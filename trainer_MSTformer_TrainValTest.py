# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import platform
import random
import numpy as np
import pytz
import tqdm
import socket
import math
import time
import yaml

from datetime import datetime

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from MSTformer_model.MSTformer_fundus_deepGF_TimeRegAtten import MSTformer
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.dist_util import get_world_size
from torch.optim.lr_scheduler import LambdaLR
from utils.fundus_dataset_fullSeq_deepGF_v3_trainValTest import fundus_DataSet, fundus_DataSet_AC
from utils.loss_utils import BalancedSoftmaxCE
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, roc_curve, auc
import matplotlib.pyplot as plt
from einops import rearrange, repeat
import ipdb


class Trainer(object):

    def __init__(self, args):
        self.args = args
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        self.model = self.setup(args)
        
        if args.loss_type == "CrossEntropyLoss":
            self.criterion_cls = torch.nn.CrossEntropyLoss()
            self.criterion_label_prob = torch.nn.CrossEntropyLoss()
        elif args.loss_type == "BCELoss":
            self.criterion_cls = torch.nn.BCELoss()
        elif args.loss_type == "BalancedSoftmaxCE":
            self.balanceSoftCE_para = {'num_classes':2, 
                                       'num_class_list':[748,2930], 
                                       'device':self.device, 
                                       'two_stage_train_flag':args.BalancedSoftmaxCE_drw, 
                                       'two_stage_train_epoch':args.BalancedSoftmaxCE_drw_epoch,
                                       'BalancedSoftmaxCE_T':args.BalancedSoftmaxCE_T}
            self.criterion_cls = BalancedSoftmaxCE(self.balanceSoftCE_para) # number of samples for every classes will be changed in each strategy
            self.criterion_label_prob = BalancedSoftmaxCE(self.balanceSoftCE_para)

        self.time_zone = 'Asia/Hong_Kong'
        self.timestamp_start = datetime.now(pytz.timezone(self.time_zone))
        if self.args.suffix is not None:
            self.log_dir = os.path.join(self.args.output_dir, 'tensorboard_MSTformer_JBHIrevision',
                                datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname() + '_' + self.args.suffix)
        else:
            self.log_dir = os.path.join(self.args.output_dir, 'tensorboard_MSTformer_JBHIrevision',
                                datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.train_record_txt = os.path.join(self.log_dir,'train_record.txt')
        self.val_record_txt = os.path.join(self.log_dir,'val_record.txt')
        self.test_record_txt = os.path.join(self.log_dir,'test_record.txt')
        self.throw_txt = os.path.join(self.log_dir,'throw_record.txt')
        self.config_txt = os.path.join(self.log_dir,'config_record.txt')
        self.test_details_txt = os.path.join(self.log_dir,'test_details.txt')

        self.throw_rate=args.throw_rate
        self.num_strategy=args.num_strategy
        self.strategy_epoch_duration=args.strategy_epoch_duration
        self.Pos_coef_list = args.Pos_coef_list
        self.train_record_diff_PosCoef_list = []
        for Pos_coef in self.Pos_coef_list:
            self.train_record_diff_PosCoef_list.append(os.path.join(self.log_dir,'train_record_Pos_coef_{}.txt'.format(Pos_coef))) 

        self._write_config_txt()
        with open(os.path.join(self.log_dir, 'config.yaml'), 'w') as f:
            yaml.safe_dump(args.__dict__, f, default_flow_style=False)
    
    
    def _write_config_txt(self):
        with open(self.config_txt, "w+") as f:
            f.write("***** Configuation *****")
            f.write('\n')
            f.write("  Sequence length = %d" % self.args.seq_len)
            f.write('\n')
            f.write("  loss type = %s" % self.args.loss_type)
            f.write('\n')
            f.write(" train batch size = %d" % self.args.train_batch_size)
            f.write(" val batch size = %d" % self.args.eval_batch_size)
            f.write('\n')
            f.write(" learning rate = %f" % self.args.learning_rate)
            f.write('\n')
            f.write(" better model criterion = {}" .format(self.args.better_model_criterion))
            f.write('\n')
            if self.args.flag_strategy_lr:
                f.write(" num strategy = %d" % self.args.num_strategy)
                f.write('\n')
                f.write(" learning rate = {}" .format(self.args.strategy_learn_rate))
                f.write('\n')
                f.write(" throw rate = {}" .format(self.args.throw_rate))
                f.write('\n')
                f.write(" strategy epoch duration = {}" .format(self.args.strategy_epoch_duration))
                f.write('\n')
                f.flush()
            else:
                f.write(" No use AC strategy")
                f.write('\n')
                f.write(" learning rate = {}" .format(self.args.learning_rate))
                f.write('\n')
                f.flush()
            
            f.write(" Pad mode = {}" .format(self.args.pad_mode))
            f.write('\n')
            f.flush()


    def setup(self,args):
        
        model = MSTformer(img_len=self.args.seq_len, 
                            win_size=self.args.merge_ratio,
                            e_layers=self.args.layer_num,
                            block_depth=self.args.block_depth,
                            useTime=self.args.useTime, 
                            useAttenMap=self.args.useAttenFlag)
        
        model.to(self.device)
        num_params = self.count_parameters(model)
        print("Sum num of params: \t%2.1fM" % num_params)
        return model
    
    def count_parameters(self, model):
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return params/1000000
    
    def save_model(self,epoch):
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_checkpoint = os.path.join(self.log_dir, "{}_model_batch{}_epoch{}.ckpt" 
                                        .format(self.args.name,self.args.train_batch_size,epoch))
        torch.save(model_to_save.state_dict(), model_checkpoint)

    def simple_accuracy(self, preds, labels):
        return (preds == labels).mean()


    def test(self, test_loader, epoch_i, global_step):
        # Validation!
        eval_losses = AverageMeter()
        if self.args.Exp_label_loss_flag == True:
            eval_future_losses = AverageMeter()
            eval_label_losses = AverageMeter()

        with open(self.test_record_txt, "a+") as f3:
            if epoch_i<8:
                f3.write("***** Running Validation *****")
                f3.write('\n')
                f3.write("  Num steps = %d" % len(test_loader))
                f3.write('\n')
                f3.write("  Batch size = %d" % self.args.eval_batch_size)
                f3.write('\n')
                f3.flush()

            self.model.eval()
            all_preds_reshape, all_label_reshape = [], []
            all_label_seq = []
            all_pred_now_label_seq = []
            all_imgname = []
            all_next_label_seq = []
            all_score_seq = []
            epoch_iterator = tqdm(test_loader,
                                desc="Validating... (loss=X.X)",
                                bar_format="{l_bar}{r_bar}",
                                dynamic_ncols=True)
            
            for step, sample_seq_batch in enumerate(epoch_iterator):
                img_batch = sample_seq_batch['image']
                atten_batch = sample_seq_batch['atten']
                polar_batch = sample_seq_batch['polar']
                next_seq_label_batch = sample_seq_batch['next_seq_label'] # shape: B x T
                year_batch = sample_seq_batch['delta_year']
                future_year_tag_batch = sample_seq_batch['future_year_tag']
                label_seq_batch = sample_seq_batch['label_seq']
                imgname_batch = sample_seq_batch['img_name']
                Tmatrix_batch = sample_seq_batch['Tmatrix']

                img_batch = img_batch.to(self.device, dtype=torch.float)
                atten_batch = atten_batch.to(self.device, dtype=torch.float)
                polar_batch = polar_batch.to(self.device, dtype=torch.float)
                next_seq_label_batch = next_seq_label_batch.to(self.device, torch.int64)
                year_batch = year_batch.to(self.device, dtype=torch.float)
                future_year_tag_batch = future_year_tag_batch.to(self.device, dtype=torch.float)
                if self.args.TEM_flag:
                    Tmatrix_batch = Tmatrix_batch.to(self.device, dtype=torch.float)
                else:
                    Tmatrix_batch = None

                with torch.no_grad():
                   if self.args.Exp_label_loss_flag == True:
                        label_seq_batch = label_seq_batch.to(self.device, torch.int64)
                        logits_cls, seq_pos_prob, pesudo_seq_label = self.model(img_batch, atten_batch, polar_batch, year_batch, future_year_tag_batch, Tmatrix=Tmatrix_batch)
                        logits_cls_reshape = rearrange(logits_cls, "b l cls -> (b l) cls")
                        next_seq_label_batch_reshape = rearrange(next_seq_label_batch, "b l -> (b l)")
                        seq_pos_prob_reshape = rearrange(seq_pos_prob, "b l cls -> (b l) cls")
                        label_seq_batch_reshape = rearrange(label_seq_batch, "b l -> (b l)")
                        label_loss = self.criterion_label_prob(seq_pos_prob_reshape, label_seq_batch_reshape)
                        future_loss = self.criterion_cls(logits_cls_reshape, next_seq_label_batch_reshape)
                        if epoch_i < self.args.warmup_epoch:
                            eval_loss = label_loss
                        else:
                            eval_loss = future_loss + self.args.Exp_label_loss_weight*label_loss
                    else:
                        label_seq_batch = label_seq_batch.to(self.device, torch.float)
                        logits_cls, pesudo_seq_label = self.model(img_batch, atten_batch, polar_batch, year_batch, future_year_tag_batch, label_seq_batch, Tmatrix=Tmatrix_batch)
                        logits_cls_reshape = rearrange(logits_cls, "b l cls -> (b l) cls")
                        next_seq_label_batch_reshape = rearrange(next_seq_label_batch, "b l -> (b l)")
                        eval_loss = self.criterion_cls(logits_cls_reshape, next_seq_label_batch_reshape)
 
                    eval_losses.update(eval_loss.item())
                    if self.args.Exp_label_loss_flag == True:
                        eval_future_losses.update(future_loss.item())
                        eval_label_losses.update(label_loss.item())

                    preds_reshape = F.softmax(logits_cls_reshape,dim=1)

                    score_reshape = preds_reshape[:, 1]
                    score = rearrange(score_reshape.squeeze(), "(b l)-> b l", l=self.args.seq_len)
                     
                    logits_cls_reshape = logits_cls_reshape.detach().cpu().numpy()

                if len(all_preds_reshape) == 0:
                    all_preds_reshape.append(preds_reshape.detach().cpu().numpy())
                    all_label_reshape.append(next_seq_label_batch_reshape.squeeze().detach().cpu().numpy())
                    all_label_seq.append(label_seq_batch.squeeze().detach().cpu().numpy())
                    all_imgname.append(np.array(imgname_batch['seq name']).transpose((1,0)))
                    all_next_label_seq.append(next_seq_label_batch.squeeze().detach().cpu().numpy())
                    all_score_seq.append(score.detach().cpu().numpy())
                    all_pred_now_label_seq.append(pesudo_seq_label.squeeze().detach().cpu().numpy())
                else:
                    all_preds_reshape[0] = np.append(all_preds_reshape[0], preds_reshape.detach().cpu().numpy(), axis=0)
                    all_label_reshape[0] = np.append(all_label_reshape[0], next_seq_label_batch_reshape.squeeze().detach().cpu().numpy(), axis=0)
                    all_label_seq[0] = np.append(all_label_seq[0], label_seq_batch.squeeze().detach().cpu().numpy(), axis=0)
                    all_imgname[0] = np.append(all_imgname[0], np.array(imgname_batch['seq name']).transpose((1,0)), axis=0)
                    all_next_label_seq[0] = np.append(all_next_label_seq[0], next_seq_label_batch.squeeze().detach().cpu().numpy(), axis=0)
                    all_score_seq[0] = np.append(all_score_seq[0], score.detach().cpu().numpy(), axis=0)
                    all_pred_now_label_seq[0] = np.append(all_pred_now_label_seq[0], pesudo_seq_label.squeeze().detach().cpu().numpy(), axis=0)

                epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

            all_preds_reshape, all_label_reshape = all_preds_reshape[0], all_label_reshape[0] # list --> numpy
            all_label_seq = all_label_seq[0]
            all_imgname = all_imgname[0]
            all_next_label_seq = all_next_label_seq[0]
            all_score_seq = all_score_seq[0]
            all_pred_now_label_seq = all_pred_now_label_seq[0]

            acc, Sen, Spe, auc, y_scores = self.evaluate_results_all_metric(all_label_reshape, all_preds_reshape)
            
            ########## loop for every pos_coef in pos_coef_list to find best Sen value: #######################
            for i in range(len(self.Pos_coef_list)):
                acc_dummy, Sen_dummy, Spe_dummy, auc_dummy, y_scores_dummy = self.evaluate_results_all_metric_improSen(all_label_reshape, all_preds_reshape, self.Pos_coef_list[i])
                # acc_dummy, Sen_dummy, Spe_dummy, auc_dummy, y_scores_dummy = self.evaluate_results_all_metric(all_label_reshape, all_preds_reshape, self.Pos_coef_list[i])
                with open(self.train_record_diff_PosCoef_list[i], "a+") as f_dummy:
                   f_dummy.write("Epoch:%d | Current Accuracy:%.3f%% | Current AUC:%.3f%% | Current sensitivity:%.3f%% | Current sepcificity:%.3f%%" % (epoch_i,acc_dummy,auc_dummy,Sen_dummy,Spe_dummy))
                   f_dummy.write('\n')
                   f_dummy.flush()
            ###################################################################################################
            
            f3.write("Global epoch: %d" % epoch_i)
            f3.write('|| ')
            f3.write("Global Steps: %d" % global_step)
            f3.write('|| ')
            f3.write("Test Loss: %2.5f" % eval_losses.avg)
            f3.write('|| ')
            f3.write("Test Accuracy: %2.5f" % acc)
            f3.write('|| ')
            f3.write("Test AUC value: %2.5f" % auc)
            f3.write('|| ')
            f3.write("Test Sensitive: %2.5f" % Sen)
            f3.write('|| ')
            f3.write("Test Specificity: %2.5f" % Spe)
            f3.write('\n')
            f3.flush()

            self.writer.add_scalar("test/loss", scalar_value=eval_losses.val, global_step=epoch_i)
            if self.args.Exp_label_loss_flag == True:
                self.writer.add_scalar("test/future_loss", scalar_value=eval_future_losses.val, global_step=epoch_i)
                self.writer.add_scalar("test/label_loss", scalar_value=eval_label_losses.val, global_step=epoch_i)
            self.writer.add_scalar("test/accuracy", scalar_value=acc, global_step=epoch_i)
            self.writer.add_scalar("test/auc", scalar_value=auc, global_step=epoch_i)
            self.writer.add_scalar("test/sensitive", scalar_value=Sen, global_step=epoch_i)
            self.writer.add_scalar("test/specificity", scalar_value=Spe, global_step=epoch_i)
            
        return acc, auc, Sen, Spe, all_label_reshape, y_scores, all_label_seq, all_next_label_seq, all_score_seq, all_imgname, all_pred_now_label_seq
        
        
    def valid(self, val_loader, epoch_i, global_step):
        # Validation!
        eval_losses = AverageMeter()
        if self.args.Exp_label_loss_flag == True:
            eval_future_losses = AverageMeter()
            eval_label_losses = AverageMeter()

        with open(self.val_record_txt, "a+") as f2:
            if epoch_i<8:
                f2.write("***** Running Validation *****")
                f2.write('\n')
                f2.write("  Num steps = %d" % len(val_loader))
                f2.write('\n')
                f2.write("  Batch size = %d" % self.args.eval_batch_size)
                f2.write('\n')
                f2.flush()

            self.model.eval()
            all_preds_reshape, all_label_reshape = [], []
            all_label_seq = []
            all_pred_now_label_seq = []
            all_imgname = []
            all_next_label_seq = []
            all_score_seq = []
            epoch_iterator = tqdm(val_loader,
                                desc="Validating... (loss=X.X)",
                                bar_format="{l_bar}{r_bar}",
                                dynamic_ncols=True)
            
            for step, sample_seq_batch in enumerate(epoch_iterator):
                img_batch = sample_seq_batch['image']
                atten_batch = sample_seq_batch['atten']
                polar_batch = sample_seq_batch['polar']
                next_seq_label_batch = sample_seq_batch['next_seq_label'] # shape: B x T
                year_batch = sample_seq_batch['delta_year']
                future_year_tag_batch = sample_seq_batch['future_year_tag']
                label_seq_batch = sample_seq_batch['label_seq']
                imgname_batch = sample_seq_batch['img_name']
                Tmatrix_batch = sample_seq_batch['Tmatrix']

                img_batch = img_batch.to(self.device, dtype=torch.float)
                atten_batch = atten_batch.to(self.device, dtype=torch.float)
                polar_batch = polar_batch.to(self.device, dtype=torch.float)
                next_seq_label_batch = next_seq_label_batch.to(self.device, torch.int64)
                year_batch = year_batch.to(self.device, dtype=torch.float)
                future_year_tag_batch = future_year_tag_batch.to(self.device, dtype=torch.float)
                if self.args.TEM_flag:
                    Tmatrix_batch = Tmatrix_batch.to(self.device, dtype=torch.float)
                else:
                    Tmatrix_batch = None

                with torch.no_grad():
                    if self.args.Exp_label_loss_flag == True:
                        label_seq_batch = label_seq_batch.to(self.device, torch.int64)
                        logits_cls, seq_pos_prob, pesudo_seq_label = self.model(img_batch, atten_batch, polar_batch, year_batch, future_year_tag_batch, Tmatrix=Tmatrix_batch)
                        logits_cls_reshape = rearrange(logits_cls, "b l cls -> (b l) cls")
                        next_seq_label_batch_reshape = rearrange(next_seq_label_batch, "b l -> (b l)")
                        seq_pos_prob_reshape = rearrange(seq_pos_prob, "b l cls -> (b l) cls")
                        label_seq_batch_reshape = rearrange(label_seq_batch, "b l -> (b l)")
                        label_loss = self.criterion_label_prob(seq_pos_prob_reshape, label_seq_batch_reshape)
                        future_loss = self.criterion_cls(logits_cls_reshape, next_seq_label_batch_reshape)
                        if epoch_i < self.args.warmup_epoch:
                            eval_loss = label_loss
                        else:
                            eval_loss = future_loss + self.args.Exp_label_loss_weight*label_loss
                    else:
                        label_seq_batch = label_seq_batch.to(self.device, torch.float)
                        logits_cls, pesudo_seq_label = self.model(img_batch, atten_batch, polar_batch, year_batch, future_year_tag_batch, label_seq_batch, Tmatrix=Tmatrix_batch)
                        logits_cls_reshape = rearrange(logits_cls, "b l cls -> (b l) cls")
                        next_seq_label_batch_reshape = rearrange(next_seq_label_batch, "b l -> (b l)")
                        eval_loss = self.criterion_cls(logits_cls_reshape, next_seq_label_batch_reshape)
 
                    eval_losses.update(eval_loss.item())
                    if self.args.Exp_label_loss_flag == True:
                        eval_future_losses.update(future_loss.item())
                        eval_label_losses.update(label_loss.item())

                    preds_reshape = F.softmax(logits_cls_reshape,dim=1)

                    score_reshape = preds_reshape[:, 1]
                    score = rearrange(score_reshape.squeeze(), "(b l)-> b l", l=self.args.seq_len)
                     
                    logits_cls_reshape = logits_cls_reshape.detach().cpu().numpy()

                if len(all_preds_reshape) == 0:
                    all_preds_reshape.append(preds_reshape.detach().cpu().numpy())
                    all_label_reshape.append(next_seq_label_batch_reshape.squeeze().detach().cpu().numpy())
                    all_label_seq.append(label_seq_batch.squeeze().detach().cpu().numpy())
                    all_imgname.append(np.array(imgname_batch['seq name']).transpose((1,0)))
                    all_next_label_seq.append(next_seq_label_batch.squeeze().detach().cpu().numpy())
                    all_score_seq.append(score.detach().cpu().numpy())
                    all_pred_now_label_seq.append(pesudo_seq_label.squeeze().detach().cpu().numpy())
                else:
                    all_preds_reshape[0] = np.append(all_preds_reshape[0], preds_reshape.detach().cpu().numpy(), axis=0)
                    all_label_reshape[0] = np.append(all_label_reshape[0], next_seq_label_batch_reshape.squeeze().detach().cpu().numpy(), axis=0)
                    all_label_seq[0] = np.append(all_label_seq[0], label_seq_batch.squeeze().detach().cpu().numpy(), axis=0)
                    all_imgname[0] = np.append(all_imgname[0], np.array(imgname_batch['seq name']).transpose((1,0)), axis=0)
                    all_next_label_seq[0] = np.append(all_next_label_seq[0], next_seq_label_batch.squeeze().detach().cpu().numpy(), axis=0)
                    all_score_seq[0] = np.append(all_score_seq[0], score.detach().cpu().numpy(), axis=0)
                    all_pred_now_label_seq[0] = np.append(all_pred_now_label_seq[0], pesudo_seq_label.squeeze().detach().cpu().numpy(), axis=0)

                epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

            all_preds_reshape, all_label_reshape = all_preds_reshape[0], all_label_reshape[0] # list --> numpy
            all_label_seq = all_label_seq[0]
            all_imgname = all_imgname[0]
            all_next_label_seq = all_next_label_seq[0]
            all_score_seq = all_score_seq[0]
            all_pred_now_label_seq = all_pred_now_label_seq[0]

            acc, Sen, Spe, auc, y_scores = self.evaluate_results_all_metric(all_label_reshape, all_preds_reshape, self.args.Pos_coef_selected)
            
            f2.write("Global epoch: %d" % epoch_i)
            f2.write('|| ')
            f2.write("Global Steps: %d" % global_step)
            f2.write('|| ')
            f2.write("Valid Loss: %2.5f" % eval_losses.avg)
            f2.write('|| ')
            f2.write("Valid Accuracy: %2.5f" % acc)
            f2.write('|| ')
            f2.write("Valid AUC value: %2.5f" % auc)
            f2.write('|| ')
            f2.write("Valid Sensitive: %2.5f" % Sen)
            f2.write('|| ')
            f2.write("Valid Specificity: %2.5f" % Spe)
            f2.write('\n')
            f2.flush()
            

            self.writer.add_scalar("validation/loss", scalar_value=eval_losses.val, global_step=epoch_i)
            if self.args.Exp_label_loss_flag == True:
                self.writer.add_scalar("validation/future_loss", scalar_value=eval_future_losses.val, global_step=epoch_i)
                self.writer.add_scalar("validation/label_loss", scalar_value=eval_label_losses.val, global_step=epoch_i)
            self.writer.add_scalar("validation/accuracy", scalar_value=acc, global_step=epoch_i)
            self.writer.add_scalar("validation/auc", scalar_value=auc, global_step=epoch_i)
            self.writer.add_scalar("validation/sensitive", scalar_value=Sen, global_step=epoch_i)
            self.writer.add_scalar("validation/specificity", scalar_value=Spe, global_step=epoch_i)
            
        return acc, auc, Sen, Spe, all_label_reshape, y_scores, all_label_seq, all_next_label_seq, all_score_seq, all_imgname, all_pred_now_label_seq
    


    def train(self):
        """ Train the model """
        # Prepare dataset
        train_loader, val_loader, test_loader, self.train_fundus_DataSet = self.get_loader()
        Next_label_pool = self.train_fundus_DataSet.get_label_pool()
        sum_num = len(Next_label_pool)
        pos_num = sum(Next_label_pool)
        neg_num = sum_num - pos_num
        print("Sum number of samples:{};\t Postive samples:{};\t Negative samples:{}"
            .format(sum_num,pos_num,neg_num))
        
        if self.args.loss_type == "BalancedSoftmaxCE":
            self.update_BalancedSoftmaxCE(self.train_fundus_DataSet)
        # Prepare optimizer and scheduler
        if not self.args.flag_strategy_lr:
            optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=self.args.learning_rate,
                                        momentum=0.9,
                                        weight_decay=self.args.weight_decay)
        else:
            optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=self.args.strategy_learn_rate[0],
                                        momentum=0.9,
                                        weight_decay=self.args.weight_decay)
        
        if not self.args.flag_strategy_lr: 
            t_total = self.args.num_steps
            if self.args.decay_type == "cosine":
                scheduler = WarmupCosineSchedule(optimizer, warmup_steps=self.args.warmup_steps, t_total=t_total)
            else:
                scheduler = WarmupLinearSchedule(optimizer, warmup_steps=self.args.warmup_steps, t_total=t_total)
        # Train!
        with open(self.train_record_txt, "w+") as f:
            f.write("***** Running training *****")
            f.write('\n')
            f.write("  Total optimization steps = %d" % self.args.num_steps)
            f.write('\n')
            f.write("  Sum number of samples:{};\t Postive samples:{};\t Negative samples:{}" .format(sum_num,pos_num,neg_num))
            f.write('\n')
            f.write("  Instantaneous batch size per GPU = %d" % self.args.train_batch_size)
            f.write('\n')
            f.write("  Gradient Accumulation steps = %d" % self.args.gradient_accumulation_steps)
            f.write('\n')
            f.flush()

            self.model.zero_grad()
            self.set_seed(self.args)  # Added here for reproducibility (even between python 2 and 3)
            losses = AverageMeter()
            if self.args.Exp_label_loss_flag == True:
                future_losses = AverageMeter()
                label_losses = AverageMeter()
            global_step, best_acc, best_AUC, best_sen, best_spe = 0, 0, 0, 0, 0
            count_strategy=0
            flag_strategy = self.args.flag_strategy
            last_strategy_epoch = 0

            for epoch_i in range(self.args.num_epoch):
                print("Starting train epoch {}...".format(epoch_i))

                """-----------------------------------------throw------------------------------------------"""
                if epoch_i>0 and flag_strategy and ((epoch_i-last_strategy_epoch) % self.strategy_epoch_duration[count_strategy]) == 0:
                
                    self.train_fundus_DataSet = self.throw(self.train_fundus_DataSet, count_strategy)
                    if self.args.loss_type == "BalancedSoftmaxCE":
                        self.update_BalancedSoftmaxCE(self.train_fundus_DataSet)
                    train_loader = DataLoader(dataset=self.train_fundus_DataSet,batch_size=self.train_batch_size,shuffle=True,drop_last=True)
                    
                    if self.args.flag_strategy_lr:
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = self.args.strategy_learn_rate[count_strategy]

                    count_strategy += 1  # 1~5
                    last_strategy_epoch = epoch_i
                    
                    if count_strategy == self.num_strategy:
                        flag_strategy = False
                """-----------------------------------------train------------------------------------------"""    
                if self.args.loss_type == "BalancedSoftmaxCE":
                    if self.args.Exp_label_loss_flag == True:
                        self.criterion_cls.update(epoch_i-last_strategy_epoch-self.args.warmup_epoch) # initialize loss_weight and determine whether to start two-stage training
                        self.criterion_label_prob.update(epoch_i)
                    else:
                        self.criterion_cls.update(epoch_i-last_strategy_epoch)

                self.model.train()
                epoch_iterator = tqdm(train_loader,
                                    desc="Training (X / X Steps) (loss=X.X) (global step=X) (avg_loss_epoch=X.X)",
                                    bar_format="{l_bar}{r_bar}",
                                    dynamic_ncols=True)
                for step, sample_seq_batch in enumerate(epoch_iterator):
                    img_batch = sample_seq_batch['image']
                    atten_batch = sample_seq_batch['atten']
                    polar_batch = sample_seq_batch['polar']
                    next_seq_label_batch = sample_seq_batch['next_seq_label'] # shape: B x T
                    year_batch = sample_seq_batch['delta_year']
                    future_year_tag_batch = sample_seq_batch['future_year_tag']
                    label_seq_batch = sample_seq_batch['label_seq']
                    Tmatrix_batch = sample_seq_batch['Tmatrix']

                    img_batch = img_batch.to(self.device, dtype=torch.float)
                    atten_batch = atten_batch.to(self.device, dtype=torch.float)
                    polar_batch = polar_batch.to(self.device, dtype=torch.float)
                    next_seq_label_batch = next_seq_label_batch.to(self.device, torch.int64)
                    year_batch = year_batch.to(self.device, dtype=torch.float)
                    future_year_tag_batch = future_year_tag_batch.to(self.device, dtype=torch.float)
                    if self.args.TEM_flag:
                        Tmatrix_batch = Tmatrix_batch.to(self.device, dtype=torch.float)
                    else:
                        Tmatrix_batch = None

                    if self.args.Exp_label_loss_flag == True:
                        label_seq_batch = label_seq_batch.to(self.device, torch.int64)
                        logits_cls, seq_pos_prob, pesudo_seq_label = self.model(img_batch, atten_batch, polar_batch, year_batch, future_year_tag_batch, Tmatrix=Tmatrix_batch)
                        logits_cls_reshape = rearrange(logits_cls, "b l cls -> (b l) cls")
                        next_seq_label_batch_reshape = rearrange(next_seq_label_batch, "b l -> (b l)")
                        seq_pos_prob_reshape = rearrange(seq_pos_prob, "b l cls -> (b l) cls")
                        label_seq_batch_reshape = rearrange(label_seq_batch, "b l -> (b l)")
                        label_loss = self.criterion_label_prob(seq_pos_prob_reshape, label_seq_batch_reshape)
                        
                        future_loss = self.criterion_cls(logits_cls_reshape, next_seq_label_batch_reshape)
                        if epoch_i < self.args.warmup_epoch:
                            loss = label_loss
                        else:
                            loss = future_loss + self.args.Exp_label_loss_weight*label_loss
                    else:
                        label_seq_batch = label_seq_batch.to(self.device, torch.float)
                        logits_cls, pesudo_seq_label = self.model(img_batch, atten_batch, polar_batch, year_batch, future_year_tag_batch, label_seq_batch, Tmatrix=Tmatrix_batch)
                        logits_cls_reshape = rearrange(logits_cls, "b l cls -> (b l) cls")
                        next_seq_label_batch_reshape = rearrange(next_seq_label_batch, "b l -> (b l)")
                        loss = self.criterion_cls(logits_cls_reshape, next_seq_label_batch_reshape)
                       
                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps
                        
                    loss.backward()

                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        losses.update(loss.item()*self.args.gradient_accumulation_steps)
                        if self.args.Exp_label_loss_flag == True:
                            future_losses.update(future_loss.item()*self.args.gradient_accumulation_steps)
                            label_losses.update(label_loss.item()*self.args.gradient_accumulation_steps)
                        if self.args.grad_clip_flag:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                        
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1
                        
                        if not self.args.flag_strategy_lr:
                            scheduler.step()

                        epoch_iterator.set_description(
                            "Training (%d / %d Steps) (loss=%2.5f) (global step=%d) (avg_loss_epoch=%2.5f)"
                                            % (step, len(train_loader), losses.val, global_step, losses.avg)
                        )

                        self.writer.add_scalar("train/loss_batch", scalar_value=losses.val, global_step=global_step)
                        if self.args.Exp_label_loss_flag == True:
                            self.writer.add_scalar("train/future_loss_batch", scalar_value=future_losses.val, global_step=global_step)
                            self.writer.add_scalar("train/label_loss_batch", scalar_value=label_losses.val, global_step=global_step)
                        self.writer.add_scalar("train/lr", scalar_value=optimizer.param_groups[0]['lr'], global_step=global_step)

                self.writer.add_scalar("train/loss_epoch", scalar_value=losses.avg, global_step=epoch_i)
                if self.args.Exp_label_loss_flag == True:
                    self.writer.add_scalar("train/future_loss_epoch", scalar_value=future_losses.avg, global_step=epoch_i)
                    self.writer.add_scalar("train/label_loss_epoch", scalar_value=label_losses.avg, global_step=epoch_i)

                if (epoch_i+1) % self.args.eval_every == 0:
                    accuracy_val,AUC_value_val,sensitivity_val, sepcificity_val,all_label_val,y_scores_val, all_label_seq_val, all_next_label_seq_val, all_score_seq_val, all_imgname_val, all_pred_now_label_seq_val = self.valid(val_loader, epoch_i, global_step)
                    
                    accuracy,AUC_value,sensitivity, sepcificity,all_label,y_scores, all_label_seq, all_next_label_seq, all_score_seq, all_imgname, all_pred_now_label_seq = self.test(test_loader, epoch_i, global_step)
                    # if best_AUC < AUC_value: # better model criterion
                    if self.args.better_model_criterion[0]*best_AUC+self.args.better_model_criterion[1]*best_acc+self.args.better_model_criterion[2]*best_sen+self.args.better_model_criterion[3]*best_spe < self.args.better_model_criterion[0]*AUC_value_val+self.args.better_model_criterion[1]*accuracy_val+self.args.better_model_criterion[2]*sensitivity_val+self.args.better_model_criterion[3]*sepcificity_val:
                        best_acc = accuracy_val
                        best_AUC = AUC_value_val
                        best_sen = sensitivity_val
                        best_spe = sepcificity_val
                        self.save_model(epoch_i)
                    
                        savepath = os.path.join(self.log_dir, "{}_ROC_fig_epoch{}.png" 
                                        .format(self.args.name,epoch_i))
                        self.plot_ROC(y_label=all_label,y_pre=y_scores,savepath=savepath)
                        save_dict = {'y_label':all_label,'y_pre':y_scores}
                        savedictpath = os.path.join(self.log_dir, "{}_Pred_GT_label_epoch{}.npy" .format(self.args.name,epoch_i)) # save pred data for recover ROC curve
                        np.save(savedictpath, save_dict)
                        
                        all_imgname = all_imgname.tolist()
                        all_label_seq = all_label_seq.tolist()
                        all_next_label_seq = all_next_label_seq.tolist()
                        all_score_seq = all_score_seq.tolist()
                        all_pred_now_label_seq = all_pred_now_label_seq.tolist()
                        with torch.no_grad():
                            with open(self.test_details_txt, "a+") as f_test:
                                f_test.write("***** Test epoch %d *****" % epoch_i)
                                f_test.write('\n')
                                for i in range(len(all_imgname)):
                                    y_scores_seq = all_score_seq[i]
                                    pred_label_seq = []
                                    for j in range(len(y_scores_seq)):
                                        if y_scores_seq[j] > self.args.Pos_coef_selected:
                                            pred_label_seq.append(1)
                                        else:
                                            pred_label_seq.append(0)
                                    f_test.write('Image Seq name:' + str(all_imgname[i]) + ' || current seq_label:'+str(all_label_seq[i]) + ' || Pred current seq_label:' + str(all_pred_now_label_seq[i]) + ' || GT next seq label:'+ str(all_next_label_seq[i]) + ' || Pred next seq label:'+ str(pred_label_seq))
                                    f_test.write('\n')
                                    f_test.flush()
                                f_test.write("End Testing!")
                                f_test.write('\n')
                                f_test.write('\n')
                                f_test.flush()  

                        
                    print('Validation -- Best Accuracy:{}, best AUC:{}, best sensitivity:{}, best sepcificity:{}\n'.format(best_acc,best_AUC,best_sen,best_spe))   
                    print('Test -- Accuracy:{}, AUC:{}, sensitivity:{}, sepcificity:{}\n'.format(accuracy,AUC_value,sensitivity,sepcificity))   
                    f.write("(Epoch %d) Current Test Accuracy:%.3f%% | Current Test AUC:%.3f%% | Current Test sensitivity:%.3f%% | Current Test sepcificity:%.3f%%" % (epoch_i+1,accuracy,AUC_value,sensitivity,sepcificity))
                    f.write('\n')
                    f.flush()
                    self.model.train()

                losses.reset()
                if self.args.Exp_label_loss_flag == True:
                    future_losses.reset()
                    label_losses.reset()
               
            f.write("Overall Test Accuracy: \t%.03f%%" % accuracy)
            f.write('\n')
            f.write("Overall Test AUC: \t%.03f%%" % AUC_value)
            f.write('\n')
            f.write("Overall Test sensitivity: \t%.03f%%" % sensitivity)
            f.write('\n')
            f.write("Overall Test sepcificity: \t%.03f%%" % sepcificity)
            f.write('\n')
            f.write("End Training!")
            f.write('\n')
            f.flush()

    
    def throw(self, old_train_dataset, count_strategy):
        # go though all training samples
        loss_pool = []

        image_pool = []
        atten_pool = []
        polar_pool = []
        next_label_pool = []
        label_seq_pool = []
        year_pool = []
        imgname_pool = []
        gla_year_pool = []
        
        new_image_pool = []
        new_atten_pool = []
        new_polar_pool = []
        new_next_label_pool = []
        new_label_seq_pool = []
        new_year_pool = []
        new_imgname_pool = []
        new_gla_year_pool = []
        old_train_loader = DataLoader(dataset=old_train_dataset,batch_size=1,shuffle=False,drop_last=True) # read sample one by one
        pos_num = 0
        neg_num = 0

        print("Start sample throwing process...\n")
        for step, sample_seq_batch in enumerate(tqdm(old_train_loader)):
            img_batch = sample_seq_batch['image']
            atten_batch = sample_seq_batch['atten']
            polar_batch = sample_seq_batch['polar']
            imgname_batch = sample_seq_batch['img_name']
            label_seq_batch = sample_seq_batch['label_seq']
            next_label_batch = sample_seq_batch['next_label']
            next_seq_label_batch = sample_seq_batch['next_seq_label'] # shape: B x T
            Gla_year_batch = sample_seq_batch['gla_eye_year']
            year_batch = sample_seq_batch['delta_year']
            future_year_tag_batch = sample_seq_batch['future_year_tag']
            Tmatrix_batch = sample_seq_batch['Tmatrix']

            img_batch = img_batch.to(self.device, dtype=torch.float)
            atten_batch = atten_batch.to(self.device, dtype=torch.float)
            polar_batch = polar_batch.to(self.device, dtype=torch.float)
            year_batch = year_batch.to(self.device, dtype=torch.float)
            next_seq_label_batch = next_seq_label_batch.to(self.device, torch.int64)
            future_year_tag_batch = future_year_tag_batch.to(self.device, dtype=torch.float)
            next_label_batch = next_label_batch.to(self.device, torch.int64)
            Gla_year_batch = Gla_year_batch.to(self.device, dtype=torch.float)
            if self.args.TEM_flag:
                Tmatrix_batch = Tmatrix_batch.to(self.device, dtype=torch.float)
            else:
                Tmatrix_batch = None

            with torch.no_grad():
                if self.args.Exp_label_loss_flag == True:
                    label_seq_batch = label_seq_batch.to(self.device, torch.int64)
                    logits_cls, seq_pos_prob, pesudo_seq_label = self.model(img_batch, atten_batch, polar_batch, year_batch, future_year_tag_batch, Tmatrix=Tmatrix_batch)
                    logits_cls_reshape = rearrange(logits_cls, "b l cls -> (b l) cls")
                    next_seq_label_batch_reshape = rearrange(next_seq_label_batch, "b l -> (b l)")
                    seq_pos_prob_reshape = rearrange(seq_pos_prob, "b l cls -> (b l) cls")
                    label_seq_batch_reshape = rearrange(label_seq_batch, "b l -> (b l)")
                    loss = self.criterion_cls(logits_cls_reshape, next_seq_label_batch_reshape) + \
                                    self.args.Exp_label_loss_weight*self.criterion_label_prob(seq_pos_prob_reshape, label_seq_batch_reshape)            
                else:
                    label_seq_batch = label_seq_batch.to(self.device, torch.float)
                    logits_cls, pesudo_seq_label = self.model(img_batch, atten_batch, polar_batch, year_batch, future_year_tag_batch, label_seq_batch, Tmatrix=Tmatrix_batch)
                    logits_cls_reshape = rearrange(logits_cls, "b l cls -> (b l) cls")
                    next_seq_label_batch_reshape = rearrange(next_seq_label_batch, "b l -> (b l)")
                    loss = self.criterion_cls(logits_cls_reshape, next_seq_label_batch_reshape)

            loss_pool.append(loss.detach().cpu().numpy().squeeze())
            image_pool.append(img_batch.detach().cpu().numpy().squeeze())
            atten_pool.append(atten_batch.detach().cpu().numpy().squeeze())
            polar_pool.append(polar_batch.detach().cpu().numpy().squeeze())
            next_label_pool.append(next_label_batch.detach().cpu().numpy().squeeze())
            year_pool.append(year_batch.detach().cpu().numpy().squeeze())
            label_seq_pool.append(label_seq_batch.detach().cpu().numpy().squeeze())
            imgname_pool.append(imgname_batch)
            gla_year_pool.append(Gla_year_batch.detach().cpu().numpy().squeeze())

        if count_strategy == 0:
            with open(self.throw_txt, "a")as f3:
                f3.write('Initial dataset:')
                f3.write('\n')
                f3.flush()
                
                for i in range(len(image_pool)): # ranked id less than new_sample_len
                    imgname = imgname_pool[i]['seq name']
                    imgname_list = [i[0] for i in imgname]

                    f3.write('Seq name:' + str(imgname_list) + '  Sep next label:'+ str(next_label_pool[i]) + 'seq_label:'+str(label_seq_pool[i]) + '  Glaucoma year interval: %.3f%%' %(gla_year_pool[i]*30))
                    f3.flush()
                    f3.write('\n')
                    f3.flush()
                
                f3.write('Positive Number:{}, Negative Number:{}'.format(sum(next_label_pool), len(next_label_pool)-sum(next_label_pool)))
                f3.write('\n')
                f3.flush()
                f3.write('\n')
                f3.flush()
                f3.write('\n')
                f3.flush()


        ori_sample_num = len(image_pool)
        new_sample_len = int(ori_sample_num*self.throw_rate[count_strategy])

        # sort sample based loss list (reverse order big-->small)
        sorted_id = sorted(range(len(loss_pool)), key=lambda k: loss_pool[k], reverse=True)

        assert len(sorted_id) == len(image_pool)
        with open(self.throw_txt, "a")as f4:
            f4.write('strategy: %d ' % count_strategy)
            f4.write('\n')
            f4.flush()
            for i in range(new_sample_len): # ranked id less than new_sample_len
                new_image_pool.append(image_pool[sorted_id[i]])
                new_atten_pool.append(atten_pool[sorted_id[i]])
                new_polar_pool.append(polar_pool[sorted_id[i]])
                # new_next_label_pool.append(next_label_pool[sorted_id[i]])
                new_year_pool.append(year_pool[sorted_id[i]])
                new_next_label_pool.append(next_label_pool[sorted_id[i]])
                new_gla_year_pool.append(gla_year_pool[sorted_id[i]])
                new_label_seq_pool.append(label_seq_pool[sorted_id[i]])

                pos_num += next_label_pool[sorted_id[i]]

                imgname_new = imgname_pool[sorted_id[i]]['seq name']
                imgname_new_list = [i[0] for i in imgname_new]
                imgname_new_dict = {'seq name':imgname_new_list}
                new_imgname_pool.append(imgname_new_dict)
                f4.write('Seq name:' + str(imgname_new_list) + '  Sep next label:'+ str(next_label_pool[sorted_id[i]]) + 'seq_label:'+str(label_seq_pool[sorted_id[i]]) + '  Glaucoma year interval: %d' %(gla_year_pool[sorted_id[i]]*30))
                f4.flush()
                f4.write('\n')
                f4.flush()
            
            f4.write('Positive Number:{}, Negative Number:{}'.format(pos_num, new_sample_len-pos_num))
            f4.flush()

            f4.write('\n')
            f4.flush()
            f4.write('\n')
            f4.flush()
            f4.write('\n')
            f4.flush()

        new_train_fundus_DataSet = fundus_DataSet_AC(image_pool=new_image_pool,
                                                     atten_pool=new_atten_pool,
                                                     polar_pool=new_polar_pool,
                                                     next_label_pool=new_next_label_pool,
                                                     label_seq_pool=new_label_seq_pool,
                                                    #  isGla_Xyear_pool=new_isGla_Xyear_pool,
                                                     year_pool=new_year_pool,
                                                     imgname_pool=new_imgname_pool,
                                                     gla_year_pool=new_gla_year_pool)

        return new_train_fundus_DataSet


    def get_loader(self):
        train_path_to_image = f'../Dataset/{self.args.foldname}/'+'train'+'/image/all/' 
        train_path_to_label = f'../Dataset/{self.args.foldname}/'+ 'train'+'/label/all/' 
        val_path_to_image = f'../Dataset/{self.args.foldname}/'+'validation'+'/image/all/' 
        val_path_to_label = f'../Dataset/{self.args.foldname}/'+ 'validation'+'/label/all/' 
        test_path_to_image = f'../Dataset/{self.args.foldname}/'+'test'+'/image/all/'
        test_path_to_label = f'../Dataset/{self.args.foldname}/'+ 'test'+'/label/all/'
        path_to_atten = f'../Dataset/final_atten/'
        path_to_polar = f'../Dataset/final_polar/'
        
        train_fundus_DataSet = fundus_DataSet(train_path_to_image,
                                            train_path_to_label, 
                                            pred_xYear=self.args.pred_xYear, 
                                            seq_len=self.args.seq_len, 
                                            path_to_atten=path_to_atten, 
                                            path_to_polar=path_to_polar, 
                                            transform=None, 
                                            seq_aug_flag=self.args.SeqAug_flag, 
                                            pad_mode=self.args.pad_mode,
                                            time_emd_type=self.args.time_emd_type,
                                            TEM_a=self.args.TEM_a,
                                            TEM_b=self.args.TEM_b)
        
        val_fundus_DataSet = fundus_DataSet(val_path_to_image,
                                            val_path_to_label, 
                                            pred_xYear=self.args.pred_xYear, 
                                            seq_len=self.args.seq_len, 
                                            path_to_atten=path_to_atten, 
                                            path_to_polar=path_to_polar, 
                                            transform=None, 
                                            seq_aug_flag=self.args.SeqAug_flag, 
                                            pad_mode=self.args.pad_mode,
                                            time_emd_type=self.args.time_emd_type,
                                            TEM_a=self.args.TEM_a,
                                            TEM_b=self.args.TEM_b)

        test_fundus_DataSet = fundus_DataSet(test_path_to_image, 
                                            test_path_to_label, 
                                            pred_xYear=self.args.pred_xYear, 
                                            seq_len=self.args.seq_len, 
                                            path_to_atten=path_to_atten, 
                                            path_to_polar=path_to_polar, 
                                            transform=None, 
                                            seq_aug_flag=False, 
                                            pad_mode="NO_PAD", # self.args.pad_mode
                                            time_emd_type=self.args.time_emd_type,
                                            TEM_a=self.args.TEM_a,
                                            TEM_b=self.args.TEM_b)
                                            
        train_loader=DataLoader(dataset=train_fundus_DataSet,batch_size=self.train_batch_size,shuffle=True,drop_last=True)
        val_loader=DataLoader(dataset=val_fundus_DataSet,batch_size=self.train_batch_size,shuffle=False,drop_last=True)
        test_loader=DataLoader(dataset=test_fundus_DataSet,batch_size=self.eval_batch_size,shuffle=False,drop_last=True)

        train_next_label_pool = train_fundus_DataSet.get_label_pool()
        train_sum_num = len(train_next_label_pool)
        train_pos_num = sum(train_next_label_pool)
        train_neg_num = train_sum_num - train_pos_num
        print("Training -- Sum number of samples:{};\t Postive samples:{};\t Negative samples:{}".format(train_sum_num,train_pos_num,train_neg_num))
        val_next_label_pool = val_fundus_DataSet.get_label_pool()
        val_sum_num = len(val_next_label_pool)
        val_pos_num = sum(val_next_label_pool)
        val_neg_num = val_sum_num - val_pos_num
        print("Validation -- Sum number of samples:{};\t Postive samples:{};\t Negative samples:{}".format(val_sum_num,val_pos_num,val_neg_num))
        test_next_label_pool = test_fundus_DataSet.get_label_pool()
        test_sum_num = len(test_next_label_pool)
        test_pos_num = sum(test_next_label_pool)
        test_neg_num = test_sum_num -test_pos_num
        print("Testing -- Sum number of samples:{};\t Postive samples:{};\t Negative samples:{}".format(test_sum_num,test_pos_num,test_neg_num))
        
        return train_loader, val_loader, test_loader, train_fundus_DataSet
    
    
    def update_BalancedSoftmaxCE(self, load_dataset):
        label_pool = load_dataset.get_label_pool()
        label_0_num = len(label_pool) - sum(label_pool)
        label_1_num = sum(label_pool)
        self.balanceSoftCE_para['num_class_list'] = [label_0_num,label_1_num]
        print("Creating BalancedSoftmaxCE class with config: \n{}".format(self.balanceSoftCE_para))
        self.criterion_cls = BalancedSoftmaxCE(self.balanceSoftCE_para) 
    
    
    def plot_ROC(self,y_label,y_pre,savepath):
        fpr, tpr, thersholds = roc_curve(y_label, y_pre)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
        
        plt.xlim([-0.05, 1.05]) 
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate') 
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig(savepath)
    

    def evaluate_results_all_metric(self, GT_label, pred_label, pos_coef=None):
        tp = 0.0
        fn = 0.0
        tn = 0.0
        fp = 0.0
        if len(GT_label.shape) == 1:
            num_sample = GT_label.shape[0]
        elif len(GT_label.shape) == 2:
            num_sample = max(GT_label.shape)
        else:
            raise ValueError
        
        label_predict_0 = pred_label[:, 0]  
        label_predict_1 = pred_label[:, 1] 
        # y_scores = np.zeros_like(GT_label)
        y_scores = pred_label[:, 1] 
        
        for nb in range(num_sample):
            if GT_label[nb] == 1 and (label_predict_1[nb] > label_predict_0[nb]):
                tp = tp + 1
            if GT_label[nb] == 0 and (label_predict_1[nb] < label_predict_0[nb]):
                tn = tn + 1
            if GT_label[nb] == 1 and (label_predict_1[nb] < label_predict_0[nb]):
                fn = fn + 1
            if GT_label[nb] == 0 and (label_predict_1[nb] > label_predict_0[nb]):
                fp = fp + 1
            
            # y_scores[nb] = (math.exp(label_predict_1[nb])) / (math.exp(label_predict_1[nb]) + math.exp(label_predict_0[nb]))

        acc = (tp + tn) / (tp + tn + fp + fn)
        Sen = tp / (tp + fn)
        Spe = tn / (tn + fp)
        auc = roc_auc_score(GT_label, y_scores)
        
        return acc, Sen, Spe, auc, y_scores
    
    
    def evaluate_results_all_metric_improSen(self, GT_label, pred_label, pos_coef):
        tp = 0.0
        fn = 0.0
        tn = 0.0
        fp = 0.0
        if len(GT_label.shape) == 1:
            num_sample = GT_label.shape[0]
        elif len(GT_label.shape) == 2:
            num_sample = max(GT_label.shape)
        else:
            raise ValueError
        
        label_predict_0 = pred_label[:, 0]  
        label_predict_1 = pred_label[:, 1] 
        y_scores = pred_label[:, 1] 
        
        for nb in range(num_sample):
            
            if GT_label[nb] == 1 and (label_predict_1[nb] > pos_coef):
                tp = tp + 1
            if GT_label[nb] == 0 and (label_predict_1[nb] < pos_coef):
                tn = tn + 1
            if GT_label[nb] == 1 and (label_predict_1[nb] < pos_coef):
                fn = fn + 1
            if GT_label[nb] == 0 and (label_predict_1[nb] > pos_coef):
                fp = fp + 1

        acc = (tp + tn) / (tp + tn + fp + fn)
        Sen = tp / (tp + fn)
        Spe = tn / (tn + fp)
        auc = roc_auc_score(GT_label, y_scores)
        
        return acc, Sen, Spe, auc, y_scores
        
    
    def get_file_list(self, dir_path, suffix):
        file_list = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.startswith(suffix):
                    file_list.append(file[:-4])
        return file_list
    

    def set_seed(self, args):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count