import argparse
import os
import platform
import random
import numpy as np

from trainer_MSTformer_TrainValTest import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # saving params
    parser.add_argument("--name", default="Pred_glaucoma_MSTformer",
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--suffix", default="MSTformer",
                        help="add suffix onto subdir.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    # dataset params
    parser.add_argument("--pred_xYear", default=10, type=int,
                        help="choose to pred X year after the observed visit.")
    parser.add_argument("--seq_len", default=5, type=int,
                        help="sequence window size for model")
    parser.add_argument("--pad_mode", type=str, default="NO_PAD", choices= ["ZEROS", "NO_PAD", "REPLICATE"],
                        help="padding mode for sequential image generation.")
    parser.add_argument("--foldname", type=str, default="Original_Fold", choices= ["Data_unlabel", "Data_unlabel_fold1", "Data_unlabel_fold2", "Data_unlabel_fold3"],
                        help="choose fold name.")
    parser.add_argument("--SeqAug_flag", default=False, type=bool,
                        help="Whether to use sequence augmentation.")
    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    # model params
    parser.add_argument("--layer_num", default=3, type=int,help="Number of layers in MST-former.")
    parser.add_argument("--block_depth", default=1, type=int,help="Number of blocks for each scale.")
    parser.add_argument("--merge_ratio", default=4, type=int,help="How many patches will be combined during the assemble layer.")
    parser.add_argument("--TEM_flag", default=True, type=bool,
                        help="Whether to use time sensitive attention.")
    parser.add_argument("--TEM_a", default=0.5, type=float,
                        help="TEM param.")
    parser.add_argument("--TEM_b", default=0.5, type=float,
                        help="TEM param.")
                        parser.add_argument("--useTime", default=True, type=bool,
                        help="Whether to use time interval information.")
    parser.add_argument("--time_emd_type", type=str, default="year", choices= ["year", "month", "day"],  # default setting is year
                        help="choose loss type.")
    parser.add_argument("--useAttenFlag", default=False, type=bool,
                        help="Whether to use attention maps.")
    parser.add_argument("--usePolarFlag", default=False, type=bool,
                        help="Whether to use polar images.")
    # loss params
    parser.add_argument("--loss_type", type=str, default="BalancedSoftmaxCE", choices= ["CrossEntropyLoss", "BCELoss", "BalancedSoftmaxCE"],
                        help="choose loss type.")
    parser.add_argument("--BalancedSoftmaxCE_drw", default=True, type=bool,
                        help="Whether to use two-stage training.")
    parser.add_argument("--BalancedSoftmaxCE_drw_epoch", default=5, type=int,
                        help="When to start two-stage training.")
    parser.add_argument("--BalancedSoftmaxCE_T", default=2.0, type=float,  # 2.0 is the selected best
                        help="Temperature to control balanced weights.")
    parser.add_argument("--Exp_label_loss_flag", default=False, type=bool,
                        help="Whether to use exponential label supervised loss.")
    parser.add_argument("--Exp_label_loss_weight", default=10, type=float,
                        help="Exponential label supervised loss weight.")
    parser.add_argument("--warmup_epoch", default=20, type=int,
                        help="When to start training forecast head loss.")
    # training params
    parser.add_argument("--train_batch_size", default=5, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=5, type=int, 
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=1, type=int, # 5
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")
    parser.add_argument("--learning_rate", default=3*1e-4, type=float,  # 3e-2
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,  
                        help="Total number of training steps to perform.")
    parser.add_argument("--num_epoch", default=300, type=int,  
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--grad_clip_flag", default=True, type=bool,
                        help="Flag of Max gradient norm.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    # AC strategy params
    parser.add_argument("--flag_strategy", default=False, type=bool,
                        help="Whether to use AC strategy.")
    parser.add_argument("--flag_strategy_lr", default=False, type=bool,
                        help="Whether to use diff learning rates for AC strategy.")
    parser.add_argument("--num_strategy", default=2, type=int, # 5
                        help="Num of taking AC strategy.")
    parser.add_argument("--throw_rate", default=[0.5,0.5,0.8,0.8,0.8,0.8], type=list, # [0.5,0.5,0.8,0.8,0.8]
                        help="Rate to throw imbalanced sample.")
    parser.add_argument("--strategy_epoch_duration", default=[100,100,10,10,10,10], type=list,  # [3,10,10,10,10]
                        help="epoch duration for every strategy.")
    parser.add_argument("--strategy_learn_rate", default=[1*1e-7, 4*1e-7, 4*1e-6, 4*1e-6, 4*1e-5, 2*1e-4], type=float,  # [1*1e-7, 4*1e-7, 4*1e-6, 4*1e-6, 4*1e-5, 2*1e-4]
                        help="lr for every strategy.")
    # evaluation params
    parser.add_argument("--better_model_criterion", default=[1.0,0.5,0.5,0.1], type=list,  # AUC, acc, sen, spe
                        help="how to decide better model.")
    parser.add_argument("--flag_use_posCoef", default=False, type=bool,
                        help="Whether to use positive coefficient to determine class.")
    parser.add_argument("--Pos_coef_selected", default=0.5, type=float,  # 0.00023
                        help="Selected positive sample thres.")
    parser.add_argument("--Pos_coef_list", default=[(i+1)*0.1 for i in range(10)], type=list, 
                        help="Adjust pos determine coef to improve Sen.")
    # dist params
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    args = parser.parse_args()

    # start training
    MSTformer_trainer = Trainer(args)
    MSTformer_trainer.train()
    


