import os
import numpy as np
from torchvision import transforms as T
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import glob
from PIL import Image
from tqdm import tqdm
import cv2
import platform
import itertools as it
import random
import ipdb
# from random import choice

class fundus_DataSet(Dataset):
    def __init__(self, 
                 path_to_image, 
                 path_to_label, 
                 pred_xYear=2, 
                 seq_len=5, 
                 path_to_atten=None, 
                 path_to_polar=None, 
                 transform=None, 
                 seq_aug_flag=True, 
                 pad_mode='ZEROS',
                 time_emd_type='year',
                 TEM_a = 0.5,
                 TEM_b = 0.5,
                 fix_data_num = None):
        self.path_to_image = path_to_image
        self.path_to_label = path_to_label
        self.path_to_atten = path_to_atten
        self.path_to_polar = path_to_polar
        self.time_emd_type = time_emd_type
        self.list_img_path = self.get_file_list(path_to_label,'SD')
        self.eye_image_dict, self.eye_atten_dict, self.eye_polar_dict, self.eye_label_dict, self.eye_year_dict, self.eye_imgpath_dict, self.eye_name_dict, self.eye_isGla_dict, self.eye_GlaYear_dict = self._revover_full_seq()
        self.transform = transform
        self.pred_xYear = pred_xYear
        self.seq_len = seq_len
        self.TEM_a = TEM_a
        self.TEM_b = TEM_b
        self.fix_data_num = fix_data_num

        self.image_pool = [] # store image with a seq level (seq_len = 5)
        self.atten_pool = []
        self.polar_pool = []
        self.next_label_pool = [] # store the image class label at T (seq_len+1) time
        self.next_seq_label_pool = [] # store the next timestamp image label in a sequence way
        self.label_seq_pool = []
        # self.currect_label_pool = []
        self.year_pool = [] # store the delta year sequence
        self.future_year_tag_pool = [] # store the time interval at the next timestamp
        self.img_name_pool = []
        self.isGla_pool = [] # store the flag whether this eye will become glaucoma eye
        self.GlaYear_pool = [] # store the time interval (not normalized by 30) from the last inspection time in the sequence to the glaucoma inspection time, if this is not a glaucoma eye, the time interval is -15
        self.isGla_Xyear_pool = [] # store the flag whether this eye will become glaucoma eye after X years
        self._read_seq_img_label_into_memory(seq_len=seq_len,seq_aug_flag=seq_aug_flag,pad_mode=pad_mode)
        self._display_data_info()
        assert len(self.image_pool) == len(self.next_seq_label_pool)
      
        
    def __getitem__(self, index):
        img_seq_single = self.image_pool[index]
        next_label_single = self.next_label_pool[index]
        next_seq_label_single = self.next_seq_label_pool[index]
        year_seq_single = self.year_pool[index]
        future_year_seq_single = self.future_year_tag_pool[index]

        # Normalize the time info
        if self.time_emd_type == "year":
            year_seq_single_Nor = year_seq_single / 30.0 # the maximum deta year is 30
            future_year_tag = future_year_seq_single / 30.0
        elif self.time_emd_type == "month":
            year_seq_single_Nor = year_seq_single / (30.0*12) # the maximum deta month is 30*12
            future_year_tag = future_year_seq_single / (30.0*12)
        elif self.time_emd_type == "day":
            year_seq_single_Nor = year_seq_single / (30.0*365) # the maximum deta day is 30*365
            future_year_tag = future_year_seq_single / (30.0*365)
            
        time_matrix_single_1 = np.zeros((self.seq_len,self.seq_len))
        for i in range(self.seq_len):
            for j in range(self.seq_len):
                time_matrix_single_1[i][j] = year_seq_single_Nor[i] - year_seq_single_Nor[j]
                
        time_matrix_single = 1.0 / (1+np.exp(self.TEM_a*time_matrix_single_1-self.TEM_b))
        
        imgname_seq_single = self.img_name_pool[index]
        atten_seq_single = self.atten_pool[index]
        polar_seq_single = self.polar_pool[index]
        label_seq_single = self.label_seq_pool[index]
        Gla_label = self.isGla_pool[index]
        Gla_year = self.GlaYear_pool[index]
        isGla_Xyear = self.isGla_Xyear_pool[index]
       
        sample_seq = {'image':img_seq_single, 
                      'isGla_Xyear':isGla_Xyear, 
                      'gla_eye_label':Gla_label, 
                      'gla_eye_year':Gla_year, 
                      'next_label':next_label_single, 
                      'delta_year':year_seq_single, 
                      'future_year_tag':future_year_tag,
                      'img_name':imgname_seq_single, 
                      'atten':atten_seq_single, 
                      'polar':polar_seq_single, 
                      'label_seq':label_seq_single,
                      'next_seq_label':next_seq_label_single,
                      'Tmatrix':time_matrix_single}

        return sample_seq


    def __len__(self):
        if self.fix_data_num is not None:
            return self.fix_data_num
        else:
            return len(self.image_pool)
    

    def _display_data_info(self):
        t_variant_eye_list = []
        t_invariant_eye_list = []
        for eye_id in self.eye_image_dict.keys():
            label_seq = self.eye_label_dict[eye_id]
            if 1 in label_seq:
                t_variant_eye_list.append(eye_id)
            else:
                t_invariant_eye_list.append(eye_id)
        t_variant_eye_list.sort()
        t_invariant_eye_list.sort()
        print("there are {} total seqs, {} t_variant seqs, {} t_invariant seqs\n".format(len(self.eye_image_dict), len(t_variant_eye_list),len(t_invariant_eye_list)))
        

    def _read_seq_img_label_into_memory(self, seq_len, seq_aug_flag, max_aug_ratio=20, pad_mode="ZEROS"):
        with tqdm(total=len(self.eye_image_dict)) as pbar:
            pbar.set_description('Reading Seq into memory')
            for eye_id in self.eye_image_dict.keys():
                # print('Generating sequence from eye--{}'.format(eye_id))
                img_seq = self.eye_image_dict[eye_id]
                label_seq = self.eye_label_dict[eye_id]
                year_seq = self.eye_year_dict[eye_id]
                imgname_seq = self.eye_name_dict[eye_id]
                atten_seq = self.eye_atten_dict[eye_id]
                polar_seq = self.eye_polar_dict[eye_id]
                # if 1 in label_seq:
                #     seq_end_id = label_seq.index(1) + 1
                #     # seq_end_id = label_seq.index(1)
                # else:
                #     seq_end_id = len(label_seq)
                
                seq_end_id = len(label_seq) # In deepGF version, pos sample will be considered into input sequence
                
                for i in range(1,seq_end_id):
                    if i-seq_len<0:
                        if pad_mode == "ZEROS":
                            pad_img = self._return_pad_img(img_seq[0].shape,seq_len-i)
                            pad_atten = self._return_pad_img(atten_seq[0].shape,seq_len-i)
                            pad_polar = self._return_pad_img(polar_seq[0].shape,seq_len-i)
                            pad_year = [0 for i in range(seq_len-i)]
                            pad_imgname = ['PAD_imgname' for i in range(seq_len-i)]
                            pad_seq_label = [0 for i in range(seq_len-i)]
                        elif pad_mode == "REPLICATE":
                            pad_img = self._return_replica_pad_img(img_seq[0],seq_len-i)
                            pad_atten = self._return_replica_pad_img(atten_seq[0],seq_len-i)
                            pad_polar = self._return_replica_pad_img(polar_seq[0],seq_len-i)
                            pad_year = [0 for i in range(seq_len-i)]
                            pad_imgname = [imgname_seq[0] for i in range(seq_len-i)]
                            pad_seq_label = [label_seq[0] for i in range(seq_len-i)]
                        elif pad_mode == "NO_PAD":
                            pad_img = "NO_PAD"
                            pad_year = "NO_PAD"
                            pad_imgname = "NO_PAD"
                            pad_seq_label = "NO_PAD"
                        else:
                            raise ValueError("Please choose right PADDING type!")
                        # pad_year = [0 for i in range(seq_len-i)]
                        # pad_imgname = ['PAD_imgname' for i in range(seq_len-i)]
                    else: pad_img = None

                    # self.next_label_pool.append(label_seq[i])
                    if pad_img is not None and pad_mode != "NO_PAD":
                        self.image_pool.append(np.array(pad_img+img_seq[:i]))
                        self.year_pool.append(np.array(pad_year+year_seq[:i]))
                        self.future_year_tag_pool.append(np.array(pad_year[1:]+year_seq[:i+1]))
                        self.atten_pool.append(np.array(pad_atten+atten_seq[:i]))
                        self.polar_pool.append(np.array(pad_polar+polar_seq[:i]))
                        self.img_name_pool.append({'seq name':pad_imgname+imgname_seq[:i]})
                        self.next_label_pool.append(label_seq[i])
                        self.label_seq_pool.append(np.array(pad_seq_label+label_seq[:i]))
                        self.isGla_pool.append(self.eye_isGla_dict[eye_id])
                        if self.eye_isGla_dict[eye_id] == 1:
                            self.GlaYear_pool.append(self.eye_GlaYear_dict[eye_id] - self.year_pool[-1][-1])
                        else:
                            self.GlaYear_pool.append(self.eye_GlaYear_dict[eye_id])
                            
                        if self.isGla_pool[-1]==1 and self.GlaYear_pool[-1] < self.pred_xYear: # this is a glaucoma eye and the time interval between the last image and onset less than required years
                            self.isGla_Xyear_pool.append(1)
                        elif self.isGla_pool[-1]==1 and self.GlaYear_pool[-1] > self.pred_xYear:
                            self.isGla_Xyear_pool.append(0)
                        else:
                            self.isGla_Xyear_pool.append(0)
                        
                        
                    elif pad_img is not None and pad_mode == "NO_PAD":
                        continue                   
                    
                    else:
                        self.image_pool.append(np.array(img_seq[i-seq_len:i]))
                        self.year_pool.append(np.array(year_seq[i-seq_len:i]))
                        self.future_year_tag_pool.append(np.array(year_seq[i-seq_len+1:i+1]))
                        self.atten_pool.append(np.array(atten_seq[i-seq_len:i]))
                        self.polar_pool.append(np.array(polar_seq[i-seq_len:i]))
                        self.img_name_pool.append({'seq name':imgname_seq[i-seq_len:i]})
                        self.next_label_pool.append(label_seq[i])
                        self.next_seq_label_pool.append(np.array(label_seq[i-seq_len+1:i+1]))
                        self.label_seq_pool.append(np.array(label_seq[i-seq_len:i]))
                        self.isGla_pool.append(self.eye_isGla_dict[eye_id])
                        # self.GlaYear_pool.append(self.eye_GlaYear_dict[eye_id])
                        if self.eye_isGla_dict[eye_id] == 1:
                            self.GlaYear_pool.append(self.eye_GlaYear_dict[eye_id] - self.year_pool[-1][-1])
                        else:
                            self.GlaYear_pool.append(self.eye_GlaYear_dict[eye_id])

                        if self.isGla_pool[-1]==1 and self.GlaYear_pool[-1] < self.pred_xYear: 
                            self.isGla_Xyear_pool.append(1)
                        elif self.isGla_pool[-1]==1 and self.GlaYear_pool[-1] > self.pred_xYear:
                            self.isGla_Xyear_pool.append(0)
                        else:
                            self.isGla_Xyear_pool.append(0)
                            
                        # sequential augmentation.
                        # if seq_aug_flag and label_seq[i] == 1: 
                        if seq_aug_flag and self.eye_isGla_dict[eye_id] == 1: 
                            all_idx = []
                            for e in it.combinations(np.arange(i-1),seq_len-1):
                                all_idx.append(list(e))
                            if len(all_idx)>max_aug_ratio:
                                all_idx = random.sample(all_idx,max_aug_ratio)
                            # print("Sequential augmentation: {} additional seqs...".format(len(all_idx)))
                            for idx in all_idx:
                                idx.append(i-1)
                                self.next_label_pool.append(label_seq[i])
                                self.next_seq_label_pool.append(np.array([label_seq[i+1] for i in idx]))
                                self.image_pool.append(np.array([img_seq[i] for i in idx])) 
                                self.year_pool.append(np.array([year_seq[i] for i in idx])) 
                                self.future_year_tag_pool.append(np.array([year_seq[i+1] for i in idx])) 
                                self.atten_pool.append(np.array([atten_seq[i] for i in idx])) 
                                self.polar_pool.append(np.array([polar_seq[i] for i in idx])) 
                                self.img_name_pool.append({'seq name':[imgname_seq[i] for i in idx]}) 
                                self.label_seq_pool.append(np.array([label_seq[i] for i in idx]))
                                self.isGla_pool.append(self.eye_isGla_dict[eye_id])
                                # self.GlaYear_pool.append(self.eye_GlaYear_dict[eye_id])
                                self.GlaYear_pool.append(self.eye_GlaYear_dict[eye_id] - self.year_pool[-1][-1])
                                
                                if self.isGla_pool[-1]==1 and self.GlaYear_pool[-1] < self.pred_xYear: 
                                    self.isGla_Xyear_pool.append(1)
                                elif self.isGla_pool[-1]==1 and self.GlaYear_pool[-1] > self.pred_xYear:
                                    self.isGla_Xyear_pool.append(0)
                                else:
                                    self.isGla_Xyear_pool.append(0)
                pbar.update(1)        
                    


    def _revover_full_seq(self):
        all_eyes = {}
        for seq_id in self.list_img_path:
        # for seq_id in self.list_img_path[:30]: # this is used for debug mode
            tmp = seq_id.split('_')
            eye_id = tmp[0] + '_' + tmp[1]
            if eye_id not in all_eyes.keys():
                all_eyes[eye_id] = 1
            else:
                all_eyes[eye_id] += 1
        
        eye_name_dict = {}
        eye_imgpath_dict = {}
        eye_image_dict = {}
        eye_label_dict = {}
        eye_year_dict = {}
        eye_time_dict = {}
        eye_atten_dict = {}
        eye_polar_dict = {}
        eye_isGla_dict = {}
        eye_GlaYear_dict = {}
        with tqdm(total=len(all_eyes)) as pbar:
            pbar.set_description('Reorganizing images of eye')
            for eye_id in all_eyes.keys():
                # print('Reorganize images of eye--{}'.format(eye_id))
                eye_name_dict[eye_id] = []
                eye_imgpath_dict[eye_id] = []
                eye_image_dict[eye_id] = []
                eye_label_dict[eye_id] = []
                eye_year_dict[eye_id] = []
                eye_time_dict[eye_id] = []
                eye_atten_dict[eye_id] = []
                eye_polar_dict[eye_id] = []
                
                # for i in range(1,all_eyes[eye_id]+1):
                for i in range(all_eyes[eye_id]):

                    img_subdir_path = os.path.join(self.path_to_image, eye_id + '_' + str(i))
                    label_subdir_txt = os.path.join(self.path_to_label, eye_id + '_' + str(i) + '.txt')

                    image_sublist = glob.glob(img_subdir_path+ '/' + 'SD*')
                    image_sublist.sort()  # fixed bug
                    label_set = self._read_txt(label_subdir_txt)
                    assert len(image_sublist) == len(label_set)

                    for i in range(len(image_sublist)):
                        if platform.system() == 'Windows':
                            image_name = image_sublist[i].split('\\')[-1]
                        else:
                            image_name = image_sublist[i].split('/')[-1]
                            
                        # if eye_id == 'SD1284_OS' and i == 1:
                        #     ipdb.set_trace()
                            
                        if image_name not in eye_name_dict[eye_id]:
                            eye_name_dict[eye_id].append(image_name)
                            image = self._read_transform_img(image_sublist[i])
                            eye_image_dict[eye_id].append(image)
                            eye_imgpath_dict[eye_id].append(image_sublist[i])
                            eye_label_dict[eye_id].append(label_set[i])
                            delta_year_ori = int(os.path.split(image_sublist[i])[-1][7:11]) \
                                            - int(eye_name_dict[eye_id][0][7:11])
                            eye_year_dict[eye_id].append(delta_year_ori)
                            # delta_year = delta_year_ori / 30.0 # the maximum deta year is 30
                            if self.time_emd_type == "year":
                                eye_time_dict[eye_id].append(delta_year_ori)
                            elif self.time_emd_type == "month":
                                delta_month_tmp = int(os.path.split(image_sublist[i])[-1][12:14]) - int(eye_name_dict[eye_id][0][12:14])
                                delta_month_ori = delta_year_ori*12 + delta_month_tmp
                                eye_time_dict[eye_id].append(delta_month_ori)
                            elif self.time_emd_type == "day":
                                delta_month_tmp = int(os.path.split(image_sublist[i])[-1][12:14]) - int(eye_name_dict[eye_id][0][12:14])
                                delta_day_tmp = int(os.path.split(image_sublist[i])[-1][15:17]) - int(eye_name_dict[eye_id][0][15:17])
                                delta_day_ori = delta_year_ori*365 + delta_month_tmp*30 + delta_day_tmp
                                eye_time_dict[eye_id].append(delta_day_ori)
                            # print(image_name+'\n')

                            atten_path = os.path.join(self.path_to_atten, image_name.replace("JPG","jpg"))
                            atten_map = self._read_transform_atten(atten_path)
                            eye_atten_dict[eye_id].append(atten_map)
                            polar_path = os.path.join(self.path_to_polar, image_name.replace("JPG","jpg"))
                            polar_map = self._read_transform_polar(polar_path)
                            eye_polar_dict[eye_id].append(polar_map)

                if 1 in eye_label_dict[eye_id]:
                    eye_isGla_dict[eye_id] = 1
                    eye_GlaYear_dict[eye_id] = eye_year_dict[eye_id][eye_label_dict[eye_id].index(1)] # obtain the year info of the first glaucoma inspection
                else:
                    eye_isGla_dict[eye_id] = 0
                    # eye_GlaYear_dict[eye_id] = -15 / 30.0
                    eye_GlaYear_dict[eye_id] = -15

                pbar.update(1)

        return eye_image_dict, eye_atten_dict, eye_polar_dict, eye_label_dict, eye_time_dict, eye_imgpath_dict, eye_name_dict, eye_isGla_dict, eye_GlaYear_dict # eye_year_dict
    

    def _read_txt(self,txt_filepath):
        label_set = []
        with open(txt_filepath, 'r') as f:
            K = f.readlines()
            for i_line in range(6):
                line= K[i_line]
                line = line.strip('\n')
                line = int(line)
                label_set.append(line)
        return label_set


    def _read_transform_img(self,imgpath):
        image = Image.open(imgpath)
        image = image.resize((224, 224))
        image = np.asarray(image, np.uint8)
        image = np.transpose(image,(2,0,1))
        image = image / 255.0
        return image
    

    def _read_transform_atten(self,imgpath):
        image = Image.open(imgpath)
        image = image.resize((224, 224))
        image = np.asarray(image, np.uint8)
        image = np.transpose(image,(2,0,1))
        image = image / 255.0
        return image
    

    def _read_transform_polar(self,imgpath):
        image = Image.open(imgpath)
        image = image.resize((224, 224))
        image = np.asarray(image, np.uint8)
        image = np.transpose(image,(2,0,1))
        image = image / 255.0
        return image
    

    def _return_pad_img(self,img_size,pad_len):
        pad_img_single = np.zeros(img_size)
        pad_img = []
        for i in range(pad_len):
            pad_img.append(pad_img_single)
        
        return pad_img 
    
    def _return_replica_pad_img(self,first_img,pad_len):
        # pad_img = []
        # for i in range(pad_len):
        #     pad_img.append(first_img)
            
        pad_img = [first_img for i in range(pad_len)]
        
        return pad_img 

    def get_file_list(self, dir_path, suffix):
        file_list = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.startswith(suffix):
                    file_list.append(file[:-4])
        return file_list


    def get_label_pool(self):
        return self.next_label_pool
    
    def get_seq_next_label_pool(self):
        return self.next_seq_label_pool
    

    def get_statistic_from_dict(self,image_dict):
        image_pool = []
        for eye_id in image_dict.keys():
            image_pool += image_dict[eye_id]

        R,G,B = 0. , 0. , 0.
        R_2,G_2,B_2 = 0. , 0. , 0.
        image_pool_array = np.array(image_pool)
        image_pool_array = np.transpose(image_pool_array,(1,0,2,3))
        n_pixel = image_pool_array.shape[1]* image_pool_array.shape[2]* image_pool_array.shape[3]
        one_order =  np.sum(image_pool_array,axis=(1,2,3))
        second_order = np.sum(np.power(image_pool_array,2.0),axis=(1,2,3))
        mean = one_order / n_pixel
        std = np.sqrt(second_order/n_pixel - np.square(mean))
        return mean.tolist(), std.tolist()
    

    def img_norm(self,img,img_mean,img_std):
        for c in range(3):
            img[c,:,:] = (img[c,:,:] - img_mean[c]) / img_std[c]
        return img



# ############################### test dataloader complies the original folder ##################################################

    

# class fundus_DataSet_test_fixLen(Dataset):
#


############################### dataloader after throwing stage ##################################################


class fundus_DataSet_AC(Dataset):
    def __init__(self, image_pool=None, 
                 atten_pool=None, polar_pool=None, 
                 isGla_Xyear_pool=None, next_label_pool=None, 
                 year_pool=None, label_seq_pool=None, 
                 imgname_pool=None, gla_pool=None, 
                 gla_year_pool=None,
                 next_seq_label_pool=None):
        
        self.image_pool = image_pool 
        # self.atten_pool = []
        # self.polar_pool = []
        self.next_label_pool = next_label_pool 
        self.next_seq_label_pool = next_seq_label_pool
        self.year_pool = year_pool
        self.imgname_pool = imgname_pool 
        self.atten_pool = atten_pool 
        self.polar_pool = polar_pool 
        # self.currect_label_pool = []
        # self.isGla_pool = gla_pool
        self.GlaYear_pool = gla_year_pool
        # self.isGla_Xyear_pool = isGla_Xyear_pool
        self.label_seq_pool = label_seq_pool

    def __getitem__(self, index):
        img_single = self.image_pool[index]
        next_label_single = self.next_label_pool[index]
        next_seq_label_single = self.next_seq_label_pool[index]
        label_seq_single = self.label_seq_pool[index]
        year_single = self.year_pool[index]
        imgname_single = self.imgname_pool[index]
        atten_single = self.atten_pool[index]
        polar_single = self.polar_pool[index]
        # Gla_label = self.isGla_pool[index]
        Gla_year = self.GlaYear_pool[index]
       
        
        sample_seq = {'image':img_single, 
                    #   'isGla_Xyear':isGla_Xyear, 
                      'gla_eye_year':Gla_year, 
                      'next_label':next_label_single, 
                      'delta_year':year_single, 
                      'img_name':imgname_single, 
                      'atten':atten_single, 
                      'polar':polar_single, 
                      'label_seq':label_seq_single,
                      'next_seq_label':next_seq_label_single}
        
        return sample_seq

    def __len__(self):
        return len(self.image_pool)
    
    def get_label_pool(self):
        return self.next_label_pool