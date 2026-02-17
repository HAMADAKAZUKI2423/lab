import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
#from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import cv2
import csv
import pathlib
import re

import pandas as pd
#import os
from scipy.stats import spearmanr, pearsonr
import scipy.io as sio

use_aggregated_data = True
# perfect_cv = False



def randomflip(imglist):
    width = imglist[0].shape[1]
    height = imglist[0].shape[0]
    if np.random.rand()>0.5:
        #vertical flip
        for i in range(len(imglist)):
            imglist[i]=cv2.flip(imglist[i], 0)
        #fg = torch.flip(fg,dims=[2])#BCHW
        #bg = torch.flip(bg,dims=[2])
    
    if np.random.rand()>0.5:
        #horizontal flip
        for i in range(len(imglist)):
            imglist[i]=cv2.flip(imglist[i], 1)
        # fg = torch.flip(fg,dims=[3])#BCHW
        # bg = torch.flip(bg,dims=[3])
    
     
    return imglist#o_fg, o_bg, o_mask

def randomtransform_scale10(imglist):
    width = imglist[0].shape[1]
    height = imglist[0].shape[0]
    # if np.random.rand()>0.5:
    #     #vertical flip
    #     for i in range(len(imglist)):
    #         imglist[i]=cv2.flip(imglist[i], 0)
    #     #fg = torch.flip(fg,dims=[2])#BCHW
    #     #bg = torch.flip(bg,dims=[2])
    
    # if np.random.rand()>0.5:
    #     #horizontal flip
    #     for i in range(len(imglist)):
    #         imglist[i]=cv2.flip(imglist[i], 1)
    #     # fg = torch.flip(fg,dims=[3])#BCHW
    #     # bg = torch.flip(bg,dims=[3])
    
    if True:
        #回転角を指定
        angle = 0.0#np.random.rand()*360.0
        #スケールを指定
        scale = 0.9 + np.random.rand()*0.2
        #scale = 0.8 + np.random.rand()*0.4
        # rand_num = np.random.rand()
        # if rand_num <= 0.33:
        #     #拡大
        #     scale = np.random.rand()*0.25 + 1.0
        # elif rand_num >=0.67:
        #     #縮小
        #     scale = 1.0 - np.random.rand()*0.5
        # else:
        #     scale = 1.0

        center = (width//2, height//2)

        #getRotationMatrix2D関数を使用
        trans = cv2.getRotationMatrix2D(center, angle , scale)
        
        #アフィン変換実行
        for i in range(len(imglist)):
            if i==2 or i==5:
                imglist[i] = cv2.warpAffine(imglist[i], trans, (width,height))#マスクはリピートされたら困る
            else:
                imglist[i] = cv2.warpAffine(imglist[i], trans, (width,height), borderMode=cv2.BORDER_REFLECT)
        #o_bg = cv2.warpAffine(bg, trans, (mask.shape[1],mask.shape[0]), borderMode=cv2.BORDER_REFLECT)
        #o_mask = cv2.warpAffine(mask, trans, (mask.shape[1],mask.shape[0]))
    if False:
        #平行移動
        x_trans = np.random.rand()*10.0-5.0
        y_trans = np.random.rand()*10.0-5.0
        trans = np.float32([[1,0,x_trans],[0,1,y_trans]])

        for i in range(len(imglist)):
            if i==2 or i==5:
                imglist[i] = cv2.warpAffine(imglist[i], trans, (width,height))
            else:
                imglist[i] = cv2.warpAffine(imglist[i], trans, (width,height), borderMode=cv2.BORDER_REFLECT)
        # o_fg = cv2.warpAffine(o_fg, trans, (mask.shape[1],mask.shape[0]), borderMode=cv2.BORDER_REFLECT)
        # o_bg = cv2.warpAffine(o_bg, trans, (mask.shape[1],mask.shape[0]), borderMode=cv2.BORDER_REFLECT)
        # o_mask = cv2.warpAffine(o_mask, trans, (mask.shape[1],mask.shape[0]))

        
    return imglist#o_fg, o_bg, o_mask

def randomtransform_scale20(imglist):
    width = imglist[0].shape[1]
    height = imglist[0].shape[0]
    

    #回転角を指定
    angle = 0.0#np.random.rand()*360.0
    #スケールを指定
    #scale = 0.9 + np.random.rand()*0.2
    scale = 0.8 + np.random.rand()*0.4
    
    center = (width//2, height//2)

    #getRotationMatrix2D関数を使用
    trans = cv2.getRotationMatrix2D(center, angle , scale)
    
    #アフィン変換実行
    for i in range(len(imglist)):
        if i==2 or i==5:
            imglist[i] = cv2.warpAffine(imglist[i], trans, (width,height))#マスクはリピートされたら困る
        else:
            imglist[i] = cv2.warpAffine(imglist[i], trans, (width,height), borderMode=cv2.BORDER_REFLECT)

    return imglist#o_fg, o_bg, o_mask


class VisDataset(Dataset):
    #@profile
    def __init__(self, condition_list, path_to_data, exclude_high_alpha=False, transform=None, consistent_transform=False):
        #make a list of the raw data
        self.transform=transform
        self.consistent_transform = consistent_transform

        imgsize = 256
        e_sigma = imgsize/4/2
        axis = np.arange(-imgsize/2,imgsize/2)
        xx,yy = np.meshgrid(axis,axis)
        r_2 = xx*xx+yy*yy
        r = np.sqrt(r_2)
        #mask = np.exp(-r_2/(2.0*e_sigma**2.))
        mask_large = np.float32(1.-1./(1+np.exp(30.-(r/3.))))
        #mask_large = np.float32(np.dstack((mask_large,mask_large,mask_large)))

        self.mask_large = cv2.resize(mask_large,(imgsize,imgsize),interpolation=cv2.INTER_LINEAR)
        self.mask_small = cv2.resize(mask_large,(imgsize//2,imgsize//2),interpolation=cv2.INTER_LINEAR)

        self.path_to_img = path_to_data + 'patches/'
        self.agg_df = pd.read_csv(path_to_data + 'mean_data.csv')


        self.use_texture_only = False
        #self.pass_rawImage = True #weight計算などのために元画像をbatchに含める

        self.scale_invariant_mode = False

        if self.use_texture_only:
            filter_bg_list = ['dtd']
            filter_ref_fg_list = ['dtd']
            filter_test_fg_list = ['dtd']
        else:
            filter_bg_list = []
            filter_ref_fg_list = []
            filter_test_fg_list = []

            

        #ここでcross validation用にデータ分割用ID振り分けを行う
        
        self.agg_df = self.agg_df.assign(cv_group_id=-1)


        self.N_fold = 5

        
        self.agg_df.loc[self.agg_df["condition"]=="same",'cv_group_id'] = self.agg_df[self.agg_df["condition"]=="same"]['fold_id']
        self.agg_df.loc[self.agg_df["condition"]=="different",'cv_group_id'] = self.agg_df[self.agg_df["condition"]=="different"]['fold_id']


        for filter_name in filter_bg_list:
            self.agg_df = self.agg_df[self.agg_df['test background'].str.contains(filter_name)]
        for filter_name in filter_ref_fg_list:
            self.agg_df = self.agg_df[self.agg_df['reference background'].str.contains(filter_name)]
        for filter_name in filter_test_fg_list:
            self.agg_df = self.agg_df[self.agg_df['test foreground'].str.contains(filter_name)]
    
        tmp_df = pd.DataFrame()
        for cond_name in condition_list:
            tmp_df = pd.concat([tmp_df, self.agg_df[self.agg_df["condition"]==cond_name]])
            # tmp_df = tmp_df.append(self.agg_df[self.agg_df["condition"]==cond_name])
        self.agg_df = tmp_df

        if exclude_high_alpha:
            self.agg_df = self.agg_df[self.agg_df['response_mean']<0.9]

        # self.agg_df=pd.concat([self.agg_df, self.standard_all, self.natural_all])
        # self.agg_df=pd.concat(self.agg_df)


        
        # self.condname_strict_cv = []
        

        print(self.agg_df[0:10])
        #print(self.agg_df.response)

        
        self.agg_df = self.agg_df.assign(new_index=-1)
        self.agg_df = self.agg_df.assign(unique_index=-1)
        #self.condition_index_dict = {}

        #データ（条件）数を計算しておく
        self.conditions = self.agg_df["condition"].unique()
        self.N_combinations = 0
        self.combination_list = {}
        N_unique_ind = 0
        for cond in self.conditions:
        
            self.agg_df.loc[self.agg_df['condition']==cond, 'new_index'] = self.agg_df[self.agg_df["condition"]==cond]["combination id"]+self.N_combinations
            
            self.combination_list[cond]=self.agg_df[self.agg_df["condition"]==cond]["combination id"].unique()
            self.combination_list[cond].sort()
            #self.condition_index_dict[cond] = [self.N_combinations, self.N_combinations+len(self.combination_list[cond])]
            self.N_combinations += len(self.combination_list[cond])

            self.agg_df.loc[self.agg_df['condition']==cond, 'unique_index'] = self.agg_df[self.agg_df["condition"]==cond]["subcondition id"]+N_unique_ind
            N_unique_ind += len(self.agg_df[self.agg_df["condition"]==cond]["subcondition id"].unique())

        self.agg_df = self.agg_df.assign(weight=1.0)
    

    def __len__(self):
        return self.N_combinations        
        
    def get_condition_indices(self, cond_name):
        return sorted(self.agg_df[self.agg_df["condition"]==cond_name]["new_index"].unique())
        
    def get_cv_indices(self, cv_group_id):
        return sorted(self.agg_df[self.agg_df["cv_group_id"]==cv_group_id]["new_index"].unique())
    
    def get_condition_cv_indices(self, cond_name, cv_group_id):
        return sorted(self.agg_df[(self.agg_df["condition"]==cond_name) & (self.agg_df["cv_group_id"]==cv_group_id)]["new_index"].unique())

    def set_weight_vals(self, input_df):
        for trial in input_df.itertuples():
            self.agg_df.loc[self.agg_df['unique_index']==trial.unique_index, 'weight'] = trial.new_weight
    def init_weight_vals(self):
        self.agg_df['weight'] = 1.0
    
    #def add_exclusive_data(num, alpha=0):


    def recalculate_weights(self, results_df):

        fig, axs = plt.subplots()
        stdval = results_df["loss_per_example"].std()
        mean_val = results_df["loss_per_example"].mean()
        #target_tensor=np.clip(target_tensor,mean_val-3*stdval,mean_val+3*stdval)

        axs.hist(results_df["loss_per_example"], bins=50, density=True)
        #axs.set_xlim(mean_val-3*stdval,mean_val+3*stdval)
        plt.show()

        #compute weights
        tolerable_loss = results_df['loss_per_example'].median()*10.0
        print("tolerable loss:",tolerable_loss)
        results_df['new_weight']=results_df['loss_per_example']
        #print(_df[['weight','loss_per_example']][0:30])
        results_df.loc[results_df['new_weight']>tolerable_loss, 'new_weight'] = tolerable_loss
        #print(_df[['weight','loss_per_example']][0:30])
        results_df['new_weight']=(1.0-(results_df['new_weight']/tolerable_loss)**2.0)**2.0
        #print(results_df[['unique_index','new_weight','loss_per_example','ref_val','test_val']][0:30])

        self.set_weight_vals(results_df)

    #@profile
    def __getitem__(self, idx):
        
        #indexはcondition_id毎に1つ割り当てる(前景背景の組が同じでref_alphaが異なる条件は同じindexにまとめられる)
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        batch_df = self.agg_df[self.agg_df['new_index']==idx]
        #batch_df = tmp_df[tmp_df["condition_id"]==trial_condition_id]
        
        batch_dict={'ref_fg': [],
                    #'ref': [],
                    'ref_bg': [],
                    #'ref_DC': [],
                    'test_fg': [],
                    'test_bg': [],
                    'ref_mask': [],
                    'test_mask': [],
                    'ref_fg_label': [],
                    'ref_bg_label': [],
                    'test_fg_label': [],
                    'test_bg_label': [],
                    'ref_alpha': [],
                    'test_alpha': [],
                    'response_std': [],
                    'condition': [],
                    'unique_index': [],
                    'weight': []}
                    #'dist': []}

        ref_fg_list = []
        ref_bg_list = []

        first_exmaple = True
        for tid, trial in batch_df.iterrows():
            #print(trial)

            if first_exmaple:

                path_to_stimuli = self.path_to_img

                #read info about reference image
                #ref_alpha = trial.ref_alpha
            
                ref_fg_name = trial["reference foreground"]
                ref_bg_name = trial["reference background"]
                test_fg_name = trial["test foreground"]
                test_bg_name = trial["test background"]
                

                ref_fg = cv2.imread(path_to_stimuli + ref_fg_name)

                if trial.condition == 'same' or trial.condition == 'different':
                    ref_bg = np.ones(ref_fg.shape)*int(ref_bg_name)
                elif ref_bg_name == 'none':
                    ref_bg = np.ones(ref_fg.shape)*128
                else:
                    ref_bg = cv2.imread(path_to_stimuli + ref_bg_name)


                if ref_fg_name == test_fg_name:
                    test_fg = ref_fg
                else:
                    test_fg = cv2.imread(path_to_stimuli + test_fg_name)

                #read info about test image
                    
                if test_bg_name == 'none':
                    test_bg = np.ones(test_fg.shape)*128
                else:
                    test_bg = cv2.imread(path_to_stimuli + test_bg_name)
                
                t_mask = self.mask_large

                if self.transform:
                    if self.consistent_transform:
                        (ref_fg, ref_bg, ref_mask, test_fg, test_bg, test_mask) = self.transform([ref_fg, ref_bg, t_mask, test_fg, test_bg, t_mask])
                    else:
                        (ref_fg, ref_bg, ref_mask) = self.transform([ref_fg, ref_bg, t_mask])
                        (test_fg, test_bg, test_mask) = self.transform([test_fg, test_bg, t_mask])
                else:
                    ref_mask = t_mask
                    test_mask = t_mask
                
                ref_fg_array = np.float32(ref_fg.transpose([2,0,1]))/255
                ref_bg_array = np.float32(ref_bg.transpose([2,0,1]))/255
                ref_fg_tensor = torch.as_tensor(ref_fg_array)
                ref_bg_tensor = torch.as_tensor(ref_bg_array)
                ref_mask_tensor = torch.as_tensor(ref_mask)

                test_fg_array = np.float32(test_fg.transpose([2,0,1]))/255
                test_bg_array = np.float32(test_bg.transpose([2,0,1]))/255
                test_fg_tensor = torch.as_tensor(test_fg_array)
                test_bg_tensor = torch.as_tensor(test_bg_array)
                test_mask_tensor = torch.as_tensor(test_mask)

            batch_dict['ref_fg'].append(ref_fg_tensor)
            batch_dict['ref_bg'].append(ref_bg_tensor)
            batch_dict['test_fg'].append(test_fg_tensor)
            batch_dict['test_bg'].append(test_bg_tensor)
            batch_dict['ref_mask'].append(ref_mask_tensor.unsqueeze(0))#[CHW]
            batch_dict['test_mask'].append(test_mask_tensor.unsqueeze(0))#[CHW]
            batch_dict['ref_fg_label'].append(trial["reference foreground"])
            batch_dict['ref_bg_label'].append(trial["reference background"])
            batch_dict['test_fg_label'].append(trial["test foreground"])
            batch_dict['test_bg_label'].append(trial["test background"])
            batch_dict['ref_alpha'].append(torch.tensor(trial["reference alpha"],dtype=torch.float32))
            batch_dict['test_alpha'].append(torch.tensor(trial["response_mean"],dtype=torch.float32))
            batch_dict['response_std'].append(torch.tensor(trial["response_std"],dtype=torch.float32))
            batch_dict['condition'].append(trial.condition)
            batch_dict['unique_index'].append(trial.unique_index)
            batch_dict['weight'].append(torch.tensor(trial.weight,dtype=torch.float32))
            #batch_dict['dist'].append(torch.tensor(trial.dist,dtype=torch.float32))
        
        batch_dict['ref_fg'] = torch.stack(batch_dict['ref_fg'])#.to(device=cuda)
        batch_dict['ref_bg'] = torch.stack(batch_dict['ref_bg'])#.to(device=cuda)
        batch_dict['test_fg'] = torch.stack(batch_dict['test_fg'])#.to(device=cuda)
        batch_dict['test_bg'] = torch.stack(batch_dict['test_bg'])#.to(device=cuda)
        batch_dict['ref_mask'] = torch.stack(batch_dict['ref_mask'])#.to(device=cuda)#[BCHW]
        batch_dict['test_mask'] = torch.stack(batch_dict['test_mask'])#.to(device=cuda)#[BCHW]
        batch_dict['ref_alpha'] = torch.stack(batch_dict['ref_alpha'])#.to(device=cuda)#alpha_batch
        batch_dict['test_alpha'] = torch.stack(batch_dict['test_alpha'])#.to(device=cuda)
        batch_dict['weight'] = torch.stack(batch_dict['weight'])#.to(device=cuda)
        batch_dict['response_std'] = torch.stack(batch_dict['response_std'])#.to(device=cuda)
        #batch_dict['dist'] = torch.stack(batch_dict['dist'])#.to(device=cuda)

        # if self.transform:
        #     batch_dict['ref_fg'], batch_dict['ref_bg'] = self.transform(batch_dict['ref_fg'], batch_dict['ref_bg'])
        #     batch_dict['test_fg'], batch_dict['test_bg'] = self.transform(batch_dict['test_fg'], batch_dict['test_bg'])

        #from IPython.core.debugger import Pdb; Pdb().set_trace()
        return batch_dict
