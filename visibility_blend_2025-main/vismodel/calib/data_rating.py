import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

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


class RatingDataset(Dataset):
    #@profile
    def __init__(self, path_to_data, transform=None, consistent_transform=False, use_clahe_for_ref=True, zero_augment=False):

        self.use_clahe_for_ref = use_clahe_for_ref# added 2023/9/26
        # CLAHEの場合にもReference(opaque target)としてclahe画像を使用する

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

        self.path_to_img = path_to_data + 'stimuli_rating_2021_preblend/'
        self.agg_df = pd.read_csv(path_to_data + 'agg_df_rating.csv')


        # self.use_texture_only = False
        # #self.pass_rawImage = True #weight計算などのために元画像をbatchに含める

        # self.scale_invariant_mode = False

        # if self.use_texture_only:
        #     filter_bg_list = ['dtd']
        #     filter_ref_fg_list = ['dtd']
        #     filter_test_fg_list = ['dtd']
        # else:
        #     filter_bg_list = []
        #     filter_ref_fg_list = []
        #     filter_test_fg_list = []

        #ここでcross validation用にデータ分割用ID振り分けを行う

        
        self.agg_df = self.agg_df.assign(cv_group_id=-1)
        

        # combination_idのlistを作る
        combination_id_list = self.agg_df["combination_id"].unique()

        self.N_fold = 5

        # combination_id_listをN_fold分割する
        # random seedを固定しておく
        np.random.seed(0)
        combination_id_list = np.random.permutation(combination_id_list)
        combination_id_list = np.array_split(combination_id_list, self.N_fold)

        # cv_group_idにfold_idを振り分ける
        for fold_id in range(self.N_fold):
            self.agg_df.loc[self.agg_df["combination_id"].isin(combination_id_list[fold_id]),'cv_group_id'] = fold_id
        
        # cv_group_idの型をintに変換
        # self.agg_df = self.agg_df.astype({'cv_group_id':int})

        # alpha=0のdataをaugmentする
        self.zero_augment = zero_augment
        if self.zero_augment:
            # blend_mode=='linear'で，かつlevel=2のデータを抽出
            zero_augment_df = self.agg_df[(self.agg_df["blend_mode"]=='linear') & (self.agg_df["level"]==2)]
            # zero_augment_dfのresponseを1.0にする
            zero_augment_df["response"] = 1.0
            # zero_augment_dfのblend_paramを0.0にする
            zero_augment_df["blend_param"] = 0.0

            #zero_augment_dfのunique_condition_idを振りなおす 1800から始まる連番にする
            id_array = np.arange(1800,1800+len(zero_augment_df))
            zero_augment_df["unique_condition_id"] = id_array

            # zero_augment_dfをagg_dfに追加
            self.agg_df = pd.concat([self.agg_df, zero_augment_df], ignore_index=True)
            
            

        
        print(self.agg_df[0:10])
        #print(self.agg_df.response)

        
        # self.agg_df = self.agg_df.assign(new_index=-1)
        # self.agg_df = self.agg_df.assign(unique_index=-1)
        # #self.condition_index_dict = {}

        # #データ（条件）数を計算しておく
        # self.conditions = self.agg_df["condition"].unique()
        # self.N_combinations = 0
        # self.combination_list = {}
        # N_unique_ind = 0
        # for cond in self.conditions:
        
        #     self.agg_df.loc[self.agg_df['condition']==cond, 'new_index'] = self.agg_df[self.agg_df["condition"]==cond]["combination id"]+self.N_combinations
            
        #     self.combination_list[cond]=self.agg_df[self.agg_df["condition"]==cond]["combination id"].unique()
        #     self.combination_list[cond].sort()
        #     #self.condition_index_dict[cond] = [self.N_combinations, self.N_combinations+len(self.combination_list[cond])]
        #     self.N_combinations += len(self.combination_list[cond])

        #     self.agg_df.loc[self.agg_df['condition']==cond, 'unique_index'] = self.agg_df[self.agg_df["condition"]==cond]["subcondition id"]+N_unique_ind
        #     N_unique_ind += len(self.agg_df[self.agg_df["condition"]==cond]["subcondition id"].unique())

        # self.agg_df = self.agg_df.assign(weight=1.0)
    

    def __len__(self):
        return len(self.agg_df)  
        
    def get_condition_indices(self, cond_name):
        return sorted(self.agg_df[self.agg_df["blend_mode"]==cond_name]["unique_condition_id"].unique())
        
    def get_cv_indices(self, cv_group_id):
        return sorted(self.agg_df[self.agg_df["cv_group_id"]==cv_group_id]["unique_condition_id"].unique())
    
    def get_condition_cv_indices(self, cond_name, cv_group_id):
        return sorted(self.agg_df[(self.agg_df["blend_mode"]==cond_name) & (self.agg_df["cv_group_id"]==cv_group_id)]["unique_condition_id"].unique())
    


    def set_loader(self, condition_list, batch_size):
        
        all_indices = []
        for cond in condition_list:
            all_indices += self.get_condition_indices(cond)
        
        all_sampler = SubsetRandomSampler(all_indices)
        all_loader = DataLoader(self, batch_size=batch_size, sampler=all_sampler,collate_fn=self.my_collate_fn)
        print("len all loader", len(all_loader))

        n_fold = self.N_fold #5

        train_loader_list = []
        valid_loader_list = []

        for i in range(n_fold):

            valid_indices = []
            train_indices = []
            for cond in condition_list:
                for j in range(n_fold):
                    if j==i:
                        valid_indices += self.get_condition_cv_indices(cond,j)
                    else:
                        train_indices += self.get_condition_cv_indices(cond,j)
                    

            train_sampler = SubsetRandomSampler(train_indices)
            train_loader = DataLoader(self, batch_size=batch_size, sampler=train_sampler,collate_fn=self.my_collate_fn)
            valid_sampler = SubsetRandomSampler(valid_indices)
            valid_loader = DataLoader(self, batch_size=batch_size, sampler=valid_sampler,collate_fn=self.my_collate_fn)
            

            print("len valid loader",len(valid_loader))
            print("len train loader", len(train_loader))
            
            train_loader_list.append(train_loader)
            valid_loader_list.append(valid_loader)
        
        return all_loader,train_loader_list,valid_loader_list
            
    #@profile
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # unique_indexがidxの行を取り出す
        trial = self.agg_df.loc[self.agg_df['unique_condition_id']==idx]
        
        test_fg = cv2.imread(self.path_to_img+trial.test_fg.values[0])
        test_bg = cv2.imread(self.path_to_img+trial.test_bg.values[0])

        test_fg = np.float32(test_fg/255.0)
        test_bg = np.float32(test_bg/255.0)

        if trial.blend_mode.values[0] == 'CLAHE':
            clahe = cv2.createCLAHE(clipLimit=trial.blend_param.values[0], tileGridSize=(8,8))
            img_yuv = cv2.cvtColor(test_fg, cv2.COLOR_BGR2YUV)
            img_yuv[:,:,0] = np.float32(clahe.apply(np.uint8(255*img_yuv[:,:,0]))/255.0)
            fg_mod = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR).clip(0,1)

            if self.use_clahe_for_ref:
                test_fg = fg_mod#.copy()

            alpha = 0.5
        else:
            fg_mod = test_fg.copy()

            alpha = trial.blend_param.values[0]
        
        # fg_modを表示
        # plt.imshow(fg_mod, vmin=0, vmax=1)
        # plt.show()

        fg_mod = torch.as_tensor(np.float32(fg_mod.transpose([2,0,1]))).unsqueeze(0)

        ref_maskimg = torch.as_tensor(np.float32(self.mask_large)).unsqueeze(0).unsqueeze(0)
        test_fg = torch.as_tensor(np.float32(test_fg.transpose([2,0,1]))).unsqueeze(0)
        test_bg = torch.as_tensor(np.float32(test_bg.transpose([2,0,1]))).unsqueeze(0)
        
        response = torch.as_tensor(trial.response.values[0], dtype=torch.float32).unsqueeze(0)
        alpha = torch.as_tensor(alpha, dtype=torch.float32).unsqueeze(0)

        unique_index = trial.unique_condition_id.values[0]
        combination_id = trial.combination_id.values[0]

        blend_mode = trial.blend_mode.values[0]

        return unique_index, combination_id, test_bg, test_fg, fg_mod, ref_maskimg, response, blend_mode, alpha

    def my_collate_fn(self, batch):

        # batchの中の各要素を取り出す
        unique_index, combination_id, test_bg, test_fg, fg_mod, ref_maskimg, response, blend_mode, alpha = zip(*batch)

        # featureのbatch数xfeature数のtensorを作成
        test_bg = torch.cat(test_bg, dim=0)
        test_fg = torch.cat(test_fg, dim=0)
        fg_mod = torch.cat(fg_mod, dim=0)
        ref_maskimg = torch.cat(ref_maskimg, dim=0)
        response = torch.cat(response, dim=0)
        alpha = torch.cat(alpha, dim=0)

        # # targetのbatch数x1のtensorを作成
        # target = torch.stack(target, dim=0)
        # # targetのbatch数x1のtensorを作成
        # # idx = torch.stack(idx, dim=0)

        # # weight = torch.stack(weight, dim=0)
        # # idxのbatch数x1のtensorを作成
        # idx = torch.as_tensor(idx)
        # # weightのbatch数x1のtensorを作成
        # weight = torch.as_tensor(weight)

        return unique_index, combination_id, test_bg, test_fg, fg_mod, ref_maskimg, response, blend_mode, alpha


def rename_single(cond, img):
    if cond == 'same' or cond == 'different' or cond =='rating':
        # '_'で分割した後，最後の要素を除く全ての要素を結合する
        return "_".join(img.split("_")[:-1])
    elif cond == 'same2' or cond == 'different2':
        # '.'で分割した後，最後の要素を除く全ての要素を結合する
        return ".".join(img.split(".")[:-1])


def rename(cond, img_list):
    new_list = []
    for img in img_list:

        new_list.append(rename_single(cond, img))

    return new_list


def filterout_training_stimuli(vis_dataset, rating_dataset, test_dataset, test_condition_list, training_condition_list):

    vis_df = vis_dataset.agg_df
    
    if rating_dataset is not None:
        rating_df = rating_dataset.agg_df

    # test_condition_list = ['different2']
    # training_condition_list = ['same', 'same2', 'different'] #['same', 'same2', 'different']



    ref_fg_list = []
    test_fg_list = []
    test_bg_list = []

    for cond in training_condition_list:
        ref_fg_list += rename(cond, list(vis_df[vis_df.condition==cond]['reference foreground'].unique()))
        test_fg_list += rename(cond, list(vis_df[vis_df.condition==cond]['test foreground'].unique()))
        test_bg_list += rename(cond, list(vis_df[vis_df.condition==cond]['test background'].unique()))

    ref_fg_list = list(set(ref_fg_list))
    test_fg_list = list(set(test_fg_list))
    test_bg_list = list(set(test_bg_list))

    train_img_list = list(set(ref_fg_list + test_fg_list + test_bg_list))

    train_fg_list = list(set(ref_fg_list + test_fg_list))
    train_bg_list = list(set(test_bg_list))

    if rating_dataset is not None:
        test_fg_list_r = rename('rating', list(set(list(rating_df.test_fg.unique()))))
        test_bg_list_r = rename('rating', list(set(list(rating_df.test_bg.unique()))))

        train_img_list += list(set(test_fg_list_r + test_bg_list_r))
        train_img_list = list(set(train_img_list))

        train_fg_list += test_fg_list_r
        train_fg_list = list(set(train_fg_list))
        train_bg_list += test_bg_list_r
        train_bg_list = list(set(train_bg_list))

        test_fg_list = list(set(test_fg_list + test_fg_list_r))
        test_bg_list = list(set(test_bg_list + test_bg_list_r))
    

    filtered_idx_dict = {}

    for cond in test_condition_list:

        unused_unique_id_list = [] 

        test_df = test_dataset.agg_df[test_dataset.agg_df.condition==cond]

        for idx, trial in test_df.iterrows():
            if False:
                if rename_single(cond, trial['test foreground']) in train_img_list:
                    continue
                elif rename_single(cond, trial['test background']) in train_img_list:
                    continue
                elif rename_single(cond, trial['reference foreground']) in train_img_list:
                    continue
                else:
                    unused_unique_id_list.append(trial['unique_index'])
            elif False:
                if rename_single(cond, trial['test foreground']) in train_fg_list:
                    continue
                elif rename_single(cond, trial['test background']) in train_bg_list:
                    continue
                elif rename_single(cond, trial['reference foreground']) in train_fg_list:
                    continue
                else:
                    unused_unique_id_list.append(trial['unique_index'])
            else:
                if rename_single(cond, trial['reference foreground']) in ref_fg_list:
                    continue
                elif rename_single(cond, trial['test foreground']) in test_fg_list:
                    if rename_single(cond, trial['test background']) in test_bg_list:
                        continue
                
                unused_unique_id_list.append(trial['new_index'])
                        

            
        print("test condition:{}".format(cond))
        print("unused_unique_id_list:{}".format(len(unused_unique_id_list)))

        filtered_idx_dict[cond] = sorted(unused_unique_id_list)
    
    return filtered_idx_dict
