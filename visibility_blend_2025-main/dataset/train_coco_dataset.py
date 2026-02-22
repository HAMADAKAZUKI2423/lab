from pycocotools.coco import COCO
import numpy as np
import random
import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms, io
import os
import os.path
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
#import matplotlib.pyplot as plt

from .utils_dataset import RandomResizedCropCustom

#from alpha_blend import compute_loss_multi, compute_loss, calc_band_simga,compute_corr_resp_loss, compute_weighted_resp_loss, get_custom_gaussian_kernel, calc_img_sigma, local_optimization, local_optimization_lab, global_optimization
#from anlp2 import NLP_Z

#matplotlib.use('agg')

COCO_ROOT = f'{os.path.dirname(__file__)}/dataset/coco/val2017'
COCO_ANN_ROOT = f'{os.path.dirname(__file__)}/dataset/coco/annotations_trainval2017/annotations/instances_val2017.json'
COCO_CAT = ["person", "vehicle","animal","food"]

NUM_PAIRS = 8192
DATA_RATE = [16,4,5] # [train, val, test]

OUTPUT_BGR = True

class CocoDataset(Dataset): 
    def __init__(self,
                 target_type:str,
                 device:torch.device,
                 wo_semi:bool = False,
                 half_semi:str = "uniform",
                 all_semi:bool = False,
                 tv_type:str = "map",
                 uniform_map:bool = False,
                 crop_size:int = 256,
                 categories = ["person", "vehicle","animal","food"],
                 num_pairs = None,
                 apply_blur = False,
                 apply_grayscale = False,
                 apply_colorjitter = False
                 ):
        assert target_type in ["content","background"]
        assert tv_type in ["map","scalar"]
        self.target_type = target_type

        if num_pairs is None:
            self.num_pairs = NUM_PAIRS
        else:
            self.num_pairs = num_pairs
        
        self.categories = categories

        self.train_img_ids, self.val_img_ids, self.test_img_ids = self.__split_dataset(COCO_ROOT, COCO_ANN_ROOT, categories)
        self.coco_img_data = self.coco_annotations.loadImgs(self.train_img_ids + self.val_img_ids + self.test_img_ids)
        self.coco_files = [str(COCO_ROOT + '/' + img["file_name"]) for img in self.coco_img_data]
        # 画像IDからインデックスへのマッピングを作成
        self.coco_id_to_index = {img['id']: idx for idx, img in enumerate(self.coco_img_data)}
        self.data_pairs = sorted(self.__make_pairs(self.train_img_ids, self.val_img_ids, self.test_img_ids, self.num_pairs))
        self.wo_semi= wo_semi
        self.half_semi = half_semi
        self.all_semi = all_semi
        self.device = device
        self.tv_type = tv_type
        self.uniform_map = uniform_map
        self.crop_size = crop_size

        self.apply_blur = apply_blur
        self.apply_grayscale = apply_grayscale
        self.apply_colorjitter = apply_colorjitter

        # 元画像のスケールが保たれるようにcropした状態が1に対応するようなscaleパラメータ
        self.use_original_randomcrop = True
        if self.use_original_randomcrop:
            self.RandomResizedCrop = RandomResizedCropCustom(size=(self.crop_size,self.crop_size), scale=(0.25, 4.0))

        

        # self.data_pairs = sorted(self.__make_dataset(COCO_ROOT, COCO_ANN_ROOT, categories, NUM_PAIRS))
        # self.wo_semi= wo_semi
        # self.half_semi = half_semi
        # self.device = device
        # self.tv_type = tv_type
        # self.uniform_map = uniform_map
        # self.crop_size = crop_size
    
    @classmethod
    def __split_dataset(self, cocoDir, cocoAnn, coco_category):
        assert os.path.isdir(cocoDir), '%s is not a valid directory' % cocoDir

        coco_annotations = COCO(cocoAnn)
        self.coco_annotations = coco_annotations

        if len(coco_category) > 0:
            cat_ids = coco_annotations.getCatIds(supNms=coco_category)
        else:
            # use all categories
            cat_ids = coco_annotations.getCatIds()
        self.coco_cat_ids = cat_ids
        valid_img_ids = []
        for cat in cat_ids:
            valid_img_ids.extend(coco_annotations.getImgIds(catIds=cat))

        valid_img_ids = list(set(valid_img_ids))
        random.seed(1)
        random.shuffle(valid_img_ids)

        total_images = len(valid_img_ids)
        train_end = int(total_images * (DATA_RATE[0] / sum(DATA_RATE)))
        val_end = train_end + int(total_images * (DATA_RATE[1] / sum(DATA_RATE)))

        train_img_ids = valid_img_ids[:train_end]
        val_img_ids = valid_img_ids[train_end:val_end]
        test_img_ids = valid_img_ids[val_end:]

        print(f"Number of coco images: {total_images}")
        print(f"Train/Val/Test split: {len(train_img_ids)}/{len(val_img_ids)}/{len(test_img_ids)}")

        return train_img_ids, val_img_ids, test_img_ids


    def __make_pairs(self, train_img_ids, val_img_ids, test_img_ids, num_examples):
        def pair_generator(imlist):
            """Return an iterator of random pairs from a list of numbers."""
            # Keep track of already generated pairs
            used_pairs = set()

            while True:
                pair = (random.choice(imlist), random.choice(imlist))
                # Avoid generating both (1, 2) and (2, 1)
                pair = tuple(pair)
                if pair not in used_pairs:
                    used_pairs.add(pair)
                    yield pair

        train_gen = pair_generator(train_img_ids)
        val_gen = pair_generator(val_img_ids)
        test_gen = pair_generator(test_img_ids)

        # Get pairs:
        pair_list = []
        for _ in range(num_examples):
            if len(pair_list) < num_examples * (DATA_RATE[0] / sum(DATA_RATE)):
                pair_list.append(next(train_gen))
            elif len(pair_list) < num_examples * ((DATA_RATE[0] + DATA_RATE[1]) / sum(DATA_RATE)):
                pair_list.append(next(val_gen))
            else:
                pair_list.append(next(test_gen))

        return pair_list
    
    # @classmethod
    # def __make_dataset(self, cocoDir, cocoAnn, coco_category, num_examples):
    #     assert os.path.isdir(cocoDir), '%s is not a valid directory' % cocoDir

    #     coco_annotations = COCO(cocoAnn)
    #     self.coco_annotations = coco_annotations

    #     if len(coco_category)>0:
    #         cat_ids = coco_annotations.getCatIds(supNms=coco_category)
    #     else:
    #         # use all categories
    #         cat_ids = coco_annotations.getCatIds()
    #     self.coco_cat_ids = cat_ids
    #     valid_img_ids = []
    #     for cat in cat_ids:
    #         valid_img_ids.extend(coco_annotations.getImgIds(catIds=cat))

    #     valid_img_ids = list(set(valid_img_ids))

    #     coco_img_data = coco_annotations.loadImgs(valid_img_ids)
    #     self.coco_img_data = coco_img_data
    #     self.coco_files = [str(cocoDir +'/' +img["file_name"]) for img in coco_img_data]
    #     print(f"Number of coco images: {len(valid_img_ids)}")

    #     random.seed(1)

    #     def pair_generator(imlist1, imlist2):
    #         """Return an iterator of random pairs from a list of numbers.""" 
    #         # Keep track of already generated pairs 
    #         used_pairs = set() 
            
    #         while True: 
    #             pair = (random.choice(imlist1), random.choice(imlist2)) 
    #             # Avoid generating both (1, 2) and (2, 1) 
    #             pair = tuple(pair) 
    #             if pair not in used_pairs: 
    #                 used_pairs.add(pair) 
    #                 yield pair 

    #     gen = pair_generator(range(len(valid_img_ids)), range(len(valid_img_ids))) 
        
    #     # Get pairs: 
    #     pair_list = []
    #     for i in range(num_examples): 
    #         pair = gen.__next__() 
    #         pair_list.append(pair)
    #         #print(pair) 
        
    #     return pair_list

    def __transform(self, param: dict, imgData:bool = False):
        list = []

        if self.use_original_randomcrop:
            list.append(transforms.Lambda(lambda img: self.RandomResizedCrop(img, param['crop_param'])))
        else:
            list.append(transforms.Lambda(lambda img: transforms.functional.resized_crop(img,*param['crop_param'],size = (self.crop_size,self.crop_size), antialias=True)))
        
        if param['h_flip']:
            list.append(transforms.Lambda(lambda img: transforms.functional.hflip(img)))
        if param['v_flip']:
            list.append(transforms.Lambda(lambda img: transforms.functional.vflip(img)))
        
        # ランダムにガウシアンブラーを適用（50%の確率で適用）
        if param['apply_blur']:
            list.append(transforms.GaussianBlur(kernel_size=(13, 13), sigma=(0.1, 5)))
        if param['apply_grayscale']:
            list.append(transforms.Grayscale(num_output_channels=3))
        if param['apply_colorjitter']:
            list.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5))

        if imgData:
            list.append(transforms.Lambda(lambda img: img.float() /255.))
            list.append(transforms.Lambda(lambda img: img * 2. - 1.))
        #list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        return transforms.Compose(list)

    def __transform_param(self, img: torch.Tensor):

        if self.use_original_randomcrop:
            params = self.RandomResizedCrop.get_params(img)
        else:
            params = transforms.RandomResizedCrop.get_params(img, scale=(0.01, 1.0), ratio=(0.75, 1.33))

        h_flip = random.random() > 0.5
        v_flip = random.random() > 0.5

        apply_blur = False
        apply_grayscale = False
        apply_colorjitter = False

        if self.apply_blur:
            apply_blur = random.random() < 0.2
        if self.apply_grayscale:
            apply_grayscale = random.random() < 0.05
        if self.apply_colorjitter:
            apply_colorjitter = random.random() < 0.2

        return {'crop_param': params, 'h_flip': h_flip, 'v_flip': v_flip, 'apply_blur': apply_blur, 'apply_grayscale': apply_grayscale, 'apply_colorjitter': apply_colorjitter}
    
    def __make_binary_map(self, coco_img_id:int, coco_param:dict, alpha:float) -> torch.Tensor:
        #cocoAnnotationからvisibilityを作成
        index = self.coco_id_to_index[coco_img_id]
        ann_ids = self.coco_annotations.getAnnIds(
            imgIds=self.coco_img_data[index]['id'], 
            catIds=self.coco_cat_ids, 
            iscrowd=None
        )
        anns = self.coco_annotations.loadAnns(ann_ids)
        
        # modified by TF 2024/10/3
        # if self.target_type == "background":
        #     object_vis = np.random.rand() * (1 - alpha) + alpha
        #     surround_vis = np.random.rand() * (1 - alpha) + alpha
        # else:
        #     object_vis = np.random.rand() * alpha
        #     surround_vis = np.random.rand() * alpha

        if False:
            # modified by TF 2024/10/10
            # これだと透過度が高いtargetのときはtarget visibilityが低いという相関関係をもつデータに適応してしまう。
            # 実際にはtargetの元の透過度にかかわらず、任意のtarget visibilityを受け付けるべき
            object_vis = np.random.rand() * alpha
            surround_vis = np.random.rand() * alpha
        else:
            object_vis = np.random.rand()
            surround_vis = np.random.rand()
        
        #sigma_vis = np.exp(np.random.random()*3)-1 # smoothing parameter
        #tvismap = 1.0-np.float32(gaussian_filter(tvismap,sigma_vis))
        mask = torch.tensor(np.max(np.stack([self.coco_annotations.annToMask(ann) * 1
                                                 for ann in anns]), axis=0),dtype=torch.float32,device=self.device).unsqueeze(0)
        tvismap = mask * (object_vis - surround_vis) + surround_vis
        
        coco_param['apply_blur']=False
        coco_param['apply_grayscale']=False
        coco_param['apply_colorjitter']=False

        ann_transform = self.__transform(coco_param)
        tvismap = ann_transform(tvismap)
        gaussian_blur = transforms.GaussianBlur((5,5), (0.1,2.0))
        tvismap = gaussian_blur(tvismap)
        return tvismap
    
    def __load_coco_data(self, id: int) -> [dict, torch.Tensor]:
        #cocoからロード
        index = self.coco_id_to_index[id]
        img = io.read_image(self.coco_files[index], mode = io.ImageReadMode.RGB).to(self.device)
        if OUTPUT_BGR:
            img = img[[2,1,0]]
        
        trans_param = self.__transform_param(img)
        transform_func = self.__transform(trans_param, imgData= True)
        trans_img = transform_func(img)
        return trans_param, trans_img

    def __getitem__(self, index):
        ref_img_id, tgt_img_id = self.data_pairs[index]

        tgt_param, tgt_transImg = self.__load_coco_data(tgt_img_id)
        _, ref_transImg = self.__load_coco_data(ref_img_id)

        alpha_val=1.
        
        alphamask = torch.ones_like(tgt_transImg[0,:,:].unsqueeze(0),dtype=torch.float32,device=self.device)
        if self.half_semi=="map":
            if np.random.rand() >= 0.5:
                fg_alpha = self.__make_binary_map(tgt_img_id, tgt_param, alpha_val)
            else:
                fg_alpha = torch.ones_like(tgt_transImg[0,:,:].unsqueeze(0),dtype=torch.float32,device=self.device)
        else:
            if self.wo_semi == True:
                alpha_val = 1
            elif self.all_semi:
                # for testing model with all semi-transparent data
                alpha_val = np.random.rand()
            else:
                alpha_val = np.random.rand()
                if self.half_semi=="uniform" and np.random.rand() >= 0.5:
                    alpha_val = 1
            fg_alpha = alpha_val * torch.ones_like(tgt_transImg[0,:,:].unsqueeze(0),dtype=torch.float32,device=self.device)

        if self.tv_type == "map":
            if self.uniform_map:
                tvismap = np.random.rand() * torch.ones_like(tgt_transImg[0,:,:].unsqueeze(0),dtype=torch.float32,device=self.device)
            else:
                tvismap = self.__make_binary_map(tgt_img_id, tgt_param, alpha_val)
        else:
            tvismap = np.random.rand() * torch.ones(1,dtype=torch.float32,device=self.device)

        if self.target_type == "background":
            fg_color = ref_transImg
            bg_color = tgt_transImg
        elif self.target_type == "content":
            fg_color = tgt_transImg
            bg_color = ref_transImg

        fg =  fg_color * fg_alpha + bg_color * (1 - fg_alpha)
        # fg =  fg_color * alpha_val + bg_color * (1 - alpha_val)

        return {'fg': fg, 'bg': bg_color, 'alpha_mask': alphamask, 'fg_color':fg_color, 'fg_alpha':fg_alpha,'target_vis': tvismap}

    def __len__(self):
        return len(self.data_pairs)
    
    def __str__(self):
        message = f'''
        CocoDataset:
        COCO_ROOT: {COCO_ROOT},
        COCO_ANN_ROOT: {COCO_ANN_ROOT},
        COCO_CAT: {self.categories},
        NUM_PAIRS: {self.num_pairs},
        DATA_RATE: {DATA_RATE},
        OUTPUT_BGR: {OUTPUT_BGR},
        self.wo_semi: {self.wo_semi},
        self.tv_type: {self.tv_type},
        self.uniform_map: {self.uniform_map},
        self.crop_size: {self.crop_size}'''
        return message
    
    def make_loader(self, batchsize:int):
        train_counts = int(self.num_pairs*(DATA_RATE[0]/sum(DATA_RATE)))
        val_counts = int(self.num_pairs*(DATA_RATE[1]/sum(DATA_RATE)))
        train_indices = list(range(train_counts))
        valid_indices = list(range(train_counts,train_counts + val_counts))
        test_indices = list(range(train_counts + val_counts,self.num_pairs))
        train_sampler = SubsetRandomSampler(train_indices)
        train_loader = DataLoader(self, batch_size=batchsize, sampler=train_sampler)
        valid_sampler = SubsetRandomSampler(valid_indices)
        valid_loader = DataLoader(self, batch_size=batchsize, sampler=valid_sampler)
        test_sampler = SubsetRandomSampler(test_indices)
        test_loader = DataLoader(self, batch_size=batchsize, sampler=test_sampler)

        return train_loader, valid_loader, test_loader