import argparse
import os
import os.path
import random
import re
import time

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as trans_func
import torchvision.utils as vutils
from matplotlib import pyplot as plt
#import matplotlib.pyplot as plt
from matplotlib.path import Path
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

#from alpha_blend import compute_loss_multi, compute_loss, calc_band_simga,compute_corr_resp_loss, compute_weighted_resp_loss, get_custom_gaussian_kernel, calc_img_sigma, local_optimization, local_optimization_lab, global_optimization
#from anlp2 import NLP_Z

matplotlib.use('agg')
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
from skimage import color as skolor  # see the docs at scikit-image.org/

class AlignedDataset(Dataset):
    IMG_EXTENSIONS = ['.png', 'jpg', 'tif']

    def __init__(self, config):
        self.config = config
        
        #dir = os.path.join(config.dataroot, config.phase)
        dir = config.dataroot

        if self.config.coco_background:
            num_paired_imgs = 1

        else:
            num_paired_imgs = 3

        self.data_pairs = sorted(self.__make_dataset(dir,config.num_pairs,num_paired_imgs))

    @classmethod
    def is_image_file(self, fname):
        return any(fname.endswith(ext) for ext in self.IMG_EXTENSIONS)

    @classmethod
    def __make_dataset(self, dir, num_examples,num_paired_imgs):
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        random.seed(1)

        

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if fname[0] != '.': 
                    if self.is_image_file(fname):
                        path = os.path.join(root, fname)
                        images.append(path)

        def pair_generator(imlist): 
            """Return an iterator of random pairs from a list of numbers.""" 
            # Keep track of already generated pairs 
            used_pairs = set() 
            
            while True: 
                pair = random.sample(imlist, num_paired_imgs) 
                # Avoid generating both (1, 2) and (2, 1) 
                pair = tuple(sorted(pair)) 
                if pair not in used_pairs: 
                    used_pairs.add(pair) 
                    yield pair 

        gen = pair_generator(images) 
        
        # Get pairs: 
        pair_list = []
        for i in range(num_examples): 
            pair = gen.__next__() 
            pair_list.append(pair)
            #print(pair) 
        
        return pair_list


        # #texture
        # path_to_imgs = path_to_dataset + 'texture/dtd/images/'
        # attribute_list = [p for p in pathlib.Path(path_to_imgs).iterdir() if p.is_dir()]
        # imgext = 'jpg'

        # imglist['texture'] = []
        # for i in range(num_fold):
        #     imglist['texture'].append([])

        # num_attrib_per_fold = len(attribute_list)//num_fold

        # min_attrib_num = 10000
        # max_attrib_num = 0
        # for i,attrib in enumerate(attribute_list):
        #     cur_fold = i//num_attrib_per_fold
        #     if cur_fold > 4:
        #         cur_fold = 4
        #     tmplist = list(pathlib.Path(path_to_imgs + attrib.name).glob('[!._]*.'+imgext))
        #     imglist['texture'][cur_fold] += tmplist
        #     min_attrib_num = min(min_attrib_num,len(tmplist))
        #     max_attrib_num = max(max_attrib_num,len(tmplist))
        #     print(attrib.name, len(tmplist))

    def __transform(self, param):
        list = []

        list.append(transforms.Lambda(lambda img: img.resize([int(img.height*param['scale']), int(img.width*param['scale'])], Image.BICUBIC)))
        
        crop_size = self.config.crop_size
        
        list.append(transforms.RandomCrop((crop_size,crop_size), padding=None, pad_if_needed=True, fill=0, padding_mode='reflect'))

        if param['h_flip']:
            list.append(transforms.Lambda(lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)))
        if param['v_flip']:
            list.append(transforms.Lambda(lambda img: img.transpose(Image.FLIP_TOP_BOTTOM)))
        
        list.append(transforms.ToTensor())

        #list.append(transforms.Lambda(lambda img: img[[2,1,0]]))

        
        #list.append(transforms.Lambda(lambda self, img, crop_size: self.random_crop(img,crop_size)))

        list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        return transforms.Compose(list)

    def __transform_param(self):
        x_max = self.config.load_size - self.config.crop_size
        x = random.randint(0, np.maximum(0, x_max))
        y = random.randint(0, np.maximum(0, x_max))

        h_flip = random.random() > 0.5
        v_flip = random.random() > 0.5

        return {'crop_pos': (x, y), 'h_flip': h_flip, 'v_flip': v_flip, 'scale': random.uniform(0.5, 2.0)}
    
    # def random_crop(self,img,crop_size):
    #     width = img.shape[-1]
    #     height = img.shape[-2]
    #     # pad the width if needed
    #     if width < crop_size[1]:
    #         padding = [crop_size[1] - width, 0,0,0]
    #         img = F.pad(img, padding, mode='reflect')
    #     # pad the height if needed
    #     if height < crop_size[0]:
    #         padding = [0,0,crop_size[0] - height,0]
    #         img = F.pad(img, padding, mode='reflect')

    #     #i, j, h, w = self.get_params(img, self.size)
    #     w = img.shape[-1]
    #     h = img.shape[-2]
    #     tw = crop_size[1]
    #     th = crop_size[0]

    #     # if h + 1 < th or w + 1 < tw:
    #     #     raise ValueError(
    #     #         "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
    #     #     )

    #     if w == tw and h == th:
    #         i=0
    #         j=0
    #     else:
    #         i = torch.randint(0, h - th + 1, size=(1, )).item()
    #         j = torch.randint(0, w - tw + 1, size=(1, )).item()
    #     #return i, j, th, tw

    #     return img[..., i:i + th, j:j + tw]#F.crop(img, i, j, h, w)

    def __getitem__(self, index):
        pair_path = self.data_pairs[index]
        imglist = []
        for fpath in pair_path:
            img = Image.open(fpath).convert('RGB')

            param = self.__transform_param()
            # w, h = AB.size
            # w2 = int(w / 2)

            transform = self.__transform(param)
            transImg = transform(img)
            
            #transImg = self.random_crop(transImg.unsqueeze(0),(self.config.crop_size,self.config.crop_size)).squeeze(0)

            imglist.append(transImg)
            # B = transform(AB.crop((w2, 0, w, h)))
        
        #tvis = torch.rand(1)[0] * 9.0

        alphamask = torch.ones_like(imglist[0][0,:,:].unsqueeze(0))

        show_imgs = False
        
        #tensor_transform = transforms.ToTensor()
        #前景画像を２つ組み合わせるパターン
            #前景画像マスクをランダム生成
            #前景画像マスクをランダムにぼかす
            #前景画像マスクをつかって２つの前景画像をブレンド

            #前景画像マスクをランダムにdilateしてtarget vis maskを生成
            #target vis maskをランダムにぼかす

        #１つの前景画像を切り抜くパターン(未実装)
            #target vis maskをランダム生成
            #target vis maskをランダムにぼかす
                
        n_edges = np.random.randint(0,8) # Number of possibly sharp edges
        r = np.random.random() # magnitude of the perturbation from the unit circle, 
        # should be between 0 and 1
        N = n_edges*3+1 # number of points in the Path
        # There is the initial point and 3 points per cubic bezier curve. Thus, the curve will only pass though n points, which will be the sharp edges, the other 2 modify the shape of the bezier curve

        angles = np.linspace(0,2*np.pi,N)
        codes = np.full(N,Path.CURVE4)
        codes[0] = Path.MOVETO

        verts = np.stack((np.cos(angles),np.sin(angles))).T*(2*r*np.random.random(N)+1-r)[:,None]
        verts[-1,:] = verts[0,:] # Using this instad of Path.CLOSEPOLY avoids an innecessary straight line
        path = Path(verts, codes)


        sigma = np.exp(np.random.random()*2)-1 #np.random.random()*5 # smoothing parameter
        # ...
        #path = Path(verts, codes)
        dpi=100.0
        fig = plt.figure(figsize=(self.config.crop_size/dpi, self.config.crop_size/dpi))
        ax = fig.add_axes([0,0,1,1]) # create the subplot filling the whole figure
        patch = patches.PathPatch(path, facecolor='k', lw=2) # Fill the shape in black
        ax.add_patch(patch)
        ax.set_xlim(np.min(verts)*1.1, np.max(verts)*1.1)
        ax.set_ylim(np.min(verts)*1.1, np.max(verts)*1.1)
        # ...
        ax.axis('off')

        fig.canvas.draw()

        ##### Smoothing ####
        # get the image as an array of values between 0 and 1
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        gray_image = np.float32(skolor.rgb2gray(data))
        gimg = Image.fromarray(gray_image,mode='F')
        scale_param = random.uniform(0.5, 2.0)
        gimg = trans_func.resize(gimg,[int(gimg.height*scale_param), int(gimg.width*scale_param)], Image.BICUBIC)
        tmp_crop = transforms.RandomCrop((self.config.crop_size,self.config.crop_size), padding=None, pad_if_needed=True, fill=1, padding_mode='constant')
        gimg = tmp_crop(gimg)
        gray_image = np.array(gimg)
        if show_imgs:
            cv2.imshow('sharp', gray_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # filter the image
        smoothed_image = np.float32(gaussian_filter(gray_image,sigma))
        if show_imgs:
            cv2.imshow('smoothed', smoothed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        bg_mask = 1-trans_func.to_tensor(smoothed_image)

        new_bg = bg_mask * imglist[0] + (1-bg_mask) * imglist[1]
        if show_imgs:
            new_bg_np = new_bg.cpu().detach().numpy().transpose([1,2,0])#np.asarray(trans_func.to_pil_image(new_bg))
            cv2.imshow('blend', new_bg_np)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        #generate vis map
        sigma_vis = np.exp(np.random.random()*3)-1 # smoothing parameter
        vis_map = 1.0-np.float32(gaussian_filter(gray_image,sigma_vis))

        center_vis = np.random.rand()
        surround_vis = np.random.rand()
        vis_map = vis_map*(center_vis-surround_vis) + surround_vis
        if show_imgs:
            cv2.imshow('vis_map', vis_map)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        tvismap = trans_func.to_tensor(vis_map)

        #randomly dilate vismap
        dilate_size = 2*np.random.randint(0,4)+1
        tvismap = F.max_pool2d(tvismap,dilate_size,stride=1,padding=(dilate_size-1)//2)
        #self.dilate = nn.MaxPool2d(3, stride=1, padding=1)

        plt.clf()
        plt.close()
        

        ################################

        im_gray = 0.299*imglist[2][0]+0.587*imglist[2][1]+0.114*imglist[2][2]
        if random.random() < 0.5:
            im_gray = im_gray * -1
        
        alpha_scale = random.uniform(0.5 , 2)
        im_gray = im_gray * alpha_scale

        im_gray = im_gray * 0.5 + 0.5

        alpha_center = random.random()
        im_gray = torch.clip(im_gray + (0.5 - alpha_center),0,1)
    
        new_fg = imglist[2] * im_gray + new_bg * ( 1 - im_gray)
        fg_color = imglist[2]
        fg_alpha = im_gray

        return {'fg': new_fg, 'bg': new_bg, 'alpha_mask': alphamask, 'target_vis': tvismap,
                'fg_color':fg_color, 'fg_alpha':fg_alpha, 'fg_paths': pair_path[0], 'bg_paths': pair_path[1]}

    def __len__(self):
        return len(self.data_pairs)