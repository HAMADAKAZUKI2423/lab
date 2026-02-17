from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import json
import numpy as np
import cv2
from matplotlib import pyplot as plt
from utils import stimulus, save_img_torch, save_grayimg_plt
from .IBlender import IBlender
from vismodel.supermodels.visModel import VisModel
from vismodel.vismodel_mlp import VisModel_MLP
from vismodel.loss.loader import load_lossFunction
from torch.nn.utils import clip_grad_norm_
import torchvision.transforms as T

eps = 0.1#1e-8

class visibilityBlender(IBlender):
    def __init__(self, vismodel: VisModel,
                 config: dict[str],
                 target_type: str,
                 device: torch.device,
                 save_only_img:bool = False):
        super().__init__()
        self.__device = device
        self.__vismodel = vismodel
        self.__save_only_img = save_only_img
        
        self.__lr: float  = config['lr']
        self.__num_epochs: int = np.int16(config.get('num_epochs',500))
        self.__vismodel._set_target_type(target_type)
        self.__lossF = load_lossFunction(config.get('loss_type','original'), config, device)
        self.__vissize_alpha = config.get('vissize_alpha', False)
        
        self.__alpha_initialize_by_tvismap: bool = config.get('alpha_initialize_by_tvismap',False)
        if not self.__alpha_initialize_by_tvismap:
            self.__init_alpha_val = 0.5
        
        self.__apply_blur: bool = config.get('apply_blur',False)
        self.__clip_norm_size: float = config.get('clip_norm_size',0.0)

        self.resize = config.get('resize', 1.0)
        self.input_size = config.get('input_size', None)
        

        
        self.reset_state()

    def _is_cuda_device(self) -> bool:
        if self.__device is None:
            return False
        device = torch.device(self.__device)
        return device.type == "cuda"

    def _maybe_cuda_synchronize(self) -> None:
        if self._is_cuda_device():
            torch.cuda.synchronize()

    def blend(self, stim: stimulus):
        assert None not in [stim.bg, stim.content_color, stim.content_alpha, stim.mask, stim.vismap, stim.ovl]

        data_height = stim.ovl.shape[2]
        data_width = stim.ovl.shape[3]

        if self.input_size is None:
            calc_height = int(data_height * self.resize)
            calc_width = int(data_width * self.resize)
        else:
            calc_height = self.input_size[0]
            calc_width = self.input_size[1]

        calc_ovl = T.functional.resize(img=stim.ovl, size=(calc_height, calc_width),antialias = True)
        calc_bg = T.functional.resize(img=stim.bg, size=(calc_height, calc_width),antialias = True)
        calc_content_color = T.functional.resize(img=stim.content_color, size=(calc_height, calc_width),antialias = True)
        calc_content_alpha = T.functional.resize(img=stim.content_alpha, size=(calc_height, calc_width),antialias = True)
        calc_mask = T.functional.resize(img=stim.mask, size=(calc_height, calc_width),antialias = True)
        calc_tv = T.functional.resize(img=stim.vismap, size=(calc_height, calc_width),antialias = True)
        
        self.__vismodel.set_inputs_bg_ovl_contents(calc_bg, calc_ovl, calc_content_color, calc_content_alpha, calc_mask)
        self.__lossF.reset_loss()
        self.alphamap = self.__iterative_optim(calc_tv)

        self.alphamap = T.functional.resize(img=self.alphamap, size=(data_height, data_width),antialias = True)
        self.blendimg = self.alphamap * stim.ovl + (1.0-self.alphamap) * stim.bg
        self.__opt_vismap_rawscale = self.__vismodel.vis_map
        self.__opt_vismap_norm = self.__vismodel.norm_vismap
            
        # self.__vismodel.set_inputs_bg_ovl_contents(stim.bg, stim.ovl, stim.content_color, stim.content_alpha, stim.mask)
        # self.__lossF.reset_loss()
        # self.alphamap = self.__iterative_optim(stim.vismap)
        # self.blendimg = self.alphamap * stim.ovl + (1.0-self.alphamap) * stim.bg
        # self.__opt_vismap_rawscale = self.__vismodel.vis_map
        # self.__opt_vismap_norm = self.__vismodel.norm_vismap
    
    def save_imgs(self, save_dir: str):
        if self.__save_only_img:
            self.__lossF.save_loss(save_dir)
        self.__save_img(save_dir)
    
    def reset_state(self):
        self.__vismodel.reset_state()
        self.__lossF.reset_state()
        self.__lossF.reset_loss()
    
    def __save_img(self, dir_path):
        save_img_torch(f'{dir_path}blend.png', self.blendimg)
        save_img_torch(f'{dir_path}alphamap.png', self.alphamap)
        gray_imgs = {
            "vismap":self.__opt_vismap_norm,
            "vismap_rawscale":self.__opt_vismap_rawscale
        }
        norm_imgs = ["vismap"]
        for key, img in gray_imgs.items():
            if key in norm_imgs:
                save_grayimg_plt(f'{dir_path}{key}_plt.png', torch.clip(img, 0, 1), norm = True)
                save_img_torch(f'{dir_path}{key}.png', torch.clip(img, 0, 1))
            else:
                save_grayimg_plt(f'{dir_path}{key}_plt.png', img, norm = False)
    
    def blur_image(self, image):
        kernel = torch.tensor([[1., 2., 1.],
                            [2., 4., 2.],
                            [1., 2., 1.]], device=image.device)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, 3, 3)
        blurred = F.conv2d(image, kernel, padding=1)
        return blurred

    def __iterative_optim(self, target_vis:torch.Tensor) -> torch.Tensor:
        self.__lossF.compute_loss_preprocess(target_vis, self.__vismodel)
        
        if self.__vissize_alpha:
            assert isinstance(self.__vismodel, VisModel_MLP)
            opt_alphamap = nn.init.uniform_(torch.empty_like(torch.zeros(self.__vismodel.ds_map_shape[0],1,self.__vismodel.ds_map_shape[2],self.__vismodel.ds_map_shape[3]),device=self.__device), a=-0.2, b=0.2)
        else:
            opt_alphamap = nn.init.uniform_(torch.empty_like(torch.zeros(self.__vismodel.ref.shape[0],1,self.__vismodel.ref.shape[2],self.__vismodel.ref.shape[3]),device=self.__device), a=-0.2, b=0.2)

        if self.__alpha_initialize_by_tvismap:
            alpha_base_init = (self.__vismodel.dilated_mask_gp[0] * (target_vis)).clamp(min=eps,max=1-eps)
        else:
            alpha_base_init = (torch.ones_like(opt_alphamap)*self.__init_alpha_val).clamp(min=eps,max=1-eps)#.unsqueeze(1)

        alpha_base_init =  torch.log(alpha_base_init/(1-alpha_base_init))
        opt_alphamap = opt_alphamap + alpha_base_init
        opt_alphamap.requires_grad=True

        optimgs = []
        optimgs.append(opt_alphamap)
        optimizer = optim.Adam(optimgs, lr=self.__lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

        run = [0]
        epoch_list = []
        self._maybe_cuda_synchronize()
        iteration_timer = time.time()
        while run[0]<self.__num_epochs:

            # Apply blurring to opt_alphamap
            if self.__apply_blur:
                opt_alphamap.data = self.blur_image(opt_alphamap.data)
            optimizer.zero_grad()

            # if True:
            #     # Apply blurring to opt_alphamap
            #     blurred_opt_alphamap = self.blur_image(opt_alphamap)

            #     sig_alpha = F.sigmoid(blurred_opt_alphamap.expand(-1,3,-1,-1))
            # else:
            sig_alpha = F.sigmoid(opt_alphamap.expand(-1,3,-1,-1))

            if self.__vissize_alpha:
                sig_alpha = self.__vismodel.get_upsampled_map(sig_alpha)
            sig_alpha = sig_alpha * self.__vismodel.dilated_mask_gp[0]
            self.__lossF.compute_loss(self.__vismodel, sig_alpha)
            self.__lossF.all_loss.backward(retain_graph=False)

            # 勾配クリッピングを適用
            if self.__clip_norm_size > 0:
                clip_grad_norm_(optimgs, self.__clip_norm_size)
            
            optimizer.step()

            epoch_list.append(run[0])
            run[0] += 1
            
            self._maybe_cuda_synchronize()
            elasped_time_iteration = time.time() - iteration_timer
            if elasped_time_iteration > 5:
                self._maybe_cuda_synchronize()
                iteration_timer = time.time()
                print("{:1f} percent completed".format(run[0]/self.__num_epochs*100))
                print('Loss : {:4f}'.format(self.__lossF.all_loss.item()))
            # scheduler.step()
            
        with torch.no_grad():
            sig_alpha = F.sigmoid(opt_alphamap.expand(-1,3,-1,-1))
            if self.__vissize_alpha:
                sig_alpha = self.__vismodel.get_upsampled_map(sig_alpha)
            sig_alpha = sig_alpha * self.__vismodel.dilated_mask_gp[0]
            self.__lossF.compute_loss(self.__vismodel, sig_alpha)
            if self.__vissize_alpha:
                opt_alphamap = self.__vismodel.get_upsampled_map(opt_alphamap)
            # torch.clip(opt_alphamap,0,1)

            opt_alphamap = F.sigmoid(opt_alphamap.expand(-1,3,-1,-1))
            opt_alphamap = opt_alphamap * self.__vismodel.dilated_mask_gp[0]
            
            epoch_list.append(run[0])
        
        return opt_alphamap