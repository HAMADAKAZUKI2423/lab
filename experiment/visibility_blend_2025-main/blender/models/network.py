from __future__ import annotations
import torch
from torch import nn
import torchvision.transforms as T

from utils import save_img_torch, stimulus
from .IBlender import IBlender
from networks.models.INetwork import INetwork

class networkBlender(IBlender):
    def __init__(self, type: str, 
                 model: INetwork, 
                 resize: float = 1.0,
                 input_size: tuple[int, int] | None = None,
                 blur_mode: int = False,
                 target_type: str = "content"):
        super().__init__()
        self.type = type
        self.model = model
        self.resize = resize
        self.input_size = input_size

        self.mean_alpha: torch.float | None = None

        self.blur_mode = blur_mode

        self.target_type = target_type

    def blend(self, stim: stimulus):
        assert None not in [stim.bg, stim.ovl, stim.mask, stim.vismap]
        

        if self.blur_mode:

            # blur_modeでは、入力画像のresizeはnetwork内で行う

            if self.model.tv_input == "map":
                self.filtered_bg: torch.Tensor = self.model((stim.ovl-0.5)/0.5, (stim.bg-0.5)/0.5, stim.vismap, self.resize)
            else:
                self.filtered_bg: torch.Tensor = self.model((stim.ovl-0.5)/0.5, (stim.bg-0.5)/0.5, stim.vismap[:,:,0,0], self.resize)
            self.filtered_bg = self.filtered_bg * 0.5 + 0.5
            self.alphamap = self.model.base_alpha * stim.mask
            self.blendimg = stim.ovl * self.alphamap + self.filtered_bg * (1.0-self.alphamap)

        else:
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
            if self.model.tv_input == "map":
                calc_tv = T.functional.resize(img=stim.vismap, size=(calc_height, calc_width),antialias = True)
            else:
                calc_tv = stim.vismap
            

            if self.model.tv_input == "map":
                if self.target_type == "background":
                    self.alphamap: torch.Tensor = self.model((calc_bg-0.5)/0.5, (calc_ovl-0.5)/0.5, calc_tv)
                else:
                    self.alphamap: torch.Tensor = self.model((calc_ovl-0.5)/0.5, (calc_bg-0.5)/0.5, calc_tv)
            else:
                if self.target_type == "background":
                    self.alphamap: torch.Tensor = self.model((calc_bg-0.5)/0.5, (calc_ovl-0.5)/0.5, calc_tv[:,:,0,0])
                else:
                    self.alphamap: torch.Tensor = self.model((calc_ovl-0.5)/0.5, (calc_bg-0.5)/0.5, calc_tv[:,:,0,0])
            self.alphamap = T.functional.resize(img=self.alphamap, size=(data_height, data_width),antialias = True)
            
            if self.target_type == "background":
                self.alphamap = 1.0-self.alphamap
            self.alphamap = self.alphamap.expand(-1,3,-1,-1)
            self.alphamap = self.alphamap * stim.mask
            self.blendimg = stim.ovl * self.alphamap + stim.bg * (1.0-self.alphamap)

            self.mean_alpha = torch.sum(self.alphamap) / (torch.sum(stim.mask)*3)
    
    def save_imgs(self, save_path: str):
        

        if self.blur_mode:
            data_list = [self.filtered_bg, self.blendimg]
            path_list = [save_path + name for name in ["filtered_bg.png","blend.png"]]
        else:
            if self.mean_alpha:
                with open(save_path + 'mean_alpha.text','w') as f:
                    f.write(str(self.mean_alpha))
            data_list = [self.alphamap, self.blendimg]
            path_list = [save_path + name for name in ["alphamap.png","blend.png"]]
        for (data, path) in zip(data_list, path_list):
            save_img_torch(path, data)