from __future__ import annotations
import torch
from torch.nn import functional as F

from vismodel.fukiage2014 import Fukiage2014, local_optimization_lab
from .IBlender import IBlender
from utils import inv_sigmoid, save_img_torch, stimulus

eps = 1e-8
param_fukiage2014 = [0.81293315, 0.89033534, 0.95519662]
DEFAULT_BLUR = 65

class fukiage2014Blender(IBlender):
    def __init__(self, model: Fukiage2014, blursize: int = DEFAULT_BLUR, target_type: str = "content"):
        super().__init__()
        self.model: Fukiage2014 = model
        self.blursize: int = blursize
        self.set_target_type(target_type)
    
    def set_target_type(self, target_type: str):
        assert target_type in ["content", "background"]
        self.target_type = target_type

    def blend(self, stim: stimulus):
        assert None not in [stim.bg, stim.ovl, stim.mask, stim.vismap]

        self.model.generate_maskPyr(stim.mask)
        target_vis_level = stim.vismap[:,0,0,0]#torch.ones((bgimg.shape[0])).to(cuda)*lev
        target_vis_level = inv_sigmoid(target_vis_level, param_fukiage2014)

        if self.target_type == "background":
            fglab = self.model.bgr2lab(stim.bg)
            bglab = self.model.bgr2lab(stim.ovl)
        else:
            fglab = self.model.bgr2lab(stim.ovl)
            bglab = self.model.bgr2lab(stim.bg)

        #identifier = 'batch'+str(bc)+'_level'+str(target_vis_level)+'_blursize'+str(blursize)
        #opt_alpha_mask = local_optimization(model, fgimg, bgimg, target_vis_level)
        opt_alpha_mask = local_optimization_lab(self.model, fglab, bglab, target_vis_level)
        
        blurkernel = torch.ones((1,1,self.blursize,self.blursize),dtype=torch.float32,device=self.model.device)#/(blursize*blursize)
        pad_num = (blurkernel.shape[-1]-1)//2
        blur_opt_alpha_mask = F.conv2d(F.pad(opt_alpha_mask * self.model.dilated_mask_gp[0], (pad_num,pad_num,pad_num,pad_num), mode='reflect'), blurkernel, stride=1, padding=0, groups=opt_alpha_mask.shape[1])
        div_val = F.conv2d(F.pad(self.model.dilated_mask_gp[0], (pad_num,pad_num,pad_num,pad_num), mode='reflect'), blurkernel, stride=1, padding=0, groups=opt_alpha_mask.shape[1])
        blur_opt_alpha_mask = blur_opt_alpha_mask / (div_val+eps)

        blur_opt_alpha_mask = blur_opt_alpha_mask.expand(-1,3,-1,-1) * self.model.mask_gp[0]
        opt_alpha_mask = blur_opt_alpha_mask#opt_alpha_mask.expand(-1,3,-1,-1)# * model.mask_gp[0]
        #opt_alpha_mask = alpha_map * model.dilated_mask_gp[0]
        
        # blendimg = (1.0-blur_opt_alpha_mask) * bgimg + blur_opt_alpha_mask * fgimg
        self.blendimg = (1.0-blur_opt_alpha_mask) * bglab + blur_opt_alpha_mask * fglab
        self.blendimg = self.model.lab2bgr(self.blendimg)
        self.blendimg = self.blendimg.clamp(min = 0.0, max=1.0)

        self.alphamap = (1 - blur_opt_alpha_mask) * stim.mask
    
    def save_imgs(self, save_path: str):
        data_list = [self.alphamap, self.blendimg]
        path_list = [save_path + name for name in ["alphamap.png","blend.png"]]
        for (data, path) in zip(data_list, path_list):
            save_img_torch(path, data)