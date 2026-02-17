from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .superModel import SuperModel

class PyramidFunctionModel(SuperModel):
    def __init__(self, level: int,
                dims: int,
                device: torch.device):
        super().__init__(level , device)

        self.dims = dims
        self.filt = self.gauss_kernel(dims)
        self.pad_two = nn.ReflectionPad2d(2)
        
    def gen_Lpyr(self, image: torch.Tensor, level: int, get_gpyr=False, channels = 3) -> list[torch.Tensor]:
        J = image
        dims = image.shape[1]
        pyr = []
        gpyr=[]
        for i in range(0, level-1):
            I = F.conv2d(self.pad_two(J), self.filt[:channels], stride=2, padding=0,
                         groups=dims)
            I_up = self.upsample(I)
            pyr.append(J - I_up)
            if get_gpyr:
                gpyr.append(I_up)

            J = I
        pyr.append(J)
        if get_gpyr:
            #Gaussian pyramid upsampled by one level
            I = F.conv2d(self.pad_two(J), self.filt[:channels], stride=2, padding=0,
                         groups=dims)
            I_up = self.upsample(I)
            gpyr.append(I_up)
            return pyr, gpyr
        else:
            return pyr
    
    def gen_Gpyr_normal(self, image: torch.Tensor, level: int) -> list[torch.Tensor]:
        #Gaussian pyramid upsampled by one level
        
        J = image
        dims = image.shape[1]
        #pyr = []
        gpyr=[J]
        for i in range(level):
            I = F.conv2d(self.pad_two(J), self.filt, stride=2, padding=0,
                            groups=dims)
            #I_up = self.upsample(I)#include conv
            gpyr.append(I)

            J = I
        return gpyr
    
    def gen_Gpyr(self, image: torch.Tensor, level: int) -> list[torch.Tensor]:
        #Gaussian pyramid upsampled by one level
        
        J = image
        dims = image.shape[1]
        #pyr = []
        gpyr=[]
        for i in range(level):
            I = F.conv2d(self.pad_two(J), self.filt, stride=2, padding=0,
                         groups=dims)
            I_up = self.upsample(I)#include conv
            gpyr.append(I_up)

            J = I
        return gpyr
    
    def recon_Lpyr(self, pyr: list[torch.Tensor]) -> torch.Tensor:
        num_level = len(pyr)
        reconimg = pyr[-1]+0.0
        for i in range(num_level-1):
            reconimg = self.upsample(reconimg) + pyr[num_level-2-i]
        return reconimg
    
    def calc_lowfreq_img(self, img:torch.Tensor) -> torch.Tensor:
        img_pyr = self.gen_Lpyr(img, self.level)

        sum_img = torch.zeros_like(img_pyr[-1])
        for i,img_frac in enumerate(reversed(img_pyr)):
            if i not in [0]:
                img_frac = torch.zeros_like(img_frac)
            sum_img += img_frac
            if i != len(img_pyr)-1:
                sum_img = self.upsample(sum_img)

        return sum_img