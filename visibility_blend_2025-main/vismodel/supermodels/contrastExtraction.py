from __future__ import annotations
import torch
from torch.nn import functional as F

eps = 1e-8

class ContrastExtraction():
    def __init__(self,
                corr_ksize: int,
                device: torch.device):
        self.precise_contrast_extraction = False
        self.contrast_extraction_residual_fit = False
        self.contrast_extraction_correlation = False
        self.allow_enhanced = True
        # target extraction時に，targetのweightが1以上になることを許す
        self.kernel = torch.ones((3,1,corr_ksize,corr_ksize),dtype=torch.float32,device=device)/(corr_ksize*corr_ksize)

    def contrast_extraction(self,
                            y_pyr: torch.Tensor,
                            blend_pyr: torch.Tensor,
                            level: int,
                            return_weight: bool = False) -> tuple[list[torch.Tensor]]:
        # Extraction x_img component from blendimg
        # Blendimg has component of x_img and y_img
        pad_num = (self.kernel.shape[-1]-1)//2
        x_res_pyr = []
        y_fit_pyr = []
        weights_list = []
        fitting_level = level if self.contrast_extraction_residual_fit else level - 1

        for i in range(fitting_level):
            if self.precise_contrast_extraction == False:

                y_var = F.conv2d(F.pad(y_pyr[i]**2, (pad_num,pad_num,pad_num,pad_num), mode='reflect'), self.kernel, stride=1, padding=0, groups=y_pyr[i].shape[1])
                blend_var = F.conv2d(F.pad(blend_pyr[i]**2, (pad_num,pad_num,pad_num,pad_num), mode='reflect'), self.kernel, stride=1, padding=0, groups=y_pyr[i].shape[1])
                covar = F.conv2d(F.pad(y_pyr[i]*blend_pyr[i], (pad_num,pad_num,pad_num,pad_num), mode='reflect'), self.kernel, stride=1, padding=0, groups=y_pyr[i].shape[1])
            
            else:
                y_ave = F.conv2d(F.pad(y_pyr[i], (pad_num,pad_num,pad_num,pad_num), mode='reflect'), self.kernel, stride=1, padding=0, groups=y_pyr[i].shape[1])
                blend_ave = F.conv2d(F.pad(blend_pyr[i], (pad_num,pad_num,pad_num,pad_num), mode='reflect'), self.kernel, stride=1, padding=0, groups=y_pyr[i].shape[1])

                y_var = F.conv2d(F.pad((y_pyr[i] - y_ave)**2, (pad_num,pad_num,pad_num,pad_num), mode='reflect'), self.kernel, stride=1, padding=0, groups=y_pyr[i].shape[1])
                blend_var = F.conv2d(F.pad((blend_pyr[i] - blend_ave)**2, (pad_num,pad_num,pad_num,pad_num), mode='reflect'), self.kernel, stride=1, padding=0, groups=y_pyr[i].shape[1])
                covar = F.conv2d(F.pad((y_pyr[i] - y_ave)*(blend_pyr[i] - blend_ave), (pad_num,pad_num,pad_num,pad_num), mode='reflect'), self.kernel, stride=1, padding=0, groups=y_pyr[i].shape[1])
            
            if self.contrast_extraction_correlation:
                weights = covar/(torch.sqrt(y_var*blend_var)+eps)
            else:
                weights = covar/(y_var+eps)

            weights = torch.clamp(weights,min=0,max=1)
            weights_list.append(weights)

            x_res_pyr.append(blend_pyr[i] - weights*y_pyr[i])
            y_fit_pyr.append(weights*y_pyr[i])

        if not self.contrast_extraction_residual_fit:
            if self.weight_mode == "3-way-extract":
                x_res_pyr.append(blend_pyr[-1])
            else:
                x_res_pyr.append(blend_pyr[-1]-y_pyr[-1])
            y_fit_pyr.append(y_pyr[-1])
            

        if return_weight:
            return x_res_pyr, y_fit_pyr, weights_list
        else:
            return x_res_pyr, y_fit_pyr
    
    def partial_correlation_extraction(self, 
                                    x_pyr: torch.Tensor,
                                    y_pyr: torch.Tensor,
                                    blend_pyr: torch.Tensor,
                                    level: int,
                                    return_weight = False) -> tuple[list[torch.Tensor]]:
        # Extraction x_img component from blendimg
        # Blendimg has component of x_img and y_img
        pad_num = (self.kernel.shape[-1]-1)//2

        x_res_pyr = []
        y_fit_pyr = []
        weights_x_y_list = []
        weights_x_bl_list = []
        weights_y_bl_list = []

        fitting_level = level if self.contrast_extraction_residual_fit else level - 1
        
        for i in range(fitting_level):

            x_var = F.conv2d(F.pad(x_pyr[i]**2, (pad_num,pad_num,pad_num,pad_num), mode='reflect'), self.kernel, stride=1, padding=0, groups=x_pyr[i].shape[1])
            covar_x_y = F.conv2d(F.pad(x_pyr[i]*y_pyr[i], (pad_num,pad_num,pad_num,pad_num), mode='reflect'), self.kernel, stride=1, padding=0, groups=x_pyr[i].shape[1])
            covar_x_bl = F.conv2d(F.pad(x_pyr[i]*blend_pyr[i], (pad_num,pad_num,pad_num,pad_num), mode='reflect'), self.kernel, stride=1, padding=0, groups=x_pyr[i].shape[1])

            weights_x_y = torch.clamp(covar_x_y/(x_var+eps), min=0, max=1)

            if self.allow_enhanced:
                weights_x_bl = torch.clamp(covar_x_bl/(x_var+eps), min=0, max=4)
            else:
                weights_x_bl = torch.clamp(covar_x_bl/(x_var+eps), min=0, max=1)

            y_minus_x = y_pyr[i] - weights_x_y*x_pyr[i]
            bl_minus_x = blend_pyr[i] - weights_x_bl*x_pyr[i]

            y_var = F.conv2d(F.pad(y_minus_x**2, (pad_num,pad_num,pad_num,pad_num), mode='reflect'), self.kernel, stride=1, padding=0, groups=x_pyr[i].shape[1])
            covar_y_bl = F.conv2d(F.pad(y_minus_x*bl_minus_x, (pad_num,pad_num,pad_num,pad_num), mode='reflect'), self.kernel, stride=1, padding=0, groups=x_pyr[i].shape[1])

            weights_y_bl = torch.clamp(covar_y_bl/(y_var+eps), min=0, max=1)

            x_res_pyr.append(blend_pyr[i] - weights_y_bl*y_pyr[i])
            y_fit_pyr.append(weights_y_bl*y_pyr[i])

            weights_x_y_list.append(weights_x_y)
            weights_x_bl_list.append(weights_x_bl)
            weights_y_bl_list.append(weights_y_bl)


        if not self.contrast_extraction_residual_fit:
            if self.weight_mode == "3-way-extract":
                x_res_pyr.append(blend_pyr[-1])
            else:
                x_res_pyr.append(blend_pyr[-1]-y_pyr[-1])
            y_fit_pyr.append(y_pyr[-1])
        

        if return_weight:
            return x_res_pyr, y_fit_pyr, weights_x_y_list, weights_x_bl_list, weights_y_bl_list
        else:
            return x_res_pyr, y_fit_pyr

    def partial_correlation_extraction_precise(self,
                                            x_pyr: torch.Tensor,
                                            y_pyr: torch.Tensor,
                                            blend_pyr: torch.Tensor,
                                            level: int,
                                            return_weight = False) -> tuple[list[torch.Tensor]]:
        # Extraction x_img component from blendimg
        # Blendimg has component of x_img and y_img
        pad_num = (self.kernel.shape[-1]-1)//2

        x_res_pyr = []
        y_fit_pyr = []
        weights_x_y_list = []
        weights_x_bl_list = []
        weights_y_bl_list = []

        fitting_level = level if self.contrast_extraction_residual_fit else level - 1
        
        for i in range(fitting_level):
            x_mean = F.conv2d(F.pad(x_pyr[i], (pad_num,pad_num,pad_num,pad_num), mode='reflect'), self.kernel, stride=1, padding=0, groups=x_pyr[i].shape[1])
            y_mean = F.conv2d(F.pad(y_pyr[i], (pad_num,pad_num,pad_num,pad_num), mode='reflect'), self.kernel, stride=1, padding=0, groups=x_pyr[i].shape[1])
            blend_mean = F.conv2d(F.pad(blend_pyr[i], (pad_num,pad_num,pad_num,pad_num), mode='reflect'), self.kernel, stride=1, padding=0, groups=x_pyr[i].shape[1])
            x_var = F.conv2d(F.pad((x_pyr[i]-x_mean)**2, (pad_num,pad_num,pad_num,pad_num), mode='reflect'), self.kernel, stride=1, padding=0, groups=x_pyr[i].shape[1])
            covar_x_y = F.conv2d(F.pad((x_pyr[i]-x_mean)*(y_pyr[i]-y_mean), (pad_num,pad_num,pad_num,pad_num), mode='reflect'), self.kernel, stride=1, padding=0, groups=x_pyr[i].shape[1])
            covar_x_bl = F.conv2d(F.pad((x_pyr[i]-x_mean)*(blend_pyr[i]-blend_mean), (pad_num,pad_num,pad_num,pad_num), mode='reflect'), self.kernel, stride=1, padding=0, groups=x_pyr[i].shape[1])

            weights_x_y = torch.clamp(covar_x_y/(x_var+eps), min=0, max=1)

            if self.allow_enhanced:
                weights_x_bl = torch.clamp(covar_x_bl/(x_var+eps), min=0, max=4)
            else:
                weights_x_bl = torch.clamp(covar_x_bl/(x_var+eps), min=0, max=1)

            y_minus_x = y_pyr[i] - weights_x_y*x_pyr[i]
            bl_minus_x = blend_pyr[i] - weights_x_bl*x_pyr[i]

            y_minus_x_mean = F.conv2d(F.pad(y_minus_x, (pad_num,pad_num,pad_num,pad_num), mode='reflect'), self.kernel, stride=1, padding=0, groups=y_pyr[i].shape[1])
            bl_minus_x_mean = F.conv2d(F.pad(bl_minus_x, (pad_num,pad_num,pad_num,pad_num), mode='reflect'), self.kernel, stride=1, padding=0, groups=y_pyr[i].shape[1])
            y_var = F.conv2d(F.pad((y_minus_x-y_minus_x_mean)**2, (pad_num,pad_num,pad_num,pad_num), mode='reflect'), self.kernel, stride=1, padding=0, groups=x_pyr[i].shape[1])
            covar_y_bl = F.conv2d(F.pad((y_minus_x-y_minus_x_mean)*(bl_minus_x-bl_minus_x_mean), (pad_num,pad_num,pad_num,pad_num), mode='reflect'), self.kernel, stride=1, padding=0, groups=x_pyr[i].shape[1])

            weights_y_bl = torch.clamp(covar_y_bl/(y_var+eps), min=0, max=1)

            x_res_pyr.append(blend_pyr[i] - weights_y_bl*y_pyr[i])
            y_fit_pyr.append(weights_y_bl*y_pyr[i])

            weights_x_y_list.append(weights_x_y)
            weights_x_bl_list.append(weights_x_bl)
            weights_y_bl_list.append(weights_y_bl)


        if not self.contrast_extraction_residual_fit:
            if self.weight_mode == "3-way-extract":
                x_res_pyr.append(blend_pyr[-1])
            else:
                x_res_pyr.append(blend_pyr[-1]-y_pyr[-1])
            y_fit_pyr.append(y_pyr[-1])
        

        if return_weight:
            return x_res_pyr, y_fit_pyr, weights_x_y_list, weights_x_bl_list, weights_y_bl_list
        else:
            return x_res_pyr, y_fit_pyr
    