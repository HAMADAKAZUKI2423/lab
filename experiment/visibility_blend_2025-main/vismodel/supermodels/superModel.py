from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class SuperModel(nn.Module):
    
    def __init__(self, level, device):
        super(SuperModel, self).__init__()

        self.level = level
        self.device = device

        self.dilate = nn.MaxPool2d(3, stride=1, padding=1)

        self.g_pad_two = nn.ReflectionPad2d(2)

        self.running_std=False

        self.mat_bgr2yuv = torch.tensor([[0.114,0.5,-0.0813],
                            [0.587,-0.3313,-0.4187],
                            [0.299,-0.1687,0.5]]).to(device=self.device)
        self.mat_yuv2bgr = torch.inverse(self.mat_bgr2yuv)
        
        self.mat_bgr2xyz = torch.tensor([[0.1804375,0.0721750,0.9503041],
                                [0.3575761,0.7151522,0.1191920],
                                [0.4124564,0.2126729,0.0193339]]).to(device=device)
        
        self.D65 = torch.tensor([0.95047, 1.00000, 1.08883]).to(device=device)
        self.D50 =  torch.tensor([0.96422, 1.00000, 0.82521]).to(device=device)

        self.mat_xyz2bgr = torch.inverse(self.mat_bgr2xyz)

        kernel = np.array([
            [1, 5, 8, 5, 1],
            [5, 25, 40, 25, 5],
            [8, 40, 64, 40, 8],
            [5, 25, 40, 25, 5],
            [1, 5, 8, 5, 1]], np.float32) / 400.0
        
        self.gfilt = torch.as_tensor(np.reshape(kernel,(1, 1, 5, 5))).to(device=self.device)
    
    def blending(self, fg, bg, alpha, blend_mode = 'linear'):

        # blend_modeがlistの場合は、それぞれのbatchで異なるblend_modeを適用する
        if isinstance(blend_mode, tuple) or isinstance(blend_mode, list):
            blend_list = []
            for i in range(fg.shape[0]):
                blend = self.blending(fg[i], bg[i], alpha[i], blend_mode[i])
                blend_list.append(blend)
            return torch.stack(blend_list, dim=0)
        
        else:

            if blend_mode == 'linear':
                return alpha * fg + (1-alpha) * bg
            elif blend_mode == 'multiply':
                return alpha * (fg * bg) + (1-alpha) * bg
            elif blend_mode == 'screen':
                return alpha * (1-((1-fg) * (1-bg))) + (1-alpha) * bg
        
    def gauss_kernel(self,channels: int=3):
        kernel = torch.tensor([
                [1, 5, 8, 5, 1],
                [5, 25, 40, 25, 5],
                [8, 40, 64, 40, 8],
                [5, 25, 40, 25, 5],
                [1, 5, 8, 5, 1]],dtype=torch.float32,device=self.device)
        kernel /= 400.
        kernel = kernel.repeat(channels, 1, 1, 1)
        return kernel
    
    def upsample(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)


    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out
    
    def downsample(self, x, kernel=None):
        if kernel is None:
            kernel = self.gauss_kernel(channels=x.shape[1])
        img = torch.nn.functional.pad(x, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1],stride=2)
        return out

        
    def generate_maskPyr(self, maskimg: torch.Tensor):
        
        with torch.no_grad():
        
            x=maskimg
            self.mask_gp = [maskimg]
            self.dilated_mask_gp = [self.dilate(maskimg)]
            for i in range(self.level-1):
                x = F.conv2d(self.g_pad_two(x), self.gfilt, stride=2, padding=0,groups=1)
                self.mask_gp.append(x)
                self.dilated_mask_gp.append(self.dilate(x))

            self.mask_gp_sum = []
            self.dilated_mask_gp_sum = []
            for i in range(len(self.mask_gp)):
                self.mask_gp_sum.append(self.mask_gp[i].sum(dim=(1,2,3)))
                self.dilated_mask_gp_sum.append(self.dilated_mask_gp[i].sum(dim=(1,2,3)))
            return

    def setOptimizeFlag(self, param_name_list):
        
        for name, param in self.named_parameters(): ######### changed here
            if name in param_name_list:
                param.requires_grad = True
                print(name, param.data)
            else:
                param.requires_grad = False
        return

    def grad_off(self):
        self.eval()#this does not turn off requires_grad

    def grad_on(self):
        self.train()

    def get_optimize_params(self):
        #return [param["value"] for param in self.param_dict if param["optimize"]]
        return self.parameters()
    
    def get_param(self, param_name):
        for name, param in self.named_parameters():
            if name == param_name:
                return param
        for name, param in self.__dict__.items():
            if name == param_name:
                return param

    def set_param(self, param_name, val):
        for name, param in self.named_parameters():
            if name == param_name:
                param.data = torch.tensor(val,dtype=torch.float32,device=self.device)
        
        #hyper parameters
        for name, param in self.__dict__.items():
            if name == param_name:
                param = val
                
    
    def LinearizeRgb(self, img):
        #img can be a tensor of arbitrary size
        img[img<=0.04045] /= 12.92
        img[img>0.04045] = torch.pow((img[img>0.04045]+0.055)/1.055,2.4)
        return img
    
    # def linear2sRGB(self, img):
    #     threshold2 = 0.04045/12.92
    #     img[img<=threshold2] *= 12.92
    #     img[img>threshold2] = torch.pow(img[img>threshold2],1/2.4)*1.055-0.055

    #     return img

    def bgr2xyz(self,rgb):
        #rgb is a tensor of [batch,color,y,x]
        #making collapted tensor [batch, color, x*y] then excange axis -> [batch, x*y, color] 
        t_rgb = rgb.view(-1,3,rgb.shape[2]*rgb.shape[3]).permute(0,2,1)
        t_xyz = torch.matmul(t_rgb, self.mat_bgr2xyz)
        xyz = t_xyz.permute(0,2,1).view(-1,3,rgb.shape[2],rgb.shape[3])
        return xyz

    def xyz2lab(self, xyz, weight=1.0, weight_ratio=1.0):

        #xyz is a tensor of [batch, color, y, x]
        f_xyz = xyz/self.D65.view(1,3,1,1).expand_as(xyz)
        f_xyz[f_xyz>6.0/29.0*6.0/29.0*6.0/29.0]=torch.pow(f_xyz[f_xyz>6.0/29.0*6.0/29.0*6.0/29.0], 1.0/3.0)
        f_xyz[f_xyz<=6.0/29.0*6.0/29.0*6.0/29.0]=f_xyz[f_xyz<=6.0/29.0*6.0/29.0*6.0/29.0]/3.0*29.0/6.0*29.0/6.0 + 4.0/29.0

        lab = torch.empty_like(xyz)
        lab[:,0,:,:] = 1.16*f_xyz[:,1,:,:] - 0.16
        lab[:,1,:,:] = weight * 5.0 * (f_xyz[:,0,:,:] - f_xyz[:,1,:,:])
        lab[:,2,:,:] = weight * weight_ratio * 2.0 * (f_xyz[:,1,:,:] - f_xyz[:,2,:,:])
        return lab

    def bgr2lab(self, bgr, weight=1.0, weight_ratio=1.0):
        if len(bgr.shape) < 4:
            t_bgr = bgr.unsqueeze(0)
            return self.xyz2lab(self.bgr2xyz(self.LinearizeRgb(t_bgr)), weight, weight_ratio).squeeze(0)
        else:
            return self.xyz2lab(self.bgr2xyz(self.LinearizeRgb(bgr.clone())), weight, weight_ratio)
    
    def lab2bgr(self, lab):
        if len(lab.shape) < 4:
            t_lab = lab.unsqueeze(0)
        else:
            t_lab = lab

        f_y = (t_lab[:,0,:,:]+0.16)/1.16
        f_x = t_lab[:,1,:,:]/5.0 + f_y
        f_z = -t_lab[:,2,:,:]/2.0 + f_y

        f_x = torch.clamp(f_x,min=0)
        f_z = torch.clamp(f_z,min=0)

        f_xyz = torch.stack([f_x, f_y, f_z],dim=1)

        threshold = (6.0/29.0*6.0/29.0*6.0/29.0)/3.0*29.0/6.0*29.0/6.0 + 4.0/29.0 # = 0.20689655172
        f_xyz[f_xyz<=threshold] = (f_xyz[f_xyz<=threshold]-4.0/29.0)*3.0/29.0*6.0/29.0*6.0
        f_xyz[f_xyz>threshold] = torch.pow(f_xyz[f_xyz>threshold], 3.0)
        xyz = f_xyz*self.D65.view(1,3,1,1).expand_as(f_xyz)

        t_xyz = xyz.view(-1,3,xyz.shape[2]*xyz.shape[3]).permute(0,2,1)
        t_bgr = torch.matmul(t_xyz, self.mat_xyz2bgr)
        L_bgr = t_bgr.permute(0,2,1).view(-1,3,xyz.shape[2],xyz.shape[3])
        
        threshold2 = 0.04045/12.92
        L_bgr[L_bgr<=threshold2] *= 12.92
        L_bgr[L_bgr>threshold2] = torch.pow(L_bgr[L_bgr>threshold2],1/2.4)*1.055-0.055

        if len(lab.shape) < 4:
            L_bgr = L_bgr.squeeze(0)

        return L_bgr

    def bgr2linearyuv(self, bgr):
        return self.bgr2yuv(self.LinearizeRgb(bgr.clone()))

    def bgr2yuv(self, bgr):
        
        if len(bgr.shape) < 4:
            t_bgr = bgr.unsqueeze(0).view(-1,3,bgr.shape[2]*bgr.shape[3]).permute(0,2,1)
        else:
            t_bgr = bgr.view(-1,3,bgr.shape[2]*bgr.shape[3]).permute(0,2,1)
        t_yuv = torch.matmul(t_bgr, self.mat_bgr2yuv)
        yuv = t_yuv.permute(0,2,1).view(-1,3,bgr.shape[2],bgr.shape[3])
        
        if len(bgr.shape) < 4:
            yuv = yuv.squeeze(0)

        return yuv
    
    def convert_color_v1(self, tensor, mode='lab'):
        
        if mode == 'yuv':
            converted_tensor = self.bgr2yuv(tensor)
        elif mode == 'lab':
            converted_tensor = self.bgr2lab(tensor)
        else:
            converted_tensor = tensor

        return converted_tensor