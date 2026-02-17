from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from utils import save_img_torch, stimulus
from .IBlender import IBlender

eps = 1e-12

class sandorBlender(IBlender):
    def __init__(self, device: torch.device, target_type: str = "content", use_motion: bool = False):
        super().__init__()
        self.device = device
        self.set_target_type(target_type)

        self.filt = self.gauss_kernel(3)
        self.pad_two = nn.ReflectionPad2d(2)

        self.use_motion = use_motion

        self.prev_ovl = None
        self.prev_bg = None

    def set_target_type(self, target_type: str):
        assert target_type in ["content", "background"]
        self.target_type = target_type

    def blend(self, stim: stimulus):
        assert None not in [stim.bg, stim.ovl, stim.mask, stim.vismap]

        alphamap = stim.vismap * stim.mask
        alphamap = alphamap.expand(-1,3,-1,-1)
        if self.target_type == "content":
            self.blendimg, self.alphamap = self.process(stim.ovl, stim.bg, alphamap)
        else:
            self.blendimg, self.alphamap = self.process(stim.bg, stim.ovl, alphamap)
        
        self.prev_ovl = stim.ovl
        self.prev_bg = stim.bg

    def save_imgs(self, save_path: str):
        data_list = [self.alphamap, self.blendimg]
        path_list = [save_path + name for name in ["alphamap.png","blend.png"]]
        for (data, path) in zip(data_list, path_list):
            save_img_torch(path, data)

    def process(self, occluder, occluded, mask, save = False):
        derSal = self.makeSaliencyMap(occluder, self.prev_ovl)
        derEdg = self.makeEdgeMap(occluder, derSal, weight = 3)
        dedSal = self.makeSaliencyMap(occluded, self.prev_bg)
        derSalDash = torch.clip(derSal - dedSal + derEdg, 0, 1)
        if save:
            save_img_torch("sandor_derSal.png", derSal)
            save_img_torch("sandor_derEdg.png", derEdg)
            save_img_torch("sandor_dedSal.png", dedSal)
            save_img_torch("sandor_derSalDash.png", derSalDash)
        blend = occluder * mask + (occluder * derSalDash + occluded * (1 - derSalDash)) * (1 - mask)
        return blend, mask + derSalDash * (1 - mask)

    def makeEdgeMap(self, img, saliency, weight = 1):
        sobelEdge = self.sobelFunction(img)
        return torch.abs(sobelEdge * saliency * weight)
    
    def sobelFunction(self, img):
        pad_img = F.pad(img,(1,1,1,1),mode='reflect')

        hkernel = torch.Tensor([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]]).to(self.device)

        hkernel = hkernel.view((1,1,3,3))
        hkernel = torch.cat([hkernel]*3,dim=1)

        vkernel = torch.Tensor([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]]).to(self.device)

        vkernel = vkernel.view((1,1,3,3))
        vkernel = torch.cat([vkernel]*3,dim=1)

        dkernel1 = torch.Tensor([[0, 1, 2],
                                    [-1, 0, 1],
                                    [-2, -1, 0]]).to(self.device)
            
        dkernel1 = dkernel1.view((1,1,3,3))
        dkernel1 = torch.cat([dkernel1]*3,dim=1)
        
        dkernel2 = torch.Tensor([[2, 1, 0],
                                [1, 0, -1],
                                [0, -1, -2]]).to(self.device)
        
        dkernel2 = dkernel2.view((1,1,3,3))
        dkernel2 = torch.cat([dkernel2]*3,dim=1)

        G_x = torch.abs(F.conv2d(pad_img, hkernel))
        G_y = torch.abs(F.conv2d(pad_img, vkernel))
        G_d1 = torch.abs(F.conv2d(pad_img, dkernel1))
        G_d2 = torch.abs(F.conv2d(pad_img, dkernel2))

        return (G_x + G_y + G_d1 + G_d2)/4

    def makeSaliencyMap(self, img, prev_img=None):# bgr
        b = img[:,:1]
        g = img[:,1:2]
        r = img[:,2:]
        maxBgr = torch.max(img, dim = 1, keepdim = True).values
        luminosity = (b + g + r) / 3.
        rg_opponency = (r - g) / (maxBgr + eps)
        by_opponency = (b - torch.min(torch.cat([g,r], dim=1), dim = 1, keepdim = True).values) / (maxBgr + eps)
        saliencyMaterial = torch.cat([luminosity,rg_opponency,by_opponency], dim=1)
        # original paper consider motion saliency, but we dont

        if self.use_motion and type(prev_img) == torch.Tensor:
            b = prev_img[:,:1]
            g = prev_img[:,1:2]
            r = prev_img[:,2:]
            prev_luminosity = (b + g + r) / 3.
            motion_map = luminosity - prev_luminosity

        
        saliencyPyr = self.gen_originalScale_Gpyr(saliencyMaterial,8) # 0-7
        saliencyTmp = torch.zeros_like(saliencyPyr[0])
        for up_layer in [1,2,3]:
            for layer_diff in [3,4]:
                saliencyTmp += self.min_max_Norm(torch.abs(saliencyPyr[up_layer] - saliencyPyr[up_layer+layer_diff]))
        
        return self.min_max_Norm(torch.mean(saliencyTmp, dim = 1, keepdim=True))

    def gen_originalScale_Gpyr(self, image, level):
        J = image
        dims = image.shape[1]
        #pyr = []
        gpyr=[]
        for i in range(level):
            I = F.conv2d(self.pad_two(J), self.filt, stride=2, padding=0,
                         groups=dims)
            I_up = I
            for j in range(i+1):
                I_up = self.upsample(I_up)#include conv
            I_up = torchvision.transforms.functional.resize(img=I_up, size=(image.shape[2], image.shape[3]),antialias = True)
            gpyr.append(I_up)

            J = I
        return gpyr
    
    def gen_Gpyr(self, image, level):
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
    
    def gauss_kernel(self,channels=3):
        kernel = torch.tensor([
                [1, 5, 8, 5, 1],
                [5, 25, 40, 25, 5],
                [8, 40, 64, 40, 8],
                [5, 25, 40, 25, 5],
                [1, 5, 8, 5, 1]],dtype=torch.float32,device=self.device)
        kernel /= 400.
        kernel = kernel.repeat(channels, 1, 1, 1)
        return kernel

    def upsample(self, x, kernel=None):
        # cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        # cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
        # cc = cc.permute(0,1,3,2)
        # cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2, device=x.device)], dim=3)
        # cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
        # x_up = cc.permute(0,1,3,2)
        # if kernel is None:
        #     kernel = self.gauss_kernel(channels=x.shape[1])
        # return self.conv_gauss(x_up, 4*kernel)
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
    
    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out
    
    def min_max_Norm(self, img, new_min = 0, new_max = 1):
        img_min, img_max = img.min(), img.max()
        #print(f"SandorModel min_max_Norm: min({img_min}), max({img_max})")
        return (img - img_min)/(img_max - img_min + eps)*(new_max - new_min) + new_min