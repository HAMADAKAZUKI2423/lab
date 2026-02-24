from __future__ import annotations
import math
import cv2
import torch
import torch.nn.functional as F
from typing import List, Tuple

from utils import save_img_torch, stimulus
from .IBlender import IBlender

DEFAULT_PYR_LEVELS = 4

class crossDissolveContrastBlender(IBlender):
    def __init__(self, target_type: str = "content"):
        super().__init__()
        self.set_target_type(target_type)
    
    def set_target_type(self, target_type: str):
        assert target_type in ["content", "background"]
        self.target_type = target_type

    def blend(self, stim: stimulus):
        assert None not in [stim.bg, stim.ovl, stim.mask, stim.vismap]

        tau = 1
        if self.target_type == "background":
            v = 1 - stim.vismap
            w = stim.vismap
        else:
            v = stim.vismap
            w = 1 - stim.vismap

        mue_fg = torch.sum(v * stim.ovl)/torch.sum(v)
        sigma2_fg = torch.sum(v * torch.pow(stim.ovl - mue_fg, 2))/torch.sum(v)

        mue_bg = torch.sum(w * stim.bg)/torch.sum(w)
        sigma2_bg = torch.sum(w * torch.pow(stim.bg - mue_bg, 2))/torch.sum(w)

        sigma_fgbg = torch.sum(torch.sqrt(v*w)*(stim.ovl - mue_fg)*(stim.bg - mue_bg))/torch.sum(torch.sqrt(v*w))

        sigma2_p = sigma2_fg*torch.pow(v,2) + sigma2_bg*torch.pow(w,2) + 2*v*w*sigma_fgbg

        mue_p = v*mue_fg + w*mue_bg
        sigma_p_dash = v*torch.sqrt(sigma2_fg) + w*torch.sqrt(sigma2_bg)

        self.blendimg = v*stim.ovl + w*stim.bg
        self.blendimg = tau * sigma_p_dash / torch.sqrt(sigma2_p) * (self.blendimg - mue_p) + mue_p
        self.alphamap = sigma_p_dash / torch.sqrt(sigma2_p)
        self.alphamap /= 2.0

    
    def save_imgs(self, save_path: str):
        # data_list = [self.blendimg]
        # path_list = [save_path + name for name in ["blend.png"]]
        data_list = [self.alphamap, self.blendimg]
        path_list = [save_path + name for name in ["alphamap.png","blend.png"]]
        for (data, path) in zip(data_list, path_list):
            save_img_torch(path, data)


class crossDissolveContrastMultiBlender(IBlender):
    def __init__(self, target_type: str = "content", pyramid_levels: int = DEFAULT_PYR_LEVELS):
        super().__init__()
        self.set_target_type(target_type)
        self.pyramid_levels = pyramid_levels
    
    def set_target_type(self, target_type: str):
        assert target_type in ["content", "background"]
        self.target_type = target_type

    def blend(self, stim: stimulus):
        assert None not in [stim.bg, stim.ovl, stim.mask, stim.vismap]

        tau = 1
        if self.target_type == "background":
            v = 1 - stim.vismap
            w = stim.vismap
        else:
            v = stim.vismap
            w = 1 - stim.vismap
        
        # Build pyramids for images and salience maps
        ovl_pyr = build_laplacian_pyramid(stim.ovl, self.pyramid_levels)
        bg_pyr = build_laplacian_pyramid(stim.bg, self.pyramid_levels)
        v_pyr = build_gaussian_pyramid(v, self.pyramid_levels)
        w_pyr = build_gaussian_pyramid(w, self.pyramid_levels)

        # Multiresolution blending with salience maps calculated at each level
        blend_pyr = []
        v_dash_pyr = []
        for l in range(self.pyramid_levels):

            mue_fg = torch.sum(v_pyr[l] * ovl_pyr[l])/torch.sum(v_pyr[l])
            sigma2_fg = torch.sum(v_pyr[l] * torch.pow(ovl_pyr[l] - mue_fg, 2))/torch.sum(v_pyr[l])

            mue_bg = torch.sum(w_pyr[l] * bg_pyr[l])/torch.sum(w_pyr[l])
            sigma2_bg = torch.sum(w_pyr[l] * torch.pow(bg_pyr[l] - mue_bg, 2))/torch.sum(w_pyr[l])

            sigma_fgbg = torch.sum(torch.sqrt(v_pyr[l]*w_pyr[l])*(ovl_pyr[l] - mue_fg)*(bg_pyr[l] - mue_bg))/torch.sum(torch.sqrt(v_pyr[l]*w_pyr[l]))

            sigma2_p = sigma2_fg*torch.pow(v_pyr[l],2) + sigma2_bg*torch.pow(w_pyr[l],2) + 2*v_pyr[l]*w_pyr[l]*sigma_fgbg

            mue_p = v_pyr[l]*mue_fg + w_pyr[l]*mue_bg
            sigma_p_dash = v_pyr[l]*torch.sqrt(sigma2_fg) + w_pyr[l]*torch.sqrt(sigma2_bg)

            self.blendimg = v_pyr[l]*ovl_pyr[l] + w_pyr[l]*bg_pyr[l]
            blended_level = tau * sigma_p_dash / torch.sqrt(sigma2_p) * (self.blendimg - mue_p) + mue_p
            blend_pyr.append(blended_level)
            v_dash_pyr.append(sigma_p_dash / torch.sqrt(sigma2_p))
        
        self.blendimg = reconstruct_laplacian_pyramid(blend_pyr)
        self.blendimg = torch.clamp(self.blendimg, max = 1, min = 0)

        self.alphamap = v_dash_pyr[0]/2.0

    
    def save_imgs(self, save_path: str):
        # data_list = [self.blendimg]
        # path_list = [save_path + name for name in ["blend.png"]]
        data_list = [self.alphamap, self.blendimg]
        path_list = [save_path + name for name in ["alphamap.png","blend.png"]]
        for (data, path) in zip(data_list, path_list):
            save_img_torch(path, data)

class crossDissolveColorBlender(IBlender):
    def __init__(self, target_type: str = "content"):
        self.set_target_type(target_type)
        
        self.alphamap: torch.Tensor | None = None
        self.blendimg: torch.Tensor | None = None
    
    def set_target_type(self, target_type: str):
        assert target_type in ["content", "background"]
        self.target_type = target_type

    def blend(self, stim: stimulus):
        assert None not in [stim.bg, stim.ovl, stim.mask, stim.vismap]

        lamb = math.exp(2)
        epsiron = math.pow(2,-6)
        if self.target_type == "background":
            v = 1 - stim.vismap
            w = stim.vismap
        else:
            v = stim.vismap
            w = 1 - stim.vismap

        fg_ast = (1-epsiron)*(2*stim.ovl - 1)
        fg_strength = torch.max(torch.abs(fg_ast), 1, keepdim = True).values
        fg_xs = (lamb - 1)/ lamb * torch.log((lamb -1)/ (torch.pow(lamb, (1-fg_strength)) - 1))/math.log(lamb)
        fg_x = F.normalize(fg_ast) * fg_xs
        bg_ast = (1-epsiron)*(2*stim.bg - 1)
        bg_strength = torch.max(torch.abs(bg_ast), 1, keepdim = True).values
        bg_xs = (lamb - 1)/ lamb * torch.log((lamb -1)/ (torch.pow(lamb, (1-bg_strength)) - 1))/math.log(lamb)
        bg_x = F.normalize(bg_ast) * bg_xs

        blend_x = fg_x * v + bg_x * w
        blend_x_strength = torch.max(torch.abs(blend_x), 1, keepdim = True).values
        blend_x_norm = torch.norm(blend_x, dim = 1, keepdim = True)
        blend_ast = blend_x / blend_x_strength * (1 - torch.log(1 + (lamb - 1)/torch.pow(lamb, (blend_x_norm * lamb)/(lamb - 1)))/math.log(lamb))
        self.blendimg = 0.5 + 0.5*(1 - epsiron)**-1 * blend_ast
        self.alphamap = v

    def save_imgs(self, save_path: str):
        # data_list = [self.blendimg]
        # path_list = [save_path + name for name in ["blend.png"]]
        data_list = [self.alphamap, self.blendimg]
        path_list = [save_path + name for name in ["alphamap.png","blend.png"]]
        for (data, path) in zip(data_list, path_list):
            save_img_torch(path, data)

class crossDissolveSaliencyBlender(IBlender):
    def __init__(self, target_type: str = "content", save_only_img:bool = False, median_ksize=15):
        self.set_target_type(target_type)
        self.save_only_img = save_only_img
        
        self.v_dash: torch.Tensor | None = None
        self.w_dash: torch.Tensor | None = None
        self.blendimg: torch.Tensor | None = None

        self.median_ksize = median_ksize
    
    def set_target_type(self, target_type: str):
        assert target_type in ["content", "background"]
        self.target_type = target_type

    def blend(self, stim: stimulus):
        assert None not in [stim.bg, stim.ovl, stim.mask, stim.vismap]

        bin_n: int = 16
        omega: float  = 1.
        sigma: float  = 0.005
        gamma: float  = 1.
        if self.target_type == "background":
            v = 1 - stim.vismap
            w = stim.vismap
        else:
            v = stim.vismap
            w = 1 - stim.vismap

        h_fg = self.MakeColorProbabilities(stim.ovl, bin_n, gaussian = True)
        h_bg = self.MakeColorProbabilities(stim.bg, bin_n, gaussian = True)
        saliency_fg = self.CalcSaliency(h_fg, omega, median=True, kernel_size=self.median_ksize)
        saliency_bg = self.CalcSaliency(h_bg, omega, median=True, kernel_size=self.median_ksize)
        saliency_mean = saliency_fg * v + saliency_bg * w

        if True:
            norm_saliency_fg = (saliency_fg-saliency_mean)
            sorted_saliency_fg, _ = torch.sort(norm_saliency_fg.view(-1))
            r_fg = torch.searchsorted(sorted_saliency_fg, norm_saliency_fg.view(-1)).view_as(saliency_fg) / saliency_fg.numel()

            norm_saliency_bg = (saliency_bg-saliency_mean)
            sorted_saliency_bg, _ = torch.sort(norm_saliency_bg.view(-1))
            r_bg = torch.searchsorted(sorted_saliency_bg, norm_saliency_bg.view(-1)).view_as(saliency_bg) / saliency_bg.numel()
        else:
            normal = torch.distributions.normal.Normal(0,sigma)
            r_fg = normal.cdf(saliency_fg - saliency_mean)
            r_bg = normal.cdf(saliency_bg - saliency_mean)

        vr = torch.pow(v*r_fg,gamma)
        wr = torch.pow(w*r_bg,gamma)
        self.v_dash = vr/(vr+wr)
        self.w_dash = wr/(vr+wr)
        self.blendimg = stim.ovl*self.v_dash + stim.bg*self.w_dash
        self.alphamap = r_fg#self.v_dash
    
    def MakeColorProbabilities(self, img: torch.Tensor, bin_n: int, gaussian: bool = True, kernel_size: int = 5, blur_sigma: int = 2) -> torch.Tensor:
        device = img.device
        img_whc = torch.permute(img, (0,2,3,1))

        #ヒストグラム計算
        hist, bins = torch.histogramdd(img_whc.to('cpu'), bins=[bin_n, bin_n, bin_n], range=[0,1,0,1,0,1])
        hist = hist.to(device)
        hist = hist/(img_whc.shape[1]*img_whc.shape[2])

        #3dガウシアンフィルター
        if(gaussian):
            hist = hist.reshape(1,1,*hist.shape)
            k = torch.from_numpy(cv2.getGaussianKernel(kernel_size, blur_sigma)).squeeze().float().to(device)
            k3d = torch.einsum('i,j,k->ijk', k, k, k)
            k3d = k3d / k3d.sum()
            hist = F.conv3d(hist, k3d.reshape(1, 1, *k3d.shape), stride=1, padding=len(k) // 2)
            hist = hist[0,0]

        ind = torch.Tensor(img_whc.shape).to(device)
        ind[...,0] = torch.bucketize(img_whc[...,0],bins[0].to(device))
        ind[...,1] = torch.bucketize(img_whc[...,1],bins[1].to(device))
        ind[...,2] = torch.bucketize(img_whc[...,2],bins[2].to(device))
        ind = torch.clip(ind - 1, min = 0, max = bin_n -1).long()

        h = torch.Tensor(img_whc.shape[:-1]).to(device)
        h[:] = hist[ind[...,0],ind[...,1],ind[...,2]]

        return h.view(h.shape[0],-1,h.shape[1],h.shape[2])

    def CalcSaliency(self, h: torch.Tensor, omega: float, median: bool = True, kernel_size: int = 3) -> torch.Tensor:
        if(omega <= 0):
            saliency = -1 * torch.log2(h)
        else:
            saliency = (1 - torch.pow(h, omega))/(omega * math.log(2))
        if(median):
            saliency = median_blur(saliency, (kernel_size,kernel_size))
        
        return saliency
    
    def save_imgs(self, save_path: str):
        if self.save_only_img:
            # data_list = [self.blendimg]
            # path_list = [save_path + name for name in ["blend.png"]]
            data_list = [self.alphamap, self.blendimg]
            path_list = [save_path + name for name in ["alphamap.png","blend.png"]]
        else:
            data_list = [self.blendimg, self.v_dash, self.w_dash]
            path_list = [save_path + name for name in ["blend.png", "ovl_dash.png", "bg_dash.png"]]
        for (data, path) in zip(data_list, path_list):
            save_img_torch(path, data)

def median_blur(input: torch.Tensor, kernel_size: Tuple[int, int]) -> torch.Tensor:
    r"""Blur an image using the median filter.

    .. image:: _static/img/median_blur.png

    Args:
        input: the input image with shape :math:`(B,C,H,W)`.
        kernel_size: the blurring kernel size.

    Returns:
        the blurred input tensor with shape :math:`(B,C,H,W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       filtering_operators.html>`__.

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> output = median_blur(input, (3, 3))
        >>> output.shape
        torch.Size([2, 4, 5, 7])
    """

    padding: Tuple[int, int] = _compute_zero_padding(kernel_size)

    # prepare kernel
    kernel: torch.Tensor = get_binary_kernel2d(kernel_size).to(input)
    b, c, h, w = input.shape

    # map the local window to single vector
    features: torch.Tensor = F.conv2d(input.reshape(b * c, 1, h, w), kernel, padding=padding, stride=1)
    features = features.view(b, c, -1, h, w)  # BxCx(K_h * K_w)xHxW

    # compute the median along the feature axis
    median: torch.Tensor = torch.median(features, dim=2)[0]

    return median

def _compute_zero_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
    r"""Utility function that computes zero padding tuple."""
    computed: List[int] = [(k - 1) // 2 for k in kernel_size]
    return computed[0], computed[1]

def get_binary_kernel2d(window_size: Tuple[int, int]) -> torch.Tensor:
    r"""Create a binary kernel to extract the patches.

    If the window size is HxW will create a (H*W)xHxW kernel.
    """
    window_range: int = window_size[0] * window_size[1]
    kernel: torch.Tensor = torch.zeros(window_range, window_range)
    for i in range(window_range):
        kernel[i, i] += 1.0
    return kernel.view(window_range, 1, window_size[0], window_size[1])


def build_gaussian_pyramid(img: torch.Tensor, levels: int) -> list:
    pyr = [img]
    for _ in range(1, levels):
        img = F.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=False)
        pyr.append(img)
    return pyr

def build_laplacian_pyramid(img: torch.Tensor, levels: int) -> list:
    gaussian_pyr = build_gaussian_pyramid(img, levels)
    laplacian_pyr = []
    for i in range(levels - 1):
        upsampled = F.interpolate(gaussian_pyr[i + 1], scale_factor=2, mode='bilinear', align_corners=False)
        laplacian = gaussian_pyr[i] - upsampled
        laplacian_pyr.append(laplacian)
    laplacian_pyr.append(gaussian_pyr[-1])  # The final level is just the Gaussian image
    return laplacian_pyr

def reconstruct_laplacian_pyramid(laplacian_pyr: list) -> torch.Tensor:
    img = laplacian_pyr[-1]
    for i in range(len(laplacian_pyr) - 2, -1, -1):
        img = F.interpolate(img, scale_factor=2, mode='bilinear', align_corners=False)
        img = img + laplacian_pyr[i]
    return img

class crossDissolveSaliencyMultiBlender(IBlender):
    def __init__(self, target_type: str = "content", save_only_img: bool = False, pyramid_levels: int = DEFAULT_PYR_LEVELS, median_ksize=21):
        self.set_target_type(target_type)
        self.save_only_img = save_only_img
        self.pyramid_levels = pyramid_levels
        
        self.v_dash: torch.Tensor | None = None
        self.w_dash: torch.Tensor | None = None
        self.blendimg: torch.Tensor | None = None

        self.median_ksize = median_ksize

        # usertest_video_make.py内で使用するパラメータ
        self.resize = 1.0
        self.input_size = None

    def set_target_type(self, target_type: str):
        assert target_type in ["content", "background"]
        self.target_type = target_type

    def blend(self, stim: stimulus):
        assert None not in [stim.bg, stim.ovl, stim.mask, stim.vismap]

        bin_n: int = 16
        omega: float  = 1.
        sigma: float  = 0.005
        gamma: float  = 1.
        
        # Set target maps based on type
        if self.target_type == "background":
            v = 1 - stim.vismap
            w = stim.vismap
        else:
            v = stim.vismap
            w = 1 - stim.vismap

        # Build pyramids for images and salience maps
        ovl_pyr = build_laplacian_pyramid(stim.ovl, self.pyramid_levels)
        bg_pyr = build_laplacian_pyramid(stim.bg, self.pyramid_levels)
        ovl_gpyr = build_gaussian_pyramid(stim.ovl, self.pyramid_levels)
        bg_gpyr = build_gaussian_pyramid(stim.bg, self.pyramid_levels)
        v_pyr = build_gaussian_pyramid(v, self.pyramid_levels)
        w_pyr = build_gaussian_pyramid(w, self.pyramid_levels)

        # Multiresolution blending with salience maps calculated at each level
        blend_pyr = []
        v_dash_pyr = []
        for l in range(self.pyramid_levels):
            # Reduce median filter size by half at each level
            median_filter_size = max(self.median_ksize // (2 ** l), 1)
            if median_filter_size%2==0:
                median_filter_size += 1

            # Calculate salience for the current level
            saliency_fg = self.CalcSaliency(self.MakeColorProbabilities(ovl_gpyr[l], bin_n), omega, median=True, kernel_size=median_filter_size)
            saliency_bg = self.CalcSaliency(self.MakeColorProbabilities(bg_gpyr[l], bin_n), omega, median=True, kernel_size=median_filter_size)
            # Calculate mean saliency
            saliency_mean = saliency_fg * v_pyr[l] + saliency_bg * w_pyr[l]
            
            if True:
                norm_saliency_fg = (saliency_fg-saliency_mean)
                sorted_saliency_fg, _ = torch.sort(norm_saliency_fg.view(-1))
                r_fg = torch.searchsorted(sorted_saliency_fg, norm_saliency_fg.view(-1)).view_as(saliency_fg) / saliency_fg.numel()

                norm_saliency_bg = (saliency_bg-saliency_mean)
                sorted_saliency_bg, _ = torch.sort(norm_saliency_bg.view(-1))
                r_bg = torch.searchsorted(sorted_saliency_bg, norm_saliency_bg.view(-1)).view_as(saliency_bg) / saliency_bg.numel()
            else:
                
                normal = torch.distributions.normal.Normal(0, sigma)
                r_fg = normal.cdf(saliency_fg - saliency_mean)
                r_bg = normal.cdf(saliency_bg - saliency_mean)

            # Power law application
            vr = torch.pow(v_pyr[l] * r_fg, gamma)
            wr = torch.pow(w_pyr[l] * r_bg, gamma)
            v_dash_l = vr / (vr + wr)
            w_dash_l = wr / (vr + wr)
            v_dash_pyr.append(v_dash_l)

            # Blend at current pyramid level
            blended_level = ovl_pyr[l] * v_dash_l + bg_pyr[l] * w_dash_l
            blend_pyr.append(blended_level)

        # Reconstruct the final blended image from the pyramid
        self.blendimg = reconstruct_laplacian_pyramid(blend_pyr)
        self.blendimg = torch.clamp(self.blendimg, max = 1, min = 0)

        self.alphamap = v_dash_pyr[0]

    # def build_gaussian_pyramid(self, img: torch.Tensor, levels: int) -> list:
    #     pyr = [img]
    #     for _ in range(1, levels):
    #         img = F.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=False)
    #         pyr.append(img)
    #     return pyr

    # def build_laplacian_pyramid(self, img: torch.Tensor, levels: int) -> list:
    #     gaussian_pyr = self.build_gaussian_pyramid(img, levels)
    #     laplacian_pyr = []
    #     for i in range(levels - 1):
    #         upsampled = F.interpolate(gaussian_pyr[i + 1], scale_factor=2, mode='bilinear', align_corners=False)
    #         laplacian = gaussian_pyr[i] - upsampled
    #         laplacian_pyr.append(laplacian)
    #     laplacian_pyr.append(gaussian_pyr[-1])  # The final level is just the Gaussian image
    #     return laplacian_pyr

    # def reconstruct_laplacian_pyramid(self, laplacian_pyr: list) -> torch.Tensor:
    #     img = laplacian_pyr[-1]
    #     for i in range(len(laplacian_pyr) - 2, -1, -1):
    #         img = F.interpolate(img, scale_factor=2, mode='bilinear', align_corners=False)
    #         img = img + laplacian_pyr[i]
    #     return img

    def MakeColorProbabilities(self, img: torch.Tensor, bin_n: int, gaussian: bool = True, kernel_size: int = 5, blur_sigma: int = 2) -> torch.Tensor:
        device = img.device
        img_whc = torch.permute(img, (0,2,3,1))

        # Histogram calculation
        hist, bins = torch.histogramdd(img_whc.to('cpu'), bins=[bin_n, bin_n, bin_n], range=[0,1,0,1,0,1])
        hist = hist.to(device)
        hist = hist / (img_whc.shape[1] * img_whc.shape[2])

        # 3D Gaussian filtering
        if gaussian:
            hist = hist.reshape(1,1,*hist.shape)
            k = torch.from_numpy(cv2.getGaussianKernel(kernel_size, blur_sigma)).squeeze().float().to(device)
            k3d = torch.einsum('i,j,k->ijk', k, k, k)
            k3d = k3d / k3d.sum()
            hist = F.conv3d(hist, k3d.reshape(1, 1, *k3d.shape), stride=1, padding=len(k) // 2)
            hist = hist[0,0]

        ind = torch.Tensor(img_whc.shape).to(device)
        ind[...,0] = torch.bucketize(img_whc[...,0],bins[0].to(device))
        ind[...,1] = torch.bucketize(img_whc[...,1],bins[1].to(device))
        ind[...,2] = torch.bucketize(img_whc[...,2],bins[2].to(device))
        ind = torch.clip(ind - 1, min=0, max=bin_n - 1).long()

        h = torch.Tensor(img_whc.shape[:-1]).to(device)
        h[:] = hist[ind[...,0],ind[...,1],ind[...,2]]

        return h.view(h.shape[0],-1,h.shape[1],h.shape[2])

    def CalcSaliency(self, h: torch.Tensor, omega: float, median: bool = True, kernel_size: int = 3) -> torch.Tensor:
        if omega <= 0:
            saliency = -1 * torch.log2(h)
        else:
            saliency = (1 - torch.pow(h, omega)) / (omega * math.log(2))
        if median:
            saliency = median_blur(saliency, (kernel_size, kernel_size))
        
        return saliency

    def save_imgs(self, save_path: str):
        if self.save_only_img:
            # data_list = [self.blendimg]
            # path_list = [save_path + name for name in ["blend.png"]]
            data_list = [self.alphamap, self.blendimg]
            path_list = [save_path + name for name in ["alphamap.png","blend.png"]]
        else:
            data_list = [self.blendimg, self.v_dash, self.w_dash]
            path_list = [save_path + name for name in ["blend.png", "ovl_dash.png", "bg_dash.png"]]
        for (data, path) in zip(data_list, path_list):
            save_img_torch(path, data)
