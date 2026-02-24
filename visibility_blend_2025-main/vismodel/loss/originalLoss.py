from __future__ import annotations
import torch
import torch.nn.functional as F
import math
import numpy as np
import json
from matplotlib import pyplot as plt
from .ILossFunction import ILossFunction
from ..supermodels.visModel import VisModel
from ..vismodel_mlp import VisModel_MLP
from utils import save_img_torch, save_grayimg_plt


eps = 1e-8

class OriginalLoss(ILossFunction):
    def __init__(self,
                 smo_lambda:float,
                 vis_lambda:float,
                 fid_lambda:float,
                 calc_max_vis: bool,
                 clip_target_vis: bool,
                 use_norm_vis_for_loss: bool,
                 smooth_loss_grad_weight: int,
                 device: torch.device,
                 l2_loss: bool = False,
                 lp_loss: float = 0,
                 use_spatial_weight: bool = False,
                 asymmetric_loss: bool = False,
                 penalty_factor: float = 1.0,
                 aggregate_vismap: bool = False,
                 vis_agg_ksize: int = 15):
        self.smo_lambda = smo_lambda
        self.vis_lambda = vis_lambda
        self.fid_lambda = fid_lambda
        self.calc_max_vis = calc_max_vis
        self.clip_target_vis = clip_target_vis
        self.use_norm_vis_for_loss = use_norm_vis_for_loss
        self.smooth_loss_grad_weight = smooth_loss_grad_weight

        self.use_spatial_weight = use_spatial_weight
        self.asymmetric_loss = asymmetric_loss
        self.penalty_factor = penalty_factor

        self.corr_ksize: int = 15
        self.corr_sigma: float = 6.0
        self.precise_fid: bool = True,
        self.fid_loss_numband: int | None = None
        self.device = device
        self.l2_loss = l2_loss

        self.lp_loss = lp_loss

        self.aggregate_vismap = aggregate_vismap
        self.vis_agg_ksize = vis_agg_ksize
        self.vis_agg_sigma = self.vis_agg_ksize/2.5

        if self.aggregate_vismap:
            self.vis_agg_kernel = self.__get_vis_agg_kernel()

        self.fid_kernel = self.__get_fid_rho_kernel()
        self.reset_state()
        self.reset_loss()
    
    def reset_state(self):
        super().reset_state()
        self.smooth_kernel: dict[str, torch.Tensor] | None = None
        self.content_info: list[dict[str]] | None  = None
        self.vis_map_max: torch.Tensor | None  = None
        self.vis_map_min: torch.Tensor | None  = None
        self.vis_scale_max: torch.Tensor | None  = None
        self.vis_scale_min: torch.Tensor | None  = None

    def reset_loss(self):
        self.vis_scale_min: torch.Tensor | None  = None
        self.vis_loss: torch.Tensor | None  = None
        self.vis_loss_map: torch.Tensor | None  = None
        self.smo_loss: torch.Tensor | None  = None
        self.smo_loss_map: torch.Tensor | None  = None
        self.fid_loss: torch.Tensor | None  = None
        self.fid_loss_map: torch.Tensor | None  = None
        self.all_loss: torch.Tensor | None  = None

        self.vis_loss_list: list[float] = []
        self.fid_loss_list: list[float] = []
        self.smo_loss_list: list[float] = []
        self.all_loss_list: list[float] = []
        self.loss_count: int = 0

    def compute_loss_preprocess(self, target_vis:torch.Tensor, vismodel:VisModel):
        self.smooth_kernel = self.__get_smooth_loss_data(vismodel, vismodel.get_overlaid())
        if vismodel.target_type == "background":
            self.content_info = self.__calc_band_simga(vismodel.ref_pyr)
        elif vismodel.target_type == "content":
            self.content_info = self.__calc_band_simga(vismodel.target_pyr)
        vismodel.compute_weights()

        self.target_vis = target_vis
        if isinstance(vismodel, VisModel_MLP):
            self.target_vis = vismodel.get_downsampled_map(self.target_vis)
        
        # if self.calc_max_vis:
        self.__set_visloss_scale(vismodel)

        if self.clip_target_vis:
            # self.target_vis = torch.clamp(self.target_vis, max = self.norm_vis_map_max)
            self.target_vis = torch.minimum(self.target_vis, self.norm_vis_map_max)
        
        if self.use_spatial_weight:
            if vismodel.spatial_weight is not None:
                self.spatial_weight = vismodel.spatial_weight
            else:
                self.spatial_weight = self.norm_vis_map_max ** 2
        else:
            self.spatial_weight = None
            

        self.target_vis_rawscale = vismodel.visibility_to_rawscale(self.target_vis)
        
        # if self.spatial_weighting:
        #     if self.target_type == "background":
        #         self.vismodel.set_blend(self.bg)
        #     elif self.target_type == "content":
        #         self.vismodel.set_blend(self.ovl)
        #     self.vismodel.compute_spatial_weights()

    def compute_loss(self,
                      vismodel:VisModel,
                      alphamap:torch.Tensor):
                      # spatial_weight: torch.Tensor | None = None):
        if self.spatial_weight is not None:
            spatial_weight = self.spatial_weight * vismodel.mask_data#vismodel.dilated_mask_gp[0]
            spatial_weight_sum = spatial_weight.sum(dim=(1,2,3))
        elif isinstance(vismodel, VisModel_MLP):
            spatial_weight = vismodel.mask_data
            spatial_weight_sum = vismodel.mask_data_sum
        else:
            spatial_weight = vismodel.dilated_mask_gp[0]
            spatial_weight_sum = vismodel.dilated_mask_gp_sum[0]
            
        blend: torch.Tensor = vismodel.blending(vismodel.get_raw_overlaid(), vismodel.get_background(), alphamap)
        vismodel.set_blend(blend)
        vismodel.compute_visibility_wo_weight()
        vis_map = vismodel.vis_map


        if vismodel.target_type == "background":
            # fidelity lossはforegroundを対象とする
            # content_component, _ = self.vismodel.contrast_extraction(self.vismodel.target_pyr, self.vismodel.blend_pyr)
            if self.precise_fid:
                content_component, _ = vismodel.partial_correlation_extraction_precise(vismodel.ref_pyr, vismodel.target_pyr, vismodel.blend_pyr, vismodel.level)
            else:
                content_component, _ = vismodel.partial_correlation_extraction(vismodel.ref_pyr, vismodel.target_pyr, vismodel.blend_pyr, vismodel.level)
            
        elif vismodel.target_type == "content":
            # content_component, _ = self.vismodel.contrast_extraction(self.vismodel.ref_pyr, self.vismodel.blend_pyr)
            if self.precise_fid:
                content_component, _ = vismodel.partial_correlation_extraction_precise(vismodel.target_pyr, vismodel.ref_pyr, vismodel.blend_pyr, vismodel.level)
            else:
                content_component, _ = vismodel.partial_correlation_extraction(vismodel.target_pyr, vismodel.ref_pyr, vismodel.blend_pyr, vismodel.level)
        

        fid_loss, fid_loss_map = self.__compute_corr_fid_loss(vismodel, content_component, numband=self.fid_loss_numband)
        smo_loss, smo_loss_map = self.__calc_smooth_loss(alphamap)

        if self.aggregate_vismap:
            pad_num = (self.vis_agg_kernel.shape[-1]-1)//2
        
            _agg_vismap = F.conv2d(F.pad( (spatial_weight * vismodel.norm_vismap) ** vismodel.lp_norm, (pad_num,pad_num,pad_num,pad_num), mode='reflect'), self.vis_agg_kernel, stride=1, padding=0, groups=1)
            vismodel.norm_vismap = (_agg_vismap + eps) ** (1/vismodel.lp_norm)

        suppress_rate = 0.1
        if self.calc_max_vis:
            if vismodel.target_type == "background":
                visloss_map: torch.Tensor = self.vis_scale_max * self.vis_scale_min * torch.abs(vis_map - self.target_vis_rawscale) + suppress_rate*((1.-self.vis_scale_max)*alphamap+(1.-self.vis_scale_min)*(1.-alphamap))
            if vismodel.target_type == "content":
                visloss_map: torch.Tensor = self.vis_scale_max * self.vis_scale_min * torch.abs(vis_map - self.target_vis_rawscale) + suppress_rate*((1.-self.vis_scale_max)*(1.-alphamap)+(1.-self.vis_scale_min)*alphamap)
            #visloss_map = scaleMax * scaleMin * torch.abs(target_vis - vismap)
            vis_loss_map = spatial_weight * visloss_map
            visloss = visloss_map.sum(dim=(1,2,3))/spatial_weight_sum
            vis_loss = visloss.mean()
        else:
            if self.use_norm_vis_for_loss:
                if self.asymmetric_loss:
                    over_loss = torch.relu(vismodel.norm_vismap - self.target_vis)# * self.penalty_factor
                    under_loss = torch.relu(self.target_vis - vismodel.norm_vismap)

                    if self.lp_loss>0:
                        over_loss = spatial_weight * torch.pow(over_loss + eps, self.lp_loss)
                        under_loss = spatial_weight * torch.pow(under_loss + eps, self.lp_loss)
                    elif self.l2_loss:
                        over_loss = spatial_weight * over_loss * over_loss
                        under_loss = spatial_weight * under_loss * under_loss
                    else:
                        over_loss = spatial_weight * over_loss
                        under_loss = spatial_weight * under_loss
                else:
                    # vis_loss_map = spatial_weight*torch.abs(self.vismodel.visibility_to_norm(vis_map) - self.tg_vis)
                    _vis_loss_map = torch.abs(vismodel.norm_vismap - self.target_vis)
                    if self.lp_loss>0:
                        vis_loss_map = spatial_weight * torch.pow(_vis_loss_map + eps, self.lp_loss)
                    elif self.l2_loss:
                        vis_loss_map = spatial_weight * _vis_loss_map * torch.abs(vismodel.norm_vismap - self.target_vis)
                    else:
                        vis_loss_map = spatial_weight * _vis_loss_map
            else:
                _vis_loss_map = spatial_weight*torch.abs(vis_map - self.target_vis_rawscale)
                if self.lp_loss>0:
                    vis_loss_map = torch.pow(_vis_loss_map + eps, self.lp_loss)
                elif self.l2_loss:
                    vis_loss_map = _vis_loss_map * torch.abs(vis_map - self.target_vis_rawscale)
                else:
                    vis_loss_map = _vis_loss_map
            
            if self.asymmetric_loss:
                vis_loss_map = over_loss + under_loss
                _over_loss = over_loss.sum(dim=(1,2,3))/spatial_weight_sum
                _under_loss = under_loss.sum(dim=(1,2,3))/spatial_weight_sum
                if self.lp_loss>0:
                    visloss = self.penalty_factor * torch.pow(_over_loss, 1/self.lp_loss) + torch.pow(_under_loss, 1/self.lp_loss)
                else:
                    visloss = self.penalty_factor * _over_loss + _under_loss
            else:
                _visloss = vis_loss_map.sum(dim=(1,2,3))/spatial_weight_sum
                if self.lp_loss>0:
                    visloss = torch.pow(_visloss, 1/self.lp_loss)
                else:
                    visloss = _visloss
            vis_loss = visloss.mean()

        self.vis_loss = vis_loss 
        self.vis_loss_map = vis_loss_map 
        self.smo_loss = smo_loss 
        self.smo_loss_map = smo_loss_map 
        self.fid_loss = fid_loss 
        self.fid_loss_map = fid_loss_map 
        self.all_loss = self.fid_loss * self.fid_lambda + self.vis_loss * self.vis_lambda+ self.smo_loss * self.smo_lambda

        self.vis_loss_list.append(self.vis_loss.item())
        self.fid_loss_list.append(self.fid_loss.item())
        self.smo_loss_list.append(self.smo_loss.item())
        self.all_loss_list.append(self.all_loss.item())
        self.loss_count += 1

    def save_loss(self, dir_path: str):
        assert self.all_loss_list != None
        epoch_list = list(range(len(self.all_loss_list)))

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(np.array(epoch_list),np.array(self.all_loss_list), color='black',  linestyle='solid', linewidth = 1.0, label='all')
        if self.fid_lambda>0:
            ax.plot(np.array(epoch_list),np.array(self.fid_loss_list), color='red',  linestyle='solid', linewidth = 1.0, label='fid_loss')
        if self.vis_lambda>0:
            ax.plot(np.array(epoch_list),np.array(self.vis_loss_list), color='blue',  linestyle='solid', linewidth = 1.0, label='vis_loss')
        if self.smo_lambda>0:
            ax.plot(np.array(epoch_list),np.array(self.smo_loss_list), color='green',  linestyle='solid', linewidth = 1.0, label='smo_loss')
        
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.legend()
        plt.savefig(f"{dir_path}visblend_lr.png")

        output_dic = {
                'vis_loss': self.vis_loss_list[-1] if self.vis_lambda>0 else 'none',
                'fidelity_loss':self.fid_loss_list[-1] if self.fid_lambda>0 else 'none',
                'smooth_loss':self.smo_loss_list[-1] if self.smo_lambda>0 else 'none',
                'all_loss':self.all_loss_list[-1],
            }
        with open(f'{dir_path}output_info.json', 'w') as f:
            json.dump(output_dic, f, indent=4)

    def save_img(self, dir_path: str):
        img_dict = {
            "vis_map_max":self.vis_map_max,
            "vis_map_min":self.vis_map_min,
            "vis_scale_max":self.vis_scale_max,
            "vis_scale_min":self.vis_scale_min,
            "target_vis_rawscale":self.target_vis_rawscale,
            "fid_loss_map":self.fid_loss_map,
            "smo_loss_map":self.smo_loss_map,
            "vis_loss_map":self.vis_loss_map
        }
        norm_imgs = ["vis_scale_max","vis_scale_min","opt_alphamap"]
        for key, img in img_dict.items():
            path = f'{dir_path}{key}.png'
            if key in norm_imgs:
                save_grayimg_plt(path, img, norm = True)
            else:
                save_grayimg_plt(path, img, norm = False)
        
            if key in norm_imgs:
                save_img_torch(f'{dir_path}{key}_gray.png', img)
    
    def print_loss(self):
        message = f'loss: {sum(self.all_loss_list)/self.loss_count:.7f}, ' \
            f'vis_loss: {sum(self.vis_loss_list)/self.loss_count:.7f}, ' \
            f'fidelity_loss: {sum(self.fid_loss_list)/self.loss_count:.7f}, ' \
            f'smooth_loss: {sum(self.smo_loss_list)/self.loss_count:.7f}, '
        self.reset_loss()
        return message
    
    def __set_visloss_scale(self, vismodel: VisModel):
        # 10/15 fixed a bug
        # vismodel.set_blend(vismodel.target)
        vismodel.set_blend(vismodel.raw_ovl)
        
        with torch.no_grad():
            vismodel.compute_visibility_wo_weight()
        vis_map_max = vismodel.vis_map

        if False:
            from PIL import Image
            ref_arr = vismodel.ref[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            tgt_arr = vismodel.raw_ovl[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            op_arr = vismodel.opaque_blend[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()            
            arr = np.hstack([ref_arr, tgt_arr, op_arr])
            im = Image.fromarray(arr)
            im.save("debug_tgt.png")

            vis_arr = vismodel.norm_vismap[0,0].mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
            im = Image.fromarray(vis_arr)
            im.save("debug_vismax.png")

        self.norm_vis_map_max = vismodel.norm_vismap.clone()

        with torch.no_grad():
            vismodel.set_blend(vismodel.ref)
            vismodel.compute_visibility_wo_weight()
        vis_map_min = vismodel.vis_map

        self.vis_map_max = vis_map_max
        self.vis_map_min = vis_map_min

        
        # self.vis_scale_max = torch.exp(-1 * torch.clamp((self.target_vis_rawscale - self.vis_map_max),min=0))
        # self.vis_scale_min = torch.exp(-1 * torch.clamp((self.vis_map_min - self.target_vis_rawscale),min=0))

        if vismodel.sigmoid_type.startswith('bezier'):
            vismodel.max_vis_map = vis_map_max.clone()
            vismodel.max_vis_score = vismodel.raw_score.clone()

    def __get_smooth_loss_data(self, vismodel: VisModel, image: torch.Tensor, use_advanced_sobel_filter: bool = True) -> dict[str, torch.Tensor]:
        image_yuv = vismodel.convert_color_v1(image, vismodel.col_conversion)
        image_y = image_yuv[:,0,:,:].unsqueeze(1)

        hkernel = torch.Tensor([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]]).to(self.device)

        hkernel = hkernel.view((1,1,3,3))

        vkernel = torch.Tensor([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]]).to(self.device)
                
        vkernel = vkernel.view((1,1,3,3))

        G_x = F.conv2d(F.pad(image_y,pad=(1,1,1,1),mode='reflect'), hkernel)
        G_y = F.conv2d(F.pad(image_y,pad=(1,1,1,1),mode='reflect'), vkernel)

        hweight = torch.exp(-torch.abs(G_x*self.smooth_loss_grad_weight))
        vweight = torch.exp(-torch.abs(G_y*self.smooth_loss_grad_weight))

        grad_kernels = {'h':hkernel, 'v':vkernel, 'hw':hweight, 'vw':vweight}

        if use_advanced_sobel_filter:
            dkernel1 = torch.Tensor([[0, 1, 2],
                                    [-1, 0, 1],
                                    [-2, -1, 0]]).to(self.device)
            
            dkernel1 = dkernel1.view((1,1,3,3))
            
            dkernel2 = torch.Tensor([[2, 1, 0],
                                    [1, 0, -1],
                                    [0, -1, -2]]).to(self.device)
            
            dkernel2 = dkernel2.view((1,1,3,3))

            G_d1 = F.conv2d(F.pad(image_y,pad=(1,1,1,1),mode='reflect'), dkernel1)
            G_d2 = F.conv2d(F.pad(image_y,pad=(1,1,1,1),mode='reflect'), dkernel2)

            d1weight = torch.exp(-torch.abs(G_d1*self.smooth_loss_grad_weight))
            d2weight = torch.exp(-torch.abs(G_d2*self.smooth_loss_grad_weight))

            grad_kernels['d1']=dkernel1
            grad_kernels['d2']=dkernel2
            grad_kernels['d1w']=d1weight
            grad_kernels['d2w']=d2weight
        
        return grad_kernels
    
    def __calc_smooth_loss(self, alphamap: torch.Tensor) -> tuple[torch.Tensor]:
        assert self.smooth_kernel != None
        G_x = F.conv2d(F.pad(alphamap[:,0,:,:].unsqueeze(1),pad=(1,1,1,1),mode='reflect'), self.smooth_kernel ['h'])
        G_y = F.conv2d(F.pad(alphamap[:,0,:,:].unsqueeze(1),pad=(1,1,1,1),mode='reflect'), self.smooth_kernel ['v'])

        if len(self.smooth_kernel.keys())>4:
            #use advanced sobel filter
            G_d1 = F.conv2d(F.pad(alphamap[:,0,:,:].unsqueeze(1),pad=(1,1,1,1),mode='reflect'), self.smooth_kernel ['d1'])
            G_d2 = F.conv2d(F.pad(alphamap[:,0,:,:].unsqueeze(1),pad=(1,1,1,1),mode='reflect'), self.smooth_kernel ['d2'])

            smooth_loss_map = (torch.abs(G_x)*self.smooth_kernel ['hw'] + torch.abs(G_y)*self.smooth_kernel ['vw'] + 
                        torch.abs(G_d1)*self.smooth_kernel ['d1w'] + torch.abs(G_d2)*self.smooth_kernel ['d2w'])/2.
            smooth_loss = smooth_loss_map.mean()        
            #smooth_loss = (torch.abs(G_x)*grad_kernels['hw'] + torch.abs(G_y)*grad_kernels['vw'] + torch.abs(G_d1)*grad_kernels['d1w'] + torch.abs(G_d2)*grad_kernels['d2w']).mean()*0.5
        else:
            smooth_loss_map = torch.abs(G_x)*self.smooth_kernel ['hw'] + torch.abs(G_y)*self.smooth_kernel ['vw']
            smooth_loss = smooth_loss_map.mean()

        return smooth_loss, smooth_loss_map

    def __get_fid_rho_kernel(self) -> torch.Tensor:
        assert self.corr_ksize != None and self.corr_sigma != None
        kernel = self.__get_custom_gaussian_kernel(self.corr_ksize,self.corr_sigma,self.device)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(3, *[1] * (kernel.dim() - 1))
        return kernel

    def __get_vis_agg_kernel(self) -> torch.Tensor:
        assert self.vis_agg_ksize != None and self.vis_agg_sigma != None
        kernel = self.__get_custom_gaussian_kernel(self.vis_agg_ksize,self.vis_agg_sigma,self.device)
        kernel = kernel.view(1, 1, *kernel.size())
        # kernel = kernel.repeat(3, *[1] * (kernel.dim() - 1))
        return kernel
    
    def __get_custom_gaussian_kernel(self, ksize: int, sigma: float, device: torch.device) -> torch.Tensor:
        meshgrids = torch.meshgrid(
            [
                torch.arange(ksize, dtype=torch.float32).to(device=device),
                torch.arange(ksize, dtype=torch.float32).to(device=device)
            ]
        , indexing="ij")
        kernel = 1
        mean = (ksize - 1) / 2
        for mgrid in meshgrids:
            kernel *= 1 / (sigma * math.sqrt(2 * math.pi)) * \
                        torch.exp(-((mgrid - mean) / sigma) ** 2 / 2)
        
        kernel = kernel / torch.sum(kernel)

        return kernel
    
    def __calc_band_simga(self, pyr: list[torch.Tensor]) -> list[dict[str]]:
        pad_num = (self.fid_kernel.shape[-1]-1)//2
        band_val = []
        for i in range(len(pyr)-1):
            if self.precise_fid:
                resp_mean = F.conv2d(F.pad(pyr[i], (pad_num,pad_num,pad_num,pad_num), mode='reflect'), self.fid_kernel, stride=1, padding=0, groups=pyr[i].shape[1])
                resp_sigma = (pyr[i]-resp_mean)**2
            else:
                resp_sigma = pyr[i]**2
            resp_sigma = F.conv2d(F.pad(resp_sigma, (pad_num,pad_num,pad_num,pad_num), mode='reflect'), self.fid_kernel, stride=1, padding=0, groups=resp_sigma.shape[1])
            
            band_val.append({'val':pyr[i],'sigma':(resp_sigma+eps)**0.5, 'mean':resp_mean})
        return band_val
    
    def __compute_corr_fid_loss(self, vismodel: VisModel, content_component: list[torch.Tensor], numband: int | None = None) -> tuple[torch.Tensor]:
        assert self.content_info != None
        pad_num = (self.fid_kernel.shape[-1]-1)//2

        if numband is None:
            numband = len(content_component)-1#model.level-1
        
        cur_dict = self.__calc_band_simga(content_component)

        cov_eps = 0.01

        total = []
        total_map = []
        for i in range(numband):
            if self.precise_fid:
                co_var = (cur_dict[i]['val']-cur_dict[i]['mean'])*(self.content_info[i]['val']-self.content_info[i]['mean'])
            else:
                co_var = cur_dict[i]['val']*self.content_info[i]['val']
            co_var = F.conv2d(F.pad(co_var, (pad_num,pad_num,pad_num,pad_num), mode='reflect'), self.fid_kernel, stride=1, padding=0, groups=co_var.shape[1])
            corr_resp: torch.Tensor = (F.relu(co_var)+cov_eps)/(cur_dict[i]['sigma']*self.content_info[i]['sigma']+cov_eps)

            if False:
                # cur_dict[i]['val'], self.content_info[i]['val'], corr_respを表示 [1,3,H,W]
                # subplotで３つ並べて表示
                plt.subplot(1,3,1)
                corr_resp_np = corr_resp.detach().cpu().numpy()[0]
                corr_resp_np = np.transpose(corr_resp_np, [1,2,0])
                cur_val_np = cur_dict[i]['val'].detach().cpu().numpy()[0] + 0.5
                cur_val_np = np.transpose(cur_val_np, [1,2,0])
                content_val_np = self.content_info[i]['val'].detach().cpu().numpy()[0] + 0.5
                content_val_np = np.transpose(content_val_np, [1,2,0])
                plt.imshow(cur_val_np)
                plt.subplot(1,3,2)
                plt.imshow(content_val_np)
                plt.subplot(1,3,3)
                plt.imshow(corr_resp_np)
                plt.show()
            loss_tmp = 1-corr_resp
            total.append(loss_tmp.mean(dim=(2,3)))
            total_map.append(loss_tmp)
        total = torch.cat(total,dim=1)
        out = total.mean()
        total_map = vismodel.compute_masked_response(total_map,numband = numband)
        total_map = total_map.mean(dim=1).unsqueeze(1)
        return out, total_map