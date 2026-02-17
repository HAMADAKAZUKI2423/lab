from __future__ import annotations
import torch
import torch.nn.functional as F
import math
import numpy as np
import cv2
import pyiqa
import json
from .ILossFunction import ILossFunction
from ..supermodels.visModel import VisModel
from ..vismodel_mlp import VisModel_MLP
from matplotlib import pyplot as plt
from utils import save_img_torch, save_grayimg_plt

eps = 1e-8

class IqaLoss(ILossFunction):
    def __init__(self,
                 iqa_metric:str,
                 vis_lambda:float,
                 iqa_lambda:float,
                 use_norm_vis_for_loss: bool,
                 device: torch.device):
        self.iqa_metric = iqa_metric
        self.vis_lambda = vis_lambda
        self.iqa_lambda = iqa_lambda
        self.use_norm_vis_for_loss = use_norm_vis_for_loss
        self.device = device

        self.iqa = pyiqa.create_metric(iqa_metric, device=device, as_loss=True)

        self.reset_state()
        self.reset_loss()
    
    def reset_state(self):
        super().reset_state()
    
    def reset_loss(self):
        self.vis_loss: torch.Tensor | None  = None
        self.vis_loss_map: torch.Tensor | None  = None
        self.iqa_loss: torch.Tensor | None  = None

        self.vis_loss_list: list[float] = []
        self.iqa_loss_list: list[float] = []
        self.all_loss_list: list[float] = []
        self.loss_count: int = 0
    
    def compute_loss_preprocess(self, target_vis:torch.Tensor, vismodel:VisModel):
        vismodel.compute_weights()
        self.target_vis = target_vis
        if isinstance(vismodel, VisModel_MLP):
            self.target_vis = vismodel.get_downsampled_map(self.target_vis)
        self.target_vis_rawscale = vismodel.visibility_to_rawscale(self.target_vis)
    
    def compute_loss(self,
                      vismodel:VisModel,
                      alphamap:torch.Tensor, 
                      spatial_weight: torch.Tensor | None = None):
        if spatial_weight is not None:
            spatial_weight = spatial_weight * vismodel.dilated_mask_gp[0]
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

        if self.use_norm_vis_for_loss:
            # vis_loss_map = spatial_weight*torch.abs(self.vismodel.visibility_to_norm(vis_map) - self.tg_vis)
            vis_loss_map = spatial_weight*torch.abs(vismodel.norm_vismap - self.target_vis)
        else:
            vis_loss_map = spatial_weight*torch.abs(vis_map - self.target_vis_rawscale)
        visloss = vis_loss_map.sum(dim=(1,2,3))/spatial_weight_sum
        vis_loss = visloss.mean()

        if self.iqa.metric_mode == 'NR':
            if self.iqa.lower_better:
                iqa_loss: torch.Tensor = F.softplus(self.iqa(blend))
                #iqa_loss: torch.Tensor = torch.sigmoid(self.iqa(blend))
            else:
                iqa_loss: torch.Tensor = F.softplus(-1 * self.iqa(blend))
        elif self.iqa.metric_mode == 'FR':
            if self.iqa.lower_better:
                iqa_loss: torch.Tensor = F.softplus(self.iqa(blend, vismodel.target)) + F.softplus(self.iqa(blend, vismodel.ref))
                #iqa_loss: torch.Tensor = torch.sigmoid(self.iqa(blend))
            else:
                iqa_loss: torch.Tensor = F.softplus(-1 * self.iqa(blend, vismodel.target)) + F.softplus(-1 * self.iqa(blend, vismodel.ref))
       
        self.vis_loss = vis_loss 
        self.vis_loss_map = vis_loss_map 
        self.iqa_loss = iqa_loss 
        self.all_loss = self.vis_loss * self.vis_lambda+ self.iqa_loss * self.iqa_lambda

        self.vis_loss_list.append(self.vis_loss.item())
        self.iqa_loss_list.append(self.iqa_loss.item())
        self.all_loss_list.append(self.all_loss.item())
        self.loss_count += 1
    
    def save_loss(self, dir_path: str):
        assert self.all_loss_list != None
        epoch_list = list(range(len(self.all_loss_list)))

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(np.array(epoch_list),np.array(self.all_loss_list), color='black',  linestyle='solid', linewidth = 1.0, label='all')
        if self.vis_lambda>0:
            ax.plot(np.array(epoch_list),np.array(self.vis_loss_list), color='blue',  linestyle='solid', linewidth = 1.0, label='vis_loss')
        if self.iqa_lambda>0:
            ax.plot(np.array(epoch_list),np.array(self.iqa_loss_list), color='green',  linestyle='solid', linewidth = 1.0, label='iqa_loss')
        
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.legend()
        plt.savefig(f"{dir_path}visblend_lr.png")

        output_dic = {
                'vis_loss': self.vis_loss_list[-1] if self.vis_lambda>0 else 'none',
                'iqa_loss':self.iqa_loss_list[-1] if self.iqa_lambda>0 else 'none',
                'all_loss':self.all_loss_list[-1],
            }
        with open(f'{dir_path}output_info.json', 'w') as f:
            json.dump(output_dic, f, indent=4)
    
    def save_img(self, dir_path: str):
        pass

    def print_loss(self):
        message = f'loss: {sum(self.all_loss_list)/self.loss_count:.3f}, ' \
            f'vis_loss: {sum(self.vis_loss_list)/self.loss_count:.3f}, ' \
            f'iqa_loss: {sum(self.iqa_loss_list)/self.loss_count:.3f}, '
        self.reset_loss()
        return message

