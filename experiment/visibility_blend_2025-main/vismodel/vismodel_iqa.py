import argparse
import sys
import cv2
import numpy as np
import seaborn as sns
import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from utils import inv_sigmoid, generalized_sigmoid, printImgCore
from .supermodels.customGDN import CustomGDN_NLP_Y
from .supermodels.visModel import VisModel
import copy
import pyiqa

import pycvvdp
from .eval.ciede2000 import deltaE_ciede2000_torch

eps = 1e-8

class VisModel_IQA(VisModel):
    
    def __init__(self, level,
                device,
                metric_name='lpips',
                sigmoid_type = 'generalized_sigmoid',
                target_type = "content",
                extraction_mode = "none",
                weight_mode = "none",
                sigmoid_param = None,
                mask_loss_weight = 0.0):
        super(VisModel_IQA, self).__init__(
            level = level,
            dims = 3,
            corr_ksize = 9,
            col_conversion = 'lab',
            no_mask = False,
            target_type = target_type,
            ovl_input = 'lowfreq',
            device = device)
        
        self.linear_map = nn.Parameter(torch.tensor(1.0))
        self.mask_loss_weight = mask_loss_weight

        if metric_name == 'ciede':

            self.map_available = True

            def delta_e(refimg, blendimg):
                ref_lab = self.convert_color_v1(refimg, 'lab')
                blend_lab = self.convert_color_v1(blendimg, 'lab')
                diff_map = (torch.sum((ref_lab - blend_lab) ** 2, dim=1)+1e-8) ** 0.5
                return diff_map

            self.iqa_metric = delta_e
        
        elif metric_name == 'ciede2000':

            self.map_available = True

            def delta_e(refimg, blendimg):
                ref_lab = self.convert_color_v1(refimg, 'lab') * 100
                blend_lab = self.convert_color_v1(blendimg, 'lab') * 100

                delta_e_map = deltaE_ciede2000_torch(ref_lab.permute(0, 2, 3, 1), blend_lab.permute(0, 2, 3, 1))

                return delta_e_map

            self.iqa_metric = delta_e

        elif metric_name == 'cvvdp':

            self.map_available = True

            self.cvvdp_metric = pycvvdp.cvvdp(display_name='standard_fhd', heatmap='raw', device=torch.device(device))
            #  "standard_fhd": {
            #                     "name": "24-inch FullHD monitor, peak luminance 200 cd/m^2, viewed under office light levels (250 lux), seen from 2 x display height", 
            #                     "resolution": [1920, 1080], 
            #                     "viewing_distance_meters":  0.6,  
            #                     "diagonal_size_inches": 24,   
            #                     "max_luminance": 200,   
            #                     "contrast": 1000,
            #                     "E_ambient": 250,
            #                     "source": "none" },

            # pixels per degree = 38 pix/deg
            # To do: input a more accurate setting (i.e., 32 pixels per degree)

            disp_geo = pycvvdp.vvdp_display_geometry((1920, 1080), diagonal_size_inches=24, distance_m=0.51)
            self.cvvdp_metric.set_display_model(display_name='standard_fhd', display_geometry=disp_geo)
            print("current ppd:", disp_geo.get_ppd())

            def metric(refimg, blendimg):
                num_batches = refimg.shape[0]
                vismap_list = []
                for bb in range(num_batches):
                    ref_rgb = refimg[bb:bb+1, [2, 1, 0], :, :]
                    blend_rgb = blendimg[bb:bb+1, [2, 1, 0], :, :]
                    Q_JOD, stats = self.cvvdp_metric.predict_map( ref_rgb, blend_rgb, dim_order="BCHW" )
                    #return 10.0 - Q_JOD

                    vismap_list.append(stats['heatmap'])

                vismaps = torch.cat(vismap_list, dim=0)

                return vismaps

            self.iqa_metric = metric
        elif metric_name == 'lpips':

            self.map_available = True

            self._iqa_metric = pyiqa.create_metric(metric_name, device=device, spatial=True, as_loss=True, loss_reduction='none')

            def metric(refimg, blendimg):
                ref_rgb = refimg[:, [2, 1, 0], :, :]
                blend_rgb = blendimg[:, [2, 1, 0], :, :]
                vis_map = self._iqa_metric(ref_rgb, blend_rgb)

                # vis_score = vis_map.mean()
                # vis_score = torch.sum(vis_map)/self.dilated_mask_gp_sum[0]
                
                return vis_map
            
            self.iqa_metric = metric
        else:

            self.map_available = False

            self._iqa_metric = pyiqa.create_metric(metric_name, device=device, as_loss=True)

            def metric(refimg, blendimg):
                ref_rgb = refimg[:, [2, 1, 0], :, :]
                blend_rgb = blendimg[:, [2, 1, 0], :, :]
                vis_score = self._iqa_metric(ref_rgb, blend_rgb)
                
                return vis_score
            
            self.iqa_metric = metric


        
        self.sigmoid_type = sigmoid_type
        if sigmoid_param == None:
            self.sigmoid_param: list[float] = [1.,1.,1.]
        else:
            self.sigmoid_param = sigmoid_param

        self._set_target_type(target_type)
        self._set_weight_mode(weight_mode)
        self._set_extraction_mode(extraction_mode)

        self.reset_state()

    def _set_weight_mode(self, weight_mode):
        assert weight_mode in ["3-way", "2-way", "2-way-extract", "3-way-extract", "original", "none"]
        self.weight_mode = weight_mode
    
    def _set_extraction_mode(self, extraction_mode):
        assert extraction_mode in ["normal", "partial", "partial-precise", "lowpass", "none"]
        self.extraction_mode = extraction_mode

    ######### Mains
    def compute_visibility(self):
        assert self.target != None and self.ref != None and self.blend != None and self.mask != None
        # Tensor inputs, img_tensor_x/y: (N, 3, H, W), RGB, 0 ~ 1
        # ref_rgb = self.ref[:, [2, 1, 0], :, :]
        # blend_rgb = self.blend[:, [2, 1, 0], :, :]
        # self.vis_score = self.iqa_metric(ref_rgb, blend_rgb)
        # if self.vis_score.ndim == 0:
        #     self.vis_score = [self.vis_score]

        with torch.no_grad():
            _vis_map = self.iqa_metric(self.ref, self.blend)

        scale_val  = F.relu(self.linear_map)

        self.vis_map = scale_val * _vis_map

        if self.map_available:
            self.vis_map = self.vis_map.view(-1,1,self.vis_map.shape[-2],self.vis_map.shape[-1])
            self.norm_vismap = self.visibility_to_norm(self.vis_map)

            score = torch.sum(self.vis_map, dim=(1,2,3))/self.dilated_mask_gp_sum[0]
        else:
            score = self.vis_map

        self.raw_score = score
        
        self.vis_score = score
        self.norm_score = self.visibility_to_norm(score)

        if self.vis_score.ndim == 0:
            self.vis_score = [self.vis_score]

        
    
    def compute_visibility_wo_weight(self):
        assert self.target != None and self.ref != None and self.blend != None and self.mask != None
        self.compute_visibility()
        
    def compute_weights(self):
        return
    
    def compute_vis_resp_alpha(self):
        return
    
    def compute_aggregate_resps(self):
        return

    ######### Utils
    def visibility_to_norm(self, vis):
        # return generalized_sigmoid(vis, self.param_fullmodel)
        
        if self.sigmoid_type == 'linear':
            _slope = torch.sigmoid(self.vis_slope)
            _vis_limit = 1.0/_slope
            # _vis = torch.clamp(vis, 0, _vis_limit.item())
            _vis = torch.min(vis, _vis_limit)
            return _vis*_slope
        
        elif self.sigmoid_type == "generalized_sigmoid":
            # (X, param_A, param_B, param_v, min_val=1, max_val=5)
            
            A = np.log(np.exp(self.sigmoid_param[0]) + 1) # force_positive
            B = np.log(np.exp(self.sigmoid_param[1]) + 1) # force_positive
            v = np.log(np.exp(self.sigmoid_param[2]) + 1) # force_positive

            Q = ((1+A)/A)**v - 1
            
            y = -A + (1+A)/((1+Q*torch.exp(-B*vis.view(-1,1)))**(1/v))
            return y.view(vis.shape)

    def visibility_to_rawscale(self, vis, mask = True):
        # if mask:
        #     vis = vis * self.dilated_mask_gp[0]
        # return inv_sigmoid(vis, self.param_fullmodel)
        if mask:
            # msk, msksum = self.get_mask_for_maskloss()
            vis = vis * self.dilated_mask_gp[0]
        
        if self.sigmoid_type == 'linear':
            _slope = torch.sigmoid(self.vis_slope)
            return vis / _slope
        
        elif self.sigmoid_type == 'generalized_sigmoid':
            A = np.log(np.exp(self.sigmoid_param[0]) + 1) # force_positive
            B = np.log(np.exp(self.sigmoid_param[1]) + 1) # force_positive
            v = np.log(np.exp(self.sigmoid_param[2]) + 1) # force_positive

            Q = ((1+A)/A)**v - 1
            _vis = torch.clamp(vis,min=0,max=0.99999)
            x = -torch.log((((1+A)/(_vis+A))**v - 1)/Q)/B
            return x
    
    def save_img(self, out_path, only_image = False):
        print("save_img function not implemented")

    def get_name(self):
        return 'VisModel_IQA'
    
    def projection(self):
        return
    
    def visualize_weights(self, showplot=False):
        return
           
    def get_params(self):
        param={}
        
        return param
