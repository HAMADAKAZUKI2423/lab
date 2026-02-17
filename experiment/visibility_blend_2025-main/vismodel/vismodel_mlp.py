from __future__ import annotations
import argparse
import sys
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from utils import printImgCore
from .supermodels.customGDN import CustomGDN_NLP_Y
from .supermodels.visModel import VisModel

eps = 1e-8


class GaussianBlurPool2d(nn.Module):
    def __init__(self, channels: int, offset, device):
        super(GaussianBlurPool2d, self).__init__()
        self.device = device
        self.channels = channels
        self.offset = offset

        base_kernel = torch.tensor([
            [1, 5, 8, 5, 1],
            [5, 25, 40, 25, 5],
            [8, 40, 64, 40, 8],
            [5, 25, 40, 25, 5],
            [1, 5, 8, 5, 1]], dtype=torch.float32, device=device)
        
        base_kernel /= 400.
        self.kernel = base_kernel.view(1, 1, 5, 5).expand(channels, 1, 5, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.offset:
            # 連続してdownsampleしていくと右下に重心がずれていくので，２回に１回sampleする場所をずらす
            return F.conv2d(F.pad(x, (1, 3, 1, 3), mode='reflect'), self.kernel, groups=x.shape[1], stride=2)
        else:
            return F.conv2d(F.pad(x, (2, 2, 2, 2), mode='reflect'), self.kernel, groups=x.shape[1], stride=2)
        # return F.conv2d(F.pad(x, (2, 2, 2, 2), mode='reflect'), self.kernel, groups=x.shape[1], stride=2)

class PositiveLinearLayer(nn.Module):
    def __init__(self, in_channels):
        super(PositiveLinearLayer, self).__init__()
        self.in_channels = in_channels
        # Initialize weights as a learnable parameter with all ones
        self.weights = nn.Parameter(torch.ones(in_channels))

    def forward(self, x):
        # Ensure weights are positive using ReLU
        positive_weights = F.relu(self.weights)
        positive_weights = positive_weights.view(1, -1, 1, 1)
        # Weighted sum along the channel dimension
        output = torch.sum(x * positive_weights, dim=1, keepdim=True)
        return output

class Square(nn.Module):
    def forward(self, x):
        return torch.square(x)
    
class VisModel_MLP(VisModel):
    
    def __init__(self, level,
                device,
                ksize = 5,
                corr_ksize = 9,
                weight_mode = "original",
                target_type = "content",
                ovl_input = "lowfreq",
                extraction_mode = "none",
                sigmoid_type = 'custom_sigmoid',
                # use_lowpass_diff_op_bl = False,
                num_hidden_layer=2,
                mlp_dim=32,
                skip_dn = False,
                norm_mode = "none",
                no_mask=False,
                nobound_opaque=False,
                fc_downsample_factor=4,
                drop_out_rate=0.0,
                lp_norm=1,
                mask_loss_weight = 1.0,
                adaptive_max_vis = False,
                # bezier_fix_sat = None,
                aliasing_free_pooling=False,
                sigmoid_param=[],
                final_activation='relu',
                # ignore_residual_for_spatial_weight=False,
                ):
        super(VisModel_MLP, self).__init__(
            level = level,
            dims = 3,
            corr_ksize = corr_ksize,
            col_conversion = 'lab',
            no_mask = no_mask,
            target_type = target_type,
            ovl_input = ovl_input,
            device = device)

        self.aliasing_free_pooling = aliasing_free_pooling

        # self.no_extraction = no_extraction
        # target extractionを行わない．bg, fg, blendをconcatenateしてMLPにつっこむ
        # mlp_predict=Trueとする
        # take_vis_degradation=Trueの場合，blend-bg, fg-blendをconcatenateしてMLPにつっこむ

        self.fc_downsample_factor = fc_downsample_factor
        # no_extraction modeのとき，fc層でダウンサンプルを行う
        if isinstance(lp_norm, str) and lp_norm == 'trainable':
            self.lp_norm = nn.Parameter(torch.tensor(2.0))  # 文字列ならtrainableパラメータ
        elif isinstance(lp_norm, (int, float)):  # 数値ならそのまま代入
            self.lp_norm = lp_norm
        else:
            raise ValueError("lp_norm must be either 'trainable' or a numeric value")

        # vis scoreをlpnormでpooling

        
        # raw_visをMLP->reluで得た後，sigmoidでratingを予測するパスと，linear layerを経てmatchingを予測するパスに分ける
        # 理論上linear layerは無くても良いはずだが，学習を安定させるかもしれない
        self.lin_map_to_match = nn.Linear(1,1)

        
        # self.use_simple_sigmoid = use_simple_sigmoid
        # generalized logisticではなくtorch.sigmoidを使う
        self.sigmoid_type = sigmoid_type
        # 'tanh' or 'sigmoid' or 'custom_sigmoid'
        # 'custom_sigmoid', 'tanh'の場合，raw visibilityはreluして出力.

        self.skip_dn = skip_dn
        # GDNを使わず，batchnormを使う

        self.no_mask = no_mask
        # responseに対してmaskを適用しない
        # MLP modelにおいて，何もないところでの反応を抑制するよう学習してほしいため

        self.mask_loss_weight = mask_loss_weight
        # mask外の領域の視認性マップ値が0になるように学習させる
        # rating taskでのみ使用される
        # no_mask=Trueでないと意味をなさない

        self.nobound_opaque = nobound_opaque
        # opaque target生成時に，maskを適用しない．

        self.norm_mode = norm_mode
        # skip_dn = Trueの場合に使うnormalizationの種類 "none", "bn"
        
        self._set_weight_mode(weight_mode)
        self._set_extraction_mode(extraction_mode)
        self.residual_fit = False
        # self.param_fullmodel = sigmoid_param

        # self.mlp_predict = mlp_predict
        # self.take_vis_degradation = take_vis_degradation # opaque targetからのresponseの減少分を計算してconcatenateする

        # self.train_sigmoid_param = train_sigmoid_param
        # if self.train_sigmoid_param and not self.use_simple_sigmoid:
        # self.param_fullmodel = nn.Parameter(torch.tensor(sigmoid_param, dtype=torch.float32, device=device))

        self._set_target_type(target_type)

        # self.force_symmetric = force_symmetric
        # self.force_diagonal_positive = force_diagonal_positive

        # if self.mlp_predict:
        #     self.force_symmetric = False
        #     self.force_diagonal_positive = False


        self.precision = False
        self.correlation = False

        self.ignore_residual = False
        # self.contrast_energy_based_spatial_weighting = spatial_weighting
        # self.use_guided_upsampling = use_guided_upsampling
        # self.low_res_visibility = low_res_visibility
        
        # self.lev_cont = lev_cont #band-limited contrastのlocal mean
        # self.lev_filt=lev_filt #resp_alphaにおける背景のlocal mean
        
        # self.col_sigma = nn.Parameter(torch.tensor(0.1),requires_grad=False)#no longer used
        
        num_channels_lum = self.level-1
        num_channels_col = self.level-1

        self.reset_state()
        
        self.channel_list = [0,1,2]
        self.num_channels = [num_channels_lum, num_channels_col, num_channels_col]
        self.num_channels_all = num_channels_lum + num_channels_col*2
        self.num_freq_list = [self.level-1,self.level-1,self.level-1]

        self.num_channels_all+=3

        self.std_vector = nn.Parameter(torch.zeros(self.level,self.dims),requires_grad=False)
        # self.col_sigma.requires_grad=False

        # self.channel_exp = nn.Parameter(torch.tensor(1.0))
        # self.vis_exp = nn.Parameter(torch.tensor(2.0))
        # if self.mlp_predict or self.sigmoid_splitpath or self.sigmoid_first:
        #     self.scaling = 1.0
        # else:
        #     self.scaling = nn.Parameter(torch.tensor(2.0))
        
        # if self.use_simple_sigmoid:
        if self.sigmoid_type == 'linear':
            self.vis_slope = nn.Parameter(torch.tensor(-1.0))
        elif self.sigmoid_type == "generalized_sigmoid":
            self.sigmoid_param = sigmoid_param
        
        elif self.sigmoid_type == 'custom_sigmoid':
            self.sig_scale = nn.Parameter(torch.tensor(1.0))
            self.sig_shift = nn.Parameter(torch.tensor(0.0))
        elif self.sigmoid_type == 'custom_sigmoid_v2':
            self.sig_a = nn.Parameter(torch.tensor(1.0))
            self.sig_b = nn.Parameter(torch.tensor(0.0))
            self.sig_c = nn.Parameter(torch.tensor(1.0))
        elif self.sigmoid_type == 'tanh':
            self.lin_map = nn.Linear(1,1,bias=False)
        else:
            self.lin_map = nn.Linear(1,1)

        self.spatial_weight = None

        
        if self.sigmoid_type == 'custom_sigmoid_v2':
            self.adaptive_max_vis = False
        else:
            self.adaptive_max_vis = adaptive_max_vis
            if self.adaptive_max_vis:
                self.vis_max = nn.Parameter(torch.tensor(0.0))
                self.vis_max_func = nn.Sigmoid()
        

        if self.skip_dn:
            self.GDN_band = None

            if self.norm_mode == "bn":
                self.norm = nn.BatchNorm2d(num_in_channels)
            elif self.norm_mode == "none":
                self.norm = None
        else:
            self.GDN_band = CustomGDN_NLP_Y(self.dims, ksize, self.device, self.level)

        if self.weight_mode == "original":
            self.linear_map = nn.Parameter(torch.tensor(1.0))

            self.fc = nn.Sequential(
                nn.Conv2d(self.num_channels_all, self.num_channels_all, (1,1), stride = 1, padding =(0,0), bias=True),
                nn.Sigmoid()
            )

            self.channel_exp = nn.Parameter(torch.tensor(1.0))

        else:

            if self.weight_mode == "3-way" or self.weight_mode == "3-vis-fusion":
                num_in_channels = self.num_channels_all*3
                num_out_channels = 1
            elif self.weight_mode == "2-way" or self.weight_mode == "2-vis-fusion":
                num_in_channels = self.num_channels_all*2
                num_out_channels = 1
            elif self.weight_mode == "4-vis-fusion":
                num_in_channels = self.num_channels_all*4
                num_out_channels = 1
            
            
            layer_list = []
            for hi in range(num_hidden_layer+1):
                if hi == 0:
                    nic = num_in_channels
                else:
                    nic = mlp_dim
                if hi == num_hidden_layer:
                    noc = num_out_channels
                else:
                    noc = mlp_dim
                layer_list.append(nn.Conv2d(nic, noc, (1,1), stride = 1, padding =(0,0), bias=True))
                torch.nn.init.xavier_uniform_(layer_list[-1].weight)*2

                if hi != num_hidden_layer:
                    layer_list.append(nn.ReLU())
                    if drop_out_rate>0.0:
                        layer_list.append(nn.Dropout(p=drop_out_rate))
                    if self.fc_downsample_factor>1:
                        if self.aliasing_free_pooling:
                            num_pooling = int(np.log2(self.fc_downsample_factor))
                            for npi in range(num_pooling):
                                offset = npi%2==1
                                layer_list.append(GaussianBlurPool2d(noc, offset, device))

                            # num_pooling = int(np.log2(self.fc_downsample_factor))
                            # for npi in range(num_pooling):
                            #     layer_list.append(GaussianBlurPool2d(noc, device))
                        else:
                            layer_list.append(nn.AvgPool2d(self.fc_downsample_factor))
                
            # layer_list.append(nn.Softplus())
            if final_activation=='softplus':
                layer_list.append(nn.Softplus())
            elif final_activation=='relu':
                layer_list.append(nn.ReLU())
            elif final_activation=='gelu':
                layer_list.append(nn.GELU())
            elif final_activation=='silu':
                layer_list.append(nn.SiLU())
            elif final_activation=='elu':
                layer_list.append(nn.ELU())

            self.fc = nn.Sequential(*layer_list)

        self.num_hidden_layer = num_hidden_layer
    
        
    def _set_weight_mode(self, weight_mode):
        assert weight_mode in ["3-way", "2-way", "original", "2-vis-fusion", "3-vis-fusion", "4-vis-fusion", "none"]
        self.weight_mode = weight_mode
    
    def _set_extraction_mode(self, extraction_mode):
        assert extraction_mode in ["normal", "partial", "partial-precise", "lowpass", "none"]
        self.extraction_mode = extraction_mode
    
    def reset_state(self):
        super().reset_state()
        self.org_map_shape: torch.Size | None = None
        self.ds_map_shape: torch.Size | None = None
        self.mask_data: torch.Tensor | None = None
        self.mask_data_sum: torch.Tensor | None = None
        self.dilated_mask_data: torch.Tensor | None = None
        self.dilated_mask_data_sum: torch.Tensor | None = None
        
    def set_mask(self, mask):
        VisModel.set_mask(self, mask)

        self.mask_data = self.get_downsampled_map(self.mask_gp[0].clone())
        self.mask_data_sum = self.mask_data.sum(dim=(1,2,3))

        self.dilated_mask_data = self.get_downsampled_map(self.dilated_mask_gp[0].clone())
        self.dilated_mask_data_sum = self.dilated_mask_data.sum(dim=(1,2,3))

    
    def get_downsampled_map(self, org_map: torch.Tensor) -> torch.Tensor:
        self.org_map_shape = org_map.shape
        # vismapと同じサイズにダウンサンプルする
        if self.fc_downsample_factor>1:
            ds_map = org_map
            for i in range(self.num_hidden_layer):
                ds_map = F.interpolate(ds_map, scale_factor=1/self.fc_downsample_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        else:
            ds_map = org_map
        self.ds_map_shape = ds_map.shape
        return ds_map
    
    def get_upsampled_map(self, ds_map: torch.Tensor) -> torch.Tensor:
        assert self.org_map_shape != None
        # 元画像のサイズにupsampleする
        if self.fc_downsample_factor>1:
            us_map = ds_map
            for i in range(self.num_hidden_layer):
                us_map = F.interpolate(us_map, scale_factor=self.fc_downsample_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True)
            us_map = F.interpolate(us_map, size=[self.org_map_shape[2],self.org_map_shape[3]], mode='bilinear', align_corners=False)
        else:
            us_map = ds_map
        return us_map
    
    def set_opaque_blend(self, opaque_blend=None, blend_mode='linear'):

        assert self.target != None and self.ref != None
        # if self.target_type == "background":
        #     opaque_blend = self.blending(self.ref, self.target, self.mask, blend_mode)
        # elif self.target_type == "content":
        if opaque_blend is None:
            if self.nobound_opaque:
                opaque_blend = self.target.clone()
            else:
                if self.extraction_mode == "lowpass":
                    lowref = self.calc_lowfreq_img(self.ref)
                    opaque_blend = self.blending(self.target, lowref, self.mask, blend_mode)
                else:
                    opaque_blend = self.blending(self.target, self.ref, self.mask, blend_mode)
        # for debug
        # self.opaque_blend = opaque_blend
        # opaque_belndを表示
        # plt.imshow(opaque_blend[0].cpu().numpy().transpose(1,2,0), vmin=0, vmax=1)
        # plt.show()
        
        opaque_blend_img = self.convert_color_v1(opaque_blend, self.col_conversion)
        self.opaque_blend_pyr = self.gen_Lpyr(opaque_blend_img, level = self.level)

        if self.extraction_mode == "normal":
            self.opaque_pyr, _ = self.contrast_extraction(self.ref_pyr, self.opaque_blend_pyr, level = self.level)
        elif self.extraction_mode == "partial":
            self.opaque_pyr, _ = self.partial_correlation_extraction(self.target_pyr, self.ref_pyr, self.opaque_blend_pyr, level = self.level)
        elif self.extraction_mode == "partial-precise":
            self.opaque_pyr, _ = self.partial_correlation_extraction_precise(self.target_pyr, self.ref_pyr, self.opaque_blend_pyr, level = self.level)
        # elif self.extraction_mode == "lowpass":
            
        else:
            self.opaque_pyr = self.opaque_blend_pyr
        
        

    ######### Mains
    def compute_visibility(self):
        assert self.target != None and self.ref != None and self.blend != None and self.mask != None
        self.compute_weights()
        self.compute_vis_resp_alpha()
        self.compute_aggregate_resps()
    
    def compute_visibility_wo_weight(self):
        assert self.target != None and self.ref != None and self.blend != None and self.mask != None
        self.compute_vis_resp_alpha()
        self.compute_aggregate_resps()
    
    def compute_spatial_weights(self):
        assert self.target != None and self.ref != None and self.opaque_pyr != None and self.mask != None
        

        concat_pyr = self.compute_masked_response(self.opaque_pyr)
        concat_pyr = concat_pyr ** 2
        if self.ignore_residual:
            contrast_energy = concat_pyr.sum(dim=1)
        else:
            contrast_energy = concat_pyr[:,0:-3].sum(dim=1)
        
        self.spatial_weight = 1-torch.exp(-100.0*contrast_energy)
        self.spatial_weight_sum = self.spatial_weight.sum(dim=(1,2))
        
        # plt.imshow(contrast_energy[0].detach().cpu().numpy())
        # plt.show()

        # # spatial_weightを表示 範囲は0~1
        # plt.imshow(self.spatial_weight[0].detach().cpu().numpy(), vmin=0, vmax=1)
        # plt.show()
    

        
    def compute_weights(self, opaque_blend = None, blend_mode = 'linear'):

        with torch.no_grad():
            self.set_opaque_blend(opaque_blend, blend_mode)

            # if self.contrast_energy_based_spatial_weighting:
            #     self.compute_spatial_weights()

        self.weight_pyr = self.opaque_pyr
        if not self.skip_dn:
            self.weight_resp = self.GDN_band(self.weight_pyr, self.weight_pyr, self.std_vector)
        else:
            self.weight_resp = self.weight_pyr
        
        if self.weight_mode == "original":
            self.weight_resp = self.compute_masked_response(self.weight_resp)
            self.weight_map = self.fc(self.weight_resp)
        
    def compute_vis_resp_alpha(self):

        if self.weight_mode == "original":

            if self.extraction_mode == "normal":
                result_pyr, sub_pyr = self.contrast_extraction(self.ref_pyr, self.blend_pyr, self.level)
            elif self.extraction_mode == "partial":
                result_pyr, sub_pyr = self.partial_correlation_extraction(self.target_pyr, self.ref_pyr, self.blend_pyr, self.level)
            elif self.extraction_mode == "partial-precise":
                result_pyr, sub_pyr = self.partial_correlation_extraction_precise(self.target_pyr, self.ref_pyr, self.blend_pyr, self.level)
            else:
                print("compute_vis_resp_partial Error")
                sys.exit(1)
            
            self.extracted_pyr = result_pyr

            # self.result_pyr = result_pyr# stored for fidelity loss computation
            if not self.skip_dn:
                self.extracted_resp = self.GDN_band(result_pyr, self.blend_pyr, self.std_vector)
            else:
                self.extracted_resp = result_pyr


        else:
            if not self.skip_dn:
                self.ref_resp = self.GDN_band(self.ref_pyr, self.ref_pyr, self.std_vector)
                self.blend_resp = self.GDN_band(self.blend_pyr, self.blend_pyr, self.std_vector)
            else:
                self.ref_resp = self.ref_pyr
                self.blend_resp = self.blend_pyr
            
    
    def compute_aggregate_resps(self):

        if self.weight_mode == "original":
            ex_resp = self.compute_masked_response(self.extracted_resp)
            exp_resp = torch.pow(ex_resp + eps, self.channel_exp)
            self.weigheted_resp = exp_resp * self.weight_map
            self.aggregated_resp = torch.pow(self.weigheted_resp.sum(dim=1)+eps, 1/self.channel_exp)
        
        elif self.weight_mode == "2-vis-fusion" or self.weight_mode == "3-vis-fusion" or self.weight_mode == "4-vis-fusion":
            op_resp = self.compute_masked_response(self.weight_resp, no_abs=True)
            bg_resp = self.compute_masked_response(self.ref_resp, no_abs=True)
            bl_resp = self.compute_masked_response(self.blend_resp, no_abs=True)

            target_vis = torch.abs(bl_resp - bg_resp)
            ref_vis = torch.abs(bl_resp - op_resp)

            if self.weight_mode == "3-vis-fusion":
                cat_resp = torch.cat([op_resp, target_vis, ref_vis], dim=1)
            elif self.weight_mode == "4-vis-fusion":
                cat_resp = torch.cat([op_resp, target_vis, ref_vis, bg_resp], dim=1)
            else:
                cat_resp = torch.cat([target_vis, ref_vis], dim=1)

            if self.no_mask:
                self.aggregated_resp = self.fc(cat_resp)
            else:
                self.aggregated_resp = self.fc(cat_resp) * self.dilated_mask_data

            self.aggregated_resp = self.aggregated_resp.squeeze(1)

        else:

            op_resp = self.compute_masked_response(self.weight_resp, no_abs=True)
            bg_resp = self.compute_masked_response(self.ref_resp, no_abs=True)
            bl_resp = self.compute_masked_response(self.blend_resp, no_abs=True)
            
            if self.weight_mode == "2-way":
                cat_resp = torch.cat([op_resp, bl_resp - bg_resp], dim=1)
            else:
                cat_resp = torch.cat([op_resp, bg_resp, bl_resp], dim=1)

                if self.skip_dn:
                    if self.norm is not None:
                        cat_resp = self.norm(cat_resp)

            if self.no_mask:
                self.aggregated_resp = self.fc(cat_resp)
            else:
                self.aggregated_resp = self.fc(cat_resp) * self.dilated_mask_data

            self.aggregated_resp = self.aggregated_resp.squeeze(1)

            
        self.vis_map = self.aggregated_resp.unsqueeze(1)
        if self.weight_mode == "original":
            scale_val  = F.relu(self.linear_map)
            self.vis_map = self.vis_map * scale_val
        self.norm_vismap = self.visibility_to_norm(self.vis_map)

        # vis_mapを表示 shape:[b,1,h,w]
        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(self.vis_map[0,0,:,:].detach().cpu().numpy(), vmin=0, vmax=10)
        # vismap_raw = F.avg_pool2d(self.vis_map, 32, stride=1, padding=16)
        # vismap = self.visibility_to_norm(vismap_raw)
        # ax[1].imshow(vismap[0,0,:,:].detach().cpu().numpy(), vmin=0, vmax=1)
        # plt.show()
        
        if self.spatial_weight is not None:
            weighted_agg_resp = self.aggregated_resp * self.spatial_weight

            if isinstance(self.lp_norm, nn.Parameter):
                self.lp_norm.data = torch.clamp(self.lp_norm.data, min=0.5)
                score = torch.pow( torch.sum(weighted_agg_resp ** self.lp_norm, dim=(1,2))/self.spatial_weight_sum + eps, (1/self.lp_norm))
            else:
                if self.lp_norm>1:
                    score = torch.pow( torch.sum(weighted_agg_resp ** self.lp_norm, dim=(1,2))/self.spatial_weight_sum + eps, (1/self.lp_norm))
                else:
                    score = torch.sum(weighted_agg_resp, dim=(1,2))/self.spatial_weight_sum

        else:
            if isinstance(self.lp_norm, nn.Parameter):
                self.lp_norm.data = torch.clamp(self.lp_norm.data, min=0.5)
                score = torch.pow( torch.sum(self.aggregated_resp ** self.lp_norm, dim=(1,2))/self.dilated_mask_data_sum + eps, (1/self.lp_norm))
            else:
                if self.lp_norm>1:
                    score = torch.pow( torch.sum(self.aggregated_resp ** self.lp_norm, dim=(1,2))/self.dilated_mask_data_sum + eps, (1/self.lp_norm))
                else:
                    score = torch.sum(self.aggregated_resp, dim=(1,2))/self.dilated_mask_data_sum

        if self.weight_mode == "original":
            scale_val  = F.relu(self.linear_map)
            self.raw_score = score * scale_val

            self.vis_score = score * scale_val

        else:
            self.raw_score = score
            

            # self.vis_score = self.lin_map_to_match(score.view(-1,1)).view(score.shape)
            self.vis_score = score
        
        self.norm_score = self.visibility_to_norm(score)



    ######### Utils
    def adjust_norm_vis(self, vis):
        # adaptive vis max使用時に，target visibilityの値をratingのスケールに合うように補正する
        # visは[0,1]の範囲
        if self.adaptive_max_vis:
            
            return vis / (1 + self.vis_max_func(self.vis_max)/4)
        else:
            return vis
    
    
    def visibility_to_norm(self, vis):
        # if self.sigmoid_first:

        # if not force_compute:
        #     if len(vis.shape) > 2:
        #         if self.norm_vismap is not None:
        #             return self.norm_vismap
        #     else:
        #         if self.norm_score is not None:
        #             return self.norm_score
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
        
            
        elif self.sigmoid_type == 'custom_sigmoid':
            c = -torch.exp(self.sig_shift) # force negative
            y = c*(1-c) / (c-torch.exp( -self.sig_scale * vis.view(-1,1) )) + c
            return y.view(vis.shape)
        
        elif self.sigmoid_type == 'custom_sigmoid_v2':
            # no upper bound but forced to cross the origin
            a = F.softplus(self.sig_a) # force_positive
            b = self.sig_b
            c = -F.softplus(self.sig_c) # force negative
        
            y = (-c*(1+torch.exp(b))) / (1+torch.exp(-a*vis.view(-1,1)+b)) + c
            return y.view(vis.shape)
            
        elif self.sigmoid_type == 'tanh':
            return torch.tanh(self.lin_map(vis.view(-1,1))).view(vis.shape)
        else:
            return torch.sigmoid(self.lin_map(vis.view(-1,1))).view(vis.shape)
        
    def visibility_to_rawscale(self, vis, mask = True):
        if mask:
            # msk, msksum = self.get_mask_for_maskloss()
            vis = vis * self.dilated_mask_data
        
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
        
        elif self.sigmoid_type == 'custom_sigmoid':
            c = -torch.exp(self.sig_shift) # force negative
            return (torch.log(vis-c) - torch.log(c*(vis-1)))/self.sig_scale
        elif self.sigmoid_type == 'custom_sigmoid_v2':
            a = F.softplus(self.sig_a)
            b = self.sig_b
            c = -F.softplus(self.sig_c)
            return (torch.log(vis-c) - torch.log(c*(vis-1)) + torch.log(1+torch.exp(-a*vis+b)) - torch.log(1+torch.exp(-a*(vis-1)+b)))/a

        elif self.sigmoid_type == 'tanh':
            return torch.atanh(vis) / self.lin_map.weight[0]
        else:
            return (torch.logit(vis) - self.lin_map.bias[0]) / self.lin_map.weight[0] 
    
    def save_img(self, out_path, only_image = False):
        tensors_dict = {
            "extracted_resp":self.extracted_resp,
            "weight_resp":self.weight_resp,
            "weight_map":self.weight_map,
            "weigheted_resp":self.weigheted_resp,
        }
        Lists_dict = {
            "blend_pyr":self.blend_pyr,
            "target_pyr":self.target_pyr,
            "ref_pyr":self.ref_pyr,
            "extracted_pyr":self.extracted_pyr,
            "weight_pyr":self.weight_pyr
        }
        images_dict = {
            "vis_map_rawscale":self.vis_map,
            "vis_map":self.norm_vismap
        }

        for k, v in tensors_dict.items():
            print(f'{k} printing')
            channels = v.shape[1]
            levels = channels//3
            label = ['l','a','b']
            for level in range(levels):
                for color in range(3):
                    channel = level * 3 + color
                    write_image = v[0,channel].detach().cpu().numpy()
                    printImgCore(write_image, f"{out_path}{k}_{str(level)}{label[color]}.png", onlyImg = only_image)

        for k, v in Lists_dict.items():
            print(f'{k} printing')
            levels = len(v)
            shape = (v[0].shape[3],v[0].shape[2])
            for level in range(levels):
                out_image = v[level][0].detach().cpu().numpy()
                out_image = np.transpose(out_image, [1,2,0])
                resized = cv2.resize(out_image, shape, interpolation=cv2.INTER_LINEAR)
                label = ['l','a','b']
                for channel in range(3):
                    write_image = resized[:,:,channel]
                    scale = None
                    printImgCore(write_image, f'{out_path}{k}_{str(level)}{label[channel]}.png', scale = scale, onlyImg = only_image)

        for k, v in images_dict.items():
            print(f'{k} printing')
            write_image = v[0,0].detach().cpu().numpy()
            scale = None
            if k in ['vis_map']:
                scale = (0,1)
            printImgCore(write_image, f'{out_path}{k}.png', scale = scale, onlyImg = only_image)
    
    def get_name(self):
        return 'VisModel_MLP'

    def projection(self):
        return 
    
    def visualize_weights(self, showplot=False):
        
        # print(self.std_vector)
        # print("col_sigma:", self.col_sigma.data)

        if not self.skip_dn:
            
            beta = ((self.GDN_band.beta.data)**2-self.GDN_band.pedestal.data).clone().cpu().numpy()#self.GDN_list_high[i].beta.data.clone().cpu().numpy()
            print('beta band:', beta)
            
            alpha = ((self.GDN_band.alpha.data)**2-self.GDN_band.pedestal.data).clone().cpu().numpy()
            print('alpha band:',alpha)

        if self.weight_mode == "original":
            print("channel_exp", self.channel_exp.data)
        # print("vis_exp", self.vis_exp.data)

        if self.adaptive_max_vis:
            print("max_vis", self.vis_max_func(self.vis_max.data))
        
        if self.sigmoid_type == 'custom_sigmoid_v2':
            # no upper bound but forced to cross the origin
            a = F.softplus(self.sig_a) # force_positive
            b = self.sig_b
            c = -F.softplus(self.sig_c) # force negative

            print("sigmoid a", a.data)
            print("sigmoid b", b.data)
            print("sigmoid c", c.data)

            if showplot:

                x = np.linspace(0, 10, 100)

                def custom_sigmoid(x, a, b, c):
                    
                    y = (-c*(1+np.exp(b))) / (1+np.exp(-a*x+b)) + c
                    return y

                y = custom_sigmoid(x, a.data.cpu().numpy(), b.data.cpu().numpy(), c.data.cpu().numpy())
                plt.plot(x, y)
                # y=0に線を引く
                plt.plot(x, np.zeros_like(x), linestyle="--", color="black")
                # y=1に線を引く
                plt.plot(x, np.ones_like(x), linestyle="--", color="black")
                # x=0に線を引く
                plt.plot(np.zeros_like(x), x, linestyle="--", color="black")

                plt.legend()
                # -1, -1にy軸を限定
                plt.ylim([0, 2])
                plt.show()

        
            # y = (-c*(1+torch.exp(b))) / (1+torch.exp(-a*vis.view(-1,1)+b)) + c

        
           
           
    def get_params(self):
        param={}

        if not self.skip_dn:
            beta = ((self.GDN_band.beta.data)**2-self.GDN_band.pedestal.data).clone().cpu().numpy()
            param['beta']=beta
            
            alpha = ((self.GDN_band.alpha.data)**2-self.GDN_band.pedestal.data).clone().cpu().numpy()
            param['alpha']=alpha

            dnfilt = (self.GDN_band.dn_filt.data**2-self.GDN_band.pedestal.data)
            dnfilt = (dnfilt/dnfilt.sum(dim=(2,3),keepdim=True)).clone().cpu().numpy()

            for i in range(dnfilt.shape[0]):
                name = 'dn_filt_'+str(i)
                param[name]=dnfilt[i]
        
        if self.weight_mode == "original":
            param['channel_exp']=self.channel_exp.data.clone().cpu().numpy()
    
        return param
