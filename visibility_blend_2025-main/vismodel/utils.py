from __future__ import annotations
import torch
import json
import os
from .supermodels.visModel import VisModel
from .vismodel_mlp import VisModel_MLP
from .vismodel_iqa import VisModel_IQA
from .fukiage2014 import Fukiage2014

def load_fukiage2014_model(device: torch.device) -> Fukiage2014:
    model = Fukiage2014(6, device=device).to(device=device)
    model.load_state_dict(torch.load(f"{os.path.dirname(__file__)}/weights/fukiage2014.pth"))
    for name, param in model.named_parameters():
        print(name, "set grad: off")
        param.requires_grad = False
    return model

def load_vismodel(type: str, device: torch.device, load_param: bool = True) -> VisModel:
    vismodel_presets = open(f"{os.path.dirname(__file__)}/vismodel_configs.json", 'r')
    vismodel_presets: dict = json.load(vismodel_presets)
    vismodel_presets = vismodel_presets[type]

    vismodel: VisModel
    if vismodel_presets.get("MLP", False):
        vismodel = load_vismodel_mlp(vismodel_presets, device)
    elif vismodel_presets.get("IQA", False):
        vismodel = load_vismodel_iqa(vismodel_presets, device)
        # load_param = False
    else:
        raise RuntimeError(f"[load_vismodel] Unsupported vismodel preset type: '{type}'. Please check the configuration in vismodel_configs.json.")
        
    if load_param:
        if vismodel_presets['path']=='none':
            pass
        else:
            state_dict = torch.load(f"{os.path.dirname(__file__)}/weights/{vismodel_presets['path']}", map_location=device)
            incompatible = vismodel.load_state_dict(state_dict, strict=False)
            if incompatible.missing_keys:
                print("[load_vismodel] Warning: missing keys", incompatible.missing_keys)
            if incompatible.unexpected_keys:
                print("[load_vismodel] Warning: unexpected keys", incompatible.unexpected_keys)

            for name, param in vismodel.named_parameters():
                print(name, "set grad: off")
                param.requires_grad = False
    
    return vismodel

def load_vismodels(model_list:list[dict[str]], device: torch.device) -> list[dict[str]]:
    for md in model_list:
        model = load_vismodel(md['type'], device)
        print(f'')
        md['model']=model
    return model_list


def load_vismodel_iqa(presets: dict[str], device: torch.device) -> VisModel_IQA:
    return VisModel_IQA(level = 6,
                            device = device,
                            metric_name=presets.get("metric_name","lpips"),
                            sigmoid_type=presets.get("sigmoid_type", "linear"), 
                            sigmoid_param = presets.get("sigmoid_param",None)
                            ).to(device)

def load_vismodel_mlp(presets: dict[str], device: torch.device) -> VisModel_MLP:
    return VisModel_MLP(level = 6,
                        device = device,
                        corr_ksize=presets.get('corr_ksize',9),
                        weight_mode=presets.get('weight_mode','3-way'),
                        extraction_mode=presets.get('extraction_mode','none'),
                        sigmoid_type=presets.get('sigmoid_type','custom_sigmoid'),
                        # use_lowpass_diff_op_bl=presets.get('use_lowpass_diff_op_bl',False),
                        num_hidden_layer=presets.get('num_hidden_layer',2),
                        mlp_dim=presets.get('mlp_dim',32),
                        skip_dn=presets.get('skip_dn',False),
                        norm_mode=presets.get('norm_mode','none'),
                        no_mask=presets.get('no_mask',True),
                        nobound_opaque=presets.get('nobound_opaque',False),
                        fc_downsample_factor=presets.get('fc_downsample_factor',4),
                        drop_out_rate=presets.get('drop_out_rate',0.0),
                        lp_norm=presets.get('lp_norm',1),
                        mask_loss_weight=presets.get('mask_loss_weight',1.0),
                        adaptive_max_vis=presets.get('adaptive_max_vis',False),
                        # bezier_fix_sat=presets.get('bezier_fix_sat',None),
                        aliasing_free_pooling=presets.get('aliasing_free_pooling',False),
                        sigmoid_param=presets.get('sigmoid_param',[]),
                        final_activation=presets.get('final_activation','softplus'),
                        # ignore_residual_for_spatial_weight=presets.get('ignore_residual_for_spatial_weight',False)
                        ).to(device)