from __future__ import annotations
import torch
from torch import nn
import sys
import os
import json
from .models.adaptation_net import AdaptationNet
from .models.alpha_net import AlphaGeneratorLight
from .models.matting_net import MattingNetBase
from .models.deeplabv3 import MobileV2DeeplabV3Net, Res101DeeplabV3Net, MobileV2DeeplabV3Net_Scalar, Res101DeeplabV3Net_Scalar
from .models.TestNet import TestNet
from .models.INetwork import INetwork
from .models.scalaer_testnet import ScalarTestNet
from .models.OnnxNet import OnnxNet

NETWORK_TYPE_LIST = ['alphanet',
                     'adaptnet',
                     'mattingnet',
                     'deeplab',
                     'testnet',
                     'onnx',
                     'deeplabv3',
                     'deeplabv3_res',
                     'scalar_testnet',
                     'scalar_deeplab',
                     'scalar_deeplabv3_res']

def load_network_configs() -> dict[str,dict]:
    network_configs = open(f"{os.path.dirname(__file__)}/network_configs.json", 'r')
    network_configs = json.load(network_configs)
    return network_configs

def load_network_path(md: dict) -> str:
    path_dicts = load_network_configs()[md["type"]]
    if md["type"] in ['testnet', 'scalar_testnet']:
        path_dicts = path_dicts[str(md["mode"])]
    assert "load" in md.keys()
    load_name = md["load"]
    assert load_name in path_dicts.keys()
    print(f"load: {load_name}")
    return path_dicts[load_name]

def load_network(blender_dict: dict, device: torch.device, train: bool = False, blur_mode = False) -> INetwork:
    netModel: INetwork = None 
    if blender_dict['type'] == 'alphanet':
        netModel = AlphaGeneratorLight().to(device)
    elif blender_dict['type'] == 'adaptnet':
        netModel = AdaptationNet().to(device)
    elif blender_dict['type'] == 'mattingnet':
        netModel = MattingNetBase().to(device)
    elif blender_dict['type'] == 'deeplabv3':
        netModel = MobileV2DeeplabV3Net().to(device)
    elif blender_dict['type'] == 'testnet':
        netModel = TestNet(mode = blender_dict["mode"], blur_mode = blur_mode, base_alpha = blender_dict.get('base_alpha', 0.5)).to(device)
    elif blender_dict['type'] == 'deeplabv3_res':
        netModel = Res101DeeplabV3Net().to(device)
    elif blender_dict['type'] == 'scalar_testnet':
        netModel = ScalarTestNet(mode = blender_dict["mode"]).to(device)
    elif blender_dict['type'] == 'scalar_deeplab':
        netModel = MobileV2DeeplabV3Net_Scalar().to(device)
    elif blender_dict['type'] == 'scalar_deeplabv3_res':
        netModel = Res101DeeplabV3Net_Scalar().to(device)
    elif blender_dict['type'] == 'onnx':
        netModel = OnnxNet()
        

    else:
        sys.exit("Error: No definition of network")
    if train:
        return netModel.apply(train_weights_init)
    
    if blender_dict['type'] == 'onnx':
        netModel.load_model(f"{os.path.dirname(__file__)}/weights/{load_network_path(blender_dict)}")
    else:
        netModel.load_state_dict(torch.load(f"{os.path.dirname(__file__)}/weights/{load_network_path(blender_dict)}", map_location=device), strict=False)
    netModel.eval()
    return netModel

def train_weights_init(m: nn.Module):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        if classname.find('Conv2dNormActivation') != -1:
            return
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)