from __future__ import annotations
import torch
from .models.IBlender import IBlender
from .models.standard import standardBlender
from .models.network import networkBlender
from networks.utils import load_network, NETWORK_TYPE_LIST
from .models.crossDissolve import crossDissolveColorBlender, crossDissolveContrastBlender, crossDissolveSaliencyBlender, crossDissolveSaliencyMultiBlender, crossDissolveContrastMultiBlender, DEFAULT_PYR_LEVELS
from .models.fukiage2014 import fukiage2014Blender, DEFAULT_BLUR
from .models.visibility import visibilityBlender
# from .models.visibility_blur import visibilityBlurBlender
from vismodel.utils import load_vismodel, load_fukiage2014_model
from .models.sandor import sandorBlender

def load_blender(blender_dict: dict, device: torch.device, save_only_img: bool = False) -> IBlender:
    # save_only_img defines will save_imgs() save not only results images but also other image
    blender: IBlender | None = None
    type: str = blender_dict['type']
    shortname: str = blender_dict.get('shortname', "Nameless")
    target_type: str = blender_dict['target_type']
    print(f'load: {type} as {shortname}')
    print(f'target_type: {target_type}')

    if type == 'standard':
        blender = standardBlender(target_type,
                                  blender_dict.get('apply_sigmoid', False),
                                  blender_dict.get('resize', 1.0),
                                  blender_dict.get('input_size', None))

    elif type in NETWORK_TYPE_LIST:
        netModel = load_network(blender_dict, device)
        blender = networkBlender(type, 
                                 netModel,
                                 blender_dict.get('resize', 1.0),
                                 blender_dict.get('input_size', None),
                                 target_type=target_type)
    
    elif type == 'CrossDissolveContrast':
        blender = crossDissolveContrastBlender(target_type)
    elif type == 'CrossDissolveContrastMulti':
        blender = crossDissolveContrastMultiBlender(target_type, blender_dict.get('pyramid_levels', DEFAULT_PYR_LEVELS))
        
    elif type == 'CrossDissolveColor':
        blender = crossDissolveColorBlender(target_type)

    elif type == 'CrossDissolveSaliency':
        blender = crossDissolveSaliencyBlender(target_type,save_only_img)
    elif type == 'CrossDissolveSaliencyMulti':
        blender = crossDissolveSaliencyMultiBlender(target_type, save_only_img, blender_dict.get('pyramid_levels', DEFAULT_PYR_LEVELS))

    elif type == 'fukiage2014':
        fukiage2014_model = load_fukiage2014_model(device)
        blender = fukiage2014Blender(fukiage2014_model, blender_dict.get('blursize', DEFAULT_BLUR), target_type)

    elif type == 'sandor':
        blender = sandorBlender(device, target_type, blender_dict.get('use_motion', False), apply_sigmoid=blender_dict.get('apply_sigmoid', False))

    elif type == 'blur':

        model_type: str = blender_dict['model_type']

        if model_type in NETWORK_TYPE_LIST:
            blender_dict["type"] = model_type
            netModel = load_network(blender_dict, device, blur_mode=True)
            blender = networkBlender(model_type, 
                                    netModel,
                                    blender_dict.get('resize', 1.0),
                                    blender_dict.get('input_size', None),
                                    blur_mode=True)
        else:
            raise ValueError(f"Invalid model type: {model_type}")
            # vismodel = load_vismodel(model_type, device)
            # blender = visibilityBlurBlender(vismodel, blender_dict, target_type, device, save_only_img)
    else:
        vismodel = load_vismodel(type, device)
        blender = visibilityBlender(vismodel, blender_dict, target_type, device, save_only_img)
    print(f'')
    return blender

def load_blenders(blenders_list: list[dict], device: torch.device, save_only_img: bool = False) -> list[dict]:
    # add ('blender': IBlender) in each dict in blenders_list
    for blender_dict in blenders_list:
        blender = load_blender(blender_dict, device, save_only_img)
        blender_dict['blender']=blender

    return blenders_list

