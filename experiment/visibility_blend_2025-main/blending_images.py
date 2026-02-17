from __future__ import annotations
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import argparse
import json
import datetime
import time
from utils import stimulus, load_stimulus, save_img_torch, save_grayimg_plt
from blender.models.IBlender import IBlender
from blender.models.visibility import visibilityBlender
from blender.loader import load_blenders
from dataset.loader import load_blend_dataset
from vismodel.supermodels.visModel import VisModel
from vismodel.utils import load_vismodel

def _is_cuda_device(device: torch.device | str | None) -> bool:
    if device is None:
        return False
    dev = torch.device(device)
    return dev.type == "cuda"


def start(device: torch.device | str | None) -> float:
    if _is_cuda_device(device):
        torch.cuda.synchronize()
    return time.time()


def elapsed(start_time: float, blender_name: str, device: torch.device | str | None) -> float:
    if _is_cuda_device(device):
        torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    print(f"time taken to generate image by {blender_name}: {elapsed_time}")
    return elapsed_time

def blend_main(blender_dict: dict, stim: stimulus, out_path: str, check_vismodel: VisModel | None, device: torch.device):
    blender: IBlender = blender_dict["blender"]
    blender_name: str = blender_dict['shortname']
    check_vismodel._set_target_type(blender_dict["target_type"])

    start_time = start(device)
    blender.blend(stim)
    elapsed_time = elapsed(start_time, blender_name, device)
    blender.save_imgs(out_path)

    if check_vismodel != None:# and not isinstance(blender, visibilityBlender):
        check_vismodel.set_inputs_bg_ovl_contents_blended(stim.bg, stim.ovl, stim.content_color, stim.content_alpha, blender.blendimg, stim.mask)
        save_img_torch(f'{out_path}input_ovl.png', check_vismodel.get_overlaid())
        check_vismodel.compute_visibility()
        gray_imgs = {
            "vismap":check_vismodel.norm_vismap,
            "vismap_rawscale":check_vismodel.vis_map
        }
        if check_vismodel.spatial_weight is not None:
            gray_imgs["spatial_weight"] = check_vismodel.spatial_weight.unsqueeze(1)
        norm_imgs = ["vismap"]
        for key, img in gray_imgs.items():
            if key in norm_imgs:
                save_grayimg_plt(f'{out_path}{key}_plt_check.png', torch.clip(img, 0, 1), norm = True)
                save_img_torch(f'{out_path}{key}_check.png', torch.clip(img, 0, 1))
            else:
                save_grayimg_plt(f'{out_path}{key}_plt_check.png', img, norm = False)

    if os.path.isdir(out_path):
        with open(save_path+'result_info.json', 'w') as f:
            json.dump({
                'model':blender_name,
                'type':blender_dict['type'],
                'time':elapsed_time
            }, f, indent=4)
            

if __name__ == '__main__':
    

    default_blender_path = os.getcwd()+'/settings_blender/settings_blenders_default.json'
    default_image_path = os.getcwd()+'/default_images.json'
    default_data_path = os.getcwd()+'/'
    default_out_path = os.getcwd()+'/results/blend_images/'

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--device', default='cuda:0', help='device name')
    parser.add_argument('-b','--blender', default=default_blender_path, help='output directory')
    parser.add_argument('-i','--image', default=default_image_path, help='output directory')
    parser.add_argument('-d','--data_path', default=default_data_path, help='output directory')
    parser.add_argument('-o','--output', default=default_out_path, help='output directory')
    parser.add_argument('--not_check', action='store_true', help='dont check vismap')

    parser.add_argument('--exp', action='store_true', help='use exp dataset')
    parser.add_argument('--exptrans', action='store_true', help='use exp trans dataset')
    parser.add_argument('--exptranstgbg', action='store_true', help='use exp trans dataset')
    parser.add_argument('--test', action='store_true', help='use test dataset')
    parser.add_argument('--expalpha', action='store_true', help='use exp alpha dataset')
    parser.add_argument('--res', action='store_true', help='use resolution dataset')

    parser.add_argument('--cocodtd', action='store_true', help='use cocodtd dataset')
    parser.add_argument('--coco', action='store_true', help='use coco dataset')
    parser.add_argument('--size', type=int, default=256)
    args = parser.parse_args()
    # args.exp=True
    print(args)

    if not torch.cuda.is_available():
        args.device = "cpu"
    args.device: torch.device  = torch.device(args.device)
    print("device name:",args.device)
    
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)

    blender_json = open(args.blender, 'r')
    blender_json: dict = json.load(blender_json)

    blend_dataset = load_blend_dataset(args, 
                                       args.data_path, 
                                       args.device,
                                       {"size":args.size})

    if blend_dataset != None:
        exp_path = f"{args.output}{blend_dataset.name}_{now.strftime('%y%m%d_%H%M')}/"
    else:
        image_json = open(args.image, 'r')
        image_json: dict = json.load(image_json)
        image_filename = os.path.basename(args.image)
        image_filename = os.path.splitext(image_filename)[0] # split extention
        blender_filename = os.path.basename(args.blender)
        blender_filename = os.path.splitext(blender_filename)[0] # split extention
        exp_path = f"{args.output}{image_filename}/{blender_filename}_{now.strftime('%y%m%d_%H%M')}/"

    os.makedirs(exp_path, exist_ok=True)
    with open(exp_path+'blender_info.json', 'w') as f:
        json.dump(blender_json['model'], f, indent=4)

    blenders_list: list[dict] = load_blenders(blender_json['model'], args.device, save_only_img = True)
    check_vismodel: None | VisModel = None
    if not args.not_check:
        if blender_json.get("check_vismodel",None) != None:
            check_vismodel = load_vismodel(blender_json["check_vismodel"], args.device)
        else:
            print(f"set check_vismodel")

    target_visibilities = ['-']
    if blend_dataset != None:
        #target_visibilities = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        #target_visibilities = [0,0.25,0.5,0.75, 1.0]
        # target_visibilities = [0,0.2,0.4,0.6,0.8,1.0]
        # target_visibilities = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
        target_visibilities = [0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
        # target_visibilities = [0.5]
        
    for vis_idx, visibility in enumerate(target_visibilities):
        if blend_dataset != None :
            stimlus_list = blend_dataset.load_dataset(visibility)
            os.makedirs(exp_path + f"original/", exist_ok=True)
            blend_dataset.save_dataset(exp_path + f"original/")
        else:
            stimlus_list = load_stimulus(image_json['image'], args.data_path, args.device)

        for ind, stim in enumerate(stimlus_list):
            assert stim.index != None 
            for blender_idx, blender_dict in enumerate(blenders_list):
                if blend_dataset != None:
                    save_dir = exp_path+ f"/{blender_dict['shortname']}/{stim.index[0]}/"
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = save_dir + stim.index[1]
                else:
                    index_dir = exp_path+'combid'+stim.index[0]+'/'
                    os.makedirs(index_dir, exist_ok=True)
                    save_path = index_dir+blender_dict['shortname']+'/'
                    os.makedirs(save_path, exist_ok=True)  
                    with open(index_dir+'input_info.json', 'w') as f:
                        json.dump(image_json['image'][ind], f, indent=4)

                print(f"{vis_idx}/{len(target_visibilities)}, {ind}/{len(stimlus_list)}, {blender_idx}/{len(blenders_list)}")
                blend_main(blender_dict, stim, save_path, check_vismodel, args.device)
    
    if blend_dataset != None:
        for blender_idx, blender_dict in enumerate(blenders_list):
            blend_dataset.tile_show(f"{exp_path}/{blender_dict['shortname']}", target_visibilities)