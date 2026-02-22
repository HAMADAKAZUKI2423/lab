from __future__ import annotations
import os
import torch
import argparse
import json
import datetime
import time
from utils import stimulus, load_stimulus, save_img_torch, save_grayimg_plt, load_img_torch
from blender.models.IBlender import IBlender
from blender.models.visibility import visibilityBlender
from blender.loader import load_blenders, load_blender
from dataset.loader import load_blend_dataset
from vismodel.supermodels.visModel import VisModel
from vismodel.utils import load_vismodel

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def start() -> float:
    torch.cuda.synchronize()
    return time.time()

def elapsed(start_time: float, blender_name: str) -> float:
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    print(f"time taken to generate image by {blender_name}: {elapsed_time}")
    return elapsed_time

def blend_main(blender_dict: dict, stim: stimulus, out_path: str, check_vismodel: VisModel | None = None):
    blender: IBlender = blender_dict["blender"]
    blender_name: str = blender_dict['shortname']

    start_time = start()
    blender.blend(stim)
    elapsed_time = elapsed(start_time,blender_name)
    blender.save_imgs(out_path)

    if check_vismodel != None and not isinstance(blender, visibilityBlender):
        check_vismodel.set_inputs_bg_ovl_blended(stim.bg, stim.ovl, blender.blendimg, stim.mask)
        check_vismodel.compute_visibility()
        gray_imgs = {
            "vismap":check_vismodel.norm_vismap,
            "vismap_rawscale":check_vismodel.vis_map
        }
        norm_imgs = ["vismap"]
        for key, img in gray_imgs.items():
            if key in norm_imgs:
                save_grayimg_plt(f'{out_path}{key}_plt.png', torch.clip(img, 0, 1), norm = True)
                save_img_torch(f'{out_path}{key}.png', torch.clip(img, 0, 1))
            else:
                save_grayimg_plt(f'{out_path}{key}_plt.png', img, norm = False)

    if os.path.isdir(out_path):
        with open(out_path+'result_info.json', 'w') as f:
            json.dump({
                'model':blender_name,
                'type':blender_dict['type'],
                'time':elapsed_time
            }, f, indent=4)
            

if __name__ == '__main__':
    default_data_path = os.getcwd()+'/'
    default_out_path = os.getcwd()+'/results/paper_luminance/'

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--device', default='cuda:0', help='device name')
    parser.add_argument('--data', default=default_data_path, help='path to the data directory')
    parser.add_argument('-o','--output', default=default_out_path, help='output directory')
    parser.add_argument('--not_check', action='store_true', help='dont check vismap')
    args = parser.parse_args()
    print(args)

    if not torch.cuda.is_available():
        args.device = "cpu"
    args.device: torch.device  = torch.device(args.device)
    print("device name:",args.device)

    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)

    exp_path = f"{args.output}{now.strftime('%y%m%d_%H%M')}/"
    os.makedirs(exp_path, exist_ok=True)

    alpha_model_dict = {
            "type":"standard",
            "shortname":"alphablend",
            "target_type":"content"
        }
    net_model_dict = {
            "type":"testnet",
            "mode":0,
            "load":"1225_501",
            "shortname":"testnet_1225_501",
            "target_type":"content"
        }
    with open(exp_path+'net_model_info.json', 'w') as f:
        json.dump(net_model_dict, f, indent=4)

    alpha_model_dict["blender"] = load_blender(alpha_model_dict,args.device)
    net_model_dict["blender"]  = load_blender(net_model_dict,args.device)

    fg = load_img_torch(args.data + "imgs/exp/test/stripes.png", args.device,"alpha")
    height = fg.shape[2]
    width = fg.shape[3]

    target_visibilities = [0.2,0.4,0.6]
    bg_initial_color = 0.5
    bg_luminance = [0,0.25,0.5,0.75,1.0]
    initial_alpha = []
    figsize = 3
    
    net_path = f"{exp_path}net/"
    os.makedirs(net_path, exist_ok=True)

    print(f"saving: vis")
    rows = len(bg_luminance)
    lines = len(target_visibilities)
    fig, ax = plt.subplots(rows, lines, figsize=(figsize*lines, figsize*rows))
    fig.subplots_adjust(hspace=0, wspace=0)
    for vis_idx, visibility in enumerate(target_visibilities):
        vismap = torch.ones((1,1,height,width),dtype=torch.float32,device=args.device)*visibility
        for luminance_idx, luminance in enumerate(bg_luminance):
            stim = stimulus()
            stim.set_bg(torch.ones_like(fg[:,:3],device=args.device, dtype=torch.float32) * luminance)
            stim.set_mask(torch.where(fg[:,3:] > 0,1.,0.).to(torch.float32))
            stim.set_content_color(fg[:,:3])
            stim.set_content_alpha(fg[:,3:])
            stim.set_ovl(stim.content_color * stim.content_alpha + stim.bg * (1 - stim.content_alpha))
            stim.set_vismap(vismap)
            
            save_name = net_path + 'vis'+str(int(visibility*100)).zfill(3) + '_bg' +str(int(luminance*100)).zfill(3) + '_'
            blend_main(net_model_dict, stim, save_name)
            
            if luminance_idx == 0:
                blender = net_model_dict["blender"]
                tmp = blender.alphamap * stim.mask 
                alpha_mean = torch.sum(tmp) / (torch.sum(stim.mask) * 3)
                print(alpha_mean)
                initial_alpha.append(alpha_mean.item())
            
            data = Image.open(net_path + 'vis'+str(int(visibility*100)).zfill(3) + '_bg' +str(int(luminance*100)).zfill(3) + '_blend.png')
            data = np.asarray(data)
            ax[luminance_idx, vis_idx].xaxis.set_major_locator(plt.NullLocator())
            ax[luminance_idx, vis_idx].yaxis.set_major_locator(plt.NullLocator())
            ax[luminance_idx, vis_idx].imshow(data)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(net_path + "arrange.png")
    plt.close()

    with open(exp_path+'initial_alpha.txt', 'w', encoding='utf-8') as f:
        for data in initial_alpha:
            f.write(f"{data}\n")
    
    alpha_path = f"{exp_path}alpha/"
    os.makedirs(alpha_path, exist_ok=True)

    print(f"saving: alpha")
    rows = len(bg_luminance)
    lines = len(target_visibilities)
    fig, ax = plt.subplots(rows, lines, figsize=(figsize*lines, figsize*rows))
    fig.subplots_adjust(hspace=0, wspace=0)
    for alpha_idx, alpha in enumerate(initial_alpha):
        vismap4alpha = torch.ones((1,1,height,width),dtype=torch.float32,device=args.device)*alpha
        for luminance_idx, luminance in enumerate(bg_luminance):
            stim = stimulus()
            stim.set_bg(torch.ones_like(fg[:,:3],device=args.device, dtype=torch.float32) * luminance)
            stim.set_mask(torch.where(fg[:,3:] > 0,1.,0.).to(torch.float32))
            stim.set_content_color(fg[:,:3])
            stim.set_content_alpha(fg[:,3:])
            stim.set_ovl(stim.content_color * stim.content_alpha + stim.bg * (1 - stim.content_alpha))
            stim.set_vismap(vismap4alpha)
            
            save_name = alpha_path + 'alpha'+str(int(alpha*100)).zfill(3) + '_bg' +str(int(luminance*100)).zfill(3) + '_'
            blend_main(alpha_model_dict, stim, save_name)

            data = Image.open(alpha_path + 'alpha'+str(int(alpha*100)).zfill(3) + '_bg' +str(int(luminance*100)).zfill(3) + '_blend.png')
            data = np.asarray(data)
            ax[luminance_idx, alpha_idx].xaxis.set_major_locator(plt.NullLocator())
            ax[luminance_idx, alpha_idx].yaxis.set_major_locator(plt.NullLocator())
            ax[luminance_idx, alpha_idx].imshow(data)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(alpha_path + "arrange.png")
    plt.close()