from __future__ import annotations
import os
import torch
import argparse
import json
import datetime
import time
import cv2
import numpy as np
from utils import load_video_stimulus, stimulusVideo, stimulus, bilinear_resize, shrink_and_center
from blender.models.IBlender import IBlender
from blender.loader import load_blenders

output_image = False

def resize_img(img,wsize,hsize):
    re_img = cv2.resize(img,(wsize,hsize))
    return re_img

def start() -> float:
    torch.cuda.synchronize()
    return time.time()

def elapsed(start_time: float, blender_name: str) -> float:
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    print(f"time taken to generate image by {blender_name}: {elapsed_time}")
    return elapsed_time

def blend_main(blender_dict: dict, stim: stimulus) -> IBlender:
    blender: IBlender = blender_dict["blender"]
    blender_name: str = blender_dict['shortname']

    blender.blend(stim)
    return blender

if __name__ == '__main__':

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
    default_blender_path = os.getcwd()+'/default_blenders.json'
    default_video_path = os.getcwd()+'/default_videos.json'
    default_data_path = os.getcwd()+'/'
    default_out_path = os.getcwd()+'/results/blend_videos/'
    default_input_path = os.getcwd()+'/inputs.json'

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--device', default='cuda:0', help='device name')
    parser.add_argument('--blender', default=default_blender_path, help='output directory')
    parser.add_argument('--video', default=default_video_path, help='output directory')
    parser.add_argument('--data', default=default_data_path, help='path to the data directory')
    parser.add_argument('--output', default=default_out_path, help='output directory')
    parser.add_argument('--input', default = default_input_path, help='file path to input json file')
    args = parser.parse_args()
    print(args)

    if not torch.cuda.is_available():
        args.device = "cpu"
    args.device: torch.device = torch.device(args.device)
    print("device name:",args.device)

    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)

    blender_json = open(args.blender, 'r')
    blender_json: dict = json.load(blender_json)

    video_json = open(args.video, 'r')
    video_dict: dict = json.load(video_json)
    input_filename = os.path.basename(args.input)
    stimlus_list: list[stimulus] = load_video_stimulus(video_dict['image'], args.data, args.device)
    input_filename = os.path.splitext(input_filename)[0] # split extention
    exp_path = f"{args.output}{input_filename}/{now.strftime('%y%m%d_%H%M')}/"
    os.makedirs(exp_path, exist_ok=True)
    with open(exp_path+'blender_info.json', 'w') as f:
        json.dump(blender_json['model'], f, indent=4)
    
    blenders_list: list[dict] = load_blenders(blender_json['model'], args.device, save_only_img = True)
        
    for stim in stimlus_list:
        input_index = stim.index
        identifier = 'combid'+str(input_index)+'/'
        os.makedirs(exp_path+identifier, exist_ok=True)
        with open(exp_path+identifier+'input_info.json', 'w') as f:
            json.dump(video_dict['image'][input_index], f, indent=4)

        ## materials
        bgFilename = exp_path+identifier+'material_bg.mov'
        bg_video: cv2.VideoWriter = cv2.VideoWriter(bgFilename,fourcc, stim.fps, (stim.width,stim.height))

        ovlFilename = exp_path+identifier + 'material_ovl.mov'
        ovl_video: cv2.VideoWriter = cv2.VideoWriter(ovlFilename,fourcc, stim.fps, (stim.width,stim.height))

        for blender_dict in blenders_list:
            blendFilename = exp_path+identifier + blender_dict['shortname']+'_blend.mov'
            blender_dict['blend_video']: cv2.VideoWriter = cv2.VideoWriter(blendFilename,fourcc, stim.fps, (stim.width,stim.height))
            
            alphaFilename = exp_path+identifier + blender_dict['shortname']+'_alpha.mov'
            blender_dict['alpha_video']: cv2.VideoWriter = cv2.VideoWriter(alphaFilename,fourcc, stim.fps, (stim.width,stim.height))
        
        for frame_id in range(stim.num_frame):
            print(frame_id+1,'of',stim.num_frame)
            img_stim = stimulus()

            # mask video to tensor
            if stim.mask_video == None:
                mask_frame = torch.ones((1,1,stim.height,stim.width),dtype=torch.float32,device=args.device)
                # img_stim.set_mask()
            else:
                if type(stim.mask_video) == torch.Tensor:
                    mask_frame = stim.mask_video
                    mask_frame = bilinear_resize(mask_frame, stim.height, stim.width)
                    # img_stim.set_mask(mask_frame)
                else: 
                    _num_frames = stim.mask_video.get(cv2.CAP_PROP_FRAME_COUNT)
                    stim.mask_video.set(cv2.CAP_PROP_POS_FRAMES, frame_id%_num_frames)
                    ret, mask_frame = stim.mask_video.read()
                    if ret:
                        mask_frame = resize_img(mask_frame,stim.width,stim.height)
                        mask_frame = torch.permute(torch.as_tensor(mask_frame, dtype=torch.float32, device=args.device)[:,:,:1]/255, (2, 0, 1)).unsqueeze(0) 
                        # img_stim.set_mask(mask_frame)
            
            mask_frame = shrink_and_center(mask_frame, stim.shrink)
            img_stim.set_mask(mask_frame)
            
            # vismap video to tensor
            if stim.vismap_video == None:
                vismap_frame = torch.ones((1,1,stim.height,stim.width),dtype=torch.float32,device=args.device) * stim.obj_vis_value
                # img_stim.set_vismap()
            else:
                if type(stim.vismap_video) == torch.Tensor:
                    vismap_frame = stim.vismap_video 
                    vismap_frame = bilinear_resize(vismap_frame, stim.height, stim.width)
                else:
                    _num_frames = stim.vismap_video.get(cv2.CAP_PROP_FRAME_COUNT)
                    stim.vismap_video.set(cv2.CAP_PROP_POS_FRAMES, frame_id%_num_frames)
                    ret, vismap_frame = stim.vismap_video.read()
                    if ret:
                        vismap_frame = resize_img(vismap_frame,stim.width,stim.height)
                        vismap_frame = torch.permute(torch.as_tensor(vismap_frame, dtype=torch.float32, device=args.device)[:,:,:1]/255, (2, 0, 1)).unsqueeze(0) 

                if stim.obj_vis_value != None and stim.else_vis_value != None:
                    vismap_frame = vismap_frame * (stim.obj_vis_value - stim.else_vis_value) + stim.else_vis_value
                #     img_stim.set_vismap()
                # else:
                #     img_stim.set_vismap(vismap_frame)
            
            vismap_frame = shrink_and_center(vismap_frame, stim.shrink, boundary="replicate")
            img_stim.set_vismap(vismap_frame)
            
            # bg video to tensor
            _num_frames = stim.bg_video.get(cv2.CAP_PROP_FRAME_COUNT)
            stim.bg_video.set(cv2.CAP_PROP_POS_FRAMES, frame_id%_num_frames)
            ret, bg_frame = stim.bg_video.read()
            if stim.bg_flip:
                bg_frame = cv2.flip(bg_frame,1)
            if ret:
                bg_frame = resize_img(bg_frame,stim.width,stim.height)
                bg_frame = torch.permute(torch.as_tensor(bg_frame, dtype=torch.float32, device=args.device)/255, (2, 0, 1)).unsqueeze(0)
                img_stim.set_bg(bg_frame)
            
            # ovl video to tensor
            if type(stim.ovl_video) == torch.Tensor:
                content_color = stim.ovl_video[:,:3]
                content_alpha = stim.ovl_video[:,3:]
                content_color = shrink_and_center(content_color, stim.shrink)
                content_alpha = shrink_and_center(content_alpha, stim.shrink)
                ovl_frame = content_color * content_alpha + img_stim.bg * (1 - content_alpha)
            else:
                _num_frames = stim.ovl_video.get(cv2.CAP_PROP_FRAME_COUNT)
                stim.ovl_video.set(cv2.CAP_PROP_POS_FRAMES, frame_id%_num_frames)
                ret, ovl_frame = stim.ovl_video.read()
                if ret:
                    ovl_frame = resize_img(ovl_frame,stim.width,stim.height)
                    ovl_frame = torch.permute(torch.as_tensor(ovl_frame, dtype=torch.float32, device=args.device)/255, (2, 0, 1)).unsqueeze(0) 
                    content_color = ovl_frame
                    content_alpha = torch.ones((1,1,ovl_frame.shape[2],ovl_frame.shape[3]), dtype=torch.float32, device=args.device)
                    content_color = shrink_and_center(content_color, stim.shrink)
                    content_alpha = shrink_and_center(content_alpha, stim.shrink)
                    ovl_frame = content_color * content_alpha + img_stim.bg * (1 - content_alpha)
            img_stim.set_content_color(content_color)
            img_stim.set_content_alpha(content_alpha)
            img_stim.set_ovl(ovl_frame)

            # write material
            bg_out = bg_frame[0].detach().cpu().numpy()
            bg_out = np.transpose(bg_out, [1,2,0]) * 255
            bg_video.write(np.uint8(bg_out))

            ovl_out = ovl_frame[0].detach().cpu().numpy()
            ovl_out = np.transpose(ovl_out, [1,2,0]) * 255
            ovl_video.write(np.uint8(ovl_out))

            for blender_idx, blender_dict in enumerate(blenders_list):
                blender = blend_main(blender_dict, img_stim)

                #Write Process
                alpha_out = blender.alphamap
                # if alpha_out != None:
                if type(alpha_out) == torch.Tensor:
                    alpha_out = alpha_out[0].detach().cpu().numpy()
                    alpha_out = np.transpose(alpha_out, [1,2,0])
                    if alpha_out.shape[2] == 1:
                        alpha_out = np.concatenate([alpha_out,alpha_out,alpha_out],2)
                
                blend_out = blender.blendimg
                if type(blend_out) == torch.Tensor:
                    blend_out = blend_out[0].detach().cpu().numpy()
                    blend_out = np.transpose(blend_out, [1,2,0]) * 255
                
                # (org_width, org_height) = stim.get_original_size()
                # if (org_width, org_height) != (stim.width, stim.height):
                #     alpha_out = cv2.resize(alpha_out,(org_height,org_width))
                #     blend_out = (1.0-alpha_out) * bg_frame + alpha_out * ovl_frame
                
                blender_dict['blend_video'].write(np.uint8(blend_out))
                blender_dict['alpha_video'].write(np.uint8(255*alpha_out))

        bg_video.release()
        ovl_video.release()
        for blender_dict in blenders_list:
            blender_dict['blend_video'].release()
            blender_dict['alpha_video'].release()
