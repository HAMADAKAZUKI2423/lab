from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
import time
import re
import os
import cv2
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from vismodel.supermodels.visModel import VisModel
from vismodel.loss.loader import load_lossFunction
from vismodel.vismodel_describe import VisModel_Describe
from utils import nearest_resize
import numpy as np
import pandas as pd


class NetworkTrainer():
    def __init__(self,
                 network: nn.Module,
                 vismodel: VisModel,
                 config: dict[str],
                 lr: float,
                 vis_exp: float,
                 handle_tv_edge: bool,
                 output_dir: str,
                 device: torch.device,
                 train_size: int = 256):
        
        self.__net = network
        self.__vismodel = vismodel
        self.shortname = config["shortname"]
        self.lossF = load_lossFunction(config.get('loss_type','original'), config, device)
        self.__vismodel._set_target_type(config['target_type'])
        self.handle_tv_edge = handle_tv_edge
        self.output_dir = output_dir
        self.device = device
        self.train_size = train_size
        
        self.__net.train()

        if isinstance(self.__vismodel, VisModel_Describe) and vis_exp>0:
            self.__vismodel.vis_exp.data *= 0.0
            self.__vismodel.vis_exp.data += vis_exp
    
        self.optimizerG = optim.Adam(self.__net.parameters(), lr=lr, betas=(0.5, 0.999))
        self.schedulerG = optim.lr_scheduler.LambdaLR(self.optimizerG, self.__train_modify_learning_rate)

        self.base_resp = 'linear'
        self.linear_corr=True
        self.recon_corr=False

        self.upcorr = False
        self.bgr_corr = True
        self.training_start_time = time.time()

        #self.test_colums = []
        self.test_df = pd.DataFrame()

        os.makedirs(f'{self.output_dir}/{self.shortname}/', exist_ok=True)
        self.train_append_log(self.__net)

    def train_process(self, data: dict[str, torch.Tensor], phase: str="train"):
        assert phase in ["train","val","test"]
        ovl = data['fg'].to(self.device)
        bg = data['bg'].to(self.device)
        content_color = data['fg_color'].to(self.device)
        content_alpha = data['fg_alpha'].to(self.device)
        mask = data['alpha_mask'].to(self.device)
        target_vis = data['target_vis'].to(self.device)
        
        if len(target_vis.shape)>=4:
            target_vismap = target_vis
        else:
            target_vismap = target_vis.unsqueeze(-1).unsqueeze(-1).expand(
                target_vis.shape[0],target_vis.shape[1],mask.shape[2],mask.shape[3])

        if self.handle_tv_edge:
            #target visibility map中のエッジ付近では，視認性ロスの重みを小さくとった方が訓練が安定するかもしれない．．というアイデア
            self.vis_spat_weight = self.__tv_smooth_weight(target_vismap)
            if False:
                #視認性ロスの重みの確認用
                for i in range(self.vis_spat_weight.shape[0]):
                    target_vismap_np = self.target_vismap[i,0].cpu().detach().numpy()#.transpose([1,2,0])#np.asarray(trans_func.to_pil_image(new_fg))
                    cv2.imshow('target_vismap_np', target_vismap_np)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    visweight_np = self.vis_spat_weight[i,0].cpu().detach().numpy()#.transpose([1,2,0])#np.asarray(trans_func.to_pil_image(new_fg))
                    cv2.imshow('visweight_np', visweight_np)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        else:
            self.vis_spat_weight = None

        data_height = ovl.shape[2]
        data_width = ovl.shape[3]
        ovl_re = T.functional.resize(img=ovl, size=(self.train_size, self.train_size),antialias = True)
        bg_re = T.functional.resize(img=bg, size=(self.train_size, self.train_size),antialias = True)
        target_vis_re = T.functional.resize(img=target_vis, size=(self.train_size, self.train_size),antialias = True)

        #modelにはvisibilityのrangeを0-1として学習させる
        alphamap_re = self.__net(ovl_re, bg_re, target_vis_re)
        alphamap = T.functional.resize(img=alphamap_re, size=(data_height, data_width),antialias = True)

        ovl = ovl * 0.5 + 0.5
        bg = bg * 0.5 + 0.5
        content_color = content_color * 0.5 + 0.5

        self.__vismodel.set_inputs_bg_ovl_contents(bg, ovl, content_color, content_alpha, mask)
        self.lossF.compute_loss_preprocess(target_vismap, self.__vismodel)
        alphamap = alphamap.expand(-1,3,-1,-1) * self.__vismodel.dilated_mask_gp[0]

        # self.lossF.compute_loss(self.__vismodel, alphamap, self.vis_spat_weight)
        self.lossF.compute_loss(self.__vismodel, alphamap)
        self.train_alphamap = alphamap

        if phase == "train":
            self.optimizerG.zero_grad()
            self.lossF.all_loss.backward()
            self.optimizerG.step()
        
        elif phase == "test":
            tmp_df = pd.DataFrame()
            if len(target_vis.shape)>=4:
                tmp_df['target_vis'] = torch.mean(target_vis,(1,2,3)).cpu().detach().numpy()
            else:
                tmp_df['target_vis'] = torch.mean(target_vis,(1)).cpu().detach().numpy()
            tmp_df['alpha_mean'] = torch.mean(alphamap,(1,2,3)).cpu().detach().numpy()
            tmp_df['alpha_std'] = torch.std(alphamap,(1,2,3)).cpu().detach().numpy()
            tmp_df['pred_vis_mean'] = torch.mean(self.__vismodel.norm_vismap,(1,2,3)).cpu().detach().numpy()
            tmp_df['pred_vis_std'] = torch.std(self.__vismodel.norm_vismap,(1,2,3)).cpu().detach().numpy()
            self.test_df = pd.concat([self.test_df,tmp_df])
    
    def train_update_learning_rate(self):
        self.schedulerG.step()
        #self.schedulerD.step()
    
    def train_save(self, epoch: int):
        torch.save(self.__net.state_dict(), '{}/{}/visrender_G_epoch_{}'.format(self.output_dir, self.shortname, epoch))
    
    def train_save_image(self, epoch: int):
        ovl = self.__vismodel.get_raw_overlaid()
        bg = self.__vismodel.get_background() 
        blendimg = ovl * self.train_alphamap + bg * (1.0-self.train_alphamap)

        target_vismap_out = self.lossF.target_vis
        target_vismap_out = target_vismap_out.expand(-1,3,-1,-1)
        
        vismap_out = self.__vismodel.norm_vismap
        vismap_out = vismap_out.expand(-1,3,-1,-1)
        
        output_image = torch.cat([ovl, bg, self.train_alphamap, blendimg], dim=3)
        vutils.save_image(output_image[:,[2,1,0]],
                '{}/{}/visrender_epoch_{}.png'.format(self.output_dir, self.shortname, epoch),
                normalize=False)
        
        output_image = torch.cat([target_vismap_out, vismap_out], dim=3)
        MIN_HEIGHT = 256
        if output_image.shape[1] < MIN_HEIGHT:
            output_image = nearest_resize(output_image, MIN_HEIGHT, int(MIN_HEIGHT * (output_image.shape[3]/output_image.shape[2])))
        vutils.save_image(output_image[:,[2,1,0]],
                '{}/{}/visrender_epoch_{}_vismap.png'.format(self.output_dir, self.shortname, epoch),
                normalize=False)
    
    def train_append_log(self, message):
        log_file = '{}/{}/visbasedrender.log'.format(self.output_dir, self.shortname)
        with open(log_file, "a") as log_file:
            log_file.write('{}\n'.format(message))  # save the message
    
    def train_print_loss(self, epoch: int, phase: str = "train"):
        assert phase in ["train","val","test"]
        elapsed_time = time.time() - self.training_start_time

        if phase == "val":
            phase = 'valid_'
        elif phase == "test":
            phase = 'test_'
        else:
            phase = 'train_'
        message = phase + f'epoch: {epoch}, time: {elapsed_time:.7f}, {self.lossF.print_loss()}lr: {self.optimizerG.param_groups[0]["lr"]:.5f}'
        self.train_append_log(message)
    
    def train_save_loss(self):
        val=[]
        train=[]
        count = 0
        count_list = []
        with open('{}/{}/visbasedrender.log'.format(self.output_dir, self.shortname)) as f:
            for s_line in f:
                if 'valid_epoch' in s_line:
                    m = re.search(', loss:', s_line)
                    ind=m.end()+1
                    loss_val = float(s_line[ind:ind+9])
                    val.append(loss_val)
                    count_list.append(count)
                    count += 1

                elif 'train_epoch' in s_line:
                    m = re.search(', loss:', s_line)
                    ind=m.end()+1
                    loss_val = float(s_line[ind:ind+9])
                    train.append(loss_val)

        plt.plot(count_list,train,label="train")
        plt.plot(count_list,val,label="valid")
        #plt.show()
        plt.legend()
        plot_file = '{}/{}/loss_plot.png'.format(self.output_dir, self.shortname)
        plt.savefig(plot_file)
    
    def save_test_df(self):
        self.test_df.to_pickle(f"{self.output_dir}/{self.shortname}/test_df.pkl")
    
    def __train_modify_learning_rate(self, epoch: int) -> float:
        #if 40 < 0:
        #    return 1.0

        delta = max(0, epoch - 100) / float(400)
        return max(0.0, 1.0 - delta)

    def __tv_smooth_weight(self, image: torch.Tensor) -> torch.Tensor:

        pad_img = F.pad(image,(1,1,1,1),mode='reflect')

        hkernel = torch.Tensor([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]]).to(self.device)

        hkernel = hkernel.view((1,1,3,3))

        vkernel = torch.Tensor([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]]).to(self.device)

        vkernel = vkernel.view((1,1,3,3))

        G_x = F.conv2d(pad_img, hkernel)
        G_y = F.conv2d(pad_img, vkernel)
        
        hweight = torch.exp(-torch.abs(G_x))
        vweight = torch.exp(-torch.abs(G_y))

        return hweight * vweight