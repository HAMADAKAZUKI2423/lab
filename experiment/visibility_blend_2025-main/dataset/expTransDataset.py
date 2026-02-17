from __future__ import annotations
import torch
import os
import sys
sys.path.append(os.getcwd())
from utils import load_img_list
from .IBlendDataset import IBlendDataset

BG_NUM = 10 # max 10
#FG_TP_CAT = ["art","natural"]
FG_TP_CAT = ["natural"]
FG_TP_NUM = 8 # max 8

#FG_OP_CAT = ["face","illust","object","text"]
FG_OP_CAT = ["face","illust","object","text"]
FG_OP_NUM = 4 # max 4

class expTransDataset(IBlendDataset):
    def __init__(self, base_path: str, device: torch.device):
        self.device = device
        self.name = "exp_trans_dataset"

        bg_name_list = [f"{i}" for i in range(BG_NUM)]
        bg_name_list.append("gray50")
        bg_dir_path = "imgs/exp/background/"
        bg_lists = load_img_list(base_path + bg_dir_path, bg_name_list, self.device)
        self.bg_lists = bg_lists

        fg_tp_list = []
        for cat in FG_TP_CAT:
            fg_tp_list.extend([f"{cat}/{i}" for i in range(FG_TP_NUM)])
        fg_tp_dir_path = "imgs/exp/target/transparent/"
        fg_tp_lists = load_img_list(base_path + fg_tp_dir_path, fg_tp_list, self.device, flag="alpha")
        self.fg_lists = fg_tp_lists

class expTransTgbgDataset(IBlendDataset):
    def __init__(self, base_path: str, device: torch.device):
        self.device = device
        self.name = "exp_trans_tgbg_dataset"

        bg_name_list = []
        for cat in FG_OP_CAT:
            bg_name_list.extend([f"{cat}/{i}" for i in range(FG_OP_NUM)])
        bg_dir_path = "imgs/exp/target/opacity/"
        bg_lists = load_img_list(base_path + bg_dir_path, bg_name_list, self.device)
        self.bg_lists= bg_lists

        fg_tp_list = []
        for cat in FG_TP_CAT:
            fg_tp_list.extend([f"{cat}/{i}" for i in range(FG_TP_NUM)])
        fg_tp_dir_path = "imgs/exp/target/transparent/"
        fg_tp_lists = load_img_list(base_path + fg_tp_dir_path, fg_tp_list, self.device, flag="alpha")
        self.fg_lists = fg_tp_lists