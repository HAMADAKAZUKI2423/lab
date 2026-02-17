from __future__ import annotations
import torch
import os
import sys
sys.path.append(os.getcwd())
from utils import load_img_list
from .IBlendDataset import IBlendDataset


BG_NUM = 1#10 # max 10
#FG_OP_CAT = ["face","illust","object","text"]
FG_OP_CAT = ["face","illust","text"]#["face","illust","object","text"]
FG_OP_NUM = 4#4 # max 4

class expDataset(IBlendDataset):
    def __init__(self, base_path: str, device: torch.device):
        self.device = device
        self.name = "exp_dataset"
        
        bg_name_list = [f"{i}" for i in range(BG_NUM)]
        bg_name_list.append("gray50")
        bg_dir_path = "imgs/exp/background/"
        bg_lists = load_img_list(base_path + bg_dir_path, bg_name_list, self.device)
        self.bg_lists= bg_lists

        fg_op_list = []
        for cat in FG_OP_CAT:
            fg_op_list.extend([f"{cat}/{i}" for i in range(FG_OP_NUM)])
        fg_op_dir_path = "imgs/exp/target/opacity/"
        fg_op_lists = load_img_list(base_path + fg_op_dir_path, fg_op_list, self.device)
        self.fg_lists= fg_op_lists

ALPHA = 0.5
class expAlphaDataset(expDataset):
    def __init__(self, base_path: str, device: torch.device):
        super().__init__(self,base_path,device)
        self.name = "exp_alpha_dataset"
        for fg_dict in self.fg_lists:
            fg_dict["alpha"] = fg_dict["alpha"]*ALPHA