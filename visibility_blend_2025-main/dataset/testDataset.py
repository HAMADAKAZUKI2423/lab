from __future__ import annotations
import torch
from utils import load_img_list
from .IBlendDataset import IBlendDataset

class testDataset(IBlendDataset):
    def __init__(self, base_path: str, device: torch.device):
        self.device = device
        self.name = "test_dataset"

        bg_name_list = []
        bg_name_list.append("noise")
        bg_dir_path = "imgs/exp/test/"
        bg_lists = load_img_list(base_path + bg_dir_path, bg_name_list, self.device)
        bg_master = bg_lists[0]
        bg_lists = []
        for contrast in [1,0.75,0.5,0.25,0]:
            new_dict = {
                "name": bg_master["name"]+str(int(contrast*100)).replace('.',''),
                "img": bg_master["img"]*contrast + 0.5 * (1 - contrast),
                "alpha": bg_master["alpha"]
                }
            bg_lists.append(new_dict)
        self.bg_lists= bg_lists

        fg_list = []
        fg_list.append("stripes")
        fg_dir_path = "imgs/exp/test/"
        fg_lists = load_img_list(base_path + fg_dir_path, fg_list, self.device, flag="alpha")
        fg_master = fg_lists[0]
        fg_lists = []
        for initial_alpha in [1,0.75,0.5,0.25]:
            new_dict = {
                "name": fg_master["name"]+str(int(initial_alpha*100)).replace('.',''),
                "img": fg_master["img"],
                "alpha": fg_master["alpha"] * initial_alpha
                }
            fg_lists.append(new_dict)
        self.fg_lists= fg_lists
