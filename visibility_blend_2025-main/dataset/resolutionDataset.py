from __future__ import annotations
import torch
from utils import load_img_list
from .utils_dataset import tile_show_core
from utils import stimulus
from .IBlendDataset import IBlendDataset

BG_NUM = 3 # max 3
FG_OP_NUM = 3 # max 3
#RES_LIST = ["384x256","768x512","1536x1024","3072x2048"]
RES_LIST = ["384x256","768x512","1536x1024"]

class resolutionDataset(IBlendDataset):
    def __init__(self, base_path: str, device: torch.device):
        self.device = device
        self.name = "resolution_dataset"

        self.imgs_dict = {}
        for res in RES_LIST:
            bg_name_list = [f"bg{i}" for i in range(BG_NUM)]
            bg_dir_path = f"imgs/exp/hd/{res}/"
            bg_lists = load_img_list(base_path + bg_dir_path, bg_name_list, self.device)

            fg_op_list = []
            fg_op_list = [f"{res}/fg{i}" for i in range(FG_OP_NUM)]
            fg_op_dir_path = f"imgs/exp/hd/"
            fg_op_lists = load_img_list(base_path + fg_op_dir_path, fg_op_list, self.device)

            self.imgs_dict[res] = (fg_op_lists, bg_lists)
    
    def load_dataset(self, visibility: float) -> list[stimulus]:
        assert 0. <= visibility and visibility <= 1.
        stimulus_list = []
        for res in RES_LIST:
            fg_lists, bg_lists = self.imgs_dict[res]

            height = bg_lists[0]["img"].shape[2]
            width = bg_lists[0]["img"].shape[3]
            ones = torch.ones((1,1,height,width),dtype=torch.float32,device=self.device)
            vismap = torch.ones((1,1,height,width),dtype=torch.float32,device=self.device)*visibility
            for bg_dict in bg_lists:
                for fg_dict in fg_lists:
                    stim = stimulus()
                    stim.set_bg(bg_dict["img"])
                    stim.set_mask(ones)
                    stim.set_content_color(fg_dict["img"])
                    stim.set_content_alpha(fg_dict["alpha"])
                    stim.set_ovl(fg_dict["img"] * fg_dict["alpha"] + bg_dict["img"] * (1 - fg_dict["alpha"]))
                    stim.set_vismap(vismap)
                    stim.set_index([fg_dict["name"], f"{bg_dict['name']}_{str(int(visibility*100))}_"])

                    stimulus_list.append(stim)
        return stimulus_list
    
    def tile_show(self, blend_dir: str, vis_list: list[float]) -> None:
        for res in RES_LIST:
            fg_lists, bg_lists = self.imgs_dict[res]

            bg_names = [name["name"] for name in bg_lists]
            fg_names = [name["name"] for name in fg_lists]
            tile_show_core(blend_dir, bg_names, fg_names, vis_list)

