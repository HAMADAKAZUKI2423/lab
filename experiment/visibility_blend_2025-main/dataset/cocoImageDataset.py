from __future__ import annotations
import torch
from utils import load_img_list, stimulus
from .IBlendDataset import IBlendDataset
from dataset.train_coco_dataset import CocoDataset
from utils import save_img_torch

ID_LIST = [100,150,200,250,300,350,400,450,500]
WO_SEMI = True

class cocoImageTestDataset(IBlendDataset):
    def __init__(self, device: torch.device, size: int = 256):
        self.device = device
        self.name = "coco_dataset"
        self.size = size

        self.dataset = CocoDataset("content", device=device, wo_semi=WO_SEMI,tv_type="map",crop_size=size)
        self.data_list = []
        for id in ID_LIST:
            data_dict = self.dataset[id]
            data_dict['fg'] = data_dict['fg'] * 0.5 + 0.5
            data_dict['bg'] = data_dict['bg'] * 0.5 + 0.5
            data_dict['fg_color'] = data_dict['fg_color'] * 0.5 + 0.5
            data_dict['id'] = id
            self.data_list.append(data_dict)
    
    def load_dataset(self, visibility: float) -> list[stimulus]:
        assert 0. <= visibility and visibility <= 1.
        stimulus_list = []

        for data_dict in self.data_list:
            stim = stimulus()
            stim.set_ovl(data_dict['fg'].unsqueeze(0))
            stim.set_bg(data_dict['bg'].unsqueeze(0))
            stim.set_mask(data_dict['alpha_mask'].unsqueeze(0))
            stim.set_content_color(data_dict['fg_color'].unsqueeze(0))
            stim.set_content_alpha(data_dict['fg_alpha'].unsqueeze(0))
            vismap = torch.ones_like(stim.mask,dtype=torch.float32,device=self.device)*visibility
            stim.set_vismap(vismap)
            stim.set_index([str(data_dict['id']), f"{str(int(visibility*100))}_"])

            stimulus_list.append(stim)
        return stimulus_list
    
    def tile_show(self, blend_dir: str, vis_list: list[float]) -> None:
        pass

    def save_dataset(self, dir_path: str):
        for data_dict in self.data_list:
            save_img_torch(dir_path+f'{data_dict["id"]}_fg.png',data_dict['fg'],None)
            save_img_torch(dir_path+f'{data_dict["id"]}_bg.png',data_dict['bg'],None)
            save_img_torch(dir_path+f'{data_dict["id"]}_fg_color.png',data_dict['fg_color'],None)
            save_img_torch(dir_path+f'{data_dict["id"]}_fg_alpha.png',data_dict['fg_alpha'],None)