from __future__ import annotations
import torch
import abc
from utils import stimulus
from .utils_dataset import tile_show_core

class IBlendDataset(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, base_path: str, device: torch.device):
        self.device = device
        self.name = "iblend_dataset"
        self.bg_lists:list[dict[str]]
        self.fg_lists:list[dict[str]]
        raise NotImplementedError()

    def load_dataset(self, visibility: float) -> list[stimulus]:
        assert 0. <= visibility and visibility <= 1.
        stimulus_list = []
        height = self.bg_lists[0]["img"].shape[2]
        width = self.bg_lists[0]["img"].shape[3]
        vismap = torch.ones((1,1,height,width),dtype=torch.float32,device=self.device)*visibility
        for bg_dict in self.bg_lists:
            for fg_dict in self.fg_lists:
                stim = stimulus()
                stim.set_bg(bg_dict["img"])
                stim.set_mask(torch.where(fg_dict["alpha"] > 0,1.,0.).to(torch.float32))
                stim.set_content_color(fg_dict["img"])
                stim.set_content_alpha(fg_dict["alpha"])
                stim.set_ovl(fg_dict["img"] * fg_dict["alpha"] + bg_dict["img"] * (1 - fg_dict["alpha"]))
                stim.set_vismap(vismap)
                stim.set_index([fg_dict["name"], f"{bg_dict['name']}_{str(int(visibility*100))}_"])

                stimulus_list.append(stim)
        return stimulus_list
    
    def tile_show(self, blend_dir: str, vis_list: list[float]) -> None:
        bg_names = [name["name"] for name in self.bg_lists]
        fg_names = [name["name"] for name in self.fg_lists]
        tile_show_core(blend_dir, bg_names, fg_names, vis_list)
    
    def save_dataset(self, path: str):
        pass