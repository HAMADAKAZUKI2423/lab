from __future__ import annotations
import abc
import torch
from ..supermodels.visModel import VisModel

class ILossFunction(metaclass=abc.ABCMeta):
    def reset_state(self):
        self.target_vis: torch.Tensor | None  = None
        self.target_vis_rawscale: torch.Tensor | None  = None
    
    @abc.abstractmethod
    def reset_loss(self):
        self.all_loss: torch.Tensor | None = None
        raise NotImplementedError()
    
    @abc.abstractmethod
    def compute_loss_preprocess(self, target_vis:torch.Tensor, vismodel:VisModel):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def compute_loss(self,
                      vismodel:VisModel,
                      alphamap:torch.Tensor, 
                      spatial_weight: torch.Tensor | None = None):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def save_loss(self, dir_path: str):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def save_img(self, dir_path: str):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def print_loss(self) -> str:
        raise NotImplementedError()