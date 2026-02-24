from __future__ import annotations
import abc
import torch
from utils import stimulus

class IBlender(metaclass=abc.ABCMeta):
    def __init__(self):
        self.alphamap: torch.Tensor | None = None
        self.blendimg: torch.Tensor | None = None

    @abc.abstractmethod
    def blend(self, stim: stimulus):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def save_imgs(self, save_path: str):
        raise NotImplementedError()