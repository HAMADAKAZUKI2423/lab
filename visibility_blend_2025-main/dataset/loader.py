from __future__ import annotations
import torch
from .IBlendDataset import IBlendDataset
from .expDataset import expDataset, expAlphaDataset
from .expTransDataset import expTransDataset, expTransTgbgDataset
from .resolutionDataset import resolutionDataset
from .testDataset import testDataset
# from .cocoDtdImageDataset import cocoDtdImageTestDataset
from .cocoImageDataset import cocoImageTestDataset

def load_blend_dataset(args, base_path: str, device: torch.device, config:dict = {}) -> IBlendDataset | None:
    if args.exp:
        return expDataset(base_path, device)
    elif args.exptrans:
        return expTransDataset(base_path, device)
    elif args.exptranstgbg:
        return expTransTgbgDataset(base_path, device)
    elif args.res:
        return resolutionDataset(base_path, device)
    elif args.test:
        return testDataset(base_path, device)
    elif args.expalpha:
        return expAlphaDataset(base_path, device)
    # elif args.cocodtd:
    #     return cocoDtdImageTestDataset(device)
    elif args.coco:
        return cocoImageTestDataset(device,config.get("size",256))
    else:
        return None