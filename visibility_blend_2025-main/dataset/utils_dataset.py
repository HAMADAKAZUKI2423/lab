from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import random
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from PIL import Image

import random
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

class RandomResizedCropCustom(object):
    """
    scale=1 であれば、
      - 出力サイズ(size)と同じ大きさの領域を元画像からクロップし
      - リサイズ時に拡大・縮小が行われない（等倍）状態になるようにする。
    scale>1 の場合はダウンスケール、scale<1 の場合はアップスケールとなる。
    """
    def __init__(self, size, scale=(0.8, 1.2), interpolation=InterpolationMode.BILINEAR):
        """
        Args:
            size (int or tuple): 最終的にリサイズする (width, height) あるいは int(正方形)。
            scale (tuple of float): (min_scale, max_scale) の範囲。
            interpolation: リサイズの補間方法 (デフォルト: BILINEAR)。
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.scale_range = scale
        self.interpolation = interpolation

    def get_params(self, img):
        """
        画像 img をもとにクロップ領域のパラメータをランダムに決定し、dict として返す。

        Returns:
            params (dict): 
                {
                    'scale': scale_value,
                    'x1': 左上x座標,
                    'y1': 左上y座標,
                    'crop_width': クロップ幅,
                    'crop_height': クロップ高さ
                }
        """
        orig_width, orig_height = F.get_image_size(img)
        target_width, target_height = self.size

        # ランダムに scale をサンプリング
        scale = random.uniform(self.scale_range[0], self.scale_range[1])

        # クロップ領域のサイズを決定
        crop_width = int(target_width * scale)
        crop_height = int(target_height * scale)

        # 元画像より大きくならないようにクリップ
        crop_width = min(crop_width, orig_width)
        crop_height = min(crop_height, orig_height)

        # 左上座標をランダムに決定
        if orig_width == crop_width:
            x1 = 0
        else:
            x1 = random.randint(0, orig_width - crop_width)

        if orig_height == crop_height:
            y1 = 0
        else:
            y1 = random.randint(0, orig_height - crop_height)

        return {
            'scale': scale,
            'x1': x1,
            'y1': y1,
            'crop_width': crop_width,
            'crop_height': crop_height
        }
    
    def __call__(self, img, params=None):
        """
        Args:
            img (PIL Image または Tensor):
            params (dict または None): 
                - None の場合は get_params によりランダムにパラメータをサンプリング
                - dict の場合は { 'scale', 'x1', 'y1', 'crop_width', 'crop_height' } を想定
        """
        if params is None:
            # パラメータをランダムにサンプリング
            params = self.get_params(img, self.size, self.scale_range)

        # パラメータを展開
        scale = params['scale']
        x1 = params['x1']
        y1 = params['y1']
        crop_width = params['crop_width']
        crop_height = params['crop_height']

        # クロップ
        img = F.crop(img, y1, x1, crop_height, crop_width)

        # リサイズ (常に self.size へ)
        img = F.resize(img, self.size, interpolation=self.interpolation)

        return img



def tile_show_core(blend_dir: str, bg_names: list[str], fg_names: list[str], vis_list: list[float]):
    vis_names = [str(int(visibility*100)) for visibility in vis_list]

    type_names = ["alphamap","blend","vismap"]
    color_types = ["blend"]
    rows = len(vis_names)
    lines = len(type_names)
    figsize = 3

    for fg_idx, fg_name in enumerate(fg_names):
        for bg_idx, bg_name in enumerate(bg_names):
            print(f"saving: {fg_idx}/{len(fg_names)}, {bg_idx}/{len(bg_names)}")
            fig, ax = plt.subplots(rows, lines, figsize=(figsize*lines, figsize*rows))
            fig.subplots_adjust(hspace=0, wspace=0)
            for vis_idx, vis_name in enumerate(vis_names):
                for type_idx, type_name in enumerate(type_names):
                    data = Image.open(f"{blend_dir}/{fg_name}/{bg_name}_{vis_name}_{type_name}.png")
                    data = np.asarray(data)
                    data = cv2.resize(data, (512, 512), cv2.INTER_LANCZOS4)
                    ax[vis_idx, type_idx].xaxis.set_major_locator(plt.NullLocator())
                    ax[vis_idx, type_idx].yaxis.set_major_locator(plt.NullLocator())
                    if type_name in color_types:
                        ax[vis_idx, type_idx].imshow(data)
                    else:
                        ax[vis_idx, type_idx].imshow(data,cmap="gray", vmin=0, vmax=255)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            plt.savefig(f"{blend_dir}/{fg_name}/{bg_name}_arrange.png")
            plt.close()