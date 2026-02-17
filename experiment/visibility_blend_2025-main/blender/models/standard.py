from __future__ import annotations
import numpy as np
import torch
from utils import save_img_torch, stimulus
from .IBlender import IBlender

class standardBlender(IBlender):
    def __init__(self, target_type: str = "content", apply_sigmoid=False, resize: float = 1.0,
                 input_size: tuple[int, int] | None = None):
        super().__init__()
        self.set_target_type(target_type)

        self.apply_sigmoid = apply_sigmoid

        # sigmoid fit by rating dataset 2024
        self.sigmoid_param = np.float64([-1.1679441703841447,
                                            393.5802369104692,
                                            261.28469316948247])
        
        # usertest_make_rating2025.py内で使用するパラメータ
        self.resize = resize
        self.input_size = input_size

    def set_target_type(self, target_type: str):
        assert target_type in ["content", "background"]
        self.target_type = target_type

    def blend(self, stim: stimulus):
        assert None not in [stim.bg, stim.ovl, stim.mask, stim.vismap]

        if self.apply_sigmoid:
            _alphamap = self.visibility_to_rawscale(stim.vismap)
        else:
            _alphamap = stim.vismap

        if self.target_type == "background":
            self.alphamap: torch.Tensor = (1-_alphamap) * stim.mask
        else:
            self.alphamap: torch.Tensor = _alphamap * stim.mask
        self.alphamap = self.alphamap.expand(-1,3,-1,-1).to(torch.float32)
        self.blendimg = self.alphamap * stim.ovl + (1.0-self.alphamap) * stim.bg
    
    def save_imgs(self, save_path: str):
        data_list = [self.alphamap, self.blendimg]
        path_list = [save_path + name for name in ["alphamap.png","blend.png"]]
        for (data, path) in zip(data_list, path_list):
            save_img_torch(path, data)
    
    def visibility_to_rawscale(self, vis):
        # 1. 元のdeviceを記憶（例えば vis の device を利用）
        original_device = vis.device

        # 2. すべてのテンソルをCPUに移動して detach し、NumPy配列に変換
        
        vis_np = np.float64(vis.cpu().detach().numpy())

        # 3. force_positive の計算（np.log(np.exp(x)+1)）
        A = np.log(np.exp(self.sigmoid_param[0]) + 1) # force_positive
        B = np.log(np.exp(self.sigmoid_param[1]) + 1) # force_positive
        v = np.log(np.exp(self.sigmoid_param[2]) + 1) # force_positive

        Q = ((1+A)/A)**v - 1
        # print("A:", A, "\nB:", B, "\nv:", v, "\nQ:", Q)

        # 5. vis の clamping（torch.clamp と同等に [0, 0.99999] の範囲にする）
        _vis = np.clip(vis_np, 0, 0.99999)

        # 6. x の計算
        #    ※ 演算順序は元のコードと同様に、まず ((1+A)/(_vis+A))**v, その後の処理
        x_np = -np.log((((1 + A) / (_vis + A)) ** v - 1) / Q) / B

        # 7. 中間の結果を確認（例として x[...,300,300] を表示）
        #    ※ x_np の次元数が十分である前提です
        # print("x[...,300,300]:", x_np[..., 300, 300] if x_np.ndim > 2 else x_np[300, 300])

        # 8. NumPy 配列を torch.Tensor に変換し、元の device に戻す
        x_tensor = torch.from_numpy(x_np).to(original_device)

        # 9. 結果を return
        return x_tensor
        # if mask:
        #     vis = vis * self.dilated_mask_gp[0]
        # return inv_sigmoid(vis, self.param_fullmodel)
    
        # A = np.log(np.exp(self.sigmoid_param[0]) + 1) # force_positive
        # B = np.log(np.exp(self.sigmoid_param[1]) + 1) # force_positive
        # v = np.log(np.exp(self.sigmoid_param[2]) + 1) # force_positive

        # Q = ((1+A)/A)**v - 1
        # print(A,B,v,Q)
        # _vis = torch.clamp(vis,min=0,max=0.99999)
        # x = -torch.log((((1+A)/(_vis+A))**v - 1)/Q)/B
        # print(x[...,300,300])
        # return x