from __future__ import annotations

local_mode = False
if local_mode:
    import sys
    import os

    BASE_DIR = os.path.dirname(__file__)   # sandor.py があるディレクトリ
    sys.path.append(BASE_DIR)

    # 例: 2つ上のディレクトリをpathに追加
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    sys.path.append(BASE_DIR)

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from utils import save_img_torch, stimulus

if local_mode:
    from IBlender import IBlender
else:
    from .IBlender import IBlender



eps = 1e-12

class sandorBlender(IBlender):
    def __init__(self, device: torch.device, target_type: str = "content", use_motion: bool = False, resize: float = 1.0,
                 input_size: tuple[int, int] | None = None, apply_sigmoid=False):
        super().__init__()
        self.device = device
        self.set_target_type(target_type)

        self.filt = self.gauss_kernel(3)
        self.pad_two = nn.ReflectionPad2d(2)

        self.use_motion = use_motion

        self.prev_ovl = None
        self.prev_bg = None

        # usertest_video_make.py内で使用するパラメータ
        self.resize = resize
        self.input_size = input_size

        self.apply_sigmoid = apply_sigmoid

        # sigmoid fit by rating dataset 2024
        self.sigmoid_param = np.float64([-1.1679441703841447,
                                            393.5802369104692,
                                            261.28469316948247])

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
            alphamap: torch.Tensor = (1-_alphamap)# * stim.mask
        else:
            alphamap: torch.Tensor = _alphamap# * stim.mask

        # alphamap = stim.vismap * stim.mask
        alphamap = alphamap.expand(-1,3,-1,-1).to(torch.float32)
        # if self.target_type == "content":
        #     self.blendimg, self.alphamap = self.process(stim.ovl, stim.bg, alphamap)
        # else:
        #     self.blendimg, self.alphamap = self.process(stim.bg, stim.ovl, alphamap)
        
        self.blendimg, self.alphamap = self.process(stim.ovl, stim.bg, alphamap, stim.mask)
        
        self.prev_ovl = stim.ovl
        self.prev_bg = stim.bg

    def save_imgs(self, save_path: str):
        data_list = [self.alphamap, self.blendimg]
        path_list = [save_path + name for name in ["alphamap.png","blend.png"]]
        for (data, path) in zip(data_list, path_list):
            save_img_torch(path, data)

    def process(self, occluder, occluded, mask, hardmask, save = False):
        derSal = self.makeSaliencyMap(occluder, self.prev_ovl)
        derEdg = self.makeEdgeMap(occluder, derSal, weight = 3)
        dedSal = self.makeSaliencyMap(occluded, self.prev_bg)
        derSalDash = torch.clip(derSal - dedSal + derEdg, 0, 1)
        if save:
            save_img_torch("sandor_derSal.png", derSal)
            save_img_torch("sandor_derEdg.png", derEdg)
            save_img_torch("sandor_dedSal.png", dedSal)
            save_img_torch("sandor_derSalDash.png", derSalDash)
        blend = occluder * mask + (occluder * derSalDash + occluded * (1 - derSalDash)) * (1 - mask)

        blend = occluded * (1-hardmask) + blend * hardmask
        return blend, (mask + derSalDash * (1 - mask)) * hardmask

    def makeEdgeMap(self, img, saliency, weight = 1):
        sobelEdge = self.sobelFunction(img)
        return torch.abs(sobelEdge * saliency * weight)
    
    def sobelFunction(self, img):
        pad_img = F.pad(img,(1,1,1,1),mode='reflect')

        hkernel = torch.Tensor([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]]).to(self.device)

        hkernel = hkernel.view((1,1,3,3))
        hkernel = torch.cat([hkernel]*3,dim=1)

        vkernel = torch.Tensor([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]]).to(self.device)

        vkernel = vkernel.view((1,1,3,3))
        vkernel = torch.cat([vkernel]*3,dim=1)

        dkernel1 = torch.Tensor([[0, 1, 2],
                                    [-1, 0, 1],
                                    [-2, -1, 0]]).to(self.device)
            
        dkernel1 = dkernel1.view((1,1,3,3))
        dkernel1 = torch.cat([dkernel1]*3,dim=1)
        
        dkernel2 = torch.Tensor([[2, 1, 0],
                                [1, 0, -1],
                                [0, -1, -2]]).to(self.device)
        
        dkernel2 = dkernel2.view((1,1,3,3))
        dkernel2 = torch.cat([dkernel2]*3,dim=1)

        G_x = torch.abs(F.conv2d(pad_img, hkernel))
        G_y = torch.abs(F.conv2d(pad_img, vkernel))
        G_d1 = torch.abs(F.conv2d(pad_img, dkernel1))
        G_d2 = torch.abs(F.conv2d(pad_img, dkernel2))

        return (G_x + G_y + G_d1 + G_d2)/4

    def makeSaliencyMap(self, img, prev_img=None):# bgr
        # b = img[:,:1]
        # g = img[:,1:2]
        # r = img[:,2:]
        # maxBgr = torch.max(img, dim = 1, keepdim = True).values
        # luminosity = (b + g + r) / 3.
        # rg_opponency = (r - g) / (maxBgr + eps)
        # by_opponency = (b - torch.min(torch.cat([g,r], dim=1), dim = 1, keepdim = True).values) / (maxBgr + eps)
        # saliencyMaterial = torch.cat([luminosity,rg_opponency,by_opponency], dim=1)
        # # original paper consider motion saliency, but we dont

        # if self.use_motion and type(prev_img) == torch.Tensor:
        #     b = prev_img[:,:1]
        #     g = prev_img[:,1:2]
        #     r = prev_img[:,2:]
        #     prev_luminosity = (b + g + r) / 3.
        #     motion_map = luminosity - prev_luminosity

        
        # saliencyPyr = self.gen_originalScale_Gpyr(saliencyMaterial,8) # 0-7
        # saliencyTmp = torch.zeros_like(saliencyPyr[0])
        # for up_layer in [1,2,3]:
        #     for layer_diff in [3,4]:
        #         saliencyTmp += self.min_max_Norm(torch.abs(saliencyPyr[up_layer] - saliencyPyr[up_layer+layer_diff]))
        
        # return self.min_max_Norm(torch.mean(saliencyTmp, dim = 1, keepdim=True))
        """
        入力 img（BGR, [B,3,H,W]）に対し、まずガウシアンピラミッド \(P_\sigma\) を生成し、
        その各スケールから輝度 l と色反対性 c を抽出する。
        前フレーム prev_img が与えられた場合、同様にピラミッドを生成し、
        各スケールの輝度差からモーション t を計算する。
        その後、各特徴について論文の指示に沿む Center-Surround 差分を計算し、
        最終的に l, c, (t) を平均して [0,1] に正規化したサリエンシーマップを返す。
        """
        # --- 1. 入力画像からガウシアンピラミッドを生成 ---
        P = self.gen_originalScale_Gpyr(img, level=8)
        # 各レベルから輝度と色特徴を抽出
        L, C_feat = self.extract_features(P)
        
        # --- 2. モーション特徴 ---
        if self.use_motion and (prev_img is not None):
            P_prev = self.gen_originalScale_Gpyr(prev_img, level=8)
            L_prev, _ = self.extract_features(P_prev)
            # 各レベルでの輝度差分（絶対値）をモーション特徴とする
            T = [torch.abs(L[i] - L_prev[i]) for i in range(len(L))]
        else:
            T = None
        
        # --- 3. 各特徴について Center-Surround 差分（顕著性マップ）を計算 ---
        C_lum = self.compute_conspicuity(L)
        C_color = self.compute_conspicuity(C_feat)
        if T is not None:
            C_motion = self.compute_conspicuity(T)
        else:
            C_motion = 0
        
        # --- 4. 最終サリエンシーマップの統合 ---
        if T is not None:
            saliency = (C_lum + C_color + C_motion) / 3.0
        else:
            saliency = (C_lum + C_color) / 2.0
        
        saliency = self.min_max_Norm(saliency)
        return saliency

    def extract_features(self, P):
        """
        ガウシアンピラミッド P から各レベルの特徴を抽出する。
        入力 P は各レベルが [B,3,H,W] のテンソルのリストとする。
        輝度 l は各レベルで (B+G+R)/3.
        色反対性 c は、BGR順において、
          - RG-opponency = (r - g)/(max(r,g,b) + eps)
          - BY-opponency = (b - min(r,g))/(max(r,g,b) + eps)
        を絶対値取り、平均することで 1チャネル化する。
        """
        L = []       # 輝度リスト
        C_feat = []  # 色特徴リスト
        for p in P:
            # p は [B,3,H,W] (BGR)
            lum = (p[:, 0:1, :, :] + p[:, 1:2, :, :] + p[:, 2:3, :, :]) / 3.0
            L.append(lum)
            
            maxBgr = torch.max(p, dim=1, keepdim=True).values
            # RG-opponency: (r - g) / (max(r,g,b) + eps)
            rg = (p[:, 2:3, :, :] - p[:, 1:2, :, :]) / (maxBgr + eps)
            # BY-opponency: (b - min(r, g)) / (max(r,g,b) + eps)
            min_rg = torch.min(torch.cat([p[:, 2:3, :, :], p[:, 1:2, :, :]], dim=1), dim=1, keepdim=True).values
            by = (p[:, 0:1, :, :] - min_rg) / (maxBgr + eps)
            # 絶対値をとって平均する
            color = 0.5 * (torch.abs(rg) + torch.abs(by))
            C_feat.append(color)
        return L, C_feat
    
    def compute_conspicuity(self, P_feature):
        """
        ピラミッド各レベルの特徴 P_feature（リスト）から、論文記述に沿い
        p ∈ {2,3,4}（論文は1から数えるため、Pythonでは [1,2,3] に対応）と
        s = p+{3,4}（Pythonではそれぞれ p+3, p+4）との間の差分（Center-Surround差分）
        を計算し、正規化して平均することで顕著性マップを返す。
        """
        F_total = 0
        count = 0
        # 論文の p = {2,3,4} に対応する Python のインデックスは {1,2,3} となる
        for p in [1, 2, 3]:
            for s in [p + 3, p + 4]:
                if s < len(P_feature):
                    diff = torch.abs(P_feature[p] - P_feature[s])
                    F_total += self.min_max_Norm(diff)
                    count += 1
        if count > 0:
            return F_total / count
        else:
            return F_total

    def gen_originalScale_Gpyr(self, image, level):
        J = image
        dims = image.shape[1]
        #pyr = []
        gpyr=[]
        for i in range(level):
            I = F.conv2d(self.pad_two(J), self.filt, stride=2, padding=0,
                         groups=dims)
            I_up = I
            for j in range(i+1):
                I_up = self.upsample(I_up)#include conv
            I_up = torchvision.transforms.functional.resize(img=I_up, size=(image.shape[2], image.shape[3]),antialias = True)
            gpyr.append(I_up)

            J = I
        return gpyr
    
    def gen_Gpyr(self, image, level):
        #Gaussian pyramid upsampled by one level
        
        J = image
        dims = image.shape[1]
        #pyr = []
        gpyr=[]
        for i in range(level):
            I = F.conv2d(self.pad_two(J), self.filt, stride=2, padding=0,
                         groups=dims)
            I_up = self.upsample(I)#include conv
            gpyr.append(I_up)

            J = I
        return gpyr
    
    def gauss_kernel(self,channels=3):
        kernel = torch.tensor([
                [1, 5, 8, 5, 1],
                [5, 25, 40, 25, 5],
                [8, 40, 64, 40, 8],
                [5, 25, 40, 25, 5],
                [1, 5, 8, 5, 1]],dtype=torch.float32,device=self.device)
        kernel /= 400.
        kernel = kernel.repeat(channels, 1, 1, 1)
        return kernel

    def upsample(self, x, kernel=None):
        # cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        # cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
        # cc = cc.permute(0,1,3,2)
        # cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2, device=x.device)], dim=3)
        # cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
        # x_up = cc.permute(0,1,3,2)
        # if kernel is None:
        #     kernel = self.gauss_kernel(channels=x.shape[1])
        # return self.conv_gauss(x_up, 4*kernel)
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
    
    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out
    
    def min_max_Norm(self, img, new_min = 0, new_max = 1):
        img_min, img_max = img.min(), img.max()
        #print(f"SandorModel min_max_Norm: min({img_min}), max({img_max})")
        return (img - img_min)/(img_max - img_min + eps)*(new_max - new_min) + new_min
    
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
    
# ===== 動作確認用サンプル =====
if __name__ == '__main__':
    # ダミーの入力画像（バッチサイズ=1, 3チャネル, 256x256, BGR）
    dummy_img = torch.rand(1, 3, 512, 512)
    # 前フレーム（モーション用）のダミー画像
    dummy_prev = torch.rand(1, 3, 512, 512)
    
    saliency_gen = sandorBlender(device='cpu', use_motion=True)
    saliency_map = saliency_gen.makeSaliencyMap(dummy_img, prev_img=dummy_prev)
    print("Saliency map shape:", saliency_map.shape)  # 期待: [1, 1, 256, 256]

    import cv2
    import torch
    import numpy as np

    # 動画ソースを設定（0ならウェブカメラ、ファイルパスなら動画ファイル）
    cap = cv2.VideoCapture(0)

    saliency_gen = sandorBlender(device='cpu', use_motion=True)
    saliency_map = saliency_gen.makeSaliencyMap(dummy_img, prev_img=dummy_prev)

    prev_frame_tensor = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCVで読み込んだフレームは (H, W, 3) の uint8(BGR) なので、
        # [0,1] の浮動小数点に変換し、PyTorchのテンソル形式 [1,3,H,W] に変換
        frame_float = frame.astype(np.float32) / 255.0
        frame_tensor = torch.from_numpy(frame_float).permute(2, 0, 1).unsqueeze(0)

        # 前フレームがあればモーション計算に利用
        if prev_frame_tensor is None:
            sal_map = saliency_gen.makeSaliencyMap(frame_tensor, prev_img=None)
        else:
            sal_map = saliency_gen.makeSaliencyMap(frame_tensor, prev_img=prev_frame_tensor)

        # 現在のフレームを前フレームとして保存（cloneを忘れずに）
        prev_frame_tensor = frame_tensor.clone()

        # サリエンシーマップは [1,1,H,W] のテンソル。squeezeして (H,W) にし、[0,255] に変換
        sal_map_np = sal_map.squeeze().detach().cpu().numpy()
        sal_map_np = (sal_map_np * 255).astype(np.uint8)

        # 視覚化のため、カラーマップ（JET）を適用
        sal_map_color = cv2.applyColorMap(sal_map_np, cv2.COLORMAP_JET)

        # もしグレースケール画像のままで表示したいなら、3チャネルに変換
        sal_map_gray = cv2.cvtColor(sal_map_np, cv2.COLOR_GRAY2BGR)


        # オリジナルのフレームとサリエンシーマップを横に連結して表示
        combined = np.hstack((frame, sal_map_gray))
        cv2.imshow('Original and Saliency', combined)

        # 'q' キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()