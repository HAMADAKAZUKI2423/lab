import torch
import numpy as np
from skimage.color import deltaE_ciede2000

def deltaE_ciede2000_torch(lab1: torch.Tensor, lab2: torch.Tensor, kL=1.0, kC=1.0, kH=1.0) -> torch.Tensor:
    """
    PyTorch版のCIEDE2000 ΔE計算関数
    lab1, lab2: shape (..., 3) の Tensor (L, a, b)
    kL, kC, kH: 定数 (通常は1.0)
    return: 同じshape(...,)のΔE値のTensor
    """

    # skimage実装のロジックに沿って計算する
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    # Step 1: Cの計算
    C1 = torch.sqrt(a1**2 + b1**2)
    C2 = torch.sqrt(a2**2 + b2**2)
    Cm = (C1 + C2) / 2.0

    # G係数
    Cm7 = Cm**7
    G = 0.5 * (1 - torch.sqrt(Cm7 / (Cm7 + 25.0**7)))

    # a'計算
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2

    # C'計算
    C1p = torch.sqrt(a1p**2 + b1**2)
    C2p = torch.sqrt(a2p**2 + b2**2)

    # h'計算: arctan2(b', a')
    # skimageは h' = atan2(b', a') (度数に変換) のあと %360で0~360度範囲化
    def hp_func(ax, bx):
        # atan2でラジアン取得、degにして0-360度
        h = torch.rad2deg(torch.atan2(bx, ax)) % 360
        return h

    h1p = hp_func(a1p, b1)
    h2p = hp_func(a2p, b2)

    # ΔL', ΔC'
    dLp = L2 - L1
    dCp = C2p - C1p

    # Δh'計算
    dhp = h2p - h1p
    # np.mod(dhp,360)と同等の処理
    dhp_mod = dhp % 360
    # dhp_modが180度より大きい場合、360度を引いて -180~180へ
    dhp_mod = torch.where(dhp_mod > 180, dhp_mod - 360, dhp_mod)

    # ΔH'
    dHp = 2.0 * torch.sqrt(C1p * C2p) * torch.sin(torch.deg2rad(dhp_mod / 2.0))

    # 平均値計算
    Lp = (L1 + L2) / 2.0
    Cp = (C1p + C2p) / 2.0

    # h_bar'
    # skimage: hp_bar = (h1p + h2p)/2
    # |h1p-h2p|>180の画素には +180度してからmod360
    hp_bar = (h1p + h2p) / 2.0
    cond = (torch.abs(h1p - h2p) > 180)
    hp_bar = torch.where(cond, hp_bar + 180.0, hp_bar)
    hp_bar = hp_bar % 360

    # T係数
    T = (1
         - 0.17*torch.cos(torch.deg2rad(hp_bar - 30))
         + 0.24*torch.cos(torch.deg2rad(2*hp_bar))
         + 0.32*torch.cos(torch.deg2rad(3*hp_bar + 6))
         - 0.20*torch.cos(torch.deg2rad(4*hp_bar - 63)))

    # ΔΘ, R_C, R_T
    Delta_theta = 30.0 * torch.exp(-(((hp_bar - 275.0)/25.0)**2))
    R_C = 2.0 * torch.sqrt((Cp**7) / (Cp**7 + 25.0**7))
    R_T = - R_C * torch.sin(torch.deg2rad(2.0 * Delta_theta))

    # S_L, S_C, S_H
    S_L = 1 + (0.015 * (Lp - 50)**2) / torch.sqrt(20 + (Lp - 50)**2)
    S_C = 1 + 0.045 * Cp
    S_H = 1 + 0.015 * Cp * T

    # ΔE2000計算
    delta_E = torch.sqrt((dLp/(S_L*kL))**2 + (dCp/(S_C*kC))**2 + (dHp/(S_H*kH))**2
                         + R_T*(dCp/(S_C*kC))*(dHp/(S_H*kH)))
    return delta_E


# ===== テストコード =====
if __name__ == "__main__":
    # ランダムなLab画像を生成 (H=64, W=64)
    H, W = 64, 64
    np.random.seed(0)
    lab1_np = np.empty((H, W, 3), dtype=np.float64)
    lab2_np = np.empty((H, W, 3), dtype=np.float64)
    # L:0~100, a,b:-128~127
    lab1_np[..., 0] = np.random.uniform(0, 100, (H, W))
    lab1_np[..., 1] = np.random.uniform(-128, 127, (H, W))
    lab1_np[..., 2] = np.random.uniform(-128, 127, (H, W))

    lab2_np[..., 0] = np.random.uniform(0, 100, (H, W))
    lab2_np[..., 1] = np.random.uniform(-128, 127, (H, W))
    lab2_np[..., 2] = np.random.uniform(-128, 127, (H, W))

    # skimageでの計算
    deltaE_skimage = deltaE_ciede2000(lab1_np, lab2_np)

    # PyTorchでの計算
    lab1_t = torch.from_numpy(lab1_np)
    lab2_t = torch.from_numpy(lab2_np)

    deltaE_torch = deltaE_ciede2000_torch(lab1_t, lab2_t).numpy()

    # 差分確認
    diff = np.abs(deltaE_skimage - deltaE_torch)
    print("Max difference:", diff.max())
    print("Mean difference:", diff.mean())
