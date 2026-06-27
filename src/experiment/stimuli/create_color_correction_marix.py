import numpy as np

# =============================================================
# 入力: 各行が R, G, B / 各列が Y(輝度), x, y の 3x3 行列
# 計測値をここに入れてください
# =============================================================
BG  = np.array([   # 1. BG  (生背景)
    [39.73, 0.6427, 0.3282],   # R: (Y, x, y)
    [150.57, 0.3074, 0.5959],   # G: (Y, x, y)
    [12.73,  0.1546, 0.052],   # B: (Y, x, y)
])
BGT = np.array([   # 2. BGT (背景の透過)
    [7.17, 0.6289, 0.3252],
    [31.60, 0.2942, 0.5892],
    [3.10, 0.1547, 0.0467],
])
FG  = np.array([   # 3. FG  (生前景)
    [30.50, 0.6570, 0.3206],
    [114.00, 0.3290, 0.6140],
    [11.17, 0.1595, 0.0496],
])
FGR = np.array([   # 4. FGR (前景の反射)
    [20.13, 0.6450, 0.3324],
    [73.50, 0.3194, 0.6238],
    [6.80, 0.1524, 0.0515],
])


def Yxy_to_XYZ(row):
    """1行 (Y, x, y) を (X, Y, Z) に変換"""
    Y, x, y = row
    if y == 0:
        return np.array([0.0, 0.0, 0.0])
    X = (x / y) * Y
    Z = ((1.0 - x - y) / y) * Y
    return np.array([X, Y, Z])


def to_XYZ_columns(M_Yxy):
    """
    行=R,G,B / 列=Y,x,y の行列を受け取り、
    各原色を XYZ に変換したうえで「列=原色(R,G,B), 行=X,Y,Z」の行列にして返す。
    （v' = M v の列ベクトル規約で扱うため転置して列に並べる）
    """
    rows_xyz = np.array([Yxy_to_XYZ(M_Yxy[i]) for i in range(3)])  # 行=R,G,B, 列=X,Y,Z
    return rows_xyz.T  # 列=R,G,B, 行=X,Y,Z


# --- XYZ へ変換（原色を列に並べた行列）---
BG_xyz  = to_XYZ_columns(BG)
BGT_xyz = to_XYZ_columns(BGT)
FG_xyz  = to_XYZ_columns(FG)
FGR_xyz = to_XYZ_columns(FGR)

# --- 変換行列 T, R を計算 ---
#   BGT_xyz = T @ BG_xyz   ->  T = BGT_xyz @ inv(BG_xyz)
#   FGR_xyz = R @ FG_xyz   ->  R = FGR_xyz @ inv(FG_xyz)
T = BGT_xyz @ np.linalg.inv(BG_xyz)
R = FGR_xyz @ np.linalg.inv(FG_xyz)

# --- T = R C を満たす C を C = R^{-1} T で導出 ---
C = np.linalg.inv(R) @ T

# =============================================================
# 結果表示
# =============================================================
np.set_printoptions(precision=6, suppress=True)
print("BG_xyz (列=R,G,B / 行=X,Y,Z):\n", BG_xyz)
print("BGT_xyz:\n", BGT_xyz)
print("FG_xyz:\n", FG_xyz)
print("FGR_xyz:\n", FGR_xyz)
print("\nT (BG -> BGT, 透過の変換行列):\n", T)
print("\nR (FG -> FGR, 反射の変換行列):\n", R)
print("\nC = R^{-1} T:\n", C)

# --- 検算 ---
print("\n[検算] R @ C と T の差(最大):", np.max(np.abs(R @ C - T)))


# =============================================================
# 追加処理: テスト色 (R, G, B + グレースケール画像) に C を掛けて補正し保存
# =============================================================
import os
import matplotlib.pyplot as plt

# --- sRGB <-> 線形RGB 変換 (IEC 61966-2-1) ---
def srgb_to_linear(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=float), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * (c ** (1 / 2.4)) )

# --- 保存先 (スクリプト内で指定。環境に合わせて書き換えてください) ---
SAVE_DIR = r"C:\Users\Hamada.MSI\Desktop\Hamada\lab\archive"

# --- テスト色を sRGB 値として定義 (各行が1色 / 列 = R, G, B / 範囲 0-1) ---
base_srgb = np.array([
    [1.0, 0.0, 0.0],   # R
    [0.0, 1.0, 0.0],   # G
    [0.0, 0.0, 1.0],   # B
    [1.0, 1.0, 1.0],   # White (255)
    [127/255.0, 127/255.0, 127/255.0], # Gray (127)
    [0.0, 0.0, 0.0],   # Black (0)
])

# --- sRGB -> 線形 にしてから行列を適用 (T, R, C は線形XYZ空間の行列) ---
base_lin = srgb_to_linear(base_srgb) # (6, 3)
t_lin = (T @ base_lin.T).T               # T を通した線形RGB (透過)
r_lin = (R @ base_lin.T).T               # R を通した線形RGB (反射)
c_lin = (C @ base_lin.T).T               # C を通した線形RGB (R^-1 T)

# --- 保存・表示用に 線形 -> sRGB へ戻す (元の色はそのまま sRGB) ---
base_colors = base_srgb
t_colors = linear_to_srgb(t_lin)
r_colors = linear_to_srgb(r_lin)
c_colors = linear_to_srgb(c_lin)

# --- 1色ずつ個別の画像として保存 (元 -> T -> R -> C の順) ---
os.makedirs(SAVE_DIR, exist_ok=True)
labels = ["R", "G", "B", "White", "Gray", "Black"]
PATCH_SIZE = 256   # 出力画像の1辺 (px)

# 保存する順番: 元のRGB -> Tを通したRGB -> Rを通したRGB -> Cを通したRGB
variants = [
    ("1_original", base_colors),
    ("2_T", t_colors),
    ("3_R", r_colors),
    ("4_C", c_colors),
]

saved_paths = []
for idx, label in enumerate(labels):
    for suffix, colors in variants:
        # 表示用に [0,1] にクリップ (行列適用で範囲外になり得るため)
        patch = np.tile(np.clip(colors[idx], 0.0, 1.0), (PATCH_SIZE, PATCH_SIZE, 1))
        path = os.path.join(SAVE_DIR, f"{label}_{suffix}.png")
        plt.imsave(path, patch)
        saved_paths.append(path)

print(f"\n[保存] {len(saved_paths)} 枚の画像を保存しました -> {SAVE_DIR}")
for p in saved_paths:
    print("  ", p)
print("base_colors:\n", base_colors)
print("t_colors:\n", t_colors)
print("r_colors:\n", r_colors)
print("c_colors:\n", c_colors)
print("最大誤差:", np.max(np.abs(R @ C @ base_lin.T - T @ base_lin.T)))

# =============================================================
# 追加処理: 2つの Yxy を Lab 空間で比較・評価 (ΔE)
#   - 行列計算は XYZ(線形)、評価のみ Lab で行う
# =============================================================
# Lab には基準白色点が必要。計測した白(背景白など)の Yxy を入れてください。
# ここで基準にした白に対する“相対的な見え”として L*a*b* が決まります。
WHITE_REF_Yxy = np.array([213.3, 0.3084, 0.3220])  # 基準白 (Y, x, y) ※色度は D65 の例

# 比較したい 2 色 (Y, x, y) を入力 (例: 実測 TV と シミュレーション TV)
SAMPLE_1_Yxy = np.array([45.9, 0.2802, 0.2912])  # 例: 実測
SAMPLE_2_Yxy = np.array([45.8, 0.2934, 0.2860])  # 例: シミュレーション


def XYZ_to_Lab(XYZ, white_XYZ):
    """XYZ -> CIELAB (基準白 white_XYZ で正規化)"""
    xr, yr, zr = np.asarray(XYZ, dtype=float) / np.asarray(white_XYZ, dtype=float)
    eps = 216.0 / 24389.0      # (6/29)^3
    kappa = 24389.0 / 27.0     # (29/3)^3

    def f(t):
        return np.cbrt(t) if t > eps else (kappa * t + 16.0) / 116.0

    fx, fy, fz = f(xr), f(yr), f(zr)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.array([L, a, b])


def deltaE76(lab1, lab2):
    """CIE76 ΔE*ab (Lab のユークリッド距離)"""
    return float(np.sqrt(np.sum((np.asarray(lab1) - np.asarray(lab2)) ** 2)))


def deltaE2000(lab1, lab2, kL=1.0, kC=1.0, kH=1.0):
    """CIEDE2000 色差 (知覚均等性を考慮した色差)"""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    avg_L = (L1 + L2) / 2.0
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    avg_C = (C1 + C2) / 2.0
    G = 0.5 * (1 - np.sqrt(avg_C ** 7 / (avg_C ** 7 + 25.0 ** 7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)
    avg_Cp = (C1p + C2p) / 2.0
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    if C1p * C2p == 0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    elif h2p - h1p > 180:
        dhp = h2p - h1p - 360
    else:
        dhp = h2p - h1p + 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)
    if C1p * C2p == 0:
        avg_hp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        avg_hp = (h1p + h2p) / 2.0
    elif h1p + h2p < 360:
        avg_hp = (h1p + h2p + 360) / 2.0
    else:
        avg_hp = (h1p + h2p - 360) / 2.0
    T_ = (1 - 0.17 * np.cos(np.radians(avg_hp - 30))
          + 0.24 * np.cos(np.radians(2 * avg_hp))
          + 0.32 * np.cos(np.radians(3 * avg_hp + 6))
          - 0.20 * np.cos(np.radians(4 * avg_hp - 63)))
    d_ro = 30 * np.exp(-((avg_hp - 275) / 25.0) ** 2)
    Rc = 2 * np.sqrt(avg_Cp ** 7 / (avg_Cp ** 7 + 25.0 ** 7))
    Sl = 1 + (0.015 * (avg_L - 50) ** 2) / np.sqrt(20 + (avg_L - 50) ** 2)
    Sc = 1 + 0.045 * avg_Cp
    Sh = 1 + 0.015 * avg_Cp * T_
    Rt = -np.sin(np.radians(2 * d_ro)) * Rc
    dE = np.sqrt(
        (dLp / (kL * Sl)) ** 2
        + (dCp / (kC * Sc)) ** 2
        + (dHp / (kH * Sh)) ** 2
        + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh))
    )
    return float(dE)


# --- Yxy -> XYZ -> Lab に変換して評価 (Yxy_to_XYZ は上で定義済み) ---
white_XYZ = Yxy_to_XYZ(WHITE_REF_Yxy)
xyz1 = Yxy_to_XYZ(SAMPLE_1_Yxy)
xyz2 = Yxy_to_XYZ(SAMPLE_2_Yxy)
lab1 = XYZ_to_Lab(xyz1, white_XYZ)
lab2 = XYZ_to_Lab(xyz2, white_XYZ)

print("\n==== Lab 空間での評価 ====")
print("基準白 (Y,x,y):", WHITE_REF_Yxy, "-> XYZ:", white_XYZ)
print("Sample1 (Y,x,y):", SAMPLE_1_Yxy, "-> XYZ:", xyz1, "-> Lab:", lab1)
print("Sample2 (Y,x,y):", SAMPLE_2_Yxy, "-> XYZ:", xyz2, "-> Lab:", lab2)
print("ΔL*={:.3f}  Δa*={:.3f}  Δb*={:.3f}".format(*(lab2 - lab1)))
print("ΔE76  (CIE76)    :", round(deltaE76(lab1, lab2), 3))
print("ΔE00  (CIEDE2000):", round(deltaE2000(lab1, lab2), 3))
# 目安: ΔE00 < 1 はほぼ知覚不能, 1-2 はよく見れば分かる, >3-5 で明確に分かる