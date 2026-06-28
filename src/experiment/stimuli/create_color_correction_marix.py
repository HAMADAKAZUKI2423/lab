import numpy as np
import os
import matplotlib.pyplot as plt

# =============================================================
# After方針: C = (R·Df)^(-1)(T·Db) = Df^(-1) R^(-1) T Db
#   計測は純色フルスケール駆動 -> 入力原色 O = 単位行列 I
#   生背景BG・生前景FG は不要。BGT(透過) と FGR(反射) のみ使用。
# 入力: 各行が R,G,B / 各列が Y(輝度),x,y の 3x3 行列
# =============================================================
BGT = np.array([   # 背景の透過 (= T·Db·O)
    [8.00,  0.6327, 0.3286],   # R
    [34.36, 0.3097, 0.6264],   # G
    [3.60,  0.1543, 0.0457],   # B
])
FGR = np.array([   # 前景の反射 (= R·Df·O)
    [19.90, 0.6448, 0.3320],   # R
    [71.80, 0.3191, 0.6231],   # G
    [6.97,  0.1525, 0.0527],   # B
])

# 入力原色 O: 純色RGBをそのまま基底として使う -> 単位行列
O = np.eye(3)


def Yxy_to_XYZ(row):
    """1行 (Y, x, y) を (X, Y, Z) に変換"""
    Y, x, y = row
    if y == 0:
        return np.array([0.0, 0.0, 0.0])
    X = (x / y) * Y
    Z = ((1.0 - x - y) / y) * Y
    return np.array([X, Y, Z])


def to_XYZ_columns(M_Yxy):
    """行=R,G,B / 列=Y,x,y を受け取り、列=原色(R,G,B)/行=X,Y,Z にして返す"""
    rows_xyz = np.array([Yxy_to_XYZ(M_Yxy[i]) for i in range(3)])
    return rows_xyz.T


BGT_xyz = to_XYZ_columns(BGT)
FGR_xyz = to_XYZ_columns(FGR)

# T' = T·Db = BGT_xyz @ inv(O),  R' = R·Df = FGR_xyz @ inv(O)
T_prime = BGT_xyz @ np.linalg.inv(O)
R_prime = FGR_xyz @ np.linalg.inv(O)

# C = R'^(-1) T' = (R·Df)^(-1)(T·Db) = Df^(-1) R^(-1) T Db
C = np.linalg.inv(R_prime) @ T_prime

np.set_printoptions(precision=6, suppress=True)
print("BGT_xyz:\n", BGT_xyz)
print("FGR_xyz:\n", FGR_xyz)
print("T_prime (=T·Db):\n", T_prime)
print("R_prime (=R·Df):\n", R_prime)
print("C = inv(R_prime) @ T_prime:\n", C)
print("[検算] max|R'C - T'| =", np.max(np.abs(R_prime @ C - T_prime)))

# =============================================================
# 追加処理(After): C補正画像の生成と「見え」の照合
#   T_prime = T·Db, R_prime = R·Df は RGB->XYZ、C は RGB->RGB
# =============================================================

# --- sRGB <-> 線形RGB 変換 (IEC 61966-2-1) ---
def srgb_to_linear(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=float), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * (c ** (1 / 2.4)) - 0.055)  # -0.055 を復活

# --- XYZ(D65) -> linear sRGB 標準行列 ---
M_XYZ2RGB = np.array([
    [ 3.2406, -1.5372, -0.4986],
    [-0.9689,  1.8758,  0.0415],
    [ 0.0557, -0.2040,  1.0570],
])

SAVE_DIR = r"C:\Users\Hamada.MSI\Desktop\Hamada\lab\archive"

# --- テスト色 (各行が1色 / 列=R,G,B / 0-1) ---
base_srgb = np.array([
    [1.0, 0.0, 0.0],            # R
    [0.0, 1.0, 0.0],            # G
    [0.0, 0.0, 1.0],            # B
    [1.0, 1.0, 1.0],            # White
    [127/255.0, 127/255.0, 127/255.0],  # Gray
    [0.0, 0.0, 0.0],            # Black
    [1.0, 1.0, 0.0],            # Yellow
    [1.0, 0.0, 1.0],            # Magenta
    [0.0, 1.0, 1.0],            # Cyan
])
labels = ["R", "G", "B", "White", "Gray", "Black", "Yellow", "Magenta", "Cyan"]

# --- sRGB -> 線形 ---
base_lin = srgb_to_linear(base_srgb)         # (9,3) 線形RGB

# --- 補正後の前景駆動値 (RGB->RGB)【主成果物】---
c_lin = (C @ base_lin.T).T                   # 前景に送る線形RGB
c_colors = linear_to_srgb(c_lin)             # 保存用 sRGB

# --- 「見え」(XYZ) の計算 ---
trans_xyz = (T_prime @ base_lin.T).T        # 透過背景の見え (ターゲット)
refl_xyz = (R_prime @ c_lin.T).T           # 補正前景の反射の見え (結果)

# --- 見えプレビュー: 白Yで正規化して XYZ->sRGB ---
white_Y = trans_xyz[labels.index("White"), 1]
def xyz_to_srgb_img(xyz, wY):
    lin = (M_XYZ2RGB @ (xyz / wY).T).T
    return linear_to_srgb(lin)
trans_view = xyz_to_srgb_img(trans_xyz, white_Y)
refl_view = xyz_to_srgb_img(refl_xyz, white_Y)

# --- 保存 ---
os.makedirs(SAVE_DIR, exist_ok=True)
PATCH = 256
variants = [
    ("1_original",      base_srgb),    # 元のRGB
    ("2_corrected_fg",  c_colors),     # C補正後 前景画像(投影用)
    ("3_trans_view",   trans_view),  # 透過背景の見え
    ("4_refl_view",   refl_view),  # 補正前景の反射の見え
]
saved_paths = []
for idx, label in enumerate(labels):
    for suffix, colors in variants:
        patch = np.tile(np.clip(colors[idx], 0.0, 1.0), (PATCH, PATCH, 1))
        path = os.path.join(SAVE_DIR, f"{label}_{suffix}.png")
        plt.imsave(path, patch)
        saved_paths.append(path)

print(f"\n[保存] {len(saved_paths)} 枚 -> {SAVE_DIR}")
print("c_colors (補正後前景 sRGB):\n", c_colors)

# --- 検証 ---
print("最大誤差(XYZ):", np.max(np.abs(refl_xyz - trans_xyz)))
# 各色の trans_xyz vs refl_xyz を deltaE2000 で評価すれば
# グレーの紫転びが解消したかを数値で確認できる。

# =============================================================
# 追加処理: 2つの Yxy を Lab 空間で比較・評価 (ΔE)
#   - 行列計算は XYZ(線形)、評価のみ Lab で行う
# =============================================================
# Lab には基準白色点が必要。計測した白(背景白など)の Yxy を入れてください。
# ここで基準にした白に対する“相対的な見え”として L*a*b* が決まります。
WHITE_REF_Yxy = np.array([213.3, 0.3084, 0.3220])  # 基準白 (Y, x, y) ※色度は D65 の例

# 比較したい 2 色 (Y, x, y) を入力 (例: 実測 TV と シミュレーション TV)
SAMPLE_1_Yxy = np.array([39.73, 0.6427, 0.3282])  # 例: 実測
SAMPLE_2_Yxy = np.array([30.5, 0.657, 0.3206])  # 例: シミュレーション


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