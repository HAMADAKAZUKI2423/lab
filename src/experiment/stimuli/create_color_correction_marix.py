import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np

# =============================================================
# 最小二乗版: T', R' を「純色 + 白 + CMY + グレー」から推定
#   モデル: XYZ = M @ rgb_lin   (M = T'(透過) または R'(反射), 3x3)
#   劣加法(混色で輝度過大)を平均的に吸収させるのが狙い。
#   ※ 純色のみだと白/混色は span 内で情報ゼロ。混色を足して初めて効く。
# =============================================================

# ---- 入力パッチ: name, sRGB(0-1), BGT実測(Y,x,y), FGR実測(Y,x,y) ----
#  純色3つは元データ。White/CMY/Gray の実測値を埋めて使う。
PATCHES = [
    # name      sRGB(R,G,B)             BGT (Y, x, y)          FGR (Y, x, y)
    ("R",     (1.0, 0.0, 0.0),        (8.00,  0.6327, 0.3286), (19.90, 0.6448, 0.3320)),
    ("G",     (0.0, 1.0, 0.0),        (34.36, 0.3097, 0.6264), (71.80, 0.3191, 0.6231)),
    ("B",     (0.0, 0.0, 1.0),        (3.60,  0.1543, 0.0457), (6.97,  0.1525, 0.0527)),
    # --- ここから追加測定（実測値を入れる）---
    ("W",     (1.0, 1.0, 1.0),        (46.7, 0.2803, 0.2921),      (101.4, 0.3130, 0.3222)),
    ("semiR",     (0.5, 0.0, 0.0),        (2.1, 0.5723, 0.3091),      (4.8, 0.6284, 0.3214)),
    ("semiG",     (0.0, 0.5, 0.0),        (8.3, 0.3003, 0.6163),      (17.2, 0.3181, 0.6094)),
    ("semiB",     (0.0, 0.0, 0.5),        (1.1, 0.1541, 0.0521),      (1.9, 0.1540, 0.0567)),
    ("Gray",  (0.5, 0.5, 0.5),        (12.0, 0.2795, 0.2852),      (25.3, 0.3084, 0.3169)),
]

# ---- パッチごとの重み（輝度が最重要なら効かせたいパッチを大きく）----
# 例: 混色・白を重めにして劣加法をしっかり吸わせる
WEIGHTS = {
    "R": 1.0, "G": 1.0, "B": 1.0, "W": 1.0, "semiR": 1.0, "semiG": 1.0, "semiB": 1.0, "Gray": 1.0,
}

# ---- sRGB -> 線形RGB ----
def srgb_to_linear(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.0)

def linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=float), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * (c ** (1 / 2.0)) - 0.055)

def Yxy_to_XYZ(row):
    Y, x, y = row
    if y == 0:
        return np.array([0.0, 0.0, 0.0])
    X = (x / y) * Y
    Z = ((1.0 - x - y) / y) * Y
    return np.array([X, Y, Z])

def collect(channel):
    """channel='BGT' or 'FGR' の有効パッチを (rgb_lin, XYZ, w, names) で返す"""
    idx = 2 if channel == "BGT" else 3
    rgb_lin, XYZ, w, names = [], [], [], []
    for p in PATCHES:
        meas = p[idx]
        if meas is None or any(v is None for v in meas):
            continue  # 未測定はスキップ
        rgb_lin.append(srgb_to_linear(p[1]))
        XYZ.append(Yxy_to_XYZ(meas))
        w.append(WEIGHTS.get(p[0], 1.0))
        names.append(p[0])
    return np.array(rgb_lin), np.array(XYZ), np.array(w), names

def fit_matrix(rgb_lin, XYZ, weights=None):
    """
    XYZ = M @ rgb_lin を最小二乗で解く (M: 3x3)
    行形式: rgb_lin(N,3) @ M.T = XYZ(N,3)
    """
    A, Bm = rgb_lin, XYZ
    if weights is not None:
        s = np.sqrt(weights).reshape(-1, 1)
        A, Bm = A * s, Bm * s
    Mt, *_ = np.linalg.lstsq(A, Bm, rcond=None)  # A @ Mt = Bm -> Mt = M.T
    return Mt.T

# ===================== 推定 =====================
rgb_T, XYZ_T, w_T, names_T = collect("BGT")
rgb_R, XYZ_R, w_R, names_R = collect("FGR")

T_prime = fit_matrix(rgb_T, XYZ_T, w_T)   # T·Db
R_prime = fit_matrix(rgb_R, XYZ_R, w_R)   # R·Df
C = np.linalg.inv(R_prime) @ T_prime      # = (R·Df)^(-1)(T·Db)

np.set_printoptions(precision=6, suppress=True)
print("T_prime:\n", T_prime)
print("R_prime:\n", R_prime)
print("C = inv(R') @ T':\n", C)

# =============================================================
# 追加処理(After): C補正画像の生成と「見え」の照合
#   T_prime = T·Db, R_prime = R·Df は RGB->XYZ、C は RGB->RGB
# =============================================================

# --- sRGB <-> 線形RGB 変換 (IEC 61966-2-1) ---

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
    [0.5, 0.0, 0.0],            # R/2
    [0.0, 0.5, 0.0],            # G/2
    [0.0, 0.0, 0.5],            # B/2
])
labels = ["R", "G", "B", "White", "Gray", "Black", "semiR", "semiG", "semiB"]

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

# ===================== 当てはめ残差の評価 =====================
# ※ 最小二乗では厳密解でないので、max|R'C-T'| ではなく
#   「モデル予測 vs 実測」の残差を見る（特に輝度Y）。
def report(channel, M, rgb_lin, XYZ_meas, names):
    print(f"\n==== {channel} 残差 (モデル予測 vs 実測) ====")
    print(f"{'patch':>6} {'Y_meas':>8} {'Y_pred':>8} {'ΔY':>7}")
    for n, rgb, xyz in zip(names, rgb_lin, XYZ_meas):
        pred = M @ rgb
        dY = pred[1] - xyz[1]
        print(f"{n:>6} {xyz[1]:8.2f} {pred[1]:8.2f} {dY:+7.2f}")

report("BGT(T')", T_prime, rgb_T, XYZ_T, names_T)
report("FGR(R')", R_prime, rgb_R, XYZ_R, names_R)