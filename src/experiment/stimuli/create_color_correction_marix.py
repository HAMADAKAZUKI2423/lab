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

# =============================================================
# After2: チャネル別EOTF g を実測ランプから構築 (g(1)=1 正規化)
#   - R,G,B 各チャネルの階調ランプ(255~0)の Yxy を記録
#   - 各チャネルの輝度Yをフルスケールで割って g を正規化 (g(0)=0, g(1)=1)
#   - 背景パス=透過(T·Db)用 g_b, 前景パス=反射(R·Df)用 g_f を別々に作る
#   - 補正: v_fg = g_f^{-1}( C · g_b(v) )   ※C は従来どおり R''^{-1} T''
# 記録フォーマット: (画素値0-255, Y, x, y)。未計測は None。
#   ※255行はフルスケール原色(BGT/FGR)と一致するはず。0行は Y=0。
#   ※xy は「階調を通して一定か」のチェック用。Y のみが g に効く。
# =============================================================
RAMP_BG = {  # 背景パス (T·Db) の階調ランプ
    "R": [
        (255, 7.5, 0.6235, 0.3300),
        (224, 6.0, 0.6183, 0.3278),
        (192, 4.4, 0.6173, 0.3245),
        (160, 3.0, 0.6049, 0.3191),
        (128, 1.9, 0.5671, 0.3127),
        (96,  1.2, 0.5391, 0.3032),
        (64,  0.6, 0.4342, 0.2782),
        (32,  0.3, 0.2983, 0.2362),
        (0,   0.2, 0.2717, 0.2359),
    ],
    "G": [
        (255, 32.6, 0.3021, 0.6246),
        (224, 25.5, 0.3019, 0.6202),
        (192, 18.8, 0.3015, 0.6154),
        (160, 12.6, 0.2986, 0.6140),
        (128, 7.9, 0.2971, 0.5993),
        (96,  4.3, 0.3010, 0.5833),
        (64,  2.0, 0.2945, 0.5139),
        (32,  0.6, 0.2620, 0.3997),
        (0,   0.2, 0.2576, 0.2164),
    ],
    "B": [
        (255, 3.5, 0.1570, 0.0456),
        (224, 2.8, 0.1572, 0.0459),
        (192, 2.1, 0.1536, 0.0469),
        (160, 1.6, 0.1562, 0.0498),
        (128, 1.0, 0.1558, 0.0521),
        (96,  0.7, 0.1599, 0.0634),
        (64,  0.4, 0.1871, 0.0850),
        (32,  0.3, 0.2187, 0.1561),
        (0,   0.2, 0.2667, 0.1960),
    ],
}
RAMP_FG = {  # 前景パス (R·Df) の階調ランプ
    "R": [
        (255, 20.4, 0.6436, 0.3312),
        (224, 16.0, 0.6411, 0.3311),
        (192, 11.7, 0.6407, 0.3309),
        (160, 8.0, 0.6271, 0.3291),
        (128, 5.0, 0.6258, 0.3263),
        (96,  2.8, 0.6216, 0.3179),
        (64,  1.3, 0.5439, 0.3155),
        (32,  0.5, 0.3936, 0.3120),
        (0,   0.3,  0.2753, 0.2479),
    ],
    "G": [
        (255, 74.1, 0.3181, 0.6227),
        (224, 57.9, 0.3186, 0.6216),
        (192, 42.1, 0.3179, 0.6226),
        (160, 28.5, 0.3199, 0.6181),
        (128, 17.7, 0.3134, 0.6154),
        (96,  9.4, 0.3114, 0.6038),
        (64,  4.1, 0.2811, 0.5819),
        (32,  1.1, 0.2856, 0.4658),
        (0,   0.2,  0.3005, 0.1945),
    ],
    "B": [
        (255, 7.2, 0.1528, 0.0519),
        (224, 5.7, 0.1532, 0.0519),
        (192, 4.3, 0.1536, 0.0530),
        (160, 3.0, 0.1564, 0.0529),
        (128, 1.9, 0.1552, 0.0550),
        (96,  1.1, 0.1624, 0.0601),
        (64,  0.6, 0.1714, 0.0739),
        (32,  0.3, 0.1813, 0.1073),
        (0,   0.3,  0.2485, 0.2818),
    ],
}
CHANNELS = ("R", "G", "B")

def build_eotf(ramp_channel):
    """1チャネルのランプ [(v255,Y,x,y),...] から正規化EOTF を作る。
       画素値(0-1) <-> 正規化線形(0-1)。Y をフルスケールで割って g(1)=1。
       未計測(None)はスキップ。点が2つ(0と255)だけなら線形になる。"""
    pts = [(p[0] / 255.0, p[1]) for p in ramp_channel if p[1] is not None]
    pts.sort(key=lambda t: t[0])
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    y_full = ys[np.argmax(xs)]          # フルスケールの輝度
    yn = ys / y_full                    # g(1)=1 へ正規化
    def g(v):                           # 画素値 -> 正規化線形
        return np.interp(np.asarray(v, dtype=float), xs, yn)
    def g_inv(y):                       # 正規化線形 -> 画素値
        return np.interp(np.asarray(y, dtype=float), yn, xs)
    return g, g_inv

# 各チャネルのEOTFを構築 (前景・背景で別々)
_gb = {c: build_eotf(RAMP_BG[c]) for c in CHANNELS}
_gf = {c: build_eotf(RAMP_FG[c]) for c in CHANNELS}

def _apply(funcs, arr, idx):
    arr = np.asarray(arr, dtype=float)
    out = np.empty_like(arr)
    for i, c in enumerate(CHANNELS):
        out[..., i] = funcs[c][idx](arr[..., i])
    return out

g_b     = lambda v: _apply(_gb, v, 0)   # 背景画素値 -> 正規化線形
g_f     = lambda v: _apply(_gf, v, 0)   # 前景画素値 -> 正規化線形
g_f_inv = lambda l: _apply(_gf, l, 1)   # 正規化線形 -> 前景画素値

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
os.makedirs(SAVE_DIR, exist_ok=True)

# =============================================================
# 【最初に実行】階調ランプ計測用画像の生成・保存
#   R/G/B 各チャネル × 各階調レベル(RAMP_BGと同じ) の単色ベタ画像を出力。
#   これを背景・前景にそれぞれ表示して輝度色彩計で Yxy を計測し、
#   RAMP_BG / RAMP_FG の None を埋める。
#   ファイル名: ramp_<channel>_<level3桁>.png  (level は 0-255 の画素値)
#   ※画素値をそのまま出したいのでカラーマネジメント/ガンマ補正OFFのビューアで表示すること。
# =============================================================
RAMP_DIR = os.path.join(SAVE_DIR, "ramp_patches")
os.makedirs(RAMP_DIR, exist_ok=True)
RAMP_LEVELS = [p[0] for p in RAMP_BG["R"]]   # レベルは RAMP 定義と一致させる
_ch_index = {"R": 0, "G": 1, "B": 2}
PATCH_RAMP = 512
ramp_paths = []
for _ch in CHANNELS:
    for _lv in RAMP_LEVELS:
        _rgb = np.zeros(3, dtype=float)
        _rgb[_ch_index[_ch]] = _lv / 255.0     # 該当チャネルにのみ画素値 _lv
        _patch = np.tile(_rgb, (PATCH_RAMP, PATCH_RAMP, 1))
        _path = os.path.join(RAMP_DIR, f"ramp_{_ch}_{_lv:03d}.png")
        plt.imsave(_path, np.clip(_patch, 0.0, 1.0))
        ramp_paths.append(_path)
print(f"[ランプ画像] {len(ramp_paths)} 枚 -> {RAMP_DIR}")

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

# --- 実測EOTF g_b で線形化 (After2: 旧 srgb_to_linear を置換) ---
# base_srgb は「背景に出す画素値(0-1)」とみなす
base_lin = g_b(base_srgb)                    # 背景画素値 -> 正規化線形 (g_b)

# --- 補正後の前景駆動値 (RGB->RGB)【主成果物】---
c_lin = (C @ base_lin.T).T                   # C = R''^{-1} T'' (従来と同一)
c_colors = g_f_inv(c_lin)                    # 正規化線形 -> 前景画素値 (g_f^{-1}, 旧 linear_to_srgb)

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
