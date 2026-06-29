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
    ("R",     (1.0, 0.0, 0.0),        (7.5, 0.6235, 0.3300), (20.4, 0.6436, 0.3312)),
    ("G",     (0.0, 1.0, 0.0),        (32.6, 0.3021, 0.6246), (74.1, 0.3181, 0.6227)),
    ("B",     (0.0, 0.0, 1.0),        (3.5, 0.1570, 0.0456), (7.2, 0.1528, 0.0519)),
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
    "R": 1.0, "G": 2.0, "B": 3.0, "W": 2.0, "semiR": 1.0, "semiG": 2.0, "semiB": 3.0, "Gray": 2.0,
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

# =============================================================
# 階調ランプ補正(After2): チャネル別EOTF g を実測ランプから構築 (g(1)=1)
#   - 旧 srgb_to_linear/linear_to_srgb の仮定ガンマを実測EOTFに置換
#   - 背景パス=透過(T·Db)用 g_b / 前景パス=反射(R·Df)用 g_f を別々に作る
#   - 行列フィットの線形化にも g を使い、モデル XYZ = M·g(v) を一貫させる
#   - 補正: v_fg = g_f^{-1}( C · g_b(v) )
# 記録フォーマット: (画素値0-255, Y, x, y)。Y のみが g に効く。
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
       画素値(0-1) <-> 正規化線形(0-1)。Y をフルスケールで割って g(1)=1。"""
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

def collect(channel):
    """channel='BGT' or 'FGR' の有効パッチを (rgb_lin, XYZ, w, names) で返す
       線形化は実測EOTF(g_b/g_f)を使用 (旧 srgb_to_linear を置換)"""
    idx = 2 if channel == "BGT" else 3
    lin_fn = g_b if channel == "BGT" else g_f   # 背景=g_b / 前景=g_f
    rgb_lin, XYZ, w, names = [], [], [], []
    for p in PATCHES:
        meas = p[idx]
        if meas is None or any(v is None for v in meas):
            continue  # 未測定はスキップ
        rgb_lin.append(lin_fn(np.asarray(p[1], dtype=float)))
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
    [0.0, 1.0, 1.0],            # Cyan
    [1.0, 0.0, 1.0],            # Magenta
    [1.0, 1.0, 0.0],            # Yellow
    [0.5, 0.0, 0.0],            # semiR
    [0.0, 0.5, 0.0],            # semiG
    [0.0, 0.0, 0.5]             # semiB
])
labels = ["R", "G", "B", "White", "Gray", "Black", "Cyan", "Magenta", "Yellow", "semiR", "semiG", "semiB"]

# --- 背景画素値 -> 正規化線形 (After2: 実測EOTF g_b で線形化) ---
base_lin = g_b(base_srgb)                    # (N,3) 正規化線形RGB

# --- 補正後の前景駆動値 (RGB->RGB)【主成果物】---
c_lin = (C @ base_lin.T).T                   # 前景に送る正規化線形RGB
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

# =============================================================
# 追加処理: 実測の透過T と シミュレート反射R の色差 (Yxy入力)
#   W, R, G, B, C, M, Y, Gray の 8 色について
#   T(Y,x,y) と R(Y,x,y) を入力し、CIEDE2000 等で色差を算出
# =============================================================
import math

# ---- 入力リスト: name, T実測(Y,x,y), R(シミュレート)(Y,x,y) ----
#   未測定/未算出は None のままにすればスキップされる
COMPARE = [
    # name     T (Y, x, y)             R (Y, x, y)
    ("W",     (43.9, 0.2794, 0.2904), (47.4, 0.2816, 0.2978)),
    ("R",     (7.5, 0.6323, 0.3285), (9.0, 0.5721, 0.3316)),
    ("G",     (32.4, 0.3033, 0.6244), (33.3, 0.3172, 0.6208)),
    ("B",     (3.6, 0.1550, 0.0455), (4.4, 0.1578, 0.0567)),
    ("C",     (36.0, 0.2161, 0.2836), (37.8, 0.2204, 0.2902)),
    ("M",     (10.9, 0.2673, 0.1101), (12.8, 0.2692, 0.1239)),
    ("Y",     (40.2, 0.4055, 0.5366), (42.9, 0.3987, 0.5282)),
    ("Gray",  (11.2, 0.2795, 0.2833), (11.3, 0.2740, 0.2862)),
]

# ---- 白色点の扱い ----
# Lab は基準白(ホワイトポイント)を必要とする。
# T と R は明るさのスケールが違う(反射の方が明るい)ので、
# それぞれの "W" の XYZ を各セットの白として正規化(Yw=100)してから比較する。
# → 絶対輝度差ではなく、白を揃えた上での色ズレを評価する。
# 共通白(例: D65)で評価したい場合は USE_OWN_WHITE = False にする。
WHITE_XYZ = np.array([102.0, 105.1, 120.3])  # 共通白を使う場合

def _get_white_xyz(channel_idx):
    """COMPARE 内の 'W' から、指定チャンネルの白XYZを返す (Yw=100に正規化)"""
    for name, t, r in COMPARE:
        if name == "W":
            row = t if channel_idx == 1 else r
            if row is None or any(v is None for v in row):
                return None
            xyz = Yxy_to_XYZ(row)
            if xyz[1] == 0:
                return None
            return xyz / xyz[1] * 100.0  # Yw=100 に正規化
    return None

def _f_lab(t):
    delta = 6.0 / 29.0
    return np.where(t > delta**3, np.cbrt(t), t / (3 * delta**2) + 4.0 / 29.0)

def XYZ_to_Lab(xyz, white_xyz):
    """XYZ -> CIE Lab* (white_xyz は Yw=100 スケール)"""
    xr, yr, zr = xyz / white_xyz
    fx, fy, fz = _f_lab(xr), _f_lab(yr), _f_lab(zr)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.array([L, a, b])

def delta_e_76(Lab1, Lab2):
    """単純なユークリッド色差 ΔE*ab (CIE1976)"""
    return float(np.linalg.norm(np.asarray(Lab1) - np.asarray(Lab2)))

def ciede2000(Lab1, Lab2, kL=1.0, kC=1.0, kH=1.0):
    """CIEDE2000 色差"""
    L1, a1, b1 = Lab1
    L2, a2, b2 = Lab2
    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    Cbar = (C1 + C2) / 2.0
    G = 0.5 * (1 - math.sqrt(Cbar**7 / (Cbar**7 + 25.0**7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = math.hypot(a1p, b1)
    C2p = math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360.0
    dLp = L2 - L1
    dCp = C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    else:
        diff = h2p - h1p
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        dhp = diff
    dHp = 2.0 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp) / 2.0)
    Lbar = (L1 + L2) / 2.0
    Cbarp = (C1p + C2p) / 2.0
    if C1p * C2p == 0:
        hbarp = h1p + h2p
    else:
        if abs(h1p - h2p) <= 180:
            hbarp = (h1p + h2p) / 2.0
        elif (h1p + h2p) < 360:
            hbarp = (h1p + h2p + 360) / 2.0
        else:
            hbarp = (h1p + h2p - 360) / 2.0
    T = (1 - 0.17 * math.cos(math.radians(hbarp - 30)) + 0.24 * math.cos(math.radians(2 * hbarp)) + 0.32 * math.cos(math.radians(3 * hbarp + 6)) - 0.20 * math.cos(math.radians(4 * hbarp - 63)))
    d_theta = 30 * math.exp(-(((hbarp - 275) / 25)**2))
    Rc = 2 * math.sqrt(Cbarp**7 / (Cbarp**7 + 25.0**7))
    Sl = 1 + (0.015 * (Lbar - 50)**2) / math.sqrt(20 + (Lbar - 50)**2)
    Sc = 1 + 0.045 * Cbarp
    Sh = 1 + 0.015 * Cbarp * T
    Rt = -math.sin(math.radians(2 * d_theta)) * Rc
    dE = math.sqrt((dLp / (kL * Sl))**2 + (dCp / (kC * Sc))**2 + (dHp / (kH * Sh))**2 + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh)))
    return float(dE)

# ===================== 色差の算出 =====================
white_T = white_R = WHITE_XYZ

print("\n==== T(透過・実測) vs R(反射・シミュレート) 色差 ====")
print(f"{'color':>6} {'ΔE00':>8} {'ΔE76':>8} {'L_T':>7} {'L_R':>7} {'ΔL':>7}")
dE00_list = []
for name, t_row, r_row in COMPARE:
    if name == "W" and (t_row is None or r_row is None or any(v is None for v in t_row) or any(v is None for v in r_row)):
        continue
    if t_row is None or r_row is None or any(v is None for v in t_row) or any(v is None for v in r_row):
        continue
    if white_T is None or white_R is None:
        print(" 'W' の T/R が未入力のため、自分白での正規化ができません。USE_OWN_WHITE=False にするか W を入力してください。")
        break

    xyz_T = Yxy_to_XYZ(t_row)
    xyz_R = Yxy_to_XYZ(r_row)

    Lab_T = XYZ_to_Lab(xyz_T, white_T)
    Lab_R = XYZ_to_Lab(xyz_R, white_R)

    dE00 = ciede2000(Lab_T, Lab_R)
    dE76 = delta_e_76(Lab_T, Lab_R)
    dE00_list.append((name, dE00))

    print(f"{name:>6} {dE00:8.3f} {dE76:8.3f} {Lab_T[0]:7.2f} {Lab_R[0]:7.2f} {Lab_R[0]-Lab_T[0]:7.2f}")

if dE00_list:
    avg = sum(d for _, d in dE00_list) / len(dE00_list)
    worst = max(dE00_list, key=lambda x: x[1])
    print(f"\n平均 ΔE00 = {avg:.3f} / 最大 ΔE00 = {worst[1]:.3f} ({worst[0]})")
