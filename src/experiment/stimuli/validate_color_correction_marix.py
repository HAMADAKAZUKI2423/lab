# =============================================================
# コード2: 検証パート（独立実行版）
#   コード1と独立して実行できるよう、必要な定義をここで再掲する。
#   行列 R'/T'/C はコード1が保存した CSV から読み込む
#   （コード1を先に一度実行して results/tables/DisplayBrightness/*.csv を生成しておくこと）
#   ※ 階調解析プロットはコード1で生成済みのため、ここでは plot_dir=None で省略する。
# =============================================================
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---- 入力パッチ: name, sRGB(0-1), BGT実測(Y,x,y), FGR実測(Y,x,y) ----
PATCHES = [
    ("R",     (1.0, 0.0, 0.0),        (8.00,  0.6327, 0.3286), (19.90, 0.6448, 0.3320)),
    ("G",     (0.0, 1.0, 0.0),        (34.36, 0.3097, 0.6264), (71.80, 0.3191, 0.6231)),
    ("B",     (0.0, 0.0, 1.0),        (3.60,  0.1543, 0.0457), (6.97,  0.1525, 0.0527)),
    ("W",     (1.0, 1.0, 1.0),        (46.7, 0.2803, 0.2921),      (101.4, 0.3130, 0.3222)),
    ("semiR",     (0.5, 0.0, 0.0),        (2.1, 0.5723, 0.3091),      (4.8, 0.6284, 0.3214)),
    ("semiG",     (0.0, 0.5, 0.0),        (8.3, 0.3003, 0.6163),      (17.2, 0.3181, 0.6094)),
    ("semiB",     (0.0, 0.0, 0.5),        (1.1, 0.1541, 0.0521),      (1.9, 0.1540, 0.0567)),
    ("Gray",  (0.5, 0.5, 0.5),        (12.0, 0.2795, 0.2852),      (25.3, 0.3084, 0.3169)),
]
WEIGHTS = {
    "R": 1.0, "G": 1.0, "B": 1.0, "W": 1.0, "semiR": 1.0, "semiG": 1.0, "semiB": 1.0, "Gray": 1.0,
}

# ---- 色変換ユーティリティ ----
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

# ---- 実測EOTF g の構築に使う階調ランプ ----
CHANNELS = ("R", "G", "B")

RAMP_BG = {
    "R": [(255,7.5,0.6235,0.3300),(224,6.0,0.6183,0.3278),(192,4.4,0.6173,0.3245),
          (160,3.0,0.6049,0.3191),(128,1.9,0.5671,0.3127),(96,1.2,0.5391,0.3032),
          (64,0.6,0.4342,0.2782),(32,0.3,0.2983,0.2362),(0,0.2,0.2717,0.2359)],
    "G": [(255,32.6,0.3021,0.6246),(224,25.5,0.3019,0.6202),(192,18.8,0.3015,0.6154),
          (160,12.6,0.2986,0.6140),(128,7.9,0.2971,0.5993),(96,4.3,0.3010,0.5833),
          (64,2.0,0.2945,0.5139),(32,0.6,0.2620,0.3997),(0,0.2,0.2576,0.2164)],
    "B": [(255,3.5,0.1570,0.0456),(224,2.8,0.1572,0.0459),(192,2.1,0.1536,0.0469),
          (160,1.6,0.1562,0.0498),(128,1.0,0.1558,0.0521),(96,0.7,0.1599,0.0634),
          (64,0.4,0.1871,0.0850),(32,0.3,0.2187,0.1561),(0,0.2,0.2667,0.1960)],
}
RAMP_FG = {
    "R": [(255,20.4,0.6436,0.3312),(224,16.0,0.6411,0.3311),(192,11.7,0.6407,0.3309),
          (160,8.0,0.6271,0.3291),(128,5.0,0.6258,0.3263),(96,2.8,0.6216,0.3179),
          (64,1.3,0.5439,0.3155),(32,0.5,0.3936,0.3120),(0,0.3,0.2753,0.2479)],
    "G": [(255,74.1,0.3181,0.6227),(224,57.9,0.3186,0.6216),(192,42.1,0.3179,0.6226),
          (160,28.5,0.3199,0.6181),(128,17.7,0.3134,0.6154),(96,9.4,0.3114,0.6038),
          (64,4.1,0.2811,0.5819),(32,1.1,0.2856,0.4658),(0,0.2,0.3005,0.1945)],
    "B": [(255,7.2,0.1528,0.0519),(224,5.7,0.1532,0.0519),(192,4.3,0.1536,0.0530),
          (160,3.0,0.1564,0.0529),(128,1.9,0.1552,0.0550),(96,1.1,0.1624,0.0601),
          (64,0.6,0.1714,0.0739),(32,0.3,0.1813,0.1073),(0,0.3,0.2485,0.2818)],
}

def _prep_ramp(ramp_channel, remove_black=True):
    """ramp [(pixel,Y,x,y),...] -> (v, yn, Y0)"""
    pts = [(p[0] / 255.0, p[1]) for p in ramp_channel if p[1] is not None]
    pts.sort(key=lambda t: t[0])
    v = np.array([p[0] for p in pts], dtype=float)
    Y = np.array([p[1] for p in pts], dtype=float)
    Y0 = Y[np.argmin(v)] if remove_black else 0.0
    Ymax = Y[np.argmax(v)]
    denom = (Ymax - Y0) if (Ymax - Y0) != 0 else 1.0
    yn = np.clip((Y - Y0) / denom, 0.0, None)
    return v, yn, Y0

def analyze_gamma(name, ramp_channel,
                  target_gamma=2.2, resid_tol=0.02, trend_tol=0.5, r2_tol=0.99,
                  plot_dir=None):
    """1チャネルの直線性チェック・ガンマ推定・モデル選択（プロットはコード1側で生成済みのため省略）。"""
    v, yn, Y0 = _prep_ramp(ramp_channel)
    resid_22 = yn - v ** target_gamma
    m = (v > 0) & (yn > 0)
    gamma, _ = curve_fit(lambda x, g: x ** g, v[m], yn[m], p0=[2.2])
    gamma = float(gamma[0])
    resid_fit = yn - v ** gamma
    ss_res = np.sum(resid_fit[m] ** 2)
    ss_tot = np.sum((yn[m] - yn[m].mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rms_resid = float(np.sqrt(np.mean(resid_fit[m] ** 2)))
    if np.std(resid_fit[m]) > 0 and np.std(v[m]) > 0:
        trend = float(abs(np.corrcoef(v[m], resid_fit[m])[0, 1]))
    else:
        trend = 0.0
    uniform = (rms_resid <= resid_tol) and (trend <= trend_tol) and (r2 >= r2_tol)
    use = "power" if uniform else "interp"
    print(f"\n---- [{name}] 直線性 & ガンマ解析 ----")
    print(f"  fit gamma = {gamma:.3f}  (target {target_gamma})  Δ={gamma-target_gamma:+.3f}")
    print(f"  R^2(log-log) = {r2:.4f} / 残差RMS = {rms_resid:.4f} / トレンド = {trend:.3f}")
    print(f"  → 残差は{'均一' if uniform else '不均一'} → 採用: "
          f"{'べき乗(v^gamma)' if use=='power' else '線形補完'}")
    return {
        "name": name, "gamma": float(gamma),
        "r2": float(r2), "rms_resid": rms_resid, "trend": trend,
        "uniform": bool(uniform), "use": use,
        "v": v, "yn": yn, "Y0": float(Y0),
        "resid_22": resid_22, "resid_fit": resid_fit,
    }

def build_eotf_auto(ramp_channel, analysis=None, **kwargs):
    """解析結果に基づき EOTF を構築して (g, g_inv, info) を返す。"""
    if analysis is None:
        name = kwargs.pop("name", "(auto)")
        analysis = analyze_gamma(name, ramp_channel, **kwargs)
    v, yn = analysis["v"], analysis["yn"]
    if analysis["use"] == "power":
        gm = analysis["gamma"]
        g     = lambda val: np.clip(np.asarray(val, dtype=float), 0.0, None) ** gm
        g_inv = lambda y:   np.clip(np.asarray(y,   dtype=float), 0.0, None) ** (1.0 / gm)
    else:
        g     = lambda val: np.interp(np.asarray(val, dtype=float), v, yn)
        g_inv = lambda y:   np.interp(np.asarray(y,   dtype=float), yn, v)
    return g, g_inv, analysis

# ---- 図の出力先 ----
FIG_DIR       = os.path.join("results", "figures", "DisplayBrightness")
PATCH_FIG_DIR = os.path.join(FIG_DIR, "patches")          # テスト色パッチ & 輝度2倍版
RAMP_FIG_DIR  = os.path.join(FIG_DIR, "ramps")            # ガンマ検証用ランプ画像
FG_ADD_DIR    = os.path.join(FIG_DIR, "foreground_add")   # 白輝度指定(前景加算)画像

# ---- チャネル別 EOTF を構築（プロットは省略: plot_dir=None）----
_gb = {}
_gf = {}
for c in CHANNELS:
    g_b, g_b_inv, _ = build_eotf_auto(RAMP_BG[c], name=f"BG_{c}", plot_dir=None)
    _gb[c] = (g_b, g_b_inv)
for c in CHANNELS:
    g_f, g_f_inv, _ = build_eotf_auto(RAMP_FG[c], name=f"FG_{c}", plot_dir=None)
    _gf[c] = (g_f, g_f_inv)

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
    """channel='BGT' or 'FGR' の有効パッチを (rgb_lin, XYZ, w, names) で返す"""
    idx = 2 if channel == "BGT" else 3
    rgb_lin, XYZ, w, names = [], [], [], []
    for p in PATCHES:
        meas = p[idx]
        if meas is None or any(v is None for v in meas):
            continue
        lin = g_b(p[1]) if channel == "BGT" else g_f(p[1])
        rgb_lin.append(lin)
        XYZ.append(Yxy_to_XYZ(meas))
        w.append(WEIGHTS.get(p[0], 1.0))
        names.append(p[0])
    return np.array(rgb_lin), np.array(XYZ), np.array(w), names

# 残差レポート用の実測パッチ（コード1と同じ）
rgb_T, XYZ_T, w_T, names_T = collect("BGT")
rgb_R, XYZ_R, w_R, names_R = collect("FGR")

# ---- コード1が保存した行列を CSV から読み込む ----
TABLE_DIR = os.path.join("results", "tables", "DisplayBrightness")

def load_matrix_csv(name):
    """results/tables/DisplayBrightness/<name>.csv を 3x3 行列として読み込む。"""
    path = os.path.join(TABLE_DIR, f"{name}.csv")
    return np.loadtxt(path, delimiter=",")   # '#' ヘッダ行は自動スキップ

T_prime = load_matrix_csv("T_prime")   # T·Db : RGB_linear -> XYZ
R_prime = load_matrix_csv("R_prime")   # R·Df : RGB_linear -> XYZ
C       = load_matrix_csv("C")         # C = inv(R') @ T'

np.set_printoptions(precision=6, suppress=True)
print("[CSV読込] R'/T'/C を読み込みました <-", os.path.abspath(TABLE_DIR))
print("T_prime:\n", T_prime)
print("R_prime:\n", R_prime)
print("C:\n", C)

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

# --- sRGB -> 線形 ---
base_lin = g_b(base_srgb)                     # ← 実測 g_b に置換（背景画素値 -> 正規化線形）

# --- 補正後の前景駆動値 (RGB->RGB)【主成果物】---
c_lin = (C @ base_lin.T).T                    # v_fg = g_f^{-1}(C_lin · g_b(v)) の内側
c_colors = g_f_inv(c_lin)                     # ← 実測 g_f^{-1} に置換（正規化線形 -> 前景画素値）

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

# --- 保存 (results/figures/DisplayBrightness/patches) ---
os.makedirs(PATCH_FIG_DIR, exist_ok=True)
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
        path = os.path.join(PATCH_FIG_DIR, f"{label}_{suffix}.png")
        plt.imsave(path, patch)
        saved_paths.append(path)

print(f"\n[保存] {len(saved_paths)} 枚 -> {PATCH_FIG_DIR}")
print("c_colors (補正後前景 sRGB):\n", c_colors)

# =============================================================
# 追加①: ガンマ確認用ランプ画像の生成・保存
#   目的: 「補正後(前景に送る駆動値)」でも Y が入力に対して
#         ガンマ2.0 に乗っているかを検証する。
#   手順: R/G/B 各チャンネルを 0~255 の複数ステップで単色駆動し、
#         本編と同じ補正(sRGB->線形-> C ->sRGB)をかけて保存。
#         これを前景に表示し、輝度色彩計で各画像の Yxy を計測する。
#   ファイル名: gamma_<channel>_<level3桁>.png (level は 0-255 の入力画素値)
#   ※画素値をそのまま出したいので、カラマネ/ガンマ補正OFFのビューアで表示すること。
# =============================================================
# ramps 配下に 生(row) と 補正後(corrected) の2種類のサブフォルダを作成
ROW_RAMP_DIR       = os.path.join(RAMP_FIG_DIR, "row_ramps")        # 生(未補正)ランプ画像: 素のガンマ測定用
CORRECTED_RAMP_DIR = os.path.join(RAMP_FIG_DIR, "corrected_ramps")  # 補正後ランプ画像: 補正後ガンマ検証用
os.makedirs(ROW_RAMP_DIR, exist_ok=True)
os.makedirs(CORRECTED_RAMP_DIR, exist_ok=True)

# 0~255 の確認ステップ(必要に応じて増減可)。0 と 255 は必ず含める。
GAMMA_LEVELS = [0, 16, 32, 48, 64, 96, 128, 160, 192, 224, 255]
_ch_index = {"R": 0, "G": 1, "B": 2}
PATCH_GAMMA = 512

def correct_srgb(rgb_srgb):
    """本編と同一の補正: sRGB(0-1) -> 線形 -> C -> sRGB(0-1)。
       C はチャンネルを混ぜるので、純色入力でも出力は混色になる(それが投影値)。"""
    lin = g_b(np.asarray(rgb_srgb, dtype=float))
    c_lin_local = (C @ np.atleast_2d(lin).T).T
    return g_f_inv(c_lin_local).reshape(np.asarray(rgb_srgb).shape)

def _ramp_rgb(ch, lv):
    """ch in ('R','G','B','W'), lv(0-255) -> 生の sRGB(0-1) ベクトル。W は R=G=B。"""
    v = lv / 255.0
    if ch == "W":
        return np.array([v, v, v], dtype=float)
    rgb = np.zeros(3, dtype=float)
    rgb[_ch_index[ch]] = v
    return rgb

# R/G/B/W 各チャンネル × 各レベルで「生」と「補正後」の2枚を出力
row_ramp_paths, corrected_ramp_paths = [], []
for _ch in ("R", "G", "B", "W"):
    for _lv in GAMMA_LEVELS:
        _rgb = _ramp_rgb(_ch, _lv)
        # (1) 生ランプ(未補正): 素のディスプレイEOTF(ガンマ)測定用
        _raw   = np.clip(_rgb, 0.0, 1.0)
        _p_raw = os.path.join(ROW_RAMP_DIR, f"ramp_{_ch}_{_lv:03d}.png")
        plt.imsave(_p_raw, np.tile(_raw, (PATCH_GAMMA, PATCH_GAMMA, 1)))
        row_ramp_paths.append(_p_raw)
        # (2) 補正後ランプ: sRGB -> 線形 -> C -> sRGB(前景駆動値)
        _corr   = np.clip(correct_srgb(_rgb), 0.0, 1.0)
        _p_corr = os.path.join(CORRECTED_RAMP_DIR, f"ramp_{_ch}_{_lv:03d}.png")
        plt.imsave(_p_corr, np.tile(_corr, (PATCH_GAMMA, PATCH_GAMMA, 1)))
        corrected_ramp_paths.append(_p_corr)

print(f"\n[生ランプ画像]   {len(row_ramp_paths)} 枚 -> {ROW_RAMP_DIR}")
print(f"[補正ランプ画像] {len(corrected_ramp_paths)} 枚 -> {CORRECTED_RAMP_DIR}")
print("  各チャンネル×各レベルの画像を前景に表示し、Yxy を計測して")
print("  下の GAMMA_MEAS に (level, Y, x, y) で記入してください。")

# =============================================================
# 追加②: 計測 Yxy からガンマを算出
#   モデル: Y(v) = Y_max * (v/255)^gamma
#   -> log(Y/Y_max) = gamma * log(v/255)  の傾きが gamma。
#   各チャンネル独立に、v>0 かつ Y>0 の点で対数最小二乗フィット。
#   目標(2.0)との差も表示する。
#   ※ここは計測後に GAMMA_MEAS を埋めてから再実行する。
# =============================================================
# 記入フォーマット: 各チャンネル [(level 0-255, Y, x, y), ...]
#   level は GAMMA_LEVELS と揃える。未計測はコメントアウト/削除で可。
GAMMA_MEAS = {
    "R": [
         (0,  0.2, 0.2476, 0.2608),
         (16,  0.2, 0.2860, 0.2123),
         (32,  0.2, 0.2596, 0.2189),
         (48,  0.5, 0.3244, 0.2763),
         (64,  0.6, 0.4792, 0.3004),
         (96,  1.3, 0.5651, 0.3083),
         (128, 1.9, 0.5825, 0.3020),
         (160, 3.0, 0.6102, 0.3292),
         (192, 4.6, 0.6214, 0.3289),
         (224, 6.2, 0.6316, 0.3288),
         (255, 7.9, 0.6299, 0.3297)
    ],
    "G": [
        (0,   0.3, 0.2074, 0.2191),
        (16,  0.3, 0.3821, 0.2310),
        (32,  0.5, 0.2505, 0.3016),
        (48,  1.3, 0.2499, 0.4777),
        (64,  1.8, 0.3280, 0.5255),
        (96,  4.4, 0.3107, 0.5672),
        (128, 8.2, 0.3126, 0.5923),
        (160, 13.1, 0.3134, 0.6074),
        (192, 19.6, 0.3166, 0.6113),
        (224, 27.3, 0.3150, 0.6220),
        (255, 34.5, 0.3182, 0.6208)
    ],
    "B": [
        (0,   0.2, 0.1210, 0.2329),
        (16,  0.3, 0.1782, 0.1192),
        (32,  0.4, 0.1878, 0.1145),
        (48,  0.4, 0.1727, 0.0941),
        (64,  0.4, 0.1826, 0.0807),
        (96,  0.8, 0.1702, 0.0630),
        (128, 1.2, 0.1581, 0.0590),
        (160, 2.0, 0.1495, 0.0550),
        (192, 2.6, 0.1549, 0.0556),
        (224, 3.3, 0.1526, 0.0230),
        (255, 4.2, 0.1531, 0.0538)
    ],
    "W": [
        (0,   0.2, 0.2827, 0.2077),
        (16,  0.3, 0.2146, 0.1294),
        (32,  0.6, 0.1820, 0.1565),
        (48,  1.6, 0.2517, 0.2538),
        (64,  2.4, 0.2601, 0.2605),
        (96,  6.2, 0.2675, 0.2756),
        (128, 11.3, 0.2714, 0.2867),
        (160, 18.3, 0.2724, 0.2759),
        (192, 27.4, 0.2813, 0.2944),
        (224, 37.5, 0.2827, 0.2996),
        (255, 47.1, 0.2827, 0.2966)
    ]
}

def estimate_gamma(meas, y_black=None):
    """[(level,Y,x,y),...] からガンマを推定。黒レベル Y_0 を差し引く。
       y_black=None のときは level 0 の実測 Y を Y_0 として使う。
       返り値: (gamma, Y_max_net, Y_0, 使用点数, R^2) / 不足なら None。"""
    pts = [(m[0], m[1]) for m in meas if m[0] is not None and m[1] is not None]
    if len(pts) < 2:
        return None
    lv = np.array([p[0] for p in pts], dtype=float)
    Y  = np.array([p[1] for p in pts], dtype=float)

    # --- 黒レベル Y_0 の決定 ---
    if y_black is None:
        zero = Y[lv == 0]
        y0 = float(zero[0]) if zero.size else 0.0   # level 0 が無ければ 0
    else:
        y0 = float(y_black)

    Y_net = Y - y0                          # ★ オフセット除去
    Y_max = Y_net[np.argmax(lv)]
    mask = (lv > 0) & (Y_net > 0) & (Y_max > 0)
    if mask.sum() < 2:
        return None
    x = np.log(lv[mask] / 255.0)
    y = np.log(Y_net[mask] / Y_max)
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, Y_max, y0, int(mask.sum()), r2

# =============================================================
# 基準ガンマ = 各chの実測ランプ(全点フィット)。目標2.0固定は廃止。
#   補正後(前景反射)は背景透過の見えを再現するのが狙い →
#   基準は背景パス RAMP_BG のガンマ。W は R/G/B の平均。
#   ※ 反射パス基準にしたい場合は REF_RAMP = RAMP_FG に変更。
# =============================================================
REF_RAMP = RAMP_BG   # 基準となる素のランプ（背景透過パス）
ref_gamma = {}
for _ch in CHANNELS:                       # ("R", "G", "B")
    _res = estimate_gamma(REF_RAMP[_ch])   # 全点フィット(黒レベル除去)
    ref_gamma[_ch] = _res[0] if _res is not None else float("nan")
ref_gamma["W"] = float(np.nanmean([ref_gamma[c] for c in CHANNELS]))  # W = R/G/B 平均

print("\n==== 各chの基準ガンマ（実測ランプ全点フィット, 黒レベル除去）====")
for _ch in ("R", "G", "B", "W"):
    tag = "(R/G/B平均)" if _ch == "W" else ""
    print(f"{_ch:>3} γ_ref = {ref_gamma[_ch]:6.3f} {tag}")

print("\n==== 補正後ガンマの検証（基準 = 各chの実測ランプ全点フィット）====")
print(f"{'ch':>3} {'gamma':>8} {'γ_ref':>8} {'Δ':>8} {'Y_max_net':>10} {'Y_0':>7} {'点数':>5} {'R^2':>7}")
for _ch in ("R", "G", "B", "W"):
    res = estimate_gamma(GAMMA_MEAS.get(_ch, []))
    tgt = ref_gamma.get(_ch, float("nan"))
    if res is None:
        print(f"{_ch:>3} {'--':>8} {tgt:8.3f}")
        continue
    g, ymax, y0, n, r2 = res
    print(f"{_ch:>3} {g:8.3f} {tgt:8.3f} {g - tgt:+8.3f} {ymax:10.2f} {y0:7.2f} {n:5d} {r2:7.4f}")

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

# フィッティングした行列
# [[ 0.385676 -0.029594  0.007298] [ 0.002786  0.485416 -0.011852] [ 0.005025  0.003184  0.601995]]

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
    ("W",     (45.5, 0.2805, 0.2911), (47.1, 0.2830, 0.2966)),
    ("R",     (7.8, 0.6228, 0.3262), (7.8, 0.6334, 0.3281)),
    ("G",     (33.6, 0.3052, 0.6257), (34.6, 0.3195, 0.6195)),
    ("B",     (3.2, 0.1553, 0.0428), (4.2, 0.1524, 0.0533)),
    ("C",     (37.3, 0.2166, 0.2837), (38.9, 0.2198, 0.2873)),
    ("M",     (11.3, 0.2675, 0.1114), (12.2, 0.2682, 0.1182)),
    ("Y",     (41.7, 0.4049, 0.5376), (42.6, 0.4084, 0.5424)),
    ("Gray",  (11.6, 0.2783, 0.2849), (11.2, 0.2727, 0.2829)),
    
]

# ---- 白色点の扱い ----
# Lab は基準白(ホワイトポイント)を必要とする。
# T と R は明るさのスケールが違う(反射の方が明るい)ので、
# それぞれの "W" の XYZ を各セットの白として正規化(Yw=100)してから比較する。
# → 絶対輝度差ではなく、白を揃えた上での色ズレを評価する。
# 共通白(例: D65)で評価したい場合は USE_OWN_WHITE = False にする。
WHITE_XYZ = np.array([97.5, 100.9, 116.1])  # 共通白を使う場合

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
print(f"{'color':>6} {'ΔE00':>8} {'ΔE76':>8} {'Y_T':>7} {'Y_R':>7} {'ΔY':>7} ")
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

    print(f"{name:>6} {dE00:8.3f} {dE76:8.3f} {t_row[0]:7.2f} {r_row[0]:7.2f} {r_row[0]-t_row[0]:+7.2f} ")

if dE00_list:
    avg = sum(d for _, d in dE00_list) / len(dE00_list)
    worst = max(dE00_list, key=lambda x: x[1])
    print(f"\n平均 ΔE00 = {avg:.3f} / 最大 ΔE00 = {worst[1]:.3f} ({worst[0]})")

    # =============================================================
# 追加: 輝度2倍バージョン（色味そのまま・最大輝度を2倍に）
#   原理: R'は線形 → c_lin をk倍すると反射XYZもk倍（色度不変・輝度Yだけk倍）。
#   「XYZ空間で2倍」= c_lin を2倍 = C を 2C にする、と等価。
#   注意: k倍後に正規化線形が1.0(=画素値255)を超えるchは前景の物理上限で飽和し、
#         そこから先は色味が崩れる。色味を保てる上限 k_max も併記。
# =============================================================
LUM_GAIN = 2.0

c_lin_boost = LUM_GAIN * c_lin                          # XYZ空間で2倍（線形空間で等価）
eps = 1e-12
k_max   = 1.0 / np.maximum(np.max(c_lin, axis=1), eps)  # 各色で色味を保てる最大ゲイン
clipped = np.max(c_lin_boost, axis=1) > 1.0             # 2倍で飽和＝色ずれする色

c_lin_boost_clip = np.clip(c_lin_boost, 0.0, 1.0)
c_colors_x2 = g_f_inv(c_lin_boost_clip)                 # 2倍版の前景画素値【成果物】
refl_xyz_x2 = (R_prime @ c_lin_boost_clip.T).T          # クランプ後に実際に出る反射XYZ

print("\n==== 輝度2倍版（色味固定）====")
print(f"{'color':>8} {'Y_x1':>7} {'狙2x':>7} {'実x2':>7} {'k_max':>6} {'飽和':>4}")
for i, label in enumerate(labels):
    Y1 = refl_xyz[i, 1]
    print(f"{label:>8} {Y1:7.2f} {2*Y1:7.2f} {refl_xyz_x2[i,1]:7.2f} "
          f"{k_max[i]:6.2f} {'*' if clipped[i] else '':>4}")

# 2倍版の前景画像を保存（suffix: 5_corrected_fg_x2, -> patches）
for idx, label in enumerate(labels):
    patch = np.tile(np.clip(c_colors_x2[idx], 0.0, 1.0), (PATCH, PATCH, 1))
    plt.imsave(os.path.join(PATCH_FIG_DIR, f"{label}_5_corrected_fg_x2.png"), patch)
print("[保存] 2倍版 (5_corrected_fg_x2) を出力しました")
print("k_max<2.0 の色は物理上限に達し、2倍まで到達できず色がずれます")

# =============================================================
# 追加: White を「指定した輝度」で出す（LUT一度だけ版・色味は背景透過の白）
#   ・前景(反射)で狙い輝度を出す画像  + 検証用に背景(透過)で狙い輝度になる画像
#   ・この範囲の出力は専用フォルダ RANGE_DIR にまとめて保存
# =============================================================
RANGE_DIR = FG_ADD_DIR   # 白輝度指定(前景加算) -> results/figures/DisplayBrightness/foreground_add
os.makedirs(RANGE_DIR, exist_ok=True)

# ---- 前景(反射)側: Y -> 前景画素値 の1D LUT を一度だけ構築（g_f_inv はここのみ）----
w_idx       = labels.index("White")
c_lin_white = c_lin[w_idx]                 # 白の補正後 線形駆動（背景白の色度を保持）
Y_white_now = refl_xyz[w_idx, 1]           # 現状で出る白輝度(反射, ≒背景透過白)
k_max_white = 1.0 / np.max(c_lin_white)    # 色味を保てる最大ゲイン
Y_white_max = Y_white_now * k_max_white    # 反射で出せる白輝度の上限

LUT_N   = 1024
Y_grid  = np.linspace(0.0, Y_white_max, LUT_N)
c_grid  = np.clip((Y_grid[:, None] / Y_white_now) * c_lin_white[None, :], 0.0, 1.0)
px_grid = g_f_inv(c_grid)                                            # (LUT_N, 3)

def _lut_lookup(Y):
    """狙い輝度(スカラー/配列) -> 前景画素値。正規格子なので直接補間。"""
    Yc  = np.clip(np.asarray(Y, dtype=float), 0.0, Y_white_max)
    idx = Yc / Y_white_max * (LUT_N - 1)
    i0  = np.floor(idx).astype(int)
    i1  = np.minimum(i0 + 1, LUT_N - 1)
    w   = (idx - i0)[..., None]
    return px_grid[i0] * (1.0 - w) + px_grid[i1] * w

def white_drive_for_luminance(Y_target, save=True, verbose=True):
    """指定輝度 Y_target の白を出す【前景】画素値を LUT から取得。"""
    feasible = Y_target <= Y_white_max + 1e-9
    Y_eff    = min(float(Y_target), Y_white_max)
    px       = _lut_lookup(Y_eff)
    if verbose:
        print(f"[FG 反射] 狙い={Y_target:6.2f}  実効={Y_eff:6.2f}  "
              f"{'OK' if feasible else '★上限超過→クランプ'}")
        print(f"          画素値(0-1)={np.round(px,4)}  (0-255)={np.round(px*255).astype(int)}")
    if save:
        patch = np.tile(px, (PATCH, PATCH, 1))
        plt.imsave(os.path.join(RANGE_DIR, f"FG_White_Y{int(round(Y_target))}.png"), patch)
    return px, Y_eff, feasible

def white_image_from_luminance(Ymap):
    """狙い輝度マップ Ymap(H,W) -> 前景画像(H,W,3)。"""
    return _lut_lookup(Ymap)

# ---- 背景(透過)側: 検証用に「透過後に狙い輝度になる」背景画像を出力 ----
g_b_inv       = lambda l: _apply(_gb, l, 1)          # 正規化線形 -> 背景画素値(逆EOTF)
base_lin_white = base_lin[w_idx]                     # = g_b([1,1,1]) = [1,1,1]
Y_bg_white_now = trans_xyz[w_idx, 1]                 # 現状の透過白輝度(背景の白上限)

def bg_image_for_transmitted_luminance(Y_target, save=True, verbose=True):
    """透過後に Y_target になる【背景】白画素値を返す。色度は背景白のまま。"""
    feasible = Y_target <= Y_bg_white_now + 1e-9
    Y_eff    = min(float(Y_target), Y_bg_white_now)  # 背景は自分の白を超えられない
    c_lin_bg = np.clip((Y_eff / Y_bg_white_now) * base_lin_white, 0.0, 1.0)
    px       = g_b_inv(c_lin_bg)                     # 背景画素値(0-1)
    if verbose:
        print(f"[BG 透過] 狙い={Y_target:6.2f}  実効={Y_eff:6.2f}  "
              f"{'OK' if feasible else '★背景上限超過→クランプ'}")
        print(f"          画素値(0-1)={np.round(px,4)}  (0-255)={np.round(px*255).astype(int)}")
    if save:
        patch = np.tile(px, (PATCH, PATCH, 1))
        plt.imsave(os.path.join(RANGE_DIR, f"BG_White_Y{int(round(Y_target))}.png"), patch)
    return px, Y_eff, feasible

print("\n==== White 指定輝度モード（LUT一度だけ版）====")
print(f"保存先          RANGE_DIR    = {RANGE_DIR}")
print(f"現状の白輝度    Y_white_now  = {Y_white_now:6.2f} cd/m^2 (反射)")
print(f"反射の出せる上限 Y_white_max = {Y_white_max:6.2f} cd/m^2 (k_max={k_max_white:.3f})")
print(f"透過白の上限    Y_bg_white   = {Y_bg_white_now:6.2f} cd/m^2 (背景)")
print(f"LUT: {LUT_N} 点（g_f_inv の評価は構築時のみ）")

# --- 前景: 好きな輝度を指定（上限 Y_white_max 以下）---
print("\n-- 前景(反射)画像 --")
for Yt in [0.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0]:
    white_drive_for_luminance(Yt)

# --- 背景: 検証用に透過後 15, 30 になる画像 ---
print("\n-- 背景(透過)検証画像 --")
for Yt in [0.0, 10.0, 15.0, 20.0, 30.0]:
    bg_image_for_transmitted_luminance(Yt)

# =============================================================
# 追加: 加算(Add) vs シミュレート(sim) の色差・輝度差
#   White を対象に、狙い輝度ごとに Add と sim の Yxy を比較。
#   出力: ΔE00, ΔL*, Δa*, Δb*, ΔY   （符号は sim - add）
#   ※Lab の基準白は既存 WHITE_XYZ を流用（add/sim 共通）。別の白にするなら差し替え。
# =============================================================
# name(狙い輝度), Add(Y,x,y), sim(Y,x,y)
ADDSIM = [
    ("Y15", (15.0, 0.2894, 0.3075), (14.9, 0.2866, 0.3071)),
    ("Y30", (29.4, 0.2862, 0.3070), (31.1, 0.2869, 0.3035)),
    ("Y45", (45.1, 0.2861, 0.3038), (46.2, 0.2842, 0.3000)),
    ("Y60", (59.9, 0.2837, 0.2985), (61.4, 0.2838, 0.2989)),
]

ADDSIM_WHITE = WHITE_XYZ   # Lab 基準白（Yw=100スケール）。必要なら実測白に変更可。

print("\n==== Add(加算) vs sim(シミュレート) 色差・輝度差  [符号 = sim - add] ====")
print(f"{'target':>7} {'ΔE00':>8} {'ΔL*':>8} {'Δa*':>8} {'Δb*':>8} "
      f"{'ΔY':>7} {'Y_add':>7} {'Y_sim':>7}")

_addsim_dE = []
for name, add_row, sim_row in ADDSIM:
    if add_row is None or sim_row is None:
        continue
    if any(v is None for v in add_row) or any(v is None for v in sim_row):
        continue

    xyz_add = Yxy_to_XYZ(add_row)
    xyz_sim = Yxy_to_XYZ(sim_row)
    Lab_add = XYZ_to_Lab(xyz_add, ADDSIM_WHITE)
    Lab_sim = XYZ_to_Lab(xyz_sim, ADDSIM_WHITE)

    dE00 = ciede2000(Lab_add, Lab_sim)
    dL = Lab_sim[0] - Lab_add[0]        # ΔL* = sim - add
    da = Lab_sim[1] - Lab_add[1]        # Δa*
    db = Lab_sim[2] - Lab_add[2]        # Δb*
    dY = sim_row[0] - add_row[0]        # ΔY (絶対輝度差)

    _addsim_dE.append((name, dE00))
    print(f"{name:>7} {dE00:8.3f} {dL:+8.3f} {da:+8.3f} {db:+8.3f} "
          f"{dY:+7.2f} {add_row[0]:7.2f} {sim_row[0]:7.2f}")

if _addsim_dE:
    _avg = sum(d for _, d in _addsim_dE) / len(_addsim_dE)
    _worst = max(_addsim_dE, key=lambda x: x[1])
    print(f"\n平均 ΔE00 = {_avg:.3f} / 最大 ΔE00 = {_worst[1]:.3f} ({_worst[0]})")