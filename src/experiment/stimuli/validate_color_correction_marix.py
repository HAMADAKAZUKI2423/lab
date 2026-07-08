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

# =============================================================
# 入力パッチ & 階調ランプを CSV から読み込み（コード1と同じローダを再掲）
#   results/tables/DisplayBrightness/{background,foreground}/{patches,ramps}/*.csv
#   ・patches 列: name, sR, sG, sB, Y, x, y
#   ・ramps   列: channel, pixel, Y, x, y
#   ・平均は Yxy->XYZ に変換して XYZ空間で平均 -> Yxy に戻す（欠損は「あるものだけ」）
# =============================================================
import csv
import glob

DATA_ROOT   = os.path.join("results", "tables", "DisplayBrightness")
PATCH_ORDER = ["R", "G", "B", "W", "semiR", "semiG", "semiB", "Gray"]

def _yxy2xyz(Y, x, y):
    if y == 0:
        return np.array([0.0, 0.0, 0.0])
    X = (x / y) * Y
    Z = ((1.0 - x - y) / y) * Y
    return np.array([X, float(Y), Z])

def _xyz2yxy(xyz):
    X, Y, Z = (float(v) for v in xyz)
    s = X + Y + Z
    if s == 0:
        return (0.0, 0.0, 0.0)
    return (Y, X / s, Y / s)   # (Y, x=X/s, y=Y/s)

def _read_csv_dicts(path):
    """ヘッダ付きCSVを {列名(大小保持): 値} の dict のリストで返す。"""
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for raw in csv.DictReader(f):
            row, empty = {}, True
            for k, v in raw.items():
                if k is None:
                    continue
                val = (v or "").strip()
                row[k.strip()] = val
                if val != "":
                    empty = False
            if not empty:
                rows.append(row)
    return rows

def _get(row, *names):
    """候補列名のうち最初に見つかった非空の値を返す。"""
    for nm in names:
        if nm in row and row[nm] != "":
            return row[nm]
    raise KeyError(names)

def load_patches_avg(sub):
    """{sub}/patches/*.csv を name ごとにXYZ平均。 -> ({name:{'srgb','yxy','n'}}, ファイル数)"""
    d = os.path.join(DATA_ROOT, sub, "patches")
    files = sorted(glob.glob(os.path.join(d, "*.csv")))
    if not files:
        raise FileNotFoundError(f"CSVが見つかりません: {d}")
    srgb_acc, xyz_acc = {}, {}
    for path in files:
        for row in _read_csv_dicts(path):
            try:
                name = _get(row, "name", "Name")
                srgb = (float(_get(row, "sR", "sr")),
                        float(_get(row, "sG", "sg")),
                        float(_get(row, "sB", "sb")))
                Y = float(_get(row, "Y")); x = float(_get(row, "x")); y = float(_get(row, "y"))
            except (KeyError, ValueError):
                continue
            srgb_acc.setdefault(name, []).append(srgb)
            xyz_acc.setdefault(name, []).append(_yxy2xyz(Y, x, y))
    out = {}
    for name, xyzs in xyz_acc.items():
        out[name] = {
            "srgb": tuple(np.mean(np.array(srgb_acc[name]), axis=0)),
            "yxy":  _xyz2yxy(np.mean(np.array(xyzs), axis=0)),
            "n":    len(xyzs),
        }
    return out, len(files)

def load_ramps_avg(sub):
    """{sub}/ramps/*.csv を (channel,pixel) ごとにXYZ平均。 -> {ch:[(pixel,Y,x,y),...]}"""
    d = os.path.join(DATA_ROOT, sub, "ramps")
    files = sorted(glob.glob(os.path.join(d, "*.csv")))
    if not files:
        raise FileNotFoundError(f"CSVが見つかりません: {d}")
    xyz_acc = {}
    for path in files:
        for row in _read_csv_dicts(path):
            try:
                ch = _get(row, "channel", "Channel", "ch").upper()
                px = int(round(float(_get(row, "pixel", "level"))))
                Y = float(_get(row, "Y")); x = float(_get(row, "x")); y = float(_get(row, "y"))
            except (KeyError, ValueError):
                continue
            xyz_acc.setdefault((ch, px), []).append(_yxy2xyz(Y, x, y))
    ramp = {}
    for (ch, px), xyzs in xyz_acc.items():
        Yc, xc, yc = _xyz2yxy(np.mean(np.array(xyzs), axis=0))
        ramp.setdefault(ch, []).append((px, Yc, xc, yc))
    for ch in ramp:
        ramp[ch].sort(key=lambda t: -t[0])   # 255 -> 0 の順
    return ramp

# ---- パッチ平均を読み込み、背景(BGT)・前景(FGR)を name で突き合わせて PATCHES を生成 ----
_bg_p, _n_bg = load_patches_avg("background")
_fg_p, _n_fg = load_patches_avg("foreground")

_names  = [n for n in PATCH_ORDER if (n in _bg_p or n in _fg_p)]
_names += [n for n in sorted(set(_bg_p) | set(_fg_p)) if n not in _names]

PATCHES = []
for n in _names:
    _src = _bg_p.get(n) or _fg_p.get(n)
    srgb = tuple(float(c) for c in _src["srgb"])
    bgt  = tuple(_bg_p[n]["yxy"]) if n in _bg_p else None
    fgr  = tuple(_fg_p[n]["yxy"]) if n in _fg_p else None
    PATCHES.append((n, srgb, bgt, fgr))

print(f"[CSV読込] パッチ: background {_n_bg} ファイル / foreground {_n_fg} ファイル -> {len(PATCHES)} パッチ")

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

# ---- 実測EOTF g の構築に使う階調ランプ（CSVからXYZ平均で生成）----
CHANNELS = ("R", "G", "B")

RAMP_BG = load_ramps_avg("background")   # 背景パス (T·Db) の階調ランプ {ch:[(pixel,Y,x,y),...]}
RAMP_FG = load_ramps_avg("foreground")   # 前景パス (R·Df) の階調ランプ

print("[CSV読込] 階調ランプ (レベル数):")
for _side, _ramp in (("BG", RAMP_BG), ("FG", RAMP_FG)):
    print(f"  {_side}: " + ", ".join(f"{_ch}={len(_ramp.get(_ch, []))}" for _ch in CHANNELS))

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
    # 常に個別EOTF（実測点の線形補完）を採用する。
    # ※power/interp の自動判定は使わず、interp に固定。
    #   直線性チェック・残差プロットは analyze_gamma 側で従来どおり生成される。
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
print("  各チャンネル×各レベルの画像(生/補正後)を前景に表示して Yxy を計測し、")
print(f"  生を {os.path.join('foreground', 'ramps')} / 補正後を {os.path.join('foreground', 'corrected_ramps')} に CSV 保存してください。")

# =============================================================
# 追加②: 補正後の直線性チェック（個別EOTF が補正後も効いているか）
#   生ランプ   : results/tables/DisplayBrightness/foreground/ramps           の平均
#   補正後ランプ: results/tables/DisplayBrightness/foreground/corrected_ramps の平均
#   コード1と同様に「横軸 v^2.2 vs 測定輝度」と残差プロットを、生・補正後で重ね描き。
#   ※どちらも実測点を結ぶ線グラフ(個別EOTF=線形補完)のため、
#     ガンマ推定・決定係数などのフィットは行わない。
# =============================================================
def _load_ramps_dir(dir_path):
    """任意ディレクトリの *.csv を (channel,pixel) ごとにXYZ平均。 -> {ch:[(pixel,Y,x,y),...]}"""
    files = sorted(glob.glob(os.path.join(dir_path, "*.csv")))
    if not files:
        raise FileNotFoundError(f"CSVが見つかりません: {dir_path}")
    xyz_acc = {}
    for path in files:
        for row in _read_csv_dicts(path):
            try:
                ch = _get(row, "channel", "Channel", "ch").upper()
                px = int(round(float(_get(row, "pixel", "level"))))
                Y = float(_get(row, "Y")); x = float(_get(row, "x")); y = float(_get(row, "y"))
            except (KeyError, ValueError):
                continue
            xyz_acc.setdefault((ch, px), []).append(_yxy2xyz(Y, x, y))
    ramp = {}
    for (ch, px), xyzs in xyz_acc.items():
        Yc, xc, yc = _xyz2yxy(np.mean(np.array(xyzs), axis=0))
        ramp.setdefault(ch, []).append((px, Yc, xc, yc))
    for ch in ramp:
        ramp[ch].sort(key=lambda t: -t[0])   # 255 -> 0 の順
    return ramp

def plot_corrected_linearity(name, ramp_raw, ramp_corr, target_gamma=2.2, plot_dir=None):
    """生ランプ(未補正)と補正後ランプの直線性・残差を重ね描き。
       個別EOTF(線形補完)が補正後も効いているかの確認用。
       横軸 v^2.2 / 縦軸 正規化輝度 yn。理想(=完全な線形補完)なら y=x に乗る。
       戻り値: (rms_raw, rms_corr) = 各々の y=x からの残差RMS。"""
    v_raw,  yn_raw,  _ = _prep_ramp(ramp_raw)
    v_corr, yn_corr, _ = _prep_ramp(ramp_corr)
    x_raw,  x_corr = v_raw ** target_gamma, v_corr ** target_gamma
    resid_raw, resid_corr = yn_raw - x_raw, yn_corr - x_corr   # y=x(理想γ2.2)からのズレ
    rms_raw  = float(np.sqrt(np.mean(resid_raw ** 2)))
    rms_corr = float(np.sqrt(np.mean(resid_corr ** 2)))
    print(f"\n---- [{name}] 補正前後の直線性（横軸 v^{target_gamma}）----")
    print(f"  残差RMS(vs y=x): raw={rms_raw:.4f} / corrected={rms_corr:.4f}")
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        # (左) 直線性: v^2.2 vs 測定輝度 yn を 生・補正後で重ね描き
        ax[0].plot([0, 1], [0, 1], "k--", lw=1, label=f"ideal γ={target_gamma} (y=x)")
        ax[0].plot(x_raw,  yn_raw,  "o-", label="raw (uncorrected)")
        ax[0].plot(x_corr, yn_corr, "s-", label="corrected")
        ax[0].set_title(f"[{name}] linearity: pixel^{target_gamma} vs measured")
        ax[0].set_xlabel(f"v^{target_gamma}"); ax[0].set_ylabel("yn (measured, 0-1)")
        ax[0].legend(); ax[0].grid(True, alpha=.3)
        # (右) 残差プロット: yn - v^2.2 を 生・補正後で重ね描き
        ax[1].axhline(0, color="k", lw=1)
        ax[1].plot(v_raw,  resid_raw,  "o-", label=f"raw (RMS={rms_raw:.4f})")
        ax[1].plot(v_corr, resid_corr, "s-", label=f"corrected (RMS={rms_corr:.4f})")
        ax[1].set_title(f"[{name}] residual vs γ={target_gamma}")
        ax[1].set_xlabel("v (0-1)"); ax[1].set_ylabel(f"yn - v^{target_gamma}")
        ax[1].legend(); ax[1].grid(True, alpha=.3)
        fig.tight_layout()
        path = os.path.join(plot_dir, f"corrected_linearity_{name}.png")
        fig.savefig(path, dpi=120); plt.close(fig)
        print(f"  [plot] {path}")
    return rms_raw, rms_corr

# 生(未補正)= foreground/ramps の平均(=RAMP_FG) / 補正後 = foreground/corrected_ramps の平均
RAMP_FG_RAW       = RAMP_FG
CORR_RAMP_CSV_DIR = os.path.join(DATA_ROOT, "foreground", "corrected_ramps")
try:
    RAMP_FG_CORR = _load_ramps_dir(CORR_RAMP_CSV_DIR)
except FileNotFoundError as _e:
    RAMP_FG_CORR = {}
    print(f"\n[補正後ランプ] 未測定のためプロットをスキップ: {_e}")

CORR_PLOT_DIR = os.path.join(RAMP_FIG_DIR, "corrected_linearity")
for _ch in [c for c in RAMP_FG_RAW if c in RAMP_FG_CORR]:
    plot_corrected_linearity(_ch, RAMP_FG_RAW[_ch], RAMP_FG_CORR[_ch], plot_dir=CORR_PLOT_DIR)

# =============================================================
# 追加処理: 透過T(背景ランプ) vs 反射R(補正後ランプ) の色差
#   コード1の背景ランプ RAMP_BG を透過ターゲット T、
#   B-3 で読み込んだ補正後ランプ RAMP_FG_CORR を反射結果 R とみなし、
#   同一画素レベルで CIEDE2000(ΔE00) と ΔY を算出する。
# =============================================================
import math

# Lab の基準白（共通白）。R/T 両方の Lab をこの白基準で評価する。
WHITE_XYZ = np.array([97.5, 100.9, 116.1])

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

# ===================== 色差の算出（チャンネル×全レベル平均）=====================
#   T = 背景ランプ RAMP_BG（透過ターゲット）/ R = 補正後ランプ RAMP_FG_CORR（反射結果）
#   同一画素レベルで対応付けし、各chの全レベル平均 ΔE00・ΔY を出力する。
def _ramp_by_level(ramp_channel):
    """[(pixel,Y,x,y),...] -> {pixel: (Y, x, y)}"""
    return {p[0]: (p[1], p[2], p[3]) for p in ramp_channel}

print("\n==== T(透過=背景ランプ) vs R(反射=補正後ランプ) 色差 [チャンネル×全レベル平均] ====")
if not RAMP_FG_CORR:
    print("  補正後ランプ(foreground/corrected_ramps)が未読込のためスキップ")
else:
    print(f"{'ch':>3} {'ΔE00_avg':>9} {'ΔE00_max':>9} {'ΔY_avg':>8} {'点数':>5}")
    for _ch in [c for c in CHANNELS if c in RAMP_BG and c in RAMP_FG_CORR]:
        _T = _ramp_by_level(RAMP_BG[_ch])
        _R = _ramp_by_level(RAMP_FG_CORR[_ch])
        _levels = sorted(set(_T) & set(_R))
        _dE, _dY = [], []
        for _lv in _levels:
            _Lab_T = XYZ_to_Lab(Yxy_to_XYZ(_T[_lv]), WHITE_XYZ)
            _Lab_R = XYZ_to_Lab(Yxy_to_XYZ(_R[_lv]), WHITE_XYZ)
            _dE.append(ciede2000(_Lab_T, _Lab_R))
            _dY.append(_R[_lv][0] - _T[_lv][0])   # ΔY = Y_R - Y_T
        if not _dE:
            print(f"{_ch:>3} {'--':>9}  (共通レベルなし)")
            continue
        print(f"{_ch:>3} {np.mean(_dE):9.3f} {np.max(_dE):9.3f} {np.mean(_dY):+8.2f} {len(_dE):5d}")

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
for Yt in [15.0, 30.0, 45.0, 60.0]:
    white_drive_for_luminance(Yt)

# --- 背景: 検証用に透過後 15, 30 になる画像 ---
print("\n-- 背景(透過)検証画像 --")
for Yt in [15.0, 30.0]:
    bg_image_for_transmitted_luminance(Yt)

# =============================================================
# 追加: 加算(Add) vs シミュレート(sim) の色差・輝度差
#   White を対象に、狙い輝度ごとに Add と sim の Yxy を比較。
#   出力: ΔE00, ΔL*, Δa*, Δb*, ΔY   （符号は sim - add）
#   ※Lab の基準白は既存 WHITE_XYZ を流用（add/sim 共通）。別の白にするなら差し替え。
# =============================================================
# ---- Add(加算) は実測を引用 / sim は既存の測定値をそのまま使用 ----
#   Add: results/tables/DisplayBrightness/foreground_add/*.csv（解決策1と同じCSV群）の
#        「加算後」Yxy を、狙い輝度(=bg+fg)ごとに XYZ空間で平均して引用する。
#   sim: 2026/07/07 08:33 の測定データ（数値は変更しない）。
_SIM_BY_TARGET = [
    # name,  target,  sim(Y, x, y)
    ("Y0",    0, (0.4,  0.1707, 0.1190)),
    ("Y10",  10, (10.7, 0.2827, 0.2875)),
    ("Y20",  20, (20.4, 0.2813, 0.2871)),
    ("Y30",  30, (30.4, 0.2849, 0.2927)),
    ("Y40",  40, (40.3, 0.2836, 0.2928)),
    ("Y50",  50, (49.8, 0.2841, 0.2939)),
    ("Y60",  60, (59.7, 0.2847, 0.2978)),
]

_ADDSIM_CSV_DIR = os.path.join(TABLE_DIR, "foreground_add")

def _load_add_by_target(dir_path):
    """foreground_add/*.csv を読み、狙い輝度(=bg+fg)ごとに加算後Yxyを XYZ空間で平均。
       -> {target(int): (Y, x, y)}。bg/fg 列が無い行は加算後Yの丸めをキーにする。"""
    files = sorted(glob.glob(os.path.join(dir_path, "*.csv")))
    if not files:
        print(f"[Add実測] CSVが見つかりません（Add列は空でスキップ）: {dir_path}")
        return {}
    def _pick(row, names, required=True):
        for nm in names:
            if nm in row and row[nm] != "":
                return row[nm]
        if required:
            raise KeyError(names)
        return None
    acc = {}   # target(int) -> [XYZ, ...]
    for path in files:
        with open(path, newline="", encoding="utf-8-sig") as f:
            for raw in csv.DictReader(f):
                row = {(k or "").strip(): (v or "").strip()
                       for k, v in raw.items() if k is not None}
                try:
                    Y = float(_pick(row, ("Y", "Y_add", "Yadd")))
                    x = float(_pick(row, ("x",)))
                    y = float(_pick(row, ("y",)))
                except (KeyError, ValueError):
                    continue
                bg = _pick(row, ("bg", "BG", "bg_Y", "Ybg"), required=False)
                fg = _pick(row, ("fg", "FG", "fg_Y", "Yfg"), required=False)
                try:
                    key = int(round(float(bg) + float(fg)))   # 狙い輝度 = bg + fg
                except (TypeError, ValueError):
                    key = int(round(Y))                        # bg/fg 無ければ加算後Yで代用
                acc.setdefault(key, []).append(_yxy2xyz(Y, x, y))
    return {k: _xyz2yxy(np.mean(np.array(v), axis=0)) for k, v in acc.items()}

_add_by_target = _load_add_by_target(_ADDSIM_CSV_DIR)

def _add_meas_for(target, tol=5):
    """狙い輝度 target に対応する実測加算Yxyを返す。完全一致優先、無ければ
       ±tol 以内の最も近いキーを使う（該当なしは None でスキップ）。"""
    if not _add_by_target:
        return None
    if target in _add_by_target:
        return _add_by_target[target]
    k = min(_add_by_target, key=lambda kk: abs(kk - target))
    return _add_by_target[k] if abs(k - target) <= tol else None

# name(狙い輝度), Add(Y,x,y)=実測加算(引用), sim(Y,x,y)=既存のまま
ADDSIM = [(name, _add_meas_for(tgt), sim) for name, tgt, sim in _SIM_BY_TARGET]

print("\n[Add実測] foreground_add の加算後Yxyを引用（狙い輝度=bg+fg で平均）")
for _name, _add_row, _sim in ADDSIM:
    _s = "（実測なし→スキップ）" if _add_row is None \
         else f"Y={_add_row[0]:.2f}, x={_add_row[1]:.4f}, y={_add_row[2]:.4f}"
    print(f"  {_name:>4}: {_s}")

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

# =============================================================
# 追加: 解決策1 — 実測加算結果ベースの単一プレーンLUT & 加算シミュレート画像
#   参照: 画素値飽和問題ページ 手順１
#   入力: results/tables/DisplayBrightness/foreground_add/*.csv
#         bg,fg ともに輝度{0,10,15,20,30}の 5x5=25 通りで測定した「加算後」Yxy
#         列: Y, x, y (ヘッダ行あり) ─ bg,fg 列は無くても可 / 大小・別名許容
#   方法: 各測定の加算後 Yxy -> XYZ -> inv(R') -> 前景線形 -> g_f_inv で
#         前景画素値を算出し、加算後輝度 Y をキーに (Y -> 前景画素値) を
#         線形補間する 1D LUT を構築（範囲外は端点クランプ）。
#   出力: 目標輝度 [0,10,20,30,40,50,60] を「前景1枚」で再現する
#         加算シミュレート画像を FG_ADD_DIR に保存 (AddSim_Y**.png)。
# =============================================================
import csv, glob

ADD_CSV_DIR = os.path.join(TABLE_DIR, "foreground_add")   # 25通りの加算測定CSV置き場
ADD_TARGETS = [0, 10, 20, 30, 40, 50, 60]                 # 前景で再現したい加算輝度

R_prime_inv = np.linalg.inv(R_prime)                      # XYZ -> 前景線形RGB

def _read_add_measurements(dir_path):
    """foreground_add/*.csv を読み込み [(bg, fg, Y, x, y), ...] を返す。
       Y=加算後輝度, x,y=色度。bg,fg(目標輝度)は無ければ None。"""
    files = sorted(glob.glob(os.path.join(dir_path, "*.csv")))
    if not files:
        raise FileNotFoundError(f"加算測定CSVが見つかりません: {dir_path}")
    def pick(row, names, required=True):
        for nm in names:
            if nm in row and row[nm] != "":
                return row[nm]
        if required:
            raise KeyError(names)
        return None
    out = []
    for path in files:
        with open(path, newline="", encoding="utf-8-sig") as f:
            for raw in csv.DictReader(f):
                row = {(k or "").strip(): (v or "").strip()
                       for k, v in raw.items() if k is not None}
                try:
                    Y = float(pick(row, ("Y", "Y_add", "Yadd")))
                    x = float(pick(row, ("x",)))
                    y = float(pick(row, ("y",)))
                except (KeyError, ValueError):
                    continue
                bg = pick(row, ("bg", "BG", "bg_Y", "Ybg"), required=False)
                fg = pick(row, ("fg", "FG", "fg_Y", "Yfg"), required=False)
                out.append((float(bg) if bg else None,
                            float(fg) if fg else None, Y, x, y))
    return out

_add_meas = _read_add_measurements(ADD_CSV_DIR)

# 各測定: 加算後 Yxy -> XYZ -> inv(R') -> 前景線形 -> g_f_inv -> 前景画素値
_Y_samp, _px_samp = [], []
for _bg, _fg, _Y, _x, _y in _add_meas:
    _xyz    = Yxy_to_XYZ((_Y, _x, _y))
    _fg_lin = np.clip(R_prime_inv @ _xyz, 0.0, None)   # 目標XYZを出す前景線形RGB
    _px     = np.clip(g_f_inv(_fg_lin), 0.0, 1.0)      # 前景画素値(0-1)
    _Y_samp.append(_Y)
    _px_samp.append(_px)

_Y_samp  = np.array(_Y_samp, dtype=float)
_px_samp = np.array(_px_samp, dtype=float)             # (N, 3)

if _Y_samp.size == 0:
    raise ValueError("加算測定の有効行がありません（列 Y,x,y を確認してください）")

# 加算後輝度 Y をキーに昇順化。同一Yは平均して単調(np.interp用)にする。
_uY, _inv = np.unique(np.round(_Y_samp, 4), return_inverse=True)
_uPx = np.array([_px_samp[_inv == i].mean(axis=0) for i in range(len(_uY))])

print("\n==== 解決策1: 実測加算LUT（単一プレーン再現用）====")
print(f"入力CSV: {ADD_CSV_DIR}")
print(f"測定点 {len(_add_meas)} 件 / 有効輝度 {len(_uY)} 段 / 加算後輝度レンジ {_uY.min():.2f} 〜 {_uY.max():.2f} cd/m^2")

def add_sim_pixel(Y_target):
    """加算後の目標輝度 -> 前景画素値(0-1)。範囲外は端点にクランプ。"""
    px = np.array([np.interp(float(Y_target), _uY, _uPx[:, i]) for i in range(3)])
    return np.clip(px, 0.0, 1.0)

# ---- 加算シミュレート画像（前景1枚で目標輝度を再現）----
os.makedirs(FG_ADD_DIR, exist_ok=True)
print("\n-- 加算シミュレート画像 (前景1枚 / 目標輝度を再現) --")
print(f"{'target':>7} {'画素値(0-255)':>16} {'備考':>10}")
add_sim_paths = []
for Yt in ADD_TARGETS:
    px   = add_sim_pixel(Yt)
    note = "" if (_uY.min() - 1e-9) <= Yt <= (_uY.max() + 1e-9) else "★範囲外→クランプ"
    plt.imsave(os.path.join(FG_ADD_DIR, f"AddSim_Y{int(Yt):02d}.png"),
               np.tile(px, (PATCH, PATCH, 1)))
    add_sim_paths.append(f"AddSim_Y{int(Yt):02d}.png")
    print(f"{Yt:7d} {str(np.round(px*255).astype(int)):>16} {note:>10}")
print(f"[保存] {len(add_sim_paths)} 枚 -> {FG_ADD_DIR}")