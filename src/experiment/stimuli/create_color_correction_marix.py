import os
import csv
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =============================================================
# 実測パッチから色変換行列 T'(透過)・R'(反射)・C(=inv(R')·T') を推定し CSV 保存する
#   モデル: XYZ = M @ rgb_lin  (M は 3x3 / 純色+白+CMY+グレーを最小二乗で当てはめ)
# =============================================================

# ---- 入力パッチを CSV から読み込み（複数ファイルを XYZ 空間で平均）----
#   results/tables/DisplayBrightness/{background,foreground}/patches/*.csv
#   列: name, sR, sG, sB, Y, x, y

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
for _p in PATCHES:
    _b = None if _p[2] is None else tuple(round(v, 4) for v in _p[2])
    _f = None if _p[3] is None else tuple(round(v, 4) for v in _p[3])
    print(f"  {_p[0]:>6}  sRGB={tuple(round(c,3) for c in _p[1])}  BGT={_b}  FGR={_f}")

# ============================================================
# 実測EOTF g の構築
#   背景パス(T·Db)用 g_b、前景パス(R·Df)用 g_f、逆写像 g_f_inv
#   画素値(0-1) <-> 正規化線形(0-1)、g(1)=1 に正規化
# ============================================================
CHANNELS = ("R", "G", "B")

# 階調ランプを CSV から生成（{background,foreground}/ramps/*.csv をXYZ空間で平均, 欠損は無視）
RAMP_BG = load_ramps_avg("background")   # 背景パス (T·Db) の階調ランプ {ch:[(pixel,Y,x,y),...]}
RAMP_FG = load_ramps_avg("foreground")   # 前景パス (R·Df) の階調ランプ

print("[CSV読込] 階調ランプ (レベル数):")
for _side, _ramp in (("BG", RAMP_BG), ("FG", RAMP_FG)):
    print(f"  {_side}: " + ", ".join(f"{_ch}={len(_ramp.get(_ch, []))}" for _ch in CHANNELS))

# =============================================================
# 階調ランプの直線性チェック & ガンマ推定（プロットのみ）
#   (1) XYZ^(1/2.2) を横軸に取り、画素値との直線性を確認
#         完全に gamma=2.2 なら yn = v^2.2  →  yn^(1/2.2) = v (直線 y=x)
#         → y=x からの残差プロットで「2.2 とのズレ」を見る
#   (2) 最小二乗(log-log)で実ガンマをフィット
#         log(yn) = gamma*log(v)  の傾き=gamma / 決定係数 R^2
#         → フィットしたガンマとの残差(yn - v^gamma)プロット
#   ※ 解析でfitしたチャンネル別ガンマをEOTFとして採用する。
#      プロット構成は変更せず、fit品質の確認に用いる。
# =============================================================

def _prep_ramp(ramp_channel, remove_black=True):
    """ramp [(pixel,Y,x,y),...] -> (v, yn, Y0)
    v : 正規化画素値(0-1)
    yn: 黒レベル除去 & フルスケール正規化した輝度(0-1, yn(1)=1)
    """
    pts = [(p[0] / 255.0, p[1]) for p in ramp_channel if p[1] is not None]
    pts.sort(key=lambda t: t[0])
    v = np.array([p[0] for p in pts], dtype=float)
    Y = np.array([p[1] for p in pts], dtype=float)
    Y0 = Y[np.argmin(v)] if remove_black else 0.0     # v=0 の輝度を黒レベルに
    Ymax = Y[np.argmax(v)]
    denom = (Ymax - Y0) if (Ymax - Y0) != 0 else 1.0
    yn = np.clip((Y - Y0) / denom, 0.0, None)          # 0-1 正規化
    return v, yn, Y0

def analyze_gamma(name, ramp_channel,
                  target_gamma=2.2,
                  plot_dir=None):
    """1チャネルの (1)直線性チェック (2)ガンマ推定 を実行し、プロットを出力する。
    戻り値 dict: gamma, r2, rms_resid, v, yn, Y0, resid_22, resid_fit
    ※ モデル選択は行わず、算出したfitガンマを各チャンネルのEOTFに採用する。
    """
    v, yn, Y0 = _prep_ramp(ramp_channel)
    # 非線形空間のプロット用に、正規化前の実測輝度 [cd/m^2] も保持する
    pts_raw = sorted(
        [(p[0] / 255.0, p[1]) for p in ramp_channel if p[1] is not None],
        key=lambda t: t[0]
    )
    Y_meas = np.array([p[1] for p in pts_raw], dtype=float)
    Ymax = Y_meas[-1]
    Yrange = (Ymax - Y0) if (Ymax - Y0) != 0 else 1.0
    # ---- (1) gamma=2.2 からのズレ: 画素値^2.2 を横軸、測定輝度を縦軸に ----
    x_22     = v ** target_gamma      # 横軸: 画素値を2.2乗
    resid_22 = yn - x_22              # 直線 y=x からの残差（測定輝度基準）
    # ---- (2) 最小二乗(log-log)で実ガンマをフィット ----
    m = (v > 0) & (yn > 0)
    gamma, _ = curve_fit(lambda x, g: x ** g, v[m], yn[m], p0=[2.2])
    gamma = float(gamma[0])
    # フィットしたガンマでの線形空間残差 (yn vs v^gamma)
    yn_fit    = v ** gamma
    resid_fit = yn - yn_fit
    ss_res = np.sum(resid_fit[m] ** 2)
    ss_tot = np.sum((yn[m] - yn[m].mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rms_resid = float(np.sqrt(np.mean(resid_fit[m] ** 2)))
    # ---- ログ出力（直線性・ガンマの把握用。モデル選択は行わない）----
    print(f"\n---- [{name}] 直線性 & ガンマ解析 ----")
    print(f"  fit gamma = {gamma:.3f}  (target {target_gamma})  Δ={gamma-target_gamma:+.3f}  (by non-linear LSQ)")
    print(f"  R^2(log-log) = {r2:.4f}")
    print(f"  残差RMS(vs v^gamma) = {rms_resid:.4f}")
    # ---- プロット ----
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        fig, ax = plt.subplots(2, 2, figsize=(11, 8))
        # (1) 直線性: 横軸 v^2.2 vs 測定輝度 yn
        ax[0, 0].plot([0, 1], [0, 1], "k--", lw=1, label=f"ideal γ={target_gamma} (y=x)")
        ax[0, 0].plot(x_22, yn, "o-", label="measured")
        ax[0, 0].set_title(f"[{name}] linearity: pixel^{target_gamma} vs measured")
        ax[0, 0].set_xlabel(f"v^{target_gamma}"); ax[0, 0].set_ylabel("yn (measured, 0-1)")
        ax[0, 0].legend(); ax[0, 0].grid(True, alpha=.3)
        # (1) 2.2 との残差プロット
        ax[0, 1].axhline(0, color="k", lw=1)
        ax[0, 1].plot(v, resid_22, "o-")
        ax[0, 1].set_title(f"[{name}] residual vs γ={target_gamma}")
        ax[0, 1].set_xlabel("v (0-1)"); ax[0, 1].set_ylabel(f"yn - v^{target_gamma}")
        ax[0, 1].grid(True, alpha=.3)
        # (2) フィット後γで左上と同じ見方に:
        #     横軸 = 画素値にフィットγを適用 (v^γ_fit), 縦軸 = 測定輝度 yn, 理想線 y=x
        x_fit = v ** gamma
        ax[1, 0].plot([0, 1], [0, 1], "k--", lw=1, label="ideal (y=x)")
        ax[1, 0].plot(x_fit, yn, "o-", label=f"measured (γ_fit={gamma:.3f})")
        ax[1, 0].set_title(f"[{name}] linearity: pixel^γ_fit vs measured (R^2={r2:.4f})")
        ax[1, 0].set_xlabel("v^γ_fit"); ax[1, 0].set_ylabel("yn (measured, 0-1)")
        ax[1, 0].legend(); ax[1, 0].grid(True, alpha=.3)
        # (2) フィットガンマとの残差プロット
        ax[1, 1].axhline(0, color="k", lw=1)
        ax[1, 1].plot(v, resid_fit, "o-")
        ax[1, 1].set_title(f"[{name}] residual vs v^γ  (RMS={rms_resid:.4f})")
        ax[1, 1].set_xlabel("v (0-1)"); ax[1, 1].set_ylabel("yn - v^γ")
        ax[1, 1].grid(True, alpha=.3)
        fig.tight_layout()
        path = os.path.join(plot_dir, f"gamma_analysis_{name}.png")
        fig.savefig(path, dpi=120); plt.close(fig)
        print(f"  [plot] {path}")

        # ---- 非線形空間: 横軸=画素値、縦軸=実測輝度 [cd/m^2] ----
        # 黒レベルと最大輝度を実測値に合わせたガンマ曲線と比較する
        Y_model_22  = Y0 + Yrange * (v ** target_gamma)
        Y_model_fit = Y0 + Yrange * (v ** gamma)

        fig_nl, ax_nl = plt.subplots(1, 2, figsize=(12, 4.5))

        ax_nl[0].plot(v, Y_meas, "o-", label="measured")
        ax_nl[0].plot(v, Y_model_22, "k--", lw=2,
                      label=f"gamma curve (γ={target_gamma})")
        ax_nl[0].set_title(f"[{name}] nonlinear space: measured vs γ={target_gamma}")
        ax_nl[0].set_xlabel("pixel value v (0-1)")
        ax_nl[0].set_ylabel("luminance Y [cd/m²]")
        ax_nl[0].legend(); ax_nl[0].grid(True, alpha=.3)

        ax_nl[1].plot(v, Y_meas, "o-", label="measured")
        ax_nl[1].plot(v, Y_model_fit, "k--", lw=2,
                      label=f"fitted gamma curve (γ={gamma:.3f})")
        ax_nl[1].set_title(f"[{name}] nonlinear space: measured vs fitted gamma")
        ax_nl[1].set_xlabel("pixel value v (0-1)")
        ax_nl[1].set_ylabel("luminance Y [cd/m²]")
        ax_nl[1].legend(); ax_nl[1].grid(True, alpha=.3)

        fig_nl.tight_layout()
        path_nl = os.path.join(plot_dir, f"gamma_nonlinear_{name}.png")
        fig_nl.savefig(path_nl, dpi=120); plt.close(fig_nl)
        print(f"  [plot] {path_nl}")
    return {
        "name": name, "gamma": float(gamma),
        "r2": float(r2), "rms_resid": rms_resid,
        "v": v, "yn": yn, "Y": Y_meas, "Y0": float(Y0),
        "resid_22": resid_22, "resid_fit": resid_fit,
    }

def build_eotf_auto(ramp_channel, analysis=None, **kwargs):
    """fitしたチャンネル別ガンマから (g, g_inv, info) を構築する。
    g(v)  = v ** gamma          画素値 -> 正規化線形
    g_inv = linear ** (1/gamma) 正規化線形 -> 画素値
    """
    if analysis is None:
        name = kwargs.pop("name", "(auto)")
        analysis = analyze_gamma(name, ramp_channel, **kwargs)
    gamma = float(analysis["gamma"])
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError(f"invalid fitted gamma: {gamma}")
    g = lambda val: np.power(
        np.clip(np.asarray(val, dtype=float), 0.0, 1.0), gamma
    )
    g_inv = lambda linear: np.power(
        np.clip(np.asarray(linear, dtype=float), 0.0, None), 1.0 / gamma
    )
    return g, g_inv, analysis

# ---- 図の出力先 ----
FIG_DIR      = os.path.join("results", "figures", "DisplayBrightness")
RAMP_FIG_DIR = os.path.join(FIG_DIR, "ramps")
# 階調解析結果(残差・直線性プロット)は ramps/<出力日付> フォルダに保存
_date          = datetime.datetime.now().strftime("%Y-%m-%d")
GAMMA_PLOT_DIR = os.path.join(RAMP_FIG_DIR, _date)

def _build_eotf_set(ramp, tag):
    """各チャネルのfitガンマ関数と解析情報を構築する。"""
    funcs, infos = {}, {}
    for c in CHANNELS:
        g, g_inv, info = build_eotf_auto(
            ramp[c], name=f"{tag}_{c}", plot_dir=GAMMA_PLOT_DIR
        )
        funcs[c] = (g, g_inv)
        infos[c] = info
    return funcs, infos

_gb, _gamma_info_bg = _build_eotf_set(RAMP_BG, "BG")
_gf, _gamma_info_fg = _build_eotf_set(RAMP_FG, "FG")

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
    """channel='BGT' or 'FGR' の有効パッチを (rgb_lin, XYZ, names) で返す"""
    idx = 2 if channel == "BGT" else 3
    rgb_lin, XYZ, names = [], [], []
    for p in PATCHES:
        meas = p[idx]
        if meas is None or any(v is None for v in meas):
            continue  # 未測定はスキップ
        lin = g_b(p[1]) if channel == "BGT" else g_f(p[1])   # 背景=g_b / 前景=g_f で線形化
        rgb_lin.append(lin)
        XYZ.append(_yxy2xyz(*meas))
        names.append(p[0])
    return np.array(rgb_lin), np.array(XYZ), names

def fit_matrix(rgb_lin, XYZ):
    """
    XYZ = M @ rgb_lin を最小二乗で解く (M: 3x3)
    行形式: rgb_lin(N,3) @ M.T = XYZ(N,3)
    """
    Mt, *_ = np.linalg.lstsq(rgb_lin, XYZ, rcond=None)  # A @ Mt = Bm -> Mt = M.T
    return Mt.T

# ===================== 推定 =====================
rgb_T, XYZ_T, names_T = collect("BGT")
rgb_R, XYZ_R, names_R = collect("FGR")

T_prime = fit_matrix(rgb_T, XYZ_T)   # T·Db
R_prime = fit_matrix(rgb_R, XYZ_R)   # R·Df
C = np.linalg.inv(R_prime) @ T_prime      # = (R·Df)^(-1)(T·Db)

np.set_printoptions(precision=6, suppress=True)
print("T_prime:\n", T_prime)
print("R_prime:\n", R_prime)
print("C = inv(R') @ T':\n", C)

# =============================================================
# R', T', C を CSV 保存（別プログラムから参照する用）
#   保存先: results/tables/DisplayBrightness/
#   ・毎回同じファイル名で上書き保存
#   ・区切りはカンマ、# 始まりのヘッダ行付き（np.loadtxt は自動スキップ）
# =============================================================
TABLE_DIR = DATA_ROOT   # 行列CSVの保存先（入力CSVと同じルート）
os.makedirs(TABLE_DIR, exist_ok=True)

def save_matrix_csv(mat, name, header):
    """3x3 行列を CSV で上書き保存。np.loadtxt(delimiter=',') でそのまま読める。"""
    path = os.path.join(TABLE_DIR, f"{name}.csv")
    np.savetxt(
        path,
        np.asarray(mat, dtype=float),
        delimiter=",",
        fmt="%.10g",
        header=header,        # comments 既定"#"なので loadtxt はこの行をスキップ
    )
    return path

_saved = [
    save_matrix_csv(T_prime, "T_prime", "T_prime (T*Db): RGB_linear -> XYZ, row-major 3x3"),
    save_matrix_csv(R_prime, "R_prime", "R_prime (R*Df): RGB_linear -> XYZ, row-major 3x3"),
    save_matrix_csv(C,       "C",       "C = inv(R') @ T': RGB_linear(bg) -> RGB_linear(fg), row-major 3x3"),
]
print("\n[CSV保存] R'/T'/C を上書き保存しました ->", os.path.abspath(TABLE_DIR))
for _p in _saved:
    print("  ", _p)

# =============================================================
# fitしたチャンネル別ガンマと拡張輝度LUTをCSV保存
# =============================================================
def save_gamma_csv(gamma_info, name):
    """channel,gamma を保存。実験プログラムはこの値でべき乗変換する。"""
    path = os.path.join(TABLE_DIR, f"{name}.csv")
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["channel", "gamma"])
        for ch in CHANNELS:
            writer.writerow([ch, f"{float(gamma_info[ch]['gamma']):.10g}"])
    return path

_saved_gamma = [
    save_gamma_csv(_gamma_info_bg, "gamma_bg"),
    save_gamma_csv(_gamma_info_fg, "gamma_fg"),
]
print("\n[CSV保存] チャンネル別fitガンマを保存しました ->", os.path.abspath(TABLE_DIR))
for _p in _saved_gamma:
    print("  ", _p)

# =============================================================
# 単一プレーンLUT: 実測の加算結果から「前景1枚で目標輝度を再現」する画像を生成
#   入力: results/tables/DisplayBrightness/foreground_add/*.csv （加算後の測定値 Y,x,y）
#   方法: 加算後 Yxy -> XYZ -> inv(R') -> 前景線形 -> g_f_inv で前景画素値を求め、
#         加算後輝度 Y をキーに 1D LUT(線形補間・範囲外は端点クランプ)を構築する。
#   出力: 目標輝度ごとの前景画像 AddSim_Y**.png を foreground_add に保存。
# =============================================================
FG_ADD_DIR  = os.path.join(FIG_DIR, "foreground_add")
ADD_CSV_DIR = os.path.join(TABLE_DIR, "foreground_add")
ADD_TARGETS = [0, 10, 20, 30, 40, 50, 60]   # 前景1枚で再現したい加算輝度(cd/m^2)
PATCH       = 256                            # 出力画像の一辺(px)

R_prime_inv = np.linalg.inv(R_prime)         # XYZ -> 前景線形RGB

def load_add_measurements(dir_path):
    """foreground_add/*.csv から加算後の (Y, x, y) を読み込む。CSVが無ければ空リスト。"""
    files = sorted(glob.glob(os.path.join(dir_path, "*.csv")))
    if not files:
        print(f"[単一プレーンLUT] 加算測定CSVが見つかりません（スキップ）: {dir_path}")
        return []
    out = []
    for path in files:
        for row in _read_csv_dicts(path):
            try:
                Y = float(_get(row, "Y", "Y_add", "Yadd"))
                x = float(_get(row, "x")); y = float(_get(row, "y"))
            except (KeyError, ValueError):
                continue
            out.append((Y, x, y))
    return out

_add_meas = load_add_measurements(ADD_CSV_DIR)
if not _add_meas:
    print("[単一プレーンLUT] 加算測定が無いため LUT 構築・画像出力をスキップします")
else:
    # 各測定: 加算後 Yxy -> XYZ -> inv(R') -> 前景線形 -> g_f_inv -> 前景画素値(0-1)
    _Y  = np.array([m[0] for m in _add_meas], dtype=float)
    _px = np.array([np.clip(g_f_inv(np.clip(R_prime_inv @ _yxy2xyz(*m), 0.0, None)), 0.0, 1.0)
                    for m in _add_meas])

    # 加算後輝度 Y をキーに昇順化（同一 Y は平均して単調化: np.interp 用）
    _uY, _inv = np.unique(np.round(_Y, 4), return_inverse=True)
    _uPx = np.array([_px[_inv == i].mean(axis=0) for i in range(len(_uY))])

    def add_sim_pixel(Y_target):
        """加算後の目標輝度 -> 前景画素値(0-1)。範囲外は端点にクランプ。"""
        px = np.array([np.interp(float(Y_target), _uY, _uPx[:, i]) for i in range(3)])
        return np.clip(px, 0.0, 1.0)
    
    # 拡張輝度LUTをCSVに保存
    Y_grid = np.linspace(0.0, float(_uY.max()), 1024)
    px_grid = np.stack([np.interp(Y_grid, _uY, _uPx[:,c]) for c in range(3)], axis=1)
    lut_path = os.path.join(TABLE_DIR, "ext_lum_lut.csv")
    np.savetxt(lut_path,
               np.column_stack([Y_grid, px_grid]), delimiter=",",
               header="Y,pxR,pxG,pxB", comments="")
    print(f"\n[CSV保存] 拡張輝度LUTを保存しました -> {os.path.abspath(lut_path)}")

    os.makedirs(FG_ADD_DIR, exist_ok=True)
    print("\n==== 単一プレーンLUT（実測加算ベース）====")
    print(f"入力CSV: {ADD_CSV_DIR}")
    print(f"測定点 {len(_add_meas)} 件 / 有効輝度 {len(_uY)} 段 / レンジ {_uY.min():.2f}〜{_uY.max():.2f} cd/m^2")
    print(f"{'target':>7} {'画素値(0-255)':>16} {'備考':>10}")

    for Yt in ADD_TARGETS:
        px   = add_sim_pixel(Yt)
        note = "" if (_uY.min() - 1e-9) <= Yt <= (_uY.max() + 1e-9) else "★範囲外→クランプ"
        plt.imsave(os.path.join(FG_ADD_DIR, f"AddSim_Y{int(Yt):02d}.png"),
                   np.tile(px, (PATCH, PATCH, 1)))
        print(f"{Yt:7d} {str(np.round(px*255).astype(int)):>16} {note:>10}")
    print(f"[保存] 前景画像 {len(ADD_TARGETS)} 枚 -> {FG_ADD_DIR}")