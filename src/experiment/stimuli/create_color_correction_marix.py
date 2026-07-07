import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# =============================================================
# 最小二乗版: T', R' を「純色 + 白 + CMY + グレー」から推定
#   モデル: XYZ = M @ rgb_lin   (M = T'(透過) または R'(反射), 3x3)
#   劣加法(混色で輝度過大)を平均的に吸収させるのが狙い。
#   ※ 純色のみだと白/混色は span 内で情報ゼロ。混色を足して初めて効く。
# =============================================================

# =============================================================
# 入力パッチを CSV から読み込み（複数ファイルをXYZ空間で平均）
#   results/tables/DisplayBrightness/{background,foreground}/patches/*.csv
#   列: name, sR, sG, sB, Y, x, y  （ヘッダ行あり / 1ファイル=1測定）
#   ・平均は Yxy->XYZ に変換して XYZ空間で平均 -> Yxy に戻す
#   ・欠損（あるファイルに無いパッチ）は「あるものだけ」で平均
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
for _p in PATCHES:
    _b = None if _p[2] is None else tuple(round(v, 4) for v in _p[2])
    _f = None if _p[3] is None else tuple(round(v, 4) for v in _p[3])
    print(f"  {_p[0]:>6}  sRGB={tuple(round(c,3) for c in _p[1])}  BGT={_b}  FGR={_f}")

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

# ============================================================
# 実測EOTF g の構築（ページ「階調ランプコード」より移植）
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

def build_eotf(ramp_channel):
    """1チャネルのランプ [(pixel,Y,x,y),...] -> 正規化EOTF。g(1)=1。未計測(None)はスキップ。"""
    pts = [(p[0] / 255.0, p[1]) for p in ramp_channel if p[1] is not None]
    pts.sort(key=lambda t: t[0])
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    y_full = ys[np.argmax(xs)]           # フルスケール輝度
    yn = ys / y_full                     # g(1)=1 へ正規化
    def g(v):     return np.interp(np.asarray(v, dtype=float), xs, yn)   # 画素値 -> 正規化線形
    def g_inv(y): return np.interp(np.asarray(y, dtype=float), yn, xs)   # 正規化線形 -> 画素値
    return g, g_inv

# =============================================================
# 追加: 階調ランプの直線性チェック & ガンマ推定 & モデル自動選択
#   (1) XYZ^(1/2.2) を横軸に取り、画素値との直線性を確認
#         完全に gamma=2.2 なら yn = v^2.2  →  yn^(1/2.2) = v (直線 y=x)
#         → y=x からの残差プロットで「2.2 とのズレ」を見る
#   (2) 最小二乗(log-log)で実ガンマをフィット
#         log(yn) = gamma*log(v)  の傾き=gamma / 決定係数 R^2
#         → フィットしたガンマとの残差(yn - v^gamma)プロット
#   (3) 残差が均一  → フィットしたガンマ(べき乗モデル)を採用
#       残差が不均一→ 線形補完(np.interp: 既存 build_eotf 相当)を採用
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
                  resid_tol=0.02,   # 均一とみなす残差RMSの上限(正規化輝度)
                  trend_tol=0.5,    # 残差と入力の相関係数の上限(系統ズレ判定)
                  r2_tol=0.99,      # log-log フィットの決定係数の下限
                  plot_dir=None):
    """1チャネルの (1)直線性チェック (2)ガンマ推定 (3)モデル選択 を実行。
    戻り値 dict: gamma, r2, use('power'|'interp'), rms_resid, trend, uniform ...
    """
    v, yn, Y0 = _prep_ramp(ramp_channel)
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
    # ---- (3) 残差の均一性を判定 → モデル選択 ----
    # 均一の目安:
    # (a) 残差RMSが小さい (resid_tol以下)
    # (b) 残差に系統的トレンドが無い(残差とvの相関が小さい)
    # (c) log-log フィットの当てはまりが良い(R^2が高い)
    rms_resid = float(np.sqrt(np.mean(resid_fit[m] ** 2)))
    if np.std(resid_fit[m]) > 0 and np.std(v[m]) > 0:
        trend = float(abs(np.corrcoef(v[m], resid_fit[m])[0, 1]))
    else:
        trend = 0.0
    uniform = (rms_resid <= resid_tol) and (trend <= trend_tol) and (r2 >= r2_tol)
    use = "power" if uniform else "interp"
    # ---- ログ出力 ----
    print(f"\n---- [{name}] 直線性 & ガンマ解析 ----")
    print(f"  fit gamma = {gamma:.3f}  (target {target_gamma})  Δ={gamma-target_gamma:+.3f}  (by non-linear LSQ)")
    print(f"  R^2(log-log) = {r2:.4f}")
    print(f"  残差RMS(vs v^gamma) = {rms_resid:.4f} / トレンド(相関) = {trend:.3f}")
    print(f"  → 残差は{'均一' if uniform else '不均一'} → 採用モデル: "
          f"{'べき乗ガンマ(v^gamma)' if use=='power' else '線形補完(np.interp)'}")
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
    return {
        "name": name, "gamma": float(gamma),
        "r2": float(r2), "rms_resid": rms_resid, "trend": trend,
        "uniform": bool(uniform), "use": use,
        "v": v, "yn": yn, "Y0": float(Y0),
        "resid_22": resid_22, "resid_fit": resid_fit,
    }

def build_eotf_auto(ramp_channel, analysis=None, **kwargs):
    """解析結果に基づき EOTF を構築して (g, g_inv, info) を返す。
    use='power'  : g(v)=v^gamma,  g_inv(y)=y^(1/gamma)  (残差が均一)
    use='interp' : 既存 build_eotf と同じ線形補完          (残差が不均一)
    g(1)=1 に正規化。既存 _gb/_gf の代わりに使える。
    """
    if analysis is None:
        name = kwargs.pop("name", "(auto)")
        analysis = analyze_gamma(name, ramp_channel, **kwargs)
    v, yn = analysis["v"], analysis["yn"]
    if analysis["use"] == "power":
        gm = analysis["gamma"]
        g     = lambda val: np.clip(np.asarray(val, dtype=float), 0.0, None) ** gm
        g_inv = lambda y:   np.clip(np.asarray(y,   dtype=float), 0.0, None) ** (1.0 / gm)
    else:
        # 線形補完(既存 build_eotf 相当)。yn は単調前提で np.interp。
        g     = lambda val: np.interp(np.asarray(val, dtype=float), v, yn)
        g_inv = lambda y:   np.interp(np.asarray(y,   dtype=float), yn, v)
    return g, g_inv, analysis

import datetime

# ---- 図の出力先 ----
FIG_DIR      = os.path.join("results", "figures", "DisplayBrightness")
RAMP_FIG_DIR = os.path.join(FIG_DIR, "ramps")
# 階調解析結果(残差・直線性プロット)は ramps/<出力日付> フォルダに保存
_date          = datetime.datetime.now().strftime("%Y-%m-%d")
GAMMA_PLOT_DIR = os.path.join(RAMP_FIG_DIR, _date)

_gb = {}
_gf = {}
for c in CHANNELS:
    g_b, g_b_inv, _ = build_eotf_auto(RAMP_BG[c], name=f"BG_{c}", plot_dir=GAMMA_PLOT_DIR)
    _gb[c] = (g_b, g_b_inv)

for c in CHANNELS:
    g_f, g_f_inv, _ = build_eotf_auto(RAMP_FG[c], name=f"FG_{c}", plot_dir=GAMMA_PLOT_DIR)
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
            continue  # 未測定はスキップ
        lin = g_b(p[1]) if channel == "BGT" else g_f(p[1])   # 背景=g_b / 前景=g_f で線形化
        rgb_lin.append(lin)
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
# 追加: R', T', C を CSV 保存（実験プログラムから参照する用）
#   保存先: results/tables/DisplayBrightness/
#   ・毎回同じファイル名で np.savetxt するので実行のたびに上書きされる
#   ・区切りはカンマ、# 始まりのヘッダ行付き（np.loadtxt は自動スキップ）
# =============================================================
TABLE_DIR = os.path.join("results", "tables", "DisplayBrightness")
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