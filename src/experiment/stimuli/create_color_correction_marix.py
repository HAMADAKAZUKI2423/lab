import os
import csv
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =============================================================
# 実測階調ランプから色変換行列 T'(透過)・R'(反射)・C(=inv(R')·T') を推定し CSV 保存する
#   モデル: XYZ = M @ rgb_lin
#   background/foreground の ramps 全CSVを(channel, pixel)ごとにXYZ空間で平均し、
#   pixel=128,255 の R/G/B/W を最小二乗フィットに使用する。
# =============================================================

DATA_ROOT = os.path.join("results", "tables", "DisplayBrightness")

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


def estimate_ramp_common_black_xyz(*ramps):
    """BG/FGランプの全channel・pixel=0をXYZ空間で平均する。"""
    samples = []
    for ramp in ramps:
        for channel in ("R", "G", "B", "W"):
            for pixel, Y, x, y in ramp.get(channel, []):
                if int(pixel) == 0:
                    samples.append(_yxy2xyz(Y, x, y))
    if not samples:
        raise ValueError("BG/FGランプにpixel=0の共通黒測定がありません")
    return np.mean(np.asarray(samples, dtype=float), axis=0)


RAMP_COMMON_BLACK_XYZ = estimate_ramp_common_black_xyz(RAMP_BG, RAMP_FG)
RAMP_COMMON_BLACK_Y = float(RAMP_COMMON_BLACK_XYZ[1])
print("[共通黒] ramp XYZ =", np.round(RAMP_COMMON_BLACK_XYZ, 6))

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
    Y0 = RAMP_COMMON_BLACK_Y if remove_black else 0.0  # 全ランプのpixel=0平均を共通黒に
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

FIT_LEVELS = (128, 255)
FIT_CHANNELS = ("R", "G", "B", "W")


def _ramp_input_rgb(channel, pixel):
    """ランプのchannel/pixelを表示入力RGB(0-1)へ戻す。"""
    value = float(pixel) / 255.0
    if channel == "R":
        return np.array([value, 0.0, 0.0])
    if channel == "G":
        return np.array([0.0, value, 0.0])
    if channel == "B":
        return np.array([0.0, 0.0, value])
    if channel == "W":
        return np.array([value, value, value])
    raise ValueError(f"unsupported ramp channel: {channel}")


def collect_ramp_fit(ramp, eotf, side):
    """平均済みrampsのpixel=128,255から行列フィット用(rgb_lin, XYZ, names)を作る。"""
    rgb_lin, xyz_values, names = [], [], []
    missing = []
    for channel in FIT_CHANNELS:
        by_pixel = {
            int(pixel): (float(Y), float(x), float(y))
            for pixel, Y, x, y in ramp.get(channel, [])
        }
        for pixel in FIT_LEVELS:
            if pixel not in by_pixel:
                missing.append(f"{channel}_{pixel}")
                continue
            Y, x, y = by_pixel[pixel]
            rgb_lin.append(eotf(_ramp_input_rgb(channel, pixel)))
            xyz_increment = _yxy2xyz(Y, x, y) - RAMP_COMMON_BLACK_XYZ
            xyz_values.append(xyz_increment)
            names.append(f"{channel}_{pixel}")

        selected = [(pixel, by_pixel[pixel][0]) for pixel in FIT_LEVELS if pixel in by_pixel]
        if len(selected) >= 2:
            luminances = np.array([item[1] for item in selected], dtype=float)
            if np.any(np.diff(luminances) < 0):
                print(f"WARN: {side}/{channel} のYが128から255で増加していません: {selected}")

    if missing:
        raise ValueError(
            f"{side} rampsに行列フィット用測定点が不足しています: {', '.join(missing)}"
        )
    return np.array(rgb_lin), np.array(xyz_values), names


def fit_matrix(rgb_lin, XYZ):
    """
    XYZ = M @ rgb_lin を最小二乗で解く (M: 3x3)
    行形式: rgb_lin(N,3) @ M.T = XYZ(N,3)
    """
    Mt, *_ = np.linalg.lstsq(rgb_lin, XYZ, rcond=None)
    return Mt.T


def report_matrix_fit(name, matrix, rgb_lin, xyz_measured, labels):
    """行列フィットのXYZ残差と入力行列の条件数を表示する。"""
    xyz_predicted = rgb_lin @ matrix.T
    residual = xyz_measured - xyz_predicted
    rmse_xyz = np.sqrt(np.mean(residual ** 2, axis=0))
    rmse_total = float(np.sqrt(np.mean(residual ** 2)))
    print(f"\n---- [{name}] ramps pixel=128,255 行列フィット検証 ----")
    print(f"  points={len(labels)} / condition={np.linalg.cond(rgb_lin):.3f}")
    print(f"  RMSE XYZ={np.round(rmse_xyz, 6)} / total={rmse_total:.6f}")
    for channel in FIT_CHANNELS:
        mask = np.array([label.startswith(f"{channel}_") for label in labels])
        channel_rmse = float(np.sqrt(np.mean(residual[mask] ** 2)))
        print(f"  {channel}: RMSE={channel_rmse:.6f}")


# ===================== rampsから推定 =====================
rgb_T, XYZ_T, names_T = collect_ramp_fit(RAMP_BG, g_b, "background")
rgb_R, XYZ_R, names_R = collect_ramp_fit(RAMP_FG, g_f, "foreground")

T_prime = fit_matrix(rgb_T, XYZ_T)   # T·Db
R_prime = fit_matrix(rgb_R, XYZ_R)   # R·Df
C = np.linalg.inv(R_prime) @ T_prime  # = (R·Df)^(-1)(T·Db)

print(f"cond(T_prime): {np.linalg.cond(T_prime):.3f}")
print(f"cond(R_prime): {np.linalg.cond(R_prime):.3f}")
print(f"cond(C): {np.linalg.cond(C):.3f}")

np.set_printoptions(precision=6, suppress=True)
print("T_prime:\n", T_prime)
print("R_prime:\n", R_prime)
print("C = inv(R') @ T':\n", C)
report_matrix_fit("T_prime/background", T_prime, rgb_T, XYZ_T, names_T)
report_matrix_fit("R_prime/foreground", R_prime, rgb_R, XYZ_R, names_R)

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
# 背景Whiteランプの絶対輝度LUTを保存
#   入力: background/ramps/*.csv の channel=W
#   出力: bg_luminance_lut.csv
#   実験プログラムはこのLUTを使って、目標輝度[cd/m²]を背景画素値へ変換する。
# =============================================================
def save_bg_luminance_lut(ramp_bg, table_dir):
    white_ramp = ramp_bg.get("W")
    if not white_ramp:
        raise ValueError(
            "背景ランプにWhiteチャンネル(W)がありません。"
            "background/ramps/*.csv の channel=W を確認してください。"
        )

    # load_ramps_avg()で同一画素値の複数測定はすでにXYZ空間平均されている。
    # np.interp用に実測輝度の昇順で保存する。
    rows = sorted(white_ramp, key=lambda row: row[1])
    path = os.path.join(table_dir, "bg_luminance_lut.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Target_Luminance(cd/m2)", "Pixel_Value", "x", "y"
        ])
        for pixel, Y, x, y in rows:
            luminance_increment = (
                0.0 if int(pixel) == 0
                else max(0.0, float(Y) - RAMP_COMMON_BLACK_Y)
            )
            writer.writerow([
                f"{luminance_increment:.10g}", int(pixel),
                f"{float(x):.10g}", f"{float(y):.10g}",
            ])
    return path

_bg_lut_path = save_bg_luminance_lut(RAMP_BG, TABLE_DIR)
print("\n[CSV保存] 背景Whiteランプの絶対輝度LUTを保存しました")
print("  ", os.path.abspath(_bg_lut_path))

# =============================================================
# 単一プレーンLUT: 実測の加算結果から「前景1枚で目標輝度を再現」する画像を生成
#   入力: results/tables/DisplayBrightness/foreground_add/add/*.csv （加算後の測定値 Y,x,y）
#   方法: 加算後 Yxy -> XYZ -> inv(R') -> 前景線形 -> g_f_inv で前景画素値を求め、
#         加算後輝度 Y をキーに 1D LUT(線形補間・範囲外は端点クランプ)を構築する。
#   出力: 目標輝度ごとの前景画像 AddSim_Y**.png を foreground_add に保存。
# =============================================================
FG_ADD_DIR  = os.path.join(FIG_DIR, "foreground_add")
ADD_CSV_DIR = os.path.join(TABLE_DIR, "foreground_add", "add")
ADD_TARGETS = [0, 10, 15, 25, 30, 35, 45, 50, 60]   # 前景1枚で再現したい加算輝度(cd/m^2)
PATCH       = 256                            # 出力画像の一辺(px)

R_prime_inv = np.linalg.inv(R_prime)         # XYZ -> 前景線形RGB

def load_add_measurements(dir_path):
    """foreground_add/add/*.csv から加算後の (Y, x, y) を読み込む。CSVが無ければ空リスト。"""
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

def load_add_common_black_xyz(dir_path):
    """加算CSVのbg=0・fg=0をXYZ空間で平均して共通黒を返す。"""
    samples = []
    for path in sorted(glob.glob(os.path.join(dir_path, "*.csv"))):
        for row in _read_csv_dicts(path):
            try:
                bg = float(_get(row, "bg", "BG", "bg_Y", "Ybg"))
                fg = float(_get(row, "fg", "FG", "fg_Y", "Yfg"))
                Y = float(_get(row, "Y", "Y_add", "Yadd"))
                x = float(_get(row, "x"))
                y = float(_get(row, "y"))
            except (KeyError, ValueError):
                continue
            if np.isclose(bg, 0.0) and np.isclose(fg, 0.0):
                samples.append(_yxy2xyz(Y, x, y))
    if not samples:
        raise ValueError("加算CSVにbg=0・fg=0の共通黒測定がありません")
    return np.mean(np.asarray(samples, dtype=float), axis=0)


_add_meas = load_add_measurements(ADD_CSV_DIR)
if not _add_meas:
    print("[単一プレーンLUT] 加算測定が無いため LUT 構築・画像出力をスキップします")
else:
    ADD_COMMON_BLACK_XYZ = load_add_common_black_xyz(ADD_CSV_DIR)
    ADD_COMMON_BLACK_Y = float(ADD_COMMON_BLACK_XYZ[1])
    print("[共通黒] add XYZ =", np.round(ADD_COMMON_BLACK_XYZ, 6))

    def _add_increment_xyz(measurement):
        return _yxy2xyz(*measurement) - ADD_COMMON_BLACK_XYZ

    # 各測定: 共通黒差引き後XYZ -> inv(R') -> 前景線形 -> g_f_inv -> 前景画素値(0-1)
    _Y = np.array([
        max(0.0, float(m[0]) - ADD_COMMON_BLACK_Y)
        for m in _add_meas
    ], dtype=float)
    _px = np.array([
        np.clip(
            g_f_inv(np.clip(R_prime_inv @ _add_increment_xyz(m), 0.0, None)),
            0.0,
            1.0,
        )
        for m in _add_meas
    ])

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

    # 加算実測を(BG設定輝度, FG設定輝度)ごとにXYZ空間で平均する。
    def load_component_add_measurements(dir_path):
        grouped = {}
        files = sorted(glob.glob(os.path.join(dir_path, "*.csv")))
        for path in files:
            for row in _read_csv_dicts(path):
                try:
                    bg = float(_get(row, "bg", "BG", "bg_Y", "Ybg"))
                    fg = float(_get(row, "fg", "FG", "fg_Y", "Yfg"))
                    measured_Y = float(_get(row, "Y", "Y_add", "Yadd"))
                    x = float(_get(row, "x"))
                    y = float(_get(row, "y"))
                except (KeyError, ValueError):
                    continue
                key = (round(bg, 4), round(fg, 4))
                grouped.setdefault(key, []).append(
                    _yxy2xyz(measured_Y, x, y)
                )
        return {
            key: np.mean(np.asarray(values), axis=0)
            for key, values in grouped.items()
        }

    sp_y_grid = None
    sp_px_grid = None
    _component_xyz = load_component_add_measurements(ADD_CSV_DIR)
    if _component_xyz:
        _component_xyz = {
            key: xyz - ADD_COMMON_BLACK_XYZ
            for key, xyz in _component_xyz.items()
        }
    if not _component_xyz:
        print("[加算sim LUT] bg/fg付き加算測定がないため生成をスキップします")
    else:
        # SP系test用の連続1次元LUTを作る。
        # 0〜15: 実測黒点とBG=15/FG=0のXYZをアフィン補間する。
        # 15〜45: BG=15の実測XYZを補間する。
        # 45〜60: BG=15/FG=30（合計45）とBG=30/FG=30（合計60）の
        # 実測XYZを補間する。物理的に一意な0/60条件をLUT端点へ固定する。
        # 同じ15/45アンカーを隣接区間で共有し、境界を連続にする。
        black_key = (0.0, 0.0)
        low_anchor_key = (15.0, 0.0)
        middle_end_key = (15.0, 30.0)
        high_endpoint_key = (30.0, 30.0)
        required_anchor_keys = (
            black_key, low_anchor_key,
            middle_end_key, high_endpoint_key,
        )
        missing_anchor_keys = [
            key for key in required_anchor_keys if key not in _component_xyz
        ]
        middle_samples = sorted(
            (round(bg + fg, 4), xyz)
            for (bg, fg), xyz in _component_xyz.items()
            if np.isclose(bg, 15.0) and 0.0 <= fg <= 30.0
        )
        required_middle_totals = {
            15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0,
        }
        available_middle_totals = {
            total for total, _ in middle_samples
        }
        missing_middle_totals = sorted(
            required_middle_totals - available_middle_totals
        )
        if missing_anchor_keys:
            print(
                "WARN: Single plane加算LUTの必須アンカーが不足しています: "
                f"{missing_anchor_keys}"
            )
        if missing_middle_totals:
            print(
                "WARN: Single plane加算LUTの中央区間の測定点が不足しています: "
                f"{missing_middle_totals}"
            )
        if missing_anchor_keys or len(middle_samples) < 2:
            print("WARN: Single plane加算LUTの有効測定点が不足しているため生成をスキップします")
        else:
            black_xyz = _component_xyz[black_key]
            xyz_15 = _component_xyz[low_anchor_key]
            xyz_45 = _component_xyz[middle_end_key]
            xyz_60 = _component_xyz[high_endpoint_key]
            middle_y = np.array(
                [total for total, _ in middle_samples], dtype=float
            )
            middle_xyz = np.array(
                [xyz for _, xyz in middle_samples], dtype=float
            )

            sp_y_grid = np.linspace(0.0, 60.0, 2048)
            sp_xyz_grid = np.empty((len(sp_y_grid), 3), dtype=float)

            low_mask = sp_y_grid <= 15.0
            low_alpha = sp_y_grid[low_mask, None] / 15.0
            sp_xyz_grid[low_mask] = (
                black_xyz[None, :]
                + low_alpha * (xyz_15 - black_xyz)[None, :]
            )

            middle_mask = (sp_y_grid > 15.0) & (sp_y_grid < 45.0)
            sp_xyz_grid[middle_mask] = np.stack([
                np.interp(
                    sp_y_grid[middle_mask], middle_y,
                    middle_xyz[:, channel],
                )
                for channel in range(3)
            ], axis=1)

            high_mask = sp_y_grid >= 45.0
            high_alpha = (sp_y_grid[high_mask, None] - 45.0) / 15.0
            sp_xyz_grid[high_mask] = (
                xyz_45[None, :]
                + high_alpha * (xyz_60 - xyz_45)[None, :]
            )

            sp_linear_rgb = sp_xyz_grid @ R_prime_inv.T
            sp_out_of_gamut = np.any(
                (sp_linear_rgb < 0.0) | (sp_linear_rgb > 1.0), axis=1
            )
            if np.any(sp_out_of_gamut):
                print(
                    "WARN: Single plane加算LUTに前景色域外の点があります。最初の入力輝度="
                    f"{sp_y_grid[np.flatnonzero(sp_out_of_gamut)[0]]:.2f}"
                )
            sp_px_grid = np.clip(
                g_f_inv(np.clip(sp_linear_rgb, 0.0, None)), 0.0, 1.0
            )
            singleplane_add_lut_path = os.path.join(TABLE_DIR, "singleplane_add_lut.csv")
            np.savetxt(
                singleplane_add_lut_path,
                np.column_stack([sp_y_grid, sp_px_grid]),
                delimiter=",",
                header="Y,pxR,pxG,pxB",
                comments="",
            )
            print(
                "\n[CSV保存] 区間別Single plane加算LUTを保存しました -> "
                f"{os.path.abspath(singleplane_add_lut_path)}"
            )
            print(
                "  0〜15=実測0–15補間 / 15〜45=実測補間 / "
                "45〜60=実測45–60補間"
            )

    os.makedirs(FG_ADD_DIR, exist_ok=True)
    print("\n==== 単一プレーンLUT（実測加算ベース）====")
    print(f"入力CSV: {ADD_CSV_DIR}")
    print(f"測定点 {len(_add_meas)} 件 / 有効輝度 {len(_uY)} 段 / レンジ {_uY.min():.2f}〜{_uY.max():.2f} cd/m^2")
    print(f"{'target':>7} {'画素値(0-255)':>16} {'備考':>10}")

    # 比較画像は必要なときだけTrueにする。本番Matrix生成には依存しない。
    SAVE_LUT_COMPARISON_IMAGES = False
    SAVE_TWO_C_COMPARISON_IMAGES = False

    if SAVE_LUT_COMPARISON_IMAGES:
        print(f"{'target':>7} {'LUT画素値(0-255)':>20} {'備考':>16}")
        for Yt in ADD_TARGETS:
            if sp_y_grid is not None and sp_px_grid is not None:
                px = np.array([
                    np.interp(
                        float(Yt), sp_y_grid, sp_px_grid[:, channel]
                    )
                    for channel in range(3)
                ])
                note = "区間別SP LUT"
            else:
                px = add_sim_pixel(Yt)
                note = "★SP LUT未生成→旧LUT"
            plt.imsave(
                os.path.join(
                    FG_ADD_DIR, f"AddSim_Y{int(Yt):02d}.png"
                ),
                np.tile(px, (PATCH, PATCH, 1)),
            )
            print(
                f"{Yt:7d} "
                f"{str(np.round(px * 255).astype(int)):>20} "
                f"{note:>16}"
            )

    # Matrix方式は実際の加算測定と同じBG/FG条件を個別に予測し、
    # 同一合計輝度ごとにXYZ空間で平均してからFG再現画像へ変換する。
    MATRIX_SIM_DIR = os.path.join(FG_ADD_DIR, "matrix_simulation")
    TWO_C_SIM_DIR = os.path.join(FG_ADD_DIR, "2c_simulation")
    os.makedirs(MATRIX_SIM_DIR, exist_ok=True)
    os.makedirs(TWO_C_SIM_DIR, exist_ok=True)
    MATRIX_BG_LEVELS = (0, 15, 30)
    MATRIX_FG_LEVELS = (0, 5, 10, 15, 20, 25, 30)

    bg_white_rows = sorted(
        (
            0.0 if int(pixel) == 0
            else max(0.0, float(Y) - RAMP_COMMON_BLACK_Y),
            float(pixel) / 255.0,
        )
        for pixel, Y, _x, _y in RAMP_BG["W"]
    )
    bg_white_y = np.array([row[0] for row in bg_white_rows])
    bg_white_pixel = np.array([row[1] for row in bg_white_rows])

    def reference_linear_for_luminance(Y_target):
        pixel = np.interp(
            float(Y_target), bg_white_y, bg_white_pixel
        )
        return g_b(np.array([pixel, pixel, pixel], dtype=float))

    matrix_xyz_by_total = {}
    for bg_target in MATRIX_BG_LEVELS:
        for fg_target in MATRIX_FG_LEVELS:
            linear_bg = reference_linear_for_luminance(bg_target)
            linear_fg_reference = reference_linear_for_luminance(
                fg_target
            )
            linear_fg = C @ linear_fg_reference
            xyz_sum = T_prime @ linear_bg + R_prime @ linear_fg
            total_target = int(bg_target + fg_target)
            matrix_xyz_by_total.setdefault(total_target, []).append(
                xyz_sum
            )

    print("\n==== Matrix平均加算シミュレート画像 ====")
    print(f"{'target':>7} {'n':>3} {'Matrix画素値(0-255)':>23}")
    for total_target in sorted(matrix_xyz_by_total):
        mean_xyz = np.mean(
            np.asarray(matrix_xyz_by_total[total_target]), axis=0
        )
        foreground_linear = R_prime_inv @ mean_xyz
        if np.any(foreground_linear < -1e-9) or np.any(
            foreground_linear > 1.0 + 1e-9
        ):
            print(
                "WARN: averaged MatrixSim is out of gamut: "
                f"Y={total_target}, "
                f"linearRGB={np.round(foreground_linear, 6)}"
            )
        matrix_px = np.clip(
            g_f_inv(np.clip(foreground_linear, 0.0, None)),
            0.0,
            1.0,
        )
        plt.imsave(
            os.path.join(
                MATRIX_SIM_DIR,
                f"MatrixSim_Y{int(total_target):02d}.png",
            ),
            np.tile(matrix_px, (PATCH, PATCH, 1)),
        )
        print(
            f"{total_target:7d} "
            f"{len(matrix_xyz_by_total[total_target]):3d} "
            f"{str(np.round(matrix_px * 255).astype(int)):>23}"
        )

    # 比較用2C方式は明示的に有効化したときだけ生成する。
    if SAVE_TWO_C_COMPARISON_IMAGES:
        print("\n==== 2Cシミュレート画像 ====")
        print(f"{'target':>7} {'2C画素値(0-255)':>20}")
        for total_target in sorted(matrix_xyz_by_total):
            half_reference_linear = reference_linear_for_luminance(
                float(total_target) / 2.0
            )
            foreground_linear_2c = 2.0 * (
                C @ half_reference_linear
            )
            if np.any(foreground_linear_2c < -1e-9) or np.any(
                foreground_linear_2c > 1.0 + 1e-9
            ):
                print(
                    "WARN: 2C simulation is out of gamut: "
                    f"Y={total_target}, "
                    f"linearRGB={np.round(foreground_linear_2c, 6)}"
                )
            two_c_px = np.clip(
                g_f_inv(np.clip(foreground_linear_2c, 0.0, None)),
                0.0,
                1.0,
            )
            plt.imsave(
                os.path.join(
                    TWO_C_SIM_DIR,
                    f"2CSim_Y{int(total_target):02d}.png",
                ),
                np.tile(two_c_px, (PATCH, PATCH, 1)),
            )
            print(
                f"{total_target:7d} "
                f"{str(np.round(two_c_px * 255).astype(int)):>20}"
            )

    if SAVE_LUT_COMPARISON_IMAGES:
        print(
            f"[保存] LUT前景画像 {len(ADD_TARGETS)}枚 "
            f"-> {FG_ADD_DIR}"
        )
    print(
        f"[保存] Matrix平均画像 {len(matrix_xyz_by_total)}枚 "
        f"-> {MATRIX_SIM_DIR}"
    )
    if SAVE_TWO_C_COMPARISON_IMAGES:
        print(
            f"[保存] 2C画像 {len(matrix_xyz_by_total)}枚 "
            f"-> {TWO_C_SIM_DIR}"
        )