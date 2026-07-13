# py .\src\experiment\stimuli\generate_gabor_and_noise_stimuli.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
import argparse
import os
from itertools import product
import glob
import csv
import math
import shutil
import datetime

# ==========================================
# 1. ユーザー環境設定 (ここを書き換えてください)
# ==========================================
# 実験環境の物理パラメータ
FG_DISTANCES_CM = [50, 60, 81] # 前景用の距離リスト (cm)
BG_DISTANCES_CM = [70, 100, 150] # 背景用の距離リスト (cm)
SCREEN_WIDTH_CM = 59.67    # 画面の横幅実寸 (cm) ※ベゼルを含まない表示領域
SCREEN_RES_X_PX = 2560    # 画面の横解像度 (px)
STIM_WIDTH_DEG = 7.9     # 刺激の幅 (度)
STIM_HEIGHT_DEG = 7.9     # 刺激の高さ (度)

# デフォーカスマッチング用刺激パラメータ
DEFOCUS_PATTERNS = ["checker", "checker_45", "stripe", "border", "noise"]
DEFOCUS_CPDS = [2, 4]
MATCH_MEAN_LUM = 15
MATCH_CONTRAST = 1.0


# キャリブレーション結果の保存先ディレクトリ (create_calibrated_gray_patches.py の出力先)
script_dir = os.path.dirname(os.path.abspath(__file__))
lab_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

# --- 保存設定 ---
OUTPUT_DIR_MATCHING = os.path.join(lab_root, "data", "processed", "images", "pre-experiment-matching")

# ==========================================
# 2. 計算用関数
# ==========================================
def calculate_ppd(distance_cm, width_cm, res_x):
    """
    視聴距離と画面サイズから PPD (Pixels Per Degree) を計算する
    """
    # 1cmあたりのピクセル数
    px_per_cm = res_x / width_cm
    
    # 視野角1度あたりの画面上の実寸 (cm)
    # tan(1deg) 近似を使用
    cm_per_degree = distance_cm * np.tan(np.deg2rad(1.0))
    
    # PPD算出
    ppd = cm_per_degree * px_per_cm
    return ppd

def get_stimulus_pixel_size(deg_w, deg_h, ppd):
    """
    視野角(度)から必要なピクセルサイズを算出
    """
    px_w = int(np.round(deg_w * ppd))
    px_h = int(np.round(deg_h * ppd))
    return px_w, px_h

def luminance_to_pixel(target_lum_map, lum_points, pixel_points):
    """
    輝度マップをキャリブレーションデータに基づいてピクセル値マップに変換する。
    np.interpを用いて線形補間を行う。
    :param target_lum_map: 変換したい輝度値のNumpy配列
    :param lum_points: キャリブレーション済みの輝度値のリスト（昇順）
    :param pixel_points: lum_pointsに対応するピクセル値のリスト（昇順）
    :return: 0-255にクリップされたピクセル値のNumpy配列
    """
    # np.interpで線形補間
    pixel_map = np.interp(target_lum_map, lum_points, pixel_points)
    # 0-255の範囲にクリップして返す
    return np.clip(pixel_map, 0, 255)

def load_and_prepare_calibration_data(log_dir, name="Calibration"):
    """
    指定ディレクトリ内の全てのCSVファイルを読み込み、輝度レベルごとに
    ピクセル値の平均を計算し、np.interp用にソートされた輝度とピクセルの配列を返す。
    """
    if not os.path.exists(log_dir):
        print(f"警告: ディレクトリが見つかりません: {log_dir}")
        return None, None
        
    csv_files = glob.glob(os.path.join(log_dir, "*.csv"))
    if not csv_files:
        print(f"警告: ディレクトリ内にCSVファイルが見つかりません: {log_dir}")
        return None, None
        
    print(f"{len(csv_files)}個のキャリブレーションデータを読み込み、平均化します: {log_dir}")
    
    # {target_lum: [pixel_val1, pixel_val2, ...]} の形式でデータを保持
    lum_pixel_data = {}
    
    for csv_file in csv_files:
        try:
            with open(csv_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        t_lum = float(row["Target_Luminance(cd/m2)"])
                        p_val = int(row["Pixel_Value"])
                        
                        if t_lum not in lum_pixel_data:
                            lum_pixel_data[t_lum] = []
                        lum_pixel_data[t_lum].append(p_val)
                        
                    except (ValueError, KeyError):
                        continue
        except Exception as e:
            print(f"警告: CSV読み込み中にエラーが発生しました ({os.path.basename(csv_file)}): {e}")
            continue
            
    if not lum_pixel_data:
        print(f"エラー: {log_dir} から有効なキャリブレーションデータを読み込めませんでした。")
        return None, None

    # 平均値を計算してリストに格納
    avg_lum_pixel_map = []
    # lum_pixel_data.items()をt_lumでソートする
    for t_lum, pixel_list in sorted(lum_pixel_data.items()):
        if pixel_list:
            avg_pixel = int(np.round(np.mean(pixel_list)))
            avg_lum_pixel_map.append((t_lum, avg_pixel))
            
    if not avg_lum_pixel_map:
        print(f"警告: {name} の有効なキャリブレーションデータがありません。")
        return None, None
    
    # ソート済みのタプルのリストを2つのNumpy配列に展開する
    calibrated_lums = np.array([item[0] for item in avg_lum_pixel_map])
    calibrated_pixels = np.array([item[1] for item in avg_lum_pixel_map])
    
    return calibrated_lums, calibrated_pixels

# ==========================================
# 3. 刺激生成関数 (前回のコードをベースに調整)
# ==========================================
def create_band_limited_noise(width_px, height_px, ppd, f_center_cpd, bandwidth_octave=1.0):
    """
    指定されたピクセルサイズで帯域制限ノイズを作成
    """
    # ホワイトノイズの生成
    white_noise = np.random.normal(0, 1, (height_px, width_px))
    
    # フーリエ変換
    ft_noise = fft2(white_noise)
    ft_noise = fftshift(ft_noise)
    
    # 周波数座標の作成 (cpd単位)
    fx = fftshift(fftfreq(width_px, d=1/ppd))
    fy = fftshift(fftfreq(height_px, d=1/ppd))
    FX, FY = np.meshgrid(fx, fy)
    R = np.sqrt(FX**2 + FY**2) # 中心からの距離（空間周波数）

    # バンドパスフィルタの作成
    f_min = f_center_cpd / (2 ** (bandwidth_octave / 2))
    f_max = f_center_cpd * (2 ** (bandwidth_octave / 2))
    
    # 理想的な矩形フィルタを適用 (実験用途によってはバターワース等を検討)
    mask = (R >= f_min) & (R <= f_max)
    
    # フィルタ適用
    ft_filtered = ft_noise * mask
    
    # 逆フーリエ変換
    noise_filtered = np.real(ifft2(ifftshift(ft_filtered)))
    
    # 正規化 (-1 〜 1 の範囲に収める)
    if np.max(np.abs(noise_filtered)) > 0:
        noise_filtered = noise_filtered / np.max(np.abs(noise_filtered))
    
    return noise_filtered

def create_defocus_pattern(pattern, w, h, ppd, cpd):
    """defocus matching 用の輝度変調 m∈[-1,+1] を返す。"""
    s = max(1, int(ppd / (2 * cpd)))       # 矩形波の半周期(px)
    x = np.arange(w); y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    
    if pattern == "checker":
        m = ((X // s) + (Y // s)) % 2
    elif pattern == "checker_45":
        # 回転座標 U=(X+Y)/(√2·s), V=(X-Y)/(√2·s) の (⌊U⌋+⌊V⌋) % 2
        U = np.floor((X + Y) / (math.sqrt(2.0) * s)).astype(int)
        V = np.floor((X - Y) / (math.sqrt(2.0) * s)).astype(int)
        m = (U + V) % 2
    elif pattern == "stripe":              # 矩形波・縦じま（x依存）
        m = (X // s) % 2
    elif pattern == "border":              # stripe を90°回転＝横じま（y依存）
        m = (Y // s) % 2
    elif pattern == "noise":
        return create_band_limited_noise(w, h, ppd, f_center_cpd=cpd)
    else:
        raise ValueError(f"unknown pattern: {pattern}")
    return m.astype(np.float64) * 2.0 - 1.0

def generate_defocus_matching_stimuli(output_dir, distances_fg, distances_bg):
    """デフォーカスマッチング用刺激(5パターン)を生成する。
    checker / checker_45 / stripe / border / noise を FG・BG それぞれ生成する。
    noise は生成スクリプト内の create_band_limited_noise を用いる（stimuli_utils は import しない）。

    ※ 校正・輝度補正は焼き込まない。experiment (defocus_matching.py) 側で test/ref と同じ
      「目標輝度→背景画素→C→前景画素」パイプラインで変換するため、ここでは正規化模様
      base∈[0,1]（m∈[-1,1] を (m+1)/2 で 0-255 に写像）のみを出力する。
    """
    print("\n--- Generating Defocus Matching Patterns (checker/checker_45/stripe/border/noise) ---")

    matching_dir = os.path.join(output_dir, "defocus-matching")
    os.makedirs(matching_dir, exist_ok=True)

    # (prefix, 距離リスト) で前景・背景をまとめて処理（校正は使わない）
    targets = [
        ("FG", distances_fg),
        ("BG", distances_bg),
    ]

    for prefix, distances in targets:
        for distance in distances:
            ppd = calculate_ppd(distance, SCREEN_WIDTH_CM, SCREEN_RES_X_PX)
            req_w, req_h = get_stimulus_pixel_size(STIM_WIDTH_DEG, STIM_HEIGHT_DEG/2, ppd)
            for pattern in DEFOCUS_PATTERNS:
                for cpd in DEFOCUS_CPDS:
                    mod_v = create_defocus_pattern(pattern, req_w, req_h, ppd, cpd)
                    # 生パターンのみ出力（校正・輝度補正なし）。m∈[-1,1] → base∈[0,1] → 0-255
                    pixel_map_v = np.clip((mod_v + 1.0) / 2.0 * 255.0, 0, 255)

                    filename = os.path.join(matching_dir, f"{prefix}_{pattern}_{distance}cm_{cpd}cpd.png")
                    plt.imsave(filename, pixel_map_v, cmap='gray', vmin=0, vmax=255)
                    print(f"  Saved {prefix} {pattern} ({distance}cm, {cpd}cpd): {filename}")

# ==========================================
# 4. メイン実行ブロック
# ==========================================
if __name__ == "__main__":
    # --- コマンドライン引数解析 ---
    parser = argparse.ArgumentParser(description="Defocus matching stimuli generator (pre-experiment-matching)")
    parser.parse_args()

    # --- [追加] 既存の出力フォルダをクリーンアップ ---
    # pre-experiment-matching 用のフォルダをクリーンアップ
    match_output_dir_new = os.path.join(OUTPUT_DIR_MATCHING, "defocus-matching")
    if os.path.exists(match_output_dir_new):
        print(f"既存のフォルダを削除します: {match_output_dir_new}")
        shutil.rmtree(match_output_dir_new)



    # --- デフォーカスマッチング用刺激(5パターン)の生成 (pre-experiment-matching用) ---
    # 校正は焼き込まず生パターンのみ生成（輝度・C変換は experiment 側で実行）。
    generate_defocus_matching_stimuli(
        output_dir=OUTPUT_DIR_MATCHING,
        distances_fg=FG_DISTANCES_CM,
        distances_bg=BG_DISTANCES_CM,
    )

    print("\n--- デフォーカスマッチング用刺激画像の生成が完了しました ---")