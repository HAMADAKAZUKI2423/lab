# py .\src\experiment\stimuli\generate_gabor_and_noise_stimuli.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
import argparse
import os
from itertools import product
import glob
import csv
import shutil
import datetime

# ==========================================
# 1. ユーザー環境設定 (ここを書き換えてください)
# ==========================================
# 実験環境の物理パラメータ
FG_DISTANCES_CM = [50] # 前景用の距離リスト (cm)
BG_DISTANCES_CM = [150] # 背景用の距離リスト (cm)
SCREEN_WIDTH_CM = 59.67    # 画面の横幅実寸 (cm) ※ベゼルを含まない表示領域
SCREEN_RES_X_PX = 2560    # 画面の横解像度 (px)
STIM_WIDTH_DEG = 7.9     # 刺激の幅 (度)
STIM_HEIGHT_DEG = 7.9     # 刺激の高さ (度)

# 刺激画像パラメータ
FG_SPATIAL_FREQS_CPD = [4] # 前景ガボールパッチの空間周波数のリスト (cpd)
BG_SPATIAL_FREQS_CPD = [4] # 背景ノイズの空間周波数リスト (cpd)
FG_MEAN_LUMINANCES_CDM2 = [35]   # 前景用平均輝度リスト (cd/m^2)
BG_MEAN_LUMINANCES_CDM2 = [15]   # 背景用平均輝度リスト (cd/m^2)
FG_CONTRASTS = [0.2, 0.5]      # 前景用コントラストリスト
BG_CONTRASTS = [1.0]      # 背景用コントラストリスト
# ガボールパッチパラメータ 
GABOR_SIGMA_DEG = 1.0     # ガボールパッチの標準偏差 (度)

# キャリブレーション結果の保存先ディレクトリ (create_calibrated_gray_patches.py の出力先)
script_dir = os.path.dirname(os.path.abspath(__file__))
lab_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

# --- 保存設定 ---
OUTPUT_DIR_PRE = os.path.join(lab_root, "data", "processed", "images", "pre-experiment-matching")
OUTPUT_DIR_MAIN = os.path.join(lab_root, "data", "processed", "images", "main-experiment-gabor")

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
def create_gabor(width_px, height_px, ppd, cpd, sigma_deg, orientation_deg=90, phase=0):
    """
    指定されたピクセルサイズでガボールパッチを作成
    """
    # グリッドの作成 (度単位の座標系)
    # 中心を0とする
    x = np.linspace(-width_px/2, width_px/2, width_px) / ppd
    y = np.linspace(-height_px/2, height_px/2, height_px) / ppd
    X, Y = np.meshgrid(x, y)
    
    # 向きの回転
    theta = np.deg2rad(orientation_deg)
    X_rot = X * np.cos(theta) + Y * np.sin(theta)
    
    # 正弦波成分 (Carrier)
    grating = np.sin(2 * np.pi * cpd * X_rot + phase)
    
    # ガウス窓成分 (Envelope)
    # 楕円ではなく円形の窓を適用するため、アスペクト比に関わらず距離を使用
    envelope = np.exp(-(X**2 + Y**2) / (2 * sigma_deg**2))
    
    # ガボールパッチ
    gabor = grating * envelope
    
    return gabor

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

def create_checkerboard(width_px, height_px, ppd, cpd, angle_deg=0):
    """
    指定されたピクセルサイズと空間周波数(cpd)で白黒のチェッカーボードを作成する。
    値は -1 と 1 で返す。角度指定(angle_deg)で回転可能。
    """
    square_size = ppd / (2 * cpd)
    
    x = np.arange(width_px) - width_px / 2
    y = np.arange(height_px) - height_px / 2
    X, Y = np.meshgrid(x, y)
    
    theta = np.deg2rad(angle_deg)
    X_rot = X * np.cos(theta) - Y * np.sin(theta)
    Y_rot = X * np.sin(theta) + Y * np.cos(theta)
    
    checker = ((np.floor(X_rot / square_size).astype(int)) + (np.floor(Y_rot / square_size).astype(int))) % 2
    checker = checker * 2 - 1
    
    return checker

def create_stripe(width_px, height_px, ppd, cpd, angle_deg=0):
    """
    指定されたピクセルサイズと空間周波数(cpd)で白黒のストライプ（またはボーダー）を作成する。
    値は -1 と 1 で返す。angle_deg=0で縦縞(ストライプ)、angle_deg=90で横縞(ボーダー)。
    """
    stripe_width = ppd / (2 * cpd)
    
    x = np.arange(width_px) - width_px / 2
    y = np.arange(height_px) - height_px / 2
    X, Y = np.meshgrid(x, y)
    
    theta = np.deg2rad(angle_deg)
    X_rot = X * np.cos(theta) - Y * np.sin(theta)
    
    stripe = (np.floor(X_rot / stripe_width).astype(int)) % 2
    stripe = stripe * 2 - 1
    
    return stripe

def generate_gabor_stimuli(output_dir, distances, fg_params_list, bg_params_list, calib_data):
    """前景用のガボール刺激画像をまとめて生成する"""
    print("--- Generating Gabor Patches (Foreground) ---")
    sorted_lums, sorted_pixels = calib_data

    for distance in distances:
        ppd = calculate_ppd(distance, SCREEN_WIDTH_CM, SCREEN_RES_X_PX)
        req_w, req_h = get_stimulus_pixel_size(STIM_WIDTH_DEG, STIM_HEIGHT_DEG, ppd)
        print(f"  Distance: {distance}cm (PPD: {ppd:.2f}, Size: {req_w}x{req_h}px)")
        
        gabor_base_dir = os.path.join(output_dir, "fg_gabor", f"{distance}cm")
        os.makedirs(gabor_base_dir, exist_ok=True)

        # 前景パラメータでループ
        for fg_cpd, fg_mean_lum, fg_contrast in fg_params_list:
            # 空間周波数のフォルダを作成
            gabor_dir = os.path.join(gabor_base_dir, f"{fg_cpd}cpd")
            os.makedirs(gabor_dir, exist_ok=True)

            # この前景条件に対するガボール画像を1枚生成
            gabor_mod_v = create_gabor(req_w, req_h, ppd, fg_cpd, GABOR_SIGMA_DEG, orientation_deg=0)
            lum_map_v = fg_mean_lum * (1 + fg_contrast * gabor_mod_v)
            pixel_map_v = luminance_to_pixel(lum_map_v, sorted_lums, sorted_pixels)

            # 背景パラメータでループし、同じ画像を異なる名前で保存
            for bg_cpd, bg_mean_lum, bg_contrast in bg_params_list:
                # 前景と背景の空間周波数が同じ場合のみ生成
                if fg_cpd != bg_cpd:
                    continue
                print(f"    Generating Gabor for FG(f={fg_cpd},L={fg_mean_lum},C={fg_contrast}) BG(f={bg_cpd},L={bg_mean_lum},C={bg_contrast})")

                # ファイル名の設定と保存
                gabor_filename = os.path.join(gabor_dir, f"FG_{fg_mean_lum}_{fg_contrast}_BG_{bg_mean_lum}_{bg_contrast}.png")
                plt.imsave(gabor_filename, pixel_map_v, cmap='gray', vmin=0, vmax=255)

def generate_noise_stimuli(output_dir, distances, fg_params_list, bg_params_list, calib_data):
    """背景用の帯域制限ノイズ画像をまとめて生成する"""
    print("\n--- Generating Band-limited Noise (Background) ---")
    sorted_lums, sorted_pixels = calib_data

    for distance in distances:
        ppd = calculate_ppd(distance, SCREEN_WIDTH_CM, SCREEN_RES_X_PX)
        # 背景ノイズは横長にする
        req_w, req_h = get_stimulus_pixel_size(STIM_WIDTH_DEG * 2, STIM_HEIGHT_DEG, ppd)
        print(f"  Distance: {distance}cm (PPD: {ppd:.2f}, Size: {req_w}x{req_h}px)")
        
        noise_base_dir = os.path.join(output_dir, "bg_noise", f"{distance}cm")
        os.makedirs(noise_base_dir, exist_ok=True)

        # 全パラメータの組み合わせでループ
        for fg_cpd, fg_mean_lum, fg_contrast in fg_params_list:
            for bg_cpd, bg_mean_lum, bg_contrast in bg_params_list:
                # 前景と背景の空間周波数が同じ場合のみ生成
                if fg_cpd != bg_cpd:
                    continue

                # 空間周波数のフォルダを作成
                noise_dir = os.path.join(noise_base_dir, f"{bg_cpd}cpd")
                os.makedirs(noise_dir, exist_ok=True)

                # このループの内部で毎回ノイズを生成し、全条件で異なるパターンにする
                stim_noise = create_band_limited_noise(req_w, req_h, ppd, f_center_cpd=bg_cpd)
                print(f"    Generating for FG(f={fg_cpd},L={fg_mean_lum},C={fg_contrast}) BG(f={bg_cpd},L={bg_mean_lum},C={bg_contrast})")

                # 輝度マップの計算とピクセル値への変換
                lum_map_noise = bg_mean_lum * (1 + bg_contrast * stim_noise)
                pixel_map_noise = luminance_to_pixel(lum_map_noise, sorted_lums, sorted_pixels)

                # ファイル名の設定と保存
                noise_filename = os.path.join(noise_dir, f"FG_{fg_mean_lum}_{fg_contrast}_BG_{bg_mean_lum}_{bg_contrast}.png")
                plt.imsave(noise_filename, pixel_map_noise, cmap='gray', vmin=0, vmax=255)

def generate_matching_stimuli(output_dir, distances_fg, distances_bg, calib_data):
    """デフォーカスマッチング用の刺激を生成する"""
    print("\n--- Generating Matching Patches (Defocus Matching) ---")
    fg_sorted_lums, fg_sorted_pixels = calib_data[0]
    bg_sorted_lums, bg_sorted_pixels = calib_data[1]

    matching_dir = os.path.join(output_dir, "defocus-matching")
    os.makedirs(matching_dir, exist_ok=True)

    cpds = [2, 4]  # 空間周波数 (cpd)
    mean_lum = 15
    contrast = 1.0

    # 刺激パターンのリスト: (名前, 関数, 角度)
    patterns = [
        ("checker", create_checkerboard, 0),
        ("checker_45", create_checkerboard, 45),
        ("stripe", create_stripe, 0),
        ("border", create_stripe, 90)
    ]

    # 前景用
    for distance in distances_fg:
        for cpd in cpds:
            ppd = calculate_ppd(distance, SCREEN_WIDTH_CM, SCREEN_RES_X_PX)
            req_w, req_h = get_stimulus_pixel_size(STIM_WIDTH_DEG, STIM_HEIGHT_DEG/2, ppd)
            
            for name, func, angle in patterns:
                mod_v = func(req_w, req_h, ppd, cpd, angle_deg=angle)
                lum_map_v = mean_lum * (1 + contrast * mod_v)
                pixel_map_v = luminance_to_pixel(lum_map_v, fg_sorted_lums, fg_sorted_pixels)
                
                filename = os.path.join(matching_dir, f"FG_{name}_{distance}cm_{cpd}cpd.png")
                plt.imsave(filename, pixel_map_v, cmap='gray', vmin=0, vmax=255)
                print(f"  Saved FG {name} ({distance}cm, {cpd}cpd): {filename}")

    # 背景用
    for distance in distances_bg:
        for cpd in cpds:
            ppd = calculate_ppd(distance, SCREEN_WIDTH_CM, SCREEN_RES_X_PX)
            req_w, req_h = get_stimulus_pixel_size(STIM_WIDTH_DEG, STIM_HEIGHT_DEG/2, ppd)
            
            for name, func, angle in patterns:
                mod_v = func(req_w, req_h, ppd, cpd, angle_deg=angle)
                lum_map_v = mean_lum * (1 + contrast * mod_v)
                pixel_map_v = luminance_to_pixel(lum_map_v, bg_sorted_lums, bg_sorted_pixels)
                
                filename = os.path.join(matching_dir, f"BG_{name}_{distance}cm_{cpd}cpd.png")
                plt.imsave(filename, pixel_map_v, cmap='gray', vmin=0, vmax=255)
                print(f"  Saved BG {name} ({distance}cm, {cpd}cpd): {filename}")

# ==========================================
# 4. メイン実行ブロック
# ==========================================
if __name__ == "__main__":
    # --- コマンドライン引数解析 ---
    parser = argparse.ArgumentParser(description="Gabor/noise stimuli generator: choose output target folder")
    parser.add_argument("--target", choices=["main", "pre"], required=True,
                        help="保存先を選択します: main または pre (必須)")
    args = parser.parse_args()

    if args.target == "main":
        OUTPUT_DIR = OUTPUT_DIR_MAIN
    elif args.target == "pre":
        OUTPUT_DIR = OUTPUT_DIR_PRE
    else:
        raise ValueError("不正なtarget指定です。mainかpreを指定してください。")

    # キャリブレーションフォルダの場所を固定
    fg_calib_dir = os.path.join(lab_root, "results", "tables", "DisplayBrightness", "fg_calibration_log")
    bg_calib_dir = os.path.join(lab_root, "results", "tables", "DisplayBrightness", "bg_calibration_log")

    print(f"OUTPUT_DIR = {OUTPUT_DIR}")

    # --- [追加] 既存の出力フォルダをクリーンアップ ---
    fg_output_base_dir = os.path.join(OUTPUT_DIR, "fg_gabor")
    bg_output_base_dir = os.path.join(OUTPUT_DIR, "bg_noise")
    match_output_dir = os.path.join(OUTPUT_DIR, "defocus-matching")

    if os.path.exists(fg_output_base_dir):
        print(f"既存のフォルダを削除します: {fg_output_base_dir}")
        shutil.rmtree(fg_output_base_dir)
    
    if os.path.exists(bg_output_base_dir):
        print(f"既存のフォルダを削除します: {bg_output_base_dir}")
        shutil.rmtree(bg_output_base_dir)

    if os.path.exists(match_output_dir):
        print(f"既存のフォルダを削除します: {match_output_dir}")
        shutil.rmtree(match_output_dir)

    # --- [新規] 輝度-ピクセル変換の準備 ---
    # 最新のキャリブレーション結果を読み込む (Foreground)
    print("\n--- [FG] キャリブレーションデータの読み込み ---")
    fg_sorted_lums, fg_sorted_pixels = load_and_prepare_calibration_data(fg_calib_dir, "Foreground")
    
    # 最新のキャリブレーション結果を読み込む (Background)
    print("\n--- [BG] キャリブレーションデータの読み込み ---")
    bg_sorted_lums, bg_sorted_pixels = load_and_prepare_calibration_data(bg_calib_dir, "Background")
    
    if fg_sorted_lums is None or bg_sorted_lums is None:
        print("エラー: 有効なキャリブレーションデータがないため、処理を中断します。")
        exit(1)

    # --- パラメータの組み合わせを生成 ---
    fg_params_list = list(product(FG_SPATIAL_FREQS_CPD, FG_MEAN_LUMINANCES_CDM2, FG_CONTRASTS))
    bg_params_list = list(product(BG_SPATIAL_FREQS_CPD, BG_MEAN_LUMINANCES_CDM2, BG_CONTRASTS))

    # --- ガボールパッチの生成 (前景距離に基づく) ---
    generate_gabor_stimuli(
        output_dir=OUTPUT_DIR,
        distances=FG_DISTANCES_CM,
        fg_params_list=fg_params_list,
        bg_params_list=bg_params_list,
        calib_data=(fg_sorted_lums, fg_sorted_pixels)
    )

    # --- 帯域制限ノイズの生成 (背景距離に基づく) ---
    generate_noise_stimuli(
        output_dir=OUTPUT_DIR,
        distances=BG_DISTANCES_CM,
        fg_params_list=fg_params_list,
        bg_params_list=bg_params_list,
        calib_data=(bg_sorted_lums, bg_sorted_pixels)
    )

    # --- デフォーカスマッチング用の刺激画像の生成 ---
    generate_matching_stimuli(
        output_dir=OUTPUT_DIR,
        distances_fg=FG_DISTANCES_CM,
        distances_bg=BG_DISTANCES_CM,
        calib_data=((fg_sorted_lums, fg_sorted_pixels), (bg_sorted_lums, bg_sorted_pixels))
    )

    print("\n--- 全ての刺激画像の生成が完了しました ---")