# py .\src\experiment\stimuli\generate_gabor_and_noise_stimuli.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import argparse
import os
import glob
import csv

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

# 刺激画像パラメータ
FG_SPATIAL_FREQS_CPD = [2, 4] # 前景ガボールパッチの空間周波数のリスト (cpd)
BG_SPATIAL_FREQS_CPD = [2, 4] # 背景ノイズの空間周波数リスト (cpd)
FG_MEAN_LUMINANCES_CDM2 = [50, 5]   # 前景用平均輝度リスト (cd/m^2)
BG_MEAN_LUMINANCES_CDM2 = [15, 5]   # 背景用平均輝度リスト (cd/m^2)
FG_CONTRASTS = [0.5, 1.0]      # 前景用コントラストリスト
BG_CONTRASTS = [0.5, 1.0]      # 背景用コントラストリスト

# ガボールパッチパラメータ 
GABOR_SIGMA_DEG = 1.0     # ガボールパッチの標準偏差 (度)

# キャリブレーション結果の保存先ディレクトリ (create_calibrated_gray_patches.py の出力先)
script_dir = os.path.dirname(os.path.abspath(__file__))
lab_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

# キャリブレーションログのベースパス
CALIB_LOG_BASE_DIR = os.path.join(lab_root, "results", "tables")

# --- 保存設定 ---
OUTPUT_DIR_PRE = os.path.join(lab_root, "data", "processed", "images", "pre-experiment-gabor")
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

def load_latest_calibration_map(log_dir):
    """
    指定ディレクトリ内の最新のCSVファイルを読み込み、
    (Target_Luminance, Pixel_Value) のリストを返す。
    """
    if not os.path.exists(log_dir):
        print(f"警告: ディレクトリが見つかりません: {log_dir}")
        return None
        
    csv_files = glob.glob(os.path.join(log_dir, "*.csv"))
    if not csv_files:
        print(f"警告: ディレクトリ内にCSVファイルが見つかりません: {log_dir}")
        return None
        
    # 作成日時が新しい順にソート
    latest_csv = max(csv_files, key=os.path.getctime)
    print(f"キャリブレーションデータを読み込み中: {latest_csv}")
    
    lum_pixel_map = []
    try:
        with open(latest_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    t_lum = float(row["Target_Luminance(cd/m2)"])
                    p_val = int(row["Pixel_Value"])
                    lum_pixel_map.append((t_lum, p_val))
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        print(f"エラー: CSV読み込み失敗: {e}")
        return None
        
    # 輝度でソート（昇順）
    lum_pixel_map.sort(key=lambda x: x[0])
    return lum_pixel_map

def get_calibrated_map_arrays(log_dir, name="Calibration"):
    """
    キャリブレーションディレクトリから最新のデータを読み込み、
    np.interp用にソートされた輝度とピクセルの配列を返す。
    """
    lum_pixel_map = load_latest_calibration_map(log_dir)
    if not lum_pixel_map:
        print(f"警告: {name} の有効なキャリブレーションデータがありません。")
        return None, None
    
    calibrated_lums = np.array([item[0] for item in lum_pixel_map])
    calibrated_pixels = np.array([item[1] for item in lum_pixel_map])
    
    # np.interpは輝度が昇順である必要がある
    sort_indices = np.argsort(calibrated_lums)
    return calibrated_lums[sort_indices], calibrated_pixels[sort_indices]

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
    fx = np.fft.fftshift(np.fft.fftfreq(width_px, d=1/ppd))
    fy = np.fft.fftshift(np.fft.fftfreq(height_px, d=1/ppd))
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
        fg_calib_dir = os.path.join(CALIB_LOG_BASE_DIR, "main-experiment-gabor", "fg_calibration_log")
        bg_calib_dir = os.path.join(CALIB_LOG_BASE_DIR, "main-experiment-gabor", "bg_calibration_log")
    elif args.target == "pre":
        OUTPUT_DIR = OUTPUT_DIR_PRE
        fg_calib_dir = os.path.join(CALIB_LOG_BASE_DIR, "pre-experiment-gabor", "fg_calibration_log")
        bg_calib_dir = os.path.join(CALIB_LOG_BASE_DIR, "pre-experiment-gabor", "bg_calibration_log")
    else:
        raise ValueError("不正なtarget指定です。mainかpreを指定してください。")

    print(f"OUTPUT_DIR = {OUTPUT_DIR}")

    # --- [新規] 輝度-ピクセル変換の準備 ---
    # 最新のキャリブレーション結果を読み込む (Foreground)
    fg_sorted_lums, fg_sorted_pixels = get_calibrated_map_arrays(fg_calib_dir, "Foreground")
    
    # 最新のキャリブレーション結果を読み込む (Background)
    bg_sorted_lums, bg_sorted_pixels = get_calibrated_map_arrays(bg_calib_dir, "Background")
    
    if fg_sorted_lums is None or bg_sorted_lums is None:
        print("エラー: 有効なキャリブレーションデータがないため、処理を中断します。")
        exit(1)

    # --- ガボールパッチの生成 (前景距離に基づく) ---
    print("--- Generating Gabor Patches (Foreground) ---")
    for fg_distance in FG_DISTANCES_CM:
        # A. 環境に応じたPPDの計算
        fg_ppd = calculate_ppd(fg_distance, SCREEN_WIDTH_CM, SCREEN_RES_X_PX)
        
        # B. 生成すべき画像サイズの計算 (px)
        req_w, req_h = get_stimulus_pixel_size(STIM_WIDTH_DEG, STIM_HEIGHT_DEG, fg_ppd)
        
        print(f"  Distance: {fg_distance}cm (PPD: {fg_ppd:.2f}, Size: {req_w}x{req_h}px)")
        
        # 保存先フォルダの準備
        gabor_dir = os.path.join(OUTPUT_DIR, "fg_gabor", f"{fg_distance}cm")
        os.makedirs(gabor_dir, exist_ok=True)

        # C. 前景パラメータでループ (8通り)
        for fg_cpd in FG_SPATIAL_FREQS_CPD:
            for fg_mean_lum in FG_MEAN_LUMINANCES_CDM2:
                for fg_contrast in FG_CONTRASTS:
                    # --- この前景条件に対するガボール画像を1枚生成 ---
                    gabor_mod_v = create_gabor(req_w, req_h, fg_ppd, fg_cpd, GABOR_SIGMA_DEG, orientation_deg=0)
                    lum_map_v = fg_mean_lum * (1 + fg_contrast * gabor_mod_v)
                    pixel_map_v = luminance_to_pixel(lum_map_v, fg_sorted_lums, fg_sorted_pixels)

                    # D. 背景パラメータでループし (8通り)、同じ画像を異なる名前で保存
                    for bg_cpd in BG_SPATIAL_FREQS_CPD:
                        for bg_mean_lum in BG_MEAN_LUMINANCES_CDM2:
                            for bg_contrast in BG_CONTRASTS:
                                print(f"    Generating Gabor for FG(f={fg_cpd},L={fg_mean_lum},C={fg_contrast}) BG(f={bg_cpd},L={bg_mean_lum},C={bg_contrast})")

                                # ファイル名の設定と保存
                                gabor_filename = os.path.join(gabor_dir, f"FG_{fg_cpd}_{fg_mean_lum}_{fg_contrast}_BG_{bg_cpd}_{bg_mean_lum}_{bg_contrast}.png")
                                plt.imsave(gabor_filename, pixel_map_v, cmap='gray', vmin=0, vmax=255)

    print("\n--- Generating Band-limited Noise (Background) ---")
    # --- 帯域制限ノイズの生成 (背景距離に基づく) ---
    for bg_distance in BG_DISTANCES_CM:
        # A. 環境に応じたPPDの計算
        bg_ppd = calculate_ppd(bg_distance, SCREEN_WIDTH_CM, SCREEN_RES_X_PX)
        # B. 生成すべき画像サイズの計算 (px) - 横長にする
        req_w, req_h = get_stimulus_pixel_size(STIM_WIDTH_DEG * 2, STIM_HEIGHT_DEG, bg_ppd)
        
        print(f"  Distance: {bg_distance}cm (PPD: {bg_ppd:.2f}, Size: {req_w}x{req_h}px)")
        
        # E. 保存先フォルダの準備
        noise_dir = os.path.join(OUTPUT_DIR, "bg_noise", f"{bg_distance}cm")
        os.makedirs(noise_dir, exist_ok=True)

        # C. 前景と背景の全パラメータでループし、64通りの画像を生成
        for fg_cpd in FG_SPATIAL_FREQS_CPD:
            for fg_mean_lum in FG_MEAN_LUMINANCES_CDM2:
                for fg_contrast in FG_CONTRASTS:
                    for bg_cpd in BG_SPATIAL_FREQS_CPD:
                        for bg_mean_lum in BG_MEAN_LUMINANCES_CDM2:
                            for bg_contrast in BG_CONTRASTS:
                                # このループの内部で毎回ノイズを生成し、全条件で異なるパターンにする
                                stim_noise = create_band_limited_noise(req_w, req_h, bg_ppd, f_center_cpd=bg_cpd)
                                print(f"    Generating for FG(f={fg_cpd},L={fg_mean_lum},C={fg_contrast}) BG(f={bg_cpd},L={bg_mean_lum},C={bg_contrast})")

                                # D. 輝度マップの計算とピクセル値への変換
                                lum_map_noise = bg_mean_lum * (1 + bg_contrast * stim_noise)
                                pixel_map_noise = luminance_to_pixel(lum_map_noise, bg_sorted_lums, bg_sorted_pixels)

                                # F. ファイル名の設定と保存
                                noise_filename = os.path.join(noise_dir, f"FG_{fg_cpd}_{fg_mean_lum}_{fg_contrast}_BG_{bg_cpd}_{bg_mean_lum}_{bg_contrast}.png")
                                plt.imsave(noise_filename, pixel_map_noise, cmap='gray', vmin=0, vmax=255)

    print("\n--- 全ての刺激画像の生成が完了しました ---")