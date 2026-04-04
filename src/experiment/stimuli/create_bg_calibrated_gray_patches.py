# py .\src\experiment\stimuli\create_bg_calibrated_gray_patches.py
import numpy as np
import cv2
import os
import csv
import datetime
import argparse
 

def generate_calibrated_gray_patches(cd_m2_levels, max_cd_m2, min_cd_m2, offset=0.0, gamma=2.2, size=(400, 300), create_images=True, base_output_dir=None):
    """
    指定された物理輝度(cd/m^2)に対応するグレーパッチ画像を生成し、対応するピクセル値のリストを返す。
    ディスプレイの最大輝度(白)と最小輝度(黒)を考慮して計算する。
 
    :param cd_m2_levels: 生成したい輝度値のリスト (例: [150, 135, 120, ...])
    :param max_cd_m2: デジタル値255に対応する最大輝度 (例: 150)
    :param min_cd_m2: デジタル値0に対応する最小輝度 (黒浮き, 例: 0.5)
    :param offset: 実測値とのずれを補正するためのオフセット値 (例: 5.0)
    :param gamma: ディスプレイのガンマ値 (通常は2.2)
    :param size: 生成する画像のサイズ (幅, 高さ)
    :param create_images: Trueの場合、画像をファイルに保存する
    :return: 計算されたピクセル値(0-255)のリスト
    """
    WIDTH, HEIGHT = size
    INV_GAMMA = 1.0 / gamma
 
    # 保存用ディレクトリの作成
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lab_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    default_output_dir = os.path.join(lab_root, "data", "processed", "images", "pre-experiment-gabor", "bg_calibrated_gray_patches")
    output_dir = base_output_dir if base_output_dir is not None else default_output_dir
    if create_images:
        os.makedirs(output_dir, exist_ok=True)
        offset_str = f"{offset:.2f}" if np.isscalar(offset) else "variable"
        print(f"--- [Background] 物理輝度準拠のグレーパッチ (max={max_cd_m2}, min={min_cd_m2}, offset={offset_str} cd/m^2, gamma={gamma}) を生成中 ---")
 
    pixel_values = []
    
    # offsetが単一の数値ならリストに変換、リストならそのまま使用
    if np.isscalar(offset):
        offsets = [float(offset)] * len(cd_m2_levels)
    else:
        offsets = offset

    for i, y_cd_m2 in enumerate(cd_m2_levels):
        # 1. 輝度をディスプレイのダイナミックレンジに合わせて正規化 (0.0 - 1.0)
        # (目標輝度 + オフセット - 黒輝度) / (白輝度 - 黒輝度)
        target_cd_m2 = y_cd_m2 + offsets[i]
        if (max_cd_m2 - min_cd_m2) <= 0:
            y_normalized = 0
        else:
            y_normalized = (target_cd_m2 - min_cd_m2) / (max_cd_m2 - min_cd_m2)
        
        # 範囲外の値をクリップ
        y_linear = np.clip(y_normalized, 0.0, 1.0)
 
        # 2. ガンマ補正 (エンコーディング) を適用して sRGB 値 (0.0 - 1.0) を計算
        # L_display = (V_digital / 255)^gamma  =>  V_digital = 255 * (L_display)^(1/gamma)
        srgb_value = np.power(y_linear, INV_GAMMA)
 
        # 3. 8ビット整数 (0-255) にスケール変換
        gray_value = int(round(srgb_value * 255))
 
        pixel_values.append(gray_value)
 
        if create_images:
            # 4. 画像データで塗りつぶしと保存
            # OpenCVはBGR順だが、グレーなので (gray, gray, gray) でOK
            final_image = np.full((HEIGHT, WIDTH, 3), gray_value, dtype=np.uint8)
 
            # ファイル名に輝度値を含める
            filename = os.path.join(output_dir, f'gray_{y_cd_m2}cdm2.png')
            cv2.imwrite(filename, final_image)

            B, G, R = final_image[0, 0] # OpenCVのimread/imwriteはBGR順
            print(f"Y={y_cd_m2} cd/m^2 -> RGB({R},{G},{B}) -> '{filename}' を生成しました。")
 
    if create_images:
        print("-" * 60)
        print(f"画像を '{output_dir}' フォルダに保存しました。")
    
    return pixel_values


def get_measured_values_from_user(target_levels):
    """
    ユーザーに測定値の入力を促し、数値のリストとして返す。
    """
    print("" + "="*60)
    print("輝度測定と入力 (Background Display)")
    print("="*60)
    print("生成された各グレーパッチを背景ディスプレイに表示し、輝度計で実際の輝度(cd/m^2)を測定してください。")
    print("以下の目標輝度に対応する測定値を、順番にスペースで区切って入力してください。")
    print(f"目標輝度リスト: {target_levels}")
    
    while True:
        user_input = input("測定値を入力してください > ")
        try:
            measured_values = [float(v) for v in user_input.split()]
            if len(measured_values) == len(target_levels):
                return measured_values
            else:
                print(f"エラー: 値の数が合いません。{len(target_levels)}個の数値を入力してください。(入力された数: {len(measured_values)})")
        except ValueError:
            print("エラー: 数値として解釈できない入力が含まれています。スペース区切りの数値のみを入力してください。")
 
# --- 実行部分 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BG calibrated gray patch generator")
    parser.add_argument("--target", choices=["main", "pre"], required=True, help="保存先を選択します: main または pre (必須)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    lab_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    if args.target == "main":
        csv_log_dir = os.path.join(lab_root, "results", "tables", "main-experiment-gabor", "bg_calibration_log")
        image_dir = os.path.join(lab_root, "data", "processed", "images", "main-experiment-gabor", "bg_calibrated_gray_patches")
    else:
        csv_log_dir = os.path.join(lab_root, "results", "tables", "pre-experiment-gabor", "bg_calibration_log")
        image_dir = os.path.join(lab_root, "data", "processed", "images", "pre-experiment-gabor", "bg_calibrated_gray_patches")

    start_time = datetime.datetime.now()
    # --- 基本設定 (背景ディスプレイ用に必要に応じて調整してください) ---
    MAX_LUMINANCE = 30.0
    MIN_LUMINANCE = 0.0
    TARGET_LUMINANCE_LEVELS = [30, 25, 20, 15, 10, 5, 0]
    IMAGE_SIZE = (400, 300)
    GAMMA = 2.2
    
    # --- 初回測定 ---
    # 最初に基準となるパッチ（オフセット=0の状態）を測定してもらう
    print("" + "="*60)
    print("初回測定: [Background] 基準となるグレーパッチを測定してください。")
    print("="*60)
    print("まず、オフセット 0 で生成された基準のグレーパッチ（または、それに相当する画像）を輝度計で測定してください。")
    
    measured_values = get_measured_values_from_user(TARGET_LUMINANCE_LEVELS)
    
    # 誤差と最初の補正オフセットを計算
    targets = np.array(TARGET_LUMINANCE_LEVELS)
    errors = np.array(measured_values) - targets
    
    # 全ての誤差が2以下か判定するため、最大絶対誤差を使用
    max_abs_error = np.max(np.abs(errors))
    
    # 最初の補正値は、観測された平均誤差を打ち消す値
    # 各輝度レベルごとに個別の補正値を計算する (初期オフセットは0なので、0 - error)
    correction_offsets = -errors

    print("--- 初回計算結果 ---")
    print(f"最大誤差 (絶対値): {max_abs_error:.4f} cd/m^2")

    # --- ループ初期化 ---
    iteration = 0
    max_iterations = 10 # 無限ループ防止

    while max_abs_error > 1.0:
        iteration += 1
        if iteration > max_iterations:
            print(f"反復回数が{max_iterations}回を超えたため、処理を中断します。")
            break

        print(f"{'='*20} [Background] キャリブレーション: Iteration {iteration} {'='*20}")
        
        # 1. 現在の補正オフセットでグレーパッチを生成
        generate_calibrated_gray_patches(
            cd_m2_levels=TARGET_LUMINANCE_LEVELS, max_cd_m2=MAX_LUMINANCE, min_cd_m2=MIN_LUMINANCE,
            offset=correction_offsets, gamma=GAMMA, size=IMAGE_SIZE, create_images=True
        )
        
        # 2. ユーザーからの実測値入力を受け取る
        measured_values = get_measured_values_from_user(TARGET_LUMINANCE_LEVELS)
        
        # 3. 誤差と次の補正オフセットを計算
        errors = np.array(measured_values) - targets
        max_abs_error = np.max(np.abs(errors))
        
        # 次の補正値は、現在の補正値から観測された誤差を引くことで更新する
        correction_offsets = correction_offsets - errors

        print("--- 計算結果 ---")
        print(f"最大誤差 (絶対値): {max_abs_error:.4f} cd/m^2")

    # --- ループ終了後 ---
    print(f"{'='*20} [Background] キャリブレーション完了 {'='*20}")
    if iteration <= max_iterations:
        print(f"最大誤差 {max_abs_error:.4f} cd/m^2 となり、閾値(2.0)内に収束しました。")
    
    print("最終的な補正オフセットを用いて、目標輝度とピクセル値の対応表を作成します。")
    
    final_pixel_values = generate_calibrated_gray_patches(
        cd_m2_levels=TARGET_LUMINANCE_LEVELS, max_cd_m2=MAX_LUMINANCE, min_cd_m2=MIN_LUMINANCE,
        offset=correction_offsets, gamma=GAMMA, size=IMAGE_SIZE, create_images=False,
        base_output_dir=image_dir
    )
    
    print("--- [Background] 【最終結果】目標輝度とピクセル値の対応表 ---")
    print("-" * 50)
    print(f"{'Target Luminance (cd/m^2)':<30} | {'Pixel Value (0-255)'}")
    print("-" * 50)
    for target, pixel in zip(TARGET_LUMINANCE_LEVELS, final_pixel_values):
        print(f"{target:<30.1f} | {pixel}")
    print("-" * 50)

    # --- 結果をCSVに保存 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lab_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    os.makedirs(csv_log_dir, exist_ok=True)
    
    log_filename = os.path.join(csv_log_dir, start_time.strftime("%Y%m%d_%H%M%S") + ".csv")
    now_str = start_time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        with open(log_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # ファイルが新規作成の場合はヘッダーを書き込む
            header = ["Date", "Target_Luminance(cd/m2)", "Pixel_Value", "Final_Offset", "Gamma", "Max_Lum", "Min_Lum"]
            writer.writerow(header)
            
            # 各輝度レベルごとのデータを書き込む
            for target, pixel, off_val in zip(TARGET_LUMINANCE_LEVELS, final_pixel_values, correction_offsets):
                writer.writerow([now_str, target, pixel, off_val, GAMMA, MAX_LUMINANCE, MIN_LUMINANCE])
        
        print(f"キャリブレーション結果を '{log_filename}' に保存しました。")
    except Exception as e:
        print(f"エラー: CSVファイルへの保存に失敗しました。 ({e})")