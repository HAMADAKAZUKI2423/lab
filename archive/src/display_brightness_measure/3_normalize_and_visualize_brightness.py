import cv2
import numpy as np
import matplotlib.pyplot as plt
## ディスプレイを撮影した画像から、正規化された物理輝度マップを計算・表示するクラス
# 1. 画像を読み込む (16bit TIFF対応)
# 2. 輝度情報を抽出 (カラーならグレースケールに変換)
# 3. 正規化 (Min-Max Normalization)

def process_tiff_normalization(input_path, output_path):
    # 1. 画像の読み込み (cv2.IMREAD_UNCHANGEDで元のビット深度を保持)
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"エラー: ファイルが見つかりません -> {input_path}")
        return

    print(f"元の画像データ型: {img.dtype}")
    print(f"元の画像形状: {img.shape}")

    # 2. 輝度情報の抽出
    # カラー画像(3チャンネル)の場合、グレースケール変換して輝度を取り出す
    if len(img.shape) == 3:
        print("カラー画像を検出しました。輝度(グレースケール)に変換します。")
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        # すでに1チャンネル(グレースケール)の場合
        gray_img = img

    # データ型をfloatに変換して計算精度を確保
    img_float = gray_img.astype(np.float32)

    # 3. 正規化 (Min-Max Normalization)
    min_val = np.min(img_float)
    max_val = np.max(img_float)

    # --- 正規化前の統計量 ---
    mean_raw = np.mean(img_float)
    std_raw = np.std(img_float)
    median_raw = np.median(img_float)

    print(f"輝度最小値 (Raw): {min_val}")
    print(f"輝度最大値 (Raw): {max_val}")
    print(f"輝度平均値 (Raw): {mean_raw:.5f}")
    print(f"輝度中央値 (Raw): {median_raw:.5f}")
    print(f"輝度標準偏差 (Raw): {std_raw:.5f}")

    if max_val - min_val == 0:
        print("警告: 画像の最大値と最小値が同じです。すべて0として扱います。")
        img_norm = np.zeros_like(img_float)
    else:
        # 正規化: 0.0 ～ 1.0
        img_norm = (img_float - min_val) / (max_val - min_val)

    # --- [追加機能] 統計量の計算 (正規化後のデータ) ---
    mean_val = np.mean(img_norm)       # 平均
    std_val = np.std(img_norm)         # 標準偏差 (Standard Deviation)
    median_val = np.median(img_norm)   # 中央値
    
    print("-" * 30)
    print(f"【正規化後 (0.0-1.0) の統計量】")
    print(f"  平均 (Mean)      : {mean_val:.5f}")
    print(f"  中央値 (Median)  : {median_val:.5f}")
    print(f"  標準偏差 (Std)   : {std_val:.5f}")
    print("-" * 30)
    # -----------------------------------------------

    # 4. マッピングと保存
    # 0.0-1.0 のデータを 0-255 (8bit整数) にマッピング
    img_8bit = (img_norm * 255).astype(np.uint8)

    # 画像として保存
    cv2.imwrite(output_path, img_8bit)
    print(f"正規化された画像を保存しました -> {output_path}")

    # --- 可視化 (ヒートマップ表示) ---
    plt.figure(figsize=(12, 6))

    # 左：正規化後のグレースケール画像
    plt.subplot(1, 2, 1)
    plt.imshow(img_norm, cmap='gray', vmin=0, vmax=1)
    # タイトルに平均と標準偏差を表示
    plt.title(f'Normalized Grayscale\nMean: {mean_val:.4f}, Median: {median_val:.4f}, Std: {std_val:.4f}')
    plt.axis('off')

    # 右：輝度の分布（ヒートマップ）
    plt.subplot(1, 2, 2)
    pos = plt.imshow(img_norm, cmap='jet', vmin=0, vmax=1)
    plt.title('Luminance Heatmap')
    plt.colorbar(pos, label='Normalized Intensity (0.0 - 1.0)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# --- 実行部分 ---
if __name__ == "__main__":
    # ここに処理したいTIFF画像のパスを指定してください
    # Windowsのパス形式に対応するため、r"..."を使用するか、スラッシュ(/)を使ってください
    input_filename = r"C:\Users\HamaKazu\Desktop\GradSchool\lab\experiment\DisplayBrightness\pattern\BG_brightness_pattern\Bnless_bg_green_channel_clipped_cutbt.png"
    output_filename = r"C:\Users\HamaKazu\Desktop\GradSchool\lab\experiment\DisplayBrightness\pattern\BG_brightness_pattern\Bnless_bg_green_channel_clipped_cutbt_norm.png"
    # 実行
    process_tiff_normalization(input_filename, output_filename)