import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_glass_simulation(image_path, M_rgb, gamma=2.2):
    """
    指定された画像に対して、ガラスの分光特性行列を適用します。
    BGR画像に対して直接計算できるよう、行列を内部で変換します。
    """
    
    # 1. 画像の読み込み (OpenCVはデフォルトでBGRとして読み込みます)
    # 画像がない場合のために、デモ用のテスト画像を生成します
    if image_path is None:
        print("画像パスが指定されていないため、テスト用画像を生成します...")
        img_bgr = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(img_bgr, (50, 50), (150, 150), (0, 0, 255), -1)   # Red box
        cv2.rectangle(img_bgr, (150, 50), (250, 150), (0, 255, 0), -1)   # Green box
        cv2.rectangle(img_bgr, (100, 150), (200, 250), (255, 0, 0), -1)  # Blue box
        cv2.putText(img_bgr, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"画像が見つかりません: {image_path}")

    # 2. 行列の BGR 補正 (ここがポイントです)
    # 入力されたMは RGB順 (Row0=R, Row1=G, Row2=B) ですが、
    # 画像は BGR順 (Ch0=B, Ch1=G, Ch2=R) です。
    # したがって、行列の「行」と「列」を両方反転させることで BGR用行列 になります。
    M_bgr = np.flip(np.flip(M_rgb, 0), 1)
    
    print("--- RGB用 行列 ---")
    print(M_rgb)
    print("\n--- BGR変換用 行列 (自動調整済み) ---")
    print(M_bgr)

    # 3. 線形変換の実行 (cv2.transform)
    # 各ピクセル p_in に対して、 p_out = M_bgr * p_in を計算します
    # 画像は通常ガンマ補正がかかっているため、物理演算を行うためにリニア空間に戻します
    img_float = img_bgr.astype(np.float32) / 255.0
    img_linear = np.power(img_float, gamma)

    # 線形空間で行列演算: p_out_linear = M_bgr * p_in_linear
    img_linear_transformed = cv2.transform(img_linear, M_bgr)

    # 負の値は物理的にあり得ないので0クリップしてから、ガンマ補正を再適用
    img_linear_transformed = np.maximum(img_linear_transformed, 0)
    img_transformed_gamma = np.power(img_linear_transformed, 1.0 / gamma)

    # 4. クリップ処理 (重要)
    # 計算結果が小数になったり、255を超えたり負になったりするため、
    # 0-255の範囲に収めて整数型(uint8)に戻します。
    img_transformed = np.clip(img_transformed_gamma * 255, 0, 255).astype(np.uint8)

    return img_bgr, img_transformed

# --- メイン処理 ---

# 1. 画像から読み取った行列 M (RGB定義)
# Row 0: R output coefficients
# Row 1: G output coefficients
# Row 2: B output coefficients
matrix_data = np.array([
   [ 0.1854, -0.0033, 0.0243],
   [-0.0037,  0.2241, 0.0056],
   [-0.0052, -0.0187, 0.2764]
])

# 2. 実行 (お手持ちの画像ファイル名があれば書き換えてください)
# ファイルパスを None にすると、R/G/Bの箱を描いたデモ画像で動きます
original, transformed = apply_glass_simulation(r"C:\Users\HamaKazu\Desktop\GradSchool\lab\McGill Calibrated Color Image Database\Textures\Textures\pippin0224.tif", matrix_data) 
# 例: apply_glass_simulation("my_photo.jpg", matrix_data)

# 画像の保存
output_filename = "simulated_glass_output.png"
cv2.imwrite(output_filename, transformed)
print(f"補正後の画像を保存しました: {output_filename}")

# 3. 結果の表示 (matplotlibはRGBで表示するため変換して表示)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title("Before Glass (Original)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))
plt.title("After Glass (Simulated)")
plt.axis('off')

plt.tight_layout()
plt.show()