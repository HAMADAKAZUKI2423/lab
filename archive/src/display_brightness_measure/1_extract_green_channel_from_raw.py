import cv2
import numpy as np

## raw画像からGチャンネルを抽出して輝度マップとして保存するスクリプト
# 1. TIFFファイルを読み込む
input_path = r"C:\Users\HamaKazu\Desktop\GradSchool\lab\experiment\DisplayBrightness\pattern\BG_brightness_pattern\Bnless_bg.tiff"
output_path = r"C:\Users\HamaKazu\Desktop\GradSchool\lab\experiment\DisplayBrightness\pattern\BG_brightness_pattern\Bnless_bg_green_channel.png"
# cv2.IMREAD_UNCHANGED を指定することで、勝手に変換されるのを防ぎます
raw_img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

if raw_img is None:
    print(f"エラー: ファイルが読み込めませんでした -> {input_path}")
else:
    # 2. カラー化（デモザイク処理）
    # Bayer BG8 の場合、OpenCVでは COLOR_BayerBG2RGB を指定します
    color_img = cv2.cvtColor(raw_img, cv2.COLOR_BayerBG2RGB)

    # 3. Gチャンネルのみを抽出 (輝度として扱う)
    # RGBの順なので、Gは2番目 (インデックス 1)
    green_channel = color_img[:, :, 1]

    # 4. 結果の表示または保存
    cv2.imshow('Green Channel as Luminance', green_channel)
    cv2.imwrite(output_path, green_channel)
    print(f"Gチャンネル画像を保存しました: {output_path}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()