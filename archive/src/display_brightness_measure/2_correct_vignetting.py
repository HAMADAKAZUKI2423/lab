import cv2
import numpy as np

## ディスプレイの撮影画像から、周辺減光（ビネット）を補正するためのスクリプト
# 1. ターゲット画像とリファレンス画像を読み込む
# 2. リファレンス画像から補正マップ（ゲインマップ）を作成
# 3. ターゲット画像に補正マップを適用して、周辺減光を補正
def correct_vignetting(target_path, ref_path, output_path):
    # 1. 画像の読み込み (元のビット深度とチャンネルを保持)
    target = cv2.imread(target_path, cv2.IMREAD_UNCHANGED)
    reference = cv2.imread(ref_path, cv2.IMREAD_UNCHANGED)

    if target is None:
        print(f"エラー: ターゲット画像が読み込めませんでした: {target_path}")
        return None
    if reference is None:
        print(f"エラー: リファレンス画像が読み込めませんでした: {ref_path}")
        return None

    # --- データ型と最大値の決定 ---
    if target.dtype == np.uint16:
        max_dtype_val = 65535
        output_dtype = np.uint16
        print("16-bit RAWデータモードで処理します。")
    elif target.dtype == np.uint8:
        max_dtype_val = 255
        output_dtype = np.uint8
        print("8-bitモードで処理します。")
    else:
        print(f"未対応のデータ型です: {target.dtype}")
        return None

    # 計算用にfloatに変換
    target_float = target.astype(np.float32)
    
    # リファレンスがカラーならグレースケールに変換
    if len(reference.shape) == 3:
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    reference_float = reference.astype(np.float32)

    # ターゲットとリファレンスのサイズを合わせる
    if target_float.shape[:2] != reference_float.shape:
        print("サイズが異なるため、リファレンス画像をリサイズします。")
        reference_float = cv2.resize(reference_float, (target_float.shape[1], target_float.shape[0]))

    # 2. 補正マップ（ゲインマップ）の作成
    max_ref_val = np.max(reference_float)
    if max_ref_val == 0:
        print("エラー: リファレンス画像が真っ黒です。補正できません。")
        return None
    gain_map = reference_float / max_ref_val

    # 0除算を防ぐためのクリッピング
    gain_map = np.maximum(gain_map, 1e-6)

    # 3. 補正の適用
    if len(target_float.shape) == 2:
        print("シングルチャンネル画像（RAWデータ）として補正します。")
        corrected_float = target_float / gain_map
    else: # 3チャンネル以上の場合
        print("マルチチャンネル画像として補正します。")
        # NumPyのブロードキャスト機能を使って、全チャンネルにゲインマップを適用
        corrected_float = target_float / gain_map[..., np.newaxis]

    # 4. 後処理
    # 元のデータ型の最大値を超えた値をクリップし、元の型に戻す
    corrected = np.clip(corrected_float, 0, max_dtype_val).astype(output_dtype)
    print("補正が完了しました。")

    cv2.imwrite(output_path, corrected)
    print(f"補正後の画像を保存しました: {output_path}")
    return corrected

# 実行例
target_image_path = r"C:\Users\HamaKazu\Desktop\GradSchool\lab\experiment\DisplayBrightness\pattern\BG_brightness_pattern\bg.tiff"
reference_image_path = r"C:\Users\HamaKazu\Desktop\GradSchool\lab\experiment\DisplayBrightness\pattern\BG_brightness_pattern\bn.tiff"
output_image_path = r"C:\Users\HamaKazu\Desktop\GradSchool\lab\experiment\DisplayBrightness\pattern\BG_brightness_pattern\Bnless_bg.tiff"

result = correct_vignetting(target_image_path, reference_image_path, output_image_path)