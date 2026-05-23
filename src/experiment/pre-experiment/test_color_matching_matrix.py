# test_color_matching_matrix.py
# カラーマッチング結果を用いた画像変換のテストスクリプト

import os
import sys
import numpy as np
from PIL import Image, ImageDraw

# 必要なモジュールへのパスを追加
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from color_matching import rgb_to_xyz, xyz_to_rgb
import stimuli_utils

def create_color_gabor(size_px, ppd, cpd):
    """RGBそれぞれのチャンネルで異なるパラメータのガボールパッチを生成して合成する"""
    # RGBで別々のガボールを生成
    gabor_r = stimuli_utils.create_gabor_image(size_px, ppd, cpd, contrast=0.8, orientation=0, phase=0)
    gabor_g = stimuli_utils.create_gabor_image(size_px, ppd, cpd*1.2, contrast=0.8, orientation=45, phase=np.pi/4)
    gabor_b = stimuli_utils.create_gabor_image(size_px, ppd, cpd*0.8, contrast=0.8, orientation=90, phase=np.pi/2)
    
    # 合成して[0, 1]に正規化
    img = Image.merge("RGB", (gabor_r, gabor_g, gabor_b))
    return np.array(img).astype(np.float32) / 255.0

def calculate_matching_matrices(results):
    """
    カラーマッチング結果 (X_Factor, Z_Factor) から3x3の変換行列を計算する
    """
    # 基準となるRGB値
    rgb_r = np.array([1.0, 0.0, 0.0])
    rgb_g = np.array([0.0, 1.0, 0.0])
    rgb_b = np.array([0.0, 0.0, 1.0])
    
    # Reference側のXYZ値 (Yは各色ごとの元の値)
    xyz_ref_r = rgb_to_xyz(rgb_r)
    xyz_ref_g = rgb_to_xyz(rgb_g)
    xyz_ref_b = rgb_to_xyz(rgb_b)
    
    # 列ベクトルとしてまとめる
    XYZ_ref = np.column_stack((xyz_ref_r, xyz_ref_g, xyz_ref_b))
    
    # Test側のXYZ値 (XとZはファクターを掛け、Yは初期指定の0.3に固定)
    xyz_test_r = np.array([xyz_ref_r[0] * results['R'][0], 0.3, xyz_ref_r[2] * results['R'][1]])
    xyz_test_g = np.array([xyz_ref_g[0] * results['G'][0], 0.3, xyz_ref_g[2] * results['G'][1]])
    xyz_test_b = np.array([xyz_ref_b[0] * results['B'][0], 0.3, xyz_ref_b[2] * results['B'][1]])
    
    XYZ_test = np.column_stack((xyz_test_r, xyz_test_g, xyz_test_b))
    
    # 変換行列の計算
    # XYZ_test = M_ref_to_test @ XYZ_ref  ==>  M_ref_to_test = XYZ_test @ inv(XYZ_ref)
    # XYZ_ref = M_test_to_ref @ XYZ_test  ==>  M_test_to_ref = XYZ_ref @ inv(XYZ_test)
    M_test_to_ref = XYZ_ref @ np.linalg.inv(XYZ_test)
    M_ref_to_test = XYZ_test @ np.linalg.inv(XYZ_ref)
    
    return M_test_to_ref, M_ref_to_test, XYZ_ref, XYZ_test

def apply_matrix_to_image(img_rgb, matrix):
    """RGB画像にXYZ空間での変換行列を適用する"""
    h, w, c = img_rgb.shape
    # RGBからXYZへ変換
    img_xyz = rgb_to_xyz(img_rgb)
    
    # 行列演算のため、(H*W, 3)の形に平坦化
    img_xyz_flat = img_xyz.reshape(-1, 3)
    
    # 行列適用 (v^T @ M^T の形をとる)
    img_xyz_converted_flat = img_xyz_flat @ matrix.T
    
    # 元の画像サイズに戻し、XYZからRGBへ変換
    img_xyz_converted = img_xyz_converted_flat.reshape(h, w, 3)
    img_rgb_converted = xyz_to_rgb(img_xyz_converted)
    
    return img_rgb_converted

def add_label(img_pil, text):
    """画像上部にテキストラベルを追加する簡単なユーティリティ"""
    w, h = img_pil.size
    new_img = Image.new("RGB", (w, h + 30), "black")
    new_img.paste(img_pil, (0, 30))
    draw = ImageDraw.Draw(new_img)
    draw.text((5, 5), text, fill="white")
    return new_img

def main():
    # 仮のカラーマッチング結果 (X_Factor, Z_Factor)
    # 実運用時はCSVファイルなどから読み込むように改変できます
    mock_results = {
        'R': (1.2, 0.8),
        'G': (0.9, 1.1),
        'B': (1.1, 1.3)
    }
    
    M_test_to_ref, M_ref_to_test, _, _ = calculate_matching_matrices(mock_results)
    
    print("=== Matrix (Test -> Ref) ===")
    print(M_test_to_ref)
    
    # ガボールパッチを生成
    print("\nGenerating color gabor patch...")
    img_rgb = create_color_gabor(size_px=300, ppd=40, cpd=2)
    
    # 変換行列を適用
    print("Applying conversion matrix...")
    img_test_to_ref = apply_matrix_to_image(img_rgb, M_test_to_ref)
    
    # PIL画像に変換してクリップ
    img_orig_pil = Image.fromarray(np.uint8(np.clip(img_rgb * 255, 0, 255)))
    img_test2ref_pil = Image.fromarray(np.uint8(np.clip(img_test_to_ref * 255, 0, 255)))
    
    # ラベルを追加して結合
    img_orig_pil = add_label(img_orig_pil, "Original Image")
    img_test2ref_pil = add_label(img_test2ref_pil, "Converted (Test -> Ref)")
    
    w, h = img_orig_pil.size
    combined = Image.new('RGB', (w * 2, h))
    combined.paste(img_orig_pil, (0, 0))
    combined.paste(img_test2ref_pil, (w, 0))
    
    # 画像の表示
    combined.show()
    # combined.save("color_matrix_conversion_result.png")

if __name__ == "__main__":
    main()