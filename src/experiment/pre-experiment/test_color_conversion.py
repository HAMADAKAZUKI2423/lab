# test_color_conversion.py
# RGB <-> XYZ 変換機能のテストスクリプト

import sys
import os
import numpy as np

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from color_matching import rgb_to_xyz, xyz_to_rgb


def test_basic_colors():
    """基本色での変換テスト"""
    print("=" * 60)
    print("Basic Color Conversion Test")
    print("=" * 60)
    
    test_colors = {
        'Red': np.array([1.0, 0.0, 0.0]),
        'Green': np.array([0.0, 1.0, 0.0]),
        'Blue': np.array([0.0, 0.0, 1.0]),
        'White': np.array([1.0, 1.0, 1.0]),
        'Black': np.array([0.0, 0.0, 0.0]),
        'Gray': np.array([0.5, 0.5, 0.5]),
    }
    
    for name, rgb in test_colors.items():
        xyz = rgb_to_xyz(rgb)
        rgb_back = xyz_to_rgb(xyz)
        error = np.linalg.norm(rgb - rgb_back)
        print(f"\n{name}:")
        print(f"  RGB: {rgb}")
        print(f"  XYZ: {xyz}")
        print(f"  RGB (converted back): {rgb_back}")
        print(f"  Error: {error:.6f}")


def test_custom_colors():
    """カスタム色での変換テスト"""
    print("\n" + "=" * 60)
    print("Custom Color Conversion Test")
    print("=" * 60)
    
    # FG条件のオレンジ
    fg_orange = np.array([200, 130, 50], dtype=np.float32) / 255.0
    print(f"\nFG Orange (RGB 0-1): {fg_orange}")
    xyz_fg = rgb_to_xyz(fg_orange)
    print(f"FG Orange (XYZ): {xyz_fg}")
    
    # XYZ要因を調整
    x_factor = 1.2
    z_factor = 0.9
    adjusted_xyz = np.array([
        xyz_fg[0] * x_factor,
        xyz_fg[1],  # Y固定
        xyz_fg[2] * z_factor
    ])
    print(f"Adjusted XYZ (X*{x_factor}, Y*1.0, Z*{z_factor}): {adjusted_xyz}")
    
    adjusted_rgb = xyz_to_rgb(adjusted_xyz)
    print(f"Adjusted RGB (0-1): {adjusted_rgb}")
    print(f"Adjusted RGB (0-255): {adjusted_rgb * 255}")
    
    # BG条件の青
    bg_blue = np.array([50, 180, 255], dtype=np.float32) / 255.0
    print(f"\nBG Blue (RGB 0-1): {bg_blue}")
    xyz_bg = rgb_to_xyz(bg_blue)
    print(f"BG Blue (XYZ): {xyz_bg}")


def test_round_trip():
    """往復変換テスト"""
    print("\n" + "=" * 60)
    print("Round-Trip Conversion Test")
    print("=" * 60)
    
    # ランダムな色を生成
    np.random.seed(42)
    for i in range(5):
        rgb_orig = np.random.rand(3)
        xyz = rgb_to_xyz(rgb_orig)
        rgb_final = xyz_to_rgb(xyz)
        error = np.linalg.norm(rgb_orig - rgb_final)
        print(f"\nTest {i+1}:")
        print(f"  Original RGB: {rgb_orig}")
        print(f"  Final RGB: {rgb_final}")
        print(f"  Error: {error:.8f}")


def test_factor_adjustment():
    """X/Zファクター調整のテスト"""
    print("\n" + "=" * 60)
    print("Factor Adjustment Test")
    print("=" * 60)
    
    ref_rgb = np.array([150, 100, 50], dtype=np.float32) / 255.0
    print(f"Reference RGB (0-1): {ref_rgb}")
    print(f"Reference RGB (0-255): {ref_rgb * 255}")
    
    ref_xyz = rgb_to_xyz(ref_rgb)
    print(f"Reference XYZ: {ref_xyz}")
    
    # 複数のファクター組み合わせをテスト
    factors = [
        (0.8, 1.2),
        (1.0, 1.0),
        (1.2, 0.8),
    ]
    
    for x_factor, z_factor in factors:
        adjusted_xyz = np.array([
            ref_xyz[0] * x_factor,
            ref_xyz[1],
            ref_xyz[2] * z_factor
        ])
        adjusted_rgb = xyz_to_rgb(adjusted_xyz)
        print(f"\nX_factor={x_factor}, Z_factor={z_factor}:")
        print(f"  Adjusted RGB (0-1): {adjusted_rgb}")
        print(f"  Adjusted RGB (0-255): {adjusted_rgb * 255}")
        
        # クリップされた値をチェック
        clipped = np.sum(adjusted_rgb != np.clip(adjusted_rgb, 0, 1))
        if clipped > 0:
            print(f"  Warning: {clipped} values clipped!")


if __name__ == "__main__":
    test_basic_colors()
    test_custom_colors()
    test_round_trip()
    test_factor_adjustment()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
