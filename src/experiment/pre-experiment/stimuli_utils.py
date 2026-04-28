"""
ステートレスなユーティリティ関数モジュール
画像生成、マーカー描画、視角計算など、状態を持たない関数を集約
"""
import math
import numpy as np
from PIL import Image, ImageFilter


# ==========================================
# 定数（defocus_matching.pyから参照可能にしたい場合は調整）
# ==========================================
SQUARE_SIZE = 30        # 四隅のマーカーの辺の長さ (px)
CROSS_SIZE = 30        # 中央の十字マーカーのサイズ (px)
MARKER_LINE_WIDTH = 5  # マーカーの線の太さ
PIXELS_PER_CM = 1/0.02331  # モニタのPPC


# ==========================================
# 画像生成関数
# ==========================================

def create_gabor_image(size_px, ppd, cpd, contrast, orientation=0, phase=0, blur_sigma=0):
    """
    Gaborパターン画像を生成する
    
    Args:
        size_px: 画像サイズ (ピクセル)
        ppd: ピクセル/度数 (pixels per degree)
        cpd: 空間周波数 (cycles per degree)
        contrast: コントラスト (0~1)
        orientation: 向き (度数, 0~180)
        phase: 位相 (ラジアン)
        blur_sigma: ガウシアンブラーの標準偏差 (0で無効)
    
    Returns:
        PIL.Image: L モード（グレースケール）の画像
    """
    x = np.linspace(-size_px/2, size_px/2, size_px) / ppd
    y = np.linspace(-size_px/2, size_px/2, size_px) / ppd
    X, Y = np.meshgrid(x, y)
    theta = np.deg2rad(orientation)
    X_rot = X * np.cos(theta) + Y * np.sin(theta)
    grating = np.sin(2 * np.pi * cpd * X_rot + phase)
    sigma_deg = 1.0
    envelope = np.exp(-(X**2 + Y**2) / (2 * sigma_deg**2))
    gabor = grating * envelope
    lum = 50.0 * (1.0 + contrast * gabor)
    pixel_map = np.clip(lum / 100.0 * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixel_map, mode='L')
    if blur_sigma > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
    return img


def create_checkerboard_image(width_px, height_px, ppd, cpd):
    """
    チェッカーボード（市松模様）画像を生成する
    
    Args:
        width_px: 画像幅 (ピクセル)
        height_px: 画像高さ (ピクセル)
        ppd: ピクセル/度数 (pixels per degree)
        cpd: 空間周波数 (cycles per degree)
    
    Returns:
        PIL.Image: L モード（グレースケール）の画像
    """
    square_size = max(1, int(ppd / (2 * cpd)))
    x = np.arange(width_px)
    y = np.arange(height_px)
    X, Y = np.meshgrid(x, y)
    checker = ((X // square_size) + (Y // square_size)) % 2
    checker = checker * 2 - 1
    lum = 50.0 * (1.0 + checker)
    pixel_map = np.clip(lum / 100.0 * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(pixel_map, mode='L')


def create_noise_base(width_px, height_px, ppd, f_center_cpd, bandwidth_octave=1.0):
    """
    フィルターされたノイズ基盤を生成する
    
    Args:
        width_px: 画像幅 (ピクセル)
        height_px: 画像高さ (ピクセル)
        ppd: ピクセル/度数 (pixels per degree)
        f_center_cpd: 中心周波数 (cycles per degree)
        bandwidth_octave: 周波数帯域幅 (オクターブ単位)
    
    Returns:
        np.ndarray: 正規化されたノイズ配列 (-1~1)
    """
    white_noise = np.random.normal(0, 1, (height_px, width_px))
    ft_noise = np.fft.fftshift(np.fft.fft2(white_noise))
    
    fx = np.fft.fftshift(np.fft.fftfreq(width_px, d=1/ppd))
    fy = np.fft.fftshift(np.fft.fftfreq(height_px, d=1/ppd))
    FX, FY = np.meshgrid(fx, fy)
    R = np.sqrt(FX**2 + FY**2)
    
    f_min = f_center_cpd / (2 ** (bandwidth_octave / 2))
    f_max = f_center_cpd * (2 ** (bandwidth_octave / 2))
    mask = (R >= f_min) & (R <= f_max)
    
    ft_filtered = ft_noise * mask
    noise_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(ft_filtered)))
    
    max_val = np.max(np.abs(noise_filtered))
    if max_val > 0:
        noise_filtered = noise_filtered / max_val
        
    return noise_filtered


def create_gabor_base(width_px, height_px, ppd, cpd, orientation=0, phase=0, sigma_deg=1.0):
    """
    Gaborの基盤パターン（-1~1）を生成する
    
    Args:
        width_px: 画像幅 (ピクセル)
        height_px: 画像高さ (ピクセル)
        ppd: ピクセル/度数 (pixels per degree)
        cpd: 空間周波数 (cycles per degree)
        orientation: 向き (度数, 0~180)
        phase: 位相 (ラジアン)
        sigma_deg: ガウシアンエンベロープの標準偏差 (度数)
    
    Returns:
        np.ndarray: 正規化されたGaborパターン (-1~1)
    """
    x = np.linspace(-width_px/2, width_px/2, width_px) / ppd
    y = np.linspace(-height_px/2, height_px/2, height_px) / ppd
    X, Y = np.meshgrid(x, y)
    theta = np.deg2rad(orientation)
    X_rot = X * np.cos(theta) + Y * np.sin(theta)
    grating = np.sin(2 * np.pi * cpd * X_rot + phase)
    envelope = np.exp(-(X**2 + Y**2) / (2 * sigma_deg**2))
    return grating * envelope


# ==========================================
# ユーティリティ関数
# ==========================================

def get_size_for_visual_angle(distance_cm, angle_deg, pixels_per_cm=PIXELS_PER_CM):
    """
    指定された視角と距離から、対応するピクセルサイズを計算する
    
    Args:
        distance_cm: 観視距離 (cm)
        angle_deg: 視角 (度数)
        pixels_per_cm: モニタのPPC (ピクセル/cm)
    
    Returns:
        int: 対応するピクセルサイズ (四捨五入)
    """
    if distance_cm <= 0:
        return 0
    # 物理サイズ[cm] = 2 * 距離[cm] * tan(視角[rad] / 2)
    angle_rad = math.radians(angle_deg)
    size_cm = 2 * distance_cm * math.tan(angle_rad / 2)
    # ピクセルサイズ = 物理サイズ[cm] * PPC
    return round(size_cm * pixels_per_cm)


# ==========================================
# 描画関数（マーカー）
# ==========================================

def draw_image_corner_brackets(canvas, size_w, size_h, offset_x=0, offset_y=0, color='white', flip_x=False, line_width=MARKER_LINE_WIDTH):
    """
    指定された画像表示領域の四隅に、鍵括弧状のマーカーを描画する
    
    Args:
        canvas: Tkinter Canvas ウィジェット
        size_w: マーカー領域の幅 (ピクセル)
        size_h: マーカー領域の高さ (ピクセル)
        offset_x: X方向のオフセット (ピクセル)
        offset_y: Y方向のオフセット (ピクセル)
        color: マーカーの色 (CSS色名または16進数)
        flip_x: X座標を反転するか (左右反転)
        line_width: 線の太さ (ピクセル)
    """
    s = SQUARE_SIZE
    
    # 画面の中心座標
    cx, cy = canvas.winfo_width() // 2, canvas.winfo_height() // 2
    # 画像表示領域の左上と右下の座標を計算 (オフセット適用)
    x0 = cx - size_w // 2 + offset_x
    y0 = cy - size_h // 2 + offset_y
    x1 = cx + size_w // 2 + offset_x
    y1 = cy + size_h // 2 + offset_y

    # X座標変換関数 (flip_xがTrueなら左右反転)
    def tx(x):
        return canvas.winfo_width() - x if flip_x else x

    # Top-left
    canvas.create_line(tx(x0), y0, tx(x0 + s), y0, fill=color, width=line_width, tags="calib")
    canvas.create_line(tx(x0), y0, tx(x0), y0 + s, fill=color, width=line_width, tags="calib")
    # Top-right
    canvas.create_line(tx(x1 - s), y0, tx(x1), y0, fill=color, width=line_width, tags="calib")
    canvas.create_line(tx(x1), y0, tx(x1), y0 + s, fill=color, width=line_width, tags="calib")
    # Bottom-left
    canvas.create_line(tx(x0), y1 - s, tx(x0), y1, fill=color, width=line_width, tags="calib")
    canvas.create_line(tx(x0), y1, tx(x0 + s), y1, fill=color, width=line_width, tags="calib")
    # Bottom-right
    canvas.create_line(tx(x1 - s), y1, tx(x1), y1, fill=color, width=line_width, tags="calib")
    canvas.create_line(tx(x1), y1 - s, tx(x1), y1, fill=color, width=line_width, tags="calib")


def draw_center_cross(canvas, offset_x=0, offset_y=0, color='white', gap=0):
    """
    画面中央に一点へ向かう4つの矢尻（棒なし）を描画する
    
    Args:
        canvas: Tkinter Canvas ウィジェット
        offset_x: X方向のオフセット (ピクセル)
        offset_y: Y方向のオフセット (ピクセル)
        color: 矢尻の色 (CSS色名または16進数)
        gap: 中央の隙間サイズ (ピクセル)
    """
    cx = canvas.winfo_width() // 2 + offset_x
    cy = canvas.winfo_height() // 2 + offset_y
    
    # 矢尻の形状パラメータ（CROSS_SIZEを基準にサイズを大きく設定）
    d2 = int(CROSS_SIZE * 1.2)  # 先端から底辺までの距離
    d1 = int(CROSS_SIZE * 0.9)  # 先端から凹みまでの距離
    d3 = int(CROSS_SIZE * 0.5)  # 幅の半分

    # 4方向の矢尻の頂点座標
    pts_left = [cx - gap, cy, cx - d2 - gap, cy - d3, cx - d1 - gap, cy, cx - d2 - gap, cy + d3]
    pts_right = [cx + gap, cy, cx + d2 + gap, cy - d3, cx + d1 + gap, cy, cx + d2 + gap, cy + d3]
    pts_up = [cx, cy - gap, cx - d3, cy - d2 - gap, cx, cy - d1 - gap, cx + d3, cy - d2 - gap]
    pts_down = [cx, cy + gap, cx - d3, cy + d2 + gap, cx, cy + d1 + gap, cx + d3, cy + d2 + gap]

    for pts in [pts_left, pts_right, pts_up, pts_down]:
        canvas.create_polygon(pts, fill=color, outline="black", width=2, tags="calib")
