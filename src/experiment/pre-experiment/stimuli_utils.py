"""
ステートレスなユーティリティ関数モジュール
画像生成、マーカー描画、視角計算など、状態を持たない関数を集約
"""
import math
import numpy as np
from PIL import Image, ImageFilter, ImageTk
import os
import csv
import torch


# ==========================================
# 定数（defocus_matching.pyから参照可能にしたい場合は調整）
# ==========================================
SQUARE_SIZE = 30        # 四隅のマーカーの辺の長さ (px)
CROSS_SIZE = 30        # 中央の十字マーカーのサイズ (px)
MARKER_LINE_WIDTH = 5  # マーカーの線の太さ
PIXELS_PER_CM = 1/0.02331  # モニタのPPC

# sRGB(D65) <-> XYZ 標準変換行列
M_RGB2XYZ = np.array([
    [0.41239080, 0.35758434, 0.18048079],
    [0.21263901, 0.71516868, 0.07219232],
    [0.01933082, 0.11919478, 0.95053215],
], dtype=np.float64)

M_XYZ2RGB = np.array([
    [ 3.24096994, -1.53738318, -0.49861076],
    [-0.96924364,  1.87596750,  0.04155506],
    [ 0.05563008, -0.20397696,  1.05697151],
], dtype=np.float64)



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


def create_cosine_windowed_disk(width_px, height_px, ppd, disk_diameter_deg=3.0, fade_width_deg=0.5):
    """
    一定コントラストの円形刺激（端はコサイン波で減衰）を生成する

    Args:
        width_px (int): 画像幅 (ピクセル)
        height_px (int): 画像高さ (ピクセル)
        ppd (float): ピクセル/度数 (pixels per degree)
        disk_diameter_deg (float): フェードアウト部分を含む円盤の総直径（度数）
        fade_width_deg (float): コサイン波で減衰するエッジ部分の幅（度数）

    Returns:
        np.ndarray: 刺激マスクを表す (height_px, width_px) 形状の 0.0 から 1.0 の値を持つ
                    2D numpy 配列
    """
    # 度数からピクセルへ変換
    total_radius_px = (disk_diameter_deg / 2.0) * ppd
    fade_width_px = fade_width_deg * ppd
    flat_radius_px = total_radius_px - fade_width_px

    if flat_radius_px < 0:
        # フェード幅が半径より大きい場合、円全体がフェードになる
        flat_radius_px = 0
        fade_width_px = total_radius_px

    # 座標グリッドを作成
    x = np.linspace(-width_px / 2, width_px / 2, width_px)
    y = np.linspace(-height_px / 2, height_px / 2, height_px)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    # マスクを作成
    mask = np.zeros((height_px, width_px))

    # 平坦な領域
    mask[R <= flat_radius_px] = 1.0

    # コサインフェード領域
    fade_zone = (R > flat_radius_px) & (R <= total_radius_px)
    if fade_width_px > 0:
        r_norm = (R[fade_zone] - flat_radius_px) / fade_width_px
        mask[fade_zone] = (np.cos(r_norm * np.pi) + 1.0) / 2.0

    return mask


def create_cosine_windowed_grating_base(width_px, height_px, ppd, cpd, orientation=0, phase=0, disk_diameter_deg=3.0, fade_width_deg=0.5):
    """
    コサイン窓を用いた円形グレーティングの基盤パターン（-1~1）を生成する
    
    Args:
        width_px: 画像幅 (ピクセル)
        height_px: 画像高さ (ピクセル)
        ppd: ピクセル/度数 (pixels per degree)
        cpd: 空間周波数 (cycles per degree)
        orientation: 向き (度数, 0~180)
        phase: 位相 (ラジアン)
        disk_diameter_deg: フェードアウト部分を含む円盤の総直径（度数）
        fade_width_deg: コサイン波で減衰するエッジ部分の幅（度数）
    
    Returns:
        np.ndarray: 正規化されたパターン (-1~1)
    """
    x = np.linspace(-width_px/2, width_px/2, width_px) / ppd
    y = np.linspace(-height_px/2, height_px/2, height_px) / ppd
    X, Y = np.meshgrid(x, y)
    theta = np.deg2rad(orientation)
    X_rot = X * np.cos(theta) + Y * np.sin(theta)
    grating = np.sin(2 * np.pi * cpd * X_rot + phase)
    
    envelope = create_cosine_windowed_disk(width_px, height_px, ppd, disk_diameter_deg, fade_width_deg)
    
    return grating * envelope

def srgb_to_linear(c):
    """
    sRGB EOTF (gamma correction)
    """
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.0)

def linear_to_srgb(c):
    """
    sRGB OETF (inverse gamma correction)
    """
    c = np.asarray(c, dtype=np.float64)
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1.0 / 2.0)) - 0.055)








# ==========================================
# ユーティリティ関数
# ==========================================

def get_size_for_visual_angle(distance_cm, angle_deg, pixels_per_cm=PIXELS_PER_CM, canvas=None):
    """
    指定された視角と距離から、対応するピクセルサイズを計算する

    Args:
        distance_cm: 観視距離 (cm)
        angle_deg: 視角 (度数)
        pixels_per_cm: モニタのPPC (ピクセル/cm)。`canvas`を指定した場合は無視されます。
        canvas: (optional) Tkinter Canvas/Widget。与えるとそのウィジェットのDPIからピクセル密度を取得します。

    Returns:
        int: 対応するピクセルサイズ (四捨五入)
    """
    if distance_cm <= 0:
        return 0

    # canvasが与えられたらそのウィジェットの DPI 情報から pixels_per_cm を計算する
    if canvas is not None:
        try:
            # winfo_fpixels('1i') は1インチあたりのピクセル数を返す
            ppi = float(canvas.winfo_fpixels('1i'))
            pixels_per_cm = ppi / 2.54
        except Exception:
            # 取得できなければ既定値を使う
            pixels_per_cm = pixels_per_cm

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


# ==========================================
# キャリブレーション・画像処理関数
# ==========================================

def load_calibration_data(log_dir):
    """
    DisplayBrightness キャリブレーション結果をCSVから読み込む
    
    Args:
        log_dir: キャリブレーションログディレクトリ
        
    Returns:
        tuple: (lums, pixels) - numpy配列
               lums: ルミナンス値の配列
               pixels: ピクセル値の配列
    """
    csv_files = []
    if not os.path.exists(log_dir):
        return None, None
    
    for file in os.listdir(log_dir):
        if file.endswith('.csv'):
            csv_files.append(os.path.join(log_dir, file))
    
    if not csv_files:
        return None, None
    
    lum_pixel_data = {}
    for filepath in csv_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # ヘッダ名が異なる場合に対応
                    t_lum = None
                    p_val = None
                    if "Target_Luminance(cd/m2)" in row and "Pixel_Value" in row:
                        t_lum = row["Target_Luminance(cd/m2)"]
                        p_val = row["Pixel_Value"]
                    elif "luminance" in row and "pixel_value" in row:
                        t_lum = row["luminance"]
                        p_val = row["pixel_value"]
                    
                    if t_lum is not None and p_val is not None:
                        try:
                            t_lum_f = float(t_lum)
                            p_val_i = int(p_val)
                            lum_pixel_data.setdefault(t_lum_f, []).append(p_val_i)
                        except ValueError:
                            pass
        except Exception as e:
            print(f"Warning: Could not read {filepath}: {e}")
            continue
    
    avg_map = []
    for t_lum, p_list in sorted(lum_pixel_data.items()):
        if p_list:
            avg_map.append((t_lum, int(np.round(np.mean(p_list)))))
            
    if avg_map:
        lums = np.array([x[0] for x in avg_map])
        pixels = np.array([x[1] for x in avg_map])
        return lums, pixels

    return None, None


def apply_torch_fft_blur(img_pil, D, pd_mm, pixels_per_deg):
    """
    Torch FFT を使用したデフォーカスブラー適用 (PIL Image用)
    
    Args:
        img_pil: PIL Image (RGB または L mode)
        D: Diopter (度数)
        pd_mm: 瞳孔径 (mm)
        pixels_per_deg: ピクセル/度数
        
    Returns:
        PIL Image: ブラー後の画像
    """
    if D <= 0 or pd_mm <= 0:
        return img_pil
        
    rad2deg = 180.0 / math.pi
    mm = 1e-3
    bd_deg = rad2deg * D * pd_mm * mm
    DEFOCUS_BLUR_SCALE_FACTOR = 0.55
    sigma = DEFOCUS_BLUR_SCALE_FACTOR * bd_deg / 2.0
    
    if sigma <= 0:
        return img_pil
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    img_np = np.array(img_pil).astype(np.float32)
    is_rgb = len(img_np.shape) == 3
    
    if is_rgb:
        img_np = np.transpose(img_np, (2, 0, 1))
        
    img_tensor = torch.from_numpy(img_np).to(device)
    h, w = img_tensor.shape[-2:]
    
    x_deg = torch.linspace(-w/2/pixels_per_deg, w/2/pixels_per_deg, w).to(device)
    y_deg = torch.linspace(-h/2/pixels_per_deg, h/2/pixels_per_deg, h).to(device)
    Y_deg, X_deg = torch.meshgrid(y_deg, x_deg, indexing='ij')
    
    psf = torch.exp(-((torch.sqrt(X_deg**2 + Y_deg**2)) ** 2) / (2 * sigma ** 2))
    psf = psf / torch.sum(psf)
    
    def FT2(tensor):
        tensor_shift = torch.fft.ifftshift(tensor, dim=(-2,-1))
        tensor_ft_shift = torch.fft.fft2(tensor_shift, norm='ortho')
        return torch.fft.fftshift(tensor_ft_shift, dim=(-2,-1))

    def iFT2(tensor):
        tensor_shift = torch.fft.ifftshift(tensor, dim=(-2,-1))
        tensor_ift_shift = torch.fft.ifft2(tensor_shift, norm='ortho')
        return torch.fft.fftshift(tensor_ift_shift, dim=(-2,-1))

    if is_rgb:
        psf = psf.unsqueeze(0)
        
    img_ft = FT2(img_tensor)
    psf_ft = FT2(psf)
    blur_tensor = torch.abs(iFT2(img_ft * psf_ft))
    
    if is_rgb:
        for c in range(3):
            blur_tensor[c] = blur_tensor[c] * torch.sum(img_tensor[c]) / (torch.sum(blur_tensor[c]) + 1e-8)
    else:
        blur_tensor = blur_tensor * torch.sum(img_tensor) / (torch.sum(blur_tensor) + 1e-8)
        
    blur_np = blur_tensor.cpu().numpy()
    
    if is_rgb:
        blur_np = np.transpose(blur_np, (1, 2, 0))
        
    blur_np = np.clip(blur_np, 0, 255).astype(np.uint8)
    return Image.fromarray(blur_np, mode=img_pil.mode)


def apply_torch_fft_blur_luminance(lum_np, D, pd_mm, pixels_per_deg):
    """
    Torch FFT を使用したデフォーカスブラー適用 (輝度配列用)
    
    Args:
        lum_np: 輝度配列 (numpy float32)
        D: Diopter (度数)
        pd_mm: 瞳孔径 (mm)
        pixels_per_deg: ピクセル/度数
        
    Returns:
        numpy array: ブラー後の輝度配列
    """
    if D <= 0 or pd_mm <= 0:
        return lum_np
        
    rad2deg = 180.0 / math.pi
    mm = 1e-3
    bd_deg = rad2deg * D * pd_mm * mm
    DEFOCUS_BLUR_SCALE_FACTOR = 0.55
    sigma = DEFOCUS_BLUR_SCALE_FACTOR * bd_deg / 2.0
    
    if sigma <= 0:
        return lum_np
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    img_tensor = torch.from_numpy(lum_np.astype(np.float32)).to(device)
    h, w = img_tensor.shape[-2:]
    
    x_deg = torch.linspace(-w/2/pixels_per_deg, w/2/pixels_per_deg, w).to(device)
    y_deg = torch.linspace(-h/2/pixels_per_deg, h/2/pixels_per_deg, h).to(device)
    Y_deg, X_deg = torch.meshgrid(y_deg, x_deg, indexing='ij')
    
    psf = torch.exp(-((torch.sqrt(X_deg**2 + Y_deg**2)) ** 2) / (2 * sigma ** 2))
    psf = psf / torch.sum(psf)
    
    def FT2(tensor):
        tensor_shift = torch.fft.ifftshift(tensor, dim=(-2,-1))
        tensor_ft_shift = torch.fft.fft2(tensor_shift, norm='ortho')
        return torch.fft.fftshift(tensor_ft_shift, dim=(-2,-1))

    def iFT2(tensor):
        tensor_shift = torch.fft.ifftshift(tensor, dim=(-2,-1))
        tensor_ift_shift = torch.fft.ifft2(tensor_shift, norm='ortho')
        return torch.fft.fftshift(tensor_ift_shift, dim=(-2,-1))

    img_ft = FT2(img_tensor)
    psf_ft = FT2(psf)
    blur_tensor = torch.abs(iFT2(img_ft * psf_ft))
    
    blur_tensor = blur_tensor * torch.sum(img_tensor) / (torch.sum(blur_tensor) + 1e-8)
        
    return blur_tensor.cpu().numpy()


def _apply_eotf(eotf, ch, vals, inverse=False):
    v, yn = eotf[ch]
    if inverse:            # 線形 -> 画素
        return np.interp(vals, yn, v)
    return np.interp(vals, v, yn)   # 画素 -> 線形

def lum_to_photo_dualplane_fg(lum_np, bg_lums, bg_pixels, C, eotf_bg, eotf_fg):
    """Dual Planeの前景用。実測EOTFでCを適用。"""
    # 1) 目標輝度 -> 背景画素(0-1) （背景校正）
    pix = np.clip(np.interp(lum_np, bg_lums, bg_pixels), 0, 255) / 255.0
    # 2) g_b で背景線形へ（グレースケールなのでR=G=B=pix）
    lin_bg = np.stack([_apply_eotf(eotf_bg, ch, pix) for ch in ("R","G","B")], axis=-1)
    # 3) 線形空間でC適用（背景線形 -> 前景線形）
    lin_fg = lin_bg @ C.T
    lin_fg = np.clip(lin_fg, 0.0, None)
    # 4) g_f_inv で前景画素へ
    out = np.empty_like(lin_fg)
    for i, ch in enumerate(("R","G","B")):
        out[..., i] = _apply_eotf(eotf_fg, ch, lin_fg[..., i], inverse=True)
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return ImageTk.PhotoImage(Image.fromarray(out, mode="RGB"))

def lum_to_photo_singleplane(lum_np, Y_grid, px_grid):
    """Single Plane用。拡張輝度LUTで変換。"""
    lum = np.asarray(lum_np, dtype=np.float64)
    out = np.empty(lum.shape + (3,), dtype=np.float64)
    for c in range(3):
        out[..., c] = np.interp(lum, Y_grid, px_grid[:, c])  # 端点クランプはinterp既定
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return ImageTk.PhotoImage(Image.fromarray(out, mode="RGB"))


def lum_to_photo_window2(lum_np, lums, pixels, color_matrix, eotf_bg=None, eotf_fg=None, Y_grid=None, px_grid=None, cond=""):
    """
    Window2用の刺激を生成するディスパッチャ。
    条件に応じて、C適用 or LUT適用を切り替える。
    フォールバックとして、EOTF/LUTがない場合は従来のsRGB近似でCを適用する。
    """
    # --- 新しい方式：EOTF/LUTが利用可能な場合 ---
    if cond in ["Dual plane", "Dual plane flat"] and eotf_bg and eotf_fg and color_matrix is not None:
        return lum_to_photo_dualplane_fg(lum_np, lums, pixels, color_matrix, eotf_bg, eotf_fg)
    
    if cond.startswith("Single plane") and Y_grid is not None and px_grid is not None:
        return lum_to_photo_singleplane(lum_np, Y_grid, px_grid)

    # --- フォールバック：従来のsRGB近似方式 ---
    if color_matrix is None: # Cがない場合は無補正グレー
        pix = np.interp(lum_np, lums, pixels)
        gray = np.clip(pix, 0, 255).astype(np.uint8)
        return ImageTk.PhotoImage(Image.fromarray(gray, mode='L'))

    # 輝度 -> 画素値(0-255)
    pix = np.interp(lum_np, lums, pixels)
    g = np.clip(pix, 0, 255).astype(np.float64) / 255.0  # sRGB[0,1], R=G=B

    lin = srgb_to_linear(g)                          # (H,W)
    lin_rgb = np.stack([lin, lin, lin], axis=-1)     # (H,W,3)
    lin_rgb_corr = lin_rgb @ color_matrix.T          # 線形RGB空間でCを適用
    srgb_corr = linear_to_srgb(lin_rgb_corr)         # 内部で[0,1]クリップ
    out = np.clip(srgb_corr * 255.0, 0, 255).astype(np.uint8)  # (H,W,3)
    return ImageTk.PhotoImage(Image.fromarray(out, mode='RGB'))

def calculate_defocus_blur_parameters(D, pd_mm, pixels_per_deg):
    """
    デフォーカスブラーのパラメータを計算
    
    Args:
        D: Diopter (度数)
        pd_mm: 瞳孔径 (mm)
        pixels_per_deg: ピクセル/度数
        
    Returns:
        dict: パラメータ辞書 {sigma, bd_deg}
    """
    if D <= 0 or pd_mm <= 0:
        return {"sigma": 0.0, "bd_deg": 0.0}
        
    rad2deg = 180.0 / math.pi
    mm = 1e-3
    bd_deg = rad2deg * D * pd_mm * mm
    DEFOCUS_BLUR_SCALE_FACTOR = 0.55
    sigma = DEFOCUS_BLUR_SCALE_FACTOR * bd_deg / 2.0
    
    return {"sigma": sigma, "bd_deg": bd_deg}


def create_block_trials(param_dict, num_repetitions, shuffle=True):
    """
    実験ブロック用の試行リストを生成
    
    Args:
        param_dict: パラメータ辞書
                   matching: {"ref_contrasts": [...], "orientations": [...]}
                   gabor: {"spatial_freqs": [...], ...}
                   image: {"images": [...]}
        num_repetitions: 反復回数
        shuffle: シャッフルするか (default True)
        
    Returns:
        list: 試行リスト [{"param1": val1, "param2": val2, ...}, ...]
    """
    trials = []
    
    # param_dict のすべてのキーと値の組み合わせを生成
    keys = list(param_dict.keys())
    values_lists = [param_dict[k] if isinstance(param_dict[k], list) else [param_dict[k]] for k in keys]
    
    # 全組み合わせを生成
    import itertools
    combinations = list(itertools.product(*values_lists))
    
    # 反復回数分複製
    for _ in range(num_repetitions):
        for combo in combinations:
            trial_dict = {keys[i]: combo[i] for i in range(len(keys))}
            trials.append(trial_dict)
    
    # シャッフル
    if shuffle:
        import random
        random.shuffle(trials)
    
    return trials


def lum_to_photo(lum_np, lums, pixels, color_matrix=None):
    """
    輝度配列をピクセル値に変換して ImageTk.PhotoImage を返す
    
    色補正行列が指定された場合、色補正を行いつつ輝度ドリフトを補正する
    （色補正行列による輝度変化を補正し、元の物理輝度を保持）

    Args:
        lum_np: 輝度配列 (numpy float)
        lums: 参照ルミナンス配列
        pixels: 参照ピクセル値配列
        color_matrix: (optional) 3x3 の色補正行列 (numpy array)
                      グレースケール(G,G,G) に適用される

    Returns:
        ImageTk.PhotoImage (グレースケール or RGB)
    """
    pix = np.interp(lum_np, lums, pixels)
    img = Image.fromarray(pix.astype(np.uint8), mode='L')
    
    if color_matrix is not None:
        img = apply_color_matrix_preserve_luminance(img, color_matrix)
        img = scale_image_to_target_luminance(img, lum_np, lums=lums, pixels=pixels)

    return ImageTk.PhotoImage(img)


def apply_color_matrix_preserve_luminance(img, color_matrix, luma_weights=(0.2126, 0.7152, 0.0722)):
    """
    Apply a 3x3 color matrix to a PIL image while preserving luminance.

    Args:
        img: PIL.Image (RGB or L)
        color_matrix: numpy array shape (3, 3)
        luma_weights: weights for luminance calculation

    Returns:
        PIL.Image in RGB mode
    """
    img_rgb = img.convert('RGB')
    arr = np.asarray(img_rgb, dtype=np.float32)
    corrected = np.dot(arr, color_matrix.T)
    luma = np.array(luma_weights, dtype=np.float32)
    original_luminance = np.dot(arr, luma)
    corrected_luminance = np.dot(corrected, luma)
    scale = np.where(corrected_luminance > 1e-6,
                    original_luminance / corrected_luminance,
                    1.0)
    corrected_scaled = corrected * scale[:, :, np.newaxis]
    corrected_scaled = np.clip(corrected_scaled, 0, 255).astype(np.uint8)
    return Image.fromarray(corrected_scaled, mode='RGB')


def scale_image_to_target_luminance(img, target_lum, lums=None, pixels=None, luma_weights=(0.2126, 0.7152, 0.0722)):
    """
    Scale an image so its mean luminance becomes target_lum (cd/m2).

    If calibration data are available, map pixel values to luminance using the provided
    (lums, pixels) pair and then convert the scaled luminance back to pixel values.
    Otherwise use grayscale image mean as a proxy.
    """
    img_rgb = img.convert('RGB')
    arr = np.asarray(img_rgb, dtype=np.float32)
    luma_weights = np.array(luma_weights, dtype=np.float32)

    gray = np.dot(arr, luma_weights)
    if lums is not None and pixels is not None and len(lums) > 1 and len(pixels) > 1:
        # Map grayscale intensity through calibration curve to cd/m2
        lum_map = np.interp(gray, pixels, lums)
        target_lum_arr = np.asarray(target_lum, dtype=np.float32)

        if target_lum_arr.ndim == 0:
            mean_lum = float(np.mean(lum_map))
            if mean_lum <= 1e-6:
                return img_rgb

            target_lum_map = lum_map * float(target_lum) / mean_lum
            scaled_gray = np.interp(target_lum_map, lums, pixels)
            ratio = np.where(gray > 1e-6, scaled_gray / gray, 0.0)
        else:
            if target_lum_arr.shape != gray.shape:
                target_lum_arr = np.broadcast_to(target_lum_arr, gray.shape)
            scaled_gray = np.interp(target_lum_arr, lums, pixels)
            ratio = np.where(gray > 1e-6, scaled_gray / gray, 0.0)

        scaled_arr = np.clip(arr * ratio[:, :, np.newaxis], 0, 255).astype(np.uint8)
        return Image.fromarray(scaled_arr, mode='RGB')
    else:
        mean_lum = float(np.mean(gray))
        if mean_lum <= 1e-6:
            return img_rgb

        scale = float(target_lum) / mean_lum
        scaled = np.clip(arr * scale, 0, 255).astype(np.uint8)
        return Image.fromarray(scaled, mode='RGB')


def lum_to_pil(lum_np, lums, pixels):
    """
    輝度配列を PIL.Image (Lモード) に変換して返す（保存や加工に使用）
    """
    pix = np.interp(lum_np, lums, pixels).astype(np.uint8)
    return Image.fromarray(pix, mode='L')


def lum_to_pil_window2(lum_np, lums, pixels, color_matrix):
    """
    Window2プレビュー保存用。
    lum_to_photo_window2() と同じ処理を行うが、
    ImageTk.PhotoImage ではなく PIL.Image を返す。
    """
    pix = np.interp(lum_np, lums, pixels)
    g = np.clip(pix, 0, 255).astype(np.float64) / 255.0

    if color_matrix is None:
        gray = np.clip(g * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(gray, mode='L')

    lin = srgb_to_linear(g)
    lin_rgb = np.stack([lin, lin, lin], axis=-1)
    lin_rgb_corr = lin_rgb @ color_matrix.T
    srgb_corr = linear_to_srgb(lin_rgb_corr)
    out = np.clip(srgb_corr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode='RGB')


def generate_matching_photos(gabor_base, cached_lum_noise,
                             fg_lums, fg_pixels, bg_lums, bg_pixels,
                             L_fg, L_bg, L_ref, c_test, ref_c, cond,
                             color_matrix, eotf_bg, eotf_fg, Y_grid, px_grid):
    """
    gabor_base とノイズ基盤から、表示に使う PhotoImage を生成するヘルパ。
    条件に応じて適切な色変換（C適用 or LUT）をディスパッチする。
    """
    out = {}
    lum_ref_fg = L_ref * (1.0 + ref_c * gabor_base)

    if cond in ["Dual plane", "Dual plane flat"]:
        # --- Dual plane系: C適用 ---
        # 参照刺激 (Window2 上側)
        out['photo_ref_fg'] = lum_to_photo_window2(
            lum_ref_fg, bg_lums, bg_pixels, color_matrix,
            eotf_bg, eotf_fg, cond=cond
        )
        # Window2 下側: 前景テスト刺激
        lum_test_fg = L_fg * (1.0 + c_test * gabor_base)
        out['photo_test_fg'] = lum_to_photo_window2(
            lum_test_fg, bg_lums, bg_pixels, color_matrix,
            eotf_bg, eotf_fg, cond=cond
        )

        # Window1: 背景刺激。補正なし。
        lum_noise_bg = cached_lum_noise
        out['photo_noise_bg'] = lum_to_photo(lum_noise_bg, bg_lums, bg_pixels, None)

    else:
        # --- Single plane系: 拡張輝度LUT適用 ---
        # 参照刺激 (Window2 上側)
        out['photo_ref_fg'] = lum_to_photo_window2(
            lum_ref_fg, bg_lums, bg_pixels, color_matrix,
            Y_grid=Y_grid, px_grid=px_grid, cond=cond
        )
        # Window2 下側: 合成刺激
        lum_test_fg = L_fg * (1.0 + c_test * gabor_base)
        lum_noise = cached_lum_noise
        lum_test_total = lum_noise + lum_test_fg
        out['photo_test'] = lum_to_photo_window2(
            lum_test_total, bg_lums, bg_pixels, color_matrix,
            Y_grid=Y_grid, px_grid=px_grid, cond=cond
        )

    return out
