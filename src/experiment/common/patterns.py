"""実験刺激に用いるステートレスな数値パターン生成。"""

import numpy as np


def create_noise_base(
    width_px: int,
    height_px: int,
    ppd: float,
    f_center_cpd: float,
    bandwidth_octave: float = 1.0,
) -> np.ndarray:
    """中心周波数周辺へ帯域制限したノイズを-1から1で返す。"""
    white_noise = np.random.normal(0, 1, (height_px, width_px))
    noise_spectrum = np.fft.fftshift(np.fft.fft2(white_noise))
    frequency_x = np.fft.fftshift(np.fft.fftfreq(width_px, d=1 / ppd))
    frequency_y = np.fft.fftshift(np.fft.fftfreq(height_px, d=1 / ppd))
    grid_x, grid_y = np.meshgrid(frequency_x, frequency_y)
    radius = np.sqrt(grid_x**2 + grid_y**2)
    minimum = f_center_cpd / (2 ** (bandwidth_octave / 2))
    maximum = f_center_cpd * (2 ** (bandwidth_octave / 2))
    filtered = np.real(
        np.fft.ifft2(
            np.fft.ifftshift(noise_spectrum * ((radius >= minimum) & (radius <= maximum)))
        )
    )
    maximum_value = float(np.max(np.abs(filtered)))
    return filtered / maximum_value if maximum_value > 0 else filtered


def create_cosine_windowed_disk(
    width_px: int,
    height_px: int,
    ppd: float,
    disk_diameter_deg: float = 3.0,
    fade_width_deg: float = 0.5,
) -> np.ndarray:
    """周辺をコサイン減衰させた円形マスクを返す。"""
    total_radius = disk_diameter_deg * ppd / 2.0
    fade_width = fade_width_deg * ppd
    flat_radius = max(0.0, total_radius - fade_width)
    if flat_radius == 0:
        fade_width = total_radius
    x = np.linspace(-width_px / 2, width_px / 2, width_px)
    y = np.linspace(-height_px / 2, height_px / 2, height_px)
    grid_x, grid_y = np.meshgrid(x, y)
    radius = np.sqrt(grid_x**2 + grid_y**2)
    mask = np.zeros((height_px, width_px))
    mask[radius <= flat_radius] = 1.0
    fade_zone = (radius > flat_radius) & (radius <= total_radius)
    if fade_width > 0:
        normalized = (radius[fade_zone] - flat_radius) / fade_width
        mask[fade_zone] = (np.cos(normalized * np.pi) + 1.0) / 2.0
    return mask


def create_cosine_windowed_grating_base(
    width_px: int,
    height_px: int,
    ppd: float,
    cpd: float,
    orientation: float = 0,
    phase: float = 0,
    disk_diameter_deg: float = 3.0,
    fade_width_deg: float = 0.5,
) -> np.ndarray:
    """コサイン窓を適用した正弦波グレーティングを返す。"""
    x = np.linspace(-width_px / 2, width_px / 2, width_px) / ppd
    y = np.linspace(-height_px / 2, height_px / 2, height_px) / ppd
    grid_x, grid_y = np.meshgrid(x, y)
    theta = np.deg2rad(orientation)
    rotated_x = grid_x * np.cos(theta) + grid_y * np.sin(theta)
    grating = np.sin(2 * np.pi * cpd * rotated_x + phase)
    window = create_cosine_windowed_disk(
        width_px, height_px, ppd, disk_diameter_deg, fade_width_deg
    )
    return grating * window
