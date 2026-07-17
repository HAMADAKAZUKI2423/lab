"""デフォーカスに関する光学計算とFFT画像処理。"""

import math

import numpy as np
from PIL import Image
import torch

DEFOCUS_BLUR_SCALE_FACTOR = 0.55


def calculate_defocus_parameters(
    diopter_difference: float,
    pupil_diameter_mm: float,
) -> tuple[float, float]:
    """デフォーカス円直径[deg]とガウス近似sigma[deg]を返す。"""
    if diopter_difference <= 0 or pupil_diameter_mm <= 0:
        return 0.0, 0.0
    blur_diameter_deg = (
        180.0 / math.pi * diopter_difference * pupil_diameter_mm * 1e-3
    )
    sigma_deg = DEFOCUS_BLUR_SCALE_FACTOR * blur_diameter_deg / 2.0
    return blur_diameter_deg, sigma_deg


def _apply_fft_blur(values: np.ndarray, sigma_deg: float, pixels_per_deg: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.from_numpy(values.astype(np.float32)).to(device)
    height, width = tensor.shape[-2:]
    x_deg = torch.linspace(
        -width / 2 / pixels_per_deg, width / 2 / pixels_per_deg, width,
        device=device,
    )
    y_deg = torch.linspace(
        -height / 2 / pixels_per_deg, height / 2 / pixels_per_deg, height,
        device=device,
    )
    grid_y, grid_x = torch.meshgrid(y_deg, x_deg, indexing="ij")
    psf = torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma_deg**2))
    psf = psf / torch.sum(psf)
    if tensor.ndim == 3:
        psf = psf.unsqueeze(0)

    def transform(value):
        return torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(value, dim=(-2, -1)), norm="ortho"),
            dim=(-2, -1),
        )

    def inverse_transform(value):
        return torch.fft.fftshift(
            torch.fft.ifft2(torch.fft.ifftshift(value, dim=(-2, -1)), norm="ortho"),
            dim=(-2, -1),
        )

    blurred = torch.abs(inverse_transform(transform(tensor) * transform(psf)))
    if tensor.ndim == 3:
        for channel in range(tensor.shape[0]):
            blurred[channel] *= torch.sum(tensor[channel]) / (
                torch.sum(blurred[channel]) + 1e-8
            )
    else:
        blurred *= torch.sum(tensor) / (torch.sum(blurred) + 1e-8)
    return blurred.cpu().numpy()


def apply_defocus_blur_to_image(
    image: Image.Image,
    diopter_difference: float,
    pupil_diameter_mm: float,
    pixels_per_deg: float,
) -> Image.Image:
    """PIL画像へデフォーカスブラーを適用する。"""
    _, sigma_deg = calculate_defocus_parameters(
        diopter_difference, pupil_diameter_mm
    )
    if sigma_deg <= 0:
        return image
    values = np.asarray(image, dtype=np.float32)
    is_rgb = values.ndim == 3
    if is_rgb:
        values = np.transpose(values, (2, 0, 1))
    blurred = _apply_fft_blur(values, sigma_deg, pixels_per_deg)
    if is_rgb:
        blurred = np.transpose(blurred, (1, 2, 0))
    return Image.fromarray(
        np.clip(blurred, 0, 255).astype(np.uint8), mode=image.mode
    )


def apply_defocus_blur_to_luminance(
    luminance: np.ndarray,
    diopter_difference: float,
    pupil_diameter_mm: float,
    pixels_per_deg: float,
) -> np.ndarray:
    """輝度配列へデフォーカスブラーを適用する。"""
    _, sigma_deg = calculate_defocus_parameters(
        diopter_difference, pupil_diameter_mm
    )
    if sigma_deg <= 0:
        return luminance
    return _apply_fft_blur(luminance, sigma_deg, pixels_per_deg)
