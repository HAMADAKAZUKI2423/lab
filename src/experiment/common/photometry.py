"""輝度・画素値・ガンマ・色補正の変換。"""

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk


def load_luminance_lut(path: str | Path):
    """輝度LUTのCSVまたはCSVディレクトリを読み、平均値を返す。"""
    path = Path(path)
    if not path.exists():
        return None, None
    csv_files = [path] if path.is_file() and path.suffix == ".csv" else (
        sorted(path.glob("*.csv")) if path.is_dir() else []
    )
    values: dict[float, list[int]] = {}
    for csv_path in csv_files:
        try:
            with csv_path.open(newline="", encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    luminance = row.get("Target_Luminance(cd/m2)", row.get("luminance"))
                    pixel = row.get("Pixel_Value", row.get("pixel_value"))
                    if luminance not in (None, "") and pixel not in (None, ""):
                        values.setdefault(float(luminance), []).append(int(float(pixel)))
        except (OSError, ValueError) as exc:
            print(f"WARN: {csv_path} could not be read: {exc}")
    if not values:
        return None, None
    averaged = [
        (luminance, int(np.round(np.mean(pixels))))
        for luminance, pixels in sorted(values.items())
    ]
    return (
        np.array([item[0] for item in averaged]),
        np.array([item[1] for item in averaged]),
    )


def srgb_to_linear(values):
    values = np.asarray(values, dtype=np.float64)
    return np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.0,
    )


def linear_to_srgb(values):
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return np.where(
        values <= 0.0031308,
        values * 12.92,
        1.055 * values ** (1.0 / 2.0) - 0.055,
    )


def _apply_gamma(gamma_by_channel, channel, values, inverse=False):
    gamma = float(gamma_by_channel[channel])
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError(f"invalid gamma for {channel}: {gamma}")
    values = np.asarray(values, dtype=np.float64)
    if inverse:
        return np.power(np.clip(values, 0.0, None), 1.0 / gamma)
    return np.power(np.clip(values, 0.0, 1.0), gamma)


def luminance_to_dualplane_photo(
    luminance,
    background_luminances,
    background_pixels,
    color_matrix,
    gamma_background,
    gamma_foreground,
):
    """目標輝度を色補正済みの前景PhotoImageへ変換する。"""
    pixels = np.clip(
        np.interp(luminance, background_luminances, background_pixels), 0, 255
    ) / 255.0
    linear_background = np.stack(
        [_apply_gamma(gamma_background, channel, pixels) for channel in "RGB"],
        axis=-1,
    )
    linear_foreground = np.clip(linear_background @ color_matrix.T, 0.0, None)
    output = np.empty_like(linear_foreground)
    for index, channel in enumerate("RGB"):
        output[..., index] = _apply_gamma(
            gamma_foreground, channel, linear_foreground[..., index], inverse=True
        )
    image = Image.fromarray(
        np.clip(output * 255.0, 0, 255).astype(np.uint8), mode="RGB"
    )
    return ImageTk.PhotoImage(image)


def luminance_components_to_matrix_singleplane_photo(
    background_luminance,
    foreground_luminance,
    background_luminances,
    background_pixels,
    color_matrix,
    t_prime,
    r_prime,
    r_prime_inv,
    gamma_background,
    gamma_foreground,
):
    """T'とR'を明示的に使い、2経路のXYZ増分を加算してFGで再現する。"""

    def to_background_linear(luminance):
        pixels = np.clip(
            np.interp(
                luminance,
                background_luminances,
                background_pixels,
            ),
            0,
            255,
        ) / 255.0
        return np.stack(
            [
                _apply_gamma(gamma_background, channel, pixels)
                for channel in "RGB"
            ],
            axis=-1,
        )

    # 実験で表示する実際の入力を個別に構築する。
    linear_bg = to_background_linear(background_luminance)
    linear_fg_reference = to_background_linear(foreground_luminance)
    linear_fg = linear_fg_reference @ color_matrix.T

    # PDF一般式: XYZ_sum = T'd_bg + R'd_fg
    xyz_bg = linear_bg @ t_prime.T
    xyz_fg = linear_fg @ r_prime.T
    xyz_sum = xyz_bg + xyz_fg

    # BGは黒表示のままなので、共通黒を除いたXYZ増分だけをFG入力へ戻す。
    linear_foreground_sum = xyz_sum @ r_prime_inv.T

    out_of_gamut = (
        np.any(linear_foreground_sum < -1e-9)
        or np.any(linear_foreground_sum > 1.0 + 1e-9)
    )
    if out_of_gamut:
        print(
            "WARN: matrix Single plane RGB is out of gamut: "
            f"min={float(np.min(linear_foreground_sum)):.6f}, "
            f"max={float(np.max(linear_foreground_sum)):.6f}"
        )

    linear_foreground_sum = np.clip(
        linear_foreground_sum, 0.0, 1.0
    )
    output = np.empty_like(linear_foreground_sum)
    for index, channel in enumerate("RGB"):
        output[..., index] = _apply_gamma(
            gamma_foreground,
            channel,
            linear_foreground_sum[..., index],
            inverse=True,
        )
    return ImageTk.PhotoImage(
        Image.fromarray(
            np.clip(output * 255.0, 0, 255).astype(np.uint8),
            mode="RGB",
        )
    )


def luminance_to_singleplane_photo(luminance, luminance_grid, pixel_grid):
    """1次元カラーLUTを使ってPhotoImageへ変換する。"""
    luminance = np.asarray(luminance, dtype=np.float64)
    output = np.empty(luminance.shape + (3,), dtype=np.float64)
    for channel in range(3):
        output[..., channel] = np.interp(
            luminance, luminance_grid, pixel_grid[:, channel]
        )
    return ImageTk.PhotoImage(
        Image.fromarray(
            np.clip(output * 255.0, 0, 255).astype(np.uint8), mode="RGB"
        )
    )


def luminance_to_window2_photo(
    luminance,
    luminances,
    pixels,
    color_matrix,
    gamma_background=None,
    gamma_foreground=None,
    luminance_grid=None,
    pixel_grid=None,
    condition="",
):
    """実験条件に応じたWindow 2用PhotoImageを返す。"""
    if (
        condition in {"Dual plane", "Dual plane flat"}
        and gamma_background
        and gamma_foreground
        and color_matrix is not None
    ):
        return luminance_to_dualplane_photo(
            luminance, luminances, pixels, color_matrix,
            gamma_background, gamma_foreground,
        )
    if condition.startswith("Single plane") and luminance_grid is not None:
        return luminance_to_singleplane_photo(
            luminance, luminance_grid, pixel_grid
        )
    pixel_values = np.interp(luminance, luminances, pixels)
    if color_matrix is None:
        return ImageTk.PhotoImage(
            Image.fromarray(
                np.clip(pixel_values, 0, 255).astype(np.uint8), mode="L"
            )
        )
    gray = np.clip(pixel_values, 0, 255).astype(np.float64) / 255.0
    linear_rgb = np.stack([srgb_to_linear(gray)] * 3, axis=-1)
    corrected = linear_to_srgb(linear_rgb @ color_matrix.T)
    return ImageTk.PhotoImage(
        Image.fromarray(
            np.clip(corrected * 255.0, 0, 255).astype(np.uint8), mode="RGB"
        )
    )


def luminance_to_photo(luminance, luminances, pixels):
    """輝度配列をグレースケールPhotoImageへ変換する。"""
    return ImageTk.PhotoImage(luminance_to_pil(luminance, luminances, pixels))


def luminance_to_pil(luminance, luminances, pixels):
    """輝度配列をグレースケールPIL画像へ変換する。"""
    pixel_values = np.interp(luminance, luminances, pixels)
    return Image.fromarray(
        np.clip(pixel_values, 0, 255).astype(np.uint8), mode="L"
    )
