# py .\src\experiment\stimuli\defocus_stimuli.py
import math
import os
import zlib

import numpy as np
from PIL import Image


SCREEN_WIDTH_CM = 59.67
SCREEN_RES_X_PX = 2560
PIXELS_PER_CM = SCREEN_RES_X_PX / SCREEN_WIDTH_CM
STIM_WIDTH_DEG = 7.9
STIM_HEIGHT_DEG = 3.95
DEFAULT_PATTERNS = ("checker", "checker_45", "stripe", "border", "noise")
DEFAULT_CPDS = (2, 4)

script_dir = os.path.dirname(os.path.abspath(__file__))
lab_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
DEFAULT_OUTPUT_DIR = os.path.join(
    lab_root, "data", "processed", "images", "pre-experiment-matching"
)


def get_size_for_visual_angle(distance_cm, angle_deg):
    """観視距離と視角から刺激サイズをピクセル単位で求める。"""
    if distance_cm <= 0:
        raise ValueError(f"distance_cm must be positive: {distance_cm}")
    size_cm = 2.0 * distance_cm * math.tan(math.radians(angle_deg) / 2.0)
    return round(size_cm * PIXELS_PER_CM)


def pixels_per_degree(distance_cm):
    """1度に対応するピクセル数を返す。"""
    return float(get_size_for_visual_angle(distance_cm, 1.0))


def _distance_label(distance_cm):
    value = float(distance_cm)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def get_stimulus_path(output_dir, prefix, pattern, distance_cm, cpd):
    """指定条件に対応するdefocus matching刺激の保存先を返す。"""
    return os.path.join(
        output_dir,
        "defocus-matching",
        f"{prefix}_{pattern}_{_distance_label(distance_cm)}cm_{cpd}cpd.png",
    )


def _condition_seed(prefix, pattern, distance_cm, cpd):
    """noiseを同一条件で再現できるよう、条件から固定seedを作る。"""
    key = f"{prefix}|{pattern}|{_distance_label(distance_cm)}|{cpd}"
    return zlib.crc32(key.encode("utf-8"))


def create_band_limited_noise(
    width_px,
    height_px,
    ppd,
    f_center_cpd,
    rng,
    bandwidth_octave=1.0,
):
    """固定された乱数生成器を使って帯域制限ノイズを作る。"""
    white_noise = rng.normal(0.0, 1.0, (height_px, width_px))
    ft_noise = np.fft.fftshift(np.fft.fft2(white_noise))

    fx = np.fft.fftshift(np.fft.fftfreq(width_px, d=1.0 / ppd))
    fy = np.fft.fftshift(np.fft.fftfreq(height_px, d=1.0 / ppd))
    fx_grid, fy_grid = np.meshgrid(fx, fy)
    radius = np.sqrt(fx_grid**2 + fy_grid**2)

    f_min = f_center_cpd / (2.0 ** (bandwidth_octave / 2.0))
    f_max = f_center_cpd * (2.0 ** (bandwidth_octave / 2.0))
    mask = (radius >= f_min) & (radius <= f_max)

    filtered = np.real(
        np.fft.ifft2(np.fft.ifftshift(ft_noise * mask))
    )
    max_value = float(np.max(np.abs(filtered)))
    return filtered / max_value if max_value > 0.0 else filtered


def create_defocus_pattern(pattern, width_px, height_px, ppd, cpd, rng):
    """defocus matching用の輝度変調を-1から1の範囲で生成する。"""
    half_period_px = max(1, int(ppd / (2.0 * cpd)))
    x = np.arange(width_px)
    y = np.arange(height_px)
    x_grid, y_grid = np.meshgrid(x, y)

    if pattern == "checker":
        modulation = (
            (x_grid // half_period_px) + (y_grid // half_period_px)
        ) % 2
    elif pattern == "checker_45":
        scale = math.sqrt(2.0) * half_period_px
        u = np.floor((x_grid + y_grid) / scale).astype(int)
        v = np.floor((x_grid - y_grid) / scale).astype(int)
        modulation = (u + v) % 2
    elif pattern == "stripe":
        modulation = (x_grid // half_period_px) % 2
    elif pattern == "border":
        modulation = (y_grid // half_period_px) % 2
    elif pattern == "noise":
        return create_band_limited_noise(
            width_px,
            height_px,
            ppd,
            f_center_cpd=cpd,
            rng=rng,
        )
    else:
        raise ValueError(f"unknown defocus pattern: {pattern}")

    return modulation.astype(np.float64) * 2.0 - 1.0


def _generate_one(output_path, prefix, pattern, distance_cm, cpd):
    ppd = pixels_per_degree(distance_cm)
    width_px = get_size_for_visual_angle(distance_cm, STIM_WIDTH_DEG)
    height_px = get_size_for_visual_angle(distance_cm, STIM_HEIGHT_DEG)
    rng = np.random.default_rng(
        _condition_seed(prefix, pattern, distance_cm, cpd)
    )
    modulation = create_defocus_pattern(
        pattern, width_px, height_px, ppd, cpd, rng
    )
    pixel_map = np.clip(
        (modulation + 1.0) * 0.5 * 255.0, 0.0, 255.0
    ).astype(np.uint8)
    Image.fromarray(pixel_map, mode="L").save(output_path)


def ensure_defocus_stimuli(
    distance_fg,
    distance_bg,
    patterns=DEFAULT_PATTERNS,
    cpds=DEFAULT_CPDS,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    """必要な刺激を確認し、存在しない画像だけを生成する。"""
    patterns = tuple(dict.fromkeys(patterns))
    cpds = tuple(dict.fromkeys(cpds))
    unknown = sorted(set(patterns) - set(DEFAULT_PATTERNS))
    if unknown:
        raise ValueError(f"unsupported defocus patterns: {unknown}")
    if not cpds or any(float(cpd) <= 0 for cpd in cpds):
        raise ValueError(f"cpds must be positive: {cpds}")

    matching_dir = os.path.join(output_dir, "defocus-matching")
    os.makedirs(matching_dir, exist_ok=True)

    generated_paths = []
    targets = (("FG", distance_fg), ("BG", distance_bg))
    for prefix, distance_cm in targets:
        for pattern in patterns:
            for cpd in cpds:
                path = get_stimulus_path(
                    output_dir, prefix, pattern, distance_cm, cpd
                )
                if os.path.isfile(path):
                    continue
                _generate_one(path, prefix, pattern, distance_cm, cpd)
                generated_paths.append(path)
                print(f"Generated missing defocus stimulus: {path}")

    required_paths = [
        get_stimulus_path(output_dir, prefix, pattern, distance_cm, cpd)
        for prefix, distance_cm in targets
        for pattern in patterns
        for cpd in cpds
    ]
    missing_paths = [path for path in required_paths if not os.path.isfile(path)]
    if missing_paths:
        raise FileNotFoundError(
            "defocus matching stimuli could not be prepared:\n"
            + "\n".join(missing_paths)
        )
    return generated_paths


if __name__ == "__main__":
    generated = ensure_defocus_stimuli(distance_fg=50, distance_bg=150)
    if generated:
        print(f"Generated {len(generated)} missing defocus stimuli.")
    else:
        print("All defocus matching stimuli already exist.")