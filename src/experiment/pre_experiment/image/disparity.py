"""Single Plane条件で使う左右眼背景像とbinocular overlayを生成する。"""

import math

import numpy as np
from PIL import Image


def right_aligned_disparity_px(
    *,
    ipd_mm: float,
    distance_fg_cm: float,
    distance_bg_cm: float,
    pixels_per_degree: float,
) -> int:
    if ipd_mm <= 0:
        raise ValueError(f"IPD must be positive: {ipd_mm}")
    if not 0 < distance_fg_cm < distance_bg_cm:
        raise ValueError("distance_fg_cm must be smaller than distance_bg_cm")
    if pixels_per_degree <= 0:
        raise ValueError(f"pixels_per_degree must be positive: {pixels_per_degree}")
    ipd_cm = ipd_mm / 10.0
    disparity_deg = math.degrees(
        math.atan2(ipd_cm, distance_fg_cm)
        - math.atan2(ipd_cm, distance_bg_cm)
    )
    return round(disparity_deg * pixels_per_degree)


def make_constant_eye_views(
    background: Image.Image,
    disparity_px: int,
) -> tuple[Image.Image, Image.Image]:
    """背景全幅を保ち、端を反射補完した一様視差の左右眼像を返す。"""
    values = np.asarray(background.convert("RGB"), dtype=np.uint8)
    shift = int(disparity_px)
    if shift == 0:
        copied = Image.fromarray(values.copy(), "RGB")
        return copied, copied.copy()

    padding = abs(shift)
    padding_mode = "reflect" if values.shape[1] > 1 else "edge"
    padded = np.pad(
        values,
        ((0, 0), (padding, padding), (0, 0)),
        mode=padding_mode,
    )
    width = values.shape[1]
    right_start = padding
    left_start = padding + shift
    right = padded[:, right_start:right_start + width]
    left = padded[:, left_start:left_start + width]
    return Image.fromarray(right, "RGB"), Image.fromarray(left, "RGB")

def fuse_eye_views(
    right: Image.Image,
    left: Image.Image,
    *,
    dominant_eye: str,
    dominant_weight: float = 0.5,
) -> Image.Image:
    """左右眼像を単一面提示用に線形overlayする。"""
    if right.size != left.size:
        raise ValueError(f"Eye-view sizes differ: {right.size} != {left.size}")
    if not 0.0 <= dominant_weight <= 1.0:
        raise ValueError("dominant_weight must be in [0, 1]")
    right_values = np.asarray(right.convert("RGB"), dtype=np.float32)
    left_values = np.asarray(left.convert("RGB"), dtype=np.float32)
    dominant = str(dominant_eye).strip().capitalize()
    if dominant == "Right":
        right_weight = dominant_weight
    elif dominant == "Left":
        right_weight = 1.0 - dominant_weight
    else:
        raise ValueError(f"dominant_eye must be Left or Right: {dominant_eye}")
    output = right_weight * right_values + (1.0 - right_weight) * left_values
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), "RGB")