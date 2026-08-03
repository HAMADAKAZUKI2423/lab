"""Image実験の試行生成と画像前処理。"""

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import random

import numpy as np
from PIL import Image
from experiment.common import geometry, optics, photometry

from .config import ImageSessionConfig


SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}
DUAL_PLANE = "Dual plane"
SINGLE_PLANE = "Single plane"
SINGLE_DEFOCUS = "Single plane + defocus simulation"
SINGLE_DEFOCUS_BINOCULAR = (
    "Single plane + defocus + binocular overlay"
)
SUPPORTED_CONDITIONS = {
    DUAL_PLANE,
    SINGLE_PLANE,
    SINGLE_DEFOCUS,
    SINGLE_DEFOCUS_BINOCULAR,
}


@dataclass(frozen=True)
class ImageTrial:
    condition: str
    background_path: Path
    foreground_path: Path


@dataclass
class PreparedImageStimulus:
    foreground: Image.Image
    background: Image.Image | None = None
    singleplane: Image.Image | None = None
    disparity_total_px: float = 0.0
    defocus_difference_d: float = 0.0
    out_of_gamut_ratio: float = 0.0


def discover_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def build_condition_blocks(
    config: ImageSessionConfig,
    background_paths: list[Path],
    foreground_paths: list[Path],
    rng: random.Random,
) -> list[list[ImageTrial]]:
    """条件ごとに試行をまとめ、ブロック順とブロック内順を別々にランダム化する。"""
    unknown = sorted(set(config.conditions) - SUPPORTED_CONDITIONS)
    if unknown:
        raise ValueError(f"unsupported image conditions: {unknown}")

    condition_order = list(config.conditions)
    rng.shuffle(condition_order)
    blocks: list[list[ImageTrial]] = []
    for condition in condition_order:
        trials = [
            ImageTrial(condition, background, foreground)
            for background, foreground in product(
                background_paths, foreground_paths
            )
        ]
        rng.shuffle(trials)
        blocks.append(trials)
    return blocks


def _load_source_images(trial: ImageTrial) -> tuple[Image.Image, Image.Image]:
    with Image.open(trial.background_path) as source:
        background = source.convert("RGB")
    with Image.open(trial.foreground_path) as source:
        foreground = source.convert("RGB")
    background = background.resize((512, 512), Image.Resampling.LANCZOS)
    background = background.crop((0, 128, 512, 384))
    foreground = foreground.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return background, foreground


def _translate_without_wrap(image: Image.Image, shift_x_px: float) -> Image.Image:
    """画像を水平方向へ移動し、はみ出しを反対側へ回り込ませない。"""
    return image.transform(
        image.size,
        Image.Transform.AFFINE,
        (1.0, 0.0, -float(shift_x_px), 0.0, 1.0, 0.0),
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0),
    )


def _calculate_disparity_px(
    ipd_mm: float,
    distance_fg_cm: float,
    distance_bg_cm: float,
) -> float:
    """背景の左右眼像間距離を前景面上のピクセル数で返す。"""
    if ipd_mm <= 0 or distance_fg_cm <= 0 or distance_bg_cm <= 0:
        raise ValueError("IPD and viewing distances must be positive")
    ipd_cm = ipd_mm / 10.0
    shift_cm = ipd_cm * (1.0 - distance_fg_cm / distance_bg_cm)
    return shift_cm * geometry.PIXELS_PER_CM


def _prepare_singleplane_components(
    background: Image.Image,
    foreground: Image.Image,
    config: ImageSessionConfig,
) -> tuple[Image.Image, Image.Image]:
    """BGとFGを前景面の共通キャンバスへ配置する。"""
    canvas_width = geometry.get_size_for_visual_angle(
        config.distance_fg_cm, config.visual_angle_deg * 2.0
    )
    canvas_height = geometry.get_size_for_visual_angle(
        config.distance_fg_cm, config.visual_angle_deg
    )
    foreground_size = geometry.get_size_for_visual_angle(
        config.distance_fg_cm, config.visual_angle_deg
    )
    background = background.resize(
        (canvas_width, canvas_height), Image.Resampling.LANCZOS
    )
    foreground = foreground.resize(
        (foreground_size, foreground_size), Image.Resampling.LANCZOS
    )
    foreground_canvas = Image.new(
        "RGB", (canvas_width, canvas_height), (0, 0, 0)
    )
    foreground_canvas.paste(
        foreground,
        (
            (canvas_width - foreground_size) // 2,
            (canvas_height - foreground_size) // 2,
        ),
    )
    return background, foreground_canvas


def _blur_for_eye(
    background: Image.Image,
    config: ImageSessionConfig,
    pupil_diameter_mm: float,
) -> Image.Image:
    diopter_difference = abs(
        100.0 / config.distance_fg_cm
        - 100.0 / config.distance_bg_cm
    )
    pixels_per_degree = geometry.get_size_for_visual_angle(
        config.distance_fg_cm, 1.0
    )
    return optics.apply_defocus_blur_to_image(
        background,
        diopter_difference,
        pupil_diameter_mm,
        pixels_per_degree,
    )


def prepare_trial_stimulus(
    trial: ImageTrial,
    config: ImageSessionConfig,
    calibration,
    *,
    left_pupil_mm: float,
    right_pupil_mm: float,
    ipd_mm: float,
) -> PreparedImageStimulus:
    background_source, foreground_source = _load_source_images(trial)
    foreground_size = geometry.get_size_for_visual_angle(
        config.distance_fg_cm, config.visual_angle_deg
    )
    foreground_display = foreground_source.resize(
        (foreground_size, foreground_size), Image.Resampling.LANCZOS
    )

    if trial.condition == DUAL_PLANE:
        background_height = geometry.get_size_for_visual_angle(
            config.distance_bg_cm, config.visual_angle_deg
        )
        background_width = geometry.get_size_for_visual_angle(
            config.distance_bg_cm, config.visual_angle_deg * 2.0
        )
        return PreparedImageStimulus(
            foreground=foreground_display,
            background=background_source.resize(
                (background_width, background_height),
                Image.Resampling.LANCZOS,
            ),
        )

    background, foreground_canvas = _prepare_singleplane_components(
        background_source, foreground_source, config
    )
    defocus_difference = 0.0
    disparity_total_px = 0.0

    if trial.condition in {SINGLE_DEFOCUS, SINGLE_DEFOCUS_BINOCULAR}:
        defocus_difference = abs(
            100.0 / config.distance_fg_cm
            - 100.0 / config.distance_bg_cm
        )
        left_background = _blur_for_eye(
            background, config, left_pupil_mm
        )
        right_background = _blur_for_eye(
            background, config, right_pupil_mm
        )
        if trial.condition == SINGLE_DEFOCUS_BINOCULAR:
            disparity_total_px = _calculate_disparity_px(
                ipd_mm,
                config.distance_fg_cm,
                config.distance_bg_cm,
            )
            left_background = _translate_without_wrap(
                left_background, -disparity_total_px / 2.0
            )
            right_background = _translate_without_wrap(
                right_background, disparity_total_px / 2.0
            )
        # 常に左右0.5:0.5。片眼像しかない端部も再正規化しない。
        background = Image.blend(
            left_background, right_background, 0.5
        )
    elif trial.condition != SINGLE_PLANE:
        raise ValueError(f"unsupported image condition: {trial.condition}")

    singleplane, out_of_gamut_ratio = (
        photometry.rgb_paths_to_matrix_singleplane_image(
            background,
            foreground_canvas,
            calibration.t_prime,
            calibration.r_prime,
            calibration.r_prime_inv,
            calibration.gamma_bg,
            calibration.gamma_fg,
        )
    )
    return PreparedImageStimulus(
        foreground=foreground_display,
        singleplane=singleplane,
        disparity_total_px=disparity_total_px,
        defocus_difference_d=defocus_difference,
        out_of_gamut_ratio=out_of_gamut_ratio,
    )