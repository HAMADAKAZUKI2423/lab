"""matching固有の刺激準備。数値処理をTkinter UIから分離する。"""

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
from PIL import Image
from experiment.common import geometry, optics, patterns, photometry

from .calibration import DisplayCalibration
from .config import MatchingSessionConfig


DUAL_CONDITIONS = {"Dual plane", "Dual plane flat"}


@dataclass
class PreparedTrialStimulus:
    gabor_base: np.ndarray
    background_luminance: np.ndarray
    background_center_offset_x: int
    ppd_fg: float
    ppd_bg: float


def slider_to_contrast(value: float) -> float:
    return (10.0 ** (2.0 * value) - 1.0) / 99.0


def contrast_to_slider(contrast: float) -> float:
    contrast = min(1.0, max(0.0, float(contrast)))
    return math.log10(contrast * 99.0 + 1.0) / 2.0


def build_blocks(config: MatchingSessionConfig, rng) -> list[dict]:
    groups = [
        [{"condition": condition, "ocularity": ocularity}
         for condition in config.conditions]
        for ocularity in config.ocularities
    ]
    for group in groups:
        rng.shuffle(group)
    rng.shuffle(groups)
    return [block for group in groups for block in group]


def build_trials(config: MatchingSessionConfig, rng) -> list[dict]:
    trials = [
        {"ref_contrast": contrast, "orientation": orientation}
        for contrast in config.ref_contrasts
        for orientation in config.orientations
        for _ in range(config.repetitions)
    ]
    rng.shuffle(trials)
    return trials


def background_geometry(
    base_width: int,
    condition: str,
    ocularity: str,
    dominant_eye: str,
    config: MatchingSessionConfig,
) -> tuple[int, int]:
    if condition not in DUAL_CONDITIONS:
        return base_width, 0
    if ocularity == "monocular":
        return int(base_width * config.asym_width_factor_large * 2.0), 0

    large = config.asym_width_factor_large
    small = config.asym_width_factor_small
    left, right = (small, large) if dominant_eye == "Right" else (large, small)
    width = int(base_width * (left + right))
    offset = int(base_width * (right - left) / 2.0)
    return width, offset


def prepare_trial_stimulus(
    *,
    condition: str,
    ocularity: str,
    dominant_eye: str,
    orientation: int,
    pupil_diameter_mm: float,
    config: MatchingSessionConfig,
    rng: np.random.Generator | None = None,
) -> PreparedTrialStimulus:
    ppd_fg = geometry.get_size_for_visual_angle(config.distance_fg_cm, 1.0)
    ppd_bg = geometry.get_size_for_visual_angle(config.distance_bg_cm, 1.0)
    width_fg = int(
        config.visual_angle_width_deg * ppd_fg * config.win2_total_width_factor
    )
    height_fg = int(config.visual_angle_height_deg * ppd_fg)
    width_bg_base = int(config.visual_angle_width_deg * ppd_bg)
    height_bg = int(config.visual_angle_height_deg * ppd_bg)
    width_bg, offset = background_geometry(
        width_bg_base, condition, ocularity, dominant_eye, config
    )

    gabor = patterns.create_cosine_windowed_grating_base(
        width_fg, height_fg, ppd_fg, config.spatial_frequency,
        orientation=orientation,
    )
    if condition == "Dual plane flat":
        background = np.full((height_bg, width_bg), config.l_bg, dtype=np.float32)
    else:
        if condition == "Dual plane":
            noise = patterns.create_noise_base(
                width_bg, height_bg, ppd_bg, config.spatial_frequency,
                rng=rng,
            )
        else:
            noise = patterns.create_noise_base(
                width_fg, height_fg, ppd_fg, config.spatial_frequency,
                rng=rng,
            )
        background = config.l_bg * (1.0 + noise)

    if condition == "Single plane + defocus simulation":
        d_fg_m = config.distance_fg_cm / 100.0
        d_bg_m = config.distance_bg_cm / 100.0
        diopter_difference = abs(1.0 / d_fg_m - 1.0 / d_bg_m)
        background = optics.apply_defocus_blur_to_luminance(
            background, diopter_difference, pupil_diameter_mm, ppd_fg
        )

    return PreparedTrialStimulus(gabor, background, offset, ppd_fg, ppd_bg)


def generate_trial_photos(
    prepared: PreparedTrialStimulus,
    *,
    condition: str,
    test_contrast: float,
    reference_contrast: float,
    config: MatchingSessionConfig,
    calibration: DisplayCalibration,
):
    """matching条件を共通の輝度変換関数へ割り当てる。"""
    reference_luminance = config.l_ref * (
        1.0 + reference_contrast * prepared.gabor_base
    )
    if condition in DUAL_CONDITIONS:
        if calibration.ext_lum_y is not None and calibration.ext_lum_px is not None:
            reference_photo = photometry.luminance_to_singleplane_photo(
                reference_luminance,
                calibration.ext_lum_y,
                calibration.ext_lum_px,
            )
        else:
            reference_photo = photometry.luminance_to_window2_photo(
                reference_luminance,
                calibration.bg_lums,
                calibration.bg_pixels,
                calibration.color_matrix,
                calibration.gamma_bg,
                calibration.gamma_fg,
                condition=condition,
            )
        test_luminance = config.l_fg * (
            1.0 + test_contrast * prepared.gabor_base
        )
        return {
            "photo_ref_fg": reference_photo,
            "photo_test_fg": photometry.luminance_to_window2_photo(
                test_luminance,
                calibration.bg_lums,
                calibration.bg_pixels,
                calibration.color_matrix,
                calibration.gamma_bg,
                calibration.gamma_fg,
                condition=condition,
            ),
            "photo_noise_bg": photometry.luminance_to_photo(
                prepared.background_luminance,
                calibration.bg_lums,
                calibration.bg_pixels,
            ),
        }
    test_luminance = (
        prepared.background_luminance
        + config.l_fg * (1.0 + test_contrast * prepared.gabor_base)
    )
    return {
        "photo_ref_fg": photometry.luminance_to_window2_photo(
            reference_luminance,
            calibration.bg_lums,
            calibration.bg_pixels,
            calibration.color_matrix,
            luminance_grid=calibration.ext_lum_y,
            pixel_grid=calibration.ext_lum_px,
            condition=condition,
        ),
        "photo_test": photometry.luminance_to_window2_photo(
            test_luminance,
            calibration.bg_lums,
            calibration.bg_pixels,
            calibration.color_matrix,
            luminance_grid=calibration.ext_lum_y,
            pixel_grid=calibration.ext_lum_px,
            condition=condition,
        ),
    }


def save_preview_images(
    save_dir: Path,
    config: MatchingSessionConfig,
    calibration: DisplayCalibration,
    pupil_diameter_mm: float,
) -> None:
    """乱数状態を変えず、代表刺激を保存する。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for condition in config.conditions:
        prepared = prepare_trial_stimulus(
            condition=condition,
            ocularity="binocular",
            dominant_eye="Right",
            orientation=0,
            pupil_diameter_mm=pupil_diameter_mm,
            config=config,
            rng=rng,
        )
        photos = generate_trial_photos(
            prepared,
            condition=condition,
            test_contrast=1.0,
            reference_contrast=0.2,
            config=config,
            calibration=calibration,
        )
        for key, photo in photos.items():
            image = getattr(photo, "_PhotoImage__photo", None)
            if isinstance(image, Image.Image):
                image.save(save_dir / f"{condition}_{key}.png")