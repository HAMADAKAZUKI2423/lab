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
    """Single planeはT'・R'のMatrix法、DPは実際の光学加算で生成する。"""
    # ReferenceもBG/FGの2経路に分け、T'とR'でXYZ加算してFG1画面へ再現する。
    component_total = config.l_bg + config.l_fg
    if component_total <= 0:
        raise ValueError("l_bg + l_fg must be positive")
    reference_modulation = 1.0 + (
        reference_contrast * prepared.gabor_base
    )
    reference_background_luminance = (
        config.l_ref * config.l_bg / component_total
        * reference_modulation
    )
    reference_foreground_luminance = (
        config.l_ref * config.l_fg / component_total
        * reference_modulation
    )
    reference_photo = (
        photometry.luminance_components_to_matrix_singleplane_photo(
            reference_background_luminance,
            reference_foreground_luminance,
            calibration.bg_lums,
            calibration.bg_pixels,
            calibration.color_matrix,
            calibration.t_prime,
            calibration.r_prime,
            calibration.r_prime_inv,
            calibration.gamma_bg,
            calibration.gamma_fg,
        )
    )

    foreground_luminance = config.l_fg * (
        1.0 + test_contrast * prepared.gabor_base
    )
    if condition in DUAL_CONDITIONS:
        return {
            "photo_ref_fg": reference_photo,
            "photo_test_fg": photometry.luminance_to_window2_photo(
                foreground_luminance,
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

    # PDF一般式をそのまま用い、BGにはT'、FGにはR'を適用した
    # XYZ増分を加算してから、inv(R')でFG1画面へ戻す。
    test_photo = (
        photometry.luminance_components_to_matrix_singleplane_photo(
            prepared.background_luminance,
            foreground_luminance,
            calibration.bg_lums,
            calibration.bg_pixels,
            calibration.color_matrix,
            calibration.t_prime,
            calibration.r_prime,
            calibration.r_prime_inv,
            calibration.gamma_bg,
            calibration.gamma_fg,
        )
    )
    return {
        "photo_ref_fg": reference_photo,
        "photo_test": test_photo,
    }


def save_preview_images(
    save_dir: Path,
    config: MatchingSessionConfig,
    calibration: DisplayCalibration,
    pupil_diameter_mm: float,
) -> None:
    """重複を除いた代表刺激を、指定コントラストごとに保存する。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    test_contrasts = (0.1, 1.0)
    reference_contrasts = (0.1, 0.2)

    def save_photo(photo, filename: str) -> None:
        tk_photo = getattr(photo, "_PhotoImage__photo", None)
        if tk_photo is None:
            print(f"WARN: preview image could not be saved: {filename}")
            return
        tk_photo.write(str(save_dir / filename), format="png")

    prepared_by_condition = {
        condition: prepare_trial_stimulus(
            condition=condition,
            ocularity="binocular",
            dominant_eye="Right",
            orientation=0,
            pupil_diameter_mm=pupil_diameter_mm,
            config=config,
            rng=rng,
        )
        for condition in config.conditions
    }

    # ref前景は条件間で共通なので、参照コントラストごとに1枚だけ保存する。
    reference_condition = config.conditions[0]
    reference_prepared = prepared_by_condition[reference_condition]
    for contrast in reference_contrasts:
        photos = generate_trial_photos(
            reference_prepared,
            condition=reference_condition,
            test_contrast=test_contrasts[0],
            reference_contrast=contrast,
            config=config,
            calibration=calibration,
        )
        save_photo(photos["photo_ref_fg"], f"ref_fg_c{contrast:g}.png")

    # Single plane系は背景と前景を合成したtest画像を条件・コントラスト別に保存する。
    for condition in ("Single plane", "Single plane + defocus simulation"):
        if condition not in prepared_by_condition:
            continue
        prepared = prepared_by_condition[condition]
        condition_label = (
            "single_defocus" if condition.endswith("defocus simulation") else "single"
        )
        for contrast in test_contrasts:
            photos = generate_trial_photos(
                prepared,
                condition=condition,
                test_contrast=contrast,
                reference_contrast=reference_contrasts[0],
                config=config,
                calibration=calibration,
            )
            save_photo(
                photos["photo_test"],
                f"{condition_label}_test_c{contrast:g}.png",
            )

    # Dual plane系のtest前景は条件間で共通なので、コントラストごとに1枚だけ保存する。
    dual_condition = next(
        (condition for condition in ("Dual plane", "Dual plane flat")
         if condition in prepared_by_condition),
        None,
    )
    if dual_condition is not None:
        prepared = prepared_by_condition[dual_condition]
        for contrast in test_contrasts:
            photos = generate_trial_photos(
                prepared,
                condition=dual_condition,
                test_contrast=contrast,
                reference_contrast=reference_contrasts[0],
                config=config,
                calibration=calibration,
            )
            save_photo(
                photos["photo_test_fg"],
                f"dual_test_fg_c{contrast:g}.png",
            )

    # Dual plane背景はnoise/flatごとに1枚だけ保存する。
    for condition, label in (
        ("Dual plane", "noise"),
        ("Dual plane flat", "flat"),
    ):
        if condition not in prepared_by_condition:
            continue
        photos = generate_trial_photos(
            prepared_by_condition[condition],
            condition=condition,
            test_contrast=test_contrasts[0],
            reference_contrast=reference_contrasts[0],
            config=config,
            calibration=calibration,
        )
        save_photo(photos["photo_noise_bg"], f"dual_test_bg_{label}.png")