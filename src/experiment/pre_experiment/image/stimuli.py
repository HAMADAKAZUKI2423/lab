"""20画像ペア×有効条件の試行生成と、条件別の表示画像生成。"""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import random

import numpy as np
from PIL import Image

from experiment.common import geometry, optics, photometry

from .calibration import DisplayCalibration
from .conditions import get_condition, resolve_reference_eye
from .config import ImageSessionConfig
from .disparity import (
    fuse_eye_views,
    make_constant_eye_views,
    right_aligned_disparity_px,
)
from .selected_pairs import SelectedPair


RESAMPLE = Image.Resampling.LANCZOS


@dataclass(frozen=True)
class ImageTrial:
    trial_order: int
    block_id: int
    within_block_order: int
    condition_id: str
    pair: SelectedPair


@dataclass
class PreparedImageStimulus:
    window1_both: Image.Image | None
    window2_foreground_only: Image.Image
    window2_both: Image.Image
    metadata: dict[str, object]


def participant_seed(base_seed: int, participant_id: str) -> int:
    digest = hashlib.sha256(
        f"{base_seed}:{participant_id.strip()}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big")


def condition_order_for_participant(
    condition_ids: tuple[str, ...],
    participant_id: str,
    base_seed: int,
) -> tuple[str, ...]:
    """奇数・偶数条件に対応した均衡順序を参加者IDから決定する。"""
    count = len(condition_ids)
    if count < 2:
        raise ValueError("At least two image conditions are required")
    first_row = [0]
    low, high = 1, count - 1
    while len(first_row) < count:
        first_row.append(low)
        low += 1
        if len(first_row) < count:
            first_row.append(high)
            high -= 1

    base_rows = [
        [((index + shift) % count) for index in first_row]
        for shift in range(count)
    ]
    # 奇数条件では逆順行も加え、直前条件の方向を均衡させる。
    design_rows = (
        base_rows
        if count % 2 == 0
        else base_rows + [list(reversed(row)) for row in base_rows]
    )
    row_index = (
        participant_seed(base_seed, participant_id) % len(design_rows)
    )
    return tuple(condition_ids[index] for index in design_rows[row_index])


def _stratified_pair_order(
    pairs: list[SelectedPair],
    rng: random.Random,
) -> list[SelectedPair]:
    high = [pair for pair in pairs if pair.difference_group == "high"]
    low = [pair for pair in pairs if pair.difference_group == "low"]
    if len(high) != 10 or len(low) != 10:
        raise ValueError("Each condition requires high 10 and low 10 pairs")
    rng.shuffle(high)
    rng.shuffle(low)
    first_half = high[:5] + low[:5]
    second_half = high[5:] + low[5:]
    rng.shuffle(first_half)
    rng.shuffle(second_half)
    return first_half + second_half


def build_trials(
    pairs: list[SelectedPair],
    condition_order: tuple[str, ...],
    rng: random.Random,
) -> list[ImageTrial]:
    trials: list[ImageTrial] = []
    trial_order = 1
    for block_id, condition_id in enumerate(condition_order, start=1):
        get_condition(condition_id)
        for within_block_order, pair in enumerate(
            _stratified_pair_order(pairs, rng), start=1
        ):
            trials.append(ImageTrial(
                trial_order=trial_order,
                block_id=block_id,
                within_block_order=within_block_order,
                condition_id=condition_id,
                pair=pair,
            ))
            trial_order += 1
    expected_trial_count = len(condition_order) * len(pairs)
    if len(trials) != expected_trial_count:
        raise RuntimeError(
            f"Expected {expected_trial_count} trials, generated {len(trials)}"
        )
    return trials


def _read_rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        return source.convert("RGB").copy()


def _prepare_sources(
    pair: SelectedPair,
    config: ImageSessionConfig,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    foreground_size = geometry.get_size_for_visual_angle(
        config.distance_fg_cm, config.visual_angle_deg
    )
    background_height = geometry.get_size_for_visual_angle(
        config.distance_bg_cm, config.visual_angle_deg
    )
    background_width = geometry.get_size_for_visual_angle(
        config.distance_bg_cm, config.background_visual_angle_width_deg
    )
    singleplane_background_height = geometry.get_size_for_visual_angle(
        config.distance_fg_cm, config.visual_angle_deg
    )
    singleplane_background_width = geometry.get_size_for_visual_angle(
        config.distance_fg_cm, config.background_visual_angle_width_deg
    )

    foreground = _read_rgb(pair.foreground_path).resize(
        (foreground_size, foreground_size), RESAMPLE
    )
    background_square = _read_rgb(pair.background_path).resize(
        (512, 512), RESAMPLE
    )
    background_strip = background_square.crop((0, 128, 512, 384))
    physical_background = background_strip.resize(
        (background_width, background_height), RESAMPLE
    )
    singleplane_background = background_strip.resize(
        (singleplane_background_width, singleplane_background_height),
        RESAMPLE,
    )
    return foreground, physical_background, singleplane_background


def _apply_channel_gamma(
    values: np.ndarray,
    gamma: dict[str, float],
    *,
    inverse: bool,
) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    for index, channel in enumerate("RGB"):
        exponent = float(gamma[channel])
        if exponent <= 0:
            raise ValueError(f"Invalid gamma for {channel}: {exponent}")
        power = 1.0 / exponent if inverse else exponent
        output[..., index] = np.clip(values[..., index], 0.0, 1.0) ** power
    return output


def _color_correct_foreground(
    foreground: Image.Image,
    calibration: DisplayCalibration,
) -> Image.Image:
    if calibration.gamma_bg is None or calibration.gamma_fg is None:
        raise RuntimeError("gamma_bg.csv and gamma_fg.csv are required")
    encoded = np.asarray(foreground.convert("RGB"), dtype=np.float64) / 255.0
    linear_background_reference = _apply_channel_gamma(
        encoded, calibration.gamma_bg, inverse=False
    )
    linear_foreground = np.clip(
        linear_background_reference @ calibration.color_matrix.T,
        0.0,
        None,
    )
    output = _apply_channel_gamma(
        linear_foreground, calibration.gamma_fg, inverse=True
    )
    return Image.fromarray(
        np.clip(output * 255.0, 0, 255).astype(np.uint8), "RGB"
    )


def _blur_eye_background(
    image: Image.Image,
    *,
    pupil_mm: float,
    config: ImageSessionConfig,
    pixels_per_degree: float,
) -> Image.Image:
    distance_fg_m = config.distance_fg_cm / 100.0
    distance_bg_m = config.distance_bg_cm / 100.0
    diopter_difference = abs(1.0 / distance_fg_m - 1.0 / distance_bg_m)
    pupil = (
        pupil_mm
        if pupil_mm > 0 else config.initial_pupil_diameter_mm
    )
    return optics.apply_defocus_blur_to_image(
        image,
        diopter_difference,
        pupil,
        pixels_per_degree,
    )


def prepare_trial_stimulus(
    trial: ImageTrial,
    config: ImageSessionConfig,
    *,
    calibration: DisplayCalibration,
    dominant_eye: str,
    left_pupil_mm: float,
    right_pupil_mm: float,
    ipd_mm: float,
) -> PreparedImageStimulus:
    spec = get_condition(trial.condition_id)
    (
        foreground_source,
        physical_background,
        singleplane_background,
    ) = _prepare_sources(trial.pair, config)
    corrected_foreground = _color_correct_foreground(
        foreground_source, calibration
    )
    foreground_display = corrected_foreground.transpose(
        Image.Transpose.FLIP_LEFT_RIGHT
    )

    base_metadata: dict[str, object] = {
        "Plane_Mode": spec.plane_mode,
        "Viewing_Mode": spec.viewing_mode,
        "Reference_Eye": resolve_reference_eye(spec, dominant_eye),
        "Defocus_Mode": spec.defocus_mode,
        "Disparity_Mode": spec.disparity_mode,
        "Reproduced_Eye": spec.reproduced_eye,
        "Disparity_Px": 0,
        "Disparity_Map_Path": "",
        "SinglePlane_OutOfGamut_Ratio": 0.0,
    }
    if spec.plane_mode == "dual":
        return PreparedImageStimulus(
            window1_both=physical_background,
            window2_foreground_only=foreground_display,
            window2_both=foreground_display,
            metadata=base_metadata,
        )

    pixels_per_degree = foreground_source.width / config.visual_angle_deg
    if spec.disparity_mode == "simple":
        disparity_px = right_aligned_disparity_px(
            ipd_mm=ipd_mm,
            distance_fg_cm=config.distance_fg_cm,
            distance_bg_cm=config.distance_bg_cm,
            pixels_per_degree=pixels_per_degree,
        )
        right_background, left_background = make_constant_eye_views(
            singleplane_background, disparity_px
        )
    else:
        disparity_px = 0
        right_background = singleplane_background.copy()
        left_background = singleplane_background.copy()

    if spec.defocus_mode == "matched_simulation":
        right_background = _blur_eye_background(
            right_background,
            pupil_mm=right_pupil_mm,
            config=config,
            pixels_per_degree=pixels_per_degree,
        )
        left_background = _blur_eye_background(
            left_background,
            pupil_mm=left_pupil_mm,
            config=config,
            pixels_per_degree=pixels_per_degree,
        )

    if spec.reproduced_eye == "Left":
        reproduced_background = left_background
    elif spec.reproduced_eye == "Right":
        reproduced_background = right_background
    elif spec.reproduced_eye == "Both":
        reproduced_background = fuse_eye_views(
            right_background,
            left_background,
            dominant_eye=dominant_eye,
            dominant_weight=config.binocular_fusion_dominant_weight,
        )
    else:
        reproduced_background = singleplane_background

    if (
        corrected_foreground.width > reproduced_background.width
        or corrected_foreground.height > reproduced_background.height
    ):
        raise ValueError(
            "Foreground image does not fit inside the Single Plane background"
        )
    foreground_layer = Image.new(
        "RGB", reproduced_background.size, (0, 0, 0)
    )
    foreground_position = (
        (reproduced_background.width - corrected_foreground.width) // 2,
        (reproduced_background.height - corrected_foreground.height) // 2,
    )
    foreground_layer.paste(corrected_foreground, foreground_position)

    singleplane, out_of_gamut_ratio = (
        photometry.rgb_paths_to_matrix_singleplane_image(
            reproduced_background,
            foreground_layer,
            calibration.t_prime,
            calibration.r_prime,
            calibration.r_prime_inv,
            calibration.gamma_bg,
            calibration.gamma_fg,
        )
    )
    singleplane_display = singleplane.transpose(
        Image.Transpose.FLIP_LEFT_RIGHT
    )
    base_metadata.update({
        "Disparity_Px": disparity_px,
        "SinglePlane_OutOfGamut_Ratio": out_of_gamut_ratio,
    })
    return PreparedImageStimulus(
        window1_both=None,
        window2_foreground_only=foreground_display,
        window2_both=singleplane_display,
        metadata=base_metadata,
    )