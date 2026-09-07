"""個別・training用グラフで使う試行単位のcontrast指標を計算する。"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .config import AR_VALUE_COLUMN, DP_CONDITION, SPD_CONDITION


RAW_CONTRAST_COLUMN = "Matched_Contrast"
AR_CONTRAST_COLUMN = AR_VALUE_COLUMN
EXTENDED_CONTRAST_COLUMN = "Matched_Contrast_Enhanced"
DOMINANT_PD_COLUMN = "Dominant_PD_mm"
BLUR_ATTENUATION_COLUMN = "Blur_Attenuation"
EFFECTIVE_BG_CONTRAST_COLUMN = "Effective_BG_Contrast"

VISUAL_ANGLE_WIDTH_DEG = 7.9
VISUAL_ANGLE_HEIGHT_DEG = 3.95
WIN2_TOTAL_WIDTH_FACTOR = 2.6
SPATIAL_FREQUENCY_CPD = 4.0
DEFAULT_DISTANCE_FG_CM = 50.0
DEFAULT_DISTANCE_BG_CM = 150.0
DEFAULT_BACKGROUND_LUMINANCE = 15.0
BACKGROUND_CONTRAST = 1.0
_BLURRED_BACKGROUND_CONDITIONS = (SPD_CONDITION, DP_CONDITION)
_REQUIRED_COLUMNS = (
    "Condition",
    RAW_CONTRAST_COLUMN,
    "L_fg",
    "L_bg",
    "Dominance",
    "PD_Right",
    "PD_Left",
)


class ContrastMetricError(ValueError):
    """グラフ用contrast指標を計算できない入力に対する例外。"""


def _validate_and_normalize(trials: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in _REQUIRED_COLUMNS if column not in trials.columns]
    if missing:
        raise ContrastMetricError(
            "raw/ar/extendedグラフに必要な列が不足しています: "
            f"{missing}"
        )
    if trials.empty:
        raise ContrastMetricError("グラフ用の試行データが空です")

    frame = trials.copy(deep=True)
    frame["Condition"] = frame["Condition"].astype("string").str.strip()
    if (frame["Condition"].isna() | frame["Condition"].eq("")).any():
        raise ContrastMetricError("Conditionに空値があります")

    dominance = frame["Dominance"].astype("string").str.strip().str.lower()
    invalid_dominance = dominance.isna() | ~dominance.isin(("left", "right"))
    if invalid_dominance.any():
        rows = frame.index[invalid_dominance].tolist()[:10]
        raise ContrastMetricError(
            "DominanceはLeftまたはRightにしてください: "
            f"row_index={rows}"
        )
    frame["Dominance"] = dominance

    for column in (
        RAW_CONTRAST_COLUMN,
        "L_fg",
        "L_bg",
        "PD_Right",
        "PD_Left",
    ):
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = values.isna() | ~np.isfinite(values.to_numpy(dtype=float))
        if invalid.any():
            rows = frame.index[invalid].tolist()[:10]
            raise ContrastMetricError(
                f"{column}を有限な数値へ変換できません: row_index={rows}"
            )
        frame[column] = values.astype(float)

    denominator = frame["L_fg"] + frame["L_bg"]
    invalid_denominator = denominator <= 0
    if invalid_denominator.any():
        rows = frame.index[invalid_denominator].tolist()[:10]
        raise ContrastMetricError(
            f"L_fg + L_bg > 0が必要です: row_index={rows}"
        )
    if (frame[RAW_CONTRAST_COLUMN] <= 0).any():
        rows = frame.index[frame[RAW_CONTRAST_COLUMN] <= 0].tolist()[:10]
        raise ContrastMetricError(
            f"{RAW_CONTRAST_COLUMN} > 0が必要です: row_index={rows}"
        )
    return frame


def _dominant_pupil_diameter(frame: pd.DataFrame) -> np.ndarray:
    return np.where(
        frame["Dominance"].eq("left"),
        frame["PD_Left"].to_numpy(dtype=float),
        frame["PD_Right"].to_numpy(dtype=float),
    )


@lru_cache(maxsize=None)
def _blur_attenuation_cached(
    pupil_diameter_mm: float,
    distance_fg_cm: float,
    distance_bg_cm: float,
    spatial_frequency_cpd: float,
) -> float:
    """実験と同じFFTデフォーカス処理からRMS contrast減衰率を返す。"""
    if (
        pupil_diameter_mm <= 0
        or distance_fg_cm <= 0
        or distance_bg_cm <= 0
        or spatial_frequency_cpd <= 0
    ):
        return 1.0

    # full-analysisモードではtorchを必要としないよう、光学依存を遅延読込する。
    from experiment.common import geometry, optics, patterns

    ppd_fg = geometry.get_size_for_visual_angle(distance_fg_cm, 1.0)
    width_base = int(VISUAL_ANGLE_WIDTH_DEG * ppd_fg)
    width_px = int(width_base * WIN2_TOTAL_WIDTH_FACTOR)
    height_px = int(VISUAL_ANGLE_HEIGHT_DEG * ppd_fg)
    if ppd_fg <= 0 or width_px <= 0 or height_px <= 0:
        return 1.0

    noise_base = patterns.create_noise_base(
        width_px,
        height_px,
        ppd_fg,
        spatial_frequency_cpd,
        rng=np.random.default_rng(42),
    )
    luminance_original = DEFAULT_BACKGROUND_LUMINANCE * (
        1.0 + BACKGROUND_CONTRAST * noise_base
    )
    diopter_difference = abs(
        100.0 / distance_fg_cm - 100.0 / distance_bg_cm
    )
    luminance_blurred = optics.apply_defocus_blur_to_luminance(
        luminance_original,
        diopter_difference,
        pupil_diameter_mm,
        ppd_fg,
    )
    rms_original = float(np.std(luminance_original))
    rms_blurred = float(np.std(luminance_blurred))
    attenuation = 1.0 if rms_original <= 1e-12 else rms_blurred / rms_original
    return float(np.clip(attenuation, 0.0, 1.0))


def calculate_blur_attenuation(
    pupil_diameter_mm: float,
    *,
    distance_fg_cm: float = DEFAULT_DISTANCE_FG_CM,
    distance_bg_cm: float = DEFAULT_DISTANCE_BG_CM,
    spatial_frequency_cpd: float = SPATIAL_FREQUENCY_CPD,
) -> float:
    """丸めた入力をキャッシュし、同じ瞳孔径のFFT計算を再利用する。"""
    return _blur_attenuation_cached(
        round(float(pupil_diameter_mm), 2),
        round(float(distance_fg_cm), 3),
        round(float(distance_bg_cm), 3),
        round(float(spatial_frequency_cpd), 3),
    )


def build_legacy_contrast_frame(trials: pd.DataFrame) -> pd.DataFrame:
    """添付旧版と同じraw・AR・extended指標を試行表へ追加する。

    DPF補正や参加者集約は行わず、入力の各試行をそのまま保持する。
    """
    frame = _validate_and_normalize(trials)
    frame[DOMINANT_PD_COLUMN] = _dominant_pupil_diameter(frame)

    requires_blur = frame["Condition"].isin(_BLURRED_BACKGROUND_CONDITIONS)
    frame[BLUR_ATTENUATION_COLUMN] = 1.0
    if requires_blur.any():
        pupil_values = sorted(
            set(frame.loc[requires_blur, DOMINANT_PD_COLUMN].astype(float))
        )
        attenuation_by_pupil = {
            value: calculate_blur_attenuation(value) for value in pupil_values
        }
        frame.loc[requires_blur, BLUR_ATTENUATION_COLUMN] = (
            frame.loc[requires_blur, DOMINANT_PD_COLUMN]
            .map(attenuation_by_pupil)
            .to_numpy(dtype=float)
        )

    frame[EFFECTIVE_BG_CONTRAST_COLUMN] = (
        BACKGROUND_CONTRAST * frame[BLUR_ATTENUATION_COLUMN]
    )
    denominator = frame["L_fg"] + frame["L_bg"]
    calculated_ar = (
        frame["L_fg"] * frame[RAW_CONTRAST_COLUMN] / denominator
    )
    if AR_CONTRAST_COLUMN in frame.columns:
        recorded_ar = pd.to_numeric(frame[AR_CONTRAST_COLUMN], errors="coerce")
        consistent = np.isfinite(recorded_ar.to_numpy(dtype=float)) & np.isclose(
            recorded_ar.to_numpy(dtype=float),
            calculated_ar.to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-12,
        )
        if not bool(consistent.all()):
            rows = frame.index[~consistent].tolist()[:10]
            raise ContrastMetricError(
                f"既存の{AR_CONTRAST_COLUMN}が計算式と一致しません: "
                f"row_index={rows}"
            )
    frame[AR_CONTRAST_COLUMN] = calculated_ar
    frame[EXTENDED_CONTRAST_COLUMN] = (
        frame[RAW_CONTRAST_COLUMN] * frame["L_fg"]
        + frame[EFFECTIVE_BG_CONTRAST_COLUMN] * frame["L_bg"]
    ) / denominator

    for column in (AR_CONTRAST_COLUMN, EXTENDED_CONTRAST_COLUMN):
        values = frame[column].to_numpy(dtype=float)
        invalid = ~np.isfinite(values) | (values <= 0)
        if invalid.any():
            rows = frame.index[invalid].tolist()[:10]
            raise ContrastMetricError(
                f"{column} > 0の有限値が必要です: row_index={rows}"
            )
    return frame


__all__ = [
    "AR_CONTRAST_COLUMN",
    "BACKGROUND_CONTRAST",
    "BLUR_ATTENUATION_COLUMN",
    "ContrastMetricError",
    "DOMINANT_PD_COLUMN",
    "EFFECTIVE_BG_CONTRAST_COLUMN",
    "EXTENDED_CONTRAST_COLUMN",
    "RAW_CONTRAST_COLUMN",
    "build_legacy_contrast_frame",
    "calculate_blur_attenuation",
]