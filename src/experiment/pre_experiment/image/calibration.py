"""Image実験のキャリブレーション状態を扱う。"""

from pathlib import Path

from experiment.common.display_calibration import (
    DisplayCalibration,
    load_display_calibration,
)


def initialize_defocus_calibration(
    app,
    display_dir: Path,
) -> DisplayCalibration:
    """defocus matchingとSingle plane生成用の校正を読み込む。"""
    calibration = load_display_calibration(display_dir)

    app.color_matrix = calibration.color_matrix
    app.gamma_bg = calibration.gamma_bg
    app.gamma_fg = calibration.gamma_fg
    app.bg_lums = calibration.bg_lums
    app.bg_pixels = calibration.bg_pixels

    return calibration


def apply_dominant_eye_calibration(app) -> None:
    """優位眼の位置と推定瞳孔径を本試行へ適用する。"""
    eye = app.participant_dominance.get()
    if eye not in app.calib_results:
        eye = "Right"
    result = app.calib_results[eye]
    app.offset_x.set(result["offset_x"])
    app.offset_y.set(result["offset_y"])
    app.current_pd_mean = result["pd_mean"]
