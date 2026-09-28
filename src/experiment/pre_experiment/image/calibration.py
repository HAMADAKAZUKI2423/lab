"""Image実験の表示校正と、条件ごとの眼位置合わせを扱う。"""

from pathlib import Path

from experiment.common.display_calibration import (
    DisplayCalibration,
    load_display_calibration,
)

from .conditions import get_condition, resolve_reference_eye


def initialize_defocus_calibration(
    app,
    display_dir: Path,
) -> DisplayCalibration:
    calibration = load_display_calibration(display_dir)
    app.color_matrix = calibration.color_matrix
    app.gamma_bg = calibration.gamma_bg
    app.gamma_fg = calibration.gamma_fg
    app.bg_lums = calibration.bg_lums
    app.bg_pixels = calibration.bg_pixels
    return calibration


def trial_alignment_eye(app, condition_id: str) -> str:
    """片眼DPは指定眼、それ以外はdominant eyeの位置合わせを使う。"""
    spec = get_condition(condition_id)
    reference = resolve_reference_eye(
        spec, app.participant_dominance.get()
    )
    return reference if reference in {"Left", "Right"} else (
        app.participant_dominance.get().strip().capitalize()
    )


def apply_eye_calibration(app, eye: str) -> None:
    eye = str(eye).strip().capitalize()
    if eye not in app.calib_results:
        raise KeyError(f"Calibration result is missing for {eye} eye")
    result = app.calib_results[eye]
    app.offset_x.set(int(result["offset_x"]))
    app.offset_y.set(int(result["offset_y"]))
    app.current_pd_mean = float(result["pd_mean"])
    app.current_alignment_eye = eye


def apply_trial_calibration(app, condition_id: str) -> str:
    eye = trial_alignment_eye(app, condition_id)
    apply_eye_calibration(app, eye)
    return eye


def apply_dominant_eye_calibration(app) -> None:
    """旧previewコードとの互換用。"""
    eye = app.participant_dominance.get().strip().capitalize()
    if eye not in app.calib_results:
        eye = "Right"
    apply_eye_calibration(app, eye)