"""Image実験固有のキャリブレーション状態を扱う。"""

import numpy as np


def initialize_defocus_compatibility(app) -> None:
    """共通defocus_matching.pyが参照する表示属性を初期化する。"""
    app.color_matrix = None
    app.gamma_bg = None
    app.gamma_fg = None
    app.bg_lums = np.array([0.0, 30.0], dtype=np.float64)
    app.bg_pixels = np.array([0.0, 255.0], dtype=np.float64)


def apply_dominant_eye_calibration(app) -> None:
    """優位眼の位置と推定瞳孔径を本試行へ適用する。"""
    eye = app.participant_dominance.get()
    if eye not in app.calib_results:
        eye = "Right"
    result = app.calib_results[eye]
    app.offset_x.set(result["offset_x"])
    app.offset_y.set(result["offset_y"])
    app.current_pd_mean = result["pd_mean"]