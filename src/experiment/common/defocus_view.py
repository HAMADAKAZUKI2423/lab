"""デフォーカスマッチング刺激の準備とCanvas描画。"""

import numpy as np
from PIL import Image

from . import geometry, markers, optics, patterns, photometry

VISUAL_ANGLE_WIDTH_DEG = 7.9
VISUAL_ANGLE_HEIGHT_DEG = 3.95
WIN2_MARKER_COLOR = "white"
MATCH_MEAN_LUMINANCE = 15.0
MATCH_CONTRAST = 1.0


def _resize_noise(noise: np.ndarray, width: int, height: int) -> np.ndarray:
    """角度空間で生成したノイズを表示面の画素数へ変換する。"""
    image = Image.fromarray(noise.astype(np.float32), mode="F")
    resized = image.resize((width, height), Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.float64)


def prepare_defocus_trial(app, *, cpd: float, seed: int) -> None:
    """1試行分の4 cpdノイズを生成し、スライダー操作中は固定する。"""
    ppd_fg = geometry.get_size_for_visual_angle(app.distance1, 1.0)
    ppd_bg = geometry.get_size_for_visual_angle(app.distance2, 1.0)
    canonical_ppd = max(ppd_fg, ppd_bg)
    canonical_width = int(VISUAL_ANGLE_WIDTH_DEG * canonical_ppd)
    canonical_height = int(VISUAL_ANGLE_HEIGHT_DEG * canonical_ppd)

    rng = np.random.default_rng(seed)
    canonical_noise = patterns.create_noise_base(
        canonical_width,
        canonical_height,
        canonical_ppd,
        cpd,
        rng=rng,
    )

    foreground_width = int(VISUAL_ANGLE_WIDTH_DEG * ppd_fg)
    foreground_height = int(VISUAL_ANGLE_HEIGHT_DEG * ppd_fg)
    background_width = int(VISUAL_ANGLE_WIDTH_DEG * ppd_bg)
    background_height = int(VISUAL_ANGLE_HEIGHT_DEG * ppd_bg)

    foreground_noise = _resize_noise(
        canonical_noise, foreground_width, foreground_height
    )
    background_noise = _resize_noise(
        canonical_noise, background_width, background_height
    )
    app.defocus_foreground_luminance = MATCH_MEAN_LUMINANCE * (
        1.0 + MATCH_CONTRAST * foreground_noise
    )
    app.defocus_background_luminance = MATCH_MEAN_LUMINANCE * (
        1.0 + MATCH_CONTRAST * background_noise
    )
    app.defocus_foreground_size = (foreground_width, foreground_height)
    app.defocus_background_size = (background_width, background_height)
    app.photo_match_bg = photometry.luminance_to_photo(
        app.defocus_background_luminance,
        app.bg_lums,
        app.bg_pixels,
    )


def update_defocus_view(app) -> None:
    """固定したノイズへ現在の瞳孔径のブラーを適用して描画する。"""
    app.canvas1.delete("match")
    app.canvas2.delete("match")
    app.canvas2.delete("calib")

    distance_fg = app.distance1
    distance_bg = app.distance2
    diopter_difference = (
        abs(100.0 / distance_fg - 100.0 / distance_bg)
        if distance_fg > 0 and distance_bg > 0 else 0.0
    )
    ppd_fg = geometry.get_size_for_visual_angle(distance_fg, 1.0)
    blurred_luminance = optics.apply_defocus_blur_to_luminance(
        app.defocus_foreground_luminance,
        diopter_difference,
        app.pupil_diameter_val.get(),
        ppd_fg,
    )
    blurred_luminance = np.fliplr(blurred_luminance)

    if (
        getattr(app, "color_matrix", None) is not None
        and getattr(app, "gamma_bg", None) is not None
        and getattr(app, "gamma_fg", None) is not None
    ):
        app.photo_match_fg = photometry.luminance_to_dualplane_photo(
            blurred_luminance,
            app.bg_lums,
            app.bg_pixels,
            app.color_matrix,
            app.gamma_bg,
            app.gamma_fg,
        )
    else:
        app.photo_match_fg = photometry.luminance_to_photo(
            blurred_luminance,
            app.bg_lums,
            app.bg_pixels,
        )

    foreground_width, foreground_height = app.defocus_foreground_size
    _, background_height = app.defocus_background_size
    foreground_offset = foreground_height // 2
    background_offset = -background_height // 2

    app.canvas1.create_image(
        app.width // 2 + app.offset_x.get(),
        app.height // 2 + app.offset_y.get() + background_offset,
        image=app.photo_match_bg,
        anchor="center",
        tags="match",
    )
    app.canvas2.create_image(
        app.canvas2.winfo_width() // 2,
        app.canvas2.winfo_height() // 2 + foreground_offset,
        image=app.photo_match_fg,
        anchor="center",
        tags="match",
    )
    for offset in (-foreground_offset, foreground_offset):
        markers.draw_image_corner_brackets(
            app.canvas2,
            foreground_width,
            foreground_height,
            offset_y=offset,
            color=WIN2_MARKER_COLOR,
        )
        markers.draw_center_cross(
            app.canvas2,
            offset_y=offset,
            color=WIN2_MARKER_COLOR,
        )
