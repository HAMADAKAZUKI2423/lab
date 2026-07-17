"""デフォーカスマッチング刺激の準備とCanvas描画。"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageTk

from experiment.stimuli import defocus_stimuli

from . import geometry, markers, optics, photometry

VISUAL_ANGLE_DEG = 7.9
WIN2_MARKER_COLOR = "white"
MATCH_MEAN_LUMINANCE = 15.0
MATCH_CONTRAST = 1.0
EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
LAB_ROOT = EXPERIMENT_DIR.parents[1]
STIMULUS_OUTPUT_DIR = LAB_ROOT / "data" / "processed" / "images" / "pre-experiment-matching"


def _ensure_stimuli(app):
    conditions = tuple(sorted(set(app.defocus_match_patterns)))
    cache_key = (float(app.distance1), float(app.distance2), conditions)
    if getattr(app, "_prepared_defocus_stimuli_key", None) == cache_key:
        return
    patterns = tuple(dict.fromkeys(pattern for pattern, _ in conditions))
    cpds = tuple(sorted({cpd for _, cpd in conditions}))
    defocus_stimuli.ensure_defocus_stimuli(
        distance_fg=app.distance1,
        distance_bg=app.distance2,
        patterns=patterns,
        cpds=cpds,
        output_dir=str(STIMULUS_OUTPUT_DIR),
    )
    app._prepared_defocus_stimuli_key = cache_key
    return


def update_defocus_view(app) -> None:
    """現在の条件と瞳孔径に対応する刺激を描画する。"""
    _ensure_stimuli(app)
    app.canvas1.delete("match")
    app.canvas2.delete("match")
    app.canvas2.delete("calib")
    distance_fg = app.distance1
    distance_bg = app.distance2
    foreground_size = geometry.get_size_for_visual_angle(distance_fg, VISUAL_ANGLE_DEG)
    background_size = geometry.get_size_for_visual_angle(distance_bg, VISUAL_ANGLE_DEG)
    diopter_difference = (
        abs(100.0 / distance_fg - 100.0 / distance_bg)
        if distance_fg > 0 and distance_bg > 0 else 0.0
    )
    pixels_per_degree = geometry.get_size_for_visual_angle(distance_fg, 1.0)
    pattern_name, cpd = app.defocus_match_patterns[app.current_match_idx]
    foreground_path = defocus_stimuli.get_stimulus_path(
        str(STIMULUS_OUTPUT_DIR), "FG", pattern_name, distance_fg, cpd
    )
    background_path = defocus_stimuli.get_stimulus_path(
        str(STIMULUS_OUTPUT_DIR), "BG", pattern_name, distance_bg, cpd
    )
    with Image.open(foreground_path) as source:
        foreground = source.convert("L")
    with Image.open(background_path) as source:
        background = source.convert("L")
    foreground = foreground.resize(
        (foreground_size, foreground_size // 2), Image.Resampling.LANCZOS
    )
    foreground = optics.apply_defocus_blur_to_image(
        foreground, diopter_difference, app.pupil_diameter_val.get(), pixels_per_degree
    ).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    background = background.resize(
        (background_size, background_size // 2), Image.Resampling.LANCZOS
    )
    foreground_base = np.asarray(foreground, dtype=np.float64) / 255.0
    foreground_luminance = MATCH_MEAN_LUMINANCE * (
        1.0 + MATCH_CONTRAST * (2.0 * foreground_base - 1.0)
    )
    if (
        getattr(app, "color_matrix", None) is not None
        and getattr(app, "gamma_bg", None) is not None
        and getattr(app, "gamma_fg", None) is not None
    ):
        app.photo_match_fg = photometry.luminance_to_dualplane_photo(
            foreground_luminance, app.bg_lums, app.bg_pixels,
            app.color_matrix, app.gamma_bg, app.gamma_fg,
        )
    else:
        app.photo_match_fg = ImageTk.PhotoImage(
            photometry.luminance_to_pil(foreground_luminance, app.bg_lums, app.bg_pixels)
        )
    background_base = np.asarray(background, dtype=np.float64) / 255.0
    background_luminance = MATCH_MEAN_LUMINANCE * (
        1.0 + MATCH_CONTRAST * (2.0 * background_base - 1.0)
    )
    app.photo_match_bg = ImageTk.PhotoImage(
        photometry.luminance_to_pil(background_luminance, app.bg_lums, app.bg_pixels)
    )
    foreground_offset = foreground_size // 4
    background_offset = -background_size // 4
    app.canvas1.create_image(
        app.width // 2 + app.offset_x.get(),
        app.height // 2 + app.offset_y.get() + background_offset,
        image=app.photo_match_bg, anchor="center", tags="match",
    )
    app.canvas2.create_image(
        app.canvas2.winfo_width() // 2,
        app.canvas2.winfo_height() // 2 + foreground_offset,
        image=app.photo_match_fg, anchor="center", tags="match",
    )
    for offset in (-foreground_offset, foreground_offset):
        markers.draw_image_corner_brackets(
            app.canvas2, foreground_size, foreground_size // 2,
            offset_y=offset, color=WIN2_MARKER_COLOR,
        )
        markers.draw_center_cross(app.canvas2, offset_y=offset, color=WIN2_MARKER_COLOR)
