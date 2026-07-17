"""観視距離・視角と表示ピクセル数の変換。"""

import math

PIXELS_PER_CM = 1 / 0.02331


def get_size_for_visual_angle(
    distance_cm: float,
    angle_deg: float,
    pixels_per_cm: float = PIXELS_PER_CM,
    canvas=None,
) -> int:
    """指定した観視距離と視角に対応する表示サイズを返す。"""
    if distance_cm <= 0:
        return 0
    if canvas is not None:
        try:
            pixels_per_cm = float(canvas.winfo_fpixels("1i")) / 2.54
        except Exception:
            pass
    size_cm = 2.0 * distance_cm * math.tan(math.radians(angle_deg) / 2.0)
    return round(size_cm * pixels_per_cm)
