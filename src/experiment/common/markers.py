"""Tkinter Canvas上の位置合わせマーカー描画。"""


SQUARE_SIZE = 30
CROSS_SIZE = 30
MARKER_LINE_WIDTH = 5


def draw_image_corner_brackets(
    canvas,
    size_w: int,
    size_h: int,
    offset_x: int = 0,
    offset_y: int = 0,
    color: str = "white",
    flip_x: bool = False,
    line_width: float = MARKER_LINE_WIDTH,
) -> None:
    """指定領域の四隅に鍵括弧状のマーカーを描く。"""
    center_x = canvas.winfo_width() // 2
    center_y = canvas.winfo_height() // 2
    x0 = center_x - size_w // 2 + offset_x
    y0 = center_y - size_h // 2 + offset_y
    x1 = center_x + size_w // 2 + offset_x
    y1 = center_y + size_h // 2 + offset_y

    def transform_x(x):
        return canvas.winfo_width() - x if flip_x else x

    segments = (
        (x0, y0, x0 + SQUARE_SIZE, y0),
        (x0, y0, x0, y0 + SQUARE_SIZE),
        (x1 - SQUARE_SIZE, y0, x1, y0),
        (x1, y0, x1, y0 + SQUARE_SIZE),
        (x0, y1 - SQUARE_SIZE, x0, y1),
        (x0, y1, x0 + SQUARE_SIZE, y1),
        (x1 - SQUARE_SIZE, y1, x1, y1),
        (x1, y1 - SQUARE_SIZE, x1, y1),
    )
    for start_x, start_y, end_x, end_y in segments:
        canvas.create_line(
            transform_x(start_x),
            start_y,
            transform_x(end_x),
            end_y,
            fill=color,
            width=line_width,
            tags="calib",
        )


def draw_center_cross(
    canvas,
    offset_x: int = 0,
    offset_y: int = 0,
    color: str = "white",
    gap: int = 0,
) -> None:
    """画面中央へ向かう4つの矢尻を描く。"""
    center_x = canvas.winfo_width() // 2 + offset_x
    center_y = canvas.winfo_height() // 2 + offset_y
    length = int(CROSS_SIZE * 1.2)
    notch = int(CROSS_SIZE * 0.9)
    half_width = int(CROSS_SIZE * 0.5)
    points = (
        [center_x - gap, center_y, center_x - length - gap, center_y - half_width,
         center_x - notch - gap, center_y, center_x - length - gap, center_y + half_width],
        [center_x + gap, center_y, center_x + length + gap, center_y - half_width,
         center_x + notch + gap, center_y, center_x + length + gap, center_y + half_width],
        [center_x, center_y - gap, center_x - half_width, center_y - length - gap,
         center_x, center_y - notch - gap, center_x + half_width, center_y - length - gap],
        [center_x, center_y + gap, center_x - half_width, center_y + length + gap,
         center_x, center_y + notch + gap, center_x + half_width, center_y + length + gap],
    )
    for polygon in points:
        canvas.create_polygon(
            polygon, fill=color, outline="red", width=2, tags="calib"
        )
