"""ディスプレイ校正ファイルの読み込み。"""

from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np
from common.photometry import load_luminance_lut


FALLBACK_COLOR_MATRIX = np.array(
    [
        [0.385676, -0.029594, 0.007298],
        [0.002786, 0.485416, -0.011852],
        [0.005025, 0.003184, 0.601995],
    ],
    dtype=np.float64,
)


@dataclass
class DisplayCalibration:
    color_matrix: np.ndarray
    gamma_bg: dict[str, float] | None
    gamma_fg: dict[str, float] | None
    bg_lums: np.ndarray
    bg_pixels: np.ndarray
    ext_lum_y: np.ndarray | None
    ext_lum_px: np.ndarray | None


def _load_matrix(path: Path) -> np.ndarray:
    try:
        return np.loadtxt(path, delimiter=",")
    except (OSError, ValueError) as exc:
        print(f"WARN: {path.name} could not be loaded ({exc}); using fallback.")
        return FALLBACK_COLOR_MATRIX.copy()


def _load_gamma(path: Path) -> dict[str, float] | None:
    try:
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        gamma = {row["channel"].upper(): float(row["gamma"]) for row in rows}
        missing = {"R", "G", "B"} - set(gamma)
        if missing:
            raise ValueError(f"missing channels: {sorted(missing)}")
        return gamma
    except (OSError, KeyError, ValueError) as exc:
        print(f"WARN: {path.name} could not be loaded ({exc}).")
        return None


def _load_extended_lut(path: Path):
    try:
        values = np.loadtxt(path, delimiter=",", skiprows=1)
        if values.ndim != 2 or values.shape[1] < 4:
            raise ValueError("expected Y,R,G,B columns")
        return values[:, 0], values[:, 1:4]
    except (OSError, ValueError) as exc:
        print(f"WARN: {path.name} could not be loaded ({exc}).")
        return None, None


def load_display_calibration(display_dir: Path) -> DisplayCalibration:
    bg_lums, bg_pixels = load_luminance_lut(
        str(display_dir / "bg_luminance_lut.csv")
    )
    if bg_lums is None or bg_pixels is None:
        print("WARN: background LUT was not found; using linear fallback.")
        bg_lums = np.array([0.0, 100.0], dtype=np.float64)
        bg_pixels = np.array([0.0, 255.0], dtype=np.float64)

    ext_lum_y, ext_lum_px = _load_extended_lut(display_dir / "ext_lum_lut.csv")
    return DisplayCalibration(
        color_matrix=_load_matrix(display_dir / "C.csv"),
        gamma_bg=_load_gamma(display_dir / "gamma_bg.csv"),
        gamma_fg=_load_gamma(display_dir / "gamma_fg.csv"),
        bg_lums=np.asarray(bg_lums, dtype=np.float64),
        bg_pixels=np.asarray(bg_pixels, dtype=np.float64),
        ext_lum_y=ext_lum_y,
        ext_lum_px=ext_lum_px,
    )