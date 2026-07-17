"""Image実験の試行生成と画像前処理。"""

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import random

from PIL import Image
from common import geometry

from .config import ImageSessionConfig


SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}


@dataclass(frozen=True)
class ImageTrial:
    background_path: Path
    foreground_path: Path


@dataclass
class PreparedImageStimulus:
    background: Image.Image
    foreground: Image.Image


def discover_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def build_trials(
    background_paths: list[Path],
    foreground_paths: list[Path],
    rng: random.Random,
) -> list[ImageTrial]:
    trials = [
        ImageTrial(background, foreground)
        for background, foreground in product(
            background_paths, foreground_paths
        )
    ]
    rng.shuffle(trials)
    return trials


def prepare_trial_stimulus(
    trial: ImageTrial,
    config: ImageSessionConfig,
) -> PreparedImageStimulus:
    fg_size = geometry.get_size_for_visual_angle(
        config.distance_fg_cm, config.visual_angle_deg
    )
    bg_height = geometry.get_size_for_visual_angle(
        config.distance_bg_cm, config.visual_angle_deg
    )
    bg_width = geometry.get_size_for_visual_angle(
        config.distance_bg_cm, config.visual_angle_deg * 2.0
    )

    with Image.open(trial.background_path) as source:
        background = source.copy()
    with Image.open(trial.foreground_path) as source:
        foreground = source.copy()

    background = background.resize((512, 512))
    background = background.crop((0, 128, 512, 384))
    background = background.resize((bg_width, bg_height))

    foreground = foreground.resize((fg_size, fg_size))
    foreground = foreground.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    return PreparedImageStimulus(background, foreground)