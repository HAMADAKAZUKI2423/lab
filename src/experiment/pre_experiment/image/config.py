"""Image evaluation予備実験の条件と保存先を一元管理する。"""

from dataclasses import dataclass
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
LAB_ROOT = EXPERIMENT_DIR.parents[1]


@dataclass(frozen=True)
class ImageSessionConfig:
    result_root: Path
    participant_data_dir: Path
    background_image_dir: Path
    foreground_image_dir: Path
    display_dir: Path
    visual_angle_deg: float = 7.9
    trials_before_break: int = 38
    time_foreground_only_ms: int = 500
    time_isi_ms: int = 1000
    time_both_ms: int = 500
    distance_fg_cm: int = 50
    distance_bg_cm: int = 150
    background_color: str = "black"
    initial_pupil_diameter_mm: float = 4.0
    defocus_cpd: float = 4.0
    defocus_repetitions: int = 5


def create_image_config() -> ImageSessionConfig:
    config = ImageSessionConfig(
        result_root=(
            LAB_ROOT / "results" / "tables" / "pre-experiment-image"
        ),
        participant_data_dir=(
            LAB_ROOT / "data" / "processed" / "tables"
            / "pre-experiment-image"
        ),
        background_image_dir=(
            LAB_ROOT / "data" / "processed" / "images"
            / "pre-experiment-image" / "bg_imgs"
        ),
        foreground_image_dir=(
            LAB_ROOT / "data" / "processed" / "images"
            / "pre-experiment-image" / "fg_imgs"
        ),
        display_dir=(
            LAB_ROOT / "results" / "tables" / "DisplayBrightness"
        ),
    )
    config.result_root.mkdir(parents=True, exist_ok=True)
    config.participant_data_dir.mkdir(parents=True, exist_ok=True)
    return config