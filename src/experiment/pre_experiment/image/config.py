"""Image evaluation予備実験の5条件・入力・保存先を一元管理する。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiment import experiment_config

from .conditions import DEFAULT_CONDITION_IDS


EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
LAB_ROOT = EXPERIMENT_DIR.parents[1]
RUNTIME_CONFIG: dict[str, Any] = experiment_config.get_config() or {}


@dataclass(frozen=True)
class ImageSessionConfig:
    result_root: Path
    participant_data_dir: Path
    selected_pairs_root: Path
    selected_pairs_dir: Path | None
    display_dir: Path
    conditions: tuple[str, ...] = DEFAULT_CONDITION_IDS
    random_seed: int = 20260928
    visual_angle_deg: float = 7.9
    background_width_factor: float = 2.0
    trials_per_condition: int = 20
    trials_before_break: int = 10
    time_foreground_only_ms: int = 500
    time_isi_ms: int = 1000
    time_both_ms: int = 500
    distance_fg_cm: float = 50.0
    distance_bg_cm: float = 125.0
    background_color: str = "black"
    initial_pupil_diameter_mm: float = 4.0
    defocus_cpd: float = 4.0
    defocus_repetitions: int = 5
    binocular_fusion_dominant_weight: float = 0.5

    @property
    def defocus_result_filename(self) -> str:
        return "defocus_matching.csv"

    @property
    def expected_trial_count(self) -> int:
        return len(self.conditions) * self.trials_per_condition

    @property
    def background_visual_angle_width_deg(self) -> float:
        """DP・SPで共通化する背景の水平視角を返す。"""
        return self.visual_angle_deg * self.background_width_factor


def create_image_config(
    selected_pairs_dir: Path | None = None,
    random_seed: int | None = None,
) -> ImageSessionConfig:
    config = ImageSessionConfig(
        result_root=(
            LAB_ROOT / "results" / "tables" / "pre-experiment-image"
        ),
        participant_data_dir=(
            LAB_ROOT / "data" / "processed" / "tables"
            / "pre-experiment-image"
        ),
        selected_pairs_root=(
            LAB_ROOT / "results" / "fukiage-disparity-selection"
        ),
        selected_pairs_dir=(
            selected_pairs_dir.expanduser().resolve()
            if selected_pairs_dir is not None else None
        ),
        display_dir=(
            LAB_ROOT / "results" / "tables" / "DisplayBrightness"
        ),
        random_seed=(
            int(random_seed)
            if random_seed is not None
            else int(RUNTIME_CONFIG.get("IMAGE_EXPERIMENT_SEED", 20260928))
        ),
        visual_angle_deg=float(RUNTIME_CONFIG.get("VISUAL_ANGLE_DEG", 7.9)),
        background_width_factor=float(
            RUNTIME_CONFIG.get("IMAGE_BACKGROUND_WIDTH_FACTOR", 2.0)
        ),
        distance_fg_cm=float(RUNTIME_CONFIG.get("DISTANCE_FG", 50.0)),
        distance_bg_cm=float(RUNTIME_CONFIG.get("DISTANCE_BG", 125.0)),
        background_color=str(RUNTIME_CONFIG.get("BG_COLOR", "black")),
    )
    if not 0 < config.distance_fg_cm < config.distance_bg_cm:
        raise ValueError("DISTANCE_FG must be positive and smaller than DISTANCE_BG")
    if config.trials_per_condition != 20:
        raise ValueError("This experiment requires 20 selected pairs per condition")
    if config.background_width_factor <= 0:
        raise ValueError("background_width_factor must be positive")
    if len(config.conditions) != 5:
        raise ValueError("This experiment temporarily requires exactly five conditions")
    if not 0.0 <= config.binocular_fusion_dominant_weight <= 1.0:
        raise ValueError("binocular_fusion_dominant_weight must be in [0, 1]")
    config.result_root.mkdir(parents=True, exist_ok=True)
    config.participant_data_dir.mkdir(parents=True, exist_ok=True)
    config.selected_pairs_root.mkdir(parents=True, exist_ok=True)
    return config