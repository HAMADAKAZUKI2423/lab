"""Image evaluation予備実験の有効条件・入力・保存先を一元管理する。"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import math

from experiment import experiment_config

from .conditions import ALL_CONDITION_SPECS, get_condition, get_enabled_condition_ids


EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
LAB_ROOT = EXPERIMENT_DIR.parents[1]
RUNTIME_CONFIG: dict[str, Any] = experiment_config.get_config() or {}

# 条件T/Fとは独立。旧ratingへ戻す場合は "rating"、または --task-mode rating。
IMAGE_TASK_MODE = "matching"


@dataclass(frozen=True)
class ImageSessionConfig:
    result_root: Path
    participant_data_dir: Path
    selected_pairs_root: Path
    selected_pairs_dir: Path | None
    display_dir: Path
    conditions: tuple[str, ...] = field(default_factory=get_enabled_condition_ids)
    task_mode: str = IMAGE_TASK_MODE
    reference_gain: float = 0.5
    gain_step: float = 0.01
    gain_fine_step: float = 0.001
    render_interval_ms: int = 33
    foreground_pixels_per_cm: float = 1.0 / 0.02331
    background_pixels_per_cm: float = 1.0 / 0.02331
    gamut_tolerance: float = 1e-9
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

    def __post_init__(self) -> None:
        if self.task_mode not in {"matching", "rating"}:
            raise ValueError("task_mode must be matching or rating")
        for value in (self.visual_angle_deg, self.distance_fg_cm, self.distance_bg_cm,
                      self.foreground_pixels_per_cm, self.background_pixels_per_cm,
                      self.gain_step, self.gain_fine_step):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Angles, distances, pixel density and steps must be finite and positive")
        if not 0 < self.distance_fg_cm < self.distance_bg_cm:
            raise ValueError("DISTANCE_FG must be smaller than DISTANCE_BG")
        if self.visual_angle_deg >= 80 or not 0 < self.reference_gain < 1:
            raise ValueError("Invalid visual angle or reference gain")
        if (not math.isfinite(self.reference_gain) or not math.isfinite(self.gamut_tolerance)
                or self.render_interval_ms < 1 or self.gamut_tolerance < 0):
            raise ValueError("Invalid update interval or gamut tolerance")
        if self.task_mode == "matching" and self.background_color.lower() != "black":
            raise ValueError("Matching requires black blank regions")
        # 有効条件をセッション生成時にtupleへ固定し、開始後の変更を防ぐ。
        object.__setattr__(self, "conditions", tuple(self.conditions))
        if not self.conditions:
            raise ValueError("At least one image condition must be enabled")
        if len(set(self.conditions)) != len(self.conditions):
            raise ValueError("Image condition IDs must be unique")
        for condition_id in self.conditions:
            get_condition(condition_id)

    @property
    def condition_count(self) -> int:
        return len(self.conditions)

    @property
    def condition_enabled(self) -> dict[str, bool]:
        """グローバル変数ではなく、実際のセッション条件から保存用T/Fを生成。"""
        enabled = set(self.conditions)
        return {
            spec.condition_id: spec.condition_id in enabled
            for spec in ALL_CONDITION_SPECS
        }

    @property
    def experiment_start_message(self) -> str:
        labels = "\n".join(
            f"{index}. {get_condition(condition_id).label}"
            for index, condition_id in enumerate(self.conditions, start=1)
        )
        return (
            f"Image {self.task_mode}: {self.condition_count} conditions\n"
            f"{self.expected_trial_count} trials: "
            f"{self.trials_per_condition} selected pairs per condition.\n\n"
            f"{labels}\n\nPress Enter to start."
        )

    @property
    def reference_center_deg(self) -> float:
        return self.visual_angle_deg / 2.0

    @property
    def test_center_deg(self) -> float:
        return -self.visual_angle_deg / 2.0

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
    task_mode: str | None = None,
) -> ImageSessionConfig:
    mode = IMAGE_TASK_MODE if task_mode is None else task_mode
    config = ImageSessionConfig(
        task_mode=mode,
        foreground_pixels_per_cm=float(RUNTIME_CONFIG.get("IMAGE_FG_PIXELS_PER_CM", 1.0 / 0.02331)),
        background_pixels_per_cm=float(RUNTIME_CONFIG.get("IMAGE_BG_PIXELS_PER_CM", 1.0 / 0.02331)),
        result_root=(
            LAB_ROOT / "results" / "tables" / (
                "pre-experiment-image-matching" if mode == "matching"
                else "pre-experiment-image"
            )
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
    if not config.conditions:
        raise ValueError("At least one image condition must be enabled")
    if len(set(config.conditions)) != len(config.conditions):
        raise ValueError("Image condition IDs must be unique")
    for condition_id in config.conditions:
        get_condition(condition_id)
    if not 0.0 <= config.binocular_fusion_dominant_weight <= 1.0:
        raise ValueError("binocular_fusion_dominant_weight must be in [0, 1]")
    config.result_root.mkdir(parents=True, exist_ok=True)
    config.participant_data_dir.mkdir(parents=True, exist_ok=True)
    config.selected_pairs_root.mkdir(parents=True, exist_ok=True)
    return config