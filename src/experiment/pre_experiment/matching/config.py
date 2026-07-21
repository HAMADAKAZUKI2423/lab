"""本実験とtrainingの条件・保存先を一元管理する。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
LAB_ROOT = EXPERIMENT_DIR.parents[1]

try:
    from experiment import experiment_config
    RUNTIME_CONFIG: dict[str, Any] = experiment_config.get_config() or {}
except Exception as exc:
    print(f"WARN: experiment_config could not be loaded: {exc}")
    RUNTIME_CONFIG = {}


@dataclass(frozen=True)
class MatchingSessionConfig:
    session_type: str
    conditions: tuple[str, ...]
    ocularities: tuple[str, ...]
    ref_contrasts: tuple[float, ...]
    orientations: tuple[int, ...]
    repetitions: int
    defocus_patterns: tuple[str, ...]
    defocus_cpds: tuple[int, ...]
    save_preview: bool
    result_root: Path
    figure_root: Path
    participant_data_dir: Path
    display_dir: Path
    spatial_frequency: float = 4.0
    visual_angle_width_deg: float = 7.9
    visual_angle_height_deg: float = 3.95
    win2_total_width_factor: float = 2.6
    asym_width_factor_large: float = 1.3
    asym_width_factor_small: float = 0.7
    initial_pupil_diameter_mm: float = 4.0
    slider_step: float = 0.005
    background_color: str = "black"
    l_fg: float = 15.0
    l_bg: float = 15.0
    l_ref: float = 30.0
    distance_fg_cm: int = 50
    distance_bg_cm: int = 150

    @property
    def contrast_result_filename(self) -> str:
        return (
            "contrast_matching_training.csv"
            if self.session_type == "training"
            else "contrast_matching.csv"
        )

    @property
    def defocus_result_filename(self) -> str:
        return (
            "defocus_matching_training.csv"
            if self.session_type == "training"
            else "defocus_matching.csv"
        )


def _base_config(session_type: str, **overrides: Any) -> MatchingSessionConfig:
    result_root = (
        LAB_ROOT / "results" / "tables" / "pre-experiment-matching" / session_type
    )
    figure_root = (
        LAB_ROOT / "results" / "figures" / "pre-experiment-matching" / session_type
    )
    participant_dir = (
        LAB_ROOT / "data" / "processed" / "tables" / "pre-experiment-matching"
    )
    values: dict[str, Any] = {
        "session_type": session_type,
        "conditions": (),
        "ocularities": ("binocular", "monocular"),
        "ref_contrasts": (),
        "orientations": (0,),
        "repetitions": 5,
        "defocus_patterns": (
            "checker", "checker_45", "stripe", "border", "noise"
        ),
        "defocus_cpds": (2, 4),
        "save_preview": True,
        "result_root": result_root,
        "figure_root": figure_root,
        "participant_data_dir": participant_dir,
        "display_dir": LAB_ROOT / "results" / "tables" / "DisplayBrightness",
        "background_color": str(RUNTIME_CONFIG.get("BG_COLOR", "black")),
        "l_fg": float(RUNTIME_CONFIG.get("L_fg", 15.0)),
        "l_bg": float(RUNTIME_CONFIG.get("L_bg", 15.0)),
        "l_ref": float(RUNTIME_CONFIG.get("L_ref", 30.0)),
        "distance_fg_cm": int(RUNTIME_CONFIG.get("DISTANCE_FG", 50)),
        "distance_bg_cm": int(RUNTIME_CONFIG.get("DISTANCE_BG", 150)),
    }
    values.update(overrides)
    config = MatchingSessionConfig(**values)
    for path in (config.result_root, config.figure_root, config.participant_data_dir):
        path.mkdir(parents=True, exist_ok=True)
    return config


def create_experiment_config() -> MatchingSessionConfig:
    return _base_config(
        "experiment",
        conditions=(
            "Single plane",
            "Single plane + defocus simulation",
            "Dual plane",
            "Dual plane flat",
        ),
        ref_contrasts=(0.1, 0.2),
        defocus_cpds=(2, 4),
        save_preview=True,
    )


def create_training_config() -> MatchingSessionConfig:
    return _base_config(
        "training",
        conditions=("Single plane", "Dual plane flat"),
        ref_contrasts=(0.1, 0.2),
        defocus_cpds=(4,),
        save_preview=False,
    )