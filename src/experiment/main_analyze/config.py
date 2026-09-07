"""contrast matching解析で共有する固定設定と出力パス規則。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import math
import re


PACKAGE_DIR = Path(__file__).resolve().parent
LAB_ROOT = PACKAGE_DIR.parents[2]

EXPERIMENT_RESULT_ROOT = (
    LAB_ROOT
    / "results"
    / "tables"
    / "main-experiment-matching"
    / "experiment"
)
ANALYSIS_TABLE_ROOT = (
    LAB_ROOT
    / "results"
    / "tables"
    / "main-experiment-matching"
    / "analysis"
    / "experiment"
)
EXPERIMENT_FIGURE_ROOT = (
    LAB_ROOT
    / "results"
    / "figures"
    / "main-experiment-matching"
    / "experiment"
)
TRAINING_FIGURE_ROOT = (
    LAB_ROOT
    / "results"
    / "figures"
    / "main-experiment-matching"
    / "training"
)
COMBINED_FIGURE_ROOT = (
    LAB_ROOT
    / "results"
    / "figures"
    / "main-experiment-matching"
    / "combined"
)

SESSION_RESULT_FILENAMES = {
    "experiment": "contrast_matching.csv",
    "training": "contrast_matching_training.csv",
}
DEFAULT_SESSION_TYPE = "experiment"

SP_CONDITION = "Single plane"
SPD_CONDITION = "Single plane + defocus simulation"
DP_CONDITION = "Dual plane"
DPF_CONDITION = "Dual plane flat"

CONDITION_ORDER = (
    SP_CONDITION,
    SPD_CONDITION,
    DP_CONDITION,
    DPF_CONDITION,
)
CORRECTED_CONDITIONS = (
    SP_CONDITION,
    SPD_CONDITION,
    DP_CONDITION,
)
OCULARITY_ORDER = ("monocular", "binocular")

# H1〜H4で実際に使用する入力列。年齢・優位眼・瞳孔径などは任意列として扱う。
REQUIRED_COLUMNS = (
    "ID",
    "Condition",
    "Ocularity",
    "Ref_Contrast",
    "Orientation",
    "Matched_Contrast",
    "L_fg",
    "L_bg",
    "L_ref",
)
OPTIONAL_METADATA_COLUMNS = (
    "Age",
    "Gender",
    "IPD(mm)",
    "Dominance",
    "PD_Right",
    "PD_Left",
    "Block_ID",
    "Trial_ID",
)

ANALYSIS_GROUP_COLUMNS = (
    "Session_Type",
    "Ref_Contrast",
    "Orientation",
)
DPF_MATCH_COLUMNS = (
    "ID",
    *ANALYSIS_GROUP_COLUMNS,
    "Ocularity",
)
PARTICIPANT_SUMMARY_KEY_COLUMNS = (
    *DPF_MATCH_COLUMNS,
    "Condition",
)

AR_VALUE_COLUMN = "AR_Matched_Contrast"
TRIAL_LOG10_COLUMN = "Log10_AR_Matched_Contrast"
TRIAL_COUNT_COLUMN = "Trial_Count"
MEAN_LOG10_COLUMN = "Mean_Log10_AR"
GEOMETRIC_MEAN_COLUMN = "Geometric_Mean_AR"

ANALYSIS_SCALE = "log10_AR_Matched_Contrast"
PARTICIPANT_AGGREGATION = "mean_of_log10_trials"
ALPHA = 0.05

# 通常の条件差とH4 interactionでは意味が異なるため、別々に固定する。
DEFAULT_EQUIVALENCE_RATIO_BOUND = 1.10
LOG10_EQUIVALENCE_MARGIN = math.log10(DEFAULT_EQUIVALENCE_RATIO_BOUND)
DEFAULT_INTERACTION_EQUIVALENCE_RATIO_BOUND = 1.10
LOG10_INTERACTION_EQUIVALENCE_MARGIN = math.log10(
    DEFAULT_INTERACTION_EQUIVALENCE_RATIO_BOUND
)

UNCORRECTED_SUMMARY_FILENAME = "participant_summary_uncorrected.csv"
CORRECTED_SUMMARY_FILENAME = "participant_summary_dpf_corrected.csv"
HYPOTHESIS_TESTS_FILENAME = "planned_hypothesis_tests.csv"
EXCLUSIONS_FILENAME = "analysis_exclusions.csv"
MANIFEST_FILENAME = "analysis_manifest.json"

UNCORRECTED_FIGURE_SUBDIR = "uncorrected"
CORRECTED_FIGURE_SUBDIR = "dpf_corrected"
H4_FIGURE_SUBDIR = "h4_interaction"

SESSION_DIR_PATTERN = re.compile(
    r"^(?P<participant_id>.+)_(?P<timestamp>\d{8}_\d{6})$"
)
_SAFE_RUN_COMPONENT = re.compile(r"[^0-9A-Za-z_-]+")


@dataclass(frozen=True)
class AnalysisOutputPaths:
    """1回の解析で共有する表・図の保存先。"""

    run_name: str
    table_dir: Path
    figure_dir: Path

    @property
    def uncorrected_figure_dir(self) -> Path:
        return self.figure_dir / UNCORRECTED_FIGURE_SUBDIR

    @property
    def corrected_figure_dir(self) -> Path:
        return self.figure_dir / CORRECTED_FIGURE_SUBDIR

    @property
    def h4_figure_dir(self) -> Path:
        return self.figure_dir / H4_FIGURE_SUBDIR


def _participant_sort_key(value: str) -> tuple[int, int | str]:
    normalized = value.strip()
    return (0, int(normalized)) if normalized.isdigit() else (1, normalized)


def sanitize_run_component(value: str) -> str:
    """パス要素として安全なASCII文字列へ正規化する。"""
    normalized = _SAFE_RUN_COMPONENT.sub("_", value.strip()).strip("_")
    return normalized or "unknown"


def build_run_name(
    *,
    all_participants: bool,
    participant_ids: Sequence[str],
    session_timestamps: Sequence[str],
) -> str:
    """全参加者または個別指定時のrun名を一意な規則で返す。"""
    if all_participants:
        return "all_participants"

    ids = sorted(
        {str(value).strip() for value in participant_ids if str(value).strip()},
        key=_participant_sort_key,
    )
    if not ids:
        raise ValueError("個別解析のrun名を作るには参加者IDが必要です")

    timestamps = sorted(
        {str(value).strip() for value in session_timestamps if str(value).strip()}
    )
    if not timestamps:
        raise ValueError(
            "個別解析のrun名を作るにはYYYYMMDD_HHMMSS形式の日時が必要です"
        )

    participant_label = sanitize_run_component("-".join(ids))
    return f"analyze_{participant_label}_{timestamps[-1]}"


def resolve_output_paths(
    run_name: str,
    *,
    table_output_dir: str | Path | None = None,
    figure_output_dir: str | Path | None = None,
    default_figure_root: str | Path = EXPERIMENT_FIGURE_ROOT,
) -> AnalysisOutputPaths:
    """表と図で同じrun名を使い、保存先を一度だけ解決する。"""
    safe_run_name = sanitize_run_component(run_name)
    table_dir = (
        Path(table_output_dir).expanduser().resolve()
        if table_output_dir is not None
        else ANALYSIS_TABLE_ROOT / safe_run_name
    )
    figure_dir = (
        Path(figure_output_dir).expanduser().resolve()
        if figure_output_dir is not None
        else Path(default_figure_root).expanduser().resolve() / safe_run_name
    )
    return AnalysisOutputPaths(
        run_name=safe_run_name,
        table_dir=table_dir,
        figure_dir=figure_dir,
    )


__all__ = [
    "ALPHA",
    "ANALYSIS_GROUP_COLUMNS",
    "ANALYSIS_SCALE",
    "ANALYSIS_TABLE_ROOT",
    "AR_VALUE_COLUMN",
    "AnalysisOutputPaths",
    "COMBINED_FIGURE_ROOT",
    "CONDITION_ORDER",
    "CORRECTED_CONDITIONS",
    "CORRECTED_FIGURE_SUBDIR",
    "CORRECTED_SUMMARY_FILENAME",
    "DEFAULT_EQUIVALENCE_RATIO_BOUND",
    "DEFAULT_INTERACTION_EQUIVALENCE_RATIO_BOUND",
    "DEFAULT_SESSION_TYPE",
    "DPF_CONDITION",
    "DPF_MATCH_COLUMNS",
    "EXCLUSIONS_FILENAME",
    "EXPERIMENT_FIGURE_ROOT",
    "EXPERIMENT_RESULT_ROOT",
    "GEOMETRIC_MEAN_COLUMN",
    "H4_FIGURE_SUBDIR",
    "HYPOTHESIS_TESTS_FILENAME",
    "LAB_ROOT",
    "LOG10_EQUIVALENCE_MARGIN",
    "LOG10_INTERACTION_EQUIVALENCE_MARGIN",
    "MANIFEST_FILENAME",
    "MEAN_LOG10_COLUMN",
    "OCULARITY_ORDER",
    "OPTIONAL_METADATA_COLUMNS",
    "PARTICIPANT_AGGREGATION",
    "PARTICIPANT_SUMMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
    "SESSION_DIR_PATTERN",
    "SESSION_RESULT_FILENAMES",
    "SP_CONDITION",
    "SPD_CONDITION",
    "DP_CONDITION",
    "TRAINING_FIGURE_ROOT",
    "TRIAL_COUNT_COLUMN",
    "TRIAL_LOG10_COLUMN",
    "UNCORRECTED_FIGURE_SUBDIR",
    "UNCORRECTED_SUMMARY_FILENAME",
    "build_run_name",
    "resolve_output_paths",
    "sanitize_run_component",
]