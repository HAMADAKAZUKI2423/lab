"""Image実験の参加者、試行manifest、評価結果を保存する。
"Observed_Eye", "Reference_Eye", "Reproduced_Eye",
"Defocus_Mode", "Disparity_Mode",
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any
import csv
import json

from .conditions import (
    get_condition,
    resolve_observed_eye,
    resolve_reference_eye,
)
from .config import ImageSessionConfig, RUNTIME_CONFIG


PARTICIPANT_FIELDS = ["ID", "Age", "Gender", "IPD", "Dominance"]
RESULT_FIELDS = [
    "ID", "Age", "Gender", "IPD(mm)", "Dominance",
    "Trial_ID", "Block_ID", "Within_Block_Order",
    "Condition", "Condition_Label", "Plane_Mode", "Viewing_Mode",
    "Observed_Eye", "Reference_Eye", "Defocus_Mode", "Disparity_Mode",
    "Pair_ID", "Difference_Group", "Group_Rank", "Global_Rank",
    "Foreground_Texture_Class",
    "Image_Win1", "Image_Win2", "Background_Path", "Foreground_Path",
    "Distance_FG(cm)", "Distance_BG(cm)", "Visual_Angle(deg)",
    "PD_Right", "OffsetX_Right", "OffsetY_Right",
    "PD_Left", "OffsetX_Left", "OffsetY_Left",
    "Disparity_Px", "Disparity_Map_Path",
    "SinglePlane_OutOfGamut_Ratio",
    "Right_Visibility_Score", "Left_Visibility_Score",
    "Absolute_Score_Difference",
    "Source_Result_Dir", "Participant_Seed", "Score",
]
MANIFEST_FIELDS = [
    field for field in RESULT_FIELDS
    if field not in {
        "ID", "Age", "Gender", "IPD(mm)", "Dominance", "Score",
        "PD_Right", "OffsetX_Right", "OffsetY_Right",
        "PD_Left", "OffsetX_Left", "OffsetY_Left",
        "Disparity_Px", "SinglePlane_OutOfGamut_Ratio",
    }
]


def load_participant(
    path: Path, participant_id: str
) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("ID") == participant_id:
                return row
    return None


def save_participant(path: Path, participant: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
    normalized = {
        field: participant.get(field, "") for field in PARTICIPANT_FIELDS
    }
    for index, row in enumerate(rows):
        if row.get("ID") == participant["ID"]:
            rows[index] = normalized
            break
    else:
        rows.append(normalized)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=PARTICIPANT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _trial_metadata(
    trial,
    *,
    dominant_eye: str,
    selection_dir: Path,
    participant_seed: int,
) -> dict[str, Any]:
    spec = get_condition(trial.condition_id)
    pair = trial.pair
    return {
        "Trial_ID": trial.trial_order,
        "Block_ID": trial.block_id,
        "Within_Block_Order": trial.within_block_order,
        "Condition": spec.condition_id,
        "Condition_Label": spec.label,
        "Plane_Mode": spec.plane_mode,
        "Viewing_Mode": spec.viewing_mode,
        "Observed_Eye": resolve_observed_eye(spec, dominant_eye),
        "Reference_Eye": resolve_reference_eye(spec, dominant_eye),
        "Reproduced_Eye": spec.reproduced_eye,
        "Defocus_Mode": spec.defocus_mode,
        "Disparity_Mode": spec.disparity_mode,
        "Pair_ID": pair.pair_id,
        "Difference_Group": pair.difference_group,
        "Group_Rank": pair.group_rank,
        "Global_Rank": pair.global_rank,
        "Foreground_Texture_Class": pair.foreground_texture_class,
        "Image_Win1": pair.background_path.name,
        "Image_Win2": pair.foreground_path.name,
        "Background_Path": str(pair.background_path),
        "Foreground_Path": str(pair.foreground_path),
        "Disparity_Map_Path": "",
        "Right_Visibility_Score": pair.right_visibility_score,
        "Left_Visibility_Score": pair.left_visibility_score,
        "Absolute_Score_Difference": pair.absolute_score_difference,
        "Source_Result_Dir": str(selection_dir),
        "Participant_Seed": participant_seed,
    }


def build_result_row(app, score: int) -> dict[str, Any]:
    trial = app.current_trial
    row = _trial_metadata(
        trial,
        dominant_eye=app.participant_dominance.get(),
        selection_dir=app.selected_pairs_dir,
        participant_seed=app.participant_seed,
    )
    right = app.calib_results.get("Right", {})
    left = app.calib_results.get("Left", {})
    prepared_metadata = (
        app.current_prepared.metadata if app.current_prepared else {}
    )
    row.update({
        "ID": app.participant_id.get(),
        "Age": app.participant_age.get(),
        "Gender": app.participant_gender.get(),
        "IPD(mm)": app.participant_ipd.get(),
        "Dominance": app.participant_dominance.get(),
        "Distance_FG(cm)": app.session_config.distance_fg_cm,
        "Distance_BG(cm)": app.session_config.distance_bg_cm,
        "Visual_Angle(deg)": app.session_config.visual_angle_deg,
        "PD_Right": right.get("pd_mean"),
        "OffsetX_Right": right.get("offset_x"),
        "OffsetY_Right": right.get("offset_y"),
        "PD_Left": left.get("pd_mean"),
        "OffsetX_Left": left.get("offset_x"),
        "OffsetY_Left": left.get("offset_y"),
        "Reproduced_Eye": prepared_metadata.get(
            "Reproduced_Eye", row["Reproduced_Eye"]
        ),
        "Disparity_Px": prepared_metadata.get("Disparity_Px", 0),
        "Disparity_Map_Path": prepared_metadata.get(
            "Disparity_Map_Path", row["Disparity_Map_Path"]
        ),
        "SinglePlane_OutOfGamut_Ratio": prepared_metadata.get(
            "SinglePlane_OutOfGamut_Ratio", 0.0
        ),
        "Score": int(score),
    })
    return {field: row.get(field, "") for field in RESULT_FIELDS}


def save_trial_manifest(
    result_dir: Path,
    trials: list,
    *,
    config: ImageSessionConfig,
    dominant_eye: str,
    selection_dir: Path,
    participant_seed: int,
) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    path = result_dir / "trial_manifest.csv"
    rows = []
    for trial in trials:
        row = _trial_metadata(
            trial,
            dominant_eye=dominant_eye,
            selection_dir=selection_dir,
            participant_seed=participant_seed,
        )
        row.update({
            "Distance_FG(cm)": config.distance_fg_cm,
            "Distance_BG(cm)": config.distance_bg_cm,
            "Visual_Angle(deg)": config.visual_angle_deg,
        })
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in MANIFEST_FIELDS}
            for row in rows
        )
    return path


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def save_session_results(
    result_dir: Path,
    rows: list[dict[str, Any]],
    *,
    config: ImageSessionConfig,
    selection_dir: Path,
    participant_seed: int,
    condition_order: tuple[str, ...],
) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    output = result_dir / "image_evaluation.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    used_config = _jsonable(asdict(config))
    used_config.update({
        "runtime_config": RUNTIME_CONFIG,
        "selected_pairs_dir": str(selection_dir),
        "participant_seed": participant_seed,
        "condition_order": list(condition_order),
        "actual_trial_count": len(rows),
    })
    with (result_dir / "used_experiment_config.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(used_config, file, indent=2, ensure_ascii=False)
    return output