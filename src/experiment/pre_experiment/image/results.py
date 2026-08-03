"""Image実験の参加者情報と試行結果を保存する。"""

from pathlib import Path
from typing import Any
import csv


PARTICIPANT_FIELDS = ["ID", "Age", "Gender", "IPD", "Dominance"]
RESULT_FIELDS = [
    "ID", "Age", "Gender", "IPD(mm)", "Dominance",
    "Distance_FG(cm)", "Distance_BG(cm)",
    "PD_Right", "OffsetX_Right", "OffsetY_Right",
    "PD_Left", "OffsetX_Left", "OffsetY_Left",
    "Trial_ID", "Block_ID", "Condition", "Image_Win1", "Image_Win2", "Score",
    "Defocus_Difference(D)", "Disparity_Total(px)",
    "Background_Left_Weight", "Background_Right_Weight",
    "Out_Of_Gamut_Ratio",
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


def build_result_row(app, score: int) -> dict[str, Any]:
    trial = app.current_trial
    right = app.calib_results.get("Right", {})
    left = app.calib_results.get("Left", {})
    return {
        "ID": app.participant_id.get(),
        "Age": app.participant_age.get(),
        "Gender": app.participant_gender.get(),
        "IPD(mm)": app.participant_ipd.get(),
        "Dominance": app.participant_dominance.get(),
        "Distance_FG(cm)": app.session_config.distance_fg_cm,
        "Distance_BG(cm)": app.session_config.distance_bg_cm,
        "PD_Right": right.get("pd_mean"),
        "OffsetX_Right": right.get("offset_x"),
        "OffsetY_Right": right.get("offset_y"),
        "PD_Left": left.get("pd_mean"),
        "OffsetX_Left": left.get("offset_x"),
        "OffsetY_Left": left.get("offset_y"),
        "Trial_ID": app.current_trial_index + 1,
        "Block_ID": app.current_block_index + 1,
        "Condition": trial.condition,
        "Image_Win1": trial.background_path.name,
        "Image_Win2": trial.foreground_path.name,
        "Score": int(score),
        "Defocus_Difference(D)": app.prepared_stimulus.defocus_difference_d,
        "Disparity_Total(px)": app.prepared_stimulus.disparity_total_px,
        "Background_Left_Weight": 0.5,
        "Background_Right_Weight": 0.5,
        "Out_Of_Gamut_Ratio": app.prepared_stimulus.out_of_gamut_ratio,
    }


def save_session_results(
    result_dir: Path, rows: list[dict[str, Any]]
) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    output = result_dir / "image_evaluation.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output