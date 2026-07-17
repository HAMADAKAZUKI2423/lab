"""参加者情報、試行結果、セッション設定の保存。"""

from pathlib import Path
import csv
import json
from typing import Any

from .config import MatchingSessionConfig, RUNTIME_CONFIG


PARTICIPANT_FIELDS = ["ID", "Age", "Gender", "IPD", "Dominance"]


def load_participant(path: Path, participant_id: str) -> dict[str, str] | None:
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
    replaced = False
    for index, row in enumerate(rows):
        if row.get("ID") == participant["ID"]:
            rows[index] = {field: participant.get(field, "") for field in PARTICIPANT_FIELDS}
            replaced = True
            break
    if not replaced:
        rows.append({field: participant.get(field, "") for field in PARTICIPANT_FIELDS})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=PARTICIPANT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_result_row(app, matched_contrast: float) -> dict[str, Any]:
    trial = app.trial_list[app.current_trial_in_block]
    right = app.calib_results.get(
        "Right", {"offset_x": 0, "offset_y": 0, "pd_mean": 0}
    )
    left = app.calib_results.get(
        "Left", {"offset_x": 0, "offset_y": 0, "pd_mean": 0}
    )
    return {
        "ID": app.participant_id.get(),
        "Age": app.participant_age.get(),
        "Gender": app.participant_gender.get(),
        "IPD(mm)": app.participant_ipd.get(),
        "Dominance": app.participant_dominance.get(),
        "Block_ID": app.current_block_index + 1,
        "Condition": app.current_block_cond["condition"],
        "Ocularity": app.current_block_cond["ocularity"],
        "Trial_ID": app.current_trial_in_experiment + 1,
        "Orientation": trial["orientation"],
        "Ref_Contrast": trial["ref_contrast"],
        "Matched_Contrast": round(float(matched_contrast), 4),
        "L_fg": app.session_config.l_fg,
        "L_bg": app.session_config.l_bg,
        "L_ref": app.session_config.l_ref,
        "Config_JSON": json.dumps(RUNTIME_CONFIG, ensure_ascii=False),
        "PD_Right": right["pd_mean"],
        "OffsetX_Right": right["offset_x"],
        "OffsetY_Right": right["offset_y"],
        "PD_Left": left["pd_mean"],
        "OffsetX_Left": left["offset_x"],
        "OffsetY_Left": left["offset_y"],
    }


def save_session_results(
    result_dir: Path,
    rows: list[dict[str, Any]],
    config: MatchingSessionConfig,
) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    output = result_dir / config.contrast_result_filename
    if rows:
        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (result_dir / "used_experiment_config.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(RUNTIME_CONFIG, file, indent=2, ensure_ascii=False)
    return output