"""全参加者のdefocus matchingとDP眼条件差を要約・可視化する。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import re
from uuid import uuid4

import numpy as np
import pandas as pd

from .config import (
    ANALYSIS_GROUP_COLUMNS,
    DP_CONDITION,
    OCULARITY_ORDER,
    PARTICIPANT_AGGREGATION,
    SESSION_DIR_PATTERN,
)
from .dpf_correction import CORRECTED_AR_COLUMN, CORRECTED_LOG10_COLUMN
from .hypothesis_tests import holm_adjust


DEFOCUS_RESULT_FILENAME = "defocus_matching.csv"
CONTRAST_RESULT_FILENAME = "contrast_matching.csv"
DEFOCUS_SUMMARY_FILENAME = "defocus_participant_summary.csv"
DEFOCUS_DP_PAIRS_FILENAME = "defocus_dp_ocularity_participant_pairs.csv"
DEFOCUS_DP_CORRELATION_FILENAME = "defocus_dp_ocularity_correlation.csv"
DEFOCUS_FIGURE_FILENAME = "defocus_matching_by_participant.png"
DEFOCUS_DP_CORRELATION_FIGURE_PREFIX = (
    "defocus_dp_ocularity_correlation"
)
_DEFOCUS_CORRELATION_X_COLUMN = "Defocus_Dom_Minus_NonDom_mm"
_DEFOCUS_CORRELATION_Y_COLUMN = (
    "DP_Monocular_Minus_Binocular_Log10"
)

_REQUIRED_DEFOCUS_COLUMNS = (
    "ID",
    "Eye",
    "Trial",
    "Spatial_Freq(cpd)",
    "Matched_PD(mm)",
)
_REQUIRED_CONTRAST_COLUMNS = ("ID", "Dominance")
_EYE_ORDER = ("right", "left")
_EYE_LABELS = {"right": "Right", "left": "Left"}
_MIN_PUPIL_DIAMETER_MM = 1.0
_MAX_PUPIL_DIAMETER_MM = 6.0
_DPI = 180
_NUMERIC_ID_PATTERN = re.compile(r"^([+-]?\d+)\.0+$")


class DefocusAnalysisError(ValueError):
    """Defocus matchingの読込・集約・描画に使えない入力への例外。"""


@dataclass(frozen=True)
class DefocusAnalysisResult:
    """Defocus matchingの要約表と保存済みファイル。"""

    participant_summary: pd.DataFrame
    participant_pairs: pd.DataFrame
    correlation_summary: pd.DataFrame
    summary_file: Path
    participant_pairs_file: Path
    correlation_file: Path
    figure_file: Path | None
    correlation_figure_files: tuple[Path, ...]


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _normalize_identifier(value: object) -> str:
    if pd.isna(value):
        return ""
    normalized = str(value).strip()
    match = _NUMERIC_ID_PATTERN.fullmatch(normalized)
    return match.group(1) if match else normalized


def _participant_sort_key(value: str) -> tuple[int, int | str]:
    normalized = str(value).strip()
    return (0, int(normalized)) if normalized.isdigit() else (1, normalized)


def _session_metadata(session_dir: Path) -> tuple[str, str]:
    match = SESSION_DIR_PATTERN.fullmatch(session_dir.name)
    if match is None:
        raise DefocusAnalysisError(
            "セッションフォルダ名はID_YYYYMMDD_HHMMSS形式にしてください: "
            f"{session_dir.name}"
        )
    participant_id = _normalize_identifier(match.group("participant_id"))
    if not participant_id:
        raise DefocusAnalysisError(
            f"セッションフォルダから参加者IDを取得できません: {session_dir}"
        )
    return participant_id, match.group("timestamp")


def _read_csv(path: Path, *, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label}が見つかりません: {path}")
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=True,
    )
    if frame.empty:
        raise DefocusAnalysisError(f"空の{label}は使用できません: {path}")
    return frame


def _single_identifier(
    frame: pd.DataFrame,
    *,
    path: Path,
    expected_id: str,
) -> pd.Series:
    if "ID" not in frame.columns:
        raise DefocusAnalysisError(f"{path}: ID列がありません")
    identifiers = frame["ID"].map(_normalize_identifier)
    if identifiers.eq("").any():
        rows = (identifiers.index[identifiers.eq("")] + 2).tolist()[:10]
        raise DefocusAnalysisError(f"{path}: IDが空の行があります: {rows}")
    found = sorted(set(identifiers), key=_participant_sort_key)
    if found != [expected_id]:
        raise DefocusAnalysisError(
            "フォルダIDとCSV内IDが一致しません: "
            f"folder={expected_id}, csv={found}, path={path}"
        )
    return identifiers


def _load_dominance(
    contrast_path: Path,
    *,
    participant_id: str,
) -> str:
    frame = _read_csv(contrast_path, label=CONTRAST_RESULT_FILENAME)
    missing = [
        column for column in _REQUIRED_CONTRAST_COLUMNS
        if column not in frame.columns
    ]
    if missing:
        raise DefocusAnalysisError(
            f"{contrast_path}: 必須列が不足しています: {missing}"
        )
    frame["ID"] = _single_identifier(
        frame,
        path=contrast_path,
        expected_id=participant_id,
    )
    values = (
        frame["Dominance"]
        .astype("string")
        .str.strip()
        .str.lower()
    )
    invalid = values.isna() | ~values.isin(_EYE_ORDER)
    if invalid.any():
        rows = (values.index[invalid] + 2).tolist()[:10]
        raise DefocusAnalysisError(
            f"{contrast_path}: DominanceはLeftまたはRightにしてください: {rows}"
        )
    unique = sorted(set(values.astype(str)))
    if len(unique) != 1:
        raise DefocusAnalysisError(
            f"{contrast_path}: Dominanceが参加者内で一意ではありません: {unique}"
        )
    return unique[0]


def _load_defocus_session(session_dir: Path) -> pd.DataFrame:
    session_dir = Path(session_dir).expanduser().resolve()
    participant_id, timestamp = _session_metadata(session_dir)
    contrast_path = session_dir / CONTRAST_RESULT_FILENAME
    defocus_path = session_dir / DEFOCUS_RESULT_FILENAME
    dominance = _load_dominance(
        contrast_path,
        participant_id=participant_id,
    )
    frame = _read_csv(defocus_path, label=DEFOCUS_RESULT_FILENAME)
    missing = [
        column for column in _REQUIRED_DEFOCUS_COLUMNS
        if column not in frame.columns
    ]
    if missing:
        raise DefocusAnalysisError(
            f"{defocus_path}: 必須列が不足しています: {missing}"
        )

    frame = frame.copy()
    frame["ID"] = _single_identifier(
        frame,
        path=defocus_path,
        expected_id=participant_id,
    )
    eyes = frame["Eye"].astype("string").str.strip().str.lower()
    invalid_eyes = eyes.isna() | ~eyes.isin(_EYE_ORDER)
    if invalid_eyes.any():
        rows = (eyes.index[invalid_eyes] + 2).tolist()[:10]
        raise DefocusAnalysisError(
            f"{defocus_path}: EyeはLeftまたはRightにしてください: {rows}"
        )
    frame["Eye"] = eyes
    if set(frame["Eye"].astype(str)) != set(_EYE_ORDER):
        raise DefocusAnalysisError(
            f"{defocus_path}: Right・Left両眼の試行が必要です"
        )

    for column in ("Trial", "Spatial_Freq(cpd)", "Matched_PD(mm)"):
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = values.isna() | ~np.isfinite(values.to_numpy(dtype=float))
        if invalid.any():
            rows = (values.index[invalid] + 2).tolist()[:10]
            raise DefocusAnalysisError(
                f"{defocus_path}: {column}を有限な数値へ変換できません: {rows}"
            )
        frame[column] = values.astype(float)

    trials = frame["Trial"].to_numpy(dtype=float)
    invalid_trials = (trials <= 0) | ~np.isclose(trials, np.round(trials))
    if invalid_trials.any():
        rows = (frame.index[invalid_trials] + 2).tolist()[:10]
        raise DefocusAnalysisError(
            f"{defocus_path}: Trialは正の整数にしてください: {rows}"
        )
    frame["Trial"] = np.round(trials).astype(int)
    duplicate_trials = frame.duplicated(["Eye", "Trial"], keep=False)
    if duplicate_trials.any():
        duplicate = frame.loc[duplicate_trials, ["Eye", "Trial"]]
        raise DefocusAnalysisError(
            f"{defocus_path}: Eye×Trialが重複しています: "
            f"{duplicate.head(10).to_dict('records')}"
        )

    pupil = frame["Matched_PD(mm)"].to_numpy(dtype=float)
    invalid_pupil = (
        (pupil < _MIN_PUPIL_DIAMETER_MM)
        | (pupil > _MAX_PUPIL_DIAMETER_MM)
    )
    if invalid_pupil.any():
        rows = (frame.index[invalid_pupil] + 2).tolist()[:10]
        raise DefocusAnalysisError(
            f"{defocus_path}: Matched_PD(mm)は1〜6 mmにしてください: {rows}"
        )

    frequencies = sorted(set(frame["Spatial_Freq(cpd)"].astype(float)))
    if len(frequencies) != 1 or frequencies[0] <= 0:
        raise DefocusAnalysisError(
            f"{defocus_path}: Spatial_Freq(cpd)は1種類の正値にしてください: "
            f"{frequencies}"
        )

    frame["Dominance"] = dominance
    frame["Eye_Role"] = np.where(
        frame["Eye"].eq(dominance),
        "dominant",
        "non_dominant",
    )
    frame["Session_Timestamp"] = timestamp
    frame["Session_Dir"] = str(session_dir)
    frame["Source_CSV"] = str(defocus_path)
    return frame.loc[
        :,
        [
            "ID",
            "Dominance",
            "Eye",
            "Eye_Role",
            "Trial",
            "Spatial_Freq(cpd)",
            "Matched_PD(mm)",
            "Session_Timestamp",
            "Session_Dir",
            "Source_CSV",
        ],
    ]


def load_defocus_trials(
    session_dirs: Sequence[str | Path],
) -> pd.DataFrame:
    """全参加者セッションのdefocus matching試行を検証して結合する。"""
    directories = [Path(path).expanduser().resolve() for path in session_dirs]
    if not directories:
        raise FileNotFoundError("Defocus matchingの対象セッションがありません")
    if len(set(directories)) != len(directories):
        raise DefocusAnalysisError("同じセッションが複数回指定されています")

    frames = [_load_defocus_session(directory) for directory in directories]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    duplicated_ids = (
        combined.loc[:, ["ID", "Session_Dir"]]
        .drop_duplicates()
        .duplicated("ID", keep=False)
    )
    if duplicated_ids.any():
        repeated = (
            combined.loc[:, ["ID", "Session_Dir"]]
            .drop_duplicates()
            .loc[duplicated_ids]
            .to_dict("records")
        )
        raise DefocusAnalysisError(
            f"同一参加者の複数セッションが含まれています: {repeated}"
        )
    return combined


def build_defocus_participant_summary(
    trials: pd.DataFrame,
) -> pd.DataFrame:
    """参加者ごとに優位眼・非優位眼の算術平均、SD、試行数を返す。"""
    required = {
        "ID",
        "Dominance",
        "Eye",
        "Eye_Role",
        "Trial",
        "Spatial_Freq(cpd)",
        "Matched_PD(mm)",
        "Session_Timestamp",
        "Session_Dir",
        "Source_CSV",
    }
    missing = sorted(required - set(trials.columns))
    if missing:
        raise DefocusAnalysisError(
            f"参加者要約に必要な列が不足しています: {missing}"
        )
    if trials.empty:
        raise DefocusAnalysisError("Defocus matching試行が空です")

    long_summary = (
        trials.groupby(
            [
                "ID",
                "Dominance",
                "Eye",
                "Eye_Role",
                "Spatial_Freq(cpd)",
                "Session_Timestamp",
                "Session_Dir",
                "Source_CSV",
            ],
            as_index=False,
            sort=True,
            dropna=False,
        )
        .agg(
            Trial_Count=("Matched_PD(mm)", "size"),
            Mean_Matched_PD_mm=("Matched_PD(mm)", "mean"),
            SD_Matched_PD_mm=("Matched_PD(mm)", "std"),
        )
    )

    rows: list[dict[str, object]] = []
    for participant_id, group in long_summary.groupby("ID", sort=False):
        if set(group["Eye_Role"].astype(str)) != {"dominant", "non_dominant"}:
            raise DefocusAnalysisError(
                f"参加者{participant_id}の優位眼・非優位眼対応が不完全です"
            )
        if group["Eye_Role"].duplicated().any():
            raise DefocusAnalysisError(
                f"参加者{participant_id}の眼役割が重複しています"
            )
        dominant = group.loc[group["Eye_Role"] == "dominant"].iloc[0]
        non_dominant = group.loc[
            group["Eye_Role"] == "non_dominant"
        ].iloc[0]
        rows.append(
            {
                "ID": str(participant_id),
                "Dominance": _EYE_LABELS[str(dominant["Eye"])],
                "Spatial_Freq_cpd": float(dominant["Spatial_Freq(cpd)"]),
                "Dominant_Eye": _EYE_LABELS[str(dominant["Eye"])],
                "Dominant_PD_Mean_mm": float(dominant["Mean_Matched_PD_mm"]),
                "Dominant_PD_SD_mm": float(dominant["SD_Matched_PD_mm"]),
                "Dominant_Trial_Count": int(dominant["Trial_Count"]),
                "Non_Dominant_Eye": _EYE_LABELS[str(non_dominant["Eye"])],
                "Non_Dominant_PD_Mean_mm": float(
                    non_dominant["Mean_Matched_PD_mm"]
                ),
                "Non_Dominant_PD_SD_mm": float(
                    non_dominant["SD_Matched_PD_mm"]
                ),
                "Non_Dominant_Trial_Count": int(
                    non_dominant["Trial_Count"]
                ),
                "Session_Timestamp": str(dominant["Session_Timestamp"]),
                "Session_Dir": str(dominant["Session_Dir"]),
                "Source_CSV": str(dominant["Source_CSV"]),
            }
        )

    rows.sort(key=lambda row: _participant_sort_key(str(row["ID"])))
    return pd.DataFrame(rows).reset_index(drop=True)


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    label: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise DefocusAnalysisError(f"{label}に必要な列が不足しています: {missing}")
    if frame.empty:
        raise DefocusAnalysisError(f"{label}が空です")


def build_defocus_dp_participant_pairs(
    defocus_summary: pd.DataFrame,
    corrected_summary: pd.DataFrame,
) -> pd.DataFrame:
    """参加者ごとのdefocus眼差とDPのmonocular－binocular差を返す。"""
    defocus_columns = [
        "ID", "Dominance", "Spatial_Freq_cpd",
        "Dominant_Eye", "Dominant_PD_Mean_mm", "Dominant_PD_SD_mm",
        "Dominant_Trial_Count", "Non_Dominant_Eye",
        "Non_Dominant_PD_Mean_mm", "Non_Dominant_PD_SD_mm",
        "Non_Dominant_Trial_Count", "Session_Timestamp", "Session_Dir",
        "Source_CSV",
    ]
    corrected_columns = [
        "ID", *ANALYSIS_GROUP_COLUMNS, "Ocularity", "Condition",
        "Participant_Aggregation", CORRECTED_LOG10_COLUMN,
        CORRECTED_AR_COLUMN,
    ]
    _require_columns(defocus_summary, defocus_columns, label="defocus参加者要約")
    _require_columns(corrected_summary, corrected_columns, label="DPF補正後参加者要約")

    defocus = defocus_summary.loc[:, defocus_columns].copy()
    defocus["ID"] = defocus["ID"].map(_normalize_identifier)
    if defocus["ID"].eq("").any() or defocus["ID"].duplicated().any():
        raise DefocusAnalysisError("defocus参加者要約のIDが空または重複しています")
    for column in ("Dominant_PD_Mean_mm", "Non_Dominant_PD_Mean_mm"):
        defocus[column] = pd.to_numeric(defocus[column], errors="coerce")
    if not np.isfinite(
        defocus[["Dominant_PD_Mean_mm", "Non_Dominant_PD_Mean_mm"]]
        .to_numpy(dtype=float)
    ).all():
        raise DefocusAnalysisError("defocus参加者要約の平均瞳孔径が有限値ではありません")
    defocus["Defocus_Dom_Minus_NonDom_mm"] = (
        defocus["Dominant_PD_Mean_mm"]
        - defocus["Non_Dominant_PD_Mean_mm"]
    )

    dp = corrected_summary.loc[
        corrected_summary["Condition"].astype(str).str.strip().eq(DP_CONDITION),
        corrected_columns,
    ].copy()
    if dp.empty:
        raise DefocusAnalysisError("DPF補正後参加者要約にDual planeがありません")
    dp["ID"] = dp["ID"].map(_normalize_identifier)
    dp["Ocularity"] = dp["Ocularity"].astype("string").str.strip()
    if set(dp["Ocularity"].dropna().astype(str)) != set(OCULARITY_ORDER):
        raise DefocusAnalysisError("DPのmonocular・binocularがそろっていません")
    if set(dp["Participant_Aggregation"].astype(str)) != {
        PARTICIPANT_AGGREGATION
    }:
        raise DefocusAnalysisError("DPF補正後参加者要約の集約方法が不正です")
    for column in (
        "Ref_Contrast", "Orientation", CORRECTED_LOG10_COLUMN,
        CORRECTED_AR_COLUMN,
    ):
        dp[column] = pd.to_numeric(dp[column], errors="coerce")
    numeric_columns = [
        "Ref_Contrast", "Orientation", CORRECTED_LOG10_COLUMN,
        CORRECTED_AR_COLUMN,
    ]
    if not np.isfinite(dp[numeric_columns].to_numpy(dtype=float)).all():
        raise DefocusAnalysisError("DPF補正後DPに有限でない数値があります")
    if (dp[CORRECTED_AR_COLUMN] <= 0).any():
        raise DefocusAnalysisError("DPF補正後DPのコントラストには正値が必要です")

    keys = ["ID", *ANALYSIS_GROUP_COLUMNS]
    if dp.duplicated([*keys, "Ocularity"]).any():
        raise DefocusAnalysisError("DPF補正後DPに参加者×解析群×眼条件の重複があります")

    def eye_values(ocularity: str, label: str) -> pd.DataFrame:
        return dp.loc[
            dp["Ocularity"].eq(ocularity),
            [*keys, CORRECTED_LOG10_COLUMN, CORRECTED_AR_COLUMN],
        ].rename(
            columns={
                CORRECTED_LOG10_COLUMN: f"DP_{label}_Corrected_Log10_AR",
                CORRECTED_AR_COLUMN: f"DP_{label}_Corrected_AR_Contrast",
            }
        )

    pairs = eye_values("monocular", "Monocular").merge(
        eye_values("binocular", "Binocular"),
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not pairs["_merge"].eq("both").all():
        raise DefocusAnalysisError("DPのmonocular・binocular対応が不完全です")
    pairs = pairs.drop(columns="_merge")

    defocus_ids = set(defocus["ID"].astype(str))
    dp_ids = set(pairs["ID"].astype(str))
    if defocus_ids != dp_ids:
        raise DefocusAnalysisError(
            "defocus matchingとDPF補正後DPの参加者が一致しません: "
            f"defocus_only={sorted(defocus_ids - dp_ids)}, "
            f"dp_only={sorted(dp_ids - defocus_ids)}"
        )
    pairs = pairs.merge(defocus, on="ID", how="left", validate="many_to_one")
    pairs["Defocus_Dom_Minus_NonDom_mm"] = (
        pairs["Dominant_PD_Mean_mm"]
        - pairs["Non_Dominant_PD_Mean_mm"]
    )
    pairs["DP_Monocular_Minus_Binocular_Log10"] = (
        pairs["DP_Monocular_Corrected_Log10_AR"]
        - pairs["DP_Binocular_Corrected_Log10_AR"]
    )
    pairs["DP_Monocular_to_Binocular_Ratio"] = np.power(
        10.0, pairs["DP_Monocular_Minus_Binocular_Log10"]
    )
    pairs["DP_Monocular_Minus_Binocular_AR"] = (
        pairs["DP_Monocular_Corrected_AR_Contrast"]
        - pairs["DP_Binocular_Corrected_AR_Contrast"]
    )
    pairs["Participant_Aggregation"] = PARTICIPANT_AGGREGATION
    pairs["_ID_Order"] = pairs["ID"].map(_participant_sort_key)
    pairs = pairs.sort_values(
        [*ANALYSIS_GROUP_COLUMNS, "_ID_Order"], kind="stable"
    ).drop(columns="_ID_Order")
    return pairs.reset_index(drop=True)


def _fisher_r_ci(correlation: float, n: int) -> tuple[float, float]:
    if n <= 3 or not np.isfinite(correlation):
        return float("nan"), float("nan")
    from scipy import stats

    clipped = float(np.clip(correlation, -1.0 + 1e-15, 1.0 - 1e-15))
    z_value = np.arctanh(clipped)
    half_width = stats.norm.ppf(0.975) / np.sqrt(n - 3)
    return (
        float(np.tanh(z_value - half_width)),
        float(np.tanh(z_value + half_width)),
    )


def build_defocus_dp_correlations(
    participant_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """解析群別にPearson相関を主解析、Spearman相関を補助解析として返す。"""
    from scipy import stats

    x_column = _DEFOCUS_CORRELATION_X_COLUMN
    y_column = _DEFOCUS_CORRELATION_Y_COLUMN
    required = ["ID", *ANALYSIS_GROUP_COLUMNS, x_column, y_column]
    _require_columns(participant_pairs, required, label="相関用参加者対応表")
    rows: list[dict[str, object]] = []
    for group_values, group in participant_pairs.groupby(
        list(ANALYSIS_GROUP_COLUMNS), sort=True, dropna=False
    ):
        metadata = dict(zip(ANALYSIS_GROUP_COLUMNS, tuple(group_values)))
        x = pd.to_numeric(group[x_column], errors="coerce").to_numpy(float)
        y = pd.to_numeric(group[y_column], errors="coerce").to_numpy(float)
        complete = np.isfinite(x) & np.isfinite(y)
        x, y = x[complete], y[complete]
        n_total, n_complete = len(group), int(complete.sum())
        pearson_r = pearson_p = spearman_rho = spearman_p = float("nan")
        ci_lower = ci_upper = float("nan")
        if n_complete < 3:
            status = "n_lt_3"
        elif np.isclose(np.std(x, ddof=1), 0.0):
            status = "zero_variance_x"
        elif np.isclose(np.std(y, ddof=1), 0.0):
            status = "zero_variance_y"
        else:
            pearson_r, pearson_p = map(float, stats.pearsonr(x, y))
            spearman_rho, spearman_p = map(float, stats.spearmanr(x, y))
            ci_lower, ci_upper = _fisher_r_ci(pearson_r, n_complete)
            status = "ok"
        rows.append(
            {
                **metadata,
                "X_Variable": x_column,
                "Y_Variable": y_column,
                "n_total_participants": n_total,
                "n_complete_participants": n_complete,
                "n_excluded_participants": n_total - n_complete,
                "pearson_r": pearson_r,
                "pearson_p_value_two_sided": pearson_p,
                "pearson_ci95_lower": ci_lower,
                "pearson_ci95_upper": ci_upper,
                "holm_adjusted_pearson_p_value": float("nan"),
                "significant_holm_alpha_0_05": False,
                "spearman_rho": spearman_rho,
                "spearman_p_value_two_sided": spearman_p,
                "Correlation_Status": status,
            }
        )
    adjusted = holm_adjust(
        [row["pearson_p_value_two_sided"] for row in rows]
    )
    for row, adjusted_p in zip(rows, adjusted):
        row["holm_adjusted_pearson_p_value"] = float(adjusted_p)
        row["significant_holm_alpha_0_05"] = bool(
            np.isfinite(adjusted_p) and adjusted_p < 0.05
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def _atomic_replace(temp_path: Path, destination: Path) -> None:
    try:
        os.replace(temp_path, destination)
    except PermissionError as error:
        raise PermissionError(
            "保存先を置換できません。Excelや画像ビューアで同名ファイルを"
            f"開いている場合は閉じてください: {destination}"
        ) from error


def _save_dataframe_csv(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        frame.to_csv(temp_path, index=False, encoding="utf-8-sig")
        _atomic_replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)
    return destination


def _participant_role_values(
    trials: pd.DataFrame,
    participant_id: str,
    role: str,
) -> np.ndarray:
    return (
        trials.loc[
            trials["ID"].astype(str).eq(participant_id)
            & trials["Eye_Role"].eq(role),
            ["Trial", "Matched_PD(mm)"],
        ]
        .sort_values("Trial", kind="stable")["Matched_PD(mm)"]
        .to_numpy(dtype=float)
    )


def _defocus_figure(trials: pd.DataFrame, summary: pd.DataFrame):
    plt = _pyplot()
    participant_ids = summary["ID"].astype(str).tolist()
    x = np.arange(len(participant_ids), dtype=float)
    role_offset = {"dominant": -0.12, "non_dominant": 0.12}
    figure_width = max(8.0, 1.15 * len(participant_ids) + 2.5)
    figure, axis = plt.subplots(figsize=(figure_width, 5.8))

    dominant_x = x + role_offset["dominant"]
    non_dominant_x = x + role_offset["non_dominant"]
    axis.errorbar(
        dominant_x,
        summary["Dominant_PD_Mean_mm"].to_numpy(dtype=float),
        yerr=summary["Dominant_PD_SD_mm"].to_numpy(dtype=float),
        fmt="o",
        color="red",
        ecolor="black",
        elinewidth=1.8,
        capsize=5,
        capthick=1.8,
        label="Dominant eye mean ± SD",
        zorder=4,
    )
    axis.errorbar(
        non_dominant_x,
        summary["Non_Dominant_PD_Mean_mm"].to_numpy(dtype=float),
        yerr=summary["Non_Dominant_PD_SD_mm"].to_numpy(dtype=float),
        fmt="o",
        color="gray",
        ecolor="gray",
        elinewidth=1.8,
        capsize=5,
        capthick=1.8,
        label="Non-dominant eye mean ± SD",
        zorder=4,
    )

    axis.set_xticks(x)
    axis.set_xticklabels(participant_ids)
    axis.set_xlim(-0.6, len(participant_ids) - 0.4)
    axis.set_ylim(_MIN_PUPIL_DIAMETER_MM, _MAX_PUPIL_DIAMETER_MM)
    axis.set_yticks(np.arange(1.0, 6.1, 0.5))
    axis.tick_params(axis="y", direction="in")
    axis.set_xlabel("Participant ID")
    axis.set_ylabel("Estimated pupil diameter (mm)")
    axis.set_title("Defocus matching by participant")
    for boundary in np.arange(0.5, len(participant_ids) - 0.5, 1.0):
        axis.axvline(boundary, linestyle=":", color="gray", alpha=0.4)
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure


def _save_figure(
    trials: pd.DataFrame,
    summary: pd.DataFrame,
    destination: Path,
) -> Path:
    plt = _pyplot()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    figure = _defocus_figure(trials, summary)
    try:
        figure.savefig(
            temp_path,
            format="png",
            dpi=_DPI,
            bbox_inches="tight",
            facecolor="white",
        )
        _atomic_replace(temp_path, destination)
    finally:
        plt.close(figure)
        temp_path.unlink(missing_ok=True)
    return destination


def _limits_including_zero(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return -1.0, 1.0
    lower = min(0.0, float(np.min(finite)))
    upper = max(0.0, float(np.max(finite)))
    span = upper - lower
    scale = max(abs(lower), abs(upper), 0.01)
    padding = max(0.12 * span, 0.05 * scale)
    return lower - padding, upper + padding


def _correlation_group_key(values) -> tuple[str, float, float]:
    session_type, reference, orientation = tuple(values)
    return str(session_type), float(reference), float(orientation)


def _safe_filename_token(value: object) -> str:
    token = re.sub(r"[^0-9A-Za-z_-]+", "_", str(value).strip()).strip("_")
    return token or "unknown"


def _number_filename_token(value: object) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _correlation_figure_filename(metadata: dict[str, object]) -> str:
    return (
        f"{DEFOCUS_DP_CORRELATION_FIGURE_PREFIX}__"
        f"{_safe_filename_token(metadata['Session_Type'])}__"
        f"ref_{_number_filename_token(metadata['Ref_Contrast'])}__"
        f"ori_{_number_filename_token(metadata['Orientation'])}.png"
    )


def _defocus_dp_correlation_figure(
    participant_group: pd.DataFrame,
    correlation_row: pd.Series,
):
    """相関計算に使用した参加者ペアを同じ尺度で散布図にする。"""
    required = [
        "ID",
        *ANALYSIS_GROUP_COLUMNS,
        _DEFOCUS_CORRELATION_X_COLUMN,
        _DEFOCUS_CORRELATION_Y_COLUMN,
    ]
    _require_columns(participant_group, required, label="相関図用参加者対応表")
    if str(correlation_row["X_Variable"]) != _DEFOCUS_CORRELATION_X_COLUMN:
        raise DefocusAnalysisError("相関表のX変数が相関図の定義と一致しません")
    if str(correlation_row["Y_Variable"]) != _DEFOCUS_CORRELATION_Y_COLUMN:
        raise DefocusAnalysisError("相関表のY変数が相関図の定義と一致しません")

    points = participant_group.loc[:, required].copy()
    points[_DEFOCUS_CORRELATION_X_COLUMN] = pd.to_numeric(
        points[_DEFOCUS_CORRELATION_X_COLUMN], errors="coerce"
    )
    points[_DEFOCUS_CORRELATION_Y_COLUMN] = pd.to_numeric(
        points[_DEFOCUS_CORRELATION_Y_COLUMN], errors="coerce"
    )
    complete = np.isfinite(
        points[_DEFOCUS_CORRELATION_X_COLUMN].to_numpy(dtype=float)
    ) & np.isfinite(
        points[_DEFOCUS_CORRELATION_Y_COLUMN].to_numpy(dtype=float)
    )
    points = points.loc[complete].copy()
    expected_n = int(correlation_row["n_complete_participants"])
    if len(points) != expected_n:
        raise DefocusAnalysisError(
            "相関図の完全ケース数が相関表と一致しません: "
            f"figure={len(points)}, table={expected_n}"
        )
    if points.empty:
        raise DefocusAnalysisError("相関図に描画できる参加者ペアがありません")

    x = points[_DEFOCUS_CORRELATION_X_COLUMN].to_numpy(dtype=float)
    y = points[_DEFOCUS_CORRELATION_Y_COLUMN].to_numpy(dtype=float)
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7.6, 6.2))
    axis.axvline(
        0.0, color="#666666", linestyle="--", linewidth=1.1, zorder=1
    )
    axis.axhline(
        0.0, color="#666666", linestyle="--", linewidth=1.1, zorder=1
    )
    axis.scatter(
        x,
        y,
        s=58,
        facecolors="royalblue",
        edgecolors="black",
        linewidths=0.8,
        alpha=0.9,
        label="Participants",
        zorder=3,
    )
    for participant_id, x_value, y_value in zip(points["ID"], x, y):
        axis.annotate(
            str(participant_id),
            (x_value, y_value),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color="#222222",
            zorder=4,
        )

    status = str(correlation_row["Correlation_Status"])
    if status == "ok":
        if len(x) < 2 or np.isclose(np.std(x, ddof=1), 0.0):
            raise DefocusAnalysisError(
                "Correlation_Status=okですが回帰直線を計算できません"
            )
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.linspace(float(np.min(x)), float(np.max(x)), 200)
        axis.plot(
            line_x,
            slope * line_x + intercept,
            color="darkorange",
            linewidth=2.0,
            label="OLS fit",
            zorder=2,
        )

    metadata = {
        column: correlation_row[column]
        for column in ANALYSIS_GROUP_COLUMNS
    }
    axis.set_xlim(*_limits_including_zero(x))
    axis.set_ylim(*_limits_including_zero(y))
    axis.set_xlabel("Defocus matched PD: dominant − non-dominant (mm)")
    axis.set_ylabel("DP corrected log10 AR: monocular − binocular")
    axis.set_title(
        "Defocus matching vs DP ocular difference\n"
        f"Session={metadata['Session_Type']}, "
        f"Ref={float(metadata['Ref_Contrast']):g}, "
        f"Orientation={float(metadata['Orientation']):g}°"
    )
    axis.grid(True, linestyle=":", alpha=0.35)
    axis.legend(frameon=False, loc="best")
    figure.tight_layout()
    return figure


def _save_defocus_dp_correlation_figures(
    participant_pairs: pd.DataFrame,
    correlation_summary: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    """解析群ごとに相関散布図を保存し、そのパスを返す。"""
    required_statistics = [
        *ANALYSIS_GROUP_COLUMNS,
        "X_Variable",
        "Y_Variable",
        "n_complete_participants",
        "Correlation_Status",
    ]
    _require_columns(correlation_summary, required_statistics, label="相関結果表")
    if correlation_summary.duplicated(
        list(ANALYSIS_GROUP_COLUMNS), keep=False
    ).any():
        raise DefocusAnalysisError("相関結果表に解析群の重複があります")

    statistics_by_key = {
        _correlation_group_key(
            row[column] for column in ANALYSIS_GROUP_COLUMNS
        ): row
        for _, row in correlation_summary.iterrows()
    }
    groups = [
        (_correlation_group_key(group_values), group)
        for group_values, group in participant_pairs.groupby(
            list(ANALYSIS_GROUP_COLUMNS), sort=True, dropna=False
        )
    ]
    if {key for key, _ in groups} != set(statistics_by_key):
        raise DefocusAnalysisError(
            "相関用参加者表と相関結果表の解析群が一致しません"
        )

    plt = _pyplot()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for old_path in destination.glob(
        f"{DEFOCUS_DP_CORRELATION_FIGURE_PREFIX}__*.png"
    ):
        old_path.unlink()

    outputs: list[Path] = []
    for key, group in groups:
        statistics = statistics_by_key[key]
        metadata = dict(zip(ANALYSIS_GROUP_COLUMNS, key))
        output_path = destination / _correlation_figure_filename(metadata)
        temp_path = output_path.with_name(
            f".{output_path.name}.{uuid4().hex}.tmp"
        )
        figure = _defocus_dp_correlation_figure(group, statistics)
        try:
            figure.savefig(
                temp_path,
                format="png",
                dpi=_DPI,
                bbox_inches="tight",
                facecolor="white",
            )
            _atomic_replace(temp_path, output_path)
        finally:
            plt.close(figure)
            temp_path.unlink(missing_ok=True)
        outputs.append(output_path)
    return tuple(outputs)


def save_defocus_matching_outputs(
    session_dirs: Sequence[str | Path],
    *,
    corrected_summary: pd.DataFrame,
    table_output_dir: str | Path,
    figure_output_dir: str | Path,
    save_figure: bool = True,
) -> DefocusAnalysisResult:
    """Defocus要約、DP眼条件差との参加者対応、相関を保存する。"""
    trials = load_defocus_trials(session_dirs)
    summary = build_defocus_participant_summary(trials)
    participant_pairs = build_defocus_dp_participant_pairs(
        summary, corrected_summary
    )
    correlation_summary = build_defocus_dp_correlations(participant_pairs)
    table_directory = Path(table_output_dir)
    summary_file = _save_dataframe_csv(
        summary, table_directory / DEFOCUS_SUMMARY_FILENAME
    )
    participant_pairs_file = _save_dataframe_csv(
        participant_pairs, table_directory / DEFOCUS_DP_PAIRS_FILENAME
    )
    correlation_file = _save_dataframe_csv(
        correlation_summary, table_directory / DEFOCUS_DP_CORRELATION_FILENAME
    )
    figure_directory = Path(figure_output_dir)
    if save_figure:
        figure_file = _save_figure(
            trials,
            summary,
            figure_directory / DEFOCUS_FIGURE_FILENAME,
        )
        correlation_figure_files = _save_defocus_dp_correlation_figures(
            participant_pairs,
            correlation_summary,
            figure_directory,
        )
    else:
        figure_file = None
        correlation_figure_files = tuple()

    print(f"Saved defocus participant summary: {summary_file}")
    print(f"Saved defocus/DP participant pairs: {participant_pairs_file}")
    print(f"Saved defocus/DP correlations: {correlation_file}")
    if figure_file is not None:
        print(f"Saved defocus matching figure: {figure_file}")
        print(
            "Saved defocus/DP correlation figure(s): "
            f"{len(correlation_figure_files)}"
        )
    return DefocusAnalysisResult(
        participant_summary=summary,
        participant_pairs=participant_pairs,
        correlation_summary=correlation_summary,
        summary_file=summary_file,
        participant_pairs_file=participant_pairs_file,
        correlation_file=correlation_file,
        figure_file=figure_file,
        correlation_figure_files=correlation_figure_files,
    )


__all__ = [
    "DEFOCUS_DP_CORRELATION_FILENAME",
    "DEFOCUS_DP_PAIRS_FILENAME",
    "DEFOCUS_FIGURE_FILENAME",
    "DEFOCUS_SUMMARY_FILENAME",
    "DefocusAnalysisError",
    "DefocusAnalysisResult",
    "build_defocus_dp_correlations",
    "build_defocus_dp_participant_pairs",
    "build_defocus_participant_summary",
    "load_defocus_trials",
    "save_defocus_matching_outputs",
]