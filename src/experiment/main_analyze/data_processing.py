"""本実験のCSV探索・検証・参加者単位集約を担当する。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .config import (
    ANALYSIS_GROUP_COLUMNS,
    AR_VALUE_COLUMN,
    CONDITION_ORDER,
    EXPERIMENT_RESULT_ROOT,
    GEOMETRIC_MEAN_COLUMN,
    MEAN_LOG10_COLUMN,
    OCULARITY_ORDER,
    OPTIONAL_METADATA_COLUMNS,
    PARTICIPANT_AGGREGATION,
    PARTICIPANT_SUMMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    SESSION_DIR_PATTERN,
    SESSION_RESULT_FILENAMES,
    TRIAL_COUNT_COLUMN,
    TRIAL_LOG10_COLUMN,
)


NUMERIC_COLUMNS = (
    "Ref_Contrast",
    "Orientation",
    "Matched_Contrast",
    "L_fg",
    "L_bg",
    "L_ref",
)
EXCLUSION_COLUMNS = (
    "Scope",
    "Source_CSV",
    "Source_Row",
    "ID",
    "Reason",
    "Detail",
)
_PROVENANCE_COLUMNS = (
    "Session_Timestamp",
    "Session_Dir",
    "Source_CSV",
)
_NUMERIC_ID_PATTERN = re.compile(r"^([+-]?\d+)\.0+$")


class DataValidationError(ValueError):
    """解析対象データが事前に定めた構造を満たさない場合の例外。"""


@dataclass(frozen=True)
class SessionSource:
    """1つのセッションフォルダと結果CSVの対応。"""

    session_dir: Path
    csv_path: Path
    session_type: str
    participant_id: str
    timestamp: str


@dataclass(frozen=True)
class LoadedTrialData:
    """検証済み試行データと入力セッションの記録。"""

    trials: pd.DataFrame
    sessions: pd.DataFrame
    exclusions: pd.DataFrame

    @property
    def participant_ids(self) -> tuple[str, ...]:
        return tuple(self.sessions["ID"].astype(str).tolist())

    @property
    def session_timestamps(self) -> tuple[str, ...]:
        return tuple(self.sessions["Session_Timestamp"].astype(str).tolist())


def empty_exclusion_log() -> pd.DataFrame:
    """除外がない場合にも固定列を持つ空の記録表を返す。"""
    return pd.DataFrame(columns=list(EXCLUSION_COLUMNS))


def _as_path_sequence(
    paths: Sequence[str | Path] | str | Path | None,
) -> list[Path]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        paths = [paths]
    return [Path(path).expanduser().resolve() for path in paths]


def _contains_result_csv(directory: Path) -> bool:
    return any(
        (directory / filename).is_file()
        for filename in SESSION_RESULT_FILENAMES.values()
    )


def discover_session_dirs(
    explicit_paths: Sequence[str | Path] | str | Path | None = None,
    *,
    result_root: str | Path = EXPERIMENT_RESULT_ROOT,
) -> list[Path]:
    """セッション、結果CSV、または親フォルダから解析対象を列挙する。

    引数を省略した場合は本実験の既定ルート直下を探索する。探索は
    1階層だけとし、同じセッションが複数回指定されても1件へまとめる。
    """
    requested = _as_path_sequence(explicit_paths)
    roots = requested or [Path(result_root).expanduser().resolve()]
    candidates: list[Path] = []

    for path in roots:
        if not path.exists():
            raise FileNotFoundError(f"入力パスが見つかりません: {path}")
        if path.is_file():
            if path.name not in SESSION_RESULT_FILENAMES.values():
                raise DataValidationError(
                    "結果CSVとして指定できるファイル名ではありません: "
                    f"{path.name}"
                )
            candidates.append(path.parent)
            continue
        if not path.is_dir():
            raise DataValidationError(f"入力パスがディレクトリではありません: {path}")
        if _contains_result_csv(path):
            candidates.append(path)
            continue
        candidates.extend(
            child
            for child in sorted(path.iterdir(), key=lambda item: item.name)
            if child.is_dir() and _contains_result_csv(child)
        )

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)

    if not unique:
        target = ", ".join(str(path) for path in roots)
        raise FileNotFoundError(f"解析可能な結果CSVがありません: {target}")
    return unique


def _normalize_identifier(value: object) -> str:
    if pd.isna(value):
        return ""
    normalized = str(value).strip()
    numeric_match = _NUMERIC_ID_PATTERN.fullmatch(normalized)
    return numeric_match.group(1) if numeric_match else normalized


def _session_source(session_dir: Path) -> SessionSource:
    existing = [
        (session_type, session_dir / filename)
        for session_type, filename in SESSION_RESULT_FILENAMES.items()
        if (session_dir / filename).is_file()
    ]
    if not existing:
        raise FileNotFoundError(f"結果CSVがありません: {session_dir}")
    if len(existing) > 1:
        filenames = ", ".join(path.name for _, path in existing)
        raise DataValidationError(
            f"同一セッションに複数種類の結果CSVがあります: {filenames}"
        )

    folder_match = SESSION_DIR_PATTERN.fullmatch(session_dir.name)
    if folder_match is None:
        raise DataValidationError(
            "セッションフォルダ名はID_YYYYMMDD_HHMMSS形式にしてください: "
            f"{session_dir.name}"
        )

    session_type, csv_path = existing[0]
    participant_id = _normalize_identifier(folder_match.group("participant_id"))
    if not participant_id:
        raise DataValidationError(
            f"セッションフォルダから参加者IDを取得できません: {session_dir}"
        )
    return SessionSource(
        session_dir=session_dir,
        csv_path=csv_path,
        session_type=session_type,
        participant_id=participant_id,
        timestamp=folder_match.group("timestamp"),
    )


def _build_sources(session_dirs: Sequence[str | Path]) -> list[SessionSource]:
    sources = [_session_source(Path(path).expanduser().resolve()) for path in session_dirs]
    duplicates: dict[tuple[str, str], list[SessionSource]] = {}
    for source in sources:
        duplicates.setdefault(
            (source.session_type, source.participant_id), []
        ).append(source)
    repeated = {
        key: values for key, values in duplicates.items() if len(values) > 1
    }
    if repeated:
        details = "; ".join(
            f"{session_type}/{participant_id}: "
            + ", ".join(source.session_dir.name for source in values)
            for (session_type, participant_id), values in sorted(repeated.items())
        )
        raise DataValidationError(
            "同一参加者の複数セッションが同時に選択されています。"
            f"再試行を混在させず1件だけ指定してください: {details}"
        )
    return sources


def _require_columns(frame: pd.DataFrame, source: SessionSource) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataValidationError(
            f"{source.csv_path}: 必須列が不足しています: {missing}"
        )


def _normalize_required_strings(
    frame: pd.DataFrame,
    source: SessionSource,
) -> None:
    for column in ("ID", "Condition", "Ocularity"):
        values = frame[column].astype("string").str.strip()
        if column == "Ocularity":
            values = values.str.lower()
        frame[column] = values
        missing = values.isna() | values.eq("")
        if missing.any():
            rows = frame.loc[missing, "Source_Row"].astype(int).tolist()[:10]
            raise DataValidationError(
                f"{source.csv_path}: {column}が空の行があります: {rows}"
            )


def _convert_numeric_columns(
    frame: pd.DataFrame,
    source: SessionSource,
) -> None:
    for column in NUMERIC_COLUMNS:
        converted = pd.to_numeric(frame[column], errors="coerce")
        invalid = converted.isna() | ~np.isfinite(converted.to_numpy(dtype=float))
        if invalid.any():
            rows = frame.loc[invalid, "Source_Row"].astype(int).tolist()[:10]
            raise DataValidationError(
                f"{source.csv_path}: {column}を有限な数値へ変換できません: {rows}"
            )
        frame[column] = converted.astype(float)


def _validate_categories(frame: pd.DataFrame, source: SessionSource) -> None:
    condition_values = set(frame["Condition"].astype(str))
    unexpected_conditions = sorted(condition_values - set(CONDITION_ORDER))
    if unexpected_conditions:
        raise DataValidationError(
            f"{source.csv_path}: 未定義のConditionがあります: "
            f"{unexpected_conditions}"
        )

    ocularity_values = set(frame["Ocularity"].astype(str))
    unexpected_ocularities = sorted(ocularity_values - set(OCULARITY_ORDER))
    if unexpected_ocularities:
        raise DataValidationError(
            f"{source.csv_path}: 未定義のOcularityがあります: "
            f"{unexpected_ocularities}"
        )


def _validate_numeric_ranges(frame: pd.DataFrame, source: SessionSource) -> None:
    checks = {
        "Ref_Contrast > 0": frame["Ref_Contrast"] > 0,
        "Matched_Contrast > 0": frame["Matched_Contrast"] > 0,
        "L_fg >= 0": frame["L_fg"] >= 0,
        "L_bg >= 0": frame["L_bg"] >= 0,
        "L_ref > 0": frame["L_ref"] > 0,
        "L_fg + L_bg > 0": (frame["L_fg"] + frame["L_bg"]) > 0,
    }
    for rule, valid in checks.items():
        if not bool(valid.all()):
            rows = frame.loc[~valid, "Source_Row"].astype(int).tolist()[:10]
            raise DataValidationError(
                f"{source.csv_path}: {rule}を満たさない行があります: {rows}"
            )


def _validate_source_id(frame: pd.DataFrame, source: SessionSource) -> None:
    normalized_ids = frame["ID"].map(_normalize_identifier)
    frame["ID"] = normalized_ids
    csv_ids = sorted(set(normalized_ids) - {""})
    if len(csv_ids) != 1:
        raise DataValidationError(
            f"{source.csv_path}: CSV内のIDは1種類にしてください: {csv_ids}"
        )
    if csv_ids[0] != source.participant_id:
        raise DataValidationError(
            "フォルダIDとCSV内IDが一致しません: "
            f"folder={source.participant_id}, csv={csv_ids[0]}, "
            f"path={source.csv_path}"
        )


def _validate_optional_trial_ids(
    frame: pd.DataFrame,
    source: SessionSource,
) -> None:
    if "Trial_ID" not in frame.columns:
        return
    trial_ids = frame["Trial_ID"].astype("string").str.strip()
    present = trial_ids.notna() & trial_ids.ne("")
    duplicated = present & trial_ids.duplicated(keep=False)
    if duplicated.any():
        values = sorted(set(trial_ids.loc[duplicated].astype(str)))[:10]
        raise DataValidationError(
            f"{source.csv_path}: Trial_IDが重複しています: {values}"
        )


def _prepare_session_frame(source: SessionSource) -> pd.DataFrame:
    frame = pd.read_csv(
        source.csv_path,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=True,
    )
    if frame.empty:
        raise DataValidationError(f"空のCSVは解析できません: {source.csv_path}")

    frame = frame.copy()
    frame["Source_Row"] = np.arange(2, len(frame) + 2, dtype=int)
    _require_columns(frame, source)
    _normalize_required_strings(frame, source)
    _convert_numeric_columns(frame, source)
    _validate_categories(frame, source)
    _validate_numeric_ranges(frame, source)
    _validate_source_id(frame, source)
    _validate_optional_trial_ids(frame, source)

    if "Session_Type" in frame.columns:
        recorded = set(
            frame["Session_Type"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
        )
        if recorded and recorded != {source.session_type}:
            raise DataValidationError(
                f"{source.csv_path}: Session_Typeとファイル名が矛盾します: "
                f"{sorted(recorded)} vs {source.session_type}"
            )

    frame["Session_Type"] = source.session_type
    frame["Session_Timestamp"] = source.timestamp
    frame["Session_Dir"] = str(source.session_dir)
    frame["Source_CSV"] = str(source.csv_path)

    denominator = frame["L_fg"] + frame["L_bg"]
    frame[AR_VALUE_COLUMN] = (
        frame["L_fg"] * frame["Matched_Contrast"] / denominator
    )
    invalid_ar = (
        ~np.isfinite(frame[AR_VALUE_COLUMN].to_numpy(dtype=float))
        | (frame[AR_VALUE_COLUMN] <= 0)
    )
    if invalid_ar.any():
        rows = frame.loc[invalid_ar, "Source_Row"].astype(int).tolist()[:10]
        raise DataValidationError(
            f"{source.csv_path}: {AR_VALUE_COLUMN} > 0が必要です: {rows}"
        )
    frame[TRIAL_LOG10_COLUMN] = np.log10(
        frame[AR_VALUE_COLUMN].to_numpy(dtype=float)
    )
    return frame


def load_trials(session_dirs: Sequence[str | Path]) -> LoadedTrialData:
    """結果CSVを読み込み、ID・列・値域を検証して派生値を追加する。

    不正な行を黙って除外せず、問題を検出した時点で停止する。そのため、
    正常終了時のexclusionsは固定列を持つ空表となる。
    """
    sources = _build_sources(session_dirs)
    if not sources:
        raise FileNotFoundError("解析対象のセッションがありません")

    frames: list[pd.DataFrame] = []
    session_rows: list[dict[str, object]] = []
    for source in sources:
        frame = _prepare_session_frame(source)
        frames.append(frame)
        session_rows.append(
            {
                "ID": source.participant_id,
                "Session_Type": source.session_type,
                "Session_Timestamp": source.timestamp,
                "Session_Dir": str(source.session_dir),
                "Source_CSV": str(source.csv_path),
                "Trial_Count": len(frame),
            }
        )

    trials = pd.concat(frames, ignore_index=True, sort=False)
    sessions = pd.DataFrame(session_rows).sort_values(
        ["Session_Type", "ID", "Session_Timestamp"],
        kind="stable",
        ignore_index=True,
    )
    return LoadedTrialData(
        trials=trials,
        sessions=sessions,
        exclusions=empty_exclusion_log(),
    )


def _validate_complete_design(summary: pd.DataFrame) -> None:
    expected = {
        (ocularity, condition)
        for ocularity in OCULARITY_ORDER
        for condition in CONDITION_ORDER
    }
    design_group_columns = ["ID", *ANALYSIS_GROUP_COLUMNS]
    errors: list[str] = []

    for group_values, group in summary.groupby(
        design_group_columns,
        sort=True,
        dropna=False,
    ):
        present = set(zip(group["Ocularity"], group["Condition"]))
        missing = sorted(expected - present)
        if missing:
            metadata = dict(zip(design_group_columns, group_values))
            errors.append(f"{metadata}: missing={missing}")
            continue
        trial_counts = group[TRIAL_COUNT_COLUMN].astype(int)
        if trial_counts.nunique() != 1:
            counts = {
                f"{row.Ocularity}/{row.Condition}": int(
                    getattr(row, TRIAL_COUNT_COLUMN)
                )
                for row in group.itertuples(index=False)
            }
            metadata = dict(zip(design_group_columns, group_values))
            errors.append(f"{metadata}: unequal_trial_counts={counts}")

    if errors:
        preview = "\n".join(errors[:10])
        suffix = "\n..." if len(errors) > 10 else ""
        raise DataValidationError(
            "H1〜H4に必要な4条件×2眼、または反復数がそろっていません:\n"
            f"{preview}{suffix}"
        )


def build_participant_summary(trials: pd.DataFrame) -> pd.DataFrame:
    """試行のlog10 ARを参加者×条件内で算術平均する。"""
    required = [
        *PARTICIPANT_SUMMARY_KEY_COLUMNS,
        AR_VALUE_COLUMN,
        TRIAL_LOG10_COLUMN,
        *_PROVENANCE_COLUMNS,
    ]
    missing = [column for column in required if column not in trials.columns]
    if missing:
        raise DataValidationError(
            f"参加者集約に必要な列が不足しています: {missing}"
        )
    if trials.empty:
        raise DataValidationError("試行データが空です")

    source_counts = trials.groupby(
        list(PARTICIPANT_SUMMARY_KEY_COLUMNS),
        dropna=False,
    )["Source_CSV"].nunique()
    if (source_counts != 1).any():
        raise DataValidationError(
            "同じ参加者・条件の試行が複数CSVにまたがっています"
        )

    summary = (
        trials.groupby(
            list(PARTICIPANT_SUMMARY_KEY_COLUMNS),
            as_index=False,
            sort=True,
            dropna=False,
        )
        .agg(
            **{
                TRIAL_COUNT_COLUMN: (TRIAL_LOG10_COLUMN, "size"),
                MEAN_LOG10_COLUMN: (TRIAL_LOG10_COLUMN, "mean"),
                "Session_Timestamp": ("Session_Timestamp", "first"),
                "Session_Dir": ("Session_Dir", "first"),
                "Source_CSV": ("Source_CSV", "first"),
            }
        )
    )
    summary[GEOMETRIC_MEAN_COLUMN] = np.power(
        10.0,
        summary[MEAN_LOG10_COLUMN].to_numpy(dtype=float),
    )
    summary["Participant_Aggregation"] = PARTICIPANT_AGGREGATION

    _validate_complete_design(summary)

    ordered_columns = [
        *PARTICIPANT_SUMMARY_KEY_COLUMNS,
        TRIAL_COUNT_COLUMN,
        MEAN_LOG10_COLUMN,
        GEOMETRIC_MEAN_COLUMN,
        "Participant_Aggregation",
        "Session_Timestamp",
        "Session_Dir",
        "Source_CSV",
    ]
    return summary.loc[:, ordered_columns].reset_index(drop=True)


__all__ = [
    "DataValidationError",
    "EXCLUSION_COLUMNS",
    "LoadedTrialData",
    "NUMERIC_COLUMNS",
    "SessionSource",
    "build_participant_summary",
    "discover_session_dirs",
    "empty_exclusion_log",
    "load_trials",
]