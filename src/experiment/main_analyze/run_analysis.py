"""全参加者解析と個別・training描画を切り替えるCLI入口。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from .config import (
    ANALYSIS_GROUP_COLUMNS,
    COMBINED_FIGURE_ROOT,
    CORRECTED_SUMMARY_FILENAME,
    EXCLUSIONS_FILENAME,
    EXPERIMENT_FIGURE_ROOT,
    EXPERIMENT_RESULT_ROOT,
    HYPOTHESIS_TESTS_FILENAME,
    MANIFEST_FILENAME,
    PARTICIPANT_AGGREGATION,
    TRAINING_FIGURE_ROOT,
    UNCORRECTED_SUMMARY_FILENAME,
    AnalysisOutputPaths,
    build_run_name,
    resolve_output_paths,
)
from .contrast_metrics import build_legacy_contrast_frame
from .data_processing import (
    LoadedTrialData,
    build_participant_summary,
    discover_session_dirs,
    load_trials,
)
from .dpf_correction import CORRECTION_METHOD, apply_dpf_correction
from .figures import save_all_figures, save_legacy_contrast_figures
from .hypothesis_tests import (
    EXPECTED_ROWS_PER_ANALYSIS_GROUP,
    run_hypothesis_tests,
)


@dataclass(frozen=True)
class AnalysisRunResult:
    """1回の解析で生成した表・図・保存先。"""

    output_paths: AnalysisOutputPaths
    session_dirs: tuple[Path, ...]
    uncorrected_summary: pd.DataFrame
    corrected_summary: pd.DataFrame
    hypothesis_tests: pd.DataFrame
    exclusions: pd.DataFrame
    figure_files: dict[str, tuple[Path, ...]]
    analysis_mode: str = "full_analysis"


def _resolved_paths(
    paths: Sequence[str | Path] | str | Path | None,
) -> list[Path]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        paths = [paths]
    return [Path(path).expanduser().resolve() for path in paths]


def _is_all_participants_request(
    input_paths: Sequence[str | Path] | str | Path | None,
    *,
    result_root: str | Path,
) -> bool:
    """省略時と実験ルート明示時を同じall_participants扱いにする。"""
    requested = _resolved_paths(input_paths)
    if not requested:
        return True
    root = Path(result_root).expanduser().resolve()
    return len(requested) == 1 and requested[0] == root



FULL_ANALYSIS_MODE = "full_analysis"
FIGURE_ONLY_MODE = "figure_only"


def _session_types(loaded: LoadedTrialData) -> frozenset[str]:
    values = frozenset(
        loaded.sessions["Session_Type"].dropna().astype(str).str.strip()
    )
    if not values:
        raise ValueError("Session_Typeを判定できません")
    return values


def _analysis_mode(
    *,
    all_participants: bool,
    session_types: frozenset[str],
) -> str:
    if all_participants and session_types == {"experiment"}:
        return FULL_ANALYSIS_MODE
    return FIGURE_ONLY_MODE


def _default_figure_root(
    session_types: frozenset[str],
) -> Path:
    if session_types == {"experiment"}:
        return EXPERIMENT_FIGURE_ROOT
    if session_types == {"training"}:
        return TRAINING_FIGURE_ROOT
    return COMBINED_FIGURE_ROOT


def _empty_result_frame() -> pd.DataFrame:
    return pd.DataFrame()


def _atomic_replace(temp_path: Path, destination: Path) -> None:
    try:
        os.replace(temp_path, destination)
    except PermissionError as error:
        raise PermissionError(
            "保存先を置換できません。Excelなどで同名ファイルを開いている場合は"
            f"閉じてから再実行してください: {destination}"
        ) from error
    except OSError as error:
        raise OSError(
            f"解析出力を保存できません: {destination} ({error})"
        ) from error


def _atomic_dataframe_csv(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(
        f".{destination.name}.{uuid4().hex}.tmp"
    )
    try:
        frame.to_csv(temp_path, index=False, encoding="utf-8-sig")
        _atomic_replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)
    return destination


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"JSONへ変換できない値です: {type(value).__name__}")


def _atomic_json(payload: dict[str, object], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(
        f".{destination.name}.{uuid4().hex}.tmp"
    )
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            )
            stream.write("\n")
        _atomic_replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)
    return destination


def _save_tables(
    *,
    output_paths: AnalysisOutputPaths,
    uncorrected_summary: pd.DataFrame,
    corrected_summary: pd.DataFrame,
    hypothesis_tests: pd.DataFrame,
    exclusions: pd.DataFrame,
) -> dict[str, Path]:
    output_paths.table_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "participant_summary_uncorrected": output_paths.table_dir
        / UNCORRECTED_SUMMARY_FILENAME,
        "participant_summary_dpf_corrected": output_paths.table_dir
        / CORRECTED_SUMMARY_FILENAME,
        "planned_hypothesis_tests": output_paths.table_dir
        / HYPOTHESIS_TESTS_FILENAME,
        "analysis_exclusions": output_paths.table_dir / EXCLUSIONS_FILENAME,
    }
    _atomic_dataframe_csv(
        uncorrected_summary,
        outputs["participant_summary_uncorrected"],
    )
    _atomic_dataframe_csv(
        corrected_summary,
        outputs["participant_summary_dpf_corrected"],
    )
    _atomic_dataframe_csv(
        hypothesis_tests,
        outputs["planned_hypothesis_tests"],
    )
    _atomic_dataframe_csv(exclusions, outputs["analysis_exclusions"])
    return outputs


def _manifest_payload(
    *,
    result: AnalysisRunResult,
    loaded: LoadedTrialData,
    all_participants: bool,
    table_files: dict[str, Path],
) -> dict[str, object]:
    figure_files = {
        family: [str(path) for path in paths]
        for family, paths in result.figure_files.items()
    }
    participant_ids = sorted(set(loaded.sessions["ID"].astype(str)))
    group_count = len(
        result.uncorrected_summary.loc[:, list(ANALYSIS_GROUP_COLUMNS)]
        .drop_duplicates()
    )
    return {
        "Manifest_Version": 1,
        "Generated_At_UTC": datetime.now(timezone.utc).isoformat(),
        "Run_Name": result.output_paths.run_name,
        "Analysis_Mode": result.analysis_mode,
        "Input_Mode": "all_participants" if all_participants else "explicit_selection",
        "Participant_Aggregation": PARTICIPANT_AGGREGATION,
        "DPF_Correction_Method": CORRECTION_METHOD,
        "Hypothesis_Rows_Per_Analysis_Group": EXPECTED_ROWS_PER_ANALYSIS_GROUP,
        "Analysis_Group_Count": group_count,
        "Session_Count": len(loaded.sessions),
        "Participant_Count": len(participant_ids),
        "Participant_IDs": participant_ids,
        "Trial_Row_Count": len(loaded.trials),
        "Uncorrected_Summary_Row_Count": len(result.uncorrected_summary),
        "Corrected_Summary_Row_Count": len(result.corrected_summary),
        "Hypothesis_Test_Row_Count": len(result.hypothesis_tests),
        "Exclusion_Row_Count": len(result.exclusions),
        "Input_Sessions": loaded.sessions.to_dict(orient="records"),
        "Output_Table_Directory": str(result.output_paths.table_dir),
        "Output_Figure_Directory": str(result.output_paths.figure_dir),
        "Table_Files": {name: str(path) for name, path in table_files.items()},
        "Figure_Files": figure_files,
    }


def run_analysis(
    input_paths: Sequence[str | Path] | str | Path | None = None,
    *,
    result_root: str | Path = EXPERIMENT_RESULT_ROOT,
    table_output_dir: str | Path | None = None,
    figure_output_dir: str | Path | None = None,
    save_figures: bool = True,
) -> AnalysisRunResult:
    """入力範囲に応じて全参加者解析またはグラフ専用処理を実行する。"""
    all_participants = _is_all_participants_request(
        input_paths,
        result_root=result_root,
    )
    session_dirs = discover_session_dirs(
        input_paths,
        result_root=result_root,
    )
    loaded = load_trials(session_dirs)
    session_types = _session_types(loaded)
    analysis_mode = _analysis_mode(
        all_participants=all_participants,
        session_types=session_types,
    )

    run_name = build_run_name(
        all_participants=all_participants,
        participant_ids=loaded.participant_ids,
        session_timestamps=loaded.session_timestamps,
    )
    output_paths = resolve_output_paths(
        run_name,
        table_output_dir=table_output_dir,
        figure_output_dir=figure_output_dir,
        default_figure_root=_default_figure_root(session_types),
    )

    if analysis_mode == FIGURE_ONLY_MODE:
        if save_figures:
            figure_frame = build_legacy_contrast_frame(loaded.trials)
            saved_figures = save_legacy_contrast_figures(
                figure_frame,
                output_paths.figure_dir,
            )
            figure_files = {
                family: tuple(paths)
                for family, paths in saved_figures.items()
            }
        else:
            figure_files = {
                "raw_contrast": tuple(),
                "ar_contrast": tuple(),
                "extended_contrast": tuple(),
            }

        result = AnalysisRunResult(
            output_paths=output_paths,
            session_dirs=tuple(session_dirs),
            uncorrected_summary=_empty_result_frame(),
            corrected_summary=_empty_result_frame(),
            hypothesis_tests=_empty_result_frame(),
            exclusions=loaded.exclusions,
            figure_files=figure_files,
            analysis_mode=analysis_mode,
        )
        print(
            f"Loaded {len(loaded.sessions)} session(s), "
            f"{loaded.sessions['ID'].nunique()} participant(s)."
        )
        print(
            "Figure-only mode: participant aggregation, DPF correction, "
            "H1-H4, and analysis tables were skipped."
        )
        if save_figures:
            figure_count = sum(len(paths) for paths in figure_files.values())
            print(
                f"Saved {figure_count} raw/AR/extended figure(s): "
                f"{output_paths.figure_dir}"
            )
        else:
            print("Figure output skipped by request; no files were generated.")
        return result

    uncorrected_summary = build_participant_summary(loaded.trials)
    corrected_summary = apply_dpf_correction(uncorrected_summary)
    hypothesis_tests = run_hypothesis_tests(
        uncorrected_summary,
        corrected_summary,
    )

    output_paths.table_dir.mkdir(parents=True, exist_ok=True)
    output_paths.figure_dir.mkdir(parents=True, exist_ok=True)
    table_files = _save_tables(
        output_paths=output_paths,
        uncorrected_summary=uncorrected_summary,
        corrected_summary=corrected_summary,
        hypothesis_tests=hypothesis_tests,
        exclusions=loaded.exclusions,
    )
    if save_figures:
        saved_figures = save_all_figures(
            uncorrected_summary,
            corrected_summary,
            output_paths,
        )
        figure_files = {
            family: tuple(paths) for family, paths in saved_figures.items()
        }
    else:
        figure_files = {
            "uncorrected": tuple(),
            "dpf_corrected": tuple(),
            "h4_interaction": tuple(),
        }

    result = AnalysisRunResult(
        output_paths=output_paths,
        session_dirs=tuple(session_dirs),
        uncorrected_summary=uncorrected_summary,
        corrected_summary=corrected_summary,
        hypothesis_tests=hypothesis_tests,
        exclusions=loaded.exclusions,
        figure_files=figure_files,
        analysis_mode=analysis_mode,
    )
    manifest = _manifest_payload(
        result=result,
        loaded=loaded,
        all_participants=all_participants,
        table_files=table_files,
    )
    _atomic_json(manifest, output_paths.table_dir / MANIFEST_FILENAME)

    print(
        f"Loaded {len(loaded.sessions)} session(s), "
        f"{loaded.sessions['ID'].nunique()} participant(s)."
    )
    print(f"Saved analysis tables: {output_paths.table_dir}")
    if save_figures:
        figure_count = sum(len(paths) for paths in figure_files.values())
        print(
            f"Saved {figure_count} figure(s): "
            f"{output_paths.figure_dir}"
        )
    else:
        print("Figure output skipped by request.")
    return result

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "全参加者の本実験データではDPF補正・H1〜H4を実行し、"
            "個別指定またはtrainingでは3種類の図だけを出力します。"
        )
    )
    parser.add_argument(
        "input_paths",
        nargs="*",
        help=(
            "セッションフォルダ、結果CSV、またはそれらを含む親フォルダ。"
            "省略時は既定の本実験ルートを使用します。"
        ),
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=EXPERIMENT_RESULT_ROOT,
        help="入力省略時に探索する本実験結果ルート。",
    )
    parser.add_argument(
        "--table-output-dir",
        type=Path,
        help="全参加者解析モードの解析CSV・manifest明示保存先。",
    )
    parser.add_argument(
        "--figure-output-dir",
        type=Path,
        help="図の明示保存先。",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="PNG生成を省略します。グラフ専用モードでは出力がなくなります。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        result = run_analysis(
            args.input_paths or None,
            result_root=args.result_root,
            table_output_dir=args.table_output_dir,
            figure_output_dir=args.figure_output_dir,
            save_figures=not args.skip_figures,
        )
        if result.analysis_mode == FULL_ANALYSIS_MODE:
            # 既存解析の完了後に、all_participants専用の独立出力を追加する。
            from .defocus_analysis import save_defocus_matching_outputs

            save_defocus_matching_outputs(
                result.session_dirs,
                table_output_dir=result.output_paths.table_dir,
                figure_output_dir=(
                    result.output_paths.figure_dir / "defocus_matching"
                ),
                save_figure=not args.skip_figures,
            )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
        print(f"Analysis failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AnalysisRunResult",
    "FIGURE_ONLY_MODE",
    "FULL_ANALYSIS_MODE",
    "build_argument_parser",
    "main",
    "run_analysis",
]