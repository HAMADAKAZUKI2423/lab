"""1名の予備実験の試行効果を真の参加者効果と仮定して必要人数を推定する。

このモジュールは通常のH1〜H4解析とは独立して実行する。1セッション・1参加者の
反復試行から試行単位Cohen's dzを求め、その値を本実験の真の参加者レベルdzと
仮定した両側t検定の事前検出力分析を行う。TOSTの必要人数は算出しない。

実行例:
  cd src
  py -m experiment.main_analyze.pilot_power_analysis \
    ../results/tables/pre-experiment-matching/experiment/1_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd

from .config import (
    ALPHA,
    ANALYSIS_GROUP_COLUMNS,
    CONDITION_ORDER,
    CORRECTED_CONDITIONS,
    DP_CONDITION,
    DPF_CONDITION,
    OCULARITY_ORDER,
    SP_CONDITION,
    SPD_CONDITION,
    TRIAL_LOG10_COLUMN,
)
from .data_processing import discover_session_dirs, load_trials


DEFAULT_TARGET_POWER = 0.80
DEFAULT_MAX_PARTICIPANTS = 10000
EXPECTED_ROWS_PER_ANALYSIS_GROUP = 15
DETAIL_FILENAME = "pilot_trial_effect_based_required_participants.csv"
SUMMARY_FILENAME = (
    "pilot_trial_effect_based_required_participants_summary.csv"
)
EFFECT_ASSUMPTION = (
    "single_participant_trial_effect_assumed_as_true_participant_effect"
)
TRIAL_PAIRING_METHOD = "within_cell_order_by_trial_id_or_source_row"
POWER_METHOD = "two_sided_one_sample_noncentral_t"


class PilotPowerAnalysisError(ValueError):
    """単一参加者パイロット解析の入力または対応関係が不正な場合の例外。"""


@dataclass(frozen=True)
class PilotPowerAnalysisResult:
    """試行由来効果量の詳細表と、最大必要人数の要約表。"""

    details: pd.DataFrame
    summary: pd.DataFrame


def required_participants_from_assumed_effect(
    effect_size: float,
    *,
    alpha: float = ALPHA,
    target_power: float = DEFAULT_TARGET_POWER,
    max_participants: int = DEFAULT_MAX_PARTICIPANTS,
) -> int | float:
    """試行由来dzを真の参加者dzと仮定し、必要参加者数を返す。

    対応のあるt検定は参加者内差分の1標本t検定として扱う。候補人数Nごとに
    非心度 ``abs(dz) * sqrt(N)`` の非心t分布から両側検出力を計算し、
    目標検出力へ初めて到達するNを返す。Holm補正とTOSTは含めない。
    """
    from scipy import stats

    if not 0.0 < float(alpha) < 1.0:
        raise PilotPowerAnalysisError(
            f"alpha must be between 0 and 1: {alpha}"
        )
    if not 0.0 < float(target_power) < 1.0:
        raise PilotPowerAnalysisError(
            f"target_power must be between 0 and 1: {target_power}"
        )
    if int(max_participants) < 2:
        raise PilotPowerAnalysisError(
            "max_participants must be at least 2: "
            f"{max_participants}"
        )
    if np.isnan(effect_size):
        return float("nan")
    if np.isinf(effect_size):
        return 2

    absolute_effect = abs(float(effect_size))
    if np.isclose(absolute_effect, 0.0):
        return float("inf")

    for participant_count in range(2, int(max_participants) + 1):
        degrees_of_freedom = participant_count - 1
        critical_t = stats.t.ppf(
            1.0 - float(alpha) / 2.0,
            degrees_of_freedom,
        )
        noncentrality = absolute_effect * np.sqrt(participant_count)
        power = (
            stats.nct.cdf(
                -critical_t,
                degrees_of_freedom,
                noncentrality,
            )
            + stats.nct.sf(
                critical_t,
                degrees_of_freedom,
                noncentrality,
            )
        )
        if power >= float(target_power):
            return participant_count
    return float("inf")


def _validate_single_participant_trials(
    trials: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """1参加者の完全な4条件×2眼データへ反復番号を付ける。"""
    required = [
        "ID",
        *ANALYSIS_GROUP_COLUMNS,
        "Ocularity",
        "Condition",
        TRIAL_LOG10_COLUMN,
        "Source_Row",
    ]
    missing = [column for column in required if column not in trials.columns]
    if missing:
        raise PilotPowerAnalysisError(
            f"パイロット解析に必要な列が不足しています: {missing}"
        )
    if trials.empty:
        raise PilotPowerAnalysisError("パイロット試行データが空です")

    frame = trials.copy(deep=True)
    for column in ("ID", "Session_Type", "Ocularity", "Condition"):
        values = frame[column].astype("string").str.strip()
        if column == "Ocularity":
            values = values.str.lower()
        frame[column] = values
        if (values.isna() | values.eq("")).any():
            raise PilotPowerAnalysisError(f"{column}に空値があります")

    participant_ids = sorted(set(frame["ID"].astype(str)))
    if len(participant_ids) != 1:
        raise PilotPowerAnalysisError(
            "この解析は1セッション・1参加者専用です: "
            f"found={participant_ids}"
        )
    participant_id = participant_ids[0]

    found_conditions = set(frame["Condition"].astype(str))
    if found_conditions != set(CONDITION_ORDER):
        raise PilotPowerAnalysisError(
            "4条件が完全にそろっていません: "
            f"expected={list(CONDITION_ORDER)}, "
            f"found={sorted(found_conditions)}"
        )
    found_ocularities = set(frame["Ocularity"].astype(str))
    if found_ocularities != set(OCULARITY_ORDER):
        raise PilotPowerAnalysisError(
            "monocular・binocularが完全にそろっていません: "
            f"found={sorted(found_ocularities)}"
        )

    log_values = pd.to_numeric(frame[TRIAL_LOG10_COLUMN], errors="coerce")
    invalid_log = log_values.isna() | ~np.isfinite(
        log_values.to_numpy(dtype=float)
    )
    if invalid_log.any():
        rows = frame.loc[invalid_log, "Source_Row"].tolist()[:10]
        raise PilotPowerAnalysisError(
            f"log10 AR値に有限でない値があります: Source_Row={rows}"
        )
    frame[TRIAL_LOG10_COLUMN] = log_values.astype(float)

    if "Trial_ID" in frame.columns:
        trial_order = pd.to_numeric(frame["Trial_ID"], errors="coerce")
        present = frame["Trial_ID"].notna() & frame["Trial_ID"].astype(
            "string"
        ).str.strip().ne("")
        if present.any() and not present.all():
            raise PilotPowerAnalysisError(
                "Trial_IDは全試行へ設定するか、列自体を省略してください"
            )
        if present.all():
            invalid_order = trial_order.isna() | ~np.isfinite(
                trial_order.to_numpy(dtype=float)
            )
            if invalid_order.any():
                raise PilotPowerAnalysisError(
                    "Trial_IDを有限な数値へ変換できません"
                )
            frame["_Pilot_Trial_Order"] = trial_order.astype(float)
        else:
            frame["_Pilot_Trial_Order"] = pd.to_numeric(
                frame["Source_Row"], errors="raise"
            ).astype(float)
    else:
        frame["_Pilot_Trial_Order"] = pd.to_numeric(
            frame["Source_Row"], errors="raise"
        ).astype(float)

    cell_columns = [
        *ANALYSIS_GROUP_COLUMNS,
        "Ocularity",
        "Condition",
    ]
    frame = frame.sort_values(
        [*cell_columns, "_Pilot_Trial_Order", "Source_Row"],
        kind="stable",
        ignore_index=True,
    )
    frame["Pilot_Repeat_Index"] = (
        frame.groupby(cell_columns, sort=False, dropna=False)
        .cumcount()
        .add(1)
    )

    expected_cells = {
        (ocularity, condition)
        for ocularity in OCULARITY_ORDER
        for condition in CONDITION_ORDER
    }
    for group_values, group in frame.groupby(
        list(ANALYSIS_GROUP_COLUMNS),
        sort=True,
        dropna=False,
    ):
        counts = group.groupby(
            ["Ocularity", "Condition"],
            dropna=False,
        ).size()
        found_cells = set(counts.index.tolist())
        if found_cells != expected_cells:
            raise PilotPowerAnalysisError(
                "解析群の4条件×2眼が不完全です: "
                f"group={group_values}, "
                f"missing={sorted(expected_cells - found_cells)}"
            )
        if counts.nunique() != 1:
            raise PilotPowerAnalysisError(
                "対応付ける試行数が条件間で一致しません: "
                f"group={group_values}, counts={counts.to_dict()}"
            )
        if int(counts.iloc[0]) < 2:
            raise PilotPowerAnalysisError(
                "効果量には各条件2試行以上が必要です: "
                f"group={group_values}"
            )
    return frame, participant_id


def _trial_pivot(group: pd.DataFrame) -> pd.DataFrame:
    """反復番号を行、眼×条件を列とする完全対応表を返す。"""
    duplicated = group.duplicated(
        ["Pilot_Repeat_Index", "Ocularity", "Condition"],
        keep=False,
    )
    if duplicated.any():
        raise PilotPowerAnalysisError(
            "Pilot_Repeat_Index×眼×条件が重複しています"
        )
    pivot = group.pivot(
        index="Pilot_Repeat_Index",
        columns=["Ocularity", "Condition"],
        values=TRIAL_LOG10_COLUMN,
    ).sort_index()
    expected_columns = {
        (ocularity, condition)
        for ocularity in OCULARITY_ORDER
        for condition in CONDITION_ORDER
    }
    if set(pivot.columns.tolist()) != expected_columns:
        raise PilotPowerAnalysisError(
            "試行対応表の4条件×2眼が不完全です"
        )
    if pivot.isna().any().any():
        raise PilotPowerAnalysisError("試行対応表に欠測値があります")
    return pivot


def _cell(
    pivot: pd.DataFrame,
    ocularity: str,
    condition: str,
) -> np.ndarray:
    key = (ocularity, condition)
    if key not in pivot.columns:
        raise PilotPowerAnalysisError(f"試行セルがありません: {key}")
    values = pivot[key].to_numpy(dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise PilotPowerAnalysisError(f"試行セルが有限な1次元値ではありません: {key}")
    return values


def _pilot_effect_statistics(
    differences,
    *,
    alpha: float,
    target_power: float,
    max_participants: int,
) -> dict[str, object]:
    """試行差分のdzと、それを真値とした必要参加者数を返す。"""
    from scipy import stats

    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise PilotPowerAnalysisError(
            "効果量計算には2件以上の有限な試行差分が必要です"
        )

    trial_count = len(values)
    mean_difference = float(np.mean(values))
    sd_difference = float(np.std(values, ddof=1))
    sem_difference = sd_difference / np.sqrt(trial_count)

    if np.isclose(sd_difference, 0.0):
        ci_lower = ci_upper = mean_difference
        if np.isclose(mean_difference, 0.0):
            statistic = 0.0
            p_value = 1.0
            effect_size = 0.0
            status = "zero_sd_zero_effect"
        else:
            statistic = float(np.sign(mean_difference) * np.inf)
            p_value = 0.0
            effect_size = float(np.sign(mean_difference) * np.inf)
            status = "zero_sd_infinite_effect"
    else:
        statistic = mean_difference / sem_difference
        p_value = float(
            2.0 * stats.t.sf(abs(statistic), df=trial_count - 1)
        )
        ci_lower, ci_upper = stats.t.interval(
            1.0 - float(alpha),
            trial_count - 1,
            loc=mean_difference,
            scale=sem_difference,
        )
        effect_size = mean_difference / sd_difference
        status = "ok"

    required = required_participants_from_assumed_effect(
        effect_size,
        alpha=alpha,
        target_power=target_power,
        max_participants=max_participants,
    )
    return {
        "Pilot_Trial_Count": trial_count,
        "Pilot_Mean_Log10_Difference": mean_difference,
        "Pilot_SD_Log10_Difference": sd_difference,
        "Pilot_SEM_Log10_Difference": sem_difference,
        "Pilot_Mean_CI_Level": 1.0 - float(alpha),
        "Pilot_Mean_CI_Log10_Lower": float(ci_lower),
        "Pilot_Mean_CI_Log10_Upper": float(ci_upper),
        "Pilot_Geometric_Mean_Ratio": float(10.0**mean_difference),
        "Pilot_Mean_CI_Ratio_Lower": float(10.0**ci_lower),
        "Pilot_Mean_CI_Ratio_Upper": float(10.0**ci_upper),
        "Pilot_Trial_T_Statistic": float(statistic),
        "Pilot_Trial_Degrees_Of_Freedom": trial_count - 1,
        "Pilot_Trial_P_Value_Two_Sided": float(p_value),
        "Pilot_Trial_Cohens_Dz": float(effect_size),
        "Assumed_True_Participant_Cohens_Dz": float(effect_size),
        "Required_Participants_For_Target_Power": required,
        "Effect_Estimate_Status": status,
    }


def _result_row(
    *,
    participant_id: str,
    metadata: dict[str, object],
    hypothesis: str,
    component: str,
    family: str,
    comparison: str,
    condition: str,
    baseline_condition: str,
    ocularity: str,
    current_primary_test: str,
    difference_definition: str,
    differences,
    alpha: float,
    target_power: float,
    max_participants: int,
) -> dict[str, object]:
    statistics = _pilot_effect_statistics(
        differences,
        alpha=alpha,
        target_power=target_power,
        max_participants=max_participants,
    )
    return {
        "Pilot_ID": participant_id,
        "Hypothesis": hypothesis,
        "Component": component,
        "Family": family,
        **metadata,
        "Ocularity": ocularity,
        "Comparison": comparison,
        "Condition": condition,
        "Baseline_Condition": baseline_condition,
        "Current_Primary_Test": current_primary_test,
        "Sample_Size_Target_Test": "two_sided_t_component",
        "Difference_Definition": difference_definition,
        "Analysis_Scale": "log10_AR_Matched_Contrast",
        "Trial_Pairing_Method": TRIAL_PAIRING_METHOD,
        **statistics,
        "Power_Method": POWER_METHOD,
        "Power_Alpha": float(alpha),
        "Target_Power": float(target_power),
        "Maximum_Participants_Searched": int(max_participants),
        "Multiple_Comparison_Adjustment_In_Power": "none",
        "TOST_Sample_Size_Included": False,
        "Effect_Assumption": EFFECT_ASSUMPTION,
    }


def _build_group_rows(
    *,
    participant_id: str,
    metadata: dict[str, object],
    pivot: pd.DataFrame,
    alpha: float,
    target_power: float,
    max_participants: int,
) -> list[dict[str, object]]:
    """1解析群について現行H1〜H4と同じ15効果を試行から作る。"""
    reference_log10 = float(np.log10(float(metadata["Ref_Contrast"])))
    h1: list[dict[str, object]] = []
    h2: list[dict[str, object]] = []
    h3: list[dict[str, object]] = []
    h4: list[dict[str, object]] = []

    def append(
        target: list[dict[str, object]],
        *,
        hypothesis: str,
        component: str,
        family: str,
        comparison: str,
        condition: str,
        baseline_condition: str,
        ocularity: str,
        current_primary_test: str,
        difference_definition: str,
        differences,
    ) -> None:
        target.append(
            _result_row(
                participant_id=participant_id,
                metadata=metadata,
                hypothesis=hypothesis,
                component=component,
                family=family,
                comparison=comparison,
                condition=condition,
                baseline_condition=baseline_condition,
                ocularity=ocularity,
                current_primary_test=current_primary_test,
                difference_definition=difference_definition,
                differences=differences,
                alpha=alpha,
                target_power=target_power,
                max_participants=max_participants,
            )
        )

    # H1: DPF試行と参照コントラストのlog10差。
    for ocularity in OCULARITY_ORDER:
        dpf = _cell(pivot, ocularity, DPF_CONDITION)
        append(
            h1,
            hypothesis="H1",
            component=f"H1_{ocularity}",
            family="H1_DPF_vs_reference",
            comparison=f"{DPF_CONDITION} vs Reference contrast",
            condition=DPF_CONDITION,
            baseline_condition="Reference contrast",
            ocularity=ocularity,
            current_primary_test="dual_inference",
            difference_definition="log10(DPF_trial) - log10(Ref)",
            differences=dpf - reference_log10,
        )

    # H2: 試行ごとのCondition-DPF。DPF補正後値とRefとの差に等しい。
    for ocularity in OCULARITY_ORDER:
        dpf = _cell(pivot, ocularity, DPF_CONDITION)
        for condition in CORRECTED_CONDITIONS:
            condition_values = _cell(pivot, ocularity, condition)
            append(
                h2,
                hypothesis="H2",
                component=f"H2_{ocularity}_{condition}",
                family="H2_corrected_conditions_vs_reference",
                comparison=f"DPF-corrected {condition} vs Reference contrast",
                condition=condition,
                baseline_condition=DPF_CONDITION,
                ocularity=ocularity,
                current_primary_test="t",
                difference_definition=(
                    f"log10({condition}_trial) - log10(DPF_trial)"
                ),
                differences=condition_values - dpf,
            )

    # H3: DPFは差し引きで相殺されるため、SPD/SPまたはDP/SPを直接作る。
    for ocularity in OCULARITY_ORDER:
        sp = _cell(pivot, ocularity, SP_CONDITION)
        for condition in (SPD_CONDITION, DP_CONDITION):
            condition_values = _cell(pivot, ocularity, condition)
            append(
                h3,
                hypothesis="H3",
                component=f"H3_{ocularity}_{condition}",
                family="H3_SP_vs_SPD_DP",
                comparison=f"{condition} vs {SP_CONDITION}",
                condition=condition,
                baseline_condition=SP_CONDITION,
                ocularity=ocularity,
                current_primary_test="t",
                difference_definition=(
                    f"log10({condition}_trial) - log10({SP_CONDITION}_trial)"
                ),
                differences=condition_values - sp,
            )

    eye_effects: dict[str, np.ndarray] = {}
    for ocularity in OCULARITY_ORDER:
        dp = _cell(pivot, ocularity, DP_CONDITION)
        spd = _cell(pivot, ocularity, SPD_CONDITION)
        effect = dp - spd
        eye_effects[ocularity] = effect
        monocular = ocularity == "monocular"
        append(
            h4,
            hypothesis="H4",
            component="H4-2" if monocular else "H4-3",
            family="H4_SPD_DP_ocularity",
            comparison=f"{DP_CONDITION} vs {SPD_CONDITION}",
            condition=DP_CONDITION,
            baseline_condition=SPD_CONDITION,
            ocularity=ocularity,
            current_primary_test="TOST" if monocular else "t",
            difference_definition=(
                "log10(DP_trial) - log10(SPD_trial)"
            ),
            differences=effect,
        )

    interaction = eye_effects["binocular"] - eye_effects["monocular"]
    append(
        h4,
        hypothesis="H4",
        component="H4-1",
        family="H4_SPD_DP_ocularity",
        comparison="(DP/SPD)_binocular vs (DP/SPD)_monocular",
        condition="binocular log10(DP/SPD)",
        baseline_condition="monocular log10(DP/SPD)",
        ocularity="binocular_minus_monocular",
        current_primary_test="t",
        difference_definition=(
            "[log10(DP)-log10(SPD)]_binocular - "
            "[log10(DP)-log10(SPD)]_monocular"
        ),
        differences=interaction,
    )
    h4.sort(key=lambda row: str(row["Component"]))

    rows = [*h1, *h2, *h3, *h4]
    if len(rows) != EXPECTED_ROWS_PER_ANALYSIS_GROUP:
        raise RuntimeError(
            "パイロット効果量の行数が不正です: "
            f"expected={EXPECTED_ROWS_PER_ANALYSIS_GROUP}, actual={len(rows)}"
        )
    return rows


def _limiting_label(row: pd.Series) -> str:
    return (
        f"{row['Hypothesis']}/{row['Component']}"
        f"/ref={float(row['Ref_Contrast']):g}"
        f"/ori={float(row['Orientation']):g}"
    )


def _build_summary(
    details: pd.DataFrame,
    *,
    participant_id: str,
    total_trial_rows: int,
    dropout_rate: float,
) -> pd.DataFrame:
    """全t成分の最大値から完遂人数と募集人数を1行で返す。"""
    if not 0.0 <= float(dropout_rate) < 1.0:
        raise PilotPowerAnalysisError(
            f"dropout_rate must be in [0, 1): {dropout_rate}"
        )
    required = pd.to_numeric(
        details["Required_Participants_For_Target_Power"],
        errors="coerce",
    ).to_numpy(dtype=float)

    if np.isnan(required).any():
        complete = float("nan")
        recruit = float("nan")
        status = "not_evaluable"
        limiting = ""
    elif np.isinf(required).any():
        complete = float("inf")
        recruit = float("inf")
        status = "infinite_required_n_present"
        limiting = "; ".join(
            _limiting_label(row)
            for _, row in details.loc[np.isinf(required)].iterrows()
        )
    else:
        complete = int(np.max(required))
        recruit = int(math.ceil(complete / (1.0 - float(dropout_rate))))
        status = "ok"
        limiting_mask = np.isclose(required, float(complete))
        limiting = "; ".join(
            _limiting_label(row)
            for _, row in details.loc[limiting_mask].iterrows()
        )

    return pd.DataFrame(
        [
            {
                "Pilot_ID": participant_id,
                "Pilot_Total_Trial_Rows": int(total_trial_rows),
                "Analysis_Group_Count": int(
                    len(
                        details.loc[:, list(ANALYSIS_GROUP_COLUMNS)]
                        .drop_duplicates()
                    )
                ),
                "Effect_Estimate_Count": int(len(details)),
                "Scope": (
                    "maximum_across_all_H1_H4_two_sided_t_components_"
                    "including_auxiliary_t_tests"
                ),
                "Required_Complete_Participants": complete,
                "Dropout_Rate": float(dropout_rate),
                "Required_Recruited_Participants": recruit,
                "Limiting_Comparisons": limiting,
                "Summary_Status": status,
                "Power_Alpha": float(details["Power_Alpha"].iloc[0]),
                "Target_Power": float(details["Target_Power"].iloc[0]),
                "Multiple_Comparison_Adjustment_In_Power": "none",
                "TOST_Sample_Size_Included": False,
                "Effect_Assumption": EFFECT_ASSUMPTION,
            }
        ]
    )


def run_pilot_power_analysis(
    trials: pd.DataFrame,
    *,
    alpha: float = ALPHA,
    target_power: float = DEFAULT_TARGET_POWER,
    max_participants: int = DEFAULT_MAX_PARTICIPANTS,
    dropout_rate: float = 0.0,
) -> PilotPowerAnalysisResult:
    """1人の反復試行効果を真値と仮定した必要参加者数を返す。"""
    prepared, participant_id = _validate_single_participant_trials(trials)
    rows: list[dict[str, object]] = []
    group_count = 0
    for group_values, group in prepared.groupby(
        list(ANALYSIS_GROUP_COLUMNS),
        sort=True,
        dropna=False,
    ):
        group_count += 1
        metadata = dict(zip(ANALYSIS_GROUP_COLUMNS, tuple(group_values)))
        if float(metadata["Ref_Contrast"]) <= 0:
            raise PilotPowerAnalysisError(
                f"Ref_Contrast > 0が必要です: {metadata['Ref_Contrast']}"
            )
        rows.extend(
            _build_group_rows(
                participant_id=participant_id,
                metadata=metadata,
                pivot=_trial_pivot(group),
                alpha=float(alpha),
                target_power=float(target_power),
                max_participants=int(max_participants),
            )
        )

    details = pd.DataFrame(rows).reset_index(drop=True)
    expected_rows = group_count * EXPECTED_ROWS_PER_ANALYSIS_GROUP
    if len(details) != expected_rows:
        raise RuntimeError(
            "パイロット解析表の総行数が不正です: "
            f"expected={expected_rows}, actual={len(details)}"
        )
    summary = _build_summary(
        details,
        participant_id=participant_id,
        total_trial_rows=len(prepared),
        dropout_rate=float(dropout_rate),
    )
    return PilotPowerAnalysisResult(details=details, summary=summary)


def save_pilot_power_outputs(
    result: PilotPowerAnalysisResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """詳細結果と最大必要人数をUTF-8 BOM付きCSVで保存する。"""
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    detail_path = destination / DETAIL_FILENAME
    summary_path = destination / SUMMARY_FILENAME
    result.details.to_csv(detail_path, index=False, encoding="utf-8-sig")
    result.summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return {"details": detail_path, "summary": summary_path}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "1名の予備実験の試行単位dzを真の参加者dzと仮定し、"
            "両側t検定の必要参加者数を推定します。"
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help=(
            "contrast_matching.csvを含む1セッションフォルダ、"
            "またはそのCSVファイル。"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "CSV保存先。省略時は入力セッション内の"
            "pilot_power_analysisフォルダ。"
        ),
    )
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument(
        "--target-power",
        type=float,
        default=DEFAULT_TARGET_POWER,
    )
    parser.add_argument(
        "--max-participants",
        type=int,
        default=DEFAULT_MAX_PARTICIPANTS,
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.0,
        help="募集人数へ反映する脱落率。例: 0.10。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        session_dirs = discover_session_dirs([args.input_path])
        if len(session_dirs) != 1:
            raise PilotPowerAnalysisError(
                "1つのセッションだけを指定してください: "
                f"found={len(session_dirs)}"
            )
        loaded = load_trials(session_dirs)
        result = run_pilot_power_analysis(
            loaded.trials,
            alpha=args.alpha,
            target_power=args.target_power,
            max_participants=args.max_participants,
            dropout_rate=args.dropout_rate,
        )
        output_dir = args.output_dir or (
            session_dirs[0] / "pilot_power_analysis"
        )
        files = save_pilot_power_outputs(result, output_dir)
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
        print(f"Pilot power analysis failed: {error}", file=sys.stderr)
        return 1

    print(f"Saved pilot effect details: {files['details']}")
    print(f"Saved participant requirement summary: {files['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DETAIL_FILENAME",
    "EFFECT_ASSUMPTION",
    "PilotPowerAnalysisError",
    "PilotPowerAnalysisResult",
    "SUMMARY_FILENAME",
    "build_argument_parser",
    "main",
    "required_participants_from_assumed_effect",
    "run_pilot_power_analysis",
    "save_pilot_power_outputs",
]