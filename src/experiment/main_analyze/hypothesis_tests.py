"""未補正・DPF補正後の参加者集約値からH1〜H4を検定する。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    ALPHA,
    ANALYSIS_GROUP_COLUMNS,
    ANALYSIS_SCALE,
    CORRECTED_CONDITIONS,
    DP_CONDITION,
    DPF_CONDITION,
    LOG10_EQUIVALENCE_MARGIN,
    LOG10_INTERACTION_EQUIVALENCE_MARGIN,
    MEAN_LOG10_COLUMN,
    OCULARITY_ORDER,
    PARTICIPANT_AGGREGATION,
    SP_CONDITION,
    SPD_CONDITION,
)
from .dpf_correction import CORRECTED_LOG10_COLUMN


HYPOTHESIS_ROW_COUNTS = {"H1": 2, "H2": 6, "H3": 4, "H4": 3}
EXPECTED_ROWS_PER_ANALYSIS_GROUP = sum(HYPOTHESIS_ROW_COUNTS.values())


class HypothesisTestError(ValueError):
    """仮説検定の入力構造または参加者対応が不正な場合の例外。"""


def holm_adjust(p_values) -> np.ndarray:
    """評価不能値も予定ファミリー数へ数えたHolm補正値を返す。"""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.size, np.nan)
    valid = [index for index, value in enumerate(values) if np.isfinite(value)]
    ordered = sorted(valid, key=lambda index: values[index])
    running = 0.0
    for rank, index in enumerate(ordered):
        running = max(running, (values.size - rank) * float(values[index]))
        adjusted[index] = min(running, 1.0)
    return adjusted


def _pow10(value: float) -> float:
    return float(10.0**value) if np.isfinite(value) else float("nan")


def _difference_statistics(values) -> dict[str, float | int]:
    """log10差分の記述統計と両側1標本t検定。"""
    from scipy import stats

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise HypothesisTestError("検定差分には1次元の有限値が必要です")
    n = len(array)
    mean = float(np.mean(array)) if n else float("nan")
    result: dict[str, float | int] = {
        "n": n,
        "mean_log10_difference": mean,
        "sd_log10_difference": float("nan"),
        "sem_log10_difference": float("nan"),
        "ci95_log10_lower": float("nan"),
        "ci95_log10_upper": float("nan"),
        "t_statistic": float("nan"),
        "degrees_of_freedom": n - 1 if n else float("nan"),
        "p_value_two_sided": float("nan"),
        "cohens_dz": float("nan"),
    }
    if n < 2:
        return result

    sd = float(np.std(array, ddof=1))
    sem = sd / np.sqrt(n)
    if np.isclose(sem, 0.0):
        if np.isclose(mean, 0.0):
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = float(np.sign(mean) * np.inf), 0.0
        ci_lower = ci_upper = mean
    else:
        statistic = mean / sem
        p_value = float(2.0 * stats.t.sf(abs(statistic), df=n - 1))
        ci_lower, ci_upper = stats.t.interval(
            0.95, n - 1, loc=mean, scale=sem
        )
    if np.isclose(sd, 0.0):
        effect_size = 0.0 if np.isclose(mean, 0.0) else float(np.sign(mean) * np.inf)
    else:
        effect_size = mean / sd
    result.update(
        {
            "sd_log10_difference": sd,
            "sem_log10_difference": sem,
            "ci95_log10_lower": float(ci_lower),
            "ci95_log10_upper": float(ci_upper),
            "t_statistic": float(statistic),
            "p_value_two_sided": p_value,
            "cohens_dz": effect_size,
        }
    )
    return result


def _empty_tost() -> dict[str, float]:
    return {
        "equivalence_margin_log10": float("nan"),
        "equivalence_ratio_lower": float("nan"),
        "equivalence_ratio_upper": float("nan"),
        "tost_t_lower": float("nan"),
        "tost_t_upper": float("nan"),
        "tost_p_lower": float("nan"),
        "tost_p_upper": float("nan"),
        "tost_p_value": float("nan"),
        "tost_ci90_log10_lower": float("nan"),
        "tost_ci90_log10_upper": float("nan"),
    }


def _tost_statistics(values, margin: float) -> dict[str, float]:
    """平均log10差が[-margin,+margin]内かをTOSTで検定する。"""
    from scipy import stats

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise HypothesisTestError("TOST差分には1次元の有限値が必要です")
    bound = float(10.0**margin)
    result = _empty_tost()
    result.update(
        {
            "equivalence_margin_log10": float(margin),
            "equivalence_ratio_lower": 1.0 / bound,
            "equivalence_ratio_upper": bound,
        }
    )
    n = len(array)
    if n < 2:
        return result
    mean = float(np.mean(array))
    sem = float(np.std(array, ddof=1)) / np.sqrt(n)
    if np.isclose(sem, 0.0):
        def zero_sem_t(numerator: float) -> float:
            return 0.0 if np.isclose(numerator, 0.0) else float(np.sign(numerator) * np.inf)
        t_lower = zero_sem_t(mean + margin)
        t_upper = zero_sem_t(mean - margin)
        ci_lower = ci_upper = mean
    else:
        t_lower = (mean + margin) / sem
        t_upper = (mean - margin) / sem
        ci_lower, ci_upper = stats.t.interval(
            1.0 - 2.0 * ALPHA, n - 1, loc=mean, scale=sem
        )
    p_lower = float(stats.t.sf(t_lower, df=n - 1))
    p_upper = float(stats.t.cdf(t_upper, df=n - 1))
    result.update(
        {
            "tost_t_lower": float(t_lower),
            "tost_t_upper": float(t_upper),
            "tost_p_lower": p_lower,
            "tost_p_upper": p_upper,
            "tost_p_value": max(p_lower, p_upper),
            "tost_ci90_log10_lower": float(ci_lower),
            "tost_ci90_log10_upper": float(ci_upper),
        }
    )
    return result


def _validate_summary(
    frame: pd.DataFrame,
    *,
    label: str,
    conditions: tuple[str, ...],
    value_column: str,
) -> pd.DataFrame:
    required = [
        "ID",
        *ANALYSIS_GROUP_COLUMNS,
        "Ocularity",
        "Condition",
        "Participant_Aggregation",
        value_column,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise HypothesisTestError(f"{label}に必要な列が不足しています: {missing}")
    if frame.empty:
        raise HypothesisTestError(f"{label}が空です")
    validated = frame.copy(deep=True)
    for column in ("ID", "Session_Type", "Ocularity", "Condition"):
        validated[column] = validated[column].astype("string").str.strip()
        invalid = validated[column].isna() | validated[column].eq("")
        if invalid.any():
            raise HypothesisTestError(f"{label}の{column}に空値があります")
    for column in ("Ref_Contrast", "Orientation", value_column):
        values = pd.to_numeric(validated[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise HypothesisTestError(f"{label}の{column}に有限でない値があります")
        validated[column] = values.astype(float)
    if (validated["Ref_Contrast"] <= 0).any():
        raise HypothesisTestError(f"{label}にはRef_Contrast > 0が必要です")
    aggregations = set(validated["Participant_Aggregation"].astype(str))
    if aggregations != {PARTICIPANT_AGGREGATION}:
        raise HypothesisTestError(f"{label}の参加者集約方法が不正です: {aggregations}")
    if set(validated["Condition"].astype(str)) != set(conditions):
        raise HypothesisTestError(
            f"{label}の条件が不正です: expected={list(conditions)}, "
            f"found={sorted(set(validated['Condition'].astype(str)))}"
        )
    if set(validated["Ocularity"].astype(str)) != set(OCULARITY_ORDER):
        raise HypothesisTestError(f"{label}の眼条件が不正です")
    key_columns = ["ID", *ANALYSIS_GROUP_COLUMNS, "Ocularity", "Condition"]
    if validated.duplicated(key_columns, keep=False).any():
        raise HypothesisTestError(f"{label}に参加者×条件の重複行があります")
    for group_values, group in validated.groupby(
        list(ANALYSIS_GROUP_COLUMNS), sort=True, dropna=False
    ):
        participant_ids = set(group["ID"].astype(str))
        for ocularity in OCULARITY_ORDER:
            for condition in conditions:
                ids = set(
                    group.loc[
                        (group["Ocularity"] == ocularity)
                        & (group["Condition"] == condition),
                        "ID",
                    ].astype(str)
                )
                if ids != participant_ids:
                    raise HypothesisTestError(
                        f"{label}の参加者対応が不完全です: group={group_values}, "
                        f"ocularity={ocularity}, condition={condition}, "
                        f"missing={sorted(participant_ids - ids)}"
                    )
    return validated


def _group_keys(frame: pd.DataFrame) -> set[tuple[object, ...]]:
    return set(
        frame.loc[:, list(ANALYSIS_GROUP_COLUMNS)]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )


def _cell(
    group: pd.DataFrame,
    condition: str,
    ocularity: str,
    value_column: str,
) -> pd.DataFrame:
    return (
        group.loc[
            (group["Condition"] == condition)
            & (group["Ocularity"] == ocularity),
            ["ID", value_column],
        ]
        .rename(columns={value_column: "value"})
        .sort_values("ID", kind="stable", ignore_index=True)
    )


def _paired(
    group: pd.DataFrame,
    condition: str,
    baseline: str,
    ocularity: str,
    value_column: str,
) -> pd.DataFrame:
    left = _cell(group, condition, ocularity, value_column).rename(
        columns={"value": "condition_value"}
    )
    right = _cell(group, baseline, ocularity, value_column).rename(
        columns={"value": "baseline_value"}
    )
    merged = left.merge(
        right,
        on="ID",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise HypothesisTestError(f"{condition}と{baseline}の参加者対応が不完全です")
    return merged.drop(columns="_merge").sort_values("ID", ignore_index=True)


def _build_row(
    *,
    metadata: dict[str, object],
    hypothesis: str,
    component: str,
    family: str,
    data_state: str,
    comparison: str,
    test_type: str,
    primary_test: str,
    condition: str,
    baseline_condition: str,
    ocularity: str,
    condition_values,
    baseline_values,
    difference_values,
    equivalence_margin: float | None = None,
    effect_scale: str = "log10_condition_ratio",
) -> dict[str, object]:
    condition_array = np.asarray(condition_values, dtype=float)
    baseline_array = np.asarray(baseline_values, dtype=float)
    difference_array = np.asarray(difference_values, dtype=float)
    if not (
        condition_array.ndim == baseline_array.ndim == difference_array.ndim == 1
        and len(condition_array) == len(baseline_array) == len(difference_array)
        and np.isfinite(condition_array).all()
        and np.isfinite(baseline_array).all()
        and np.isfinite(difference_array).all()
    ):
        raise HypothesisTestError("検定値の参加者対応または有限性が不正です")
    stats = _difference_statistics(difference_array)
    tost = _tost_statistics(difference_array, equivalence_margin) if equivalence_margin is not None else _empty_tost()
    mean_condition = float(np.mean(condition_array)) if len(condition_array) else float("nan")
    mean_baseline = float(np.mean(baseline_array)) if len(baseline_array) else float("nan")
    return {
        "Hypothesis": hypothesis,
        "Component": component,
        "Family": family,
        **metadata,
        "Ocularity": ocularity,
        "Data_State": data_state,
        "Comparison": comparison,
        "Test_Type": test_type,
        "Primary_Test": primary_test,
        "Condition": condition,
        "Baseline_Condition": baseline_condition,
        "Effect_Scale": effect_scale,
        "Analysis_Scale": ANALYSIS_SCALE,
        "Participant_Aggregation": PARTICIPANT_AGGREGATION,
        "mean_log10_condition": mean_condition,
        "mean_log10_baseline": mean_baseline,
        "geometric_mean_condition": _pow10(mean_condition),
        "geometric_mean_baseline": _pow10(mean_baseline),
        **stats,
        "geometric_mean_ratio": _pow10(float(stats["mean_log10_difference"])),
        "ci95_ratio_lower": _pow10(float(stats["ci95_log10_lower"])),
        "ci95_ratio_upper": _pow10(float(stats["ci95_log10_upper"])),
        **tost,
        "tost_ci90_ratio_lower": _pow10(float(tost["tost_ci90_log10_lower"])),
        "tost_ci90_ratio_upper": _pow10(float(tost["tost_ci90_log10_upper"])),
        "holm_adjusted_p_value": float("nan"),
        "significant_holm_alpha_0_05": False,
        "holm_adjusted_tost_p_value": float("nan"),
        "equivalent_holm_alpha_0_05": False,
        "Primary_P_Value": float("nan"),
        "Primary_Adjusted_P_Value": float("nan"),
        "Primary_Pass_Alpha_0_05": False,
        "Conclusion_Code": "not_evaluable",
        "H4_Conjunction_Evaluable": False,
        "H4_Conjunction_All_Pass": False,
    }


def _apply_holm(
    rows: list[dict[str, object]],
    p_column: str,
    adjusted_column: str,
    decision_column: str,
) -> None:
    adjusted = holm_adjust([row[p_column] for row in rows])
    for row, value in zip(rows, adjusted):
        row[adjusted_column] = float(value)
        row[decision_column] = bool(np.isfinite(value) and value < ALPHA)


def _dual_conclusion(row: dict[str, object]) -> str:
    t_value = float(row["holm_adjusted_p_value"])
    tost_value = float(row["holm_adjusted_tost_p_value"])
    if not np.isfinite(t_value) or not np.isfinite(tost_value):
        return "not_evaluable"
    different, equivalent = t_value < ALPHA, tost_value < ALPHA
    if different and equivalent:
        return "different_but_within_equivalence_bounds"
    if different:
        return "different_not_equivalent"
    if equivalent:
        return "equivalent_no_detected_difference"
    return "inconclusive"


def _set_primary(
    h1: list[dict[str, object]],
    h2: list[dict[str, object]],
    h3: list[dict[str, object]],
    h4: list[dict[str, object]],
) -> None:
    for row in [*h1, *h2]:
        row["Conclusion_Code"] = _dual_conclusion(row)
    for row in h3:
        raw = float(row["p_value_two_sided"])
        adjusted = float(row["holm_adjusted_p_value"])
        row["Primary_P_Value"] = raw
        row["Primary_Adjusted_P_Value"] = adjusted
        row["Primary_Pass_Alpha_0_05"] = bool(np.isfinite(adjusted) and adjusted < ALPHA)
        row["Conclusion_Code"] = (
            "difference_detected"
            if row["Primary_Pass_Alpha_0_05"]
            else "no_detected_difference"
            if np.isfinite(adjusted)
            else "not_evaluable"
        )
    for row in h4:
        if row["Primary_Test"] == "TOST":
            raw = float(row["tost_p_value"])
            adjusted = float(row["holm_adjusted_tost_p_value"])
            success = "equivalence_supported"
        else:
            raw = float(row["p_value_two_sided"])
            adjusted = float(row["holm_adjusted_p_value"])
            success = "difference_detected"
        row["Primary_P_Value"] = raw
        row["Primary_Adjusted_P_Value"] = adjusted
        # IUTでは各主判定の生p値を使用する。Holm値は単体報告用。
        row["Primary_Pass_Alpha_0_05"] = bool(np.isfinite(raw) and raw < ALPHA)
        row["Conclusion_Code"] = success if row["Primary_Pass_Alpha_0_05"] else "primary_test_not_supported" if np.isfinite(raw) else "not_evaluable"
    evaluable = len(h4) == 3 and all(np.isfinite(float(row["Primary_P_Value"])) for row in h4)
    all_pass = evaluable and all(bool(row["Primary_Pass_Alpha_0_05"]) for row in h4)
    for row in h4:
        row["H4_Conjunction_Evaluable"] = evaluable
        row["H4_Conjunction_All_Pass"] = all_pass


def run_hypothesis_tests(
    uncorrected_df: pd.DataFrame,
    corrected_df: pd.DataFrame,
) -> pd.DataFrame:
    """解析計画どおり、1解析群あたりH1〜H4の15行を返す。"""
    uncorrected = _validate_summary(
        uncorrected_df,
        label="未補正参加者表",
        conditions=(*CORRECTED_CONDITIONS, DPF_CONDITION),
        value_column=MEAN_LOG10_COLUMN,
    )
    corrected = _validate_summary(
        corrected_df,
        label="DPF補正後参加者表",
        conditions=CORRECTED_CONDITIONS,
        value_column=CORRECTED_LOG10_COLUMN,
    )
    if _group_keys(uncorrected) != _group_keys(corrected):
        raise HypothesisTestError("未補正表と補正後表の解析群が一致しません")
    corrected_groups = {
        tuple(key): group
        for key, group in corrected.groupby(
            list(ANALYSIS_GROUP_COLUMNS), sort=True, dropna=False
        )
    }
    all_rows: list[dict[str, object]] = []
    for key, raw_group in uncorrected.groupby(
        list(ANALYSIS_GROUP_COLUMNS), sort=True, dropna=False
    ):
        group_key = tuple(key)
        fixed_group = corrected_groups[group_key]
        if set(raw_group["ID"].astype(str)) != set(fixed_group["ID"].astype(str)):
            raise HypothesisTestError(f"未補正表と補正後表の参加者が一致しません: {group_key}")
        metadata = dict(zip(ANALYSIS_GROUP_COLUMNS, group_key))
        reference_log10 = float(np.log10(float(metadata["Ref_Contrast"])))
        h1: list[dict[str, object]] = []
        h2: list[dict[str, object]] = []
        h3: list[dict[str, object]] = []
        h4: list[dict[str, object]] = []

        for ocularity in OCULARITY_ORDER:
            values = _cell(raw_group, DPF_CONDITION, ocularity, MEAN_LOG10_COLUMN)["value"].to_numpy(float)
            baseline = np.full(len(values), reference_log10)
            h1.append(
                _build_row(
                    metadata=metadata,
                    hypothesis="H1",
                    component=f"H1_{ocularity}",
                    family="H1_DPF_vs_reference",
                    data_state="uncorrected",
                    comparison=f"{DPF_CONDITION} vs Reference contrast",
                    test_type="one_sample_t_and_tost",
                    primary_test="dual_inference",
                    condition=DPF_CONDITION,
                    baseline_condition="Reference contrast",
                    ocularity=ocularity,
                    condition_values=values,
                    baseline_values=baseline,
                    difference_values=values - reference_log10,
                    equivalence_margin=LOG10_EQUIVALENCE_MARGIN,
                )
            )

        for ocularity in OCULARITY_ORDER:
            for condition in CORRECTED_CONDITIONS:
                values = _cell(fixed_group, condition, ocularity, CORRECTED_LOG10_COLUMN)["value"].to_numpy(float)
                baseline = np.full(len(values), reference_log10)
                h2.append(
                    _build_row(
                        metadata=metadata,
                        hypothesis="H2",
                        component=f"H2_{ocularity}_{condition}",
                        family="H2_corrected_conditions_vs_reference",
                        data_state="dpf_corrected",
                        comparison=f"{condition} vs Reference contrast",
                        test_type="one_sample_t_and_tost",
                        primary_test="dual_inference",
                        condition=condition,
                        baseline_condition="Reference contrast",
                        ocularity=ocularity,
                        condition_values=values,
                        baseline_values=baseline,
                        difference_values=values - reference_log10,
                        equivalence_margin=LOG10_EQUIVALENCE_MARGIN,
                    )
                )

        for ocularity in OCULARITY_ORDER:
            for condition in (SPD_CONDITION, DP_CONDITION):
                paired = _paired(fixed_group, condition, SP_CONDITION, ocularity, CORRECTED_LOG10_COLUMN)
                condition_values = paired["condition_value"].to_numpy(float)
                baseline_values = paired["baseline_value"].to_numpy(float)
                h3.append(
                    _build_row(
                        metadata=metadata,
                        hypothesis="H3",
                        component=f"H3_{ocularity}_{condition}",
                        family="H3_SP_vs_SPD_DP",
                        data_state="dpf_corrected",
                        comparison=f"{condition} vs {SP_CONDITION}",
                        test_type="paired_t",
                        primary_test="t",
                        condition=condition,
                        baseline_condition=SP_CONDITION,
                        ocularity=ocularity,
                        condition_values=condition_values,
                        baseline_values=baseline_values,
                        difference_values=condition_values - baseline_values,
                    )
                )

        eye_effects: dict[str, pd.DataFrame] = {}
        for ocularity in OCULARITY_ORDER:
            paired = _paired(fixed_group, DP_CONDITION, SPD_CONDITION, ocularity, CORRECTED_LOG10_COLUMN)
            condition_values = paired["condition_value"].to_numpy(float)
            baseline_values = paired["baseline_value"].to_numpy(float)
            paired["effect"] = condition_values - baseline_values
            eye_effects[ocularity] = paired[["ID", "effect"]]
            monocular = ocularity == "monocular"
            h4.append(
                _build_row(
                    metadata=metadata,
                    hypothesis="H4",
                    component="H4-1" if monocular else "H4-2",
                    family="H4_SPD_DP_ocularity",
                    data_state="dpf_corrected",
                    comparison=f"{DP_CONDITION} vs {SPD_CONDITION}",
                    test_type="paired_t_and_tost",
                    primary_test="TOST" if monocular else "t",
                    condition=DP_CONDITION,
                    baseline_condition=SPD_CONDITION,
                    ocularity=ocularity,
                    condition_values=condition_values,
                    baseline_values=baseline_values,
                    difference_values=condition_values - baseline_values,
                    equivalence_margin=LOG10_EQUIVALENCE_MARGIN,
                )
            )

        interaction = eye_effects["binocular"].rename(columns={"effect": "bino"}).merge(
            eye_effects["monocular"].rename(columns={"effect": "mono"}),
            on="ID",
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        if not interaction["_merge"].eq("both").all():
            raise HypothesisTestError("H4 interactionの眼間参加者対応が不完全です")
        interaction = interaction.drop(columns="_merge").sort_values("ID", ignore_index=True)
        bino = interaction["bino"].to_numpy(float)
        mono = interaction["mono"].to_numpy(float)
        h4.append(
            _build_row(
                metadata=metadata,
                hypothesis="H4",
                component="H4-3",
                family="H4_SPD_DP_ocularity",
                data_state="dpf_corrected",
                comparison="(DP/SPD)_binocular vs (DP/SPD)_monocular",
                test_type="paired_t_and_tost",
                primary_test="t",
                condition="binocular log10(DP/SPD)",
                baseline_condition="monocular log10(DP/SPD)",
                ocularity="binocular_minus_monocular",
                condition_values=bino,
                baseline_values=mono,
                difference_values=bino - mono,
                equivalence_margin=LOG10_INTERACTION_EQUIVALENCE_MARGIN,
                effect_scale="log10_ratio_of_ratios",
            )
        )

        for family_rows in (h1, h2, h4):
            _apply_holm(family_rows, "p_value_two_sided", "holm_adjusted_p_value", "significant_holm_alpha_0_05")
            _apply_holm(family_rows, "tost_p_value", "holm_adjusted_tost_p_value", "equivalent_holm_alpha_0_05")
        _apply_holm(h3, "p_value_two_sided", "holm_adjusted_p_value", "significant_holm_alpha_0_05")
        _set_primary(h1, h2, h3, h4)
        group_rows = [*h1, *h2, *h3, *h4]
        counts = pd.Series([row["Hypothesis"] for row in group_rows]).value_counts()
        actual = {name: int(counts.get(name, 0)) for name in HYPOTHESIS_ROW_COUNTS}
        if actual != HYPOTHESIS_ROW_COUNTS:
            raise RuntimeError(f"仮説行数が不正です: {actual}")
        all_rows.extend(group_rows)

    result = pd.DataFrame(all_rows).reset_index(drop=True)
    expected = len(_group_keys(uncorrected)) * EXPECTED_ROWS_PER_ANALYSIS_GROUP
    if len(result) != expected:
        raise RuntimeError(f"検定表の総行数が不正です: expected={expected}, actual={len(result)}")
    return result


__all__ = [
    "EXPECTED_ROWS_PER_ANALYSIS_GROUP",
    "HYPOTHESIS_ROW_COUNTS",
    "HypothesisTestError",
    "holm_adjust",
    "run_hypothesis_tests",
]