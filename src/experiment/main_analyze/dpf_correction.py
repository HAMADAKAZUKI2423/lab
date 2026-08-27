"""参加者別のDPFバイアスをlog10空間で補正する。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    CORRECTED_CONDITIONS,
    DPF_CONDITION,
    DPF_MATCH_COLUMNS,
    GEOMETRIC_MEAN_COLUMN,
    MEAN_LOG10_COLUMN,
    OCULARITY_ORDER,
    PARTICIPANT_AGGREGATION,
    PARTICIPANT_SUMMARY_KEY_COLUMNS,
    TRIAL_COUNT_COLUMN,
)


ORIGINAL_LOG10_COLUMN = "Original_Log10_AR"
ORIGINAL_AR_COLUMN = "Original_AR_Contrast"
REFERENCE_LOG10_COLUMN = "Reference_Log10"
DPF_LOG10_COLUMN = "DPF_Log10_AR"
DPF_AR_COLUMN = "DPF_Geometric_Mean_AR"
DPF_TRIAL_COUNT_COLUMN = "DPF_Trial_Count"
DPF_BIAS_LOG10_COLUMN = "DPF_Bias_Log10"
DPF_BIAS_RATIO_COLUMN = "DPF_Bias_Ratio"
CORRECTED_LOG10_COLUMN = "Corrected_Log10_AR"
CORRECTED_AR_COLUMN = "Corrected_AR_Contrast"
CORRECTED_OFFSET_COLUMN = "Corrected_Log10_Offset_From_Ref"
CORRECTED_RATIO_COLUMN = "Corrected_Ratio_To_Ref"
CORRECTION_METHOD_COLUMN = "Bias_Correction_Method"
CORRECTION_METHOD = "z_corrected = z_condition - z_DPF + log10(Ref)"

_REQUIRED_COLUMNS = (
    *PARTICIPANT_SUMMARY_KEY_COLUMNS,
    TRIAL_COUNT_COLUMN,
    MEAN_LOG10_COLUMN,
    GEOMETRIC_MEAN_COLUMN,
    "Participant_Aggregation",
)
_STRING_KEY_COLUMNS = ("ID", "Session_Type", "Ocularity", "Condition")
_NUMERIC_COLUMNS = (
    "Ref_Contrast",
    "Orientation",
    TRIAL_COUNT_COLUMN,
    MEAN_LOG10_COLUMN,
    GEOMETRIC_MEAN_COLUMN,
)


class DPFCorrectionError(ValueError):
    """DPF補正に必要な対応関係または値が不正な場合の例外。"""


def _records_preview(
    frame: pd.DataFrame,
    columns: tuple[str, ...] | list[str],
    *,
    limit: int = 10,
) -> list[dict[str, object]]:
    return frame.loc[:, list(columns)].head(limit).to_dict("records")


def _validate_required_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DPFCorrectionError(
            f"DPF補正に必要な参加者集約列が不足しています: {missing}"
        )


def _normalize_and_validate_values(frame: pd.DataFrame) -> pd.DataFrame:
    validated = frame.copy(deep=True)
    _validate_required_columns(validated)
    if validated.empty:
        raise DPFCorrectionError("未補正の参加者集約表が空です")

    for column in _STRING_KEY_COLUMNS:
        values = validated[column].astype("string").str.strip()
        validated[column] = values
        missing = values.isna() | values.eq("")
        if missing.any():
            rows = validated.index[missing].tolist()[:10]
            raise DPFCorrectionError(
                f"DPF補正キーの{column}が空です: row_index={rows}"
            )

    for column in _NUMERIC_COLUMNS:
        converted = pd.to_numeric(validated[column], errors="coerce")
        invalid = converted.isna() | ~np.isfinite(converted.to_numpy(dtype=float))
        if invalid.any():
            rows = validated.index[invalid].tolist()[:10]
            raise DPFCorrectionError(
                f"{column}を有限な数値へ変換できません: row_index={rows}"
            )
        validated[column] = converted.astype(float)

    if (validated["Ref_Contrast"] <= 0).any():
        rows = validated.index[validated["Ref_Contrast"] <= 0].tolist()[:10]
        raise DPFCorrectionError(
            f"Ref_Contrast > 0が必要です: row_index={rows}"
        )
    if (validated[GEOMETRIC_MEAN_COLUMN] <= 0).any():
        rows = validated.index[
            validated[GEOMETRIC_MEAN_COLUMN] <= 0
        ].tolist()[:10]
        raise DPFCorrectionError(
            f"{GEOMETRIC_MEAN_COLUMN} > 0が必要です: row_index={rows}"
        )

    trial_counts = validated[TRIAL_COUNT_COLUMN]
    invalid_counts = (trial_counts <= 0) | ~np.isclose(
        trial_counts,
        np.round(trial_counts),
    )
    if invalid_counts.any():
        rows = validated.index[invalid_counts].tolist()[:10]
        raise DPFCorrectionError(
            f"{TRIAL_COUNT_COLUMN}は正の整数にしてください: row_index={rows}"
        )
    validated[TRIAL_COUNT_COLUMN] = np.round(trial_counts).astype(int)

    aggregation_values = set(
        validated["Participant_Aggregation"].dropna().astype(str)
    )
    if aggregation_values != {PARTICIPANT_AGGREGATION}:
        raise DPFCorrectionError(
            "参加者集約方法が現行計画と一致しません: "
            f"expected={PARTICIPANT_AGGREGATION}, "
            f"found={sorted(aggregation_values)}"
        )

    conditions = set(validated["Condition"].astype(str))
    expected_conditions = {*CORRECTED_CONDITIONS, DPF_CONDITION}
    unexpected_conditions = sorted(conditions - expected_conditions)
    if unexpected_conditions:
        raise DPFCorrectionError(
            f"未定義のConditionがあります: {unexpected_conditions}"
        )

    ocularities = set(validated["Ocularity"].astype(str))
    unexpected_ocularities = sorted(ocularities - set(OCULARITY_ORDER))
    if unexpected_ocularities:
        raise DPFCorrectionError(
            f"未定義のOcularityがあります: {unexpected_ocularities}"
        )

    duplicated = validated.duplicated(
        list(PARTICIPANT_SUMMARY_KEY_COLUMNS),
        keep=False,
    )
    if duplicated.any():
        preview = _records_preview(
            validated.loc[duplicated],
            list(PARTICIPANT_SUMMARY_KEY_COLUMNS),
        )
        raise DPFCorrectionError(
            "参加者×解析群×眼×条件の行が重複しています: "
            f"{preview}"
        )

    expected_geometric_mean = np.power(
        10.0,
        validated[MEAN_LOG10_COLUMN].to_numpy(dtype=float),
    )
    consistent = np.isclose(
        validated[GEOMETRIC_MEAN_COLUMN].to_numpy(dtype=float),
        expected_geometric_mean,
        rtol=1e-9,
        atol=1e-12,
    )
    if not bool(consistent.all()):
        rows = validated.index[~consistent].tolist()[:10]
        raise DPFCorrectionError(
            f"{MEAN_LOG10_COLUMN}と{GEOMETRIC_MEAN_COLUMN}が矛盾します: "
            f"row_index={rows}"
        )
    return validated


def _validate_condition_grid(frame: pd.DataFrame) -> None:
    expected = {*CORRECTED_CONDITIONS, DPF_CONDITION}
    errors: list[dict[str, object]] = []
    for group_values, group in frame.groupby(
        list(DPF_MATCH_COLUMNS),
        sort=True,
        dropna=False,
    ):
        present = set(group["Condition"].astype(str))
        if present != expected:
            metadata = dict(zip(DPF_MATCH_COLUMNS, group_values))
            metadata["missing"] = sorted(expected - present)
            metadata["unexpected"] = sorted(present - expected)
            errors.append(metadata)
    if errors:
        raise DPFCorrectionError(
            "DPF補正には参加者・解析群・眼ごとにSP/SPD/DP/DPFが"
            f"1行ずつ必要です: {errors[:10]}"
        )


def _build_dpf_table(frame: pd.DataFrame) -> pd.DataFrame:
    dpf = frame.loc[
        frame["Condition"] == DPF_CONDITION,
        [
            *DPF_MATCH_COLUMNS,
            TRIAL_COUNT_COLUMN,
            MEAN_LOG10_COLUMN,
            GEOMETRIC_MEAN_COLUMN,
        ],
    ].copy()
    duplicated = dpf.duplicated(list(DPF_MATCH_COLUMNS), keep=False)
    if duplicated.any():
        preview = _records_preview(
            dpf.loc[duplicated],
            list(DPF_MATCH_COLUMNS),
        )
        raise DPFCorrectionError(f"DPF行が重複しています: {preview}")
    return dpf.rename(
        columns={
            TRIAL_COUNT_COLUMN: DPF_TRIAL_COUNT_COLUMN,
            MEAN_LOG10_COLUMN: DPF_LOG10_COLUMN,
            GEOMETRIC_MEAN_COLUMN: DPF_AR_COLUMN,
        }
    )


def _validate_pair_coverage(
    target: pd.DataFrame,
    dpf: pd.DataFrame,
) -> None:
    target_keys = target.loc[:, list(DPF_MATCH_COLUMNS)].drop_duplicates()
    dpf_keys = dpf.loc[:, list(DPF_MATCH_COLUMNS)].drop_duplicates()
    coverage = target_keys.merge(
        dpf_keys,
        on=list(DPF_MATCH_COLUMNS),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    unmatched = coverage.loc[coverage["_merge"] != "both"]
    if not unmatched.empty:
        preview = _records_preview(
            unmatched,
            [*DPF_MATCH_COLUMNS, "_merge"],
        )
        raise DPFCorrectionError(
            f"補正対象とDPFの対応が1対1でそろっていません: {preview}"
        )


def apply_dpf_correction(uncorrected_df: pd.DataFrame) -> pd.DataFrame:
    """DPFバイアスをSP・SPD・DPから差し引いた参加者表を返す。

    補正式は次のとおり。DPFは参加者・解析群・眼ごとに対応付ける。

    ``z_corrected = z_condition - z_DPF + log10(Ref)``

    入力DataFrameは変更せず、DPF行を除いた3条件の新しい表を返す。
    """
    validated = _normalize_and_validate_values(uncorrected_df)
    _validate_condition_grid(validated)

    target = validated.loc[
        validated["Condition"].isin(CORRECTED_CONDITIONS)
    ].copy()
    dpf = _build_dpf_table(validated)
    _validate_pair_coverage(target, dpf)

    corrected = target.merge(
        dpf,
        on=list(DPF_MATCH_COLUMNS),
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not corrected["_merge"].eq("both").all():
        raise DPFCorrectionError("DPF結合後に未対応行が残りました")
    corrected = corrected.drop(columns="_merge")

    corrected[ORIGINAL_LOG10_COLUMN] = corrected[MEAN_LOG10_COLUMN]
    corrected[ORIGINAL_AR_COLUMN] = corrected[GEOMETRIC_MEAN_COLUMN]
    corrected[REFERENCE_LOG10_COLUMN] = np.log10(
        corrected["Ref_Contrast"].to_numpy(dtype=float)
    )
    corrected[DPF_BIAS_LOG10_COLUMN] = (
        corrected[DPF_LOG10_COLUMN] - corrected[REFERENCE_LOG10_COLUMN]
    )
    corrected[DPF_BIAS_RATIO_COLUMN] = (
        corrected[DPF_AR_COLUMN] / corrected["Ref_Contrast"]
    )
    corrected[CORRECTED_LOG10_COLUMN] = (
        corrected[ORIGINAL_LOG10_COLUMN]
        - corrected[DPF_BIAS_LOG10_COLUMN]
    )
    corrected[CORRECTED_OFFSET_COLUMN] = (
        corrected[ORIGINAL_LOG10_COLUMN] - corrected[DPF_LOG10_COLUMN]
    )
    corrected[CORRECTED_RATIO_COLUMN] = (
        corrected[ORIGINAL_AR_COLUMN] / corrected[DPF_AR_COLUMN]
    )
    corrected[CORRECTED_AR_COLUMN] = np.power(
        10.0,
        corrected[CORRECTED_LOG10_COLUMN].to_numpy(dtype=float),
    )
    corrected[CORRECTION_METHOD_COLUMN] = CORRECTION_METHOD

    corrected_ar_from_ratio = (
        corrected[CORRECTED_RATIO_COLUMN] * corrected["Ref_Contrast"]
    )
    if not np.allclose(
        corrected[CORRECTED_AR_COLUMN],
        corrected_ar_from_ratio,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise RuntimeError("DPF補正のlog10式と比率式が一致しません")
    if not np.allclose(
        corrected[CORRECTED_LOG10_COLUMN]
        - corrected[REFERENCE_LOG10_COLUMN],
        corrected[CORRECTED_OFFSET_COLUMN],
        rtol=1e-10,
        atol=1e-12,
    ):
        raise RuntimeError("DPF補正値とRefオフセットが一致しません")

    ocularity_order = {
        value: index for index, value in enumerate(OCULARITY_ORDER)
    }
    condition_order = {
        value: index for index, value in enumerate(CORRECTED_CONDITIONS)
    }
    corrected["_Ocularity_Order"] = corrected["Ocularity"].map(
        ocularity_order
    )
    corrected["_Condition_Order"] = corrected["Condition"].map(
        condition_order
    )
    corrected = corrected.sort_values(
        [
            "Session_Type",
            "Ref_Contrast",
            "Orientation",
            "ID",
            "_Ocularity_Order",
            "_Condition_Order",
        ],
        kind="stable",
        ignore_index=True,
    ).drop(columns=["_Ocularity_Order", "_Condition_Order"])

    priority_columns = [
        *PARTICIPANT_SUMMARY_KEY_COLUMNS,
        TRIAL_COUNT_COLUMN,
        MEAN_LOG10_COLUMN,
        GEOMETRIC_MEAN_COLUMN,
        ORIGINAL_LOG10_COLUMN,
        ORIGINAL_AR_COLUMN,
        REFERENCE_LOG10_COLUMN,
        DPF_TRIAL_COUNT_COLUMN,
        DPF_LOG10_COLUMN,
        DPF_AR_COLUMN,
        DPF_BIAS_LOG10_COLUMN,
        DPF_BIAS_RATIO_COLUMN,
        CORRECTED_LOG10_COLUMN,
        CORRECTED_AR_COLUMN,
        CORRECTED_OFFSET_COLUMN,
        CORRECTED_RATIO_COLUMN,
        "Participant_Aggregation",
        CORRECTION_METHOD_COLUMN,
    ]
    remaining_columns = [
        column for column in corrected.columns if column not in priority_columns
    ]
    return corrected.loc[:, [*priority_columns, *remaining_columns]]


__all__ = [
    "CORRECTED_AR_COLUMN",
    "CORRECTED_LOG10_COLUMN",
    "CORRECTED_OFFSET_COLUMN",
    "CORRECTED_RATIO_COLUMN",
    "CORRECTION_METHOD",
    "DPFCorrectionError",
    "DPF_AR_COLUMN",
    "DPF_BIAS_LOG10_COLUMN",
    "DPF_BIAS_RATIO_COLUMN",
    "DPF_LOG10_COLUMN",
    "ORIGINAL_AR_COLUMN",
    "ORIGINAL_LOG10_COLUMN",
    "REFERENCE_LOG10_COLUMN",
    "apply_dpf_correction",
]