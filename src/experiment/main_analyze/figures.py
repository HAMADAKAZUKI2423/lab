"""全参加者解析図と、個別・training用の試行ベース図を保存する。"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ANALYSIS_GROUP_COLUMNS,
    CONDITION_ORDER,
    CORRECTED_CONDITIONS,
    DP_CONDITION,
    DPF_CONDITION,
    MEAN_LOG10_COLUMN,
    OCULARITY_ORDER,
    SP_CONDITION,
    SPD_CONDITION,
    AnalysisOutputPaths,
    sanitize_run_component,
)
from .contrast_metrics import (
    AR_CONTRAST_COLUMN,
    EXTENDED_CONTRAST_COLUMN,
    RAW_CONTRAST_COLUMN,
)
from .dpf_correction import CORRECTED_LOG10_COLUMN


_CONDITION_LABELS = {
    SP_CONDITION: "SP",
    SPD_CONDITION: "SPD",
    DP_CONDITION: "DP",
    DPF_CONDITION: "DPF",
}
_EYE_LABELS = {"monocular": "Monocular", "binocular": "Binocular"}
_EYE_COLORS = {"monocular": "royalblue", "binocular": "darkorange"}
_BOOTSTRAP_SAMPLES = 10_000
_DPI = 180
_CONTRAST_Y_TICKS = np.array(
    [0.05, *np.arange(0.1, 1.01, 0.1)],
    dtype=float,
)
_CONTRAST_Y_TICK_LABELS = (
    "0.05",
    "0.1",
    "0.2",
    "0.3",
    "0.4",
    "0.5",
    "0.6",
    "0.7",
    "0.8",
    "0.9",
    "1.0",
)
_CONTRAST_ANNOTATION_Y = 0.055
_H4_Y_LIMITS = (-0.25, 0.15)
_H4_SYMLOG_LINTHRESH = 0.01


class FigureInputError(ValueError):
    """描画用参加者表の構造が不正な場合の例外。"""


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _configure_contrast_axis(axis) -> None:
    """contrast図を0.05〜1.0の固定log目盛りへ統一する。"""
    axis.set_yscale("log")
    axis.set_ylim(0.05, 1.0)
    axis.set_yticks(_CONTRAST_Y_TICKS)
    axis.set_yticklabels(_CONTRAST_Y_TICK_LABELS)
    axis.set_yticks([], minor=True)
    axis.tick_params(axis="y", which="both", direction="in")


def _annotate_mean_and_sd(
    axis,
    bars,
    means,
    standard_deviations,
) -> None:
    """各バーの足元へ平均mと標準偏差dを表示する。"""
    for bar, mean, standard_deviation in zip(
        bars,
        means,
        standard_deviations,
    ):
        deviation_text = (
            "nan"
            if not np.isfinite(standard_deviation)
            else f"{float(standard_deviation):.2f}"
        )
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            _CONTRAST_ANNOTATION_Y,
            f"m={float(mean):.2f}\nd={deviation_text}",
            ha="center",
            va="bottom",
            color="black",
            fontsize=9,
            bbox={
                "facecolor": "white",
                "alpha": 0.75,
                "edgecolor": "none",
                "pad": 1,
            },
            zorder=5,
        )


def _stable_seed(parts: tuple[object, ...]) -> int:
    token = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(sha256(token).digest()[:8], "big", signed=False)


def _bootstrap_mean_ci(
    values,
    *,
    seed_parts: tuple[object, ...],
) -> tuple[float, float, float]:
    """参加者を再標本化し、log10平均と95% percentile CIを返す。"""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise FigureInputError("CI計算には1人以上の有限な参加者値が必要です")
    mean = float(np.mean(array))
    if len(array) == 1 or np.allclose(array, array[0]):
        return mean, mean, mean
    random = np.random.default_rng(_stable_seed(seed_parts))
    samples = random.choice(
        array,
        size=(_BOOTSTRAP_SAMPLES, len(array)),
        replace=True,
    ).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return mean, float(lower), float(upper)


def _validate_frame(
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
        value_column,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise FigureInputError(f"{label}に必要な列が不足しています: {missing}")
    if frame.empty:
        raise FigureInputError(f"{label}が空です")
    validated = frame.copy(deep=True)
    for column in ("ID", "Session_Type", "Ocularity", "Condition"):
        validated[column] = validated[column].astype("string").str.strip()
        if (validated[column].isna() | validated[column].eq("")).any():
            raise FigureInputError(f"{label}の{column}に空値があります")
    for column in ("Ref_Contrast", "Orientation", value_column):
        values = pd.to_numeric(validated[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(float)).all():
            raise FigureInputError(f"{label}の{column}に有限でない値があります")
        validated[column] = values.astype(float)
    if (validated["Ref_Contrast"] <= 0).any():
        raise FigureInputError(f"{label}にはRef_Contrast > 0が必要です")
    if set(validated["Condition"].astype(str)) != set(conditions):
        raise FigureInputError(
            f"{label}の条件が不正です: expected={list(conditions)}, "
            f"found={sorted(set(validated['Condition'].astype(str)))}"
        )
    if set(validated["Ocularity"].astype(str)) != set(OCULARITY_ORDER):
        raise FigureInputError(f"{label}の眼条件が不正です")
    key_columns = ["ID", *ANALYSIS_GROUP_COLUMNS, "Ocularity", "Condition"]
    if validated.duplicated(key_columns, keep=False).any():
        raise FigureInputError(f"{label}に参加者×条件の重複行があります")
    for group_values, group in validated.groupby(
        list(ANALYSIS_GROUP_COLUMNS), sort=True, dropna=False
    ):
        expected_ids = set(group["ID"].astype(str))
        for eye in OCULARITY_ORDER:
            for condition in conditions:
                ids = set(
                    group.loc[
                        (group["Ocularity"] == eye)
                        & (group["Condition"] == condition),
                        "ID",
                    ].astype(str)
                )
                if ids != expected_ids:
                    raise FigureInputError(
                        f"{label}の参加者対応が不完全です: "
                        f"group={group_values}, eye={eye}, condition={condition}"
                    )
    return validated


def _number_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _file_name(prefix: str, metadata: dict[str, object]) -> str:
    return "__".join(
        [
            prefix,
            sanitize_run_component(str(metadata["Session_Type"])),
            f"ref_{_number_token(float(metadata['Ref_Contrast']))}",
            f"ori_{_number_token(float(metadata['Orientation']))}",
        ]
    ) + ".png"


def _title(label: str, metadata: dict[str, object]) -> str:
    return (
        f"{label}\nSession={metadata['Session_Type']}, "
        f"Ref={float(metadata['Ref_Contrast']):g}, "
        f"Orientation={float(metadata['Orientation']):g}°"
    )


def _condition_figure(
    group: pd.DataFrame,
    *,
    conditions: tuple[str, ...],
    value_column: str,
    label: str,
    metadata: dict[str, object],
):
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(9.2, 5.8))
    x = np.arange(len(conditions), dtype=float)
    width = 0.34
    for eye_index, eye in enumerate(OCULARITY_ORDER):
        positions = x + (eye_index - 0.5) * width
        mean_logs: list[float] = []
        lower_logs: list[float] = []
        upper_logs: list[float] = []
        participant_values: list[np.ndarray] = []
        participant_standard_deviations: list[float] = []
        for condition in conditions:
            values = group.loc[
                (group["Ocularity"] == eye)
                & (group["Condition"] == condition),
                value_column,
            ].to_numpy(float)
            mean, lower, upper = _bootstrap_mean_ci(
                values,
                seed_parts=(label, *metadata.values(), eye, condition),
            )
            mean_logs.append(mean)
            lower_logs.append(lower)
            upper_logs.append(upper)
            participant_values.append(values)
            linear_values = 10.0**values
            participant_standard_deviations.append(
                float(np.std(linear_values, ddof=1))
                if len(linear_values) > 1
                else float("nan")
            )
        means = 10.0 ** np.asarray(mean_logs)
        lowers = 10.0 ** np.asarray(lower_logs)
        uppers = 10.0 ** np.asarray(upper_logs)
        bars = axis.bar(
            positions,
            means,
            width=width * 0.92,
            color=_EYE_COLORS[eye],
            alpha=0.82,
            label=_EYE_LABELS[eye],
            zorder=2,
        )
        axis.errorbar(
            positions,
            means,
            yerr=np.vstack([means - lowers, uppers - means]),
            fmt="none",
            ecolor="#222222",
            elinewidth=1.2,
            capsize=4,
            zorder=4,
        )
        # mはバーと同じ幾何平均、dは参加者単位AR値の標準偏差。
        _annotate_mean_and_sd(
            axis,
            bars,
            means,
            participant_standard_deviations,
        )
        for position, values in zip(positions, participant_values):
            linear_values = 10.0**values
            jitter = np.linspace(-0.045, 0.045, len(linear_values))
            axis.scatter(
                np.full(len(linear_values), position) + jitter,
                linear_values,
                s=24,
                color="#202020",
                alpha=0.5,
                linewidths=0,
                zorder=3,
            )
    reference = float(metadata["Ref_Contrast"])
    axis.axhline(
        reference,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Reference",
        zorder=1,
    )
    _configure_contrast_axis(axis)
    axis.set_xticks(x)
    axis.set_xticklabels([_CONDITION_LABELS[condition] for condition in conditions])
    axis.set_xlabel("Condition")
    axis.set_ylabel("AR matched contrast (geometric mean)")
    axis.set_title(_title(label, metadata))
    axis.grid(axis="y", which="both", linestyle=":", alpha=0.4)
    axis.legend(frameon=False, ncol=3)
    figure.tight_layout()
    return figure


def _save_condition_figures(
    frame: pd.DataFrame,
    *,
    output_dir: str | Path,
    conditions: tuple[str, ...],
    value_column: str,
    label: str,
    prefix: str,
) -> list[Path]:
    plt = _pyplot()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for group_values, group in frame.groupby(
        list(ANALYSIS_GROUP_COLUMNS), sort=True, dropna=False
    ):
        metadata = dict(zip(ANALYSIS_GROUP_COLUMNS, group_values))
        figure = _condition_figure(
            group,
            conditions=conditions,
            value_column=value_column,
            label=label,
            metadata=metadata,
        )
        output_path = destination / _file_name(prefix, metadata)
        try:
            figure.savefig(output_path, dpi=_DPI, bbox_inches="tight", facecolor="white")
        finally:
            plt.close(figure)
        outputs.append(output_path)
    return outputs


def save_uncorrected_figures(
    uncorrected_df: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """未補正4条件を参加者単位の平均・95% CIで描画する。"""
    validated = _validate_frame(
        uncorrected_df,
        label="未補正参加者表",
        conditions=CONDITION_ORDER,
        value_column=MEAN_LOG10_COLUMN,
    )
    return _save_condition_figures(
        validated,
        output_dir=output_dir,
        conditions=CONDITION_ORDER,
        value_column=MEAN_LOG10_COLUMN,
        label="Uncorrected participant summaries",
        prefix="uncorrected",
    )


def save_corrected_figures(
    corrected_df: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """DPF補正後3条件を参加者単位の平均・95% CIで描画する。"""
    validated = _validate_frame(
        corrected_df,
        label="DPF補正後参加者表",
        conditions=CORRECTED_CONDITIONS,
        value_column=CORRECTED_LOG10_COLUMN,
    )
    return _save_condition_figures(
        validated,
        output_dir=output_dir,
        conditions=CORRECTED_CONDITIONS,
        value_column=CORRECTED_LOG10_COLUMN,
        label="DPF-corrected participant summaries",
        prefix="dpf_corrected",
    )


def _h4_effects(group: pd.DataFrame) -> pd.DataFrame:
    effects: dict[str, pd.Series] = {}
    for eye in OCULARITY_ORDER:
        pivot = group.loc[
            group["Ocularity"] == eye,
            ["ID", "Condition", CORRECTED_LOG10_COLUMN],
        ].pivot(index="ID", columns="Condition", values=CORRECTED_LOG10_COLUMN)
        effects[eye] = pivot[DP_CONDITION] - pivot[SPD_CONDITION]
    paired = pd.concat(
        [
            effects["monocular"].rename("monocular"),
            effects["binocular"].rename("binocular"),
        ],
        axis=1,
        join="outer",
    )
    if paired.isna().any().any():
        raise FigureInputError("H4 interactionの眼間参加者対応が不完全です")
    paired["interaction"] = paired["binocular"] - paired["monocular"]
    return paired.sort_index()


def _h4_figure(group: pd.DataFrame, metadata: dict[str, object]):
    plt = _pyplot()
    effect_frame = _h4_effects(group)
    columns = ("monocular", "binocular", "interaction")
    labels = (
        "Monocular\nlog10(DP/SPD)",
        "Binocular\nlog10(DP/SPD)",
        "Interaction\nBino − Mono",
    )
    colors = (_EYE_COLORS["monocular"], _EYE_COLORS["binocular"], "seagreen")
    summaries = [
        _bootstrap_mean_ci(
            effect_frame[column].to_numpy(float),
            seed_parts=("H4", *metadata.values(), column),
        )
        for column in columns
    ]
    means = np.asarray([summary[0] for summary in summaries])
    lowers = np.asarray([summary[1] for summary in summaries])
    uppers = np.asarray([summary[2] for summary in summaries])
    x = np.arange(3, dtype=float)
    figure, axis = plt.subplots(figsize=(8.0, 5.6))
    axis.bar(x, means, width=0.62, color=colors, alpha=0.84, zorder=2)
    axis.errorbar(
        x,
        means,
        yerr=np.vstack([means - lowers, uppers - means]),
        fmt="none",
        ecolor="#222222",
        elinewidth=1.2,
        capsize=4,
        zorder=4,
    )
    for index, column in enumerate(columns):
        values = effect_frame[column].to_numpy(float)
        jitter = np.linspace(-0.08, 0.08, len(values))
        axis.scatter(
            np.full(len(values), x[index]) + jitter,
            values,
            s=32,
            color="#202020",
            alpha=0.55,
            linewidths=0,
            zorder=3,
        )
        axis.annotate(
            f"ratio={10.0 ** means[index]:.3f}",
            (x[index], means[index]),
            xytext=(4, 5 if means[index] >= 0 else -12),
            textcoords="offset points",
            fontsize=8.5,
        )
    axis.axhline(0.0, color="#444444", linestyle="--", linewidth=1.2, zorder=1)
    axis.set_yscale(
        "symlog",
        base=10,
        linthresh=_H4_SYMLOG_LINTHRESH,
        linscale=1.0,
    )
    axis.set_ylim(*_H4_Y_LIMITS)
    axis.tick_params(axis="y", which="both", direction="in")
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Participant-level log10 effect")
    axis.set_title(_title("H4 SPD→DP effects and interaction", metadata))
    axis.grid(axis="y", linestyle=":", alpha=0.4)
    figure.tight_layout()
    return figure


def save_h4_interaction_figures(
    corrected_df: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """H4の眼別SPD→DP効果とinteractionを描画する。"""
    plt = _pyplot()
    validated = _validate_frame(
        corrected_df,
        label="DPF補正後参加者表",
        conditions=CORRECTED_CONDITIONS,
        value_column=CORRECTED_LOG10_COLUMN,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for group_values, group in validated.groupby(
        list(ANALYSIS_GROUP_COLUMNS), sort=True, dropna=False
    ):
        metadata = dict(zip(ANALYSIS_GROUP_COLUMNS, group_values))
        figure = _h4_figure(group, metadata)
        output_path = destination / _file_name("h4_interaction", metadata)
        try:
            figure.savefig(output_path, dpi=_DPI, bbox_inches="tight", facecolor="white")
        finally:
            plt.close(figure)
        outputs.append(output_path)
    return outputs



_LEGACY_DPI = 300
_LEGACY_METRIC_SPECS = (
    ("raw_contrast", RAW_CONTRAST_COLUMN, "Matched Contrast (Raw)"),
    ("ar_contrast", AR_CONTRAST_COLUMN, "Matched Contrast (AR)"),
    (
        "extended_contrast",
        EXTENDED_CONTRAST_COLUMN,
        "Matched Contrast (Extended)",
    ),
)


def _validate_legacy_trial_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "ID",
        *ANALYSIS_GROUP_COLUMNS,
        "Ocularity",
        "Condition",
        RAW_CONTRAST_COLUMN,
        AR_CONTRAST_COLUMN,
        EXTENDED_CONTRAST_COLUMN,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise FigureInputError(
            f"個別・training図に必要な列が不足しています: {missing}"
        )
    if frame.empty:
        raise FigureInputError("個別・training図の試行データが空です")

    validated = frame.copy(deep=True)
    for column in ("ID", "Session_Type", "Ocularity", "Condition"):
        values = validated[column].astype("string").str.strip()
        if column == "Ocularity":
            values = values.str.lower()
        validated[column] = values
        if (values.isna() | values.eq("")).any():
            raise FigureInputError(f"個別・training図の{column}に空値があります")

    for column in (
        "Ref_Contrast",
        "Orientation",
        RAW_CONTRAST_COLUMN,
        AR_CONTRAST_COLUMN,
        EXTENDED_CONTRAST_COLUMN,
    ):
        values = pd.to_numeric(validated[column], errors="coerce")
        invalid = values.isna() | ~np.isfinite(values.to_numpy(dtype=float))
        if column not in ("Ref_Contrast", "Orientation"):
            invalid = invalid | (values <= 0)
        if invalid.any():
            raise FigureInputError(
                f"個別・training図の{column}に描画不能な値があります"
            )
        validated[column] = values.astype(float)

    conditions = set(validated["Condition"].astype(str))
    unexpected_conditions = sorted(conditions - set(CONDITION_ORDER))
    if unexpected_conditions:
        raise FigureInputError(
            f"個別・training図に未定義の条件があります: {unexpected_conditions}"
        )
    ocularities = set(validated["Ocularity"].astype(str))
    unexpected_eyes = sorted(ocularities - set(OCULARITY_ORDER))
    if unexpected_eyes:
        raise FigureInputError(
            f"個別・training図に未定義の眼条件があります: {unexpected_eyes}"
        )
    return validated


def _bootstrap_linear_mean_ci(
    values,
    *,
    seed_parts: tuple[object, ...],
) -> tuple[float, float, float]:
    """試行を再標本化し、算術平均と95% percentile CIを返す。"""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise FigureInputError("試行ベースCIには1件以上の有限値が必要です")
    mean = float(np.mean(array))
    if len(array) == 1 or np.allclose(array, array[0]):
        return mean, mean, mean
    random = np.random.default_rng(_stable_seed(seed_parts))
    samples = random.choice(
        array,
        size=(_BOOTSTRAP_SAMPLES, len(array)),
        replace=True,
    ).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return mean, float(lower), float(upper)


def _legacy_number_token(value: float) -> str:
    return f"{float(value):g}"


def _legacy_file_name(
    metric_name: str,
    metadata: dict[str, object],
) -> str:
    session_type = sanitize_run_component(str(metadata["Session_Type"]))
    reference = _legacy_number_token(float(metadata["Ref_Contrast"]))
    orientation = _legacy_number_token(float(metadata["Orientation"]))
    return (
        f"{session_type}_matched_{metric_name}_"
        f"ref_{reference}_ori_{orientation}.png"
    )


def _legacy_metric_figure(
    group: pd.DataFrame,
    *,
    value_column: str,
    ylabel: str,
    metric_name: str,
    metadata: dict[str, object],
):
    plt = _pyplot()

    available_conditions = set(group["Condition"].astype(str))
    conditions = tuple(
        condition for condition in CONDITION_ORDER
        if condition in available_conditions
    )
    available_eyes = set(group["Ocularity"].astype(str))
    eyes = tuple(eye for eye in OCULARITY_ORDER if eye in available_eyes)
    if not conditions or not eyes:
        raise FigureInputError(
            f"描画可能な条件または眼条件がありません: {metadata}"
        )

    figure, axis = plt.subplots(figsize=(11, 6))
    x = np.arange(len(conditions), dtype=float)
    width = min(0.34, 0.78 / len(eyes))
    for eye_index, eye in enumerate(eyes):
        positions = x + (eye_index - (len(eyes) - 1) / 2.0) * width
        means: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []
        standard_deviations: list[float] = []
        for condition in conditions:
            values = group.loc[
                (group["Condition"] == condition)
                & (group["Ocularity"] == eye),
                value_column,
            ].to_numpy(dtype=float)
            if len(values) == 0:
                raise FigureInputError(
                    "個別・training図の条件セルが欠けています: "
                    f"group={metadata}, eye={eye}, condition={condition}"
                )
            mean, lower, upper = _bootstrap_linear_mean_ci(
                values,
                seed_parts=(
                    "legacy",
                    metric_name,
                    *metadata.values(),
                    eye,
                    condition,
                ),
            )
            means.append(mean)
            lowers.append(lower)
            uppers.append(upper)
            standard_deviations.append(
                float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
            )
        mean_array = np.asarray(means)
        lower_array = np.asarray(lowers)
        upper_array = np.asarray(uppers)
        bars = axis.bar(
            positions,
            mean_array,
            width=width * 0.92,
            color=_EYE_COLORS[eye],
            alpha=0.82,
            label=_EYE_LABELS[eye],
            zorder=2,
        )
        axis.errorbar(
            positions,
            mean_array,
            yerr=np.vstack(
                [mean_array - lower_array, upper_array - mean_array]
            ),
            fmt="none",
            ecolor="#222222",
            elinewidth=1.5,
            capsize=4,
            zorder=4,
        )
        _annotate_mean_and_sd(
            axis,
            bars,
            means,
            standard_deviations,
        )

    reference = float(metadata["Ref_Contrast"])
    axis.axhline(
        reference,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Ref Contrast ({reference:g})",
        zorder=1,
    )
    axis.set_title(
        f"{metadata['Session_Type']}: {ylabel} "
        f"(Ref={reference:g}, "
        f"Ori={float(metadata['Orientation']):g}°)"
    )
    axis.set_xlabel("Condition")
    axis.set_ylabel(ylabel)
    _configure_contrast_axis(axis)
    axis.set_xticks(x)
    axis.set_xticklabels([_CONDITION_LABELS[value] for value in conditions])
    axis.grid(axis="y", which="both", linestyle=":", alpha=0.4)
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis.legend(unique.values(), unique.keys(), loc="upper left")
    figure.tight_layout()
    return figure


def save_legacy_contrast_figures(
    trial_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, list[Path]]:
    """個別・trainingのraw・AR・extended図を試行データから保存する。"""
    plt = _pyplot()
    validated = _validate_legacy_trial_frame(trial_df)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    # 同じrunへ再出力したときに、なくなった解析群の古い図を残さない。
    for pattern in (
        "*_matched_raw_contrast_*.png",
        "*_matched_ar_contrast_*.png",
        "*_matched_extended_contrast_*.png",
        "*_matched_enhanced_contrast_*.png",
    ):
        for old_path in destination.glob(pattern):
            old_path.unlink()

    outputs: dict[str, list[Path]] = {
        metric_name: [] for metric_name, _, _ in _LEGACY_METRIC_SPECS
    }
    for group_values, group in validated.groupby(
        list(ANALYSIS_GROUP_COLUMNS),
        sort=True,
        dropna=False,
    ):
        metadata = dict(zip(ANALYSIS_GROUP_COLUMNS, group_values))
        for metric_name, value_column, ylabel in _LEGACY_METRIC_SPECS:
            figure = _legacy_metric_figure(
                group,
                value_column=value_column,
                ylabel=ylabel,
                metric_name=metric_name,
                metadata=metadata,
            )
            output_path = destination / _legacy_file_name(
                metric_name,
                metadata,
            )
            try:
                figure.savefig(
                    output_path,
                    dpi=_LEGACY_DPI,
                    bbox_inches="tight",
                    facecolor="white",
                )
            finally:
                plt.close(figure)
            outputs[metric_name].append(output_path)
    return outputs


def save_all_figures(
    uncorrected_df: pd.DataFrame,
    corrected_df: pd.DataFrame,
    output_paths: AnalysisOutputPaths,
) -> dict[str, list[Path]]:
    """同じrun名の3サブフォルダへ全図を保存する。"""
    return {
        "uncorrected": save_uncorrected_figures(
            uncorrected_df, output_paths.uncorrected_figure_dir
        ),
        "dpf_corrected": save_corrected_figures(
            corrected_df, output_paths.corrected_figure_dir
        ),
        "h4_interaction": save_h4_interaction_figures(
            corrected_df, output_paths.h4_figure_dir
        ),
    }


__all__ = [
    "FigureInputError",
    "save_all_figures",
    "save_corrected_figures",
    "save_h4_interaction_figures",
    "save_legacy_contrast_figures",
    "save_uncorrected_figures",
]