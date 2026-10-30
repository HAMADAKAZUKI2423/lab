"""全参加者解析図と、個別・training用の試行ベース図を保存する。"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ANALYSIS_GROUP_COLUMNS,
    ANALYSIS_SCALE,
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

# 追加出力 *_significance_no_title.png だけの見た目設定。
# FIGSIZE: 基準PNGを指定しない単独呼び出し時の(幅, 高さ) [inch]。
# 通常のsave_all_figuresではuncorrectedの_no_title PNGと同じ画素数にする。
# FONT_SIZE: [pt]。
_SIGNIFICANCE_FIGSIZE = (9.2, 5.8)
_SIGNIFICANCE_AXIS_FONT_SIZE = 22
_SIGNIFICANCE_YLABEL_FONT_SIZE = 22
_SIGNIFICANCE_LABEL_FONT_SIZE = 16       # アスタリスク / n.s. の希望サイズ
_SIGNIFICANCE_LEGEND_FONT_SIZE = 22      # Reference / Monocular / Binocular（自動縮小しない）
_SIGNIFICANCE_MIN_FONT_SIZE = 6         # 空き領域不足時の自動縮小の下限
_SIGNIFICANCE_TOP_PADDING_PT = 2
_SIGNIFICANCE_DATA_GAP_PT = 2
_SIGNIFICANCE_LEGEND_GAP_PT = 1
_SIGNIFICANCE_ROW_GAP_PT = 0.4
_SIGNIFICANCE_CAP_HEIGHT_PT = 4.0
_SIGNIFICANCE_LABEL_GAP_PT = 0.3
_SIGNIFICANCE_LINE_WIDTH = 1.0
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
_H4_SYMLOG_LINTHRESH = 0.01
_H4_SYMLOG_LINSCALE = 1.0  # 上端10^1でも収まらない場合のみ0〜10^-2区間を縮める
_H4_SYMLOG_MIN_LINSCALE = 0.1


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


def _annotate_median_and_iqr(
    axis,
    bars,
    medians,
    interquartile_ranges,
) -> None:
    """各バーの足元へ中央値medと四分位範囲IQRを表示する。"""
    for bar, median, interquartile_range in zip(
        bars,
        medians,
        interquartile_ranges,
    ):
        iqr_text = (
            "nan"
            if not np.isfinite(interquartile_range)
            else f"{float(interquartile_range):.2f}"
        )
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            _CONTRAST_ANNOTATION_Y,
            f"med={float(median):.2f}\nIQR={iqr_text}",
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


def _mean_t_ci(values) -> tuple[float, float, float]:
    """参加者単位の対数値から平均と両側95% t信頼区間を返す。

    自由度はn-1、標準偏差はddof=1。参加者の対数値が独立で、
    平均のt区間を適用できる分布であることを前提とする。
    元のcontrast尺度への逆変換は呼び出し側で行う。
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise FigureInputError("t信頼区間には1次元の有限な参加者値が必要です")
    n = len(array)
    if n < 2:
        raise FigureInputError("95% t信頼区間の計算には2人以上の参加者が必要です")
    from scipy.stats import t

    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1))
    sem = sd / np.sqrt(n)
    critical_t = float(t.ppf(0.975, df=n - 1))
    half_width = critical_t * sem
    if not np.isfinite([mean, half_width]).all():
        raise FigureInputError("95% t信頼区間を有限値として計算できません")
    return mean, float(mean - half_width), float(mean + half_width)


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
            # 参加者ごとのlog10代表値に基づく、平均の95% t信頼区間。
            mean, lower, upper = _mean_t_ci(values)
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


def _save_with_title_variants(
    figure,
    output_path: Path,
    *,
    dpi: int,
) -> list[Path]:
    """従来版と、点なし・文字サイズ2.4倍・利用可能な指定フォントの_no_title版を保存する。"""
    import textwrap

    from matplotlib import font_manager
    from matplotlib.collections import PathCollection
    from matplotlib.text import Text

    # 実行環境に存在するフォントだけを選び、暗黙の代替フォントを防ぐ。
    selected_font = None
    for candidate in ("Times New Roman", "Arial", "Helvetica", "Times"):
        try:
            font_manager.findfont(candidate, fallback_to_default=False)
        except ValueError:
            continue
        selected_font = candidate
        break
    if selected_font is None:
        raise RuntimeError(
            "Times New Roman / Arial / Helvetica / Timesのいずれかをインストールしてください。"
            "Symbolは本文・数字用のフォントとして使用しません。"
        )

    output_path = Path(output_path)
    save_options = dict(dpi=dpi, bbox_inches="tight", facecolor="white")
    # タイトルあり版は、点・文字サイズとも従来どおり保存する。
    figure.savefig(output_path, **save_options)
    no_title_path = output_path.with_name(
        f"{output_path.stem}_no_title{output_path.suffix}"
    )
    # 描画を確定し、目盛り・軸ラベル・凡例・注記を含む全Textを取得する。
    figure.canvas.draw()
    text_sizes = [(text, text.get_fontsize()) for text in figure.findobj(Text)]
    text_families = [(text, text.get_fontfamily()) for text, _ in text_sizes]
    hidden_artists = []
    for axis in figure.axes:
        hidden_artists.extend([axis.title, axis._left_title, axis._right_title])
        # scatterだけを非表示にする。errorbarのLineCollectionは保持する。
        # 個別・training図の試行点も_no_title版では非表示にする。
        hidden_artists.extend(
            collection for collection in axis.collections
            if isinstance(collection, PathCollection)
        )
    if figure._suptitle is not None:
        hidden_artists.append(figure._suptitle)
    # 平均・標準偏差の数値注記だけを隠す（棒の高さやCIは維持）。
    for axis in figure.axes:
        hidden_artists.extend(
            text for text in axis.texts
            if text.get_text().startswith("m=") and "\nd=" in text.get_text()
        )
    # タイトルなし版では0.6〜0.9の縦軸数字だけを隠す。目盛線・グリッドは保持する。
    for axis in figure.axes:
        for tick in axis.yaxis.get_major_ticks() + axis.yaxis.get_minor_ticks():
            if any(abs(float(tick.get_loc()) - value) < 1e-8 for value in (0.6, 0.7, 0.8, 0.9)):
                hidden_artists.extend([tick.label1, tick.label2])
    ylabel_texts = [(axis.yaxis.label, axis.yaxis.label.get_text()) for axis in figure.axes]
    visibility = [(artist, artist.get_visible()) for artist in hidden_artists]
    original_positions = [
        (axis, axis.get_position(original=True).frozen(), axis.get_position().frozen())
        for axis in figure.axes
    ]
    original_subplotpars = {
        key: getattr(figure.subplotpars, key)
        for key in ("left", "bottom", "right", "top", "wspace", "hspace")
    }
    try:
        for artist, _ in visibility:
            artist.set_visible(False)
        for text, size in text_sizes:
            text.set_fontsize(size * 2.4)
            text.set_fontfamily([selected_font])
        # 文字拡大に合わせて余白を再計算する。棒とCIの値は変更しない。
        figure.tight_layout()
        # 縦向きの軸名がプロットの高さを超えたら、文字サイズを保って改行する。
        # bbox_inches="tight"と余白再計算により、左端の切れも防ぐ。
        for axis in figure.axes:
            label = axis.yaxis.label
            original_label = label.get_text()
            prefix, bracket, suffix = original_label.partition("(")
            if bracket:
                original_label = prefix.rstrip() + "\n(" + suffix
                label.set_text(original_label)
                figure.tight_layout()
            # 長い場合の追加折り返しでも、括弧直前の改行は維持する。
            for width in range(max(len(line) for line in original_label.split("\n")), 7, -1):
                figure.canvas.draw()
                renderer = figure.canvas.get_renderer()
                if label.get_window_extent(renderer).height <= axis.get_window_extent(renderer).height * 0.95:
                    break
                label.set_text("\n".join(
                    "\n".join(textwrap.wrap(line, width=width))
                    for line in original_label.split("\n")
                ))
                figure.tight_layout()
        figure.savefig(no_title_path, **save_options)
    finally:
        for text, family in text_families:
            text.set_fontfamily(family)
        for label, original_text in ylabel_texts:
            label.set_text(original_text)
        for text, size in text_sizes:
            text.set_fontsize(size)
        for artist, was_visible in visibility:
            artist.set_visible(was_visible)
        figure.subplotpars.update(**original_subplotpars)
        for axis, original, active in original_positions:
            axis.set_position(original, which="original")
            axis.set_position(active, which="active")
    return [output_path, no_title_path]


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
            saved_paths = _save_with_title_variants(figure, output_path, dpi=_DPI)
        finally:
            plt.close(figure)
        outputs.extend(saved_paths)
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


# 以下は既存出力を置換せず、補正後・6比較注記付きの別PNGだけを追加する。
def _significance_label(adjusted_p: float) -> str:
    """両側t検定のHolm補正後p値を表示用記号へ変換する。"""
    value = float(adjusted_p)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        # 評価不能・欠損をn.s.として表示しない。
        raise FigureInputError(f"有意差注記に使えない補正後p値です: {value}")
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return "n.s."


def _condition_significance_rows(
    group: pd.DataFrame,
    tests: pd.DataFrame,
    metadata: dict[str, object],
) -> list[dict[str, object]]:
    """解析群・集約方法・眼・比較を照合し、予定した6比較だけを返す。"""
    required = [
        *ANALYSIS_GROUP_COLUMNS,
        "Participant_Aggregation", "Analysis_Scale", "Data_State",
        "Hypothesis", "Ocularity", "Condition", "Baseline_Condition",
        "holm_adjusted_p_value", "n",
    ]
    missing = [column for column in required if column not in tests.columns]
    if missing:
        raise FigureInputError(f"有意差注記の検定表に必要な列がありません: {missing}")
    if "Participant_Aggregation" not in group:
        raise FigureInputError("補正後参加者表にParticipant_Aggregationがありません")
    aggregations = group["Participant_Aggregation"].astype("string").str.strip()
    if aggregations.isna().any() or aggregations.eq("").any() or aggregations.nunique() != 1:
        raise FigureInputError("補正後参加者表の集約方法を一意に特定できません")
    aggregation = str(aggregations.iloc[0])
    mask = (
        tests["Session_Type"].astype("string").eq(str(metadata["Session_Type"]))
        & tests["Participant_Aggregation"].astype("string").eq(aggregation)
    )
    for column in ("Ref_Contrast", "Orientation"):
        numeric = pd.to_numeric(tests[column], errors="coerce").to_numpy(float)
        mask &= np.isclose(numeric, float(metadata[column]), rtol=0.0, atol=1e-12)
    matched = tests.loc[mask.fillna(False)]
    # 各ペアについて単眼→両眼の順。長いSP–DP括弧は上段に置く。
    comparisons = (
        ("H3", SPD_CONDITION, SP_CONDITION),
        ("H4", DP_CONDITION, SPD_CONDITION),
        ("H3", DP_CONDITION, SP_CONDITION),
    )
    annotations: list[dict[str, object]] = []
    for hypothesis, condition, baseline in comparisons:
        for eye in OCULARITY_ORDER:
            rows = matched.loc[
                matched["Hypothesis"].eq(hypothesis)
                & matched["Ocularity"].eq(eye)
                & matched["Condition"].eq(condition)
                & matched["Baseline_Condition"].eq(baseline)
            ]
            if len(rows) != 1:
                raise FigureInputError(
                    "有意差注記の検定結果は1行必要です: "
                    f"group={metadata}, aggregation={aggregation}, "
                    f"eye={eye}, {condition} vs {baseline}, found={len(rows)}"
                )
            row = rows.iloc[0]
            if row["Data_State"] != "dpf_corrected" or row["Analysis_Scale"] != ANALYSIS_SCALE:
                raise FigureInputError("有意差注記の検定表の補正状態・解析尺度が一致しません")
            n_expected = group.loc[group["Ocularity"].eq(eye), "ID"].nunique()
            try:
                n_test = float(row["n"])
                adjusted_p = float(row["holm_adjusted_p_value"])
            except (TypeError, ValueError) as error:
                raise FigureInputError("有意差注記のnまたはp値が数値ではありません") from error
            if not np.isfinite(n_test) or n_test < 2 or n_test != n_expected:
                raise FigureInputError(
                    f"図と検定の参加者数が一致しません: plot={n_expected}, test={n_test}"
                )
            annotations.append({
                "Ocularity": eye,
                "Condition": condition,
                "Baseline_Condition": baseline,
                # Primary_Adjusted_P_Valueは使わない。H4単眼ではTOSTを指すため。
                "Adjusted_P_Value": adjusted_p,
                "Label": _significance_label(adjusted_p),
            })
    return annotations


def _corrected_significance_figure(
    group: pd.DataFrame,
    metadata: dict[str, object],
    annotations: list[dict[str, object]],
    *,
    reference_pixel_size: tuple[int, int] | None = None,
):
    """未補正4条件版と同じサイズに、補正後3条件の6比較を収める。"""
    from matplotlib import font_manager
    from matplotlib.collections import PathCollection
    from matplotlib.container import BarContainer, ErrorbarContainer
    from matplotlib.text import Text

    selected_font = None
    for candidate in ("Times New Roman", "Arial", "Helvetica", "Times"):
        try:
            font_manager.findfont(candidate, fallback_to_default=False)
        except ValueError:
            continue
        selected_font = candidate
        break
    if selected_font is None:
        raise RuntimeError(
            "Times New Roman / Arial / Helvetica / Timesのいずれかをインストールしてください。"
        )

    figure = _condition_figure(
        group,
        conditions=CORRECTED_CONDITIONS,
        value_column=CORRECTED_LOG10_COLUMN,
        # 棒・CIとも既存の補正後図と同じ計算を共有する。
        label="DPF-corrected participant summaries",
        metadata=metadata,
    )
    try:
        # 既存PNGと既存_no_title版の描画処理は触らない。
        if reference_pixel_size is None:
            figure.set_size_inches(*_SIGNIFICANCE_FIGSIZE)
        else:
            # tight crop済みの既存PNGの画素数をそのままキャンバスに使う。
            # 0.01 pixelの余裕はfloat→整数変換の丸め落ち防止用。
            width_px, height_px = reference_pixel_size
            if width_px <= 0 or height_px <= 0:
                raise FigureInputError("基準PNGの画像サイズが不正です")
            figure.set_size_inches((width_px + 0.01) / _DPI, (height_px + 0.01) / _DPI)
        axis = figure.axes[0]
        for title in (axis.title, axis._left_title, axis._right_title):
            title.set_visible(False)
        if figure._suptitle is not None:
            figure._suptitle.set_visible(False)
        for collection in axis.collections:
            if isinstance(collection, PathCollection):
                collection.set_visible(False)
        for text in axis.texts:
            if text.get_text().startswith("m=") and "\nd=" in text.get_text():
                text.set_visible(False)
        figure.canvas.draw()
        for text in figure.findobj(Text):
            text.set_fontsize(_SIGNIFICANCE_AXIS_FONT_SIZE)
            text.set_fontfamily([selected_font])
        axis.set_ylabel(
            "DPF-corrected matched contrast\n" + r"$C'_{\mathrm{AR}}$ (geometric mean)",
            fontsize=_SIGNIFICANCE_YLABEL_FONT_SIZE,
        )

        bar_containers = [c for c in axis.containers if isinstance(c, BarContainer)]
        if len(bar_containers) != len(OCULARITY_ORDER):
            raise FigureInputError("6比較注記の棒の配置を特定できません")
        centers: dict[tuple[str, str], float] = {}
        for eye, bars in zip(OCULARITY_ORDER, bar_containers):
            if len(bars) != len(CORRECTED_CONDITIONS):
                raise FigureInputError("6比較注記の棒の数が不正です")
            for condition, bar in zip(CORRECTED_CONDITIONS, bars):
                centers[(eye, condition)] = bar.get_x() + bar.get_width() / 2.0

        # CIの数値・対数尺度は変更しない。各括弧の横幅内にある
        # 棒・CIだけを避けて配置し、SPD対DPをSPの高さまで持ち上げない。
        reference = float(metadata["Ref_Contrast"])
        ci_values = [reference]
        ci_tops = {
            (eye, condition): float(bar.get_height())
            for eye, bars in zip(OCULARITY_ORDER, bar_containers)
            for condition, bar in zip(CORRECTED_CONDITIONS, bars)
        }
        errorbars = [c for c in axis.containers if isinstance(c, ErrorbarContainer)]
        if len(errorbars) != len(OCULARITY_ORDER):
            raise FigureInputError("CIの眼条件の配置を特定できません")
        for eye, container in zip(OCULARITY_ORDER, errorbars):
            segments = container.lines[2][0].get_segments()
            if len(segments) != len(CORRECTED_CONDITIONS):
                raise FigureInputError("CIの条件数が不正です")
            for condition, segment in zip(CORRECTED_CONDITIONS, segments):
                ys = np.asarray(segment)[:, 1]
                ci_values.extend(ys.tolist())
                ci_tops[(eye, condition)] = float(max(ys))
        low_ci, high_ci = min(ci_values), max(ci_values)
        if low_ci <= 0.0 or not np.isfinite(ci_values).all():
            raise FigureInputError("対数軸で表示できない補正後CIです")
        if high_ci >= 1.0:
            raise FigureInputError(
                "CIまたはreferenceが1.0以上のため、全注記をその上かつ1.0未満には配置できません。"
                "値を切り捨てずに停止しました。"
            )
        lower = min(axis.get_ylim()[0], low_ci / 1.1)
        axis.set_ylim(lower, 1.0)
        ticks = sorted(set([
            *_CONTRAST_Y_TICKS,
            *([lower] if lower < _CONTRAST_Y_TICKS[0] else []),
        ]))
        tick_labels = [
            "" if any(abs(value - hidden) < 1e-8 for hidden in (0.6, 0.7, 0.8, 0.9))
            else f"{value:.1f}" if 0.1 <= value <= 1.0 else f"{value:g}"
            for value in ticks
        ]
        axis.set_yticks(ticks)
        axis.set_yticklabels(
            tick_labels, fontsize=_SIGNIFICANCE_AXIS_FONT_SIZE, fontfamily=selected_font,
        )
        # 古い枠外凡例を外してから、軸名・目盛りのための余白を確定する。
        if axis.get_legend() is not None:
            axis.get_legend().remove()
        figure.tight_layout()
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        bounds = axis.get_window_extent(renderer)
        px_per_pt = figure.dpi / 72.0
        pt = lambda value: value * px_per_pt
        region_top = bounds.y1 - pt(_SIGNIFICANCE_TOP_PADDING_PT)
        if len(annotations) != 6:
            raise FigureInputError("比較注記は6本必要です")
        if not (0 < _SIGNIFICANCE_MIN_FONT_SIZE <= min(
            _SIGNIFICANCE_LABEL_FONT_SIZE, _SIGNIFICANCE_LEGEND_FONT_SIZE,
        )):
            raise FigureInputError("注記・凡例の希望サイズはMIN_FONT_SIZE以上にしてください")

        # 全6比較を一律に最高の棒の上へ積まず、各比較の横範囲に
        # 必要な高さから詰める。線と文字の占有領域が重ならないようにする。
        intervals = []
        for level, annotation in enumerate(annotations):
            eye = str(annotation["Ocularity"])
            left = centers[(eye, str(annotation["Baseline_Condition"]))]
            right = centers[(eye, str(annotation["Condition"]))]
            left, right = sorted((left, right))
            local_high = max([
                reference,
                *[value for key, value in ci_tops.items()
                  if left - 0.02 <= centers[key] <= right + 0.02],
            ])
            floor = axis.transData.transform((0.0, local_high))[1]
            intervals.append((level, annotation, left, right, floor))
        # 同じ条件ペアをまとめ、必ず単眼→両眼の順に配置する。
        # CIの高さで眼条件を別々に並べ替えると上下が逆転するため、
        # ペア単位の必要高さ・横幅で配置順を決める。
        def comparison_pair(annotation):
            return tuple(sorted((
                str(annotation["Baseline_Condition"]),
                str(annotation["Condition"]),
            )))

        pair_priority = {}
        for _, annotation, left, right, floor in intervals:
            pair = comparison_pair(annotation)
            if pair not in pair_priority:
                pair_priority[pair] = (floor, right - left)
            else:
                previous_floor, previous_width = pair_priority[pair]
                pair_priority[pair] = (
                    min(previous_floor, floor), max(previous_width, right - left),
                )
        eye_priority = {"monocular": 0, "binocular": 1}
        intervals.sort(key=lambda item: (
            *pair_priority[comparison_pair(item[1])],
            comparison_pair(item[1]), eye_priority[str(item[1]["Ocularity"])],
            item[0],
        ))
        fitted = None
        for scale in np.linspace(
            1.0, _SIGNIFICANCE_MIN_FONT_SIZE / _SIGNIFICANCE_LABEL_FONT_SIZE, 51,
        ):
            label_size = max(_SIGNIFICANCE_MIN_FONT_SIZE, _SIGNIFICANCE_LABEL_FONT_SIZE * scale)
            legend_size = _SIGNIFICANCE_LEGEND_FONT_SIZE  # 凡例は固定サイズ。星だけ縮小する。
            legend = axis.legend(
                frameon=False, ncol=3, loc="upper center",
                bbox_to_anchor=(0.5, (region_top - bounds.y0) / bounds.height),
                fontsize=legend_size, borderaxespad=0.0,
                handlelength=1.2, handletextpad=0.5, columnspacing=0.8,
                borderpad=0.0,
            )
            for text in legend.get_texts():
                text.set_fontfamily([selected_font])
            probe = axis.text(
                0.5, 0.5, "n.s.", transform=axis.transAxes,
                fontsize=label_size, fontfamily=selected_font, va="bottom",
            )
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            text_height = probe.get_window_extent(renderer).height
            probe.remove()
            legend_box = legend.get_window_extent(renderer)
            cap = pt(_SIGNIFICANCE_CAP_HEIGHT_PT)
            label_gap = pt(_SIGNIFICANCE_LABEL_GAP_PT)
            row_gap = pt(_SIGNIFICANCE_ROW_GAP_PT)
            row_height = cap + label_gap + text_height
            annotation_top = legend_box.y0 - pt(_SIGNIFICANCE_LEGEND_GAP_PT)
            placed = []
            for level, annotation, left, right, floor in intervals:
                # xも表示座標にして、文字の横幅込みで衝突判定する。
                x0, x1 = axis.transData.transform([(left, lower), (right, lower)])[:, 0]
                prop = font_manager.FontProperties(family=selected_font, size=label_size)
                text_width = renderer.get_text_width_height_descent(
                    str(annotation["Label"]), prop, ismath=False,
                )[0]
                center_x = (x0 + x1) / 2.0
                x0 = min(x0, center_x - text_width / 2.0) - pt(1)
                x1 = max(x1, center_x + text_width / 2.0) + pt(1)
                bottom = floor + pt(_SIGNIFICANCE_DATA_GAP_PT)
                if str(annotation["Ocularity"]) == "binocular":
                    # 横方向の衝突の有無によらず、対応する単眼注記の
                    # 占有領域より上に両眼の比較線を配置する。
                    monocular = [
                        item for item in placed
                        if str(item["annotation"]["Ocularity"]) == "monocular"
                        and comparison_pair(item["annotation"]) == comparison_pair(annotation)
                    ]
                    if len(monocular) != 1:
                        raise FigureInputError("両眼比較線に対応する単眼比較線が一意ではありません")
                    bottom = max(bottom, monocular[0]["top"] + row_gap)
                for other in sorted(placed, key=lambda item: item["bottom"]):
                    overlaps_x = x0 < other["x1"] and x1 > other["x0"]
                    if (overlaps_x and bottom < other["top"] + row_gap
                            and bottom + row_height + row_gap > other["bottom"]):
                        bottom = other["top"] + row_gap
                if bottom + row_height > annotation_top:
                    break
                placed.append(dict(
                    level=level, annotation=annotation, left=left, right=right,
                    x0=x0, x1=x1, bottom=bottom, top=bottom + row_height,
                ))
            if (len(placed) == len(annotations)
                    and legend_box.x0 >= bounds.x0 and legend_box.x1 <= bounds.x1):
                fitted = (label_size, cap, label_gap, placed)
                break
            legend.remove()
        if fitted is None:
            raise FigureInputError(
                "1.0未満の空き領域に6比較と凡例が収まりません。"
                "_SIGNIFICANCE_FIGSIZEの高さを増やすか、文字サイズ・余白を小さくしてください。"
                "CIや値は切り捨てていません。"
            )
        label_size, cap, label_gap, placed = fitted
        transform = axis.get_xaxis_transform()
        for item in placed:
            level, annotation = item["level"], item["annotation"]
            eye = str(annotation["Ocularity"])
            left, right = item["left"], item["right"]
            y_px = item["bottom"] + cap
            y = (y_px - bounds.y0) / bounds.height
            cap_y = cap / bounds.height
            color = _EYE_COLORS[eye]
            line, = axis.plot(
                [left, left, right, right], [y - cap_y, y, y, y - cap_y],
                transform=transform, clip_on=True,
                color=color, linewidth=_SIGNIFICANCE_LINE_WIDTH, zorder=10,
            )
            line.set_gid(f"significance_bracket_{level}")
            text = axis.text(
                (left + right) / 2.0, y + label_gap / bounds.height,
                str(annotation["Label"]), transform=transform, clip_on=True,
                ha="center", va="bottom", fontsize=label_size,
                fontfamily=selected_font, color=color, zorder=11,
            )
            text.set_gid(f"significance_label_{level}")
        # ここから先はtight_layoutを呼ばない（測定済みの軸サイズを変えない）。
        return figure
    except Exception:
        _pyplot().close(figure)
        raise


def save_corrected_significance_figures(
    corrected_df: pd.DataFrame,
    hypothesis_tests: pd.DataFrame,
    output_dir: str | Path,
    *,
    uncorrected_output_dir: str | Path | None = None,
) -> list[Path]:
    """補正後3条件×2眼の6比較付きPNGを既存図とは別名で追加保存する。

    uncorrected_output_dir指定時は、同じパネルの未補正4条件_no_title PNGと
    画素数を一致させる。既存PNGは読み取りのみで変更しない。

    注記は既存の両側t検定のholm_adjusted_p_valueのみを参照する。
    H3内4検定・H4内3検定の補正を維持し、描画対象6行で再補正しない。
    H2/reference比較、H4 interaction、H6はこの図には注記しない。
    """
    validated = _validate_frame(
        corrected_df, label="有意差注記用DPF補正後参加者表",
        conditions=CORRECTED_CONDITIONS, value_column=CORRECTED_LOG10_COLUMN,
    )
    # 全パネルを先に照合する。欠損・重複・不正p値はn.s.扱いにしない。
    panels = []
    for values, group in validated.groupby(list(ANALYSIS_GROUP_COLUMNS), sort=True, dropna=False):
        metadata = dict(zip(ANALYSIS_GROUP_COLUMNS, values))
        annotations = _condition_significance_rows(group, hypothesis_tests, metadata)
        panels.append((group, metadata, annotations))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    from PIL import Image

    for group, metadata, annotations in panels:
        reference_pixel_size = None
        if uncorrected_output_dir is not None:
            reference_base = Path(uncorrected_output_dir) / _file_name("uncorrected", metadata)
            reference_path = reference_base.with_name(
                f"{reference_base.stem}_no_title{reference_base.suffix}"
            )
            if not reference_path.is_file():
                raise FigureInputError(f"サイズ基準の未補正PNGがありません: {reference_path}")
            with Image.open(reference_path) as image:
                reference_pixel_size = image.size
        figure = _corrected_significance_figure(
            group, metadata, annotations, reference_pixel_size=reference_pixel_size,
        )
        base_path = destination / _file_name("dpf_corrected", metadata)
        output_path = base_path.with_name(f"{base_path.stem}_significance_no_title{base_path.suffix}")
        try:
            # tight cropすると注記や軸名で縦横サイズが変わるため、この追加版だけ固定保存。
            with _pyplot().rc_context({"savefig.bbox": None}):
                figure.savefig(output_path, dpi=_DPI, bbox_inches=None, facecolor="white")
            if reference_pixel_size is not None:
                with Image.open(output_path) as image:
                    if image.size != reference_pixel_size:
                        raise FigureInputError("追加PNGの画素数が未補正4条件版と一致しません")
        finally:
            _pyplot().close(figure)
        outputs.append(output_path)
    return outputs


# ---- 4条件図の半分の横幅で描く追加図（DPF眼間・H4低減量・H6低減量比較） ----
# 見た目は*_significance_no_title.pngに揃える（点・m/d注記なし、t CI、括弧注記）。
_HALF_WIDTH_RATIO = 0.5


def _selected_font() -> str:
    from matplotlib import font_manager

    for candidate in ("Times New Roman", "Arial", "Helvetica", "Times"):
        try:
            font_manager.findfont(candidate, fallback_to_default=False)
        except ValueError:
            continue
        return candidate
    raise RuntimeError(
        "Times New Roman / Arial / Helvetica / Timesのいずれかをインストールしてください。"
    )


def _half_width_canvas(
    metadata: dict[str, object],
    uncorrected_output_dir: str | Path | None,
) -> tuple[tuple[float, float], tuple[int, int] | None]:
    """未補正4条件_no_title PNGと同じ高さ・半分の横幅のキャンバスを返す。"""
    if uncorrected_output_dir is None:
        width, height = _SIGNIFICANCE_FIGSIZE
        return (width * _HALF_WIDTH_RATIO, height), None
    from PIL import Image

    base = Path(uncorrected_output_dir) / _file_name("uncorrected", metadata)
    reference_path = base.with_name(f"{base.stem}_no_title{base.suffix}")
    if not reference_path.is_file():
        raise FigureInputError(f"サイズ基準の未補正PNGがありません: {reference_path}")
    with Image.open(reference_path) as image:
        width_px, height_px = image.size
    target = (int(width_px * _HALF_WIDTH_RATIO), int(height_px))
    return ((target[0] + 0.01) / _DPI, (target[1] + 0.01) / _DPI), target


def _effect_test_p(
    group: pd.DataFrame,
    tests: pd.DataFrame,
    metadata: dict[str, object],
    *,
    hypothesis: str,
    ocularity: str,
) -> float:
    """既存の両側t検定のHolm補正p値を1行だけ取得する（再検定・再補正しない）。"""
    required = [
        *ANALYSIS_GROUP_COLUMNS, "Participant_Aggregation", "Hypothesis",
        "Ocularity", "Data_State", "Analysis_Scale", "holm_adjusted_p_value", "n",
    ]
    missing = [column for column in required if column not in tests.columns]
    if missing:
        raise FigureInputError(f"有意差注記の検定表に必要な列がありません: {missing}")
    aggregations = group["Participant_Aggregation"].astype("string").str.strip()
    if aggregations.isna().any() or aggregations.nunique() != 1:
        raise FigureInputError("参加者表の集約方法を一意に特定できません")
    mask = (
        tests["Session_Type"].astype("string").eq(str(metadata["Session_Type"]))
        & tests["Participant_Aggregation"].astype("string").eq(str(aggregations.iloc[0]))
        & tests["Hypothesis"].eq(hypothesis)
        & tests["Ocularity"].eq(ocularity)
    )
    for column in ("Ref_Contrast", "Orientation"):
        numeric = pd.to_numeric(tests[column], errors="coerce").to_numpy(float)
        mask &= np.isclose(numeric, float(metadata[column]), rtol=0.0, atol=1e-12)
    rows = tests.loc[mask.fillna(False)]
    if hypothesis == "H4":
        # H4は眼別DP-SPD（H4-2/3）ではなく眼間差H4-1だけを使う。
        rows = rows.loc[rows["Component"].eq("H4-1")]
    if len(rows) != 1:
        raise FigureInputError(
            f"有意差注記の検定結果は1行必要です: {hypothesis}, {ocularity}, found={len(rows)}"
        )
    row = rows.iloc[0]
    if row["Data_State"] != "dpf_corrected" or row["Analysis_Scale"] != ANALYSIS_SCALE:
        raise FigureInputError("有意差注記の検定表の補正状態・解析尺度が一致しません")
    if float(row["n"]) != group["ID"].nunique():
        raise FigureInputError("図と検定の参加者数が一致しません")
    adjusted_p = float(row["holm_adjusted_p_value"])
    _significance_label(adjusted_p)  # 欠損・不正値をn.s.扱いにしない。
    return adjusted_p


def _half_width_bar_figure(
    bars: list[dict[str, object]],
    *,
    xticks: list[float],
    xticklabels: list[str],
    ylabel: str,
    contrast: bool,
    figsize: tuple[float, float],
    legend: bool,
    brackets: list[dict[str, object]],
    xlim: tuple[float, float] | None = None,
    reference: float | None = None,
):
    """bars: x・width・color・values(log10参加者値)。brackets: left・right・p・color。

    エラーバーは既存C_AR図と同じ参加者log10値の95% t CI。
    contrast=Trueでは10**で幾何平均へ戻し、0.05〜1.0のlog軸に描く。
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text

    font = _selected_font()
    figure, axis = _pyplot().subplots(figsize=figsize)
    try:
        floors: dict[float, float] = {}
        lows: list[float] = []
        highs: list[float] = []
        for bar in bars:
            mean, lower, upper = _mean_t_ci(np.asarray(bar["values"], dtype=float))
            if contrast:
                mean, lower, upper = (10.0 ** value for value in (mean, lower, upper))
            x = float(bar["x"])
            axis.bar(x, mean, width=float(bar["width"]), color=bar["color"],
                     alpha=0.82, zorder=2)
            axis.errorbar(x, mean, yerr=[[mean - lower], [upper - mean]], fmt="none",
                          ecolor="#222222", elinewidth=1.2, capsize=4, zorder=4)
            floors[x] = max(mean, upper, 0.0 if not contrast else upper)
            lows.append(min(mean, lower))
            highs.append(max(mean, upper))
        if contrast:
            _configure_contrast_axis(axis)
            if max(highs) >= 1.0:
                raise FigureInputError("CIが1.0以上のため表示範囲に収まりません")
            lower_limit = min(0.05, min(lows) / 1.1)
            axis.set_ylim(lower_limit, 1.0)
            ticks = sorted(set([*_CONTRAST_Y_TICKS, *([lower_limit] if lower_limit < 0.05 else [])]))
            axis.set_yticks(ticks)
            axis.set_yticklabels([
                "" if any(abs(value - hidden) < 1e-8 for hidden in (0.6, 0.7, 0.8, 0.9))
                else f"{value:.1f}" if 0.1 <= value <= 1.0 else f"{value:g}"
                for value in ticks
            ])
            upper_limit = 1.0
            if reference is not None:
                axis.axhline(reference, color="red", linestyle="--", linewidth=1.5, zorder=1)
        else:
            # 0〜10^0（上端=10^0）から始め、注記が収まらなければ上端を10^1にする。
            # 負のCIがある場合のみ下側へ広げる。
            if max(highs) >= 1.0:
                raise FigureInputError("CIが10^0以上のため表示範囲に収まりません")
            linscale = _H4_SYMLOG_LINSCALE
            lower_limit = 0.0 if min(lows) >= 0.0 else min(lows) * 1.1
            upper_limit = 1.0
            axis.axhline(0.0, color="#444444", linestyle="--", linewidth=1.2, zorder=1)

            def apply_symlog_axis() -> None:
                axis.set_yscale("symlog", base=10, linthresh=_H4_SYMLOG_LINTHRESH,
                                linscale=linscale)
                axis.set_ylim(lower_limit, upper_limit)
                # 0付近を縮めた場合は、ラベル重なりを避けるため±10^-2目盛りを省く。
                exponents = (0, -1, -2) if linscale >= _H4_SYMLOG_LINSCALE else (0, -1)
                negative = [-(10.0 ** k) for k in exponents if -(10.0 ** k) >= lower_limit]
                positive = [10.0 ** k for k in sorted(exponents)]
                positive += [10.0] if upper_limit > 1.0 else []
                axis.set_yticks([*negative, 0.0, *positive])
                axis.set_yticks([], minor=True)
                axis.tick_params(axis="y", which="both", direction="in")

            apply_symlog_axis()
        axis.set_xticks(xticks)
        axis.set_xticklabels(xticklabels)
        if xlim is not None:
            axis.set_xlim(*xlim)
        axis.grid(axis="y", which="both", linestyle=":", alpha=0.4)
        axis.set_ylabel(ylabel)
        for text in figure.findobj(Text):
            text.set_fontsize(_SIGNIFICANCE_AXIS_FONT_SIZE)
            text.set_fontfamily([font])
        axis.yaxis.label.set_fontsize(_SIGNIFICANCE_YLABEL_FONT_SIZE)

        px_per_pt = figure.dpi / 72.0
        pt = lambda value: value * px_per_pt
        # 半分幅で凡例が1行に収まらなければ、文字サイズを保って2行にする。
        legend_handles = [Patch(color=_EYE_COLORS[eye], alpha=0.82, label=_EYE_LABELS[eye])
                          for eye in OCULARITY_ORDER]
        if reference is not None:
            legend_handles.append(Line2D([], [], color="red", linestyle="--",
                                         linewidth=1.5, label="Reference"))
        legend_columns = len(legend_handles)
        for _ in range(80):
            if axis.get_legend() is not None:
                axis.get_legend().remove()
            figure.tight_layout()
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            bounds = axis.get_window_extent(renderer)
            region_top = bounds.y1 - pt(_SIGNIFICANCE_TOP_PADDING_PT)
            annotation_top = region_top
            legend_box = None
            if legend:
                handle = axis.legend(
                    handles=legend_handles,
                    frameon=False, ncol=legend_columns, loc="upper center",
                    bbox_to_anchor=(0.5, (region_top - bounds.y0) / bounds.height),
                    fontsize=_SIGNIFICANCE_LEGEND_FONT_SIZE, borderaxespad=0.0,
                    handlelength=1.2, handletextpad=0.5, columnspacing=0.8, borderpad=0.0,
                )
                for text in handle.get_texts():
                    text.set_fontfamily([font])
                figure.canvas.draw()
                legend_box = handle.get_window_extent(renderer)
                annotation_top = legend_box.y0 - pt(_SIGNIFICANCE_LEGEND_GAP_PT)
            # 既存の6比較図と同じく、表示時の行高さで占有領域を測る。
            probe = axis.text(0.5, 0.5, "n.s.", transform=axis.transAxes, va="bottom",
                              fontsize=_SIGNIFICANCE_LABEL_FONT_SIZE, fontfamily=font)
            figure.canvas.draw()
            text_height = probe.get_window_extent(renderer).height
            probe.remove()
            cap = pt(_SIGNIFICANCE_CAP_HEIGHT_PT)
            label_gap = pt(_SIGNIFICANCE_LABEL_GAP_PT)
            row_gap = pt(_SIGNIFICANCE_ROW_GAP_PT)
            row_height = cap + label_gap + text_height
            placed = []
            for bracket in brackets:
                left, right = sorted((float(bracket["left"]), float(bracket["right"])))
                local_high = max(value for x, value in floors.items() if left - 0.02 <= x <= right + 0.02)
                x0, x1 = axis.transData.transform([(left, local_high), (right, local_high)])[:, 0]
                bottom = axis.transData.transform((left, local_high))[1] + pt(_SIGNIFICANCE_DATA_GAP_PT)
                # 指定順（単眼→両眼）に下から積み、横に重なる括弧とは高さを分ける。
                for other in placed:
                    if x0 < other["x1"] and x1 > other["x0"]:
                        bottom = max(bottom, other["top"] + row_gap)
                placed.append(dict(bracket, left=left, right=right, x0=x0, x1=x1,
                                   bottom=bottom, top=bottom + row_height))
            fits = all(item["top"] <= annotation_top for item in placed)
            if legend_box is not None:
                fits &= legend_box.x0 >= bounds.x0 and legend_box.x1 <= bounds.x1
            if fits:
                break
            if legend_box is not None and legend_columns > 1 and (
                    legend_box.x0 < bounds.x0 or legend_box.x1 > bounds.x1):
                legend_columns -= 1
                continue
            if contrast or linscale <= _H4_SYMLOG_MIN_LINSCALE:
                raise FigureInputError("10^1未満の空き領域に注記と凡例が収まりません")
            if upper_limit < 10.0:
                # まず上端を10^1へ広げて注記領域を確保する。
                upper_limit = 10.0
            else:
                # それでも収まらない場合のみ、0〜10^-2区間の目盛り間隔を縮める。
                linscale = max(_H4_SYMLOG_MIN_LINSCALE, linscale * 0.85)
            apply_symlog_axis()
        else:
            raise FigureInputError("注記と凡例を図内に配置できません")
        transform = axis.get_xaxis_transform()
        for index, item in enumerate(placed):
            y = (item["bottom"] + cap - bounds.y0) / bounds.height
            cap_y = cap / bounds.height
            line, = axis.plot(
                [item["left"], item["left"], item["right"], item["right"]],
                [y - cap_y, y, y, y - cap_y], transform=transform, clip_on=True,
                color=item["color"], linewidth=_SIGNIFICANCE_LINE_WIDTH, zorder=10,
            )
            line.set_gid(f"significance_bracket_{index}")
            text = axis.text(
                (item["left"] + item["right"]) / 2.0, y + label_gap / bounds.height,
                _significance_label(float(item["p"])), transform=transform, clip_on=True,
                ha="center", va="bottom", fontsize=_SIGNIFICANCE_LABEL_FONT_SIZE,
                fontfamily=font, color=item["color"], zorder=11,
            )
            text.set_gid(f"significance_label_{index}")
        # ここから先はtight_layoutを呼ばない（測定済みの軸サイズを変えない）。
        return figure
    except Exception:
        _pyplot().close(figure)
        raise


def _save_half_width(figure, output_path: Path, target_px: tuple[int, int] | None) -> Path:
    """tight cropせず、4条件図の半分の画素幅で_no_title版を保存する。"""
    from PIL import Image

    path = output_path.with_name(f"{output_path.stem}_no_title{output_path.suffix}")
    try:
        with _pyplot().rc_context({"savefig.bbox": None}):
            figure.savefig(path, dpi=_DPI, bbox_inches=None, facecolor="white")
    finally:
        _pyplot().close(figure)
    if target_px is not None:
        with Image.open(path) as image:
            if image.size != target_px:
                raise FigureInputError("追加PNGの画素数が4条件図の半分幅と一致しません")
    return path


def _eye_pivot(group: pd.DataFrame, eye: str) -> pd.DataFrame:
    pivot = group.loc[
        group["Ocularity"] == eye, ["ID", "Condition", CORRECTED_LOG10_COLUMN],
    ].pivot(index="ID", columns="Condition", values=CORRECTED_LOG10_COLUMN)
    return pivot.sort_index()


def _h4_effects(group: pd.DataFrame) -> pd.DataFrame:
    """masking低減が正になるよう、眼別にSPD−DP（log10）を返す。"""
    effects = {
        eye: (lambda pivot: pivot[SPD_CONDITION] - pivot[DP_CONDITION])(_eye_pivot(group, eye))
        for eye in OCULARITY_ORDER
    }
    paired = pd.concat(
        [effects["monocular"].rename("monocular"), effects["binocular"].rename("binocular")],
        axis=1, join="outer",
    )
    if paired.isna().any().any():
        raise FigureInputError("H4 interactionの眼間参加者対応が不完全です")
    # 検定表H4-1（DP/SPDのBino−Mono）とは符号のみ逆。両側p値は同じ。
    paired["interaction"] = paired["binocular"] - paired["monocular"]
    return paired.sort_index()


def _h4_figure(group, metadata, hypothesis_tests=None, *, figsize):
    effects = _h4_effects(group)
    bars = [
        dict(x=float(index), width=0.62, color=_EYE_COLORS[eye], values=effects[eye].to_numpy(float))
        for index, eye in enumerate(("monocular", "binocular"))
    ]
    brackets = []
    if hypothesis_tests is not None:
        brackets.append(dict(left=0.0, right=1.0, color="black", p=_effect_test_p(
            group, hypothesis_tests, metadata,
            hypothesis="H4", ocularity="binocular_minus_monocular",
        )))
    return _half_width_bar_figure(
        bars, xticks=[0.0, 1.0], xticklabels=["Mono", "Bino"],
        ylabel=r"$C'_{\mathrm{AR}}$ reduction" + "\n" + r"$\log_{10}$(SPD/DP)",
        contrast=False, figsize=figsize, legend=False, brackets=brackets,
    )


def save_h4_interaction_figures(
    corrected_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    hypothesis_tests: pd.DataFrame | None = None,
    uncorrected_output_dir: str | Path | None = None,
) -> list[Path]:
    """H4の眼別SPD−DP低減量とinteractionを、4条件図の半分幅で描画する。"""
    validated = _validate_frame(
        corrected_df, label="DPF補正後参加者表",
        conditions=CORRECTED_CONDITIONS, value_column=CORRECTED_LOG10_COLUMN,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for values, group in validated.groupby(list(ANALYSIS_GROUP_COLUMNS), sort=True, dropna=False):
        metadata = dict(zip(ANALYSIS_GROUP_COLUMNS, values))
        figsize, target_px = _half_width_canvas(metadata, uncorrected_output_dir)
        figure = _h4_figure(group, metadata, hypothesis_tests, figsize=figsize)
        outputs.append(_save_half_width(
            figure, destination / _file_name("h4_interaction", metadata), target_px,
        ))
    return outputs


def save_successive_reduction_figures(
    corrected_df: pd.DataFrame,
    hypothesis_tests: pd.DataFrame,
    output_dir: str | Path,
    *,
    uncorrected_output_dir: str | Path | None = None,
) -> list[Path]:
    """H6: SP−SPDとSPD−DPの低減量を眼別に並べ、眼別の有意差を示す。"""
    validated = _validate_frame(
        corrected_df, label="DPF補正後参加者表",
        conditions=CORRECTED_CONDITIONS, value_column=CORRECTED_LOG10_COLUMN,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    width = 0.34
    for values, group in validated.groupby(list(ANALYSIS_GROUP_COLUMNS), sort=True, dropna=False):
        metadata = dict(zip(ANALYSIS_GROUP_COLUMNS, values))
        bars, brackets = [], []
        for eye_index, eye in enumerate(OCULARITY_ORDER):
            pivot = _eye_pivot(group, eye)
            offset = (eye_index - 0.5) * width
            for x, reduction in enumerate((
                pivot[SP_CONDITION] - pivot[SPD_CONDITION],
                pivot[SPD_CONDITION] - pivot[DP_CONDITION],
            )):
                bars.append(dict(x=x + offset, width=width * 0.92, color=_EYE_COLORS[eye],
                                 values=reduction.to_numpy(float)))
            brackets.append(dict(left=offset, right=1.0 + offset, color=_EYE_COLORS[eye],
                                 p=_effect_test_p(group, hypothesis_tests, metadata,
                                                  hypothesis="H6", ocularity=eye)))
        figsize, target_px = _half_width_canvas(metadata, uncorrected_output_dir)
        figure = _half_width_bar_figure(
            bars, xticks=[0.0, 1.0],
            xticklabels=[r"SP$\rightarrow$SPD", r"SPD$\rightarrow$DP"],
            ylabel=r"$C'_{\mathrm{AR}}$ reduction" + "\n" + r"($\log_{10}$ ratio)",
            contrast=False, figsize=figsize, legend=True, brackets=brackets,
        )
        outputs.append(_save_half_width(
            figure, destination / _file_name("successive_reductions", metadata), target_px,
        ))
    return outputs


def save_dpf_ocularity_figures(
    uncorrected_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    uncorrected_output_dir: str | Path | None = None,
) -> list[Path]:
    """未補正DPFのC_ARを単眼・両眼の2本で示す（reference線あり・有意差なし）。"""
    validated = _validate_frame(
        uncorrected_df, label="未補正参加者表",
        conditions=CONDITION_ORDER, value_column=MEAN_LOG10_COLUMN,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    width = 0.34
    for values, group in validated.groupby(list(ANALYSIS_GROUP_COLUMNS), sort=True, dropna=False):
        metadata = dict(zip(ANALYSIS_GROUP_COLUMNS, values))
        bars = [
            dict(x=(eye_index - 0.5) * width, width=width * 0.92, color=_EYE_COLORS[eye],
                 values=group.loc[group["Ocularity"].eq(eye) & group["Condition"].eq(DPF_CONDITION),
                                  MEAN_LOG10_COLUMN].to_numpy(float))
            for eye_index, eye in enumerate(OCULARITY_ORDER)
        ]
        figsize, target_px = _half_width_canvas(metadata, uncorrected_output_dir)
        figure = _half_width_bar_figure(
            bars, xticks=[0.0], xticklabels=[_CONDITION_LABELS[DPF_CONDITION]],
            ylabel="Matched contrast\n" + r"$C_{\mathrm{AR}}$ (geometric mean)",
            # 4条件図と同じ棒幅になるよう、2条件分の横範囲を確保する。
            contrast=True, figsize=figsize, legend=True, brackets=[], xlim=(-1.0, 1.0),
            reference=float(metadata["Ref_Contrast"]),
        )
        outputs.append(_save_half_width(
            figure, destination / _file_name("dpf_ocularity", metadata), target_px,
        ))
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


def _bootstrap_linear_median_ci(
    values,
    *,
    seed_parts: tuple[object, ...],
) -> tuple[float, float, float]:
    """試行を再標本化し、中央値と95% percentile CIを返す。"""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise FigureInputError("中央値CIには1件以上の有限な試行値が必要です")
    median = float(np.median(array))
    if len(array) == 1 or np.allclose(array, array[0]):
        return median, median, median
    random = np.random.default_rng(_stable_seed(seed_parts))
    samples = random.choice(
        array,
        size=(_BOOTSTRAP_SAMPLES, len(array)),
        replace=True,
    )
    lower, upper = np.quantile(np.median(samples, axis=1), [0.025, 0.975])
    return median, float(lower), float(upper)


def _legacy_number_token(value: float) -> str:
    return f"{float(value):g}"


def _legacy_file_name(
    metric_name: str,
    metadata: dict[str, object],
    *,
    center_method: str,
) -> str:
    session_type = sanitize_run_component(str(metadata["Session_Type"]))
    reference = _legacy_number_token(float(metadata["Ref_Contrast"]))
    orientation = _legacy_number_token(float(metadata["Orientation"]))
    center_token = "" if center_method == "mean" else "median_"
    return (
        f"{session_type}_matched_{metric_name}_{center_token}"
        f"ref_{reference}_ori_{orientation}.png"
    )


def _legacy_metric_figure(
    group: pd.DataFrame,
    *,
    value_column: str,
    ylabel: str,
    metric_name: str,
    metadata: dict[str, object],
    center_method: str,
):
    plt = _pyplot()
    if center_method not in {"mean", "median"}:
        raise FigureInputError(
            f"未定義の中心統計量です: {center_method}"
        )

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
        trial_values: list[np.ndarray] = []
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
            trial_values.append(values)
            seed_parts = (
                "legacy",
                center_method,
                metric_name,
                *metadata.values(),
                eye,
                condition,
            )
            if center_method == "mean":
                center, lower, upper = _bootstrap_linear_mean_ci(
                    values,
                    seed_parts=seed_parts,
                )
                spread = (
                    float(np.std(values, ddof=1))
                    if len(values) > 1
                    else float("nan")
                )
            else:
                center, lower, upper = _bootstrap_linear_median_ci(
                    values,
                    seed_parts=seed_parts,
                )
                spread = (
                    float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
                    if len(values) > 1
                    else float("nan")
                )
            means.append(center)
            lowers.append(lower)
            uppers.append(upper)
            standard_deviations.append(spread)
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
        if center_method == "mean":
            _annotate_mean_and_sd(
                axis,
                bars,
                means,
                standard_deviations,
            )
        else:
            _annotate_median_and_iqr(
                axis,
                bars,
                means,
                standard_deviations,
            )
        # 各バー上へ、その条件・眼条件に含まれる全試行を表示する。
        for position, values in zip(positions, trial_values):
            jitter = np.linspace(
                -width * 0.18,
                width * 0.18,
                len(values),
            )
            axis.scatter(
                np.full(len(values), position) + jitter,
                values,
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
        linewidth=2,
        label=f"Ref Contrast ({reference:g})",
        zorder=1,
    )
    center_label = "Mean" if center_method == "mean" else "Median"
    axis.set_title(
        f"{metadata['Session_Type']}: {ylabel} ({center_label}) "
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

    center_methods = ("mean", "median")
    outputs: dict[str, list[Path]] = {
        (metric_name if method == "mean" else f"{metric_name}_median"): []
        for metric_name, _, _ in _LEGACY_METRIC_SPECS
        for method in center_methods
    }
    for group_values, group in validated.groupby(
        list(ANALYSIS_GROUP_COLUMNS),
        sort=True,
        dropna=False,
    ):
        metadata = dict(zip(ANALYSIS_GROUP_COLUMNS, group_values))
        for metric_name, value_column, ylabel in _LEGACY_METRIC_SPECS:
            for center_method in center_methods:
                output_key = (
                    metric_name
                    if center_method == "mean"
                    else f"{metric_name}_median"
                )
                figure = _legacy_metric_figure(
                    group,
                    value_column=value_column,
                    ylabel=ylabel,
                    metric_name=metric_name,
                    metadata=metadata,
                    center_method=center_method,
                )
                output_path = destination / _legacy_file_name(
                    metric_name,
                    metadata,
                    center_method=center_method,
                )
                try:
                    saved_paths = _save_with_title_variants(
                        figure, output_path, dpi=_LEGACY_DPI
                    )
                finally:
                    plt.close(figure)
                outputs[output_key].extend(saved_paths)
    return outputs


def save_all_figures(
    uncorrected_df: pd.DataFrame,
    corrected_df: pd.DataFrame,
    output_paths: AnalysisOutputPaths,
    *,
    hypothesis_tests: pd.DataFrame | None = None,
) -> dict[str, list[Path]]:
    """既存図を維持し、検定表があれば6比較注記付き別PNGを追加する。"""
    outputs = {
        "uncorrected": save_uncorrected_figures(
            uncorrected_df, output_paths.uncorrected_figure_dir
        ),
        "dpf_corrected": save_corrected_figures(
            corrected_df, output_paths.corrected_figure_dir
        ),
    }
    # 半分幅の図は未補正4条件_no_title PNGの画素数を基準にする。
    outputs["h4_interaction"] = save_h4_interaction_figures(
        corrected_df, output_paths.h4_figure_dir, hypothesis_tests=hypothesis_tests,
        uncorrected_output_dir=output_paths.uncorrected_figure_dir,
    )
    outputs["dpf_ocularity"] = save_dpf_ocularity_figures(
        uncorrected_df, output_paths.uncorrected_figure_dir,
        uncorrected_output_dir=output_paths.uncorrected_figure_dir,
    )
    if hypothesis_tests is not None:
        outputs["successive_reductions"] = save_successive_reduction_figures(
            corrected_df, hypothesis_tests, output_paths.h4_figure_dir,
            uncorrected_output_dir=output_paths.uncorrected_figure_dir,
        )
        outputs["dpf_corrected_significance"] = save_corrected_significance_figures(
            corrected_df, hypothesis_tests, output_paths.corrected_figure_dir,
            uncorrected_output_dir=output_paths.uncorrected_figure_dir,
        )
    return outputs


__all__ = [
    "FigureInputError",
    "save_all_figures",
    "save_corrected_figures",
    "save_corrected_significance_figures",
    "save_dpf_ocularity_figures",
    "save_h4_interaction_figures",
    "save_legacy_contrast_figures",
    "save_successive_reduction_figures",
    "save_uncorrected_figures",
]