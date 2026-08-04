"""
現行の main_experiment_matching.py に対応した解析スクリプト。

入力:
  results/tables/main-experiment-matching/experiment/
    {participant_id}_YYYYMMDD_HHMMSS/contrast_matching.csv

出力:
  results/figures/main-experiment-matching/experiment/
    analysis_YYYYMMDD_HHMMSS/

Defocus処理は実験と同じcommon.opticsを使用する。
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
from experiment.common import geometry, optics, patterns


EXPERIMENT_RESULT_ROOT = os.path.join(
    LAB_ROOT, "results", "tables", "main-experiment-matching", "experiment"
)
EXPERIMENT_FIGURE_ROOT = os.path.join(
    LAB_ROOT, "results", "figures", "main-experiment-matching", "experiment"
)
TRAINING_FIGURE_ROOT = os.path.join(
    LAB_ROOT, "results", "figures", "main-experiment-matching", "training"
)
COMBINED_FIGURE_ROOT = os.path.join(
    LAB_ROOT, "results", "figures", "main-experiment-matching", "combined"
)

VISUAL_ANGLE_WIDTH_DEG = 7.9
VISUAL_ANGLE_HEIGHT_DEG = 3.95
WIN2_TOTAL_WIDTH_FACTOR = 2.6
SPATIAL_FREQ_CPD = 4.0
DEFAULT_L_BG = 15.0
BACKGROUND_CONTRAST = 1.0

REQUIRED_COLUMNS = [
    "ID", "Condition", "Ocularity", "Ref_Contrast", "Matched_Contrast",
    "L_fg", "L_bg", "L_ref", "Dominance", "PD_Right", "PD_Left",
"Orientation",
]

_blur_attenuation_cache = {}


def matching_result_path(session_dir):
    """sessionフォルダから本実験またはtrainingの結果CSVを判定する。"""
    candidates = [
        os.path.join(session_dir, "contrast_matching.csv"),
        os.path.join(session_dir, "contrast_matching_training.csv"),
    ]
    existing = [path for path in candidates if os.path.isfile(path)]
    if len(existing) > 1:
        raise ValueError(f"本実験とtrainingのCSVが同じフォルダにあります: {session_dir}")
    return existing[0] if existing else None


def discover_session_dirs(explicit_dirs=None):
    """セッションフォルダ、またはその親フォルダから解析対象を見つける。"""
    if explicit_dirs:
        candidate_dirs = []
        for path in explicit_dirs:
            path = os.path.abspath(path)
            if matching_result_path(path) is not None:
                # 例: .../experiment/3_20260730_164234
                candidate_dirs.append(path)
            elif os.path.isdir(path):
                # 例: .../experiment を指定した場合は、直下の全セッションを使う。
                candidate_dirs.extend(
                    sorted(
                        child for child in glob.glob(os.path.join(path, "*"))
                        if os.path.isdir(child)
                    )
                )
            else:
                print(f"WARN: ディレクトリが見つかりません: {path}")
    else:
        candidate_dirs = sorted(
            path for path in glob.glob(os.path.join(EXPERIMENT_RESULT_ROOT, "*"))
            if os.path.isdir(path)
        )

    valid_dirs = []
    for session_dir in candidate_dirs:
        csv_path = matching_result_path(session_dir)
        if csv_path is not None:
            valid_dirs.append(session_dir)
        else:
            print(
                "WARN: contrast_matching.csv / contrast_matching_training.csv "
                f"がないため除外: {session_dir}"
            )
    return list(dict.fromkeys(valid_dirs))


def load_matching_results(session_dirs):
    frames = []
    for session_dir in session_dirs:
        csv_path = matching_result_path(session_dir)
        if csv_path is None:
            continue
        df = pd.read_csv(csv_path, encoding="utf-8")
        missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"{csv_path}: 必須列が不足しています: {missing}")
        session_type = (
            "training"
            if os.path.basename(csv_path) == "contrast_matching_training.csv"
            else "experiment"
        )
        df["Session_Type"] = session_type
        frames.append(df)
        print(f"Loaded ({session_type}): {csv_path}")

    if not frames:
        raise FileNotFoundError(
            "解析可能な contrast_matching.csv または "
            "contrast_matching_training.csv がありません"
        )
    return pd.concat(frames, ignore_index=True)


def dominant_pd_mm(row):
    dominance = str(row.get("Dominance", "Right")).strip().lower()
    value = row["PD_Left"] if dominance == "left" else row["PD_Right"]
    return float(value)


def calculate_blur_attenuation_cached(
    pd_mm,
    d_fg=50.0,
    d_bg=150.0,
    f_center_cpd=SPATIAL_FREQ_CPD,
):
    """実験と同じ輝度配列用FFTデフォーカス処理からRMS減衰率を求める。"""
    key = (
        round(float(pd_mm), 2), round(float(d_fg), 3),
        round(float(d_bg), 3), round(float(f_center_cpd), 3),
    )
    if key in _blur_attenuation_cache:
        return _blur_attenuation_cache[key]

    if pd_mm <= 0 or d_fg <= 0 or d_bg <= 0:
        _blur_attenuation_cache[key] = 1.0
        return 1.0

    ppd_fg = geometry.get_size_for_visual_angle(d_fg, 1.0)
    width_base = int(VISUAL_ANGLE_WIDTH_DEG * ppd_fg)
    width_px = int(width_base * WIN2_TOTAL_WIDTH_FACTOR)
    height_px = int(VISUAL_ANGLE_HEIGHT_DEG * ppd_fg)

    # 実験のnoise生成と同じ関数を、解析ではseed固定で決定論的に使う。
    noise_base = patterns.create_noise_base(
        width_px,
        height_px,
        ppd_fg,
        f_center_cpd,
        rng=np.random.default_rng(42),
    )

    lum_original = DEFAULT_L_BG * (1.0 + BACKGROUND_CONTRAST * noise_base)
    d_fg_m = d_fg / 100.0
    d_bg_m = d_bg / 100.0
    diopter_difference = abs(1.0 / d_fg_m - 1.0 / d_bg_m)

    lum_blurred = optics.apply_defocus_blur_to_luminance(
        lum_original, diopter_difference, float(pd_mm), ppd_fg
    )

    rms_original = float(np.std(lum_original))
    rms_blurred = float(np.std(lum_blurred))
    attenuation = 1.0 if rms_original <= 1e-12 else rms_blurred / rms_original
    attenuation = float(np.clip(attenuation, 0.0, 1.0))
    _blur_attenuation_cache[key] = attenuation
    return attenuation


def add_analysis_columns(df):
    analyzed = df.copy()
    analyzed["Dominant_PD_mm"] = analyzed.apply(dominant_pd_mm, axis=1)

    def attenuation_for_row(row):
        condition = str(row["Condition"])
        if condition not in {"Single plane + defocus simulation", "Dual plane"}:
            return 1.0
        return calculate_blur_attenuation_cached(row["Dominant_PD_mm"])

    analyzed["Blur_Attenuation"] = analyzed.apply(attenuation_for_row, axis=1)
    analyzed["Effective_BG_Contrast"] = (
        BACKGROUND_CONTRAST * analyzed["Blur_Attenuation"]
    )
    analyzed["AR_Matched_Contrast"] = (
        analyzed["L_fg"] * analyzed["Matched_Contrast"]
        / (analyzed["L_fg"] + analyzed["L_bg"])
    )
    analyzed["matched_ar_contrast"] = analyzed["AR_Matched_Contrast"]
    analyzed["Matched_Contrast_Enhanced"] = (
        analyzed["Matched_Contrast"] * analyzed["L_fg"]
        + analyzed["Effective_BG_Contrast"] * analyzed["L_bg"]
    ) / (analyzed["L_fg"] + analyzed["L_bg"])
    return analyzed


def _cohens_d_one_sample(values, reference):
    """参照コントラストとの差を標本SDで標準化した効果量。"""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    sd = np.std(values, ddof=1)
    difference = float(np.mean(values) - reference)
    if sd == 0:
        return 0.0 if difference == 0 else float(np.sign(difference) * np.inf)
    return difference / sd


def _cohens_d_paired(values_a, values_b):
    """対応ありデータの差分に対するCohen's d（d_z）。"""
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    valid = np.isfinite(values_a) & np.isfinite(values_b)
    if valid.sum() < 2:
        return float("nan")
    differences = values_a[valid] - values_b[valid]
    mean_difference = float(np.mean(differences))
    sd = np.std(differences, ddof=1)
    if sd == 0:
        return 0.0 if mean_difference == 0 else float(np.sign(mean_difference) * np.inf)
    return mean_difference / sd


def _holm_adjust(p_values):
    """Holm-Bonferroni補正済みp値（NaNはそのまま）を返す。"""
    adjusted = np.full(len(p_values), np.nan)
    valid = [i for i, p in enumerate(p_values) if np.isfinite(p)]
    ordered = sorted(valid, key=lambda i: p_values[i])
    running_maximum = 0.0
    for rank, index in enumerate(ordered):
        running_maximum = max(
            running_maximum, (len(ordered) - rank) * p_values[index]
        )
        adjusted[index] = min(running_maximum, 1.0)
    return adjusted


def _posthoc_power_and_required_n(effect_size, n, alpha=0.05, target_power=0.80):
    """両側1標本t検定の事後検定力と、80%検定力の必要人数を返す。"""
    from scipy import stats

    if not np.isfinite(effect_size) or effect_size == 0:
        return float("nan"), float("inf")
    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    z_power = stats.norm.ppf(target_power)
    required_n = int(np.ceil(((z_alpha + z_power) / abs(effect_size)) ** 2))
    if n < 2:
        return float("nan"), required_n
    degrees_of_freedom = n - 1
    critical_t = stats.t.ppf(1.0 - alpha / 2.0, degrees_of_freedom)
    noncentrality = effect_size * np.sqrt(n)
    observed_power = (
        stats.nct.cdf(-critical_t, degrees_of_freedom, noncentrality)
        + stats.nct.sf(critical_t, degrees_of_freedom, noncentrality)
    )
    return float(observed_power), required_n


def save_statistical_csv_outputs(df, output_dir):
    """4条件×2眼条件の統計解析をCSVへ保存する。

    各参加者の反復試行を条件内で平均して独立な解析単位とし、
    Ref_Contrast・Orientation・Session_Typeごとに8本の両側1標本t検定を
    行う。各8本のp値にHolm-Bonferroni補正を適用する。
    """
    from scipy import stats

    value_column = "matched_ar_contrast"
    condition_order = [
        "Single plane", "Single plane + defocus simulation",
        "Dual plane", "Dual plane flat",
    ]
    ocularity_order = ["binocular", "monocular"]
    keys = [
        "ID", "Session_Type", "Condition", "Ocularity",
        "Ref_Contrast", "Orientation",
    ]
    participant_means = (
        df[keys + [value_column]]
        .dropna(subset=keys + [value_column])
        .groupby(keys, as_index=False)[value_column].mean()
    )

    test_rows = []
    analysis_groups = ["Session_Type", "Ref_Contrast", "Orientation"]
    for group_values, group_df in participant_means.groupby(analysis_groups):
        session_type, reference, orientation = group_values
        eight_rows = []
        for condition in condition_order:
            for ocularity in ocularity_order:
                condition_df = group_df.loc[
                    (group_df["Condition"] == condition)
                    & (group_df["Ocularity"] == ocularity)
                ].copy()
                reference_df = group_df.loc[
                    (group_df["Condition"] == "Single plane")
                    & (group_df["Ocularity"] == ocularity)
                ].copy()

                if condition == "Single plane":
                    values = condition_df[value_column].to_numpy(dtype=float)
                    values = values[np.isfinite(values)]
                    n = len(values)
                    mean = float(np.mean(values)) if n else float("nan")
                    sd = float(np.std(values, ddof=1)) if n >= 2 else float("nan")
                    sem = sd / np.sqrt(n) if n >= 2 else float("nan")
                    reference_mean = mean if n else float("nan")
                    mean_difference = 0.0 if n else float("nan")
                    t_statistic = p_value = ci_low = ci_high = float("nan")
                    cohens_d = float("nan")
                    observed_power, required_n = float("nan"), float("inf")
                else:
                    merged = condition_df.merge(
                        reference_df[["ID", value_column]].rename(
                            columns={value_column: f"{value_column}_baseline"}
                        ),
                        on="ID",
                        how="inner",
                    )
                    values = merged[value_column].to_numpy(dtype=float)
                    reference_values = merged[f"{value_column}_baseline"].to_numpy(dtype=float)
                    values = values[np.isfinite(values)]
                    reference_values = reference_values[np.isfinite(reference_values)]
                    n = len(values)
                    mean = float(np.mean(values)) if n else float("nan")
                    sd = float(np.std(values, ddof=1)) if n >= 2 else float("nan")
                    sem = sd / np.sqrt(n) if n >= 2 else float("nan")
                    reference_mean = float(np.mean(reference_values)) if len(reference_values) else float("nan")
                    mean_difference = float(np.mean(values - reference_values)) if n else float("nan")
                    if n >= 2:
                        result = stats.ttest_rel(values, reference_values)
                        t_statistic, p_value = float(result.statistic), float(result.pvalue)
                        diff_sd = float(np.std(values - reference_values, ddof=1)) if n >= 2 else float("nan")
                        sem_diff = diff_sd / np.sqrt(n) if n >= 2 else float("nan")
                        ci_low, ci_high = stats.t.interval(
                            0.95, n - 1, loc=mean_difference, scale=sem_diff
                        )
                    else:
                        t_statistic = p_value = ci_low = ci_high = float("nan")
                    cohens_d = _cohens_d_paired(values, reference_values)
                    observed_power, required_n = _posthoc_power_and_required_n(cohens_d, n)

                eight_rows.append({
                    "Session_Type": session_type,
                    "Ref_Contrast": reference,
                    "Orientation": orientation,
                    "Condition": condition,
                    "Ocularity": ocularity,
                    "n_participants": n,
                    "mean_matched_ar_contrast": mean,
                    "sd_matched_ar_contrast": sd,
                    "sem_matched_ar_contrast": sem,
                    "reference_condition": "Single plane",
                    "reference_single_plane_mean": reference_mean,
                    "reference_contrast": reference_mean,
                    "mean_difference_from_reference": mean_difference,
                    "ci95_lower": ci_low,
                    "ci95_upper": ci_high,
                    "t_statistic": t_statistic,
                    "degrees_of_freedom": n - 1 if n else float("nan"),
                    "p_value_two_sided": p_value,
                    "cohens_d": cohens_d,
                    "posthoc_power": observed_power,
                    "required_n_for_80pct_power": required_n,
                })
        adjusted = _holm_adjust([row["p_value_two_sided"] for row in eight_rows])
        for row, adjusted_p in zip(eight_rows, adjusted):
            row["holm_adjusted_p_value"] = adjusted_p
            row["significant_holm_alpha_0_05"] = bool(
                np.isfinite(adjusted_p) and adjusted_p < 0.05
            )
        test_rows.extend(eight_rows)

    pd.DataFrame(test_rows).to_csv(
        os.path.join(output_dir, "one_sample_ttests_holm_cohens_d_power.csv"),
        index=False, encoding="utf-8-sig",
    )

    # 同一参加者・同一眼条件・同一参照値・同一方位のSingle plane値を
    # ベースラインにして、個人固有のマッチング偏りを除去する。
    baseline = participant_means[
        participant_means["Condition"] == "Single plane"
    ].drop(columns=["Condition"]).rename(
        columns={value_column: "single_plane_matched_ar_contrast"}
    )
    corrected = participant_means.merge(
        baseline,
        on=["ID", "Session_Type", "Ocularity", "Ref_Contrast", "Orientation"],
        how="left",
    )
    corrected["bias_corrected_difference_from_single_plane"] = (
        corrected[value_column] - corrected["single_plane_matched_ar_contrast"]
    )
    corrected.to_csv(
        os.path.join(output_dir, "participant_bias_corrected_values.csv"),
        index=False, encoding="utf-8-sig",
    )
    (
        corrected.groupby(
            ["Session_Type", "Ref_Contrast", "Orientation", "Condition", "Ocularity"],
            as_index=False,
        )["bias_corrected_difference_from_single_plane"]
        .agg(["count", "mean", "std", "sem"])
        .reset_index()
        .rename(columns={
            "count": "n_participants",
            "mean": "mean_bias_corrected_difference",
            "std": "sd_bias_corrected_difference",
            "sem": "sem_bias_corrected_difference",
        })
        .to_csv(
            os.path.join(output_dir, "bias_corrected_summary.csv"),
            index=False, encoding="utf-8-sig",
        )
    )
    print(f"Saved statistical CSV outputs: {output_dir}")


def save_outputs(df, output_dir):
    """解析結果はグラフだけを保存する。"""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 旧版と同じ、Condition × Ocularityの棒グラフだけを出力する。
    condition_order = [
        "Single plane",
        "Single plane + defocus simulation",
        "Dual plane",
        "Dual plane flat",
    ]
    ocularity_order = ["monocular", "binocular"]
    bar_specs = [
        ("Matched_Contrast", "raw", "Matched Contrast (Raw)"),
        ("matched_ar_contrast", "ar", "Matched Contrast (AR Extended)"),
        (
            "Matched_Contrast_Enhanced",
            "enhanced",
            "Matched Contrast (Enhanced)",
        ),
    ]

    for session_type in sorted(df["Session_Type"].dropna().unique()):
        session_df = df[df["Session_Type"] == session_type]
        ref_contrasts = sorted(
            session_df["Ref_Contrast"].dropna().unique(), reverse=True
        )
        orientations = sorted(session_df["Orientation"].dropna().unique())

        for ref_contrast in ref_contrasts:
            for orientation in orientations:
                plot_df = session_df[
                    (session_df["Ref_Contrast"] == ref_contrast)
                    & (session_df["Orientation"] == orientation)
                ]
                if plot_df.empty:
                    continue

                for value_column, suffix, ylabel in bar_specs:
                    fig, ax = plt.subplots(figsize=(11, 6))
                    sns.barplot(
                        data=plot_df,
                        x="Condition",
                        y=value_column,
                        hue="Ocularity",
                        order=condition_order,
                        hue_order=ocularity_order,
                        errorbar=("ci", 95),
                        capsize=0.1,
                        err_kws={"linewidth": 1.5},
                        ax=ax,
                    )
                    ax.axhline(
                        y=ref_contrast,
                        color="red",
                        linestyle="--",
                        linewidth=2,
                        label=f"Ref Contrast ({ref_contrast})",
                    )

                    # 各バーの足元に平均値(m)と標準偏差(d)を表示する。
                    grouped = (
                        plot_df.groupby(["Condition", "Ocularity"])[value_column]
                        .agg(["mean", "std"])
                    )
                    for container_index, container in enumerate(ax.containers):
                        if container_index >= len(ocularity_order):
                            continue
                        ocularity = ocularity_order[container_index]
                        for condition_index, bar in enumerate(container):
                            if condition_index >= len(condition_order):
                                continue
                            condition = condition_order[condition_index]
                            key = (condition, ocularity)
                            if key not in grouped.index:
                                continue
                            mean_value = grouped.loc[key, "mean"]
                            std_value = grouped.loc[key, "std"]
                            if not np.isfinite(mean_value):
                                continue
                            std_text = (
                                "nan" if pd.isna(std_value) else f"{std_value:.2f}"
                            )
                            x_position = bar.get_x() + bar.get_width() / 2
                            ax.text(
                                x_position,
                                0.055,
                                f"m={mean_value:.2f}\nd={std_text}",
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

                    ax.set_title(
                        f"{session_type}: {ylabel} "
                        f"(Ref={ref_contrast}, Ori={orientation}°)"
                    )
                    ax.set_ylabel(ylabel)
                    ax.set_xlabel("Condition")
                    ax.set_yscale("log")
                    ax.set_ylim(0.05, 1.0)
                    ax.set_yticks(np.arange(0.1, 1.01, 0.1))
                    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
                    ax.tick_params(axis="x", rotation=0)

                    handles, legend_labels = ax.get_legend_handles_labels()
                    unique_legend = dict(zip(legend_labels, handles))
                    ax.legend(
                        unique_legend.values(),
                        unique_legend.keys(),
                        loc="upper left",
                    )
                    plt.tight_layout()

                    output_filename = (
                        f"{session_type}_matched_{suffix}_contrast_"
                        f"ref_{ref_contrast}_ori_{int(orientation)}.png"
                    )
                    plt.savefig(
                        os.path.join(output_dir, output_filename),
                        dpi=300,
                    )
                    plt.close(fig)
                    print(f"Saved bar chart: {output_filename}")

    print(f"Saved analysis outputs: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze contrast matching results."
    )
    parser.add_argument(
        "session_dirs",
        nargs="*",
        help="省略時はexperiment配下を自動探索。指定時は参加者・日時フォルダを列挙。",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="省略時はresults/figures側にanalysis_日時フォルダを作成。",
    )
    args = parser.parse_args()

    session_dirs = discover_session_dirs(args.session_dirs)
    if not session_dirs:
        raise FileNotFoundError(
            f"解析対象がありません。既定探索先: {EXPERIMENT_RESULT_ROOT}"
        )

    df = load_matching_results(session_dirs)
    analyzed = add_analysis_columns(df)

    session_types = set(analyzed["Session_Type"].dropna().astype(str))
    if session_types == {"training"}:
        default_figure_root = TRAINING_FIGURE_ROOT
    elif session_types == {"experiment"}:
        default_figure_root = EXPERIMENT_FIGURE_ROOT
    else:
        default_figure_root = COMBINED_FIGURE_ROOT

    session_timestamps = []
    for session_dir in session_dirs:
        folder_name = os.path.basename(os.path.normpath(session_dir))
        match = re.search(r"(\d{8}_\d{6})$", folder_name)
        if match:
            session_timestamps.append(match.group(1))

    if not session_timestamps:
        raise ValueError(
            "実験結果フォルダ名から日時を取得できませんでした。"
            "フォルダ名は ID_YYYYMMDD_HHMMSS の形式にしてください。"
        )

    # 複数セッションの場合は、解析対象中で最も新しい実験日時を使う。
    timestamp = max(session_timestamps)

    participant_ids = sorted(
        analyzed["ID"].dropna().astype(str).str.strip().unique()
    )
    participant_ids = [value for value in participant_ids if value]
    participant_label = "-".join(participant_ids) or "unknown"
    participant_label = re.sub(r"[^0-9A-Za-z_-]+", "_", participant_label)

    all_participants_mode = (
        not args.session_dirs
        or any(
            os.path.abspath(path) == os.path.abspath(EXPERIMENT_RESULT_ROOT)
            for path in args.session_dirs
        )
    )
    figure_output_dir = args.output_dir or os.path.join(
        default_figure_root,
        "all_participants"
        if all_participants_mode
        else f"analyze_{participant_label}_{timestamp}",
    )

    if all_participants_mode:
        stats_output_dir = os.path.join(EXPERIMENT_RESULT_ROOT, "all_participants")
    elif len(session_dirs) == 1:
        stats_output_dir = os.path.abspath(session_dirs[0])
    else:
        stats_output_dir = os.path.join(
            EXPERIMENT_RESULT_ROOT,
            f"analyze_{participant_label}_{timestamp}",
        )

    save_outputs(analyzed, figure_output_dir)
    save_statistical_csv_outputs(analyzed, stats_output_dir)


if __name__ == "__main__":
    main()