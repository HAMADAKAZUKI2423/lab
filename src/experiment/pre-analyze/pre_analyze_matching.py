"""
現行の pre_experiment_matching.py に対応した解析スクリプト。

入力:
  results/tables/pre-experiment-matching/experiment/
    {participant_id}_YYYYMMDD_HHMMSS/contrast_matching.csv

出力:
  results/figures/pre-experiment-matching/experiment/
    analysis_YYYYMMDD_HHMMSS/

Defocus処理は実験と同じcommon.opticsを使用する。
"""

import argparse
import glob
import sys
import os
import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
PRE_EXPERIMENT_DIR = os.path.join(LAB_ROOT, "src", "experiment", "pre-experiment")
if PRE_EXPERIMENT_DIR not in sys.path:
    sys.path.insert(0, PRE_EXPERIMENT_DIR)

from common import geometry, optics, patterns


EXPERIMENT_RESULT_ROOT = os.path.join(
    LAB_ROOT, "results", "tables", "pre-experiment-matching", "experiment"
)
EXPERIMENT_FIGURE_ROOT = os.path.join(
    LAB_ROOT, "results", "figures", "pre-experiment-matching", "experiment"
)
TRAINING_FIGURE_ROOT = os.path.join(
    LAB_ROOT, "results", "figures", "pre-experiment-matching", "training"
)
COMBINED_FIGURE_ROOT = os.path.join(
    LAB_ROOT, "results", "figures", "pre-experiment-matching", "combined"
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
    """省略時は本実験、明示時は本実験・trainingの両方を受け付ける。"""
    if explicit_dirs:
        session_dirs = [os.path.abspath(path) for path in explicit_dirs]
    else:
        session_dirs = sorted(
            path for path in glob.glob(os.path.join(EXPERIMENT_RESULT_ROOT, "*"))
            if os.path.isdir(path)
        )

    valid_dirs = []
    for session_dir in session_dirs:
        csv_path = matching_result_path(session_dir)
        if csv_path is not None:
            valid_dirs.append(session_dir)
        else:
            print(
                "WARN: contrast_matching.csv / contrast_matching_training.csv "
                f"がないため除外: {session_dir}"
            )
    return valid_dirs


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
    random_state = np.random.get_state()
    np.random.seed(42)
    try:
        noise_base = patterns.create_noise_base(
            width_px, height_px, ppd_fg, f_center_cpd
        )
    finally:
        np.random.set_state(random_state)

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
    analyzed["Matched_Contrast_Enhanced"] = (
        analyzed["Matched_Contrast"] * analyzed["L_fg"]
        + analyzed["Effective_BG_Contrast"] * analyzed["L_bg"]
    ) / (analyzed["L_fg"] + analyzed["L_bg"])
    return analyzed


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
        ("AR_Matched_Contrast", "ar", "Matched Contrast (AR Extended)"),
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
                    ax.set_ylim(0.05, 1.2)
                    ax.set_yticks([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
                    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2g"))
                    ax.tick_params(axis="x", rotation=15)

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
        description="Analyze pre-experiment contrast matching results."
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

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        default_figure_root, f"analysis_{timestamp}"
    )
    save_outputs(analyzed, output_dir)


if __name__ == "__main__":
    main()