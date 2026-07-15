"""
現行の pre-experiment-gabor-matching.py に対応した解析スクリプト。

入力:
  results/tables/pre-experiment-matching/experiment/
    {participant_id}_YYYYMMDD_HHMMSS/contrast_matching.csv

出力:
  results/figures/pre-experiment-matching/experiment/
    analysis_YYYYMMDD_HHMMSS/

Defocus処理は実験と同じ stimuli_utils.apply_torch_fft_blur_luminance() を使用する。
"""

import argparse
import glob
import importlib.util
import json
import os
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
PRE_EXPERIMENT_DIR = os.path.join(LAB_ROOT, "src", "experiment", "pre-experiment")
STIMULI_UTILS_PATH = os.path.join(PRE_EXPERIMENT_DIR, "stimuli_utils.py")

# pre-analyze と pre-experiment は兄弟ディレクトリのため直接importしない。
# ファイルパスから明示的に stimuli_utils.py をロードする。
spec = importlib.util.spec_from_file_location("pre_experiment_stimuli_utils", STIMULI_UTILS_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"stimuli_utils.py をロードできません: {STIMULI_UTILS_PATH}")
stimuli_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stimuli_utils)


RESULT_ROOT = os.path.join(
    LAB_ROOT, "results", "tables", "pre-experiment-matching", "experiment"
)
FIGURE_ROOT = os.path.join(
    LAB_ROOT, "results", "figures", "pre-experiment-matching", "experiment"
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
]

_blur_attenuation_cache = {}


def discover_session_dirs(explicit_dirs=None):
    """実験側の参加者・日時フォルダ構造に従って解析対象を探索する。"""
    if explicit_dirs:
        session_dirs = [os.path.abspath(path) for path in explicit_dirs]
    else:
        session_dirs = sorted(
            path for path in glob.glob(os.path.join(RESULT_ROOT, "*"))
            if os.path.isdir(path)
        )

    valid_dirs = []
    for session_dir in session_dirs:
        csv_path = os.path.join(session_dir, "contrast_matching.csv")
        if os.path.isfile(csv_path):
            valid_dirs.append(session_dir)
        else:
            print(f"WARN: contrast_matching.csv がないため除外: {session_dir}")
    return valid_dirs


def load_matching_results(session_dirs):
    frames = []
    for session_dir in session_dirs:
        csv_path = os.path.join(session_dir, "contrast_matching.csv")
        df = pd.read_csv(csv_path, encoding="utf-8")
        missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"{csv_path}: 必須列が不足しています: {missing}")
        df["Session_Dir"] = session_dir
        frames.append(df)
        print(f"Loaded: {csv_path}")

    if not frames:
        raise FileNotFoundError(
            f"解析可能な contrast_matching.csv がありません: {RESULT_ROOT}"
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

    ppd_fg = stimuli_utils.get_size_for_visual_angle(d_fg, 1.0)
    width_base = int(VISUAL_ANGLE_WIDTH_DEG * ppd_fg)
    width_px = int(width_base * WIN2_TOTAL_WIDTH_FACTOR)
    height_px = int(VISUAL_ANGLE_HEIGHT_DEG * ppd_fg)

    # 実験のnoise生成と同じ関数を、解析ではseed固定で決定論的に使う。
    random_state = np.random.get_state()
    np.random.seed(42)
    try:
        noise_base = stimuli_utils.create_noise_base(
            width_px, height_px, ppd_fg, f_center_cpd
        )
    finally:
        np.random.set_state(random_state)

    lum_original = DEFAULT_L_BG * (1.0 + BACKGROUND_CONTRAST * noise_base)
    d_fg_m = d_fg / 100.0
    d_bg_m = d_bg / 100.0
    diopter_difference = abs(1.0 / d_fg_m - 1.0 / d_bg_m)

    lum_blurred = stimuli_utils.apply_torch_fft_blur_luminance(
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
    analyzed["Matching_Ratio"] = (
        analyzed["Matched_Contrast"] / analyzed["Ref_Contrast"]
    )
    return analyzed


def save_outputs(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    trial_path = os.path.join(output_dir, "matching_trials_analyzed.csv")
    df.to_csv(trial_path, index=False, encoding="utf-8-sig")

    group_columns = ["Condition", "Ocularity", "Ref_Contrast"]
    summary = (
        df.groupby(group_columns, dropna=False)
        .agg(
            n=("Matched_Contrast", "size"),
            matched_mean=("Matched_Contrast", "mean"),
            matched_std=("Matched_Contrast", "std"),
            ar_matched_mean=("AR_Matched_Contrast", "mean"),
            ar_matched_std=("AR_Matched_Contrast", "std"),
            blur_attenuation_mean=("Blur_Attenuation", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(
        os.path.join(output_dir, "matching_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    sns.set_theme(style="whitegrid")
    for value_column, filename, ylabel in [
        ("Matched_Contrast", "matched_contrast.png", "Matched contrast"),
        ("AR_Matched_Contrast", "ar_matched_contrast.png", "AR matched contrast"),
        ("Matching_Ratio", "matching_ratio.png", "Matched / reference contrast"),
    ]:
        plt.figure(figsize=(12, 6))
        sns.pointplot(
            data=df,
            x="Condition",
            y=value_column,
            hue="Ocularity",
            errorbar="sd",
            dodge=True,
            markers=["o", "s"],
        )
        plt.ylabel(ylabel)
        plt.xlabel("Condition")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=200)
        plt.close()

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
        raise FileNotFoundError(f"解析対象がありません: {RESULT_ROOT}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        FIGURE_ROOT, f"analysis_{timestamp}"
    )

    df = load_matching_results(session_dirs)
    analyzed = add_analysis_columns(df)
    save_outputs(analyzed, output_dir)

    with open(
        os.path.join(output_dir, "source_sessions.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(session_dirs, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()