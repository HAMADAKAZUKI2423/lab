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

    # 実験のnoise生��と同じ関数を、解析ではseed固定で決定論的に使う。
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
    analyzed["Matched_Contrast_Enhanced"] = (
        analyzed["Matched_Contrast"] * analyzed["L_fg"]
        + analyzed["Effective_BG_Contrast"] * analyzed["L_bg"]
    ) / (analyzed["L_fg"] + analyzed["L_bg"])
    return analyzed


# Statistical analysis constants and helpers.
ALPHA = 0.05
ANALYSIS_SCALE = "log10_AR_Matched_Contrast"
PARTICIPANT_AGGREGATION = "median_of_log10_trials"
ANALYSIS_GROUP_COLUMNS = ("Session_Type", "Ref_Contrast", "Orientation")
OCULARITIES = ("monocular", "binocular")

SP_CONDITION = "Single plane"
SPD_CONDITION = "Single plane + defocus simulation"
DP_CONDITION = "Dual plane"
DPF_CONDITION = "Dual plane flat"
H1_CONDITIONS = (SPD_CONDITION, DP_CONDITION, DPF_CONDITION)

# H2/H3の条件差に対する等価幅。比で1/1.10〜1.10を暫定SESOIとする。
DEFAULT_EQUIVALENCE_RATIO_BOUND = 1.10
LOG10_EQUIVALENCE_MARGIN = np.log10(DEFAULT_EQUIVALENCE_RATIO_BOUND)

# H2-3の差の差に対する等価幅。比の比で1/1.10〜1.10を暫定値とする。
# H2/H3とは意味が異なるため独立した定数にし、最終解析前に別途固定する。
DEFAULT_INTERACTION_EQUIVALENCE_RATIO_BOUND = 1.10
LOG10_INTERACTION_EQUIVALENCE_MARGIN = np.log10(
    DEFAULT_INTERACTION_EQUIVALENCE_RATIO_BOUND
)


def _finite_values(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _pow10_or_nan(value):
    return float(10.0 ** value) if np.isfinite(value) else float("nan")


def _cohens_dz(diff_values):
    """対応差の平均を差の標本SDで割ったCohen's dz。"""
    values = _finite_values(diff_values)
    if len(values) < 2:
        return float("nan")
    mean_difference = float(np.mean(values))
    sd_difference = float(np.std(values, ddof=1))
    if np.isclose(sd_difference, 0.0):
        if np.isclose(mean_difference, 0.0):
            return 0.0
        return float(np.sign(mean_difference) * np.inf)
    return mean_difference / sd_difference


def _required_n_for_80pct_power(
    effect_size, alpha=ALPHA, target_power=0.80, max_n=10000
):
    """観測dzを仮定した両側1標本t検定の80%検出力に必要なn。"""
    from scipy import stats

    if np.isnan(effect_size) or effect_size == 0:
        return float("inf")
    if np.isinf(effect_size):
        return 2
    effect_size = abs(float(effect_size))
    for n in range(2, max_n + 1):
        degrees_of_freedom = n - 1
        critical_t = stats.t.ppf(1.0 - alpha / 2.0, degrees_of_freedom)
        noncentrality = effect_size * np.sqrt(n)
        power = (
            stats.nct.cdf(-critical_t, degrees_of_freedom, noncentrality)
            + stats.nct.sf(critical_t, degrees_of_freedom, noncentrality)
        )
        if power >= target_power:
            return n
    return float("inf")


def _holm_adjust(p_values):
    """Holm補正。NaNも事前に定めた検定数へ数えるが、出力はNaNのまま。"""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan)
    valid_indices = [
        index for index, p_value in enumerate(p_values) if np.isfinite(p_value)
    ]
    ordered_indices = sorted(valid_indices, key=lambda index: p_values[index])
    family_size = len(p_values)
    running_maximum = 0.0
    for rank, index in enumerate(ordered_indices):
        running_maximum = max(
            running_maximum,
            (family_size - rank) * float(p_values[index]),
        )
        adjusted[index] = min(running_maximum, 1.0)
    return adjusted


def _mean_difference_statistics(diff_values):
    """log10差分系列の両側1標本t検定・95% CI・効果量を返す。"""
    from scipy import stats

    values = _finite_values(diff_values)
    n = len(values)
    mean_difference = float(np.mean(values)) if n else float("nan")
    result = {
        "n": n,
        "mean_log10_difference": mean_difference,
        "sd_log10_difference": float("nan"),
        "sem_log10_difference": float("nan"),
        "ci95_log10_lower": float("nan"),
        "ci95_log10_upper": float("nan"),
        "t_statistic": float("nan"),
        "degrees_of_freedom": n - 1 if n else float("nan"),
        "p_value_two_sided": float("nan"),
        "cohens_dz": float("nan"),
        "required_n_for_80pct_power": float("inf"),
    }
    if n < 2:
        return result

    sd_difference = float(np.std(values, ddof=1))
    sem_difference = sd_difference / np.sqrt(n)
    result["sd_log10_difference"] = sd_difference
    result["sem_log10_difference"] = sem_difference

    if np.isclose(sd_difference, 0.0):
        if np.isclose(mean_difference, 0.0):
            t_statistic, p_value = 0.0, 1.0
        else:
            t_statistic = float(np.sign(mean_difference) * np.inf)
            p_value = 0.0
        ci_lower = ci_upper = mean_difference
    else:
        t_statistic = mean_difference / sem_difference
        p_value = float(2.0 * stats.t.sf(abs(t_statistic), df=n - 1))
        ci_lower, ci_upper = stats.t.interval(
            0.95,
            n - 1,
            loc=mean_difference,
            scale=sem_difference,
        )

    effect_size = _cohens_dz(values)
    result.update(
        {
            "ci95_log10_lower": float(ci_lower),
            "ci95_log10_upper": float(ci_upper),
            "t_statistic": float(t_statistic),
            "p_value_two_sided": p_value,
            "cohens_dz": effect_size,
            "required_n_for_80pct_power": _required_n_for_80pct_power(
                effect_size
            ),
        }
    )
    return result


def _empty_tost_statistics():
    return {
        "equivalence_margin_log10": float("nan"),
        "tost_p_lower": float("nan"),
        "tost_p_upper": float("nan"),
        "tost_p_value": float("nan"),
        "tost_ci90_log10_lower": float("nan"),
        "tost_ci90_log10_upper": float("nan"),
    }


def _tost_equivalence_statistics(diff_values, margin, alpha=ALPHA):
    """log10平均差が[-margin,+margin]内かをTOSTで検定する。"""
    from scipy import stats

    values = _finite_values(diff_values)
    result = _empty_tost_statistics()
    result["equivalence_margin_log10"] = float(margin)
    n = len(values)
    if n < 2 or not np.isfinite(margin) or margin <= 0:
        return result

    mean_difference = float(np.mean(values))
    sd_difference = float(np.std(values, ddof=1))
    sem_difference = sd_difference / np.sqrt(n)

    if np.isclose(sem_difference, 0.0):
        def zero_se_t(numerator):
            if np.isclose(numerator, 0.0):
                return 0.0
            return float(np.sign(numerator) * np.inf)

        t_lower = zero_se_t(mean_difference + margin)
        t_upper = zero_se_t(mean_difference - margin)
        ci_lower = ci_upper = mean_difference
    else:
        t_lower = (mean_difference + margin) / sem_difference
        t_upper = (mean_difference - margin) / sem_difference
        ci_lower, ci_upper = stats.t.interval(
            1.0 - 2.0 * alpha,
            n - 1,
            loc=mean_difference,
            scale=sem_difference,
        )

    p_lower = float(stats.t.sf(t_lower, df=n - 1))
    p_upper = float(stats.t.cdf(t_upper, df=n - 1))
    result.update(
        {
            "tost_p_lower": p_lower,
            "tost_p_upper": p_upper,
            "tost_p_value": max(p_lower, p_upper),
            "tost_ci90_log10_lower": float(ci_lower),
            "tost_ci90_log10_upper": float(ci_upper),
        }
    )
    return result


def _paired_condition_frame(
    group_df, baseline_condition, condition, ocularity, value_column
):
    """同一IDの2条件を1対1で対応付ける。"""
    baseline = group_df.loc[
        (group_df["Condition"] == baseline_condition)
        & (group_df["Ocularity"] == ocularity),
        ["ID", value_column],
    ].rename(columns={value_column: "baseline_value"})
    comparison = group_df.loc[
        (group_df["Condition"] == condition)
        & (group_df["Ocularity"] == ocularity),
        ["ID", value_column],
    ].rename(columns={value_column: "condition_value"})
    return pd.merge(
        baseline,
        comparison,
        on="ID",
        how="inner",
        validate="one_to_one",
    ).dropna()


def _build_test_row(
    *,
    group_metadata,
    hypothesis,
    family,
    claim_component,
    comparison,
    test_type,
    condition,
    baseline_condition,
    ocularity,
    condition_values,
    baseline_values,
    diff_values,
    effect_scale="log10_condition_ratio",
    equivalence_margin_log10=None,
    primary_test="",
):
    """1比較分のt検定・TOST・比率換算を同一形式の行へまとめる。"""
    condition_values = _finite_values(condition_values)
    baseline_values = _finite_values(baseline_values)
    diff_values = _finite_values(diff_values)
    if not (
        len(condition_values) == len(baseline_values) == len(diff_values)
    ):
        raise ValueError("condition/baseline/difference lengths must match")

    summary = _mean_difference_statistics(diff_values)
    run_tost = equivalence_margin_log10 is not None
    tost = (
        _tost_equivalence_statistics(diff_values, equivalence_margin_log10)
        if run_tost
        else _empty_tost_statistics()
    )
    ratio_bound = (
        float(10.0 ** equivalence_margin_log10)
        if run_tost
        else float("nan")
    )
    row = {
        "Hypothesis": hypothesis,
        "Family": family,
        "Claim_Component": claim_component,
        **group_metadata,
        "Ocularity": ocularity,
        "Comparison": comparison,
        "Test_Type": test_type,
        "Primary_Test": primary_test,
        "Condition": condition,
        "Baseline_Condition": baseline_condition,
        "Effect_Scale": effect_scale,
        "Analysis_Scale": ANALYSIS_SCALE,
        "Participant_Aggregation": PARTICIPANT_AGGREGATION,
        "n": summary["n"],
        "mean_log10_condition": (
            float(np.mean(condition_values))
            if len(condition_values)
            else float("nan")
        ),
        "mean_log10_baseline": (
            float(np.mean(baseline_values))
            if len(baseline_values)
            else float("nan")
        ),
        "geometric_mean_condition": (
            _pow10_or_nan(float(np.mean(condition_values)))
            if len(condition_values)
            else float("nan")
        ),
        "geometric_mean_baseline": (
            _pow10_or_nan(float(np.mean(baseline_values)))
            if len(baseline_values)
            else float("nan")
        ),
        **summary,
        "geometric_mean_ratio": _pow10_or_nan(
            summary["mean_log10_difference"]
        ),
        "ci95_ratio_lower": _pow10_or_nan(summary["ci95_log10_lower"]),
        "ci95_ratio_upper": _pow10_or_nan(summary["ci95_log10_upper"]),
        "holm_adjusted_p_value": float("nan"),
        "significant_holm_alpha_0_05": False,
        "equivalence_ratio_lower": (
            1.0 / ratio_bound if run_tost else float("nan")
        ),
        "equivalence_ratio_upper": ratio_bound,
        **tost,
        "tost_ci90_ratio_lower": _pow10_or_nan(
            tost["tost_ci90_log10_lower"]
        ),
        "tost_ci90_ratio_upper": _pow10_or_nan(
            tost["tost_ci90_log10_upper"]
        ),
        "holm_adjusted_tost_p_value": float("nan"),
        "equivalent_holm_alpha_0_05": False,
        "Primary_P_Value": float("nan"),
        "Primary_Reject_Alpha_0_05": False,
        "Decision_Basis": "",
        "H2_Conjunction_Evaluable": False,
        "H2_Conjunction_All_Pass": False,
    }
    return row


def _apply_holm_to_rows(rows, p_key, adjusted_key, decision_key):
    adjusted_values = _holm_adjust([row[p_key] for row in rows])
    for row, adjusted_p in zip(rows, adjusted_values):
        row[adjusted_key] = adjusted_p
        row[decision_key] = bool(
            np.isfinite(adjusted_p) and adjusted_p < ALPHA
        )


def _raw_t_tost_interpretation(row):
    t_p = row["p_value_two_sided"]
    tost_p = row["tost_p_value"]
    if not np.isfinite(tost_p):
        return ""
    t_significant = np.isfinite(t_p) and t_p < ALPHA
    tost_significant = tost_p < ALPHA
    if t_significant and tost_significant:
        return "different_but_within_equivalence_bounds"
    if t_significant:
        return "different_not_equivalent"
    if tost_significant:
        return "equivalent_no_detected_difference"
    return "inconclusive"


def _set_primary_decisions(group_test_rows):
    """Holm判定とH2連言（IUT）の主判定を明示的な列へ格納する。"""
    for row in group_test_rows:
        hypothesis = row["Hypothesis"]
        primary_test = row["Primary_Test"]
        if hypothesis == "H1":
            primary_p = row["holm_adjusted_p_value"]
            basis = "Holm-adjusted two-sided t test"
        elif hypothesis == "H3":
            primary_p = row["holm_adjusted_tost_p_value"]
            basis = "Holm-adjusted TOST"
        elif hypothesis in {"H2", "H2_interaction"}:
            primary_p = (
                row["tost_p_value"]
                if primary_test == "TOST"
                else row["p_value_two_sided"]
            )
            basis = f"Unadjusted {primary_test} as IUT component"
        else:
            primary_p = float("nan")
            basis = ""
        row["Primary_P_Value"] = primary_p
        row["Primary_Reject_Alpha_0_05"] = bool(
            np.isfinite(primary_p) and primary_p < ALPHA
        )
        row["Decision_Basis"] = basis
        row["Raw_T_and_TOST_Interpretation"] = _raw_t_tost_interpretation(
            row
        )

    h2_rows = [
        row
        for row in group_test_rows
        if row["Hypothesis"] in {"H2", "H2_interaction"}
    ]
    expected_components = {"H2-1", "H2-2", "H2-3"}
    present_components = {row["Claim_Component"] for row in h2_rows}
    evaluable = (
        present_components == expected_components
        and all(np.isfinite(row["Primary_P_Value"]) for row in h2_rows)
    )
    all_pass = evaluable and all(
        row["Primary_Reject_Alpha_0_05"] for row in h2_rows
    )
    for row in h2_rows:
        row["H2_Conjunction_Evaluable"] = evaluable
        row["H2_Conjunction_All_Pass"] = all_pass


def _prepare_participant_medians(df):
    source_column = "AR_Matched_Contrast"
    value_column = "Log10_AR_Matched_Contrast"
    keys = [
        "ID",
        *ANALYSIS_GROUP_COLUMNS,
        "Condition",
        "Ocularity",
    ]
    statistical_data = df[keys + [source_column]].copy()
    statistical_data[source_column] = pd.to_numeric(
        statistical_data[source_column], errors="coerce"
    )
    statistical_data = statistical_data.dropna(subset=keys)
    invalid_mask = (
        ~np.isfinite(statistical_data[source_column])
        | (statistical_data[source_column] <= 0)
    )
    if invalid_mask.any():
        raise ValueError(
            "log10統計解析にはAR_Matched_Contrast > 0が必要です。"
            f"無効な試行数: {int(invalid_mask.sum())}"
        )
    statistical_data[value_column] = np.log10(
        statistical_data[source_column].to_numpy(dtype=float)
    )
    participant_medians = (
        statistical_data.groupby(keys, as_index=False)[value_column].median()
    )
    return participant_medians, value_column


def save_statistical_csv_outputs(df, output_dir):
    """解析計画H1〜H3を実行し、検定表とH3相関表を保存する。

    H1: SP vs SPD/DP/DPF（眼別の対応t、6検定をHolm補正）
    H2-1: monocularのSPD vs DP等価性（主判定TOST、tも併記）
    H2-2: binocularのSPD vs DP差（主判定t、TOSTも併記）
    H2-3: SPD→DP効果のbinocular vs monocular差（対応t、TOSTも併記）
           3主張の連言は各主判定の生p値によるIUTで評価する。
           単体報告用として3行のt/TOSTには別途Holm補正値も出力する。
    H3: DPF vs Ref（眼別の1標本t・TOST、2検定をHolm補正）
        およびSP/DPFの参照オフセット相関（2眼をHolm補正）

    各試行をlog10変換後、参加者×条件内の中央値を代表値とする。
    """
    from scipy import stats

    os.makedirs(output_dir, exist_ok=True)
    participant_medians, value_column = _prepare_participant_medians(df)
    test_rows = []
    correlation_rows = []

    for group_values, group_df in participant_medians.groupby(
        list(ANALYSIS_GROUP_COLUMNS), sort=True
    ):
        group_metadata = dict(zip(ANALYSIS_GROUP_COLUMNS, group_values))
        reference = float(group_metadata["Ref_Contrast"])
        if reference <= 0:
            raise ValueError(
                f"Ref_Contrast must be positive for log10 analysis: {reference}"
            )
        reference_log10 = np.log10(reference)
        group_test_rows = []
        group_correlation_rows = []
        h2_effects = {}

        for ocularity in OCULARITIES:
            # H1: SPを基準とする3つの条件差。
            for condition in H1_CONDITIONS:
                paired = _paired_condition_frame(
                    group_df,
                    SP_CONDITION,
                    condition,
                    ocularity,
                    value_column,
                )
                baseline_values = paired["baseline_value"].to_numpy(dtype=float)
                condition_values = paired["condition_value"].to_numpy(dtype=float)
                group_test_rows.append(
                    _build_test_row(
                        group_metadata=group_metadata,
                        hypothesis="H1",
                        family="H1_SP_vs_others",
                        claim_component="",
                        comparison=f"{condition} vs {SP_CONDITION}",
                        test_type="paired_t",
                        condition=condition,
                        baseline_condition=SP_CONDITION,
                        ocularity=ocularity,
                        condition_values=condition_values,
                        baseline_values=baseline_values,
                        diff_values=condition_values - baseline_values,
                        primary_test="t",
                    )
                )

            # H2-1/H2-2: 眼別のSPD→DP差。両方でtとTOSTを実行する。
            paired_h2 = _paired_condition_frame(
                group_df,
                SPD_CONDITION,
                DP_CONDITION,
                ocularity,
                value_column,
            )
            spd_values = paired_h2["baseline_value"].to_numpy(dtype=float)
            dp_values = paired_h2["condition_value"].to_numpy(dtype=float)
            h2_effects[ocularity] = pd.Series(
                dp_values - spd_values,
                index=paired_h2["ID"],
                name=ocularity,
                dtype=float,
            )
            is_monocular = ocularity == "monocular"
            group_test_rows.append(
                _build_test_row(
                    group_metadata=group_metadata,
                    hypothesis="H2",
                    family="H2_SPD_DP_ocularity",
                    claim_component="H2-1" if is_monocular else "H2-2",
                    comparison=f"{DP_CONDITION} vs {SPD_CONDITION}",
                    test_type="paired_t_and_tost",
                    condition=DP_CONDITION,
                    baseline_condition=SPD_CONDITION,
                    ocularity=ocularity,
                    condition_values=dp_values,
                    baseline_values=spd_values,
                    diff_values=dp_values - spd_values,
                    equivalence_margin_log10=LOG10_EQUIVALENCE_MARGIN,
                    primary_test="TOST" if is_monocular else "t",
                )
            )

            # H3: DPFと物理的参照コントラストの一致。
            dpf_values = group_df.loc[
                (group_df["Condition"] == DPF_CONDITION)
                & (group_df["Ocularity"] == ocularity),
                value_column,
            ].to_numpy(dtype=float)
            reference_values = np.full(len(dpf_values), reference_log10)
            group_test_rows.append(
                _build_test_row(
                    group_metadata=group_metadata,
                    hypothesis="H3",
                    family="H3_DPF_vs_reference",
                    claim_component="",
                    comparison=f"{DPF_CONDITION} vs Reference contrast",
                    test_type="one_sample_t_and_tost",
                    condition=DPF_CONDITION,
                    baseline_condition="Reference contrast",
                    ocularity=ocularity,
                    condition_values=dpf_values,
                    baseline_values=reference_values,
                    diff_values=dpf_values - reference_log10,
                    equivalence_margin_log10=LOG10_EQUIVALENCE_MARGIN,
                    primary_test="TOST",
                )
            )

            # H3補助: SPとDPFの参照オフセットの参加者間相関。
            paired_sp_dpf = _paired_condition_frame(
                group_df,
                SP_CONDITION,
                DPF_CONDITION,
                ocularity,
                value_column,
            )
            sp_offset = (
                paired_sp_dpf["baseline_value"].to_numpy(dtype=float)
                - reference_log10
            )
            dpf_offset = (
                paired_sp_dpf["condition_value"].to_numpy(dtype=float)
                - reference_log10
            )
            n_correlation = len(paired_sp_dpf)
            if (
                n_correlation >= 3
                and not np.isclose(np.std(sp_offset, ddof=1), 0.0)
                and not np.isclose(np.std(dpf_offset, ddof=1), 0.0)
            ):
                pearson_r, pearson_p = stats.pearsonr(sp_offset, dpf_offset)
                pearson_r, pearson_p = float(pearson_r), float(pearson_p)
            else:
                pearson_r = pearson_p = float("nan")
            group_correlation_rows.append(
                {
                    "Hypothesis": "H3",
                    **group_metadata,
                    "Ocularity": ocularity,
                    "n_participants": n_correlation,
                    "Analysis_Scale": ANALYSIS_SCALE,
                    "Participant_Aggregation": PARTICIPANT_AGGREGATION,
                    "mean_sp_log10_offset_from_reference": (
                        float(np.mean(sp_offset))
                        if n_correlation
                        else float("nan")
                    ),
                    "mean_dpf_log10_offset_from_reference": (
                        float(np.mean(dpf_offset))
                        if n_correlation
                        else float("nan")
                    ),
                    "geometric_mean_sp_ratio_to_reference": (
                        _pow10_or_nan(float(np.mean(sp_offset)))
                        if n_correlation
                        else float("nan")
                    ),
                    "geometric_mean_dpf_ratio_to_reference": (
                        _pow10_or_nan(float(np.mean(dpf_offset)))
                        if n_correlation
                        else float("nan")
                    ),
                    "pearson_r": pearson_r,
                    "p_value_two_sided": pearson_p,
                    "holm_adjusted_p_value": float("nan"),
                    "significant_holm_alpha_0_05": False,
                }
            )

        # H2-3: 各参加者のlog10(DP/SPD)を眼間で直接比較する。
        interaction_frame = pd.concat(
            [h2_effects["binocular"], h2_effects["monocular"]],
            axis=1,
            join="inner",
        ).dropna()
        binocular_effect = interaction_frame["binocular"].to_numpy(dtype=float)
        monocular_effect = interaction_frame["monocular"].to_numpy(dtype=float)
        group_test_rows.append(
            _build_test_row(
                group_metadata=group_metadata,
                hypothesis="H2_interaction",
                family="H2_SPD_DP_ocularity",
                claim_component="H2-3",
                comparison="(DP/SPD)_binocular vs (DP/SPD)_monocular",
                test_type="paired_t_and_tost",
                condition="binocular log10(DP/SPD)",
                baseline_condition="monocular log10(DP/SPD)",
                ocularity="binocular_minus_monocular",
                condition_values=binocular_effect,
                baseline_values=monocular_effect,
                diff_values=binocular_effect - monocular_effect,
                effect_scale="log10_ratio_of_ratios",
                equivalence_margin_log10=(
                    LOG10_INTERACTION_EQUIVALENCE_MARGIN
                ),
                primary_test="t",
            )
        )

        # 事前に定めたファミリーごとのHolm補正（単体報告用）。
        h1_rows = [row for row in group_test_rows if row["Hypothesis"] == "H1"]
        h2_rows = [
            row
            for row in group_test_rows
            if row["Hypothesis"] in {"H2", "H2_interaction"}
        ]
        h3_rows = [row for row in group_test_rows if row["Hypothesis"] == "H3"]
        _apply_holm_to_rows(
            h1_rows,
            "p_value_two_sided",
            "holm_adjusted_p_value",
            "significant_holm_alpha_0_05",
        )
        for family_rows in (h2_rows, h3_rows):
            _apply_holm_to_rows(
                family_rows,
                "p_value_two_sided",
                "holm_adjusted_p_value",
                "significant_holm_alpha_0_05",
            )
            _apply_holm_to_rows(
                family_rows,
                "tost_p_value",
                "holm_adjusted_tost_p_value",
                "equivalent_holm_alpha_0_05",
            )

        _apply_holm_to_rows(
            group_correlation_rows,
            "p_value_two_sided",
            "holm_adjusted_p_value",
            "significant_holm_alpha_0_05",
        )
        _set_primary_decisions(group_test_rows)
        test_rows.extend(group_test_rows)
        correlation_rows.extend(group_correlation_rows)

    legacy_statistical_files = (
        "one_sample_ttests_holm_cohens_d_power.csv",
        "paired_ttests_vs_single_plane_holm_cohens_d.csv",
        "participant_bias_corrected_values.csv",
        "bias_corrected_summary.csv",
    )
    for filename in legacy_statistical_files:
        legacy_path = os.path.join(output_dir, filename)
        if os.path.isfile(legacy_path):
            os.remove(legacy_path)

    test_frame = pd.DataFrame(test_rows)
    correlation_frame = pd.DataFrame(correlation_rows)
    test_frame.to_csv(
        os.path.join(output_dir, "planned_hypothesis_tests.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    correlation_frame.to_csv(
        os.path.join(output_dir, "h3_dpf_sp_offset_correlation.csv"),
        index=False,
        encoding="utf-8-sig",
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
    output_dir = args.output_dir or os.path.join(
        default_figure_root,
        "all_participants"
        if all_participants_mode
        else f"analyze_{participant_label}_{timestamp}",
    )
    save_outputs(analyzed, output_dir)
    save_statistical_csv_outputs(analyzed, output_dir)


if __name__ == "__main__":
    main()