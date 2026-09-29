"""DP優位眼・非優位眼条件の同一画像ペアに対する評定差を解析する。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
LAB_ROOT = SCRIPT_DIR.parents[2]
DATA_BASE_DIR = LAB_ROOT / "results" / "tables" / "pre-experiment-image"
FIGURE_BASE_DIR = LAB_ROOT / "results" / "figures" / "pre-experiment-image"

DOM_CONDITION = "DP_MONO_DOM"
NONDOM_CONDITION = "DP_MONO_NONDOM"
GROUP_ORDER = ("overall", "high", "low")
REQUIRED_COLUMNS = {
    "ID",
    "Condition",
    "Pair_ID",
    "Difference_Group",
    "Group_Rank",
    "Foreground_Texture_Class",
    "Score",
}


def resolve_target_dir(argument: str | None) -> Path:
    """明示されたセッション、または最新の結果セッションを返す。"""
    if argument:
        target = Path(argument).expanduser().resolve()
        if not target.is_dir():
            raise FileNotFoundError(f"解析対象ディレクトリがありません: {target}")
        return target
    if not DATA_BASE_DIR.is_dir():
        raise FileNotFoundError(f"結果ルートがありません: {DATA_BASE_DIR}")
    candidates = [path for path in DATA_BASE_DIR.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"解析対象セッションがありません: {DATA_BASE_DIR}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_results(target_dir: Path) -> pd.DataFrame:
    """image_evaluation.csvを読み、必要列と評定範囲を検証する。"""
    csv_path = target_dir / "image_evaluation.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"画像評価結果がありません: {csv_path}")
    frame = pd.read_csv(csv_path, encoding="utf-8")
    if frame.empty:
        raise ValueError(f"画像評価結果が空です: {csv_path}")
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{csv_path}: 必須列が不足しています: {missing}")
    frame["Score"] = pd.to_numeric(frame["Score"], errors="coerce")
    if frame["Score"].isna().any():
        raise ValueError("Scoreに数値へ変換できない値があります")
    if not frame["Score"].between(1, 5).all():
        invalid = sorted(frame.loc[~frame["Score"].between(1, 5), "Score"].unique())
        raise ValueError(f"Scoreは1〜5である必要があります: {invalid}")
    return frame


def build_paired_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """参加者・Pair_IDごとにDP優位眼と非優位眼の評定を対応付ける。"""
    filtered = frame[frame["Condition"].isin((DOM_CONDITION, NONDOM_CONDITION))].copy()
    if filtered.empty:
        raise ValueError("DP_MONO_DOM / DP_MONO_NONDOMの行がありません")

    duplicates = filtered.duplicated(
        subset=["ID", "Condition", "Pair_ID"], keep=False
    )
    if duplicates.any():
        duplicate_keys = (
            filtered.loc[duplicates, ["ID", "Condition", "Pair_ID"]]
            .drop_duplicates()
            .to_dict("records")
        )
        raise ValueError(
            "同一参加者・条件・Pair_IDに重複行があります: "
            f"{duplicate_keys}"
        )

    metadata_columns = [
        "ID",
        "Pair_ID",
        "Difference_Group",
        "Group_Rank",
        "Foreground_Texture_Class",
    ]
    metadata = filtered[metadata_columns].drop_duplicates()
    inconsistent = metadata.duplicated(subset=["ID", "Pair_ID"], keep=False)
    if inconsistent.any():
        keys = (
            metadata.loc[inconsistent, ["ID", "Pair_ID"]]
            .drop_duplicates()
            .to_dict("records")
        )
        raise ValueError(f"条件間で画像ペア属性が一致しません: {keys}")

    scores = filtered.pivot(
        index=["ID", "Pair_ID"], columns="Condition", values="Score"
    )
    missing_conditions = [
        condition for condition in (DOM_CONDITION, NONDOM_CONDITION)
        if condition not in scores.columns
    ]
    if missing_conditions:
        raise ValueError(f"比較条件が不足しています: {missing_conditions}")
    incomplete = scores[[DOM_CONDITION, NONDOM_CONDITION]].isna().any(axis=1)
    if incomplete.any():
        raise ValueError(
            "優位眼・非優位眼の両条件がそろわない画像ペアがあります: "
            f"{list(scores.index[incomplete])}"
        )

    paired = (
        scores[[DOM_CONDITION, NONDOM_CONDITION]]
        .rename(columns={
            DOM_CONDITION: "Score_DOM",
            NONDOM_CONDITION: "Score_NONDOM",
        })
        .reset_index()
        .merge(metadata, on=["ID", "Pair_ID"], how="left", validate="one_to_one")
    )
    paired["Difference_Group"] = paired["Difference_Group"].astype(str).str.lower()
    unexpected_groups = sorted(set(paired["Difference_Group"]) - {"high", "low"})
    if unexpected_groups:
        raise ValueError(f"Difference_Groupにhigh/low以外があります: {unexpected_groups}")

    for participant_id, participant_df in paired.groupby("ID", sort=False):
        if len(participant_df) != 20:
            raise ValueError(
                f"ID={participant_id}: 対応する画像ペアは20組必要です: {len(participant_df)}"
            )
        group_counts = participant_df["Difference_Group"].value_counts().to_dict()
        if group_counts.get("high", 0) != 10 or group_counts.get("low", 0) != 10:
            raise ValueError(
                f"ID={participant_id}: high 10組・low 10組が必要です: {group_counts}"
            )

    paired["Signed_Difference"] = paired["Score_DOM"] - paired["Score_NONDOM"]
    paired["Absolute_Difference"] = paired["Signed_Difference"].abs()
    paired["Squared_Difference"] = paired["Signed_Difference"] ** 2
    return paired.sort_values(
        ["ID", "Difference_Group", "Group_Rank"],
        key=lambda values: values.map({"high": 0, "low": 1}).fillna(values)
        if values.name == "Difference_Group" else values,
    ).reset_index(drop=True)


def summarize_rmsd(paired: pd.DataFrame) -> pd.DataFrame:
    """参加者ごとにoverall/high/lowのRMSDと補助指標を計算する。"""
    rows: list[dict[str, object]] = []
    for participant_id, participant_df in paired.groupby("ID", sort=False):
        subsets = {
            "overall": participant_df,
            "high": participant_df[participant_df["Difference_Group"] == "high"],
            "low": participant_df[participant_df["Difference_Group"] == "low"],
        }
        for group in GROUP_ORDER:
            subset = subsets[group]
            difference = subset["Signed_Difference"].to_numpy(dtype=float)
            rows.append({
                "ID": participant_id,
                "Group": group,
                "N_Pairs": len(subset),
                "RMSD": float(np.sqrt(np.mean(difference ** 2))),
                "Mean_Signed_Difference": float(np.mean(difference)),
                "Mean_Absolute_Difference": float(np.mean(np.abs(difference))),
                "Agreement_Rate": float(np.mean(difference == 0)),
            })
    summary = pd.DataFrame(rows)
    summary["Group"] = pd.Categorical(
        summary["Group"], categories=GROUP_ORDER, ordered=True
    )
    return summary.sort_values(["ID", "Group"]).reset_index(drop=True)


def _safe_id(value: object) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(value)
    )


def plot_rmsd(
    participant_summary: pd.DataFrame,
    output: Path,
    participant_id: object,
) -> None:
    """overall/high/lowのRMSD棒グラフを保存する。"""
    values = participant_summary.set_index("Group").reindex(GROUP_ORDER)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(
        np.arange(len(GROUP_ORDER)),
        values["RMSD"].to_numpy(dtype=float),
        color=["#7f7f7f", "#d62728", "#1f77b4"],
    )
    ax.set_xticks(np.arange(len(GROUP_ORDER)))
    ax.set_xticklabels(["Overall", "High", "Low"])
    ax.set_ylim(0, 4)
    ax.set_ylabel("RMS difference in visibility score")
    ax.set_title(f"DP dominant vs non-dominant eye RMSD\nID={participant_id}")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, value in zip(bars, values["RMSD"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            float(value) + 0.06,
            f"{float(value):.3f}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_score_scatter(
    participant_df: pd.DataFrame,
    participant_summary: pd.DataFrame,
    output: Path,
    participant_id: object,
) -> None:
    """優位眼・非優位眼評定の対応散布図を保存する。"""
    rng = np.random.default_rng(20260929)
    jitter_x = rng.uniform(-0.06, 0.06, len(participant_df))
    jitter_y = rng.uniform(-0.06, 0.06, len(participant_df))
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for group, color in (("high", "#d62728"), ("low", "#1f77b4")):
        mask = participant_df["Difference_Group"] == group
        ax.scatter(
            participant_df.loc[mask, "Score_DOM"] + jitter_x[mask.to_numpy()],
            participant_df.loc[mask, "Score_NONDOM"] + jitter_y[mask.to_numpy()],
            s=70,
            alpha=0.8,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            label=group,
        )
    ax.plot([0.8, 5.2], [0.8, 5.2], "--", color="black", linewidth=1.2)
    ax.set_xlim(0.8, 5.2)
    ax.set_ylim(0.8, 5.2)
    ax.set_xticks(range(1, 6))
    ax.set_yticks(range(1, 6))
    ax.set_xlabel("DP dominant-eye score")
    ax.set_ylabel("DP non-dominant-eye score")
    ax.set_title(f"Matched image-pair scores\nID={participant_id}")
    overall = participant_summary[
        participant_summary["Group"] == "overall"
    ].iloc[0]
    ax.text(
        0.04,
        0.96,
        f"RMSD = {overall['RMSD']:.3f}\n"
        f"Mean signed difference = {overall['Mean_Signed_Difference']:+.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    ax.legend(title="Difference group")
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_pair_differences(
    participant_df: pd.DataFrame,
    output: Path,
    participant_id: object,
) -> None:
    """画像ペアごとの符号付き評定差を保存する。"""
    ordered = participant_df.sort_values(
        ["Difference_Group", "Group_Rank"],
        key=lambda values: values.map({"high": 0, "low": 1}).fillna(values)
        if values.name == "Difference_Group" else values,
    ).reset_index(drop=True)
    colors = ordered["Difference_Group"].map({
        "high": "#d62728",
        "low": "#1f77b4",
    })
    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x, ordered["Signed_Difference"], color=colors)
    ax.axhline(0, color="black", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["Pair_ID"], rotation=60, ha="right")
    ax.set_ylim(-4.3, 4.3)
    ax.set_ylabel("Score difference: dominant - non-dominant")
    ax.set_xlabel("Matched image pair")
    ax.set_title(f"Pairwise DP eye-condition differences\nID={participant_id}")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def save_outputs(
    paired: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """解析表と参加者別の3図を保存する。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(
        output_dir / "dp_dom_nondom_pair_differences.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        output_dir / "dp_dom_nondom_rmsd_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    for participant_id, participant_df in paired.groupby("ID", sort=False):
        participant_summary = summary[summary["ID"] == participant_id]
        suffix = _safe_id(participant_id)
        plot_rmsd(
            participant_summary,
            output_dir / f"dp_dom_nondom_rmsd_{suffix}.png",
            participant_id,
        )
        plot_score_scatter(
            participant_df,
            participant_summary,
            output_dir / f"dp_dom_vs_nondom_scatter_{suffix}.png",
            participant_id,
        )
        plot_pair_differences(
            participant_df,
            output_dir / f"dp_dom_nondom_pair_differences_{suffix}.png",
            participant_id,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DP優位眼・非優位眼条件の同一画像ペア評定差を解析する"
    )
    parser.add_argument(
        "target_dir",
        nargs="?",
        help="image_evaluation.csvを含むセッションディレクトリ",
    )
    args = parser.parse_args()
    target_dir = resolve_target_dir(args.target_dir)
    output_dir = FIGURE_BASE_DIR / target_dir.name / "dp_eye_difference"
    frame = load_results(target_dir)
    paired = build_paired_scores(frame)
    summary = summarize_rmsd(paired)
    save_outputs(paired, summary, output_dir)
    print(f"Input: {target_dir / 'image_evaluation.csv'}")
    print(f"Output: {output_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()