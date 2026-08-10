# cd src && py -m experiment.pre_analyze.pre_analyze_image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import glob
import argparse

# 解析対象のフォルダ
script_dir = os.path.dirname(os.path.abspath(__file__))
lab_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

# 入力データのベースディレクトリ (pre_experiment_image.py の出力先)
DATA_BASE_DIR = os.path.join(lab_root, "results", "tables", "pre-experiment-image")

parser = argparse.ArgumentParser(description='Analyze Image experiment results.')
parser.add_argument('target_dir', nargs='?', help='Path to the target directory')
args = parser.parse_args()

if args.target_dir:
    TARGET_DIR = args.target_dir
    if not os.path.exists(TARGET_DIR):
        print(f"指定されたディレクトリが見つかりません: {TARGET_DIR}")
        exit()
else:
    # 最新のフォルダを検索
    if not os.path.exists(DATA_BASE_DIR):
        print(f"データディレクトリが見つかりません: {DATA_BASE_DIR}")
        exit()

    subdirs = [d for d in glob.glob(os.path.join(DATA_BASE_DIR, "*")) if os.path.isdir(d)]
    if not subdirs:
        print(f"データフォルダが見つかりません: {DATA_BASE_DIR}")
        exit()

    # 最新のフォルダを取得 (更新日時順)
    TARGET_DIR = max(subdirs, key=os.path.getmtime)

target_folder_name = os.path.basename(os.path.normpath(TARGET_DIR))
print(f"解析対象フォルダ: {TARGET_DIR}")

# 出力先ディレクトリの設定
OUTPUT_DIR = os.path.join(lab_root, "results", "figures", "pre-experiment-image", target_folder_name)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
print(f"結果出力先: {OUTPUT_DIR}")

# image実験本試行の結果だけを解析する。
# 同じフォルダに保存されるdefocus_matching.csvは列構成が異なるため対象外。
result_path = os.path.join(TARGET_DIR, "image_evaluation.csv")
if not os.path.isfile(result_path):
    raise FileNotFoundError(
        f"画像評価結果が見つかりません: {result_path}"
    )
file_paths = [result_path]

all_data = []

CURRENT_DISTANCE_COLUMNS = ("Distance_FG(cm)", "Distance_BG(cm)")
LEGACY_DISTANCE_COLUMNS = ("Distance1(cm)", "Distance2(cm)")
REQUIRED_COLUMNS = {"Condition", "Image_Win1", "Image_Win2", "Score"}

for file in file_paths:
    # 現行のimage実験が出力するCSVを読み込む。
    df = pd.read_csv(file, encoding="utf-8")
    if df.empty:
        print(f"WARN: 空のCSVを除外します: {file}")
        continue

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{file}: 必須列が不足しています: {missing}")

    if all(column in df.columns for column in CURRENT_DISTANCE_COLUMNS):
        fg_column, bg_column = CURRENT_DISTANCE_COLUMNS
    elif all(column in df.columns for column in LEGACY_DISTANCE_COLUMNS):
        # 過去に保存したCSVも解析できるよう、旧列名をフォールバックとして受け付ける。
        fg_column, bg_column = LEGACY_DISTANCE_COLUMNS
    else:
        raise ValueError(
            f"{file}: 距離列が見つかりません。"
            f"現行列={CURRENT_DISTANCE_COLUMNS}, 旧列={LEGACY_DISTANCE_COLUMNS}"
        )

    distance_fg = float(df[fg_column].iloc[0])
    distance_bg = float(df[bg_column].iloc[0])
    df["distance"] = f"{distance_fg:g}-{distance_bg:g} cm"
    # ソート用にディオプトリ差と前景距離を計算する。
    df["d1"] = distance_fg
    df["diopter_diff"] = (
        abs(100 / distance_fg - 100 / distance_bg)
        if distance_fg > 0 and distance_bg > 0
        else 0
    )
    all_data.append(df)

# 全データを統合
if not all_data:
    print("No data to process.")
    exit()
final_df = pd.concat(all_data, ignore_index=True)

# 3. 画像名からラベルを抽出 (fg/bg_texX_LumX)
def extract_label(filename, prefix):
    # ファイル名から texX と LumX を抽出
    tex_match = re.search(r'(tex\d+)', str(filename), re.IGNORECASE)
    # 新しい形式 (lumi_default, lumi_half, lumi_quarter)
    lum_match = re.search(r'(lumi_(?:default|half|quarter))', str(filename), re.IGNORECASE)
    # 古い形式 (LumL, LumM, LumH)
    if not lum_match:
        lum_match = re.search(r'(Lum[LMH])', str(filename), re.IGNORECASE)
    
    parts = [prefix]
    if tex_match:
        parts.append(tex_match.group(1))
    if lum_match:
        parts.append(lum_match.group(1))
    
    if len(parts) > 1:
        return "_".join(parts)
    
    # 旧形式などのフォールバック
    match = re.search(r'lum(\d+\.?\d*)', str(filename), re.IGNORECASE)
    if match:
        return f"{prefix}_{match.group(1)}nit"
    return str(filename)

# 元のファイル名でソートして順序を決定（ファイル名先頭の番号が輝度ランク順になっているため）
unique_files = sorted(final_df['Image_Win2'].unique())
fg_labels = []
seen = set()
for f in unique_files:
    lbl = extract_label(f, "fg")
    if lbl not in seen:
        fg_labels.append(lbl)
        seen.add(lbl)

final_df['fg_image'] = final_df['Image_Win2'].apply(lambda x: extract_label(x, "fg"))
final_df['fg_image'] = pd.Categorical(final_df['fg_image'], categories=fg_labels, ordered=True)

unique_bg_files = sorted(final_df['Image_Win1'].unique())
bg_labels = []
seen_bg = set()
for f in unique_bg_files:
    lbl = extract_label(f, "bg")
    if lbl not in seen_bg:
        bg_labels.append(lbl)
        seen_bg.add(lbl)

final_df['bg_image'] = final_df['Image_Win1'].apply(lambda x: extract_label(x, "bg"))
final_df['bg_image'] = pd.Categorical(final_df['bg_image'], categories=bg_labels, ordered=True)

# 4条件を固定順で解析する。
CONDITION_ORDER = [
    "Single plane",
    "Single plane + defocus simulation",
    "Single plane + defocus + binocular overlay",
    "Dual plane",
]
CONDITION_SLUGS = {
    "Single plane": "single_plane",
    "Single plane + defocus simulation": "single_plane_defocus",
    "Single plane + defocus + binocular overlay": "single_plane_defocus_binocular",
    "Dual plane": "dual_plane",
}
BASELINE_CONDITION = "Single plane"

final_df["Score"] = pd.to_numeric(final_df["Score"], errors="coerce")
if final_df["Score"].isna().any():
    raise ValueError("Scoreに数値へ変換できない値が含まれています")

available_conditions = set(final_df["Condition"].dropna().astype(str))
missing_conditions = [
    condition for condition in CONDITION_ORDER
    if condition not in available_conditions
]
if missing_conditions:
    raise ValueError(
        "解析に必要なConditionが不足しています: "
        f"{missing_conditions}; found={sorted(available_conditions)}"
    )
unexpected_conditions = sorted(available_conditions - set(CONDITION_ORDER))
if unexpected_conditions:
    print(f"WARN: 未定義のConditionは解析対象外です: {unexpected_conditions}")

# この解析の出力を毎回5枚だけにするため、同じ出力先の既存PNGを削除する。
for old_png in glob.glob(os.path.join(OUTPUT_DIR, "*.png")):
    os.remove(old_png)

# --- ConditionごとのBG×FGヒートマップ（4枚） ---
for condition in CONDITION_ORDER:
    condition_df = final_df[final_df["Condition"] == condition]
    heatmap_df = (
        condition_df
        .groupby(["bg_image", "fg_image"], observed=False)["Score"]
        .mean()
        .reset_index()
    )
    heatmap_pivot = heatmap_df.pivot(
        index="bg_image", columns="fg_image", values="Score"
    ).reindex(
        index=final_df["bg_image"].cat.categories,
        columns=final_df["fg_image"].cat.categories,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(
        heatmap_pivot.to_numpy(dtype=float),
        cmap="Reds",
        vmin=1,
        vmax=5,
        origin="lower",
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(heatmap_pivot.columns)))
    ax.set_yticks(np.arange(len(heatmap_pivot.index)))
    ax.set_xticklabels(heatmap_pivot.columns, rotation=45, ha="right")
    ax.set_yticklabels(heatmap_pivot.index)
    ax.set_xlabel("Foreground image")
    ax.set_ylabel("Background image")
    ax.set_title(f"BG × FG score heatmap\n{condition}")

    for row_index in range(len(heatmap_pivot.index)):
        for column_index in range(len(heatmap_pivot.columns)):
            value = heatmap_pivot.iloc[row_index, column_index]
            if pd.notna(value):
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > 3.5 else "black",
                )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Mean score")
    fig.tight_layout()
    output_path = os.path.join(
        OUTPUT_DIR,
        f"score_heatmap_{CONDITION_SLUGS[condition]}.png",
    )
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved heatmap: {output_path}")

# --- 画像ペアごとにSingle planeとの差を求め、その平均を4本で表示（1枚） ---
# 差の符号は「各ConditionのScore - Single planeのScore」とする。
# 同一条件・同一画像ペアに複数試行がある場合は、先にペア内平均を取る。
pair_scores = (
    final_df[final_df["Condition"].isin(CONDITION_ORDER)]
    .groupby(
        ["Image_Win1", "Image_Win2", "Condition"],
        as_index=False,
    )["Score"]
    .mean()
)
pair_pivot = pair_scores.pivot(
    index=["Image_Win1", "Image_Win2"],
    columns="Condition",
    values="Score",
)

complete_pairs = pair_pivot.dropna(subset=CONDITION_ORDER).copy()
if complete_pairs.empty:
    raise ValueError(
        "4条件がそろった画像ペアがないため、Single plane基準のスコア差を計算できません"
    )

score_differences = complete_pairs[CONDITION_ORDER].subtract(
    complete_pairs[BASELINE_CONDITION], axis=0
)
# 棒の高さは画像ペアごとのスコア差の平均、誤差棒はそのSEM。
difference_mean = score_differences.mean(axis=0).reindex(CONDITION_ORDER)
difference_sem = score_differences.sem(axis=0).reindex(CONDITION_ORDER).fillna(0.0)
# 基準条件自身との差は各画像ペアで必ず0なので、明示的に固定する。
difference_mean.loc[BASELINE_CONDITION] = 0.0
difference_sem.loc[BASELINE_CONDITION] = 0.0

fig, ax = plt.subplots(figsize=(12, 7))
x_positions = np.arange(len(CONDITION_ORDER))
bars = ax.bar(
    x_positions,
    difference_mean.to_numpy(dtype=float),
    yerr=difference_sem.to_numpy(dtype=float),
    capsize=5,
    color=["#7f7f7f", "#4c78a8", "#f58518", "#54a24b"],
)
ax.axhline(0.0, color="black", linestyle="--", linewidth=1.5)
ax.set_xticks(x_positions)
ax.set_xticklabels(CONDITION_ORDER, rotation=15, ha="right")
ax.set_xlabel("Condition")
ax.set_ylabel("Score difference from Single plane")
ax.set_title(
    "Mean score difference across matched BG × FG image pairs\n"
    f"n = {len(score_differences)} complete image pairs"
)
error_low = difference_mean.to_numpy() - difference_sem.to_numpy()
error_high = difference_mean.to_numpy() + difference_sem.to_numpy()
limit = max(
    0.5,
    float(np.nanmax(np.abs(np.concatenate([error_low, error_high])))) * 1.2,
)
ax.set_ylim(-limit, limit)
ax.grid(axis="y", linestyle="--", alpha=0.5)
for bar, value in zip(bars, difference_mean):
    vertical_alignment = "bottom" if value >= 0 else "top"
    offset = limit * 0.02 if value >= 0 else -limit * 0.02
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + offset,
        f"{value:+.3f}",
        ha="center",
        va=vertical_alignment,
    )
fig.tight_layout()
bar_path = os.path.join(OUTPUT_DIR, "score_difference_vs_single_plane.png")
fig.savefig(bar_path, dpi=300)
plt.close(fig)
print(f"Saved score-difference bar chart: {bar_path}")
print(f"Saved exactly 5 figures to: {OUTPUT_DIR}")