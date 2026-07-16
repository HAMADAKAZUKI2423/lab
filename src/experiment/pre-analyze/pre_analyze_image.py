# py .\src\experiment\pre-analyze\pre_analyze_image.py
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

# フォルダ内のすべてのCSVファイルを取得
file_paths = glob.glob(os.path.join(TARGET_DIR, '*.csv'))

# ファイルが見つからない場合はメッセージを表示して終了
if not file_paths:
    print(f"解析対象のCSVファイルがフォルダ '{TARGET_DIR}' に見つかりませんでした。")
    exit()

all_data = []

for file in file_paths:
    # 1. データをヘッダー付きで正しく読み込む
    df = pd.read_csv(file, encoding='utf-8')
    d1 = df['Distance1(cm)'].iloc[0]
    d2 = df['Distance2(cm)'].iloc[0]
    df['distance'] = f"{d1}-{d2} cm"
    # ソート用にディオプトリ差と前景距離(d1)を計算
    df['d1'] = d1
    df['diopter_diff'] = abs(100/d1 - 100/d2) if d1 > 0 and d2 > 0 else 0
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

# 4. 条件ごとの平均値と標準誤差を算出
summary_df = final_df.groupby(['distance', 'diopter_diff', 'd1', 'fg_image'])['Score'].agg(['mean', 'sem']).reset_index()

# 指定された順序 `50-70, 81-150, 50-100, 60-150` でグラフのx軸を並べ替える
custom_order = ['50-70 cm', '81-150 cm', '50-100 cm', '60-150 cm']
summary_df['distance'] = pd.Categorical(summary_df['distance'], categories=custom_order, ordered=True)
# 指定したカテゴリ順でソート
summary_df = summary_df.sort_values('distance')

# グラフ描画用にピボット（行：距離ラベル、列：前景画像、値：平均スコア）
pivot_df = summary_df.pivot_table(index='distance', columns='fg_image', values='mean')
error_df = summary_df.pivot_table(index='distance', columns='fg_image', values='sem')

# 4. グラフの描画
# 棒グラフに変更
fig, ax = plt.subplots(figsize=(12, 6))

# pandasのplot機能を使って棒グラフを描画
# x軸: distance (index), 凡例: fg_image (columns)
pivot_df.plot(kind='bar', yerr=error_df, ax=ax, capsize=4, rot=0)

# 軸ラベルとタイトルの設定
ax.set_xlabel('Distance Combination', fontsize=12)
ax.set_ylabel('Average Score', fontsize=12)
ax.set_title('Average Score by Distance and Foreground Image', fontsize=14)
ax.set_ylim(0, 5.5) # スコアの範囲に合わせて調整
ax.legend(title='FG Image', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'score_vs_distance_bar.png'))
# 続けて背景画像のグラフを描画するため、ここではshow()せず最後にまとめて表示

# --- 追加: 背景画像ごとの集計とグラフ描画 ---
summary_bg_df = final_df.groupby(['distance', 'bg_image'])['Score'].agg(['mean', 'sem']).reset_index()
summary_bg_df['distance'] = pd.Categorical(summary_bg_df['distance'], categories=custom_order, ordered=True)
summary_bg_df = summary_bg_df.sort_values('distance')

pivot_bg_df = summary_bg_df.pivot_table(index='distance', columns='bg_image', values='mean')
error_bg_df = summary_bg_df.pivot_table(index='distance', columns='bg_image', values='sem')

fig_bg, ax_bg = plt.subplots(figsize=(12, 6))
pivot_bg_df.plot(kind='bar', yerr=error_bg_df, ax=ax_bg, capsize=4, rot=0)

ax_bg.set_xlabel('Distance Combination', fontsize=12)
ax_bg.set_ylabel('Average Score', fontsize=12)
ax_bg.set_title('Average Score by Distance and Background Luminance', fontsize=14)
ax_bg.set_ylim(0, 5.5)
ax_bg.legend(title='BG Image', bbox_to_anchor=(1.05, 1), loc='upper left')
ax_bg.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'score_vs_distance_bg_bar.png'))

# --- 追加: 前景・背景輝度のヒートマップ (距離ごとに作成) ---
unique_distances = final_df['distance'].unique()
# custom_order にあるものはその順で、ないものはその後に追加
sorted_distances = [d for d in custom_order if d in unique_distances] + [d for d in unique_distances if d not in custom_order]

for dist in sorted_distances:
    dist_df = final_df[final_df['distance'] == dist]
    
    # 集計
    heatmap_df = dist_df.groupby(['bg_image', 'fg_image'])['Score'].mean().reset_index()
    heatmap_pivot = heatmap_df.pivot(index='bg_image', columns='fg_image', values='Score')
    # カテゴリ順序を保持して再インデックス（データが存在しない組み合わせも表示するため）
    heatmap_pivot = heatmap_pivot.reindex(index=final_df['bg_image'].cat.categories, columns=final_df['fg_image'].cat.categories)

    fig_hm, ax_hm = plt.subplots(figsize=(8, 8))
    im = ax_hm.imshow(heatmap_pivot, cmap='Reds', vmin=1, vmax=5, origin='lower')

    ax_hm.set_xticks(np.arange(len(heatmap_pivot.columns)))
    ax_hm.set_yticks(np.arange(len(heatmap_pivot.index)))
    ax_hm.set_xticklabels(heatmap_pivot.columns, rotation=45, ha="right")
    ax_hm.set_yticklabels(heatmap_pivot.index)
    ax_hm.set_xlabel('Foreground Luminance')
    ax_hm.set_ylabel('Background Luminance')
    ax_hm.set_title(f'Average Score Heatmap (FG vs BG)\nDistance: {dist}')

    for i in range(len(heatmap_pivot.index)):
        for j in range(len(heatmap_pivot.columns)):
            val = heatmap_pivot.iloc[i, j]
            if not np.isnan(val):
                text_color = "white" if val > 3.5 else "black"
                ax_hm.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color)

    cbar = ax_hm.figure.colorbar(im, ax=ax_hm)
    cbar.set_label('Average Score')

    plt.tight_layout()
    safe_dist = str(dist).replace(' ', '_')
    plt.savefig(os.path.join(OUTPUT_DIR, f'score_heatmap_fg_bg_{safe_dist}.png'))

# --- 追加: 全距離平均のヒートマップ ---
heatmap_all_df = final_df.groupby(['bg_image', 'fg_image'])['Score'].mean().reset_index()
heatmap_all_pivot = heatmap_all_df.pivot(index='bg_image', columns='fg_image', values='Score')
# カテゴリ順序を保持して再インデックス
heatmap_all_pivot = heatmap_all_pivot.reindex(index=final_df['bg_image'].cat.categories, columns=final_df['fg_image'].cat.categories)

fig_hm_all, ax_hm_all = plt.subplots(figsize=(8, 8))
im_all = ax_hm_all.imshow(heatmap_all_pivot, cmap='Reds', vmin=1, vmax=5, origin='lower')

ax_hm_all.set_xticks(np.arange(len(heatmap_all_pivot.columns)))
ax_hm_all.set_yticks(np.arange(len(heatmap_all_pivot.index)))
ax_hm_all.set_xticklabels(heatmap_all_pivot.columns, rotation=45, ha="right")
ax_hm_all.set_yticklabels(heatmap_all_pivot.index)
ax_hm_all.set_xlabel('Foreground Luminance')
ax_hm_all.set_ylabel('Background Luminance')
ax_hm_all.set_title('Average Score Heatmap (FG vs BG)\nDistance: ALL')

for i in range(len(heatmap_all_pivot.index)):
    for j in range(len(heatmap_all_pivot.columns)):
        val = heatmap_all_pivot.iloc[i, j]
        if not np.isnan(val):
            text_color = "white" if val > 3.5 else "black"
            ax_hm_all.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color)

cbar_all = ax_hm_all.figure.colorbar(im_all, ax=ax_hm_all)
cbar_all.set_label('Average Score')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'score_heatmap_fg_bg_ALL.png'))

plt.show()