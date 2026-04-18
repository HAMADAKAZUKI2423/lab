# py .\src\experiment\pre-analyze\pre-analyze-gabor-noise.py
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

# 入力データのベースディレクトリ (実験名に合わせてフォルダ名を指定)
# generate_gabor_and_noise_stimuli.py を使用した実験の結果を想定
DATA_BASE_DIR = os.path.join(lab_root, "results", "tables", "pre-experiment-gabor")

parser = argparse.ArgumentParser(description='Analyze Gabor and Noise experiment results.')
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
        print("ヒント: ディレクトリ名が異なる場合は、スクリプト内の DATA_BASE_DIR を修正してください。")
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
OUTPUT_DIR = os.path.join(lab_root, "results", "figures", "pre-experiment-gabor-noise", target_folder_name)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
print(f"結果出力先: {OUTPUT_DIR}")

# フォルダ内のすべてのCSVファイルを取得
file_paths = glob.glob(os.path.join(TARGET_DIR, '*.csv'))

if not file_paths:
    print(f"解析対象のCSVファイルがフォルダ '{TARGET_DIR}' に見つかりませんでした。")
    exit()

all_data = []

for file in file_paths:
    df = pd.read_csv(file, encoding='utf-8')

    if 'Viewing_Condition' not in df.columns:
        df['Viewing_Condition'] = 'Binocular'

    d1 = df['Distance1(cm)'].iloc[0]
    d2 = df['Distance2(cm)'].iloc[0]
    df['distance'] = f"{d1}-{d2} cm"
    print(f"ファイル '{os.path.basename(file)}' を読み込みました。距離: {d1} cm - {d2} cm")
    
    all_data.append(df)

if not all_data:
    print("No data to process.")
    exit()
final_df = pd.concat(all_data, ignore_index=True)

# --- パラメータ抽出 ---
def extract_parameters(row):
    # ファイル名形式1: {freq}cpd_{lum}nit_{contrast}... を想定
    # ファイル名形式2: FG_{lum}_{contrast}_BG_{lum}_{contrast}.png
    fg_file = str(row['Image_Win2'])
    bg_file = str(row['Image_Win1'])
    
    # 形式1 (旧形式)
    match_fg = re.search(r'(\d+\.?\d*)cpd_(\d+\.?\d*)nit_(\d+\.?\d*)', fg_file)
    match_bg = re.search(r'(\d+\.?\d*)cpd_(\d+\.?\d*)nit_(\d+\.?\d*)', bg_file)
    
    if match_fg and match_bg:
        return pd.Series([
            float(match_fg.group(2)), float(match_fg.group(1)), float(match_fg.group(3)),
            float(match_bg.group(2)), float(match_bg.group(1)), float(match_bg.group(3))
        ])
    
    # 形式2 (新形式: generate_gabor_and_noise_stimuli.py)
    match_new = re.search(r'FG_(\d+\.?\d*)_(\d+\.?\d*)_BG_(\d+\.?\d*)_(\d+\.?\d*)', fg_file)
    if match_new:
        fg_lum = float(match_new.group(1))
        fg_contrast = float(match_new.group(2))
        bg_lum = float(match_new.group(3))
        bg_contrast = float(match_new.group(4))
        # Spatial_Freq(cpd) 列から取得
        cpd = row.get('Spatial_Freq(cpd)', None)
        return pd.Series([fg_lum, cpd, fg_contrast, bg_lum, cpd, bg_contrast])
        
    return pd.Series([None] * 6)

# パラメータ抽出の実行
param_cols = ['fg_lum', 'fg_cpd', 'fg_contrast', 'bg_lum', 'bg_cpd', 'bg_contrast']
final_df[param_cols] = final_df.apply(extract_parameters, axis=1)

# 欠損値の除去
final_df.dropna(subset=param_cols, inplace=True)

if final_df.empty:
    print("有効なパラメータを抽出できませんでした。ファイル名形式やCSVの列を確認してください。")
    exit()

# データ型変換
final_df['fg_cpd'] = final_df['fg_cpd'].astype(float)
final_df['bg_cpd'] = final_df['bg_cpd'].astype(float)

# --- 集計 ---
# グルーピング: 距離, 背景空間周波数
# グルーピング: 前景空間周波数, 距離, 背景空間周波数
# 他の条件（輝度、コントラスト）はすべて平均してまとめる
summary_df = final_df.groupby(['fg_cpd', 'distance', 'bg_cpd'])['Score'].mean().reset_index()

# 距離のソート順序指定
custom_order_base = ['50-70 cm', '81-150 cm', '50-100 cm', '60-150 cm']
found_distances = summary_df['distance'].unique().tolist()
custom_order = [d for d in custom_order_base if d in found_distances]
for d in found_distances:
    if d not in custom_order:
        custom_order.append(d)

try:
    summary_df['distance'] = pd.Categorical(summary_df['distance'], categories=custom_order, ordered=True)
except ValueError:
    pass

# --- 要望1: 単眼/複眼 x 背景コントラストごとのヒートマップ ---
print("\n--- Generating heatmaps by Viewing Condition and Background Contrast ---")
summary_view_contrast = final_df.groupby(['Viewing_Condition', 'bg_contrast', 'distance', 'bg_cpd'])['Score'].mean().reset_index()
try:
    summary_view_contrast['distance'] = pd.Categorical(summary_view_contrast['distance'], categories=custom_order, ordered=True)
except ValueError:
    pass

unique_conditions = summary_view_contrast[['Viewing_Condition', 'bg_contrast']].drop_duplicates().sort_values(by=['Viewing_Condition', 'bg_contrast'])

for idx, row in unique_conditions.iterrows():
    view_cond = row['Viewing_Condition']
    bg_contrast = row['bg_contrast']
    
    plot_df = summary_view_contrast[(summary_view_contrast['Viewing_Condition'] == view_cond) & (summary_view_contrast['bg_contrast'] == bg_contrast)]
    if plot_df.empty:
        continue
    
    pivot_table = plot_df.pivot(index='bg_cpd', columns='distance', values='Score')
    pivot_table.sort_index(ascending=True, inplace=True)
    pivot_table = pivot_table.reindex(columns=custom_order)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot_table, cmap='coolwarm', vmin=1, vmax=5, origin='lower', aspect='auto')

    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            val = pivot_table.iloc[i, j]
            if not np.isnan(val):
                text_color = "white" if (val < 2.5 or val > 4.0) else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=10)

    ax.set_xticks(np.arange(len(pivot_table.columns)))
    ax.set_yticks(np.arange(len(pivot_table.index)))
    ax.set_xticklabels(pivot_table.columns)
    ax.set_yticklabels(pivot_table.index)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Background Spatial Frequency (cpd)')
    ax.set_title(f'Average Score Heatmap\n({view_cond}, BG Contrast: {bg_contrast})', fontsize=14)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Score')
    plt.tight_layout()

    filename = f'heatmap_score_view_{view_cond}_bgcontrast_{bg_contrast}.png'
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    print(f"グラフ保存: {filename}")
    plt.close(fig)

# --- 要望2: 輝度組み合わせごとのヒートマップ ---
print("\n--- Generating heatmaps by Luminance Combination and Viewing Condition ---")
summary_lum = final_df.groupby(['Viewing_Condition', 'fg_lum', 'bg_lum', 'distance', 'bg_cpd'])['Score'].mean().reset_index()
try:
    summary_lum['distance'] = pd.Categorical(summary_lum['distance'], categories=custom_order, ordered=True)
except ValueError:
    pass

unique_lum_conditions = summary_lum[['Viewing_Condition', 'fg_lum', 'bg_lum']].drop_duplicates().sort_values(by=['Viewing_Condition', 'fg_lum', 'bg_lum'])

for idx, row in unique_lum_conditions.iterrows():
    view_cond = row['Viewing_Condition']
    fglum = row['fg_lum']
    bglum = row['bg_lum']
    
    plot_df = summary_lum[(summary_lum['Viewing_Condition'] == view_cond) & (summary_lum['fg_lum'] == fglum) & (summary_lum['bg_lum'] == bglum)]
    pivot_table = plot_df.pivot(index='bg_cpd', columns='distance', values='Score')
    pivot_table.sort_index(ascending=True, inplace=True)
    pivot_table = pivot_table.reindex(columns=custom_order)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot_table, cmap='coolwarm', vmin=1, vmax=5, origin='lower', aspect='auto')
    
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            val = pivot_table.iloc[i, j]
            if not np.isnan(val):
                text_color = "white" if (val < 2.5 or val > 4.0) else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=10)
    
    ax.set_xticks(np.arange(len(pivot_table.columns)))
    ax.set_yticks(np.arange(len(pivot_table.index)))
    ax.set_xticklabels(pivot_table.columns)
    ax.set_yticklabels(pivot_table.index)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Background Spatial Frequency (cpd)')
    ax.set_title(f'Average Score Heatmap\n({view_cond}, FG: {fglum}nit, BG: {bglum}nit)', fontsize=14)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Score')
    plt.tight_layout()
    
    filename = f'heatmap_score_view_{view_cond}_lum_fg{fglum}_bg{bglum}.png'
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    print(f"グラフ保存: {filename}")
    plt.close(fig)