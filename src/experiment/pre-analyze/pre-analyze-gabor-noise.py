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
parser.add_argument('target_dirs', nargs='+', help='Paths to the target directories (e.g. 0_20260413 0_20260418)')
args = parser.parse_args()

TARGET_DIRS = []
for d in args.target_dirs:
    if os.path.exists(d):
        TARGET_DIRS.append(d)
    else:
        d_path = os.path.join(DATA_BASE_DIR, d)
        if os.path.exists(d_path):
            TARGET_DIRS.append(d_path)
        else:
            print(f"指定されたディレクトリが見つかりません: {d}")
            exit()

target_folder_name = "_".join([os.path.basename(os.path.normpath(d)) for d in TARGET_DIRS])
print(f"解析対象フォルダ: {TARGET_DIRS}")

# 出力先ディレクトリの設定
OUTPUT_DIR = os.path.join(lab_root, "results", "figures", "pre-experiment-gabor-noise", target_folder_name)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
print(f"結果出力先: {OUTPUT_DIR}")

# 対象フォルダ内のすべてのCSVファイルを取得
all_data = []
for target_dir in TARGET_DIRS:
    file_paths = glob.glob(os.path.join(target_dir, '*.csv'))
    if not file_paths:
        print(f"警告: フォルダ '{target_dir}' にCSVファイルが見つかりませんでした。")
        continue

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

# --- グラフ生成 ---
print("\n--- Generating heatmaps ---")

custom_order = ['50-70 cm', '81-150 cm', '50-100 cm', '60-150 cm']
cpd_order = [2.0, 4.0, 6.0, 8.0]

view_conds = ['Binocular', 'Monocular']
# データに存在する輝度の組み合わせを取得
lum_combinations = final_df[['bg_lum', 'fg_lum']].drop_duplicates().sort_values(by=['bg_lum', 'fg_lum']).values.tolist()

for view_cond in view_conds:
    for bg_lum, fg_lum in lum_combinations:
        subset_df_0 = final_df[(final_df['Viewing_Condition'] == view_cond) & 
                               (final_df['bg_lum'] == bg_lum) & 
                               (final_df['fg_lum'] == fg_lum) &
                               (final_df['bg_contrast'] == 0.0)]
        
        subset_df_1 = final_df[(final_df['Viewing_Condition'] == view_cond) & 
                               (final_df['bg_lum'] == bg_lum) & 
                               (final_df['fg_lum'] == fg_lum) &
                               (final_df['bg_contrast'] == 1.0)]
        
        if subset_df_0.empty or subset_df_1.empty:
            continue
            
        summary_0 = subset_df_0.groupby(['bg_cpd', 'distance'])['Score'].mean().reset_index()
        pivot_0 = summary_0.pivot(index='bg_cpd', columns='distance', values='Score')
        
        summary_1 = subset_df_1.groupby(['bg_cpd', 'distance'])['Score'].mean().reset_index()
        pivot_1 = summary_1.pivot(index='bg_cpd', columns='distance', values='Score')
        
        # 差分を計算 (bg_contrast: 0.0 - 1.0)
        pivot_table = pivot_0 - pivot_1
        
        # 軸の順序を適用 (存在しない列・行はNaNになる)
        pivot_table = pivot_table.reindex(index=cpd_order, columns=custom_order)
        
        # 絶対値の最大値より大きい中で最小の整数を求める
        max_abs_val = np.nanmax(np.abs(pivot_table.values))
        if np.isnan(max_abs_val):
            continue
        max_int = int(np.ceil(max_abs_val))
        if max_int == 0:
            max_int = 1
            
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(pivot_table, cmap='coolwarm', vmin=-2, vmax=2, origin='lower', aspect='auto')
        
        ax.set_xticks(np.arange(len(pivot_table.columns)))
        ax.set_yticks(np.arange(len(pivot_table.index)))
        ax.set_xticklabels(pivot_table.columns)
        ax.set_yticklabels([f"{int(c)}" for c in pivot_table.index])
        
        ax.set_xlabel('Distance')
        ax.set_ylabel('Spatial Frequency (cpd)')
        
        title = f'Score Difference (Contrast 0.0 - 1.0)\n({view_cond}, BG: {bg_lum}nit, FG: {fg_lum}nit)'
        ax.set_title(title, fontsize=14)
        
        # 各セルに数値を書き込む
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                val = pivot_table.iloc[i, j]
                if not np.isnan(val):
                    text_color = "white" if abs(val) > (max_int * 0.6) else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color)
                    
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Score Difference')
        
        plt.tight_layout()
        
        filename = f'heatmap_diff_{view_cond}_bg{bg_lum}_fg{fg_lum}.png'
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        print(f"グラフ保存: {filename}")
        plt.close(fig)