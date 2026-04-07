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

# 入力データのベースディレクトリ (calculate_visibility_scores_gabor.py の出力先)
DATA_BASE_DIR = os.path.join(lab_root, "results", "tables", "pre-experiment-gabor", "predicted_visibility_score")

parser = argparse.ArgumentParser(description='Analyze predicted Gabor and Noise visibility scores.')
parser.add_argument('target_file', nargs='?', help='Path to the target CSV file')
args = parser.parse_args()

if args.target_file:
    TARGET_FILE = args.target_file
    if not os.path.exists(TARGET_FILE):
        print(f"指定されたファイルが見つかりません: {TARGET_FILE}")
        exit()
else:
    # 最新のCSVファイルを検索
    if not os.path.exists(DATA_BASE_DIR):
        print(f"データディレクトリが見つかりません: {DATA_BASE_DIR}")
        exit()

    csv_files = glob.glob(os.path.join(DATA_BASE_DIR, "*.csv"))
    if not csv_files:
        print(f"データファイルが見つかりません: {DATA_BASE_DIR}")
        exit()

    # 最新のファイルを取得 (更新日時順)
    TARGET_FILE = max(csv_files, key=os.path.getmtime)

target_file_name = os.path.basename(TARGET_FILE)
print(f"解析対象ファイル: {target_file_name}")

# 出力先ディレクトリの設定
OUTPUT_DIR = os.path.join(lab_root, "results", "figures", "pre-experiment-gabor-noise", "predicted-visibility-score")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
print(f"結果出力先: {OUTPUT_DIR}")

# CSVファイルを読み込み
final_df = pd.read_csv(TARGET_FILE)

# --- パラメータ抽出 ---
def extract_gabor_noise_parameters(filename):
    # ファイル名形式: FG_{fg_cpd}_{fg_lum}_{fg_contrast}_BG_{bg_cpd}_{bg_lum}_{bg_contrast}.png
    pattern = r'FG_(\d+\.?\d*)_(\d+\.?\d*)_(\d+\.?\d*)_BG_(\d+\.?\d*)_(\d+\.?\d*)_(\d+\.?\d*)\.png'
    match = re.search(pattern, str(filename))
    if match:
        fg_cpd = float(match.group(1))
        fg_lum = float(match.group(2))
        fg_contrast = float(match.group(3))
        bg_cpd = float(match.group(4))
        bg_lum = float(match.group(5))
        bg_contrast = float(match.group(6))
        return fg_cpd, fg_lum, fg_contrast, bg_cpd, bg_lum, bg_contrast
    return None, None, None, None, None, None

# Image_Win1 (or Image_Win2, they are the same) からパラメータを抽出
params = final_df['Image_Win1'].apply(lambda x: pd.Series(extract_gabor_noise_parameters(x)))
params.columns = ['fg_cpd', 'fg_lum', 'fg_contrast', 'bg_cpd', 'bg_lum', 'bg_contrast']
final_df = pd.concat([final_df, params], axis=1)

# 距離カラムの生成
final_df['distance'] = final_df['Distance1(cm)'].astype(str) + '-' + final_df['Distance2(cm)'].astype(str) + ' cm'

# 欠損値の除去
final_df.dropna(subset=['fg_cpd', 'fg_lum', 'fg_contrast', 'bg_cpd', 'bg_lum', 'bg_contrast'], inplace=True)

if final_df.empty:
    print("有効なパラメータを抽出できませんでした。ファイル名形式を確認してください。")
    exit()

# データ型変換
final_df['fg_cpd'] = final_df['fg_cpd'].astype(int)
final_df['bg_cpd'] = final_df['bg_cpd'].astype(int)

# --- 集計 ---
# グルーピング: 距離, 背景空間周波数
# グルーピング: 前景空間周波数, 距離, 背景空間周波数
# 他の条件（輝度、コントラスト）はすべて平均してまとめる
summary_df = final_df.groupby(['fg_cpd', 'distance', 'bg_cpd'])['Score'].mean().reset_index()

# 距離のソート順序指定
custom_order = ['50-70 cm', '81-150 cm', '50-100 cm', '60-150 cm']
try:
    summary_df['distance'] = pd.Categorical(summary_df['distance'], categories=custom_order, ordered=True)
except ValueError:
    pass

# --- グラフ描画 ---
# 要件: ヒートマップ
# 縦軸: 背景の空間周波数 (bg_cpd)
# 横軸: 距離 (distance, 4パターン)
# 値: スコア (全条件の平均)
# 値: スコア (平均)
# 前景の空間周波数(fg_cpd)でグラフを分ける

unique_fg_cpds = sorted(summary_df['fg_cpd'].unique())

for fgcpd in unique_fg_cpds:
    plot_df = summary_df[summary_df['fg_cpd'] == fgcpd]

    # ピボットテーブル作成 (行: bg_cpd, 列: distance, 値: Score)
    pivot_table = plot_df.pivot(index='bg_cpd', columns='distance', values='Score')

    # 軸の整列
    # 縦軸(bg_cpd): 数値順 (昇順) -> origin='lower' で小さい値が下に来るようにする
    pivot_table.sort_index(ascending=True, inplace=True)
    # 横軸(distance): 指定順序に並べ替え
    pivot_table = pivot_table.reindex(columns=custom_order)
        
    fig, ax = plt.subplots(figsize=(8, 6))

    # ヒートマップ描画
    # vmin/vmax を予測スコアの範囲(0-1)に調整
    im = ax.imshow(pivot_table, cmap='viridis', vmin=0, vmax=1, origin='lower', aspect='auto')

    # 数値の表示
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            val = pivot_table.iloc[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 0.5 else "black" # 視認性調整
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=10)

    # 軸ラベル設定
    ax.set_xticks(np.arange(len(pivot_table.columns)))
    ax.set_yticks(np.arange(len(pivot_table.index)))
    ax.set_xticklabels(pivot_table.columns)
    ax.set_yticklabels(pivot_table.index)

    ax.set_xlabel('Distance')
    ax.set_ylabel('Background Spatial Frequency (cpd)')
    ax.set_title(f'Predicted Score Heatmap (FG: {fgcpd} cpd)', fontsize=14)

    # カラーバー
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Predicted Score')

    plt.tight_layout()

    filename = f'heatmap_score_fgcpd_{fgcpd}.png'
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    print(f"グラフ保存: {filename}")
    plt.close(fig)

# --- 追加: 前景コントラストごとに分割 ---
summary_contrast = final_df.groupby(['fg_contrast', 'distance', 'bg_cpd'])['Score'].mean().reset_index()
try:
    summary_contrast['distance'] = pd.Categorical(summary_contrast['distance'], categories=custom_order, ordered=True)
except ValueError:
    pass

unique_contrasts = sorted(summary_contrast['fg_contrast'].unique())

for contrast in unique_contrasts:
    plot_df = summary_contrast[summary_contrast['fg_contrast'] == contrast]
    
    pivot_table = plot_df.pivot(index='bg_cpd', columns='distance', values='Score')
    pivot_table.sort_index(ascending=True, inplace=True)
    pivot_table = pivot_table.reindex(columns=custom_order)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot_table, cmap='viridis', vmin=0, vmax=1, origin='lower', aspect='auto')
    
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            val = pivot_table.iloc[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=10)
    
    ax.set_xticks(np.arange(len(pivot_table.columns)))
    ax.set_yticks(np.arange(len(pivot_table.index)))
    ax.set_xticklabels(pivot_table.columns)
    ax.set_yticklabels(pivot_table.index)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Background Spatial Frequency (cpd)')
    ax.set_title(f'Predicted Score Heatmap (FG Contrast: {contrast})', fontsize=14)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Predicted Score')
    plt.tight_layout()
    
    filename = f'heatmap_score_contrast_{contrast}.png'
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    print(f"グラフ保存: {filename}")
    plt.close(fig)

# --- 追加: 輝度条件(FG/BG)ごとに分割 ---
summary_lum = final_df.groupby(['fg_lum', 'bg_lum', 'distance', 'bg_cpd'])['Score'].mean().reset_index()
try:
    summary_lum['distance'] = pd.Categorical(summary_lum['distance'], categories=custom_order, ordered=True)
except ValueError:
    pass

unique_lum_pairs = summary_lum[['fg_lum', 'bg_lum']].drop_duplicates().sort_values(by=['fg_lum', 'bg_lum'])

for idx, row in unique_lum_pairs.iterrows():
    fglum = row['fg_lum']
    bglum = row['bg_lum']
    
    plot_df = summary_lum[(summary_lum['fg_lum'] == fglum) & (summary_lum['bg_lum'] == bglum)]
    pivot_table = plot_df.pivot(index='bg_cpd', columns='distance', values='Score')
    pivot_table.sort_index(ascending=True, inplace=True)
    pivot_table = pivot_table.reindex(columns=custom_order)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot_table, cmap='viridis', vmin=0, vmax=1, origin='lower', aspect='auto')
    
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            val = pivot_table.iloc[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=10)
    
    ax.set_xticks(np.arange(len(pivot_table.columns)))
    ax.set_yticks(np.arange(len(pivot_table.index)))
    ax.set_xticklabels(pivot_table.columns)
    ax.set_yticklabels(pivot_table.index)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Background Spatial Frequency (cpd)')
    ax.set_title(f'Predicted Score Heatmap (FG: {fglum}nit, BG: {bglum}nit)', fontsize=14)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Predicted Score')
    plt.tight_layout()
    
    filename = f'heatmap_score_lum_fg{fglum}_bg{bglum}.png'
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    print(f"グラフ保存: {filename}")
    plt.close(fig)

# --- 追加: 距離ごとに x=前景cpd, y=背景cpd のヒートマップを作成 ---
for distance in custom_order:
    dist_df = final_df[final_df['distance'] == distance]
    if dist_df.empty:
        print(f"距離 '{distance}' のデータがありません。スキップします。")
        continue

    summary_dist = dist_df.groupby(['fg_cpd', 'bg_cpd'])['Score'].mean().reset_index()
    pivot_table = summary_dist.pivot(index='bg_cpd', columns='fg_cpd', values='Score')

    if pivot_table.empty:
        print(f"距離 '{distance}' で有効な集計がありません。スキップします。")
        continue

    pivot_table.sort_index(ascending=True, inplace=True)
    pivot_table = pivot_table.reindex(columns=sorted(pivot_table.columns))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot_table, cmap='viridis', vmin=0, vmax=1, origin='lower', aspect='auto')

    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            val = pivot_table.iloc[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=10)

    ax.set_xticks(np.arange(len(pivot_table.columns)))
    ax.set_yticks(np.arange(len(pivot_table.index)))
    ax.set_xticklabels(pivot_table.columns)
    ax.set_yticklabels(pivot_table.index)

    ax.set_xlabel('Foreground Spatial Frequency (cpd)')
    ax.set_ylabel('Background Spatial Frequency (cpd)')
    ax.set_title(f'Predicted Score Heatmap (Distance: {distance})', fontsize=14)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Predicted Score')
    plt.tight_layout()

    filename = f'heatmap_score_dist_{distance.replace(" ", "_").replace("-", "to")}.png'
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    print(f"グラフ保存: {filename}")
    plt.close(fig)