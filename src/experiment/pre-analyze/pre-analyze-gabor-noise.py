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
DATA_BASE_DIR = os.path.join(lab_root, "results", "tables", "pre-experiment-gabor-noise")

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
    # ヘッダー列名定義
    col_names = [
        "ID", "Age", "Gender", "IPD(mm)", "Distance1(cm)", "Distance2(cm)",
        "Offset_X", "Offset_Y", "Trial_ID", "Image_Win1", "Image_Win2", "Score"
    ]
    # header=0 で既存ヘッダーを無視して names で上書き
    df = pd.read_csv(file, encoding='utf-8', header=0, names=col_names)

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
def extract_parameters(filename):
    # ファイル名形式: {freq}cpd_{lum}nit_{contrast}... を想定
    # 背景がノイズ画像の場合も cpd がファイル名に含まれていると仮定
    match = re.search(r'(\d+)cpd_(\d+\.?\d*)nit_(\d+\.?\d*)', str(filename))
    if match:
        cpd = int(match.group(1))
        luminance = float(match.group(2))
        contrast = float(match.group(3))
        return luminance, cpd, contrast
    return None, None, None

# FGパラメータ抽出 (Image_Win2: 前景)
final_df[['fg_lum', 'fg_cpd', 'fg_contrast']] = final_df['Image_Win2'].apply(
    lambda x: pd.Series(extract_parameters(x))
)

# BGパラメータ抽出 (Image_Win1: 背景)
final_df[['bg_lum', 'bg_cpd', 'bg_contrast']] = final_df['Image_Win1'].apply(
    lambda x: pd.Series(extract_parameters(x))
)

# 欠損値の除去 (bg_cpdなどが取得できなかった場合など)
final_df.dropna(subset=['fg_lum', 'fg_cpd', 'fg_contrast', 'bg_lum', 'bg_cpd'], inplace=True)

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
    # vmin=1, vmax=5 (スコア範囲)
    im = ax.imshow(pivot_table, cmap='coolwarm', vmin=1, vmax=5, origin='lower', aspect='auto')

    # 数値の表示
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            val = pivot_table.iloc[i, j]
            if not np.isnan(val):
                text_color = "white" if (val < 2.5 or val > 4.0) else "black" # 視認性調整
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=10)

    # 軸ラベル設定
    ax.set_xticks(np.arange(len(pivot_table.columns)))
    ax.set_yticks(np.arange(len(pivot_table.index)))
    ax.set_xticklabels(pivot_table.columns)
    ax.set_yticklabels(pivot_table.index)

    ax.set_xlabel('Distance')
    ax.set_ylabel('Background Spatial Frequency (cpd)')
    ax.set_title(f'Average Score Heatmap (FG: {fgcpd} cpd)', fontsize=14)

    # カラーバー
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Score')

    plt.tight_layout()

    filename = f'heatmap_score_fgcpd_{fgcpd}.png'
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    print(f"グラフ保存: {filename}")
    plt.close(fig)