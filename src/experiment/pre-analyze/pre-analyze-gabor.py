# py .\src\experiment\pre-analyze\pre-analyze-gabor.py
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

# 入力データのベースディレクトリ (pre-experiment-gabor.py の出力先)
DATA_BASE_DIR = os.path.join(lab_root, "results", "tables", "pre-experiment-gabor")

parser = argparse.ArgumentParser(description='Analyze Gabor experiment results.')
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
OUTPUT_DIR = os.path.join(lab_root, "results", "figures", "pre-experiment-gabor", target_folder_name)
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
    # pre.pyの出力形式に合わせてカラム名を明示的に指定する
    # (既存のデータはヘッダー列数が不足している可能性があるため、header=0で無視してnamesで上書きする)
    col_names = [
        "ID", "Age", "Gender", "IPD(mm)", "Distance1(cm)", "Distance2(cm)",
        "Offset_X", "Offset_Y", "Trial_ID", "Image_Win1", "Image_Win2", "Score"
    ]
    df = pd.read_csv(file, encoding='utf-8', header=0, names=col_names)

    d1 = df['Distance1(cm)'].iloc[0]
    d2 = df['Distance2(cm)'].iloc[0]
    df['distance'] = f"{d1}-{d2} cm"
    print(f"ファイル '{os.path.basename(file)}' を読み込みました。距離: {d1} cm - {d2} cm")
    # ソート用にディオプトリ差と前景距離(d1)を計算
    df['d1'] = d1
    df['diopter_diff'] = abs(100/d1 - 100/d2) if d1 > 0 and d2 > 0 else 0
    all_data.append(df)

# 全データを統合
if not all_data:
    print("No data to process.")
    exit()
final_df = pd.concat(all_data, ignore_index=True)

# 3. 画像名からパラメータ（平均輝度、空間周波数、コントラスト）を抽出する関数
def extract_parameters(filename):
    # 例: 2cpd_50nit_0.8_h.png
    match = re.search(r'(\d+)cpd_(\d+\.?\d*)nit_(\d+\.?\d*)', str(filename))
    if match:
        cpd = int(match.group(1))
        luminance = float(match.group(2))
        contrast = float(match.group(3))
        return luminance, cpd, contrast
    return None, None, None

# 新しい列を一度に追加
final_df[['mean_luminance', 'cpd', 'contrast']] = final_df['Image_Win2'].apply(
    lambda x: pd.Series(extract_parameters(x))
)

# 不要な行（パラメータを抽出できなかったもの）を削除
final_df.dropna(subset=['mean_luminance', 'cpd', 'contrast'], inplace=True)
# データ型を適切に変換
if not final_df.empty:
    final_df['cpd'] = final_df['cpd'].astype(int)

# 4. 条件ごとの平均値と標準誤差を算出
summary_df = final_df.groupby(['distance', 'diopter_diff', 'd1', 'mean_luminance', 'cpd', 'contrast'])['Score'].agg(['mean', 'sem']).reset_index()

# 指定された順序 `50-70, 81-150, 50-100, 60-150` でグラフのx軸を並べ替える
custom_order = ['50-70 cm', '81-150 cm', '50-100 cm', '60-150 cm']
try:
    summary_df['distance'] = pd.Categorical(summary_df['distance'], categories=custom_order, ordered=True)
    # 指定したカテゴリ順でソート
    summary_df = summary_df.sort_values('distance')
except ValueError:
    print("警告: CSVデータに 'custom_order' にない距離の組み合わせが含まれています。ソートはスキップされます。")

# 5. グラフの描画
# スタイルのためのユニークな値を取得
unique_distances = summary_df['distance'].unique()
unique_lums = sorted(final_df['mean_luminance'].dropna().unique())
unique_cpds = sorted(final_df['cpd'].dropna().unique())

# --- スタイル定義 ---
# 空間周波数ごとの色
cpd_colors = ['tab:blue', 'tab:green', 'tab:orange']
cpd_color_map = {cpd: cpd_colors[i % len(cpd_colors)] for i, cpd in enumerate(unique_cpds)}

# 平均輝度ごとのマーカー
lum_marker_map = {lum: marker for lum, marker in zip(unique_lums, ['o', 's', '^', 'D'])}

# --- グラフ作成 ---
if not unique_distances.size:
    print("描画するデータがありません。")
    exit()

# --- 凡例と全体ラベルの設定 ---
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=cpd_color_map.get(cpd), lw=2, label=f'{cpd} cpd') for cpd in unique_cpds]
legend_elements.extend([Line2D([0], [0], marker=lum_marker_map.get(lum), color='gray', label=f'{lum} nit', linestyle='None') for lum in unique_lums])

# 各距離の組み合わせについて個別のグラフを生成
for distance in unique_distances:
    # 1. FigureとAxesを新規作成
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 2. 対象のデータフレームをフィルタリング
    distance_df = summary_df[summary_df['distance'] == distance]

    # 3. データのプロット
    for lum in unique_lums:
        for cpd in unique_cpds:
            plot_df = distance_df[(distance_df['mean_luminance'] == lum) & (distance_df['cpd'] == cpd)].sort_values('contrast')
            if not plot_df.empty:
                ax.errorbar(plot_df['contrast'], plot_df['mean'], yerr=plot_df['sem'], 
                            marker=lum_marker_map.get(lum, 'o'), 
                            color=cpd_color_map.get(cpd, 'black'), 
                            linestyle='-', capsize=4)

    # 4. グラフの装飾
    ax.set_title(f'Visibility Score vs. Contrast\nDistance Combination: {distance}', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Michelson Contrast', fontsize=12)
    ax.set_ylabel('Average Score', fontsize=12)
    ax.set_ylim(0.5, 5.5)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.legend(handles=legend_elements, title="Parameters")

    # 5. 保存と表示
    plt.tight_layout()
    
    # ファイル名に距離を含める (例: '50-70 cm' -> '50-70_cm')
    safe_distance_str = str(distance).replace(' ', '_')
    output_filename = os.path.join(OUTPUT_DIR, f'contrast_vs_score_{safe_distance_str}.png')
    plt.savefig(output_filename)
    print(f"グラフを保存しました: {output_filename}")
    
    plt.show()