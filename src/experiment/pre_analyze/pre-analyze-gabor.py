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

# FGパラメータ抽出
final_df[['fg_lum', 'fg_cpd', 'fg_contrast']] = final_df['Image_Win2'].apply(
    lambda x: pd.Series(extract_parameters(x))
)

# BGパラメータ抽出
final_df[['bg_lum', 'bg_cpd', 'bg_contrast']] = final_df['Image_Win1'].apply(
    lambda x: pd.Series(extract_parameters(x))
)

# 不要な行を削除
final_df.dropna(subset=['fg_lum', 'fg_cpd', 'fg_contrast', 'bg_lum'], inplace=True)
# データ型を適切に変換
if not final_df.empty:
    final_df['fg_cpd'] = final_df['fg_cpd'].astype(int)

# 4. 条件ごとの平均値と標準誤差を算出
# グルーピングキーを変更: 距離, FG輝度, BG輝度, FG周波数, FGコントラスト
summary_df = final_df.groupby(['distance', 'fg_lum', 'bg_lum', 'fg_cpd', 'fg_contrast'])['Score'].agg(['mean', 'sem']).reset_index()

# 指定された順序 `50-70, 81-150, 50-100, 60-150` でグラフのx軸を並べ替える
custom_order = ['50-70 cm', '81-150 cm', '50-100 cm', '60-150 cm']
try:
    summary_df['distance'] = pd.Categorical(summary_df['distance'], categories=custom_order, ordered=True)
    # 指定したカテゴリ順でソート
    summary_df = summary_df.sort_values('distance')
except ValueError:
    print("警告: CSVデータに 'custom_order' にない距離の組み合わせが含まれています。ソートはスキップされます。")

# 5. グラフの描画
unique_distances = summary_df['distance'].unique()

if not unique_distances.size:
    print("描画するデータがありません。")
    exit()

# 8本の線のための条件リストを作成 (FG Lum x BG Lum x FG CPD)
# データに含まれる全組み合わせを取得して色を割り当てる
conditions = summary_df[['fg_lum', 'bg_lum', 'fg_cpd']].drop_duplicates().sort_values(by=['fg_lum', 'bg_lum', 'fg_cpd'])

# 色分けルールのための値の取得
unique_fg_lums = sorted(conditions['fg_lum'].unique())
unique_bg_lums = sorted(conditions['bg_lum'].unique())
unique_fg_cpds = sorted(conditions['fg_cpd'].unique())

max_fg_lum = max(unique_fg_lums) if unique_fg_lums else 0
max_bg_lum = max(unique_bg_lums) if unique_bg_lums else 0
max_fg_cpd = max(unique_fg_cpds) if unique_fg_cpds else 0

condition_styles = {}

for idx, row in enumerate(conditions.itertuples(index=False)):
    # (fg_lum, bg_lum, fg_cpd) -> (color, label)
    key = (row.fg_lum, row.bg_lum, row.fg_cpd)
    label = f"FG:{row.fg_lum}nit, BG:{row.bg_lum}nit, {row.fg_cpd}cpd"
    
    # 色の決定: 背景輝度が高い(赤系)/低い(青系)、FG輝度が高い(濃い)/低い(薄い)
    is_high_bg = (row.bg_lum == max_bg_lum)
    is_high_fg = (row.fg_lum == max_fg_lum)
    is_high_cpd = (row.fg_cpd == max_fg_cpd)
    
    if is_high_bg:
        color = 'red' if is_high_fg else 'orange'
    else:
        color = 'blue' if is_high_fg else 'skyblue'

    # 線の種類: 周波数が高い(実線)/低い(点線)
    linestyle = '-' if is_high_cpd else '--'
    
    condition_styles[key] = {'color': color, 'label': label, 'linestyle': linestyle}

# 各距離の組み合わせについて個別のグラフを生成
for distance in unique_distances:
    # 1. FigureとAxesを新規作成
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 2. 対象のデータフレームをフィルタリング
    distance_df = summary_df[summary_df['distance'] == distance]

    # 3. データのプロット
    # 定義した条件スタイルに基づいてループ
    for key, style in condition_styles.items():
        fg_l, bg_l, cpd = key
        
        # 条件に合致するデータを抽出
        plot_df = distance_df[
            (distance_df['fg_lum'] == fg_l) & 
            (distance_df['bg_lum'] == bg_l) & 
            (distance_df['fg_cpd'] == cpd)
        ].sort_values('fg_contrast')
        
        if not plot_df.empty:
            ax.errorbar(plot_df['fg_contrast'], plot_df['mean'], yerr=plot_df['sem'], 
                        label=style['label'],
                        color=style['color'], 
                        marker='o',
                        linestyle=style['linestyle'], capsize=4)

    # 4. グラフの装飾
    ax.set_title(f'Visibility Score vs. Contrast\nDistance Combination: {distance}', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Michelson Contrast', fontsize=12)
    ax.set_ylabel('Average Score', fontsize=12)
    ax.set_ylim(0.5, 5.5)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.legend(title="Conditions (FG, BG, CPD)", bbox_to_anchor=(1.05, 1), loc='upper left')

    # 5. 保存と表示
    plt.tight_layout()
    
    # ファイル名に距離を含める (例: '50-70 cm' -> '50-70_cm')
    safe_distance_str = str(distance).replace(' ', '_')
    output_filename = os.path.join(OUTPUT_DIR, f'contrast_vs_score_{safe_distance_str}.png')
    plt.savefig(output_filename)
    print(f"グラフを保存しました: {output_filename}")
    
    plt.show()

# --- 追加: AR拡張コントラスト vs スコア (平均 + 標準偏差) ---
print("AR拡張コントラストの分析を実行中...")

# AR拡張コントラストの計算
# c_AR = (Y_max - Y_min) / (Y_max + Y_min + 2*Y_BG)
# ここで、Y_max = fg_lum * (1 + fg_contrast), Y_min = fg_lum * (1 - fg_contrast)
# これを代入すると: c_AR = (fg_lum * fg_contrast) / (fg_lum + bg_lum)
final_df['ar_contrast'] = (final_df['fg_lum'] * final_df['fg_contrast']) / (final_df['fg_lum'] + final_df['bg_lum'])

# グラフ描画
fig_ar, ax_ar = plt.subplots(figsize=(10, 7))

# FG輝度とBG輝度の組み合わせごとに集計してプロット
grouped_ar = final_df.groupby(['fg_lum', 'bg_lum', 'ar_contrast'])['Score'].agg(['mean', 'std']).reset_index()
grouped_ar['std'] = grouped_ar['std'].fillna(0)

# 組み合わせのユニークなリストを作成
conditions_ar = grouped_ar[['fg_lum', 'bg_lum']].drop_duplicates().sort_values(by=['fg_lum', 'bg_lum'])

for idx, (fg_l, bg_l) in enumerate(conditions_ar.itertuples(index=False, name=None)):
    subset = grouped_ar[(grouped_ar['fg_lum'] == fg_l) & (grouped_ar['bg_lum'] == bg_l)]
    label = f"FG:{fg_l}nit, BG:{bg_l}nit"
    # 線なし (fmt='o') でプロット
    ax_ar.errorbar(subset['ar_contrast'], subset['mean'], yerr=subset['std'],
                 fmt='o', capsize=4, label=label, alpha=0.8)

ax_ar.set_title('Visibility Score vs. AR Extended Contrast', fontsize=14)
ax_ar.set_xlabel(r'AR Extended Contrast ($c_{AR}$)', fontsize=12)
ax_ar.set_ylabel('Average Score', fontsize=12)
ax_ar.set_ylim(0.5, 5.5)
ax_ar.set_xlim(0, 1.05)
ax_ar.grid(True, which='both', linestyle='--', linewidth=0.5)
ax_ar.legend(loc='upper left')

output_filename_ar = os.path.join(OUTPUT_DIR, 'ar_contrast_vs_score_mean_std.png')
plt.savefig(output_filename_ar)
print(f"AR拡張コントラストのグラフを保存しました: {output_filename_ar}")
plt.show()

# --- 追加: 輝度比 (FG/BG) vs スコア ---
print("輝度比 (FG/BG) の分析を実行中...")

# 輝度比の計算
final_df['lum_ratio'] = final_df['fg_lum'] / final_df['bg_lum']

# 輝度比ごとの平均と標準偏差を算出
ratio_summary_df = final_df.groupby('lum_ratio')['Score'].agg(['mean', 'std']).reset_index()
ratio_summary_df['std'] = ratio_summary_df['std'].fillna(0)
ratio_summary_df = ratio_summary_df.sort_values('lum_ratio')

# グラフ描画
fig_ratio, ax_ratio = plt.subplots(figsize=(10, 7))

ax_ratio.errorbar(ratio_summary_df['lum_ratio'], ratio_summary_df['mean'], yerr=ratio_summary_df['std'],
             fmt='-o', capsize=4, color='green', label='Mean Score ± STD')

ax_ratio.set_title('Visibility Score vs. Luminance Ratio (FG/BG)', fontsize=14)
ax_ratio.set_xlabel(r'Luminance Ratio ($Y_{FG}/Y_{BG}$)', fontsize=12)
ax_ratio.set_ylabel('Average Score', fontsize=12)
ax_ratio.set_ylim(0.5, 5.5)
ax_ratio.set_xscale('log')
ax_ratio.grid(True, which='both', linestyle='--', linewidth=0.5)
ax_ratio.legend(loc='upper left')

output_filename_ratio = os.path.join(OUTPUT_DIR, 'lum_ratio_vs_score_mean_std.png')
plt.savefig(output_filename_ratio)
print(f"輝度比のグラフを保存しました: {output_filename_ratio}")
plt.show()