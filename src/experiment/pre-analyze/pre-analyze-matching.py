# py .\src\experiment\pre-analyze\pre-analyze-matching.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os
import glob
import argparse

# 解析対象のフォルダ
script_dir = os.path.dirname(os.path.abspath(__file__))
lab_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

# 入力データのベースディレクトリ
DATA_BASE_DIR = os.path.join(lab_root, "results", "tables", "pre-experiment-matching")

parser = argparse.ArgumentParser(description='Analyze defocus matching experiment results.')
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
OUTPUT_DIR = os.path.join(lab_root, "results", "figures", "pre-experiment-matching", target_folder_name)
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
    all_data.append(df)

if not all_data:
    print("No data to process.")
    exit()
    
final_df = pd.concat(all_data, ignore_index=True)

# 必要な列が揃っているか確認
required_columns = ["Condition", "Ocularity", "Ref_Contrast", "Matched_Contrast"]
for col in required_columns:
    if col not in final_df.columns:
        print(f"Error: 必要な列 '{col}' がCSVファイルに見つかりません。")
        exit()

print("\n--- Generating bar charts ---")
        
# 描画用の設定
sns.set_theme(style="whitegrid")
condition_order = ["Single plane", "Single plane + defocus simulation", "OST-AR"]
ocularity_order = ["monocular", "binocular"]

unique_ref_contrasts = sorted(final_df['Ref_Contrast'].dropna().unique(), reverse=True)

for ref_c in unique_ref_contrasts:
    plot_df = final_df[final_df['Ref_Contrast'] == ref_c]
    if plot_df.empty:
        continue
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 棒グラフと95%信頼区間のエラーバーを描画
    # errorbar='ci' はデフォルトで 95% 信頼区間になります
    sns.barplot(
        data=plot_df, 
        x='Condition', 
        y='Matched_Contrast', 
        hue='Ocularity',
        order=condition_order,
        hue_order=ocularity_order,
        errorbar=('ci', 95), 
        capsize=0.1, 
        err_kws={'linewidth': 1.5},
        ax=ax
    )
    
    # Ref_Contrastの値の高さに横方向の点線を引く
    ax.axhline(y=ref_c, color='red', linestyle='--', linewidth=2, label=f'Ref Contrast ({ref_c})')
    
    # 各バーの足元（内側）に平均(m)と分散/標準偏差(d)を記入
    patch_idx = 0
    for oc in ocularity_order:
        for cond in condition_order:
            if patch_idx < len(ax.patches):
                p = ax.patches[patch_idx]
                height = p.get_height()
                if pd.notna(height) and height > 0:
                    subset = plot_df[(plot_df['Ocularity'] == oc) & (plot_df['Condition'] == cond)]
                    if not subset.empty:
                        m = subset['Matched_Contrast'].mean()
                        # "d" は標準偏差 (Standard Deviation) と解釈して std() を使用しています。
                        # もし厳密な分散 (Variance) を表示したい場合は .std() を .var() に変更してください。
                        d = subset['Matched_Contrast'].std() 
                        
                        x = p.get_x() + p.get_width() / 2
                        y = 0.1  # バーの足元付近 (ログスケールに合わせて調整)
                        
                        ax.text(x, y, f"m={m:.2f}\nd={d:.2f}", ha='center', va='bottom', color='black', fontsize=10,
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                patch_idx += 1

    # グラフの見た目調整
    ax.set_title(f'Matched Contrast by Condition and Ocularity (Ref Contrast: {ref_c})', fontsize=14)
    ax.set_ylabel('Matched Contrast', fontsize=12)
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_yscale('log') # 縦軸をログスケールに設定
    ax.set_ylim(0.1, 1.0) # コントラストの最小値を0.1に固定
    
    # y軸の数字を指数表記ではなく通常の小数表記にし、目盛りを細かく表示する
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    
    # x軸のラベルをそのまま表示（回転なし）
    labels = ax.get_xticklabels()
    ax.set_xticklabels(labels)
    
    # 凡例の設定
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=0.5)
    
    plt.tight_layout()
    
    # 画像として保存
    filename = f'matched_contrast_ref_{ref_c}.png'
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    print(f"グラフ保存: {filename}")
    plt.close(fig)

print("解析完了")