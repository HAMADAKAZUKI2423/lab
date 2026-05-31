# py .\src\experiment\pre-analyze\pre-analyze-matching-ar.py
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

parser = argparse.ArgumentParser(description='Analyze defocus matching experiment results with AR contrast.')
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
required_columns = ["Condition", "Ocularity", "Ref_Contrast", "Matched_Contrast", "Orientation"]
for col in required_columns:
    if col not in final_df.columns:
        print(f"Error: 必要な列 '{col}' がCSVファイルに見つかりません。")
        exit()

# --- 記録された前景単体コントラストを合成後のAR拡張コントラストに変換 ---
L_fg = 35.0
L_bg = 15.0
final_df['Matched_Contrast_AR'] = final_df['Matched_Contrast'] * (L_fg / (L_fg + L_bg))

print("\n--- Generating bar charts (AR Extended Contrast) ---")
        
# 描画用の設定
sns.set_theme(style="whitegrid")
condition_order = ["Single plane", "Single plane + defocus simulation", "Dual plane", "Dual plane flat"]

# ラベルの揺れ(大文字・小文字)を統一
final_df['Ocularity'] = final_df['Ocularity'].str.lower()

# データに存在する条件のみを順に抽出 (過去の monocular にも対応)
ocularity_order = [oc for oc in ["left", "right", "binocular", "monocular"] if oc in final_df['Ocularity'].unique()]

unique_ref_contrasts = sorted(final_df['Ref_Contrast'].dropna().unique(), reverse=True)
unique_orientations = sorted(final_df['Orientation'].dropna().unique())

for ref_c in unique_ref_contrasts:
    for ori in unique_orientations:
        plot_df = final_df[(final_df['Ref_Contrast'] == ref_c) & (final_df['Orientation'] == ori)]
        if plot_df.empty:
            continue
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 棒グラフと95%信頼区間のエラーバーを描画
        sns.barplot(
            data=plot_df, 
            x='Condition', 
            y='Matched_Contrast_AR', 
            hue='Ocularity',
            order=condition_order,
            hue_order=ocularity_order,
            errorbar=('ci', 95), 
            capsize=0.1, 
            err_kws={'linewidth': 1.5},
            ax=ax
        )
        
        # Ref_Contrastの値の高さに横方向の点線を引く (リファレンスは単体表示のため値がそのまま見た目のコントラスト)
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
                            m = subset['Matched_Contrast_AR'].mean()
                            d = subset['Matched_Contrast_AR'].std() 
                            
                            x = p.get_x() + p.get_width() / 2
                            y = 0.1  # バーの足元付近
                            
                            ax.text(x, y, f"m={m:.2f}\nd={d:.2f}", ha='center', va='bottom', color='black', fontsize=10,
                                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                    patch_idx += 1
    
        # グラフの見た目調整
        ax.set_title(f'Matched AR Contrast by Condition and Ocularity (Ref Contrast: {ref_c}, Ori: {ori}°)', fontsize=14)
        ax.set_ylabel('Matched Contrast (AR Extended)', fontsize=12)
        ax.set_xlabel('Condition', fontsize=12)
        ax.set_yscale('log') # 縦軸をログスケールに設定
        
        ax.set_ylim(0.1, 1.0) 
        ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        
        # x軸のラベルをそのまま表示（回転なし）
        labels = ax.get_xticklabels()
        ax.set_xticklabels(labels)
        
        # 凡例の設定
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, bbox_to_anchor=(0, 0.25), loc='upper left', borderaxespad=0.5)
        
        plt.tight_layout()
        
        # 画像として保存 (ファイル名に _ar を追加)
        filename = f'matched_ar_contrast_ref_{ref_c}_ori_{int(ori)}.png'
        save_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(save_path, dpi=300)
        print(f"グラフ保存: {filename}")
        plt.close(fig)

# --- 追加: エンハンスコントラストの計算とグラフ出力 ---
# 背景ノイズのコントラスト（CSVに含まれていない場合は1.0と仮定）
C_bg = 1.0

# エンハンスコントラストの計算式:
# Y_max = L_fg * (1 + C_fg) + L_bg * (1 + C_bg)
# Y_min = L_fg * (1 - C_fg) + L_bg * (1 - C_bg)
# C_enhanced = (Y_max - Y_min) / (Y_max + Y_min) = (L_fg * C_fg + L_bg * C_bg) / (L_fg + L_bg)
final_df['Matched_Contrast_Enhanced'] = (final_df['Matched_Contrast'] * L_fg + C_bg * L_bg) / (L_fg + L_bg)

print("\n--- Generating bar charts (Enhanced Contrast) ---")
for ref_c in unique_ref_contrasts:
    for ori in unique_orientations:
        plot_df = final_df[(final_df['Ref_Contrast'] == ref_c) & (final_df['Orientation'] == ori)]
        if plot_df.empty:
            continue
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.barplot(
            data=plot_df, 
            x='Condition', 
            y='Matched_Contrast_Enhanced', 
            hue='Ocularity',
            order=condition_order,
            hue_order=ocularity_order,
            errorbar=('ci', 95), 
            capsize=0.1, 
            err_kws={'linewidth': 1.5},
            ax=ax
        )
        
        ax.axhline(y=ref_c, color='red', linestyle='--', linewidth=2, label=f'Ref Contrast ({ref_c})')
        
        patch_idx = 0
        for oc in ocularity_order:
            for cond in condition_order:
                if patch_idx < len(ax.patches):
                    p = ax.patches[patch_idx]
                    height = p.get_height()
                    if pd.notna(height) and height > 0:
                        subset = plot_df[(plot_df['Ocularity'] == oc) & (plot_df['Condition'] == cond)]
                        if not subset.empty:
                            m = subset['Matched_Contrast_Enhanced'].mean()
                            d = subset['Matched_Contrast_Enhanced'].std() 
                            
                            x = p.get_x() + p.get_width() / 2
                            y = 0.15  # バーの足元付近
                            
                            ax.text(x, y, f"m={m:.2f}\nd={d:.2f}", ha='center', va='bottom', color='black', fontsize=10,
                                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                    patch_idx += 1
    
        ax.set_title(f'Matched Enhanced Contrast by Condition and Ocularity (Ref Contrast: {ref_c}, Ori: {ori}°)', fontsize=14)
        ax.set_ylabel('Matched Contrast (Enhanced)', fontsize=12)
        ax.set_xlabel('Condition', fontsize=12)
        ax.set_yscale('log')
        
        ax.set_ylim(0.1, 1.0) 
        ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        
        labels = ax.get_xticklabels()
        ax.set_xticklabels(labels)
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, bbox_to_anchor=(0, 0.25), loc='upper left', borderaxespad=0.5)
        
        plt.tight_layout()
        
        filename = f'matched_enhanced_contrast_ref_{ref_c}_ori_{int(ori)}.png'
        save_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(save_path, dpi=300)
        print(f"グラフ保存: {filename}")
        plt.close(fig)

print("解析完了")