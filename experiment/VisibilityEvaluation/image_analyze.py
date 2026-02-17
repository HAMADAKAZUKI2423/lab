import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import glob

# 解析対象のフォルダ
 # main.py の出力フォルダ (./main_results) や、その中の日付フォルダなどを指定します
TARGET_DIR = r"C:\Users\HamaKazu\Desktop\GradSchool\lab\experiment\VisibilityEvaluation\main_results"

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

# 3. 前景画像(Image_Win2)のファイル名をカテゴリとして使用
final_df['fg_image'] = final_df['Image_Win2']

# 4. 条件ごとの平均値と標準誤差を算出
summary_df = final_df.groupby(['distance', 'diopter_diff', 'd1', 'fg_image'])['Score'].agg(['mean', 'sem']).reset_index()

# 指定された順序 `50-70, 81-150, 50-100, 60-150` でグラフのx軸を並べ替える
custom_order = ['50-70 cm', '81-150 cm', '50-100 cm', '60-150 cm']
summary_df['distance'] = pd.Categorical(summary_df['distance'], categories=custom_order, ordered=True)
# 指定したカテゴリ順でソート
summary_df = summary_df.sort_values('distance')

# グラフ描画用にピボット（行：前景画像、列：距離ラベル、値：平均スコア）
pivot_df = summary_df.pivot_table(index='fg_image', columns='distance', values='mean')
error_df = summary_df.pivot_table(index='fg_image', columns='distance', values='sem')

# 4. グラフの描画
# 点と線グラフで、距離ごとに色分けして表示
fig, ax = plt.subplots(figsize=(15, 7)) # 横幅を広げてラベルの重なりを防ぐ

# 前景画像名でソートしてx軸の順序を固定
pivot_df.sort_index(inplace=True)
error_df.sort_index(inplace=True)

# 各距離の組み合わせ（ピボットテーブルの各列）についてプロット
for distance_condition in pivot_df.columns:
    ax.errorbar(
        pivot_df.index,
        pivot_df[distance_condition],
        yerr=error_df[distance_condition],
        marker='o',
        linestyle='-',
        label=distance_condition,
        capsize=4
    )

# 軸ラベルとタイトルの設定
ax.set_xlabel('Foreground Image', fontsize=12)
ax.set_ylabel('Average Score', fontsize=12)
ax.set_title('Average Score by Foreground Image and Distance', fontsize=14)
plt.xticks(rotation=45, ha='right') # 横軸ラベルを回転させて見やすくする
ax.set_ylim(0, 5) # スコアの範囲に合わせて調整
ax.legend(title='Distance Combination')
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('image_vs_score_line.png')
plt.show()