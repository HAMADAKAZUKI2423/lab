import os
import glob
import shutil
import random
import numpy as np
import cv2
from sklearn.cluster import KMeans

# ==========================================
# 設定エリア
# ==========================================
# パス設定 (相対パス)
script_dir = os.path.dirname(os.path.abspath(__file__))
lab_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

# 入力画像のフォルダパス
INPUT_DIR = os.path.join(lab_root, "data", "raw", "images", "McGill Calibrated Color Image Database", "Textures", "Textures")

# 出力先のフォルダパス (BGとFG)
OUTPUT_BG_DIR = os.path.join(lab_root, "data", "processed", "images", "pre-experiment-image", "bg_imgs")
OUTPUT_FG_DIR = os.path.join(lab_root, "data", "processed", "images", "pre-experiment-image", "fg_imgs")

# クラスタ数（選択する画像の枚数）
N_CLUSTERS = 5

# 対象とする画像の拡張子
EXTENSIONS = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif']
# ==========================================

def main():
    # 1. 画像ファイルパスの取得
    image_paths = []
    for ext in EXTENSIONS:
        # 指定フォルダ直下のファイルを探す
        image_paths.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    
    if not image_paths:
        print(f"エラー: 指定されたフォルダ '{INPUT_DIR}' に画像ファイルが見つかりませんでした。")
        return

    print(f"画像ファイル数: {len(image_paths)}")

    # 2. 各画像の平均輝度を計算
    print("平均輝度を計算中...")
    data = [] # 辞書 {'path': ..., 'lum': ...} のリスト
    luminances = []

    for path in image_paths:
        try:
            # OpenCVで画像を読み込む
            img = cv2.imread(path)
            if img is None:
                print(f"警告: ファイル '{os.path.basename(path)}' を読み込めませんでした。")
                continue

            # グレースケールに変換
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_lum = np.mean(gray_img)
            
            data.append({'path': path, 'lum': mean_lum})
            luminances.append(mean_lum)
        except Exception as e:
            print(f"警告: ファイル '{os.path.basename(path)}' の読み込みに失敗しました: {e}")

    if not data:
        print("有効な画像がありませんでした。")
        return

    # 画像数がクラスタ数より少ない場合の安全策
    n_clusters = min(N_CLUSTERS, len(data))
    if n_clusters < N_CLUSTERS:
        print(f"警告: 画像数({len(data)})が指定された枚数({N_CLUSTERS})より少ないため、{n_clusters}枚のみ選択します。")

    # 3. k-meansクラスタリングを実行
    # scikit-learnのKMeansは2次元配列を期待するためreshapeする
    X = np.array(luminances).reshape(-1, 1)
    
    print(f"{n_clusters}段階の輝度レベルに分類中...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # 結果をクラスタごとにまとめる
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(data[i])

    # 4. 出力フォルダの準備
    for d in [OUTPUT_BG_DIR, OUTPUT_FG_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)
            
    print(f"出力フォルダ(BG): {OUTPUT_BG_DIR}")
    print(f"出力フォルダ(FG): {OUTPUT_FG_DIR}")

    # 5. 各クラスタからランダムに2枚選んでコピー
    # クラスタの中心輝度でソートして、暗い順に番号を振る
    cluster_centers = kmeans.cluster_centers_.flatten()
    sorted_indices = np.argsort(cluster_centers)

    print("\n--- 選択された画像 ---")
    for rank, cluster_idx in enumerate(sorted_indices):
        cluster_items = clusters.get(cluster_idx, [])
        if not cluster_items: continue

        # ランダムに2枚選択 (画像が足りない場合はあるだけ選択)
        n_select = min(2, len(cluster_items))
        selected_items = random.sample(cluster_items, n_select)
        
        for i, item in enumerate(selected_items):
            # 1枚目をBG、2枚目をFGに振り分け
            # ファイル名の先頭をランク番号(1-5)にすることで、実験プログラム側でペアリング可能にする
            if i == 0:
                target_dir = OUTPUT_BG_DIR
                prefix = "bg"
            else:
                target_dir = OUTPUT_FG_DIR
                prefix = "fg"

            new_name = f"{rank+1}_{prefix}_lum{item['lum']:.1f}_{os.path.basename(item['path'])}"
            dst_path = os.path.join(target_dir, new_name)
            
            shutil.copy2(item['path'], dst_path)
            print(f"Class {rank+1} ({prefix}) (Center~{cluster_centers[cluster_idx]:.1f}): {new_name}")

if __name__ == "__main__":
    main()