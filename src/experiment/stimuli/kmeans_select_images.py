# py .\src\experiment\stimuli\kmeans_select_images.py
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

# クラスタ設定
N_TEX_CLUSTERS = 3  # テクスチャ（ラプラシアン）のクラス数
N_LUM_CLUSTERS = 3  # 各テクスチャクラス内の輝度クラス数
# 合計クラス数 = 3 * 3 = 9

# 対象とする画像の拡張子
EXTENSIONS = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif']
# ==========================================

def main():
    # 1. 画像ファイルパスの取得
    def compute_laplacian_mad(img_gray, levels=3):
        """
        ラプラシアンピラミッドの各レベルにおける平均絶対偏差(MAD)を計算する
        """
        features = []
        current_img = img_gray.astype(np.float32)
        
        for _ in range(levels):
            rows, cols = current_img.shape
            # 画像サイズが小さすぎる場合は中断
            if rows < 2 or cols < 2:
                break
                
            down = cv2.pyrDown(current_img)
            up = cv2.pyrUp(down, dstsize=(cols, rows))
            lap = current_img - up
            features.append(np.mean(np.abs(lap)))
            current_img = down
        
        # 残差（低周波成分）のMADも追加（あるいは単に平均輝度情報として扱うかだが、ここではMADを一貫して使用）
        features.append(np.mean(np.abs(current_img - np.mean(current_img))))
        return np.array(features)

    image_paths = []
    for ext in EXTENSIONS:
        # 指定フォルダ直下のファイルを探す
        image_paths.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    
    if not image_paths:
        print(f"エラー: 指定されたフォルダ '{INPUT_DIR}' に画像ファイルが見つかりませんでした。")
        return

    print(f"画像ファイル数: {len(image_paths)}")

    # 2. 特徴量（ラプラシアンピラミッドのMAD）を計算
    print("特徴量(ラプラシアンピラミッドMAD)を計算中...")
    data = [] # 辞書 {'path': ..., 'lum': ..., 'features': ...} のリスト
    features_list = []

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
            # ラプラシアンピラミッド特徴量の計算 (レベル数は適宜調整、ここでは3+残差)
            feats = compute_laplacian_mad(gray_img, levels=3)
            
            data.append({'path': path, 'lum': mean_lum, 'features': feats})
            features_list.append(feats)
        except Exception as e:
            print(f"警告: ファイル '{os.path.basename(path)}' の読み込みに失敗しました: {e}")

    if not data:
        print("有効な画像がありませんでした。")
        return

    # 3. 階層的クラスタリングを実行 (1 + 3回)
    
    # --- Step 1: テクスチャ（ラプラシアン特徴量）で3クラスに分類 ---
    print(f"Step 1: テクスチャに基づいて {N_TEX_CLUSTERS} クラスに分類中...")
    X_tex = np.array(features_list)
    kmeans_tex = KMeans(n_clusters=N_TEX_CLUSTERS, random_state=42, n_init=10)
    tex_labels = kmeans_tex.fit_predict(X_tex)

    # テクスチャの強さ（エネルギー総和）でソート
    tex_centers = kmeans_tex.cluster_centers_
    tex_magnitudes = np.sum(tex_centers, axis=1)
    tex_sorted_indices = np.argsort(tex_magnitudes) # [小(平坦) -> 大(複雑)]
    tex_rank_map = {label: rank for rank, label in enumerate(tex_sorted_indices)}

    # データをテクスチャクラスごとに分ける
    tex_clusters = {rank: [] for rank in range(N_TEX_CLUSTERS)}
    for i, item in enumerate(data):
        rank = tex_rank_map[tex_labels[i]]
        tex_clusters[rank].append(item)

    # --- Step 2: 各テクスチャクラス内で輝度で3クラスに分類 ---
    final_groups = [] # (tex_rank, lum_rank, items) のリスト
    
    for t_rank in range(N_TEX_CLUSTERS):
        items = tex_clusters[t_rank]
        print(f"Step 2: Texture Class {t_rank+1} (n={len(items)}) を輝度で {N_LUM_CLUSTERS} 分割中...")
        
        if len(items) < N_LUM_CLUSTERS:
            print(f"  警告: 画像数が足りないためスキップします。")
            continue

        # 輝度データの準備
        X_lum = np.array([item['lum'] for item in items]).reshape(-1, 1)
        
        kmeans_lum = KMeans(n_clusters=N_LUM_CLUSTERS, random_state=42, n_init=10)
        lum_labels = kmeans_lum.fit_predict(X_lum)
        
        # 輝度でソート (暗 -> 明)
        lum_centers = kmeans_lum.cluster_centers_.flatten()
        lum_sorted_indices = np.argsort(lum_centers)
        lum_rank_map = {label: rank for rank, label in enumerate(lum_sorted_indices)}
        
        # グループ化して保存
        lum_subclusters = {rank: [] for rank in range(N_LUM_CLUSTERS)}
        for i, item in enumerate(items):
            l_rank = lum_rank_map[lum_labels[i]]
            lum_subclusters[l_rank].append(item)
            
        for l_rank in range(N_LUM_CLUSTERS):
            final_groups.append({
                'tex_rank': t_rank,
                'lum_rank': l_rank,
                'items': lum_subclusters[l_rank],
                'avg_lum': lum_centers[lum_sorted_indices[l_rank]]
            })

    # 4. 出力フォルダの準備
    for d in [OUTPUT_BG_DIR, OUTPUT_FG_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)
            
    print(f"出力フォルダ(BG): {OUTPUT_BG_DIR}")
    print(f"出力フォルダ(FG): {OUTPUT_FG_DIR}")

    # 5. 各グループから画像を抽出して保存
    # final_groups は既に [Tex1-Lum1, Tex1-Lum2, ..., Tex3-Lum3] の順で追加されているはずだが念のためソート
    # ソートキー: テクスチャランク優先、次に輝度ランク
    final_groups.sort(key=lambda x: (x['tex_rank'], x['lum_rank']))

    print("\n--- 選択された画像 ---")
    for global_rank, group in enumerate(final_groups):
        cluster_items = group['items']
        tex_r = group['tex_rank'] + 1
        lum_r = group['lum_rank'] + 1
        lum_label = ["L", "M", "H"][group['lum_rank']]

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

            # ファイル名: {通し番号1-9}_Tex{1-3}_Lum{L/M/H}_{元ファイル名}
            new_name = f"{global_rank+1}_{prefix}_Tex{tex_r}_Lum{lum_label}_{os.path.basename(item['path'])}"
            dst_path = os.path.join(target_dir, new_name)
            
            shutil.copy2(item['path'], dst_path)
            print(f"Class {global_rank+1} [Tex{tex_r}-Lum{lum_label}] ({prefix}): {new_name}")

if __name__ == "__main__":
    main()