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
INPUT_DIR = os.path.join(lab_root, "data", "raw", "images")

# 出力先のフォルダパス (BGとFG)
OUTPUT_BG_DIR = os.path.join(lab_root, "data", "processed", "images", "pre-experiment-image", "bg_imgs")
OUTPUT_FG_DIR = os.path.join(lab_root, "data", "processed", "images", "pre-experiment-image", "fg_imgs")

# クラスタ設定
N_TEX_CLUSTERS = 5  # テクスチャ（ラプラシアン）のクラス数
N_SELECT_PER_TEX_CLUSTER = 1 # 各テクスチャクラスタから選択する前景・背景画像の数

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
        # フォルダ内のすべてのサブフォルダから画像ファイルを探す（再帰的）
        image_paths.extend(glob.glob(os.path.join(INPUT_DIR, "**", ext), recursive=True))
    
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
    
    # --- Step 1: テクスチャ（ラプラシアン特徴量）で5クラスに分類 ---
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

    # 4. 出力フォルダの準備
    for d in [OUTPUT_BG_DIR, OUTPUT_FG_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)
            
    print(f"出力フォルダ(BG): {OUTPUT_BG_DIR}")
    print(f"出力フォルダ(FG): {OUTPUT_FG_DIR}")

    # 5. 各テクスチャクラスタから画像をランダムに抽出して保存
    print("\n--- 選択された画像 ---")
    global_img_idx = 0

    # テクスチャランクでループ (t_rankは 0:平坦 -> 4:複雑)
    for t_rank in range(N_TEX_CLUSTERS):
        items_in_cluster = tex_clusters[t_rank]

        num_required = N_SELECT_PER_TEX_CLUSTER * 2
        if len(items_in_cluster) < num_required:
            print(f"警告: Texture Class {t_rank+1} (n={len(items_in_cluster)}) の画像数が足りないためスキップします。 ({num_required}枚以上必要)")
            continue

        # 輝度を考慮せず、クラスタからランダムに画像を選択
        selected_items = random.sample(items_in_cluster, num_required)

        # 選択した画像を前景用と背景用に半分ずつ分ける
        fg_items = selected_items[:N_SELECT_PER_TEX_CLUSTER]
        bg_items = selected_items[N_SELECT_PER_TEX_CLUSTER:]

        # 前景画像をコピー
        for i, item in enumerate(fg_items):
            global_img_idx += 1
            tex_r = t_rank + 1
            prefix = "fg"
            original_path = item['path']
            original_basename = os.path.basename(original_path)

            # 元の画像をコピー
            new_name = f"{global_img_idx}_{prefix}_Tex{tex_r}_{original_basename}"
            dst_path = os.path.join(OUTPUT_FG_DIR, new_name)
            shutil.copy2(original_path, dst_path)
            print(f"Selected ({prefix.upper()}) [Tex{tex_r}]: {new_name}")

            # 輝度を半分にした画像を生成して保存
            try:
                img = cv2.imread(original_path)
                if img is None:
                    print(f"  警告: 輝度変更のため {original_basename} を読み込めませんでした。")
                    continue

                # HSV色空間に変換し、輝度(V)を半分にする
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                hsv[:, :, 2] //= 2
                darker_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                # 新しいファイル名を生成して保存
                darker_basename = f"dark_{original_basename}"
                darker_new_name = f"{global_img_idx}_{prefix}_Tex{tex_r}_{darker_basename}"
                darker_dst_path = os.path.join(OUTPUT_FG_DIR, darker_new_name)
                cv2.imwrite(darker_dst_path, darker_img)
                print(f"  -> Generated darker version: {darker_new_name}")
            except Exception as e:
                print(f"  エラー: 輝度変更処理中にエラーが発生しました ({original_basename}): {e}")

        # 背景画像をコピー
        for i, item in enumerate(bg_items):
            global_img_idx += 1
            tex_r = t_rank + 1
            prefix = "bg"

            new_name = f"{global_img_idx}_{prefix}_Tex{tex_r}_{os.path.basename(item['path'])}"
            dst_path = os.path.join(OUTPUT_BG_DIR, new_name)
            shutil.copy2(item['path'], dst_path)
            print(f"Selected ({prefix.upper()}) [Tex{tex_r}]: {new_name}")


if __name__ == "__main__":
    main()