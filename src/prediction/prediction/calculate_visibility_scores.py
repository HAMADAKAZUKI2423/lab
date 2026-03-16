import os
import glob
import torch
import numpy as np
import cv2
import pandas as pd
import sys
from datetime import datetime

# プロジェクトルートをパスに追加
PROJECT_ROOT = r"C:\Users\HamaKazu\Desktop\GradSchool\lab"
VIS_BLEND_PATH = os.path.join(PROJECT_ROOT, "visibility_blend_2025-main")
sys.path.append(VIS_BLEND_PATH)

from vismodel.utils import load_vismodel # type: ignore

def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return img

def ensure_bgr(img):
    if img.ndim == 2:
        return np.repeat(img[:, :, None], 3, axis=2)
    if img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)
    if img.shape[2] == 4:
        return img[:, :, :3]
    return img

def np_to_tensor(img, device):
    if img.ndim == 2:
        img = img[:, :, None]
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return tensor

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 視認性スコア計算開始 ---")
    print(f"デバイス: {device}")

    # モデルのロード
    model_name = "vismlp_norm"
    print(f"モデル '{model_name}' をロード中...")
    vismodel = load_vismodel(model_name, device, load_param=True)
    vismodel.eval()

    # データディレクトリの設定
    base_data_dir = os.path.join(PROJECT_ROOT, "data", "processed", "images", "pre-experiment-gabor")
    overlapped_dir = os.path.join(base_data_dir, "overlapped")
    fg_gabor_dir = os.path.join(base_data_dir, "fg_gabor")
    bg_noise_dir = os.path.join(base_data_dir, "bg_noise")
    
    # 出力先の設定
    output_dir = os.path.join(PROJECT_ROOT, "results", "tables", "pre-experiment-gabor", "visibility_score")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{timestamp}.csv")

    # overlappedディレクトリ内の全画像をリストアップ
    all_image_files = []
    sub_dirs = [d for d in os.listdir(overlapped_dir) if os.path.isdir(os.path.join(overlapped_dir, d))]
    for sub_dir in sub_dirs:
        current_path = os.path.join(overlapped_dir, sub_dir)
        files = glob.glob(os.path.join(current_path, "*.png"))
        for f in files:
            all_image_files.append((sub_dir, f))

    total_files = len(all_image_files)
    print(f"総処理対象ファイル数: {total_files}")
    print(f"結果保存先: {output_path}")
    print("-" * 50)

    results = []

    for i, (sub_dir, img_path) in enumerate(all_image_files, 1):
        filename = os.path.basename(img_path)
        
        try:
            # 命名規則: overlap_FG-{fg_name}_BG-{bg_name}.png
            parts = filename.replace("overlap_FG-", "").replace(".png", "").split("_BG-")
            if len(parts) != 2:
                print(f"[{i}/{total_files}] スキップ (不正なファイル名): {filename}")
                continue
            fg_name, bg_name = parts

            # FG画像のパスを探索
            fg_base = os.path.join(fg_gabor_dir, sub_dir)
            fg_img_path = None
            for root, dirs, files in os.walk(fg_base):
                if f"{fg_name}.png" in files:
                    fg_img_path = os.path.join(root, f"{fg_name}.png")
                    break
            
            # BG画像のパスを探索
            bg_base = os.path.join(bg_noise_dir, sub_dir)
            bg_img_path = None
            for root, dirs, files in os.walk(bg_base):
                if f"{bg_name}.png" in files:
                    bg_img_path = os.path.join(root, f"{bg_name}.png")
                    break

            if not fg_img_path or not bg_img_path:
                print(f"[{i}/{total_files}] 失敗 (元画像未検出): {filename}")
                continue

            # 画像の読み込みと推論
            bg_tensor = np_to_tensor(ensure_bgr(read_image(bg_img_path)), device)
            fg_tensor = np_to_tensor(ensure_bgr(read_image(fg_img_path)), device)
            blend_tensor = np_to_tensor(ensure_bgr(read_image(img_path)), device)
            mask_tensor = np_to_tensor(np.ones(bg_tensor.shape[2:], dtype=np.float32)[..., None], device)

            with torch.no_grad():
                vismodel.set_inputs_tg_ref_blended(fg_tensor, bg_tensor, blend_tensor, mask_tensor)
                vismodel.compute_weights()
                vismodel.compute_visibility_wo_weight()

            # スコア算出
            avg_score = float(vismodel.norm_vismap.mean().cpu().numpy())
            
            # 記録
            res_dict = {
                "directory": sub_dir,
                "filename": filename,
                "fg_name": fg_name,
                "bg_name": bg_name,
                "visibility_score": avg_score
            }
            results.append(res_dict)

            # 逐次保存（ヘッダーは初回のみ）
            pd.DataFrame([res_dict]).to_csv(output_path, mode='a', index=False, header=not os.path.exists(output_path))
            
            print(f"[{i}/{total_files}] 完了: {filename} -> スコア: {avg_score:.4f}")

        except Exception as e:
            print(f"[{i}/{total_files}] エラー発生 ({filename}): {e}")

    print("-" * 50)
    print(f"全工程が完了しました。結果は {output_path} に保存されています。")

if __name__ == "__main__":
    main()
