import os
import glob
import torch
import numpy as np
import cv2
import pandas as pd
import sys
import torch.nn.functional as F
from datetime import datetime
from unittest.mock import MagicMock

# --- 外部依存関係の回避策 ---
sys.modules["pyiqa"] = MagicMock()
# ---------------------------

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

def center_crop(img, target_h, target_w):
    """画像を中央で指定サイズにクロップする"""
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    x1 = cx - target_w // 2
    y1 = cy - target_h // 2
    x2 = x1 + target_w
    y2 = y1 + target_h
    
    # 範囲外チェック（念のため）
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    return img[y1:y2, x1:x2]

def pad_to_multiple(tensor, multiple=32):
    """テンソルの H, W を指定の倍数にパディングする"""
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return tensor, h, w

def np_to_tensor(img, device):
    if img.ndim == 2:
        img = img[:, :, None]
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return tensor

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 視認性スコア計算開始 (Center Crop対応版) ---")
    print(f"デバイス: {device}")

    model_name = "vismlp_norm"
    print(f"モデル '{model_name}' をロード中...")
    vismodel = load_vismodel(model_name, device, load_param=True)
    vismodel.eval()

    base_data_dir = os.path.join(PROJECT_ROOT, "data", "processed", "images", "pre-experiment-gabor")
    overlapped_dir = os.path.join(base_data_dir, "overlapped")
    fg_gabor_dir = os.path.join(base_data_dir, "fg_gabor")
    bg_noise_dir = os.path.join(base_data_dir, "bg_noise")
    
    output_dir = os.path.join(PROJECT_ROOT, "results", "tables", "pre-experiment-gabor", "visibility_score")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{timestamp}.csv")

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

    for i, (sub_dir, img_path) in enumerate(all_image_files, 1):
        filename = os.path.basename(img_path)
        
        try:
            parts = filename.replace("overlap_FG-", "").replace(".png", "").split("_BG-")
            if len(parts) != 2:
                print(f"[{i}/{total_files}] スキップ: {filename}")
                continue
            fg_name, bg_name = parts

            fg_base = os.path.join(fg_gabor_dir, sub_dir)
            fg_img_path = None
            for root, dirs, files in os.walk(fg_base):
                if f"{fg_name}.png" in files:
                    fg_img_path = os.path.join(root, f"{fg_name}.png")
                    break
            
            bg_base = os.path.join(bg_noise_dir, sub_dir)
            bg_img_path = None
            for root, dirs, files in os.walk(bg_base):
                if f"{bg_name}.png" in files:
                    bg_img_path = os.path.join(root, f"{bg_name}.png")
                    break

            if not fg_img_path or not bg_img_path:
                print(f"[{i}/{total_files}] 未検出: {filename}")
                continue

            # 画像読み込み
            bg_img_raw = read_image(bg_img_path)
            fg_img_raw = read_image(fg_img_path)
            blend_img_raw = read_image(img_path)

            # OVL (Blend) のサイズを取得
            th, tw = blend_img_raw.shape[:2]

            # BG をクロップ
            bg_img_cropped = center_crop(bg_img_raw, th, tw)
            
            # FG もサイズが違う場合はクロップ (通常は一致しているはず)
            fg_img_cropped = fg_img_raw
            if fg_img_raw.shape[:2] != (th, tw):
                fg_img_cropped = center_crop(fg_img_raw, th, tw)

            # テンソル変換
            bg_tensor = np_to_tensor(ensure_bgr(bg_img_cropped), device)
            fg_tensor = np_to_tensor(ensure_bgr(fg_img_cropped), device)
            blend_tensor = np_to_tensor(ensure_bgr(blend_img_raw), device)
            mask_tensor = np_to_tensor(np.ones((th, tw), dtype=np.float32)[..., None], device)

            # --- パディング処理 (32の倍数) ---
            bg_tensor, orig_h, orig_w = pad_to_multiple(bg_tensor, 32)
            fg_tensor, _, _ = pad_to_multiple(fg_tensor, 32)
            blend_tensor, _, _ = pad_to_multiple(blend_tensor, 32)
            mask_tensor, _, _ = pad_to_multiple(mask_tensor, 32)

            with torch.no_grad():
                vismodel.set_inputs_tg_ref_blended(fg_tensor, bg_tensor, blend_tensor, mask_tensor)
                vismodel.compute_weights()
                vismodel.compute_visibility_wo_weight()

            vismap = vismodel.norm_vismap
            vismap_cropped = vismap[:, :, :orig_h, :orig_w]
            avg_score = float(vismap_cropped.mean().cpu().numpy())
            
            res_dict = {
                "directory": sub_dir,
                "filename": filename,
                "fg_name": fg_name,
                "bg_name": bg_name,
                "visibility_score": avg_score
            }
            
            pd.DataFrame([res_dict]).to_csv(output_path, mode='a', index=False, header=not os.path.exists(output_path))
            print(f"[{i}/{total_files}] 完了: {filename} -> {avg_score:.4f}")

        except Exception as e:
            print(f"[{i}/{total_files}] エラー ({filename}): {e}")

    print("-" * 50)
    print(f"完了。結果: {output_path}")

if __name__ == "__main__":
    main()
