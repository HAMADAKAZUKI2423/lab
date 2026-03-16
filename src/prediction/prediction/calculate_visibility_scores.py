import os
import glob
import torch
import numpy as np
import cv2
import pandas as pd
import sys

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
    print(f"Using device: {device}")

    # モデルのロード
    model_name = "vismlp_norm"
    vismodel = load_vismodel(model_name, device, load_param=True)
    vismodel.eval()

    # データディレクトリの設定
    base_data_dir = os.path.join(PROJECT_ROOT, "data", "processed", "images", "pre-experiment-gabor")
    overlapped_dir = os.path.join(base_data_dir, "overlapped")
    fg_gabor_dir = os.path.join(base_data_dir, "fg_gabor")
    bg_noise_dir = os.path.join(base_data_dir, "bg_noise")
    
    # 記録用リスト
    results = []

    # overlappedディレクトリ内の各サブディレクトリを走査
    sub_dirs = [d for d in os.listdir(overlapped_dir) if os.path.isdir(os.path.join(overlapped_dir, d))]
    
    for sub_dir in sub_dirs:
        current_overlapped_path = os.path.join(overlapped_dir, sub_dir)
        # 画像ファイルを取得
        image_files = glob.glob(os.path.join(current_overlapped_path, "*.png"))
        print(f"Processing directory: {sub_dir} ({len(image_files)} files)")

        # FG, BGの検索用ベースパス
        # ここでは sub_dir (2260314_163430など) が一致すると仮定
        fg_base = os.path.join(fg_gabor_dir, sub_dir)
        bg_base = os.path.join(bg_noise_dir, sub_dir)

        for img_path in image_files:
            filename = os.path.basename(img_path)
            # 命名規則: overlap_FG-{fg_name}_BG-{bg_name}.png
            # "overlap_FG-" と ".png" を除いて "BG-" で分割
            try:
                parts = filename.replace("overlap_FG-", "").replace(".png", "").split("_BG-")
                if len(parts) != 2:
                    print(f"Skipping invalid filename: {filename}")
                    continue
                fg_name, bg_name = parts
            except Exception as e:
                print(f"Error parsing filename {filename}: {e}")
                continue

            # FG画像のパスを探索 (50cm, 60cm, 81cmなどのサブディレクトリ内を探す)
            fg_img_path = None
            for root, dirs, files in os.walk(fg_base):
                if f"{fg_name}.png" in files:
                    fg_img_path = os.path.join(root, f"{fg_name}.png")
                    break
            
            # BG画像のパスを探索
            bg_img_path = None
            for root, dirs, files in os.walk(bg_base):
                if f"{bg_name}.png" in files:
                    bg_img_path = os.path.join(root, f"{bg_name}.png")
                    break

            if not fg_img_path or not bg_img_path:
                print(f"Source files not found for {filename} (FG: {fg_img_path}, BG: {bg_img_path})")
                continue

            # 画像の読み込み
            bg_img = read_image(bg_img_path)
            bg_bgr = ensure_bgr(bg_img)
            
            fg_img = read_image(fg_img_path)
            fg_bgr = ensure_bgr(fg_img)
            
            blend_img = read_image(img_path)
            blend_bgr = ensure_bgr(blend_img)

            # アルファマップの生成（ガボールパッチ画像は背景がグレーなので、
            # 視認性モデルの要件に合わせて、輝度変化がある場所をマスクとするなどの処理が必要な場合があるが、
            # ここではモデルに画像をそのまま渡し、アルファは0.5（典型的な重ね合わせ）または
            # 重ね合わせ画像から逆算する形式になる。
            # 今回は、bg, fg, blendが既にあるので、set_inputs_tg_ref_blended を使用する。
            
            # マスクの作成 (ここでは全域 1.0 または ガボールパッチの範囲)
            mask_np = np.ones(bg_bgr.shape[:2], dtype=np.float32)

            # テンソル変換
            bg_tensor = np_to_tensor(bg_bgr, device)
            fg_tensor = np_to_tensor(fg_bgr, device)
            blend_tensor = np_to_tensor(blend_bgr, device)
            mask_tensor = np_to_tensor(mask_np[..., None], device)

            with torch.no_grad():
                # vismlp_norm は target_type="content" を期待
                # tg = fg, ref = bg としてセット
                vismodel.set_inputs_tg_ref_blended(
                    fg_tensor,
                    bg_tensor,
                    blend_tensor,
                    mask_tensor
                )
                vismodel.compute_weights()
                vismodel.compute_visibility_wo_weight()

            # 視認性マップの取得
            vismap = vismodel.norm_vismap # [0, 1] に正規化されたマップ
            avg_score = float(vismap.mean().cpu().numpy())

            results.append({
                "directory": sub_dir,
                "filename": filename,
                "fg_name": fg_name,
                "bg_name": bg_name,
                "visibility_score": avg_score
            })
            print(f"Score for {filename}: {avg_score:.4f}")

    # 結果の保存
    if results:
        df = pd.DataFrame(results)
        output_dir = os.path.join(PROJECT_ROOT, "results", "tables", "pre-experiment-gabor")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "visibility_scores.csv")
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
