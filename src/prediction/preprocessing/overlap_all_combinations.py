import cv2
import numpy as np
import os
import glob

def apply_gamma(img, gamma=2.2, decode=True):
    """
    ガンマ補正の適用/解除
    decode=True: sRGB (0-255) の非線形 -> Linear (0.0-1.0) の線形空間へ
    decode=False: Linear (0.0-1.0) -> sRGB (0-255) へエンコード
    """
    if decode:
        norm = img.astype(np.float32) / 255.0
        return np.power(norm, gamma)
    else:
        corrected = np.power(img, 1.0 / gamma)
        return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)

def main():
    # パスの構築
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lab_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    
    base_dir = os.path.join(lab_root, "data", "processed", "images", "pre-experiment-gabor")

    # 物理輝度の最大値設定 (キャリブレーション時の MAX_LUMINANCE を設定)
    FG_MAX_LUM = 100.0
    BG_MAX_LUM = 30.0

    # 処理する距離の組み合わせ
    DISTANCE_PAIRS = [
        (50, 70),
        (50, 100),
        (60, 150),
        (81, 150)
    ]

    total_count = 0

    for fg_dist, bg_dist in DISTANCE_PAIRS:
        fg_dir = os.path.join(base_dir, "fg_gabor", f"{fg_dist}cm")
        bg_dir = os.path.join(base_dir, "bg_noise", f"{bg_dist}cm")
        out_dir = os.path.join(base_dir, "overlapped", f"{fg_dist}-{bg_dist}")

        # フォルダの存在確認
        if not os.path.exists(fg_dir):
            print(f"警告: 前景フォルダが見つかりません、スキップします: {fg_dir}")
            continue
        if not os.path.exists(bg_dir):
            print(f"警告: 背景フォルダが見つかりません、スキップします: {bg_dir}")
            continue
            
        os.makedirs(out_dir, exist_ok=True)
        
        fg_files = glob.glob(os.path.join(fg_dir, "*.png"))
        bg_files = glob.glob(os.path.join(bg_dir, "*.png"))
        
        if not fg_files or not bg_files:
            print(f"警告: 処理対象の画像が見つかりませんでした。({fg_dist}cm - {bg_dist}cm)")
            continue

        print(f"--- 処理開始: {fg_dist}cm (FG) と {bg_dist}cm (BG) ---")
        print(f"前景画像: {len(fg_files)}枚, 背景画像: {len(bg_files)}枚")

        # 背景画像を空間周波数でグループ化
        bg_groups = {}
        for path in bg_files:
            prefix = os.path.basename(path).split('cpd')[0]
            if prefix not in bg_groups:
                bg_groups[prefix] = []
            bg_groups[prefix].append(path)

        count = 0
        for fg_path in fg_files:
            fg_prefix = os.path.basename(fg_path).split('cpd')[0]

            if fg_prefix not in bg_groups:
                continue

            for bg_path in bg_groups[fg_prefix]:
                fg_name = os.path.basename(fg_path)
                bg_name = os.path.basename(bg_path)
                
                fg_img = cv2.imread(fg_path)
                bg_img = cv2.imread(bg_path)
                
                h_g, w_g = fg_img.shape[:2]
                h_n, w_n = bg_img.shape[:2]
                
                # 背景画像を前景画像サイズに合わせて中央でクロップ
                cx, cy = w_n // 2, h_n // 2
                x1 = cx - w_g // 2
                y1 = cy - h_g // 2
                x2 = x1 + w_g
                y2 = y1 + h_g
                bg_cropped = bg_img[y1:y2, x1:x2]
                
                # 1. 画像を線形空間(リニア)に戻す (0.0 - 1.0)
                fg_linear = apply_gamma(fg_img, decode=True)
                bg_linear = apply_gamma(bg_cropped, decode=True)
                
                # 2. 物理輝度(cd/m^2)を掛ける
                fg_lum = fg_linear * FG_MAX_LUM
                bg_lum = bg_linear * BG_MAX_LUM
                
                # 3. 光学的な重ね合わせ (加算) -> HDRな物理輝度
                hdr_lum = fg_lum + bg_lum
                
                # 4. Reinhard トーンマッピング (SDRへ圧縮)
                # x / (1 + x) の x に物理輝度(例: 130.0)をそのまま入れるとほぼ 1 (真っ白) になってしまうため、
                # 基準となる最大輝度 (FG_MAX_LUM) で割って相対スケールに戻してから適用します。
                x = hdr_lum / FG_MAX_LUM
                sdr_linear = x / (1.0 + x)
                
                # 5. SDR画像としてガンマエンコード (非線形空間に戻す)
                out_img = apply_gamma(sdr_linear, decode=False)
                
                # ファイル名の生成と保存
                out_name = f"overlap_FG-{fg_name[:-4]}_BG-{bg_name[:-4]}.png"
                out_path = os.path.join(out_dir, out_name)
                
                cv2.imwrite(out_path, out_img)
                count += 1
        
        total_count += count
        print(f"完了! {count} 枚の重ね合わせ画像を保存しました:\n{out_dir}")
        print("-" * 40)

    print(f"=== 全ての処理が完了しました。合計 {total_count} 枚の画像を生成しました。 ===")

if __name__ == "__main__":
    main()