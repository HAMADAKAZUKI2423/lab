import cv2
import glob
import os
import sys
import numpy as np
import csv

# 自然順ソート用
try:
    from natsort import natsorted
except ImportError:
    natsorted = sorted

class Bayer8StatsCalculator:
    def __init__(self, folder_path, extension="*.tiff", gamma=1.0):
        self.folder_path = folder_path
        self.gamma = gamma
        search_path = os.path.join(folder_path, extension)
        self.image_paths = natsorted(glob.glob(search_path))
        
        if not self.image_paths:
            print(f"エラー: 画像が見つかりません: {search_path}")
            sys.exit(1)

        self.images = []
        self.filenames = []
        
        print(f"画像を読み込んでいます (Gamma={self.gamma})...")
        
        # 画像読み込み & デモザイク
        for path in self.image_paths:
            raw_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            
            if raw_img is not None:
                # デモザイク処理 (BayerBG -> BGR)
                if len(raw_img.shape) != 3:
                    try:
                        # ※色が変な場合は cv2.COLOR_BayerGB2BGR などを試してください
                        img_bgr = cv2.cvtColor(raw_img, cv2.COLOR_BayerBG2BGR)
                    except Exception as e:
                        print(f"変換エラー: {path}\n{e}")
                        continue
                else:
                    img_bgr = raw_img

                self.images.append(img_bgr)
                self.filenames.append(os.path.basename(path))
            else:
                print(f"警告: 読み込めないファイル: {path}")

        if not self.images:
            sys.exit(1)

        self.ref_image = self.images[0]
        self.max_val = 255.0
        
        # 範囲選択用
        self.drawing = False
        self.ix, self.iy = -1, -1

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                display_img = self.ref_image.copy()
                cv2.rectangle(display_img, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
                cv2.imshow("Select Region", display_img)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x_min, x_max = sorted([self.ix, x])
            y_min, y_max = sorted([self.iy, y])
            
            if (x_max - x_min) > 0 and (y_max - y_min) > 0:
                print(f"\n範囲選択: x=[{x_min}:{x_max}], y=[{y_min}:{y_max}]")
                # 矩形を固定表示
                display_img = self.ref_image.copy()
                cv2.rectangle(display_img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                cv2.imshow("Select Region", display_img)
                
                # 計算実行
                self.calculate_stats(x_min, x_max, y_min, y_max)

    def calculate_stats(self, x1, x2, y1, y2):
        print("-" * 80)
        print(f"{'Filename':<20} | {'R_Mean':<8} {'R_Std':<8} | {'G_Mean':<8} {'G_Std':<8} | {'B_Mean':<8} {'B_Std':<8} | {'Y_Mean':<8} {'Y_Std':<8}")
        print("-" * 80)

        # CSV保存用のデータリスト
        csv_data = []
        header = ["Filename", "R_Mean", "R_Std", "G_Mean", "G_Std", "B_Mean", "B_Std", "Y_Mean", "Y_Std"]
        csv_data.append(header)

        for i, img in enumerate(self.images):
            # ROI切り出し
            roi = img[y1:y2, x1:x2]
            
            # --- RGB統計 (0-255) ---
            b_roi = roi[:, :, 0].astype(float)
            g_roi = roi[:, :, 1].astype(float)
            r_roi = roi[:, :, 2].astype(float)
            
            r_m, r_s = np.mean(r_roi), np.std(r_roi)
            g_m, g_s = np.mean(g_roi), np.std(g_roi)
            b_m, b_s = np.mean(b_roi), np.std(b_roi)

            # --- 輝度Y統計 (物理推定 0.0-1.0) ---
            r_lin = (r_roi / self.max_val) ** self.gamma
            g_lin = (g_roi / self.max_val) ** self.gamma
            b_lin = (b_roi / self.max_val) ** self.gamma
            
            y_roi = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin
            # 0.0-1.0 の値を 0-255 スケールに戻す
            y_roi = y_roi * self.max_val
            y_m, y_s = np.mean(y_roi), np.std(y_roi)

            # コンソール出力
            fname = self.filenames[i]
            # 長すぎるファイル名は省略
            disp_name = (fname[:17] + '..') if len(fname) > 19 else fname
            
            print(f"{disp_name:<20} | {r_m:8.2f} {r_s:8.2f} | {g_m:8.2f} {g_s:8.2f} | {b_m:8.2f} {b_s:8.2f} | {y_m:8.2f} {y_s:8.2f}")

            # CSVデータ追加
            csv_data.append([fname, r_m, r_s, g_m, g_s, b_m, b_s, y_m, y_s])

        # CSVファイル書き出し
        try:
            with open("stats_result.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(csv_data)
            print("-" * 80)
            print(">> 計算結果を 'stats_result.csv' に保存しました。")
        except PermissionError:
            print(">> エラー: 'stats_result.csv' を書き込めませんでした。ファイルが開かれていませんか？")

    def run(self):
        window_name = "Select Region"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.on_mouse)

        print("ウィンドウ上でマウスをドラッグして範囲を選択してください。")
        print("終了するには 'q' または ESC を押してください。")

        while True:
            # 描画更新待ち
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q') or key == 27:
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # --- 研究用設定 ---
    TARGET_FOLDER = r"C:\Users\HamaKazu\Desktop\GradSchool\lab\experiment\DisplayBrightness\FGDisplay\harfmirror\light"
    FILE_EXTENSION = "*.tiff" 
    GAMMA_VALUE = 1.0 
    # ----------------
    
    if not os.path.exists(TARGET_FOLDER):
        print(f"フォルダが見つかりません: {TARGET_FOLDER}")
    else:
        calc = Bayer8StatsCalculator(TARGET_FOLDER, FILE_EXTENSION, gamma=GAMMA_VALUE)
        calc.run()