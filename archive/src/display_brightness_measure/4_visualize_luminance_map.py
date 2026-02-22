import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

## ディスプレイを撮影した画像から、正規化された物理輝度マップを計算・表示するクラス
# 1. 画像を読み込む (16bit TIFF対応)
# 2. 輝度情報を抽出 (カラーならグレースケールに変換)
# 3. 正規化 (Min-Max Normalization)

class NormalizedLuminanceMapper:
    def __init__(self, image_path, gamma=1.0):
        """
        :param image_path: 画像のパス
        :param gamma: 逆ガンマ補正値（計測用生データなら1.0, 通常画像なら2.2）
        """
        self.image_path = image_path
        self.gamma = gamma
        
        if not os.path.exists(image_path):
            print(f"エラー: ファイルが見つかりません: {image_path}")
            sys.exit(1)
            
        # 画像読み込み (16bit対応)
        self.img_raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if self.img_raw is None:
            print("エラー: 画像を読み込めませんでした。")
            sys.exit(1)

        # ビット深度判定と最大値設定（基礎的な正規化用）
        if self.img_raw.dtype == np.uint16:
            self.max_val = 65535.0
            print(f"画像タイプ: 16-bit TIFF (Max: {self.max_val})")
        else:
            self.max_val = 255.0
            print(f"画像タイプ: 8-bit Image (Max: {self.max_val})")

    def compute_luminance(self):
        """
        画像をリニア化し、物理輝度マップを計算する
        この時点の値は、センサ入力に応じた絶対的な相対値(0.0~1.0の範囲内)
        """
        # カラー(3ch)かモノクロ(1ch)かで処理を分岐
        if len(self.img_raw.shape) == 3:
            # BGR -> ビット深度で正規化
            b = self.img_raw[:, :, 0] / self.max_val
            g = self.img_raw[:, :, 1] / self.max_val
            r = self.img_raw[:, :, 2] / self.max_val
            
            # リニアライズ (逆ガンマ)
            r_lin = r ** self.gamma
            g_lin = g ** self.gamma
            b_lin = b ** self.gamma
            
            # 輝度Y (Rec.709) 計算
            self.luminance_map = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin
            
        else:
            # モノクロ画像の場合
            norm = self.img_raw / self.max_val
            self.luminance_map = norm ** self.gamma

        return self.luminance_map

    def show_normalized_plot(self, output_path=None):
        """
        画像内の最小値・最大値に基づいて正規化（コントラストストレッチ）して表示・保存する
        :param output_path: プロットを保存するパス。Noneの場合は表示のみ。
        """
        # 物理輝度データの計算
        y_map = self.compute_luminance()
        
        # --- 正規化のためのデータ範囲取得 ---
        # 画像内の実際の最小値と最大値を取得
        act_min = np.min(y_map)
        act_max = np.max(y_map)
        
        print(f"データ範囲: Min={act_min:.6f}, Max={act_max:.6f}")
        if act_max - act_min < 1e-9:
             print("警告: 画像が均一（真っ黒または単色）なため、正しく正規化できない可能性があります。")

        # 図の作成
        fig = plt.figure(figsize=(14, 6))
        
        # --- 1. 正規化ヒートマップ (Left) ---
        ax1 = fig.add_subplot(1, 2, 1)
        
        # 【重要】vmin, vmax に実測値を指定することで、表示を正規化する
        im = ax1.imshow(y_map, cmap='inferno', vmin=act_min, vmax=act_max)
        
        ax1.set_title(f"Normalized Luminance Heatmap\n(Stretched to Min-Max)")
        ax1.axis('off')
        
        # カラーバー (実際の値の範囲を示す)
        cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label(f'Physical Intensity Range ({act_min:.2f} - {act_max:.2f})')

        # --- 2. ヒストグラム (Right) ---
        ax2 = fig.add_subplot(1, 2, 2)
        
        flat_data = y_map.flatten()
        # ヒストグラムの表示範囲も実測値に合わせる
        ax2.hist(flat_data, bins=100, range=(act_min, act_max), color='orange', alpha=0.7)
        
        ax2.set_title("Luminance Distribution (Histogram)")
        ax2.set_xlabel("Physical Intensity Value")
        ax2.set_ylabel("Pixel Count")
        ax2.grid(True, linestyle='--', alpha=0.5)

        # 統計量表示
        text_str = f"Min: {act_min:.4f}\nMax: {act_max:.4f}\nMean:{np.mean(flat_data):.4f}"
        ax2.text(0.95, 0.95, text_str, transform=ax2.transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        plt.tight_layout()
        
        # ファイルへの保存処理
        if output_path:
            try:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"プロットを保存しました: {output_path}")
            except Exception as e:
                print(f"エラー: プロットの保存に失敗しました - {e}")

        print("正規化されたマップを表示します。")
        plt.show()

if __name__ == "__main__":
    # --- 設定 ---
    # 解析したい単一画像のパス
    TARGET_IMAGE = r"C:\Users\HamaKazu\Desktop\GradSchool\lab\experiment\DisplayBrightness\pattern\BG_brightness_pattern\Bnless_bg_clipped.png"
    
    # 計測用TIFFなら 1.0, 通常画像なら 2.2
    GAMMA_VALUE = 1.0

    # 出力ファイル名 (Noneにすると保存しない)
    base, _ = os.path.splitext(os.path.basename(TARGET_IMAGE))
    OUTPUT_PLOT_PATH = f"{base}_luminance_map.png"
    # -----------

    mapper = NormalizedLuminanceMapper(TARGET_IMAGE, gamma=GAMMA_VALUE)
    mapper.show_normalized_plot(output_path=OUTPUT_PLOT_PATH)