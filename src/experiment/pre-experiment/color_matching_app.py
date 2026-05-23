# color_matching_app.py
# カラーマッチング実験用スタンドアロンアプリケーション
# Window 1（参考画像）と Window 2（テスト画像）に分けて表示
# XYZ色空間でのカラーマッチングタスクを実行

import tkinter as tk
from tkinter import ttk
import os
import sys

# Add parent directories to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import color_matching
import stimuli_utils


class ColorMatchingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Color Matching - Experimenter (Window 2)")
        self.root.configure(bg='black')
        self.root.geometry("1920x1080")
        self.root.state('zoomed')
        
        # 基本設定
        self.participant_id = tk.StringVar(value="0")
        self.offset_x = tk.DoubleVar(value=0)
        self.offset_y = tk.DoubleVar(value=0)
        self.pupil_diameter_val = tk.DoubleVar(value=4.0)
        
        # カラーマッチング用変数
        self.color_x_factor_val = tk.DoubleVar(value=1.0)
        self.color_z_factor_val = tk.DoubleVar(value=1.0)
        
        # 実験パラメータ
        self.distance1 = 50  # cm (参考画像までの距離)
        self.distance2 = 150  # cm (背景までの距離)
        
        # Canvas 設定
        self.width = 1920
        self.height = 1080
        
        # キーバインディング管理
        self.key_bindings = {}
        
        # Create Window 1 immediately (reference display)
        self.win1 = tk.Toplevel(self.root)
        self.win1.title("Color Matching - Reference (Window 1)")
        self.win1.configure(bg='black')
        self.win1.state('zoomed')

        # Window 1 canvas
        self.canvas1 = tk.Canvas(self.win1, bg='black', highlightthickness=0)
        self.canvas1.pack(fill=tk.BOTH, expand=True)

        # Canvas 2 (experimenter/main window)
        self.canvas2 = tk.Canvas(self.root, bg='black', highlightthickness=0)
        self.canvas2.pack(fill=tk.BOTH, expand=True)

        # 結果ディレクトリ
        lab_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
        self.result_dir = os.path.join(lab_root, "results", "tables", "pre-experiment-color-matching")

        # UI初期化
        # Update and set canvas explicit sizes to match screen (like gabor app)
        self.root.update_idletasks()
        try:
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
            self.width = screen_w
            self.height = screen_h
            self.canvas1.config(width=self.width, height=self.height)
            self.canvas2.config(width=self.width, height=self.height)
        except Exception:
            # keep defaults if measurement fails
            pass

        self._setup_ui()
        
    def _setup_ui(self):
        """基本的なUI構造をセットアップ"""
        # Canvas を黒でクリア
        self.canvas2.delete("all")
        self.canvas1.delete("all")

        # 参加者入力と開始ボタンを中央に表示
        self.participant_frame = tk.Frame(self.root, bg='gray20', padx=20, pady=20)
        self.participant_frame.place(relx=0.5, rely=0.5, anchor='center')

        tk.Label(self.participant_frame, text="Participant ID:", bg='gray20', fg='white', font=("Arial", 14)).grid(row=0, column=0, sticky='e', padx=5, pady=5)
        tk.Entry(self.participant_frame, textvariable=self.participant_id, width=20, font=("Arial", 14)).grid(row=0, column=1, sticky='w', padx=5, pady=5)
        
        start_btn = tk.Button(self.participant_frame, text="Start Color Matching", font=("Arial", 14), command=self.start_color_matching)
        start_btn.grid(row=1, column=0, columnspan=2, pady=15)
        start_btn.focus_set()

    def start_color_matching(self):
        """カラーマッチング実験を開始"""
        print(f"Starting color matching with participant: {self.participant_id.get()}")

        if hasattr(self, 'participant_frame') and self.participant_frame.winfo_exists():
            self.participant_frame.destroy()

        # Canvas をクリアして準備
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        self.win1.lift()

        # ウィジェットサイズ・DPI情報が更新されるよう待つ
        self.root.update_idletasks()
        self.win1.update_idletasks()
        self.canvas1.update_idletasks()
        self.canvas2.update_idletasks()

        # Canvas の DPI (ppi) を用いて pixels_per_cm を計算して保存
        try:
            ppi1 = float(self.canvas1.winfo_fpixels('1i'))
            self.pixels_per_cm_win1 = ppi1 / 2.54
        except Exception:
            self.pixels_per_cm_win1 = stimuli_utils.PIXELS_PER_CM

        try:
            ppi2 = float(self.canvas2.winfo_fpixels('1i'))
            self.pixels_per_cm_win2 = ppi2 / 2.54
        except Exception:
            self.pixels_per_cm_win2 = stimuli_utils.PIXELS_PER_CM

        # DEBUG: print computed DPI / pixels_per_cm and derived square size
        try:
            print(f"DEBUG: ppi1={ppi1 if 'ppi1' in locals() else 'n/a'}, pixels_per_cm_win1={self.pixels_per_cm_win1}")
            print(f"DEBUG: ppi2={ppi2 if 'ppi2' in locals() else 'n/a'}, pixels_per_cm_win2={self.pixels_per_cm_win2}")
            # compute example square size for verification
            test_sq = stimuli_utils.get_size_for_visual_angle(self.distance1, 7.9, canvas=self.canvas1)
            print(f"DEBUG: test square_size (1) = {test_sq}")
        except Exception:
            pass

        self.finish_color_matching_callback = self._on_color_matching_finished
        color_matching.setup_color_matching_calibration(self)
        
    def _on_color_matching_finished(self):
        """カラーマッチング完了時の処理"""
        print("Color matching completed")
        # 結果が保存されました
        if hasattr(self, 'color_match_results'):
            for result in self.color_match_results:
                print(f"  {result['condition']}: X={result['x_factor']:.2f}, Z={result['z_factor']:.2f}")


def main():
    root = tk.Tk()
    app = ColorMatchingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
