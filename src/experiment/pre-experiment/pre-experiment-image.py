# py .\src\experiment\pre-experiment\pre-experiment-image.py
import tkinter as tk
from tkinter import ttk, messagebox
import os
import csv
import datetime
from PIL import Image, ImageTk
import glob
import itertools
import defocus_matching
import random
import math
import numpy as np

from experiment_base_ui import ExperimentBaseUI

class ImagePupilDiameterVar(tk.DoubleVar):
    """最終matching確定時に、image側のdefocus操作UIだけを先に破棄する。"""

    def __init__(self, app, *args, **kwargs):
        self.app = app
        super().__init__(*args, **kwargs)

    def set(self, value):
        patterns = getattr(self.app, "defocus_match_patterns", None)
        current_idx = getattr(self.app, "current_match_idx", -1)
        if patterns is not None and current_idx >= len(patterns):
            frame = getattr(self.app, "ctrl_frame", None)
            if frame is not None and frame.winfo_exists():
                frame.destroy()
        super().set(value)

# ==========================================
# 定数設定エリア (実験条件やデザインはここを変更)
# ==========================================
# --- 実験設定 ---
VISUAL_ANGLE_DEG = 7.9   # 画像の視角 (degree)
NUM_TRIALS_BEFORE_BREAK = 38 # 休憩に入るまでの試行回数
script_dir = os.path.dirname(os.path.abspath(__file__))
lab_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
IMG_DIR_1 = os.path.join(lab_root, "data", "processed", "images", "pre-experiment-image", "bg_imgs")
IMG_DIR_2 = os.path.join(lab_root, "data", "processed", "images", "pre-experiment-image", "fg_imgs")
RESULT_DIR = os.path.join(lab_root, "results", "tables", "pre-experiment-image")
PARTICIPANT_DATA_DIR = os.path.join(
    lab_root, "data", "processed", "tables", "pre-experiment-image"
)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PARTICIPANT_DATA_DIR, exist_ok=True)

# --- 時間設定 (ミリ秒) ---
TIME_PHASE_1 = 500    # Phase 1: Image display only on Win2
TIME_ISI = 1000        # Phase 2: Inter Stimulus Interval (black screen)
TIME_PHASE_2 = 500    # Phase 3: Image display on both windows
DISTANCE_FG = 50
DISTANCE_BG = 150

# --- UIデザイン設定 ---
BG_COLOR = 'black'     # 全体の背景色
PIXELS_PER_CM = 1/0.02331  # モニタのPPC (1mmあたり0.2331画素の場合)
SQUARE_SIZE = 30       # 四隅のマーカーの辺の長さ (px)
CROSS_SIZE = 30        # 中央の十字マーカーのサイズ (px)
MARKER_LINE_WIDTH = 5  # マーカーの線の太さ
WIN1_MARKER_COLOR = 'red'      # Window 1 (被験者側) のマーカー色
WIN2_MARKER_COLOR = 'white'    # Window 2 (実験者側) のマーカー色

# ==========================================

class ExperimentApp(ExperimentBaseUI):
    def __init__(self, root):
        ExperimentBaseUI.__init__(self, root, PARTICIPANT_DATA_DIR)

        self.root = root
        self.root.title("Controller (Window 2)")
        self.root.configure(bg=BG_COLOR)
        
        # --- image実験固有の状態 ---
        self.evaluation_val = tk.IntVar(value=3)
        self.participant_dominance = tk.StringVar(value="Right")
        self.pupil_diameter_val = ImagePupilDiameterVar(self, value=4.0)

        # 距離条件はmatching実験と同じく前景50 cm／背景150 cmに固定する。
        self.distance1 = DISTANCE_FG
        self.distance2 = DISTANCE_BG

        # 両眼で位置合わせとdefocus matchingを行う。
        self.calibration_eyes = ["Right", "Left"]
        self.current_calib_eye_idx = 0
        self.calib_results = {}
        self.current_pd_mean = 0.0
        self.current_pd_std = 0.0
        self.detailed_defocus_results = []

        # image実験では色・輝度補正を行わない。共通defocus matchingが使う
        # 輝度→画素変換だけを線形マッピングとして用意する。
        self.color_matrix = None
        self.gamma_bg = None
        self.gamma_fg = None
        self.bg_lums = np.array([0.0, 30.0], dtype=np.float64)
        self.bg_pixels = np.array([0.0, 255.0], dtype=np.float64)

        self.trial_list = []
        self.current_trial_index = 0
        self.results = []
        self.key_bindings = {}
        self.eval_buttons = []
        self.result_dir = None
        
        # --- 画像ファイルの読み込み ---
        
        # 現在の試行で表示する画像を保持する変数 (Tkinterで画像を表示する際の必須処理)
        self.current_img_path_1 = None
        self.current_img_path_2 = None
        self.photo1 = None
        self.photo2 = None
 
        # --- ウィンドウのセットアップ ---
        # プライマリモニタのスクリーンサイズを取得
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        
        # Window 2 (実験者用操作画面): 最大化表示
        self.root.state('zoomed')

        # Window 1 (被験者用表示画面): スクリーンの右半分に表示
        self.win1 = tk.Toplevel(self.root)
        self.win1.title("Display (Window 1)")
        self.win1.state('zoomed')

        # ウィンドウの最大化をUIに反映させ、実際のサイズを取得して更新する
        self.root.update_idletasks()
        self.width = screen_w
        self.height = screen_h

        # Canvasの作成 (サイズ取得後)
        self.canvas2 = tk.Canvas(self.root, width=self.width, height=self.height, bg=BG_COLOR, highlightthickness=0)
        self.canvas2.pack(fill="both", expand=True)

        self.win1.configure(bg=BG_COLOR)
        self.canvas1 = tk.Canvas(self.win1, width=self.width, height=self.height, bg=BG_COLOR, highlightthickness=0)
        self.canvas1.pack(fill="both", expand=True)

        # --- ステップ0: 実験設定UIの表示 ---
        self.setup_participant_info_ui()

    def setup_participant_info_ui(self):
        """matching実験と同じく、参加者IDを先に確認する。"""
        self.participant_frame = tk.Frame(self.root, bg="gray", padx=20, pady=20)
        self.participant_frame.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(self.participant_frame, text="Enter Participant ID", font=("Arial", 16)).grid(
            row=0, column=0, columnspan=2, pady=10
        )
        tk.Label(self.participant_frame, text="Participant ID:").grid(
            row=1, column=0, sticky="w", padx=5, pady=5
        )
        entry_id = tk.Entry(self.participant_frame, textvariable=self.participant_id)
        entry_id.grid(row=1, column=1, padx=5, pady=5)
        entry_id.focus_set()
        next_button = tk.Button(
            self.participant_frame,
            text="Next",
            command=self.check_participant_id,
        )
        next_button.grid(row=2, column=0, columnspan=2, pady=20)
        next_button.bind("<Return>", self.check_participant_id)
        entry_id.bind("<Return>", self.check_participant_id)

    def check_participant_id(self, event=None):
        participant_id = self.participant_id.get().strip()
        if not participant_id:
            messagebox.showwarning("Input Error", "Please enter a Participant ID.")
            return
        participant_path = os.path.join(PARTICIPANT_DATA_DIR, "participants.csv")
        found = False
        if os.path.exists(participant_path):
            with open(participant_path, "r", encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    if row["ID"] == participant_id:
                        self.participant_age.set(row.get("Age", ""))
                        self.participant_gender.set(row.get("Gender", ""))
                        self.participant_ipd.set(row.get("IPD", ""))
                        self.participant_dominance.set(row.get("Dominance", "Right"))
                        found = True
                        break
        self.participant_frame.destroy()
        if found:
            self.start_calibration_sequence()
        else:
            self.setup_new_participant_ui()

    def setup_new_participant_ui(self):
        self.participant_frame = tk.Frame(self.root, bg="gray", padx=20, pady=20)
        self.participant_frame.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(
            self.participant_frame,
            text=f"New Participant Registration (ID: {self.participant_id.get()})",
            font=("Arial", 16),
        ).grid(row=0, column=0, columnspan=2, pady=10)
        tk.Label(self.participant_frame, text="Age:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(self.participant_frame, textvariable=self.participant_age).grid(row=1, column=1, padx=5, pady=5)
        tk.Label(self.participant_frame, text="Gender:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        gender_combo = ttk.Combobox(self.participant_frame, textvariable=self.participant_gender, values=["Male", "Female", "Other"])
        gender_combo.grid(row=2, column=1, padx=5, pady=5)
        gender_combo.set("Male")
        tk.Label(self.participant_frame, text="IPD (mm):").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(self.participant_frame, textvariable=self.participant_ipd).grid(row=3, column=1, padx=5, pady=5)
        tk.Label(self.participant_frame, text="Eye Dominance:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        dominance_combo = ttk.Combobox(self.participant_frame, textvariable=self.participant_dominance, values=["Right", "Left"])
        dominance_combo.grid(row=4, column=1, padx=5, pady=5)
        dominance_combo.set("Right")
        tk.Button(self.participant_frame, text="Register and Next", command=self.register_and_start).grid(
            row=5, column=0, columnspan=2, pady=20
        )

    def register_and_start(self, event=None):
        try:
            int(self.participant_age.get())
            float(self.participant_ipd.get())
        except ValueError:
            messagebox.showwarning("Input Error", "Please enter valid Age and IPD values.")
            return
        self.save_participant_data()
        self.participant_frame.destroy()
        self.start_calibration_sequence()

    def save_participant_data(self):
        participant_path = os.path.join(PARTICIPANT_DATA_DIR, "participants.csv")
        fields = ["ID", "Age", "Gender", "IPD", "Dominance"]
        rows = []
        updated = False
        if os.path.exists(participant_path):
            with open(participant_path, "r", encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    if row["ID"] == self.participant_id.get():
                        row.update({
                            "Age": self.participant_age.get(),
                            "Gender": self.participant_gender.get(),
                            "IPD": self.participant_ipd.get(),
                            "Dominance": self.participant_dominance.get(),
                        })
                        updated = True
                    rows.append(row)
        if not updated:
            rows.append({
                "ID": self.participant_id.get(),
                "Age": self.participant_age.get(),
                "Gender": self.participant_gender.get(),
                "IPD": self.participant_ipd.get(),
                "Dominance": self.participant_dominance.get(),
            })
        with open(participant_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def start_calibration_sequence(self):
        self.win1.update_idletasks()
        self.width = self.win1.winfo_width()
        self.height = self.win1.winfo_height()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = os.path.join(RESULT_DIR, f"{self.participant_id.get()}_{timestamp}")
        os.makedirs(self.result_dir, exist_ok=True)
        self.current_calib_eye_idx = 0
        self.calib_results = {}
        self.detailed_defocus_results = []
        self.current_trial_index = 0
        self.results = []
        self.start_eye_calibration()

    def start_eye_calibration(self):
        if self.current_calib_eye_idx >= len(self.calibration_eyes):
            self.apply_dominant_eye_calibration()
            self.show_image_experiment_start_ui()
            return
        eye = self.calibration_eyes[self.current_calib_eye_idx]
        messagebox.showinfo("Calibration", f"Next: Calibration for {eye} Eye.\nPlease cover the other eye.")
        self.offset_x.set(0)
        self.offset_y.set(0)
        self.pupil_diameter_val.set(4.0)
        self.setup_calibration_ui(is_break=False)

    def apply_dominant_eye_calibration(self):
        """優位眼の位置と瞳孔径をimage実験へ適用する。"""
        eye = self.participant_dominance.get()
        result = self.calib_results.get(eye, self.calib_results["Right"])
        self.offset_x.set(result["offset_x"])
        self.offset_y.set(result["offset_y"])
        self.current_pd_mean = result["pd_mean"]

    def show_image_experiment_start_ui(self):
        """defocus matching終了後、image実験の開始確認を表示する。"""
        self.clear_key_bindings()
        if hasattr(self, "ctrl_frame") and self.ctrl_frame.winfo_exists():
            self.ctrl_frame.destroy()

        self.canvas1.delete("all")
        self.canvas2.delete("all")

        self.ctrl_frame = tk.Frame(self.root, bg="gray", padx=30, pady=30)
        self.ctrl_frame.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(
            self.ctrl_frame,
            text="The image experiment will now begin.\nPress Enter to start.",
            bg="gray",
            fg="white",
            font=("Arial", 16),
        ).pack(pady=15)

        start_button = tk.Button(
            self.ctrl_frame,
            text="Start Image Experiment",
            command=self.begin_image_experiment,
        )
        start_button.pack(pady=10)
        start_button.focus_set()
        self.key_bindings["<Return>"] = self.root.bind(
            "<Return>", self.begin_image_experiment
        )

    def begin_image_experiment(self, event=None):
        """開始確認画面を閉じてimage実験を開始する。"""
        self.clear_key_bindings()
        if hasattr(self, "ctrl_frame") and self.ctrl_frame.winfo_exists():
            self.ctrl_frame.destroy()
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        self.start_experiment()

    def start_eye_defocus_matching(self):
        """位置合わせ後、matching実験と共通のdefocus matchingへ進む。"""
        if hasattr(self, "ctrl_frame") and self.ctrl_frame.winfo_exists():
            self.ctrl_frame.destroy()
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        defocus_matching.setup_defocus_matching_ui(self)

    def _reset_to_setup_ui(self):
        """UIをクリアし、初期設定画面に戻る"""
        # ctrl_frameやinstruction_frameが存在し、破棄されていない場合のみdestroyを呼ぶ
        if hasattr(self, 'ctrl_frame') and self.ctrl_frame.winfo_exists():
            self.ctrl_frame.destroy()
        if hasattr(self, 'instruction_frame') and self.instruction_frame.winfo_exists():
            self.instruction_frame.destroy()
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        self.setup_participant_info_ui()

    def clear_key_bindings(self):
        """登録済みのrootキーバインドを解除する。"""
        for key, binding_id in list(self.key_bindings.items()):
            self.root.unbind(key, binding_id)
        self.key_bindings.clear()

    def get_size_for_visual_angle(self, distance_cm, angle_deg):
        """指定された視角と距離から、対応するピクセルサイズを計算する"""
        if distance_cm <= 0:
            return 0
        # 物理サイズ[cm] = 2 * 距離[cm] * tan(視角[rad] / 2)
        angle_rad = math.radians(angle_deg)

        size_cm = 2 * distance_cm * math.tan(angle_rad / 2)
        # ピクセルサイズ = 物理サイズ[cm] * PPC
        return round(size_cm * PIXELS_PER_CM)

    def draw_image_corner_brackets(self, canvas, size_w, size_h, offset_x=0, offset_y=0, color='white', flip_x=False, line_width=MARKER_LINE_WIDTH):
        """指定された画像表示領域の四隅に、鍵括弧状のマーカーを描画する"""
        s = SQUARE_SIZE
        
        # 画面の中心座標
        cx, cy = canvas.winfo_width() // 2, canvas.winfo_height() // 2
        # 画像表示領域の左上と右下の座標を計算 (オフセット適用)
        x0 = cx - size_w // 2 + offset_x
        y0 = cy - size_h // 2 + offset_y
        x1 = cx + size_w // 2 + offset_x
        y1 = cy + size_h // 2 + offset_y

        # X座標変換関数 (flip_xがTrueなら左右反転)
        def tx(x):
            return canvas.winfo_width() - x if flip_x else x

        # Top-left
        canvas.create_line(tx(x0), y0, tx(x0 + s), y0, fill=color, width=line_width, tags="calib")
        canvas.create_line(tx(x0), y0, tx(x0), y0 + s, fill=color, width=line_width, tags="calib")
        # Top-right
        canvas.create_line(tx(x1 - s), y0, tx(x1), y0, fill=color, width=line_width, tags="calib")
        canvas.create_line(tx(x1), y0, tx(x1), y0 + s, fill=color, width=line_width, tags="calib")
        # Bottom-left
        canvas.create_line(tx(x0), y1 - s, tx(x0), y1, fill=color, width=line_width, tags="calib")
        canvas.create_line(tx(x0), y1, tx(x0 + s), y1, fill=color, width=line_width, tags="calib")
        # Bottom-right
        canvas.create_line(tx(x1 - s), y1, tx(x1), y1, fill=color, width=line_width, tags="calib")
        canvas.create_line(tx(x1), y1 - s, tx(x1), y1, fill=color, width=line_width, tags="calib")

    def draw_center_cross(self, canvas, offset_x=0, offset_y=0, color='white'):
        """画面中央に一点へ向かう4つの矢尻（棒なし）を描画する"""
        cx = canvas.winfo_width() // 2 + offset_x
        cy = canvas.winfo_height() // 2 + offset_y
        
        # 矢尻の形状パラメータ（CROSS_SIZEを基準にサイズを大きく設定）
        d2 = int(CROSS_SIZE * 1.2)  # 先端から底辺までの距離
        d1 = int(CROSS_SIZE * 0.9)  # 先端から凹みまでの距離
        d3 = int(CROSS_SIZE * 0.5)  # 幅の半分

        # 4方向の矢尻の頂点座標
        pts_left = [cx, cy, cx - d2, cy - d3, cx - d1, cy, cx - d2, cy + d3]
        pts_right = [cx, cy, cx + d2, cy - d3, cx + d1, cy, cx + d2, cy + d3]
        pts_up = [cx, cy, cx - d3, cy - d2, cx, cy - d1, cx + d3, cy - d2]
        pts_down = [cx, cy, cx - d3, cy + d2, cx, cy + d1, cx + d3, cy + d2]

        for pts in [pts_left, pts_right, pts_up, pts_down]:
            canvas.create_polygon(pts, fill=color, outline="black", width=2, tags="calib")

    def update_calibration_view(self, *args):
        """キャリブレーション画面の表示を更新する (スライダー操作時に呼ばれる)"""
        # 既存のマーカーを一旦すべて削除
        self.canvas1.delete("calib")
        self.canvas2.delete("calib")

        d_fg = self.distance1
        d_bg = self.distance2

        # --- 前景マーカーのサイズ計算 (正方形) ---
        fg_size = self.get_size_for_visual_angle(d_fg, VISUAL_ANGLE_DEG)
        
        # --- 背景マーカーのサイズ計算 (横長) ---
        bg_h = self.get_size_for_visual_angle(d_bg, VISUAL_ANGLE_DEG)
        bg_w = self.get_size_for_visual_angle(d_bg, VISUAL_ANGLE_DEG * 2)

        # --- Window 1 (被験者側) のマーカー描画 ---
        # 1. 背景全体（横長）の四隅にマーカーを描画
        self.draw_image_corner_brackets(self.canvas1, bg_w, bg_h, self.offset_x.get(), self.offset_y.get(), color=WIN1_MARKER_COLOR, line_width=MARKER_LINE_WIDTH * 1.5)
        # 2. 背景の中央に、正方形マーカーを描画
        self.draw_image_corner_brackets(self.canvas1, bg_h, bg_h, self.offset_x.get(), self.offset_y.get(), color=WIN1_MARKER_COLOR, line_width=MARKER_LINE_WIDTH * 1.5)
        
        # Window 2 (実験者側): 基準となるマーカーと十字を描画
        self.draw_image_corner_brackets(self.canvas2, fg_size, fg_size, 0, 0, color=WIN2_MARKER_COLOR, flip_x=False)
        self.draw_center_cross(self.canvas2, color=WIN2_MARKER_COLOR)

    def adjust_offset(self, dx, dy):
        """矢印キーによるオフセット調整用関数"""
        self.offset_x.set(self.offset_x.get() + dx)
        self.offset_y.set(self.offset_y.get() + dy)
        self.update_calibration_view()
        return "break" # デフォルトのイベント処理（スライダーの移動など）を停止する

    def setup_calibration_ui(self, is_break=False):
        """ステップ1: キャリブレーション用UIを構築し表示する"""
        self.update_calibration_view()
        
        # 操作用UIをまとめるためのフレーム
        self.ctrl_frame = tk.Frame(self.root, bg='gray')
        self.ctrl_frame.place(relx=0.5, rely=0.8, anchor='center')

        if is_break:
            instruction_text = "This is a break. You can adjust the position if needed.\nPress 'Resume Experiment' to continue."
            button_text = "Resume Experiment"
            button_command = self.resume_experiment
        else:
            instruction_text = "Use the arrow keys to adjust the position of the red frame."
            button_text = "Calibration Done, Next"
            button_command = self.start_eye_defocus_matching

        # 位置調整の指示ラベル
        tk.Label(self.ctrl_frame, text=instruction_text, bg='gray', fg='white', font=("Arial", 12)).pack(pady=10, padx=20)

        # ボタン
        btn = tk.Button(self.ctrl_frame, text=button_text, command=button_command)
        btn.pack(pady=10)
        btn.focus_set() # ボタンにフォーカスを当ててキー入力を受け付ける
        btn.bind('<Return>', lambda event: button_command())
        btn.bind('<Left>', lambda e: self.adjust_offset(-1, 0))
        btn.bind('<Right>', lambda e: self.adjust_offset(1, 0))
        btn.bind('<Up>', lambda e: self.adjust_offset(0, -1))
        btn.bind('<Down>', lambda e: self.adjust_offset(0, 1))

    def resume_experiment(self):
        """休憩を終了し、実験を再開する"""
        self.ctrl_frame.destroy()
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        self.run_trial()

    def start_break(self):
        """休憩を開始し、キャリブレーションUIを表示する"""
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        self.setup_calibration_ui(is_break=True)
    def _update_eval_highlight(self):
        """Highlights the currently selected evaluation button."""
        current_value = self.evaluation_val.get()
        for i, button_items in enumerate(self.eval_buttons):
            value = 5 - i
            canvas = button_items['canvas']
            dot = button_items['dot']
            label = button_items['label']
            
            if value == current_value:
                canvas.itemconfig(dot, fill='black', outline='black')
                label.config(font=("Arial", 12, "bold"))
            else:
                canvas.itemconfig(dot, fill='white', outline='white')
                label.config(font=("Arial", 12, "normal"))

    def _move_selection(self, delta):
        """Moves the evaluation selection left or right."""
        current_value = self.evaluation_val.get()
        new_value = current_value + delta
        new_value = max(1, min(5, new_value)) # Clamp between 1 and 5
        if new_value != current_value:
            self.evaluation_val.set(new_value)
            self._update_eval_highlight()
        return "break" # Stop event propagation

    def start_experiment(self):
        """実験ループを開始する"""
        self.clear_key_bindings()

        # --- 1. Get experiment parameters and load image paths ---
        bg_img_paths = sorted(glob.glob(os.path.join(IMG_DIR_1, '*')))
        fg_img_paths = sorted(glob.glob(os.path.join(IMG_DIR_2, '*')))

        if not bg_img_paths or not fg_img_paths:
            messagebox.showerror("Error", f"Image folder not found or is empty.\n\nBG path: {IMG_DIR_1}\nFG path: {IMG_DIR_2}")
            self._reset_to_setup_ui()
            return

        # 全背景画像×全前景画像の組み合わせをランダムな順で提示する。
        self.trial_list = [
            {"bg_image": bg_path, "fg_image": fg_path}
            for bg_path, fg_path in itertools.product(bg_img_paths, fg_img_paths)
        ]
        random.shuffle(self.trial_list)

        print(f"Found {len(bg_img_paths)} background images and {len(fg_img_paths)} foreground images.")
        print(f"Total trials to be generated: {len(self.trial_list)}")
            
        if hasattr(self, "ctrl_frame") and self.ctrl_frame.winfo_exists():
            self.ctrl_frame.destroy()
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        self.run_trial()
        
    def run_trial(self):
        """1試行分の実験シーケンスを実行する"""
        # 全ての試行が終了したら実験を終了
        if self.current_trial_index >= len(self.trial_list):
            self.finish_experiment()
            return

        # --- 1. Get trial conditions and load/process images ---
        trial_cond = self.trial_list[self.current_trial_index]
        self.current_img_path_1 = trial_cond["bg_image"]
        self.current_img_path_2 = trial_cond["fg_image"]
        img1 = Image.open(self.current_img_path_1)
        img2 = Image.open(self.current_img_path_2)
        
        # --- 画像処理 ---
        d_fg = self.distance1
        d_bg = self.distance2
        fg_size = self.get_size_for_visual_angle(d_fg, VISUAL_ANGLE_DEG)

        # 背景は横長(幅が視角の2倍)なので、幅と高さを別々に計算してリサイズ
        bg_h = self.get_size_for_visual_angle(d_bg, VISUAL_ANGLE_DEG)
        bg_w = self.get_size_for_visual_angle(d_bg, VISUAL_ANGLE_DEG * 2)

        # 背景画像の前処理: 512x512にリサイズし、中央の512x256(アスペクト比2:1)を切り出す
        # これにより、元画像の中央256x256の領域が、前景画像と重なる領域となる
        img1 = img1.resize((512, 512))
        img1 = img1.crop((0, 128, 512, 384)) # top=128, bottom=384 (高さ256px分を切り出し)

        # 最終的な表示サイズ(視角依存)にリサイズ
        img1 = img1.resize((bg_w, bg_h))
        img2 = img2.resize((fg_size, fg_size))

        img2 = img2.transpose(Image.FLIP_LEFT_RIGHT) # Flip for experimenter view

        # Convert to PhotoImage (must be done before displaying)
        self.photo1 = ImageTk.PhotoImage(img1)
        self.photo2 = ImageTk.PhotoImage(img2)

        # --- シーケンス開始 ---
        # フェーズ1: Window 2にのみ画像を表示
        self.canvas1.configure(bg='black')
        self.canvas1.delete("all")
        
        self.canvas2.create_image(self.canvas2.winfo_width()//2, self.canvas2.winfo_height()//2, image=self.photo2, anchor='center', tags="img")
        
        # 指定時間後に次のフェーズへ
        self.root.after(TIME_PHASE_1, self.phase_isi)

    def phase_isi(self):
        """フェーズ2: ISI (Inter Stimulus Interval)。両画面を暗転させる。"""
        # Window 1は既に暗転している
        self.canvas2.delete("img") # Window 2の画像を削除
        
        # 暗転中、Window 2には基準マーカーを表示
        d_fg = self.distance1
        fg_size = self.get_size_for_visual_angle(d_fg, VISUAL_ANGLE_DEG)
        self.draw_image_corner_brackets(self.canvas2, fg_size, fg_size, 0, 0, color=WIN2_MARKER_COLOR, flip_x=True)
        self.draw_center_cross(self.canvas2, color=WIN2_MARKER_COLOR)

        # 指定時間後に次のフェーズへ
        self.root.after(TIME_ISI, self.phase_both)

    def phase_both(self):
        """フェーズ3: 両方のウィンドウに画像を表示する"""
        # Window 2のマーカーを削除
        self.canvas2.delete("calib")

        # Window 1 に画像を表示 (キャリブレーションで調整したオフセットを適用)
        ox, oy = self.offset_x.get(), self.offset_y.get()
        self.canvas1.create_image(self.width//2 + ox, self.height//2 + oy, image=self.photo1, anchor='center', tags="img")
        
        # Window 2 に画像を表示
        self.canvas2.create_image(self.canvas2.winfo_width()//2, self.canvas2.winfo_height()//2, image=self.photo2, anchor='center', tags="img")

        # 指定時間後に次のフェーズへ
        self.root.after(TIME_PHASE_2, self.phase_end_trial)

    def phase_end_trial(self):
        """試行終了処理。両画面の画像を消去し、評価UIを表示する"""
        self.canvas1.delete("img")
        self.canvas2.delete("img")
        self.show_evaluation_ui()

    def show_evaluation_ui(self):
        """ステップ4: 評価用UIを表示する"""
        self.eval_frame = tk.Frame(self.root, bg='white', padx=20, pady=20, relief="solid", borderwidth=1)
        self.eval_frame.place(relx=0.5, rely=0.5, anchor='center')

        tk.Label(self.eval_frame, text=f"Trial No.{self.current_trial_index + 1} の評価", font=("Arial", 16), bg='white').pack(pady=(0, 20))
        
        # デフォルト値を3に設定
        self.evaluation_val.set(3) # Reset to default for each trial

        # --- 評価選択UI ---
        options_frame = tk.Frame(self.eval_frame, bg='white')
        options_frame.pack(pady=10, padx=20)

        self.eval_buttons.clear()
        for i in range(5, 0, -1):
            option_frame = tk.Frame(options_frame, bg='white')
            option_frame.pack(side='left', padx=15)

            canvas = tk.Canvas(option_frame, width=30, height=30, bg='white', highlightthickness=0)
            canvas.pack()
            
            # Draw outer circle
            canvas.create_oval(5, 5, 25, 25, outline='black', width=2)
            
            # Create placeholder for inner dot (initially hidden)
            dot_item = canvas.create_oval(10, 10, 20, 20, fill='white', outline='white')
            
            # Draw number label below
            label = tk.Label(option_frame, text=str(i), font=("Arial", 12), bg='white')
            label.pack()
            
            self.eval_buttons.append({'canvas': canvas, 'dot': dot_item, 'label': label})

        # --- 評価基準ラベル ---
        desc_frame = tk.Frame(self.eval_frame, bg='white')
        desc_frame.pack(fill='x', padx=10, pady=(5, 10))
        tk.Label(desc_frame, text="5: Very clear", bg='white').pack(side='left')
        tk.Label(desc_frame, text="1: Invisible", bg='white').pack(side='right')

        # --- 操作説明 ---
        tk.Label(self.eval_frame, text="◀ / ▶ で選択, ▼ で決定", font=("Arial", 10), bg='white').pack(pady=(10, 0))

        # --- 初期化とキーバインド ---
        self._update_eval_highlight()

        self.key_bindings['<Left>'] = self.root.bind('<Left>', lambda e: self._move_selection(-1))
        self.key_bindings['<Right>'] = self.root.bind('<Right>', lambda e: self._move_selection(1))
        self.key_bindings['<Down>'] = self.root.bind('<Down>', lambda e: self.save_and_next())
        self.root.focus_set()

    def save_and_next(self):
        """評価データを保存し、次の試行に進む"""
        self.clear_key_bindings()
        
        score = self.evaluation_val.get()
        f1 = os.path.basename(self.current_img_path_1)
        f2 = os.path.basename(self.current_img_path_2)
        
        right = self.calib_results.get("Right", {})
        left = self.calib_results.get("Left", {})
        self.results.append([
            self.participant_id.get(),
            self.participant_age.get(),
            self.participant_gender.get(),
            self.participant_ipd.get(),
            self.participant_dominance.get(),
            self.distance1,
            self.distance2,
            right.get("pd_mean"),
            right.get("offset_x"),
            right.get("offset_y"),
            left.get("pd_mean"),
            left.get("offset_x"),
            left.get("offset_y"),
            self.current_trial_index + 1,
            f1,
            f2,
            score,
        ])
        
        self.eval_frame.destroy()
        self.current_trial_index += 1
        
        # 休憩を挟むかチェック
        is_break_time = (
            self.current_trial_index % NUM_TRIALS_BEFORE_BREAK == 0
            and self.current_trial_index < len(self.trial_list)
        )

        if is_break_time:
            self.root.after(500, self.start_break)
        else:
            # 少し待ってから次の試行を開始 (UIの応答性を保つため)
            self.root.after(500, self.run_trial)

    def finish_experiment(self):
        """実験終了処理。結果をCSVファイルに保存する"""
        # IDと日付でフォルダを作成 (id_YYYYMMDD)
        p_id = self.participant_id.get()
        now = datetime.datetime.now()
        date_str = now.strftime("%Y%m%d")
        save_folder = self.result_dir or os.path.join(RESULT_DIR, f"{p_id}_{date_str}")
        
        os.makedirs(save_folder, exist_ok=True)
            
        # ファイル名に被験者IDと現在時刻を含める
        filename = os.path.join(save_folder, "image_evaluation.csv")
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = [
                "ID", "Age", "Gender", "IPD(mm)", "Dominance",
                "Distance_FG(cm)", "Distance_BG(cm)",
                "PD_Right", "OffsetX_Right", "OffsetY_Right",
                "PD_Left", "OffsetX_Left", "OffsetY_Left",
                "Trial_ID", "Image_Win1", "Image_Win2", "Score",
            ]
            writer.writerow(header)
            writer.writerows(self.results)
            
        messagebox.showinfo("Finished", f"Experiment finished.\nData saved to: {filename}")
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ExperimentApp(root)
    root.mainloop()