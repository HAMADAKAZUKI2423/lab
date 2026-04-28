# 実行例: 
# py .\src\experiment\pre-experiment\pre-experiment-gabor.py
import tkinter as tk
from tkinter import ttk, messagebox
import os
import csv
import datetime
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import glob
import random
import math
import numpy as np
import defocus_matching
import stimuli_utils

# ==========================================
# 定数設定エリア (実験条件やデザインはここを変更)
# ==========================================
# --- 実験設定 ---
VISUAL_ANGLE_DEG = 7.9   # 画像の視角 (degree)
NUM_REPETITIONS = 5      # 試行の反復回数
script_dir = os.path.dirname(os.path.abspath(__file__))
lab_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
RESULT_DIR = os.path.join(lab_root, "results", "tables", "pre-experiment-matching")
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
PARTICIPANT_DATA_DIR = os.path.join(lab_root, "data", "processed", "tables", "pre-experiment-matching")
if not os.path.exists(PARTICIPANT_DATA_DIR):
    os.makedirs(PARTICIPANT_DATA_DIR)

# --- デフォーカスマッチング設定 ---
# DEFOCUS_BLUR_SCALE_FACTOR = 0.55  # defocus_matching.py に移動
PUPIL_DIAMETER_MM = 4.0 # 瞳孔径 (mm)


# --- UIデザイン設定 ---
BG_COLOR = 'black'     # 全体の背景色
# NOTE: PPCはstimuli_utils.pyで定義
# PIXELS_PER_CM, SQUARE_SIZE, CROSS_SIZE, MARKER_LINE_WIDTH は stimuli_utils.py で定義
WIN1_MARKER_COLOR = 'red'      # Window 1 (被験者側) のマーカー色
WIN2_MARKER_COLOR = 'white'    # Window 2 (実験者側) のマーカー色

# ==========================================

class ExperimentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Controller (Window 2)")
        self.root.configure(bg=BG_COLOR)
        
        # --- 変数の初期化 ---
        self.offset_x = tk.IntVar(value=0)
        self.offset_y = tk.IntVar(value=0)
        self.pupil_diameter_val = tk.DoubleVar(value=4.0)
        self.evaluation_val = tk.IntVar(value=3)
        self.participant_age = tk.StringVar()
        self.participant_gender = tk.StringVar()
        self.participant_ipd = tk.StringVar()
        self.participant_id = tk.StringVar()
        self.participant_dominance = tk.StringVar(value="Right")

        # --- 実験条件用変数 ---
        self.distance1 = 50  # Foreground fixed (cm)
        self.distance2 = 77  # Background fixed (cm)
        self.spatial_freq = 4 # Spatial frequency fixed (cpd)
        
        self.trial_list = []
        self.blocks = []
        self.current_block_index = 0
        self.current_trial_in_block = 0
        self.current_trial_in_experiment = 0
        self.current_block_cond = None
        self.results = []
        self.key_bindings = {}
        self.current_pd_mean = 0.0
        self.current_pd_std = 0.0
        
        self.calibration_eyes = ["Right", "Left"]
        self.current_calib_eye_idx = 0
        self.calib_results = {}
        
        self.knob_val = 0.0
        self.photo_ref = None
        self.photo_test = None

        # --- キャリブレーションデータのロード ---
        fg_calib_dir = os.path.join(lab_root, "results", "tables", "DisplayBrightness", "fg_calibration_log")
        bg_calib_dir = os.path.join(lab_root, "results", "tables", "DisplayBrightness", "bg_calibration_log")
        
        self.fg_lums, self.fg_pixels = self.load_calibration_data(fg_calib_dir)
        self.bg_lums, self.bg_pixels = self.load_calibration_data(bg_calib_dir)
        
        if self.fg_lums is None or self.bg_lums is None:
            print("Warning: Calibration data not found. Linear mapping will be used.")
            self.fg_lums = np.array([0.0, 100.0])
            self.fg_pixels = np.array([0, 255])
            self.bg_lums = np.array([0.0, 100.0])
            self.bg_pixels = np.array([0, 255])

        # --- ウィンドウのセットアップ ---
        # プライマリモニタのスクリーンサイズを取得
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        
        # Window 2 (実験者用操作画面): 最大化表示
        self.root.state('zoomed')

        # Window 1 (被験者用表示画面): スクリーンの右半分に表示
        self.win1 = tk.Toplevel(self.root)
        self.win1.title("Display (Window 1)")
        
        # 確実にサブディスプレイ（メインの右側）で開くように、初期位置をメイン画面の幅の分だけ右にオフセットする
        self.win1.geometry(f"+{screen_w}+0")
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
        """ステップ0-1: ID入力UIを構築し表示する"""
        # UI要素をまとめるためのフレーム
        self.participant_frame = tk.Frame(self.root, bg='gray', padx=20, pady=20)
        self.participant_frame.place(relx=0.5, rely=0.5, anchor='center')

        tk.Label(self.participant_frame, text="Enter Participant ID", font=("Arial", 16)).grid(row=0, column=0, columnspan=2, pady=10)

        tk.Label(self.participant_frame, text="Participant ID:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        entry_id = tk.Entry(self.participant_frame, textvariable=self.participant_id)
        entry_id.grid(row=1, column=1, padx=5, pady=5)
        entry_id.focus_set()

        btn = tk.Button(self.participant_frame, text="Next", command=self.check_participant_id)
        btn.grid(row=2, column=0, columnspan=2, pady=20)
        btn.bind('<Return>', lambda event: self.check_participant_id())
        entry_id.bind('<Return>', lambda event: self.check_participant_id())

    def check_participant_id(self, event=None):
        pid = self.participant_id.get().strip()
        if not pid:
            messagebox.showwarning("Input Error", "Please enter a Participant ID.")
            return
        
        filepath = os.path.join(PARTICIPANT_DATA_DIR, "participants.csv")
        found = False
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["ID"] == pid:
                        self.participant_age.set(row["Age"])
                        self.participant_gender.set(row["Gender"])
                        self.participant_ipd.set(row["IPD"])
                        self.participant_dominance.set(row.get("Dominance", "Right"))
                        found = True
                        break
        
        self.participant_frame.destroy()
        
        if found:
            self.start_calibration_sequence()
        else:
            self.setup_new_participant_ui()

    def setup_new_participant_ui(self):
        """ステップ0-2: 新規被験者の情報入力UI"""
        self.participant_frame = tk.Frame(self.root, bg='gray', padx=20, pady=20)
        self.participant_frame.place(relx=0.5, rely=0.5, anchor='center')

        tk.Label(self.participant_frame, text=f"New Participant Registration (ID: {self.participant_id.get()})", font=("Arial", 16)).grid(row=0, column=0, columnspan=2, pady=10)

        tk.Label(self.participant_frame, text="Age:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        entry_age = tk.Entry(self.participant_frame, textvariable=self.participant_age)
        entry_age.grid(row=1, column=1, padx=5, pady=5)
        entry_age.focus_set()

        tk.Label(self.participant_frame, text="Gender:").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        gender_combo = ttk.Combobox(self.participant_frame, textvariable=self.participant_gender, values=["Male", "Female", "Other"])
        gender_combo.grid(row=2, column=1, padx=5, pady=5)
        gender_combo.set("Male")

        tk.Label(self.participant_frame, text="IPD (mm):").grid(row=3, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(self.participant_frame, textvariable=self.participant_ipd).grid(row=3, column=1, padx=5, pady=5)

        tk.Label(self.participant_frame, text="Eye Dominance:").grid(row=4, column=0, sticky='w', padx=5, pady=5)
        dom_combo = ttk.Combobox(self.participant_frame, textvariable=self.participant_dominance, values=["Right", "Left"])
        dom_combo.grid(row=4, column=1, padx=5, pady=5)
        dom_combo.set("Right")

        btn = tk.Button(self.participant_frame, text="Register and Next", command=self.register_and_start)
        btn.grid(row=5, column=0, columnspan=2, pady=20)
        btn.bind('<Return>', lambda event: self.register_and_start())
        
    def register_and_start(self, event=None):
        if not self.participant_age.get() or not self.participant_ipd.get():
            messagebox.showwarning("Input Error", "Please enter Age and IPD.")
            return
        self.save_participant_data()
        self.participant_frame.destroy()
        self.start_calibration_sequence()

    def save_participant_data(self):
        if not os.path.exists(PARTICIPANT_DATA_DIR):
            os.makedirs(PARTICIPANT_DATA_DIR)
        filepath = os.path.join(PARTICIPANT_DATA_DIR, "participants.csv")
        fieldnames = ["ID", "Age", "Gender", "IPD", "Dominance"]
        rows = []
        found = False
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r["ID"] == self.participant_id.get():
                        r["Age"] = self.participant_age.get()
                        r["Gender"] = self.participant_gender.get()
                        r["IPD"] = self.participant_ipd.get()
                        r["Dominance"] = self.participant_dominance.get()
                        found = True
                    rows.append(r)
        if not found:
            rows.append({
                "ID": self.participant_id.get(), "Age": self.participant_age.get(),
                "Gender": self.participant_gender.get(), "IPD": self.participant_ipd.get(),
                "Dominance": self.participant_dominance.get()
            })
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def start_calibration_sequence(self):
        # Window 1 (被験者用画面) の実際のサイズを取得して更新する
        self.win1.update_idletasks()
        self.width = self.win1.winfo_width()
        self.height = self.win1.winfo_height()

        self.calibration_eyes = ["Right", "Left"]
        self.current_calib_eye_idx = 0
        self.calib_results = {}
        self.start_eye_calibration()
        
    def start_eye_calibration(self):
        if self.current_calib_eye_idx >= len(self.calibration_eyes):
            self.setup_experiment_blocks()
            return
            
        current_eye = self.calibration_eyes[self.current_calib_eye_idx]
        messagebox.showinfo("Calibration", f"Next: Calibration for {current_eye} Eye.\nPlease cover the other eye.")
        self.offset_x.set(0)
        self.offset_y.set(0)
        self.pupil_diameter_val.set(4.0)
        self.setup_calibration_ui(is_new_eye=True)

    def setup_experiment_blocks(self):
        dom_eye = self.participant_dominance.get()
        if dom_eye not in self.calib_results:
            dom_eye = "Right"
        self.offset_x.set(self.calib_results[dom_eye]["offset_x"])
        self.offset_y.set(self.calib_results[dom_eye]["offset_y"])
        self.current_pd_mean = self.calib_results[dom_eye]["pd_mean"]
        
        conditions = ["Single plane", "Single plane + defocus simulation", "OST-AR (dual plane)"]
        ocularities = ["monocular", "binocular"]
        self.blocks = []
        for cond in conditions:
            for oc in ocularities:
                self.blocks.append({"condition": cond, "ocularity": oc})
        
        # ブロックの順番をランダムにシャッフル
        random.shuffle(self.blocks)

        self.current_block_index = 0
        self.current_trial_in_experiment = 0
        self.start_block()

    def start_block(self):
        """新しいブロックを開始する"""
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        
        if self.current_block_index >= len(self.blocks):
            self.finish_experiment()
            return
            
        self.current_block_cond = self.blocks[self.current_block_index]
        self.current_trial_in_block = 0
        
        self.setup_block_confirmation_ui()

    def setup_block_confirmation_ui(self):
        """ブロック開始前の確認画面を表示する"""
        if hasattr(self, 'ctrl_frame') and self.ctrl_frame.winfo_exists():
            self.ctrl_frame.destroy()
        self.canvas1.delete("all")
        self.canvas2.delete("all")

        # Clear previous key bindings
        for key, binding_id in self.key_bindings.items():
            self.root.unbind(key, binding_id)
        self.key_bindings.clear()

        self.ctrl_frame = tk.Frame(self.root, bg='gray')
        self.ctrl_frame.place(relx=0.5, rely=0.5, anchor='center')

        cond = self.current_block_cond["condition"]
        oc = self.current_block_cond["ocularity"]
        dom_eye = self.participant_dominance.get()
        
        if oc == "monocular":
            eye_inst = f"Please use your DOMINANT eye ({dom_eye}) and COVER the other eye."
        else:
            eye_inst = "Please use BOTH eyes."
            
        instruction_text = f"[Block {self.current_block_index + 1}/{len(self.blocks)}]\n" \
                           f"Condition: {cond}\nOcularity: {oc}\n\n" \
                           f"{eye_inst}\n\nPress 'Enter' to start."
        
        tk.Label(self.ctrl_frame, text=instruction_text, 
                 bg='gray', fg='white', font=("Arial", 16)).pack(pady=20, padx=40)

        btn = tk.Button(self.ctrl_frame, text="OK", command=self._start_calibration_from_confirmation, font=("Arial", 14))
        btn.pack(pady=10)
        btn.focus_set()
        self.key_bindings['<Return>'] = self.root.bind('<Return>', lambda event: self._start_calibration_from_confirmation())

    def _start_calibration_from_confirmation(self):
        for key, binding_id in self.key_bindings.items():
            self.root.unbind(key, binding_id)
        self.key_bindings.clear()
        
        if hasattr(self, 'ctrl_frame') and self.ctrl_frame.winfo_exists():
            self.ctrl_frame.destroy()
        self.setup_calibration_ui()

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
    def load_calibration_data(self, log_dir):
        lum_pixel_data = {}
        csv_files = glob.glob(os.path.join(log_dir, "*.csv"))
        for csv_file in csv_files:
            try:
                with open(csv_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            t_lum = float(row["Target_Luminance(cd/m2)"])
                            p_val = int(row["Pixel_Value"])
                            lum_pixel_data.setdefault(t_lum, []).append(p_val)
                        except:
                            pass
            except:
                pass
        
        avg_map = []
        for t_lum, p_list in sorted(lum_pixel_data.items()):
            if p_list:
                avg_map.append((t_lum, int(np.round(np.mean(p_list)))))
                
        if not avg_map:
            return None, None
            
        lums = np.array([x[0] for x in avg_map])
        pixels = np.array([x[1] for x in avg_map])
        return lums, pixels

    def update_calibration_view(self, *args):
        """キャリブレーション画面の表示を更新する (スライダー操作時に呼ばれる)"""
        # 既存のマーカーを一旦すべて削除
        self.canvas1.delete("calib")
        self.canvas2.delete("calib")

        d_fg = self.distance1
        d_bg = self.distance2

        # --- 前景マーカーのサイズ計算 (正方形) ---
        fg_marker_size = stimuli_utils.get_size_for_visual_angle(d_fg, VISUAL_ANGLE_DEG)
        
        # --- 背景マーカーのサイズ計算 (横長) ---
        # 背景画像は前景の2倍の幅を持つ
        bg_marker_h = stimuli_utils.get_size_for_visual_angle(d_bg, VISUAL_ANGLE_DEG)
        bg_marker_w = stimuli_utils.get_size_for_visual_angle(d_bg, VISUAL_ANGLE_DEG * 2) # 幅は視角2倍で計算

        # --- Window 1 (被験者側) のマーカー描画 ---
        # 1. 背景全体（横長）の四隅にマーカーを描画
        stimuli_utils.draw_image_corner_brackets(self.canvas1, bg_marker_w, bg_marker_h, self.offset_x.get(), self.offset_y.get(), color=WIN1_MARKER_COLOR, line_width=stimuli_utils.MARKER_LINE_WIDTH * 1.5)
        # 2. 背景の中央に、前景と同じサイズの正方形マーカーを描画
        stimuli_utils.draw_image_corner_brackets(self.canvas1, bg_marker_h, bg_marker_h, self.offset_x.get(), self.offset_y.get(), color=WIN1_MARKER_COLOR, line_width=stimuli_utils.MARKER_LINE_WIDTH * 1.5)
        
        # --- Window 2 (実験者側) のマーカー描画 ---
        # 基準となる前景サイズのマーカーと十字を描画
        stimuli_utils.draw_image_corner_brackets(self.canvas2, fg_marker_size, fg_marker_size, 0, 0, color=WIN2_MARKER_COLOR, flip_x=False, line_width=stimuli_utils.MARKER_LINE_WIDTH)
        stimuli_utils.draw_center_cross(self.canvas2, color=WIN2_MARKER_COLOR)

    def adjust_offset(self, dx, dy):
        """矢印キーによるオフセット調整用関数"""
        self.offset_x.set(self.offset_x.get() + dx)
        self.offset_y.set(self.offset_y.get() + dy)
        self.update_calibration_view()
        return "break" # デフォルトのイベント処理（スライダーの移動など）を停止する

    def setup_calibration_ui(self, is_break=False, is_new_eye=False):
        """ステップ1: キャリブレーション用UIを構築し表示する"""
        self.update_calibration_view()
        
        if hasattr(self, 'ctrl_frame') and self.ctrl_frame.winfo_exists():
            self.ctrl_frame.destroy()
            
        # 操作用UIをまとめるためのフレーム
        self.ctrl_frame = tk.Frame(self.root, bg='gray')
        self.ctrl_frame.place(relx=0.5, rely=0.8, anchor='center')

        # Clear previous key bindings
        for key, binding_id in self.key_bindings.items():
            self.root.unbind(key, binding_id)
        self.key_bindings.clear()

        if is_break:
            instruction_text = "This is a break. You can adjust the position if needed.\nPress 'Resume Experiment' to continue."
            button_text = "Resume Experiment"
            button_command = self.resume_experiment
        elif is_new_eye:
            eye = self.calibration_eyes[self.current_calib_eye_idx]
            instruction_text = f"[{eye} Eye Calibration]\nUse the arrow keys to adjust the position of the red frame."
            button_text = "Calibration Done, Next"
            button_command = lambda: defocus_matching.setup_defocus_matching_ui(self)
        else:
            instruction_text = "Use the arrow keys to adjust the position of the red frame."
            button_text = "Calibration Done, Start Block"
            button_command = self.start_experiment_block

        # 位置調整の指示ラベル
        tk.Label(self.ctrl_frame, text=instruction_text, bg='gray', fg='white', font=("Arial", 12)).pack(pady=10, padx=20)

        # ボタン
        btn = tk.Button(self.ctrl_frame, text=button_text, command=button_command)
        btn.pack(pady=10)
        
        self.key_bindings['<Return>'] = self.root.bind('<Return>', lambda event: button_command())
        self.key_bindings['<Left>'] = self.root.bind('<Left>', lambda e: self.adjust_offset(-1, 0))
        self.key_bindings['<Right>'] = self.root.bind('<Right>', lambda e: self.adjust_offset(1, 0))
        self.key_bindings['<Up>'] = self.root.bind('<Up>', lambda e: self.adjust_offset(0, -1))
        self.key_bindings['<Down>'] = self.root.bind('<Down>', lambda e: self.adjust_offset(0, 1))
        self.root.focus_set()

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
    # def setup_defocus_matching_ui(self):  # 削除済み、defocus_matching.py に移動
    # ... (methods moved to defocus_matching.py)

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

    def start_experiment_block(self):
        """ブロックの実験ループを開始する"""
        # Clear key bindings from defocus matching phase
        for key, binding_id in self.key_bindings.items():
            self.root.unbind(key, binding_id)
        self.key_bindings.clear()

        block_trials = []
        ref_contrasts = [0.5, 0.2]
        orientations = [0, 90]
        for ref_c in ref_contrasts:
            for ori in orientations:
                for _ in range(NUM_REPETITIONS):
                    block_trials.append({"ref_contrast": ref_c, "orientation": ori})

        random.shuffle(block_trials)
        self.trial_list = block_trials
            
        # --- 5. Start the experiment ---
        self.ctrl_frame.destroy() # キャリブレーションUIを削除
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        self.run_trial()
        
    def run_trial(self):
        """1試行分の実験シーケンスを実行する"""
        trial_cond = self.trial_list[self.current_trial_in_block]
        ref_c = trial_cond["ref_contrast"]
        
        # 初期コントラストを0.0〜1.0の間でランダムに設定
        self.init_contrast = random.uniform(0.0, 1.0)
        
        self.canvas1.configure(bg='black')
        self.canvas2.configure(bg='black')
        
        d_fg, d_bg = self.distance1, self.distance2
        ppd_fg = stimuli_utils.PIXELS_PER_CM * d_fg * math.tan(math.radians(1.0))
        ppd_bg = stimuli_utils.PIXELS_PER_CM * d_bg * math.tan(math.radians(1.0))
        
        width_deg = 7.9
        height_deg = 3.95
        width_fg = int(width_deg * ppd_fg)
        height_fg = int(height_deg * ppd_fg)
        width_bg = int(width_deg * ppd_bg)
        height_bg = int(height_deg * ppd_bg)
        
        ori = trial_cond["orientation"]
        cond = self.current_block_cond["condition"]
        
        # 試行ごとのベースモジュレーション(-1~1)を生成してキャッシュ
        self.gabor_base = stimuli_utils.create_gabor_base(width_fg, height_fg, ppd_fg, self.spatial_freq, orientation=ori)
        
        if cond == "OST-AR (dual plane)":
            self.noise_base = stimuli_utils.create_noise_base(width_bg, height_bg, ppd_bg, self.spatial_freq)
        else:
            self.noise_base = stimuli_utils.create_noise_base(width_fg, height_fg, ppd_fg, self.spatial_freq)
        
        # コントラストマッチング用スライダーUIをセットアップ
        self.setup_contrast_matching_ui()

    def setup_contrast_matching_ui(self):
        """コントラストマッチング用UIを構築"""
        if hasattr(self, 'ctrl_frame') and self.ctrl_frame.winfo_exists():
            self.ctrl_frame.destroy()
        
        for key, binding_id in self.key_bindings.items():
            self.root.unbind(key, binding_id)
        self.key_bindings.clear()
        
        # 操作用UIフレーム
        self.ctrl_frame = tk.Frame(self.root, bg='gray')
        self.ctrl_frame.place(relx=0.5, rely=0.8, anchor='center')
        
        # スライダー (0.0 - 1.0, 左が1.0, 右が0.0)
        self.contrast_val = tk.DoubleVar(value=self.init_contrast)
        slider = tk.Scale(self.ctrl_frame, from_=1.0, to=0.0, resolution=0.01, orient=tk.HORIZONTAL,
                          length=400, variable=self.contrast_val, showvalue=0, command=lambda *args: self.update_stimuli())
        slider.pack(pady=10)
        
        # 決定ボタン
        btn = tk.Button(self.ctrl_frame, text="Next Trial", command=self.save_and_next)
        btn.pack(pady=10)
        btn.focus_set()
        self.key_bindings['<Down>'] = self.root.bind('<Down>', lambda event: self.save_and_next())
        
        # 指示
        instruction_text = "Adjust the slider (Left/Right arrow keys) to match the contrast.\nPress 'Down' arrow to confirm."
        tk.Label(self.ctrl_frame, text=instruction_text,
                 bg='gray', fg='white', font=("Arial", 12)).pack(pady=10, padx=20)
        
        # Bind keys for contrast adjustment
        self.key_bindings['<Left>'] = self.root.bind('<Left>', lambda e: self._handle_contrast_key_press(e))
        self.key_bindings['<Right>'] = self.root.bind('<Right>', lambda e: self._handle_contrast_key_press(e))
        self.root.focus_set()
        
        # 初回表示
        self.update_stimuli()
    
    def _handle_contrast_key_press(self, event):
        """コントラスト調整用キープレスハンドラ"""
        step = 0.01
        current_val = self.contrast_val.get()
        min_val = 0.0
        max_val = 1.0
        
        if event.keysym == 'Left':  # 左矢印: 値を減少させる (1.0->0.0方向) -> スライダーが右に移動
            new_val = max(min_val, current_val - step)
            self.contrast_val.set(new_val)
        elif event.keysym == 'Right':  # 右矢印: 値を増加させる (0.0->1.0方向) -> スライダーが左に移動
            new_val = min(max_val, current_val + step)
            self.contrast_val.set(new_val)
        
        self.update_stimuli()
        return "break"

    def update_stimuli(self):
        # スライダーがある場合はそこからコントラスト値を取得
        if hasattr(self, 'contrast_val'):
            c_test = self.contrast_val.get()
        else:
            c_test = self.init_contrast
        trial = self.trial_list[self.current_trial_in_block]
        cond = self.current_block_cond["condition"]
        ref_c = trial["ref_contrast"]
        d_fg, d_bg = self.distance1, self.distance2
        ppd_fg = stimuli_utils.PIXELS_PER_CM * d_fg * math.tan(math.radians(1.0))
        ppd_bg = stimuli_utils.PIXELS_PER_CM * d_bg * math.tan(math.radians(1.0))
        
        gap_y_deg = 2.0
        gap_y_fg = int(gap_y_deg * ppd_fg)
        gap_y_bg = int(gap_y_deg * ppd_bg)
        
        blur_sigma = 0.0
        if cond == "Single plane + defocus simulation":
            D = abs(1/0.5 - 1/0.77)
            bd_deg = D * self.current_pd_mean * (180.0 / math.pi) / 1000.0
            blur_sigma = 0.55 * bd_deg / 2.0 * ppd_fg
            
        self.canvas1.delete("stim")
        self.canvas2.delete("stim")
        self.canvas1.delete("calib")
        self.canvas2.delete("calib")
        
        cx1, cy1 = self.width//2 + self.offset_x.get(), self.height//2 + self.offset_y.get()
        cx2, cy2 = self.canvas2.winfo_width()//2, self.canvas2.winfo_height()//2
        
        # 実験での物理輝度設定 (cd/m^2)
        L_fg = 35.0
        L_bg = 15.0
        C_bg = 1.0
        
        # Ref (Foreground only, Top)
        lum_ref_fg = L_fg * (1.0 + ref_c * self.gabor_base)
        if cond == "Single plane + defocus simulation" and blur_sigma > 0:
            img_f = Image.fromarray(lum_ref_fg.astype(np.float32), mode='F')
            img_f = img_f.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
            lum_ref_fg = np.array(img_f)
        pix_ref_fg = np.interp(lum_ref_fg, self.fg_lums, self.fg_pixels).astype(np.uint8)
        img_ref_fg = Image.fromarray(pix_ref_fg, mode='L')
        self.photo_ref_fg = ImageTk.PhotoImage(img_ref_fg)
        
        if cond == "OST-AR (dual plane)":
            # Test: Gabor on Foreground, Noise on Background (Bottom)
            lum_test_fg = L_fg * (1.0 + c_test * self.gabor_base)
            pix_test_fg = np.interp(lum_test_fg, self.fg_lums, self.fg_pixels).astype(np.uint8)
            img_test_fg = Image.fromarray(pix_test_fg, mode='L')
            
            lum_noise_bg = L_bg * (1.0 + C_bg * self.noise_base)
            pix_noise_bg = np.interp(lum_noise_bg, self.bg_lums, self.bg_pixels).astype(np.uint8)
            img_noise_bg = Image.fromarray(pix_noise_bg, mode='L')
            
            self.photo_test_fg = ImageTk.PhotoImage(img_test_fg)
            self.photo_noise_bg = ImageTk.PhotoImage(img_noise_bg)
            
            # Draw
            self.canvas2.create_image(cx2, cy2 - gap_y_fg, image=self.photo_ref_fg, anchor='center', tags="stim")
            
            self.canvas1.create_image(cx1, cy1 + gap_y_bg, image=self.photo_noise_bg, anchor='center', tags="stim")
            self.canvas2.create_image(cx2, cy2 + gap_y_fg, image=self.photo_test_fg, anchor='center', tags="stim")
        else:
            # Single plane (Bottom)
            lum_noise = L_bg * (1.0 + C_bg * self.noise_base)
            
            lum_test_fg = L_fg * (1.0 + c_test * self.gabor_base)
            if cond == "Single plane + defocus simulation" and blur_sigma > 0:
                img_f = Image.fromarray(lum_test_fg.astype(np.float32), mode='F')
                img_f = img_f.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
                lum_test_fg = np.array(img_f)
                
            lum_test_total = lum_noise + lum_test_fg
            pix_test = np.interp(lum_test_total, self.fg_lums, self.fg_pixels).astype(np.uint8)
            img_test = Image.fromarray(pix_test, mode='L')
            
            self.photo_test = ImageTk.PhotoImage(img_test)
            
            # Draw
            self.canvas2.create_image(cx2, cy2 - gap_y_fg, image=self.photo_ref_fg, anchor='center', tags="stim")
            self.canvas2.create_image(cx2, cy2 + gap_y_fg, image=self.photo_test, anchor='center', tags="stim")

    def save_and_next(self, event=None):
        """評価データを保存し、次の試行に進む"""
        # UIをクリア
        if hasattr(self, 'ctrl_frame') and self.ctrl_frame.winfo_exists():
            self.ctrl_frame.destroy()
        
        for key, binding_id in self.key_bindings.items():
            self.root.unbind(key, binding_id)
        self.key_bindings.clear()
        
        # スライダーがある場合はそこからコントラスト値を取得
        if hasattr(self, 'contrast_val'):
            c_test = self.contrast_val.get()
        else:
            c_test = self.init_contrast
        trial = self.trial_list[self.current_trial_in_block]
        
        right_res = self.calib_results.get("Right", {"offset_x": 0, "offset_y": 0, "pd_mean": 0})
        left_res = self.calib_results.get("Left", {"offset_x": 0, "offset_y": 0, "pd_mean": 0})

        self.results.append([
            self.participant_id.get(), self.participant_age.get(), self.participant_gender.get(), 
            self.participant_ipd.get(), self.participant_dominance.get(),
            self.current_block_index + 1, self.current_block_cond["condition"], self.current_block_cond["ocularity"],
            self.current_trial_in_experiment + 1,
            trial["orientation"], trial["ref_contrast"], round(c_test, 4),
            right_res["pd_mean"], right_res["offset_x"], right_res["offset_y"],
            left_res["pd_mean"], left_res["offset_x"], left_res["offset_y"]
        ])
        
        self.current_trial_in_block += 1
        self.current_trial_in_experiment += 1
        
        if self.current_trial_in_block >= len(self.trial_list):
            self.current_block_index += 1
            if self.current_block_index >= len(self.blocks):
                self.finish_experiment()
            elif self.current_block_index % 2 == 0:
                self.root.after(500, self.start_break)
            else:
                self.root.after(500, self.start_block)
        else:
            self.root.after(500, self.run_trial)

    def finish_experiment(self):
        """実験終了処理。結果をCSVファイルに保存する"""
        if not os.path.exists(RESULT_DIR):
            os.makedirs(RESULT_DIR)
            
        # IDと日付でフォルダを作成 (id_YYYYMMDD)
        p_id = self.participant_id.get()
        now = datetime.datetime.now()
        date_str = now.strftime("%Y%m%d")
        save_folder = os.path.join(RESULT_DIR, f"{p_id}_{date_str}")
        
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        # ファイル名に被験者IDと現在時刻を含める
        filename = os.path.join(save_folder, f"result_{p_id}_{now.strftime('%Y%m%d_%H%M%S')}.csv")
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = [
                "ID", "Age", "Gender", "IPD(mm)", "Dominance", 
                "Block_ID", "Condition", "Ocularity", "Trial_ID", 
                "Orientation", "Ref_Contrast", "Matched_Contrast",
                "PD_Right", "OffsetX_Right", "OffsetY_Right", "PD_Left", "OffsetX_Left", "OffsetY_Left"
            ]
            writer.writerow(header)
            writer.writerows(self.results)
            
        messagebox.showinfo("Finished", f"Experiment finished.\nData saved to: {filename}")
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ExperimentApp(root)
    root.mainloop()