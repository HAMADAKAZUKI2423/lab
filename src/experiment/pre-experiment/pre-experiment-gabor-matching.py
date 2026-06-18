"""
Gabor Matching 実験スクリプト（削減版）

前景と背景のコントラストマッチングを行う実験
基盤クラス（ExperimentBaseUI, ExperimentTrialLoop）から共通処理を継承
matching固有のロジックのみを実装
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os
import csv
import datetime
import random
import math
import numpy as np
import glob
from PIL import Image, ImageTk
import json

# 基盤クラスと共通ユーティリティをインポート
from experiment_base_ui import ExperimentBaseUI
from experiment_trial_loop import ExperimentTrialLoop
import stimuli_utils
import defocus_matching


# 固定色補正行列（Window 2 に表示される画像に適用）
COLOR_MATRIX = np.array([
    [ 0.33169778,  0.01128241,  0.0258315 ],
    [-0.00844114,  0.41731136,  0.01354067],
    [-0.0107871 , -0.04633671,  0.55739266]
], dtype=np.float32)


# ==========================================
# 定数設定エリア (実験条件やデザインはここを変更)
# ==========================================
VISUAL_ANGLE_DEG = 7.9
NUM_REPETITIONS = 5
SPATIAL_FREQ = 4  # Spatial frequency in cpd
PUPIL_DIAMETER_MM = 4.0

script_dir = os.path.dirname(os.path.abspath(__file__))
lab_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

RESULT_DIR = os.path.join(lab_root, "results", "tables", "pre-experiment-matching")
FIGURE_DIR = os.path.join(lab_root, "results", "figures", "pre-experiment-matching")
PARTICIPANT_DATA_DIR = os.path.join(lab_root, "data", "processed", "tables", "pre-experiment-matching")

for dir_path in [RESULT_DIR, FIGURE_DIR, PARTICIPANT_DATA_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

BG_COLOR = 'black'
WIN1_MARKER_COLOR = 'red'
WIN2_MARKER_COLOR = 'white'

# Load experiment config (falls back to defaults)
import sys
sys.path.insert(0, os.path.join(lab_root, 'src', 'experiment'))
try:
    import experiment_config
    CFG = experiment_config.get_config()
except Exception:
    CFG = {}

BG_COLOR = CFG.get('BG_COLOR', BG_COLOR)


class ExperimentApp(ExperimentBaseUI, ExperimentTrialLoop):
    """
    Gabor Matching 実験
    
    基盤クラスから継承：
    - ExperimentBaseUI: UI処理（参加者情報、キャリブレーション）
    - ExperimentTrialLoop: 試行ループ（フェーズ管理、結果保存）
    
    matching固有の処理のみを実装
    """
    
    def __init__(self, root):
        # 基盤クラスの初期化
        ExperimentBaseUI.__init__(self, root, PARTICIPANT_DATA_DIR)
        ExperimentTrialLoop.__init__(self)
        
        self.root = root
        self.root.title("Gabor Matching Experiment - Controller (Window 2)")
        self.root.configure(bg=BG_COLOR)
        
        # matching固有の変数
        self.distance1 = 50  # Foreground distance (cm)
        self.distance2 = 100  # Background distance (cm)
        self.spatial_freq = SPATIAL_FREQ
        self.pupil_diameter_val = tk.DoubleVar(value=4.0)
        
        self.calibration_eyes = ["Right", "Left"]
        self.current_calib_eye_idx = 0
        self.calib_results = {}
        
        self.blocks = []
        self.current_block_index = 0
        self.current_trial_in_block = 0
        self.current_block_cond = None
        
        self.current_pd_mean = 0.0
        self.current_pd_std = 0.0
        
        self.participant_dominance = tk.StringVar(value="Right")
        
        # キャリブレーションデータ
        fg_calib_dir = os.path.join(lab_root, "results", "tables", "DisplayBrightness", "fg_calibration_log")
        bg_calib_dir = os.path.join(lab_root, "results", "tables", "DisplayBrightness", "bg_calibration_log")
        
        self.fg_lums, self.fg_pixels = stimuli_utils.load_calibration_data(fg_calib_dir)
        self.bg_lums, self.bg_pixels = stimuli_utils.load_calibration_data(bg_calib_dir)
        self.color_matrix = COLOR_MATRIX
        
        if self.fg_lums is None or self.bg_lums is None:
            print("Warning: Calibration data not found. Linear mapping will be used.")
            self.fg_lums = np.array([0.0, 100.0])
            self.fg_pixels = np.array([0, 255])
            self.bg_lums = np.array([0.0, 100.0])
            self.bg_pixels = np.array([0, 255])

        # Apply config overrides
        self.config = CFG or {}
        self.L_fg = float(self.config.get('L_fg', 15.0))
        self.L_bg = float(self.config.get('L_bg', 15.0))
        self.L_ref = float(self.config.get('L_ref', 30.0))
        self.distance1 = int(self.config.get('DISTANCE_FG', self.distance1))
        self.distance2 = int(self.config.get('DISTANCE_BG', self.distance2))
        
        # Window1とcanvasのセットアップ
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        
        self.root.state('zoomed')
        
        self.win1 = tk.Toplevel(self.root)
        self.win1.title("Gabor Matching Experiment - Display (Window 1)")
        self.win1.geometry(f"+{screen_w}+0")
        self.win1.state('zoomed')
        
        self.root.update_idletasks()
        self.width = screen_w
        self.height = screen_h
        
        self.canvas2 = tk.Canvas(self.root, width=self.width, height=self.height, 
                                bg=BG_COLOR, highlightthickness=0)
        self.canvas2.pack(fill="both", expand=True)
        
        self.win1.configure(bg=BG_COLOR)
        self.canvas1 = tk.Canvas(self.win1, width=self.width, height=self.height, 
                                bg=BG_COLOR, highlightthickness=0)
        self.canvas1.pack(fill="both", expand=True)
        
        # 試行ループの初期化
        self.result_dir = RESULT_DIR
        self.setup_trial_phases(time_phase1=1600, time_isi=1, time_phase2=5000)
        
        # UIを開始
        self.setup_participant_info_ui()
    
    # ========== 基盤クラスから継承したメソッドを実装 ==========
    
    def setup_participant_info_ui(self):
        """ステップ0-1: ID入力UIを構築し表示する"""
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
        self.win1.update_idletasks()
        self.width = self.win1.winfo_width()
        self.height = self.win1.winfo_height()

        self.calibration_eyes = ["Right", "Left"]
        self.current_calib_eye_idx = 0
        self.calib_results = {}
        self.start_eye_calibration()

    def on_participant_confirmed(self):
        """参加者確認後、キャリブレーション開始"""
        self.win1.update_idletasks()
        self.width = self.win1.winfo_width()
        self.height = self.win1.winfo_height()
        
        self.calibration_eyes = ["Right", "Left"]
        self.current_calib_eye_idx = 0
        self.calib_results = {}
        self.start_eye_calibration()
    
    def save_preview_images(self):
        """デフォーカスの効き方などの確認用画像を保存する"""
        now = datetime.datetime.now()
        date_str = now.strftime("%Y%m%d_%H%M%S")
        p_id = self.participant_id.get()
        save_dir = os.path.join(FIGURE_DIR, f"{p_id}_{date_str}", "stimuli")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        d_fg, d_bg = self.distance1, self.distance2
        ppd_fg = stimuli_utils.PIXELS_PER_CM * d_fg * math.tan(math.radians(1.0))
        
        width_deg = 7.9
        height_deg = 3.95
        width_fg = int(width_deg * ppd_fg)
        height_fg = int(height_deg * ppd_fg)
        
        ori = 0
        gabor_base = stimuli_utils.create_gabor_base(width_fg, height_fg, ppd_fg, self.spatial_freq, orientation=ori)
        noise_base = stimuli_utils.create_noise_base(width_fg, height_fg, ppd_fg, self.spatial_freq)
        
        L_fg = 15.0
        L_bg = 15.0
        C_bg = 1.0
        L_ref = 30.0
        
        # Reference Gabor patches
        for ref_c in [0.0,0.2, 0.4]:
            lum_ref_fg = L_ref * (1.0 + ref_c * gabor_base)
            pil_ref = stimuli_utils.lum_to_pil(lum_ref_fg, self.fg_lums, self.fg_pixels)
            pil_ref.save(os.path.join(save_dir, f"ref_gabor_contrast_{ref_c}.png"))
            
        # Single plane stimulus
        c_test = 0.0
        lum_test_fg = L_fg * (1.0 + c_test * gabor_base)
        pil_test_fg = stimuli_utils.lum_to_pil(lum_test_fg, self.fg_lums, self.fg_pixels)
        pil_test_fg.save(os.path.join(save_dir, "single_plane_foreground.png"))

        lum_noise = L_bg * (1.0 + C_bg * noise_base)
        pil_noise = stimuli_utils.lum_to_pil(lum_noise, self.fg_lums, self.fg_pixels)
        pil_noise.save(os.path.join(save_dir, "single_plane_background.png"))

        lum_total = lum_noise + lum_test_fg
        pil_total = stimuli_utils.lum_to_pil(lum_total, self.fg_lums, self.fg_pixels)
        pil_total.save(os.path.join(save_dir, "single_plane_combined.png"))

    def setup_experiment_blocks(self):
        dom_eye = self.participant_dominance.get()
        if dom_eye not in self.calib_results:
            dom_eye = "Right"
        self.offset_x.set(self.calib_results[dom_eye]["offset_x"])
        self.offset_y.set(self.calib_results[dom_eye]["offset_y"])
        self.current_pd_mean = self.calib_results[dom_eye]["pd_mean"]
        
        self.save_preview_images()
        
        # Setup experiment blocks
        """キャリブレーション画面の更新"""
        self.canvas1.delete("calib")
        self.canvas2.delete("calib")
        
        fg_marker_size = stimuli_utils.get_size_for_visual_angle(self.distance1, VISUAL_ANGLE_DEG)
        bg_marker_h = stimuli_utils.get_size_for_visual_angle(self.distance2, VISUAL_ANGLE_DEG)
        bg_marker_w = stimuli_utils.get_size_for_visual_angle(self.distance2, VISUAL_ANGLE_DEG * 2)
        
        # Window 1
        stimuli_utils.draw_image_corner_brackets(
            self.canvas1, bg_marker_w, bg_marker_h, 
            self.offset_x.get(), self.offset_y.get(), 
            color=WIN1_MARKER_COLOR, line_width=stimuli_utils.MARKER_LINE_WIDTH * 1.5
        )
        stimuli_utils.draw_image_corner_brackets(
            self.canvas1, bg_marker_h, bg_marker_h, 
            self.offset_x.get(), self.offset_y.get(), 
            color=WIN1_MARKER_COLOR, line_width=stimuli_utils.MARKER_LINE_WIDTH * 1.5
        )
        
        # Window 2
        stimuli_utils.draw_image_corner_brackets(
            self.canvas2, fg_marker_size, fg_marker_size, 
            0, 0, color=WIN2_MARKER_COLOR, line_width=stimuli_utils.MARKER_LINE_WIDTH
        )
        stimuli_utils.draw_center_cross(self.canvas2, color=WIN2_MARKER_COLOR)
    
    def on_calibration_complete(self):
        """キャリブレーション完了"""
        if self.current_calib_eye_idx < len(self.calibration_eyes):
            # デフォーカスマッチングへ
            defocus_matching.setup_defocus_matching_ui(self)
        else:
            # 実験ブロックへ
            self.setup_experiment_blocks()
    
    # ========== matching固有のメソッド ==========
    
    def start_eye_calibration(self):
        """瞳孔径キャリブレーション開始"""
        if self.current_calib_eye_idx >= len(self.calibration_eyes):
            self.setup_experiment_blocks()
            return
        
        current_eye = self.calibration_eyes[self.current_calib_eye_idx]
        tk.messagebox.showinfo("Calibration", 
                              f"Next: Calibration for {current_eye} Eye.\nPlease cover the other eye.")
        
        self.offset_x.set(0)
        self.offset_y.set(0)
        self.pupil_diameter_val.set(4.0)
        
        self.setup_calibration_ui_matching(is_new_eye=True)
    
    def update_calibration_view(self, *args):
        """キャリブレーション画面の表示を更新する"""
        self.canvas1.delete("calib")
        self.canvas2.delete("calib")

        d_fg = self.distance1
        d_bg = self.distance2

        # 前景マーカーのサイズ計算 (正方形)
        fg_marker_size = stimuli_utils.get_size_for_visual_angle(d_fg, VISUAL_ANGLE_DEG)
        
        # 背景マーカーのサイズ計算 (横長)
        bg_marker_h = stimuli_utils.get_size_for_visual_angle(d_bg, VISUAL_ANGLE_DEG)
        bg_marker_w = stimuli_utils.get_size_for_visual_angle(d_bg, VISUAL_ANGLE_DEG * 2)

        # Window 1 (被験者側)のマーカー描画
        stimuli_utils.draw_image_corner_brackets(
            self.canvas1, bg_marker_w, bg_marker_h, 
            self.offset_x.get(), self.offset_y.get(), 
            color=WIN1_MARKER_COLOR, line_width=stimuli_utils.MARKER_LINE_WIDTH * 1.5
        )
        stimuli_utils.draw_image_corner_brackets(
            self.canvas1, bg_marker_h, bg_marker_h, 
            self.offset_x.get(), self.offset_y.get(), 
            color=WIN1_MARKER_COLOR, line_width=stimuli_utils.MARKER_LINE_WIDTH * 1.5
        )
        
        # Window 2 (実験者側) のマーカー描画
        stimuli_utils.draw_image_corner_brackets(
            self.canvas2, fg_marker_size, fg_marker_size, 
            0, 0, color=WIN2_MARKER_COLOR, flip_x=False, 
            line_width=stimuli_utils.MARKER_LINE_WIDTH
        )
        stimuli_utils.draw_center_cross(self.canvas2, color=WIN2_MARKER_COLOR)

    def adjust_offset(self, dx, dy):
        """矢印キーによるオフセット調整用関数"""
        self.offset_x.set(self.offset_x.get() + dx)
        self.offset_y.set(self.offset_y.get() + dy)
        self.update_calibration_view()
        return "break"

    def clear_key_bindings(self):
        """キーバインディングをクリア"""
        if not hasattr(self, 'key_bindings'):
            self.key_bindings = {}
        for key, binding_id in self.key_bindings.items():
            try:
                self.root.unbind(key, binding_id)
            except:
                pass
        self.key_bindings.clear()
    
    def setup_calibration_ui_matching(self, is_new_eye=False, is_new_block=False):
        """matching用キャリブレーション画面"""
        self.update_calibration_view()
        
        if hasattr(self, 'ctrl_frame') and self.ctrl_frame and self.ctrl_frame.winfo_exists():
            self.ctrl_frame.destroy()
        
        self.ctrl_frame = tk.Frame(self.root, bg='gray')
        self.ctrl_frame.place(relx=0.5, rely=0.8, anchor='center')
        
        self.clear_key_bindings()
        
        if is_new_eye:
            eye = self.calibration_eyes[self.current_calib_eye_idx]
            instruction_text = f"[{eye} Eye Calibration]\nUse the arrow keys to adjust the position of the red frame."
            button_text = "Calibration Done, Next"
            button_command = self._on_calibration_done
        else:
            instruction_text = "Use the arrow keys to adjust the position of the red frame."
            button_text = "Calibration Done, Start Block"
            button_command = self.start_experiment_block
        
        tk.Label(self.ctrl_frame, text=instruction_text, bg='gray', fg='white', 
                font=("Arial", 12)).pack(pady=10, padx=20)
        
        btn = tk.Button(self.ctrl_frame, text=button_text, command=button_command)
        btn.pack(pady=10)
        
        self.key_bindings['<Return>'] = self.root.bind('<Return>', 
                                                       lambda e: button_command())
        self.key_bindings['<Left>'] = self.root.bind('<Left>', 
                                                     lambda e: self.adjust_offset(-1, 0))
        self.key_bindings['<Right>'] = self.root.bind('<Right>', 
                                                      lambda e: self.adjust_offset(1, 0))
        self.key_bindings['<Up>'] = self.root.bind('<Up>', 
                                                   lambda e: self.adjust_offset(0, -1))
        self.key_bindings['<Down>'] = self.root.bind('<Down>', 
                                                     lambda e: self.adjust_offset(0, 1))
        self.root.focus_set()
    
    def _on_calibration_done(self):
        """瞳孔径キャリブレーション完了後の処理"""
        
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        
        # デフォーカスマッチング
        defocus_matching.setup_defocus_matching_ui(self)

    def setup_experiment_blocks(self):
        """実験ブロック構成を設定"""
        self.color_matrix = COLOR_MATRIX
        self.avg_color_match_results = []

        dom_eye = self.participant_dominance.get()
        if dom_eye not in self.calib_results:
            dom_eye = "Right"

        self.offset_x.set(self.calib_results[dom_eye]["offset_x"])
        self.offset_y.set(self.calib_results[dom_eye]["offset_y"])
        self.current_pd_mean = self.calib_results[dom_eye]["pd_mean"]

        self.save_preview_images()


        # キャリブレーション画面の更新（マーカー描画）
        self.canvas1.delete("calib")
        self.canvas2.delete("calib")

        fg_marker_size = stimuli_utils.get_size_for_visual_angle(self.distance1, VISUAL_ANGLE_DEG)
        bg_marker_h = stimuli_utils.get_size_for_visual_angle(self.distance2, VISUAL_ANGLE_DEG)
        bg_marker_w = stimuli_utils.get_size_for_visual_angle(self.distance2, VISUAL_ANGLE_DEG * 2)

        # Window 1
        stimuli_utils.draw_image_corner_brackets(
            self.canvas1, bg_marker_w, bg_marker_h, 
            self.offset_x.get(), self.offset_y.get(), 
            color=WIN1_MARKER_COLOR, line_width=stimuli_utils.MARKER_LINE_WIDTH * 1.5
        )
        stimuli_utils.draw_image_corner_brackets(
            self.canvas1, bg_marker_h, bg_marker_h, 
            self.offset_x.get(), self.offset_y.get(), 
            color=WIN1_MARKER_COLOR, line_width=stimuli_utils.MARKER_LINE_WIDTH * 1.5
        )

        # Window 2
        stimuli_utils.draw_image_corner_brackets(
            self.canvas2, fg_marker_size, fg_marker_size, 
            0, 0, color=WIN2_MARKER_COLOR, line_width=stimuli_utils.MARKER_LINE_WIDTH
        )
        stimuli_utils.draw_center_cross(self.canvas2, color=WIN2_MARKER_COLOR)

        conditions = ["Single plane", "Single plane + defocus simulation", "Dual plane", "Dual plane flat"]

        # ブロックごとに条件を作成してシャッフル
        bino_block = [{"condition": c, "ocularity": "binocular"} for c in conditions]
        mono_block = [{"condition": c, "ocularity": "monocular"} for c in conditions]

        random.shuffle(bino_block)
        random.shuffle(mono_block)

        # どちらのブロックを先にするかランダムに決定して結合
        if random.choice([True, False]):
            self.blocks = bino_block + mono_block
        else:
            self.blocks = mono_block + bino_block

        self.current_block_index = 0
        self.current_trial_in_experiment = 0
        self.start_block()
    
    def save_preview_images(self):
        """プレビュー画像を保存"""
        now = datetime.datetime.now()
        date_str = now.strftime("%Y%m%d_%H%M%S")
        p_id = self.participant_id.get()
        save_dir = os.path.join(FIGURE_DIR, f"{p_id}_{date_str}", "stimulis")
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        d_fg = self.distance1
        ppd_fg = stimuli_utils.get_size_for_visual_angle(d_fg, 1.0)
        
        width_deg = 7.9
        height_deg = 3.95
        width_fg = int(width_deg * ppd_fg)
        height_fg = int(height_deg * ppd_fg)
        
        ori = 0
        gabor_base = stimuli_utils.create_cosine_windowed_grating_base(width_fg, height_fg, ppd_fg, 
                                                                       self.spatial_freq, orientation=ori)
        noise_base = stimuli_utils.create_noise_base(width_fg, height_fg, ppd_fg, 
                                                     self.spatial_freq)
        
        L_fg = 15.0
        L_bg = 15.0
        C_bg = 0.0
        L_ref = 30.0
        
        for ref_c in [0.0, 0.2, 0.4]:
            lum_ref_fg = L_ref * (1.0 + ref_c * gabor_base)
            pix_ref_fg = np.interp(lum_ref_fg, self.fg_lums, self.fg_pixels).astype(np.uint8)
            img_ref = Image.fromarray(pix_ref_fg, mode='L')
            if getattr(self, 'color_matrix', None) is not None:
                img_ref = stimuli_utils.apply_color_matrix_preserve_luminance(img_ref, self.color_matrix)
            img_ref.save(os.path.join(save_dir, f"ref_gabor_contrast_{ref_c}.png"))
        
        c_test = 0.0
        lum_test_fg = L_fg * (1.0 + c_test * gabor_base)
        pix_test_fg = np.interp(lum_test_fg, self.fg_lums, self.fg_pixels).astype(np.uint8)
        img_test_fg = Image.fromarray(pix_test_fg, mode='L')
        if getattr(self, 'color_matrix', None) is not None:
            img_test_fg = stimuli_utils.apply_color_matrix_preserve_luminance(img_test_fg, self.color_matrix)
        img_test_fg.save(os.path.join(save_dir, "single_plane_foreground.png"))
        
        lum_noise = L_bg * (1.0 + C_bg * noise_base)
        # Background should be mapped using background calibration
        pix_noise = np.interp(lum_noise, self.bg_lums, self.bg_pixels).astype(np.uint8)
        img_noise = Image.fromarray(pix_noise, mode='L')
        img_noise.save(os.path.join(save_dir, "single_plane_background.png"))

        # Also save dual-plane specific foreground/background using respective calibrations
        # Foreground (uses foreground calibration)
        img_test_fg.save(os.path.join(save_dir, "dual_plane_foreground.png"))

        # Background for dual plane (use background calibration mapping)
        pix_noise_bg = np.interp(lum_noise, self.bg_lums, self.bg_pixels).astype(np.uint8)
        img_noise_bg = Image.fromarray(pix_noise_bg, mode='L')
        img_noise_bg.save(os.path.join(save_dir, "dual_plane_background.png"))
        
        lum_total = lum_noise + lum_test_fg
        pix_total = np.interp(lum_total, self.fg_lums, self.fg_pixels).astype(np.uint8)
        img_total = Image.fromarray(pix_total, mode='L')
        if getattr(self, 'color_matrix', None) is not None:
            img_total = stimuli_utils.apply_color_matrix_preserve_luminance(img_total, self.color_matrix)
        img_total.save(os.path.join(save_dir, "single_plane_combined.png"))
        
        D = abs(1/(self.distance1/100.0) - 1/(self.distance2/100.0))
        pd_mm = self.current_pd_mean if self.current_pd_mean > 0 else 4.0
        lum_noise_defocus = stimuli_utils.apply_torch_fft_blur_luminance(lum_noise, D, pd_mm, ppd_fg)
        pix_noise_defocus = np.interp(lum_noise_defocus, self.bg_lums, self.bg_pixels).astype(np.uint8)
        img_noise_defocus = Image.fromarray(pix_noise_defocus, mode='L')
        img_noise_defocus.save(os.path.join(save_dir, "single_plane_defocus_background.png"))

        # Dual-plane defocus background (mapped with background calibration)
        img_noise_defocus.save(os.path.join(save_dir, "dual_plane_defocus_background.png"))
        
        lum_total_defocus = lum_noise_defocus + lum_test_fg
        pix_total_defocus = np.interp(lum_total_defocus, self.fg_lums, self.fg_pixels).astype(np.uint8)
        img_total_defocus = Image.fromarray(pix_total_defocus, mode='L')
        if getattr(self, 'color_matrix', None) is not None:
            img_total_defocus = stimuli_utils.apply_color_matrix_preserve_luminance(img_total_defocus, self.color_matrix)
        img_total_defocus.save(os.path.join(save_dir, "single_plane_defocus_combined.png"))
        
        # --- Dual plane previews ---
        # Dual plane: expanded background noise with foreground overlaid (center crop)
        width_bg_expanded = int(width_fg * 2.5)
        noise_base_expanded = stimuli_utils.create_noise_base(width_bg_expanded, height_fg, ppd_fg, self.spatial_freq)
        lum_noise_expanded = L_bg * (1.0 + noise_base_expanded)

        # save an example background crop (center) for dual plane
        start_x = (width_bg_expanded - width_fg) // 2
        end_x = start_x + width_fg
        lum_noise_crop = lum_noise_expanded[:, start_x:end_x]
        pix_noise_crop = np.interp(lum_noise_crop, self.fg_lums, self.fg_pixels).astype(np.uint8)
        Image.fromarray(pix_noise_crop, mode='L').save(os.path.join(save_dir, "dual_plane_background.png"))

        # foreground (same size as foreground window)
        pix_test_fg = np.interp(lum_test_fg, self.fg_lums, self.fg_pixels).astype(np.uint8)
        Image.fromarray(pix_test_fg, mode='L').save(os.path.join(save_dir, "dual_plane_foreground.png"))

        # combined (overlay fg on center of expanded background -> crop to fg size)
        lum_combined_expanded = lum_noise_expanded.copy()
        # place foreground at center
        fg_x0 = (width_bg_expanded - width_fg) // 2
        lum_combined_expanded[:, fg_x0:fg_x0+width_fg] = lum_combined_expanded[:, fg_x0:fg_x0+width_fg] + lum_test_fg
        lum_combined_crop = lum_combined_expanded[:, start_x:end_x]
        pix_combined_crop = np.interp(lum_combined_crop, self.fg_lums, self.fg_pixels).astype(np.uint8)
        Image.fromarray(pix_combined_crop, mode='L').save(os.path.join(save_dir, "dual_plane_combined.png"))

        # --- Dual plane flat previews ---
        # Background is flat (no noise)
        lum_flat_bg = np.full((height_fg, width_fg), L_bg, dtype=np.float32)
        pix_flat_bg = np.interp(lum_flat_bg, self.fg_lums, self.fg_pixels).astype(np.uint8)
        Image.fromarray(pix_flat_bg, mode='L').save(os.path.join(save_dir, "dual_plane_flat_background.png"))

        # foreground (same as before)
        Image.fromarray(pix_test_fg, mode='L').save(os.path.join(save_dir, "dual_plane_flat_foreground.png"))

        # combined flat
        lum_flat_combined = lum_flat_bg + lum_test_fg
        pix_flat_combined = np.interp(lum_flat_combined, self.fg_lums, self.fg_pixels).astype(np.uint8)
        Image.fromarray(pix_flat_combined, mode='L').save(os.path.join(save_dir, "dual_plane_flat_combined.png"))
    
    def start_block(self):
        """ブロック開始"""
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        
        if self.current_block_index >= len(self.blocks):
            self.finish_experiment()
            return
        
        self.current_block_cond = self.blocks[self.current_block_index]
        self.current_trial_in_block = 0
        
        self.setup_block_confirmation_ui()
    
    def setup_block_confirmation_ui(self):
        """ブロック確認画面"""
        if hasattr(self, 'ctrl_frame') and self.ctrl_frame and self.ctrl_frame.winfo_exists():
            self.ctrl_frame.destroy()
        
        self.clear_key_bindings()
        
        self.ctrl_frame = tk.Frame(self.root, bg='gray')
        self.ctrl_frame.place(relx=0.5, rely=0.5, anchor='center')
        
        cond = self.current_block_cond["condition"]
        oc = self.current_block_cond["ocularity"]
        
        eye_inst = "Please use your DOMINANT eye and COVER the other eye." if oc == "monocular" \
                   else "Please use BOTH eyes."
        
        instruction_text = (f"[Block {self.current_block_index + 1}/{len(self.blocks)}]\n"
                          f"Condition: {cond}\nOcularity: {oc}\n\n"
                          f"{eye_inst}\n\nPress 'Enter' to start.")
        
        tk.Label(self.ctrl_frame, text=instruction_text, bg='gray', fg='white', 
                font=("Arial", 16)).pack(pady=20, padx=40)
        
        btn = tk.Button(self.ctrl_frame, text="OK", command=self._start_block_calibration, 
                       font=("Arial", 14))
        btn.pack(pady=10)
        btn.focus_set()
        
        self.key_bindings['<Return>'] = self.root.bind('<Return>', 
                                                       lambda e: self._start_block_calibration())
    
    def _start_block_calibration(self):
        """ブロック開始時のキャリブレーション"""
        self.clear_key_bindings()
        
        if hasattr(self, 'ctrl_frame') and self.ctrl_frame and self.ctrl_frame.winfo_exists():
            self.ctrl_frame.destroy()
        
        self.setup_calibration_ui_matching(is_new_block=True)
    
    def run_trial(self):
        """試行実行"""
        if self.current_trial_in_block >= len(self.trial_list):
            self.current_block_index += 1
            if self.current_block_index >= len(self.blocks):
                self._save_results_and_finish()
            else:
                self.root.after(500, self.start_block)
            return
        
        trial = self.trial_list[self.current_trial_in_block]
        ref_c = trial["ref_contrast"]
        cond = self.current_block_cond["condition"]
        
        self.init_contrast = random.uniform(0.0, 1.0)
        self.init_slider_val = math.log10(self.init_contrast * 99.0 + 1.0) / 2.0
        
        self.canvas1.configure(bg='black')
        self.canvas2.configure(bg='black')
        
        d_fg = self.distance1
        d_bg = self.distance2
        ppd_fg = stimuli_utils.get_size_for_visual_angle(d_fg, 1.0)
        ppd_bg = stimuli_utils.get_size_for_visual_angle(d_bg, 1.0)
        
        width_deg = 7.9
        height_deg = 3.95
        width_fg = int(width_deg * ppd_fg)
        height_fg = int(height_deg * ppd_fg)
        width_bg = int(width_deg * ppd_bg)
        height_bg = int(height_deg * ppd_bg)
        
        ori = trial["orientation"]
        
        self.gabor_base = stimuli_utils.create_cosine_windowed_grating_base(width_fg, height_fg, ppd_fg, 
                                                                            self.spatial_freq, orientation=ori)
        
        L_bg = self.L_bg
        
        if cond == "Dual plane flat":
            # コントラスト0の場合、重いFFTノイズ生成処理をスキップ
            width_bg_expanded = int(width_bg * 2.5)
            self.noise_base = None
            lum_noise_temp = np.full((height_bg, width_bg_expanded), L_bg, dtype=np.float32)
        else:
            if cond == "Dual plane":
                width_bg_expanded = int(width_bg * 2.5)
                self.noise_base = stimuli_utils.create_noise_base(width_bg_expanded, height_bg, ppd_bg, self.spatial_freq)
            else:
                self.noise_base = stimuli_utils.create_noise_base(width_fg, height_fg, ppd_fg, self.spatial_freq)
            
            lum_noise_temp = L_bg * (1.0 + self.noise_base)
        
        cond = self.current_block_cond["condition"]
        if cond == "Single plane + defocus simulation":
            D = abs(1/(self.distance1/100.0) - 1/(self.distance2/100.0))
            self.cached_lum_noise = stimuli_utils.apply_torch_fft_blur_luminance(
                lum_noise_temp, D, self.current_pd_mean, ppd_fg
            )
        else:
            self.cached_lum_noise = lum_noise_temp
        
        self.setup_contrast_matching_ui()
    
    def setup_contrast_matching_ui(self):
        """コントラストマッチングUI"""
        if hasattr(self, 'ctrl_frame') and self.ctrl_frame and self.ctrl_frame.winfo_exists():
            self.ctrl_frame.destroy()
        
        self.clear_key_bindings()
        
        self.ctrl_frame = tk.Frame(self.root, bg='gray')
        self.ctrl_frame.place(relx=0.5, rely=0.8, anchor='center')
        
        self.slider_val = tk.DoubleVar(value=self.init_slider_val)
        slider = tk.Scale(self.ctrl_frame, from_=1.0, to=0.0, resolution=0.001, 
                         orient=tk.HORIZONTAL, length=400, variable=self.slider_val, 
                         showvalue=0, command=lambda *args: self.update_stimuli())
        slider.pack(pady=10)
        
        btn = tk.Button(self.ctrl_frame, text="Next Trial", command=self.save_and_next)
        btn.pack(pady=10)
        btn.focus_set()
        
        instruction_text = "Adjust the slider to match the contrast.\nPress 'Down' arrow to confirm."
        tk.Label(self.ctrl_frame, text=instruction_text, bg='gray', fg='white', 
                font=("Arial", 12)).pack(pady=10, padx=20)
        
        self.key_bindings['<Down>'] = self.root.bind('<Down>', lambda e: self.save_and_next())
        self.key_bindings['<Left>'] = self.root.bind('<Left>', lambda e: self._handle_contrast_key_press(e))
        self.key_bindings['<Right>'] = self.root.bind('<Right>', lambda e: self._handle_contrast_key_press(e))
        self.root.focus_set()
        
        self.update_stimuli()
    
    def _handle_contrast_key_press(self, event):
        """コントラスト調整用キー処理"""
        step = 0.005
        current_val = self.slider_val.get()
        
        if event.keysym == 'Left':
            new_val = max(0.0, current_val - step)
            self.slider_val.set(new_val)
        elif event.keysym == 'Right':
            new_val = min(1.0, current_val + step)
            self.slider_val.set(new_val)
        
        self.update_stimuli()
        return "break"
    
    def update_stimuli(self):
        """刺激表示を更新"""
        v = self.slider_val.get() if hasattr(self, 'slider_val') else self.init_slider_val
        c_test = (10**(2.0 * v) - 1.0) / 99.0
        
        trial = self.trial_list[self.current_trial_in_block]
        cond = self.current_block_cond["condition"]
        ref_c = trial["ref_contrast"]
        
        d_fg = self.distance1
        d_bg = self.distance2
        ppd_fg = stimuli_utils.get_size_for_visual_angle(d_fg, 1.0)
        ppd_bg = stimuli_utils.get_size_for_visual_angle(d_bg, 1.0)
        
        gap_y_deg = 2.0
        gap_y_fg = int(gap_y_deg * ppd_fg)
        gap_y_bg = int(gap_y_deg * ppd_bg)
        
        self.canvas1.delete("stim")
        self.canvas2.delete("stim")
        self.canvas1.delete("calib")
        self.canvas2.delete("calib")
        
        cx1, cy1 = self.width//2 + self.offset_x.get(), self.height//2 + self.offset_y.get()
        cx2, cy2 = self.canvas2.winfo_width()//2, self.canvas2.winfo_height()//2
        
        L_fg = 15.0
        L_bg = 15.0
        L_ref = 30.0
        
        # Generate PhotoImage objects for reference/test/background using helper
        photos = stimuli_utils.generate_matching_photos(
            self.gabor_base, self.cached_lum_noise,
            self.fg_lums, self.fg_pixels, self.bg_lums, self.bg_pixels,
            L_fg=L_fg, L_bg=L_bg, L_ref=L_ref, c_test=c_test, ref_c=ref_c, cond=cond,
            color_matrix=getattr(self, 'color_matrix', None)
        )

        self.photo_ref_fg = photos.get('photo_ref_fg')
        if cond in ["Dual plane", "Dual plane flat"]:
            self.photo_test_fg = photos.get('photo_test_fg')
            self.photo_noise_bg = photos.get('photo_noise_bg')

            self.canvas2.create_image(cx2, cy2 - gap_y_fg, image=self.photo_ref_fg, anchor='center', tags="stim")
            self.canvas1.create_image(cx1, cy1 + gap_y_bg, image=self.photo_noise_bg, anchor='center', tags="stim")
            self.canvas2.create_image(cx2, cy2 + gap_y_fg, image=self.photo_test_fg, anchor='center', tags="stim")
        else:
            self.photo_test = photos.get('photo_test')
            self.canvas2.create_image(cx2, cy2 - gap_y_fg, image=self.photo_ref_fg, anchor='center', tags="stim")
            self.canvas2.create_image(cx2, cy2 + gap_y_fg, image=self.photo_test, anchor='center', tags="stim")
    
    def save_and_next(self, event=None):
        """結果保存して次へ"""
        if hasattr(self, 'ctrl_frame') and self.ctrl_frame and self.ctrl_frame.winfo_exists():
            self.ctrl_frame.destroy()
        
        self.clear_key_bindings()
        
        v = self.slider_val.get() if hasattr(self, 'slider_val') else self.init_slider_val
        c_test = (10**(2.0 * v) - 1.0) / 99.0
        trial = self.trial_list[self.current_trial_in_block]
        
        right_res = self.calib_results.get("Right", {"offset_x": 0, "offset_y": 0, "pd_mean": 0})
        left_res = self.calib_results.get("Left", {"offset_x": 0, "offset_y": 0, "pd_mean": 0})
        
        self.add_trial_result({
            "ID": self.participant_id.get(),
            "Age": self.participant_age.get(),
            "Gender": self.participant_gender.get(),
            "IPD(mm)": self.participant_ipd.get(),
            "Dominance": self.participant_dominance.get(),
            "Block_ID": self.current_block_index + 1,
            "Condition": self.current_block_cond["condition"],
            "Ocularity": self.current_block_cond["ocularity"],
            "Trial_ID": self.current_trial_in_experiment + 1,
            "Orientation": trial["orientation"],
            "Ref_Contrast": trial["ref_contrast"],
            "Matched_Contrast": round(c_test, 4),
            "L_fg": self.L_fg,
            "L_bg": self.L_bg,
            "L_ref": self.L_ref,
            "Config_JSON": json.dumps(self.config if getattr(self, 'config', None) is not None else {}),
            "PD_Right": right_res["pd_mean"],
            "OffsetX_Right": right_res["offset_x"],
            "OffsetY_Right": right_res["offset_y"],
            "PD_Left": left_res["pd_mean"],
            "OffsetX_Left": left_res["offset_x"],
            "OffsetY_Left": left_res["offset_y"]
        })
        
        self.current_trial_in_block += 1
        self.current_trial_in_experiment += 1
        
        if self.current_trial_in_block >= len(self.trial_list):
            self.current_block_index += 1
            if self.current_block_index >= len(self.blocks):
                self._save_results_and_finish()
            else:
                self.root.after(500, self.start_block)
        else:
            self.root.after(500, self.run_trial)
    
    def start_experiment_block(self):
        """ブロック実験開始"""
        self.clear_key_bindings()
        
        if hasattr(self, 'ctrl_frame') and self.ctrl_frame and self.ctrl_frame.winfo_exists():
            self.ctrl_frame.destroy()
        
        # 試行リストを生成
        ref_contrasts = [0.0, 0.2, 0.4]
        orientations = [0]
        
        self.trial_list = []
        for ref_c in ref_contrasts:
            for ori in orientations:
                for _ in range(NUM_REPETITIONS):
                    self.trial_list.append({
                        "ref_contrast": ref_c,
                        "orientation": ori
                    })
        
        random.shuffle(self.trial_list)
        self.current_trial_in_block = 0
        
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        
        self.run_trial()
    
    def _save_results_and_finish(self):
        """結果を保存して終了"""
        if not os.path.exists(RESULT_DIR):
            os.makedirs(RESULT_DIR)
        
        p_id = self.participant_id.get()
        now = datetime.datetime.now()
        date_str = now.strftime("%Y%m%d_%H%M%S")
        save_folder = os.path.join(RESULT_DIR, f"{p_id}_{date_str}")
        
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        filename = os.path.join(save_folder, 
                               f"result_{p_id}_{date_str}.csv")
        
        if self.results:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                writer.writeheader()
                writer.writerows(self.results)
            # Save the config used for this experiment run into the results folder
            try:
                cfg_to_save = getattr(self, 'config', None) or {}
                with open(os.path.join(save_folder, 'used_experiment_config.json'), 'w', encoding='utf-8') as cf:
                    json.dump(cfg_to_save, cf, indent=2)
            except Exception:
                pass
        
        tk.messagebox.showinfo("Completed", 
                              f"Experiment finished.\nData saved to: {filename}")
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ExperimentApp(root)
    root.mainloop()
