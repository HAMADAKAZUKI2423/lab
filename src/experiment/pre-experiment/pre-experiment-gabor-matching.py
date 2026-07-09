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


# ==========================================
# 定数設定エリア (実験条件やデザインはここを変更)
# ==========================================
VISUAL_ANGLE_DEG = 7.9
NUM_REPETITIONS = 5
SPATIAL_FREQ = 4  # Spatial frequency in cpd
PUPIL_DIAMETER_MM = 4.0
# Dual plane / Dual plane flat の背景(window1)の横幅拡張に使う倍率。
#   - binocular: 長い側 + 短い側 = 1.3 + 0.7 = 2.0倍（優位眼依存の左右非対称）。
#   - monocular: 1.3 × 2 = 2.6倍（左右対称・中央）。
ASYM_WIDTH_FACTOR_LARGE = 1.3
ASYM_WIDTH_FACTOR_SMALL = 0.7

# Window2(canvas2)に表示する前景(ref/test)の左右対称拡張倍率。
# 中心から片側 = 元画像幅の1.3倍 → 左右合計2.6倍。
WIN2_HALF_WIDTH_FACTOR = 1.3
WIN2_TOTAL_WIDTH_FACTOR = WIN2_HALF_WIDTH_FACTOR * 2  # = 2.6



script_dir = os.path.dirname(os.path.abspath(__file__))
lab_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
DISPLAY_DIR = os.path.join(lab_root, "results", "tables", "DisplayBrightness")

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

def load_matrix_csv(name):
    try:
        return np.loadtxt(os.path.join(DISPLAY_DIR, f"{name}.csv"), delimiter=",")
    except IOError:
        print(f"WARN: {name}.csv not found. Using fallback.")
        return None

def load_eotf(path):
    # channel,v,yn -> {ch: (v_array, yn_array)}
    import csv
    d = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ch = r["channel"].upper()
                d.setdefault(ch, ([], []))
                d[ch][0].append(float(r["v"])); d[ch][1].append(float(r["yn"]))
        return {ch: (np.array(v), np.array(y)) for ch,(v,y) in d.items()}
    except IOError:
        return None

def load_ext_lum_lut(path):
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    return arr[:,0], arr[:,1:4]   # Y_grid(N,), px_grid(N,3)


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
        self.distance2 = 150  # Background distance (cm)
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
        
        # 前景補正は背景校正(bg)から得た画素値に C を掛けて行うため、fg校正は使用しない。
        # self.fg_lums, self.fg_pixels = stimuli_utils.load_calibration_data(fg_calib_dir)
        self.color_matrix = load_matrix_csv("C")
        if self.color_matrix is None:
            print("WARN: C.csv not found. Using hardcoded fallback matrix.")
            self.color_matrix = np.array([
                [ 0.385676, -0.029594,  0.007298],
                [ 0.002786,  0.485416, -0.011852],
                [ 0.005025,  0.003184,  0.601995],
            ], dtype=np.float64)

        self.eotf_bg = load_eotf(os.path.join(DISPLAY_DIR, "eotf_bg.csv"))
        self.eotf_fg = load_eotf(os.path.join(DISPLAY_DIR, "eotf_fg.csv"))
        try:
            self.ext_lum_Y, self.ext_lum_px = load_ext_lum_lut(os.path.join(DISPLAY_DIR, "ext_lum_lut.csv"))
        except (IOError, ValueError):
            print("WARN: ext_lum_lut.csv not found or invalid. Single plane will not work correctly.")
            self.ext_lum_Y, self.ext_lum_px = None, None

        self.bg_lums, self.bg_pixels = stimuli_utils.load_calibration_data(bg_calib_dir)
        
        if self.bg_lums is None:
            print("Warning: Calibration data not found. Linear mapping will be used.")
            self.bg_lums = np.array([0.0, 100.0])
            self.bg_pixels = np.array([0, 255])

        self.fg_lums, self.fg_pixels = self.bg_lums, self.bg_pixels

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
        # 実表示(generate_matching_photos)と同様、refは前景C経路の再現上限クリップを避けるため
        # Single planeと同じ拡張輝度LUTで変換する（LUT未整備時のみ従来のC経路にフォールバック）。
        for ref_c in [0.1, 0.2]:
            lum_ref_fg = L_ref * (1.0 + ref_c * gabor_base)
            if self.ext_lum_Y is not None and self.ext_lum_px is not None:
                img_ref = stimuli_utils.lum_to_pil_singleplane(lum_ref_fg, self.ext_lum_Y, self.ext_lum_px)
            else:
                img_ref = stimuli_utils.lum_to_pil_window2(lum_ref_fg, self.bg_lums, self.bg_pixels, self.color_matrix)
            img_ref.save(os.path.join(save_dir, f"ref_gabor_contrast_{ref_c}.png"))
            
        # Single plane stimulus
        c_test = 0.0
        lum_test_fg = L_fg * (1.0 + c_test * gabor_base)
        img_test_fg = stimuli_utils.lum_to_pil_window2(lum_test_fg, self.bg_lums, self.bg_pixels, getattr(self, 'color_matrix_xyz', None))
        img_test_fg.save(os.path.join(save_dir, "single_plane_foreground.png"))

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
        # 元画像（1枚・約7.9deg）の前景幅。背景の非対称化はこの基準幅では使わないが、
        # 「2.6倍前」の基準として保持する。
        width_fg_base = int(width_deg * ppd_fg)
        height_fg = int(height_deg * ppd_fg)
        width_bg = int(width_deg * ppd_bg)   # 背景(window1)の元幅。非対称化の基準（変更しない）
        height_bg = int(height_deg * ppd_bg)
        # Window2(ref/test)はデフォルトで左右対称2.6倍幅にする（毎回paddingしない）
        width_fg = int(width_fg_base * WIN2_TOTAL_WIDTH_FACTOR)
        
        ori = trial["orientation"]
        
        self.gabor_base = stimuli_utils.create_cosine_windowed_grating_base(width_fg, height_fg, ppd_fg, 
                                                                            self.spatial_freq, orientation=ori)
        
        L_bg = self.L_bg
        
        # === 背景(window1)の横幅設定 ===
        # Dual plane / Dual plane flat では、monocular・binocular の両方で背景の総横幅を
        # 「合計2.0倍（= 1.3 + 0.7）」に統一する。
        #   - binocular: 優位眼依存の左右非対称(1.3 / 0.7)オフセットを与える（従来どおり）。
        #   - monocular: 横幅は binocular と同じだが、左右対称（中央配置・オフセット0）。
        ocularity = self.current_block_cond["ocularity"]
        dom_eye = self.participant_dominance.get()
        is_dual = cond in ["Dual plane", "Dual plane flat"]
        
        if is_dual:
            if ocularity == "binocular":
                # 背景は合計2.0倍（1.3 + 0.7）・優位眼依存の非対称（従来どおり）
                width_bg_expanded = int(width_bg * (ASYM_WIDTH_FACTOR_LARGE + ASYM_WIDTH_FACTOR_SMALL))  # = 2.0
                if dom_eye == "Right":
                    bg_left_factor = ASYM_WIDTH_FACTOR_SMALL   # 0.7
                    bg_right_factor = ASYM_WIDTH_FACTOR_LARGE  # 1.3
                else:  # "Left"
                    bg_left_factor = ASYM_WIDTH_FACTOR_LARGE   # 1.3
                    bg_right_factor = ASYM_WIDTH_FACTOR_SMALL  # 0.7
                # 提示中央(cx1)に対する背景の描画オフセット（右が大きいほど +x 側へ）
                self.bg_center_offset_x = int(width_bg * (bg_right_factor - bg_left_factor) / 2.0)
            else:
                # monocular：左右とも1.3倍＝合計2.6倍・左右対称（中央配置）
                width_bg_expanded = int(width_bg * (ASYM_WIDTH_FACTOR_LARGE * 2))  # = 2.6
                self.bg_center_offset_x = 0
        else:
            # Single plane 系：window1 背景は単独表示しない（等倍・中央）
            width_bg_expanded = width_bg
            self.bg_center_offset_x = 0
        # ============================================================================
        
        if cond == "Dual plane flat":
            # コントラスト0の場合、重いFFTノイズ生成処理をスキップ
            lum_noise_temp = np.full((height_bg, width_bg_expanded), L_bg, dtype=np.float32)
        else:
            if cond == "Dual plane":
                self.noise_base = stimuli_utils.create_noise_base(width_bg_expanded, height_bg, ppd_bg, self.spatial_freq)
            else:
                self.noise_base = stimuli_utils.create_noise_base(width_fg, height_fg, ppd_fg, self.spatial_freq)
            
            lum_noise_temp = L_bg * (1.0 + self.noise_base)
        
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
            L_fg=L_fg, L_bg=L_bg, L_ref=L_ref, c_test=c_test, ref_c=ref_c, 
            cond=cond, color_matrix=self.color_matrix,
            eotf_bg=self.eotf_bg, eotf_fg=self.eotf_fg,
            Y_grid=self.ext_lum_Y, px_grid=self.ext_lum_px
        )

        self.photo_ref_fg = photos.get('photo_ref_fg')
        if cond in ["Dual plane", "Dual plane flat"]:
            self.photo_test_fg = photos.get('photo_test_fg')
            self.photo_noise_bg = photos.get('photo_noise_bg')

            # ref/test は width_fg(=2.6倍) で生成済みのため、左右対称・中央配置でよい。
            # 旧実装では binocular 時に ref のみ非対称paddingしていたが、本変更で廃止する。
            # 背景(window1)の「合計2倍・非対称」は bg_center_offset_x として引き続き維持する。
            ref_offset_x = 0
            bg_offset_x = getattr(self, 'bg_center_offset_x', 0)
            self.canvas2.create_image(cx2 + ref_offset_x, cy2 - gap_y_fg, image=self.photo_ref_fg, anchor='center', tags="stim")
            self.canvas1.create_image(cx1 + bg_offset_x, cy1 + gap_y_bg, image=self.photo_noise_bg, anchor='center', tags="stim")
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
        ref_contrasts = [0.1, 0.2]
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
