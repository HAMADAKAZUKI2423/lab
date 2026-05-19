"""
画像実験スクリプト（削減版）

自然画像の視認性評価実験
基盤クラス（ExperimentBaseUI, ExperimentTrialLoop）から共通処理を継承
image固有のロジックのみを実装
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

from experiment_base_ui import ExperimentBaseUI
from experiment_trial_loop import ExperimentTrialLoop
import stimuli_utils
import defocus_matching


# ==========================================
# 定数設定エリア
# ==========================================
VISUAL_ANGLE_DEG = 7.9
NUM_TRIALS_BEFORE_BREAK = 100
NUM_REPETITIONS = 2
PUPIL_DIAMETER_MM = 4.0

# **CRITICAL**: image実験用タイミング設定（matching/gaborと異なる）
# Phase 1=500ms (show image on Win2 only)
# ISI=1000ms (black screen both)
# Phase 2=500ms (show both)
TIME_PHASE_1 = 500
TIME_ISI = 1000
TIME_PHASE_2 = 500

script_dir = os.path.dirname(os.path.abspath(__file__))
lab_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

BASE_IMG_DIR_1 = os.path.join(lab_root, "data", "processed", "images", "pre-experiment-image", "bg_noise")
BASE_IMG_DIR_2 = os.path.join(lab_root, "data", "processed", "images", "pre-experiment-image", "fg_image")
RESULT_DIR = os.path.join(lab_root, "results", "tables", "pre-experiment-image")
FIGURE_DIR = os.path.join(lab_root, "results", "figures", "pre-experiment-image")
PARTICIPANT_DATA_DIR = os.path.join(lab_root, "data", "processed", "tables", "pre-experiment-image")

for dir_path in [RESULT_DIR, FIGURE_DIR, PARTICIPANT_DATA_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

BG_COLOR = 'black'
WIN1_MARKER_COLOR = 'red'
WIN2_MARKER_COLOR = 'white'


class ExperimentApp(ExperimentBaseUI, ExperimentTrialLoop):
    """
    画像実験
    
    基盤クラスから継承：
    - ExperimentBaseUI: UI処理
    - ExperimentTrialLoop: 試行ループ
    
    image固有の処理を実装
    """
    
    def __init__(self, root):
        ExperimentBaseUI.__init__(self, root, PARTICIPANT_DATA_DIR)
        ExperimentTrialLoop.__init__(self)
        
        self.root = root
        self.root.title("Image Experiment - Controller (Window 2)")
        self.root.configure(bg=BG_COLOR)
        
        # image固有の変数
        self.distance1 = 50
        self.distance2 = 70
        self.defocus_val = tk.DoubleVar(value=0.0)
        
        self.blocks = []
        self.current_block_index = 0
        self.current_trial_in_block = 0
        self.current_block_cond = None
        
        self.pupil_diameter_val = tk.DoubleVar(value=4.0)
        self.evaluation_val = tk.IntVar(value=3)
        
        self.calibration_eyes = ["Right", "Left"]
        self.current_calib_eye_idx = 0
        self.calib_results = {}
        
        self.current_pd_mean = 0.0
        self.current_pd_std = 0.0
        
        self.participant_dominance = tk.StringVar(value="Right")
        
        # キャリブレーションデータ
        fg_calib_dir = os.path.join(lab_root, "results", "tables", "DisplayBrightness", "fg_calibration_log")
        bg_calib_dir = os.path.join(lab_root, "results", "tables", "DisplayBrightness", "bg_calibration_log")
        
        self.fg_lums, self.fg_pixels = stimuli_utils.load_calibration_data(fg_calib_dir)
        self.bg_lums, self.bg_pixels = stimuli_utils.load_calibration_data(bg_calib_dir)
        
        if self.fg_lums is None or self.bg_lums is None:
            print("Warning: Calibration data not found. Linear mapping will be used.")
            self.fg_lums = np.array([0.0, 100.0])
            self.fg_pixels = np.array([0, 255])
            self.bg_lums = np.array([0.0, 100.0])
            self.bg_pixels = np.array([0, 255])
        
        # ウィンドウのセットアップ
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        
        self.root.state('zoomed')
        
        self.win1 = tk.Toplevel(self.root)
        self.win1.title("Image Experiment - Display (Window 1)")
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
        
        self.result_dir = RESULT_DIR
        self.setup_trial_phases(TIME_PHASE_1, TIME_ISI, TIME_PHASE_2)
        
        self.setup_participant_info_ui()
        
        # 画像リスト
        self.image_list = []
        self.load_images()
    
    def load_images(self):
        """画像をロード"""
        image_dirs = [BASE_IMG_DIR_1, BASE_IMG_DIR_2]
        for img_dir in image_dirs:
            if os.path.exists(img_dir):
                self.image_list += glob.glob(os.path.join(img_dir, "*.png"))
                self.image_list += glob.glob(os.path.join(img_dir, "*.jpg"))
        
        if not self.image_list:
            print("Warning: No images found in specified directories")
    
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
    
    def on_participant_confirmed(self):
        """参加者確認後"""
        self.win1.update_idletasks()
        self.width = self.win1.winfo_width()
        self.height = self.win1.winfo_height()
        self.setup_experiment_ui()
    
    def setup_experiment_ui(self):
        """実験設定UI"""
        frame = tk.Frame(self.root, bg='gray', padx=20, pady=20)
        frame.place(relx=0.5, rely=0.5, anchor='center')
        
        tk.Label(frame, text="Experiment Setup", font=("Arial", 16)).grid(row=0, column=0, columnspan=2, pady=10)
        
        tk.Label(frame, text="Foreground Distance (cm):").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        dist1_entry = tk.Entry(frame)
        dist1_entry.insert(0, str(self.distance1))
        dist1_entry.grid(row=1, column=1, padx=5, pady=5)
        
        tk.Label(frame, text="Background Distance (cm):").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        dist2_entry = tk.Entry(frame)
        dist2_entry.insert(0, str(self.distance2))
        dist2_entry.grid(row=2, column=1, padx=5, pady=5)
        
        btn = tk.Button(frame, text="Setup Complete", command=lambda: self._on_setup_complete(frame, dist1_entry, dist2_entry))
        btn.grid(row=3, column=0, columnspan=2, pady=20)
    
    def _on_setup_complete(self, frame, dist1_entry, dist2_entry):
        """実験設定完了"""
        try:
            self.distance1 = int(dist1_entry.get())
            self.distance2 = int(dist2_entry.get())
        except (ValueError, tk.TclError):
            messagebox.showwarning("Input Error", "Please enter valid numbers.")
            return
        
        frame.destroy()
        self.setup_experiment_blocks()
    
    def setup_experiment_blocks(self):
        """実験ブロック構成"""
        blur_conditions = ["No Blur", "Blur"]
        
        self.blocks = []
        for blur in blur_conditions:
            self.blocks.append({"blur_condition": blur})
        
        random.shuffle(self.blocks)
        
        self.current_block_index = 0
        self.current_trial_in_experiment = 0
        self.start_block()
    
    def start_block(self):
        """ブロック開始"""
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        
        if self.current_block_index >= len(self.blocks):
            self._save_results_and_finish()
            return
        
        self.current_block_cond = self.blocks[self.current_block_index]
        self.current_trial_in_block = 0
        self.setup_block_confirmation_ui()
    
    def setup_block_confirmation_ui(self):
        """ブロック確認UI"""
        if hasattr(self, 'ctrl_frame') and self.ctrl_frame and self.ctrl_frame.winfo_exists():
            self.ctrl_frame.destroy()
        
        self.clear_key_bindings()
        
        self.ctrl_frame = tk.Frame(self.root, bg='gray')
        self.ctrl_frame.place(relx=0.5, rely=0.5, anchor='center')
        
        blur_cond = self.current_block_cond["blur_condition"]
        instruction_text = f"Next section is {blur_cond}, OK?\n\nPress 'Enter' to continue."
        
        tk.Label(self.ctrl_frame, text=instruction_text, bg='gray', fg='white', 
                font=("Arial", 16)).pack(pady=20, padx=40)
        
        btn = tk.Button(self.ctrl_frame, text="OK", command=self._start_calibration_from_confirmation, 
                       font=("Arial", 14))
        btn.pack(pady=10)
        btn.focus_set()
        
        self.key_bindings['<Return>'] = self.root.bind('<Return>', 
                                                       lambda e: self._start_calibration_from_confirmation())
    
    def _start_calibration_from_confirmation(self):
        """キャリブレーション開始"""
        self.clear_key_bindings()
        
        if hasattr(self, 'ctrl_frame') and self.ctrl_frame and self.ctrl_frame.winfo_exists():
            self.ctrl_frame.destroy()
        
        self.offset_x.set(0)
        self.offset_y.set(0)
        self.setup_calibration_ui(is_new_block=True)
    
    def update_calibration_view(self):
        """キャリブレーション画面更新"""
        self.canvas1.delete("calib")
        self.canvas2.delete("calib")
        
        d_fg = self.distance1.get()
        d_bg = self.distance2.get()
        
        fg_size = stimuli_utils.get_size_for_visual_angle(d_fg, VISUAL_ANGLE_DEG)
        bg_size = stimuli_utils.get_size_for_visual_angle(d_bg, VISUAL_ANGLE_DEG)
        
        stimuli_utils.draw_image_corner_brackets(self.canvas1, bg_size, bg_size, 
                                                self.offset_x.get(), self.offset_y.get(), 
                                                color=WIN1_MARKER_COLOR, line_width=stimuli_utils.MARKER_LINE_WIDTH * 1.5)
        
        stimuli_utils.draw_image_corner_brackets(self.canvas2, fg_size, fg_size, 
                                                0, 0, color=WIN2_MARKER_COLOR, 
                                                line_width=stimuli_utils.MARKER_LINE_WIDTH)
        stimuli_utils.draw_center_cross(self.canvas2, color=WIN2_MARKER_COLOR)
    
    def on_calibration_complete(self):
        """キャリブレーション完了"""
        defocus_matching.setup_defocus_matching_ui_image(self)
    
    def start_experiment_block(self):
        """ブロック実験開始"""
        self.clear_key_bindings()
        
        if hasattr(self, 'ctrl_frame') and self.ctrl_frame and self.ctrl_frame.winfo_exists():
            self.ctrl_frame.destroy()
        
        self.trial_list = []
        for img_path in self.image_list[:NUM_REPETITIONS]:  # 最初の数枚を使用
            self.trial_list.append({"image_path": img_path})
        
        random.shuffle(self.trial_list)
        self.current_trial_in_block = 0
        
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        
        self.run_trial()
    
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
        
        d_fg = self.distance1.get()
        ppd_fg = stimuli_utils.get_size_for_visual_angle(d_fg, 1.0)
        
        width_deg = 7.9
        height_deg = 3.95
        width_fg = int(width_deg * ppd_fg)
        height_fg = int(height_deg * ppd_fg)
        
        self.canvas1.configure(bg='black')
        self.canvas2.configure(bg='black')
        
        self.current_image_path = trial["image_path"]
        
        self.setup_stimulus_display_ui()
    
    def setup_stimulus_display_ui(self):
        """刺激表示UI"""
        if hasattr(self, 'ctrl_frame') and self.ctrl_frame and self.ctrl_frame.winfo_exists():
            self.ctrl_frame.destroy()
        
        self.clear_key_bindings()
        
        self.ctrl_frame = tk.Frame(self.root, bg='gray')
        self.ctrl_frame.place(relx=0.5, rely=0.8, anchor='center')
        
        self.evaluation_val.set(3)
        
        instruction_text = "Rate the visibility of the image.\nUse arrow keys ← → to select, Press 'Enter' to confirm."
        tk.Label(self.ctrl_frame, text=instruction_text, bg='gray', fg='white', 
                font=("Arial", 12)).pack(pady=10, padx=20)
        
        choice_frame = tk.Frame(self.ctrl_frame, bg='gray')
        choice_frame.pack(pady=10)
        
        self.eval_buttons = []
        for i in range(1, 6):
            btn = tk.Button(choice_frame, text=str(i), width=3, 
                           command=lambda val=i: self._on_evaluation_selected(val))
            btn.pack(side=tk.LEFT, padx=5)
            self.eval_buttons.append(btn)
        
        self._update_eval_highlight()
        
        self.key_bindings['<Left>'] = self.root.bind('<Left>', lambda e: self._move_selection(-1))
        self.key_bindings['<Right>'] = self.root.bind('<Right>', lambda e: self._move_selection(1))
        self.key_bindings['<Return>'] = self.root.bind('<Return>', 
                                                       lambda e: self._on_evaluation_selected(self.evaluation_val.get()))
        
        self.display_stimulus()
    
    def _update_eval_highlight(self):
        """評価ハイライト更新"""
        for i, btn in enumerate(self.eval_buttons):
            if i + 1 == self.evaluation_val.get():
                btn.config(bg='yellow', fg='black')
            else:
                btn.config(bg='lightgray', fg='black')
    
    def _move_selection(self, direction):
        """選択移動"""
        current = self.evaluation_val.get()
        new_val = current + direction
        new_val = max(1, min(5, new_val))
        self.evaluation_val.set(new_val)
        self._update_eval_highlight()
    
    def _on_evaluation_selected(self, value):
        """評価確定"""
        self.evaluation_val.set(value)
        self.save_and_next()
    
    def display_stimulus(self):
        """刺激表示"""
        try:
            img = Image.open(self.current_image_path)
            
            d_fg = self.distance1.get()
            ppd_fg = stimuli_utils.get_size_for_visual_angle(d_fg, 1.0)
            
            width_deg = 7.9
            height_deg = 3.95
            width_fg = int(width_deg * ppd_fg)
            height_fg = int(height_deg * ppd_fg)
            
            img = img.resize((width_fg, height_fg), Image.LANCZOS)
            
            self.photo_img = ImageTk.PhotoImage(img)
            
            self.canvas1.delete("stim")
            self.canvas2.delete("stim")
            
            cx1 = self.width // 2 + self.offset_x.get()
            cy1 = self.height // 2 + self.offset_y.get()
            cx2 = self.canvas2.winfo_width() // 2
            cy2 = self.canvas2.winfo_height() // 2
            
            self.canvas1.create_image(cx1, cy1, image=self.photo_img, anchor='center', tags="stim")
            self.canvas2.create_image(cx2, cy2, image=self.photo_img, anchor='center', tags="stim")
        except Exception as e:
            print(f"Error loading image: {e}")
    
    def save_and_next(self, event=None):
        """結果保存して次へ"""
        if hasattr(self, 'ctrl_frame') and self.ctrl_frame and self.ctrl_frame.winfo_exists():
            self.ctrl_frame.destroy()
        
        self.clear_key_bindings()
        
        self.add_trial_result({
            "participant_id": self.participant_id.get(),
            "block_id": self.current_block_index + 1,
            "trial_id": self.current_trial_in_experiment + 1,
            "blur_condition": self.current_block_cond["blur_condition"],
            "image_file": os.path.basename(self.current_image_path),
            "visibility_rating": self.evaluation_val.get()
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
    
    def _save_results_and_finish(self):
        """結果保存して終了"""
        if not os.path.exists(RESULT_DIR):
            os.makedirs(RESULT_DIR)
        
        p_id = self.participant_id.get()
        now = datetime.datetime.now()
        date_str = now.strftime("%Y%m%d")
        save_folder = os.path.join(RESULT_DIR, f"{p_id}_{date_str}")
        
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        filename = os.path.join(save_folder, 
                               f"result_{p_id}_{now.strftime('%Y%m%d_%H%M%S')}.csv")
        
        if self.results:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                writer.writeheader()
                writer.writerows(self.results)
        
        messagebox.showinfo("Completed", 
                           f"Experiment finished.\nData saved to: {filename}")
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ExperimentApp(root)
    root.mainloop()
