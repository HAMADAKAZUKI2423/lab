"""
Gabor 実験スクリプト（削減版）

Gabor刺激とノイズの視認性評価実験
基盤クラス（ExperimentBaseUI, ExperimentTrialLoop）から共通処理を継承
gabor固有のロジックのみを実装
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

TIME_PHASE_1 = 1600
TIME_ISI = 1
TIME_PHASE_2 = 5000

script_dir = os.path.dirname(os.path.abspath(__file__))
lab_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

BASE_IMG_DIR_1 = os.path.join(lab_root, "data", "processed", "images", "pre-experiment-gabor", "bg_noise")
BASE_IMG_DIR_2 = os.path.join(lab_root, "data", "processed", "images", "pre-experiment-gabor", "fg_gabor")
RESULT_DIR = os.path.join(lab_root, "results", "tables", "pre-experiment-gabor")
FIGURE_DIR = os.path.join(lab_root, "results", "figures", "pre-experiment-gabor")
PARTICIPANT_DATA_DIR = os.path.join(lab_root, "data", "processed", "tables", "pre-experiment-gabor")

for dir_path in [RESULT_DIR, FIGURE_DIR, PARTICIPANT_DATA_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

BG_COLOR = 'black'
WIN1_MARKER_COLOR = 'red'
WIN2_MARKER_COLOR = 'white'


class ExperimentApp(ExperimentBaseUI, ExperimentTrialLoop):
    """
    Gabor 実験
    
    基盤クラスから継承：
    - ExperimentBaseUI: UI処理
    - ExperimentTrialLoop: 試行ループ
    
    gabor固有の処理をオリジナルと全く同じ挙動で実装
    """
    
    def __init__(self, root):
        ExperimentBaseUI.__init__(self, root, PARTICIPANT_DATA_DIR)
        ExperimentTrialLoop.__init__(self)
        
        self.root = root
        self.root.title("Gabor Experiment - Controller (Window 2)")
        self.root.configure(bg=BG_COLOR)
        
        # gabor固有の変数
        self.distance1 = tk.IntVar(value=50)
        self.distance2 = tk.IntVar(value=70)
        self.viewing_condition = tk.StringVar(value="Binocular")
        self.spatial_freq = tk.StringVar(value="2,8")
        
        self.blocks = []
        self.current_block_index = 0
        self.current_trial_in_block = 0
        self.current_block_cond = None
        self.eval_buttons = []
        
        self.pupil_diameter_val = tk.DoubleVar(value=4.0)
        self.current_pd_mean = 0.0
        self.current_pd_std = 0.0
        
        self.current_img_path_1 = None
        self.current_img_path_2 = None
        self.photo1 = None
        self.photo2 = None
        self.max_contrast_gabor_path = None
        self.photo_max_gabor = None
        
        # ウィンドウのセットアップ
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        
        self.root.state('zoomed')
        
        self.win1 = tk.Toplevel(self.root)
        self.win1.title("Gabor Experiment - Display (Window 1)")
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
    
    def setup_participant_info_ui(self):
        """ステップ0: 実験設定UIを構築し表示する"""
        if hasattr(self, 'participant_frame') and self.participant_frame and self.participant_frame.winfo_exists():
            self.participant_frame.destroy()
            
        self.participant_frame = tk.Frame(self.root, bg='gray', padx=20, pady=20)
        self.participant_frame.place(relx=0.5, rely=0.5, anchor='center')

        tk.Label(self.participant_frame, text="Experiment Setup", font=("Arial", 16)).grid(row=0, column=0, columnspan=2, pady=10)

        tk.Label(self.participant_frame, text="Participant ID:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        entry_id = tk.Entry(self.participant_frame, textvariable=self.participant_id)
        entry_id.grid(row=1, column=1, padx=5, pady=5)
        entry_id.focus_set()

        tk.Label(self.participant_frame, text="Age:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(self.participant_frame, textvariable=self.participant_age).grid(row=2, column=1, padx=5, pady=5)

        tk.Label(self.participant_frame, text="Gender:").grid(row=3, column=0, sticky='w', padx=5, pady=5)
        gender_combo = ttk.Combobox(self.participant_frame, textvariable=self.participant_gender, values=["Male", "Female", "Other"])
        gender_combo.grid(row=3, column=1, padx=5, pady=5)
        gender_combo.set("Male")

        tk.Label(self.participant_frame, text="IPD (mm):").grid(row=4, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(self.participant_frame, textvariable=self.participant_ipd).grid(row=4, column=1, padx=5, pady=5)

        tk.Label(self.participant_frame, text="Foreground Distance (cm):").grid(row=5, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(self.participant_frame, textvariable=self.distance1).grid(row=5, column=1, padx=5, pady=5)

        tk.Label(self.participant_frame, text="Background Distance (cm):").grid(row=6, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(self.participant_frame, textvariable=self.distance2).grid(row=6, column=1, padx=5, pady=5)

        tk.Label(self.participant_frame, text="First Viewing Condition:").grid(row=7, column=0, sticky='w', padx=5, pady=5)
        view_combo = ttk.Combobox(self.participant_frame, textvariable=self.viewing_condition, values=["Binocular", "Monocular"])
        view_combo.grid(row=7, column=1, padx=5, pady=5)
        view_combo.set("Binocular")

        tk.Label(self.participant_frame, text="Spatial Freqs (cpd, comma-separated):").grid(row=8, column=0, sticky='w', padx=5, pady=5)
        cpd_entry = tk.Entry(self.participant_frame, textvariable=self.spatial_freq)
        cpd_entry.grid(row=8, column=1, padx=5, pady=5)

        btn = tk.Button(self.participant_frame, text="Setup Complete, Next", command=self.start_calibration)
        btn.grid(row=9, column=0, columnspan=2, pady=20)
        btn.bind('<Return>', lambda event: self.start_calibration())

    def start_calibration(self):
        """実験設定の入力を検証し、問題なければブロックを初期化して開始する"""
        if not self.participant_id.get() or not self.participant_age.get() or not self.participant_ipd.get():
            messagebox.showwarning("Input Error", "Please enter ID, Age, and IPD.")
            return
        
        try:
            self.distance1.get()
            self.distance2.get()
            spatial_freqs_str = self.spatial_freq.get().strip()
            if not spatial_freqs_str:
                raise ValueError("Spatial frequency cannot be empty.")
            selected_spatial_freqs = [int(s.strip()) for s in spatial_freqs_str.split(',')]
        except (ValueError, tk.TclError):
            messagebox.showwarning("Input Error", "Please enter valid numbers for experiment settings.\nSpatial Freqs must be comma-separated numbers (e.g., 2,8).")
            return

        self.win1.update_idletasks()
        self.width = self.win1.winfo_width()
        self.height = self.win1.winfo_height()

        first_view = self.viewing_condition.get()
        second_view = "Monocular" if first_view == "Binocular" else "Binocular"
        view_conditions = [first_view, second_view]
        self.blocks = []
        for view in view_conditions:
            for cpd in selected_spatial_freqs:
                self.blocks.append({"viewing_condition": view, "spatial_freq": str(cpd)})
        
        random.shuffle(self.blocks)
        
        self.current_block_index = 0
        self.current_trial_in_experiment = 0

        self.participant_frame.destroy()
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
        """ブロック確認UI"""
        if hasattr(self, 'ctrl_frame') and self.ctrl_frame and self.ctrl_frame.winfo_exists():
            self.ctrl_frame.destroy()
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        
        self.clear_key_bindings()
        
        self.ctrl_frame = tk.Frame(self.root, bg='gray')
        self.ctrl_frame.place(relx=0.5, rely=0.5, anchor='center')
        
        v_cond = self.current_block_cond["viewing_condition"]
        cpd_cond = self.current_block_cond["spatial_freq"]
        instruction_text = f"Next section is {cpd_cond} cpd, {v_cond}, OK?\n\nPress 'Enter' to continue."
        
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
        
        self.setup_calibration_ui(is_new_block=True)
        
    def update_calibration_view(self):
        """キャリブレーション画面更新"""
        self.canvas1.delete("calib")
        self.canvas2.delete("calib")
        
        d_fg = self.distance1.get()
        d_bg = self.distance2.get()
        
        fg_size = stimuli_utils.get_size_for_visual_angle(d_fg, VISUAL_ANGLE_DEG)
        bg_h = stimuli_utils.get_size_for_visual_angle(d_bg, VISUAL_ANGLE_DEG)
        bg_w = stimuli_utils.get_size_for_visual_angle(d_bg, VISUAL_ANGLE_DEG * 2)
        
        stimuli_utils.draw_image_corner_brackets(self.canvas1, bg_w, bg_h, 
                                                self.offset_x.get(), self.offset_y.get(), 
                                                color=WIN1_MARKER_COLOR, line_width=stimuli_utils.MARKER_LINE_WIDTH * 1.5)
        
        stimuli_utils.draw_image_corner_brackets(self.canvas1, bg_h, bg_h, 
                                                self.offset_x.get(), self.offset_y.get(), 
                                                color=WIN1_MARKER_COLOR, line_width=stimuli_utils.MARKER_LINE_WIDTH * 1.5)
        
        stimuli_utils.draw_image_corner_brackets(self.canvas2, fg_size, fg_size, 0, 0, 
                                                color=WIN2_MARKER_COLOR, flip_x=False, line_width=stimuli_utils.MARKER_LINE_WIDTH)
        stimuli_utils.draw_center_cross(self.canvas2, color=WIN2_MARKER_COLOR)

    def setup_calibration_ui(self, is_break=False, is_new_block=False):
        self.update_calibration_view()
        
        self.ctrl_frame = tk.Frame(self.root, bg='gray')
        self.ctrl_frame.place(relx=0.5, rely=0.8, anchor='center')

        self.clear_key_bindings()

        if is_break:
            instruction_text = "This is a break. You can adjust the position if needed.\nPress 'Resume Experiment' to continue."
            button_text = "Resume Experiment"
            button_command = self.resume_experiment
        elif is_new_block:
            v_cond = self.current_block_cond["viewing_condition"]
            cpd_cond = self.current_block_cond["spatial_freq"]
            instruction_text = f"[Block {self.current_block_index + 1}/4]\nCondition: {v_cond}, Spatial Freq: {cpd_cond} cpd\n\nPlease set up for {v_cond} viewing.\nUse the arrow keys to adjust the position of the red frame."
            button_text = "Calibration Done, Next"
            button_command = lambda: defocus_matching.setup_defocus_matching_ui_gabor(self)
        else:
            instruction_text = "Use the arrow keys to adjust the position of the red frame."
            button_text = "Calibration Done, Next"
            button_command = lambda: defocus_matching.setup_defocus_matching_ui_gabor(self)

        tk.Label(self.ctrl_frame, text=instruction_text, bg='gray', fg='white', font=("Arial", 12)).pack(pady=10, padx=20)

        btn = tk.Button(self.ctrl_frame, text=button_text, command=button_command)
        btn.pack(pady=10)
        btn.focus_set()
        
        self.key_bindings['<Return>'] = self.root.bind('<Return>', lambda event: button_command())
        self.key_bindings['<Left>'] = self.root.bind('<Left>', lambda e: self.adjust_offset(-1, 0))
        self.key_bindings['<Right>'] = self.root.bind('<Right>', lambda e: self.adjust_offset(1, 0))
        self.key_bindings['<Up>'] = self.root.bind('<Up>', lambda e: self.adjust_offset(0, -1))
        self.key_bindings['<Down>'] = self.root.bind('<Down>', lambda e: self.adjust_offset(0, 1))

    def resume_experiment(self):
        self.ctrl_frame.destroy()
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        self.run_trial()

    def start_break(self):
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        self.setup_calibration_ui(is_break=True)

    def start_experiment_block(self):
        """ブロック実験開始"""
        self.clear_key_bindings()
        
        d_fg = self.distance1.get()
        d_bg = self.distance2.get()
        cpd = self.current_block_cond["spatial_freq"]
            
        bg_img_dir = os.path.join(BASE_IMG_DIR_1, f'{d_bg}cm', f'{cpd}cpd')
        fg_img_dir = os.path.join(BASE_IMG_DIR_2, f'{d_fg}cm', f'{cpd}cpd')

        max_gabor_files = glob.glob(os.path.join(fg_img_dir, 'FG_50_1.0_*.png'))
        if not max_gabor_files:
            messagebox.showerror("Error", f"Max contrast Gabor patch (FG_50_1.0_*.png) not found in:\n{fg_img_dir}")
            self._reset_to_setup_ui()
            return
        self.max_contrast_gabor_path = max_gabor_files[0]

        bg_img_paths = sorted(glob.glob(os.path.join(bg_img_dir, '*')))
        fg_img_paths = sorted(glob.glob(os.path.join(fg_img_dir, '*')))

        if not bg_img_paths or not fg_img_paths:
            messagebox.showerror("Error", f"Image folder not found or is empty for the specified distances.\n\nBG path: {bg_img_dir}\nFG path: {fg_img_dir}")
            self._reset_to_setup_ui()
            return

        block_trials = []
        bg_path_dict = {os.path.basename(p): p for p in bg_img_paths}

        for fg_path in fg_img_paths:
            fg_basename = os.path.basename(fg_path)
            if fg_basename in bg_path_dict:
                bg_path = bg_path_dict[fg_basename]
                block_trials.append({"bg_image": bg_path, "fg_image": fg_path})
            else:
                print(f"Warning: Corresponding background image not found for '{fg_basename}'. Skipping.")

        block_trials = block_trials * NUM_REPETITIONS
        random.shuffle(block_trials)
        self.trial_list = block_trials
            
        if hasattr(self, 'ctrl_frame') and self.ctrl_frame and self.ctrl_frame.winfo_exists():
            self.ctrl_frame.destroy()
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        self.run_trial()
    
    def run_trial(self):
        """試行実行"""
        trial = self.trial_list[self.current_trial_in_block]
        self.current_img_path_1 = trial["bg_image"]
        self.current_img_path_2 = trial["fg_image"]
        img1 = Image.open(self.current_img_path_1)
        img2 = Image.open(self.current_img_path_2)

        img_max_gabor = Image.open(self.max_contrast_gabor_path)

        d_fg = self.distance1.get()
        d_bg = self.distance2.get()
        fg_size = stimuli_utils.get_size_for_visual_angle(d_fg, VISUAL_ANGLE_DEG)

        bg_h = stimuli_utils.get_size_for_visual_angle(d_bg, VISUAL_ANGLE_DEG)
        bg_w = stimuli_utils.get_size_for_visual_angle(d_bg, VISUAL_ANGLE_DEG * 2)
        img1 = img1.resize((bg_w, bg_h))
        img2 = img2.resize((fg_size, fg_size))
        img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)

        img_max_gabor = img_max_gabor.resize((fg_size, fg_size))
        img_max_gabor = img_max_gabor.transpose(Image.FLIP_LEFT_RIGHT)

        self.photo1 = ImageTk.PhotoImage(img1)
        self.photo2 = ImageTk.PhotoImage(img2)
        self.photo_max_gabor = ImageTk.PhotoImage(img_max_gabor)
        
        self.canvas1.configure(bg='black')
        self.canvas1.delete("all")
        
        self.canvas2.create_image(self.canvas2.winfo_width()//2, self.canvas2.winfo_height()//2, image=self.photo_max_gabor, anchor='center', tags="img")
        
        stimuli_utils.draw_image_corner_brackets(self.canvas2, fg_size, fg_size, 0, 0, color=WIN2_MARKER_COLOR, flip_x=True)
        stimuli_utils.draw_center_cross(self.canvas2, color=WIN2_MARKER_COLOR, gap=30)
        
        self.root.after(TIME_PHASE_1, self.phase_isi)

    def phase_isi(self):
        self.canvas2.delete("img")
        self.canvas2.delete("calib")
        d_fg = self.distance1.get()
        fg_marker_size = stimuli_utils.get_size_for_visual_angle(d_fg, VISUAL_ANGLE_DEG)
        stimuli_utils.draw_image_corner_brackets(self.canvas2, fg_marker_size, fg_marker_size, 0, 0, color=WIN2_MARKER_COLOR, flip_x=True)
        stimuli_utils.draw_center_cross(self.canvas2, color=WIN2_MARKER_COLOR)
        self.root.after(TIME_ISI, self.phase_both)

    def phase_both(self):
        self.canvas2.delete("calib")
        ox, oy = self.offset_x.get(), self.offset_y.get()
        self.canvas1.create_image(self.width//2 + ox, self.height//2 + oy, image=self.photo1, anchor='center', tags="img")
        self.canvas2.create_image(self.canvas2.winfo_width()//2, self.canvas2.winfo_height()//2, image=self.photo2, anchor='center', tags="img")
        d_fg = self.distance1.get()
        fg_marker_size = stimuli_utils.get_size_for_visual_angle(d_fg, VISUAL_ANGLE_DEG)
        stimuli_utils.draw_image_corner_brackets(self.canvas2, fg_marker_size, fg_marker_size, 0, 0, color=WIN2_MARKER_COLOR, flip_x=True)
        stimuli_utils.draw_center_cross(self.canvas2, color=WIN2_MARKER_COLOR, gap=30)
        self.root.after(TIME_PHASE_2, self.phase_end_trial)

    def phase_end_trial(self):
        self.canvas1.delete("img")
        self.canvas2.delete("img")
        self.canvas2.delete("calib")
        self.show_evaluation_ui()

    def show_evaluation_ui(self, num_choices=5, callback=None):
        self.eval_frame = tk.Frame(self.root, bg='white', padx=20, pady=20, relief="solid", borderwidth=1)
        self.eval_frame.place(relx=0.5, rely=0.5, anchor='center')

        tk.Label(self.eval_frame, text=f"Trial No.{self.current_trial_in_experiment + 1} の評価", font=("Arial", 16), bg='white').pack(pady=(0, 20))
        
        self.evaluation_val.set(3)

        options_frame = tk.Frame(self.eval_frame, bg='white')
        options_frame.pack(pady=10, padx=20)

        self.eval_buttons.clear()
        for i in range(5, 0, -1):
            option_frame = tk.Frame(options_frame, bg='white')
            option_frame.pack(side='left', padx=15)

            canvas = tk.Canvas(option_frame, width=30, height=30, bg='white', highlightthickness=0)
            canvas.pack()
            
            canvas.create_oval(5, 5, 25, 25, outline='black', width=2)
            dot_item = canvas.create_oval(10, 10, 20, 20, fill='white', outline='white')
            
            label = tk.Label(option_frame, text=str(i), font=("Arial", 12), bg='white')
            label.pack()
            
            self.eval_buttons.append({'canvas': canvas, 'dot': dot_item, 'label': label})

        desc_frame = tk.Frame(self.eval_frame, bg='white')
        desc_frame.pack(fill='x', padx=10, pady=(5, 10))
        tk.Label(desc_frame, text="5: Very clear", bg='white').pack(side='left')
        tk.Label(desc_frame, text="1: Invisible", bg='white').pack(side='right')

        tk.Label(self.eval_frame, text="◀ / ▶ で選択, ▼ で決定", font=("Arial", 10), bg='white').pack(pady=(10, 0))

        self._update_eval_highlight()

        self.key_bindings['<Left>'] = self.root.bind('<Left>', lambda e: self._move_selection(-1))
        self.key_bindings['<Right>'] = self.root.bind('<Right>', lambda e: self._move_selection(1))
        self.key_bindings['<Down>'] = self.root.bind('<Down>', lambda e: self.save_and_next())
        self.root.focus_set()

    def _update_eval_highlight(self):
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
        current = self.evaluation_val.get()
        new_val = current + delta
        new_val = max(1, min(5, new_val))
        if new_val != current:
            self.evaluation_val.set(new_val)
            self._update_eval_highlight()
        return "break"

    def save_and_next(self, callback=None):
        self.clear_key_bindings()
        
        score = self.evaluation_val.get()
        f1 = os.path.basename(self.current_img_path_1)
        f2 = os.path.basename(self.current_img_path_2)
        
        self.results.append([
            self.participant_id.get(), self.participant_age.get(), self.participant_gender.get(), self.participant_ipd.get(),
            self.current_block_cond["viewing_condition"],
            self.distance1.get(), self.distance2.get(),
            self.current_block_cond["spatial_freq"],
            self.offset_x.get(), self.offset_y.get(),
            round(self.current_pd_mean, 3), round(self.current_pd_std, 3),
            self.current_trial_in_experiment + 1,
            f1, f2, score
        ])
        
        self.eval_frame.destroy()
        
        self.current_trial_in_block += 1
        self.current_trial_in_experiment += 1
        
        is_break_time = (self.current_trial_in_experiment > 0) and \
                        (self.current_trial_in_experiment % NUM_TRIALS_BEFORE_BREAK == 0)

        if self.current_trial_in_block >= len(self.trial_list):
            self.current_block_index += 1
            if self.current_block_index >= len(self.blocks):
                self.finish_experiment()
            else:
                self.root.after(500, self.start_block)
        elif is_break_time:
            self.root.after(500, self.start_break)
        else:
            self.root.after(500, self.run_trial)
    
    def finish_experiment(self):
        if not os.path.exists(RESULT_DIR):
            os.makedirs(RESULT_DIR)
            
        p_id = self.participant_id.get()
        now = datetime.datetime.now()
        date_str = now.strftime("%Y%m%d")
        save_folder = os.path.join(RESULT_DIR, f"{p_id}_{date_str}")
        
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        filename = os.path.join(save_folder, f"result_{p_id}_{now.strftime('%Y%m%d_%H%M%S')}.csv")
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = [
                "ID", "Age", "Gender", "IPD(mm)", "Viewing_Condition", "Distance1(cm)", "Distance2(cm)",
                "Spatial_Freq(cpd)",
                "Offset_X", "Offset_Y", "Match_PD_Mean", "Match_PD_Std", "Trial_ID", "Image_Win1", "Image_Win2", "Score"
            ]
            writer.writerow(header)
            writer.writerows(self.results)
            
        messagebox.showinfo("Finished", f"Experiment finished.\nData saved to: {filename}")
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ExperimentApp(root)
    root.mainloop()
