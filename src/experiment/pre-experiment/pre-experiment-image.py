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
IMG_DIR_1 = BASE_IMG_DIR_1
IMG_DIR_2 = BASE_IMG_DIR_2
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
    
    image固有の処理をオリジナルと全く同じ挙動で実装
    """
    
    def __init__(self, root):
        ExperimentBaseUI.__init__(self, root, PARTICIPANT_DATA_DIR)
        ExperimentTrialLoop.__init__(self)
        
        self.root = root
        self.root.title("Image Experiment - Controller (Window 2)")
        self.root.configure(bg=BG_COLOR)
        
        # image固有の変数
        self.defocus_val = tk.DoubleVar(value=0.0)
        self.distance1 = tk.IntVar(value=50)
        self.distance2 = tk.IntVar(value=70)

        self.trial_list = []
        self.current_trial_index = 0
        self.eval_buttons = []
        
        self.current_img_path_1 = None
        self.current_img_path_2 = None
        self.photo1 = None
        self.photo2 = None
        
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
    
    def setup_experiment_ui(self):
        """
        実験固有の設定UI（被験者登録後に表示される）
        参加者ID/年齢などの登録UIは `ExperimentBaseUI.setup_participant_info_ui` を使用します。
        """
        if hasattr(self, 'participant_frame') and self.participant_frame and self.participant_frame.winfo_exists():
            self.participant_frame.destroy()

        self.participant_frame = tk.Frame(self.root, bg='gray', padx=20, pady=20)
        self.participant_frame.place(relx=0.5, rely=0.5, anchor='center')

        tk.Label(self.participant_frame, text="Experiment Setup", font=("Arial", 16)).grid(row=0, column=0, columnspan=2, pady=10)

        # Participant info fields are managed by base class variables and were populated by check_participant_id/register_and_start
        tk.Label(self.participant_frame, text="Participant ID:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        tk.Label(self.participant_frame, text=self.participant_id.get()).grid(row=1, column=1, padx=5, pady=5)

        tk.Label(self.participant_frame, text="Age:").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        tk.Label(self.participant_frame, text=self.participant_age.get()).grid(row=2, column=1, padx=5, pady=5)

        tk.Label(self.participant_frame, text="Gender:").grid(row=3, column=0, sticky='w', padx=5, pady=5)
        tk.Label(self.participant_frame, text=self.participant_gender.get()).grid(row=3, column=1, padx=5, pady=5)

        tk.Label(self.participant_frame, text="IPD (mm):").grid(row=4, column=0, sticky='w', padx=5, pady=5)
        tk.Label(self.participant_frame, text=self.participant_ipd.get()).grid(row=4, column=1, padx=5, pady=5)

        tk.Label(self.participant_frame, text="Foreground Distance (cm):").grid(row=5, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(self.participant_frame, textvariable=self.distance1).grid(row=5, column=1, padx=5, pady=5)

        tk.Label(self.participant_frame, text="Background Distance (cm):").grid(row=6, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(self.participant_frame, textvariable=self.distance2).grid(row=6, column=1, padx=5, pady=5)

        btn = tk.Button(self.participant_frame, text="Setup Complete, Next", command=self.start_calibration)
        btn.grid(row=7, column=0, columnspan=2, pady=20)
        btn.bind('<Return>', lambda event: self.start_calibration())

    def start_calibration(self):
        # distance の妥当性チェックのみ行い、キャリブレーションへ移行
        try:
            self.distance1.get()
            self.distance2.get()
        except (ValueError, tk.TclError):
            messagebox.showwarning("Input Error", "Please enter valid numbers for experiment settings.")
            return

        self.win1.update_idletasks()
        self.width = self.win1.winfo_width()
        self.height = self.win1.winfo_height()

        if hasattr(self, 'participant_frame') and self.participant_frame and self.participant_frame.winfo_exists():
            self.participant_frame.destroy()
        self.setup_calibration_ui()

    def on_participant_confirmed(self):
        """
        ExperimentBaseUI.check_participant_id() または register_and_start() から呼ばれるコールバック。
        被験者情報がロードされた後、画像実験固有の設定UIを表示する。
        """
        self.win1.update_idletasks()
        self.width = self.win1.winfo_width()
        self.height = self.win1.winfo_height()
        self.setup_experiment_ui()

    def _reset_to_setup_ui(self):
        if hasattr(self, 'ctrl_frame') and self.ctrl_frame and self.ctrl_frame.winfo_exists():
            self.ctrl_frame.destroy()
        if hasattr(self, 'instruction_frame') and self.instruction_frame and self.instruction_frame.winfo_exists():
            self.instruction_frame.destroy()
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        self.setup_participant_info_ui()

    def update_calibration_view(self, *args):
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
        
        stimuli_utils.draw_image_corner_brackets(self.canvas2, fg_size, fg_size, 0, 0, color=WIN2_MARKER_COLOR, flip_x=False)
        stimuli_utils.draw_center_cross(self.canvas2, color=WIN2_MARKER_COLOR)

    def setup_calibration_ui(self, is_break=False):
        # Use base-class calibration UI to handle controls and bindings
        super().setup_calibration_ui(is_break=is_break)

    def resume_experiment(self):
        self.ctrl_frame.destroy()
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        self.run_trial()

    def on_calibration_complete(self):
        """
        キャリブレーション完了時に呼ばれるコールバック。
        実験開始前のキャリブなら `start_experiment` を呼び、休憩後のキャリブなら `resume_experiment` を呼ぶ。
        """
        # remove control frame and calibration markers
        try:
            if self.ctrl_frame and self.ctrl_frame.winfo_exists():
                self.ctrl_frame.destroy()
        except Exception:
            pass

        # decide whether we are mid-experiment (break) or initial calibration
        if hasattr(self, 'trial_list') and self.trial_list and self.current_trial_index > 0 and self.current_trial_index < len(self.trial_list):
            # mid-experiment break -> resume
            self.canvas1.delete("all")
            self.canvas2.delete("all")
            self.resume_experiment()
        else:
            # initial calibration -> start experiment
            self.canvas1.delete("all")
            self.canvas2.delete("all")
            self.start_experiment()

    def start_break(self):
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        self.setup_calibration_ui(is_break=True)

    def start_experiment(self):
        self.clear_key_bindings()

        bg_img_paths = sorted(glob.glob(os.path.join(IMG_DIR_1, '*')))
        fg_img_paths = sorted(glob.glob(os.path.join(IMG_DIR_2, '*')))

        if not bg_img_paths or not fg_img_paths:
            messagebox.showerror("Error", f"Image folder not found or is empty.\n\nBG path: {IMG_DIR_1}\nFG path: {IMG_DIR_2}")
            self._reset_to_setup_ui()
            return

        import itertools
        all_combinations = list(itertools.product(bg_img_paths, fg_img_paths))

        print(f"Found {len(bg_img_paths)} background images and {len(fg_img_paths)} foreground images.")
        print(f"Total trials to be generated: {len(all_combinations)}")

        block_trials = []
        for bg_path, fg_path in all_combinations:
            block_trials.append({"bg_image": bg_path, "fg_image": fg_path})
        
        random.shuffle(block_trials)
        self.trial_list = block_trials
            
        if hasattr(self, 'ctrl_frame') and self.ctrl_frame and self.ctrl_frame.winfo_exists():
            self.ctrl_frame.destroy()
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        self.run_trial()

    def run_trial(self):
        if self.current_trial_index >= len(self.trial_list):
            self.finish_experiment()
            return

        trial_cond = self.trial_list[self.current_trial_index]
        self.current_img_path_1 = trial_cond["bg_image"]
        self.current_img_path_2 = trial_cond["fg_image"]
        img1 = Image.open(self.current_img_path_1)
        img2 = Image.open(self.current_img_path_2)
        
        d_fg = self.distance1.get()
        d_bg = self.distance2.get()
        fg_size = stimuli_utils.get_size_for_visual_angle(d_fg, VISUAL_ANGLE_DEG)

        bg_h = stimuli_utils.get_size_for_visual_angle(d_bg, VISUAL_ANGLE_DEG)
        bg_w = stimuli_utils.get_size_for_visual_angle(d_bg, VISUAL_ANGLE_DEG * 2)

        img1 = img1.resize((512, 512))
        img1 = img1.crop((0, 128, 512, 384))

        img1 = img1.resize((bg_w, bg_h))
        img2 = img2.resize((fg_size, fg_size))

        img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)

        self.photo1 = ImageTk.PhotoImage(img1)
        self.photo2 = ImageTk.PhotoImage(img2)

        self.canvas1.configure(bg='black')
        self.canvas1.delete("all")
        
        self.canvas2.create_image(self.canvas2.winfo_width()//2, self.canvas2.winfo_height()//2, image=self.photo2, anchor='center', tags="img")
        
        self.root.after(TIME_PHASE_1, self.phase_isi)

    def phase_isi(self):
        self.canvas2.delete("img")
        d_fg = self.distance1.get()
        fg_size = stimuli_utils.get_size_for_visual_angle(d_fg, VISUAL_ANGLE_DEG)
        stimuli_utils.draw_image_corner_brackets(self.canvas2, fg_size, fg_size, 0, 0, color=WIN2_MARKER_COLOR, flip_x=True)
        stimuli_utils.draw_center_cross(self.canvas2, color=WIN2_MARKER_COLOR)
        self.root.after(TIME_ISI, self.phase_both)

    def phase_both(self):
        self.canvas2.delete("calib")

        ox, oy = self.offset_x.get(), self.offset_y.get()
        self.canvas1.create_image(self.width//2 + ox, self.height//2 + oy, image=self.photo1, anchor='center', tags="img")
        
        self.canvas2.create_image(self.canvas2.winfo_width()//2, self.canvas2.winfo_height()//2, image=self.photo2, anchor='center', tags="img")

        self.root.after(TIME_PHASE_2, self.phase_end_trial)

    def phase_end_trial(self):
        self.canvas1.delete("img")
        self.canvas2.delete("img")
        self.show_evaluation_ui()

    def show_evaluation_ui(self, num_choices=5, callback=None):
        self.eval_frame = tk.Frame(self.root, bg='white', padx=20, pady=20, relief="solid", borderwidth=1)
        self.eval_frame.place(relx=0.5, rely=0.5, anchor='center')

        tk.Label(self.eval_frame, text=f"Trial No.{self.current_trial_index + 1} の評価", font=("Arial", 16), bg='white').pack(pady=(0, 20))
        
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
            self.distance1.get(), self.distance2.get(),
            self.offset_x.get(), self.offset_y.get(), self.defocus_val.get(),
            self.current_trial_index + 1,
            f1, f2, score
        ])
        
        self.eval_frame.destroy()
        self.current_trial_index += 1
        
        is_break_time = (self.current_trial_index > 0) and \
                        (self.current_trial_index % NUM_TRIALS_BEFORE_BREAK == 0) and \
                        (self.current_trial_index < len(self.trial_list))
        
        if is_break_time:
            self.root.after(500, self.start_break)
        else:
            self.root.after(500, self.run_trial)

    def finish_experiment(self):
        if not os.path.exists(RESULT_DIR):
            os.makedirs(RESULT_DIR)
            
        p_id = self.participant_id.get()
        now = datetime.datetime.now()
        date_str = now.strftime("%Y%m%d_%H%M%S")
        save_folder = os.path.join(RESULT_DIR, f"{p_id}_{date_str}")
        
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        filename = os.path.join(save_folder, f"result_{p_id}_{date_str}.csv")
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = [
                "ID", "Age", "Gender", "IPD(mm)", "Distance1(cm)", "Distance2(cm)",
                "Offset_X", "Offset_Y", "Defocus_D", "Trial_ID", "Image_Win1", "Image_Win2", "Score"
            ]
            writer.writerow(header)
            writer.writerows(self.results)
            
        messagebox.showinfo("Finished", f"Experiment finished.\nData saved to: {filename}")
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ExperimentApp(root)
    root.mainloop()
