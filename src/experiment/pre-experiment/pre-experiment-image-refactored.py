"""
画像実験スクリプト（削減版）

自然画像の視認性評価実験
基盤クラス（ExperimentBaseUI, ExperimentTrialLoop）から共通処理を継承
image固有のロジックのみを実装
"""

import tkinter as tk
from tkinter import messagebox
import os
import csv
import datetime
import random
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

TIME_PHASE_1 = 1600
TIME_ISI = 1
TIME_PHASE_2 = 5000

script_dir = os.path.dirname(os.path.abspath(__file__))
lab_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

IMG_DIR = os.path.join(lab_root, "data", "processed", "images", "pre-experiment-image")
RESULT_DIR = os.path.join(lab_root, "results", "tables", "pre-experiment-image")

for dir_path in [RESULT_DIR]:
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
        ExperimentBaseUI.__init__(self, root, os.path.join(lab_root, "data", "processed", "tables", "pre-experiment-image"))
        ExperimentTrialLoop.__init__(self)
        
        self.root = root
        self.root.title("Image Experiment - Controller (Window 2)")
        self.root.configure(bg=BG_COLOR)
        
        # image固有の変数
        self.distance1 = tk.IntVar(value=50)
        self.distance2 = tk.IntVar(value=70)
        self.defocus_val = tk.DoubleVar(value=0.0)
        
        self.blocks = []
        self.current_block_index = 0
        self.current_trial_in_block = 0
        self.current_block_cond = None
        
        self.evaluation_val = tk.IntVar(value=3)
        
        # 画像リスト
        self.image_list = []
        self.load_images()
        
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
    
    def load_images(self):
        """画像をロード"""
        if os.path.exists(IMG_DIR):
            self.image_list = glob.glob(os.path.join(IMG_DIR, "*.png"))
            self.image_list += glob.glob(os.path.join(IMG_DIR, "*.jpg"))
        
        if not self.image_list:
            print("Warning: No images found in", IMG_DIR)
    
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
        tk.Entry(frame, textvariable=self.distance1).grid(row=1, column=1, padx=5, pady=5)
        
        tk.Label(frame, text="Background Distance (cm):").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(frame, textvariable=self.distance2).grid(row=2, column=1, padx=5, pady=5)
        
        btn = tk.Button(frame, text="Setup Complete", command=lambda: self._on_setup_complete(frame))
        btn.grid(row=3, column=0, columnspan=2, pady=20)
    
    def _on_setup_complete(self, frame):
        """実験設定完了"""
        try:
            self.distance1.get()
            self.distance2.get()
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
