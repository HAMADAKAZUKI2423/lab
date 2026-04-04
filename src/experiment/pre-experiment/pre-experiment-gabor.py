# py .\src\experiment\pre-experiment\pre-experiment-gabor.py
import tkinter as tk
from tkinter import ttk, messagebox
import os
import csv
import datetime
from PIL import Image, ImageTk, ImageEnhance
import glob
import random
import math

# ==========================================
# 定数設定エリア (実験条件やデザインはここを変更)
# ==========================================
# --- 実験設定 ---
VISUAL_ANGLE_DEG = 7.9   # 画像の視角 (degree)
script_dir = os.path.dirname(os.path.abspath(__file__))
lab_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
IMG_DIR_1 = os.path.join(lab_root, "data", "processed", "images", "pre-experiment-gabor", "bg_noise")
IMG_DIR_2 = os.path.join(lab_root, "data", "processed", "images", "pre-experiment-gabor", "fg_gabor")
RESULT_DIR = os.path.join(lab_root, "results", "tables", "pre-experiment-gabor")

# --- 時間設定 (ミリ秒) ---
TIME_PHASE_1 = 1600    # Phase 1: Image display only on Win2
TIME_ISI = 1000        # Phase 2: Inter Stimulus Interval (black screen)
TIME_PHASE_2 = 1600    # Phase 3: Image display on both windows

# --- UIデザイン設定 ---
BG_COLOR = 'black'     # 全体の背景色
# NOTE: 以下のPPC(Pixel Per Centimeter)は使用するモニタに合わせて要調整
PIXELS_PER_CM = 1/0.02331  # モニタのPPC (1mmあたり0.2331画素の場合)
SQUARE_SIZE = 30       # 四隅のマーカーの辺の長さ (px)
CROSS_SIZE = 30        # 中央の十字マーカーのサイズ (px)
MARKER_LINE_WIDTH = 5  # マーカーの線の太さ
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
        self.evaluation_val = tk.IntVar(value=3)
        self.participant_age = tk.StringVar()
        self.participant_gender = tk.StringVar()
        self.participant_ipd = tk.StringVar()
        self.participant_id = tk.StringVar()

        # --- 実験条件用変数 ---
        self.distance1 = tk.IntVar(value=50)
        self.distance2 = tk.IntVar(value=70)
        self.trial_list = []
        self.num_trials_per_block = 0
        self.current_trial_index = 0
        self.results = []
        self.key_bindings = {}
        self.eval_buttons = []
        
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
        """ステップ0: 実験設定UIを構築し表示する"""
        # UI要素をまとめるためのフレーム
        self.participant_frame = tk.Frame(self.root, bg='gray', padx=20, pady=20)
        self.participant_frame.place(relx=0.5, rely=0.5, anchor='center')

        tk.Label(self.participant_frame, text="Experiment Setup", font=("Arial", 16)).grid(row=0, column=0, columnspan=2, pady=10)

        tk.Label(self.participant_frame, text="Participant ID:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        entry_id = tk.Entry(self.participant_frame, textvariable=self.participant_id)
        entry_id.grid(row=1, column=1, padx=5, pady=5)
        entry_id.focus_set()

        tk.Label(self.participant_frame, text="Age:").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(self.participant_frame, textvariable=self.participant_age).grid(row=2, column=1, padx=5, pady=5)

        tk.Label(self.participant_frame, text="Gender:").grid(row=3, column=0, sticky='w', padx=5, pady=5)
        gender_combo = ttk.Combobox(self.participant_frame, textvariable=self.participant_gender, values=["Male", "Female", "Other"])
        gender_combo.grid(row=3, column=1, padx=5, pady=5)
        gender_combo.set("Male") # Default value

        tk.Label(self.participant_frame, text="IPD (mm):").grid(row=4, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(self.participant_frame, textvariable=self.participant_ipd).grid(row=4, column=1, padx=5, pady=5)

        tk.Label(self.participant_frame, text="Foreground Distance (cm):").grid(row=5, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(self.participant_frame, textvariable=self.distance1).grid(row=5, column=1, padx=5, pady=5)

        tk.Label(self.participant_frame, text="Background Distance (cm):").grid(row=6, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(self.participant_frame, textvariable=self.distance2).grid(row=6, column=1, padx=5, pady=5)

        btn = tk.Button(self.participant_frame, text="Setup Complete, Next", command=self.start_calibration)
        btn.grid(row=7, column=0, columnspan=2, pady=20)
        btn.bind('<Return>', lambda event: self.start_calibration())

    def start_calibration(self):
        """実験設定の入力を検証し、問題なければキャリブレーションステップに進む"""
        if not self.participant_id.get() or not self.participant_age.get() or not self.participant_ipd.get():
            messagebox.showwarning("Input Error", "Please enter ID, Age, and IPD.")
            return
        
        try:
            self.distance1.get()
            self.distance2.get()
        except (ValueError, tk.TclError):
            messagebox.showwarning("Input Error", "Please enter valid numbers for experiment settings.")
            return

        # 実験設定確定時に、Window 1 (被験者用画面) の実際のサイズを取得して更新する
        self.win1.update_idletasks()
        self.width = self.win1.winfo_width()
        self.height = self.win1.winfo_height()

        self.participant_frame.destroy()
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

    def draw_center_cross(self, canvas, color='white'):
        """画面中央に十字マーカーを描画する"""
        cx, cy = canvas.winfo_width() // 2, canvas.winfo_height() // 2
        l = CROSS_SIZE // 2
        canvas.create_line(cx - l, cy, cx + l, cy, fill=color, width=5, tags="calib")
        canvas.create_line(cx, cy - l, cx, cy + l, fill=color, width=5, tags="calib")

    def update_calibration_view(self, *args):
        """キャリブレーション画面の表示を更新する (スライダー操作時に呼ばれる)"""
        # 既存のマーカーを一旦すべて削除
        self.canvas1.delete("calib")
        self.canvas2.delete("calib")

        d_fg = self.distance1.get()
        d_bg = self.distance2.get()

        # --- 前景マーカーのサイズ計算 (正方形) ---
        fg_marker_size = self.get_size_for_visual_angle(d_fg, VISUAL_ANGLE_DEG)
        
        # --- 背景マーカーのサイズ計算 (横長) ---
        # 背景画像は前景の2倍の幅を持つ
        bg_marker_h = self.get_size_for_visual_angle(d_bg, VISUAL_ANGLE_DEG)
        bg_marker_w = self.get_size_for_visual_angle(d_bg, VISUAL_ANGLE_DEG * 2) # 幅は視角2倍で計算

        # --- Window 1 (被験者側) のマーカー描画 ---
        # 1. 背景全体（横長）の四隅にマーカーを描画
        self.draw_image_corner_brackets(self.canvas1, bg_marker_w, bg_marker_h, self.offset_x.get(), self.offset_y.get(), color=WIN1_MARKER_COLOR, line_width=MARKER_LINE_WIDTH * 1.5)
        # 2. 背景の中央に、前景と同じサイズの正方形マーカーを描画
        self.draw_image_corner_brackets(self.canvas1, bg_marker_h, bg_marker_h, self.offset_x.get(), self.offset_y.get(), color=WIN1_MARKER_COLOR, line_width=MARKER_LINE_WIDTH * 1.5)
        
        # --- Window 2 (実験者側) のマーカー描画 ---
        # 基準となる前景サイズのマーカーと十字を描画
        self.draw_image_corner_brackets(self.canvas2, fg_marker_size, fg_marker_size, 0, 0, color=WIN2_MARKER_COLOR, flip_x=False, line_width=MARKER_LINE_WIDTH)
        self.draw_center_cross(self.canvas2, color=WIN2_MARKER_COLOR)

    def adjust_offset(self, dx, dy):
        """矢印キーによるオフセット調整用関数"""
        self.offset_x.set(self.offset_x.get() + dx)
        self.offset_y.set(self.offset_y.get() + dy)
        self.update_calibration_view()
        return "break" # デフォルトのイベント処理（スライダーの移動など）を停止する

    def setup_calibration_ui(self):
        """ステップ1: キャリブレーション用UIを構築し表示する"""
        self.update_calibration_view()
        
        # 操作用UIをまとめるためのフレーム
        self.ctrl_frame = tk.Frame(self.root, bg='gray')
        self.ctrl_frame.place(relx=0.5, rely=0.8, anchor='center')

        # 位置調整の指示ラベル
        instruction_text = "Use the arrow keys to adjust the position of the red frame."
        tk.Label(self.ctrl_frame, text=instruction_text, bg='gray', fg='white', font=("Arial", 12)).pack(pady=10, padx=20)

        # 実験開始ボタン
        btn = tk.Button(self.ctrl_frame, text="Start Experiment", command=self.start_experiment)
        btn.pack(pady=10)
        btn.focus_set() # ボタンにフォーカスを当ててキー入力を受け付ける
        btn.bind('<Return>', lambda event: self.start_experiment())
        btn.bind('<Left>', lambda e: self.adjust_offset(-1, 0))
        btn.bind('<Right>', lambda e: self.adjust_offset(1, 0))
        btn.bind('<Up>', lambda e: self.adjust_offset(0, -1))
        btn.bind('<Down>', lambda e: self.adjust_offset(0, 1))

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
        # --- 1. Get experiment parameters and load image paths ---
        d_fg = self.distance1.get()
        d_bg = self.distance2.get()
        bg_img_dir = os.path.join(IMG_DIR_1, f'{d_bg}cm')
        fg_img_dir = os.path.join(IMG_DIR_2, f'{d_fg}cm')

        bg_img_paths = sorted(glob.glob(os.path.join(bg_img_dir, '*')))
        fg_img_paths = sorted(glob.glob(os.path.join(fg_img_dir, '*')))

        if not bg_img_paths or not fg_img_paths:
            messagebox.showerror("Error", f"Image folder not found or is empty for the specified distances.\n\nBG path: {bg_img_dir}\nFG path: {fg_img_dir}")
            self._reset_to_setup_ui()
            return

        # --- 3. Build the trial list ---
        block_trials = []
        
        # Create a dictionary of background images for quick lookup
        bg_path_dict = {os.path.basename(p): p for p in bg_img_paths}

        # For each foreground image, find the background image with the same name to create a pair
        for fg_path in fg_img_paths:
            fg_basename = os.path.basename(fg_path)
            if fg_basename in bg_path_dict:
                bg_path = bg_path_dict[fg_basename]
                block_trials.append({"bg_image": bg_path, "fg_image": fg_path})
            else:
                print(f"Warning: Corresponding background image not found for '{fg_basename}'. Skipping.")

        print(f"Total trials generated: {len(block_trials)}") # 64試行になるはず

        random.shuffle(block_trials)
        self.trial_list = block_trials
            
        # --- 5. Start the experiment ---
        self.ctrl_frame.destroy() # キャリブレーションUIを削除
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

        # Resize images based on visual angle and distance
        d_fg = self.distance1.get()
        d_bg = self.distance2.get()
        fg_size = self.get_size_for_visual_angle(d_fg, VISUAL_ANGLE_DEG)

        # 背景は横長(幅が視角の2倍)なので、幅と高さを別々に計算してリサイズ
        bg_h = self.get_size_for_visual_angle(d_bg, VISUAL_ANGLE_DEG)
        bg_w = self.get_size_for_visual_angle(d_bg, VISUAL_ANGLE_DEG * 2)
        img1 = img1.resize((bg_w, bg_h))
        img2 = img2.resize((fg_size, fg_size))
        img2 = img2.transpose(Image.FLIP_LEFT_RIGHT) # Flip for experimenter view

        # Convert to PhotoImage (must be done before displaying)
        self.photo1 = ImageTk.PhotoImage(img1)
        self.photo2 = ImageTk.PhotoImage(img2)
        
        # --- 2. Start trial sequence ---
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
        d_fg = self.distance1.get()
        fg_marker_size = self.get_size_for_visual_angle(d_fg, VISUAL_ANGLE_DEG)
        self.draw_image_corner_brackets(self.canvas2, fg_marker_size, fg_marker_size, 0, 0, color=WIN2_MARKER_COLOR, flip_x=True)
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
        for key, binding_id in self.key_bindings.items():
            self.root.unbind(key, binding_id)
        self.key_bindings.clear()
        
        score = self.evaluation_val.get()
        f1 = os.path.basename(self.current_img_path_1)
        f2 = os.path.basename(self.current_img_path_2)
        
        self.results.append([
            self.participant_id.get(), self.participant_age.get(), self.participant_gender.get(), self.participant_ipd.get(),
            self.distance1.get(), self.distance2.get(),
            self.offset_x.get(), self.offset_y.get(),
            self.current_trial_index + 1,
            f1, f2, score
        ])
        
        self.eval_frame.destroy()
        self.current_trial_index += 1
        
        # 少し待ってから次の試行を開始 (UIの応答性を保つため)
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
                "ID", "Age", "Gender", "IPD(mm)", "Distance1(cm)", "Distance2(cm)",
                "Offset_X", "Offset_Y", "Trial_ID", "Image_Win1", "Image_Win2", "Score"
            ]
            writer.writerow(header)
            writer.writerows(self.results)
            
        messagebox.showinfo("Finished", f"Experiment finished.\nData saved to: {filename}")
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ExperimentApp(root)
    root.mainloop()