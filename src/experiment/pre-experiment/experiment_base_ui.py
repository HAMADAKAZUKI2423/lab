"""
実験アプリケーション共通のUI処理を提供するベースクラス

全実験（matching、gabor、image）で共通の機能を実装：
- 参加者情報の入力・管理
- キーバインドの統一管理
- キャリブレーション画面の構築
- 評価UI（Likert scale）
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os
import csv


class ExperimentBaseUI:
    """
    全実験共通のUI操作を提供するベースクラス
    """
    
    def __init__(self, root, participant_data_dir):
        """
        基盤UIを初期化
        
        Args:
            root: Tkinterのルートウィンドウ
            participant_data_dir: 参加者データディレクトリ
        """
        self.root = root
        self.participant_data_dir = participant_data_dir
        
        # UI用変数
        self.participant_id = tk.StringVar()
        self.participant_age = tk.StringVar()
        self.participant_gender = tk.StringVar()
        self.participant_ipd = tk.StringVar()
        self.offset_x = tk.IntVar(value=0)
        self.offset_y = tk.IntVar(value=0)
        self.evaluation_val = tk.IntVar(value=3)
        
        # キーバインド管理
        self.key_bindings = {}
        
        # UI要素への参照
        self.participant_frame = None
        self.ctrl_frame = None
        self.instruction_frame = None
    
    def setup_participant_info_ui(self):
        """参加者情報入力UIを構築・表示"""
        if hasattr(self, 'participant_frame') and self.participant_frame and self.participant_frame.winfo_exists():
            self.participant_frame.destroy()
        
        self.participant_frame = tk.Frame(self.root, bg='gray', padx=20, pady=20)
        self.participant_frame.place(relx=0.5, rely=0.5, anchor='center')
        
        tk.Label(self.participant_frame, text="Enter Participant ID", 
                font=("Arial", 16)).grid(row=0, column=0, columnspan=2, pady=10)
        
        tk.Label(self.participant_frame, text="Participant ID:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        entry_id = tk.Entry(self.participant_frame, textvariable=self.participant_id)
        entry_id.grid(row=1, column=1, padx=5, pady=5)
        entry_id.focus_set()
        
        btn = tk.Button(self.participant_frame, text="Next", command=self.check_participant_id)
        btn.grid(row=2, column=0, columnspan=2, pady=20)
        btn.bind('<Return>', lambda event: self.check_participant_id())
        entry_id.bind('<Return>', lambda event: self.check_participant_id())
    
    def check_participant_id(self, event=None):
        """参加者IDから参加者データをロード、または新規登録へ"""
        pid = self.participant_id.get().strip()
        if not pid:
            messagebox.showwarning("Input Error", "Please enter a Participant ID.")
            return
        
        filepath = os.path.join(self.participant_data_dir, "participants.csv")
        found = False
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["ID"] == pid:
                        self.participant_age.set(row.get("Age", ""))
                        self.participant_gender.set(row.get("Gender", ""))
                        self.participant_ipd.set(row.get("IPD", ""))
                        found = True
                        break
        
        if self.participant_frame and self.participant_frame.winfo_exists():
            self.participant_frame.destroy()
        
        if found:
            self.on_participant_confirmed()
        else:
            self.setup_new_participant_ui()
    
    def setup_new_participant_ui(self):
        """新規参加者登録UIを構築"""
        self.participant_frame = tk.Frame(self.root, bg='gray', padx=20, pady=20)
        self.participant_frame.place(relx=0.5, rely=0.5, anchor='center')
        
        tk.Label(self.participant_frame, 
                text=f"New Participant Registration (ID: {self.participant_id.get()})", 
                font=("Arial", 16)).grid(row=0, column=0, columnspan=2, pady=10)
        
        tk.Label(self.participant_frame, text="Age:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        entry_age = tk.Entry(self.participant_frame, textvariable=self.participant_age)
        entry_age.grid(row=1, column=1, padx=5, pady=5)
        entry_age.focus_set()
        
        tk.Label(self.participant_frame, text="Gender:").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        gender_combo = ttk.Combobox(self.participant_frame, textvariable=self.participant_gender, 
                                   values=["Male", "Female", "Other"])
        gender_combo.grid(row=2, column=1, padx=5, pady=5)
        gender_combo.set("Male")
        
        tk.Label(self.participant_frame, text="IPD (mm):").grid(row=3, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(self.participant_frame, textvariable=self.participant_ipd).grid(row=3, column=1, padx=5, pady=5)
        
        btn = tk.Button(self.participant_frame, text="Register and Next", 
                       command=self.register_and_start)
        btn.grid(row=4, column=0, columnspan=2, pady=20)
        btn.bind('<Return>', lambda event: self.register_and_start())
    
    def register_and_start(self, event=None):
        """新規参加者を登録して開始"""
        if not self.participant_age.get() or not self.participant_ipd.get():
            messagebox.showwarning("Input Error", "Please enter Age and IPD.")
            return
        self.save_participant_data()
        if self.participant_frame and self.participant_frame.winfo_exists():
            self.participant_frame.destroy()
        self.on_participant_confirmed()
    
    def save_participant_data(self):
        """参加者データをCSVに保存"""
        if not os.path.exists(self.participant_data_dir):
            os.makedirs(self.participant_data_dir)
        
        filepath = os.path.join(self.participant_data_dir, "participants.csv")
        fieldnames = ["ID", "Age", "Gender", "IPD"]
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
                        found = True
                    rows.append(r)
        
        if not found:
            rows.append({
                "ID": self.participant_id.get(),
                "Age": self.participant_age.get(),
                "Gender": self.participant_gender.get(),
                "IPD": self.participant_ipd.get()
            })
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    
    def on_participant_confirmed(self):
        """参加者確認後の処理（継承先で実装）"""
        raise NotImplementedError("Subclasses must implement on_participant_confirmed()")
    
    def setup_calibration_ui(self, is_break=False, is_new_block=False):
        """キャリブレーション画面を構築"""
        if hasattr(self, 'ctrl_frame') and self.ctrl_frame and self.ctrl_frame.winfo_exists():
            self.ctrl_frame.destroy()
        
        self.clear_key_bindings()
        
        self.ctrl_frame = tk.Frame(self.root, bg='gray')
        self.ctrl_frame.place(relx=0.5, rely=0.8, anchor='center')
        
        # ボタンテキストの決定
        if is_break:
            instruction = "Break - Adjust if needed.\nPress 'Enter' to continue."
            button_text = "Resume"
        elif is_new_block:
            instruction = "New Block - Adjust the offset to align the crosshairs.\nUse arrow keys: ← → ↑ ↓\nPress 'Enter' to confirm."
            button_text = "Calibration Complete"
        else:
            instruction = "Calibration - Adjust the offset to align the crosshairs.\nUse arrow keys: ← → ↑ ↓\nPress 'Enter' to confirm."
            button_text = "Next"
        
        tk.Label(self.ctrl_frame, text=instruction, bg='gray', fg='white', 
                font=("Arial", 12)).pack(pady=10, padx=20)
        
        btn = tk.Button(self.ctrl_frame, text=button_text, 
                       command=self.on_calibration_complete)
        btn.pack(pady=10)
        btn.focus_set()
        
        self.key_bindings['<Return>'] = self.root.bind('<Return>', 
                                                       lambda e: self.on_calibration_complete())
        self.key_bindings['<Left>'] = self.root.bind('<Left>', 
                                                     lambda e: self.adjust_offset(-1, 0))
        self.key_bindings['<Right>'] = self.root.bind('<Right>', 
                                                      lambda e: self.adjust_offset(1, 0))
        self.key_bindings['<Up>'] = self.root.bind('<Up>', 
                                                   lambda e: self.adjust_offset(0, -1))
        self.key_bindings['<Down>'] = self.root.bind('<Down>', 
                                                     lambda e: self.adjust_offset(0, 1))
        
        self.update_calibration_view()
    
    def adjust_offset(self, dx, dy):
        """画像表示のオフセットを調整"""
        self.offset_x.set(self.offset_x.get() + dx)
        self.offset_y.set(self.offset_y.get() + dy)
        self.update_calibration_view()
    
    def update_calibration_view(self):
        """キャリブレーション画面を更新（継承先で実装）"""
        raise NotImplementedError("Subclasses must implement update_calibration_view()")
    
    def on_calibration_complete(self):
        """キャリブレーション完了時の処理（継承先で実装）"""
        raise NotImplementedError("Subclasses must implement on_calibration_complete()")
    
    def clear_key_bindings(self):
        """すべてのキーバインドをクリア"""
        for key, binding_id in self.key_bindings.items():
            try:
                self.root.unbind(key, binding_id)
            except:
                pass
        self.key_bindings.clear()
    
    def _reset_to_setup_ui(self):
        """セットアップUIにリセット"""
        if hasattr(self, 'ctrl_frame') and self.ctrl_frame and self.ctrl_frame.winfo_exists():
            self.ctrl_frame.destroy()
        if hasattr(self, 'instruction_frame') and self.instruction_frame and self.instruction_frame.winfo_exists():
            self.instruction_frame.destroy()
        
        self.clear_key_bindings()
        self.setup_participant_info_ui()
    
    def show_evaluation_ui(self, num_choices=5, callback=None):
        """評価UI（Likert scale）を表示"""
        if hasattr(self, 'instruction_frame') and self.instruction_frame and self.instruction_frame.winfo_exists():
            self.instruction_frame.destroy()
        
        self.clear_key_bindings()
        self.evaluation_val.set(num_choices // 2)
        
        self.instruction_frame = tk.Frame(self.root, bg='gray', padx=20, pady=20)
        self.instruction_frame.place(relx=0.5, rely=0.5, anchor='center')
        
        tk.Label(self.instruction_frame, text="Rate the stimulus:", 
                font=("Arial", 14), bg='gray', fg='white').pack(pady=10)
        
        choice_frame = tk.Frame(self.instruction_frame, bg='gray')
        choice_frame.pack(pady=10)
        
        self.eval_buttons = []
        for i in range(1, num_choices + 1):
            btn = tk.Button(choice_frame, text=str(i), width=3, 
                           command=lambda val=i: self._select_evaluation(val, callback))
            btn.pack(side=tk.LEFT, padx=5)
            self.eval_buttons.append(btn)
        
        self._update_eval_highlight()
        
        # キーバインド
        self.key_bindings['<Left>'] = self.root.bind('<Left>', 
                                                     lambda e: self._move_selection(-1))
        self.key_bindings['<Right>'] = self.root.bind('<Right>', 
                                                      lambda e: self._move_selection(1))
        self.key_bindings['<Return>'] = self.root.bind('<Return>', 
                                                       lambda e: self._select_evaluation(
                                                           self.evaluation_val.get(), callback))
    
    def _update_eval_highlight(self):
        """評価UIのハイライトを更新"""
        for i, btn in enumerate(self.eval_buttons):
            if i + 1 == self.evaluation_val.get():
                btn.config(bg='yellow', fg='black')
            else:
                btn.config(bg='lightgray', fg='black')
    
    def _move_selection(self, direction):
        """評価の選択肢をナビゲート"""
        current = self.evaluation_val.get()
        new_val = current + direction
        new_val = max(1, min(new_val, len(self.eval_buttons)))
        self.evaluation_val.set(new_val)
        self._update_eval_highlight()
    
    def _select_evaluation(self, value, callback=None):
        """評価値を確定"""
        self.evaluation_val.set(value)
        if hasattr(self, 'instruction_frame') and self.instruction_frame and self.instruction_frame.winfo_exists():
            self.instruction_frame.destroy()
        self.clear_key_bindings()
        if callback:
            callback(value)
    
    def save_and_next(self, callback=None):
        """評価結果を保存して次へ進む"""
        if callback:
            callback(self.evaluation_val.get())
