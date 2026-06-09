"""
実験の試行ループ共通処理を提供するクラス

全実験（matching、gabor、image）で共通の機能：
- フェーズ管理（ISI、提示、終了）
- タイミング制御
- 結果保存
"""

import tkinter as tk
import os
import csv
import datetime


class ExperimentTrialLoop:
    """
    試行実行の共通ロジックを提供するクラス
    """
    
    def __init__(self):
        """試行ループを初期化"""
        self.trial_list = []
        self.current_trial_in_experiment = 0
        self.current_trial_in_block = 0
        self.results = []
        self.result_dir = None
        self.canvas1 = None
        self.canvas2 = None
        self.root = None
    
    def setup_trial_phases(self, time_phase1=1600, time_isi=1, time_phase2=5000):
        """
        試行フェーズのタイミングを設定
        
        Args:
            time_phase1: フェーズ1の提示時間 (ms)
            time_isi: ISIの時間 (ms)
            time_phase2: フェーズ2の提示時間 (ms)
        """
        self.time_phase1 = time_phase1
        self.time_isi = time_isi
        self.time_phase2 = time_phase2
    
    def phase_isi(self, callback=None):
        """
        ISI (Inter Stimulus Interval) フェーズ
        黒い画面を表示し、指定時間後に次のフェーズへ遷移
        
        Args:
            callback: ISI後に実行するコールバック関数
        """
        if self.canvas1:
            self.canvas1.delete("all")
        if self.canvas2:
            self.canvas2.delete("all")
        
        if callback:
            self.root.after(self.time_isi, callback)
    
    def phase_end_trial(self):
        """
        試行終了時の処理
        Canvas をクリアし、次の試行または終了に進む
        """
        if self.canvas1:
            self.canvas1.delete("all")
        if self.canvas2:
            self.canvas2.delete("all")
        
        self.current_trial_in_block += 1
        self.current_trial_in_experiment += 1
    
    def finish_experiment(self):
        """
        実験終了処理
        結果ファイルを保存し、完了メッセージを表示
        """
        # 結果ファイルを保存
        if self.results and self.result_dir:
            self._save_trial_results()
        
        # 完了ダイアログを表示
        from tkinter import messagebox
        messagebox.showinfo("Experiment Complete", "Thank you for your participation!")
        
        # ウィンドウを閉じる
        if self.root:
            self.root.quit()
    
    def _save_trial_results(self):
        """試行結果をCSVに保存"""
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        
        filename = self._generate_result_filename()
        filepath = os.path.join(self.result_dir, filename)
        
        if not self.results:
            return
        
        fieldnames = list(self.results[0].keys())
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
        
        print(f"Results saved to {filepath}")
    
    def _generate_result_filename(self, prefix="result"):
        """
        結果ファイルのファイル名を生成
        
        Args:
            prefix: ファイル名の接頭辞
            
        Returns:
            str: ファイル名 (例: result_20260518_143022.csv)
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}.csv"
    
    def add_trial_result(self, trial_result):
        """
        試行結果をリストに追加
        
        Args:
            trial_result: 試行結果の辞書
        """
        self.results.append(trial_result)
