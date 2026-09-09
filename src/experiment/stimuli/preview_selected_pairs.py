"""選定ペアをpre_experiment.imageのUI・画像前処理で確認する。
配置: src/experiment/stimuli/preview_selected_pairs.py
起動: cd src; python -m experiment.stimuli.preview_selected_pairs
既存実験のソース・画像フォルダ・参加者CSVは変更しない。
位置合わせ後、前景・背景を最初から同時表示し、Enterで評価画面へ進む。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk

from experiment import experiment_config
from experiment.pre_experiment.image.app import ImageExperimentApp
from experiment.pre_experiment.image.config import create_image_config
from experiment.pre_experiment.image.stimuli import ImageTrial, prepare_trial_stimulus
from experiment.pre_experiment.image.evaluation import show_evaluation_ui

LAB_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = LAB_ROOT / "results" / "fukiage-disparity-selection"
PREVIEW_ROOT = LAB_ROOT / "results" / "tables" / "preview-selected-pairs"


def load_selected_pairs(directory: Path):
    with (directory / "top20.csv").open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"rank", "foreground_path", "background_path"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV必須列: {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError("top20.csvにペアがありません")
    rows.sort(key=lambda r: int(r["rank"]))
    ranks = [int(r["rank"]) for r in rows]
    if min(ranks) < 1 or len(set(ranks)) != len(ranks):
        raise ValueError("rankは重複のない正整数にしてください")
    trials = []
    for row in rows:
        paths = {}
        for role in ("foreground", "background"):
            # 保存された元画像を優先。加工済み・合成済み画像は読み込まない。
            saved = sorted((directory / "top20" / f"rank_{int(row['rank']):02d}").glob(f"{role}_original.*"))
            if len(saved) > 1:
                raise ValueError(f"元画像が複数あります: {saved}")
            path = saved[0] if saved else Path(row[f"{role}_path"])
            if not path.is_absolute():
                path = directory / path
            if not path.is_file():
                raise FileNotFoundError(path)
            paths[role] = path.resolve()
        trials.append(
            ImageTrial(
                background_path=paths["background"],
                foreground_path=paths["foreground"],
                condition="Dual plane",
            )
        )
    return rows, trials


def make_config():
    runtime = experiment_config.get_config()
    fg, bg, angle = (float(runtime[k]) for k in ("DISTANCE_FG", "DISTANCE_BG", "VISUAL_ANGLE_DEG"))
    if not all(math.isfinite(x) and x > 0 for x in (fg, bg, angle)) or fg >= bg:
        raise ValueError("experiment_configの距離・視角が不正です")
    return replace(create_image_config(), distance_fg_cm=fg, distance_bg_cm=bg,
                   visual_angle_deg=angle, background_color=str(runtime.get("BG_COLOR", "black")))


class SelectedPairPreview(ImageExperimentApp):
    def __init__(self, root, directory, config, rows, trials):
        self.selected_directory = directory
        self.selected_rows, self.selected_trials = rows, trials
        self.pending_after = None
        self.preview_state = "starting"
        self.rating_rows = []
        self.preview_dir = PREVIEW_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # 親は最後にsetup_participant_info_uiを呼ぶ。それだけを下で差し替える。
        super().__init__(root, config)
        self.root.title("Selected pair preview - Window 2 (FG)")
        self.win1.title("Selected pair preview - Window 1 (BG)")
        self.root.protocol("WM_DELETE_WINDOW", self.close_preview)
        self.win1.protocol("WM_DELETE_WINDOW", self.close_preview)

    def setup_participant_info_ui(self):
        """参加者登録・デフォーカス測定を開始せず、右眼位置合わせへ進む。"""
        self.trial_list = list(self.selected_trials)
        self.current_trial_index = 0
        self.open_alignment()

    def bind_preview(self, key, callback):
        self.key_bindings[key] = self.root.bind(key, lambda event: callback())

    def cancel_pending(self):
        if self.pending_after is not None:
            self.root.after_cancel(self.pending_after)
            self.pending_after = None

    def schedule(self, ms, callback):
        self.cancel_pending()
        def fire():
            self.pending_after = None
            callback()
        self.pending_after = self.root.after(ms, fire)

    def clear_preview(self):
        self.cancel_pending()
        self.clear_key_bindings()
        for name in ("ctrl_frame", "eval_frame", "participant_frame"):
            self._destroy_frame(name)
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        self.bind_preview("<Escape>", self.close_preview)

    def open_alignment(self):
        self.clear_preview()
        self.preview_state = "alignment"
        self.root.update_idletasks()
        self.width, self.height = self.win1.winfo_width(), self.win1.winfo_height()
        # 既存の赤い二重BG枠・白いFG枠と注視マーカーをそのまま使用。
        self.update_calibration_view()
        self.ctrl_frame = tk.Frame(self.root, bg="gray", padx=15, pady=10)
        self.ctrl_frame.place(relx=0.5, rely=0.8, anchor="center")
        tk.Label(self.ctrl_frame, bg="gray", fg="white", text=
                 "右眼だけで位置合わせ（左眼を覆う）\n矢印: 赤い背景枠を移動 / Enter: ペアを提示\n"
                 "前景・背景を同時表示 / Enterまで維持 / Enter後に評価").pack()
        tk.Button(self.ctrl_frame, text="提示開始", command=self.run_trial).pack(pady=8)
        for key, delta in (("Left",(-1,0)),("Right",(1,0)),("Up",(0,-1)),("Down",(0,1))):
            self.bind_preview(f"<{key}>", lambda d=delta: self.adjust_offset(*d))
        self.bind_preview("<Return>", self.run_trial)
        self.root.focus_set()

    def run_trial(self):
        self.clear_preview()
        self.preview_state = "presenting"
        self.current_trial = self.trial_list[self.current_trial_index]
        try:
            prepared = prepare_trial_stimulus( self.current_trial, self.session_config, calibration=self.defocus_display_calibration, left_pupil_mm=2.2, right_pupil_mm=2.2, ipd_mm=60.0, )
            self.photo_background = ImageTk.PhotoImage(prepared.background)
            self.photo_foreground = ImageTk.PhotoImage(prepared.foreground)
        except Exception as exc:
            messagebox.showerror("画像の準備に失敗", str(exc), parent=self.root)
            self.open_alignment()
            return
        self.width, self.height = self.win1.winfo_width(), self.win1.winfo_height()
        self.phase_both()
        self.root.focus_set()

    def draw_fg(self):
        self.canvas2.create_image(self.canvas2.winfo_width()//2, self.canvas2.winfo_height()//2,
                                  image=self.photo_foreground, anchor="center", tags="img")

    def phase_both(self):
        """前景・背景を同時に描画し、時間制限なしで維持する。"""
        self.cancel_pending()
        self.canvas1.delete("img")
        self.canvas2.delete("img")
        self.canvas2.delete("calib")
        self.canvas1.create_image(self.width//2+self.offset_x.get(), self.height//2+self.offset_y.get(),
                                  image=self.photo_background, anchor="center", tags="img")
        self.draw_fg()
        self.bind_preview("<Return>", self.phase_end_trial)

    def phase_end_trial(self):
        if self.preview_state != "presenting":
            return
        self.clear_preview()
        self.preview_state = "review"
        show_evaluation_ui(self, self.save_and_next)
        rank = self.selected_rows[self.current_trial_index]["rank"]
        tk.Label(self.eval_frame, bg="white", text=
                 f"選定rank {rank} / {self.current_trial_index+1} of {len(self.trial_list)}\n"
                 "R / Space: 再提示、P: 前へ、N: 評価せず次へ、C: 位置合わせ").pack(pady=8)
        bar = tk.Frame(self.eval_frame, bg="white")
        bar.pack()
        for text, command in (("再提示", self.run_trial), ("前へ", lambda: self.move(-1)),
                              ("評価せず次へ", lambda: self.move(1)), ("位置合わせ", self.open_alignment),
                              ("終了", self.close_preview)):
            tk.Button(bar, text=text, command=command).pack(side="left", padx=3)
        for key, command in (("<r>",self.run_trial),("<space>",self.run_trial),
                             ("<p>",lambda:self.move(-1)),("<n>",lambda:self.move(1)),
                             ("<c>",self.open_alignment)):
            self.bind_preview(key, command)

    def move(self, delta):
        self.current_trial_index = (self.current_trial_index + delta) % len(self.trial_list)
        self.run_trial()

    def save_and_next(self):
        if self.preview_state != "review":
            return
        row = self.selected_rows[self.current_trial_index]
        record = {
            "timestamp": datetime.now().isoformat(), "rank": row["rank"],
            "foreground_path": str(self.current_trial.foreground_path),
            "background_path": str(self.current_trial.background_path),
            "score": int(self.evaluation_val.get()),
            "offset_x_right": self.offset_x.get(), "offset_y_right": self.offset_y.get(),
            "distance_fg_cm": self.distance1, "distance_bg_cm": self.distance2,
            "visual_angle_deg": self.session_config.visual_angle_deg,
            "source_result_dir": str(self.selected_directory),
        }
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        path = self.preview_dir / "preview_ratings.csv"
        try:
            first = not path.exists()
            with path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(record))
                if first:
                    writer.writeheader()
                writer.writerow(record)
        except OSError as exc:
            messagebox.showerror("評価を保存できません", str(exc), parent=self.root)
            return
        self.rating_rows.append(record)
        self.move(1)

    def close_preview(self):
        self.cancel_pending()
        self.clear_key_bindings()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", nargs="?", type=Path, help="top20.csvのある結果フォルダ")
    args = parser.parse_args()
    root = tk.Tk()
    root.withdraw()
    try:
        directory = args.result_dir
        if directory is None:
            chosen = filedialog.askdirectory(parent=root, title="top20.csvのある結果フォルダ", initialdir=str(RESULT_ROOT))
            if not chosen:
                return
            directory = Path(chosen)
        directory = directory.expanduser().resolve()
        rows, trials = load_selected_pairs(directory)
        config = make_config()
        config_path = directory / "config.json"
        if config_path.is_file():
            previous = json.loads(config_path.read_text(encoding="utf-8"))
            changes = []
            for key, current in (("distance_fg_cm", config.distance_fg_cm),
                                 ("distance_bg_cm", config.distance_bg_cm),
                                 ("visual_angle_deg", config.visual_angle_deg)):
                if key in previous and not math.isclose(float(previous[key]), current):
                    changes.append(f"{key}: 選定時={previous[key]}, 現在={current}")
            if changes and not messagebox.askyesno("選定時と現在の設定が異なります", "\n".join(changes)+"\n現在の設定で続けますか？", parent=root):
                return
        root.deiconify()
        app = SelectedPairPreview(root, directory, config, rows, trials)
        print(f"評価保存先: {app.preview_dir}")
        root.mainloop()
    except Exception as exc:
        messagebox.showerror("Preview起動エラー", str(exc), parent=root)
        raise
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass


if __name__ == "__main__":
    main()
