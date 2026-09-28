"""20画像ペア×5条件（計100試行）のImage evaluationアプリ。"""

from datetime import datetime
from pathlib import Path
import random
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import ImageTk

from experiment.common import geometry, markers
from experiment.common.defocus_controller import setup_defocus_matching_ui
from ..experiment_base_ui import ExperimentBaseUI

from .calibration import (
    apply_trial_calibration,
    initialize_defocus_calibration,
)
from .conditions import get_condition, resolve_observed_eye
from .config import ImageSessionConfig
from .evaluation import show_evaluation_ui
from .results import (
    build_result_row,
    load_participant,
    save_participant,
    save_session_results,
    save_trial_manifest,
)
from .selected_pairs import (
    load_selected_pairs,
)
from .stimuli import (
    build_trials,
    condition_order_for_participant,
    participant_seed,
    prepare_trial_stimulus,
)


WIN1_MARKER_COLOR = "red"
WIN2_MARKER_COLOR = "white"


class ImageExperimentApp(ExperimentBaseUI):
    """high/low各10ペアを暫定5条件で反復評価する。"""

    def __init__(self, root: tk.Tk, session_config: ImageSessionConfig):
        super().__init__(root)
        self.root = root
        self.session_config = session_config
        self.session_type = "image"
        self.root.title("Image Evaluation - Controller (Window 2)")
        self.root.configure(bg=session_config.background_color)

        self.selected_pairs_dir = self._resolve_selected_pairs_dir()
        self.selected_pairs = load_selected_pairs(self.selected_pairs_dir)
        self.pupil_diameter_val = tk.DoubleVar(
            value=session_config.initial_pupil_diameter_mm
        )
        self.distance1 = session_config.distance_fg_cm
        self.distance2 = session_config.distance_bg_cm
        self.calibration_eyes = ["Right", "Left"]
        self.current_calib_eye_idx = 0
        self.current_alignment_eye = "Right"
        self.calib_results: dict[str, dict] = {}
        self.detailed_defocus_results: list[dict] = []
        self.current_pd_mean = 0.0
        self.current_pd_std = 0.0
        self.defocus_display_calibration = initialize_defocus_calibration(
            self, session_config.display_dir
        )

        self.trial_list = []
        self.condition_order: tuple[str, ...] = ()
        self.participant_seed = 0
        self.current_trial_index = 0
        self.active_block_id = 0
        self.results: list[dict] = []
        self.eval_buttons: list[dict] = []
        self.current_trial = None
        self.current_prepared = None
        self.photo_background = None
        self.photo_foreground_only = None
        self.photo_both = None
        self.result_dir: Path = session_config.result_root

        self._setup_windows()
        self.setup_participant_info_ui()

    def _resolve_selected_pairs_dir(self) -> Path:
        configured = self.session_config.selected_pairs_dir
        if configured is not None:
            if not configured.is_dir():
                raise FileNotFoundError(configured)
            return configured.resolve()
        chosen = filedialog.askdirectory(
            parent=self.root,
            title="high10.csvとlow10.csvのある選定結果フォルダ",
            initialdir=str(self.session_config.selected_pairs_root),
        )
        if not chosen:
            raise RuntimeError("Selected-pair directory was not chosen")
        return Path(chosen).expanduser().resolve()

    def _setup_windows(self) -> None:
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.state("zoomed")

        self.win1 = tk.Toplevel(self.root)
        self.win1.title("Image Evaluation - Display (Window 1)")
        self.win1.geometry(f"+{screen_width}+0")
        self.win1.state("zoomed")
        self.root.update_idletasks()

        self.width = screen_width
        self.height = screen_height
        self.canvas2 = tk.Canvas(
            self.root,
            width=screen_width,
            height=screen_height,
            bg=self.session_config.background_color,
            highlightthickness=0,
        )
        self.canvas2.pack(fill="both", expand=True)
        self.canvas1 = tk.Canvas(
            self.win1,
            width=screen_width,
            height=screen_height,
            bg=self.session_config.background_color,
            highlightthickness=0,
        )
        self.canvas1.pack(fill="both", expand=True)

    @property
    def participant_csv_path(self) -> Path:
        return self.session_config.participant_data_dir / "participants.csv"

    # ---------- participant ----------

    def setup_participant_info_ui(self) -> None:
        self._destroy_frame("participant_frame")
        self.participant_frame = tk.Frame(
            self.root, bg="gray", padx=20, pady=20
        )
        self.participant_frame.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(
            self.participant_frame,
            text="Enter Participant ID",
            font=("Arial", 16),
        ).grid(row=0, column=0, columnspan=2, pady=10)
        tk.Label(self.participant_frame, text="Participant ID:").grid(
            row=1, column=0, sticky="w", padx=5, pady=5
        )
        entry = tk.Entry(
            self.participant_frame, textvariable=self.participant_id
        )
        entry.grid(row=1, column=1, padx=5, pady=5)
        entry.bind("<Return>", self.check_participant_id)
        button = tk.Button(
            self.participant_frame,
            text="Next",
            command=self.check_participant_id,
        )
        button.grid(row=2, column=0, columnspan=2, pady=20)
        button.bind("<Return>", self.check_participant_id)
        entry.focus_set()

    def check_participant_id(self, event=None) -> None:
        participant_id = self.participant_id.get().strip()
        if not participant_id:
            messagebox.showwarning(
                "Input Error", "Please enter a Participant ID."
            )
            return
        row = load_participant(self.participant_csv_path, participant_id)
        self._destroy_frame("participant_frame")
        if row is None:
            self.setup_new_participant_ui()
            return
        self.participant_age.set(row.get("Age", ""))
        self.participant_gender.set(row.get("Gender", ""))
        self.participant_ipd.set(row.get("IPD", ""))
        self.participant_dominance.set(row.get("Dominance", "Right"))
        self.start_calibration_sequence()

    def setup_new_participant_ui(self) -> None:
        self.participant_frame = tk.Frame(
            self.root, bg="gray", padx=20, pady=20
        )
        self.participant_frame.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(
            self.participant_frame,
            text=(
                "New Participant Registration "
                f"(ID: {self.participant_id.get()})"
            ),
            font=("Arial", 16),
        ).grid(row=0, column=0, columnspan=2, pady=10)
        tk.Label(self.participant_frame, text="Age:").grid(
            row=1, column=0, sticky="w", padx=5, pady=5
        )
        tk.Entry(
            self.participant_frame, textvariable=self.participant_age
        ).grid(row=1, column=1, padx=5, pady=5)
        tk.Label(self.participant_frame, text="Gender:").grid(
            row=2, column=0, sticky="w", padx=5, pady=5
        )
        gender = ttk.Combobox(
            self.participant_frame,
            textvariable=self.participant_gender,
            values=["Male", "Female", "Other"],
            state="readonly",
        )
        gender.grid(row=2, column=1, padx=5, pady=5)
        self.participant_gender.set("Male")
        tk.Label(self.participant_frame, text="IPD (mm):").grid(
            row=3, column=0, sticky="w", padx=5, pady=5
        )
        tk.Entry(
            self.participant_frame, textvariable=self.participant_ipd
        ).grid(row=3, column=1, padx=5, pady=5)
        tk.Label(self.participant_frame, text="Eye Dominance:").grid(
            row=4, column=0, sticky="w", padx=5, pady=5
        )
        dominance = ttk.Combobox(
            self.participant_frame,
            textvariable=self.participant_dominance,
            values=["Right", "Left"],
            state="readonly",
        )
        dominance.grid(row=4, column=1, padx=5, pady=5)
        self.participant_dominance.set("Right")
        tk.Button(
            self.participant_frame,
            text="Register and Next",
            command=self.register_and_start,
        ).grid(row=5, column=0, columnspan=2, pady=20)

    def register_and_start(self, event=None) -> None:
        try:
            age = int(self.participant_age.get())
            ipd = float(self.participant_ipd.get())
            if age <= 0 or ipd <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning(
                "Input Error", "Please enter valid positive Age and IPD values."
            )
            return
        save_participant(
            self.participant_csv_path,
            {
                "ID": self.participant_id.get(),
                "Age": self.participant_age.get(),
                "Gender": self.participant_gender.get(),
                "IPD": self.participant_ipd.get(),
                "Dominance": self.participant_dominance.get(),
            },
        )
        self._destroy_frame("participant_frame")
        self.start_calibration_sequence()

    # ---------- calibration / defocus ----------

    def start_calibration_sequence(self) -> None:
        self.win1.update_idletasks()
        self.width = self.win1.winfo_width()
        self.height = self.win1.winfo_height()
        participant_id = self.participant_id.get().strip() or "participant"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = (
            self.session_config.result_root
            / f"{participant_id}_{timestamp}"
        )
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.current_calib_eye_idx = 0
        self.calib_results = {}
        self.detailed_defocus_results = []
        self.current_trial_index = 0
        self.active_block_id = 0
        self.results = []
        self.start_eye_calibration()

    def start_eye_calibration(self) -> None:
        self._destroy_frame("ctrl_frame")
        self.clear_key_bindings()
        if self.current_calib_eye_idx >= len(self.calibration_eyes):
            self.show_experiment_start_ui()
            return
        eye = self.calibration_eyes[self.current_calib_eye_idx]
        messagebox.showinfo(
            "Calibration",
            f"Next: Calibration for {eye} Eye.\nPlease cover the other eye.",
        )
        self.offset_x.set(0)
        self.offset_y.set(0)
        self.pupil_diameter_val.set(
            self.session_config.initial_pupil_diameter_mm
        )
        self.setup_calibration_ui()

    def update_calibration_view(self, *args) -> None:
        self.canvas1.delete("calib")
        self.canvas2.delete("calib")
        foreground = geometry.get_size_for_visual_angle(
            self.distance1, self.session_config.visual_angle_deg
        )
        background_height = geometry.get_size_for_visual_angle(
            self.distance2, self.session_config.visual_angle_deg
        )
        background_width = geometry.get_size_for_visual_angle(
            self.distance2, self.session_config.visual_angle_deg * 2.0
        )
        for width in (background_width, background_height):
            markers.draw_image_corner_brackets(
                self.canvas1,
                width,
                background_height,
                self.offset_x.get(),
                self.offset_y.get(),
                color=WIN1_MARKER_COLOR,
                line_width=markers.MARKER_LINE_WIDTH * 1.5,
            )
        markers.draw_image_corner_brackets(
            self.canvas2,
            foreground,
            foreground,
            color=WIN2_MARKER_COLOR,
        )
        markers.draw_center_cross(
            self.canvas2, color=WIN2_MARKER_COLOR
        )

    def setup_calibration_ui(self) -> None:
        self._destroy_frame("ctrl_frame")
        self.clear_key_bindings()
        self.update_calibration_view()
        eye = self.calibration_eyes[self.current_calib_eye_idx]
        self.ctrl_frame = tk.Frame(self.root, bg="gray")
        self.ctrl_frame.place(relx=0.5, rely=0.8, anchor="center")
        tk.Label(
            self.ctrl_frame,
            text=f"[{eye} eye] Use the arrow keys to align the red frame.",
            bg="gray",
            fg="white",
            font=("Arial", 12),
        ).pack(pady=10, padx=20)
        tk.Button(
            self.ctrl_frame,
            text="Calibration Done, Next",
            command=self.start_eye_defocus_matching,
        ).pack(pady=10)
        self.key_bindings["<Return>"] = self.root.bind(
            "<Return>", lambda event: self.start_eye_defocus_matching()
        )
        self.key_bindings["<Left>"] = self.root.bind(
            "<Left>", lambda event: self.adjust_offset(-1, 0)
        )
        self.key_bindings["<Right>"] = self.root.bind(
            "<Right>", lambda event: self.adjust_offset(1, 0)
        )
        self.key_bindings["<Up>"] = self.root.bind(
            "<Up>", lambda event: self.adjust_offset(0, -1)
        )
        self.key_bindings["<Down>"] = self.root.bind(
            "<Down>", lambda event: self.adjust_offset(0, 1)
        )
        self.root.focus_set()

    def start_eye_defocus_matching(self) -> None:
        self._destroy_frame("ctrl_frame")
        self.clear_key_bindings()
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        setup_defocus_matching_ui(
            self,
            cpd=self.session_config.defocus_cpd,
            repetitions=self.session_config.defocus_repetitions,
        )

    # ---------- block / trial ----------

    def show_experiment_start_ui(self) -> None:
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        self.clear_key_bindings()
        self.ctrl_frame = tk.Frame(
            self.root, bg="gray", padx=30, pady=30
        )
        self.ctrl_frame.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(
            self.ctrl_frame,
            text=(
                "The 5-condition image experiment will now begin.\n"
                "100 trials: 20 selected pairs in every condition."
            ),
            bg="gray",
            fg="white",
            font=("Arial", 16),
        ).pack(pady=15)
        tk.Button(
            self.ctrl_frame,
            text="Start Image Experiment",
            command=self.begin_experiment,
        ).pack(pady=10)
        self.key_bindings["<Return>"] = self.root.bind(
            "<Return>", self.begin_experiment
        )

    def begin_experiment(self, event=None) -> None:
        self._destroy_frame("ctrl_frame")
        self.clear_key_bindings()
        participant_id = self.participant_id.get().strip()
        self.participant_seed = participant_seed(
            self.session_config.random_seed, participant_id
        )
        self.condition_order = condition_order_for_participant(
            self.session_config.conditions,
            participant_id,
            self.session_config.random_seed,
        )
        rng = random.Random(self.participant_seed)
        self.trial_list = build_trials(
            self.selected_pairs, self.condition_order, rng
        )
        save_trial_manifest(
            self.result_dir,
            self.trial_list,
            config=self.session_config,
            dominant_eye=self.participant_dominance.get(),
            selection_dir=self.selected_pairs_dir,
            participant_seed=self.participant_seed,
        )
        print(f"Selected pairs: {self.selected_pairs_dir}")
        print(f"Condition order: {self.condition_order}")
        print(f"Participant seed: {self.participant_seed}")
        print(f"Total trials: {len(self.trial_list)}")
        self.current_trial_index = 0
        self.active_block_id = 0
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        self.run_trial()

    def run_trial(self) -> None:
        if self.current_trial_index >= len(self.trial_list):
            self.finish_experiment()
            return
        trial = self.trial_list[self.current_trial_index]
        if trial.block_id != self.active_block_id:
            self.show_block_confirmation(trial)
            return

        self.current_trial = trial
        apply_trial_calibration(self, trial.condition_id)
        right = self.calib_results.get("Right", {})
        left = self.calib_results.get("Left", {})
        try:
            self.current_prepared = prepare_trial_stimulus(
                trial,
                self.session_config,
                calibration=self.defocus_display_calibration,
                dominant_eye=self.participant_dominance.get(),
                left_pupil_mm=float(left.get("pd_mean", 0.0)),
                right_pupil_mm=float(right.get("pd_mean", 0.0)),
                ipd_mm=float(self.participant_ipd.get()),
            )
        except Exception as exc:
            messagebox.showerror(
                "Stimulus preparation failed",
                f"Trial {trial.trial_order}: {exc}",
                parent=self.root,
            )
            raise

        prepared = self.current_prepared
        self.photo_background = (
            ImageTk.PhotoImage(prepared.window1_both)
            if prepared.window1_both is not None else None
        )
        self.photo_foreground_only = ImageTk.PhotoImage(
            prepared.window2_foreground_only
        )
        self.photo_both = ImageTk.PhotoImage(prepared.window2_both)

        self.canvas1.configure(bg=self.session_config.background_color)
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        self.canvas2.create_image(
            self.canvas2.winfo_width() // 2,
            self.canvas2.winfo_height() // 2,
            image=self.photo_foreground_only,
            anchor="center",
            tags="img",
        )
        self.root.after(
            self.session_config.time_foreground_only_ms, self.phase_isi
        )

    def show_block_confirmation(self, trial) -> None:
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        self._destroy_frame("ctrl_frame")
        self.clear_key_bindings()
        spec = get_condition(trial.condition_id)
        observed_eye = resolve_observed_eye(
            spec, self.participant_dominance.get()
        )
        if observed_eye == "Both":
            eye_instruction = "Use BOTH eyes."
        else:
            other_eye = "Left" if observed_eye == "Right" else "Right"
            eye_instruction = (
                f"Use the {observed_eye.upper()} eye and cover the "
                f"{other_eye.lower()} eye."
            )
        self.ctrl_frame = tk.Frame(
            self.root, bg="gray", padx=30, pady=30
        )
        self.ctrl_frame.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(
            self.ctrl_frame,
            text=(
                f"Block {trial.block_id}/{len(self.condition_order)}\n"
                f"{spec.label}\n\n{eye_instruction}\n"
                "20 trials; a short break follows trial 10."
            ),
            bg="gray",
            fg="white",
            font=("Arial", 16),
        ).pack(pady=15, padx=20)
        tk.Button(
            self.ctrl_frame,
            text="Start Block",
            command=lambda: self.start_block(trial.block_id),
        ).pack(pady=10)
        self.key_bindings["<Return>"] = self.root.bind(
            "<Return>", lambda event: self.start_block(trial.block_id)
        )

    def start_block(self, block_id: int) -> None:
        self._destroy_frame("ctrl_frame")
        self.clear_key_bindings()
        self.active_block_id = block_id
        self.run_trial()

    def phase_isi(self) -> None:
        self.canvas2.delete("img")
        foreground = geometry.get_size_for_visual_angle(
            self.distance1, self.session_config.visual_angle_deg
        )
        markers.draw_image_corner_brackets(
            self.canvas2,
            foreground,
            foreground,
            color=WIN2_MARKER_COLOR,
            flip_x=True,
        )
        markers.draw_center_cross(
            self.canvas2, color=WIN2_MARKER_COLOR
        )
        self.root.after(self.session_config.time_isi_ms, self.phase_both)

    def phase_both(self) -> None:
        self.canvas1.delete("img")
        self.canvas2.delete("calib")
        if self.photo_background is not None:
            self.canvas1.create_image(
                self.width // 2 + self.offset_x.get(),
                self.height // 2 + self.offset_y.get(),
                image=self.photo_background,
                anchor="center",
                tags="img",
            )
        self.canvas2.create_image(
            self.canvas2.winfo_width() // 2,
            self.canvas2.winfo_height() // 2,
            image=self.photo_both,
            anchor="center",
            tags="img",
        )
        self.root.after(
            self.session_config.time_both_ms, self.phase_end_trial
        )

    def phase_end_trial(self) -> None:
        self.canvas1.delete("img")
        self.canvas2.delete("img")
        show_evaluation_ui(self, self.save_and_next)

    def save_and_next(self) -> None:
        self.clear_key_bindings()
        self.results.append(
            build_result_row(self, self.evaluation_val.get())
        )
        self._checkpoint_results()
        self._destroy_frame("eval_frame")
        self.current_trial_index += 1
        if self.current_trial_index >= len(self.trial_list):
            self.root.after(500, self.finish_experiment)
            return
        next_trial = self.trial_list[self.current_trial_index]
        break_due = (
            next_trial.block_id == self.active_block_id
            and next_trial.within_block_order
            == self.session_config.trials_before_break + 1
        )
        if break_due:
            self.root.after(500, self.start_break)
        else:
            self.root.after(500, self.run_trial)

    def start_break(self) -> None:
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        self._destroy_frame("ctrl_frame")
        self.clear_key_bindings()
        self.ctrl_frame = tk.Frame(
            self.root, bg="gray", padx=30, pady=30
        )
        self.ctrl_frame.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(
            self.ctrl_frame,
            text=(
                "Short break: 10/20 trials completed in this block.\n"
                "Press Enter when you are ready to continue."
            ),
            bg="gray",
            fg="white",
            font=("Arial", 16),
        ).pack(pady=15)
        tk.Button(
            self.ctrl_frame,
            text="Resume",
            command=self.resume_experiment,
        ).pack(pady=10)
        self.key_bindings["<Return>"] = self.root.bind(
            "<Return>", lambda event: self.resume_experiment()
        )

    def resume_experiment(self) -> None:
        self._destroy_frame("ctrl_frame")
        self.clear_key_bindings()
        self.run_trial()

    def _checkpoint_results(self) -> Path:
        return save_session_results(
            self.result_dir,
            self.results,
            config=self.session_config,
            selection_dir=self.selected_pairs_dir,
            participant_seed=self.participant_seed,
            condition_order=self.condition_order,
        )

    def finish_experiment(self) -> None:
        output = self._checkpoint_results()
        messagebox.showinfo(
            "Finished",
            f"Experiment finished.\nData saved to: {output}",
        )
        self.root.destroy()