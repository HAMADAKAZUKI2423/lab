"""Image evaluation予備実験のTkinterアプリ。"""

from datetime import datetime
from pathlib import Path
import random
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import ImageTk

from common import geometry, markers
from common.defocus_controller import setup_defocus_matching_ui
from experiment_base_ui import ExperimentBaseUI

from .calibration import (
    apply_dominant_eye_calibration,
    initialize_defocus_compatibility,
)
from .config import ImageSessionConfig
from .evaluation import show_evaluation_ui
from .results import (
    build_result_row,
    load_participant,
    save_participant,
    save_session_results,
)
from .stimuli import build_trials, discover_images, prepare_trial_stimulus


WIN1_MARKER_COLOR = "red"
WIN2_MARKER_COLOR = "white"


class ImageExperimentApp(ExperimentBaseUI):
    """画像の前景・背景組み合わせを5段階で評価する。"""

    def __init__(self, root: tk.Tk, session_config: ImageSessionConfig):
        super().__init__(root)
        self.root = root
        self.session_config = session_config
        self.session_type = "image"
        self.root.title("Image Evaluation - Controller (Window 2)")
        self.root.configure(bg=session_config.background_color)

        self.pupil_diameter_val = tk.DoubleVar(
            value=session_config.initial_pupil_diameter_mm
        )
        self.distance1 = session_config.distance_fg_cm
        self.distance2 = session_config.distance_bg_cm
        self.calibration_eyes = ["Right", "Left"]
        self.current_calib_eye_idx = 0
        self.calib_results: dict[str, dict] = {}
        self.detailed_defocus_results: list[dict] = []
        self.current_pd_mean = 0.0
        self.current_pd_std = 0.0
        initialize_defocus_compatibility(self)

        self.rng = random.Random()
        self.trial_list = []
        self.current_trial_index = 0
        self.results: list[dict] = []
        self.eval_buttons: list[dict] = []
        self.current_trial = None
        self.photo_background = None
        self.photo_foreground = None
        self.result_dir: Path = session_config.result_root

        self._setup_windows()
        self.setup_participant_info_ui()

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
            int(self.participant_age.get())
            float(self.participant_ipd.get())
        except ValueError:
            messagebox.showwarning(
                "Input Error", "Please enter valid Age and IPD values."
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
        self.results = []
        self.start_eye_calibration()

    def start_eye_calibration(self) -> None:
        self._destroy_frame("ctrl_frame")
        self.clear_key_bindings()
        if self.current_calib_eye_idx >= len(self.calibration_eyes):
            apply_dominant_eye_calibration(self)
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
        self.setup_calibration_ui(is_break=False)

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

    def setup_calibration_ui(self, is_break: bool = False) -> None:
        self._destroy_frame("ctrl_frame")
        self.clear_key_bindings()
        self.update_calibration_view()
        self.ctrl_frame = tk.Frame(self.root, bg="gray")
        self.ctrl_frame.place(relx=0.5, rely=0.8, anchor="center")
        if is_break:
            instruction = (
                "This is a break. You can adjust the position if needed.\n"
                "Press Enter to resume."
            )
            button_text = "Resume Experiment"
            command = self.resume_experiment
        else:
            instruction = "Use the arrow keys to adjust the red frame."
            button_text = "Calibration Done, Next"
            command = self.start_eye_defocus_matching
        tk.Label(
            self.ctrl_frame,
            text=instruction,
            bg="gray",
            fg="white",
            font=("Arial", 12),
        ).pack(pady=10, padx=20)
        tk.Button(
            self.ctrl_frame, text=button_text, command=command
        ).pack(pady=10)
        self.key_bindings["<Return>"] = self.root.bind(
            "<Return>", lambda event: command()
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
            patterns=self.session_config.defocus_patterns,
            cpds=self.session_config.defocus_cpds,
        )

    def show_experiment_start_ui(self) -> None:
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        self.ctrl_frame = tk.Frame(
            self.root, bg="gray", padx=30, pady=30
        )
        self.ctrl_frame.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(
            self.ctrl_frame,
            text="The image experiment will now begin.\nPress Enter to start.",
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

    # ---------- trial ----------

    def begin_experiment(self, event=None) -> None:
        self._destroy_frame("ctrl_frame")
        self.clear_key_bindings()
        background_paths = discover_images(
            self.session_config.background_image_dir
        )
        foreground_paths = discover_images(
            self.session_config.foreground_image_dir
        )
        if not background_paths or not foreground_paths:
            messagebox.showerror(
                "Error",
                "Image folder not found or is empty.\n\n"
                f"BG path: {self.session_config.background_image_dir}\n"
                f"FG path: {self.session_config.foreground_image_dir}",
            )
            self._reset_to_setup_ui()
            return
        self.trial_list = build_trials(
            background_paths, foreground_paths, self.rng
        )
        print(
            f"Found {len(background_paths)} background images and "
            f"{len(foreground_paths)} foreground images."
        )
        print(f"Total trials: {len(self.trial_list)}")
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        self.run_trial()

    def run_trial(self) -> None:
        if self.current_trial_index >= len(self.trial_list):
            self.finish_experiment()
            return
        self.current_trial = self.trial_list[self.current_trial_index]
        prepared = prepare_trial_stimulus(
            self.current_trial, self.session_config
        )
        self.photo_background = ImageTk.PhotoImage(prepared.background)
        self.photo_foreground = ImageTk.PhotoImage(prepared.foreground)

        self.canvas1.configure(bg=self.session_config.background_color)
        self.canvas1.delete("all")
        self.canvas2.create_image(
            self.canvas2.winfo_width() // 2,
            self.canvas2.winfo_height() // 2,
            image=self.photo_foreground,
            anchor="center",
            tags="img",
        )
        self.root.after(
            self.session_config.time_foreground_only_ms, self.phase_isi
        )

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
        self.canvas2.delete("calib")
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
            image=self.photo_foreground,
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
        self._destroy_frame("eval_frame")
        self.current_trial_index += 1
        break_due = (
            self.current_trial_index
            % self.session_config.trials_before_break == 0
            and self.current_trial_index < len(self.trial_list)
        )
        if break_due:
            self.root.after(500, self.start_break)
        else:
            self.root.after(500, self.run_trial)

    def start_break(self) -> None:
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        self.setup_calibration_ui(is_break=True)

    def resume_experiment(self) -> None:
        self._destroy_frame("ctrl_frame")
        self.clear_key_bindings()
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        self.run_trial()

    def finish_experiment(self) -> None:
        output = save_session_results(self.result_dir, self.results)
        messagebox.showinfo(
            "Finished",
            f"Experiment finished.\nData saved to: {output}",
        )
        self.root.destroy()

    def _reset_to_setup_ui(self) -> None:
        self._destroy_frame("ctrl_frame")
        self._destroy_frame("eval_frame")
        self.clear_key_bindings()
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        self.setup_participant_info_ui()

