"""本実験とtrainingで共有するTkinterアプリ。"""

from datetime import datetime
from pathlib import Path
import random
import tkinter as tk
from tkinter import messagebox, ttk

import defocus_matching
import stimuli_utils
from experiment_base_ui import ExperimentBaseUI
from experiment_trial_loop import ExperimentTrialLoop

from .calibration import load_display_calibration
from .config import MatchingSessionConfig
from .results import (
    build_result_row,
    load_participant,
    save_participant,
    save_session_results,
)
from .stimuli import (
    DUAL_CONDITIONS,
    build_blocks,
    build_trials,
    contrast_to_slider,
    generate_trial_photos,
    prepare_trial_stimulus,
    slider_to_contrast,
)


WIN1_MARKER_COLOR = "red"
WIN2_MARKER_COLOR = "white"


class MatchingExperimentApp(ExperimentBaseUI, ExperimentTrialLoop):
    """条件だけを差し替えて本実験とtrainingを実行する。"""

    def __init__(self, root: tk.Tk, session_config: MatchingSessionConfig):
        ExperimentBaseUI.__init__(
            self, root, str(session_config.participant_data_dir)
        )
        ExperimentTrialLoop.__init__(self)

        self.root = root
        self.session_config = session_config
        self.session_type = session_config.session_type
        self.config = {
            "L_fg": session_config.l_fg,
            "L_bg": session_config.l_bg,
            "L_ref": session_config.l_ref,
            "DISTANCE_FG": session_config.distance_fg_cm,
            "DISTANCE_BG": session_config.distance_bg_cm,
        }
        self.root.title(
            f"Gabor Matching {self.session_type.title()} - Controller (Window 2)"
        )
        self.root.configure(bg=session_config.background_color)

        self.distance1 = session_config.distance_fg_cm
        self.distance2 = session_config.distance_bg_cm
        self.spatial_freq = session_config.spatial_frequency
        self.L_fg = session_config.l_fg
        self.L_bg = session_config.l_bg
        self.L_ref = session_config.l_ref

        self.participant_dominance = tk.StringVar(value="Right")
        self.pupil_diameter_val = tk.DoubleVar(
            value=session_config.initial_pupil_diameter_mm
        )
        self.calibration_eyes = ["Right", "Left"]
        self.current_calib_eye_idx = 0
        self.calib_results: dict[str, dict] = {}
        self.detailed_defocus_results: list[dict] = []
        self.current_pd_mean = 0.0
        self.current_pd_std = 0.0

        self.blocks: list[dict] = []
        self.current_block_index = 0
        self.current_block_cond: dict | None = None
        self.prepared_stimulus = None
        self.rng = random.Random()

        calibration = load_display_calibration(session_config.display_dir)
        self.display_calibration = calibration
        # defocus_matching.pyが参照する互換属性
        self.color_matrix = calibration.color_matrix
        self.gamma_bg = calibration.gamma_bg
        self.gamma_fg = calibration.gamma_fg
        self.bg_lums = calibration.bg_lums
        self.bg_pixels = calibration.bg_pixels
        self.fg_lums = calibration.bg_lums
        self.fg_pixels = calibration.bg_pixels
        self.ext_lum_Y = calibration.ext_lum_y
        self.ext_lum_px = calibration.ext_lum_px

        self._setup_windows()
        self.result_dir = session_config.result_root
        self.setup_trial_phases(time_phase1=1600, time_isi=1, time_phase2=5000)
        self.setup_participant_info_ui()

    def _setup_windows(self) -> None:
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        self.root.state("zoomed")

        self.win1 = tk.Toplevel(self.root)
        self.win1.title("Gabor Matching - Display (Window 1)")
        self.win1.geometry(f"+{screen_w}+0")
        self.win1.state("zoomed")
        self.root.update_idletasks()

        self.width = screen_w
        self.height = screen_h
        self.canvas2 = tk.Canvas(
            self.root, width=screen_w, height=screen_h,
            bg=self.session_config.background_color, highlightthickness=0,
        )
        self.canvas2.pack(fill="both", expand=True)
        self.canvas1 = tk.Canvas(
            self.win1, width=screen_w, height=screen_h,
            bg=self.session_config.background_color, highlightthickness=0,
        )
        self.canvas1.pack(fill="both", expand=True)

    # ---------- participant ----------

    @property
    def participant_csv_path(self) -> Path:
        return self.session_config.participant_data_dir / "participants.csv"

    def setup_participant_info_ui(self) -> None:
        self._destroy_frame("participant_frame")
        self.participant_frame = tk.Frame(
            self.root, bg="gray", padx=20, pady=20
        )
        self.participant_frame.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(
            self.participant_frame, text="Enter Participant ID", font=("Arial", 16)
        ).grid(row=0, column=0, columnspan=2, pady=10)
        tk.Label(self.participant_frame, text="Participant ID:").grid(
            row=1, column=0, sticky="w", padx=5, pady=5
        )
        entry = tk.Entry(self.participant_frame, textvariable=self.participant_id)
        entry.grid(row=1, column=1, padx=5, pady=5)
        entry.bind("<Return>", self.check_participant_id)

        next_button = tk.Button(
            self.participant_frame, text="Next", command=self.check_participant_id
        )
        next_button.grid(row=2, column=0, columnspan=2, pady=20)
        next_button.bind("<Return>", self.check_participant_id)
        entry.focus_set()

    def check_participant_id(self, event=None) -> None:
        participant_id = self.participant_id.get().strip()
        if not participant_id:
            messagebox.showwarning("Input Error", "Please enter a Participant ID.")
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
        fields = [
            ("Age:", self.participant_age),
            ("IPD (mm):", self.participant_ipd),
        ]
        tk.Label(
            self.participant_frame,
            text=f"New Participant Registration (ID: {self.participant_id.get()})",
            font=("Arial", 16),
        ).grid(row=0, column=0, columnspan=2, pady=10)
        for index, (label, variable) in enumerate(fields, start=1):
            tk.Label(self.participant_frame, text=label).grid(
                row=index, column=0, sticky="w", padx=5, pady=5
            )
            tk.Entry(self.participant_frame, textvariable=variable).grid(
                row=index, column=1, padx=5, pady=5
            )
        tk.Label(self.participant_frame, text="Gender:").grid(
            row=3, column=0, sticky="w", padx=5, pady=5
        )
        ttk.Combobox(
            self.participant_frame,
            textvariable=self.participant_gender,
            values=["Male", "Female", "Other"],
        ).grid(row=3, column=1, padx=5, pady=5)
        self.participant_gender.set("Male")
        tk.Label(self.participant_frame, text="Eye Dominance:").grid(
            row=4, column=0, sticky="w", padx=5, pady=5
        )
        ttk.Combobox(
            self.participant_frame,
            textvariable=self.participant_dominance,
            values=["Right", "Left"],
        ).grid(row=4, column=1, padx=5, pady=5)
        tk.Button(
            self.participant_frame,
            text="Register and Next",
            command=self.register_and_start,
        ).grid(row=5, column=0, columnspan=2, pady=20)

    def register_and_start(self, event=None) -> None:
        if not self.participant_age.get() or not self.participant_ipd.get():
            messagebox.showwarning("Input Error", "Please enter Age and IPD.")
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

    def on_participant_confirmed(self) -> None:
        self.start_calibration_sequence()

    # ---------- calibration / defocus ----------

    def start_calibration_sequence(self) -> None:
        self.win1.update_idletasks()
        self.width = self.win1.winfo_width()
        self.height = self.win1.winfo_height()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        participant_id = self.participant_id.get().strip() or "participant"
        self.result_dir = self.session_config.result_root / f"{participant_id}_{timestamp}"
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.current_calib_eye_idx = 0
        self.calib_results = {}
        self.detailed_defocus_results = []
        self.start_eye_calibration()

    def start_eye_calibration(self) -> None:
        if self.current_calib_eye_idx >= len(self.calibration_eyes):
            self.setup_experiment_blocks()
            return
        eye = self.calibration_eyes[self.current_calib_eye_idx]
        messagebox.showinfo(
            "Calibration", f"Next: Calibration for {eye} Eye.\nPlease cover the other eye."
        )
        self.offset_x.set(0)
        self.offset_y.set(0)
        self.pupil_diameter_val.set(self.session_config.initial_pupil_diameter_mm)
        self.setup_calibration_ui_matching(is_new_eye=True)

    def update_calibration_view(self, *args) -> None:
        self.canvas1.delete("calib")
        self.canvas2.delete("calib")
        fg = stimuli_utils.get_size_for_visual_angle(
            self.distance1, self.session_config.visual_angle_width_deg
        )
        bg_h = stimuli_utils.get_size_for_visual_angle(
            self.distance2, self.session_config.visual_angle_width_deg
        )
        bg_w = stimuli_utils.get_size_for_visual_angle(
            self.distance2, self.session_config.visual_angle_width_deg * 2.0
        )
        for width in (bg_w, bg_h):
            stimuli_utils.draw_image_corner_brackets(
                self.canvas1, width, bg_h,
                self.offset_x.get(), self.offset_y.get(),
                color=WIN1_MARKER_COLOR,
                line_width=stimuli_utils.MARKER_LINE_WIDTH * 1.5,
            )
        stimuli_utils.draw_image_corner_brackets(
            self.canvas2, fg, fg, 0, 0,
            color=WIN2_MARKER_COLOR,
            line_width=stimuli_utils.MARKER_LINE_WIDTH,
        )
        stimuli_utils.draw_center_cross(self.canvas2, color=WIN2_MARKER_COLOR)

    def adjust_offset(self, dx: int, dy: int):
        self.offset_x.set(self.offset_x.get() + dx)
        self.offset_y.set(self.offset_y.get() + dy)
        self.update_calibration_view()
        return "break"

    def setup_calibration_ui_matching(
        self, *, is_new_eye: bool = False, is_new_block: bool = False
    ) -> None:
        self.update_calibration_view()
        self._destroy_frame("ctrl_frame")
        self.clear_key_bindings()
        self.ctrl_frame = tk.Frame(self.root, bg="gray")
        self.ctrl_frame.place(relx=0.5, rely=0.8, anchor="center")
        if is_new_eye:
            eye = self.calibration_eyes[self.current_calib_eye_idx]
            text = f"[{eye} Eye Calibration]\nUse arrow keys to align the red frame."
            command = self._start_defocus_matching
        else:
            text = "Block calibration: align the red frame, then press Enter."
            command = self.start_experiment_block
        tk.Label(
            self.ctrl_frame, text=text, bg="gray", fg="white", font=("Arial", 12)
        ).pack(pady=10, padx=20)
        tk.Button(self.ctrl_frame, text="Calibration Done", command=command).pack(pady=10)
        self.key_bindings["<Return>"] = self.root.bind("<Return>", lambda e: command())
        self.key_bindings["<Left>"] = self.root.bind(
            "<Left>", lambda e: self.adjust_offset(-1, 0)
        )
        self.key_bindings["<Right>"] = self.root.bind(
            "<Right>", lambda e: self.adjust_offset(1, 0)
        )
        self.key_bindings["<Up>"] = self.root.bind(
            "<Up>", lambda e: self.adjust_offset(0, -1)
        )
        self.key_bindings["<Down>"] = self.root.bind(
            "<Down>", lambda e: self.adjust_offset(0, 1)
        )
        self.root.focus_set()

    def _start_defocus_matching(self) -> None:
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        defocus_matching.setup_defocus_matching_ui(
            self,
            patterns=self.session_config.defocus_patterns,
            cpds=self.session_config.defocus_cpds,
        )

    def on_calibration_complete(self) -> None:
        self._start_defocus_matching()

    # ---------- block / trial ----------

    def setup_experiment_blocks(self) -> None:
        dominant_eye = self.participant_dominance.get()
        if dominant_eye not in self.calib_results:
            dominant_eye = "Right"
        calibration = self.calib_results[dominant_eye]
        self.offset_x.set(calibration["offset_x"])
        self.offset_y.set(calibration["offset_y"])
        self.current_pd_mean = calibration["pd_mean"]
        self.blocks = build_blocks(self.session_config, self.rng)
        self.current_block_index = 0
        self.current_trial_in_experiment = 0
        self.start_block()

    def start_block(self) -> None:
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        if self.current_block_index >= len(self.blocks):
            self._save_results_and_finish()
            return
        self.current_block_cond = self.blocks[self.current_block_index]
        self.current_trial_in_block = 0
        self._show_block_confirmation()

    def _show_block_confirmation(self) -> None:
        self._destroy_frame("ctrl_frame")
        self.clear_key_bindings()
        condition = self.current_block_cond["condition"]
        ocularity = self.current_block_cond["ocularity"]
        eye_text = (
            "Use your DOMINANT eye and COVER the other eye."
            if ocularity == "monocular" else "Use BOTH eyes."
        )
        self.ctrl_frame = tk.Frame(self.root, bg="gray")
        self.ctrl_frame.place(relx=0.5, rely=0.5, anchor="center")
        text = (
            f"[Block {self.current_block_index + 1}/{len(self.blocks)}]\n"
            f"Condition: {condition}\nOcularity: {ocularity}\n\n{eye_text}"
        )
        tk.Label(
            self.ctrl_frame, text=text, bg="gray", fg="white", font=("Arial", 16)
        ).pack(pady=20, padx=40)
        tk.Button(
            self.ctrl_frame, text="OK", command=self._start_block_calibration
        ).pack(pady=10)
        self.key_bindings["<Return>"] = self.root.bind(
            "<Return>", lambda e: self._start_block_calibration()
        )

    def _start_block_calibration(self) -> None:
        self._destroy_frame("ctrl_frame")
        self.clear_key_bindings()
        self.setup_calibration_ui_matching(is_new_block=True)

    def start_experiment_block(self) -> None:
        self.clear_key_bindings()
        self._destroy_frame("ctrl_frame")
        self.trial_list = build_trials(self.session_config, self.rng)
        self.current_trial_in_block = 0
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        self.run_trial()

    def run_trial(self) -> None:
        if self.current_trial_in_block >= len(self.trial_list):
            self.current_block_index += 1
            self.root.after(500, self.start_block)
            return
        trial = self.trial_list[self.current_trial_in_block]
        self.init_contrast = self.rng.uniform(0.0, 1.0)
        self.init_slider_val = contrast_to_slider(self.init_contrast)
        self.prepared_stimulus = prepare_trial_stimulus(
            condition=self.current_block_cond["condition"],
            ocularity=self.current_block_cond["ocularity"],
            dominant_eye=self.participant_dominance.get(),
            orientation=trial["orientation"],
            pupil_diameter_mm=self.current_pd_mean,
            config=self.session_config,
        )
        self.setup_contrast_matching_ui()

    def setup_contrast_matching_ui(self) -> None:
        self._destroy_frame("ctrl_frame")
        self.clear_key_bindings()
        self.ctrl_frame = tk.Frame(self.root, bg="gray")
        self.ctrl_frame.place(relx=0.5, rely=0.8, anchor="center")
        self.slider_val = tk.DoubleVar(value=self.init_slider_val)
        tk.Scale(
            self.ctrl_frame, from_=1.0, to=0.0, resolution=0.001,
            orient=tk.HORIZONTAL, length=400, variable=self.slider_val,
            showvalue=0, command=lambda *_: self.update_stimuli(),
        ).pack(pady=10)
        tk.Button(
            self.ctrl_frame, text="Next Trial", command=self.save_and_next
        ).pack(pady=10)
        tk.Label(
            self.ctrl_frame,
            text="Adjust the slider to match the contrast.\nPress Down to confirm.",
            bg="gray", fg="white", font=("Arial", 12),
        ).pack(pady=10, padx=20)
        self.key_bindings["<Down>"] = self.root.bind(
            "<Down>", lambda e: self.save_and_next()
        )
        self.key_bindings["<Left>"] = self.root.bind(
            "<Left>", lambda e: self._move_contrast(-self.session_config.slider_step)
        )
        self.key_bindings["<Right>"] = self.root.bind(
            "<Right>", lambda e: self._move_contrast(self.session_config.slider_step)
        )
        self.update_stimuli()

    def _move_contrast(self, delta: float):
        self.slider_val.set(min(1.0, max(0.0, self.slider_val.get() + delta)))
        self.update_stimuli()
        return "break"

    def update_stimuli(self) -> None:
        test_contrast = slider_to_contrast(self.slider_val.get())
        trial = self.trial_list[self.current_trial_in_block]
        condition = self.current_block_cond["condition"]
        photos = generate_trial_photos(
            self.prepared_stimulus,
            condition=condition,
            test_contrast=test_contrast,
            reference_contrast=trial["ref_contrast"],
            config=self.session_config,
            calibration=self.display_calibration,
        )
        self.canvas1.delete("stim")
        self.canvas2.delete("stim")
        cx1 = self.width // 2 + self.offset_x.get()
        cy1 = self.height // 2 + self.offset_y.get()
        cx2 = self.canvas2.winfo_width() // 2
        cy2 = self.canvas2.winfo_height() // 2
        gap_fg = int(2.0 * self.prepared_stimulus.ppd_fg)
        gap_bg = int(2.0 * self.prepared_stimulus.ppd_bg)

        self.photo_ref_fg = photos["photo_ref_fg"]
        if condition in DUAL_CONDITIONS:
            self.photo_test_fg = photos["photo_test_fg"]
            self.photo_noise_bg = photos["photo_noise_bg"]
            self.canvas2.create_image(
                cx2, cy2 - gap_fg, image=self.photo_ref_fg,
                anchor="center", tags="stim",
            )
            self.canvas1.create_image(
                cx1 + self.prepared_stimulus.background_center_offset_x,
                cy1 + gap_bg, image=self.photo_noise_bg,
                anchor="center", tags="stim",
            )
            self.canvas2.create_image(
                cx2, cy2 + gap_fg, image=self.photo_test_fg,
                anchor="center", tags="stim",
            )
        else:
            self.photo_test = photos["photo_test"]
            self.canvas2.create_image(
                cx2, cy2 - gap_fg, image=self.photo_ref_fg,
                anchor="center", tags="stim",
            )
            self.canvas2.create_image(
                cx2, cy2 + gap_fg, image=self.photo_test,
                anchor="center", tags="stim",
            )

    def save_and_next(self, event=None) -> None:
        self._destroy_frame("ctrl_frame")
        self.clear_key_bindings()
        matched = slider_to_contrast(self.slider_val.get())
        self.add_trial_result(build_result_row(self, matched))
        self.current_trial_in_block += 1
        self.current_trial_in_experiment += 1
        self.root.after(500, self.run_trial)

    def _save_results_and_finish(self) -> None:
        output = save_session_results(
            Path(self.result_dir), self.results, self.session_config
        )
        messagebox.showinfo(
            "Completed", f"{self.session_type.title()} finished.\nData saved to: {output}"
        )
        self.root.destroy()

    def _destroy_frame(self, attribute: str) -> None:
        frame = getattr(self, attribute, None)
        if frame is not None and frame.winfo_exists():
            frame.destroy()