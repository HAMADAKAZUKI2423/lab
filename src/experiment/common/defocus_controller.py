"""両実験で共有するデフォーカスマッチングの進行制御。"""

import csv
import math
from pathlib import Path
import random
import tkinter as tk

from .defocus_view import prepare_defocus_trial, update_defocus_view


DEFAULT_CPD = 4.0
DEFAULT_REPETITIONS = 5


def setup_defocus_matching_ui(
    app,
    *,
    cpd: float = DEFAULT_CPD,
    repetitions: int = DEFAULT_REPETITIONS,
) -> None:
    """4 cpdの帯域制限ノイズを使ったdefocus matchingを開始する。"""
    if cpd <= 0:
        raise ValueError(f"cpd must be positive: {cpd}")
    if repetitions <= 0:
        raise ValueError(f"repetitions must be positive: {repetitions}")

    app._destroy_frame("ctrl_frame")
    app.clear_key_bindings()
    app.canvas1.delete("all")
    app.canvas2.delete("all")
    app.defocus_match_trials = [
        {
            "trial": index + 1,
            "cpd": float(cpd),
            "seed": random.randrange(2**32),
        }
        for index in range(repetitions)
    ]
    app.current_match_idx = 0
    app.match_pd_results = []
    _show_step(app)


def _show_step(app) -> None:
    app._destroy_frame("ctrl_frame")
    app.clear_key_bindings()
    trial = app.defocus_match_trials[app.current_match_idx]
    prepare_defocus_trial(app, cpd=trial["cpd"], seed=trial["seed"])
    app.ctrl_frame = tk.Frame(app.root, bg="gray")
    app.ctrl_frame.place(relx=0.5, rely=0.8, anchor="center")
    app.pupil_diameter_val.set(4.0)
    # 右眼・左眼それぞれの最初の1試行だけ表示する。
    if app.current_match_idx == 0:
        tk.Scale(
            app.ctrl_frame,
            from_=6.0,
            to=1.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            length=400,
            variable=app.pupil_diameter_val,
            command=lambda *_: update_defocus_view(app),
        ).pack(pady=10)
    current_step = app.current_match_idx + 1
    total_steps = len(app.defocus_match_trials)
    button_text = "Matching Done" if current_step == total_steps else "Next Matching"
    button = tk.Button(
        app.ctrl_frame, text=button_text, command=lambda: _record_and_continue(app)
    )
    button.pack(pady=10)
    button.focus_set()
    instruction = (
        f"Defocus Matching ({current_step}/{total_steps})\n"
        "Use ← and → to adjust the test stimulus.\n"
        "Press ↓ to confirm."
    )
    tk.Label(
        app.ctrl_frame,
        text=instruction,
        bg="gray",
        fg="white",
        font=("Arial", 16, "bold"),
    ).pack(pady=10, padx=20)
    app.key_bindings["<Down>"] = app.root.bind(
        "<Down>", lambda event: _record_and_continue(app)
    )
    app.key_bindings["<Left>"] = app.root.bind(
        "<Left>", lambda event: _handle_key(app, event)
    )
    app.key_bindings["<Right>"] = app.root.bind(
        "<Right>", lambda event: _handle_key(app, event)
    )
    app.root.focus_set()
    update_defocus_view(app)


def _record_and_continue(app) -> None:
    if app.current_match_idx >= len(app.defocus_match_trials):
        return
    trial = app.defocus_match_trials[app.current_match_idx]
    pupil_diameter = app.pupil_diameter_val.get()
    app.match_pd_results.append(pupil_diameter)
    current_eye = app.calibration_eyes[app.current_calib_eye_idx]
    app.detailed_defocus_results.append(
        {
            "ID": app.participant_id.get(),
            "Eye": current_eye,
            "Trial": trial["trial"],
            "Spatial_Freq(cpd)": trial["cpd"],
            "Noise_Seed": trial["seed"],
            "Matched_PD(mm)": pupil_diameter,
        }
    )
    print(
        f"Defocus match result: trial={trial['trial']}, "
        f"{trial['cpd']}cpd -> {pupil_diameter}mm"
    )
    app.current_match_idx += 1
    if app.current_match_idx < len(app.defocus_match_trials):
        _show_step(app)
        return
    average = sum(app.match_pd_results) / len(app.match_pd_results)
    app.pupil_diameter_val.set(round(average, 2))
    app.current_pd_mean = average
    current_eye = app.calibration_eyes[app.current_calib_eye_idx]
    print(
        f"Defocus matching mean: eye={current_eye}, "
        f"n={len(app.match_pd_results)}, mean={average:.3f}mm"
    )
    count = len(app.match_pd_results)
    app.current_pd_std = (
        math.sqrt(
            sum((value - average) ** 2 for value in app.match_pd_results)
            / (count - 1)
        )
        if count > 1
        else 0.0
    )
    _finish_eye(app)


def _finish_eye(app) -> None:
    current_eye = app.calibration_eyes[app.current_calib_eye_idx]
    app.calib_results[current_eye] = {
        "offset_x": app.offset_x.get(),
        "offset_y": app.offset_y.get(),
        "pd_mean": sum(app.match_pd_results) / len(app.match_pd_results),
    }
    app.current_calib_eye_idx += 1
    app.canvas1.delete("all")
    app.canvas2.delete("all")
    if app.current_calib_eye_idx >= len(app.calibration_eyes):
        _save_results(app)
    app.clear_key_bindings()
    app.start_eye_calibration()


def _save_results(app) -> None:
    if not app.detailed_defocus_results:
        return
    result_dir = Path(app.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    config_filename = getattr(
        getattr(app, "session_config", None), "defocus_result_filename", None
    )
    filename = config_filename or (
        "defocus_matching_training.csv"
        if getattr(app, "session_type", "experiment") == "training"
        else "defocus_matching.csv"
    )
    output = result_dir / filename
    fields = [
        "ID", "Eye", "Trial", "Spatial_Freq(cpd)",
        "Noise_Seed", "Matched_PD(mm)",
    ]
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(app.detailed_defocus_results)
    print(f"Detailed defocus matching results saved to {output}")


def _handle_key(app, event):
    current = app.pupil_diameter_val.get()
    if event.keysym == "Left":
        app.pupil_diameter_val.set(max(1.0, current - 0.1))
    elif event.keysym == "Right":
        app.pupil_diameter_val.set(min(6.0, current + 0.1))
    update_defocus_view(app)
    return "break"
