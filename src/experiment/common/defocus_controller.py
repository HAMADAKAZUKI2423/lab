"""両実験で共有するデフォーカスマッチングの進行制御。"""

import csv
import math
from pathlib import Path
import random
import tkinter as tk

from .defocus_view import update_defocus_view


DEFAULT_PATTERNS = ("checker", "checker_45", "stripe", "border", "noise")
DEFAULT_CPDS = (2, 4)


def setup_defocus_matching_ui(
    app,
    *,
    patterns=DEFAULT_PATTERNS,
    cpds=DEFAULT_CPDS,
) -> None:
    """指定したパターンと空間周波数でmatchingを開始する。"""
    app._destroy_frame("ctrl_frame")
    app.clear_key_bindings()
    app.canvas1.delete("all")
    app.canvas2.delete("all")
    app.defocus_match_patterns = [
        (pattern, cpd) for pattern in patterns for cpd in cpds
    ]
    random.shuffle(app.defocus_match_patterns)
    app.current_match_idx = 0
    app.match_pd_results = []
    _show_step(app)


def _show_step(app) -> None:
    app._destroy_frame("ctrl_frame")
    app.clear_key_bindings()
    app.ctrl_frame = tk.Frame(app.root, bg="gray")
    app.ctrl_frame.place(relx=0.5, rely=0.8, anchor="center")
    app.pupil_diameter_val.set(4.0)
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
    total_steps = len(app.defocus_match_patterns)
    button_text = "Matching Done" if current_step == total_steps else "Next Matching"
    button = tk.Button(
        app.ctrl_frame, text=button_text, command=lambda: _record_and_continue(app)
    )
    button.pack(pady=10)
    button.focus_set()
    instruction = (
        f"Defocus Matching ({current_step}/{total_steps})\n"
        "Adjust the slider to match the blur on Window 2 with Window 1.\n"
        "Press Down to confirm."
    )
    tk.Label(
        app.ctrl_frame,
        text=instruction,
        bg="gray",
        fg="white",
        font=("Arial", 12),
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
    if app.current_match_idx >= len(app.defocus_match_patterns):
        return
    pattern, cpd = app.defocus_match_patterns[app.current_match_idx]
    pupil_diameter = app.pupil_diameter_val.get()
    app.match_pd_results.append(pupil_diameter)
    current_eye = app.calibration_eyes[app.current_calib_eye_idx]
    app.detailed_defocus_results.append(
        {
            "ID": app.participant_id.get(),
            "Eye": current_eye,
            "Pattern": pattern,
            "Spatial_Freq(cpd)": cpd,
            "Matched_PD(mm)": pupil_diameter,
        }
    )
    print(
        f"Defocus match result: {pattern}_{cpd}cpd -> {pupil_diameter}mm"
    )
    app.current_match_idx += 1
    if app.current_match_idx < len(app.defocus_match_patterns):
        _show_step(app)
        return
    average = sum(app.match_pd_results) / len(app.match_pd_results)
    app.pupil_diameter_val.set(round(average, 2))
    app.current_pd_mean = average
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
        "ID", "Eye", "Pattern", "Spatial_Freq(cpd)", "Matched_PD(mm)"
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
