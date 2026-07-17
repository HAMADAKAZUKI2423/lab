"""Image実験の5段階評価UI。"""

import tkinter as tk
from collections.abc import Callable


def _update_highlight(app) -> None:
    current = app.evaluation_val.get()
    for index, items in enumerate(app.eval_buttons):
        value = 5 - index
        selected = value == current
        items["canvas"].itemconfig(
            items["dot"],
            fill="black" if selected else "white",
            outline="black" if selected else "white",
        )
        items["label"].config(
            font=("Arial", 12, "bold" if selected else "normal")
        )


def _move_selection(app, delta: int):
    current = app.evaluation_val.get()
    app.evaluation_val.set(max(1, min(5, current + delta)))
    _update_highlight(app)
    return "break"


def show_evaluation_ui(
    app,
    on_confirm: Callable[[], None],
) -> None:
    app.eval_frame = tk.Frame(
        app.root,
        bg="white",
        padx=20,
        pady=20,
        relief="solid",
        borderwidth=1,
    )
    app.eval_frame.place(relx=0.5, rely=0.5, anchor="center")
    tk.Label(
        app.eval_frame,
        text=f"Trial No.{app.current_trial_index + 1} の評価",
        font=("Arial", 16),
        bg="white",
    ).pack(pady=(0, 20))

    app.evaluation_val.set(3)
    options_frame = tk.Frame(app.eval_frame, bg="white")
    options_frame.pack(pady=10, padx=20)
    app.eval_buttons.clear()

    for value in range(5, 0, -1):
        option_frame = tk.Frame(options_frame, bg="white")
        option_frame.pack(side="left", padx=15)
        canvas = tk.Canvas(
            option_frame, width=30, height=30,
            bg="white", highlightthickness=0,
        )
        canvas.pack()
        canvas.create_oval(5, 5, 25, 25, outline="black", width=2)
        dot = canvas.create_oval(
            10, 10, 20, 20, fill="white", outline="white"
        )
        label = tk.Label(
            option_frame, text=str(value), font=("Arial", 12), bg="white"
        )
        label.pack()
        app.eval_buttons.append(
            {"canvas": canvas, "dot": dot, "label": label}
        )

    description = tk.Frame(app.eval_frame, bg="white")
    description.pack(fill="x", padx=10, pady=(5, 10))
    tk.Label(description, text="5: Very clear", bg="white").pack(
        side="left"
    )
    tk.Label(description, text="1: Invisible", bg="white").pack(
        side="right"
    )
    tk.Label(
        app.eval_frame,
        text="◀ / ▶ で選択, ▼ で決定",
        font=("Arial", 10),
        bg="white",
    ).pack(pady=(10, 0))

    _update_highlight(app)
    app.key_bindings["<Left>"] = app.root.bind(
        "<Left>", lambda event: _move_selection(app, -1)
    )
    app.key_bindings["<Right>"] = app.root.bind(
        "<Right>", lambda event: _move_selection(app, 1)
    )
    app.key_bindings["<Down>"] = app.root.bind(
        "<Down>", lambda event: on_confirm()
    )
    app.root.focus_set()