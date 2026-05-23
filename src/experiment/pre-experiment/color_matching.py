# color_matching.py
# カラーマッチング関連の関数とクラス
# XYZ色空間でのカラーマッチングを実装
# UIはdefocus_matchingをベースに、左右配置で参考画像とテスト画像を表示

import tkinter as tk
import os
import random
import math
import csv
import datetime
from PIL import Image, ImageTk
import stimuli_utils
import numpy as np

# --- カラーマッチング設定 ---
VISUAL_ANGLE_DEG = 7.9   # 画像の視角 (degree)
WIN2_MARKER_COLOR = 'white'    # Window 2 (実験者側) のマーカー色
MARKER_LINE_WIDTH = 5  # マーカーの線の太さ

script_dir = os.path.dirname(os.path.abspath(__file__))
lab_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

# ============================================================
# RGB <-> XYZ 変換関数
# ============================================================

def rgb_to_xyz(rgb_normalized):
    """
    RGB [0-1] をXYZ [0-1] に変換
    sRGBガンマ補正を考慮
    
    Args:
        rgb_normalized: np.array shape (3,) or (..., 3) with values in [0, 1]
    
    Returns:
        xyz_normalized: np.array same shape with XYZ values
    """
    # ガンマ逆補正 (sRGB -> linear)
    rgb_linear = np.where(
        rgb_normalized <= 0.04045,
        rgb_normalized / 12.92,
        np.power((rgb_normalized + 0.055) / 1.055, 2.4)
    )
    
    # RGB (linear) -> XYZ 変換行列 (D65光源)
    # sRGB to XYZ matrix
    transform_matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    
    # 行列乗算
    if rgb_linear.ndim == 1:
        xyz_linear = transform_matrix @ rgb_linear
    else:
        xyz_linear = np.dot(rgb_linear, transform_matrix.T)
    
    return xyz_linear

def xyz_to_rgb(xyz_normalized):
    """
    XYZ [0-1] をRGB [0-1] に変換
    
    Args:
        xyz_normalized: np.array shape (3,) or (..., 3) with values in [0, 1]
    
    Returns:
        rgb_normalized: np.array same shape with RGB values in [0, 1]
    """
    # XYZ -> RGB (linear) 逆行列 (D65光源)
    inverse_matrix = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ])
    
    # 行列乗算
    if xyz_normalized.ndim == 1:
        rgb_linear = inverse_matrix @ xyz_normalized
    else:
        rgb_linear = np.dot(xyz_normalized, inverse_matrix.T)
    
    # クリップ
    rgb_linear = np.clip(rgb_linear, 0, 1)
    
    # ガンマ補正 (linear -> sRGB)
    rgb_normalized = np.where(
        rgb_linear <= 0.0031308,
        12.92 * rgb_linear,
        1.055 * np.power(rgb_linear, 1.0/2.4) - 0.055
    )
    
    # クリップ
    rgb_normalized = np.clip(rgb_normalized, 0, 1)
    
    return rgb_normalized


def rgb_to_hex(rgb_normalized):
    """RGB [0-1] を Tkinter で使える #RRGGBB 形式に変換"""
    rgb_int = np.round(np.clip(rgb_normalized, 0, 1) * 255).astype(int)
    return "#{:02x}{:02x}{:02x}".format(*rgb_int)


# ============================================================
# カラーマッチング UI 関数
# ============================================================

def _clear_key_bindings(app):
    if hasattr(app, 'key_bindings'):
        for key, binding_id in list(app.key_bindings.items()):
            try:
                app.root.unbind(key, binding_id)
            except Exception:
                pass
        app.key_bindings.clear()


def setup_color_matching_calibration(app):
    """最初に位置キャリブレーションを行う"""
    if hasattr(app, 'ctrl_frame') and app.ctrl_frame.winfo_exists():
        app.ctrl_frame.destroy()
    app.canvas1.delete("all")
    app.canvas2.delete("all")
    _clear_key_bindings(app)

    if not hasattr(app, 'color_x_factor_val'):
        app.color_x_factor_val = tk.DoubleVar(value=1.0)
    if not hasattr(app, 'color_z_factor_val'):
        app.color_z_factor_val = tk.DoubleVar(value=1.0)
    if not hasattr(app, 'offset_x'):
        app.offset_x = tk.DoubleVar(value=0)
    if not hasattr(app, 'offset_y'):
        app.offset_y = tk.DoubleVar(value=0)

    _show_color_matching_calibration(app)


def _show_color_matching_calibration(app):
    if hasattr(app, 'ctrl_frame') and app.ctrl_frame.winfo_exists():
        app.ctrl_frame.destroy()
    _clear_key_bindings(app)

    app.ctrl_frame = tk.Frame(app.root, bg='gray')
    app.ctrl_frame.place(relx=0.5, rely=0.8, anchor='center')

    instruction_text = (
        "Calibration - Adjust the offset to align the crosshairs.\n"
        "Use arrow keys: ← → ↑ ↓\n"
        "Press 'Enter' to confirm."
    )
    tk.Label(app.ctrl_frame, text=instruction_text, bg='gray', fg='white', font=("Arial", 12)).pack(pady=10, padx=20)

    btn = tk.Button(app.ctrl_frame, text="Next", command=lambda: _on_color_matching_calibration_complete(app))
    btn.pack(pady=10)
    btn.focus_set()

    app.key_bindings['<Left>'] = app.root.bind('<Left>', lambda event: _handle_color_calibration_key(app, -1, 0))
    app.key_bindings['<Right>'] = app.root.bind('<Right>', lambda event: _handle_color_calibration_key(app, 1, 0))
    app.key_bindings['<Up>'] = app.root.bind('<Up>', lambda event: _handle_color_calibration_key(app, 0, -1))
    app.key_bindings['<Down>'] = app.root.bind('<Down>', lambda event: _handle_color_calibration_key(app, 0, 1))
    app.key_bindings['<Return>'] = app.root.bind('<Return>', lambda event: _on_color_matching_calibration_complete(app))
    app.root.focus_set()

    update_color_matching_calibration_view(app)


def _handle_color_calibration_key(app, dx, dy):
    step = 1
    app.offset_x.set(app.offset_x.get() + dx)
    app.offset_y.set(app.offset_y.get() + dy)
    update_color_matching_calibration_view(app)
    return "break"


def _on_color_matching_calibration_complete(app):
    _clear_key_bindings(app)
    if hasattr(app, 'ctrl_frame') and app.ctrl_frame.winfo_exists():
        app.ctrl_frame.destroy()
    setup_color_matching_ui(app)


def update_color_matching_calibration_view(app):
    app.canvas1.delete("calib")
    app.canvas2.delete("calib")

    d_fg = app.distance1
    square_size = stimuli_utils.get_size_for_visual_angle(d_fg, VISUAL_ANGLE_DEG)
    square_size = min(square_size,
                      app.canvas1.winfo_height() - 120,
                      app.canvas1.winfo_width() - 120,
                      app.canvas2.winfo_height() - 120,
                      app.canvas2.winfo_width() - 120)
    square_size = max(100, square_size)

    cx1 = app.canvas1.winfo_width() // 2 + int(app.offset_x.get())
    cy1 = app.canvas1.winfo_height() // 2 + int(app.offset_y.get())
    cx2 = app.canvas2.winfo_width() // 2
    cy2 = app.canvas2.winfo_height() // 2

    half = square_size // 2
    x01 = cx1 - half
    y01 = cy1 - half
    x11 = cx1 + half
    y11 = cy1 + half
    x02 = cx2 - half
    y02 = cy2 - half
    x12 = cx2 + half
    y12 = cy2 + half

    stimuli_utils.draw_image_corner_brackets(
        app.canvas1, square_size, square_size,
        offset_x=int(app.offset_x.get()), offset_y=int(app.offset_y.get()),
        color='red', line_width=stimuli_utils.MARKER_LINE_WIDTH * 1.5
    )
    stimuli_utils.draw_image_corner_brackets(
        app.canvas2, square_size, square_size,
        offset_x=0, offset_y=0,
        color='white', line_width=stimuli_utils.MARKER_LINE_WIDTH
    )
    stimuli_utils.draw_center_cross(app.canvas2, offset_x=0, offset_y=0, color='white')
    app.canvas1.create_text(cx1, y01 - 20, text='Window 1 Calibration', fill='red', font=("Arial", 14), tags='calib')
    app.canvas2.create_text(cx2, y02 - 20, text='Window 2 Calibration', fill='white', font=("Arial", 14), tags='calib')


def setup_color_matching_ui(app):
    """ステップ1: カラーマッチング用UIを構築し表示する"""
    if hasattr(app, 'ctrl_frame') and app.ctrl_frame.winfo_exists():
        app.ctrl_frame.destroy()
    app.canvas1.delete("all")
    app.canvas2.delete("all")
    _clear_key_bindings(app)

    # 条件を R, G, B の3色に設定
    app.color_match_conditions = ["R", "G", "B"]
    app.current_condition_idx = 0
    app.color_match_results = []  # 各条件の調整結果を保存
    
    _show_color_matching_step(app)


def _show_color_matching_step(app):
    """カラーマッチングの1ステップを表示"""
    if hasattr(app, 'ctrl_frame') and app.ctrl_frame.winfo_exists():
        app.ctrl_frame.destroy()
    _clear_key_bindings(app)

    app.ctrl_frame = tk.Frame(app.root, bg='gray')
    app.ctrl_frame.place(relx=0.5, rely=0.8, anchor='center')

    # XとZのスライダーを表示
    app.color_x_factor_val.set(1.0)
    app.color_z_factor_val.set(1.0)
    tk.Label(app.ctrl_frame, text='X adjust', bg='gray', fg='white', font=("Arial", 11)).pack()
    slider_x = tk.Scale(app.ctrl_frame, from_=0.5, to=1.5, resolution=0.01, orient=tk.HORIZONTAL,
                        length=400, variable=app.color_x_factor_val,
                        command=lambda *args: update_color_view(app))
    slider_x.pack(pady=5)

    tk.Label(app.ctrl_frame, text='Z adjust', bg='gray', fg='white', font=("Arial", 11)).pack()
    slider_z = tk.Scale(app.ctrl_frame, from_=0.5, to=1.5, resolution=0.01, orient=tk.HORIZONTAL,
                        length=400, variable=app.color_z_factor_val,
                        command=lambda *args: update_color_view(app))
    slider_z.pack(pady=5)

    total_steps = len(app.color_match_conditions)
    current_step = app.current_condition_idx + 1
    button_text = "Matching Done" if current_step == total_steps else "Next Matching"

    btn = tk.Button(app.ctrl_frame, text=button_text, command=lambda: _next_color_matching_step(app))
    btn.pack(pady=10)
    btn.focus_set()
    app.key_bindings['<Down>'] = app.root.bind('<Down>', lambda event: _next_color_matching_step(app))

    condition = app.color_match_conditions[app.current_condition_idx]
    instruction_text = (
        f"Color Matching - {condition} Condition ({current_step}/{total_steps})\n"
        "Adjust X with the slider or Left/Right arrows.\n"
        "Adjust Z with Up/Down arrows.\n"
        "Press Down or Enter to confirm."
    )
    tk.Label(app.ctrl_frame, text=instruction_text,
             bg='gray', fg='white', font=("Arial", 12)).pack(pady=10, padx=20)

    app.key_bindings['<Left>'] = app.root.bind('<Left>', lambda e: _handle_color_key_press(app, e, 'x_decrease'))
    app.key_bindings['<Right>'] = app.root.bind('<Right>', lambda e: _handle_color_key_press(app, e, 'x_increase'))
    app.key_bindings['<Up>'] = app.root.bind('<Up>', lambda e: _handle_color_key_press(app, e, 'z_increase'))
    app.key_bindings['<Down>'] = app.root.bind('<Down>', lambda e: _handle_color_key_press(app, e, 'z_decrease'))
    app.key_bindings['<Return>'] = app.root.bind('<Return>', lambda event: _next_color_matching_step(app))
    app.root.focus_set()

    update_color_view(app)


def _next_color_matching_step(app):
    """次のマッチングステップへ"""
    # 結果を記録
    condition = app.color_match_conditions[app.current_condition_idx]
    x_val = app.color_x_factor_val.get()
    z_val = app.color_z_factor_val.get()
    app.color_match_results.append({
        'condition': condition,
        'x_factor': x_val,
        'z_factor': z_val
    })
    print(f"Color match result: {condition} -> X={x_val:.2f}, Z={z_val:.2f}")

    app.current_condition_idx += 1
    if app.current_condition_idx < len(app.color_match_conditions):
        _show_color_matching_step(app)
    else:
        # すべて終わったら結果を保存
        save_color_matching_results(app)


def _handle_color_key_press(app, event, action):
    """キー入力でカラーマッチングパラメータを調整
    
    Args:
        app: アプリケーション
        event: キーイベント
        action: 'x_increase', 'x_decrease', 'z_increase', 'z_decrease'
    """
    step = 0.01
    min_val = 0.5
    max_val = 1.5

    if action == 'x_increase':
        current_x = app.color_x_factor_val.get()
        new_val = min(max_val, current_x + step)
        app.color_x_factor_val.set(new_val)
    elif action == 'x_decrease':
        current_x = app.color_x_factor_val.get()
        new_val = max(min_val, current_x - step)
        app.color_x_factor_val.set(new_val)
    elif action == 'z_increase':
        current_z = app.color_z_factor_val.get()
        new_val = min(max_val, current_z + step)
        app.color_z_factor_val.set(new_val)
    elif action == 'z_decrease':
        current_z = app.color_z_factor_val.get()
        new_val = max(min_val, current_z - step)
        app.color_z_factor_val.set(new_val)
    else:
        return
    
    update_color_view(app)
    return "break"


def update_color_view(app):
    """カラーマッチング画面の表示を更新"""
    app.canvas1.delete("match")
    app.canvas2.delete("match")

    condition = app.color_match_conditions[app.current_condition_idx]
    
    if condition == "R":
        ref_color_rgb = np.array([255, 0, 0], dtype=np.float32) / 255.0
    elif condition == "G":
        ref_color_rgb = np.array([0, 255, 0], dtype=np.float32) / 255.0
    else:
        ref_color_rgb = np.array([0, 0, 255], dtype=np.float32) / 255.0

    x_factor = app.color_x_factor_val.get()
    z_factor = app.color_z_factor_val.get()
    ref_xyz = rgb_to_xyz(ref_color_rgb)
    y_fixed = ref_xyz[1]
    adjusted_xyz = np.array([
        ref_xyz[0] * x_factor,
        y_fixed,
        ref_xyz[2] * z_factor
    ])
    test_color_rgb = xyz_to_rgb(adjusted_xyz)

    d_fg = app.distance1
    square_size = stimuli_utils.get_size_for_visual_angle(d_fg, VISUAL_ANGLE_DEG)
    square_size = min(square_size,
                      app.canvas1.winfo_height() - 120,
                      app.canvas1.winfo_width() - 120,
                      app.canvas2.winfo_height() - 120,
                      app.canvas2.winfo_width() - 120)
    square_size = max(120, square_size)
    half = square_size // 2

    def draw_half_square(canvas, offset_x=0, offset_y=0, label='', show_ref=False, show_test=False):
        width = canvas.winfo_width() if canvas.winfo_width() > 1 else 1920
        height = canvas.winfo_height() if canvas.winfo_height() > 1 else 1080
        cx = width // 2 + offset_x
        cy = height // 2 + offset_y
        x0 = cx - half
        y0 = cy - half
        x1 = cx + half
        y1 = cy + half
        mid_y = y0 + square_size // 2

        canvas.create_rectangle(x0, y0, x1, y1, outline='white', width=4, tags='match')
        if show_ref:
            canvas.create_rectangle(x0, y0, x1, mid_y,
                                    fill=rgb_to_hex(ref_color_rgb),
                                    outline='', tags='match')
        if show_test:
            canvas.create_rectangle(x0, mid_y, x1, y1,
                                    fill=rgb_to_hex(test_color_rgb),
                                    outline='', tags='match')
        canvas.create_line(x0, mid_y, x1, mid_y, fill='white', width=2, tags='match')
        canvas.create_text(cx, y0 - 20, text=label, fill='white', font=("Arial", 12), tags='match')

    draw_half_square(app.canvas1, offset_x=int(app.offset_x.get()), offset_y=int(app.offset_y.get()),
                     label='Window 1', show_ref=False, show_test=True)
    draw_half_square(app.canvas2, offset_x=0, offset_y=0,
                     label='Window 2', show_ref=True, show_test=False)


def save_color_matching_results(app):
    """カラーマッチング結果をCSVに保存"""
    if not hasattr(app, 'color_match_results') or not app.color_match_results:
        print("No color matching results to save")
        return

    p_id = app.participant_id.get() if hasattr(app, 'participant_id') else "Unknown"
    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d")

    result_dir = getattr(app, 'result_dir', os.path.join(lab_root, "results", "tables", "pre-experiment-color-matching"))
    save_folder = os.path.join(result_dir, f"{p_id}_{date_str}")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    filename = os.path.join(save_folder, f"color_matching_{p_id}_{now.strftime('%Y%m%d_%H%M%S')}.csv")
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Condition", "X_Factor", "Z_Factor"])
        writer.writeheader()
        for result in app.color_match_results:
            writer.writerow({
                "ID": p_id,
                "Condition": result['condition'],
                "X_Factor": result['x_factor'],
                "Z_Factor": result['z_factor']
            })
    print(f"Color matching results saved to {filename}")

    # Canvas をクリアして次のステップへ
    app.canvas1.delete("all")
    app.canvas2.delete("all")
    
    # 実験へ進むか、終了するかは呼び出し元で決定
    # ここではカラーマッチングの完了を示す
    if hasattr(app, 'finish_color_matching_callback'):
        app.finish_color_matching_callback()
