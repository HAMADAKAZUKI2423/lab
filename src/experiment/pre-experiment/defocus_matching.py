# defocus_matching.py
# デフォーカスマッチング関連の関数とクラス

import tkinter as tk
import os
import random
import math
import csv
import datetime
from PIL import Image, ImageTk, ImageFilter
import stimuli_utils
import numpy as np
import torch

# --- デフォーカスマッチング設定 ---
DEFOCUS_BLUR_SCALE_FACTOR = 0.55
VISUAL_ANGLE_DEG = 7.9   # 画像の視角 (degree)
WIN2_MARKER_COLOR = 'white'    # Window 2 (実験者側) のマーカー色
MARKER_LINE_WIDTH = 5  # マーカーの線の太さ

script_dir = os.path.dirname(os.path.abspath(__file__))
lab_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

def apply_torch_fft_blur(img_pil, D, pd_mm, pixels_per_deg):
    """optics_model.pyに基づくFFTを用いたデフォーカスブラー適用関数"""
    if D <= 0 or pd_mm <= 0:
        return img_pil
        
    rad2deg = 180.0 / math.pi
    mm = 1e-3
    bd_deg = rad2deg * D * pd_mm * mm
    sigma = DEFOCUS_BLUR_SCALE_FACTOR * bd_deg / 2.0
    
    if sigma <= 0:
        return img_pil
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    img_np = np.array(img_pil).astype(np.float32)
    is_rgb = len(img_np.shape) == 3
    
    if is_rgb:
        img_np = np.transpose(img_np, (2, 0, 1)) # HWC to CHW
        
    img_tensor = torch.from_numpy(img_np).to(device)
    h, w = img_tensor.shape[-2:]
    
    x_deg = torch.linspace(-w/2/pixels_per_deg, w/2/pixels_per_deg, w).to(device)
    y_deg = torch.linspace(-h/2/pixels_per_deg, h/2/pixels_per_deg, h).to(device)
    Y_deg, X_deg = torch.meshgrid(y_deg, x_deg, indexing='ij')
    
    psf = torch.exp(-((torch.sqrt(X_deg**2 + Y_deg**2)) ** 2) / (2 * sigma ** 2))
    psf = psf / torch.sum(psf)
    
    def FT2(tensor):
        tensor_shift = torch.fft.ifftshift(tensor, dim=(-2,-1))
        tensor_ft_shift = torch.fft.fft2(tensor_shift, norm='ortho')
        return torch.fft.fftshift(tensor_ft_shift, dim=(-2,-1))

    def iFT2(tensor):
        tensor_shift = torch.fft.ifftshift(tensor, dim=(-2,-1))
        tensor_ift_shift = torch.fft.ifft2(tensor_shift, norm='ortho')
        return torch.fft.fftshift(tensor_ift_shift, dim=(-2,-1))

    if is_rgb:
        psf = psf.unsqueeze(0)
        
    img_ft = FT2(img_tensor)
    psf_ft = FT2(psf)
    blur_tensor = torch.abs(iFT2(img_ft * psf_ft))
    
    if is_rgb:
        for c in range(3):
            blur_tensor[c] = blur_tensor[c] * torch.sum(img_tensor[c]) / (torch.sum(blur_tensor[c]) + 1e-8)
    else:
        blur_tensor = blur_tensor * torch.sum(img_tensor) / (torch.sum(blur_tensor) + 1e-8)
        
    blur_np = blur_tensor.cpu().numpy()
    
    if is_rgb:
        blur_np = np.transpose(blur_np, (1, 2, 0)) # CHW to HWC
        
    blur_np = np.clip(blur_np, 0, 255).astype(np.uint8)
    return Image.fromarray(blur_np, mode=img_pil.mode)

def setup_defocus_matching_ui(app):
    """ステップ2: デフォーカスマッチング用UIを構築し表示する"""
    if hasattr(app, 'ctrl_frame') and app.ctrl_frame.winfo_exists():
        app.ctrl_frame.destroy()
    app.canvas1.delete("all")
    app.canvas2.delete("all")

    # Clear previous key bindings (from calibration)
    for key, binding_id in app.key_bindings.items():
        app.root.unbind(key, binding_id)
    app.key_bindings.clear()

    # パターンと空間周波数の組み合わせを作成
    patterns = ["checker", "checker_45", "stripe", "border", "noise"]
    cpds = [2, 4]
    app.defocus_match_patterns = [(p, c) for p in patterns for c in cpds]
    random.shuffle(app.defocus_match_patterns)
    
    app.current_match_idx = 0
    app.match_pd_results = [] # 各パターンの調整結果を保存

    _show_defocus_matching_step(app)

def _show_defocus_matching_step(app):
    if hasattr(app, 'ctrl_frame') and app.ctrl_frame.winfo_exists():
        app.ctrl_frame.destroy()
    
    # Clear previous key bindings
    for key, binding_id in app.key_bindings.items():
        app.root.unbind(key, binding_id)
    app.key_bindings.clear()

    # 操作用UIフレーム
    app.ctrl_frame = tk.Frame(app.root, bg='gray')
    app.ctrl_frame.place(relx=0.5, rely=0.8, anchor='center')

    # スライダー (初期値は4.0にリセット)
    app.pupil_diameter_val.set(4.0)
    slider = tk.Scale(app.ctrl_frame, from_=6.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, 
                      length=400, variable=app.pupil_diameter_val, command=lambda *args: update_defocus_view(app))
    slider.pack(pady=10)

    total_steps = len(app.defocus_match_patterns)
    current_step = app.current_match_idx + 1
    is_last = (current_step == total_steps)
    button_text = "Matching Done" if is_last else "Next Matching"

    # 実験開始/次へボタン
    btn = tk.Button(app.ctrl_frame, text=button_text, command=lambda: _next_defocus_matching_step(app))
    btn.pack(pady=10)
    btn.focus_set()
    app.key_bindings['<Down>'] = app.root.bind('<Down>', lambda event: _next_defocus_matching_step(app))

    # 指示
    instruction_text = f"Defocus Matching ({current_step}/{total_steps})\nAdjust the slider (Left/Right arrow keys) to change the pupil diameter and match the blur on Window 2 (simulated) with Window 1 (natural blur).\nPress 'Down' arrow to confirm."
    tk.Label(app.ctrl_frame, text=instruction_text, 
             bg='gray', fg='white', font=("Arial", 12)).pack(pady=10, padx=20)

    # Bind keys for defocus matching
    app.key_bindings['<Left>'] = app.root.bind('<Left>', lambda e: _handle_defocus_key_press(app, e))
    app.key_bindings['<Right>'] = app.root.bind('<Right>', lambda e: _handle_defocus_key_press(app, e))
    app.root.focus_set() # Ensure root has focus for key events

    # 初回表示
    update_defocus_view(app)

def _next_defocus_matching_step(app):
    # Guard: if called after completion, ignore
    if not hasattr(app, 'defocus_match_patterns') or app.current_match_idx >= len(app.defocus_match_patterns):
        return

    # 結果を記録
    pattern, cpd = app.defocus_match_patterns[app.current_match_idx]
    pd_val = app.pupil_diameter_val.get()
    app.match_pd_results.append(pd_val)
    print(f"Defocus match result: {pattern}_{cpd}cpd -> {pd_val}mm")
    
    if not hasattr(app, 'detailed_defocus_results'):
        app.detailed_defocus_results = []
        
    current_eye = app.calibration_eyes[app.current_calib_eye_idx] if hasattr(app, 'calibration_eyes') else "Unknown"
    app.detailed_defocus_results.append({
        "ID": app.participant_id.get() if hasattr(app, 'participant_id') else "Unknown",
        "Eye": current_eye,
        "Pattern": pattern,
        "Spatial_Freq(cpd)": cpd,
        "Matched_PD(mm)": pd_val
    })

    app.current_match_idx += 1
    if app.current_match_idx < len(app.defocus_match_patterns):
        _show_defocus_matching_step(app)
    else:
        # すべて終わったら平均値を計算して設定
        avg_pd = sum(app.match_pd_results) / len(app.match_pd_results)
        app.pupil_diameter_val.set(round(avg_pd, 2))
        
        app.current_pd_mean = avg_pd
        n = len(app.match_pd_results)
        app.current_pd_std = math.sqrt(sum((x - avg_pd)**2 for x in app.match_pd_results) / (n - 1)) if n > 1 else 0.0
        
        print(f"Average pupil diameter set to: {app.pupil_diameter_val.get()}mm")
        
        finish_eye_defocus_matching(app)
        
def finish_eye_defocus_matching(app):
    current_eye = app.calibration_eyes[app.current_calib_eye_idx]
    app.calib_results[current_eye] = {
        "offset_x": app.offset_x.get(),
        "offset_y": app.offset_y.get(),
        "pd_mean": sum(app.match_pd_results)/len(app.match_pd_results)
    }
    app.current_calib_eye_idx += 1
    # Canvas をクリアしてから次のステップへ
    app.canvas1.delete("all")
    app.canvas2.delete("all")
    
    # すべての目 (左右) のデフォーカスマッチングが終わったらCSVに保存する
    if app.current_calib_eye_idx >= len(app.calibration_eyes):
        if hasattr(app, 'detailed_defocus_results') and app.detailed_defocus_results:
            p_id = app.participant_id.get() if hasattr(app, 'participant_id') else "Unknown"
            now = datetime.datetime.now()
            date_str = now.strftime("%Y%m%d")
            
            result_dir = getattr(app, 'result_dir', os.path.join(lab_root, "results", "tables", "pre-experiment-matching"))
            save_folder = os.path.join(result_dir, f"{p_id}_{date_str}")
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
                
            filename = os.path.join(save_folder, f"defocus_matching_{p_id}_{now.strftime('%Y%m%d_%H%M%S')}.csv")
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["ID", "Eye", "Pattern", "Spatial_Freq(cpd)", "Matched_PD(mm)"])
                writer.writeheader()
                writer.writerows(app.detailed_defocus_results)
            print(f"Detailed defocus matching results saved to {filename}")

    # Clear any remaining key bindings to avoid callbacks after finish
    try:
        for key, binding_id in list(app.key_bindings.items()):
            app.root.unbind(key, binding_id)
    except Exception:
        pass
    app.key_bindings.clear()

    app.start_eye_calibration()

def _handle_defocus_key_press(app, event):
    """Handles key presses for defocus matching UI.
    Left arrow decreases pupil diameter, Right arrow increases it.
    """
    step = 0.1  # Step for pupil diameter adjustment
    current_pd = app.pupil_diameter_val.get()
    min_val = 1.0
    max_val = 6.0 # From the slider definition in setup_defocus_matching_ui

    if event.keysym == 'Left': # Decrease pupil diameter
        new_val = max(min_val, current_pd - step)
        app.pupil_diameter_val.set(new_val)
    elif event.keysym == 'Right': # Increase pupil diameter
        new_val = min(max_val, current_pd + step)
        app.pupil_diameter_val.set(new_val)
    
    update_defocus_view(app)
    return "break"

def update_defocus_view(app):
    """デフォーカスマッチング画面の表示を更新する"""
    app.canvas1.delete("match")
    app.canvas2.delete("match")
    app.canvas2.delete("calib") # キャリブレーション枠を一旦消して再描画

    d_fg = app.distance1
    d_bg = app.distance2
    fg_size = stimuli_utils.get_size_for_visual_angle(d_fg, VISUAL_ANGLE_DEG)
    bg_size = stimuli_utils.get_size_for_visual_angle(d_bg, VISUAL_ANGLE_DEG)

    # 1. 前景と背景の距離からディオプトリ差Dを計算
    d_fg_m = d_fg / 100.0
    d_bg_m = d_bg / 100.0
    if d_fg_m <= 0 or d_bg_m <= 0:
        D = 0
    else:
        D = abs(1/d_fg_m - 1/d_bg_m)

    pd_mm = app.pupil_diameter_val.get()
    
    pixels_per_deg_fg = stimuli_utils.get_size_for_visual_angle(d_fg, 1.0)

    pattern_name, cpd = app.defocus_match_patterns[app.current_match_idx]
    
    matching_dir = os.path.join(lab_root, "data", "processed", "images", "pre-experiment-matching", "defocus-matching")
    fg_img_path = os.path.join(matching_dir, f"FG_{pattern_name}_{d_fg}cm_{cpd}cpd.png")
    bg_img_path = os.path.join(matching_dir, f"BG_{pattern_name}_{d_bg}cm_{cpd}cpd.png")

    try:
        img_fg = Image.open(fg_img_path).convert('L')
    except Exception as e:
        img_fg = Image.new('L', (fg_size, fg_size // 2), 128)
        print(f"Error loading matching FG image: {e}")

    try:
        img_bg = Image.open(bg_img_path).convert('L')
    except Exception as e:
        img_bg = Image.new('L', (bg_size, bg_size // 2), 128)
        print(f"Error loading matching BG image: {e}")

    img_fg = img_fg.resize((fg_size, fg_size // 2), Image.LANCZOS)
    img_fg = apply_torch_fft_blur(img_fg, D, pd_mm, pixels_per_deg_fg)
    img_bg = img_bg.resize((bg_size, bg_size // 2), Image.LANCZOS)
    
    img_fg = img_fg.transpose(Image.FLIP_LEFT_RIGHT) # 実験者ビュー用に左右反転
    color_matrix = getattr(app, 'color_matrix', None)
    if color_matrix is not None:
        img_fg = stimuli_utils.apply_color_matrix_preserve_luminance(img_fg, color_matrix.T)

    img_fg = stimuli_utils.scale_image_to_target_luminance(
        img_fg, 15.0,
        getattr(app, 'fg_lums', None), getattr(app, 'fg_pixels', None)
    )
    img_bg = stimuli_utils.scale_image_to_target_luminance(
        img_bg, 15.0,
        getattr(app, 'bg_lums', None), getattr(app, 'bg_pixels', None)
    )

    dy_fg = fg_size // 4
    dy_bg = -bg_size // 4

    app.photo_match_fg = ImageTk.PhotoImage(img_fg)
    app.photo_match_bg = ImageTk.PhotoImage(img_bg)

    # 3. 描画
    # Window 1 (Background Display)
    ox, oy = app.offset_x.get(), app.offset_y.get()
    cx1, cy1 = app.width//2 + ox, app.height//2 + oy
    app.canvas1.create_image(cx1, cy1 + dy_bg, image=app.photo_match_bg, anchor='center', tags="match")

    # Window 2 (Foreground/Simulated)
    cx2, cy2 = app.canvas2.winfo_width()//2, app.canvas2.winfo_height()//2
    app.canvas2.create_image(cx2, cy2 + dy_fg, image=app.photo_match_fg, anchor='center', tags="match")
    
    # 枠と十字マーカーの再描画 (上下それぞれの領域に)
    stimuli_utils.draw_image_corner_brackets(app.canvas2, fg_size, fg_size // 2, 0, -fg_size // 4, color=WIN2_MARKER_COLOR, line_width=MARKER_LINE_WIDTH)
    stimuli_utils.draw_center_cross(app.canvas2, offset_x=0, offset_y=-fg_size // 4, color=WIN2_MARKER_COLOR)
    
    stimuli_utils.draw_image_corner_brackets(app.canvas2, fg_size, fg_size // 2, 0, fg_size // 4, color=WIN2_MARKER_COLOR, line_width=MARKER_LINE_WIDTH)
    stimuli_utils.draw_center_cross(app.canvas2, offset_x=0, offset_y=fg_size // 4, color=WIN2_MARKER_COLOR)


# ============================================================
# pre-experiment-gabor.py 用の関数
# ============================================================

def setup_defocus_matching_ui_gabor(app):
    """pre-experiment-gabor.py 用デフォーカスマッチングUI"""
    if hasattr(app, 'ctrl_frame') and app.ctrl_frame.winfo_exists():
        app.ctrl_frame.destroy()
    app.canvas1.delete("all")
    app.canvas2.delete("all")

    # Clear previous key bindings (from calibration)
    for key, binding_id in app.key_bindings.items():
        app.root.unbind(key, binding_id)
    app.key_bindings.clear()

    # 1, 2, 4 cpd のリストを作成してシャッフル
    app.defocus_match_cpds = [1, 2, 4]
    random.shuffle(app.defocus_match_cpds)
    app.current_match_idx = 0
    app.match_pd_results = [] # 各cpdでの瞳孔径の調整結果を保存

    _show_defocus_matching_step_gabor(app)

def _show_defocus_matching_step_gabor(app):
    if hasattr(app, 'ctrl_frame') and app.ctrl_frame.winfo_exists():
        app.ctrl_frame.destroy()
    
    # Clear previous key bindings
    for key, binding_id in app.key_bindings.items():
        app.root.unbind(key, binding_id)
    app.key_bindings.clear()

    # 操作用UIフレーム
    app.ctrl_frame = tk.Frame(app.root, bg='gray')
    app.ctrl_frame.place(relx=0.5, rely=0.8, anchor='center')

    # スライダー (初期値は4.0にリセット)
    app.pupil_diameter_val.set(4.0)
    slider = tk.Scale(app.ctrl_frame, from_=6.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, 
                      length=400, variable=app.pupil_diameter_val, command=lambda *args: update_defocus_view_gabor(app))
    slider.pack(pady=10)

    total_steps = len(app.defocus_match_cpds)
    current_step = app.current_match_idx + 1
    is_last = (current_step == total_steps)
    button_text = "Start Experiment" if is_last else "Next Matching"

    # 実験開始/次へボタン
    btn = tk.Button(app.ctrl_frame, text=button_text, command=lambda: _next_defocus_matching_step_gabor(app))
    btn.pack(pady=10)
    btn.focus_set()
    app.key_bindings['<Down>'] = app.root.bind('<Down>', lambda event: _next_defocus_matching_step_gabor(app))

    # 指示
    instruction_text = f"Defocus Matching ({current_step}/{total_steps})\nAdjust the slider (Left/Right arrow keys) to change the pupil diameter and match the blur on Window 2 (simulated) with Window 1 (natural blur).\nPress 'Down' arrow to confirm."
    tk.Label(app.ctrl_frame, text=instruction_text, 
             bg='gray', fg='white', font=("Arial", 12)).pack(pady=10, padx=20)

    # Bind keys for defocus matching
    app.key_bindings['<Left>'] = app.root.bind('<Left>', lambda e: _handle_defocus_key_press_gabor(app, e))
    app.key_bindings['<Right>'] = app.root.bind('<Right>', lambda e: _handle_defocus_key_press_gabor(app, e))
    app.root.focus_set()

    # 初回表示
    update_defocus_view_gabor(app)

def _next_defocus_matching_step_gabor(app):
    # 結果を記録
    cpd = app.defocus_match_cpds[app.current_match_idx]
    pd_val = app.pupil_diameter_val.get()
    app.match_pd_results.append(pd_val)
    print(f"Defocus match result: {cpd}cpd -> {pd_val}mm")

    app.current_match_idx += 1
    if app.current_match_idx < len(app.defocus_match_cpds):
        _show_defocus_matching_step_gabor(app)
    else:
        # すべて終わったら平均値を計算して設定
        avg_pd = sum(app.match_pd_results) / len(app.match_pd_results)
        app.pupil_diameter_val.set(round(avg_pd, 2))
        
        app.current_pd_mean = avg_pd
        n = len(app.match_pd_results)
        app.current_pd_std = math.sqrt(sum((x - avg_pd)**2 for x in app.match_pd_results) / (n - 1)) if n > 1 else 0.0
        
        print(f"Average pupil diameter set to: {app.pupil_diameter_val.get()}mm")
        
        # Canvas をクリア
        app.canvas1.delete("all")
        app.canvas2.delete("all")
        
        # 実験へ
        app.start_experiment_block()

def _handle_defocus_key_press_gabor(app, event):
    step = 0.1
    current_pd = app.pupil_diameter_val.get()
    min_val = 1.0
    max_val = 6.0

    if event.keysym == 'Left':
        new_val = max(min_val, current_pd - step)
        app.pupil_diameter_val.set(new_val)
    elif event.keysym == 'Right':
        new_val = min(max_val, current_pd + step)
        app.pupil_diameter_val.set(new_val)
    
    update_defocus_view_gabor(app)
    return "break"

def update_defocus_view_gabor(app):
    """pre-experiment-gabor.py 用デフォーカスマッチング画面更新"""
    app.canvas1.delete("match")
    app.canvas2.delete("match")
    app.canvas2.delete("calib")

    d_fg = app.distance1
    d_bg = app.distance2
    fg_size = app.get_size_for_visual_angle(d_fg, VISUAL_ANGLE_DEG)
    bg_size = app.get_size_for_visual_angle(d_bg, VISUAL_ANGLE_DEG)

    # 1. 前景と背景の距離からディオプトリ差Dを計算
    d_fg_m = d_fg / 100.0
    d_bg_m = d_bg / 100.0
    if d_fg_m <= 0 or d_bg_m <= 0:
        D = 0
    else:
        D = abs(1/d_fg_m - 1/d_bg_m)

    pd_mm = app.pupil_diameter_val.get()
    
    pixels_per_deg_fg = app.get_size_for_visual_angle(d_fg, 1.0)

    # 動的にマッチング用画像を取得
    cpd = app.defocus_match_cpds[app.current_match_idx]
    matching_dir = os.path.join(lab_root, "data", "processed", "images", "pre-experiment-gabor", "defocus-matching")
    fg_img_path = os.path.join(matching_dir, f"FG_checker_{d_fg}cm_{cpd}cpd.png")
    bg_img_path = os.path.join(matching_dir, f"BG_checker_{d_bg}cm_{cpd}cpd.png")

    # 画像の読み込みと加工
    try:
        img_fg = Image.open(fg_img_path)
    except Exception as e:
        img_fg = Image.new('RGB', (fg_size, fg_size // 2), (255, 255, 255))
        print(f"Error loading matching FG image: {e}")

    try:
        img_bg = Image.open(bg_img_path)
    except Exception as e:
        img_bg = Image.new('RGB', (bg_size, bg_size // 2), (255, 255, 255))
        print(f"Error loading matching BG image: {e}")

    # 前景用 (Blurred)
    img_fg = img_fg.resize((fg_size, fg_size // 2), Image.LANCZOS)
    img_fg = apply_torch_fft_blur(img_fg, D, pd_mm, pixels_per_deg_fg)
    img_fg = img_fg.transpose(Image.FLIP_LEFT_RIGHT)
    color_matrix = getattr(app, 'color_matrix', None)
    if color_matrix is not None:
        img_fg = stimuli_utils.apply_color_matrix_preserve_luminance(img_fg, color_matrix.T)

    img_fg = stimuli_utils.scale_image_to_target_luminance(
        img_fg, 30.0,
        getattr(app, 'fg_lums', None), getattr(app, 'fg_pixels', None)
    )
    img_bg = img_bg.resize((bg_size, bg_size // 2), Image.LANCZOS)
    img_bg = stimuli_utils.scale_image_to_target_luminance(
        img_bg, 30.0,
        getattr(app, 'bg_lums', None), getattr(app, 'bg_pixels', None)
    )

    dy_fg = fg_size // 4
    dy_bg = -bg_size // 4

    app.photo_match_fg = ImageTk.PhotoImage(img_fg)
    app.photo_match_bg = ImageTk.PhotoImage(img_bg)

    # 描画
    ox, oy = app.offset_x.get(), app.offset_y.get()
    cx1, cy1 = app.width//2 + ox, app.height//2 + oy
    app.canvas1.create_image(cx1, cy1 + dy_bg, image=app.photo_match_bg, anchor='center', tags="match")

    cx2, cy2 = app.canvas2.winfo_width()//2, app.canvas2.winfo_height()//2
    app.canvas2.create_image(cx2, cy2 + dy_fg, image=app.photo_match_fg, anchor='center', tags="match")
    
    # 枠と十字マーカーの再描画
    app.draw_image_corner_brackets(app.canvas2, fg_size, fg_size // 2, 0, -fg_size // 4, color=WIN2_MARKER_COLOR, line_width=MARKER_LINE_WIDTH)
    app.draw_center_cross(app.canvas2, offset_x=0, offset_y=-fg_size // 4, color=WIN2_MARKER_COLOR)
    
    app.draw_image_corner_brackets(app.canvas2, fg_size, fg_size // 2, 0, fg_size // 4, color=WIN2_MARKER_COLOR, line_width=MARKER_LINE_WIDTH)
    app.draw_center_cross(app.canvas2, offset_x=0, offset_y=fg_size // 4, color=WIN2_MARKER_COLOR)


# ============================================================
# pre-experiment-image.py 用の関数
# ============================================================

def setup_defocus_matching_ui_image(app):
    """pre-experiment-image.py 用デフォーカスマッチングUI (シングルスライダー)"""
    if hasattr(app, 'ctrl_frame') and app.ctrl_frame.winfo_exists():
        app.ctrl_frame.destroy()
    app.canvas1.delete("all")
    app.canvas2.delete("all")

    # Clear previous key bindings
    for key, binding_id in app.key_bindings.items():
        app.root.unbind(key, binding_id)
    app.key_bindings.clear()

    # 操作用UIフレーム
    app.ctrl_frame = tk.Frame(app.root, bg='gray')
    app.ctrl_frame.place(relx=0.5, rely=0.8, anchor='center')

    # スライダー
    slider = tk.Scale(app.ctrl_frame, from_=0.0, to=3.0, resolution=0.01, orient=tk.HORIZONTAL, 
                      length=400, variable=app.defocus_val, command=lambda *args: update_defocus_view_image(app))
    slider.pack(pady=10)

    # 実験開始ボタン
    btn = tk.Button(app.ctrl_frame, text="Start Experiment", command=lambda: app.start_experiment())
    btn.pack(pady=10)
    btn.focus_set()
    app.key_bindings['<Return>'] = app.root.bind('<Return>', lambda event: app.start_experiment())

    # 指示
    instruction_text = "Adjust the slider (Left/Right arrow keys) until the blur on Window 2 (simulated) matches Window 1 (natural blur)."
    tk.Label(app.ctrl_frame, text=instruction_text, 
             bg='gray', fg='white', font=("Arial", 12)).pack(pady=10, padx=20)

    # Bind keys for defocus matching
    app.key_bindings['<Left>'] = app.root.bind('<Left>', lambda e: _handle_defocus_key_press_image(app, e))
    app.key_bindings['<Right>'] = app.root.bind('<Right>', lambda e: _handle_defocus_key_press_image(app, e))
    app.root.focus_set()

    # 初回表示
    update_defocus_view_image(app)

def _handle_defocus_key_press_image(app, event):
    """pre-experiment-image.py 用キープレスハンドラ"""
    step = 0.05
    current_defocus = app.defocus_val.get()
    min_val = 0.0
    max_val = 3.0

    if event.keysym == 'Left':
        new_val = min(max_val, current_defocus + step)
        app.defocus_val.set(new_val)
    elif event.keysym == 'Right':
        new_val = max(min_val, current_defocus - step)
        app.defocus_val.set(new_val)
    
    update_defocus_view_image(app)
    return "break"

def update_defocus_view_image(app):
    """pre-experiment-image.py 用デフォーカスマッチング画面更新"""
    app.canvas1.delete("match")
    app.canvas2.delete("match")
    app.canvas2.delete("calib")

    d_fg = app.distance1.get()
    d_bg = app.distance2.get()
    fg_size = app.get_size_for_visual_angle(d_fg, VISUAL_ANGLE_DEG)
    bg_size = app.get_size_for_visual_angle(d_bg, VISUAL_ANGLE_DEG)

    # 1. 瞳孔径とディオプトリからシグマ(pixel)を計算
    PUPIL_DIAMETER_MM = 4.0  # image用の固定値
    D = app.defocus_val.get()
    bd_deg = D * PUPIL_DIAMETER_MM * (180.0 / math.pi) / 1000.0
    sigma_deg = DEFOCUS_BLUR_SCALE_FACTOR * bd_deg / 2.0
    
    # pixels_per_deg を簡易計算
    pixels_per_deg_fg = app.get_size_for_visual_angle(d_fg, 1.0)
    sigma_pixels = sigma_deg * pixels_per_deg_fg

    # 動的にマッチング用画像を取得
    matching_dir = os.path.join(lab_root, "data", "processed", "images", "pre-experiment-gabor", "defocus-matching")
    fg_img_path = os.path.join(matching_dir, f"FG_checker_{d_fg}cm.png")
    bg_img_path = os.path.join(matching_dir, f"BG_checker_{d_bg}cm.png")

    # 画像の読み込みと加工
    try:
        img_fg = Image.open(fg_img_path)
    except Exception as e:
        img_fg = Image.new('RGB', (fg_size, fg_size // 2), (255, 255, 255))
        print(f"Error loading matching FG image: {e}")

    try:
        img_bg = Image.open(bg_img_path)
    except Exception as e:
        img_bg = Image.new('RGB', (bg_size, bg_size // 2), (255, 255, 255))
        print(f"Error loading matching BG image: {e}")

    img_fg = img_fg.resize((fg_size, fg_size // 2), Image.LANCZOS)
    img_fg = apply_torch_fft_blur(img_fg, D, PUPIL_DIAMETER_MM, pixels_per_deg_fg)
    img_fg = img_fg.transpose(Image.FLIP_LEFT_RIGHT)
    color_matrix = getattr(app, 'color_matrix', None)
    if color_matrix is not None:
        img_fg = stimuli_utils.apply_color_matrix_preserve_luminance(img_fg, color_matrix.T)

    img_fg = stimuli_utils.scale_image_to_target_luminance(
        img_fg, 30.0,
        getattr(app, 'fg_lums', None), getattr(app, 'fg_pixels', None)
    )
    img_bg = img_bg.resize((bg_size, bg_size // 2), Image.LANCZOS)
    img_bg = stimuli_utils.scale_image_to_target_luminance(
        img_bg, 30.0,
        getattr(app, 'bg_lums', None), getattr(app, 'bg_pixels', None)
    )

    dy_fg = fg_size // 4
    dy_bg = -bg_size // 4

    app.photo_match_fg = ImageTk.PhotoImage(img_fg)
    app.photo_match_bg = ImageTk.PhotoImage(img_bg)

    # 描画
    ox, oy = app.offset_x.get(), app.offset_y.get()
    cx1, cy1 = app.width//2 + ox, app.height//2 + oy
    app.canvas1.create_image(cx1, cy1 + dy_bg, image=app.photo_match_bg, anchor='center', tags="match")

    cx2, cy2 = app.canvas2.winfo_width()//2, app.canvas2.winfo_height()//2
    app.canvas2.create_image(cx2, cy2 + dy_fg, image=app.photo_match_fg, anchor='center', tags="match")
    
    # 枠と十字マーカーの再描画
    app.draw_image_corner_brackets(app.canvas2, fg_size, fg_size // 2, 0, -fg_size // 4, color=WIN2_MARKER_COLOR, line_width=MARKER_LINE_WIDTH)
    app.draw_center_cross(app.canvas2, offset_x=0, offset_y=-fg_size // 4, color=WIN2_MARKER_COLOR)
    
    app.draw_image_corner_brackets(app.canvas2, fg_size, fg_size // 2, 0, fg_size // 4, color=WIN2_MARKER_COLOR, line_width=MARKER_LINE_WIDTH)
    app.draw_center_cross(app.canvas2, offset_x=0, offset_y=fg_size // 4, color=WIN2_MARKER_COLOR)