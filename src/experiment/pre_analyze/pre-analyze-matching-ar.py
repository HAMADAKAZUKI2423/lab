# py .\src\experiment\pre-analyze\pre-analyze-matching-ar.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os
import glob
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 解析対象のフォルダ
script_dir = os.path.dirname(os.path.abspath(__file__))
lab_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

# 入力データのベースディレクトリ
DATA_BASE_DIR = os.path.join(lab_root, "results", "tables", "pre-experiment-matching")

parser = argparse.ArgumentParser(description='Analyze defocus matching experiment results with AR contrast.')
parser.add_argument('target_dir', nargs='?', help='Path to the target directory')
args = parser.parse_args()

if args.target_dir:
    TARGET_DIR = args.target_dir
    if not os.path.exists(TARGET_DIR):
        print(f"指定されたディレクトリが見つかりません: {TARGET_DIR}")
        exit()
else:
    # 最新のフォルダを検索
    if not os.path.exists(DATA_BASE_DIR):
        print(f"データディレクトリが見つかりません: {DATA_BASE_DIR}")
        exit()

    subdirs = [d for d in glob.glob(os.path.join(DATA_BASE_DIR, "*")) if os.path.isdir(d)]
    if not subdirs:
        print(f"データフォルダが見つかりません: {DATA_BASE_DIR}")
        exit()

    # 最新のフォルダを取得 (更新日時順)
    TARGET_DIR = max(subdirs, key=os.path.getmtime)

target_folder_name = os.path.basename(os.path.normpath(TARGET_DIR))
print(f"解析対象フォルダ: {TARGET_DIR}")

# 出力先ディレクトリの設定
OUTPUT_DIR = os.path.join(lab_root, "results", "figures", "pre-experiment-matching", target_folder_name)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
print(f"結果出力先: {OUTPUT_DIR}")

# フォルダ内のすべてのCSVファイルを取得
file_paths = glob.glob(os.path.join(TARGET_DIR, '*.csv'))

if not file_paths:
    print(f"解析対象のCSVファイルがフォルダ '{TARGET_DIR}' に見つかりませんでした。")
    exit()

all_data = []
for file in file_paths:
    df = pd.read_csv(file, encoding='utf-8')
    all_data.append(df)

if not all_data:
    print("No data to process.")
    exit()
    
final_df = pd.concat(all_data, ignore_index=True)

# 必要な列が揃っているか確認
required_columns = ["Condition", "Ocularity", "Ref_Contrast", "Matched_Contrast", "Orientation"]
for col in required_columns:
    if col not in final_df.columns:
        print(f"Error: 必要な列 '{col}' がCSVファイルに見つかりません。")
        exit()

# --- 記録された前景単体コントラストを合成後のAR拡張コントラストに変換 ---
# Load luminance config: prefer per-run config saved in the target folder, then global config
L_fg = 15.0
L_bg = 15.0
L_ref = 30.0
used_cfg_path = os.path.join(TARGET_DIR, 'used_experiment_config.json')
if os.path.exists(used_cfg_path):
    try:
        import json
        used_cfg = json.load(open(used_cfg_path, 'r', encoding='utf-8'))
        L_fg = float(used_cfg.get('L_fg', L_fg))
        L_bg = float(used_cfg.get('L_bg', L_bg))
        L_ref = float(used_cfg.get('L_ref', L_ref))
    except Exception:
        pass
else:
    config_path = os.path.join(lab_root, 'config', 'experiment_conditions.json')
    if os.path.exists(config_path):
        try:
            import json
            cfg = json.load(open(config_path, 'r', encoding='utf-8'))
            L_fg = float(cfg.get('L_fg', L_fg))
            L_bg = float(cfg.get('L_bg', L_bg))
            L_ref = float(cfg.get('L_ref', L_ref))
        except Exception:
            pass

# Ensure luminance columns exist in the concatenated dataframe; prefer existing per-row values
if 'L_fg' not in final_df.columns:
    final_df['L_fg'] = L_fg
if 'L_bg' not in final_df.columns:
    final_df['L_bg'] = L_bg
if 'L_ref' not in final_df.columns:
    final_df['L_ref'] = L_ref

# Compute AR-extended matched contrast using per-row luminance values
final_df['Matched_Contrast_AR'] = final_df['Matched_Contrast'] * (final_df['L_fg'] / (final_df['L_fg'] + final_df['L_bg']))

print("\n--- Generating bar charts (AR Extended Contrast) ---")
        
# 描画用の設定
sns.set_theme(style="whitegrid")
condition_order = ["Single plane", "Single plane + defocus simulation", "Dual plane", "Dual plane flat"]
ocularity_order = ["monocular", "binocular"]

unique_ref_contrasts = sorted(final_df['Ref_Contrast'].dropna().unique(), reverse=True)
unique_orientations = sorted(final_df['Orientation'].dropna().unique())

# Load shared models and utilities from the prepared models file
_blur_attenuation_cache = {}
import importlib.util
models_file = os.path.join(script_dir, "pre-analyze-matching-models.py")
spec = importlib.util.spec_from_file_location("preanalyze_models", models_file)
preanalyze_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preanalyze_models)

# Reuse blur calculation and model classes from the external module
calculate_blur_attenuation_cached = preanalyze_models.calculate_blur_attenuation_cached
ModelA = preanalyze_models.ModelA
ModelB = preanalyze_models.ModelB
ModelC1 = preanalyze_models.ModelC1
ModelC2 = preanalyze_models.ModelC2

def get_effective_c_bg(row):
    if row['Condition'] == 'Single plane + defocus simulation':
        dom = row.get('Dominance', 'Right')
        if dom == 'Right':
            pd_val = row.get('PD_Right', 0)
        else:
            pd_val = row.get('PD_Left', 0)
            
        if pd.isna(pd_val) or pd_val <= 0:
            pd_val = 4.0
        return calculate_blur_attenuation_cached(pd_val)
    return 1.0


def get_effective_c_bg_eye(row, eye):
    """指定眼(eye='Left'/'Right')のPDから背景blur減衰率を返す。
       defocus simulation条件以外は1.0。"""
    if row['Condition'] != 'Single plane + defocus simulation':
        return 1.0
    if eye == 'Left':
        pd_val = row.get('PD_Left', 0)
    else:
        pd_val = row.get('PD_Right', 0)
    if pd.isna(pd_val) or pd_val <= 0:
        pd_val = 4.0
    return calculate_blur_attenuation_cached(pd_val)

# --- Enhanced contrast values ---
print("\n--- Calculating effective background contrast and Enhanced Contrast ---")
final_df['Effective_C_bg'] = final_df.apply(get_effective_c_bg, axis=1)
final_df['Matched_Contrast_Enhanced'] = (final_df['Matched_Contrast'] * L_fg + final_df['Effective_C_bg'] * L_bg) / (L_fg + L_bg)

# Train ModelA on Single plane and compute predictions using external model classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_df = final_df[final_df['Condition'] == 'Single plane']
models = {}
if not train_df.empty:
    S_train = torch.tensor(train_df['Matched_Contrast_AR'].values, dtype=torch.float32).to(device)
    M_train = torch.ones(len(train_df), dtype=torch.float32).to(device)
    C_train = torch.tensor(train_df['Ref_Contrast'].values, dtype=torch.float32).to(device)

    modelA = ModelA().to(device)
    opt = optim.Adam(modelA.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    epochs = 300
    for ep in range(epochs):
        opt.zero_grad()
        pred = modelA(S_train, M_train)
        loss = loss_fn(pred, C_train)
        loss.backward()
        opt.step()
        
        # 回帰タスクのため、Accuracyの代わりにMAE(平均絶対誤差)を計算して表示します
        with torch.no_grad():
            mae = torch.mean(torch.abs(pred - C_train)).item()
        print(f"Epoch {ep+1}/{epochs} | Loss (MSE): {loss.item():.6f} | MAE: {mae:.6f}")

    # 学習済みパラメータを他モデルに流用
    models = {
        'ModelA': modelA,
        'ModelB': ModelB().to(device),
        'ModelC1': ModelC1().to(device),
        'ModelC2': ModelC2().to(device),
    }
    for name, m in models.items():
        if name != 'ModelA':
            m.raw_sigma.data = modelA.raw_sigma.data.clone()
            m.raw_beta.data = modelA.raw_beta.data.clone()

    # データフレーム全体に対して予測を計算
    preds_A, preds_B, preds_C1, preds_C2 = [], [], [], []
    for _, row in final_df.iterrows():
        C_val = torch.tensor(float(row['Ref_Contrast']), dtype=torch.float32).to(device)
        M_val = torch.tensor(1.0, dtype=torch.float32).to(device)

        # dominant-eye ベースの既存の Effective_C_bg は ModelB/ModelC1 の blur に使う
        blur_val = torch.tensor(float(row.get('Effective_C_bg', 1.0)), dtype=torch.float32).to(device)

        # 左右眼それぞれの blur を取得（defocus 条件以外は 1.0）
        blur_left = torch.tensor(float(get_effective_c_bg_eye(row, 'Left')), dtype=torch.float32).to(device)
        blur_right = torch.tensor(float(get_effective_c_bg_eye(row, 'Right')), dtype=torch.float32).to(device)

        dom = row.get('Dominance', 'Right')
        if dom == 'Right':
            pd_val = row.get('PD_Right', 0)
        else:
            pd_val = row.get('PD_Left', 0)
        if pd.isna(pd_val) or pd_val <= 0:
            pd_val = 0.0
        delta_D_val = torch.tensor(pd_val, dtype=torch.float32).to(device)

        with torch.no_grad():
            gamma_A = models['ModelA'].gamma
            sigma_A = models['ModelA'].sigma
            beta_A = models['ModelA'].beta
            pa = torch.pow(C_val * (torch.pow(sigma_A, gamma_A) + beta_A * torch.pow(M_val, gamma_A)), 1.0 / gamma_A).item()

            gamma_B = models['ModelB'].gamma
            sigma_B = models['ModelB'].sigma
            beta_B = models['ModelB'].beta
            pb = torch.pow(C_val * (torch.pow(sigma_B, gamma_B) + beta_B * torch.pow(M_val * blur_val, gamma_B)), 1.0 / gamma_B).item()

            # C1: g(delta_D) を定数 0.2 に置換
            gamma_C1 = models['ModelC1'].gamma
            sigma_C1 = models['ModelC1'].sigma
            beta_C1 = models['ModelC1'].beta
            pc1 = torch.pow(C_val * (torch.pow(sigma_C1, gamma_C1) + beta_C1 * torch.pow(M_val * blur_val * 0.2, gamma_C1)), 1.0 / gamma_C1).item()

            # C2: 左右の blur を使って各眼で B を計算し、その平均を予測値とする
            gamma_C2 = models['ModelC2'].gamma
            sigma_C2 = models['ModelC2'].sigma
            beta_C2 = models['ModelC2'].beta
            b_left = torch.pow(C_val * (torch.pow(sigma_C2, gamma_C2) + beta_C2 * torch.pow(M_val * blur_left, gamma_C2)), 1.0 / gamma_C2)
            b_right = torch.pow(C_val * (torch.pow(sigma_C2, gamma_C2) + beta_C2 * torch.pow(M_val * blur_right, gamma_C2)), 1.0 / gamma_C2)
            pc2 = (0.5 * (b_left + b_right)).item()

        preds_A.append(pa)
        preds_B.append(pb)
        preds_C1.append(pc1)
        preds_C2.append(pc2)

    final_df['Pred_ModelA'] = preds_A
    final_df['Pred_ModelB'] = preds_B
    final_df['Pred_ModelC1'] = preds_C1
    final_df['Pred_ModelC2'] = preds_C2
else:
    print('Warning: No Single plane data found for training ModelA. Skipping model predictions.')

print("\n--- Generating bar charts (Enhanced Contrast) ---")
for ref_c in unique_ref_contrasts:
    for ori in unique_orientations:
        plot_df = final_df[(final_df['Ref_Contrast'] == ref_c) & (final_df['Orientation'] == ori)]
        if plot_df.empty:
            continue
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.barplot(
            data=plot_df, 
            x='Condition', 
            y='Matched_Contrast_Enhanced', 
            hue='Ocularity',
            order=condition_order,
            hue_order=ocularity_order,
            errorbar=('ci', 95), 
            capsize=0.1, 
            err_kws={'linewidth': 1.5},
            ax=ax
        )
        
        ax.axhline(y=ref_c, color='red', linestyle='--', linewidth=2, label=f'Ref Contrast ({ref_c})')
        
        patch_idx = 0
        for oc in ocularity_order:
            for cond in condition_order:
                if patch_idx < len(ax.patches):
                    p = ax.patches[patch_idx]
                    height = p.get_height()
                    if pd.notna(height) and height > 0:
                        subset = plot_df[(plot_df['Ocularity'] == oc) & (plot_df['Condition'] == cond)]
                        if not subset.empty:
                            m = subset['Matched_Contrast_Enhanced'].mean()
                            d = subset['Matched_Contrast_Enhanced'].std() 
                            
                            x = p.get_x() + p.get_width() / 2
                            y = 0.15  # バーの足元付近
                            
                            ax.text(x, y, f"m={m:.2f}\nd={d:.2f}", ha='center', va='bottom', color='black', fontsize=10,
                                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                    patch_idx += 1
    
        ax.set_title(f'Matched Enhanced Contrast by Condition and Ocularity (Ref Contrast: {ref_c}, Ori: {ori}°)', fontsize=14)
        ax.set_ylabel('Matched Contrast (Enhanced)', fontsize=12)
        ax.set_xlabel('Condition', fontsize=12)
        ax.set_yscale('log')
        
        ax.set_ylim(0.05, 1.0) 
        ax.set_yticks([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        
        labels = ax.get_xticklabels()
        ax.set_xticklabels(labels)
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, bbox_to_anchor=(0, 0.25), loc='upper left', borderaxespad=0.5)
        
        plt.tight_layout()
        
        filename = f'matched_enhanced_contrast_ref_{ref_c}_ori_{int(ori)}.png'
        save_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(save_path, dpi=300)
        print(f"グラフ保存: {filename}")
        plt.close(fig)

print("\n--- Generating bar charts (Raw Contrast) ---")
for ref_c in unique_ref_contrasts:
    for ori in unique_orientations:
        plot_df = final_df[(final_df['Ref_Contrast'] == ref_c) & (final_df['Orientation'] == ori)]
        if plot_df.empty:
            continue
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.barplot(
            data=plot_df, 
            x='Condition', 
            y='Matched_Contrast', 
            hue='Ocularity',
            order=condition_order,
            hue_order=ocularity_order,
            errorbar=('ci', 95), 
            capsize=0.1, 
            err_kws={'linewidth': 1.5},
            ax=ax
        )
        
        ax.axhline(y=ref_c, color='red', linestyle='--', linewidth=2, label=f'Ref Contrast ({ref_c})')
        
        patch_idx = 0
        for oc in ocularity_order:
            for cond in condition_order:
                if patch_idx < len(ax.patches):
                    p = ax.patches[patch_idx]
                    height = p.get_height()
                    if pd.notna(height) and height > 0:
                        subset = plot_df[(plot_df['Ocularity'] == oc) & (plot_df['Condition'] == cond)]
                        if not subset.empty:
                            m = subset['Matched_Contrast'].mean()
                            d = subset['Matched_Contrast'].std() 
                            
                            x = p.get_x() + p.get_width() / 2
                            y = 0.05  # バーの足元付近
                            
                            ax.text(x, y, f"m={m:.2f}\nd={d:.2f}", ha='center', va='bottom', color='black', fontsize=10,
                                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                    patch_idx += 1

        ax.set_title(f'Matched Contrast by Condition and Ocularity (Ref Contrast: {ref_c}, Ori: {ori}°)', fontsize=14)
        ax.set_ylabel('Matched Contrast (Raw)', fontsize=12)
        ax.set_xlabel('Condition', fontsize=12)
        ax.set_yscale('log')
        
        ax.set_ylim(0.05, 1.0) 
        ax.set_yticks([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        
        labels = ax.get_xticklabels()
        ax.set_xticklabels(labels)
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, bbox_to_anchor=(0, 0.25), loc='upper left', borderaxespad=0.5)
        
        plt.tight_layout()
        
        filename = f'matched_raw_contrast_ref_{ref_c}_ori_{int(ori)}.png'
        save_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(save_path, dpi=300)
        print(f"グラフ保存: {filename}")
        plt.close(fig)

print("\n--- Generating bar charts (AR Extended Contrast) ---")
for ref_c in unique_ref_contrasts:
    for ori in unique_orientations:
        plot_df = final_df[(final_df['Ref_Contrast'] == ref_c) & (final_df['Orientation'] == ori)]
        if plot_df.empty:
            continue
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.barplot(
            data=plot_df, 
            x='Condition', 
            y='Matched_Contrast_AR', 
            hue='Ocularity',
            order=condition_order,
            hue_order=ocularity_order,
            errorbar=('ci', 95), 
            capsize=0.1, 
            err_kws={'linewidth': 1.5},
            ax=ax
        )
        
        ax.axhline(y=ref_c, color='red', linestyle='--', linewidth=2, label=f'Ref Contrast ({ref_c})')
        
        patch_idx = 0
        for oc in ocularity_order:
            for cond in condition_order:
                if patch_idx < len(ax.patches):
                    p = ax.patches[patch_idx]
                    height = p.get_height()
                    if pd.notna(height) and height > 0:
                        subset = plot_df[(plot_df['Ocularity'] == oc) & (plot_df['Condition'] == cond)]
                        if not subset.empty:
                            m = subset['Matched_Contrast_AR'].mean()
                            d = subset['Matched_Contrast_AR'].std() 
                            
                            x = p.get_x() + p.get_width() / 2
                            y = 0.05  # バーの足元付近
                            
                            ax.text(x, y, f"m={m:.2f}\nd={d:.2f}", ha='center', va='bottom', color='black', fontsize=10,
                                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                    patch_idx += 1
        # もしモデルが学習済みなら、Dual plane flatの右側に各モデルの予測値を1本ずつバーとしてプロット
        if models:
            pds_left = [r for r in final_df['PD_Left'].tolist() if pd.notna(r) and r > 0]
            pds_right = [r for r in final_df['PD_Right'].tolist() if pd.notna(r) and r > 0]
            avg_pd_left = sum(pds_left) / len(pds_left) if pds_left else 4.0
            avg_pd_right = sum(pds_right) / len(pds_right) if pds_right else 4.0
            avg_blur_left = calculate_blur_attenuation_cached(avg_pd_left)
            avg_blur_right = calculate_blur_attenuation_cached(avg_pd_right)

            C_val = torch.tensor(float(ref_c), dtype=torch.float32).to(device)
            M_val = torch.tensor(1.0, dtype=torch.float32).to(device)
            blur_val = torch.tensor(0.5 * (avg_blur_left + avg_blur_right), dtype=torch.float32).to(device)

            with torch.no_grad():
                gamma_A = models['ModelA'].gamma
                sigma_A = models['ModelA'].sigma
                beta_A = models['ModelA'].beta
                pred_A = torch.pow(C_val * (torch.pow(sigma_A, gamma_A) + beta_A * torch.pow(M_val, gamma_A)), 1.0 / gamma_A).item()

                gamma_B = models['ModelB'].gamma
                sigma_B = models['ModelB'].sigma
                beta_B = models['ModelB'].beta
                pred_B = torch.pow(C_val * (torch.pow(sigma_B, gamma_B) + beta_B * torch.pow(M_val * blur_val, gamma_B)), 1.0 / gamma_B).item()

                # C1: g を定数0.2
                gamma_C1 = models['ModelC1'].gamma
                sigma_C1 = models['ModelC1'].sigma
                beta_C1 = models['ModelC1'].beta
                pred_C1 = torch.pow(C_val * (torch.pow(sigma_C1, gamma_C1) + beta_C1 * torch.pow(M_val * blur_val * 0.2, gamma_C1)), 1.0 / gamma_C1).item()

                # C2: 左右Bの平均
                gamma_C2 = models['ModelC2'].gamma
                sigma_C2 = models['ModelC2'].sigma
                beta_C2 = models['ModelC2'].beta
                bl = torch.tensor(avg_blur_left, dtype=torch.float32).to(device)
                br = torch.tensor(avg_blur_right, dtype=torch.float32).to(device)
                b_left = torch.pow(C_val * (torch.pow(sigma_C2, gamma_C2) + beta_C2 * torch.pow(M_val * bl, gamma_C2)), 1.0 / gamma_C2)
                b_right = torch.pow(C_val * (torch.pow(sigma_C2, gamma_C2) + beta_C2 * torch.pow(M_val * br, gamma_C2)), 1.0 / gamma_C2)
                pred_C2 = (0.5 * (b_left + b_right)).item()

            model_preds = [pred_A, pred_B, pred_C1, pred_C2]
            model_labels = ['Model A', 'Model B', 'Model C1', 'Model C2']
            model_colors = ['#A0A0A0', '#B0B0B0', '#C0C0C0', '#D0D0D0'] # バーの色
            
            base_x = len(condition_order)
            bar_width = 0.6
            
            for i, (pred_val, label, color) in enumerate(zip(model_preds, model_labels, model_colors)):
                x_pos = base_x + i
                ax.bar(x_pos, pred_val, width=bar_width, color=color, alpha=0.8, edgecolor='black', zorder=3)
                ax.text(x_pos, 0.05, f"pred\n={pred_val:.2f}", ha='center', va='bottom', color='black', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            
            # x軸の範囲を更新
            ax.set_xlim(-0.5, base_x + len(model_labels) - 0.5)

        ax.set_title(f'Matched AR Contrast by Condition and Ocularity (Ref Contrast: {ref_c}, Ori: {ori}°)', fontsize=14)
        ax.set_ylabel('Matched Contrast (AR Extended)', fontsize=12)
        ax.set_xlabel('Condition', fontsize=12)
        ax.set_yscale('log')
        
        ax.set_ylim(0.05, 1.0) 
        ax.set_yticks([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        
        if models:
            new_xticks = list(range(len(condition_order))) + [base_x + i for i in range(len(model_labels))]
            new_xticklabels = condition_order + model_labels
            ax.set_xticks(new_xticks)
            ax.set_xticklabels(new_xticklabels, rotation=15, ha='right')
        else:
            labels = ax.get_xticklabels()
            ax.set_xticklabels(labels)
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, bbox_to_anchor=(0.78, 1.0), loc='upper left', borderaxespad=0.5)
        
        plt.tight_layout()
        
        filename = f'matched_ar_contrast_ref_{ref_c}_ori_{int(ori)}.png'
        save_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(save_path, dpi=300)
        print(f"グラフ保存: {filename}")
        plt.close(fig)

print(f"解析完了: {target_folder_name} (パス: {TARGET_DIR})")