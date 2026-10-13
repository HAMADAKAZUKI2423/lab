#!/usr/bin/env python3
r"""選定済みhigh10/low10の背景をぼかし、左右の光学加算重畳PNGを保存する。

配置: lab/src/experiment/stimuli/export_selected_blur_overlays.py
lab/srcから、例:
    py -m experiment.stimuli.export_selected_blur_overlays "..\results\fukiage-disparity-selection\選定フォルダ名"
瞳孔径が選定config.jsonにない場合（例。2.2は実測値に置き換える）:
    py -m experiment.stimuli.export_selected_blur_overlays "選定フォルダ" --pupil-diameter-mm 2.2
校正フォルダを移動した場合:
    py -m experiment.stimuli.export_selected_blur_overlays "選定フォルダ" --display-dir "校正フォルダ"

現在のexperiment_configではなく、選定時のconfig.jsonの距離・視角・IPD・解像度を使う。
瞳孔径はCLI > 選定config.jsonの順。65 mmはIPDであり、瞳孔径には使わない。
前景はぼかさず、元の背景パノラマを校正線形RGBでGaussian blur後に左右眼像へ切り出す。
既存selectorの処理を再利用するため、ページの現行select_fukiage_disparity_pairs.pyが必要。
再選定・モデル読込・視認性再計算・元画像/CSV更新はしない。GPUは不要。

既定出力: 選定フォルダ/blur_overlay_check/実行日時/
各high10/low10/rank_XXに、左右のぼかし重畳像・ぼかしなし重畳像・背景像・比較画像を保存。
comparison_no_blur_top_blur_bottom.pngは左列=右眼、右列=左眼、上段=ぼかしなし、下段=ぼかしあり。

重畳はselectorと同じXYZ光学加算であり、alphaブレンドではない。
PNGは等価前景ディスプレイRGBへ変換して色域外をクリップした確認用画像。
全眼光学系/両眼融合の完全再現ではない。校正ファイルは読み出した時点のものを使用し、
選定時から変更されていないことまでは保証しない。入力/校正のSHA256を記録する。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import MISSING, asdict, fields
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from experiment.common.display_calibration import load_display_calibration
from experiment.stimuli.select_fukiage_disparity_pairs import (
    DEFOCUS_BLUR_SCALE_FACTOR,
    RunSettings,
    calculate_defocus_parameters,
    make_eye_backgrounds,
    optical_addition_bgr,
    prepare_background_panorama,
    prepare_foreground,
    prepare_scoring_backgrounds,
    require_calibration,
    right_aligned_disparity_deg,
)

CALIBRATION_FILES = ("C.csv", "T_prime.csv", "R_prime.csv", "gamma_bg.csv", "gamma_fg.csv")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_pairs(directory: Path) -> list[dict]:
    rows = []
    required = {"difference_group", "group_rank", "foreground_path", "background_path"}
    for group in ("high", "low"):
        path = directory / f"{group}10.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{path.name}の必須列: {sorted(required)}")
            part = list(reader)
        if len(part) != 10 or any(row["difference_group"] != group for row in part):
            raise ValueError(f"{path.name}は{group}の10組である必要があります")
        ranks = [int(row["group_rank"]) for row in part]
        if sorted(ranks) != list(range(1, 11)):
            raise ValueError(f"{path.name}のgroup_rankは重複なしの1..10が必要です")
        rows.extend(sorted(part, key=lambda row: int(row["group_rank"])))
    return rows


def original_path(directory: Path, row: dict, role: str) -> Path:
    pair_dir = directory / f"{row['difference_group']}10" / f"rank_{int(row['group_rank']):02d}"
    saved = sorted(path for path in pair_dir.glob(f"{role}_original.*")
                   if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if len(saved) > 1:
        raise ValueError(f"元画像が複数あります: {saved}")
    path = saved[0] if saved else Path(row[f"{role}_path"]).expanduser()
    if not path.is_absolute():
        path = directory / path
    if not path.is_file():
        raise FileNotFoundError(f"元の{role}画像がありません: {path}")
    return path.resolve()


def build_settings(config: dict, pupil_override: float | None) -> RunSettings:
    missing = [field.name for field in fields(RunSettings)
               if field.default is MISSING and field.default_factory is MISSING
               and field.name not in config]
    if missing:
        raise ValueError(f"選定config.jsonの必須設定がありません: {missing}")
    values = {field.name: config[field.name] for field in fields(RunSettings) if field.name in config}
    fg, bg, angle, ipd = (float(config[key]) for key in
                         ("distance_fg_cm", "distance_bg_cm", "visual_angle_deg", "ipd_mm"))
    size_value = float(config["image_size_px"])
    if not all(math.isfinite(x) for x in (fg, bg, angle, ipd, size_value)):
        raise ValueError("距離・視角・IPD・解像度には有限な値が必要です")
    if not (0 < fg < bg and 0 < angle < 90 and ipd > 0 and size_value >= 1 and size_value.is_integer()):
        raise ValueError("選定config.jsonの距離・視角・IPD・解像度が不正です")
    pupil = pupil_override if pupil_override is not None else config.get("pupil_diameter_mm")
    if pupil is None:
        raise ValueError("瞳孔径の記録がありません。--pupil-diameter-mmで明示指定してください（IPDとは別）")
    pupil = float(pupil)
    if not math.isfinite(pupil) or not 0 < pupil <= 10:
        raise ValueError("瞳孔径は0より大きく10 mm以下の有限値です。65 mmはIPDです")
    scale = float(config.get("blur_scale_factor", DEFOCUS_BLUR_SCALE_FACTOR))
    if not math.isclose(scale, DEFOCUS_BLUR_SCALE_FACTOR, rel_tol=0, abs_tol=1e-12):
        raise ValueError("選定時と現行common.opticsのブラー係数が異なります。設定を確認してください")
    truncate = float(config.get("gaussian_truncate", 4.0))
    if not math.isfinite(truncate) or truncate <= 0:
        raise ValueError("gaussian_truncateには正の有限値が必要です")
    size = int(size_value)
    diopter = abs(100.0 / fg - 100.0 / bg)
    diameter, sigma_deg = calculate_defocus_parameters(diopter, pupil)
    disparity_deg = right_aligned_disparity_deg(ipd, fg, bg)
    values.update(distance_fg_cm=fg, distance_bg_cm=bg, visual_angle_deg=angle,
                  ipd_mm=ipd, image_size_px=size, device="cpu", apply_defocus=True,
                  pupil_diameter_mm=pupil, diopter_difference=diopter,
                  blur_diameter_deg=diameter, blur_sigma_deg=sigma_deg,
                  blur_sigma_px=sigma_deg * size / angle,
                  disparity_deg=disparity_deg, disparity_px=round(size * disparity_deg / angle),
                  blur_scale_factor=scale, gaussian_truncate=truncate)
    return RunSettings(**values)


def load_strict_calibration(directory: Path):
    # C.csvの欠損/破損時に既存ローダーが既定行列へ逃げることを防ぐ。
    matrices = {}
    for name in CALIBRATION_FILES:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"表示校正ファイルがありません: {path}")
        if name in {"C.csv", "T_prime.csv", "R_prime.csv"}:
            matrix = np.loadtxt(path, delimiter=",")
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                raise ValueError(f"校正行列は有限な3x3である必要があります: {path}")
            matrices[name] = matrix
    calibration = load_display_calibration(directory)
    require_calibration(calibration)
    if not np.array_equal(calibration.color_matrix, matrices["C.csv"]):
        raise ValueError("C.csvが正常に読み込まれていません")
    for gamma in (calibration.gamma_bg, calibration.gamma_fg):
        if gamma is None or any(not math.isfinite(float(gamma[c])) or float(gamma[c]) <= 0 for c in "RGB"):
            raise ValueError("校正ガンマRGBは正の有限値である必要があります")
    if not np.allclose(calibration.r_prime @ calibration.r_prime_inv, np.eye(3), rtol=1e-6, atol=1e-8):
        raise ValueError("R_primeの逆行列が不整合です")
    return calibration


def save_png(path: Path, image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3 or not np.isfinite(image).all():
        raise ValueError(f"保存画像が不正です: {path}")
    # selectorの保存方法（8bit変換時は切捨て）を維持する。
    pixels = np.clip(image * 255, 0, 255).astype(np.uint8)
    # 日本語のWindowsパスにも対応する。
    ok, encoded = cv2.imencode(".png", pixels)
    if not ok:
        raise OSError(f"PNGエンコードに失敗しました: {path}")
    encoded.tofile(str(path))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result_dir", type=Path, help="high10.csv、low10.csv、config.jsonがある選定フォルダ")
    parser.add_argument("--pupil-diameter-mm", type=float, help="左右共通瞳孔径。省略時は選定config.jsonの記録値")
    parser.add_argument("--display-dir", type=Path, help="校正フォルダの明示指定。省略時は選定config.jsonのdisplay_dir")
    parser.add_argument("--output-dir", type=Path, help="新規出力フォルダ。既存フォルダは上書きしない")
    args = parser.parse_args(argv)
    source = args.result_dir.expanduser().resolve()
    config_path = source / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    settings = build_settings(config, args.pupil_diameter_mm)
    rows = read_pairs(source)
    inputs = [(row, original_path(source, row, "foreground"), original_path(source, row, "background"))
              for row in rows]
    display_value = args.display_dir if args.display_dir is not None else config.get("display_dir")
    if not display_value:
        raise ValueError("校正フォルダの記録がありません。--display-dirで明示指定してください")
    display_dir = Path(display_value).expanduser()
    if not display_dir.is_absolute():
        display_dir = source / display_dir
    display_dir = display_dir.resolve()
    calibration_hashes = {name: sha256(display_dir / name) for name in CALIBRATION_FILES}
    calibration = load_strict_calibration(display_dir)
    source_hashes = {name: sha256(source / name) for name in ("config.json", "high10.csv", "low10.csv")}
    input_hashes = {str(path): sha256(path) for _, fg, bg in inputs for path in (fg, bg)}
    output = (args.output_dir.expanduser().resolve() if args.output_dir is not None else
              source / "blur_overlay_check" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"source_result_dir": str(source), "source_config": config,
                "simulation_settings": asdict(settings), "display_dir": str(display_dir),
                "calibration_sha256": calibration_hashes, "source_sha256": source_hashes,
                "input_sha256": input_hashes, "eye_boundary_mode": settings.eye_boundary_mode,
                "blur_method": "gaussian_linear_rgb", "blur_boundary_mode": "reflect101",
                "composite_method": "calibrated_XYZ_addition_to_equivalent_FG_RGB",
                "gamut_policy": "clip_linear_RGB_0_1", "PNG_quantization": "uint8_floor",
                "source_calibration_version_verified": False,
                "model_evaluation_performed": False, "reselection_performed": False,
                "pupil_source": "CLI" if args.pupil_diameter_mm is not None else "selection_config",
                "comparison_layout": "columns:right_eye,left_eye; rows:no_blur,blur",
                "no_blur_comparison_boundary": "same_reflect_as_blurred_simulation",
                "note": "ブラー追加/変更後の画像です。保存済みhigh/lowスコアの順位保証や再評価は行いません",
                "complete": False}
    write_json(output / "simulation_config.json", metadata)
    print(f"Source: {source}\nOutput: {output}", flush=True)
    print(f"Pupil={settings.pupil_diameter_mm:g} mm, IPD={settings.ipd_mm:g} mm, "
          f"sigma={settings.blur_sigma_px:.4f} px, disparity={settings.disparity_px} px", flush=True)
    print("WARN: 校正ファイルは現在の内容を使用。選定時の版と同一かは未確認です。", flush=True)
    report = []
    for index, (row, fg_path, bg_path) in enumerate(inputs, start=1):
        pair_dir = output / f"{row['difference_group']}10" / f"rank_{int(row['group_rank']):02d}"
        pair_dir.mkdir(parents=True, exist_ok=False)
        foreground = prepare_foreground(fg_path, settings.image_size_px)
        panorama = prepare_background_panorama(bg_path, settings.image_size_px)
        blurred_views = prepare_scoring_backgrounds(panorama, calibration, settings)
        clear_views = make_eye_backgrounds(panorama, settings.image_size_px, settings.disparity_px,
                                         boundary_mode=settings.eye_boundary_mode)
        record = {"difference_group": row["difference_group"], "group_rank": int(row["group_rank"]),
                  "source_rank": row.get("rank", ""), "foreground_path": str(fg_path),
                  "background_path": str(bg_path), "pupil_diameter_mm": settings.pupil_diameter_mm,
                  "blur_sigma_px": settings.blur_sigma_px, "disparity_px": settings.disparity_px,
                  "pair_output_dir": str(pair_dir)}
        composites = {}
        save_png(pair_dir / "foreground_prepared.png", foreground)
        for eye, blurred_bg, clear_bg in zip(("right", "left"), blurred_views, clear_views):
            composite, clip_ratio = optical_addition_bgr(foreground, blurred_bg, calibration)
            clear_composite, clear_clip_ratio = optical_addition_bgr(foreground, clear_bg, calibration)
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in (clip_ratio, clear_clip_ratio)):
                raise ValueError(f"{pair_dir.name}: クリップ率が不正です")
            save_png(pair_dir / f"{eye}_background_view_blurred.png", blurred_bg)
            save_png(pair_dir / f"{eye}_background_view_unblurred.png", clear_bg)
            save_png(pair_dir / f"{eye}_optical_composite.png", composite)
            save_png(pair_dir / f"{eye}_optical_composite_unblurred.png", clear_composite)
            composites[eye] = (clear_composite, composite)
            record[f"{eye}_clip_ratio_blurred"] = clip_ratio
            record[f"{eye}_clip_ratio_unblurred"] = clear_clip_ratio
        comparison = np.concatenate((np.concatenate((composites["right"][0], composites["left"][0]), axis=1),
                                     np.concatenate((composites["right"][1], composites["left"][1]), axis=1)), axis=0)
        save_png(pair_dir / "comparison_no_blur_top_blur_bottom.png", comparison)
        write_json(pair_dir / "pair_metadata.json", {**record, "source_selection_row": row})
        report.append(record)
        print(f"[{index}/20] {row['difference_group']} rank {int(row['group_rank']):02d}: saved", flush=True)
    with (output / "simulation_manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    # 処理中の入力/校正変更も検出。部分出力ではcomplete=Falseのまま。
    for name, expected in calibration_hashes.items():
        if sha256(display_dir / name) != expected:
            raise RuntimeError(f"処理中に校正ファイルが変更されました: {name}")
    for name, expected in source_hashes.items():
        if sha256(source / name) != expected:
            raise RuntimeError(f"処理中に選定ファイルが変更されました: {name}")
    for path, expected in input_hashes.items():
        if sha256(Path(path)) != expected:
            raise RuntimeError(f"処理中に元画像が変更されました: {path}")
    metadata.update(complete=True, saved_pair_count=len(report), finished_at=datetime.now().isoformat())
    write_json(output / "simulation_config.json", metadata)
    print(f"Saved all {len(report)} pairs: {output}", flush=True)


if __name__ == "__main__":
    main()