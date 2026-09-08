#!/usr/bin/env python3
"""Fukiageモデルで左右視認性差が大きいFG/BG画像ペアを抽出する。

実行例（srcから）:
    python -m experiment.stimuli.select_fukiage_disparity_pairs --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import torch

from experiment import experiment_config
from experiment.common.display_calibration import DisplayCalibration, load_display_calibration
from experiment.pre_experiment.image.config import create_image_config
from experiment.pre_experiment.image.stimuli import discover_images

SCRIPT_PATH = Path(__file__).resolve()
LAB_ROOT = SCRIPT_PATH.parents[3]
VISIBILITY_REPO = LAB_ROOT / "visibility_blend_2025-main"
DEFAULT_OUTPUT_ROOT = LAB_ROOT / "results" / "fukiage-disparity-selection"


@dataclass(frozen=True)
class RunSettings:
    ipd_mm: float
    distance_fg_cm: float
    distance_bg_cm: float
    visual_angle_deg: float
    nominal_l_fg: float
    nominal_l_bg: float
    image_size_px: int
    disparity_deg: float
    disparity_px: int
    alpha: float
    model: str
    device: str
    top_k: int
    apply_defocus: bool = False


@dataclass
class PairResult:
    foreground_path: str
    background_path: str
    right_visibility_score: float
    left_visibility_score: float
    signed_score_difference: float
    absolute_score_difference: float
    right_clip_ratio: float
    left_clip_ratio: float
    disparity_deg: float
    disparity_px: int
    rank: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipd-mm", type=float, default=60.0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--model", default="vismlp_norm")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--foreground-dir", type=Path)
    parser.add_argument("--background-dir", type=Path)
    return parser.parse_args()


def right_aligned_disparity_deg(ipd_mm: float, fg_cm: float, bg_cm: float) -> float:
    ipd_cm = ipd_mm / 10.0
    return math.degrees(math.atan2(ipd_cm, fg_cm) - math.atan2(ipd_cm, bg_cm))


def load_settings(args: argparse.Namespace) -> RunSettings:
    runtime = experiment_config.get_config() or {}
    required = ("DISTANCE_FG", "DISTANCE_BG", "VISUAL_ANGLE_DEG", "L_fg", "L_bg")
    missing = [key for key in required if key not in runtime]
    if missing:
        raise KeyError(f"experiment_config.pyに必要なキーがありません: {missing}")
    fg_cm = float(runtime["DISTANCE_FG"])
    bg_cm = float(runtime["DISTANCE_BG"])
    angle = float(runtime["VISUAL_ANGLE_DEG"])
    l_fg = float(runtime["L_fg"])
    l_bg = float(runtime["L_bg"])
    if fg_cm >= bg_cm:
        raise ValueError("DISTANCE_FGはDISTANCE_BGより小さくしてください")
    disparity_deg = right_aligned_disparity_deg(args.ipd_mm, fg_cm, bg_cm)
    disparity_px = round(args.image_size * disparity_deg / angle)
    return RunSettings(
        ipd_mm=args.ipd_mm,
        distance_fg_cm=fg_cm,
        distance_bg_cm=bg_cm,
        visual_angle_deg=angle,
        nominal_l_fg=l_fg,
        nominal_l_bg=l_bg,
        image_size_px=args.image_size,
        disparity_deg=disparity_deg,
        disparity_px=disparity_px,
        alpha=l_fg / (l_fg + l_bg),
        model=args.model,
        device=args.device,
        top_k=args.top_k,
    )


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image.astype(np.float32) / 255.0


def prepare_foreground(path: Path, size: int) -> np.ndarray:
    return cv2.resize(read_bgr(path), (size, size), interpolation=cv2.INTER_AREA)


def prepare_background_panorama(path: Path, size: int) -> np.ndarray:
    """現行Image実験と同じ中央横長領域を、2視野幅の背景として準備する。"""
    square = cv2.resize(read_bgr(path), (512, 512), interpolation=cv2.INTER_AREA)
    strip = square[128:384, :, :]
    return cv2.resize(strip, (size * 2, size), interpolation=cv2.INTER_AREA)


def crop_with_black(panorama: np.ndarray, start_x: int, width: int) -> np.ndarray:
    height, panorama_width = panorama.shape[:2]
    output = np.zeros((height, width, 3), dtype=np.float32)
    src_l = max(0, start_x)
    src_r = min(panorama_width, start_x + width)
    if src_r > src_l:
        dst_l = src_l - start_x
        output[:, dst_l:dst_l + src_r - src_l] = panorama[:, src_l:src_r]
    return output


def make_eye_backgrounds(
    panorama: np.ndarray, output_width: int, disparity_px: int
) -> tuple[np.ndarray, np.ndarray]:
    right_start = (panorama.shape[1] - output_width) // 2
    # 右側をサンプリングすると、左眼画像中の背景内容は左へ移動する。
    left_start = right_start + disparity_px
    return (
        crop_with_black(panorama, right_start, output_width),
        crop_with_black(panorama, left_start, output_width),
    )


def apply_gamma(rgb: np.ndarray, gamma: dict[str, float]) -> np.ndarray:
    out = np.empty_like(rgb, dtype=np.float64)
    for index, channel in enumerate("RGB"):
        out[..., index] = np.clip(rgb[..., index], 0, 1) ** float(gamma[channel])
    return out


def invert_gamma(linear_rgb: np.ndarray, gamma: dict[str, float]) -> np.ndarray:
    out = np.empty_like(linear_rgb, dtype=np.float64)
    for index, channel in enumerate("RGB"):
        out[..., index] = np.clip(linear_rgb[..., index], 0, None) ** (
            1.0 / float(gamma[channel])
        )
    return out


def require_calibration(calibration: DisplayCalibration) -> None:
    if calibration.gamma_bg is None or calibration.gamma_fg is None:
        raise RuntimeError("gamma_bg.csvとgamma_fg.csvが必要です")
    for name in ("color_matrix", "t_prime", "r_prime", "r_prime_inv"):
        matrix = np.asarray(getattr(calibration, name))
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise RuntimeError(f"表示校正行列が不正です: {name}")


def calibrated_model_components(
    foreground_bgr: np.ndarray,
    background_bgr: np.ndarray,
    calibration: DisplayCalibration,
) -> tuple[np.ndarray, np.ndarray]:
    """FGと透過BGを共通の等価前景RGB表現へ変換する。"""
    assert calibration.gamma_bg is not None
    assert calibration.gamma_fg is not None
    fg_rgb = foreground_bgr[..., ::-1]
    bg_rgb = background_bgr[..., ::-1]
    linear_fg_ref = apply_gamma(fg_rgb, calibration.gamma_bg)
    linear_fg = np.clip(linear_fg_ref @ calibration.color_matrix.T, 0, None)
    linear_bg = apply_gamma(bg_rgb, calibration.gamma_bg)
    equivalent_bg = (linear_bg @ calibration.t_prime.T) @ calibration.r_prime_inv.T
    target_rgb = invert_gamma(linear_fg, calibration.gamma_fg)
    reference_rgb = invert_gamma(equivalent_bg, calibration.gamma_fg)
    return (
        np.clip(target_rgb[..., ::-1], 0, 1).astype(np.float32),
        np.clip(reference_rgb[..., ::-1], 0, 1).astype(np.float32),
    )


def optical_addition_bgr(
    foreground_bgr: np.ndarray,
    background_bgr: np.ndarray,
    calibration: DisplayCalibration,
) -> tuple[np.ndarray, float]:
    """XYZ_sum=T'd_bg+R'd_fgを計算し、等価前景BGR画像へ戻す。"""
    assert calibration.gamma_bg is not None
    assert calibration.gamma_fg is not None
    fg_rgb = foreground_bgr[..., ::-1]
    bg_rgb = background_bgr[..., ::-1]
    linear_bg = apply_gamma(bg_rgb, calibration.gamma_bg)
    linear_fg_ref = apply_gamma(fg_rgb, calibration.gamma_bg)
    linear_fg = np.clip(linear_fg_ref @ calibration.color_matrix.T, 0, None)
    xyz_sum = linear_bg @ calibration.t_prime.T + linear_fg @ calibration.r_prime.T
    equivalent = xyz_sum @ calibration.r_prime_inv.T
    clipped = (equivalent < 0) | (equivalent > 1)
    clip_ratio = float(np.mean(np.any(clipped, axis=2)))
    encoded_rgb = invert_gamma(np.clip(equivalent, 0, 1), calibration.gamma_fg)
    return np.clip(encoded_rgb[..., ::-1], 0, 1).astype(np.float32), clip_ratio


def to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    chw = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)
    return torch.from_numpy(chw).unsqueeze(0).to(device)


def load_visibility_model(model_name: str, device: torch.device):
    if not VISIBILITY_REPO.is_dir():
        raise FileNotFoundError(f"visibility_blend_2025-mainがありません: {VISIBILITY_REPO}")
    sys.path.insert(0, str(VISIBILITY_REPO))
    from vismodel.utils import load_vismodel  # type: ignore
    model = load_vismodel(model_name, device, load_param=True)
    model.eval()
    return model


def compute_visibility_map(
    model, target_bgr: np.ndarray, reference_bgr: np.ndarray,
    alpha: float, device: torch.device,
) -> np.ndarray:
    height, width = target_bgr.shape[:2]
    alpha_map = np.full((height, width, 1), alpha, dtype=np.float32)
    full_mask = np.ones((height, width, 1), dtype=np.float32)
    with torch.no_grad():
        model.set_inputs_tg_ref_alphamap(
            to_tensor(target_bgr, device),
            to_tensor(reference_bgr, device),
            to_tensor(alpha_map, device),
            to_tensor(full_mask, device),
            blend_mode="linear",
        )
        model.compute_weights()
        model.compute_visibility_wo_weight()
    return model.norm_vismap.squeeze().detach().cpu().numpy().astype(np.float32)


def evaluate_pair(
    fg_path: Path, bg_path: Path, foreground: np.ndarray,
    right_bg: np.ndarray, left_bg: np.ndarray,
    calibration: DisplayCalibration, model, settings: RunSettings,
    device: torch.device,
) -> PairResult:
    right_target, right_reference = calibrated_model_components(
        foreground, right_bg, calibration
    )
    left_target, left_reference = calibrated_model_components(
        foreground, left_bg, calibration
    )
    right_map = compute_visibility_map(
        model, right_target, right_reference, settings.alpha, device
    )
    left_map = compute_visibility_map(
        model, left_target, left_reference, settings.alpha, device
    )
    _, right_clip = optical_addition_bgr(foreground, right_bg, calibration)
    _, left_clip = optical_addition_bgr(foreground, left_bg, calibration)
    right_score = float(right_map.mean())
    left_score = float(left_map.mean())
    signed = left_score - right_score
    return PairResult(
        str(fg_path), str(bg_path), right_score, left_score,
        signed, abs(signed), right_clip, left_clip,
        settings.disparity_deg, settings.disparity_px,
    )


def save_bgr(path: Path, image: np.ndarray) -> None:
    cv2.imwrite(str(path), np.clip(image * 255, 0, 255).astype(np.uint8))


def save_vismap(path: Path, vismap: np.ndarray, heatmap: bool = False) -> None:
    gray = np.clip(vismap * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(
        str(path), cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS) if heatmap else gray
    )


def write_csv(path: Path, rows: list[PairResult]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def save_top_pair(
    output_dir: Path, result: PairResult, foreground: np.ndarray,
    panorama: np.ndarray, calibration: DisplayCalibration, model,
    settings: RunSettings, device: torch.device,
) -> None:
    assert result.rank is not None
    pair_dir = output_dir / "top20" / f"rank_{result.rank:02d}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    right_bg, left_bg = make_eye_backgrounds(
        panorama, settings.image_size_px, settings.disparity_px
    )
    right_target, right_ref = calibrated_model_components(foreground, right_bg, calibration)
    left_target, left_ref = calibrated_model_components(foreground, left_bg, calibration)
    right_map = compute_visibility_map(model, right_target, right_ref, settings.alpha, device)
    left_map = compute_visibility_map(model, left_target, left_ref, settings.alpha, device)
    right_composite, _ = optical_addition_bgr(foreground, right_bg, calibration)
    left_composite, _ = optical_addition_bgr(foreground, left_bg, calibration)

    fg_source = Path(result.foreground_path)
    bg_source = Path(result.background_path)
    shutil.copy2(fg_source, pair_dir / f"foreground_original{fg_source.suffix}")
    shutil.copy2(bg_source, pair_dir / f"background_original{bg_source.suffix}")
    save_bgr(pair_dir / "foreground_prepared.png", foreground)
    save_bgr(pair_dir / "right_background_view.png", right_bg)
    save_bgr(pair_dir / "left_background_view.png", left_bg)
    save_bgr(pair_dir / "right_optical_composite.png", right_composite)
    save_bgr(pair_dir / "left_optical_composite.png", left_composite)
    save_vismap(pair_dir / "right_vismap_gray.png", right_map)
    save_vismap(pair_dir / "left_vismap_gray.png", left_map)
    save_vismap(pair_dir / "right_vismap_heat.png", right_map, True)
    save_vismap(pair_dir / "left_vismap_heat.png", left_map, True)


def main() -> None:
    args = parse_args()
    settings = load_settings(args)
    image_config = create_image_config()
    fg_dir = args.foreground_dir or image_config.foreground_image_dir
    bg_dir = args.background_dir or image_config.background_image_dir
    fg_paths = discover_images(fg_dir)
    bg_paths = discover_images(bg_dir)
    if not fg_paths or not bg_paths:
        raise FileNotFoundError(f"画像がありません: FG={fg_dir}, BG={bg_dir}")

    output_dir = args.output_dir or (
        DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    calibration = load_display_calibration(image_config.display_dir)
    require_calibration(calibration)
    device = torch.device(settings.device)
    model = load_visibility_model(settings.model, device)

    fg_cache = {path: prepare_foreground(path, settings.image_size_px) for path in fg_paths}
    panorama_cache = {
        path: prepare_background_panorama(path, settings.image_size_px) for path in bg_paths
    }
    eye_bg_cache = {
        path: make_eye_backgrounds(
            panorama_cache[path], settings.image_size_px, settings.disparity_px
        ) for path in bg_paths
    }

    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump({
            **asdict(settings),
            "foreground_dir": str(fg_dir),
            "background_dir": str(bg_dir),
            "display_dir": str(image_config.display_dir),
            "visibility_repo": str(VISIBILITY_REPO),
        }, file, ensure_ascii=False, indent=2)

    pairs = list(product(fg_paths, bg_paths))
    results: list[PairResult] = []
    print(
        f"FG={settings.distance_fg_cm:g}cm, BG={settings.distance_bg_cm:g}cm, "
        f"IPD={settings.ipd_mm:g}mm, disparity={settings.disparity_deg:.4f}deg "
        f"({settings.disparity_px}px), pairs={len(pairs)}"
    )
    for index, (fg_path, bg_path) in enumerate(pairs, 1):
        right_bg, left_bg = eye_bg_cache[bg_path]
        result = evaluate_pair(
            fg_path, bg_path, fg_cache[fg_path], right_bg, left_bg,
            calibration, model, settings, device,
        )
        results.append(result)
        print(f"[{index}/{len(pairs)}] {fg_path.name} x {bg_path.name}: {result.absolute_score_difference:.6f}")

    results.sort(key=lambda row: row.absolute_score_difference, reverse=True)
    for rank, result in enumerate(results, 1):
        result.rank = rank
    top_results = results[:min(settings.top_k, len(results))]
    write_csv(output_dir / "all_pairs.csv", results)
    write_csv(output_dir / "top20.csv", top_results)

    for result in top_results:
        fg_path = Path(result.foreground_path)
        bg_path = Path(result.background_path)
        save_top_pair(
            output_dir, result, fg_cache[fg_path], panorama_cache[bg_path],
            calibration, model, settings, device,
        )
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()