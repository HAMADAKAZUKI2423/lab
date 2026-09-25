#!/usr/bin/env python3
"""事前K-means抽出後、Fukiageモデルで左右視認性差が大きい画像ペアを選ぶ。

全画像のラプラシアンピラミッド特徴量をK-meansで5クラスへ分類し、
平均相対輝度は昇順に並べて等数5分割する。両軸のtop/bottomが交差する
4カテゴリから50枚ずつを
再現可能な乱数で抽出する。その200枚を背景候補とし、うち低輝度側の
100枚だけを前景候補として視認性評価する。

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

SCRIPT_PATH = Path(__file__).resolve()
LAB_ROOT = SCRIPT_PATH.parents[3]
VISIBILITY_REPO = LAB_ROOT / "visibility_blend_2025-main"
DEFAULT_OUTPUT_ROOT = LAB_ROOT / "results" / "fukiage-disparity-selection"
RAW_IMAGE_DIR = LAB_ROOT / "data" / "raw" / "images" 
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
TEXTURE_CLASS_COUNT = 3
LUMINANCE_CLASS_COUNT = 3
LAPLACIAN_LEVELS = 3
KMEANS_RANDOM_STATE = 42
SAMPLES_PER_CATEGORY = 50
SAMPLING_RANDOM_SEED = 42
CORNER_CATEGORIES = (
    "texture_top__luminance_top",
    "texture_top__luminance_bottom",
    "texture_bottom__luminance_top",
    "texture_bottom__luminance_bottom",
)


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
    texture_class_count: int
    luminance_class_count: int
    laplacian_levels: int
    kmeans_random_state: int
    samples_per_category: int
    sampling_random_seed: int
    apply_defocus: bool = False


@dataclass
class PairResult:
    foreground_path: str
    background_path: str
    foreground_category: str
    background_category: str
    right_visibility_score: float
    left_visibility_score: float
    signed_score_difference: float
    absolute_score_difference: float
    right_clip_ratio: float
    left_clip_ratio: float
    disparity_deg: float
    disparity_px: int
    rank: int | None = None


@dataclass(frozen=True)
class ImageClusterItem:
    path: Path
    mean_relative_luminance: float
    laplacian_features: tuple[float, ...]
    texture_cluster_label: int
    texture_rank: int
    texture_extreme: str
    luminance_class: int
    luminance_rank: int
    luminance_extreme: str
    corner_category: str
    foreground_candidate: bool
    background_candidate: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipd-mm", type=float, default=60.0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--model", default="vismlp_norm")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--image-dir", type=Path, default=RAW_IMAGE_DIR)
    parser.add_argument(
        "--samples-per-category", type=int, default=SAMPLES_PER_CATEGORY,
        help="4カテゴリそれぞれから抽出する画像数（既定: 50）",
    )
    parser.add_argument(
        "--sampling-seed", type=int, default=SAMPLING_RANDOM_SEED,
        help="カテゴリ内ランダム抽出の乱数シード（既定: 42）",
    )
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
    if args.samples_per_category < 1:
        raise ValueError("--samples-per-categoryは1以上にしてください")
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
        texture_class_count=TEXTURE_CLASS_COUNT,
        luminance_class_count=LUMINANCE_CLASS_COUNT,
        laplacian_levels=LAPLACIAN_LEVELS,
        kmeans_random_state=KMEANS_RANDOM_STATE,
        samples_per_category=args.samples_per_category,
        sampling_random_seed=args.sampling_seed,
    )


def discover_images_recursive(directory: Path) -> list[Path]:
    """data/raw/images以下の対応画像をサブフォルダも含めて列挙する。"""
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image.astype(np.float32) / 255.0


def mean_relative_luminance_from_bgr(bgr: np.ndarray) -> float:
    """sRGBを線形化し、Rec.709係数による平均相対輝度Yを返す。"""
    rgb = bgr[..., ::-1].astype(np.float64)
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    relative_y = (
        0.2126 * linear[..., 0]
        + 0.7152 * linear[..., 1]
        + 0.0722 * linear[..., 2]
    )
    return float(np.mean(relative_y))


def compute_laplacian_mad(bgr: np.ndarray, levels: int) -> np.ndarray:
    """ラプラシアンピラミッド各段と最終低周波残差のMADを返す。"""
    gray = cv2.cvtColor(
        np.clip(bgr * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY
    ).astype(np.float32)
    features: list[float] = []
    current = gray
    for _ in range(levels):
        rows, cols = current.shape
        if rows < 2 or cols < 2:
            raise ValueError("画像が小さすぎて指定段数のラプラシアン特徴を計算できません")
        down = cv2.pyrDown(current)
        up = cv2.pyrUp(down, dstsize=(cols, rows))
        features.append(float(np.mean(np.abs(current - up))))
        current = down
    features.append(float(np.mean(np.abs(current - np.mean(current)))))
    return np.asarray(features, dtype=np.float64)


def _rank_clusters(center_scores: np.ndarray) -> dict[int, int]:
    """クラスタ中心を小さい順に1..Kへ順位付けする。"""
    order = np.argsort(np.asarray(center_scores), kind="stable")
    return {int(label): rank for rank, label in enumerate(order, start=1)}


def _extreme_name(rank: int, class_count: int) -> str:
    if rank == 1:
        return "bottom"
    if rank == class_count:
        return "top"
    return "middle"


def classify_image_extremes(
    paths: list[Path],
    texture_class_count: int,
    luminance_class_count: int,
    laplacian_levels: int,
    random_state: int,
) -> list[ImageClusterItem]:
    """テクスチャはK-means、輝度は等数5分割し、四隅を抽出する。"""
    required = max(texture_class_count, luminance_class_count)
    if len(paths) < required:
        raise ValueError(
            f"K-means分類には少なくとも{required}枚必要です: {len(paths)}枚"
        )
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise RuntimeError(
            'scikit-learnが必要です。python -m pip install "scikit-learn" を実行してください'
        ) from exc

    measured: list[tuple[Path, float, np.ndarray]] = []
    for path in paths:
        bgr = read_bgr(path)
        measured.append(
            (
                path,
                mean_relative_luminance_from_bgr(bgr),
                compute_laplacian_mad(bgr, laplacian_levels),
            )
        )

    texture_x = np.vstack([features for _, _, features in measured])
    texture_model = KMeans(
        n_clusters=texture_class_count, random_state=random_state, n_init=10
    )
    texture_labels = texture_model.fit_predict(texture_x)
    if len(np.unique(texture_labels)) != texture_class_count:
        raise ValueError("ラプラシアンK-meansで5個の異なるクラスタを作れませんでした")
    texture_rank_map = _rank_clusters(texture_model.cluster_centers_.sum(axis=1))

    # 輝度はK-meansを使わず、値とパスで安定ソートして等数に5分割する。
    luminance_order = sorted(
        range(len(measured)),
        key=lambda index: (
            measured[index][1], str(measured[index][0]).casefold()
        ),
    )
    luminance_ranks = np.empty(len(measured), dtype=np.int32)
    for class_rank, index_group in enumerate(
        np.array_split(np.asarray(luminance_order, dtype=np.int64), luminance_class_count),
        start=1,
    ):
        luminance_ranks[index_group] = class_rank

    items: list[ImageClusterItem] = []
    for index, (path, luminance, features) in enumerate(measured):
        texture_label = int(texture_labels[index])
        texture_rank = texture_rank_map[texture_label]
        luminance_rank = int(luminance_ranks[index])
        texture_extreme = _extreme_name(texture_rank, texture_class_count)
        luminance_extreme = _extreme_name(luminance_rank, luminance_class_count)
        is_corner = texture_extreme != "middle" and luminance_extreme != "middle"
        category = (
            f"texture_{texture_extreme}__luminance_{luminance_extreme}"
            if is_corner else ""
        )
        items.append(
            ImageClusterItem(
                path=path,
                mean_relative_luminance=luminance,
                laplacian_features=tuple(float(value) for value in features),
                texture_cluster_label=texture_label,
                texture_rank=texture_rank,
                texture_extreme=texture_extreme,
                luminance_class=luminance_rank,
                luminance_rank=luminance_rank,
                luminance_extreme=luminance_extreme,
                corner_category=category,
                foreground_candidate=is_corner and luminance_extreme == "bottom",
                background_candidate=is_corner,
            )
        )
    return items


def sample_corner_categories(
    items: list[ImageClusterItem],
    samples_per_category: int,
    random_seed: int,
) -> list[ImageClusterItem]:
    """4カテゴリから同数を非復元抽出する。カテゴリ不足時は実行しない。"""
    if samples_per_category < 1:
        raise ValueError("カテゴリごとの抽出数は1以上にしてください")
    rng = np.random.default_rng(random_seed)
    selected: list[ImageClusterItem] = []
    for category in CORNER_CATEGORIES:
        candidates = sorted(
            (item for item in items if item.corner_category == category),
            key=lambda item: str(item.path).casefold(),
        )
        if len(candidates) < samples_per_category:
            raise ValueError(
                f"カテゴリ {category} は{len(candidates)}枚しかありません。"
                f"{samples_per_category}枚の非復元抽出はできません。"
            )
        indices = rng.choice(
            len(candidates), size=samples_per_category, replace=False
        )
        # 抽出結果の並びはパス順に戻し、ペア生成順も再現可能にする。
        selected.extend(candidates[int(index)] for index in sorted(indices.tolist()))
    return selected


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
    fg_category: str, bg_category: str,
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
        str(fg_path), str(bg_path), fg_category, bg_category,
        right_score, left_score, signed, abs(signed), right_clip, left_clip,
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


def select_unique_image_pairs(
    sorted_results: list[PairResult], top_k: int
) -> list[PairResult]:
    """画像重複なしで厳密にK組を選び、スコア合計を最大化する(MILP)。

    同一性は従来通り正規化ファイルパスで判定する。
    別名コピーの画像内容の重複は検出しない。
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    if top_k < 1:
        raise ValueError("--top-kは1以上にしてください")
    key_cache: dict[str, str] = {}
    def image_key(path: str) -> str:
        if path not in key_cache:
            key_cache[path] = str(Path(path).resolve()).casefold()
        return key_cache[path]

    # FG/BGを入れ替えた同じ2画像の候補は、スコアが高い向きだけ残す。
    # 残す候補は評価済みの行なので、前景の輝度クラス制限は維持される。
    best: dict[tuple[str, str], PairResult] = {}
    for row in sorted_results:
        a, b = image_key(row.foreground_path), image_key(row.background_path)
        score = row.absolute_score_difference
        if a == b:
            continue
        if not math.isfinite(score):
            raise ValueError(f"スコアが有限値ではありません: {row.foreground_path}, {row.background_path}")
        edge = tuple(sorted((a, b)))
        if edge not in best or score > best[edge].absolute_score_difference:
            best[edge] = row
    edges = list(best)
    candidates = list(best.values())
    vertices = sorted({v for edge in edges for v in edge})
    if len(vertices) < 2 * top_k or len(edges) < top_k:
        raise ValueError(f"画像重複なしで{top_k}組を作る有効な候補が不足しています")
    vertex_index = {v: i for i, v in enumerate(vertices)}
    n, m = len(vertices), len(edges)
    # 各列が候補ペア。画像ごとの使用回数<=1、最終行の採用数==K。
    rr, cc = [], []
    for j, (a, b) in enumerate(edges):
        rr.extend((vertex_index[a], vertex_index[b], n))
        cc.extend((j, j, j))
    matrix = coo_matrix((np.ones(3 * m), (rr, cc)), shape=(n + 1, m)).tocsc()
    lower = np.r_[np.zeros(n), float(top_k)]
    upper = np.r_[np.ones(n), float(top_k)]
    scores = np.array([row.absolute_score_difference for row in candidates], dtype=float)
    # 目的関数の数値スケールを調整（最適解は変わらない）。
    scale = max(float(np.abs(scores).max()), 1e-12)
    solution = milp(
        c=-scores / scale,
        integrality=np.ones(m),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"mip_rel_gap": 0.0},
    )
    if solution.status == 2:
        raise ValueError(
            f"現在の候補では画像重複なしの{top_k}組は実現不可能です。"
            "入力画像数または事前抽出条件を見直してください。"
        )
    if not solution.success or solution.x is None:
        raise RuntimeError(f"最適な{top_k}組を確定できませんでした: {solution.message}")
    selected = [row for row, x in zip(candidates, solution.x) if x > 0.5]
    keys = [image_key(p) for row in selected for p in (row.foreground_path, row.background_path)]
    if len(selected) != top_k or len(set(keys)) != 2 * top_k:
        raise RuntimeError("最適化結果の組数・画像重複検証に失敗しました")
    return sorted(selected, key=lambda row: row.absolute_score_difference, reverse=True)

def write_prefilter_clusters_csv(
    path: Path,
    items: list[ImageClusterItem],
    selected_paths: set[Path],
) -> None:
    fieldnames = (
        "path",
        "mean_relative_luminance",
        "laplacian_features",
        "texture_cluster_label",
        "texture_rank",
        "texture_extreme",
        "luminance_class",
        "luminance_rank",
        "luminance_extreme",
        "corner_category",
        "selected_for_visibility",
        "foreground_candidate",
        "background_candidate",
    )
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "path": str(item.path),
                    "mean_relative_luminance": item.mean_relative_luminance,
                    "laplacian_features": json.dumps(item.laplacian_features),
                    "texture_cluster_label": item.texture_cluster_label,
                    "texture_rank": item.texture_rank,
                    "texture_extreme": item.texture_extreme,
                    "luminance_class": item.luminance_class,
                    "luminance_rank": item.luminance_rank,
                    "luminance_extreme": item.luminance_extreme,
                    "corner_category": item.corner_category,
                    "selected_for_visibility": item.path in selected_paths,
                    "foreground_candidate": (
                        item.path in selected_paths and item.foreground_candidate
                    ),
                    "background_candidate": (
                        item.path in selected_paths and item.background_candidate
                    ),
                }
            )


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
    image_paths = discover_images_recursive(args.image_dir)
    if not image_paths:
        raise FileNotFoundError(f"画像がありません: {args.image_dir}")

    # 高コストな視認性評価より先に、テクスチャK-meansと輝度等数分割で絞る。
    cluster_items = classify_image_extremes(
        image_paths,
        settings.texture_class_count,
        settings.luminance_class_count,
        settings.laplacian_levels,
        settings.kmeans_random_state,
    )
    sampled_items = sample_corner_categories(
        cluster_items,
        settings.samples_per_category,
        settings.sampling_random_seed,
    )
    selected_paths = {item.path for item in sampled_items}
    item_by_path = {item.path: item for item in sampled_items}
    fg_paths = [item.path for item in sampled_items if item.foreground_candidate]
    bg_paths = [item.path for item in sampled_items if item.background_candidate]
    if settings.top_k < 1:
        raise ValueError("--top-kは1以上にしてください")
    fg_keys = {str(path.resolve()).casefold() for path in fg_paths}
    bg_keys = {str(path.resolve()).casefold() for path in bg_paths}
    max_pair_count = min(len(fg_keys), len(bg_keys), len(fg_keys | bg_keys) // 2)
    if settings.top_k > max_pair_count:
        raise ValueError(
            f"重複なしの{settings.top_k}組には事前抽出後の候補が不足しています: "
            f"FG={len(fg_keys)}, BG={len(bg_keys)}, 最大={max_pair_count}組。"
            "入力画像数または事前抽出条件を見直してください。"
        )
    try:
        from scipy.optimize import milp  # noqa: F401
    except ImportError as exc:
        raise RuntimeError('python -m pip install "scipy>=1.9" を実行してください') from exc

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
    category_counts_before_sampling = {
        category: sum(item.corner_category == category for item in cluster_items)
        for category in CORNER_CATEGORIES
    }
    category_counts_after_sampling = {
        category: sum(item.corner_category == category for item in sampled_items)
        for category in CORNER_CATEGORIES
    }

    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump({
            **asdict(settings),
            "image_dir": str(args.image_dir),
            "image_count": len(image_paths),
            "corner_category_counts_before_sampling": category_counts_before_sampling,
            "corner_category_counts_after_sampling": category_counts_after_sampling,
            "sampled_image_count": len(sampled_items),
            "foreground_candidate_count": len(fg_paths),
            "background_candidate_count": len(bg_paths),
            "display_dir": str(image_config.display_dir),
            "visibility_repo": str(VISIBILITY_REPO),
        }, file, ensure_ascii=False, indent=2)
    write_prefilter_clusters_csv(
        output_dir / "image_prefilter_clusters.csv", cluster_items, selected_paths
    )

    pairs = [
        (fg_path, bg_path)
        for fg_path, bg_path in product(fg_paths, bg_paths)
        if str(fg_path.resolve()).casefold() != str(bg_path.resolve()).casefold()
    ]
    if not pairs:
        raise ValueError("事前抽出後に評価可能な前景・背景ペアがありません")
    results: list[PairResult] = []
    print(
        f"FG={settings.distance_fg_cm:g}cm, BG={settings.distance_bg_cm:g}cm, "
        f"IPD={settings.ipd_mm:g}mm, disparity={settings.disparity_deg:.4f}deg "
        f"({settings.disparity_px}px), images={len(image_paths)}, "
        f"categories_before={category_counts_before_sampling}, "
        f"categories_sampled={category_counts_after_sampling}, "
        f"FG candidates={len(fg_paths)}, BG candidates={len(bg_paths)}, "
        f"pairs={len(pairs)}"
    )
    for index, (fg_path, bg_path) in enumerate(pairs, 1):
        right_bg, left_bg = eye_bg_cache[bg_path]
        result = evaluate_pair(
            fg_path, bg_path, fg_cache[fg_path], right_bg, left_bg,
            item_by_path[fg_path].corner_category,
            item_by_path[bg_path].corner_category,
            calibration, model, settings, device,
        )
        results.append(result)
        print(
            f"[{index}/{len(pairs)}] {fg_path.name} x {bg_path.name}: "
            f"{result.absolute_score_difference:.6f}"
        )

    results.sort(key=lambda row: row.absolute_score_difference, reverse=True)
    write_csv(output_dir / "all_pairs.csv", results)  # 最適化前にも評価結果を保存
    top_results = select_unique_image_pairs(results, settings.top_k)
    for rank, result in enumerate(top_results, 1):
        result.rank = rank
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