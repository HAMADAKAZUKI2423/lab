#!/usr/bin/env python3
"""輝度で絞った背景の左右周波数クラス差を検査し、high/low各10組を選定する。

既定入力の前景はMcGill全件＋DTD各カテゴリ10枚（seed42・非復元抽出）。
背景はMcGill・DTD全件のまま。前景・背景それぞれの母集団で輝度を先に
5分位化する。前景は下位1クラスだけでラプラシアン特徴を抽出・10クラス化。
背景は上位2クラスだけでブラー後の
左右像を共通KMeansで10クラス化し、低→高周波順位に並べる。
high背景は左右順位差3以上を全件使用。low背景は右=左=kの画像を
k=1..10から固定seed42で非復元5枚ずつ抽出し、計50枚を候補とする。
不足クラスは実枚数全件を使用。別クラス補充・閾値緩和は行わない。
最終high/lowは各専用背景から選び、前景各クラスhigh/low各1組を維持。
--background-onlyで背景レポートだけを保存し、視認性モデルを読み込まない。
画像実験専用の視角（既定7.5°）を使用。common・実験コードは変更しない。

低輝度前景をテクスチャ10クラスに分け、全前景×背景を評価する。
各クラスから異なる前景を使うhigh 1組 / low 1組を選び、前景・背景は
最終20組内で重複させない。前景はぼかさず、モデル・alpha方式は維持する。

背景は校正ガンマで線形化し、パノラマ全体をGaussian blurしてから
左右眼像を切り出す。sigmaはcommon.opticsと同じ0.55係数の近似を使う。
瞳孔径は左右共通で明示指定する。IPDとは別であり、65 mmを流用しない。
--no-defocusは黒補完・ぼかしなしに戻るが、新しい背景順位差フィルタは維持する。
元画像は上書きせず、DP実機提示にぼかし済み画像を再投入しない。

実行例（lab/src、PowerShell）:
    $pupilMm = [double](Read-Host "採用する瞳孔径(mm)")
    py -m experiment.stimuli.select_fukiage_disparity_pairs --device cuda:0 --pupil-diameter-mm $pupilMm
ぼかし無効（背景フィルタは有効）:
    py -m experiment.stimuli.select_fukiage_disparity_pairs --device cuda:0 --no-defocus
"""
from __future__ import annotations

import argparse
import hashlib
from functools import lru_cache
import csv
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from experiment import experiment_config
from experiment.common.optics import (
    DEFOCUS_BLUR_SCALE_FACTOR,
    calculate_defocus_parameters,
)
from experiment.common.display_calibration import DisplayCalibration, load_display_calibration
from experiment.pre_experiment.image.config import create_image_config

SCRIPT_PATH = Path(__file__).resolve()
LAB_ROOT = SCRIPT_PATH.parents[3]
VISIBILITY_REPO = LAB_ROOT / "visibility_blend_2025-main"
DEFAULT_OUTPUT_ROOT = LAB_ROOT / "results" / "fukiage-disparity-selection"
MCGILL_IMAGE_DIR = (
    LAB_ROOT / "data" / "raw" / "images"
    / "McGill Calibrated Color Image Database" / "Textures"
)
DTD_IMAGE_DIR = (
    LAB_ROOT / "data" / "raw" / "images"
    / "Describable Textures Dataset" / "images"
)
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LUMINANCE_CLASS_COUNT = 5
TEXTURE_CLASS_COUNT = 10
LAPLACIAN_LEVELS = 3
KMEANS_RANDOM_STATE = 42
GAUSSIAN_TRUNCATE = 4.0
BACKGROUND_CLUSTER_COUNT = 10
MIN_BACKGROUND_RANK_GAP = 3
LOW_BACKGROUNDS_PER_CLASS = 5
LOW_BACKGROUND_RANDOM_SEED = 42
FOREGROUND_DTD_SAMPLES_PER_CATEGORY = 20
FOREGROUND_DTD_SAMPLING_SEED = 42


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
    luminance_class_count: int = LUMINANCE_CLASS_COUNT
    texture_class_count: int = TEXTURE_CLASS_COUNT
    laplacian_levels: int = LAPLACIAN_LEVELS
    kmeans_random_state: int = KMEANS_RANDOM_STATE
    apply_defocus: bool = True
    pupil_diameter_mm: float | None = None
    diopter_difference: float = 0.0
    blur_diameter_deg: float = 0.0
    blur_sigma_deg: float = 0.0
    blur_sigma_px: float = 0.0
    blur_scale_factor: float = DEFOCUS_BLUR_SCALE_FACTOR
    gaussian_truncate: float = GAUSSIAN_TRUNCATE
    background_cluster_count: int = BACKGROUND_CLUSTER_COUNT
    min_background_rank_gap: int = MIN_BACKGROUND_RANK_GAP
    low_backgrounds_per_class: int = LOW_BACKGROUNDS_PER_CLASS
    low_background_random_seed: int = LOW_BACKGROUND_RANDOM_SEED

    @property
    def eye_boundary_mode(self) -> str:
        # --no-defocusは旧selectorの数値を再現するため黒補完を保持する。
        return "reflect" if self.apply_defocus else "black"


@dataclass(frozen=True)
class ImageItem:
    path: Path
    mean_relative_luminance: float
    luminance_class: int
    laplacian_features: tuple[float, ...]
    foreground_texture_label: int | None
    foreground_texture_class: int | None
    distance_to_texture_center: float | None
    foreground_candidate: bool
    background_candidate: bool


@dataclass
class PairResult:
    foreground_path: str
    background_path: str
    foreground_texture_class: int
    foreground_luminance_class: int
    background_luminance_class: int
    right_visibility_score: float
    left_visibility_score: float
    signed_score_difference: float
    absolute_score_difference: float
    right_clip_ratio: float
    left_clip_ratio: float
    disparity_deg: float
    disparity_px: int
    difference_group: str = ""
    group_rank: int | None = None
    rank: int | None = None
    apply_defocus: bool = False
    pupil_diameter_mm: float | None = None
    diopter_difference: float = 0.0
    blur_sigma_deg: float = 0.0
    blur_sigma_px: float = 0.0
    eye_boundary_mode: str = "black"
    background_right_texture_class: int | None = None
    background_left_texture_class: int | None = None
    background_texture_rank_gap: int | None = None
    background_candidate_group: str = ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ipd-mm", type=float, default=60.0)
    parser.add_argument(
        "--pupil-diameter-mm", type=float, default=None,
        help="背景defocus用の左右共通瞳孔径[mm]。defocus有効時は必須。IPDとは別",
    )
    parser.add_argument(
        "--no-defocus", action="store_true",
        help="背景defocusを無効化し、旧版と同じ黒補完・ぼかしなし評価を行う",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--model", default="vismlp_norm")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--image-dir", type=Path, action="append", dest="image_dirs",
        help=("任意の入力画像ルート。複数回指定できます。指定した場合は、その"
              "ディレクトリ群を前景・背景とも全件使用します。既定入力は前景のみDTD各カテゴリ10枚です"),
    )
    parser.add_argument(
        "--selection-pool-size", type=int, default=200,
        help=("各テクスチャクラスについて、最適化へ渡す上位/下位候補数。"
              "0なら評価済み全候補を使用（既定: 200）"),
    )
    parser.add_argument(
        "--max-clip-ratio", type=float, default=1.0,
        help="左右いずれかの光学加算クリップ率がこの値を超える候補を除外（既定: 1.0）",
    )
    parser.add_argument('--background-clusters', type=int, default=BACKGROUND_CLUSTER_COUNT)
    parser.add_argument('--min-background-rank-gap', type=int, default=MIN_BACKGROUND_RANK_GAP)
    parser.add_argument('--background-only', action='store_true',
                        help='背景の輝度・左右周波数分類と確認画像だけを保存し、モデル評価はしない')
    args = parser.parse_args(argv)
    if args.background_clusters != 10:
        parser.error('low背景は10クラス各5枚（計50枚）に固定されています')
    if not 1 <= args.min_background_rank_gap < args.background_clusters:
        parser.error('背景の順位差は1以上、背景クラスタ数未満にしてください')
    if not args.no_defocus and args.pupil_diameter_mm is None:
        parser.error(
            "背景defocusには--pupil-diameter-mmが必要です。"
            "IPD（例: 65 mm）ではなく、採用する瞳孔径を指定してください"
        )
    if args.pupil_diameter_mm is not None and (
        not math.isfinite(args.pupil_diameter_mm)
        or not 0.0 < args.pupil_diameter_mm <= 10.0
    ):
        parser.error("瞳孔径は有限な0より大きく10 mm以下の値にしてください。IPDとは別です")
    return args


def right_aligned_disparity_deg(ipd_mm: float, fg_cm: float, bg_cm: float) -> float:
    ipd_cm = ipd_mm / 10.0
    return math.degrees(math.atan2(ipd_cm, fg_cm) - math.atan2(ipd_cm, bg_cm))


def load_settings(args: argparse.Namespace) -> RunSettings:
    runtime = experiment_config.get_config() or {}
    required = ("L_fg", "L_bg")
    missing = [key for key in required if key not in runtime]
    if missing:
        raise KeyError(f"experiment_config.pyに必要なキーがありません: {missing}")
    image_config = create_image_config()
    fg_cm = float(image_config.distance_fg_cm)
    bg_cm = float(image_config.distance_bg_cm)
    # 画像実験と同じ専用設定を使う。共通defocusや他実験の視角は変えない。
    angle = float(image_config.visual_angle_deg)
    l_fg = float(runtime["L_fg"])
    l_bg = float(runtime["L_bg"])
    if not all(math.isfinite(value) for value in (fg_cm, bg_cm, angle, l_fg, l_bg)):
        raise ValueError("距離・視角・輝度設定には有限な値が必要です")
    if not (0 < fg_cm < bg_cm and angle > 0 and l_fg > 0 and l_bg > 0):
        raise ValueError("距離・視角・輝度設定が不正です")
    if not math.isfinite(args.ipd_mm) or args.ipd_mm <= 0:
        raise ValueError("--ipd-mmには正の有限値が必要です")
    if args.image_size <= 0:
        raise ValueError("--image-sizeには正の整数が必要です")
    apply_defocus = not args.no_defocus
    pupil_mm = args.pupil_diameter_mm
    if pupil_mm is not None and (not math.isfinite(pupil_mm) or not 0 < pupil_mm <= 10):
        raise ValueError("瞳孔径は0より大きく10 mm以下の有限値にしてください")
    if apply_defocus and pupil_mm is None:
        raise ValueError("背景defocusには--pupil-diameter-mmの明示指定が必要です")
    diopter_difference = abs(100.0 / fg_cm - 100.0 / bg_cm)
    blur_diameter_deg, sigma_deg = (
        calculate_defocus_parameters(diopter_difference, float(pupil_mm))
        if apply_defocus else (0.0, 0.0)
    )
    pixels_per_degree = args.image_size / angle
    sigma_px = sigma_deg * pixels_per_degree
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
        apply_defocus=apply_defocus,
        pupil_diameter_mm=pupil_mm,
        diopter_difference=diopter_difference,
        blur_diameter_deg=blur_diameter_deg,
        blur_sigma_deg=sigma_deg,
        blur_sigma_px=sigma_px,
        background_cluster_count=args.background_clusters,
        min_background_rank_gap=args.min_background_rank_gap,
    )


def discover_images_recursive(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def sample_dtd_images_by_category(directory: Path) -> tuple[list[Path], dict[str, int]]:
    """前景専用：DTDの各カテゴリから10枚ずつ非復元抽出（seed42）。"""
    categories = sorted((path for path in directory.iterdir() if path.is_dir()),
                        key=lambda path: str(path).casefold())
    if not categories:
        raise ValueError(f'DTDカテゴリフォルダがありません: {directory}')
    rng = np.random.default_rng(FOREGROUND_DTD_SAMPLING_SEED)
    selected, counts = [], {}
    for category in categories:
        candidates = discover_images_recursive(category)
        counts[category.name] = len(candidates)
        if len(candidates) < FOREGROUND_DTD_SAMPLES_PER_CATEGORY:
            raise ValueError(f'DTDカテゴリ{category.name}: {len(candidates)}枚しかなく、前景用10枚を重複なしで抽出できません')
        indices = rng.choice(len(candidates), size=FOREGROUND_DTD_SAMPLES_PER_CATEGORY, replace=False)
        selected.extend(candidates[int(i)] for i in sorted(indices.tolist()))
    return selected, counts


def collect_image_candidate_paths(image_dirs, *, use_default_datasets, background_only):
    """背景は常に全件。既定入力の前景だけMcGill全件＋DTD各カテゴリ10枚。"""
    def unique_sorted(paths):
        return sorted({path.resolve() for path in paths}, key=lambda path: str(path).casefold())
    background_paths = unique_sorted([path for directory in image_dirs
                                      for path in discover_images_recursive(directory)])
    foreground_paths, sampled, category_counts = [], [], {}
    sampling_applied = use_default_datasets and not background_only
    if use_default_datasets:
        mcgill_paths = discover_images_recursive(image_dirs[0])
        dtd_paths = discover_images_recursive(image_dirs[1])
        if not background_only:
            sampled, category_counts = sample_dtd_images_by_category(image_dirs[1])
            foreground_paths = unique_sorted(mcgill_paths + sampled)
    else:
        mcgill_paths, dtd_paths = [], []
        if not background_only:
            foreground_paths = background_paths.copy()
    info = {
        'dataset_image_counts': {'foreground_pool': len(foreground_paths),
            'background_all': len(background_paths), 'mcgill_all': len(mcgill_paths),
            'dtd_all': len(dtd_paths), 'dtd_foreground_sampled': len(sampled)},
        'dtd_sampling_applied': sampling_applied,
        'dtd_sampling_scope': 'foreground_only' if sampling_applied else 'none',
        'dtd_samples_per_category': FOREGROUND_DTD_SAMPLES_PER_CATEGORY if sampling_applied else None,
        'dtd_sampling_seed': FOREGROUND_DTD_SAMPLING_SEED if sampling_applied else None,
        'dtd_category_counts_before_sampling': category_counts,
        'dtd_foreground_sample_paths': [str(path.resolve()) for path in sampled],
        'foreground_processing_performed': not background_only,
        'background_dtd_sampling_applied': False,
        'luminance_classification_scope': 'separate_foreground_pool_and_full_background_pool',
    }
    return foreground_paths, background_paths, info


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image.astype(np.float32) / 255.0


def mean_relative_luminance_from_bgr(bgr: np.ndarray) -> float:
    rgb = bgr[..., ::-1].astype(np.float64)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    y = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]
    return float(np.mean(y))


def compute_laplacian_mad(bgr: np.ndarray, levels: int) -> np.ndarray:
    gray = cv2.cvtColor(
        np.clip(bgr * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY
    ).astype(np.float32)
    features: list[float] = []
    current = gray
    for _ in range(levels):
        rows, cols = current.shape
        if rows < 2 or cols < 2:
            raise ValueError("画像が小さすぎます")
        down = cv2.pyrDown(current)
        up = cv2.pyrUp(down, dstsize=(cols, rows))
        features.append(float(np.mean(np.abs(current - up))))
        current = down
    features.append(float(np.mean(np.abs(current - np.mean(current)))))
    return np.asarray(features, dtype=np.float64)


def classify_images(paths: list[Path], settings: RunSettings, *, luminance_items=None) -> list[ImageItem]:
    """輝度を等数5分割し、最下位前景候補だけをテクスチャ10クラス化する。"""
    if len(paths) < settings.luminance_class_count * settings.texture_class_count:
        raise ValueError("画像数が少なすぎます")
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise RuntimeError('python -m pip install "scikit-learn" を実行してください') from exc

    # 輝度を先に確定。ラプラシアン特徴は低輝度前景だけに計算する。
    base = classify_background_luminance(paths, settings) if luminance_items is None else luminance_items
    if [item.path for item in base] != list(paths):
        raise ValueError('共通の輝度分類と入力画像の順序が不一致です')
    measured = [(item.path, item.mean_relative_luminance,
                 compute_laplacian_mad(read_bgr(item.path), settings.laplacian_levels)
                 if item.luminance_class == 1 else np.empty(0, dtype=float))
                for item in base]
    luminance_classes = np.asarray([item.luminance_class for item in base], dtype=np.int32)

    fg_indices = np.flatnonzero(luminance_classes == 1)
    if len(fg_indices) < 2 * settings.texture_class_count:
        raise ValueError(
            f"各テクスチャクラスから異なる前景を2枚選ぶには、低輝度候補が最低"
            f"{2 * settings.texture_class_count}枚必要です: {len(fg_indices)}枚"
        )
    x = np.vstack([measured[int(i)][2] for i in fg_indices])
    model = KMeans(
        n_clusters=settings.texture_class_count,
        random_state=settings.kmeans_random_state,
        n_init=10,
    )
    labels = model.fit_predict(x)
    if len(np.unique(labels)) != settings.texture_class_count:
        raise ValueError("前景候補を10個の異なるテクスチャクラスタへ分類できませんでした")
    center_order = np.argsort(model.cluster_centers_.sum(axis=1), kind="stable")
    rank_by_label = {int(label): rank for rank, label in enumerate(center_order, start=1)}
    texture_info: dict[int, tuple[int, int, float]] = {}
    for local_i, global_i in enumerate(fg_indices):
        label = int(labels[local_i])
        distance = float(np.linalg.norm(x[local_i] - model.cluster_centers_[label]))
        texture_info[int(global_i)] = (label, rank_by_label[label], distance)

    counts = {rank: 0 for rank in range(1, settings.texture_class_count + 1)}
    for _, rank, _ in texture_info.values():
        counts[rank] += 1
    insufficient = {rank: count for rank, count in counts.items() if count < 2}
    if insufficient:
        raise ValueError(f"異なる前景をhigh/lowへ割り当てられないクラスがあります: {insufficient}")

    items: list[ImageItem] = []
    for i, (path, luminance, features) in enumerate(measured):
        lum_class = int(luminance_classes[i])
        texture = texture_info.get(i)
        items.append(ImageItem(
            path=path,
            mean_relative_luminance=luminance,
            luminance_class=lum_class,
            laplacian_features=tuple(float(v) for v in features),
            foreground_texture_label=None if texture is None else texture[0],
            foreground_texture_class=None if texture is None else texture[1],
            distance_to_texture_center=None if texture is None else texture[2],
            foreground_candidate=lum_class == 1,
            background_candidate=lum_class in (4, 5),
        ))
    return items


def prepare_foreground(path: Path, size: int) -> np.ndarray:
    return cv2.resize(read_bgr(path), (size, size), interpolation=cv2.INTER_AREA)


def prepare_background_panorama(path: Path, size: int) -> np.ndarray:
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


def crop_with_reflect(panorama: np.ndarray, start_x: int, width: int) -> np.ndarray:
    """背景の範囲外を反射補完する。黒い縁による人工的な左右差を避ける。"""
    left_pad = max(0, -start_x)
    right_pad = max(0, start_x + width - panorama.shape[1])
    mode = "reflect" if panorama.shape[1] > 1 else "edge"
    padded = np.pad(
        panorama, ((0, 0), (left_pad, right_pad), (0, 0)), mode=mode
    )
    return padded[:, start_x + left_pad:start_x + left_pad + width].copy()


def make_eye_backgrounds(
    panorama: np.ndarray, output_width: int, disparity_px: int,
    *, boundary_mode: str = "black",
) -> tuple[np.ndarray, np.ndarray]:
    if panorama.ndim != 3 or panorama.shape[2] != 3 or min(panorama.shape[:2]) <= 0:
        raise ValueError("背景には空でないH×W×3配列が必要です")
    if output_width <= 0:
        raise ValueError("output_widthには正値が必要です")
    if boundary_mode not in {"black", "reflect"}:
        raise ValueError(f"未定義の境界処理: {boundary_mode}")
    crop = crop_with_reflect if boundary_mode == "reflect" else crop_with_black
    right_start = (panorama.shape[1] - output_width) // 2
    left_start = right_start + disparity_px
    return (
        crop(panorama, right_start, output_width),
        crop(panorama, left_start, output_width),
    )


def apply_gamma(rgb: np.ndarray, gamma: dict[str, float]) -> np.ndarray:
    out = np.empty_like(rgb, dtype=np.float64)
    for i, channel in enumerate("RGB"):
        out[..., i] = np.clip(rgb[..., i], 0, 1) ** float(gamma[channel])
    return out


def invert_gamma(linear_rgb: np.ndarray, gamma: dict[str, float]) -> np.ndarray:
    out = np.empty_like(linear_rgb, dtype=np.float64)
    for i, channel in enumerate("RGB"):
        out[..., i] = np.clip(linear_rgb[..., i], 0, None) ** (1.0 / float(gamma[channel]))
    return out


def blur_background_panorama(
    panorama_bgr: np.ndarray,
    calibration: DisplayCalibration,
    settings: RunSettings,
) -> np.ndarray:
    """背景全体を線形RGBでぼかし、校正ガンマ付きBGRへ戻す。

    common.opticsと同じsigma式を使うが、畳み込みは反射境界Gaussian。
    FFTの周期的回り込み、途中の8bit量子化、コントラスト再正規化は行わない。
    """
    if not settings.apply_defocus:
        return panorama_bgr.copy()
    if settings.pupil_diameter_mm is None:
        raise ValueError("背景defocus用の瞳孔径が設定されていません")
    if calibration.gamma_bg is None:
        raise RuntimeError("背景defocusにはgamma_bg.csvが必要です")
    sigma = settings.blur_sigma_px
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("背景defocusのsigma_pxには正の有限値が必要です")
    linear_rgb = apply_gamma(panorama_bgr[..., ::-1], calibration.gamma_bg)
    radius = max(1, math.ceil(settings.gaussian_truncate * sigma))
    kernel_size = radius * 2 + 1
    blurred_linear = cv2.GaussianBlur(
        linear_rgb, (kernel_size, kernel_size),
        sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101,
    )
    encoded_rgb = invert_gamma(blurred_linear, calibration.gamma_bg)
    return np.ascontiguousarray(
        np.clip(encoded_rgb[..., ::-1], 0.0, 1.0), dtype=np.float32
    )


def prepare_scoring_backgrounds(
    panorama: np.ndarray,
    calibration: DisplayCalibration,
    settings: RunSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """全候補評価と選定ペア保存で共有する背景準備。前景には触れない。"""
    blurred_panorama = blur_background_panorama(panorama, calibration, settings)
    return make_eye_backgrounds(
        blurred_panorama, settings.image_size_px, settings.disparity_px,
        boundary_mode=settings.eye_boundary_mode,
    )


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
    assert calibration.gamma_bg is not None and calibration.gamma_fg is not None
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
    assert calibration.gamma_bg is not None and calibration.gamma_fg is not None
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
    fg_item: ImageItem, bg_item: ImageItem, foreground: np.ndarray,
    right_bg: np.ndarray, left_bg: np.ndarray,
    calibration: DisplayCalibration, model, settings: RunSettings,
    device: torch.device,
) -> PairResult:
    assert fg_item.foreground_texture_class is not None
    right_target, right_reference = calibrated_model_components(foreground, right_bg, calibration)
    left_target, left_reference = calibrated_model_components(foreground, left_bg, calibration)
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
        foreground_path=str(fg_item.path),
        background_path=str(bg_item.path),
        foreground_texture_class=fg_item.foreground_texture_class,
        foreground_luminance_class=fg_item.luminance_class,
        background_luminance_class=bg_item.luminance_class,
        right_visibility_score=right_score,
        left_visibility_score=left_score,
        signed_score_difference=signed,
        absolute_score_difference=abs(signed),
        right_clip_ratio=right_clip,
        left_clip_ratio=left_clip,
        disparity_deg=settings.disparity_deg,
        disparity_px=settings.disparity_px,
        apply_defocus=settings.apply_defocus,
        pupil_diameter_mm=settings.pupil_diameter_mm,
        diopter_difference=settings.diopter_difference,
        blur_sigma_deg=settings.blur_sigma_deg,
        blur_sigma_px=settings.blur_sigma_px,
        eye_boundary_mode=settings.eye_boundary_mode,
    )


def _image_key(path: str) -> str:
    return str(Path(path).resolve()).casefold()


def select_high_low_pairs(
    results: list[PairResult], pool_size: int, max_clip_ratio: float,
    *, min_background_rank_gap: int = MIN_BACKGROUND_RANK_GAP,
) -> list[PairResult]:
    """各クラスhigh/low各1組、前景・背景重複なしで20組をMILP選定する。"""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    if pool_size < 0:
        raise ValueError("--selection-pool-sizeは0以上にしてください")
    if not 0 <= max_clip_ratio <= 1:
        raise ValueError("--max-clip-ratioは0以上1以下にしてください")

    for row in results:
        if row.background_candidate_group not in {'high', 'low'}:
            raise ValueError('high/low背景用途が未設定です')
        gap = row.background_texture_rank_gap
        right, left = row.background_right_texture_class, row.background_left_texture_class
        if right is None or left is None or not (1 <= right <= 10 and 1 <= left <= 10) or gap != abs(right - left):
            raise ValueError('背景クラスと順位差が不整合です')
        if row.background_candidate_group == 'low' and (
            gap != 0 or row.background_right_texture_class is None
            or row.background_right_texture_class != row.background_left_texture_class):
            raise ValueError('low候補に左右異クラスの背景があります')
        if row.background_candidate_group == 'high' and (gap is None or gap < min_background_rank_gap):
            raise ValueError('high候補に左右同クラスの背景があります')
    valid = [
        row for row in results
        if math.isfinite(row.absolute_score_difference)
        and max(row.right_clip_ratio, row.left_clip_ratio) <= max_clip_ratio
    ]
    by_class = {
        texture_class: [r for r in valid if r.foreground_texture_class == texture_class]
        for texture_class in range(1, TEXTURE_CLASS_COUNT + 1)
    }
    missing = [c for c, rows in by_class.items() if not rows]
    if missing:
        raise ValueError(f"クリップ率制約後に候補がないテクスチャクラス: {missing}")

    # 全組合せは評価・CSV保存する。MILPの変数数だけを上位/下位候補へ制限する。
    candidates: list[tuple[str, PairResult, float]] = []
    for texture_class, rows in by_class.items():
        ordered = sorted(rows, key=lambda r: r.absolute_score_difference)
        low_rows = [r for r in ordered if r.background_candidate_group == 'low']
        high_rows = [r for r in ordered if r.background_candidate_group == 'high']
        if not low_rows or not high_rows:
            raise ValueError(f'前景クラス{texture_class}にhigh/low専用候補が不足（クリップ率除外後）')
        if pool_size:
            low_rows = low_rows[:pool_size]
            high_rows = high_rows[-pool_size:]
        values = np.asarray([r.absolute_score_difference for r in ordered], dtype=float)
        lo, hi = float(values.min()), float(values.max())
        denominator = max(hi - lo, 1e-12)
        for row in high_rows:
            z = (row.absolute_score_difference - lo) / denominator
            candidates.append(("high", row, float(z)))
        for row in low_rows:
            z = (row.absolute_score_difference - lo) / denominator
            candidates.append(("low", row, float(1.0 - z)))

    group_keys = [
        (group, texture_class)
        for group in ("high", "low")
        for texture_class in range(1, TEXTURE_CLASS_COUNT + 1)
    ]
    group_index = {key: i for i, key in enumerate(group_keys)}
    fg_keys = sorted({_image_key(row.foreground_path) for _, row, _ in candidates})
    bg_keys = sorted({_image_key(row.background_path) for _, row, _ in candidates})
    fg_index = {key: i + len(group_keys) for i, key in enumerate(fg_keys)}
    bg_offset = len(group_keys) + len(fg_keys)
    bg_index = {key: i + bg_offset for i, key in enumerate(bg_keys)}
    row_count = len(group_keys) + len(fg_keys) + len(bg_keys)

    rr: list[int] = []
    cc: list[int] = []
    for column, (group, row, _) in enumerate(candidates):
        rr.extend((
            group_index[(group, row.foreground_texture_class)],
            fg_index[_image_key(row.foreground_path)],
            bg_index[_image_key(row.background_path)],
        ))
        cc.extend((column, column, column))
    matrix = coo_matrix(
        (np.ones(len(rr)), (rr, cc)), shape=(row_count, len(candidates))
    ).tocsc()
    lower = np.r_[np.ones(len(group_keys)), np.zeros(len(fg_keys) + len(bg_keys))]
    upper = np.r_[np.ones(len(group_keys)), np.ones(len(fg_keys) + len(bg_keys))]
    utilities = np.asarray([utility for _, _, utility in candidates], dtype=float)
    solution = milp(
        c=-utilities,
        integrality=np.ones(len(candidates)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"mip_rel_gap": 0.0},
    )
    if solution.status == 2:
        raise ValueError(
            "現在の候補プールでは、各クラスhigh/low各1組かつ前景・背景重複なしを"
            "満たせません。--selection-pool-sizeを増やすか0にしてください。"
        )
    if not solution.success or solution.x is None:
        raise RuntimeError(f"20組を確定できませんでした: {solution.message}")

    chosen: list[PairResult] = []
    for (group, row, _), x in zip(candidates, solution.x):
        if x > 0.5:
            row.difference_group = group
            chosen.append(row)
    if len(chosen) != 20:
        raise RuntimeError(f"選定組数が20ではありません: {len(chosen)}")

    chosen.sort(key=lambda r: (
        0 if r.difference_group == "high" else 1,
        r.foreground_texture_class,
    ))
    for rank, row in enumerate(chosen, start=1):
        row.rank = rank
        same_group = [
            r for r in chosen if r.difference_group == row.difference_group
        ]
        same_group.sort(
            key=lambda r: r.absolute_score_difference,
            reverse=row.difference_group == "high",
        )
        row.group_rank = same_group.index(row) + 1

    if any(r.difference_group != r.background_candidate_group for r in chosen):
        raise RuntimeError('high/low背景用途が最終群と一致しません')
    for group in ('high', 'low'):
        if sorted(r.foreground_texture_class for r in chosen if r.difference_group == group) != list(range(1, 11)):
            raise RuntimeError('各前景クラスhigh/low各1組の検証に失敗しました')
    if len({_image_key(r.foreground_path) for r in chosen}) != 20:
        raise RuntimeError("前景重複検証に失敗しました")
    if len({_image_key(r.background_path) for r in chosen}) != 20:
        raise RuntimeError("背景重複検証に失敗しました")
    return chosen


def write_csv(path: Path, rows: list[PairResult]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def write_image_classes_csv(path: Path, items: list[ImageItem]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = list(asdict(items[0]).keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            row = asdict(item)
            row["path"] = str(item.path)
            row["laplacian_features"] = json.dumps(item.laplacian_features)
            writer.writerow(row)


def save_bgr(path: Path, image: np.ndarray) -> None:
    cv2.imwrite(str(path), np.clip(image * 255, 0, 255).astype(np.uint8))


def save_vismap(path: Path, vismap: np.ndarray, heatmap: bool = False) -> None:
    gray = np.clip(vismap * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(
        str(path), cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS) if heatmap else gray
    )


def save_selected_pair(
    output_dir: Path, result: PairResult,
    foreground: np.ndarray, panorama: np.ndarray,
    calibration: DisplayCalibration, model,
    settings: RunSettings, device: torch.device,
    *, eye_backgrounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    if result.difference_group not in {"high", "low"}:
        raise ValueError(
            f"difference_groupはhighまたはlowにしてください: "
            f"{result.difference_group!r}"
        )
    if result.difference_group != result.background_candidate_group:
        raise ValueError('保存群と背景候補用途が不一致です')
    if result.group_rank is None:
        raise ValueError("group_rankが設定されていません")
    group_dir = output_dir / f"{result.difference_group}10"
    pair_dir = group_dir / f"rank_{result.group_rank:02d}"
    pair_dir.mkdir(parents=True, exist_ok=False)
    right_bg, left_bg = (
        eye_backgrounds if eye_backgrounds is not None
        else prepare_scoring_backgrounds(panorama, calibration, settings)
    )
    right_target, right_ref = calibrated_model_components(foreground, right_bg, calibration)
    left_target, left_ref = calibrated_model_components(foreground, left_bg, calibration)
    right_map = compute_visibility_map(model, right_target, right_ref, settings.alpha, device)
    left_map = compute_visibility_map(model, left_target, left_ref, settings.alpha, device)
    # 保存するマップがall_pairs/high10/low10のスコアと一致することを検証。
    for eye, actual, expected in (
        ("right", float(right_map.mean()), result.right_visibility_score),
        ("left", float(left_map.mean()), result.left_visibility_score),
    ):
        if not math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-6):
            raise RuntimeError(f"保存時の{eye}スコアが評価時と不一致: {actual} != {expected}")
    right_composite, _ = optical_addition_bgr(foreground, right_bg, calibration)
    left_composite, _ = optical_addition_bgr(foreground, left_bg, calibration)
    right_unblurred, left_unblurred = make_eye_backgrounds(
        panorama, settings.image_size_px, settings.disparity_px,
        boundary_mode=settings.eye_boundary_mode,
    )

    fg_source = Path(result.foreground_path)
    bg_source = Path(result.background_path)
    shutil.copy2(fg_source, pair_dir / f"foreground_original{fg_source.suffix}")
    shutil.copy2(bg_source, pair_dir / f"background_original{bg_source.suffix}")
    save_bgr(pair_dir / "foreground_prepared.png", foreground)
    save_bgr(pair_dir / "right_background_view_unblurred.png", right_unblurred)
    save_bgr(pair_dir / "left_background_view_unblurred.png", left_unblurred)
    save_bgr(pair_dir / "right_background_view.png", right_bg)
    save_bgr(pair_dir / "left_background_view.png", left_bg)
    save_bgr(pair_dir / "right_optical_composite.png", right_composite)
    save_bgr(pair_dir / "left_optical_composite.png", left_composite)
    save_vismap(pair_dir / "right_vismap_gray.png", right_map)
    save_vismap(pair_dir / "left_vismap_gray.png", left_map)
    save_vismap(pair_dir / "right_vismap_heat.png", right_map, True)
    save_vismap(pair_dir / "left_vismap_heat.png", left_map, True)
    with (pair_dir / "pair_metadata.json").open("w", encoding="utf-8") as file:
        json.dump({
            **asdict(result),
            "blur_method": "gaussian_linear_rgb" if settings.apply_defocus else "none",
            "blur_boundary_mode": "reflect101" if settings.apply_defocus else "none",
            "blur_scale_factor": settings.blur_scale_factor,
            "gaussian_truncate": settings.gaussian_truncate,
            "model_input_mode": "calibrated_components_linear_alpha",
            "original_images_preserved": True,
            "dp_presentation_source": "background_original (not the blurred view)",
        }, file, ensure_ascii=False, indent=2)


@dataclass
class BackgroundEyeClass:
    background_path: str
    luminance_class: int
    mean_relative_luminance: float
    right_features: tuple = ()
    left_features: tuple = ()
    right_label: int | None = None
    left_label: int | None = None
    right_rank: int | None = None
    left_rank: int | None = None
    rank_gap: int | None = None
    accepted: bool = False
    reason: str = "luminance"
    source_sha256: str = ""
    high_candidate: bool = False
    low_eligible: bool = False
    low_candidate: bool = False
    candidate_group: str = ""


def source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def classify_background_luminance(paths, settings, *, mean_luminances=None):
    """背景の全入力を輝度5分位化。重い左右像・pyramid処理は上位2クラスだけ。"""
    if len(paths) < settings.luminance_class_count:
        raise ValueError('背景の輝度分類に必要な画像数がありません')
    # 輝度値は再利用するが、分位クラスはこの母集団内で付け直す。
    measured = [(path, mean_relative_luminance_from_bgr(read_bgr(path))
                 if mean_luminances is None else mean_luminances[path]) for path in paths]
    order = sorted(range(len(measured)), key=lambda i: (measured[i][1], str(measured[i][0]).casefold()))
    classes = np.zeros(len(measured), dtype=int)
    for rank, indices in enumerate(np.array_split(order, settings.luminance_class_count), start=1):
        classes[indices] = rank
    return [ImageItem(path, lum, int(classes[i]), (), None, None, None,
                      False, int(classes[i]) >= settings.luminance_class_count - 1)
            for i, (path, lum) in enumerate(measured)]


def background_frequency_features(bgr, calibration, levels):
    """校正線形Yの各帯域MADをL1正規化。強度ではなく周波数の構成比。"""
    linear = apply_gamma(bgr[..., ::-1], calibration.gamma_bg)
    gray = linear @ np.asarray(calibration.t_prime)[1, :]
    if not np.isfinite(gray).all() or np.min(gray) < -1e-9:
        raise ValueError('背景校正の線形Yが非有限または負値です')
    current = np.maximum(gray, 0).astype(np.float64)
    features = []
    for _ in range(levels):
        rows, cols = current.shape
        if min(rows, cols) < 2:
            raise ValueError('背景画像がピラミッドの段数に対して小さすぎます')
        down = cv2.pyrDown(current)
        up = cv2.pyrUp(down, dstsize=(cols, rows))
        features.append(float(np.mean(np.abs(current - up))))
        current = down
    features.append(float(np.mean(np.abs(current - current.mean()))))
    values = np.asarray(features, dtype=np.float64)
    if values.sum() <= 1e-10:
        return None  # 一様画像の周波数傾向は定義しない。
    return values / values.sum()


def fit_background_frequency_classes(features, settings):
    from sklearn.cluster import KMeans
    x = np.asarray(features, dtype=np.float64)
    k = settings.background_cluster_count
    if len(x) < k or len(np.unique(x, axis=0)) < k:
        raise ValueError(f'背景左右像を{k}個の異なるクラスタへ分けられません。valid eye views={len(x)}')
    model = KMeans(n_clusters=k, random_state=settings.kmeans_random_state, n_init=10)
    labels = model.fit_predict(x)
    if len(np.unique(labels)) != k:
        raise ValueError('背景クラスタ数が指定数に達しませんでした')
    centers = model.cluster_centers_
    # 特徴量順は高周波→低周波→残差。Fは実測cpdではなく相対指標。
    frequency = centers @ np.arange(settings.laplacian_levels, -1, -1, dtype=float)
    order = np.argsort(frequency, kind='stable')
    if np.any(np.diff(frequency[order]) <= 1e-9):
        raise ValueError('周波数指標が同じ背景クラスタがあり、意味のある順位を定義できません')
    ranks = {int(label): rank for rank, label in enumerate(order, start=1)}
    return labels, centers, frequency, ranks


def classify_background_eyes(items, settings, calibration):
    """輝度通過背景のみ逐次処理し、左右2N特徴を同じKMeansへ投入する。"""
    records, features, owners = [], [], []
    for index, item in enumerate(items, start=1):
        record = BackgroundEyeClass(str(item.path), item.luminance_class, item.mean_relative_luminance)
        records.append(record)
        if not item.background_candidate:
            continue
        record.source_sha256 = source_hash(item.path)
        panorama = prepare_background_panorama(item.path, settings.image_size_px)
        eyes = prepare_scoring_backgrounds(panorama, calibration, settings)
        if source_hash(item.path) != record.source_sha256:
            raise RuntimeError(f'背景読み込み中に元画像が変更されました: {item.path}')
        pair = [background_frequency_features(view, calibration, settings.laplacian_levels) for view in eyes]
        if any(value is None for value in pair):
            record.reason = 'flat_or_undefined_frequency'
            continue
        record.right_features, record.left_features = (tuple(value.tolist()) for value in pair)
        record.reason = 'rank_gap'
        for eye, vector in zip(('right', 'left'), pair):
            features.append(vector)
            owners.append((record, eye))
        print(f'Background features [{index}/{len(items)}]: {item.path.name}', flush=True)
    labels, centers, frequency, ranks = fit_background_frequency_classes(features, settings)
    for (record, eye), label in zip(owners, labels):
        setattr(record, eye + '_label', int(label))
        setattr(record, eye + '_rank', ranks[int(label)])
    # high/lowの分岐は全左右像の共通fitが完了してから行う。
    for record in records:
        if record.right_rank is None:
            continue
        record.rank_gap = abs(record.right_rank - record.left_rank)
        record.reason = 'unassigned'
    return records, centers, frequency, ranks


def assign_background_pools(records, settings):
    """スコアを参照せず、high全件と同クラスlow候補を分離する。

    lowは各クラス最大5枚を抽出し、不足クラスは実枚数全件を使用する。
    """
    if settings.background_cluster_count != 10:
        raise ValueError('low用背景には共通10クラスが必要です')
    if settings.low_backgrounds_per_class != 5:
        raise ValueError('low用背景は各クラス5枚に固定されています')
    by_class = {rank: [] for rank in range(1, settings.background_cluster_count + 1)}
    paths = [r.background_path for r in records]
    if len(set(paths)) != len(paths):
        raise ValueError('背景分類結果の画像パスが重複しています')
    for r in records:
        r.high_candidate = r.low_eligible = r.low_candidate = r.accepted = False
        r.candidate_group = ''
        if r.right_rank is None or r.left_rank is None:
            continue
        r.rank_gap = abs(r.right_rank - r.left_rank)
        r.high_candidate = r.rank_gap >= settings.min_background_rank_gap
        r.low_eligible = r.right_rank == r.left_rank
        if r.high_candidate:
            r.candidate_group = 'high'; r.accepted = True; r.reason = 'high_accepted'
        elif r.low_eligible:
            by_class[r.right_rank].append(r); r.reason = 'low_not_sampled'
        else:
            r.reason = 'intermediate_rank_gap'
    counts = {rank: len(rows) for rank, rows in by_class.items()}
    insufficient = {rank: count for rank, count in counts.items()
                    if count < settings.low_backgrounds_per_class}
    rng = np.random.default_rng(settings.low_background_random_seed)
    for rank, rows in by_class.items():
        ordered = sorted(rows, key=lambda r: (r.background_path.casefold(), r.background_path))
        sample_count = min(len(ordered), settings.low_backgrounds_per_class)
        indices = rng.choice(len(ordered), size=sample_count, replace=False)
        for index in indices:
            r = ordered[int(index)]
            r.low_candidate = r.accepted = True
            r.candidate_group = 'low'; r.reason = 'low_sampled'
    return counts, insufficient


def write_background_pool_manifest(output, records, counts, insufficient, settings):
    fields = ['background_path', 'candidate_group', 'luminance_class', 'right_rank',
              'left_rank', 'rank_gap', 'source_sha256']
    for group in ('high', 'low'):
        with (output / f'{group}_background_candidates.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            for r in sorted((r for r in records if r.candidate_group == group), key=lambda r: (r.right_rank, r.background_path)):
                writer.writerow({key: getattr(r, key) for key in fields})
    with (output / 'background_same_class_counts.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['background_class', 'eligible_count', 'sampled_count', 'required_count'])
        writer.writeheader()
        for rank, count in counts.items():
            writer.writerow({'background_class': rank, 'eligible_count': count,
                             'sampled_count': sum(r.low_candidate and r.right_rank == rank for r in records),
                             'required_count': settings.low_backgrounds_per_class})
    (output / 'background_pool_status.json').write_text(json.dumps({
        'complete': not bool(insufficient), 'same_class_counts': counts,
        'insufficient_classes': insufficient, 'low_sampling_seed': settings.low_background_random_seed,
        'low_sampling_policy': 'uniform_without_replacement_before_model_evaluation',
        'high_count': sum(r.high_candidate for r in records),
        'low_count': sum(r.low_candidate for r in records)}, ensure_ascii=False, indent=2), encoding='utf-8')
    # 同クラス判定は画素や視認性スコアの一致を意味しない。


def write_background_reports(output, records, centers, frequency, ranks, settings, calibration):
    fields = list(asdict(records[0]))
    with (output / 'background_eye_classes.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            for name in ('right_features', 'left_features'):
                row[name] = json.dumps(row[name])
            writer.writerow(row)
    center_rows = []
    for label in sorted(ranks, key=ranks.get):
        center_rows.append({'label': label, 'frequency_rank': ranks[label],
            'frequency_index': float(frequency[label]), 'features': json.dumps(centers[label].tolist()),
            'eye_count': sum(int(r.right_label == label) + int(r.left_label == label) for r in records)})
    with (output / 'background_cluster_centers.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(center_rows[0])); writer.writeheader(); writer.writerows(center_rows)
    details = {'background_cluster_count': settings.background_cluster_count,
        'min_background_rank_gap': settings.min_background_rank_gap,
        'laplacian_levels': settings.laplacian_levels, 'random_state': settings.kmeans_random_state,
        'n_init': 10, 'feature': 'calibrated_linear_Y_band_MAD_L1_normalized',
        'frequency_weights': list(range(settings.laplacian_levels, -1, -1)),
        'frequency_index_is_cpd': False, 'fit_scope': 'both_eyes_of_bright_backgrounds',
        'flat_pair_policy': 'exclude_pair_from_fit_and_selection',
        'centers': centers.tolist(), 'label_to_rank': ranks,
        'run_settings': asdict(settings), 'source_count': len(records),
        'bright_count': sum(r.luminance_class >= settings.luminance_class_count - 1 for r in records),
        'accepted_count': sum(r.accepted for r in records),
        'high_count': sum(r.high_candidate for r in records),
        'low_eligible_count': sum(r.low_eligible for r in records),
        'low_sampled_count': sum(r.low_candidate for r in records)}
    (output / 'background_clustering_config.json').write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding='utf-8')
    transitions = np.zeros((settings.background_cluster_count, settings.background_cluster_count), dtype=int)
    for r in records:
        if r.right_rank is not None:
            transitions[r.right_rank - 1, r.left_rank - 1] += 1
    # 行:右眼順位、列:左眼順位。countsもCSVで保存する。
    np.savetxt(output / 'background_rank_transitions.csv', transitions, delimiter=',', fmt='%d',
               header='right_rank_rows_left_rank_columns_1_to_K')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(transitions, origin='lower', cmap='viridis')
    ax.set(xticks=range(settings.background_cluster_count), yticks=range(settings.background_cluster_count),
           xticklabels=range(1, settings.background_cluster_count + 1), yticklabels=range(1, settings.background_cluster_count + 1),
           xlabel='Left-eye frequency rank', ylabel='Right-eye frequency rank', title='Bright-background class transitions')
    fig.colorbar(image, ax=ax, label='Background count'); fig.tight_layout()
    fig.savefig(output / 'background_rank_transitions.png', dpi=140); plt.close(fig)
    # 各クラスタに最も近い左右像を1枚ずつ選ぶ。保持する画像はこの小さい一覧のみ。
    example_tiles = []
    for label in sorted(ranks, key=ranks.get):
        candidates = [(float(np.linalg.norm(np.asarray(getattr(r, eye + '_features')) - centers[label])), r, eye)
                      for r in records for eye in ('right', 'left') if getattr(r, eye + '_label') == label]
        _, r, eye = min(candidates, key=lambda row: row[0])
        if source_hash(Path(r.background_path)) != r.source_sha256:
            raise RuntimeError('代表画像保存前に背景が変更されました')
        views = prepare_scoring_backgrounds(prepare_background_panorama(Path(r.background_path), settings.image_size_px), calibration, settings)
        tile = cv2.resize(views[0 if eye == 'right' else 1], (160, 160))
        tile = np.clip(tile * 255, 0, 255).astype(np.uint8)
        tile = cv2.copyMakeBorder(tile, 24, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(tile, f'Rank {ranks[label]} {eye}', (4, 17), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
        example_tiles.append(tile)
    cv2.imwrite(str(output / 'background_cluster_examples.png'), np.concatenate(example_tiles, axis=1))
    for group in ('high', 'low'):
        comparisons = []
        for r in sorted((r for r in records if r.candidate_group == group), key=lambda r: (-r.rank_gap, r.background_path))[:20]:
            if source_hash(Path(r.background_path)) != r.source_sha256:
                raise RuntimeError('左右比較保存前に背景が変更されました')
            eyes = prepare_scoring_backgrounds(prepare_background_panorama(Path(r.background_path), settings.image_size_px), calibration, settings)
            tile = np.concatenate([cv2.resize(view, (128,128)) for view in eyes], axis=1)
            tile = cv2.copyMakeBorder(np.clip(tile * 255,0,255).astype(np.uint8),24,0,0,0,cv2.BORDER_CONSTANT)
            cv2.putText(tile, f'R={r.right_rank} L={r.left_rank} gap={r.rank_gap}',(4,17),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
            comparisons.append(tile)
        if comparisons:
            cv2.imwrite(str(output / f'background_{group}_examples_first20.png'), np.concatenate(comparisons,axis=0))


def main() -> None:
    args = parse_args()
    settings = load_settings(args)
    use_default_datasets = not args.image_dirs
    image_dirs = [
        path.expanduser().resolve()
        for path in (
            args.image_dirs
            or (MCGILL_IMAGE_DIR, DTD_IMAGE_DIR)
        )
    ]
    missing_image_dirs = [path for path in image_dirs if not path.is_dir()]
    if missing_image_dirs:
        raise FileNotFoundError(
            "入力画像ディレクトリがありません: "
            + ", ".join(str(path) for path in missing_image_dirs)
        )

    paths, background_paths, dataset_info = collect_image_candidate_paths(
        image_dirs, use_default_datasets=use_default_datasets,
        background_only=args.background_only,
    )
    if not background_paths:
        raise FileNotFoundError('画像がありません: ' + ', '.join(str(path) for path in image_dirs))
    if not args.background_only:
        try:
            from scipy.optimize import milp  # noqa: F401
        except ImportError as exc:
            raise RuntimeError('python -m pip install "scipy>=1.9" を実行してください') from exc

    # 背景専用モードでは前景のクラスタリング・モデル読込を一切行わない。
    bg_luminance_items = classify_background_luminance(background_paths, settings)
    # 背景全件の輝度クラスを前景へ流用しない。前景の抽出済み母集団で5分位化する。
    if args.background_only:
        items = []
    else:
        mean_luminances = {item.path: item.mean_relative_luminance for item in bg_luminance_items}
        fg_luminance_items = classify_background_luminance(paths, settings, mean_luminances=mean_luminances)
        items = classify_images(paths, settings, luminance_items=fg_luminance_items)
    fg_items = [item for item in items if item.foreground_candidate]
    output_dir = args.output_dir or (
        DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    image_config = create_image_config()
    calibration = load_display_calibration(image_config.display_dir)
    require_calibration(calibration)
    config = {**asdict(settings), 'complete': False,
        'mode': 'background_only' if args.background_only else 'full_selection',
        'pixels_per_degree': settings.image_size_px / settings.visual_angle_deg,
        'blur_method': 'gaussian_linear_rgb' if settings.apply_defocus else 'none',
        'blur_boundary_mode': 'reflect101' if settings.apply_defocus else 'none',
        'eye_boundary_mode': settings.eye_boundary_mode,
        'model_input_mode': 'calibrated_components_linear_alpha',
        'original_images_preserved': True, 'image_dirs': [str(path) for path in image_dirs],
        'image_count': len(paths), 'background_source_count': len(background_paths),
        **dataset_info,
        'foreground_candidate_count': len(fg_items),
        'background_bright_count': sum(item.background_candidate for item in bg_luminance_items),
        'selection_pool_size': args.selection_pool_size, 'max_clip_ratio': args.max_clip_ratio,
        'selection_rule': 'FG McGill all plus DTD sample10 per category; BG all; separate luminance quantiles first; high BG gap>=threshold; low BG same class sample5 per cluster; high/low per FG class; no duplicate FG/BG',
        'display_dir': str(image_config.display_dir), 'visibility_repo': str(VISIBILITY_REPO)}
    def save_config():
        (output_dir / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    save_config()
    # 抽出された前景入力一覧を保存（輝度・テクスチャ選別前）。
    if not args.background_only:
        sampled_paths = set(dataset_info['dtd_foreground_sample_paths'])
        with (output_dir / 'foreground_input_images.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=['path', 'sampled_from_dtd'])
            writer.writeheader()
            writer.writerows({'path': str(path), 'sampled_from_dtd': str(path) in sampled_paths} for path in paths)
    if items:
        write_image_classes_csv(output_dir / 'image_prefilter_classes.csv', items)
    write_image_classes_csv(output_dir / 'background_luminance_classes.csv', bg_luminance_items)
    if not settings.apply_defocus:
        print('WARN: --no-defocusは背景順位差フィルタを無効化しません。旧方式の黒補完は人工的な左右差を生む可能性があります。', flush=True)
    records, centers, frequency, ranks = classify_background_eyes(bg_luminance_items, settings, calibration)
    same_class_counts, insufficient = assign_background_pools(records, settings)
    write_background_pool_manifest(output_dir, records, same_class_counts, insufficient, settings)
    write_background_reports(output_dir, records, centers, frequency, ranks, settings, calibration)
    config.update(low_background_same_class_counts=same_class_counts,
                  low_background_insufficient_classes=insufficient,
                  background_clustering_complete=True,
                  high_background_candidate_count=sum(r.high_candidate for r in records),
                  low_background_candidate_count=sum(r.low_candidate for r in records),
                  background_pool_selection_complete=not bool(insufficient),
                  luminance_filter_before_feature_extraction=True)
    save_config()
    if insufficient:
        print(
            f'WARN: low背景が各クラス5枚に満たないクラスは実枚数全件を使用します: {insufficient}',
            flush=True,
        )
    accepted = {record.background_path: record for record in records if record.accepted}
    bg_items = [item for item in bg_luminance_items if str(item.path) in accepted]
    config.update(background_candidate_count=len(bg_items),
                  evaluated_pair_count=0 if args.background_only else len(fg_items) * len(bg_items),
                  background_clustering_complete=True)
    save_config()
    print(f'BG source={len(background_paths)}, bright={config["background_bright_count"]}, high={config["high_background_candidate_count"]}, low={config["low_background_candidate_count"]} (5 per class)', flush=True)
    if args.background_only:
        config.update(complete=True, model_evaluation_performed=False)
        save_config(); print(f'Background reports saved: {output_dir}', flush=True)
        return
    if len(fg_items) < 20 or config['high_background_candidate_count'] < 10:
        raise ValueError(f'重複なし選定に候補が不足。FG={len(fg_items)}, high_BG={config["high_background_candidate_count"]} (minimum 10)。low_BG={config["low_background_candidate_count"]}。条件は緩和しません')
    device = torch.device(settings.device)
    model = load_visibility_model(settings.model, device)
    fg_cache = {item.path: prepare_foreground(item.path, settings.image_size_px) for item in fg_items}
    # 全BGの画像配列は保持しない。BGを外側ループにし、保存用キャッシュも4件まで。
    @lru_cache(maxsize=4)
    def get_background(path):
        record = accepted[str(path)]
        if source_hash(path) != record.source_sha256:
            raise RuntimeError(f'分類後に背景が変更されました: {path}')
        panorama = prepare_background_panorama(path, settings.image_size_px)
        eyes = prepare_scoring_backgrounds(panorama, calibration, settings)
        if source_hash(path) != record.source_sha256:
            raise RuntimeError(f'背景処理中に画像が変更されました: {path}')
        return panorama, eyes

    total = len(fg_items) * len(bg_items)
    results: list[PairResult] = []
    print(
        f"FG candidates={len(fg_items)}, BG candidates={len(bg_items)}, pairs={total}, "
        f"disparity={settings.disparity_deg:.4f}deg ({settings.disparity_px}px)"
    )
    print(
        f"Background defocus={settings.apply_defocus}, "
        f"pupil={settings.pupil_diameter_mm}mm, "
        f"diopter_difference={settings.diopter_difference:.4f}D, "
        f"sigma={settings.blur_sigma_deg:.6f}deg ({settings.blur_sigma_px:.4f}px), "
        f"eye_boundary={settings.eye_boundary_mode}"
    )
    index = 0
    for bg_item in bg_items:
        _, (right_bg, left_bg) = get_background(bg_item.path)
        record = accepted[str(bg_item.path)]
        for fg_item in fg_items:
            index += 1
            result = evaluate_pair(fg_item, bg_item, fg_cache[fg_item.path], right_bg, left_bg,
                                   calibration, model, settings, device)
            result.background_right_texture_class = record.right_rank
            result.background_left_texture_class = record.left_rank
            result.background_texture_rank_gap = record.rank_gap
            result.background_candidate_group = record.candidate_group
            results.append(result)
            print(f'[{index}/{total}] Tex{result.foreground_texture_class}: {fg_item.path.name} x {bg_item.path.name} = {result.absolute_score_difference:.6f}', flush=True)

    results.sort(key=lambda row: (
        row.foreground_texture_class, -row.absolute_score_difference
    ))
    write_csv(output_dir / "all_pairs.csv", results)
    selected = select_high_low_pairs(
        results, args.selection_pool_size, args.max_clip_ratio,
        min_background_rank_gap=settings.min_background_rank_gap,
    )
    write_csv(output_dir / "top20.csv", selected)

    high_results = sorted(
        (row for row in selected if row.difference_group == "high"),
        key=lambda row: row.group_rank if row.group_rank is not None else math.inf,
    )
    low_results = sorted(
        (row for row in selected if row.difference_group == "low"),
        key=lambda row: row.group_rank if row.group_rank is not None else math.inf,
    )
    if len(high_results) != TEXTURE_CLASS_COUNT:
        raise RuntimeError(
            f"highの選定数が{TEXTURE_CLASS_COUNT}ではありません: "
            f"{len(high_results)}"
        )
    if len(low_results) != TEXTURE_CLASS_COUNT:
        raise RuntimeError(
            f"lowの選定数が{TEXTURE_CLASS_COUNT}ではありません: "
            f"{len(low_results)}"
        )
    write_csv(output_dir / "high10.csv", high_results)
    write_csv(output_dir / "low10.csv", low_results)

    for result in selected:
        fg_path = Path(result.foreground_path)
        bg_path = Path(result.background_path)
        if source_hash(bg_path) != accepted[str(bg_path)].source_sha256:
            raise RuntimeError(f'保存前に背景が変更されました: {bg_path}')
        panorama, eyes = get_background(bg_path)
        save_selected_pair(output_dir, result, fg_cache[fg_path], panorama,
                           calibration, model, settings, device, eye_backgrounds=eyes)
    config.update(complete=True, model_evaluation_performed=True, selected_pair_count=len(selected))
    save_config()
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()