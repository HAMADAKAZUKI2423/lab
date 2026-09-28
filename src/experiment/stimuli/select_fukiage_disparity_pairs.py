#!/usr/bin/env python3
"""低輝度前景をテクスチャ10クラスに分け、全前景×背景を評価する。

各テクスチャクラスから、異なる前景を使う high 1組 / low 1組を選ぶ。
high は左右眼の平均視認性スコア差を最大化し、low は最小化する。
前景・背景は最終20組内で重複させない。

実行例（lab/src から）:
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
MCGILL_IMAGE_DIR = (
    LAB_ROOT / "data" / "raw" / "images"
    / "McGill Calibrated Color Image Database" / "Textures"
)
DTD_IMAGE_DIR = (
    LAB_ROOT / "data" / "raw" / "images"
    / "Describable Textures Dataset" / "images"
)
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LUMINANCE_CLASS_COUNT = 3
TEXTURE_CLASS_COUNT = 10
LAPLACIAN_LEVELS = 3
KMEANS_RANDOM_STATE = 42
DTD_SAMPLES_PER_CATEGORY = 10
DTD_SAMPLING_RANDOM_SEED = 42


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
    apply_defocus: bool = False


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ipd-mm", type=float, default=60.0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--model", default="vismlp_norm")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--image-dir", type=Path, action="append", dest="image_dirs",
        help=("任意の入力画像ルート。複数回指定できます。指定した場合は、その"
              "ディレクトリ群を全件使用し、既定のDTDカテゴリ抽出は行いません"),
    )
    parser.add_argument(
        "--dtd-samples-per-category", type=int,
        default=DTD_SAMPLES_PER_CATEGORY,
        help="DTDの各カテゴリフォルダから抽出する画像数（既定: 10）",
    )
    parser.add_argument(
        "--dtd-sampling-seed", type=int,
        default=DTD_SAMPLING_RANDOM_SEED,
        help="DTDカテゴリ内ランダム抽出の乱数シード（既定: 42）",
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
    if not (0 < fg_cm < bg_cm and angle > 0 and l_fg > 0 and l_bg > 0):
        raise ValueError("距離・視角・輝度設定が不正です")
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
    )


def discover_images_recursive(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def sample_dtd_images_by_category(
    directory: Path, samples_per_category: int, random_seed: int,
) -> tuple[list[Path], dict[str, int]]:
    """DTDの各直下カテゴリから、再現可能な乱数で同数を非復元抽出する。"""
    if samples_per_category < 1:
        raise ValueError("--dtd-samples-per-categoryは1以上にしてください")
    categories = sorted(
        (path for path in directory.iterdir() if path.is_dir()),
        key=lambda path: str(path).casefold(),
    )
    if not categories:
        raise ValueError(f"DTDカテゴリフォルダがありません: {directory}")
    rng = np.random.default_rng(random_seed)
    selected: list[Path] = []
    category_counts: dict[str, int] = {}
    for category in categories:
        candidates = discover_images_recursive(category)
        category_counts[category.name] = len(candidates)
        if len(candidates) < samples_per_category:
            raise ValueError(
                f"DTDカテゴリ {category.name} は{len(candidates)}枚しかありません。"
                f"{samples_per_category}枚を非復元抽出できません"
            )
        indices = rng.choice(
            len(candidates), size=samples_per_category, replace=False
        )
        selected.extend(candidates[int(index)] for index in sorted(indices.tolist()))
    return selected, category_counts


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


def classify_images(paths: list[Path], settings: RunSettings) -> list[ImageItem]:
    """輝度を等数3分割し、低輝度前景候補だけをテクスチャ10クラス化する。"""
    if len(paths) < settings.luminance_class_count * settings.texture_class_count:
        raise ValueError("画像数が少なすぎます")
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise RuntimeError('python -m pip install "scikit-learn" を実行してください') from exc

    measured: list[tuple[Path, float, np.ndarray]] = []
    for path in paths:
        bgr = read_bgr(path)
        measured.append((
            path,
            mean_relative_luminance_from_bgr(bgr),
            compute_laplacian_mad(bgr, settings.laplacian_levels),
        ))

    order = sorted(
        range(len(measured)),
        key=lambda i: (measured[i][1], str(measured[i][0]).casefold()),
    )
    luminance_classes = np.empty(len(measured), dtype=np.int32)
    for class_number, indices in enumerate(
        np.array_split(np.asarray(order), settings.luminance_class_count), start=1
    ):
        luminance_classes[indices] = class_number

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
            background_candidate=lum_class in (2, 3),
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


def make_eye_backgrounds(
    panorama: np.ndarray, output_width: int, disparity_px: int
) -> tuple[np.ndarray, np.ndarray]:
    right_start = (panorama.shape[1] - output_width) // 2
    left_start = right_start + disparity_px
    return (
        crop_with_black(panorama, right_start, output_width),
        crop_with_black(panorama, left_start, output_width),
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
    )


def _image_key(path: str) -> str:
    return str(Path(path).resolve()).casefold()


def select_high_low_pairs(
    results: list[PairResult], pool_size: int, max_clip_ratio: float
) -> list[PairResult]:
    """各クラスhigh/low各1組、前景・背景重複なしで20組をMILP選定する。"""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    if pool_size < 0:
        raise ValueError("--selection-pool-sizeは0以上にしてください")
    if not 0 <= max_clip_ratio <= 1:
        raise ValueError("--max-clip-ratioは0以上1以下にしてください")

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
        low_rows = ordered if pool_size == 0 else ordered[:pool_size]
        high_rows = ordered if pool_size == 0 else ordered[-pool_size:]
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
) -> None:
    assert result.rank is not None
    pair_dir = output_dir / "top20" / f"rank_{result.rank:02d}"
    pair_dir.mkdir(parents=True, exist_ok=False)
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

    dtd_category_counts: dict[str, int] = {}
    if use_default_datasets:
        mcgill_paths = discover_images_recursive(image_dirs[0])
        dtd_paths, dtd_category_counts = sample_dtd_images_by_category(
            image_dirs[1],
            args.dtd_samples_per_category,
            args.dtd_sampling_seed,
        )
        paths = sorted(
            {path.resolve() for path in mcgill_paths + dtd_paths},
            key=lambda path: str(path).casefold(),
        )
    else:
        mcgill_paths = []
        dtd_paths = []
        paths = sorted(
            {
                path.resolve()
                for directory in image_dirs
                for path in discover_images_recursive(directory)
            },
            key=lambda path: str(path).casefold(),
        )
    if not paths:
        raise FileNotFoundError(
            "画像がありません: " + ", ".join(str(path) for path in image_dirs)
        )
    try:
        from scipy.optimize import milp  # noqa: F401
    except ImportError as exc:
        raise RuntimeError('python -m pip install "scipy>=1.9" を実行してください') from exc

    items = classify_images(paths, settings)
    fg_items = [item for item in items if item.foreground_candidate]
    bg_items = [item for item in items if item.background_candidate]
    if len(fg_items) < 20 or len(bg_items) < 20:
        raise ValueError(f"重複なし20組に候補が不足しています: FG={len(fg_items)}, BG={len(bg_items)}")

    output_dir = args.output_dir or (
        DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    image_config = create_image_config()
    calibration = load_display_calibration(image_config.display_dir)
    require_calibration(calibration)
    device = torch.device(settings.device)
    model = load_visibility_model(settings.model, device)

    # 現行実装と同様に前処理画像をキャッシュする。メモリ不足時は画像サイズを下げて確認する。
    fg_cache = {
        item.path: prepare_foreground(item.path, settings.image_size_px)
        for item in fg_items
    }
    panorama_cache = {
        item.path: prepare_background_panorama(item.path, settings.image_size_px)
        for item in bg_items
    }
    eye_bg_cache = {
        item.path: make_eye_backgrounds(
            panorama_cache[item.path], settings.image_size_px, settings.disparity_px
        )
        for item in bg_items
    }

    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump({
            **asdict(settings),
            "image_dirs": [str(path) for path in image_dirs],
            "image_count": len(paths),
            "dataset_image_counts": (
                {
                    "mcgill_all": len(mcgill_paths),
                    "dtd_selected": len(dtd_paths),
                }
                if use_default_datasets
                else {"custom_all": len(paths)}
            ),
            "dtd_sampling_applied": use_default_datasets,
            "dtd_samples_per_category": (
                args.dtd_samples_per_category if use_default_datasets else None
            ),
            "dtd_sampling_seed": (
                args.dtd_sampling_seed if use_default_datasets else None
            ),
            "dtd_category_counts_before_sampling": dtd_category_counts,
            "foreground_candidate_count": len(fg_items),
            "background_candidate_count": len(bg_items),
            "evaluated_pair_count": len(fg_items) * len(bg_items),
            "selection_pool_size": args.selection_pool_size,
            "max_clip_ratio": args.max_clip_ratio,
            "selection_rule": (
                "10 texture classes; one high and one low per class; "
                "different foregrounds; unique foregrounds/backgrounds globally"
            ),
            "display_dir": str(image_config.display_dir),
            "visibility_repo": str(VISIBILITY_REPO),
        }, file, ensure_ascii=False, indent=2)
    write_image_classes_csv(output_dir / "image_prefilter_classes.csv", items)

    total = len(fg_items) * len(bg_items)
    results: list[PairResult] = []
    print(
        f"FG candidates={len(fg_items)}, BG candidates={len(bg_items)}, pairs={total}, "
        f"disparity={settings.disparity_deg:.4f}deg ({settings.disparity_px}px)"
    )
    index = 0
    for fg_item in fg_items:
        foreground = fg_cache[fg_item.path]
        for bg_item in bg_items:
            index += 1
            right_bg, left_bg = eye_bg_cache[bg_item.path]
            result = evaluate_pair(
                fg_item, bg_item, foreground, right_bg, left_bg,
                calibration, model, settings, device,
            )
            results.append(result)
            print(
                "[{}/{}] Tex{}: {} x {} = {:.6f}".format(
                    index,
                    total,
                    result.foreground_texture_class,
                    fg_item.path.name,
                    bg_item.path.name,
                    result.absolute_score_difference,
                )
            )

    results.sort(key=lambda row: (
        row.foreground_texture_class, -row.absolute_score_difference
    ))
    write_csv(output_dir / "all_pairs.csv", results)
    selected = select_high_low_pairs(
        results, args.selection_pool_size, args.max_clip_ratio
    )
    write_csv(output_dir / "top20.csv", selected)

    for result in selected:
        fg_path = Path(result.foreground_path)
        bg_path = Path(result.background_path)
        save_selected_pair(
            output_dir, result,
            fg_cache[fg_path], panorama_cache[bg_path],
            calibration, model, settings, device,
        )
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()