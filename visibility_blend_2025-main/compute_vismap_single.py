#!/usr/bin/env python3
"""compute_vismap_single.py

Minimal utility to compute and save a visibility map for a single
background/foreground pair and model.

Usage example
------------
python compute_vismap_single.py \
    --model vismlp_norm \
    --bg imgs/parasol.png \
    --fg imgs/boy.png \
    --alpha_map imgs/boy_vis_50_30.png \
    --out_blend vismaps/blend.png \
    --out_gray vismaps/vismap_gray.png \
    --out_heat vismaps/vismap_heat.png
------------
python compute_vismap_single.py \
    --model vismlp_norm \
    --bg imgs/parasol.png \
    --fg imgs/boy.png \
    --alpha 0.5 \
    --out_blend vismaps/blend.png \
    --out_gray vismaps/vismap_gray.png \
    --out_heat vismaps/vismap_heat.png

If the foreground lacks an alpha channel, supply either --alpha (scalar in
[0,1]) or --alpha_map (grayscale image path). Batch processing and automatic
resizing are intentionally omitted for clarity.
"""

import argparse
import os
from typing import Optional

import cv2
import numpy as np
import torch

from vismodel.utils import load_vismodel  # type: ignore


def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return img


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.repeat(img[:, :, None], 3, axis=2)
    if img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)
    if img.shape[2] == 4:
        return img[:, :, :3]
    return img


def np_to_tensor(img: np.ndarray) -> torch.Tensor:
    if img.ndim == 2:
        img = img[:, :, None]
    tensor = torch.from_numpy(img.transpose(2, 0, 1))
    return tensor.unsqueeze(0)


def save_vismap(vismap: torch.Tensor, path: str) -> None:
    vis_np = vismap.squeeze().cpu().numpy()
    vis_norm = np.clip(vis_np, 0.0, 1.0)
    vis_img = (vis_norm * 255).astype(np.uint8)
    cv2.imwrite(path, vis_img)


def save_vismap_color(vismap: torch.Tensor, path: str, cmap: str) -> None:
    import matplotlib.pyplot as plt

    vis_np = vismap.squeeze().cpu().numpy()
    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.imshow(vis_np, cmap=cmap, vmin=0, vmax=1)
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_bgr(img_bgr: np.ndarray, path: str) -> None:
    cv2.imwrite(path, (np.clip(img_bgr, 0, 1) * 255).astype(np.uint8))


def read_alpha_map(path: str, expected_shape: tuple[int, int]) -> np.ndarray:
    alpha_img = read_image(path)
    if alpha_img.ndim == 3:
        alpha_img = alpha_img[:, :, 0]
    if alpha_img.shape[0] != expected_shape[0] or alpha_img.shape[1] != expected_shape[1]:
        raise ValueError("Alpha map shape does not match background image.")
    return np.clip(alpha_img, 0.0, 1.0)


def solve_alpha(
    fg_img: np.ndarray,
    alpha_scalar: Optional[float],
    alpha_map_path: Optional[str],
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if fg_img.ndim != 3:
        raise ValueError("Foreground image must have 3 or 4 channels.")

    if fg_img.shape[2] == 4:
        fg_bgr = fg_img[:, :, :3]
        alpha = fg_img[:, :, 3]
    else:
        fg_bgr = fg_img
        alpha = None

    if alpha_map_path is not None:
        alpha = read_alpha_map(alpha_map_path, expected_shape)
    elif alpha is None and alpha_scalar is not None:
        alpha = np.full(expected_shape, alpha_scalar, dtype=np.float32)
    elif alpha is None:
        raise ValueError("No alpha information available. Provide RGBA foreground, --alpha, or --alpha_map.")

    if alpha_scalar is not None and alpha_map_path is not None:
        alpha = np.clip(alpha * alpha_scalar, 0.0, 1.0)

    if alpha.shape != expected_shape:
        raise ValueError("Alpha map shape does not match background image.")

    return fg_bgr, np.clip(alpha, 0.0, 1.0)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute a visibility map for a single foreground/background pair.")
    parser.add_argument("--model", required=True, help="Model name defined in vismodel_configs.json.")
    parser.add_argument("--bg", required=True, help="Background image path (PNG/JPG).")
    parser.add_argument("--fg", required=True, help="Foreground image path (PNG/JPG/PNG with alpha).")
    parser.add_argument("--alpha", type=float, help="Uniform alpha value or multiplier in [0,1].")
    parser.add_argument("--alpha_map", help="Grayscale alpha map image path (same size as background).")
    parser.add_argument("--blend", help="Optional pre-blended image path (if not provided, will be computed from fg/bg/alpha).")
    parser.add_argument("--out_gray", required=True, help="Output path for grayscale visibility map (PNG).")
    parser.add_argument("--out_heat", help="Optional output path for colored heatmap (PNG).")
    parser.add_argument("--out_blend", help="Optional path to save the blended image.")
    parser.add_argument("--cmap", default="viridis", help="Colormap name for heatmap output (default: viridis).")
    parser.add_argument("--device", default="cpu", help="Torch device specifier, e.g., 'cpu' or 'cuda:0'.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    bg_img = read_image(args.bg)
    bg_bgr = ensure_bgr(bg_img)

    fg_img = read_image(args.fg)
    expected_shape = bg_bgr.shape[:2]
    if fg_img.shape[0] != expected_shape[0] or fg_img.shape[1] != expected_shape[1]:
        raise ValueError("Foreground and background images must share the same spatial resolution.")

    fg_bgr, alpha = solve_alpha(fg_img, args.alpha, args.alpha_map, expected_shape)

    mask_np = (alpha > 1e-3).astype(np.float32)

    blend_np = None
    if args.blend:
        blend_img = read_image(args.blend)
        if blend_img.shape[0] != expected_shape[0] or blend_img.shape[1] != expected_shape[1]:
            raise ValueError("Blend image shape does not match background image.")
        blend_np = ensure_bgr(blend_img)
    else:
        blend_np = fg_bgr * alpha[..., None] + bg_bgr * (1.0 - alpha[..., None])

    device = torch.device(args.device)
    vismodel = load_vismodel(args.model, device, load_param=True)
    vismodel.eval()

    bg_tensor = np_to_tensor(bg_bgr).to(device)
    fg_tensor = np_to_tensor(fg_bgr).to(device)
    alpha_tensor = np_to_tensor(alpha[..., None]).to(device)
    mask_tensor = np_to_tensor(mask_np[..., None]).to(device)

    with torch.no_grad():
        vismodel.set_inputs_tg_ref_alphamap(
            fg_tensor,
            bg_tensor,
            alpha_tensor,
            mask_tensor,
            blend_mode="linear",
        )
        vismodel.compute_weights()
        vismodel.compute_visibility_wo_weight()

    vismap = vismodel.norm_vismap
    save_vismap(vismap, args.out_gray)

    if args.out_heat:
        save_vismap_color(vismap, args.out_heat, args.cmap)

    if args.out_blend:
        if blend_np is None:
            blend_np = fg_bgr * alpha[..., None] + bg_bgr * (1.0 - alpha[..., None])
        save_bgr(blend_np, args.out_blend)


if __name__ == "__main__":
    main()



