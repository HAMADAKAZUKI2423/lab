#!/usr/bin/env python3
"""CSVで再選定し、保存用の視認性マップ等は採用18組だけ再計算する。"""
import argparse
import csv
import hashlib
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

LEVELS = ("low", "mid", "high")
GROUPS = ("high", "low")


def image_key(value):
    return str(Path(value).expanduser().resolve()).casefold()


def select(rows, pool_size, max_clip, use_laplacian, use_model):
    valid = []
    for row in rows:
        try:
            score = float(row["right_visibility_score"])
            l2 = float(row["visibility_map_l2"])
            clip = max(float(row["right_clip_ratio"]), float(row["left_clip_ratio"]))
        except (KeyError, TypeError, ValueError):
            continue
        if all(map(math.isfinite, (score, l2, clip))) and clip <= max_clip:
            row["right_visibility_score"], row["visibility_map_l2"] = score, l2
            valid.append(row)
    if not valid:
        raise ValueError("有効な候補がありません（CSV列・clip率を確認してください）")

    q1, q2 = np.quantile([r["right_visibility_score"] for r in valid], [1/3, 2/3])
    for row in valid:
        score = row["right_visibility_score"]
        row["visibility_level"] = "low" if score <= q1 else "mid" if score <= q2 else "high"

    candidates = []
    for group in GROUPS:
        for level in LEVELS:
            part = [r for r in valid if r["visibility_level"] == level and
                    (not use_laplacian or r.get("background_candidate_group") == group)]
            if use_model:
                part.sort(key=lambda r: r["visibility_map_l2"], reverse=group == "high")
            else:
                part.sort(key=lambda r: hashlib.sha256(
                    f"{group}|{level}|{r['foreground_path']}|{r['background_path']}".encode()
                ).hexdigest())
            if pool_size:
                part = part[:pool_size]
            if len(part) < 3:
                raise ValueError(f"{group}/{level}候補が3組未満です: {len(part)}")
            candidates.extend((group, level, r, (len(part)-i)/len(part)) for i, r in enumerate(part))

    strata = [(g, level) for g in GROUPS for level in LEVELS]
    images = sorted({image_key(p) for _, _, r, _ in candidates
                     for p in (r["foreground_path"], r["background_path"])})
    index = {("s", *key): i for i, key in enumerate(strata)}
    index.update({("i", key): len(strata)+i for i, key in enumerate(images)})
    rr, cc = [], []
    for col, (group, level, row, _) in enumerate(candidates):
        for key in (("s", group, level), ("i", image_key(row["foreground_path"])),
                    ("i", image_key(row["background_path"]))):
            rr.append(index[key]); cc.append(col)
    matrix = coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(len(index), len(candidates))).tocsc()
    lower = np.r_[np.full(len(strata), 3), np.zeros(len(images))]
    upper = np.r_[np.full(len(strata), 3), np.ones(len(images))]
    result = milp(c=-np.asarray([item[3] for item in candidates]),
                  integrality=np.ones(len(candidates)), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(matrix, lower, upper),
                  options={"mip_rel_gap": 0.0})
    if not result.success or result.x is None:
        raise ValueError("制約を満たす18組がありません。候補を確認してください（条件は自動緩和しません）")
    chosen = []
    for (group, _, row, _), value in zip(candidates, result.x):
        if value > .5:
            row["difference_group"] = group
            chosen.append(row)
    chosen.sort(key=lambda r: (GROUPS.index(r["difference_group"]), LEVELS.index(r["visibility_level"]),
                               -r["visibility_map_l2"] if r["difference_group"] == "high" else r["visibility_map_l2"]))
    if len(chosen) != 18 or len({image_key(p) for r in chosen for p in
                                  (r["foreground_path"], r["background_path"])}) != 36:
        raise RuntimeError("18組または前景・背景横断の重複なし条件を満たしません")
    for rank, row in enumerate(chosen, 1):
        row["rank"] = rank
    for group in GROUPS:
        for rank, row in enumerate((r for r in chosen if r["difference_group"] == group), 1):
            row["group_rank"] = rank
    return chosen, (float(q1), float(q2))


def save_pair_artifacts(source, output, chosen, args, thresholds):
    """選定は変えず、元selectの保存関数で採用18組だけを再描画する。"""
    from dataclasses import fields
    from typing import get_args, get_type_hints
    import torch
    from experiment.stimuli import select_fukiage_disparity_pairs as original

    config_path = source.parent / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    missing = {f.name for f in fields(original.RunSettings)} - config.keys()
    if missing:
        raise ValueError(f"元config.jsonの保存用設定が不足しています: {sorted(missing)}")
    if config.get("model_input_mode") != "calibrated_components_linear_alpha":
        raise ValueError("元config.jsonのモデル入力方式に対応していません")
    values = {f.name: config[f.name] for f in fields(original.RunSettings)}
    values["device"] = args.device or config["device"]
    settings = original.RunSettings(**values)
    display_dir = Path(config["display_dir"]).expanduser()
    if not display_dir.is_absolute():
        display_dir = source.parent / display_dir
    for name in ("C.csv", "T_prime.csv", "R_prime.csv", "gamma_bg.csv", "gamma_fg.csv"):
        if not (display_dir / name).is_file():
            raise FileNotFoundError(f"表示校正ファイルがありません: {display_dir / name}")
    calibration = original.load_display_calibration(display_dir)
    original.require_calibration(calibration)
    if "visibility_repo" in config:
        original.VISIBILITY_REPO = Path(config["visibility_repo"]).expanduser()
    device = torch.device(settings.device)
    model = original.load_visibility_model(settings.model, device)
    hints = get_type_hints(original.PairResult)

    def restore_pair(row):
        data = {}
        for field in fields(original.PairResult):
            value = row.get(field.name)
            if value is None or value == "":
                continue
            kind = hints[field.name]
            alternatives = get_args(kind)
            if alternatives:
                kind = next(t for t in alternatives if t is not type(None))
            if kind is bool:
                text = str(value).lower()
                if text not in {"true", "false", "1", "0"}:
                    raise ValueError(f"CSVのbool値が不正です: {field.name}={value}")
                data[field.name] = text in {"true", "1"}
            else:
                data[field.name] = kind(value)
        pair = original.PairResult(**data)
        for key in ("apply_defocus", "pupil_diameter_mm", "disparity_deg", "disparity_px",
                    "diopter_difference", "blur_sigma_deg", "blur_sigma_px", "eye_boundary_mode"):
            actual, expected = getattr(pair, key), getattr(settings, key)
            same = (math.isclose(float(actual), float(expected), rel_tol=1e-7, abs_tol=1e-9)
                    if actual is not None and expected is not None
                    and isinstance(expected, (int, float)) else actual == expected)
            if not same:
                raise ValueError(f"CSVと元config.jsonの設定が不一致です: {key}")
        # Laplacian OFF時の保存用途だけbothと記録。選定結果は変更しない。
        if args.no_laplacian_difference:
            pair.background_candidate_group = "both"
        return pair

    pairs = [restore_pair(row) for row in chosen]
    for pair in pairs:
        for role in ("foreground", "background"):
            path = Path(getattr(pair, role + "_path"))
            if not path.is_file():
                raise FileNotFoundError(f"元画像がありません: {path}")
    config.update(complete=False, mode="selection_from_csv", source_result_dir=str(source.parent),
                  source_csv=str(source), device=settings.device,
                  selection_pool_size=args.selection_pool_size, max_clip_ratio=args.max_clip_ratio,
                  use_laplacian_difference=not args.no_laplacian_difference,
                  use_model_difference=not args.no_model_difference,
                  visibility_level_thresholds={"low_mid": thresholds[0], "mid_high": thresholds[1]},
                  selected_per_stratum=3, selected_pair_count=len(pairs),
                  all_pairs_recomputed=False, visibility_maps_regenerated=True,
                  source_calibration_version_verified=False)
    def write_config():
        (output / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_config()
    shutil.copy2(config_path, output / "source_config.json")
    shutil.copy2(source, output / "all_pairs.csv")
    for pattern in ("background_*.csv", "background_*.json", "background_*.png",
                    "high_background_candidates.csv", "low_background_candidates.csv",
                    "foreground_input_images.csv", "image_prefilter_classes.csv"):
        for path in source.parent.glob(pattern):
            if path.is_file():
                shutil.copy2(path, output / path.name)
    print("保存用に採用18組だけ再計算します（校正・モデル重みは元の評価時と同じものが必要）。")
    for index, pair in enumerate(pairs, 1):
        foreground = original.prepare_foreground(Path(pair.foreground_path), settings.image_size_px)
        panorama = original.prepare_background_panorama(Path(pair.background_path), settings.image_size_px)
        original.save_selected_pair(output, pair, foreground, panorama,
                                    calibration, model, settings, device)
        print(f"保存 [{index}/{len(pairs)}] {pair.difference_group}/{pair.group_rank}", flush=True)
    config.update(complete=True, model_evaluation_performed=True,
                  saved_pair_count=len(pairs), saved_pairs_recomputed=len(pairs))
    write_config()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input-dir", type=Path,
                        help="元のselect_fukiageの結果フォルダ。省略時は選択ダイアログ")
    inputs.add_argument("--input-csv", type=Path,
                        help="all_pairs.csvを直接指定する場合")
    parser.add_argument("--device", help="保存時の計算デバイス。省略時は元config.jsonの値")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--selection-pool-size", type=int, default=0,
                        help="各層の候補上限。0=上限なし（既定）")
    parser.add_argument("--max-clip-ratio", type=float, default=1.0)
    parser.add_argument("--no-laplacian-difference", action="store_true")
    parser.add_argument("--no-model-difference", action="store_true")
    args = parser.parse_args()
    if args.selection_pool_size < 0 or not 0 <= args.max_clip_ratio <= 1:
        parser.error("pool sizeは0以上、clip ratioは0〜1です")
    if args.no_laplacian_difference and args.no_model_difference:
        parser.error("左右差判定を両方OFFにはできません（high/lowの意味がなくなるため）")
    if args.input_csv is not None:
        source = args.input_csv.expanduser().resolve()
    else:
        directory = args.input_dir
        if directory is None:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            try:
                selected = filedialog.askdirectory(
                    parent=root,
                    title="select_fukiageの結果フォルダ（all_pairs.csvがあるフォルダ）を選択",
                    mustexist=True,
                )
            finally:
                root.destroy()
            if not selected:
                print("フォルダ選択をキャンセルしました。")
                return
            directory = Path(selected)
        directory = directory.expanduser().resolve()
        if not directory.is_dir():
            parser.error(f"入力フォルダがありません: {directory}")
        source = directory / "all_pairs.csv"
    if not source.is_file():
        parser.error(f"all_pairs.csvが見つかりません: {source}")
    print(f"入力CSV: {source}")
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"foreground_path", "background_path", "right_visibility_score",
                    "right_clip_ratio", "left_clip_ratio", "visibility_map_l2",
                    "background_candidate_group"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV必須列がありません: {sorted(required - set(reader.fieldnames or []))}")
        rows = list(reader)
    chosen, thresholds = select(rows, args.selection_pool_size, args.max_clip_ratio,
                                not args.no_laplacian_difference,
                                not args.no_model_difference)
    output = (args.output_dir.expanduser().resolve() if args.output_dir else
              source.parent / f"selected_from_csv_{datetime.now():%Y%m%d_%H%M%S}")
    output.mkdir(parents=True, exist_ok=False)
    fields = list(dict.fromkeys([*(reader.fieldnames or []), "difference_group", "visibility_level", "rank", "group_rank"]))
    def write_csv(path, data):
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(data)
    write_csv(output / "selected_pairs.csv", chosen)
    for group in GROUPS:
        selected = [r for r in chosen if r["difference_group"] == group]
        write_csv(output / f"{group}9.csv", selected)
    save_pair_artifacts(source, output, chosen, args, thresholds)
    (output / "selection_config.json").write_text(json.dumps({
        "input_csv": str(source), "candidate_count": len(rows), "selected_count": len(chosen),
        "selection_pool_size": args.selection_pool_size, "max_clip_ratio": args.max_clip_ratio,
        "visibility_tercile_thresholds": {"low_mid": thresholds[0], "mid_high": thresholds[1]},
        "use_laplacian_difference": not args.no_laplacian_difference,
        "use_model_difference": not args.no_model_difference,
        "visibility_recomputed": "selected_18_only", "visibility_maps_regenerated": True,
        "cross_role_image_duplicates": 0,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"18組を保存: {output}")


if __name__ == "__main__":
    main()