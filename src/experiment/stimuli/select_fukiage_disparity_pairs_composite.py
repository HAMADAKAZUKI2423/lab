#!/usr/bin/env python3
"""左右で異なるテクスチャクラスの画像を人工合成した背景でhigh5/low5を選定する。
lab/srcから:
  py -m experiment.stimuli.select_fukiage_disparity_pairs_composite --pupil-diameter-mm 2.2 --device cuda:0
  # 背景作成・左右像保存のみは --background-only
- 全背景元画像をラプラシアン帯域構成比+平均Yで10クラス化（従来と同じ周波数順位）。
- high: 順位差5の組(1-6,2-7,3-8,4-9,5-10)ごとに5枚。元画像は重複なしでランダム、左右配置もランダム。
  縦横比を保つ中央クロップで横に並べ、右端から全体幅の--overlap-center-from-right(既定1/3)を重なり中心とし、全体幅の--overlap-ratio(既定0.2)をcosineで接続。
- low: 未使用元画像の通常背景で、右眼像と左眼像のクラス差<=1。右眼クラスごと3枚。
- 前景抽出・総当たり・high5/low5選定は従来どおり。前景・背景を含め画像の重複なし。
"""
import argparse
import csv
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from experiment.stimuli import select_fukiage_disparity_pairs as base

K = 10
PAIRS = [(k, k + 5) for k in range(1, 6)]


def write_rows(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def center_crop(bgr, width, height):
    """縦横比を保って中央クロップし、伸縮なしの比率で指定サイズへ縮小する。"""
    h, w = bgr.shape[:2]
    if w * height > h * width:
        cw, ch = round(h * width / height), h
    else:
        cw, ch = w, round(w * height / width)
    x, y = (w - cw) // 2, (h - ch) // 2
    return cv2.resize(bgr[y:y + ch, x:x + cw], (width, height), interpolation=cv2.INTER_AREA)


def compose(first, second, calibration, overlap):
    """線形RGBで幅の異なる2枚を横に並べ、重なり部分をraised-cosineのαで移行する。"""
    lw, rw = first.shape[1], second.shape[1]
    lin = [base.apply_gamma(img[..., ::-1], calibration.gamma_bg) for img in (first, second)]
    width = lw + rw - overlap
    weight = np.zeros(width)
    weight[:lw - overlap] = 1
    u = (np.arange(overlap) + 0.5) / overlap
    weight[lw - overlap:lw] = 0.5 * (1 + np.cos(np.pi * u))
    a = np.zeros((first.shape[0], width, 3)); b = np.zeros_like(a)
    a[:, :lw] = lin[0]; b[:, lw - overlap:] = lin[1]
    out = weight[None, :, None] * a + (1 - weight[None, :, None]) * b
    return base.invert_gamma(out, calibration.gamma_bg)[..., ::-1].astype(np.float32)


def fit_classes(features, settings):
    """共通標準化+KMeans(10)。順位は従来どおり帯域構成比の周波数指標で付ける。"""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    x = np.asarray(features, dtype=np.float64)
    scaler = StandardScaler().fit(x)
    model = KMeans(n_clusters=K, random_state=settings.kmeans_random_state, n_init=10)
    model.fit(scaler.transform(x))
    centers = scaler.inverse_transform(model.cluster_centers_)
    frequency = centers[:, :-1] @ np.arange(settings.laplacian_levels, -1, -1, dtype=float)
    order = np.argsort(frequency, kind='stable')
    if np.any(np.diff(frequency[order]) <= 1e-9):
        raise ValueError('周波数指標が同じクラスタがあり順位を定義できません')
    ranks = {int(label): rank for rank, label in enumerate(order, start=1)}
    return lambda v: ranks[int(model.predict(scaler.transform(np.asarray(v)[None]))[0])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overlap-ratio', type=float, default=0.2)
    parser.add_argument('--overlap-center-from-right', type=float, default=1 / 3)
    parser.add_argument('--composites-per-pair', type=int, default=5)
    parser.add_argument('--low-per-class', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--image-dir', dest='image_dirs', type=Path, action='append')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--ipd-mm', type=float, default=60.0)
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--model', default='vismlp_norm')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--pupil-diameter-mm', type=float)
    parser.add_argument('--no-defocus', action='store_true')
    parser.add_argument('--background-only', action='store_true')
    parser.add_argument('--selection-pool-size', type=int, default=0)
    parser.add_argument('--max-clip-ratio', type=float, default=1.0)
    parser.add_argument('--no-model-difference', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--no-laplacian-difference', action='store_true', help=argparse.SUPPRESS)
    parser.set_defaults(background_clusters=7, min_background_rank_gap=0)
    args = parser.parse_args()
    r, c = args.overlap_ratio, args.overlap_center_from_right
    if not (0 < r and r / 2 < c < 1 - r / 2):
        parser.error('重なり範囲が全体幅に収まりません')
    settings = replace(base.load_settings(args), texture_class_count=5)
    rng = np.random.default_rng(args.seed)
    n = settings.image_size_px
    total = round(n * settings.background_width_ratio)
    overlap = max(2, round(total * args.overlap_ratio))
    left_w = round(total * (1 - args.overlap_center_from_right) + overlap / 2)
    right_w = total + overlap - left_w
    tile = left_w  # 分類用クロップ幅（左タイル幅）
    image_dirs = [p.expanduser().resolve() for p in
                  (args.image_dirs or (base.MCGILL_IMAGE_DIR, base.DTD_IMAGE_DIR))]
    _, background_paths, dataset_info = base.collect_image_candidate_paths(
        image_dirs, use_default_datasets=not args.image_dirs, background_only=True)
    output = args.output_dir or (base.DEFAULT_OUTPUT_ROOT /
                                ('composite_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    output.mkdir(parents=True, exist_ok=False)
    display_dir = base.create_image_config().display_dir
    calibration = base.load_display_calibration(display_dir)
    base.require_calibration(calibration)
    levels = settings.laplacian_levels
    # 1) 元画像（合成時と同じ中央クロップ）を10クラスに分類・順位付け。
    sources = {}
    for i, path in enumerate(background_paths, 1):
        crop = center_crop(base.read_bgr(path), tile, n)
        feature = base.background_frequency_features(crop, calibration, levels)
        if feature is not None:
            sources[str(path)] = (crop, feature)
        print(f'Source features [{i}/{len(background_paths)}]', flush=True)
    if len(sources) < K:
        raise RuntimeError(f'有効なSource featuresがKMeansのクラス数未満です: {len(sources)} < {K}')
    print(f'[1/4] Source features完了: valid={len(sources)}。KMeansを開始します', flush=True)
    try:
        predict = fit_classes([f for _, f in sources.values()], settings)
    except Exception as exc:
        raise RuntimeError('[1/4] KMeansによる10クラス分類に失敗しました') from exc
    print('[1/4] KMeans完了。各画像のクラス予測を開始します', flush=True)
    source_rank = {}
    source_items = list(sources.items())
    for i, (path, (_, feature)) in enumerate(source_items, 1):
        try:
            source_rank[path] = predict(feature)
        except Exception as exc:
            raise RuntimeError(
                f'[1/4] Source class予測に失敗 [{i}/{len(source_items)}]: {path}'
            ) from exc
        if i == 1 or i % 100 == 0 or i == len(source_items):
            print(f'[1/4] Source classes [{i}/{len(source_items)}]', flush=True)
    write_rows(output / 'source_classes.csv',
               [dict(source_path=p, texture_rank=r) for p, r in source_rank.items()])
    print(f'[1/4] source_classes.csv保存完了: {output}', flush=True)
    used, candidates = set(), []

    def add_candidate(group, name, panorama, meta):
        eyes = base.prepare_scoring_backgrounds(panorama, calibration, settings)
        ranks = [predict(base.background_frequency_features(v, calibration, levels)) for v in eyes]
        directory = output / 'background_candidates' / group / name
        directory.mkdir(parents=True)
        path = directory / 'background_panorama.png'
        base.save_bgr(path, panorama)
        for eye, view in zip(('right', 'left'), eyes):
            base.save_bgr(directory / f'{eye}_background.png', view)
        item = base.ImageItem(path, base.mean_relative_luminance_from_bgr(panorama), 0,
                              (), None, None, None, False, True)
        candidates.append(dict(group=group, name=name, item=item, panorama=panorama, eyes=eyes,
                               right_rank=ranks[0], left_rank=ranks[1], **meta))

    # 2) high: 順位差5の組ごとに重複なし・左右ランダムで合成。
    high_total = len(PAIRS) * args.composites_per_pair
    high_done = 0
    print(f'[2/4] high合成背景の生成を開始: total={high_total}', flush=True)
    for low_rank, high_rank in PAIRS:
        pools = []
        for rank in (low_rank, high_rank):
            pool = sorted(p for p, r in source_rank.items() if r == rank and p not in used)
            if len(pool) < args.composites_per_pair:
                raise ValueError(f'クラス{rank}の元画像が不足しています: {len(pool)}')
            pick = [pool[i] for i in rng.choice(len(pool), args.composites_per_pair, replace=False)]
            used.update(pick); pools.append(pick)
        for index, (a, b) in enumerate(zip(*pools), 1):
            low_left = bool(rng.random() < 0.5)
            first, second = (a, b) if low_left else (b, a)
            panorama = compose(center_crop(base.read_bgr(Path(first)), left_w, n),
                               center_crop(base.read_bgr(Path(second)), right_w, n), calibration, overlap)
            try:
                add_candidate('high', f'pair{low_rank}-{high_rank}_{index:02d}', panorama, dict(
                    left_source=first, right_source=second,
                    left_source_rank=source_rank[first], right_source_rank=source_rank[second],
                    low_class_side='left' if low_left else 'right', overlap_px=overlap))
            except Exception as exc:
                raise RuntimeError(
                    f'[2/4] high背景処理に失敗: pair={low_rank}-{high_rank}, index={index}'
                ) from exc
            high_done += 1
            print(f'[2/4] High candidates [{high_done}/{high_total}]', flush=True)
    print('[2/4] high合成背景の生成完了', flush=True)

    # 3) low: 未使用元画像の通常背景で左右クラス差<=1、右眼クラスごと3枚。
    by_class = {rank: [] for rank in range(1, K + 1)}
    low_paths = sorted(p for p in sources if p not in used)
    print(f'[3/4] low候補の全画像走査を開始: total={len(low_paths)}', flush=True)
    for i, path in enumerate(low_paths, 1):
        if i == 1 or i % 10 == 0 or i == len(low_paths):
            print(f'[3/4] Low candidate scan [{i}/{len(low_paths)}]: {path}', flush=True)
        try:
            panorama = base.prepare_background_panorama(
                Path(path), n, settings.background_width_ratio
            )
            eyes = base.prepare_scoring_backgrounds(panorama, calibration, settings)
            feats = [
                base.background_frequency_features(v, calibration, levels)
                for v in eyes
            ]
            if any(f is None for f in feats):
                print(f'WARN: low候補の特徴抽出をスキップ: {path}', flush=True)
                continue
            right_rank, left_rank = (predict(f) for f in feats)
        except Exception as exc:
            raise RuntimeError(
                f'[3/4] low候補走査に失敗 [{i}/{len(low_paths)}]: {path}'
            ) from exc
        if abs(right_rank - left_rank) <= 1:
            by_class[right_rank].append(path)
    print(
        f'[3/4] low候補走査完了: accepted={sum(map(len, by_class.values()))}',
        flush=True,
    )
    print('[4/4] low背景候補の保存を開始', flush=True)
    for rank, pool in by_class.items():
        if len(pool) < args.low_per_class:
            print(f'WARN: low右眼クラス{rank}は{len(pool)}枚のみ使用します', flush=True)
        for i, j in enumerate(rng.choice(len(pool), min(len(pool), args.low_per_class), replace=False), 1):
            path = pool[int(j)]; used.add(path)
            try:
                add_candidate('low', f'class{rank:02d}_{i:02d}', base.prepare_background_panorama(
                    Path(path), n, settings.background_width_ratio), dict(
                    left_source=path, right_source=path, left_source_rank=source_rank[path],
                    right_source_rank=source_rank[path], low_class_side='', overlap_px=0))
            except Exception as exc:
                raise RuntimeError(
                    f'[4/4] low背景候補の保存に失敗: class={rank}, index={i}, path={path}'
                ) from exc
            print(f'[4/4] Low candidate saved: class={rank}, index={i}', flush=True)
    print('[4/4] low背景候補の保存完了', flush=True)
    write_rows(output / 'background_candidates.csv', [
        {k: (str(v.path) if k == 'item' else v) for k, v in c.items() if k not in ('panorama', 'eyes')}
        for c in candidates])
    config = {**asdict(settings), **dataset_info, 'mode': 'composite', 'complete': False,
              'texture_class_count_background': K, 'class_pairs': PAIRS,
              'overlap_ratio': args.overlap_ratio, 'overlap_px': overlap, 'overlap_center_from_right': args.overlap_center_from_right,
              'left_tile_width_px': left_w, 'right_tile_width_px': right_w,
              'blend': 'raised_cosine_alpha_in_linear_RGB', 'crop': 'aspect_preserving_center_crop',
              'side': 'random', 'duplicate_images': False, 'seed': args.seed,
              'low_rule': 'abs(right_rank-left_rank)<=1', 'low_per_right_class': args.low_per_class,
              'selection_metric': 'sqrt(var(L-R)+2*mean(L-R)**2)'}
    def save_config():
        (output / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2,
                                                      default=str), encoding='utf-8')
    save_config()
    print(f'背景候補 high={sum(c["group"] == "high" for c in candidates)}, '
          f'low={sum(c["group"] == "low" for c in candidates)}: {output}', flush=True)
    if args.background_only:
        config['complete'] = True; save_config(); return
    # 4) 前景抽出は従来通り。背景に使った元画像は前景から除外。
    foreground_paths, _, foreground_info = base.collect_image_candidate_paths(
        image_dirs, use_default_datasets=not args.image_dirs, background_only=False)
    used_keys = {base._image_key(p) for p in used}
    fg_items = [f for f in base.classify_images(foreground_paths, settings)
                if base._image_key(str(f.path)) not in used_keys]
    base.write_image_classes_csv(output / 'image_prefilter_classes.csv', fg_items)
    foregrounds = {f.path: base.prepare_foreground(f.path, n) for f in fg_items}
    device = torch.device(settings.device)
    model = base.load_visibility_model(settings.model, device)
    by_path, results = {}, []
    for c in candidates:
        by_path[str(c['item'].path)] = c
        right, left = c['eyes']
        for fg in fg_items:
            r = base.evaluate_pair(fg, c['item'], foregrounds[fg.path], right, left,
                                   calibration, model, settings, device)
            r.background_candidate_group = c['group']
            r.background_right_texture_class = c['right_rank']
            r.background_left_texture_class = c['left_rank']
            r.background_texture_rank_gap = abs(c['right_rank'] - c['left_rank'])
            results.append(r)
        print(f'Evaluated {c["name"]}', flush=True)
    chosen, thresholds = base.select_high_low_pairs(
        results, args.selection_pool_size, args.max_clip_ratio,
        use_laplacian_difference=True, use_model_difference=True, balance_foreground_classes=True)
    base.write_csv(output / 'all_pairs.csv', results)
    base.write_csv(output / 'selected_pairs.csv', chosen)
    for group in ('high', 'low'):
        base.write_csv(output / f'{group}5.csv', [r for r in chosen if r.difference_group == group])
    for r in chosen:
        c = by_path[r.background_path]
        base.save_selected_pair(output, r, foregrounds[Path(r.foreground_path)], c['panorama'],
                                calibration, model, settings, device, eye_backgrounds=c['eyes'])
    config.update(foreground_info, complete=True, selected_pair_count=len(chosen),
                  visibility_level_thresholds=dict(zip(('low_mid', 'mid_high'), thresholds)))
    save_config(); print(f'Saved: {output}')


if __name__ == '__main__':
    main()