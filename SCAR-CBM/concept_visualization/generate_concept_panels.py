#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from concept_visualization.common import (
    ROOT,
    build_model,
    choose_sample_indices,
    forward_stage5_spatial_heatmap,
    load_adj,
    load_attribute_names,
    load_cub_train_items,
    load_image_chw,
    normalize_heatmap,
    overlay_rgb,
    save_bar_chart,
    topk_ids_from_two,
    upsample_heatmaps,
)


def label_for_concept(concept_id: int, attribute_names: List[str]) -> str:
    if 0 <= concept_id < len(attribute_names):
        return f"{concept_id}: {attribute_names[concept_id]}"
    return f"{concept_id}"


def save_overlay_panel(
    image_chw: torch.Tensor,
    heatmaps_lowres: torch.Tensor,
    concept_ids: List[int],
    attribute_names: List[str],
    title_prefix: str,
    out_path: str,
    cmap_name: str,
    alpha: float,
) -> None:
    img = image_chw.numpy().transpose(1, 2, 0)
    hi = upsample_heatmaps(heatmaps_lowres[concept_ids], img.shape[:2]).numpy()
    n = len(concept_ids)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows))
    axes = np.atleast_2d(axes)
    for i, cid in enumerate(concept_ids):
        ax = axes[i // cols, i % cols]
        hm = normalize_heatmap(hi[i])
        ax.imshow(overlay_rgb(img, hm, cmap_name, alpha))
        ax.set_title(f"{title_prefix}\n{label_for_concept(cid, attribute_names)}", fontsize=10)
        ax.axis('off')
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis('off')
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_compare_panel(
    image_chw: torch.Tensor,
    stage2_lowres: torch.Tensor,
    stage5_lowres: torch.Tensor,
    concept_ids: List[int],
    attribute_names: List[str],
    out_path: str,
    overlay_cmap: str,
    delta_cmap: str,
    alpha: float,
) -> List[Dict[str, float | int | str]]:
    img = image_chw.numpy().transpose(1, 2, 0)
    stage2_hi = upsample_heatmaps(stage2_lowres[concept_ids], img.shape[:2]).numpy()
    stage5_hi = upsample_heatmaps(stage5_lowres[concept_ids], img.shape[:2]).numpy()
    fig, axes = plt.subplots(len(concept_ids), 3, figsize=(10.5, 3.2 * len(concept_ids)))
    axes = np.atleast_2d(axes)
    rows: List[Dict[str, float | int | str]] = []
    for i, cid in enumerate(concept_ids):
        before = stage2_hi[i]
        after = stage5_hi[i]
        delta = after - before
        before_norm = normalize_heatmap(before)
        after_norm = normalize_heatmap(after)
        delta_scale = float(np.max(np.abs(delta)) + 1e-8)
        label = label_for_concept(cid, attribute_names)

        axes[i, 0].imshow(overlay_rgb(img, before_norm, overlay_cmap, alpha))
        axes[i, 0].set_title(f"Stage2\n{label}", fontsize=10)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(overlay_rgb(img, after_norm, overlay_cmap, alpha))
        axes[i, 1].set_title(f"Stage5\n{label}", fontsize=10)
        axes[i, 1].axis('off')

        im = axes[i, 2].imshow(delta, cmap=delta_cmap, vmin=-delta_scale, vmax=delta_scale)
        axes[i, 2].set_title(f"Delta\n{label}", fontsize=10)
        axes[i, 2].axis('off')
        fig.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)

        rows.append(
            {
                'concept_id': int(cid),
                'concept_name': label,
                'stage2_max': float(before.max()),
                'stage5_max': float(after.max()),
                'delta_abs_max': float(np.abs(delta).max()),
            }
        )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description='生成 SCAR-CBM concept 可视化面板资源')
    parser.add_argument('--cub-root', type=str, required=True)
    parser.add_argument('--attributes', type=str, default=os.path.join(ROOT, 'Data', 'attributes.txt'))
    parser.add_argument('--stage1-out', type=str, required=True)
    parser.add_argument('--stage2-npz', type=str, required=True)
    parser.add_argument('--stage4-npz', type=str, default=None)
    parser.add_argument('--stage5-weights', type=str, required=True)
    parser.add_argument('--num-concepts', type=int, default=312)
    parser.add_argument('--feature-dim', type=int, default=256)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--sample-count', type=int, default=8)
    parser.add_argument('--top-k', type=int, default=6)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--overlay-cmap', type=str, default='turbo')
    parser.add_argument('--delta-cmap', type=str, default='coolwarm')
    parser.add_argument('--alpha', type=float, default=0.45)
    parser.add_argument('--out-dir', type=str, required=True)
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    samples_dir = os.path.join(out_dir, 'samples')
    summary_dir = os.path.join(out_dir, 'summary')
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    attribute_names = load_attribute_names(args.attributes)
    cub_items = load_cub_train_items(args.cub_root)
    device = torch.device(args.device)
    adj = load_adj(device, args.stage1_out, None)
    model = build_model(args.stage5_weights, args.num_concepts, args.feature_dim, args.num_heads, device)

    stage4_indices = None
    if args.stage4_npz:
        with np.load(args.stage4_npz, allow_pickle=False) as z:
            stage4_indices = z['sampled_indices']

    with np.load(args.stage2_npz, allow_pickle=False) as z:
        c_heatmap = z['c_heatmap']
        spatial_stage2 = z['spatial_concept_heatmap']
        selected_indices = choose_sample_indices(c_heatmap, stage4_indices, args.sample_count)

        summary_stage2_counter: Counter[int] = Counter()
        summary_stage5_counter: Counter[int] = Counter()
        sample_rows = []

        for rank, row_idx in enumerate(selected_indices, start=1):
            item = cub_items[row_idx]
            image_chw = load_image_chw(item.image_path, args.image_size)
            stage2_lowres = torch.from_numpy(spatial_stage2[row_idx].astype(np.float32))
            stage5_lowres = forward_stage5_spatial_heatmap(model, image_chw.to(device), adj)
            concept_ids = topk_ids_from_two(stage2_lowres, stage5_lowres, args.top_k)

            for cid in torch.topk(torch.from_numpy(c_heatmap[row_idx]), min(args.top_k, c_heatmap.shape[1])).indices.tolist():
                summary_stage2_counter[int(cid)] += 1
            for cid in torch.topk(stage5_lowres.amax(dim=(1, 2)), min(args.top_k, stage5_lowres.shape[0])).indices.tolist():
                summary_stage5_counter[int(cid)] += 1

            sample_slug = f'{rank:02d}_row_{row_idx:04d}'
            stage2_png = os.path.join(samples_dir, f'{sample_slug}_stage2_overlay.png')
            stage5_png = os.path.join(samples_dir, f'{sample_slug}_stage5_overlay.png')
            compare_png = os.path.join(samples_dir, f'{sample_slug}_compare.png')

            save_overlay_panel(image_chw, stage2_lowres, concept_ids, attribute_names, 'Stage2', stage2_png, args.overlay_cmap, args.alpha)
            save_overlay_panel(image_chw, stage5_lowres, concept_ids, attribute_names, 'Stage5', stage5_png, args.overlay_cmap, args.alpha)
            compare_stats = save_compare_panel(image_chw, stage2_lowres, stage5_lowres, concept_ids, attribute_names, compare_png, args.overlay_cmap, args.delta_cmap, args.alpha)

            sample_rows.append(
                {
                    'row_index': int(row_idx),
                    'image_id': int(item.image_id),
                    'relpath': item.relpath,
                    'concept_ids': [int(x) for x in concept_ids],
                    'concept_labels': [label_for_concept(int(x), attribute_names) for x in concept_ids],
                    'stage2_overlay': os.path.relpath(stage2_png, out_dir),
                    'stage5_overlay': os.path.relpath(stage5_png, out_dir),
                    'compare_panel': os.path.relpath(compare_png, out_dir),
                    'compare_stats': compare_stats,
                }
            )

    top_stage2 = summary_stage2_counter.most_common(12)
    top_stage5 = summary_stage5_counter.most_common(12)
    stage2_bar = os.path.join(summary_dir, 'top_stage2_concepts.png')
    stage5_bar = os.path.join(summary_dir, 'top_stage5_concepts.png')
    save_bar_chart([label_for_concept(cid, attribute_names) for cid, _ in top_stage2], [float(v) for _, v in top_stage2], 'Top Concepts Across Selected Samples - Stage2', stage2_bar)
    save_bar_chart([label_for_concept(cid, attribute_names) for cid, _ in top_stage5], [float(v) for _, v in top_stage5], 'Top Concepts Across Selected Samples - Stage5', stage5_bar)

    manifest = {
        'meta': {
            'cub_root': os.path.abspath(args.cub_root),
            'stage1_out': os.path.abspath(args.stage1_out),
            'stage2_npz': os.path.abspath(args.stage2_npz),
            'stage4_npz': os.path.abspath(args.stage4_npz) if args.stage4_npz else None,
            'stage5_weights': os.path.abspath(args.stage5_weights),
            'sample_count': len(sample_rows),
            'top_k': int(args.top_k),
            'image_size': int(args.image_size),
        },
        'summary': {
            'top_stage2': [{'concept_id': int(cid), 'concept_name': label_for_concept(cid, attribute_names), 'count': int(v)} for cid, v in top_stage2],
            'top_stage5': [{'concept_id': int(cid), 'concept_name': label_for_concept(cid, attribute_names), 'count': int(v)} for cid, v in top_stage5],
            'top_stage2_chart': os.path.relpath(stage2_bar, out_dir),
            'top_stage5_chart': os.path.relpath(stage5_bar, out_dir),
        },
        'samples': sample_rows,
    }

    manifest_path = os.path.join(out_dir, 'manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f'已写入: {manifest_path}')
    print(f'样本数: {len(sample_rows)}')
    print(f'汇总图: {stage2_bar}, {stage5_bar}')


if __name__ == '__main__':
    main()
