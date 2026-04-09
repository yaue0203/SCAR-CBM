#!/usr/bin/env python3
"""
对比两份 CompleteModel 权重的 spatial_concept_heatmap，可视化 before / after / delta。

示例:
  python compare_spatial_heatmaps.py       --before-weights tmp_compare_weights/before.pt       --after-weights tmp_compare_weights/after.pt       --image Data/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg       --stage1-out Stage1/stage1_cub_output       --num-concepts 312       --out-dir tmp_compare_viz
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from visualize_spatial_concept_heatmap import (
    CompleteModel,
    collect_image_paths,
    load_adj,
    load_cub_one,
    load_image_tensor,
    maybe_load_weights,
    overlay_out_path,
    overlay_rgb,
    parse_concept_ids,
    pick_top_k_compare,
    upsample_heatmaps,
    _forward_spatial_heatmaps,
)


def pick_top_k_delta(
    heatmaps_before: torch.Tensor,
    heatmaps_after: torch.Tensor,
    k: int,
) -> List[int]:
    diff = (heatmaps_after - heatmaps_before).abs().amax(dim=(1, 2))
    base = torch.maximum(
        heatmaps_before.amax(dim=(1, 2)),
        heatmaps_after.amax(dim=(1, 2)),
    )
    score = diff + 0.25 * base
    k = min(k, score.shape[0])
    _, idx = torch.topk(score, k)
    return idx.cpu().tolist()


def save_triplet_panel(
    image_chw: torch.Tensor,
    heatmaps_before_k_hw: torch.Tensor,
    heatmaps_after_k_hw: torch.Tensor,
    concept_ids: List[int],
    out_path: str,
    overlay_cmap: str,
    delta_cmap: str,
    alpha: float,
    label_before: str,
    label_after: str,
) -> List[dict]:
    if heatmaps_before_k_hw.shape != heatmaps_after_k_hw.shape:
        raise ValueError(
            f"before/after 热力图形状不一致: {tuple(heatmaps_before_k_hw.shape)} vs {tuple(heatmaps_after_k_hw.shape)}"
        )

    img = image_chw.detach().cpu().numpy().transpose(1, 2, 0)
    before_np = heatmaps_before_k_hw.detach().cpu().numpy()
    after_np = heatmaps_after_k_hw.detach().cpu().numpy()
    delta_np = after_np - before_np

    n = len(concept_ids)
    fig, axes = plt.subplots(n, 3, figsize=(9.6, 3.0 * n))
    if n == 1:
        axes = np.array([axes])

    stats = []
    for i, cid in enumerate(concept_ids):
        hm_before = before_np[i]
        hm_after = after_np[i]
        hm_delta = delta_np[i]

        p_lo_before, p_hi_before = np.percentile(hm_before, [2, 98])
        p_lo_after, p_hi_after = np.percentile(hm_after, [2, 98])
        spread_before = p_hi_before - p_lo_before
        spread_after = p_hi_after - p_lo_after
        before_norm = (
            np.zeros_like(hm_before) if spread_before < 0.05
            else np.clip((hm_before - p_lo_before) / (spread_before + 1e-8), 0.0, 1.0)
        )
        after_norm = (
            np.zeros_like(hm_after) if spread_after < 0.05
            else np.clip((hm_after - p_lo_after) / (spread_after + 1e-8), 0.0, 1.0)
        )
        delta_scale = np.max(np.abs(hm_delta)) + 1e-8

        blended_before = overlay_rgb(img, before_norm, cmap_name=overlay_cmap, alpha=alpha)
        blended_after = overlay_rgb(img, after_norm, cmap_name=overlay_cmap, alpha=alpha)

        ax0, ax1, ax2 = axes[i]
        ax0.imshow(blended_before)
        ax0.set_title(f"concept {cid} | {label_before}")
        ax0.axis("off")

        ax1.imshow(blended_after)
        ax1.set_title(f"concept {cid} | {label_after}")
        ax1.axis("off")

        im = ax2.imshow(hm_delta, cmap=delta_cmap, vmin=-delta_scale, vmax=delta_scale)
        ax2.set_title(f"concept {cid} | delta")
        ax2.axis("off")
        fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

        stats.append(
            {
                "concept_id": int(cid),
                "before_max": float(hm_before.max()),
                "after_max": float(hm_after.max()),
                "delta_abs_max": float(np.abs(hm_delta).max()),
                "before_mean": float(hm_before.mean()),
                "after_mean": float(hm_after.mean()),
            }
        )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return stats


def build_models(args, device: torch.device):
    before_model = CompleteModel(
        input_dim=3,
        feature_dim=args.feature_dim,
        num_concepts=args.num_concepts,
        num_heads=args.num_heads,
    ).to(device)
    after_model = CompleteModel(
        input_dim=3,
        feature_dim=args.feature_dim,
        num_concepts=args.num_concepts,
        num_heads=args.num_heads,
    ).to(device)
    maybe_load_weights(before_model, args.before_weights)
    maybe_load_weights(after_model, args.after_weights)
    return before_model, after_model


def run_one(
    x: torch.Tensor,
    src_tag: str,
    out_png: str,
    before_model: torch.nn.Module,
    after_model: torch.nn.Module,
    adj: torch.Tensor,
    args,
) -> None:
    before_low = _forward_spatial_heatmaps(before_model, x, adj)
    after_low = _forward_spatial_heatmaps(after_model, x, adj)

    num_c = before_low.shape[0]
    ids = parse_concept_ids(args.concepts, num_c)
    if not ids:
        if args.selection == 'delta':
            ids = pick_top_k_delta(before_low.cpu(), after_low.cpu(), args.top_k)
        else:
            ids = pick_top_k_compare(before_low.cpu(), after_low.cpu(), args.top_k)

    _, _, H, W = x.shape
    before_hi = upsample_heatmaps(before_low, H, W)[ids]
    after_hi = upsample_heatmaps(after_low, H, W)[ids]
    stats = save_triplet_panel(
        x[0].cpu(),
        before_hi.cpu(),
        after_hi.cpu(),
        ids,
        out_png,
        overlay_cmap=args.cmap,
        delta_cmap=args.delta_cmap,
        alpha=args.alpha,
        label_before=args.before_label,
        label_after=args.after_label,
    )

    sidecar = {
        'source': src_tag,
        'before_weights': os.path.abspath(args.before_weights),
        'after_weights': os.path.abspath(args.after_weights),
        'concept_ids': ids,
        'lowres_shape': list(before_low.shape),
        'image_hw': [H, W],
        'selection': args.selection,
        'stats': stats,
    }
    json_path = os.path.splitext(out_png)[0] + '.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(sidecar, f, ensure_ascii=False, indent=2)
    print(f'已保存: {out_png}')
    print(f'  来源: {src_tag}')
    print(f'  概念: {ids}')
    print(f'  统计: {json_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description='before/after 空间热力图对比可视化')
    parser.add_argument('--before-weights', type=str, required=True)
    parser.add_argument('--after-weights', type=str, required=True)
    parser.add_argument('--before-label', type=str, default='before')
    parser.add_argument('--after-label', type=str, default='after')
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--images-dir', type=str, default=None)
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--cub-root', type=str, default=None)
    parser.add_argument('--cub-split', type=str, default='train', choices=('train', 'test'))
    parser.add_argument('--sample-index', type=int, default=None)
    parser.add_argument('--stage1-out', type=str, default=None)
    parser.add_argument('--adj-path', type=str, default=None)
    parser.add_argument('--num-concepts', type=int, default=312)
    parser.add_argument('--feature-dim', type=int, default=256)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--concepts', type=str, default=None)
    parser.add_argument('--top-k', type=int, default=8)
    parser.add_argument('--selection', type=str, choices=('joint', 'delta'), default='delta')
    parser.add_argument('--cmap', type=str, default='turbo')
    parser.add_argument('--delta-cmap', type=str, default='coolwarm')
    parser.add_argument('--alpha', type=float, default=0.45)
    parser.add_argument('--out-dir', type=str, default='tmp_compare_viz')
    args = parser.parse_args()

    device = torch.device(args.device)
    before_model, after_model = build_models(args, device)
    adj = load_adj(device, args.stage1_out, args.adj_path)
    if adj.shape[0] != args.num_concepts:
        raise ValueError(f'邻接矩阵 {tuple(adj.shape)} 与 num_concepts={args.num_concepts} 不一致')

    os.makedirs(args.out_dir, exist_ok=True)

    if args.images_dir:
        paths = collect_image_paths(args.images_dir)
        if args.max_images > 0:
            paths = paths[:args.max_images]
        if not paths:
            raise FileNotFoundError(f'目录下未找到图像: {args.images_dir}')
        for path in paths:
            x = load_image_tensor(path, args.image_size).to(device)
            out_png = overlay_out_path(args.images_dir, path, args.out_dir, suffix='_before_after_delta.png')
            os.makedirs(os.path.dirname(out_png), exist_ok=True)
            run_one(x, path, out_png, before_model, after_model, adj, args)
        return

    if args.image:
        x = load_image_tensor(args.image, args.image_size).to(device)
        src_tag = args.image
        safe = 'single'
    elif args.cub_root is not None and args.sample_index is not None:
        x = load_cub_one(args.cub_root, args.cub_split, args.sample_index, args.image_size).to(device)
        src_tag = f'cub:{args.cub_root}[{args.sample_index}]'
        safe = f'cub_{args.sample_index}'
    else:
        parser.error('请提供 --images-dir、--image，或同时提供 --cub-root 与 --sample-index')

    out_png = os.path.join(os.path.abspath(args.out_dir), f'compare_{safe}_before_after_delta.png')
    run_one(x, src_tag, out_png, before_model, after_model, adj, args)


if __name__ == '__main__':
    main()
