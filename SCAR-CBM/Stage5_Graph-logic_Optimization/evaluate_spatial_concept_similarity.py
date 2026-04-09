#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from network_architecture import CompleteModel
from visualize_spatial_concept_heatmap import (
    collect_image_paths,
    load_adj,
    load_image_tensor,
    maybe_load_weights,
)


def load_mask(mask_path: str, image_size: int) -> torch.Tensor:
    from PIL import Image

    with Image.open(mask_path) as img:
        img = img.convert("L")
        if image_size > 0:
            img = img.resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def forward_spatial_heatmaps(model: CompleteModel, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    return out["spatial_concept_heatmap"]


def map_image_to_mask(cub_root: str, image_path: str) -> str:
    rel = os.path.relpath(image_path, os.path.join(cub_root, "images"))
    base, _ = os.path.splitext(rel)
    return os.path.join(cub_root, "segmentations", base + ".png")


def summarize_image(
    heatmap: torch.Tensor,
    mask: torch.Tensor,
    top_k: int,
    min_conf: float,
    eps: float = 1e-6,
) -> Dict[str, float | List[int]]:
    _, c, h, w = heatmap.shape
    hm = heatmap[0]
    mask_rs = F.interpolate(mask.float(), size=(h, w), mode="bilinear", align_corners=True).clamp(0.0, 1.0)[0, 0]
    bg = 1.0 - mask_rs

    fg_mass = (hm * mask_rs.unsqueeze(0)).sum(dim=(1, 2))
    bg_mass = (hm * bg.unsqueeze(0)).sum(dim=(1, 2))
    total_mass = (fg_mass + bg_mass).clamp_min(eps)
    fg_ratio = fg_mass / total_mass
    peak = hm.amax(dim=(1, 2))

    active = torch.nonzero(peak > min_conf, as_tuple=False).squeeze(1)
    if active.numel() > top_k:
        chosen = torch.topk(peak[active], top_k).indices
        active = active[chosen]
    if active.numel() < 2:
        return {
            "num_active": int(active.numel()),
            "mean_fg_ratio": float(fg_ratio.mean().item()),
            "active_mean_fg_ratio": float(fg_ratio[active].mean().item()) if active.numel() > 0 else 0.0,
            "active_mean_peak": float(peak[active].mean().item()) if active.numel() > 0 else 0.0,
            "active_mean_similarity": 0.0,
            "active_max_similarity": 0.0,
            "active_concepts": active.cpu().tolist(),
        }

    feats = hm[active] * mask_rs.unsqueeze(0)
    feats = feats.view(feats.shape[0], -1)
    feats = feats - feats.mean(dim=1, keepdim=True)
    feats = F.normalize(feats, dim=1, eps=eps)
    sim = feats @ feats.t()
    off_diag = sim[~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)]

    return {
        "num_active": int(active.numel()),
        "mean_fg_ratio": float(fg_ratio.mean().item()),
        "active_mean_fg_ratio": float(fg_ratio[active].mean().item()),
        "active_mean_peak": float(peak[active].mean().item()),
        "active_mean_similarity": float(off_diag.mean().item()),
        "active_max_similarity": float(off_diag.max().item()),
        "active_concepts": active.cpu().tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="量化评估空间热力图的前景占比与概念相似度")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--cub-root", type=str, required=True)
    parser.add_argument("--stage1-out", type=str, default=None)
    parser.add_argument("--adj-path", type=str, default=None)
    parser.add_argument("--num-concepts", type=int, default=312)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--min-conf", type=float, default=0.6)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = CompleteModel(
        input_dim=3,
        feature_dim=args.feature_dim,
        num_concepts=args.num_concepts,
        num_heads=args.num_heads,
    ).to(device)
    maybe_load_weights(model, args.weights)
    _ = load_adj(device, args.stage1_out, args.adj_path)
    model.eval()

    paths = collect_image_paths(args.images_dir)
    if args.max_images > 0:
        paths = paths[: args.max_images]

    rows: List[Dict[str, object]] = []
    agg = defaultdict(list)
    for image_path in paths:
        mask_path = map_image_to_mask(args.cub_root, image_path)
        if not os.path.exists(mask_path):
            continue
        x = load_image_tensor(image_path, args.image_size).to(device)
        mask = load_mask(mask_path, args.image_size).to(device)
        heatmap = forward_spatial_heatmaps(model, x)
        row = summarize_image(heatmap, mask, args.top_k, args.min_conf)
        row["image"] = image_path
        rows.append(row)
        for key in ("num_active", "mean_fg_ratio", "active_mean_fg_ratio", "active_mean_peak", "active_mean_similarity", "active_max_similarity"):
            agg[key].append(float(row[key]))

    summary = {
        "weights": os.path.abspath(args.weights),
        "images_dir": os.path.abspath(args.images_dir),
        "num_images": len(rows),
        "metrics": {k: (float(np.mean(v)) if v else 0.0) for k, v in agg.items()},
        "rows": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"已写入: {args.out}")
    print(json.dumps(summary["metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
