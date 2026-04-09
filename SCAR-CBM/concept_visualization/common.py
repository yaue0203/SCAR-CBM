from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from network_architecture import CompleteModel
from visualize_spatial_concept_heatmap import load_adj, maybe_load_weights


@dataclass
class CubTrainItem:
    row_index: int
    image_id: int
    relpath: str
    image_path: str


def load_attribute_names(path: str) -> List[str]:
    names: List[str] = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            names.append(parts[1] if len(parts) == 2 else parts[0])
    return names


def load_cub_train_items(cub_root: str) -> List[CubTrainItem]:
    cub_root = os.path.abspath(cub_root)
    images_txt = os.path.join(cub_root, 'images.txt')
    split_txt = os.path.join(cub_root, 'train_test_split.txt')
    images_dir = os.path.join(cub_root, 'images')

    id_to_relpath: Dict[int, str] = {}
    with open(images_txt, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                id_to_relpath[int(parts[0])] = parts[1]

    items: List[CubTrainItem] = []
    with open(split_txt, 'r', encoding='utf-8', errors='replace') as f:
        row_idx = 0
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            image_id = int(parts[0])
            is_train = int(parts[1])
            if is_train != 1:
                continue
            relpath = id_to_relpath[image_id]
            items.append(CubTrainItem(row_idx, image_id, relpath, os.path.join(images_dir, relpath)))
            row_idx += 1
    return items


def load_image_chw(path: str, image_size: int) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert('RGB')
        if image_size > 0:
            img = img.resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(arr, (2, 0, 1)))


def overlay_rgb(image_hwc: np.ndarray, heatmap_hw: np.ndarray, cmap_name: str, alpha: float) -> np.ndarray:
    cmap = mpl.colormaps[cmap_name] if hasattr(mpl, 'colormaps') else plt.cm.get_cmap(cmap_name)
    color = cmap(np.clip(heatmap_hw, 0.0, 1.0))[..., :3].astype(np.float32)
    return np.clip((1.0 - alpha) * image_hwc + alpha * color, 0.0, 1.0)


def normalize_heatmap(heatmap_hw: np.ndarray) -> np.ndarray:
    p_lo, p_hi = np.percentile(heatmap_hw, [2, 98])
    spread = p_hi - p_lo
    if spread < 0.05:
        return np.zeros_like(heatmap_hw)
    return np.clip((heatmap_hw - p_lo) / (spread + 1e-8), 0.0, 1.0)


def upsample_heatmaps(heatmaps: torch.Tensor, out_hw: Sequence[int]) -> torch.Tensor:
    x = F.interpolate(heatmaps.unsqueeze(0).float(), size=tuple(out_hw), mode='bilinear', align_corners=False)
    return x.squeeze(0)


def topk_ids_from_two(stage2_lowres: torch.Tensor, stage5_lowres: torch.Tensor, k: int) -> List[int]:
    score = torch.maximum(stage2_lowres.amax(dim=(1, 2)), stage5_lowres.amax(dim=(1, 2)))
    k = min(int(k), int(score.shape[0]))
    return torch.topk(score, k).indices.cpu().tolist()


def choose_sample_indices(stage2_heatmap: np.ndarray, stage4_indices: np.ndarray | None, sample_count: int) -> List[int]:
    score = stage2_heatmap.max(axis=1)
    if stage4_indices is not None and stage4_indices.size > 0:
        unique = np.unique(stage4_indices.astype(np.int64).ravel())
        unique = unique[(unique >= 0) & (unique < stage2_heatmap.shape[0])]
        if unique.size > 0:
            ranked = unique[np.argsort(score[unique])[::-1]]
            return ranked[:sample_count].tolist()
    ranked = np.argsort(score)[::-1]
    return ranked[:sample_count].tolist()


def build_model(weights: str, num_concepts: int, feature_dim: int, num_heads: int, device: torch.device) -> CompleteModel:
    model = CompleteModel(input_dim=3, feature_dim=feature_dim, num_concepts=num_concepts, num_heads=num_heads).to(device)
    maybe_load_weights(model, weights)
    model.eval()
    return model


@torch.no_grad()
def forward_stage5_spatial_heatmap(model: CompleteModel, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    out = model(x.unsqueeze(0), adj)
    return out['spatial_concept_heatmap'][0].detach().cpu()


def save_bar_chart(labels: List[str], values: List[float], title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(labels))))
    ypos = np.arange(len(labels))
    ax.barh(ypos, values, color='#1f6feb')
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
