#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

PART_GROUP_ALIASES = {
    'head': {'bill', 'beak', 'crown', 'forehead', 'eye', 'throat', 'nape', 'head'},
    'wing': {'wing'},
    'tail': {'tail'},
    'leg': {'leg', 'foot'},
    'back': {'back', 'upperparts'},
    'belly': {'belly', 'breast', 'underparts', 'abdomen'},
    'body': {'body', 'shape', 'size'},
}

POINT_GROUPS = {
    'head': [2, 5, 6, 7, 10, 11, 15],
    'wing': [9, 13],
    'tail': [14],
    'leg': [8, 12],
    'back': [1, 10],
    'belly': [3, 4, 15],
}


@dataclass
class CUBPartRecord:
    image_id: int
    part_id: int
    x: float
    y: float
    visible: int


def load_attribute_names(attr_txt: str) -> List[str]:
    names: List[str] = []
    with open(attr_txt, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, raw = line.split(' ', 1)
            names.append(raw)
    return names


def build_attribute_part_groups(attr_txt: str) -> Dict[str, List[int]]:
    names = load_attribute_names(attr_txt)
    grouped: Dict[str, List[int]] = {k: [] for k in PART_GROUP_ALIASES}
    grouped['other'] = []
    for idx, name in enumerate(names):
        lowered = name.lower()
        assigned = False
        for group, aliases in PART_GROUP_ALIASES.items():
            if any(alias in lowered for alias in aliases):
                grouped[group].append(idx)
                assigned = True
                break
        if not assigned:
            grouped['other'].append(idx)
    return grouped


def load_image_id_map(cub_root: str) -> Dict[int, str]:
    images_txt = os.path.join(cub_root, 'images.txt')
    out: Dict[int, str] = {}
    with open(images_txt, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                out[int(parts[0])] = parts[1]
    return out


def load_part_locs(cub_root: str) -> Dict[int, List[CUBPartRecord]]:
    loc_path = os.path.join(cub_root, 'parts', 'part_locs.txt')
    out: Dict[int, List[CUBPartRecord]] = {}
    with open(loc_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            rec = CUBPartRecord(
                image_id=int(parts[0]),
                part_id=int(parts[1]),
                x=float(parts[2]),
                y=float(parts[3]),
                visible=int(parts[4]),
            )
            out.setdefault(rec.image_id, []).append(rec)
    return out


def build_part_group_mask(
    image_hw: Tuple[int, int],
    part_records: List[CUBPartRecord],
    target_group: str,
    radius: int = 14,
) -> np.ndarray:
    h, w = image_hw
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)
    valid_ids = set(POINT_GROUPS.get(target_group, []))
    for rec in part_records:
        if rec.visible != 1 or rec.part_id not in valid_ids:
            continue
        x, y = rec.x, rec.y
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    return (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)


def export_part_supervision_summary(cub_root: str, out_json: str) -> None:
    attr_txt = os.path.join(os.path.dirname(cub_root), 'attributes.txt')
    grouped = build_attribute_part_groups(attr_txt)
    image_map = load_image_id_map(cub_root)
    part_locs = load_part_locs(cub_root)
    counts = {k: len(v) for k, v in grouped.items()}
    visible_counts = {k: 0 for k in POINT_GROUPS}
    for image_id, records in part_locs.items():
        for group, part_ids in POINT_GROUPS.items():
            if any(r.visible == 1 and r.part_id in part_ids for r in records):
                visible_counts[group] += 1
    payload = {
        'attribute_groups': grouped,
        'attribute_group_sizes': counts,
        'part_visibility_counts': visible_counts,
        'num_images_with_parts': len(part_locs),
        'num_images_indexed': len(image_map),
    }
    Path(out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')



def load_image_sizes(cub_root: str) -> Dict[int, Tuple[int, int]]:
    image_map = load_image_id_map(cub_root)
    out: Dict[int, Tuple[int, int]] = {}
    for image_id, rel in image_map.items():
        img_path = os.path.join(cub_root, 'images', rel)
        with Image.open(img_path) as img:
            out[image_id] = (img.height, img.width)
    return out


def load_train_image_ids(cub_root: str) -> List[int]:
    split_txt = os.path.join(cub_root, 'train_test_split.txt')
    ids: List[int] = []
    with open(split_txt, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and int(parts[1]) == 1:
                ids.append(int(parts[0]))
    return ids


def export_group_part_masks_npz(cub_root: str, out_npz: str, image_size: int = 224) -> None:
    train_ids = load_train_image_ids(cub_root)
    image_sizes = load_image_sizes(cub_root)
    part_locs = load_part_locs(cub_root)
    group_names = ['head', 'wing', 'tail', 'leg', 'back', 'belly']
    masks = {g: [] for g in group_names}

    for image_id in train_ids:
        h, w = image_sizes[image_id]
        records = part_locs.get(image_id, [])
        for group in group_names:
            arr = build_part_group_mask((h, w), records, group)
            img = Image.fromarray(arr * 255)
            if image_size > 0:
                img = img.resize((image_size, image_size), Image.NEAREST)
            out = (np.asarray(img, dtype=np.uint8) > 0).astype(np.uint8)
            masks[group].append(out)

    payload = {f'{g}_mask': np.stack(v, axis=0)[:, None, :, :] for g, v in masks.items()}
    payload['train_image_ids'] = np.asarray(train_ids, dtype=np.int64)
    payload['group_names'] = np.asarray(group_names)
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **payload)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="导出 CUB 部位监督摘要/掩码")
    parser.add_argument("--cub-root", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--export-masks", action='store_true')
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    if args.export_masks:
        export_group_part_masks_npz(args.cub_root, args.out, image_size=args.image_size)
    else:
        export_part_supervision_summary(args.cub_root, args.out)
    print(f"已写入: {args.out}")
