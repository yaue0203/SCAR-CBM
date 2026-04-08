#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

NUM_CUB_ATTRIBUTES = 312


def parse_ratio_arg(value: str) -> float:
    text = str(value).strip()
    if text.endswith('%'):
        ratio = float(text[:-1]) / 100.0
    else:
        ratio = float(text)
    if not (0.0 < ratio <= 1.0):
        raise argparse.ArgumentTypeError(
            f'比例必须在 (0, 1] 范围内，当前为 {value}'
        )
    return ratio


def sample_labeled_rows(
    concept_ids: np.ndarray,
    ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples = int(concept_ids.shape[0])
    n_selected = min(n_samples, max(1, int(np.ceil(n_samples * ratio))))
    if n_selected == n_samples:
        indices = np.arange(n_samples, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(n_samples, size=n_selected, replace=False))
    return concept_ids[indices], indices


def complement_indices(n_samples: int, selected_indices: np.ndarray) -> np.ndarray:
    all_idx = np.arange(int(n_samples), dtype=np.int64)
    selected = np.asarray(selected_indices, dtype=np.int64)
    return np.setdiff1d(all_idx, selected, assume_unique=True)


def _source_signature(split_path: str, attr_path: str) -> dict[str, Any]:
    split_stat = os.stat(split_path)
    attr_stat = os.stat(attr_path)
    return {
        'split_path': os.path.abspath(split_path),
        'split_mtime_ns': int(split_stat.st_mtime_ns),
        'split_size': int(split_stat.st_size),
        'attr_path': os.path.abspath(attr_path),
        'attr_mtime_ns': int(attr_stat.st_mtime_ns),
        'attr_size': int(attr_stat.st_size),
        'num_concepts': NUM_CUB_ATTRIBUTES,
    }


def _cache_dir() -> Path:
    return Path(__file__).resolve().parent / 'Stage1' / '.cache'


def _cache_path(cub_root: str) -> Path:
    key = hashlib.sha1(os.path.abspath(cub_root).encode('utf-8')).hexdigest()[:12]
    return _cache_dir() / f'cub_train_concepts_{key}.npz'


def _parse_cub_train_concept_matrix(cub_root: str) -> tuple[np.ndarray, dict[str, Any]]:
    split_path = os.path.join(cub_root, 'train_test_split.txt')
    attr_path = os.path.join(cub_root, 'attributes', 'image_attribute_labels.txt')

    train_ids: list[int] = []
    with open(split_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            img_id, is_train = int(parts[0]), int(parts[1])
            if is_train == 1:
                train_ids.append(img_id)

    train_set = frozenset(train_ids)
    n_train = len(train_ids)
    img_to_row = {img_id: i for i, img_id in enumerate(train_ids)}

    mat = np.zeros((n_train, NUM_CUB_ATTRIBUTES), dtype=np.float64)
    counts = np.zeros((n_train, NUM_CUB_ATTRIBUTES), dtype=np.float64)

    with open(attr_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            img_id = int(parts[0])
            if img_id not in train_set:
                continue
            attr_id = int(parts[1])
            is_present = float(parts[2])
            c = attr_id - 1
            if not (0 <= c < NUM_CUB_ATTRIBUTES):
                continue
            r = img_to_row[img_id]
            mat[r, c] += is_present
            counts[r, c] += 1.0

    nz = counts > 0
    mat[nz] /= counts[nz]
    parsed_meta = {
        'num_train': n_train,
        'num_concepts': NUM_CUB_ATTRIBUTES,
        'cub_root': os.path.abspath(cub_root),
        'rows_without_any_label': int(np.sum(counts.sum(axis=1) == 0)),
    }
    return mat.astype(np.float32), parsed_meta


def load_cub_train_concept_matrix(
    cub_root: str,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    cub_root = os.path.abspath(cub_root)
    split_path = os.path.join(cub_root, 'train_test_split.txt')
    attr_path = os.path.join(cub_root, 'attributes', 'image_attribute_labels.txt')

    if not os.path.isfile(split_path):
        raise FileNotFoundError(f'缺少 {split_path}')
    if not os.path.isfile(attr_path):
        raise FileNotFoundError(f'缺少 {attr_path}')

    signature = _source_signature(split_path, attr_path)
    cache_path = _cache_path(cub_root)

    if use_cache and not refresh_cache and cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as payload:
            cached_signature = json.loads(str(payload['source_signature'].item()))
            if cached_signature == signature:
                concept_ids = payload['concept_ids'].astype(np.float32, copy=False)
                meta = json.loads(str(payload['meta_json'].item()))
                meta.update({
                    'cache_hit': True,
                    'cache_path': str(cache_path),
                    'source_signature': signature,
                })
                return concept_ids, meta

    concept_ids, meta = _parse_cub_train_concept_matrix(cub_root)

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            concept_ids=concept_ids,
            source_signature=json.dumps(signature, ensure_ascii=False),
            meta_json=json.dumps(meta, ensure_ascii=False),
        )

    meta = dict(meta)
    meta.update({
        'cache_hit': False,
        'cache_path': str(cache_path),
        'source_signature': signature,
    })
    return concept_ids, meta
