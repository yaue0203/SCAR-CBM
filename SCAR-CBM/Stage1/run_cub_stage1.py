#!/usr/bin/env python3
"""
使用 Caltech-UCSD Birds-200-2011 (CUB_200_2011) 做训练框架中的 Stage 1。

数据来源（相对 --cub-root）:
  - train_test_split.txt          → 仅 is_training_image=1 的样本
  - attributes/image_attribute_labels.txt → <image_id> <attribute_id> <is_present> ...

输出概念矩阵形状 [N_train, 312]；若同一 (image, attribute) 有多条 MTurk 记录，对 is_present 取平均。
随后调用与全量训练相同的 stage1_hypergraph.build_concept_hypergraph。

用法（在 SCAR-CBM 根目录）:
  python Stage1/run_cub_stage1.py
  python Stage1/run_cub_stage1.py --cub-root Data/CUB_200_2011 --out Stage1/stage1_cub_output --threshold 0.01
  python Stage1/run_cub_stage1.py --labeled-ratio 10%
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from stage1_hypergraph import build_concept_hypergraph
from stage_output_utils import resolve_default_output_dir

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


def load_cub_train_concept_matrix(cub_root: str) -> tuple[np.ndarray, dict]:
    cub_root = os.path.abspath(cub_root)
    split_path = os.path.join(cub_root, 'train_test_split.txt')
    attr_path = os.path.join(cub_root, 'attributes', 'image_attribute_labels.txt')

    if not os.path.isfile(split_path):
        raise FileNotFoundError(f'缺少 {split_path}')
    if not os.path.isfile(attr_path):
        raise FileNotFoundError(f'缺少 {attr_path}')

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
    meta = {
        'num_train': n_train,
        'num_concepts': NUM_CUB_ATTRIBUTES,
        'cub_root': cub_root,
        'rows_without_any_label': int(np.sum(counts.sum(axis=1) == 0)),
    }
    return mat.astype(np.float32), meta


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    default_cub = os.path.join(_ROOT, 'Data', 'CUB_200_2011')

    parser = argparse.ArgumentParser(description='CUB-200-2011 Stage 1（概念超图）')
    parser.add_argument(
        '--cub-root',
        type=str,
        default=default_cub,
        help='CUB_200_2011 根目录（含 train_test_split.txt、attributes/）',
    )
    parser.add_argument(
        '--out',
        type=str,
        default=None,
        help='输出目录；未提供时默认自动创建 Stage1/stage1_cub_output_<HH-MM_YYMMDD>/',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.01,
        help='共现归一化边权阈值。CUB 上归一化共现通常 <0.03，默认 0.01；通用随机示例可用 0.1',
    )
    parser.add_argument(
        '--labeled-ratio',
        type=parse_ratio_arg,
        default=0.1,
        help='用于构建超图的有标签数据比例，支持 0.1 或 10%%，默认 10%%',
    )
    parser.add_argument('--seed', type=int, default=42, help='有标签样本抽样随机种子')
    parser.add_argument(
        '--save-hypergraph-pickle',
        action='store_true',
        help='额外保存 concept_hypergraph.pkl（NetworkX 序列化可能达数百 MB，易占内存，默认关闭）',
    )
    args = parser.parse_args()

    out_dir = resolve_default_output_dir(
        os.path.dirname(os.path.abspath(__file__)),
        'stage1_cub_output',
        args.out,
    )
    os.makedirs(out_dir, exist_ok=True)

    logging.info('读取 CUB 训练集概念标注 …')
    concept_ids, meta = load_cub_train_concept_matrix(args.cub_root)
    np.save(os.path.join(out_dir, 'c_labeled_train.npy'), concept_ids)
    logging.info(
        '概念矩阵: shape=%s, 无标注行数=%d',
        concept_ids.shape,
        meta['rows_without_any_label'],
    )

    concept_ids_for_graph, selected_indices = sample_labeled_rows(
        concept_ids,
        ratio=args.labeled_ratio,
        seed=args.seed,
    )
    np.save(os.path.join(out_dir, 'labeled_row_indices.npy'), selected_indices.astype(np.int64))
    logging.info(
        '用于构图的有标签样本: %d/%d (ratio=%.4f)',
        len(selected_indices),
        len(concept_ids),
        args.labeled_ratio,
    )

    hg = build_concept_hypergraph(
        concept_ids_for_graph,
        num_concepts=NUM_CUB_ATTRIBUTES,
        cooccurrence_threshold=args.threshold,
    )

    adj_cpu = hg.get_adjacency_matrix('cpu')
    torch.save(adj_cpu, os.path.join(out_dir, 'adj_norm_gcn.pt'))

    cm = hg.cooccurrence_matrix
    if cm is not None:
        if hasattr(cm, 'detach'):
            cm = cm.detach().cpu().numpy()
        np.save(os.path.join(out_dir, 'cooccurrence_raw.npy'), np.asarray(cm))

    edges_path = os.path.join(out_dir, 'hypergraph_edges.json')
    edges_compact = [
        {'u': int(u), 'v': int(v), 'weight': float(d.get('weight', 1.0))}
        for u, v, d in hg.hypergraph.edges(data=True)
    ]
    with open(edges_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'num_concepts': NUM_CUB_ATTRIBUTES,
                'threshold': args.threshold,
                'labeled_ratio': args.labeled_ratio,
                'num_labeled_total': int(len(concept_ids)),
                'num_labeled_used': int(len(selected_indices)),
                'sample_seed': int(args.seed),
                'labeled_row_indices_path': os.path.join(out_dir, 'labeled_row_indices.npy'),
                'edges': edges_compact,
            },
            f,
        )

    pkl_path = None
    if args.save_hypergraph_pickle:
        import pickle

        pkl_path = os.path.join(out_dir, 'concept_hypergraph.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(hg, f, protocol=4)

    with open(os.path.join(out_dir, 'cub_stage1_meta.txt'), 'w', encoding='utf-8') as f:
        for k, v in meta.items():
            f.write(f'{k}: {v}\n')
        f.write(f'labeled_ratio: {args.labeled_ratio}\n')
        f.write(f'num_labeled_total: {len(concept_ids)}\n')
        f.write(f'num_labeled_used: {len(selected_indices)}\n')
        f.write(f'sample_seed: {args.seed}\n')
        f.write(f'cooccurrence_threshold: {args.threshold}\n')
        f.write(f'hypergraph_edges: {hg.hypergraph.number_of_edges()}\n')

    print('CUB Stage 1 完成')
    print(f'  训练图像数: {meta["num_train"]}, 概念维度: {NUM_CUB_ATTRIBUTES}')
    print(
        f'  用于构图的有标签样本: {len(selected_indices)}/{len(concept_ids)} '
        f'(ratio={args.labeled_ratio:.4f}, seed={args.seed})'
    )
    print(f'  输出目录: {out_dir}')
    print('  邻接矩阵: adj_norm_gcn.pt  |  概念矩阵: c_labeled_train.npy')
    print("  行索引: labeled_row_indices.npy")
    print(f'  轻量边表: {edges_path}')
    print(f'  超图边数: {hg.hypergraph.number_of_edges()}  (阈值={args.threshold})')
    if pkl_path:
        print(f'  已保存 pickle: {pkl_path}')
    print()
    print('后续完整训练: num_concepts=312；prebuilt_hypergraph 建议用概念矩阵即时重建（避免巨大 pickle）:')
    print('  from stage1_hypergraph import build_concept_hypergraph')
    print("  hg = build_concept_hypergraph(np.load('c_labeled_train.npy'), num_concepts=312,")
    print(f'      cooccurrence_threshold={args.threshold})')
    print('  framework.fit(..., prebuilt_hypergraph=hg, labeled_concept_ids=None)')


if __name__ == '__main__':
    main()
