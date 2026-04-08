#!/usr/bin/env python3
"""
仅执行训练框架中的 Stage 1：从有标签概念矩阵构建超图并保存邻接矩阵。

不创建模型、不进入 fit 的 epoch。输出可供后续训练传入 prebuilt_hypergraph，
或单独检查共现结构。

用法（在仓库根目录 SCAR-CBM 下）:
  python Stage1/run_stage1_train_only.py
  python Stage1/run_stage1_train_only.py --npy path/to/c_labeled.npy --out ./stage1_out
  python Stage1/run_stage1_train_only.py --npy path/to/c_labeled.npy --labeled-ratio 10%
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from stage1_data_utils import complement_indices, parse_ratio_arg, sample_labeled_rows
from stage1_hypergraph import build_concept_hypergraph
from stage_output_utils import resolve_default_output_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(
        description='只运行训练中的 Stage 1（概念超图构建）'
    )
    parser.add_argument(
        '--npy',
        type=str,
        default=None,
        help='概念矩阵 .npy，形状 [N, C]；省略则用随机示例数据',
    )
    parser.add_argument(
        '--out',
        type=str,
        default=None,
        help='输出目录；未提供时默认自动创建 Stage1/stage1_train_output_<HH-MM_YYMMDD>/',
    )
    parser.add_argument(
        '--labeled-ratio',
        type=parse_ratio_arg,
        default=0.1,
        help='用于构建超图的有标签数据比例，支持 0.1 或 10%%，默认 10%%',
    )
    parser.add_argument('--threshold', type=float, default=0.1, help='共现边权阈值')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    out_dir = resolve_default_output_dir(
        os.path.dirname(os.path.abspath(__file__)),
        'stage1_train_output',
        args.out,
    )
    os.makedirs(out_dir, exist_ok=True)

    if args.npy:
        concept_ids = np.load(args.npy).astype(np.float32)
    else:
        rng = np.random.default_rng(args.seed)
        n, c = 120, 10
        concept_ids = (rng.random((n, c)) < 0.35).astype(np.float32)
        empty = concept_ids.sum(axis=1) == 0
        if np.any(empty):
            concept_ids[empty, rng.integers(0, c, size=int(empty.sum()))] = 1.0

    concept_ids_for_graph, selected_indices = sample_labeled_rows(
        concept_ids,
        ratio=args.labeled_ratio,
        seed=args.seed,
    )
    unlabeled_indices = complement_indices(len(concept_ids), selected_indices)
    np.save(os.path.join(out_dir, 'labeled_row_indices.npy'), selected_indices.astype(np.int64))
    np.save(os.path.join(out_dir, 'unlabeled_row_indices.npy'), unlabeled_indices.astype(np.int64))

    hg = build_concept_hypergraph(
        concept_ids_for_graph,
        num_concepts=None,
        cooccurrence_threshold=args.threshold,
    )

    adj_cpu = hg.get_adjacency_matrix('cpu')
    adj_path = os.path.join(out_dir, 'adj_norm_gcn.pt')
    torch.save(adj_cpu, adj_path)

    co_path = os.path.join(out_dir, 'cooccurrence_raw.npy')
    if hg.cooccurrence_matrix is not None:
        cm = hg.cooccurrence_matrix
        if hasattr(cm, 'detach'):
            cm = cm.detach().cpu().numpy()
        np.save(co_path, np.asarray(cm))

    import pickle

    pkl_path = os.path.join(out_dir, 'concept_hypergraph.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(hg, f)

    print('Stage 1 完成（与 train_framework 中构建逻辑一致）')
    print(
        f'  使用有标签样本: {len(selected_indices)}/{len(concept_ids)} '
        f'(ratio={args.labeled_ratio:.4f}, seed={args.seed})'
    )
    print(f'  剩余未标注样本: {len(unlabeled_indices)}')
    print(f'  节点: {hg.hypergraph.number_of_nodes()}, 边: {hg.hypergraph.number_of_edges()}')
    print(f'  邻接矩阵: {adj_path}')
    print(f'  共现矩阵: {co_path}')
    print(f'  行索引: {os.path.join(out_dir, "labeled_row_indices.npy")} | {os.path.join(out_dir, "unlabeled_row_indices.npy")}')
    print(f'  超图对象: {pkl_path}  →  fit(..., prebuilt_hypergraph=pickle.load(open(...)))')


if __name__ == '__main__':
    main()
