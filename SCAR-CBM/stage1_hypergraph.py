"""
训练框架 Stage 1：从有标签概念向量构建概念关系超图（纯离线步骤）。

与 `ConceptualSSLFramework.build_concept_hypergraph` / `fit` 共用同一实现，
可单独运行本模块或 `Stage1/run_stage1_train_only.py`，不加载模型、不进入 epoch。
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from core_modules import ConceptHypergraph

logger = logging.getLogger(__name__)


def build_concept_hypergraph(
    labeled_concept_ids: np.ndarray,
    num_concepts: Optional[int] = None,
    cooccurrence_threshold: float = 0.1,
) -> ConceptHypergraph:
    """
    从有标签数据的概念 ID 矩阵构建概念超图（与训练中 Stage 1 一致）。

    Args:
        labeled_concept_ids: [N, C]，与 `ConceptHypergraph.build_from_labeled_data` 一致
        num_concepts: 概念数 C；默认从矩阵列数推断
        cooccurrence_threshold: 归一化共现边权阈值

    Returns:
        构建好的 ConceptHypergraph（可调用 get_adjacency_matrix）
    """
    if labeled_concept_ids.ndim != 2:
        raise ValueError(
            f"labeled_concept_ids 应为二维 [N, C]，当前 shape={labeled_concept_ids.shape}"
        )
    n_cols = labeled_concept_ids.shape[1]
    if num_concepts is None:
        num_concepts = n_cols
    elif n_cols != num_concepts:
        raise ValueError(
            f"num_concepts={num_concepts} 与 labeled_concept_ids 列数 {n_cols} 不一致"
        )

    hg = ConceptHypergraph(num_concepts)
    hg.build_from_labeled_data(
        labels=None,
        concept_ids=torch.FloatTensor(labeled_concept_ids.astype(np.float32)),
        cooccurrence_threshold=cooccurrence_threshold,
    )
    logger.info(
        "Stage 1 超图: %d 个节点, %d 条边 (阈值=%.4f)",
        hg.hypergraph.number_of_nodes(),
        hg.hypergraph.number_of_edges(),
        cooccurrence_threshold,
    )
    return hg
