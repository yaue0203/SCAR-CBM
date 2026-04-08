#!/usr/bin/env python3
"""
从 train_framework.py 中独立出的 Stage 3 脚本：自适应伪标签生成。

用途:
  - 使用 Stage2 导出的特征，独立运行 Stage3 伪标签流程
  - 支持按 epoch 迭代并更新伪标签历史

典型用法（在 SCAR-CBM 根目录）:
  python "Stage3_Pseudo-label/run_stage3.py"       --unlabeled-npz "Stage2_Semantic-feature-extraction/stage2_output/stage2_features.npz"       --labeled-features-npy your_labeled_features.npy       --labeled-labels-npy your_labeled_labels.npy       --num-concepts 312

  # CUB：随机 10%% 训练样本作有标签（312 维属性真值），其余作无标签查询
  python "Stage3_Pseudo-label/run_stage3.py"       --unlabeled-npz .../stage2_features.npz       --cub-root Data/CUB_200_2011       --split-seed 42       --num-concepts 312

  # 默认行为：若未提供手工 labeled 输入，则自动按 CUB train 的 10%% / 90%% 划分
  python "Stage3_Pseudo-label/run_stage3.py"       --unlabeled-npz .../stage2_features.npz       --num-concepts 312
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core_modules import PseudoLabelHistory, knn_retrieval
from pseudo_label_and_sampling import PseudoLabelGenerator
from stage1_data_utils import complement_indices, load_cub_train_concept_matrix, sample_labeled_rows
from stage_output_utils import find_latest_output_file, resolve_default_output_dir


def split_cub_labeled_fraction(
    f_visual: torch.Tensor,
    c_heatmap: torch.Tensor,
    gt_concepts: np.ndarray,
    labeled_fraction: float,
    seed: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
    np.ndarray,
]:
    """
    将 Stage2 的 train 特征按行与 CUB 官方 train 概念矩阵对齐后，
    使用与 Stage1 完全一致的抽样逻辑划分 labeled / unlabeled。

    gt_concepts: [N, num_concepts]，须与 f_visual 行顺序一致（与 Stage1 run_cub_stage1 相同）。
    """
    n = f_visual.shape[0]
    if gt_concepts.shape[0] != n:
        raise ValueError(
            f"CUB 概念矩阵行数 {gt_concepts.shape[0]} 与 Stage2 样本数 {n} 不一致；"
            "请确认 Stage2 使用同一 CUB train 划分且顺序未被重排。"
        )
    if c_heatmap.shape[0] != n:
        raise ValueError("c_heatmap 样本数与 f_visual 不一致")
    if n < 2:
        raise ValueError("Stage2 样本数至少为 2 才能做有标签/无标签划分")
    if not (0.0 < labeled_fraction <= 1.0):
        raise ValueError("--labeled-fraction 必须在 (0, 1] 内，例如 0.1 表示 10%")

    _, labeled_idx = sample_labeled_rows(
        gt_concepts,
        ratio=float(labeled_fraction),
        seed=int(seed),
    )
    labeled_idx = labeled_idx.astype(np.int64, copy=False)
    unlabeled_idx = complement_indices(n, labeled_idx)

    if unlabeled_idx.size == 0:
        raise ValueError("有标签比例过高，导致无标签集合为空；Stage3 需要至少 1 个无标签样本")

    f_l = f_visual[labeled_idx]
    y_l = torch.from_numpy(gt_concepts[labeled_idx].astype(np.float32))
    f_u = f_visual[unlabeled_idx]
    c_u = c_heatmap[unlabeled_idx]
    return f_u, c_u, f_l, y_l, labeled_idx, unlabeled_idx


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少文件: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _load_tensor(path: str, name: str) -> torch.Tensor:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少{name}文件: {path}")
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"{name}应为 2 维张量，当前为: {arr.shape}")
    return torch.from_numpy(arr.astype(np.float32))


def load_unlabeled_inputs(npz_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    payload = _load_npz(npz_path)
    if "f_visual" not in payload:
        raise KeyError(f"{npz_path} 中缺少键: f_visual")
    if "c_heatmap" not in payload:
        raise KeyError(f"{npz_path} 中缺少键: c_heatmap")

    f_visual = torch.from_numpy(payload["f_visual"].astype(np.float32))
    c_heatmap = torch.from_numpy(payload["c_heatmap"].astype(np.float32))
    if f_visual.ndim != 2 or c_heatmap.ndim != 2:
        raise ValueError(
            f"unlabeled 输入维度错误: f_visual={tuple(f_visual.shape)}, c_heatmap={tuple(c_heatmap.shape)}"
        )
    if f_visual.shape[0] != c_heatmap.shape[0]:
        raise ValueError("unlabeled f_visual 与 c_heatmap 的样本数不一致")
    return f_visual, c_heatmap


def load_labeled_refs(
    unlabeled_features: torch.Tensor,
    unlabeled_heatmap: torch.Tensor,
    labeled_npz: str | None,
    labeled_features_npy: str | None,
    labeled_labels_npy: str | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if labeled_features_npy and labeled_labels_npy:
        f_l = _load_tensor(labeled_features_npy, "labeled feature")
        y_l = _load_tensor(labeled_labels_npy, "labeled label")
    elif labeled_npz:
        payload = _load_npz(labeled_npz)
        if "f_visual" not in payload:
            raise KeyError(f"{labeled_npz} 中缺少键: f_visual")
        f_l = torch.from_numpy(payload["f_visual"].astype(np.float32))

        if "labels" in payload:
            y_l = torch.from_numpy(payload["labels"].astype(np.float32))
        elif "c_heatmap" in payload:
            y_l = torch.from_numpy(payload["c_heatmap"].astype(np.float32))
            print("警告: labeled npz 中未找到 labels，使用 c_heatmap 作为 KNN 标签参考。")
        else:
            raise KeyError(f"{labeled_npz} 中缺少 labels 或 c_heatmap 键")
    else:
        f_l = unlabeled_features
        y_l = unlabeled_heatmap
        print("警告: 未提供 labeled 参考数据，退化为使用 unlabeled 特征自身进行 KNN 检索。")

    if f_l.ndim != 2 or y_l.ndim != 2:
        raise ValueError(f"labeled 输入维度错误: f_l={tuple(f_l.shape)}, y_l={tuple(y_l.shape)}")
    if f_l.shape[0] != y_l.shape[0]:
        raise ValueError("labeled features 与 labels 的样本数不一致")
    return f_l, y_l


@torch.no_grad()
def run_stage3_epochs(
    features_unlabeled: torch.Tensor,
    heatmap_unlabeled: torch.Tensor,
    features_labeled: torch.Tensor,
    labels_labeled: torch.Tensor,
    num_concepts: int,
    alpha: float,
    epochs: int,
    batch_size: int,
    k: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    n = features_unlabeled.shape[0]
    if heatmap_unlabeled.shape[1] != num_concepts:
        raise ValueError(
            f"num_concepts={num_concepts} 与 unlabeled c_heatmap.shape[1]={heatmap_unlabeled.shape[1]} 不一致"
        )
    if labels_labeled.shape[1] != num_concepts:
        raise ValueError(
            f"num_concepts={num_concepts} 与 labeled labels.shape[1]={labels_labeled.shape[1]} 不一致"
        )

    f_u = features_unlabeled.to(device)
    c_u = heatmap_unlabeled.to(device)
    f_l = features_labeled.to(device)
    y_l = labels_labeled.to(device)

    generator = PseudoLabelGenerator(num_concepts=num_concepts, alpha=alpha)
    history = PseudoLabelHistory(num_samples=n)

    final_c_pseudo = torch.zeros_like(c_u)
    final_m_rel = torch.zeros(n, device=device)
    reliabilities = []
    pseudo_mean_abs = []

    for epoch in range(epochs):
        generator.epoch = epoch
        epoch_pseudo = []
        epoch_mrel = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            fu_b = f_u[start:end]
            cu_b = c_u[start:end]

            c_knn = knn_retrieval(fu_b, f_l, y_l, k=min(k, f_l.shape[0]))
            c_pseudo, m_rel = generator.generate(
                cu_b,
                c_knn,
                contour_mask=None,
                pseudo_label_history=history,
                sample_ids=np.arange(start, end),
            )

            sample_ids = np.arange(start, end)
            history.update(sample_ids, c_pseudo)

            epoch_pseudo.append(c_pseudo)
            epoch_mrel.append(m_rel)

        epoch_pseudo_t = torch.cat(epoch_pseudo, dim=0)
        epoch_mrel_t = torch.cat(epoch_mrel, dim=0)
        final_c_pseudo = epoch_pseudo_t
        final_m_rel = epoch_mrel_t

        reliabilities.append(epoch_mrel_t.float().mean().item())
        pseudo_mean_abs.append(epoch_pseudo_t.abs().mean().item())
        print(
            f"[Epoch {epoch + 1}/{epochs}] "
            f"reliable_ratio={reliabilities[-1]:.4f}, "
            f"mean_abs_pseudo={pseudo_mean_abs[-1]:.4f}"
        )

    return {
        "c_pseudo": final_c_pseudo.detach().cpu().numpy().astype(np.float32),
        "m_rel": final_m_rel.detach().cpu().numpy().astype(np.float32),
        "reliable_ratio_per_epoch": np.asarray(reliabilities, dtype=np.float32),
        "mean_abs_pseudo_per_epoch": np.asarray(pseudo_mean_abs, dtype=np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 独立伪标签脚本")
    default_unlabeled_npz = find_latest_output_file(
        os.path.join(_ROOT, 'Stage2_Semantic-feature-extraction'),
        'stage2_output',
        'stage2_features.npz',
    )

    parser.add_argument(
        "--unlabeled-npz",
        type=str,
        default=default_unlabeled_npz,
        help="Stage2 unlabeled 输出 .npz（默认自动读取最新的 stage2_output_<HH-MM_YYMMDD>/stage2_features.npz）",
    )
    parser.add_argument(
        "--labeled-npz",
        type=str,
        default=None,
        help="可选：Stage2 labeled 输出 .npz（优先读取 labels，否则回退 c_heatmap）",
    )
    parser.add_argument(
        "--labeled-features-npy",
        type=str,
        default=None,
        help="可选：有标签特征 .npy，形状 [N_l, feature_dim]",
    )
    parser.add_argument(
        "--labeled-labels-npy",
        type=str,
        default=None,
        help="可选：有标签概念标签 .npy，形状 [N_l, num_concepts]",
    )
    parser.add_argument(
        "--cub-root",
        type=str,
        default=None,
        help="CUB_200_2001 根目录；与 --labeled-fraction 联用时从 attributes 读取 [N,312] 真值并划分有/无标签集",
    )
    parser.add_argument(
        "--labeled-fraction",
        type=float,
        default=0.1,
        help="默认 0.1。表示从 CUB train 中随机选 10%% 作有标签（需 --cub-root 或使用默认 CUB 路径）；若提供手工 labeled 输入则忽略该参数",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="与 --labeled-fraction 联用时的随机种子（可复现划分）",
    )
    parser.add_argument("--num-concepts", type=int, default=312, help="概念数量")
    parser.add_argument("--alpha", type=float, default=0.7, help="伪标签混合系数")
    parser.add_argument("--epochs", type=int, default=5, help="Stage3 迭代轮数")
    parser.add_argument("--batch-size", type=int, default=256, help="Stage3 批大小")
    parser.add_argument("--knn-k", type=int, default=5, help="KNN 邻居数量")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="运行设备，例如 cpu / cuda",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="输出 .npz 路径；未提供时默认写入 Stage3_Pseudo-label/stage3_output_<HH-MM_YYMMDD>/stage3_pseudo_labels.npz",
    )
    args = parser.parse_args()

    manual_labeled = bool(args.labeled_features_npy and args.labeled_labels_npy) or bool(
        args.labeled_npz
    )
    use_cub_split = (args.labeled_fraction is not None) and (not manual_labeled)
    if use_cub_split and args.cub_root is None:
        args.cub_root = os.path.join(_ROOT, "Data", "CUB_200_2011")
        print(f"未指定 --cub-root，使用默认: {args.cub_root}")
    elif manual_labeled and args.labeled_fraction is not None:
        print("检测到手工 labeled 输入，忽略 --labeled-fraction，直接使用提供的 labeled 参考。")

    if args.alpha < 0.0 or args.alpha > 1.0:
        raise ValueError("--alpha 必须在 [0, 1] 范围内")
    if args.epochs <= 0:
        raise ValueError("--epochs 必须 > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须 > 0")
    if args.knn_k <= 0:
        raise ValueError("--knn-k 必须 > 0")

    device = torch.device(args.device)

    f_full, c_full = load_unlabeled_inputs(args.unlabeled_npz)
    labeled_idx_np: np.ndarray | None = None
    unlabeled_idx_np: np.ndarray | None = None

    if use_cub_split:
        gt_mat, gt_meta = load_cub_train_concept_matrix(args.cub_root)
        if gt_mat.shape[1] != args.num_concepts:
            raise ValueError(
                f"CUB 概念维度 {gt_mat.shape[1]} 与 --num-concepts={args.num_concepts} 不一致"
            )
        f_u, c_u, f_l, y_l, labeled_idx_np, unlabeled_idx_np = split_cub_labeled_fraction(
            f_full,
            c_full,
            gt_mat,
            labeled_fraction=float(args.labeled_fraction),
            seed=int(args.split_seed),
        )
        print(
            f"CUB 划分: labeled={f_l.shape[0]} ({100.0 * f_l.shape[0] / f_full.shape[0]:.2f}%%), "
            f"unlabeled={f_u.shape[0]}, cub_train_N={gt_meta.get('num_train', f_full.shape[0])}; "
            f"labeled 采样与 Stage1 保持一致, cache_hit={gt_meta.get('cache_hit', False)}"
        )
    else:
        f_u, c_u = f_full, c_full
        f_l, y_l = load_labeled_refs(
            unlabeled_features=f_u,
            unlabeled_heatmap=c_u,
            labeled_npz=args.labeled_npz,
            labeled_features_npy=args.labeled_features_npy,
            labeled_labels_npy=args.labeled_labels_npy,
        )

    outputs = run_stage3_epochs(
        features_unlabeled=f_u,
        heatmap_unlabeled=c_u,
        features_labeled=f_l,
        labels_labeled=y_l,
        num_concepts=args.num_concepts,
        alpha=args.alpha,
        epochs=args.epochs,
        batch_size=args.batch_size,
        k=args.knn_k,
        device=device,
    )

    if labeled_idx_np is not None and unlabeled_idx_np is not None:
        outputs["labeled_row_indices"] = labeled_idx_np.astype(np.int64)
        outputs["unlabeled_row_indices"] = unlabeled_idx_np.astype(np.int64)

    if args.out:
        out_path = os.path.abspath(args.out)
    else:
        out_dir = resolve_default_output_dir(
            os.path.dirname(os.path.abspath(__file__)),
            'stage3_output',
        )
        out_path = os.path.join(out_dir, 'stage3_pseudo_labels.npz')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **outputs)

    meta = {
        "unlabeled_npz": os.path.abspath(args.unlabeled_npz),
        "labeled_npz": os.path.abspath(args.labeled_npz) if args.labeled_npz else None,
        "labeled_features_npy": (
            os.path.abspath(args.labeled_features_npy) if args.labeled_features_npy else None
        ),
        "labeled_labels_npy": (
            os.path.abspath(args.labeled_labels_npy) if args.labeled_labels_npy else None
        ),
        "cub_root": os.path.abspath(args.cub_root) if args.cub_root else None,
        "labeled_fraction": args.labeled_fraction,
        "split_seed": args.split_seed if use_cub_split else None,
        "num_labeled_split": int(f_l.shape[0]) if use_cub_split else None,
        "num_unlabeled_split": int(f_u.shape[0]) if use_cub_split else None,
        "cub_cache_hit": bool(gt_meta.get('cache_hit', False)) if use_cub_split else None,
        "cub_cache_path": gt_meta.get('cache_path') if use_cub_split else None,
        "num_concepts": args.num_concepts,
        "alpha": args.alpha,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "knn_k": args.knn_k,
        "device": str(device),
    }
    meta_path = os.path.splitext(out_path)[0] + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Stage 3 伪标签生成完成")
    print(f"  输出文件: {out_path}")
    print(f"  配置文件: {meta_path}")
    print(f"  - c_pseudo: {outputs['c_pseudo'].shape}")
    print(f"  - m_rel: {outputs['m_rel'].shape}")
    print(f"  - reliable_ratio_per_epoch: {outputs['reliable_ratio_per_epoch'].shape}")
    print(f"  - mean_abs_pseudo_per_epoch: {outputs['mean_abs_pseudo_per_epoch'].shape}")


if __name__ == "__main__":
    main()
