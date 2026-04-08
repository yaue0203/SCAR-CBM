#!/usr/bin/env python3
"""
从 train_framework 中独立出的 Stage 5 脚本：图逻辑优化（完整训练循环）。

对应:
  - `ConceptualSSLFramework`：`train_epoch` / `validate` / `fit`
  - 多任务损失组合（L_align / L_consistency / L_geo）与伪标签、超图邻接

典型用法（在 SCAR-CBM 根目录）:
  python "Stage5_Graph-logic_Optimization/run_stage5.py" --demo

  # 小规模快速自检（少 epoch）
  python "Stage5_Graph-logic_Optimization/run_stage5.py" --num-epochs 2 --num-labeled 32
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STAGE5 = os.path.dirname(os.path.abspath(__file__))
for p in (_ROOT, _STAGE5):
    if p not in sys.path:
        sys.path.insert(0, p)

from core_modules import LabeledDataset, UnlabeledDataset
from network_architecture import CompleteModel
from stage5_framework import ConceptualSSLFramework, create_example_data
from stage_output_utils import resolve_default_output_dir


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_data_loaders(
    data_dict: dict,
    batch_size_labeled: int,
    batch_size_unlabeled: int,
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple:
    X_labeled = data_dict["X_labeled"]
    y_labeled = data_dict["y_labeled"]
    c_labeled = data_dict["c_labeled"]
    X_unlabeled = data_dict["X_unlabeled"]

    num_labeled = len(X_labeled)
    num_val = int(num_labeled * val_split)

    rng = np.random.RandomState(seed)
    indices = rng.permutation(num_labeled)
    train_indices = indices[num_val:]
    val_indices = indices[:num_val]

    X_labeled_train = X_labeled[train_indices]
    y_labeled_train = y_labeled[train_indices]
    c_labeled_train = c_labeled[train_indices]

    X_labeled_val = X_labeled[val_indices]
    y_labeled_val = y_labeled[val_indices]
    c_labeled_val = c_labeled[val_indices]

    train_labeled_dataset = LabeledDataset(
        X_labeled_train, y_labeled_train, c_labeled_train
    )
    train_unlabeled_dataset = UnlabeledDataset(X_unlabeled)
    val_dataset = LabeledDataset(
        X_labeled_val, y_labeled_val, c_labeled_val
    )

    train_labeled_loader = DataLoader(
        train_labeled_dataset,
        batch_size=batch_size_labeled,
        shuffle=True,
        num_workers=0,
    )
    train_unlabeled_loader = DataLoader(
        train_unlabeled_dataset,
        batch_size=batch_size_unlabeled,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_labeled,
        shuffle=False,
        num_workers=0,
    )

    return train_labeled_loader, train_unlabeled_loader, val_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5 图逻辑优化（独立训练入口）")
    parser.add_argument("--demo", action="store_true", help="等价于默认随机数据演示")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-concepts", type=int, default=10)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--num-labeled", type=int, default=100)
    parser.add_argument("--num-unlabeled", type=int, default=400)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-size-labeled", type=int, default=32)
    parser.add_argument("--batch-size-unlabeled", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="训练历史摘要输出目录；未提供时默认写入 Stage5_Graph-logic_Optimization/stage5_output_<HH-MM_YYMMDD>/",
    )
    args = parser.parse_args()

    if not args.demo and len(sys.argv) == 1:
        parser.print_help()
        print("\n提示: 使用 --demo 运行随机数据演示。")
        return

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CompleteModel(
        input_dim=3,
        feature_dim=256,
        num_concepts=args.num_concepts,
        num_heads=4,
    )

    data_dict = create_example_data(
        num_labeled=args.num_labeled,
        num_unlabeled=args.num_unlabeled,
        num_concepts=args.num_concepts,
        image_size=args.image_size,
    )

    train_labeled_loader, train_unlabeled_loader, val_loader = create_data_loaders(
        data_dict,
        batch_size_labeled=args.batch_size_labeled,
        batch_size_unlabeled=args.batch_size_unlabeled,
        seed=args.seed,
    )

    framework = ConceptualSSLFramework(
        model=model,
        device=device,
        num_concepts=args.num_concepts,
        num_epochs=args.num_epochs,
        batch_size_labeled=args.batch_size_labeled,
        batch_size_unlabeled=args.batch_size_unlabeled,
        learning_rate=args.learning_rate,
    )

    history = framework.fit(
        train_labeled_loader=train_labeled_loader,
        train_unlabeled_loader=train_unlabeled_loader,
        val_loader=val_loader,
        labeled_concept_ids=data_dict["c_labeled"],
        contour_mask=data_dict["contour_mask"],
    )

    output_dir = resolve_default_output_dir(
        os.path.dirname(os.path.abspath(__file__)),
        'stage5_output',
        args.output_dir,
    )
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "stage5_training_summary.json")
        serializable = {k: [float(x) for x in v] for k, v in history.items()}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"已写入: {out_path}")


if __name__ == "__main__":
    main()
