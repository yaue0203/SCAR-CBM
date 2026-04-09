"""
主训练框架：Stage 2～5 分阶段流程可在此串联执行，也可单独使用 `ConceptualSSLFramework`。

- Stage 2：语义特征提取（CUB train → npz）
- Stage 3：伪标签（Stage2 npz + CUB 划分 → npz）
- Stage 4：反课程采样索引（Stage2 npz → npz）
- Stage 5：图逻辑优化训练（Stage2/3/4 npz + CUB 图像 → `fit`）

用法（在 SCAR-CBM 根目录）::

  python train_framework.py --mode pipeline --cub-root Data/CUB_200_2011

  # 跳过已跑过的阶段（需磁盘上已有对应 npz）
  python train_framework.py --mode pipeline --skip-stage2 --skip-stage3 --skip-stage4

  # 仅 Stage5（等价于原 run_stage5_cub）
  python train_framework.py --mode stage5-only
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from argparse import Namespace
from typing import Any, Optional

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
_STAGE5_DIR = os.path.join(_ROOT, "Stage5_Graph-logic_Optimization")
for _p in (_ROOT, _STAGE5_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from network_architecture import CompleteModel
from stage5_framework import ConceptualSSLFramework, create_example_data

__all__ = [
    "ConceptualSSLFramework",
    "create_example_data",
    "run_pipeline_stages_2_to_5",
    "load_external_module",
]


def load_external_module(path: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _default_paths() -> dict:
    return {
        "stage2_npz": os.path.join(
            _ROOT,
            "Stage2_Semantic-feature-extraction",
            "stage2_output",
            "stage2_features.npz",
        ),
        "stage3_npz": os.path.join(
            _ROOT,
            "Stage3_Pseudo-label",
            "stage3_output",
            "stage3_pseudo_labels.npz",
        ),
        "stage4_npz": os.path.join(
            _ROOT,
            "Stage4_Anti-curriculum",
            "stage4_output",
            "stage4_sampled_indices.npz",
        ),
        "stage5_out": os.path.join(
            _ROOT,
            "Stage5_Graph-logic_Optimization",
            "stage5_output",
        ),
        "stage1_out": os.path.join(_ROOT, "Stage1", "stage1_cub_output"),
        "cub_root": os.path.join(_ROOT, "Data", "CUB_200_2011"),
    }


def run_stage2_extract(
    *,
    cub_root: str,
    stage1_out: str,
    adj_path: Optional[str],
    stage2_npz: str,
    weights: Optional[str],
    num_concepts: int,
    feature_dim: int,
    num_heads: int,
    batch_size: int,
    device: str,
    image_size: int,
) -> None:
    s2 = load_external_module(
        os.path.join(_ROOT, "Stage2_Semantic-feature-extraction", "run_stage2.py"),
        "run_stage2",
    )
    model = CompleteModel(
        input_dim=3,
        feature_dim=feature_dim,
        num_concepts=num_concepts,
        num_heads=num_heads,
    )
    w = weights
    if w and not os.path.isfile(w):
        print(f"警告: Stage2 未找到权重 {w}，随机初始化。", flush=True)
        w = None
    s2.maybe_load_weights(model, w)
    framework = ConceptualSSLFramework(
        model=model,
        device=device,
        num_concepts=num_concepts,
        num_epochs=1,
    )
    ap = s2.resolve_adj_path(adj_path, stage1_out)
    print(f"[Stage2] 邻接矩阵: {ap}", flush=True)
    adj_matrix = s2.load_adjacency(ap, framework.device)
    if adj_matrix.shape[0] != num_concepts:
        raise ValueError(
            f"num_concepts={num_concepts} 与邻接矩阵 {tuple(adj_matrix.shape)} 不一致"
        )
    images = s2.load_cub_images(
        cub_root=cub_root,
        split="train",
        image_size=image_size,
        max_samples=0,
    )
    print(f"[Stage2] CUB train 图像: {tuple(images.shape)}", flush=True)
    outputs = s2.extract_stage2_features(
        framework=framework,
        images=images,
        adj_matrix=adj_matrix,
        batch_size=batch_size,
    )
    os.makedirs(os.path.dirname(os.path.abspath(stage2_npz)), exist_ok=True)
    np.savez_compressed(stage2_npz, **outputs)
    print(f"[Stage2] 已写入: {os.path.abspath(stage2_npz)}", flush=True)


def run_stage3_pseudo(
    *,
    stage2_npz: str,
    stage3_npz: str,
    cub_root: str,
    labeled_fraction: float,
    split_seed: int,
    num_concepts: int,
    alpha: float,
    epochs: int,
    batch_size: int,
    knn_k: int,
    device: str,
) -> None:
    s3 = load_external_module(
        os.path.join(_ROOT, "Stage3_Pseudo-label", "run_stage3.py"),
        "run_stage3",
    )
    f_full, c_full = s3.load_unlabeled_inputs(stage2_npz)
    load_gt = s3._load_cub_train_concept_loader()
    gt_mat, _gt_meta = load_gt(cub_root)
    f_u, c_u, f_l, y_l, lab_idx, unl_idx = s3.split_cub_labeled_fraction(
        f_full,
        c_full,
        gt_mat,
        labeled_fraction=labeled_fraction,
        seed=split_seed,
    )
    outputs = s3.run_stage3_epochs(
        features_unlabeled=f_u,
        heatmap_unlabeled=c_u,
        features_labeled=f_l,
        labels_labeled=y_l,
        num_concepts=num_concepts,
        alpha=alpha,
        epochs=epochs,
        batch_size=batch_size,
        k=knn_k,
        device=torch.device(device),
    )
    outputs["labeled_row_indices"] = lab_idx.astype(np.int64)
    outputs["unlabeled_row_indices"] = unl_idx.astype(np.int64)
    os.makedirs(os.path.dirname(os.path.abspath(stage3_npz)), exist_ok=True)
    np.savez_compressed(stage3_npz, **outputs)
    print(f"[Stage3] 已写入: {os.path.abspath(stage3_npz)}", flush=True)


def run_stage4_sample(
    *,
    stage2_npz: str,
    stage4_npz: str,
    batch_size: int,
    low_density_ratio: float,
    uncertainty_weight: float,
    num_repeats: int,
    seed: Optional[int],
    device: str,
) -> None:
    s4 = load_external_module(
        os.path.join(_ROOT, "Stage4_Anti-curriculum", "run_stage4.py"),
        "run_stage4",
    )
    if seed is not None:
        np.random.seed(seed)
    f_vis, logits = s4.load_stage4_inputs(stage2_npz)
    indices = s4.run_stage4_sample(
        features=f_vis,
        logits=logits,
        batch_size=batch_size,
        low_density_ratio=low_density_ratio,
        uncertainty_weight=uncertainty_weight,
        num_repeats=num_repeats,
        device=torch.device(device),
    )
    os.makedirs(os.path.dirname(os.path.abspath(stage4_npz)), exist_ok=True)
    payload = {
        "sampled_indices": indices,
        "num_samples": np.array([f_vis.shape[0]], dtype=np.int64),
    }
    np.savez_compressed(stage4_npz, **payload)
    meta = {
        "unlabeled_npz": os.path.abspath(stage2_npz),
        "num_samples": int(f_vis.shape[0]),
        "batch_size": batch_size,
        "num_repeats": num_repeats,
        "low_density_ratio": low_density_ratio,
        "uncertainty_weight": uncertainty_weight,
        "seed": seed,
        "device": device,
    }
    with open(os.path.splitext(stage4_npz)[0] + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[Stage4] 已写入: {os.path.abspath(stage4_npz)}", flush=True)


def run_pipeline_stages_2_to_5(ns: Namespace) -> None:
    """
    按顺序执行 Stage2 → 3 → 4 → 5（可通过 ns.skip_stage* 跳过，但后续阶段需已有 npz）。
    """
    d = _default_paths()
    stage2_npz = ns.stage2_npz or d["stage2_npz"]
    stage3_npz = ns.stage3_npz or d["stage3_npz"]
    stage4_npz = ns.stage4_npz or d["stage4_npz"]
    stage5_out = ns.stage5_output_dir or d["stage5_out"]

    if not ns.skip_stage2:
        run_stage2_extract(
            cub_root=ns.cub_root,
            stage1_out=ns.stage1_out,
            adj_path=ns.adj_path,
            stage2_npz=stage2_npz,
            weights=ns.weights,
            num_concepts=ns.num_concepts,
            feature_dim=ns.feature_dim,
            num_heads=ns.num_heads,
            batch_size=ns.stage2_batch_size,
            device=ns.device,
            image_size=ns.image_size,
        )
    elif not os.path.isfile(stage2_npz):
        raise FileNotFoundError(f"--skip-stage2 但缺少 {stage2_npz}")

    if not ns.skip_stage3:
        run_stage3_pseudo(
            stage2_npz=stage2_npz,
            stage3_npz=stage3_npz,
            cub_root=ns.cub_root,
            labeled_fraction=ns.labeled_fraction,
            split_seed=ns.split_seed,
            num_concepts=ns.num_concepts,
            alpha=ns.stage3_alpha,
            epochs=ns.stage3_epochs,
            batch_size=ns.stage3_batch_size,
            knn_k=ns.knn_k,
            device=ns.device,
        )
    elif not os.path.isfile(stage3_npz):
        raise FileNotFoundError(f"--skip-stage3 但缺少 {stage3_npz}")

    if not ns.skip_stage4:
        run_stage4_sample(
            stage2_npz=stage2_npz,
            stage4_npz=stage4_npz,
            batch_size=ns.stage4_batch_size,
            low_density_ratio=ns.low_density_ratio,
            uncertainty_weight=ns.uncertainty_weight,
            num_repeats=ns.stage4_num_repeats,
            seed=ns.stage4_seed,
            device=ns.device,
        )
    elif ns.use_stage4_pool and not os.path.isfile(stage4_npz):
        raise FileNotFoundError(f"--use-stage4-pool 需要 {stage4_npz} 或去掉 --skip-stage4")

    r5c = load_external_module(
        os.path.join(_STAGE5_DIR, "run_stage5_cub.py"),
        "run_stage5_cub",
    )
    s5 = Namespace(
        cub_root=ns.cub_root,
        stage1_out=ns.stage1_out,
        adj_path=ns.adj_path,
        stage2_npz=stage2_npz,
        stage3_npz=stage3_npz,
        stage4_npz=stage4_npz if ns.use_stage4_pool else None,
        use_stage4_pool=bool(ns.use_stage4_pool),
        weights=ns.weights,
        num_concepts=ns.num_concepts,
        feature_dim=ns.feature_dim,
        num_heads=ns.num_heads,
        num_epochs=ns.num_epochs,
        batch_size_labeled=ns.batch_size_labeled,
        batch_size_unlabeled=ns.batch_size_unlabeled,
        learning_rate=ns.learning_rate,
        lambda_spatial_diversity=ns.lambda_spatial_diversity,
        image_size=ns.image_size,
        val_split=ns.val_split,
        seed=ns.seed,
        device=ns.device,
        output_dir=stage5_out,
        save_weights=ns.save_weights_resolved,
    )
    r5c.run_stage5_cub_training(s5)


def _build_parser() -> argparse.ArgumentParser:
    d = _default_paths()
    p = argparse.ArgumentParser(
        description="SCAR-CBM：Stage2-5 串联（pipeline）或仅 Stage5（stage5-only）"
    )
    p.add_argument(
        "--mode",
        choices=["pipeline", "stage5-only"],
        default="pipeline",
        help="pipeline: 依次跑 Stage2→5；stage5-only: 只跑 Stage5（需已有 Stage2/3 npz）",
    )
    p.add_argument("--cub-root", type=str, default=d["cub_root"])
    p.add_argument("--stage1-out", type=str, default=d["stage1_out"])
    p.add_argument("--adj-path", type=str, default=None)
    p.add_argument("--stage2-npz", type=str, default=d["stage2_npz"])
    p.add_argument("--stage3-npz", type=str, default=d["stage3_npz"])
    p.add_argument("--stage4-npz", type=str, default=d["stage4_npz"])
    p.add_argument(
        "--stage5-output-dir",
        type=str,
        default=d["stage5_out"],
        help="Stage5 日志与可选权重输出目录",
    )
    p.add_argument("--skip-stage2", action="store_true")
    p.add_argument("--skip-stage3", action="store_true")
    p.add_argument("--skip-stage4", action="store_true")
    p.add_argument("--use-stage4-pool", action="store_true")
    p.add_argument("--weights", type=str, default=None)
    p.add_argument(
        "--no-save-weights",
        action="store_true",
        help="不在 Stage5 结束时保存 complete_model.pt（默认会保存到 stage5_output_dir）",
    )
    p.add_argument("--num-concepts", type=int, default=312)
    p.add_argument("--feature-dim", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    # Stage2
    p.add_argument("--stage2-batch-size", type=int, default=32)
    # Stage3
    p.add_argument("--labeled-fraction", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--stage3-alpha", type=float, default=0.7)
    p.add_argument("--stage3-epochs", type=int, default=5)
    p.add_argument("--stage3-batch-size", type=int, default=256)
    p.add_argument("--knn-k", type=int, default=5)
    # Stage4
    p.add_argument("--stage4-batch-size", type=int, default=256)
    p.add_argument("--stage4-num-repeats", type=int, default=1)
    p.add_argument("--low-density-ratio", type=float, default=0.2)
    p.add_argument("--uncertainty-weight", type=float, default=1.0)
    p.add_argument("--stage4-seed", type=int, default=None)
    # Stage5
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument("--batch-size-labeled", type=int, default=32)
    p.add_argument("--batch-size-unlabeled", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument(
        "--lambda-spatial-diversity",
        type=float,
        default=0.3,
        help="Stage5 概念空间热力图去同质化损失权重",
    )
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument(
        "--save-weights",
        type=str,
        default=None,
        help="显式指定 Stage5 保存权重的 .pt 路径；默认写入 stage5_output_dir/complete_model.pt",
    )
    return p


def _resolve_save_weights_path(args: argparse.Namespace) -> Optional[str]:
    if args.no_save_weights:
        return None
    if args.save_weights:
        return args.save_weights
    return os.path.join(args.stage5_output_dir, "complete_model.pt")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    os.makedirs(args.stage5_output_dir, exist_ok=True)
    save_resolved = _resolve_save_weights_path(args)

    ns = Namespace(**vars(args))
    ns.save_weights_resolved = save_resolved

    if args.mode == "pipeline":
        run_pipeline_stages_2_to_5(ns)
    else:
        r5c = load_external_module(
            os.path.join(_STAGE5_DIR, "run_stage5_cub.py"),
            "run_stage5_cub",
        )
        s5 = Namespace(
            cub_root=ns.cub_root,
            stage1_out=ns.stage1_out,
            adj_path=ns.adj_path,
            stage2_npz=ns.stage2_npz,
            stage3_npz=ns.stage3_npz,
            stage4_npz=ns.stage4_npz if ns.use_stage4_pool else None,
            use_stage4_pool=bool(ns.use_stage4_pool),
            weights=ns.weights,
            num_concepts=ns.num_concepts,
            feature_dim=ns.feature_dim,
            num_heads=ns.num_heads,
            num_epochs=ns.num_epochs,
            batch_size_labeled=ns.batch_size_labeled,
            batch_size_unlabeled=ns.batch_size_unlabeled,
            learning_rate=ns.learning_rate,
            lambda_spatial_diversity=ns.lambda_spatial_diversity,
            image_size=ns.image_size,
            val_split=ns.val_split,
            seed=ns.seed,
            device=ns.device,
            output_dir=ns.stage5_output_dir,
            save_weights=save_resolved,
        )
        r5c.run_stage5_cub_training(s5)


if __name__ == "__main__":
    main()
