#!/usr/bin/env python3
"""
从 train_framework.py 中独立出的 Stage 4 脚本：不确定性引导的反课程采样。

对应:
  - `ConceptualSSLFramework.sample_unlabeled_data`
  - `pseudo_label_and_sampling.UncertaintyGuidedSampler`

典型用法（在 SCAR-CBM 根目录）:
  python "Stage4_Anti-curriculum/run_stage4.py" \\
      --unlabeled-npz "Stage2_Semantic-feature-extraction/stage2_output/stage2_features.npz" \\
      --batch-size 256

说明:
  - 输入需与 Stage2 输出一致：包含 f_visual [N, D]、c_heatmap [N, C]（作为 logits 参与熵与采样权重）
  - 输出为采样的样本行索引（相对该 npz 中的无标签集合顺序）
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

from pseudo_label_and_sampling import UncertaintyGuidedSampler
from stage_output_utils import find_latest_output_file, resolve_default_output_dir


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少文件: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def load_stage4_inputs(npz_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """读取 Stage2 风格 npz：f_visual + c_heatmap（视作 logits）。"""
    payload = _load_npz(npz_path)
    if "f_visual" not in payload:
        raise KeyError(f"{npz_path} 中缺少键: f_visual")
    if "c_heatmap" not in payload:
        raise KeyError(f"{npz_path} 中缺少键: c_heatmap")

    f_visual = torch.from_numpy(payload["f_visual"].astype(np.float32))
    logits = torch.from_numpy(payload["c_heatmap"].astype(np.float32))
    if f_visual.ndim != 2 or logits.ndim != 2:
        raise ValueError(
            f"输入维度错误: f_visual={tuple(f_visual.shape)}, c_heatmap={tuple(logits.shape)}"
        )
    if f_visual.shape[0] != logits.shape[0]:
        raise ValueError("f_visual 与 c_heatmap 的样本数不一致")
    return f_visual, logits


@torch.no_grad()
def run_stage4_sample(
    features: torch.Tensor,
    logits: torch.Tensor,
    batch_size: int,
    low_density_ratio: float,
    uncertainty_weight: float,
    num_repeats: int,
    device: torch.device,
) -> np.ndarray:
    """
    执行反课程采样，返回索引数组。

    Returns:
        indices: shape [num_repeats, batch_size]（num_repeats==1 时为 [batch_size]）
    """
    n = features.shape[0]
    if n < 2:
        raise ValueError("样本数至少为 2，k-NN 密度估计才有意义")
    if batch_size <= 0:
        raise ValueError("--batch-size 必须 > 0")
    if num_repeats <= 0:
        raise ValueError("--num-repeats 必须 > 0")

    sampler = UncertaintyGuidedSampler(
        low_density_ratio=low_density_ratio,
        uncertainty_weight=uncertainty_weight,
    )
    f = features.to(device)
    lg = logits.to(device)

    rounds = []
    for _ in range(num_repeats):
        idx = sampler.sample(f, lg, batch_size)
        rounds.append(idx.astype(np.int64))
    if num_repeats == 1:
        return rounds[0]
    return np.stack(rounds, axis=0)


def main() -> None:
    default_unlabeled_npz = find_latest_output_file(
        os.path.join(_ROOT, 'Stage2_Semantic-feature-extraction'),
        'stage2_output',
        'stage2_features.npz',
    )

    parser = argparse.ArgumentParser(description="Stage 4 反课程采样（独立脚本）")
    parser.add_argument(
        "--unlabeled-npz",
        type=str,
        default=default_unlabeled_npz,
        help="Stage2 输出 .npz（默认自动读取最新的 stage2_output_<HH-MM_YYMMDD>/stage2_features.npz）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="每轮采样的索引数量（对应训练时无标签 mini-batch 大小）",
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=1,
        help="独立重复采样轮数（不同随机种子下的多次 draw）",
    )
    parser.add_argument(
        "--low-density-ratio",
        type=float,
        default=0.2,
        help="UncertaintyGuidedSampler 的 low_density_ratio（τ_low）",
    )
    parser.add_argument(
        "--uncertainty-weight",
        type=float,
        default=1.0,
        help="UncertaintyGuidedSampler 的不确定性权重系数",
    )
    parser.add_argument("--seed", type=int, default=None, help="numpy 随机种子（可复现采样）")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="计算设备（密度与距离在 device 上计算）",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="输出 .npz 路径；未提供时默认写入 Stage4_Anti-curriculum/stage4_output_<HH-MM_YYMMDD>/stage4_sampled_indices.npz",
    )
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    device = torch.device(args.device)
    f_visual, logits = load_stage4_inputs(args.unlabeled_npz)

    indices = run_stage4_sample(
        features=f_visual,
        logits=logits,
        batch_size=args.batch_size,
        low_density_ratio=args.low_density_ratio,
        uncertainty_weight=args.uncertainty_weight,
        num_repeats=args.num_repeats,
        device=device,
    )

    if args.out:
        out_path = os.path.abspath(args.out)
    else:
        out_dir = resolve_default_output_dir(
            os.path.dirname(os.path.abspath(__file__)),
            'stage4_output',
        )
        out_path = os.path.join(out_dir, 'stage4_sampled_indices.npz')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    payload: Dict[str, np.ndarray] = {
        "sampled_indices": indices,
        "num_samples": np.array([f_visual.shape[0]], dtype=np.int64),
    }
    np.savez_compressed(out_path, **payload)

    meta = {
        "unlabeled_npz": os.path.abspath(args.unlabeled_npz),
        "num_samples": int(f_visual.shape[0]),
        "feature_dim": int(f_visual.shape[1]),
        "num_concepts": int(logits.shape[1]),
        "batch_size": args.batch_size,
        "num_repeats": args.num_repeats,
        "low_density_ratio": args.low_density_ratio,
        "uncertainty_weight": args.uncertainty_weight,
        "seed": args.seed,
        "device": str(device),
    }
    meta_path = os.path.splitext(out_path)[0] + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Stage 4 反课程采样完成")
    print(f"  输出文件: {out_path}")
    print(f"  配置文件: {meta_path}")
    print(f"  - sampled_indices: {payload['sampled_indices'].shape}")
    print(f"  - N={f_visual.shape[0]}, batch_size={args.batch_size}")


if __name__ == "__main__":
    main()
