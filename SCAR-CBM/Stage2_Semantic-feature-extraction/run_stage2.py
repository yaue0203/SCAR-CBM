#!/usr/bin/env python3
"""
从 train_framework.py 中独立出的 Stage 2 脚本：细粒度空间-语义特征提取。

用法（在 SCAR-CBM 根目录）:
  # 推荐：仅指定 Stage1 输出目录，自动识别 .pt / .npy 邻接矩阵
  python "Stage2-Semantic feature extraction/run_stage2.py" \
      --stage1-out Stage1/stage1_cub_output \
      --input-npy your_images.npy \
      --weights your_model.pt \
      --num-concepts 312

  # 也可手动指定邻接矩阵文件
  python "Stage2-Semantic feature extraction/run_stage2.py" \
      --adj-path Stage1/stage1_cub_output/adj_norm_gcn.pt \
      --input-npy your_images.npy

说明:
  - input-npy 支持 [N, C, H, W] 或 [N, H, W, C]（自动转成 NCHW）
  - 若不传 --input-npy，脚本默认优先读取 Data/CUB_200_2011；不存在时回退为 1 个随机样本自检
  - 输出为 .npz，包含 f_visual / c_heatmap / c_global / c_local / h_aligned / c_graph
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from network_architecture import CompleteModel
from train_framework import ConceptualSSLFramework


def resolve_adj_path(adj_path: str | None, stage1_out: str) -> str:
    if adj_path:
        if os.path.isfile(adj_path):
            return os.path.abspath(adj_path)
        raise FileNotFoundError(f"指定的邻接矩阵文件不存在: {adj_path}")

    stage1_out_abs = os.path.abspath(stage1_out)
    if not os.path.isdir(stage1_out_abs):
        raise FileNotFoundError(f"Stage1 输出目录不存在: {stage1_out_abs}")

    preferred = [
        "adj_norm_gcn.pt",
        "adj_norm_gcn.npy",
        "cooccurrence_raw.npy",
    ]
    for name in preferred:
        p = os.path.join(stage1_out_abs, name)
        if os.path.isfile(p):
            return p

    candidates = []
    for fname in os.listdir(stage1_out_abs):
        lower = fname.lower()
        if lower.endswith(".pt") or lower.endswith(".npy"):
            candidates.append(os.path.join(stage1_out_abs, fname))
    if not candidates:
        raise FileNotFoundError(
            f"在 {stage1_out_abs} 中未找到可用的 .pt 或 .npy 邻接矩阵文件"
        )

    # 优先包含 adj 关键词的文件，其次按文件名排序
    candidates.sort(key=lambda p: (0 if "adj" in os.path.basename(p).lower() else 1, p))
    return candidates[0]


def load_adjacency(path: str, device: torch.device) -> torch.Tensor:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少邻接矩阵文件: {path}")
    if path.endswith(".pt") or path.endswith(".pth"):
        adj = torch.load(path, map_location="cpu")
        if not isinstance(adj, torch.Tensor):
            adj = torch.as_tensor(adj, dtype=torch.float32)
    elif path.endswith(".npy"):
        adj = torch.from_numpy(np.load(path)).float()
    else:
        raise ValueError("adj-path 仅支持 .pt/.pth/.npy")

    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"邻接矩阵形状应为 [C, C]，当前为: {tuple(adj.shape)}")
    return adj.to(device=device, dtype=torch.float32)


def load_images(input_npy: str) -> torch.Tensor:
    if not os.path.isfile(input_npy):
        raise FileNotFoundError(f"缺少输入文件: {input_npy}")
    arr = np.load(input_npy)
    if arr.ndim != 4:
        raise ValueError(f"输入图像应为 4 维张量，当前为: {arr.shape}")

    # 自动支持 NHWC -> NCHW
    if arr.shape[1] in (1, 3):
        nchw = arr
    elif arr.shape[-1] in (1, 3):
        nchw = np.transpose(arr, (0, 3, 1, 2))
    else:
        raise ValueError(
            "无法识别通道维。请提供 [N,C,H,W] 或 [N,H,W,C] 且 C=1/3。"
        )
    return torch.from_numpy(nchw.astype(np.float32))


def load_cub_images(
    cub_root: str,
    split: str = "train",
    image_size: int = 224,
    max_samples: int = 0,
) -> torch.Tensor:
    cub_root_abs = os.path.abspath(cub_root)
    images_txt = os.path.join(cub_root_abs, "images.txt")
    split_txt = os.path.join(cub_root_abs, "train_test_split.txt")
    images_dir = os.path.join(cub_root_abs, "images")
    if not os.path.isfile(images_txt):
        raise FileNotFoundError(f"缺少文件: {images_txt}")
    if not os.path.isfile(split_txt):
        raise FileNotFoundError(f"缺少文件: {split_txt}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"缺少目录: {images_dir}")

    id_to_relpath = {}
    with open(images_txt, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            img_id = int(parts[0])
            id_to_relpath[img_id] = parts[1]

    selected_ids = []
    with open(split_txt, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            img_id, is_train = int(parts[0]), int(parts[1])
            if split == "train" and is_train != 1:
                continue
            if split == "test" and is_train != 0:
                continue
            selected_ids.append(img_id)

    if max_samples > 0:
        selected_ids = selected_ids[:max_samples]

    images = []
    for img_id in selected_ids:
        relpath = id_to_relpath.get(img_id)
        if relpath is None:
            continue
        img_path = os.path.join(images_dir, relpath)
        if not os.path.isfile(img_path):
            continue
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if image_size > 0:
                img = img.resize((image_size, image_size), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, 3]
        images.append(np.transpose(arr, (2, 0, 1)))  # -> [3, H, W]

    if not images:
        raise ValueError(
            f"在 CUB 数据集中未读取到图像。cub_root={cub_root_abs}, split={split}"
        )
    stacked = np.stack(images, axis=0).astype(np.float32)  # [N, 3, H, W]
    return torch.from_numpy(stacked)


def maybe_load_weights(model: torch.nn.Module, weights_path: str | None) -> None:
    if not weights_path:
        return
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"缺少权重文件: {weights_path}")

    ckpt = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError("无法解析权重文件，请提供 state_dict 或包含 state_dict 的 checkpoint。")

    # 兼容 DataParallel 前缀
    cleaned = {}
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith("module.") else k
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"加载权重: {weights_path}")
    print(f"  missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")


@torch.no_grad()
def extract_stage2_features(
    framework: ConceptualSSLFramework,
    images: torch.Tensor,
    adj_matrix: torch.Tensor,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    dataset = TensorDataset(images)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    keys: List[str] = [
        "f_visual",
        "spatial_features",
        "c_global",
        "h_aligned",
        "c_local",
        "c_heatmap",
        "c_graph",
    ]
    collector: Dict[str, List[torch.Tensor]] = {k: [] for k in keys}

    framework.model.eval()
    for (x,) in loader:
        x = x.to(framework.device)
        outputs = framework.extract_features(x, adj_matrix)
        for k in keys:
            v = outputs.get(k)
            if v is None:
                continue
            collector[k].append(v.detach().cpu())

    result: Dict[str, np.ndarray] = {}
    for k, chunks in collector.items():
        if not chunks:
            continue
        result[k] = torch.cat(chunks, dim=0).numpy()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 独立特征提取脚本")
    default_stage1_out = os.path.join(_ROOT, "Stage1", "stage1_cub_output")
    default_cub_root = os.path.join(_ROOT, "Data", "CUB_200_2011")
    parser.add_argument(
        "--adj-path",
        type=str,
        default=None,
        help="邻接矩阵路径（可选，未提供时将从 --stage1-out 自动识别 .pt/.npy）",
    )
    parser.add_argument(
        "--stage1-out",
        type=str,
        default=default_stage1_out,
        help="Stage1 输出目录（用于自动识别邻接矩阵文件）",
    )
    parser.add_argument(
        "--input-npy",
        type=str,
        default=None,
        help="输入图像 .npy（[N,C,H,W] 或 [N,H,W,C]）",
    )
    parser.add_argument(
        "--cub-root",
        type=str,
        default=None,
        help="CUB_200_2011 根目录（提供后可直接读取 CUB 图像作为 Stage2 输入）",
    )
    parser.add_argument(
        "--cub-split",
        type=str,
        choices=["train", "test", "all"],
        default="train",
        help="CUB 数据划分（仅在 --cub-root 提供时生效）",
    )
    parser.add_argument(
        "--cub-max-samples",
        type=int,
        default=0,
        help="最多读取多少张 CUB 图像；0 表示全部（仅在 --cub-root 提供时生效）",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="模型权重路径（可选）",
    )
    parser.add_argument("--num-concepts", type=int, default=312, help="概念数")
    parser.add_argument("--feature-dim", type=int, default=256, help="视觉特征维度")
    parser.add_argument("--num-heads", type=int, default=4, help="跨注意力头数")
    parser.add_argument("--batch-size", type=int, default=32, help="推理 batch 大小")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="运行设备，例如 cpu / cuda",
    )
    parser.add_argument(
        "--random-samples",
        type=int,
        default=0,
        help="若 >0，则忽略 input-npy，生成随机样本数量用于自检",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="随机样本图像尺寸（random-samples>0 时生效）",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="输出 .npz 路径，默认 Stage2-Semantic feature extraction/stage2_output/stage2_features.npz",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model = CompleteModel(
        input_dim=3,
        feature_dim=args.feature_dim,
        num_concepts=args.num_concepts,
        num_heads=args.num_heads,
    )
    maybe_load_weights(model, args.weights)

    framework = ConceptualSSLFramework(
        model=model,
        device=args.device,
        num_concepts=args.num_concepts,
        num_epochs=1,
    )

    adj_path = resolve_adj_path(args.adj_path, args.stage1_out)
    print(f"使用邻接矩阵: {adj_path}")
    adj_matrix = load_adjacency(adj_path, framework.device)
    if adj_matrix.shape[0] != args.num_concepts:
        raise ValueError(
            f"num_concepts={args.num_concepts} 与邻接矩阵大小 {tuple(adj_matrix.shape)} 不一致"
        )

    effective_cub_root = args.cub_root
    effective_random_samples = args.random_samples
    if not args.input_npy and not effective_cub_root and effective_random_samples <= 0:
        if os.path.isdir(default_cub_root):
            effective_cub_root = default_cub_root
            print(f"未提供 --input-npy，默认读取 CUB 数据: {default_cub_root}")
        else:
            # 若默认 CUB 目录不存在，则做一次最小随机自检，避免直接报错。
            effective_random_samples = 1
            print(
                "未提供 --input-npy 且默认 CUB 目录不存在，"
                "默认使用 1 个随机样本进行 Stage2 自检。"
            )

    if effective_cub_root:
        images = load_cub_images(
            cub_root=effective_cub_root,
            split=args.cub_split,
            image_size=args.image_size,
            max_samples=args.cub_max_samples,
        )
        print(
            f"加载 CUB 输入: root={os.path.abspath(effective_cub_root)}, "
            f"split={args.cub_split}, images={tuple(images.shape)}"
        )
    elif effective_random_samples > 0:
        n = effective_random_samples
        s = args.image_size
        images = torch.randn(n, 3, s, s, dtype=torch.float32)
        print(f"使用随机输入: images={tuple(images.shape)}")
    else:
        if not args.input_npy:
            raise ValueError("未提供输入。请设置 --input-npy，或使用 --random-samples。")
        images = load_images(args.input_npy)
        print(f"加载输入: {args.input_npy}, images={tuple(images.shape)}")

    outputs = extract_stage2_features(
        framework=framework,
        images=images,
        adj_matrix=adj_matrix,
        batch_size=args.batch_size,
    )

    default_out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "stage2_output",
        "stage2_features.npz",
    )
    out_path = os.path.abspath(args.out or default_out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **outputs)

    print("Stage 2 特征提取完成")
    print(f"  输出文件: {out_path}")
    print(f"  使用矩阵: {os.path.basename(adj_path)}")
    print(f"  矩阵路径: {adj_path}")
    for k, v in outputs.items():
        print(f"  - {k}: {v.shape}")


if __name__ == "__main__":
    main()
