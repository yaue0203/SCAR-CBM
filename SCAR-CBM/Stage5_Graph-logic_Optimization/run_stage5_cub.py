#!/usr/bin/env python3
"""
Stage 5（CUB）：用 Stage2 / Stage3 / Stage4 的 npz 输出驱动 `ConceptualSSLFramework.fit`。

前提（与当前仓库脚本一致）:
  - Stage2 对 CUB **train** 全量按固定顺序导出 `stage2_features.npz`（行顺序 = `run_stage2.load_cub_images` = Stage1 概念矩阵行顺序）。
  - Stage3 默认会按 CUB train 的 10%% / 90%% 划分有标签/无标签，并在 `stage3_pseudo_labels.npz` 中写入
    `labeled_row_indices` / `unlabeled_row_indices`（相对上述全量 train 的行号）；也可用 `--labeled-fraction` 显式改比例。
  - Stage4 默认在 **全量** Stage2 npz（N = CUB train 样本数）上采样，得到 **全局行号** `sampled_indices`；
    若你在「仅无标签」子集上另存了 npz 再跑 Stage4，则索引为 **局部**（`num_samples` = 无标签子集大小）。

无标签 DataLoader:
  - 默认使用 Stage3 划分出的全部无标签图像；
  - 若提供 `--stage4-npz` 且 `--use-stage4-pool`，则仅保留落在无标签子集内的 Stage4 索引并去重，作为无标签子集（反课程池）；若映射后为空则回退为全量无标签并告警。

典型用法（在 SCAR-CBM 根目录）:

  python "Stage5_Graph-logic_Optimization/run_stage5_cub.py" \\
      --cub-root Data/CUB_200_2011 \\
      --stage1-out Stage1/stage1_cub_output \\
      --stage2-npz Stage2_Semantic-feature-extraction/stage2_output/stage2_features.npz \\
      --stage3-npz Stage3_Pseudo-label/stage3_output/stage3_pseudo_labels.npz \\
      --stage4-npz Stage4_Anti-curriculum/stage4_output/stage4_sampled_indices.npz \\
      --weights path/to/stage2_model.pt \\
      --use-stage4-pool \\
      --num-epochs 5
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tarfile
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STAGE5 = os.path.dirname(os.path.abspath(__file__))
for p in (_ROOT, _STAGE5):
    if p not in sys.path:
        sys.path.insert(0, p)

from network_architecture import CompleteModel
from stage5_framework import ConceptualSSLFramework
from stage_output_utils import find_latest_output_dir, find_latest_output_file, resolve_default_output_dir


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_stage2_helpers():
    stage2_py = os.path.join(_ROOT, "Stage2_Semantic-feature-extraction", "run_stage2.py")
    m = _load_module(stage2_py, "run_stage2")
    return m.load_cub_images, m.load_adjacency, m.resolve_adj_path, m.maybe_load_weights


def _load_cub_gt_loader():
    from stage1_data_utils import load_cub_train_concept_matrix
    return load_cub_train_concept_matrix


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少文件: {path}")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _maybe_extract_segmentations(cub_root: str) -> str:
    cub_root_abs = os.path.abspath(cub_root)
    seg_dir = os.path.join(cub_root_abs, "segmentations")
    if os.path.isdir(seg_dir):
        return seg_dir

    archive = os.path.join(os.path.dirname(cub_root_abs), "segmentations.tgz")
    if not os.path.isfile(archive):
        raise FileNotFoundError(
            f"缺少 segmentation 目录且未找到压缩包: {archive}"
        )

    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(path=cub_root_abs)
    if not os.path.isdir(seg_dir):
        raise FileNotFoundError(f"解压后仍未找到 segmentation 目录: {seg_dir}")
    return seg_dir


def load_cub_segmentations(
    cub_root: str,
    split: str = "train",
    image_size: int = 224,
    max_samples: int = 0,
) -> np.ndarray:
    cub_root_abs = os.path.abspath(cub_root)
    images_txt = os.path.join(cub_root_abs, "images.txt")
    split_txt = os.path.join(cub_root_abs, "train_test_split.txt")
    seg_dir = _maybe_extract_segmentations(cub_root_abs)

    id_to_relpath = {}
    with open(images_txt, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            id_to_relpath[int(parts[0])] = parts[1]

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

    masks = []
    for img_id in selected_ids:
        relpath = id_to_relpath.get(img_id)
        if relpath is None:
            continue
        mask_rel = os.path.splitext(relpath)[0] + ".png"
        mask_path = os.path.join(seg_dir, mask_rel)
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"缺少 segmentation mask: {mask_path}")
        with Image.open(mask_path) as img:
            img = img.convert("L")
            if image_size > 0:
                img = img.resize((image_size, image_size), Image.NEAREST)
            arr = (np.asarray(img, dtype=np.uint8) > 0).astype(np.uint8)
        masks.append(arr[None, ...])

    if not masks:
        raise ValueError(
            f"在 CUB segmentation 中未读取到 mask。cub_root={cub_root_abs}, split={split}"
        )
    return np.stack(masks, axis=0)


def _random_horizontal_flip(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    if torch.rand(()) < p:
        x = torch.flip(x, dims=(2,))
    return x


def _weak_augment(x: torch.Tensor) -> torch.Tensor:
    x = x.float().clamp(0.0, 1.0)
    return _random_horizontal_flip(x, p=0.5)


def _strong_augment(x: torch.Tensor) -> torch.Tensor:
    x = x.float().clamp(0.0, 1.0)
    x = _random_horizontal_flip(x, p=0.5)

    if torch.rand(()) < 0.8:
        brightness = 0.7 + 0.6 * torch.rand((), dtype=x.dtype)
        x = x * brightness

    if torch.rand(()) < 0.8:
        mean = x.mean(dim=(1, 2), keepdim=True)
        contrast = 0.7 + 0.6 * torch.rand((), dtype=x.dtype)
        x = (x - mean) * contrast + mean

    if torch.rand(()) < 0.3:
        gray = x.mean(dim=0, keepdim=True)
        x = gray.repeat(x.shape[0], 1, 1)

    if torch.rand(()) < 0.5:
        noise = 0.03 * torch.randn_like(x)
        x = x + noise

    return x.clamp(0.0, 1.0)


def _map_stage4_to_unlabeled_local(
    sampled: np.ndarray,
    unlabeled_row_indices: np.ndarray,
    num_source_rows: int,
) -> np.ndarray:
    """将 Stage4 的 sampled_indices 映射为无标签子集内的局部下标（0..Nu-1）。"""
    unl = unlabeled_row_indices.astype(np.int64).ravel()
    pos_global = {int(g): i for i, g in enumerate(unl)}
    flat = sampled.astype(np.int64).ravel()
    nu = len(unl)

    if num_source_rows == nu:
        out = [int(g) for g in flat if 0 <= int(g) < nu]
        return np.unique(np.asarray(out, dtype=np.int64))

    # 假定 num_source_rows 为全量 CUB train 行号（与默认 Stage2 / Stage4 一致）
    out = [pos_global[int(g)] for g in flat if int(g) in pos_global]
    return np.unique(np.asarray(out, dtype=np.int64))


class _HypergraphAdjWrapper:
    """用 Stage1 导出的邻接矩阵满足 `fit(..., prebuilt_hypergraph=)`，与 Stage2 前向一致。"""

    def __init__(self, num_concepts: int, adj_cpu: torch.Tensor):
        self.num_concepts = num_concepts
        self._adj = adj_cpu

    def get_adjacency_matrix(self, device):
        return self._adj.to(device)


def _read_stage4_num_source(stage4_npz: str) -> int:
    payload = _load_npz(stage4_npz)
    if "num_samples" in payload:
        return int(np.asarray(payload["num_samples"]).ravel()[0])
    meta_path = os.path.splitext(stage4_npz)[0] + "_meta.json"
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "num_samples" in meta:
            return int(meta["num_samples"])
    raise ValueError(
        f"无法从 {stage4_npz} 或对应 _meta.json 读取 num_samples，"
        "请确认 Stage4 输出完整。"
    )


def build_loaders(
    X_labeled_train: np.ndarray,
    y_labeled_train: np.ndarray,
    c_labeled_train: np.ndarray,
    X_labeled_val: np.ndarray,
    y_labeled_val: np.ndarray,
    c_labeled_val: np.ndarray,
    X_unlabeled: np.ndarray,
    unlabeled_contour_masks: Optional[np.ndarray],
    unlabeled_pool_indices: Optional[np.ndarray],
    batch_size_labeled: int,
    batch_size_unlabeled: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    from core_modules import LabeledDataset, UnlabeledDataset

    train_lab = LabeledDataset(X_labeled_train, y_labeled_train, c_labeled_train)
    val_lab = LabeledDataset(X_labeled_val, y_labeled_val, c_labeled_val)
    unlabeled_ds = UnlabeledDataset(
        X_unlabeled,
        transform=_weak_augment,
        strong_transform=_strong_augment,
        contour_masks=unlabeled_contour_masks,
    )

    if unlabeled_pool_indices is not None and len(unlabeled_pool_indices) > 0:
        unlabeled_ds = Subset(unlabeled_ds, unlabeled_pool_indices.tolist())

    train_labeled_loader = DataLoader(
        train_lab,
        batch_size=batch_size_labeled,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_lab,
        batch_size=batch_size_labeled,
        shuffle=False,
        num_workers=0,
    )
    train_unlabeled_loader = DataLoader(
        unlabeled_ds,
        batch_size=batch_size_unlabeled,
        shuffle=True,
        num_workers=0,
    )
    return train_labeled_loader, train_unlabeled_loader, val_loader


def main() -> None:
    default_stage1 = find_latest_output_dir(
        os.path.join(_ROOT, 'Stage1'), 'stage1_cub_output'
    )
    default_cub = os.path.join(_ROOT, "Data", "CUB_200_2011")
    default_s2 = find_latest_output_file(
        os.path.join(_ROOT, 'Stage2_Semantic-feature-extraction'),
        'stage2_output',
        'stage2_features.npz',
    )
    default_s3 = find_latest_output_file(
        os.path.join(_ROOT, 'Stage3_Pseudo-label'),
        'stage3_output',
        'stage3_pseudo_labels.npz',
    )
    default_s4 = find_latest_output_file(
        os.path.join(_ROOT, 'Stage4_Anti-curriculum'),
        'stage4_output',
        'stage4_sampled_indices.npz',
    )

    parser = argparse.ArgumentParser(description="Stage 5 CUB：衔接 Stage2/3/4 npz 的训练")
    parser.add_argument("--cub-root", type=str, default=default_cub)
    parser.add_argument("--stage1-out", type=str, default=default_stage1)
    parser.add_argument("--adj-path", type=str, default=None)
    parser.add_argument(
        "--stage2-npz",
        type=str,
        default=default_s2,
        help="Stage2 输出，用于校验样本数与行顺序（与脚本读入的 CUB 图像一致）",
    )
    parser.add_argument("--stage3-npz", type=str, default=default_s3)
    parser.add_argument(
        "--stage4-npz",
        type=str,
        default=None,
        help="Stage4 输出；不配则不用 Stage4 池",
    )
    parser.add_argument(
        "--use-stage4-pool",
        action="store_true",
        help="启用时：用 Stage4 索引在无标签子集中构造子集（需同时提供 --stage4-npz）",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="可选：与 Stage2 相同的 checkpoint（.pt）；若路径不存在则警告并随机初始化",
    )
    parser.add_argument("--num-concepts", type=int, default=312)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size-labeled", type=int, default=32)
    parser.add_argument("--batch-size-unlabeled", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="从「有标签」子集中划验证集比例",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="训练摘要输出目录；未提供时默认写入 Stage5_Graph-logic_Optimization/stage5_output_<HH-MM_YYMMDD>/",
    )
    parser.add_argument(
        "--save-weights",
        type=str,
        default=None,
        help="训练结束后将 CompleteModel.state_dict() 保存到此路径（.pt），可供 Stage2 / 下次 Stage5 --weights 使用",
    )
    args = parser.parse_args()
    run_stage5_cub_training(args)


def _write_epoch_row_summary(output_dir: str, history: Dict[str, list[float]]) -> str:
    keys = list(history.keys())
    num_epochs = max((len(v) for v in history.values()), default=0)
    out_path = os.path.join(output_dir, "stage5_cub_training_summary_rows.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        for epoch_idx in range(num_epochs):
            parts = [f"epoch={epoch_idx + 1}"]
            for key in keys:
                values = history.get(key, [])
                if epoch_idx >= len(values):
                    continue
                val = float(values[epoch_idx])
                if val != val:
                    parts.append(f"{key}=nan")
                else:
                    parts.append(f"{key}={val:.6f}")
            f.write(" | ".join(parts) + "\n")

    return out_path


def run_stage5_cub_training(args) -> None:
    """
    执行 Stage5 CUB 训练（与 `main()` 中 argparse 解析后的逻辑一致）。
    `args` 须具备与 ArgumentParser 相同字段的 namespace。
    """
    load_cub_images, load_adjacency, resolve_adj_path, maybe_load_weights = _load_stage2_helpers()
    load_cub_train_concept_matrix = _load_cub_gt_loader()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    s3 = _load_npz(args.stage3_npz)
    if "labeled_row_indices" not in s3 or "unlabeled_row_indices" not in s3:
        raise KeyError(
            f"{args.stage3_npz} 中缺少 labeled_row_indices / unlabeled_row_indices。\n"
            "请使用 Stage3 的 CUB 划分模式生成（--cub-root + --labeled-fraction）。"
        )

    lab_idx = s3["labeled_row_indices"].astype(np.int64).ravel()
    unl_idx = s3["unlabeled_row_indices"].astype(np.int64).ravel()

    s2 = _load_npz(args.stage2_npz)
    n2 = int(s2["f_visual"].shape[0])

    images_t = load_cub_images(
        cub_root=args.cub_root,
        split="train",
        image_size=args.image_size,
        max_samples=0,
    )
    n_img = images_t.shape[0]
    if n_img != n2:
        raise ValueError(
            f"CUB train 图像数 {n_img} 与 Stage2 npz 样本数 {n2} 不一致；"
            "请使用同一划分与同一 cub-root/image-size。"
        )

    gt_mat, _gt_meta = load_cub_train_concept_matrix(args.cub_root)
    if gt_mat.shape[0] != n_img:
        raise ValueError(
            f"CUB 概念矩阵行数 {gt_mat.shape[0]} 与图像数 {n_img} 不一致。"
        )

    images = images_t.cpu().numpy().astype(np.float32)
    gt = gt_mat.astype(np.float32)

    X_lab = images[lab_idx]
    y_lab = gt[lab_idx]
    X_unl = images[unl_idx]
    contour_masks_all = load_cub_segmentations(
        args.cub_root,
        split="train",
        image_size=args.image_size,
        max_samples=0,
    )
    if contour_masks_all.shape[0] != n_img:
        raise ValueError(
            f"CUB contour mask 数 {contour_masks_all.shape[0]} 与图像数 {n_img} 不一致。"
        )
    contour_masks_unl = contour_masks_all[unl_idx]

    n_lab = X_lab.shape[0]
    if n_lab < 2:
        raise ValueError("有标签样本过少，无法划分 train/val。")

    n_val = max(1, int(round(n_lab * args.val_split)))
    n_val = min(n_val, n_lab - 1)
    perm = np.random.RandomState(args.seed).permutation(n_lab)
    val_ids = perm[:n_val]
    train_ids = perm[n_val:]

    X_ltr, y_ltr, c_ltr = X_lab[train_ids], y_lab[train_ids], y_lab[train_ids]
    X_lv, y_lv, c_lv = X_lab[val_ids], y_lab[val_ids], y_lab[val_ids]

    unlabeled_pool: Optional[np.ndarray] = None
    if args.use_stage4_pool:
        if not args.stage4_npz:
            raise ValueError("--use-stage4-pool 需要同时提供 --stage4-npz")
        s4 = _load_npz(args.stage4_npz)
        if "sampled_indices" not in s4:
            raise KeyError(f"{args.stage4_npz} 中缺少 sampled_indices")
        num_source = _read_stage4_num_source(args.stage4_npz)
        unlabeled_pool = _map_stage4_to_unlabeled_local(
            s4["sampled_indices"], unl_idx, num_source_rows=num_source
        )
        if unlabeled_pool.size == 0:
            print(
                "警告: Stage4 索引映射到无标签子集后为空，回退为使用全部无标签样本。"
            )
            unlabeled_pool = None
        else:
            print(
                f"Stage4 池: 无标签局部索引数={len(unlabeled_pool)} "
                f"(Stage4 num_source_rows={num_source})"
            )

    train_labeled_loader, train_unlabeled_loader, val_loader = build_loaders(
        X_ltr,
        y_ltr,
        c_ltr,
        X_lv,
        y_lv,
        c_lv,
        X_unl,
        contour_masks_unl,
        unlabeled_pool,
        args.batch_size_labeled,
        args.batch_size_unlabeled,
    )

    device = args.device
    model = CompleteModel(
        input_dim=3,
        feature_dim=args.feature_dim,
        num_concepts=args.num_concepts,
        num_heads=args.num_heads,
    )
    wpath = args.weights
    if wpath and not os.path.isfile(wpath):
        print(
            f"警告: 未找到权重文件「{wpath}」，将使用随机初始化。"
            "请传真实 checkpoint 路径，或不要写 --weights（与 Stage2 一致时建议加载）",
            flush=True,
        )
        wpath = None
    maybe_load_weights(model, wpath)

    framework = ConceptualSSLFramework(
        model=model,
        device=device,
        num_concepts=args.num_concepts,
        num_epochs=args.num_epochs,
        batch_size_labeled=args.batch_size_labeled,
        batch_size_unlabeled=args.batch_size_unlabeled,
        learning_rate=args.learning_rate,
    )

    adj_path = resolve_adj_path(args.adj_path, args.stage1_out)
    print(f"邻接矩阵（与 Stage2 一致）: {adj_path}", flush=True)
    adj_matrix = load_adjacency(adj_path, torch.device("cpu"))
    if adj_matrix.shape[0] != args.num_concepts:
        raise ValueError(
            f"num_concepts={args.num_concepts} 与邻接矩阵 {tuple(adj_matrix.shape)} 不一致"
        )
    prebuilt = _HypergraphAdjWrapper(args.num_concepts, adj_matrix.cpu())

    output_dir = resolve_default_output_dir(
        os.path.dirname(os.path.abspath(__file__)),
        'stage5_output',
        args.output_dir,
    )

    history = framework.fit(
        train_labeled_loader=train_labeled_loader,
        train_unlabeled_loader=train_unlabeled_loader,
        val_loader=val_loader,
        prebuilt_hypergraph=prebuilt,
        contour_mask=None,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_json = os.path.join(output_dir, "stage5_cub_training_summary.json")
        serializable = {k: [float(x) for x in v] for k, v in history.items()}
        payload = {
            "meta": {
                "cub_root": os.path.abspath(args.cub_root),
                "stage2_npz": os.path.abspath(args.stage2_npz),
                "stage3_npz": os.path.abspath(args.stage3_npz),
                "stage4_npz": os.path.abspath(args.stage4_npz) if args.stage4_npz else None,
                "use_stage4_pool": bool(args.use_stage4_pool),
                "num_labeled_train": int(X_ltr.shape[0]),
                "num_val": int(X_lv.shape[0]),
                "num_unlabeled": int(X_unl.shape[0]),
                "unlabeled_pool_size": int(len(train_unlabeled_loader.dataset)),
            },
            "history": serializable,
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"已写入: {out_json}")

        out_rows = _write_epoch_row_summary(output_dir, serializable)
        print(f"已写入: {out_rows}")

    if args.save_weights:
        save_path = os.path.abspath(args.save_weights)
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(framework.model.state_dict(), save_path)
        print(f"已保存模型权重（state_dict）: {save_path}", flush=True)


if __name__ == "__main__":
    main()
