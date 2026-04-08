#!/usr/bin/env python3
"""
将 spatial_concept_heatmap（低分辨率，如 14×14）上采样后叠加到原图，便于检查概念—空间对齐。

在 SCAR-CBM 根目录运行。

方式 A — 仅 .npz（需已含 spatial_concept_heatmap，例如重跑 Stage2 导出）:
  python visualize_spatial_concept_heatmap.py \\
      --npz Stage2_Semantic-feature-extraction/stage2_output/stage2_features.npz \\
      --npz-index 0 \\
      --image path/to/image.jpg \\
      --out-dir viz_out

方式 B — CUB 训练集第 i 张（与 run_stage2.load_cub_images 顺序一致）+ npz 第 i 行:
  python visualize_spatial_concept_heatmap.py \\
      --npz .../stage2_features.npz --npz-index 42 \\
      --cub-root Data/CUB_200_2011 --cub-split train \\
      --out-dir viz_out

方式 C — 仅权重现场推理（不传 --npz）:
  python visualize_spatial_concept_heatmap.py \\
      --weights model.pt --stage1-out Stage1/stage1_cub_output \\
      --num-concepts 312 --image path/to/image.jpg \\
      --out-dir viz_out

方式 D — 递归处理整个 images 文件夹（逐张与模型对齐，输出目录镜像相对路径）:
  python visualize_spatial_concept_heatmap.py \\
      --images-dir Data/CUB_200_2011/images \\
      --weights model.pt --stage1-out Stage1/stage1_cub_output \\
      --num-concepts 312 --out-dir viz_cub_all \\
      --max-images 0
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from typing import List, Optional, Sequence

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **kwargs):
        return it

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from network_architecture import CompleteModel


def _load_run_stage2():
    path = os.path.join(_ROOT, "Stage2_Semantic-feature-extraction", "run_stage2.py")
    spec = importlib.util.spec_from_file_location("run_stage2_viz", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_image_tensor(path: str, size: int) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB")
        if size > 0:
            img = img.resize((size, size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)


def load_cub_one(
    cub_root: str,
    split: str,
    sample_index: int,
    image_size: int,
) -> torch.Tensor:
    m = _load_run_stage2()
    n = sample_index + 1
    images = m.load_cub_images(
        cub_root=cub_root,
        split=split,
        image_size=image_size,
        max_samples=n,
    )
    if images.shape[0] <= sample_index:
        raise IndexError(
            f"cub 样本不足: 需要索引 {sample_index}，实际仅 {images.shape[0]} 张"
        )
    return images[sample_index : sample_index + 1]


def load_adj(device: torch.device, stage1_out: str | None, adj_path: str | None) -> torch.Tensor:
    m = _load_run_stage2()
    p = m.resolve_adj_path(adj_path, stage1_out or os.path.join(_ROOT, "Stage1", "stage1_cub_output"))
    return m.load_adjacency(p, device)


def maybe_load_weights(model: torch.nn.Module, weights_path: str | None) -> None:
    if not weights_path:
        return
    m = _load_run_stage2()
    m.maybe_load_weights(model, weights_path)


def upsample_heatmaps(
    heatmaps: torch.Tensor,
    out_h: int,
    out_w: int,
) -> torch.Tensor:
    """heatmaps: [K, h, w] -> [K, out_h, out_w]"""
    if heatmaps.ndim != 3:
        raise ValueError(f"期望 [K,h,w]，得到 {tuple(heatmaps.shape)}")
    x = heatmaps.unsqueeze(0).float()
    x = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=True)
    return x.squeeze(0)


def overlay_rgb(
    image_hwc: np.ndarray,
    heatmap_hw: np.ndarray,
    cmap_name: str = "turbo",
    alpha: float = 0.45,
) -> np.ndarray:
    """
    image_hwc: [H,W,3] float 0~1
    heatmap_hw: [H,W] float，已归一化到约 0~1
    """
    if hasattr(mpl, "colormaps"):
        cmap = mpl.colormaps[cmap_name]
    else:
        cmap = plt.cm.get_cmap(cmap_name)
    h = np.clip(heatmap_hw, 0.0, 1.0)
    color = cmap(h)[..., :3].astype(np.float32)
    base = np.clip(image_hwc.astype(np.float32), 0.0, 1.0)
    out = (1.0 - alpha) * base + alpha * color
    return np.clip(out, 0.0, 1.0)


def parse_concept_ids(s: str | None, num_concepts: int) -> List[int]:
    if not s:
        return []
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        k = int(part)
        if k < 0 or k >= num_concepts:
            raise ValueError(f"概念 id 越界: {k}（num_concepts={num_concepts}）")
        out.append(k)
    return out


def pick_top_k(heatmaps_lowres: torch.Tensor, k: int) -> List[int]:
    """heatmaps_lowres: [K, h, w]"""
    score = heatmaps_lowres.amax(dim=(1, 2))
    k = min(k, score.shape[0])
    _, idx = torch.topk(score, k)
    return idx.cpu().tolist()


def pick_top_k_compare(
    heatmaps_a: torch.Tensor,
    heatmaps_b: torch.Tensor,
    k: int,
) -> List[int]:
    """从两组热力图的联合峰值中选概念，便于 before/after 对齐比较。"""
    score_a = heatmaps_a.amax(dim=(1, 2))
    score_b = heatmaps_b.amax(dim=(1, 2))
    score = torch.maximum(score_a, score_b)
    k = min(k, score.shape[0])
    _, idx = torch.topk(score, k)
    return idx.cpu().tolist()


def collect_image_paths(root: str, extensions: Sequence[str] = (".jpg", ".jpeg", ".png", ".webp", ".bmp")) -> List[str]:
    root_abs = os.path.abspath(root)
    if not os.path.isdir(root_abs):
        raise NotADirectoryError(root)
    found: List[str] = []
    ext_set = tuple(e.lower() for e in extensions)
    for dirpath, _, filenames in os.walk(root_abs):
        for fn in filenames:
            low = fn.lower()
            if any(low.endswith(e) for e in ext_set):
                found.append(os.path.join(dirpath, fn))
    found.sort()
    return found


def overlay_out_path(images_dir: str, image_path: str, out_dir: str, suffix: str = "_spatial_overlay.png") -> str:
    """在 out_dir 下保持与 images_dir 相同的相对子目录。"""
    rel = os.path.relpath(image_path, start=os.path.abspath(images_dir))
    base, _ = os.path.splitext(rel)
    return os.path.join(os.path.abspath(out_dir), base + suffix)


def save_panel(
    image_chw: torch.Tensor,
    heatmaps_k_hw: torch.Tensor,
    concept_ids: List[int],
    out_path: str,
    cmap_name: str,
    alpha: float,
    ncols: int = 4,
) -> None:
    """heatmaps_k_hw: [K', H, W] 已与图像同分辨率"""
    img = image_chw.detach().cpu().numpy().transpose(1, 2, 0)
    hmaps = heatmaps_k_hw.detach().cpu().numpy()
    n = len(concept_ids)
    ncols = max(1, min(ncols, n))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows))
    if nrows * ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    for i, cid in enumerate(concept_ids):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        hm = hmaps[i]
        p_lo, p_hi = np.percentile(hm, [2, 98])
        hm = np.clip((hm - p_lo) / (p_hi - p_lo + 1e-8), 0.0, 1.0)
        blended = overlay_rgb(img, hm, cmap_name=cmap_name, alpha=alpha)
        ax.imshow(blended)
        ax.set_title(f"concept {cid}")
        ax.axis("off")
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _forward_spatial_heatmaps(
    model: torch.nn.Module,
    x: torch.Tensor,
    adj: torch.Tensor,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        out = model(x, adj)
    if "spatial_concept_heatmap" not in out or out["spatial_concept_heatmap"] is None:
        raise RuntimeError("模型输出中无 spatial_concept_heatmap")
    return out["spatial_concept_heatmap"][0]


def run_single_overlay(
    x: torch.Tensor,
    heatmaps_low: torch.Tensor,
    concept_ids: List[int],
    out_png: str,
    cmap_name: str,
    alpha: float,
    ncols: int,
) -> None:
    _, _, H, W = x.shape
    heat_hi = upsample_heatmaps(heatmaps_low, H, W)
    heat_sel = heat_hi[concept_ids]
    save_panel(
        x[0].cpu(),
        heat_sel.cpu(),
        concept_ids,
        out_png,
        cmap_name=cmap_name,
        alpha=alpha,
        ncols=ncols,
    )


def save_compare_panel(
    image_chw: torch.Tensor,
    heatmaps_a_k_hw: torch.Tensor,
    heatmaps_b_k_hw: torch.Tensor,
    concept_ids: List[int],
    out_path: str,
    cmap_name: str,
    alpha: float,
    label_a: str,
    label_b: str,
) -> None:
    if heatmaps_a_k_hw.shape != heatmaps_b_k_hw.shape:
        raise ValueError(
            f"before/after 热力图形状不一致: {tuple(heatmaps_a_k_hw.shape)} vs {tuple(heatmaps_b_k_hw.shape)}"
        )
    img = image_chw.detach().cpu().numpy().transpose(1, 2, 0)
    hmaps_a = heatmaps_a_k_hw.detach().cpu().numpy()
    hmaps_b = heatmaps_b_k_hw.detach().cpu().numpy()
    n = len(concept_ids)
    fig, axes = plt.subplots(n, 2, figsize=(6.4, 3.0 * n))
    if n == 1:
        axes = np.array([axes])
    for i, cid in enumerate(concept_ids):
        for j, (hmaps, label) in enumerate(((hmaps_a, label_a), (hmaps_b, label_b))):
            ax = axes[i, j]
            hm = hmaps[i]
            p_lo, p_hi = np.percentile(hm, [2, 98])
            hm = np.clip((hm - p_lo) / (p_hi - p_lo + 1e-8), 0.0, 1.0)
            blended = overlay_rgb(img, hm, cmap_name=cmap_name, alpha=alpha)
            ax.imshow(blended)
            ax.set_title(f"concept {cid} | {label}")
            ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_compare_overlay(
    x: torch.Tensor,
    heatmaps_low_a: torch.Tensor,
    heatmaps_low_b: torch.Tensor,
    concept_ids: List[int],
    out_png: str,
    cmap_name: str,
    alpha: float,
    label_a: str,
    label_b: str,
) -> None:
    _, _, H, W = x.shape
    heat_hi_a = upsample_heatmaps(heatmaps_low_a, H, W)[concept_ids]
    heat_hi_b = upsample_heatmaps(heatmaps_low_b, H, W)[concept_ids]
    save_compare_panel(
        x[0].cpu(),
        heat_hi_a.cpu(),
        heat_hi_b.cpu(),
        concept_ids,
        out_png,
        cmap_name=cmap_name,
        alpha=alpha,
        label_a=label_a,
        label_b=label_b,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="spatial_concept_heatmap 叠加原图可视化",
    )
    parser.add_argument("--npz", type=str, default=None, help="Stage2 等导出的 .npz")
    parser.add_argument(
        "--npz-index",
        type=int,
        default=0,
        help="使用 npz 中第几条样本的 spatial_concept_heatmap",
    )
    parser.add_argument("--image", type=str, default=None, help="单张图像路径")
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="递归处理文件夹内所有图像（需 --weights；不支持与 --npz 联用）",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="批量模式下最多处理张数，0 表示不限制",
    )
    parser.add_argument("--cub-root", type=str, default=None)
    parser.add_argument("--cub-split", type=str, default="train", choices=("train", "test"))
    parser.add_argument("--sample-index", type=int, default=None, help="与 --cub-root 联用")
    parser.add_argument("--weights", type=str, default=None, help="CompleteModel 权重（无 npz 时必需）")
    parser.add_argument("--compare-weights", type=str, default=None, help="第二份 CompleteModel 权重，用于 before/after 对比导出")
    parser.add_argument("--stage1-out", type=str, default=None)
    parser.add_argument("--adj-path", type=str, default=None)
    parser.add_argument("--num-concepts", type=int, default=312)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--concepts",
        type=str,
        default=None,
        help="要可视化的概念 id，逗号分隔，如 0,3,7（与 --top-k 二选一）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="按低分辨率热力图峰值选 top-k 个概念（未指定 --concepts 时生效）",
    )
    parser.add_argument("--cmap", type=str, default="turbo")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--ncols", type=int, default=4)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(_ROOT, "viz_spatial_concept_output"),
    )
    parser.add_argument(
        "--basename",
        type=str,
        default="spatial_overlay",
        help="输出 PNG 文件名前缀",
    )
    parser.add_argument("--compare-label-a", type=str, default="before")
    parser.add_argument("--compare-label-b", type=str, default="after")
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.compare_weights and args.npz:
        parser.error("--compare-weights 目前仅支持现场推理模式，不支持与 --npz 联用")

    if args.images_dir:
        if args.npz:
            parser.error("批量 --images-dir 与 --npz 不能同时使用（逐文件无 npz 行号对应）")
        if not args.weights:
            parser.error("批量模式必须提供 --weights")
        paths = collect_image_paths(args.images_dir)
        if args.max_images > 0:
            paths = paths[: args.max_images]
        if not paths:
            raise FileNotFoundError(f"目录下未找到图像: {args.images_dir}")

        model = CompleteModel(
            input_dim=3,
            feature_dim=args.feature_dim,
            num_concepts=args.num_concepts,
            num_heads=args.num_heads,
        ).to(device)
        maybe_load_weights(model, args.weights)
        compare_model: Optional[torch.nn.Module] = None
        if args.compare_weights:
            compare_model = CompleteModel(
                input_dim=3,
                feature_dim=args.feature_dim,
                num_concepts=args.num_concepts,
                num_heads=args.num_heads,
            ).to(device)
            maybe_load_weights(compare_model, args.compare_weights)
        adj = load_adj(device, args.stage1_out, args.adj_path)
        if adj.shape[0] != args.num_concepts:
            raise ValueError(
                f"邻接矩阵 {tuple(adj.shape)} 与 num_concepts={args.num_concepts} 不一致"
            )

        os.makedirs(args.out_dir, exist_ok=True)
        ids0 = parse_concept_ids(args.concepts, args.num_concepts)
        for path in tqdm(paths, desc="spatial_overlay"):
            x = load_image_tensor(path, args.image_size).to(device)
            heatmaps_low = _forward_spatial_heatmaps(model, x, adj)
            compare_heatmaps_low = (
                _forward_spatial_heatmaps(compare_model, x, adj)
                if compare_model is not None
                else None
            )
            num_c = heatmaps_low.shape[0]
            if ids0:
                ids = [i for i in ids0 if i < num_c]
                if not ids:
                    ids = (
                        pick_top_k_compare(heatmaps_low.cpu(), compare_heatmaps_low.cpu(), args.top_k)
                        if compare_heatmaps_low is not None
                        else pick_top_k(heatmaps_low.cpu(), args.top_k)
                    )
            else:
                ids = (
                    pick_top_k_compare(heatmaps_low.cpu(), compare_heatmaps_low.cpu(), args.top_k)
                    if compare_heatmaps_low is not None
                    else pick_top_k(heatmaps_low.cpu(), args.top_k)
                )
            suffix = "_spatial_compare.png" if compare_heatmaps_low is not None else "_spatial_overlay.png"
            out_png = overlay_out_path(args.images_dir, path, args.out_dir, suffix=suffix)
            os.makedirs(os.path.dirname(out_png), exist_ok=True)
            if compare_heatmaps_low is not None:
                run_compare_overlay(
                    x,
                    heatmaps_low,
                    compare_heatmaps_low,
                    ids,
                    out_png,
                    cmap_name=args.cmap,
                    alpha=args.alpha,
                    label_a=args.compare_label_a,
                    label_b=args.compare_label_b,
                )
            else:
                run_single_overlay(
                    x,
                    heatmaps_low,
                    ids,
                    out_png,
                    cmap_name=args.cmap,
                    alpha=args.alpha,
                    ncols=args.ncols,
                )
        print(f"完成 {len(paths)} 张，输出根目录: {os.path.abspath(args.out_dir)}")
        return

    # 图像 [1,3,H,W]
    x: torch.Tensor
    if args.image:
        x = load_image_tensor(args.image, args.image_size).to(device)
        src_tag = args.image
    elif args.cub_root is not None and args.sample_index is not None:
        x = load_cub_one(
            args.cub_root, args.cub_split, args.sample_index, args.image_size
        ).to(device)
        src_tag = f"cub:{args.cub_root}[{args.sample_index}]"
    else:
        parser.error("请提供 --images-dir、--image，或同时提供 --cub-root 与 --sample-index")

    _, _, H, W = x.shape

    heatmaps_low: torch.Tensor
    compare_heatmaps_low: Optional[torch.Tensor] = None
    if args.npz:
        if not os.path.isfile(args.npz):
            raise FileNotFoundError(args.npz)
        with np.load(args.npz, allow_pickle=False) as z:
            if "spatial_concept_heatmap" not in z:
                raise KeyError(
                    f"{args.npz} 中无 spatial_concept_heatmap，请用当前模型重跑 Stage2 导出"
                )
            arr = z["spatial_concept_heatmap"]
        if arr.ndim == 4:
            if args.npz_index < 0 or args.npz_index >= arr.shape[0]:
                raise IndexError(
                    f"npz-index={args.npz_index} 越界，N={arr.shape[0]}"
                )
            arr = arr[args.npz_index]
        elif arr.ndim == 3:
            if args.npz_index != 0:
                raise ValueError("npz 为 [K,h,w] 单样本时不要使用非零 --npz-index")
        else:
            raise ValueError(
                f"spatial_concept_heatmap 形状应为 [N,K,h,w] 或 [K,h,w]，当前 {arr.shape}"
            )
        heatmaps_low = torch.from_numpy(arr.astype(np.float32)).to(device)
        if heatmaps_low.shape[0] != args.num_concepts:
            print(
                f"警告: num_concepts={args.num_concepts} 与 npz 概念维 {heatmaps_low.shape[0]} 不一致，"
                f"后续以 npz 为准选 id。"
            )
    else:
        if not args.weights:
            parser.error("未指定 --npz 时，必须提供 --weights 做前向推理")
        model = CompleteModel(
            input_dim=3,
            feature_dim=args.feature_dim,
            num_concepts=args.num_concepts,
            num_heads=args.num_heads,
        ).to(device)
        maybe_load_weights(model, args.weights)
        compare_model: Optional[torch.nn.Module] = None
        if args.compare_weights:
            compare_model = CompleteModel(
                input_dim=3,
                feature_dim=args.feature_dim,
                num_concepts=args.num_concepts,
                num_heads=args.num_heads,
            ).to(device)
            maybe_load_weights(compare_model, args.compare_weights)
        adj = load_adj(device, args.stage1_out, args.adj_path)
        if adj.shape[0] != args.num_concepts:
            raise ValueError(
                f"邻接矩阵 {tuple(adj.shape)} 与 num_concepts={args.num_concepts} 不一致"
            )
        heatmaps_low = _forward_spatial_heatmaps(model, x, adj)
        compare_heatmaps_low = (
            _forward_spatial_heatmaps(compare_model, x, adj)
            if compare_model is not None
            else None
        )

    num_c = heatmaps_low.shape[0]
    ids = parse_concept_ids(args.concepts, num_c)
    if not ids:
        ids = (
            pick_top_k_compare(heatmaps_low.cpu(), compare_heatmaps_low.cpu(), args.top_k)
            if compare_heatmaps_low is not None
            else pick_top_k(heatmaps_low.cpu(), args.top_k)
        )

    os.makedirs(args.out_dir, exist_ok=True)
    safe = str(args.npz_index) if args.npz else "live"
    suffix = "compare" if compare_heatmaps_low is not None else "overlay"
    out_png = os.path.join(
        args.out_dir, f"{args.basename}_{safe}_{suffix}.png"
    )
    if compare_heatmaps_low is not None:
        run_compare_overlay(
            x,
            heatmaps_low,
            compare_heatmaps_low,
            ids,
            out_png,
            cmap_name=args.cmap,
            alpha=args.alpha,
            label_a=args.compare_label_a,
            label_b=args.compare_label_b,
        )
    else:
        run_single_overlay(
            x,
            heatmaps_low,
            ids,
            out_png,
            cmap_name=args.cmap,
            alpha=args.alpha,
            ncols=args.ncols,
        )
    print(f"已保存: {out_png}")
    print(f"  来源: {src_tag}")
    print(f"  概念: {ids}")
    print(f"  低分辨率热力图: {tuple(heatmaps_low.shape)} -> 叠加 {H}x{W}")
    if compare_heatmaps_low is not None:
        print(f"  对比权重: {args.compare_weights}")


if __name__ == "__main__":
    main()
