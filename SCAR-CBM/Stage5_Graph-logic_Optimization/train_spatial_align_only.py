#!/usr/bin/env python3
"""
仅训练 CompleteModel.concept_spatial_align，冻结主任务分支，尽量不影响 c_acc / y_acc。

核心原则：
- encoder / predictor_global / cross_attention / predictor_local / heatmap_generator / gcn 全部冻结
- 只更新 concept_spatial_align
- 主任务指标 c_acc / y_acc 低于门槛时，不保存 checkpoint；若启用严格门槛则直接报错退出
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STAGE5 = os.path.dirname(os.path.abspath(__file__))
for p in (_ROOT, _STAGE5):
    if p not in sys.path:
        sys.path.insert(0, p)

from core_modules import LabeledDataset, UnlabeledDataset
from loss_functions import (
    GeometricConstraintLoss,
    SpatialConceptAlignmentLoss,
    SpatialConsistencyLoss,
    SpatialPseudoAlignmentLoss,
)
from network_architecture import CompleteModel
from run_stage5_cub import (
    _load_cub_gt_loader,
    _load_npz,
    _load_stage2_helpers,
    _map_stage4_to_unlabeled_local,
    _read_stage4_num_source,
    _strong_augment,
    _weak_augment,
    load_cub_segmentations,
)
from stage5_framework import _fit_per_concept_thresholds, _multilabel_concept_and_class_metrics
from stage_output_utils import find_latest_output_dir, find_latest_output_file


def build_loaders(
    X_labeled_train: np.ndarray,
    y_labeled_train: np.ndarray,
    c_labeled_train: np.ndarray,
    mask_labeled_train: np.ndarray,
    X_labeled_val: np.ndarray,
    y_labeled_val: np.ndarray,
    c_labeled_val: np.ndarray,
    mask_labeled_val: np.ndarray,
    X_unlabeled: np.ndarray,
    mask_unlabeled: np.ndarray,
    unlabeled_pool_indices: Optional[np.ndarray],
    batch_size_labeled: int,
    batch_size_unlabeled: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_lab = LabeledDataset(
        X_labeled_train,
        y_labeled_train,
        c_labeled_train,
        transform=_weak_augment,
        contour_masks=mask_labeled_train,
    )
    val_lab = LabeledDataset(
        X_labeled_val,
        y_labeled_val,
        c_labeled_val,
        transform=None,
        contour_masks=mask_labeled_val,
    )
    unlabeled_ds = UnlabeledDataset(
        X_unlabeled,
        transform=_weak_augment,
        strong_transform=_strong_augment,
        contour_masks=mask_unlabeled,
    )
    if unlabeled_pool_indices is not None and len(unlabeled_pool_indices) > 0:
        unlabeled_ds = Subset(unlabeled_ds, unlabeled_pool_indices.tolist())

    train_labeled_loader = DataLoader(train_lab, batch_size=batch_size_labeled, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_lab, batch_size=batch_size_labeled, shuffle=False, num_workers=0)
    train_unlabeled_loader = DataLoader(unlabeled_ds, batch_size=batch_size_unlabeled, shuffle=True, num_workers=0)
    return train_labeled_loader, train_unlabeled_loader, val_loader


def freeze_except_spatial_align(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith('concept_spatial_align.')


@torch.no_grad()
def forward_frozen_main(
    model: CompleteModel,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    f_visual, spatial_features = model.encoder(x)
    c_global = model.predictor_global(f_visual)
    h_aligned = model.cross_attention(c_global, spatial_features)
    c_local = model.predictor_local(h_aligned)
    c_heatmap = model.heatmap_generator(c_global, c_local)
    return f_visual, spatial_features, c_global, c_local, c_heatmap


def forward_spatial_only(
    model: CompleteModel,
    x: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    f_visual, spatial_features, c_global, c_local, c_heatmap = forward_frozen_main(model, x)
    spatial_heatmap = model.concept_spatial_align(
        spatial_features.detach(),
        c_global.detach(),
        c_local=c_local.detach(),
        c_fused=c_heatmap.detach(),
    )
    return {
        'f_visual': f_visual.detach(),
        'c_heatmap': c_heatmap.detach(),
        'spatial_concept_heatmap': spatial_heatmap,
    }


def eval_main_metrics(
    model: CompleteModel,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    ys_list = []
    logits_list = []
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            out = forward_spatial_only(model, x)
            logits = out['c_heatmap']
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
            total_loss += loss.item() * x.shape[0]
            total_n += x.shape[0]
            ys_list.append(y.cpu().numpy())
            logits_list.append(logits.cpu().numpy())
    y_all = np.concatenate(ys_list, axis=0)
    logits_all = np.concatenate(logits_list, axis=0)
    m = _multilabel_concept_and_class_metrics(y_all, logits_all)
    m['val_loss'] = total_loss / max(total_n, 1)
    return m


def eval_spatial_val_loss(
    model: CompleteModel,
    val_loader: DataLoader,
    device: torch.device,
    spatial_align_loss: SpatialConceptAlignmentLoss,
    geo_loss_fn: GeometricConstraintLoss,
    lambda_geo: float,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            mask = batch[4].to(device) if len(batch) > 4 else None
            out = forward_spatial_only(model, x)
            loss = spatial_align_loss(out['spatial_concept_heatmap'], y)
            if mask is not None and lambda_geo > 0.0:
                loss = loss + lambda_geo * geo_loss_fn(out['spatial_concept_heatmap'], mask)
            total += loss.item() * x.shape[0]
            n += x.shape[0]
    return total / max(n, 1)


def main() -> None:
    default_stage1 = find_latest_output_dir(os.path.join(_ROOT, 'Stage1'), 'stage1_cub_output')
    default_cub = os.path.join(_ROOT, 'Data', 'CUB_200_2011')
    default_s2 = find_latest_output_file(os.path.join(_ROOT, 'Stage2_Semantic-feature-extraction'), 'stage2_output', 'stage2_features.npz')
    default_s3 = find_latest_output_file(os.path.join(_ROOT, 'Stage3_Pseudo-label'), 'stage3_output', 'stage3_pseudo_labels.npz')
    default_s4 = find_latest_output_file(os.path.join(_ROOT, 'Stage4_Anti-curriculum'), 'stage4_output', 'stage4_sampled_indices.npz')

    parser = argparse.ArgumentParser(description='仅训练 concept_spatial_align 的 CUB 脚本')
    parser.add_argument('--cub-root', type=str, default=default_cub)
    parser.add_argument('--stage1-out', type=str, default=default_stage1)
    parser.add_argument('--adj-path', type=str, default=None)
    parser.add_argument('--stage2-npz', type=str, default=default_s2)
    parser.add_argument('--stage3-npz', type=str, default=default_s3)
    parser.add_argument('--stage4-npz', type=str, default=None)
    parser.add_argument('--use-stage4-pool', action='store_true')
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--num-concepts', type=int, default=312)
    parser.add_argument('--feature-dim', type=int, default=256)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--batch-size-labeled', type=int, default=32)
    parser.add_argument('--batch-size-unlabeled', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--lambda-labeled', type=float, default=1.0)
    parser.add_argument('--lambda-unlabeled', type=float, default=0.5)
    parser.add_argument('--lambda-geo', type=float, default=0.3)
    parser.add_argument('--lambda-consistency', type=float, default=0.1)
    parser.add_argument('--min-c-acc', type=float, default=0.90)
    parser.add_argument('--min-y-acc', type=float, default=0.90)
    parser.add_argument('--enforce-metric-floor', action='store_true')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--save-weights', type=str, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    load_cub_images, load_adjacency, resolve_adj_path, maybe_load_weights = _load_stage2_helpers()
    load_cub_train_concept_matrix = _load_cub_gt_loader()

    device = torch.device(args.device)
    s3 = _load_npz(args.stage3_npz)
    lab_idx = s3['labeled_row_indices'].astype(np.int64).ravel()
    unl_idx = s3['unlabeled_row_indices'].astype(np.int64).ravel()

    s2 = _load_npz(args.stage2_npz)
    n2 = int(s2['f_visual'].shape[0])
    images_t = load_cub_images(cub_root=args.cub_root, split='train', image_size=args.image_size, max_samples=0)
    n_img = images_t.shape[0]
    if n_img != n2:
        raise ValueError(f'CUB train 图像数 {n_img} 与 Stage2 npz 样本数 {n2} 不一致')

    gt_mat, _ = load_cub_train_concept_matrix(args.cub_root)
    if gt_mat.shape[0] != n_img:
        raise ValueError(f'CUB 概念矩阵行数 {gt_mat.shape[0]} 与图像数 {n_img} 不一致')

    images = images_t.cpu().numpy().astype(np.float32)
    gt = gt_mat.astype(np.float32)
    masks_all = load_cub_segmentations(args.cub_root, split='train', image_size=args.image_size, max_samples=0)

    X_lab = images[lab_idx]
    y_lab = gt[lab_idx]
    X_unl = images[unl_idx]
    masks_lab = masks_all[lab_idx]
    masks_unl = masks_all[unl_idx]

    n_lab = X_lab.shape[0]
    n_val = max(1, int(round(n_lab * args.val_split)))
    n_val = min(n_val, n_lab - 1)
    perm = np.random.RandomState(args.seed).permutation(n_lab)
    val_ids = perm[:n_val]
    train_ids = perm[n_val:]

    X_ltr, y_ltr, m_ltr = X_lab[train_ids], y_lab[train_ids], masks_lab[train_ids]
    X_lv, y_lv, m_lv = X_lab[val_ids], y_lab[val_ids], masks_lab[val_ids]

    unlabeled_pool = None
    if args.use_stage4_pool:
        if not args.stage4_npz:
            raise ValueError('--use-stage4-pool 需要同时提供 --stage4-npz')
        s4 = _load_npz(args.stage4_npz)
        num_source = _read_stage4_num_source(args.stage4_npz)
        unlabeled_pool = _map_stage4_to_unlabeled_local(s4['sampled_indices'], unl_idx, num_source_rows=num_source)
        if unlabeled_pool.size == 0:
            unlabeled_pool = None

    train_labeled_loader, train_unlabeled_loader, val_loader = build_loaders(
        X_ltr, y_ltr, y_ltr, m_ltr,
        X_lv, y_lv, y_lv, m_lv,
        X_unl, masks_unl,
        unlabeled_pool,
        args.batch_size_labeled,
        args.batch_size_unlabeled,
    )

    model = CompleteModel(
        input_dim=3,
        feature_dim=args.feature_dim,
        num_concepts=args.num_concepts,
        num_heads=args.num_heads,
    ).to(device)
    maybe_load_weights(model, args.weights)
    freeze_except_spatial_align(model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError('没有可训练参数，concept_spatial_align 未找到')
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)

    adj_path = resolve_adj_path(args.adj_path, args.stage1_out)
    adj_matrix = load_adjacency(adj_path, device)

    spatial_align_loss = SpatialConceptAlignmentLoss()
    spatial_pseudo_loss = SpatialPseudoAlignmentLoss()
    spatial_consistency_loss = SpatialConsistencyLoss()
    geo_loss_fn = GeometricConstraintLoss()

    baseline_metrics = eval_main_metrics(model, val_loader, device)
    print('baseline_main_metrics=', baseline_metrics)
    if args.enforce_metric_floor:
        if baseline_metrics['c_acc'] < args.min_c_acc or baseline_metrics['y_acc'] < args.min_y_acc:
            raise RuntimeError(
                f"基线主任务指标未达门槛: c_acc={baseline_metrics['c_acc']:.4f}, y_acc={baseline_metrics['y_acc']:.4f}, "
                f"要求 >= ({args.min_c_acc:.4f}, {args.min_y_acc:.4f})"
            )

    if args.output_dir is None:
        args.output_dir = os.path.join(_STAGE5, 'stage5_align_only_output')
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_weights is None:
        args.save_weights = os.path.join(args.output_dir, 'stage5_align_only_best.pt')

    config_path = os.path.join(args.output_dir, 'run_config.json')
    last_weights_path = os.path.join(args.output_dir, 'stage5_align_only_last.pt')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    history = defaultdict(list)
    best_val_spatial = float('inf')
    best_epoch = 0

    for epoch in range(1, args.num_epochs + 1):
        model.eval()
        model.concept_spatial_align.train()
        epoch_losses = defaultdict(float)
        num_batches = 0

        for batch in train_labeled_loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            mask = batch[4].to(device) if len(batch) > 4 else None

            optimizer.zero_grad()
            out = forward_spatial_only(model, x)
            loss = args.lambda_labeled * spatial_align_loss(out['spatial_concept_heatmap'], y)
            if mask is not None and args.lambda_geo > 0.0:
                loss = loss + args.lambda_geo * geo_loss_fn(out['spatial_concept_heatmap'], mask)
            loss.backward()
            optimizer.step()

            epoch_losses['L_spatial_labeled'] += loss.item()
            num_batches += 1

        for batch in train_unlabeled_loader:
            x_w = batch[0].to(device)
            x_s = batch[1].to(device)
            mask = batch[3].to(device) if len(batch) > 3 else None

            optimizer.zero_grad()
            out_w = forward_spatial_only(model, x_w)
            out_s = forward_spatial_only(model, x_s)
            teacher_prob = torch.sigmoid(out_w['c_heatmap']).detach()
            loss = args.lambda_unlabeled * spatial_pseudo_loss(
                out_w['spatial_concept_heatmap'],
                teacher_prob,
                torch.ones(x_w.shape[0], device=device),
                None,
            )
            if args.lambda_consistency > 0.0:
                loss = loss + args.lambda_consistency * spatial_consistency_loss(
                    out_w['spatial_concept_heatmap'],
                    out_s['spatial_concept_heatmap'],
                    None,
                )
            if mask is not None and args.lambda_geo > 0.0:
                loss = loss + args.lambda_geo * geo_loss_fn(out_w['spatial_concept_heatmap'], mask)
            loss.backward()
            optimizer.step()

            epoch_losses['L_spatial_unlabeled'] += loss.item()
            num_batches += 1

        main_metrics = eval_main_metrics(model, val_loader, device)
        spatial_val = eval_spatial_val_loss(model, val_loader, device, spatial_align_loss, geo_loss_fn, args.lambda_geo)
        epoch_summary = {
            'epoch': epoch,
            'L_spatial_labeled': epoch_losses['L_spatial_labeled'] / max(num_batches, 1),
            'L_spatial_unlabeled': epoch_losses['L_spatial_unlabeled'] / max(num_batches, 1),
            'val_spatial_loss': spatial_val,
            **main_metrics,
        }
        for k, v in epoch_summary.items():
            history[k].append(float(v) if isinstance(v, (int, float, np.floating)) else v)
        print(json.dumps(epoch_summary, ensure_ascii=False))

        metrics_ok = main_metrics['c_acc'] >= args.min_c_acc and main_metrics['y_acc'] >= args.min_y_acc
        torch.save(model.state_dict(), last_weights_path)
        if metrics_ok and spatial_val < best_val_spatial:
            best_val_spatial = spatial_val
            best_epoch = epoch
            torch.save(model.state_dict(), args.save_weights)

    if best_epoch == 0:
        torch.save(model.state_dict(), args.save_weights)

    summary = {
        'best_epoch': best_epoch,
        'best_val_spatial_loss': best_val_spatial,
        'baseline_main_metrics': baseline_metrics,
        'metric_floor': {'min_c_acc': args.min_c_acc, 'min_y_acc': args.min_y_acc},
        'weights_best': os.path.abspath(args.save_weights),
        'weights_last': os.path.abspath(last_weights_path),
        'config_path': os.path.abspath(config_path),
        'history': dict(history),
    }
    summary_path = os.path.join(args.output_dir, 'train_spatial_align_only_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'已写入: {summary_path}')
    print(f'配置文件: {config_path}')
    print(f'最佳权重: {args.save_weights}')
    print(f'最后权重: {last_weights_path}')


if __name__ == '__main__':
    main()
