"""
Stage 5：图逻辑优化 — 主训练框架（从原 train_framework.py 迁移）

包含 `ConceptualSSLFramework` 的完整训练循环：监督 + 半监督多任务损失、伪标签、验证与早停。
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from typing import Dict, Optional, Tuple
from collections import defaultdict
import logging

from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from core_modules import (
    ConceptHypergraph, NegativeSampleQueue, PseudoLabelHistory,
    LabeledDataset, UnlabeledDataset,
    compute_entropy, compute_density, knn_retrieval
)
from pseudo_label_and_sampling import (
    PseudoLabelGenerator, UncertaintyGuidedSampler,
    CurriculumLearningScheduler
)
from loss_functions import MultiTaskLoss, SpatialConceptAlignmentLoss
from stage1_hypergraph import build_concept_hypergraph as build_hypergraph_stage1


def _multilabel_concept_and_class_metrics(
    y_true: np.ndarray,
    logits: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    多标签概念指标：
    - c_acc / c_auc：micro（所有样本×概念位点池化）
    - y_acc / y_auc：macro（逐概念维平均，class-level）
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    y_bin = (y_true >= 0.5).astype(np.int32)
    probs = expit(logits)
    if thresholds is None:
        pred_bin = (probs >= 0.5).astype(np.int32)
    else:
        thr = np.asarray(thresholds, dtype=np.float64).reshape(1, -1)
        pred_bin = (probs >= thr).astype(np.int32)

    concept_acc_micro = float((pred_bin == y_bin).mean())

    n_labels = y_bin.shape[1]
    class_accs: list[float] = []
    class_aucs: list[float] = []
    class_bal_accs: list[float] = []
    class_f1s: list[float] = []
    for j in range(n_labels):
        yj = y_bin[:, j]
        pj = pred_bin[:, j]
        sj = probs[:, j]
        class_accs.append(float((pj == yj).mean()))
        class_f1s.append(float(f1_score(yj, pj, zero_division=0)))
        if int(yj.min()) != int(yj.max()):
            class_bal_accs.append(float(balanced_accuracy_score(yj, pj)))
            class_aucs.append(float(roc_auc_score(yj, sj)))

    class_acc_macro = float(np.mean(class_accs)) if class_accs else float("nan")
    class_auc_macro = float(np.mean(class_aucs)) if class_aucs else float("nan")
    class_bal_acc_macro = float(np.mean(class_bal_accs)) if class_bal_accs else float("nan")
    class_f1_macro = float(np.mean(class_f1s)) if class_f1s else float("nan")

    try:
        concept_auc_micro = float(roc_auc_score(y_bin, probs, average="micro"))
    except ValueError:
        concept_auc_micro = float("nan")

    return {
        "c_acc": concept_acc_micro,
        "c_auc": concept_auc_micro,
        "y_acc": class_acc_macro,
        "y_auc": class_auc_macro,
        "y_bal_acc": class_bal_acc_macro,
        "y_f1": class_f1_macro,
    }


def _fit_per_concept_thresholds(
    y_true: np.ndarray,
    logits: np.ndarray,
    num_candidates: int = 19,
) -> np.ndarray:
    """用训练集预测为每个概念单独选阈值。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    y_bin = (y_true >= 0.5).astype(np.int32)
    probs = expit(logits)
    thresholds = np.full(y_bin.shape[1], 0.5, dtype=np.float64)
    candidates = np.linspace(0.05, 0.95, num_candidates, dtype=np.float64)

    for j in range(y_bin.shape[1]):
        yj = y_bin[:, j]
        if int(yj.min()) == int(yj.max()):
            continue
        pj = probs[:, j]
        best_thr = 0.5
        best_bal_acc = -1.0
        best_f1 = -1.0
        for thr in candidates:
            pred = (pj >= thr).astype(np.int32)
            bal_acc = float(balanced_accuracy_score(yj, pred))
            f1 = float(f1_score(yj, pred, zero_division=0))
            if bal_acc > best_bal_acc or (bal_acc == best_bal_acc and f1 > best_f1):
                best_bal_acc = bal_acc
                best_f1 = f1
                best_thr = float(thr)
        thresholds[j] = best_thr

    return thresholds


class ConceptualSSLFramework:
    """完整的概念-半监督学习框架"""

    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda',
                 num_concepts: int = 10,
                 num_epochs: int = 100,
                 batch_size_labeled: int = 32,
                 batch_size_unlabeled: int = 64,
                 learning_rate: float = 0.001,
                 lambda_spatial_align: float = 0.2,
                 lambda_spatial_consistency: float = 0.15):
        """
        初始化框架

        Args:
            model: 完整的神经网络模型
            device: 计算设备
            num_concepts: 概念数量
            num_epochs: 训练轮数
            batch_size_labeled: 有标签数据batch大小
            batch_size_unlabeled: 无标签数据batch大小
            learning_rate: 学习率
            lambda_spatial_align: 图像空间热力图与概念标签对齐损失权重（0 关闭）
            lambda_spatial_consistency: weak/strong 空间热力图一致性损失权重（0 关闭）
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.num_concepts = num_concepts
        self.num_epochs = num_epochs
        self.batch_size_labeled = batch_size_labeled
        self.batch_size_unlabeled = batch_size_unlabeled

        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        # 损失函数
        self.loss_fn = MultiTaskLoss()
        self.lambda_spatial_align = float(lambda_spatial_align)
        self.lambda_spatial_consistency = float(lambda_spatial_consistency)
        self.spatial_align_loss = SpatialConceptAlignmentLoss()

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs
        )

        # 组件初始化
        self.hypergraph = None
        self.negative_queue = NegativeSampleQueue()
        self.pseudo_label_history = None
        self.pseudo_label_generator = PseudoLabelGenerator(num_concepts)
        self.sampler = UncertaintyGuidedSampler()
        self.curriculum_scheduler = CurriculumLearningScheduler(
            total_epochs=num_epochs
        )

        # 日志
        self.logger = self._setup_logger()
        self.metrics = defaultdict(list)
        self.eval_thresholds: Optional[np.ndarray] = None

    def _setup_logger(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    # ==================== Stage 1: 概念关系图构建 ====================

    def build_concept_hypergraph(
        self,
        labeled_concept_ids: np.ndarray,
        cooccurrence_threshold: float = 0.1,
    ):
        """
        Stage 1: 离线构建概念关系超图（与 stage1_hypergraph.build_concept_hypergraph 一致）

        Args:
            labeled_concept_ids: 有标签数据的概念ID [N_labeled, num_concepts]
            cooccurrence_threshold: 共现归一化权重阈值
        """
        self.logger.info("Building concept hypergraph (Stage 1)...")
        self.hypergraph = build_hypergraph_stage1(
            labeled_concept_ids,
            num_concepts=self.num_concepts,
            cooccurrence_threshold=cooccurrence_threshold,
        )

    # ==================== Stage 2: 细粒度特征提取 ====================

    def extract_features(self,
                        x: torch.Tensor,
                        adj_matrix: torch.Tensor) -> Dict:
        """
        Stage 2: 细粒度空间-语义特征提取

        Args:
            x: 输入图像 [B, C, H, W]
            adj_matrix: 概念关系图邻接矩阵

        Returns:
            outputs: 包含各种特征和预测的字典
        """
        outputs = self.model(x, adj_matrix)
        return outputs

    # ==================== Stage 3: 自适应伪标签生成 ====================

    def generate_pseudo_labels(self,
                              features_unlabeled: torch.Tensor,
                              c_heatmap: torch.Tensor,
                              features_labeled: torch.Tensor,
                              labels_labeled: torch.Tensor,
                              contour_mask: Optional[torch.Tensor] = None,
                              sample_ids: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage 3: 自适应伪标签生成

        Args:
            features_unlabeled: 无标签数据特征 [B_u, feature_dim]
            c_heatmap: 无标签数据的概念热力图 [B_u, num_concepts]
            features_labeled: 有标签数据特征 [B_l, feature_dim]
            labels_labeled: 有标签数据标签 [B_l, num_concepts]
            contour_mask: 物理轮廓掩码

        Returns:
            c_pseudo: 伪标签 [B_u, num_concepts]
            m_rel: 可靠性掩码 [B_u]
        """
        # KNN检索
        c_knn = knn_retrieval(
            features_unlabeled, features_labeled, labels_labeled, k=5
        )

        # 伪标签生成
        c_pseudo, m_rel = self.pseudo_label_generator.generate(
            c_heatmap, c_knn, contour_mask, self.pseudo_label_history, sample_ids
        )

        return c_pseudo, m_rel

    # ==================== Stage 4: 不确定性采样 ====================

    def sample_unlabeled_data(self,
                             features: torch.Tensor,
                             logits: torch.Tensor,
                             batch_size: int) -> np.ndarray:
        """
        Stage 4: 不确定性引导的反课程采样

        Args:
            features: 特征向量 [N, feature_dim]
            logits: 模型预测 [N, num_concepts]
            batch_size: 采样大小

        Returns:
            sampled_indices: 采样的样本索引
        """
        sampled_indices = self.sampler.sample(features, logits, batch_size)
        return sampled_indices

    def build_unlabeled_epoch_loader(self,
                                     train_unlabeled_loader: DataLoader,
                                     adj_matrix: torch.Tensor) -> Tuple[DataLoader, float]:
        """
        使用 Stage 4 权重为当前 epoch 重建无标签 DataLoader。
        返回新的 loader 以及本轮采样后唯一样本占比。
        """
        dataset = train_unlabeled_loader.dataset
        total_unlabeled = len(dataset)
        if total_unlabeled == 0:
            return train_unlabeled_loader, 0.0

        was_training = self.model.training
        self.model.eval()

        all_features = []
        all_logits = []

        with torch.no_grad():
            for batch in train_unlabeled_loader:
                x_u_weak = batch[0].to(self.device)
                outputs = self.extract_features(x_u_weak, adj_matrix)
                all_features.append(outputs['f_visual'].detach())
                all_logits.append(outputs['c_heatmap'].detach())

        if was_training:
            self.model.train()

        features = torch.cat(all_features, dim=0)
        logits = torch.cat(all_logits, dim=0)

        sampled_indices = self.sample_unlabeled_data(features, logits, total_unlabeled)
        sampled_unique_ratio = float(len(np.unique(sampled_indices))) / float(total_unlabeled)

        dataset_weights = self.sampler.compute_sampling_weights(features, logits).detach().cpu()
        dataset_weights = dataset_weights.to(dtype=torch.double)

        weighted_sampler = WeightedRandomSampler(
            weights=dataset_weights,
            num_samples=total_unlabeled,
            replacement=True,
        )

        unlabeled_loader = DataLoader(
            dataset,
            batch_size=train_unlabeled_loader.batch_size,
            sampler=weighted_sampler,
            num_workers=train_unlabeled_loader.num_workers,
            collate_fn=train_unlabeled_loader.collate_fn,
            pin_memory=train_unlabeled_loader.pin_memory,
            drop_last=train_unlabeled_loader.drop_last,
        )
        return unlabeled_loader, sampled_unique_ratio

    # ==================== 训练步骤 ====================

    def train_epoch(self,
                   train_labeled_loader: DataLoader,
                   train_unlabeled_loader: DataLoader,
                   adj_matrix: torch.Tensor,
                   contour_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        训练一个epoch

        Args:
            train_labeled_loader: 有标签数据的DataLoader
            train_unlabeled_loader: 无标签数据的DataLoader
            adj_matrix: 概念关系图邻接矩阵
            contour_mask: 物理轮廓掩码

        Returns:
            metrics_dict: 当前epoch的指标
        """
        self.model.train()
        epoch_losses = defaultdict(float)
        num_batches = 0

        # 获取所有有标签数据
        all_features_labeled = []
        all_labels_labeled = []
        all_logits_labeled = []

        for batch_l in train_labeled_loader:
            x_l, y_l, c_l, _ = batch_l[:4]
            x_l = x_l.to(self.device)
            y_l = y_l.to(self.device)

            outputs_l = self.extract_features(x_l, adj_matrix)
            all_features_labeled.append(outputs_l['f_visual'].detach())
            all_labels_labeled.append(y_l.detach())
            all_logits_labeled.append(outputs_l['c_heatmap'].detach())

        features_labeled = torch.cat(all_features_labeled, dim=0)
        labels_labeled = torch.cat(all_labels_labeled, dim=0)
        logits_labeled = torch.cat(all_logits_labeled, dim=0)
        self.eval_thresholds = _fit_per_concept_thresholds(
            labels_labeled.detach().cpu().numpy(),
            logits_labeled.detach().cpu().numpy(),
        )

        # 训练有标签数据
        for batch_l in train_labeled_loader:
            x_l, y_l, c_l, idx_l = batch_l[:4]
            x_l = x_l.to(self.device)
            y_l = y_l.to(self.device)

            self.optimizer.zero_grad()

            # 前向传播
            outputs_l = self.extract_features(x_l, adj_matrix)
            c_heatmap_l = outputs_l['c_heatmap']

            # 有标签损失
            loss_supervised = nn.functional.binary_cross_entropy_with_logits(
                c_heatmap_l, y_l
            )
            loss_total = loss_supervised
            if self.lambda_spatial_align > 0.0:
                smap = outputs_l.get("spatial_concept_heatmap")
                if smap is not None:
                    loss_spa = self.spatial_align_loss(smap, y_l)
                    loss_total = loss_total + self.lambda_spatial_align * loss_spa
                    epoch_losses["L_spatial_align"] += loss_spa.item()

            loss_total.backward()
            self.optimizer.step()

            epoch_losses['L_supervised'] += loss_supervised.item()
            num_batches += 1

        # 训练无标签数据（Stage 4: 先为当前 epoch 重建采样后的无标签 loader）
        sampled_unlabeled_loader, sampled_unique_ratio = self.build_unlabeled_epoch_loader(
            train_unlabeled_loader,
            adj_matrix,
        )
        epoch_losses['stage4_unique_ratio'] += sampled_unique_ratio

        unlabeled_iterator = iter(sampled_unlabeled_loader)
        for batch_u in unlabeled_iterator:
            x_u_weak, x_u_strong, idx_u = batch_u[:3]
            batch_contour_mask = batch_u[3] if len(batch_u) > 3 else contour_mask
            x_u_weak = x_u_weak.to(self.device)
            x_u_strong = x_u_strong.to(self.device)
            if batch_contour_mask is not None:
                batch_contour_mask = batch_contour_mask.to(self.device)

            self.optimizer.zero_grad()

            # 前向传播
            outputs_weak = self.extract_features(x_u_weak, adj_matrix)
            outputs_strong = self.extract_features(x_u_strong, adj_matrix)

            c_heatmap = outputs_weak['c_heatmap']
            f_visual_weak = outputs_weak['f_visual']
            f_visual_strong = outputs_strong['f_visual']

            # 生成伪标签
            c_pseudo, m_rel = self.generate_pseudo_labels(
                f_visual_weak, c_heatmap,
                features_labeled, labels_labeled,
                batch_contour_mask,
                sample_ids=idx_u.detach().cpu().numpy().astype(np.int64),
            )

            reliable_mask = m_rel > 0
            if bool(reliable_mask.any()):
                idx_u_np = idx_u.detach().cpu().numpy().astype(np.int64)
                self.pseudo_label_history.update(
                    idx_u_np[reliable_mask.detach().cpu().numpy()],
                    c_pseudo.detach()[reliable_mask],
                )

            # 计算样本权重（基于不确定性）
            uncertainties = compute_entropy(c_heatmap)
            sample_weights = 1.0 / (uncertainties + 1e-8)

            negative_features = self.negative_queue.get_negatives(
                k=min(256, len(self.negative_queue.queue))
            )
            if negative_features is not None:
                negative_features = negative_features.to(self.device)

            # 计算多任务损失
            losses_dict = self.loss_fn(
                c_heatmap_labeled=None,
                y_true=None,
                c_heatmap_unlabeled=c_heatmap,
                c_pseudo=c_pseudo,
                f_weak=f_visual_weak,
                f_strong=f_visual_strong,
                m_rel=m_rel,
                c_graph=outputs_weak.get('c_graph'),
                contour_mask=batch_contour_mask,
                spatial_heatmap=outputs_weak.get('spatial_concept_heatmap'),
                spatial_heatmap_strong=outputs_strong.get('spatial_concept_heatmap'),
                spatial_pseudo_weight=self.lambda_spatial_align,
                negative_features=negative_features,
                sample_weights=sample_weights
            )

            loss = losses_dict['L_total']
            if self.lambda_spatial_consistency > 0.0:
                loss = loss + self.lambda_spatial_consistency * losses_dict['L_spatial_consistency']

            if loss > 0:
                loss.backward()
                self.optimizer.step()

            # 记录损失
            for key, val in losses_dict.items():
                if isinstance(val, torch.Tensor):
                    epoch_losses[key] += val.item()

            # 更新负样本队列
            self.negative_queue.update(f_visual_weak)

            num_batches += 1

        # 平均损失
        metrics_dict = {
            key: val / max(num_batches, 1)
            for key, val in epoch_losses.items()
        }

        return metrics_dict

    def validate(self,
                val_loader: DataLoader,
                adj_matrix: torch.Tensor) -> Dict[str, float]:
        """
        验证阶段

        Args:
            val_loader: 验证数据加载器
            adj_matrix: 概念关系图邻接矩阵

        Returns:
            metrics_dict: 验证指标
        """
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        ys_list: list[np.ndarray] = []
        logits_list: list[np.ndarray] = []

        with torch.no_grad():
            for x, y, _, _ in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                outputs = self.extract_features(x, adj_matrix)
                c_heatmap = outputs['c_heatmap']

                loss = nn.functional.binary_cross_entropy_with_logits(
                    c_heatmap, y
                )

                total_loss += loss.item() * x.shape[0]
                num_samples += x.shape[0]
                ys_list.append(y.detach().cpu().numpy())
                logits_list.append(c_heatmap.detach().cpu().numpy())

        avg_loss = total_loss / max(num_samples, 1)
        y_all = np.concatenate(ys_list, axis=0)
        logits_all = np.concatenate(logits_list, axis=0)
        m = _multilabel_concept_and_class_metrics(y_all, logits_all)
        tuned = {}
        if self.eval_thresholds is not None:
            tuned_metrics = _multilabel_concept_and_class_metrics(
                y_all, logits_all, thresholds=self.eval_thresholds
            )
            tuned = {
                "c_acc_tuned": tuned_metrics["c_acc"],
                "y_acc_tuned": tuned_metrics["y_acc"],
                "y_bal_acc_tuned": tuned_metrics["y_bal_acc"],
                "y_f1_tuned": tuned_metrics["y_f1"],
            }
        return {"val_loss": avg_loss, **m, **tuned}

    def fit(
        self,
        train_labeled_loader: DataLoader,
        train_unlabeled_loader: DataLoader,
        val_loader: DataLoader,
        labeled_concept_ids: Optional[np.ndarray] = None,
        prebuilt_hypergraph: Optional[ConceptHypergraph] = None,
        cooccurrence_threshold: float = 0.1,
        contour_mask: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        完整的训练流程

        Args:
            train_labeled_loader: 有标签训练数据
            train_unlabeled_loader: 无标签训练数据
            val_loader: 验证数据
            labeled_concept_ids: 有标签数据的概念ID；与 prebuilt_hypergraph 二选一
            prebuilt_hypergraph: 若已离线跑过 Stage 1，可直接传入，fit 内不再构建
            cooccurrence_threshold: 仅在根据 labeled_concept_ids 构建 Stage 1 时生效
            contour_mask: 物理轮廓掩码

        Returns:
            training_history: 训练历史
        """
        if prebuilt_hypergraph is not None:
            if prebuilt_hypergraph.num_concepts != self.num_concepts:
                raise ValueError(
                    f"prebuilt_hypergraph.num_concepts={prebuilt_hypergraph.num_concepts} "
                    f"与框架 num_concepts={self.num_concepts} 不一致"
                )
            self.hypergraph = prebuilt_hypergraph
            self.logger.info(
                "使用预构建的 Stage 1 超图（跳过 build_concept_hypergraph）"
            )
        elif labeled_concept_ids is not None:
            self.build_concept_hypergraph(
                labeled_concept_ids,
                cooccurrence_threshold=cooccurrence_threshold,
            )
        else:
            raise ValueError(
                "fit() 需要 labeled_concept_ids 或 prebuilt_hypergraph 之一"
            )

        if contour_mask is not None:
            contour_mask = contour_mask.to(self.device)

        adj_matrix = self.hypergraph.get_adjacency_matrix(self.device)

        # 初始化伪标签历史。若无标签集经过 Subset 过滤，history 仍按原始索引空间分配，
        # 以兼容 DataLoader 返回的原始 sample id。
        unlabeled_dataset = train_unlabeled_loader.dataset
        if hasattr(unlabeled_dataset, 'indices') and len(unlabeled_dataset.indices) > 0:
            total_unlabeled = int(max(unlabeled_dataset.indices)) + 1
        else:
            total_unlabeled = len(unlabeled_dataset)
        self.pseudo_label_history = PseudoLabelHistory(total_unlabeled)

        training_history = defaultdict(list)
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(self.num_epochs):
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            self.logger.info(f"{'='*50}")

            # 训练
            train_metrics = self.train_epoch(
                train_labeled_loader,
                train_unlabeled_loader,
                adj_matrix,
                contour_mask
            )

            # 验证
            val_metrics = self.validate(val_loader, adj_matrix)

            # 记录指标
            for key, val in train_metrics.items():
                training_history[f'train_{key}'].append(val)
                self.logger.info(f"  {key}: {val:.4f}")

            for key, val in val_metrics.items():
                training_history[key].append(val)
                if isinstance(val, float) and val != val:
                    self.logger.info(f"  {key}: nan")
                else:
                    self.logger.info(f"  {key}: {val:.4f}")

            def _fmt(v: float) -> str:
                if isinstance(v, float) and v != v:
                    return "nan"
                return f"{v:.4f}"

            self.logger.info(
                f"  [验证] c_acc={_fmt(val_metrics['c_acc'])} c_auc={_fmt(val_metrics['c_auc'])} "
                f"y_acc={_fmt(val_metrics['y_acc'])} y_auc={_fmt(val_metrics['y_auc'])} "
                f"y_bal_acc={_fmt(val_metrics['y_bal_acc'])} y_f1={_fmt(val_metrics['y_f1'])} "
                f"(c=micro, y=逐概念macro)"
            )
            if 'c_acc_tuned' in val_metrics:
                self.logger.info(
                    f"  [阈值校准] c_acc_tuned={_fmt(val_metrics['c_acc_tuned'])} "
                    f"y_acc_tuned={_fmt(val_metrics['y_acc_tuned'])} "
                    f"y_bal_acc_tuned={_fmt(val_metrics['y_bal_acc_tuned'])} "
                    f"y_f1_tuned={_fmt(val_metrics['y_f1_tuned'])}"
                )

            # 学习率调度
            self.scheduler.step()

            # 课程学习步骤
            self.curriculum_scheduler.step_epoch()
            self.pseudo_label_generator.epoch += 1

            # 早停
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                patience_counter = 0
                self.logger.info(f"  Best model updated! Loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        self.logger.info(f"\nTraining completed!")
        return dict(training_history)

# ==================== 使用示例 ====================

def create_example_data(num_labeled: int = 100,
                       num_unlabeled: int = 1000,
                       num_concepts: int = 10,
                       image_size: int = 32):
    """创建示例数据"""

    # 有标签数据
    X_labeled = np.random.randn(num_labeled, 3, image_size, image_size).astype(np.float32)
    y_labeled = np.random.rand(num_labeled, num_concepts).astype(np.float32)
    c_labeled = np.random.rand(num_labeled, num_concepts).astype(np.float32)

    # 无标签数据
    X_unlabeled = np.random.randn(num_unlabeled, 3, image_size, image_size).astype(np.float32)

    # 物理轮廓掩码
    contour_mask = np.random.rand(1, 1, image_size, image_size).astype(np.float32)
    contour_mask = (contour_mask > 0.5).astype(np.float32)
    contour_mask = torch.FloatTensor(contour_mask)

    return {
        'X_labeled': X_labeled,
        'y_labeled': y_labeled,
        'c_labeled': c_labeled,
        'X_unlabeled': X_unlabeled,
        'contour_mask': contour_mask
    }
