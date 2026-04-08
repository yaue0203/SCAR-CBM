"""
损失函数模块：实现Stage 5的多任务联合优化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class AlignmentLoss(nn.Module):
    """伪标签对齐损失（仅在可靠样本上）"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, 
                c_heatmap: torch.Tensor,
                c_pseudo: torch.Tensor,
                m_rel: torch.Tensor,
                sample_weights: torch.Tensor = None) -> torch.Tensor:
        """
        L_align = Σ w_x_u · BCE(c_heatmap, c_pseudo)  （仅在 m_rel=1 时）
        
        Args:
            c_heatmap: 模型预测的热力图 [B, num_concepts]
            c_pseudo: 伪标签 [B, num_concepts]
            m_rel: 可靠性掩码 [B]
            sample_weights: 样本权重 [B]
        
        Returns:
            loss: 标量损失
        """
        # BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(
            c_heatmap, c_pseudo, reduction='none'
        )  # [B, num_concepts]
        
        # 平均每个样本
        bce_loss = bce_loss.mean(dim=1)  # [B]
        
        # 仅在可靠样本上计算损失
        bce_loss = bce_loss * m_rel  # [B]
        
        # 应用样本权重
        if sample_weights is not None:
            bce_loss = bce_loss * sample_weights
        
        if self.reduction == 'mean':
            if sample_weights is not None:
                denom = (m_rel * sample_weights).sum() + 1e-8
            else:
                denom = m_rel.sum() + 1e-8
            return bce_loss.sum() / denom
        else:
            return bce_loss.sum()


class ConsistencyLoss(nn.Module):
    """自监督特征一致性损失（仅在不可靠样本上）"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self,
                f_weak: torch.Tensor,
                f_strong: torch.Tensor,
                m_rel: torch.Tensor) -> torch.Tensor:
        """
        L_consistency = (1 - m_rel) · MSE(f_weak, f_strong)
        
        Args:
            f_weak: 弱增强特征 [B, feature_dim]
            f_strong: 强增强特征 [B, feature_dim]
            m_rel: 可靠性掩码 [B]
        
        Returns:
            loss: 标量损失
        """
        # MSE损失
        mse_loss = F.mse_loss(f_weak, f_strong, reduction='none').mean(dim=1)  # [B]
        
        # 仅在不可靠样本上计算
        consistency_loss = mse_loss * (1 - m_rel)  # [B]
        
        if self.reduction == 'mean':
            return consistency_loss.sum() / ((1 - m_rel).sum() + 1e-8)
        else:
            return consistency_loss.sum()


class GeometricConstraintLoss(nn.Module):
    """几何约束损失（破除背景捷径）"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self,
                c_heatmap: torch.Tensor,
                contour_mask: torch.Tensor) -> torch.Tensor:
        """
        L_geo = || c_heatmap ⊙ (1 - contour_mask) ||^2
        
        约束：背景区域（contour_mask=1）的热力图应该为0
        
        Args:
            c_heatmap: 概念热力图，支持 [B, num_concepts] 或 [B, num_concepts, H, W]
            contour_mask: 物理轮廓掩码 [B, 1, H, W]，1表示前景，0表示背景
        
        Returns:
            loss: 标量损失
        """
        if c_heatmap.dim() == 4:
            b, _, h, w = c_heatmap.shape
            if contour_mask.dim() != 4:
                raise ValueError("空间几何约束要求 contour_mask 为 [B,1,H,W] 或 [1,1,H,W]")
            if contour_mask.shape[0] == 1 and b > 1:
                contour_mask = contour_mask.expand(b, -1, -1, -1)
            if contour_mask.shape[-2:] != (h, w):
                contour_mask = F.interpolate(
                    contour_mask.float(), size=(h, w), mode="nearest"
                )
            background_loss = (c_heatmap * (1.0 - contour_mask.float())) ** 2
        else:
            if contour_mask.dim() == 4:
                b = c_heatmap.shape[0]
                if contour_mask.shape[0] == 1 and b > 1:
                    contour_mask = contour_mask.expand(b, -1, -1, -1)
                contour_mask_pooled = F.adaptive_avg_pool2d(contour_mask, output_size=1)
                contour_mask_pooled = contour_mask_pooled.view(contour_mask.shape[0])
            else:
                contour_mask_pooled = contour_mask
            background_loss = (c_heatmap * (1.0 - contour_mask_pooled.unsqueeze(1))) ** 2
        
        if self.reduction == 'mean':
            if c_heatmap.dim() == 4:
                mask = (1.0 - contour_mask.float())
                denom = mask.sum() * c_heatmap.shape[1] + 1e-8
                return background_loss.sum() / denom
            return background_loss.mean()
        else:
            return background_loss.sum()


class GraphRegularizationLoss(nn.Module):
    """概念关系图正则化损失（逻辑自洽）"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self,
                c_heatmap: torch.Tensor,
                c_graph: torch.Tensor) -> torch.Tensor:
        """
        L_graph = || h_c - c_heatmap ||^2
        
        确保GCN推理后的结果与原始预测保持一致
        
        Args:
            c_heatmap: 原始热力图预测 [B, num_concepts]
            c_graph: GCN推理后的结果 [B, num_concepts]
        
        Returns:
            loss: 标量损失
        """
        if c_graph is None:
            return c_heatmap.new_zeros(())
        
        graph_loss = F.mse_loss(c_graph, c_heatmap, reduction=self.reduction)
        
        return graph_loss


class ContrastiveLoss(nn.Module):
    """排他性对比损失（InfoNCE 形式，使用负样本队列）"""
    
    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self,
                anchor_features: torch.Tensor,
                positive_features: torch.Tensor,
                negative_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        L_contrast = -log(exp(sim_pos) / Σ exp(sim_neg))
        
        Args:
            anchor_features: 当前样本特征 [B, D]
            positive_features: 对应正样本特征 [B, D]
            negative_features: 负样本队列 [K, D]
        
        Returns:
            loss: 标量损失
        """
        if negative_features is None or negative_features.numel() == 0:
            return anchor_features.new_zeros(())

        z_anchor = F.normalize(anchor_features, p=2, dim=1)
        z_positive = F.normalize(positive_features, p=2, dim=1)
        z_negative = F.normalize(
            negative_features.to(device=anchor_features.device, dtype=anchor_features.dtype),
            p=2,
            dim=1,
        )

        sim_pos = torch.sum(z_anchor * z_positive, dim=1, keepdim=True)
        sim_neg = torch.matmul(z_anchor, z_negative.t())

        logits = torch.cat([sim_pos, sim_neg], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels, reduction=self.reduction)


class NT_Xent_Loss(nn.Module):
    """NT-Xent损失（Normalized Temperature-scaled Cross Entropy）"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, 
                z_i: torch.Tensor,
                z_j: torch.Tensor) -> torch.Tensor:
        """
        simclr风格的对比损失
        
        Args:
            z_i: 样本i的特征向量 [B, feature_dim]
            z_j: 样本j的特征向量（同一样本的增强版本） [B, feature_dim]
        
        Returns:
            loss: 标量损失
        """
        batch_size = z_i.shape[0]
        
        # 归一化特征
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # 构建相似度矩阵
        # z_i与z_j的相似度
        similarity_matrix = torch.mm(z_i, z_j.t())  # [B, B]
        similarity_matrix = similarity_matrix / self.temperature
        
        # 正样本标签（对角线）
        pos_mask = torch.eye(batch_size, dtype=torch.bool, device=z_i.device)
        
        # logits: 正样本应该有高分，其他的低分
        # 正样本得分
        pos_logits = similarity_matrix[pos_mask].view(batch_size, 1)
        
        # 负样本得分（包括i与i的相似度、i与j^k的相似度等）
        neg_logits = similarity_matrix[~pos_mask].view(batch_size, -1)
        
        # 计算损失
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=z_i.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss


class SpatialConceptAlignmentLoss(nn.Module):
    """
    监督图像空间热力图与多标签概念一致。

    相比仅做 top-k BCE，这里额外加入：
    - 负概念 hardest-region 抑制，减少整图泛亮
    - 正概念峰谷分离，鼓励响应集中到少量真正相关区域
    """

    def __init__(
        self,
        reduction: str = "mean",
        eps: float = 1e-6,
        topk_ratio: float = 0.1,
        negative_weight: float = 0.35,
        separation_weight: float = 0.15,
        separation_margin: float = 0.25,
    ):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
        self.topk_ratio = float(topk_ratio)
        self.negative_weight = float(negative_weight)
        self.separation_weight = float(separation_weight)
        self.separation_margin = float(separation_margin)

    def forward(
        self,
        spatial_concept_heatmap: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            spatial_concept_heatmap: [B, num_concepts, H, W]，每概念的空间概率图
            y_true: [B, num_concepts]，0~1 多标签
        """
        if spatial_concept_heatmap.dim() != 4:
            raise ValueError(
                "SpatialConceptAlignmentLoss 期望 spatial_concept_heatmap 为 [B,C,H,W]"
            )
        b, c, h, w = spatial_concept_heatmap.shape
        flat = spatial_concept_heatmap.view(b, c, h * w)
        y_true = y_true.float()

        h_w = h * w
        if h_w <= 64:
            k = max(4, int(round(0.25 * h_w)))
        else:
            k = max(1, int(round(self.topk_ratio * h_w)))
        topk_vals = flat.topk(k, dim=-1).values
        pooled_topk = topk_vals.mean(dim=-1).clamp(self.eps, 1.0 - self.eps)
        loss_presence = F.binary_cross_entropy(pooled_topk, y_true, reduction='none')

        # 对缺失概念，直接压制其 hardest region 的响应。
        loss_negative = ((1.0 - y_true) * topk_vals.pow(2).mean(dim=-1))

        # 对存在概念，拉开高响应区域与低响应区域的差距，减少大面积泛亮。
        bottomk_vals = torch.topk(flat, k, dim=-1, largest=False).values
        pooled_bottomk = bottomk_vals.mean(dim=-1)
        separation_gap = pooled_topk - pooled_bottomk
        loss_separation = y_true * F.relu(self.separation_margin - separation_gap)

        total = (
            loss_presence
            + self.negative_weight * loss_negative
            + self.separation_weight * loss_separation
        )
        if self.reduction == 'sum':
            return total.sum()
        return total.mean()


class SpatialConsistencyLoss(nn.Module):
    """约束 weak/strong 增强下的空间热力图保持稳定。"""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        heatmap_weak: torch.Tensor,
        heatmap_strong: torch.Tensor,
        m_rel: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if heatmap_weak.shape != heatmap_strong.shape:
            raise ValueError(
                f"空间一致性损失要求形状一致，得到 {tuple(heatmap_weak.shape)} vs {tuple(heatmap_strong.shape)}"
            )
        per_sample = F.mse_loss(heatmap_weak, heatmap_strong, reduction="none")
        per_sample = per_sample.mean(dim=(1, 2, 3))
        if m_rel is not None:
            weights = 1.0 - m_rel.float()
            denom = weights.sum() + 1e-8
            return (per_sample * weights).sum() / denom
        if self.reduction == "mean":
            return per_sample.mean()
        return per_sample.sum()


class SpatialForegroundSeparationLoss(nn.Module):
    """利用前景 mask 把响应往主体内部收，并抑制边框背景偏置。"""

    def __init__(
        self,
        reduction: str = "mean",
        foreground_floor: float = 0.35,
        separation_margin: float = 0.12,
        border_ratio: float = 0.12,
        foreground_weight: float = 0.5,
        border_weight: float = 0.35,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.reduction = reduction
        self.foreground_floor = float(foreground_floor)
        self.separation_margin = float(separation_margin)
        self.border_ratio = float(border_ratio)
        self.foreground_weight = float(foreground_weight)
        self.border_weight = float(border_weight)
        self.eps = float(eps)

    def _border_mask(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        bw = max(1, int(round(min(h, w) * self.border_ratio)))
        mask = torch.zeros((1, 1, h, w), device=device, dtype=dtype)
        mask[:, :, :bw, :] = 1.0
        mask[:, :, -bw:, :] = 1.0
        mask[:, :, :, :bw] = 1.0
        mask[:, :, :, -bw:] = 1.0
        return mask

    def forward(
        self,
        spatial_heatmap: torch.Tensor,
        contour_mask: torch.Tensor,
        concept_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if spatial_heatmap.dim() != 4:
            raise ValueError('SpatialForegroundSeparationLoss 期望 spatial_heatmap 为 [B,C,H,W]')
        if contour_mask.dim() != 4:
            raise ValueError('SpatialForegroundSeparationLoss 期望 contour_mask 为 [B,1,H,W] 或 [1,1,H,W]')

        b, c, h, w = spatial_heatmap.shape
        mask = contour_mask.float()
        if mask.shape[0] == 1 and b > 1:
            mask = mask.expand(b, -1, -1, -1)
        if mask.shape[-2:] != (h, w):
            mask = F.interpolate(mask, size=(h, w), mode='nearest')
        mask = mask.clamp(0.0, 1.0)
        bg_mask = 1.0 - mask

        if concept_targets is None:
            concept_weights = spatial_heatmap.mean(dim=(2, 3)).detach()
        else:
            concept_weights = concept_targets.float()
        concept_weights = concept_weights.clamp(0.0, 1.0)

        fg_area = mask.sum(dim=(2, 3)).clamp_min(self.eps)
        bg_area = bg_mask.sum(dim=(2, 3)).clamp_min(self.eps)

        fg_mean = (spatial_heatmap * mask).sum(dim=(2, 3)) / fg_area
        bg_mean = (spatial_heatmap * bg_mask).sum(dim=(2, 3)) / bg_area

        loss_foreground = concept_weights * F.relu(self.foreground_floor - fg_mean)
        loss_separation = concept_weights * F.relu(self.separation_margin - (fg_mean - bg_mean))

        border_mask = self._border_mask(h, w, spatial_heatmap.device, spatial_heatmap.dtype)
        border_bg = (border_mask * bg_mask).clamp(0.0, 1.0)
        border_area = border_bg.sum(dim=(2, 3)).clamp_min(self.eps)
        border_mean = (spatial_heatmap * border_bg).sum(dim=(2, 3)) / border_area
        loss_border = concept_weights * border_mean

        total = loss_separation + self.foreground_weight * loss_foreground + self.border_weight * loss_border
        if self.reduction == 'sum':
            return total.sum()
        denom = concept_weights.sum().clamp_min(self.eps)
        return total.sum() / denom


class SpatialPseudoAlignmentLoss(nn.Module):
    """将空间热力图池化为概念 logits，并与伪标签对齐。"""

    def __init__(self, topk_ratio: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.topk_ratio = float(topk_ratio)
        self.eps = float(eps)
        self.base_loss = AlignmentLoss()

    def pool_to_logits(self, spatial_heatmap: torch.Tensor) -> torch.Tensor:
        if spatial_heatmap.dim() != 4:
            raise ValueError("SpatialPseudoAlignmentLoss 期望 spatial_heatmap 为 [B,C,H,W]")
        b, c, h, w = spatial_heatmap.shape
        flat = spatial_heatmap.view(b, c, h * w)
        h_w = h * w
        if h_w <= 64:
            k = max(4, int(round(0.25 * h_w)))
        else:
            k = max(1, int(round(self.topk_ratio * h_w)))
        pooled = flat.topk(k, dim=-1).values.mean(dim=-1)
        pooled = pooled.clamp(self.eps, 1.0 - self.eps)
        return torch.logit(pooled)

    def forward(
        self,
        spatial_heatmap: torch.Tensor,
        c_pseudo: torch.Tensor,
        m_rel: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pooled_logits = self.pool_to_logits(spatial_heatmap)
        return self.base_loss(pooled_logits, c_pseudo, m_rel, sample_weights)


class MultiTaskLoss(nn.Module):
    """多任务联合损失"""
    
    def __init__(self,
                 lambda1: float = 0.5,
                 lambda2: float = 0.4,
                 lambda3: float = 2.5,
                 lambda4: float = 0.08,
                 lambda5: float = 0.02):
        super().__init__()
        # 归一化相对权重，使 λ1…λ5 之和为 1（保持比例不变）
        total = lambda1 + lambda2 + lambda3 + lambda4 + lambda5
        if total <= 0:
            raise ValueError("MultiTaskLoss 的 λ1…λ5 之和必须为正")
        self.lambda1 = lambda1 / total
        self.lambda2 = lambda2 / total
        self.lambda3 = lambda3 / total
        self.lambda4 = lambda4 / total
        self.lambda5 = lambda5 / total
        
        # 初始化各个损失函数
        self.loss_alignment = AlignmentLoss()
        self.loss_consistency = ConsistencyLoss()
        self.loss_geometric = GeometricConstraintLoss()
        self.loss_graph = GraphRegularizationLoss()
        self.loss_contrast = ContrastiveLoss()
        self.loss_spatial_consistency = SpatialConsistencyLoss()
        self.loss_spatial_pseudo = SpatialPseudoAlignmentLoss()
    
    def forward(self,
                # 监督部分
                c_heatmap_labeled: torch.Tensor,
                y_true: torch.Tensor,
                # 半监督部分
                c_heatmap_unlabeled: torch.Tensor,
                c_pseudo: torch.Tensor,
                f_weak: torch.Tensor,
                f_strong: torch.Tensor,
                m_rel: torch.Tensor,
                # 几何和图部分
                c_graph: Optional[torch.Tensor] = None,
                contour_mask: Optional[torch.Tensor] = None,
                spatial_heatmap: Optional[torch.Tensor] = None,
                spatial_heatmap_strong: Optional[torch.Tensor] = None,
                spatial_pseudo_weight: float = 0.0,
                # 对比学习部分
                negative_features: Optional[torch.Tensor] = None,
                sample_weights: Optional[torch.Tensor] = None) -> dict:
        """
        计算多任务联合损失
        
        L_total = L_supervised + λ1·L_align + λ2·L_consistency + λ3·L_geo + λ4·L_graph + λ5·L_contrast
        
        Returns:
            loss_dict: 包含各项损失的字典
        """
        losses = {}
        total_loss = 0.0
        
        # L_supervised: 有标签数据的标准损失（半监督分支可不传）
        if c_heatmap_labeled is not None and y_true is not None:
            loss_supervised = F.binary_cross_entropy_with_logits(
                c_heatmap_labeled, y_true
            )
            losses['L_supervised'] = loss_supervised
            total_loss = total_loss + loss_supervised
        else:
            losses['L_supervised'] = c_heatmap_unlabeled.new_zeros(())
        
        # L_align: 伪标签对齐损失
        loss_align = self.loss_alignment(
            c_heatmap_unlabeled, c_pseudo, m_rel, sample_weights
        )
        losses['L_align'] = loss_align
        total_loss = total_loss + self.lambda1 * loss_align
        
        # L_consistency: 特征一致性损失
        loss_consistency = self.loss_consistency(f_weak, f_strong, m_rel)
        losses['L_consistency'] = loss_consistency
        total_loss = total_loss + self.lambda2 * loss_consistency
        
        # L_geo: 几何约束损失
        if contour_mask is not None:
            geo_target = spatial_heatmap if spatial_heatmap is not None else c_heatmap_unlabeled
            loss_geo = self.loss_geometric(geo_target, contour_mask)
            losses['L_geo'] = loss_geo
            total_loss = total_loss + self.lambda3 * loss_geo
        else:
            losses['L_geo'] = c_heatmap_unlabeled.new_zeros(())

        if spatial_heatmap is not None and spatial_heatmap_strong is not None:
            loss_spatial_consistency = self.loss_spatial_consistency(
                spatial_heatmap, spatial_heatmap_strong, m_rel
            )
            losses['L_spatial_consistency'] = loss_spatial_consistency
        else:
            losses['L_spatial_consistency'] = c_heatmap_unlabeled.new_zeros(())

        if spatial_heatmap is not None:
            loss_spatial_pseudo = self.loss_spatial_pseudo(
                spatial_heatmap, c_pseudo, m_rel, sample_weights
            )
            losses['L_spatial_pseudo'] = loss_spatial_pseudo
            if spatial_pseudo_weight > 0.0:
                total_loss = total_loss + float(spatial_pseudo_weight) * loss_spatial_pseudo
        else:
            losses['L_spatial_pseudo'] = c_heatmap_unlabeled.new_zeros(())
        
        # L_graph: 图正则化损失
        if c_graph is not None:
            loss_graph = self.loss_graph(c_heatmap_unlabeled, c_graph)
            losses['L_graph'] = loss_graph
            total_loss = total_loss + self.lambda4 * loss_graph
        else:
            losses['L_graph'] = c_heatmap_unlabeled.new_zeros(())
        
        # L_contrast: 对比损失
        if negative_features is not None:
            loss_contrast = self.loss_contrast(f_weak, f_strong, negative_features)
            losses['L_contrast'] = loss_contrast
            total_loss = total_loss + self.lambda5 * loss_contrast
        else:
            losses['L_contrast'] = c_heatmap_unlabeled.new_zeros(())
        
        losses['L_total'] = total_loss
        
        return losses


class DynamicWeightedLoss(nn.Module):
    """动态加权的多任务损失（自动调整各任务权重）"""
    
    def __init__(self, num_tasks: int = 5, initial_weights=None):
        super().__init__()
        
        if initial_weights is None:
            initial_weights = torch.ones(num_tasks) / num_tasks
        
        self.initial_weights = initial_weights
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, 
                losses: Dict[str, torch.Tensor],
                task_names: list) -> torch.Tensor:
        """
        使用不确定性加权的多任务学习
        
        L_total = Σ (1 / (2 * σ_i^2)) * L_i + log(σ_i)
        
        Args:
            losses: 各任务的损失值
            task_names: 任务名称列表
        
        Returns:
            weighted_loss: 加权后的总损失
        """
        weighted_loss = 0.0
        
        for i, task_name in enumerate(task_names):
            if task_name in losses:
                loss_val = losses[task_name]
                precision = torch.exp(-self.log_vars[i])
                weighted_loss += precision * loss_val + self.log_vars[i]
        
        return weighted_loss
