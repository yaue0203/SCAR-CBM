"""
损失函数模块：实现Stage 5的多任务联合优化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


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
            return bce_loss.sum() / (m_rel.sum() + 1e-8)
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
            c_heatmap: 概念热力图 [B, num_concepts]
            contour_mask: 物理轮廓掩码 [B, 1, H, W]，1表示背景，0表示前景
        
        Returns:
            loss: 标量损失
        """
        # 如果contour_mask不是正确的维度，进行调整
        if contour_mask.dim() == 4:
            # 背景区域（contour_mask=1）应该对应热力图的低值
            # 这里假设我们有空间信息，需要平均池化
            contour_mask_pooled = F.adaptive_avg_pool2d(contour_mask, output_size=1)  # [B, 1, 1, 1]
            contour_mask_pooled = contour_mask_pooled.squeeze()  # [B]
        else:
            contour_mask_pooled = contour_mask  # [B]
        
        # 在背景区域应该为0
        background_loss = (c_heatmap * contour_mask_pooled.unsqueeze(1)) ** 2
        
        if self.reduction == 'mean':
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
            return torch.tensor(0.0, device=c_heatmap.device)
        
        graph_loss = F.mse_loss(c_graph, c_heatmap, reduction=self.reduction)
        
        return graph_loss


class ContrastiveLoss(nn.Module):
    """排他性对比损失（拉远易混淆概念）"""
    
    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self,
                c_heatmap: torch.Tensor,
                confusable_pairs: Tuple[torch.Tensor, torch.Tensor],
                negative_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        L_contrast = -log(exp(sim_pos) / Σ exp(sim_neg))
        
        Args:
            c_heatmap: 概念热力图预测 [B, num_concepts]
            confusable_pairs: (positive_idx, negative_idx) 混淆概念对
            negative_features: 负样本特征队列 [K, num_concepts]
        
        Returns:
            loss: 标量损失
        """
        pos_idx, neg_idx = confusable_pairs
        
        # 获取正和负样本的热力图
        c_pos = c_heatmap[:, pos_idx]  # [B, num_pos_concepts]
        c_neg = c_heatmap[:, neg_idx]  # [B, num_neg_concepts]
        
        # 计算相似度
        sim_pos = F.cosine_similarity(c_pos, c_pos, dim=-1)  # [B]
        sim_neg = F.cosine_similarity(c_pos, c_neg, dim=-1)  # [B]
        
        # 对比损失
        logits = torch.stack([sim_pos, sim_neg], dim=1) / self.temperature  # [B, 2]
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        contrast_loss = F.cross_entropy(logits, labels, reduction=self.reduction)
        
        return contrast_loss


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


class MultiTaskLoss(nn.Module):
    """多任务联合损失"""
    
    def __init__(self,
                 lambda1: float = 0.5,
                 lambda2: float = 0.3,
                 lambda3: float = 0.2,
                 lambda4: float = 0.1,
                 lambda5: float = 0.1):
        super().__init__()
        
        self.lambda1 = lambda1  # L_align
        self.lambda2 = lambda2  # L_consistency
        self.lambda3 = lambda3  # L_geo
        self.lambda4 = lambda4  # L_graph
        self.lambda5 = lambda5  # L_contrast
        
        # 初始化各个损失函数
        self.loss_alignment = AlignmentLoss()
        self.loss_consistency = ConsistencyLoss()
        self.loss_geometric = GeometricConstraintLoss()
        self.loss_graph = GraphRegularizationLoss()
        self.loss_contrast = ContrastiveLoss()
    
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
                # 对比学习部分
                confusable_pairs: Optional[Tuple] = None,
                sample_weights: Optional[torch.Tensor] = None) -> dict:
        """
        计算多任务联合损失
        
        L_total = L_supervised + λ1·L_align + λ2·L_consistency + λ3·L_geo + λ4·L_graph + λ5·L_contrast
        
        Returns:
            loss_dict: 包含各项损失的字典
        """
        losses = {}
        total_loss = 0.0
        
        # L_supervised: 有标签数据的标准损失
        loss_supervised = F.binary_cross_entropy_with_logits(
            c_heatmap_labeled, y_true
        )
        losses['L_supervised'] = loss_supervised
        total_loss = total_loss + loss_supervised
        
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
            loss_geo = self.loss_geometric(c_heatmap_unlabeled, contour_mask)
            losses['L_geo'] = loss_geo
            total_loss = total_loss + self.lambda3 * loss_geo
        
        # L_graph: 图正则化损失
        if c_graph is not None:
            loss_graph = self.loss_graph(c_heatmap_unlabeled, c_graph)
            losses['L_graph'] = loss_graph
            total_loss = total_loss + self.lambda4 * loss_graph
        
        # L_contrast: 对比损失
        if confusable_pairs is not None:
            loss_contrast = self.loss_contrast(c_heatmap_unlabeled, confusable_pairs)
            losses['L_contrast'] = loss_contrast
            total_loss = total_loss + self.lambda5 * loss_contrast
        
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
