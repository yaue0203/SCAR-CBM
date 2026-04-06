"""
伪标签生成与采样模块：实现Stage 3和Stage 4的核心算法
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
from scipy.stats import entropy

from core_modules import gaussian_mixture_noise_detection


class PseudoLabelGenerator:
    """自适应伪标签生成器"""
    
    def __init__(self, num_concepts: int, 
                 alpha: float = 0.7,
                 entropy_threshold_schedule=None):
        self.num_concepts = num_concepts
        self.alpha = alpha
        self.entropy_threshold_schedule = entropy_threshold_schedule or {}
        self.epoch = 0
    
    def generate(self, 
                 c_heatmap: torch.Tensor,
                 c_knn: torch.Tensor,
                 contour_mask: torch.Tensor = None,
                 pseudo_label_history = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage 3: 自适应伪标签生成
        
        Args:
            c_heatmap: 模型预测的热力图 [B, num_concepts]
            c_knn: KNN检索的标签 [B, num_concepts]
            contour_mask: 物理轮廓掩码 [B, H, W]
            pseudo_label_history: 伪标签历史记录
        
        Returns:
            c_pseudo: 最终伪标签 [B, num_concepts]
            m_rel: 可靠性掩码 [B]
        """
        batch_size = c_heatmap.shape[0]
        device = c_heatmap.device
        
        # 步骤1: GMM噪声检测
        heatmap_diff = torch.abs(c_heatmap - c_knn)  # [B, num_concepts]
        mean_diff = heatmap_diff.mean(dim=1)  # [B]
        
        # 与 core_modules.gaussian_mixture_noise_detection 一致：1=保留，0=滤除
        m_noise = self._gmm_noise_detection(mean_diff)  # [B]
        
        # 步骤2: 动态混合
        c_mixed = self.alpha * c_heatmap + (1 - self.alpha) * c_knn  # [B, num_concepts]
        
        # 步骤3: 历史验证
        if pseudo_label_history is not None:
            c_accepted = self._verify_consistency(c_mixed, pseudo_label_history)
        else:
            c_accepted = c_mixed
        
        # 步骤4: 动态可靠性掩码（基于熵）
        m_rel = self._compute_reliability_mask(c_accepted, device)  # [B]
        
        # 步骤5: 噪声过滤
        c_pseudo = m_noise.unsqueeze(1) * c_accepted  # [B, num_concepts]
        
        return c_pseudo, m_rel
    
    def _gmm_noise_detection(self, differences: torch.Tensor) -> torch.Tensor:
        """GMM 噪声过滤掩码（与 ``core_modules.gaussian_mixture_noise_detection`` 共享逻辑）。"""
        return gaussian_mixture_noise_detection(differences, n_components=2)
    
    def _verify_consistency(self, current_labels: torch.Tensor,
                           history) -> torch.Tensor:
        """与历史伪标签验证一致性"""
        # 这里简化实现，实际应该根据具体的history结构调整
        return current_labels
    
    def _compute_reliability_mask(self, 
                                 labels: torch.Tensor,
                                 device: torch.device) -> torch.Tensor:
        """
        计算可靠性掩码（基于信息熵）
        
        H(c) <= τ_batch 时，mask = 1（可靠）
        H(c) > τ_batch 时，mask = 0（不可靠）
        """
        # 转换为概率分布
        probs = torch.sigmoid(labels)
        
        # 计算熵
        entropy_vals = -torch.sum(
            probs * torch.log(probs + 1e-8) + 
            (1 - probs) * torch.log(1 - probs + 1e-8),
            dim=1
        )  # [B]
        
        # 动态阈值：使用batch内的中位数
        tau_batch = torch.median(entropy_vals)
        
        # 可靠性掩码
        m_rel = (entropy_vals <= tau_batch).float()
        
        return m_rel


class UncertaintyGuidedSampler:
    """不确定性引导的反课程采样器"""
    
    def __init__(self, 
                 low_density_ratio: float = 0.2,
                 uncertainty_weight: float = 1.0):
        """
        Args:
            low_density_ratio: 低密度样本的比例 (τ_low)
            uncertainty_weight: 不确定性权重系数
        """
        self.low_density_ratio = low_density_ratio
        self.uncertainty_weight = uncertainty_weight
    
    def sample(self, 
               features: torch.Tensor,
               logits: torch.Tensor,
               batch_size: int) -> np.ndarray:
        """
        Stage 4: 不确定性引导反课程采样
        
        Args:
            features: 样本特征 [N, feature_dim]
            logits: 模型输出 [N, num_concepts]
            batch_size: 采样的batch大小
        
        Returns:
            sampled_indices: 采样的样本索引
        """
        N = features.shape[0]
        device = features.device
        
        # 计算密度：使用k-NN
        densities = self._compute_density(features)  # [N]
        
        # 计算不确定性（熵）
        uncertainties = self._compute_uncertainty(logits)  # [N]
        
        # 反课程权重：优先选择低密度高不确定性的样本
        weights = (1.0 / (densities + 1e-8)) * uncertainties  # [N]
        
        # 归一化权重
        weights = weights / (weights.sum() + 1e-8)
        
        # 采样（float32→numpy 后概率和可能略偏离 1，np.random.choice 会报错）
        probs = np.asarray(weights.detach().cpu().numpy(), dtype=np.float64)
        probs = np.clip(probs, 0.0, None)
        s = float(probs.sum())
        if not np.isfinite(s) or s <= 0:
            probs = np.full(N, 1.0 / N, dtype=np.float64)
        else:
            probs /= s
        sampled_indices = np.random.choice(
            N, size=batch_size, p=probs, replace=True
        )
        
        return sampled_indices
    
    def _compute_density(self, features: torch.Tensor, k: int = 5) -> torch.Tensor:
        """计算k-NN密度"""
        distances = torch.cdist(features, features)
        
        # 获取k+1个最近邻（包括自身）
        knn_dists, _ = torch.topk(distances, k=k+1, dim=1, largest=False)
        
        # 使用第k个邻居的距离
        density = 1.0 / (knn_dists[:, k] + 1e-8)
        
        return density
    
    def _compute_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        """计算样本不确定性（熵）"""
        probs = torch.sigmoid(logits)
        
        # 计算信息熵
        entropy = -(
            probs * torch.log(probs + 1e-8) +
            (1 - probs) * torch.log(1 - probs + 1e-8)
        ).sum(dim=1)
        
        # 不确定性权重
        uncertainty_weights = 1.0 / (entropy + 1e-8)
        
        return uncertainty_weights


class BatchIntensityAwareSampler:
    """Batch强度感知采样器（课程学习变体）"""
    
    def __init__(self, 
                 initial_sample_rate: float = 0.1,
                 final_sample_rate: float = 1.0,
                 total_epochs: int = 100):
        """
        Args:
            initial_sample_rate: 初始采样比例
            final_sample_rate: 最终采样比例
            total_epochs: 总epoch数（用于线性增长）
        """
        self.initial_sample_rate = initial_sample_rate
        self.final_sample_rate = final_sample_rate
        self.total_epochs = total_epochs
        self.current_epoch = 0
    
    def get_sample_rate(self) -> float:
        """获取当前采样比例（线性增长）"""
        progress = self.current_epoch / self.total_epochs
        sample_rate = (
            self.initial_sample_rate + 
            progress * (self.final_sample_rate - self.initial_sample_rate)
        )
        return min(sample_rate, self.final_sample_rate)
    
    def sample(self, 
               sample_scores: torch.Tensor,
               batch_size: int) -> np.ndarray:
        """
        基于样本得分的采样
        
        Args:
            sample_scores: 每个样本的得分 [N]
            batch_size: 要采样的大小
        
        Returns:
            sampled_indices: 采样的索引
        """
        N = sample_scores.shape[0]
        sample_rate = self.get_sample_rate()
        
        # 根据当前采样比例，计算要采样多少样本
        num_samples = max(1, int(N * sample_rate))
        
        # 获取top-K样本
        _, top_indices = torch.topk(
            sample_scores, 
            k=min(num_samples, N),
            dim=0
        )
        
        # 从top-K中随机采样batch_size个
        sampled_indices = top_indices[
            np.random.choice(len(top_indices), batch_size, replace=True)
        ].cpu().numpy()
        
        return sampled_indices
    
    def step_epoch(self):
        """更新epoch计数"""
        self.current_epoch += 1


class TripleLossSampler:
    """三元损失采样（用于对比学习中的负样本采样）"""
    
    @staticmethod
    def hard_negative_sampling(embeddings: torch.Tensor,
                               labels: torch.Tensor,
                               margin: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        硬负样本采样
        
        Args:
            embeddings: 特征向量 [B, feature_dim]
            labels: 标签 [B]
            margin: 三元损失的边际
        
        Returns:
            positive_indices: 正样本索引
            negative_indices: 负样本索引
        """
        # 计算相似度矩阵
        similarities = torch.mm(embeddings, embeddings.t())  # [B, B]
        
        # 同标签的为正样本，不同标签的为负样本
        same_label = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        different_label = 1.0 - same_label
        
        # 对角线设为0（自身不是正样本）
        same_label.fill_diagonal_(0)
        
        # 选择最难的正样本（相似度最低）
        same_label_mask = same_label > 0
        similarities_masked = similarities.clone()
        similarities_masked[~same_label_mask] = float('inf')
        
        hard_positive_sim, hard_positive_idx = similarities_masked.min(dim=1)
        
        # 选择最难的负样本（相似度最高）
        different_label_mask = different_label > 0
        similarities_masked = similarities.clone()
        similarities_masked[~different_label_mask] = float('-inf')
        
        hard_negative_sim, hard_negative_idx = similarities_masked.max(dim=1)
        
        return hard_positive_idx, hard_negative_idx


class CurriculumLearningScheduler:
    """课程学习调度器"""
    
    def __init__(self, 
                 strategy: str = 'easy_to_hard',
                 total_epochs: int = 100):
        """
        Args:
            strategy: 'easy_to_hard' 或 'hard_to_easy'
            total_epochs: 总epoch数
        """
        self.strategy = strategy
        self.total_epochs = total_epochs
        self.current_epoch = 0
    
    def get_difficulty_threshold(self) -> float:
        """获取当前难度阈值"""
        progress = self.current_epoch / self.total_epochs
        
        if self.strategy == 'easy_to_hard':
            # 从容易到困难：阈值从高到低
            threshold = 1.0 - 0.9 * progress
        else:  # hard_to_easy
            # 从困难到容易：阈值从低到高
            threshold = 0.1 + 0.9 * progress
        
        return threshold
    
    def filter_by_difficulty(self,
                            sample_scores: torch.Tensor,
                            threshold: float = None) -> np.ndarray:
        """
        根据难度过滤样本
        
        Args:
            sample_scores: 样本得分 [N]
            threshold: 难度阈值
        
        Returns:
            filtered_indices: 过滤后的索引
        """
        if threshold is None:
            threshold = self.get_difficulty_threshold()
        
        if self.strategy == 'easy_to_hard':
            # 选择得分 >= 阈值的样本（容易的样本）
            mask = sample_scores >= threshold
        else:
            # 选择得分 <= 阈值的样本（困难的样本）
            mask = sample_scores <= threshold
        
        indices = torch.where(mask)[0].cpu().numpy()
        
        return indices if len(indices) > 0 else np.array([0])
    
    def step_epoch(self):
        """更新epoch计数"""
        self.current_epoch += 1
