"""
核心模块：数据结构、工具类和基础操作
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import networkx as nx
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
import math


class ConceptHypergraph:
    """概念关系超图构建与管理"""
    
    def __init__(self, num_concepts: int):
        self.num_concepts = num_concepts
        self.hypergraph = nx.Graph()
        self.concept_embeddings = None
        self.cooccurrence_matrix = None
        
    def build_from_labeled_data(self, labels: np.ndarray, 
                                concept_ids: np.ndarray,
                                cooccurrence_threshold: float = 0.1):
        """
        从有标签数据提取概念共现模式
        
        Args:
            labels: 样本标签 [N]
            concept_ids: 样本对应的概念ID [N, num_concepts]
            cooccurrence_threshold: 共现频率阈值
        """
        # 计算概念共现矩阵
        self.cooccurrence_matrix = concept_ids.T @ concept_ids  # [num_concepts, num_concepts]
        
        # 归一化
        row_sum = self.cooccurrence_matrix.sum(axis=1, keepdims=True) + 1e-8
        cooccurrence_norm = self.cooccurrence_matrix / row_sum
        
        # 构建超图边
        for i in range(self.num_concepts):
            self.hypergraph.add_node(i)
            for j in range(i+1, self.num_concepts):
                weight = cooccurrence_norm[i, j]
                if weight > cooccurrence_threshold:
                    self.hypergraph.add_edge(i, j, weight=weight)
    
    def get_adjacency_matrix(self, device='cpu'):
        """获取邻接矩阵用于GCN"""
        adj = nx.to_numpy_array(self.hypergraph, nodelist=range(self.num_concepts))
        # 添加自环
        adj = adj + np.eye(self.num_concepts)
        # 度数归一化
        degree = np.sum(adj, axis=1)
        degree_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-8))
        adj_norm = degree_inv_sqrt @ adj @ degree_inv_sqrt
        
        return torch.FloatTensor(adj_norm).to(device)


class NegativeSampleQueue:
    """负样本特征队列（用于对比学习）"""
    
    def __init__(self, queue_size: int = 65536, feature_dim: int = 256):
        self.queue_size = queue_size
        self.feature_dim = feature_dim
        self.queue = deque(maxlen=queue_size)
        
    def update(self, features: torch.Tensor):
        """更新队列"""
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        for feat in features:
            self.queue.append(feat.detach().cpu())
    
    def get_negatives(self, k: int = 256) -> Optional[torch.Tensor]:
        """随机采样负样本"""
        if k <= 0 or len(self.queue) < k:
            return None
        
        indices = np.random.choice(len(self.queue), k, replace=False)
        negatives = torch.stack([self.queue[i] for i in indices])
        return negatives


class PseudoLabelHistory:
    """伪标签历史管理（用于一致性验证）"""
    
    def __init__(self, num_samples: int, memory_size: int = 3):
        self.num_samples = num_samples
        self.memory_size = memory_size
        self.history = [deque(maxlen=memory_size) for _ in range(num_samples)]
        
    def update(self, sample_ids: np.ndarray, pseudo_labels: torch.Tensor):
        """记录伪标签"""
        for sid, label in zip(sample_ids, pseudo_labels):
            self.history[sid].append(label.detach().cpu())
    
    def get_consistency(self, sample_ids: np.ndarray, 
                       new_labels: torch.Tensor) -> torch.Tensor:
        """计算与历史标签的一致性"""
        consistency = torch.ones(
            len(sample_ids), device=new_labels.device, dtype=new_labels.dtype
        )
        
        for idx, sid in enumerate(sample_ids):
            if len(self.history[sid]) > 0:
                # 计算余弦相似度
                past_labels = torch.stack(list(self.history[sid])).to(
                    device=new_labels.device, dtype=new_labels.dtype
                )
                curr_label = new_labels[idx]
                sim = F.cosine_similarity(
                    curr_label.unsqueeze(0),
                    past_labels
                ).mean()
                consistency[idx] = sim
        
        return consistency

    def get_history_mean(
        self, sample_ids: np.ndarray, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """返回每个样本的历史伪标签均值；无历史则返回 0。"""
        if len(sample_ids) == 0:
            return torch.empty(0, device=device, dtype=dtype)

        feature_dim = None
        for sid in sample_ids:
            if len(self.history[sid]) > 0:
                feature_dim = int(self.history[sid][0].numel())
                break
        if feature_dim is None:
            raise ValueError("get_history_mean 需要至少一个存在历史记录的样本")

        out = torch.zeros(len(sample_ids), feature_dim, device=device, dtype=dtype)
        for idx, sid in enumerate(sample_ids):
            if len(self.history[sid]) == 0:
                continue
            past_labels = torch.stack(list(self.history[sid])).to(device=device, dtype=dtype)
            out[idx] = past_labels.mean(dim=0)
        return out


def _to_tensor(sample):
    if isinstance(sample, torch.Tensor):
        return sample.detach().clone().float()
    arr = np.asarray(sample)
    return torch.from_numpy(np.ascontiguousarray(arr)).float()


class UnlabeledDataset(Dataset):
    """无标签数据集"""
    
    def __init__(self, data: np.ndarray, transform=None, strong_transform=None,
                 contour_masks: Optional[np.ndarray] = None):
        self.data = data
        self.transform = transform
        self.strong_transform = strong_transform
        self.contour_masks = contour_masks
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = _to_tensor(self.data[idx])
        base = x.clone()

        if self.transform:
            x_weak = self.transform(base.clone())
        else:
            x_weak = base.clone()
        
        if self.strong_transform:
            x_strong = self.strong_transform(base.clone())
        else:
            x_strong = base.clone()

        if self.contour_masks is None:
            return x_weak, x_strong, idx

        contour_mask = _to_tensor(self.contour_masks[idx])
        return x_weak, x_strong, idx, contour_mask


class LabeledDataset(Dataset):
    """有标签数据集"""
    
    def __init__(self, images: np.ndarray, labels: np.ndarray, 
                 concept_ids: np.ndarray, transform=None,
                 contour_masks: Optional[np.ndarray] = None):
        self.images = images
        self.labels = labels
        self.concept_ids = concept_ids
        self.transform = transform
        self.contour_masks = contour_masks
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        x = _to_tensor(self.images[idx])
        if self.transform:
            x = self.transform(x)

        y = _to_tensor(self.labels[idx])
        c = _to_tensor(self.concept_ids[idx])
        if self.contour_masks is None:
            return x, y, c, idx

        contour_mask = _to_tensor(self.contour_masks[idx])
        return x, y, c, idx, contour_mask


# ============ 辅助函数 ============

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    多标签概念预测的不确定性：各概念独立二值熵之和（按样本）。

    输入为 **logits**（与 BCEWithLogits、c_heatmap 一致）。勿对概念维做 softmax（会误当成互斥多类）。
    """
    probs = torch.sigmoid(logits)
    ent = -(
        probs * torch.log(probs + 1e-8)
        + (1.0 - probs) * torch.log(1.0 - probs + 1e-8)
    )
    return ent.sum(dim=-1)


def compute_density(features: torch.Tensor, k: int = 5) -> torch.Tensor:
    """计算样本的密度（基于k-NN）"""
    # 计算特征的L2距离
    distances = torch.cdist(features, features)
    # 获取k最近邻的距离
    knn_dist, _ = torch.topk(distances, k=k+1, dim=1, largest=False)
    # 取最后一个（第k个邻居的距离）
    density = 1.0 / (knn_dist[:, k] + 1e-8)
    return density


def apply_contour_mask(heatmap: torch.Tensor, 
                       contour_mask: torch.Tensor) -> torch.Tensor:
    """应用物理轮廓掩码破除背景捷径"""
    # contour_mask: 1表示背景，0表示前景
    masked_heatmap = heatmap * (1 - contour_mask)
    return masked_heatmap


def knn_retrieval(query_features: torch.Tensor,
                  db_features: torch.Tensor,
                  db_labels: torch.Tensor,
                  k: int = 5) -> torch.Tensor:
    """KNN检索最近的有标签样本"""
    # 计算距离
    distances = torch.cdist(query_features, db_features)
    
    # 获取k最近邻
    _, knn_indices = torch.topk(distances, k=k, dim=1, largest=False)
    
    # 获取对应的标签（取平均）
    knn_labels = db_labels[knn_indices]  # [batch_size, k, num_concepts]
    retrieved_labels = knn_labels.mean(dim=1)  # [batch_size, num_concepts]
    
    return retrieved_labels


def gaussian_mixture_noise_detection(values: torch.Tensor, 
                                     n_components: int = 2) -> torch.Tensor:
    """
    基于 GMM 对数似然的样本筛选。

    返回与 ``values`` 逐元素对齐的掩码：高似然（被 GMM 较好解释）为 1，
    低似然（离群/噪声倾向）为 0。与 ``PseudoLabelGenerator`` 中用法一致（1 表示保留）。
    """
    values_np = values.detach().cpu().numpy().reshape(-1, 1)

    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(values_np)

    ll = gmm.score_samples(values_np)
    threshold = np.percentile(ll, 30)
    keep_mask = (ll >= threshold).astype(np.float32)

    return torch.from_numpy(keep_mask).to(device=values.device, dtype=torch.float32)
