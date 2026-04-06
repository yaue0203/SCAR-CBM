"""
网络架构模块：编码器、预测器、GCN等核心组件
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BaseEncoder(nn.Module):
    """基础视觉编码器（可替换为ResNet、ViT等）"""
    
    def __init__(self, input_dim: int = 3, feature_dim: int = 256, 
                 hidden_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        
        # 简化的CNN编码器结构
        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet-like blocks
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        
        # 全局平均池化后的特征维度
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, feature_dim)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        # 第一个block可能需要改变维度
        first_conv_in = in_channels
        if stride != 1 or in_channels != out_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, 1, stride=stride))
            layers.append(nn.BatchNorm2d(out_channels))
            first_conv_in = out_channels
        
        layers.append(nn.Conv2d(first_conv_in, out_channels, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回两个输出：
        - features: 全局特征 [B, feature_dim]
        - spatial_features: 空间特征用于注意力 [B, C, H, W]
        """
        if x.ndim != 4:
            raise ValueError(f"BaseEncoder 期望输入形状 [B,C,H,W]，得到 ndim={x.ndim}")
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"BaseEncoder 期望通道数 {self.input_dim}，得到 {x.shape[1]}"
            )
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        spatial_features = self.layer3(x)  # 保存空间特征
        
        # 全局特征
        x = self.global_avgpool(spatial_features)
        x = x.flatten(1)
        features = self.fc(x)
        
        return features, spatial_features


class CrossAttentionModule(nn.Module):
    """跨注意力区域聚焦模块"""
    
    def __init__(self, feature_dim: int = 256, num_heads: int = 4, query_dim: int = None):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.query_dim = query_dim if query_dim is not None else feature_dim
        
        assert feature_dim % num_heads == 0
        
        # 投影层
        self.query_proj = nn.Linear(self.query_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, concept_embeddings: torch.Tensor,
                spatial_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            concept_embeddings: 全局概念预测 [B, num_concepts]
            spatial_features: 空间特征 [B, C, H, W]
        
        Returns:
            aligned_features: 对齐后的局部特征 [B, C]
        """
        B, C, H, W = spatial_features.shape
        
        # 展开空间特征
        spatial_features_flat = spatial_features.view(B, C, -1)  # [B, C, HW]
        spatial_features_flat = spatial_features_flat.permute(0, 2, 1)  # [B, HW, C]
        
        # 生成Q, K, V
        Q = self.query_proj(concept_embeddings).unsqueeze(1)  # [B, 1, feature_dim]
        K = self.key_proj(spatial_features_flat)  # [B, HW, feature_dim]
        V = self.value_proj(spatial_features_flat)  # [B, HW, feature_dim]
        
        # 分头
        Q = Q.view(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(B, H*W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(B, H*W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 计算注意力
        scores = (Q @ K.transpose(-2, -1)) * self.scale  # [B, num_heads, 1, HW]
        attn_weights = F.softmax(scores, dim=-1)
        
        # 加权求和
        attn_output = attn_weights @ V  # [B, num_heads, 1, head_dim]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(B, 1, self.feature_dim)
        
        # 输出投影
        output = self.out_proj(attn_output.squeeze(1))  # [B, feature_dim]
        
        return output


class GlobalConceptPredictor(nn.Module):
    """全局概念预测器"""
    
    def __init__(self, input_dim: int = 256, num_concepts: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_concepts)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.fc1(features)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # logits，未激活


class LocalConceptPredictor(nn.Module):
    """局部概念预测器"""
    
    def __init__(self, input_dim: int = 256, num_concepts: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_concepts)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.fc1(features)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # logits，未激活


class FusionGate(nn.Module):
    """全局-局部融合门控"""
    
    def __init__(self, num_concepts: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(num_concepts * 2, 128)
        self.fc2 = nn.Linear(128, num_concepts)
    
    def forward(self, c_global: torch.Tensor, 
                c_local: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c_global: 全局概念预测 [B, num_concepts]
            c_local: 局部概念预测 [B, num_concepts]
        
        Returns:
            gate_weights: 融合权重 [B, num_concepts]
        """
        combined = torch.cat([c_global, c_local], dim=1)
        gate = self.fc1(combined)
        gate = F.relu(gate)
        gate = torch.sigmoid(self.fc2(gate))
        return gate


class GraphConvolutionalNetwork(nn.Module):
    """简化的GCN用于概念关系推理"""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64,
                 output_dim: int = 10, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.layers.append(nn.Linear(hidden_dim, output_dim))
    
    def forward(self, x: torch.Tensor, 
                adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 节点特征，支持：
                - [num_concepts, feature_dim]：标准 GCN，聚合为 A @ X
                - [B, num_concepts]：按样本在概念维上传播，每行为 x @ A^T
                - [B, num_concepts, feature_dim]：批内 A @ X_b
            adj_matrix: 邻接矩阵 [num_concepts, num_concepts]（行/列为概念索引）
        
        Returns:
            与 layer 输出维度一致的张量（最后一层为 output_dim）
        """
        adj_dim = adj_matrix.shape[0]
        for i, layer in enumerate(self.layers):
            if x.ndim == 2 and x.shape[0] == adj_dim:
                # [num_concepts, F] -> A @ X
                x = torch.matmul(adj_matrix, x)
            elif x.ndim == 3 and x.shape[1] == adj_dim:
                # [B, C, F] -> 对每个 batch 做 A @ X
                x = torch.einsum('ij,bjf->bif', adj_matrix, x)
            elif x.ndim == 2 and x.shape[-1] == adj_dim:
                # [B, C]：概念维在最后一维，等价于 (A @ x^T)^T = x @ A^T
                x = torch.matmul(x, adj_matrix.t())
            elif x.shape[-1] == adj_dim:
                raise ValueError(
                    f"GCN 不支持的输入形状 {tuple(x.shape)}（邻接维为 {adj_dim}）"
                )

            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        
        return x


class ConceptHeatmapGenerator(nn.Module):
    """概念热力图生成器（整合全局和局部预测）"""
    
    def __init__(self, num_concepts: int = 10):
        super().__init__()
        self.fusion_gate = FusionGate(num_concepts)
    
    def forward(self, c_global: torch.Tensor, 
                c_local: torch.Tensor) -> torch.Tensor:
        """
        生成最终的概念热力图
        
        Args:
            c_global: 全局概念预测 [B, num_concepts]
            c_local: 局部概念预测 [B, num_concepts]
        
        Returns:
            c_heatmap: 最终概念热力图 [B, num_concepts]
        """
        # 获取融合权重
        g = self.fusion_gate(c_global, c_local)
        
        # 应用sigmoid确保在0-1之间
        c_global_prob = torch.sigmoid(c_global)
        c_local_prob = torch.sigmoid(c_local)
        
        # 加权融合
        c_heatmap = g * c_local_prob + (1 - g) * c_global_prob
        
        return c_heatmap


class CompleteModel(nn.Module):
    """完整模型（集成所有组件）"""
    
    def __init__(self, input_dim: int = 3, feature_dim: int = 256,
                 num_concepts: int = 10, num_heads: int = 4):
        super().__init__()
        
        self.encoder = BaseEncoder(input_dim, feature_dim)
        self.cross_attention = CrossAttentionModule(
            feature_dim=feature_dim,
            num_heads=num_heads,
            query_dim=num_concepts,
        )
        self.predictor_global = GlobalConceptPredictor(feature_dim, num_concepts)
        self.predictor_local = LocalConceptPredictor(feature_dim, num_concepts)
        self.heatmap_generator = ConceptHeatmapGenerator(num_concepts)
        self.gcn = GraphConvolutionalNetwork(num_concepts, 64, num_concepts, 2)
    
    def forward(self, x: torch.Tensor,
                adj_matrix: torch.Tensor = None) -> dict:
        """
        完整前向传播
        
        Args:
            x: 输入图像 [B, C, H, W]
            adj_matrix: 概念关系图的邻接矩阵 [num_concepts, num_concepts]
        
        Returns:
            包含各种中间输出和最终输出的字典
        """
        # Stage 2: 细粒度特征提取
        f_visual, spatial_features = self.encoder(x)
        
        # 全局概念预测
        c_global = self.predictor_global(f_visual)
        
        # 跨注意力对齐
        h_aligned = self.cross_attention(c_global, spatial_features)
        
        # 局部概念预测
        c_local = self.predictor_local(h_aligned)
        
        # 概念热力图
        c_heatmap = self.heatmap_generator(c_global, c_local)
        
        # GCN推理（如果提供邻接矩阵）
        c_graph = None
        if adj_matrix is not None:
            c_graph = self.gcn(c_heatmap, adj_matrix)
        
        return {
            'f_visual': f_visual,
            'spatial_features': spatial_features,
            'c_global': c_global,
            'h_aligned': h_aligned,
            'c_local': c_local,
            'c_heatmap': c_heatmap,
            'c_graph': c_graph
        }
