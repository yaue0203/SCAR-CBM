"""
完整使用示例和实战指南
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import json

# 导入框架模块
from train_framework import ConceptualSSLFramework, create_example_data
from network_architecture import CompleteModel
from core_modules import LabeledDataset, UnlabeledDataset


class Config:
    """配置类"""
    
    # 模型参数
    input_dim = 3
    feature_dim = 256
    num_concepts = 10
    num_heads = 4
    hidden_dim = 512
    
    # 训练参数
    num_epochs = 100
    batch_size_labeled = 32
    batch_size_unlabeled = 64
    learning_rate = 0.001
    weight_decay = 1e-5
    
    # 损失权重
    lambda_align = 0.5
    lambda_consistency = 0.3
    lambda_geo = 0.2
    lambda_graph = 0.1
    lambda_contrast = 0.1
    
    # 采样参数
    low_density_ratio = 0.2
    alpha_mix = 0.7
    entropy_threshold = 0.5
    
    # 设备和日志
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 42
    save_dir = './checkpoints'


def set_seed(seed: int):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def create_data_loaders(data_dict: dict,
                       config: Config,
                       val_split: float = 0.2) -> tuple:
    """
    创建数据加载器
    
    Args:
        data_dict: 包含数据的字典
        config: 配置对象
        val_split: 验证集划分比例
    
    Returns:
        (train_labeled_loader, train_unlabeled_loader, val_loader)
    """
    X_labeled = data_dict['X_labeled']
    y_labeled = data_dict['y_labeled']
    c_labeled = data_dict['c_labeled']
    X_unlabeled = data_dict['X_unlabeled']
    
    # 划分有标签数据为训练和验证
    num_labeled = len(X_labeled)
    num_val = int(num_labeled * val_split)
    
    indices = np.random.permutation(num_labeled)
    train_indices = indices[num_val:]
    val_indices = indices[:num_val]
    
    X_labeled_train = X_labeled[train_indices]
    y_labeled_train = y_labeled[train_indices]
    c_labeled_train = c_labeled[train_indices]
    
    X_labeled_val = X_labeled[val_indices]
    y_labeled_val = y_labeled[val_indices]
    c_labeled_val = c_labeled[val_indices]
    
    # 创建数据集
    train_labeled_dataset = LabeledDataset(
        X_labeled_train, y_labeled_train, c_labeled_train
    )
    
    train_unlabeled_dataset = UnlabeledDataset(
        X_unlabeled
    )
    
    val_dataset = LabeledDataset(
        X_labeled_val, y_labeled_val, c_labeled_val
    )
    
    # 创建加载器
    train_labeled_loader = DataLoader(
        train_labeled_dataset,
        batch_size=config.batch_size_labeled,
        shuffle=True,
        num_workers=2
    )
    
    train_unlabeled_loader = DataLoader(
        train_unlabeled_dataset,
        batch_size=config.batch_size_unlabeled,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size_labeled,
        shuffle=False,
        num_workers=2
    )
    
    return train_labeled_loader, train_unlabeled_loader, val_loader


def main():
    """主函数"""
    
    # 设置配置
    config = Config()
    set_seed(config.seed)
    
    print("="*60)
    print("概念级别的多模态半监督学习框架")
    print("="*60)
    
    # 第1步：创建模型
    print("\n[Step 1] 创建模型...")
    model = CompleteModel(
        input_dim=config.input_dim,
        feature_dim=config.feature_dim,
        num_concepts=config.num_concepts,
        num_heads=config.num_heads
    )
    print(f"  - 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 第2步：创建数据
    print("\n[Step 2] 创建训练数据...")
    data_dict = create_example_data(
        num_labeled=100,
        num_unlabeled=400,
        num_concepts=config.num_concepts,
        image_size=32
    )
    print(f"  - 有标签数据: {data_dict['X_labeled'].shape}")
    print(f"  - 无标签数据: {data_dict['X_unlabeled'].shape}")
    print(f"  - 物理掩码: {data_dict['contour_mask'].shape}")
    
    # 第3步：创建数据加载器
    print("\n[Step 3] 创建数据加载器...")
    train_labeled_loader, train_unlabeled_loader, val_loader = create_data_loaders(
        data_dict, config
    )
    print(f"  - 有标签训练样本: {len(train_labeled_loader.dataset)}")
    print(f"  - 无标签训练样本: {len(train_unlabeled_loader.dataset)}")
    print(f"  - 验证样本: {len(val_loader.dataset)}")
    
    # 第4步：创建框架和优化器
    print("\n[Step 4] 初始化训练框架...")
    framework = ConceptualSSLFramework(
        model=model,
        device=config.device,
        num_concepts=config.num_concepts,
        num_epochs=config.num_epochs,
        batch_size_labeled=config.batch_size_labeled,
        batch_size_unlabeled=config.batch_size_unlabeled,
        learning_rate=config.learning_rate
    )
    print(f"  - 设备: {config.device}")
    print(f"  - 学习率: {config.learning_rate}")
    print(f"  - Epochs: {config.num_epochs}")
    
    # 第5步：训练
    print("\n[Step 5] 开始训练...")
    print("-"*60)
    
    training_history = framework.fit(
        train_labeled_loader=train_labeled_loader,
        train_unlabeled_loader=train_unlabeled_loader,
        val_loader=val_loader,
        labeled_concept_ids=data_dict['c_labeled'],
        contour_mask=data_dict['contour_mask']
    )
    
    print("-"*60)
    print("\n[完成] 训练结束！")
    
    # 打印最终结果
    print("\n训练历史摘要:")
    for key in sorted(training_history.keys()):
        values = training_history[key]
        if len(values) > 0:
            print(f"  {key}:")
            print(f"    - 初始: {values[0]:.4f}")
            print(f"    - 最终: {values[-1]:.4f}")
            print(f"    - 最优: {min(values):.4f}")
    
    return framework, training_history


# ============================================================
# 实战指南和最佳实践
# ============================================================

BEST_PRACTICES = """
╔════════════════════════════════════════════════════════════════╗
║           概念级别半监督学习框架 - 实战指南                  ║
╚════════════════════════════════════════════════════════════════╝

### 1. 数据准备
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ 有标签数据格式:
  - 图像: [N, C, H, W] (float32, 0-1 or -1-1)
  - 标签: [N, num_concepts] (float32, 0-1范围)
  - 概念ID: [N, num_concepts] (float32, 用于构建超图)

✓ 无标签数据格式:
  - 图像: [M, C, H, W] (float32)
  - 通常M >> N（无标签数据应远多于有标签）

✓ 物理轮廓掩码:
  - 形状: [1, 1, H, W]
  - 1表示背景，0表示前景
  - 用于破除背景捷径


### 2. Stage 1: 概念关系图构建
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

这个阶段是离线的，只在开始时执行一次:

  framework.build_concept_hypergraph(labeled_concept_ids)

关键参数:
  - cooccurrence_threshold: 共现频率阈值（默认0.1）
  - 更大的阈值 → 更稀疏的图
  - 更小的阈值 → 更密集的图


### 3. Stage 2: 细粒度特征提取
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

关键组件:
  1. BaseEncoder: 提取全局视觉特征和空间特征
  2. GlobalConceptPredictor: 基于全局特征的概念预测
  3. CrossAttentionModule: 聚焦到相关的空间区域
  4. LocalConceptPredictor: 基于局部特征的概念预测
  5. FusionGate: 融合全局和局部预测

优化建议:
  - 增加attention head数量以捕捉更多细节
  - 使用不同大小的CNN层获取多尺度特征
  - 考虑使用Vision Transformer替代CNN


### 4. Stage 3: 自适应伪标签生成
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

关键参数:
  
  alpha_mix (default: 0.7)
    伪标签 = 0.7 * heatmap + 0.3 * KNN
    - 更高的alpha: 更相信模型预测
    - 更低的alpha: 更相信KNN检索

  entropy_threshold
    使用信息熵判断伪标签可靠性
    H(c) <= τ_batch → 可靠（m_rel = 1）
    H(c) > τ_batch → 不可靠（m_rel = 0）

GMM噪声检测:
  - 自动识别异常的KNN检索结果
  - 剔除似然低的样本


### 5. Stage 4: 不确定性采样
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

反课程策略（Easy to Hard）:
  
  w(x) = (1 / density(x)) * entropy(x)
  
  - 优先选择: 低密度 + 高熵 的样本
  - 这些是模型不确定的、具有挑战的样本
  - 类似于困难样本挖掘

参数调整:
  - low_density_ratio: 低密度样本的比例
  - 更高的比例 → 更容易的样本
  - 更低的比例 → 更困难的样本


### 6. Stage 5: 多任务联合优化
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

L_total = L_supervised + λ1·L_align + λ2·L_consistency + λ3·L_geo + λ4·L_graph + λ5·L_contrast

各损失项说明:

  L_supervised (supervised loss)
    - 仅在有标签数据上
    - 标准BCE损失

  L_align (alignment loss)
    - 伪标签与模型预测的对齐
    - 仅在可靠样本上（m_rel = 1）
    - 权重: λ1 = 0.5 (推荐)

  L_consistency (consistency loss)
    - 弱增强与强增强特征的一致性
    - 仅在不可靠样本上（m_rel = 0）
    - 权重: λ2 = 0.3 (推荐)

  L_geo (geometric constraint loss)
    - 破除背景捷径
    - 背景区域的热力图应该为0
    - 权重: λ3 = 0.2 (推荐)

  L_graph (graph regularization loss)
    - GCN推理与直接预测的一致性
    - 权重: λ4 = 0.1 (推荐)

  L_contrast (contrastive loss)
    - 拉远易混淆的概念
    - 权重: λ5 = 0.1 (推荐)

权重调整策略:
  - 初期: 强调L_align，弱化其他
  - 后期: 逐步增加L_graph和L_contrast权重
  - 使用DynamicWeightedLoss自动调整


### 7. 超参数调优
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

学习率策略:
  - 初始LR: 0.001（推荐）
  - 调度器: CosineAnnealing（效果好）
  - Warmup: 前5% epochs增暖

批大小:
  - 有标签batch: 32-64
  - 无标签batch: 64-256
  - 无标签应该大于有标签

增强策略:
  - 弱增强: 随机裁剪、旋转
  - 强增强: RandAugment、Mixup、CutMix

其他:
  - dropout: 0.3
  - weight decay: 1e-5
  - 梯度裁剪: max_norm=1.0 (可选)


### 8. 监控和调试
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

关键指标:

  训练损失:
    - L_supervised应该单调递减
    - L_align在可靠样本上应该降低
    - L_consistency在不可靠样本上应该降低

  伪标签质量:
    - m_rel的平均值 (应该在0.4-0.6)
    - 噪声检测的覆盖率
    - 与历史标签的一致性

  特征质量:
    - 密度分布 (应该有较好的多样性)
    - 熵值分布 (应该逐步降低)
    - 簇内距离 vs 簇间距离比例

调试建议:
  ✗ 损失不下降
    → 检查学习率是否过小
    → 检查数据是否有问题
    → 尝试调整loss权重

  ✗ 过拟合
    → 增加dropout
    → 增加正则化权重
    → 增加强增强比例

  ✗ 伪标签质量差
    → 增加alpha_mix中KNN的权重
    → 调整entropy_threshold
    → 增加有标签数据


### 9. 推理和评估
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

推理模式:
  framework.model.eval()
  with torch.no_grad():
      outputs = framework.extract_features(x, adj_matrix)
      predictions = outputs['c_heatmap']

常用输出:
  - c_heatmap: 最终概念概率 [B, num_concepts]
  - c_global: 全局预测 [B, num_concepts]
  - c_local: 局部预测 [B, num_concepts]
  - c_graph: GCN推理结果 [B, num_concepts]
  - h_aligned: 对齐后的局部特征 [B, feature_dim]

评估指标:
  - mAP (mean Average Precision)
  - F1-score (per-concept)
  - 混淆矩阵
  - ROC-AUC


### 10. 常见问题解决
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: 如何处理不平衡的数据？
A: 使用样本权重或加权采样器

Q: 如何扩展到大规模数据？
A: 使用分布式训练、混合精度、数据并行

Q: 如何选择num_concepts？
A: 根据任务特性，通常为5-100之间

Q: 如何处理域迁移？
A: 增加对比损失权重，使用域自适应技术

Q: 内存不足怎么办？
A: 减小batch_size、使用梯度累积、使用更小的模型

"""

print(BEST_PRACTICES)


if __name__ == '__main__':
    framework, history = main()
    
    print("\n✓ 框架已准备好！")
    print("  - 模型已保存")
    print("  - 训练历史已记录")
    print("  - 可以进行推理了")
