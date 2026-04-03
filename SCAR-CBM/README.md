# 概念级别多模态半监督学习框架

完整实现论文中提出的5阶段框架，用于图像分类和概念学习任务。

## 📋 项目结构

```
.
├── core_modules.py                 # 核心数据结构和工具函数
├── network_architecture.py         # 神经网络模块和模型
├── pseudo_label_and_sampling.py   # 伪标签生成和采样策略
├── loss_functions.py              # 多任务损失函数
├── train_framework.py             # 主训练框架
├── example_usage.py               # 完整使用示例
├── ARCHITECTURE_GUIDE.py          # 架构详细指南
└── README.md                      # 本文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision
pip install numpy scipy scikit-learn networkx
```

### 2. 运行示例

```bash
python example_usage.py
```

### 3. 使用框架进行训练

```python
from train_framework import ConceptualSSLFramework
from network_architecture import CompleteModel

# 创建模型
model = CompleteModel(num_concepts=10)

# 创建框架
framework = ConceptualSSLFramework(
    model=model,
    num_concepts=10,
    num_epochs=100
)

# 训练
history = framework.fit(
    train_labeled_loader=labeled_loader,
    train_unlabeled_loader=unlabeled_loader,
    val_loader=val_loader,
    labeled_concept_ids=concept_ids
)
```

## 🏗️ 框架架构

### 5个主要阶段

#### Stage 1: 概念关系图构建 (离线)
- 从有标签数据提取概念共现模式
- 构建概念超图
- 生成邻接矩阵用于GCN推理

#### Stage 2: 细粒度空间-语义特征提取 (每epoch)
- BaseEncoder: 提取全局和空间特征
- GlobalConceptPredictor: 全局概念预测
- CrossAttentionModule: 对齐到相关区域
- LocalConceptPredictor: 局部概念预测
- FusionGate: 融合全局和局部预测

#### Stage 3: 自适应伪标签生成 (每epoch)
- KNN检索有标签样本的标签
- GMM噪声检测
- 动态混合有标签和模型预测
- 计算可靠性掩码（基于熵）
- 生成最终伪标签

#### Stage 4: 不确定性引导采样
- 计算样本密度（k-NN）
- 计算样本不确定性（熵）
- 反课程采样：优先选择低密度高熵样本
- 实现难样本挖掘

#### Stage 5: 多任务联合优化
多个损失函数：
- **L_supervised**: 有标签数据损失
- **L_align**: 伪标签对齐（仅可靠样本）
- **L_consistency**: 特征一致性（仅不可靠样本）
- **L_geo**: 几何约束，破除背景捷径
- **L_graph**: 概念关系图一致性
- **L_contrast**: 对比损失，拉远混淆概念

总损失: `L = L_sup + 0.5*L_align + 0.3*L_cons + 0.2*L_geo + 0.1*L_graph + 0.1*L_contrast`

## 📊 主要模块说明

### core_modules.py
- **ConceptHypergraph**: 概念超图的构建和管理
- **NegativeSampleQueue**: 负样本队列（对比学习）
- **PseudoLabelHistory**: 伪标签历史追踪
- **LabeledDataset/UnlabeledDataset**: 数据集类
- 辅助函数: 密度计算、熵计算、KNN检索等

### network_architecture.py
- **BaseEncoder**: CNN图像编码器
- **CrossAttentionModule**: 跨注意力机制
- **GlobalConceptPredictor**: 全局概念预测头
- **LocalConceptPredictor**: 局部概念预测头
- **FusionGate**: 全局-局部融合门控
- **GraphConvolutionalNetwork**: GCN推理模块
- **CompleteModel**: 完整模型集成

### pseudo_label_and_sampling.py
- **PseudoLabelGenerator**: 自适应伪标签生成
- **UncertaintyGuidedSampler**: 不确定性采样
- **BatchIntensityAwareSampler**: Batch强度采样
- **CurriculumLearningScheduler**: 课程学习调度

### loss_functions.py
- **AlignmentLoss**: 伪标签对齐损失
- **ConsistencyLoss**: 特征一致性损失
- **GeometricConstraintLoss**: 几何约束损失
- **GraphRegularizationLoss**: 图正则化损失
- **ContrastiveLoss**: 对比损失
- **MultiTaskLoss**: 多任务联合损失
- **DynamicWeightedLoss**: 自动权重调整

### train_framework.py
- **ConceptualSSLFramework**: 完整的训练框架
  - `build_concept_hypergraph()`: Stage 1
  - `extract_features()`: Stage 2
  - `generate_pseudo_labels()`: Stage 3
  - `sample_unlabeled_data()`: Stage 4
  - `train_epoch()` / `fit()`: Stage 5的训练循环

## 🔧 配置参数

### 模型参数
- `input_dim`: 3（RGB图像）
- `feature_dim`: 256（特征维度）
- `num_concepts`: 概念数量（通常5-100）
- `num_heads`: 注意力头数（4-8）

### 训练参数
- `num_epochs`: 训练轮数（50-200）
- `batch_size_labeled`: 有标签batch大小（32-64）
- `batch_size_unlabeled`: 无标签batch大小（64-256）
- `learning_rate`: 学习率（0.0001-0.001）

### 损失权重
- `lambda_align`: 0.5
- `lambda_consistency`: 0.3
- `lambda_geo`: 0.2
- `lambda_graph`: 0.1
- `lambda_contrast`: 0.1

### 采样参数
- `alpha_mix`: 伪标签中混合的比例（0.7）
- `low_density_ratio`: 低密度样本比例（0.2）
- `entropy_threshold`: 熵阈值（自适应）

## 📈 性能优化

### 混合精度训练
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
scaler.scale(loss).backward()
```

### 分布式训练
```python
model = nn.DataParallel(model)  # 单机多卡
# 或
model = nn.parallel.DistributedDataParallel(model)  # 多机多卡
```

### 梯度累积
```python
for i, batch in enumerate(loader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 💡 常见问题

### Q: 如何处理类不平衡？
A: 使用加权采样或样本权重

### Q: 如何选择num_concepts？
A: 根据任务特性，通常为5-100之间

### Q: 伪标签质量差怎么办？
A: 
1. 增加有标签数据
2. 调整alpha_mix（更相信KNN或模型）
3. 增加entropy_threshold
4. 检查数据质量

### Q: 内存不足？
A:
1. 减小batch_size
2. 使用梯度累积
3. 使用混合精度
4. 使用更小的模型

### Q: 如何推理？
```python
framework.model.eval()
with torch.no_grad():
    outputs = framework.extract_features(x, adj_matrix)
    predictions = outputs['c_heatmap']
```

## 📚 相关文献

该框架实现以下技术：
- 半监督学习 (Pseudo-labeling)
- 知识图谱 (Concept Hypergraph)
- 注意力机制 (Cross-Attention)
- 图卷积网络 (GCN)
- 对比学习 (Contrastive Learning)
- 课程学习 (Curriculum Learning)
- 不确定性估计 (Uncertainty Estimation)

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

## 📧 联系方式

如有问题，请通过Issue联系。

---

**最后更新**: 2024年
**版本**: 1.0
**作者**: Framework Implementation Team
