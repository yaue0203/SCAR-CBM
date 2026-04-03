# 概念级别多模态半监督学习框架 - 完整实现指南

## 📋 项目总览

这是对论文中提出的**概念级别的多模态半监督学习框架**的完整实现，包含所有5个阶段的算法和相关支撑模块。

### 项目特点
- ✅ **完整实现**：论文中所有5个阶段都已实现
- ✅ **生产级代码**：模块化设计、完整文档、错误处理
- ✅ **易于使用**：提供配置类、示例代码和详细说明
- ✅ **高度可扩展**：支持自定义编码器、损失函数、采样策略
- ✅ **性能优化**：支持混合精度、分布式训练、梯度累积

---

## 📦 文件列表

### 核心实现文件（6个）

| 文件名 | 行数 | 功能 | 关键类 |
|-------|------|------|-------|
| **core_modules.py** | 800 | 核心数据结构和工具 | ConceptHypergraph, NegativeSampleQueue, PseudoLabelHistory |
| **network_architecture.py** | 600 | 神经网络模块 | BaseEncoder, CrossAttentionModule, CompleteModel |
| **pseudo_label_and_sampling.py** | 600 | 伪标签和采样 | PseudoLabelGenerator, UncertaintyGuidedSampler |
| **loss_functions.py** | 500 | 所有损失函数 | MultiTaskLoss, AlignmentLoss, ConsistencyLoss等 |
| **train_framework.py** | 700 | 训练框架 | ConceptualSSLFramework |
| **example_usage.py** | 400 | 使用示例 | Config, main() |

### 文档文件（4个）

| 文件名 | 内容 |
|-------|------|
| **README.md** | 快速开始、框架说明、常见问题 |
| **ARCHITECTURE_GUIDE.py** | 详细的架构说明和优化建议 |
| **IMPLEMENTATION_SUMMARY.py** | 论文伪代码与实现的对应表 |
| **FILE_INDEX.py** | 完整的文件索引和快速查找 |

### 依赖文件

| 文件名 | 内容 |
|-------|------|
| **requirements.txt** | Python依赖列表 |

**总代码量**：~3200行核心代码 + ~2500行文档

---

## 🚀 快速开始（5分钟）

### 1. 安装依赖
```bash
pip install -r requirements.txt
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

---

## 🏗️ 框架的5个阶段

### Stage 1: 概念关系图构建（离线）

**目标**：从有标签数据中提取概念关系

**实现**：
```python
# core_modules.py: ConceptHypergraph
framework.build_concept_hypergraph(labeled_concept_ids)
adj_matrix = framework.hypergraph.get_adjacency_matrix()
```

**工作原理**：
1. 计算概念共现矩阵
2. 构建概念超图（边权重 = 共现频率）
3. 生成邻接矩阵用于GCN推理

---

### Stage 2: 细粒度空间-语义特征提取（每epoch）

**目标**：提取融合全局和局部信息的概念特征

**实现**：
```python
# network_architecture.py: CompleteModel
outputs = framework.extract_features(x, adj_matrix)
```

**数据流**：
```
输入图像 x [B, 3, H, W]
    ↓
BaseEncoder
    ├→ f_visual: [B, 256]           # 全局特征
    └→ spatial_features: [B, C, H, W]  # 空间特征
        ├→ GlobalConceptPredictor → c_global: [B, 10]
        └→ CrossAttentionModule
            └→ h_aligned: [B, 256]
                └→ LocalConceptPredictor → c_local: [B, 10]
    ├→ FusionGate(c_global, c_local) → g: [B, 10]
    └→ c_heatmap = g * sigmoid(c_local) + (1-g) * sigmoid(c_global)
        └→ c_heatmap: [B, 10]
```

---

### Stage 3: 自适应伪标签生成（每epoch）

**目标**：为无标签数据生成高质量的伪标签

**实现**：
```python
# pseudo_label_and_sampling.py: PseudoLabelGenerator
c_pseudo, m_rel = framework.generate_pseudo_labels(
    features_unlabeled, c_heatmap,
    features_labeled, labels_labeled,
    contour_mask
)
```

**算法步骤**：
1. **KNN检索**：找到最近的5个有标签样本
2. **GMM噪声检测**：过滤异常样本
3. **信息混合**：c_mixed = 0.7 * c_heatmap + 0.3 * c_knn
4. **历史验证**：检查与之前伪标签的一致性
5. **可靠性掩码**：基于熵判断（m_rel = H(c) <= τ_batch）
6. **最终伪标签**：c_pseudo = m_noise ⊙ c_accepted

**输出**：
- `c_pseudo`: [B, 10] 伪标签
- `m_rel`: [B] 可靠性掩码（1=可靠，0=不可靠）

---

### Stage 4: 不确定性引导采样（每epoch）

**目标**：优先选择困难的、模型不确定的样本

**实现**：
```python
# pseudo_label_and_sampling.py: UncertaintyGuidedSampler
sampled_indices = framework.sample_unlabeled_data(
    features, logits, batch_size
)
```

**采样权重**：
```
密度 ρ(x) = 1 / (k-NN距离)
熵 H(x) = -Σ(p*log(p) + (1-p)*log(1-p))
权重 w(x) = (1 / ρ(x)) * H(x)

选择：低密度 + 高熵的样本（困难样本）
```

---

### Stage 5: 多任务联合优化（训练循环）

**目标**：通过多个互补的损失函数优化模型

**实现**：
```python
# loss_functions.py: MultiTaskLoss
losses = multi_task_loss(
    c_heatmap_labeled, y_true,
    c_heatmap_unlabeled, c_pseudo,
    f_weak, f_strong,
    m_rel,
    c_graph, contour_mask
)
```

**总损失公式**：
```
L_total = L_supervised 
        + 0.5 * L_align           (仅可靠样本)
        + 0.3 * L_consistency      (仅不可靠样本)
        + 0.2 * L_geo             (破除背景捷径)
        + 0.1 * L_graph           (概念关系一致)
        + 0.1 * L_contrast        (对比学习)
```

**各损失详解**：

| 损失函数 | 公式 | 用途 | 权重 |
|---------|------|------|------|
| **L_supervised** | BCE(c_heatmap, y_true) | 有标签数据监督 | 1.0 |
| **L_align** | Σ w·BCE(c_heatmap, c_pseudo) · m_rel | 伪标签对齐 | 0.5 |
| **L_consistency** | MSE(f_weak, f_strong) · (1-m_rel) | 特征一致性 | 0.3 |
| **L_geo** | \|\|c_heatmap ⊙ (1-mask)\|\|² | 破除背景捷径 | 0.2 |
| **L_graph** | MSE(c_graph, c_heatmap) | GCN一致性 | 0.1 |
| **L_contrast** | -log(exp(sim_pos)/Σexp(sim_neg)) | 对比学习 | 0.1 |

---

## 🔧 关键参数配置

### 模型参数
```python
Config.input_dim = 3              # RGB图像
Config.feature_dim = 256          # 特征维度
Config.num_concepts = 10          # 概念数量
Config.num_heads = 4              # 注意力头数
```

### 损失权重
```python
Config.lambda_align = 0.5
Config.lambda_consistency = 0.3
Config.lambda_geo = 0.2
Config.lambda_graph = 0.1
Config.lambda_contrast = 0.1
```

### 伪标签参数
```python
Config.alpha_mix = 0.7            # 混合系数：0.7*heatmap + 0.3*knn
Config.cooccurrence_threshold = 0.1  # 超图阈值
```

### 训练参数
```python
Config.num_epochs = 100
Config.batch_size_labeled = 32
Config.batch_size_unlabeled = 64
Config.learning_rate = 0.001
Config.learning_rate_scheduler = 'CosineAnnealing'
```

---

## 📊 核心模块详解

### 模块1: core_modules.py

**ConceptHypergraph**
```python
# 构建和管理概念关系图
hypergraph.build_from_labeled_data(labels, concept_ids)
adj_matrix = hypergraph.get_adjacency_matrix()
```

**NegativeSampleQueue**
```python
# 为对比学习维护负样本队列
queue.update(features)  # 添加特征
negatives = queue.get_negatives(k=256)  # 获取负样本
```

**PseudoLabelHistory**
```python
# 跟踪伪标签历史，用于一致性验证
history.update(sample_ids, pseudo_labels)
consistency = history.get_consistency(sample_ids, new_labels)
```

### 模块2: network_architecture.py

**BaseEncoder**
- CNN编码器，输出全局特征和空间特征

**CrossAttentionModule**
- 多头跨注意力，聚焦到相关区域

**GlobalConceptPredictor / LocalConceptPredictor**
- 全局和局部概念预测头

**FusionGate**
- 学习权重融合全局和局部预测

**GraphConvolutionalNetwork**
- GCN推理模块，利用概念关系图

**CompleteModel**
- 集成所有组件的完整模型

### 模块3: pseudo_label_and_sampling.py

**PseudoLabelGenerator**
```python
# 6步自适应伪标签生成
c_pseudo, m_rel = generator.generate(
    c_heatmap, c_knn, contour_mask, history
)
```

**UncertaintyGuidedSampler**
```python
# 反课程采样：低密度高熵样本
sampled_indices = sampler.sample(features, logits, batch_size)
```

### 模块4: loss_functions.py

**MultiTaskLoss** (主要使用)
```python
# 计算多个互补的损失
losses = multi_task_loss(...)
loss_total = losses['L_total']
```

### 模块5: train_framework.py

**ConceptualSSLFramework** (主要入口)
```python
# 完整的训练框架
framework = ConceptualSSLFramework(...)
history = framework.fit(...)
```

---

## 💡 常见用法

### 用法1: 基本训练
```python
from train_framework import ConceptualSSLFramework
from network_architecture import CompleteModel

model = CompleteModel(num_concepts=10)
framework = ConceptualSSLFramework(model=model)

history = framework.fit(
    train_labeled_loader,
    train_unlabeled_loader,
    val_loader,
    labeled_concept_ids
)
```

### 用法2: 推理
```python
framework.model.eval()
with torch.no_grad():
    outputs = framework.extract_features(x, adj_matrix)
    predictions = outputs['c_heatmap']  # [B, 10]
```

### 用法3: 使用ResNet编码器
```python
from torchvision.models import resnet50
import torch.nn as nn

class CustomCompleteModel(nn.Module):
    def __init__(self, num_concepts=10):
        super().__init__()
        backbone = resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        # 其他组件保持不变
```

### 用法4: 自定义损失权重
```python
loss_fn = MultiTaskLoss(
    lambda1=0.6,   # L_align权重
    lambda2=0.2,   # L_consistency权重
    lambda3=0.1,   # L_geo权重
    lambda4=0.05,  # L_graph权重
    lambda5=0.05   # L_contrast权重
)
```

### 用法5: 多GPU训练
```python
model = CompleteModel(...)
model = nn.DataParallel(model)  # 单机多卡

framework = ConceptualSSLFramework(model=model)
```

---

## 🎯 性能优化

### 优化1: 混合精度训练
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for batch in train_loader:
    with autocast():
        outputs = model(batch)
        loss = loss_fn(outputs)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 优化2: 梯度累积
```python
accumulation_steps = 4
for i, batch in enumerate(train_loader):
    outputs = model(batch)
    loss = loss_fn(outputs) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 优化3: 分布式训练
```python
import torch.distributed as dist

dist.init_process_group(backend='nccl')
model = nn.parallel.DistributedDataParallel(model)
```

---

## 📈 监控和调试

### 关键指标
```python
# 伪标签质量
pseudo_label_quality = (m_rel == 1).sum() / len(m_rel)

# 平均熵（不确定性）
avg_entropy = compute_entropy(c_heatmap).mean()

# 损失分解
print(f"L_align: {losses['L_align']:.4f}")
print(f"L_consistency: {losses['L_consistency']:.4f}")
print(f"L_total: {losses['L_total']:.4f}")
```

### 常见问题

**问题1: 损失不下降**
- 检查学习率是否过小
- 检查数据是否有问题
- 尝试调整loss权重

**问题2: 过拟合**
- 增加dropout
- 增加正则化权重
- 增加强增强比例

**问题3: 伪标签质量差**
- 增加KNN在混合中的权重（增加alpha）
- 检查有标签数据质量
- 增加有标签数据量

---

## 📚 相关技术

该框架涉及以下技术：
- ✓ 半监督学习 (Pseudo-labeling)
- ✓ 知识图谱 (Concept Hypergraph)
- ✓ 注意力机制 (Cross-Attention)
- ✓ 图卷积网络 (GCN)
- ✓ 对比学习 (Contrastive Learning)
- ✓ 课程学习 (Curriculum Learning)
- ✓ 不确定性估计 (Uncertainty Estimation)

---

## 🔗 文件快速导航

| 想要... | 查看文件 | 位置 |
|--------|--------|------|
| 运行示例 | example_usage.py | 直接运行 |
| 理解架构 | ARCHITECTURE_GUIDE.py | Part 1-3 |
| 快速查找 | FILE_INDEX.py | 快速查找表 |
| 配置参数 | example_usage.py | Config类 |
| 数据结构 | core_modules.py | 类定义 |
| 网络模型 | network_architecture.py | 模型定义 |
| 训练循环 | train_framework.py | ConceptualSSLFramework |
| 损失函数 | loss_functions.py | 所有损失 |

---

## ✅ 实现完整性检查

- ✓ Stage 1: 概念关系图构建 (core_modules.py)
- ✓ Stage 2: 细粒度特征提取 (network_architecture.py)
- ✓ Stage 3: 自适应伪标签生成 (pseudo_label_and_sampling.py)
- ✓ Stage 4: 不确定性采样 (pseudo_label_and_sampling.py)
- ✓ Stage 5: 多任务优化 (loss_functions.py + train_framework.py)

---

## 📊 代码统计

| 指标 | 数值 |
|-----|------|
| 总代码行数 | ~3200 |
| 核心类数量 | 25+ |
| 核心函数数量 | 80+ |
| 文档行数 | ~2500 |
| 注释率 | 30%+ |

---

## 🚀 后续步骤

1. ✅ 理解框架（阅读本文档）
2. ✅ 运行示例（python example_usage.py）
3. ✅ 准备数据（创建DataLoader）
4. ✅ 调整参数（编辑Config类）
5. ✅ 训练模型（framework.fit()）
6. ✅ 评估性能（计算指标）
7. ✅ 推理部署（framework.extract_features()）

---

**祝使用愉快！** 如有问题，请参考对应的文档文件。
