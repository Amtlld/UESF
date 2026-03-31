# 数据泄露防护机制

数据泄露（Data Leakage）是 EEG 机器学习研究中最常见的方法论问题之一。UESF 从框架层面提供了两种强制性的防护机制，本文详细解释这两种机制的原理和使用方式。

---

## EEG 研究中的两类数据泄露

### 1. 被试内泄露（Subject-wise Leakage）

**问题**：按样本随机切分时，同一被试的不同 Epoch 可能同时出现在训练集和测试集中。

由于同一被试在不同时刻的 EEG 信号具有高度相似的个体特征（信道阻抗、信号波形习惯等），模型实际上在测试集上"见过"被试，测试的不是模型的泛化能力，而是记忆能力。

**结果**：实验报告的准确率虚高，模型在真实新被试上表现大幅下降。

### 2. 统计信息泄露（Statistical Leakage）

**问题**：在切分前对全体数据做全局 Z-Score 标准化时，测试集的统计信息（均值、标准差）已经参与了参数计算。

即使在测试阶段没有直接使用测试标签，测试集的特征分布信息已通过标准化参数"泄露"给了模型训练过程。

**结果**：模型在训练集和测试集上的分布差异被人为消除，评估结果不可信。

---

## UESF 的防护方案一：维度隔离切分

UESF 的 Splitter 支持按数据的特定维度进行隔离切分，确保同一维度的所有数据只归属于训练集或测试集之一。

### split.dimension 的含义

| 配置值 | 切分粒度 | 保证 |
|--------|----------|------|
| `subject` | 按被试切分 | 同一被试的所有 Epoch 只在训练集或测试集中出现 |
| `session` | 按会话切分 | 同一被试同一会话的数据不跨集合 |
| `recording` | 按录制段切分 | 同一录制段的 Epoch 不跨集合 |
| `none` | 按样本随机切分 | 无隔离保证，研究被试内泛化时使用 |

### 正确配置（跨被试泛化）

```yaml
datasets:
  main:
    name: seed_preprocessed
    split:
      strategy: k-fold
      dimension: subject    # 按被试隔离
      k-folds: 5
```

5-Fold 时，14 名被试会被分为 5 组，每次用 4 组（约 11 名被试）训练，1 组（约 3 名被试）测试。同一被试的所有 Epoch **绝对不会**跨越训练集和测试集。

### 错误配置（存在泄露风险）

```yaml
datasets:
  main:
    name: seed_preprocessed
    split:
      strategy: holdout
      dimension: none       # 按样本随机切分，存在泄露
      train_ratio: 0.8
      val_ratio: 0.1
      test_ratio: 0.1
```

`dimension: none` 允许同一被试的 Epoch 同时出现在训练集和测试集。除非你的研究问题明确是被试内泛化（即模型只需要在同一被试的不同时段泛化），否则不应使用此设置。

---

## UESF 的防护方案二：Fit-on-Train 在线变换

全局标准化（Z-Score）必须在切分完成**之后**执行，且只能使用训练集数据拟合参数。

### 执行顺序保证

UESF 严格执行以下顺序：

```
数据加载 → 切分（生成 train/val/test 索引）→ 在线变换（仅fit_on训练集）→ 训练
```

在实验 YAML 中配置：

```yaml
datasets:
  main:
    name: seed_preprocessed
    split:
      strategy: k-fold
      dimension: subject
      k-folds: 5
    transforms:
      - name: zscore_normalize
        fit_on: train      # 只从训练集计算均值(μ)和标准差(σ)
        apply_to: all      # 用同一组μ/σ变换 train/val/test
```

### 为什么不在预处理阶段做全局标准化？

预处理（`preprocess.yml`）在数据切分之前执行，此时 train/val/test 的划分还未确定，无法实现 Fit-on-Train。因此：

- **预处理阶段**只允许 Epoch 内部的标准化（`epoch_normalize`），这不会引入跨被试的统计信息
- **全局跨被试的 Z-Score 标准化**必须放在实验的 `transforms` 中

---

## K-Fold 结果聚合的统计正确性

K-Fold 实验产生 K 个测试集上的预测结果，如何聚合影响最终指标的统计含义。

### concat 模式（推荐）

将所有折的测试集预测和标签拼接后，一次性计算指标：

```yaml
evaluation:
  k_fold_aggregation: concat
```

**优点**：对类别不均衡具有鲁棒性。若某一折的某类样本极少，该折的 F1 分数会偏低，但在全局拼接后被正确权重化，不会放大偶然误差。

**适用场景**：大多数 EEG 分类实验，特别是各折之间类别分布有差异时。

### mean_std 模式

每折独立计算指标，报告均值和标准差：

```yaml
evaluation:
  k_fold_aggregation: mean_std
```

**输出**：如 `accuracy: 0.7856 ± 0.0234`，提供置信区间信息。

**适用场景**：论文中需要报告标准差作为稳定性指标时；或各折样本量基本均衡时。

---

## 快速检查清单

在提交实验结果前，检查以下配置：

- [ ] `split.dimension` 设置为 `subject`（或你研究问题对应的正确维度，而非 `none`）
- [ ] 全局 Z-Score 标准化配置在 `transforms` 中，而非在 `preprocess.yml` 的 `data` 流中
- [ ] `transforms.fit_on` 设置为 `train`
- [ ] 实验配置了 `seed` 确保可复现性
