# 如何配置实验 YAML

本指南详细说明实验配置文件的每个字段。实验配置文件位于项目目录的 `experiments/<name>.yml`，通过 `uesf experiment add --name <name>` 创建。

---

## 文件结构总览

```yaml
# ==================== 基础元信息 ====================
name: baseline_cnn
description: "1D CNN 跨被试情绪识别，5-Fold 交叉验证"
seed: 42

# ==================== 组件挂载 ====================
model:
  name: emotion_cnn
  params: { hidden_size: 128, dropout_rate: 0.5 }

trainer:
  name: emotion_trainer
  params: {}

# ==================== 数据集与切分 ====================
datasets:
  main:
    name: seed_preprocessed
    split:
      strategy: k-fold
      dimension: subject
      k-folds: 5
      val_ratio_in_train: 0.1
      shuffle: true
    transforms:
      - name: zscore_normalize
        fit_on: train
        apply_to: all

# ==================== DataLoader 通道映射 ====================
dataloaders:
  train:
    main: "main.train"
  val:
    main: "main.val"
  test:
    main: "main.test"

# ==================== 训练超参数 ====================
training:
  epochs: 100
  batch_size: 64
  optimizer:
    name: adam
    params: { lr: 0.001, weight_decay: 1e-4 }
  scheduler:
    name: cosine_annealing_lr
    params: { T_max: 50, eta_min: 1e-6 }
  gradient_clip:
    max_norm: 1.0
  early_stopping:
    monitor: val_accuracy
    patience: 15
    mode: max

# ==================== 评估配置 ====================
evaluation:
  metrics: [accuracy, f1_score, auroc]
  k_fold_aggregation: concat

# ==================== 日志配置 ====================
logging:
  use_wandb: false
  checkpoint_metric: val_accuracy
```

---

## 基础元信息

```yaml
name: baseline_cnn          # 实验名称，与文件名一致
description: "实验描述"      # 可选，用于 experiment list 显示
seed: 42                    # 随机种子
```

`seed` 控制数据切分的随机性，相同 seed 保证每次实验的切分方式完全一致，是实验可复现性的基础。

---

## 组件挂载（model / trainer）

```yaml
model:
  name: emotion_cnn         # 对应 project.yml 中 models 块的键名
  params:
    hidden_size: 128        # 传入模型 __init__ 的 **kwargs
    dropout_rate: 0.5

trainer:
  name: emotion_trainer     # 对应 project.yml 中 trainers 块的键名
  params: {}                # 传入训练器 __init__ 的 **kwargs（通常为空）
```

框架在运行时按三级优先级解析组件名：项目级（`project.yml`）> 全局库（`uesf model list`）> 内置（如 `dummy`）。

---

## 数据集定义与切分策略

### 数据集别名

`datasets` 下的每个键是这次实验中给数据集起的**临时别名**（如 `main`），用于在 `dataloaders` 中引用。同一实验可以挂载多个数据集（多源域场景）。

### 切分策略（strategy）

**Holdout 切分**：将数据一次性分为训练/验证/测试集。

```yaml
datasets:
  main:
    name: seed_preprocessed
    split:
      strategy: holdout
      dimension: subject
      shuffle: true
      train_ratio: 0.70
      val_ratio: 0.15
      test_ratio: 0.15
```

**K-Fold 交叉验证**：将数据切为 K 折，循环使用每一折作为测试集。

```yaml
datasets:
  main:
    name: seed_preprocessed
    split:
      strategy: k-fold
      dimension: subject
      k-folds: 5             # K 值；填 -1 或 "total" 表示留一法（LOOCV）
      val_ratio_in_train: 0.1 # 每折内从训练集划出多少比例用于早停验证
      shuffle: true
```

### 切分维度（dimension）

`dimension` 控制**以什么粒度**分配数据，这对防止数据泄露至关重要：

| dimension | 含义 | 适用场景 |
|-----------|------|----------|
| `subject` | 按被试切分：同一被试的所有数据只归属于一个集合 | 跨被试泛化（最严格，推荐） |
| `session` | 按会话切分：同一会话的数据不跨集合 | 跨时间泛化 |
| `recording` | 按录制段切分 | 较宽松的切分 |
| `none` | 按样本随机切分 | 研究被试内泛化能力（可能存在泄露） |

> 使用 `dimension: subject` 可以确保同一被试的 EEG 数据不会同时出现在训练集和测试集，这是跨被试 EEG 研究中防止数据泄露的标准做法。详见[数据泄露防护机制](../concepts/02_data_leakage_prevention.md)。

### shuffle

```yaml
shuffle: true   # 切分前随机打乱（适用于大多数分类实验）
shuffle: false  # 保持固有顺序（适用于按时序前后测对比）
```

---

## 在线变换（transforms）

变换在切分完成后执行，绑定在数据集配置下：

```yaml
datasets:
  main:
    name: seed_preprocessed
    split: ...
    transforms:
      - name: zscore_normalize    # 全局 Z-Score 标准化
        fit_on: train             # 只从训练集计算均值和标准差
        apply_to: all             # 用同一组参数变换 train/val/test
```

`fit_on: train` + `apply_to: all` 是 Fit-on-Train 原则的配置方式：标准化参数仅从训练集学习，然后统一应用到所有集合，确保验证集和测试集的统计信息对模型完全保密。

---

## DataLoader 通道映射（dataloaders）

```yaml
dataloaders:
  train:               # 训练阶段使用的 batch 来源
    main: "main.train" # 格式：<数据集别名>.<切分相>
  val:
    main: "main.val"
  test:
    main: "main.test"
```

通道名（如 `main`）对应 Trainer 接收到的 `batch` 字典中的键名：

```python
# 对应上述配置，batch 结构为：
batch = {
    "main": (data_tensor, labels_tensor)
}
```

**多数据源示例**（域自适应场景）：

```yaml
dataloaders:
  train:
    source: "source_domain.train"
    target: "target_domain.train"
  val:
    target: "target_domain.val"
  test:
    target: "target_domain.test"
```

---

## 训练超参数（training）

### 基础参数

```yaml
training:
  epochs: 100       # 最大训练轮数
  batch_size: 64    # 每个批次的样本数
```

### 优化器（optimizer）

```yaml
optimizer:
  name: adam        # 优化器名称（见内置组件列表）
  params:
    lr: 0.001       # 学习率（必填）
    weight_decay: 1e-4  # 权重衰减（可选）
    # betas, eps 等 Adam 特有参数也可以在这里填写
```

### 学习率调度器（scheduler，可选）

```yaml
scheduler:
  name: cosine_annealing_lr
  params:
    T_max: 50        # 余弦退火周期（epoch 数）
    eta_min: 1e-6    # 最小学习率
```

### 梯度裁剪（gradient_clip，可选）

```yaml
gradient_clip:
  max_norm: 1.0    # 梯度范数上限
  norm_type: 2     # 范数类型（默认 L2）
```

### 早停（early_stopping，可选）

```yaml
early_stopping:
  monitor: val_accuracy  # 监控的指标名称
  patience: 15           # 指标连续多少轮不改善时停止
  min_delta: 0.001       # 视为改善的最小变化量（可选，默认 0.0）
  mode: max              # "max" 表示越大越好，"min" 表示越小越好
```

---

## 评估配置（evaluation）

```yaml
evaluation:
  metrics: [accuracy, f1_score, precision, recall, auroc]
  k_fold_aggregation: concat    # "concat" 或 "mean_std"
```

### k_fold_aggregation 的选择

| 方式 | 行为 | 适用场景 |
|------|------|----------|
| `concat`（推荐） | 将所有折的预测和目标拼接后，一次性计算指标 | 各折样本量不均衡，或标签分布不平衡时 |
| `mean_std` | 每折独立计算指标，最终输出均值和标准差 | 需要置信区间的传统论文写作 |

---

## 日志配置（logging）

```yaml
logging:
  use_wandb: false              # 是否接入 Weights & Biases
  checkpoint_metric: val_accuracy  # 保存最优检查点的依据指标
```

`checkpoint_metric` 指定的指标越大越好时，框架会在该指标达到历史最优时保存检查点。

---

## 实验 YAML 常见问题

**Q：组件名找不到怎么办？**  
A：检查 `project.yml` 中的 `models` 和 `trainers` 块，确认键名与实验 YAML 中的 `name` 字段一致，且 entrypoint 路径正确。也可以运行 `uesf project info` 查看当前项目可用的所有组件。

**Q：`val_ratio_in_train` 在 K-Fold 中的作用是什么？**  
A：K-Fold 切分产生了训练折和测试折，但早停需要验证集。`val_ratio_in_train` 从每折的训练集中再划出一部分用于早停监控，这部分数据不参与训练。

**Q：scheduler 的参数名从哪里查？**  
A：参数名与 PyTorch 官方文档完全一致，参见[内置组件列表](../reference/04_builtin_components.md)中的调度器对应关系。
