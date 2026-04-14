# 二、YAML 配置接口设计

## 2.1 划分原语定义

框架内部有三种划分原语，在不同上下文中使用：

### Split Block（完整划分 → train/val/test）

用于 Regular 模式顶层 `split` 和 UDA 模式 `target.split`（inductive）。

```yaml
strategy: holdout | k-fold        # 划分策略
dimension: subject | session | recording  # 划分维度（隔离单位）
shuffle: false                     # 是否打乱
seed: 42                           # 随机种子（可省略，继承全局 seed）

# holdout 专有
train_ratio: 0.8
val_ratio: 0.1
test_ratio: 0.1

# k-fold 专有
k: 5                              # 折数，-1 表示 leave-one-out
val_ratio: 0.1                    # 可选，从整体数据中切出验证集的比例
```

> **val_ratio 语义说明**：`val_ratio` 始终表示**占整体数据的比例**。
> - Holdout：`train_ratio + val_ratio + test_ratio = 1.0`，直接按比例切分。
> - K-Fold：每折的测试集由 k 决定（占 `1/k`），`val_ratio` 表示从整体中切出的验证集比例。框架内部自动换算为 `val_ratio_in_train = val_ratio / (1 - 1/k)`，从每折的训练部分中切出对应数量。
>
> 例如：`k=5, val_ratio=0.1` → 每折中测试集占 20%，验证集占 10%，训练集占 70%（等效 `val_ratio_in_train = 0.125`）。

### ValSplit Block（仅切出验证集 → train/val）

用于 UDA 模式 `source.split` 和 transductive 模式 `target.split`。源域永远不做测试，因此不需要 `strategy`、`train_ratio`、`test_ratio`。

```yaml
dimension: subject | session | recording  # 划分维度（隔离单位）
val_ratio: 0.2                             # 从全域数据中切出的验证集比例
shuffle: true                              # 是否打乱
```

> **dimension 生效**：ValSplit 也按 dimension 分组后再切分。例如 `dimension: session` 时，先按 session 分组，再按 val_ratio 将部分 session 组整体划入验证集，确保同一 session 的数据不会跨 train/val 泄漏。

### DomainPartition（域划分 → source/target）

用于 UDA 模式 `uda.domain`，将数据划分为源域和目标域。

```yaml
strategy: holdout | k-fold
dimension: dataset | subject | session
# holdout + dataset: 显式指定 source/target
# holdout + subject/session: target_count 或 target_ratio（二者互斥）
# k-fold: k（-1 = leave-one-out）
```

> **设计说明**：`dataset` 维度仅出现在 DomainPartition 中，不出现在 Split Block / ValSplit Block 中。跨数据集的 Regular 模式操作由 `split.dimension: dataset` 专门处理。

---

## 2.2 Regular 模式（常规深度学习）

**场景 A：单数据集，subject 级 K-Fold**

```yaml
experiment_name: regular_kfold_subject
seed: 42

datasets:
  bcic4_2a:
    name: BCIC4_2a_preprocessed

split:
  strategy: k-fold
  dimension: subject
  k: 9
  val_ratio: 0.1
  shuffle: false

model:
  name: EEGNet
  params:
    dropout: 0.5

training:
  epochs: 100
  batch_size: 64
  optimizer:
    name: adam
    params: { lr: 0.001 }
  logging:
    backend: tensorboard

evaluation:
  metrics: [accuracy, f1_score]
```

**场景 B：单数据集，recording 级 holdout**

```yaml
experiment_name: regular_holdout_recording
seed: 42

datasets:
  bcic4_2a:
    name: BCIC4_2a_preprocessed

split:
  strategy: holdout
  dimension: recording
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  shuffle: true

model: { name: EEGNet }
training: { epochs: 50, batch_size: 32 }
evaluation: { metrics: [accuracy] }
```

**场景 C：跨数据集常规训练 — Holdout（显式指派）**

```yaml
experiment_name: cross_dataset_regular
seed: 42

datasets:
  bcic4_2a:
    name: BCIC4_2a_preprocessed
  bcic4_2b:
    name: BCIC4_2b_preprocessed
  physionet:
    name: PhysioNet_preprocessed

alignment:
  channel: intersection
  label: true

split:
  strategy: holdout
  dimension: dataset
  assign:
    train: [bcic4_2a, bcic4_2b]
    test: [physionet]
  val_ratio: 0.1

model: { name: EEGNet }
training: { epochs: 100, batch_size: 64 }
evaluation: { metrics: [accuracy] }
```

> **验证集处理**：`dimension=dataset` + `holdout` 时不支持 `assign.val`。验证集从 `assign.train` 指定的训练数据集展平合并后，按 `val_ratio` 随机切出（默认 `shuffle=true`）。若未指定 `val_ratio`，则无验证集。

**场景 D：跨数据集 K-Fold（各数据集轮流做测试集）**

```yaml
split:
  strategy: k-fold
  dimension: dataset
  k: -1   # leave-one-dataset-out
  val_ratio: 0.1
```

> **验证集处理**：`dimension=dataset` + `k-fold` 时，每折的测试集为一个数据集，其余数据集作为训练集。验证集通过在训练集内部按 `val_ratio` 切分产生（按 recording 维度随机抽取），复用顶层 `split.val_ratio` 配置。若未指定 `val_ratio`，则该折无验证集。

> **Regular 模式小结**：`split` 统一位于顶层，使用 Split Block 结构。`dimension` 可为 `subject | session | recording | dataset` 四选一。当 `dimension: dataset` 时需声明多个 datasets 并配置 alignment。

---

## 2.3 UDA 模式（无监督域适应）

UDA 配置为三层结构：**域划分**（DomainPartition）+ **适应模式** + **域内划分**（source 用 ValSplit，target 用 Split 或 ValSplit）。

```yaml
mode: uda

uda:
  # -------- 第一层：域划分（DomainPartition） --------
  domain:
    strategy: holdout | k-fold
    dimension: dataset | subject | session
    # holdout + dataset 维度时显式指定：
    source: [ds_a, ds_b]
    target: ds_c
    # holdout + subject/session 维度时（二者互斥，不可同时出现）：
    target_count: 1          # 或 target_ratio: 0.2
    # k-fold 时：
    k: 5                     # -1 = leave-one-out

  # -------- 第二层：适应模式 --------
  adaptation: transductive | inductive

  # -------- 第三层：域内划分 --------
  source:
    split:                    # ValSplit — 源域 train/val 划分
      dimension: recording    # 划分维度（不可与 domain.dimension 相同）
      val_ratio: 0.2
      shuffle: true

  target:
    split:                    # inductive: Split Block（必填）; transductive: ValSplit（可选）
      # --- inductive 时使用完整 Split Block ---
      strategy: holdout       # holdout 或 k-fold
      dimension: recording    # 划分维度（不可与 domain.dimension 相同）
      train_ratio: 0.7
      val_ratio: 0.1
      test_ratio: 0.2
      shuffle: true
      # --- transductive 时使用 ValSplit（可省略） ---
      # dimension: session
      # val_ratio: 0.15
      # shuffle: false
```

> **关键约束**：
> - `source.split` 使用 **ValSplit** 结构：仅 `dimension + val_ratio + shuffle`，不含 `strategy`、`train_ratio`、`test_ratio`（源域永远不做测试）
> - `transductive` 模式下，`target.split` 使用 **ValSplit** 结构或省略（目标域全域 = 训练 + 测试）
> - `inductive` 模式下，`target.split` 使用完整 **Split Block** 结构（必须包含 `test_ratio` 或 `k`）
> - `domain.dimension` 不可与 `source.split.dimension` 或 `target.split.dimension` 相同（防止语义冲突和边界退化）
> - `target_count` 和 `target_ratio` 互斥，不可同时出现
>
> **Transductive target_val 的用途**：transductive 模式下 `target.split` 切出的 `target_val` 不用于监督评估，而是用于计算**目标域信息熵**和**伪标签稳定性**（观察连续若干 Epoch 目标域预测结果的变化比例），以实现早停和模型选择。

---

**UDA 场景 A：跨数据集 UDA — Transductive — Holdout**

```yaml
experiment_name: cross_dataset_uda_transductive
seed: 42

datasets:
  bcic4_2a:
    name: BCIC4_2a_preprocessed
  bcic4_2b:
    name: BCIC4_2b_preprocessed

alignment:
  channel: intersection
  label: true

mode: uda

uda:
  domain:
    strategy: holdout
    dimension: dataset
    source: [bcic4_2a]
    target: bcic4_2b

  adaptation: transductive

  source:
    split:
      dimension: recording
      val_ratio: 0.2
      shuffle: true

  target:
    split:
      dimension: session
      val_ratio: 0.15
      shuffle: false

model: { name: DANN }
training: { epochs: 200, batch_size: 128 }
evaluation: { metrics: [accuracy, f1_score] }
```

**UDA 场景 B：数据集内 UDA — Subject 级 K-Fold — Inductive**

```yaml
experiment_name: intra_dataset_uda_kfold_inductive
seed: 42

datasets:
  bcic4_2a:
    name: BCIC4_2a_preprocessed

mode: uda

uda:
  domain:
    strategy: k-fold
    dimension: subject
    k: -1                    # leave-one-subject-out

  adaptation: inductive

  source:
    split:
      dimension: recording
      val_ratio: 0.2
      shuffle: true

  target:
    split:
      strategy: holdout
      dimension: recording
      train_ratio: 0.6
      val_ratio: 0.1
      test_ratio: 0.3
      shuffle: true

model: { name: DANN }
training: { epochs: 150, batch_size: 64 }
evaluation: { metrics: [accuracy] }
```

**UDA 场景 C：跨数据集 UDA — K-Fold — Inductive**

```yaml
uda:
  domain:
    strategy: k-fold
    dimension: dataset
    k: -1                    # leave-one-dataset-out，各数据集轮流做目标域

  adaptation: inductive

  source:
    split:
      dimension: subject
      val_ratio: 0.2

  target:
    split:
      strategy: k-fold
      dimension: session
      k: 3
      val_ratio: 0.1
```

> 注意：inductive 模式下 target split 使用 k-fold 时，每个 domain fold 内部再产生 k 个子 fold，形成**嵌套折叠**。框架需处理此情况。

**UDA 场景 D：数据集内 UDA — Session 级 Holdout — Transductive**

```yaml
uda:
  domain:
    strategy: holdout
    dimension: session
    target_count: 1

  adaptation: transductive

  source:
    split:
      dimension: recording
      val_ratio: 0.2
```

> Transductive 模式 + 省略 `target.split` = 目标域全域作为无监督训练数据和测试数据，无需再切。

---

## 2.4 配置完整结构总览

```yaml
# ==================== 实验 YAML 完整结构 ====================

experiment_name: string           # 必填
description: string               # 可选
seed: int                         # 可选，默认 42

# ---- 数据源 ----
datasets:                         # 必填，至少一个
  <alias>:
    name: string                  # 预处理数据集名
    transforms:                   # 可选，后处理变换
      - name: zscore_normalize

# ---- 跨数据集对齐（多数据集时） ----
alignment:                        # 可选
  channel: intersection           # 通道对齐方式（目前仅支持 intersection）
  label: true | false             # 是否检查标签一致性，默认 true

# ---- 实验模式 ----
mode: regular | uda               # 可选，默认 regular

# ---- Regular 模式配置（Split Block） ----
split:                            # mode=regular 时必填
  strategy: holdout | k-fold
  dimension: subject | session | recording | dataset
  shuffle: bool
  # holdout 参数:
  train_ratio: float
  val_ratio: float
  test_ratio: float
  # holdout + dimension=dataset 参数:
  assign:                         # 显式指派数据集到各阶段（不支持 assign.val）
    train: [alias, ...]
    test: [alias, ...]
  # k-fold 参数:
  k: int                          # 折数，-1 = leave-one-out
  val_ratio: float                # 可选，从整体数据中切出验证集的比例

# ---- UDA 模式配置 ----
uda:                              # mode=uda 时必填
  domain:                         # DomainPartition
    strategy: holdout | k-fold
    dimension: dataset | subject | session
    # holdout + dataset:
    source: [alias, ...]
    target: alias
    # holdout + subject/session（二者互斥）:
    target_count: int             # 或 target_ratio: float
    # k-fold:
    k: int                        # -1 = leave-one-out
  adaptation: transductive | inductive
  source:                         # ValSplit
    split:
      dimension: subject | session | recording  # 不可与 domain.dimension 相同
      val_ratio: float
      shuffle: bool
  target:
    split:                        # inductive: Split Block（必填）; transductive: ValSplit（可选）
      # inductive 使用完整 Split Block:
      strategy: holdout | k-fold
      dimension: subject | session | recording  # 不可与 domain.dimension 相同
      # holdout: train_ratio, val_ratio, test_ratio
      # k-fold: k, val_ratio
      shuffle: bool
      # transductive 使用 ValSplit（可省略）:
      # dimension, val_ratio, shuffle

# ---- 模型 ----
model:
  name: string
  params: { ... }

# ---- 训练器 ----
trainer:                          # 可选，默认 DummyTrainer
  name: string
  params: { ... }

# ---- 设备 ----
device: string                    # 可选，覆盖全局配置中的 device 设置（如 "cuda:0"），省略时使用全局配置（默认 "cpu"）

# ---- 训练配置 ----
training:
  epochs: int
  batch_size: int                 # 默认 32
  optimizer:
    name: string                  # 默认 adam
    params: { ... }
  scheduler:                      # 可选
    name: string
    params: { ... }
  early_stopping:                 # 可选
    metric: string
    patience: int
    mode: min | max
  gradient_clip: float            # 可选
  checkpoint:                     # 可选，省略则不保存 checkpoint
    metric: string                # 用于选择 best model 的指标名（如 "val_accuracy"）
    mode: min | max               # metric 越小越好还是越大越好，默认 max
    dir: string                   # checkpoint 保存目录，默认 experiments/results/<experiment_name>/checkpoints
  logging:                        # 可选，省略则不记录训练日志
    backend: tensorboard          # 日志后端，目前仅支持 tensorboard
    log_every_n_epochs: int       # 每 N 个 epoch 记录一次，默认 1
    log_graph: bool               # 是否记录模型计算图，默认 false

# ---- 评估 ----
evaluation:
  metrics: [string]               # 默认 [accuracy]
  k_fold_aggregation: concat | mean_std  # 默认 concat

# ---- 训练过程监测 ----
# 位于 training 块内部，见下方 training.logging
```
