# 如何配置实验 YAML

本指南详细说明实验配置文件的每个字段。实验配置文件位于项目目录的 `experiments/<name>.yml`，通过 `uesf experiment add --name <name>` 创建。

UESF 支持两大类实验模式：**常规深度学习**（`mode: regular`，默认）和 **UDA 无监督域自适应**（`mode: uda`）。

---

## 文件结构总览（常规模式）

```yaml
# ==================== 基础元信息 ====================
experiment_name: baseline_cnn
description: "1D CNN 跨被试情绪识别，5-Fold 交叉验证"
seed: 42
# mode: regular  # 可省略，默认 regular

# ==================== 数据集 ====================
datasets:
  main:
    name: seed_preprocessed
    transforms:
      - name: zscore_normalize
        scope: per_dataset        # per_dataset (默认) | global

# ==================== 切分 ====================
# 顶层 split：整个实验只有一份切分配置，框架自动接线 dataloaders
split:
  strategy: k-fold
  dimension: subject              # subject | session | recording | flatten | dataset
  k: 5
  val_ratio: 0.1                  # 从整体切出 10%，框架内部按折换算
  shuffle: true

# ==================== 组件挂载 ====================
model:
  name: emotion_cnn
  params: { hidden_size: 128, dropout_rate: 0.5 }

trainer:
  name: emotion_trainer
  params: {}

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
    metric: val_accuracy
    patience: 15
    mode: max
  checkpoint:
    metric: val_accuracy
    mode: max                     # 可省略，默认 max
  logging:                        # 可选；启用需安装 uesf[tensorboard]
    backend: tensorboard
    log_every_n_epochs: 1
    log_graph: false

# ==================== 评估配置 ====================
evaluation:
  metrics: [accuracy, f1_score, auroc]
  k_fold_aggregation: concat      # concat | mean_std
```

---

## 基础元信息

```yaml
experiment_name: baseline_cnn     # 实验名称，与文件名一致
description: "实验描述"            # 可选，用于 experiment list 显示
seed: 42                          # 随机种子
```

`seed` 控制数据切分的随机性，相同 seed 保证每次实验的切分方式完全一致，是实验可复现性的基础。框架为多层切分派生独立种子（domain=+0、source=+1、target=+2），不同层随机性互不影响。

---

## 组件挂载（model / trainer）

```yaml
model:
  name: emotion_cnn               # 对应 project.yml 中 models 块的键名
  params:
    hidden_size: 128              # 传入模型 __init__ 的 **kwargs
    dropout_rate: 0.5

trainer:
  name: emotion_trainer           # 对应 project.yml 中 trainers 块的键名
  params: {}                      # 传入训练器 __init__ 的 **kwargs（通常为空）
```

框架在运行时按三级优先级解析组件名：项目级（`project.yml`）> 全局库（`uesf model list`）> 内置（如 `dummy_model` / `dummy_trainer`）。

---

## 数据集定义

`datasets` 下的每个键是这次实验中给数据集起的**临时别名**（如 `main`）。同一实验可以挂载多个数据集（跨数据集训练或 UDA）。dev6 不再需要 `dataloaders` 显式接线 —— 框架根据 `mode` 与 `split` 自动决定通道名，见下文 "自动通道接线"。

```yaml
datasets:
  main:
    name: seed_preprocessed       # 预处理数据集名
    # transforms: ...             # 可选，见"在线变换"
```

---

## 切分（split）

**顶层 `split` 在常规模式下必填**。三种划分原语各司其职：

- **Split Block**（完整 train/val/test）—— 用于顶层 `split` 与 UDA inductive 的 `target.split`
- **ValSplit Block**（仅 train/val，切出验证集）—— 用于独立的 `val_split` 子块与 UDA 的 `source.split`
- **DomainPartition**（source/target 域划分）—— 用于 UDA 的 `uda.domain`

### 策略（strategy）

**Holdout**：一次性切出 train/val/test。

```yaml
split:
  strategy: holdout
  dimension: subject
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  shuffle: true
```

**K-Fold 交叉验证**：切为 K 折，每折轮流做测试集。`k: -1` 表示留一法（LOOCV，即 k = 切分维度上的组数）。

```yaml
split:
  strategy: k-fold
  dimension: subject
  k: 5
  val_ratio: 0.1                  # 框架内部按 0.1 / (1 - 1/5) = 0.125 换算到每折训练部分
  shuffle: true
```

> **框架内部换算**：k-fold 时 `val_ratio` 表达的是"从整体数据中切出的验证集比例"。框架按 `val_ratio / (1 - 1/k)` 换算为每折训练部分中的比例，用户无感知。

### 切分维度（dimension）

`dimension` 控制**以什么粒度**分配数据，这对防止数据泄露至关重要：

| dimension | 含义 | 适用场景 |
|-----------|------|----------|
| `subject` | 按被试切分：同一被试的所有数据只归属于一个集合 | 跨被试泛化（最严格，推荐） |
| `session` | 按会话切分：同一 (被试, 会话) 内的 recording 整体分配 | 跨时间泛化 |
| `recording` | 每个 recording 作为独立组 | 最细粒度的维度切分 |
| `flatten` | 将 (subject, session, recording) 三元组展平后随机切分 | 研究被试内泛化（无隔离） |
| `dataset` | 整个数据集作为一个组分配 | 跨数据集泛化（需 ≥2 个数据集） |

> 使用 `dimension: subject` 可以确保同一被试的 EEG 数据不会同时出现在训练集和测试集，这是跨被试 EEG 研究中防止数据泄露的标准做法。详见[数据泄露防护机制](../concepts/02_data_leakage_prevention.md)。

**R26 约束**：`dimension: flatten` 必须配合 `shuffle: true`（展平后无序遍历无物理意义，配置校验阶段直接拒绝）。

### shuffle

```yaml
shuffle: true    # 打乱切分维度上的分组顺序（R27 默认值）
shuffle: false   # holdout: 按维度原始顺序依次分配 train → val → test
                 # k-fold:  按维度原始顺序轮换，fold i 的 test = groups[i]
                 # dimension=flatten 时 shuffle 必须为 true
```

### 验证集独立划分（val_split，可选）

默认情况下，val 与 test 在同一 `dimension` 上按比例切出。如果想在**另一个维度**上切出 val（例如主切分按 subject、val 按 session），使用 `val_split` 子块：

```yaml
split:
  strategy: k-fold
  dimension: subject
  k: 5
  val_split:
    dimension: session
    val_ratio: 0.1
    shuffle: true                 # 默认 true（R27）
  shuffle: true
```

> **与主 `val_ratio` 互斥（R19）**：同一 Split Block 中 `val_split` 和 `val_ratio` 只能出现一个。`val_split` 存在时主划分只产生 train/test，ValSplitter 从每折的 train 中切出 val。
>
> **R30 偏序约束**：主 `dimension: flatten` 时，`val_split.dimension` 必须也为 `flatten`（flatten 切分后训练子集不再保持 5-D 结构，无法在其他维度上分组）。
>
> **R21 值域**：`val_split.val_ratio` 必须在 `(0, 1)` 内（val_split 出现时 val_ratio 必须 > 0，否则不应配置）。

---

## 在线变换（transforms）

变换在切分完成后按"fit-on-train / apply-to-all"原则执行，避免测试集统计信息泄露。dev6 引入 `scope` 字段来描述变换的 **作用范围**：

```yaml
datasets:
  main:
    name: seed_preprocessed
    transforms:
      - name: zscore_normalize
        scope: per_dataset        # 每数据集独立（默认）
        # scope: global           # 所有训练数据集合并 fit（仅 Regular 模式）
```

### 两种 scope 的语义

| scope | fit 范围 | 适用场景 |
|-------|----------|----------|
| `per_dataset`（默认） | **单数据集 / 数据集内切分**：fit on 该数据集的 train 部分<br>**跨数据集（dimension=dataset）**：训练数据集 fit on train；测试数据集 fit on 自身全量（无训练部分） | 大多数场景；单数据集实验下与 global 等价 |
| `global` | 所有训练数据集合并 train 数据 → 单次 fit → 应用到所有数据 | 需要用训练数据集统计量来 transform 测试数据集 |

**单数据集退化**：单数据集场景下，`per_dataset` 与 `global` 行为完全相同（合并集合只有一个参与者）。

**UDA 约束（R23）**：`mode: uda` 时 `scope` 必须为 `per_dataset`，框架会为源域、目标域分别 fit 对应数据集的相应部分：

- 源域各数据集：fit on source_train，transform source_train / source_val
- 目标域（inductive）：fit on target_train，transform target_train / target_val / target_test
- 目标域（transductive）：fit on 目标域全量数据，transform 同一份数据

**跨数据集一致性（R29）**：多数据集场景下，各 alias 的 `transforms` 列表必须长度相同且对应位置的 `(name, scope)` 完全一致，否则 `ConfigValidator.validate()` 拒绝。

---

## 自动通道接线

dev6 删除了原来的 `dataloaders` 段，框架根据 `mode` 与 `split.dimension` 自动生成 DataLoader 通道字典：

| mode | dimension | train 批次键 | val 批次键 | test 批次键 |
|------|-----------|-------------|-----------|-----------|
| regular | 单数据集任一维度 | `main` | `main` | `main` |
| regular | `dataset` | `main`（训练数据集合并） | `main`（训练数据集合并） | `main`（测试数据集合并） |
| uda | — | `source`, `target` | `source_val`, `target_val` | `main`（目标域测试） |

Trainer 接到的 `batch` 结构即为这些键的 `dict[str, (data, labels)]`。Regular 模式下 Trainer 通常只需 `batch["main"]`；UDA 模式下从 `batch["source"]` / `batch["target"]` 分别取有监督损失 / 域对齐项。

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
    lr: 0.001
    weight_decay: 1e-4
```

### 学习率调度器（scheduler，可选）

```yaml
scheduler:
  name: cosine_annealing_lr
  params:
    T_max: 50
    eta_min: 1e-6
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
  metric: val_accuracy   # 监控的指标名称
  patience: 15           # 指标连续多少轮不改善时停止
  min_delta: 0.001       # 视为改善的最小变化量（可选，默认 0.0）
  mode: max              # "max" 越大越好；"min" 越小越好
```

### 检查点（checkpoint，可选）

```yaml
checkpoint:
  metric: val_accuracy   # 用于选择 best model 的指标名
  mode: max              # 默认 max
  # dir: ...             # 可选，默认 <output_dir>/fold_<i>/checkpoints
```

### 训练日志（logging，可选）

最小配置（仅启用后端，其他字段走默认）：

```yaml
logging:
  backend: tensorboard         # 当前仅支持 tensorboard（R17）
  log_every_n_epochs: 1        # 正整数（R18），epoch 级写入频率
  log_graph: false             # 是否记录模型计算图（默认 false）
```

完整字段一览（全部可选；未声明的字段保持 dev6 早期默认行为）：

```yaml
logging:
  backend: tensorboard
  log_every_n_epochs: 1
  log_every_n_steps: 10            # 每 10 个 batch 写一次 step 级标量（tag 前缀 step/）；省略 = 关闭
  log_graph: false
  log_lr: true                     # 是否写学习率（epoch 级 lr 与 step 级 step/lr）

  # 训练集（两类独立配置）
  train_step_scalars: [loss]       # 白名单筛选 training_step 返回的标量；省略 = 记录全部
  train_metrics: [accuracy]        # 需 Trainer 的 training_step 返回 preds/targets（opt-in）

  # 验证集（evaluation.metrics 的子集白名单）
  val_metrics: [accuracy]          # 必须是 evaluation.metrics 的子集（R37）；省略 = 记录全部

  # 测试集（opt-in；触发数据泄露 WARN）
  test_metrics: [accuracy]         # 每 log_every_n_epochs 在测试集上运行一次评估
```

**TensorBoard tag 命名**：

| 来源 | tag | step 轴 |
|------|-----|---------|
| `training_step` 标量（epoch 平均） | `<name>`（如 `loss`） | epoch |
| 训练集 computed 指标（`train_metrics`） | `train_<name>` | epoch |
| 验证集 computed 指标（`evaluation.metrics`，经 `val_metrics` 过滤） | `val_<name>` | epoch |
| 测试集 computed 指标（`test_metrics`） | `test_<name>` | epoch |
| 学习率 | `lr` | epoch |
| Step 级训练标量 / lr | `step/<name>`、`step/lr` | 全局 step |

**Trainer preds/targets 返回约定**（用于 `train_metrics`）：

```python
class MyTrainer(BaseTrainer):
    def training_step(self, batch, batch_idx, optimizer):
        data, labels = batch["main"]
        logits = self.model(data)
        loss = criterion(logits, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        return {
            "loss": loss.item(),
            "preds": logits.detach().argmax(dim=1),   # 或保留 logits 交给 metric 自行 argmax
            "targets": labels.detach(),
        }
```

如果 `training.logging.train_metrics` 已声明但 `training_step` 没有返回 `preds`/`targets`，
框架会 WARN 并静默跳过训练集指标（不阻断训练）。

**自定义指标**：`train_metrics` / `val_metrics` / `test_metrics` 中都可以使用
`project.yml` 中 `metrics:` 注册的项目级指标、或 `uesf metric add` 添加的全局指标——
指标名解析链复用 `evaluation.metrics` 的逻辑，无需额外配置。

> **依赖**：启用 `tensorboard` 后端需安装可选依赖 `uv pip install uesf[tensorboard]`。
> 每个 fold 独立写入 `experiments/results/<exp>/fold_<i>/tb_logs/`，
> `tensorboard --logdir experiments/results/<exp>` 可一次查看所有 fold 曲线。

> **数据泄露风险（test_metrics）**：训练过程中观察测试集曲线会把测试集信息泄露到
> 模型选择 / 超参调整环节，即便不做 early_stopping / checkpoint，研究人员也会基于
> 曲线隐式调优，导致最终报告的 test 指标乐观偏差。声明 `test_metrics` 时框架必然
> 发出一条 WARN——请仅在事后分析 / 消融研究 / 教学演示等接受风险的场景使用，日常
> 监测请用 `val_metrics`。

> **R24 约束**：`adaptation: transductive` 下 `checkpoint` 与 `early_stopping` 都不可出现（当前版本不支持 transductive 的模型选择指标）。

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
| `concat`（推荐） | 将所有折的预测和目标拼接后，一次性计算指标 | 每个样本恰好被测试一次（标准 k-fold） |
| `mean_std` | 每折独立计算指标，最终报告 mean ± std | 各折测试集重叠 / 规模不一致；或论文需要置信区间 |

---

## 按数据集维度切分（跨数据集常规 DL）

当 `dimension: dataset` 时，整个数据集作为一个组分配到 train/test。**dev6 要求显式 `assign`**（R13 覆盖所有声明的 aliases），且**验证集必须通过 `val_split` 配置**（R20 不允许主 `val_ratio`）。

### Holdout + 显式 assign

```yaml
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
  val_split:
    dimension: session
    val_ratio: 0.1
    shuffle: true
```

框架对 `assign.train` 指定的每个训练数据集独立调用 ValSplitter —— 各数据集保持 5-D 结构，在自身维度空间内按 `val_split.dimension` 切出验证集，避免不同数据集的 subject/session/recording 命名空间混淆。

### K-Fold（各数据集轮流做测试集）

```yaml
split:
  strategy: k-fold
  dimension: dataset
  k: -1                        # leave-one-dataset-out；或 k == len(datasets)
  val_split:
    dimension: session
    val_ratio: 0.1
    shuffle: true
```

> **R14 约束**：`dimension: dataset` + `k-fold` 时 `k` 必须为 `-1` 或等于数据集数量。
>
> **R3 约束**：`dimension: dataset` 需要 `len(datasets) ≥ 2`。

---

## 跨数据集对齐（alignment）

多数据集实验需要对齐通道和标签空间。`alignment` 位于顶层：

```yaml
alignment:
  channel: intersection        # 通道对齐方式（目前仅支持 intersection）
  label: true                  # 是否检查标签一致性，默认 true
```

**通道对齐**：`intersection` 保留所有数据集的通道交集，丢弃各自独有通道。对齐后通道顺序跟随第一个数据集的 `electrode_list`。前提是所有相关数据集在 `raw.yml` 中配置了 `electrode_list`；若交集为空，框架抛出 `ShapeMismatchError`。

**标签对齐**：开启后验证所有数据集的 `n_classes` 与 `numeric_to_semantic` 映射一致。框架同时校验 `n_samples`（采样点数）一致，不一致抛 `ShapeMismatchError`；采样率不一致（`n_samples` 一致但 Hz 不同）发出 warning。

> **R16**：单数据集实验（`len(datasets) == 1`）时若配置 `alignment`，框架忽略并 warning。

---

## UDA 无监督域自适应配置

设置 `mode: uda` 即可启用域自适应模式。dev6 的 UDA 配置是三层正交结构：

1. **域划分（DomainPartition）** `uda.domain` —— 把数据分为源域和目标域
2. **适应模式** `uda.adaptation` —— `transductive` 或 `inductive`
3. **域内划分** `uda.source.split`（ValSplit）与 `uda.target.split`（Split Block）

### 域维度（domain.dimension）

dev6 用 `domain.dimension` 统一描述跨 / 数据集内 UDA。**R28 白名单**仅允许：

| 配置值 | 含义 |
|--------|------|
| `dataset` | 跨数据集 UDA：整个数据集作为一个组 |
| `subject` | 数据集内 UDA：按被试划分源域 / 目标域 |
| `session` | 数据集内 UDA：按会话划分源域 / 目标域 |

（当前版本不支持 `recording` 与 `flatten` 作为 `domain.dimension`：前者跨 recording 域偏移过小，后者与域隔离语义冲突。）

### 适应模式（adaptation）

| 配置值 | 说明 |
|--------|------|
| `transductive` | 目标域全部数据同时用于无监督训练和最终测试（`target_train == target_test` 的拷贝，`target_val` 为空） |
| `inductive` | 目标域再按 `target.split` 切为 train/val/test，在训练集上无监督训练，在测试集上评估 |

### 跨数据集 UDA — Transductive

```yaml
mode: uda

datasets:
  bcic4_2a:
    name: BCIC4_2a_preprocessed
  bcic4_2b:
    name: BCIC4_2b_preprocessed

alignment:
  channel: intersection
  label: true

uda:
  domain:
    strategy: holdout
    dimension: dataset
    source: [bcic4_2a]           # holdout + dataset 维度必填（R8）
    target: bcic4_2b

  adaptation: transductive

  source:                        # 可选；省略则源域全量 train，source_val 为空
    split:
      dimension: recording       # ValSplit；必填 dimension，不可与 domain.dimension 相同（R11）
      val_ratio: 0.2
      shuffle: true

# transductive 模式下不配置 target.split（R5）
# 当前版本不支持 transductive 的 checkpoint / early_stopping（R24）

model: { name: DANN }
trainer: { name: dann_trainer }
training: { epochs: 200, batch_size: 128, optimizer: { name: adam, params: { lr: 0.001 }}}
evaluation: { metrics: [accuracy, f1_score] }
```

### 跨数据集 UDA — Inductive + K-Fold（嵌套折叠）

```yaml
mode: uda

uda:
  domain:
    strategy: k-fold
    dimension: dataset
    k: -1                        # leave-one-dataset-out
  adaptation: inductive
  source:
    split:
      dimension: subject
      val_ratio: 0.2
  target:
    split:                       # 完整 Split Block
      strategy: k-fold
      dimension: session
      k: 3
      val_ratio: 0.1             # 或 val_split 子块
```

> **嵌套折叠**：inductive + target k-fold 时 domain fold × target inner fold 会展平为一维列表，`fold_info` 同时携带 `domain_fold` 与 `inner_fold`，方便 TensorBoard / 查询时回溯层次。

### 数据集内 UDA — Transductive + LOOCV

```yaml
mode: uda

datasets:
  main:
    name: seed_preprocessed

uda:
  domain:
    strategy: k-fold
    dimension: subject
    k: -1                        # 每个被试轮流做 target
  adaptation: transductive
  source:
    split:
      dimension: recording
      val_ratio: 0.15
```

### 数据集内 UDA — Inductive + Holdout

```yaml
mode: uda

datasets:
  main:
    name: seed_preprocessed

uda:
  domain:
    strategy: holdout
    dimension: subject
    target_count: 1              # 或 target_ratio: 0.2（R12 互斥）
  adaptation: inductive
  source:
    split:
      dimension: recording
      val_ratio: 0.15
  target:
    split:
      strategy: holdout
      dimension: recording       # R11：必须 ≠ domain.dimension
      train_ratio: 0.7
      val_ratio: 0.1
      test_ratio: 0.2
```

### 约束提示（UDA）

- **R9 / R25**：`domain.dimension ∈ {subject, session}` 要求 `len(datasets) == 1`；`len(datasets) > 1` 强制 `domain.dimension: dataset`
- **R10**：`source.split` 只能含 `dimension / val_ratio / shuffle`（ValSplit 语义），不能出现 `strategy / train_ratio / test_ratio`
- **R11**：`domain.dimension` 不可与 `source.split.dimension` 或 `target.split.dimension` 相同
- **R4 / R5**：`adaptation: inductive` 下 `target.split` 必填；`transductive` 下 `target.split` 不可出现
- **R12**：holdout + subject/session 时 `target_count` 与 `target_ratio` 互斥
- **R24**：`transductive` 下禁止 `training.early_stopping` 与 `training.checkpoint`

### UDA 的 Batch 结构

UDA 模式下，框架自动接线的批次字典为：

```python
# train batch
batch = {
    "source": (source_data, source_labels),      # 有标签训练
    "target": (target_data, target_labels),      # 目标域数据；标签不应在训练中使用
}
# val batch（若配置源域 / 目标域 val）
batch = {
    "source_val": (...),
    "target_val": (...),
}
# test batch（目标域测试）
batch = {
    "main": (target_test_data, target_test_labels),
}
```

Trainer 的 `training_step` 按这些键取出数据，实现 MMD、对抗训练等域适应算法。

---

## 实验 YAML 常见问题

**Q：组件名找不到怎么办？**
A：检查 `project.yml` 中的 `models` 和 `trainers` 块，确认键名与实验 YAML 中的 `name` 字段一致，且 entrypoint 路径正确。也可以运行 `uesf project info` 查看当前项目可用的所有组件。

**Q：k-fold 的 `val_ratio` 是每折中还是整体中的比例？**
A：整体中的比例。框架按 `val_ratio / (1 - 1/k)` 自动换算为每折训练部分中的比例。例如 `k=5, val_ratio=0.1` → 每折测试集占 20%、验证集占 10%、训练集占 70%。

**Q：想在 val 与 test 上使用不同维度切分，怎么配置？**
A：使用 `val_split` 子块。例如主 `dimension: subject`（跨被试泛化测试）+ `val_split.dimension: session`（折内按会话切验证集）。注意 R19 / R30 约束。

**Q：跨数据集实验为什么不能用主 `val_ratio`？**
A：验证集无法在 `dataset` 维度上切分（否则整个验证数据集都会属于另一个 alias）。R20 要求 `dimension: dataset` 时通过 `val_split` 在每个训练数据集内部独立切出验证集。

**Q：单数据集用 `scope: global` 和 `scope: per_dataset` 有什么区别？**
A：没有。`global` 的"合并训练数据集" 在单数据集下退化为单个参与者，行为与 `per_dataset` 完全一致。

**Q：scheduler 的参数名从哪里查？**
A：参数名与 PyTorch 官方文档完全一致，参见[内置组件列表](../reference/04_builtin_components.md)中的调度器对应关系。
