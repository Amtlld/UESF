# Experiment Manager 详细设计

Experiment Manager 是 UESF 引擎的核心调度枢纽，其核心职责是将静态的配置解析为执行流。为了保证框架的绝对通用性（兼容普通分类乃至非对称架构的无监督域自适应等复杂场景），Experiment Manager 在设计上严格落实了**数据流与控制流解耦**的原则。

数据库表结构详见 [`experiments` 表](../03_storage/02_database_schema.md#experiments-表)。

## 1. 核心抽象与控制反转 (Inversion of Control)

- **数据集切分器 (Splitter)**: 框架内建了一系列切分策略模块（如 `K-Fold`, `Holdout`, `Leave-One-Out`），依据 `datasets.split` 配置动态生成每一折的训练/验证/测试数据集索引快照。尤其对于跨域 EEG 问题，切分器内置了基于分组维度的隔离设计 (`dimension: subject` 等)，从根源上防止时序信息的泄露。
- **多通道字典映射加载器 (Multi-channel Dataloader Interface)**: DataLoader Builder 在实例化时并不会简单合并并输出 `(X, y)` 元组。相对的，它会根据实验配置定义的通道名（如 `src_labeled`, `tgt_unlabeled`），平行地初始化多组 DataLoader，并在最上层使用联合迭代器 (Combined Iterator) 将一个 step 内获得的数据全部打包成一个字典结构。
- **训练委托 (Delegated Training Step)**: UESF 自带的基础 `Runner` 极度瘦身，不再去维护任何特定于某种算法的 Loss 计算流与梯度管理。在训练步 (`training_step`) 中，`Runner` 仅将上一步组装的多通道 `batch` 字典与优化器实例委托下发给当前挂载的 `Trainer`。**梯度的反向传播（`loss.backward()`）和优化器步进（`optimizer.step()`）由 Trainer 全权负责**，Runner 不介入任何优化过程。这一设计使得任何自定义的前沿 UDA 计算图、GAN 交替更新以及多阶段参数冻结等复杂优化策略得以完全地闭环隔离在用户的工程级代码内部。

## 2. 结果管理与断点系统

- **实验日志与指标追踪**: 原生集成标准日志库处理输出流，无缝衔接 `Weights & Biases` (W&B) 记录 Loss 收敛趋势及动态指标曲线。
- **模型检查点 (Checkpoints)**: 基于实验中给定的 `checkpoint_metric`（如 `val_f1_score`），评估阶段会触发预设的 Monitor Hook，自动存储表现最优权重的 `.pt` 或 `.pth` 快照至 `<project-dir>/experiments/results/<experiment-name>/checkpoints/` 下。

## 3. 实验配置语法结构

UESF 支持用户通过 YAML 进行灵活配置。下面是完整的 `<experiment_name>.yml` 准语法规范：

```yaml
# ====== 基础元信息 ======
name: <experiment-name>
description: <experiment-description>
seed: <int> # 随机种子，确保研究基准及数据划分的绝对可复现性

# ====== 组件挂载 (Registry Components) ======
model:
  name: <model-name>  # 对应 project.yml 中已注册的模型名
  params: { ... }     # 针对该模型初始化接收的 kwargs 字典

trainer:
  name: <trainer-name> # 对应 project.yml 中已注册的训练器名
  params: { ... }      # UDA 中的独有超参 (如 GRL_lambda 起始权重) 可被隔离于此

# ====== 数据集定义与切分策略 (Datasets & Splitters) ======
datasets:
  <dataset-alias-A>:  # 给参与实验的数据集起的临时别名
    name: <preprocessed-dataset-name-1> 
    split:
      strategy: <strategy>   # "holdout" 或 "k-fold"
      dimension: <dimension> # "subject", "dataset", "record", 或 "none"
      
      # [k-fold 专有]:
      k-folds: <int>         # 若为 "total" 或 -1 则表示留一法 (LOOCV)
      val_ratio_in_train: <float> # 每折内部用于 Early Stop 的比例
      
      # [holdout 专有]:
      train_ratio: <float>   
      val_ratio: <float>
      test_ratio: <float>

  # ... 多源域时可继续挂载 <dataset-alias-B>
  
# ====== 通道拆分与映射层 (Multi-channel Mapping) ======
# 将上面 Splitter 产生的切分相 (如 .train) 自由装配送入网络的哪一数据流入口
dataloaders:
  train:
    <channel_name>: "<dataset-alias>.<split-phase>" 
    # 例如： src_labeled: "alias_A.train"
  val:
    <channel_name>: "<dataset-alias>.val"
  test:
    <channel_name>: "<dataset-alias>.test"

# ====== 训练超参数 (Overrides) ======
# 注意：若 Trainer.configure_optimizers() 返回了非 None 值，
#       下方 optimizer / learning_rate / weight_decay / scheduler 将被忽略，
#       系统会在日志中发出 Warning 提示。
training:
  epochs: <int>
  batch_size: <int>
  learning_rate: <float>
  optimizer: <optimizer>          # 优化器名称，如 "adam", "sgd", "adamw" 等。仅支持 PyTorch 内置优化器，名称到具体类的映射规则由另外的文档说明。若需自定义优化器，用户可通过继承 BaseTrainer 并重写 configure_optimizers() 方法实现
  weight_decay: <float>           # (可选) 权重衰减系数，默认 0.0
  gradient_clip:                  # (可选) 梯度裁剪配置
    max_norm: <float>             # 梯度最大范数，如 1.0
    norm_type: <int>              # (可选) 范数类型，默认 2
  scheduler:                      # (可选) 学习率调度器配置
    name: <scheduler-name>        # 调度器名称，如 "cosine_annealing", "step_lr", "reduce_on_plateau" 等
    params: { ... }               # 调度器参数（如 {T_max: 50, eta_min: 1e-6}）
  early_stopping:                 # (可选) 早停策略配置
    monitor: <metric-name>        # 监控的指标名称，如 "val_loss", "val_accuracy"
    patience: <int>               # 指标无改善的容忍轮数
    min_delta: <float>            # (可选) 视为改善的最小变化量，默认 0.0
    mode: <mode>                  # (可选) "min"（指标越小越好）或 "max"（指标越大越好），默认 "min"

# ====== 宏观评估与记录 ======
evaluation:
  metrics: [<metric-name>]  # e.g., ["accuracy", "f1_score", ...]
  k_fold_aggregation: "concat" # 针对交叉验证时多折结果的汇总策略：可选 "concat"（全局拼接计算，推荐）或 "mean_std"（各折取平均与方差）

logging:
  use_wandb: <bool>
  checkpoint_metric: <metric-name>
```

## 4. 评估模块与指标体系 (Evaluator & Metrics)

UESF 提出了"数据的延迟聚合与一次性结算"指标计算方案，并通过类似于模型注入的方式，使得评价指标（Metric）也可以被灵活地定义与调用。

### 4.1 Epoch-level Aggregation (Epoch级聚合延迟验证)

为了避免"批次数值求平均"所导致的非平滑指标计算失真问题，UESF 采取了延时计算的设计。在底层的业务控制流中：
- `Trainer.validation_step()` 返回时不会立刻计算分类准确度，而是仅回传该批次的原始预测张量（`preds`）与对应的目标标签（`targets`）。
- UESF 底层 `Runner` 在内存/显存中自动拼接（Concat）并缓存一整个验证或测试周期的全部批次张量结果。
- 在 Epoch 周期结尾，由框架独立的 **`Evaluator`（评估执行组件）** 对这个无切分、最完整的表现全量预测张量执行一次性的评价运算。

> **针对交叉验证（如 K-Fold）的综合评估策略规范：**
> 在进行多折验证时，由于各折间数据量可能存在差异或标签分布不平衡，针对指标的最终取值，UESF 在实验配置的 `evaluation: k_fold_aggregation` 中提供参数接口供用户自主选择：
> - **`concat` (大满贯模式, 推荐)**：系统会将验证集中生成的完整 `preds`、`targets` 张量全局拼接聚合，并在完全落幕后执行一次全维度指标计算运算；
> - **`mean_std` (独立平均模式)**：维持传统作法，系统独立得出单折精度，最终求取所有单折指标集合的均值（Mean）与标准差（Std）。
> 这样设计一方面保障了由于各折不平衡造成的统计失真的规避情况，同时也能灵活适应需要独立标准差作为置信区间的传统发文要求。

### 4.2 统一 Metric 接口规范

任何在实验中挂载的指标函数（无论是内置封装的还是用户自定义的），都遵循统一的函数签名规范：
```python
import torch
from typing import Dict, Any

def my_metric_func(preds: torch.Tensor, targets: torch.Tensor, **kwargs) -> float | Dict[str, Any]:
    """
    统一的指标计算委托接口。

    :param preds: 聚合了完整 Epoch 流程数据的模型预测结局张量。
    :param targets: 对应的真实标签张量。
    :param kwargs: 实验 YAML 配置中通过字典键名注入的控制参数（如 average='macro'）。
    :return: 返回纯数值（float），或一个可直接 JSON 序列化的多层指标字典。
             所有的返回数据通过 Evaluator 都会被记录并打包放入数据库。
    """
    pass
```

> 内置指标的名称、参数及其含义由专项文档另行规定。

### 4.3 指标组件管理机制

指标（Metric）采用与模型、训练器相同的三类组件管理方式：

| 类型 | 标识 | 说明 |
|------|------|------|
| 内置指标 | `EMBEDDED` | UESF 开发者维护的标准评估指标（如准确率、F1 等） |
| 已注册指标 | `REGISTERED` | 用户在 `project.yml` 中通过 `entrypoint` 注册的项目级自定义指标 |
| 全局指标 | `GLOBAL` | 通过 `uesf metric add` 导入的系统级共享指标 |

所有指标元信息（含源码快照）统一记录在数据库 `metrics` 表中，详见 [`metrics` 表](../03_storage/02_database_schema.md#metrics-表)。

在 `project.yml` 中注册项目级自定义指标的示例：
```yaml
metrics:
  my_custom_score:
    entrypoint: "./src/utils/custom_metric.py:my_metric_func"
```

由于评估结果返回被约束为基础标量或是标准的字典对象，这能保障系统可不加障碍地将数据转化为 JSON 序列，存进 `experiments` 状态追踪表的 `results` 快照块中。

**REGISTERED 指标的源码变更自动检测**与模型、训练器机制完全一致：在 `uesf experiment run` 的组件初始化阶段检测源文件哈希，若发生变更则将旧记录归档为 `<name>_<sha256前8位>`（`is_obsolete = 1`），以原始名称创建新记录并更新快照。详见 [Model Manager §3.1](03_model_manager.md#31-registered-模型的源码变更自动检测与重新注册)。

UESF 支持通过以下命令管理全局指标：`uesf metric add` / `uesf metric list` / `uesf metric remove` / `uesf metric edit`。`uesf metric list` 默认隐藏过时记录，可通过 `--show-obsolete` 参数显示。

## 5. 添加、删除和查询实验

UESF 提供命令，允许用户添加、删除和查询实验。

1. 添加实验：用户可以添加空白实验或从现有实验添加新实验
  - 添加空白实验：UESF 创建空白实验配置文件，用户需自行指定全部参数。
  - 从现有实验添加新实验：UESF 复制现有实验的配置产生新实验配置，用户可以在此基础上进行修改
  - 用户使用以上两条命令创建实验时，UESF 提供命令参数供用户指定实验名称等配置。若用户没有在命令中指定实验名，UESF 使用项目名称和当前系统时间拼接成为新实验名
2. 删除实验：用户可以删除实验或仅删除实验结果
  - 删除实验：该命令将一并删除实验配置和实验结果（包括实验产生的检查点模型等）
  - 仅删除实验结果：该命令仅删除实验结果（包括实验产生的检查点模型等）
3. 查询实验：用户可以查询实验结果。在命令参数中，用户可以指定自己关注的指标（如精度、F1分数、查全率、查准率、AUC-ROC、混淆矩阵等）
