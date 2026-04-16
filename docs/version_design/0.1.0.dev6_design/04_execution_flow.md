# 四、执行流程详解

## 4.1 Regular 模式流程

```
load datasets → alignment (if multi-dataset)
    → create_splitter(split_config)
    → splitter.split(data) → folds: list[SplitResult]
    → for each fold:
        → for phase in [train, val, test]:
            prepare_channel_data(split_result, dataset_cache, phase)
                → channel_data, channel_labels
        → apply transforms (scope-dependent, see below)
        → for phase in [train, val, test]:
            DataloaderBuilder.build(channel_data, channel_labels, batch_size, shuffle)
                → DataLoader
        → init fresh model + trainer
        → create_logger(logging_config, fold_dir/tb_logs)  # 可选
        → runner.run(logger=logger) → fold_result
    → aggregate fold results → final_results
```

> **val_ratio 换算**：无 `val_split` 时，`ConfigValidator.normalize()` 在 K-Fold 场景中自动将用户配置的 `val_ratio`（整体比例）换算为 `val_ratio_in_train = val_ratio / (1 - 1/k)`，传给 `KFoldSplitter`。用户无感知。

> **val_split 独立划分**：若配置了 `val_split`，splitter 的主划分仅产生 train/test。splitter 内部使用 `ValSplitter`（基于 `val_split.dimension`）从每折的 train 中切出 val，最终 `SplitResult` 仍包含 train/val/test 三组索引。对于 `dimension=dataset` 场景，`RegularExecutionStrategy` 先将训练数据集合并，再用 `ValSplitter` 按 `val_split.dimension` 分组切出验证集。

> **Transform fit 策略（Regular 模式）**：
> - `scope: per_dataset`（默认）：对每个数据集独立执行——fit 仅在该数据集的 train 部分，transform 在该数据集的 train/val/test 部分。无论单数据集还是多数据集，fit 始终仅在训练集上进行，验证集和测试集不参与 fit。跨数据集场景下，各数据集使用各自的统计量。
> - `scope: global`：所有数据集的 train 数据合并后 fit，统一 transform 到所有 train/val/test 数据。仅 Regular 模式可用。
> - 无论何种 scope，验证集和测试集均不参与 fit，以避免数据泄漏。

**自动通道映射规则（Regular 模式）**：
- 单数据集：channel name = `"main"`
- 多数据集 + `dimension=dataset`：所有训练集数据合并到 `"main"` 通道

> **多数据集合并策略（`dimension=dataset`）**：
> - 各数据集将前三维展平：`[n_subjects, n_sessions, n_recordings, n_channels, n_samples]` → `[N_i, n_channels, n_samples]`
> - 属于同一 phase 的数据集沿 axis=0 concat：`[sum(N_i), n_channels, n_samples]` → 构建 `"main"` 通道 dataloader
> - `DatasetLevelSplitter` 返回的是 alias 级别的划分（哪些数据集做 train / test），不涉及 sample 级索引
> - `get_groups()` 不适用于 `dimension=dataset`，仅在数据集内部按 subject / session / recording / flatten 分组时使用
> - 合并前提：
>   - `n_channels`：由 `alignment.channel: intersection` 保证一致
>   - `n_samples`（采样点数）：由用户在预处理阶段保证一致（统一采样率和时间窗口），框架在合并时校验，不一致则抛出 `ShapeMismatchError`
>   - 采样率：框架检查各数据集元信息中的采样率，不一致时发出 warning（采样点数可能一致但采样率不同，意味着时间窗口长度不同）

## 4.2 UDA 模式流程

```
load datasets → alignment (if cross-dataset)
    → create_uda_orchestrator(uda_config)
    → orchestrator.split(dataset_cache) → uda_folds: list[UDASplitResult]
    → for each uda_fold:
        → prepare_uda_channel_data(uda_split, dataset_cache)
            → {"train": (channel_data, channel_labels),
               "val":   (channel_data, channel_labels),
               "test":  (channel_data, channel_labels)}
        → apply transforms per dataset (see below)
        → for phase in [train, val, test]:
            DataloaderBuilder.build(channel_data, channel_labels, batch_size, shuffle)
                → DataLoader
        → init fresh model + trainer
        → create_logger(logging_config, fold_dir/tb_logs)  # 可选
        → runner.run(logger=logger) → fold_result
    → aggregate results → final_results
```

**自动通道映射规则（UDA 模式）**：

| Dataloader | 通道 | 内容 |
|:-----------|:-----|:-----|
| train | `"source"` | 源域训练集（有标签） |
| train | `"target"` | 目标域训练集（无标签/有标签，由 Trainer 决定如何使用） |
| val | `"source_val"` | 源域验证集 |
| val | `"target_val"` | 目标域验证集（用于信息熵/伪标签稳定性监控） |
| test | `"main"` | 目标域测试集（transductive 时与 target train 相同数据） |

> Trainer 的 `training_step(batch)` 签名不变。UDA Trainer 自行从 `batch["source"]` 和 `batch["target"]` 中取数据，实现域适应逻辑。

> **Transform fit 策略（UDA 模式）**：
> UDA 模式始终使用 per_dataset 语义——各数据集内独立 fit & transform：
> - **源域各数据集**：fit on 该数据集的 source_train 部分，transform 该数据集的 source_train / source_val
> - **目标域各数据集（inductive）**：fit on 该数据集的 target_train 部分，transform 该数据集的 target_train / target_val / target_test
> - **目标域各数据集（transductive）**：fit on 该数据集的目标域全量数据，transform 同一份数据。当前版本不支持 transductive 下的 checkpoint/早停
> - **单数据集 UDA**：源域 fit on source_train，目标域 fit on 对应部分（行为一致，只是只有一个数据集）
> - 每个数据集使用各自的统计量，不同数据集之间互不影响

## 4.3 跨数据集索引合并与还原

当 UDA `domain.dimension=dataset` 且源域包含多个数据集时，ValSplitter 需要在合并后的数据上统一按维度分组，再将切分结果还原回各 alias 的局部索引存入 `UDASplitResult`。

**为什么需要合并**：ValSplitter 只接受单一 ndarray，跨数据集场景必须先将多个 alias 的数据展平合并，才能在统一的维度空间中分组。

**为什么切分后还要还原**：`UDASplitResult` 的字段类型为 `dict[str, np.ndarray]`（alias → 局部索引），这保证了：
- 自包含性：UDASplitResult + dataset_cache 即可还原数据，不依赖合并顺序等隐式信息
- 下游统一性：`prepare_uda_channel_data` 对单 alias（数据集内 UDA）和多 alias（跨数据集 UDA）使用同一套循环逻辑

两次合并操作的数据不同——第一次合并全量数据用于分组决策，第二次在下游合并切片后的子集用于构建训练数据。还原步骤是两次操作之间的桥梁。

**UDAOrchestrator 内部流程（以 source split 为例）**：

```python
# 伪代码

# 1. 各 alias 展平为 [N_i, C, T]，记录偏移量
offset_map = {}   # alias → (start, end)
arrays = []
cursor = 0
for alias in source_aliases:
    flat = flatten_3d(dataset_cache[alias])  # [N_i, C, T]
    offset_map[alias] = (cursor, cursor + len(flat))
    arrays.append(flat)
    cursor += len(flat)

merged = np.concatenate(arrays, axis=0)  # [sum(N_i), C, T]

# 2. 在合并数据上执行 ValSplit（跨 alias 统一按维度分组）
split_result = source_splitter.split(merged)
# split_result.train_indices / val_indices 是合并后的全局索引

# 3. 将全局索引还原回各 alias 的局部索引
source_train = {}
source_val = {}
for alias, (start, end) in offset_map.items():
    mask_train = (split_result.train_indices >= start) & (split_result.train_indices < end)
    source_train[alias] = split_result.train_indices[mask_train] - start

    mask_val = (split_result.val_indices >= start) & (split_result.val_indices < end)
    source_val[alias] = split_result.val_indices[mask_val] - start
```

> **数据集内 UDA**（`domain.dimension=subject/session`）：dict 中只有单个 alias，无需合并和还原，ValSplitter 直接在该 alias 的数据上操作。

## 4.4 ValSplit 维度分组处理

`ValSplitter` 在切出验证集时按 `dimension` 分组，确保维度隔离：

```python
# 伪代码
groups = get_groups(data, dimension)  # 按维度分组，按维度原始顺序排列
n = len(groups)
n_val = max(1, round(n * val_ratio))

if shuffle:
    rng.shuffle(group_indices)        # 随机打乱后再切分

# shuffle=False 时按原始顺序：前部为 train，尾部为 val
train_groups = groups[:-n_val]        # 前 n-n_val 个组进入 train
val_groups = groups[-n_val:]          # 尾部 n_val 个组整体进入 val
```

例如 `source.split.dimension: session`，源域有 6 个 session，`val_ratio: 0.2`：
→ 1 个 session 整体作为验证集，5 个 session 作为训练集。
同一 session 内的所有 recording 不会被拆散到 train 和 val 两端。

## 4.5 嵌套折叠处理（Inductive + Target K-Fold）

当 UDA inductive 模式的 target split 使用 k-fold 时，形成两层循环：

```
total_folds = domain_folds × target_inner_folds
```

`UDAOrchestrator.split()` 将嵌套折叠展平为一维列表：

```python
# 伪代码
all_splits = []
for d_idx, domain_fold in enumerate(domain_splitter.split(...)):
    source_split = source_splitter.split(source_data)     # 单个 SplitResult
    target_splits = target_splitter.split(target_data)     # k 个 SplitResult
    for t_idx, target_split in enumerate(target_splits):
        all_splits.append(UDASplitResult(
            source_train = apply_indices(source_split.train_indices, domain_fold.source),
            source_val   = apply_indices(source_split.val_indices, domain_fold.source),
            target_train = apply_indices(target_split.train_indices, domain_fold.target),
            target_val   = apply_indices(target_split.val_indices, domain_fold.target),
            target_test  = apply_indices(target_split.test_indices, domain_fold.target),
            fold_info    = {
                "domain_fold": d_idx,
                "inner_fold": t_idx,
                **domain_fold.fold_info,
            },
        ))
return all_splits
```

> **source split 共享**：同一 domain fold 内，源域的 train/val 划分在所有 target inner fold 间共享（即源域只切分一次）。这是有意为之——源域划分与目标域内部折叠正交，每个 inner fold 仅改变目标域的 train/val/test 分配。

## 4.6 结果聚合策略

`k_fold_aggregation` 的两种模式：

| 模式 | 语义 | 适用场景 |
|:-----|:-----|:---------|
| `concat` | 将所有 fold 的预测结果拼接后，在全量预测上计算一次 metric | 每个样本恰好被测试一次的场景（如标准 k-fold） |
| `mean_std` | 每个 fold 独立计算 metric，最终报告 mean ± std | 各 fold 测试集可能重叠或规模不一致的场景 |

**聚合层次**：

- **Regular k-fold**：在所有 fold 上聚合
- **UDA 嵌套折叠**（domain fold × inner fold）：展平后视为一维 fold 列表，统一聚合。`fold_info` 保留层次信息供用户按需分析
- **跨数据集 `dimension=dataset`**：每个 fold 的测试集为单个数据集，聚合时同时报告各 fold（即各测试数据集）的独立 metric 和整体聚合 metric

## 4.7 随机性管理

- 全局 `seed` 作为基础种子，所有随机操作从该种子派生
- 多层划分时，每层使用独立的 `numpy.random.Generator`，种子按 `seed + layer_offset` 派生，确保各层随机性独立且可复现：
  - domain split: `seed + 0`
  - source inner split: `seed + 1`
  - target inner split: `seed + 2`
- 每个 fold 内 model 权重初始化和 dataloader shuffle 使用 `seed + fold_idx` 派生，确保各 fold 间随机性不同但可复现
- **可复现保证**：相同配置 + 相同 seed → 完全可复现的划分结果和训练过程（前提：单 GPU、确定性算法模式）

## 4.8 多 fold 场景下的 Checkpoint 与日志

每个 fold 的输出目录结构：

```
{output_dir}/fold_{idx}/
├── best.pt                   # best checkpoint
└── tb_logs/                  # TensorBoard 日志（仅在配置 logging 时生成）
    └── events.out.tfevents.*
```

- 每个 fold 独立保存 best checkpoint，路径格式：`{output_dir}/fold_{idx}/best.pt`
- `checkpoint_metric` 在每个 fold 内独立生效（基于该 fold 的 val metric 选择最优模型）
- 实验结束后不保留所有 fold 的 checkpoint，仅保留 best（可通过配置开启全量保留）
- TensorBoard 日志每个 fold 独立写入，`tensorboard --logdir {output_dir}` 可一次查看所有 fold 的训练曲线

## 4.9 错误恢复与资源清理

- **fold 级隔离**：单个 fold 训练失败时，捕获异常并记录到该 fold 的 `FoldResult`（标记为 failed），继续执行后续 fold。实验最终结果中包含失败 fold 的信息，聚合指标仅基于成功的 fold 计算。若所有 fold 均失败，实验整体标记为 failed。
- **Logger 安全关闭**：Logger 使用 context manager 模式（`__enter__` / `__exit__`），确保异常时也能 flush 并关闭。`runner.run()` 中改用 `with logger:` 包裹训练循环。`TrainingLogger` 协议新增 `__enter__` 和 `__exit__` 方法。
- **DB 状态**：实验开始时标记为 `running`，每个 fold 完成/失败时更新进度，实验结束时根据 fold 结果标记为 `completed` 或 `partial`（部分 fold 失败）或 `failed`（全部失败）。
