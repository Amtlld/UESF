# 四、执行流程详解

## 4.1 Regular 模式流程

```
load datasets → alignment (if multi-dataset)
    → create_splitter(split_config)
    → splitter.split(data) → folds: list[SplitResult]
    → for each fold:
        → slice data by indices
        → fit transforms on train, apply to all
        → build dataloaders (auto-channel)
        → init fresh model + trainer
        → create_logger(logging_config, fold_dir/tb_logs)  # 可选
        → runner.run(logger=logger) → fold_result
    → aggregate fold results → final_results
```

> **val_ratio 换算**：`ConfigValidator.normalize()` 在 K-Fold 场景中自动将用户配置的 `val_ratio`（整体比例）换算为 `val_ratio_in_train = val_ratio / (1 - 1/k)`，传给 `KFoldSplitter`。用户无感知。

> **Transform fit 策略（Regular 模式）**：仅在 train 集上 fit，apply 到 train/val/test。验证集和测试集不参与 fit，以避免数据泄漏。

**自动通道映射规则（Regular 模式）**：
- 单数据集：channel name = `"main"`
- 多数据集 + `dimension=dataset`：所有训练集数据合并到 `"main"` 通道

> **多数据集合并策略（`dimension=dataset`）**：
> - 各数据集将前三维展平：`[n_subjects, n_sessions, n_recordings, n_channels, n_samples]` → `[N_i, n_channels, n_samples]`
> - 属于同一 phase 的数据集沿 axis=0 concat：`[sum(N_i), n_channels, n_samples]` → 构建 `"main"` 通道 dataloader
> - `DatasetLevelSplitter` 返回的是 alias 级别的划分（哪些数据集做 train / test），不涉及 sample 级索引
> - `get_groups()` 不适用于 `dimension=dataset`，仅在数据集内部按 subject / session / recording 分组时使用
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
        → slice source/target data by indices
        → fit transforms on source_train, apply to all
        → build dataloaders:
            train: {"source": src_train, "target": tgt_train}
            val:   {"source_val": src_val, "target_val": tgt_val}
            test:  {"main": tgt_test}
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
> - 仅在 **source_train** 上 fit，apply 到所有通道（source_train / source_val / target_train / target_val / target_test）
> - 目标域数据**不参与** fit，以避免目标域统计信息泄漏到训练阶段
> - 跨数据集场景下，各数据集的数据使用同一组 fit 参数（源域训练集统计量）进行 transform

## 4.3 ValSplit 维度分组处理

`ValSplitter` 在切出验证集时按 `dimension` 分组，确保维度隔离：

```python
# 伪代码
groups = get_groups(data, dimension)  # 按维度分组
n = len(groups)
n_val = max(1, round(n * val_ratio))

if shuffle:
    rng.shuffle(group_indices)

val_groups = groups[-n_val:]          # 尾部 n_val 个组整体进入 val
train_groups = groups[:-n_val]        # 其余进入 train
```

例如 `source.split.dimension: session`，源域有 6 个 session，`val_ratio: 0.2`：
→ 1 个 session 整体作为验证集，5 个 session 作为训练集。
同一 session 内的所有 recording 不会被拆散到 train 和 val 两端。

## 4.4 嵌套折叠处理（Inductive + Target K-Fold）

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

## 4.5 结果聚合策略

`k_fold_aggregation` 的两种模式：

| 模式 | 语义 | 适用场景 |
|:-----|:-----|:---------|
| `concat` | 将所有 fold 的预测结果拼接后，在全量预测上计算一次 metric | 每个样本恰好被测试一次的场景（如标准 k-fold） |
| `mean_std` | 每个 fold 独立计算 metric，最终报告 mean ± std | 各 fold 测试集可能重叠或规模不一致的场景 |

**聚合层次**：

- **Regular k-fold**：在所有 fold 上聚合
- **UDA 嵌套折叠**（domain fold × inner fold）：展平后视为一维 fold 列表，统一聚合。`fold_info` 保留层次信息供用户按需分析
- **跨数据集 `dimension=dataset`**：每个 fold 的测试集为单个数据集，聚合时同时报告各 fold（即各测试数据集）的独立 metric 和整体聚合 metric

## 4.6 随机性管理

- 全局 `seed` 作为基础种子，所有随机操作从该种子派生
- 多层划分时，每层使用独立的 `numpy.random.Generator`，种子按 `seed + layer_offset` 派生，确保各层随机性独立且可复现：
  - domain split: `seed + 0`
  - source inner split: `seed + 1`
  - target inner split: `seed + 2`
- 每个 fold 内 model 权重初始化和 dataloader shuffle 使用 `seed + fold_idx` 派生，确保各 fold 间随机性不同但可复现
- **可复现保证**：相同配置 + 相同 seed → 完全可复现的划分结果和训练过程（前提：单 GPU、确定性算法模式）

## 4.7 多 fold 场景下的 Checkpoint 与日志

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

## 4.8 错误恢复与资源清理

- **fold 级隔离**：单个 fold 训练失败时，捕获异常并记录到该 fold 的 `FoldResult`（标记为 failed），继续执行后续 fold。实验最终结果中包含失败 fold 的信息，聚合指标仅基于成功的 fold 计算。若所有 fold 均失败，实验整体标记为 failed。
- **Logger 安全关闭**：Logger 使用 context manager 模式（`__enter__` / `__exit__`），确保异常时也能 flush 并关闭。`runner.run()` 中改用 `with logger:` 包裹训练循环。`TrainingLogger` 协议新增 `__enter__` 和 `__exit__` 方法。
- **DB 状态**：实验开始时标记为 `running`，每个 fold 完成/失败时更新进度，实验结束时根据 fold 结果标记为 `completed` 或 `partial`（部分 fold 失败）或 `failed`（全部失败）。
