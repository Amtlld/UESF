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
        → runner.run() → fold_result
    → aggregate fold results → final_results
```

> **val_ratio 换算**：`ConfigValidator.normalize()` 在 K-Fold 场景中自动将用户配置的 `val_ratio`（整体比例）换算为 `val_ratio_in_train = val_ratio / (1 - 1/k)`，传给 `KFoldSplitter`。用户无感知。

**自动通道映射规则（Regular 模式）**：
- 单数据集：channel name = `"main"`
- 多数据集 + `dimension=dataset`：所有训练集数据合并到 `"main"` 通道

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
        → runner.run() → fold_result
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
