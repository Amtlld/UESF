# 七、配置校验规则汇总

| 编号 | 规则 | 校验内容 |
|:-----|:-----|:---------|
| R1 | mode/split 互斥 | `mode=regular` ↔ 有 `split`，无 `uda` |
| R2 | mode/uda 互斥 | `mode=uda` ↔ 有 `uda`，无 `split` |
| R3 | 数据集级需多数据集 | `split.dimension=dataset` → `len(datasets) > 1` |
| R4 | inductive 需测试集 | `adaptation=inductive` → `target.split` 中有 `test_ratio`（holdout）或 `k`（k-fold） |
| R5 | transductive 无 target.split | `adaptation=transductive` → `target.split` 不可出现 |
| R6 | holdout ratio 求和 | 无 `val_split` 时：`train_ratio + val_ratio + test_ratio` 之和满足 `abs(sum - 1.0) < 1e-4`；有 `val_split` 时：`train_ratio + test_ratio` 之和满足 `abs(sum - 1.0) < 1e-4` |
| R7 | k-fold 合法 k 值 | k-fold 的 `k > 1` 或 `k == -1` |
| R8 | 跨数据集域需指定 | `domain.dimension=dataset` + `strategy=holdout` → 必须指定 `source` 和 `target` |
| R9 | 数据集内域需单数据集 | `domain.dimension=subject\|session` → `len(datasets) == 1`（不同数据集的 subject/session 编号可能冲突，无法统一分组）。注：此规则为 R25 UDA 部分的逆否等价形式，保留以提供更直接的错误提示 |
| R10 | 源域无测试集 | `source.split`（若存在）不含 `strategy`、`test_ratio`、`train_ratio`。`source` 块整体可选，省略时源域全量数据作为训练集，`source_val` 为空 |
| R11 | 域维度与内部维度不同 | `domain.dimension` ≠ `source.split.dimension`（若 `source.split` 存在）且 `domain.dimension` ≠ `target.split.dimension`（若 `target.split` 存在） |
| R12 | target 计数互斥 | `domain` 中 `target_count` 和 `target_ratio` 不可同时出现 |
| R13 | dataset holdout 需 assign | `split.dimension=dataset` + `strategy=holdout` → `assign` 必填，且 `assign` 中的 alias 须覆盖所有声明的 datasets |
| R14 | dataset k-fold 的 k 值约束 | `split.dimension=dataset` + `strategy=k-fold` → `k == -1` 或 `k == len(datasets)` |
| R15 | val_ratio 值域 | `0 <= val_ratio < 1`；k-fold 换算后 `val_ratio_in_train < 1` |
| R16 | 单数据集 alignment 处理 | `len(datasets) == 1` 时若存在 `alignment` 块，忽略并发出 warning |
| R17 | logging.backend 合法值 | `training.logging.backend` 仅允许 `"tensorboard"`，其他值抛出 `TypeMismatchError` |
| R17b | log_graph 类型 | `training.logging.log_graph`（若声明）必须为 `bool`，否则抛出 `TypeMismatchError` |
| R17c | log_lr 类型 | `training.logging.log_lr`（若声明）必须为 `bool`，否则抛出 `TypeMismatchError` |
| R18 | log_every_n_epochs 正整数 | `training.logging.log_every_n_epochs` 必须为正整数（> 0），否则抛出 `TypeMismatchError` |
| R18a | log_every_n_steps 正整数 | `training.logging.log_every_n_steps`（若声明）必须为正整数（> 0）。未声明 / 为 `null` 时关闭 step 级日志 |
| R19 | val_split 与 val_ratio 互斥 | `split.val_split` 存在时，`split.val_ratio` 不可出现；反之亦然 |
| R20 | dataset 维度必须用 val_split | `split.dimension=dataset` 时，若需要验证集则必须使用 `val_split`，不可使用主划分的 `val_ratio` |
| R21 | val_split.val_ratio 值域 | `0 < val_split.val_ratio < 1`（val_split 存在时 val_ratio 必须 > 0，否则不应配置 val_split） |
| R22 | transform scope 合法值 | `transforms[].scope` 仅允许 `"per_dataset"` 或 `"global"`，其他值抛出 `TypeMismatchError` |
| R23 | UDA 禁止 global scope | `mode=uda` 时，`transforms[].scope` 不可为 `"global"`（UDA 跨数据集场景下各数据集必须独立标准化） |
| R24 | transductive 禁止 checkpoint/早停 | `adaptation=transductive` 时，`training.early_stopping` 和 `training.checkpoint` 不可出现（当前版本不支持 transductive 下的 checkpoint/早停逻辑） |
| R25 | 多数据集需 dataset 维度 | `mode=regular` 且 `len(datasets) > 1` → `split.dimension` 必须为 `dataset`。多数据集场景下各数据集的 subject/session/recording 结构可能不同，无法在非 dataset 维度上统一分组。同理 `mode=uda` 且 `len(datasets) > 1` → `domain.dimension` 必须为 `dataset` |
| R26 | flatten 维度禁止 shuffle=False | 任意划分原语（Split Block / ValSplit / val_split 子块）中 `dimension=flatten` 时，`shuffle` 必须为 `true`。flatten 语义为"将 (subject, session, recording) 展平后随机切分"，顺序遍历无明确物理意义，抛出 `TypeMismatchError` |
| R27 | shuffle 默认值 | 所有划分原语（Split Block / ValSplit / DomainPartition）中 `shuffle` 字段未显式指定时，默认为 `true`。由 `ConfigValidator.normalize()` 填充 |
| R28 | DomainPartition 维度白名单 | `uda.domain.dimension` 仅允许 `dataset | subject | session`。当前版本不支持 `recording`（跨 recording 域偏移小，UDA 意义有限）和 `flatten`（与域划分的隔离语义冲突）。其他值抛出 `TypeMismatchError` |
| R29 | 跨数据集 transforms 管线一致 | 多数据集场景（`split.dimension=dataset` 或 `uda.domain.dimension=dataset`）下，参与同一次 `apply_transforms`/`apply_transforms_uda` 调用的各 alias，其 `transforms` 列表必须**长度相同**，且**相同位置的 `name` 与 `scope` 完全一致**。否则无法在步骤级按 scope 分发。校验由 `ConfigValidator.validate()` 在规范化后执行；不一致时抛出 `ConfigError` |
| R30 | val_split 维度偏序约束 | 当 Regular 主 `split.dimension` ∈ `{subject, session, recording}` 时，`val_split.dimension` 可为任意 `subject | session | recording | flatten`（训练子集沿主维度切片后仍为干净 5D，`ValSplitter` 可直接 `get_groups`）；当主 `split.dimension=flatten` 时，`val_split.dimension` 必须也为 `flatten`——flatten 切分后训练子集不再保持 5D 结构，无法在其他维度上分组。其他组合抛出 `ConfigError` |
| R35 | train_step_scalars 类型 | `training.logging.train_step_scalars`（若声明）必须为**非空字符串**的 list。用于在 TensorBoard 中筛选 `training_step` 返回 dict 的标量 key；未声明 / 为 `null` 时记录所有数值标量 |
| R36 | train_metrics 类型 | `training.logging.train_metrics`（若声明）必须为**非空字符串**的 list。声明后框架将在训练集 preds/targets 上用 Evaluator 计算对应指标并以 `train_<name>` 写入 TensorBoard；Trainer 的 `training_step` 需返回 `preds` 与 `targets` 才能生效 |
| R37 | val_metrics 为 evaluation.metrics 子集 | `training.logging.val_metrics`（若声明）必须为**非空字符串**的 list，且所有元素必须出现在 `evaluation.metrics` 中（因此 `evaluation.metrics` 必须已声明）。未声明 / 为 `null` 时记录 `evaluation.metrics` 全部 |
| R38 | test_metrics 类型 | `training.logging.test_metrics`（若声明）必须为**非空字符串**的 list。声明后框架每 `log_every_n_epochs` 在测试集上运行一次评估并以 `test_<name>` 写入 TensorBoard；触发数据泄露风险 warning，详见 [09_training_logger.md §9.13](09_training_logger.md) |

---

## 边界条件处理

以下场景需在运行时（数据加载后、划分前）进行检查，因为涉及实际数据的分组数量，无法在纯配置校验阶段完成：

| 场景 | 处理方式 |
|:-----|:---------|
| 任意维度分组数为 1 | 抛出 `SplitError`。仅有 1 组时无法进行任何有意义的切分（无论 holdout/k-fold/ValSplit），一律拦截 |
| `val_ratio=0`（ValSplit 或 Split Block） | 合法。`val_indices` 为空数组，不构建 val dataloader，Trainer 跳过验证步骤 |
| 分组后组数 < 所需切分数量（如 2 组但 `val_ratio=0.4`，训练只剩 1 组） | 运行时 warning，不阻断。允许退化但记录日志 |
| `target_count` 或 `target_ratio` 换算后 ≥ 实际组数 | 抛出 `SplitError`（目标域不能占满所有组，源域不能为空） |
| `k` > 分组数 | 抛出 `SplitError`（折数不能超过组数） |
| k-fold 场景下 `val_ratio_in_train` 换算后 > 0.5 | 运行时 warning，提示训练数据占比过低（不阻断） |
