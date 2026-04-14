# 七、配置校验规则汇总

| 编号 | 规则 | 校验内容 |
|:-----|:-----|:---------|
| R1 | mode/split 互斥 | `mode=regular` ↔ 有 `split`，无 `uda` |
| R2 | mode/uda 互斥 | `mode=uda` ↔ 有 `uda`，无 `split` |
| R3 | 数据集级需多数据集 | `split.dimension=dataset` → `len(datasets) > 1` |
| R4 | inductive 需测试集 | `adaptation=inductive` → `target.split` 中有 `test_ratio`（holdout）或 `k`（k-fold） |
| R5 | transductive 无测试集 | `adaptation=transductive` → `target.split` 无 `test_ratio`，无 `strategy` |
| R6 | holdout ratio 求和 | holdout 的 `train_ratio + val_ratio + test_ratio` 之和满足 `abs(sum - 1.0) < 1e-4` |
| R7 | k-fold 合法 k 值 | k-fold 的 `k > 1` 或 `k == -1` |
| R8 | 跨数据集域需指定 | `domain.dimension=dataset` + `strategy=holdout` → 必须指定 `source` 和 `target` |
| R9 | 数据集内域需单数据集 | `domain.dimension=subject\|session` → `len(datasets) == 1`（不同数据集的 subject/session 编号可能冲突，无法统一分组） |
| R10 | 源域无测试集 | `source.split` 不含 `strategy`、`test_ratio`、`train_ratio` |
| R11 | 域维度与内部维度不同 | `domain.dimension` ≠ `source.split.dimension` 且 `domain.dimension` ≠ `target.split.dimension` |
| R12 | target 计数互斥 | `domain` 中 `target_count` 和 `target_ratio` 不可同时出现 |
| R13 | dataset holdout 需 assign | `split.dimension=dataset` + `strategy=holdout` → `assign` 必填，且 `assign` 中的 alias 须覆盖所有声明的 datasets |
| R14 | dataset k-fold 的 k 值约束 | `split.dimension=dataset` + `strategy=k-fold` → `k == -1` 或 `k == len(datasets)` |
| R15 | val_ratio 值域 | `0 <= val_ratio < 1`；k-fold 换算后 `val_ratio_in_train < 1` |
| R16 | 单数据集 alignment 处理 | `len(datasets) == 1` 时若存在 `alignment` 块，忽略并发出 warning |
| R17 | logging.backend 合法值 | `training.logging.backend` 仅允许 `"tensorboard"`，其他值抛出 `TypeMismatchError` |
| R18 | log_every_n_epochs 正整数 | `training.logging.log_every_n_epochs` 必须为正整数（> 0），否则抛出 `TypeMismatchError` |

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
