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
