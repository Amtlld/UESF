# 五、跨数据集对齐模块

保持现有 `alignment.py` 接口不变，简化配置入口：

```yaml
alignment:
  channel: intersection       # 通道对齐策略（可扩展: interpolation 等）
  label: true                 # 标签一致性校验
```

**触发条件**：框架检测到 `datasets` 中包含多个数据集 **且** 存在跨数据集操作时，自动应用对齐。

| 场景 | 触发 | 说明 |
|:-----|:-----|:-----|
| `mode: regular` + `dimension: dataset` | 自动触发 | 多数据集合并训练，需对齐通道和标签 |
| `mode: uda` + `domain.dimension: dataset` | 自动触发 | 跨数据集域适应，需对齐通道和标签 |
| 单数据集内实验（`len(datasets) == 1`） | 跳过 | 无需对齐 |
| 多数据集 + `dimension ≠ dataset`（Regular）| 被 R25 拦截 | 配置校验阶段即报错，不会到达对齐阶段 |
| 多数据集 + `domain.dimension ≠ dataset`（UDA）| 被 R25 拦截 | 配置校验阶段即报错，不会到达对齐阶段 |

**对齐时机**：在数据集加载完成后、任何划分操作之前执行。对齐结果直接更新 `dataset_cache`（原地替换），后续划分和训练使用对齐后的数据。

**`n_samples` 校验**：对齐阶段检查各数据集的 `n_samples`（采样点数）是否一致。不一致时抛出 `ShapeMismatchError`。采样率不一致但 `n_samples` 一致时发出 warning（可能意味着时间窗口长度不同）。

**处理流程**：

1. `ChannelAligner.align(datasets)` → 各数据集保留交集通道，返回对齐后的数据和公共通道列表
2. `LabelAligner.validate(metadata)` → 校验标签一致性
3. 校验 `n_samples` 一致性
4. 更新 `dataset_cache` 和 `metadata_cache`

**接口保留**：

- `ChannelAligner` ABC 不变，`CHANNEL_ALIGNER_REGISTRY` 支持后续注册 `interpolation` 等新策略
- `LabelAligner` 不变，检查 `n_classes` 和 `numeric_to_semantic` 映射一致性
