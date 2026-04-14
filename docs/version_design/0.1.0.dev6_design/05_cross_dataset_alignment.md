# 五、跨数据集对齐模块

保持现有 `alignment.py` 接口不变，简化配置入口：

```yaml
alignment:
  channel: intersection       # 通道对齐策略（可扩展: interpolation 等）
  label: true                 # 标签一致性校验
```

**触发条件**：框架检测到 `datasets` 中包含多个数据集 **且** 存在跨数据集操作时，自动应用对齐。

| 场景 | 触发 |
|:-----|:-----|
| `mode: regular` + `dimension: dataset` | 自动触发 |
| `mode: uda` + `domain.dimension: dataset` | 自动触发 |
| 单数据集内实验 | 跳过 |

**接口保留**：

- `ChannelAligner` ABC 不变，`CHANNEL_ALIGNER_REGISTRY` 支持后续注册 `interpolation` 等新策略
- `LabelAligner` 不变，检查 `n_classes` 和 `numeric_to_semantic` 映射一致性
