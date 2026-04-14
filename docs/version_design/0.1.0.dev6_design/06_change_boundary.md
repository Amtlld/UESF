# 六、与现有模块的变更边界

| 模块 | 变更程度 | 说明 |
|:-----|:---------|:-----|
| `runner.py` | **不变** | 仍接收 train/val/test dataloader，接口完全兼容 |
| `evaluator.py` | **不变** | 仍聚合 fold 结果 |
| `dataset.py` | **不变** | EEGDataset 封装不变 |
| `transforms.py` | **不变** | fit/transform 接口不变 |
| `alignment.py` | **微调** | 配置键名简化（`channels.method` → `channel`，`labels.check_consistency` → `label`） |
| `dataloader_builder.py` | **重构** | 移除手动通道映射，改为接收已构建的 `dict[str, EEGDataset]` |
| `splitter.py` | **重写** | 拆分为 `splitter/` 子包，逻辑重新组织 |
| `experiment_manager.py` | **重构** | 提取 `ConfigValidator`（normalize + validate 分离）+ `ExperimentExecutor`，策略模式分发 |
| `BaseTrainer` | **不变** | `training_step(batch)` 签名不变 |
| `BaseModel` | **不变** | `forward(x)` 签名不变 |

> 本次重构为 UESF 初版开发阶段的 breaking change，不考虑后向兼容。
