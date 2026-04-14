# 六、与现有模块的变更边界

| 模块 | 变更程度 | 说明 |
|:-----|:---------|:-----|
| `runner.py` | **微调** | `run()` 新增可选参数 `logger` 和 `log_every_n_epochs`，无 logger 时行为完全不变 |
| `evaluator.py` | **不变** | 仍聚合 fold 结果 |
| `dataset.py` | **不变** | EEGDataset 封装不变 |
| `transforms.py` | **不变** | fit/transform 接口不变 |
| `alignment.py` | **微调** | 配置键名简化（`channels.method` → `channel`，`labels.check_consistency` → `label`） |
| `dataloader_builder.py` | **重构** | 移除手动通道映射，改为接收按通道组织的 `dict[str, np.ndarray]`，内部封装 `EEGDataset` 并构建 `DataLoader` |
| `splitter.py` | **重写** | 拆分为 `splitter/` 子包，逻辑重新组织。`UDASplitResult` 字段去除 `_indices` 后缀（如 `source_train_indices` → `source_train`），新增 `fold_info` 字段 |
| `experiment_manager.py` | **重构** | 提取 `ConfigValidator`（normalize + validate 分离）+ `ExperimentExecutor`，策略模式分发 |
| `BaseTrainer` | **不变** | `training_step(batch)` 签名不变 |
| `BaseModel` | **不变** | `forward(x)` 签名不变 |
| `logger.py` | **新增** | TrainingLogger 协议 + TensorBoardLogger 实现，位于 `experiment/` |

> 本次重构为 UESF 初版开发阶段的 breaking change，不考虑后向兼容。
