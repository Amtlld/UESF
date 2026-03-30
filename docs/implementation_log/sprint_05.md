# Sprint 5: 实验基础设施

## 目标

实现完整的实验执行 pipeline：Splitter、Online Transforms、Multi-channel DataLoader、Runner、Evaluator、ExperimentManager。

## 完成内容

### 数据集切分器 (Splitter)

- **splitter.py** (`src/uesf/experiment/splitter.py`)
  - `HoldoutSplitter` — train/val/test 按比例切分
  - `KFoldSplitter` — K 折交叉验证，k=-1 或 "total" 为 LOOCV
  - 维度隔离 (`dimension`): subject/session/recording/none
  - `shuffle` + `seed` 控制确定性打乱
  - `val_ratio_in_train` 从训练集中切出验证集（用于 K-Fold 早停）
  - `SplitResult` 数据类持有 train/val/test 索引数组
  - `_get_groups()` 按指定维度分组索引

### 在线变换 (Transforms)

- **transforms.py** (`src/uesf/experiment/transforms.py`)
  - `ZScoreNormalize` — fit-on-train, apply-to-all 设计
  - 仅从 train split 计算 mean/std，防止数据泄露
  - `TRANSFORM_REGISTRY` 注册表 + `create_transform()` 工厂

### 数据集包装 (Dataset)

- **dataset.py** (`src/uesf/experiment/dataset.py`)
  - `EEGDataset(Dataset)` — numpy 到 PyTorch 的薄包装
  - 自动转换 float32 + long 类型

### 多通道 DataLoader

- **dataloader_builder.py** (`src/uesf/experiment/dataloader_builder.py`)
  - `CombinedIterator` — 同步迭代多个 DataLoader，产出 batch 字典
  - `build_dataloaders()` — 按 phase 控制 shuffle
  - 以最短 DataLoader 为准停止迭代

### 评估器 (Evaluator)

- **evaluator.py** (`src/uesf/experiment/evaluator.py`)
  - Epoch 级聚合：拼接全部 batch 的 preds/targets 后一次性计算指标
  - K-Fold 聚合：`concat` 模式（全局拼接重算）和 `mean_std` 模式（均值±标准差）
  - 单个指标失败不影响其他指标计算

### 训练循环 (Runner)

- **runner.py** (`src/uesf/experiment/runner.py`)
  - 极度精简的 Runner，仅负责循环编排
  - `train_epoch()` — 委托 Trainer.training_step()
  - `validate_epoch()` — 委托 Trainer.validation_step()，收集 preds/targets
  - `run()` — 完整训练循环：epoch loop + scheduler step + early stopping + checkpoint
  - `EarlyStopping` — min/max 模式，patience + min_delta
  - 梯度裁剪支持 (max_norm, norm_type)

### 实验管理器 (ExperimentManager)

- **experiment_manager.py** (`src/uesf/managers/experiment_manager.py`)
  - `add()` — 创建实验配置 YAML（空白或从现有复制）
  - `list()` — 列出项目下所有实验
  - `remove()` — 删除实验配置和/或结果
  - `run()` — 完整编排：config → component init → split → transform → train → evaluate → save
  - `query()` — 跨项目查询实验结果
  - 状态机：PENDING → RUNNING → COMPLETED/FAILED
  - 崩溃时状态仍会被记录到数据库
  - 多折每折新建 model/trainer 实例

### 测试

| 测试文件 | 测试数 | 覆盖内容 |
|----------|--------|----------|
| `tests/experiment/test_splitter.py` | 13 | Holdout/KFold/LOOCV、维度隔离、确定性、val_ratio |
| `tests/experiment/test_transforms.py` | 5 | ZScore fit/transform/fit_transform、工厂、未知变换 |
| `tests/experiment/test_dataset.py` | 3 | 长度、getitem、不匹配检查 |
| `tests/experiment/test_dataloader_builder.py` | 5 | 多通道、最短停止、单通道、val不打乱、长度 |
| `tests/experiment/test_evaluator.py` | 5 | epoch 指标、空输入、异常处理、mean_std/concat 聚合 |
| `tests/experiment/test_runner.py` | 7 | EarlyStopping、train/validate epoch、完整 run、早停 |

**总测试数：321（含 Sprint 0-4 的 282 个）**

## 关键设计决策

1. **Trainer 全权负责梯度**：Runner 从不调用 backward()/step()，仅委托 Trainer
2. **Epoch 级聚合**：不对 batch 平均，而是拼接完整 epoch 的 preds/targets 后一次性计算
3. **Fit-on-Train 原则**：ZScore 统计量仅从 train split 计算，防止数据泄露
4. **维度隔离**：同一 subject 不跨 fold，从根源防止泄露
5. **多通道字典 batch**：`{channel: (data, labels)}` 格式支持域自适应等复杂场景

## 验收

- [x] 321 个测试全部通过
- [x] ruff check 无错误
