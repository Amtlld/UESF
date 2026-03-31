# UESF - Universal EEG Study Framework

UESF 是一个面向 EEG 深度学习研究的标准化实验管理框架。它提供从原始数据管理、预处理流水线、模型/训练器注册，到实验执行与结果查询的完整生命周期支持，通过 CLI 和 YAML 配置驱动全部工作流。

## 核心特性

- **数据全生命周期管理** -- 原始数据集注册/导入、三流预处理流水线（数据流/标签流/联合流）、标签重映射（Masked Dataset）
- **三类组件管理** -- 内置（EMBEDDED）、全局（GLOBAL）、项目级（REGISTERED）三层优先级解析，SHA256 自动检测源码变更并归档旧版本
- **防数据泄露设计** -- 维度隔离切分（按 subject/session 等）、Fit-on-Train 在线变换、Epoch 级指标聚合
- **控制反转架构** -- Trainer 全权负责梯度操作，Runner 极度精简；多通道字典 DataLoader 原生支持域自适应等复杂场景
- **灵活的实验管理** -- Holdout/K-Fold/LOOCV 切分策略、concat/mean_std 多折聚合、早停、检查点自动保存、实验结果数据库持久化

## 安装

```bash
# 推荐使用 uv
uv venv && source .venv/bin/activate
uv pip install -e .

# 开发依赖
uv pip install pytest pytest-cov ruff
```

## 快速开始

### 1. 全局配置

```bash
uesf config show
uesf config set default_device cuda
uesf config set data_dir ~/eeg_data
```

### 2. 注册原始数据集

准备 `raw.yml` 描述数据集元信息，然后：

```bash
# 仅注册（数据保留在原位）
uesf data raw register /path/to/dataset/

# 或导入（复制到 UESF 数据目录）
uesf data raw import /path/to/dataset/
```

### 3. 预处理

编写 `preprocess.yml` 定义预处理流水线：

```yaml
source_dataset: my_raw_dataset
out_name: my_preprocessed

data_stream:
  - name: bandpass_filter
    params: { low_freq: 1.0, high_freq: 40.0 }
  - name: resample
    params: { target_sr: 128.0 }

joint_stream:
  - name: sliding_window
    params: { window_size: 4.0, step_size: 1.0 }
  - name: epoch_normalize
    params: { method: zscore }
```

```bash
uesf data preprocess run -c preprocess.yml
```

### 4. 初始化项目

```bash
mkdir my_eeg_project && cd my_eeg_project
uesf project init
```

编辑 `project.yml` 注册模型和训练器：

```yaml
project-name: emotion_recognition

preprocessed_datasets:
  - my_preprocessed

models:
  my_cnn:
    entrypoint: "./src/models/cnn.py:EmotionCNN"

trainers:
  my_trainer:
    entrypoint: "./src/trainers/trainer.py:EmotionTrainer"
```

### 5. 配置并运行实验

```bash
uesf experiment add --name baseline_exp
```

编辑 `experiments/baseline_exp.yml`：

```yaml
name: baseline_exp
seed: 42

model:
  name: my_cnn
  params: { dropout: 0.5 }

trainer:
  name: my_trainer
  params: {}

datasets:
  main:
    name: my_preprocessed
    split:
      strategy: k-fold
      dimension: subject
      k-folds: 5
    transforms:
      - name: zscore_normalize
        fit_on: train
        apply_to: all

dataloaders:
  train:
    main: "main.train"
  val:
    main: "main.val"
  test:
    main: "main.test"

training:
  epochs: 100
  batch_size: 64
  optimizer:
    name: adam
    params: { lr: 0.001 }
  early_stopping:
    monitor: val_accuracy
    patience: 10
    mode: max

evaluation:
  metrics: [accuracy, f1_score, auroc]
  k_fold_aggregation: concat

logging:
  checkpoint_metric: val_accuracy
```

```bash
uesf experiment run --exp baseline_exp
```

### 6. 查询结果

```bash
uesf experiment query --metrics accuracy,f1_score --status COMPLETED
```

## 自定义组件

### 模型

继承 `BaseModel`，实现 `forward()` 方法：

```python
from uesf.components.base_model import BaseModel

class MyModel(BaseModel):
    def __init__(self, n_channels, n_samples, n_classes, **kwargs):
        super().__init__(n_channels, n_samples, n_classes, **kwargs)
        # n_channels, n_samples, n_classes 由框架自动从数据集注入
        self.conv = nn.Conv1d(n_channels, 64, 3)
        # ...

    def forward(self, x, **kwargs):
        return self.fc(self.conv(x))
```

### 训练器

继承 `BaseTrainer`，实现 `training_step()` 和 `validation_step()`：

```python
from uesf.components.base_trainer import BaseTrainer

class MyTrainer(BaseTrainer):
    def training_step(self, batch, batch_idx, optimizer):
        # batch: {channel_name: (data, labels)}
        for name, (data, labels) in batch.items():
            output = self.model(data.to(self.device))
            loss = F.cross_entropy(output, labels.to(self.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return {"loss": loss.item()}

    def validation_step(self, batch, batch_idx):
        # 返回 preds 和 targets，由 Evaluator 在 Epoch 级聚合后计算指标
        preds, targets = [], []
        for name, (data, labels) in batch.items():
            preds.append(self.model(data.to(self.device)).argmax(1))
            targets.append(labels)
        return {"preds": torch.cat(preds), "targets": torch.cat(targets)}
```

### 评估指标

遵循统一签名 `(preds, targets, **kwargs) -> float | dict`：

```python
def my_custom_metric(preds, targets, **kwargs):
    correct = (preds == targets).sum().item()
    return correct / len(targets)
```

## CLI 命令参考

| 命令 | 说明 |
|------|------|
| `uesf config show/set` | 查看/修改全局配置 |
| `uesf data raw register/import/list/remove/edit/info` | 原始数据集管理 |
| `uesf data preprocess run` | 执行预处理流水线 |
| `uesf data preprocessed list/remove` | 预处理数据集管理 |
| `uesf data preprocessed mask` | 创建标签重映射数据集 |
| `uesf model add/list/remove/edit` | 全局模型管理 |
| `uesf trainer add/list/remove/edit` | 全局训练器管理 |
| `uesf metric add/list/remove/edit` | 全局评估指标管理 |
| `uesf project init/info` | 项目初始化与信息查看 |
| `uesf experiment add/list/remove/run/query` | 实验全生命周期管理 |

## 内置组件

### 预处理算子

| 算子 | 类型 | 说明 |
|------|------|------|
| `resample` | data | 重采样至目标采样率 |
| `bandpass_filter` | data | 带通滤波 |
| `notch_filter` | data | 陷波滤波 |
| `reference` | data | 参考电极（CAR） |
| `smooth` | label | 标签平滑 |
| `sliding_window` | joint | 滑动窗口分段 |
| `epoch_normalize` | joint | Epoch 级标准化（zscore/minmax） |

### 评估指标

`accuracy`, `f1_score`, `precision`, `recall`, `auroc`, `confusion_matrix`

### 优化器

`sgd`, `adam`, `adamw`, `adagrad`, `adadelta`, `rmsprop`, `radam`, `nadam`

### 调度器

`step_lr`, `multi_step_lr`, `exponential_lr`, `linear_lr`, `cosine_annealing_lr`, `cosine_annealing_warm_restarts`, `reduce_lr_on_plateau`, `one_cycle_lr`

## 项目结构

```
src/uesf/
  cli/           # Typer CLI 命令
  components/    # BaseModel, BaseTrainer, 内置映射与指标
  core/          # 异常、日志、数据库、配置
  experiment/    # Splitter, Transforms, Dataset, DataLoader, Runner, Evaluator
  managers/      # DataManager, ModelManager, TrainerManager, MetricManager,
                 # ProjectManager, ExperimentManager
  pipeline/      # 预处理流水线与算子
```

## 开发

```bash
# 运行测试
python -m pytest tests/ -v

# 代码检查
ruff check src/ tests/
```

## 许可证

MIT
