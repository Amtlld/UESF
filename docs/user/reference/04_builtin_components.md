# 内置组件参考

UESF 提供开箱即用的内置组件，无需注册即可在配置文件中直接使用。

---

## 预处理算子

### 数据流算子（`pipeline.data`）

| 算子名 | 说明 | 参数 | 参数说明 |
|--------|------|------|----------|
| `filter` | 带通/高通/低通滤波 | `l_freq: float` | 高通截止频率（Hz），`null` 表示不做高通 |
| | | `h_freq: float` | 低通截止频率（Hz），`null` 表示不做低通 |
| `notch_filter` | 陷波滤波，去除市电干扰 | `notch_freq: float` | 陷波频率（50 或 60 Hz） |
| `ica` | 独立成分分析，去眼电/心电伪影 | `method: str` | ICA 算法：`fastica`、`picard` |
| | | `n_components: float\|int` | 浮点数表示方差解释比例（如 0.95），整数表示成分数量 |
| | | `exclude_eog_ecg: bool` | 是否自动识别并去除眼电/心电成分 |
| `resample` | 频率重采样 | `target_rate: int` | 目标采样率（Hz） |
| `reference` | 重参考 | `type: str` | 参考方式：`CAR`（公共平均参考）、`mastoid` 等 |

**示例**：

```yaml
data:
  - name: filter
    params: { l_freq: 1.0, h_freq: 40.0 }
  - name: notch_filter
    params: { notch_freq: 50.0 }
  - name: ica
    params: { method: fastica, n_components: 0.95, exclude_eog_ecg: true }
  - name: resample
    params: { target_rate: 128 }
```

### 标签流算子（`pipeline.label`）

| 算子名 | 说明 | 参数 | 参数说明 |
|--------|------|------|----------|
| `smooth` | 滑窗标签平滑 | `window_size: int` | 滑动窗口大小（样本数） |

### 联合流算子（`pipeline.joint`）

| 算子名 | 说明 | 参数 | 参数说明 |
|--------|------|------|----------|
| `sliding_window` | 滑窗切片（Epoching）| `window_size_sec: float` | 窗口时长（秒） |
| | | `stride_sec: float` | 滑动步长（秒） |
| | | `window_type: str` | 窗函数：`rect`（矩形）、`hanning`（汉宁窗） |
| | | `label_strategy: str` | 窗口标签：`mode`（众数）、`last`（末尾标签） |
| `epoch_normalize` | Epoch 内独立标准化 | `method: str` | 标准化方法：`zscore`、`minmax` |
| | | `axis: int` | 计算轴，`-1` 表示时间轴 |

---

## 在线变换

在线变换配置在实验 YAML 的 `datasets.<alias>.transforms` 下，在数据切分完成后执行。

| 变换名 | 说明 | 参数 | 说明 |
|--------|------|------|------|
| `zscore_normalize` | 全局 Z-Score 标准化 | `fit_on: str` | 用于拟合参数的集合：`train` |
| | | `apply_to: str` | 应用范围：`all`（train/val/test 全部） |

**示例**：

```yaml
transforms:
  - name: zscore_normalize
    fit_on: train
    apply_to: all
```

---

## 内置评估指标

直接在实验 YAML 的 `evaluation.metrics` 列表中使用，无需注册。

| 指标名 | 说明 | 返回类型 |
|--------|------|----------|
| `accuracy` | 分类准确率 | float |
| `f1_score` | 加权 F1 分数 | float |
| `precision` | 加权精确率 | float |
| `recall` | 加权召回率 | float |
| `auroc` | AUROC（曲线下面积）| float |
| `confusion_matrix` | 混淆矩阵 | dict（含 `matrix` 键） |

**使用示例**：

```yaml
evaluation:
  metrics: [accuracy, f1_score, precision, recall, auroc, confusion_matrix]
```

---

## 内置优化器

名称直接用于实验 YAML 的 `training.optimizer.name`。参数通过 `training.optimizer.params` 传入，名称与 PyTorch 官方 API 完全一致。

| 名称 | 对应 PyTorch 类 | 常用参数 |
|------|----------------|----------|
| `sgd` | `torch.optim.SGD` | `lr`, `momentum`, `weight_decay`, `nesterov` |
| `adam` | `torch.optim.Adam` | `lr`, `betas`, `eps`, `weight_decay` |
| `adamw` | `torch.optim.AdamW` | `lr`, `betas`, `eps`, `weight_decay` |
| `adagrad` | `torch.optim.Adagrad` | `lr`, `weight_decay`, `eps` |
| `adadelta` | `torch.optim.Adadelta` | `lr`, `rho`, `eps`, `weight_decay` |
| `rmsprop` | `torch.optim.RMSprop` | `lr`, `alpha`, `eps`, `weight_decay`, `momentum` |
| `radam` | `torch.optim.RAdam` | `lr`, `betas`, `eps`, `weight_decay` |
| `nadam` | `torch.optim.NAdam` | `lr`, `betas`, `eps`, `weight_decay` |

**示例**：

```yaml
optimizer:
  name: adam
  params:
    lr: 0.001
    weight_decay: 1e-4
    betas: [0.9, 0.999]
```

---

## 内置学习率调度器

名称用于 `training.scheduler.name`，参数用于 `training.scheduler.params`，与 PyTorch 官方 API 一致。

| 名称 | 对应 PyTorch 类 | 必填参数 |
|------|----------------|----------|
| `step_lr` | `StepLR` | `step_size`, `gamma` |
| `multi_step_lr` | `MultiStepLR` | `milestones`, `gamma` |
| `exponential_lr` | `ExponentialLR` | `gamma` |
| `linear_lr` | `LinearLR` | `start_factor`, `end_factor`, `total_iters` |
| `cosine_annealing_lr` | `CosineAnnealingLR` | `T_max`, `eta_min` |
| `cosine_annealing_warm_restarts` | `CosineAnnealingWarmRestarts` | `T_0`, `T_mult`, `eta_min` |
| `reduce_lr_on_plateau` | `ReduceLROnPlateau` | `factor`, `patience`, `mode` |
| `one_cycle_lr` | `OneCycleLR` | `max_lr`, `total_steps` |

**示例**：

```yaml
scheduler:
  name: cosine_annealing_lr
  params:
    T_max: 50
    eta_min: 1.0e-6
```
