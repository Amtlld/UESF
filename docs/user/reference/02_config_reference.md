# 配置文件格式参考

UESF 使用 4 种配置文件，本文档列出每种配置文件的完整字段说明。

---

## 1. 全局配置文件（`~/.uesf/config.yml`）

由 `uesf config set` 命令修改，或手动编辑。

```yaml
data_dir: ~/eeg_data
default_device: cuda
num_workers: 4
log_level: INFO
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data_dir` | string | `~/.uesf/data` | 数据文件存储目录（`raw/`、`preprocessed/`、`masked/` 均在此目录下） |
| `default_device` | string | `cpu` | 默认计算设备（`cpu`、`cuda`、`cuda:0` 等） |
| `num_workers` | int | 0 | DataLoader 的 worker 进程数 |
| `log_level` | string | `INFO` | 日志级别：`DEBUG`、`INFO`、`WARNING`、`ERROR` |

---

## 2. 原始数据集描述文件（`raw.yml`）

放在原始数据集目录下，`uesf data raw register/import` 时读取。

```yaml
raw:
  name: seed_raw
  description: SEED 情绪脑电数据集，62 通道，200Hz
  eeg_data_key: data
  label_key: label
  sampling_rate: 200
  n_subjects: 14
  n_sessions: 3
  n_recordings: 1
  n_channels: 62
  n_samples: 800
  electrode_list:
    - Fp1
    - Fp2
    # ...（按通道顺序排列）
  dimension_info:
    - subject
    - session
    - recording
  numeric_to_semantic:
    0: negative
    1: neutral
    2: positive
```

| 字段 | 是否必填 | 类型 | 说明 |
|------|----------|------|------|
| `name` | 必填 | string | 数据集唯一名称（只含字母、数字、下划线） |
| `description` | 可选 | string | 描述文字 |
| `eeg_data_key` | 必填 | string | `.mat` 文件中 EEG 数据的键名 |
| `label_key` | 必填 | string | `.mat` 文件中标签数组的键名 |
| `sampling_rate` | 必填 | float | 采样率（Hz） |
| `n_subjects` | 必填 | int | 被试数量 |
| `n_sessions` | 必填 | int | 每名被试的会话数 |
| `n_recordings` | 必填 | int | 每个会话的录制段数（若未切分则为 1） |
| `n_channels` | 必填 | int | EEG 通道数 |
| `n_samples` | 可选 | int | 每段录制的采样点数；注册时会从文件自动推断并校验 |
| `electrode_list` | 可选 | list[str] | 电极名称列表，框架会注入模型初始化参数 |
| `dimension_info` | 必填 | list[str] | 数据维度语义，通常为 `[subject, session, recording]` |
| `numeric_to_semantic` | 必填 | dict | 数字标签到语义标签的映射，决定类别数 |

> `data_shape` 和 `label_shape` 无需填写，注册时自动推断。

---

## 3. 预处理配置文件（`preprocess.yml`）

```yaml
preprocess:
  source_dataset: seed_raw        # 输入原始数据集名称（可被 CLI --dataset 覆盖）
  out_name: seed_preprocessed     # 输出数据集名称（可被 CLI --out-name 覆盖）

  pipeline:
    data:                         # 数据流：滤波、重采样等
      - name: filter
        params:
          l_freq: 1.0
          h_freq: 40.0
      - name: notch_filter
        params:
          notch_freq: 50.0
      - name: ica
        params:
          method: fastica
          n_components: 0.95
          exclude_eog_ecg: true
      - name: resample
        params:
          target_rate: 128

    label:                        # 标签流：标签预处理
      - name: smooth
        params:
          window_size: 5

    joint:                        # 联合流：同时切分信号和标签
      - name: sliding_window
        params:
          window_size_sec: 4.0
          stride_sec: 2.0
          window_type: rect
          label_strategy: mode
      - name: epoch_normalize
        params:
          method: zscore
          axis: -1
```

### 顶层字段

| 字段 | 是否必填 | 说明 |
|------|----------|------|
| `source_dataset` | 可选 | 输入原始数据集名称（也可通过 CLI 参数指定） |
| `out_name` | 必填 | 输出预处理数据集名称 |
| `pipeline.data` | 可选 | 数据流算子列表 |
| `pipeline.label` | 可选 | 标签流算子列表 |
| `pipeline.joint` | 可选 | 联合流算子列表 |

算子完整列表参见 [内置组件列表 → 预处理算子](04_builtin_components.md#预处理算子)。

---

## 4. 项目配置文件（`project.yml`）

放在项目根目录，`uesf project init` 生成初始模板。

```yaml
project-name: emotion_recognition
description: SEED 情绪识别跨被试实验

# 方式一：直接使用预处理数据集
preprocessed_datasets:
  - seed_preprocessed
  - seed_binary          # 也可以是标签重映射数据集

# 方式二：从原始数据集 + 预处理配置自动生成（两种方式二选一）
# raw_datasets:
#   - seed_raw
# preprocess_config: ./preprocess.yml

models:
  emotion_cnn:
    entrypoint: "./src/models/cnn.py:EmotionCNN"
  transformer:
    entrypoint: "./src/models/transformer.py:EEGTransformer"

trainers:
  emotion_trainer:
    entrypoint: "./src/trainers/trainer.py:EmotionTrainer"

metrics:
  balanced_accuracy:
    entrypoint: "./src/metrics/balanced.py:balanced_accuracy"
```

| 字段 | 是否必填 | 说明 |
|------|----------|------|
| `project-name` | 必填 | 项目名称 |
| `description` | 可选 | 项目描述 |
| `preprocessed_datasets` | 与 `raw_datasets` 二选一 | 直接使用已有预处理数据集 |
| `raw_datasets` | 与 `preprocessed_datasets` 二选一 | 原始数据集名称列表 |
| `preprocess_config` | 仅 `raw_datasets` 方式需填 | 预处理配置文件路径 |
| `models.<name>.entrypoint` | 可选 | 模型入口点，格式：`"./path/file.py:ClassName"` |
| `trainers.<name>.entrypoint` | 可选 | 训练器入口点 |
| `metrics.<name>.entrypoint` | 可选 | 指标函数入口点，格式：`"./path/file.py:func_name"` |

---

## 5. 实验配置文件（`experiments/<name>.yml`）

由 `uesf experiment add` 生成，路径固定为项目 `experiments/` 目录下。

完整字段参见 [如何配置实验 YAML](../how-to/06_configure_experiment.md)。

以下是字段速查表：

### 顶层字段

| 字段 | 是否必填 | 类型 | 说明 |
|------|----------|------|------|
| `name` | 必填 | string | 实验名称（与文件名一致） |
| `description` | 可选 | string | 实验描述 |
| `seed` | 推荐 | int | 随机种子，保证切分可复现 |

### model / trainer

| 字段 | 说明 |
|------|------|
| `model.name` | 组件名，对应 `project.yml` 中 `models` 块的键名 |
| `model.params` | 传入 `__init__` 的 kwargs 字典 |
| `trainer.name` / `trainer.params` | 同上 |

### datasets

| 字段 | 说明 |
|------|------|
| `datasets.<alias>.name` | 预处理数据集名称 |
| `datasets.<alias>.split.strategy` | `holdout` 或 `k-fold` |
| `datasets.<alias>.split.dimension` | `subject`、`session`、`recording`、`none` |
| `datasets.<alias>.split.shuffle` | bool，是否随机打乱 |
| `datasets.<alias>.split.k-folds` | K-Fold 专有，折数；`-1` 为 LOOCV |
| `datasets.<alias>.split.val_ratio_in_train` | K-Fold 专有，从训练折划出的验证比例 |
| `datasets.<alias>.split.train_ratio` | Holdout 专有 |
| `datasets.<alias>.split.val_ratio` | Holdout 专有 |
| `datasets.<alias>.split.test_ratio` | Holdout 专有 |
| `datasets.<alias>.transforms` | 在线变换列表（`name`、`fit_on`、`apply_to`、`params`） |

### dataloaders

```yaml
dataloaders:
  train:
    <channel_name>: "<dataset_alias>.<split_phase>"
  val:
    <channel_name>: "<dataset_alias>.val"
  test:
    <channel_name>: "<dataset_alias>.test"
```

### training

| 字段 | 是否必填 | 类型 | 说明 |
|------|----------|------|------|
| `epochs` | 必填 | int | 最大训练轮数 |
| `batch_size` | 必填 | int | 批次大小 |
| `optimizer.name` | 必填 | string | 优化器名称 |
| `optimizer.params.lr` | 必填 | float | 学习率 |
| `optimizer.params.*` | 可选 | - | 其他 PyTorch 优化器参数（与官方 API 一致） |
| `gradient_clip.max_norm` | 可选 | float | 梯度最大范数 |
| `gradient_clip.norm_type` | 可选 | int | 范数类型，默认 2 |
| `scheduler.name` | 可选 | string | 调度器名称 |
| `scheduler.params` | 可选 | dict | 调度器参数（与 PyTorch 官方 API 一致） |
| `early_stopping.monitor` | 可选 | string | 监控的指标名称 |
| `early_stopping.patience` | 可选 | int | 容忍轮数 |
| `early_stopping.min_delta` | 可选 | float | 最小改善量，默认 0.0 |
| `early_stopping.mode` | 可选 | string | `max` 或 `min` |

### evaluation / logging

| 字段 | 是否必填 | 说明 |
|------|----------|------|
| `evaluation.metrics` | 必填 | 指标名称列表 |
| `evaluation.k_fold_aggregation` | 可选 | `concat`（默认）或 `mean_std` |
| `logging.use_wandb` | 可选 | bool，是否启用 W&B |
| `logging.checkpoint_metric` | 推荐 | 保存最优检查点的依据指标 |
