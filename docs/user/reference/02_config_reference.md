# 配置文件格式参考

UESF 使用 4 种配置文件，本文档列出每种配置文件的完整字段说明。

---

## 1. 全局配置文件（`<uesf-home>/config.yml`）

由 `uesf config set` 命令修改，或手动编辑。

```yaml
data_dir: ~/eeg_data
default_device: cuda
num_workers: 4
log_level: INFO
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data_dir` | string | `<uesf-home>/data` | 数据文件存储目录（`raw/`、`preprocessed/`、`masked/` 均在此目录下） |
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
          method: picard          # MNE ICA() 参数全部透传
          n_components: null
          random_state: 97
          max_iter: 1000
          eog_ch_names: [Fp1, Fp2]  # 作为 EOG 参考的通道名
          figures_dir: ./ica_figs   # 可选 figures 输出目录
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

以下是字段速查表（dev6 顶层 `split`，自动通道接线；配置校验规则编号参见 `docs/version_design/0.1.0.dev6_design/07_config_validation_rules.md`）：

### 顶层字段

| 字段 | 是否必填 | 类型 | 说明 |
|------|----------|------|------|
| `experiment_name` | 必填 | string | 实验名称（与文件名一致） |
| `description` | 可选 | string | 实验描述 |
| `seed` | 可选 | int | 随机种子；默认 42。框架内部派生多层种子 |
| `mode` | 可选 | string | `regular`（默认）或 `uda` |
| `device` | 可选 | string | 覆盖全局 `default_device`（如 `cuda:0`） |

### model / trainer

| 字段 | 说明 |
|------|------|
| `model.name` | 组件名，对应 `project.yml` 中 `models` 块的键名 |
| `model.params` | 传入 `__init__` 的 kwargs 字典 |
| `trainer.name` / `trainer.params` | 同上；未配置时默认使用内置 `dummy_trainer` |

### datasets

| 字段 | 说明 |
|------|------|
| `datasets.<alias>.name` | 预处理数据集名称 |
| `datasets.<alias>.transforms` | 在线变换列表，每项含 `name`、`scope: per_dataset \| global`、`params` |

> dev6 移除了 `datasets.<alias>.split` —— 整个实验只有一份顶层 `split`。

### split（Regular 模式顶层必填）

| 字段 | 说明 |
|------|------|
| `split.strategy` | `holdout` 或 `k-fold` |
| `split.dimension` | `subject` / `session` / `recording` / `flatten` / `dataset`（五选一） |
| `split.shuffle` | bool，是否打乱分组顺序（默认 `true`，R27；`flatten` 强制为 `true`，R26） |
| `split.train_ratio` | holdout 必填 |
| `split.val_ratio` | holdout 与 k-fold 可选；与 `val_split` 互斥（R19） |
| `split.test_ratio` | holdout 必填 |
| `split.k` | k-fold 必填，`>1` 或 `-1`（LOOCV）；`dimension: dataset` 时 `-1` 或 `== len(datasets)`（R14） |
| `split.assign` | `dimension: dataset` + holdout 必填：`{train: [...], test: [...]}`，覆盖全部声明的 alias（R13） |
| `split.val_split` | 独立 ValSplit 子块（可选）：`{dimension, val_ratio(0 < _ < 1), shuffle?}`；主 `dimension: flatten` 要求也为 `flatten`（R30） |

### uda（`mode: uda` 时必填）

#### uda.domain（DomainPartition）

| 字段 | 说明 |
|------|------|
| `uda.domain.strategy` | `holdout` 或 `k-fold` |
| `uda.domain.dimension` | `dataset` / `subject` / `session`（R28 白名单；不支持 recording / flatten） |
| `uda.domain.shuffle` | bool，是否打乱分组顺序（默认 `true`） |
| `uda.domain.source` | `dataset + holdout` 必填：源域数据集别名列表（R8） |
| `uda.domain.target` | `dataset + holdout` 必填：目标域数据集别名（R8） |
| `uda.domain.target_count` | `subject/session + holdout` 必填：目标域组数（与 `target_ratio` 互斥，R12） |
| `uda.domain.target_ratio` | `subject/session + holdout` 必填：目标域比例（与 `target_count` 互斥，R12） |
| `uda.domain.k` | k-fold 必填；`-1` 为 leave-one-out |

#### uda.adaptation

| 字段 | 说明 |
|------|------|
| `uda.adaptation` | `transductive` 或 `inductive` |

#### uda.source（可选；省略时源域全量作训练，`source_val` 为空）

| 字段 | 说明 |
|------|------|
| `uda.source.split.dimension` | ValSplit 维度；不可与 `domain.dimension` 相同（R11） |
| `uda.source.split.val_ratio` | `[0, 1)`；`> 0` 时切出验证集 |
| `uda.source.split.shuffle` | bool，默认 `true`；`dimension: flatten` 强制为 `true` |

> `source.split` 是 ValSplit Block，不可包含 `strategy / train_ratio / test_ratio`（R10）。

#### uda.target.split（inductive 必填；transductive 不可出现）

| 字段 | 说明 |
|------|------|
| `uda.target.split.strategy` | `holdout` 或 `k-fold` |
| `uda.target.split.dimension` | 同 Split Block；不可与 `domain.dimension` 相同（R11） |
| `uda.target.split.shuffle` | bool，默认 `true` |
| `uda.target.split.train_ratio` | holdout 必填 |
| `uda.target.split.test_ratio` | holdout 必填（R4） |
| `uda.target.split.val_ratio` | 可选；与 `val_split` 互斥（R19） |
| `uda.target.split.k` | k-fold 必填（R4） |
| `uda.target.split.val_split` | 独立 ValSplit 子块（可选） |

### alignment（多数据集实验可选）

| 字段 | 说明 |
|------|------|
| `alignment.channel` | 通道对齐方式，目前仅支持 `intersection` |
| `alignment.label` | bool，是否验证标签一致性（默认 `true`） |

> 单数据集实验配置 `alignment` 时会被忽略并发出 warning（R16）。多数据集下框架同时校验 `n_samples` 一致（`ShapeMismatchError`）并对采样率不一致 warning。

### dataloaders

dev6 **自动通道接线**，无需配置。Trainer 接到的 `batch` 键：

| mode | train | val | test |
|------|-------|-----|------|
| regular（单数据集 / `dimension: dataset`） | `main` | `main` | `main` |
| uda | `source`, `target` | `source_val`, `target_val` | `main` |

### training

| 字段 | 是否必填 | 类型 | 说明 |
|------|----------|------|------|
| `epochs` | 必填 | int | 最大训练轮数 |
| `batch_size` | 可选 | int | 批次大小，默认 32 |
| `optimizer.name` | 必填 | string | 优化器名称 |
| `optimizer.params.lr` | 必填 | float | 学习率 |
| `optimizer.params.*` | 可选 | - | 其他 PyTorch 优化器参数（与官方 API 一致） |
| `gradient_clip.max_norm` | 可选 | float | 梯度最大范数 |
| `gradient_clip.norm_type` | 可选 | int | 范数类型，默认 2 |
| `scheduler.name` | 可选 | string | 调度器名称 |
| `scheduler.params` | 可选 | dict | 调度器参数（与 PyTorch 官方 API 一致） |
| `early_stopping.metric` | 可选 | string | 监控的指标名称 |
| `early_stopping.patience` | 可选 | int | 容忍轮数 |
| `early_stopping.min_delta` | 可选 | float | 最小改善量，默认 0.0 |
| `early_stopping.mode` | 可选 | string | `max` 或 `min` |
| `checkpoint.metric` | 可选 | string | 选择 best model 的指标（如 `val_accuracy`） |
| `checkpoint.mode` | 可选 | string | `max`（默认）或 `min` |
| `checkpoint.dir` | 可选 | string | 自定义 checkpoint 目录 |
| `logging.backend` | 可选 | string | 仅允许 `tensorboard`（R17） |
| `logging.log_every_n_epochs` | 可选 | int | 正整数（R18），默认 1。epoch 级写入频率 |
| `logging.log_every_n_steps` | 可选 | int | 正整数（R18a），默认 `null`。声明后每 N 个 batch 写入一次 step 级标量（tag 前缀 `step/`）；关闭则不写 step 级 |
| `logging.log_graph` | 可选 | bool | 是否记录模型计算图（R17b），默认 `false` |
| `logging.log_lr` | 可选 | bool | 是否记录学习率（R17c），默认 `true`。同时作用于 epoch 级 `lr` 与 step 级 `step/lr` |
| `logging.train_step_scalars` | 可选 | list[str] | 白名单（R35）；筛选 `training_step` 返回的数值标量，未声明时记录全部。tag 为原始 key（epoch 级）或 `step/<name>`（step 级） |
| `logging.train_metrics` | 可选 | list[str] | 训练集 computed 指标列表（R36），默认 `[]` 表示不计算。声明后框架用训练集 preds/targets 计算指定指标并以 `train_<name>` 写入；**要求 Trainer 的 `training_step` 返回 `preds` / `targets`**，否则框架 WARN 并跳过 |
| `logging.val_metrics` | 可选 | list[str] | 验证集白名单（R37）；必须是 `evaluation.metrics` 的子集。tag 为 `val_<name>`；未声明时记录 `evaluation.metrics` 全部 |
| `logging.test_metrics` | 可选 | list[str] | 测试集 computed 指标列表（R38），默认 `[]` 表示不计算。声明后框架每 `log_every_n_epochs` 执行一次测试集评估并以 `test_<name>` 写入 TB；**触发数据泄露风险 WARN**，详见设计文档 §9.13 |

> **R24**：`uda.adaptation: transductive` 时 `early_stopping` 与 `checkpoint` 不可出现。

### evaluation

| 字段 | 是否必填 | 说明 |
|------|----------|------|
| `evaluation.metrics` | 可选 | 指标名称列表，默认 `[accuracy]` |
| `evaluation.k_fold_aggregation` | 可选 | `concat`（默认）或 `mean_std` |
| `evaluation.test_with` | 可选 | `last`（默认）使用训练结束时的模型评估测试集；`best` 加载 `training.checkpoint` 保存的 `best_model.pt` 后再评估。设为 `best` 时必须同时配置 `training.checkpoint.metric`，否则触发 `ConfigError`。若训练期间从未触发 best 保存（如无验证集 / 监控指标拼写错误），框架会 WARN 并退回 `last`。 |
