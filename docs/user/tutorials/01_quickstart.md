# 15 分钟快速上手

本教程带你用 UESF 跑通第一次完整的 EEG 分类实验。全程只需要 CLI 命令和 YAML 配置，不需要编写任何 Python 代码（使用内置的 Dummy 模型）。

**前提条件**：已安装 UESF（`pip install uesf`），有一批 `.mat` 格式的 EEG 数据文件。

---

## 步骤 1：验证安装

```bash
uesf --version
```

---

## 步骤 2：全局配置

设置数据存储目录和默认计算设备：

```bash
uesf config set data_dir ~/eeg_data
uesf config set default_device cuda      # 没有 GPU 时填 cpu
uesf config set num_workers 4
```

查看当前配置：

```bash
uesf config show
```

输出示例：

```
┌─────────────────────────────────────┐
│         UESF 全局配置               │
├──────────────────┬──────────────────┤
│ data_dir         │ ~/eeg_data       │
│ default_device   │ cuda             │
│ num_workers      │ 4                │
│ log_level        │ INFO             │
└──────────────────┴──────────────────┘
```

---

## 步骤 3：准备并注册原始数据集

### 组织数据目录

将你的数据组织成如下结构：

```
seed_dataset/
├── raw.yml          # 数据集描述文件（下面编写）
├── subject_01.mat
├── subject_02.mat
└── ...
```

每个 `.mat` 文件对应一名被试。

### 编写 raw.yml

在数据目录下创建 `raw.yml`：

```yaml
raw:
  name: seed_raw
  description: SEED 情绪脑电数据集，3 类情绪，14 名被试
  eeg_data_key: data        # .mat 文件中 EEG 数据的键名
  label_key: label           # .mat 文件中标签的键名
  sampling_rate: 200
  n_subjects: 14
  n_sessions: 3
  n_recordings: 1
  n_channels: 62
  n_samples: 800
  dimension_info:
    - subject
    - session
    - recording
  numeric_to_semantic:
    0: negative
    1: neutral
    2: positive
```

> **注意** `eeg_data_key` 和 `label_key` 必须与你的 `.mat` 文件中实际使用的键名一致。可以用 `scipy.io.loadmat` 加载一个文件检查键名。

### 注册数据集

```bash
uesf data raw register /path/to/seed_dataset/
```

注册后数据文件保留在原位，UESF 只记录元信息。若想让 UESF 接管存储（将文件复制到 `data_dir`），改用 `import`：

```bash
uesf data raw import /path/to/seed_dataset/
```

验证注册成功：

```bash
uesf data raw list
```

---

## 步骤 4：预处理

### 编写 preprocess.yml

在工作目录创建 `preprocess.yml`：

```yaml
preprocess:
  source_dataset: seed_raw
  out_name: seed_preprocessed
  pipeline:
    data:
      - name: filter
        params: { l_freq: 1.0, h_freq: 40.0 }
      - name: resample
        params: { target_rate: 128 }
    joint:
      - name: sliding_window
        params:
          window_size_sec: 4.0
          stride_sec: 2.0
          label_strategy: mode
```

### 运行预处理

```bash
uesf data preprocess run -c preprocess.yml
```

预处理按被试逐个处理（懒加载，不会 OOM），完成后结果存为 `.npy` 文件。

验证预处理完成：

```bash
uesf data preprocessed list
```

---

## 步骤 5：初始化项目

```bash
mkdir emotion_recognition && cd emotion_recognition
uesf project init
```

`project init` 创建以下结构：

```
emotion_recognition/
├── project.yml          # 项目配置（需要编辑）
└── experiments/         # 实验配置目录
```

编辑 `project.yml`，挂载预处理数据集：

```yaml
project-name: emotion_recognition
description: SEED 情绪识别实验

preprocessed_datasets:
  - seed_preprocessed
```

> 本教程使用内置 Dummy 模型，所以 `models` 和 `trainers` 字段暂时留空。

---

## 步骤 6：配置并运行实验

创建实验配置文件：

```bash
uesf experiment add --name quickstart_exp
```

编辑生成的 `experiments/quickstart_exp.yml`：

```yaml
name: quickstart_exp
description: 快速上手测试实验
seed: 42

model:
  name: dummy        # 内置 Dummy 模型，随机输出，仅用于验证流程
  params: {}

trainer:
  name: dummy        # 内置 Dummy 训练器
  params: {}

datasets:
  main:
    name: seed_preprocessed
    split:
      strategy: holdout
      dimension: subject      # 按被试切分，防止数据泄露
      shuffle: true
      train_ratio: 0.7
      val_ratio: 0.15
      test_ratio: 0.15
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
  epochs: 3
  batch_size: 64
  optimizer:
    name: adam
    params: { lr: 0.001 }

evaluation:
  metrics: [accuracy, f1_score]

logging:
  checkpoint_metric: val_accuracy
```

运行实验：

```bash
uesf experiment run --exp quickstart_exp
```

---

## 步骤 7：查询结果

```bash
uesf experiment query --metrics accuracy,f1_score --status COMPLETED
```

输出示例：

```
┌──────────────────┬──────────┬──────────────┬──────────┐
│ 实验名           │ 状态     │ accuracy     │ f1_score │
├──────────────────┼──────────┼──────────────┼──────────┤
│ quickstart_exp   │ COMPLETED│ 0.3421       │ 0.3389   │
└──────────────────┴──────────┴──────────────┴──────────┘
```

> Dummy 模型是随机输出，3 类分类的准确率在 33% 左右是正常的。这说明整个数据流水线运行正常。

---

## 下一步

快速上手到此完成。接下来：

- 编写真正的模型：[编写自定义模型](../how-to/03_write_custom_model.md)
- 编写训练器：[编写自定义训练器](../how-to/04_write_custom_trainer.md)
- 完整的端到端教程（含 K-Fold 跨被试实验）：[端到端完整实验](02_first_experiment.md)
