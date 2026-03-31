# 如何准备和注册原始数据集

本指南说明如何将你的 EEG 原始数据注册到 UESF，使其可以被预处理流水线和实验管理系统使用。

---

## 数据集目录结构要求

UESF 要求原始数据集是一个目录，包含一个 `raw.yml` 描述文件和若干 `.mat` 格式的数据文件（每个被试一个文件）：

```
seed_dataset/
├── raw.yml
├── subject_01.mat
├── subject_02.mat
├── subject_03.mat
└── ...
```

每个 `.mat` 文件存储**一名被试**的全部数据。文件内须包含两个键：EEG 信号数据和对应标签，键名可以自定义（在 `raw.yml` 中指定）。

---

## 编写 raw.yml

`raw.yml` 是对数据集的结构化描述，框架通过它理解数据的维度和语义。

**完整示例**（SEED 数据集，3 类情绪识别，62 通道，200Hz）：

```yaml
raw:
  name: seed_raw
  description: SEED 情绪脑电数据集，62 通道，200Hz，3 类情绪标签
  eeg_data_key: data          # .mat 文件中 EEG 数据的键名
  label_key: label            # .mat 文件中标签数组的键名
  sampling_rate: 200
  n_subjects: 14
  n_sessions: 3
  n_recordings: 1             # 若每个 session 只有一段连续录制，填 1
  n_channels: 62
  n_samples: 800              # 每段录制的采样点数（整条连续信号时可以省略）
  electrode_list:             # 可选，电极名称列表，框架会将其注入到模型初始化参数
    - Fp1
    - Fp2
    - F7
    - F3
    # ... 其余电极
  dimension_info:
    - subject                 # 第一维：被试
    - session                 # 第二维：会话
    - recording               # 第三维：录制段
  numeric_to_semantic:        # 数字标签到语义标签的映射
    0: negative
    1: neutral
    2: positive
```

### 字段说明

| 字段 | 是否必填 | 类型 | 说明 |
|------|----------|------|------|
| `name` | 必填 | string | 数据集名称，在 UESF 中作为唯一标识符，只能包含字母、数字和下划线 |
| `description` | 可选 | string | 数据集的文字描述 |
| `eeg_data_key` | 必填 | string | `.mat` 文件中 EEG 信号数组的键名 |
| `label_key` | 必填 | string | `.mat` 文件中标签数组的键名 |
| `sampling_rate` | 必填 | float | 采样率（Hz） |
| `n_subjects` | 必填 | int | 被试数量（即 `.mat` 文件数量） |
| `n_sessions` | 必填 | int | 每名被试的会话数 |
| `n_recordings` | 必填 | int | 每个会话的录制段数，若未切分填 1 |
| `n_channels` | 必填 | int | EEG 通道数 |
| `n_samples` | 可选 | int | 每段录制的采样点数；框架注册时会自动从文件推断并校验 |
| `electrode_list` | 可选 | list | 电极名称列表，按通道顺序排列 |
| `dimension_info` | 必填 | list | 数据维度语义，按顺序填写（通常是 `subject`、`session`、`recording`） |
| `numeric_to_semantic` | 必填 | dict | 数字标签到语义标签的映射；决定了类别数和类别名称 |

> **关于自动推断** `data_shape` 和 `label_shape` 无需填写。框架在注册或导入时会逐一读取所有 `.mat` 文件，自动推断数据形状，并校验所有文件是否维度一致。若发现不一致，操作会终止并报告差异。

### 如何确认 .mat 文件的键名

```python
import scipy.io
data = scipy.io.loadmat("subject_01.mat")
print(list(data.keys()))
# 排除以 '__' 开头的元信息键，剩余的就是数据键名
```

---

## 注册 vs. 导入：如何选择

| | 注册（register） | 导入（import） |
|--|-----------------|----------------|
| 数据文件位置 | 保留在原始路径 | 复制到 `data_dir` |
| 谁负责管理文件 | 你自己 | UESF |
| 适合场景 | 数据已在固定位置（如共享存储），不想复制 | 希望 UESF 统一管理，方便备份和迁移 |
| 删除数据集时 | 仅删除 UESF 中的元信息记录 | 同时删除 `data_dir` 中的物理文件 |

---

## 注册操作

```bash
uesf data raw register /path/to/seed_dataset/
```

UESF 会：
1. 读取 `raw.yml` 中的元信息
2. 扫描目录下所有 `.mat` 文件，校验维度一致性
3. 将数据集信息写入数据库

数据文件**不移动**，路径记录在数据库中。

---

## 导入操作

```bash
uesf data raw import /path/to/seed_dataset/
```

UESF 会：
1. 读取 `raw.yml`，校验维度一致性
2. 将所有 `.mat` 文件**复制**到 `<data_dir>/raw/seed_raw/`
3. 将数据集信息写入数据库

---

## 查看已注册的数据集

```bash
uesf data raw list
```

输出示例：

```
┌────────────┬───────────┬────────────┬──────────┬─────────────┐
│ 名称       │ 被试数    │ 通道数     │ 采样率   │ 状态        │
├────────────┼───────────┼────────────┼──────────┼─────────────┤
│ seed_raw   │ 14        │ 62         │ 200 Hz   │ REGISTERED  │
└────────────┴───────────┴────────────┴──────────┴─────────────┘
```

查看详细信息：

```bash
uesf data raw info seed_raw
```

---

## 修改数据集信息

注册后可以补充或修改描述信息：

```bash
uesf data raw edit seed_raw --description "SEED 数据集，已清理 3 名被试"
```

---

## 删除数据集

```bash
uesf data raw remove seed_raw
```

框架会列出依赖此原始数据集的所有预处理数据集，并提示你选择是否一并删除。

---

## 下一步

数据集注册完成后，进行预处理：[配置和运行预处理流水线](02_preprocessing.md)
