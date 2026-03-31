# 目录规范与数据存储结构

本文档定义 UESF 管理的所有文件与目录的物理存储布局。数据库表结构详见 [数据库 Schema 设计](02_database_schema.md)。

## 0. UESF Home 目录解析

UESF 的全局目录（`<uesf-home>`）按以下优先级解析：

| 优先级 | 来源 | 路径 |
|--------|------|------|
| 1（最高） | 环境变量 `UESF_HOME` | 用户显式指定的路径 |
| 2 | 虚拟环境自动检测 | `$VIRTUAL_ENV/.uesf` |
| 3（回退） | 用户主目录 | `~/.uesf` |

> **推荐安装方式**：创建一个持久化的专用虚拟环境，将 UESF 与所有研究依赖（PyTorch 等）安装在同一环境中，并将该环境固定用于所有 UESF 操作：
> ```bash
> uv venv ~/envs/eeg-research
> source ~/envs/eeg-research/bin/activate
> uv pip install uesf torch numpy ...
> ```
> 此后每次使用 `uesf` CLI 前激活该环境即可。`UESF_HOME` 将自动解析为 `~/envs/eeg-research/.uesf`。
>
> **不要使用 `uv tool install uesf`**。`uv tool` 会将 UESF 安装到一个隔离的工具环境中，该环境不含 PyTorch 等研究依赖。注册的 `model.py` / `trainer.py` 在运行时需要动态 `import` 这些包，工具环境无法满足该需求。

## 1. UESF 管理的文件目录设计

### 1.1 UESF 管理的数据集

UESF 管理的数据集包括原始数据集、预处理数据集和标签映射数据集，统一存放在数据目录 `<data-dir>` 下。

#### 1.1.1 原始数据集

用户注册并导入的原始数据集被视为 UESF 管理的原始数据集。导入过程中，UESF 将：
- 检查数据集是否与 `raw.yml` 文件中的信息一致
- 复制注册的原始数据集中的 `.mat` 文件到 `<data-dir>/raw` 目录下
- 将原始数据集的信息存储到数据库中

UESF 管理的原始数据集存放在 `<data-dir>/raw` 目录下。

一个典型的原始数据集的目录结构如下：
```
<data-dir>/raw/<dataset-name>
├── subject_01.mat
├── ...
└── subject_n.mat
```

其中，`subject_*.mat` 是原始数据集的原始数据文件，每个文件对应一个被试。文件中包含 EEG 数据和标签两个字段（具体的键名由 `raw.yml` 中的 `eeg_data_key` 和 `label_key` 指定）。

> 数据库中的存储结构详见 [`raw_datasets` 表](02_database_schema.md#raw_datasets-表)。

#### 1.1.2 预处理数据集

UESF 管理的预处理数据集存放在 `<data-dir>/preprocessed` 目录下。

一个典型的预处理数据集的目录结构如下：
```
<data-dir>/preprocessed/<dataset-name>
├── eeg_data.npy
└── labels.npy
```

> 数据库中的存储结构详见 [`preprocessed_datasets` 表](02_database_schema.md#preprocessed_datasets-表)。

#### 1.1.3 标签映射数据集 (Masked Datasets)

标签映射数据集的映射后标签数组存储在 `<data-dir>/masked/<name>/` 目录下：
```
<data-dir>/masked/<name>
└── labels.npy
```

特征数据不复制，运行时直接读取源预处理数据集的 `.npy` 特征张量。

> 数据库中的存储结构详见 [`masked_datasets` 表](02_database_schema.md#masked_datasets-表)。

### 1.2 训练器

UESF 支持用户自定义训练器，并将其注册为 UESF 管理的全局训练器。全局训练器存放在 `<uesf-home>/trainer` 目录下的 `<trainer_name>.py` 文件中。

> 数据库中的存储结构详见 [`trainers` 表](02_database_schema.md#trainers-表)。

### 1.3 模型

UESF 支持用户自定义模型，并将其注册为 UESF 管理的全局模型。UESF 管理的全局模型存放在 `<uesf-home>/models` 目录下的 `<model_name>.py` 文件中。

> 数据库中的存储结构详见 [`models` 表](02_database_schema.md#models-表)。

### 1.4 实验

实验的配置详情和评估结果记录在全局数据库中。实验相关的物理文件（检查点、日志等）存储在项目目录下，详见 [§2 项目目录结构](#2-uesf-项目目录设计)。

> 数据库中的存储结构详见 [`experiments` 表](02_database_schema.md#experiments-表)。

## 2. UESF 项目目录设计

一个 UESF 项目的目录结构应遵循如下规范：
```text
<project-dir>
├── experiments/
|   ├── configs/                    # 实验配置
|   |   ├── <exp-1-name>.yml
|   |   ├── ...
|   |   └── <exp-n-name>.yml
|   └── results/                    # 实验结果
|       ├── <exp-1-name>
|       |   ├── checkpoints/
|       |   |   ├── <checkpoint-name>.pth
|       |   |   └── ...
|       |   └── <exp-1-result>.yml
|       └── ...
├── logs/                           # 实验运行日志
|   ├── <exp-1-name>/
|   |   ├── stdout.log              # 标准输出日志
|   |   └── stderr.log              # 标准错误日志
|   └── ...
├── data/
|   └── preprocess.yml              # 项目级预处理配置
└── project.yml                     # 项目配置
```

> 实验名自动生成&从已有实验配置生成新实验
