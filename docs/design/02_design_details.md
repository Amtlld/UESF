# 详细设计

本文档旨在阐述UESF项目的详细设计。基于《01 总体设计》中提出的架构与核心组件，本文档将深入探讨各个模块的具体实现细节、数据结构（如配置文件规范）、接口定义以及执行流程。

### UESF 核心交互与存储原则

在深入具体模块设计之前，UESF 确立以下三个核心原则，统领系统的数据交换与状态留痕机制：

1. **YAML 是与用户的接口**：YAML 仅仅是系统与用户进行可读交互的形式。UESF 内部的数据流转基于 JSON 对象。在读取用户输入的 YAML 后，系统需即刻将其转化为 JSON 对象进行后续的业务处理；同样地，在向用户输出任何 YAML 前，系统须先产生 JSON 结果对象再进行序列化转换。
2. **JSON 对象快照**：为了保证操作行为和配置的可追溯性，所有输入 UESF 的 YAML 数据对象，在其转化为 JSON 对象后，均应将其 JSON 格式保存存储到数据库中；所有输出给用户的 YAML 数据对象，产生它们的原生 JSON 对象也应该同步存储到数据库之中。
3. **源代码快照**：为了保证跨项目科学研究的绝对可重复性和底层逻辑已知可溯，当用户向 UESF 系统中添加注册全局的自定义 Trainer（训练器）或 Model（模型）等源码类组件时，系统不仅要拷贝存储这些源代码文件本身，还必须在触发添加操作时向记录的数据库表中存储该源代码的全文快照。

## 1. 目录规范与数据存储结构

### 1.1 UESF管理的文件目录设计

#### 1.1.1 UESF管理的数据集

UESF管理的数据集包括UESF管理的原始数据集和UESF管理的预处理数据集。

UESF管理的数据集存放在数据目录`<data-dir>`下。

##### 1.1.1.1 UESF管理的原始数据集

用户注册并导入的原始数据集被视为UESF管理的原始数据集。导入过程中，UESF将：
- 检查数据集是否与`raw.yml`文件中的信息一致
- 复制注册的原始数据集中的.mat文件到`<data-dir>/raw`目录下
- 将原始数据集的信息存储到数据库中

UESF管理的原始数据集存放在`<data-dir>/raw`目录下。

UESF管理的一个典型的原始数据集的目录结构如下：
```
<data-dir>/raw/<dataset-name>
├── subject_01.mat
├── ...
└── subject_n.mat
```

其中，`subject_*.mat`是原始数据集的原始数据文件，每个文件对应一个被试。文件中包含`data`和`label`两个字段，分别表示原始数据和标签。

在数据库中，UESF管理原始数据集的信息存储在`raw_datasets`表中，其结构如下：
```sql
CREATE TABLE raw_datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    n_subjects INTEGER,
    sampling_rate REAL,
    n_recordings INTEGER,
    n_channels INTEGER,
    n_samples INTEGER,
    categories TEXT,
    electrodes TEXT,
    data_shape TEXT,
    label_shape TEXT,
    label_mapping TEXT, -- 以JSON对象形式存储的数字标签与字符串语义标签的映射关系（如 {"0": "angry", "1": "happy"}）
    raw_info_snapshot TEXT, -- 从用户的raw.yml提取转化的JSON对象快照
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

##### 1.1.1.2 UESF管理的预处理数据集

UESF管理的预处理数据集存放在`<data-dir>/preprocessed`目录下。

UESF管理的一个典型的预处理数据集的目录结构如下：
```
<data-dir>/preprocessed/<dataset-name>
├── subject_01.mat
├── ...
└── subject_n.mat
```

在数据库中，UESF管理预处理数据集的信息存储在`preprocessed_datasets`表中，其结构如下：
```sql
CREATE TABLE preprocessed_datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    n_subjects INTEGER,
    sampling_rate REAL,
    n_channels INTEGER,
    n_samples INTEGER,
    categories TEXT,
    electrodes TEXT,
    data_shape TEXT,
    label_shape TEXT,
    label_mapping TEXT, -- 以JSON对象形式存储的数字标签与字符串语义标签的映射关系（如 {"0": "angry", "1": "happy"}）
    preprocess_config_snapshot TEXT, -- 预处理时输入的preprocess.yml转化的JSON形式配置快照
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

##### 1.1.1.3 UESF管理的标签映射数据集 (Masked Datasets)

为了极低成本地实现跨数据集协议层面的标签统一，UESF 允许且支持从现有的预处理数据集中创建无底层特征数据文件复制拷贝的“标签映射数据集”。

在数据库中，Masked 数据集的映射信息与源挂载被单独存储在 `masked_datasets` 表中，其结构如下：
```sql
CREATE TABLE masked_datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,         -- 给映射形成的新数据集起的独立名称
    description TEXT,                  -- 映射行为目的描述（如：四分类降维到二分类）
    source_dataset_id INTEGER,         -- 所依赖底层的源预处理数据集外键引用
    label_mapping TEXT NOT NULL,       -- 以 JSON 字符串存储的字典映射关系（如 {"angry":"negative"}）
    n_classes INTEGER,                 -- 映射压缩完成后新类别的总数
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(source_dataset_id) REFERENCES preprocessed_datasets(id)
);
```

#### 1.1.2 训练器

UESF支持用户自定义训练器，并将其注册为UESF管理的全局训练器。全局训练器存放在`~/.uesf/trainer`目录下的`<trainer_name>.py`文件中。

在数据库中，UESF管理全局训练器的信息存储在`trainers`表中，其结构如下：
```sql
CREATE TABLE trainers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    trainer_path TEXT,
    source_code_snapshot TEXT, -- 触发注册/添加事件时当前源码的全文防篡改快照
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

#### 1.1.3 模型

UESF支持用户自定义模型，并将其注册为UESF管理的全局模型。UESF管理的全局模型存放在`~/.uesf/models`目录下的`<model_name>.py`文件中。

在数据库中，UESF管理全局模型的信息存储在`models`表中，其结构如下：
```sql
CREATE TABLE models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    model_path TEXT,
    source_code_snapshot TEXT, -- 触发注册/添加事件时当前源码的全文防篡改快照
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

#### 1.1.4 实验

UESF将实验的配置详情和评估结果记录在全局数据库中。通过将配置和结果序列化为JSON对象并统一存储，系统能够在灵活适应不同评估指标和网络架构参数的同时，完美满足用户通过命令行进行跨项目检索、筛选和历史表现对比的需求。

在数据库中，这些数据统一存储在`experiments`表中，其结构设计如下：
```sql
CREATE TABLE experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_name TEXT NOT NULL,
    experiment_name TEXT NOT NULL,
    description TEXT,
    config TEXT,          -- 以JSON字符串形式存储的实验配置对象
    results TEXT,         -- 以JSON字符串形式存储的各项评估指标与结果对象
    checkpoint_dir_path TEXT, -- 对应的最佳模型权重持久化存储路径
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### 1.2 UESF项目目录设计

一个UESF项目的目录结构应遵循如下规范：
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
├── data/
|   └── preprocess.yml              # 项目级预处理配置
└── project.yml                     # 项目配置
```

> 实验名自动生成&从已有实验配置生成新实验

### 1.3 UESF全局配置

UESF的全局配置文件存放在数据库的`configs`表中，用户可以通过创建`~/.uesf/config.yml`覆写全局设置。

例如：
```yaml
# UESF管理的数据集目录
data_dir: <path-to-your-dir>
```

`configs`表结构设计如下：
```sql
CREATE TABLE configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE NOT NULL,  -- 全局配置的键名，例如 'data_dir'
    value TEXT NOT NULL,       -- 全局配置的值，统一转换为 JSON 字符串格式以便支持布尔、列表等复杂类型
    description TEXT,          -- 针对该全局配置项的说明和备注
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## 2. 核心组件详细设计

### 2.1 Project Manager 详细设计

UESF项目被定义为一套包括数据预处理和一系列实验的工作对象。一个项目包含：
- 一个或若干个预处理数据集（或原始数据集与自定义预处理流程）
- 一个或若干个模型
- 若干个实验

为了让用户可以最大程度地操作项目，UESF完全从用户定义的`project.yml`而非数据库读取项目配置信息。

UESF不在Project处实现过多的验证和限制逻辑，以保证预处理、模型和实验功能可以相对独立地使用，项目仅作为复用配置信息的中心地位。

`project.yml`应遵循下面的格式：
```yaml
project-name: example

description: <project-description>

# 如果使用预处理数据集
preprocessed_datasets:
  - <preprocessed-dataset-name>
  # - <preprocessed-dataset-name> * n

# 如果使用原始数据集与自定义预处理流程
raw_datasets:
  - <raw-dataset-name>
  # - <raw-dataset-name> * n
preprocess_config: <path-to-preprocess-config>  # 若不填写，默认为 ./preprocess.yml

trainers:
  <trainer-1>:
    entrypoint: <entrypoint>

models:
  <model-1>:
    entrypoint: <entrypoint>  # e.g. "./src/models/transformer.py:MyTransformerClass"
  ...
```

### 2.2 Data Manager 详细设计

#### 2.2.1 Raw Datasets 管理机制

Raw Datasets是原始数据集，用户需要将原始数据集组织成特定的格式，并注册到UESF中。仅注册而未导入的原始数据集被视作非UESF管理的数据集，用户可以在UESF中使用它们，但需自行管理存储。

UESF提供对Raw Datasets的查看、注册、导入、移除、添加和修改信息、预处理等操作。

UESF要求注册的原始数据集被组织成一个如下形式的目录：
```
<path-to-your-raw-dataset>
├── raw.yml
├── subject_01.mat
├── ...
└── subject_n.mat
```
数据集文件夹下的`raw.yml`配置文件需遵循下面的格式：
```yaml
raw:
  name: <name>
  description: <description>
  sampling_rate: <sampling-rate>
  n_subjects: <number-of-subjects>
  n_recordings: <number-of-recordings>
  n_channels: <number-of-channels>
  n_samples: <number-of-samples>
  n_classes: <number-of-classes>
  electrode_list: <list-of-electrodes>

```

用户可以通过命令将注册到UESF的原始数据集导入UESF，这会将注册到UESF的原始数据集转存到UESF管理的数据目录下。导入后的数据集将存放在UESF管理的数据目录下，并被视为UESF管理的原始数据集。

UESF管理的原始数据集将数据信息记录在UESF数据库中，而不使用`raw.yml`。

UESF管理的原始数据集存储在UESF管理的数据目录`<data-dir>`下。

#### 2.2.2 Data Preprocessor (数据预处理器)

Data Preprocessor是UESF对数据进行预处理的子模块，可以通过命令`uesf data preprocess run`调用。作为一个相对独立的功能，预处理模块可以结合现有的项目配置来无缝执行，也可以完全脱离项目在任意目录单独使用。为了保证这种使用的灵活性，系统对预处理配置文件的寻址与加载采取了如下的优先级顺序（由高到低）：
1. 使用命令执行时，通过参数 `--config-path` 明确指定的配置路径。
2. 在当前目录下寻找 `project.yml`，如果检测到该文件且其中填写了 `preprocess_config` 字段，则采用该字段指向的预处理配置路径。
3. 若上述皆不满足，系统默认尝试寻找当前目录下的 `preprocess.yml` 文件。

`preprocess.yml`的格式应遵循下面的格式：
```yaml
preprocess:
  pipeline:
    data:
      <module>:
        <parameter>: <value>
        ...
      ...
    label:
      <module>:
        <parameter>: <value>
        ...
      ...
  out_name: <preprocessed-dataset-name>
```

Data Preprocessor会按照`pipeline`中定义的模块顺序依次分别处理数据和标签。

> `pipeline`中支持的模块将在另一文档中详细描述。

`out_name`字段用于指定预处理输出的预处理数据集名称。用户也可以使用`uesf data preprocess run --out-name <preprocessed-dataset-name>`。

#### 2.2.3 Preprocessed Datasets 管理机制

Preprocessed Datasets是由Data Preprocessor处理后产生的由UESF管理的数据集。

Preprocessed Datasets存储在UESF数据目录中。

Preprocessed Datasets只能从Raw Datasets经Data Preprocessor处理后产生，不能由用户直接导入。

UESF支持对Preprocessed Datasets的查询、删除操作。

#### 2.2.4 Masked Datasets 动态映射机制

Masked Dataset 是 UESF 解决多异构数据集标签统一定义的重要手段。其作为一等公民，在对外的调用方式上与普通的预处理数据集完全一致，而底层则采用对用户无感的软链接（Soft Reference）与实时字典映射转换完成。

- **Dataloader 层的动态加载劫持**：当在实验配置中请求挂载 `name: <dataset-name>` 时，数据加载器会跨表检索 `masked_datasets` 表。若判定为该类衍生数据集，引擎在物理存储上照常只去读取底层原始预处理对应的庞大 `.npy` 特征张量（做到系统级零额外存储占用）；但在标签数据读入内存时，引擎会实施动态拦截：根据表内记录的 `label_mapping` JSON 规则实施逐元素（Element-wise）的新旧标签覆盖。
- **CLI 交互生成命令**：用户需要通过专门的交互指令生成挂网规则衍生数据集：
  > 示例命令：`uesf data preprocessed mask <源数据集名> --out-name <新名称> --mapping-file rule.yml`

创建完毕后，在接下来的所有实验或项目的 YAML 配置中，用户直接填入此新名称即可使用（框架流侧完全透明隔离）。


### 2.3 Model Manager 详细设计

Model Manager负责管理模型。UESF支持三类模型：
- 内置模型
- 自定义模型
- 全局自定义模型

#### 2.3.1 内置模型

UESF开发者维护一系列已发表的EEG深度学习模型，供用户使用。

使用UESF内置模型时，用户仅需在实验配置文档中指定模型名称，UESF会自动加载内置模型。
```yaml
model: "EEGConformer"
```

#### 2.3.2 自定义模型

UESF可以作为一个Python库被导入用户Python脚本中。用户可以利用UESF提供的模型基类来自定义模型。

用户通过编写继承UESF提供的模型基类的模型，并通过实验配置进行导入的，被视作用户管理的模型。

一个可用的自定义模型需要满足如下两个必要条件：
1. 自定义模型源代码存在
2. 自定义模型在项目配置文件中注册

模型在项目配置文件`project.yml`中注册的示例：
```yaml
models:
  My_Transformer:
    entrypoint: "./src/models/transformer.py:MyTransformerClass"
  ...
```

为适应极度解耦的数据流结构，自定义模型所继承的模型基类接口规范如下：
```python
import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        """基类初始化。所有的自定义模型都必须接受 kwargs，以由框架反射注入 YAML 配置参数。"""
        super().__init__()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """标准前向传播接口。"""
        raise NotImplementedError

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """(可选) 特征表征提取接口。对需要截获分类头前一层特征的复杂任务提供结构化支撑。"""
        pass
```

#### 2.3.3 全局自定义模型

UESF支持用户将自定义模型注册为UESF管理的全局自定义模型。

> 全局自定义模型可能遭遇调试困难，需要用户自行斟酌是否使用。

UESF支持用户对全局自定义模型进行添加、查看、移除、修改信息等操作。

### 2.4 Trainer Manager 详细设计

训练器（trainer）定义了训练流程。与Model Manager类似，UESF支持三类训练器：
- 内置训练器
- 自定义训练器
- 全局自定义训练器

#### 2.4.1 内置训练器

UESF开发者维护若干个常用的深度学习训练器，供用户使用。

使用UESF内置训练器时，用户需在实验配置文档中指定训练器名称，UESF会自动加载内置训练器。
```yaml
model: "EEGConformer"
trainer: "CommonTrainer"
```

#### 2.4.2 自定义训练器

UESF可以作为一个Python库被导入用户Python脚本中。用户可以利用UESF提供的训练器基类来自定义训练器。

用户通过编写继承UESF提供的训练器基类的训练器，并通过实验配置进行导入的，被视作用户管理的训练器。

一个可用的自定义训练器需要满足如下两个必要条件：
1. 自定义训练器源代码存在
2. 自定义训练器在项目配置文件中注册

训练器在项目配置文件`project.yml`中注册的示例：
```yaml
trainers:
  MyTrainer:
    entrypoint: "./src/models/my_trainer.py:MyTrainerClass"
  ...
```

为彻底贯彻控制流委托，自定义训练器必定继承的基类接口规范如下：
```python
from typing import Dict, Any, Tuple
import torch

class BaseTrainer:
    def __init__(self, model: torch.nn.Module, device: torch.device, **kwargs):
        """初始化训练器并挂载模型实例。"""
        self.model = model.to(device)
        self.device = device
        self.config = kwargs
        
    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, Any]:
        """(可选) 配置特定优化器。默认回退使用 YAML 设定的优化器流。"""
        pass 

    def training_step(self, batch: Dict[str, Tuple[torch.Tensor, torch.Tensor]], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        核心训练步委托。Runner 会将多通道 DataLoader 得到的数据组装为字典下发。
        预期返回包含 "loss" (梯度回传) 键与额外日志张量的字典。
        """
        raise NotImplementedError

    def validation_step(self, batch: Dict[str, Tuple[torch.Tensor, torch.Tensor]], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        验证步委托。
        预期直接返回 "preds" 和 "targets" 字典。Runner 会统一收集 Epoch 队列，再集中调用外部指标包计算数值结果。
        """
        raise NotImplementedError
```

#### 2.4.3 全局自定义训练器

UESF支持用户将自定义训练器注册为UESF管理的全局自定义训练器。

> 全局自定义训练器可能遭遇调试困难，需要用户自行斟酌是否使用。

UESF支持用户对全局自定义训练器进行添加、查看、移除、修改信息等操作。

### 2.5 Experiment Manager 详细设计

Experiment Manager 是 UESF 引擎的核心调度枢纽，其核心职责是将静态的配置解析为执行流。为了保证框架的绝对通用性（兼容普通分类乃至非对称架构的无监督域自适应等复杂场景），Experiment Manager 在设计上严格落实了**数据流与控制流解耦**的原则。

#### 2.5.1 核心抽象与控制反转 (Inversion of Control)

- **数据集切分器 (Splitter)**: 框架内建了一系列切分策略模块（如 `K-Fold`, `Holdout`, `Leave-One-Out`），依据 `datasets.split` 配置动态生成每一折的训练/验证/测试数据集索引快照。尤其对于跨域 EEG 问题，切分器内置了基于分组维度的隔离设计 (`dimension: subject` 等)，从根源上防止时序信息的泄露。
- **多通道字典映射加载器 (Multi-channel Dataloader Interface)**: DataLoader Builder 在实例化时并不会简单合并并输出 `(X, y)` 元组。相对的，它会根据实验配置定义的通道名（如 `src_labeled`, `tgt_unlabeled`），平行地初始化多组 DataLoader，并在最上层使用联合迭代器 (Combined Iterator) 将一个 step 内获得的数据全部打包成一个字典结构。
- **训练委托 (Delegated Training Step)**: UESF 自带的基础 `Runner` 极度瘦身，不再去维护任何特定于某种算法的 Loss 计算流。在训练步 (`training_step`) 中，`Runner` 仅将上一步组装的多通道 `batch` 字典委托下发给当前挂载的 `Trainer` 或 `Model`。任何自定义的前沿 UDA 计算图与参数更新逻辑得以完全地闭环隔离在用户的工程级代码内部。

#### 2.5.2 结果管理与断点系统

- **实验日志与指标追踪**: 原生集成标准日志库处理输出流，无缝衔接 `Weights & Biases` (W&B) 记录 Loss 收敛趋势及动态指标曲线。
- **模型检查点 (Checkpoints)**: 基于实验中给定的 `checkpoint_metric`（如 `val_f1_score`），评估阶段会触发预设的 Monitor Hook，自动存储表现最优权重的 `.pt` 或 `.pth` 快照至 `<project-dir>/experiments/results/<experiment-name>/checkpoints/` 下。

#### 2.5.3 实验配置语法结构

UESF 支持用户通过 YAML 进行灵活配置。下面是完整的 `<experiment_name>.yml` 准语法规范：

```yaml
# ====== 基础元信息 ======
name: <experiment-name>
description: <experiment-description>
seed: <int> # 随机种子，确保研究基准及数据划分的绝对可复现性

# ====== 组件挂载 (Registry Components) ======
model:
  name: <model-name>  # 对应 project.yml 中已注册的模型名
  params: { ... }     # 针对该模型初始化接收的 kwargs 字典

trainer:
  name: <trainer-name> # 对应 project.yml 中已注册的训练器名
  params: { ... }      # UDA 中的独有超参 (如 GRL_lambda 起始权重) 可被隔离于此

# ====== 数据集定义与切分策略 (Datasets & Splitters) ======
datasets:
  <dataset-alias-A>:  # 给参与实验的数据集起的临时别名
    name: <preprocessed-dataset-name-1> 
    split:
      strategy: <strategy>   # "holdout" 或 "k-fold"
      dimension: <dimension> # "subject", "dataset", "record", 或 "none"
      
      # [k-fold 专有]:
      k-folds: <int>         # 若为 "total" 或 -1 则表示留一法 (LOOCV)
      val_ratio_in_train: <float> # 每折内部用于 Early Stop 的比例
      
      # [holdout 专有]:
      train_ratio: <float>   
      val_ratio: <float>
      test_ratio: <float>

  # ... 多源域时可继续挂载 <dataset-alias-B>
  
# ====== 通道拆分与映射层 (Multi-channel Mapping) ======
# 将上面 Splitter 产生的切分相 (如 .train) 自由装配送入网络的哪一数据流入口
dataloaders:
  train:
    <channel_name>: "<dataset-alias>.<split-phase>" 
    # 例如： src_labeled: "alias_A.train"
  val:
    <channel_name>: "<dataset-alias>.val"
  test:
    <channel_name>: "<dataset-alias>.test"

# ====== 训练超参数 (Overrides) ======
training:
  epochs: <int>
  batch-size: <int>
  learning-rate: <float>
  optimizer: <optimizer>

# ====== 宏观评估与记录 ======
evaluation:
  metrics: [<metric-name>]  # e.g., ["accuracy", "f1_score", ...]

logging:
  use_wandb: <bool>
  checkpoint_metric: <metric-name>
```

#### 2.5.4 评估模块与指标体系 (Evaluator & Metrics)

UESF 提出了“数据的延迟聚合与一次性结算”指标计算方案，并通过类似于模型注入的方式，使得评价指标（Metric）也可以被灵活地定义与调用。

**1. Epoch-level Aggregation (Epoch级聚合延迟验证)**
为了避免“批次数值求平均”所导致的非平滑指标计算失真问题，UESF 采取了延时计算的设计。在底层的业务控制流中：
- `Trainer.validation_step()` 返回时不会立刻计算分类准确度，而是仅回传该批次的原始预测张量（`preds`）与对应的目标标签（`targets`）。
- UESF 底层 `Runner` 在内存/显存中自动拼接（Concat）并缓存一整个验证或测试周期的全部批次张量结果。
- 在 Epoch 周期结尾，由框架独立的 **`Evaluator`（评估执行组件）** 对这个无切分、最完整的表现全量预测张量执行一次性的评价运算。

**2. 统一 Metric 接口规范**
任何在实验中挂载的指标方法（无论是内置封装了 `scikit-learn` 或 `torchmetrics` 的函数，还是用户自定义的打分算法），都受制于统一的函数签名规范：
```python
import torch
from typing import Dict, Any

def my_metric_func(preds: torch.Tensor, targets: torch.Tensor, **kwargs) -> float | Dict[str, Any]:
    """
    统一的指标计算委托接口。
    
    :param preds: 聚合了完整 Epoch 流程数据的模型预测结局张量。
    :param targets: 对应的真实标签张量。
    :param kwargs: YAML 配置中通过字典键名注入附带的控制参数(如 average='macro')。
    :return: 返回纯数值(float)，或是一个无需经过任何转换就能直接 JSON 序列化的多层指标字典。
             所有的返回数据通过 Evaluator 都会被记录并打包放入数据库。
    """
    pass
```

**3. 项目级别的指标注册器 (Metrics Registry)**
类似于模型与训练器注册机制，`project.yml` 现同样支持以专门的配置块注册自定义指标（使得实验配置阶段可以用最直观的代码代替硬编码）：
```yaml
# project.yml 节点扩展示例
models: ...
trainers: ...

metrics:
  my_custom_score:
    entrypoint: "./src/utils/custom_metric.py:MyCustomScoreFunc"
```

由于评估结果返回被约束为基础标量或是标准的字典对象，这能保障系统可不加障碍地将数据转化为 JSON 序列，存进 `experiments` 状态追踪表的 `results` 快照块中。

#### 2.5.5 添加、删除和查询实验

UESF提供命令，允许用户添加、删除和查询实验。

1. 添加实验：用户可以添加空白实验或从现有实验添加新实验
  - 添加空白实验：UESF创建空白实验配置文件，用户需自行指定全部参数。
  - 从现有实验添加新实验：UESF复制现有实验的配置产生新实验配置，用户可以在此基础上进行修改
  - 用户使用以上两条命令创建实验时，UESF提供命令参数供用户指定实验名称等配置。若用户没有在命令中指定实验名，UESF使用项目名称和当前系统时间拼接成为新实验名
2. 删除实验：用户可以删除实验或仅删除实验结果
  - 删除实验：该命令将一并删除实验配置和实验结果（包括实验产生的检查点模型等）
  - 仅删除实验结果：该命令仅删除实验结果（包括实验产生的检查点模型等）
3. 查询实验：用户可以查询实验结果。在命令参数中，用户可以指定自己关注的指标（如精度、F1分数、查全率、查准率、AUC-ROC、混淆矩阵等）


## 3. 核心业务流程流转细节

### 3.1 预处理流程 (Data Preprocessing Workflow)

预处理流程的主要职责是将原始的脑电数据转换为标准化、可供深度学习模型直接读取的数据格式。

1. **配置解析**: 用户通过 CLI 命令启动预处理。系统首先读取 `preprocess.yml` 配置文件，解析其中定义的各项预处理操作（如滤波、通道插值、去伪影等）及其参数。
2. **数据加载**: 系统查询数据库（SQLite），获取并定位原始数据集的存放路径。随后按被试（Subject）或会话（Session）依次将 `.mat` 文件读入内存，并将数据维度统一调整为标准的 `(样本数, 通道数, 采样点数)` 格式。
3. **执行预处理**:
   - **数据处理**: 脑电信号依次经过配置文件中定义的数据处理模块，得到清洗后的波形或分段数据。
   - **标签处理**: 同步执行标签映射（如将连续值转换为分类标签）与对齐操作。
4. **数据保存与信息记录**: 为了提升模型训练时的数据加载速度，预处理后的数据将被统一保存为 `.npy` 格式（存放在 `~/.uesf/data/preprocessed/<数据集名称>` 目录下）。同时，系统会将生成的数据集信息、引用的原始数据集 ID 以及当时的预处理配置文件内容一并记录到数据库的 `preprocessed_datasets` 表中，以便后续查阅与实验追溯。

### 3.2 实验执行流程 (Experiment Execution Workflow)

实验执行流程负责将用户的实验配置转化为具体的模型训练与评估任务。

1. **配置加载与模型初始化**:
   - 系统读取具体的实验 YAML 配置文件，并与项目默认配置 (`project.yml`) 里的参数合并。
   - 根据配置中指定的 `model` 和 `trainer` 路径，系统动态导入用户编写的 Python 类并进行实例化。系统仅作代码引用加载，不会复制或移动用户的源文件。
2. **数据划分与加载**:
   - 系统根据配置文件中定义的划分策略（如留出法、交叉验证等），将选取的数据集划分为训练集、验证集和测试集的范围。
   - 系统按照用户定义的数据数据通道（如源域数据、目标域数据），分别创建对应的数据加载器（DataLoader），并将它们组合捆绑，以保证在训练时能同时平行提供各域的数据。
3. **训练循环**:
   - 系统开启由验证策略决定的训练主循环（例如，如果是 5 折交叉验证，则重复完整执行 5 折的训练和评估过程）。
   - 在训练的每一个批次（Batch）中，系统将组合好的多通道数据（例如包含源域和目标域数据的一个大字典）交由用户模型中必须实现的 `training_step()` 接口处理。模型内部自行负责计算特征和复杂损失（Loss），然后仅向框架返回一个总损失值供框架执行梯度反向传播。
4. **评估与结果保存**:
   - 根据设定，系统会自动在验证集和测试集上计算指定的评估指标（如准确率、F1 分数等）。
   - 系统会根据用户设定的监控指标（如验证集 F1 分数），自动保存表现最好的模型权重文件，将其存放在 `<项目目录>/experiments/results/<实验名>/checkpoints/` 目录下。若开启了日志追踪工具（如 W&B），则同步存储训练曲线。

## 4. CLI 接口设计 (Commands Interface)

UESF 提供了清晰的命令行接口 (CLI)。针对不同的管理层级，命令分为系统设置、数据管理、模型与训练器管理以及特定项目的实验管理四大部分。

> 提示：在执行指令时，如果命令行直接提供的参数与 YAML 配置文件中的设定发生冲突，系统将优先采用命令行中提供的参数值。

### 4.1 全局系统设置
用于修改 UESF 框架的全局默认参数。
- `uesf set <KEY> <VALUE>`: 设置全局变量。例如使用 `uesf set data_dir ~/.uesf/data`，可以更改系统管理的数据集默认存放路径。

### 4.2 数据管理命令 (`uesf data`)
用于统一管理所有的原始脑电数据集和预处理数据集。

#### 原始数据集 (Raw Data)
原始数据集的管理分为“仅注册”和“全部归档”两种方式，具体区别在于数据文件归谁保管：
- `uesf data raw register`: 将用户自行保存的原始数据集的基本信息登记到系统中，但这不会移动用户的原始数据文件。
- `uesf data raw import`: 读取用户指定的原始数据集，将其 `.mat` 数据文件以及配置复制导入到系统专门的数据目录下（例如 `<data-dir>/raw`），交由系统集中管理保管。
- `uesf data raw list` / `uesf data raw remove`: 查看已登记的原始数据集列表，或从系统中删除指定的数据集记录和关联文件。
- `uesf data raw edit`: 修改已登记的原始数据集的信息描述参数（例如补充采样频率等属性说明）。

#### 预处理数据集 (Preprocessed Data)
- `uesf data preprocess run`: 根据指定的预处理配置文件（`preprocess.yml`）独立执行读取和数据清洗操作（如滤波、分段提取），可以在命令末尾添加 `--out-name` 参数自主命名生成的新预处理数据集。
- `uesf data preprocessed list` / `uesf data preprocessed remove`: 查看或删除系统所生成的 `.npy` 格式的预处理数据集。
- `uesf data preprocessed mask`: 为现有的预处理数据集创建一个特殊的包含标签映射关系的数据集版本（Masked Dataset）。例如可以将原数据集的细致情绪分类归合为“积极”与“消极”两类，新生成的版本将直接挂靠应用这个转换规则，以节省重复存储的硬盘空间。

### 4.3 核心算法组件库 (`uesf model` & `uesf trainer`)
除了在具体的项目中局部指定使用外，本命令模块允许把编写的模型或训练规则放入系统全局共享，方便不同的任务研究反复调用：
- `uesf model add` / `uesf trainer add`: 登记自定义的模型源码或训练器路径，将其注册为系统通用的组件。
- `uesf model list` / `uesf model remove` / `uesf model edit`: 查看、删除或修改所有已登记的全局深度学习模型的记录信息。
- `uesf trainer list` / `uesf trainer remove` / `uesf trainer edit`: 查看、删除或修改所有已登记的通用训练器的功能说明信息。

### 4.4 项目与实验工程控制 (`uesf project` & `uesf experiment`)
此类命令用于直接管理针对特定科学问题的工程研究。这些命令必须在包含 `project.yml` 配置文件的工程主目录下执行。

#### 项目工作区基础操作
- `uesf project init`: 在当前使用的空白文件夹内自动创建项目所需的标准目录结构，例如生成默认的 `project.yml` 和 `experiments/` 文件夹。
- `uesf project info`: 在终端显示当前项目的健康运行状况，包括文件路径情况和系统中可以使用的模型组件。

#### 实验迭代控制
此类命令负责根据配置完成从数据处理到模型评估的流程迭代：
- `uesf experiment add`: 生成一个新的实验配置文件。系统可以生成一份完全空白的参数模板，也支持完整复制某次旧实验的全部参数设置并予以更名，方便作变量修改（例如超参数对比）。
- `uesf experiment list`: 查看归属在当前项目目录下存在的所有实验配置概览清单。
- `uesf experiment remove`: 删除某一次特定的实验记录。用户可以通过添加参数来选择只删除该次实验产生的庞大结果文件包以及模型权重，而特地保留其实验配置记录以便来日查阅。
- `uesf experiment run --exp <experiment_name>`: 实验一站式全自动执行命令。根据指定的单一实验配置文件，系统分别依次执行模型的数据读入、执行多轮次深度学习训练流循环，并在验证评估结束后输出最佳权重。
- `uesf experiment query`: 精细化比较检索面板功能。用户可以通过指定特定的实验量化考核指标（例如分类的准确率、F1 分数或是提取评估混淆矩阵等），来搜索对比历届已完成实验的综合表现分数，并由此查询对应的检查点信息。


## 5. 技术选型

- **Python 3.10+**: 作为主要开发语言。Python 3.10 及以上版本提供了完善的类型提示（Type Hinting）和模式匹配等现代语言特性，能够显著提升框架代码的可读性、程序的安全性以及后续代码的维护效率。
- **PyTorch 2.5+**: 作为核心的深度学习计算框架。PyTorch 拥有活跃的学术生态系统，其 2.5 及以上版本对动态计算图和底层算子编译有着更好的优化支持，能为脑电信号的卷积特征提取或是复杂的域自适应模型训练提供强大的硬件运算支持。
- **SQLite**: 作为本地轻量级关系型数据库。由于系统需要统一管理大量的实验变体配置和数据信息对应关系，单纯依靠文件夹层级进行查找效率极低且缺乏系统性。SQLite 不需要单独安装和配置数据库服务器，开箱即用，是管理本地科研信息极为理想的技术方案。
- **NumPy**: 用于高性能的底层科学计算和存储。由于原始的 `.mat` 格式在深度学习框架中读取较慢，系统将清洗后供模型训练的脑电数据统一转化为 NumPy 的二进制 `.npy` 格式。其数据存取速度极快，能完美消除大规模深度神经网络数据加载时的 I/O 瓶颈。
- **Typer**: 用于构建多层级的命令行接口 (CLI)。借助于 Python 原生的类型提示，Typer 能够自动生成清晰的命令行帮助文档，并为系统复杂嵌套的命令树设计提供了优雅的代码结构实现，大幅度降低了非计算机专业研究人员在终端的使用门槛。
- **Rich**: 用于终端界面的排版与日志的格式化输出。借助 Rich，系统能够在命令行终端中绘制美观的运行进度条、数据指标表格、语法高亮的命令输出等，为研究者在查阅复杂的实验结果或监控模型训练状态时提供极佳的用户交互体验。
- **MNE 或 SciPy**: 用于专业的脑电信号前端处理。MNE 是 Python 生态中具有统治地位的处理神经生理学数据的专门标准库，而 SciPy 提供了强大的基础数学算子。它们共同为原始脑电信号的降噪、滤波和频谱特征提取等预处理环节提供科学、可靠的计算基础保障。