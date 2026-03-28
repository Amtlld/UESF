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

其中，`subject_*.mat`是原始数据集的原始数据文件，每个文件对应一个被试。文件中包含EEG数据和标签两个字段（具体的键名由`raw.yml`中的`eeg_data_key`和`label_key`指定）。

在数据库中，UESF管理原始数据集的信息存储在`raw_datasets`表中，其结构如下：
```sql
CREATE TABLE raw_datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    is_imported BOOLEAN, -- 标记数据集是否导入，0表示仅注册，1表示已导入
    data_dir_path TEXT, -- 数据集路径
    eeg_data_key TEXT NOT NULL, -- .mat文件中EEG数据存储的键名（如 "data"）
    label_key TEXT NOT NULL,    -- .mat文件中标签存储的键名（如 "label"）
    n_subjects INTEGER,
    sampling_rate REAL,
    n_recordings INTEGER,
    n_channels INTEGER,
    n_samples INTEGER,
    electrode_list TEXT, -- JSON对象，各通道对应的导联，是字符串列表
    data_shape TEXT, -- JSON对象，表示数组形状的列表，是整数列表，如 [5, 32, 500]。由系统在注册/导入时自动推断，推断时需校验各被试.mat文件的data_shape是否一致
    dimension_info TEXT NOT NULL, -- JSON对象，字符串列表，表示各维度对应的数据意义，如 ["record", "channel", "sample"]。必须由用户指明
    label_shape TEXT, -- 由系统在注册/导入时自动推断，推断时需校验各被试.mat文件的label_shape是否一致
    numeric_to_semantic TEXT NOT NULL, -- 以JSON对象形式存储的数字标签与字符串语义标签的映射关系（如 {"0": "angry", "1": "happy"}）。必须由用户指明，取代旧的 categories 字段，作为标签类别的唯一定义来源
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
├── eeg_data.npy
└── labels.npy
```

在数据库中，UESF管理预处理数据集的信息存储在`preprocessed_datasets`表中，其结构如下：
```sql
CREATE TABLE preprocessed_datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    source_raw_dataset_id INTEGER, -- 引用的源原始数据集外键，用于追溯预处理数据集的来源
    data_dir_path TEXT, -- 数据集路径
    n_subjects INTEGER,
    sampling_rate REAL,
    n_recordings INTEGER,
    n_channels INTEGER,
    n_samples INTEGER,
    electrode_list TEXT, -- JSON对象，各通道对应的导联，是字符串列表
    data_shape TEXT, -- JSON对象，表示数组形状的列表，是整数列表，如 [10, 5, 32, 500]。由系统在预处理完成后自动推断
    dimension_info TEXT NOT NULL DEFAULT '["subject", "recording", "channel", "sample"]', -- JSON对象，字符串列表，表示各维度对应的数据意义。原则上固定为 ["subject", "recording", "channel", "sample"]，由预处理模块通过必要的数组结构调整保证维度语义正确。保留此字段以支持未来扩展
    label_shape TEXT, -- 由系统在预处理完成后自动推断
    numeric_to_semantic TEXT NOT NULL, -- 以JSON对象形式存储的数字标签与字符串语义标签的映射关系（如 {"0": "angry", "1": "happy"}）。继承自源原始数据集或通过预处理管线中的标签处理模块重新定义
    preprocess_config_snapshot TEXT, -- 预处理时输入的preprocess.yml转化的JSON形式配置快照
    is_orphan BOOLEAN DEFAULT 0, -- 孤儿标记。当源原始数据集被删除且用户选择保留该预处理数据集时，系统将此字段置为1
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(source_raw_dataset_id) REFERENCES raw_datasets(id)
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
    data_dir_path TEXT,                -- 映射后标签数组的物理存储路径（<data-dir>/masked/<name>/）
    label_mapping TEXT NOT NULL,       -- 以 JSON 字符串存储的旧语义→新语义的字典映射关系（如 {"angry":"negative", "sad":"negative", "happy":"positive"}）
    numeric_to_semantic TEXT NOT NULL, -- 映射后新的数字标签→语义标签关系（如 {"0":"negative", "1":"positive"}）。新数字标签按语义标签的 ASCII 排序依次分配 0, 1, 2, ...
    n_classes INTEGER,                 -- 映射压缩完成后新类别的总数
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(source_dataset_id) REFERENCES preprocessed_datasets(id)
);
```

#### 1.1.2 训练器

UESF支持用户自定义训练器，并将其注册为UESF管理的全局训练器。全局训练器存放在`~/.uesf/trainer`目录下的`<trainer_name>.py`文件中。

在数据库中，训练器信息存储在`trainers`表中，其结构如下：
```sql
CREATE TABLE trainers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    trainer_path TEXT,
    trainer_type TEXT, -- 标记训练器类型，可能为"EMBEDDED", "REGISTERED", "GLOBAL"
    source_code_snapshot TEXT, -- 触发注册/添加事件时当前源码的全文防篡改快照
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

#### 1.1.3 模型

UESF支持用户自定义模型，并将其注册为UESF管理的全局模型。UESF管理的全局模型存放在`~/.uesf/models`目录下的`<model_name>.py`文件中。

在数据库中，模型信息存储在`models`表中，其结构如下：
```sql
CREATE TABLE models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    model_path TEXT,
    model_type TEXT, -- 标记模型类型，可能为"EMBEDDED", "REGISTERED", "GLOBAL"
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
    model_id INTEGER,     -- 关联使用的模型外键，便于跨项目检索某模型在所有实验中的表现
    trainer_id INTEGER,   -- 关联使用的训练器外键
    config TEXT,          -- 以JSON字符串形式存储的完整实验配置对象
    results TEXT,         -- 以JSON字符串形式存储的各项评估指标与结果对象
    status TEXT DEFAULT 'PENDING',  -- 实验执行状态，如 PENDING, RUNNING, COMPLETED, FAILED, INTERRUPTED 等
    environment_snapshot TEXT, -- 运行时的系统与依赖环境快照（例如 pip freeze）以保证绝对复现
    checkpoint_dir_path TEXT, -- 对应的最佳模型权重持久化存储路径
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(model_id) REFERENCES models(id),
    FOREIGN KEY(trainer_id) REFERENCES trainers(id)
);
CREATE INDEX idx_experiments_project_name ON experiments(project_name);
CREATE INDEX idx_experiments_experiment_name ON experiments(experiment_name);
CREATE INDEX idx_experiments_model_id ON experiments(model_id);
CREATE INDEX idx_experiments_trainer_id ON experiments(trainer_id);
CREATE UNIQUE INDEX idx_experiments_project_experiment ON experiments(project_name, experiment_name);
```

> **实验状态机转换规则**
> `experiments.status` 字段遵循以下状态转换约束：
> - `PENDING`：实验配置已创建但尚未执行（初始状态）
> - `RUNNING`：实验开始执行时立即设置（在 `try` 块入口处写入）
> - `COMPLETED`：实验全部流程（含所有折的训练与评估）正常结束后设置
> - `FAILED`：实验执行过程中发生任何未捕获异常（包括 GPU OOM、数据加载错误等）时设置。错误信息应同步写入 `results` 字段的 JSON 对象中（如 `{"error": "CUDA out of memory", ...}`）
>
> 状态写入必须在 `try/except/finally` 结构中执行，确保即使进程异常退出也能正确记录最终状态。

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

### 1.3 UESF全局配置

UESF 的全局配置采用"数据库存储默认值 + 文件覆写"的双层机制。

#### 1.3.1 配置项定义

当前版本仅允许以下四个全局配置键：

| 键名 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data_dir` | string | `~/.uesf/data` | UESF 管理的数据集统一存储目录 |
| `default_device` | string | `cpu` | 默认计算设备（如 `cpu`, `cuda:0`, `cuda:1`） |
| `num_workers` | int | `4` | DataLoader 的工作进程数 |
| `log_level` | string | `INFO` | 框架日志输出级别，可选 `DEBUG`, `INFO`, `WARNING`, `ERROR` |

#### 1.3.2 存储与优先级

- **数据库 `configs` 表**：存储系统的默认配置值。数据库中的全局配置**不可更改**，仅在系统初始化时写入默认值，作为基准参照
- **用户配置文件 `~/.uesf/config.yml`**：用户可通过创建此文件覆写全局设置。**`config.yml` 的优先级高于数据库表**

系统在读取全局配置时，先从数据库加载默认值，再用 `config.yml` 中的同名键覆盖。

`config.yml` 示例：
```yaml
data_dir: /data/eeg_datasets
default_device: "cuda:0"
num_workers: 8
log_level: DEBUG
```

> **未知键警告**
> 若 `config.yml` 中出现上述四个合法键以外的键名，系统在启动时抛出警告（Warning）提示用户该键不受支持，但不终止运行。

#### 1.3.3 数据库表结构

`configs` 表结构设计如下：
```sql
CREATE TABLE configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE NOT NULL,  -- 全局配置的键名，仅允许 'data_dir', 'default_device', 'num_workers', 'log_level'
    value TEXT NOT NULL,       -- 全局配置的默认值，统一转换为 JSON 字符串格式
    description TEXT,          -- 针对该全局配置项的说明和备注
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

> 系统初始化时（参见 §1.4.1），`configs` 表将被写入上述四个键的默认值。此后该表内容保持只读，用户的个性化覆写通过 `config.yml` 完成。

### 1.4 数据库生命周期管理

#### 1.4.1 数据库存储位置与初始化

UESF 的全局 SQLite 数据库文件固定存放在 `~/.uesf/uesf.db`。当用户首次执行任何 `uesf` 命令时，系统检测到该文件不存在后，将自动执行以下初始化流程：

1. 创建 `~/.uesf/` 目录（若不存在）
2. 在该目录下创建 `uesf.db` 文件
3. 执行内置的 DDL 脚本，创建所有表结构（`raw_datasets`, `preprocessed_datasets`, `masked_datasets`, `trainers`, `models`, `experiments`, `configs`, `schema_versions`）
4. 向 `schema_versions` 表写入当前 schema 版本号
5. 向 `configs` 表写入系统默认配置项（如 `data_dir` 默认值为 `~/.uesf/data`）

`schema_versions` 表结构如下：
```sql
CREATE TABLE schema_versions (
    version INTEGER PRIMARY KEY,
    description TEXT,
    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

#### 1.4.2 Schema 迁移策略

UESF 采用顺序编号的迁移脚本进行 schema 版本管理：

1. 每个 schema 变更对应一个递增编号的迁移脚本（如 `001_initial.sql`, `002_add_orphan_field.sql`）
2. 系统启动时，比对 `schema_versions` 表中的最高版本号与内置最新版本号
3. 若存在未应用的迁移，系统在事务保护下依序执行，并将迁移记录写入 `schema_versions` 表
4. 迁移失败时自动回滚事务，保持数据库在上一个稳定版本

#### 1.4.3 事务控制策略

为保证数据一致性，所有涉及数据库写操作的业务流程必须在事务中执行：

- **原子性保证**：数据集注册/导入、预处理结果写入、实验状态更新等操作，均需在单一事务中完成"文件操作 + 数据库写入"的组合动作。若任一步骤失败，事务回滚并清理已产生的文件
- **实验状态更新**：实验开始时将状态设为 `RUNNING`，正常完成后设为 `COMPLETED`，任何未捕获异常（包括 GPU OOM）导致的中断设为 `FAILED`。状态转换在 `try/except/finally` 块中执行，确保异常场景下状态也能正确写入

#### 1.4.4 单实例运行约束

当前版本的 UESF 限定为**单实例运行**。系统不显式进行多实例互斥检测，但在文档和帮助信息中明确提示用户：同时运行多个 `uesf` 命令（尤其是涉及写操作的命令）可能导致数据库锁冲突或数据不一致。多实例并发支持将在后续版本中通过 SQLite WAL 模式和文件锁机制进行增强。

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

metrics:
  <metric-name>:
    entrypoint: <entrypoint>  # e.g. "./src/utils/custom_metric.py:MyCustomScoreFunc"
  ...
```

> **规范：组件名称解析优先级**
> 当用户在实验配置中引用模型或训练器名称时，系统按照以下优先级进行解析（由高到低）：
> 1. **项目级自定义组件**：在当前 `project.yml` 的 `models` 或 `trainers` 块中注册的组件
> 2. **全局自定义组件**（GLOBAL）：通过 `uesf model add` / `uesf trainer add` 导入的全局组件
> 3. **内置组件**（EMBEDDED）：UESF 内置提供的组件
>
> 若项目级自定义名与全局或内置名冲突，系统将优先使用项目级定义，并在日志中输出一条 Warning 提示用户存在名称遮蔽。

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
  eeg_data_key: <key>       # (必填) .mat文件中EEG数据存储的键名，如 "data"
  label_key: <key>           # (必填) .mat文件中标签存储的键名，如 "label"
  sampling_rate: <sampling-rate>
  n_subjects: <number-of-subjects>
  n_recordings: <number-of-recordings>
  n_channels: <number-of-channels>
  n_samples: <number-of-samples>
  electrode_list: <list-of-electrodes>
  dimension_info: <list>     # (必填) 各维度对应的数据意义，如 ["record", "channel", "sample"]
  numeric_to_semantic:             # (必填) 数字标签与语义标签的映射，取代旧的 n_classes 字段
    0: <label-name>          # 如 0: "angry"
    1: <label-name>          # 如 1: "happy"
    ...
```

> **自动推断与一致性校验**
> `data_shape` 和 `label_shape` 字段无需用户填写，系统在注册或导入原始数据集时，会逐一读取每个被试的 `.mat` 文件，自动推断其 `data_shape` 和 `label_shape`，并校验所有被试文件的维度是否一致。若检测到不一致，系统将终止操作并报告差异详情。

用户可以通过命令将注册到UESF的原始数据集导入UESF，这会将注册到UESF的原始数据集转存到UESF管理的数据目录下。导入后的数据集将存放在UESF管理的数据目录下，并被视为UESF管理的原始数据集。

UESF管理的原始数据集将数据信息记录在UESF数据库中，而不使用`raw.yml`。

UESF管理的原始数据集存储在UESF管理的数据目录`<data-dir>`下。

> **删除操作规范与级联处理**
> 删除原始数据集时，系统执行以下流程：
> 1. 在终端向用户显示确认提示，列出该原始数据集的名称及其关联的所有预处理数据集
> 2. 用户确认删除后，系统提示用户选择是否同时删除依赖该原始数据集的预处理数据集：
>    - **同时删除**：系统在同一事务中删除原始数据集记录、关联的预处理数据集记录及其物理文件（`.npy`），并级联删除依赖这些预处理数据集的 Masked Dataset 记录
>    - **保留预处理数据集**：系统仅删除原始数据集记录及其物理文件（`.mat`），并将所有依赖它的预处理数据集的 `is_orphan` 字段置为 `1`，同时将其 `source_raw_dataset_id` 置为 `NULL`
> 3. 用户未确认时，系统取消操作

#### 2.2.2 Data Preprocessor (数据预处理器)

Data Preprocessor是UESF对数据进行预处理的子模块，可以通过命令`uesf data preprocess run`调用。作为一个相对独立的功能，预处理模块可以结合现有的项目配置来无缝执行，也可以完全脱离项目在任意目录单独使用。为了保证这种使用的灵活性，系统对预处理配置文件的寻址与加载采取了如下的优先级顺序（由高到低）：
1. 使用命令执行时，通过参数 `--config-path` 明确指定的配置路径。
2. 在当前目录下寻找 `project.yml`，如果检测到该文件且其中填写了 `preprocess_config` 字段，则采用该字段指向的预处理配置路径。
3. 若上述皆不满足，系统默认尝试寻找当前目录下的 `preprocess.yml` 文件。

`preprocess.yml`的格式应遵循下面的格式：
```yaml
preprocess:
  source_dataset: <raw-dataset-name>  # (可选) 指定要预处理的原始数据集名称
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

`source_dataset`字段用于指定要预处理的原始数据集名称。用户也可以通过CLI参数 `--dataset <raw-dataset-name>` 指定输入数据集。对输入数据集的指定采取如下优先级顺序（由高到低）：
1. CLI 参数 `--dataset` 明确指定的输入数据集
2. `preprocess.yml` 中 `source_dataset` 字段指定的输入数据集
3. 在当前目录下寻找 `project.yml`，如果检测到该文件且其中 `raw_datasets` 列表仅包含一个数据集，则自动采用该数据集

若以上三种方式均未能确定输入数据集，系统将终止操作并提示用户明确指定。

`out_name`字段用于指定预处理输出的预处理数据集名称。用户也可以使用`uesf data preprocess run --out-name <preprocessed-dataset-name>`。

> **预处理错误处理策略**
> 预处理流程采用“严格失败”策略：在处理任何一个被试的 `.mat` 文件时，若发生文件损坏、格式异常、键名不匹配或维度不一致等任何形式的错误，系统将立即终止整个预处理流程，报告具体的出错文件名及错误详情，并清理已产生的中间文件。不支持跳过单个被试继续处理。

#### 2.2.3 Preprocessed Datasets 管理机制

Preprocessed Datasets是由Data Preprocessor处理后产生的由UESF管理的数据集。

Preprocessed Datasets存储在UESF数据目录中。

Preprocessed Datasets只能从Raw Datasets经Data Preprocessor处理后产生，不能由用户直接导入。

UESF支持对Preprocessed Datasets的查询、删除操作。

> **删除操作规范**
> 删除预处理数据集时，系统在终端向用户显示确认提示，列出该数据集名称及依赖它的 Masked Dataset 列表。用户确认后，系统在同一事务中删除该预处理数据集记录、其物理文件（`.npy`）以及所有依赖它的 Masked Dataset 记录。孤儿状态的预处理数据集（`is_orphan = 1`）可正常删除，不会影响其他数据。

#### 2.2.4 Masked Datasets 动态映射机制

Masked Dataset 是 UESF 解决多异构数据集标签统一定义的重要手段。其作为一等公民，在对外的调用方式上与普通的预处理数据集完全一致。

- **标签存储机制**：创建 Masked Dataset 时，系统读取源预处理数据集的标签数组，通过源数据集的 `numeric_to_semantic` 和用户提供的 `label_mapping`（旧语义→新语义）算出新数字标签（新语义标签按 ASCII 排序后依次编号 0, 1, 2, ...），将映射后的整型标签数组存储到 `<data-dir>/masked/<name>/labels.npy`。特征数据物理上不复制，运行时直接读取源预处理数据集的 `.npy` 特征张量。
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

> **规范：路径解析基准点**
> 任何在框架（例如 project.yml 或模型、训练器的配置中）填写的相对路径（如 `./src`），系统在底层运行解析时必须严格约定：**永远相对于 `project.yml` 所在的「项目工作目录」(Project Directory) 进行拼接**，绝不可因为用户在不同层级执行 `uesf` 命令而简单挂载当前工作目录 (CWD)。这能极大减轻代码与模型找不到路径的风险。

为适应极度解耦的数据流结构，自定义模型所继承的模型基类接口规范如下：
```python
import torch
import torch.nn as nn
from typing import Optional, List

class BaseModel(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int,
        electrode_list: Optional[List[str]] = None,
        **kwargs
    ):
        """
        基类初始化。框架在实例化模型时，会从数据集元数据中自动提取
        n_channels、n_samples、n_classes 等维度信息，并与实验配置中
        model.params 的用户自定义参数一并注入。
        
        :param n_channels: 数据集的通道数（由框架从数据集元数据自动注入）
        :param n_samples: 数据集的采样点数（由框架从数据集元数据自动注入）
        :param n_classes: 分类任务的类别数（由框架从数据集 numeric_to_semantic 自动推算注入）
        :param electrode_list: (可选) 电极列表（由框架从数据集元数据自动注入）
        :param kwargs: 实验配置中 model.params 用户自定义的额外参数
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.electrode_list = electrode_list

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """标准前向传播接口。"""
        raise NotImplementedError

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """(可选) 特征表征提取接口。对需要截获分类头前一层特征的复杂任务提供结构化支撑。"""
        pass
```

#### 2.3.3 模型管理

UESF在数据库`models`表中记录模型元信息。无论是项目级自定义组件还是全局组件，均需注册到数据库中。

UESF支持用户将自定义模型导入为UESF管理的全局自定义模型。

`models`表通过`model_type`字段记录模型类型。该字段取用下列三种可能之一：
- EMBEDDED：内嵌模型，是UESF提供的模型
- REGISTERED：已注册的自定义模型。当用户首次运行使用了未注册组件的实验时，UESF自动将该组件注册到数据库（记录 entrypoint 路径并创建源代码快照），UESF也提供显式注册命令
- GLOBAL：已导入的全局自定义模型，已注册的模型通过模型导入命令成为该类型

> 全局自定义模型可能遭遇调试困难，需要用户自行斟酌是否使用。

UESF支持用户对：
- 全局自定义模型进行查看、移除、修改信息等操作；
- 已注册的自定义模型进行查看、移除、修改信息、导入（成为全局模型）等操作；
- 未注册的自定义模型进行注册、导入（成为全局模型）等操作。

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
from typing import Dict, Any, Tuple, Optional
import torch
import warnings

class BaseTrainer:
    def __init__(self, model: torch.nn.Module, device: torch.device, **kwargs):
        """初始化训练器并挂载模型实例。"""
        self.model = model.to(device)
        self.device = device
        self.config = kwargs
        
    def configure_optimizers(self) -> Optional[Tuple[torch.optim.Optimizer, Any]]:
        """
        (可选) 配置特定优化器和学习率调度器。
        
        若此方法返回非 None 值，系统将使用其返回的优化器和调度器，
        即使实验 YAML 中同时定义了 training.optimizer 等字段，
        系统也会忽略 YAML 中的优化器配置，并在日志中发出 Warning 提示。
        
        若此方法返回 None（默认行为），系统将回退使用 YAML 配置中的
        training.optimizer / training.learning_rate 等字段构建优化器。
        
        :return: (optimizer, scheduler) 元组，或 None
        """
        return None 

    def training_step(
        self,
        batch: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        batch_idx: int,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, Any]:
        """
        核心训练步委托。梯度反向传播由 Trainer 全权负责。
        
        Runner 会将多通道 DataLoader 得到的数据组装为字典下发，
        同时将当前优化器实例一并传入。Trainer 必须在此方法内部
        完成以下完整流程：
          1. 前向传播计算 loss
          2. optimizer.zero_grad()
          3. loss.backward()
          4. (可选) 梯度裁剪
          5. optimizer.step()
        
        这意味着 Runner 不会在此方法之外调用 .backward() 或
        .step()，从而使得 Trainer 可以实现任意复杂的优化策略
        （如 GAN 的交替更新、UDA 的多阶段参数冻结等）。
        
        :param batch: 多通道数据字典，键为通道名，值为 (data, label) 元组
        :param batch_idx: 当前批次索引
        :param optimizer: 当前使用的优化器实例
        :return: 包含日志信息的字典（如 {"loss": loss_value, "lr": current_lr, ...}），
                 其中的值应为 Python 标量或已 .detach() 的张量，仅用于日志记录
        """
        raise NotImplementedError

    def validation_step(self, batch: Dict[str, Tuple[torch.Tensor, torch.Tensor]], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        验证步委托。
        预期直接返回 "preds" 和 "targets" 字典。Runner 会统一收集 Epoch 队列，再集中调用外部指标包计算数值结果。
        """
        raise NotImplementedError
```

#### 2.4.3 训练器管理

UESF在数据库`trainers`表中记录训练器元信息。无论是项目级自定义组件还是全局组件，均需注册到数据库中。

UESF支持用户将自定义训练器导入为UESF管理的全局自定义训练器。

`trainers`表通过`trainer_type`字段记录训练器类型。该字段取用下列三种可能之一：
- EMBEDDED：内嵌训练器，是UESF提供的训练器
- REGISTERED：已注册的自定义训练器。当用户首次运行使用了未注册组件的实验时，UESF自动将该组件注册到数据库（记录 entrypoint 路径并创建源代码快照），UESF也提供显式注册命令
- GLOBAL：已导入的全局自定义训练器，已注册的训练器通过训练器导入命令成为该类型

> 全局自定义训练器可能遭遇调试困难，需要用户自行斟酌是否使用。

UESF支持用户对：
- 全局自定义训练器进行查看、移除、修改信息等操作；
- 已注册的自定义训练器进行查看、移除、修改信息、导入（成为全局训练器）等操作；
- 未注册的自定义训练器进行注册、导入（成为全局训练器）等操作。

### 2.5 Experiment Manager 详细设计

Experiment Manager 是 UESF 引擎的核心调度枢纽，其核心职责是将静态的配置解析为执行流。为了保证框架的绝对通用性（兼容普通分类乃至非对称架构的无监督域自适应等复杂场景），Experiment Manager 在设计上严格落实了**数据流与控制流解耦**的原则。

#### 2.5.1 核心抽象与控制反转 (Inversion of Control)

- **数据集切分器 (Splitter)**: 框架内建了一系列切分策略模块（如 `K-Fold`, `Holdout`, `Leave-One-Out`），依据 `datasets.split` 配置动态生成每一折的训练/验证/测试数据集索引快照。尤其对于跨域 EEG 问题，切分器内置了基于分组维度的隔离设计 (`dimension: subject` 等)，从根源上防止时序信息的泄露。
- **多通道字典映射加载器 (Multi-channel Dataloader Interface)**: DataLoader Builder 在实例化时并不会简单合并并输出 `(X, y)` 元组。相对的，它会根据实验配置定义的通道名（如 `src_labeled`, `tgt_unlabeled`），平行地初始化多组 DataLoader，并在最上层使用联合迭代器 (Combined Iterator) 将一个 step 内获得的数据全部打包成一个字典结构。
- **训练委托 (Delegated Training Step)**: UESF 自带的基础 `Runner` 极度瘦身，不再去维护任何特定于某种算法的 Loss 计算流与梯度管理。在训练步 (`training_step`) 中，`Runner` 仅将上一步组装的多通道 `batch` 字典与优化器实例委托下发给当前挂载的 `Trainer`。**梯度的反向传播（`loss.backward()`）和优化器步进（`optimizer.step()`）由 Trainer 全权负责**，Runner 不介入任何优化过程。这一设计使得任何自定义的前沿 UDA 计算图、GAN 交替更新以及多阶段参数冻结等复杂优化策略得以完全地闭环隔离在用户的工程级代码内部。

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
# 注意：若 Trainer.configure_optimizers() 返回了非 None 值，
#       下方 optimizer / learning_rate / weight_decay / scheduler 将被忽略，
#       系统会在日志中发出 Warning 提示。
training:
  epochs: <int>
  batch_size: <int>
  learning_rate: <float>
  optimizer: <optimizer>          # 优化器名称，如 "adam", "sgd", "adamw" 等。仅支持 PyTorch 内置优化器，名称到具体类的映射规则由另外的文档说明。若需自定义优化器，用户可通过继承 BaseTrainer 并重写 configure_optimizers() 方法实现
  weight_decay: <float>           # (可选) 权重衰减系数，默认 0.0
  gradient_clip:                  # (可选) 梯度裁剪配置
    max_norm: <float>             # 梯度最大范数，如 1.0
    norm_type: <int>              # (可选) 范数类型，默认 2
  scheduler:                      # (可选) 学习率调度器配置
    name: <scheduler-name>        # 调度器名称，如 "cosine_annealing", "step_lr", "reduce_on_plateau" 等
    params: { ... }               # 调度器参数（如 {T_max: 50, eta_min: 1e-6}）
  early_stopping:                 # (可选) 早停策略配置
    monitor: <metric-name>        # 监控的指标名称，如 "val_loss", "val_accuracy"
    patience: <int>               # 指标无改善的容忍轮数
    min_delta: <float>            # (可选) 视为改善的最小变化量，默认 0.0
    mode: <mode>                  # (可选) "min"（指标越小越好）或 "max"（指标越大越好），默认 "min"

# ====== 宏观评估与记录 ======
evaluation:
  metrics: [<metric-name>]  # e.g., ["accuracy", "f1_score", ...]
  k_fold_aggregation: "concat" # 针对交叉验证时多折结果的汇总策略：可选 "concat"（全局拼接计算，推荐）或 "mean_std"（各折取平均与方差）

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

> **针对交叉验证（如 K-Fold）的综合评估策略规范：**
> 在进行多折验证时，由于各折间数据量可能存在差异或标签分布不平衡，针对指标的最终取值，UESF 在实验配置的 `evaluation: k_fold_aggregation` 中提供参数接口供用户自主选择：
> - **`concat` (大满贯模式, 推荐)**：系统会将验证集中生成的完整 `preds`、`targets` 张量全局拼接聚合，并在完全落幕后执行一次全维度指标计算运算；
> - **`mean_std` (独立平均模式)**：维持传统作法，系统独立得出单折精度，最终求取所有单折指标集合的均值（Mean）与标准差（Std）。
> 这样设计一方面保障了由于各折不平衡造成的统计失真的规避情况，同时也能灵活适应需要独立标准差作为置信区间的传统发文要求。

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
   - 在训练的每一个批次（Batch）中，系统将组合好的多通道数据（例如包含源域和目标域数据的一个大字典）连同优化器实例一并交由 Trainer 的 `training_step()` 接口处理。Trainer 在方法内部全权负责前向传播、损失计算、`optimizer.zero_grad()`、`loss.backward()` 以及 `optimizer.step()` 的完整优化流程，仅向框架返回包含日志信息的字典用于记录。
4. **评估与结果保存**:
   - 根据设定，系统会自动在验证集和测试集上计算指定的评估指标（如准确率、F1 分数等）。
   - 系统会根据用户设定的监控指标（如验证集 F1 分数），自动保存表现最好的模型权重文件，将其存放在 `<项目目录>/experiments/results/<实验名>/checkpoints/` 目录下。若开启了日志追踪工具（如 W&B），则同步存储训练曲线。
5. **实验日志记录**:
   - 实验开始执行时，系统自动创建 `<项目目录>/logs/<实验名>/` 目录，并将实验过程中的标准输出和标准错误分别重定向至 `stdout.log` 和 `stderr.log` 文件中，供用户事后审查调试。
6. **异常处理与状态记录**:
   - 实验执行过程中发生任何未捕获异常（包括但不限于 GPU OOM、数据加载错误、模型前向传播崩溃等），系统将实验状态设为 `FAILED`，将错误信息写入数据库 `experiments.results` 字段，并将异常堆栈信息记录至 `stderr.log`。实验不会自动重试。

## 4. CLI 接口设计 (Commands Interface)

UESF 提供了清晰的命令行接口 (CLI)。针对不同的管理层级，命令分为系统设置、数据管理、模型与训练器管理以及特定项目的实验管理四大部分。

> 提示：在执行指令时，如果命令行直接提供的参数与 YAML 配置文件中的设定发生冲突，系统将优先采用命令行中提供的参数值。

### 4.1 全局系统设置
用于查看和修改 UESF 框架的全局配置参数。全局配置的数据库默认值不可更改，所有用户自定义覆写通过 `~/.uesf/config.yml` 文件完成。
- `uesf config set <KEY> <VALUE>`: 将指定的全局配置键值写入 `~/.uesf/config.yml`。仅允许设置合法键名（`data_dir`, `default_device`, `num_workers`, `log_level`），若键名不合法则报错。例如：`uesf config set default_device cuda:0`。
- `uesf config show`: 在终端显示当前生效的全局配置（合并数据库默认值与 `config.yml` 覆写后的最终结果）。

### 4.2 数据管理命令 (`uesf data`)
用于统一管理所有的原始脑电数据集和预处理数据集。

#### 原始数据集 (Raw Data)
原始数据集的管理分为“仅注册”和“全部归档”两种方式，具体区别在于数据文件归谁保管：
- `uesf data raw register`: 将用户自行保存的原始数据集的基本信息登记到系统中，但这不会移动用户的原始数据文件。
- `uesf data raw import`: 读取用户指定的原始数据集，将其 `.mat` 数据文件以及配置复制导入到系统专门的数据目录下（例如 `<data-dir>/raw`），交由系统集中管理保管。
- `uesf data raw list` / `uesf data raw remove`: 查看已登记的原始数据集列表，或从系统中删除指定的数据集记录和关联文件。
- `uesf data raw edit`: 修改已登记的原始数据集的信息描述参数（例如补充采样频率等属性说明）。

#### 预处理数据集 (Preprocessed Data)
- `uesf data preprocess run`: 根据指定的预处理配置文件（`preprocess.yml`）独立执行读取和数据清洗操作（如滤波、分段提取）。可以通过 `--dataset <raw-dataset-name>` 参数指定输入的原始数据集（优先级高于 `preprocess.yml` 中的 `source_dataset` 字段），通过 `--out-name` 参数自主命名生成的新预处理数据集。
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
- `uesf experiment resume --exp <experiment_name>`: **[计划中，当前版本不实现]** 实验断点续训与异常中断恢复命令。该功能将在后续版本中实现，支持检索 `experiments` 表内的状态与环境快照，锁定最近的 Checkpoint 权重文件，并还原 DataLoader 进度及优化器状态。
- `uesf experiment query`: 精细化比较检索面板功能。用户可以通过指定特定的实验量化考核指标（例如分类的准确率、F1 分数或是提取评估混淆矩阵等），来搜索对比历届已完成实验的综合表现分数，并由此查询对应的检查点信息。


## 5. 技术选型

- **Python 3.10+**: 作为主要开发语言。Python 3.10 及以上版本提供了完善的类型提示（Type Hinting）和模式匹配等现代语言特性，能够显著提升框架代码的可读性、程序的安全性以及后续代码的维护效率。
- **PyTorch 2.5+**: 作为核心的深度学习计算框架。PyTorch 拥有活跃的学术生态系统，其 2.5 及以上版本对动态计算图和底层算子编译有着更好的优化支持，能为脑电信号的卷积特征提取或是复杂的域自适应模型训练提供强大的硬件运算支持。
- **SQLite**: 作为本地轻量级关系型数据库。由于系统需要统一管理大量的实验变体配置和数据信息对应关系，单纯依靠文件夹层级进行查找效率极低且缺乏系统性。SQLite 不需要单独安装和配置数据库服务器，开箱即用，是管理本地科研信息极为理想的技术方案。
- **NumPy**: 用于高性能的底层科学计算和存储。由于原始的 `.mat` 格式在深度学习框架中读取较慢，系统将清洗后供模型训练的脑电数据统一转化为 NumPy 的二进制 `.npy` 格式。其数据存取速度极快，能完美消除大规模深度神经网络数据加载时的 I/O 瓶颈。
- **Typer**: 用于构建多层级的命令行接口 (CLI)。借助于 Python 原生的类型提示，Typer 能够自动生成清晰的命令行帮助文档，并为系统复杂嵌套的命令树设计提供了优雅的代码结构实现，大幅度降低了非计算机专业研究人员在终端的使用门槛。
- **Rich**: 用于终端界面的排版与日志的格式化输出。借助 Rich，系统能够在命令行终端中绘制美观的运行进度条、数据指标表格、语法高亮的命令输出等，为研究者在查阅复杂的实验结果或监控模型训练状态时提供极佳的用户交互体验。
- **MNE 或 SciPy**: 用于专业的脑电信号前端处理。MNE 是 Python 生态中具有统治地位的处理神经生理学数据的专门标准库，而 SciPy 提供了强大的基础数学算子。它们共同为原始脑电信号的降噪、滤波和频谱特征提取等预处理环节提供科学、可靠的计算基础保障。