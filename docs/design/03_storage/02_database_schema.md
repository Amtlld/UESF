# 数据库 Schema 设计

本文档集中定义 UESF 全局 SQLite 数据库的全部表结构，以及数据库的生命周期管理策略。

核心存储原则（YAML/JSON 转换、JSON 快照、源码快照）详见 [核心交互与存储原则](../02_core_principles.md)。

---

## 1. 表结构定义

### `raw_datasets` 表

存储用户注册/导入的原始数据集元信息。

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
    n_sessions INTEGER,
    n_recordings INTEGER, -- 允许原始数据集的"recording"维度置为1（未切分）
    n_channels INTEGER,
    n_samples INTEGER,
    electrode_list TEXT, -- JSON对象，各通道对应的导联，是字符串列表
    data_shape TEXT, -- JSON对象，表示数组形状的列表，是整数列表，如 [5, 1, 32, 500]。由系统在注册/导入时自动推断，推断时需校验各被试.mat文件的data_shape是否一致
    dimension_info TEXT NOT NULL, -- JSON对象，字符串列表，表示各维度对应的数据意义，如 ["session", "recording", "channel", "sample"]。必须由用户指明
    label_shape TEXT, -- 由系统在注册/导入时自动推断，推断时需校验各被试.mat文件的label_shape是否一致
    numeric_to_semantic TEXT NOT NULL, -- 以JSON对象形式存储的数字标签与字符串语义标签的映射关系（如 {"0": "angry", "1": "happy"}）。必须由用户指明，取代旧的 categories 字段，作为标签类别的唯一定义来源
    raw_info_snapshot TEXT, -- 从用户的raw.yml提取转化的JSON对象快照
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### `preprocessed_datasets` 表

存储由 Data Preprocessor 产生的预处理数据集元信息。

```sql
CREATE TABLE preprocessed_datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    source_raw_dataset_id INTEGER, -- 引用的源原始数据集外键，用于追溯预处理数据集的来源
    data_dir_path TEXT, -- 数据集路径
    n_subjects INTEGER,
    sampling_rate REAL,
    n_sessions INTEGER,
    n_recordings INTEGER,
    n_channels INTEGER,
    n_samples INTEGER,
    electrode_list TEXT, -- JSON对象，各通道对应的导联，是字符串列表
    data_shape TEXT, -- JSON对象，表示数组形状的列表，是整数列表，如 [10, 2, 5, 32, 500]。由系统在预处理完成后自动推断
    dimension_info TEXT NOT NULL DEFAULT '["subject", "session", "recording", "channel", "sample"]', -- JSON对象，字符串列表，表示各维度对应的数据意义。原则上固定为 ["subject", "session", "recording", "channel", "sample"]，由预处理模块通过必要的数组结构调整保证维度语义正确。保留此字段以支持未来扩展
    label_shape TEXT, -- 由系统在预处理完成后自动推断
    numeric_to_semantic TEXT NOT NULL, -- 以JSON对象形式存储的数字标签与字符串语义标签的映射关系（如 {"0": "angry", "1": "happy"}）。继承自源原始数据集或通过预处理管线中的标签处理模块重新定义
    preprocess_config_snapshot TEXT, -- 预处理时输入的preprocess.yml转化的JSON形式配置快照
    is_orphan BOOLEAN DEFAULT 0, -- 孤儿标记。当源原始数据集被删除且用户选择保留该预处理数据集时，系统将此字段置为1
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(source_raw_dataset_id) REFERENCES raw_datasets(id)
);
```

### `masked_datasets` 表

存储标签映射数据集的映射信息与源挂载关系。

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

### `trainers` 表

存储训练器元信息（内置、已注册、全局三类）。

```sql
CREATE TABLE trainers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    trainer_path TEXT,
    trainer_type TEXT, -- 标记训练器类型，可能为"EMBEDDED", "REGISTERED", "GLOBAL"
    is_obsolete BOOLEAN DEFAULT 0, -- 过时标记。当 REGISTERED 类型训练器的源码变更后被重新注册时，旧记录的此字段置为 1，训练器名称同步重命名为 <name>_<sha256前8位>
    source_code_snapshot TEXT, -- 触发注册/添加事件时当前源码的全文防篡改快照
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### `models` 表

存储模型元信息（内置、已注册、全局三类）。

```sql
CREATE TABLE models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    model_path TEXT,
    model_type TEXT, -- 标记模型类型，可能为"EMBEDDED", "REGISTERED", "GLOBAL"
    is_obsolete BOOLEAN DEFAULT 0, -- 过时标记。当 REGISTERED 类型模型的源码变更后被重新注册时，旧记录的此字段置为 1，模型名称同步重命名为 <name>_<sha256前8位>
    source_code_snapshot TEXT, -- 触发注册/添加事件时当前源码的全文防篡改快照
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### `metrics` 表

存储评估指标元信息（内置、已注册、全局三类）。

```sql
CREATE TABLE metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    metric_path TEXT,
    metric_type TEXT, -- 标记指标类型，可能为"EMBEDDED", "REGISTERED", "GLOBAL"
    is_obsolete BOOLEAN DEFAULT 0, -- 过时标记。当 REGISTERED 类型指标的源码变更后被重新注册时，旧记录的此字段置为 1，指标名称同步重命名为 <name>_<sha256前8位>
    source_code_snapshot TEXT, -- 触发注册/添加事件时当前源码的全文防篡改快照。EMBEDDED 类型此字段为 NULL
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### `experiments` 表

存储实验的配置详情和评估结果。通过将配置和结果序列化为 JSON 对象并统一存储，系统能够在灵活适应不同评估指标和网络架构参数的同时，完美满足用户通过命令行进行跨项目检索、筛选和历史表现对比的需求。

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
    status TEXT DEFAULT 'PENDING',  -- 实验执行状态，取值为 PENDING, RUNNING, COMPLETED, FAILED 之一
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
> - `FAILED`：实验执行过程中发生任何未捕获异常（包括 GPU OOM、数据加载错误、用户手动中断等）时设置。错误信息应同步写入 `results` 字段的 JSON 对象中（如 `{"error": "CUDA out of memory", ...}`）
>
> 状态写入必须在 `try/except/finally` 结构中执行，确保即使进程异常退出也能正确记录最终状态。

### `configs` 表

存储系统全局配置的默认值（只读）。配置项的业务语义详见 [全局配置机制](03_global_config.md)。

```sql
CREATE TABLE configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE NOT NULL,  -- 全局配置的键名，仅允许 'data_dir', 'default_device', 'num_workers', 'log_level'
    value TEXT NOT NULL,       -- 全局配置的默认值，统一转换为 JSON 字符串格式
    description TEXT,          -- 针对该全局配置项的说明和备注
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### `schema_versions` 表

用于数据库 Schema 版本管理与迁移追踪。

```sql
CREATE TABLE schema_versions (
    version INTEGER PRIMARY KEY,
    description TEXT,
    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

---

## 2. 数据库生命周期管理

### 2.1 存储位置与初始化

UESF 的全局 SQLite 数据库文件固定存放在 `~/.uesf/uesf.db`。当用户首次执行任何 `uesf` 命令时，系统检测到该文件不存在后，将自动执行以下初始化流程：

1. 创建 `~/.uesf/` 目录（若不存在）
2. 在该目录下创建 `uesf.db` 文件
3. 执行内置的 DDL 脚本，创建所有表结构（`raw_datasets`, `preprocessed_datasets`, `masked_datasets`, `trainers`, `models`, `metrics`, `experiments`, `configs`, `schema_versions`）
4. 向 `schema_versions` 表写入当前 schema 版本号
5. 向 `configs` 表写入系统默认配置项（如 `data_dir` 默认值为 `~/.uesf/data`）

### 2.2 Schema 迁移策略

UESF 采用顺序编号的迁移脚本进行 schema 版本管理：

1. 每个 schema 变更对应一个递增编号的迁移脚本（如 `001_initial.sql`, `002_add_orphan_field.sql`）
2. 系统启动时，比对 `schema_versions` 表中的最高版本号与内置最新版本号
3. 若存在未应用的迁移，系统在事务保护下依序执行，并将迁移记录写入 `schema_versions` 表
4. 迁移失败时自动回滚事务，保持数据库在上一个稳定版本

### 2.3 事务控制策略

为保证数据一致性，所有涉及数据库写操作的业务流程必须在事务中执行：

- **原子性保证**：数据集注册/导入、预处理结果写入、实验状态更新等操作，均需在单一事务中完成"文件操作 + 数据库写入"的组合动作。若任一步骤失败，事务回滚并清理已产生的文件
- **实验状态更新**：实验开始时将状态设为 `RUNNING`，正常完成后设为 `COMPLETED`，任何未捕获异常（包括 GPU OOM）导致的中断设为 `FAILED`。状态转换在 `try/except/finally` 块中执行，确保异常场景下状态也能正确写入

### 2.4 单实例运行约束

当前版本的 UESF 限定为**单实例运行**。系统不显式进行多实例互斥检测，但在文档和帮助信息中明确提示用户：同时运行多个 `uesf` 命令（尤其是涉及写操作的命令）可能导致数据库锁冲突或数据不一致。多实例并发支持将在后续版本中通过 SQLite WAL 模式和文件锁机制进行增强。
