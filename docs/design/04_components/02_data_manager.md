# Data Manager 详细设计

Data Manager 负责管理 UESF 中所有数据集相关的操作，包括原始数据集、预处理数据集和标签映射数据集。

目录存储结构详见 [目录规范](../03_storage/01_directory_layout.md)，数据库表结构详见 [数据库 Schema 设计](../03_storage/02_database_schema.md)。

## 1. Raw Datasets 管理机制

Raw Datasets 是原始数据集，用户需要将原始数据集组织成特定的格式，并注册到 UESF 中。仅注册而未导入的原始数据集被视作非 UESF 管理的数据集，用户可以在 UESF 中使用它们，但需自行管理存储。

UESF 提供对 Raw Datasets 的查看、注册、导入、移除、添加和修改信息、预处理等操作。

### 1.1 原始数据集目录格式

UESF 要求注册的原始数据集被组织成一个如下形式的目录：
```
<path-to-your-raw-dataset>
├── raw.yml
├── subject_01.mat
├── ...
└── subject_n.mat
```

### 1.2 `raw.yml` 配置规范

数据集文件夹下的 `raw.yml` 配置文件需遵循下面的格式：
```yaml
raw:
  name: <name>
  description: <description>
  eeg_data_key: <key>       # (必填) .mat文件中EEG数据存储的键名，如 "data"
  label_key: <key>           # (必填) .mat文件中标签存储的键名，如 "label"
  sampling_rate: <sampling-rate>
  n_subjects: <number-of-subjects>
  n_sessions: <number-of-sessions>
  n_recordings: <number-of-recordings> # 若未切分，该维度允许为1
  n_channels: <number-of-channels>
  n_samples: <number-of-samples>
  electrode_list: <list-of-electrodes>
  dimension_info: <list>     # (必填) 各维度对应的数据意义，如 ["session", "recording", "channel", "sample"]
  numeric_to_semantic:             # (必填) 数字标签与语义标签的映射，取代旧的 n_classes 字段
    0: <label-name>          # 如 0: "angry"
    1: <label-name>          # 如 1: "happy"
    ...
```

> **自动推断与一致性校验**
> `data_shape` 和 `label_shape` 字段无需用户填写，系统在注册或导入原始数据集时，会逐一读取每个被试的 `.mat` 文件，自动推断其 `data_shape` 和 `label_shape`，并校验所有被试文件的维度是否一致。若检测到不一致，系统将终止操作并报告差异详情。

### 1.3 注册与导入

用户可以通过命令将注册到 UESF 的原始数据集导入 UESF，这会将注册到 UESF 的原始数据集转存到 UESF 管理的数据目录下。导入后的数据集将存放在 UESF 管理的数据目录下，并被视为 UESF 管理的原始数据集。

UESF 管理的原始数据集将数据信息记录在 UESF 数据库中，而不使用 `raw.yml`。

UESF 管理的原始数据集存储在 UESF 管理的数据目录 `<data-dir>` 下。

### 1.4 删除操作规范与级联处理

删除原始数据集时，系统执行以下流程：
1. 在终端向用户显示确认提示，列出该原始数据集的名称及其关联的所有预处理数据集
2. 用户确认删除后，系统提示用户选择是否同时删除依赖该原始数据集的预处理数据集：
   - **同时删除**：系统在同一事务中删除原始数据集记录、关联的预处理数据集记录及其物理文件（`.npy`），并级联删除依赖这些预处理数据集的 Masked Dataset 记录
   - **保留预处理数据集**：系统仅删除原始数据集记录及其物理文件（`.mat`），并将所有依赖它的预处理数据集的 `is_orphan` 字段置为 `1`，同时将其 `source_raw_dataset_id` 置为 `NULL`
3. 用户未确认时，系统取消操作

## 2. Data Preprocessor (数据预处理器)

Data Preprocessor 是 UESF 对数据进行预处理的子模块，可以通过命令 `uesf data preprocess run` 调用。作为一个相对独立的功能，预处理模块可以结合现有的项目配置来无缝执行，也可以完全脱离项目在任意目录单独使用。

### 2.1 配置文件寻址优先级

为了保证使用的灵活性，系统对预处理配置文件的寻址与加载采取了如下的优先级顺序（由高到低）：
1. 使用命令执行时，通过参数 `--config-path` 明确指定的配置路径。
2. 在当前目录下寻找 `project.yml`，如果检测到该文件且其中填写了 `preprocess_config` 字段，则采用该字段指向的预处理配置路径。
3. 若上述皆不满足，系统默认尝试寻找当前目录下的 `preprocess.yml` 文件。

### 2.2 `preprocess.yml` 格式

`preprocess.yml` 的格式应遵循下面的格式：
```yaml
preprocess:
  source_dataset: <raw-dataset-name>  # (可选) 指定要预处理的原始数据集名称
  pipeline:
    data:                             # 纯特征预处理（滤波、去伪迹、重采样等）
      - name: <module_name>
        params:
          <parameter>: <value>
    label:                            # 纯标签预处理
      - name: <module_name>
        params:
          <parameter>: <value>
    joint:                            # 联合预处理（滑窗分段，需同时切分特征和标签）
      - name: <module_name>
        params:
          <parameter>: <value>
  out_name: <preprocessed-dataset-name>
```

Data Preprocessor 会以**列表定义的先后顺序**，严格执行 `data` 流、`label` 流，并在最后合并进入 `joint` 算子组，以保证转换逻辑的严谨与防止多模态特征的维度发散错乱。

> 预处理执行流的内存优化逻辑以及支持的内置模块（包含伪迹剔除 ICA）详见 [Preprocessing Pipeline 详细设计](06_preprocessing_pipeline.md)。

### 2.3 输入数据集指定优先级

`source_dataset` 字段用于指定要预处理的原始数据集名称。用户也可以通过 CLI 参数 `--dataset <raw-dataset-name>` 指定输入数据集。对输入数据集的指定采取如下优先级顺序（由高到低）：
1. CLI 参数 `--dataset` 明确指定的输入数据集
2. `preprocess.yml` 中 `source_dataset` 字段指定的输入数据集
3. 在当前目录下寻找 `project.yml`，如果检测到该文件且其中 `raw_datasets` 列表仅包含一个数据集，则自动采用该数据集

若以上三种方式均未能确定输入数据集，系统将终止操作并提示用户明确指定。

`out_name` 字段用于指定预处理输出的预处理数据集名称。用户也可以使用 `uesf data preprocess run --out-name <preprocessed-dataset-name>`。

### 2.4 预处理错误处理策略

预处理流程采用"严格失败"策略：在处理任何一个被试的 `.mat` 文件时，若发生文件损坏、格式异常、键名不匹配或维度不一致等任何形式的错误，系统将立即终止整个预处理流程，报告具体的出错文件名及错误详情，并清理已产生的中间文件。不支持跳过单个被试继续处理。

## 3. Preprocessed Datasets 管理机制

Preprocessed Datasets 是由 Data Preprocessor 处理后产生的由 UESF 管理的数据集。

Preprocessed Datasets 存储在 UESF 数据目录中。

Preprocessed Datasets 只能从 Raw Datasets 经 Data Preprocessor 处理后产生，不能由用户直接导入。

UESF 支持对 Preprocessed Datasets 的查询、删除操作。

> **删除操作规范**
> 删除预处理数据集时，系统在终端向用户显示确认提示，列出该数据集名称及依赖它的 Masked Dataset 列表。用户确认后，系统在同一事务中删除该预处理数据集记录、其物理文件（`.npy`）以及所有依赖它的 Masked Dataset 记录。孤儿状态的预处理数据集（`is_orphan = 1`）可正常删除，不会影响其他数据。

## 4. Masked Datasets 动态映射机制

Masked Dataset 是 UESF 解决多异构数据集标签统一定义的重要手段。其作为一等公民，在对外的调用方式上与普通的预处理数据集完全一致。

- **标签存储机制**：创建 Masked Dataset 时，系统读取源预处理数据集的标签数组，通过源数据集的 `numeric_to_semantic` 和用户提供的 `label_mapping`（旧语义→新语义）算出新数字标签（新语义标签按 ASCII 排序后依次编号 0, 1, 2, ...），将映射后的整型标签数组存储到 `<data-dir>/masked/<name>/labels.npy`。特征数据物理上不复制，运行时直接读取源预处理数据集的 `.npy` 特征张量。
- **CLI 交互生成命令**：用户需要通过专门的交互指令生成挂网规则衍生数据集：
  > 示例命令：`uesf data preprocessed mask <源数据集名> --out-name <新名称> --mapping-file rule.yml`

创建完毕后，在接下来的所有实验或项目的 YAML 配置中，用户直接填入此新名称即可使用（框架流侧完全透明隔离）。
