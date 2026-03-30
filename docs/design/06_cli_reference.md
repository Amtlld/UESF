# CLI 接口参考 (Commands Interface)

UESF 提供了清晰的命令行接口 (CLI)。针对不同的管理层级，命令分为系统设置、数据管理、模型与训练器管理以及特定项目的实验管理四大部分。

> 提示：在执行指令时，如果命令行直接提供的参数与 YAML 配置文件中的设定发生冲突，系统将优先采用命令行中提供的参数值。

另见 [CLI 命令结构定义](../cli_design.yml) 获取机器可读的命令树定义。

## 1. 全局系统设置

用于查看和修改 UESF 框架的全局配置参数。全局配置的数据库默认值不可更改，所有用户自定义覆写通过 `~/.uesf/config.yml` 文件完成。配置机制详见 [全局配置机制](03_storage/03_global_config.md)。

- `uesf config set <KEY> <VALUE>`: 将指定的全局配置键值写入 `~/.uesf/config.yml`。仅允许设置合法键名（`data_dir`, `default_device`, `num_workers`, `log_level`），若键名不合法则报错。例如：`uesf config set default_device cuda:0`。
- `uesf config show`: 在终端显示当前生效的全局配置（合并数据库默认值与 `config.yml` 覆写后的最终结果）。

## 2. 数据管理命令 (`uesf data`)

用于统一管理所有的原始脑电数据集和预处理数据集。组件设计详见 [Data Manager 详细设计](04_components/02_data_manager.md)。

### 原始数据集 (Raw Data)

原始数据集的管理分为"仅注册"和"全部归档"两种方式，具体区别在于数据文件归谁保管：
- `uesf data raw register`: 将用户自行保存的原始数据集的基本信息登记到系统中，但这不会移动用户的原始数据文件。
- `uesf data raw import`: 读取用户指定的原始数据集，将其 `.mat` 数据文件以及配置复制导入到系统专门的数据目录下（例如 `<data-dir>/raw`），交由系统集中管理保管。
- `uesf data raw list` / `uesf data raw remove`: 查看已登记的原始数据集列表，或从系统中删除指定的数据集记录和关联文件。
- `uesf data raw edit`: 修改已登记的原始数据集的信息描述参数（例如补充采样频率等属性说明）。

### 预处理数据集 (Preprocessed Data)

- `uesf data preprocess run`: 根据指定的预处理配置文件（`preprocess.yml`）独立执行读取和数据清洗操作（如滤波、分段提取）。可以通过 `--dataset <raw-dataset-name>` 参数指定输入的原始数据集（优先级高于 `preprocess.yml` 中的 `source_dataset` 字段），通过 `--out-name` 参数自主命名生成的新预处理数据集。
- `uesf data preprocessed list` / `uesf data preprocessed remove`: 查看或删除系统所生成的 `.npy` 格式的预处理数据集。
- `uesf data preprocessed mask`: 为现有的预处理数据集创建一个特殊的包含标签映射关系的数据集版本（Masked Dataset）。例如可以将原数据集的细致情绪分类归合为"积极"与"消极"两类，新生成的版本将直接挂靠应用这个转换规则，以节省重复存储的硬盘空间。

## 3. 核心算法组件库 (`uesf model` & `uesf trainer` & `uesf metric`)

除了在具体的项目中局部指定使用外，本命令模块允许把编写的模型、训练规则或评估指标放入系统全局共享，方便不同的任务研究反复调用：

- `uesf model add` / `uesf trainer add` / `uesf metric add`: 登记自定义的模型、训练器或评估指标源码路径，将其注册为系统通用的全局组件。
- `uesf model list` / `uesf model remove` / `uesf model edit`: 查看、删除或修改所有已登记的全局深度学习模型的记录信息。支持 `--show-obsolete` 参数显示已过时的历史版本记录。
- `uesf trainer list` / `uesf trainer remove` / `uesf trainer edit`: 查看、删除或修改所有已登记的通用训练器的功能说明信息。支持 `--show-obsolete` 参数显示已过时的历史版本记录。
- `uesf metric list` / `uesf metric remove` / `uesf metric edit`: 查看、删除或修改所有已登记的全局评估指标的记录信息。支持 `--show-obsolete` 参数显示已过时的历史版本记录。

组件设计详见 [Model Manager](04_components/03_model_manager.md)、[Trainer Manager](04_components/04_trainer_manager.md) 和 [Experiment Manager](04_components/05_experiment_manager.md#43-指标组件管理机制)。

## 4. 项目与实验工程控制 (`uesf project` & `uesf experiment`)

此类命令用于直接管理针对特定科学问题的工程研究。这些命令必须在包含 `project.yml` 配置文件的工程主目录下执行。

### 项目工作区基础操作

- `uesf project init`: 在当前使用的空白文件夹内自动创建项目所需的标准目录结构，例如生成默认的 `project.yml` 和 `experiments/` 文件夹。
- `uesf project info`: 在终端显示当前项目的健康运行状况，包括文件路径情况和系统中可以使用的模型组件。

### 实验迭代控制

此类命令负责根据配置完成从数据处理到模型评估的流程迭代：

- `uesf experiment add`: 生成一个新的实验配置文件。系统可以生成一份完全空白的参数模板，也支持完整复制某次旧实验的全部参数设置并予以更名，方便作变量修改（例如超参数对比）。
- `uesf experiment list`: 查看归属在当前项目目录下存在的所有实验配置概览清单。
- `uesf experiment remove`: 删除某一次特定的实验记录。用户可以通过添加参数来选择只删除该次实验产生的庞大结果文件包以及模型权重，而特地保留其实验配置记录以便来日查阅。
- `uesf experiment run --exp <experiment_name>`: 实验一站式全自动执行命令。根据指定的单一实验配置文件，系统分别依次执行模型的数据读入、执行多轮次深度学习训练流循环，并在验证评估结束后输出最佳权重。
- `uesf experiment query`: 精细化比较检索面板功能。用户可以通过指定特定的实验量化考核指标（例如分类的准确率、F1 分数或是提取评估混淆矩阵等），来搜索对比历届已完成实验的综合表现分数，并由此查询对应的检查点信息。

组件设计详见 [Project Manager](04_components/01_project_manager.md) 和 [Experiment Manager](04_components/05_experiment_manager.md)。
