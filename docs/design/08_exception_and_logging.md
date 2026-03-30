# 异常处理与日志诊断体系设计草案

本文档定义了 UESF 框架的统一异常处理规范与日志诊断体系架构。作为一套面向科学研究的深度学习 CLI 工具，UESF 必须在“提升终端用户体验（隐藏晦涩堆栈，提供明确行动建议）”与“保障底层问题可追溯（详尽日志与上下文留存）”之间取得平衡。

## 1. 设计目标

1. **分级可见性**：用户在 CLI 终端看到的是格式化、语义化且带有行动建议的错误提示；开发者或进阶用户可从全局或具体实验的日志文件中获取完整的上下文与溯源码追踪（Stacktrace）。
2. **上下文感知（Context-Aware）**：日志记录须自动附带当前的执行环境（如 `project=my_proj`, `experiment=exp_01`, `component=DataPreprocessor`），不再是孤立的单行文本。
3. **隔离与追踪**：全局服务日志与实验运行流水日志严格分离，确保某次具体长时训练任务的日志可独立归档溯源。
4. **防御性失败（Fail-fast）**：框架遇到配置脏数据、接口未实现、形状错位等状态不一致时应“尽早失败”，并抛出具备业务语境的特定异常，彻底避免深入 PyTorch/NumPy 底层图层报出含糊的 RuntimeError。

## 2. 统一异常层次结构 (Exception Hierarchy)

框架底层设计统一的安全屏障拦截库级报错（如 `KeyError`, `RuntimeError`），向外抛出继承自 `UESFException` 的自定义业务异常。 

```python
Exception
 └── UESFException (框架异常基类)
      ├── ConfigError (配置类异常)
      │    ├── YAMLParseError (YAML格式结构非法)
      │    ├── MissingRequiredKeyError (缺少必要的配置键)
      │    └── TypeMismatchError (配置类型或超参数取值不合规)
      ├── ComponentError (组件注册与集成异常)
      │    ├── ComponentNotFoundError (未找到对应名称的模型或优化器)
      │    └── InterfaceViolationError (用户自定义的代码未实现 BaseModel/BaseTrainer 等约定接口)
      ├── DataError (数据处理与I/O异常)
      │    ├── DatasetNotFoundError (引用的原始/预处理数据集尚未在库中注册)
      │    ├── ShapeMismatchError (数据维度与模型输入、或标签形状的不匹配)
      │    └── MemoryOutOfBoundsError (懒加载或数据处理时超出安全内存水位)
      ├── ExperimentError (实验生命周期中断)
      │    ├── InvalidExperimentStateError (尝试修改处于运行态中的实验等非法状态转移)
      │    └── TrainingDivergenceError (发现 Loss NaN 等发散情况发生中断)
      └── StorageError (环境与持久化异常)
           ├── DatabaseLockedError (SQLite 并发操作导致死锁)
           └── SnapshotCreationError (未能成功复制项目源代码作为快照)
```

**诊断强约束设计**：所有 `UESFException` 子类实例化时强制要求包含三个维度：
- `message`: 现象描述。例：“未找到对应名称为 'adamw_custom' 的已注册组件”
- `context`: 附加上下文环境（供写文件日志使用）。例：`{"module": "TrainerManager", "config": "project.yml"}`
- `hint`: 给用户的修复建议（抛向终端）。例：“请运行 `uesf model list` 全局确认该组件，或检查项目内 YAML 配置文件路径拼写。”

## 3. 日志体系架构 (Logging Architecture)

建立基于 Python `logging` 标准库与 `Rich` UI 终端的双线并行路由。

### 3.1 预设等级域定

- **`DEBUG`**: 记录参数字典合并明细、算子处理流张量形状（Shape）变换过程、底层执行的原始 SQL 查询。日常运行时仅向全局日志文件静默倾倒；CLI 通过注入 `--debug/-d` 显式开启打印。
- **`INFO`**: 主流程生命周期事件：包含管理器初始化完毕、实验正式执行、Epoch 结转统计信息、模型权重写入等里程碑。
- **`WARNING`**: 非致命缺陷/软降级：发现 YAML 内有历史弃用字段自动过滤、数据集标签与语义不完全重叠引发隐式处理。
- **`ERROR`**: 捕获 `UESFException`；引发当前 CLI 操作中断（但不引发框架级别的全局崩溃）。
- **`CRITICAL`**: 发生未被防护的操作系统级卡死、框架源码层级 Bug 的底层抛出，将建议提 Bug Issue。

### 3.2 Logger 命名空间拓扑

采用级联的命名空间控制日志传递路由：
- `uesf` (Root 控制节点)
  - `uesf.cli` 
  - `uesf.db`
  - `uesf.manager.*` (project/data/experiment/trainer)
  - `uesf.pipeline` 
  - `uesf.user_space` (捕获并通过包装器代理用户侧 Trainer/Model 里的 print 或 log，以便集成归档)

### 3.3 日志通道 (Handlers)

1. **ConsoleHandler (终端展示层)**：
   由 `rich.logging.RichHandler` 提供终端高亮与渲染能力，不包含完整 Stack trace 栈以维持简洁（除非 Debug 模式下显式展开），日志遵循用户设定的 `config.yml -> log_level` 控制。
2. **GlobalFileHandler (系统追溯层)**：
   定位于 `~/.uesf/logs/uesf.log`，通过循环滚动（RotatingFile）记录整个框架在运行周期留下的蛛丝马迹，默认最低捕获级别为 `DEBUG` 并记录所有携带源代码的堆栈。
3. **ExperimentFileHandler (实验沙盒记录层)**：
   仅在响应 `uesf experiment run` 命令时动态实例并挂载到 `<project_dir>/experiments/<exp_id>/run.log`。旨在单独无干扰记录并追溯该特殊实验的全量日志（如每一步 Loss 指标与耗时）。

## 4. CLI 交互优化反馈展示

利用 Typer 拦截所有全局异常逃逸 (Global Exception Hook)。

- **当捕获 `UESFException` 分支**：阻止堆栈打满终端，改用 `rich.panel.Panel` 高亮输出语义化结构体：
```text
╭─ ❌ [DataError]: 预处理内存超限预警 ────────────────────────────────────╮
│ 当前在为 Subject 'S01' 运行 ICA 滤波时检测到物理内存占用已超限 (98.5%)    │
│                                                                      │
│ 上下文:                                                               │
│   - 处理管线: preprocess_pipeline.yml                                 │
│   - 数据阶段: Data Preprocessor -> filter_artifacts                   │
│                                                                      │
│ 💡 修复建议:                                                          │
│ 1. 请不要通过 --n-jobs 参数开启高并发多进程                           │
│ 2. 优先对原始数据进行下采样算子降频 (Resampling) 设计                   │
╰──────────────────────────────────────────────────────────────────────╯
```

- **当面临不可抗类型的底层 Panic**：框架输出致命崩溃指引，提示用户带上 `~/.uesf/logs/uesf.log` 在开源社区提 Issues。

## 5. 诊断辅助与容错机制 (Diagnostics Tooling)

> **注意：** 为了保障初版核心实验流程能快速落地，本章节所述的全部高级诊断与并发容错功能**暂不做实现，留待未来版本规划加入**。

### 5.1 环境快照上报 (Environment Doctor) （未来规划）
开发 `uesf doctor` (或 `uesf diagnose`) CLI 接口指令，一键输出诊断报告。自动扫描汇总以查明运行环境风险：
- 主框架版本与 Python 大版本兼容性检查。
- PyTorch 编译规格与 CUDA/ROCm/MPS 可用情况。
- SQLite 库 `~/.uesf/uesf.db` 可操作读写权及 Schema 一致性断言。
- 数据管理全景分析 (`<data-dir>` 磁盘空余探测、基础目录规范排插)。

### 5.2 离线干运行 (Dry-Run 推理) （未来规划）
为了提前验证实验运行前漫长配置链路的准确性，在执行数据预处理 (`data preprocess run`) 等重载操作之上增加 `--dry-run` 选项。仅执行逻辑闭环而不实际触发计算/读盘：
1. 校验 YAML 并执行组件库依赖推导映射。
2. 推测提取数据集尺寸元信息与模型 Tensor 形态计算验证。
避免因参数对不齐导致运行三小时后报出维度异常的心智损耗。

### 5.3 数据锁防抢占 （未来规划）
为应对大规模集群下调度重叠引发的数据竞争，实验操作需加入基于 SQLite Exclusive 及基于底层文件级的跨进程多线程原子锁。操作未果进入 Exponential Backoff 重试，代替立刻抛错。
