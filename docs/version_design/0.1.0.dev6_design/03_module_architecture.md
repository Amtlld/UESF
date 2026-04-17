# 三、模块架构设计

## 3.1 目录结构

```
src/uesf/experiment/
├── __init__.py
├── config_schema.py          # YAML 配置校验与规范化
├── splitter/                 # 拆为子包
│   ├── __init__.py           # 导出 create_splitter, SplitResult, UDASplitResult 等
│   ├── base.py               # SplitResult, MultiDatasetSplitResult, DomainSplitResult, UDASplitResult
│   ├── regular.py            # HoldoutSplitter, KFoldSplitter, DatasetLevelSplitter
│   ├── uda.py                # DatasetDomainSplitter, DimensionDomainSplitter,
│   │                         # ValSplitter, UDAOrchestrator
│   └── grouping.py           # get_groups() 维度分组工具函数
├── alignment.py              # ChannelAligner, LabelAligner（保持现有接口）
├── transforms.py             # ZScoreNormalize 等（保持现有）+ apply_transforms 应用函数
├── dataset.py                # EEGDataset（保持现有）
├── dataloader_builder.py     # 重构：自动通道映射
├── logger.py                 # 新增：TrainingLogger 协议 + TensorBoardLogger 实现
├── evaluator.py              # 保持现有
└── runner.py                 # 微调：run() 新增可选 logger 参数

src/uesf/managers/
└── experiment_manager.py     # 重构：提取 ConfigValidator + ExperimentExecutor
```

## 3.2 核心类关系

```
┌─────────────────────────────────────────────────────────┐
│                  ExperimentManager                       │
│  (生命周期管理: add / remove / list / query / run)       │
│                                                         │
│  run() → 加载配置 → 创建 DB 记录 →                       │
│          ConfigValidator.normalize() →                   │
│          ConfigValidator.validate() →                    │
│          ExperimentExecutor.execute()                    │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                  ConfigValidator                         │
│                                                         │
│  normalize(raw_config) → filled_config                  │
│  - 填充默认值（seed, shuffle, mode 等）                   │
│  - 旧键名转换（k-folds/k_folds → k）                    │
│                                                         │
│  validate(config) → None | raise ConfigError            │
│  - 校验必填字段                                          │
│  - 校验 mode 与 split/uda 互斥性                         │
│  - 校验 UDA adaptation + target.split 约束               │
│  - 校验 ratio 合法性 (sum ≈ 1.0, 范围 0~1)              │
│  - 校验 domain.dimension ≠ inner split dimension        │
│  - 校验 target_count / target_ratio 互斥                │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│               ExperimentExecutor                         │
│  execute(ctx: ExperimentContext) → ExperimentResult      │
│                                                         │
│  内部流程:                                               │
│  1. 加载数据集 → dataset_cache                           │
│  2. 跨数据集对齐（如需要）                                │
│  3. 根据 mode 分发:                                      │
│     ├─ RegularExecutionStrategy                         │
│     └─ UDAExecutionStrategy                             │
│  4. 聚合结果                                             │
└───────────────────────┬─────────────────────────────────┘
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
┌──────────────────┐  ┌───────────────────────┐
│ RegularExecution │  │  UDAExecution          │
│    Strategy      │  │    Strategy            │
│                  │  │                        │
│ split →          │  │ domain_split →         │
│ for fold:        │  │ for domain_fold:       │
│   create_logger()│  │   inner_split() →      │
│   build_data()   │  │   for inner_fold:      │
│   train()        │  │     create_logger()    │
│   evaluate()     │  │     build_data()       │
│                  │  │     train()             │
│                  │  │     evaluate()          │
└──────────────────┘  └───────────────────────┘
          │                       │
          └───────────┬───────────┘
                      ▼
        ┌──────────────────────────┐
        │     TrainingLogger       │
        │       (Protocol)         │
        │                          │
        │ log_scalars(dict, step)  │
        │ log_graph(model, input)  │
        │ close()                  │
        │                          │
        │ 实现: TensorBoardLogger  │
        └──────────────────────────┘
```

### 3.3.1 Executor 与 Strategy 接口定义

```python
# ---- managers/experiment_manager.py ----

@dataclass
class ExperimentContext:
    """实验执行上下文，由 ExperimentManager 构建后传入 Executor。"""
    config: dict                          # 规范化后的完整配置
    dataset_cache: dict[str, np.ndarray]  # alias → 5D data
    metadata_cache: dict[str, dict]       # alias → 数据集元信息（采样率等）
    experiment_id: int                    # DB 记录 ID
    output_dir: Path                      # 实验输出目录
    device: torch.device                  # 实验 YAML device 字段覆盖全局配置，由 ExperimentManager 解析后传入

@dataclass
class FoldResult:
    """单个 fold 的评估结果。"""
    metrics: dict[str, float]             # {"accuracy": 0.85, "f1_score": 0.82}
    fold_info: dict                       # 来自 SplitResult/UDASplitResult 的 fold 描述
    predictions: np.ndarray | None        # concat 模式需要保留预测结果
    labels: np.ndarray | None             # concat 模式需要保留真实标签
    failed: bool = False                  # 该 fold 是否训练失败
    error: str | None = None              # 失败时的异常信息

@dataclass
class ExperimentResult:
    """实验最终结果。"""
    fold_results: list[FoldResult]
    aggregated_metrics: dict[str, float]  # 聚合后的 metric
    aggregation_mode: str                 # "concat" | "mean_std"

class ExperimentExecutor:
    """根据 mode 分发到对应 strategy 执行。"""
    
    def execute(self, ctx: ExperimentContext) -> ExperimentResult:
        ...

class RegularExecutionStrategy:
    """Regular 模式执行策略。
    
    根据 split.dimension 走两条路径：
    
    路径 A — dimension ≠ dataset（单数据集）：
      1. create_splitter(split_config) → HoldoutSplitter | KFoldSplitter
      2. splitter.split(data_5d) → list[SplitResult]
      3. prepare_channel_data(split_result, dataset_cache, phase) 构建通道数据
    
    路径 B — dimension = dataset（跨数据集）：
      1. 直接构造 DatasetLevelSplitter（不经过 create_splitter 工厂）
      2. splitter.split(aliases) → list[DatasetLevelSplitResult]
      3. 对每折的训练数据集独立调用 ValSplitter（若配置了 val_split）
      4. 组装为 MultiDatasetSplitResult
      5. prepare_channel_data(multi_result, dataset_cache, phase) 构建通道数据
    
    transforms 的 fit/transform 在每个 fold 的数据切分后、dataloader 构建前执行，
    通过 apply_transforms_per_dataset / apply_transforms_global 函数完成：
    
    scope=per_dataset（默认）：
      调用 apply_transforms_per_dataset()
      - 对每个数据集独立创建 transform 实例
      - 单数据集 / 数据集内切分（dimension=subject/session/recording/flatten）：
        fit on train 部分，transform on train/val/test
      - 跨数据集（dimension=dataset）：
        * 训练数据集：fit on train（不含 val_split 切出的 val），transform on train + val
        * 测试数据集：fit & transform on 自身全量数据
      单数据集时等价于 scope=global。
    
    scope=global：
      调用 apply_transforms_global()
      - 创建单个 transform 实例
      - transform.fit(所有训练数据集的 train 合并数据)
      - transform.transform(所有 train/val/test 数据)
    """
    
    def run(self, ctx: ExperimentContext) -> list[FoldResult]:
        """根据 dimension 选择路径 → 遍历 fold → apply transforms → prepare data → 训练评估 → 返回各 fold 结果。"""
        ...

class UDAExecutionStrategy:
    """UDA 模式执行策略。
    
    transforms 的 fit/transform 在每个 fold 的数据切分后、dataloader 构建前执行。
    UDA 模式始终使用 per_dataset 语义，通过 apply_transforms_uda() 函数完成：
    
    1. 对每个数据集独立创建 transform 实例
    2. transform.fit(该数据集内的训练部分)
       - 源域：fit on source_train
       - 目标域 inductive：fit on target_train
       - 目标域 transductive：fit on 目标域全量数据（无 train/test 之分）
    3. transform.transform(该数据集内的所有部分)
    
    单数据集 UDA 时退化为：源域 fit on source_train，目标域 fit on target 对应部分。
    
    Checkpoint/早停：
    - inductive：基于目标域验证集（target_val）上的指标评估
    - transductive：当前版本不支持 checkpoint/早停
    """
    
    def run(self, ctx: ExperimentContext) -> list[FoldResult]:
        """创建 UDAOrchestrator → 遍历展平后的 fold → apply transforms → prepare data → 训练评估 → 返回各 fold 结果。"""
        ...
```

## 3.3 异常使用规范

本模块复用 `uesf.core.exceptions` 中已有的异常体系，各阶段抛出的异常类型如下：

| 阶段 | 异常类型 | 触发场景 |
|:-----|:---------|:---------|
| 配置校验 | `MissingRequiredKeyError` | 必填字段缺失（如 `mode=uda` 但无 `uda` 块） |
| 配置校验 | `TypeMismatchError` | 值类型或范围非法（如 `k=0`、`val_ratio=1.5`） |
| 划分阶段 | `SplitError(ExperimentError)` | 分组为空、组数不足以满足划分需求 |
| 跨数据集对齐 | `ShapeMismatchError` | 通道无交集、标签映射不一致 |
| 数据集加载 | `DatasetNotFoundError` | `datasets` 中引用的预处理数据集未注册 |

> **新增异常**：`SplitError` 作为 `ExperimentError` 的子类，专用于划分阶段。定义位于 `uesf/core/exceptions.py`：
>
> ```python
> class SplitError(ExperimentError):
>     """划分阶段异常（分组为空、组数不足等）。"""
> ```

> **原则**：所有异常必须使用框架异常体系（`UESFException` 子类），禁止裸抛 `ValueError` / `RuntimeError`。异常应携带 `context`（环境元数据）和 `hint`（用户可操作的修复建议）。

---

## 3.4 关键接口定义

### 3.4.1 配置校验

```python
# ---- experiment/config_schema.py ----

class ConfigValidator:
    """校验并规范化实验 YAML 配置。
    
    职责分离：normalize() 填充默认值，validate() 纯校验。
    """

    @staticmethod
    def normalize(raw_config: dict) -> dict:
        """填充默认值，返回规范化配置。
        
        处理内容：
        - seed 默认 42
        - mode 默认 "regular"
        - shuffle 默认 true（R27；Split Block / ValSplit / val_split 子块 / DomainPartition 统一填充）
        - 旧键名转换（k-folds / k_folds → k）
        - k-fold 无 val_split 时 val_ratio → val_ratio_in_train 换算
        - val_split 存在时提取为 val_split_config
        - source.split 无 strategy 时标记为 ValSplit 类型
        """
        ...

    @staticmethod
    def validate(config: dict) -> None:
        """校验规范化后的配置，不合法则抛出 ConfigError。
        
        校验规则见配置校验规则文档（R1–R28）。重点包括：
        - R26：dimension=flatten 时 shuffle 必须为 true（全部划分原语）
        - R28：uda.domain.dimension 仅允许 dataset|subject|session
        - R6/R15/R21：各类 ratio 值域与求和约束
        - R11：domain.dimension ≠ source/target.split.dimension
        """
        ...
```

### 3.4.2 数据结构

```python
# ---- experiment/splitter/base.py ----

@dataclass
class SplitResult:
    """一次 Split 的索引结果（train/val/test）。"""
    train_indices: np.ndarray
    val_indices: np.ndarray       # 无验证集时为 empty array
    test_indices: np.ndarray      # 无测试集时为 empty array

@dataclass
class DatasetLevelSplitResult:
    """数据集级划分结果（alias 级别，不含 sample 索引）。"""
    phase_aliases: dict[str, list[str]]  # {"train": ["ds_a"], "test": ["ds_b"]}

@dataclass
class MultiDatasetSplitResult:
    """跨数据集划分的最终结果（dimension=dataset 时使用）。
    
    由 RegularExecutionStrategy 组装：先从 DatasetLevelSplitResult 获取 alias 级
    train/test 划分，再对每个训练数据集独立调用 ValSplitter 切出 val 索引，
    最终汇总为 phase → {alias → indices} 结构。
    
    与 SplitResult（单数据集索引级）和 DatasetLevelSplitResult（alias 级无索引）
    相比，此结构同时携带多数据集信息和 sample 级索引，打通了跨数据集 val_split
    到 prepare_channel_data 的数据流。
    """
    phase_indices: dict[str, dict[str, np.ndarray]]
    # phase → {alias → flatten_3d 后的 sample indices}
    # e.g. {"train": {"ds_a": array([0,1,3,5]), "ds_b": array([0,2,4])},
    #        "val":   {"ds_a": array([2,4]),     "ds_b": array([1,3])},
    #        "test":  {"ds_c": array([0,1,2,3,4,5])}}
    #
    # 测试数据集的索引为该数据集 flatten_3d 后的全量索引。
    # 无验证集时 "val" 键对应空 dict 或各 alias 对应空数组。

@dataclass
class DomainSplitResult:
    """UDA 域划分结果 — 一个 domain fold。
    
    所有索引均为 flatten_3d 后的样本级索引空间。
    alias 信息通过 dict keys 获取（source_indices.keys() / target_indices.keys()）。
    
    对于跨数据集 UDA (dimension=dataset):
        source_indices/target_indices 各含对应域的 alias 键，
        每个 alias 的值为 range(0, N_i) 即该数据集 flatten_3d 后的全量索引。
    对于数据集内 UDA (dimension=subject/session):
        dict 中仅有单个 alias，索引为该维度分组后的样本级索引。
    """
    source_indices: dict[str, np.ndarray]  # alias → flatten_3d 后的 sample indices
    target_indices: dict[str, np.ndarray]  # alias → flatten_3d 后的 sample indices
    fold_info: dict                        # 描述信息（如哪个 subject/dataset 作为 target）

@dataclass
class UDASplitResult:
    """UDA 完整划分结果 — 域划分 + 域内划分的最终产物。"""
    source_train: dict[str, np.ndarray]   # alias → indices
    source_val: dict[str, np.ndarray]     # alias → indices
    target_train: dict[str, np.ndarray]   # alias → indices
    target_val: dict[str, np.ndarray]     # alias → indices
    target_test: dict[str, np.ndarray]    # alias → indices
    fold_info: dict                       # {"domain_fold": 0, "inner_fold": 2, ...}
```

> **transductive 模式语义**：transductive 模式下 `target_train` 和 `target_test` 均为目标域全量索引的 **copy**，二者内容相同但互不引用。`target_val` 为空数组。`target_train` 用于无监督训练，`target_test` 用于最终测试。当前版本不支持 transductive 下的 checkpoint/早停逻辑。

### 3.4.3 维度分组工具

```python
# ---- experiment/splitter/grouping.py ----

def get_groups(data: np.ndarray, dimension: str) -> list[np.ndarray]:
    """将 5-D 数据按指定维度分组，返回每组的扁平化索引列表。
    
    data shape: [n_subjects, n_sessions, n_recordings, n_channels, n_samples]
    
    仅在指定维度上产生组边界，其余维度内的数据保持完整不被拆分：
    
    dimension="subject"   → 按 subject 索引分组（n_subjects 组），
                            每组包含该 subject 所有 session、recording 的数据
    dimension="session"   → 按 session 索引分组（n_sessions 组），
                            每组包含所有 subject 在该 session 下的所有 recording 数据
    dimension="recording" → 按 recording 索引分组（n_recordings 组），
                            每组包含所有 subject、所有 session 在该 recording 下的数据
    dimension="flatten"   → 每个 (subject, session, recording) 三元组为独立一组
                            （n_subjects × n_sessions × n_recordings 组）
    
    Returns:
        groups: list[np.ndarray], 每个元素是该组在扁平化后的 sample indices
    """
    ...
```

### 3.4.4 Regular Splitters

```python
# ---- experiment/splitter/regular.py ----

class HoldoutSplitter:
    """Holdout 划分：按 ratio 切分为 train/test（或 train/val/test）。
    
    两种模式：
    - 无 val_split：三路切分，train/val/test 在同一 dimension 上按 ratio 划分
    - 有 val_split：二路切分，仅产生 train/test，验证集由 ValSplitter 从 train 中切出
    """
    
    def __init__(self, dimension: str, train_ratio: float, test_ratio: float,
                 val_ratio: float = 0.0, val_split_config: dict | None = None,
                 shuffle: bool = False, seed: int = 42):
        """
        Args:
            val_ratio: 无 val_split 时，与 train_ratio/test_ratio 同维度切出验证集。
                       有 val_split 时必须为 0。
            val_split_config: 验证集独立划分配置（含 dimension, val_ratio, shuffle）。
                              与主 val_ratio 互斥。
        """
        ...
    
    def split(self, data: np.ndarray) -> list[SplitResult]:
        """返回长度为 1 的列表。
        
        若指定 val_split_config，内部先二路切分 train/test，
        再用 ValSplitter 从 train 中切出 val。
        """
        ...

class KFoldSplitter:
    """K-Fold 划分：K 折交叉验证。"""
    
    def __init__(self, dimension: str, k: int, val_ratio: float = 0.0,
                 val_split_config: dict | None = None,
                 shuffle: bool = False, seed: int = 42):
        """
        Args:
            k: 折数。k == -1 表示 leave-one-out，运行时在 split() 中解析为
               len(get_groups(data, dimension))；若解析后折数仍为 1 或超过
               组数，抛出 SplitError。
            val_ratio: 无 val_split 时，从整体数据中切出验证集的比例。
                       内部自动换算为 val_ratio_in_train = val_ratio / (1 - 1/k)。
                       有 val_split 时必须为 0。
            val_split_config: 验证集独立划分配置（含 dimension, val_ratio, shuffle）。
                              与主 val_ratio 互斥。
        """
        ...
    
    def split(self, data: np.ndarray) -> list[SplitResult]:
        """返回长度为 k 的列表。
        
        若指定 val_split_config，每折内部先确定 train/test，
        再用 ValSplitter 从 train 中切出 val。
        """
        ...

class DatasetLevelSplitter:
    """数据集级划分：整个数据集分配到 train/test。
    
    不继承 BaseSplitter — 接口签名不同（接收 alias 列表而非 np.ndarray）。
    验证集始终通过 val_split 从训练数据集内部切出（不在 dataset 维度上切分）。
    """
    
    def __init__(self, strategy: str, assign: dict | None = None,
                 k: int | None = None, shuffle: bool = True, seed: int = 42):
        """
        Args:
            shuffle: strategy=k-fold 时控制 alias 轮换顺序（shuffle=False 按
                     datasets 声明顺序，shuffle=True 按 seed 打乱后轮换）。
                     strategy=holdout 时 shuffle 无效（assign 为显式指派）。
        """
        ...
    
    def split(self, aliases: list[str]) -> list[DatasetLevelSplitResult]:
        """
        strategy=holdout + assign: 返回长度为 1 的列表
        strategy=k-fold: 返回长度为 k 的列表（各数据集轮流做测试集）；
                        k == -1 在 split() 中解析为 len(aliases)；
                        非 -1 时要求 k == len(aliases)（R14）
        
        注意：DatasetLevelSplitResult 仅包含 alias 级划分（train/test）。
        验证集划分由 RegularExecutionStrategy 在合并训练数据后，
        使用 ValSplitter（基于 val_split 配置）完成。
        """
        ...
```

### 3.4.5 UDA Splitters

```python
# ---- experiment/splitter/uda.py ----

class DatasetDomainSplitter:
    """域划分：dimension=dataset 时，按数据集划分源域/目标域。"""
    
    def __init__(self, strategy: str, source: list[str] | None = None,
                 target: str | None = None, k: int | None = None,
                 shuffle: bool = True, seed: int = 42):
        ...
    
    def split(self, aliases: list[str]) -> list[DomainSplitResult]:
        """
        holdout: source/target 显式指定，返回长度为 1 的列表
        k-fold: 各数据集轮流做目标域，返回长度为 k 的列表
        """
        ...

class DimensionDomainSplitter:
    """域划分：dimension=subject|session 时，按维度划分源域/目标域。"""
    
    def __init__(self, strategy: str, dimension: str,
                 target_count: int | None = None,
                 target_ratio: float | None = None,
                 k: int | None = None,
                 shuffle: bool = True, seed: int = 42):
        ...
    
    def split(self, data: np.ndarray, alias: str) -> list[DomainSplitResult]:
        """
        holdout: target_count/target_ratio 决定目标域大小，返回长度为 1 的列表
        k-fold: 各 subject/session 轮流做目标域，返回长度为 k 的列表
        
        fold_info 记录每个 fold 中哪些 group 属于目标域。
        """
        ...

class ValSplitter:
    """ValSplit 原语：从给定数据中按维度分组后切出验证集。
    
    用于 source.split 和 Regular 模式 val_split。
    按 dimension 分组后，将 val_ratio 比例的组整体划入验证集。
    
    split() 接收单个数据集的 5D ndarray，在该数据集的维度空间内分组。
    跨数据集场景下，UDAOrchestrator / RegularExecutionStrategy 对每个
    数据集独立调用 split()，各数据集使用各自的维度结构，避免不同数据集
    的 subject/session/recording 命名空间混淆。
    """
    
    def __init__(self, dimension: str, val_ratio: float,
                 shuffle: bool = True, seed: int = 42):
        ...
    
    def split(self, data: np.ndarray) -> SplitResult:
        """
        返回单个 SplitResult（test_indices 为 empty array）。
        
        流程：
        1. get_groups(data, dimension) → groups
        2. 按 val_ratio 将部分 group 整体划入 val
        3. 剩余 group 为 train
        """
        ...

class UDAOrchestrator:
    """组合域划分和域内划分，生成最终的 UDASplitResult 列表。
    
    内部根据 domain.dimension 选择 DatasetDomainSplitter 或 DimensionDomainSplitter，
    调用者无需关心具体使用哪个 splitter。
    """
    
    def __init__(self, uda_config: dict, seed: int = 42):
        base_seed = seed
        
        # 根据 domain.dimension 创建合适的 domain splitter（seed + 0）
        domain_cfg = uda_config["domain"]
        if domain_cfg["dimension"] == "dataset":
            self.domain_splitter = DatasetDomainSplitter(..., seed=base_seed)
        else:
            self.domain_splitter = DimensionDomainSplitter(..., seed=base_seed)
        
        # 源域：ValSplitter（seed + 1）
        self.source_splitter = ValSplitter(..., seed=base_seed + 1)
        
        # 目标域：根据 adaptation 选择（seed + 2）
        self.adaptation = uda_config["adaptation"]
        if self.adaptation == "inductive":
            target_split_cfg = uda_config["target"]["split"]
            # 完整 Split Block → HoldoutSplitter 或 KFoldSplitter
            self.target_splitter = create_splitter(target_split_cfg, seed=base_seed + 2)
        else:
            self.target_splitter = None  # transductive: 目标域不划分
    
    def split(self, dataset_cache: dict) -> list[UDASplitResult]:
        """
        完整流程：
        1. 根据 domain.dimension 调用对应的 domain_splitter:
           - dataset: domain_splitter.split(aliases)
           - subject/session: domain_splitter.split(data, alias)
        2. 对每个 domain_fold:
           a. 源域划分：对每个源域数据集独立调用 source_splitter.split(data_5d)，
              各数据集在自身维度空间内分组切出 train/val
           b. 目标域划分:
              - inductive: target_splitter.split(data_5d) → target train/val/test
              - transductive: target_train = target_test = 全域数据（各自 copy），target_val = 空
           c. 组合为 UDASplitResult（含 fold_info）
        3. 处理嵌套折叠（inductive + target k-fold 时展平）
        
        Returns:
            展平后的 UDASplitResult 列表
        """
        ...
```

### 3.4.6 工厂函数

```python
# ---- experiment/splitter/__init__.py ----

def create_splitter(config: dict, seed: int = 42) -> HoldoutSplitter | KFoldSplitter:
    """根据配置创建索引级 regular splitter（dimension ≠ dataset）。
    
    自动根据 strategy 选择:
    - strategy=holdout → HoldoutSplitter（传入 val_split_config 若存在）
    - strategy=k-fold → KFoldSplitter（传入 val_split_config 若存在）
    
    若配置中包含 val_split 子块，提取为 val_split_config 传入 splitter。
    seed 透传给 splitter 构造函数。
    
    注意：dimension=dataset 时不经过此工厂，由 RegularExecutionStrategy
    直接构造 DatasetLevelSplitter 并走独立的跨数据集执行路径。
    """
    ...

def create_uda_orchestrator(uda_config: dict, seed: int = 42) -> UDAOrchestrator:
    """根据 UDA 配置创建编排器。seed 透传给 UDAOrchestrator。"""
    ...
```

### 3.4.7 通道数据准备与 DataloaderBuilder

#### 通道数据准备函数

SplitResult / UDASplitResult 仅携带索引，需要经过"索引 → 切片 → 按通道组织 dict"的转换才能传入 DataloaderBuilder。以下两个函数承担此职责：

```python
# ---- experiment/dataloader_builder.py ----

def prepare_channel_data(
    split_result: SplitResult | MultiDatasetSplitResult,
    dataset_cache: dict[str, np.ndarray],
    phase: str,  # "train" | "val" | "test"
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """将 Regular 模式划分结果的索引转换为通道数据 dict。

    单数据集（SplitResult）：
        indices = getattr(split_result, f"{phase}_indices")
        data = flatten_3d(dataset_cache[alias])[indices]
        return {"main": data}, {"main": labels}

    多数据集 dimension=dataset（MultiDatasetSplitResult）：
        alias_indices = split_result.phase_indices[phase]  # {alias → indices}
        arrays = [flatten_3d(dataset_cache[a])[idx] for a, idx in alias_indices.items()]
        merged = np.concatenate(arrays, axis=0)
        return {"main": merged}, {"main": merged_labels}
        
        若 phase 对应空 dict（如无验证集时的 "val"），返回空数组。

    Returns:
        (channel_data, channel_labels):
            channel_data:   {"main": np.ndarray}  shape [N, C, T]
            channel_labels: {"main": np.ndarray}  shape [N]
    """
    ...


def prepare_uda_channel_data(
    uda_split: UDASplitResult,
    dataset_cache: dict[str, np.ndarray],
) -> dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]:
    """将 UDASplitResult 转换为 UDA 模式各 dataloader 的通道数据。

    流程：
    1. 遍历 uda_split 的五组索引（source_train/source_val/target_train/target_val/target_test）
    2. 对每组：按 alias 从 dataset_cache 取数据 → flatten_3d → 按索引切片
    3. 多 alias 时沿 axis=0 concat
    4. 按通道名组织为 dict

    Returns:
        {
            "train": ({"source": src_data, "target": tgt_data},
                      {"source": src_labels, "target": tgt_labels}),
            "val":   ({"source_val": src_val_data, "target_val": tgt_val_data},
                      {"source_val": src_val_labels, "target_val": tgt_val_labels}),
            "test":  ({"main": tgt_test_data},
                      {"main": tgt_test_labels}),
        }
    """
    ...
```

> **flatten_3d**：将 5D 数据 `[n_subjects, n_sessions, n_recordings, n_channels, n_samples]` 的前三维展平为 `[N, n_channels, n_samples]`。这是所有索引切片操作的前置步骤。

#### DataloaderBuilder

```python
# ---- experiment/dataloader_builder.py ----

class DataloaderBuilder:
    """根据通道-数据集映射构建 DataLoader。
    
    调用者提供已切分好的 dict[str, np.ndarray]（通道名 → 数据），
    Builder 负责封装为 EEGDataset 并构建 DataLoader。
    """
    
    def build(
        self,
        channel_data: dict[str, np.ndarray],
        channel_labels: dict[str, np.ndarray],
        batch_size: int,
        shuffle: bool = True,
    ) -> DataLoader:
        """
        Args:
            channel_data: {"main": array} 或 {"source": array, "target": array}
            channel_labels: 各通道对应的标签（通道名与 channel_data 一致）
            batch_size: 批大小
            shuffle: 是否打乱
        
        Returns:
            DataLoader，每个 batch 为 dict[str, Tensor]
        """
        ...

def _get_sample_input(train_loader, mode: str = "regular") -> torch.Tensor:
    """从 train_loader 中取第一个 batch，提取模型输入 tensor。
    
    Regular 模式：batch["main"][0]（取 data 部分）
    UDA 模式：batch["source"][0]（取 source data 部分）
    
    返回的 tensor 用于 log_graph 的 model trace。
    """
    ...
```

### 3.4.8 Transform 应用

```python
# ---- experiment/transforms.py ----

def apply_transforms_per_dataset(
    transforms_config: list[dict],
    dataset_cache: dict[str, np.ndarray],
    split_indices: dict[str, dict[str, np.ndarray]],
    fit_phase: str = "train",
) -> None:
    """对每个数据集独立 fit & transform（原地修改 dataset_cache）。
    
    用于 Regular 模式 scope=per_dataset 和所有 UDA 场景。
    
    Args:
        transforms_config: 变换配置列表，如 [{"name": "zscore_normalize"}]
        dataset_cache: alias → 5D 数据（原地修改，保持 5D 形状）
        split_indices: alias → {phase → indices}
            indices 为 flatten_3d 后的样本级索引。
            e.g. {"ds_a": {"train": array([...]), "val": array([...]), "test": array([...])}}
        fit_phase: fit 时使用的 phase 名（默认 "train"）
    
    流程：
    1. 对每个 alias 独立创建 transform 实例
    2. flatten_3d(dataset_cache[alias]) → 3D 数据（该 alias 的所有样本）
    3. 用 split_indices[alias][fit_phase] 索引取出 fit 数据 → transform.fit()
    4. 对该 alias 的所有样本执行 transform.transform()
       （alias 内的样本天然只属于该数据集参与的 phase——
        训练数据集：train + val；测试数据集：test。因此"全量 transform"
        等价于"transform 该数据集参与的所有 phase"，不会触及不属于该数据集的数据）
    5. reshape 回 5D 写回 dataset_cache[alias]
    
    特殊情况（dimension=dataset 的测试数据集）：
    - 若 split_indices[alias] 中无 fit_phase 键（测试数据集无训练部分），
      则 fit on 该数据集全量数据。调用者在构造 split_indices 时需处理此逻辑。
    """
    ...

def apply_transforms_global(
    transforms_config: list[dict],
    dataset_cache: dict[str, np.ndarray],
    split_indices: dict[str, dict[str, np.ndarray]],
    fit_phase: str = "train",
) -> None:
    """合并所有数据集的训练数据 fit，统一 transform（原地修改 dataset_cache）。
    
    仅用于 Regular 模式 scope=global。UDA 模式禁止使用（校验规则 R23）。
    
    Args:
        transforms_config: 变换配置列表
        dataset_cache: alias → 5D 数据（原地修改，保持 5D 形状）
        split_indices: alias → {phase → indices}（indices 为 flatten_3d 后的样本级索引）
        fit_phase: fit 时使用的 phase 名（默认 "train"）
    
    流程：
    1. 创建单个 transform 实例
    2. 对每个 alias: flatten_3d → 用 fit_phase 索引取出训练数据
    3. 合并所有 alias 的训练数据 → transform.fit(merged_train)
    4. 对每个 alias: flatten_3d → transform.transform() → reshape 回 5D 写回
    """
    ...

def apply_transforms_uda(
    transforms_config: list[dict],
    dataset_cache: dict[str, np.ndarray],
    uda_split: "UDASplitResult",
    adaptation: str,
) -> None:
    """UDA 模式的 transform 应用（原地修改 dataset_cache）。
    
    始终使用 per_dataset 语义，但 fit 数据的选择因域角色而异：
    
    Args:
        transforms_config: 变换配置列表
        dataset_cache: alias → 5D 数据（原地修改，保持 5D 形状）
        uda_split: 当前 fold 的 UDA 划分结果
        adaptation: "transductive" | "inductive"
    
    流程：
    1. 源域各数据集：fit on source_train[alias]，transform source_train/source_val
    2. 目标域各数据集：
       - inductive: fit on target_train[alias]，transform target_train/target_val/target_test
       - transductive: fit on 目标域全量数据，transform 同一份数据
    
    内部委托 apply_transforms_per_dataset()，通过构造不同的 split_indices
    和 fit_phase 参数来实现源域/目标域的差异化行为。
    """
    ...
```

### 3.4.9 训练日志

```python
# ---- experiment/logger.py ----

@runtime_checkable
class TrainingLogger(Protocol):
    """训练日志写入协议。Runner 仅依赖此协议。
    
    支持 context manager 模式，确保异常时也能安全关闭。
    """
    
    def log_scalars(self, tag_value_dict: dict[str, float], step: int) -> None:
        """记录一组标量指标。"""
        ...
    
    def log_graph(self, model: torch.nn.Module, input_sample: torch.Tensor) -> None:
        """记录模型计算图（可选）。"""
        ...
    
    def close(self) -> None:
        """释放资源。"""
        ...
    
    def __enter__(self) -> "TrainingLogger":
        """进入 context manager，返回自身。"""
        ...
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """退出 context manager，调用 close()。"""
        ...

class TensorBoardLogger:
    """基于 torch.utils.tensorboard 的实现。延迟 import SummaryWriter。"""
    
    def __init__(self, log_dir: Path) -> None: ...
    def log_scalars(self, tag_value_dict: dict[str, float], step: int) -> None: ...
    def log_graph(self, model: torch.nn.Module, input_sample: torch.Tensor) -> None: ...
    def close(self) -> None: ...

def create_logger(config: dict, log_dir: Path) -> TrainingLogger | None:
    """根据 training.logging 配置创建 logger，未配置时返回 None。"""
    ...
```
