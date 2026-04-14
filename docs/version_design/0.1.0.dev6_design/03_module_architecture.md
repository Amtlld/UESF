# 三、模块架构设计

## 3.1 目录结构

```
src/uesf/experiment/
├── __init__.py
├── config_schema.py          # YAML 配置校验与规范化
├── splitter/                 # 拆为子包
│   ├── __init__.py           # 导出 create_splitter, SplitResult, UDASplitResult 等
│   ├── base.py               # SplitResult, DomainSplitResult, UDASplitResult
│   ├── regular.py            # HoldoutSplitter, KFoldSplitter, DatasetLevelSplitter
│   ├── uda.py                # DatasetDomainSplitter, DimensionDomainSplitter,
│   │                         # ValSplitter, UDAOrchestrator
│   └── grouping.py           # get_groups() 维度分组工具函数
├── alignment.py              # ChannelAligner, LabelAligner（保持现有接口）
├── transforms.py             # ZScoreNormalize 等（保持现有）
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

@dataclass
class FoldResult:
    """单个 fold 的评估结果。"""
    metrics: dict[str, float]             # {"accuracy": 0.85, "f1_score": 0.82}
    fold_info: dict                       # 来自 SplitResult/UDASplitResult 的 fold 描述
    predictions: np.ndarray | None        # concat 模式需要保留预测结果
    labels: np.ndarray | None             # concat 模式需要保留真实标签

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
    """Regular 模式执行策略。"""
    
    def run(self, ctx: ExperimentContext) -> list[FoldResult]:
        """创建 splitter → 遍历 fold → 训练评估 → 返回各 fold 结果。"""
        ...

class UDAExecutionStrategy:
    """UDA 模式执行策略。"""
    
    def run(self, ctx: ExperimentContext) -> list[FoldResult]:
        """创建 UDAOrchestrator → 遍历展平后的 fold → 训练评估 → 返回各 fold 结果。"""
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
        - shuffle 默认 true
        - 旧键名转换（k-folds / k_folds → k）
        - k-fold val_ratio → val_ratio_in_train 换算
        - source.split 无 strategy 时标记为 ValSplit 类型
        - transductive target.split 无 strategy 时标记为 ValSplit 类型
        """
        ...

    @staticmethod
    def validate(config: dict) -> None:
        """校验规范化后的配置，不合法则抛出 ConfigError。
        
        校验规则见配置校验规则文档。
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
    """数据集级划分结果。"""
    phase_aliases: dict[str, list[str]]  # {"train": ["ds_a"], "test": ["ds_b"]}

@dataclass
class DomainSplitResult:
    """UDA 域划分结果 — 一个 domain fold。
    
    对于跨数据集 UDA (dimension=dataset):
        source_indices/target_indices: {alias: 全量索引}
    对于数据集内 UDA (dimension=subject/session):
        source_indices/target_indices: {alias: 对应维度的索引}
    """
    source_indices: dict[str, np.ndarray]  # alias → sample indices
    target_indices: dict[str, np.ndarray]  # alias → sample indices
    fold_info: dict                        # 描述信息（如哪个 subject 作为 target）

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

> **transductive 模式 copy 语义**：transductive 模式下 `target_test` 是 `target_train` 的 **copy**（`target_train.copy()`），而非同一引用。这确保下游对其中一个的修改不会影响另一个。

### 3.4.3 维度分组工具

```python
# ---- experiment/splitter/grouping.py ----

def get_groups(data: np.ndarray, dimension: str) -> list[np.ndarray]:
    """将 5-D 数据按指定维度分组，返回每组的扁平化索引列表。
    
    data shape: [n_subjects, n_sessions, n_recordings, n_channels, n_samples]
    
    dimension="subject"   → 每个 subject 一组（n_subjects 组）
    dimension="session"   → 每个 (subject, session) 一组（n_subjects × n_sessions 组）
    dimension="recording" → 每个 (subject, session, recording) 一组
    
    Returns:
        groups: list[np.ndarray], 每个元素是该组在扁平化后的 sample indices
    """
    ...
```

### 3.4.4 Regular Splitters

```python
# ---- experiment/splitter/regular.py ----

class HoldoutSplitter:
    """Holdout 划分：按 ratio 切分为 train/val/test。"""
    
    def __init__(self, dimension: str, train_ratio: float, val_ratio: float,
                 test_ratio: float, shuffle: bool = False, seed: int = 42):
        ...
    
    def split(self, data: np.ndarray) -> list[SplitResult]:
        """返回长度为 1 的列表。"""
        ...

class KFoldSplitter:
    """K-Fold 划分：K 折交叉验证。"""
    
    def __init__(self, dimension: str, k: int, val_ratio: float = 0.0,
                 shuffle: bool = False, seed: int = 42):
        """
        Args:
            val_ratio: 从整体数据中切出验证集的比例。
                       内部自动换算为 val_ratio_in_train = val_ratio / (1 - 1/k)。
        """
        ...
    
    def split(self, data: np.ndarray) -> list[SplitResult]:
        """返回长度为 k 的列表。"""
        ...

class DatasetLevelSplitter:
    """数据集级划分：整个数据集分配到 train/val/test。
    
    不继承 BaseSplitter — 接口签名不同（接收 alias 列表而非 np.ndarray）。
    """
    
    def __init__(self, strategy: str, assign: dict | None = None,
                 k: int | None = None, seed: int = 42):
        ...
    
    def split(self, aliases: list[str]) -> list[DatasetLevelSplitResult]:
        """
        strategy=holdout + assign: 返回长度为 1 的列表
        strategy=k-fold: 返回长度为 k 的列表（各数据集轮流做测试集）
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
    
    用于 source.split 和 transductive target.split。
    按 dimension 分组后，将 val_ratio 比例的组整体划入验证集。
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
    
    def __init__(self, uda_config: dict):
        # 根据 domain.dimension 创建合适的 domain splitter
        domain_cfg = uda_config["domain"]
        if domain_cfg["dimension"] == "dataset":
            self.domain_splitter = DatasetDomainSplitter(...)
        else:
            self.domain_splitter = DimensionDomainSplitter(...)
        
        # 源域：ValSplitter
        self.source_splitter = ValSplitter(...)
        
        # 目标域：根据 adaptation 选择
        self.adaptation = uda_config["adaptation"]
        target_split_cfg = uda_config.get("target", {}).get("split", {})
        if self.adaptation == "inductive":
            # 完整 Split Block → HoldoutSplitter 或 KFoldSplitter
            self.target_splitter = create_splitter(target_split_cfg)
        elif target_split_cfg:
            # transductive + 有 ValSplit 配置
            self.target_splitter = ValSplitter(...)
        else:
            self.target_splitter = None  # transductive 无 target.split
    
    def split(self, dataset_cache: dict) -> list[UDASplitResult]:
        """
        完整流程：
        1. 根据 domain.dimension 调用对应的 domain_splitter:
           - dataset: domain_splitter.split(aliases)
           - subject/session: domain_splitter.split(data, alias)
        2. 对每个 domain_fold:
           a. 收集源域数据 → source_splitter.split() → source train/val
           b. 收集目标域数据:
              - inductive: target_splitter.split() → target train/val/test
              - transductive + ValSplit: target_splitter.split() → target train/val,
                target_test = target_train.copy()
              - transductive 无 split: target_train = target_test = 全域数据（copy）
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

def create_splitter(config: dict) -> HoldoutSplitter | KFoldSplitter | DatasetLevelSplitter:
    """根据配置创建 regular splitter。
    
    自动根据 strategy 和 dimension 选择:
    - dimension=dataset → DatasetLevelSplitter
    - strategy=holdout → HoldoutSplitter
    - strategy=k-fold → KFoldSplitter
    """
    ...

def create_uda_orchestrator(uda_config: dict) -> UDAOrchestrator:
    """根据 UDA 配置创建编排器。"""
    ...
```

### 3.4.7 DataloaderBuilder

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
```

### 3.4.8 训练日志

```python
# ---- experiment/logger.py ----

@runtime_checkable
class TrainingLogger(Protocol):
    """训练日志写入协议。Runner 仅依赖此协议。"""
    
    def log_scalars(self, tag_value_dict: dict[str, float], step: int) -> None:
        """记录一组标量指标。"""
        ...
    
    def log_graph(self, model: torch.nn.Module, input_sample: torch.Tensor) -> None:
        """记录模型计算图（可选）。"""
        ...
    
    def close(self) -> None:
        """释放资源。"""
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
