# 九、训练过程监测（TensorBoard 集成）

## 9.1 设计目标

为实验训练过程提供可视化监测能力，首选后端为 TensorBoard。设计遵循以下原则：

- **Runner 零硬依赖**：Runner 通过 Protocol 接口写日志，不直接 import TensorBoard
- **可扩展后端**：Protocol 抽象允许未来接入 W&B 等其他后端，无需改动 Runner
- **零配置即可用**：用户不指定 `logging` 块时无任何开销；指定后框架自动管理 logdir 生命周期
- **多 fold 隔离**：每个 fold 独立 logdir，避免 step 冲突，支持 TensorBoard regex 过滤

## 9.2 YAML 配置接口

在 `training` 块中新增可选的 `logging` 子块：

```yaml
training:
  epochs: 100
  batch_size: 64
  logging:                          # 可选，省略则不记录训练日志
    backend: tensorboard            # 日志后端，目前仅支持 tensorboard
    log_every_n_epochs: 1           # 每 N 个 epoch 记录一次，默认 1
    log_graph: false                # 是否记录模型计算图，默认 false
```

**设计说明**：

- `backend` 目前仅支持 `tensorboard`，为未来扩展预留字段
- `log_every_n_epochs` 控制写入频率，避免大量 epoch 时日志膨胀
- `log_graph` 默认关闭——计算图写入需要 trace 一次前向传播，可能与某些动态模型不兼容
- 不暴露 `logdir`：由框架根据 `output_dir` 自动生成，保证目录结构一致性

## 9.3 TrainingLogger 协议与实现

### 9.3.1 协议定义

```python
# ---- experiment/logger.py ----

from typing import Protocol, runtime_checkable

@runtime_checkable
class TrainingLogger(Protocol):
    """训练日志写入协议。

    Runner 仅依赖此协议，不依赖具体实现。
    """

    def log_scalars(self, tag_value_dict: dict[str, float], step: int) -> None:
        """记录一组标量指标。

        Args:
            tag_value_dict: 指标名到值的映射，如 {"loss": 0.35, "accuracy": 0.82}。
            step: 当前步数（通常为 epoch 编号）。
        """
        ...

    def log_graph(self, model: "torch.nn.Module", input_sample: "torch.Tensor") -> None:
        """记录模型计算图（可选实现）。

        Args:
            model: 要记录的模型。
            input_sample: 用于 trace 的样本输入。
        """
        ...

    def close(self) -> None:
        """释放资源（关闭文件句柄、flush 缓冲区等）。"""
        ...
```

### 9.3.2 TensorBoard 实现

```python
# ---- experiment/logger.py ----

from pathlib import Path
import torch

class TensorBoardLogger:
    """基于 torch.utils.tensorboard 的 TrainingLogger 实现。"""

    def __init__(self, log_dir: Path) -> None:
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def log_scalars(self, tag_value_dict: dict[str, float], step: int) -> None:
        for tag, value in tag_value_dict.items():
            self.writer.add_scalar(tag, value, step)

    def log_graph(self, model: torch.nn.Module, input_sample: torch.Tensor) -> None:
        self.writer.add_graph(model, input_sample)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
```

**延迟 import 策略**：`SummaryWriter` 在 `__init__` 中才 import，未配置 logging 时 `tensorboard` 包不会被加载，不影响无 TensorBoard 环境下的框架运行。

### 9.3.3 工厂函数

```python
# ---- experiment/logger.py ----

def create_logger(config: dict, log_dir: Path) -> TrainingLogger | None:
    """根据配置创建 logger 实例。

    Args:
        config: training.logging 配置块（可能为空 dict）。
        log_dir: 日志输出目录。

    Returns:
        TrainingLogger 实例，或 None（未配置 logging 时）。
    """
    backend = config.get("backend")
    if backend is None:
        return None
    if backend == "tensorboard":
        return TensorBoardLogger(log_dir)
    raise ConfigError(
        f"Unsupported logging backend: {backend}",
        hint="Currently only 'tensorboard' is supported.",
    )
```

## 9.4 Runner 集成方式

Runner 的 `run()` 方法新增可选参数 `logger: TrainingLogger | None = None`，在训练循环中按条件写入：

```python
# ---- experiment/runner.py（变更部分） ----

class Runner:

    def run(
        self,
        train_loader,
        val_loader,
        optimizer,
        scheduler=None,
        checkpoint_dir=None,
        checkpoint_metric=None,
        early_stopping_config=None,
        logger=None,               # 新增：可选 TrainingLogger
        log_every_n_epochs=1,      # 新增：写入频率
    ) -> dict[str, Any]:

        # ... 现有初始化逻辑 ...

        for epoch in range(self.epochs):
            train_metrics = self.train_epoch(train_loader, optimizer, epoch)

            val_metrics = {}
            if val_loader and len(val_loader) > 0:
                val_metrics, _, _ = self.validate_epoch(val_loader)
                val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}

            # --- 日志记录（新增） ---
            if logger and (epoch + 1) % log_every_n_epochs == 0:
                scalars = {**train_metrics, **val_metrics}
                # 记录学习率
                scalars["lr"] = optimizer.param_groups[0]["lr"]
                logger.log_scalars(scalars, step=epoch)

            # ... 现有 scheduler / checkpoint / early_stopping 逻辑不变 ...

        # --- 关闭 logger（新增） ---
        if logger:
            logger.close()

        return { ... }
```

**改动范围**：仅 `run()` 签名增加两个可选参数 + 循环体内增加一个 `if` 块 + 循环结束后 `close()`。其余方法（`train_epoch`、`validate_epoch`）完全不变。

## 9.5 记录的指标

Runner 自动记录以下指标，无需用户手动配置：

| 类别 | 指标 | 来源 | TensorBoard tag |
|:-----|:-----|:-----|:----------------|
| 训练 | `training_step` 返回的所有标量 | `train_epoch()` 每 epoch 均值 | `loss`, `domain_loss` 等（原始 key） |
| 验证 | Evaluator 计算的所有指标 | `validate_epoch()` | `val_accuracy`, `val_f1_score` 等 |
| 学习率 | 当前学习率 | `optimizer.param_groups[0]["lr"]` | `lr` |

**UDA 模式说明**：UDA Trainer 的 `training_step` 通常返回 `{"loss": ..., "source_cls_loss": ..., "domain_loss": ...}` 等多个标量。Runner 对这些标量一视同仁地记录，无需为 UDA 做特殊处理——指标的丰富度由 Trainer 实现决定，Logger 层透明传递。

## 9.6 多 fold 日志目录结构

日志目录与 checkpoint 目录对齐，由 ExperimentExecutor 在创建每个 fold 的运行环境时自动生成：

```
{output_dir}/
├── fold_0/
│   ├── best.pt               # checkpoint
│   └── tb_logs/              # TensorBoard logdir
│       └── events.out.tfevents.*
├── fold_1/
│   ├── best.pt
│   └── tb_logs/
├── ...
```

**使用方式**：

```bash
# 查看某个实验所有 fold 的训练曲线（TensorBoard 自动识别子目录）
tensorboard --logdir {output_dir}

# 查看特定 fold
tensorboard --logdir {output_dir}/fold_0/tb_logs
```

**嵌套折叠（UDA inductive + target k-fold）**：展平后的 fold 编号已由 `UDAOrchestrator` 处理，目录为 `fold_0/`, `fold_1/`, ... `fold_{n-1}/`，`fold_info` 中的 `domain_fold` 和 `inner_fold` 字段可用于在 TensorBoard 中通过 tag 过滤追溯层次关系。

## 9.7 ExperimentExecutor 集成

ExperimentExecutor 负责在每个 fold 执行前创建 logger、执行后确保 logger 关闭：

```python
# ---- managers/experiment_manager.py（ExperimentExecutor 变更部分） ----

class RegularExecutionStrategy:

    def run(self, ctx: ExperimentContext) -> list[FoldResult]:
        logging_config = ctx.config.get("training", {}).get("logging", {})
        log_every_n = logging_config.get("log_every_n_epochs", 1)

        for fold_idx, split_result in enumerate(folds):
            fold_dir = ctx.output_dir / f"fold_{fold_idx}"

            # 创建 logger
            logger = create_logger(logging_config, fold_dir / "tb_logs")

            # 可选：记录模型计算图
            if logger and logging_config.get("log_graph", False):
                sample_input = _get_sample_input(train_loader)
                logger.log_graph(model, sample_input)

            # 训练
            result = runner.run(
                train_loader, val_loader, optimizer,
                logger=logger,
                log_every_n_epochs=log_every_n,
                ...
            )

            # logger.close() 由 runner.run() 内部调用
```

`UDAExecutionStrategy` 同理，无需额外处理。

## 9.8 配置校验规则

新增以下校验规则，补充至 [配置校验规则文档](07_config_validation_rules.md)：

| 编号 | 规则 | 异常类型 |
|:-----|:-----|:---------|
| R13 | `training.logging.backend` 仅允许 `"tensorboard"`（后续版本可扩展） | `TypeMismatchError` |
| R14 | `training.logging.log_every_n_epochs` 必须为正整数 | `TypeMismatchError` |

## 9.9 依赖管理

`tensorboard` 作为**可选依赖**（optional dependency），不加入核心依赖：

```toml
# pyproject.toml
[project.optional-dependencies]
tensorboard = ["tensorboard>=2.14"]
```

安装方式：

```bash
uv pip install uesf[tensorboard]
```

**缺失依赖时的行为**：用户配置了 `logging.backend: tensorboard` 但未安装 `tensorboard` 包时，`TensorBoardLogger.__init__` 中的 `from torch.utils.tensorboard import SummaryWriter` 会抛出 `ImportError`。框架在 `create_logger()` 中捕获此异常并转换为带 `hint` 的 `ConfigError`：

```python
def create_logger(config: dict, log_dir: Path) -> TrainingLogger | None:
    ...
    if backend == "tensorboard":
        try:
            return TensorBoardLogger(log_dir)
        except ImportError:
            raise ConfigError(
                "TensorBoard is not installed.",
                hint="Install with: uv pip install uesf[tensorboard]",
            )
```
