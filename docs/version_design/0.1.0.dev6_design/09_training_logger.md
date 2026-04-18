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
    log_every_n_steps: null         # 每 N 个 batch 记录一次 step 级标量；null/省略 = 关闭
    log_graph: false                # 是否记录模型计算图，默认 false
    log_lr: true                    # 是否记录学习率（step 级与 epoch 级均受控），默认 true
    train_step_scalars: null        # list[str] 白名单，筛选 training_step 返回的标量；null = 记录全部
    train_metrics: []               # list[str]，在训练集 preds/targets 上用 Evaluator 计算（opt-in）
    val_metrics: null               # list[str]，必须是 evaluation.metrics 子集；null = 记录全部
    test_metrics: []                # list[str]，在测试集上计算（opt-in，触发数据泄露 warning）
```

**设计说明**：

- `backend` 目前仅支持 `tensorboard`，为未来扩展预留字段
- `log_every_n_epochs` 控制 epoch 级写入频率，避免大量 epoch 时日志膨胀
- `log_every_n_steps` 控制 step 级（batch 级）写入频率，便于观察细粒度 loss / lr 波动；未声明默认关闭
- `log_graph` 默认关闭——计算图写入需要 trace 一次前向传播，可能与某些动态模型不兼容
- 不暴露 `logdir`：由框架根据 `output_dir` 自动生成，保证目录结构一致性
- `train_metrics`、`test_metrics` 的指标名走与 `evaluation.metrics` 相同的 `MetricManager` 解析链（内置 → 注册 REGISTERED → 全局 GLOBAL），所以**自定义指标**可以直接复用
- 新字段全部可选，未声明时行为与 dev6 早期版本完全一致（向后兼容）

## 9.3 TrainingLogger 协议与实现

### 9.3.1 协议定义

```python
# ---- experiment/logger.py ----

from typing import Protocol, runtime_checkable

@runtime_checkable
class TrainingLogger(Protocol):
    """训练日志写入协议。

    Runner 仅依赖此协议，不依赖具体实现。
    支持 context manager 模式，确保异常时也能 flush 并关闭。
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

    def __enter__(self) -> "TrainingLogger":
        """进入 context manager，返回自身。"""
        ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """退出 context manager，调用 close()（异常时也能安全关闭）。"""
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

    def __enter__(self) -> "TensorBoardLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
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

        # 使用 context manager 包裹训练循环，确保异常时 logger 也能安全关闭
        logger_cm = logger if logger is not None else _NullContext()
        with logger_cm:
            for epoch in range(self.epochs):
                train_metrics = self.train_epoch(train_loader, optimizer, epoch)

                val_metrics = {}
                if val_loader and len(val_loader) > 0:
                    val_metrics, _, _ = self.validate_epoch(val_loader)
                    val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}

                # --- 日志记录（新增） ---
                if logger is not None and (epoch + 1) % log_every_n_epochs == 0:
                    scalars = {**train_metrics, **val_metrics}
                    scalars["lr"] = optimizer.param_groups[0]["lr"]
                    logger.log_scalars(scalars, step=epoch)

                # ... 现有 scheduler / checkpoint / early_stopping 逻辑不变 ...

        return { ... }
```

**改动范围**：仅 `run()` 签名增加两个可选参数 + 循环体用 `with` 包裹 + 循环体内增加一个 `if` 块。其余方法（`train_epoch`、`validate_epoch`）完全不变。Logger 的关闭由 `__exit__` 自动完成，异常时也能 flush。

## 9.5 记录的指标

Runner 默认记录下表中的指标，不声明任何新字段时与 dev6 早期行为一致：

| 类别 | 来源 | TensorBoard tag | step 坐标 | 默认启用 |
|:-----|:-----|:----------------|:---------|:---------|
| 训练 step 标量（epoch 平均） | `train_epoch()` aggregate from `training_step` | `<name>`（`loss`、`domain_loss` 等） | `epoch` | ✅ |
| 训练 computed 指标 | 训练集 preds/targets → `train_metrics` Evaluator | `train_<name>` | `epoch` | ❌（需声明 `train_metrics`） |
| 验证 computed 指标 | `validate_epoch()` → `evaluation.metrics` Evaluator | `val_<name>`（`val_accuracy` 等），经 `val_metrics` 过滤 | `epoch` | ✅ |
| 测试 computed 指标 | `test_loader` → `test_metrics` Evaluator | `test_<name>` | `epoch` | ❌（需声明 `test_metrics`，附带数据泄露 WARN） |
| 学习率 | `optimizer.param_groups[0]["lr"]` | `lr` | `epoch` | ✅（`log_lr` 可关） |
| Step 级训练标量 | `training_step` 返回的数值标量（经 `train_step_scalars` 过滤） | `step/<name>`（`step/loss` 等） | 全局 step = `epoch * len(train_loader) + batch_idx` | ❌（需声明 `log_every_n_steps`） |
| Step 级学习率 | 同上 | `step/lr` | 全局 step | ❌（需声明 `log_every_n_steps`；`log_lr` 可关） |

**UDA 模式说明**：UDA Trainer 的 `training_step` 通常返回 `{"loss": ..., "source_cls_loss": ..., "domain_loss": ...}` 等多个标量。Runner 对这些标量一视同仁地记录，无需为 UDA 做特殊处理——指标的丰富度由 Trainer 实现决定，Logger 层透明传递。使用 `train_step_scalars` 可以在 TB 中只保留感兴趣的子集。

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

            # 训练（logger 的关闭由 runner.run() 内部的 with 块自动处理）
            result = runner.run(
                train_loader, val_loader, optimizer,
                logger=logger,
                log_every_n_epochs=log_every_n,
                ...
            )
```

`UDAExecutionStrategy` 同理，无需额外处理。

## 9.8 配置校验规则

相关配置校验规则见 [07_config_validation_rules.md](07_config_validation_rules.md) 中 R17、R18。

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

## 9.10 Step 级写入

`log_every_n_steps`（默认 `null`）启用后，Runner 在 `train_epoch` 内部每 N 个 batch 向
TensorBoard 写入一条记录，step 坐标为**全局 step** = `epoch * len(train_loader) + batch_idx`。
Tag 统一加 `step/` 前缀，与 epoch 级 tag 命名空间隔离：

| 来源 | tag | step 坐标 |
|------|-----|-----------|
| `training_step` 返回的每个数值标量（经 `train_step_scalars` 过滤） | `step/<name>` | 全局 step |
| 学习率（`log_lr=true` 时） | `step/lr` | 全局 step |

**写入条件**：`training_logger is not None and log_every_n_steps and (global_step + 1) % log_every_n_steps == 0`。
触发后，Runner 从当前 `step_result` 中提取数值标量、应用 `train_step_scalars` 白名单后
写入。step 级写入独立于 `log_every_n_epochs`——即使 epoch 级写入被跳过的 epoch，step 级
仍然按 N-batch 节奏进行。

**典型用法**：训练集规模较大、loss 波动剧烈时启用 `log_every_n_steps` 可得到更细的 loss
曲线；epoch 级则用于 aggregated / val / test 指标。

## 9.11 指标白名单（filter）

Runner 默认把 `training_step` 返回的全部数值标量 ∪ `evaluation.metrics` 计算出的全部
验证指标 ∪ `lr` 一并写入 TB。dev6+ 用户可以通过三条过滤策略收敛 tag 集合：

| 字段 | 作用对象 | 默认 | 语义 |
|------|---------|------|------|
| `train_step_scalars` | `training_step` 返回的数值标量（epoch 平均 + step 级） | `null` = 全部 | 白名单：仅出现在列表中的 key 被写入 TB（tag 保持原名，不加 `train_` 前缀） |
| `val_metrics` | `evaluation.metrics` 中计算出的验证指标 | `null` = 全部 | 白名单：仅出现在列表中的**裸名**对应的 `val_<name>` 会写入 TB；**必须是 `evaluation.metrics` 的子集**（R37） |
| `log_lr` | `lr` 与 `step/lr` | `true` | 布尔开关；`false` 时完全不写 lr |

**过滤对 Evaluator 计算的影响**：`val_metrics` 只影响 TB 写入，不影响 Evaluator
计算——`evaluation.metrics` 仍然全部计算（checkpoint / early_stopping / 最终报告依赖）。

## 9.12 训练集 computed 指标（train_metrics）

声明 `training.logging.train_metrics` 后，Runner 在每 epoch 结束时用一个独立的
Evaluator 对**训练集** preds/targets 计算指定指标，以 `train_<name>` 的 tag 写入 TB。
Runner 不做额外的 no-grad 前向，而是复用 `training_step` 的前向结果——**要求 Trainer
的 `training_step` 在返回 dict 中额外带上**：

- `"preds"`: `torch.Tensor`（logits 或类索引；与 `validation_step` 一致）
- `"targets"`: `torch.Tensor`

Runner 在 `train_epoch` 内部通过 `step_result.get("preds")` / `.get("targets")` 收集；
为避免多余的 CPU 拷贝，**仅在 `train_evaluator` 非空时才收集**（opt-in）。

**缺失字段时的降级行为**：若 `train_metrics` 已配置但 `training_step` 没有返回
`preds` / `targets`，Runner 在首个 epoch 结束时发出 WARN 并静默跳过训练集指标计算，
其他 TB 写入不受影响：

```
training.logging.train_metrics is configured but training_step did not return
'preds'/'targets' — skipping train-side metric computation. Update your Trainer
to return these tensors to enable train metrics.
```

**成本**：只是把 `training_step` 已经算出的 logits 额外 `.detach().cpu()` 一次并存到
`Runner._train_preds`。相对完整 no-grad 训练集评估 pass 省掉一次前向，开销可忽略。

## 9.13 测试集 computed 指标（test_metrics）与数据泄露风险

声明 `training.logging.test_metrics` 后，Runner 每 `log_every_n_epochs` 对
`test_loader` 做一次 no-grad 评估 pass（复用 `trainer.validation_step`），用独立的
test Evaluator 计算指定指标并以 `test_<name>` 写入 TB。

**强制性 WARN**：`ExperimentExecutor` 在读取到非空 `test_metrics` 时立即发出一条
WARN——这**不是**阻断式校验，用户可以选择继续执行，但框架要求显式记录：

```
training.logging.test_metrics is set — tracking test-set metrics DURING
training creates data-leakage risk via model selection / hyperparameter
tuning. Prefer val_metrics for monitoring; only enable test_metrics for
controlled ablation / analysis where you accept the risk.
```

**为什么警告**：训练过程中观测测试集曲线会把测试集信息泄露到模型选择 / 超参调整
环节——即便不直接用 test 指标做 early_stopping / checkpoint，研究人员也会基于曲线
隐式调优，导致最终报告的 test 指标乐观偏差。UESF 默认假设 `evaluation.metrics` 的
最终 test 评估仅在训练结束后执行一次（现状保留），`test_metrics` 只用于**事后分析
/ 消融研究 / 教学演示**等接受风险的场景。

**与最终 test 评估的关系**：训练结束后，`ExperimentExecutor` 仍然按 dev6 行为在测试集
上运行一次完整 `evaluation.metrics` 评估并写入 `fold_results` / 最终报告——
`test_metrics` 只新增训练过程中的 TB 追踪，不影响最终报告的内容。
