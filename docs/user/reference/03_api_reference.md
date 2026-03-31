# Python API 参考

本文档描述用户需要继承或实现的 Python 接口规范。

---

## BaseModel

`uesf.components.base_model.BaseModel`

所有自定义模型的基类，继承自 `torch.nn.Module`。

### 类签名

```python
class BaseModel(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int,
        electrode_list: Optional[list[str]] = None,
        **kwargs,
    ) -> None
```

### 构造函数参数

| 参数 | 类型 | 注入方式 | 说明 |
|------|------|----------|------|
| `n_channels` | int | 框架自动注入 | EEG 通道数，从预处理数据集元信息读取 |
| `n_samples` | int | 框架自动注入 | 每个 Epoch 的采样点数 |
| `n_classes` | int | 框架自动注入 | 类别数（从 `numeric_to_semantic` 映射条目数得出） |
| `electrode_list` | list[str] \| None | 框架自动注入 | 电极名称列表，若 `raw.yml` 未配置则为 `None` |
| `**kwargs` | - | 来自实验 YAML 的 `model.params` | 用户自定义超参数 |

> **注意** `n_channels`、`n_samples`、`n_classes` 由框架在运行时注入，无需在实验 YAML 的 `model.params` 中填写。

子类必须在 `__init__` 中调用 `super().__init__(n_channels, n_samples, n_classes, electrode_list, **kwargs)`。

### 抽象方法

#### forward

```python
@abstractmethod
def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor
```

**输入**：`x`，形状 `(batch_size, n_channels, n_samples)`，`dtype=torch.float32`

**输出**：`torch.Tensor`，形状 `(batch_size, n_classes)`，logits（未经过 softmax）

### 可选方法

#### extract_features

```python
def extract_features(self, x: torch.Tensor) -> torch.Tensor
```

用于特征提取、可视化或迁移学习场景。默认实现抛出 `NotImplementedError`，按需覆盖。

### 最小实现模板

```python
from typing import Optional
import torch
import torch.nn as nn
from uesf.components.base_model import BaseModel

class MyModel(BaseModel):
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int,
        electrode_list: Optional[list] = None,
        hidden_size: int = 128,    # 自定义超参数
        **kwargs,
    ):
        super().__init__(n_channels, n_samples, n_classes, electrode_list, **kwargs)
        # 定义网络层
        self.fc = nn.Linear(n_channels * n_samples, n_classes)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x: (batch, n_channels, n_samples)
        return self.fc(x.flatten(1))  # (batch, n_classes)
```

---

## BaseTrainer

`uesf.components.base_trainer.BaseTrainer`

所有自定义训练器的基类。

### 类签名

```python
class BaseTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        **kwargs,
    ) -> None
```

### 构造函数参数

| 参数 | 类型 | 注入方式 | 说明 |
|------|------|----------|------|
| `model` | `nn.Module` | 框架自动注入 | 已实例化的模型，已 `.to(device)` |
| `device` | `torch.device` | 框架自动注入 | 计算设备（由全局配置的 `default_device` 确定） |
| `**kwargs` | - | 来自实验 YAML 的 `trainer.params` | 用户自定义训练超参数 |

构造函数自动执行 `self.model = model.to(device)`，无需手动移动。

框架注入的属性：
- `self.model`：模型实例
- `self.device`：计算设备
- `self.config`：`kwargs` 字典（`trainer.params` 中的所有字段）

### 抽象方法

#### training_step

```python
@abstractmethod
def training_step(
    self,
    batch: dict[str, tuple[torch.Tensor, torch.Tensor]],
    batch_idx: int,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]
```

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `batch` | dict | 多通道批次数据，`{channel_name: (data, labels)}`；`data` 形状 `(B, C, T)`，`labels` 形状 `(B,)` |
| `batch_idx` | int | 当前批次索引 |
| `optimizer` | `torch.optim.Optimizer` | 框架传入的优化器实例 |

**要求**：必须完成完整的 5 步优化闭环（`zero_grad` → `forward` → `loss` → `backward` → `step`）。

**返回**：`dict`，值为 Python 标量（float/int）或已 `.detach()` 的标量 tensor。这些值会被框架聚合并写入日志。

#### validation_step

```python
@abstractmethod
def validation_step(
    self,
    batch: dict[str, tuple[torch.Tensor, torch.Tensor]],
    batch_idx: int,
) -> dict[str, torch.Tensor]
```

**参数**：同 `training_step`（无 `optimizer`）。

**返回**：`dict`，必须包含：
- `"preds"`：预测类别张量，形状 `(B,)`，`dtype=torch.long`
- `"targets"`：真实标签张量，形状 `(B,)`，`dtype=torch.long`

框架在整个验证集上将所有批次的 `preds` 和 `targets` 拼接后，统一传入指标函数计算最终结果。

### 可选方法

#### configure_optimizers

```python
def configure_optimizers(self) -> Optional[tuple[torch.optim.Optimizer, Any]]
```

覆盖此方法以自定义优化器（例如分层学习率、多优化器）。

**返回**：`None`（使用实验 YAML 配置）或 `(optimizer, scheduler)` 元组。`scheduler` 可以为 `None`。

若返回非 `None`，实验 YAML 中的 `training.optimizer` 和 `training.scheduler` 将被忽略，框架会在日志中发出警告。

### 最小实现模板

```python
from typing import Any
import torch
import torch.nn.functional as F
from uesf.components.base_trainer import BaseTrainer

class MyTrainer(BaseTrainer):
    def training_step(
        self,
        batch: dict[str, tuple[torch.Tensor, torch.Tensor]],
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, Any]:
        data, labels = batch["main"]
        data, labels = data.to(self.device), labels.to(self.device)

        logits = self.model(data)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {"loss": loss.item()}

    def validation_step(
        self,
        batch: dict[str, tuple[torch.Tensor, torch.Tensor]],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        data, labels = batch["main"]
        data = data.to(self.device)

        with torch.no_grad():
            preds = self.model(data).argmax(dim=1)

        return {"preds": preds.cpu(), "targets": labels}
```

---

## 自定义指标函数规范

指标函数不需要继承任何基类，只需遵循统一的函数签名。

### 函数签名

```python
def my_metric(
    preds: torch.Tensor,
    targets: torch.Tensor,
    **kwargs,
) -> float | dict
```

### 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `preds` | `torch.Tensor` | 整个验证/测试集的预测类别，形状 `(N,)`，已取 argmax |
| `targets` | `torch.Tensor` | 对应的真实标签，形状 `(N,)` |
| `**kwargs` | - | 额外参数（保留用于未来扩展） |

### 返回值

| 类型 | 说明 |
|------|------|
| `float` | 单值指标，直接记录 |
| `dict` | 复杂指标，值必须是 JSON 可序列化的 Python 对象（list、dict、float、int、str） |

### 示例

```python
import torch

def balanced_accuracy(preds: torch.Tensor, targets: torch.Tensor, **kwargs) -> float:
    """类别均衡准确率。"""
    classes = targets.unique()
    return sum(
        (preds[targets == c] == c).float().mean().item()
        for c in classes
    ) / len(classes)
```

### 注册到项目

```yaml
# project.yml
metrics:
  balanced_accuracy:
    entrypoint: "./src/metrics/balanced.py:balanced_accuracy"
```

### 在实验中使用

```yaml
# experiments/baseline.yml
evaluation:
  metrics: [accuracy, balanced_accuracy]
```
