# 如何编写自定义训练器

本指南说明如何编写符合 UESF 规范的自定义训练器。

---

## 设计理念：训练委托

UESF 中的 Runner（运行器）是极度精简的调度层，它**不**参与任何梯度计算。Runner 只负责：

1. 驱动训练循环（for epoch in range(epochs)）
2. 将每个 batch 字典和 optimizer 实例传递给 Trainer
3. 收集 Trainer 返回的指标数据

**梯度的完整生命周期**（`zero_grad` → `forward` → `loss.backward` → `step`）**全部由 Trainer 负责**。这个设计让你可以在 Trainer 中实现任何复杂的训练策略：GAN 的交替更新、UDA 的多阶段梯度、多优化器协作等，而不受框架的限制。

---

## BaseTrainer 接口

```python
class BaseTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        **kwargs,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.config = kwargs    # 实验 YAML 中 trainer.params 的内容

    def configure_optimizers(self) -> Optional[tuple]: ...  # 可选

    @abstractmethod
    def training_step(
        self,
        batch: dict[str, tuple[torch.Tensor, torch.Tensor]],
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def validation_step(
        self,
        batch: dict[str, tuple[torch.Tensor, torch.Tensor]],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]: ...
```

---

## batch 字典的结构

`training_step` 和 `validation_step` 接收的 `batch` 是一个字典：

```python
{
    "main": (data_tensor, labels_tensor),
    # 多数据源时可能有多个键
    # "source": (src_data, src_labels),
    # "target": (tgt_data, tgt_labels),
}
```

其中：
- 键名对应实验 YAML 的 `dataloaders` 中定义的通道名
- `data_tensor` 形状：`(batch_size, n_channels, n_samples)`
- `labels_tensor` 形状：`(batch_size,)`，类型为 `torch.long`

---

## training_step：5 步优化闭环

`training_step` 必须完成完整的一步优化：

```python
def training_step(self, batch, batch_idx, optimizer):
    data, labels = batch["main"]
    data = data.to(self.device)
    labels = labels.to(self.device)

    # 1. 前向传播
    logits = self.model(data)
    # 2. 计算损失
    loss = F.cross_entropy(logits, labels)
    # 3. 清零梯度
    optimizer.zero_grad()
    # 4. 反向传播
    loss.backward()
    # 5. 更新参数
    optimizer.step()

    return {"loss": loss.item()}   # 返回标量值（不是 tensor）
```

**返回值要求**：`dict`，值为 Python 标量（float/int）或已 detach 的标量 tensor。返回的字典会被框架聚合并写入日志。

---

## validation_step：返回预测和目标

`validation_step` 不需要计算梯度，返回的预测和目标由框架统一聚合后计算指标：

```python
def validation_step(self, batch, batch_idx):
    data, labels = batch["main"]
    data = data.to(self.device)

    with torch.no_grad():
        logits = self.model(data)
        preds = logits.argmax(dim=1)

    return {
        "preds": preds.cpu(),     # 必须包含 "preds"
        "targets": labels.cpu(), # 必须包含 "targets"
    }
```

**返回值要求**：必须包含 `"preds"` 和 `"targets"` 两个键，对应的 tensor 会被框架在整个验证集上拼接（concat），然后传入各指标函数计算最终结果。

---

## 完整示例：标准分类训练器

```python
# src/trainers/trainer.py
from typing import Any

import torch
import torch.nn.functional as F

from uesf.components.base_trainer import BaseTrainer


class EmotionTrainer(BaseTrainer):
    """标准交叉熵分类训练器。"""

    def training_step(
        self,
        batch: dict[str, tuple[torch.Tensor, torch.Tensor]],
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, Any]:
        data, labels = batch["main"]
        data = data.to(self.device)
        labels = labels.to(self.device)

        # 5 步优化闭环
        logits = self.model(data)                    # 1. 前向传播
        loss = F.cross_entropy(logits, labels)       # 2. 计算损失
        optimizer.zero_grad()                         # 3. 清零梯度
        loss.backward()                               # 4. 反向传播
        optimizer.step()                              # 5. 更新参数

        return {"loss": loss.item()}

    def validation_step(
        self,
        batch: dict[str, tuple[torch.Tensor, torch.Tensor]],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        data, labels = batch["main"]
        data = data.to(self.device)

        with torch.no_grad():
            logits = self.model(data)
            preds = logits.argmax(dim=1)

        return {
            "preds": preds.cpu(),
            "targets": labels.cpu(),
        }
```

---

## 可选：configure_optimizers

默认情况下，框架根据实验 YAML 的 `training.optimizer` 和 `training.scheduler` 创建优化器和调度器。若需要自定义（例如对不同层组使用不同学习率），可以覆盖此方法：

```python
def configure_optimizers(self):
    optimizer = torch.optim.Adam([
        {"params": self.model.conv1.parameters(), "lr": 1e-4},
        {"params": self.model.fc.parameters(), "lr": 1e-3},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    return optimizer, scheduler
```

> 若 `configure_optimizers` 返回非 `None` 值，实验 YAML 中的 `training.optimizer` 和 `training.scheduler` 字段将被忽略，框架会在日志中发出警告。

返回 `None`（默认行为）使用 YAML 配置的优化器。

---

## 在 project.yml 中注册

```yaml
trainers:
  emotion_trainer:                                          # 在实验 YAML 中使用的名称
    entrypoint: "./src/trainers/trainer.py:EmotionTrainer"
```

---

## 下一步

- 配置实验 YAML，将模型和训练器挂载到实验中：[配置实验 YAML](06_configure_experiment.md)
- 运行端到端的完整实验教程：[端到端完整实验](../tutorials/02_first_experiment.md)
