# Trainer Manager 详细设计

训练器（trainer）定义了训练流程。与 Model Manager 类似，UESF 支持三类训练器：
- 内置训练器（EMBEDDED）
- 自定义训练器（REGISTERED）
- 全局自定义训练器（GLOBAL）

数据库表结构详见 [`trainers` 表](../03_storage/02_database_schema.md#trainers-表)。

## 1. 内置训练器

UESF 开发者维护若干个常用的深度学习训练器，供用户使用。

使用 UESF 内置训练器时，用户需在实验配置文档中指定训练器名称，UESF 会自动加载内置训练器。
```yaml
model: "EEGConformer"
trainer: "CommonTrainer"
```

## 2. 自定义训练器

UESF 可以作为一个 Python 库被导入用户 Python 脚本中。用户可以利用 UESF 提供的训练器基类来自定义训练器。

用户通过编写继承 UESF 提供的训练器基类的训练器，并通过实验配置进行导入的，被视作用户管理的训练器。

一个可用的自定义训练器需要满足如下两个必要条件：
1. 自定义训练器源代码存在
2. 自定义训练器在项目配置文件中注册

训练器在项目配置文件 `project.yml` 中注册的示例：
```yaml
trainers:
  MyTrainer:
    entrypoint: "./src/models/my_trainer.py:MyTrainerClass"
  ...
```

### 2.1 BaseTrainer 接口规范

为彻底贯彻控制流委托，自定义训练器必定继承的基类接口规范如下：
```python
from typing import Dict, Any, Tuple, Optional
import torch
import warnings

class BaseTrainer:
    def __init__(self, model: torch.nn.Module, device: torch.device, **kwargs):
        """初始化训练器并挂载模型实例。"""
        self.model = model.to(device)
        self.device = device
        self.config = kwargs
        
    def configure_optimizers(self) -> Optional[Tuple[torch.optim.Optimizer, Any]]:
        """
        (可选) 配置特定优化器和学习率调度器。
        
        若此方法返回非 None 值，系统将使用其返回的优化器和调度器，
        即使实验 YAML 中同时定义了 training.optimizer 等字段，
        系统也会忽略 YAML 中的优化器配置，并在日志中发出 Warning 提示。
        
        若此方法返回 None（默认行为），系统将回退使用 YAML 配置中的
        training.optimizer / training.learning_rate 等字段构建优化器。
        
        :return: (optimizer, scheduler) 元组，或 None
        """
        return None 

    def training_step(
        self,
        batch: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        batch_idx: int,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, Any]:
        """
        核心训练步委托。梯度反向传播由 Trainer 全权负责。
        
        Runner 会将多通道 DataLoader 得到的数据组装为字典下发，
        同时将当前优化器实例一并传入。Trainer 必须在此方法内部
        完成以下完整流程：
          1. 前向传播计算 loss
          2. optimizer.zero_grad()
          3. loss.backward()
          4. (可选) 梯度裁剪
          5. optimizer.step()
        
        这意味着 Runner 不会在此方法之外调用 .backward() 或
        .step()，从而使得 Trainer 可以实现任意复杂的优化策略
        （如 GAN 的交替更新、UDA 的多阶段参数冻结等）。
        
        :param batch: 多通道数据字典，键为通道名，值为 (data, label) 元组
        :param batch_idx: 当前批次索引
        :param optimizer: 当前使用的优化器实例
        :return: 包含日志信息的字典（如 {"loss": loss_value, "lr": current_lr, ...}），
                 其中的值应为 Python 标量或已 .detach() 的张量，仅用于日志记录
        """
        raise NotImplementedError

    def validation_step(self, batch: Dict[str, Tuple[torch.Tensor, torch.Tensor]], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        验证步委托。
        预期直接返回 "preds" 和 "targets" 字典。Runner 会统一收集 Epoch 队列，再集中调用外部指标包计算数值结果。
        """
        raise NotImplementedError
```

## 3. 训练器管理

UESF 在数据库 `trainers` 表中记录训练器元信息。无论是项目级自定义组件还是全局组件，均需注册到数据库中。

UESF 支持用户将自定义训练器导入为 UESF 管理的全局自定义训练器。

`trainers` 表通过 `trainer_type` 字段记录训练器类型。该字段取用下列三种可能之一：
- EMBEDDED：内嵌训练器，是 UESF 提供的训练器
- REGISTERED：已注册的自定义训练器。当用户首次运行使用了未注册组件的实验时，UESF 自动将该组件注册到数据库（记录 entrypoint 路径并创建源代码快照），UESF 也提供显式注册命令
- GLOBAL：已导入的全局自定义训练器，已注册的训练器通过训练器导入命令成为该类型

> 全局自定义训练器可能遭遇调试困难，需要用户自行斟酌是否使用。

UESF 支持用户对：
- 全局自定义训练器进行查看、移除、修改信息等操作；
- 已注册的自定义训练器进行查看、移除、修改信息、导入（成为全局训练器）等操作；
- 未注册的自定义训练器进行注册、导入（成为全局训练器）等操作。
