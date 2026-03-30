# 07 内置优化器与调度器映射规则详细设计

本规范用于说明在实验配置（如 `<experiment_name>.yml`）的 `training` 模块下，框架如何将用户指定的字符串（标识符）解析实例化为实际的 PyTorch 组件。框架遵循 **"透明透传 (Transparent Passthrough)"** 原则。

在配置文件 `experiments` 中，优化器和调度器的声明遵循相同的结构：通过 `name` 选择组件类型，通过 `params` 透传实例化的关键字参数 (kwargs)。

## 1. 优化器映射 (Optimizers)

系统拦截特定的优化器字符串标识符，并将其映射为对应的 `torch.optim` 类。

| UESF 标识符 (`training.optimizer.name`) | 对应的 PyTorch 原生类 | 说明 |
|-------------------------|---------------------------|------|
| `sgd`                   | `torch.optim.SGD`         | 经典随机梯度下降，常需在 `params` 额外传参 `momentum`, `nesterov` 等 |
| `adam`                  | `torch.optim.Adam`        | 自适应矩估计 |
| `adamw`                 | `torch.optim.AdamW`       | 推荐的默认视觉/时序优化器 |
| `adagrad`               | `torch.optim.Adagrad`     | |
| `adadelta`              | `torch.optim.Adadelta`    | |
| `rmsprop`               | `torch.optim.RMSprop`     | |
| `radam`                 | `torch.optim.RAdam`       | |
| `nadam`                 | `torch.optim.NAdam`       | |

*注：若用户设定的字符串不在此内置列表中，且不在 `project.yml` 中注册为项目级自定义优化器，系统将向用户抛出异常。用户亦可通过继承 `BaseTrainer` 重写 `configure_optimizers()` 使用定制的参数更新策略。*

## 2. 学习率调度器映射 (LR Schedulers)

系统同样内置了一系列常用的学习率调度策略映射。

| UESF 标识符 (`training.scheduler.name`) | 对应的 PyTorch 原生类 | 官网常见必填参数示例 |
|---------------------------|--------------------------------------------|-----------------|
| `step_lr`                 | `torch.optim.lr_scheduler.StepLR`          | `step_size`, `gamma` |
| `multi_step_lr`           | `torch.optim.lr_scheduler.MultiStepLR`     | `milestones`, `gamma` |
| `exponential_lr`          | `torch.optim.lr_scheduler.ExponentialLR`   | `gamma` |
| `linear_lr`               | `torch.optim.lr_scheduler.LinearLR`        | `start_factor`, `end_factor`, `total_iters` |
| `cosine_annealing_lr`     | `torch.optim.lr_scheduler.CosineAnnealingLR` | `T_max`, `eta_min` |
| `cosine_annealing_warm_restarts` | `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`| `T_0`, `T_mult`, `eta_min` |
| `reduce_lr_on_plateau`    | `torch.optim.lr_scheduler.ReduceLROnPlateau` | `mode`, `factor`, `patience` |
| `one_cycle_lr`            | `torch.optim.lr_scheduler.OneCycleLR`      | `max_lr`, `total_steps` |

## 3. 参数透传与校验机制 (Parameter Passthrough)

- **校验缺省：** UESF 不在自身代码逻辑中硬编码验证类似 `betas`, `eps`, `momentum` 等层出不穷的超参数字段有效性。框架直接利用 Python 解包，如 `torch.optim.Adam(**kwargs)`，所有合法性依靠 PyTorch 原生 `__init__` 函数校验。
- **对照官网：** 因此，所有的配置参数命名与类型都需**严格遵守 [PyTorch 官档 (torch.optim)](https://pytorch.org/docs/stable/optim.html) 的规定**。若发生参数拼写错误引发的 `TypeError`，UESF 仅负责以清晰的 `Traceback` 提示上抛，建议用户按照 PyTorch 接口查阅文档。

## 4. 实验配置样例

经过上述映射原则，一份体现复杂优化控制流的 YAML 配置长卷展示如下，可见它的书写具备极高的原生兼容性表达。

```yaml
training:
  epochs: 100
  batch_size: 64
  
  optimizer:
    name: "adamw"             # 映射为 torch.optim.AdamW
    params:
      lr: 1e-3                # 学习率直接填进 params
      weight_decay: 1e-4
      betas: [0.9, 0.999]     # PyTorch 支持的额外高阶参数
      eps: 1e-8
  
  scheduler:
    name: "cosine_annealing_warm_restarts" # 映射为同名 PyTorch 原生类
    params:
      T_0: 10
      T_mult: 2
      eta_min: 1e-6
```
