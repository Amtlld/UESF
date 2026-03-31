# 端到端完整实验

本教程带你完成一个完整的跨被试 EEG 情绪识别实验：从编写模型和训练器，到配置 5-Fold 交叉验证实验，再到分析结果。

**场景**：SEED 数据集，14 名被试，62 通道，200Hz，3 类情绪（negative/neutral/positive），按被试做 5 折交叉验证。

**前提**：已完成[准备数据集](../how-to/01_prepare_raw_data.md)和[预处理](../how-to/02_preprocessing.md)，`seed_preprocessed` 可用。

---

## 项目结构规划

```
emotion_recognition/           # 项目根目录
├── project.yml                # 项目配置
├── experiments/
│   └── baseline_cnn.yml       # 实验配置（下面编写）
└── src/
    ├── models/
    │   └── cnn.py             # 自定义模型（下面编写）
    └── trainers/
        └── trainer.py         # 自定义训练器（下面编写）
```

初始化项目：

```bash
mkdir emotion_recognition && cd emotion_recognition
uesf project init
mkdir -p src/models src/trainers
```

---

## 步骤 1：编写模型

创建 `src/models/cnn.py`：

```python
# src/models/cnn.py
from typing import Optional

import torch
import torch.nn as nn

from uesf.components.base_model import BaseModel


class EmotionCNN(BaseModel):
    """基于 1D 卷积的 EEG 情绪分类模型。

    框架自动注入 n_channels、n_samples、n_classes，
    无需在实验 YAML 的 model.params 中填写这三个参数。
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int,
        hidden_size: int = 128,
        dropout_rate: float = 0.5,
        electrode_list: Optional[list] = None,
        **kwargs,
    ):
        super().__init__(n_channels, n_samples, n_classes, electrode_list, **kwargs)

        # 两层时域卷积
        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, hidden_size, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x: (batch, n_channels, n_samples)
        x = self.relu(self.conv1(x))   # (batch, 64, n_samples)
        x = self.relu(self.conv2(x))   # (batch, hidden_size, n_samples)
        x = self.pool(x).squeeze(-1)   # (batch, hidden_size)
        x = self.dropout(x)
        return self.fc(x)              # (batch, n_classes)
```

---

## 步骤 2：编写训练器

创建 `src/trainers/trainer.py`：

```python
# src/trainers/trainer.py
from typing import Any

import torch
import torch.nn.functional as F

from uesf.components.base_trainer import BaseTrainer


class EmotionTrainer(BaseTrainer):
    """标准交叉熵分类训练器。

    Trainer 全权负责梯度的完整生命周期，
    框架的 Runner 只负责传递 batch 和收集日志。
    """

    def training_step(
        self,
        batch: dict[str, tuple[torch.Tensor, torch.Tensor]],
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, Any]:
        data, labels = batch["main"]
        data = data.to(self.device)
        labels = labels.to(self.device)

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

        # 框架在 epoch 结束时将所有 batch 的 preds/targets 拼接，
        # 然后整体传入指标函数计算，避免批次平均引入的统计偏差
        return {
            "preds": preds.cpu(),
            "targets": labels.cpu(),
        }
```

---

## 步骤 3：配置 project.yml

编辑项目根目录的 `project.yml`：

```yaml
project-name: emotion_recognition
description: SEED 情绪识别跨被试实验

preprocessed_datasets:
  - seed_preprocessed

models:
  emotion_cnn:
    entrypoint: "./src/models/cnn.py:EmotionCNN"

trainers:
  emotion_trainer:
    entrypoint: "./src/trainers/trainer.py:EmotionTrainer"
```

验证项目配置：

```bash
uesf project info
```

---

## 步骤 4：配置实验 YAML

创建实验配置：

```bash
uesf experiment add --name baseline_cnn
```

编辑生成的 `experiments/baseline_cnn.yml`：

```yaml
name: baseline_cnn
description: "EmotionCNN 1D，5-Fold 跨被试交叉验证，Adam + 余弦退火"
seed: 42

model:
  name: emotion_cnn
  params:
    hidden_size: 128
    dropout_rate: 0.5

trainer:
  name: emotion_trainer
  params: {}

datasets:
  main:
    name: seed_preprocessed
    split:
      strategy: k-fold
      dimension: subject        # 按被试切分：同一被试不跨训练/测试集
      k-folds: 5
      val_ratio_in_train: 0.1   # 从训练折划出 10% 用于早停监控
      shuffle: true
    transforms:
      - name: zscore_normalize
        fit_on: train           # 只用训练集计算均值和标准差
        apply_to: all           # 统一应用到 train/val/test

dataloaders:
  train:
    main: "main.train"
  val:
    main: "main.val"
  test:
    main: "main.test"

training:
  epochs: 100
  batch_size: 64
  optimizer:
    name: adam
    params:
      lr: 0.001
      weight_decay: 1e-4
  scheduler:
    name: cosine_annealing_lr
    params: { T_max: 50, eta_min: 1e-6 }
  gradient_clip:
    max_norm: 1.0
  early_stopping:
    monitor: val_accuracy
    patience: 15
    mode: max

evaluation:
  metrics: [accuracy, f1_score, precision, recall, auroc]
  k_fold_aggregation: concat    # 所有折的预测拼接后一次性计算指标

logging:
  use_wandb: false
  checkpoint_metric: val_accuracy
```

---

## 步骤 5：运行实验

```bash
uesf experiment run --exp baseline_cnn
```

框架按以下顺序执行：

1. 加载 `seed_preprocessed` 的特征和标签
2. 按被试维度切分为 5 折
3. 对每一折：
   - 用训练集拟合 Z-Score 标准化参数，应用到所有集合
   - 初始化 `EmotionCNN` 和 `EmotionTrainer`
   - 执行训练循环（含早停和检查点保存）
4. 将 5 折的 test 集预测拼接，计算最终指标
5. 将结果写入数据库

训练过程中会显示每个 epoch 的指标：

```
Fold 1/5 ━━━━━━━━━━━━━━━━━━━━━━━━━ 100/100 epochs
  Epoch 100  loss=0.312  val_accuracy=0.7823  val_f1_score=0.7801
  早停：val_accuracy 在 epoch 85 达到最高值 0.7891
```

---

## 步骤 6：查询结果

```bash
uesf experiment query --metrics accuracy,f1_score,auroc --status COMPLETED
```

输出示例：

```
┌──────────────┬──────────┬──────────┬──────────┬────────┐
│ 实验名       │ 状态     │ accuracy │ f1_score │ auroc  │
├──────────────┼──────────┼──────────┼──────────┼────────┤
│ baseline_cnn │ COMPLETED│ 0.7856   │ 0.7821   │ 0.9134 │
└──────────────┴──────────┴──────────┴──────────┴────────┘
```

`concat` 模式下，准确率是所有 5 折测试集样本统一计算的结果（不是各折平均）。

---

## 步骤 7：创建对比实验

基于基线实验复制一个新配置，修改超参数进行对比：

```bash
uesf experiment add --name deeper_cnn --from baseline_cnn
```

编辑 `experiments/deeper_cnn.yml`，修改 `model.params`：

```yaml
model:
  name: emotion_cnn
  params:
    hidden_size: 256     # 从 128 增大到 256
    dropout_rate: 0.3
```

运行对比实验：

```bash
uesf experiment run --exp deeper_cnn
```

同时查询两个实验的结果：

```bash
uesf experiment query --metrics accuracy,f1_score --status COMPLETED
```

---

## 下一步

- 了解数据泄露防护的详细原理：[数据泄露防护机制](../concepts/02_data_leakage_prevention.md)
- 编写自定义评估指标：[编写自定义指标](../how-to/05_write_custom_metric.md)
- 统一多数据集的标签体系：[标签重映射](../how-to/10_label_remapping.md)
