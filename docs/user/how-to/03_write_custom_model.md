# 如何编写自定义模型

本指南说明如何编写一个符合 UESF 规范的自定义 EEG 模型，并将其注册到项目中。

---

## BaseModel 接口

所有自定义模型必须继承 `BaseModel`，它继承自 `torch.nn.Module`。

```python
class BaseModel(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int,
        electrode_list: Optional[list[str]] = None,
        **kwargs,
    ) -> None: ...

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor: ...

    def extract_features(self, x: torch.Tensor) -> torch.Tensor: ...  # 可选
```

### 框架自动注入的参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `n_channels` | int | EEG 通道数，从预处理数据集元信息读取 |
| `n_samples` | int | 每个 Epoch 的采样点数，从预处理数据集计算得出 |
| `n_classes` | int | 类别数，从 `numeric_to_semantic` 映射的条目数得出 |
| `electrode_list` | list[str] \| None | 电极名称列表，若 `raw.yml` 中未填写则为 `None` |

这些参数**由框架在运行时自动注入**，你不需要在实验 YAML 的 `model.params` 中填写它们。

---

## 最小可用示例

```python
# src/models/linear.py
import torch.nn as nn
from uesf.components.base_model import BaseModel

class LinearClassifier(BaseModel):
    def __init__(self, n_channels, n_samples, n_classes, **kwargs):
        super().__init__(n_channels, n_samples, n_classes, **kwargs)
        self.fc = nn.Linear(n_channels * n_samples, n_classes)

    def forward(self, x, **kwargs):
        # x: (batch, n_channels, n_samples)
        return self.fc(x.flatten(1))   # → (batch, n_classes)
```

注意：
- 必须调用 `super().__init__(n_channels, n_samples, n_classes, **kwargs)`
- `forward` 接收形状为 `(batch, n_channels, n_samples)` 的输入张量，返回形状为 `(batch, n_classes)` 的 logits

---

## 实用示例：1D CNN 情绪分类模型

```python
# src/models/cnn.py
import torch
import torch.nn as nn
from uesf.components.base_model import BaseModel


class EmotionCNN(BaseModel):
    """基于 1D 卷积的 EEG 情绪分类模型。"""

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int,
        hidden_size: int = 128,
        dropout_rate: float = 0.5,
        **kwargs,
    ):
        super().__init__(n_channels, n_samples, n_classes, **kwargs)

        # 时域卷积提取特征
        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, hidden_size, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x: (batch, n_channels, n_samples)
        x = self.relu(self.conv1(x))   # → (batch, 64, n_samples)
        x = self.relu(self.conv2(x))   # → (batch, hidden_size, n_samples)
        x = self.pool(x).squeeze(-1)   # → (batch, hidden_size)
        x = self.dropout(x)
        return self.fc(x)              # → (batch, n_classes)
```

### 传入自定义超参数

实验 YAML 中 `model.params` 里的任何字段都会以 `**kwargs` 形式传入 `__init__`：

```yaml
# experiments/baseline_cnn.yml
model:
  name: emotion_cnn
  params:
    hidden_size: 256
    dropout_rate: 0.3
```

对应模型代码中的 `hidden_size=256, dropout_rate=0.3`。

---

## 可选方法：extract_features

用于特征可视化、迁移学习等场景：

```python
def extract_features(self, x: torch.Tensor) -> torch.Tensor:
    """返回全连接层之前的特征向量。"""
    x = self.relu(self.conv1(x))
    x = self.relu(self.conv2(x))
    return self.pool(x).squeeze(-1)  # → (batch, hidden_size)
```

默认实现会抛出 `NotImplementedError`，只有在需要时才覆盖。

---

## 在 project.yml 中注册

编辑项目根目录的 `project.yml`，在 `models` 块中添加：

```yaml
project-name: emotion_recognition

preprocessed_datasets:
  - seed_preprocessed

models:
  emotion_cnn:                                   # 在实验 YAML 中使用的名称
    entrypoint: "./src/models/cnn.py:EmotionCNN" # 文件路径:类名
```

### 入口点（entrypoint）格式

```
"<相对路径>:<类名>"
```

相对路径相对于 `project.yml` 所在目录，与执行 `uesf` 命令时的工作目录无关。

---

## 源码变更自动检测

修改模型代码后**不需要手动重新注册**。UESF 在每次 `uesf experiment run` 时会检测源文件的 SHA256 哈希值，若发现变更：

1. 将旧版本归档（以 `emotion_cnn_<sha256前8位>` 命名，标记为 `obsolete`）
2. 以原名称 `emotion_cnn` 注册新版本
3. 在日志中输出提示

这保证每次实验结果都与当时的源码对应，具备完整的可追溯性。

---

## 注册为全局组件

若想在多个项目间复用同一个模型，可以注册到全局组件库：

```bash
uesf model add emotion_cnn ./src/models/cnn.py:EmotionCNN --description "1D CNN 情绪识别模型"
```

注册后可在任何项目的 `project.yml` 中使用 `emotion_cnn`，无需再填写 `entrypoint`。

---

## 下一步

编写模型后，编写配套的训练器：[编写自定义训练器](04_write_custom_trainer.md)
