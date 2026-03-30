# Model Manager 详细设计

Model Manager 负责管理模型。UESF 支持三类模型：
- 内置模型（EMBEDDED）
- 自定义模型（REGISTERED）
- 全局自定义模型（GLOBAL）

数据库表结构详见 [`models` 表](../03_storage/02_database_schema.md#models-表)。

## 1. 内置模型

UESF 开发者维护一系列已发表的 EEG 深度学习模型，供用户使用。

使用 UESF 内置模型时，用户仅需在实验配置文档中指定模型名称，UESF 会自动加载内置模型。
```yaml
model: "EEGConformer"
```

## 2. 自定义模型

UESF 可以作为一个 Python 库被导入用户 Python 脚本中。用户可以利用 UESF 提供的模型基类来自定义模型。

用户通过编写继承 UESF 提供的模型基类的模型，并通过实验配置进行导入的，被视作用户管理的模型。

一个可用的自定义模型需要满足如下两个必要条件：
1. 自定义模型源代码存在
2. 自定义模型在项目配置文件中注册

模型在项目配置文件 `project.yml` 中注册的示例：
```yaml
models:
  My_Transformer:
    entrypoint: "./src/models/transformer.py:MyTransformerClass"
  ...
```

> **规范：路径解析基准点**
> 任何在框架（例如 project.yml 或模型、训练器的配置中）填写的相对路径（如 `./src`），系统在底层运行解析时必须严格约定：**永远相对于 `project.yml` 所在的「项目工作目录」(Project Directory) 进行拼接**，绝不可因为用户在不同层级执行 `uesf` 命令而简单挂载当前工作目录 (CWD)。这能极大减轻代码与模型找不到路径的风险。

### 2.1 BaseModel 接口规范

为适应极度解耦的数据流结构，自定义模型所继承的模型基类接口规范如下：
```python
import torch
import torch.nn as nn
from typing import Optional, List

class BaseModel(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int,
        electrode_list: Optional[List[str]] = None,
        **kwargs
    ):
        """
        基类初始化。框架在实例化模型时，会从数据集元数据中自动提取
        n_channels、n_samples、n_classes 等维度信息，并与实验配置中
        model.params 的用户自定义参数一并注入。
        
        :param n_channels: 数据集的通道数（由框架从数据集元数据自动注入）
        :param n_samples: 数据集的采样点数（由框架从数据集元数据自动注入）
        :param n_classes: 分类任务的类别数（由框架从数据集 numeric_to_semantic 自动推算注入）
        :param electrode_list: (可选) 电极列表（由框架从数据集元数据自动注入）
        :param kwargs: 实验配置中 model.params 用户自定义的额外参数
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.electrode_list = electrode_list

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """标准前向传播接口。"""
        raise NotImplementedError

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """(可选) 特征表征提取接口。对需要截获分类头前一层特征的复杂任务提供结构化支撑。"""
        pass
```

## 3. 模型管理

UESF 在数据库 `models` 表中记录模型元信息。无论是项目级自定义组件还是全局组件，均需注册到数据库中。

UESF 支持用户将自定义模型导入为 UESF 管理的全局自定义模型。

`models` 表通过 `model_type` 字段记录模型类型。该字段取用下列三种可能之一：
- EMBEDDED：内嵌模型，是 UESF 提供的模型
- REGISTERED：已注册的自定义模型。当用户首次运行使用了未注册组件的实验时，UESF 自动将该组件注册到数据库（记录 entrypoint 路径并创建源代码快照），UESF 也提供显式注册命令
- GLOBAL：已导入的全局自定义模型，已注册的模型通过模型导入命令成为该类型

> 全局自定义模型可能遭遇调试困难，需要用户自行斟酌是否使用。

UESF 支持用户对：
- 全局自定义模型进行查看、移除、修改信息等操作；
- 已注册的自定义模型进行查看、移除、修改信息、导入（成为全局模型）等操作；
- 未注册的自定义模型进行注册、导入（成为全局模型）等操作。

### 3.1 REGISTERED 模型的源码变更自动检测与重新注册

此机制**仅作用于 `REGISTERED` 类型**（entrypoint 指向用户项目目录的模型）。`EMBEDDED` 和 `GLOBAL` 类型不受影响。

在 `uesf experiment run` 执行的"配置加载与组件初始化"阶段，框架会对实验引用的每个 `REGISTERED` 模型执行以下检测流程：

1. 读取 entrypoint 指向的源文件当前内容，计算其 SHA256 哈希值
2. 与数据库中 `source_code_snapshot` 字段存储的快照内容对比
3. **若哈希一致**：无操作，继续加载
4. **若哈希不一致**，触发重新注册流程：
   a. 取旧快照内容的 SHA256 前 8 位作为版本标识符 `<hash8>`（确保同一旧版本永远映射到同一后缀，操作幂等）
   b. 将旧记录的 `name` 重命名为 `<original-name>_<hash8>`，并将 `is_obsolete` 置为 `1`
   c. 以原始名称创建新记录，写入更新后的源码快照，`model_type` 保持 `REGISTERED`，`is_obsolete = 0`
   d. 在日志中输出 INFO 提示：`检测到模型 <name> 源码已变更，旧版本已归档为 <name>_<hash8>，已自动重新注册`

过时（`is_obsolete = 1`）的模型记录在 `uesf model list` 默认输出中隐藏，可通过 `--show-obsolete` 参数显示。历史实验记录的 `model_id` 外键仍指向对应的旧版本记录，保证可追溯性不受影响。
