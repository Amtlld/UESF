# Project Manager 详细设计

UESF 项目被定义为一套包括数据预处理和一系列实验的工作对象。一个项目包含：
- 一个或若干个预处理数据集（或原始数据集与自定义预处理流程）
- 一个或若干个模型
- 若干个实验

为了让用户可以最大程度地操作项目，UESF 完全从用户定义的 `project.yml` 而非数据库读取项目配置信息。

UESF 不在 Project 处实现过多的验证和限制逻辑，以保证预处理、模型和实验功能可以相对独立地使用，项目仅作为复用配置信息的中心地位。

## 1. `project.yml` 结构规范

`project.yml` 应遵循下面的格式：
```yaml
project-name: example

description: <project-description>

# 如果使用预处理数据集
preprocessed_datasets:
  - <preprocessed-dataset-name>
  # - <preprocessed-dataset-name> * n

# 如果使用原始数据集与自定义预处理流程
raw_datasets:
  - <raw-dataset-name>
  # - <raw-dataset-name> * n
preprocess_config: <path-to-preprocess-config>  # 若不填写，默认为 ./preprocess.yml

trainers:
  <trainer-1>:
    entrypoint: <entrypoint>

models:
  <model-1>:
    entrypoint: <entrypoint>  # e.g. "./src/models/transformer.py:MyTransformerClass"
  ...

metrics:
  <metric-name>:
    entrypoint: <entrypoint>  # e.g. "./src/utils/custom_metric.py:MyCustomScoreFunc"
  ...
```

## 2. 组件名称解析优先级

当用户在实验配置中引用模型或训练器名称时，系统按照以下优先级进行解析（由高到低）：
1. **项目级自定义组件**：在当前 `project.yml` 的 `models` 或 `trainers` 块中注册的组件
2. **全局自定义组件**（GLOBAL）：通过 `uesf model add` / `uesf trainer add` 导入的全局组件
3. **内置组件**（EMBEDDED）：UESF 内置提供的组件

若项目级自定义名与全局或内置名冲突，系统将优先使用项目级定义，并在日志中输出一条 Warning 提示用户存在名称遮蔽。
