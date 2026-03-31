# UESF 用户文档

UESF（Universal EEG Study Framework）是一个面向 EEG 深度学习研究的实验管理框架，通过 CLI 和 YAML 配置驱动从原始数据到实验结果的完整工作流。

## 快速找到你需要的文档

**我是新用户，想尽快跑起来**
→ [15 分钟快速上手](tutorials/01_quickstart.md)

**我想完成某项具体任务**
→ [操作指南](how-to/)

**我需要查某个命令或配置项的参数**
→ [参考手册](reference/)

**我想理解某个设计背后的原因**
→ [概念解释](concepts/)

---

## 安装

依赖环境：Python 3.10+，PyTorch 2.5+

```bash
# 推荐：在 venv 环境中安装
uv venv ~/envs/eeg-research
source ~/envs/eeg-research/bin/activate
uv pip install uesf
```

> **注意** 不要使用 `uv tool install uesf`。UESF 依赖运行时的 Python 环境来加载用户编写的模型和训练器代码，`uv tool install` 创建的隔离环境会导致路径解析失败。

安装完成后验证：

```bash
uesf --version
```

---

## UESF 是什么

EEG 深度学习实验通常涉及繁琐的重复工作：数据预处理参数需要反复调整、不同实验的切分策略难以保证一致性、结果散落在各处难以对比。UESF 将这些工作标准化。

框架的核心是 **YAML 配置驱动**：用户通过配置文件描述预处理流水线、实验参数、切分策略，框架负责执行。自定义模型和训练器通过继承基类编写后，通过入口点（entrypoint）挂载到配置中，框架在运行时自动注入数据集元信息（通道数、采样点数、类别数）并实例化。

UESF 在设计上着重解决 EEG 研究中常见的数据泄露问题：切分器支持按被试/会话维度隔离（同一被试的数据不会同时出现在训练集和测试集），在线变换严格执行 Fit-on-Train 原则（标准化参数只从训练集计算）。

---

## 核心工作流

```
原始数据（.mat）
    │
    ▼  uesf data raw register / import
原始数据集注册
    │
    ▼  uesf data preprocess run
预处理数据集（.npy）
    │
    ▼  uesf project init
项目初始化（project.yml + experiments/）
    │
    ▼  编写模型/训练器，注册到 project.yml
    │
    ▼  uesf experiment add → 编辑实验 YAML
    │
    ▼  uesf experiment run
实验执行（切分 → 在线变换 → 训练 → 评估）
    │
    ▼  uesf experiment query
查询与比较实验结果
```

---

## 文档索引

### 教程

| 文档 | 内容 |
|------|------|
| [15 分钟快速上手](tutorials/01_quickstart.md) | 从安装到看到第一次实验结果，全程只需 CLI 和 YAML |
| [端到端完整实验](tutorials/02_first_experiment.md) | 编写自定义 CNN 模型和训练器，运行 5-Fold 跨被试实验 |

### 操作指南

| 文档 | 内容 |
|------|------|
| [准备原始数据集](how-to/01_prepare_raw_data.md) | 组织目录结构，编写 raw.yml，注册或导入数据集 |
| [配置预处理流水线](how-to/02_preprocessing.md) | 三流架构，内置算子参数，运行预处理 |
| [编写自定义模型](how-to/03_write_custom_model.md) | 继承 BaseModel，实现 forward，注册到项目 |
| [编写自定义训练器](how-to/04_write_custom_trainer.md) | 继承 BaseTrainer，实现训练和验证步骤 |
| [编写自定义指标](how-to/05_write_custom_metric.md) | 指标函数签名规范，注册和使用 |
| [配置实验 YAML](how-to/06_configure_experiment.md) | 所有实验配置字段详解 |
| [运行实验与查看进度](how-to/07_run_and_monitor.md) | 运行命令，日志文件，检查点位置 |
| [查询和比较实验结果](how-to/08_query_results.md) | experiment query 用法，输出解读 |
| [管理全局组件库](how-to/09_manage_components.md) | 跨项目复用模型和训练器 |
| [标签重映射](how-to/10_label_remapping.md) | 统一多数据集标签体系 |

### 参考手册

| 文档 | 内容 |
|------|------|
| [CLI 命令手册](reference/01_cli_reference.md) | 所有命令的完整语法、参数和示例 |
| [配置文件格式](reference/02_config_reference.md) | raw.yml、preprocess.yml、project.yml、实验 YAML 的完整字段表 |
| [Python API 参考](reference/03_api_reference.md) | BaseModel、BaseTrainer、自定义指标接口规范 |
| [内置组件列表](reference/04_builtin_components.md) | 预处理算子、优化器、调度器、内置指标 |

### 概念解释

| 文档 | 内容 |
|------|------|
| [UESF 架构与数据流](concepts/01_architecture.md) | 分层架构，元数据与物理文件的关系 |
| [数据泄露防护机制](concepts/02_data_leakage_prevention.md) | 维度隔离切分，Fit-on-Train 保证 |
| [三级组件解析优先级](concepts/03_component_resolution.md) | 项目级 > 全局级 > 内置，名称遮蔽规则 |
| [实验可追溯性设计](concepts/04_traceability.md) | YAML 快照，源码快照，seed 可复现性 |
