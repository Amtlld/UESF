# UESF 设计文档索引

本索引文件提供 UESF 项目所有设计文档的导航。

## 文档目录

| 编号 | 文档 | 摘要 |
|------|------|------|
| 01 | [总体设计](01_overall_design.md) | 核心理念、分层架构、核心组件概述、使用流程、技术选型 |
| 02 | [核心交互与存储原则](02_core_principles.md) | YAML/JSON 转换、JSON 对象快照、源代码快照三大原则 |
| 03-01 | [目录规范与存储结构](03_storage/01_directory_layout.md) | UESF 管理的数据集、组件、项目的物理目录规范 |
| 03-02 | [数据库 Schema 设计](03_storage/02_database_schema.md) | 全部 SQLite 表结构定义、生命周期管理、迁移与事务策略 |
| 03-03 | [全局配置机制](03_storage/03_global_config.md) | 配置项定义、优先级规则、config.yml 覆写机制 |
| 04-01 | [Project Manager 详细设计](04_components/01_project_manager.md) | project.yml 结构规范、组件名称解析优先级 |
| 04-02 | [Data Manager 详细设计](04_components/02_data_manager.md) | Raw/Preprocessed/Masked 数据集管理与预处理器 |
| 04-03 | [Model Manager 详细设计](04_components/03_model_manager.md) | 三类模型管理、BaseModel 接口规范、REGISTERED 组件自动更新机制 |
| 04-04 | [Trainer Manager 详细设计](04_components/04_trainer_manager.md) | 三类训练器管理、BaseTrainer 接口规范、REGISTERED 组件自动更新机制 |
| 04-05 | [Experiment Manager 详细设计](04_components/05_experiment_manager.md) | 控制反转、实验配置语法、三类评估指标管理、Metric 接口规范 |
| 05-01 | [预处理流程](05_workflows/01_preprocessing_workflow.md) | 数据预处理的完整执行步骤 |
| 05-02 | [实验执行流程](05_workflows/02_experiment_workflow.md) | 实验从配置加载到结果保存的完整流程 |
| 06 | [CLI 接口参考](06_cli_reference.md) | 全部命令行接口的完整文档 |
| 07 | [技术选型](07_tech_stack.md) | Python、PyTorch、SQLite 等技术的选用理由 |

## 相关文档

- [CLI 命令结构定义](../cli_design.yml)
