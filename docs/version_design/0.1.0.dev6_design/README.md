# UESF 实验模块重设计方案

> 版本：0.1.0.dev6
> 范围：实验管理器（ExperimentManager）+ 实验模块（experiment/）
> 不变：Runner、Evaluator、BaseTrainer、BaseModel、组件管理器

---

## 文档索引

| 编号 | 文档 | 内容 |
|:-----|:-----|:-----|
| 01 | [设计原则](01_design_principles.md) | 五大核心设计原则 |
| 02 | [YAML 配置接口设计](02_yaml_config_interface.md) | 划分原语定义、Regular/UDA 模式配置、完整结构总览 |
| 03 | [模块架构设计](03_module_architecture.md) | 目录结构、核心类关系、关键接口定义 |
| 04 | [执行流程详解](04_execution_flow.md) | Regular/UDA 模式执行流程、ValSplit 分组处理、嵌套折叠处理 |
| 05 | [跨数据集对齐模块](05_cross_dataset_alignment.md) | 对齐配置、触发条件、接口保留 |
| 06 | [变更边界](06_change_boundary.md) | 与现有模块的变更程度说明 |
| 07 | [配置校验规则](07_config_validation_rules.md) | 12 条配置校验规则汇总 |
| 08 | [设计优势总结](08_design_summary.md) | 7 项设计优势 |
