# UESF 实验配置增强:label_mapping

> 版本:0.1.0.dev9
> 范围:实验模块(`experiment/`)+ `ExperimentManager.run` 的加载段
> 不变:Runner、Evaluator、BaseTrainer、BaseModel、组件管理器、数据管理器

---

## 文档索引

| 编号 | 文档 | 内容 |
|:-----|:-----|:-----|
| 01 | [实验级标签映射](01_label_mapping.md) | 动机、YAML 配置示例、算法、与 LabelAligner / masked_datasets / transforms 的关系、运行/配置两层校验、边界行为 |

## 本版本要点

- 在实验 YAML 的 `datasets[alias]` 下新增可选 `label_mapping: {old_semantic: new_semantic}`。
- 加载完预处理数据集后,UESF 自动按"新 semantic ASCII 升序"重新分配 numeric,原地更新 `labels_cache[alias]` 与 `metadata_cache[alias]` 的 `n_classes` / `numeric_to_semantic`,使下游 `LabelAligner`、模型实例化等看到映射后的状态。
- `ConfigValidator` 新增 R31 做结构和跨数据集目标集合一致性校验;运行时层负责"键与数据集 semantic 全集完全一致"的深度校验。
- 与既有 `masked_datasets` 并存:前者为实验级一次性声明,后者为可跨实验复用的物化副本。
