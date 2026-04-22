# 01. 实验级标签映射 (label_mapping)

## 动机

多数据集 EEG 实验经常遇到"语义等价但标签字符串/整数不同"的情况:

- FACED 使用 `anger / disgust / fear / sadness / neutrality / amusement / inspiration / joy / tenderness`
- THU-EP 使用首字母大写的 `Anger / Disgust / Fear / Sadness / Neutral / Amusement / Inspiration / Joy / Tenderness`
- 同时我们可能想把这些 9 类细粒度情绪折叠成 `positive / neutral / negative` 三分类

现状要做到这件事,必须调用 `uesf.managers.data_manager.DataManager.create_masked` 预生成一份 `masked_datasets` 物化副本,并在实验中引用该副本。每换一次映射方案就要生成一份新的物化副本,管理成本高、实验配置里也看不出语义意图。

本特性把等价能力下沉到实验 YAML,形成实验级(一次性、声明式、无副作用)标签重映射。

## YAML 接口

在 `datasets[alias]` 下增加可选块:

```yaml
datasets:
  faced_test:
    name: faced_test
    transforms:
      - name: zscore_normalize
        scope: per_dataset
    label_mapping:
      anger: negative
      disgust: negative
      fear: negative
      sadness: negative
      neutrality: neutral
      amusement: positive
      inspiration: positive
      joy: positive
      tenderness: positive
  thuep_test:
    name: thuep_test
    transforms:
      - name: zscore_normalize
        scope: per_dataset
    label_mapping:
      Anger: negative
      Disgust: negative
      Fear: negative
      Sadness: negative
      Neutral: neutral
      Amusement: positive
      Inspiration: positive
      Joy: positive
      Tenderness: positive
```

- **键**:该数据集 `numeric_to_semantic` 元数据中的旧 semantic 字符串(大小写敏感)。
- **值**:目标 semantic 字符串。不同数据集可用同一目标集合来对齐(上例两端都收敛到 `{positive, neutral, negative}`)。

## 算法

与 `DataManager.create_masked` 保持一致:

1. 从 `metadata_cache[alias]["numeric_to_semantic"]` 取 `{old_num: old_semantic}`。
2. 严格校验 `set(label_mapping.keys()) == set(numeric_to_semantic.values())`。
3. 取 `new_semantics = sorted(set(label_mapping.values()))`,按 ASCII 升序为新 semantic 分配 `0..len-1` 的 numeric。
4. 构造 `old_num → new_num` 查找表,生成新的 labels 数组,原地替换 `labels_cache[alias]`。
5. 更新 `metadata_cache[alias]` 的 `numeric_to_semantic` 和 `n_classes`。

上述 YAML 示例对两个数据集都得到 `{0: negative, 1: neutral, 2: positive}`,下游模型的输出维度、评估指标、`LabelAligner` 的一致性校验自然对齐。

## 数据流位置

`ExperimentManager.run()` 中的顺序:

```
_load_all_datasets  →  apply_label_mapping  →  _apply_alignment / LabelAligner.validate  →  ExperimentContext → ExperimentExecutor
```

`dataset_cache`(EEG 5-D 数组)不动;仅改 `labels_cache[alias]` 与 `metadata_cache[alias]`。

## 与既有机制的关系

| 机制 | 层次 | 物化 | 适用 |
|------|------|------|------|
| `masked_datasets`(`data_manager.create_masked`) | 数据管理层 | 是,落盘到 `data_dir/masked/` 并写 DB | 需要跨多个实验复用同一映射方案 |
| `label_mapping`(本特性) | 实验配置层 | 否,只在内存中生效 | 实验级消融、一次性对齐 |
| `LabelAligner`(`experiment/alignment.py`) | 实验配置层 | — | 在多数据集场景下校验 `n_classes` 与 `numeric_to_semantic` 一致;运行在 `apply_label_mapping` 之后 |
| `transforms`(`experiment/transforms.py`) | 实验配置层 | 仅内存 | 处理 EEG 数据本身(归一化等),不触碰标签 |

`label_mapping` 与 `transforms` 并列,但不进入 transform 注册表:前者是标签的元数据重映射(一次成型、无需 fit 在 train-split、无 snapshot/restore),后者是可能 fit-on-train 的数据变换。

## 校验(R31)

### 配置层(静态,`config_schema.ConfigValidator.validate`)

- `label_mapping` 若存在必须是**非空 `dict[str, str]`**,键与值均为非空字符串。
- 多数据集场景下,所有声明了 `label_mapping` 的 alias 的 `set(values())` 必须一致。
- 不读 DB,与现有 R1–R30 风格一致。

### 运行时层(`apply_label_mapping`)

- 数据集必须具备 `numeric_to_semantic` 元数据(DB 记录中 `numeric_to_semantic` 字段非空)。
- `set(mapping.keys()) == set(numeric_to_semantic.values())`,否则报错并列出 missing/extra。

## 边界与约束

- **作用于 `masked_datasets` 源**:允许。masked 记录自身也带 `numeric_to_semantic`(记录的是上一次映射结果),按本算法可以再映射一次,语义一致。
- **未声明 `label_mapping` 的 alias**:保持原标签与原元数据;若多数据集场景下与其它 alias 不兼容,由 `LabelAligner.validate` 运行时捕获。
- **幂等性**:对同一份 `labels_cache` / `metadata_cache` 重复调用 `apply_label_mapping` 会基于"已映射后的 numeric_to_semantic"做二次映射;下游实验流程在每个 `run()` 内只调用一次,此行为仅在测试或嵌套使用中暴露。
- **不影响**:DB 记录(不写 `masked_datasets` 表)、`eeg_data.npy`、`labels.npy` 磁盘文件。

## 示例:三分类情绪实验完整配置

```yaml
experiment_name: faced_thuep_three_class
seed: 42
mode: regular
datasets:
  faced_test:
    name: faced_test
    transforms:
      - name: zscore_normalize
        scope: per_dataset
    label_mapping:
      anger: negative
      disgust: negative
      fear: negative
      sadness: negative
      neutrality: neutral
      amusement: positive
      inspiration: positive
      joy: positive
      tenderness: positive
  thuep_test:
    name: thuep_test
    transforms:
      - name: zscore_normalize
        scope: per_dataset
    label_mapping:
      Anger: negative
      Disgust: negative
      Fear: negative
      Sadness: negative
      Neutral: neutral
      Amusement: positive
      Inspiration: positive
      Joy: positive
      Tenderness: positive
alignment:
  channel: intersection
  label: true
split:
  strategy: holdout
  dimension: dataset
  assign:
    train: [faced_test]
    test: [thuep_test]
model:
  name: my_model
training:
  epochs: 30
```
