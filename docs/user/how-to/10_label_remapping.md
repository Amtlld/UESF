# 如何用标签重映射统一多数据集标签

当你需要整合多个数据集时，它们的情绪标签体系往往不一致（例如一个用 5 类细粒度情绪，另一个用 3 类，你想统一为"积极/消极"二分类）。UESF 的标签重映射数据集（Masked Dataset）解决这个问题，且不占用额外的特征存储空间。

---

## 问题场景

- `seed_preprocessed`：3 类标签（`negative`、`neutral`、`positive`）
- `deap_preprocessed`：4 类标签（`HVHA`、`HVLA`、`LVHA`、`LVLA`）
- 目标：将两个数据集都转换为二分类（`negative`、`positive`），用于同一实验

---

## 创建标签映射规则文件

编写映射规则文件（`rule_seed_binary.yml`），将旧语义标签映射到新语义标签：

```yaml
# rule_seed_binary.yml
# 格式：旧语义标签: 新语义标签
negative: negative
neutral: positive    # 将 neutral 归为 positive
positive: positive
```

```yaml
# rule_deap_binary.yml
HVHA: positive    # High Valence High Arousal → positive
HVLA: positive    # High Valence Low Arousal → positive
LVHA: negative    # Low Valence High Arousal → negative
LVLA: negative    # Low Valence Low Arousal → negative
```

新的数字标签由新语义标签按 ASCII 排序后重新编号（`negative=0`，`positive=1`）。

---

## 创建标签重映射数据集

```bash
# 为 SEED 数据集创建二分类版本
uesf data preprocessed mask seed_preprocessed \
  --out-name seed_binary \
  --mapping-file rule_seed_binary.yml

# 为 DEAP 数据集创建二分类版本
uesf data preprocessed mask deap_preprocessed \
  --out-name deap_binary \
  --mapping-file rule_deap_binary.yml
```

框架会：
1. 读取源数据集的标签数组和 `numeric_to_semantic` 映射
2. 根据规则文件计算新标签
3. 将新标签数组存储为 `<data_dir>/masked/<out_name>/labels.npy`
4. 特征数据**不复制**，运行时直接引用源数据集的 `features.npy`

---

## 在项目和实验中使用

标签重映射数据集对框架完全透明，使用方式与普通预处理数据集完全一致。

在 `project.yml` 中声明：

```yaml
preprocessed_datasets:
  - seed_binary
  - deap_binary
```

在实验 YAML 中使用：

```yaml
datasets:
  seed:
    name: seed_binary        # 直接使用重映射数据集名称
    split:
      strategy: holdout
      dimension: subject
      train_ratio: 0.8
      val_ratio: 0.1
      test_ratio: 0.1
```

---

## 查看所有预处理和重映射数据集

```bash
uesf data preprocessed list
```

输出中会区分普通预处理数据集和标签重映射数据集（Masked）。

---

## 删除标签重映射数据集

```bash
uesf data preprocessed remove seed_binary
```

只删除 `labels.npy` 文件，源预处理数据集 `seed_preprocessed` 不受影响。
