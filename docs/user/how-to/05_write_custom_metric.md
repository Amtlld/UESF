# 如何编写自定义评估指标

本指南说明如何编写符合 UESF 规范的自定义指标函数，并在实验中使用。

---

## 统一指标函数签名

所有指标函数（无论是内置的还是自定义的）都遵循同一签名：

```python
def my_metric(
    preds: torch.Tensor,
    targets: torch.Tensor,
    **kwargs,
) -> float | dict:
    ...
```

### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `preds` | `torch.Tensor` | **Epoch 级**聚合后的完整预测张量，形状 `(n_samples,)`（已取 argmax），来自整个验证集或测试集 |
| `targets` | `torch.Tensor` | 对应的真实标签张量，形状 `(n_samples,)` |
| `**kwargs` | - | 可通过实验 YAML 传入额外参数（见下方说明） |

> **注意** `preds` 不是单个 batch 的输出，而是整个验证/测试集上所有 batch 的 `preds` 拼接后的结果。框架在 epoch 结束时统一执行此聚合，然后调用指标函数。这避免了批次级平均引入的统计偏差（在类别不均衡时尤为重要）。

### 返回值规范

| 类型 | 示例 | 说明 |
|------|------|------|
| `float` | `0.8523` | 单值指标，直接作为该指标的值 |
| `dict` | `{"matrix": [[...]], "per_class": [...]}` | 复杂指标，值必须是 JSON 可序列化的 Python 对象 |

---

## 示例 1：均衡准确率（Balanced Accuracy）

```python
# src/metrics/balanced_accuracy.py
import torch


def balanced_accuracy(preds: torch.Tensor, targets: torch.Tensor, **kwargs) -> float:
    """每类准确率的均值，在类别不均衡时比普通准确率更公平。"""
    classes = targets.unique()
    per_class_acc = []
    for cls in classes:
        mask = targets == cls
        acc = (preds[mask] == targets[mask]).float().mean().item()
        per_class_acc.append(acc)
    return sum(per_class_acc) / len(per_class_acc)
```

---

## 示例 2：返回字典的复杂指标

返回字典时，所有值必须是 JSON 可序列化的 Python 原生类型（不能是 `numpy.ndarray`，需要先转为 list）：

```python
# src/metrics/detailed_report.py
import torch


def class_report(
    preds: torch.Tensor,
    targets: torch.Tensor,
    class_names: list = None,
    **kwargs,
) -> dict:
    """返回每类的精确率、召回率和 F1。"""
    classes = targets.unique().tolist()
    report = {}
    for cls in classes:
        cls_name = class_names[cls] if class_names else str(cls)
        tp = ((preds == cls) & (targets == cls)).sum().item()
        fp = ((preds == cls) & (targets != cls)).sum().item()
        fn = ((preds != cls) & (targets == cls)).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        report[cls_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    return report
```

在实验 YAML 中传入 `class_names`：

```yaml
evaluation:
  metrics: [accuracy, class_report]
  # 可以通过 metric_params 传入 kwargs（功能待确认，如不支持则在函数内硬编码）
```

---

## 在 project.yml 中注册

```yaml
metrics:
  balanced_accuracy:
    entrypoint: "./src/metrics/balanced_accuracy.py:balanced_accuracy"
  class_report:
    entrypoint: "./src/metrics/detailed_report.py:class_report"
```

---

## 在实验 YAML 中使用

```yaml
evaluation:
  metrics: [accuracy, f1_score, balanced_accuracy, class_report]
  k_fold_aggregation: concat
```

`metrics` 列表中直接填写注册时使用的名称。内置指标（如 `accuracy`、`f1_score`）无需注册，直接使用。

---

## 注册为全局指标

若想在多个项目间共享同一个指标函数：

```bash
uesf metric add balanced_accuracy ./src/metrics/balanced_accuracy.py:balanced_accuracy \
  --description "类别均衡准确率，适用于类别不均衡场景"
```

注册后所有项目都可以直接在 `evaluation.metrics` 中使用 `balanced_accuracy`，不需要在 `project.yml` 中再注册。

---

## 下一步

- 运行实验并查看指标输出：[运行实验与查看进度](07_run_and_monitor.md)
- 查询和比较多实验的指标结果：[查询和比较实验结果](08_query_results.md)
