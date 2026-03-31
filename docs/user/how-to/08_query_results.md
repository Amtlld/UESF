# 如何查询和比较实验结果

---

## 基础查询

在项目根目录下运行（无参数时显示所有实验的所有指标）：

```bash
uesf experiment query
```

---

## 按指标筛选

只显示你关心的指标列：

```bash
uesf experiment query --metrics accuracy,f1_score,auroc
```

输出示例：

```
┌──────────────┬──────────┬──────────┬──────────┬────────┐
│ 实验名       │ 状态     │ accuracy │ f1_score │ auroc  │
├──────────────┼──────────┼──────────┼──────────┼────────┤
│ baseline_cnn │ COMPLETED│ 0.7856   │ 0.7821   │ 0.9134 │
│ deeper_cnn   │ COMPLETED│ 0.8012   │ 0.7988   │ 0.9267 │
│ quickstart   │ FAILED   │ -        │ -        │ -      │
└──────────────┴──────────┴──────────┴──────────┴────────┘
```

---

## 按状态筛选

只显示已完成的实验：

```bash
uesf experiment query --metrics accuracy,f1_score --status COMPLETED
```

---

## K-Fold 结果的显示

使用 `k_fold_aggregation: concat` 时，指标是所有折测试集拼接后统一计算的结果，显示为单个数值。

使用 `k_fold_aggregation: mean_std` 时，每个指标显示为 `均值 ± 标准差`：

```
┌──────────────────┬──────────────────────┬─────────────────────┐
│ 实验名           │ accuracy             │ f1_score            │
├──────────────────┼──────────────────────┼─────────────────────┤
│ baseline_cnn_ms  │ 0.7856 ± 0.0234      │ 0.7821 ± 0.0198     │
└──────────────────┴──────────────────────┴─────────────────────┘
```

---

## 查看混淆矩阵等复杂指标

若指标函数返回字典（如 `confusion_matrix`），查询结果会展开显示：

```bash
uesf experiment query --metrics confusion_matrix --status COMPLETED
```

输出以 JSON 格式嵌入表格或单独展示，具体格式由终端宽度决定。

---

## 列出实验配置

查看当前项目下的所有实验及其状态（不显示指标）：

```bash
uesf experiment list
```

输出示例：

```
┌──────────────┬──────────┬───────────────────────────────┐
│ 实验名       │ 状态     │ 描述                           │
├──────────────┼──────────┼───────────────────────────────┤
│ baseline_cnn │ COMPLETED│ EmotionCNN 1D，5-Fold 跨被试  │
│ deeper_cnn   │ COMPLETED│ 更宽的 hidden_size，对比实验  │
│ new_exp      │ PENDING  │ -                             │
└──────────────┴──────────┴───────────────────────────────┘
```

---

## 找到对应的检查点

根据实验查询结果，最优检查点存放在：

```
experiments/results/<实验名>/checkpoints/
```

K-Fold 实验每折保存一个最优检查点文件（`fold_<n>_best.pt`）。
