# 如何运行实验与查看进度

---

## 运行实验

在项目根目录（包含 `project.yml` 的目录）下执行：

```bash
uesf experiment run --exp baseline_cnn
```

> **注意** 必须在项目根目录下运行，或在其他目录运行时通过 `--project-dir` 指定项目路径：`uesf experiment run --exp baseline_cnn --project-dir /path/to/project`

---

## 框架执行流程

`uesf experiment run` 按以下步骤自动执行：

1. **加载配置**：读取实验 YAML 和 `project.yml`，解析所有组件名称
2. **组件初始化**：检测模型/训练器源码的 SHA256，若有变更则自动归档旧版本并注册新版本
3. **数据加载**：读取预处理数据集的 `.npy` 特征和标签
4. **数据切分**：按配置的切分策略生成各折的索引快照
5. **在线变换**：拟合并应用 Z-Score 等变换（仅从训练集学习参数）
6. **训练与评估**：执行训练循环，早停，保存最优检查点
7. **结果入库**：将指标结果写入数据库，状态更新为 `COMPLETED`

---

## 进度显示

框架使用 Rich 显示训练进度：

```
正在运行实验：baseline_cnn
数据集：seed_preprocessed (14 名被试, 5 折)

Fold 1/5  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  100%  epoch 100/100
  train_loss: 0.2341   val_accuracy: 0.7823   val_f1_score: 0.7801

Fold 2/5  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  72%   epoch 72/100
  train_loss: 0.2819   val_accuracy: 0.7654   val_f1_score: 0.7612
  [早停触发：val_accuracy 连续 15 轮未改善]
```

---

## 日志文件

实验运行时的完整输出写入项目目录的 `logs/` 下：

```
emotion_recognition/
└── logs/
    └── baseline_cnn/
        ├── stdout.log   # 训练过程中的所有标准输出（每 epoch 的指标）
        └── stderr.log   # 错误和警告信息
```

实时查看日志：

```bash
tail -f logs/baseline_cnn/stdout.log
```

---

## 检查点文件

框架根据 `logging.checkpoint_metric` 指定的指标，在每次指标改善时保存检查点：

```
emotion_recognition/
└── experiments/
    └── results/
        └── baseline_cnn/
            └── checkpoints/
                ├── fold_1_best.pt
                ├── fold_2_best.pt
                └── ...
```

加载检查点用于推理：

```python
import torch
checkpoint = torch.load("experiments/results/baseline_cnn/checkpoints/fold_1_best.pt")
model.load_state_dict(checkpoint["model_state_dict"])
```

---

## 实验状态

| 状态 | 含义 |
|------|------|
| `PENDING` | 已创建配置，尚未运行 |
| `RUNNING` | 正在运行中 |
| `COMPLETED` | 正常完成 |
| `FAILED` | 运行中出错 |

查看实验列表和状态：

```bash
uesf experiment list
```

---

## 运行失败时的调试

查看错误信息：

```bash
cat logs/baseline_cnn/stderr.log
```

常见错误类型：

| 错误信息 | 原因 | 解决方法 |
|----------|------|----------|
| `ComponentNotFoundError: emotion_cnn` | 模型名称找不到 | 检查 `project.yml` 中 `models` 块的键名是否与实验 YAML 中的 `name` 一致 |
| `EntrypointImportError: ./src/models/cnn.py` | 源文件导入失败 | 检查文件路径和 Python 语法；运行 `python src/models/cnn.py` 确认无导入错误 |
| `CUDA out of memory` | GPU 显存不足 | 减小 `batch_size`，或在配置中设置 `default_device: cpu` |
| `DatasetNotFoundError: seed_preprocessed` | 预处理数据集不存在 | 运行 `uesf data preprocessed list` 检查，确认数据集名称正确 |

---

## 重新运行实验

若需要保留配置但清除之前的结果重新运行：

```bash
# 只删除结果，保留配置
uesf experiment remove baseline_cnn --results-only

# 然后重新运行
uesf experiment run --exp baseline_cnn
```

---

## Weights & Biases 集成

在实验 YAML 的 `logging` 块中启用：

```yaml
logging:
  use_wandb: true
  checkpoint_metric: val_accuracy
```

确保已安装并登录 W&B：

```bash
pip install wandb
wandb login
```

---

## 下一步

实验完成后查询和比较结果：[查询和比较实验结果](08_query_results.md)
