# 如何管理全局组件库

全局组件库允许你将常用的模型、训练器和评估指标注册为全局可用，在任意项目中无需重新配置入口点即可使用。

---

## 何时使用全局组件

| 场景 | 推荐 |
|------|------|
| 模型只在当前项目使用 | 项目级（`project.yml` entrypoint） |
| 同一个模型在多个项目中复用 | 全局组件库 |
| 稳定成熟、不再频繁修改的代码 | 全局组件库 |
| 开发阶段仍在频繁迭代的代码 | 项目级（自动检测变更，无需手动重注册） |

---

## 添加全局组件

```bash
# 注册全局模型
uesf model add emotion_cnn ./src/models/cnn.py:EmotionCNN \
  --description "1D CNN 情绪分类，适用于 SEED 等数据集"

# 注册全局训练器
uesf trainer add standard_trainer ./src/trainers/trainer.py:StandardTrainer

# 注册全局指标
uesf metric add balanced_acc ./src/metrics/balanced.py:balanced_accuracy \
  --description "类别均衡准确率"
```

注册后，源码内容会被复制并存储在 `~/.uesf/` 下。

---

## 在项目中使用全局组件

全局组件注册后，可在任意项目的实验 YAML 中直接按名称使用，无需在 `project.yml` 中声明入口点：

```yaml
# experiments/baseline.yml
model:
  name: emotion_cnn    # 全局库中的组件，直接使用名称

evaluation:
  metrics: [accuracy, balanced_acc]    # balanced_acc 是全局指标
```

若 `project.yml` 中有同名的项目级组件，项目级优先（遮蔽全局）。

---

## 查看全局组件

```bash
# 查看当前有效版本
uesf model list
uesf trainer list
uesf metric list

# 同时显示已归档的旧版本
uesf model list --show-obsolete
```

---

## 更新全局组件

修改源文件后，重新运行 `add` 命令：

```bash
uesf model add emotion_cnn ./src/models/cnn.py:EmotionCNN
```

框架会检测源码是否有变更：
- 有变更：旧版本归档为 `emotion_cnn_<sha256前8位>`，以原名称创建新版本
- 无变更：不做任何操作

---

## 修改组件描述

```bash
uesf model edit emotion_cnn --description "1D CNN，已加入 BatchNorm"
```

---

## 删除全局组件

```bash
uesf model remove emotion_cnn
uesf model remove emotion_cnn -y    # 跳过确认
```

删除的是元数据记录，不影响本地源文件。已使用此组件的历史实验记录仍然保留（数据库中有源码快照）。
