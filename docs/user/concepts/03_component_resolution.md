# 三级组件解析优先级

UESF 中的模型、训练器和评估指标通过名称引用。当你在实验 YAML 中写 `model.name: emotion_cnn` 时，框架按三级优先级查找这个名称对应的实现。

---

## 三级优先级

```
项目级（REGISTERED）> 全局级（GLOBAL）> 内置（EMBEDDED）
```

### 内置（EMBEDDED）

框架自带的组件，无需任何注册即可使用：

- 模型：`dummy`（测试用随机模型）
- 训练器：`dummy`（配合 DummyModel 的测试训练器）
- 指标：`accuracy`、`f1_score`、`precision`、`recall`、`auroc`、`confusion_matrix`
- 优化器：`adam`、`adamw`、`sgd` 等（透传给 PyTorch）
- 调度器：`cosine_annealing_lr`、`step_lr` 等（透传给 PyTorch）

### 全局级（GLOBAL）

通过 `uesf model add` 等命令注册的组件，源码被复制并存储在 `~/.uesf/` 下，跨项目共享：

```bash
uesf model add emotion_cnn ./src/models/cnn.py:EmotionCNN
```

全局组件在任何项目中都可以通过名称 `emotion_cnn` 直接使用，无需在 `project.yml` 中配置入口点。

### 项目级（REGISTERED）

在 `project.yml` 的 `models`/`trainers`/`metrics` 块中通过 `entrypoint` 注册的组件。源码保留在项目目录中，运行时动态加载：

```yaml
models:
  emotion_cnn:
    entrypoint: "./src/models/cnn.py:EmotionCNN"
```

项目级组件优先级最高，意味着项目目录下的组件定义可以**遮蔽（shadow）**全局库和内置中的同名组件。

---

## 名称遮蔽（Shadowing）

若项目级组件与全局库或内置组件同名，框架使用项目级版本。例如：

```yaml
# project.yml
models:
  accuracy:          # 与内置指标同名，但这是一个模型，不会冲突
    entrypoint: ...
metrics:
  f1_score:          # 与内置指标 f1_score 同名
    entrypoint: "./src/metrics/custom_f1.py:my_f1"
```

第二个例子中，实验 YAML 中的 `f1_score` 将使用 `my_f1` 函数，而不是内置的 F1 实现。这是有意的设计，让你可以替换内置指标的实现，但需要注意命名冲突带来的意外遮蔽。

> **最佳实践**：自定义组件使用独特的名称（如 `balanced_acc`），而不是与内置组件同名，除非明确需要替换内置实现。

---

## REGISTERED vs. GLOBAL：何时使用哪种

| 场景 | 推荐方式 |
|------|----------|
| 模型只在当前项目使用，代码在项目目录下 | 项目级（`project.yml` entrypoint） |
| 同一个模型要在多个项目中复用 | 全局级（`uesf model add`） |
| 开发阶段频繁修改模型代码 | 项目级（每次 `run` 自动检测变更，无需手动重注册） |
| 稳定的、经过验证的模型，不再修改 | 全局级（源码快照固化，保证稳定性） |

---

## 源码变更检测

**项目级（REGISTERED）**组件在每次 `uesf experiment run` 时会自动检测源文件的 SHA256 哈希：

- 若哈希未变化：使用已记录的版本
- 若哈希变化：将旧版本归档为 `emotion_cnn_<sha256前8位>`（标记为 `obsolete`），以原名称 `emotion_cnn` 创建新版本，日志中输出提示

**全局级（GLOBAL）**组件在注册时已将源码快照存储在数据库中，后续源文件变更**不会**触发自动检测。若要更新全局组件，需要重新运行 `uesf model add`（旧版本会被归档）。

---

## 查看过时（obsolete）版本

```bash
uesf model list --show-obsolete
```

过时版本是完整的历史记录，每个版本都有对应的源码快照，确保任何历史实验都可以追溯到当时使用的确切代码。
