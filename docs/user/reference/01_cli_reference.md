# CLI 命令参考手册

UESF 命令行工具，主命令为 `uesf`，下设 7 个子命令组。

> CLI 参数与 YAML 配置发生冲突时，CLI 参数优先。

---

## uesf config

### uesf config show

显示当前生效的全局配置（数据库默认值与 `~/.uesf/config.yml` 覆写后的合并结果）。

```bash
uesf config show
```

### uesf config set

将指定键值写入 `~/.uesf/config.yml`。

```bash
uesf config set <KEY> <VALUE>
```

| 参数 | 说明 |
|------|------|
| `KEY` | 合法键名：`data_dir`、`default_device`、`num_workers`、`log_level` |
| `VALUE` | 对应的值 |

示例：

```bash
uesf config set data_dir ~/eeg_data
uesf config set default_device cuda:0
uesf config set num_workers 8
uesf config set log_level DEBUG
```

---

## uesf data raw

### uesf data raw register

将用户管理的原始数据集注册到 UESF（不移动数据文件）。

```bash
uesf data raw register <PATH>
```

| 参数 | 说明 |
|------|------|
| `PATH` | 数据集目录路径，目录下须包含 `raw.yml` 和 `.mat` 文件 |

示例：

```bash
uesf data raw register /data/seed_dataset/
```

### uesf data raw import

将原始数据集复制到 `data_dir` 并由 UESF 管理。

```bash
uesf data raw import <PATH>
```

示例：

```bash
uesf data raw import /data/seed_dataset/
```

### uesf data raw list

列出所有已注册的原始数据集。

```bash
uesf data raw list
```

### uesf data raw info

显示指定原始数据集的详细信息。

```bash
uesf data raw info <NAME>
```

示例：

```bash
uesf data raw info seed_raw
```

### uesf data raw edit

修改原始数据集的元信息。

```bash
uesf data raw edit <NAME> [--description TEXT] [--sampling-rate FLOAT]
```

| 选项 | 说明 |
|------|------|
| `--description` | 更新描述文字 |
| `--sampling-rate` | 更新采样率 |

示例：

```bash
uesf data raw edit seed_raw --description "SEED 数据集，已清理异常被试"
```

### uesf data raw remove

删除指定原始数据集。

```bash
uesf data raw remove <NAME> [--delete-preprocessed] [--yes]
```

| 选项 | 说明 |
|------|------|
| `--delete-preprocessed` | 同时删除依赖此数据集的预处理数据集 |
| `--yes`, `-y` | 跳过确认提示 |

示例：

```bash
uesf data raw remove seed_raw
uesf data raw remove seed_raw --delete-preprocessed -y
```

---

## uesf data preprocess

### uesf data preprocess run

根据预处理配置执行预处理流水线。

```bash
uesf data preprocess run [-c CONFIG_PATH] [--dataset NAME] [--out-name NAME]
```

| 选项 | 说明 |
|------|------|
| `-c`, `--config-path` | 预处理配置文件路径（默认当前目录的 `preprocess.yml`） |
| `--dataset` | 输入原始数据集名称（覆盖配置文件中的 `source_dataset`） |
| `--out-name` | 输出预处理数据集名称（覆盖配置文件中的 `out_name`） |

示例：

```bash
uesf data preprocess run -c preprocess.yml
uesf data preprocess run --dataset seed_raw --out-name seed_preprocessed
```

---

## uesf data preprocessed

### uesf data preprocessed list

列出所有预处理数据集（包括标签重映射数据集）。

```bash
uesf data preprocessed list
```

### uesf data preprocessed remove

删除指定预处理数据集。

```bash
uesf data preprocessed remove <NAME> [--yes]
```

### uesf data preprocessed mask

基于现有预处理数据集创建标签重映射数据集。

```bash
uesf data preprocessed mask <SOURCE> --out-name <NAME> --mapping-file <FILE>
```

| 参数/选项 | 说明 |
|-----------|------|
| `SOURCE` | 源预处理数据集名称 |
| `--out-name` | 新数据集名称 |
| `--mapping-file` | 标签映射规则文件（YAML 格式，旧语义标签 → 新语义标签） |

示例：

```bash
uesf data preprocessed mask seed_preprocessed --out-name seed_binary --mapping-file rule.yml
```

映射文件示例（`rule.yml`）：

```yaml
negative: negative
neutral: positive
positive: positive
```

---

## uesf model / uesf trainer / uesf metric

这三组命令结构相同，以 `uesf model` 为例说明。

### add

注册组件到全局组件库。

```bash
uesf model add <NAME> <SOURCE> [--description TEXT]
```

| 参数 | 说明 |
|------|------|
| `NAME` | 在全局库中的名称 |
| `SOURCE` | 源文件路径和类名，格式：`./path/to/file.py:ClassName` |
| `--description` | 可选描述 |

示例：

```bash
uesf model add emotion_cnn ./src/models/cnn.py:EmotionCNN --description "1D CNN 情绪分类"
uesf trainer add emotion_trainer ./src/trainers/trainer.py:EmotionTrainer
uesf metric add balanced_acc ./src/metrics/balanced.py:balanced_accuracy
```

### list

列出已注册的全局组件。

```bash
uesf model list [--show-obsolete]
```

| 选项 | 说明 |
|------|------|
| `--show-obsolete` | 同时显示已被归档的旧版本 |

### remove

删除指定组件。

```bash
uesf model remove <NAME> [--yes]
```

### edit

修改组件的描述信息。

```bash
uesf model edit <NAME> [--description TEXT]
```

---

## uesf project

### uesf project init

在当前目录初始化 UESF 项目，生成 `project.yml` 和 `experiments/` 目录。

```bash
uesf project init [PATH]
```

| 参数 | 说明 |
|------|------|
| `PATH` | 目标目录（默认为当前目录） |

示例：

```bash
uesf project init
uesf project init /path/to/my_project/
```

### uesf project info

显示当前项目的状态，包括可用组件列表和数据集。

```bash
uesf project info [--project-dir PATH]
```

---

## uesf experiment

以下命令默认在当前项目目录执行，可通过 `--project-dir` 指定其他路径。

### uesf experiment add

创建新的实验配置文件。

```bash
uesf experiment add [--name NAME] [--from BASE_EXP] [--description TEXT] [--project-dir PATH]
```

| 选项 | 说明 |
|------|------|
| `--name` | 实验名称（未指定时自动生成） |
| `--from` | 从现有实验复制配置 |
| `--description` | 实验描述 |

示例：

```bash
uesf experiment add --name baseline_cnn
uesf experiment add --name deeper_cnn --from baseline_cnn
```

### uesf experiment list

列出当前项目下所有实验及其状态。

```bash
uesf experiment list [--project-dir PATH]
```

### uesf experiment run

执行指定实验。

```bash
uesf experiment run --exp <NAME> [--project-dir PATH]
```

示例：

```bash
uesf experiment run --exp baseline_cnn
```

### uesf experiment remove

删除实验。

```bash
uesf experiment remove <NAME> [--results-only] [--yes] [--project-dir PATH]
```

| 选项 | 说明 |
|------|------|
| `--results-only` | 只删除实验结果（检查点等），保留配置文件 |
| `--yes`, `-y` | 跳过确认提示 |

示例：

```bash
uesf experiment remove baseline_cnn --results-only  # 保留配置，清除结果
uesf experiment remove baseline_cnn -y              # 删除配置和结果
```

### uesf experiment query

查询并对比已完成实验的指标结果。

```bash
uesf experiment query [--metrics METRICS] [--status STATUS] [--project-dir PATH]
```

| 选项 | 说明 |
|------|------|
| `--metrics` | 逗号分隔的指标名称列表，如 `accuracy,f1_score` |
| `--status` | 按状态筛选：`PENDING`、`RUNNING`、`COMPLETED`、`FAILED` |

示例：

```bash
uesf experiment query --metrics accuracy,f1_score,auroc --status COMPLETED
```
