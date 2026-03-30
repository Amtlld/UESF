# Sprint 4: Project Manager + Metric Manager

## 目标

实现项目初始化与加载、三级组件名称解析、内置评估指标、自定义指标管理。

## 完成内容

### ProjectManager

- **project_manager.py** (`src/uesf/managers/project_manager.py`)
  - `init(project_dir)` — 创建标准项目结构（project.yml + experiments/ 目录）
  - `load(project_dir)` — 解析 project.yml，验证必需字段 `project-name`
  - `info(project_dir)` — 返回项目摘要（名称、数据集、模型、训练器、指标列表）
  - `resolve_component(name, type, config, project_dir)` — 三级优先级解析：
    1. 项目级（project.yml 中定义）
    2. 全局级（数据库中 GLOBAL 类型）
    3. 内置级（EMBEDDED）
  - 名称遮蔽时输出 Warning 日志

### 内置评估指标

- **builtin_metrics.py** (`src/uesf/components/builtin_metrics.py`)
  - 6 个内置指标：accuracy, f1_score, precision, recall, auroc, confusion_matrix
  - 统一签名：`(preds: Tensor, targets: Tensor, **kwargs) -> float | dict`
  - 支持 logits 输入（自动 argmax）
  - F1/Precision/Recall 支持 macro/micro/weighted 三种 average 模式
  - AUROC 支持二分类和多分类（macro-averaged OVR）
  - confusion_matrix 返回 JSON 可序列化的 dict

### MetricManager

- **metric_manager.py** (`src/uesf/managers/metric_manager.py`)
  - 与 ModelManager/TrainerManager 完全相同的 CRUD 模式
  - `load_metric(name)` — 三级加载：entrypoint → 数据库 → 内置
  - SHA256 自动重注册机制

### CLI 命令

- `uesf project init/info` (`src/uesf/cli/project_cmd.py`)
- `uesf metric add/list/remove/edit` (`src/uesf/cli/metric_cmd.py`)
- 在 `app.py` 中注册 project_app 和 metric_app

### 测试

| 测试文件 | 测试数 | 覆盖内容 |
|----------|--------|----------|
| `tests/managers/test_project_manager.py` | 13 | init、load（正常/缺失/无效YAML/缺少字段）、info、resolve（项目级/全局/未找到/遮蔽/跨类型） |
| `tests/managers/test_metric_manager.py` | 13 | CRUD、注册/重注册、load_metric（内置/entrypoint/未找到） |
| `tests/components/test_builtin_metrics.py` | 18 | 各指标正确性、logits 输入、多种 average 模式、注册表完整性 |
| `tests/cli/test_project_cmd.py` | 4 | CLI E2E |
| `tests/cli/test_metric_cmd.py` | 8 | CLI E2E |

**总测试数：282（含 Sprint 0-3 的 219 个）**

## 验收

- [x] 282 个测试全部通过
- [x] ruff check 无错误
- [x] CLI 命令已注册并可用
