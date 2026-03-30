# Sprint 3: 组件基类 + Model/Trainer Manager

## 目标

实现 BaseModel、BaseTrainer 基类，内置优化器/调度器映射，以及 Model/Trainer 的三类组件管理（EMBEDDED/REGISTERED/GLOBAL）。

## 完成内容

### 组件基类

- **BaseModel** (`src/uesf/components/base_model.py`)
  - 继承 `nn.Module`，自动存储 `n_channels`, `n_samples`, `n_classes`, `electrode_list`
  - 抽象 `forward()` 方法，可选 `extract_features()` 钩子
  - 通过 `**kwargs` 支持自定义参数透传

- **BaseTrainer** (`src/uesf/components/base_trainer.py`)
  - 拥有完整优化循环：前向传播、损失计算、梯度管理、优化器步进
  - 自动将 model 移动到指定 device
  - `configure_optimizers()` 默认返回 None（使用 YAML 配置），子类可覆盖
  - 抽象 `training_step(batch, batch_idx, optimizer)` 和 `validation_step(batch, batch_idx)`
  - batch 格式：多通道字典 `{channel_name: (data, labels)}`

### 内置映射

- **builtin_mappings.py** (`src/uesf/components/builtin_mappings.py`)
  - 8 个优化器映射：sgd, adam, adamw, adagrad, adadelta, rmsprop, radam, nadam
  - 8 个调度器映射：step_lr, multi_step_lr, exponential_lr, linear_lr, cosine_annealing_lr, cosine_annealing_warm_restarts, reduce_lr_on_plateau, one_cycle_lr
  - Transparent Passthrough 设计：YAML 参数直接 `**kwargs` 传递给 PyTorch 构造函数
  - `resolve_optimizer()` 和 `resolve_scheduler()` 查找函数

### 组件管理器

- **ModelManager** (`src/uesf/managers/model_manager.py`)
  - `add_global(source_path, name)` — 复制文件到 `~/.uesf/models/`，存储源码快照
  - `register(name, entrypoint, project_dir)` — 项目级注册
  - `detect_and_reregister(name, entrypoint, project_dir)` — SHA256 比较，变更时归档旧版本（`<name>_<hash[:8]>`，`is_obsolete=1`）
  - `load_class(name, entrypoint)` — 动态加载 Python 类
  - `list(show_obsolete)`, `get(name)`, `remove(name)`, `edit(name, description)`
  - 共享工具函数：`_parse_entrypoint()`, `_import_class()`

- **TrainerManager** (`src/uesf/managers/trainer_manager.py`)
  - 与 ModelManager 完全相同的接口模式
  - 复用 `_parse_entrypoint()` 和 `_import_class()`

### CLI 命令

- `uesf model add/list/remove/edit` (`src/uesf/cli/model_cmd.py`)
- `uesf trainer add/list/remove/edit` (`src/uesf/cli/trainer_cmd.py`)
- 在 `app.py` 中注册 model_app 和 trainer_app

### 测试

| 测试文件 | 测试数 | 覆盖内容 |
|----------|--------|----------|
| `tests/components/test_base_model.py` | 8 | 维度存储、forward、extract_features、nn.Module 继承 |
| `tests/components/test_base_trainer.py` | 7 | device 移动、config 存储、训练/验证步骤、自定义优化器 |
| `tests/components/test_builtin_mappings.py` | 12 | 映射完整性、resolve 成功/失败、transparent passthrough |
| `tests/managers/test_model_manager.py` | 18 | CRUD、注册、SHA256 重注册、动态加载、entrypoint 解析 |
| `tests/managers/test_trainer_manager.py` | 16 | 同上模式 |
| `tests/cli/test_model_cmd.py` | 8 | CLI E2E |
| `tests/cli/test_trainer_cmd.py` | 8 | CLI E2E |

**总测试数：219（含 Sprint 0-2 的 142 个）**

## 设计决策

1. **动态加载机制**：使用 `importlib.util.spec_from_file_location` 而非 `importlib.import_module`，避免污染 `sys.modules`
2. **entrypoint 格式**：统一使用 `path/to/file.py:ClassName`，路径相对于项目目录解析
3. **SHA256 自动重注册**：仅对 REGISTERED 类型组件生效，GLOBAL 和 EMBEDDED 不受影响
4. **归档命名**：旧版本重命名为 `<name>_<sha256[:8]>`，保留完整历史

## 验收

- [x] 219 个测试全部通过
- [x] ruff check 无错误
- [x] CLI 命令已注册并可用
