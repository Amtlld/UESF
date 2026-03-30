# Sprint 0: 项目脚手架与核心基础设施

**日期**: 2026-03-30
**状态**: 已完成

## 目标

建立 Python 包骨架、构建系统和三个基础核心模块（异常、日志、数据库）+ 全局配置管理器 + CLI 入口。

## 交付物

### 包结构
- `pyproject.toml`: uv/hatchling 构建配置，声明所有依赖，入口点 `uesf`
- `src/uesf/`: 包骨架，含 `core/`, `cli/`, `managers/`, `pipeline/`, `experiment/`, `components/` 子包

### 核心模块
1. **异常体系** (`core/exceptions.py`): 完整的 `UESFException` 层次结构，5 大类 12 个具体异常，每个异常携带 `(message, context, hint)` 三元组
2. **日志系统** (`core/logging.py`): `uesf.*` 命名空间，ConsoleHandler (Rich) + GlobalFileHandler (RotatingFile)，`setup_logging()` / `get_logger()` 工厂
3. **数据库管理** (`core/database.py`): `DatabaseManager` 类，9 张表 DDL（raw_datasets, preprocessed_datasets, masked_datasets, trainers, models, metrics, experiments, configs, schema_versions），事务上下文管理器，默认配置种子数据
4. **全局配置** (`core/config.py`): `ConfigManager` 双层机制（DB 默认值 + config.yml 覆盖），4 个配置项，未知键警告

### CLI
- `uesf --version`: 版本输出
- `uesf config show`: 展示合并后的全局配置
- `uesf config set <KEY> <VALUE>`: 写入 config.yml

### 测试
- 76 个测试全部通过
- 覆盖：异常实例化与继承链、数据库建表与事务、配置 CRUD、CLI E2E
- ruff 检查无错误

## 关键设计决策

- `DatabaseManager` 支持 `:memory:` 参数，测试无需磁盘 I/O
- `UESF_HOME` 环境变量覆盖 `~/.uesf`，实现测试完全隔离
- DDL 以 Python 字符串内嵌，零外部文件依赖
- 使用 `sqlite3` 直接操作，不引入 ORM
