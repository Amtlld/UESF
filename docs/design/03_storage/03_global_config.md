# 全局配置机制

UESF 的全局配置采用"数据库存储默认值 + 文件覆写"的双层机制。

数据库表结构详见 [`configs` 表](02_database_schema.md#configs-表)。

## 1. 配置项定义

当前版本仅允许以下四个全局配置键：

| 键名 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data_dir` | string | `~/.uesf/data` | UESF 管理的数据集统一存储目录 |
| `default_device` | string | `cpu` | 默认计算设备（如 `cpu`, `cuda:0`, `cuda:1`） |
| `num_workers` | int | `4` | DataLoader 的工作进程数 |
| `log_level` | string | `INFO` | 框架日志输出级别，可选 `DEBUG`, `INFO`, `WARNING`, `ERROR` |

## 2. 存储与优先级

- **数据库 `configs` 表**：存储系统的默认配置值。数据库中的全局配置**不可更改**，仅在系统初始化时写入默认值，作为基准参照
- **用户配置文件 `~/.uesf/config.yml`**：用户可通过创建此文件覆写全局设置。**`config.yml` 的优先级高于数据库表**

系统在读取全局配置时，先从数据库加载默认值，再用 `config.yml` 中的同名键覆盖。

`config.yml` 示例：
```yaml
data_dir: /data/eeg_datasets
default_device: "cuda:0"
num_workers: 8
log_level: DEBUG
```

> **未知键警告**
> 若 `config.yml` 中出现上述四个合法键以外的键名，系统在启动时抛出警告（Warning）提示用户该键不受支持，但不终止运行。

> 系统初始化时（参见 [数据库初始化](02_database_schema.md#21-存储位置与初始化)），`configs` 表将被写入上述四个键的默认值。此后该表内容保持只读，用户的个性化覆写通过 `config.yml` 完成。
