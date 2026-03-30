# Sprint 2: 预处理 Pipeline + Preprocessed/Masked 数据集

**日期**: 2026-03-30
**状态**: 已完成

## 目标

实现三流预处理架构 (data/label/joint)、内置算子库、逐 subject 懒加载、预处理/Masked 数据集管理。

## 交付物

### 预处理算子 (`pipeline/operators/`)
- **Data 算子**: `resample`, `filter` (带通/高通/低通), `notch_filter`, `reference` (CAR)
- **Label 算子**: `smooth` (移动窗口众数)
- **Joint 算子**: `sliding_window` (滑窗切片，沿 recording 维度展开), `epoch_normalize` (Z-score/MinMax)
- 算子注册表 `OPERATOR_REGISTRY`，按名称查找

### Preprocessor (`pipeline/preprocessor.py`)
- 逐 subject 懒加载 .mat → data/label/joint 流水线 → 追加 .npy
- 严格失败策略：任何 subject 出错则中止并清理
- 配置快照存入 DB

### DataManager 扩展
- `list_preprocessed()`, `get_preprocessed()`, `remove_preprocessed()`: 预处理数据集 CRUD + 级联删除 masked
- `create_masked()`, `list_masked()`, `remove_masked()`: 标签重映射虚拟数据集，不复制特征数据

### CLI 扩展
- `uesf data preprocess run`: 执行预处理
- `uesf data preprocessed list/remove`: 管理预处理数据集
- `uesf data preprocessed mask`: 创建标签映射数据集

### 测试
- 26 个新测试（共 130 个），全部通过
- 覆盖：各算子单元测试、端到端预处理、masked 数据集创建/删除
