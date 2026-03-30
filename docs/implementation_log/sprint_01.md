# Sprint 1: 数据管理 - Raw 数据集

**日期**: 2026-03-30
**状态**: 已完成

## 目标

实现 Raw 数据集的完整生命周期管理：注册、导入、列表、删除、编辑、信息查看。

## 交付物

### DataManager (`managers/data_manager.py`)
- `register_raw()`: 解析 raw.yml → 验证 .mat 文件 → 推断 data_shape/label_shape → 校验跨 subject 一致性 → 存入 DB
- `import_raw()`: register + 拷贝 .mat 文件到 `<data_dir>/raw/<name>/`
- `list_raw()` / `get_raw()` / `edit_raw()` / `remove_raw()`: 完整 CRUD
- 删除级联：支持删除依赖的预处理数据集或标记为孤儿

### CLI (`cli/data_cmd.py`)
- `uesf data raw register/import/list/remove/edit/info` 全部实现
- Rich 表格输出、确认提示、错误面板

### 测试
- `tests/fixtures/fake_data.py`: 生成伪造 .mat 数据集的辅助函数
- 28 个新测试（共 104 个），全部通过
- 覆盖：正常注册/导入、shape 推断、一致性校验、缺失键检测、CLI E2E

## 关键实现细节

- `loadmat(squeeze_me=False)` 保留 singleton 维度，确保 shape 推断准确
- `numeric_to_semantic` 键统一转为字符串存储
- 缺少 `name` 字段时回退使用目录名
