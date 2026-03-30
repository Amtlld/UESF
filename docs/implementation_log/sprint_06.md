# Sprint 6: 实验 CLI + E2E 集成测试

## 目标

实现实验 CLI 命令、DummyModel/DummyTrainer、完整 E2E 工作流测试。

## 完成内容

### 实验 CLI 命令

- **experiment_cmd.py** (`src/uesf/cli/experiment_cmd.py`)
  - `uesf experiment add` — 创建实验配置（空白或从现有复制），支持 --name, --from, --description
  - `uesf experiment list` — 列出项目下所有实验及状态
  - `uesf experiment remove` — 删除实验或仅删除结果（--results-only）
  - `uesf experiment run --exp <name>` — 执行实验，结束后 Rich 表格展示结果
  - `uesf experiment query` — 查询实验结果，--metrics 指定关注指标，--status 过滤

### DummyModel & DummyTrainer

- **dummy.py** (`src/uesf/components/dummy.py`)
  - `DummyModel(BaseModel)` — 全连接网络，用于测试和 EMBEDDED 默认
  - `DummyTrainer(BaseTrainer)` — 交叉熵训练器，支持多通道 batch

### E2E 集成测试

| 测试文件 | 测试数 | 覆盖内容 |
|----------|--------|----------|
| `tests/e2e/test_full_workflow.py` | 4 | 完整 holdout 流程、K-Fold 流程、实验 add/list、失败状态记录 |
| `tests/cli/test_experiment_cmd.py` | 4 | CLI add/list/remove E2E |

**总测试数：329（含 Sprint 0-5 的 321 个）**

### E2E 测试覆盖的完整流程

1. 创建伪预处理数据集（.npy + DB 记录）
2. 初始化项目（project.yml + model/trainer 入口点）
3. 创建实验配置 YAML
4. 运行实验：
   - 组件初始化（模型/训练器/指标）
   - 数据加载 + 切分（Holdout/K-Fold）
   - 在线变换（ZScore fit-on-train）
   - 训练循环（多 epoch）
   - 验证评估（epoch 级聚合）
   - 结果写入数据库
5. 查询验证实验结果和状态

## 验收

- [x] 329 个测试全部通过
- [x] ruff check 无错误
- [x] CLI 命令已注册并可用
- [x] E2E 测试验证完整流程
