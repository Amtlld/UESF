# UESF 架构与数据流

本文解释 UESF 各层组件之间的关系，以及数据从原始文件到模型输入的完整流动路径。

---

## 分层架构

```
┌─────────────────────────────────────────────────┐
│                   用户界面层                      │
│   CLI（uesf命令）  ←→  YAML 配置文件              │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│                   管理器层（Managers）             │
│  DataManager  ModelManager  ExperimentManager    │
│  TrainerManager  MetricManager  ProjectManager   │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│                   实验引擎层                      │
│  Splitter  Transforms  DataLoader  Runner        │
│  Evaluator                                       │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│                   持久化层                        │
│  SQLite（元数据）    文件系统（.npy / .pt）        │
└─────────────────────────────────────────────────┘
```

**CLI 和 YAML** 是用户的操作界面，所有操作通过命令或配置文件表达，不需要编写框架调用代码。

**管理器层** 处理元数据的增删改查：注册数据集、查找组件、管理实验记录。

**实验引擎层** 负责运行时执行：数据切分、在线变换、训练循环、指标计算。

**持久化层** 存储两类信息：元数据和快照存在 SQLite，实际数据文件（`.npy` 特征、`.pt` 模型权重）存在文件系统。

---

## 数据的三种形态

UESF 管理三类数据集对象，每种在系统中的角色不同：

```
原始数据集（Raw Dataset）
  ├── .mat 文件（每个被试一个）
  └── raw.yml（元数据描述）
         │
         │  uesf data preprocess run
         ▼
预处理数据集（Preprocessed Dataset）
  ├── features.npy（全体被试的特征张量，形状 [N, C, T]）
  └── labels.npy（全体被试的标签数组，形状 [N]）
         │
         │  uesf data preprocessed mask
         ▼
标签重映射数据集（Masked Dataset）
  ├── labels.npy（重新编号的标签数组，形状 [N]）
  └── 特征：直接引用源预处理数据集的 features.npy（不复制）
```

标签重映射数据集不复制特征数据，通过引用视图（View）共享特征张量，存储开销极低。

---

## 实验执行数据流

`uesf experiment run` 执行时，数据经过以下 5 个阶段：

```
阶段 1：数据加载（Data Loading）
  └── 读取 features.npy 和 labels.npy
      Masked Dataset 在此应用标签视图映射

阶段 2：索引切分（Splitter）
  └── 按配置的 strategy 和 dimension 生成 train/val/test 索引
      不复制数据，只记录索引快照

阶段 3：在线变换（Online Transforms）
  └── zscore_normalize：
      - fit_on: train  → 只从 train 索引对应的样本计算 μ, σ
      - apply_to: all  → 用同一组 μ, σ 变换所有样本

阶段 4：DataLoader 构建（Multi-channel Dataloading）
  └── 根据 dataloaders 配置，将切分相（.train/.val/.test）
      组装为多通道字典 DataLoader

阶段 5：训练与评估（Runner + Trainer + Evaluator）
  └── Runner 驱动训练循环
      → 将 batch 字典传给 Trainer.training_step（Trainer 全权负责梯度）
      → Trainer.validation_step 返回 preds/targets
      → Evaluator 在 epoch 结束时聚合，计算指标
```

---

## 元数据与物理文件的分离

UESF 将**元数据**（谁注册了什么，形状是多少，来自哪里）与**物理文件**（实际的数组数据）分开存储：

- **SQLite 数据库**（`~/.uesf/uesf.db`）：存储所有元数据、快照和实验结果
- **文件系统**（`<data_dir>/`）：存储物理数据文件

这种分离的好处：
1. 列出所有数据集、查询实验结果等元数据操作无需读取大型数据文件，速度极快
2. 数据集可以仅注册（`register`）而不导入，物理文件保留在原位
3. 数据库损坏不会影响物理文件，反之亦然

---

## YAML 与 JSON 的角色

**YAML** 是用户的输入界面：配置可读性高，支持注释，用于描述意图（"我想用 Adam，lr=0.001"）。

**JSON** 是系统的内部记录格式：每次操作的输入/输出 YAML 在执行前会被解析为 JSON，并存储在 SQLite 中作为实验快照。这确保即使后来修改了配置文件，历史实验仍然有完整的配置记录可以追溯。
