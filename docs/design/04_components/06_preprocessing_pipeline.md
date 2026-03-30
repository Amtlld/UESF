# Preprocessing Pipeline 详细设计

本文档详细说明了 UESF 数据预处理器 (Data Preprocessor) 中 `pipeline` 模块的具体设计规范、内存优化策略以及内置算子库。

## 1. Pipeline 执行架构设计

在 EEG 深度学习研究中，预处理模块不仅负责信号清洗，还需处理好向训练张量的转换。为了解决复杂流处理和潜在的数据泄露风险，UESF 的预处理 Pipeline 遵循以下设计准则：

1. **有序列表执行**：所有预处理步骤采用 `List`（列表）配置，而非原先的字典，确保完全按照声明的先后顺序执行，并允许重复调用同一模块（如多次滤波）。
2. **三线分流**：
   - `data` 流：仅对信号时序或通道进行变换。
   - `label` 流：对标签的变换。
   - `joint` 流：同时操作数据和标签（如滑窗切片 Epoching），以保证时序和维度的强对齐。

### 1.1 内存优化：基于被试的懒加载 (Subject-wise Lazy Evaluation)

为防止大采样率下将数十名被试数据一次性读入引发内存溢出 (OOM)，Pipeline 底层执行将采用流式架构：
1. **按被试循环**：每次仅将 **1 个**被试的原始 `.mat` 数据读入内存。
2. **流水线推演**：该被试的数据顺次通过 `data`、`label` 和 `joint` pipeline。
3. **追加持久化**：将处理完毕的分块数据以 `np.memmap` 形式追加 (Append) 到统一的 `<out_name>.npy` 中，并立即释放内存资源。

## 2. YAML 配置结构规范

在 `preprocess.yml` 中，`pipeline` 的标准定义结构应如下所示：

```yaml
preprocess:
  source_dataset: "<raw-dataset-name>"
  out_name: "<preprocessed-dataset-name>"
  pipeline:
    # 1. 数据处理流 (不改变张量阶数)
    data:
      - name: filter
        params: { l_freq: 1.0, h_freq: 45.0 }
      - name: notch_filter
        params: { notch_freq: 50.0 }
      - name: ica  # ICA 去伪迹建议在带通滤波后执行
        params: { method: "fastica", n_components: 0.95, exclude_eog_ecg: true }
      - name: resample
        params: { target_rate: 200 }
        
    # 2. 标签处理流
    label:
      - name: smooth 
        params: { window_size: 5 }
        
    # 3. 联合处理流 (关键！同时修改特征和标签的 shape 以保持机制对齐)
    joint:
      - name: sliding_window  # 滑窗切片 (沿着 recording 维度展开)
        params: 
          window_size_sec: 4.0
          stride_sec: 2.0
          window_type: "rect" # 窗函数类型，支持 "rect" (矩形窗), "hanning" (汉宁窗) 等
          label_strategy: "mode" # 窗口内标签特征聚合策略 (mode/last)
```

## 3. 内置标准算子库 (Built-in Modules)

UESF 利用底层科学计算库（如 `MNE-Python`, `SciPy`）封装了标准化的预处理模块。

> **⚠️ 注意：关于全局标准化 (Normalize)**
> 为避免数据泄露（Data Leakage），UESF **禁止**在预处理阶段采用跨所有被试或跨录制段的全局 `Z-Score`。
> 全局的标准化计算必须被推迟到 `Experiment Manager` 的 `transforms`（在线变换阶段）进行（严格执行 Fit on Train 原则）。预处理阶段若需标准化，仅允许采用按 Epoch 独立的局部标准化（如 `epoch_normalize`）。

### 3.1 核心 Data 算子 (`pipeline.data`)

| 算子名称 (`name`) | 说明 | 核心参数 (`params`) |
| :--- | :--- | :--- |
| `resample` | **频率重采样**。降低数据存储和计算开销。 | `target_rate`: 目标采样率 (Hz) |
| `filter` | **带通/高通/低通滤波**。去除低频基线漂移和高频环境噪声。 | `l_freq`, `h_freq`: 截断频率 (Hz) |
| `notch_filter` | **陷波滤波**。专用于剔除 50Hz/60Hz 市电工频干扰。 | `notch_freq`: 陷波频率 |
| `ica` | **独立成分分析 (ICA)**。<br>无损地分离并滤除眼电/肌电/心电等伪影。 | `n_components`: 成分数量或解释方差比值<br>`method`: ICA算法 ("fastica", "picard")<br>`exclude_eog_ecg`: (bool) 自动剔除眼/心电 |
| `reference` | **重参考**。如转换为公共平均参考 (CAR)。 | `type`: "CAR", "mastoid" 等 |

### 3.2 标签 Label 算子 (`pipeline.label`)

| 算子名称 (`name`) | 说明 | 核心参数 (`params`) |
| :--- | :--- | :--- |
| `smooth` | **标签平滑**。针对连续标签进行基础消噪。 | `window_size` |

### 3.3 联合 Joint 算子 (`pipeline.joint`)

| 算子名称 (`name`) | 说明 | 核心参数 (`params`) |
| :--- | :--- | :--- |
| `sliding_window` | **滑窗切片 (Epoching)**。<br>将连续片段切割为供模型消费的标准 Batch，**切片生成的新片段将在 `recording` 维度上展开**。 | `window_size_sec`: 窗口截断长(秒)<br>`stride_sec`: 移动步长(秒)<br>`window_type`: 窗函数("rect","hanning"等)<br>`label_strategy`: "mode"/"last" |
| `epoch_normalize` | **Epoch 内部独立标准化**。<br>唯一安全的无泄露标准化动作，按计算生成的最小切片窗口做内部的相对缩放。 | `method`: "zscore"/"minmax", `axis`: -1 |
