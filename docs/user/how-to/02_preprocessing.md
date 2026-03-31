# 如何配置和运行预处理流水线

本指南说明如何编写 `preprocess.yml`，配置三流预处理流水线，并将原始数据集转换为可供实验使用的预处理数据集。

---

## preprocess.yml 结构概览

```yaml
preprocess:
  source_dataset: seed_raw     # 输入的原始数据集名称
  out_name: seed_preprocessed  # 输出的预处理数据集名称

  pipeline:
    data:    # 数据流：只操作 EEG 信号，不改变维度结构
      - ...
    label:   # 标签流：只操作标签数组
      - ...
    joint:   # 联合流：同时操作信号和标签（如滑窗切片）
      - ...
```

### 为什么要分三流

`data`、`label`、`joint` 三流分开处理是为了保证信号和标签的严格对齐：

- 滤波、重采样等操作只改变信号内容，不改变样本数量，放在 `data` 流
- 标签平滑等操作只处理标签，放在 `label` 流
- 滑动窗口切片同时切分信号和标签，必须原子性地操作两者，放在 `joint` 流

三流的执行顺序固定：先执行 `data` 流中的所有算子，再执行 `label` 流，最后执行 `joint` 流。

---

## 数据流（data stream）算子

### filter — 带通/高通/低通滤波

去除低频基线漂移和高频噪声：

```yaml
- name: filter
  params:
    l_freq: 1.0    # 高通截止频率（Hz），去除低于此频率的成分
    h_freq: 40.0   # 低通截止频率（Hz），去除高于此频率的成分
```

只做高通滤波（去基线漂移）：

```yaml
- name: filter
  params:
    l_freq: 0.5
    h_freq: null
```

### notch_filter — 陷波滤波

专用于去除市电干扰（中国和欧洲用 50Hz，北美用 60Hz）：

```yaml
- name: notch_filter
  params:
    notch_freq: 50.0
```

### ica — 独立成分分析

去除眼电、心电等伪影。建议在带通滤波**之后**执行：

```yaml
- name: ica
  params:
    method: fastica       # ICA 算法，支持 "fastica"、"picard"
    n_components: 0.95    # 保留解释方差比例为 95% 的成分；也可以填整数（如 20）指定成分数
    exclude_eog_ecg: true # 自动识别并去除眼电/心电成分
```

### resample — 重采样

降低采样率以减少计算量：

```yaml
- name: resample
  params:
    target_rate: 128    # 目标采样率（Hz）
```

### reference — 重参考

将信号转换为公共平均参考（CAR）：

```yaml
- name: reference
  params:
    type: CAR
```

---

## 标签流（label stream）算子

### smooth — 标签平滑

对连续标签序列进行滑窗平滑，减少瞬时噪声标注的影响：

```yaml
- name: smooth
  params:
    window_size: 5    # 滑动窗口大小（样本数）
```

---

## 联合流（joint stream）算子

### sliding_window — 滑窗切片（Epoching）

将连续 EEG 信号切分为等长的 Epoch，是最常用的联合算子：

```yaml
- name: sliding_window
  params:
    window_size_sec: 4.0       # 窗口时长（秒）
    stride_sec: 2.0            # 滑动步长（秒），小于 window_size_sec 时有重叠
    window_type: rect          # 窗函数，支持 "rect"（矩形）、"hanning"（汉宁窗）
    label_strategy: mode       # 窗口内标签聚合策略："mode"（众数）或 "last"（末尾标签）
```

> 切片生成的新 Epoch 会在 `recording` 维度上展开。

### epoch_normalize — Epoch 内标准化

对每个 Epoch 独立进行标准化，是**唯一安全的预处理阶段标准化方式**：

```yaml
- name: epoch_normalize
  params:
    method: zscore    # 标准化方法："zscore" 或 "minmax"
    axis: -1          # 沿哪个轴计算，-1 表示时间轴
```

> **关于跨被试全局标准化**：预处理阶段**禁止**做跨被试的全局 Z-Score（这会导致数据泄露）。全局标准化应在实验配置的 `transforms` 中使用 `zscore_normalize`，配合 `fit_on: train` 严格执行 Fit-on-Train 原则。详见[数据泄露防护机制](../concepts/02_data_leakage_prevention.md)。

---

## 完整的 preprocess.yml 示例

SEED 数据集的典型预处理配置（带通滤波 → 陷波 → ICA → 重采样 → 滑窗切片）：

```yaml
preprocess:
  source_dataset: seed_raw
  out_name: seed_preprocessed

  pipeline:
    data:
      - name: filter
        params: { l_freq: 1.0, h_freq: 40.0 }
      - name: notch_filter
        params: { notch_freq: 50.0 }
      - name: ica
        params:
          method: fastica
          n_components: 0.95
          exclude_eog_ecg: true
      - name: resample
        params: { target_rate: 128 }
    joint:
      - name: sliding_window
        params:
          window_size_sec: 4.0
          stride_sec: 2.0
          label_strategy: mode
      - name: epoch_normalize
        params: { method: zscore, axis: -1 }
```

---

## 指定输入数据集的三种方式

优先级由高到低：

1. **CLI 参数**（最高优先级）：`uesf data preprocess run --dataset seed_raw --out-name seed_preprocessed`
2. **preprocess.yml 中的 `source_dataset` 字段**
3. **project.yml 推断**：若当前目录有 `project.yml` 且 `raw_datasets` 列表只包含一个数据集，自动使用该数据集

---

## 运行预处理

```bash
# 指定配置文件
uesf data preprocess run -c preprocess.yml

# 或通过 CLI 参数覆盖输入/输出名称
uesf data preprocess run -c preprocess.yml --dataset seed_raw --out-name seed_preprocessed
```

预处理采用**按被试懒加载**：每次只将一个被试的数据读入内存，处理完后立即写入磁盘并释放内存。这确保在处理大数据集时不会 OOM。

---

## 查看预处理结果

```bash
uesf data preprocessed list
```

输出示例：

```
┌────────────────────┬──────────┬─────────────────┬────────────┐
│ 名称               │ 来源     │ Epoch 数        │ 占用空间   │
├────────────────────┼──────────┼─────────────────┼────────────┤
│ seed_preprocessed  │ seed_raw │ 5040            │ 1.2 GB     │
└────────────────────┴──────────┴─────────────────┴────────────┘
```

---

## 错误处理

预处理采用"严格失败"策略：任何一个被试文件出错（文件损坏、键名不匹配、维度不一致）都会**立即终止整个流程**，并清理已产生的中间文件。

常见错误：

- `KeyError: 'data'`：`.mat` 文件中不存在 `eeg_data_key` 指定的键名，检查 `raw.yml` 中的键名是否正确
- `DimensionMismatchError`：不同被试的数据维度不一致，检查数据集
- `DatasetNotFoundError: seed_raw`：指定的原始数据集未注册，先运行 `uesf data raw register`

---

## 下一步

预处理完成后：

- 初始化项目并编写自定义模型：[编写自定义模型](03_write_custom_model.md)
- 直接配置实验：[配置实验 YAML](06_configure_experiment.md)
