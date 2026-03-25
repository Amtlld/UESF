# Design of UESF

UESF（Unified EEG Study Framework）被设计为一个CLI工具，用于实现EEG数据的预处理、特征提取、模型训练和评估等功能，供研究人员使用。

## 技术栈

UESF基于Python 3.12开发，使用PyTorch 2.10.0

## 环境

### 开发环境

UESF的开发环境使用Docker容器，


## 使用

### 数据集

在UESF中，数据集被分为原始数据集和预处理数据集。您需要先导入原始数据集，然后对原始数据集进行预处理，得到预处理数据集。

#### 导入原始数据集

使用`uesf raw import /path/to/raw/dataset`命令导入原始数据集。数据集路径应当是一个目录，在这个目录下，您应当存放：
- 若干个.mat文件：每个.mat文件包含一个被试的EEG数据，形状为(记录数, 通道数, 采样点数)
- 一个raw.yml文件：包含数据集的元信息，包括数据集名称、采样率、被试数、记录数、通道数、采样点数、类别数、电极列表等。数据集名称不应与其它数据集重复

#### 数据预处理

使用`uesf raw preprocess <raw-dataset-name>`命令对原始数据集进行预处理。您可以添加参数`--config <path>`来指定预处理配置，否则UESF将自动加载当前工作区下名为`preprocess.yml`的配置文件。

> 注意：如果需要，您应当在`preprocess.yml`中指定预处理数据集的名称，或在命令中添加参数`--output-name <processed-dataset-name>`，否则UESF将基于原始数据集的名称自动生成预处理数据集的名称。

生成的预处理数据集将存放在数据目录下（默认为~/.uesf/data/processed/<processed-dataset-name>，您可以通过uesf set命令进行自定义）。预处理数据集的目录结构如下：
```
<processed-dataset-name>
├── preprocessed.yml
└── <subject-id>.npy
```
其中，`preprocessed.yml`文件包含预处理数据集的元信息，包括数据集名称、采样率、被试数、记录数、通道数、采样点数、类别数、电极列表等。`<subject-id>.npy`文件包含预处理数据集的EEG数据和标签，形状分别为(记录数, 通道数, 采样点数)和(记录数, 标签维数)。

> 标签维数大于1时，可以保存除了类别标签之外的其它信息。一般情况下，应该使用这一维度的下标0表示类别。

### 项目

UESF中的项目被定义为一套数据预处理、模型训练和评估的完整流程。一个项目包含：
- 一个或若干个预处理数据集（或原始数据集与自定义预处理流程）
- 一个或若干个模型
- 一套评估配置

> 注意：使用`uesf project`系列命令时，请确保您的终端工作目录位于UESF项目目录下。

使用`uesf project register`命令注册项目，当前工作区下应存在项目配置文件`project.yml`，其内容如下：
```yaml
name: <project-name>
description: <project-description>
# 使用预处理数据集
preprocessed_datasets:
  - <preprocessed-dataset-name>
# 使用原始数据集与自定义预处理流程
raw_datasets:
  - <raw-dataset-name>
  preprocess_config: <path-to-preprocess-config>
models:
  - <model-name>
evaluations:
  - <evaluation-name>
```

注册后，可以使用`uesf project list`命令查看已注册的项目，使用`uesf project remove`命令删除已注册的项目。

如果您的项目使用原始数据集与自定义预处理流程，您需要运行`uesf project preprocess`命令对原始数据集进行预处理。

使用`uesf project train`命令对预处理数据集进行模型训练，最后使用`uesf project evaluate`命令对模型进行评估。

您可以使用集成命令`uesf project run`来依次执行预处理、训练和评估命令。该命令会自动检测您是否修改了预处理配置、模型或评估配置，且仅在您修改了配置或没有运行过预处理、训练或评估命令时，才会执行相应的命令。

