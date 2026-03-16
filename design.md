# Design of UESF

UESF（Unified EEG Study Framework）被设计为一个CLI工具，用于实现EEG数据的预处理、特征提取、模型训练和评估等功能，供研究人员使用。

## 技术栈

UESF基于Python 3.12开发，使用PyTorch 2.10.0

## 环境

### 开发环境

UESF的开发环境使用Docker容器，


## 使用

UESF是一个CLI工具，这意味着您通过命令行使用UESF。

uesf
Usage:  uesf [OPTIONS] COMMAND

Options:
  --help  Show this message and exit.
  --version  Print version information and quit.
Management Commands:
  dataset  Manage dataset.
  project  Manage project.
  model  Manage model.


uesf dataset
Usage:  uesf dataset [OPTIONS] COMMAND

Options:
  --help  Show this message and exit.
  --version Print version information and quit.
Commands:
  import  Import dataset.
  list  List dataset.
  remove  Remove dataset.
  