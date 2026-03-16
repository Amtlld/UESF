# 1. 使用 NVIDIA 官方 CUDA 12.1 开发镜像 (包含 nvcc 编译器，方便编译自定义算子)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 2. 设置环境变量，避免交互式安装卡住
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 3. 安装 Python 3.10 和基础工具
# Ubuntu 22.04 默认 Python 版本就是 3.10，所以直接安装 python3 即可
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    p7zip-full \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. 创建软链接，让 python 命令指向 python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# 5. 升级 pip
RUN python -m pip install --upgrade pip

# 6. 设置工作目录
WORKDIR /workspace

# 7. 优先安装 PyTorch (指定 cu121 版本)
# 作者强调了 cu121，我们需要指定 index-url 从 PyTorch 官方源下载对应的版本
# 这一步放在 requirements.txt 之前，防止 requirements.txt 自动拉取错误的 CUDA 版本
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 8. 复制 requirements.txt
COPY requirements.txt .

# 9. 处理 requirements.txt 并安装其余依赖
# 使用 sed 删除 requirements.txt 中的 torch 行，因为我们上一行已经手动安装了特定版本
RUN sed -i '/^torch/d' requirements.txt && pip install -r requirements.txt

# 10. 默认命令
CMD ["/bin/bash"]