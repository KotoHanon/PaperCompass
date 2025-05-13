FROM python:3.10-slim

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    poppler-utils \
    poppler-data \
    build-essential \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY . /app/

# 安装 MinerU
RUN pip install --no-cache-dir -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com

# 安装Python依赖
RUN pip install -r requirements.txt

# 下载预训练模型（可选，取决于网络和空间情况）
RUN mkdir -p .cache/ && cd .cache/ && \
    git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 && \
    git clone https://huggingface.co/BAAI/bge-large-en-v1.5 && \
    git clone https://huggingface.co/openai/clip-vit-base-patch32

# 设置环境变量
ENV PYTHONPATH=/app

# 容器启动命令
CMD ["/bin/bash"] 