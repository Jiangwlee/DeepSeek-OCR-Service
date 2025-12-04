FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y python3-pip python3-venv git curl tzdata && \
    ln -snf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" >/etc/timezone && \
    rm -rf /var/lib/apt/lists/*

# 创建虚拟环境
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# 配置pip使用清华源
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# 升级pip
RUN pip install --upgrade pip

# 安装依赖
RUN pip install modelscope
RUN pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
# 安装Qwen3-VL所需的工具库（如果使用Qwen模型）
RUN pip install qwen-vl-utils>=0.0.14

# 复制启动脚本（注意：构建上下文为仓库根目录）
COPY services/vllm-service/run-vllm-service.sh /app/run-vllm-service.sh

# 设置环境变量
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV VLLM_USE_MODELSCOPE=True

# 设置启动命令
CMD ["bash", "/app/run-vllm-service.sh"]
