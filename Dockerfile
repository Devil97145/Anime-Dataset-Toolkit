# 使用 NVIDIA 官方 PyTorch 镜像（含 CUDA + cuDNN + PyTorch）
# 若仅需 CPU，可替换为：python:3.12-slim
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# 设置工作目录
WORKDIR /app

# 安装系统依赖（FFmpeg + aria2）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        aria2 \
        wget \
        unzip \
        libgl1 \
        libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
# 注意：onnxruntime-gpu 已包含在 pytorch 镜像中，无需重复安装
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install onnxruntime-gpu  # 显式确保使用 GPU 版本

# 复制项目代码
COPY . .

# 创建输出目录（避免权限问题）
RUN mkdir -p /app/outputs && chmod -R 777 /app/outputs

# 暴露 Gradio 端口
EXPOSE 7860

# 启动命令（允许外部访问）
CMD ["python", "app.py", "--server-name", "0.0.0.0", "--server-port", "7860"]