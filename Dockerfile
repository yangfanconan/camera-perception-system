# 多阶段构建 - 摄像头实时感知系统
FROM python:3.9-slim as base

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# ====================
# 构建阶段
# ====================
FROM base as builder

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制源代码
COPY . .

# 安装项目
RUN pip install -e .

# ====================
# 生产阶段
# ====================
FROM base as production

# 创建非 root 用户
RUN useradd --create-home --shell /bin/bash appuser

# 复制已安装的包
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制源代码
COPY --chown=appuser:appuser . .

# 创建数据目录
RUN mkdir -p /app/data /app/calibration_data /app/models /app/logs && \
    chown -R appuser:appuser /app

# 切换到非 root 用户
USER appuser

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/health', timeout=5)" || exit 1

# 环境变量
ENV PYTHONPATH=/app \
    MODEL_CACHE_DIR=/app/models \
    LOG_LEVEL=INFO \
    MAX_MEMORY_GB=4 \
    ENABLE_DEPTH_ESTIMATION=true \
    ENABLE_TENSORRT=false

# 卷挂载点
VOLUME ["/app/data", "/app/calibration_data", "/app/models", "/app/logs", "/app/configs"]

# 启动命令
CMD ["python", "-m", "src.api.main", "--host", "0.0.0.0", "--port", "8000"]

# ====================
# 开发阶段
# ====================
FROM base as development

# 安装开发依赖
RUN pip install \
    pytest \
    pytest-cov \
    black \
    isort \
    flake8 \
    mypy

# 复制源代码
COPY --chown=appuser:appuser . .

# 创建数据目录
RUN mkdir -p /app/data /app/calibration_data /app/models /app/logs && \
    chown -R appuser:appuser /app

# 切换到非 root 用户
USER appuser

# 暴露端口
EXPOSE 8000

# 默认命令
CMD ["python", "-m", "src.api.main_full", "--host", "0.0.0.0", "--port", "8000", "--reload"]
