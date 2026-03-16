#!/bin/bash
# Apple Silicon M5 Pro 环境安装脚本

set -e

echo "🍎 Apple Silicon M5 Pro 环境安装"
echo "=================================="

# 检测平台
if [[ "$(uname)" == "Darwin" ]]; then
    echo "✅ macOS 检测到"
    if [[ "$(uname -m)" == "arm64" ]]; then
        echo "✅ Apple Silicon 检测到"
    fi
fi

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✅ Python $PYTHON_VERSION"

# 创建虚拟环境
echo ""
echo "📦 创建虚拟环境..."
if [ -d "venv" ]; then
    echo "⚠️  venv 已存在，将删除并重新创建"
    rm -rf venv
fi

python3 -m venv venv
echo "✅ 虚拟环境创建成功"

# 激活虚拟环境
source venv/bin/activate

# 升级 pip
echo ""
echo "📦 升级 pip..."
pip install --upgrade pip

# 安装依赖
echo ""
echo "📦 安装依赖（针对 M5 Pro 优化）..."

# 先安装基础依赖
pip install numpy opencv-python pyyaml loguru psutil watchdog

# 安装 PyTorch（MPS 支持）
echo ""
echo "📦 安装 PyTorch（MPS 加速版）..."
pip install torch torchvision

# 安装 YOLOv8
echo ""
echo "📦 安装 YOLOv8..."
pip install ultralytics

# 安装 MediaPipe
echo ""
echo "📦 安装 MediaPipe..."
pip install mediapipe

# 安装 Web 依赖
echo ""
echo "📦 安装 Web 依赖..."
pip install fastapi uvicorn websockets python-multipart pydantic

# 安装其他依赖
pip install python-dotenv json5 pytest pytest-asyncio

echo ""
echo "✅ 所有依赖安装完成"

# 验证安装
echo ""
echo "🔍 验证安装..."
python -c "
import torch
import cv2
from ultralytics import YOLO

print(f'PyTorch: {torch.__version__}')
print(f'MPS 可用：{torch.backends.mps.is_available()}')
print(f'OpenCV: {cv2.__version__}')
print(f'平台：{torch.backends.mps.is_available() and \"MPS 加速已启用\" or \"CPU 模式\"}')
"

# 下载模型（可选）
echo ""
read -p "是否现在下载 YOLOv8 模型？（约 100MB，可稍后自动下载）[y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📥 下载 YOLOv8n 模型..."
    yolo download yolov8n.pt
    echo "📥 下载 YOLOv8n-Pose 模型..."
    yolo download yolov8n-pose.pt
    echo "✅ 模型下载完成"
fi

echo ""
echo "======================================"
echo "✅ 安装完成！"
echo ""
echo "使用方法:"
echo "  1. 激活虚拟环境：source venv/bin/activate"
echo "  2. 启动优化服务：./start_m5_optimized.sh"
echo "  3. 查看优化文档：cat APPLE_SILICON_OPTIMIZATION.md"
echo ""
