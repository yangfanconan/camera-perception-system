#!/bin/bash
# Apple Silicon M5 Pro 优化启动脚本

set -e

echo "🍎 Apple Silicon M5 Pro 优化启动脚本"
echo "======================================"

# 检测平台
if [[ "$(uname)" != "Darwin" ]]; then
    echo "❌ 此脚本仅适用于 macOS"
    exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "⚠️  未检测到 Apple Silicon，将使用通用配置"
fi

# 设置环境变量（MPS 优化）
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# 选择配置文件
if [ -f "configs/camera_apple_silicon.yaml" ]; then
    export CONFIG_FILE=configs/camera_apple_silicon.yaml
    echo "✅ 使用 M5 Pro 优化配置：$CONFIG_FILE"
else
    export CONFIG_FILE=configs/camera.yaml
    echo "⚠️  使用默认配置：$CONFIG_FILE"
fi

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate

# 检查/安装依赖
echo "📦 检查依赖..."
pip install -q -r requirements.txt

# 检查模型文件
if [ ! -f "models/yolov8n-pose.pt" ]; then
    echo "📥 下载 YOLOv8n-Pose 模型..."
    mkdir -p models
    yolo download yolov8n-pose.pt || echo "⚠️  模型下载失败，将使用自动下载"
fi

if [ ! -f "models/yolov8n.pt" ]; then
    echo "📥 下载 YOLOv8n 模型..."
    mkdir -p models
    yolo download yolov8n.pt || echo "⚠️  模型下载失败，将使用自动下载"
fi

# 创建必要目录
mkdir -p data calibration_data logs

# 启动服务
echo ""
echo "🚀 启动服务..."
echo "   - 后端：http://localhost:8000"
echo "   - API 文档：http://localhost:8000/docs"
echo "   - 前端：http://localhost:5173 (需单独启动)"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

# 运行优化测试
echo "🔍 运行系统检测..."
python3 -c "
from src.utils.apple_silicon import main
main()
"

echo ""
echo "▶️  启动 FastAPI 服务..."
python -m src.api.main_full --host 0.0.0.0 --port 8000
