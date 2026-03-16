#!/bin/bash
# 快速启动调试脚本（使用系统 Python）

set -e

echo "🚀 启动调试（使用系统 Python）"
echo "================================"

cd /Users/yangfan/camera-perception-system

# 设置环境变量
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=4
export CONFIG_FILE=configs/camera_apple_silicon.yaml

echo ""
echo "🔍 系统检测..."
python3 << 'EOF'
import platform
import sys

print(f"Python: {sys.version}")
print(f"平台：{platform.platform()}")
print(f"架构：{platform.machine()}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"MPS 可用：{torch.backends.mps.is_available()}")
except ImportError:
    print("PyTorch: 未安装")

try:
    import cv2
    print(f"OpenCV: {cv2.__version__}")
except ImportError:
    print("OpenCV: 未安装")

try:
    from ultralytics import YOLO
    print("Ultralytics YOLO: 已安装")
except ImportError:
    print("Ultralytics YOLO: 未安装")

try:
    import mediapipe
    print(f"MediaPipe: {mediapipe.__version__}")
except ImportError:
    print("MediaPipe: 未安装")

try:
    import fastapi
    print(f"FastAPI: {fastapi.__version__}")
except ImportError:
    print("FastAPI: 未安装")
EOF

echo ""
echo "📦 安装缺失的依赖..."
pip3 install -q loguru psutil pyyaml watchdog 2>/dev/null || true

echo ""
echo "🔧 测试 Apple Silicon 优化器..."
python3 -c "
from src.utils.apple_silicon import AppleSiliconOptimizer
optimizer = AppleSiliconOptimizer()
print(f'Apple Silicon: {optimizer.is_apple_silicon}')
print(f'M5 Pro: {optimizer.is_m5_pro}')
print(f'MPS 可用：{optimizer.mps_available}')
print(f'最优设备：{optimizer.device}')
"

echo ""
echo "🎯 启动后端服务（测试模式）..."
echo "   API: http://localhost:8000"
echo "   文档：http://localhost:8000/docs"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

# 启动后端（不重启模式）
python3 -m src.api.main_full --host 0.0.0.0 --port 8000 --reload
