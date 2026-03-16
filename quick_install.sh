#!/bin/bash
# 快速安装完整依赖脚本

set -e

echo "🚀 安装完整依赖（M5 Pro 优化版）"
echo "=================================="
echo ""

# 安装 YOLOv8
echo "📦 安装 YOLOv8 (ultralytics)..."
pip3 install ultralytics --quiet

# 安装 MediaPipe
echo "📦 安装 MediaPipe..."
pip3 install mediapipe --quiet

# 安装可选依赖
echo "📦 安装可选依赖..."
pip3 install psutil pyyaml watchdog --quiet

echo ""
echo "✅ 所有依赖安装完成！"
echo ""
echo "下载 YOLO 模型..."
yolo download yolov8n.pt
yolo download yolov8n-pose.pt

echo ""
echo "======================================"
echo "✅ 系统已准备就绪！"
echo ""
echo "启动完整服务：python3 -m src.api.full_server"
echo "访问地址：http://localhost:8100"
echo "======================================"
