#!/bin/bash
# 服务器启动脚本

set -e

cd /Users/yangfan/camera-perception-system

echo ""
echo "============================================================"
echo "🔧 摄像头感知系统 - M5 Pro 调试模式"
echo "============================================================"
echo ""

# 设置环境变量
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=4

# 创建日志目录
mkdir -p logs

# 检查端口
if lsof -i :8100 > /dev/null 2>&1; then
    echo "⚠️  端口 8100 已被占用，停止旧进程..."
    lsof -ti :8100 | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# 启动服务器
echo "🚀 启动服务器..."
echo ""

python3 run_server.py &
SERVER_PID=$!

# 等待服务器启动
echo "等待服务器启动..."
for i in {1..10}; do
    if curl -s http://localhost:8100/api/status > /dev/null 2>&1; then
        echo "✅ 服务器启动成功！"
        echo ""
        echo "============================================================"
        echo "访问地址:"
        echo "  Web 界面：http://localhost:8100"
        echo "  API 文档：http://localhost:8100/docs"
        echo "============================================================"
        echo ""
        echo "服务器 PID: $SERVER_PID"
        echo ""
        echo "按 Ctrl+C 停止服务器"
        echo ""
        
        # 保持运行
        wait $SERVER_PID
        exit 0
    fi
    sleep 1
done

echo "❌ 服务器启动失败"
exit 1
