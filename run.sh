#!/bin/bash
# 稳定的服务器启动脚本

set -e

cd /Users/yangfan/camera-perception-system

echo "============================================================"
echo "🚀 启动摄像头感知系统 v2"
echo "============================================================"
echo ""

# 清理旧进程
echo "📋 清理旧进程..."
lsof -ti :8100 | xargs kill -9 2>/dev/null || true
sleep 1

# 创建日志目录
mkdir -p logs

# 启动服务器（后台）
echo "🚀 启动服务器..."
nohup python3 -m src.api.full_server_v2 > logs/server.log 2>&1 &
SERVER_PID=$!

echo "✅ 服务器已启动 (PID: $SERVER_PID)"
echo ""

# 等待启动
echo "⏳ 等待服务器启动..."
for i in {1..10}; do
    if curl -s http://localhost:8100/api/status > /dev/null 2>&1; then
        echo "✅ 服务器启动成功！"
        echo ""
        echo "============================================================"
        echo "访问地址:"
        echo "  🌐 Web 界面：http://localhost:8100"
        echo "  📊 API 状态：http://localhost:8100/api/status"
        echo "  📖 API 文档：http://localhost:8100/docs"
        echo "============================================================"
        echo ""
        echo "📊 日志文件：logs/server.log"
        echo ""
        echo "停止服务器：lsof -ti :8100 | xargs kill -9"
        echo ""
        exit 0
    fi
    sleep 1
done

echo "❌ 服务器启动失败"
echo ""
echo "日志:"
tail -20 logs/server.log
exit 1
