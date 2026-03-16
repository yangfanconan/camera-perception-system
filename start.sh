#!/bin/bash
# 快速启动脚本

set -e

echo "🚀 启动摄像头实时感知系统..."

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
echo "📦 安装依赖..."
pip install -q -r requirements.txt

# 启动后端
echo "🔧 启动后端服务..."
python -m src.api.main --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# 等待后端启动
sleep 3

# 启动前端
echo "🎨 启动前端服务..."
cd web

if [ ! -d "node_modules" ]; then
    echo "📦 安装前端依赖..."
    npm install
fi

npm run dev &
FRONTEND_PID=$!

# 等待退出
echo ""
echo "✅ 系统已启动！"
echo "   - 后端：http://localhost:8000"
echo "   - 前端：http://localhost:5173"
echo "   - API 文档：http://localhost:8000/docs"
echo ""
echo "按 Ctrl+C 停止所有服务"

wait
