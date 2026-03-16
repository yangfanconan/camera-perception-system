@echo off
REM Windows 快速启动脚本

setlocal enabledelayedexpansion

echo 🚀 启动摄像头实时感知系统...

REM 检查虚拟环境
if not exist "venv" (
    echo 📦 创建虚拟环境...
    python -m venv venv
)

REM 激活虚拟环境
call venv\Scripts\activate.bat

REM 安装依赖
echo 📦 安装依赖...
pip install -q -r requirements.txt

REM 启动后端
echo 🔧 启动后端服务...
start "Backend" cmd /c "python -m src.api.main --host 0.0.0.0 --port 8000"

REM 等待后端启动
timeout /t 5 /nobreak >nul

REM 启动前端
echo 🎨 启动前端服务...
cd web

if not exist "node_modules" (
    echo 📦 安装前端依赖...
    call npm install
)

start "Frontend" cmd /c "npm run dev"

echo.
echo ✅ 系统已启动！
echo    - 后端：http://localhost:8000
echo    - 前端：http://localhost:5173
echo    - API 文档：http://localhost:8000/docs
echo.
echo 关闭所有窗口以停止服务

pause
