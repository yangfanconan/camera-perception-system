"""
调试模式 - 最小化启动（无需 YOLO/MediaPipe）
用于测试系统核心功能
"""

import asyncio
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
from pathlib import Path
from loguru import logger
import time

# 设置日志
logger.add(lambda msg: print(msg, end=''), level="INFO", colorize=True)

app = FastAPI(title="调试模式 - 摄像头感知系统")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局状态
class DebugState:
    def __init__(self):
        self.camera = None
        self.camera_opened = False
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
    
    def update_fps(self):
        current = time.time()
        if current - self.last_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_time = current
        self.frame_count += 1

state = DebugState()


@app.on_event("startup")
async def startup():
    logger.info("Starting debug server...")


@app.on_event("shutdown")
async def shutdown():
    if state.camera:
        state.camera.release()
    logger.info("Server stopped")


@app.get("/")
async def root():
    return HTMLResponse("""
    <html>
        <head>
            <title>调试模式 - 摄像头感知系统</title>
            <style>
                body { font-family: Arial; background: #0f0f1a; color: white; padding: 20px; }
                h1 { color: #00ff00; }
                .status { background: #1a1a2e; padding: 20px; border-radius: 8px; margin: 20px 0; }
                .ok { color: #00ff00; }
                .error { color: #ff4444; }
            </style>
        </head>
        <body>
            <h1>🎥 调试模式 - 摄像头感知系统</h1>
            <div class="status">
                <h2>系统状态</h2>
                <div id="status">加载中...</div>
            </div>
            <div class="status">
                <h2>操作</h2>
                <button onclick="startCamera()">启动摄像头</button>
                <button onclick="stopCamera()">停止摄像头</button>
                <button onclick="testMPS()">测试 MPS</button>
            </div>
            <script>
                async function getStatus() {
                    const r = await fetch('/api/status');
                    const d = await r.json();
                    document.getElementById('status').innerHTML = 
                        `<pre>${JSON.stringify(d, null, 2)}</pre>`;
                }
                async function startCamera() {
                    await fetch('/api/camera/start', {method: 'POST'});
                    getStatus();
                }
                async function stopCamera() {
                    await fetch('/api/camera/stop', {method: 'POST'});
                    getStatus();
                }
                async function testMPS() {
                    const r = await fetch('/api/test/mps', {method: 'POST'});
                    const d = await r.json();
                    alert(JSON.stringify(d, null, 2));
                }
                getStatus();
                setInterval(getStatus, 2000);
            </script>
        </body>
    </html>
    """)


@app.get("/api/status")
async def get_status():
    return {
        "camera_opened": state.camera_opened,
        "fps": state.fps,
        "platform": "macOS ARM64",
        "torch_mps": True,
        "debug_mode": True
    }


@app.post("/api/camera/start")
async def start_camera():
    if state.camera is None:
        state.camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        state.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        state.camera.set(cv2.CAP_PROP_FPS, 30)
        
        if state.camera.isOpened():
            state.camera_opened = True
            logger.info("Camera opened")
            return {"status": "success", "message": "Camera opened"}
        else:
            state.camera = None
            return {"status": "error", "message": "Failed to open camera"}
    
    return {"status": "info", "message": "Camera already opened"}


@app.post("/api/camera/stop")
async def stop_camera():
    if state.camera:
        state.camera.release()
        state.camera = None
        state.camera_opened = False
        logger.info("Camera stopped")
        return {"status": "success", "message": "Camera stopped"}
    return {"status": "info", "message": "Camera not opened"}


@app.post("/api/test/mps")
async def test_mps():
    """测试 MPS 加速"""
    try:
        import torch
        
        results = {
            "torch_version": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
            "device": "mps" if torch.backends.mps.is_available() else "cpu"
        }
        
        # 简单 MPS 测试
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            
            # 计算矩阵乘法
            start = time.time()
            for _ in range(100):
                z = x @ y
            elapsed = time.time() - start
            
            results["mps_test"] = {
                "operation": "100x matrix multiplication",
                "time_ms": round(elapsed * 1000, 2),
                "status": "success"
            }
        
        return results
    
    except Exception as e:
        return {"error": str(e)}


@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket video connected")
    
    try:
        while state.camera_opened and state.camera:
            ret, frame = state.camera.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue
            
            # 编码为 JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            try:
                await websocket.send_bytes(buffer.tobytes())
            except:
                break
            
            state.update_fps()
            await asyncio.sleep(1/30)
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("WebSocket video disconnected")


@app.websocket("/ws/data")
async def websocket_data(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket data connected")
    
    try:
        while True:
            # 发送模拟数据
            data = {
                "persons": [],
                "hands": [],
                "fps": state.fps,
                "timestamp": time.time()
            }
            
            try:
                await websocket.send_json(data)
            except:
                break
            
            await asyncio.sleep(0.05)
    
    except Exception as e:
        logger.error(f"WebSocket data error: {e}")


if __name__ == '__main__':
    print("\n" + "="*50)
    print("🔧 调试模式启动")
    print("="*50)
    print("\n访问地址:")
    print("  http://localhost:8100")
    print("  API 文档：http://localhost:8100/docs")
    print("\n按 Ctrl+C 停止服务\n")
    
    uvicorn.run(app, host='0.0.0.0', port=8100, log_level="info")
