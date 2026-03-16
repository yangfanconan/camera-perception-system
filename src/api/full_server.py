"""
完整功能服务器 - 支持视频流、检测、空间计量
集成 YOLOv8 + MediaPipe 检测
"""

import asyncio
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
from pathlib import Path
from loguru import logger
import time
import io
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入检测模块
try:
    from algorithms import CombinedDetector
    from algorithms.calibration import CalibrationParams
    from algorithms.spatial.core import SpatialCalculatorEnhanced
    DETECTION_AVAILABLE = True
    logger.info("Detection modules loaded successfully")
except ImportError as e:
    logger.warning(f"Detection modules not available: {e}")
    DETECTION_AVAILABLE = False

# 设置日志
logger.remove()
logger.add(lambda msg: print(msg, end=''), level="INFO", colorize=True)

app = FastAPI(
    title="摄像头感知系统 - 完整版",
    description="实时人体/手部检测与空间计量",
    version="0.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 全局状态 ====================

class SystemState:
    def __init__(self):
        self.camera = None
        self.camera_opened = False
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        self.detector = None
        self.spatial_calc = None
        self.calibrated = False
        
    def update_fps(self):
        current = time.time()
        if current - self.last_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_time = current
        self.frame_count += 1

state = SystemState()

# ==================== 数据模型 ====================

class CameraConfig(BaseModel):
    camera_id: int = 0
    resolution: list = [1920, 1080]
    fps: int = 30

# ==================== API 路由 ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    """主页面"""
    return """
    <html>
        <head>
            <title>摄像头感知系统 - 完整版</title>
            <meta charset="UTF-8">
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body { 
                    font-family: Arial, sans-serif; 
                    background: #0f0f1a; 
                    color: white; 
                    padding: 20px;
                }
                h1 { color: #00ff00; margin-bottom: 20px; }
                .container { 
                    display: flex; 
                    gap: 20px; 
                    flex-wrap: wrap; 
                    align-items: flex-start;
                }
                .video-panel { 
                    flex: 1 1 640px; 
                    min-width: 320px;
                    max-width: 100%;
                }
                .data-panel { 
                    width: 350px; 
                    flex-shrink: 0;
                }
                .video-container {
                    position: relative;
                    background: #000;
                    border-radius: 12px;
                    overflow: hidden;
                    width: 100%;
                    max-width: 1280px;
                    margin: 0 auto;
                }
                #video { 
                    width: 100%; 
                    height: auto; 
                    display: block; 
                    min-height: 360px;
                    max-height: 720px;
                    object-fit: contain;
                }
                #overlay {
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    pointer-events: none;
                }
                
                @media (max-width: 1024px) {
                    .container { flex-direction: column; }
                    .video-panel { min-width: 100%; }
                    .data-panel { width: 100%; }
                }
                .panel {
                    background: rgba(26, 26, 46, 0.8);
                    border-radius: 12px;
                    padding: 20px;
                    margin-bottom: 15px;
                    border: 1px solid #333355;
                }
                .panel h2 { color: #00ff00; margin-bottom: 15px; font-size: 18px; }
                .metrics { display: flex; gap: 15px; }
                .metric {
                    flex: 1;
                    text-align: center;
                    background: rgba(0,0,0,0.3);
                    padding: 15px;
                    border-radius: 8px;
                }
                .metric .label { color: #888; font-size: 12px; display: block; margin-bottom: 5px; }
                .metric .value { color: #00ff00; font-size: 24px; font-weight: bold; }
                .data-row {
                    display: flex;
                    justify-content: space-between;
                    padding: 8px 0;
                    border-bottom: 1px solid rgba(255,255,255,0.1);
                }
                .data-row:last-child { border-bottom: none; }
                .highlight { color: #00ff00; font-weight: bold; }
                .controls { display: flex; gap: 10px; margin-top: 15px; }
                button {
                    padding: 10px 20px;
                    border: none;
                    border-radius: 6px;
                    cursor: pointer;
                    font-size: 14px;
                    font-weight: 500;
                }
                .btn-primary { background: linear-gradient(90deg, #4a4a6a, #5a5a7a); color: white; }
                .btn-danger { background: linear-gradient(90deg, #dc3545, #c82333); color: white; }
                button:disabled { opacity: 0.5; cursor: not-allowed; }
                #topview { background: #1a1a2e; border-radius: 8px; width: 100%; }
                .status { display: flex; align-items: center; gap: 10px; margin-bottom: 20px; }
                .status-dot { width: 10px; height: 10px; border-radius: 50%; background: #ff4444; }
                .status-dot.online { background: #00ff00; box-shadow: 0 0 10px #00ff00; }
            </style>
        </head>
        <body>
            <h1>🎥 摄像头感知系统 - 完整版</h1>
            
            <div class="status">
                <div class="status-dot" id="statusDot"></div>
                <span id="statusText">未连接</span>
                <span id="fpsDisplay">FPS: 0</span>
            </div>
            
            <div class="container">
                <div class="video-panel">
                    <div class="video-container">
                        <video id="video" autoplay playsinline></video>
                        <canvas id="overlay"></canvas>
                    </div>
                    <div class="controls">
                        <button class="btn-primary" onclick="startCamera()" id="startBtn">▶ 启动摄像头</button>
                        <button class="btn-danger" onclick="stopCamera()" id="stopBtn" disabled>⏹ 停止摄像头</button>
                    </div>
                </div>
                
                <div class="data-panel">
                    <div class="panel">
                        <h2>📊 检测结果</h2>
                        <div class="metrics">
                            <div class="metric">
                                <span class="label">人数</span>
                                <span class="value" id="personCount">0</span>
                            </div>
                            <div class="metric">
                                <span class="label">手数</span>
                                <span class="value" id="handCount">0</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="panel" id="personPanel">
                        <h2>👤 人体数据</h2>
                        <div id="personData">等待检测...</div>
                    </div>
                    
                    <div class="panel" id="handPanel">
                        <h2>✋ 手部数据</h2>
                        <div id="handData">等待检测...</div>
                    </div>
                    
                    <div class="panel">
                        <h2>🗺️ 顶视图</h2>
                        <canvas id="topview" width="310" height="250"></canvas>
                    </div>
                </div>
            </div>
            
            <script>
                let videoStream = null;
                let dataWs = null;
                let isRunning = false;
                
                function updateStatus(online, fps) {
                    document.getElementById('statusDot').className = 'status-dot' + (online ? ' online' : '');
                    document.getElementById('statusText').textContent = online ? '摄像头已连接' : '未连接';
                    document.getElementById('fpsDisplay').textContent = 'FPS: ' + fps;
                }
                
                async function startCamera() {
                    try {
                        const r = await fetch('/api/camera/start', {method: 'POST'});
                        const d = await r.json();
                        if (d.status === 'success') {
                            document.getElementById('startBtn').disabled = true;
                            document.getElementById('stopBtn').disabled = false;
                            isRunning = true;
                            startVideoStream();
                            connectWebSocket();
                        } else {
                            alert('启动失败：' + d.message);
                        }
                    } catch (e) {
                        alert('启动失败：' + e.message);
                    }
                }
                
                async function stopCamera() {
                    try {
                        await fetch('/api/camera/stop', {method: 'POST'});
                        document.getElementById('startBtn').disabled = false;
                        document.getElementById('stopBtn').disabled = true;
                        isRunning = false;
                        if (videoStream) {
                            videoStream.src = '';
                            videoStream = null;
                        }
                        if (dataWs) dataWs.close();
                        updateStatus(false, 0);
                    } catch (e) {
                        console.error(e);
                    }
                }
                
                function startVideoStream() {
                    videoStream = document.getElementById('video');
                    videoStream.src = '/ws/video';
                    videoStream.onloadedmetadata = () => {
                        const canvas = document.getElementById('overlay');
                        canvas.width = videoStream.videoWidth;
                        canvas.height = videoStream.videoHeight;
                    };
                }
                
                function connectWebSocket() {
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    dataWs = new WebSocket(protocol + '//' + window.location.host + '/ws/data');
                    
                    dataWs.onopen = () => {
                        console.log('WebSocket connected');
                        updateStatus(true, 0);
                    };
                    
                    dataWs.onmessage = (event) => {
                        const data = JSON.parse(event.data);
                        
                        // 更新 FPS
                        if (data.fps) {
                            document.getElementById('fpsDisplay').textContent = 'FPS: ' + data.fps;
                        }
                        
                        // 更新人数/手数
                        document.getElementById('personCount').textContent = data.persons?.length || 0;
                        document.getElementById('handCount').textContent = data.hands?.length || 0;
                        
                        // 更新人体数据
                        const personDiv = document.getElementById('personData');
                        if (data.persons && data.persons.length > 0) {
                            personDiv.innerHTML = data.persons.map((p, i) => `
                                <div class="data-row">
                                    <span>人${i+1}:</span>
                                    <span>距离 ${p.distance}m | 身高 ${p.height}cm</span>
                                </div>
                            `).join('');
                        } else {
                            personDiv.innerHTML = '未检测到人体';
                        }
                        
                        // 更新手部数据
                        const handDiv = document.getElementById('handData');
                        if (data.hands && data.hands.length > 0) {
                            handDiv.innerHTML = data.hands.map((h, i) => `
                                <div class="data-row">
                                    <span>手${i+1}:</span>
                                    <span>大小 ${h.size}cm | 距离 ${h.distance}m</span>
                                </div>
                            `).join('');
                        } else {
                            handDiv.innerHTML = '未检测到手部';
                        }
                        
                        // 绘制顶视图
                        drawTopView(data);
                        
                        // 绘制标注
                        drawOverlay(data);
                    };
                    
                    dataWs.onclose = () => {
                        console.log('WebSocket disconnected');
                        if (isRunning) {
                            setTimeout(connectWebSocket, 1000);
                        }
                    };
                }
                
                function drawTopView(data) {
                    const canvas = document.getElementById('topview');
                    const ctx = canvas.getContext('2d');
                    
                    // 清空
                    ctx.fillStyle = '#1a1a2e';
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    
                    // 网格
                    ctx.strokeStyle = '#333355';
                    ctx.lineWidth = 1;
                    for (let x = 0; x < canvas.width; x += 25) {
                        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
                    }
                    for (let y = 0; y < canvas.height; y += 25) {
                        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
                    }
                    
                    // 摄像头位置
                    ctx.fillStyle = '#ffffff';
                    ctx.beginPath();
                    ctx.arc(canvas.width/2, canvas.height/2, 6, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.fillText('摄像头', canvas.width/2 + 10, canvas.height/2);
                    
                    // 人体位置
                    data.persons?.forEach((p, i) => {
                        if (p.topview) {
                            ctx.fillStyle = '#00ff00';
                            ctx.beginPath();
                            ctx.arc(p.topview.x, p.topview.y, 10, 0, Math.PI * 2);
                            ctx.fill();
                            ctx.fillText(`人${i+1}`, p.topview.x - 10, p.topview.y - 12);
                        }
                    });
                    
                    // 手部位置
                    data.hands?.forEach((h, i) => {
                        if (h.topview) {
                            ctx.fillStyle = '#0088ff';
                            ctx.beginPath();
                            ctx.arc(h.topview.x, h.topview.y, 8, 0, Math.PI * 2);
                            ctx.fill();
                            ctx.fillText(`手${i+1}`, h.topview.x - 10, h.topview.y - 10);
                        }
                    });
                }
                
                function drawOverlay(data) {
                    const canvas = document.getElementById('overlay');
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    
                    // 绘制人体
                    data.persons?.forEach(p => {
                        const [x, y, w, h] = p.bbox;
                        ctx.strokeStyle = '#00ff00';
                        ctx.lineWidth = 2;
                        ctx.strokeRect(x, y, w, h);
                        
                        // 关键点
                        if (p.keypoints) {
                            Object.values(p.keypoints).forEach(kp => {
                                ctx.fillStyle = '#ff0000';
                                ctx.beginPath();
                                ctx.arc(kp[0], kp[1], 4, 0, Math.PI * 2);
                                ctx.fill();
                            });
                        }
                        
                        // 文字
                        ctx.fillStyle = '#00ff00';
                        ctx.font = '14px Arial';
                        ctx.fillText(`距离:${p.distance}m 身高:${p.height}cm`, x, y - 10);
                    });
                    
                    // 绘制手部
                    data.hands?.forEach(h => {
                        const [x, y, w, h2] = h.bbox;
                        ctx.strokeStyle = '#0088ff';
                        ctx.lineWidth = 2;
                        ctx.strokeRect(x, y, w, h2);
                    });
                }
            </script>
        </body>
    </html>
    """


@app.get("/api/status")
async def get_status():
    return {
        "camera_opened": state.camera_opened,
        "fps": state.fps,
        "platform": "macOS ARM64",
        "torch_mps": True,
        "debug_mode": False,
        "calibrated": state.calibrated
    }


@app.post("/api/camera/start")
async def start_camera(config: CameraConfig = None):
    if config is None:
        config = CameraConfig()

    if state.camera:
        state.camera.release()

    state.camera = cv2.VideoCapture(config.camera_id, cv2.CAP_AVFOUNDATION)
    state.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.resolution[0])
    state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.resolution[1])
    state.camera.set(cv2.CAP_PROP_FPS, config.fps)

    if state.camera.isOpened():
        state.camera_opened = True
        
        # 初始化检测器
        if DETECTION_AVAILABLE and state.detector is None:
            try:
                logger.info("Initializing detector...")
                state.detector = CombinedDetector(
                    conf_threshold=0.5,
                    smooth=True
                )
                logger.info("Detector initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize detector: {e}")
        
        # 初始化空间计算器
        if DETECTION_AVAILABLE and state.spatial_calc is None:
            try:
                logger.info("Initializing spatial calculator...")
                calib_params = CalibrationParams(
                    fx=650.0, fy=650.0, cx=320.0, cy=240.0
                )
                state.spatial_calc = SpatialCalculatorEnhanced(calib_params)
                logger.info("Spatial calculator initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize spatial calculator: {e}")
        
        logger.info(f"Camera opened: {config.resolution[0]}x{config.resolution[1]}@{config.fps}fps")
        return {"status": "success", "message": "Camera opened"}
    else:
        state.camera = None
        return {"status": "error", "message": "Failed to open camera"}


@app.post("/api/camera/stop")
async def stop_camera():
    if state.camera:
        state.camera.release()
        state.camera = None
        state.camera_opened = False
        logger.info("Camera stopped")
        return {"status": "success", "message": "Camera stopped"}
    return {"status": "info", "message": "Camera not opened"}


@app.get("/ws/video")
async def video_stream():
    """MJPEG 视频流 - 自动缩放到合理尺寸"""
    # 目标输出尺寸（保持宽高比，最大边不超过 1280）
    TARGET_WIDTH = 1280
    TARGET_HEIGHT = 720
    
    async def generate():
        while state.camera_opened and state.camera:
            ret, frame = state.camera.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue

            # 获取原始尺寸
            orig_h, orig_w = frame.shape[:2]
            
            # 如果原始尺寸太大，进行缩放
            if orig_w > TARGET_WIDTH or orig_h > TARGET_HEIGHT:
                scale = min(TARGET_WIDTH / orig_w, TARGET_HEIGHT / orig_h)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # 编码为 JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            state.update_fps()
            await asyncio.sleep(1/30)

    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')


@app.websocket("/ws/data")
async def websocket_data(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket data connected")

    try:
        while state.camera_opened:
            if not state.camera:
                await asyncio.sleep(0.01)
                continue

            # 读取帧
            ret, frame = state.camera.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue

            # 初始化数据
            data = {
                "persons": [],
                "hands": [],
                "fps": state.fps,
                "timestamp": time.time()
            }

            # 执行检测
            if DETECTION_AVAILABLE and state.detector:
                try:
                    result = state.detector.detect(frame)
                    
                    # 处理人体检测
                    if state.spatial_calc:
                        frame_height, frame_width = frame.shape[:2]
                        for person in result.persons:
                            person_metrics = state.spatial_calc.calc_person_metrics(
                                person,
                                image=frame,
                                image_height=frame_height,
                                image_width=frame_width
                            )
                            data["persons"].append(person_metrics)
                    else:
                        data["persons"] = result.persons
                    
                    # 处理手部检测
                    if state.spatial_calc:
                        for hand in result.hands:
                            hand_metrics = state.spatial_calc.calc_hand_metrics(hand)
                            data["hands"].append(hand_metrics)
                    else:
                        data["hands"] = result.hands
                        
                except Exception as e:
                    logger.error(f"Detection error: {e}")

            try:
                await websocket.send_json(data)
            except:
                break

            await asyncio.sleep(0.05)

    except Exception as e:
        logger.error(f"WebSocket data error: {e}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🔧 摄像头感知系统 - 完整版")
    print("="*60)
    print("\n访问地址:")
    print("  http://localhost:8100")
    print("\n按 Ctrl+C 停止服务\n")
    
    uvicorn.run(app, host='0.0.0.0', port=8100, log_level="info")
