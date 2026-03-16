"""
完整功能服务器 v2 - 集成 YOLO + MediaPipe + 空间计量
支持自动降级（依赖未安装时使用模拟数据）
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
import json

# 设置日志
logger.remove()
logger.add(lambda msg: print(msg, end=''), level="INFO", colorize=True)

app = FastAPI(title="摄像头感知系统 v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 全局状态 ====================

# 共享帧缓冲区
current_frame = None
frame_lock = None
capture_task = None
capture_running = False

class SystemState:
    def __init__(self):
        self.camera = None
        self.camera_opened = False
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        self.calibrated = False

        # 检测器（自动检测可用性）
        self.yolo_available = False
        self.mediapipe_available = False
        self.detector = None
        self.spatial_calc = None
        
        # 上一帧的距离（用于速度约束）
        self.prev_distances = {}  # track_id -> distance

        # 尝试加载模块
        self._init_detectors()
    
    def _init_detectors(self):
        """初始化检测器"""
        # 检查 YOLO
        try:
            from src.algorithms.detection_mps import CombinedDetectorMPS, YOLO_AVAILABLE
            self.detector = CombinedDetectorMPS(device='mps', half=True)
            self.yolo_available = YOLO_AVAILABLE
            logger.info(f"✅ YOLO 已加载：{YOLO_AVAILABLE}")
        except Exception as e:
            logger.warning(f"⚠️  YOLO 检测器未加载：{e}")
            self.yolo_available = False

        # 检查 MediaPipe
        try:
            from src.algorithms.detection_mps import MP_AVAILABLE
            self.mediapipe_available = MP_AVAILABLE
            logger.info(f"✅ MediaPipe 状态：{MP_AVAILABLE}")
        except Exception as e:
            logger.warning(f"⚠️  MediaPipe 检测器未加载：{e}")
            self.mediapipe_available = False

        # 尝试加载标定参数
        calib_path = Path("calibration_data/calib_params.json")
        if calib_path.exists():
            try:
                from src.algorithms.spatial_enhanced import SpatialCalculatorEnhanced, CalibrationParams
                with open(calib_path, 'r') as f:
                    data = json.load(f)
                calib_params = CalibrationParams.from_dict(data)
                self.spatial_calc = SpatialCalculatorEnhanced(calib_params)
                self.calibrated = True
                logger.info("✅ 标定参数已加载")
            except Exception as e:
                logger.warning(f"标定参数加载失败：{e}")
                self._create_default_spatial_calc()
        else:
            # 创建默认的空间计算器
            self._create_default_spatial_calc()
    
    def _create_default_spatial_calc(self):
        """创建默认的空间计算器"""
        try:
            from src.algorithms.spatial_enhanced import SpatialCalculatorEnhanced, CalibrationParams
            # 使用默认参数（假设 1280x720 分辨率，焦距约 1200）
            calib_params = CalibrationParams(
                fx=1200.0,
                fy=1200.0,
                cx=640.0,
                cy=360.0,
                image_size=[1280, 720]
            )
            self.spatial_calc = SpatialCalculatorEnhanced(calib_params)
            logger.info("✅ 使用默认标定参数")
        except Exception as e:
            logger.warning(f"创建默认空间计算器失败：{e}")
            self.spatial_calc = None
    
    def update_fps(self):
        current = time.time()
        if current - self.last_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_time = current
        self.frame_count += 1


async def capture_loop():
    """后台持续读取摄像头帧"""
    global current_frame, frame_lock, capture_running
    
    capture_running = True
    frame_lock = asyncio.Lock()
    
    while capture_running and state.camera_opened and state.camera:
        ret, frame = state.camera.read()
        if ret and frame is not None:
            async with frame_lock:
                current_frame = frame
        await asyncio.sleep(0.01)


state = SystemState()

# ==================== 数据模型 ====================

class CameraConfig(BaseModel):
    camera_id: int = 0
    resolution: list = [1920, 1080]
    fps: int = 30

class CalibrationInput(BaseModel):
    """校准输入"""
    person_index: int = 0  # 要校准的人体索引
    known_distance: float = None  # 已知距离（米）
    known_height: float = None  # 已知身高（厘米）

# ==================== 校准 API ====================

@app.post("/api/calibrate")
async def calibrate_system(input: CalibrationInput):
    """使用已知数据进行校准"""
    if not state.spatial_calc:
        return {"status": "error", "message": "空间计算器未初始化"}
    
    # 获取当前检测结果
    if not state.camera or not state.camera_opened:
        return {"status": "error", "message": "摄像头未启动"}
    
    ret, frame = state.camera.read()
    if not ret:
        return {"status": "error", "message": "无法读取摄像头"}
    
    # 检测人体
    if not state.detector:
        return {"status": "error", "message": "检测器未初始化"}
    
    result = state.detector.detect(frame, smooth=False)
    persons = result.get("persons", [])
    
    if not persons or input.person_index >= len(persons):
        return {"status": "error", "message": f"未找到人体 {input.person_index}"}
    
    person = persons[input.person_index]
    
    # 使用已知距离校准
    if input.known_distance:
        state.spatial_calc.calibrate_with_known_distance(
            person["bbox"], input.known_distance
        )
        logger.info(f"距离校准完成: {input.known_distance}m")
    
    # 使用已知身高校准
    if input.known_height:
        state.spatial_calc.calibrate_with_known_height(
            person["bbox"], person.get("keypoints", {}), input.known_height
        )
        logger.info(f"身高校准完成: {input.known_height}cm")
    
    return {
        "status": "success",
        "message": "校准完成",
        "calibration_status": state.spatial_calc.get_calibration_status()
    }

@app.get("/api/calibration/status")
async def get_calibration_status():
    """获取校准状态"""
    if not state.spatial_calc:
        return {"status": "error", "message": "空间计算器未初始化"}
    
    return {
        "status": "success",
        "calibration": state.spatial_calc.get_calibration_status()
    }

@app.post("/api/calibration/reset")
async def reset_calibration():
    """重置校准参数"""
    if not state.spatial_calc:
        return {"status": "error", "message": "空间计算器未初始化"}
    
    state.spatial_calc.reset_calibration()
    return {"status": "success", "message": "校准参数已重置"}

# ==================== 主页面 ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>摄像头感知系统 v2 - M5 Pro 优化版</title>
            <meta charset="UTF-8">
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body { 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; 
                    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
                    color: white; 
                    padding: 20px;
                    min-height: 100vh;
                }
                h1 { 
                    color: #00ff00; 
                    margin-bottom: 10px;
                    text-shadow: 0 0 20px rgba(0, 255, 0, 0.5);
                }
                .subtitle { color: #888; margin-bottom: 20px; }
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
                    box-shadow: 0 4px 30px rgba(0, 255, 0, 0.2);
                    width: 100%;
                    max-width: 1280px;
                    margin: 0 auto;
                }
                /* 使用 padding-bottom 技巧保持 16:9 比例 */
                .video-wrapper {
                    position: relative;
                    width: 100%;
                    padding-bottom: 56.25%; /* 16:9 = 9/16 = 0.5625 = 56.25% */
                    height: 0;
                    overflow: hidden;
                }
                .video-wrapper #video {
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
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
                    background: rgba(26, 26, 46, 0.9);
                    border-radius: 12px;
                    padding: 20px;
                    margin-bottom: 15px;
                    border: 1px solid #333355;
                    backdrop-filter: blur(10px);
                }
                .panel h2 { 
                    color: #00ff00; 
                    margin-bottom: 15px; 
                    font-size: 18px;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }
                .metrics { display: flex; gap: 15px; }
                .metric {
                    flex: 1;
                    text-align: center;
                    background: rgba(0,0,0,0.3);
                    padding: 15px;
                    border-radius: 8px;
                    transition: transform 0.3s;
                }
                .metric:hover { transform: scale(1.05); }
                .metric .label { color: #888; font-size: 12px; display: block; margin-bottom: 5px; }
                .metric .value { color: #00ff00; font-size: 28px; font-weight: bold; }
                .data-row {
                    display: flex;
                    justify-content: space-between;
                    padding: 10px 0;
                    border-bottom: 1px solid rgba(255,255,255,0.1);
                    font-size: 14px;
                }
                .data-row:last-child { border-bottom: none; }
                .highlight { color: #00ff00; font-weight: bold; }
                .controls { display: flex; gap: 10px; margin-top: 15px; }
                button {
                    padding: 12px 24px;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                    font-size: 14px;
                    font-weight: 600;
                    transition: all 0.3s;
                }
                .btn-primary { 
                    background: linear-gradient(90deg, #00ff00, #00cc00); 
                    color: #000;
                    box-shadow: 0 4px 15px rgba(0, 255, 0, 0.3);
                }
                .btn-primary:hover { transform: translateY(-2px); }
                .btn-danger { 
                    background: linear-gradient(90deg, #ff4444, #cc0000); 
                    color: white;
                    box-shadow: 0 4px 15px rgba(255, 68, 68, 0.3);
                }
                .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
                #topview { background: #1a1a2e; border-radius: 8px; width: 100%; }
                .status { 
                    display: flex; 
                    align-items: center; 
                    gap: 10px; 
                    margin-bottom: 20px;
                    padding: 15px;
                    background: rgba(26, 26, 46, 0.8);
                    border-radius: 12px;
                }
                .status-dot { width: 12px; height: 12px; border-radius: 50%; background: #ff4444; }
                .status-dot.online { background: #00ff00; box-shadow: 0 0 15px #00ff00; animation: pulse 2s infinite; }
                @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
                .badge {
                    padding: 4px 8px;
                    border-radius: 4px;
                    font-size: 11px;
                    font-weight: bold;
                }
                .badge-mps { background: rgba(0, 255, 0, 0.2); color: #00ff00; }
                .badge-cpu { background: rgba(255, 165, 0, 0.2); color: orange; }
                .loading { text-align: center; padding: 20px; color: #888; }
            </style>
        </head>
        <body>
            <h1>🎥 摄像头感知系统 v2</h1>
            <p class="subtitle">Apple M5 Pro 优化版 | MPS 加速 | 实时空间计量</p>
            
            <div class="status">
                <div class="status-dot" id="statusDot"></div>
                <span id="statusText">未连接</span>
                <span class="badge" id="deviceBadge">检测中...</span>
                <span id="fpsDisplay" style="margin-left: auto; color: #888;">FPS: 0</span>
            </div>
            
            <div class="container">
                <div class="video-panel">
                    <div class="video-container">
                        <div class="video-wrapper">
                            <img id="video" style="width:100%;height:100%;object-fit:contain;background:#000;">
                            <canvas id="overlay"></canvas>
                        </div>
                    </div>
                    <div class="controls">
                        <button class="btn-primary" onclick="startCamera()" id="startBtn">▶ 启动摄像头</button>
                        <button class="btn-danger" onclick="stopCamera()" id="stopBtn" disabled>⏹ 停止</button>
                    </div>
                    
                    <!-- 校准面板 -->
                    <div class="panel" style="margin-top:10px;background:rgba(26,26,46,0.8);padding:15px;border-radius:8px;">
                        <h3 style="color:#00ff00;margin-bottom:10px;font-size:14px;">📐 距离/身高校准</h3>
                        <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;">
                            <div>
                                <label style="font-size:11px;color:#888;">已知距离(m):</label>
                                <input type="number" id="knownDistance" step="0.1" min="0.5" max="10" value="2.0" 
                                    style="width:70px;background:#1a1a2e;border:1px solid #333;color:#fff;padding:4px;border-radius:4px;">
                            </div>
                            <div>
                                <label style="font-size:11px;color:#888;">已知身高(cm):</label>
                                <input type="number" id="knownHeight" step="1" min="100" max="250" value="170" 
                                    style="width:70px;background:#1a1a2e;border:1px solid #333;color:#fff;padding:4px;border-radius:4px;">
                            </div>
                            <button onclick="calibrate()" style="background:#4a4a6a;border:none;color:#fff;padding:6px 12px;border-radius:4px;cursor:pointer;">校准</button>
                            <button onclick="resetCalibration()" style="background:#4a2a2a;border:none;color:#fff;padding:6px 12px;border-radius:4px;cursor:pointer;">重置</button>
                        </div>
                        <div id="calibStatus" style="margin-top:8px;font-size:11px;color:#888;"></div>
                    </div>
                </div>
                
                <div class="data-panel">
                    <div class="panel">
                        <h2>📊 实时检测</h2>
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
                        <div id="personData"><div class="loading">等待检测...</div></div>
                    </div>
                    
                    <div class="panel" id="handPanel">
                        <h2>✋ 手部数据</h2>
                        <div id="handData"><div class="loading">等待检测...</div></div>
                    </div>
                    
                    <div class="panel">
                        <h2>🗺️ 顶视图</h2>
                        <canvas id="topview" width="310" height="250"></canvas>
                        <div style="display: flex; gap: 15px; margin-top: 10px; justify-content: center; font-size: 12px;">
                            <span style="display: flex; align-items: center; gap: 5px;">
                                <span style="width: 10px; height: 10px; background: #00ff00; border-radius: 50%;"></span> 人
                            </span>
                            <span style="display: flex; align-items: center; gap: 5px;">
                                <span style="width: 10px; height: 10px; background: #0088ff; border-radius: 50%;"></span> 手
                            </span>
                            <span style="display: flex; align-items: center; gap: 5px;">
                                <span style="width: 10px; height: 10px; background: #fff; border-radius: 50%;"></span> 摄像头
                            </span>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                let videoStream = null;
                let dataWs = null;
                let isRunning = false;
                
                function updateStatus(online, fps, device) {
                    document.getElementById('statusDot').className = 'status-dot' + (online ? ' online' : '');
                    document.getElementById('statusText').textContent = online ? '摄像头已连接' : '未连接';
                    document.getElementById('fpsDisplay').textContent = 'FPS: ' + fps;
                    if (device) {
                        document.getElementById('deviceBadge').textContent = device;
                        document.getElementById('deviceBadge').className = 'badge ' + (device.includes('MPS') ? 'badge-mps' : 'badge-cpu');
                    }
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
                        updateStatus(false, 0, '');
                    } catch (e) {
                        console.error(e);
                    }
                }
                

                async function calibrate() {
                    const distance = parseFloat(document.getElementById('knownDistance').value);
                    const height = parseFloat(document.getElementById('knownHeight').value);
                    
                    try {
                        const r = await fetch('/api/calibrate', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({
                                person_index: 0,
                                known_distance: distance,
                                known_height: height
                            })
                        });
                        const d = await r.json();
                        if (d.status === 'success') {
                            document.getElementById('calibStatus').innerHTML = 
                                '<span style="color:#00ff00">✓ 校准成功</span> ' + 
                                '距离系数:' + (d.calibration_status?.distance_scale || 1).toFixed(3) + ' ' +
                                '身高系数:' + (d.calibration_status?.height_scale || 1).toFixed(3);
                        } else {
                            document.getElementById('calibStatus').innerHTML = 
                                '<span style="color:#ff4444">✗ ' + d.message + '</span>';
                        }
                    } catch (e) {
                        document.getElementById('calibStatus').innerHTML = 
                            '<span style="color:#ff4444">✗ 校准失败: ' + e.message + '</span>';
                    }
                }

                async function resetCalibration() {
                    try {
                        const r = await fetch('/api/calibration/reset', {method: 'POST'});
                        const d = await r.json();
                        if (d.status === 'success') {
                            document.getElementById('calibStatus').innerHTML = 
                                '<span style="color:#00ff00">✓ 校准已重置</span>';
                        }
                    } catch (e) {
                        console.error(e);
                    }
                }

                function startVideoStream() {
                    videoStream = document.getElementById('video');
                    videoStream.src = '/ws/video?t=' + Date.now();
                    videoStream.onload = () => {
                        const canvas = document.getElementById('overlay');
                        canvas.width = videoStream.naturalWidth || 1280;
                        canvas.height = videoStream.naturalHeight || 720;
                    };
                }
                
                function connectWebSocket() {
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    dataWs = new WebSocket(protocol + '//' + window.location.host + '/ws/data');
                    
                    dataWs.onopen = () => {
                        console.log('WebSocket connected');
                    };
                    
                    dataWs.onmessage = (event) => {
                        const data = JSON.parse(event.data);
                        
                        if (data.fps) {
                            document.getElementById('fpsDisplay').textContent = 'FPS: ' + data.fps;
                        }
                        
                        if (data.device) {
                            updateStatus(true, data.fps || 0, data.device);
                        }
                        
                        document.getElementById('personCount').textContent = data.persons?.length || 0;
                        document.getElementById('handCount').textContent = data.hands?.length || 0;
                        
                        const personDiv = document.getElementById('personData');
                        if (data.persons && data.persons.length > 0) {
                            personDiv.innerHTML = data.persons.map((p, i) => {
                                const conf = p.confidence ? (p.confidence * 100).toFixed(0) : '?';
                                const angle = p.angle_h !== undefined ? p.angle_h : '?';
                                const topX = p.topview?.x !== undefined ? p.topview.x.toFixed(0) : '?';
                                const topY = p.topview?.y !== undefined ? p.topview.y.toFixed(0) : '?';
                                return `
                                <div class="data-row">
                                    <span>👤 人${i+1} <span style="color:#888;font-size:11px">(${conf}%)</span></span>
                                    <span><span class="highlight">${p.distance}m</span> | <span class="highlight">${p.height}cm</span></span>
                                </div>
                                <div class="data-row" style="font-size:11px;color:#888">
                                    <span>角度: ${angle}° | 顶视: (${topX}, ${topY})</span>
                                </div>
                                <div class="data-row" style="font-size:11px;color:#666">
                                    <span>方法: ${p.estimate_method || 'body'} | 速度: ${p.velocity || 0} m/s</span>
                                </div>
                                <div class="data-row" style="font-size:11px;color:#555">
                                    <span>头部置信度: ${p.head_confidence || 0} | 不确定性: ${p.uncertainty || 0}</span>
                                </div>
                            `}).join('');
                        } else {
                            personDiv.innerHTML = '<div class="loading">未检测到人体</div>';
                        }

                        const handDiv = document.getElementById('handData');
                        if (data.hands && data.hands.length > 0) {
                            handDiv.innerHTML = data.hands.map((h, i) => {
                                const conf = h.confidence ? (h.confidence * 100).toFixed(0) : '?';
                                const estimated = h.estimated ? ' (估)' : '';
                                const topX = h.topview?.x !== undefined ? h.topview.x.toFixed(0) : '?';
                                const topY = h.topview?.y !== undefined ? h.topview.y.toFixed(0) : '?';
                                return `
                                <div class="data-row">
                                    <span>✋ 手${i+1} (${h.hand_type || '?'})${estimated} <span style="color:#888;font-size:11px">(${conf}%)</span></span>
                                    <span><span class="highlight">${h.size}cm</span> | <span class="highlight">${h.distance}m</span></span>
                                </div>
                                <div class="data-row" style="font-size:11px;color:#888">
                                    <span>顶视: (${topX}, ${topY})</span>
                                </div>
                            `}).join('');
                        } else {
                            handDiv.innerHTML = '<div class="loading">未检测到手部</div>';
                        }
                        
                        drawTopView(data);
                        drawOverlay(data);
                    };
                    
                    dataWs.onclose = () => {
                        if (isRunning) setTimeout(connectWebSocket, 1000);
                    };
                }
                
                function drawTopView(data) {
                    const canvas = document.getElementById('topview');
                    const ctx = canvas.getContext('2d');
                    ctx.fillStyle = '#1a1a2e';
                    ctx.fillRect(0, 0, canvas.width, canvas.height);

                    // 绘制网格
                    ctx.strokeStyle = '#333355';
                    ctx.lineWidth = 1;
                    for (let x = 0; x < canvas.width; x += 25) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke(); }
                    for (let y = 0; y < canvas.height; y += 25) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke(); }

                    // 绘制摄像头位置（底部中心）
                    const camX = canvas.width / 2;
                    const camY = canvas.height - 15;
                    ctx.fillStyle = '#ffffff';
                    ctx.beginPath();
                    ctx.arc(camX, camY, 8, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.fillStyle = '#888';
                    ctx.font = '10px Arial';
                    ctx.fillText('摄像头', camX - 18, camY + 18);

                    // 绘制人体位置
                    data.persons?.forEach((p, i) => {
                        if (p.topview && p.topview.x !== undefined) {
                            const tx = camX + (p.topview.x - 400) * 2;
                            const ty = camY - (300 - p.topview.y) * 2;
                            
                            if (tx >= 0 && tx <= canvas.width && ty >= 0 && ty <= canvas.height) {
                                ctx.fillStyle = '#00ff00';
                                ctx.beginPath();
                                ctx.arc(tx, ty, 10, 0, Math.PI * 2);
                                ctx.fill();
                                ctx.fillStyle = '#00ff00';
                                ctx.font = '10px Arial';
                                ctx.fillText(p.distance + 'm', tx - 10, ty - 15);
                            }
                        }
                    });

                    // 绘制手部位置
                    data.hands?.forEach((h, i) => {
                        if (h.topview && h.topview.x !== undefined) {
                            const tx = camX + (h.topview.x - 400) * 2;
                            const ty = camY - (300 - h.topview.y) * 2;
                            if (tx >= 0 && tx <= canvas.width && ty >= 0 && ty <= canvas.height) {
                                ctx.fillStyle = h.estimated ? '#ff8800' : '#0088ff';
                                ctx.beginPath();
                                ctx.arc(tx, ty, 7, 0, Math.PI * 2);
                                ctx.fill();
                            }
                        }
                    });
                }

                function drawOverlay(data) {
                    const canvas = document.getElementById('overlay');
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);

                    data.persons?.forEach(p => {
                        const [x, y, w, h] = p.bbox;
                        ctx.strokeStyle = '#00ff00';
                        ctx.lineWidth = 2;
                        ctx.strokeRect(x, y, w, h);

                        const conf = p.confidence ? (p.confidence * 100).toFixed(0) : '?';
                        const angle = p.angle_h !== undefined ? p.angle_h : '?';

                        // 绘制关键点
                        if (p.keypoints) {
                            Object.values(p.keypoints).forEach(kp => {
                                ctx.fillStyle = '#ff0000';
                                ctx.beginPath();
                                ctx.arc(kp[0], kp[1], 3, 0, Math.PI * 2);
                                ctx.fill();
                            });
                            
                            // 绘制头部框（基于眼睛和耳朵位置）
                            const leftEye = p.keypoints['L_eye'];
                            const rightEye = p.keypoints['R_eye'];
                            const leftEar = p.keypoints['L_ear'];
                            const rightEar = p.keypoints['R_ear'];
                            const nose = p.keypoints['nose'];
                            
                            if (leftEye && rightEye) {
                                // 计算头部区域
                                const eyeCenterX = (leftEye[0] + rightEye[0]) / 2;
                                const eyeCenterY = (leftEye[1] + rightEye[1]) / 2;
                                const eyeDist = Math.abs(rightEye[0] - leftEye[0]);
                                const headSize = eyeDist * 2.5;  // 头部大约是眼距的 2.5 倍
                                
                                ctx.strokeStyle = '#ff00ff';  // 紫色头部框
                                ctx.lineWidth = 2;
                                ctx.strokeRect(
                                    eyeCenterX - headSize/2, 
                                    eyeCenterY - headSize/2, 
                                    headSize, 
                                    headSize
                                );
                            }
                        }
                        
                        // 绘制标签
                        ctx.fillStyle = '#00ff00';
                        ctx.font = 'bold 12px Arial';
                        ctx.fillText(`人 ${conf}% | ${angle}°`, x + 2, y - 10);
                        ctx.font = '11px Arial';
                        ctx.fillText(`距离:${p.distance}m 身高:${p.height}cm`, x + 2, y + 8);
                    });

                    data.hands?.forEach(h => {
                        const [x, y, w, h2] = h.bbox;
                        ctx.strokeStyle = h.estimated ? '#ff8800' : '#0088ff';
                        ctx.lineWidth = 2;
                        ctx.strokeRect(x, y, w, h2);

                        const conf = h.confidence ? (h.confidence * 100).toFixed(0) : '?';
                        const label = h.estimated ? `手(估) ${conf}%` : `手 ${conf}%`;
                        ctx.fillStyle = h.estimated ? '#ff8800' : '#0088ff';
                        ctx.font = 'bold 12px Arial';
                        ctx.fillText(label, x + 2, y - 10);
                    });
                }
            </script>            </script>
        </body>
    </html>
    """


@app.get("/api/status")
async def get_status():
    return {
        "camera_opened": state.camera_opened,
        "fps": state.fps,
        "yolo_available": state.yolo_available,
        "mediapipe_available": state.mediapipe_available,
        "calibrated": state.calibrated
    }


@app.post("/api/camera/start")
async def start_camera(config: CameraConfig = None):
    global capture_task, frame_lock
    
    if config is None:
        config = CameraConfig()

    if state.camera:
        state.camera.release()

    state.camera = cv2.VideoCapture(config.camera_id, cv2.CAP_AVFOUNDATION)
    state.camera.set(cv2.CAP_PROP_FRAME_WIDTH, min(config.resolution[0], 1280))
    state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, min(config.resolution[1], 720))
    state.camera.set(cv2.CAP_PROP_FPS, min(config.fps, 30))

    # 预热
    for _ in range(5):
        state.camera.read()
        await asyncio.sleep(0.05)

    if state.camera.isOpened():
        state.camera_opened = True
        frame_lock = asyncio.Lock()
        
        # 启动后台读帧任务
        loop = asyncio.get_event_loop()
        capture_task = loop.create_task(capture_loop())
        
        logger.info(f"✅ Camera opened, capture task started")
        return {"status": "success", "message": "Camera opened"}
    else:
        state.camera = None
        return {"status": "error", "message": "Failed to open camera"}


@app.post("/api/camera/stop")
async def stop_camera():
    global capture_running, capture_task
    
    state.camera_opened = False
    capture_running = False
    
    # 等待后台任务停止
    if capture_task:
        capture_task.cancel()
        try:
            await capture_task
        except asyncio.CancelledError:
            pass
    
    if state.camera:
        state.camera.release()
        state.camera = None
        logger.info("Camera stopped")
        return {"status": "success", "message": "Camera stopped"}
    return {"status": "info", "message": "Camera not opened"}


@app.get("/ws/video")
async def video_stream():
    """MJPEG 视频流 - 直接读取摄像头"""
    logger.info("视频流连接开始")
    
    async def generate():
        frame_count = 0
        while state.camera_opened and state.camera:
            ret, frame = state.camera.read()
            if not ret or frame is None:
                await asyncio.sleep(0.01)
                continue

            try:
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                frame_count += 1
            except Exception as e:
                logger.error(f"视频编码错误：{e}")
                break

            state.update_fps()
            await asyncio.sleep(1/30)
        
        logger.info(f"视频流结束，已发送 {frame_count} 帧")

    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')


@app.websocket("/ws/data")
async def websocket_data(websocket: WebSocket):
    await websocket.accept()

    device_info = "MPS" if state.yolo_available else "CPU (模拟)"
    logger.info(f"数据 WebSocket 连接，使用 {device_info}")

    try:
        while state.camera_opened and state.camera:
            ret, frame = state.camera.read()
            if not ret or frame is None:
                await asyncio.sleep(0.01)
                continue

            data = {
                "persons": [],
                "hands": [],
                "fps": state.fps,
                "device": device_info,
                "timestamp": time.time()
            }

            # 使用真实检测
            if state.detector and state.yolo_available:
                try:
                    result = state.detector.detect(frame, smooth=True)
                    
                    # 计算人体空间信息
                    for person in result.get("persons", []):
                        if state.spatial_calc:
                            # 获取上一帧的距离用于速度约束
                            track_id = person.get("track_id", 0)
                            prev_dist = state.prev_distances.get(track_id)
                            
                            metrics = state.spatial_calc.calc_person_metrics(
                                person, 
                                prev_distance=prev_dist,
                                dt=0.05  # 50ms 间隔
                            )
                            person["distance"] = metrics.get("distance", 0)
                            person["height"] = metrics.get("height", 0)
                            person["topview"] = metrics.get("topview", {})
                            
                            # 计算角度（相对于画面中心）
                            bbox = person.get("bbox", [0, 0, 0, 0])
                            center_x = bbox[0] + bbox[2] / 2
                            frame_center_x = frame.shape[1] / 2
                            # 水平角度（假设焦距 1200）
                            angle_h = np.arctan((center_x - frame_center_x) / 1200) * 180 / np.pi
                            person["angle_h"] = round(angle_h, 1)
                            person["estimate_method"] = metrics.get("estimate_method", "body")
                            person["head_confidence"] = metrics.get("head_confidence", 0)
                            
                            # 保存当前距离
                            state.prev_distances[track_id] = person["distance"]
                        else:
                            # 没有标定时使用简单估计
                            bbox = person.get("bbox", [0, 0, 0, 0])
                            # 假设平均肩宽 45cm，用像素宽度估计距离
                            pixel_width = bbox[2]
                            if pixel_width > 0:
                                distance = (0.45 * 1200) / pixel_width  # 简单估计
                                person["distance"] = round(distance, 2)
                            else:
                                person["distance"] = 0
                            person["height"] = 0
                            person["topview"] = {"x": 0, "y": 0}
                            
                            # 角度
                            center_x = bbox[0] + bbox[2] / 2
                            frame_center_x = frame.shape[1] / 2
                            angle_h = np.arctan((center_x - frame_center_x) / 1200) * 180 / np.pi
                            person["angle_h"] = round(angle_h, 1)
                        
                        data["persons"].append(person)
                    
                    # 计算手部空间信息
                    for hand in result.get("hands", []):
                        if state.spatial_calc:
                            metrics = state.spatial_calc.calc_hand_metrics(hand)
                            hand["distance"] = metrics.get("distance", 0)
                            hand["size"] = metrics.get("size", 0)
                            hand["topview"] = metrics.get("topview", {})
                        else:
                            # 简单估计
                            bbox = hand.get("bbox", [0, 0, 0, 0])
                            pixel_size = max(bbox[2], bbox[3])
                            if pixel_size > 0:
                                # 假设手长约 18cm
                                distance = (0.18 * 1200) / pixel_size
                                hand["distance"] = round(distance, 2)
                            else:
                                hand["distance"] = 0
                            hand["size"] = 18  # 假设 18cm
                            hand["topview"] = {"x": 0, "y": 0}
                        
                        data["hands"].append(hand)
                        
                except Exception as e:
                    logger.debug(f"检测错误：{e}")

            try:
                await websocket.send_json(data)
            except:
                break

            await asyncio.sleep(0.05)

    except Exception as e:
        logger.error(f"WebSocket error: {e}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎥 摄像头感知系统 v2 - M5 Pro 优化版")
    print("="*60)
    print(f"\n检测器状态:")
    print(f"  YOLO: {'✅' if state.yolo_available else '⏳ 未安装'}")
    print(f"  MediaPipe: {'✅' if state.mediapipe_available else '⏳ 未安装'}")
    print(f"  标定：{'✅' if state.calibrated else '⏳ 未标定'}")
    print(f"\n访问地址:")
    print(f"  http://localhost:8100")
    print(f"\n按 Ctrl+C 停止服务\n")
    
    uvicorn.run(app, host='0.0.0.0', port=8100, log_level="info")
