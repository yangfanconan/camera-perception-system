"""
完整后端 API 实现
功能：
1. 完整的 REST API 端点
2. WebSocket 实时推送
3. 配置管理
4. 数据导出
"""

import asyncio
import json
import cv2
import numpy as np
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from pathlib import Path
from loguru import logger
import uvicorn
from datetime import datetime
import io
import base64

# 导入项目模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms import (
    CombinedDetector,
    CombinedDetectorEnhanced,
    SpatialCalculator,
    SpatialCalculatorEnhanced,
    CalibrationParams,
    CameraCalibrator,
    visualize_detections
)
from data.video_capture import VideoCapture
from data.video_encoder import MJPEGStreamer, FLVStreamer, StreamConfig


# ==================== 数据模型 ====================

class CameraConfig(BaseModel):
    """相机配置"""
    camera_id: int = 0
    resolution: List[int] = [1920, 1080]
    fps: int = 20


class DetectionConfig(BaseModel):
    """检测配置"""
    conf_threshold: float = 0.5
    iou_threshold: float = 0.45
    smooth_enabled: bool = True


class SpatialConfig(BaseModel):
    """空间计量配置"""
    ref_shoulder_width: float = 0.45
    camera_height: float = 1.8
    pitch_angle: float = 30.0
    topview_scale: float = 10.0


class SystemSettings(BaseModel):
    """系统设置"""
    camera: CameraConfig = CameraConfig()
    detection: DetectionConfig = DetectionConfig()
    spatial: SpatialConfig = SpatialConfig()


class CalibrationRequest(BaseModel):
    """标定请求"""
    checkerboard: List[int] = [9, 6]
    square_size: float = 25.0


class ExtrinsicsConfig(BaseModel):
    """外参配置"""
    height: float = 1.8
    pitch_angle: float = 30.0


class CalibrationInput(BaseModel):
    """校准输入"""
    known_height: Optional[float] = None
    known_distance: Optional[float] = None


# ==================== FastAPI 应用 ====================

app = FastAPI(
    title="摄像头实时感知系统 API",
    description="基于 YOLO+MediaPipe 的人体/手部检测与空间计量系统",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 全局状态 ====================

class SystemState:
    """系统全局状态"""
    
    def __init__(self):
        # 硬件
        self.camera: Optional[VideoCapture] = None
        self.detector: Optional[CombinedDetectorEnhanced] = None
        self.spatial_calc: Optional[SpatialCalculatorEnhanced] = None
        self.calibrator: Optional[CameraCalibrator] = None
        
        # 状态
        self.camera_opened = False
        self.calibrated = False
        self.running = False
        
        # 配置
        self.settings = SystemSettings()
        
        # 性能统计
        self.fps_counter = 0
        self.last_frame_time = 0
        self.frame_count = 0
        
        # WebSocket 连接
        self.websocket_video: List[WebSocket] = []
        self.websocket_data: List[WebSocket] = []
        
        # 标定缓存
        self.calibration_images: List[np.ndarray] = []
        self.calibration_corners: List[np.ndarray] = []
        
        # 检测数据历史
        self.detection_history: List[Dict] = []
    
    async def initialize(self):
        """初始化系统"""
        logger.info("Initializing system...")
        
        # 初始化检测器
        self.detector = CombinedDetectorEnhanced(
            pose_model_path='models/yolov8n-pose.pt',
            conf_threshold=self.settings.detection.conf_threshold
        )
        
        # 尝试加载标定参数
        calib_path = Path('calibration_data/calib_params.json')
        if calib_path.exists():
            self.load_calibration(str(calib_path))
        
        # 初始化标定器
        self.calibrator = CameraCalibrator(
            checkerboard_size=tuple(self.settings.camera.resolution)
        )
        
        logger.info("System initialized")
    
    def load_calibration(self, filepath: str):
        """加载标定参数"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            calib_params = CalibrationParams.from_dict(data)
            self.spatial_calc = SpatialCalculatorEnhanced(calib_params)
            self.spatial_calc.set_camera_extrinsics(
                height=self.settings.spatial.camera_height,
                pitch_angle=self.settings.spatial.pitch_angle
            )
            self.calibrated = True
            
            logger.info(f"Calibration loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
    
    def start_camera(self, camera_id: int = 0) -> bool:
        """启动摄像头"""
        try:
            if self.camera is not None:
                self.camera.release()
            
            self.camera = VideoCapture(
                camera_id=camera_id,
                resolution=tuple(self.settings.camera.resolution),
                fps=self.settings.camera.fps
            )
            
            if self.camera.open():
                self.camera_opened = True
                logger.info(f"Camera {camera_id} started")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            return False
    
    def stop_camera(self):
        """停止摄像头"""
        if self.camera:
            self.camera.release()
            self.camera = None
            self.camera_opened = False
            logger.info("Camera stopped")
    
    def update_fps(self):
        """更新 FPS 计数"""
        import time
        current_time = time.time()
        
        if current_time - self.last_frame_time >= 1.0:
            self.fps_counter = self.frame_count
            self.frame_count = 0
            self.last_frame_time = current_time
        
        self.frame_count += 1
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "camera_opened": self.camera_opened,
            "calibrated": self.calibrated,
            "fps": self.fps_counter,
            "persons_detected": len(self.detection_history[-1]["persons"]) if self.detection_history else 0,
            "hands_detected": len(self.detection_history[-1]["hands"]) if self.detection_history else 0
        }


# 全局状态实例
state = SystemState()


# ==================== API 路由 ====================

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    await state.initialize()


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理"""
    state.stop_camera()
    if state.detector:
        state.detector.close()


@app.get("/", response_class=HTMLResponse)
async def root():
    """根路径"""
    return """
    <html>
        <head>
            <title>摄像头实时感知系统</title>
            <style>
                body { font-family: Arial; background: #0f0f1a; color: white; padding: 40px; }
                h1 { color: #00ff00; }
                a { color: #0088ff; }
                .info { background: #1a1a2e; padding: 20px; border-radius: 8px; margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>🎥 摄像头实时感知 + 空间计量系统</h1>
            <div class="info">
                <h2>API 文档</h2>
                <ul>
                    <li><a href="/docs">Swagger UI</a> - OpenAPI 文档</li>
                    <li><a href="/redoc">ReDoc</a> - 美观文档</li>
                </ul>
            </div>
            <div class="info">
                <h2>前端页面</h2>
                <p>访问 <a href="http://localhost:5173">http://localhost:5173</a></p>
            </div>
            <div class="info">
                <h2>系统状态</h2>
                <pre id="status">加载中...</pre>
            </div>
            <script>
                fetch('/api/status')
                    .then(r => r.json())
                    .then(d => document.getElementById('status').textContent = JSON.stringify(d, null, 2));
            </script>
        </body>
    </html>
    """


# ----- 系统状态 API -----

@app.get("/api/status")
async def get_status():
    """获取系统状态"""
    return state.get_status()


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ----- 相机控制 API -----

@app.post("/api/camera/start")
async def start_camera(config: Optional[CameraConfig] = None):
    """启动摄像头"""
    if config is None:
        config = CameraConfig()
    
    # 更新配置
    state.settings.camera = config
    
    if state.start_camera(camera_id=config.camera_id):
        return {"status": "success", "message": f"Camera {config.camera_id} started"}
    else:
        raise HTTPException(status_code=500, detail="Failed to open camera")


@app.post("/api/camera/stop")
async def stop_camera():
    """停止摄像头"""
    state.stop_camera()
    return {"status": "success", "message": "Camera stopped"}


@app.get("/api/camera/config")
async def get_camera_config():
    """获取相机配置"""
    return state.settings.camera.dict()


@app.post("/api/camera/config")
async def update_camera_config(config: CameraConfig):
    """更新相机配置"""
    state.settings.camera = config
    
    # 如果摄像头已启动，重启以应用新配置
    if state.camera_opened:
        state.stop_camera()
        state.start_camera(camera_id=config.camera_id)
    
    return {"status": "success", "config": config.dict()}


# ----- 标定 API -----

@app.get("/api/calibration/status")
async def calibration_status():
    """获取标定状态"""
    return {
        "calibrated": state.calibrated,
        "spatial_calc_initialized": state.spatial_calc is not None,
        "calibration_records": len(state.calibrator.calib_params.dist_coeffs) if state.calibrator and state.calibrator.calib_params else 0
    }


@app.post("/api/calibration/load")
async def load_calibration(filepath: str):
    """加载标定参数"""
    try:
        state.load_calibration(filepath)
        return {"status": "success", "message": "Calibration loaded"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/calibration/start_capture")
async def start_calibration_capture():
    """开始标定采集"""
    state.calibration_images = []
    state.calibration_corners = []
    return {"status": "success", "message": "Calibration capture started"}


@app.get("/api/calibration/capture_status")
async def capture_status():
    """获取采集状态"""
    return {
        "image_count": len(state.calibration_images),
        "collecting": len(state.calibration_images) < 15
    }


@app.post("/api/calibration/upload", status_code=201)
async def upload_calibration_images(images: List[UploadFile] = File(...)):
    """上传标定图片"""
    for image in images:
        content = await image.read()
        img_array = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is not None:
            # 检测棋盘格
            found, corners = state.calibrator.find_checkerboard_corners(img)
            if found:
                state.calibration_images.append(img)
                state.calibration_corners.append(corners)
    
    return {
        "status": "success",
        "uploaded": len(images),
        "valid": len(state.calibration_images)
    }


@app.post("/api/calibration/run")
async def run_calibration():
    """执行标定"""
    if len(state.calibration_images) < 3:
        raise HTTPException(status_code=400, detail="Need at least 3 valid images")
    
    try:
        # 准备物体点
        objpoints = []
        for _ in state.calibration_corners:
            objpoints.append(state.calibrator.prepare_object_points())
        
        # 执行标定
        gray_size = state.calibration_images[0].shape[:2]
        
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, state.calibration_corners, gray_size[::-1], None, None,
            flags=cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST
        )
        
        # 计算重投影误差
        mean_error = state.calibrator._calc_reprojection_error(
            objpoints, state.calibration_corners, rvecs, tvecs,
            camera_matrix, dist_coeffs
        )
        
        # 构建标定参数
        calib_params = CalibrationParams(
            fx=float(camera_matrix[0, 0]),
            fy=float(camera_matrix[1, 1]),
            cx=float(camera_matrix[0, 2]),
            cy=float(camera_matrix[1, 2]),
            dist_coeffs=dist_coeffs.flatten().tolist(),
            rotation_matrix=rvecs[0].flatten().tolist() if rvecs else [0, 0, 0],
            translation_vector=tvecs[0].flatten().tolist() if tvecs else [0, 0, 0],
            image_size=(gray_size[1], gray_size[0]),
            checkerboard_size=tuple(state.settings.camera.checkerboard),
            square_size=state.settings.calibration.square_size,
            reprojection_error=mean_error,
            num_images=len(state.calibration_images)
        )
        
        # 保存
        output_path = Path('calibration_data/calib_params.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(calib_params.to_dict(), f, indent=2)
        
        # 更新空间计算器
        state.spatial_calc = SpatialCalculatorEnhanced(calib_params)
        state.calibrated = True
        
        return {
            "success": True,
            "reprojection_error": mean_error,
            "num_images": len(state.calibration_images),
            **calib_params.to_dict()
        }
    
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/calibration/extrinsics")
async def save_extrinsics(config: ExtrinsicsConfig):
    """保存外参配置"""
    state.settings.spatial.camera_height = config.height
    state.settings.spatial.pitch_angle = config.pitch_angle
    
    if state.spatial_calc:
        state.spatial_calc.set_camera_extrinsics(config.height, config.pitch_angle)
    
    return {"status": "success", "extrinsics": config.dict()}


@app.post("/api/calibration/height")
async def calibrate_height(input: CalibrationInput):
    """使用已知身高校准"""
    if not input.known_height or not state.spatial_calc:
        raise HTTPException(status_code=400, detail="Invalid input or calculator not initialized")
    
    # 获取最新检测结果
    if state.detection_history:
        latest = state.detection_history[-1]
        if latest["persons"]:
            person = latest["persons"][0]
            state.spatial_calc.calibrate_with_known_height(
                person["bbox"],
                person.get("keypoints", {}),
                input.known_height
            )
            return {"status": "success", "message": "Height calibration applied"}
    
    raise HTTPException(status_code=400, detail="No person detected for calibration")


@app.post("/api/calibration/distance")
async def calibrate_distance(input: CalibrationInput):
    """使用已知距离校准"""
    if not input.known_distance or not state.spatial_calc:
        raise HTTPException(status_code=400, detail="Invalid input or calculator not initialized")
    
    # 获取最新检测结果
    if state.detection_history:
        latest = state.detection_history[-1]
        if latest["persons"]:
            person = latest["persons"][0]
            state.spatial_calc.calibrate_with_known_distance(
                person["bbox"],
                input.known_distance
            )
            return {"status": "success", "message": "Distance calibration applied"}
    
    raise HTTPException(status_code=400, detail="No person detected for calibration")


@app.post("/api/calibration/reset")
async def reset_calibration():
    """重置校准"""
    if state.spatial_calc:
        state.spatial_calc.reset_calibration()
    return {"status": "success", "message": "Calibration reset"}


# ----- 设置 API -----

@app.get("/api/settings")
async def get_settings():
    """获取系统设置"""
    return state.settings.dict()


@app.post("/api/settings")
async def update_settings(settings: SystemSettings):
    """更新系统设置"""
    state.settings = settings
    
    # 应用空间设置
    if state.spatial_calc:
        state.spatial_calc.set_camera_extrinsics(
            settings.spatial.camera_height,
            settings.spatial.pitch_angle
        )
    
    return {"status": "success", "settings": settings.dict()}


# ----- 数据导出 API -----

@app.get("/api/export/data")
async def export_data():
    """导出检测数据"""
    if not state.detection_history:
        raise HTTPException(status_code=404, detail="No data to export")
    
    # 生成 JSON
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "total_records": len(state.detection_history),
        "data": state.detection_history[-1000:]  # 最近 1000 条
    }
    
    # 创建响应
    return JSONResponse(
        content=export_data,
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename=detection_data_{datetime.now().strftime('%Y%m%d')}.json"
        }
    )


@app.get("/api/export/image")
async def export_annotated_image():
    """导出标注图像"""
    if not state.camera or not state.detector:
        raise HTTPException(status_code=400, detail="Camera or detector not initialized")
    
    # 读取帧
    ret, frame = state.camera.read()
    if not ret:
        raise HTTPException(status_code=500, detail="Failed to read frame")
    
    # 检测
    result = state.detector.detect(frame)
    
    # 可视化
    output = visualize_detections(frame, result)
    
    # 编码
    _, buffer = cv2.imencode('.jpg', output)
    
    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg",
        headers={"Content-Disposition": "attachment; filename=annotated_image.jpg"}
    )


# ==================== WebSocket ====================

@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    """WebSocket 视频流推送"""
    await websocket.accept()
    state.websocket_video.append(websocket)
    
    logger.info("WebSocket video connection established")
    
    streamer = MJPEGStreamer(quality=85)
    
    try:
        while state.camera_opened:
            if state.camera is None:
                await asyncio.sleep(0.1)
                continue
            
            # 读取帧
            ret, frame = state.camera.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue
            
            # 编码
            data = streamer.stream_frame(frame)
            
            # 发送
            try:
                await websocket.send_bytes(data)
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket send error: {e}")
                break
            
            # 更新 FPS
            state.update_fps()
            
            # 帧率控制
            await asyncio.sleep(1 / state.settings.camera.fps)
    
    except WebSocketDisconnect:
        logger.info("WebSocket video disconnected")
    finally:
        if websocket in state.websocket_video:
            state.websocket_video.remove(websocket)


@app.websocket("/ws/data")
async def websocket_data(websocket: WebSocket):
    """WebSocket 数据推送"""
    await websocket.accept()
    state.websocket_data.append(websocket)
    
    logger.info("WebSocket data connection established")
    
    last_time = datetime.now().timestamp()
    
    try:
        while state.camera_opened:
            if state.camera is None or state.detector is None:
                await asyncio.sleep(0.1)
                continue
            
            # 读取帧
            ret, frame = state.camera.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue
            
            current_time = datetime.now().timestamp()
            
            # 检测
            result = state.detector.detect(frame, smooth=state.settings.detection.smooth_enabled)
            
            # 空间计量
            metrics = {
                "persons": [],
                "hands": [],
                "frame_shape": list(result.get("frame_shape", frame.shape)) if isinstance(result, dict) else frame.shape,
                "timestamp": current_time,
                "timestamp_diff": (current_time - last_time) * 1000
            }
            
            if state.spatial_calc:
                for person in result.get("persons", []):
                    person_metrics = state.spatial_calc.calc_person_metrics(person)
                    metrics["persons"].append(person_metrics)
                
                for hand in result.get("hands", []):
                    hand_metrics = state.spatial_calc.calc_hand_metrics(hand)
                    metrics["hands"].append(hand_metrics)
            else:
                metrics["persons"] = result.get("persons", [])
                metrics["hands"] = result.get("hands", [])
            
            # 保存历史
            state.detection_history.append(metrics)
            if len(state.detection_history) > 1000:
                state.detection_history.pop(0)
            
            # 发送
            try:
                await websocket.send_json(metrics)
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket send error: {e}")
                break
            
            last_time = current_time
            
            # 数据推送频率（20Hz）
            await asyncio.sleep(0.05)
    
    except WebSocketDisconnect:
        logger.info("WebSocket data disconnected")
    finally:
        if websocket in state.websocket_data:
            state.websocket_data.remove(websocket)


# ==================== 主程序 ====================

def main():
    """启动服务"""
    import argparse
    
    parser = argparse.ArgumentParser(description='FastAPI 服务')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='监听地址')
    parser.add_argument('--port', type=int, default=8000, help='端口号')
    parser.add_argument('--reload', action='store_true', help='启用热重载')
    
    args = parser.parse_args()
    
    # 设置运行状态
    state.running = True
    
    # 启动服务
    logger.info(f"Starting server at {args.host}:{args.port}")
    
    uvicorn.run(
        "src.api.main_full:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == '__main__':
    main()
