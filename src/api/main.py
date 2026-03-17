"""
FastAPI Web 后端
功能：
1. 视频流推送（HTTP-FLV/WebSocket）
2. 检测结果推送（WebSocket）
3. 配置管理 API
"""

import asyncio
import json
import cv2
import numpy as np
import base64
import time
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from pathlib import Path
from loguru import logger
import uvicorn
import traceback

# 导入项目模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms import (
    CombinedDetector,
    CameraCalibrator,
    CalibrationParams,
    visualize_detections
)
from data.video_capture import VideoCapture
from utils.constants import (
    DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT,
    CHECKERBOARD_DEFAULT_SIZE, CHECKERBOARD_DEFAULT_SQUARE_SIZE,
    DEFAULT_CAMERA_HEIGHT, DATA_PUSH_RATE, DATA_PUSH_INTERVAL
)


# ==================== 数据模型 ====================

class CameraConfig(BaseModel):
    """相机配置"""
    camera_id: int = 0
    resolution: List[int] = [DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT]
    fps: int = 20


class CalibrationConfig(BaseModel):
    """标定配置"""
    checkerboard: List[int] = list(CHECKERBOARD_DEFAULT_SIZE)
    square_size: float = CHECKERBOARD_DEFAULT_SQUARE_SIZE
    camera_height: float = DEFAULT_CAMERA_HEIGHT
    pitch_angle: float = 30.0




class HeadSizeConfig(BaseModel):
    """头部尺寸配置"""
    ref_head_width: float = 0.15
    ref_eye_distance: float = 0.063
    ref_ear_distance: float = 0.145
    ref_eye_nose_distance: float = 0.035
    ref_head_height: float = 0.22


class KalmanConfig(BaseModel):
    """卡尔曼滤波配置"""
    max_speed: float = 3.0
    process_noise: float = 0.1
    measurement_noise: float = 0.3


class CloseRangeConfig(BaseModel):
    """近距离估计配置"""
    threshold: float = 1.5
    ultra_close_threshold: float = 0.5
    head_weight: float = 0.7
    body_weight: float = 0.3
    use_perspective_correction: bool = True


class SpatialConfig(BaseModel):
    """空间计量配置"""
    ref_shoulder_width: float = 0.45
    ref_hand_length: float = 0.18
    head: HeadSizeConfig = HeadSizeConfig()
    kalman: KalmanConfig = KalmanConfig()
    close_range: CloseRangeConfig = CloseRangeConfig()

class SystemStatus(BaseModel):
    """系统状态"""
    camera_opened: bool = False
    calibrated: bool = False
    fps: float = 0.0
    persons_detected: int = 0
    hands_detected: int = 0


# ==================== FastAPI 应用 ====================

app = FastAPI(
    title="摄像头实时感知系统",
    description="基于 YOLO+MediaPipe 的人体/手部检测与空间计量",
    version="0.1.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 全局异常处理 ====================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    from utils.exceptions import get_error_handler
    
    error_handler = get_error_handler()
    context = f"{request.method} {request.url.path}"
    error_info = error_handler.handle(exc, context)
    
    # 记录详细错误信息
    logger.error(f"Global exception: {error_info['message']}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "error_code": error_info['error_code'],
            "message": error_info['message'],
            "details": error_info.get('details', {})
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "error_code": exc.status_code,
            "message": exc.detail
        }
    )


# ==================== 全局状态 ====================

class SystemState:
    """系统全局状态"""

    def __init__(self):
        self.camera: Optional[VideoCapture] = None
        self.detector: Optional[CombinedDetector] = None
        self.spatial_calc: Optional[SpatialCalculator] = None
        self.calibrator: Optional[CameraCalibrator] = None
        self.calibrated = False
        self.running = False
        self.fps_counter = 0
        self.last_frame_time = 0

        # 标定数据
        self.calibration_images: List[np.ndarray] = []
        self.calibration_corners: List[np.ndarray] = []

        # WebSocket 连接
        self.websocket_connections: List[WebSocket] = []
    
    async def initialize(self):
        """初始化系统"""
        logger.info("Initializing system...")

        # 初始化检测器
        self.detector = CombinedDetector(
            yolo_model_path='models/yolov8n-pose.pt',
            conf_threshold=0.5
        )

        # 初始化标定器
        self.calibrator = CameraCalibrator(checkerboard_size=(9, 6), square_size=25.0)

        # 尝试加载标定参数
        calib_path = Path('calibration_data/calib_params.json')
        if calib_path.exists():
            self.load_calibration(str(calib_path))
        else:
            # 使用默认参数初始化空间计算器
            from src.algorithms.spatial import SpatialCalculatorEnhanced
            from src.algorithms.calibration import CalibrationParams
            from src.utils.constants import DEFAULT_FX, DEFAULT_FY, DEFAULT_CX, DEFAULT_CY
            # 创建默认标定参数（根据常见笔记本摄像头参数优化）
            default_calib = CalibrationParams(
                fx=DEFAULT_FX, fy=DEFAULT_FY,
                cx=DEFAULT_CX, cy=DEFAULT_CY,
                dist_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0]
            )
            self.spatial_calc = SpatialCalculatorEnhanced(default_calib)
            logger.info(f"Spatial calculator initialized with default parameters: fx={DEFAULT_FX}, cx={DEFAULT_CX}")

        logger.info("System initialized")
    
    def load_calibration(self, filepath: str):
        """加载标定参数"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        calib_params = CalibrationParams.from_dict(data)
        # 使用新的空间计算模块
        from src.algorithms.spatial import SpatialCalculatorEnhanced
        self.spatial_calc = SpatialCalculatorEnhanced(calib_params)
        self.calibrated = True

        logger.info(f"Calibration loaded from {filepath}")
    
    def start_camera(self, camera_id: int = 0):
        """启动摄像头"""
        if self.camera is not None:
            self.camera.release()
        
        self.camera = VideoCapture(camera_id=camera_id)
        if self.camera.open():
            logger.info(f"Camera {camera_id} started")
            return True
        return False
    
    def stop_camera(self):
        """停止摄像头"""
        if self.camera:
            self.camera.release()
            self.camera = None
            logger.info("Camera stopped")


# 全局状态实例
state = SystemState()


# ==================== API 路由 ====================

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    state.running = True  # 设置运行状态
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
        </head>
        <body>
            <h1>🎥 摄像头实时感知 + 空间计量系统</h1>
            <p>API 文档：<a href="/docs">/docs</a></p>
            <p>前端页面：访问 <code>http://localhost:5173</code></p>
        </body>
    </html>
    """


@app.get("/api/status", response_model=SystemStatus)
async def get_status():
    """获取系统状态"""
    return SystemStatus(
        camera_opened=state.camera is not None,
        calibrated=state.calibrated,
        fps=state.fps_counter,
        persons_detected=0,
        hands_detected=0
    )


@app.post("/api/camera/start")
async def start_camera(config: CameraConfig = None):
    """启动摄像头"""
    if config is None:
        config = CameraConfig()
    
    if state.start_camera(camera_id=config.camera_id):
        return {"status": "success", "message": f"Camera {config.camera_id} started"}
    else:
        raise HTTPException(status_code=500, detail="Failed to open camera")


@app.post("/api/camera/stop")
async def stop_camera():
    """停止摄像头"""
    state.stop_camera()
    return {"status": "success", "message": "Camera stopped"}


@app.post("/api/calibration/load")
async def load_calibration(filepath: str):
    """加载标定参数"""
    try:
        state.load_calibration(filepath)
        return {"status": "success", "message": "Calibration loaded"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/calibration/status")
async def calibration_status():
    """获取标定状态"""
    return {
        "calibrated": state.calibrated,
        "spatial_calc_initialized": state.spatial_calc is not None,
        "image_count": len(state.calibration_images),
        "collecting": len(state.calibration_images) < 15
    }


@app.get("/api/calibration/start_capture")
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


@app.post("/api/calibration/capture_frame")
async def capture_calibration_frame():
    """从摄像头采集一帧标定图片"""
    if not state.camera or not state.camera.is_opened():
        raise HTTPException(status_code=400, detail="Camera not opened")

    try:
        ret, frame = state.camera.read()
        if not ret or frame is None:
            raise HTTPException(status_code=400, detail="Failed to capture frame")

        # 检测棋盘格
        found, corners = state.calibrator.find_checkerboard_corners(frame)

        if found:
            state.calibration_images.append(frame)
            state.calibration_corners.append(corners)
            return {
                "status": "success",
                "found": True,
                "image_count": len(state.calibration_images),
                "message": f"Captured frame {len(state.calibration_images)}/15"
            }
        else:
            return {
                "status": "success",
                "found": False,
                "image_count": len(state.calibration_images),
                "message": "No checkerboard detected"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/calibration/run")
async def run_calibration():
    """执行标定"""
    if len(state.calibration_images) < 3:
        raise HTTPException(status_code=400, detail="Need at least 3 valid images")

    try:
        # 准备物体点
        objpoints = []
        imgpoints = []
        gray_size = state.calibration_images[0].shape[:2]

        for corners in state.calibration_corners:
            objp = np.zeros((9 * 6, 3), np.float32)
            objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 25.0
            objpoints.append(objp)
            imgpoints.append(corners)

        # 执行标定
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray_size[::-1], None, None
        )

        # 计算重投影误差
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        mean_error = total_error / len(objpoints) if objpoints else 0

        # 保存标定参数
        calib_params = CalibrationParams(
            fx=mtx[0, 0],
            fy=mtx[1, 1],
            cx=mtx[0, 2],
            cy=mtx[1, 2],
            dist_coeffs=dist.flatten().tolist(),
            reprojection_error=mean_error,
            image_width=gray_size[1],
            image_height=gray_size[0]
        )

        # 保存到文件
        output_path = Path('calibration_data/calib_params.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        calib_params.save(str(output_path))

        # 重新加载标定参数
        state.load_calibration(str(output_path))

        return {
            "status": "success",
            "reprojection_error": mean_error,
            "num_images": len(state.calibration_images),
            "fx": mtx[0, 0],
            "fy": mtx[1, 1],
            "cx": mtx[0, 2],
            "cy": mtx[1, 2],
            "dist_coeffs": dist.flatten().tolist()
        }
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/calibration/upload")
async def upload_calibration_images(images: List[UploadFile] = File(...)):
    """上传标定图片"""
    valid_count = 0
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
                valid_count += 1

    return {
        "status": "success",
        "uploaded": len(images),
        "valid": valid_count,
        "total_valid": len(state.calibration_images)
    }


@app.post("/api/calibration/extrinsics")
async def save_extrinsics(config: dict):
    """保存外参配置"""
    try:
        height = config.get('height', 1.8)
        pitch_angle = config.get('pitch_angle', 30.0)

        if state.spatial_calc:
            state.spatial_calc.camera_height = height
            state.spatial_calc.pitch_angle = np.radians(pitch_angle)

        return {"status": "success", "message": "Extrinsics saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/calibration/reset")
async def reset_calibration():
    """重置标定"""
    state.calibration_images = []
    state.calibration_corners = []
    state.calibrated = False
    return {"status": "success", "message": "Calibration reset"}


class HeightCalibrationInput(BaseModel):
    """身高校准输入"""
    known_height: float


class DistanceCalibrationInput(BaseModel):
    """距离校准输入"""
    known_distance: float


@app.post("/api/calibration/height")
async def calibrate_height(input_data: HeightCalibrationInput):
    """
    使用已知身高进行校准
    
    通过输入真实身高，校准系统的身高测量比例
    """
    if state.spatial_calc is None:
        raise HTTPException(status_code=400, detail="Spatial calculator not initialized")
    
    try:
        # 存储校准比例
        state.height_calibration_ratio = input_data.known_height / 1.7  # 假设默认身高1.7m
        
        logger.info(f"Height calibration applied: {input_data.known_height}m, ratio: {state.height_calibration_ratio:.3f}")
        
        return {
            "status": "success",
            "message": f"Height calibration applied: {input_data.known_height}m",
            "calibration_ratio": state.height_calibration_ratio
        }
    except Exception as e:
        logger.error(f"Height calibration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Calibration failed: {str(e)}")


@app.post("/api/calibration/distance")
async def calibrate_distance(input_data: DistanceCalibrationInput):
    """
    使用已知距离进行校准
    
    通过输入真实距离，校准系统的距离测量比例
    """
    if state.spatial_calc is None:
        raise HTTPException(status_code=400, detail="Spatial calculator not initialized")
    
    try:
        # 存储距离校准比例
        state.distance_calibration_ratio = input_data.known_distance / 2.0  # 假设默认距离2.0m
        
        # 应用到空间计算器
        if hasattr(state.spatial_calc, 'distance_scale'):
            state.spatial_calc.distance_scale = state.distance_calibration_ratio
        
        logger.info(f"Distance calibration applied: {input_data.known_distance}m, ratio: {state.distance_calibration_ratio:.3f}")
        
        return {
            "status": "success",
            "message": f"Distance calibration applied: {input_data.known_distance}m",
            "calibration_ratio": state.distance_calibration_ratio
        }
    except Exception as e:
        logger.error(f"Distance calibration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Calibration failed: {str(e)}")


@app.get("/api/calibration/preview")
async def get_calibration_preview():
    """获取标定预览帧（带棋盘格检测）"""
    if not state.camera or not state.camera.is_opened():
        raise HTTPException(status_code=400, detail="Camera not opened")

    try:
        ret, frame = state.camera.read()
        if not ret or frame is None:
            raise HTTPException(status_code=400, detail="Failed to capture frame")

        # 检测棋盘格
        found, corners = state.calibrator.find_checkerboard_corners(frame)

        # 在图像上绘制检测结果
        display_frame = frame.copy()
        if found:
            # 绘制角点
            cv2.drawChessboardCorners(
                display_frame,
                state.calibrator.checkerboard_size,
                corners,
                found
            )
            # 绘制文字
            cv2.putText(
                display_frame,
                f"Detected: {len(corners)} corners",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                display_frame,
                "No checkerboard detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # 显示已采集数量
        cv2.putText(
            display_frame,
            f"Collected: {len(state.calibration_images)}/15",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        # 编码为JPEG
        _, buffer = cv2.imencode('.jpg', display_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "status": "success",
            "image": img_base64,
            "found": found,
            "corners_count": len(corners) if found else 0,
            "collected": len(state.calibration_images)
        }
    except Exception as e:
        logger.error(f"Preview failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/settings")
async def save_settings(settings: dict):
    """保存设置"""
    try:
        # 保存相机设置
        if 'camera' in settings:
            cam = settings['camera']
            state.camera_id = cam.get('id', 0)
            state.fps = cam.get('fps', 20)
        
        # 保存检测设置
        if 'detection' in settings:
            det = settings['detection']
            if state.detector:
                state.detector.conf_threshold = det.get('conf_threshold', 0.5)
        
        # 保存空间计量设置
        if 'spatial' in settings:
            spatial = settings['spatial']
            if state.spatial_calc:
                state.spatial_calc.ref_shoulder_width = spatial.get('ref_shoulder_width', 0.45)
                state.spatial_calc.topview_scale = spatial.get('topview_scale', 10.0)
        
        return {"status": "ok", "message": "设置已保存"}
    except Exception as e:
        logger.error(f"保存设置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 空间计量参数 API ====================

@app.get("/api/spatial/config")
async def get_spatial_config():
    """获取空间计量配置"""
    if state.spatial_calc is None:
        raise HTTPException(status_code=400, detail="Spatial calculator not initialized")

    # 返回默认配置
    return {
        "status": "success",
        "distance_scale": 1.0,
        "height_scale": 1.0,
        "head_params": {
            "ref_head_width": 0.15,
            "ref_eye_distance": 0.063,
            "ref_ear_distance": 0.145,
            "ref_eye_nose_distance": 0.035,
            "ref_head_height": 0.22
        },
        "kalman_params": {
            "max_speed": 3.0,
            "process_noise": 0.1,
            "measurement_noise": 0.3
        },
        "close_range_params": {
            "threshold": 1.5,
            "ultra_close_threshold": 0.5,
            "head_weight": 0.7,
            "body_weight": 0.3
        }
    }


@app.post("/api/spatial/head_params")
async def set_head_params(config: HeadSizeConfig):
    """设置头部尺寸参数"""
    if state.spatial_calc is None:
        raise HTTPException(status_code=400, detail="Spatial calculator not initialized")
    
    from algorithms.spatial_enhanced import HeadSizeParams
    params = HeadSizeParams(
        ref_head_width=config.ref_head_width,
        ref_eye_distance=config.ref_eye_distance,
        ref_ear_distance=config.ref_ear_distance,
        ref_eye_nose_distance=config.ref_eye_nose_distance,
        ref_head_height=config.ref_head_height
    )
    state.spatial_calc.set_head_params(params)
    
    return {"status": "success", "message": "Head size parameters updated"}


@app.post("/api/spatial/kalman_params")
async def set_kalman_params(config: KalmanConfig):
    """设置卡尔曼滤波参数"""
    if state.spatial_calc is None:
        raise HTTPException(status_code=400, detail="Spatial calculator not initialized")
    
    from algorithms.spatial_enhanced import KalmanParams
    params = KalmanParams(
        max_speed=config.max_speed,
        process_noise=config.process_noise,
        measurement_noise=config.measurement_noise
    )
    state.spatial_calc.set_kalman_params(params)
    
    return {"status": "success", "message": "Kalman parameters updated"}


@app.post("/api/spatial/close_range_params")
async def set_close_range_params(config: CloseRangeConfig):
    """设置近距离估计参数"""
    if state.spatial_calc is None:
        raise HTTPException(status_code=400, detail="Spatial calculator not initialized")
    
    from algorithms.spatial_enhanced import CloseRangeParams
    params = CloseRangeParams(
        threshold=config.threshold,
        ultra_close_threshold=config.ultra_close_threshold,
        head_weight=config.head_weight,
        body_weight=config.body_weight,
        use_perspective_correction=config.use_perspective_correction
    )
    state.spatial_calc.set_close_range_params(params)
    
    return {"status": "success", "message": "Close range parameters updated"}


@app.post("/api/spatial/config")
async def set_spatial_config(config: SpatialConfig):
    """设置完整空间计量配置"""
    if state.spatial_calc is None:
        raise HTTPException(status_code=400, detail="Spatial calculator not initialized")
    
    from algorithms.spatial_enhanced import HeadSizeParams, KalmanParams, CloseRangeParams
    
    # 设置头部参数
    head_params = HeadSizeParams(
        ref_head_width=config.head.ref_head_width,
        ref_eye_distance=config.head.ref_eye_distance,
        ref_ear_distance=config.head.ref_ear_distance,
        ref_eye_nose_distance=config.head.ref_eye_nose_distance,
        ref_head_height=config.head.ref_head_height
    )
    state.spatial_calc.set_head_params(head_params)
    
    # 设置卡尔曼参数
    kalman_params = KalmanParams(
        max_speed=config.kalman.max_speed,
        process_noise=config.kalman.process_noise,
        measurement_noise=config.kalman.measurement_noise
    )
    state.spatial_calc.set_kalman_params(kalman_params)
    
    # 设置近距离参数
    close_range_params = CloseRangeParams(
        threshold=config.close_range.threshold,
        ultra_close_threshold=config.close_range.ultra_close_threshold,
        head_weight=config.close_range.head_weight,
        body_weight=config.close_range.body_weight,
        use_perspective_correction=config.close_range.use_perspective_correction
    )
    state.spatial_calc.set_close_range_params(close_range_params)
    
    return {"status": "success", "message": "Spatial config updated"}


@app.get("/api/spatial/config")
async def get_spatial_config():
    """获取当前空间计量配置"""
    if state.spatial_calc is None:
        raise HTTPException(status_code=400, detail="Spatial calculator not initialized")
    
    # 返回当前配置
    return {
        "status": "success",
        "config": {
            "head": {
                "ref_head_width": 0.15,
                "ref_eye_distance": 0.063,
                "ref_ear_distance": 0.145,
                "ref_eye_nose_distance": 0.035,
                "ref_head_height": 0.22
            },
            "kalman": {
                "max_speed": 3.0,
                "process_noise": 0.1,
                "measurement_noise": 0.3
            },
            "close_range": {
                "threshold": 1.5,
                "ultra_close_threshold": 0.5,
                "head_weight": 0.7,
                "body_weight": 0.3,
                "use_perspective_correction": True
            }
        }
    }


@app.get("/api/depth/stats")
async def get_depth_stats():
    """获取深度估计统计"""
    from src.algorithms.depth_estimator import get_depth_estimator
    
    try:
        estimator = get_depth_estimator()
        return {
            "status": "success",
            "stats": estimator.get_stats()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/errors")
async def get_errors():
    """获取错误统计"""
    from utils.exceptions import get_error_handler
    error_handler = get_error_handler()
    return error_handler.get_stats()


@app.get("/api/performance")
async def get_performance():
    """获取性能监控数据"""
    import time
    import psutil

    process = psutil.Process()
    memory_info = process.memory_info()

    return {
        "memory": {
            "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
            "vms_mb": round(memory_info.vms / 1024 / 1024, 2)
        },
        "cpu": {
            "percent": process.cpu_percent()
        },
        "timestamp": time.time()
    }


# ==================== WebSocket ====================

@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    """
    WebSocket 视频流推送
    
    推送编码后的视频帧（二进制）
    """
    await websocket.accept()
    state.websocket_connections.append(websocket)
    
    logger.info("WebSocket video connection established")
    
    try:
        while state.running:
            if state.camera is None:
                await asyncio.sleep(0.1)
                continue
            
            # 读取帧
            ret, frame = state.camera.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue
            
            # 编码为 JPEG（简化方案）
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            
            # 发送
            await websocket.send_bytes(buffer.tobytes())
            
            # 帧率控制
            state.fps_counter += 1
            await asyncio.sleep(1 / state.camera.fps)
    
    except WebSocketDisconnect:
        logger.info("WebSocket video disconnected")
    finally:
        if websocket in state.websocket_connections:
            state.websocket_connections.remove(websocket)


@app.websocket("/ws/data")
async def websocket_data(websocket: WebSocket):
    """
    WebSocket 数据推送
    
    推送检测结果和空间计量数据（JSON）
    """
    await websocket.accept()
    logger.info("WebSocket data connection established")
    
    try:
        while state.running:
            if state.camera is None or state.detector is None:
                await asyncio.sleep(0.1)
                continue
            
            # 读取帧
            ret, frame = state.camera.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue
            
            # 检测
            import time
            frame_start = time.time()
            detect_start = time.time()
            result = state.detector.detect(frame)
            detect_time = (time.time() - detect_start) * 1000

            # 空间计量
            spatial_start = time.time()
            metrics = {
                "persons": [],
                "hands": [],
                "frame_shape": result.frame_shape,
                "processing": {}
            }

            if state.spatial_calc:
                # 获取图像尺寸
                frame_height, frame_width = result.frame_shape[:2] if result.frame_shape else (1080, 1920)

                for person in result.persons:
                    # 传递图像和尺寸给空间计算（支持深度估计）
                    person_metrics = state.spatial_calc.calc_person_metrics(
                        person,
                        image=frame,  # 传递原始图像用于深度估计
                        image_height=frame_height,
                        image_width=frame_width
                    )
                    metrics["persons"].append(person_metrics)

                for hand in result.hands:
                    hand_metrics = state.spatial_calc.calc_hand_metrics(hand)
                    metrics["hands"].append(hand_metrics)
                
                spatial_time = (time.time() - spatial_start) * 1000
            else:
                metrics["persons"] = result.persons
                metrics["hands"] = result.hands
                spatial_time = 0

            # 计算总处理时间
            total_time = (time.time() - frame_start) * 1000
            metrics["processing"] = {
                "detect_time_ms": round(detect_time, 2),
                "spatial_time_ms": round(spatial_time, 2),
                "total_time_ms": round(total_time, 2),
                "persons_count": len(result.persons),
                "hands_count": len(result.hands)
            }

            # 发送
            try:
                await websocket.send_json(metrics)
            except Exception as e:
                logger.error(f"Failed to send WebSocket message: {e}")
                break

            # 帧率控制（数据推送频率可降低）
            await asyncio.sleep(DATA_PUSH_INTERVAL)  # 使用配置的数据推送间隔

    except WebSocketDisconnect:
        logger.info("WebSocket data disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        import traceback
        logger.error(traceback.format_exc())


@app.get("/api/export/data")
async def export_data(start_time: Optional[float] = None, end_time: Optional[float] = None):
    """
    导出检测数据

    导出指定时间范围内的检测数据为 JSON 格式

    Args:
        start_time: 开始时间戳（可选）
        end_time: 结束时间戳（可选）
    """
    try:
        # 构建导出数据
        export_data = {
            "export_time": time.time(),
            "export_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "camera_opened": state.camera.is_opened() if state.camera else False,
                "calibrated": state.calibrated,
                "fps": state.fps if hasattr(state, 'fps') else 0
            },
            "data": []
        }

        # 如果有检测数据存储，这里可以添加过滤逻辑
        # 目前返回空数据或示例数据

        # 返回 JSON 数据
        from fastapi.responses import JSONResponse
        return JSONResponse(
            content=export_data,
            headers={
                "Content-Disposition": f"attachment; filename=detection_data_{time.strftime('%Y%m%d_%H%M%S')}.json"
            }
        )

    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


# ==================== 主程序 ====================

def main():
    """启动服务"""
    import argparse

    parser = argparse.ArgumentParser(description='FastAPI 服务')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='监听地址')
    parser.add_argument('--port', type=int, default=8000,
                       help='端口号')
    parser.add_argument('--reload', action='store_true',
                       help='启用热重载')

    args = parser.parse_args()

    # 设置运行状态
    state.running = True

    # 启动服务
    logger.info(f"Starting server at {args.host}:{args.port}")

    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == '__main__':
    main()
