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

# 挂载前端静态文件
WEB_DIST = Path(__file__).parent.parent.parent / "web" / "dist"
if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")


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

        # 轨迹跟踪器
        self.trajectories: Dict[int, List[Dict]] = {}  # track_id -> [{x, y, t}, ...]
        self.max_trajectory_length = 60  # 保留最近 60 帧（约 2 秒）
    
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

        # 初始化深度估计器
        try:
            from src.algorithms.depth_estimator import get_depth_estimator
            self.depth_estimator = get_depth_estimator(backend='depth_anything_v2', model_size='small')
            logger.info(f"Depth estimator initialized: backend={self.depth_estimator.backend}")
        except Exception as e:
            logger.warning(f"Depth estimator not available: {e}")
            self.depth_estimator = None

        # 尝试加载标定参数
        calib_path = Path('calibration_data/calib_params.json')
        if calib_path.exists():
            self.load_calibration(str(calib_path))
        else:
            # 使用默认参数初始化空间计算器
            from src.algorithms.spatial_enhanced import SpatialCalculatorEnhanced
            from src.algorithms.calibration import CalibrationParams
            from src.utils.constants import DEFAULT_FX, DEFAULT_FY, DEFAULT_CX, DEFAULT_CY
            # 创建默认标定参数（根据常见笔记本摄像头参数优化）
            default_calib = CalibrationParams(
                fx=DEFAULT_FX, fy=DEFAULT_FY,
                cx=DEFAULT_CX, cy=DEFAULT_CY,
                dist_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0]
            )
            self.spatial_calc = SpatialCalculatorEnhanced(default_calib)
            # 将深度估计器传递给空间计算器
            if self.depth_estimator:
                self.spatial_calc.depth_estimator = self.depth_estimator
            logger.info(f"Spatial calculator initialized with default parameters: fx={DEFAULT_FX}, cx={DEFAULT_CX}")

        logger.info("System initialized")
    
    def load_calibration(self, filepath: str):
        """加载标定参数"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        calib_params = CalibrationParams.from_dict(data)
        # 使用新的空间计算模块
        from src.algorithms.spatial_enhanced import SpatialCalculatorEnhanced
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
    """根路径 - 返回前端页面"""
    index_path = WEB_DIST / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding='utf-8'))
    else:
        return """
        <html>
            <head><title>摄像头实时感知系统</title></head>
            <body>
                <h1>🎥 摄像头实时感知系统</h1>
                <p>前端未构建，请运行: cd web && npm run build</p>
                <p>API 文档：<a href="/docs">/docs</a></p>
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


@app.get("/api/depth/heatmap")
async def get_depth_heatmap():
    """获取深度热力图"""
    if not state.camera or not state.camera.is_opened():
        return {"status": "error", "message": "Camera not opened"}

    try:
        ret, frame = state.camera.read()
        if not ret or frame is None:
            return {"status": "error", "message": "Failed to read frame"}

        if state.depth_estimator is None:
            return {"status": "error", "message": "Depth estimator not initialized"}

        # 估计深度（归一化用于显示）
        depth_map = state.depth_estimator.estimate(frame, normalize=True)
        if depth_map is None:
            return {"status": "error", "message": "Depth estimation failed"}

        # 同时获取原始深度值（用于校准计算）
        depth_raw = state.depth_estimator.estimate(frame, normalize=False)
        
        # 归一化深度用于热力图
        depth_min = float(np.min(depth_map))
        depth_max = float(np.max(depth_map))
        depth_mean = float(np.mean(depth_map))
        
        # 原始深度值
        raw_min = float(np.min(depth_raw)) if depth_raw is not None else 0
        raw_max = float(np.max(depth_raw)) if depth_raw is not None else 0
        raw_mean = float(np.mean(depth_raw)) if depth_raw is not None else 0

        # 归一化用于热力图（值小=近=红色，值大=远=蓝色）
        depth_normalized = ((depth_map - depth_min) / (depth_max - depth_min + 1e-6) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        # 编码为 base64
        _, buffer = cv2.imencode('.jpg', heatmap, [cv2.IMWRITE_JPEG_QUALITY, 80])
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 降采样深度数据用于前端鼠标悬停显示（1/8 分辨率）
        if depth_raw is not None:
            h, w = depth_raw.shape
            scale = 8
            small_h, small_w = h // scale, w // scale
            depth_small = cv2.resize(depth_raw, (small_w, small_h), interpolation=cv2.INTER_AREA)
            # 编码为 base64（float32 -> bytes）
            depth_bytes = depth_small.astype(np.float32).tobytes()
            depth_data_base64 = base64.b64encode(depth_bytes).decode('utf-8')
            depth_shape = {"width": small_w, "height": small_h, "scale": scale}
        else:
            depth_data_base64 = None
            depth_shape = None

        # 计算真实距离（使用原始深度值）
        logger.info(f"Raw depth range: min={raw_min:.2f}, max={raw_max:.2f}, mean={raw_mean:.2f}")
        logger.info(f"Calibration: scale={state.depth_estimator.scale_factor:.4f}, offset={state.depth_estimator.offset:.4f}")
        
        if state.depth_estimator.calibrated and depth_raw is not None:
            # 应用线性映射：real = scale * raw + offset
            # 不限制范围，允许合理的外推
            nearest_real = raw_min * state.depth_estimator.scale_factor + state.depth_estimator.offset
            farthest_real = raw_max * state.depth_estimator.scale_factor + state.depth_estimator.offset
            mean_real = raw_mean * state.depth_estimator.scale_factor + state.depth_estimator.offset

            # 确保 nearest < farthest（近的距离小，远的距离大）
            if nearest_real > farthest_real:
                nearest_real, farthest_real = farthest_real, nearest_real

            # 限制最小距离为 0
            nearest_real = max(0, nearest_real)
            farthest_real = max(nearest_real, farthest_real)
            mean_real = max(0, mean_real)

            calibrated = True
            
            # 返回校准点信息
            calibration_info = state.depth_estimator.calibration_point
        else:
            # 未标定时，不显示假距离
            nearest_real = None
            farthest_real = None
            mean_real = None
            calibrated = False
            calibration_info = None

        return {
            "status": "success",
            "heatmap": heatmap_base64,
            "nearest_distance": nearest_real,
            "farthest_distance": farthest_real,
            "depth_mean": mean_real,
            "calibrated": calibrated,
            "depth_data": depth_data_base64,
            "depth_shape": depth_shape,
            "scale_factor": state.depth_estimator.scale_factor if state.depth_estimator.calibrated else 1.0,
            "offset": state.depth_estimator.offset if state.depth_estimator.calibrated else 0.0,
            "calibration_info": calibration_info,
        }
    except Exception as e:
        logger.error(f"Depth heatmap error: {e}")
        return {"status": "error", "message": str(e)}


# ==================== 深度校准 API ====================
class DepthCalibrationPoint(BaseModel):
    """深度校准点"""
    relative_depth: float
    real_distance: float  # 单位：米


class DepthCalibrationRegion(BaseModel):
    """区域深度校准"""
    x: int
    y: int
    width: int
    height: int
    real_distance: float  # 单位：米


@app.post("/api/depth/calibrate_region")
async def calibrate_depth_region(data: DepthCalibrationRegion):
    """通过框选区域进行深度校准"""
    if state.depth_estimator is None:
        return {"status": "error", "message": "Depth estimator not initialized"}

    if not state.camera or not state.camera.is_opened():
        return {"status": "error", "message": "Camera not opened"}

    try:
        # 读取当前帧
        ret, frame = state.camera.read()
        if not ret or frame is None:
            return {"status": "error", "message": "Failed to read frame"}

        # 估计原始深度（不归一化，用于校准）
        depth_map = state.depth_estimator.estimate(frame, normalize=False)
        if depth_map is None:
            return {"status": "error", "message": "Depth estimation failed"}

        # 获取框选区域的平均深度
        h, w = depth_map.shape
        x1 = max(0, min(data.x, w - 1))
        y1 = max(0, min(data.y, h - 1))
        x2 = max(0, min(data.x + data.width, w))
        y2 = max(0, min(data.y + data.height, h))

        if x2 <= x1 or y2 <= y1:
            return {"status": "error", "message": "Invalid region"}

        region = depth_map[y1:y2, x1:x2]
        avg_depth = float(np.mean(region))

        # 添加校准点
        result = state.depth_estimator.add_calibration_point(avg_depth, data.real_distance)

        return {
            "status": "success",
            "region_avg_depth": avg_depth,
            "calibration": result
        }
    except Exception as e:
        logger.error(f"Region calibration error: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/api/depth/calibrate")
async def add_depth_calibration(data: DepthCalibrationPoint):
    """添加深度校准点"""
    if state.depth_estimator is None:
        return {"status": "error", "message": "Depth estimator not initialized"}
    
    result = state.depth_estimator.add_calibration_point(
        data.relative_depth, 
        data.real_distance
    )
    return {"status": "success", "calibration": result}


@app.get("/api/depth/calibration")
async def get_depth_calibration():
    """获取深度校准信息"""
    if state.depth_estimator is None:
        return {"status": "error", "message": "Depth estimator not initialized"}
    
    return {
        "status": "success",
        "calibration": state.depth_estimator.get_calibration_info()
    }


@app.delete("/api/depth/calibration")
async def clear_depth_calibration():
    """清除深度校准"""
    if state.depth_estimator is None:
        return {"status": "error", "message": "Depth estimator not initialized"}
    
    state.depth_estimator.clear_calibration()
    return {"status": "success", "message": "Calibration cleared"}


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

            # 更新轨迹
            import time as time_module
            current_time = time_module.time()
            for person in metrics["persons"]:
                track_id = person.get("track_id")
                if track_id is not None:
                    bbox = person.get("bbox", [])
                    if len(bbox) >= 4:
                        cx = bbox[0] + bbox[2] / 2
                        cy = bbox[1] + bbox[3] / 2
                        
                        # 添加轨迹点
                        if track_id not in state.trajectories:
                            state.trajectories[track_id] = []
                        
                        state.trajectories[track_id].append({
                            "x": cx,
                            "y": cy,
                            "t": current_time,
                            "distance": person.get("distance", 0)
                        })
                        
                        # 限制轨迹长度
                        if len(state.trajectories[track_id]) > state.max_trajectory_length:
                            state.trajectories[track_id].pop(0)
                        
                        # 计算速度
                        traj = state.trajectories[track_id]
                        if len(traj) >= 2:
                            dt = traj[-1]["t"] - traj[-2]["t"]
                            if dt > 0:
                                dx = traj[-1]["x"] - traj[-2]["x"]
                                dy = traj[-1]["y"] - traj[-2]["y"]
                                # 像素速度 -> 实际速度（简化）
                                pixel_speed = (dx**2 + dy**2) ** 0.5 / dt
                                # 假设 100 像素 ≈ 0.3 米（需要根据距离调整）
                                distance = person.get("distance", 2.0)
                                scale = 0.3 * distance / 2.0  # 距离越远，像素代表的实际距离越大
                                real_speed = pixel_speed * scale
                                person["velocity"] = {
                                    "vx": dx / dt * scale,
                                    "vy": dy / dt * scale,
                                    "speed": real_speed
                                }
                        
                        # 添加轨迹到输出
                        person["trajectory"] = state.trajectories[track_id][-30:]  # 最近 30 帧

            # 发送
            try:
                await websocket.send_json(metrics)
            except Exception as e:
                logger.error(f"Failed to send WebSocket message: {e}")
                break

            # ==================== 集成新功能 ====================
            try:
                # 1. 手势识别
                from algorithms.gesture_recognition import get_gesture_recognizer
                gesture_recognizer = get_gesture_recognizer()
                for hand in metrics["hands"]:
                    try:
                        gesture = gesture_recognizer.recognize(hand)
                        hand["gesture"] = gesture
                    except:
                        pass
                
                # 2. 跟踪器更新
                from algorithms.person_tracker import get_person_tracker
                tracker = get_person_tracker()
                try:
                    tracked_persons = tracker.update(metrics["persons"])
                    for i, person in enumerate(metrics["persons"]):
                        if i < len(tracked_persons):
                            person["track_id"] = tracked_persons[i].get("track_id", person.get("track_id"))
                            person["track_age"] = tracked_persons[i].get("age", 0)
                except:
                    pass
                
                # 3. 跌倒检测
                from algorithms.fall_detection import get_fall_detector
                fall_detector = get_fall_detector()
                for person in metrics["persons"]:
                    try:
                        fall_result = fall_detector.detect(person)
                        person["fall_risk"] = fall_result.get("fall_risk", 0)
                        person["is_falling"] = fall_result.get("is_falling", False)
                        if fall_result.get("is_falling"):
                            from algorithms.alert_system import get_alert_system
                            alert_system = get_alert_system()
                            alert_system.create_alert(
                                alert_type="fall",
                                severity="critical",
                                message="检测到跌倒！",
                                track_id=person.get("track_id")
                            )
                    except:
                        pass
                
                # 4. 报警系统检查
                from algorithms.alert_system import get_alert_system
                alert_system = get_alert_system()
                for person in metrics["persons"]:
                    try:
                        bbox = person.get("bbox", [0, 0, 0, 0])
                        cx = bbox[0] + bbox[2] / 2 if len(bbox) >= 4 else 0
                        cy = bbox[1] + bbox[3] / 2 if len(bbox) >= 4 else 0
                        alert_system.check_zones({"x": cx, "y": cy}, person.get("track_id"))
                    except:
                        pass
                
                # 5. 记录数据到分析模块
                from algorithms.data_analysis import get_data_analyzer
                analyzer = get_data_analyzer()
                analyzer.record_metric('person_count', len(result.persons))
                analyzer.record_metric('hand_count', len(result.hands))
                analyzer.record_metric('fps', 1000 / max(total_time, 1))
                analyzer.record_metric('detection_time', detect_time)

                # 6. 场景分析
                from algorithms.scene_analysis import get_scene_analyzer
                scene_analyzer = get_scene_analyzer()
                scene_analyzer.analyze(frame, metrics["persons"])

                # 7. 录制视频
                from algorithms.video_recording import get_video_recorder
                recorder = get_video_recorder()
                if recorder.is_recording:
                    recorder.write_frame(frame)

            except Exception as e:
                logger.warning(f"Integration error: {e}")


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


# ==================== 新增 API 端点 ====================

# 手势识别 API
@app.get("/api/gestures")
async def get_gestures():
    """获取当前识别的手势"""
    try:
        from algorithms.gesture_recognition import get_gesture_recognizer
        recognizer = get_gesture_recognizer()
        
        recent_gestures = recognizer.gesture_history[-10:] if recognizer.gesture_history else []
        
        return {
            "status": "success",
            "gestures": [g.to_dict() for g in recent_gestures],
            "smoothed_gesture": recognizer.get_smoothed_gesture()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# 跌倒检测 API
@app.get("/api/falls")
async def get_fall_events():
    """获取跌倒事件"""
    try:
        from algorithms.fall_detection import get_fall_detector
        detector = get_fall_detector()
        
        events = detector.get_fall_events()
        
        return {
            "status": "success",
            "events": [e.to_dict() for e in events[-20:]],
            "total_events": len(events)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# 跟踪状态 API
@app.get("/api/tracks")
async def get_tracks():
    """获取当前跟踪状态"""
    try:
        from algorithms.person_tracker import get_person_tracker
        tracker = get_person_tracker()
        
        tracks = tracker.get_active_tracks()
        
        return {
            "status": "success",
            "tracks": [t.to_dict() for t in tracks],
            "stats": tracker.get_stats()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# 报警 API
class AlertCreate(BaseModel):
    """报警创建请求"""
    alert_type: str
    severity: str
    message: str
    track_id: Optional[int] = None


@app.post("/api/alerts/create")
async def create_alert(data: AlertCreate):
    """创建报警事件"""
    try:
        from algorithms.alert_system import get_alert_system
        system = get_alert_system()

        alert = system.create_alert(
            alert_type=data.alert_type,
            severity=data.severity,
            message=data.message,
            track_id=data.track_id
        )

        return {
            "status": "success",
            "alert": alert.to_dict() if alert else None
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/alerts")
async def get_alerts(since: float = None, alert_type: str = None):
    """获取报警事件"""
    try:
        from algorithms.alert_system import get_alert_system
        system = get_alert_system()
        
        alerts = system.get_alerts(since=since, alert_type=alert_type)
        
        return {
            "status": "success",
            "alerts": [a.to_dict() for a in alerts[-50:]],
            "stats": system.get_stats()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: int):
    """确认报警"""
    try:
        from algorithms.alert_system import get_alert_system
        system = get_alert_system()
        
        success = system.acknowledge_alert(alert_id)
        
        return {
            "status": "success" if success else "not_found",
            "alert_id": alert_id
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# 报警区域 API
@app.get("/api/zones")
async def get_zones():
    """获取报警区域"""
    try:
        from algorithms.alert_system import get_alert_system
        system = get_alert_system()
        
        zones = [
            {
                "id": zone.zone_id,
                "name": zone.name,
                "polygon": zone.polygon,
                "type": zone.zone_type,
                "enabled": zone.enabled
            }
            for zone in system.alert_zones.values()
        ]
        
        return {
            "status": "success",
            "zones": zones
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


class ZoneConfig(BaseModel):
    """区域配置"""
    zone_id: str
    name: str
    polygon: List[List[int]]
    zone_type: str = "restricted"
    alert_type: str = "intrusion"


@app.post("/api/zones")
async def create_zone(config: ZoneConfig):
    """创建报警区域"""
    try:
        from algorithms.alert_system import get_alert_system
        system = get_alert_system()
        
        polygon = [tuple(p) for p in config.polygon]
        
        system.add_zone(
            zone_id=config.zone_id,
            name=config.name,
            polygon=polygon,
            zone_type=config.zone_type,
            alert_type=config.alert_type
        )
        
        return {
            "status": "success",
            "zone_id": config.zone_id
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.delete("/api/zones/{zone_id}")
async def delete_zone(zone_id: str):
    """删除报警区域"""
    try:
        from algorithms.alert_system import get_alert_system
        system = get_alert_system()
        
        system.remove_zone(zone_id)
        
        return {
            "status": "success",
            "zone_id": zone_id
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# 深度校准 API
@app.get("/api/depth/calibration")
async def get_depth_calibration():
    """获取深度校准参数"""
    try:
        from algorithms.depth_calibration import get_depth_calibrator
        calibrator = get_depth_calibrator()
        
        return {
            "status": "success",
            "params": calibrator.get_params()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


class DepthCalibrationPoint(BaseModel):
    """深度校准点"""
    pixel_depth: float
    real_depth: float


@app.post("/api/depth/calibration")
async def add_depth_calibration_point(point: DepthCalibrationPoint):
    """添加深度校准点"""
    try:
        from algorithms.depth_calibration import get_depth_calibrator
        calibrator = get_depth_calibrator()
        
        calibrator.add_calibration_point(
            pixel_depth=point.pixel_depth,
            real_depth=point.real_depth
        )
        
        return {
            "status": "success",
            "num_points": len(calibrator.calibration_points)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/depth/calibration/calibrate")
async def calibrate_depth(method: str = "linear"):
    """执行深度校准"""
    try:
        from algorithms.depth_calibration import get_depth_calibrator
        calibrator = get_depth_calibrator()
        
        success = calibrator.calibrate(method=method)
        
        if success:
            calibrator.save_params()
        
        return {
            "status": "success" if success else "failed",
            "params": calibrator.get_params()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# 数据记录 API
@app.get("/api/records/stats")
async def get_record_stats():
    """获取记录统计"""
    try:
        from algorithms.data_recorder import get_data_recorder
        recorder = get_data_recorder()
        
        return {
            "status": "success",
            "stats": recorder.get_session_stats()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/records/trajectories")
async def get_trajectories():
    """获取轨迹数据"""
    try:
        from algorithms.data_recorder import get_data_recorder
        recorder = get_data_recorder()
        
        trajectories = recorder.get_all_trajectories()
        
        return {
            "status": "success",
            "trajectories": [t.to_dict() for t in trajectories]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/records/export")
async def export_records(format: str = "json"):
    """导出记录数据"""
    try:
        from algorithms.data_recorder import get_data_recorder
        recorder = get_data_recorder()
        
        if format == "csv":
            filepath = recorder.export_csv()
        else:
            filepath = recorder.export_json()
        
        return {
            "status": "success",
            "filepath": filepath
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# 性能统计 API
@app.get("/api/performance")
async def get_performance():
    """获取性能统计"""
    try:
        stats = {
            "fps": state.fps,
            "frame_count": state.frame_count,
            "detection_time": state.last_detection_time,
            "persons_detected": len(state.last_persons) if hasattr(state, 'last_persons') else 0,
            "hands_detected": len(state.last_hands) if hasattr(state, 'last_hands') else 0
        }
        
        return {
            "status": "success",
            "performance": stats
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# 系统信息 API
@app.get("/api/system/info")
async def get_system_info():
    """获取系统信息"""
    import platform
    import torch
    
    try:
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            "camera_opened": state.camera is not None and state.camera.is_opened(),
            "calibrated": state.calibrator is not None and state.calibrator.is_calibrated()
        }
        
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_memory"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return {
            "status": "success",
            "system": info
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ==================== Round 201-300 新增 API ====================

# 场景分析
@app.get("/api/scene/analysis")
async def get_scene_analysis():
    """获取场景分析结果"""
    try:
        from algorithms.scene_analysis import get_scene_analyzer
        analyzer = get_scene_analyzer()
        stats = analyzer.get_statistics()
        return {"status": "success", "statistics": stats}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/scene/changes")
async def get_scene_changes():
    """获取场景变化"""
    try:
        from algorithms.scene_analysis import get_scene_analyzer
        analyzer = get_scene_analyzer()
        changes = analyzer.get_scene_changes()
        return {"status": "success", "changes": changes}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# 多摄像头管理
@app.get("/api/cameras")
async def list_cameras():
    """列出所有摄像头"""
    try:
        from algorithms.multi_camera import get_multi_camera_manager
        manager = get_multi_camera_manager()
        return {"status": "success", "cameras": [c.to_dict() for c in manager.get_all_info()]}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/cameras/{camera_id}/activate")
async def activate_camera(camera_id: int):
    """激活摄像头"""
    try:
        from algorithms.multi_camera import get_multi_camera_manager
        manager = get_multi_camera_manager()
        if manager.set_active(camera_id):
            return {"status": "success", "active_camera": camera_id}
        return {"status": "error", "message": "Camera not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/cameras/stats")
async def get_camera_stats():
    """获取摄像头统计"""
    try:
        from algorithms.multi_camera import get_multi_camera_manager
        manager = get_multi_camera_manager()
        return {"status": "success", "stats": manager.get_stats()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# 视频录制
@app.post("/api/recording/start")
async def start_recording():
    """开始录制"""
    try:
        from algorithms.video_recording import get_video_recorder
        recorder = get_video_recorder()
        if recorder.start_recording():
            return {"status": "success", "message": "Recording started"}
        return {"status": "error", "message": "Failed to start recording"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/recording/stop")
async def stop_recording():
    """停止录制"""
    try:
        from algorithms.video_recording import get_video_recorder
        recorder = get_video_recorder()
        session = recorder.stop_recording()
        if session:
            return {"status": "success", "session": session.to_dict()}
        return {"status": "error", "message": "No active recording"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/recording/status")
async def get_recording_status():
    """获取录制状态"""
    try:
        from algorithms.video_recording import get_video_recorder
        recorder = get_video_recorder()
        return {"status": "success", "recording": recorder.get_status()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# 用户认证
class LoginRequest(BaseModel):
    """登录请求"""
    username: str
    password: str


@app.post("/api/auth/login")
async def login(request: LoginRequest):
    """用户登录"""
    try:
        from algorithms.auth_system import get_auth_system
        auth = get_auth_system()
        result = auth.login(request.username, request.password)
        if result:
            return {"status": "success", "data": result}
        return {"status": "error", "message": "Invalid credentials"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/auth/logout")
async def logout(request: Request):
    """用户登出"""
    try:
        from algorithms.auth_system import get_auth_system
        auth = get_auth_system()
        session_id = request.headers.get("X-Session-ID", "")
        auth.logout(session_id)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/users")
async def list_users():
    """列出用户"""
    try:
        from algorithms.auth_system import get_auth_system
        auth = get_auth_system()
        users = auth.user_manager.get_all_users()
        return {"status": "success", "users": [u.to_dict() for u in users]}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# 数据分析
@app.get("/api/analytics/statistics")
async def get_analytics_statistics():
    """获取统计数据"""
    try:
        from algorithms.data_analysis import get_data_analyzer
        analyzer = get_data_analyzer()
        return {"status": "success", "statistics": analyzer.get_statistics()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/analytics/trends/{metric_name}")
async def get_metric_trend(metric_name: str):
    """获取指标趋势"""
    try:
        from algorithms.data_analysis import get_data_analyzer
        analyzer = get_data_analyzer()
        trend = analyzer.get_trend(metric_name)
        return {"status": "success", "trend": trend}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/analytics/report")
async def generate_report():
    """生成报告"""
    try:
        from algorithms.data_analysis import get_data_analyzer
        analyzer = get_data_analyzer()
        report = analyzer.generate_report()
        return {"status": "success", "report": report.to_dict()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# 模型管理
@app.get("/api/models")
async def list_models():
    """列出模型"""
    try:
        from algorithms.model_optimization import get_model_manager
        manager = get_model_manager()
        return {"status": "success", "models": manager.list_models()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/models/backends")
async def list_backends():
    """列出可用后端"""
    try:
        from algorithms.model_optimization import get_model_manager
        manager = get_model_manager()
        return {"status": "success", "backends": manager.accelerator.get_available_backends()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# 边缘计算
@app.get("/api/edge/device")
async def get_device_info():
    """获取设备信息"""
    try:
        from algorithms.edge_computing import get_edge_manager
        manager = get_edge_manager()
        return {"status": "success", "device": manager.get_device_info()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/edge/resources")
async def get_edge_resources():
    """获取边缘资源"""
    try:
        from algorithms.edge_computing import get_edge_manager
        manager = get_edge_manager()
        return {"status": "success", "resources": manager.get_resource_usage()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/edge/config")
async def get_edge_config():
    """获取边缘配置"""
    try:
        from algorithms.edge_computing import get_edge_manager
        manager = get_edge_manager()
        return {"status": "success", "config": manager.get_optimized_config()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# 协作功能
@app.get("/api/collaboration/users")
async def get_collaboration_users():
    """获取协作用户"""
    try:
        from algorithms.collaboration import get_collaboration_manager
        manager = get_collaboration_manager()
        return {"status": "success", "users": manager.get_users()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/collaboration/annotations")
async def get_collaboration_annotations(frame_id: int = None):
    """获取协作注释"""
    try:
        from algorithms.collaboration import get_collaboration_manager
        manager = get_collaboration_manager()
        return {"status": "success", "annotations": manager.get_annotations(frame_id)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/collaboration/session")
async def get_collaboration_session():
    """获取协作会话"""
    try:
        from algorithms.collaboration import get_collaboration_manager
        manager = get_collaboration_manager()
        return {"status": "success", "session": manager.get_session_info()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# 系统集成
@app.get("/api/integrations")
async def list_integrations():
    """列出集成"""
    try:
        from algorithms.integration import get_integration_manager
        manager = get_integration_manager()
        return {"status": "success", "integrations": manager.get_integrations()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/integrations/stats")
async def get_integration_stats():
    """获取集成统计"""
    try:
        from algorithms.integration import get_integration_manager
        manager = get_integration_manager()
        return {"status": "success", "stats": manager.get_stats()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# 部署管理
@app.get("/api/deployment/status")
async def get_deployment_status():
    """获取部署状态"""
    try:
        from algorithms.deployment import get_deployment_manager
        manager = get_deployment_manager()
        return {"status": "success", "deployment": manager.get_status()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/deployment/health")
async def get_deployment_health():
    """获取健康状态"""
    try:
        from algorithms.deployment import get_deployment_manager
        manager = get_deployment_manager()
        return {"status": "success", "health": manager.health_checker.get_status()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ==================== 综合感知 API ====================
from algorithms.perception_fusion import PerceptionFusion, visualize_perception

# 全局感知融合器
perception_fusion = None

def get_perception_fusion():
    """获取感知融合器"""
    global perception_fusion
    if perception_fusion is None:
        perception_fusion = PerceptionFusion(
            depth_estimator=state.depth_estimator,
            detector=state.detector
        )
    return perception_fusion

@app.get("/api/perception/process")
async def api_perception_process():
    """综合感知处理"""
    if not state.camera or not state.camera.is_opened():
        return {"status": "error", "message": "Camera not opened"}
    
    try:
        ret, frame = state.camera.read()
        if not ret or frame is None:
            return {"status": "error", "message": "Failed to read frame"}
        
        fusion = get_perception_fusion()
        result = fusion.process(frame)
        
        return {
            "status": "success",
            "result": result.to_dict()
        }
    except Exception as e:
        logger.error(f"Perception process error: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/perception/visualize")
async def api_perception_visualize():
    """获取可视化结果"""
    if not state.camera or not state.camera.is_opened():
        return {"status": "error", "message": "Camera not opened"}
    
    try:
        ret, frame = state.camera.read()
        if not ret or frame is None:
            return {"status": "error", "message": "Failed to read frame"}
        
        fusion = get_perception_fusion()
        result = fusion.process(frame)
        
        # 可视化
        vis = visualize_perception(frame, result)
        
        # 俯视图
        bird_view = fusion.get_bird_eye_view(result)
        
        # 编码
        _, buffer = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
        vis_base64 = base64.b64encode(buffer).decode('utf-8')
        
        _, buffer = cv2.imencode('.jpg', bird_view, [cv2.IMWRITE_JPEG_QUALITY, 80])
        bird_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "status": "success",
            "visualization": vis_base64,
            "bird_eye_view": bird_base64,
            "result": result.to_dict()
        }
    except Exception as e:
        logger.error(f"Perception visualize error: {e}")
        return {"status": "error", "message": str(e)}


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
