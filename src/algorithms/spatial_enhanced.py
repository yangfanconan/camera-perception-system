"""
增强版空间计量模块
功能：
1. 动态校准（支持用户输入参考尺寸自动修正）
2. 误差修正模型（多项式拟合）
3. 多帧融合（提高测量稳定性）
4. 置信度评估
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field
from collections import deque
from loguru import logger

from .calibration import CalibrationParams
from utils.constants import (
    # 图像常量
    DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT,
    # 身体部位常量
    BODY_PART_FULL, BODY_PART_HALF, BODY_PART_UPPER, BODY_PART_LOWER, BODY_PART_HEAD_ONLY, BODY_PART_UNKNOWN,
    BODY_PART_MIN_KEYPOINTS, BODY_PART_BBOX_RATIO_THRESHOLD_LOW, BODY_PART_BBOX_RATIO_THRESHOLD_MID,
    BODY_PART_BBOX_RATIO_THRESHOLD_HIGH,
    # 距离估计常量
    ULTRA_CLOSE_BBOX_RATIO, CLOSE_BBOX_RATIO, NEAR_BBOX_RATIO,
    REFERENCE_SHOULDER_WIDTH, REFERENCE_HEAD_WIDTH, REFERENCE_UPPER_BODY_WIDTH, REFERENCE_HEAD_HEIGHT,
    REFERENCE_EYE_DISTANCE, REFERENCE_EAR_DISTANCE, REFERENCE_EYE_NOSE_DISTANCE,
    DISTANCE_THRESHOLD_CLOSE, DISTANCE_THRESHOLD_ULTRA_CLOSE,
    BBOX_DISTANCE_PARAMS, EXTREME_CLOSE_DISTANCE_HEAD, EXTREME_CLOSE_DISTANCE_UPPER, EXTREME_CLOSE_DISTANCE_FULL,
    # 相机常量
    DEFAULT_FX, DEFAULT_FY, DEFAULT_CX, DEFAULT_CY,
    DEFAULT_CAMERA_HEIGHT, DEFAULT_PITCH_ANGLE, DEFAULT_FOV_VERTICAL, DEFAULT_FOV_HORIZONTAL,
    # 卡尔曼滤波常量
    KALMAN_MAX_SPEED, KALMAN_PROCESS_NOISE, KALMAN_MEASUREMENT_NOISE,
    KALMAN_DT_MIN, KALMAN_DT_MAX,
    # 多帧融合常量
    MEASUREMENT_BUFFER_SIZE, CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW
)


@dataclass
class MeasurementResult:
    """测量结果数据类"""
    value: float  # 测量值
    unit: str  # 单位
    confidence: float  # 置信度 (0-1)
    std_dev: float  # 标准差
    num_samples: int  # 样本数
    timestamp: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'value': round(self.value, 2),
            'unit': self.unit,
            'confidence': round(self.confidence, 3),
            'std_dev': round(self.std_dev, 4),
            'num_samples': self.num_samples
        }


@dataclass
class CalibrationRecord:
    """校准记录"""
    measured_pixels: float  # 测量像素值
    actual_value: float  # 实际值
    distance: float  # 测量时距离
    timestamp: float


@dataclass
class HeadSizeParams:
    """头部尺寸参数"""
    ref_head_width: float = REFERENCE_HEAD_WIDTH
    ref_eye_distance: float = REFERENCE_EYE_DISTANCE
    ref_ear_distance: float = REFERENCE_EAR_DISTANCE
    ref_eye_nose_distance: float = REFERENCE_EYE_NOSE_DISTANCE
    ref_head_height: float = REFERENCE_HEAD_HEIGHT


@dataclass
class KalmanParams:
    """卡尔曼滤波参数"""
    max_speed: float = KALMAN_MAX_SPEED
    process_noise: float = KALMAN_PROCESS_NOISE
    measurement_noise: float = KALMAN_MEASUREMENT_NOISE


@dataclass
class CameraExtrinsicsParams:
    """相机外参配置"""
    height: float = DEFAULT_CAMERA_HEIGHT
    pitch_angle: float = DEFAULT_PITCH_ANGLE
    fov_vertical: float = DEFAULT_FOV_VERTICAL
    fov_horizontal: float = DEFAULT_FOV_HORIZONTAL
    roll_angle: float = 0.0
    yaw_angle: float = 0.0


@dataclass
class CloseRangeParams:
    """近距离估计参数"""
    threshold: float = 2.0           # 近距离阈值 (米) - 增加到2米以更早启用近距离优化
    ultra_close_threshold: float = 0.8  # 超近距离阈值 (米) - 增加到0.8米
    head_weight: float = 0.8         # 近距离时头部估计权重 - 增加头部权重
    body_weight: float = 0.2         # 近距离时身体估计权重
    use_perspective_correction: bool = True  # 启用透视校正


class DistanceKalmanFilter:
    """
    距离估计卡尔曼滤波器

    状态向量: [distance, velocity]
    - distance: 当前距离
    - velocity: 距离变化速度（正为远离，负为靠近）

    物理模型:
    - 人最大移动速度: ~10 m/s (奔跑)
    - 正常移动速度: ~1.5 m/s (步行)
    - 测量噪声: ~0.1-0.5m (取决于检测质量)
    """

    def __init__(self,
                 initial_distance: float = 3.0,
                 max_speed: float = 3.0,  # 最大速度 m/s
                 process_noise: float = 0.1,  # 过程噪声
                 measurement_noise: float = 0.3):  # 测量噪声
        
        # 状态向量 [distance, velocity]
        self.x = np.array([initial_distance, 0.0])
        
        # 状态协方差矩阵
        self.P = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        
        # 物理约束
        self.max_speed = max_speed
        
        # 噪声参数
        self.Q = np.array([  # 过程噪声
            [process_noise ** 2, 0.0],
            [0.0, (max_speed / 10) ** 2]
        ])
        self.R = measurement_noise ** 2  # 测量噪声
        
        # 观测矩阵 (只观测距离)
        self.H = np.array([[1.0, 0.0]])
        
        # 上次更新时间
        self.last_time = None
        
        # 历史记录（用于调试）
        self.history = deque(maxlen=30)
    
    def predict(self, dt: float) -> np.ndarray:
        """
        预测步骤
        
        Args:
            dt: 时间间隔（秒）
        """
        # 状态转移矩阵（匀速模型）
        F = np.array([
            [1.0, dt],
            [0.0, 1.0]
        ])
        
        # 预测状态
        self.x = F @ self.x
        
        # 限制速度
        self.x[1] = np.clip(self.x[1], -self.max_speed, self.max_speed)
        
        # 预测协方差
        self.P = F @ self.P @ F.T + self.Q
        
        return self.x.copy()
    
    def update(self, measurement: float, confidence: float = 1.0) -> np.ndarray:
        """
        更新步骤
        
        Args:
            measurement: 测量距离
            confidence: 测量置信度 (0-1)
        """
        # 根据置信度调整测量噪声
        R = self.R / max(confidence, 0.1)
        
        # 卡尔曼增益
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T / S
        
        # 更新状态
        y = measurement - self.H @ self.x  # 残差
        self.x = self.x + K.flatten() * y
        
        # 更新协方差
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P
        
        # 记录历史
        self.history.append({
            'measurement': measurement,
            'estimate': self.x[0],
            'velocity': self.x[1],
            'confidence': confidence
        })
        
        return self.x.copy()
    
    def filter(self, measurement: float, dt: float, confidence: float = 1.0) -> Tuple[float, float]:
        """
        完整的滤波过程
        
        Args:
            measurement: 测量距离
            dt: 时间间隔（秒）
            confidence: 测量置信度
            
        Returns:
            (filtered_distance, velocity)
        """
        # 预测
        self.predict(dt)
        
        # 更新
        self.update(measurement, confidence)
        
        return float(self.x[0]), float(self.x[1])
    
    def get_state(self) -> Dict[str, float]:
        """获取当前状态"""
        return {
            'distance': float(self.x[0]),
            'velocity': float(self.x[1]),
            'uncertainty': float(np.sqrt(self.P[0, 0]))
        }
    
    def reset(self, distance: float = 3.0):
        """重置滤波器"""
        self.x = np.array([distance, 0.0])
        self.P = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        self.history.clear()

    def get_motion_state(self) -> str:
        """
        获取当前运动状态

        Returns:
            'stationary', 'walking', 'running'
        """
        velocity = abs(self.x[1])
        if velocity < 0.1:
            return 'stationary'
        elif velocity < 2.0:
            return 'walking'
        else:
            return 'running'


class SpatialCalculatorEnhanced:
    """增强版空间计算器"""

    def __init__(self, calibration_params: CalibrationParams,
                 head_params: HeadSizeParams = None,
                 kalman_params: KalmanParams = None,
                 close_range_params: CloseRangeParams = None):
        """
        初始化空间计算器

        Args:
            calibration_params: 相机标定参数
            head_params: 头部尺寸参数
            kalman_params: 卡尔曼滤波参数
            close_range_params: 近距离估计参数
        """
        self.calib = calibration_params

        # 相机内参
        self.fx = calibration_params.fx
        self.fy = calibration_params.fy
        self.cx = calibration_params.cx
        self.cy = calibration_params.cy

        # 相机外参（可配置）
        self.camera_height = 1.8  # 米
        self.pitch_angle = 30.0   # 度

        # 顶视图配置
        self.topview_scale = 10.0  # 1 米 = 10 像素
        self.topview_origin = (400, 300)

        # 头部尺寸参数
        self.head_params = head_params or HeadSizeParams()

        # 卡尔曼滤波参数
        self.kalman_params = kalman_params or KalmanParams()

        # 近距离估计参数
        self.close_range_params = close_range_params or CloseRangeParams()

        # 校准参数
        self.distance_scale = 1.0  # 距离缩放系数
        self.height_scale = 1.0    # 身高缩放系数
        self.size_scale = 1.0      # 尺寸缩放系数

        # 误差修正模型（多项式系数）
        self.distance_correction_coeffs = [0.0, 0.0, 1.0]  # ax^2 + bx + c
        self.height_correction_coeffs = [0.0, 0.0, 1.0]

        # 校准记录历史
        self.calibration_history: List[CalibrationRecord] = []

        # 多帧融合缓冲区
        self.measurement_buffer: Dict[str, deque] = {
            'distance': deque(maxlen=10),
            'height': deque(maxlen=10),
            'size': deque(maxlen=10)
        }

        # 卡尔曼滤波器（用于距离平滑）
        self.kalman_filters: Dict[int, DistanceKalmanFilter] = {}

        # 上次更新时间（用于计算 dt）
        self.last_update_time: Dict[int, float] = {}

        # 深度估计器（可选，由外部注入）
        self.depth_estimator = None

        logger.info("SpatialCalculatorEnhanced initialized")
        logger.info(f"Head params: width={self.head_params.ref_head_width}m, "
                   f"eye_dist={self.head_params.ref_eye_distance}m")
        logger.info(f"Kalman params: max_speed={self.kalman_params.max_speed}m/s")
        logger.info(f"Close range: threshold={self.close_range_params.threshold}m")

    def set_head_params(self, params: HeadSizeParams) -> None:
        """设置头部尺寸参数"""
        self.head_params = params
        logger.info(f"Head params updated: width={params.ref_head_width}m")

    def detect_body_part(self, keypoints: Dict[str, List[float]], bbox: List[int], 
                         image_height: float = 1080.0, image_width: float = 1920.0) -> Tuple[str, float, Dict]:
        """
        智能检测身体部位（全身/半身/上半身/下半身/仅头部）
        
        改进点：
        1. 考虑关键点质量和分布
        2. 结合边界框和关键点位置关系
        3. 检测身体是否被裁剪
        4. 更准确的极近距离判断
        
        Args:
            keypoints: 人体关键点 {name: [x, y]}
            bbox: 边界框 [x, y, w, h]
            image_height: 图像高度
            image_width: 图像宽度
            
        Returns:
            (部位类型, 完整度分数, 详细信息)
            部位类型: 'full_body', 'half_body', 'upper_body', 'lower_body', 'head_only'
        """
        # 定义关键点组
        head_points = ['nose', 'L_eye', 'R_eye', 'L_ear', 'R_ear']
        upper_points = ['L_shoulder', 'R_shoulder', 'L_elbow', 'R_elbow', 'L_wrist', 'R_wrist']
        lower_points = ['L_hip', 'R_hip', 'L_knee', 'R_knee', 'L_ankle', 'R_ankle']
        
        # 获取实际存在的各部位关键点
        head_kps = {p: keypoints[p] for p in head_points if p in keypoints}
        upper_kps = {p: keypoints[p] for p in upper_points if p in keypoints}
        lower_kps = {p: keypoints[p] for p in lower_points if p in keypoints}
        
        # 计算各部位关键点数量和质量
        head_count = len(head_kps)
        upper_count = len(upper_kps)
        lower_count = len(lower_kps)
        
        # 判断各部位是否存在（至少2个关键点）
        has_head = head_count >= 2
        has_upper = upper_count >= 2
        has_lower = lower_count >= 2
        
        # 计算关键点垂直分布范围
        all_kp_y = [kp[1] for kp in keypoints.values()]
        if all_kp_y:
            min_y = min(all_kp_y)
            max_y = max(all_kp_y)
            kp_span = max_y - min_y  # 关键点垂直跨度
        else:
            kp_span = 0
        
        # 边界框分析
        bbox_y1 = bbox[1]
        bbox_y2 = bbox[1] + bbox[3]
        bbox_height = bbox[3]
        bbox_width = bbox[2]
        bbox_area = bbox_width * bbox_height
        image_area = image_width * image_height
        
        bbox_height_ratio = bbox_height / image_height
        bbox_width_ratio = bbox_width / image_width
        bbox_area_ratio = bbox_area / image_area
        bbox_bottom_ratio = bbox_y2 / image_height
        bbox_top_ratio = bbox_y1 / image_height
        
        # 检测身体是否被画面裁剪
        body_clipped_top = bbox_y1 < 5  # 顶部被裁剪
        body_clipped_bottom = bbox_y2 > image_height - 5  # 底部被裁剪
        body_clipped_left = bbox[0] < 5  # 左侧被裁剪
        body_clipped_right = bbox[0] + bbox_width > image_width - 5  # 右侧被裁剪
        body_clipped = body_clipped_top or body_clipped_bottom or body_clipped_left or body_clipped_right
        
        # 检测关键点分布（判断身体姿态）
        if has_head and has_upper and has_lower:
            # 获取关键参考点
            nose_y = head_kps.get('nose', [0, image_height/2])[1]
            shoulder_y = min(upper_kps.get('L_shoulder', [0, image_height])[1], 
                           upper_kps.get('R_shoulder', [0, image_height])[1])
            hip_y = min(lower_kps.get('L_hip', [0, image_height])[1],
                       lower_kps.get('R_hip', [0, image_height])[1])
            ankle_y = max(lower_kps.get('L_ankle', [0, 0])[1],
                         lower_kps.get('R_ankle', [0, 0])[1])
            
            # 计算身体各部分比例
            head_to_shoulder = shoulder_y - nose_y
            shoulder_to_hip = hip_y - shoulder_y
            hip_to_ankle = ankle_y - hip_y
            
            # 正常人体比例：头:上身:下身 ≈ 1:2:3
            # 如果下身比例很小，可能是坐着或弯腰
            if hip_to_ankle < head_to_shoulder * 1.5:
                # 下身较短，可能是坐着或半身
                body_part = 'half_body'
                confidence = 0.75
            elif body_clipped_bottom and not body_clipped_top:
                # 底部被裁剪，但顶部完整 -> 全身但脚部出画面
                body_part = 'full_body'
                confidence = 0.85
            elif bbox_height_ratio > 0.6:
                # 占据画面高度超过60%，很可能是全身
                body_part = 'full_body'
                confidence = 0.9
            else:
                body_part = 'full_body'
                confidence = 0.8
                
        elif has_head and has_upper and not has_lower:
            # 有头、上半身，无下半身
            if body_clipped_bottom:
                # 底部被裁剪，可能是全身被截断
                body_part = 'upper_body'
                confidence = 0.7
            else:
                # 明确的上半身
                body_part = 'upper_body'
                confidence = 0.9
                
        elif not has_head and has_upper and has_lower:
            # 无头，有上半身、下半身
            if body_clipped_top:
                # 顶部被裁剪，可能是全身但头部出画面
                body_part = 'lower_body'
                confidence = 0.75
            else:
                body_part = 'lower_body'
                confidence = 0.8
                
        elif has_head and not has_upper and not has_lower:
            # 只有头（或头部附近）
            if bbox_area_ratio > 0.15 or bbox_height_ratio > 0.25:
                # 占据画面较大，可能是特写
                body_part = 'head_only'
                confidence = 0.9
            else:
                body_part = 'head_only'
                confidence = 0.7
                
        elif has_head and has_lower and not has_upper:
            # 有头、下半身，无上半身（弯腰、倒立等特殊情况）
            body_part = 'half_body'
            confidence = 0.6
            
        else:
            # 关键点很少，根据边界框推断
            if bbox_height_ratio > 0.5 and bbox_area_ratio > 0.2:
                body_part = 'full_body'
                confidence = 0.4
            elif bbox_height_ratio > 0.3:
                if bbox_bottom_ratio > 0.8:
                    body_part = 'lower_body'
                elif bbox_top_ratio < 0.2:
                    body_part = 'upper_body'
                else:
                    body_part = 'half_body'
                confidence = 0.4
            elif bbox_height_ratio > 0.15:
                body_part = 'head_only'
                confidence = 0.5
            else:
                body_part = 'unknown'
                confidence = 0.2
        
        # 极近距离修正（更激进的修正策略）
        # 当画面占比很大时，修正判断
        if bbox_area_ratio > 0.35:  # 从0.4降低到0.35
            if body_part == 'full_body':
                # 全身占35%以上画面，说明人非常近
                if bbox_height_ratio > 0.6:  # 从0.7降低到0.6
                    body_part = 'upper_body'
                    confidence = 0.85
                elif bbox_area_ratio > 0.5:  # 新增：占比超过50%直接判定为上半身
                    body_part = 'upper_body'
                    confidence = 0.8
            elif body_part == 'upper_body':
                # 上半身占比超过45%就可能是仅头部
                if bbox_area_ratio > 0.45:  # 从0.6降低到0.45
                    body_part = 'head_only'
                    confidence = 0.9
            elif body_part == 'half_body' and bbox_area_ratio > 0.4:
                # 半身占比超过40%可能是上半身特写
                body_part = 'upper_body'
                confidence = 0.85
        
        # 超近距离强制修正（画面占比超过60%）
        if bbox_area_ratio > 0.6:
            # 无论如何都判定为仅头部（脸贴摄像头的情况）
            body_part = 'head_only'
            confidence = 0.95
        
        info = {
            'has_head': has_head,
            'has_upper': has_upper,
            'has_lower': has_lower,
            'head_count': head_count,
            'upper_count': upper_count,
            'lower_count': lower_count,
            'kp_span': kp_span,
            'bbox_height_ratio': bbox_height_ratio,
            'bbox_width_ratio': bbox_width_ratio,
            'bbox_area_ratio': bbox_area_ratio,
            'bbox_bottom_ratio': bbox_bottom_ratio,
            'bbox_top_ratio': bbox_top_ratio,
            'body_clipped': body_clipped,
            'body_clipped_top': body_clipped_top,
            'body_clipped_bottom': body_clipped_bottom
        }
        
        return body_part, confidence, info

    def set_kalman_params(self, params: KalmanParams) -> None:
        """设置卡尔曼滤波参数"""
        self.kalman_params = params
        # 更新现有滤波器参数
        for kf in self.kalman_filters.values():
            kf.max_speed = params.max_speed
            kf.Q = np.array([
                [params.process_noise ** 2, 0.0],
                [0.0, (params.max_speed / 10) ** 2]
            ])
            kf.R = params.measurement_noise ** 2
        logger.info(f"Kalman params updated: max_speed={params.max_speed}m/s")

    def set_close_range_params(self, params: CloseRangeParams) -> None:
        """设置近距离估计参数"""
        self.close_range_params = params
        logger.info(f"Close range params updated: threshold={params.threshold}m")

    def set_camera_extrinsics(self, height: float, pitch_angle: float) -> None:
        """设置相机外参"""
        self.camera_height = height
        self.pitch_angle = pitch_angle
        logger.info(f"Camera extrinsics set: height={height}m, pitch={pitch_angle}°")
    
    # ==================== 基础计算（同原版本） ====================
    
    def pixel_to_camera_coords(
        self, u: float, v: float, z_c: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """像素坐标→相机坐标系"""
        x_norm = (u - self.cx) / self.fx
        y_norm = (v - self.cy) / self.fy
        
        if z_c is None:
            return (x_norm, y_norm, 1.0)
        
        return (x_norm * z_c, y_norm * z_c, z_c)
    
    def calc_distance_to_camera(
        self,
        pixel_width: float,
        ref_width: float
    ) -> float:
        """通过相似三角形计算距离"""
        if pixel_width <= 0:
            return 0.0
        
        distance = (ref_width * self.fx) / pixel_width
        return distance * self.distance_scale
    
    def calc_height(
        self,
        head_point: List[int],
        foot_point: List[int],
        distance: float
    ) -> float:
        """计算身高"""
        pixel_height = abs(foot_point[1] - head_point[1])
        height_m = (pixel_height * distance) / self.fx
        height_cm = height_m * 100
        
        return height_cm * self.height_scale
    
    def calc_hand_size(
        self,
        palm_point: List[int],
        finger_point: List[int],
        distance: float
    ) -> float:
        """计算手大小"""
        pixel_dist = np.sqrt(
            (finger_point[0] - palm_point[0]) ** 2 +
            (finger_point[1] - palm_point[1]) ** 2
        )
        
        size_m = (pixel_dist * distance) / self.fx
        size_cm = size_m * 100
        
        return size_cm * self.size_scale
    
    # ==================== 动态校准功能 ====================
    
    def add_calibration_record(
        self,
        measured_pixels: float,
        actual_value: float,
        distance: float,
        measurement_type: str = 'distance'
    ) -> None:
        """
        添加校准记录
        
        Args:
            measured_pixels: 测量的像素值
            actual_value: 实际值（米或厘米）
            distance: 测量时的距离
            measurement_type: 测量类型 ('distance', 'height', 'size')
        """
        record = CalibrationRecord(
            measured_pixels=measured_pixels,
            actual_value=actual_value,
            distance=distance,
            timestamp=self._get_timestamp()
        )
        
        self.calibration_history.append(record)
        logger.info(f"Calibration record added: {measurement_type}={actual_value}, "
                   f"pixels={measured_pixels}, distance={distance}")
        
        # 更新缩放系数
        self._update_scale_factors(measurement_type)
    
    def _update_scale_factors(self, measurement_type: str) -> None:
        """更新缩放系数"""
        if len(self.calibration_history) < 2:
            return
        
        # 计算平均缩放系数
        for record in self.calibration_history:
            if record.measured_pixels > 0:
                expected = (record.actual_value * self.fx) / record.measured_pixels
                if expected > 0:
                    scale = record.distance / expected
                    
                    if measurement_type == 'distance':
                        self.distance_scale *= 0.7 + 0.3 * scale
                    elif measurement_type == 'height':
                        self.height_scale *= 0.7 + 0.3 * (record.actual_value / expected)
    
    def calibrate_with_known_height(
        self,
        person_bbox: List[int],
        keypoints: Dict[str, List[int]],
        known_height: float
    ) -> Dict[str, float]:
        """
        使用已知身高进行校准
        
        Args:
            person_bbox: 人体边界框
            keypoints: 人体关键点
            known_height: 已知身高（厘米）
            
        Returns:
            校准参数
        """
        # 计算初始距离
        distance = self.calc_person_distance(person_bbox)
        
        # 计算初始身高
        if 'nose' in keypoints and 'left_ankle' in keypoints:
            head = keypoints['nose']
            foot = keypoints['left_ankle']
            calculated_height = self.calc_height(head, foot, distance)
            
            # 计算修正系数
            if calculated_height > 0:
                self.height_scale = known_height / calculated_height
                
                logger.info(f"Height calibration: calculated={calculated_height:.1f}cm, "
                           f"known={known_height}cm, scale={self.height_scale:.3f}")
        
        return {
            'height_scale': self.height_scale,
            'distance': distance
        }
    
    def calibrate_with_known_distance(
        self,
        person_bbox: List[int],
        known_distance: float
    ) -> Dict[str, float]:
        """
        使用已知距离进行校准
        
        Args:
            person_bbox: 人体边界框
            known_distance: 已知距离（米）
            
        Returns:
            校准参数
        """
        # 计算初始距离
        calculated_distance = self.calc_person_distance(person_bbox)
        
        # 计算修正系数
        if calculated_distance > 0:
            self.distance_scale = known_distance / calculated_distance
            
            logger.info(f"Distance calibration: calculated={calculated_distance:.2f}m, "
                       f"known={known_distance}m, scale={self.distance_scale:.3f}")
        
        return {
            'distance_scale': self.distance_scale,
            'calculated_distance': calculated_distance
        }
    
    # ==================== 误差修正模型 ====================
    
    def fit_distance_correction(self) -> None:
        """拟合距离误差修正模型（多项式）"""
        if len(self.calibration_history) < 3:
            logger.warning("Not enough calibration records for fitting")
            return
        
        # 准备数据
        x_data = []  # 测量距离
        y_data = []  # 误差
        
        for record in self.calibration_history:
            if record.measured_pixels > 0:
                measured_dist = (0.45 * self.fx) / record.measured_pixels
                error = record.actual_value - measured_dist
                x_data.append(measured_dist)
                y_data.append(error)
        
        # 多项式拟合（二次）
        coeffs = np.polyfit(x_data, y_data, 2)
        self.distance_correction_coeffs = coeffs.tolist()
        
        logger.info(f"Distance correction model fitted: {coeffs}")
    
    def apply_distance_correction(self, distance: float) -> float:
        """
        应用距离误差修正
        
        Args:
            distance: 测量距离
            
        Returns:
            修正后的距离
        """
        a, b, c = self.distance_correction_coeffs
        correction = a * distance ** 2 + b * distance + c
        return distance + correction
    
    # ==================== 多帧融合 ====================
    
    def add_measurement(
        self,
        measurement_type: str,
        value: float,
        track_id: Optional[int] = None
    ) -> MeasurementResult:
        """
        添加测量值到缓冲区（多帧融合）
        
        Args:
            measurement_type: 测量类型 ('distance', 'height', 'size')
            value: 测量值
            track_id: 追踪 ID（用于区分不同目标）
            
        Returns:
            融合后的测量结果
        """
        key = f"{measurement_type}_{track_id}" if track_id else measurement_type
        
        if key not in self.measurement_buffer:
            self.measurement_buffer[key] = deque(maxlen=10)
        
        self.measurement_buffer[key].append(value)
        
        # 计算统计量
        values = list(self.measurement_buffer[key])
        
        if len(values) < 2:
            return MeasurementResult(
                value=value,
                unit='m' if measurement_type == 'distance' else 'cm',
                confidence=0.5,
                std_dev=0.0,
                num_samples=1
            )
        
        mean = np.mean(values)
        std = np.std(values)
        
        # 置信度计算（基于标准差）
        if mean > 0:
            cv = std / mean  # 变异系数
            confidence = max(0.0, min(1.0, 1.0 - cv * 10))
        else:
            confidence = 0.5
        
        return MeasurementResult(
            value=mean,
            unit='m' if measurement_type == 'distance' else 'cm',
            confidence=confidence,
            std_dev=std,
            num_samples=len(values),
            timestamp=self._get_timestamp()
        )
    
    def clear_buffer(self, track_id: Optional[int] = None):
        """清空缓冲区"""
        if track_id is None:
            for buffer in self.measurement_buffer.values():
                buffer.clear()
        else:
            keys_to_clear = [k for k in self.measurement_buffer.keys() 
                           if str(track_id) in k]
            for key in keys_to_clear:
                self.measurement_buffer[key].clear()
    
    # ==================== 综合测量函数 ====================
    
    def calc_person_metrics(
        self,
        person: Dict[str, Any],
        image: np.ndarray = None,
        use_correction: bool = True,
        use_fusion: bool = True,
        prev_distance: float = None,
        dt: float = 0.05,
        image_height: float = 1080.0,
        image_width: float = 1920.0
    ) -> Dict[str, Any]:
        """
        计算人体综合指标

        Args:
            person: 人体检测结果
            image: 原始图像（用于深度估计）
            use_correction: 是否应用误差修正
            use_fusion: 是否应用多帧融合
            prev_distance: 上一帧的距离（用于速度约束）
            dt: 时间间隔（秒）
            image_height: 图像高度
            image_width: 图像宽度

        Returns:
            包含各项指标的字典
        """
        bbox = person['bbox']
        keypoints = person.get('keypoints', {})
        track_id = person.get('track_id')

        # ========== 多维度距离估计 ==========
        # 1. 基于身体宽度的距离
        body_distance = self.calc_person_distance(bbox)

        # 2. 基于头部尺寸的距离
        head_distance, head_confidence = self.calc_head_distance(keypoints)

        # 3. 基于人体姿态的距离（使用更多关键点）
        pose_distance, pose_confidence = self.calc_pose_based_distance(keypoints, bbox)

        # 4. 近距离优化估计（传入实际图像尺寸）
        distance, estimate_method = self.calc_close_range_distance(
            body_distance, head_distance, head_confidence, bbox, keypoints,
            image_height=image_height, image_width=image_width
        )
        
        # 5. 深度估计融合（暂时禁用 - Depth Anything V2 输出相对深度，需要校准）
        # TODO: 添加深度校准功能后启用
        # Depth Anything V2 输出的是相对深度，不是绝对深度
        # 需要通过已知距离的参考物体进行尺度校准才能使用
        # 调试日志
        bbox_area_ratio = (bbox[2] * bbox[3]) / (image_width * image_height)
        logger.info(f"Distance calc: body={body_distance:.2f}, head={head_distance:.2f}, "
                   f"final={distance:.2f}, method={estimate_method}, bbox_ratio={bbox_area_ratio:.2f}, "
                   f"img_size={image_width:.0f}x{image_height:.0f}")

        # 检查是否为极近距离（画面占比超过50%）
        is_ultra_close_by_bbox = bbox_area_ratio > 0.5

        # 5. 如果姿态估计可用，进一步融合
        if pose_distance > 0 and pose_confidence > 0.3:
            if abs(pose_distance - distance) < 0.5:
                distance = distance * 0.7 + pose_distance * 0.3
                estimate_method += "+pose"

        # ========== 卡尔曼滤波平滑 ==========
        import time
        current_time = time.time()

        # 极近距离时跳过卡尔曼滤波（直接使用估计值）
        if is_ultra_close_by_bbox:
            # 极近距离时不使用卡尔曼滤波，避免平滑到旧值
            filtered_distance = distance
            velocity = 0.0
            motion_state = 'static'
            kalman_info = {'velocity': 0.0, 'uncertainty': 0.1, 'motion_state': 'static'}
            estimate_method += "_no_kf"
        else:
            # 获取或创建卡尔曼滤波器（使用配置参数）
            if track_id not in self.kalman_filters:
                self.kalman_filters[track_id] = DistanceKalmanFilter(
                    initial_distance=distance,
                    max_speed=self.kalman_params.max_speed,
                    process_noise=self.kalman_params.process_noise,
                    measurement_noise=self.kalman_params.measurement_noise
                )
                self.last_update_time[track_id] = current_time

            # 计算实际时间间隔
            last_time = self.last_update_time.get(track_id, current_time)
            actual_dt = min(max(current_time - last_time, 0.01), 0.5)  # 限制在 10ms-500ms
            self.last_update_time[track_id] = current_time

            # 计算测量置信度（基于头部检测和身体完整度）
            bbox_ratio = bbox[3] / max(bbox[2], 1)
            measurement_confidence = 0.5  # 基础置信度
            if head_confidence > 0.3:
                measurement_confidence += 0.3 * head_confidence
            if bbox_ratio > 1.0:
                measurement_confidence += 0.2

            # 使用卡尔曼滤波器
            kf = self.kalman_filters[track_id]
            filtered_distance, velocity = kf.filter(distance, actual_dt, measurement_confidence)
            
            # 获取运动状态
            motion_state = kf.get_motion_state()
            
            # 记录滤波信息
            kalman_info = kf.get_state()
            kalman_info['motion_state'] = motion_state
            estimate_method += f"_kf(v={velocity:.1f})"

        # 应用误差修正（对滤波后的距离）
        if use_correction:
            filtered_distance = self.apply_distance_correction(filtered_distance)

        # 多帧融合
        if use_fusion:
            distance_result = self.add_measurement('distance', filtered_distance, track_id)
            filtered_distance = distance_result.value
        else:
            distance_result = None
        
        # 使用滤波后的距离
        distance = filtered_distance

        # 身体部位检测（使用实际图像尺寸）
        body_part, part_confidence, part_info = self.detect_body_part(
            keypoints, bbox, image_height=image_height, image_width=image_width
        )

        # 计算身高
        height = 0.0
        height_result = None

        if keypoints:
            # 查找头部和脚部关键点
            head = None
            foot = None

            for kp_name in ['nose', 'L_eye', 'R_eye']:
                if kp_name in keypoints:
                    head = keypoints[kp_name]
                    break

            for kp_name in ['L_ankle', 'R_ankle']:
                if kp_name in keypoints:
                    foot = keypoints[kp_name]
                    break

            if head and foot:
                height = self.calc_height(head, foot, distance)

                if use_fusion:
                    height_result = self.add_measurement('height', height, track_id)
                    height = height_result.value

        # 顶视图坐标
        x_center = bbox[0] + bbox[2] // 2
        y_center = bbox[1] + bbox[3] // 2
        topview = self.pixel_to_topview(x_center, y_center, distance)

        # 构建结果
        result = {
            "distance": round(distance, 2),
            "height": round(height, 1),
            "topview": {
                "x": round(topview[0], 1),
                "y": round(topview[1], 1)
            },
            "estimate_method": estimate_method,
            "head_confidence": round(head_confidence, 2) if head_confidence else 0,
            "pose_confidence": round(pose_confidence, 2) if pose_confidence else 0,
            "velocity": round(kalman_info.get('velocity', 0), 2),
            "motion_state": kalman_info.get('motion_state', 'unknown'),
            "uncertainty": round(kalman_info.get('uncertainty', 0), 3),
            "bbox": bbox,
            "keypoints": keypoints,
            "track_id": track_id,
            # 身体部位信息
            "body_part": body_part,
            "body_part_confidence": round(part_confidence, 2),
            "bbox_area_ratio": round(part_info['bbox_area_ratio'], 3),
            "bbox_height_ratio": round(part_info['bbox_height_ratio'], 3),
            # 调试信息
            "image_size": [int(image_width), int(image_height)]
        }

        # 添加置信度信息
        if distance_result:
            result["distance_confidence"] = distance_result.confidence
            result["distance_std"] = distance_result.std_dev

        if height_result:
            result["height_confidence"] = height_result.confidence
            result["height_std"] = height_result.std_dev

        return result

    def calc_hand_metrics(
        self,
        hand: Dict[str, Any],
        use_fusion: bool = True
    ) -> Dict[str, Any]:
        """
        计算手部综合指标
        
        Args:
            hand: 手部检测结果
            use_fusion: 是否应用多帧融合
            
        Returns:
            包含各项指标的字典
        """
        keypoints = hand.get('keypoints', [])
        bbox = hand['bbox']
        
        if len(keypoints) < 21:
            return {}
        
        # 掌根到中指指尖
        palm = keypoints[0]
        middle_finger = keypoints[12]
        
        # 估算距离
        hand_width = bbox[2]
        ref_hand_width = 0.08  # 8cm
        distance = (ref_hand_width * self.fx) / hand_width if hand_width > 0 else 1.0
        
        # 计算手大小
        hand_size = self.calc_hand_size(palm, middle_finger, distance)
        
        # 多帧融合
        if use_fusion:
            size_result = self.add_measurement('size', hand_size)
            hand_size = size_result.value
        else:
            size_result = None
        
        # 顶视图坐标
        palm_center = [
            bbox[0] + bbox[2] // 2,
            bbox[1] + bbox[3] // 2
        ]
        topview = self.pixel_to_topview(palm_center[0], palm_center[1], distance)
        
        result = {
            "size": round(hand_size, 1),
            "distance": round(distance, 2),
            "topview": {
                "x": round(topview[0], 1),
                "y": round(topview[1], 1)
            },
            "keypoints": keypoints
        }
        
        if size_result:
            result["size_confidence"] = size_result.confidence
            result["size_std"] = size_result.std_dev
        
        return result
    
    # ==================== 辅助函数 ====================
    
    def calc_person_distance(self, person_bbox: List[int], 
                             image_width: float = DEFAULT_IMAGE_WIDTH, 
                             image_height: float = DEFAULT_IMAGE_HEIGHT) -> float:
        """计算人到摄像头的距离"""
        w = person_bbox[2]
        if w <= 0:
            return 0.0

        # 根据边界框占比调整参考宽度（极近距离时边界框很宽）
        bbox_area_ratio = (person_bbox[2] * person_bbox[3]) / (image_width * image_height)
        
        # 极近距离时（占比>40%），使用头部宽度而非肩宽
        if bbox_area_ratio > 0.4:
            ref_width = REFERENCE_HEAD_WIDTH
        elif bbox_area_ratio > 0.2:
            ref_width = REFERENCE_UPPER_BODY_WIDTH
        else:
            ref_width = REFERENCE_SHOULDER_WIDTH
        
        distance = (ref_width * self.fx) / w
        actual_distance = distance * self.distance_scale
        
        logger.debug(f"Body distance calc: w={w}px, fx={self.fx}, ref={ref_width}m, "
                    f"dist={actual_distance:.2f}m, bbox_ratio={bbox_area_ratio:.2f}")
        
        return actual_distance
    

    def calc_head_distance(self, keypoints: Dict[str, List[float]]) -> Tuple[float, float]:
        """
        基于头部尺寸计算距离
        
        Args:
            keypoints: 人体关键点字典
            
        Returns:
            (距离, 置信度) - 置信度表示头部检测的可靠性
        """
        # 头部关键点：眼睛和耳朵
        left_eye = keypoints.get('L_eye')
        right_eye = keypoints.get('R_eye')
        left_ear = keypoints.get('L_ear')
        right_ear = keypoints.get('R_ear')
        nose = keypoints.get('nose')
        
        distances = []

        # 方法1：双眼间距
        if left_eye and right_eye:
            eye_dist = np.sqrt((left_eye[0] - right_eye[0])**2 + (left_eye[1] - right_eye[1])**2)
            if eye_dist > 5:  # 最小像素距离
                dist = (REFERENCE_EYE_DISTANCE * self.fx) / eye_dist
                distances.append(dist)

        # 方法2：耳朵间距
        if left_ear and right_ear:
            ear_dist = np.sqrt((left_ear[0] - right_ear[0])**2 + (left_ear[1] - right_ear[1])**2)
            if ear_dist > 8:
                dist = (REFERENCE_EAR_DISTANCE * self.fx) / ear_dist
                distances.append(dist)

        # 方法3：眼到鼻子的距离
        if nose:
            if left_eye:
                eye_nose_dist = np.sqrt((left_eye[0] - nose[0])**2 + (left_eye[1] - nose[1])**2)
                if eye_nose_dist > 3:
                    dist = (REFERENCE_EYE_NOSE_DISTANCE * self.fx) / eye_nose_dist
                    distances.append(dist)
            if right_eye:
                eye_nose_dist = np.sqrt((right_eye[0] - nose[0])**2 + (right_eye[1] - nose[1])**2)
                if eye_nose_dist > 3:
                    dist = (REFERENCE_EYE_NOSE_DISTANCE * self.fx) / eye_nose_dist
                    distances.append(dist)
        
        if not distances:
            return 0.0, 0.0
        
        # 取中位数作为估计
        median_dist = np.median(distances)
        confidence = len(distances) / 4.0  # 最多 4 种方法
        
        return float(median_dist * self.distance_scale), float(confidence)


    def calc_close_range_distance(
        self,
        body_distance: float,
        head_distance: float,
        head_confidence: float,
        bbox: List[int],
        keypoints: Dict[str, List[float]],
        image_height: float = 1080.0,
        image_width: float = 1920.0
    ) -> Tuple[float, str]:
        """
        近距离距离估计优化

        在近距离时，使用三角函数和AOA方法提高精度

        Args:
            body_distance: 基于身体的距离估计
            head_distance: 基于头部的距离估计
            head_confidence: 头部检测置信度
            bbox: 人体边界框
            keypoints: 人体关键点
            image_height: 图像高度
            image_width: 图像宽度

        Returns:
            (优化后的距离, 估计方法描述)
        """
        params = self.close_range_params

        # 计算边界框占比（使用传入的实际图像尺寸）
        bbox_area_ratio = (bbox[2] * bbox[3]) / (image_width * image_height)

        # 极近距离时强制使用V2方法（不管body_distance是多少）
        is_ultra_close_by_bbox = bbox_area_ratio > 0.5

        # 判断是否为近距离
        is_close_range = body_distance < params.threshold or (head_distance > 0 and head_distance < params.threshold) or is_ultra_close_by_bbox
        is_ultra_close = body_distance < params.ultra_close_threshold or (head_distance > 0 and head_distance < params.ultra_close_threshold) or is_ultra_close_by_bbox

        # 近距离时使用V2方法（三角函数+AOA）
        if is_close_range:
            dist_v2, method_v2, conf_v2 = self.calc_close_range_distance_v2(
                bbox, keypoints, image_height=image_height, image_width=image_width
            )
            # 极近距离时降低置信度阈值，确保能返回结果
            conf_threshold = 0.3 if is_ultra_close_by_bbox else 0.5
            if conf_v2 > conf_threshold:
                logger.info(f"V2 distance: {dist_v2:.2f}m, method: {method_v2}, conf: {conf_v2:.2f}, bbox_ratio: {bbox_area_ratio:.2f}")
                return dist_v2, f"v2_{method_v2}"

        if not is_close_range:
            # 远距离：正常融合
            if head_distance > 0 and head_confidence > 0.3:
                distance = body_distance * 0.6 + head_distance * 0.4
                method = "body+head"
            else:
                distance = body_distance
                method = "body"
            return distance, method

        # 检查身体是否完整（bbox 高度是否合理）
        bbox_ratio = bbox[3] / max(bbox[2], 1)  # 高宽比
        body_complete = bbox_ratio > 1.0  # 正常人体高宽比 > 1

        # 计算身体在画面中的位置（判断是否超出画面边界）
        bbox_y2 = bbox[1] + bbox[3]
        bbox_x2 = bbox[0] + bbox[2]

        body_clipped = (bbox_y2 >= image_height - 10 or  # 底部被裁剪
                       bbox[1] <= 10 or                   # 顶部被裁剪
                       bbox_x2 >= image_width - 10 or     # 右侧被裁剪
                       bbox[0] <= 10)                     # 左侧被裁剪

        # 根据情况选择估计策略
        if is_ultra_close:
            # 超近距离：主要依赖头部
            if head_distance > 0 and head_confidence > 0.3:
                distance = head_distance * params.head_weight + body_distance * params.body_weight
                method = "ultra_close_head"
            else:
                # 头部不可靠，使用透视校正
                distance = self._apply_perspective_correction(body_distance, bbox, keypoints)
                method = "ultra_close_corrected"
        elif body_clipped:
            # 身体被裁剪：主要依赖头部
            if head_distance > 0 and head_confidence > 0.3:
                distance = head_distance * 0.8 + body_distance * 0.2
                method = "close_clipped_head"
            else:
                distance = body_distance
                method = "close_clipped_body"
        elif not body_complete:
            # 身体不完整但未被裁剪：融合估计
            if head_distance > 0 and head_confidence > 0.3:
                distance = head_distance * params.head_weight + body_distance * params.body_weight
                method = "close_incomplete_fused"
            else:
                distance = body_distance
                method = "close_incomplete_body"
        else:
            # 身体完整：正常融合
            if head_distance > 0 and head_confidence > 0.3:
                distance = body_distance * 0.6 + head_distance * 0.4
                method = "close_fused"
            else:
                distance = body_distance
                method = "close_body"

        return distance, method

    def _apply_perspective_correction(
        self,
        distance: float,
        bbox: List[int],
        keypoints: Dict[str, List[float]]
    ) -> float:
        """
        应用透视校正

        近距离时，由于透视效应，简单的相似三角形计算会有误差
        使用关键点位置进行校正

        Args:
            distance: 原始距离估计
            bbox: 人体边界框
            keypoints: 人体关键点

        Returns:
            校正后的距离
        """
        if not self.close_range_params.use_perspective_correction:
            return distance

        # 获取脚踝位置
        left_ankle = keypoints.get('L_ankle')
        right_ankle = keypoints.get('R_ankle')

        # 如果脚踝在画面底部附近，说明人很近
        image_height = 1080
        if left_ankle and left_ankle[1] > image_height * 0.9:
            # 脚踝在画面底部 10%，人非常近
            # 使用脚踝到画面底部的距离进行校正
            ankle_offset = image_height - left_ankle[1]
            if ankle_offset < 50:  # 脚踝几乎贴底
                # 校正系数：根据脚踝位置调整
                correction_factor = 0.85 + 0.15 * (ankle_offset / 50)
                distance = distance * correction_factor

        # 使用头部位置进行校正
        nose = keypoints.get('nose')
        if nose:
            # 头部在画面中心偏上时，人可能更近
            head_y_ratio = nose[1] / image_height
            if head_y_ratio < 0.3:  # 头部在画面上 30%
                # 人可能站得很近，向下看摄像头
                correction_factor = 0.9 + 0.1 * head_y_ratio
                distance = distance * correction_factor

        return distance

    def calc_pose_based_distance(
        self,
        keypoints: Dict[str, List[float]],
        bbox: List[int]
    ) -> Tuple[float, float]:
        """
        基于人体姿态的距离估计

        使用多个身体部位的尺寸比例来估计距离

        Args:
            keypoints: 人体关键点
            bbox: 人体边界框

        Returns:
            (距离, 置信度)
        """
        distances = []
        weights = []

        # 1. 基于肩宽估计
        left_shoulder = keypoints.get('L_shoulder')
        right_shoulder = keypoints.get('R_shoulder')
        if left_shoulder and right_shoulder:
            shoulder_width_px = np.sqrt(
                (right_shoulder[0] - left_shoulder[0])**2 +
                (right_shoulder[1] - left_shoulder[1])**2
            )
            if shoulder_width_px > 20:
                dist = (0.45 * self.fx) / shoulder_width_px  # 平均肩宽 45cm
                distances.append(dist)
                weights.append(1.0)

        # 2. 基于臀宽估计
        left_hip = keypoints.get('L_hip')
        right_hip = keypoints.get('R_hip')
        if left_hip and right_hip:
            hip_width_px = np.sqrt(
                (right_hip[0] - left_hip[0])**2 +
                (right_hip[1] - left_hip[1])**2
            )
            if hip_width_px > 15:
                dist = (0.35 * self.fx) / hip_width_px  # 平均臀宽 35cm
                distances.append(dist)
                weights.append(0.8)

        # 3. 基于躯干高度估计
        if left_shoulder and left_hip:
            torso_height_px = np.sqrt(
                (left_shoulder[0] - left_hip[0])**2 +
                (left_shoulder[1] - left_hip[1])**2
            )
            if torso_height_px > 30:
                dist = (0.50 * self.fx) / torso_height_px  # 平均躯干高 50cm
                distances.append(dist)
                weights.append(0.7)

        # 4. 基于大腿长度估计
        left_knee = keypoints.get('L_knee')
        if left_hip and left_knee:
            thigh_length_px = np.sqrt(
                (left_hip[0] - left_knee[0])**2 +
                (left_hip[1] - left_knee[1])**2
            )
            if thigh_length_px > 20:
                dist = (0.45 * self.fx) / thigh_length_px  # 平均大腿长 45cm
                distances.append(dist)
                weights.append(0.6)

        if not distances:
            return 0.0, 0.0

        # 加权平均
        distances = np.array(distances)
        weights = np.array(weights)
        weighted_dist = np.sum(distances * weights) / np.sum(weights)

        # 置信度基于方法数量
        confidence = min(1.0, len(distances) / 3.0)

        return float(weighted_dist * self.distance_scale), float(confidence)



    def calc_trigonometric_distance(
        self,
        foot_y: float,
        image_height: float = 1080.0
    ) -> Tuple[float, float]:
        """
        基于三角函数的距离估计（利用相机俯角和像素位置）

        原理：
        - 相机安装高度 h，俯角 θ
        - 人体脚部在画面中的位置对应一个视角 α
        - 利用三角函数：d = h * tan(θ + α) 或根据具体情况调整

        Args:
            foot_y: 脚部像素Y坐标
            image_height: 图像高度

        Returns:
            (距离, 置信度)
        """
        # 相机参数
        h = self.camera_height  # 安装高度
        theta = np.radians(self.pitch_angle)  # 俯角转弧度
        fov_v = np.radians(self.close_range_params.threshold)  # 这里借用，实际应该用fov

        # 计算脚部位置对应的角度
        # 画面中心对应俯角方向
        # 画面底部对应更近的距离
        y_center = image_height / 2

        # 像素位置转换为角度偏移
        # 假设垂直视场角为60度
        fov_vertical = np.radians(60.0)  # 垂直视场角
        pixel_angle = (foot_y - y_center) / y_center * (fov_vertical / 2)

        # 计算距离
        # 当人站在相机正前方时：
        # - 脚在画面底部：人很近
        # - 脚在画面中心：人较远
        # - 脚在画面顶部：人很远或超出视野

        # 角度关系：
        # 相机光轴方向与地面的夹角 = 90° - θ
        # 脚部方向与光轴的夹角 = pixel_angle
        # 脚部方向与地面的夹角 = 90° - θ + pixel_angle

        alpha = pixel_angle  # 脚部方向与光轴的夹角

        # 计算脚部方向与地面的夹角
        ground_angle = np.pi / 2 - theta + alpha

        # 避免除零和负角度
        if ground_angle <= 0.01:
            # 人太近，脚部超出画面顶部
            return 0.2, 0.3  # 返回更小的最小距离估计

        # 三角函数计算距离
        # d = h / tan(ground_angle)
        distance = h / np.tan(ground_angle)
        
        # 限制最大和最小距离
        distance = np.clip(distance, 0.15, 10.0)  # 最小15cm，最大10米

        # 置信度：脚部越靠近画面底部，置信度越高
        confidence = min(1.0, foot_y / image_height)

        return float(max(0.3, distance)), float(confidence)

    def calc_aoa_distance(
        self,
        keypoints: Dict[str, List[float]],
        bbox: List[int],
        image_height: float = 1080.0,
        image_width: float = 1920.0
    ) -> Tuple[float, float]:
        """
        基于AOA（Angle of Arrival）的距离估计

        利用人体多个关键点的角度信息来估计距离

        原理：
        - 头部和脚部在画面中的角度差可以反映距离
        - 近距离时角度差大，远距离时角度差小

        Args:
            keypoints: 人体关键点
            bbox: 人体边界框
            image_height: 图像高度
            image_width: 图像宽度

        Returns:
            (距离, 置信度)
        """
        # 获取关键点
        nose = keypoints.get('nose')
        left_ankle = keypoints.get('L_ankle')
        right_ankle = keypoints.get('R_ankle')

        # 如果没有脚踝，使用bbox底部
        if left_ankle or right_ankle:
            foot_y = max(
                left_ankle[1] if left_ankle else 0,
                right_ankle[1] if right_ankle else 0
            )
        else:
            foot_y = bbox[1] + bbox[3]  # bbox底部

        # 如果没有鼻子，使用bbox顶部
        head_y = nose[1] if nose else bbox[1]

        # 计算人体在画面中的角度跨度
        fov_vertical = np.radians(60.0)  # 垂直视场角
        y_center = image_height / 2

        # 头部和脚部的角度
        head_angle = (head_y - y_center) / y_center * (fov_vertical / 2)
        foot_angle = (foot_y - y_center) / y_center * (fov_vertical / 2)

        # 角度跨度
        angle_span = abs(foot_angle - head_angle)

        if angle_span < 0.01:
            return 0.0, 0.0

        # 利用角度跨度和人体实际高度估计距离
        # 假设人体高度约1.7m
        person_height = 1.7

        # 距离 = 身高 / (2 * tan(angle_span/2))
        # 简化：distance ≈ height / angle_span (小角度近似)
        distance = person_height / (2 * np.tan(angle_span / 2))

        # 置信度：基于关键点检测
        confidence = 0.5
        if nose:
            confidence += 0.2
        if left_ankle or right_ankle:
            confidence += 0.3

        return float(distance), float(min(1.0, confidence))

    def calc_close_range_distance_v2(
        self,
        bbox: List[int],
        keypoints: Dict[str, List[float]],
        image_height: float = 1080.0,
        image_width: float = 1920.0
    ) -> Tuple[float, str, float]:
        """
        近距离距离估计V2 - 综合三角函数和AOA方法

        专门针对近距离（<1.5m）优化，支持身体部位检测

        Args:
            bbox: 人体边界框
            keypoints: 人体关键点
            image_height: 图像高度
            image_width: 图像宽度

        Returns:
            (距离, 方法描述, 置信度)
        """
        distances = []
        methods = []
        weights = []
        
        # ===== 身体部位检测 =====
        body_part, part_confidence, part_info = self.detect_body_part(
            keypoints, bbox, image_height=image_height, image_width=image_width
        )
        bbox_area_ratio = part_info['bbox_area_ratio']
        bbox_height_ratio = part_info['bbox_height_ratio']
        
        # ===== 超近距离强制判断（脸贴摄像头） =====
        # 当画面占比超过70%时，直接判定为超近距离（0.2-0.35米）
        if bbox_area_ratio > 0.70:
            # 脸贴摄像头的情况，直接返回固定距离
            if body_part == 'head_only':
                return 0.25, "extreme_close_head_only", 0.95
            elif body_part in ['upper_body', 'half_body']:
                return 0.30, "extreme_close_upper_body", 0.95
            else:
                return 0.35, "extreme_close_full_body", 0.95
        
        # ===== 极近距离判断（边界框占比） =====
        # 当边界框占画面很大比例时，说明人非常近
        # 重新校准的距离公式（基于实测数据）
        if bbox_area_ratio > 0.20:  # 从0.25降低到0.20，更早启用
            # 根据身体部位调整估算
            if body_part == 'head_only':
                # 只有头部：占比20%时约0.5米，占比60%时约0.25米，占比80%时约0.15米
                # 公式：distance = 0.7 - 1.1 * (ratio - 0.2)
                estimated_dist = 0.7 - (bbox_area_ratio - 0.2) * 1.1
                estimated_dist = max(0.12, min(0.7, estimated_dist))
            elif body_part in ['upper_body', 'half_body']:
                # 上半身/半身：占比20%时约0.65米，占比60%时约0.3米，占比80%时约0.15米
                # 公式：distance = 0.85 - 1.25 * (ratio - 0.2)
                estimated_dist = 0.85 - (bbox_area_ratio - 0.2) * 1.25
                estimated_dist = max(0.15, min(0.85, estimated_dist))
            else:
                # 全身或其他：占比20%时约0.8米，占比60%时约0.35米，占比80%时约0.15米
                # 公式：distance = 0.95 - 1.3 * (ratio - 0.2)
                estimated_dist = 0.95 - (bbox_area_ratio - 0.2) * 1.3
                estimated_dist = max(0.15, min(1.0, estimated_dist))

            distances.append(estimated_dist)
            methods.append(f"bbox_ratio_{body_part}")
            weights.append(0.95)  # 极高权重

            # 如果占比超过45%，直接返回这个估计（人已经非常近了）
            if bbox_area_ratio > 0.45:  # 从0.5降低到0.45
                return estimated_dist, f"ultra_close_{body_part}", 0.9
        
        # ===== 根据身体部位选择估计策略 =====
        
        # 1. 基于三角函数的距离（利用脚部位置）- 仅当有下半身时
        if body_part in ['full_body', 'lower_body']:
            left_ankle = keypoints.get('L_ankle')
            right_ankle = keypoints.get('R_ankle')
            
            if left_ankle or right_ankle:
                foot_y = max(
                    left_ankle[1] if left_ankle else 0,
                    right_ankle[1] if right_ankle else 0
                )
                dist_trig, conf_trig = self.calc_trigonometric_distance(foot_y, image_height)
                if dist_trig > 0:
                    distances.append(dist_trig)
                    methods.append("trigonometric")
                    weights.append(conf_trig * 1.0)

        # 2. 基于AOA的距离
        dist_aoa, conf_aoa = self.calc_aoa_distance(keypoints, bbox, image_height, image_width)
        if dist_aoa > 0:
            distances.append(dist_aoa)
            methods.append("aoa")
            weights.append(conf_aoa * 0.8)

        # 3. 基于头部尺寸的距离（近距离时头部检测更可靠）
        if body_part in ['full_body', 'upper_body', 'half_body', 'head_only']:
            head_dist, head_conf = self.calc_head_distance(keypoints)
            if head_dist > 0 and head_conf > 0.2:
                distances.append(head_dist)
                methods.append("head")
                # 根据身体部位调整头部权重
                if body_part == 'head_only':
                    weights.append(head_conf * 1.0)  # 只有头时完全依赖头部
                elif body_part == 'upper_body':
                    weights.append(head_conf * 0.95)  # 上半身时头部很可靠
                else:
                    weights.append(head_conf * 0.9)

        # 4. 基于身体宽度的距离（仅当身体较完整时）
        if body_part in ['full_body', 'half_body']:
            body_dist = self.calc_person_distance(bbox)
            if body_dist > 0:
                distances.append(body_dist)
                methods.append("body")
                weights.append(0.5)

        if not distances:
            # 如果边界框很大但没有其他估计，使用边界框比例估算
            if bbox_area_ratio > 0.15:
                estimated_dist = 0.8 - bbox_area_ratio * 1.5
                return max(0.15, estimated_dist), f"bbox_fallback_{body_part}", 0.5
            return 0.5, f"fallback_{body_part}", 0.3

        # 加权平均
        distances = np.array(distances)
        weights = np.array(weights)
        weighted_dist = np.sum(distances * weights) / np.sum(weights)

        # 综合置信度
        confidence = np.mean(weights) * part_confidence

        method_str = "+".join(methods)

        return float(weighted_dist), f"{method_str}_{body_part}", float(confidence)

    def pixel_to_topview(
        self,
        u: float,
        v: float,
        distance: float
    ) -> Tuple[float, float]:
        """
        像素坐标→顶视图坐标
        
        顶视图坐标系：
        - 原点在摄像头位置（画面底部中心）
        - X轴：水平方向，正方向向右
        - Y轴：深度方向，正方向向前（远离摄像头）
        """
        # 计算相对于画面中心的水平偏移（米）
        x_offset = (u - self.cx) / self.fx * distance
        
        # 深度直接使用距离
        depth = distance
        
        # 转换为顶视图坐标
        # 摄像头在底部中心，人向前走时 Y 坐标减小
        x_top = self.topview_origin[0] + x_offset * self.topview_scale
        y_top = self.topview_origin[1] - depth * self.topview_scale

        return (x_top, y_top)

    def _get_timestamp(self) -> float:
        """获取时间戳"""
        import time
        return time.time()
    
    def get_calibration_status(self) -> Dict[str, Any]:
        """获取校准状态"""
        return {
            'distance_scale': round(self.distance_scale, 4),
            'height_scale': round(self.height_scale, 4),
            'size_scale': round(self.size_scale, 4),
            'calibration_records': len(self.calibration_history),
            'distance_correction_coeffs': self.distance_correction_coeffs,
            # 新增参数状态
            'head_params': {
                'ref_head_width': self.head_params.ref_head_width,
                'ref_eye_distance': self.head_params.ref_eye_distance,
                'ref_ear_distance': self.head_params.ref_ear_distance,
                'ref_eye_nose_distance': self.head_params.ref_eye_nose_distance,
                'ref_head_height': self.head_params.ref_head_height
            },
            'kalman_params': {
                'max_speed': self.kalman_params.max_speed,
                'process_noise': self.kalman_params.process_noise,
                'measurement_noise': self.kalman_params.measurement_noise
            },
            'close_range_params': {
                'threshold': self.close_range_params.threshold,
                'ultra_close_threshold': self.close_range_params.ultra_close_threshold,
                'head_weight': self.close_range_params.head_weight,
                'body_weight': self.close_range_params.body_weight,
                'use_perspective_correction': self.close_range_params.use_perspective_correction
            }
        }
    
    def reset_calibration(self):
        """重置校准参数"""
        self.distance_scale = 1.0
        self.height_scale = 1.0
        self.size_scale = 1.0
        self.distance_correction_coeffs = [0.0, 0.0, 1.0]
        self.calibration_history = []
        self.clear_buffer()
        logger.info("Calibration reset")


def create_sample_calib_params() -> CalibrationParams:
    """创建示例标定参数"""
    return CalibrationParams(
        fx=1200.0,
        fy=1200.0,
        cx=960.0,
        cy=540.0,
        dist_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
        image_size=(1920, 1080)
    )


def main():
    """测试增强版空间计算器"""
    # 创建标定参数
    calib_params = create_sample_calib_params()
    
    # 初始化计算器
    calc = SpatialCalculatorEnhanced(calib_params)
    calc.set_camera_extrinsics(height=1.8, pitch_angle=30.0)
    
    # 测试：添加校准记录
    calc.add_calibration_record(
        measured_pixels=100,
        actual_value=5.0,
        distance=5.0,
        measurement_type='distance'
    )
    
    # 测试：多帧融合
    print("Testing multi-frame fusion...")
    for i in range(10):
        # 模拟测量值（带噪声）
        measured_distance = 3.0 + np.random.randn() * 0.1
        result = calc.add_measurement('distance', measured_distance, track_id=1)
        print(f"  Sample {i+1}: {result.value:.3f}m ± {result.std_dev:.3f}m "
              f"(confidence: {result.confidence:.2f})")
    
    # 测试：误差修正
    print("\nTesting error correction...")
    raw_distance = 3.0
    corrected = calc.apply_distance_correction(raw_distance)
    print(f"  Raw: {raw_distance:.2f}m, Corrected: {corrected:.2f}m")
    
    # 获取校准状态
    print("\nCalibration Status:")
    status = calc.get_calibration_status()
    for key, value in status.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
