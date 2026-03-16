"""
空间计算核心模块
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from loguru import logger

from .kalman import DistanceKalmanFilter, KalmanParams
from .body_detection import BodyPartDetector
from .distance_estimation import DistanceEstimator
from src.algorithms.calibration import CalibrationParams
from src.algorithms.depth_enhanced_estimation import DepthEnhancedDistanceEstimator
from src.utils.constants import (
    DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT,
    ULTRA_CLOSE_BBOX_RATIO,
    MEASUREMENT_BUFFER_SIZE,
    KALMAN_DT_MIN, KALMAN_DT_MAX
)


@dataclass
class MeasurementResult:
    """测量结果"""
    value: float
    unit: str
    confidence: float
    std_dev: float
    num_samples: int
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            'value': round(self.value, 2),
            'unit': self.unit,
            'confidence': round(self.confidence, 3),
            'std_dev': round(self.std_dev, 4),
            'num_samples': self.num_samples
        }


class SpatialCalculatorEnhanced:
    """增强版空间计量器"""

    def __init__(self, calibration_params: CalibrationParams,
                 kalman_params: Optional[KalmanParams] = None):
        """
        初始化空间计量器

        Args:
            calibration_params: 标定参数
            kalman_params: 卡尔曼滤波参数
        """
        self.calib = calibration_params
        self.fx = calibration_params.fx
        self.fy = calibration_params.fy
        self.cx = calibration_params.cx
        self.cy = calibration_params.cy

        # 初始化子模块
        self.body_detector = BodyPartDetector()
        self.distance_estimator = DistanceEstimator(fx=self.fx)
        
        # 深度增强估计器（可选）
        self.depth_enhanced_estimator = None
        self.use_depth_estimation = True  # 默认启用

        # 卡尔曼滤波器管理
        self.kalman_filters: Dict[int, DistanceKalmanFilter] = {}
        self.last_update_time: Dict[int, float] = {}
        self.kalman_params = kalman_params or KalmanParams()

        # 测量缓冲区
        self.measurement_buffers: Dict[str, deque] = {}

        logger.info(f"SpatialCalculatorEnhanced initialized: fx={self.fx}, fy={self.fy}")

    def calc_person_metrics(self, person: Dict[str, Any],
                           image: Optional[np.ndarray] = None,
                           image_height: float = DEFAULT_IMAGE_HEIGHT,
                           image_width: float = DEFAULT_IMAGE_WIDTH) -> Dict[str, Any]:
        """
        计算人体综合指标（支持深度增强）

        Args:
            person: 人体检测结果
            image: 原始图像（用于深度估计）
            image_height: 图像高度
            image_width: 图像宽度

        Returns:
            包含各项指标的字典
        """
        import time
        start_time = time.time()

        try:
            bbox = person['bbox']
            keypoints = person.get('keypoints', {})
            track_id = person.get('track_id')

            # 身体部位检测
            body_part, part_confidence, part_info = self.body_detector.detect(
                keypoints, bbox, image_height, image_width
            )

            # 距离估计：优先使用深度增强估计
            if self.use_depth_estimation and image is not None:
                # 初始化深度增强估计器
                if self.depth_enhanced_estimator is None:
                    self.depth_enhanced_estimator = DepthEnhancedDistanceEstimator(
                        fx=self.fx, use_depth=True
                    )
                
                # 使用深度增强估计
                distance, confidence, method = self.depth_enhanced_estimator.estimate(
                    image, bbox, keypoints, body_part
                )
                estimate_method = method
                
                # 备用：边界框估计
                body_distance = self.distance_estimator.estimate_from_bbox(
                    bbox, image_width, image_height
                )
            else:
                # 传统几何估计
                body_distance = self.distance_estimator.estimate_from_bbox(
                    bbox, image_width, image_height
                )

                # 尝试头部估计
                head_distance, head_confidence = self.distance_estimator.estimate_from_head(keypoints)

                # 如果头部估计不可用，使用身体关键点估计
                body_keypoint_distance = 0.0
                body_keypoint_confidence = 0.0
                if head_distance == 0 or head_confidence < 0.3:
                    body_keypoint_distance, body_keypoint_confidence = \
                        self.distance_estimator.estimate_from_body_keypoints(keypoints)

                # 选择最佳估计
                if head_confidence >= 0.3:
                    distance = head_distance
                    confidence = head_confidence
                    primary_method = "head"
                elif body_keypoint_confidence >= 0.3:
                    distance = body_keypoint_distance
                    confidence = body_keypoint_confidence
                    primary_method = "body_keypoints"
                else:
                    distance = body_distance
                    confidence = 0.5
                    primary_method = "bbox"

                # 近距离优化
                distance, estimate_method = self.distance_estimator.estimate_close_range(
                    body_distance, distance, confidence,
                    bbox, keypoints, image_height, image_width, body_part
                )

                if primary_method == "body_keypoints":
                    estimate_method += "_body_kp"
        except Exception as e:
            logger.error(f"Error in spatial calculation: {e}")
            # 返回默认值
            return {
                "distance": 0.0,
                "height": 0.0,
                "topview": {"x": 0, "y": 0},
                "estimate_method": "error",
                "body_part": "unknown",
                "error": str(e)
            }

        # 检查是否为极近距离
        bbox_area_ratio = (bbox[2] * bbox[3]) / (image_width * image_height)
        is_ultra_close = bbox_area_ratio > ULTRA_CLOSE_BBOX_RATIO

        # 卡尔曼滤波
        if is_ultra_close:
            # 极近距离跳过滤波
            filtered_distance = distance
            kalman_info = {'velocity': 0.0, 'uncertainty': 0.1, 'motion_state': 'static'}
            estimate_method += "_no_kf"
        else:
            filtered_distance, kalman_info = self._apply_kalman_filter(
                distance, track_id, head_confidence, bbox
            )
            estimate_method += f"_kf(v={kalman_info['velocity']:.1f})"

        # 计算身高
        height = self._estimate_height(keypoints, filtered_distance)

        # 顶视图坐标
        topview = self._pixel_to_topview(
            bbox[0] + bbox[2] // 2,
            bbox[1] + bbox[3] // 2,
            filtered_distance
        )

        # 构建结果
        # 计算处理时间
        calc_time = (time.time() - start_time) * 1000

        result = {
            "distance": round(filtered_distance, 2),
            "height": round(height, 1),
            "topview": {"x": round(topview[0], 1), "y": round(topview[1], 1)},
            "estimate_method": estimate_method,
            "body_part": body_part,
            "body_part_confidence": round(part_confidence, 2),
            "bbox_area_ratio": round(part_info['bbox_area_ratio'], 3),
            "bbox_height_ratio": round(part_info['bbox_height_ratio'], 3),
            "image_size": [int(image_width), int(image_height)],
            "bbox": bbox,
            "keypoints": keypoints,
            "track_id": track_id,
            "velocity": round(kalman_info['velocity'], 2),
            "motion_state": kalman_info['motion_state'],
            "calc_time_ms": round(calc_time, 2)  # 添加计算时间
        }

        return result

    def _apply_kalman_filter(self, distance: float, track_id: Optional[int],
                            head_confidence: float, bbox: List[int]) -> Tuple[float, Dict]:
        """应用卡尔曼滤波"""
        if track_id is None:
            return distance, {'velocity': 0.0, 'uncertainty': 0.1, 'motion_state': 'unknown'}

        current_time = time.time()

        # 初始化或获取滤波器
        if track_id not in self.kalman_filters:
            self.kalman_filters[track_id] = DistanceKalmanFilter(
                initial_distance=distance,
                max_speed=self.kalman_params.max_speed,
                process_noise=self.kalman_params.process_noise,
                measurement_noise=self.kalman_params.measurement_noise
            )
            self.last_update_time[track_id] = current_time

        # 计算时间间隔
        last_time = self.last_update_time.get(track_id, current_time)
        dt = min(max(current_time - last_time, KALMAN_DT_MIN), KALMAN_DT_MAX)
        self.last_update_time[track_id] = current_time

        # 计算测量置信度
        measurement_confidence = 0.5
        if head_confidence > 0.3:
            measurement_confidence += 0.3 * head_confidence
        if bbox[3] / max(bbox[2], 1) > 1.0:
            measurement_confidence += 0.2

        # 应用滤波
        kf = self.kalman_filters[track_id]
        filtered_distance, velocity = kf.filter(distance, dt, measurement_confidence)

        return filtered_distance, kf.get_state()

    def _estimate_height(self, keypoints: Dict[str, List[float]], distance: float) -> float:
        """估算身高"""
        if not keypoints:
            return 0.0

        head = keypoints.get('nose') or keypoints.get('L_eye') or keypoints.get('R_eye')
        left_ankle = keypoints.get('L_ankle')
        right_ankle = keypoints.get('R_ankle')

        if head and (left_ankle or right_ankle):
            foot = left_ankle or right_ankle
            pixel_height = abs(foot[1] - head[1])
            # 简化的身高估算
            height = (pixel_height * distance) / self.fx
            return min(height * 1.2, 2.5)  # 限制在合理范围

        return 0.0

    def _pixel_to_topview(self, u: int, v: int, distance: float) -> Tuple[float, float]:
        """像素坐标转顶视图坐标"""
        # 简化的顶视图转换
        x = (u - self.cx) * distance / self.fx
        y = distance
        return x, y

    def calc_hand_metrics(self, hand: Dict[str, Any]) -> Dict[str, Any]:
        """计算手部指标"""
        keypoints = hand.get('keypoints', [])
        bbox = hand['bbox']

        if len(keypoints) < 21:
            return {}

        hand_width = bbox[2]
        ref_hand_width = 0.08  # 8cm
        distance = (ref_hand_width * self.fx) / hand_width if hand_width > 0 else 1.0

        return {
            "size": round(hand_width * 0.1, 1),
            "distance": round(distance, 2),
            "handedness": hand.get('handedness', 'Unknown')
        }
