"""
空间计量算法包

包含以下模块：
- core: 核心计算类
- kalman: 卡尔曼滤波器
- body_detection: 身体部位检测
- distance_estimation: 距离估计算法
- utils: 工具函数
"""

from .core import SpatialCalculatorEnhanced
from .kalman import DistanceKalmanFilter
from .body_detection import BodyPartDetector
from .distance_estimation import DistanceEstimator

__all__ = [
    'SpatialCalculatorEnhanced',
    'DistanceKalmanFilter',
    'BodyPartDetector',
    'DistanceEstimator',
]
