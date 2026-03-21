"""
算法模块初始化
"""

from .calibration import CameraCalibrator, CalibrationParams
from .detection import (
    PersonDetector,
    HandDetector,
    CombinedDetector,
    DetectionResult,
    visualize_detections
)
from .spatial import SpatialCalculator
from .spatial_enhanced import (
    SpatialCalculatorEnhanced,
    HeadSizeParams,
    KalmanParams,
    CloseRangeParams
)

__all__ = [
    'CameraCalibrator',
    'CalibrationParams',
    'PersonDetector',
    'HandDetector',
    'CombinedDetector',
    'DetectionResult',
    'visualize_detections',
    'SpatialCalculator',
    'SpatialCalculatorEnhanced',
    'HeadSizeParams',
    'KalmanParams',
    'CloseRangeParams',
]