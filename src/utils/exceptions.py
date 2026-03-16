"""
统一异常处理模块

定义所有自定义异常类和错误处理工具
"""

from typing import Optional, Dict, Any
from enum import Enum
from loguru import logger


class ErrorCode(Enum):
    """错误代码枚举"""
    # 系统错误 (1000-1999)
    UNKNOWN_ERROR = 1000
    INITIALIZATION_ERROR = 1001
    CONFIGURATION_ERROR = 1002
    
    # 摄像头错误 (2000-2999)
    CAMERA_NOT_FOUND = 2000
    CAMERA_OPEN_FAILED = 2001
    CAMERA_READ_FAILED = 2002
    CAMERA_RESOLUTION_ERROR = 2003
    
    # 检测错误 (3000-3999)
    DETECTION_MODEL_ERROR = 3000
    DETECTION_INFERENCE_ERROR = 3001
    DETECTION_TIMEOUT = 3002
    
    # 空间计算错误 (4000-4999)
    SPATIAL_CALCULATION_ERROR = 4000
    CALIBRATION_ERROR = 4001
    INVALID_MEASUREMENT = 4002
    
    # 网络错误 (5000-5999)
    NETWORK_ERROR = 5000
    WEBSOCKET_ERROR = 5001
    API_ERROR = 5002


class CameraPerceptionException(Exception):
    """基础异常类"""
    
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'error_code': self.error_code.value,
            'error_name': self.error_code.name,
            'message': self.message,
            'details': self.details
        }


class CameraException(CameraPerceptionException):
    """摄像头相关异常"""
    pass


class DetectionException(CameraPerceptionException):
    """检测相关异常"""
    pass


class SpatialException(CameraPerceptionException):
    """空间计算相关异常"""
    pass


class CalibrationException(CameraPerceptionException):
    """标定相关异常"""
    pass


class ErrorHandler:
    """错误处理器"""
    
    def __init__(self):
        self.error_counts: Dict[ErrorCode, int] = {}
        self.error_history = []
        self.max_history = 100
        
    def handle(self, exception: Exception, context: str = "") -> Dict[str, Any]:
        """
        处理异常
        
        Args:
            exception: 异常对象
            context: 错误上下文
            
        Returns:
            错误信息字典
        """
        if isinstance(exception, CameraPerceptionException):
            error_info = exception.to_dict()
            error_code = exception.error_code
        else:
            error_info = {
                'error_code': ErrorCode.UNKNOWN_ERROR.value,
                'error_name': type(exception).__name__,
                'message': str(exception),
                'details': {'context': context}
            }
            error_code = ErrorCode.UNKNOWN_ERROR
        
        # 记录错误统计
        self.error_counts[error_code] = self.error_counts.get(error_code, 0) + 1
        
        # 记录错误历史
        import time
        self.error_history.append({
            'timestamp': time.time(),
            'error_code': error_code.value,
            'message': error_info['message'],
            'context': context
        })
        
        # 限制历史记录大小
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        # 记录日志
        logger.error(f"[{context}] {error_info['message']}", 
                    extra={'error_code': error_code.value})
        
        return error_info
    
    def get_stats(self) -> Dict[str, Any]:
        """获取错误统计"""
        return {
            'error_counts': {k.value: v for k, v in self.error_counts.items()},
            'total_errors': sum(self.error_counts.values()),
            'recent_errors': self.error_history[-10:]
        }


# 全局错误处理器
_error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """获取全局错误处理器"""
    return _error_handler
