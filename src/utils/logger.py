"""
日志和错误处理模块
功能：
1. 统一日志配置
2. 异常捕获和记录
3. 错误响应格式化
4. 性能监控
"""

import sys
import time
import traceback
from functools import wraps
from typing import Optional, Callable, Any, Dict, List
from pathlib import Path
from contextlib import contextmanager
from loguru import logger
import threading
import json
from datetime import datetime


# ==================== 日志配置 ====================

class LogFormatter:
    """日志格式化器"""
    
    @staticmethod
    def format(record: dict) -> str:
        """自定义日志格式"""
        # 添加异常堆栈
        if record['exception']:
            exc_type, exc_value, exc_traceback = record['exception']
            record['extra']['exc_info'] = ''.join(
                traceback.format_exception(exc_type, exc_value, exc_traceback)
            )
        else:
            record['extra']['exc_info'] = ''
        
        # 时间戳
        timestamp = record['time'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # 颜色
        colors = {
            'TRACE': '<cyan>',
            'DEBUG': '<blue>',
            'INFO': '<green>',
            'SUCCESS': '<bold green>',
            'WARNING': '<yellow>',
            'ERROR': '<red>',
            'CRITICAL': '<bold red>'
        }
        
        color = colors.get(record['level'].name, '<white>')
        
        # 格式化
        return (
            f"{timestamp} | "
            f"{color}{record['level'].name:^8}{'</>'}"
            f" | <cyan>{record['name']}</cyan>:<cyan>{record['function']}</cyan>:<cyan>{record['line']}</cyan> "
            f"| {record['message']}\n"
            f"{record['extra']['exc_info']}"
        )


def setup_logger(
    level: str = "INFO",
    log_file: str = "logs/app.log",
    rotation: str = "10 MB",
    retention: str = "7 days",
    compression: str = "zip"
) -> None:
    """
    设置日志
    
    Args:
        level: 日志级别
        log_file: 日志文件路径
        rotation: 轮转大小
        retention: 保留时间
        compression: 压缩格式
    """
    # 移除默认处理器
    logger.remove()
    
    # 控制台输出
    logger.add(
        sys.stderr,
        format=LogFormatter.format,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # 文件输出
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        str(log_path),
        format=LogFormatter.format,
        level=level,
        rotation=rotation,
        retention=retention,
        compression=compression,
        backtrace=True,
        diagnose=True
    )
    
    logger.info(f"Logger initialized: level={level}, file={log_file}")


# ==================== 异常处理 ====================

class AppException(Exception):
    """应用基础异常"""
    
    def __init__(self, message: str, code: str = "APP_ERROR", status_code: int = 500):
        self.message = message
        self.code = code
        self.status_code = status_code
        super().__init__(self.message)
    
    def to_dict(self) -> dict:
        return {
            "error": self.code,
            "message": self.message,
            "status_code": self.status_code
        }


class ConfigException(AppException):
    """配置异常"""
    def __init__(self, message: str):
        super().__init__(message, code="CONFIG_ERROR", status_code=500)


class CameraException(AppException):
    """摄像头异常"""
    def __init__(self, message: str):
        super().__init__(message, code="CAMERA_ERROR", status_code=500)


class DetectionException(AppException):
    """检测异常"""
    def __init__(self, message: str):
        super().__init__(message, code="DETECTION_ERROR", status_code=500)


class CalibrationException(AppException):
    """标定异常"""
    def __init__(self, message: str):
        super().__init__(message, code="CALIBRATION_ERROR", status_code=500)


class SpatialException(AppException):
    """空间计量异常"""
    def __init__(self, message: str):
        super().__init__(message, code="SPATIAL_ERROR", status_code=500)


# ==================== 装饰器 ====================

def log_execution(func: Callable) -> Callable:
    """记录函数执行日志"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Executing {func.__name__}")
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"{func.__name__} completed in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
            raise
    
    return wrapper


def handle_exceptions(default_return: Any = None, reraise: bool = False):
    """
    异常处理装饰器
    
    Args:
        default_return: 异常时的默认返回值
        reraise: 是否重新抛出异常
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except AppException as e:
                logger.error(f"App exception in {func.__name__}: {e.message}")
                if reraise:
                    raise
                return default_return
            except Exception as e:
                logger.exception(f"Unexpected exception in {func.__name__}: {e}")
                if reraise:
                    raise
                return default_return
        
        return wrapper
    return decorator


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    重试装饰器
    
    Args:
        max_attempts: 最大重试次数
        delay: 初始延迟（秒）
        backoff: 延迟倍数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts")
                        raise
                    
                    logger.warning(f"{func.__name__} attempt {attempts} failed: {e}, retrying in {current_delay}s")
                    time.sleep(current_delay)
                    current_delay *= backoff
        
        return wrapper
    return decorator


# ==================== 性能监控 ====================

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self._metrics: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
    
    def record(self, name: str, value: float) -> None:
        """记录指标"""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = []
            
            self._metrics[name].append(value)
            
            # 保留最近 1000 个数据
            if len(self._metrics[name]) > 1000:
                self._metrics[name] = self._metrics[name][-1000:]
    
    @contextmanager
    def measure(self, name: str):
        """上下文管理器：测量执行时间"""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.record(name, elapsed)
    
    def get_stats(self, name: str) -> Optional[Dict[str, float]]:
        """获取统计信息"""
        with self._lock:
            if name not in self._metrics or not self._metrics[name]:
                return None
            
            values = self._metrics[name]
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "latest": values[-1]
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """获取所有统计信息"""
        with self._lock:
            return {
                name: self.get_stats(name)
                for name in self._metrics
                if self.get_stats(name) is not None
            }
    
    def reset(self, name: Optional[str] = None) -> None:
        """重置指标"""
        with self._lock:
            if name:
                self._metrics[name] = []
            else:
                self._metrics.clear()


# 全局性能监控器
perf_monitor = PerformanceMonitor()


def measure_performance(name: str):
    """性能测量装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with perf_monitor.measure(name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# ==================== FastAPI 异常处理 ====================

def register_exception_handlers(app) -> None:
    """注册 FastAPI 异常处理器"""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from fastapi.exceptions import RequestValidationError
    
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        logger.error(f"App exception: {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict()
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning(f"Validation error: {exc.errors()}")
        return JSONResponse(
            status_code=400,
            content={
                "error": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": exc.errors()
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_ERROR",
                "message": "An internal error occurred"
            }
        )


# ==================== 中间件 ====================

def create_logging_middleware():
    """创建日志中间件"""
    from fastapi import Request
    import time
    
    async def logging_middleware(request: Request, call_next):
        # 请求信息
        method = request.method
        path = request.url.path
        client = request.client.host if request.client else "unknown"
        
        logger.info(f"Incoming request: {method} {path} from {client}")
        
        # 处理请求
        start_time = time.time()
        response = await call_next(request)
        elapsed = time.time() - start_time
        
        # 响应信息
        logger.info(f"Outgoing response: {method} {path} - {response.status_code} in {elapsed:.3f}s")
        
        # 记录性能指标
        perf_monitor.record(f"request_{path}", elapsed)
        
        return response
    
    return logging_middleware


# ==================== 主函数（测试） ====================

def main():
    """测试日志和异常处理"""
    # 设置日志
    setup_logger(level="DEBUG")
    
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # 测试异常
    try:
        raise AppException("Test exception", code="TEST_ERROR")
    except AppException as e:
        logger.error(f"Caught exception: {e.to_dict()}")
    
    # 测试性能监控
    @measure_performance("test_function")
    def test_function():
        time.sleep(0.1)
        return "done"
    
    for _ in range(10):
        test_function()
    
    # 打印统计
    stats = perf_monitor.get_stats("test_function")
    logger.info(f"Performance stats: {stats}")


if __name__ == '__main__':
    main()
