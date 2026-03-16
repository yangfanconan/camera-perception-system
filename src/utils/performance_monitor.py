"""
性能监控模块
功能：
1. FPS监控
2. 内存使用监控
3. 算法耗时统计
4. 性能瓶颈分析
"""

import time
import psutil
import os
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
from loguru import logger
from functools import wraps


@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: float
    fps: float
    memory_mb: float
    cpu_percent: float
    detection_time_ms: float
    spatial_calc_time_ms: float
    total_frame_time_ms: float


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self, history_size: int = 100):
        """
        初始化性能监控器

        Args:
            history_size: 历史数据保留数量
        """
        self.history_size = history_size
        self.metrics_history: deque = deque(maxlen=history_size)
        self.process = psutil.Process(os.getpid())

        # 帧率计算
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0

        # 算法耗时统计
        self.detection_times: deque = deque(maxlen=30)
        self.spatial_times: deque = deque(maxlen=30)
        self.frame_times: deque = deque(maxlen=30)

        # 回调函数
        self.alert_callbacks: List[Callable] = []

        logger.info("PerformanceMonitor initialized")

    def start_frame(self) -> float:
        """开始一帧处理"""
        return time.time()

    def end_frame(self, start_time: float, detection_time: float = 0, spatial_time: float = 0):
        """结束一帧处理"""
        end_time = time.time()
        frame_time = (end_time - start_time) * 1000  # ms

        self.frame_count += 1
        self.frame_times.append(frame_time)

        if detection_time > 0:
            self.detection_times.append(detection_time)
        if spatial_time > 0:
            self.spatial_times.append(spatial_time)

        # 每秒更新FPS
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time

            # 记录指标
            self._record_metrics(detection_time, spatial_time, frame_time)

        return frame_time

    def _record_metrics(self, detection_time: float, spatial_time: float, frame_time: float):
        """记录性能指标"""
        try:
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            cpu_percent = self.process.cpu_percent()

            metrics = PerformanceMetrics(
                timestamp=time.time(),
                fps=self.current_fps,
                memory_mb=memory_mb,
                cpu_percent=cpu_percent,
                detection_time_ms=detection_time,
                spatial_calc_time_ms=spatial_time,
                total_frame_time_ms=frame_time
            )

            self.metrics_history.append(metrics)

            # 检查性能警报
            self._check_alerts(metrics)

        except Exception as e:
            logger.error(f"Failed to record metrics: {e}")

    def _check_alerts(self, metrics: PerformanceMetrics):
        """检查性能警报"""
        alerts = []

        if metrics.fps < 10:
            alerts.append(f"Low FPS: {metrics.fps:.1f}")
        if metrics.memory_mb > 1000:
            alerts.append(f"High memory usage: {metrics.memory_mb:.1f}MB")
        if metrics.detection_time_ms > 100:
            alerts.append(f"Slow detection: {metrics.detection_time_ms:.1f}ms")
        if metrics.total_frame_time_ms > 100:
            alerts.append(f"Slow frame processing: {metrics.total_frame_time_ms:.1f}ms")

        for alert in alerts:
            logger.warning(f"Performance alert: {alert}")
            for callback in self.alert_callbacks:
                callback(alert)

    def get_stats(self) -> Dict:
        """获取性能统计"""
        if not self.metrics_history:
            return {}

        recent = list(self.metrics_history)[-10:]

        return {
            'fps': {
                'current': self.current_fps,
                'avg': sum(m.fps for m in recent) / len(recent),
                'min': min(m.fps for m in recent),
                'max': max(m.fps for m in recent)
            },
            'memory_mb': {
                'current': recent[-1].memory_mb if recent else 0,
                'avg': sum(m.memory_mb for m in recent) / len(recent)
            },
            'detection_time_ms': {
                'avg': sum(self.detection_times) / len(self.detection_times) if self.detection_times else 0,
                'max': max(self.detection_times) if self.detection_times else 0
            },
            'spatial_time_ms': {
                'avg': sum(self.spatial_times) / len(self.spatial_times) if self.spatial_times else 0,
                'max': max(self.spatial_times) if self.spatial_times else 0
            }
        }

    def on_alert(self, callback: Callable):
        """注册警报回调"""
        self.alert_callbacks.append(callback)

    def reset(self):
        """重置监控器"""
        self.metrics_history.clear()
        self.detection_times.clear()
        self.spatial_times.clear()
        self.frame_times.clear()
        self.frame_count = 0
        self.current_fps = 0.0


def timed(func):
    """装饰器：统计函数执行时间"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = (time.time() - start) * 1000  # ms
        logger.debug(f"{func.__name__} took {elapsed:.2f}ms")
        return result
    return wrapper


# 全局监控器实例
_monitor: Optional[PerformanceMonitor] = None


def get_monitor() -> PerformanceMonitor:
    """获取全局监控器实例"""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor()
    return _monitor
