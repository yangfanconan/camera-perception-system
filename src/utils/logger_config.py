"""
日志配置模块
功能：
1. 结构化日志
2. 日志分级
3. 日志轮转
4. 日志分析
"""

import json
import sys
from pathlib import Path
from loguru import logger
from typing import Dict, Any, Optional
from datetime import datetime


class StructuredLog:
    """结构化日志记录器"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 配置日志
        self._setup_logger()
        
    def _setup_logger(self):
        """配置日志处理器"""
        # 移除默认处理器
        logger.remove()
        
        # 控制台输出
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                   "<level>{message}</level>",
            level="INFO",
            colorize=True
        )
        
        # 文件输出 - 详细日志
        logger.add(
            self.log_dir / "app_{time:YYYY-MM-DD}.log",
            rotation="00:00",  # 每天轮转
            retention="30 days",  # 保留30天
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            level="DEBUG",
            encoding="utf-8"
        )
        
        # 文件输出 - 错误日志
        logger.add(
            self.log_dir / "error_{time:YYYY-MM-DD}.log",
            rotation="00:00",
            retention="30 days",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            level="ERROR",
            encoding="utf-8"
        )
        
        # 结构化JSON日志
        logger.add(
            self.log_dir / "structured_{time:YYYY-MM-DD}.json",
            rotation="00:00",
            retention="30 days",
            format="{message}",
            level="INFO",
            serialize=True
        )
        
    def log_detection(self, person_count: int, hand_count: int, 
                     processing_time: float, **kwargs):
        """记录检测日志"""
        logger.info(
            f"Detection: {person_count} persons, {hand_count} hands, "
            f"time: {processing_time:.1f}ms",
            extra={
                "event": "detection",
                "person_count": person_count,
                "hand_count": hand_count,
                "processing_time_ms": processing_time,
                **kwargs
            }
        )
        
    def log_distance(self, distance: float, method: str, 
                    confidence: float, **kwargs):
        """记录距离测量日志"""
        logger.info(
            f"Distance: {distance:.2f}m ({method}), confidence: {confidence:.2f}",
            extra={
                "event": "distance_measurement",
                "distance": distance,
                "method": method,
                "confidence": confidence,
                **kwargs
            }
        )
        
    def log_performance(self, fps: float, memory_mb: float, 
                       cpu_percent: float, **kwargs):
        """记录性能日志"""
        logger.debug(
            f"Performance: FPS={fps:.1f}, Memory={memory_mb:.1f}MB, CPU={cpu_percent:.1f}%",
            extra={
                "event": "performance",
                "fps": fps,
                "memory_mb": memory_mb,
                "cpu_percent": cpu_percent,
                **kwargs
            }
        )


# 全局日志配置
_log_config: Optional[StructuredLog] = None

def setup_logging(log_dir: str = "logs") -> StructuredLog:
    """设置日志"""
    global _log_config
    if _log_config is None:
        _log_config = StructuredLog(log_dir)
    return _log_config
