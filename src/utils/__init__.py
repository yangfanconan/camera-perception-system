"""
工具函数模块
"""

from .config_manager import ConfigManager, get_config_manager, get_config, setup_logging
from .logger import (
    setup_logger,
    log_execution,
    handle_exceptions,
    retry,
    measure_performance,
    perf_monitor,
    AppException,
    ConfigException,
    CameraException,
    DetectionException,
    CalibrationException,
    SpatialException,
    register_exception_handlers,
    create_logging_middleware
)
from .apple_silicon import (
    AppleSiliconOptimizer,
    get_optimizer,
    get_device,
    optimize_model,
    apply_apple_silicon_optimizations
)

__all__ = [
    'ConfigManager',
    'get_config_manager',
    'get_config',
    'setup_logging',
    'setup_logger',
    'log_execution',
    'handle_exceptions',
    'retry',
    'measure_performance',
    'perf_monitor',
    'AppException',
    'ConfigException',
    'CameraException',
    'DetectionException',
    'CalibrationException',
    'SpatialException',
    'register_exception_handlers',
    'create_logging_middleware',
    'AppleSiliconOptimizer',
    'get_optimizer',
    'get_device',
    'optimize_model',
    'apply_apple_silicon_optimizations',
]
