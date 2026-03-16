"""
系统配置管理模块
功能：
1. YAML 配置文件加载
2. 环境变量覆盖
3. 配置验证
4. 热重载支持
"""

import os
import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional, List, Union
from dataclasses import dataclass, field, asdict
from loguru import logger
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


@dataclass
class CameraConfig:
    """相机配置"""
    id: int = 0
    resolution: List[int] = field(default_factory=lambda: [1920, 1080])
    fps: int = 20
    auto_focus: bool = True
    exposure: int = -1  # -1 = 自动


@dataclass
class CalibrationConfig:
    """标定配置"""
    checkerboard: List[int] = field(default_factory=lambda: [9, 6])
    square_size: float = 25.0
    min_images: int = 15
    file_path: str = "calibration_data/calib_params.json"


@dataclass
class DetectionConfig:
    """检测配置"""
    pose_model: str = "models/yolov8n-pose.pt"
    hand_model: str = "mediapipe"  # mediapipe 或 yolo
    conf_threshold: float = 0.5
    iou_threshold: float = 0.45
    smooth_enabled: bool = True
    max_persons: int = 10
    max_hands: int = 4


@dataclass
class SpatialConfig:
    """空间计量配置"""
    ref_shoulder_width: float = 0.45
    ref_hand_length: float = 0.18
    camera_height: float = 1.8
    pitch_angle: float = 30.0
    topview_scale: float = 10.0
    enable_correction: bool = True
    enable_fusion: bool = True


@dataclass
class StreamConfig:
    """流配置"""
    type: str = "mjpeg"  # mjpeg, flv, webrtc
    quality: int = 85
    bitrate: int = 2000000
    gop_size: int = 20


@dataclass
class ServerConfig:
    """服务器配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1


@dataclass
class LogConfig:
    """日志配置"""
    level: str = "INFO"
    file: str = "logs/app.log"
    rotation: str = "10 MB"
    retention: str = "7 days"
    format: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"


@dataclass
class SystemConfig:
    """系统配置"""
    camera: CameraConfig = field(default_factory=CameraConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    log: LogConfig = field(default_factory=LogConfig)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)
    
    def to_yaml(self) -> str:
        """转换为 YAML 字符串"""
        return yaml.dump(self.to_dict(), default_flow_style=False, allow_unicode=True)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SystemConfig':
        """从字典创建"""
        config = cls()
        
        if 'camera' in data:
            for k, v in data['camera'].items():
                if hasattr(config.camera, k):
                    setattr(config.camera, k, v)
        
        if 'calibration' in data:
            for k, v in data['calibration'].items():
                if hasattr(config.calibration, k):
                    setattr(config.calibration, k, v)
        
        if 'detection' in data:
            for k, v in data['detection'].items():
                if hasattr(config.detection, k):
                    setattr(config.detection, k, v)
        
        if 'spatial' in data:
            for k, v in data['spatial'].items():
                if hasattr(config.spatial, k):
                    setattr(config.spatial, k, v)
        
        if 'stream' in data:
            for k, v in data['stream'].items():
                if hasattr(config.stream, k):
                    setattr(config.stream, k, v)
        
        if 'server' in data:
            for k, v in data['server'].items():
                if hasattr(config.server, k):
                    setattr(config.server, k, v)
        
        if 'log' in data:
            for k, v in data['log'].items():
                if hasattr(config.log, k):
                    setattr(config.log, k, v)
        
        return config


class ConfigChangeHandler(FileSystemEventHandler):
    """配置文件变更处理器"""
    
    def __init__(self, config_manager: 'ConfigManager', callback: callable = None):
        self.config_manager = config_manager
        self.callback = callback
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        if str(event.src_path).endswith('.yaml') or str(event.src_path).endswith('.yml'):
            logger.info(f"Config file changed: {event.src_path}")
            
            try:
                self.config_manager.reload()
                if self.callback:
                    self.callback(self.config_manager.config)
            except Exception as e:
                logger.error(f"Failed to reload config: {e}")


class ConfigManager:
    """配置管理器"""
    
    DEFAULT_CONFIG_PATH = "configs/camera.yaml"
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.config = SystemConfig()
        self._lock = threading.Lock()
        self._observer: Optional[Observer] = None
        
        # 加载配置
        self.load()
        
        logger.info(f"ConfigManager initialized with {self.config_path}")
    
    def load(self) -> SystemConfig:
        """加载配置文件"""
        with self._lock:
            config_path = Path(self.config_path)
            
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                    
                    if data:
                        self.config = SystemConfig.from_dict(data)
                        logger.info(f"Config loaded from {self.config_path}")
                except Exception as e:
                    logger.warning(f"Failed to load config: {e}, using defaults")
            else:
                logger.warning(f"Config file not found: {self.config_path}, using defaults")
                # 创建默认配置
                self.save()
            
            # 应用环境变量覆盖
            self._apply_env_overrides()
            
            return self.config
    
    def save(self, path: Optional[str] = None) -> None:
        """保存配置文件"""
        with self._lock:
            save_path = Path(path) if path else Path(self.config_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config.to_dict(), f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"Config saved to {save_path}")
    
    def reload(self) -> SystemConfig:
        """重新加载配置"""
        return self.load()
    
    def _apply_env_overrides(self) -> None:
        """应用环境变量覆盖"""
        # 相机配置
        if env_val := os.getenv('CAMERA_ID'):
            self.config.camera.id = int(env_val)
        if env_val := os.getenv('RESOLUTION'):
            self.config.camera.resolution = [int(x) for x in env_val.split('x')]
        if env_val := os.getenv('FPS'):
            self.config.camera.fps = int(env_val)
        
        # 检测配置
        if env_val := os.getenv('CONF_THRESHOLD'):
            self.config.detection.conf_threshold = float(env_val)
        if env_val := os.getenv('SMOOTH_ENABLED'):
            self.config.detection.smooth_enabled = env_val.lower() == 'true'
        
        # 空间计量配置
        if env_val := os.getenv('CAMERA_HEIGHT'):
            self.config.spatial.camera_height = float(env_val)
        if env_val := os.getenv('PITCH_ANGLE'):
            self.config.spatial.pitch_angle = float(env_val)
        
        # 日志配置
        if env_val := os.getenv('LOG_LEVEL'):
            self.config.log.level = env_val
        if env_val := os.getenv('LOG_FILE'):
            self.config.log.file = env_val
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值（支持点分隔路径）"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = getattr(value, k)
            return value
        except (AttributeError, KeyError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """设置配置值"""
        keys = key.split('.')
        obj = self.config
        
        try:
            for k in keys[:-1]:
                obj = getattr(obj, k)
            setattr(obj, keys[-1], value)
        except (AttributeError, KeyError) as e:
            logger.error(f"Failed to set config {key}: {e}")
    
    def start_watching(self, callback: Optional[callable] = None) -> None:
        """开始监听配置文件变化"""
        if self._observer:
            return
        
        self._observer = Observer()
        handler = ConfigChangeHandler(self, callback)
        
        config_dir = str(Path(self.config_path).parent)
        self._observer.schedule(handler, config_dir, recursive=False)
        self._observer.start()
        
        logger.info(f"Started watching config directory: {config_dir}")
    
    def stop_watching(self) -> None:
        """停止监听"""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            logger.info("Stopped watching config directory")
    
    def validate(self) -> List[str]:
        """验证配置"""
        errors = []
        
        # 相机配置验证
        if self.config.camera.fps < 1 or self.config.camera.fps > 60:
            errors.append("FPS must be between 1 and 60")
        
        if len(self.config.camera.resolution) != 2:
            errors.append("Resolution must have 2 values")
        
        # 检测配置验证
        if not 0 <= self.config.detection.conf_threshold <= 1:
            errors.append("Confidence threshold must be between 0 and 1")
        
        # 空间计量配置验证
        if self.config.spatial.camera_height <= 0:
            errors.append("Camera height must be positive")
        
        if not 0 <= self.config.spatial.pitch_angle <= 90:
            errors.append("Pitch angle must be between 0 and 90")
        
        if errors:
            for error in errors:
                logger.error(f"Config validation error: {error}")
        
        return errors
    
    def __getattr__(self, name: str) -> Any:
        """允许直接访问配置子项"""
        if hasattr(self.config, name):
            return getattr(self.config, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


# ==================== 全局配置实例 ====================

_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """获取配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def get_config(key: str, default: Any = None) -> Any:
    """获取配置值"""
    manager = get_config_manager()
    return manager.get(key, default)


def setup_logging() -> None:
    """设置日志"""
    manager = get_config_manager()
    config = manager.config.log
    
    # 移除默认处理器
    logger.remove()
    
    # 添加控制台处理器
    logger.add(
        lambda msg: print(msg, end=''),
        format=config.format,
        level=config.level,
        colorize=True
    )
    
    # 添加文件处理器
    log_file = Path(config.file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        str(log_file),
        format=config.format,
        level=config.level,
        rotation=config.rotation,
        retention=config.retention
    )
    
    logger.info(f"Logging configured: level={config.level}, file={config.file}")


# ==================== 主函数（测试） ====================

def main():
    """测试配置管理"""
    # 创建配置管理器
    manager = ConfigManager()
    
    # 打印配置
    print("Current Configuration:")
    print(f"  Camera ID: {manager.config.camera.id}")
    print(f"  Resolution: {manager.config.camera.resolution}")
    print(f"  FPS: {manager.config.camera.fps}")
    print(f"  Confidence Threshold: {manager.config.detection.conf_threshold}")
    print(f"  Camera Height: {manager.config.spatial.camera_height}m")
    
    # 验证配置
    errors = manager.validate()
    if errors:
        print("\nValidation Errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n✓ Configuration is valid")
    
    # 测试获取配置
    print(f"\nUsing get(): camera.fps = {manager.get('camera.fps')}")
    
    # 测试设置配置
    manager.set('camera.fps', 30)
    print(f"After set(): camera.fps = {manager.get('camera.fps')}")
    
    # 保存配置
    manager.save('configs/test_config.yaml')
    print("\nConfig saved to configs/test_config.yaml")


if __name__ == '__main__':
    main()
