"""
配置管理模块

功能：
1. 配置加载和保存
2. 配置验证
3. 默认配置
4. 配置热更新
"""

import json
import yaml
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
from loguru import logger
import copy
import time


@dataclass
class CameraConfig:
    """相机配置"""
    camera_id: int = 0
    resolution: List[int] = field(default_factory=lambda: [1920, 1080])
    fps: int = 30
    auto_start: bool = False


@dataclass
class DetectionConfig:
    """检测配置"""
    person_confidence: float = 0.5
    hand_confidence: float = 0.5
    face_confidence: float = 0.5
    vehicle_confidence: float = 0.5
    max_detections: int = 100
    nms_threshold: float = 0.45


@dataclass
class TrackingConfig:
    """跟踪配置"""
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    max_tracks: int = 50


@dataclass
class AlertConfig:
    """报警配置"""
    enabled: bool = True
    fall_detection: bool = True
    zone_monitoring: bool = True
    crowd_detection: bool = True
    cooldown: float = 10.0
    crowd_threshold: int = 5


@dataclass
class SpeechConfig:
    """语音配置"""
    enabled: bool = False
    language: str = "zh-CN"
    rate: float = 1.0
    announce_falls: bool = True
    announce_alerts: bool = True


@dataclass
class UIConfig:
    """UI 配置"""
    theme: str = "dark"
    show_trajectory: bool = True
    show_velocity: bool = True
    show_depth_heatmap: bool = False
    trajectory_length: int = 30


@dataclass
class SystemConfig:
    """系统配置"""
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    # 其他设置
    log_level: str = "INFO"
    data_dir: str = "data"
    model_dir: str = "models"


class ConfigManager:
    """
    配置管理器
    
    管理系统配置
    """
    
    DEFAULT_CONFIG_FILE = "config/config.yaml"
    
    def __init__(self, config_file: str = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = Path(config_file or self.DEFAULT_CONFIG_FILE)
        self.config = SystemConfig()
        
        # 配置变更回调
        self.callbacks: List[callable] = []
        
        # 加载配置
        self.load()
        
        logger.info(f"ConfigManager initialized (config_file={self.config_file})")
    
    def load(self) -> bool:
        """
        加载配置
        
        Returns:
            是否成功
        """
        if not self.config_file.exists():
            logger.info(f"Config file not found, using defaults: {self.config_file}")
            self.save()  # 保存默认配置
            return True
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                if self.config_file.suffix in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            # 更新配置
            self._update_config(data)
            
            logger.info(f"Config loaded from {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return False
    
    def save(self) -> bool:
        """
        保存配置
        
        Returns:
            是否成功
        """
        try:
            # 创建目录
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换为字典
            data = self._config_to_dict()
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                if self.config_file.suffix in ['.yaml', '.yml']:
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
                else:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Config saved to {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False
    
    def _update_config(self, data: Dict):
        """更新配置"""
        if 'camera' in data:
            self.config.camera = CameraConfig(**data['camera'])
        if 'detection' in data:
            self.config.detection = DetectionConfig(**data['detection'])
        if 'tracking' in data:
            self.config.tracking = TrackingConfig(**data['tracking'])
        if 'alert' in data:
            self.config.alert = AlertConfig(**data['alert'])
        if 'speech' in data:
            self.config.speech = SpeechConfig(**data['speech'])
        if 'ui' in data:
            self.config.ui = UIConfig(**data['ui'])
        
        # 其他设置
        for key in ['log_level', 'data_dir', 'model_dir']:
            if key in data:
                setattr(self.config, key, data[key])
    
    def _config_to_dict(self) -> Dict:
        """配置转字典"""
        return {
            'camera': asdict(self.config.camera),
            'detection': asdict(self.config.detection),
            'tracking': asdict(self.config.tracking),
            'alert': asdict(self.config.alert),
            'speech': asdict(self.config.speech),
            'ui': asdict(self.config.ui),
            'log_level': self.config.log_level,
            'data_dir': self.config.data_dir,
            'model_dir': self.config.model_dir
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键 (支持点分隔，如 "camera.fps")
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        obj = self.config
        
        for k in keys:
            if hasattr(obj, k):
                obj = getattr(obj, k)
            else:
                return default
        
        return obj
    
    def set(self, key: str, value: Any, save: bool = True):
        """
        设置配置值
        
        Args:
            key: 配置键
            value: 配置值
            save: 是否保存
        """
        keys = key.split('.')
        obj = self.config
        
        # 遍历到最后一个键
        for k in keys[:-1]:
            if hasattr(obj, k):
                obj = getattr(obj, k)
            else:
                return
        
        # 设置值
        if hasattr(obj, keys[-1]):
            setattr(obj, keys[-1], value)
            
            # 触发回调
            self._notify_callbacks(key, value)
            
            # 保存
            if save:
                self.save()
    
    def _notify_callbacks(self, key: str, value: Any):
        """通知配置变更"""
        for callback in self.callbacks:
            try:
                callback(key, value)
            except Exception as e:
                logger.error(f"Config callback error: {e}")
    
    def on_change(self, callback: callable):
        """
        注册配置变更回调
        
        Args:
            callback: 回调函数 (key, value) -> None
        """
        self.callbacks.append(callback)
    
    def reset(self, save: bool = True):
        """
        重置为默认配置
        
        Args:
            save: 是否保存
        """
        self.config = SystemConfig()
        
        if save:
            self.save()
        
        logger.info("Config reset to defaults")
    
    def validate(self) -> List[str]:
        """
        验证配置
        
        Returns:
            错误列表
        """
        errors = []
        
        # 验证相机配置
        if self.config.camera.fps < 1 or self.config.camera.fps > 120:
            errors.append("camera.fps must be between 1 and 120")
        
        if self.config.camera.resolution[0] < 100 or self.config.camera.resolution[1] < 100:
            errors.append("camera.resolution too small")
        
        # 验证检测配置
        if not 0 <= self.config.detection.person_confidence <= 1:
            errors.append("detection.person_confidence must be between 0 and 1")
        
        # 验证跟踪配置
        if self.config.tracking.max_age < 1:
            errors.append("tracking.max_age must be positive")
        
        return errors
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return self._config_to_dict()
    
    def from_dict(self, data: Dict, save: bool = True):
        """
        从字典加载
        
        Args:
            data: 配置字典
            save: 是否保存
        """
        self._update_config(data)
        
        if save:
            self.save()


class ConfigPresets:
    """
    配置预设
    
    提供常用配置预设
    """
    
    @staticmethod
    def high_performance() -> Dict:
        """高性能配置"""
        return {
            'camera': {'fps': 60, 'resolution': [1280, 720]},
            'detection': {'person_confidence': 0.6, 'max_detections': 50},
            'tracking': {'max_age': 20, 'max_tracks': 30},
            'ui': {'show_trajectory': False, 'show_velocity': False}
        }
    
    @staticmethod
    def high_accuracy() -> Dict:
        """高精度配置"""
        return {
            'camera': {'fps': 30, 'resolution': [1920, 1080]},
            'detection': {'person_confidence': 0.7, 'nms_threshold': 0.3},
            'tracking': {'max_age': 50, 'min_hits': 5}
        }
    
    @staticmethod
    def security() -> Dict:
        """安全监控配置"""
        return {
            'alert': {
                'enabled': True,
                'fall_detection': True,
                'zone_monitoring': True,
                'crowd_detection': True
            },
            'speech': {
                'enabled': True,
                'announce_falls': True,
                'announce_alerts': True
            }
        }
    
    @staticmethod
    def demo() -> Dict:
        """演示配置"""
        return {
            'camera': {'fps': 30, 'resolution': [1280, 720]},
            'detection': {'person_confidence': 0.5},
            'ui': {
                'show_trajectory': True,
                'show_velocity': True,
                'theme': 'dark'
            },
            'speech': {'enabled': True}
        }


# 全局配置管理器
_config_manager = None

def get_config() -> ConfigManager:
    """获取配置管理器单例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


# 测试代码
if __name__ == '__main__':
    print("Testing Config Manager...")
    
    config = ConfigManager("test_config.yaml")
    
    # 获取配置
    print(f"Camera FPS: {config.get('camera.fps')}")
    print(f"Detection confidence: {config.get('detection.person_confidence')}")
    
    # 设置配置
    config.set('camera.fps', 60)
    config.set('ui.theme', 'light')
    
    # 验证
    errors = config.validate()
    print(f"Validation errors: {errors}")
    
    # 预设
    print("\nHigh performance preset:")
    print(json.dumps(ConfigPresets.high_performance(), indent=2))
    
    print("\nDone!")