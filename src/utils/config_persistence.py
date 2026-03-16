"""
配置持久化模块

自动保存和加载用户配置
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = "configs/user"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_file = self.config_dir / "settings.json"
        self.default_config = {
            "camera": {
                "camera_id": 0,
                "resolution": [1920, 1080],
                "fps": 20
            },
            "detection": {
                "conf_threshold": 0.5,
                "max_detections": 10
            },
            "depth": {
                "enabled": True,
                "model_type": "small",
                "use_cache": True
            },
            "spatial": {
                "ref_head_width": 0.15,
                "ref_shoulder_width": 0.45,
                "kalman_enabled": True
            },
            "ui": {
                "show_depth_map": False,
                "show_performance": True,
                "theme": "dark"
            }
        }
        self.config = self.load()
        
    def load(self) -> Dict[str, Any]:
        """加载配置"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    saved_config = json.load(f)
                    # 合并默认配置和保存的配置
                    config = self.default_config.copy()
                    self._deep_update(config, saved_config)
                    logger.info(f"Config loaded from {self.config_file}")
                    return config
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
        
        return self.default_config.copy()
    
    def save(self) -> bool:
        """保存配置"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            logger.info(f"Config saved to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项（支持点号路径）"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> bool:
        """设置配置项（支持点号路径）"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        return self.save()
    
    def _deep_update(self, base: Dict, update: Dict):
        """深度更新字典"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value


# 全局配置管理器
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """获取全局配置管理器"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
