"""
插件系统模块

功能：
1. 插件加载和管理
2. 插件生命周期
3. 插件通信
4. 插件配置
"""

import importlib
import inspect
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger
import time
import json
from abc import ABC, abstractmethod


@dataclass
class PluginInfo:
    """插件信息"""
    name: str
    version: str
    description: str
    author: str = ""
    enabled: bool = True
    priority: int = 0  # 优先级，数字越大越先执行


class PluginBase(ABC):
    """
    插件基类
    
    所有插件必须继承此类
    """
    
    def __init__(self):
        """初始化插件"""
        self.info = self.get_info()
        self.config: Dict = {}
        self.enabled = True
        self.initialized = False
    
    @abstractmethod
    def get_info(self) -> PluginInfo:
        """获取插件信息"""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict = None) -> bool:
        """
        初始化插件
        
        Args:
            config: 插件配置
            
        Returns:
            是否成功
        """
        pass
    
    @abstractmethod
    def process(self, data: Dict) -> Dict:
        """
        处理数据
        
        Args:
            data: 输入数据
            
        Returns:
            处理结果
        """
        pass
    
    def shutdown(self):
        """关闭插件"""
        self.initialized = False
        logger.info(f"Plugin {self.info.name} shutdown")
    
    def configure(self, config: Dict):
        """配置插件"""
        self.config.update(config)
    
    def enable(self):
        """启用插件"""
        self.enabled = True
    
    def disable(self):
        """禁用插件"""
        self.enabled = False


class PluginManager:
    """
    插件管理器
    
    管理插件的生命周期
    """
    
    def __init__(self, plugin_dir: str = "plugins"):
        """
        初始化插件管理器
        
        Args:
            plugin_dir: 插件目录
        """
        self.plugin_dir = Path(plugin_dir)
        self.plugins: Dict[str, PluginBase] = {}
        self.plugin_order: List[str] = []
        
        # 事件钩子
        self.hooks: Dict[str, List[Callable]] = {
            'before_process': [],
            'after_process': [],
            'on_detection': [],
            'on_alert': []
        }
        
        logger.info(f"PluginManager initialized (plugin_dir={plugin_dir})")
    
    def discover_plugins(self) -> List[str]:
        """
        发现可用插件
        
        Returns:
            插件名称列表
        """
        discovered = []
        
        if not self.plugin_dir.exists():
            logger.warning(f"Plugin directory not found: {self.plugin_dir}")
            return discovered
        
        for path in self.plugin_dir.iterdir():
            if path.is_dir() and not path.name.startswith('_'):
                # 检查是否有 plugin.py
                plugin_file = path / "plugin.py"
                if plugin_file.exists():
                    discovered.append(path.name)
            
            elif path.suffix == '.py' and not path.name.startswith('_'):
                discovered.append(path.stem)
        
        logger.info(f"Discovered {len(discovered)} plugins: {discovered}")
        return discovered
    
    def load_plugin(self, plugin_name: str) -> bool:
        """
        加载插件
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            是否成功
        """
        if plugin_name in self.plugins:
            logger.warning(f"Plugin already loaded: {plugin_name}")
            return True
        
        try:
            # 尝试导入插件模块
            module_path = f"plugins.{plugin_name}.plugin"
            
            try:
                module = importlib.import_module(module_path)
            except ImportError:
                # 尝试直接导入
                module = importlib.import_module(f"plugins.{plugin_name}")
            
            # 查找插件类
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, PluginBase) and obj != PluginBase:
                    plugin_class = obj
                    break
            
            if plugin_class is None:
                logger.error(f"No plugin class found in {plugin_name}")
                return False
            
            # 实例化插件
            plugin = plugin_class()
            
            # 注册插件
            self.plugins[plugin_name] = plugin
            self._update_plugin_order()
            
            logger.info(f"Plugin loaded: {plugin_name} v{plugin.info.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}")
            return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """
        卸载插件
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            是否成功
        """
        if plugin_name not in self.plugins:
            return False
        
        try:
            plugin = self.plugins[plugin_name]
            plugin.shutdown()
            
            del self.plugins[plugin_name]
            self._update_plugin_order()
            
            logger.info(f"Plugin unloaded: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload plugin {plugin_name}: {e}")
            return False
    
    def initialize_plugin(self, plugin_name: str, config: Dict = None) -> bool:
        """
        初始化插件
        
        Args:
            plugin_name: 插件名称
            config: 插件配置
            
        Returns:
            是否成功
        """
        if plugin_name not in self.plugins:
            return False
        
        plugin = self.plugins[plugin_name]
        
        try:
            success = plugin.initialize(config)
            plugin.initialized = success
            
            if success:
                logger.info(f"Plugin initialized: {plugin_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to initialize plugin {plugin_name}: {e}")
            return False
    
    def process(self, data: Dict) -> Dict:
        """
        通过所有插件处理数据
        
        Args:
            data: 输入数据
            
        Returns:
            处理结果
        """
        # 触发 before_process 钩子
        self._trigger_hooks('before_process', data)
        
        result = data.copy()
        
        # 按优先级顺序处理
        for plugin_name in self.plugin_order:
            plugin = self.plugins.get(plugin_name)
            
            if plugin and plugin.enabled and plugin.initialized:
                try:
                    result = plugin.process(result)
                except Exception as e:
                    logger.error(f"Plugin {plugin_name} process error: {e}")
        
        # 触发 after_process 钩子
        self._trigger_hooks('after_process', result)
        
        return result
    
    def _update_plugin_order(self):
        """更新插件执行顺序"""
        # 按优先级排序
        self.plugin_order = sorted(
            self.plugins.keys(),
            key=lambda x: self.plugins[x].info.priority,
            reverse=True
        )
    
    def register_hook(self, hook_name: str, callback: Callable):
        """
        注册事件钩子
        
        Args:
            hook_name: 钩子名称
            callback: 回调函数
        """
        if hook_name in self.hooks:
            self.hooks[hook_name].append(callback)
            logger.debug(f"Registered hook: {hook_name}")
    
    def _trigger_hooks(self, hook_name: str, data: Dict):
        """触发钩子"""
        if hook_name not in self.hooks:
            return
        
        for callback in self.hooks[hook_name]:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Hook callback error: {e}")
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """获取插件信息"""
        if plugin_name in self.plugins:
            return self.plugins[plugin_name].info
        return None
    
    def get_all_plugins(self) -> Dict[str, PluginInfo]:
        """获取所有插件信息"""
        return {
            name: plugin.info 
            for name, plugin in self.plugins.items()
        }
    
    def enable_plugin(self, plugin_name: str):
        """启用插件"""
        if plugin_name in self.plugins:
            self.plugins[plugin_name].enable()
    
    def disable_plugin(self, plugin_name: str):
        """禁用插件"""
        if plugin_name in self.plugins:
            self.plugins[plugin_name].disable()
    
    def load_all(self) -> Dict[str, bool]:
        """
        加载所有发现的插件
        
        Returns:
            加载结果
        """
        results = {}
        
        for plugin_name in self.discover_plugins():
            results[plugin_name] = self.load_plugin(plugin_name)
        
        return results
    
    def initialize_all(self, configs: Dict[str, Dict] = None) -> Dict[str, bool]:
        """
        初始化所有插件
        
        Args:
            configs: 插件配置字典
            
        Returns:
            初始化结果
        """
        results = {}
        configs = configs or {}
        
        for plugin_name in self.plugins:
            config = configs.get(plugin_name, {})
            results[plugin_name] = self.initialize_plugin(plugin_name, config)
        
        return results
    
    def shutdown_all(self):
        """关闭所有插件"""
        for plugin_name in list(self.plugins.keys()):
            self.unload_plugin(plugin_name)


# 示例插件
class ExamplePlugin(PluginBase):
    """示例插件"""
    
    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="example",
            version="1.0.0",
            description="Example plugin for demonstration",
            author="System",
            priority=0
        )
    
    def initialize(self, config: Dict = None) -> bool:
        logger.info("Example plugin initialized")
        return True
    
    def process(self, data: Dict) -> Dict:
        # 添加处理标记
        data['example_processed'] = True
        return data


# 全局插件管理器
_plugin_manager = None

def get_plugin_manager() -> PluginManager:
    """获取插件管理器单例"""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


# 测试代码
if __name__ == '__main__':
    print("Testing Plugin System...")
    
    manager = PluginManager()
    
    # 手动注册示例插件
    manager.plugins['example'] = ExamplePlugin()
    manager._update_plugin_order()
    
    # 初始化
    manager.initialize_plugin('example')
    
    # 处理数据
    data = {'test': 'data'}
    result = manager.process(data)
    
    print(f"Result: {result}")
    
    # 获取插件信息
    print(f"Plugins: {manager.get_all_plugins()}")
    
    print("\nDone!")