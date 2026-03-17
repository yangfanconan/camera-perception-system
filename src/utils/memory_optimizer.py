"""
内存优化模块

用于监控和优化内存使用
"""

import gc
import psutil
import torch
import numpy as np
from typing import Optional, Dict, Any
from collections import deque
from loguru import logger
import threading
import time


class MemoryOptimizer:
    """内存优化器"""
    
    def __init__(self, max_memory_gb: float = 4.0, 
                 cleanup_threshold: float = 0.8,
                 monitoring_interval: float = 5.0):
        """
        初始化内存优化器
        
        Args:
            max_memory_gb: 最大内存限制（GB）
            cleanup_threshold: 清理阈值（内存使用比例）
            monitoring_interval: 监控间隔（秒）
        """
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.cleanup_threshold = cleanup_threshold
        self.monitoring_interval = monitoring_interval
        
        self.process = psutil.Process()
        self.memory_history = deque(maxlen=100)
        self.running = False
        self.monitor_thread = None
        
        # 统计
        self.stats = {
            'cleanup_count': 0,
            'peak_memory_mb': 0,
            'avg_memory_mb': 0
        }
        
        logger.info(f"MemoryOptimizer initialized: max={max_memory_gb}GB, threshold={cleanup_threshold}")
    
    def start_monitoring(self):
        """启动内存监控"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """停止内存监控"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            memory_info = self.get_memory_info()
            self.memory_history.append(memory_info)
            
            # 更新峰值
            if memory_info['rss_mb'] > self.stats['peak_memory_mb']:
                self.stats['peak_memory_mb'] = memory_info['rss_mb']
            
            # 检查是否需要清理
            if memory_info['percent'] > self.cleanup_threshold * 100:
                logger.warning(f"Memory usage high: {memory_info['percent']:.1f}%, triggering cleanup")
                self.cleanup()
            
            time.sleep(self.monitoring_interval)
    
    def get_memory_info(self) -> Dict[str, float]:
        """获取内存信息"""
        memory = self.process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            'rss_mb': memory.rss / 1024 / 1024,
            'vms_mb': memory.vms / 1024 / 1024,
            'percent': self.process.memory_percent(),
            'system_percent': system_memory.percent,
            'available_mb': system_memory.available / 1024 / 1024
        }
    
    def cleanup(self, aggressive: bool = False):
        """
        清理内存
        
        Args:
            aggressive: 是否进行激进清理
        """
        before = self.get_memory_info()
        
        # Python 垃圾回收
        gc.collect()
        
        # PyTorch 缓存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # 激进清理
        if aggressive:
            # 强制释放未使用的内存
            gc.collect(2)  # 完整垃圾回收
            
            # 释放 numpy 内存池
            np.free = lambda: None  # 占位
        
        after = self.get_memory_info()
        freed_mb = before['rss_mb'] - after['rss_mb']
        
        self.stats['cleanup_count'] += 1
        
        logger.info(f"Memory cleanup: freed {freed_mb:.1f} MB, current: {after['rss_mb']:.1f} MB")
        
        return freed_mb
    
    def optimize_tensors(self):
        """优化张量内存"""
        # 检查并释放不需要的张量
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    if obj.device.type == 'cuda' and not obj.is_cuda:
                        # 移动到 CPU
                        obj.cpu()
            except:
                pass
    
    def get_stats(self) -> Dict[str, Any]:
        """获取内存统计"""
        if len(self.memory_history) == 0:
            return self.stats
        
        recent = list(self.memory_history)[-10:]
        avg_memory = np.mean([m['rss_mb'] for m in recent])
        
        self.stats['avg_memory_mb'] = round(avg_memory, 2)
        
        return {
            **self.stats,
            'current': self.get_memory_info(),
            'history': list(self.memory_history)
        }


class FrameBuffer:
    """帧缓冲区（限制内存使用）"""
    
    def __init__(self, max_frames: int = 30, max_memory_mb: float = 500):
        """
        初始化帧缓冲区
        
        Args:
            max_frames: 最大帧数
            max_memory_mb: 最大内存（MB）
        """
        self.max_frames = max_frames
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.buffer = deque(maxlen=max_frames)
        self.current_memory = 0
    
    def add(self, frame: np.ndarray) -> bool:
        """
        添加帧到缓冲区
        
        Args:
            frame: 帧数据
            
        Returns:
            是否成功添加
        """
        frame_size = frame.nbytes
        
        # 检查内存限制
        if self.current_memory + frame_size > self.max_memory_bytes:
            # 移除旧帧
            while self.buffer and self.current_memory + frame_size > self.max_memory_bytes:
                old_frame = self.buffer.popleft()
                self.current_memory -= old_frame.nbytes
        
        self.buffer.append(frame)
        self.current_memory += frame_size
        
        return True
    
    def get(self, index: int = -1) -> Optional[np.ndarray]:
        """获取帧"""
        if not self.buffer:
            return None
        
        if index < 0:
            return self.buffer[index]
        
        if index < len(self.buffer):
            return self.buffer[index]
        
        return None
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
        self.current_memory = 0
        gc.collect()
    
    def get_memory_usage(self) -> float:
        """获取内存使用（MB）"""
        return self.current_memory / 1024 / 1024


class LazyLoader:
    """延迟加载器"""
    
    def __init__(self, loader_func: callable, unload_after: float = 300):
        """
        初始化延迟加载器
        
        Args:
            loader_func: 加载函数
            unload_after: 卸载时间（秒）
        """
        self.loader_func = loader_func
        self.unload_after = unload_after
        self._obj = None
        self._last_access = 0
        self._lock = threading.Lock()
    
    @property
    def obj(self):
        """获取对象（延迟加载）"""
        with self._lock:
            if self._obj is None:
                logger.info("Lazy loading object...")
                self._obj = self.loader_func()
            
            self._last_access = time.time()
            return self._obj
    
    def unload_if_idle(self):
        """如果空闲则卸载"""
        with self._lock:
            if self._obj is not None:
                idle_time = time.time() - self._last_access
                if idle_time > self.unload_after:
                    logger.info(f"Unloading idle object (idle {idle_time:.0f}s)")
                    self._obj = None
                    gc.collect()


# 全局内存优化器
_memory_optimizer: Optional[MemoryOptimizer] = None


def get_memory_optimizer() -> MemoryOptimizer:
    """获取全局内存优化器"""
    global _memory_optimizer
    if _memory_optimizer is None:
        _memory_optimizer = MemoryOptimizer()
    return _memory_optimizer


def cleanup_memory(aggressive: bool = False):
    """
    便捷函数：清理内存
    
    Args:
        aggressive: 是否激进清理
    """
    optimizer = get_memory_optimizer()
    return optimizer.cleanup(aggressive)


def get_memory_stats() -> Dict[str, Any]:
    """获取内存统计"""
    optimizer = get_memory_optimizer()
    return optimizer.get_stats()
