"""
智能缓存系统
功能：
1. LRU缓存
2. TTL缓存（带过期时间）
3. 内存限制
4. 自动清理
"""

import time
import hashlib
import pickle
from typing import Any, Optional, Dict, Tuple
from collections import OrderedDict
from dataclasses import dataclass
from loguru import logger
import threading


@dataclass
class CacheEntry:
    """缓存条目"""
    value: Any
    timestamp: float
    ttl: Optional[float] = None  # 过期时间（秒）
    access_count: int = 0


class SmartCache:
    """智能缓存"""

    def __init__(self, max_size: int = 100, default_ttl: Optional[float] = None):
        """
        初始化缓存

        Args:
            max_size: 最大缓存条目数
            default_ttl: 默认过期时间（秒）
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

        logger.info(f"SmartCache initialized: max_size={max_size}")

    def _make_key(self, *args, **kwargs) -> str:
        """生成缓存键"""
        key_data = pickle.dumps((args, sorted(kwargs.items())))
        return hashlib.md5(key_data).hexdigest()

    def get(self, *args, **kwargs) -> Optional[Any]:
        """获取缓存值"""
        key = self._make_key(*args, **kwargs)

        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            # 检查是否过期
            if entry.ttl is not None:
                if time.time() - entry.timestamp > entry.ttl:
                    del self._cache[key]
                    self._misses += 1
                    return None

            # 更新访问统计
            entry.access_count += 1
            self._cache.move_to_end(key)
            self._hits += 1

            return entry.value

    def set(self, value: Any, *args, ttl: Optional[float] = None, **kwargs):
        """设置缓存值"""
        key = self._make_key(*args, **kwargs)

        with self._lock:
            # 清理过期条目
            self._cleanup_expired()

            # 如果缓存已满，移除最久未访问的
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)

            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=ttl or self.default_ttl
            )
            self._cache[key] = entry

    def _cleanup_expired(self):
        """清理过期条目"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.ttl is not None and current_time - entry.timestamp > entry.ttl
        ]
        for key in expired_keys:
            del self._cache[key]

    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def get_stats(self) -> Dict:
        """获取缓存统计"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0

            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'memory_estimate_kb': len(self._cache) * 0.5  # 粗略估计
            }

    def __contains__(self, key) -> bool:
        """检查键是否存在"""
        return self.get(key) is not None


class DetectionCache:
    """检测结果缓存"""

    def __init__(self, max_frames: int = 5):
        """
        初始化检测缓存

        Args:
            max_frames: 最大缓存帧数
        """
        self.max_frames = max_frames
        self._frame_cache: OrderedDict[int, Any] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, frame_hash: int) -> Optional[Any]:
        """获取缓存的检测结果"""
        with self._lock:
            if frame_hash in self._frame_cache:
                self._frame_cache.move_to_end(frame_hash)
                return self._frame_cache[frame_hash]
            return None

    def set(self, frame_hash: int, result: Any):
        """缓存检测结果"""
        with self._lock:
            if len(self._frame_cache) >= self.max_frames:
                self._frame_cache.popitem(last=False)
            self._frame_cache[frame_hash] = result

    def clear(self):
        """清空缓存"""
        with self._lock:
            self._frame_cache.clear()


# 全局缓存实例
_distance_cache = SmartCache(max_size=50, default_ttl=0.1)  # 100ms TTL


def get_distance_cache() -> SmartCache:
    """获取距离计算缓存"""
    return _distance_cache
