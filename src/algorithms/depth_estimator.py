"""
深度估计模块

基于 MiDaS 的单目深度估计（带缓存和异步支持）
"""

import cv2
import numpy as np
import torch
import threading
import time
from typing import Tuple, Optional
from pathlib import Path
from loguru import logger
from collections import deque


class DepthCache:
    """深度图缓存"""
    
    def __init__(self, max_size: int = 5, ttl: float = 0.5):
        """
        初始化缓存
        
        Args:
            max_size: 最大缓存帧数
            ttl: 缓存有效期（秒）
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = deque(maxlen=max_size)
        self.lock = threading.Lock()
        
    def get(self, frame_hash: str) -> Optional[np.ndarray]:
        """获取缓存的深度图"""
        with self.lock:
            now = time.time()
            for i, (hash_val, depth, timestamp) in enumerate(self.cache):
                if hash_val == frame_hash and (now - timestamp) < self.ttl:
                    return depth
            return None
    
    def put(self, frame_hash: str, depth: np.ndarray):
        """添加深度图到缓存"""
        with self.lock:
            self.cache.append((frame_hash, depth, time.time()))
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()


class AsyncDepthEstimator:
    """异步深度估计器"""
    
    def __init__(self, estimator: 'DepthEstimator'):
        self.estimator = estimator
        self.queue = deque(maxlen=3)  # 最多3帧待处理
        self.result = None
        self.lock = threading.Lock()
        self.thread = None
        self.running = False
        
    def start(self):
        """启动异步处理线程"""
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """停止异步处理"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def _process_loop(self):
        """处理循环"""
        while self.running:
            if self.queue:
                with self.lock:
                    if self.queue:
                        image, callback = self.queue.popleft()
                    else:
                        continue
                
                # 执行深度估计
                depth = self.estimator.estimate(image)
                
                # 保存结果
                with self.lock:
                    self.result = depth
                
                # 回调
                if callback:
                    callback(depth)
            else:
                time.sleep(0.01)  # 10ms 轮询
    
    def submit(self, image: np.ndarray, callback=None):
        """提交图像进行异步处理"""
        with self.lock:
            self.queue.append((image, callback))
    
    def get_result(self) -> Optional[np.ndarray]:
        """获取最新结果"""
        with self.lock:
            return self.result


class DepthEstimator:
    """
    单目深度估计器
    
    使用 MiDaS 模型进行深度估计
    """
    
    # 模型类型映射
    MODEL_TYPES = {
        'small': 'MiDaS_small',      # 轻量级，速度快
        'hybrid': 'DPT_Hybrid',       # 平衡精度速度
        'large': 'DPT_Large',         # 高精度，慢
    }
    
    def __init__(self, model_type: str = 'small', device: Optional[str] = None,
                 use_cache: bool = True, cache_size: int = 5):
        """
        初始化深度估计器
        
        Args:
            model_type: 模型类型 ('small', 'hybrid', 'large')
            device: 计算设备 ('cuda', 'mps', 'cpu')
            use_cache: 是否使用缓存
            cache_size: 缓存大小
        """
        self.model_type = model_type
        self.device = self._get_device(device)
        self.model = None
        self.transform = None
        
        # 缓存
        self.use_cache = use_cache
        self.cache = DepthCache(max_size=cache_size) if use_cache else None
        
        # 性能统计
        self.stats = {
            'total_calls': 0,
            'cache_hits': 0,
            'inference_time_ms': deque(maxlen=100)
        }
        
        logger.info(f"Initializing DepthEstimator with model: {model_type}, device: {self.device}, cache: {use_cache}")
        
    def _get_device(self, device: Optional[str]) -> str:
        """自动选择设备"""
        if device:
            return device
        
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def load_model(self) -> bool:
        """
        加载 MiDaS 模型
        
        Returns:
            是否成功
        """
        try:
            # 使用 torch.hub 加载 MiDaS
            model_name = self.MODEL_TYPES.get(self.model_type, 'MiDaS_small')
            
            logger.info(f"Loading MiDaS model: {model_name}")
            
            # 加载模型和变换
            self.model = torch.hub.load('intel-isl/MiDaS', model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # 加载对应的变换
            midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            
            if self.model_type == 'large':
                self.transform = midas_transforms.dpt_transform
            elif self.model_type == 'hybrid':
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform
            
            logger.info("MiDaS model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load MiDaS model: {e}")
            return False
    
    def _compute_frame_hash(self, image: np.ndarray) -> str:
        """计算图像哈希用于缓存"""
        # 使用图像尺寸和部分像素的哈希
        h, w = image.shape[:2]
        # 采样中心区域的像素
        sample = image[h//4:3*h//4, w//4:3*w//4:10]
        return f"{h}x{w}_{hash(sample.tobytes()) % 1000000}"
    
    def estimate(self, image: np.ndarray, use_cache: bool = True) -> Optional[np.ndarray]:
        """
        估计深度图（带缓存）
        
        Args:
            image: BGR 图像 (H, W, 3)
            use_cache: 是否使用缓存
            
        Returns:
            深度图 (H, W)，单位：米，None 表示失败
        """
        import time
        start_time = time.time()
        
        self.stats['total_calls'] += 1
        
        # 检查缓存
        if use_cache and self.use_cache and self.cache:
            frame_hash = self._compute_frame_hash(image)
            cached_depth = self.cache.get(frame_hash)
            if cached_depth is not None:
                self.stats['cache_hits'] += 1
                return cached_depth
        
        if self.model is None:
            if not self.load_model():
                return None
        
        try:
            # 确保图像格式正确
            if image is None or image.size == 0:
                logger.error("Invalid input image: None or empty")
                return None
            
            # 检查图像维度
            if len(image.shape) == 2:
                # 灰度图，转换为 3 通道
                logger.debug("Converting grayscale to 3-channel")
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) != 3:
                logger.error(f"Invalid image shape: {image.shape}, expected 2 or 3 dimensions")
                return None
            
            # 检查通道数
            if image.shape[2] not in [1, 3, 4]:
                logger.error(f"Invalid number of channels: {image.shape[2]}, expected 1, 3, or 4")
                return None
            
            # 确保图像是 uint8 类型
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
            
            # 转换为 RGB（如果是 BGR）
            if image.shape[2] == 3:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.shape[2] == 4:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif image.shape[2] == 1:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = image
            
            # 应用变换
            input_batch = self.transform(img_rgb).to(self.device)
            
            # 推理
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image.shape[:2],
                    mode='bicubic',
                    align_corners=False,
                ).squeeze()
            
            # 转换为 numpy
            depth = prediction.cpu().numpy()
            
            # MiDaS 输出是相对深度，需要转换
            # 使用逆深度表示：depth = 1.0 / (depth + epsilon)
            depth = self._convert_to_metric(depth)
            
            # 记录推理时间
            inference_time = (time.time() - start_time) * 1000
            self.stats['inference_time_ms'].append(inference_time)
            
            # 存入缓存
            if use_cache and self.use_cache and self.cache:
                self.cache.put(frame_hash, depth)
            
            return depth

        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            return None
    
    def _convert_to_metric(self, relative_depth: np.ndarray) -> np.ndarray:
        """
        将相对深度转换为米制深度

        MiDaS 输出是相对逆深度（值越大表示越近）
        需要转换为实际距离（值越大表示越远）

        Args:
            relative_depth: 相对深度图

        Returns:
            绝对深度图（米）
        """
        # 避免除零
        epsilon = 1e-6

        # 归一化到 0-1
        depth_min = relative_depth.min()
        depth_max = relative_depth.max()
        normalized = (relative_depth - depth_min) / (depth_max - depth_min + epsilon)

        # MiDaS: 值越大 = 越近
        # 实际距离: 值越大 = 越远
        # 转换: 近距离 (0.3m) 到远距离 (10m)
        # 修正：normalized=1 (近) -> 0.3m, normalized=0 (远) -> 10m
        metric_depth = 10.0 - 9.7 * normalized  # 10m ~ 0.3m

        return metric_depth
    
    def estimate_distance(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """
        估计特定区域的距离
        
        Args:
            image: 完整图像
            bbox: 边界框 (x, y, w, h)
            
        Returns:
            (距离, 置信度)
        """
        depth_map = self.estimate(image)
        if depth_map is None:
            return 0.0, 0.0
        
        x, y, w, h = bbox
        
        # 提取 ROI
        roi = depth_map[y:y+h, x:x+w]
        if roi.size == 0:
            return 0.0, 0.0
        
        # 使用中位数（更鲁棒）
        distance = np.median(roi)
        
        # 置信度基于深度方差
        variance = np.var(roi)
        confidence = max(0.0, 1.0 - variance / 10.0)
        
        return float(distance), float(confidence)
    
    def get_depth_at_point(self, image: np.ndarray, point: Tuple[int, int]) -> float:
        """
        获取特定点的深度

        Args:
            image: 图像
            point: (x, y) 坐标

        Returns:
            深度值（米）
        """
        depth_map = self.estimate(image)
        if depth_map is None:
            return 0.0

        x, y = point
        if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
            return float(depth_map[y, x])
        return 0.0
    
    def get_stats(self) -> dict:
        """获取性能统计"""
        avg_time = 0.0
        if self.stats['inference_time_ms']:
            avg_time = sum(self.stats['inference_time_ms']) / len(self.stats['inference_time_ms'])
        
        cache_hit_rate = 0.0
        if self.stats['total_calls'] > 0:
            cache_hit_rate = self.stats['cache_hits'] / self.stats['total_calls']
        
        return {
            'total_calls': self.stats['total_calls'],
            'cache_hits': self.stats['cache_hits'],
            'cache_hit_rate': round(cache_hit_rate, 3),
            'avg_inference_time_ms': round(avg_time, 2),
            'model_type': self.model_type,
            'device': self.device
        }


# 全局深度估计器实例
_depth_estimator: Optional[DepthEstimator] = None


def get_depth_estimator(model_type: str = 'small') -> DepthEstimator:
    """获取全局深度估计器实例"""
    global _depth_estimator
    if _depth_estimator is None:
        _depth_estimator = DepthEstimator(model_type)
    return _depth_estimator


def estimate_depth(image: np.ndarray) -> Optional[np.ndarray]:
    """
    便捷函数：估计深度图
    
    Args:
        image: BGR 图像
        
    Returns:
        深度图
    """
    estimator = get_depth_estimator()
    return estimator.estimate(image)
