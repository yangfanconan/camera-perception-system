"""
Depth Anything V2 深度估计模块

基于 Depth Anything V2 的高精度单目深度估计
支持 Apple Silicon MPS 加速
"""

import cv2
import numpy as np
import torch
import threading
import time
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
from loguru import logger
from collections import deque
import hashlib


class DepthAnythingV2Estimator:
    """Depth Anything V2 深度估计器"""
    
    # 模型配置
    MODEL_CONFIGS = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 2048]},
    }
    
    def __init__(
        self,
        encoder: str = 'vits',
        device: str = None,
        max_depth: float = 10.0,
        cache_enabled: bool = True,
        cache_size: int = 3
    ):
        """
        初始化 Depth Anything V2
        
        Args:
            encoder: 编码器类型 ('vits', 'vitb', 'vitl')
            device: 设备 ('mps', 'cuda', 'cpu')
            max_depth: 最大深度值（米）
            cache_enabled: 是否启用缓存
            cache_size: 缓存帧数
        """
        self.encoder = encoder
        self.max_depth = max_depth
        self.cache_enabled = cache_enabled
        
        # 设备选择
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Depth Anything V2 using device: {self.device}")
        
        # 加载模型
        self.model = None
        self._load_model()
        
        # 缓存
        if cache_enabled:
            self.cache = DepthCache(max_size=cache_size)
        else:
            self.cache = None
        
        # 性能统计
        self.stats = {
            'total_inferences': 0,
            'total_time_ms': 0,
            'avg_time_ms': 0
        }
    
    def _load_model(self):
        """加载模型"""
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
            
            config = self.MODEL_CONFIGS[self.encoder]
            self.model = DepthAnythingV2(**config)
            
            # 加载预训练权重
            model_path = self._download_weights()
            if model_path:
                state_dict = torch.load(model_path, map_location='cpu')
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded Depth Anything V2 weights: {model_path}")
            
            self.model = self.model.to(self.device).eval()
            
            # 参数量统计
            num_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Depth Anything V2 ({self.encoder}) parameters: {num_params/1e6:.2f}M")
            
        except Exception as e:
            logger.error(f"Failed to load Depth Anything V2: {e}")
            self.model = None
    
    def _download_weights(self) -> Optional[str]:
        """下载预训练权重"""
        weights_dir = Path('models/depth_anything_v2')
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        weight_urls = {
            'vits': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth',
            'vitb': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth',
            'vitl': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth',
        }
        
        weight_file = weights_dir / f'depth_anything_v2_{self.encoder}.pth'
        
        if weight_file.exists():
            return str(weight_file)
        
        # 下载权重
        import urllib.request
        url = weight_urls.get(self.encoder)
        if url:
            logger.info(f"Downloading Depth Anything V2 weights: {url}")
            try:
                urllib.request.urlretrieve(url, weight_file)
                logger.info(f"Downloaded to: {weight_file}")
                return str(weight_file)
            except Exception as e:
                logger.warning(f"Failed to download weights: {e}")
        
        return None
    
    def estimate(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        估计深度图
        
        Args:
            image: BGR 图像 (H, W, 3)
            
        Returns:
            depth: 深度图 (H, W)，单位为米
        """
        if self.model is None:
            return None
        
        start_time = time.time()
        
        # 检查缓存
        if self.cache_enabled and self.cache:
            frame_hash = self._compute_hash(image)
            cached = self.cache.get(frame_hash)
            if cached is not None:
                return cached
        
        try:
            # 预处理
            h, w = image.shape[:2]
            
            # Depth Anything V2 输入
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 推理
            with torch.no_grad():
                depth = self.model.infer_image(image_rgb)  # (H, W)
            
            # 转换为米（Depth Anything V2 输出是相对深度）
            # 使用线性映射到 0.1m - max_depth
            depth_min = depth.min()
            depth_max = depth.max()
            if depth_max - depth_min > 0:
                depth_normalized = (depth - depth_min) / (depth_max - depth_min)
            else:
                depth_normalized = np.zeros_like(depth)
            
            depth_metric = 0.1 + depth_normalized * (self.max_depth - 0.1)
            
            # 缓存结果
            if self.cache_enabled and self.cache:
                self.cache.put(frame_hash, depth_metric)
            
            # 更新统计
            elapsed_ms = (time.time() - start_time) * 1000
            self.stats['total_inferences'] += 1
            self.stats['total_time_ms'] += elapsed_ms
            self.stats['avg_time_ms'] = self.stats['total_time_ms'] / self.stats['total_inferences']
            
            return depth_metric.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            return None
    
    def estimate_at_point(self, image: np.ndarray, x: int, y: int) -> Optional[float]:
        """
        估计指定点的深度
        
        Args:
            image: BGR 图像
            x: x 坐标
            y: y 坐标
            
        Returns:
            depth: 深度值（米）
        """
        depth_map = self.estimate(image)
        if depth_map is not None and 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            return float(depth_map[y, x])
        return None
    
    def estimate_bbox_depth(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[Dict[str, float]]:
        """
        估计边界框区域的深度
        
        Args:
            image: BGR 图像
            bbox: (x1, y1, x2, y2)
            
        Returns:
            dict: {min, max, mean, bottom} 深度值
        """
        depth_map = self.estimate(image)
        if depth_map is None:
            return None
        
        x1, y1, x2, y2 = bbox
        x1, x2 = max(0, x1), min(depth_map.shape[1], x2)
        y1, y2 = max(0, y1), min(depth_map.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        region = depth_map[y1:y2, x1:x2]
        
        # 底部区域深度（用于距离估计）
        bottom_height = max(1, (y2 - y1) // 4)
        bottom_region = depth_map[y2-bottom_height:y2, x1:x2]
        
        return {
            'min': float(region.min()),
            'max': float(region.max()),
            'mean': float(region.mean()),
            'bottom': float(bottom_region.mean()) if bottom_region.size > 0 else float(region.mean()),
            'std': float(region.std())
        }
    
    def _compute_hash(self, image: np.ndarray) -> str:
        """计算图像哈希（用于缓存）"""
        # 使用图像数据的部分信息计算哈希
        data = image[::16, ::16].tobytes()
        return hashlib.md5(data).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return self.stats.copy()
    
    def clear_cache(self):
        """清空缓存"""
        if self.cache:
            self.cache.clear()


class DepthCache:
    """深度图缓存"""
    
    def __init__(self, max_size: int = 3, ttl: float = 0.5):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def get(self, frame_hash: str) -> Optional[np.ndarray]:
        with self.lock:
            now = time.time()
            for i, (hash_val, depth, timestamp) in enumerate(self.cache):
                if hash_val == frame_hash and (now - timestamp) < self.ttl:
                    return depth
            return None
    
    def put(self, frame_hash: str, depth: np.ndarray):
        with self.lock:
            self.cache.append((frame_hash, depth, time.time()))
    
    def clear(self):
        with self.lock:
            self.cache.clear()


class AsyncDepthEstimator:
    """异步深度估计器"""
    
    def __init__(self, estimator: DepthAnythingV2Estimator):
        self.estimator = estimator
        self.queue = deque(maxlen=2)
        self.result = None
        self.lock = threading.Lock()
        self.thread = None
        self.running = False
    
    def start(self):
        """启动异步处理"""
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        logger.info("Async depth estimator started")
    
    def stop(self):
        """停止异步处理"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        logger.info("Async depth estimator stopped")
    
    def _process_loop(self):
        """处理循环"""
        while self.running:
            if self.queue:
                with self.lock:
                    if self.queue:
                        image = self.queue.popleft()
                    else:
                        continue
                
                depth = self.estimator.estimate(image)
                
                with self.lock:
                    self.result = depth
            else:
                time.sleep(0.005)
    
    def submit(self, image: np.ndarray):
        """提交图像进行处理"""
        with self.lock:
            if len(self.queue) < self.queue.maxlen:
                self.queue.append(image)
    
    def get_result(self) -> Optional[np.ndarray]:
        """获取最新结果"""
        with self.lock:
            return self.result


# 工厂函数
_depth_estimator_instance = None

def get_depth_estimator(
    encoder: str = 'vits',
    device: str = None,
    **kwargs
) -> DepthAnythingV2Estimator:
    """获取深度估计器单例"""
    global _depth_estimator_instance
    
    if _depth_estimator_instance is None:
        _depth_estimator_instance = DepthAnythingV2Estimator(
            encoder=encoder,
            device=device,
            **kwargs
        )
    
    return _depth_estimator_instance


# 测试代码
if __name__ == '__main__':
    import sys
    
    print("Testing Depth Anything V2...")
    
    # 初始化
    estimator = DepthAnythingV2Estimator(encoder='vits')
    
    # 测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 测试推理
    for i in range(5):
        start = time.time()
        depth = estimator.estimate(test_image)
        elapsed = (time.time() - start) * 1000
        if depth is not None:
            print(f"Inference {i+1}: shape={depth.shape}, range=[{depth.min():.2f}, {depth.max():.2f}]m, time={elapsed:.1f}ms")
    
    # 统计
    print(f"\nStats: {estimator.get_stats()}")