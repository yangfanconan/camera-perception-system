"""
深度估计模块 - 统一接口

支持多种后端：
1. Depth Anything V2 (推荐，高精度)
2. MiDaS (备选)
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


class DepthEstimator:
    """统一深度估计接口"""
    
    def __init__(
        self,
        backend: str = 'depth_anything_v2',
        model_size: str = 'small',
        device: str = None,
        max_depth: float = 10.0,
        cache_enabled: bool = True
    ):
        """
        初始化深度估计器
        
        Args:
            backend: 后端类型 ('depth_anything_v2', 'midas')
            model_size: 模型大小 ('small', 'base', 'large')
            device: 设备 ('mps', 'cuda', 'cpu')
            max_depth: 最大深度值（米）
            cache_enabled: 是否启用缓存
        """
        self.backend = backend
        self.model_size = model_size
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
        
        # 加载模型
        self.model = None
        self.transform = None
        self._load_model()
        
        # 缓存
        if cache_enabled:
            self.cache = DepthCache(max_size=3)
        else:
            self.cache = None
        
        # 性能统计
        self.stats = {
            'total_inferences': 0,
            'total_time_ms': 0,
            'avg_time_ms': 0
        }
        
        logger.info(f"Depth estimator initialized: backend={backend}, device={self.device}")

        # 深度校准参数
        self.calibrated = False
        self.calibration_points = []  # [(relative_depth, real_distance), ...]
        self.scale_factor = 1.0  # 缩放因子
        self.offset = 0.0  # 偏移量
    
    def _load_model(self):
        """加载模型"""
        if self.backend == 'depth_anything_v2':
            self._load_depth_anything_v2()
        else:
            self._load_midas()
    
    def _load_depth_anything_v2(self):
        """加载 Depth Anything V2"""
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
            
            encoder_map = {
                'small': 'vits',
                'base': 'vitb',
                'large': 'vitl'
            }
            encoder = encoder_map.get(self.model_size, 'vits')
            
            config = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 2048]},
            }
            
            self.model = DepthAnythingV2(**config[encoder])
            
            # 加载权重
            weight_path = self._download_weights(encoder)
            if weight_path:
                state_dict = torch.load(weight_path, map_location='cpu')
                self.model.load_state_dict(state_dict)
                self.model = self.model.to(self.device).eval()
                self.encoder_type = encoder
                
                num_params = sum(p.numel() for p in self.model.parameters())
                logger.info(f"Depth Anything V2 ({encoder}) loaded with weights: {num_params/1e6:.2f}M params")
            else:
                # 没有权重，回退到 MiDaS
                logger.warning("Depth Anything V2 weights not available, falling back to MiDaS")
                self.backend = 'midas'
                self._load_midas()
                return
            
        except Exception as e:
            logger.error(f"Failed to load Depth Anything V2: {e}")
            logger.info("Falling back to MiDaS")
            self.backend = 'midas'
            self._load_midas()
    
    def _download_weights(self, encoder: str) -> Optional[str]:
        """下载 Depth Anything V2 权重"""
        weights_dir = Path('models/depth_anything_v2')
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        urls = {
            'vits': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth',
            'vitb': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth',
            'vitl': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth',
        }
        
        weight_file = weights_dir / f'depth_anything_v2_{encoder}.pth'
        
        if weight_file.exists():
            return str(weight_file)
        
        import urllib.request
        url = urls.get(encoder)
        if url:
            logger.info(f"Downloading Depth Anything V2 weights: {encoder}")
            try:
                urllib.request.urlretrieve(url, weight_file)
                logger.info(f"Downloaded: {weight_file}")
                return str(weight_file)
            except Exception as e:
                logger.warning(f"Download failed: {e}")
        
        return None
    
    def _load_midas(self):
        """加载 MiDaS"""
        try:
            self.model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
            self.model = self.model.to(self.device).eval()
            
            midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
            self.transform = midas_transforms.small_transform
            
            logger.info("MiDaS small loaded")
            
        except Exception as e:
            logger.error(f"Failed to load MiDaS: {e}")
            logger.warning("Using geometric depth estimation fallback")
            self.model = None
            self.backend = 'geometric'
    
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
            if self.backend == 'depth_anything_v2':
                depth = self._estimate_depth_anything_v2(image)
            elif self.backend == 'midas':
                depth = self._estimate_midas(image)
            else:
                # geometric backend - no depth map
                return None
            
            if depth is None:
                return None
            
            # 缓存
            if self.cache_enabled and self.cache:
                self.cache.put(frame_hash, depth)
            
            # 统计
            elapsed_ms = (time.time() - start_time) * 1000
            self.stats['total_inferences'] += 1
            self.stats['total_time_ms'] += elapsed_ms
            self.stats['avg_time_ms'] = self.stats['total_time_ms'] / self.stats['total_inferences']
            
            return depth
            
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            return None
    
    def _estimate_depth_anything_v2(self, image: np.ndarray) -> Optional[np.ndarray]:
        """使用 Depth Anything V2 估计深度"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        with torch.no_grad():
            depth = self.model.infer_image(image_rgb)
        
        # 归一化到米
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max - depth_min > 0:
            depth_norm = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = np.zeros_like(depth)
        
        depth_metric = 0.1 + depth_norm * (self.max_depth - 0.1)
        return depth_metric.astype(np.float32)
    
    def _estimate_midas(self, image: np.ndarray) -> Optional[np.ndarray]:
        """使用 MiDaS 估计深度"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(image_rgb).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze()
        
        depth = prediction.cpu().numpy()
        
        # 归一化
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max - depth_min > 0:
            depth_norm = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = np.zeros_like(depth)
        
        depth_metric = 0.1 + depth_norm * (self.max_depth - 0.1)
        return depth_metric.astype(np.float32)
    
    def estimate_at_point(self, image: np.ndarray, x: int, y: int) -> Optional[float]:
        """估计指定点的深度"""
        depth_map = self.estimate(image)
        if depth_map is not None and 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            return float(depth_map[y, x])
        return None
    
    def estimate_bbox_depth(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[Dict[str, float]]:
        """估计边界框区域的深度"""
        depth_map = self.estimate(image)
        if depth_map is None:
            return None
        
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        region = depth_map[y1:y2, x1:x2]
        bottom_h = max(1, (y2 - y1) // 4)
        bottom = depth_map[max(0, y2-bottom_h):y2, x1:x2]
        
        return {
            'min': float(region.min()),
            'max': float(region.max()),
            'mean': float(region.mean()),
            'bottom': float(bottom.mean()) if bottom.size > 0 else float(region.mean()),
            'std': float(region.std())
        }
    
    def _compute_hash(self, image: np.ndarray) -> str:
        """计算图像哈希"""
        data = image[::16, ::16].tobytes()
        return hashlib.md5(data).hexdigest()
    

    # ==================== 深度校准 ====================
    def add_calibration_point(self, relative_depth: float, real_distance: float) -> Dict[str, Any]:
        """
        添加校准点
        
        Args:
            relative_depth: 相对深度值（从深度图获取）
            real_distance: 真实距离（米）
        
        Returns:
            校准状态
        """
        self.calibration_points.append((relative_depth, real_distance))
        
        # 如果有足够的点，计算校准参数
        if len(self.calibration_points) >= 2:
            self._compute_calibration()
        
        return {
            "points_count": len(self.calibration_points),
            "calibrated": self.calibrated,
            "scale_factor": self.scale_factor,
            "offset": self.offset
        }

    def _compute_calibration(self):
        """计算校准参数（线性回归）"""
        if len(self.calibration_points) < 2:
            return
        
        # 简单线性回归: real = scale * relative + offset
        points = np.array(self.calibration_points)
        relative = points[:, 0]
        real = points[:, 1]
        
        # 最小二乘法
        n = len(relative)
        sum_x = np.sum(relative)
        sum_y = np.sum(real)
        sum_xy = np.sum(relative * real)
        sum_xx = np.sum(relative * relative)
        
        denom = n * sum_xx - sum_x * sum_x
        if abs(denom) > 1e-6:
            self.scale_factor = (n * sum_xy - sum_x * sum_y) / denom
            self.offset = (sum_y - self.scale_factor * sum_x) / n
        else:
            self.scale_factor = 1.0
            self.offset = 0.0
        
        self.calibrated = True
        logger.info(f"Depth calibration computed: scale={self.scale_factor:.4f}, offset={self.offset:.4f}")

    def apply_calibration(self, depth_map: np.ndarray) -> np.ndarray:
        """
        应用校准，将相对深度转换为真实距离
        
        Args:
            depth_map: 相对深度图
        
        Returns:
            真实距离图（米）
        """
        if not self.calibrated:
            return depth_map
        
        return depth_map * self.scale_factor + self.offset

    def clear_calibration(self):
        """清除校准"""
        self.calibrated = False
        self.calibration_points = []
        self.scale_factor = 1.0
        self.offset = 0.0
        logger.info("Depth calibration cleared")

    def get_calibration_info(self) -> Dict[str, Any]:
        """获取校准信息"""
        return {
            "calibrated": self.calibrated,
            "points_count": len(self.calibration_points),
            "points": self.calibration_points.copy(),
            "scale_factor": self.scale_factor,
            "offset": self.offset
        }

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
            for hash_val, depth, timestamp in self.cache:
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
    
    def __init__(self, estimator: DepthEstimator):
        self.estimator = estimator
        self.queue = deque(maxlen=2)
        self.result = None
        self.lock = threading.Lock()
        self.thread = None
        self.running = False
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def _process_loop(self):
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
        with self.lock:
            if len(self.queue) < self.queue.maxlen:
                self.queue.append(image)
    
    def get_result(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.result


# 单例
_estimator_instance = None

def get_depth_estimator(**kwargs) -> DepthEstimator:
    """获取深度估计器单例"""
    global _estimator_instance
    if _estimator_instance is None:
        _estimator_instance = DepthEstimator(**kwargs)
    return _estimator_instance