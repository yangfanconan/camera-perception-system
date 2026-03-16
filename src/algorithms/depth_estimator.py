"""
深度估计模块

基于 MiDaS 的单目深度估计
"""

import cv2
import numpy as np
import torch
from typing import Tuple, Optional
from pathlib import Path
from loguru import logger


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
    
    def __init__(self, model_type: str = 'small', device: Optional[str] = None):
        """
        初始化深度估计器
        
        Args:
            model_type: 模型类型 ('small', 'hybrid', 'large')
            device: 计算设备 ('cuda', 'mps', 'cpu')
        """
        self.model_type = model_type
        self.device = self._get_device(device)
        self.model = None
        self.transform = None
        
        logger.info(f"Initializing DepthEstimator with model: {model_type}, device: {self.device}")
        
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
    
    def estimate(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        估计深度图
        
        Args:
            image: BGR 图像 (H, W, 3)
            
        Returns:
            深度图 (H, W)，单位：米，None 表示失败
        """
        if self.model is None:
            if not self.load_model():
                return None
        
        try:
            # 转换为 RGB
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
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
            
            return depth
            
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            return None
    
    def _convert_to_metric(self, relative_depth: np.ndarray) -> np.ndarray:
        """
        将相对深度转换为米制深度
        
        MiDaS 输出是相对深度，需要通过标定转换为绝对深度
        这里使用简化的线性映射
        
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
        
        # 转换到实际距离范围 (0.1m - 10m)
        # 使用逆深度表示：越近值越大
        metric_depth = 0.1 + 9.9 * (1.0 - normalized)
        
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
