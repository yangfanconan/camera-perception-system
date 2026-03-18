"""
深度增强的距离估计模块

融合 MiDaS 深度估计和几何估计
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from loguru import logger

from .spatial.distance_estimation import DistanceEstimator
from .depth_estimator import DepthEstimator, get_depth_estimator


@dataclass
class DistanceEstimate:
    """距离估计结果"""
    distance: float
    confidence: float
    method: str
    details: Dict


class DepthEnhancedDistanceEstimator:
    """
    深度增强的距离估计器
    
    融合多种估计方法：
    1. MiDaS 深度估计（主要）
    2. 几何估计（头部/身体关键点）
    3. 边界框占比估计（备选）
    """
    
    def __init__(self, fx: float = 650.0, cx: float = 320.0,
                 use_depth: bool = True, depth_weight: float = 0.3):
        """
        初始化
        
        Args:
            fx: 焦距
            cx: 主点
            use_depth: 是否使用深度估计
            depth_weight: 深度估计的权重
        """
        self.fx = fx
        self.cx = cx
        self.use_depth = use_depth
        self.depth_weight = depth_weight
        
        # 几何估计器
        self.geometric_estimator = DistanceEstimator(fx, cx)
        
        # 深度估计器（延迟加载）
        self.depth_estimator: Optional[DepthEstimator] = None
        
        logger.info(f"DepthEnhancedDistanceEstimator initialized: "
                   f"use_depth={use_depth}, depth_weight={depth_weight}")
        
    def _ensure_depth_estimator(self):
        """确保深度估计器已加载"""
        if self.use_depth and self.depth_estimator is None:
            self.depth_estimator = get_depth_estimator('small')
            if not self.depth_estimator.load_model():
                logger.warning("Failed to load depth estimator, falling back to geometric only")
                self.use_depth = False
    
    def estimate(self, image: np.ndarray, bbox: Tuple[int, int, int, int],
                 keypoints: Dict, body_part: str = 'unknown') -> DistanceEstimate:
        """
        融合估计距离
        
        Args:
            image: 完整图像
            bbox: 边界框 (x, y, w, h)
            keypoints: 关键点字典
            body_part: 身体部位类型
            
        Returns:
            DistanceEstimate
        """
        estimates = []
        
        # 1. 深度估计（带合理性检查）
        if self.use_depth:
            self._ensure_depth_estimator()
            if self.depth_estimator:
                depth_dist, depth_conf = self._estimate_from_depth(image, bbox)
                # 合理性检查：深度应该在 0.2m - 5m 范围内
                if 0.2 <= depth_dist <= 5.0:
                    estimates.append({
                        'distance': depth_dist,
                        'confidence': depth_conf,
                        'method': 'depth',
                        'weight': self.depth_weight
                    })
                else:
                    logger.warning(f"Depth estimate {depth_dist:.2f}m out of reasonable range, skipping")
        
        # 2. 几何估计（头部）
        head_dist, head_conf = self.geometric_estimator.estimate_from_head(keypoints)
        if head_dist > 0 and head_conf > 0.3:
            estimates.append({
                'distance': head_dist,
                'confidence': head_conf,
                'method': 'head',
                'weight': 0.25
            })
        
        # 3. 几何估计（身体关键点）
        body_dist, body_conf = self.geometric_estimator.estimate_from_body_keypoints(keypoints)
        if body_dist > 0 and body_conf > 0.3:
            estimates.append({
                'distance': body_dist,
                'confidence': body_conf,
                'method': 'body_keypoints',
                'weight': 0.15
            })
        
        # 4. 边界框估计（备选）
        if len(estimates) < 2:
            h, w = image.shape[:2]
            bbox_dist = self.geometric_estimator.estimate_from_bbox(bbox, w, h)
            estimates.append({
                'distance': bbox_dist,
                'confidence': 0.3,
                'method': 'bbox',
                'weight': 0.1
            })
        
        # 融合估计
        return self._fuse_estimates(estimates, body_part)
    
    def _estimate_from_depth(self, image: np.ndarray, 
                            bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """
        基于深度估计距离
        
        Args:
            image: 图像
            bbox: 边界框
            
        Returns:
            (距离, 置信度)
        """
        if self.depth_estimator is None:
            return 0.0, 0.0
        
        try:
            return self.depth_estimator.estimate_distance(image, bbox)
        except Exception as e:
            logger.error(f"Depth estimation error: {e}")
            return 0.0, 0.0
    
    def _fuse_estimates(self, estimates: list, body_part: str) -> DistanceEstimate:
        """
        融合多种估计
        
        使用加权平均，考虑置信度和一致性
        """
        if not estimates:
            return DistanceEstimate(0.0, 0.0, 'unknown', {})
        
        if len(estimates) == 1:
            e = estimates[0]
            return DistanceEstimate(
                distance=e['distance'],
                confidence=e['confidence'],
                method=e['method'],
                details={'single_estimate': True}
            )
        
        # 计算加权平均
        total_weight = 0.0
        weighted_sum = 0.0
        
        for e in estimates:
            # 权重 = 配置权重 * 置信度
            w = e['weight'] * e['confidence']
            weighted_sum += e['distance'] * w
            total_weight += w
        
        if total_weight == 0:
            return DistanceEstimate(0.0, 0.0, 'unknown', {})
        
        fused_distance = weighted_sum / total_weight
        
        # 计算融合置信度
        avg_confidence = sum(e['confidence'] for e in estimates) / len(estimates)
        
        # 一致性检查
        distances = [e['distance'] for e in estimates]
        consistency = self._check_consistency(distances)
        
        # 最终置信度
        final_confidence = avg_confidence * 0.7 + consistency * 0.3
        
        # 构建方法描述
        methods = '+'.join(sorted(set(e['method'] for e in estimates)))
        
        return DistanceEstimate(
            distance=fused_distance,
            confidence=final_confidence,
            method=f'fused_{methods}',
            details={
                'estimates': estimates,
                'consistency': consistency,
                'body_part': body_part
            }
        )
    
    def _check_consistency(self, distances: list) -> float:
        """
        检查估计一致性
        
        Returns:
            一致性分数 (0-1)
        """
        if len(distances) < 2:
            return 1.0
        
        mean = np.mean(distances)
        std = np.std(distances)
        
        # 变异系数
        if mean == 0:
            return 0.0
        
        cv = std / mean
        
        # 转换为一致性分数
        consistency = max(0.0, 1.0 - cv)
        
        return float(consistency)
    
    def estimate_with_uncertainty(self, image: np.ndarray, bbox: Tuple[int, int, int, int],
                                  keypoints: Dict) -> Tuple[float, float, str]:
        """
        带不确定性的估计
        
        Returns:
            (距离, 不确定性, 方法)
        """
        result = self.estimate(image, bbox, keypoints)
        
        # 不确定性 = 1 - 置信度
        uncertainty = 1.0 - result.confidence
        
        return result.distance, uncertainty, result.method


# 便捷函数
def estimate_distance_enhanced(image: np.ndarray, bbox: Tuple[int, int, int, int],
                               keypoints: Dict, fx: float = 650.0) -> Tuple[float, float, str]:
    """
    深度增强的距离估计便捷函数
    
    Args:
        image: 图像
        bbox: 边界框
        keypoints: 关键点
        fx: 焦距
        
    Returns:
        (距离, 置信度, 方法)
    """
    estimator = DepthEnhancedDistanceEstimator(fx=fx)
    result = estimator.estimate(image, bbox, keypoints)
    return result.distance, result.confidence, result.method
