"""
深度校准模块

通过已知距离的参考物体校准深度估计，
将相对深度转换为绝对深度。

校准方法：
1. 单点校准：使用一个已知距离的点
2. 多点校准：使用多个已知距离的点进行线性/多项式拟合
3. 平面校准：使用已知尺寸的平面物体
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
from loguru import logger


@dataclass
class CalibrationPoint:
    """校准点"""
    pixel_depth: float      # 深度估计器输出的相对深度
    real_depth: float       # 实际距离（米）
    bbox: Tuple[int, int, int, int]  # 边界框
    timestamp: float        # 时间戳
    confidence: float = 1.0 # 置信度


@dataclass
class DepthCalibrationParams:
    """深度校准参数"""
    scale: float = 1.0          # 缩放系数
    offset: float = 0.0         # 偏移量
    poly_coeffs: List[float] = field(default_factory=lambda: [0.0, 1.0])  # 多项式系数 [offset, scale]
    calibration_points: List[Dict] = field(default_factory=list)
    calibrated: bool = False
    method: str = "linear"  # linear, polynomial, piecewise


class DepthCalibrator:
    """
    深度校准器
    
    将相对深度转换为绝对深度
    """
    
    def __init__(self, params_file: str = "calibration_data/depth_calib.json"):
        """
        初始化校准器
        
        Args:
            params_file: 校准参数文件路径
        """
        self.params_file = Path(params_file)
        self.params = DepthCalibrationParams()
        self.calibration_points: List[CalibrationPoint] = []
        
        # 加载已有校准参数
        self._load_params()
    
    def _load_params(self):
        """加载校准参数"""
        if self.params_file.exists():
            try:
                with open(self.params_file, 'r') as f:
                    data = json.load(f)
                
                self.params.scale = data.get('scale', 1.0)
                self.params.offset = data.get('offset', 0.0)
                self.params.poly_coeffs = data.get('poly_coeffs', [0.0, 1.0])
                self.params.calibrated = data.get('calibrated', False)
                self.params.method = data.get('method', 'linear')
                self.params.calibration_points = data.get('calibration_points', [])
                
                logger.info(f"Loaded depth calibration: scale={self.params.scale}, offset={self.params.offset}")
            except Exception as e:
                logger.warning(f"Failed to load depth calibration: {e}")
    
    def save_params(self):
        """保存校准参数"""
        self.params_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'scale': self.params.scale,
            'offset': self.params.offset,
            'poly_coeffs': self.params.poly_coeffs,
            'calibrated': self.params.calibrated,
            'method': self.params.method,
            'calibration_points': self.params.calibration_points
        }
        
        with open(self.params_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved depth calibration to {self.params_file}")
    
    def add_calibration_point(
        self,
        pixel_depth: float,
        real_depth: float,
        bbox: Tuple[int, int, int, int] = None,
        confidence: float = 1.0
    ):
        """
        添加校准点
        
        Args:
            pixel_depth: 深度估计器输出的相对深度
            real_depth: 实际距离（米）
            bbox: 边界框（可选）
            confidence: 置信度
        """
        import time
        
        point = CalibrationPoint(
            pixel_depth=pixel_depth,
            real_depth=real_depth,
            bbox=bbox or (0, 0, 0, 0),
            timestamp=time.time(),
            confidence=confidence
        )
        
        self.calibration_points.append(point)
        
        # 同时保存到参数中
        self.params.calibration_points.append({
            'pixel_depth': pixel_depth,
            'real_depth': real_depth,
            'confidence': confidence
        })
        
        logger.info(f"Added calibration point: pixel={pixel_depth:.3f}, real={real_depth:.2f}m")
    
    def calibrate(self, method: str = "linear") -> bool:
        """
        执行校准
        
        Args:
            method: 校准方法 ('linear', 'polynomial', 'robust')
            
        Returns:
            是否校准成功
        """
        if len(self.calibration_points) < 1:
            logger.warning("Need at least 1 calibration point")
            return False
        
        self.params.method = method
        
        if method == "linear":
            return self._calibrate_linear()
        elif method == "polynomial":
            return self._calibrate_polynomial()
        elif method == "robust":
            return self._calibrate_robust()
        else:
            logger.warning(f"Unknown calibration method: {method}")
            return False
    
    def _calibrate_linear(self) -> bool:
        """线性校准"""
        if len(self.calibration_points) < 1:
            return False
        
        # 单点校准
        if len(self.calibration_points) == 1:
            point = self.calibration_points[0]
            if point.pixel_depth > 0:
                self.params.scale = point.real_depth / point.pixel_depth
                self.params.offset = 0.0
                self.params.poly_coeffs = [0.0, self.params.scale]
                self.params.calibrated = True
                logger.info(f"Single point calibration: scale={self.params.scale:.4f}")
                return True
            return False
        
        # 多点线性回归
        pixel_depths = np.array([p.pixel_depth for p in self.calibration_points])
        real_depths = np.array([p.real_depth for p in self.calibration_points])
        
        # y = scale * x + offset
        # 使用最小二乘法
        A = np.vstack([pixel_depths, np.ones(len(pixel_depths))]).T
        result = np.linalg.lstsq(A, real_depths, rcond=None)
        
        self.params.scale = result[0][0]
        self.params.offset = result[0][1]
        self.params.poly_coeffs = [self.params.offset, self.params.scale]
        self.params.calibrated = True
        
        # 计算误差
        predicted = self.params.scale * pixel_depths + self.params.offset
        errors = np.abs(predicted - real_depths)
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(errors ** 2))
        
        logger.info(f"Linear calibration: scale={self.params.scale:.4f}, offset={self.params.offset:.4f}")
        logger.info(f"Calibration error: MAE={mae:.3f}m, RMSE={rmse:.3f}m")
        
        return True
    
    def _calibrate_polynomial(self, degree: int = 2) -> bool:
        """多项式校准"""
        if len(self.calibration_points) < degree + 1:
            logger.warning(f"Need at least {degree + 1} points for degree {degree} polynomial")
            return self._calibrate_linear()
        
        pixel_depths = np.array([p.pixel_depth for p in self.calibration_points])
        real_depths = np.array([p.real_depth for p in self.calibration_points])
        
        # 多项式拟合
        coeffs = np.polyfit(pixel_depths, real_depths, degree)
        self.params.poly_coeffs = coeffs.tolist()
        self.params.calibrated = True
        
        # 计算误差
        predicted = np.polyval(coeffs, pixel_depths)
        errors = np.abs(predicted - real_depths)
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(errors ** 2))
        
        logger.info(f"Polynomial calibration (degree={degree}): coeffs={coeffs}")
        logger.info(f"Calibration error: MAE={mae:.3f}m, RMSE={rmse:.3f}m")
        
        return True
    
    def _calibrate_robust(self) -> bool:
        """鲁棒校准（去除异常点）"""
        if len(self.calibration_points) < 3:
            return self._calibrate_linear()
        
        pixel_depths = np.array([p.pixel_depth for p in self.calibration_points])
        real_depths = np.array([p.real_depth for p in self.calibration_points])
        
        # 使用 RANSAC 去除异常点
        from sklearn.linear_model import RANSACRegressor, LinearRegression
        
        try:
            ransac = RANSACRegressor(
                estimator=LinearRegression(),
                min_samples=2,
                max_trials=100,
                residual_threshold=0.5
            )
            ransac.fit(pixel_depths.reshape(-1, 1), real_depths)
            
            self.params.scale = ransac.estimator_.coef_[0]
            self.params.offset = ransac.estimator_.intercept_
            self.params.poly_coeffs = [self.params.offset, self.params.scale]
            self.params.calibrated = True
            
            inlier_mask = ransac.inlier_mask_
            n_inliers = np.sum(inlier_mask)
            
            logger.info(f"Robust calibration: scale={self.params.scale:.4f}, offset={self.params.offset:.4f}")
            logger.info(f"Inliers: {n_inliers}/{len(self.calibration_points)}")
            
            return True
            
        except ImportError:
            logger.warning("sklearn not available, falling back to linear calibration")
            return self._calibrate_linear()
    
    def apply_calibration(self, depth_map: np.ndarray) -> np.ndarray:
        """
        应用校准，将相对深度转换为绝对深度
        
        Args:
            depth_map: 相对深度图
            
        Returns:
            绝对深度图（米）
        """
        if not self.params.calibrated:
            logger.warning("Depth not calibrated, returning raw values")
            return depth_map
        
        if self.params.method == "linear":
            return self.params.scale * depth_map + self.params.offset
        elif self.params.method == "polynomial":
            return np.polyval(self.params.poly_coeffs, depth_map)
        else:
            return self.params.scale * depth_map + self.params.offset
    
    def get_params(self) -> Dict[str, Any]:
        """获取校准参数"""
        return {
            'scale': self.params.scale,
            'offset': self.params.offset,
            'poly_coeffs': self.params.poly_coeffs,
            'calibrated': self.params.calibrated,
            'method': self.params.method,
            'num_points': len(self.calibration_points)
        }
    
    def reset(self):
        """重置校准"""
        self.params = DepthCalibrationParams()
        self.calibration_points = []
        logger.info("Depth calibration reset")


class DepthCalibrationUI:
    """
    深度校准 UI 辅助类
    
    提供交互式校准功能
    """
    
    def __init__(self, calibrator: DepthCalibrator):
        self.calibrator = calibrator
        self.current_mode = "idle"  # idle, collecting, calibrating
        self.collect_count = 0
        self.target_count = 5
    
    def start_collection(self, target_count: int = 5):
        """开始收集校准点"""
        self.current_mode = "collecting"
        self.collect_count = 0
        self.target_count = target_count
        logger.info(f"Started calibration point collection, target: {target_count}")
    
    def add_point_from_detection(
        self,
        pixel_depth: float,
        real_depth: float = None,
        auto_estimate: bool = True
    ):
        """
        从检测结果添加校准点
        
        Args:
            pixel_depth: 相对深度
            real_depth: 实际距离（如果为None，需要用户输入）
            auto_estimate: 是否自动估计（基于边界框）
        """
        if self.current_mode != "collecting":
            return
        
        if real_depth is None:
            # 需要用户输入实际距离
            logger.info(f"Collected point {self.collect_count + 1}: pixel_depth={pixel_depth:.3f}")
            logger.info("Please provide real distance for this point")
            return
        
        self.calibrator.add_calibration_point(pixel_depth, real_depth)
        self.collect_count += 1
        
        if self.collect_count >= self.target_count:
            logger.info(f"Collected {self.collect_count} points, ready for calibration")
            self.current_mode = "ready"
    
    def finish_collection(self) -> bool:
        """完成收集并执行校准"""
        if self.collect_count < 1:
            logger.warning("No calibration points collected")
            return False
        
        success = self.calibrator.calibrate()
        if success:
            self.calibrator.save_params()
            self.current_mode = "calibrated"
        
        return success


# 全局校准器实例
_depth_calibrator = None

def get_depth_calibrator() -> DepthCalibrator:
    """获取深度校准器单例"""
    global _depth_calibrator
    if _depth_calibrator is None:
        _depth_calibrator = DepthCalibrator()
    return _depth_calibrator


# 测试代码
if __name__ == '__main__':
    print("Testing Depth Calibrator...")
    
    calibrator = DepthCalibrator()
    
    # 添加校准点
    calibrator.add_calibration_point(0.5, 1.0)   # 相对深度 0.5 -> 实际 1.0m
    calibrator.add_calibration_point(1.0, 2.0)   # 相对深度 1.0 -> 实际 2.0m
    calibrator.add_calibration_point(2.0, 4.0)   # 相对深度 2.0 -> 实际 4.0m
    calibrator.add_calibration_point(3.0, 6.0)   # 相对深度 3.0 -> 实际 6.0m
    
    # 执行校准
    calibrator.calibrate("linear")
    
    # 测试转换
    test_depths = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    calibrated = calibrator.apply_calibration(test_depths)
    
    print(f"Input: {test_depths}")
    print(f"Calibrated: {calibrated}")
    
    # 保存
    calibrator.save_params()
    
    print("Done!")