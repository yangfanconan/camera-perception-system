"""
Apple Silicon 优化模块
针对 M 系列芯片的性能优化
"""

import os
import platform
from typing import Optional, Tuple
from loguru import logger
import torch


class AppleSiliconOptimizer:
    """Apple Silicon 优化器"""
    
    def __init__(self):
        """初始化优化器"""
        self.is_apple_silicon = self._detect_apple_silicon()
        self.mps_available = self._check_mps()
        self.device = self._get_optimal_device()
        
        # M5 Pro 特定优化
        self.is_m5_pro = self._detect_m5_pro()
        
        logger.info(f"Apple Silicon detected: {self.is_apple_silicon}")
        logger.info(f"MPS available: {self.mps_available}")
        logger.info(f"Device: {self.device}")
        logger.info(f"M5 Pro detected: {self.is_m5_pro}")
    
    def _detect_apple_silicon(self) -> bool:
        """检测是否为 Apple Silicon"""
        return (
            platform.system() == "Darwin" and 
            platform.machine() == "arm64"
        )
    
    def _detect_m5_pro(self) -> bool:
        """检测是否为 M5 Pro 芯片"""
        try:
            import subprocess
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                capture_output=True,
                text=True
            )
            return "Apple M5 Pro" in result.stdout
        except Exception:
            return False
    
    def _check_mps(self) -> bool:
        """检查 MPS 加速是否可用"""
        if not self.is_apple_silicon:
            return False
        
        try:
            return torch.backends.mps.is_available()
        except Exception:
            return False
    
    def _get_optimal_device(self) -> str:
        """获取最优设备"""
        if self.mps_available:
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def optimize_torch(self) -> None:
        """优化 PyTorch 设置"""
        if not self.is_apple_silicon:
            return
        
        # 设置 MPS 为默认设备
        if self.mps_available:
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            
            # 限制 MPS 内存使用（避免占用全部统一内存）
            import psutil
            total_memory = psutil.virtual_memory().total
            memory_limit = int(total_memory * 0.8)  # 80%
            
            try:
                torch.mps.set_per_process_memory_fraction(0.8)
                logger.info(f"MPS memory limit set to 80% ({memory_limit // 1024**3}GB)")
            except Exception as e:
                logger.warning(f"Failed to set MPS memory limit: {e}")
        
        logger.info("PyTorch optimized for Apple Silicon")
    
    def optimize_opencv(self) -> None:
        """优化 OpenCV 设置"""
        if not self.is_apple_silicon:
            return
        
        try:
            import cv2
            
            # 设置线程数
            cv2.setNumThreads(4)
            
            # 使用 AVFoundation 后端（macOS 原生）
            logger.info("OpenCV optimized for Apple Silicon")
        except Exception as e:
            logger.warning(f"Failed to optimize OpenCV: {e}")
    
    def optimize_ultralytics(self, model) -> None:
        """优化 Ultralytics YOLO 模型"""
        if not self.is_apple_silicon:
            return
        
        try:
            # 设置 MPS 设备
            if self.mps_available:
                model.to(self.device)
                logger.info(f"YOLO model moved to {self.device}")
        except Exception as e:
            logger.warning(f"Failed to optimize YOLO model: {e}")
    
    def get_optimal_workers(self) -> int:
        """获取最优工作线程数"""
        if self.is_m5_pro:
            return 4  # M5 Pro 多核性能强
        elif self.is_apple_silicon:
            return 2
        else:
            return 0
    
    def get_optimal_batch_size(self) -> int:
        """获取最优批次大小"""
        # MPS 单批处理更稳定
        return 1
    
    def get_optimal_image_size(self) -> int:
        """获取最优推理图像尺寸"""
        # 平衡速度和精度
        return 640
    
    def configure_media_pipe(self) -> dict:
        """配置 MediaPipe 优化参数"""
        return {
            'static_image_mode': False,
            'max_num_hands': 2,
            'model_complexity': 1,  # 1 = 平衡模式
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5
        }


# ==================== 全局优化器实例 ====================

_optimizer: Optional[AppleSiliconOptimizer] = None


def get_optimizer() -> AppleSiliconOptimizer:
    """获取优化器实例"""
    global _optimizer
    if _optimizer is None:
        _optimizer = AppleSiliconOptimizer()
        _optimizer.optimize_torch()
        _optimizer.optimize_opencv()
    return _optimizer


def get_device() -> str:
    """获取最优设备"""
    return get_optimizer().device


def optimize_model(model) -> None:
    """优化模型"""
    get_optimizer().optimize_ultralytics(model)


def apply_apple_silicon_optimizations() -> None:
    """应用所有 Apple Silicon 优化"""
    optimizer = get_optimizer()
    optimizer.optimize_torch()
    optimizer.optimize_opencv()


# ==================== 主函数（测试） ====================

def main():
    """测试 Apple Silicon 优化"""
    optimizer = get_optimizer()
    
    print("\n=== Apple Silicon Optimization Status ===")
    print(f"Apple Silicon: {optimizer.is_apple_silicon}")
    print(f"M5 Pro: {optimizer.is_m5_pro}")
    print(f"MPS Available: {optimizer.mps_available}")
    print(f"Optimal Device: {optimizer.device}")
    print(f"Optimal Workers: {optimizer.get_optimal_workers()}")
    print(f"Optimal Batch Size: {optimizer.get_optimal_batch_size()}")
    print(f"Optimal Image Size: {optimizer.get_optimal_image_size()}")
    
    print("\n=== MediaPipe Config ===")
    mp_config = optimizer.configure_media_pipe()
    for key, value in mp_config.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
