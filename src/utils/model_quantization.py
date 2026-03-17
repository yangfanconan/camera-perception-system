"""
模型量化模块

支持 INT8/FP16 量化，减少模型大小和推理延迟
"""

import torch
import torch.quantization
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Callable
from loguru import logger


class ModelQuantization:
    """模型量化器"""
    
    def __init__(self, model: torch.nn.Module = None):
        self.model = model
        self.quantized_model = None
        
    def quantize_dynamic(self, model_path: str, output_path: Optional[str] = None) -> str:
        """
        动态量化（适用于 CPU）
        
        将模型权重从 FP32 量化为 INT8，减少内存占用
        
        Args:
            model_path: 模型路径
            output_path: 输出路径（可选）
            
        Returns:
            量化后的模型路径
        """
        try:
            # 加载模型
            model = torch.load(model_path, map_location='cpu')
            
            # 动态量化
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=torch.qint8
            )
            
            # 保存量化模型
            if output_path is None:
                output_path = model_path.replace('.pt', '_quantized.pt')
            
            torch.save(quantized_model, output_path)
            
            # 计算压缩率
            original_size = Path(model_path).stat().st_size / (1024 * 1024)
            quantized_size = Path(output_path).stat().st_size / (1024 * 1024)
            compression_ratio = original_size / quantized_size
            
            logger.info(f"Dynamic quantization complete:")
            logger.info(f"  Original size: {original_size:.2f} MB")
            logger.info(f"  Quantized size: {quantized_size:.2f} MB")
            logger.info(f"  Compression ratio: {compression_ratio:.2f}x")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Dynamic quantization failed: {e}")
            return model_path
    
    def quantize_static(self, model_path: str, 
                       calibration_data: torch.Tensor,
                       output_path: Optional[str] = None) -> str:
        """
        静态量化（需要校准数据）
        
        提供更高效的量化，但需要代表性校准数据
        
        Args:
            model_path: 模型路径
            calibration_data: 校准数据
            output_path: 输出路径（可选）
            
        Returns:
            量化后的模型路径
        """
        try:
            # 加载模型
            model = torch.load(model_path, map_location='cpu')
            model.eval()
            
            # 融合层（优化）
            model_fused = torch.quantization.fuse_modules(
                model, 
                [['conv', 'bn', 'relu']], 
                inplace=True
            )
            
            # 配置量化
            model_fused.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            
            # 准备量化
            torch.quantization.prepare(model_fused, inplace=True)
            
            # 校准
            with torch.no_grad():
                for data in calibration_data:
                    model_fused(data)
            
            # 转换
            quantized_model = torch.quantization.convert(model_fused, inplace=True)
            
            # 保存
            if output_path is None:
                output_path = model_path.replace('.pt', '_static_quantized.pt')
            
            torch.save(quantized_model, output_path)
            
            logger.info(f"Static quantization complete: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Static quantization failed: {e}")
            return model_path
    
    def quantize_yolo(self, model_path: str, method: str = "dynamic") -> str:
        """
        量化 YOLO 模型
        
        Args:
            model_path: YOLO 模型路径
            method: 量化方法 ('dynamic', 'static')
            
        Returns:
            量化后的模型路径
        """
        if method == "dynamic":
            return self.quantize_dynamic(model_path)
        elif method == "static":
            # 生成校准数据
            calibration_data = self._generate_calibration_data()
            return self.quantize_static(model_path, calibration_data)
        else:
            return model_path
    
    def _generate_calibration_data(self, num_samples: int = 100) -> torch.Tensor:
        """生成校准数据"""
        # 生成随机图像数据作为校准
        calibration_data = []
        for _ in range(num_samples):
            # 随机图像 (3, 640, 640)
            img = torch.randn(1, 3, 640, 640)
            calibration_data.append(img)
        
        return calibration_data
    
    def benchmark(self, model_path: str, input_shape: Tuple[int, ...] = (1, 3, 640, 640),
                 num_runs: int = 100) -> dict:
        """
        基准测试
        
        Args:
            model_path: 模型路径
            input_shape: 输入形状
            num_runs: 运行次数
            
        Returns:
            性能统计
        """
        try:
            model = torch.load(model_path, map_location='cpu')
            model.eval()
            
            # 生成测试输入
            dummy_input = torch.randn(*input_shape)
            
            # 预热
            for _ in range(10):
                with torch.no_grad():
                    _ = model(dummy_input)
            
            # 测试
            import time
            times = []
            
            for _ in range(num_runs):
                start = time.time()
                with torch.no_grad():
                    _ = model(dummy_input)
                times.append(time.time() - start)
            
            avg_time = np.mean(times) * 1000  # ms
            std_time = np.std(times) * 1000
            
            # 计算模型大小
            model_size = Path(model_path).stat().st_size / (1024 * 1024)
            
            return {
                "model_path": model_path,
                "model_size_mb": round(model_size, 2),
                "avg_inference_time_ms": round(avg_time, 2),
                "std_inference_time_ms": round(std_time, 2),
                "fps": round(1000 / avg_time, 1)
            }
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return {}


class QuantizationConfig:
    """量化配置"""
    
    # 支持的量化方法
    METHODS = {
        'dynamic': '动态量化（快速，无需校准）',
        'static': '静态量化（高效，需要校准）',
        'qat': '量化感知训练（最准确，需要训练）'
    }
    
    # 推荐配置
    RECOMMENDED = {
        'cpu': 'dynamic',
        'gpu': 'none',  # GPU 通常使用 TensorRT
        'mobile': 'static'
    }


def quantize_model(model_path: str, method: str = "dynamic") -> str:
    """
    便捷函数：量化模型
    
    Args:
        model_path: 模型路径
        method: 量化方法
        
    Returns:
        量化后的模型路径
    """
    quantizer = ModelQuantization()
    return quantizer.quantize_yolo(model_path, method)


def benchmark_quantization(original_path: str, quantized_path: str) -> dict:
    """
    对比原始模型和量化模型的性能
    
    Args:
        original_path: 原始模型路径
        quantized_path: 量化模型路径
        
    Returns:
        对比结果
    """
    quantizer = ModelQuantization()
    
    original_stats = quantizer.benchmark(original_path)
    quantized_stats = quantizer.benchmark(quantized_path)
    
    if not original_stats or not quantized_stats:
        return {}
    
    comparison = {
        "original": original_stats,
        "quantized": quantized_stats,
        "improvement": {
            "size_reduction": round(
                (1 - quantized_stats["model_size_mb"] / original_stats["model_size_mb"]) * 100, 1
            ),
            "speedup": round(
                original_stats["avg_inference_time_ms"] / quantized_stats["avg_inference_time_ms"], 2
            )
        }
    }
    
    return comparison
