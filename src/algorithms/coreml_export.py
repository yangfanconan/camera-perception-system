"""
Core ML 模型导出和优化

针对 Apple Silicon 的部署优化：
1. PyTorch -> Core ML 转换
2. 模型量化 (INT8/FP16)
3. Neural Engine 加速
4. 内存优化

优势：
- 更快的推理速度
- 更低的功耗
- 更小的模型体积
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from loguru import logger
import time
import os


class CoreMLExporter:
    """
    Core ML 模型导出器
    
    将 PyTorch 模型转换为 Core ML 格式
    """
    
    def __init__(self, model: nn.Module, input_shape: Tuple[int, ...] = (1, 3, 480, 640)):
        """
        初始化导出器
        
        Args:
            model: PyTorch 模型
            input_shape: 输入形状
        """
        self.model = model
        self.input_shape = input_shape
        self.coreml_model = None
    
    def export_to_torchscript(self, output_path: str) -> str:
        """
        导出为 TorchScript 格式
        
        Args:
            output_path: 输出路径
            
        Returns:
            保存的文件路径
        """
        self.model.eval()
        
        # 创建示例输入
        example_input = torch.randn(*self.input_shape)
        
        # 追踪模型
        with torch.no_grad():
            traced_model = torch.jit.trace(self.model, example_input)
        
        # 保存
        traced_model.save(output_path)
        logger.info(f"TorchScript model saved to: {output_path}")
        
        return output_path
    
    def export_to_onnx(self, output_path: str, opset_version: int = 12) -> str:
        """
        导出为 ONNX 格式
        
        Args:
            output_path: 输出路径
            opset_version: ONNX opset 版本
            
        Returns:
            保存的文件路径
        """
        self.model.eval()
        
        # 创建示例输入
        example_input = torch.randn(*self.input_shape)
        
        # 导出
        with torch.no_grad():
            torch.onnx.export(
                self.model,
                example_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
        
        logger.info(f"ONNX model saved to: {output_path}")
        
        return output_path
    
    def export_to_coreml(self, output_path: str) -> str:
        """
        导出为 Core ML 格式
        
        Args:
            output_path: 输出路径
            
        Returns:
            保存的文件路径
        """
        try:
            import coremltools as ct
            
            self.model.eval()
            
            # 创建示例输入
            example_input = torch.randn(*self.input_shape)
            
            # 追踪模型
            with torch.no_grad():
                traced_model = torch.jit.trace(self.model, example_input)
            
            # 转换为 Core ML
            self.coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=self.input_shape, name='input')]
            )
            
            # 保存
            self.coreml_model.save(output_path)
            logger.info(f"Core ML model saved to: {output_path}")
            
            return output_path
            
        except ImportError:
            logger.warning("coremltools not installed, skipping Core ML export")
            return None
        except Exception as e:
            logger.error(f"Core ML export failed: {e}")
            return None


class ModelQuantizer:
    """
    模型量化器
    
    支持 INT8 和 FP16 量化
    """
    
    def __init__(self, model: nn.Module):
        """
        初始化量化器
        
        Args:
            model: PyTorch 模型
        """
        self.model = model
        self.quantized_model = None
    
    def quantize_dynamic(self, dtype: str = 'qint8') -> nn.Module:
        """
        动态量化
        
        Args:
            dtype: 量化数据类型 ('qint8', 'float16')
            
        Returns:
            量化后的模型
        """
        if dtype == 'qint8':
            self.quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
            logger.info("Model quantized to INT8 (dynamic)")
        elif dtype == 'float16':
            self.quantized_model = self.model.half()
            logger.info("Model quantized to FP16")
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
        return self.quantized_model
    
    def quantize_static(
        self,
        calibration_data: List[torch.Tensor],
        dtype: str = 'qint8'
    ) -> nn.Module:
        """
        静态量化（需要校准数据）
        
        Args:
            calibration_data: 校准数据
            dtype: 量化数据类型
            
        Returns:
            量化后的模型
        """
        # 设置量化配置
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # 准备量化
        self.model_prepared = torch.quantization.prepare(self.model)
        
        # 校准
        with torch.no_grad():
            for data in calibration_data:
                self.model_prepared(data)
        
        # 转换
        self.quantized_model = torch.quantization.convert(self.model_prepared)
        logger.info("Model quantized to INT8 (static)")
        
        return self.quantized_model
    
    def compare_size(self) -> Dict[str, float]:
        """
        比较模型大小
        
        Returns:
            原始和量化后模型的大小
        """
        def get_model_size(model):
            torch.save(model.state_dict(), '/tmp/temp_model.pt')
            size = os.path.getsize('/tmp/temp_model.pt') / 1024 / 1024
            os.remove('/tmp/temp_model.pt')
            return size
        
        original_size = get_model_size(self.model)
        
        if self.quantized_model is not None:
            quantized_size = get_model_size(self.quantized_model)
            compression_ratio = original_size / quantized_size
        else:
            quantized_size = 0
            compression_ratio = 0
        
        return {
            'original_mb': original_size,
            'quantized_mb': quantized_size,
            'compression_ratio': compression_ratio
        }


class MPSOptimizer:
    """
    Apple Silicon MPS 优化器
    
    针对 Metal Performance Shaders 的优化
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        """
        初始化优化器
        
        Args:
            model: PyTorch 模型
            device: 设备
        """
        self.model = model
        
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
    
    def optimize_for_mps(self) -> nn.Module:
        """
        针对 MPS 优化模型
        
        Returns:
            优化后的模型
        """
        # 移动到 MPS
        self.model = self.model.to(self.device)
        
        # 设置为评估模式
        self.model.eval()
        
        # 禁用梯度计算
        for param in self.model.parameters():
            param.requires_grad = False
        
        logger.info(f"Model optimized for MPS: {self.device}")
        
        return self.model
    
    def benchmark(self, input_shape: Tuple[int, ...], num_runs: int = 100) -> Dict[str, float]:
        """
        性能基准测试
        
        Args:
            input_shape: 输入形状
            num_runs: 运行次数
            
        Returns:
            性能统计
        """
        # 预热
        warmup_input = torch.randn(*input_shape, device=self.device)
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(warmup_input)
        
        # 同步
        if self.device.type == 'mps':
            torch.mps.synchronize()
        
        # 基准测试
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                input_tensor = torch.randn(*input_shape, device=self.device)
                
                start = time.time()
                _ = self.model(input_tensor)
                
                if self.device.type == 'mps':
                    torch.mps.synchronize()
                
                elapsed = time.time() - start
                times.append(elapsed * 1000)  # ms
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'fps': 1000 / np.mean(times)
        }


class InferencePipeline:
    """
    推理管道
    
    整合模型加载、预处理、推理、后处理
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        input_size: Tuple[int, int] = (640, 480)
    ):
        """
        初始化推理管道
        
        Args:
            model: 模型
            device: 设备
            input_size: 输入尺寸 (W, H)
        """
        self.input_size = input_size
        
        # 设备选择
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        # 优化模型
        self.optimizer = MPSOptimizer(model, self.device)
        self.model = self.optimizer.optimize_for_mps()
        
        # 性能统计
        self.stats = {
            'total_inferences': 0,
            'total_preprocess_ms': 0,
            'total_inference_ms': 0,
            'total_postprocess_ms': 0
        }
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        预处理图像
        
        Args:
            image: BGR 图像 (H, W, 3)
            
        Returns:
            tensor: (1, 3, H, W)
        """
        import cv2
        
        # 调整大小
        h, w = self.input_size[1], self.input_size[0]
        img = cv2.resize(image, (w, h))
        
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 归一化
        img = img.astype(np.float32) / 255.0
        
        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        
        # 转换为 tensor
        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
        
        return tensor
    
    def infer(self, image: np.ndarray) -> Dict[str, Any]:
        """
        执行推理
        
        Args:
            image: BGR 图像
            
        Returns:
            推理结果
        """
        # 预处理
        preprocess_start = time.time()
        tensor = self.preprocess(image)
        preprocess_time = (time.time() - preprocess_start) * 1000
        
        # 推理
        inference_start = time.time()
        with torch.no_grad():
            output = self.model(tensor)
        inference_time = (time.time() - inference_start) * 1000
        
        # 后处理
        postprocess_start = time.time()
        result = self.postprocess(output, image.shape[:2])
        postprocess_time = (time.time() - postprocess_start) * 1000
        
        # 更新统计
        self.stats['total_inferences'] += 1
        self.stats['total_preprocess_ms'] += preprocess_time
        self.stats['total_inference_ms'] += inference_time
        self.stats['total_postprocess_ms'] += postprocess_time
        
        return result
    
    def postprocess(self, output: Any, original_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        后处理
        
        Args:
            output: 模型输出
            original_shape: 原始图像尺寸
            
        Returns:
            处理后的结果
        """
        # 子类实现具体后处理逻辑
        return {'output': output}
    
    def get_stats(self) -> Dict[str, float]:
        """获取性能统计"""
        n = self.stats['total_inferences']
        if n == 0:
            return self.stats
        
        return {
            'total_inferences': n,
            'avg_preprocess_ms': self.stats['total_preprocess_ms'] / n,
            'avg_inference_ms': self.stats['total_inference_ms'] / n,
            'avg_postprocess_ms': self.stats['total_postprocess_ms'] / n,
            'avg_total_ms': (self.stats['total_preprocess_ms'] + 
                           self.stats['total_inference_ms'] + 
                           self.stats['total_postprocess_ms']) / n
        }


def export_yolo_to_coreml(yolo_model_path: str, output_path: str) -> str:
    """
    导出 YOLO 模型为 Core ML 格式
    
    Args:
        yolo_model_path: YOLO 模型路径
        output_path: 输出路径
        
    Returns:
        保存的文件路径
    """
    try:
        from ultralytics import YOLO
        
        # 加载模型
        model = YOLO(yolo_model_path)
        
        # 导出为 Core ML
        model.export(format='coreml', imgsz=640)
        
        # 移动到指定位置
        coreml_path = yolo_model_path.replace('.pt', '_coreml.mlpackage')
        if os.path.exists(coreml_path):
            import shutil
            shutil.move(coreml_path, output_path)
            logger.info(f"YOLO exported to Core ML: {output_path}")
            return output_path
        
        return None
        
    except Exception as e:
        logger.error(f"YOLO Core ML export failed: {e}")
        return None


def benchmark_models(models: Dict[str, nn.Module], input_shape: Tuple[int, ...]) -> Dict[str, Dict]:
    """
    对比多个模型的性能
    
    Args:
        models: 模型字典 {name: model}
        input_shape: 输入形状
        
    Returns:
        性能对比结果
    """
    results = {}
    
    for name, model in models.items():
        optimizer = MPSOptimizer(model)
        optimizer.optimize_for_mps()
        benchmark = optimizer.benchmark(input_shape)
        results[name] = benchmark
        logger.info(f"{name}: {benchmark['mean_ms']:.1f}ms ({benchmark['fps']:.1f} FPS)")
    
    return results


# 测试代码
if __name__ == '__main__':
    print("Testing Core ML Export...")
    
    # 创建简单模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 10)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = SimpleModel()
    
    # 测试导出
    exporter = CoreMLExporter(model, (1, 3, 224, 224))
    
    # TorchScript
    exporter.export_to_torchscript('/tmp/model.pt')
    
    # ONNX
    exporter.export_to_onnx('/tmp/model.onnx')
    
    # Core ML (如果可用)
    exporter.export_to_coreml('/tmp/model.mlpackage')
    
    # 测试量化
    quantizer = ModelQuantizer(model)
    quantized = quantizer.quantize_dynamic('qint8')
    size_info = quantizer.compare_size()
    print(f"Size comparison: {size_info}")
    
    # 测试 MPS 优化
    optimizer = MPSOptimizer(model)
    optimizer.optimize_for_mps()
    benchmark = optimizer.benchmark((1, 3, 224, 224))
    print(f"Benchmark: {benchmark}")
    
    print("Done!")