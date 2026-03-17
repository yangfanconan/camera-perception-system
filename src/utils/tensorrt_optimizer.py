"""
TensorRT 优化模块

用于加速深度学习模型推理
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
from loguru import logger


class TensorRTOptimizer:
    """TensorRT 模型优化器"""
    
    def __init__(self, model_path: Optional[str] = None, 
                 precision: str = "fp16", 
                 max_batch_size: int = 1,
                 max_workspace_size: int = 1 << 30):  # 1GB
        """
        初始化 TensorRT 优化器
        
        Args:
            model_path: 模型路径（.pt 或 .onnx）
            precision: 精度模式 ('fp32', 'fp16', 'int8')
            max_batch_size: 最大批处理大小
            max_workspace_size: 最大工作空间大小（字节）
        """
        self.model_path = model_path
        self.precision = precision
        self.max_batch_size = max_batch_size
        self.max_workspace_size = max_workspace_size
        self.engine = None
        self.context = None
        self.stream = None
        
        # 检查 TensorRT 是否可用
        self.trt_available = self._check_tensorrt()
        
        if self.trt_available:
            logger.info(f"TensorRT initialized: precision={precision}, max_batch={max_batch_size}")
        else:
            logger.warning("TensorRT not available, falling back to PyTorch")
    
    def _check_tensorrt(self) -> bool:
        """检查 TensorRT 是否可用"""
        try:
            import tensorrt as trt
            logger.info(f"TensorRT version: {trt.__version__}")
            return True
        except ImportError:
            return False
    
    def optimize_yolo(self, model_path: str, output_path: Optional[str] = None) -> str:
        """
        优化 YOLO 模型
        
        Args:
            model_path: YOLO 模型路径
            output_path: 输出路径（可选）
            
        Returns:
            优化后的模型路径
        """
        if not self.trt_available:
            logger.warning("TensorRT not available, skipping optimization")
            return model_path
        
        try:
            import tensorrt as trt
            
            # 加载 YOLO 模型
            from ultralytics import YOLO
            model = YOLO(model_path)
            
            # 导出 ONNX
            onnx_path = model_path.replace('.pt', '.onnx')
            if not Path(onnx_path).exists():
                logger.info(f"Exporting {model_path} to ONNX...")
                model.export(format='onnx', imgsz=640)
            
            # 构建 TensorRT 引擎
            engine_path = onnx_path.replace('.onnx', '.trt')
            if Path(engine_path).exists():
                logger.info(f"Loading existing TensorRT engine: {engine_path}")
                self.engine = self._load_engine(engine_path)
            else:
                logger.info(f"Building TensorRT engine: {engine_path}")
                self.engine = self._build_engine(onnx_path, engine_path)
            
            return engine_path
            
        except Exception as e:
            logger.error(f"TensorRT optimization failed: {e}")
            return model_path
    
    def _build_engine(self, onnx_path: str, engine_path: str):
        """构建 TensorRT 引擎"""
        import tensorrt as trt
        
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)
        
        # 解析 ONNX
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                raise RuntimeError("ONNX parsing failed")
        
        # 配置 builder
        config = builder.create_builder_config()
        config.max_workspace_size = self.max_workspace_size
        
        # 设置精度
        if self.precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif self.precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
            # 需要校准数据
            logger.warning("INT8 mode requires calibration data")
        
        # 构建引擎
        engine = builder.build_engine(network, config)
        
        # 保存引擎
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
        
        return engine
    
    def _load_engine(self, engine_path: str):
        """加载 TensorRT 引擎"""
        import tensorrt as trt
        
        logger = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(logger)
        
        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        return engine
    
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        执行推理
        
        Args:
            input_data: 输入数据 (N, C, H, W)
            
        Returns:
            输出数据
        """
        if self.engine is None:
            raise RuntimeError("TensorRT engine not initialized")
        
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        # 创建执行上下文
        context = self.engine.create_execution_context()
        
        # 分配内存
        h_input = cuda.mem_host_alloc(input_data.nbytes)
        h_output = cuda.mem_host_alloc(self.max_batch_size * 1000 * 4)  # 假设输出
        d_input = cuda.mem_alloc(input_data.nbytes)
        d_output = cuda.mem_alloc(self.max_batch_size * 1000 * 4)
        
        # 拷贝输入数据
        cuda.memcpy_htod(d_input, input_data)
        
        # 执行推理
        context.execute_v2([int(d_input), int(d_output)])
        
        # 拷贝输出数据
        output = np.empty((self.max_batch_size, 1000), dtype=np.float32)
        cuda.memcpy_dtoh(output, d_output)
        
        return output


class ModelOptimizer:
    """模型优化管理器"""
    
    def __init__(self):
        self.optimizers = {}
        self.optimized_models = {}
    
    def optimize_model(self, model_name: str, model_path: str, 
                      method: str = "tensorrt") -> str:
        """
        优化模型
        
        Args:
            model_name: 模型名称
            model_path: 模型路径
            method: 优化方法 ('tensorrt', 'onnx', 'torchscript')
            
        Returns:
            优化后的模型路径
        """
        if method == "tensorrt":
            optimizer = TensorRTOptimizer()
            optimized_path = optimizer.optimize_yolo(model_path)
            self.optimizers[model_name] = optimizer
            self.optimized_models[model_name] = optimized_path
            return optimized_path
        
        elif method == "torchscript":
            return self._optimize_torchscript(model_path)
        
        else:
            return model_path
    
    def _optimize_torchscript(self, model_path: str) -> str:
        """使用 TorchScript 优化"""
        try:
            from ultralytics import YOLO
            
            model = YOLO(model_path)
            ts_path = model_path.replace('.pt', '.torchscript.pt')
            
            if not Path(ts_path).exists():
                logger.info(f"Exporting to TorchScript: {ts_path}")
                model.export(format='torchscript')
            
            return ts_path
            
        except Exception as e:
            logger.error(f"TorchScript optimization failed: {e}")
            return model_path
    
    def get_optimizer(self, model_name: str) -> Optional[TensorRTOptimizer]:
        """获取优化器"""
        return self.optimizers.get(model_name)


# 全局优化器
_optimizer: Optional[ModelOptimizer] = None


def get_model_optimizer() -> ModelOptimizer:
    """获取全局模型优化器"""
    global _optimizer
    if _optimizer is None:
        _optimizer = ModelOptimizer()
    return _optimizer


def optimize_yolo_model(model_path: str, method: str = "tensorrt") -> str:
    """
    便捷函数：优化 YOLO 模型
    
    Args:
        model_path: 模型路径
        method: 优化方法
        
    Returns:
        优化后的模型路径
    """
    optimizer = get_model_optimizer()
    return optimizer.optimize_model("yolo", model_path, method)
