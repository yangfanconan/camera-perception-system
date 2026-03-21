"""
AI模型优化模块

功能：
1. 模型量化
2. 模型剪枝
3. 推理加速
4. 模型管理
"""

import numpy as np
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger
import json
import os


@dataclass
class ModelInfo:
    """模型信息"""
    model_name: str
    model_type: str  # detection, pose, classification
    framework: str   # pytorch, onnx, tensorrt
    input_size: Tuple[int, int]
    num_params: int
    model_size_mb: float
    precision: str   # fp32, fp16, int8
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'framework': self.framework,
            'input_size': self.input_size,
            'num_params': self.num_params,
            'model_size_mb': round(self.model_size_mb, 2),
            'precision': self.precision
        }


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    model_name: str
    framework: str
    precision: str
    batch_size: int
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    fps: float
    gpu_memory_mb: float
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'framework': self.framework,
            'precision': self.precision,
            'batch_size': self.batch_size,
            'avg_latency_ms': round(self.avg_latency_ms, 2),
            'min_latency_ms': round(self.min_latency_ms, 2),
            'max_latency_ms': round(self.max_latency_ms, 2),
            'fps': round(self.fps, 1),
            'gpu_memory_mb': round(self.gpu_memory_mb, 1)
        }


class ModelOptimizer:
    """
    模型优化器
    
    提供模型量化、剪枝等功能
    """
    
    def __init__(self):
        """初始化模型优化器"""
        self.optimized_models: Dict[str, str] = {}
        
        logger.info("ModelOptimizer initialized")
    
    def quantize_dynamic(
        self,
        model_path: str,
        output_path: str,
        dtype: str = 'int8'
    ) -> bool:
        """
        动态量化
        
        Args:
            model_path: 模型路径
            output_path: 输出路径
            dtype: 数据类型 (int8, fp16)
            
        Returns:
            是否成功
        """
        try:
            import torch
            
            # 加载模型
            model = torch.load(model_path, map_location='cpu')
            
            if dtype == 'int8':
                quantized = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear, torch.nn.Conv2d},
                    dtype=torch.qint8
                )
            elif dtype == 'fp16':
                quantized = model.half()
            else:
                logger.error(f"Unsupported dtype: {dtype}")
                return False
            
            # 保存
            torch.save(quantized, output_path)
            
            # 记录
            self.optimized_models[model_path] = output_path
            
            logger.info(f"Quantized model saved: {output_path}")
            return True
            
        except ImportError:
            logger.warning("PyTorch not available for quantization")
            return False
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return False
    
    def export_onnx(
        self,
        model,
        output_path: str,
        input_size: Tuple[int, int] = (640, 640),
        opset_version: int = 12
    ) -> bool:
        """
        导出 ONNX 格式
        
        Args:
            model: 模型对象
            output_path: 输出路径
            input_size: 输入尺寸
            opset_version: ONNX opset 版本
            
        Returns:
            是否成功
        """
        try:
            import torch
            
            # 创建输入
            dummy_input = torch.randn(1, 3, input_size[1], input_size[0])
            
            # 导出
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                opset_version=opset_version,
                input_names=['images'],
                output_names=['output'],
                dynamic_axes={
                    'images': {0: 'batch'},
                    'output': {0: 'batch'}
                }
            )
            
            logger.info(f"ONNX model exported: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return False
    
    def optimize_onnx(
        self,
        onnx_path: str,
        output_path: str,
        optimize_level: str = 'all'
    ) -> bool:
        """
        优化 ONNX 模型
        
        Args:
            onnx_path: ONNX 模型路径
            output_path: 输出路径
            optimize_level: 优化级别
            
        Returns:
            是否成功
        """
        try:
            import onnx
            from onnxoptimizer import optimize
            
            # 加载模型
            model = onnx.load(onnx_path)
            
            # 优化
            passes = ['eliminate_identity', 'eliminate_nop_transpose', 
                     'fuse_bn_into_conv', 'fuse_consecutive_transposes']
            
            optimized = optimize(model, passes)
            
            # 保存
            onnx.save(optimized, output_path)
            
            logger.info(f"ONNX model optimized: {output_path}")
            return True
            
        except ImportError:
            logger.warning("ONNX optimizer not available")
            return False
        except Exception as e:
            logger.error(f"ONNX optimization failed: {e}")
            return False


class InferenceAccelerator:
    """
    推理加速器
    
    提供多种推理后端
    """
    
    def __init__(self):
        """初始化推理加速器"""
        self.backends: Dict[str, Any] = {}
        self.current_backend = None
        
        # 检查可用后端
        self._check_backends()
        
        logger.info(f"InferenceAccelerator initialized, backends: {list(self.backends.keys())}")
    
    def _check_backends(self):
        """检查可用的推理后端"""
        # PyTorch
        try:
            import torch
            self.backends['pytorch'] = {
                'available': True,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'version': torch.__version__
            }
        except ImportError:
            self.backends['pytorch'] = {'available': False}
        
        # ONNX Runtime
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            self.backends['onnxruntime'] = {
                'available': True,
                'providers': providers,
                'version': ort.__version__
            }
        except ImportError:
            self.backends['onnxruntime'] = {'available': False}
        
        # TensorRT
        try:
            import tensorrt
            self.backends['tensorrt'] = {
                'available': True,
                'version': tensorrt.__version__
            }
        except ImportError:
            self.backends['tensorrt'] = {'available': False}
        
        # OpenVINO
        try:
            import openvino
            self.backends['openvino'] = {
                'available': True,
                'version': openvino.__version__
            }
        except ImportError:
            self.backends['openvino'] = {'available': False}
    
    def get_available_backends(self) -> List[str]:
        """获取可用的后端列表"""
        return [
            name for name, info in self.backends.items()
            if info.get('available', False)
        ]
    
    def create_session(
        self,
        model_path: str,
        backend: str = 'auto',
        **kwargs
    ) -> Optional[Any]:
        """
        创建推理会话
        
        Args:
            model_path: 模型路径
            backend: 后端名称
            **kwargs: 额外参数
            
        Returns:
            推理会话
        """
        if backend == 'auto':
            backend = self._select_best_backend(model_path)
        
        if backend == 'onnxruntime':
            return self._create_onnx_session(model_path, **kwargs)
        elif backend == 'tensorrt':
            return self._create_trt_session(model_path, **kwargs)
        elif backend == 'openvino':
            return self._create_ov_session(model_path, **kwargs)
        else:
            logger.warning(f"Unknown backend: {backend}")
            return None
    
    def _select_best_backend(self, model_path: str) -> str:
        """选择最佳后端"""
        if model_path.endswith('.onnx'):
            if self.backends.get('tensorrt', {}).get('available'):
                return 'tensorrt'
            elif self.backends.get('onnxruntime', {}).get('available'):
                return 'onnxruntime'
        
        if model_path.endswith('.engine') or model_path.endswith('.plan'):
            return 'tensorrt'
        
        if model_path.endswith('.xml'):
            return 'openvino'
        
        return 'pytorch'
    
    def _create_onnx_session(self, model_path: str, **kwargs) -> Optional[Any]:
        """创建 ONNX Runtime 会话"""
        try:
            import onnxruntime as ort
            
            providers = kwargs.get('providers', None)
            if providers is None:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            session = ort.InferenceSession(
                model_path,
                sess_options,
                providers=providers
            )
            
            self.current_backend = 'onnxruntime'
            logger.info(f"ONNX session created: {model_path}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to create ONNX session: {e}")
            return None
    
    def _create_trt_session(self, model_path: str, **kwargs) -> Optional[Any]:
        """创建 TensorRT 会话"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            logger = trt.Logger(trt.Logger.WARNING)
            
            with open(model_path, 'rb') as f:
                engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
            
            context = engine.create_execution_context()
            
            self.current_backend = 'tensorrt'
            logger.info(f"TensorRT session created: {model_path}")
            return {'engine': engine, 'context': context}
            
        except Exception as e:
            logger.error(f"Failed to create TensorRT session: {e}")
            return None
    
    def _create_ov_session(self, model_path: str, **kwargs) -> Optional[Any]:
        """创建 OpenVINO 会话"""
        try:
            from openvino.runtime import Core
            
            core = Core()
            model = core.read_model(model_path)
            compiled = core.compile_model(model, 'CPU')
            
            self.current_backend = 'openvino'
            logger.info(f"OpenVINO session created: {model_path}")
            return compiled
            
        except Exception as e:
            logger.error(f"Failed to create OpenVINO session: {e}")
            return None


class ModelBenchmark:
    """
    模型基准测试
    
    测试模型性能
    """
    
    def __init__(self, warmup_runs: int = 3, test_runs: int = 50):
        """
        初始化基准测试
        
        Args:
            warmup_runs: 预热次数
            test_runs: 测试次数
        """
        self.warmup_runs = warmup_runs
        self.test_runs = test_runs
        
        self.results: List[BenchmarkResult] = []
        
        logger.info(f"ModelBenchmark initialized (warmup={warmup_runs}, test={test_runs})")
    
    def benchmark(
        self,
        session: Any,
        input_shape: Tuple[int, ...],
        backend: str = 'onnxruntime',
        batch_size: int = 1
    ) -> Optional[BenchmarkResult]:
        """
        基准测试
        
        Args:
            session: 推理会话
            input_shape: 输入形状
            backend: 后端名称
            batch_size: 批大小
            
        Returns:
            测试结果
        """
        try:
            # 创建输入
            input_data = np.random.randn(*input_shape).astype(np.float32)
            
            # 预热
            for _ in range(self.warmup_runs):
                self._run_inference(session, input_data, backend)
            
            # 测试
            latencies = []
            
            for _ in range(self.test_runs):
                start = time.perf_counter()
                self._run_inference(session, input_data, backend)
                end = time.perf_counter()
                
                latencies.append((end - start) * 1000)  # ms
            
            # 计算统计
            avg_latency = np.mean(latencies)
            min_latency = np.min(latencies)
            max_latency = np.max(latencies)
            fps = 1000 / avg_latency * batch_size
            
            # GPU 内存（如果可用）
            gpu_memory = self._get_gpu_memory()
            
            result = BenchmarkResult(
                model_name='unknown',
                framework=backend,
                precision='fp32',
                batch_size=batch_size,
                avg_latency_ms=avg_latency,
                min_latency_ms=min_latency,
                max_latency_ms=max_latency,
                fps=fps,
                gpu_memory_mb=gpu_memory
            )
            
            self.results.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return None
    
    def _run_inference(self, session: Any, input_data: np.ndarray, backend: str):
        """运行推理"""
        if backend == 'onnxruntime':
            input_name = session.get_inputs()[0].name
            session.run(None, {input_name: input_data})
        elif backend == 'openvino':
            infer_request = session.create_infer_request()
            infer_request.infer({0: input_data})
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    def _get_gpu_memory(self) -> float:
        """获取 GPU 内存使用"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.max_memory_allocated() / 1024 / 1024
        except:
            pass
        return 0.0
    
    def compare_results(self) -> Dict:
        """比较测试结果"""
        if not self.results:
            return {}
        
        comparison = {
            'fastest': min(self.results, key=lambda r: r.avg_latency_ms).to_dict(),
            'highest_fps': max(self.results, key=lambda r: r.fps).to_dict(),
            'all_results': [r.to_dict() for r in self.results]
        }
        
        return comparison


class ModelManager:
    """
    模型管理器
    
    管理模型生命周期
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        初始化模型管理器
        
        Args:
            models_dir: 模型目录
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.loaded_models: Dict[str, Any] = {}
        self.model_info: Dict[str, ModelInfo] = {}
        
        self.optimizer = ModelOptimizer()
        self.accelerator = InferenceAccelerator()
        self.benchmark = ModelBenchmark()
        
        logger.info(f"ModelManager initialized (models_dir={models_dir})")
    
    def register_model(
        self,
        model_name: str,
        model_path: str,
        model_type: str,
        framework: str = 'pytorch'
    ) -> bool:
        """
        注册模型
        
        Args:
            model_name: 模型名称
            model_path: 模型路径
            model_type: 模型类型
            framework: 框架
            
        Returns:
            是否成功
        """
        path = Path(model_path)
        
        if not path.exists():
            logger.error(f"Model not found: {model_path}")
            return False
        
        # 获取模型信息
        model_size = path.stat().st_size / 1024 / 1024
        
        info = ModelInfo(
            model_name=model_name,
            model_type=model_type,
            framework=framework,
            input_size=(640, 640),
            num_params=0,
            model_size_mb=model_size,
            precision='fp32'
        )
        
        self.model_info[model_name] = info
        
        logger.info(f"Model registered: {model_name}")
        return True
    
    def load_model(
        self,
        model_name: str,
        backend: str = 'auto'
    ) -> Optional[Any]:
        """
        加载模型
        
        Args:
            model_name: 模型名称
            backend: 推理后端
            
        Returns:
            模型会话
        """
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        info = self.model_info.get(model_name)
        if not info:
            logger.error(f"Model not registered: {model_name}")
            return None
        
        # 查找模型文件
        model_path = self._find_model_file(model_name)
        if not model_path:
            return None
        
        # 创建会话
        session = self.accelerator.create_session(model_path, backend)
        
        if session:
            self.loaded_models[model_name] = session
            return session
        
        return None
    
    def _find_model_file(self, model_name: str) -> Optional[str]:
        """查找模型文件"""
        extensions = ['.pt', '.onnx', '.engine', '.plan', '.xml']
        
        for ext in extensions:
            path = self.models_dir / f"{model_name}{ext}"
            if path.exists():
                return str(path)
        
        return None
    
    def unload_model(self, model_name: str):
        """卸载模型"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            logger.info(f"Model unloaded: {model_name}")
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """获取模型信息"""
        info = self.model_info.get(model_name)
        return info.to_dict() if info else None
    
    def list_models(self) -> List[Dict]:
        """列出所有模型"""
        return [info.to_dict() for info in self.model_info.values()]
    
    def benchmark_model(
        self,
        model_name: str,
        batch_size: int = 1
    ) -> Optional[Dict]:
        """
        基准测试模型
        
        Args:
            model_name: 模型名称
            batch_size: 批大小
            
        Returns:
            测试结果
        """
        session = self.load_model(model_name)
        if not session:
            return None
        
        info = self.model_info.get(model_name)
        input_shape = (batch_size, 3, info.input_size[1], info.input_size[0])
        
        result = self.benchmark.benchmark(
            session,
            input_shape,
            self.accelerator.current_backend,
            batch_size
        )
        
        return result.to_dict() if result else None
    
    def optimize_model(
        self,
        model_name: str,
        optimization: str = 'quantize_int8'
    ) -> Optional[str]:
        """
        优化模型
        
        Args:
            model_name: 模型名称
            optimization: 优化类型
            
        Returns:
            优化后的模型路径
        """
        model_path = self._find_model_file(model_name)
        if not model_path:
            return None
        
        output_name = f"{model_name}_{optimization}"
        output_path = str(self.models_dir / f"{output_name}.pt")
        
        if optimization == 'quantize_int8':
            if self.optimizer.quantize_dynamic(model_path, output_path, 'int8'):
                return output_path
        
        elif optimization == 'quantize_fp16':
            if self.optimizer.quantize_dynamic(model_path, output_path, 'fp16'):
                return output_path
        
        return None


# 全局实例
_model_manager = None

def get_model_manager() -> ModelManager:
    """获取模型管理器单例"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


# 测试代码
if __name__ == '__main__':
    print("Testing Model Manager...")
    
    manager = ModelManager()
    
    # 列出可用后端
    backends = manager.accelerator.get_available_backends()
    print(f"Available backends: {backends}")
    
    # 列出模型
    models = manager.list_models()
    print(f"Registered models: {len(models)}")
    
    print("\nDone!")