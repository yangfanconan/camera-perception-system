"""
边缘计算支持模块

功能：
1. 边缘设备管理
2. 模型部署
3. 资源监控
4. 边云协同
"""

import numpy as np
import time
import platform
import subprocess
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
from loguru import logger
import json
import threading
import queue


@dataclass
class DeviceInfo:
    """设备信息"""
    device_id: str
    device_type: str      # jetson, raspberry_pi, pc, cloud
    name: str
    cpu: str
    cpu_cores: int
    memory_gb: float
    gpu: str = ""
    gpu_memory_gb: float = 0.0
    os: str = ""
    architecture: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'device_id': self.device_id,
            'device_type': self.device_type,
            'name': self.name,
            'cpu': self.cpu,
            'cpu_cores': self.cpu_cores,
            'memory_gb': round(self.memory_gb, 2),
            'gpu': self.gpu,
            'gpu_memory_gb': round(self.gpu_memory_gb, 2),
            'os': self.os,
            'architecture': self.architecture
        }


@dataclass
class ResourceUsage:
    """资源使用情况"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    temperature: float = 0.0
    disk_percent: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'cpu_percent': round(self.cpu_percent, 1),
            'memory_percent': round(self.memory_percent, 1),
            'gpu_percent': round(self.gpu_percent, 1),
            'gpu_memory_percent': round(self.gpu_memory_percent, 1),
            'temperature': round(self.temperature, 1),
            'disk_percent': round(self.disk_percent, 1)
        }


@dataclass
class EdgeConfig:
    """边缘配置"""
    device_id: str
    server_url: str
    sync_interval: float = 5.0
    offline_mode: bool = False
    max_queue_size: int = 1000
    compression: bool = True


class DeviceDetector:
    """
    设备检测器
    
    检测设备类型和硬件信息
    """
    
    # 设备特征
    JETSON_MODELS = [
        'jetson-nano', 'jetson-tx2', 'jetson-xavier', 'jetson-orin'
    ]
    
    RASPBERRY_PI_MODELS = [
        'raspberry-pi-3', 'raspberry-pi-4', 'raspberry-pi-5'
    ]
    
    def __init__(self):
        """初始化设备检测器"""
        logger.info("DeviceDetector initialized")
    
    def detect_device(self) -> DeviceInfo:
        """检测当前设备"""
        device_id = self._get_device_id()
        device_type = self._detect_device_type()
        
        return DeviceInfo(
            device_id=device_id,
            device_type=device_type,
            name=self._get_device_name(device_type),
            cpu=self._get_cpu_info(),
            cpu_cores=self._get_cpu_cores(),
            memory_gb=self._get_memory_size(),
            gpu=self._get_gpu_info(),
            gpu_memory_gb=self._get_gpu_memory(),
            os=self._get_os_info(),
            architecture=platform.machine()
        )
    
    def _get_device_id(self) -> str:
        """获取设备ID"""
        # 尝试获取硬件序列号
        try:
            if platform.system() == 'Linux':
                # 读取机器ID
                with open('/etc/machine-id', 'r') as f:
                    return f.read().strip()[:16]
        except:
            pass
        
        # 使用主机名
        return platform.node()[:16]
    
    def _detect_device_type(self) -> str:
        """检测设备类型"""
        system = platform.system()
        machine = platform.machine()
        
        # 检测 Jetson
        if self._is_jetson():
            return 'jetson'
        
        # 检测树莓派
        if self._is_raspberry_pi():
            return 'raspberry_pi'
        
        # 检测服务器
        if system == 'Linux':
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    content = f.read().lower()
                    if 'epyc' in content or 'xeon' in content:
                        return 'server'
            except:
                pass
        
        # 默认为 PC
        return 'pc'
    
    def _is_jetson(self) -> bool:
        """检测是否为 Jetson 设备"""
        try:
            # 检查 Jetson 特定文件
            jetson_files = [
                '/etc/nv_tegra_release',
                '/sys/module/tegra_fuse'
            ]
            for f in jetson_files:
                if os.path.exists(f):
                    return True
        except:
            pass
        return False
    
    def _is_raspberry_pi(self) -> bool:
        """检测是否为树莓派"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                content = f.read()
                if 'Raspberry Pi' in content:
                    return True
        except:
            pass
        return False
    
    def _get_device_name(self, device_type: str) -> str:
        """获取设备名称"""
        names = {
            'jetson': 'NVIDIA Jetson',
            'raspberry_pi': 'Raspberry Pi',
            'server': 'Server',
            'pc': 'Desktop/Laptop',
            'cloud': 'Cloud Instance'
        }
        return names.get(device_type, 'Unknown Device')
    
    def _get_cpu_info(self) -> str:
        """获取 CPU 信息"""
        try:
            if platform.system() == 'Linux':
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'model name' in line:
                            return line.split(':')[1].strip()
        except:
            pass
        
        return platform.processor() or 'Unknown CPU'
    
    def _get_cpu_cores(self) -> int:
        """获取 CPU 核心数"""
        import os
        return os.cpu_count() or 1
    
    def _get_memory_size(self) -> float:
        """获取内存大小 (GB)"""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024 ** 3)
        except:
            return 8.0  # 默认 8GB
    
    def _get_gpu_info(self) -> str:
        """获取 GPU 信息"""
        try:
            # 尝试 nvidia-smi
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
        except:
            pass
        
        return ""
    
    def _get_gpu_memory(self) -> float:
        """获取 GPU 内存 (GB)"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                mem_str = result.stdout.strip().split()[0]
                return float(mem_str) / 1024  # MB to GB
        except:
            pass
        
        return 0.0
    
    def _get_os_info(self) -> str:
        """获取操作系统信息"""
        return f"{platform.system()} {platform.release()}"


class ResourceMonitor:
    """
    资源监控器
    
    监控系统资源使用
    """
    
    def __init__(self, history_size: int = 100):
        """
        初始化资源监控器
        
        Args:
            history_size: 历史记录大小
        """
        self.history_size = history_size
        self.history: deque = deque(maxlen=history_size)
        
        self._monitoring = False
        self._monitor_thread = None
        
        logger.info("ResourceMonitor initialized")
    
    def start_monitoring(self, interval: float = 1.0):
        """
        开始监控
        
        Args:
            interval: 监控间隔（秒）
        """
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        
        logger.info(f"Resource monitoring started (interval={interval}s)")
    
    def stop_monitoring(self):
        """停止监控"""
        self._monitoring = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """监控循环"""
        while self._monitoring:
            usage = self.get_current_usage()
            self.history.append(usage)
            time.sleep(interval)
    
    def get_current_usage(self) -> ResourceUsage:
        """获取当前资源使用"""
        usage = ResourceUsage(timestamp=time.time())
        
        try:
            import psutil
            
            usage.cpu_percent = psutil.cpu_percent(interval=0.1)
            usage.memory_percent = psutil.virtual_memory().percent
            usage.disk_percent = psutil.disk_usage('/').percent
            
            # 温度（如果可用）
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    if entries:
                        usage.temperature = entries[0].current
                        break
            
        except ImportError:
            # psutil 不可用时的备用方案
            usage.cpu_percent = self._get_cpu_usage_fallback()
            usage.memory_percent = 50.0
        
        # GPU 使用率
        usage.gpu_percent, usage.gpu_memory_percent = self._get_gpu_usage()
        
        return usage
    
    def _get_cpu_usage_fallback(self) -> float:
        """CPU 使用率备用方案"""
        try:
            if platform.system() == 'Linux':
                with open('/proc/stat', 'r') as f:
                    line = f.readline()
                    values = line.split()[1:5]
                    values = [int(v) for v in values]
                    total = sum(values)
                    idle = values[3]
                    return 100 * (1 - idle / total) if total > 0 else 0
        except:
            pass
        return 0.0
    
    def _get_gpu_usage(self) -> Tuple[float, float]:
        """获取 GPU 使用率"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                gpu_util = float(parts[0])
                mem_parts = parts[1].split('/')
                mem_used = float(mem_parts[0].strip())
                mem_total = float(mem_parts[1].strip())
                mem_util = 100 * mem_used / mem_total if mem_total > 0 else 0
                return gpu_util, mem_util
        except:
            pass
        
        return 0.0, 0.0
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if not self.history:
            return {}
        
        cpu_values = [u.cpu_percent for u in self.history]
        mem_values = [u.memory_percent for u in self.history]
        
        return {
            'cpu': {
                'avg': np.mean(cpu_values),
                'max': np.max(cpu_values),
                'min': np.min(cpu_values)
            },
            'memory': {
                'avg': np.mean(mem_values),
                'max': np.max(mem_values),
                'min': np.min(mem_values)
            },
            'samples': len(self.history)
        }


class EdgeOptimizer:
    """
    边缘优化器
    
    根据设备能力优化配置
    """
    
    # 设备能力配置
    DEVICE_CONFIGS = {
        'jetson': {
            'max_fps': 30,
            'resolution': (1280, 720),
            'batch_size': 1,
            'use_gpu': True,
            'model_precision': 'fp16'
        },
        'raspberry_pi': {
            'max_fps': 15,
            'resolution': (640, 480),
            'batch_size': 1,
            'use_gpu': False,
            'model_precision': 'int8'
        },
        'server': {
            'max_fps': 60,
            'resolution': (1920, 1080),
            'batch_size': 4,
            'use_gpu': True,
            'model_precision': 'fp32'
        },
        'pc': {
            'max_fps': 30,
            'resolution': (1280, 720),
            'batch_size': 1,
            'use_gpu': True,
            'model_precision': 'fp16'
        }
    }
    
    def __init__(self):
        """初始化边缘优化器"""
        self.detector = DeviceDetector()
        self.device_info: Optional[DeviceInfo] = None
        self.config: Dict = {}
        
        logger.info("EdgeOptimizer initialized")
    
    def initialize(self) -> DeviceInfo:
        """初始化并检测设备"""
        self.device_info = self.detector.detect_device()
        self.config = self.DEVICE_CONFIGS.get(
            self.device_info.device_type,
            self.DEVICE_CONFIGS['pc']
        )
        
        logger.info(f"Device detected: {self.device_info.device_type}")
        return self.device_info
    
    def get_optimized_config(self) -> Dict:
        """获取优化后的配置"""
        if not self.config:
            self.initialize()
        
        return self.config.copy()
    
    def adjust_for_resources(self, usage: ResourceUsage) -> Dict:
        """
        根据资源使用情况调整配置
        
        Args:
            usage: 资源使用情况
            
        Returns:
            调整后的配置
        """
        config = self.get_optimized_config()
        
        # CPU 使用率高时降低帧率
        if usage.cpu_percent > 80:
            config['max_fps'] = max(10, config['max_fps'] // 2)
        
        # 内存使用率高时降低分辨率
        if usage.memory_percent > 85:
            w, h = config['resolution']
            config['resolution'] = (w // 2, h // 2)
        
        # 温度过高时降低性能
        if usage.temperature > 80:
            config['max_fps'] = max(5, config['max_fps'] // 3)
        
        return config
    
    def get_model_recommendation(self) -> Dict:
        """获取模型推荐"""
        if not self.device_info:
            self.initialize()
        
        device_type = self.device_info.device_type
        
        recommendations = {
            'jetson': {
                'detection': 'yolov8n-pose.engine',  # TensorRT
                'pose': 'yolov8n-pose.engine',
                'use_tensorrt': True
            },
            'raspberry_pi': {
                'detection': 'yolov8n-pose-int8.tflite',  # TFLite
                'pose': 'yolov8n-pose-int8.tflite',
                'use_tflite': True
            },
            'server': {
                'detection': 'yolov8n-pose.pt',
                'pose': 'yolov8n-pose.pt',
                'use_tensorrt': False
            },
            'pc': {
                'detection': 'yolov8n-pose.onnx',
                'pose': 'yolov8n-pose.onnx',
                'use_onnx': True
            }
        }
        
        return recommendations.get(device_type, recommendations['pc'])


class EdgeClient:
    """
    边缘客户端
    
    与云端通信
    """
    
    def __init__(self, config: EdgeConfig):
        """
        初始化边缘客户端
        
        Args:
            config: 边缘配置
        """
        self.config = config
        
        # 数据队列
        self.data_queue: queue.Queue = queue.Queue(maxsize=config.max_queue_size)
        
        # 状态
        self.connected = False
        self.sync_thread = None
        self.running = False
        
        # 统计
        self.sync_count = 0
        self.error_count = 0
        
        logger.info(f"EdgeClient initialized (server={config.server_url})")
    
    def connect(self) -> bool:
        """连接到服务器"""
        if self.config.offline_mode:
            logger.info("Running in offline mode")
            return True
        
        try:
            # 尝试连接
            import requests
            response = requests.get(
                f"{self.config.server_url}/api/status",
                timeout=5
            )
            
            if response.status_code == 200:
                self.connected = True
                logger.info("Connected to server")
                return True
            
        except Exception as e:
            logger.warning(f"Failed to connect: {e}")
        
        self.connected = False
        return False
    
    def start_sync(self):
        """开始同步"""
        if self.running:
            return
        
        self.running = True
        self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.sync_thread.start()
        
        logger.info("Sync started")
    
    def stop_sync(self):
        """停止同步"""
        self.running = False
        
        if self.sync_thread:
            self.sync_thread.join(timeout=2.0)
        
        logger.info("Sync stopped")
    
    def _sync_loop(self):
        """同步循环"""
        while self.running:
            try:
                self._sync_data()
                time.sleep(self.config.sync_interval)
            except Exception as e:
                logger.error(f"Sync error: {e}")
                self.error_count += 1
    
    def _sync_data(self):
        """同步数据"""
        if not self.connected or self.data_queue.empty():
            return
        
        # 批量获取数据
        batch = []
        while not self.data_queue.empty() and len(batch) < 100:
            try:
                batch.append(self.data_queue.get_nowait())
            except queue.Empty:
                break
        
        if not batch:
            return
        
        # 发送到服务器
        try:
            import requests
            
            data = {
                'device_id': self.config.device_id,
                'data': batch,
                'timestamp': time.time()
            }
            
            if self.config.compression:
                import gzip
                import json
                compressed = gzip.compress(json.dumps(data).encode())
                headers = {'Content-Encoding': 'gzip'}
            else:
                compressed = json.dumps(data).encode()
                headers = {}
            
            response = requests.post(
                f"{self.config.server_url}/api/edge/data",
                data=compressed,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                self.sync_count += 1
            else:
                self.error_count += 1
                
        except Exception as e:
            logger.error(f"Failed to sync data: {e}")
            self.error_count += 1
    
    def push_data(self, data: Dict):
        """
        推送数据
        
        Args:
            data: 数据字典
        """
        data['timestamp'] = time.time()
        
        try:
            self.data_queue.put(data, block=False)
        except queue.Full:
            logger.warning("Data queue full, dropping data")
    
    def get_status(self) -> Dict:
        """获取状态"""
        return {
            'device_id': self.config.device_id,
            'connected': self.connected,
            'queue_size': self.data_queue.qsize(),
            'sync_count': self.sync_count,
            'error_count': self.error_count
        }


class EdgeComputingManager:
    """
    边缘计算管理器
    
    整合所有边缘计算功能
    """
    
    def __init__(self, config: EdgeConfig = None):
        """
        初始化边缘计算管理器
        
        Args:
            config: 边缘配置
        """
        self.optimizer = EdgeOptimizer()
        self.monitor = ResourceMonitor()
        self.client: Optional[EdgeClient] = None
        
        if config:
            self.client = EdgeClient(config)
        
        self.device_info: Optional[DeviceInfo] = None
        
        logger.info("EdgeComputingManager initialized")
    
    def initialize(self) -> DeviceInfo:
        """初始化"""
        self.device_info = self.optimizer.initialize()
        self.monitor.start_monitoring()
        
        if self.client:
            self.client.connect()
            self.client.start_sync()
        
        return self.device_info
    
    def get_device_info(self) -> Dict:
        """获取设备信息"""
        if not self.device_info:
            self.initialize()
        
        return self.device_info.to_dict()
    
    def get_resource_usage(self) -> Dict:
        """获取资源使用"""
        return self.monitor.get_current_usage().to_dict()
    
    def get_optimized_config(self) -> Dict:
        """获取优化配置"""
        usage = self.monitor.get_current_usage()
        return self.optimizer.adjust_for_resources(usage)
    
    def push_data(self, data: Dict):
        """推送数据到云端"""
        if self.client:
            self.client.push_data(data)
    
    def get_status(self) -> Dict:
        """获取状态"""
        return {
            'device': self.get_device_info(),
            'resource': self.get_resource_usage(),
            'config': self.get_optimized_config(),
            'client': self.client.get_status() if self.client else None
        }
    
    def shutdown(self):
        """关闭"""
        self.monitor.stop_monitoring()
        
        if self.client:
            self.client.stop_sync()


# 全局实例
_edge_manager = None

def get_edge_manager() -> EdgeComputingManager:
    """获取边缘计算管理器单例"""
    global _edge_manager
    if _edge_manager is None:
        _edge_manager = EdgeComputingManager()
    return _edge_manager


# 测试代码
if __name__ == '__main__':
    print("Testing Edge Computing Manager...")
    
    manager = EdgeComputingManager()
    
    # 初始化
    device_info = manager.initialize()
    print(f"Device: {device_info.name}")
    print(f"Type: {device_info.device_type}")
    print(f"CPU: {device_info.cpu}")
    print(f"Memory: {device_info.memory_gb:.1f} GB")
    
    # 获取资源使用
    usage = manager.get_resource_usage()
    print(f"CPU: {usage['cpu_percent']}%")
    print(f"Memory: {usage['memory_percent']}%")
    
    # 获取优化配置
    config = manager.get_optimized_config()
    print(f"Max FPS: {config['max_fps']}")
    print(f"Resolution: {config['resolution']}")
    
    manager.shutdown()
    
    print("\nDone!")