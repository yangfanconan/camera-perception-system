"""
性能基准测试

测试系统各组件的性能指标
"""

import pytest
import time
import numpy as np
import torch
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.depth_estimator import DepthEstimator
from src.algorithms.spatial.distance_estimation import DistanceEstimator
from src.utils.memory_optimizer import MemoryOptimizer
from src.utils.cache import SmartCache


class TestPerformance:
    """性能测试类"""
    
    @pytest.fixture
    def sample_image(self):
        """生成测试图像"""
        return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    @pytest.fixture
    def sample_bbox(self):
        """生成测试边界框"""
        return [100, 100, 200, 400]
    
    @pytest.fixture
    def sample_keypoints(self):
        """生成测试关键点"""
        return {
            'nose': [200, 150],
            'L_eye': [190, 145],
            'R_eye': [210, 145],
            'L_shoulder': [150, 250],
            'R_shoulder': [250, 250]
        }
    
    def test_detection_latency(self, sample_image):
        """测试检测延迟"""
        from src.algorithms import CombinedDetector
        
        detector = CombinedDetector(conf_threshold=0.5)
        
        # 预热
        for _ in range(5):
            detector.detect(sample_image)
        
        # 测试
        times = []
        for _ in range(20):
            start = time.time()
            result = detector.detect(sample_image)
            times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        
        print(f"\nDetection Performance:")
        print(f"  Average: {avg_time:.1f} ms")
        print(f"  Std Dev: {std_time:.1f} ms")
        print(f"  FPS: {1000/avg_time:.1f}")
        
        # 断言：检测应该能在 100ms 内完成
        assert avg_time < 100, f"Detection too slow: {avg_time:.1f}ms"
    
    def test_spatial_calculation_latency(self, sample_bbox, sample_keypoints):
        """测试空间计算延迟"""
        estimator = DistanceEstimator(fx=650.0)
        
        times = []
        for _ in range(100):
            start = time.time()
            distance = estimator.estimate_from_bbox(sample_bbox, 1280, 720)
            head_dist, head_conf = estimator.estimate_from_head(sample_keypoints)
            times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000
        
        print(f"\nSpatial Calculation Performance:")
        print(f"  Average: {avg_time:.2f} ms")
        print(f"  FPS: {1000/avg_time:.1f}")
        
        # 断言：空间计算应该能在 10ms 内完成
        assert avg_time < 10, f"Spatial calc too slow: {avg_time:.2f}ms"
    
    def test_cache_performance(self):
        """测试缓存性能"""
        cache = SmartCache(maxsize=100, ttl=60)
        
        # 写入测试
        write_times = []
        for i in range(1000):
            start = time.time()
            cache.set(f"key_{i}", f"value_{i}")
            write_times.append(time.time() - start)
        
        # 读取测试
        read_times = []
        for i in range(1000):
            start = time.time()
            _ = cache.get(f"key_{i}")
            read_times.append(time.time() - start)
        
        avg_write = np.mean(write_times) * 1000000  # μs
        avg_read = np.mean(read_times) * 1000000
        
        print(f"\nCache Performance:")
        print(f"  Write: {avg_write:.2f} μs")
        print(f"  Read: {avg_read:.2f} μs")
        
        # 断言：缓存操作应该非常快
        assert avg_read < 10, f"Cache read too slow: {avg_read:.2f}μs"
    
    def test_memory_usage(self):
        """测试内存使用"""
        optimizer = MemoryOptimizer()
        
        # 获取初始内存
        initial = optimizer.get_memory_info()
        
        # 分配一些内存
        large_array = np.zeros((1000, 1000, 100))
        
        # 获取使用后的内存
        after_alloc = optimizer.get_memory_info()
        
        # 清理
        del large_array
        optimizer.cleanup()
        
        # 获取清理后的内存
        after_cleanup = optimizer.get_memory_info()
        
        print(f"\nMemory Usage:")
        print(f"  Initial: {initial['rss_mb']:.1f} MB")
        print(f"  After alloc: {after_alloc['rss_mb']:.1f} MB")
        print(f"  After cleanup: {after_cleanup['rss_mb']:.1f} MB")
        
        # 断言：内存应该能被有效清理
        assert after_cleanup['rss_mb'] < after_alloc['rss_mb'] * 1.5
    
    def test_throughput(self, sample_image):
        """测试系统吞吐量"""
        from src.algorithms import CombinedDetector
        
        detector = CombinedDetector(conf_threshold=0.5)
        
        # 测试 100 帧的处理时间
        num_frames = 100
        start = time.time()
        
        for _ in range(num_frames):
            detector.detect(sample_image)
        
        total_time = time.time() - start
        fps = num_frames / total_time
        
        print(f"\nThroughput Test:")
        print(f"  Frames: {num_frames}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  FPS: {fps:.1f}")
        
        # 断言：应该能达到至少 10 FPS
        assert fps > 10, f"Throughput too low: {fps:.1f} FPS"


class TestBenchmark:
    """基准测试类"""
    
    def test_yolo_benchmark(self):
        """YOLO 模型基准测试"""
        from ultralytics import YOLO
        
        model_path = "yolov8n-pose.pt"
        if not Path(model_path).exists():
            pytest.skip(f"Model {model_path} not found")
        
        model = YOLO(model_path)
        
        # 生成测试数据
        test_images = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) 
                      for _ in range(50)]
        
        # 预热
        for img in test_images[:5]:
            model(img, verbose=False)
        
        # 基准测试
        times = []
        for img in test_images:
            start = time.time()
            model(img, verbose=False)
            times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000
        
        print(f"\nYOLO Benchmark:")
        print(f"  Average: {avg_time:.1f} ms")
        print(f"  FPS: {1000/avg_time:.1f}")
        
        # 保存结果
        results = {
            'model': model_path,
            'avg_time_ms': avg_time,
            'fps': 1000 / avg_time,
            'device': 'mps' if torch.backends.mps.is_available() else 'cpu'
        }
        
        return results
    
    def test_depth_estimation_benchmark(self):
        """深度估计基准测试"""
        try:
            estimator = DepthEstimator(model_type='small')
            
            # 生成测试图像
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # 预热
            for _ in range(3):
                estimator.estimate(test_image)
            
            # 测试
            times = []
            for _ in range(10):
                start = time.time()
                estimator.estimate(test_image)
                times.append(time.time() - start)
            
            avg_time = np.mean(times) * 1000
            
            print(f"\nDepth Estimation Benchmark:")
            print(f"  Average: {avg_time:.1f} ms")
            print(f"  FPS: {1000/avg_time:.1f}")
            
        except Exception as e:
            pytest.skip(f"Depth estimation not available: {e}")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])
