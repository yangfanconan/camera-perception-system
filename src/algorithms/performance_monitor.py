"""
性能监控模块

功能：
1. 实时性能监控
2. 资源使用统计
3. 性能瓶颈分析
4. 自动优化建议
"""

import numpy as np
import psutil
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from loguru import logger
import threading
import json
from datetime import datetime


@dataclass
class PerformanceMetrics:
    """性能指标"""
    # 帧率
    fps: float = 0.0
    frame_time: float = 0.0  # 毫秒
    
    # CPU
    cpu_percent: float = 0.0
    cpu_count: int = 0
    
    # 内存
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    
    # GPU (如果可用)
    gpu_percent: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    
    # 检测
    detection_time: float = 0.0  # 毫秒
    persons_detected: int = 0
    hands_detected: int = 0
    
    # 网络
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0
    
    # 时间戳
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            'fps': round(self.fps, 1),
            'frame_time': round(self.frame_time, 2),
            'cpu_percent': round(self.cpu_percent, 1),
            'memory_percent': round(self.memory_percent, 1),
            'memory_used_mb': round(self.memory_used_mb, 1),
            'detection_time': round(self.detection_time, 2),
            'timestamp': self.timestamp
        }


class PerformanceMonitor:
    """
    性能监控器
    
    监控系统性能
    """
    
    def __init__(self, history_size: int = 300):
        """
        初始化性能监控器
        
        Args:
            history_size: 历史数据大小
        """
        self.history_size = history_size
        
        # 历史数据
        self.metrics_history: deque = deque(maxlen=history_size)
        self.frame_times: deque = deque(maxlen=100)
        
        # 当前指标
        self.current_metrics = PerformanceMetrics()
        
        # 帧计数
        self.frame_count = 0
        self.last_frame_time = time.time()
        
        # 检测时间
        self.detection_times: deque = deque(maxlen=100)
        
        # 监控线程
        self.running = False
        self.monitor_thread = None
        
        # 网络统计
        self.last_net_io = psutil.net_io_counters()
        
        logger.info(f"PerformanceMonitor initialized (history_size={history_size})")
    
    def start(self):
        """启动监控"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("PerformanceMonitor started")
    
    def stop(self):
        """停止监控"""
        self.running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        logger.info("PerformanceMonitor stopped")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                self._collect_metrics()
                time.sleep(1.0)  # 每秒采集一次
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
    
    def _collect_metrics(self):
        """采集性能指标"""
        metrics = PerformanceMetrics()
        
        # CPU
        metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
        metrics.cpu_count = psutil.cpu_count()
        
        # 内存
        memory = psutil.virtual_memory()
        metrics.memory_percent = memory.percent
        metrics.memory_used_mb = memory.used / (1024 * 1024)
        metrics.memory_available_mb = memory.available / (1024 * 1024)
        
        # GPU (尝试获取)
        try:
            import torch
            if torch.cuda.is_available():
                metrics.gpu_percent = 0  # 需要额外库
                metrics.gpu_memory_used_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                metrics.gpu_memory_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        except:
            pass
        
        # 网络
        net_io = psutil.net_io_counters()
        metrics.network_sent_mb = (net_io.bytes_sent - self.last_net_io.bytes_sent) / (1024 * 1024)
        metrics.network_recv_mb = (net_io.bytes_recv - self.last_net_io.bytes_recv) / (1024 * 1024)
        self.last_net_io = net_io
        
        # 帧率
        if self.frame_times:
            avg_frame_time = np.mean(list(self.frame_times))
            metrics.frame_time = avg_frame_time
            metrics.fps = 1000.0 / max(avg_frame_time, 1)
        
        # 检测时间
        if self.detection_times:
            metrics.detection_time = np.mean(list(self.detection_times))
        
        # 保存
        self.current_metrics = metrics
        self.metrics_history.append(metrics)
    
    def record_frame(self):
        """记录帧"""
        current_time = time.time()
        
        if self.last_frame_time:
            frame_time = (current_time - self.last_frame_time) * 1000  # 毫秒
            self.frame_times.append(frame_time)
        
        self.last_frame_time = current_time
        self.frame_count += 1
    
    def record_detection(self, detection_time: float, persons: int = 0, hands: int = 0):
        """
        记录检测
        
        Args:
            detection_time: 检测时间（毫秒）
            persons: 检测到的人数
            hands: 检测到的手数
        """
        self.detection_times.append(detection_time)
        self.current_metrics.persons_detected = persons
        self.current_metrics.hands_detected = hands
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """获取当前指标"""
        return self.current_metrics
    
    def get_history(self, count: int = None) -> List[PerformanceMetrics]:
        """
        获取历史数据
        
        Args:
            count: 数量
            
        Returns:
            历史数据列表
        """
        history = list(self.metrics_history)
        
        if count:
            return history[-count:]
        
        return history
    
    def get_statistics(self) -> Dict:
        """
        获取统计数据
        
        Returns:
            统计数据
        """
        if not self.metrics_history:
            return {}
        
        fps_values = [m.fps for m in self.metrics_history if m.fps > 0]
        cpu_values = [m.cpu_percent for m in self.metrics_history]
        memory_values = [m.memory_percent for m in self.metrics_history]
        
        return {
            'fps': {
                'current': self.current_metrics.fps,
                'avg': np.mean(fps_values) if fps_values else 0,
                'min': np.min(fps_values) if fps_values else 0,
                'max': np.max(fps_values) if fps_values else 0
            },
            'cpu': {
                'current': self.current_metrics.cpu_percent,
                'avg': np.mean(cpu_values) if cpu_values else 0,
                'max': np.max(cpu_values) if cpu_values else 0
            },
            'memory': {
                'current': self.current_metrics.memory_percent,
                'avg': np.mean(memory_values) if memory_values else 0,
                'max': np.max(memory_values) if memory_values else 0
            },
            'frame_count': self.frame_count,
            'uptime': time.time() - self.metrics_history[0].timestamp if self.metrics_history else 0
        }


class PerformanceAnalyzer:
    """
    性能分析器
    
    分析性能瓶颈
    """
    
    def __init__(self, monitor: PerformanceMonitor = None):
        """
        初始化性能分析器
        
        Args:
            monitor: 性能监控器
        """
        self.monitor = monitor or PerformanceMonitor()
        
        # 阈值
        self.thresholds = {
            'fps_low': 15.0,
            'fps_critical': 10.0,
            'cpu_high': 80.0,
            'cpu_critical': 95.0,
            'memory_high': 80.0,
            'memory_critical': 95.0,
            'detection_time_high': 100.0,  # 毫秒
            'detection_time_critical': 200.0
        }
        
        logger.info("PerformanceAnalyzer initialized")
    
    def analyze(self) -> Dict:
        """
        分析性能
        
        Returns:
            分析结果
        """
        metrics = self.monitor.get_current_metrics()
        stats = self.monitor.get_statistics()
        
        issues = []
        recommendations = []
        
        # 分析 FPS
        if metrics.fps < self.thresholds['fps_critical']:
            issues.append({
                'type': 'fps_critical',
                'severity': 'critical',
                'message': f"FPS 过低: {metrics.fps:.1f}"
            })
            recommendations.append("降低检测分辨率或跳帧")
        elif metrics.fps < self.thresholds['fps_low']:
            issues.append({
                'type': 'fps_low',
                'severity': 'warning',
                'message': f"FPS 偏低: {metrics.fps:.1f}"
            })
            recommendations.append("考虑优化检测参数")
        
        # 分析 CPU
        if metrics.cpu_percent > self.thresholds['cpu_critical']:
            issues.append({
                'type': 'cpu_critical',
                'severity': 'critical',
                'message': f"CPU 使用率过高: {metrics.cpu_percent:.1f}%"
            })
            recommendations.append("减少并发任务或升级硬件")
        elif metrics.cpu_percent > self.thresholds['cpu_high']:
            issues.append({
                'type': 'cpu_high',
                'severity': 'warning',
                'message': f"CPU 使用率较高: {metrics.cpu_percent:.1f}%"
            })
        
        # 分析内存
        if metrics.memory_percent > self.thresholds['memory_critical']:
            issues.append({
                'type': 'memory_critical',
                'severity': 'critical',
                'message': f"内存使用率过高: {metrics.memory_percent:.1f}%"
            })
            recommendations.append("释放内存或增加内存")
        elif metrics.memory_percent > self.thresholds['memory_high']:
            issues.append({
                'type': 'memory_high',
                'severity': 'warning',
                'message': f"内存使用率较高: {metrics.memory_percent:.1f}%"
            })
        
        # 分析检测时间
        if metrics.detection_time > self.thresholds['detection_time_critical']:
            issues.append({
                'type': 'detection_slow',
                'severity': 'warning',
                'message': f"检测时间过长: {metrics.detection_time:.1f}ms"
            })
            recommendations.append("使用更小的模型或降低检测频率")
        
        # 计算健康分数
        health_score = self._calculate_health_score(metrics)
        
        return {
            'health_score': health_score,
            'status': self._get_status(health_score),
            'issues': issues,
            'recommendations': recommendations,
            'metrics': metrics.to_dict(),
            'statistics': stats
        }
    
    def _calculate_health_score(self, metrics: PerformanceMetrics) -> float:
        """计算健康分数"""
        score = 100.0
        
        # FPS 分数 (0-30分)
        if metrics.fps >= 30:
            score -= 0
        elif metrics.fps >= 20:
            score -= 10
        elif metrics.fps >= 15:
            score -= 20
        else:
            score -= 30
        
        # CPU 分数 (0-30分)
        if metrics.cpu_percent <= 50:
            score -= 0
        elif metrics.cpu_percent <= 70:
            score -= 10
        elif metrics.cpu_percent <= 85:
            score -= 20
        else:
            score -= 30
        
        # 内存分数 (0-20分)
        if metrics.memory_percent <= 50:
            score -= 0
        elif metrics.memory_percent <= 70:
            score -= 5
        elif metrics.memory_percent <= 85:
            score -= 10
        else:
            score -= 20
        
        # 检测时间分数 (0-20分)
        if metrics.detection_time <= 30:
            score -= 0
        elif metrics.detection_time <= 50:
            score -= 5
        elif metrics.detection_time <= 100:
            score -= 10
        else:
            score -= 20
        
        return max(0, score)
    
    def _get_status(self, health_score: float) -> str:
        """获取状态"""
        if health_score >= 80:
            return 'excellent'
        elif health_score >= 60:
            return 'good'
        elif health_score >= 40:
            return 'fair'
        else:
            return 'poor'
    
    def get_optimization_suggestions(self) -> List[str]:
        """获取优化建议"""
        suggestions = []
        metrics = self.monitor.get_current_metrics()
        
        if metrics.fps < 20:
            suggestions.append("降低视频分辨率")
            suggestions.append("启用帧跳过")
            suggestions.append("使用更小的模型")
        
        if metrics.cpu_percent > 70:
            suggestions.append("减少检测频率")
            suggestions.append("启用多线程处理")
            suggestions.append("关闭不必要的功能")
        
        if metrics.memory_percent > 70:
            suggestions.append("清理历史数据")
            suggestions.append("减少缓存大小")
            suggestions.append("启用数据压缩")
        
        return suggestions


class PerformanceOptimizer:
    """
    性能优化器
    
    自动优化性能
    """
    
    def __init__(self, analyzer: PerformanceAnalyzer = None):
        """
        初始化优化器
        
        Args:
            analyzer: 性能分析器
        """
        self.analyzer = analyzer or PerformanceAnalyzer()
        
        # 优化策略
        self.strategies = {
            'reduce_fps': self._reduce_fps,
            'reduce_resolution': self._reduce_resolution,
            'skip_frames': self._skip_frames,
            'reduce_detections': self._reduce_detections
        }
        
        # 当前设置
        self.settings = {
            'target_fps': 30,
            'resolution_scale': 1.0,
            'skip_frame_count': 0,
            'detection_interval': 1
        }
        
        logger.info("PerformanceOptimizer initialized")
    
    def optimize(self) -> Dict:
        """
        执行优化
        
        Returns:
            优化结果
        """
        analysis = self.analyzer.analyze()
        
        optimizations = []
        
        # 根据健康分数优化
        if analysis['health_score'] < 60:
            # 应用优化策略
            if self.analyzer.monitor.current_metrics.fps < 15:
                result = self._reduce_fps()
                if result:
                    optimizations.append(result)
            
            if self.analyzer.monitor.current_metrics.cpu_percent > 80:
                result = self._skip_frames()
                if result:
                    optimizations.append(result)
        
        return {
            'analysis': analysis,
            'optimizations': optimizations,
            'settings': self.settings
        }
    
    def _reduce_fps(self) -> Optional[Dict]:
        """降低帧率"""
        if self.settings['target_fps'] > 15:
            self.settings['target_fps'] = max(15, self.settings['target_fps'] - 5)
            return {
                'action': 'reduce_fps',
                'new_value': self.settings['target_fps']
            }
        return None
    
    def _reduce_resolution(self) -> Optional[Dict]:
        """降低分辨率"""
        if self.settings['resolution_scale'] > 0.5:
            self.settings['resolution_scale'] = max(0.5, self.settings['resolution_scale'] - 0.1)
            return {
                'action': 'reduce_resolution',
                'new_value': self.settings['resolution_scale']
            }
        return None
    
    def _skip_frames(self) -> Optional[Dict]:
        """跳帧"""
        if self.settings['skip_frame_count'] < 3:
            self.settings['skip_frame_count'] += 1
            return {
                'action': 'skip_frames',
                'new_value': self.settings['skip_frame_count']
            }
        return None
    
    def _reduce_detections(self) -> Optional[Dict]:
        """减少检测"""
        if self.settings['detection_interval'] < 5:
            self.settings['detection_interval'] += 1
            return {
                'action': 'reduce_detections',
                'new_value': self.settings['detection_interval']
            }
        return None
    
    def get_settings(self) -> Dict:
        """获取当前设置"""
        return self.settings.copy()


# 全局实例
_performance_monitor = None
_performance_analyzer = None

def get_performance_monitor() -> PerformanceMonitor:
    """获取性能监控器单例"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor

def get_performance_analyzer() -> PerformanceAnalyzer:
    """获取性能分析器单例"""
    global _performance_analyzer
    if _performance_analyzer is None:
        _performance_analyzer = PerformanceAnalyzer()
    return _performance_analyzer


# 测试代码
if __name__ == '__main__':
    print("Testing Performance Monitor...")
    
    monitor = PerformanceMonitor()
    monitor.start()
    
    # 模拟帧
    for i in range(100):
        monitor.record_frame()
        monitor.record_detection(np.random.uniform(20, 50), 2, 4)
        time.sleep(0.03)
    
    # 获取指标
    metrics = monitor.get_current_metrics()
    print(f"Current metrics: {metrics.to_dict()}")
    
    # 分析
    analyzer = PerformanceAnalyzer(monitor)
    analysis = analyzer.analyze()
    print(f"\nAnalysis: {json.dumps(analysis, indent=2)}")
    
    monitor.stop()
    print("\nDone!")