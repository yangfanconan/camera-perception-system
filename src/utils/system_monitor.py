"""
系统监控模块
功能：
1. 系统资源监控
2. 服务健康检查
3. 自动告警
"""

import psutil
import time
import asyncio
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from loguru import logger


@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_total_mb: float
    disk_percent: float
    network_sent_mb: float
    network_recv_mb: float


class SystemMonitor:
    """系统监控器"""
    
    def __init__(self, check_interval: float = 5.0):
        """
        初始化系统监控器
        
        Args:
            check_interval: 检查间隔（秒）
        """
        self.check_interval = check_interval
        self.running = False
        self.metrics_history = []
        self.max_history = 100
        self.alert_callbacks = []
        
        # 告警阈值
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0
        }
        
    async def start(self):
        """启动监控"""
        self.running = True
        logger.info("System monitor started")
        
        while self.running:
            try:
                metrics = self._collect_metrics()
                self._store_metrics(metrics)
                self._check_alerts(metrics)
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"System monitor error: {e}")
                await asyncio.sleep(self.check_interval)
                
    def stop(self):
        """停止监控"""
        self.running = False
        logger.info("System monitor stopped")
        
    def _collect_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_percent=memory.percent,
            memory_used_mb=memory.used / 1024 / 1024,
            memory_total_mb=memory.total / 1024 / 1024,
            disk_percent=disk.percent,
            network_sent_mb=network.bytes_sent / 1024 / 1024,
            network_recv_mb=network.bytes_recv / 1024 / 1024
        )
        
    def _store_metrics(self, metrics: SystemMetrics):
        """存储指标"""
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)
            
    def _check_alerts(self, metrics: SystemMetrics):
        """检查告警"""
        alerts = []
        
        if metrics.cpu_percent > self.thresholds['cpu_percent']:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
            
        if metrics.memory_percent > self.thresholds['memory_percent']:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
            
        if metrics.disk_percent > self.thresholds['disk_percent']:
            alerts.append(f"High disk usage: {metrics.disk_percent:.1f}%")
            
        for alert in alerts:
            logger.warning(f"System alert: {alert}")
            for callback in self.alert_callbacks:
                callback(alert)
                
    def get_current_metrics(self) -> Dict[str, Any]:
        """获取当前指标"""
        if not self.metrics_history:
            return {}
        
        latest = self.metrics_history[-1]
        return {
            'timestamp': latest.timestamp,
            'cpu_percent': latest.cpu_percent,
            'memory': {
                'percent': latest.memory_percent,
                'used_mb': round(latest.memory_used_mb, 2),
                'total_mb': round(latest.memory_total_mb, 2)
            },
            'disk_percent': latest.disk_percent,
            'network': {
                'sent_mb': round(latest.network_sent_mb, 2),
                'recv_mb': round(latest.network_recv_mb, 2)
            }
        }
        
    def get_average_metrics(self, last_n: int = 10) -> Dict[str, float]:
        """获取平均指标"""
        if len(self.metrics_history) < last_n:
            return {}
            
        recent = self.metrics_history[-last_n:]
        return {
            'avg_cpu': sum(m.cpu_percent for m in recent) / len(recent),
            'avg_memory': sum(m.memory_percent for m in recent) / len(recent),
            'avg_disk': sum(m.disk_percent for m in recent) / len(recent)
        }


# 全局监控器
_monitor: Optional[SystemMonitor] = None

def get_system_monitor() -> SystemMonitor:
    """获取全局系统监控器"""
    global _monitor
    if _monitor is None:
        _monitor = SystemMonitor()
    return _monitor
