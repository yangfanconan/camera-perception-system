"""
数据可视化仪表盘模块

功能：
1. 实时统计数据
2. 图表生成
3. 报告生成
4. 数据导出
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from loguru import logger
import time
import json
from datetime import datetime, timedelta


@dataclass
class MetricData:
    """指标数据"""
    name: str
    value: float
    unit: str = ""
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    指标收集器
    
    收集和管理系统指标
    """
    
    def __init__(self, history_size: int = 1000):
        """
        初始化指标收集器
        
        Args:
            history_size: 历史数据大小
        """
        self.history_size = history_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.current_values: Dict[str, float] = {}
        
        logger.info(f"MetricsCollector initialized (history_size={history_size})")
    
    def record(self, name: str, value: float, unit: str = "", tags: Dict = None):
        """
        记录指标
        
        Args:
            name: 指标名称
            value: 指标值
            unit: 单位
            tags: 标签
        """
        metric = MetricData(
            name=name,
            value=value,
            unit=unit,
            tags=tags or {}
        )
        
        self.metrics[name].append(metric)
        self.current_values[name] = value
    
    def get_current(self, name: str) -> Optional[float]:
        """获取当前值"""
        return self.current_values.get(name)
    
    def get_history(self, name: str, count: int = None) -> List[MetricData]:
        """
        获取历史数据
        
        Args:
            name: 指标名称
            count: 数量
            
        Returns:
            指标数据列表
        """
        if name not in self.metrics:
            return []
        
        history = list(self.metrics[name])
        
        if count:
            return history[-count:]
        
        return history
    
    def get_statistics(self, name: str, window: int = 100) -> Dict:
        """
        获取统计数据
        
        Args:
            name: 指标名称
            window: 时间窗口
            
        Returns:
            统计数据
        """
        history = self.get_history(name, window)
        
        if not history:
            return {}
        
        values = [m.value for m in history]
        
        return {
            'name': name,
            'current': values[-1] if values else 0,
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'count': len(values)
        }
    
    def get_all_statistics(self) -> Dict[str, Dict]:
        """获取所有指标统计"""
        return {name: self.get_statistics(name) for name in self.metrics.keys()}


class DashboardData:
    """
    仪表盘数据
    
    汇总系统状态和统计数据
    """
    
    def __init__(self):
        """初始化仪表盘数据"""
        self.metrics = MetricsCollector()
        
        # 检测统计
        self.detection_stats = {
            'total_persons': 0,
            'total_hands': 0,
            'total_vehicles': 0,
            'total_faces': 0,
            'total_alerts': 0,
            'total_falls': 0
        }
        
        # 时间序列数据
        self.time_series = {
            'fps': deque(maxlen=300),
            'persons': deque(maxlen=300),
            'alerts': deque(maxlen=100)
        }
        
        # 会话信息
        self.session_start = time.time()
        self.frame_count = 0
        
        logger.info("DashboardData initialized")
    
    def update_detection(self, persons: int, hands: int, vehicles: int = 0, faces: int = 0):
        """更新检测统计"""
        self.detection_stats['total_persons'] += persons
        self.detection_stats['total_hands'] += hands
        self.detection_stats['total_vehicles'] += vehicles
        self.detection_stats['total_faces'] += faces
        
        # 记录时间序列
        current_time = time.time()
        self.time_series['persons'].append((current_time, persons))
        
        # 记录指标
        self.metrics.record('persons', persons)
        self.metrics.record('hands', hands)
    
    def update_fps(self, fps: float):
        """更新 FPS"""
        current_time = time.time()
        self.time_series['fps'].append((current_time, fps))
        self.metrics.record('fps', fps)
    
    def record_alert(self, alert_type: str):
        """记录报警"""
        self.detection_stats['total_alerts'] += 1
        current_time = time.time()
        self.time_series['alerts'].append((current_time, alert_type))
    
    def record_fall(self):
        """记录跌倒"""
        self.detection_stats['total_falls'] += 1
    
    def increment_frame(self):
        """增加帧计数"""
        self.frame_count += 1
    
    def get_summary(self) -> Dict:
        """获取摘要数据"""
        current_time = time.time()
        duration = current_time - self.session_start
        
        return {
            'session': {
                'start_time': self.session_start,
                'duration': round(duration, 1),
                'frame_count': self.frame_count,
                'avg_fps': self.frame_count / max(duration, 1)
            },
            'detection': self.detection_stats.copy(),
            'metrics': self.metrics.get_all_statistics()
        }
    
    def get_time_series(self, name: str, duration: float = 60.0) -> List:
        """
        获取时间序列数据
        
        Args:
            name: 数据名称
            duration: 时间范围（秒）
            
        Returns:
            时间序列数据
        """
        if name not in self.time_series:
            return []
        
        current_time = time.time()
        cutoff_time = current_time - duration
        
        data = []
        for item in self.time_series[name]:
            if isinstance(item, tuple):
                timestamp, value = item
                if timestamp >= cutoff_time:
                    data.append({
                        'time': timestamp,
                        'value': value
                    })
        
        return data
    
    def get_chart_data(self) -> Dict:
        """获取图表数据"""
        return {
            'fps': self.get_time_series('fps', 60.0),
            'persons': self.get_time_series('persons', 60.0),
            'alerts': self.get_time_series('alerts', 300.0)
        }
    
    def reset(self):
        """重置数据"""
        self.detection_stats = {k: 0 for k in self.detection_stats}
        for key in self.time_series:
            self.time_series[key].clear()
        self.session_start = time.time()
        self.frame_count = 0
        logger.info("DashboardData reset")


class ReportGenerator:
    """
    报告生成器
    
    生成统计报告
    """
    
    def __init__(self):
        """初始化报告生成器"""
        logger.info("ReportGenerator initialized")
    
    def generate_session_report(self, dashboard_data: DashboardData) -> Dict:
        """
        生成会话报告
        
        Args:
            dashboard_data: 仪表盘数据
            
        Returns:
            报告数据
        """
        summary = dashboard_data.get_summary()
        
        report = {
            'report_type': 'session',
            'generated_at': datetime.now().isoformat(),
            'session': summary['session'],
            'detection': summary['detection'],
            'analysis': self._analyze_session(summary)
        }
        
        return report
    
    def _analyze_session(self, summary: Dict) -> Dict:
        """分析会话数据"""
        session = summary['session']
        detection = summary['detection']
        
        analysis = {
            'avg_persons_per_frame': detection['total_persons'] / max(session['frame_count'], 1),
            'alerts_per_minute': detection['total_alerts'] / max(session['duration'] / 60, 1),
            'falls_per_hour': detection['total_falls'] / max(session['duration'] / 3600, 1),
            'performance': 'good' if session['avg_fps'] > 25 else 'needs_improvement'
        }
        
        return analysis
    
    def generate_alert_report(self, alerts: List[Dict]) -> Dict:
        """
        生成报警报告
        
        Args:
            alerts: 报警列表
            
        Returns:
            报告数据
        """
        if not alerts:
            return {
                'report_type': 'alerts',
                'total_alerts': 0,
                'by_type': {},
                'by_severity': {}
            }
        
        # 按类型统计
        by_type = defaultdict(int)
        by_severity = defaultdict(int)
        
        for alert in alerts:
            alert_type = alert.get('alert_type', 'unknown')
            severity = alert.get('severity', 'unknown')
            
            by_type[alert_type] += 1
            by_severity[severity] += 1
        
        return {
            'report_type': 'alerts',
            'generated_at': datetime.now().isoformat(),
            'total_alerts': len(alerts),
            'by_type': dict(by_type),
            'by_severity': dict(by_severity),
            'recent_alerts': alerts[-10:]
        }
    
    def export_report(self, report: Dict, filepath: str, format: str = "json"):
        """
        导出报告
        
        Args:
            report: 报告数据
            filepath: 文件路径
            format: 格式 (json, txt)
        """
        if format == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        
        elif format == "txt":
            with open(filepath, 'w', encoding='utf-8') as f:
                self._write_text_report(f, report)
        
        logger.info(f"Report exported to {filepath}")
    
    def _write_text_report(self, f, report: Dict):
        """写入文本报告"""
        f.write("=" * 50 + "\n")
        f.write(f"报告类型: {report.get('report_type', 'unknown')}\n")
        f.write(f"生成时间: {report.get('generated_at', 'unknown')}\n")
        f.write("=" * 50 + "\n\n")
        
        if 'session' in report:
            session = report['session']
            f.write("会话信息:\n")
            f.write(f"  持续时间: {session.get('duration', 0):.1f} 秒\n")
            f.write(f"  帧计数: {session.get('frame_count', 0)}\n")
            f.write(f"  平均 FPS: {session.get('avg_fps', 0):.1f}\n\n")
        
        if 'detection' in report:
            detection = report['detection']
            f.write("检测统计:\n")
            f.write(f"  总人数: {detection.get('total_persons', 0)}\n")
            f.write(f"  总手数: {detection.get('total_hands', 0)}\n")
            f.write(f"  总报警: {detection.get('total_alerts', 0)}\n")
            f.write(f"  总跌倒: {detection.get('total_falls', 0)}\n\n")
        
        if 'analysis' in report:
            analysis = report['analysis']
            f.write("分析结果:\n")
            f.write(f"  平均每帧人数: {analysis.get('avg_persons_per_frame', 0):.2f}\n")
            f.write(f"  每分钟报警: {analysis.get('alerts_per_minute', 0):.2f}\n")
            f.write(f"  性能评估: {analysis.get('performance', 'unknown')}\n")


class DashboardAPI:
    """
    仪表盘 API
    
    提供数据访问接口
    """
    
    def __init__(self):
        """初始化仪表盘 API"""
        self.data = DashboardData()
        self.report_generator = ReportGenerator()
        
        logger.info("DashboardAPI initialized")
    
    def get_summary(self) -> Dict:
        """获取摘要"""
        return self.data.get_summary()
    
    def get_chart_data(self) -> Dict:
        """获取图表数据"""
        return self.data.get_chart_data()
    
    def get_metrics(self) -> Dict:
        """获取指标"""
        return self.data.metrics.get_all_statistics()
    
    def generate_report(self) -> Dict:
        """生成报告"""
        return self.report_generator.generate_session_report(self.data)
    
    def export_report(self, filepath: str, format: str = "json"):
        """导出报告"""
        report = self.generate_report()
        self.report_generator.export_report(report, filepath, format)


# 全局实例
_dashboard = None

def get_dashboard() -> DashboardAPI:
    """获取仪表盘单例"""
    global _dashboard
    if _dashboard is None:
        _dashboard = DashboardAPI()
    return _dashboard


# 测试代码
if __name__ == '__main__':
    print("Testing Dashboard...")
    
    dashboard = DashboardAPI()
    
    # 模拟数据
    for i in range(100):
        dashboard.data.update_detection(
            persons=np.random.randint(0, 5),
            hands=np.random.randint(0, 10)
        )
        dashboard.data.update_fps(np.random.uniform(25, 35))
        dashboard.data.increment_frame()
    
    # 获取摘要
    summary = dashboard.get_summary()
    print("Summary:")
    print(json.dumps(summary, indent=2))
    
    # 生成报告
    report = dashboard.generate_report()
    print("\nReport generated")
    
    print("\nDone!")