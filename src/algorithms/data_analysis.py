"""
数据分析与报告模块

功能：
1. 数据统计分析
2. 趋势分析
3. 报告生成
4. 数据可视化
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger
import time
import json
from pathlib import Path
from datetime import datetime, timedelta


@dataclass
class StatisticsResult:
    """统计结果"""
    metric_name: str
    value: float
    unit: str
    period: str
    timestamp: float
    details: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'metric_name': self.metric_name,
            'value': round(self.value, 3),
            'unit': self.unit,
            'period': self.period,
            'timestamp': self.timestamp,
            'details': self.details
        }


@dataclass
class TrendPoint:
    """趋势数据点"""
    timestamp: float
    value: float
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'value': round(self.value, 3)
        }


@dataclass
class TrendAnalysis:
    """趋势分析结果"""
    metric_name: str
    points: List[TrendPoint]
    trend: str  # increasing, decreasing, stable
    slope: float
    r_squared: float
    forecast: List[TrendPoint] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'metric_name': self.metric_name,
            'points': [p.to_dict() for p in self.points],
            'trend': self.trend,
            'slope': round(self.slope, 6),
            'r_squared': round(self.r_squared, 4),
            'forecast': [p.to_dict() for p in self.forecast]
        }


@dataclass
class Report:
    """报告"""
    report_id: str
    title: str
    period_start: float
    period_end: float
    generated_at: float
    sections: List[Dict]
    summary: str
    
    def to_dict(self) -> Dict:
        return {
            'report_id': self.report_id,
            'title': self.title,
            'period_start': self.period_start,
            'period_end': self.period_end,
            'generated_at': self.generated_at,
            'sections': self.sections,
            'summary': self.summary
        }


class DataAggregator:
    """
    数据聚合器
    
    聚合检测数据
    """
    
    def __init__(self, window_size: int = 3600):
        """
        初始化数据聚合器
        
        Args:
            window_size: 时间窗口（秒）
        """
        self.window_size = window_size
        
        # 数据存储
        self.data_store: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        
        logger.info(f"DataAggregator initialized (window={window_size}s)")
    
    def add_data(self, metric_name: str, value: float, timestamp: float = None):
        """
        添加数据
        
        Args:
            metric_name: 指标名称
            value: 值
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.data_store[metric_name].append((timestamp, value))
        
        # 清理过期数据
        self._cleanup(metric_name)
    
    def _cleanup(self, metric_name: str):
        """清理过期数据"""
        cutoff = time.time() - self.window_size
        self.data_store[metric_name] = [
            (t, v) for t, v in self.data_store[metric_name]
            if t > cutoff
        ]
    
    def get_statistics(self, metric_name: str) -> Optional[StatisticsResult]:
        """
        获取统计数据
        
        Args:
            metric_name: 指标名称
            
        Returns:
            统计结果
        """
        data = self.data_store.get(metric_name, [])
        
        if not data:
            return None
        
        values = [v for _, v in data]
        
        return StatisticsResult(
            metric_name=metric_name,
            value=np.mean(values),
            unit='count' if 'count' in metric_name else 'value',
            period=f'{self.window_size}s',
            timestamp=time.time(),
            details={
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'std': float(np.std(values)),
                'count': len(values),
                'sum': float(np.sum(values))
            }
        )
    
    def get_all_statistics(self) -> Dict[str, StatisticsResult]:
        """获取所有统计数据"""
        return {
            name: self.get_statistics(name)
            for name in self.data_store
            if self.data_store[name]
        }


class TrendAnalyzer:
    """
    趋势分析器
    
    分析数据趋势
    """
    
    def __init__(self):
        """初始化趋势分析器"""
        logger.info("TrendAnalyzer initialized")
    
    def analyze(
        self,
        data: List[Tuple[float, float]],
        forecast_periods: int = 5
    ) -> Optional[TrendAnalysis]:
        """
        分析趋势
        
        Args:
            data: 数据点列表 [(timestamp, value), ...]
            forecast_periods: 预测周期数
            
        Returns:
            趋势分析结果
        """
        if len(data) < 3:
            return None
        
        # 排序数据
        data = sorted(data, key=lambda x: x[0])
        
        # 提取时间和值
        timestamps = np.array([t for t, _ in data])
        values = np.array([v for _, v in data])
        
        # 归一化时间
        t_normalized = (timestamps - timestamps[0]) / max(timestamps[-1] - timestamps[0], 1)
        
        # 线性回归
        try:
            coeffs = np.polyfit(t_normalized, values, 1)
            slope = coeffs[0]
            intercept = coeffs[1]
            
            # 计算 R²
            predicted = slope * t_normalized + intercept
            ss_res = np.sum((values - predicted) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            r_squared = 1 - (ss_res / max(ss_tot, 1e-10))
            
            # 判断趋势
            if abs(slope) < 0.01 * np.std(values):
                trend = 'stable'
            elif slope > 0:
                trend = 'increasing'
            else:
                trend = 'decreasing'
            
            # 预测
            forecast = []
            if forecast_periods > 0:
                time_step = (timestamps[-1] - timestamps[0]) / max(len(timestamps) - 1, 1)
                
                for i in range(1, forecast_periods + 1):
                    future_t = timestamps[-1] + i * time_step
                    future_t_norm = (future_t - timestamps[0]) / max(timestamps[-1] - timestamps[0], 1)
                    future_value = slope * future_t_norm + intercept
                    
                    forecast.append(TrendPoint(
                        timestamp=future_t,
                        value=max(0, future_value)  # 确保非负
                    ))
            
            # 创建趋势点
            points = [TrendPoint(timestamp=t, value=v) for t, v in data]
            
            return TrendAnalysis(
                metric_name='',
                points=points,
                trend=trend,
                slope=slope,
                r_squared=r_squared,
                forecast=forecast
            )
            
        except Exception as e:
            logger.error(f"Trend analysis error: {e}")
            return None
    
    def detect_anomaly(
        self,
        data: List[Tuple[float, float]],
        threshold: float = 2.0
    ) -> List[Tuple[float, float]]:
        """
        检测异常值
        
        Args:
            data: 数据点列表
            threshold: 标准差阈值
            
        Returns:
            异常点列表
        """
        if len(data) < 3:
            return []
        
        values = np.array([v for _, v in data])
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return []
        
        anomalies = []
        for t, v in data:
            z_score = abs(v - mean) / std
            if z_score > threshold:
                anomalies.append((t, v))
        
        return anomalies


class ReportGenerator:
    """
    报告生成器
    
    生成分析报告
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        初始化报告生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ReportGenerator initialized (output_dir={output_dir})")
    
    def generate_daily_report(
        self,
        statistics: Dict[str, StatisticsResult],
        trends: Dict[str, TrendAnalysis],
        events: List[Dict] = None
    ) -> Report:
        """
        生成日报
        
        Args:
            statistics: 统计数据
            trends: 趋势数据
            events: 事件列表
            
        Returns:
            报告对象
        """
        now = time.time()
        day_start = now - 86400
        
        report_id = f"daily_{datetime.now().strftime('%Y%m%d')}"
        
        sections = []
        
        # 概览部分
        overview = self._generate_overview_section(statistics)
        sections.append(overview)
        
        # 趋势部分
        trend_section = self._generate_trend_section(trends)
        sections.append(trend_section)
        
        # 事件部分
        if events:
            event_section = self._generate_event_section(events)
            sections.append(event_section)
        
        # 生成摘要
        summary = self._generate_summary(statistics, trends, events)
        
        report = Report(
            report_id=report_id,
            title=f"每日报告 - {datetime.now().strftime('%Y-%m-%d')}",
            period_start=day_start,
            period_end=now,
            generated_at=now,
            sections=sections,
            summary=summary
        )
        
        # 保存报告
        self._save_report(report)
        
        return report
    
    def _generate_overview_section(self, statistics: Dict[str, StatisticsResult]) -> Dict:
        """生成概览部分"""
        metrics = []
        
        for name, stat in statistics.items():
            if stat:
                metrics.append({
                    'name': name,
                    'value': stat.value,
                    'unit': stat.unit,
                    'details': stat.details
                })
        
        return {
            'title': '数据概览',
            'type': 'overview',
            'metrics': metrics
        }
    
    def _generate_trend_section(self, trends: Dict[str, TrendAnalysis]) -> Dict:
        """生成趋势部分"""
        trend_data = []
        
        for name, trend in trends.items():
            if trend:
                trend_data.append({
                    'name': name,
                    'trend': trend.trend,
                    'slope': trend.slope,
                    'r_squared': trend.r_squared
                })
        
        return {
            'title': '趋势分析',
            'type': 'trends',
            'trends': trend_data
        }
    
    def _generate_event_section(self, events: List[Dict]) -> Dict:
        """生成事件部分"""
        # 按类型分组
        by_type = defaultdict(list)
        for event in events:
            by_type[event.get('type', 'unknown')].append(event)
        
        return {
            'title': '事件记录',
            'type': 'events',
            'total': len(events),
            'by_type': {k: len(v) for k, v in by_type.items()},
            'events': events[:50]  # 限制数量
        }
    
    def _generate_summary(
        self,
        statistics: Dict[str, StatisticsResult],
        trends: Dict[str, TrendAnalysis],
        events: List[Dict]
    ) -> str:
        """生成摘要"""
        parts = []
        
        # 统计摘要
        if statistics:
            parts.append(f"共收集 {len(statistics)} 项指标数据。")
        
        # 趋势摘要
        if trends:
            increasing = sum(1 for t in trends.values() if t and t.trend == 'increasing')
            decreasing = sum(1 for t in trends.values() if t and t.trend == 'decreasing')
            stable = sum(1 for t in trends.values() if t and t.trend == 'stable')
            
            parts.append(f"趋势分析：{increasing} 项上升，{decreasing} 项下降，{stable} 项稳定。")
        
        # 事件摘要
        if events:
            parts.append(f"记录 {len(events)} 个事件。")
        
        return ' '.join(parts) if parts else "暂无数据。"
    
    def _save_report(self, report: Report):
        """保存报告"""
        filepath = self.output_dir / f"{report.report_id}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Report saved: {filepath}")
    
    def generate_html_report(self, report: Report) -> str:
        """
        生成 HTML 报告
        
        Args:
            report: 报告对象
            
        Returns:
            HTML 内容
        """
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{report.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .section {{ margin: 20px 0; padding: 15px; background: #f9f9f9; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #e8f5e9; border-radius: 5px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
        .metric-name {{ font-size: 12px; color: #666; }}
        .trend-up {{ color: #4CAF50; }}
        .trend-down {{ color: #f44336; }}
        .trend-stable {{ color: #FF9800; }}
        .summary {{ background: #e3f2fd; padding: 20px; border-radius: 5px; margin-top: 20px; }}
        .footer {{ margin-top: 30px; text-align: center; color: #999; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{report.title}</h1>
        <p>报告周期：{datetime.fromtimestamp(report.period_start).strftime('%Y-%m-%d %H:%M')} - {datetime.fromtimestamp(report.period_end).strftime('%Y-%m-%d %H:%M')}</p>
        
        <div class="summary">
            <strong>摘要：</strong> {report.summary}
        </div>
"""
        
        for section in report.sections:
            html += f"""
        <div class="section">
            <h2>{section['title']}</h2>
"""
            
            if section['type'] == 'overview':
                for metric in section.get('metrics', []):
                    html += f"""
            <div class="metric">
                <div class="metric-value">{metric['value']:.1f}</div>
                <div class="metric-name">{metric['name']}</div>
            </div>
"""
            
            elif section['type'] == 'trends':
                for trend in section.get('trends', []):
                    trend_class = f"trend-{trend['trend']}"
                    html += f"""
            <p><span class="{trend_class}">{trend['name']}: {trend['trend']}</span> (R²={trend['r_squared']:.3f})</p>
"""
            
            elif section['type'] == 'events':
                html += f"""
            <p>总事件数: {section['total']}</p>
            <ul>
"""
                for event_type, count in section.get('by_type', {}).items():
                    html += f"                <li>{event_type}: {count}</li>\n"
                html += "            </ul>\n"
            
            html += "        </div>\n"
        
        html += f"""
        <div class="footer">
            生成时间：{datetime.fromtimestamp(report.generated_at).strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
        
        # 保存 HTML
        html_path = self.output_dir / f"{report.report_id}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return html


class DataAnalyzer:
    """
    数据分析器
    
    整合所有分析功能
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        初始化数据分析器
        
        Args:
            output_dir: 输出目录
        """
        self.aggregator = DataAggregator()
        self.trend_analyzer = TrendAnalyzer()
        self.report_generator = ReportGenerator(output_dir)
        
        # 数据存储
        self.data_history: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        
        logger.info("DataAnalyzer initialized")
    
    def record_metric(self, name: str, value: float, timestamp: float = None):
        """
        记录指标
        
        Args:
            name: 指标名称
            value: 值
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.aggregator.add_data(name, value, timestamp)
        self.data_history[name].append((timestamp, value))
    
    def get_statistics(self, metric_name: str = None) -> Dict:
        """
        获取统计数据
        
        Args:
            metric_name: 指标名称（可选）
            
        Returns:
            统计数据
        """
        if metric_name:
            result = self.aggregator.get_statistics(metric_name)
            return result.to_dict() if result else {}
        
        return {
            name: stat.to_dict()
            for name, stat in self.aggregator.get_all_statistics().items()
        }
    
    def get_trend(self, metric_name: str) -> Optional[Dict]:
        """
        获取趋势分析
        
        Args:
            metric_name: 指标名称
            
        Returns:
            趋势分析结果
        """
        data = self.data_history.get(metric_name, [])
        
        if len(data) < 3:
            return None
        
        result = self.trend_analyzer.analyze(data)
        
        if result:
            result.metric_name = metric_name
            return result.to_dict()
        
        return None
    
    def get_all_trends(self) -> Dict[str, Dict]:
        """获取所有趋势"""
        trends = {}
        
        for name, data in self.data_history.items():
            if len(data) >= 3:
                result = self.trend_analyzer.analyze(data)
                if result:
                    result.metric_name = name
                    trends[name] = result.to_dict()
        
        return trends
    
    def detect_anomalies(self, metric_name: str, threshold: float = 2.0) -> List[Dict]:
        """
        检测异常
        
        Args:
            metric_name: 指标名称
            threshold: 阈值
            
        Returns:
            异常点列表
        """
        data = self.data_history.get(metric_name, [])
        anomalies = self.trend_analyzer.detect_anomaly(data, threshold)
        
        return [{'timestamp': t, 'value': v} for t, v in anomalies]
    
    def generate_report(self, events: List[Dict] = None) -> Report:
        """
        生成报告
        
        Args:
            events: 事件列表
            
        Returns:
            报告对象
        """
        statistics = self.aggregator.get_all_statistics()
        trends = {}
        
        for name, data in self.data_history.items():
            if len(data) >= 3:
                result = self.trend_analyzer.analyze(data)
                if result:
                    result.metric_name = name
                    trends[name] = result
        
        return self.report_generator.generate_daily_report(statistics, trends, events)
    
    def export_data(self, metric_name: str, format: str = 'json') -> str:
        """
        导出数据
        
        Args:
            metric_name: 指标名称
            format: 格式 (json, csv)
            
        Returns:
            导出内容
        """
        data = self.data_history.get(metric_name, [])
        
        if format == 'json':
            return json.dumps([
                {'timestamp': t, 'value': v}
                for t, v in data
            ])
        
        elif format == 'csv':
            lines = ['timestamp,value']
            lines.extend([f'{t},{v}' for t, v in data])
            return '\n'.join(lines)
        
        return ''


# 全局实例
_data_analyzer = None

def get_data_analyzer() -> DataAnalyzer:
    """获取数据分析器单例"""
    global _data_analyzer
    if _data_analyzer is None:
        _data_analyzer = DataAnalyzer()
    return _data_analyzer


# 测试代码
if __name__ == '__main__':
    print("Testing Data Analyzer...")
    
    analyzer = DataAnalyzer()
    
    # 记录数据
    for i in range(100):
        analyzer.record_metric('person_count', np.random.randint(1, 10))
        analyzer.record_metric('fps', 30 + np.random.randn() * 2)
        time.sleep(0.01)
    
    # 获取统计
    stats = analyzer.get_statistics()
    print(f"Statistics: {list(stats.keys())}")
    
    # 获取趋势
    trend = analyzer.get_trend('person_count')
    if trend:
        print(f"Trend: {trend['trend']}")
    
    # 生成报告
    report = analyzer.generate_report()
    print(f"Report: {report.title}")
    
    print("\nDone!")