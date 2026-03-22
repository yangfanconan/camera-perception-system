"""
报警系统模块

实现多种报警功能：
1. 越界报警
2. 跌倒报警
3. 逗留报警
4. 人群聚集报警
5. 异常行为报警
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger
import time
import json
from pathlib import Path


@dataclass
class Alert:
    """报警事件"""
    alert_id: int
    alert_type: str           # crossing, fall, loitering, crowd, abnormal
    severity: str             # low, medium, high, critical
    message: str
    track_id: int             # 相关跟踪ID
    position: Tuple[float, float]
    timestamp: float
    acknowledged: bool = False
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type,
            'severity': self.severity,
            'message': self.message,
            'track_id': self.track_id,
            'position': self.position,
            'timestamp': self.timestamp,
            'acknowledged': self.acknowledged
        }


@dataclass
class AlertZone:
    """报警区域"""
    zone_id: str
    name: str
    polygon: List[Tuple[int, int]]  # 多边形顶点
    zone_type: str            # forbidden, restricted, safe
    alert_type: str           # 进入报警类型
    enabled: bool = True
    
    def contains(self, point: Tuple[float, float]) -> bool:
        """检查点是否在区域内"""
        x, y = point
        n = len(self.polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = self.polygon[i]
            xj, yj = self.polygon[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            
            j = i
        
        return inside


class AlertSystem:
    """
    报警系统
    
    管理各种报警事件
    """
    
    SEVERITY_LEVELS = {
        'low': 1,
        'medium': 2,
        'high': 3,
        'critical': 4
    }
    
    ALERT_TYPES = {
        'crossing': '🚧 越界报警',
        'fall': '⚠️ 跌倒报警',
        'loitering': '⏰ 逗留报警',
        'crowd': '👥 人群聚集',
        'abnormal': '❗ 异常行为',
        'intrusion': '🚨 入侵报警',
        'exit': '🚪 离开区域'
    }
    
    def __init__(self, config_file: str = None):
        """
        初始化报警系统
        
        Args:
            config_file: 配置文件路径
        """
        self.alert_zones: Dict[str, AlertZone] = {}
        self.alerts: List[Alert] = []
        self.next_alert_id = 1
        
        # 报警回调
        self.callbacks: List[Callable] = []
        
        # 冷却时间
        self.cooldown: Dict[str, float] = {}  # (zone_id, track_id) -> last_alert_time
        self.default_cooldown = 10.0  # 秒
        
        # 统计
        self.stats = defaultdict(int)
        
        # 加载配置
        if config_file:
            self._load_config(config_file)
        
        logger.info("AlertSystem initialized")
    
    def _load_config(self, config_file: str):
        """加载配置"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            for zone_config in config.get('zones', []):
                zone = AlertZone(
                    zone_id=zone_config['id'],
                    name=zone_config['name'],
                    polygon=[tuple(p) for p in zone_config['polygon']],
                    zone_type=zone_config.get('type', 'restricted'),
                    alert_type=zone_config.get('alert_type', 'intrusion'),
                    enabled=zone_config.get('enabled', True)
                )
                self.alert_zones[zone.zone_id] = zone
            
            logger.info(f"Loaded {len(self.alert_zones)} alert zones")
            
        except Exception as e:
            logger.warning(f"Failed to load alert config: {e}")
    
    def add_zone(
        self,
        zone_id: str,
        name: str,
        polygon: List[Tuple[int, int]],
        zone_type: str = "restricted",
        alert_type: str = "intrusion"
    ):
        """
        添加报警区域
        
        Args:
            zone_id: 区域ID
            name: 区域名称
            polygon: 多边形顶点
            zone_type: 区域类型
            alert_type: 报警类型
        """
        zone = AlertZone(
            zone_id=zone_id,
            name=name,
            polygon=polygon,
            zone_type=zone_type,
            alert_type=alert_type
        )
        self.alert_zones[zone_id] = zone
        logger.info(f"Added alert zone: {name}")
    
    def remove_zone(self, zone_id: str):
        """移除报警区域"""
        if zone_id in self.alert_zones:
            del self.alert_zones[zone_id]
            logger.info(f"Removed alert zone: {zone_id}")
    
    def check_zones(
        self,
        track_id: int,
        position: Tuple[float, float],
        image_width: int = 1920,
        image_height: int = 1080
    ) -> List[Alert]:
        """
        检查区域越界
        
        Args:
            track_id: 跟踪ID
            position: 位置 (x, y)
            image_width: 图像宽度
            image_height: 图像高度
            
        Returns:
            报警列表
        """
        alerts = []
        current_time = time.time()
        
        for zone_id, zone in self.alert_zones.items():
            if not zone.enabled:
                continue
            
            # 检查是否在区域内
            in_zone = zone.contains(position)
            
            # 检查冷却时间
            cooldown_key = f"{zone_id}_{track_id}"
            if cooldown_key in self.cooldown:
                if current_time - self.cooldown[cooldown_key] < self.default_cooldown:
                    continue
            
            # 根据区域类型判断
            if in_zone and zone.zone_type in ['forbidden', 'restricted']:
                # 创建报警
                alert = self._create_alert(
                    alert_type=zone.alert_type,
                    severity='high' if zone.zone_type == 'forbidden' else 'medium',
                    message=f"检测到 {zone.name} 区域{'入侵' if zone.zone_type == 'forbidden' else '进入'}",
                    track_id=track_id,
                    position=position
                )
                
                alerts.append(alert)
                self.cooldown[cooldown_key] = current_time
        
        return alerts
    
    def check_fall(
        self,
        track_id: int,
        fall_state: str,
        position: Tuple[float, float]
    ) -> Optional[Alert]:
        """
        检查跌倒报警
        
        Args:
            track_id: 跟踪ID
            fall_state: 跌倒状态
            position: 位置
            
        Returns:
            报警事件
        """
        if fall_state != "fallen":
            return None
        
        # 检查冷却
        cooldown_key = f"fall_{track_id}"
        current_time = time.time()
        
        if cooldown_key in self.cooldown:
            if current_time - self.cooldown[cooldown_key] < self.default_cooldown:
                return None
        
        alert = self._create_alert(
            alert_type='fall',
            severity='critical',
            message=f"检测到人员跌倒！",
            track_id=track_id,
            position=position
        )
        
        self.cooldown[cooldown_key] = current_time
        return alert
    
    def check_loitering(
        self,
        track_id: int,
        positions: List[Tuple[float, float]],
        duration_threshold: float = 60.0,
        distance_threshold: float = 50.0
    ) -> Optional[Alert]:
        """
        检查逗留报警
        
        Args:
            track_id: 跟踪ID
            positions: 位置历史
            duration_threshold: 逗留时间阈值（秒）
            distance_threshold: 移动距离阈值（像素）
            
        Returns:
            报警事件
        """
        if len(positions) < 2:
            return None
        
        # 计算停留时间
        first_time = positions[0][2] if len(positions[0]) > 2 else time.time() - len(positions) / 30
        last_time = positions[-1][2] if len(positions[-1]) > 2 else time.time()
        duration = last_time - first_time
        
        if duration < duration_threshold:
            return None
        
        # 计算移动距离
        first_pos = (positions[0][0], positions[0][1])
        last_pos = (positions[-1][0], positions[-1][1])
        distance = np.sqrt((last_pos[0] - first_pos[0]) ** 2 + (last_pos[1] - first_pos[1]) ** 2)
        
        if distance > distance_threshold:
            return None
        
        # 检查冷却
        cooldown_key = f"loitering_{track_id}"
        current_time = time.time()
        
        if cooldown_key in self.cooldown:
            if current_time - self.cooldown[cooldown_key] < self.default_cooldown:
                return None
        
        alert = self._create_alert(
            alert_type='loitering',
            severity='medium',
            message=f"检测到人员逗留超过 {duration:.0f} 秒",
            track_id=track_id,
            position=last_pos
        )
        
        self.cooldown[cooldown_key] = current_time
        return alert
    
    def check_crowd(
        self,
        positions: List[Tuple[float, float]],
        threshold: int = 5,
        radius: float = 100.0
    ) -> Optional[Alert]:
        """
        检查人群聚集
        
        Args:
            positions: 所有人位置列表
            threshold: 人数阈值
            radius: 检测半径（像素）
            
        Returns:
            报警事件
        """
        if len(positions) < threshold:
            return None
        
        # 使用简单的密度检测
        for pos in positions:
            # 计算该位置附近的人数
            nearby = 0
            for other in positions:
                distance = np.sqrt((pos[0] - other[0]) ** 2 + (pos[1] - other[1]) ** 2)
                if distance < radius:
                    nearby += 1
            
            if nearby >= threshold:
                # 检查冷却
                cooldown_key = f"crowd_{int(pos[0])}_{int(pos[1])}"
                current_time = time.time()
                
                if cooldown_key in self.cooldown:
                    if current_time - self.cooldown[cooldown_key] < self.default_cooldown:
                        continue
                
                alert = self._create_alert(
                    alert_type='crowd',
                    severity='medium',
                    message=f"检测到人群聚集，约 {nearby} 人",
                    track_id=-1,
                    position=pos
                )
                
                self.cooldown[cooldown_key] = current_time
                return alert
        
        return None
    
    def _create_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        track_id: int,
        position: Tuple[float, float]
    ) -> Alert:
        """创建报警"""
        alert = Alert(
            alert_id=self.next_alert_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            track_id=track_id,
            position=position,
            timestamp=time.time()
        )
        
        self.next_alert_id += 1
        self.alerts.append(alert)
        self.stats[alert_type] += 1
        
        # 触发回调
        for callback in self.callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
        
        logger.warning(f"ALERT [{severity.upper()}] {message}")
        
        return alert
    
    def create_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        track_id: int = None
    ) -> 'Alert':
        """创建报警（公共方法）"""
        return self._create_alert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            track_id=track_id or 0,
            position=(0.0, 0.0)
        )

    def add_callback(self, callback: Callable):
        """添加报警回调"""
        self.callbacks.append(callback)
    
    def acknowledge_alert(self, alert_id: int) -> bool:
        """确认报警"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def get_alerts(
        self,
        since: float = None,
        alert_type: str = None,
        severity: str = None,
        unacknowledged_only: bool = False
    ) -> List[Alert]:
        """
        获取报警列表
        
        Args:
            since: 起始时间
            alert_type: 报警类型
            severity: 严重程度
            unacknowledged_only: 仅未确认
            
        Returns:
            报警列表
        """
        alerts = self.alerts
        
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]
        
        return alerts
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_alerts': len(self.alerts),
            'by_type': dict(self.stats),
            'unacknowledged': len([a for a in self.alerts if not a.acknowledged])
        }
    
    def clear_alerts(self):
        """清除所有报警"""
        self.alerts.clear()
        self.cooldown.clear()
    
    def save_config(self, config_file: str):
        """保存配置"""
        config = {
            'zones': [
                {
                    'id': zone.zone_id,
                    'name': zone.name,
                    'polygon': list(zone.polygon),
                    'type': zone.zone_type,
                    'alert_type': zone.alert_type,
                    'enabled': zone.enabled
                }
                for zone in self.alert_zones.values()
            ]
        }
        
        Path(config_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved alert config to {config_file}")


# 全局报警系统实例
_alert_system = None

def get_alert_system() -> AlertSystem:
    """获取报警系统单例"""
    global _alert_system
    if _alert_system is None:
        _alert_system = AlertSystem()
    return _alert_system


# 测试代码
if __name__ == '__main__':
    print("Testing Alert System...")
    
    system = AlertSystem()
    
    # 添加报警区域
    system.add_zone(
        zone_id="zone1",
        name="禁区A",
        polygon=[(100, 100), (300, 100), (300, 300), (100, 300)],
        zone_type="forbidden",
        alert_type="intrusion"
    )
    
    # 测试越界检测
    print("\n1. Testing zone crossing:")
    alerts = system.check_zones(1, (200, 200))  # 在区域内
    for alert in alerts:
        print(f"  Alert: {alert.message}")
    
    alerts = system.check_zones(2, (50, 50))  # 在区域外
    print(f"  Outside zone: {len(alerts)} alerts")
    
    # 测试跌倒检测
    print("\n2. Testing fall detection:")
    alert = system.check_fall(1, "fallen", (200, 200))
    if alert:
        print(f"  Fall alert: {alert.message}")
    
    # 测试人群聚集
    print("\n3. Testing crowd detection:")
    positions = [(100 + i * 10, 100 + i * 10) for i in range(6)]
    alert = system.check_crowd(positions, threshold=5, radius=100)
    if alert:
        print(f"  Crowd alert: {alert.message}")
    
    # 打印统计
    print("\nStats:", system.get_stats())
    
    print("\nDone!")