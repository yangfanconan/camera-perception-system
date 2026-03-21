"""
跌倒检测模块

检测人体跌倒事件，用于安全监控。

检测方法：
1. 姿态分析：检测人体是否处于异常姿态
2. 运动分析：检测快速下降运动
3. 位置分析：检测人体是否在地面上
4. 关键点分析：检测关键点位置异常
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from loguru import logger
import time
import math


@dataclass
class FallEvent:
    """跌倒事件"""
    timestamp: float
    confidence: float
    duration: float           # 跌倒持续时间
    position: Tuple[float, float]  # 跌倒位置
    state: str                # falling, fallen, recovered
    keypoints: Dict           # 关键点信息
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'confidence': round(self.confidence, 3),
            'duration': round(self.duration, 2),
            'position': self.position,
            'state': self.state
        }


@dataclass
class PersonState:
    """人体状态跟踪"""
    track_id: int
    positions: deque = field(default_factory=lambda: deque(maxlen=30))
    velocities: deque = field(default_factory=lambda: deque(maxlen=30))
    heights: deque = field(default_factory=lambda: deque(maxlen=30))
    aspect_ratios: deque = field(default_factory=lambda: deque(maxlen=30))
    last_fall_time: float = 0
    fall_state: str = "normal"  # normal, falling, fallen, recovered


class FallDetector:
    """
    跌倒检测器
    
    基于姿态和运动分析检测跌倒
    """
    
    # 跌倒检测阈值
    FALL_VELOCITY_THRESHOLD = 0.3      # 快速下降速度阈值 (像素/帧)
    FALL_HEIGHT_RATIO = 0.5            # 高度变化比例阈值
    FALL_ASPECT_RATIO = 1.5            # 宽高比阈值（跌倒后宽度大于高度）
    FALL_DURATION_THRESHOLD = 1.0      # 跌倒持续时间阈值（秒）
    RECOVERY_TIME_THRESHOLD = 5.0      # 恢复时间阈值（秒）
    GROUND_LEVEL_THRESHOLD = 0.8       # 地面高度阈值（画面底部比例）
    
    def __init__(self, fps: float = 30.0):
        """
        初始化跌倒检测器
        
        Args:
            fps: 视频帧率
        """
        self.fps = fps
        self.dt = 1.0 / fps
        
        # 人体状态跟踪
        self.person_states: Dict[int, PersonState] = {}
        
        # 跌倒事件记录
        self.fall_events: List[FallEvent] = []
        
        # 报警回调
        self.on_fall_detected = None
        self.on_recovery_detected = None
        
        logger.info(f"FallDetector initialized (fps={fps})")
    
    def update(
        self,
        track_id: int,
        bbox: List[int],
        keypoints: Dict[str, List[float]],
        image_height: int = 1080,
        image_width: int = 1920
    ) -> Optional[FallEvent]:
        """
        更新人体状态并检测跌倒
        
        Args:
            track_id: 跟踪ID
            bbox: 边界框 [x, y, w, h]
            keypoints: 关键点
            image_height: 图像高度
            image_width: 图像宽度
            
        Returns:
            FallEvent: 如果检测到跌倒，返回事件
        """
        # 获取或创建状态
        if track_id not in self.person_states:
            self.person_states[track_id] = PersonState(track_id=track_id)
        
        state = self.person_states[track_id]
        
        # 计算当前位置和属性
        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2
        
        # 归一化位置
        norm_x = cx / image_width
        norm_y = cy / image_height
        
        # 高度和宽高比
        height = h
        aspect_ratio = w / max(h, 1)
        
        # 更新状态
        current_time = time.time()
        state.positions.append((norm_x, norm_y, current_time))
        state.heights.append(height)
        state.aspect_ratios.append(aspect_ratio)
        
        # 计算速度
        if len(state.positions) >= 2:
            prev_x, prev_y, prev_t = state.positions[-2]
            dt = current_time - prev_t
            if dt > 0:
                vx = (norm_x - prev_x) / dt
                vy = (norm_y - prev_y) / dt
                state.velocities.append((vx, vy, current_time))
        
        # 检测跌倒
        fall_event = self._detect_fall(
            state, bbox, keypoints, image_height, image_width
        )
        
        return fall_event
    
    def _detect_fall(
        self,
        state: PersonState,
        bbox: List[int],
        keypoints: Dict[str, List[float]],
        image_height: int,
        image_width: int
    ) -> Optional[FallEvent]:
        """
        检测跌倒
        
        Args:
            state: 人体状态
            bbox: 边界框
            keypoints: 关键点
            image_height: 图像高度
            image_width: 图像宽度
            
        Returns:
            FallEvent: 跌倒事件
        """
        current_time = time.time()
        x, y, w, h = bbox
        
        # 获取当前状态
        fall_indicators = []
        
        # 1. 检测快速下降运动
        if len(state.velocities) >= 3:
            recent_velocities = list(state.velocities)[-5:]
            avg_vy = np.mean([v[1] for v in recent_velocities])
            
            # 快速向下运动
            if avg_vy > self.FALL_VELOCITY_THRESHOLD:
                fall_indicators.append(('velocity', 0.8, avg_vy))
        
        # 2. 检测高度变化
        if len(state.heights) >= 10:
            recent_heights = list(state.heights)[-10:]
            height_change = (recent_heights[0] - recent_heights[-1]) / max(recent_heights[0], 1)
            
            # 高度显著减小
            if height_change > self.FALL_HEIGHT_RATIO:
                fall_indicators.append(('height_change', 0.7, height_change))
        
        # 3. 检测宽高比变化
        if len(state.aspect_ratios) >= 5:
            current_ratio = state.aspect_ratios[-1]
            avg_ratio = np.mean(list(state.aspect_ratios)[:-1])
            
            # 宽高比增大（从站立变为躺下）
            if current_ratio > self.FALL_ASPECT_RATIO and current_ratio > avg_ratio * 1.5:
                fall_indicators.append(('aspect_ratio', 0.6, current_ratio))
        
        # 4. 检测关键点位置
        if keypoints:
            keypoint_fall_score = self._analyze_keypoints(keypoints, image_height)
            if keypoint_fall_score > 0.5:
                fall_indicators.append(('keypoints', keypoint_fall_score, keypoint_fall_score))
        
        # 5. 检测是否在地面
        cy = (y + h / 2) / image_height
        if cy > self.GROUND_LEVEL_THRESHOLD:
            fall_indicators.append(('ground_level', 0.5, cy))
        
        # 综合判断
        if len(fall_indicators) >= 2:
            # 计算综合置信度
            total_confidence = sum(ind[1] for ind in fall_indicators) / len(fall_indicators)
            
            # 状态转换
            if state.fall_state == "normal":
                state.fall_state = "falling"
                state.last_fall_time = current_time
                logger.warning(f"Person {state.track_id} may be falling! Indicators: {[i[0] for i in fall_indicators]}")
            
            elif state.fall_state == "falling":
                # 检查是否持续足够时间
                fall_duration = current_time - state.last_fall_time
                if fall_duration > self.FALL_DURATION_THRESHOLD:
                    state.fall_state = "fallen"
                    
                    # 创建跌倒事件
                    event = FallEvent(
                        timestamp=current_time,
                        confidence=total_confidence,
                        duration=fall_duration,
                        position=(x / image_width, y / image_height),
                        state="fallen",
                        keypoints=keypoints
                    )
                    
                    self.fall_events.append(event)
                    
                    # 触发报警
                    if self.on_fall_detected:
                        self.on_fall_detected(event)
                    
                    logger.warning(f"FALL DETECTED! Person {state.track_id}, confidence={total_confidence:.2f}")
                    
                    return event
        
        elif state.fall_state == "fallen":
            # 检查是否恢复
            if len(fall_indicators) == 0:
                recovery_time = current_time - state.last_fall_time
                if recovery_time > self.RECOVERY_TIME_THRESHOLD:
                    state.fall_state = "recovered"
                    logger.info(f"Person {state.track_id} recovered from fall")
                    
                    if self.on_recovery_detected:
                        self.on_recovery_detected(state.track_id)
        
        elif state.fall_state == "recovered":
            # 重置为正常状态
            if len(fall_indicators) == 0:
                state.fall_state = "normal"
        
        return None
    
    def _analyze_keypoints(self, keypoints: Dict[str, List[float]], image_height: int) -> float:
        """
        分析关键点判断跌倒
        
        Args:
            keypoints: 关键点
            image_height: 图像高度
            
        Returns:
            跌倒可能性分数 (0-1)
        """
        score = 0.0
        
        # 检查肩膀是否在同一水平线上
        l_shoulder = keypoints.get('L_shoulder')
        r_shoulder = keypoints.get('R_shoulder')
        
        if l_shoulder and r_shoulder:
            shoulder_diff = abs(l_shoulder[1] - r_shoulder[1]) / image_height
            # 跌倒时肩膀高度差较大
            if shoulder_diff > 0.1:
                score += 0.3
        
        # 检查臀部是否高于肩膀（倒立）
        l_hip = keypoints.get('L_hip')
        r_hip = keypoints.get('R_hip')
        
        if l_shoulder and r_shoulder and l_hip and r_hip:
            shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
            hip_y = (l_hip[1] + r_hip[1]) / 2
            
            # 臀部高于肩膀
            if hip_y < shoulder_y:
                score += 0.4
        
        # 检查脚踝位置
        l_ankle = keypoints.get('L_ankle')
        r_ankle = keypoints.get('R_ankle')
        
        if l_ankle and r_ankle:
            ankle_y = (l_ankle[1] + r_ankle[1]) / 2
            # 脚踝在画面顶部（倒立）
            if ankle_y < 0.3:
                score += 0.3
        
        return min(score, 1.0)
    
    def get_fall_events(self, since: float = None) -> List[FallEvent]:
        """
        获取跌倒事件
        
        Args:
            since: 起始时间戳
            
        Returns:
            跌倒事件列表
        """
        if since is None:
            return self.fall_events
        
        return [e for e in self.fall_events if e.timestamp >= since]
    
    def get_person_state(self, track_id: int) -> Optional[str]:
        """获取人体跌倒状态"""
        if track_id in self.person_states:
            return self.person_states[track_id].fall_state
        return None
    
    def reset(self):
        """重置检测器"""
        self.person_states.clear()
        self.fall_events.clear()
        logger.info("FallDetector reset")


class FallAlertSystem:
    """
    跌倒报警系统
    
    管理跌倒报警和通知
    """
    
    def __init__(self, cooldown: float = 10.0):
        """
        初始化报警系统
        
        Args:
            cooldown: 报警冷却时间（秒）
        """
        self.cooldown = cooldown
        self.last_alert_time: Dict[int, float] = {}
        self.alert_count = 0
    
    def should_alert(self, track_id: int) -> bool:
        """
        检查是否应该发送报警
        
        Args:
            track_id: 跟踪ID
            
        Returns:
            是否应该报警
        """
        current_time = time.time()
        
        if track_id in self.last_alert_time:
            if current_time - self.last_alert_time[track_id] < self.cooldown:
                return False
        
        self.last_alert_time[track_id] = current_time
        self.alert_count += 1
        return True
    
    def get_stats(self) -> Dict:
        """获取报警统计"""
        return {
            'total_alerts': self.alert_count,
            'active_persons': len(self.last_alert_time)
        }


# 全局检测器实例
_fall_detector = None

def get_fall_detector(fps: float = 30.0) -> FallDetector:
    """获取跌倒检测器单例"""
    global _fall_detector
    if _fall_detector is None:
        _fall_detector = FallDetector(fps=fps)
    return _fall_detector


# 测试代码
if __name__ == '__main__':
    print("Testing Fall Detector...")
    
    detector = FallDetector(fps=30.0)
    
    # 模拟正常站立
    print("\n1. Normal standing:")
    for i in range(10):
        bbox = [100, 100, 100, 300]  # 站立姿态
        keypoints = {
            'L_shoulder': [150, 150],
            'R_shoulder': [200, 150],
            'L_hip': [160, 280],
            'R_hip': [190, 280],
            'L_ankle': [160, 400],
            'R_ankle': [190, 400]
        }
        event = detector.update(1, bbox, keypoints)
        print(f"  Frame {i}: state={detector.get_person_state(1)}")
    
    # 模拟跌倒
    print("\n2. Falling:")
    for i in range(20):
        # 高度逐渐减小，宽度增大
        h = 300 - i * 15
        w = 100 + i * 10
        y = 100 + i * 10
        bbox = [100, y, w, max(h, 50)]
        
        event = detector.update(1, bbox, keypoints)
        state = detector.get_person_state(1)
        print(f"  Frame {i}: state={state}, bbox={bbox[2]}x{bbox[3]}")
        
        if event:
            print(f"  FALL DETECTED! confidence={event.confidence}")
    
    print("\nDone!")