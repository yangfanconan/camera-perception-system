"""
行为分析模块

识别和分析人体动作：
1. 基础动作：站立、坐着、躺着、行走、跑步
2. 交互动作：挥手、举手、拍手
3. 异常行为：打架、推搡、攀爬
4. 动作序列分析
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from loguru import logger
import time
import math


@dataclass
class ActionResult:
    """动作识别结果"""
    action: str              # 动作名称
    confidence: float        # 置信度
    duration: float          # 持续时间
    track_id: int            # 跟踪ID
    metadata: Dict           # 额外信息
    timestamp: float         # 时间戳
    
    def to_dict(self) -> Dict:
        return {
            'action': self.action,
            'confidence': round(self.confidence, 3),
            'duration': round(self.duration, 2),
            'track_id': self.track_id,
            'timestamp': self.timestamp
        }


@dataclass
class PoseSequence:
    """姿态序列"""
    track_id: int
    poses: deque = field(default_factory=lambda: deque(maxlen=90))  # 3秒 @ 30fps
    timestamps: deque = field(default_factory=lambda: deque(maxlen=90))
    
    def add_pose(self, keypoints: Dict[str, List[float]], timestamp: float):
        """添加姿态"""
        self.poses.append(keypoints)
        self.timestamps.append(timestamp)
    
    def get_duration(self) -> float:
        """获取序列时长"""
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]


class ActionRecognizer:
    """
    动作识别器
    
    基于关键点序列识别动作
    """
    
    # 动作类型
    ACTIONS = {
        # 基础动作
        'standing': '🧍 站立',
        'sitting': '🪑 坐着',
        'lying': '🛏️ 躺着',
        'walking': '🚶 行走',
        'running': '🏃 跑步',
        'jumping': '🦘 跳跃',
        
        # 手势动作
        'waving': '👋 挥手',
        'raising_hand': '🙋 举手',
        'clapping': '👏 拍手',
        
        # 异常行为
        'fighting': '👊 打架',
        'pushing': '🤼 推搡',
        'climbing': '🧗 攀爬',
        'falling': '⬇️ 跌倒',
        
        # 其他
        'unknown': '❓ 未知'
    }
    
    def __init__(self, fps: float = 30.0):
        """
        初始化动作识别器
        
        Args:
            fps: 视频帧率
        """
        self.fps = fps
        
        # 姿态序列存储
        self.pose_sequences: Dict[int, PoseSequence] = {}
        
        # 动作历史
        self.action_history: Dict[int, deque] = {}
        self.history_size = 30
        
        # 动作持续时间
        self.action_start_time: Dict[int, Tuple[str, float]] = {}
        
        logger.info(f"ActionRecognizer initialized (fps={fps})")
    
    def update(
        self,
        track_id: int,
        keypoints: Dict[str, List[float]],
        bbox: List[int] = None
    ) -> ActionResult:
        """
        更新并识别动作
        
        Args:
            track_id: 跟踪ID
            keypoints: 关键点
            bbox: 边界框
            
        Returns:
            ActionResult: 识别结果
        """
        current_time = time.time()
        
        # 获取或创建姿态序列
        if track_id not in self.pose_sequences:
            self.pose_sequences[track_id] = PoseSequence(track_id=track_id)
        
        sequence = self.pose_sequences[track_id]
        sequence.add_pose(keypoints, current_time)
        
        # 识别动作
        action, confidence = self._recognize_action(sequence, bbox)
        
        # 更新动作历史
        if track_id not in self.action_history:
            self.action_history[track_id] = deque(maxlen=self.history_size)
        self.action_history[track_id].append(action)
        
        # 计算持续时间
        duration = 0.0
        if track_id in self.action_start_time:
            prev_action, start_time = self.action_start_time[track_id]
            if prev_action == action:
                duration = current_time - start_time
            else:
                self.action_start_time[track_id] = (action, current_time)
        else:
            self.action_start_time[track_id] = (action, current_time)
        
        return ActionResult(
            action=action,
            confidence=confidence,
            duration=duration,
            track_id=track_id,
            metadata={},
            timestamp=current_time
        )
    
    def _recognize_action(
        self,
        sequence: PoseSequence,
        bbox: List[int] = None
    ) -> Tuple[str, float]:
        """
        识别动作
        
        Args:
            sequence: 姿态序列
            bbox: 边界框
            
        Returns:
            (action_name, confidence)
        """
        if len(sequence.poses) < 5:
            return 'unknown', 0.5
        
        # 获取最新姿态
        current_pose = sequence.poses[-1]
        
        # 1. 检测基础姿态
        base_action, base_conf = self._detect_base_pose(current_pose, bbox)
        
        # 2. 检测运动动作
        motion_action, motion_conf = self._detect_motion(sequence)
        
        # 3. 检测手势动作
        gesture_action, gesture_conf = self._detect_gesture_action(sequence)
        
        # 综合判断
        actions = [
            (base_action, base_conf, 0.4),
            (motion_action, motion_conf, 0.4),
            (gesture_action, gesture_conf, 0.2)
        ]
        
        # 加权选择
        best_action = 'unknown'
        best_score = 0.0
        
        for action, conf, weight in actions:
            score = conf * weight
            if score > best_score and action != 'unknown':
                best_score = score
                best_action = action
        
        return best_action, min(best_score + 0.3, 1.0)
    
    def _detect_base_pose(
        self,
        keypoints: Dict[str, List[float]],
        bbox: List[int] = None
    ) -> Tuple[str, float]:
        """检测基础姿态"""
        
        # 获取关键点
        l_shoulder = keypoints.get('L_shoulder')
        r_shoulder = keypoints.get('R_shoulder')
        l_hip = keypoints.get('L_hip')
        r_hip = keypoints.get('R_hip')
        l_knee = keypoints.get('L_knee')
        r_knee = keypoints.get('R_knee')
        l_ankle = keypoints.get('L_ankle')
        r_ankle = keypoints.get('R_ankle')
        
        if not all([l_shoulder, r_shoulder, l_hip, r_hip]):
            return 'unknown', 0.5
        
        # 计算躯干角度
        shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
        hip_y = (l_hip[1] + r_hip[1]) / 2
        torso_height = abs(hip_y - shoulder_y)
        
        # 计算躯干宽度
        shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
        
        # 宽高比
        aspect_ratio = shoulder_width / max(torso_height, 1)
        
        # 判断姿态
        if l_knee and r_knee and l_ankle and r_ankle:
            knee_y = (l_knee[1] + r_knee[1]) / 2
            ankle_y = (l_ankle[1] + r_ankle[1]) / 2
            leg_height = abs(ankle_y - knee_y)
            
            # 躺着：躯干接近水平
            if aspect_ratio > 1.5:
                return 'lying', 0.8
            
            # 坐着：腿弯曲
            if leg_height < torso_height * 0.5:
                return 'sitting', 0.7
        
        # 站立：躯干垂直
        if aspect_ratio < 0.8:
            return 'standing', 0.8
        
        return 'unknown', 0.5
    
    def _detect_motion(self, sequence: PoseSequence) -> Tuple[str, float]:
        """检测运动动作"""
        if len(sequence.poses) < 15:
            return 'unknown', 0.5
        
        # 计算关键点移动速度
        poses = list(sequence.poses)
        timestamps = list(sequence.timestamps)
        
        # 计算躯干中心移动
        centers = []
        for pose in poses[-15:]:
            l_hip = pose.get('L_hip')
            r_hip = pose.get('R_hip')
            if l_hip and r_hip:
                cx = (l_hip[0] + r_hip[0]) / 2
                cy = (l_hip[1] + r_hip[1]) / 2
                centers.append((cx, cy))
        
        if len(centers) < 10:
            return 'unknown', 0.5
        
        # 计算速度
        speeds = []
        for i in range(1, len(centers)):
            dx = centers[i][0] - centers[i-1][0]
            dy = centers[i][1] - centers[i-1][1]
            speed = np.sqrt(dx**2 + dy**2)
            speeds.append(speed)
        
        avg_speed = np.mean(speeds)
        speed_variance = np.var(speeds)
        
        # 判断运动类型
        if avg_speed > 20:  # 高速移动
            return 'running', 0.8
        elif avg_speed > 5:  # 中速移动
            # 检查是否有节奏（行走）
            if speed_variance > 10:
                return 'walking', 0.7
            return 'walking', 0.6
        elif avg_speed > 2:  # 轻微移动
            return 'standing', 0.6
        
        return 'unknown', 0.5
    
    def _detect_gesture_action(self, sequence: PoseSequence) -> Tuple[str, float]:
        """检测手势动作"""
        if len(sequence.poses) < 10:
            return 'unknown', 0.5
        
        poses = list(sequence.poses[-10:])
        
        # 检测挥手
        wave_score = self._detect_waving(poses)
        if wave_score > 0.6:
            return 'waving', wave_score
        
        # 检测举手
        raise_score = self._detect_raising_hand(poses[-1])
        if raise_score > 0.6:
            return 'raising_hand', raise_score
        
        # 检测拍手
        clap_score = self._detect_clapping(poses)
        if clap_score > 0.6:
            return 'clapping', clap_score
        
        return 'unknown', 0.5
    
    def _detect_waving(self, poses: List[Dict]) -> float:
        """检测挥手"""
        # 获取手腕位置序列
        l_wrist_positions = []
        r_wrist_positions = []
        
        for pose in poses:
            l_wrist = pose.get('L_wrist')
            r_wrist = pose.get('R_wrist')
            
            if l_wrist:
                l_wrist_positions.append(l_wrist[0])
            if r_wrist:
                r_wrist_positions.append(r_wrist[0])
        
        # 检测左右摆动
        for positions in [l_wrist_positions, r_wrist_positions]:
            if len(positions) >= 5:
                # 计算方向变化次数
                changes = 0
                for i in range(2, len(positions)):
                    if (positions[i] - positions[i-1]) * (positions[i-1] - positions[i-2]) < 0:
                        changes += 1
                
                if changes >= 2:
                    return 0.7
        
        return 0.0
    
    def _detect_raising_hand(self, pose: Dict) -> float:
        """检测举手"""
        l_shoulder = pose.get('L_shoulder')
        r_shoulder = pose.get('R_shoulder')
        l_wrist = pose.get('L_wrist')
        r_wrist = pose.get('R_wrist')
        
        if not all([l_shoulder, r_shoulder]):
            return 0.0
        
        shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
        
        # 检查手腕是否高于肩膀
        if l_wrist and l_wrist[1] < shoulder_y:
            return 0.8
        if r_wrist and r_wrist[1] < shoulder_y:
            return 0.8
        
        return 0.0
    
    def _detect_clapping(self, poses: List[Dict]) -> float:
        """检测拍手"""
        # 获取最近几帧的手腕位置
        for pose in poses[-3:]:
            l_wrist = pose.get('L_wrist')
            r_wrist = pose.get('R_wrist')
            
            if l_wrist and r_wrist:
                # 计算双手距离
                distance = np.sqrt(
                    (l_wrist[0] - r_wrist[0]) ** 2 +
                    (l_wrist[1] - r_wrist[1]) ** 2
                )
                
                # 双手很近
                if distance < 50:
                    return 0.7
        
        return 0.0
    
    def get_action_name(self, action: str) -> str:
        """获取动作显示名称"""
        return self.ACTIONS.get(action, action)
    
    def get_current_action(self, track_id: int) -> Optional[str]:
        """获取当前动作"""
        if track_id in self.action_history:
            history = list(self.action_history[track_id])
            if history:
                # 返回最常见的动作
                from collections import Counter
                counts = Counter(history)
                return counts.most_common(1)[0][0]
        return None
    
    def get_action_stats(self, track_id: int) -> Dict[str, int]:
        """获取动作统计"""
        if track_id not in self.action_history:
            return {}
        
        from collections import Counter
        return dict(Counter(self.action_history[track_id]))
    
    def clear_track(self, track_id: int):
        """清除跟踪数据"""
        if track_id in self.pose_sequences:
            del self.pose_sequences[track_id]
        if track_id in self.action_history:
            del self.action_history[track_id]
        if track_id in self.action_start_time:
            del self.action_start_time[track_id]


class BehaviorAnalyzer:
    """
    行为分析器
    
    分析异常行为和交互
    """
    
    def __init__(self):
        """初始化行为分析器"""
        self.interaction_history: deque = deque(maxlen=100)
        self.abnormal_events: List[Dict] = []
        
        logger.info("BehaviorAnalyzer initialized")
    
    def analyze_interaction(
        self,
        tracks: List[Dict],
        current_time: float = None
    ) -> List[Dict]:
        """
        分析人与人之间的交互
        
        Args:
            tracks: 跟踪数据列表
            current_time: 当前时间
            
        Returns:
            交互事件列表
        """
        if current_time is None:
            current_time = time.time()
        
        interactions = []
        
        # 检测近距离交互
        for i, track1 in enumerate(tracks):
            for j, track2 in enumerate(tracks):
                if i >= j:
                    continue
                
                # 计算距离
                pos1 = track1.get('position', (0, 0))
                pos2 = track2.get('position', (0, 0))
                
                distance = np.sqrt(
                    (pos1[0] - pos2[0]) ** 2 +
                    (pos1[1] - pos2[1]) ** 2
                )
                
                # 近距离交互
                if distance < 100:  # 像素
                    interaction = {
                        'type': 'close_proximity',
                        'track_ids': [track1['track_id'], track2['track_id']],
                        'distance': distance,
                        'timestamp': current_time
                    }
                    interactions.append(interaction)
                    
                    self.interaction_history.append(interaction)
        
        return interactions
    
    def detect_abnormal_behavior(
        self,
        tracks: List[Dict],
        actions: Dict[int, str]
    ) -> List[Dict]:
        """
        检测异常行为
        
        Args:
            tracks: 跟踪数据
            actions: 动作字典
            
        Returns:
            异常事件列表
        """
        events = []
        current_time = time.time()
        
        for track in tracks:
            track_id = track.get('track_id')
            action = actions.get(track_id, 'unknown')
            
            # 检测打架
            if action == 'fighting':
                event = {
                    'type': 'fighting',
                    'track_id': track_id,
                    'severity': 'high',
                    'timestamp': current_time
                }
                events.append(event)
                self.abnormal_events.append(event)
            
            # 检测攀爬
            elif action == 'climbing':
                event = {
                    'type': 'climbing',
                    'track_id': track_id,
                    'severity': 'medium',
                    'timestamp': current_time
                }
                events.append(event)
                self.abnormal_events.append(event)
        
        return events
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_interactions': len(self.interaction_history),
            'total_abnormal_events': len(self.abnormal_events)
        }


# 全局实例
_action_recognizer = None
_behavior_analyzer = None

def get_action_recognizer(fps: float = 30.0) -> ActionRecognizer:
    """获取动作识别器单例"""
    global _action_recognizer
    if _action_recognizer is None:
        _action_recognizer = ActionRecognizer(fps=fps)
    return _action_recognizer

def get_behavior_analyzer() -> BehaviorAnalyzer:
    """获取行为分析器单例"""
    global _behavior_analyzer
    if _behavior_analyzer is None:
        _behavior_analyzer = BehaviorAnalyzer()
    return _behavior_analyzer


# 测试代码
if __name__ == '__main__':
    print("Testing Action Recognizer...")
    
    recognizer = ActionRecognizer(fps=30.0)
    
    # 模拟站立姿态
    standing_pose = {
        'L_shoulder': [150, 150],
        'R_shoulder': [200, 150],
        'L_hip': [160, 280],
        'R_hip': [190, 280],
        'L_knee': [160, 380],
        'R_knee': [190, 380],
        'L_ankle': [160, 450],
        'R_ankle': [190, 450],
        'L_wrist': [120, 200],
        'R_wrist': [230, 200]
    }
    
    # 更新多次
    for i in range(20):
        result = recognizer.update(1, standing_pose)
    
    print(f"Action: {result.action}")
    print(f"Display: {recognizer.get_action_name(result.action)}")
    print(f"Confidence: {result.confidence}")
    
    print("\nDone!")