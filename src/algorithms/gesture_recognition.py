"""
手势识别模块

识别常见手势：
1. 竖起大拇指 👍
2. 胜利手势 ✌️
3. OK手势 👌
4. 握拳 ✊
5. 张开手掌 🖐️
6. 指向 👆
7. 数字手势 (1-5)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from loguru import logger
import time


@dataclass
class GestureResult:
    """手势识别结果"""
    gesture: str           # 手势名称
    confidence: float      # 置信度
    hand: str              # 左手/右手
    landmarks: Dict        # 关键点信息
    timestamp: float       # 时间戳
    
    def to_dict(self) -> Dict:
        return {
            'gesture': self.gesture,
            'confidence': round(self.confidence, 3),
            'hand': self.hand,
            'timestamp': self.timestamp
        }


class GestureRecognizer:
    """
    手势识别器
    
    基于 MediaPipe 手部关键点识别手势
    """
    
    # MediaPipe 手部关键点索引
    # 0: WRIST
    # 1-4: THUMB (CMC, MCP, IP, TIP)
    # 5-8: INDEX (MCP, PIP, DIP, TIP)
    # 9-12: MIDDLE (MCP, PIP, DIP, TIP)
    # 13-16: RING (MCP, PIP, DIP, TIP)
    # 17-20: PINKY (MCP, PIP, DIP, TIP)
    
    WRIST = 0
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
    RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20
    
    # 手势定义
    GESTURES = {
        'thumbs_up': '👍 竖起大拇指',
        'thumbs_down': '👎 大拇指朝下',
        'victory': '✌️ 胜利手势',
        'ok': '👌 OK手势',
        'fist': '✊ 握拳',
        'open_palm': '🖐️ 张开手掌',
        'pointing': '👆 指向',
        'one': '☝️ 数字1',
        'two': '✌️ 数字2',
        'three': '3️⃣ 数字3',
        'four': '4️⃣ 数字4',
        'five': '🖐️ 数字5',
        'rock': '🤘 摇滚手势',
        'call_me': '🤙 打电话',
        'unknown': '❓ 未知手势'
    }
    
    def __init__(self):
        """初始化手势识别器"""
        self.gesture_history: List[GestureResult] = []
        self.max_history = 30
        self.smoothing_window = 5
        
        logger.info("GestureRecognizer initialized")
    
    def recognize(self, landmarks: List[List[float]], hand_label: str = "Right") -> GestureResult:
        """
        识别手势
        
        Args:
            landmarks: 21个关键点，每个点 [x, y, z, visibility]
            hand_label: "Left" 或 "Right"
            
        Returns:
            GestureResult: 识别结果
        """
        if not landmarks or len(landmarks) < 21:
            return GestureResult(
                gesture='unknown',
                confidence=0.0,
                hand=hand_label,
                landmarks={},
                timestamp=time.time()
            )
        
        # 提取关键点
        points = np.array(landmarks)
        
        # 计算手指状态
        finger_states = self._get_finger_states(points, hand_label)
        
        # 识别手势
        gesture, confidence = self._classify_gesture(finger_states, points)
        
        result = GestureResult(
            gesture=gesture,
            confidence=confidence,
            hand=hand_label,
            landmarks={'finger_states': finger_states},
            timestamp=time.time()
        )
        
        # 添加到历史
        self.gesture_history.append(result)
        if len(self.gesture_history) > self.max_history:
            self.gesture_history.pop(0)
        
        return result
    
    def _get_finger_states(self, points: np.ndarray, hand_label: str) -> Dict[str, bool]:
        """
        获取各手指的伸展状态
        
        Returns:
            dict: {finger_name: is_extended}
        """
        states = {}
        
        # 大拇指 - 特殊处理（水平方向）
        thumb_tip = points[self.THUMB_TIP]
        thumb_ip = points[self.THUMB_IP]
        thumb_mcp = points[self.THUMB_MCP]
        
        # 根据左右手判断大拇指方向
        if hand_label == "Right":
            states['thumb'] = thumb_tip[0] < thumb_ip[0]  # 右手大拇指在左边
        else:
            states['thumb'] = thumb_tip[0] > thumb_ip[0]  # 左手大拇指在右边
        
        # 其他手指 - 垂直方向
        fingers = [
            ('index', self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP),
            ('middle', self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP),
            ('ring', self.RING_TIP, self.RING_PIP, self.RING_MCP),
            ('pinky', self.PINKY_TIP, self.PINKY_PIP, self.PINKY_MCP)
        ]
        
        for name, tip_idx, pip_idx, mcp_idx in fingers:
            tip = points[tip_idx]
            pip = points[pip_idx]
            mcp = points[mcp_idx]
            
            # 手指伸展：指尖高于PIP关节（y值更小）
            # 同时考虑弯曲程度
            is_extended = tip[1] < pip[1] and pip[1] < mcp[1] + 0.1
            states[name] = is_extended
        
        return states
    
    def _classify_gesture(self, finger_states: Dict[str, bool], points: np.ndarray) -> Tuple[str, float]:
        """
        根据手指状态分类手势
        
        Args:
            finger_states: 手指状态
            points: 关键点
            
        Returns:
            (gesture_name, confidence)
        """
        thumb = finger_states.get('thumb', False)
        index = finger_states.get('index', False)
        middle = finger_states.get('middle', False)
        ring = finger_states.get('ring', False)
        pinky = finger_states.get('pinky', False)
        
        # 计算伸展的手指数量
        extended_count = sum([thumb, index, middle, ring, pinky])
        
        # 张开手掌 (5个手指都伸展)
        if all([thumb, index, middle, ring, pinky]):
            return 'open_palm', 0.95
        
        # 握拳 (所有手指都弯曲)
        if not any([index, middle, ring, pinky]):
            if thumb:
                return 'thumbs_up', 0.9
            return 'fist', 0.9
        
        # 胜利手势 / 数字2 (食指和中指伸展)
        if index and middle and not ring and not pinky:
            return 'victory', 0.9
        
        # 数字1 (只有食指伸展)
        if index and not middle and not ring and not pinky:
            # 检查是否是指向手势（食指指向前方）
            return 'one', 0.85
        
        # 数字3 (食指、中指、无名指伸展)
        if index and middle and ring and not pinky:
            return 'three', 0.85
        
        # 数字4 (除大拇指外都伸展)
        if not thumb and index and middle and ring and pinky:
            return 'four', 0.85
        
        # OK手势 (大拇指和食指形成圆圈)
        if self._check_ok_gesture(points):
            return 'ok', 0.85
        
        # 摇滚手势 (食指和小指伸展)
        if index and not middle and not ring and pinky:
            return 'rock', 0.8
        
        # 打电话手势 (大拇指和小指伸展)
        if thumb and not index and not middle and not ring and pinky:
            return 'call_me', 0.8
        
        # 指向 (只有食指伸展，且大拇指弯曲)
        if index and not middle and not ring and not pinky and not thumb:
            return 'pointing', 0.75
        
        # 大拇指朝下
        if thumb and not any([index, middle, ring, pinky]):
            # 检查大拇指方向
            thumb_tip = points[self.THUMB_TIP]
            wrist = points[self.WRIST]
            if thumb_tip[1] > wrist[1]:  # 大拇指在手腕下方
                return 'thumbs_down', 0.8
            return 'thumbs_up', 0.8
        
        # 未知手势
        return 'unknown', 0.5
    
    def _check_ok_gesture(self, points: np.ndarray) -> bool:
        """检查是否是OK手势"""
        thumb_tip = points[self.THUMB_TIP]
        index_tip = points[self.INDEX_TIP]
        
        # 大拇指和食指尖端的距离
        distance = np.sqrt(
            (thumb_tip[0] - index_tip[0]) ** 2 +
            (thumb_tip[1] - index_tip[1]) ** 2
        )
        
        # 如果距离很小，且其他手指伸展
        if distance < 0.1:  # 阈值
            middle = points[self.MIDDLE_TIP]
            ring = points[self.RING_TIP]
            pinky = points[self.PINKY_TIP]
            mcp = points[self.MIDDLE_MCP]
            
            # 其他手指伸展
            other_extended = (
                middle[1] < mcp[1] and
                ring[1] < mcp[1] and
                pinky[1] < mcp[1]
            )
            
            return other_extended
        
        return False
    
    def get_smoothed_gesture(self, window: int = None) -> Optional[str]:
        """
        获取平滑后的手势（基于历史记录）
        
        Args:
            window: 平滑窗口大小
            
        Returns:
            最常见的手势
        """
        if not self.gesture_history:
            return None
        
        window = window or self.smoothing_window
        recent = self.gesture_history[-window:]
        
        # 统计各手势出现次数
        gesture_counts = {}
        for result in recent:
            if result.gesture not in gesture_counts:
                gesture_counts[result.gesture] = 0
            gesture_counts[result.gesture] += 1
        
        # 返回最常见的手势
        if gesture_counts:
            return max(gesture_counts, key=gesture_counts.get)
        
        return None
    
    def get_gesture_name(self, gesture: str) -> str:
        """获取手势的显示名称"""
        return self.GESTURES.get(gesture, gesture)


class GestureActionMapper:
    """
    手势动作映射器
    
    将手势映射到具体动作
    """
    
    # 默认动作映射
    DEFAULT_ACTIONS = {
        'thumbs_up': 'confirm',
        'thumbs_down': 'cancel',
        'victory': 'peace',
        'ok': 'accept',
        'fist': 'grab',
        'open_palm': 'release',
        'pointing': 'select',
        'one': 'select_1',
        'two': 'select_2',
        'three': 'select_3',
        'four': 'select_4',
        'five': 'select_5',
    }
    
    def __init__(self, action_map: Dict[str, str] = None):
        """
        初始化映射器
        
        Args:
            action_map: 自定义动作映射
        """
        self.action_map = action_map or self.DEFAULT_ACTIONS.copy()
        self.gesture_cooldowns: Dict[str, float] = {}
        self.cooldown_time = 1.0  # 秒
    
    def map_gesture_to_action(self, gesture: str) -> Optional[str]:
        """
        将手势映射到动作
        
        Args:
            gesture: 手势名称
            
        Returns:
            动作名称
        """
        # 检查冷却时间
        if gesture in self.gesture_cooldowns:
            if time.time() - self.gesture_cooldowns[gesture] < self.cooldown_time:
                return None
        
        action = self.action_map.get(gesture)
        if action:
            self.gesture_cooldowns[gesture] = time.time()
        
        return action
    
    def set_cooldown(self, cooldown_time: float):
        """设置冷却时间"""
        self.cooldown_time = cooldown_time


# 全局识别器实例
_gesture_recognizer = None

def get_gesture_recognizer() -> GestureRecognizer:
    """获取手势识别器单例"""
    global _gesture_recognizer
    if _gesture_recognizer is None:
        _gesture_recognizer = GestureRecognizer()
    return _gesture_recognizer


# 测试代码
if __name__ == '__main__':
    print("Testing Gesture Recognizer...")
    
    recognizer = GestureRecognizer()
    
    # 模拟手势关键点
    # 张开手掌
    open_palm_landmarks = [
        [0.5, 0.8, 0, 1],  # wrist
        [0.4, 0.7, 0, 1],  # thumb_cmc
        [0.35, 0.6, 0, 1],  # thumb_mcp
        [0.3, 0.5, 0, 1],  # thumb_ip
        [0.25, 0.4, 0, 1],  # thumb_tip
        [0.45, 0.5, 0, 1],  # index_mcp
        [0.45, 0.35, 0, 1],  # index_pip
        [0.45, 0.25, 0, 1],  # index_dip
        [0.45, 0.15, 0, 1],  # index_tip
        [0.5, 0.5, 0, 1],  # middle_mcp
        [0.5, 0.35, 0, 1],  # middle_pip
        [0.5, 0.25, 0, 1],  # middle_dip
        [0.5, 0.15, 0, 1],  # middle_tip
        [0.55, 0.5, 0, 1],  # ring_mcp
        [0.55, 0.35, 0, 1],  # ring_pip
        [0.55, 0.25, 0, 1],  # ring_dip
        [0.55, 0.15, 0, 1],  # ring_tip
        [0.6, 0.5, 0, 1],  # pinky_mcp
        [0.6, 0.4, 0, 1],  # pinky_pip
        [0.6, 0.3, 0, 1],  # pinky_dip
        [0.6, 0.2, 0, 1],  # pinky_tip
    ]
    
    result = recognizer.recognize(open_palm_landmarks, "Right")
    print(f"Gesture: {result.gesture}")
    print(f"Display: {recognizer.get_gesture_name(result.gesture)}")
    print(f"Confidence: {result.confidence}")
    
    print("\nDone!")