"""
智能场景分析模块

功能：
1. 场景理解与描述
2. 场景分类
3. 场景变化检测
4. 场景摘要生成
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from loguru import logger
import time


@dataclass
class SceneObject:
    """场景对象"""
    object_type: str       # person, vehicle, furniture, etc.
    bbox: List[int]
    confidence: float
    attributes: Dict = field(default_factory=dict)
    track_id: Optional[int] = None


@dataclass
class SceneDescription:
    """场景描述"""
    scene_type: str        # indoor, outdoor, office, home, street, etc.
    objects: List[SceneObject]
    activities: List[str]  # walking, sitting, talking, etc.
    crowd_density: float   # 0-1
    scene_summary: str     # 自然语言描述
    timestamp: float
    
    def to_dict(self) -> Dict:
        return {
            'scene_type': self.scene_type,
            'object_count': len(self.objects),
            'activities': self.activities,
            'crowd_density': round(self.crowd_density, 2),
            'summary': self.scene_summary,
            'timestamp': self.timestamp
        }


class SceneClassifier:
    """
    场景分类器
    
    识别场景类型
    """
    
    # 场景类型
    SCENE_TYPES = {
        'indoor': ['office', 'home', 'corridor', 'room', 'lobby', 'store'],
        'outdoor': ['street', 'parking', 'park', 'entrance', 'yard'],
        'special': ['construction', 'restricted', 'emergency']
    }
    
    # 场景特征
    SCENE_FEATURES = {
        'office': {'has_desks': True, 'has_computers': True, 'typical_objects': ['chair', 'desk', 'monitor']},
        'home': {'has_furniture': True, 'typical_objects': ['sofa', 'bed', 'tv']},
        'corridor': {'long_narrow': True, 'typical_objects': ['door', 'exit_sign']},
        'street': {'outdoor': True, 'typical_objects': ['car', 'person', 'traffic_light']},
        'parking': {'outdoor': True, 'typical_objects': ['car', 'parking_line']},
        'entrance': {'has_door': True, 'typical_objects': ['door', 'person']}
    }
    
    def __init__(self):
        """初始化场景分类器"""
        self.scene_history: deque = deque(maxlen=30)
        
        logger.info("SceneClassifier initialized")
    
    def classify(self, frame: np.ndarray, objects: List[SceneObject]) -> str:
        """
        分类场景
        
        Args:
            frame: 图像帧
            objects: 检测到的对象
            
        Returns:
            场景类型
        """
        # 基于对象推断场景
        object_types = [obj.object_type for obj in objects]
        
        # 计算场景得分
        scores = defaultdict(float)
        
        # 室内/室外判断
        person_count = object_types.count('person')
        vehicle_count = object_types.count('vehicle') + object_types.count('car')
        
        if vehicle_count > person_count:
            scores['outdoor'] += 0.5
            scores['street'] += 0.3
            scores['parking'] += 0.2
        elif person_count > 0:
            scores['indoor'] += 0.3
        
        # 基于图像特征
        if frame is not None:
            # 分析颜色分布
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 室外通常有更多天空（蓝色）和绿色
            blue_ratio = np.sum((hsv[:, :, 0] >= 100) & (hsv[:, :, 0] <= 130)) / frame.size
            green_ratio = np.sum((hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85)) / frame.size
            
            if blue_ratio > 0.1 or green_ratio > 0.1:
                scores['outdoor'] += 0.3
            else:
                scores['indoor'] += 0.3
        
        # 返回得分最高的场景
        if scores:
            best_scene = max(scores, key=scores.get)
            
            # 细化场景类型
            if best_scene == 'outdoor':
                if vehicle_count > 2:
                    return 'parking' if 'parking' in scores else 'street'
                return 'street'
            elif best_scene == 'indoor':
                if person_count > 5:
                    return 'office'
                return 'room'
            
            return best_scene
        
        return 'unknown'
    
    def get_scene_type_category(self, scene_type: str) -> str:
        """获取场景大类"""
        for category, types in self.SCENE_TYPES.items():
            if scene_type in types:
                return category
        return 'unknown'


class ActivityAnalyzer:
    """
    活动分析器
    
    分析场景中的活动
    """
    
    # 活动类型
    ACTIVITIES = {
        'movement': ['walking', 'running', 'standing'],
        'interaction': ['talking', 'handshake', 'fighting'],
        'work': ['typing', 'reading', 'meeting'],
        'leisure': ['sitting', 'resting', 'eating']
    }
    
    def __init__(self):
        """初始化活动分析器"""
        self.activity_history: deque = deque(maxlen=100)
        
        logger.info("ActivityAnalyzer initialized")
    
    def analyze(
        self,
        persons: List[Dict],
        actions: Dict[int, str]
    ) -> List[str]:
        """
        分析活动
        
        Args:
            persons: 人员列表
            actions: 动作字典
            
        Returns:
            活动列表
        """
        activities = set()
        
        for person in persons:
            track_id = person.get('track_id')
            if track_id in actions:
                action = actions[track_id]
                activities.add(action)
        
        # 分析群体活动
        if len(persons) > 1:
            # 检查是否在交谈
            positions = [(p.get('bbox', [0, 0, 0, 0])[0], p.get('bbox', [0, 0, 0, 0])[1]) for p in persons]
            
            for i, pos1 in enumerate(positions):
                for j, pos2 in enumerate(positions):
                    if i >= j:
                        continue
                    
                    distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    
                    if distance < 200:  # 近距离
                        activities.add('talking')
        
        return list(activities)


class CrowdAnalyzer:
    """
    人群分析器
    
    分析人群密度和分布
    """
    
    def __init__(self, grid_size: int = 10):
        """
        初始化人群分析器
        
        Args:
            grid_size: 网格大小
        """
        self.grid_size = grid_size
        
        logger.info("CrowdAnalyzer initialized")
    
    def analyze(
        self,
        persons: List[Dict],
        frame_width: int = 1920,
        frame_height: int = 1080
    ) -> Dict:
        """
        分析人群
        
        Args:
            persons: 人员列表
            frame_width: 帧宽度
            frame_height: 帧高度
            
        Returns:
            分析结果
        """
        if not persons:
            return {
                'count': 0,
                'density': 0.0,
                'distribution': [],
                'hotspots': []
            }
        
        # 计算密度
        area = frame_width * frame_height
        density = len(persons) / (area / 100000)  # 每100k像素的人数
        
        # 创建密度网格
        cell_width = frame_width // self.grid_size
        cell_height = frame_height // self.grid_size
        
        grid = np.zeros((self.grid_size, self.grid_size))
        
        for person in persons:
            bbox = person.get('bbox', [0, 0, 0, 0])
            cx = bbox[0] + bbox[2] / 2
            cy = bbox[1] + bbox[3] / 2
            
            grid_x = min(int(cx / cell_width), self.grid_size - 1)
            grid_y = min(int(cy / cell_height), self.grid_size - 1)
            
            grid[grid_y, grid_x] += 1
        
        # 找热点
        hotspots = []
        threshold = max(2, len(persons) // 5)
        
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if grid[y, x] >= threshold:
                    hotspots.append({
                        'position': (x * cell_width + cell_width // 2, 
                                    y * cell_height + cell_height // 2),
                        'count': int(grid[y, x])
                    })
        
        return {
            'count': len(persons),
            'density': min(density / 10, 1.0),  # 归一化到 0-1
            'distribution': grid.tolist(),
            'hotspots': hotspots
        }


class SceneDescriptionGenerator:
    """
    场景描述生成器
    
    生成自然语言场景描述
    """
    
    # 描述模板
    TEMPLATES = {
        'empty': "场景为空，没有检测到任何对象。",
        'single_person': "场景中有1个人正在{activity}。",
        'multiple_persons': "场景中有{count}个人，主要活动是{activities}。",
        'crowded': "场景拥挤，有{count}个人，密度为{density}。",
        'with_vehicles': "场景中有{person_count}个人和{vehicle_count}辆车。",
        'activity_focused': "当前主要活动：{activities}。"
    }
    
    def __init__(self):
        """初始化描述生成器"""
        logger.info("SceneDescriptionGenerator initialized")
    
    def generate(
        self,
        scene_type: str,
        objects: List[SceneObject],
        activities: List[str],
        crowd_density: float
    ) -> str:
        """
        生成场景描述
        
        Args:
            scene_type: 场景类型
            objects: 对象列表
            activities: 活动列表
            crowd_density: 人群密度
            
        Returns:
            场景描述
        """
        person_count = sum(1 for obj in objects if obj.object_type == 'person')
        vehicle_count = sum(1 for obj in objects if obj.object_type in ['vehicle', 'car'])
        
        # 选择模板
        if person_count == 0 and vehicle_count == 0:
            return self.TEMPLATES['empty']
        
        elif person_count == 1:
            activity = activities[0] if activities else '站立'
            return self.TEMPLATES['single_person'].format(activity=self._translate_activity(activity))
        
        elif crowd_density > 0.5:
            return self.TEMPLATES['crowded'].format(
                count=person_count,
                density=f"{crowd_density * 100:.0f}%"
            )
        
        elif vehicle_count > 0:
            return self.TEMPLATES['with_vehicles'].format(
                person_count=person_count,
                vehicle_count=vehicle_count
            )
        
        elif person_count > 1:
            activity_str = '、'.join([self._translate_activity(a) for a in activities[:3]])
            return self.TEMPLATES['multiple_persons'].format(
                count=person_count,
                activities=activity_str or '活动'
            )
        
        return f"检测到{len(objects)}个对象。"
    
    def _translate_activity(self, activity: str) -> str:
        """翻译活动名称"""
        translations = {
            'standing': '站立',
            'walking': '行走',
            'running': '跑步',
            'sitting': '坐着',
            'lying': '躺着',
            'waving': '挥手',
            'talking': '交谈',
            'unknown': '活动'
        }
        return translations.get(activity, activity)


class SceneAnalyzer:
    """
    场景分析器
    
    整合所有场景分析功能
    """
    
    def __init__(self):
        """初始化场景分析器"""
        self.classifier = SceneClassifier()
        self.activity_analyzer = ActivityAnalyzer()
        self.crowd_analyzer = CrowdAnalyzer()
        self.description_generator = SceneDescriptionGenerator()
        
        # 历史记录
        self.scene_history: deque = deque(maxlen=100)
        
        logger.info("SceneAnalyzer initialized")
    
    def analyze(
        self,
        frame: np.ndarray,
        persons: List[Dict],
        actions: Dict[int, str] = None,
        vehicles: List[Dict] = None
    ) -> SceneDescription:
        """
        分析场景
        
        Args:
            frame: 图像帧
            persons: 人员列表
            actions: 动作字典
            vehicles: 车辆列表
            
        Returns:
            SceneDescription: 场景描述
        """
        # 构建对象列表
        objects = []
        
        for person in persons:
            obj = SceneObject(
                object_type='person',
                bbox=person.get('bbox', [0, 0, 0, 0]),
                confidence=person.get('confidence', 1.0),
                track_id=person.get('track_id')
            )
            objects.append(obj)
        
        for vehicle in (vehicles or []):
            obj = SceneObject(
                object_type='vehicle',
                bbox=vehicle.get('bbox', [0, 0, 0, 0]),
                confidence=vehicle.get('confidence', 1.0),
                track_id=vehicle.get('track_id')
            )
            objects.append(obj)
        
        # 分类场景
        scene_type = self.classifier.classify(frame, objects)
        
        # 分析活动
        activities = self.activity_analyzer.analyze(persons, actions or {})
        
        # 分析人群
        crowd_result = self.crowd_analyzer.analyze(persons)
        
        # 生成描述
        summary = self.description_generator.generate(
            scene_type=scene_type,
            objects=objects,
            activities=activities,
            crowd_density=crowd_result['density']
        )
        
        # 创建结果
        result = SceneDescription(
            scene_type=scene_type,
            objects=objects,
            activities=activities,
            crowd_density=crowd_result['density'],
            scene_summary=summary,
            timestamp=time.time()
        )
        
        # 保存历史
        self.scene_history.append(result)
        
        return result
    
    def get_scene_changes(self, window: int = 10) -> List[Dict]:
        """
        检测场景变化
        
        Args:
            window: 时间窗口
            
        Returns:
            变化列表
        """
        if len(self.scene_history) < 2:
            return []
        
        changes = []
        history = list(self.scene_history)[-window:]
        
        for i in range(1, len(history)):
            prev = history[i - 1]
            curr = history[i]
            
            # 检测场景类型变化
            if prev.scene_type != curr.scene_type:
                changes.append({
                    'type': 'scene_change',
                    'from': prev.scene_type,
                    'to': curr.scene_type,
                    'timestamp': curr.timestamp
                })
            
            # 检测人群密度变化
            density_change = abs(curr.crowd_density - prev.crowd_density)
            if density_change > 0.3:
                changes.append({
                    'type': 'density_change',
                    'change': density_change,
                    'timestamp': curr.timestamp
                })
        
        return changes
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if not self.scene_history:
            return {}
        
        history = list(self.scene_history)
        
        # 场景类型分布
        scene_counts = defaultdict(int)
        for scene in history:
            scene_counts[scene.scene_type] += 1
        
        # 平均人群密度
        avg_density = np.mean([s.crowd_density for s in history])
        
        # 活动统计
        activity_counts = defaultdict(int)
        for scene in history:
            for activity in scene.activities:
                activity_counts[activity] += 1
        
        return {
            'total_scenes': len(history),
            'scene_distribution': dict(scene_counts),
            'avg_crowd_density': round(avg_density, 3),
            'activity_distribution': dict(activity_counts)
        }


# 全局实例
_scene_analyzer = None

def get_scene_analyzer() -> SceneAnalyzer:
    """获取场景分析器单例"""
    global _scene_analyzer
    if _scene_analyzer is None:
        _scene_analyzer = SceneAnalyzer()
    return _scene_analyzer


# 测试代码
if __name__ == '__main__':
    print("Testing Scene Analyzer...")
    
    analyzer = SceneAnalyzer()
    
    # 模拟数据
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    persons = [
        {'bbox': [100, 100, 50, 150], 'confidence': 0.9, 'track_id': 1},
        {'bbox': [300, 100, 50, 150], 'confidence': 0.85, 'track_id': 2}
    ]
    actions = {1: 'walking', 2: 'standing'}
    
    # 分析场景
    result = analyzer.analyze(frame, persons, actions)
    
    print(f"Scene type: {result.scene_type}")
    print(f"Activities: {result.activities}")
    print(f"Crowd density: {result.crowd_density}")
    print(f"Summary: {result.scene_summary}")
    
    print("\nDone!")