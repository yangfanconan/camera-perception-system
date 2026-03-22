"""
综合感知融合模块
功能：
1. 融合深度估计、目标检测、关键点检测
2. 3D位置估计
3. 多目标跟踪（基于深度）
4. 置信度评估
"""

import cv2
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import time
from loguru import logger


@dataclass
class DetectedObject:
    """检测到的目标"""
    id: int                          # 目标ID
    label: str                       # 标签（person, hand等）
    bbox: Tuple[int, int, int, int]  # 边界框 (x1, y1, x2, y2)
    confidence: float                # 检测置信度
    
    # 3D信息
    distance: float = 0.0            # 距离（米）
    position_3d: Tuple[float, float, float] = (0, 0, 0)  # 3D坐标 (x, y, z) 米
    
    # 关键点
    keypoints: List[Dict] = field(default_factory=list)
    
    # 深度信息
    depth_mean: float = 0.0          # 平均深度
    depth_std: float = 0.0           # 深度标准差
    
    # 跟踪信息
    track_id: int = -1               # 跟踪ID
    velocity: Tuple[float, float, float] = (0, 0, 0)  # 速度 (vx, vy, vz)
    
    # 时间戳
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'label': self.label,
            'bbox': self.bbox,
            'confidence': round(self.confidence, 3),
            'distance': round(self.distance, 2),
            'position_3d': tuple(round(v, 2) for v in self.position_3d),
            'keypoints': self.keypoints,
            'depth_mean': round(self.depth_mean, 2),
            'depth_std': round(self.depth_std, 3),
            'track_id': self.track_id,
            'velocity': tuple(round(v, 2) for v in self.velocity),
            'timestamp': self.timestamp
        }


@dataclass 
class PerceptionResult:
    """感知结果"""
    objects: List[DetectedObject] = field(default_factory=list)
    frame_shape: Tuple[int, int] = (0, 0)
    depth_calibrated: bool = False
    timestamp: float = 0.0
    
    # 统计信息
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'objects': [obj.to_dict() for obj in self.objects],
            'frame_shape': self.frame_shape,
            'depth_calibrated': self.depth_calibrated,
            'timestamp': self.timestamp,
            'stats': self.stats
        }


class SimpleTracker:
    """简单的多目标跟踪器（基于IoU和深度）"""
    
    def __init__(self, max_age: int = 30, min_hits: int = 3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: Dict[int, Dict] = {}
        self.next_id = 1
        self.frame_count = 0
    
    def update(self, detections: List[DetectedObject]) -> List[DetectedObject]:
        """更新跟踪"""
        self.frame_count += 1
        
        if not self.tracks:
            # 初始化所有检测为新跟踪
            for det in detections:
                det.track_id = self.next_id
                self.tracks[self.next_id] = {
                    'bbox': det.bbox,
                    'depth': det.distance,
                    'age': 0,
                    'hits': 1,
                    'position_history': deque(maxlen=10)
                }
                self.tracks[self.next_id]['position_history'].append(det.position_3d)
                self.next_id += 1
            return detections
        
        # 计算IoU矩阵
        matched = set()
        unmatched_tracks = set(self.tracks.keys())
        
        for det in detections:
            best_iou = 0
            best_track_id = -1
            
            for track_id in unmatched_tracks:
                track = self.tracks[track_id]
                iou = self._compute_iou(det.bbox, track['bbox'])
                
                # 考虑深度相似性
                depth_sim = 1.0 - min(1.0, abs(det.distance - track['depth']) / max(det.distance, 0.1))
                score = iou * 0.7 + depth_sim * 0.3
                
                if score > best_iou and score > 0.3:
                    best_iou = score
                    best_track_id = track_id
            
            if best_track_id >= 0:
                det.track_id = best_track_id
                self.tracks[best_track_id]['bbox'] = det.bbox
                self.tracks[best_track_id]['depth'] = det.distance
                self.tracks[best_track_id]['age'] = 0
                self.tracks[best_track_id]['hits'] += 1
                self.tracks[best_track_id]['position_history'].append(det.position_3d)
                
                # 计算速度
                history = list(self.tracks[best_track_id]['position_history'])
                if len(history) >= 2:
                    dt = 0.1  # 假设10fps
                    vx = (history[-1][0] - history[-2][0]) / dt
                    vy = (history[-1][1] - history[-2][1]) / dt
                    vz = (history[-1][2] - history[-2][2]) / dt
                    det.velocity = (vx, vy, vz)
                
                matched.add(best_track_id)
                unmatched_tracks.discard(best_track_id)
            else:
                # 新目标
                det.track_id = self.next_id
                self.tracks[self.next_id] = {
                    'bbox': det.bbox,
                    'depth': det.distance,
                    'age': 0,
                    'hits': 1,
                    'position_history': deque(maxlen=10)
                }
                self.tracks[self.next_id]['position_history'].append(det.position_3d)
                self.next_id += 1
        
        # 更新未匹配的跟踪
        for track_id in unmatched_tracks:
            self.tracks[track_id]['age'] += 1
        
        # 删除过期的跟踪
        to_delete = [tid for tid, t in self.tracks.items() if t['age'] > self.max_age]
        for tid in to_delete:
            del self.tracks[tid]
        
        return detections
    
    def _compute_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """计算IoU"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        inter = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0.0


class PerceptionFusion:
    """综合感知融合器"""
    
    def __init__(
        self,
        depth_estimator=None,
        detector=None,
        camera_params: Dict = None
    ):
        """
        初始化感知融合器
        
        Args:
            depth_estimator: 深度估计器
            detector: 目标检测器
            camera_params: 相机参数 {fx, fy, cx, cy, image_width, image_height}
        """
        self.depth_estimator = depth_estimator
        self.detector = detector
        self.tracker = SimpleTracker()
        
        # 相机参数
        self.camera_params = camera_params or {
            'fx': 1000.0,
            'fy': 1000.0,
            'cx': 960.0,
            'cy': 540.0,
            'image_width': 1920,
            'image_height': 1080
        }
        
        self.next_obj_id = 1
    
    def process(
        self,
        image: np.ndarray,
        depth_map: np.ndarray = None,
        detections: Dict = None
    ) -> PerceptionResult:
        """
        处理一帧图像
        
        Args:
            image: BGR图像
            depth_map: 深度图（可选，如果不提供则自动估计）
            detections: 检测结果（可选，如果不提供则自动检测）
        
        Returns:
            PerceptionResult: 感知结果
        """
        start_time = time.time()
        h, w = image.shape[:2]
        
        # 1. 获取深度图
        if depth_map is None and self.depth_estimator is not None:
            depth_map = self.depth_estimator.estimate(image, normalize=False)
        
        # 2. 获取检测结果
        if detections is None and self.detector is not None:
            try:
                det_result = self.detector.detect(image)
                if det_result:
                    detections = det_result.to_dict()
            except Exception as e:
                logger.warning(f"Detection failed: {e}")
                detections = None
        
        # 3. 融合处理
        objects = []
        
        if depth_map is not None:
            # 处理检测结果（如果有）
            if detections:
                # 处理人体检测
                for person in detections.get('persons', []):
                    obj = self._process_detection(person, 'person', depth_map, w, h)
                    if obj:
                        objects.append(obj)
                
                # 处理手部检测
                for hand in detections.get('hands', []):
                    obj = self._process_detection(hand, 'hand', depth_map, w, h)
                    if obj:
                        objects.append(obj)
            
            # 如果没有检测到目标，创建一个全图深度对象
            if not objects:
                # 计算全图深度统计
                depth_mean = float(np.mean(depth_map))
                depth_std = float(np.std(depth_map))
                
                if self.depth_estimator and self.depth_estimator.calibrated:
                    distance = depth_mean * self.depth_estimator.scale_factor + self.depth_estimator.offset
                    distance = max(0, distance)
                else:
                    distance = depth_mean
                
                # 创建全图深度对象
                obj = DetectedObject(
                    id=1,
                    label='scene',
                    bbox=(0, 0, w, h),
                    confidence=1.0,
                    distance=distance,
                    position_3d=(0, 0, distance),
                    depth_mean=depth_mean,
                    depth_std=depth_std,
                    timestamp=time.time()
                )
                objects.append(obj)
        
        # 4. 跟踪
        objects = self.tracker.update(objects)
        
        # 5. 构建结果
        result = PerceptionResult(
            objects=objects,
            frame_shape=(h, w),
            depth_calibrated=self.depth_estimator.calibrated if self.depth_estimator else False,
            timestamp=time.time()
        )
        
        # 统计信息
        result.stats = {
            'num_objects': len(objects),
            'num_persons': len([o for o in objects if o.label == 'person']),
            'num_hands': len([o for o in objects if o.label == 'hand']),
            'processing_time_ms': round((time.time() - start_time) * 1000, 1)
        }
        
        return result
    
    def _process_detection(
        self,
        detection: Dict,
        label: str,
        depth_map: np.ndarray,
        img_w: int,
        img_h: int
    ) -> Optional[DetectedObject]:
        """处理单个检测"""
        bbox = detection.get('bbox')
        if not bbox:
            return None
        
        # bbox 可能是列表或元组
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[:4]
        else:
            return None
        
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(img_w, int(x2)), min(img_h, int(y2))
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # 获取区域深度
        region = depth_map[y1:y2, x1:x2]
        if region.size == 0:
            return None
        
        depth_values = region.flatten()
        depth_mean = float(np.mean(depth_values))
        depth_std = float(np.std(depth_values))
        
        # 计算真实距离
        if self.depth_estimator and self.depth_estimator.calibrated:
            distance = depth_mean * self.depth_estimator.scale_factor + self.depth_estimator.offset
            distance = max(0, distance)
        else:
            distance = depth_mean
        
        # 计算3D位置
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        position_3d = self._pixel_to_3d(center_x, center_y, distance, img_w, img_h)
        
        # 处理关键点
        keypoints = detection.get('keypoints', {})
        if isinstance(keypoints, dict):
            # 转换为列表格式
            kp_list = []
            for name, coords in keypoints.items():
                if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    kp_list.append({'name': name, 'x': coords[0], 'y': coords[1]})
            keypoints = kp_list
        
        # 创建目标对象
        obj = DetectedObject(
            id=self.next_obj_id,
            label=label,
            bbox=(x1, y1, x2, y2),
            confidence=detection.get('confidence', 0.0),
            distance=distance,
            position_3d=position_3d,
            keypoints=keypoints,
            depth_mean=depth_mean,
            depth_std=depth_std,
            timestamp=time.time()
        )
        
        self.next_obj_id += 1
        
        return obj
    
    def _pixel_to_3d(
        self,
        px: float,
        py: float,
        depth: float,
        img_w: int,
        img_h: int
    ) -> Tuple[float, float, float]:
        """
        像素坐标转3D坐标
        
        Args:
            px, py: 像素坐标
            depth: 深度（米）
            img_w, img_h: 图像尺寸
        
        Returns:
            (x, y, z): 3D坐标（米）
        """
        # 使用相机参数
        fx = self.camera_params.get('fx', 1000.0)
        fy = self.camera_params.get('fy', 1000.0)
        cx = self.camera_params.get('cx', img_w / 2)
        cy = self.camera_params.get('cy', img_h / 2)
        
        # 反投影
        z = depth
        x = (px - cx) * z / fx
        y = (py - cy) * z / fy
        
        return (x, y, z)
    
    def get_bird_eye_view(
        self,
        result: PerceptionResult,
        view_size: Tuple[int, int] = (400, 400),
        max_distance: float = 10.0
    ) -> np.ndarray:
        """
        生成俯视图
        
        Args:
            result: 感知结果
            view_size: 视图尺寸 (width, height)
            max_distance: 最大显示距离（米）
        
        Returns:
            俯视图图像
        """
        view_w, view_h = view_size
        view = np.ones((view_h, view_w, 3), dtype=np.uint8) * 30  # 深灰色背景
        
        # 绘制网格
        grid_step = view_w // 10
        for i in range(0, view_w, grid_step):
            cv2.line(view, (i, 0), (i, view_h), (50, 50, 50), 1)
        for i in range(0, view_h, grid_step):
            cv2.line(view, (0, i), (view_w, i), (50, 50, 50), 1)
        
        # 绘制相机位置（底部中心）
        camera_pos = (view_w // 2, view_h - 20)
        cv2.circle(view, camera_pos, 10, (0, 255, 0), -1)
        cv2.putText(view, "CAM", (camera_pos[0] - 15, camera_pos[1] + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # 绘制目标
        for obj in result.objects:
            # 转换坐标 (x, z) -> 俯视图坐标
            # x: 左右, z: 前后（深度）
            x, y, z = obj.position_3d
            
            # 映射到视图
            view_x = int(view_w // 2 + x / max_distance * view_w // 2)
            view_y = int(view_h - 40 - z / max_distance * (view_h - 60))
            
            # 边界检查
            view_x = max(0, min(view_w - 1, view_x))
            view_y = max(0, min(view_h - 1, view_y))
            
            # 根据标签选择颜色
            if obj.label == 'person':
                color = (0, 200, 255)  # 橙色
            else:
                color = (255, 200, 0)  # 青色
            
            # 绘制目标
            cv2.circle(view, (view_x, view_y), 8, color, -1)
            
            # 显示距离
            cv2.putText(view, f"{obj.distance:.1f}m", (view_x + 10, view_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 显示跟踪ID
            cv2.putText(view, f"#{obj.track_id}", (view_x - 15, view_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # 添加标题
        cv2.putText(view, "Bird Eye View", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 添加距离刻度
        for d in range(2, int(max_distance) + 1, 2):
            y_pos = int(view_h - 40 - d / max_distance * (view_h - 60))
            cv2.line(view, (0, y_pos), (view_w, y_pos), (100, 100, 100), 1)
            cv2.putText(view, f"{d}m", (5, y_pos - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        
        return view


def visualize_perception(
    image: np.ndarray,
    result: PerceptionResult,
    show_distance: bool = True,
    show_3d: bool = True,
    show_keypoints: bool = True
) -> np.ndarray:
    """
    可视化感知结果
    
    Args:
        image: 原始图像
        result: 感知结果
        show_distance: 显示距离
        show_3d: 显示3D坐标
        show_keypoints: 显示关键点
    
    Returns:
        可视化后的图像
    """
    vis = image.copy()
    
    for obj in result.objects:
        x1, y1, x2, y2 = obj.bbox
        
        # 根据标签选择颜色
        if obj.label == 'person':
            color = (0, 200, 255)  # 橙色
        else:
            color = (255, 200, 0)  # 青色
        
        # 绘制边界框
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        
        # 绘制信息标签
        label_parts = [f"#{obj.track_id}", obj.label]
        if show_distance:
            label_parts.append(f"{obj.distance:.1f}m")
        
        label = " | ".join(label_parts)
        
        # 标签背景
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - label_h - 8), (x1 + label_w + 8, y1), color, -1)
        cv2.putText(vis, label, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示3D坐标
        if show_3d:
            x, y, z = obj.position_3d
            pos_text = f"3D: ({x:.1f}, {y:.1f}, {z:.1f})m"
            cv2.putText(vis, pos_text, (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # 绘制关键点
        if show_keypoints and obj.keypoints:
            for kp in obj.keypoints:
                kx, ky = int(kp.get('x', 0)), int(kp.get('y', 0))
                if 0 <= kx < vis.shape[1] and 0 <= ky < vis.shape[0]:
                    cv2.circle(vis, (kx, ky), 3, (0, 255, 255), -1)
    
    # 绘制统计信息
    stats_text = f"Objects: {result.stats.get('num_objects', 0)} | " \
                 f"Persons: {result.stats.get('num_persons', 0)} | " \
                 f"Time: {result.stats.get('processing_time_ms', 0):.1f}ms"
    cv2.putText(vis, stats_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return vis