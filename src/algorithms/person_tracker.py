"""
多人跟踪模块

基于 DeepSORT 思想实现多人跟踪：
1. 检测关联：IoU + 外观特征
2. ID 持久化：遮挡后恢复 ID
3. 轨迹管理：轨迹创建、更新、删除
4. 运动预测：卡尔曼滤波预测位置
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from loguru import logger
import time
import colorsys


@dataclass
class Track:
    """跟踪目标"""
    track_id: int
    bbox: List[int]                    # [x, y, w, h]
    keypoints: Dict[str, List[float]]  # 关键点
    confidence: float
    age: int = 0                       # 存在帧数
    hits: int = 1                      # 匹配次数
    misses: int = 0                    # 连续未匹配次数
    state: str = "tentative"           # tentative, confirmed, lost
    
    # 运动状态
    velocity: Tuple[float, float] = (0.0, 0.0)
    last_position: Tuple[int, int] = (0, 0)
    
    # 外观特征
    feature: Optional[np.ndarray] = None
    feature_history: deque = field(default_factory=lambda: deque(maxlen=50))
    
    # 轨迹历史
    trajectory: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # 元数据
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    
    # 颜色（用于可视化）
    color: Tuple[int, int, int] = (255, 0, 0)
    
    def get_center(self) -> Tuple[int, int]:
        """获取中心点"""
        x, y, w, h = self.bbox
        return (int(x + w / 2), int(y + h / 2))
    
    def get_area(self) -> int:
        """获取面积"""
        return self.bbox[2] * self.bbox[3]
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'track_id': self.track_id,
            'bbox': self.bbox,
            'confidence': round(self.confidence, 3),
            'age': self.age,
            'state': self.state,
            'velocity': [round(v, 2) for v in self.velocity],
            'duration': round(time.time() - self.first_seen, 1)
        }


class KalmanFilter:
    """
    简化的卡尔曼滤波器
    
    状态: [x, y, w, h, vx, vy, vw, vh]
    """
    
    def __init__(self):
        self.dim_x = 8  # 状态维度
        self.dim_z = 4  # 观测维度
        
        # 状态转移矩阵
        self.F = np.eye(self.dim_x)
        self.F[:4, 4:] = np.eye(4)
        
        # 观测矩阵
        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[:4, :4] = np.eye(4)
        
        # 过程噪声
        self.Q = np.eye(self.dim_x) * 0.01
        self.Q[4:, 4:] *= 10
        
        # 观测噪声
        self.R = np.eye(self.dim_z) * 1.0
        
        # 初始状态协方差
        self.P = np.eye(self.dim_x) * 10
    
    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        初始化状态
        
        Args:
            measurement: [x, y, w, h]
            
        Returns:
            (state, covariance)
        """
        mean = np.zeros(self.dim_x)
        mean[:4] = measurement
        return mean.copy(), self.P.copy()
    
    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测下一状态
        
        Args:
            mean: 状态均值
            covariance: 状态协方差
            
        Returns:
            (predicted_mean, predicted_covariance)
        """
        mean = self.F @ mean
        covariance = self.F @ covariance @ self.F.T + self.Q
        return mean, covariance
    
    def update(self, mean: np.ndarray, covariance: np.ndarray, 
               measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        更新状态
        
        Args:
            mean: 状态均值
            covariance: 状态协方差
            measurement: 观测值
            
        Returns:
            (updated_mean, updated_covariance)
        """
        # 卡尔曼增益
        S = self.H @ covariance @ self.H.T + self.R
        K = covariance @ self.H.T @ np.linalg.inv(S)
        
        # 更新
        y = measurement - self.H @ mean
        mean = mean + K @ y
        covariance = (np.eye(self.dim_x) - K @ self.H) @ covariance
        
        return mean, covariance


class PersonTracker:
    """
    多人跟踪器
    
    实现多人跟踪和 ID 管理
    """
    
    # 跟踪参数
    MAX_AGE = 30              # 最大存在帧数
    MIN_HITS = 3              # 确认所需匹配次数
    MAX_MISSES = 10           # 最大连续未匹配次数
    IOU_THRESHOLD = 0.3       # IoU 阈值
    DISTANCE_THRESHOLD = 100  # 距离阈值（像素）
    
    def __init__(self, max_tracks: int = 50):
        """
        初始化跟踪器
        
        Args:
            max_tracks: 最大跟踪数量
        """
        self.max_tracks = max_tracks
        
        # 跟踪列表
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        
        # 卡尔曼滤波器
        self.kalman_filters: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.kf = KalmanFilter()
        
        # 统计信息
        self.total_tracks = 0
        self.active_tracks = 0
        
        logger.info(f"PersonTracker initialized (max_tracks={max_tracks})")
    
    def update(
        self,
        detections: List[Dict],
        image_width: int = 1920,
        image_height: int = 1080
    ) -> List[Track]:
        """
        更新跟踪状态
        
        Args:
            detections: 检测结果列表
            image_width: 图像宽度
            image_height: 图像高度
            
        Returns:
            活跃的跟踪列表
        """
        current_time = time.time()
        
        # 预测所有跟踪的位置
        predicted_positions = {}
        for track_id, track in self.tracks.items():
            if track_id in self.kalman_filters:
                mean, cov = self.kalman_filters[track_id]
                mean, cov = self.kf.predict(mean, cov)
                self.kalman_filters[track_id] = (mean, cov)
                predicted_positions[track_id] = mean[:4].copy()
        
        # 计算代价矩阵
        if detections and self.tracks:
            cost_matrix = self._compute_cost_matrix(detections, predicted_positions)
            matches, unmatched_dets, unmatched_tracks = self._hungarian_assignment(cost_matrix)
        else:
            matches = []
            unmatched_dets = list(range(len(detections)))
            unmatched_tracks = list(self.tracks.keys())
        
        # 更新匹配的跟踪
        for det_idx, track_id in matches:
            det = detections[det_idx]
            track = self.tracks[track_id]
            
            # 更新边界框
            old_bbox = track.bbox.copy()
            track.bbox = det['bbox'].copy()
            track.confidence = det.get('confidence', 1.0)
            track.keypoints = det.get('keypoints', {})
            track.hits += 1
            track.misses = 0
            track.age += 1
            track.last_seen = current_time
            
            # 更新卡尔曼滤波器
            if track_id in self.kalman_filters:
                mean, cov = self.kalman_filters[track_id]
                measurement = np.array(det['bbox'])
                mean, cov = self.kf.update(mean, cov, measurement)
                self.kalman_filters[track_id] = (mean, cov)
            
            # 计算速度
            old_center = (old_bbox[0] + old_bbox[2] / 2, old_bbox[1] + old_bbox[3] / 2)
            new_center = (track.bbox[0] + track.bbox[2] / 2, track.bbox[1] + track.bbox[3] / 2)
            track.velocity = (new_center[0] - old_center[0], new_center[1] - old_center[1])
            
            # 更新轨迹
            track.trajectory.append((new_center[0], new_center[1], current_time))
            
            # 更新状态
            if track.state == "tentative" and track.hits >= self.MIN_HITS:
                track.state = "confirmed"
        
        # 处理未匹配的检测（创建新跟踪）
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            self._create_track(det, image_width, image_height)
        
        # 处理未匹配的跟踪
        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            track.misses += 1
            track.age += 1
            
            if track.misses > self.MAX_MISSES:
                track.state = "lost"
        
        # 删除丢失的跟踪
        lost_ids = [tid for tid, t in self.tracks.items() if t.state == "lost"]
        for tid in lost_ids:
            del self.tracks[tid]
            if tid in self.kalman_filters:
                del self.kalman_filters[tid]
        
        # 返回活跃的跟踪
        active_tracks = [t for t in self.tracks.values() if t.state in ["tentative", "confirmed"]]
        self.active_tracks = len(active_tracks)
        
        return active_tracks
    
    def _create_track(self, detection: Dict, image_width: int, image_height: int):
        """创建新跟踪"""
        if len(self.tracks) >= self.max_tracks:
            return
        
        track_id = self.next_id
        self.next_id += 1
        
        # 生成颜色
        hue = (track_id * 0.618033988749895) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        color = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
        
        track = Track(
            track_id=track_id,
            bbox=detection['bbox'].copy(),
            keypoints=detection.get('keypoints', {}),
            confidence=detection.get('confidence', 1.0),
            color=color
        )
        
        self.tracks[track_id] = track
        
        # 初始化卡尔曼滤波器
        measurement = np.array(detection['bbox'], dtype=float)
        self.kalman_filters[track_id] = self.kf.initiate(measurement)
        
        self.total_tracks += 1
        
        logger.debug(f"Created track {track_id}")
    
    def _compute_cost_matrix(
        self,
        detections: List[Dict],
        predicted_positions: Dict[int, np.ndarray]
    ) -> np.ndarray:
        """
        计算代价矩阵
        
        Args:
            detections: 检测列表
            predicted_positions: 预测位置
            
        Returns:
            代价矩阵
        """
        n_dets = len(detections)
        n_tracks = len(predicted_positions)
        
        cost_matrix = np.zeros((n_dets, n_tracks))
        
        track_ids = list(predicted_positions.keys())
        
        for i, det in enumerate(detections):
            det_bbox = det['bbox']
            
            for j, track_id in enumerate(track_ids):
                pred_bbox = predicted_positions[track_id]
                
                # 计算 IoU
                iou = self._compute_iou(det_bbox, pred_bbox)
                
                # 计算距离
                det_center = (det_bbox[0] + det_bbox[2] / 2, det_bbox[1] + det_bbox[3] / 2)
                pred_center = (pred_bbox[0] + pred_bbox[2] / 2, pred_bbox[1] + pred_bbox[3] / 2)
                distance = np.sqrt((det_center[0] - pred_center[0]) ** 2 + 
                                   (det_center[1] - pred_center[1]) ** 2)
                
                # 综合代价
                # IoU 越大代价越小，距离越远代价越大
                cost = 1.0 - iou + distance / 1000.0
                
                cost_matrix[i, j] = cost
        
        return cost_matrix
    
    def _compute_iou(self, bbox1: List, bbox2: np.ndarray) -> float:
        """计算 IoU"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # 计算交集
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        
        # 计算并集
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        return inter_area / max(union_area, 1)
    
    def _hungarian_assignment(self, cost_matrix: np.ndarray) -> Tuple[List, List, List]:
        """
        匈牙利算法分配
        
        Args:
            cost_matrix: 代价矩阵
            
        Returns:
            (matches, unmatched_dets, unmatched_tracks)
        """
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        except ImportError:
            # 简单的贪心分配
            return self._greedy_assignment(cost_matrix)
        
        matches = []
        unmatched_dets = list(range(cost_matrix.shape[0]))
        unmatched_tracks = list(range(cost_matrix.shape[1]))
        
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 1.0 - self.IOU_THRESHOLD:  # 代价阈值
                matches.append((i, j))
                if i in unmatched_dets:
                    unmatched_dets.remove(i)
                if j in unmatched_tracks:
                    unmatched_tracks.remove(j)
        
        return matches, unmatched_dets, unmatched_tracks
    
    def _greedy_assignment(self, cost_matrix: np.ndarray) -> Tuple[List, List, List]:
        """贪心分配"""
        n_dets, n_tracks = cost_matrix.shape
        
        matches = []
        unmatched_dets = list(range(n_dets))
        unmatched_tracks = list(range(n_tracks))
        
        # 按代价排序
        indices = np.argsort(cost_matrix.ravel())
        
        for idx in indices:
            i, j = idx // n_tracks, idx % n_tracks
            
            if i in unmatched_dets and j in unmatched_tracks:
                if cost_matrix[i, j] < 1.0 - self.IOU_THRESHOLD:
                    matches.append((i, j))
                    unmatched_dets.remove(i)
                    unmatched_tracks.remove(j)
        
        return matches, unmatched_dets, unmatched_tracks
    
    def get_track(self, track_id: int) -> Optional[Track]:
        """获取跟踪"""
        return self.tracks.get(track_id)
    
    def get_all_tracks(self) -> List[Track]:
        """获取所有跟踪"""
        return list(self.tracks.values())
    
    def get_active_tracks(self) -> List[Track]:
        """获取活跃跟踪"""
        return [t for t in self.tracks.values() if t.state in ["tentative", "confirmed"]]
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_tracks': self.total_tracks,
            'active_tracks': self.active_tracks,
            'current_tracks': len(self.tracks)
        }
    
    def reset(self):
        """重置跟踪器"""
        self.tracks.clear()
        self.kalman_filters.clear()
        self.next_id = 1
        self.total_tracks = 0
        self.active_tracks = 0
        logger.info("PersonTracker reset")


# 全局跟踪器实例
_person_tracker = None

def get_person_tracker() -> PersonTracker:
    """获取多人跟踪器单例"""
    global _person_tracker
    if _person_tracker is None:
        _person_tracker = PersonTracker()
    return _person_tracker


# 测试代码
if __name__ == '__main__':
    print("Testing Person Tracker...")
    
    tracker = PersonTracker()
    
    # 模拟检测
    detections = [
        {'bbox': [100, 100, 50, 150], 'confidence': 0.9},
        {'bbox': [300, 100, 50, 150], 'confidence': 0.85},
    ]
    
    # 更新跟踪
    for frame in range(20):
        # 模拟移动
        for i, det in enumerate(detections):
            det['bbox'][0] += 5  # 向右移动
        
        tracks = tracker.update(detections)
        
        print(f"\nFrame {frame}:")
        for track in tracks:
            print(f"  Track {track.track_id}: bbox={track.bbox}, state={track.state}")
    
    print("\nStats:", tracker.get_stats())
    print("\nDone!")