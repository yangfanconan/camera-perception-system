"""
时序融合模块

参考 Tesla FSD 的时序架构：
1. 特征队列 (Feature Queue): 缓存历史帧特征
2. 运动补偿 (Motion Compensation): 相机/物体运动补偿
3. 时序融合 (Temporal Fusion): 跨帧特征聚合
4. 视频目标跟踪 (Video Object Tracking): 时序一致性

优势：
- 提高检测稳定性
- 减少抖动和误检
- 估计运动速度和方向
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from loguru import logger
import time
import cv2


@dataclass
class TrackedObject:
    """跟踪目标"""
    track_id: int
    bbox: np.ndarray          # [x, y, w, h]
    keypoints: Dict           # 关键点
    depth: float              # 深度/距离
    velocity: np.ndarray      # [vx, vy, vz] 速度
    confidence: float         # 置信度
    age: int                  # 存活帧数
    hits: int                 # 连续检测次数
    last_seen: float          # 最后出现时间
    
    # 历史轨迹
    trajectory: List[Tuple[float, float, float]] = field(default_factory=list)  # [(x, y, t), ...]
    
    def update(self, bbox: np.ndarray, keypoints: Dict, depth: float, confidence: float):
        """更新目标状态"""
        self.bbox = bbox
        self.keypoints = keypoints
        self.depth = depth
        self.confidence = confidence
        self.hits += 1
        self.age += 1
        self.last_seen = time.time()
        
        # 记录轨迹
        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        self.trajectory.append((cx, cy, self.last_seen))
        
        # 限制轨迹长度
        if len(self.trajectory) > 30:
            self.trajectory.pop(0)


@dataclass
class TemporalFeature:
    """时序特征"""
    features: torch.Tensor    # 特征图
    timestamp: float          # 时间戳
    pose: np.ndarray          # 相机位姿 (可选)
    

class FeatureQueue:
    """
    特征队列
    
    缓存历史帧的特征，用于时序融合
    """
    
    def __init__(self, max_size: int = 8, feature_dim: int = 256):
        """
        初始化特征队列
        
        Args:
            max_size: 最大缓存帧数
            feature_dim: 特征维度
        """
        self.max_size = max_size
        self.feature_dim = feature_dim
        self.queue: deque = deque(maxlen=max_size)
        self.timestamps: deque = deque(maxlen=max_size)
    
    def push(self, features: torch.Tensor, timestamp: float = None):
        """
        添加特征到队列
        
        Args:
            features: 特征图 (C, H, W) 或 (1, C, H, W)
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = time.time()
        
        # 确保维度正确
        if features.dim() == 4:
            features = features.squeeze(0)
        
        self.queue.append(features.detach().cpu())
        self.timestamps.append(timestamp)
    
    def get_features(self, device: torch.device = None) -> Optional[torch.Tensor]:
        """
        获取堆叠的特征
        
        Returns:
            features: (T, C, H, W) 时序特征
        """
        if len(self.queue) == 0:
            return None
        
        features = torch.stack(list(self.queue))
        
        if device is not None:
            features = features.to(device)
        
        return features
    
    def get_time_deltas(self) -> np.ndarray:
        """获取时间间隔"""
        if len(self.timestamps) < 2:
            return np.array([])
        
        timestamps = np.array(list(self.timestamps))
        deltas = timestamps[-1] - timestamps[:-1]
        return deltas
    
    def clear(self):
        """清空队列"""
        self.queue.clear()
        self.timestamps.clear()
    
    def __len__(self):
        return len(self.queue)


class MotionCompensator:
    """
    运动补偿
    
    补偿相机运动，对齐历史帧特征
    """
    
    def __init__(self):
        self.prev_frame = None
        self.motion_matrix = np.eye(3)
    
    def compute_motion(self, frame: np.ndarray) -> np.ndarray:
        """
        计算帧间运动
        
        Args:
            frame: 当前帧
            
        Returns:
            motion_matrix: 3x3 变换矩阵
        """
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return np.eye(3)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 使用光流计算运动
        try:
            # 特征点检测
            prev_pts = cv2.goodFeaturesToTrack(
                self.prev_frame, maxCorners=200, qualityLevel=0.01, minDistance=30
            )
            
            if prev_pts is not None and len(prev_pts) > 10:
                # 计算光流
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_frame, gray, prev_pts, None
                )
                
                # 筛选有效点
                good_prev = prev_pts[status == 1]
                good_next = next_pts[status == 1]
                
                if len(good_prev) > 10:
                    # 计算仿射变换
                    motion_matrix, _ = cv2.estimateAffinePartial2D(good_prev, good_next)
                    
                    if motion_matrix is not None:
                        # 转换为 3x3
                        full_matrix = np.eye(3)
                        full_matrix[:2, :] = motion_matrix
                        self.motion_matrix = full_matrix
        except Exception as e:
            logger.debug(f"Motion computation failed: {e}")
        
        self.prev_frame = gray
        return self.motion_matrix
    
    def warp_features(self, features: torch.Tensor, motion_matrix: np.ndarray) -> torch.Tensor:
        """
        使用运动矩阵变换特征
        
        Args:
            features: 特征图 (C, H, W)
            motion_matrix: 3x3 变换矩阵
            
        Returns:
            warped_features: 变换后的特征
        """
        # 简化实现：直接返回原特征
        # 实际应用中需要实现特征变换
        return features


class TemporalFusion(nn.Module):
    """
    时序融合模块
    
    使用注意力机制聚合历史帧特征
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        num_heads: int = 4,
        num_frames: int = 8
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_frames = num_frames
        
        # 时间位置编码
        self.time_pos = nn.Parameter(torch.randn(1, num_frames, 1, 1, feature_dim))
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
    
    def forward(
        self,
        current_features: torch.Tensor,
        history_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        时序融合
        
        Args:
            current_features: 当前帧特征 (B, C, H, W)
            history_features: 历史帧特征 (B, T, C, H, W)
            
        Returns:
            fused_features: 融合后的特征 (B, C, H, W)
        """
        B, C, H, W = current_features.shape
        
        # 如果没有历史特征，直接返回当前特征
        if history_features is None:
            return current_features
        
        # history_features: (B, T, C, H, W)
        BT, C_h, H_h, W_h = history_features.shape[0] * history_features.shape[1], history_features.shape[2], history_features.shape[3], history_features.shape[4]
        T = history_features.shape[1]
        
        # 展平空间维度
        current_flat = current_features.flatten(2).transpose(1, 2)  # (B, HW, C)
        
        # 重塑历史特征: (B, T, C, H, W) -> (B, T, H, W, C)
        history_flat = history_features.permute(0, 1, 3, 4, 2).reshape(B, T * H * W, C)
        
        # 自注意力
        attn_out, _ = self.attention(current_flat, history_flat, history_flat)
        current_flat = self.norm1(current_flat + attn_out)
        
        # 前馈网络
        ffn_out = self.ffn(current_flat)
        current_flat = self.norm2(current_flat + ffn_out)
        
        # 恢复空间维度
        fused_features = current_flat.transpose(1, 2).reshape(B, C, H, W)
        
        return fused_features


class SimpleTracker:
    """
    简单目标跟踪器
    
    使用 IoU 匹配的简单跟踪
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        """
        初始化跟踪器
        
        Args:
            max_age: 最大丢失帧数
            min_hits: 确认跟踪的最小检测次数
            iou_threshold: IoU 匹配阈值
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks: Dict[int, TrackedObject] = {}
        self.next_id = 1
        self.frame_count = 0
    
    def update(self, detections: List[Dict]) -> List[TrackedObject]:
        """
        更新跟踪
        
        Args:
            detections: 检测结果列表
            
        Returns:
            active_tracks: 活跃的跟踪目标
        """
        self.frame_count += 1
        current_time = time.time()
        
        # 提取检测框
        det_boxes = np.array([d['bbox'] for d in detections]) if detections else np.array([])
        
        # 预测现有轨迹位置（简单实现：使用上一帧位置）
        track_boxes = np.array([t.bbox for t in self.tracks.values()]) if self.tracks else np.array([])
        track_ids = list(self.tracks.keys())
        
        # IoU 匹配
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(track_ids)))
        
        if len(det_boxes) > 0 and len(track_boxes) > 0:
            iou_matrix = self._compute_iou_matrix(det_boxes, track_boxes)
            
            # 贪婪匹配
            while iou_matrix.size > 0:
                max_iou = iou_matrix.max()
                if max_iou < self.iou_threshold:
                    break
                
                det_idx, track_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                matched.append((det_idx, track_idx))
                
                unmatched_dets.remove(det_idx)
                unmatched_tracks.remove(track_idx)
                
                iou_matrix = np.delete(iou_matrix, det_idx, axis=0)
                iou_matrix = np.delete(iou_matrix, track_idx, axis=1)
        
        # 更新匹配的轨迹
        for det_idx, track_idx in matched:
            track_id = track_ids[track_idx]
            det = detections[det_idx]
            self.tracks[track_id].update(
                bbox=np.array(det['bbox']),
                keypoints=det.get('keypoints', {}),
                depth=det.get('distance', 0),
                confidence=det.get('confidence', 0)
            )
        
        # 创建新轨迹
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            new_track = TrackedObject(
                track_id=self.next_id,
                bbox=np.array(det['bbox']),
                keypoints=det.get('keypoints', {}),
                depth=det.get('distance', 0),
                velocity=np.zeros(3),
                confidence=det.get('confidence', 0),
                age=1,
                hits=1,
                last_seen=current_time
            )
            self.tracks[self.next_id] = new_track
            self.next_id += 1
        
        # 标记丢失的轨迹
        for track_idx in unmatched_tracks:
            track_id = track_ids[track_idx]
            self.tracks[track_id].age += 1
        
        # 移除过老的轨迹
        to_remove = [
            tid for tid, track in self.tracks.items()
            if track.age > self.max_age
        ]
        for tid in to_remove:
            del self.tracks[tid]
        
        # 返回确认的活跃轨迹
        active_tracks = [
            track for track in self.tracks.values()
            if track.hits >= self.min_hits and track.age < self.max_age
        ]
        
        return active_tracks
    
    def _compute_iou_matrix(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """计算 IoU 矩阵"""
        if len(boxes1) == 0 or len(boxes2) == 0:
            return np.array([])
        
        # 转换为 [x1, y1, x2, y2]
        b1 = np.copy(boxes1)
        b1[:, 2:] = b1[:, :2] + b1[:, 2:]
        
        b2 = np.copy(boxes2)
        b2[:, 2:] = b2[:, :2] + b2[:, 2:]
        
        # 计算交集
        x1 = np.maximum(b1[:, None, 0], b2[None, :, 0])
        y1 = np.maximum(b1[:, None, 1], b2[None, :, 1])
        x2 = np.minimum(b1[:, None, 2], b2[None, :, 2])
        y2 = np.minimum(b1[:, None, 3], b2[None, :, 3])
        
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # 计算并集
        area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
        area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
        union = area1[:, None] + area2[None, :] - inter
        
        return inter / (union + 1e-6)
    
    def get_track(self, track_id: int) -> Optional[TrackedObject]:
        """获取指定轨迹"""
        return self.tracks.get(track_id)
    
    def get_all_tracks(self) -> List[TrackedObject]:
        """获取所有轨迹"""
        return list(self.tracks.values())
    
    def clear(self):
        """清空跟踪器"""
        self.tracks.clear()
        self.next_id = 1
        self.frame_count = 0


class VelocityEstimator:
    """
    速度估计器
    
    基于轨迹估计目标运动速度
    """
    
    def __init__(self, fps: float = 30.0, smoothing: float = 0.5):
        """
        初始化速度估计器
        
        Args:
            fps: 帧率
            smoothing: 平滑系数 (0-1)
        """
        self.fps = fps
        self.smoothing = smoothing
        self.dt = 1.0 / fps
    
    def estimate(self, track: TrackedObject) -> np.ndarray:
        """
        估计目标速度
        
        Args:
            track: 跟踪目标
            
        Returns:
            velocity: [vx, vy, vz] 速度 (m/s)
        """
        if len(track.trajectory) < 2:
            return np.zeros(3)
        
        # 获取最近的轨迹点
        recent = track.trajectory[-5:]  # 最近 5 帧
        
        if len(recent) < 2:
            return np.zeros(3)
        
        # 计算位移
        positions = np.array([(p[0], p[1]) for p in recent])
        times = np.array([p[2] for p in recent])
        
        # 线性拟合
        if len(positions) >= 2:
            dt_total = times[-1] - times[0]
            if dt_total > 0:
                dx = positions[-1, 0] - positions[0, 0]
                dy = positions[-1, 1] - positions[0, 1]
                
                # 像素速度 -> 实际速度（需要相机参数）
                # 简化：假设 100 像素 = 1 米
                vx = dx / dt_total / 100.0
                vy = dy / dt_total / 100.0
                
                # 深度变化
                vz = 0.0  # 需要多帧深度数据
                
                # 平滑
                velocity = np.array([vx, vy, vz])
                track.velocity = track.velocity * self.smoothing + velocity * (1 - self.smoothing)
                
                return track.velocity
        
        return np.zeros(3)


class TemporalProcessor:
    """
    时序处理器
    
    整合所有时序功能
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        num_frames: int = 8,
        device: str = None
    ):
        """
        初始化时序处理器
        
        Args:
            feature_dim: 特征维度
            num_frames: 缓存帧数
            device: 设备
        """
        # 设备选择
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # 特征队列
        self.feature_queue = FeatureQueue(num_frames, feature_dim)
        
        # 运动补偿
        self.motion_compensator = MotionCompensator()
        
        # 时序融合
        self.temporal_fusion = TemporalFusion(feature_dim, num_heads=4, num_frames=num_frames)
        self.temporal_fusion = self.temporal_fusion.to(self.device)
        
        # 目标跟踪
        self.tracker = SimpleTracker()
        
        # 速度估计
        self.velocity_estimator = VelocityEstimator()
        
        logger.info(f"TemporalProcessor initialized on {self.device}")
    
    def process_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        features: torch.Tensor = None
    ) -> Dict[str, Any]:
        """
        处理单帧
        
        Args:
            frame: 图像
            detections: 检测结果
            features: 特征图 (可选)
            
        Returns:
            result: 处理结果
        """
        result = {
            'tracks': [],
            'motion': None,
            'fused_features': None
        }
        
        # 运动补偿
        motion = self.motion_compensator.compute_motion(frame)
        result['motion'] = motion
        
        # 目标跟踪
        tracks = self.tracker.update(detections)
        
        # 速度估计
        for track in tracks:
            velocity = self.velocity_estimator.estimate(track)
            track.velocity = velocity
        
        result['tracks'] = tracks
        
        # 特征融合
        if features is not None:
            self.feature_queue.push(features)
            
            history = self.feature_queue.get_features(self.device)
            if history is not None and len(history) > 1:
                # 添加 batch 维度
                current = features.unsqueeze(0).to(self.device)
                history = history.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    fused = self.temporal_fusion(current, history)
                    result['fused_features'] = fused.squeeze(0).cpu()
        
        return result
    
    def reset(self):
        """重置处理器"""
        self.feature_queue.clear()
        self.tracker.clear()
        self.motion_compensator = MotionCompensator()


# 测试代码
if __name__ == '__main__':
    print("Testing TemporalProcessor...")
    
    # 创建处理器
    processor = TemporalProcessor(feature_dim=256, num_frames=8)
    
    # 模拟数据
    for i in range(10):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = [
            {
                'bbox': [100 + i*5, 100, 50, 100],
                'confidence': 0.9,
                'keypoints': {},
                'distance': 2.0 + i*0.1
            }
        ]
        features = torch.randn(256, 30, 40)
        
        result = processor.process_frame(frame, detections, features)
        
        print(f"Frame {i}: tracks={len(result['tracks'])}, "
              f"motion={result['motion'][0, :2]}, "
              f"fused_shape={result['fused_features'].shape if result['fused_features'] is not None else None}")
    
    print("Done!")