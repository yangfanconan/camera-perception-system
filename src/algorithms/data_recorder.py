"""
数据记录模块

记录检测数据和历史：
1. 检测结果记录
2. 轨迹数据记录
3. 报警事件记录
4. 统计数据记录
5. 数据导出功能
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from loguru import logger
import time
import json
import csv
from pathlib import Path
from datetime import datetime
import threading
import queue


@dataclass
class DetectionRecord:
    """检测记录"""
    timestamp: float
    frame_id: int
    track_id: int
    bbox: List[int]
    confidence: float
    keypoints: Dict[str, List[float]]
    distance: float
    velocity: Tuple[float, float]
    gesture: str
    fall_state: str
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'frame_id': self.frame_id,
            'track_id': self.track_id,
            'bbox': self.bbox,
            'confidence': round(self.confidence, 3),
            'distance': round(self.distance, 2),
            'velocity': [round(v, 2) for v in self.velocity],
            'gesture': self.gesture,
            'fall_state': self.fall_state
        }


@dataclass
class TrajectoryRecord:
    """轨迹记录"""
    track_id: int
    start_time: float
    end_time: float
    positions: List[Tuple[float, float, float]]  # (x, y, timestamp)
    total_distance: float
    avg_speed: float
    duration: float
    
    def to_dict(self) -> Dict:
        return {
            'track_id': self.track_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': round(self.duration, 2),
            'total_distance': round(self.total_distance, 2),
            'avg_speed': round(self.avg_speed, 2),
            'point_count': len(self.positions)
        }


@dataclass
class SessionStats:
    """会话统计"""
    session_id: str
    start_time: float
    end_time: float = 0
    total_frames: int = 0
    total_detections: int = 0
    unique_persons: int = 0
    total_alerts: int = 0
    avg_fps: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'session_id': self.session_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': round(self.end_time - self.start_time, 2) if self.end_time else 0,
            'total_frames': self.total_frames,
            'total_detections': self.total_detections,
            'unique_persons': self.unique_persons,
            'total_alerts': self.total_alerts,
            'avg_fps': round(self.avg_fps, 2)
        }


class DataRecorder:
    """
    数据记录器
    
    记录和管理检测数据
    """
    
    def __init__(
        self,
        output_dir: str = "recordings",
        auto_save: bool = True,
        save_interval: float = 60.0
    ):
        """
        初始化数据记录器
        
        Args:
            output_dir: 输出目录
            auto_save: 自动保存
            save_interval: 自动保存间隔（秒）
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.auto_save = auto_save
        self.save_interval = save_interval
        
        # 数据存储
        self.detections: List[DetectionRecord] = []
        self.trajectories: Dict[int, TrajectoryRecord] = {}
        self.alerts: List[Dict] = []
        
        # 会话统计
        self.session = SessionStats(
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            start_time=time.time()
        )
        
        # 跟踪ID集合
        self.seen_track_ids: set = set()
        
        # 帧计数
        self.frame_count = 0
        
        # 写入队列
        self.write_queue: queue.Queue = queue.Queue()
        self.write_thread = None
        self.running = False
        
        # 启动写入线程
        if auto_save:
            self._start_write_thread()
        
        logger.info(f"DataRecorder initialized (output_dir={output_dir})")
    
    def _start_write_thread(self):
        """启动写入线程"""
        self.running = True
        self.write_thread = threading.Thread(target=self._write_worker, daemon=True)
        self.write_thread.start()
    
    def _write_worker(self):
        """写入工作线程"""
        last_save = time.time()
        
        while self.running:
            try:
                # 定期保存
                if time.time() - last_save > self.save_interval:
                    self._save_data()
                    last_save = time.time()
                
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Write worker error: {e}")
    
    def record_detection(
        self,
        frame_id: int,
        track_id: int,
        bbox: List[int],
        confidence: float,
        keypoints: Dict[str, List[float]],
        distance: float = 0.0,
        velocity: Tuple[float, float] = (0.0, 0.0),
        gesture: str = "unknown",
        fall_state: str = "normal"
    ):
        """
        记录检测结果
        
        Args:
            frame_id: 帧ID
            track_id: 跟踪ID
            bbox: 边界框
            confidence: 置信度
            keypoints: 关键点
            distance: 距离
            velocity: 速度
            gesture: 手势
            fall_state: 跌倒状态
        """
        record = DetectionRecord(
            timestamp=time.time(),
            frame_id=frame_id,
            track_id=track_id,
            bbox=bbox.copy() if isinstance(bbox, list) else list(bbox),
            confidence=confidence,
            keypoints=keypoints.copy() if keypoints else {},
            distance=distance,
            velocity=velocity,
            gesture=gesture,
            fall_state=fall_state
        )
        
        self.detections.append(record)
        self.session.total_detections += 1
        self.seen_track_ids.add(track_id)
        self.session.unique_persons = len(self.seen_track_ids)
        
        # 更新轨迹
        self._update_trajectory(track_id, bbox, record.timestamp)
    
    def _update_trajectory(self, track_id: int, bbox: List[int], timestamp: float):
        """更新轨迹"""
        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        
        if track_id not in self.trajectories:
            self.trajectories[track_id] = TrajectoryRecord(
                track_id=track_id,
                start_time=timestamp,
                end_time=timestamp,
                positions=[(cx, cy, timestamp)],
                total_distance=0.0,
                avg_speed=0.0,
                duration=0.0
            )
        else:
            traj = self.trajectories[track_id]
            
            # 计算移动距离
            if traj.positions:
                last_pos = traj.positions[-1]
                dist = np.sqrt((cx - last_pos[0]) ** 2 + (cy - last_pos[1]) ** 2)
                traj.total_distance += dist
            
            traj.positions.append((cx, cy, timestamp))
            traj.end_time = timestamp
            traj.duration = timestamp - traj.start_time
            
            if traj.duration > 0:
                traj.avg_speed = traj.total_distance / traj.duration
    
    def record_alert(self, alert: Dict):
        """记录报警"""
        alert['recorded_at'] = time.time()
        self.alerts.append(alert)
        self.session.total_alerts += 1
    
    def record_frame(self):
        """记录帧"""
        self.frame_count += 1
        self.session.total_frames = self.frame_count
        
        # 计算平均 FPS
        duration = time.time() - self.session.start_time
        if duration > 0:
            self.session.avg_fps = self.frame_count / duration
    
    def get_recent_detections(self, count: int = 100) -> List[DetectionRecord]:
        """获取最近的检测记录"""
        return self.detections[-count:]
    
    def get_trajectory(self, track_id: int) -> Optional[TrajectoryRecord]:
        """获取轨迹"""
        return self.trajectories.get(track_id)
    
    def get_all_trajectories(self) -> List[TrajectoryRecord]:
        """获取所有轨迹"""
        return list(self.trajectories.values())
    
    def get_session_stats(self) -> Dict:
        """获取会话统计"""
        return self.session.to_dict()
    
    def export_csv(self, filename: str = None) -> str:
        """
        导出为 CSV
        
        Args:
            filename: 文件名
            
        Returns:
            文件路径
        """
        if filename is None:
            filename = f"detections_{self.session.session_id}.csv"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # 写入表头
            writer.writerow([
                'timestamp', 'frame_id', 'track_id',
                'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
                'confidence', 'distance', 'velocity_x', 'velocity_y',
                'gesture', 'fall_state'
            ])
            
            # 写入数据
            for record in self.detections:
                writer.writerow([
                    record.timestamp,
                    record.frame_id,
                    record.track_id,
                    record.bbox[0], record.bbox[1], record.bbox[2], record.bbox[3],
                    record.confidence,
                    record.distance,
                    record.velocity[0], record.velocity[1],
                    record.gesture,
                    record.fall_state
                ])
        
        logger.info(f"Exported {len(self.detections)} records to {filepath}")
        return str(filepath)
    
    def export_json(self, filename: str = None) -> str:
        """
        导出为 JSON
        
        Args:
            filename: 文件名
            
        Returns:
            文件路径
        """
        if filename is None:
            filename = f"session_{self.session.session_id}.json"
        
        filepath = self.output_dir / filename
        
        data = {
            'session': self.session.to_dict(),
            'detections': [r.to_dict() for r in self.detections],
            'trajectories': [t.to_dict() for t in self.trajectories.values()],
            'alerts': self.alerts
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported session data to {filepath}")
        return str(filepath)
    
    def export_trajectories(self, filename: str = None) -> str:
        """
        导出轨迹数据
        
        Args:
            filename: 文件名
            
        Returns:
            文件路径
        """
        if filename is None:
            filename = f"trajectories_{self.session.session_id}.json"
        
        filepath = self.output_dir / filename
        
        data = {
            'session_id': self.session.session_id,
            'trajectories': {}
        }
        
        for track_id, traj in self.trajectories.items():
            data['trajectories'][str(track_id)] = {
                'info': traj.to_dict(),
                'points': traj.positions
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported {len(self.trajectories)} trajectories to {filepath}")
        return str(filepath)
    
    def _save_data(self):
        """保存数据"""
        if not self.detections:
            return
        
        # 保存检测数据
        self.export_json()
        
        logger.debug(f"Auto-saved session data")
    
    def end_session(self):
        """结束会话"""
        self.session.end_time = time.time()
        self.running = False
        
        # 保存最终数据
        self._save_data()
        
        logger.info(f"Session ended: {self.session.session_id}")
    
    def clear(self):
        """清除数据"""
        self.detections.clear()
        self.trajectories.clear()
        self.alerts.clear()
        self.seen_track_ids.clear()
        self.frame_count = 0
        
        # 重置会话
        self.session = SessionStats(
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            start_time=time.time()
        )
        
        logger.info("DataRecorder cleared")


class VideoRecorder:
    """
    视频记录器
    
    录制视频和关键帧
    """
    
    def __init__(self, output_dir: str = "recordings/videos"):
        """
        初始化视频记录器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer: Optional[cv2.VideoWriter] = None
        self.recording = False
        self.start_time = 0
        self.frame_count = 0
        
        # 关键帧
        self.keyframes: List[Tuple[np.ndarray, float, str]] = []
    
    def start_recording(
        self,
        width: int = 1920,
        height: int = 1080,
        fps: float = 30.0,
        filename: str = None
    ):
        """
        开始录制
        
        Args:
            width: 视频宽度
            height: 视频高度
            fps: 帧率
            filename: 文件名
        """
        if self.recording:
            return
        
        if filename is None:
            filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        
        filepath = self.output_dir / filename
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))
        
        self.recording = True
        self.start_time = time.time()
        self.frame_count = 0
        
        logger.info(f"Started recording: {filepath}")
    
    def write_frame(self, frame: np.ndarray):
        """写入帧"""
        if self.writer and self.recording:
            self.writer.write(frame)
            self.frame_count += 1
    
    def save_keyframe(self, frame: np.ndarray, label: str = ""):
        """
        保存关键帧
        
        Args:
            frame: 图像帧
            label: 标签
        """
        timestamp = time.time()
        self.keyframes.append((frame.copy(), timestamp, label))
        
        # 同时保存到文件
        filename = f"keyframe_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        filepath = self.output_dir / filename
        cv2.imwrite(str(filepath), frame)
        
        logger.debug(f"Saved keyframe: {filepath}")
    
    def stop_recording(self) -> Dict:
        """停止录制"""
        if not self.recording:
            return {}
        
        if self.writer:
            self.writer.release()
            self.writer = None
        
        duration = time.time() - self.start_time
        self.recording = False
        
        result = {
            'duration': duration,
            'frames': self.frame_count,
            'fps': self.frame_count / max(duration, 0.001)
        }
        
        logger.info(f"Stopped recording: {result}")
        return result
    
    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            'recording': self.recording,
            'frames': self.frame_count,
            'keyframes': len(self.keyframes),
            'duration': time.time() - self.start_time if self.recording else 0
        }


# 全局记录器实例
_data_recorder = None
_video_recorder = None

def get_data_recorder() -> DataRecorder:
    """获取数据记录器单例"""
    global _data_recorder
    if _data_recorder is None:
        _data_recorder = DataRecorder()
    return _data_recorder

def get_video_recorder() -> VideoRecorder:
    """获取视频记录器单例"""
    global _video_recorder
    if _video_recorder is None:
        _video_recorder = VideoRecorder()
    return _video_recorder


# 测试代码
if __name__ == '__main__':
    print("Testing Data Recorder...")
    
    recorder = DataRecorder(output_dir="test_recordings", auto_save=False)
    
    # 记录检测
    for i in range(10):
        recorder.record_detection(
            frame_id=i,
            track_id=1,
            bbox=[100 + i * 10, 100, 50, 150],
            confidence=0.9,
            keypoints={},
            distance=2.5,
            velocity=(5.0, 0.0)
        )
        recorder.record_frame()
    
    # 打印统计
    print("\nSession stats:", recorder.get_session_stats())
    
    # 导出
    csv_path = recorder.export_csv()
    json_path = recorder.export_json()
    
    print(f"\nExported to:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    
    recorder.end_session()
    print("\nDone!")