"""
视频录制与回放模块

功能：
1. 视频录制
2. 视频回放
3. 关键帧提取
4. 视频剪辑
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger
import time
import json
import threading
import queue
from datetime import datetime


@dataclass
class RecordingConfig:
    """录制配置"""
    output_dir: str = "recordings"
    filename_prefix: str = "recording"
    codec: str = "mp4v"
    fps: float = 30.0
    resolution: Tuple[int, int] = (1920, 1080)
    max_duration: float = 600.0  # 最大时长（秒）
    split_interval: float = 300.0  # 分割间隔（秒）
    save_keyframes: bool = True
    keyframe_interval: float = 5.0  # 关键帧间隔（秒）


@dataclass
class RecordingSession:
    """录制会话"""
    session_id: str
    start_time: float
    end_time: float = 0
    duration: float = 0
    frame_count: int = 0
    file_path: str = ""
    keyframes: List[str] = field(default_factory=list)
    events: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'session_id': self.session_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': round(self.duration, 2),
            'frame_count': self.frame_count,
            'file_path': self.file_path,
            'keyframes': self.keyframes,
            'events': self.events
        }


class VideoRecorder:
    """
    视频录制器
    
    支持实时录制和事件触发录制
    """
    
    def __init__(self, config: RecordingConfig = None):
        """
        初始化视频录制器
        
        Args:
            config: 录制配置
        """
        self.config = config or RecordingConfig()
        
        # 创建输出目录
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 录制状态
        self.is_recording = False
        self.writer: Optional[cv2.VideoWriter] = None
        self.session: Optional[RecordingSession] = None
        
        # 帧队列
        self.frame_queue: queue.Queue = queue.Queue(maxsize=100)
        
        # 录制线程
        self.recording_thread = None
        self.running = False
        
        # 关键帧
        self.last_keyframe_time = 0
        
        # 统计
        self.total_sessions = 0
        
        logger.info(f"VideoRecorder initialized (output_dir={self.output_dir})")
    
    def start_recording(self, resolution: Tuple[int, int] = None) -> bool:
        """
        开始录制
        
        Args:
            resolution: 视频分辨率
            
        Returns:
            是否成功
        """
        if self.is_recording:
            logger.warning("Already recording")
            return False
        
        try:
            # 确定分辨率
            res = resolution or self.config.resolution
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = f"{self.config.filename_prefix}_{timestamp}"
            filename = f"{session_id}.mp4"
            filepath = self.output_dir / filename
            
            # 创建 VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*self.config.codec)
            self.writer = cv2.VideoWriter(
                str(filepath),
                fourcc,
                self.config.fps,
                res
            )
            
            if not self.writer.isOpened():
                logger.error("Failed to create video writer")
                return False
            
            # 创建会话
            self.session = RecordingSession(
                session_id=session_id,
                start_time=time.time(),
                file_path=str(filepath)
            )
            
            # 启动录制线程
            self.is_recording = True
            self.running = True
            self.recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
            self.recording_thread.start()
            
            self.total_sessions += 1
            
            logger.info(f"Started recording: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            return False
    
    def stop_recording(self) -> Optional[RecordingSession]:
        """
        停止录制
        
        Returns:
            录制会话信息
        """
        if not self.is_recording:
            return None
        
        self.is_recording = False
        self.running = False
        
        # 等待线程结束
        if self.recording_thread:
            self.recording_thread.join(timeout=2.0)
        
        # 释放写入器
        if self.writer:
            self.writer.release()
            self.writer = None
        
        # 完成会话
        if self.session:
            self.session.end_time = time.time()
            self.session.duration = self.session.end_time - self.session.start_time
            
            # 保存会话信息
            self._save_session_info()
            
            logger.info(f"Stopped recording: duration={self.session.duration:.1f}s, frames={self.session.frame_count}")
        
        session = self.session
        self.session = None
        
        return session
    
    def write_frame(self, frame: np.ndarray):
        """
        写入帧
        
        Args:
            frame: 图像帧
        """
        if not self.is_recording:
            return
        
        try:
            self.frame_queue.put(frame, block=False)
        except queue.Full:
            logger.warning("Frame queue full, dropping frame")
    
    def _recording_loop(self):
        """录制循环"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                
                if self.writer and self.writer.isOpened():
                    self.writer.write(frame)
                    self.session.frame_count += 1
                    
                    # 检查关键帧
                    if self.config.save_keyframes:
                        self._check_keyframe(frame)
                    
                    # 检查最大时长
                    if self.session.duration > self.config.max_duration:
                        logger.info("Max duration reached, stopping")
                        self.stop_recording()
                        break
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Recording error: {e}")
    
    def _check_keyframe(self, frame: np.ndarray):
        """检查并保存关键帧"""
        current_time = time.time()
        
        if current_time - self.last_keyframe_time >= self.config.keyframe_interval:
            # 保存关键帧
            keyframe_name = f"{self.session.session_id}_kf_{len(self.session.keyframes)}.jpg"
            keyframe_path = self.output_dir / keyframe_name
            
            cv2.imwrite(str(keyframe_path), frame)
            
            self.session.keyframes.append(str(keyframe_path))
            self.last_keyframe_time = current_time
    
    def _save_session_info(self):
        """保存会话信息"""
        if not self.session:
            return
        
        info_path = self.output_dir / f"{self.session.session_id}.json"
        
        with open(info_path, 'w') as f:
            json.dump(self.session.to_dict(), f, indent=2)
    
    def add_event(self, event: Dict):
        """
        添加事件标记
        
        Args:
            event: 事件信息
        """
        if self.session:
            event['timestamp'] = time.time()
            event['relative_time'] = event['timestamp'] - self.session.start_time
            self.session.events.append(event)
    
    def get_status(self) -> Dict:
        """获取录制状态"""
        return {
            'is_recording': self.is_recording,
            'session': self.session.to_dict() if self.session else None,
            'queue_size': self.frame_queue.qsize(),
            'total_sessions': self.total_sessions
        }


class VideoPlayer:
    """
    视频播放器
    
    支持播放、暂停、跳转
    """
    
    def __init__(self):
        """初始化视频播放器"""
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_playing = False
        self.is_paused = False
        
        # 视频信息
        self.fps = 30.0
        self.frame_count = 0
        self.duration = 0.0
        self.current_frame = 0
        self.resolution = (1920, 1080)
        
        # 播放控制
        self.playback_speed = 1.0
        self.loop = False
        
        # 回调
        self.on_frame_callback = None
        self.on_end_callback = None
        
        # 播放线程
        self.play_thread = None
        self.running = False
        
        logger.info("VideoPlayer initialized")
    
    def load(self, filepath: str) -> bool:
        """
        加载视频
        
        Args:
            filepath: 视频文件路径
            
        Returns:
            是否成功
        """
        try:
            self.close()
            
            self.cap = cv2.VideoCapture(filepath)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open video: {filepath}")
                return False
            
            # 获取视频信息
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.duration = self.frame_count / max(self.fps, 1)
            self.resolution = (
                int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
            
            logger.info(f"Loaded video: {filepath} ({self.duration:.1f}s, {self.fps:.1f}fps)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load video: {e}")
            return False
    
    def play(self):
        """开始播放"""
        if self.cap is None:
            return
        
        self.is_playing = True
        self.is_paused = False
        self.running = True
        
        self.play_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.play_thread.start()
    
    def pause(self):
        """暂停播放"""
        self.is_paused = True
    
    def resume(self):
        """恢复播放"""
        self.is_paused = False
    
    def stop(self):
        """停止播放"""
        self.is_playing = False
        self.running = False
        
        if self.play_thread:
            self.play_thread.join(timeout=2.0)
        
        self.current_frame = 0
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def close(self):
        """关闭视频"""
        self.stop()
        
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def seek(self, position: float):
        """
        跳转到指定位置
        
        Args:
            position: 位置（0-1）
        """
        if self.cap is None:
            return
        
        frame_idx = int(position * self.frame_count)
        frame_idx = max(0, min(frame_idx, self.frame_count - 1))
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self.current_frame = frame_idx
    
    def seek_time(self, time_seconds: float):
        """
        跳转到指定时间
        
        Args:
            time_seconds: 时间（秒）
        """
        position = time_seconds / max(self.duration, 1)
        self.seek(position)
    
    def set_speed(self, speed: float):
        """
        设置播放速度
        
        Args:
            speed: 速度倍率
        """
        self.playback_speed = max(0.1, min(speed, 4.0))
    
    def _playback_loop(self):
        """播放循环"""
        frame_interval = 1.0 / max(self.fps, 1)
        
        while self.running and self.is_playing:
            if self.is_paused:
                time.sleep(0.1)
                continue
            
            try:
                ret, frame = self.cap.read()
                
                if ret:
                    self.current_frame += 1
                    
                    if self.on_frame_callback:
                        self.on_frame_callback(frame, self.current_frame / max(self.frame_count, 1))
                    
                    # 控制播放速度
                    time.sleep(frame_interval / self.playback_speed)
                else:
                    # 播放结束
                    if self.loop:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.current_frame = 0
                    else:
                        self.is_playing = False
                        
                        if self.on_end_callback:
                            self.on_end_callback()
                        break
                
            except Exception as e:
                logger.error(f"Playback error: {e}")
                break
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """获取当前帧"""
        if self.cap is None:
            return None
        
        return self.cap.read()[1]
    
    def get_progress(self) -> float:
        """获取播放进度"""
        return self.current_frame / max(self.frame_count, 1)
    
    def get_info(self) -> Dict:
        """获取视频信息"""
        return {
            'fps': self.fps,
            'frame_count': self.frame_count,
            'duration': self.duration,
            'resolution': self.resolution,
            'current_frame': self.current_frame,
            'progress': self.get_progress(),
            'is_playing': self.is_playing,
            'is_paused': self.is_paused
        }


class VideoClipper:
    """
    视频剪辑器
    
    提取视频片段
    """
    
    def __init__(self):
        """初始化视频剪辑器"""
        logger.info("VideoClipper initialized")
    
    def extract_clip(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        end_time: float
    ) -> bool:
        """
        提取视频片段
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            
        Returns:
            是否成功
        """
        try:
            cap = cv2.VideoCapture(input_path)
            
            if not cap.isOpened():
                logger.error(f"Failed to open video: {input_path}")
                return False
            
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 创建输出写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # 跳转到开始位置
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # 提取帧
            frame_count = 0
            while frame_count < end_frame - start_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                writer.write(frame)
                frame_count += 1
            
            cap.release()
            writer.release()
            
            logger.info(f"Extracted clip: {start_time:.1f}s - {end_time:.1f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract clip: {e}")
            return False
    
    def extract_frames(
        self,
        input_path: str,
        output_dir: str,
        interval: float = 1.0
    ) -> List[str]:
        """
        提取视频帧
        
        Args:
            input_path: 输入视频路径
            output_dir: 输出目录
            interval: 提取间隔（秒）
            
        Returns:
            提取的帧文件列表
        """
        frames = []
        
        try:
            cap = cv2.VideoCapture(input_path)
            
            if not cap.isOpened():
                return frames
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps * interval)
            
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            frame_idx = 0
            saved_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    filename = f"frame_{saved_idx:06d}.jpg"
                    filepath = Path(output_dir) / filename
                    cv2.imwrite(str(filepath), frame)
                    frames.append(str(filepath))
                    saved_idx += 1
                
                frame_idx += 1
            
            cap.release()
            
            logger.info(f"Extracted {len(frames)} frames")
            
        except Exception as e:
            logger.error(f"Failed to extract frames: {e}")
        
        return frames


# 全局实例
_video_recorder = None
_video_player = None

def get_video_recorder() -> VideoRecorder:
    """获取视频录制器单例"""
    global _video_recorder
    if _video_recorder is None:
        _video_recorder = VideoRecorder()
    return _video_recorder

def get_video_player() -> VideoPlayer:
    """获取视频播放器单例"""
    global _video_player
    if _video_player is None:
        _video_player = VideoPlayer()
    return _video_player


# 测试代码
if __name__ == '__main__':
    print("Testing Video Recorder...")
    
    recorder = VideoRecorder()
    
    # 开始录制
    recorder.start_recording((640, 480))
    
    # 模拟写入帧
    for i in range(100):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        recorder.write_frame(frame)
        time.sleep(0.03)
    
    # 停止录制
    session = recorder.stop_recording()
    
    if session:
        print(f"Recording saved: {session.file_path}")
        print(f"Duration: {session.duration:.1f}s")
        print(f"Frames: {session.frame_count}")
    
    print("\nDone!")