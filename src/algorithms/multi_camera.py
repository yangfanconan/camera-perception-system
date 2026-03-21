"""
多摄像头管理模块

功能：
1. 多路摄像头管理
2. 摄像头切换
3. 多画面显示
4. 同步录制
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
from loguru import logger
import time
import threading
import queue


@dataclass
class CameraInfo:
    """摄像头信息"""
    camera_id: int
    name: str
    source: str           # 设备路径或RTSP地址
    resolution: Tuple[int, int]
    fps: float
    is_opened: bool = False
    last_frame_time: float = 0
    frame_count: int = 0
    error_count: int = 0


@dataclass
class CameraConfig:
    """摄像头配置"""
    camera_id: int
    source: str
    name: str = ""
    enabled: bool = True
    resolution: Tuple[int, int] = (1920, 1080)
    fps: int = 30
    buffer_size: int = 1


class CameraCapture:
    """
    摄像头捕获类
    
    单个摄像头的捕获管理
    """
    
    def __init__(self, config: CameraConfig):
        """
        初始化摄像头捕获
        
        Args:
            config: 摄像头配置
        """
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False
        
        # 帧缓冲
        self.frame_buffer: queue.Queue = queue.Queue(maxsize=config.buffer_size)
        
        # 捕获线程
        self.running = False
        self.capture_thread = None
        
        # 统计
        self.frame_count = 0
        self.last_frame_time = 0
        self.error_count = 0
        
        logger.info(f"CameraCapture initialized: {config.name} (source={config.source})")
    
    def open(self) -> bool:
        """打开摄像头"""
        if self.is_opened:
            return True
        
        try:
            # 尝试不同的源类型
            if isinstance(self.config.source, int):
                self.cap = cv2.VideoCapture(self.config.source)
            elif self.config.source.startswith('rtsp://') or self.config.source.startswith('http://'):
                self.cap = cv2.VideoCapture(self.config.source)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                self.cap = cv2.VideoCapture(self.config.source)
            
            if self.cap.isOpened():
                # 设置分辨率
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
                self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
                
                self.is_opened = True
                self.running = True
                
                # 启动捕获线程
                self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
                self.capture_thread.start()
                
                logger.info(f"Camera opened: {self.config.name}")
                return True
            else:
                logger.error(f"Failed to open camera: {self.config.name}")
                return False
                
        except Exception as e:
            logger.error(f"Camera open error: {e}")
            return False
    
    def close(self):
        """关闭摄像头"""
        self.running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.is_opened = False
        logger.info(f"Camera closed: {self.config.name}")
    
    def _capture_loop(self):
        """捕获循环"""
        while self.running and self.is_opened:
            try:
                ret, frame = self.cap.read()
                
                if ret:
                    self.frame_count += 1
                    self.last_frame_time = time.time()
                    
                    # 放入缓冲区
                    try:
                        self.frame_buffer.put(frame, block=False)
                    except queue.Full:
                        # 丢弃旧帧
                        try:
                            self.frame_buffer.get_nowait()
                            self.frame_buffer.put(frame, block=False)
                        except:
                            pass
                else:
                    self.error_count += 1
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Capture error: {e}")
                self.error_count += 1
                time.sleep(0.1)
    
    def read(self) -> Optional[np.ndarray]:
        """读取帧"""
        if not self.is_opened:
            return None
        
        try:
            return self.frame_buffer.get(timeout=1.0)
        except queue.Empty:
            return None
    
    def get_info(self) -> CameraInfo:
        """获取摄像头信息"""
        return CameraInfo(
            camera_id=self.config.camera_id,
            name=self.config.name,
            source=self.config.source,
            resolution=self.config.resolution,
            fps=self.config.fps,
            is_opened=self.is_opened,
            last_frame_time=self.last_frame_time,
            frame_count=self.frame_count,
            error_count=self.error_count
        )


class MultiCameraManager:
    """
    多摄像头管理器
    
    管理多个摄像头
    """
    
    MAX_CAMERAS = 8
    
    def __init__(self):
        """初始化多摄像头管理器"""
        self.cameras: OrderedDict[int, CameraCapture] = OrderedDict()
        self.active_camera_id: Optional[int] = None
        
        # 帧回调
        self.on_frame_callback = None
        
        # 统计
        self.total_frames = 0
        
        logger.info("MultiCameraManager initialized")
    
    def add_camera(self, config: CameraConfig) -> bool:
        """
        添加摄像头
        
        Args:
            config: 摄像头配置
            
        Returns:
            是否成功
        """
        if len(self.cameras) >= self.MAX_CAMERAS:
            logger.warning(f"Maximum cameras reached: {self.MAX_CAMERAS}")
            return False
        
        if config.camera_id in self.cameras:
            logger.warning(f"Camera already exists: {config.camera_id}")
            return False
        
        camera = CameraCapture(config)
        self.cameras[config.camera_id] = camera
        
        logger.info(f"Added camera: {config.name} (id={config.camera_id})")
        return True
    
    def remove_camera(self, camera_id: int) -> bool:
        """移除摄像头"""
        if camera_id not in self.cameras:
            return False
        
        camera = self.cameras[camera_id]
        camera.close()
        del self.cameras[camera_id]
        
        if self.active_camera_id == camera_id:
            self.active_camera_id = None
        
        logger.info(f"Removed camera: {camera_id}")
        return True
    
    def open_camera(self, camera_id: int) -> bool:
        """打开摄像头"""
        if camera_id not in self.cameras:
            return False
        
        return self.cameras[camera_id].open()
    
    def close_camera(self, camera_id: int):
        """关闭摄像头"""
        if camera_id in self.cameras:
            self.cameras[camera_id].close()
    
    def open_all(self) -> Dict[int, bool]:
        """打开所有摄像头"""
        results = {}
        for camera_id in self.cameras:
            results[camera_id] = self.open_camera(camera_id)
        return results
    
    def close_all(self):
        """关闭所有摄像头"""
        for camera in self.cameras.values():
            camera.close()
    
    def set_active(self, camera_id: int) -> bool:
        """设置活动摄像头"""
        if camera_id in self.cameras:
            self.active_camera_id = camera_id
            return True
        return False
    
    def read_active(self) -> Optional[np.ndarray]:
        """读取活动摄像头帧"""
        if self.active_camera_id is None:
            return None
        
        return self.cameras[self.active_camera_id].read()
    
    def read_all(self) -> Dict[int, np.ndarray]:
        """读取所有摄像头帧"""
        frames = {}
        for camera_id, camera in self.cameras.items():
            if camera.is_opened:
                frame = camera.read()
                if frame is not None:
                    frames[camera_id] = frame
        return frames
    
    def get_camera_info(self, camera_id: int) -> Optional[CameraInfo]:
        """获取摄像头信息"""
        if camera_id in self.cameras:
            return self.cameras[camera_id].get_info()
        return None
    
    def get_all_info(self) -> List[CameraInfo]:
        """获取所有摄像头信息"""
        return [camera.get_info() for camera in self.cameras.values()]
    
    def get_active_info(self) -> Optional[CameraInfo]:
        """获取活动摄像头信息"""
        if self.active_camera_id is not None:
            return self.get_camera_info(self.active_camera_id)
        return None
    
    def create_mosaic(
        self,
        frames: Dict[int, np.ndarray],
        grid_size: Tuple[int, int] = None
    ) -> np.ndarray:
        """
        创建多画面拼接
        
        Args:
            frames: 帧字典
            grid_size: 网格大小 (cols, rows)
            
        Returns:
            拼接后的图像
        """
        if not frames:
            return None
        
        n_frames = len(frames)
        
        # 自动计算网格大小
        if grid_size is None:
            cols = int(np.ceil(np.sqrt(n_frames)))
            rows = int(np.ceil(n_frames / cols))
        else:
            cols, rows = grid_size
        
        # 获取帧尺寸
        first_frame = list(frames.values())[0]
        h, w = first_frame.shape[:2]
        
        # 缩放尺寸
        target_w = 640
        target_h = int(h * target_w / w)
        
        # 创建画布
        canvas = np.zeros((target_h * rows, target_w * cols, 3), dtype=np.uint8)
        
        # 放置帧
        for i, (camera_id, frame) in enumerate(frames.items()):
            row = i // cols
            col = i % cols
            
            # 缩放
            resized = cv2.resize(frame, (target_w, target_h))
            
            # 添加标签
            label = f"Camera {camera_id}"
            cv2.putText(resized, label, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 放置到画布
            y1 = row * target_h
            y2 = y1 + target_h
            x1 = col * target_w
            x2 = x1 + target_w
            
            canvas[y1:y2, x1:x2] = resized
        
        return canvas
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_cameras': len(self.cameras),
            'active_cameras': sum(1 for c in self.cameras.values() if c.is_opened),
            'active_camera_id': self.active_camera_id,
            'total_frames': sum(c.frame_count for c in self.cameras.values())
        }


class CameraSwitcher:
    """
    摄像头切换器
    
    自动或手动切换摄像头
    """
    
    def __init__(self, manager: MultiCameraManager):
        """
        初始化切换器
        
        Args:
            manager: 摄像头管理器
        """
        self.manager = manager
        
        # 自动切换
        self.auto_switch = False
        self.switch_interval = 5.0  # 秒
        self.last_switch_time = 0
        
        # 切换回调
        self.on_switch_callback = None
        
        logger.info("CameraSwitcher initialized")
    
    def switch_next(self) -> Optional[int]:
        """切换到下一个摄像头"""
        camera_ids = list(self.manager.cameras.keys())
        
        if not camera_ids:
            return None
        
        if self.manager.active_camera_id is None:
            next_id = camera_ids[0]
        else:
            current_idx = camera_ids.index(self.manager.active_camera_id)
            next_idx = (current_idx + 1) % len(camera_ids)
            next_id = camera_ids[next_idx]
        
        self.manager.set_active(next_id)
        self.last_switch_time = time.time()
        
        if self.on_switch_callback:
            self.on_switch_callback(next_id)
        
        return next_id
    
    def switch_prev(self) -> Optional[int]:
        """切换到上一个摄像头"""
        camera_ids = list(self.manager.cameras.keys())
        
        if not camera_ids:
            return None
        
        if self.manager.active_camera_id is None:
            prev_id = camera_ids[-1]
        else:
            current_idx = camera_ids.index(self.manager.active_camera_id)
            prev_idx = (current_idx - 1) % len(camera_ids)
            prev_id = camera_ids[prev_idx]
        
        self.manager.set_active(prev_id)
        self.last_switch_time = time.time()
        
        if self.on_switch_callback:
            self.on_switch_callback(prev_id)
        
        return prev_id
    
    def switch_to(self, camera_id: int) -> bool:
        """切换到指定摄像头"""
        if self.manager.set_active(camera_id):
            self.last_switch_time = time.time()
            
            if self.on_switch_callback:
                self.on_switch_callback(camera_id)
            
            return True
        return False
    
    def enable_auto_switch(self, interval: float = 5.0):
        """启用自动切换"""
        self.auto_switch = True
        self.switch_interval = interval
        logger.info(f"Auto switch enabled (interval={interval}s)")
    
    def disable_auto_switch(self):
        """禁用自动切换"""
        self.auto_switch = False
        logger.info("Auto switch disabled")
    
    def update(self) -> Optional[int]:
        """
        更新切换状态
        
        Returns:
            如果发生切换，返回新的摄像头ID
        """
        if not self.auto_switch:
            return None
        
        if time.time() - self.last_switch_time >= self.switch_interval:
            return self.switch_next()
        
        return None


# 全局实例
_multi_camera_manager = None

def get_multi_camera_manager() -> MultiCameraManager:
    """获取多摄像头管理器单例"""
    global _multi_camera_manager
    if _multi_camera_manager is None:
        _multi_camera_manager = MultiCameraManager()
    return _multi_camera_manager


# 测试代码
if __name__ == '__main__':
    print("Testing Multi-Camera Manager...")
    
    manager = MultiCameraManager()
    
    # 添加摄像头
    manager.add_camera(CameraConfig(
        camera_id=0,
        source=0,
        name="Webcam 1"
    ))
    
    # 获取信息
    print("Camera info:", manager.get_all_info())
    
    # 打开摄像头
    manager.open_camera(0)
    
    # 读取帧
    frame = manager.read_active()
    if frame is not None:
        print(f"Frame shape: {frame.shape}")
    
    # 关闭
    manager.close_all()
    
    print("\nDone!")