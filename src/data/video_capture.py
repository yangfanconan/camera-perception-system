"""
视频采集模块
功能：
1. 摄像头视频流采集
2. 帧预处理（去畸变、缩放）
3. 视频流编码输出
"""

import cv2
import numpy as np
from typing import Optional, Generator, Tuple
from pathlib import Path
from loguru import logger

from src.algorithms.calibration import CalibrationParams
from src.utils.constants import (
    DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT,
    COMMON_RESOLUTIONS
)


class VideoCapture:
    """视频采集器"""
    
    def __init__(
        self,
        camera_id: int = 0,
        resolution: Tuple[int, int] = (DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT),
        fps: int = 20
    ):
        """
        初始化视频采集器
        
        Args:
            camera_id: 摄像头设备 ID
            resolution: 分辨率 (width, height)
            fps: 帧率
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.calib_params: Optional[CalibrationParams] = None
        
        logger.info(f"VideoCapture initialized: camera_id={camera_id}, "
                   f"resolution={resolution}, fps={fps}")
    
    def open(self) -> bool:
        """打开摄像头"""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera {self.camera_id}")
            return False
        
        # 设置分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # 获取实际分辨率
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        logger.info(f"Camera opened: {actual_width}x{actual_height}@{actual_fps}fps")
        
        return True
    
    def set_calibration(self, calib_params: CalibrationParams) -> None:
        """设置标定参数（用于去畸变）"""
        self.calib_params = calib_params
        logger.info("Calibration parameters set")
    
    def is_opened(self) -> bool:
        """检查摄像头是否已打开"""
        return self.cap is not None and self.cap.isOpened()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        读取一帧（带异常处理）

        Returns:
            (success, frame): 是否成功，帧图像
        """
        if self.cap is None:
            return False, None

        try:
            ret, frame = self.cap.read()

            if not ret:
                logger.warning("Failed to read frame from camera")
                return False, None

            # 检查帧是否有效
            if frame is None or frame.size == 0:
                logger.warning("Invalid frame received")
                return False, None

            # 去畸变
            if self.calib_params is not None:
                try:
                    frame = self.undistort(frame)
                except Exception as e:
                    logger.error(f"Undistort failed: {e}")
                    # 返回原始帧

            return True, frame
            
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return False, None
    
    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """去畸变"""
        if self.calib_params is None:
            return frame
        
        camera_matrix = self.calib_params.get_camera_matrix()
        dist_coeffs = self.calib_params.get_dist_coeffs()
        
        return cv2.undistort(frame, camera_matrix, dist_coeffs)
    
    def read_stream(self) -> Generator[np.ndarray, None, None]:
        """
        视频流生成器
        
        Yields:
            帧图像
        """
        while True:
            ret, frame = self.read()
            if not ret:
                break
            yield frame
    
    def release(self) -> None:
        """释放摄像头"""
        if self.cap is not None:
            self.cap.release()
            logger.info("Camera released")
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class FrameEncoder:
    """帧编码器（用于视频流编码）"""
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (1920, 1080),
        fps: int = 20,
        bitrate: int = 2000000  # 2000kbps
    ):
        """
        初始化帧编码器
        
        Args:
            resolution: 分辨率
            fps: 帧率
            bitrate: 码率 (bps)
        """
        self.resolution = resolution
        self.fps = fps
        self.bitrate = bitrate
        
        # 编码器配置
        self.codec = cv2.VideoWriter_fourcc(*'H264')
        
        logger.info(f"FrameEncoder initialized: {resolution}@{fps}fps, "
                   f"bitrate={bitrate//1000}kbps")
    
    def encode(self, frame: np.ndarray) -> Optional[bytes]:
        """
        编码单帧
        
        Args:
            frame: BGR 帧图像
            
        Returns:
            编码后的数据（H.264 裸流）
        """
        # 注意：OpenCV 的 VideoWriter 不适合实时流编码
        # 实际项目中应使用 FFmpeg 或 aiortc 进行编码
        # 这里仅作为示例框架
        
        # 转换为 RGB（Web 端要求）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 实际编码逻辑（待实现，使用 aiortc 或 FFmpeg）
        # 这里返回原始数据作为占位
        return frame_rgb.tobytes()
    
    def encode_to_flv(self, frame: np.ndarray) -> Optional[bytes]:
        """
        编码为 FLV 格式
        
        Args:
            frame: BGR 帧图像
            
        Returns:
            FLV 格式数据
        """
        # 实际项目中应使用 qwencode 或 FFmpeg
        # 这里仅作为示例框架
        logger.debug("FLV encoding (placeholder)")
        return None


def main():
    """测试视频采集"""
    import argparse
    
    parser = argparse.ArgumentParser(description='视频采集测试')
    parser.add_argument('--camera', '-c', type=int, default=0,
                       help='摄像头设备 ID')
    parser.add_argument('--resolution', '-r', type=int, nargs=2,
                       default=[1920, 1080],
                       help='分辨率 [width, height]')
    parser.add_argument('--fps', '-f', type=int, default=20,
                       help='帧率')
    
    args = parser.parse_args()
    
    # 打开摄像头
    cap = VideoCapture(
        camera_id=args.camera,
        resolution=tuple(args.resolution),
        fps=args.fps
    )
    
    if not cap.open():
        return
    
    logger.info("Press 'q' to quit")
    
    try:
        for frame in cap.read_stream():
            cv2.imshow('Video Stream', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
