"""
macOS 优化版视频采集模块
支持 AVFoundation 后端、VideoToolbox 硬件编解码
"""

import cv2
import numpy as np
import platform
from typing import Optional, Generator, Tuple
from pathlib import Path
from loguru import logger

from .algorithms.calibration import CalibrationParams


# macOS 特定优化
IS_MACOS = platform.system() == "Darwin"
IS_APPLE_SILICON = IS_MACOS and platform.machine() == "arm64"


class VideoCaptureMacOS:
    """macOS 优化版视频采集器"""
    
    # macOS 推荐的摄像头分辨率和帧率组合
    MACOS_OPTIMAL_SETTINGS = {
        (1920, 1080): 30,  # Full HD @ 30fps
        (1280, 720): 60,   # HD @ 60fps
        (640, 480): 60,    # VGA @ 60fps
    }
    
    def __init__(
        self,
        camera_id: int = 0,
        resolution: Tuple[int, int] = (1920, 1080),
        fps: int = 30,
        backend: str = 'auto'
    ):
        """
        初始化视频采集器
        
        Args:
            camera_id: 摄像头设备 ID
            resolution: 分辨率 (width, height)
            fps: 帧率
            backend: OpenCV 后端 ('auto', 'avfoundation', 'builtin')
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        self.backend = backend
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.calib_params: Optional[CalibrationParams] = None
        
        # macOS 优化设置
        self.use_avfoundation = IS_MACOS
        self.use_metal = IS_APPLE_SILICON
        
        logger.info(f"VideoCaptureMacOS initialized: camera_id={camera_id}, "
                   f"resolution={resolution}, fps={fps}")
        logger.info(f"Platform: macOS={IS_MACOS}, Apple Silicon={IS_APPLE_SILICON}")
    
    def _get_optimal_backend(self) -> int:
        """获取最优后端"""
        if self.backend == 'avfoundation' or (self.backend == 'auto' and IS_MACOS):
            # AVFoundation 是 macOS 原生后端，性能最好
            logger.info("Using AVFoundation backend")
            return cv2.CAP_AVFOUNDATION
        elif self.backend == 'builtin':
            logger.info("Using builtin backend")
            return cv2.CAP_ANY
        else:
            return cv2.CAP_ANY
    
    def open(self) -> bool:
        """打开摄像头"""
        backend = self._get_optimal_backend()
        
        self.cap = cv2.VideoCapture(self.camera_id, backend)
        
        if not self.cap.isOpened():
            # 尝试回退到默认后端
            logger.warning("AVFoundation failed, trying builtin backend")
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False
        
        # 设置分辨率和帧率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # macOS 特定优化
        if IS_MACOS:
            # 设置缓冲区大小（减少延迟）
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # 尝试设置硬件加速
            try:
                self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, 1)
            except Exception:
                pass
        
        # 获取实际设置
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        logger.info(f"Camera opened: {actual_width}x{actual_height}@{actual_fps}fps")
        
        # 检查是否达到目标设置
        if actual_width != self.resolution[0] or actual_height != self.resolution[1]:
            logger.warning(f"Requested {self.resolution}, got {actual_width}x{actual_height}")
        
        return True
    
    def set_calibration(self, calib_params: CalibrationParams) -> None:
        """设置标定参数"""
        self.calib_params = calib_params
        logger.info("Calibration parameters set")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """读取一帧"""
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        
        if not ret:
            logger.warning("Failed to read frame")
            return False, None
        
        # 去畸变
        if self.calib_params is not None:
            frame = self.undistort(frame)
        
        return True, frame
    
    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """去畸变"""
        if self.calib_params is None:
            return frame
        
        camera_matrix = self.calib_params.get_camera_matrix()
        dist_coeffs = self.calib_params.get_dist_coeffs()
        
        return cv2.undistort(frame, camera_matrix, dist_coeffs)
    
    def read_stream(self) -> Generator[np.ndarray, None, None]:
        """视频流生成器"""
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


class VideoEncoderMacOS:
    """macOS 优化版视频编码器（VideoToolbox 硬件加速）"""
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (1920, 1080),
        fps: int = 30,
        bitrate: int = 3000000,
        use_videotoolbox: bool = True
    ):
        """
        初始化视频编码器
        
        Args:
            resolution: 分辨率
            fps: 帧率
            bitrate: 码率 (bps)
            use_videotoolbox: 是否使用 VideoToolbox 硬件加速
        """
        self.resolution = resolution
        self.fps = fps
        self.bitrate = bitrate
        self.use_videotoolbox = use_videotoolbox and IS_MACOS
        
        self.codec = None
        self.out = None
        
        if self.use_videotoolbox:
            logger.info("VideoToolbox hardware encoding enabled")
        else:
            logger.info("Using software encoding")
    
    def open(self, output_path: str = '') -> bool:
        """
        打开编码器
        
        Args:
            output_path: 输出文件路径（空字符串表示内存输出）
        """
        if self.use_videotoolbox:
            # 尝试使用 H.264 + VideoToolbox
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'X264')
        
        try:
            self.out = cv2.VideoWriter(
                output_path,
                fourcc,
                self.fps,
                self.resolution,
                True
            )
            return self.out.isOpened()
        except Exception as e:
            logger.error(f"Failed to open encoder: {e}")
            return False
    
    def write(self, frame: np.ndarray) -> bool:
        """写入帧"""
        if self.out is None:
            return False
        
        self.out.write(frame)
        return True
    
    def release(self) -> None:
        """释放编码器"""
        if self.out is not None:
            self.out.release()
            logger.info("Encoder released")


def list_cameras_macos() -> list:
    """
    列出 macOS 上可用的摄像头
    
    Returns:
        摄像头设备 ID 列表
    """
    if not IS_MACOS:
        return [0]
    
    cameras = []
    
    # 尝试打开不同的摄像头 ID
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            cameras.append(i)
            cap.release()
    
    logger.info(f"Available cameras: {cameras}")
    return cameras


def get_optimal_settings_for_camera(
    camera_id: int = 0
) -> Tuple[Tuple[int, int], int]:
    """
    获取摄像头的最优设置
    
    Returns:
        (resolution, fps) 最优分辨率和帧率
    """
    cap = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION if IS_MACOS else cv2.CAP_ANY)
    
    if not cap.isOpened():
        return (1920, 1080), 30
    
    # 测试不同设置
    best_resolution = (1920, 1080)
    best_fps = 30
    
    for resolution, target_fps in VideoCaptureMacOS.MACOS_OPTIMAL_SETTINGS.items():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        cap.set(cv2.CAP_PROP_FPS, target_fps)
        
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        if actual_fps > 0:
            best_resolution = resolution
            best_fps = int(actual_fps)
            break
    
    cap.release()
    
    return best_resolution, best_fps


# 兼容性别名
VideoCapture = VideoCaptureMacOS


def main():
    """测试 macOS 优化视频采集"""
    import argparse
    
    parser = argparse.ArgumentParser(description='macOS 视频采集测试')
    parser.add_argument('--camera', '-c', type=int, default=0, help='摄像头 ID')
    parser.add_argument('--resolution', '-r', type=int, nargs=2, default=[1920, 1080])
    parser.add_argument('--fps', '-f', type=int, default=30)
    parser.add_argument('--backend', choices=['auto', 'avfoundation', 'builtin'], default='auto')
    
    args = parser.parse_args()
    
    # 列出可用摄像头
    if IS_MACOS:
        cameras = list_cameras_macos()
        logger.info(f"Available cameras: {cameras}")
        
        # 获取最优设置
        optimal_res, optimal_fps = get_optimal_settings_for_camera(args.camera)
        logger.info(f"Optimal settings: {optimal_res[0]}x{optimal_res[1]}@{optimal_fps}fps")
    
    # 创建采集器
    cap = VideoCaptureMacOS(
        camera_id=args.camera,
        resolution=tuple(args.resolution),
        fps=args.fps,
        backend=args.backend
    )
    
    if not cap.open():
        logger.error("Failed to open camera")
        return
    
    logger.info("Press 'q' to quit")
    
    frame_count = 0
    start_time = cv2.getTickCount()
    
    try:
        for frame in cap.read_stream():
            # 计算 FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
                if elapsed > 0:
                    fps = frame_count / elapsed
                    logger.info(f"Actual FPS: {fps:.1f}")
            
            cv2.imshow('macOS Video Capture', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
