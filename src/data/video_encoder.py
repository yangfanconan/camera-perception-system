"""
视频流编码模块 - 基于 aiortc 的 H.264 编码
功能：
1. H.264 硬件/软件编码
2. 低延迟流传输
3. 自适应码率控制
4. 支持 WebRTC/HTTP-FLV 输出
"""

import asyncio
import cv2
import numpy as np
from typing import Optional, Tuple, Callable, Awaitable
from pathlib import Path
from loguru import logger
from dataclasses import dataclass
from enum import Enum

# aiortc 相关
try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
    from aiortc.contrib.media import MediaPlayer, MediaRecorder
    from av import VideoFrame
    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False
    logger.warning("aiortc not installed, WebRTC features disabled")

# av (PyAV) 相关
try:
    import av
    from av.codec import Codec
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False
    logger.warning("PyAV not installed, using OpenCV fallback")


class CodecType(Enum):
    """编码器类型"""
    H264 = "h264"
    H265 = "h265"
    VP8 = "vp8"
    VP9 = "vp9"
    JPEG = "jpeg"  # 降级方案


@dataclass
class StreamConfig:
    """流配置"""
    width: int = 1920
    height: int = 1080
    fps: int = 20
    bitrate: int = 2000000  # 2000kbps
    codec: CodecType = CodecType.H264
    gop_size: int = 20  # 关键帧间隔
    preset: str = "ultrafast"  # 编码速度预设
    zero_latency: bool = True  # 零延迟模式


class H264Encoder:
    """H.264 编码器（基于 PyAV）"""
    
    def __init__(self, config: StreamConfig = None):
        """
        初始化 H.264 编码器
        
        Args:
            config: 流配置
        """
        self.config = config or StreamConfig()
        self.codec = None
        self.stream = None
        self.container = None
        self.frame_count = 0
        
        if AV_AVAILABLE:
            self._init_encoder()
            logger.info(f"H264Encoder initialized: {self.config.width}x{self.config.height}"
                       f"@{self.config.fps}fps, {self.config.bitrate//1000}kbps")
    
    def _init_encoder(self):
        """初始化编码器"""
        # 创建输出容器（内存中的 MP4/FLV）
        self.container = av.open(format='mp4', mode='w')
        
        # 创建视频流
        self.stream = self.container.add_stream('h264', rate=self.config.fps)
        self.stream.width = self.config.width
        self.stream.height = self.config.height
        self.stream.bit_rate = self.config.bitrate
        self.stream.gop_size = self.config.gop_size
        
        # 零延迟选项
        if self.config.zero_latency:
            self.stream.options = {
                'preset': self.config.preset,
                'tune': 'zerolatency',
                'profile': 'baseline'
            }
    
    def encode(self, frame: np.ndarray) -> Optional[bytes]:
        """
        编码单帧
        
        Args:
            frame: BGR 帧图像
            
        Returns:
            编码后的数据包
        """
        if not AV_AVAILABLE or self.stream is None:
            return self._encode_fallback(frame)
        
        try:
            # BGR → RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 创建 VideoFrame
            av_frame = VideoFrame.from_ndarray(frame_rgb, format='rgb24')
            av_frame.pts = self.frame_count
            av_frame.time_base = self.stream.codec_context.time_base
            
            # 编码
            packets = self.stream.encode(av_frame)
            
            self.frame_count += 1
            
            # 合并所有输出包
            output = b''
            for packet in packets:
                output += packet.to_bytes()
            
            return output if output else None
        
        except Exception as e:
            logger.error(f"Encoding error: {e}")
            return self._encode_fallback(frame)
    
    def _encode_fallback(self, frame: np.ndarray) -> Optional[bytes]:
        """降级编码方案（JPEG）"""
        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return buffer.tobytes()
        except Exception as e:
            logger.error(f"Fallback encoding error: {e}")
            return None
    
    def flush(self) -> bytes:
        """刷新编码器缓冲区"""
        if not AV_AVAILABLE or self.stream is None:
            return b''
        
        packets = self.stream.encode(None)
        output = b''
        for packet in packets:
            output += packet.to_bytes()
        return output
    
    def close(self):
        """关闭编码器"""
        if self.container:
            self.container.close()
            logger.info("Encoder closed")


class WebRTCTrack(MediaStreamTrack):
    """WebRTC 媒体流轨道"""
    
    kind = "video"
    
    def __init__(self, encoder: H264Encoder):
        super().__init__()
        self.encoder = encoder
        self.frame_queue = asyncio.Queue(maxsize=10)
        self.timestamp = 0
    
    async def recv(self):
        """接收下一帧"""
        try:
            frame_data = await asyncio.wait_for(
                self.frame_queue.get(),
                timeout=1.0
            )
            
            # 创建 VideoFrame
            frame = VideoFrame.from_ndarray(
                frame_data,
                format='rgb24'
            )
            frame.pts = self.timestamp
            frame.time_base = '1/90000'
            
            self.timestamp += 4500  # 90000 / 20fps
            
            return frame
        
        except asyncio.TimeoutError:
            # 超时返回黑屏
            return VideoFrame(np.zeros((480, 640, 3), dtype=np.uint8), format='rgb24')
    
    def send_frame(self, frame_data: np.ndarray):
        """发送帧到队列"""
        try:
            self.frame_queue.put_nowait(frame_data)
        except asyncio.QueueFull:
            pass  # 队列满时丢弃


class WebRTCServer:
    """WebRTC 服务器"""
    
    def __init__(self, config: StreamConfig = None):
        """
        初始化 WebRTC 服务器
        
        Args:
            config: 流配置
        """
        self.config = config or StreamConfig()
        self.encoder = H264Encoder(self.config)
        self.connections = {}
        self.pc_counter = 0
        
        if AIORTC_AVAILABLE:
            logger.info("WebRTC server initialized")
        else:
            logger.warning("aiortc not available, WebRTC disabled")
    
    def create_peer_connection(self) -> RTCPeerConnection:
        """创建对等连接"""
        pc = RTCPeerConnection()
        pc_id = self.pc_counter
        self.pc_counter += 1
        
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection {pc_id} state: {pc.connectionState}")
            
            if pc.connectionState in ["failed", "closed"]:
                await pc.close()
                if pc_id in self.connections:
                    del self.connections[pc_id]
        
        self.connections[pc_id] = pc
        return pc
    
    async def handle_offer(
        self,
        offer: RTCSessionDescription
    ) -> RTCSessionDescription:
        """
        处理 SDP Offer
        
        Args:
            offer: SDP Offer
            
        Returns:
            SDP Answer
        """
        pc = self.create_peer_connection()
        
        # 添加视频轨道
        track = WebRTCTrack(self.encoder)
        pc.addTrack(track)
        
        # 处理 Offer
        await pc.setRemoteDescription(offer)
        
        # 创建 Answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        return answer
    
    def send_frame(self, frame: np.ndarray):
        """发送帧到所有连接"""
        # BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 编码
        encoded = self.encoder.encode(frame_rgb)
        
        # 发送到所有轨道
        for pc in self.connections.values():
            for track in pc.getTransceivers():
                if isinstance(track.sender.track, WebRTCTrack):
                    track.sender.track.send_frame(frame_rgb)


class FLVStreamer:
    """FLV 流推送器（HTTP-FLV 协议）"""
    
    def __init__(self, config: StreamConfig = None):
        """
        初始化 FLV 流推送器
        
        Args:
            config: 流配置
        """
        self.config = config or StreamConfig()
        self.encoder = H264Encoder(self.config)
        
        # FLV 头
        self.flv_header = self._create_flv_header()
        self.flv_started = False
        
        logger.info("FLV streamer initialized")
    
    def _create_flv_header(self) -> bytes:
        """创建 FLV 文件头"""
        # FLV v1, Video+Audio
        header = bytearray()
        header.extend(b'FLV')
        header.append(0x01)  # Version 1
        header.append(0x05)  # HasVideo | HasAudio
        header.extend(b'\x00\x00\x00\x09')  # DataOffset
        header.extend(b'\x00\x00\x00\x00')  # PreviousTagSize0
        return bytes(header)
    
    def _create_flv_tag(
        self,
        data: bytes,
        timestamp: int,
        tag_type: int = 0x09  # Video tag
    ) -> bytes:
        """
        创建 FLV Tag
        
        Args:
            data: 数据
            timestamp: 时间戳（毫秒）
            tag_type: Tag 类型 (0x09=Video, 0x08=Audio)
            
        Returns:
            FLV Tag 数据
        """
        tag = bytearray()
        
        # Tag Type
        tag.append(tag_type)
        
        # DataSize
        data_size = len(data)
        tag.extend(data_size.to_bytes(3, 'big'))
        
        # Timestamp (Extended)
        tag.extend((timestamp & 0xFFFFFF).to_bytes(3, 'big'))
        tag.append((timestamp >> 24) & 0xFF)
        
        # StreamID
        tag.extend(b'\x00\x00\x00')
        
        # Data
        tag.extend(data)
        
        # PreviousTagSize
        tag.extend((11 + data_size).to_bytes(4, 'big'))
        
        return bytes(tag)
    
    def stream_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[int] = None
    ) -> bytes:
        """
        生成单帧 FLV 流
        
        Args:
            frame: BGR 帧图像
            timestamp: 时间戳（毫秒）
            
        Returns:
            FLV 格式数据
        """
        # BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 编码
        encoded = self.encoder.encode(frame_rgb)
        
        if encoded is None:
            return b''
        
        # 时间戳
        if timestamp is None:
            timestamp = int(self.encoder.frame_count * 1000 / self.config.fps)
        
        # 创建 FLV Tag
        flv_tag = self._create_flv_tag(encoded, timestamp)
        
        # 添加 FLV 头（第一帧）
        if not self.flv_started:
            self.flv_started = True
            return self.flv_header + flv_tag
        
        return flv_tag
    
    def reset(self):
        """重置流状态"""
        self.flv_started = False
        if self.encoder:
            self.encoder.frame_count = 0


class MJPEGStreamer:
    """MJPEG 流推送器（降级方案）"""
    
    def __init__(self, quality: int = 85):
        """
        初始化 MJPEG 流推送器
        
        Args:
            quality: JPEG 质量 (1-100)
        """
        self.quality = quality
        logger.info(f"MJPEG streamer initialized (quality={quality})")
    
    def stream_frame(self, frame: np.ndarray) -> bytes:
        """
        生成单帧 MJPEG 数据
        
        Args:
            frame: BGR 帧图像
            
        Returns:
            JPEG 数据
        """
        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            return buffer.tobytes()
        except Exception as e:
            logger.error(f"MJPEG encoding error: {e}")
            return b''


class VideoStreamPipeline:
    """视频流处理管道"""
    
    def __init__(
        self,
        source: int = 0,
        config: StreamConfig = None,
        stream_type: str = 'mjpeg'
    ):
        """
        初始化视频流处理管道
        
        Args:
            source: 摄像头设备 ID
            config: 流配置
            stream_type: 流类型 ('webrtc', 'flv', 'mjpeg')
        """
        self.source = source
        self.config = config or StreamConfig()
        self.stream_type = stream_type
        
        # 选择流推送器
        if stream_type == 'webrtc':
            self.streamer = WebRTCServer(self.config)
        elif stream_type == 'flv':
            self.streamer = FLVStreamer(self.config)
        else:
            self.streamer = MJPEGStreamer()
        
        # 摄像头
        self.cap: Optional[cv2.VideoCapture] = None
        
        # 回调
        self.on_frame: Optional[Callable[[np.ndarray], Awaitable[None]]] = None
        
        # 运行状态
        self.running = False
        
        logger.info(f"VideoStreamPipeline initialized: source={source}, "
                   f"type={stream_type}")
    
    def open_camera(self) -> bool:
        """打开摄像头"""
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera {self.source}")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        logger.info(f"Camera opened: {actual_width}x{actual_height}@{actual_fps}fps")
        return True
    
    async def run(self):
        """运行流处理管道"""
        if not self.cap:
            if not self.open_camera():
                return
        
        self.running = True
        frame_interval = 1.0 / self.config.fps
        
        logger.info(f"Starting video stream pipeline...")
        
        try:
            while self.running:
                import time
                start_time = time.time()
                
                # 读取帧
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning("Failed to read frame")
                    await asyncio.sleep(0.1)
                    continue
                
                # 处理帧（如果有回调）
                if self.on_frame:
                    await self.on_frame(frame)
                
                # 控制帧率
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_interval - elapsed)
                await asyncio.sleep(sleep_time)
        
        except asyncio.CancelledError:
            logger.info("Stream pipeline cancelled")
        finally:
            self.stop()
    
    def stop(self):
        """停止流处理管道"""
        self.running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        logger.info("Stream pipeline stopped")
    
    def set_frame_callback(
        self,
        callback: Callable[[np.ndarray], Awaitable[None]]
    ):
        """设置帧处理回调"""
        self.on_frame = callback


def create_streamer(
    stream_type: str = 'mjpeg',
    config: StreamConfig = None
):
    """
    创建流推送器工厂函数
    
    Args:
        stream_type: 流类型 ('webrtc', 'flv', 'mjpeg')
        config: 流配置
        
    Returns:
        流推送器实例
    """
    if stream_type == 'webrtc':
        return WebRTCServer(config)
    elif stream_type == 'flv':
        return FLVStreamer(config)
    else:
        return MJPEGStreamer()


# ==================== 主函数（测试） ====================

async def main():
    """测试视频流编码"""
    config = StreamConfig(
        width=1920,
        height=1080,
        fps=20,
        bitrate=2000000
    )
    
    # 创建 MJPEG 流推送器（最简单）
    streamer = MJPEGStreamer(quality=85)
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    if not cap.isOpened():
        logger.error("Cannot open camera")
        return
    
    logger.info("Testing MJPEG encoding... Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 编码
            data = streamer.stream_frame(frame)
            
            logger.debug(f"Encoded frame: {len(data)} bytes")
            
            # 显示
            cv2.imshow('Stream Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    asyncio.run(main())
