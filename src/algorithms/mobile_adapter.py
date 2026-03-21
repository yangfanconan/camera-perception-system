"""
移动端适配模块

功能：
1. 移动端检测
2. 响应式布局适配
3. 触摸手势支持
4. 性能优化
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from loguru import logger
import time
import re


@dataclass
class DeviceInfo:
    """设备信息"""
    user_agent: str = ""
    is_mobile: bool = False
    is_tablet: bool = False
    is_desktop: bool = True
    os: str = "unknown"  # ios, android, windows, macos, linux
    browser: str = "unknown"
    screen_width: int = 1920
    screen_height: int = 1080
    pixel_ratio: float = 1.0
    touch_support: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'is_mobile': self.is_mobile,
            'is_tablet': self.is_tablet,
            'is_desktop': self.is_desktop,
            'os': self.os,
            'browser': self.browser,
            'screen': f"{self.screen_width}x{self.screen_height}",
            'pixel_ratio': self.pixel_ratio,
            'touch_support': self.touch_support
        }


class DeviceDetector:
    """
    设备检测器
    
    检测客户端设备类型
    """
    
    # 移动设备正则
    MOBILE_PATTERNS = [
        r'Android',
        r'iPhone',
        r'iPod',
        r'Windows Phone',
        r'BlackBerry',
        r'webOS'
    ]
    
    # 平板正则
    TABLET_PATTERNS = [
        r'iPad',
        r'Android(?!.*Mobile)',
        r'Tablet',
        r'Kindle'
    ]
    
    # OS 正则
    OS_PATTERNS = {
        'ios': [r'iPhone', r'iPad', r'iPod'],
        'android': [r'Android'],
        'windows': [r'Windows NT', r'Windows Phone'],
        'macos': [r'Macintosh', r'Mac OS X'],
        'linux': [r'Linux']
    }
    
    # 浏览器正则
    BROWSER_PATTERNS = {
        'chrome': [r'Chrome/(?!.*Edg)'],
        'safari': [r'Safari/(?!.*Chrome)'],
        'firefox': [r'Firefox/'],
        'edge': [r'Edg/'],
        'opera': [r'OPR/', r'Opera']
    }
    
    def detect(self, user_agent: str) -> DeviceInfo:
        """
        检测设备
        
        Args:
            user_agent: User-Agent 字符串
            
        Returns:
            DeviceInfo: 设备信息
        """
        info = DeviceInfo(user_agent=user_agent)
        
        # 检测移动设备
        for pattern in self.MOBILE_PATTERNS:
            if re.search(pattern, user_agent, re.IGNORECASE):
                info.is_mobile = True
                info.is_desktop = False
                break
        
        # 检测平板
        for pattern in self.TABLET_PATTERNS:
            if re.search(pattern, user_agent, re.IGNORECASE):
                info.is_tablet = True
                info.is_mobile = False
                info.is_desktop = False
                break
        
        # 检测 OS
        for os_name, patterns in self.OS_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, user_agent, re.IGNORECASE):
                    info.os = os_name
                    break
            if info.os != 'unknown':
                break
        
        # 检测浏览器
        for browser_name, patterns in self.BROWSER_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, user_agent, re.IGNORECASE):
                    info.browser = browser_name
                    break
            if info.browser != 'unknown':
                break
        
        return info
    
    def detect_from_headers(self, headers: Dict) -> DeviceInfo:
        """
        从请求头检测设备
        
        Args:
            headers: HTTP 请求头
            
        Returns:
            DeviceInfo: 设备信息
        """
        user_agent = headers.get('user-agent', '')
        info = self.detect(user_agent)
        
        # 检测触摸支持
        if headers.get('touch-support') == 'true':
            info.touch_support = True
        
        # 解析屏幕尺寸
        screen_info = headers.get('screen-info', '')
        if screen_info:
            try:
                parts = screen_info.split('x')
                if len(parts) >= 2:
                    info.screen_width = int(parts[0])
                    info.screen_height = int(parts[1])
                if len(parts) >= 3:
                    info.pixel_ratio = float(parts[2])
            except:
                pass
        
        return info


class ResponsiveConfig:
    """
    响应式配置
    
    根据设备类型调整配置
    """
    
    # 断点
    BREAKPOINTS = {
        'xs': 576,    # 手机
        'sm': 768,    # 平板竖屏
        'md': 992,    # 平板横屏
        'lg': 1200,   # 桌面
        'xl': 1920    # 大屏
    }
    
    # 预设配置
    PRESETS = {
        'mobile': {
            'video_width': 640,
            'video_height': 480,
            'fps': 15,
            'detection_interval': 3,
            'show_trajectory': False,
            'show_velocity': False,
            'heatmap_resolution': 0.5
        },
        'tablet': {
            'video_width': 1280,
            'video_height': 720,
            'fps': 20,
            'detection_interval': 2,
            'show_trajectory': True,
            'show_velocity': False,
            'heatmap_resolution': 0.75
        },
        'desktop': {
            'video_width': 1920,
            'video_height': 1080,
            'fps': 30,
            'detection_interval': 1,
            'show_trajectory': True,
            'show_velocity': True,
            'heatmap_resolution': 1.0
        }
    }
    
    def __init__(self):
        """初始化响应式配置"""
        self.current_config = self.PRESETS['desktop'].copy()
        
        logger.info("ResponsiveConfig initialized")
    
    def get_config(self, device_info: DeviceInfo) -> Dict:
        """
        获取适配配置
        
        Args:
            device_info: 设备信息
            
        Returns:
            配置字典
        """
        if device_info.is_mobile:
            self.current_config = self.PRESETS['mobile'].copy()
        elif device_info.is_tablet:
            self.current_config = self.PRESETS['tablet'].copy()
        else:
            self.current_config = self.PRESETS['desktop'].copy()
        
        # 根据屏幕尺寸微调
        if device_info.screen_width < self.BREAKPOINTS['sm']:
            self.current_config['video_width'] = min(
                self.current_config['video_width'],
                device_info.screen_width
            )
        
        return self.current_config
    
    def get_breakpoint(self, width: int) -> str:
        """
        获取断点名称
        
        Args:
            width: 屏幕宽度
            
        Returns:
            断点名称
        """
        if width < self.BREAKPOINTS['xs']:
            return 'xs'
        elif width < self.BREAKPOINTS['sm']:
            return 'sm'
        elif width < self.BREAKPOINTS['md']:
            return 'md'
        elif width < self.BREAKPOINTS['lg']:
            return 'lg'
        else:
            return 'xl'


class TouchGestureHandler:
    """
    触摸手势处理器
    
    处理移动端触摸手势
    """
    
    def __init__(self):
        """初始化触摸手势处理器"""
        self.gestures: Dict[str, callable] = {}
        self.touch_start: Optional[Dict] = None
        self.touch_history: List[Dict] = []
        
        # 手势参数
        self.tap_threshold = 10  # 像素
        self.swipe_threshold = 50  # 像素
        self.pinch_threshold = 10  # 像素变化
        
        logger.info("TouchGestureHandler initialized")
    
    def register_gesture(self, gesture_name: str, callback: callable):
        """
        注册手势回调
        
        Args:
            gesture_name: 手势名称 (tap, swipe_left, swipe_right, pinch, zoom)
            callback: 回调函数
        """
        self.gestures[gesture_name] = callback
    
    def touch_start_event(self, x: float, y: float, timestamp: float = None):
        """
        触摸开始事件
        
        Args:
            x: X 坐标
            y: Y 坐标
            timestamp: 时间戳
        """
        self.touch_start = {
            'x': x,
            'y': y,
            'timestamp': timestamp or time.time()
        }
        self.touch_history = [self.touch_start.copy()]
    
    def touch_move_event(self, x: float, y: float):
        """触摸移动事件"""
        if self.touch_start:
            self.touch_history.append({
                'x': x,
                'y': y,
                'timestamp': time.time()
            })
    
    def touch_end_event(self, x: float, y: float):
        """
        触摸结束事件
        
        Args:
            x: X 坐标
            y: Y 坐标
        """
        if not self.touch_start:
            return
        
        # 计算移动距离
        dx = x - self.touch_start['x']
        dy = y - self.touch_start['y']
        distance = (dx**2 + dy**2) ** 0.5
        
        # 判断手势
        if distance < self.tap_threshold:
            # 点击
            self._trigger_gesture('tap', {'x': x, 'y': y})
        
        elif abs(dx) > self.swipe_threshold and abs(dx) > abs(dy):
            # 水平滑动
            if dx > 0:
                self._trigger_gesture('swipe_right', {'dx': dx})
            else:
                self._trigger_gesture('swipe_left', {'dx': dx})
        
        elif abs(dy) > self.swipe_threshold and abs(dy) > abs(dx):
            # 垂直滑动
            if dy > 0:
                self._trigger_gesture('swipe_down', {'dy': dy})
            else:
                self._trigger_gesture('swipe_up', {'dy': dy})
        
        # 清理
        self.touch_start = None
        self.touch_history = []
    
    def _trigger_gesture(self, gesture_name: str, data: Dict):
        """触发的手势"""
        if gesture_name in self.gestures:
            try:
                self.gestures[gesture_name](data)
            except Exception as e:
                logger.error(f"Gesture callback error: {e}")


class MobileOptimizer:
    """
    移动端优化器
    
    优化移动端性能
    """
    
    def __init__(self):
        """初始化优化器"""
        self.optimizations = {
            'reduce_fps': True,
            'lower_resolution': True,
            'skip_frames': True,
            'compress_images': True,
            'lazy_loading': True
        }
        
        self.stats = {
            'frames_skipped': 0,
            'bytes_saved': 0
        }
        
        logger.info("MobileOptimizer initialized")
    
    def optimize_frame(self, frame_data: bytes, device_info: DeviceInfo) -> bytes:
        """
        优化帧数据
        
        Args:
            frame_data: 帧数据
            device_info: 设备信息
            
        Returns:
            优化后的数据
        """
        if not device_info.is_mobile:
            return frame_data
        
        # 移动端压缩
        if self.optimizations['compress_images']:
            # 这里可以添加图像压缩逻辑
            pass
        
        return frame_data
    
    def should_skip_frame(self, device_info: DeviceInfo, fps: float) -> bool:
        """
        判断是否跳过帧
        
        Args:
            device_info: 设备信息
            fps: 当前 FPS
            
        Returns:
            是否跳过
        """
        if not device_info.is_mobile:
            return False
        
        # 移动端降低帧率
        target_fps = 15 if device_info.is_mobile else 30
        
        if fps > target_fps:
            self.stats['frames_skipped'] += 1
            return True
        
        return False
    
    def get_optimized_settings(self, device_info: DeviceInfo) -> Dict:
        """
        获取优化设置
        
        Args:
            device_info: 设备信息
            
        Returns:
            设置字典
        """
        if device_info.is_mobile:
            return {
                'video_quality': 'medium',
                'detection_fps': 10,
                'stream_fps': 15,
                'resolution_scale': 0.5,
                'enable_cache': True
            }
        elif device_info.is_tablet:
            return {
                'video_quality': 'high',
                'detection_fps': 15,
                'stream_fps': 20,
                'resolution_scale': 0.75,
                'enable_cache': True
            }
        else:
            return {
                'video_quality': 'best',
                'detection_fps': 30,
                'stream_fps': 30,
                'resolution_scale': 1.0,
                'enable_cache': False
            }
    
    def get_stats(self) -> Dict:
        """获取统计"""
        return self.stats.copy()


class MobileAdapter:
    """
    移动端适配器
    
    整合移动端相关功能
    """
    
    def __init__(self):
        """初始化适配器"""
        self.detector = DeviceDetector()
        self.responsive = ResponsiveConfig()
        self.touch_handler = TouchGestureHandler()
        self.optimizer = MobileOptimizer()
        
        logger.info("MobileAdapter initialized")
    
    def adapt(self, user_agent: str, headers: Dict = None) -> Dict:
        """
        适配请求
        
        Args:
            user_agent: User-Agent
            headers: 请求头
            
        Returns:
            适配结果
        """
        # 检测设备
        device_info = self.detector.detect(user_agent)
        
        if headers:
            device_info = self.detector.detect_from_headers(headers)
        
        # 获取配置
        config = self.responsive.get_config(device_info)
        
        # 获取优化设置
        optimizations = self.optimizer.get_optimized_settings(device_info)
        
        return {
            'device': device_info.to_dict(),
            'config': config,
            'optimizations': optimizations
        }
    
    def is_mobile_request(self, user_agent: str) -> bool:
        """判断是否移动端请求"""
        device_info = self.detector.detect(user_agent)
        return device_info.is_mobile or device_info.is_tablet


# 全局实例
_mobile_adapter = None

def get_mobile_adapter() -> MobileAdapter:
    """获取移动端适配器单例"""
    global _mobile_adapter
    if _mobile_adapter is None:
        _mobile_adapter = MobileAdapter()
    return _mobile_adapter


# 测试代码
if __name__ == '__main__':
    print("Testing Mobile Adapter...")
    
    adapter = MobileAdapter()
    
    # 测试设备检测
    test_agents = [
        'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)',
        'Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X)',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0',
        'Mozilla/5.0 (Linux; Android 11; Pixel 4)'
    ]
    
    for agent in test_agents:
        result = adapter.adapt(agent)
        print(f"\nUser-Agent: {agent[:50]}...")
        print(f"Device: {result['device']}")
        print(f"Config: {result['config']}")
    
    print("\nDone!")