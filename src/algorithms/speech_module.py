"""
语音播报模块

功能：
1. 文本转语音 (TTS)
2. 事件语音播报
3. 多语言支持
4. 语音队列管理
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import deque
from loguru import logger
import time
import threading
import queue


@dataclass
class SpeechTask:
    """语音任务"""
    text: str
    priority: int = 0        # 优先级，数字越大越优先
    language: str = "zh-CN"
    rate: float = 1.0        # 语速
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class SpeechEngine:
    """
    语音引擎
    
    支持多种 TTS 后端
    """
    
    def __init__(self, backend: str = "auto"):
        """
        初始化语音引擎
        
        Args:
            backend: TTS 后端 ("auto", "pyttsx3", "gtts", "macos")
        """
        self.backend = backend
        self.engine = None
        self.available = False
        
        self._init_engine()
        
        logger.info(f"SpeechEngine initialized (backend={self.backend}, available={self.available})")
    
    def _init_engine(self):
        """初始化 TTS 引擎"""
        if self.backend == "auto":
            # 尝试不同的后端
            backends = ["macos", "pyttsx3", "gtts"]
            for b in backends:
                if self._try_backend(b):
                    self.backend = b
                    self.available = True
                    return
            
            logger.warning("No TTS backend available")
            return
        
        self._try_backend(self.backend)
    
    def _try_backend(self, backend: str) -> bool:
        """尝试初始化指定后端"""
        try:
            if backend == "pyttsx3":
                import pyttsx3
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 150)
                return True
            
            elif backend == "gtts":
                from gtts import gTTS
                self.engine = gTTS
                return True
            
            elif backend == "macos":
                import subprocess
                # 测试 say 命令
                result = subprocess.run(['which', 'say'], capture_output=True)
                if result.returncode == 0:
                    self.engine = "macos"
                    return True
            
            return False
            
        except ImportError:
            return False
        except Exception as e:
            logger.debug(f"Backend {backend} init error: {e}")
            return False
    
    def speak(self, text: str, language: str = "zh-CN", rate: float = 1.0, blocking: bool = False):
        """
        播放语音
        
        Args:
            text: 要播放的文本
            language: 语言
            rate: 语速
            blocking: 是否阻塞
        """
        if not self.available:
            logger.warning(f"TTS not available, would say: {text}")
            return
        
        if blocking:
            self._speak_sync(text, language, rate)
        else:
            thread = threading.Thread(
                target=self._speak_sync,
                args=(text, language, rate)
            )
            thread.start()
    
    def _speak_sync(self, text: str, language: str, rate: float):
        """同步播放"""
        try:
            if self.backend == "pyttsx3":
                self.engine.say(text)
                self.engine.runAndWait()
            
            elif self.backend == "gtts":
                from gtts import gTTS
                import tempfile
                import os
                
                # 创建临时文件
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                    temp_file = f.name
                
                # 生成语音
                tts = gTTS(text=text, lang=language.split('-')[0])
                tts.save(temp_file)
                
                # 播放
                os.system(f'afplay {temp_file}' if os.name != 'nt' else f'start {temp_file}')
                
                # 清理
                time.sleep(0.5)
                os.remove(temp_file)
            
            elif self.backend == "macos":
                import subprocess
                
                # 使用 say 命令
                cmd = ['say']
                
                if language.startswith('zh'):
                    cmd.extend(['-v', 'Ting-Ting'])
                elif language.startswith('en'):
                    cmd.extend(['-v', 'Samantha'])
                
                cmd.append(text)
                
                subprocess.run(cmd)
            
        except Exception as e:
            logger.error(f"Speech error: {e}")
    
    def save_to_file(self, text: str, filepath: str, language: str = "zh-CN"):
        """
        保存语音到文件
        
        Args:
            text: 文本
            filepath: 文件路径
            language: 语言
        """
        if not self.available:
            return False
        
        try:
            if self.backend == "gtts":
                from gtts import gTTS
                tts = gTTS(text=text, lang=language.split('-')[0])
                tts.save(filepath)
                return True
            
            elif self.backend == "macos":
                import subprocess
                subprocess.run(['say', '-o', filepath, text])
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Save speech error: {e}")
            return False


class SpeechQueue:
    """
    语音队列管理器
    
    管理语音播报队列
    """
    
    def __init__(self, max_size: int = 20, cooldown: float = 2.0):
        """
        初始化语音队列
        
        Args:
            max_size: 最大队列大小
            cooldown: 相同内容冷却时间（秒）
        """
        self.max_size = max_size
        self.cooldown = cooldown
        
        self.queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_size)
        self.recent_texts: Dict[str, float] = {}
        
        self.running = False
        self.worker_thread = None
        
        self.speech_engine = SpeechEngine()
        
        logger.info("SpeechQueue initialized")
    
    def start(self):
        """启动队列处理"""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        
        logger.info("SpeechQueue started")
    
    def stop(self):
        """停止队列处理"""
        self.running = False
        
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
        
        logger.info("SpeechQueue stopped")
    
    def add_task(self, task: SpeechTask) -> bool:
        """
        添加语音任务
        
        Args:
            task: 语音任务
            
        Returns:
            是否成功添加
        """
        current_time = time.time()
        
        # 检查冷却
        text_key = task.text[:50]  # 使用前50字符作为key
        if text_key in self.recent_texts:
            if current_time - self.recent_texts[text_key] < self.cooldown:
                return False
        
        # 更新最近记录
        self.recent_texts[text_key] = current_time
        
        # 清理过期记录
        expired = [k for k, v in self.recent_texts.items() 
                   if current_time - v > self.cooldown * 2]
        for k in expired:
            del self.recent_texts[k]
        
        # 添加到队列
        try:
            # 使用负优先级，因为 PriorityQueue 是最小堆
            self.queue.put((-task.priority, task), block=False)
            return True
        except queue.Full:
            logger.warning("Speech queue is full")
            return False
    
    def speak(self, text: str, priority: int = 0, language: str = "zh-CN") -> bool:
        """
        添加语音播报
        
        Args:
            text: 文本
            priority: 优先级
            language: 语言
            
        Returns:
            是否成功添加
        """
        task = SpeechTask(
            text=text,
            priority=priority,
            language=language
        )
        return self.add_task(task)
    
    def _worker(self):
        """工作线程"""
        while self.running:
            try:
                # 获取任务
                priority, task = self.queue.get(timeout=1.0)
                
                # 播放语音
                self.speech_engine.speak(
                    text=task.text,
                    language=task.language,
                    rate=task.rate,
                    blocking=True
                )
                
                # 间隔
                time.sleep(0.5)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Speech worker error: {e}")
    
    def get_queue_size(self) -> int:
        """获取队列大小"""
        return self.queue.qsize()


class EventAnnouncer:
    """
    事件播报器
    
    将检测事件转换为语音播报
    """
    
    # 事件播报模板
    TEMPLATES = {
        'person_detected': '检测到人员',
        'person_entered': '有人进入区域',
        'person_left': '有人离开区域',
        'fall_detected': '警告！检测到有人跌倒',
        'gesture_thumbs_up': '检测到点赞手势',
        'gesture_victory': '检测到胜利手势',
        'gesture_ok': '检测到OK手势',
        'crowd_detected': '警告！检测到人群聚集',
        'vehicle_detected': '检测到车辆',
        'alert_triggered': '警报！请立即查看',
        'system_started': '系统已启动',
        'system_stopped': '系统已停止'
    }
    
    def __init__(self, enabled: bool = True):
        """
        初始化事件播报器
        
        Args:
            enabled: 是否启用
        """
        self.enabled = enabled
        self.speech_queue = SpeechQueue()
        
        # 统计
        self.announcement_count = 0
        
        logger.info(f"EventAnnouncer initialized (enabled={enabled})")
    
    def start(self):
        """启动播报器"""
        if self.enabled:
            self.speech_queue.start()
    
    def stop(self):
        """停止播报器"""
        self.speech_queue.stop()
    
    def announce(self, event_type: str, details: Dict = None, priority: int = 0):
        """
        播报事件
        
        Args:
            event_type: 事件类型
            details: 事件详情
            priority: 优先级
        """
        if not self.enabled:
            return
        
        # 获取播报文本
        text = self._get_announcement_text(event_type, details)
        
        if text:
            self.speech_queue.speak(text, priority=priority)
            self.announcement_count += 1
    
    def _get_announcement_text(self, event_type: str, details: Dict = None) -> Optional[str]:
        """获取播报文本"""
        # 检查模板
        if event_type in self.TEMPLATES:
            text = self.TEMPLATES[event_type]
            
            # 添加详情
            if details:
                if 'count' in details:
                    text += f"，共{details['count']}人"
                if 'zone' in details:
                    text += f"，区域：{details['zone']}"
            
            return text
        
        # 自定义事件
        if details and 'message' in details:
            return details['message']
        
        return None
    
    def announce_person_detected(self, count: int):
        """播报人员检测"""
        self.announce('person_detected', {'count': count})
    
    def announce_fall(self, track_id: int):
        """播报跌倒事件"""
        self.announce('fall_detected', {'track_id': track_id}, priority=10)
    
    def announce_gesture(self, gesture: str):
        """播报手势"""
        event_type = f'gesture_{gesture}'
        self.announce(event_type, priority=5)
    
    def announce_alert(self, message: str):
        """播报警报"""
        self.announce('alert_triggered', {'message': message}, priority=10)
    
    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            'enabled': self.enabled,
            'announcement_count': self.announcement_count,
            'queue_size': self.speech_queue.get_queue_size()
        }


# 全局实例
_speech_queue = None
_event_announcer = None

def get_speech_queue() -> SpeechQueue:
    """获取语音队列单例"""
    global _speech_queue
    if _speech_queue is None:
        _speech_queue = SpeechQueue()
    return _speech_queue

def get_event_announcer() -> EventAnnouncer:
    """获取事件播报器单例"""
    global _event_announcer
    if _event_announcer is None:
        _event_announcer = EventAnnouncer()
    return _event_announcer


# 测试代码
if __name__ == '__main__':
    print("Testing Speech Module...")
    
    announcer = EventAnnouncer()
    announcer.start()
    
    # 测试播报
    announcer.announce_person_detected(1)
    announcer.announce_gesture('thumbs_up')
    announcer.announce_fall(1)
    
    # 等待播报完成
    time.sleep(5)
    
    announcer.stop()
    print("\nDone!")