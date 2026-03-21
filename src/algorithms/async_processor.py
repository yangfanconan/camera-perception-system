"""
异步处理模块

实现高性能异步处理：
1. 帧队列管理
2. 多线程处理
3. 流水线并行
4. 结果缓存
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Lock, Event
from loguru import logger
import time
import queue


@dataclass
class FrameData:
    """帧数据"""
    frame_id: int
    frame: np.ndarray
    timestamp: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class ProcessResult:
    """处理结果"""
    frame_id: int
    result: Dict
    timestamp: float
    process_time: float


class FrameQueue:
    """
    帧队列
    
    管理输入帧的缓冲
    """
    
    def __init__(self, max_size: int = 30):
        """
        初始化帧队列
        
        Args:
            max_size: 最大队列大小
        """
        self.max_size = max_size
        self.queue: deque = deque(maxlen=max_size)
        self.lock = Lock()
        self.latest_frame_id = -1
    
    def put(self, frame_data: FrameData) -> bool:
        """
        添加帧到队列
        
        Args:
            frame_data: 帧数据
            
        Returns:
            是否成功添加
        """
        with self.lock:
            if len(self.queue) >= self.max_size:
                # 丢弃最旧的帧
                self.queue.popleft()
            
            self.queue.append(frame_data)
            self.latest_frame_id = frame_data.frame_id
            return True
    
    def get(self) -> Optional[FrameData]:
        """
        获取最新的帧
        
        Returns:
            帧数据
        """
        with self.lock:
            if self.queue:
                return self.queue.pop()
            return None
    
    def get_all(self) -> List[FrameData]:
        """获取所有帧"""
        with self.lock:
            return list(self.queue)
    
    def clear(self):
        """清空队列"""
        with self.lock:
            self.queue.clear()
    
    def size(self) -> int:
        """获取队列大小"""
        return len(self.queue)


class ResultCache:
    """
    结果缓存
    
    缓存处理结果以避免重复计算
    """
    
    def __init__(self, max_size: int = 100, ttl: float = 1.0):
        """
        初始化缓存
        
        Args:
            max_size: 最大缓存大小
            ttl: 缓存过期时间（秒）
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[int, Tuple[ProcessResult, float]] = {}
        self.lock = Lock()
    
    def put(self, result: ProcessResult):
        """添加结果到缓存"""
        with self.lock:
            # 清理过期缓存
            current_time = time.time()
            expired = [k for k, (v, t) in self.cache.items() 
                       if current_time - t > self.ttl]
            for k in expired:
                del self.cache[k]
            
            # 检查大小
            if len(self.cache) >= self.max_size:
                # 删除最旧的
                oldest = min(self.cache.keys(), 
                            key=lambda k: self.cache[k][1])
                del self.cache[oldest]
            
            self.cache[result.frame_id] = (result, current_time)
    
    def get(self, frame_id: int) -> Optional[ProcessResult]:
        """获取缓存结果"""
        with self.lock:
            if frame_id in self.cache:
                result, timestamp = self.cache[frame_id]
                if time.time() - timestamp < self.ttl:
                    return result
                else:
                    del self.cache[frame_id]
            return None
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()


class AsyncProcessor:
    """
    异步处理器
    
    实现多线程异步处理
    """
    
    def __init__(
        self,
        process_func: Callable,
        num_workers: int = 4,
        queue_size: int = 30
    ):
        """
        初始化异步处理器
        
        Args:
            process_func: 处理函数
            num_workers: 工作线程数
            queue_size: 队列大小
        """
        self.process_func = process_func
        self.num_workers = num_workers
        
        # 帧队列
        self.input_queue = FrameQueue(max_size=queue_size)
        self.output_queue = FrameQueue(max_size=queue_size)
        
        # 结果缓存
        self.result_cache = ResultCache()
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.futures: Dict[int, Future] = {}
        
        # 控制
        self.running = False
        self.stop_event = Event()
        
        # 统计
        self.stats = {
            'frames_processed': 0,
            'frames_dropped': 0,
            'total_process_time': 0.0,
            'avg_process_time': 0.0
        }
        self.stats_lock = Lock()
        
        logger.info(f"AsyncProcessor initialized (workers={num_workers})")
    
    def start(self):
        """启动处理器"""
        if self.running:
            return
        
        self.running = True
        self.stop_event.clear()
        
        # 启动工作线程
        for i in range(self.num_workers):
            self.executor.submit(self._worker, i)
        
        logger.info("AsyncProcessor started")
    
    def stop(self):
        """停止处理器"""
        self.running = False
        self.stop_event.set()
        self.executor.shutdown(wait=False)
        
        logger.info("AsyncProcessor stopped")
    
    def submit(self, frame: np.ndarray, metadata: Dict = None) -> int:
        """
        提交帧进行处理
        
        Args:
            frame: 图像帧
            metadata: 元数据
            
        Returns:
            帧ID
        """
        frame_id = int(time.time() * 1000) % (2 ** 31)
        
        frame_data = FrameData(
            frame_id=frame_id,
            frame=frame.copy(),
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        self.input_queue.put(frame_data)
        
        return frame_id
    
    def get_result(self, timeout: float = 0.1) -> Optional[ProcessResult]:
        """
        获取处理结果
        
        Args:
            timeout: 超时时间
            
        Returns:
            处理结果
        """
        return self.output_queue.get()
    
    def _worker(self, worker_id: int):
        """
        工作线程
        
        Args:
            worker_id: 工作线程ID
        """
        logger.debug(f"Worker {worker_id} started")
        
        while self.running and not self.stop_event.is_set():
            try:
                # 获取帧
                frame_data = self.input_queue.get()
                if frame_data is None:
                    time.sleep(0.001)
                    continue
                
                # 检查缓存
                cached = self.result_cache.get(frame_data.frame_id)
                if cached:
                    self.output_queue.put(cached)
                    continue
                
                # 处理
                start_time = time.time()
                result = self.process_func(frame_data.frame, frame_data.metadata)
                process_time = time.time() - start_time
                
                # 创建结果
                process_result = ProcessResult(
                    frame_id=frame_data.frame_id,
                    result=result,
                    timestamp=time.time(),
                    process_time=process_time
                )
                
                # 缓存结果
                self.result_cache.put(process_result)
                
                # 添加到输出队列
                self.output_queue.put(process_result)
                
                # 更新统计
                with self.stats_lock:
                    self.stats['frames_processed'] += 1
                    self.stats['total_process_time'] += process_time
                    self.stats['avg_process_time'] = (
                        self.stats['total_process_time'] / 
                        self.stats['frames_processed']
                    )
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        logger.debug(f"Worker {worker_id} stopped")
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        with self.stats_lock:
            return self.stats.copy()


class PipelineProcessor:
    """
    流水线处理器
    
    实现多阶段流水线处理
    """
    
    def __init__(self, stages: List[Tuple[str, Callable]]):
        """
        初始化流水线处理器
        
        Args:
            stages: 处理阶段列表 [(name, func), ...]
        """
        self.stages = stages
        self.stage_queues: List[queue.Queue] = [
            queue.Queue(maxsize=10) for _ in range(len(stages) + 1)
        ]
        
        self.running = False
        self.threads = []
        
        # 统计
        self.stage_stats = [{**{'name': name, 'count': 0, 'total_time': 0}} 
                           for name, _ in stages]
        
        logger.info(f"PipelineProcessor initialized with {len(stages)} stages")
    
    def start(self):
        """启动流水线"""
        if self.running:
            return
        
        self.running = True
        
        # 启动各阶段线程
        for i, (name, func) in enumerate(self.stages):
            thread = threading.Thread(
                target=self._stage_worker,
                args=(i, name, func),
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
        
        logger.info("PipelineProcessor started")
    
    def stop(self):
        """停止流水线"""
        self.running = False
        
        for q in self.stage_queues:
            # 发送停止信号
            try:
                q.put(None)
            except:
                pass
        
        for thread in self.threads:
            thread.join(timeout=1.0)
        
        self.threads.clear()
        logger.info("PipelineProcessor stopped")
    
    def submit(self, frame: np.ndarray, metadata: Dict = None):
        """提交帧到流水线"""
        self.stage_queues[0].put({
            'frame': frame,
            'metadata': metadata or {},
            'results': {}
        })
    
    def get_result(self, timeout: float = 1.0) -> Optional[Dict]:
        """获取最终结果"""
        try:
            return self.stage_queues[-1].get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _stage_worker(self, stage_id: int, name: str, func: Callable):
        """阶段工作线程"""
        import threading
        
        while self.running:
            try:
                # 获取输入
                data = self.stage_queues[stage_id].get(timeout=0.1)
                if data is None:
                    continue
                
                # 处理
                start_time = time.time()
                result = func(data['frame'], data['metadata'], data['results'])
                process_time = time.time() - start_time
                
                # 更新结果
                data['results'][name] = result
                
                # 更新统计
                self.stage_stats[stage_id]['count'] += 1
                self.stage_stats[stage_id]['total_time'] += process_time
                
                # 传递到下一阶段
                self.stage_queues[stage_id + 1].put(data)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Stage {name} error: {e}")
    
    def get_stats(self) -> List[Dict]:
        """获取各阶段统计"""
        stats = []
        for i, stat in enumerate(self.stage_stats):
            avg_time = stat['total_time'] / max(stat['count'], 1)
            stats.append({
                'stage': stat['name'],
                'count': stat['count'],
                'avg_time': round(avg_time * 1000, 2)  # ms
            })
        return stats


class FrameSkipper:
    """
    帧跳过器
    
    根据处理能力动态调整帧率
    """
    
    def __init__(self, target_fps: float = 30.0, max_skip: int = 3):
        """
        初始化帧跳过器
        
        Args:
            target_fps: 目标帧率
            max_skip: 最大跳帧数
        """
        self.target_fps = target_fps
        self.max_skip = max_skip
        
        self.frame_interval = 1.0 / target_fps
        self.last_process_time = 0
        self.skip_count = 0
        self.total_frames = 0
        self.processed_frames = 0
    
    def should_process(self) -> bool:
        """
        判断是否应该处理当前帧
        
        Returns:
            是否处理
        """
        self.total_frames += 1
        current_time = time.time()
        
        # 检查时间间隔
        if current_time - self.last_process_time >= self.frame_interval:
            self.last_process_time = current_time
            self.processed_frames += 1
            self.skip_count = 0
            return True
        
        # 检查跳帧计数
        if self.skip_count < self.max_skip:
            self.skip_count += 1
            return False
        
        # 强制处理
        self.last_process_time = current_time
        self.processed_frames += 1
        self.skip_count = 0
        return True
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_frames': self.total_frames,
            'processed_frames': self.processed_frames,
            'skipped_frames': self.total_frames - self.processed_frames,
            'actual_fps': self.processed_frames / max(time.time() - (self.last_process_time - self.frame_interval * self.processed_frames), 0.001)
        }


# 测试代码
if __name__ == '__main__':
    print("Testing Async Processor...")
    
    # 测试处理函数
    def process_frame(frame: np.ndarray, metadata: Dict) -> Dict:
        time.sleep(0.01)  # 模拟处理
        return {'detected': True, 'count': 1}
    
    # 创建处理器
    processor = AsyncProcessor(process_func=process_frame, num_workers=2)
    processor.start()
    
    # 提交帧
    for i in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_id = processor.submit(frame, {'index': i})
        print(f"Submitted frame {i}, id={frame_id}")
        
        # 获取结果
        result = processor.get_result()
        if result:
            print(f"  Got result: frame_id={result.frame_id}, time={result.process_time:.3f}s")
    
    # 打印统计
    print("\nStats:", processor.get_stats())
    
    processor.stop()
    print("\nDone!")