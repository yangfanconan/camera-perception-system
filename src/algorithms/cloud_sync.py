"""
云端同步模块

功能：
1. 数据上传到云端
2. 配置同步
3. 远程监控
4. 数据备份
"""

import json
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger
import time
import threading
import queue
from datetime import datetime


@dataclass
class SyncTask:
    """同步任务"""
    task_id: str
    task_type: str       # upload, download, sync
    data_type: str       # config, data, model
    payload: Any
    timestamp: float = field(default_factory=time.time)
    retries: int = 0
    max_retries: int = 3


@dataclass
class SyncStatus:
    """同步状态"""
    last_sync: float = 0
    sync_count: int = 0
    failed_count: int = 0
    pending_count: int = 0
    is_syncing: bool = False


class CloudStorage:
    """
    云存储接口
    
    支持多种云存储后端
    """
    
    def __init__(self, backend: str = "local", config: Dict = None):
        """
        初始化云存储
        
        Args:
            backend: 存储后端 ("local", "s3", "oss", "http")
            config: 配置
        """
        self.backend = backend
        self.config = config or {}
        self.client = None
        
        self._init_backend()
        
        logger.info(f"CloudStorage initialized (backend={backend})")
    
    def _init_backend(self):
        """初始化后端"""
        if self.backend == "local":
            # 本地存储
            self.storage_dir = Path(self.config.get('path', 'cloud_storage'))
            self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        elif self.backend == "s3":
            # AWS S3
            try:
                import boto3
                self.client = boto3.client('s3')
                self.bucket = self.config.get('bucket')
                logger.info("S3 client initialized")
            except ImportError:
                logger.warning("boto3 not available, using local storage")
                self.backend = "local"
                self._init_backend()
        
        elif self.backend == "oss":
            # 阿里云 OSS
            try:
                import oss2
                auth = oss2.Auth(
                    self.config.get('access_key'),
                    self.config.get('secret_key')
                )
                self.client = oss2.Bucket(
                    auth,
                    self.config.get('endpoint'),
                    self.config.get('bucket')
                )
                logger.info("OSS client initialized")
            except ImportError:
                logger.warning("oss2 not available, using local storage")
                self.backend = "local"
                self._init_backend()
    
    def upload(self, key: str, data: bytes) -> bool:
        """
        上传数据
        
        Args:
            key: 键名
            data: 数据
            
        Returns:
            是否成功
        """
        try:
            if self.backend == "local":
                filepath = self.storage_dir / key
                filepath.parent.mkdir(parents=True, exist_ok=True)
                with open(filepath, 'wb') as f:
                    f.write(data)
                return True
            
            elif self.backend == "s3":
                self.client.put_object(Bucket=self.bucket, Key=key, Body=data)
                return True
            
            elif self.backend == "oss":
                self.client.put_object(key, data)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return False
    
    def download(self, key: str) -> Optional[bytes]:
        """
        下载数据
        
        Args:
            key: 键名
            
        Returns:
            数据
        """
        try:
            if self.backend == "local":
                filepath = self.storage_dir / key
                if filepath.exists():
                    with open(filepath, 'rb') as f:
                        return f.read()
                return None
            
            elif self.backend == "s3":
                response = self.client.get_object(Bucket=self.bucket, Key=key)
                return response['Body'].read()
            
            elif self.backend == "oss":
                return self.client.get_object(key).read()
            
            return None
            
        except Exception as e:
            logger.error(f"Download error: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """删除数据"""
        try:
            if self.backend == "local":
                filepath = self.storage_dir / key
                if filepath.exists():
                    filepath.unlink()
                return True
            
            elif self.backend == "s3":
                self.client.delete_object(Bucket=self.bucket, Key=key)
                return True
            
            elif self.backend == "oss":
                self.client.delete_object(key)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return False
    
    def list_keys(self, prefix: str = "") -> List[str]:
        """列出键"""
        try:
            if self.backend == "local":
                keys = []
                for path in self.storage_dir.rglob("*"):
                    if path.is_file():
                        key = str(path.relative_to(self.storage_dir))
                        if key.startswith(prefix):
                            keys.append(key)
                return keys
            
            # 其他后端需要实现
            return []
            
        except Exception as e:
            logger.error(f"List error: {e}")
            return []


class SyncManager:
    """
    同步管理器
    
    管理数据同步
    """
    
    def __init__(self, storage: CloudStorage = None):
        """
        初始化同步管理器
        
        Args:
            storage: 云存储实例
        """
        self.storage = storage or CloudStorage()
        self.status = SyncStatus()
        
        # 同步队列
        self.queue: queue.Queue = queue.Queue()
        
        # 工作线程
        self.running = False
        self.worker_thread = None
        
        # 同步间隔
        self.sync_interval = 60.0  # 秒
        
        logger.info("SyncManager initialized")
    
    def start(self):
        """启动同步"""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        
        logger.info("SyncManager started")
    
    def stop(self):
        """停止同步"""
        self.running = False
        
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        
        logger.info("SyncManager stopped")
    
    def _worker(self):
        """工作线程"""
        while self.running:
            try:
                # 处理队列中的任务
                try:
                    task = self.queue.get(timeout=1.0)
                    self._process_task(task)
                except queue.Empty:
                    pass
                
                # 定期同步
                if time.time() - self.status.last_sync > self.sync_interval:
                    self._periodic_sync()
                
            except Exception as e:
                logger.error(f"Sync worker error: {e}")
    
    def _process_task(self, task: SyncTask):
        """处理同步任务"""
        try:
            if task.task_type == "upload":
                success = self._upload_data(task)
            elif task.task_type == "download":
                success = self._download_data(task)
            else:
                success = False
            
            if success:
                self.status.sync_count += 1
            else:
                # 重试
                if task.retries < task.max_retries:
                    task.retries += 1
                    self.queue.put(task)
                else:
                    self.status.failed_count += 1
                    logger.error(f"Task failed after {task.max_retries} retries: {task.task_id}")
            
        except Exception as e:
            logger.error(f"Task process error: {e}")
            self.status.failed_count += 1
    
    def _upload_data(self, task: SyncTask) -> bool:
        """上传数据"""
        key = f"{task.data_type}/{task.task_id}"
        
        if isinstance(task.payload, dict):
            data = json.dumps(task.payload).encode('utf-8')
        elif isinstance(task.payload, str):
            data = task.payload.encode('utf-8')
        else:
            data = task.payload
        
        return self.storage.upload(key, data)
    
    def _download_data(self, task: SyncTask) -> bool:
        """下载数据"""
        key = f"{task.data_type}/{task.task_id}"
        data = self.storage.download(key)
        
        if data:
            task.payload = data
            return True
        
        return False
    
    def _periodic_sync(self):
        """定期同步"""
        self.status.is_syncing = True
        
        try:
            # 同步配置
            self._sync_config()
            
            # 更新状态
            self.status.last_sync = time.time()
            
        except Exception as e:
            logger.error(f"Periodic sync error: {e}")
        
        self.status.is_syncing = False
    
    def _sync_config(self):
        """同步配置"""
        try:
            config_path = Path("config/config.yaml")
            if config_path.exists():
                with open(config_path, 'rb') as f:
                    data = f.read()
                
                key = f"config/config_{datetime.now().strftime('%Y%m%d')}.yaml"
                self.storage.upload(key, data)
                
        except Exception as e:
            logger.error(f"Config sync error: {e}")
    
    def upload_data(self, data_type: str, data_id: str, data: Any) -> bool:
        """
        上传数据
        
        Args:
            data_type: 数据类型
            data_id: 数据ID
            data: 数据
            
        Returns:
            是否成功添加到队列
        """
        task = SyncTask(
            task_id=data_id,
            task_type="upload",
            data_type=data_type,
            payload=data
        )
        
        try:
            self.queue.put(task)
            self.status.pending_count = self.queue.qsize()
            return True
        except Exception as e:
            logger.error(f"Upload queue error: {e}")
            return False
    
    def download_data(self, data_type: str, data_id: str) -> Optional[Any]:
        """
        下载数据
        
        Args:
            data_type: 数据类型
            data_id: 数据ID
            
        Returns:
            数据
        """
        task = SyncTask(
            task_id=data_id,
            task_type="download",
            data_type=data_type,
            payload=None
        )
        
        self._process_task(task)
        
        return task.payload
    
    def backup_session(self, session_data: Dict) -> bool:
        """
        备份会话数据
        
        Args:
            session_data: 会话数据
            
        Returns:
            是否成功
        """
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        return self.upload_data('sessions', session_id, session_data)
    
    def get_status(self) -> Dict:
        """获取同步状态"""
        return {
            'last_sync': self.status.last_sync,
            'sync_count': self.status.sync_count,
            'failed_count': self.status.failed_count,
            'pending_count': self.queue.qsize(),
            'is_syncing': self.status.is_syncing
        }


class RemoteMonitor:
    """
    远程监控
    
    提供远程监控接口
    """
    
    def __init__(self, sync_manager: SyncManager = None):
        """
        初始化远程监控
        
        Args:
            sync_manager: 同步管理器
        """
        self.sync_manager = sync_manager or SyncManager()
        
        # 监控数据
        self.monitor_data: Dict = {}
        self.last_update = 0
        self.update_interval = 5.0  # 秒
        
        logger.info("RemoteMonitor initialized")
    
    def update_status(self, status: Dict):
        """
        更新状态
        
        Args:
            status: 状态数据
        """
        self.monitor_data.update(status)
        self.monitor_data['timestamp'] = time.time()
        
        # 定期上传
        if time.time() - self.last_update > self.update_interval:
            self._upload_status()
            self.last_update = time.time()
    
    def _upload_status(self):
        """上传状态"""
        self.sync_manager.upload_data(
            'monitor',
            datetime.now().strftime('%Y%m%d_%H%M%S'),
            self.monitor_data
        )
    
    def get_remote_status(self) -> Optional[Dict]:
        """获取远程状态"""
        # 获取最新的监控数据
        keys = self.sync_manager.storage.list_keys('monitor/')
        
        if keys:
            latest_key = sorted(keys)[-1]
            data = self.sync_manager.storage.download(latest_key)
            
            if data:
                return json.loads(data.decode('utf-8'))
        
        return None


# 全局实例
_sync_manager = None
_remote_monitor = None

def get_sync_manager() -> SyncManager:
    """获取同步管理器单例"""
    global _sync_manager
    if _sync_manager is None:
        _sync_manager = SyncManager()
    return _sync_manager

def get_remote_monitor() -> RemoteMonitor:
    """获取远程监控单例"""
    global _remote_monitor
    if _remote_monitor is None:
        _remote_monitor = RemoteMonitor()
    return _remote_monitor


# 测试代码
if __name__ == '__main__':
    print("Testing Cloud Sync...")
    
    # 使用本地存储测试
    storage = CloudStorage(backend="local", config={'path': 'test_cloud'})
    
    # 上传测试
    storage.upload('test/data.json', json.dumps({'test': 'data'}).encode())
    
    # 下载测试
    data = storage.download('test/data.json')
    print(f"Downloaded: {data}")
    
    # 同步管理器
    sync = SyncManager(storage)
    
    # 上传数据
    sync.upload_data('detections', 'test_001', {'persons': 3})
    
    print(f"Status: {sync.get_status()}")
    
    print("\nDone!")