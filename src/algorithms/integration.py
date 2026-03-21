"""
系统集成模块

功能：
1. 第三方系统集成
2. Webhook 支持
3. API 网关
4. 消息队列
"""

import numpy as np
import time
import json
import hashlib
import hmac
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger
import threading
import queue
import asyncio
import aiohttp


@dataclass
class WebhookConfig:
    """Webhook 配置"""
    webhook_id: str
    name: str
    url: str
    secret: str
    events: List[str]  # 订阅的事件类型
    enabled: bool = True
    retry_count: int = 3
    timeout: float = 10.0


@dataclass
class WebhookEvent:
    """Webhook 事件"""
    event_id: str
    event_type: str
    timestamp: float
    data: Dict
    source: str = "camera-perception-system"
    
    def to_dict(self) -> Dict:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'timestamp': self.timestamp,
            'source': self.source,
            'data': self.data
        }


@dataclass
class Integration:
    """集成配置"""
    integration_id: str
    name: str
    integration_type: str  # webhook, mqtt, kafka, etc.
    config: Dict
    enabled: bool = True
    last_sync: float = 0
    
    def to_dict(self) -> Dict:
        return {
            'integration_id': self.integration_id,
            'name': self.name,
            'integration_type': self.integration_type,
            'enabled': self.enabled,
            'last_sync': self.last_sync
        }


class WebhookManager:
    """
    Webhook 管理器
    
    管理 Webhook 发送
    """
    
    def __init__(self):
        """初始化 Webhook 管理器"""
        self.webhooks: Dict[str, WebhookConfig] = {}
        self.event_queue: queue.Queue = queue.Queue()
        
        self.running = False
        self.worker_thread = None
        
        # 统计
        self.stats = defaultdict(lambda: {'sent': 0, 'failed': 0})
        
        logger.info("WebhookManager initialized")
    
    def register_webhook(self, config: WebhookConfig) -> bool:
        """注册 Webhook"""
        self.webhooks[config.webhook_id] = config
        logger.info(f"Webhook registered: {config.name}")
        return True
    
    def unregister_webhook(self, webhook_id: str) -> bool:
        """注销 Webhook"""
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            return True
        return False
    
    def trigger(self, event_type: str, data: Dict):
        """
        触发事件
        
        Args:
            event_type: 事件类型
            data: 事件数据
        """
        event = WebhookEvent(
            event_id=self._generate_id(),
            event_type=event_type,
            timestamp=time.time(),
            data=data
        )
        
        self.event_queue.put(event)
    
    def start(self):
        """启动工作线程"""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        logger.info("Webhook worker started")
    
    def stop(self):
        """停止工作线程"""
        self.running = False
        
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
        
        logger.info("Webhook worker stopped")
    
    def _worker_loop(self):
        """工作线程循环"""
        while self.running:
            try:
                event = self.event_queue.get(timeout=1.0)
                self._process_event(event)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Webhook worker error: {e}")
    
    def _process_event(self, event: WebhookEvent):
        """处理事件"""
        for webhook in self.webhooks.values():
            if not webhook.enabled:
                continue
            
            if event.event_type not in webhook.events and '*' not in webhook.events:
                continue
            
            self._send_webhook(webhook, event)
    
    def _send_webhook(self, webhook: WebhookConfig, event: WebhookEvent):
        """发送 Webhook"""
        import requests
        
        payload = json.dumps(event.to_dict())
        
        # 计算签名
        signature = self._compute_signature(payload, webhook.secret)
        
        headers = {
            'Content-Type': 'application/json',
            'X-Signature': signature,
            'X-Event-Type': event.event_type
        }
        
        for attempt in range(webhook.retry_count):
            try:
                response = requests.post(
                    webhook.url,
                    data=payload,
                    headers=headers,
                    timeout=webhook.timeout
                )
                
                if response.status_code == 200:
                    self.stats[webhook.webhook_id]['sent'] += 1
                    logger.debug(f"Webhook sent: {webhook.name}")
                    return
                else:
                    logger.warning(f"Webhook failed: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Webhook error: {e}")
            
            time.sleep(1)  # 重试延迟
        
        self.stats[webhook.webhook_id]['failed'] += 1
    
    def _compute_signature(self, payload: str, secret: str) -> str:
        """计算签名"""
        return hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def _generate_id(self) -> str:
        """生成事件ID"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def get_stats(self) -> Dict:
        """获取统计"""
        return dict(self.stats)


class MessageQueue:
    """
    消息队列
    
    支持多种消息队列后端
    """
    
    def __init__(self):
        """初始化消息队列"""
        self.queues: Dict[str, queue.Queue] = {}
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        
        logger.info("MessageQueue initialized")
    
    def create_queue(self, name: str, max_size: int = 1000) -> bool:
        """创建队列"""
        if name in self.queues:
            return False
        
        self.queues[name] = queue.Queue(maxsize=max_size)
        return True
    
    def publish(self, queue_name: str, message: Dict) -> bool:
        """
        发布消息
        
        Args:
            queue_name: 队列名称
            message: 消息内容
            
        Returns:
            是否成功
        """
        if queue_name not in self.queues:
            return False
        
        try:
            self.queues[queue_name].put(message, block=False)
            
            # 通知订阅者
            for callback in self.subscribers.get(queue_name, []):
                try:
                    callback(message)
                except Exception as e:
                    logger.error(f"Subscriber error: {e}")
            
            return True
        except queue.Full:
            logger.warning(f"Queue full: {queue_name}")
            return False
    
    def consume(self, queue_name: str, timeout: float = 1.0) -> Optional[Dict]:
        """
        消费消息
        
        Args:
            queue_name: 队列名称
            timeout: 超时时间
            
        Returns:
            消息内容
        """
        if queue_name not in self.queues:
            return None
        
        try:
            return self.queues[queue_name].get(timeout=timeout)
        except queue.Empty:
            return None
    
    def subscribe(self, queue_name: str, callback: Callable):
        """订阅队列"""
        self.subscribers[queue_name].append(callback)
    
    def get_queue_size(self, queue_name: str) -> int:
        """获取队列大小"""
        if queue_name in self.queues:
            return self.queues[queue_name].qsize()
        return 0


class APIGateway:
    """
    API 网关
    
    统一 API 管理
    """
    
    def __init__(self):
        """初始化 API 网关"""
        self.routes: Dict[str, Dict] = {}
        self.middlewares: List[Callable] = []
        self.rate_limits: Dict[str, Dict] = {}
        
        logger.info("APIGateway initialized")
    
    def register_route(
        self,
        path: str,
        method: str,
        handler: Callable,
        rate_limit: int = None
    ):
        """
        注册路由
        
        Args:
            path: 路径
            method: HTTP 方法
            handler: 处理函数
            rate_limit: 速率限制（请求/分钟）
        """
        key = f"{method}:{path}"
        
        self.routes[key] = {
            'path': path,
            'method': method,
            'handler': handler,
            'rate_limit': rate_limit
        }
        
        if rate_limit:
            self.rate_limits[key] = {
                'limit': rate_limit,
                'requests': defaultdict(list)
            }
        
        logger.debug(f"Route registered: {key}")
    
    def add_middleware(self, middleware: Callable):
        """添加中间件"""
        self.middlewares.append(middleware)
    
    def check_rate_limit(self, key: str, client_id: str) -> bool:
        """检查速率限制"""
        if key not in self.rate_limits:
            return True
        
        limit_config = self.rate_limits[key]
        now = time.time()
        
        # 清理过期请求
        limit_config['requests'][client_id] = [
            t for t in limit_config['requests'][client_id]
            if now - t < 60
        ]
        
        # 检查限制
        if len(limit_config['requests'][client_id]) >= limit_config['limit']:
            return False
        
        # 记录请求
        limit_config['requests'][client_id].append(now)
        return True
    
    def get_routes(self) -> List[Dict]:
        """获取所有路由"""
        return [
            {
                'path': route['path'],
                'method': route['method'],
                'rate_limit': route['rate_limit']
            }
            for route in self.routes.values()
        ]


class IntegrationManager:
    """
    集成管理器
    
    管理第三方集成
    """
    
    # 支持的集成类型
    INTEGRATION_TYPES = [
        'webhook',
        'mqtt',
        'kafka',
        'rabbitmq',
        'slack',
        'teams',
        'email',
        'sms'
    ]
    
    def __init__(self):
        """初始化集成管理器"""
        self.integrations: Dict[str, Integration] = {}
        self.webhook_manager = WebhookManager()
        self.message_queue = MessageQueue()
        self.api_gateway = APIGateway()
        
        logger.info("IntegrationManager initialized")
    
    def register_integration(
        self,
        name: str,
        integration_type: str,
        config: Dict
    ) -> Optional[Integration]:
        """
        注册集成
        
        Args:
            name: 集成名称
            integration_type: 集成类型
            config: 配置
            
        Returns:
            集成对象
        """
        if integration_type not in self.INTEGRATION_TYPES:
            logger.error(f"Unsupported integration type: {integration_type}")
            return None
        
        import uuid
        integration_id = str(uuid.uuid4())[:8]
        
        integration = Integration(
            integration_id=integration_id,
            name=name,
            integration_type=integration_type,
            config=config
        )
        
        self.integrations[integration_id] = integration
        
        # 初始化特定类型的集成
        if integration_type == 'webhook':
            webhook_config = WebhookConfig(
                webhook_id=integration_id,
                name=name,
                url=config.get('url', ''),
                secret=config.get('secret', ''),
                events=config.get('events', ['*'])
            )
            self.webhook_manager.register_webhook(webhook_config)
        
        logger.info(f"Integration registered: {name}")
        return integration
    
    def remove_integration(self, integration_id: str) -> bool:
        """移除集成"""
        if integration_id in self.integrations:
            integration = self.integrations[integration_id]
            
            if integration.integration_type == 'webhook':
                self.webhook_manager.unregister_webhook(integration_id)
            
            del self.integrations[integration_id]
            return True
        
        return False
    
    def trigger_event(self, event_type: str, data: Dict):
        """
        触发事件
        
        Args:
            event_type: 事件类型
            data: 事件数据
        """
        # 发送到 Webhook
        self.webhook_manager.trigger(event_type, data)
        
        # 发送到消息队列
        self.message_queue.publish('events', {
            'event_type': event_type,
            'data': data,
            'timestamp': time.time()
        })
    
    def start(self):
        """启动所有集成"""
        self.webhook_manager.start()
        logger.info("Integrations started")
    
    def stop(self):
        """停止所有集成"""
        self.webhook_manager.stop()
        logger.info("Integrations stopped")
    
    def get_integrations(self) -> List[Dict]:
        """获取所有集成"""
        return [i.to_dict() for i in self.integrations.values()]
    
    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            'total_integrations': len(self.integrations),
            'webhook_stats': self.webhook_manager.get_stats(),
            'queue_sizes': {
                name: self.message_queue.get_queue_size(name)
                for name in self.message_queue.queues
            }
        }


class NotificationService:
    """
    通知服务
    
    发送各种通知
    """
    
    def __init__(self, integration_manager: IntegrationManager):
        """
        初始化通知服务
        
        Args:
            integration_manager: 集成管理器
        """
        self.manager = integration_manager
        
        logger.info("NotificationService initialized")
    
    def send_alert(
        self,
        title: str,
        message: str,
        severity: str = 'info',
        channels: List[str] = None
    ):
        """
        发送告警通知
        
        Args:
            title: 标题
            message: 消息
            severity: 严重程度
            channels: 通知渠道
        """
        data = {
            'title': title,
            'message': message,
            'severity': severity,
            'timestamp': time.time()
        }
        
        self.manager.trigger_event('alert', data)
    
    def send_detection(
        self,
        detection_type: str,
        details: Dict
    ):
        """
        发送检测通知
        
        Args:
            detection_type: 检测类型
            details: 详情
        """
        data = {
            'detection_type': detection_type,
            'details': details,
            'timestamp': time.time()
        }
        
        self.manager.trigger_event('detection', data)
    
    def send_system_event(
        self,
        event_type: str,
        details: Dict
    ):
        """
        发送系统事件
        
        Args:
            event_type: 事件类型
            details: 详情
        """
        data = {
            'event_type': event_type,
            'details': details,
            'timestamp': time.time()
        }
        
        self.manager.trigger_event('system', data)


# 全局实例
_integration_manager = None

def get_integration_manager() -> IntegrationManager:
    """获取集成管理器单例"""
    global _integration_manager
    if _integration_manager is None:
        _integration_manager = IntegrationManager()
    return _integration_manager


# 测试代码
if __name__ == '__main__':
    print("Testing Integration Manager...")
    
    manager = IntegrationManager()
    
    # 注册 Webhook
    webhook = manager.register_integration(
        name="Test Webhook",
        integration_type="webhook",
        config={
            'url': 'https://example.com/webhook',
            'secret': 'test-secret',
            'events': ['alert', 'detection']
        }
    )
    
    print(f"Integration: {webhook.name if webhook else 'Failed'}")
    
    # 列出集成
    integrations = manager.get_integrations()
    print(f"Total integrations: {len(integrations)}")
    
    # 触发事件
    manager.start()
    manager.trigger_event('alert', {'message': 'Test alert'})
    
    # 获取统计
    stats = manager.get_stats()
    print(f"Stats: {stats}")
    
    manager.stop()
    
    print("\nDone!")