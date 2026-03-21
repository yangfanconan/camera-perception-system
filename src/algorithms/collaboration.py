"""
实时协作模块

功能：
1. 多用户协作
2. 实时同步
3. 注释标记
4. 协作会话
"""

import numpy as np
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger
import json
import threading
import queue
import uuid


@dataclass
class User:
    """用户"""
    user_id: str
    username: str
    color: str  # 用户标识颜色
    cursor: Optional[Dict] = None
    last_active: float = 0
    
    def to_dict(self) -> Dict:
        return {
            'user_id': self.user_id,
            'username': self.username,
            'color': self.color,
            'cursor': self.cursor,
            'last_active': self.last_active
        }


@dataclass
class Annotation:
    """注释"""
    annotation_id: str
    user_id: str
    annotation_type: str  # point, box, line, text
    data: Dict            # 注释数据
    timestamp: float
    frame_id: int = 0
    is_resolved: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'annotation_id': self.annotation_id,
            'user_id': self.user_id,
            'annotation_type': self.annotation_type,
            'data': self.data,
            'timestamp': self.timestamp,
            'frame_id': self.frame_id,
            'is_resolved': self.is_resolved
        }


@dataclass
class CollaborationSession:
    """协作会话"""
    session_id: str
    name: str
    created_at: float
    created_by: str
    users: Set[str] = field(default_factory=set)
    annotations: List[Annotation] = field(default_factory=list)
    is_active: bool = True
    
    def to_dict(self) -> Dict:
        return {
            'session_id': self.session_id,
            'name': self.name,
            'created_at': self.created_at,
            'created_by': self.created_by,
            'users': list(self.users),
            'annotation_count': len(self.annotations),
            'is_active': self.is_active
        }


class UserManager:
    """
    用户管理器
    
    管理协作用户
    """
    
    # 用户颜色列表
    USER_COLORS = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
        '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F',
        '#BB8FCE', '#85C1E9', '#F8B500', '#00CED1'
    ]
    
    def __init__(self):
        """初始化用户管理器"""
        self.users: Dict[str, User] = {}
        self.color_index = 0
        
        logger.info("UserManager initialized")
    
    def add_user(self, user_id: str, username: str) -> User:
        """
        添加用户
        
        Args:
            user_id: 用户ID
            username: 用户名
            
        Returns:
            用户对象
        """
        color = self.USER_COLORS[self.color_index % len(self.USER_COLORS)]
        self.color_index += 1
        
        user = User(
            user_id=user_id,
            username=username,
            color=color,
            last_active=time.time()
        )
        
        self.users[user_id] = user
        
        logger.info(f"User added: {username}")
        return user
    
    def remove_user(self, user_id: str):
        """移除用户"""
        if user_id in self.users:
            del self.users[user_id]
            logger.info(f"User removed: {user_id}")
    
    def get_user(self, user_id: str) -> Optional[User]:
        """获取用户"""
        return self.users.get(user_id)
    
    def update_cursor(self, user_id: str, cursor: Dict):
        """更新用户光标"""
        user = self.users.get(user_id)
        if user:
            user.cursor = cursor
            user.last_active = time.time()
    
    def get_active_users(self, timeout: float = 30.0) -> List[User]:
        """
        获取活跃用户
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            活跃用户列表
        """
        now = time.time()
        return [
            user for user in self.users.values()
            if now - user.last_active < timeout
        ]
    
    def get_all_users(self) -> List[Dict]:
        """获取所有用户"""
        return [user.to_dict() for user in self.users.values()]


class AnnotationManager:
    """
    注释管理器
    
    管理注释标记
    """
    
    def __init__(self):
        """初始化注释管理器"""
        self.annotations: Dict[str, Annotation] = {}
        self.frame_annotations: Dict[int, List[str]] = defaultdict(list)
        
        logger.info("AnnotationManager initialized")
    
    def add_annotation(
        self,
        user_id: str,
        annotation_type: str,
        data: Dict,
        frame_id: int = 0
    ) -> Annotation:
        """
        添加注释
        
        Args:
            user_id: 用户ID
            annotation_type: 注释类型
            data: 注释数据
            frame_id: 帧ID
            
        Returns:
            注释对象
        """
        annotation_id = str(uuid.uuid4())[:8]
        
        annotation = Annotation(
            annotation_id=annotation_id,
            user_id=user_id,
            annotation_type=annotation_type,
            data=data,
            timestamp=time.time(),
            frame_id=frame_id
        )
        
        self.annotations[annotation_id] = annotation
        self.frame_annotations[frame_id].append(annotation_id)
        
        logger.debug(f"Annotation added: {annotation_id}")
        return annotation
    
    def remove_annotation(self, annotation_id: str) -> bool:
        """移除注释"""
        if annotation_id in self.annotations:
            annotation = self.annotations[annotation_id]
            
            # 从帧索引中移除
            if annotation.frame_id in self.frame_annotations:
                self.frame_annotations[annotation.frame_id].remove(annotation_id)
            
            del self.annotations[annotation_id]
            return True
        
        return False
    
    def resolve_annotation(self, annotation_id: str) -> bool:
        """解决注释"""
        annotation = self.annotations.get(annotation_id)
        if annotation:
            annotation.is_resolved = True
            return True
        return False
    
    def get_annotations(self, frame_id: int = None) -> List[Annotation]:
        """获取注释"""
        if frame_id is not None:
            ids = self.frame_annotations.get(frame_id, [])
            return [self.annotations[aid] for aid in ids if aid in self.annotations]
        
        return list(self.annotations.values())
    
    def get_unresolved(self) -> List[Annotation]:
        """获取未解决的注释"""
        return [
            a for a in self.annotations.values()
            if not a.is_resolved
        ]
    
    def clear_frame(self, frame_id: int):
        """清除帧的所有注释"""
        ids = self.frame_annotations.get(frame_id, [])
        for aid in ids:
            if aid in self.annotations:
                del self.annotations[aid]
        
        del self.frame_annotations[frame_id]


class SyncManager:
    """
    同步管理器
    
    管理实时同步
    """
    
    def __init__(self):
        """初始化同步管理器"""
        self.event_queue: queue.Queue = queue.Queue()
        self.subscribers: Dict[str, List[callable]] = defaultdict(list)
        
        self.running = False
        self.sync_thread = None
        
        logger.info("SyncManager initialized")
    
    def start(self):
        """开始同步"""
        if self.running:
            return
        
        self.running = True
        self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.sync_thread.start()
        
        logger.info("Sync started")
    
    def stop(self):
        """停止同步"""
        self.running = False
        
        if self.sync_thread:
            self.sync_thread.join(timeout=2.0)
        
        logger.info("Sync stopped")
    
    def _sync_loop(self):
        """同步循环"""
        while self.running:
            try:
                event = self.event_queue.get(timeout=1.0)
                self._dispatch_event(event)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Sync error: {e}")
    
    def publish(self, event_type: str, data: Dict):
        """
        发布事件
        
        Args:
            event_type: 事件类型
            data: 事件数据
        """
        event = {
            'type': event_type,
            'data': data,
            'timestamp': time.time()
        }
        
        try:
            self.event_queue.put(event, block=False)
        except queue.Full:
            logger.warning("Event queue full")
    
    def subscribe(self, event_type: str, callback: callable):
        """
        订阅事件
        
        Args:
            event_type: 事件类型
            callback: 回调函数
        """
        self.subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: str, callback: callable):
        """取消订阅"""
        if callback in self.subscribers.get(event_type, []):
            self.subscribers[event_type].remove(callback)
    
    def _dispatch_event(self, event: Dict):
        """分发事件"""
        event_type = event['type']
        
        for callback in self.subscribers.get(event_type, []):
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Callback error: {e}")
        
        # 分发到通配符订阅者
        for callback in self.subscribers.get('*', []):
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Callback error: {e}")


class CollaborationManager:
    """
    协作管理器
    
    整合所有协作功能
    """
    
    def __init__(self):
        """初始化协作管理器"""
        self.user_manager = UserManager()
        self.annotation_manager = AnnotationManager()
        self.sync_manager = SyncManager()
        
        self.sessions: Dict[str, CollaborationSession] = {}
        self.current_session: Optional[CollaborationSession] = None
        
        # 状态
        self.current_frame_id = 0
        
        logger.info("CollaborationManager initialized")
    
    def create_session(self, name: str, created_by: str) -> CollaborationSession:
        """
        创建协作会话
        
        Args:
            name: 会话名称
            created_by: 创建者ID
            
        Returns:
            会话对象
        """
        session_id = str(uuid.uuid4())[:8]
        
        session = CollaborationSession(
            session_id=session_id,
            name=name,
            created_at=time.time(),
            created_by=created_by
        )
        
        session.users.add(created_by)
        self.sessions[session_id] = session
        
        logger.info(f"Session created: {name}")
        return session
    
    def join_session(self, session_id: str, user_id: str) -> bool:
        """加入会话"""
        session = self.sessions.get(session_id)
        
        if session and session.is_active:
            session.users.add(user_id)
            
            # 发布事件
            self.sync_manager.publish('user_joined', {
                'session_id': session_id,
                'user_id': user_id
            })
            
            return True
        
        return False
    
    def leave_session(self, session_id: str, user_id: str):
        """离开会话"""
        session = self.sessions.get(session_id)
        
        if session:
            session.users.discard(user_id)
            
            self.sync_manager.publish('user_left', {
                'session_id': session_id,
                'user_id': user_id
            })
    
    def set_current_session(self, session_id: str):
        """设置当前会话"""
        self.current_session = self.sessions.get(session_id)
    
    def add_user(self, username: str) -> User:
        """添加用户"""
        user_id = str(uuid.uuid4())[:8]
        return self.user_manager.add_user(user_id, username)
    
    def update_cursor(self, user_id: str, x: float, y: float):
        """更新光标位置"""
        self.user_manager.update_cursor(user_id, {
            'x': x,
            'y': y,
            'frame_id': self.current_frame_id
        })
        
        # 同步
        self.sync_manager.publish('cursor_update', {
            'user_id': user_id,
            'x': x,
            'y': y
        })
    
    def add_annotation(
        self,
        user_id: str,
        annotation_type: str,
        data: Dict,
        frame_id: int = None
    ) -> Annotation:
        """添加注释"""
        if frame_id is None:
            frame_id = self.current_frame_id
        
        annotation = self.annotation_manager.add_annotation(
            user_id=user_id,
            annotation_type=annotation_type,
            data=data,
            frame_id=frame_id
        )
        
        # 同步
        self.sync_manager.publish('annotation_added', {
            'annotation': annotation.to_dict()
        })
        
        return annotation
    
    def remove_annotation(self, annotation_id: str) -> bool:
        """移除注释"""
        result = self.annotation_manager.remove_annotation(annotation_id)
        
        if result:
            self.sync_manager.publish('annotation_removed', {
                'annotation_id': annotation_id
            })
        
        return result
    
    def get_annotations(self, frame_id: int = None) -> List[Dict]:
        """获取注释"""
        annotations = self.annotation_manager.get_annotations(frame_id)
        return [a.to_dict() for a in annotations]
    
    def get_users(self) -> List[Dict]:
        """获取用户列表"""
        return self.user_manager.get_all_users()
    
    def get_cursors(self) -> Dict[str, Dict]:
        """获取所有光标"""
        return {
            user_id: user.cursor
            for user_id, user in self.user_manager.users.items()
            if user.cursor
        }
    
    def set_frame(self, frame_id: int):
        """设置当前帧"""
        self.current_frame_id = frame_id
        
        self.sync_manager.publish('frame_changed', {
            'frame_id': frame_id
        })
    
    def start(self):
        """开始协作"""
        self.sync_manager.start()
    
    def stop(self):
        """停止协作"""
        self.sync_manager.stop()
    
    def subscribe(self, event_type: str, callback: callable):
        """订阅事件"""
        self.sync_manager.subscribe(event_type, callback)
    
    def get_session_info(self) -> Optional[Dict]:
        """获取当前会话信息"""
        if self.current_session:
            return self.current_session.to_dict()
        return None


class CollaborationRenderer:
    """
    协作渲染器
    
    在帧上渲染协作元素
    """
    
    def __init__(self, collaboration_manager: CollaborationManager):
        """
        初始化渲染器
        
        Args:
            collaboration_manager: 协作管理器
        """
        self.manager = collaboration_manager
        
        logger.info("CollaborationRenderer initialized")
    
    def render(self, frame: np.ndarray, frame_id: int = None) -> np.ndarray:
        """
        渲染协作元素
        
        Args:
            frame: 图像帧
            frame_id: 帧ID
            
        Returns:
            渲染后的帧
        """
        if frame_id is None:
            frame_id = self.manager.current_frame_id
        
        result = frame.copy()
        
        # 渲染注释
        annotations = self.manager.annotation_manager.get_annotations(frame_id)
        
        for annotation in annotations:
            result = self._render_annotation(result, annotation)
        
        # 渲染光标
        cursors = self.manager.get_cursors()
        
        for user_id, cursor in cursors.items():
            if cursor and cursor.get('frame_id') == frame_id:
                result = self._render_cursor(result, user_id, cursor)
        
        return result
    
    def _render_annotation(self, frame: np.ndarray, annotation: Annotation) -> np.ndarray:
        """渲染注释"""
        import cv2
        
        user = self.manager.user_manager.get_user(annotation.user_id)
        color = self._hex_to_bgr(user.color) if user else (0, 255, 0)
        
        if annotation.annotation_type == 'point':
            x, y = annotation.data.get('x', 0), annotation.data.get('y', 0)
            cv2.circle(frame, (int(x), int(y)), 5, color, -1)
            
        elif annotation.annotation_type == 'box':
            x1, y1 = annotation.data.get('x1', 0), annotation.data.get('y1', 0)
            x2, y2 = annotation.data.get('x2', 0), annotation.data.get('y2', 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
        elif annotation.annotation_type == 'line':
            x1, y1 = annotation.data.get('x1', 0), annotation.data.get('y1', 0)
            x2, y2 = annotation.data.get('x2', 0), annotation.data.get('y2', 0)
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
        elif annotation.annotation_type == 'text':
            x, y = annotation.data.get('x', 0), annotation.data.get('y', 0)
            text = annotation.data.get('text', '')
            cv2.putText(frame, text, (int(x), int(y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def _render_cursor(self, frame: np.ndarray, user_id: str, cursor: Dict) -> np.ndarray:
        """渲染光标"""
        import cv2
        
        user = self.manager.user_manager.get_user(user_id)
        if not user:
            return frame
        
        color = self._hex_to_bgr(user.color)
        x, y = cursor.get('x', 0), cursor.get('y', 0)
        
        # 绘制光标
        cv2.circle(frame, (int(x), int(y)), 8, color, 2)
        cv2.line(frame, (int(x), int(y) - 12), (int(x), int(y) + 12), color, 1)
        cv2.line(frame, (int(x) - 12, int(y)), (int(x) + 12, int(y)), color, 1)
        
        # 绘制用户名
        cv2.putText(frame, user.username, (int(x) + 10, int(y) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def _hex_to_bgr(self, hex_color: str) -> tuple:
        """十六进制颜色转 BGR"""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (b, g, r)


# 全局实例
_collaboration_manager = None

def get_collaboration_manager() -> CollaborationManager:
    """获取协作管理器单例"""
    global _collaboration_manager
    if _collaboration_manager is None:
        _collaboration_manager = CollaborationManager()
    return _collaboration_manager


# 测试代码
if __name__ == '__main__':
    print("Testing Collaboration Manager...")
    
    manager = CollaborationManager()
    
    # 添加用户
    user1 = manager.add_user("Alice")
    user2 = manager.add_user("Bob")
    
    print(f"Users: {[u['username'] for u in manager.get_users()]}")
    
    # 创建会话
    session = manager.create_session("Test Session", user1.user_id)
    print(f"Session: {session.name}")
    
    # 加入会话
    manager.join_session(session.session_id, user2.user_id)
    
    # 添加注释
    annotation = manager.add_annotation(
        user_id=user1.user_id,
        annotation_type='point',
        data={'x': 100, 'y': 100}
    )
    print(f"Annotation: {annotation.annotation_id}")
    
    # 更新光标
    manager.update_cursor(user2.user_id, 200, 200)
    
    # 获取光标
    cursors = manager.get_cursors()
    print(f"Cursors: {list(cursors.keys())}")
    
    print("\nDone!")