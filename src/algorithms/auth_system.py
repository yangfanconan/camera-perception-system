"""
用户权限管理模块

功能：
1. 用户认证
2. 权限管理
3. 会话管理
4. 操作日志
"""

import hashlib
import secrets
import time
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger
from enum import Enum
import json
from pathlib import Path


class UserRole(Enum):
    """用户角色"""
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"
    GUEST = "guest"


class Permission(Enum):
    """权限"""
    # 摄像头权限
    CAMERA_VIEW = "camera_view"
    CAMERA_CONTROL = "camera_control"
    CAMERA_CONFIG = "camera_config"
    
    # 检测权限
    DETECTION_VIEW = "detection_view"
    DETECTION_CONFIG = "detection_config"
    
    # 报警权限
    ALERT_VIEW = "alert_view"
    ALERT_ACKNOWLEDGE = "alert_acknowledge"
    ALERT_CONFIG = "alert_config"
    
    # 录制权限
    RECORDING_VIEW = "recording_view"
    RECORDING_CREATE = "recording_create"
    RECORDING_DELETE = "recording_delete"
    
    # 系统权限
    SYSTEM_CONFIG = "system_config"
    USER_MANAGEMENT = "user_management"
    LOG_VIEW = "log_view"


# 角色权限映射
ROLE_PERMISSIONS: Dict[UserRole, Set[Permission]] = {
    UserRole.ADMIN: set(Permission),  # 管理员拥有所有权限
    
    UserRole.OPERATOR: {
        Permission.CAMERA_VIEW,
        Permission.CAMERA_CONTROL,
        Permission.DETECTION_VIEW,
        Permission.DETECTION_CONFIG,
        Permission.ALERT_VIEW,
        Permission.ALERT_ACKNOWLEDGE,
        Permission.RECORDING_VIEW,
        Permission.RECORDING_CREATE,
    },
    
    UserRole.VIEWER: {
        Permission.CAMERA_VIEW,
        Permission.DETECTION_VIEW,
        Permission.ALERT_VIEW,
        Permission.RECORDING_VIEW,
    },
    
    UserRole.GUEST: {
        Permission.CAMERA_VIEW,
    }
}


@dataclass
class User:
    """用户"""
    user_id: str
    username: str
    password_hash: str
    role: UserRole
    email: str = ""
    created_at: float = 0
    last_login: float = 0
    is_active: bool = True
    
    def to_dict(self) -> Dict:
        return {
            'user_id': self.user_id,
            'username': self.username,
            'role': self.role.value,
            'email': self.email,
            'created_at': self.created_at,
            'last_login': self.last_login,
            'is_active': self.is_active
        }


@dataclass
class Session:
    """会话"""
    session_id: str
    user_id: str
    created_at: float
    expires_at: float
    ip_address: str = ""
    user_agent: str = ""
    
    def is_expired(self) -> bool:
        return time.time() > self.expires_at
    
    def to_dict(self) -> Dict:
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'created_at': self.created_at,
            'expires_at': self.expires_at,
            'ip_address': self.ip_address
        }


@dataclass
class AuditLog:
    """审计日志"""
    log_id: str
    user_id: str
    action: str
    resource: str
    timestamp: float
    ip_address: str = ""
    details: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'log_id': self.log_id,
            'user_id': self.user_id,
            'action': self.action,
            'resource': self.resource,
            'timestamp': self.timestamp,
            'ip_address': self.ip_address,
            'details': self.details
        }


class PasswordManager:
    """密码管理器"""
    
    @staticmethod
    def hash_password(password: str, salt: str = None) -> tuple:
        """
        哈希密码
        
        Args:
            password: 明文密码
            salt: 盐值（可选）
            
        Returns:
            (哈希值, 盐值)
        """
        if salt is None:
            salt = secrets.token_hex(16)
        
        hash_value = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        ).hex()
        
        return hash_value, salt
    
    @staticmethod
    def verify_password(password: str, hash_value: str, salt: str) -> bool:
        """验证密码"""
        new_hash, _ = PasswordManager.hash_password(password, salt)
        return secrets.compare_digest(new_hash, hash_value)


class UserManager:
    """
    用户管理器
    
    管理用户账户
    """
    
    def __init__(self, data_dir: str = "data/users"):
        """
        初始化用户管理器
        
        Args:
            data_dir: 数据目录
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.users: Dict[str, User] = {}
        self.salt_store: Dict[str, str] = {}  # user_id -> salt
        
        # 加载用户数据
        self._load_users()
        
        # 创建默认管理员
        self._create_default_admin()
        
        logger.info(f"UserManager initialized with {len(self.users)} users")
    
    def _load_users(self):
        """加载用户数据"""
        users_file = self.data_dir / "users.json"
        
        if users_file.exists():
            try:
                with open(users_file, 'r') as f:
                    data = json.load(f)
                
                for user_data in data.get('users', []):
                    user = User(
                        user_id=user_data['user_id'],
                        username=user_data['username'],
                        password_hash=user_data['password_hash'],
                        role=UserRole(user_data['role']),
                        email=user_data.get('email', ''),
                        created_at=user_data.get('created_at', 0),
                        last_login=user_data.get('last_login', 0),
                        is_active=user_data.get('is_active', True)
                    )
                    self.users[user.user_id] = user
                
                self.salt_store = data.get('salts', {})
                
            except Exception as e:
                logger.error(f"Failed to load users: {e}")
    
    def _save_users(self):
        """保存用户数据"""
        users_file = self.data_dir / "users.json"
        
        try:
            data = {
                'users': [user.to_dict() for user in self.users.values()],
                'salts': self.salt_store
            }
            
            with open(users_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save users: {e}")
    
    def _create_default_admin(self):
        """创建默认管理员"""
        if not any(u.role == UserRole.ADMIN for u in self.users.values()):
            self.create_user(
                username="admin",
                password="admin123",
                role=UserRole.ADMIN,
                email="admin@localhost"
            )
            logger.info("Created default admin user (admin/admin123)")
    
    def create_user(
        self,
        username: str,
        password: str,
        role: UserRole,
        email: str = ""
    ) -> Optional[User]:
        """
        创建用户
        
        Args:
            username: 用户名
            password: 密码
            role: 角色
            email: 邮箱
            
        Returns:
            用户对象
        """
        # 检查用户名是否存在
        if any(u.username == username for u in self.users.values()):
            logger.warning(f"Username already exists: {username}")
            return None
        
        # 生成用户ID
        user_id = secrets.token_hex(8)
        
        # 哈希密码
        hash_value, salt = PasswordManager.hash_password(password)
        
        # 创建用户
        user = User(
            user_id=user_id,
            username=username,
            password_hash=hash_value,
            role=role,
            email=email,
            created_at=time.time()
        )
        
        self.users[user_id] = user
        self.salt_store[user_id] = salt
        
        self._save_users()
        
        logger.info(f"Created user: {username} (role={role.value})")
        return user
    
    def get_user(self, user_id: str) -> Optional[User]:
        """获取用户"""
        return self.users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """通过用户名获取用户"""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def update_user(self, user_id: str, **kwargs) -> bool:
        """更新用户信息"""
        user = self.users.get(user_id)
        if not user:
            return False
        
        for key, value in kwargs.items():
            if hasattr(user, key) and key not in ['user_id', 'password_hash']:
                setattr(user, key, value)
        
        self._save_users()
        return True
    
    def change_password(self, user_id: str, new_password: str) -> bool:
        """修改密码"""
        user = self.users.get(user_id)
        if not user:
            return False
        
        hash_value, salt = PasswordManager.hash_password(new_password)
        user.password_hash = hash_value
        self.salt_store[user_id] = salt
        
        self._save_users()
        return True
    
    def delete_user(self, user_id: str) -> bool:
        """删除用户"""
        if user_id in self.users:
            del self.users[user_id]
            if user_id in self.salt_store:
                del self.salt_store[user_id]
            
            self._save_users()
            return True
        return False
    
    def verify_user(self, username: str, password: str) -> Optional[User]:
        """
        验证用户
        
        Args:
            username: 用户名
            password: 密码
            
        Returns:
            用户对象（验证成功）或 None
        """
        user = self.get_user_by_username(username)
        
        if not user or not user.is_active:
            return None
        
        salt = self.salt_store.get(user.user_id, "")
        
        if PasswordManager.verify_password(password, user.password_hash, salt):
            user.last_login = time.time()
            self._save_users()
            return user
        
        return None
    
    def get_all_users(self) -> List[User]:
        """获取所有用户"""
        return list(self.users.values())


class SessionManager:
    """
    会话管理器
    
    管理用户会话
    """
    
    SESSION_TIMEOUT = 3600  # 1小时
    
    def __init__(self):
        """初始化会话管理器"""
        self.sessions: Dict[str, Session] = {}
        
        logger.info("SessionManager initialized")
    
    def create_session(
        self,
        user_id: str,
        ip_address: str = "",
        user_agent: str = ""
    ) -> Session:
        """
        创建会话
        
        Args:
            user_id: 用户ID
            ip_address: IP地址
            user_agent: 用户代理
            
        Returns:
            会话对象
        """
        session_id = secrets.token_urlsafe(32)
        
        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=time.time(),
            expires_at=time.time() + self.SESSION_TIMEOUT,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.sessions[session_id] = session
        
        logger.info(f"Created session for user: {user_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """获取会话"""
        session = self.sessions.get(session_id)
        
        if session and session.is_expired():
            self.delete_session(session_id)
            return None
        
        return session
    
    def refresh_session(self, session_id: str) -> bool:
        """刷新会话"""
        session = self.sessions.get(session_id)
        
        if session:
            session.expires_at = time.time() + self.SESSION_TIMEOUT
            return True
        
        return False
    
    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def delete_user_sessions(self, user_id: str):
        """删除用户所有会话"""
        to_delete = [
            sid for sid, session in self.sessions.items()
            if session.user_id == user_id
        ]
        
        for sid in to_delete:
            del self.sessions[sid]
    
    def cleanup_expired(self):
        """清理过期会话"""
        expired = [
            sid for sid, session in self.sessions.items()
            if session.is_expired()
        ]
        
        for sid in expired:
            del self.sessions[sid]
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")


class AuditLogger:
    """
    审计日志记录器
    
    记录用户操作
    """
    
    def __init__(self, data_dir: str = "data/logs"):
        """
        初始化审计日志记录器
        
        Args:
            data_dir: 数据目录
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.logs: List[AuditLog] = []
        
        logger.info("AuditLogger initialized")
    
    def log(
        self,
        user_id: str,
        action: str,
        resource: str,
        ip_address: str = "",
        details: Dict = None
    ):
        """
        记录操作日志
        
        Args:
            user_id: 用户ID
            action: 操作
            resource: 资源
            ip_address: IP地址
            details: 详情
        """
        log_entry = AuditLog(
            log_id=secrets.token_hex(8),
            user_id=user_id,
            action=action,
            resource=resource,
            timestamp=time.time(),
            ip_address=ip_address,
            details=details or {}
        )
        
        self.logs.append(log_entry)
        
        # 限制内存中的日志数量
        if len(self.logs) > 10000:
            self._save_logs()
            self.logs = self.logs[-1000:]
        
        logger.debug(f"Audit log: {action} by {user_id}")
    
    def _save_logs(self):
        """保存日志到文件"""
        if not self.logs:
            return
        
        timestamp = time.strftime("%Y%m%d")
        log_file = self.data_dir / f"audit_{timestamp}.json"
        
        try:
            # 追加模式
            existing = []
            if log_file.exists():
                with open(log_file, 'r') as f:
                    existing = json.load(f)
            
            existing.extend([log.to_dict() for log in self.logs])
            
            with open(log_file, 'w') as f:
                json.dump(existing, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save audit logs: {e}")
    
    def get_logs(
        self,
        user_id: str = None,
        action: str = None,
        start_time: float = None,
        end_time: float = None,
        limit: int = 100
    ) -> List[AuditLog]:
        """
        获取日志
        
        Args:
            user_id: 用户ID过滤
            action: 操作过滤
            start_time: 开始时间
            end_time: 结束时间
            limit: 数量限制
            
        Returns:
            日志列表
        """
        filtered = self.logs
        
        if user_id:
            filtered = [l for l in filtered if l.user_id == user_id]
        
        if action:
            filtered = [l for l in filtered if l.action == action]
        
        if start_time:
            filtered = [l for l in filtered if l.timestamp >= start_time]
        
        if end_time:
            filtered = [l for l in filtered if l.timestamp <= end_time]
        
        return filtered[-limit:]


class AuthSystem:
    """
    认证系统
    
    整合用户管理、会话管理、审计日志
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        初始化认证系统
        
        Args:
            data_dir: 数据目录
        """
        self.user_manager = UserManager(data_dir=f"{data_dir}/users")
        self.session_manager = SessionManager()
        self.audit_logger = AuditLogger(data_dir=f"{data_dir}/logs")
        
        logger.info("AuthSystem initialized")
    
    def login(
        self,
        username: str,
        password: str,
        ip_address: str = "",
        user_agent: str = ""
    ) -> Optional[Dict]:
        """
        用户登录
        
        Args:
            username: 用户名
            password: 密码
            ip_address: IP地址
            user_agent: 用户代理
            
        Returns:
            登录结果（包含会话信息）
        """
        user = self.user_manager.verify_user(username, password)
        
        if not user:
            self.audit_logger.log(
                user_id="unknown",
                action="login_failed",
                resource="auth",
                ip_address=ip_address,
                details={"username": username}
            )
            return None
        
        # 创建会话
        session = self.session_manager.create_session(
            user_id=user.user_id,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # 记录日志
        self.audit_logger.log(
            user_id=user.user_id,
            action="login",
            resource="auth",
            ip_address=ip_address
        )
        
        return {
            'session_id': session.session_id,
            'user': user.to_dict(),
            'expires_at': session.expires_at
        }
    
    def logout(self, session_id: str, ip_address: str = ""):
        """用户登出"""
        session = self.session_manager.get_session(session_id)
        
        if session:
            self.audit_logger.log(
                user_id=session.user_id,
                action="logout",
                resource="auth",
                ip_address=ip_address
            )
            
            self.session_manager.delete_session(session_id)
    
    def validate_session(self, session_id: str) -> Optional[User]:
        """验证会话"""
        session = self.session_manager.get_session(session_id)
        
        if session:
            return self.user_manager.get_user(session.user_id)
        
        return None
    
    def check_permission(self, user: User, permission: Permission) -> bool:
        """检查权限"""
        role_permissions = ROLE_PERMISSIONS.get(user.role, set())
        return permission in role_permissions
    
    def require_permission(self, session_id: str, permission: Permission) -> Optional[User]:
        """
        要求权限
        
        Args:
            session_id: 会话ID
            permission: 所需权限
            
        Returns:
            用户对象（有权限）或 None
        """
        user = self.validate_session(session_id)
        
        if user and self.check_permission(user, permission):
            return user
        
        return None
    
    def log_action(
        self,
        session_id: str,
        action: str,
        resource: str,
        details: Dict = None
    ):
        """记录操作"""
        session = self.session_manager.get_session(session_id)
        
        if session:
            self.audit_logger.log(
                user_id=session.user_id,
                action=action,
                resource=resource,
                ip_address=session.ip_address,
                details=details
            )


# 全局实例
_auth_system = None

def get_auth_system() -> AuthSystem:
    """获取认证系统单例"""
    global _auth_system
    if _auth_system is None:
        _auth_system = AuthSystem()
    return _auth_system


# 测试代码
if __name__ == '__main__':
    print("Testing Auth System...")
    
    auth = AuthSystem()
    
    # 测试登录
    result = auth.login("admin", "admin123", "127.0.0.1")
    
    if result:
        print(f"Login successful: {result['user']['username']}")
        print(f"Session ID: {result['session_id']}")
        
        # 测试权限
        user = auth.validate_session(result['session_id'])
        if user:
            print(f"Has SYSTEM_CONFIG: {auth.check_permission(user, Permission.SYSTEM_CONFIG)}")
        
        # 登出
        auth.logout(result['session_id'])
        print("Logged out")
    else:
        print("Login failed")
    
    print("\nDone!")