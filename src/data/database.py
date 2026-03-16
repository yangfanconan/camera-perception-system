"""
数据库模块 - 检测数据持久化
功能：
1. SQLite 数据库存储
2. 检测记录管理
3. 数据统计分析
4. 历史数据查询
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from loguru import logger
import json
import threading


@dataclass
class DetectionRecord:
    """检测记录"""
    id: Optional[int] = None
    timestamp: str = ""
    person_count: int = 0
    hand_count: int = 0
    persons_json: str = ""
    hands_json: str = ""
    frame_width: int = 0
    frame_height: int = 0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_detection_data(
        cls,
        persons: List[Dict],
        hands: List[Dict],
        frame_shape: Tuple[int, int, int]
    ) -> 'DetectionRecord':
        """从检测数据创建记录"""
        return cls(
            timestamp=datetime.now().isoformat(),
            person_count=len(persons),
            hand_count=len(hands),
            persons_json=json.dumps(persons),
            hands_json=json.dumps(hands),
            frame_width=frame_shape[1] if len(frame_shape) > 1 else 0,
            frame_height=frame_shape[0] if len(frame_shape) > 0 else 0
        )


@dataclass
class PersonMetrics:
    """人体指标记录"""
    id: Optional[int] = None
    record_id: int = 0
    timestamp: str = ""
    track_id: Optional[int] = None
    distance: float = 0.0
    height: float = 0.0
    topview_x: float = 0.0
    topview_y: float = 0.0
    confidence: float = 0.0


@dataclass
class HandMetrics:
    """手部指标记录"""
    id: Optional[int] = None
    record_id: int = 0
    timestamp: str = ""
    hand_type: str = ""
    size: float = 0.0
    distance: float = 0.0
    topview_x: float = 0.0
    topview_y: float = 0.0


class Database:
    """数据库管理类"""
    
    def __init__(self, db_path: str = "data/perception.db"):
        """
        初始化数据库
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self._local = threading.local()
        
        # 确保目录存在
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据库
        self._init_db()
        
        logger.info(f"Database initialized: {db_path}")
    
    @property
    def connection(self) -> sqlite3.Connection:
        """获取线程本地连接"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    @contextmanager
    def cursor(self):
        """上下文管理器获取游标"""
        conn = self.connection
        try:
            cur = conn.cursor()
            yield cur
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cur.close()
    
    def _init_db(self):
        """初始化数据库表"""
        with self.cursor() as cur:
            # 检测记录表
            cur.execute('''
                CREATE TABLE IF NOT EXISTS detection_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    person_count INTEGER DEFAULT 0,
                    hand_count INTEGER DEFAULT 0,
                    persons_json TEXT,
                    hands_json TEXT,
                    frame_width INTEGER,
                    frame_height INTEGER
                )
            ''')
            
            # 人体指标表
            cur.execute('''
                CREATE TABLE IF NOT EXISTS person_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id INTEGER,
                    timestamp TEXT,
                    track_id INTEGER,
                    distance REAL,
                    height REAL,
                    topview_x REAL,
                    topview_y REAL,
                    confidence REAL,
                    FOREIGN KEY (record_id) REFERENCES detection_records(id)
                )
            ''')
            
            # 手部指标表
            cur.execute('''
                CREATE TABLE IF NOT EXISTS hand_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id INTEGER,
                    timestamp TEXT,
                    hand_type TEXT,
                    size REAL,
                    distance REAL,
                    topview_x REAL,
                    topview_y REAL,
                    FOREIGN KEY (record_id) REFERENCES detection_records(id)
                )
            ''')
            
            # 创建索引
            cur.execute('CREATE INDEX IF NOT EXISTS idx_records_timestamp ON detection_records(timestamp)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_person_timestamp ON person_metrics(timestamp)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_hand_timestamp ON hand_metrics(timestamp)')
            
            # 创建配置表
            cur.execute('''
                CREATE TABLE IF NOT EXISTS system_config (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                )
            ''')
    
    # ==================== 检测记录 CRUD ====================
    
    def add_detection_record(
        self,
        persons: List[Dict],
        hands: List[Dict],
        frame_shape: Tuple[int, int, int]
    ) -> int:
        """
        添加检测记录
        
        Args:
            persons: 人体检测结果
            hands: 手部检测结果
            frame_shape: 帧形状
            
        Returns:
            记录 ID
        """
        record = DetectionRecord.from_detection_data(persons, hands, frame_shape)
        
        with self.cursor() as cur:
            cur.execute('''
                INSERT INTO detection_records 
                (timestamp, person_count, hand_count, persons_json, hands_json, frame_width, frame_height)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.timestamp,
                record.person_count,
                record.hand_count,
                record.persons_json,
                record.hands_json,
                record.frame_width,
                record.frame_height
            ))
            return cur.lastrowid
    
    def get_detection_record(self, record_id: int) -> Optional[DetectionRecord]:
        """获取检测记录"""
        with self.cursor() as cur:
            cur.execute('SELECT * FROM detection_records WHERE id = ?', (record_id,))
            row = cur.fetchone()
            if row:
                return DetectionRecord(**dict(row))
        return None
    
    def get_detection_records(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[DetectionRecord]:
        """获取检测记录列表"""
        query = 'SELECT * FROM detection_records WHERE 1=1'
        params = []
        
        if start_time:
            query += ' AND timestamp >= ?'
            params.append(start_time.isoformat())
        
        if end_time:
            query += ' AND timestamp <= ?'
            params.append(end_time.isoformat())
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        with self.cursor() as cur:
            cur.execute(query, params)
            return [DetectionRecord(**dict(row)) for row in cur.fetchall()]
    
    def delete_old_records(self, days: int = 7):
        """删除旧记录"""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        with self.cursor() as cur:
            cur.execute('DELETE FROM detection_records WHERE timestamp < ?', (cutoff,))
            cur.execute('DELETE FROM person_metrics WHERE timestamp < ?', (cutoff,))
            cur.execute('DELETE FROM hand_metrics WHERE timestamp < ?', (cutoff,))
        
        logger.info(f"Deleted records older than {days} days")
    
    # ==================== 人体指标 CRUD ====================
    
    def add_person_metrics(self, record_id: int, person: Dict[str, Any]) -> int:
        """添加人体指标"""
        metrics = PersonMetrics(
            record_id=record_id,
            timestamp=datetime.now().isoformat(),
            track_id=person.get('track_id'),
            distance=person.get('distance', 0.0),
            height=person.get('height', 0.0),
            topview_x=person.get('topview', {}).get('x', 0.0),
            topview_y=person.get('topview', {}).get('y', 0.0),
            confidence=person.get('distance_confidence', 0.0)
        )
        
        with self.cursor() as cur:
            cur.execute('''
                INSERT INTO person_metrics 
                (record_id, timestamp, track_id, distance, height, topview_x, topview_y, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.record_id,
                metrics.timestamp,
                metrics.track_id,
                metrics.distance,
                metrics.height,
                metrics.topview_x,
                metrics.topview_y,
                metrics.confidence
            ))
            return cur.lastrowid
    
    def get_person_metrics(
        self,
        record_id: Optional[int] = None,
        track_id: Optional[int] = None,
        limit: int = 100
    ) -> List[PersonMetrics]:
        """获取人体指标"""
        query = 'SELECT * FROM person_metrics WHERE 1=1'
        params = []
        
        if record_id:
            query += ' AND record_id = ?'
            params.append(record_id)
        
        if track_id is not None:
            query += ' AND track_id = ?'
            params.append(track_id)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        with self.cursor() as cur:
            cur.execute(query, params)
            return [PersonMetrics(**dict(row)) for row in cur.fetchall()]
    
    # ==================== 手部指标 CRUD ====================
    
    def add_hand_metrics(self, record_id: int, hand: Dict[str, Any]) -> int:
        """添加手部指标"""
        metrics = HandMetrics(
            record_id=record_id,
            timestamp=datetime.now().isoformat(),
            hand_type=hand.get('hand_type', 'Unknown'),
            size=hand.get('size', 0.0),
            distance=hand.get('distance', 0.0),
            topview_x=hand.get('topview', {}).get('x', 0.0),
            topview_y=hand.get('topview', {}).get('y', 0.0)
        )
        
        with self.cursor() as cur:
            cur.execute('''
                INSERT INTO hand_metrics 
                (record_id, timestamp, hand_type, size, distance, topview_x, topview_y)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.record_id,
                metrics.timestamp,
                metrics.hand_type,
                metrics.size,
                metrics.distance,
                metrics.topview_x,
                metrics.topview_y
            ))
            return cur.lastrowid
    
    def get_hand_metrics(
        self,
        record_id: Optional[int] = None,
        limit: int = 100
    ) -> List[HandMetrics]:
        """获取手部指标"""
        query = 'SELECT * FROM hand_metrics WHERE 1=1'
        params = []
        
        if record_id:
            query += ' AND record_id = ?'
            params.append(record_id)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        with self.cursor() as cur:
            cur.execute(query, params)
            return [HandMetrics(**dict(row)) for row in cur.fetchall()]
    
    # ==================== 系统配置 ====================
    
    def save_config(self, key: str, value: Any):
        """保存配置"""
        with self.cursor() as cur:
            cur.execute('''
                INSERT OR REPLACE INTO system_config (key, value, updated_at)
                VALUES (?, ?, ?)
            ''', (key, json.dumps(value), datetime.now().isoformat()))
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置"""
        with self.cursor() as cur:
            cur.execute('SELECT value FROM system_config WHERE key = ?', (key,))
            row = cur.fetchone()
            if row:
                return json.loads(row['value'])
        return default
    
    # ==================== 统计分析 ====================
    
    def get_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """获取统计数据"""
        query_params = []
        time_filter = ''
        
        if start_time:
            time_filter += ' AND timestamp >= ?'
            query_params.append(start_time.isoformat())
        
        if end_time:
            time_filter += ' AND timestamp <= ?'
            query_params.append(end_time.isoformat())
        
        with self.cursor() as cur:
            # 总记录数
            cur.execute(f'''
                SELECT COUNT(*) as count FROM detection_records WHERE 1=1 {time_filter}
            ''', query_params)
            total_records = cur.fetchone()['count']
            
            # 平均人数
            cur.execute(f'''
                SELECT AVG(person_count) as avg FROM detection_records WHERE 1=1 {time_filter}
            ''', query_params)
            avg_persons = cur.fetchone()['avg'] or 0
            
            # 平均手数
            cur.execute(f'''
                SELECT AVG(hand_count) as avg FROM detection_records WHERE 1=1 {time_filter}
            ''', query_params)
            avg_hands = cur.fetchone()['avg'] or 0
            
            # 人体指标统计
            cur.execute(f'''
                SELECT AVG(distance) as avg_dist, AVG(height) as avg_height,
                       MAX(height) as max_height, MIN(height) as min_height
                FROM person_metrics WHERE 1=1 {time_filter}
            ''', query_params)
            person_stats = cur.fetchone()
            
            # 手部指标统计
            cur.execute(f'''
                SELECT AVG(size) as avg_size, AVG(distance) as avg_distance
                FROM hand_metrics WHERE 1=1 {time_filter}
            ''', query_params)
            hand_stats = cur.fetchone()
        
        return {
            "total_records": total_records,
            "avg_persons": round(avg_persons, 2),
            "avg_hands": round(avg_hands, 2),
            "person_metrics": {
                "avg_distance": round(person_stats['avg_dist'] or 0, 2),
                "avg_height": round(person_stats['avg_height'] or 0, 1),
                "max_height": round(person_stats['max_height'] or 0, 1),
                "min_height": round(person_stats['min_height'] or 0, 1)
            },
            "hand_metrics": {
                "avg_size": round(hand_stats['avg_size'] or 0, 1),
                "avg_distance": round(hand_stats['avg_distance'] or 0, 2)
            }
        }
    
    def get_hourly_distribution(self, date: Optional[datetime] = None) -> List[Dict]:
        """获取小时分布"""
        if date is None:
            date = datetime.now()
        
        date_str = date.strftime('%Y-%m-%d')
        
        with self.cursor() as cur:
            cur.execute('''
                SELECT 
                    substr(timestamp, 12, 2) as hour,
                    AVG(person_count) as avg_persons,
                    AVG(hand_count) as avg_hands
                FROM detection_records
                WHERE timestamp LIKE ?
                GROUP BY hour
                ORDER BY hour
            ''', (f'{date_str}%',))
            
            return [dict(row) for row in cur.fetchall()]
    
    def close(self):
        """关闭数据库"""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
            logger.info("Database connection closed")


# ==================== 全局数据库实例 ====================

_db_instance: Optional[Database] = None


def get_database(db_path: str = "data/perception.db") -> Database:
    """获取数据库实例"""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database(db_path)
    return _db_instance


def close_database():
    """关闭数据库"""
    global _db_instance
    if _db_instance:
        _db_instance.close()
        _db_instance = None


# ==================== 主函数（测试） ====================

def main():
    """测试数据库功能"""
    db = Database("data/test_perception.db")
    
    # 添加测试记录
    record_id = db.add_detection_record(
        persons=[
            {
                "bbox": [100, 100, 200, 500],
                "distance": 3.5,
                "height": 175.0,
                "topview": {"x": 450, "y": 250}
            }
        ],
        hands=[
            {
                "hand_type": "Right",
                "size": 18.5,
                "distance": 1.2,
                "topview": {"x": 380, "y": 280}
            }
        ],
        frame_shape=(1080, 1920, 3)
    )
    
    print(f"Created record: {record_id}")
    
    # 添加指标
    db.add_person_metrics(record_id, {
        "distance": 3.5,
        "height": 175.0,
        "topview": {"x": 450, "y": 250},
        "distance_confidence": 0.95
    })
    
    db.add_hand_metrics(record_id, {
        "hand_type": "Right",
        "size": 18.5,
        "distance": 1.2,
        "topview": {"x": 380, "y": 280}
    })
    
    # 获取记录
    records = db.get_detection_records(limit=10)
    print(f"Records: {len(records)}")
    
    # 获取统计
    stats = db.get_statistics()
    print(f"Statistics: {json.dumps(stats, indent=2)}")
    
    # 保存配置
    db.save_config("camera_id", 0)
    db.save_config("calibration_file", "calibration_data/calib_params.json")
    
    # 获取配置
    camera_id = db.get_config("camera_id")
    print(f"Camera ID: {camera_id}")
    
    db.close()


if __name__ == '__main__':
    main()
