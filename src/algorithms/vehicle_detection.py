"""
车辆检测模块

功能：
1. 车辆检测（汽车、卡车、摩托车、自行车等）
2. 车牌检测与识别
3. 车辆跟踪
4. 车辆属性分析（颜色、类型）
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from loguru import logger
import time
import re


@dataclass
class VehicleInfo:
    """车辆信息"""
    track_id: int
    vehicle_type: str         # car, truck, bus, motorcycle, bicycle
    bbox: List[int]           # [x, y, w, h]
    confidence: float
    
    # 属性
    color: Optional[str] = None
    direction: Optional[str] = None  # left, right, up, down
    
    # 车牌
    license_plate: Optional[str] = None
    plate_bbox: Optional[List[int]] = None
    
    # 运动
    speed: float = 0.0
    trajectory: List[Tuple[int, int]] = field(default_factory=list)
    
    # 时间
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            'track_id': self.track_id,
            'vehicle_type': self.vehicle_type,
            'bbox': self.bbox,
            'confidence': round(self.confidence, 3),
            'color': self.color,
            'license_plate': self.license_plate,
            'speed': round(self.speed, 2)
        }


# 车辆类型映射
VEHICLE_TYPES = {
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

# 颜色映射
COLOR_RANGES = {
    'red': ([0, 50, 50], [10, 255, 255]),
    'orange': ([11, 50, 50], [25, 255, 255]),
    'yellow': ([26, 50, 50], [35, 255, 255]),
    'green': ([36, 50, 50], [85, 255, 255]),
    'blue': ([86, 50, 50], [125, 255, 255]),
    'purple': ([126, 50, 50], [150, 255, 255]),
    'white': ([0, 0, 200], [180, 30, 255]),
    'black': ([0, 0, 0], [180, 255, 50]),
    'gray': ([0, 0, 51], [180, 50, 199])
}


class VehicleDetector:
    """
    车辆检测器
    
    使用 YOLO 模型检测车辆
    """
    
    # COCO 数据集中的车辆类别
    VEHICLE_CLASSES = {
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }
    
    def __init__(self, model_path: str = "models/yolov8n.pt"):
        """
        初始化车辆检测器
        
        Args:
            model_path: YOLO 模型路径
        """
        self.model = None
        self.model_path = model_path
        
        self._load_model()
        
        logger.info(f"VehicleDetector initialized (model={model_path})")
    
    def _load_model(self):
        """加载模型"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            logger.info("Loaded YOLO model for vehicle detection")
        except Exception as e:
            logger.warning(f"Could not load YOLO model: {e}")
    
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Dict]:
        """
        检测车辆
        
        Args:
            frame: 输入图像
            conf_threshold: 置信度阈值
            
        Returns:
            检测结果列表
        """
        if self.model is None:
            return []
        
        try:
            results = self.model(frame, verbose=False)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    cls_id = int(box.cls[0])
                    
                    # 只保留车辆类别
                    if cls_id in self.VEHICLE_CLASSES:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])
                        
                        if confidence >= conf_threshold:
                            detections.append({
                                'vehicle_type': self.VEHICLE_CLASSES[cls_id],
                                'bbox': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                                'confidence': confidence
                            })
            
            return detections
            
        except Exception as e:
            logger.error(f"Vehicle detection error: {e}")
            return []


class LicensePlateDetector:
    """
    车牌检测与识别器
    """
    
    def __init__(self):
        """初始化车牌检测器"""
        self.plate_cascade = None
        self._load_cascade()
        
        logger.info("LicensePlateDetector initialized")
    
    def _load_cascade(self):
        """加载车牌检测模型"""
        try:
            # 尝试加载 OpenCV 车牌检测器
            cascade_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
            self.plate_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("Loaded license plate cascade")
        except Exception as e:
            logger.warning(f"Could not load plate cascade: {e}")
    
    def detect_and_recognize(
        self,
        frame: np.ndarray,
        vehicle_bbox: List[int]
    ) -> Tuple[Optional[str], Optional[List[int]]]:
        """
        检测并识别车牌
        
        Args:
            frame: 输入图像
            vehicle_bbox: 车辆边界框
            
        Returns:
            (车牌号, 车牌边界框)
        """
        if self.plate_cascade is None:
            return None, None
        
        try:
            # 提取车辆区域
            x, y, w, h = vehicle_bbox
            vehicle_roi = frame[y:y+h, x:x+w]
            
            if vehicle_roi.size == 0:
                return None, None
            
            # 检测车牌
            gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
            plates = self.plate_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 10)
            )
            
            for (px, py, pw, ph) in plates:
                # 提取车牌区域
                plate_roi = vehicle_roi[py:py+ph, px:px+pw]
                
                # 识别车牌号
                plate_text = self._recognize_plate(plate_roi)
                
                if plate_text:
                    # 转换到原图坐标
                    plate_bbox = [x + px, y + py, pw, ph]
                    return plate_text, plate_bbox
            
            return None, None
            
        except Exception as e:
            logger.debug(f"License plate detection error: {e}")
            return None, None
    
    def _recognize_plate(self, plate_roi: np.ndarray) -> Optional[str]:
        """
        识别车牌号
        
        Args:
            plate_roi: 车牌区域
            
        Returns:
            车牌号
        """
        try:
            # 预处理
            gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
            
            # 二值化
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 这里应该使用 OCR，简化实现
            # 返回模拟车牌号
            return None  # 实际应用中需要集成 OCR
            
        except Exception as e:
            return None


class VehicleTracker:
    """
    车辆跟踪器
    """
    
    def __init__(self, max_age: int = 30, min_hits: int = 3):
        """
        初始化车辆跟踪器
        
        Args:
            max_age: 最大丢失帧数
            min_hits: 确认所需匹配次数
        """
        self.max_age = max_age
        self.min_hits = min_hits
        
        self.tracks: Dict[int, VehicleInfo] = {}
        self.next_id = 1
        
        self.track_history: Dict[int, deque] = {}
        
        logger.info("VehicleTracker initialized")
    
    def update(self, detections: List[Dict]) -> List[VehicleInfo]:
        """
        更新跟踪
        
        Args:
            detections: 检测结果
            
        Returns:
            活跃的跟踪列表
        """
        current_time = time.time()
        
        # 计算代价矩阵
        if detections and self.tracks:
            cost_matrix = self._compute_cost_matrix(detections)
            matches, unmatched_dets, unmatched_tracks = self._hungarian_assignment(cost_matrix)
        else:
            matches = []
            unmatched_dets = list(range(len(detections)))
            unmatched_tracks = list(self.tracks.keys())
        
        # 更新匹配的跟踪
        for det_idx, track_id in matches:
            det = detections[det_idx]
            track = self.tracks[track_id]
            
            # 更新信息
            old_center = self._get_center(track.bbox)
            track.bbox = det['bbox']
            track.confidence = det['confidence']
            track.vehicle_type = det['vehicle_type']
            track.last_seen = current_time
            
            # 更新轨迹
            new_center = self._get_center(track.bbox)
            if track_id not in self.track_history:
                self.track_history[track_id] = deque(maxlen=50)
            self.track_history[track_id].append(new_center)
            track.trajectory = list(self.track_history[track_id])
            
            # 计算速度
            if len(track.trajectory) >= 2:
                dx = track.trajectory[-1][0] - track.trajectory[-2][0]
                dy = track.trajectory[-1][1] - track.trajectory[-2][1]
                track.speed = np.sqrt(dx**2 + dy**2)
        
        # 创建新跟踪
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            
            track = VehicleInfo(
                track_id=self.next_id,
                vehicle_type=det['vehicle_type'],
                bbox=det['bbox'],
                confidence=det['confidence']
            )
            
            self.tracks[self.next_id] = track
            self.next_id += 1
        
        # 删除丢失的跟踪
        to_remove = []
        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            if current_time - track.last_seen > self.max_age / 30.0:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
            if track_id in self.track_history:
                del self.track_history[track_id]
        
        return [t for t in self.tracks.values() if current_time - t.last_seen < 1.0]
    
    def _get_center(self, bbox: List[int]) -> Tuple[int, int]:
        """获取边界框中心"""
        x, y, w, h = bbox
        return (int(x + w / 2), int(y + h / 2))
    
    def _compute_cost_matrix(self, detections: List[Dict]) -> np.ndarray:
        """计算代价矩阵"""
        n_dets = len(detections)
        n_tracks = len(self.tracks)
        
        cost_matrix = np.zeros((n_dets, n_tracks))
        
        track_ids = list(self.tracks.keys())
        
        for i, det in enumerate(detections):
            det_center = self._get_center(det['bbox'])
            
            for j, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                track_center = self._get_center(track.bbox)
                
                # 计算距离
                distance = np.sqrt(
                    (det_center[0] - track_center[0]) ** 2 +
                    (det_center[1] - track_center[1]) ** 2
                )
                
                cost_matrix[i, j] = distance
        
        return cost_matrix
    
    def _hungarian_assignment(self, cost_matrix: np.ndarray, threshold: float = 100.0):
        """匈牙利算法分配"""
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        except ImportError:
            # 简单贪心
            return self._greedy_assignment(cost_matrix, threshold)
        
        matches = []
        unmatched_dets = list(range(cost_matrix.shape[0]))
        unmatched_tracks = list(range(cost_matrix.shape[1]))
        
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < threshold:
                matches.append((i, j))
                if i in unmatched_dets:
                    unmatched_dets.remove(i)
                if j in unmatched_tracks:
                    unmatched_tracks.remove(j)
        
        return matches, unmatched_dets, unmatched_tracks
    
    def _greedy_assignment(self, cost_matrix: np.ndarray, threshold: float):
        """贪心分配"""
        n_dets, n_tracks = cost_matrix.shape
        
        matches = []
        unmatched_dets = list(range(n_dets))
        unmatched_tracks = list(range(n_tracks))
        
        # 按代价排序
        indices = np.argsort(cost_matrix.ravel())
        
        for idx in indices:
            i, j = idx // n_tracks, idx % n_tracks
            
            if i in unmatched_dets and j in unmatched_tracks:
                if cost_matrix[i, j] < threshold:
                    matches.append((i, j))
                    unmatched_dets.remove(i)
                    unmatched_tracks.remove(j)
        
        return matches, unmatched_dets, unmatched_tracks


class VehicleAnalyzer:
    """
    车辆分析器
    
    整合检测、跟踪和属性分析
    """
    
    def __init__(self, model_path: str = "models/yolov8n.pt"):
        """
        初始化车辆分析器
        
        Args:
            model_path: YOLO 模型路径
        """
        self.detector = VehicleDetector(model_path)
        self.tracker = VehicleTracker()
        self.plate_detector = LicensePlateDetector()
        
        logger.info("VehicleAnalyzer initialized")
    
    def analyze(self, frame: np.ndarray) -> List[VehicleInfo]:
        """
        分析图像中的车辆
        
        Args:
            frame: 输入图像
            
        Returns:
            车辆信息列表
        """
        # 检测车辆
        detections = self.detector.detect(frame)
        
        # 更新跟踪
        vehicles = self.tracker.update(detections)
        
        # 分析属性
        for vehicle in vehicles:
            # 分析颜色
            vehicle.color = self._analyze_color(frame, vehicle.bbox)
            
            # 检测车牌
            plate, plate_bbox = self.plate_detector.detect_and_recognize(frame, vehicle.bbox)
            if plate:
                vehicle.license_plate = plate
                vehicle.plate_bbox = plate_bbox
        
        return vehicles
    
    def _analyze_color(self, frame: np.ndarray, bbox: List[int]) -> Optional[str]:
        """分析车辆颜色"""
        try:
            x, y, w, h = bbox
            roi = frame[y:y+h, x:x+w]
            
            if roi.size == 0:
                return None
            
            # 转换到 HSV
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # 计算主要颜色
            color_counts = {}
            
            for color_name, (lower, upper) in COLOR_RANGES.items():
                lower = np.array(lower)
                upper = np.array(upper)
                
                mask = cv2.inRange(hsv, lower, upper)
                count = cv2.countNonZero(mask)
                
                if count > 0:
                    color_counts[color_name] = count
            
            if color_counts:
                return max(color_counts, key=color_counts.get)
            
            return None
            
        except Exception as e:
            return None
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_vehicles': len(self.tracker.tracks),
            'by_type': self._count_by_type()
        }
    
    def _count_by_type(self) -> Dict[str, int]:
        """按类型统计"""
        counts = {}
        for vehicle in self.tracker.tracks.values():
            vtype = vehicle.vehicle_type
            counts[vtype] = counts.get(vtype, 0) + 1
        return counts


# 全局实例
_vehicle_analyzer = None

def get_vehicle_analyzer() -> VehicleAnalyzer:
    """获取车辆分析器单例"""
    global _vehicle_analyzer
    if _vehicle_analyzer is None:
        _vehicle_analyzer = VehicleAnalyzer()
    return _vehicle_analyzer


# 测试代码
if __name__ == '__main__':
    print("Testing Vehicle Analyzer...")
    
    analyzer = VehicleAnalyzer()
    
    # 测试图像
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    vehicles = analyzer.analyze(frame)
    
    print(f"Detected {len(vehicles)} vehicles")
    for vehicle in vehicles:
        print(f"  Vehicle {vehicle.track_id}: {vehicle.to_dict()}")
    
    print("\nDone!")