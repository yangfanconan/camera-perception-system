"""
人脸检测与属性分析模块

功能：
1. 人脸检测
2. 人脸属性分析（年龄、性别、表情）
3. 人脸质量评估
4. 人脸特征提取
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from loguru import logger
import time


@dataclass
class FaceInfo:
    """人脸信息"""
    face_id: int
    bbox: List[int]              # [x, y, w, h]
    landmarks: List[Tuple[int, int]]  # 5点或68点
    confidence: float
    
    # 属性
    age: Optional[int] = None
    gender: Optional[str] = None
    emotion: Optional[str] = None
    
    # 质量
    quality_score: float = 0.0
    blur_score: float = 0.0
    brightness: float = 0.0
    
    # 特征
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        return {
            'face_id': self.face_id,
            'bbox': self.bbox,
            'confidence': round(self.confidence, 3),
            'age': self.age,
            'gender': self.gender,
            'emotion': self.emotion,
            'quality_score': round(self.quality_score, 3)
        }


class FaceDetector:
    """
    人脸检测器
    
    使用 OpenCV DNN 或 MediaPipe
    """
    
    def __init__(self, backend: str = "opencv"):
        """
        初始化人脸检测器
        
        Args:
            backend: 检测后端 ("opencv", "mediapipe")
        """
        self.backend = backend
        self.face_cascade = None
        self.face_net = None
        self.mp_face_detection = None
        
        self._init_detector()
        
        self.next_face_id = 1
        
        logger.info(f"FaceDetector initialized (backend={backend})")
    
    def _init_detector(self):
        """初始化检测器"""
        if self.backend == "opencv":
            # 使用 Haar 级联分类器
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # 尝试加载 DNN 模型
            try:
                model_file = "models/opencv_face_detector/opencv_face_detector_uint8.pb"
                config_file = "models/opencv_face_detector/opencv_face_detector.pbtxt"
                
                import os
                if os.path.exists(model_file):
                    self.face_net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
                    logger.info("Loaded DNN face detector")
            except Exception as e:
                logger.warning(f"Could not load DNN model: {e}")
        
        elif self.backend == "mediapipe":
            try:
                import mediapipe as mp
                self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=0.5
                )
                logger.info("Loaded MediaPipe face detector")
            except ImportError:
                logger.warning("MediaPipe not available, falling back to OpenCV")
                self.backend = "opencv"
                self._init_detector()
    
    def detect(self, frame: np.ndarray) -> List[FaceInfo]:
        """
        检测人脸
        
        Args:
            frame: 输入图像
            
        Returns:
            人脸信息列表
        """
        if self.backend == "opencv":
            return self._detect_opencv(frame)
        elif self.backend == "mediapipe":
            return self._detect_mediapipe(frame)
        
        return []
    
    def _detect_opencv(self, frame: np.ndarray) -> List[FaceInfo]:
        """OpenCV 检测"""
        faces = []
        
        if self.face_net is not None:
            # DNN 检测
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
            self.face_net.setInput(blob)
            detections = self.face_net.forward()
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > 0.5:
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    
                    face = FaceInfo(
                        face_id=self.next_face_id,
                        bbox=[x1, y1, x2 - x1, y2 - y1],
                        landmarks=[],
                        confidence=float(confidence)
                    )
                    faces.append(face)
                    self.next_face_id += 1
        
        elif self.face_cascade is not None:
            # Haar 级联检测
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in detections:
                face = FaceInfo(
                    face_id=self.next_face_id,
                    bbox=[int(x), int(y), int(w), int(h)],
                    landmarks=[],
                    confidence=0.9
                )
                faces.append(face)
                self.next_face_id += 1
        
        return faces
    
    def _detect_mediapipe(self, frame: np.ndarray) -> List[FaceInfo]:
        """MediaPipe 检测"""
        faces = []
        
        if self.mp_face_detection is None:
            return faces
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face_detection.process(rgb_frame)
        
        if results.detections:
            h, w = frame.shape[:2]
            
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                
                # 获取关键点
                landmarks = []
                for keypoint in detection.location_data.relative_keypoints:
                    landmarks.append((int(keypoint.x * w), int(keypoint.y * h)))
                
                face = FaceInfo(
                    face_id=self.next_face_id,
                    bbox=[x, y, width, height],
                    landmarks=landmarks,
                    confidence=detection.score[0] if detection.score else 0.9
                )
                faces.append(face)
                self.next_face_id += 1
        
        return faces


class FaceAttributeAnalyzer:
    """
    人脸属性分析器
    
    分析年龄、性别、表情等
    """
    
    # 表情类别
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    def __init__(self):
        """初始化属性分析器"""
        self.age_net = None
        self.gender_net = None
        self.emotion_net = None
        
        self._load_models()
        
        logger.info("FaceAttributeAnalyzer initialized")
    
    def _load_models(self):
        """加载模型"""
        try:
            # 尝试加载年龄和性别模型
            model_dir = "models/face_attributes"
            
            import os
            if os.path.exists(f"{model_dir}/age_net.caffemodel"):
                self.age_net = cv2.dnn.readNetFromCaffe(
                    f"{model_dir}/age_deploy.prototxt",
                    f"{model_dir}/age_net.caffemodel"
                )
                logger.info("Loaded age model")
            
            if os.path.exists(f"{model_dir}/gender_net.caffemodel"):
                self.gender_net = cv2.dnn.readNetFromCaffe(
                    f"{model_dir}/gender_deploy.prototxt",
                    f"{model_dir}/gender_net.caffemodel"
                )
                logger.info("Loaded gender model")
                
        except Exception as e:
            logger.warning(f"Could not load attribute models: {e}")
    
    def analyze(self, frame: np.ndarray, face: FaceInfo) -> FaceInfo:
        """
        分析人脸属性
        
        Args:
            frame: 输入图像
            face: 人脸信息
            
        Returns:
            更新后的人脸信息
        """
        x, y, w, h = face.bbox
        
        # 确保边界框在图像内
        h_img, w_img = frame.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        
        if w <= 0 or h <= 0:
            return face
        
        # 提取人脸区域
        face_roi = frame[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            return face
        
        # 分析年龄
        face.age = self._estimate_age(face_roi)
        
        # 分析性别
        face.gender = self._estimate_gender(face_roi)
        
        # 分析表情
        face.emotion = self._estimate_emotion(face_roi)
        
        # 评估质量
        face.quality_score = self._assess_quality(face_roi)
        
        return face
    
    def _estimate_age(self, face_roi: np.ndarray) -> Optional[int]:
        """估计年龄"""
        if self.age_net is None:
            # 简单估计（基于人脸大小）
            h, w = face_roi.shape[:2]
            # 假设较大的人脸可能是成年人
            return 25  # 默认值
        
        try:
            blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227))
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            
            # 年龄区间
            age_intervals = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
            age_idx = age_preds[0].argmax()
            
            # 解析年龄
            age_str = age_intervals[age_idx]
            # 简单取中间值
            ages = [int(s) for s in age_str if s.isdigit()]
            if ages:
                return sum(ages) // len(ages)
            
            return 25
            
        except Exception as e:
            logger.debug(f"Age estimation error: {e}")
            return None
    
    def _estimate_gender(self, face_roi: np.ndarray) -> Optional[str]:
        """估计性别"""
        if self.gender_net is None:
            return None
        
        try:
            blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227))
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            
            gender = 'Male' if gender_preds[0][0] > gender_preds[0][1] else 'Female'
            return gender
            
        except Exception as e:
            logger.debug(f"Gender estimation error: {e}")
            return None
    
    def _estimate_emotion(self, face_roi: np.ndarray) -> Optional[str]:
        """估计表情"""
        # 简单实现：基于亮度分布
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # 计算亮度分布
            mean_brightness = np.mean(gray)
            
            # 简单规则
            if mean_brightness > 150:
                return 'happy'
            elif mean_brightness < 80:
                return 'sad'
            else:
                return 'neutral'
                
        except Exception as e:
            return 'neutral'
    
    def _assess_quality(self, face_roi: np.ndarray) -> float:
        """评估人脸质量"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # 计算清晰度（Laplacian 方差）
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 计算亮度
            brightness = np.mean(gray)
            
            # 综合评分
            quality = min(1.0, blur_score / 100.0) * 0.5
            
            # 亮度评分（适中亮度更好）
            brightness_score = 1.0 - abs(brightness - 128) / 128.0
            quality += brightness_score * 0.5
            
            return min(1.0, quality)
            
        except Exception as e:
            return 0.5


class FaceAnalyzer:
    """
    人脸分析器
    
    整合检测和属性分析
    """
    
    def __init__(self, detection_backend: str = "opencv"):
        """
        初始化人脸分析器
        
        Args:
            detection_backend: 检测后端
        """
        self.detector = FaceDetector(backend=detection_backend)
        self.attribute_analyzer = FaceAttributeAnalyzer()
        
        # 人脸跟踪
        self.face_tracks: Dict[int, FaceInfo] = {}
        self.next_track_id = 1
        
        logger.info("FaceAnalyzer initialized")
    
    def analyze(self, frame: np.ndarray) -> List[FaceInfo]:
        """
        分析图像中的人脸
        
        Args:
            frame: 输入图像
            
        Returns:
            人脸信息列表
        """
        # 检测人脸
        faces = self.detector.detect(frame)
        
        # 分析属性
        for face in faces:
            self.attribute_analyzer.analyze(frame, face)
        
        # 跟踪关联
        faces = self._associate_tracks(faces)
        
        return faces
    
    def _associate_tracks(self, faces: List[FaceInfo]) -> List[FaceInfo]:
        """关联跟踪ID"""
        if not self.face_tracks:
            # 首次检测
            for face in faces:
                face.face_id = self.next_track_id
                self.face_tracks[self.next_track_id] = face
                self.next_track_id += 1
            return faces
        
        # 简单的 IoU 匹配
        matched = set()
        new_faces = []
        
        for face in faces:
            best_iou = 0
            best_id = None
            
            for track_id, tracked_face in self.face_tracks.items():
                if track_id in matched:
                    continue
                
                iou = self._compute_iou(face.bbox, tracked_face.bbox)
                
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_id = track_id
            
            if best_id is not None:
                face.face_id = best_id
                self.face_tracks[best_id] = face
                matched.add(best_id)
            else:
                face.face_id = self.next_track_id
                self.face_tracks[self.next_track_id] = face
                self.next_track_id += 1
            
            new_faces.append(face)
        
        # 清理旧的跟踪
        current_ids = {f.face_id for f in faces}
        to_remove = [tid for tid in self.face_tracks if tid not in current_ids]
        for tid in to_remove:
            del self.face_tracks[tid]
        
        return new_faces
    
    def _compute_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """计算 IoU"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - inter
        
        return inter / max(union, 1)


# 全局实例
_face_analyzer = None

def get_face_analyzer() -> FaceAnalyzer:
    """获取人脸分析器单例"""
    global _face_analyzer
    if _face_analyzer is None:
        _face_analyzer = FaceAnalyzer()
    return _face_analyzer


# 测试代码
if __name__ == '__main__':
    print("Testing Face Analyzer...")
    
    analyzer = FaceAnalyzer()
    
    # 测试图像
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    faces = analyzer.analyze(frame)
    
    print(f"Detected {len(faces)} faces")
    for face in faces:
        print(f"  Face {face.face_id}: {face.to_dict()}")
    
    print("\nDone!")