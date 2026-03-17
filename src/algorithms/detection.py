"""
目标检测模块 - YOLOv8 + MediaPipe
功能：
1. 人体检测（YOLOv8）：头部、脚踝关键点
2. 手部检测（MediaPipe）：21 个手部关键点
3. 输出统一的检测结果格式
"""

import cv2
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path
from loguru import logger

# YOLOv8
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("YOLOv8 not installed, person detection disabled")

# MediaPipe
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    logger.warning("MediaPipe not installed, hand detection disabled")


# 关键点索引定义
class PersonKeypoints:
    """人体关键点索引"""
    HEAD_TOP = 0      # 头部顶点
    ANKLE_LEFT = 1    # 左脚踝
    ANKLE_RIGHT = 2   # 右脚踝
    SHOULDER_LEFT = 3 # 左肩
    SHOULDER_RIGHT = 4  # 右肩


class HandKeypoints:
    """手部关键点索引 (MediaPipe)"""
    WRIST = 0         # 掌根
    THUMB_TIP = 4     # 拇指指尖
    INDEX_TIP = 8     # 食指指尖
    MIDDLE_TIP = 12   # 中指指尖
    RING_TIP = 16     # 无名指指尖
    PINKY_TIP = 20    # 小指指尖


class DetectionResult:
    """检测结果数据类"""
    
    def __init__(self):
        self.persons: List[Dict[str, Any]] = []
        self.hands: List[Dict[str, Any]] = []
        self.frame_shape: tuple = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于 JSON 序列化）"""
        return {
            "persons": self.persons,
            "hands": self.hands,
            "frame_shape": self.frame_shape
        }
    
    def __repr__(self) -> str:
        return f"DetectionResult(persons={len(self.persons)}, hands={len(self.hands)})"


class PersonDetector:
    """人体检测器 (YOLOv8)"""
    
    def __init__(self, model_path: str = 'models/yolov8n.pt', 
                 conf_threshold: float = 0.5):
        """
        初始化人体检测器
        
        Args:
            model_path: YOLO 模型路径
            conf_threshold: 置信度阈值
        """
        self.conf_threshold = conf_threshold
        self.model = None
        
        if YOLO_AVAILABLE and Path(model_path).exists():
            self.model = YOLO(model_path)
            logger.info(f"Person detector loaded: {model_path}")
        else:
            if not YOLO_AVAILABLE:
                logger.warning("YOLOv8 not available")
            else:
                logger.warning(f"Model not found: {model_path}")
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        检测人体
        
        Args:
            image: BGR 图像
            
        Returns:
            人体检测结果列表
        """
        if self.model is None:
            return []
        
        # 运行 YOLO 检测
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        
        persons = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                # 只处理人体类别 (COCO: 0=person)
                if int(box.cls[0]) != 0:
                    continue
                
                # 提取边界框
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                w, h = x2 - x1, y2 - y1
                
                # 估算关键点（简化：从边界框推算）
                # 实际项目中可换用 YOLO-Pose 获取更精确的关键点
                keypoints = self._estimate_keypoints(x1, y1, w, h)
                
                persons.append({
                    "bbox": [x1, y1, w, h],
                    "confidence": float(box.conf[0].cpu().numpy()),
                    "keypoints": keypoints
                })
        
        return persons
    
    def _estimate_keypoints(
        self, x1: int, y1: int, w: int, h: int
    ) -> Dict[str, List[int]]:
        """
        从边界框估算关键点
        
        Returns:
            关键点字典 {"head": [x, y], "ankle_left": [x, y], "ankle_right": [x, y]}
        """
        # 简化估算（实际应使用 Pose 模型）
        head_x = x1 + w // 2
        head_y = y1 + int(h * 0.15)
        
        ankle_y = y1 + h
        ankle_left_x = x1 + int(w * 0.35)
        ankle_right_x = x1 + int(w * 0.65)
        
        return {
            "head": [head_x, head_y],
            "ankle_left": [ankle_left_x, ankle_y],
            "ankle_right": [ankle_right_x, ankle_y],
            "shoulder_left": [x1 + int(w * 0.3), y1 + int(h * 0.2)],
            "shoulder_right": [x1 + int(w * 0.7), y1 + int(h * 0.2)]
        }


class HandDetector:
    """手部检测器 (MediaPipe tasks API)"""

    def __init__(self, static_image_mode: bool = False,
                 max_num_hands: int = 2,
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        初始化手部检测器

        Args:
            static_image_mode: 是否静态图像模式
            max_num_hands: 最大检测手数
            model_complexity: 模型复杂度 (0/1)
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
        """
        self.hand_landmarker = None

        if MP_AVAILABLE:
            try:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                
                model_path = Path('models/hand_landmarker.task')
                if model_path.exists():
                    base_options = python.BaseOptions(model_asset_path=str(model_path))
                    options = vision.HandLandmarkerOptions(
                        base_options=base_options,
                        num_hands=max_num_hands,
                        min_hand_detection_confidence=min_detection_confidence,
                        min_hand_presence_confidence=min_tracking_confidence,
                        min_tracking_confidence=min_tracking_confidence
                    )
                    self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
                    logger.info("Hand detector (MediaPipe tasks) initialized")
                else:
                    logger.warning(f"Hand landmarker model not found: {model_path}")
            except Exception as e:
                logger.warning(f"Failed to initialize hand detector: {e}")

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        检测手部

        Args:
            image: BGR 图像

        Returns:
            手部检测结果列表
        """
        if self.hand_landmarker is None:
            return []

        try:
            from mediapipe import Image as MPImage
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = MPImage(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            results = self.hand_landmarker.detect(mp_image)

            hands = []
            if results.hand_landmarks:
                h, w = image.shape[:2]

                for idx, landmarks in enumerate(results.hand_landmarks):
                    keypoints = []
                    for lm in landmarks:
                        x, y = int(lm.x * w), int(lm.y * h)
                        keypoints.append([x, y])

                    keypoints_np = np.array(keypoints)
                    x_min, y_min = keypoints_np.min(axis=0)
                    x_max, y_max = keypoints_np.max(axis=0)

                    handedness = "Unknown"
                    if results.handedness and idx < len(results.handedness):
                        handedness = results.handedness[idx][0].category_name

                    hands.append({
                        "keypoints": keypoints,
                        "bbox": [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)],
                        "handedness": handedness,
                        "confidence": 0.9
                    })

            return hands
        except Exception as e:
            logger.error(f"Hand detection error: {e}")
            return []

    def close(self):
        """关闭手部检测器"""
        try:
            if self.hand_landmarker is not None:
                # MediaPipe tasks API 没有明确的 close 方法
                # 但我们可以设置为 None 来释放引用
                self.hand_landmarker = None
                logger.info("Hand detector closed")
        except Exception as e:
            logger.warning(f"Error closing hand detector: {e}")



class CombinedDetector:
    """组合检测器 (人体 + 手部)"""
    
    def __init__(self, 
                 yolo_model_path: str = 'models/yolov8n.pt',
                 conf_threshold: float = 0.5):
        """
        初始化组合检测器
        
        Args:
            yolo_model_path: YOLO 模型路径
            conf_threshold: 置信度阈值
        """
        self.person_detector = PersonDetector(
            model_path=yolo_model_path,
            conf_threshold=conf_threshold
        )
        self.hand_detector = HandDetector()
    
    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        同时检测人体和手部
        
        Args:
            image: BGR 图像
            
        Returns:
            DetectionResult: 检测结果
        """
        result = DetectionResult()
        result.frame_shape = image.shape
        
        # 检测人体
        result.persons = self.person_detector.detect(image)
        
        # 检测手部
        result.hands = self.hand_detector.detect(image)
        
        return result
    
    def close(self):
        """关闭检测器"""
        try:
            if hasattr(self.hand_detector, 'close'):
                self.hand_detector.close()
        except Exception as e:
            logger.warning(f"Error closing hand detector: {e}")


def visualize_detections(
    image: np.ndarray, 
    result: DetectionResult,
    show_bbox: bool = True,
    show_keypoints: bool = True,
    show_labels: bool = True
) -> np.ndarray:
    """
    可视化检测结果
    
    Args:
        image: BGR 图像
        result: 检测结果
        show_bbox: 是否显示边界框
        show_keypoints: 是否显示关键点
        show_labels: 是否显示标签
        
    Returns:
        标注后的图像
    """
    output = image.copy()
    
    # 绘制人体检测
    for person in result.persons:
        x, y, w, h = person['bbox']
        conf = person['confidence']
        
        # 边界框
        if show_bbox:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 标签
        if show_labels:
            label = f"Person: {conf:.2f}"
            cv2.putText(output, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 关键点
        if show_keypoints:
            keypoints = person.get('keypoints', {})
            for name, kp in keypoints.items():
                x_kp, y_kp = int(kp[0]), int(kp[1])
                cv2.circle(output, (x_kp, y_kp), 5, (0, 0, 255), -1)
    
    # 绘制手部检测
    for hand in result.hands:
        keypoints = hand['keypoints']
        
        # 边界框
        if show_bbox:
            x, y, w, h = hand['bbox']
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # 关键点连线
        if show_keypoints:
            # 绘制手掌骨架
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
                (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
                (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
                (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
                (0, 17), (17, 18), (18, 19), (19, 20),  # 小指
            ]
            for conn in connections:
                pt1 = tuple(keypoints[conn[0]])
                pt2 = tuple(keypoints[conn[1]])
                cv2.line(output, pt1, pt2, (255, 0, 255), 2)
            
            # 绘制关键点
            for kp in keypoints:
                x_kp, y_kp = int(kp[0]), int(kp[1])
                cv2.circle(output, (x_kp, y_kp), 4, (255, 0, 255), -1)
    
    return output


def main():
    """测试检测器"""
    import argparse
    
    parser = argparse.ArgumentParser(description='目标检测测试')
    parser.add_argument('--source', '-s', type=int, default=0,
                       help='摄像头设备 ID')
    parser.add_argument('--model', '-m', type=str, 
                       default='models/yolov8n.pt',
                       help='YOLO 模型路径')
    parser.add_argument('--conf', '-c', type=float, default=0.5,
                       help='置信度阈值')
    
    args = parser.parse_args()
    
    # 初始化检测器
    detector = CombinedDetector(
        yolo_model_path=args.model,
        conf_threshold=args.conf
    )
    
    # 打开摄像头
    cap = cv2.VideoCapture(args.source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    if not cap.isOpened():
        logger.error("Cannot open camera")
        return
    
    logger.info("Camera opened. Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测
            result = detector.detect(frame)
            
            # 可视化
            output = visualize_detections(frame, result)
            
            # 显示信息
            cv2.putText(output, f"Persons: {len(result.persons)}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(output, f"Hands: {len(result.hands)}",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            cv2.imshow('Detection', output)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
