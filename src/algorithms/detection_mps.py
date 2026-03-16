"""
Apple Silicon 优化版目标检测模块
支持 MPS 加速、半精度推理、统一内存优化
"""

import cv2
import numpy as np
import platform
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from collections import deque
from loguru import logger

# YOLOv8
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("YOLOv8 not installed")

# MediaPipe - 支持新旧版本
MP_AVAILABLE = False
MP_HANDS = None
try:
    import mediapipe as mp
    # 尝试旧版 API (0.10 之前)
    if hasattr(mp, 'solutions'):
        MP_HANDS = mp.solutions.hands.Hands
        MP_AVAILABLE = True
        logger.info("MediaPipe (legacy API) available")
    else:
        # 新版 API (0.10+) - 使用 tasks vision
        try:
            from mediapipe import tasks
            from mediapipe.tasks import vision
            MP_HANDS = vision.HandLandmarker
            MP_AVAILABLE = True
            logger.info("MediaPipe (new tasks API) available")
        except ImportError as e:
            logger.warning(f"MediaPipe tasks API not available: {e}")
except ImportError:
    logger.warning("MediaPipe not installed")
except Exception as e:
    logger.warning(f"MediaPipe error: {e}")

# PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed")


# ==================== COCO 关键点定义 ====================

class COCOKeyPoints:
    """COCO 人体关键点索引"""
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    
    NAMES = {
        0: 'nose', 1: 'L_eye', 2: 'R_eye', 3: 'L_ear', 4: 'R_ear',
        5: 'L_shoulder', 6: 'R_shoulder', 7: 'L_elbow', 8: 'R_elbow',
        9: 'L_wrist', 10: 'R_wrist', 11: 'L_hip', 12: 'R_hip',
        13: 'L_knee', 14: 'R_knee', 15: 'L_ankle', 16: 'R_ankle'
    }


class HandKeyPoints:
    """MediaPipe 手部关键点"""
    WRIST = 0
    THUMB_TIP = 4
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_TIP = 16
    PINKY_FINGER_TIP = 20


# ==================== 卡尔曼滤波器 ====================

class KalmanFilter:
    """一维卡尔曼滤波器（用于关键点平滑）"""
    
    def __init__(self, process_variance: float = 1e-5, 
                 measurement_variance: float = 1e-2):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.state = np.zeros((2, 1))
        self.covariance = np.eye(2) * 1.0
        self.transition = np.array([[1, 1], [0, 1]], dtype=np.float32)
        self.observation = np.array([[1, 0]], dtype=np.float32)
        self.process_noise_cov = np.eye(2) * process_variance
        self.measurement_noise_cov = np.array([[measurement_variance]], dtype=np.float32)
    
    def predict(self) -> np.ndarray:
        self.state = self.transition @ self.state
        self.covariance = (
            self.transition @ self.covariance @ self.transition.T + 
            self.process_noise_cov
        )
        return self.state[0, 0]
    
    def update(self, measurement: float) -> float:
        innovation = measurement - self.observation @ self.state
        innovation_covariance = (
            self.observation @ self.covariance @ self.observation.T + 
            self.measurement_noise_cov
        )
        kalman_gain = self.covariance @ self.observation.T @ np.linalg.inv(innovation_covariance)
        self.state = self.state + kalman_gain @ innovation
        identity = np.eye(2)
        self.covariance = (identity - kalman_gain @ self.observation) @ self.covariance
        return self.state[0, 0]


class KeyPointSmoother:
    """关键点平滑器"""
    
    def __init__(self, max_keypoints: int = 21):
        self.max_keypoints = max_keypoints
        self.filters_x = [KalmanFilter() for _ in range(max_keypoints)]
        self.filters_y = [KalmanFilter() for _ in range(max_keypoints)]
    
    def smooth(self, keypoints: List[List[float]]) -> List[List[float]]:
        smoothed = []
        for i, kp in enumerate(keypoints[:self.max_keypoints]):
            smooth_x = self.filters_x[i].update(kp[0])
            smooth_y = self.filters_y[i].update(kp[1])
            smoothed.append([smooth_x, smooth_y])
        return smoothed


# ==================== 人体姿态检测器（MPS 优化） ====================

class PoseDetectorMPS:
    """人体姿态检测器（支持 MPS 加速）"""
    
    def __init__(self, 
                 model_path: str = 'models/yolov8n-pose.pt',
                 conf_threshold: float = 0.5,
                 device: Optional[str] = None,
                 half: bool = True):
        """
        初始化姿态检测器
        
        Args:
            model_path: YOLO-Pose 模型路径
            conf_threshold: 置信度阈值
            device: 推理设备 ('mps', 'cuda', 'cpu', 'auto')
            half: 是否使用半精度（FP16）
        """
        self.conf_threshold = conf_threshold
        self.half = half
        self.model = None
        self.device = device
        self.smoothers: Dict[int, KeyPointSmoother] = {}
        
        # 自动检测设备
        if device is None or device == 'auto':
            self.device = self._detect_optimal_device()

        logger.info(f"Initializing PoseDetector on {self.device}")

        if YOLO_AVAILABLE and Path(model_path).exists():
            self.model = YOLO(model_path)

            # 移动到指定设备
            if self.device != 'cpu':
                try:
                    self.model.to(self.device)
                    logger.info(f"Model moved to {self.device}")
                except Exception as e:
                    logger.warning(f"Failed to move model to {self.device}: {e}")
                    self.device = 'cpu'

            # 半精度推理 - MPS 上 YOLOv8 有兼容性问题，禁用
            if self.half and self.device == 'cuda':
                self.model.half()
                logger.info("Half precision (FP16) enabled")
            elif self.half and self.device == 'mps':
                logger.info("Half precision disabled on MPS (compatibility)")

            logger.info(f"Pose detector loaded: {model_path}")
        else:
            if not YOLO_AVAILABLE:
                logger.warning("YOLOv8-Pose not available")
            else:
                logger.warning(f"Model not found: {model_path}")
    
    def _detect_optimal_device(self) -> str:
        """检测最优设备"""
        if not TORCH_AVAILABLE:
            return 'cpu'
        
        # Apple Silicon MPS
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("Apple Silicon detected, using MPS acceleration")
                return 'mps'
        
        # NVIDIA CUDA
        if torch.cuda.is_available():
            return 'cuda'
        
        return 'cpu'
    
    def detect(self, image: np.ndarray, smooth: bool = True) -> List[Dict[str, Any]]:
        """检测人体姿态"""
        if self.model is None:
            return []
        
        try:
            # YOLOv8 推理（自动处理设备）
            results = self.model(image, conf=self.conf_threshold, verbose=False)
            
            persons = []
            
            for result in results:
                # 提取关键点
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints_data = result.keypoints.xy.cpu().numpy()
                    confidences = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
                else:
                    keypoints_data = None
                
                # 处理边界框
                boxes = result.boxes
                if boxes is None:
                    continue
                
                for idx, box in enumerate(boxes):
                    if int(box.cls[0]) != 0:  # 只处理人体
                        continue
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    w, h = x2 - x1, y2 - y1
                    
                    # 提取关键点
                    keypoints_dict = {}
                    if keypoints_data is not None and idx < len(keypoints_data):
                        kp_array = keypoints_data[idx]
                        for kp_idx in range(len(kp_array)):
                            kp_name = COCOKeyPoints.NAMES.get(kp_idx, f'kp_{kp_idx}')
                            kp = kp_array[kp_idx].tolist()
                            
                            if confidences is not None:
                                if confidences[idx, kp_idx] < 0.5:
                                    continue
                            
                            keypoints_dict[kp_name] = kp
                    
                    # 应用平滑
                    if smooth and keypoints_dict:
                        person_id = hash((x1, y1, w, h)) % 1000
                        if person_id not in self.smoothers:
                            self.smoothers[person_id] = KeyPointSmoother(max_keypoints=17)
                        
                        kp_list = list(keypoints_dict.values())
                        smoothed_list = self.smoothers[person_id].smooth(kp_list)
                        kp_names = list(keypoints_dict.keys())
                        keypoints_dict = {kp_names[i]: smoothed_list[i] for i in range(len(smoothed_list))}
                    
                    persons.append({
                        "bbox": [x1, y1, w, h],
                        "confidence": float(box.conf[0].cpu().numpy()),
                        "keypoints": keypoints_dict,
                        "track_id": person_id if smooth else None
                    })
            
            return persons
        
        except Exception as e:
            logger.error(f"Pose detection error: {e}")
            return []
    
    def close(self):
        """关闭检测器"""
        self.smoothers.clear()


# ==================== 手部检测器 ====================

class HandDetectorMPS:
    """手部检测器（MediaPipe）"""

    def __init__(self,
                 static_image_mode: bool = False,
                 max_num_hands: int = 2,
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self.hands = None
        self.smoother: Optional[KeyPointSmoother] = None
        self.use_new_api = False

        if MP_AVAILABLE:
            # 检查是否为新 API
            if MP_HANDS == vision.HandLandmarker:
                # 新版 API
                try:
                    base_options = vision.BaseOptions(model_asset_path='models/hand_landmarker.task')
                    options = vision.HandLandmarkerOptions(
                        base_options=base_options,
                        running_mode=vision.RunningMode.IMAGE,
                        num_hands=max_num_hands,
                        min_hand_detection_confidence=min_detection_confidence,
                        min_hand_presence_confidence=min_tracking_confidence
                    )
                    self.hands = vision.HandLandmarker.create_from_options(options)
                    self.use_new_api = True
                    logger.info("Hand detector (MediaPipe new API) initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize new MediaPipe API: {e}")
                    return
            else:
                # 旧版 API
                self.hands = MP_HANDS(
                    static_image_mode=static_image_mode,
                    max_num_hands=max_num_hands,
                    model_complexity=model_complexity,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
                self.use_new_api = False
                logger.info("Hand detector (MediaPipe legacy API) initialized")
            
            self.smoother = KeyPointSmoother(max_keypoints=21)
    
    def detect(self, image: np.ndarray, smooth: bool = True) -> List[Dict[str, Any]]:
        """检测手部"""
        if self.hands is None:
            return []

        hands = []
        h, w = image.shape[:2]

        if self.use_new_api:
            # 新版 API
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                results = self.hands.detect(mp_image)
                
                if results.hand_landmarks:
                    for landmarks in results.hand_landmarks:
                        keypoints = []
                        for lm in landmarks:
                            x, y = int(lm.x * w), int(lm.y * h)
                            keypoints.append([float(x), float(y)])

                        if smooth:
                            keypoints = self.smoother.smooth(keypoints)
                            keypoints = [[int(kp[0]), int(kp[1])] for kp in keypoints]

                        keypoints_np = np.array(keypoints)
                        x_min, y_min = keypoints_np.min(axis=0)
                        x_max, y_max = keypoints_np.max(axis=0)

                        handedness = results.handedness[0][0].category_name if results.handedness else "Unknown"
                        hand_type = "Right" if handedness == "Right" else "Left"

                        hands.append({
                            "keypoints": keypoints,
                            "bbox": [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)],
                            "hand_type": hand_type
                        })
            except Exception as e:
                logger.debug(f"Hand detection error (new API): {e}")
        else:
            # 旧版 API
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    keypoints = []
                    for lm in landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        keypoints.append([float(x), float(y)])

                    if smooth:
                        keypoints = self.smoother.smooth(keypoints)
                        keypoints = [[int(kp[0]), int(kp[1])] for kp in keypoints]

                    keypoints_np = np.array(keypoints)
                    x_min, y_min = keypoints_np.min(axis=0)
                    x_max, y_max = keypoints_np.max(axis=0)

                    hand_type = "Right" if keypoints[4][0] < keypoints[17][0] else "Left"

                    hands.append({
                        "keypoints": keypoints,
                        "bbox": [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)],
                        "hand_type": hand_type
                    })

        return hands
    
    def close(self):
        if self.hands:
            self.hands.close()


# ==================== 组合检测器 ====================

class CombinedDetectorMPS:
    """组合检测器（MPS 优化）"""

    def __init__(self,
                 pose_model_path: str = 'models/yolov8n-pose.pt',
                 conf_threshold: float = 0.5,
                 device: Optional[str] = None,
                 half: bool = True):
        self.pose_detector = PoseDetectorMPS(
            model_path=pose_model_path,
            conf_threshold=conf_threshold,
            device=device,
            half=half
        )
        self.hand_detector = HandDetectorMPS()
        self.mp_available = MP_AVAILABLE

    def detect(self, image: np.ndarray, smooth: bool = True) -> Dict[str, Any]:
        """同时检测人体和手部"""
        persons = self.pose_detector.detect(image, smooth=smooth)
        hands = []
        
        # 如果 MediaPipe 可用，使用专门的手部检测
        if self.mp_available:
            hands = self.hand_detector.detect(image, smooth=smooth)
        
        # 如果没有检测到手部，从人体关键点中估计手部位置
        if not hands:
            for person in persons:
                keypoints = person.get('keypoints', {})
                # 从手腕关键点估计手部位置
                for wrist_name, hand_type in [('L_wrist', 'Left'), ('R_wrist', 'Right')]:
                    if wrist_name in keypoints:
                        wrist = keypoints[wrist_name]
                        # 估计手部边界框（以手腕为中心，大小约为人体框的 1/6）
                        bbox = person['bbox']
                        hand_size = max(bbox[2], bbox[3]) // 6
                        hand_size = max(hand_size, 40)  # 最小 40 像素
                        
                        hands.append({
                            "keypoints": [[int(wrist[0]), int(wrist[1])]],
                            "bbox": [int(wrist[0] - hand_size//2), int(wrist[1] - hand_size//2), hand_size, hand_size],
                            "hand_type": hand_type,
                            "confidence": person.get('confidence', 0.5),
                            "estimated": True  # 标记为估计的手部
                        })

        return {
            "persons": persons,
            "hands": hands,
            "frame_shape": image.shape
        }
    
    def close(self):
        self.pose_detector.close()
        self.hand_detector.close()


# ==================== 可视化 ====================

def visualize_detections_mps(
    image: np.ndarray,
    result: Dict[str, Any]
) -> np.ndarray:
    """可视化检测结果"""
    output = image.copy()
    
    # 绘制人体
    for person in result.get('persons', []):
        x, y, w, h = person['bbox']
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        for kp in person.get('keypoints', {}).values():
            cv2.circle(output, (int(kp[0]), int(kp[1])), 5, (0, 0, 255), -1)
    
    # 绘制手部
    for hand in result.get('hands', []):
        x, y, w, h = hand['bbox']
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        for kp in hand.get('keypoints', []):
            cv2.circle(output, (int(kp[0]), int(kp[1])), 3, (255, 0, 255), -1)
    
    return output


# ==================== 主函数 ====================

def main():
    """测试 MPS 优化检测器"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MPS 优化目标检测测试')
    parser.add_argument('--source', type=int, default=0, help='摄像头 ID')
    parser.add_argument('--device', choices=['mps', 'cuda', 'cpu', 'auto'], default='auto')
    parser.add_argument('--no-half', action='store_true', help='禁用半精度')
    
    args = parser.parse_args()
    
    detector = CombinedDetectorMPS(
        device=args.device,
        half=not args.no_half
    )
    
    cap = cv2.VideoCapture(args.source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    if not cap.isOpened():
        logger.error("Cannot open camera")
        return
    
    logger.info("Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result = detector.detect(frame)
            output = visualize_detections_mps(frame, result)
            
            cv2.putText(output, f"Persons: {len(result['persons'])}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(output, f"Hands: {len(result['hands'])}",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            cv2.imshow('MPS Detection', output)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
