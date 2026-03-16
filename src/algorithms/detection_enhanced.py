"""
增强版目标检测模块 - 集成 YOLO-Pose 精确人体关键点
功能：
1. YOLOv8-Pose 人体关键点检测（18 个 COCO 关键点）
2. MediaPipe 手部 21 个关键点
3. 关键点平滑滤波（卡尔曼滤波）
4. 多人物/多手部追踪
"""

import cv2
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from collections import deque
from loguru import logger

from ..utils.constants import (
    KEYPOINT_CONFIDENCE_THRESHOLD,
    KEYPOINT_CONFIDENCE_THRESHOLD_NORMAL,
    YOLO_INPUT_SIZE
)

# YOLOv8
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("YOLOv8 not installed")

# MediaPipe
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    logger.warning("MediaPipe not installed")


# ==================== 关键点定义 ====================

class COCOKeyPoints:
    """COCO 人体关键点索引（YOLO-Pose）"""
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
    
    # 关键点名称映射
    NAMES = {
        0: 'nose', 1: 'L_eye', 2: 'R_eye', 3: 'L_ear', 4: 'R_ear',
        5: 'L_shoulder', 6: 'R_shoulder', 7: 'L_elbow', 8: 'R_elbow',
        9: 'L_wrist', 10: 'R_wrist', 11: 'L_hip', 12: 'R_hip',
        13: 'L_knee', 14: 'R_knee', 15: 'L_ankle', 16: 'R_ankle'
    }
    
    # 用于身高计算的关键点
    HEAD_KEYPOINTS = [0, 1, 2, 3, 4]  # 鼻子、眼睛、耳朵
    FOOT_KEYPOINTS = [15, 16]  # 脚踝


class HandKeyPoints:
    """MediaPipe 手部关键点"""
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_FINGER_MCP = 17
    PINKY_FINGER_PIP = 18
    PINKY_FINGER_DIP = 19
    PINKY_FINGER_TIP = 20


# ==================== 卡尔曼滤波器（关键点平滑） ====================

class KalmanFilter:
    """一维卡尔曼滤波器（用于关键点平滑）"""
    
    def __init__(self, process_variance: float = 1e-5, 
                 measurement_variance: float = 1e-2):
        """
        初始化卡尔曼滤波器
        
        Args:
            process_variance: 过程噪声方差
            measurement_variance: 测量噪声方差
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        
        # 状态：[位置，速度]
        self.state = np.zeros((2, 1))
        
        # 协方差矩阵
        self.covariance = np.eye(2) * 1.0
        
        # 状态转移矩阵
        self.transition = np.array([[1, 1], [0, 1]], dtype=np.float32)
        
        # 观测矩阵
        self.observation = np.array([[1, 0]], dtype=np.float32)
        
        # 过程噪声协方差
        self.process_noise_cov = np.eye(2) * process_variance
        
        # 测量噪声协方差
        self.measurement_noise_cov = np.array([[measurement_variance]], dtype=np.float32)
    
    def predict(self) -> np.ndarray:
        """预测下一步状态"""
        # 预测状态
        self.state = self.transition @ self.state
        
        # 预测协方差
        self.covariance = (
            self.transition @ self.covariance @ self.transition.T + 
            self.process_noise_cov
        )
        
        return self.state[0, 0]
    
    def update(self, measurement: float) -> float:
        """
        更新状态
        
        Args:
            measurement: 测量值
            
        Returns:
            修正后的状态
        """
        # 计算卡尔曼增益
        innovation = measurement - self.observation @ self.state
        innovation_covariance = (
            self.observation @ self.covariance @ self.observation.T + 
            self.measurement_noise_cov
        )
        kalman_gain = self.covariance @ self.observation.T @ np.linalg.inv(innovation_covariance)
        
        # 更新状态
        self.state = self.state + kalman_gain @ innovation
        
        # 更新协方差
        identity = np.eye(2)
        self.covariance = (identity - kalman_gain @ self.observation) @ self.covariance
        
        return self.state[0, 0]


class KeyPointSmoother:
    """关键点平滑器（管理多个卡尔曼滤波器）"""
    
    def __init__(self, max_keypoints: int = 21):
        """
        初始化平滑器
        
        Args:
            max_keypoints: 最大关键点数量
        """
        self.max_keypoints = max_keypoints
        self.filters_x: List[KalmanFilter] = []
        self.filters_y: List[KalmanFilter] = []
        
        # 初始化滤波器
        for _ in range(max_keypoints):
            self.filters_x.append(KalmanFilter())
            self.filters_y.append(KalmanFilter())
    
    def smooth(self, keypoints: List[List[float]]) -> List[List[float]]:
        """
        平滑关键点
        
        Args:
            keypoints: 关键点列表 [[x1, y1], [x2, y2], ...]
            
        Returns:
            平滑后的关键点
        """
        smoothed = []
        
        for i, kp in enumerate(keypoints):
            if i >= self.max_keypoints:
                break
            
            # X 坐标平滑
            self.filters_x[i].predict()
            smooth_x = self.filters_x[i].update(kp[0])
            
            # Y 坐标平滑
            self.filters_y[i].predict()
            smooth_y = self.filters_y[i].update(kp[1])
            
            smoothed.append([smooth_x, smooth_y])
        
        return smoothed


# ==================== 人体检测器（YOLO-Pose） ====================

class PoseDetector:
    """人体姿态检测器（YOLOv8-Pose）"""
    
    def __init__(self, model_path: str = 'models/yolov8n-pose.pt',
                 conf_threshold: float = 0.5,
                 device: Optional[str] = None):
        """
        初始化姿态检测器
        
        Args:
            model_path: YOLO-Pose 模型路径
            conf_threshold: 置信度阈值
            device: 推理设备 ('mps', 'cuda', 'cpu', 'auto')
        """
        self.conf_threshold = conf_threshold
        self.model = None
        self.smoothers: Dict[int, KeyPointSmoother] = {}  # 按追踪 ID 管理平滑器
        
        # Apple Silicon 优化：自动检测最优设备
        if device is None or device == 'auto':
            device = self._detect_optimal_device()
        
        self.device = device
        logger.info(f"Using device: {device} for pose detection")
        
        if YOLO_AVAILABLE and Path(model_path).exists():
            # MPS 加速支持
            self.model = YOLO(model_path)
            if device != 'cpu':
                self.model.to(device)
                logger.info(f"Model moved to {device}")
            logger.info(f"Pose detector loaded: {model_path}")
        else:
            if not YOLO_AVAILABLE:
                logger.warning("YOLOv8-Pose not available")
            else:
                logger.warning(f"Model not found: {model_path}, falling back to detection model")
                # 降级使用普通检测模型
                if Path(model_path.replace('-pose', '')).exists():
                    self.model = YOLO(model_path.replace('-pose', ''))
                    if device != 'cpu':
                        self.model.to(device)
    
    def _detect_optimal_device(self) -> str:
        """检测最优设备"""
        try:
            # Apple Silicon MPS 加速
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return 'mps'
            
            # NVIDIA CUDA
            if torch.cuda.is_available():
                return 'cuda'
            
            # 回退到 CPU
            return 'cpu'
        except Exception:
            return 'cpu'
    
    def detect(self, image: np.ndarray,
               smooth: bool = True) -> List[Dict[str, Any]]:
        """
        检测人体姿态

        Args:
            image: BGR 图像
            smooth: 是否应用关键点平滑

        Returns:
            人体姿态检测结果
        """
        import time
        start_time = time.time()

        if self.model is None:
            logger.warning("Model not initialized")
            return []

        try:
            # 运行 YOLO-Pose 检测
            results = self.model(image, conf=self.conf_threshold, verbose=False,
                                imgsz=640,  # 固定输入尺寸提高性能
                                max_det=10)  # 限制最大检测数量
        except Exception as e:
            logger.error(f"Detection inference failed: {e}")
            return []

        persons = []
        
        for result in results:
            # 检查是否有关键点
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoints_data = result.keypoints.xy.cpu().numpy()  # [N, 17, 2]
                confidences = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
            else:
                keypoints_data = None
            
            # 处理边界框
            boxes = result.boxes
            if boxes is None:
                continue
            
            for idx, box in enumerate(boxes):
                # 只处理人体类别
                if int(box.cls[0]) != 0:
                    continue
                
                # 提取边界框
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                w, h = x2 - x1, y2 - y1
                
                # 提取关键点
                keypoints_dict = {}
                if keypoints_data is not None and idx < len(keypoints_data):
                    kp_array = keypoints_data[idx]  # [17, 2]
                    
                    for kp_idx in range(len(kp_array)):
                        kp_name = COCOKeyPoints.NAMES.get(kp_idx, f'kp_{kp_idx}')
                        kp = kp_array[kp_idx].tolist()
                        
                        # 检查置信度（使用配置阈值）
                        if confidences is not None:
                            if confidences[idx, kp_idx] < KEYPOINT_CONFIDENCE_THRESHOLD:
                                continue
                        
                        keypoints_dict[kp_name] = kp
                
                # 应用平滑
                if smooth and len(keypoints_dict) > 0:
                    person_id = hash((x1, y1, w, h)) % 1000
                    if person_id not in self.smoothers:
                        self.smoothers[person_id] = KeyPointSmoother(max_keypoints=17)
                    
                    # 转换为列表格式进行平滑
                    kp_list = list(keypoints_dict.values())
                    smoothed_list = self.smoothers[person_id].smooth(kp_list)
                    
                    # 转回字典格式
                    kp_names = list(keypoints_dict.keys())
                    keypoints_dict = {
                        kp_names[i]: smoothed_list[i] 
                        for i in range(len(smoothed_list))
                    }
                
                persons.append({
                    "bbox": [x1, y1, w, h],
                    "confidence": float(box.conf[0].cpu().numpy()),
                    "keypoints": keypoints_dict,
                    "track_id": person_id if smooth else None
                })

        # 记录检测时间
        detection_time = (time.time() - start_time) * 1000
        if detection_time > 50:  # 超过50ms记录警告
            logger.debug(f"Detection took {detection_time:.1f}ms for {len(persons)} persons")

        return persons

    def reset_smoother(self, track_id: int):
        """重置指定 ID 的平滑器"""
        if track_id in self.smoothers:
            del self.smoothers[track_id]
    
    def close(self):
        """关闭检测器"""
        self.smoothers.clear()


# ==================== 手部检测器（MediaPipe） ====================

class HandDetectorEnhanced:
    """增强版手部检测器"""
    
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
        self.hands = None
        self.smoother: Optional[KeyPointSmoother] = None
        
        if MP_AVAILABLE:
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_num_hands,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.smoother = KeyPointSmoother(max_keypoints=21)
            logger.info("Hand detector (MediaPipe) initialized")
    
    def detect(self, image: np.ndarray, 
               smooth: bool = True) -> List[Dict[str, Any]]:
        """
        检测手部
        
        Args:
            image: BGR 图像
            smooth: 是否应用关键点平滑
            
        Returns:
            手部检测结果
        """
        if self.hands is None:
            return []
        
        # 转换为 RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 检测
        results = self.hands.process(image_rgb)
        
        hands = []
        
        if results.multi_hand_landmarks:
            h, w = image.shape[:2]
            
            for idx, landmarks in enumerate(results.multi_hand_landmarks):
                # 提取 21 个关键点
                keypoints = []
                for lm in landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    keypoints.append([float(x), float(y)])
                
                # 应用平滑
                if smooth:
                    keypoints = self.smoother.smooth(keypoints)
                    # 转回整数
                    keypoints = [[int(kp[0]), int(kp[1])] for kp in keypoints]
                
                # 计算边界框
                keypoints_np = np.array(keypoints)
                x_min, y_min = keypoints_np.min(axis=0)
                x_max, y_max = keypoints_np.max(axis=0)
                
                # 判断左右手（通过拇指位置）
                hand_type = self._classify_hand(keypoints)
                
                hands.append({
                    "keypoints": keypoints,
                    "bbox": [
                        int(x_min), int(y_min),
                        int(x_max - x_min), int(y_max - y_min)
                    ],
                    "hand_type": hand_type,
                    "landmarks": landmarks  # 原始 landmarks
                })
        
        return hands
    
    def _classify_hand(self, keypoints: List[List[int]]) -> str:
        """
        判断左右手
        
        通过拇指（4 号）和小指（17 号）的 X 坐标关系判断
        """
        if len(keypoints) < 21:
            return "Unknown"
        
        thumb_x = keypoints[HandKeyPoints.THUMB_TIP][0]
        pinky_mcp_x = keypoints[HandKeyPoints.PINKY_FINGER_MCP][0]
        
        # 对于右手，拇指在左侧；对于左手，拇指在右侧
        # 但这取决于手掌朝向，这里简化处理
        if thumb_x < pinky_mcp_x:
            return "Right"
        else:
            return "Left"
    
    def close(self):
        """关闭检测器"""
        if self.hands:
            self.hands.close()


# ==================== 组合检测器 ====================

class CombinedDetectorEnhanced:
    """增强版组合检测器"""
    
    def __init__(self,
                 pose_model_path: str = 'models/yolov8n-pose.pt',
                 conf_threshold: float = 0.5):
        """
        初始化组合检测器
        
        Args:
            pose_model_path: YOLO-Pose 模型路径
            conf_threshold: 置信度阈值
        """
        self.pose_detector = PoseDetector(
            model_path=pose_model_path,
            conf_threshold=conf_threshold
        )
        self.hand_detector = HandDetectorEnhanced()
    
    def detect(self, image: np.ndarray,
               smooth: bool = True) -> Dict[str, Any]:
        """
        同时检测人体姿态和手部
        
        Args:
            image: BGR 图像
            smooth: 是否应用关键点平滑
            
        Returns:
            检测结果字典
        """
        return {
            "persons": self.pose_detector.detect(image, smooth=smooth),
            "hands": self.hand_detector.detect(image, smooth=smooth),
            "frame_shape": image.shape
        }
    
    def close(self):
        """关闭检测器"""
        self.pose_detector.close()
        self.hand_detector.close()


# ==================== 可视化函数 ====================

def visualize_pose_detections(
    image: np.ndarray,
    persons: List[Dict[str, Any]],
    show_skeleton: bool = True,
    show_bbox: bool = True
) -> np.ndarray:
    """
    可视化人体姿态检测
    
    Args:
        image: BGR 图像
        persons: 人体检测结果
        show_skeleton: 是否显示骨架
        show_bbox: 是否显示边界框
        
    Returns:
        标注后的图像
    """
    output = image.copy()
    
    # COCO 骨架连接
    skeleton_connections = [
        (15, 13), (13, 11), (16, 14), (14, 12),  # 腿
        (11, 12),  # 髋部
        (5, 6),  # 肩
        (5, 7), (7, 9),  # 左臂
        (6, 8), (8, 10),  # 右臂
        (11, 5), (12, 6),  # 躯干
        (0, 1), (0, 2), (1, 3), (2, 4)  # 头
    ]
    
    keypoint_colors = [
        (255, 0, 0), (255, 50, 0), (255, 100, 0), (255, 150, 0), (255, 200, 0),  # 红 - 头
        (0, 255, 0), (50, 255, 0), (100, 255, 0), (150, 255, 0), (200, 255, 0),  # 绿 - 左
        (0, 0, 255), (0, 50, 255), (0, 100, 255), (0, 150, 255), (0, 200, 255),  # 蓝 - 右
    ]
    
    for person in persons:
        keypoints = person.get('keypoints', {})
        bbox = person.get('bbox')
        confidence = person.get('confidence', 0)
        
        # 绘制边界框
        if show_bbox and bbox:
            x, y, w, h = bbox
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output, f"Person: {confidence:.2f}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 绘制骨架
        if show_skeleton and keypoints:
            # 绘制关键点
            kp_list = list(keypoints.values())
            for kp in kp_list:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(output, (x, y), 5, (0, 255, 255), -1)
            
            # 绘制连接线
            for conn in skeleton_connections:
                if conn[0] < len(kp_list) and conn[1] < len(kp_list):
                    pt1 = tuple(map(int, kp_list[conn[0]]))
                    pt2 = tuple(map(int, kp_list[conn[1]]))
                    cv2.line(output, pt1, pt2, (255, 0, 255), 2)
    
    return output


def visualize_hand_detections(
    image: np.ndarray,
    hands: List[Dict[str, Any]]
) -> np.ndarray:
    """
    可视化手部检测
    
    Args:
        image: BGR 图像
        hands: 手部检测结果
        
    Returns:
        标注后的图像
    """
    output = image.copy()
    
    for hand in hands:
        keypoints = hand.get('keypoints', [])
        bbox = hand.get('bbox')
        hand_type = hand.get('hand_type', 'Unknown')
        
        # 绘制边界框
        if bbox:
            x, y, w, h = bbox
            color = (255, 0, 0) if hand_type == 'Right' else (0, 0, 255)
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            cv2.putText(output, f"{hand_type} Hand", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 绘制骨架
        if keypoints and len(keypoints) >= 21:
            # 手指连接
            finger_tips = [4, 8, 12, 16, 20]
            finger_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), 
                           (255, 255, 0), (255, 0, 255)]
            
            # 绘制手指
            for finger_idx, tip_idx in enumerate(finger_tips):
                if finger_idx == 0:  # 拇指
                    conn = [(0, 1), (1, 2), (2, 3), (3, 4)]
                else:  # 其他手指
                    base_idx = finger_idx * 4 + 1
                    conn = [(0, base_idx), (base_idx, base_idx + 1),
                           (base_idx + 1, base_idx + 2), (base_idx + 2, base_idx + 3)]
                
                for c in conn:
                    if c[0] < len(keypoints) and c[1] < len(keypoints):
                        pt1 = tuple(keypoints[c[0]])
                        pt2 = tuple(keypoints[c[1]])
                        cv2.line(output, pt1, pt2, finger_colors[finger_idx], 2)
            
            # 绘制关键点
            for kp in keypoints:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(output, (x, y), 3, (255, 255, 255), -1)
    
    return output


# ==================== 主函数（测试） ====================

def main():
    """测试增强版检测器"""
    import argparse
    
    parser = argparse.ArgumentParser(description='增强版目标检测测试')
    parser.add_argument('--source', '-s', type=int, default=0,
                       help='摄像头设备 ID')
    parser.add_argument('--model', '-m', type=str,
                       default='models/yolov8n-pose.pt',
                       help='YOLO-Pose 模型路径')
    parser.add_argument('--no-smooth', action='store_true',
                       help='禁用关键点平滑')
    
    args = parser.parse_args()
    
    # 初始化检测器
    detector = CombinedDetectorEnhanced(
        pose_model_path=args.model,
        conf_threshold=0.5
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
            result = detector.detect(frame, smooth=not args.no_smooth)
            
            # 可视化
            output = frame.copy()
            output = visualize_pose_detections(output, result['persons'])
            output = visualize_hand_detections(output, result['hands'])
            
            # 显示信息
            cv2.putText(output, f"Persons: {len(result['persons'])}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(output, f"Hands: {len(result['hands'])}",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            cv2.imshow('Enhanced Detection', output)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
