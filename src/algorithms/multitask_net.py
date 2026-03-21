"""
多任务网络架构

参考 Tesla FSD 的设计理念：
1. 共享特征提取骨干网络 (Shared Backbone)
2. 特征金字塔网络 (FPN) 多尺度特征
3. 多任务头：检测、姿态估计、深度估计
4. 高效推理优化

优势：
- 减少计算量（共享特征提取）
- 提高精度（多任务协同学习）
- 降低延迟（单次前向传播）
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from loguru import logger
import time
import threading
from collections import deque


@dataclass
class MultiTaskOutput:
    """多任务网络输出"""
    # 检测结果
    boxes: np.ndarray          # (N, 4) [x1, y1, x2, y2]
    scores: np.ndarray         # (N,) 置信度
    classes: np.ndarray        # (N,) 类别
    
    # 姿态估计
    keypoints: np.ndarray      # (N, 17, 3) [x, y, conf]
    
    # 深度估计
    depth_map: np.ndarray      # (H, W) 深度图
    
    # 处理时间
    inference_time_ms: float
    
    def to_dict(self) -> Dict:
        return {
            'boxes': self.boxes.tolist() if len(self.boxes) > 0 else [],
            'scores': self.scores.tolist() if len(self.scores) > 0 else [],
            'classes': self.classes.tolist() if len(self.classes) > 0 else [],
            'keypoints': self.keypoints.tolist() if len(self.keypoints) > 0 else [],
            'depth_map': None,  # 太大不序列化
            'inference_time_ms': self.inference_time_ms
        }


class ConvBlock(nn.Module):
    """卷积块：Conv + BN + SiLU"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """瓶颈块：1x1 -> 3x3 -> 1x1"""
    
    def __init__(self, channels: int, expansion: float = 0.5):
        super().__init__()
        hidden = int(channels * expansion)
        self.cv1 = ConvBlock(channels, hidden, 1)
        self.cv2 = ConvBlock(hidden, channels, 3)
    
    def forward(self, x):
        return x + self.cv2(self.cv1(x))


class SPPF(nn.Module):
    """空间金字塔池化 - 快速版"""
    
    def __init__(self, channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        hidden = channels // 2
        self.cv1 = ConvBlock(channels, hidden, 1)
        self.cv2 = ConvBlock(hidden * 4, out_channels, 1)
        self.m = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)
    
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], dim=1))


class FeaturePyramidNetwork(nn.Module):
    """特征金字塔网络 (FPN)"""
    
    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            self.lateral_convs.append(ConvBlock(in_channels, out_channels, 1))
            self.fpn_convs.append(ConvBlock(out_channels, out_channels, 3))
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: [P3, P4, P5] 多尺度特征
        Returns:
            fpn_features: [P3', P4', P5'] 增强后的特征
        """
        # 自顶向下路径
        laterals = [conv(f) for f, conv in zip(features, self.lateral_convs)]
        
        # 融合
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode='nearest'
            )
        
        # 输出
        return [conv(lat) for lat, conv in zip(laterals, self.fpn_convs)]


class DetectionHead(nn.Module):
    """检测头"""
    
    def __init__(self, in_channels: int = 256, num_classes: int = 1, num_anchors: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # 每个锚点输出：4 (bbox) + 1 (obj) + num_classes
        self.output_channels = num_anchors * (5 + num_classes)
        
        self.conv = nn.Sequential(
            ConvBlock(in_channels, in_channels, 3),
            ConvBlock(in_channels, in_channels, 3),
            nn.Conv2d(in_channels, self.output_channels, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class KeypointHead(nn.Module):
    """关键点检测头"""
    
    def __init__(self, in_channels: int = 256, num_keypoints: int = 17):
        super().__init__()
        self.num_keypoints = num_keypoints
        
        # 每个关键点：x, y, confidence
        self.conv = nn.Sequential(
            ConvBlock(in_channels, in_channels, 3),
            ConvBlock(in_channels, in_channels, 3),
            nn.Conv2d(in_channels, num_keypoints * 3, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DepthHead(nn.Module):
    """深度估计头"""
    
    def __init__(self, in_channels: int = 256):
        super().__init__()
        
        # 解码器
        self.decoder = nn.Sequential(
            ConvBlock(in_channels, in_channels // 2, 3),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(in_channels // 2, in_channels // 4, 3),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(in_channels // 4, in_channels // 8, 3),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(in_channels // 8, 32, 3),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()  # 归一化到 0-1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class MultiTaskNetwork(nn.Module):
    """
    多任务网络
    
    架构：
    1. Backbone: 轻量级特征提取
    2. FPN: 多尺度特征融合
    3. Heads: 检测、关键点、深度
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        num_keypoints: int = 17,
        fpn_channels: int = 256
    ):
        super().__init__()
        
        # 简化的骨干网络 (类似 YOLOv8)
        self.backbone = nn.ModuleDict({
            'stem': ConvBlock(3, 32, 3, 2),
            'stage1': nn.Sequential(ConvBlock(32, 64, 3, 2), Bottleneck(64)),
            'stage2': nn.Sequential(ConvBlock(64, 128, 3, 2), Bottleneck(128), Bottleneck(128)),
            'stage3': nn.Sequential(ConvBlock(128, 256, 3, 2), Bottleneck(256), Bottleneck(256), Bottleneck(256)),
            'stage4': nn.Sequential(ConvBlock(256, 512, 3, 2), Bottleneck(512), Bottleneck(512)),
            'sppf': SPPF(512, 512)
        })
        
        # FPN
        self.fpn = FeaturePyramidNetwork([128, 256, 512], fpn_channels)
        
        # 多任务头
        self.detection_head = DetectionHead(fpn_channels, num_classes)
        self.keypoint_head = KeypointHead(fpn_channels, num_keypoints)
        self.depth_head = DepthHead(fpn_channels)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入图像 (B, 3, H, W)
            
        Returns:
            dict: {
                'detection': 检测输出,
                'keypoints': 关键点输出,
                'depth': 深度图
            }
        """
        # Backbone
        features = []
        x = self.backbone['stem'](x)
        x = self.backbone['stage1'](x)
        x = self.backbone['stage2'](x)
        features.append(x)  # P3
        x = self.backbone['stage3'](x)
        features.append(x)  # P4
        x = self.backbone['stage4'](x)
        x = self.backbone['sppf'](x)
        features.append(x)  # P5
        
        # FPN
        fpn_features = self.fpn(features)
        
        # 多任务头
        outputs = {
            'detection': [self.detection_head(f) for f in fpn_features],
            'keypoints': [self.keypoint_head(f) for f in fpn_features],
            'depth': self.depth_head(fpn_features[0])  # 使用最高分辨率特征
        }
        
        return outputs


class MultiTaskInference:
    """多任务推理引擎"""
    
    def __init__(
        self,
        device: str = None,
        input_size: Tuple[int, int] = (640, 480),
        max_depth: float = 10.0
    ):
        """
        初始化推理引擎
        
        Args:
            device: 设备 ('mps', 'cuda', 'cpu')
            input_size: 输入尺寸 (W, H)
            max_depth: 最大深度值（米）
        """
        # 设备选择
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        self.input_size = input_size
        self.max_depth = max_depth
        
        # 创建模型
        self.model = MultiTaskNetwork().to(self.device)
        self.model.eval()
        
        # 性能统计
        self.stats = {
            'total_inferences': 0,
            'total_time_ms': 0,
            'avg_time_ms': 0
        }
        
        logger.info(f"MultiTaskNetwork initialized on {self.device}")
        
        # 参数量统计
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Total parameters: {num_params/1e6:.2f}M")
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        预处理图像
        
        Args:
            image: BGR 图像 (H, W, 3)
            
        Returns:
            tensor: (1, 3, H, W)
        """
        # 调整大小
        h, w = self.input_size[1], self.input_size[0]
        img = cv2.resize(image, (w, h))
        
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 归一化
        img = img.astype(np.float32) / 255.0
        
        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        
        # 转换为 tensor
        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
        
        return tensor
    
    def infer(self, image: np.ndarray) -> MultiTaskOutput:
        """
        执行推理
        
        Args:
            image: BGR 图像
            
        Returns:
            MultiTaskOutput: 多任务输出
        """
        start_time = time.time()
        
        # 预处理
        tensor = self.preprocess(image)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(tensor)
        
        # 后处理
        boxes, scores, classes = self._decode_detections(outputs['detection'], image.shape[:2])
        keypoints = self._decode_keypoints(outputs['keypoints'], image.shape[:2])
        depth_map = self._decode_depth(outputs['depth'], image.shape[:2])
        
        # 统计
        elapsed_ms = (time.time() - start_time) * 1000
        self.stats['total_inferences'] += 1
        self.stats['total_time_ms'] += elapsed_ms
        self.stats['avg_time_ms'] = self.stats['total_time_ms'] / self.stats['total_inferences']
        
        return MultiTaskOutput(
            boxes=boxes,
            scores=scores,
            classes=classes,
            keypoints=keypoints,
            depth_map=depth_map,
            inference_time_ms=elapsed_ms
        )
    
    def _decode_detections(
        self,
        detections: List[torch.Tensor],
        original_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """解码检测结果"""
        # 简化实现：返回空结果
        # 实际应用中需要实现完整的 NMS 后处理
        return np.array([]), np.array([]), np.array([])
    
    def _decode_keypoints(
        self,
        keypoints: List[torch.Tensor],
        original_shape: Tuple[int, int]
    ) -> np.ndarray:
        """解码关键点"""
        # 简化实现
        return np.array([])
    
    def _decode_depth(
        self,
        depth: torch.Tensor,
        original_shape: Tuple[int, int]
    ) -> np.ndarray:
        """解码深度图"""
        depth = depth.squeeze().cpu().numpy()
        depth = cv2.resize(depth, (original_shape[1], original_shape[0]))
        # 转换为米
        depth_meters = depth * self.max_depth
        return depth_meters.astype(np.float32)
    
    def get_stats(self) -> Dict[str, float]:
        """获取性能统计"""
        return self.stats.copy()


class HybridInference:
    """
    混合推理引擎
    
    结合预训练模型和多任务网络：
    - 使用 YOLOv8 进行检测和姿态估计（高精度）
    - 使用 Depth Anything V2 进行深度估计（高精度）
    - 多任务网络作为轻量级备选方案
    """
    
    def __init__(
        self,
        yolo_model: str = 'models/yolov8n-pose.pt',
        use_depth_anything: bool = True,
        device: str = None
    ):
        """
        初始化混合推理引擎
        
        Args:
            yolo_model: YOLO 模型路径
            use_depth_anything: 是否使用 Depth Anything V2
            device: 设备
        """
        # 设备选择
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # YOLO 检测器
        self.yolo = None
        try:
            from ultralytics import YOLO
            self.yolo = YOLO(yolo_model)
            logger.info(f"YOLO loaded: {yolo_model}")
        except Exception as e:
            logger.warning(f"YOLO not available: {e}")
        
        # 深度估计器
        self.depth_estimator = None
        if use_depth_anything:
            try:
                from .depth_estimator import get_depth_estimator
                self.depth_estimator = get_depth_estimator()
                logger.info(f"Depth estimator loaded: backend={self.depth_estimator.backend}")
            except Exception as e:
                logger.warning(f"Depth estimator not available: {e}")
        
        # 性能统计
        self.stats = {
            'total_inferences': 0,
            'yolo_time_ms': 0,
            'depth_time_ms': 0,
            'total_time_ms': 0
        }
    
    def infer(self, image: np.ndarray) -> Dict[str, Any]:
        """
        执行混合推理
        
        Args:
            image: BGR 图像
            
        Returns:
            dict: 检测和深度结果
        """
        start_time = time.time()
        result = {
            'persons': [],
            'hands': [],
            'depth_map': None
        }
        
        # YOLO 检测
        yolo_start = time.time()
        if self.yolo is not None:
            try:
                yolo_result = self.yolo(image, verbose=False)
                if yolo_result and len(yolo_result) > 0:
                    r = yolo_result[0]
                    
                    # 解析人体检测
                    if r.boxes is not None:
                        for i, box in enumerate(r.boxes):
                            if int(box.cls[0]) == 0:  # person class
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                conf = float(box.conf[0])
                                
                                person = {
                                    'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                                    'confidence': conf,
                                    'keypoints': {}
                                }
                                
                                # 关键点
                                if r.keypoints is not None and i < len(r.keypoints.data):
                                    kpts = r.keypoints.data[i].cpu().numpy()
                                    person['keypoints'] = self._parse_keypoints(kpts)
                                
                                result['persons'].append(person)
            except Exception as e:
                logger.error(f"YOLO inference error: {e}")
        
        self.stats['yolo_time_ms'] = (time.time() - yolo_start) * 1000
        
        # 深度估计
        depth_start = time.time()
        if self.depth_estimator is not None:
            try:
                depth_map = self.depth_estimator.estimate(image)
                result['depth_map'] = depth_map
            except Exception as e:
                logger.error(f"Depth estimation error: {e}")
        
        self.stats['depth_time_ms'] = (time.time() - depth_start) * 1000
        
        # 统计
        self.stats['total_inferences'] += 1
        self.stats['total_time_ms'] = (time.time() - start_time) * 1000
        
        return result
    
    def _parse_keypoints(self, kpts: np.ndarray) -> Dict[str, List[float]]:
        """解析关键点"""
        keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        keypoints = {}
        for i, name in enumerate(keypoint_names):
            if i < len(kpts):
                x, y, conf = kpts[i]
                keypoints[name] = [float(x), float(y), float(conf)]
        
        return keypoints
    
    def get_stats(self) -> Dict[str, float]:
        """获取性能统计"""
        return self.stats.copy()


# 测试代码
if __name__ == '__main__':
    print("Testing MultiTaskNetwork...")
    
    # 创建模型
    model = MultiTaskNetwork()
    
    # 测试输入
    x = torch.randn(1, 3, 480, 640)
    
    # 前向传播
    with torch.no_grad():
        outputs = model(x)
    
    print(f"Detection outputs: {[o.shape for o in outputs['detection']]}")
    print(f"Keypoint outputs: {[o.shape for o in outputs['keypoints']]}")
    print(f"Depth output: {outputs['depth'].shape}")
    
    # 参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params/1e6:.2f}M")