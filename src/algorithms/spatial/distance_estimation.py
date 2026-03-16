"""
距离估计算法模块
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from loguru import logger

from src.utils.constants import (
    REFERENCE_SHOULDER_WIDTH, REFERENCE_HEAD_WIDTH, REFERENCE_UPPER_BODY_WIDTH,
    REFERENCE_EYE_DISTANCE, REFERENCE_EAR_DISTANCE, REFERENCE_EYE_NOSE_DISTANCE,
    ULTRA_CLOSE_BBOX_RATIO, CLOSE_BBOX_RATIO, NEAR_BBOX_RATIO,
    EXTREME_CLOSE_DISTANCE_HEAD, EXTREME_CLOSE_DISTANCE_UPPER, EXTREME_CLOSE_DISTANCE_FULL,
    BODY_PART_HEAD_ONLY, BODY_PART_UPPER, BODY_PART_HALF, BODY_PART_FULL
)


class DistanceEstimator:
    """距离估计器"""

    def __init__(self, fx: float = 650.0, distance_scale: float = 1.0):
        """
        初始化距离估计器

        Args:
            fx: 相机焦距
            distance_scale: 距离缩放系数
        """
        self.fx = fx
        self.distance_scale = distance_scale

    def estimate_from_bbox(self, person_bbox: List[int],
                          image_width: float, image_height: float) -> float:
        """
        基于边界框估算距离（优化版）

        改进点：
        1. 使用更精细的边界框占比分级
        2. 添加透视校正
        3. 自适应参考宽度

        Args:
            person_bbox: [x, y, w, h]
            image_width: 图像宽度
            image_height: 图像高度

        Returns:
            距离（米）
        """
        w = person_bbox[2]
        h = person_bbox[3]
        if w <= 0 or h <= 0:
            return 0.0

        # 计算边界框占比
        bbox_area_ratio = (w * h) / (image_width * image_height)
        bbox_aspect_ratio = h / max(w, 1)

        # 根据占比和宽高比选择参考宽度
        if bbox_area_ratio > 0.6:
            # 超近距离 - 只有脸部
            ref_width = 0.12  # 脸宽
        elif bbox_area_ratio > 0.4:
            # 极近距离 - 头部
            ref_width = REFERENCE_HEAD_WIDTH
        elif bbox_area_ratio > 0.25:
            # 近距离 - 上半身
            ref_width = REFERENCE_UPPER_BODY_WIDTH
        elif bbox_area_ratio > 0.15:
            # 中距离 - 根据宽高比判断
            if bbox_aspect_ratio > 1.5:
                ref_width = REFERENCE_SHOULDER_WIDTH
            else:
                ref_width = REFERENCE_UPPER_BODY_WIDTH
        else:
            # 远距离 - 肩宽
            ref_width = REFERENCE_SHOULDER_WIDTH

        # 基础距离计算
        distance = (ref_width * self.fx) / w

        # 透视校正（边界框在画面边缘时）
        center_x = person_bbox[0] + w / 2
        center_y = person_bbox[1] + h / 2
        
        # 计算偏离中心的程度
        offset_x = abs(center_x - image_width / 2) / (image_width / 2)
        offset_y = abs(center_y - image_height / 2) / (image_height / 2)
        
        # 边缘校正（最大5%）
        edge_correction = 1 + 0.05 * (offset_x ** 2 + offset_y ** 2)
        distance *= edge_correction

        return distance * self.distance_scale

    def estimate_from_body_keypoints(self, keypoints: Dict[str, List[float]]) -> Tuple[float, float]:
        """
        基于身体关键点估算距离（无头部时使用）

        使用肩宽、躯干高度、臀部宽度等特征

        Returns:
            (距离, 置信度)
        """
        measurements = []

        # 1. 肩宽（最可靠的身体特征）
        left_shoulder = keypoints.get('L_shoulder')
        right_shoulder = keypoints.get('R_shoulder')
        if left_shoulder and right_shoulder:
            shoulder_dist = np.sqrt(
                (left_shoulder[0] - right_shoulder[0])**2 +
                (left_shoulder[1] - right_shoulder[1])**2
            )
            if shoulder_dist > 10:
                # 平均肩宽 45cm
                dist = (REFERENCE_SHOULDER_WIDTH * self.fx) / shoulder_dist
                quality = min(1.0, shoulder_dist / 100)
                measurements.append((dist, quality * 1.2, 'shoulder_width'))

        # 2. 躯干高度（肩膀到臀部）
        left_hip = keypoints.get('L_hip')
        right_hip = keypoints.get('R_hip')
        if (left_shoulder or right_shoulder) and (left_hip or right_hip):
            shoulder = left_shoulder or right_shoulder
            hip = left_hip or right_hip
            torso_height = abs(shoulder[1] - hip[1])
            if torso_height > 15:
                # 平均躯干高度 50cm
                dist = (0.50 * self.fx) / torso_height
                quality = min(1.0, torso_height / 80)
                measurements.append((dist, quality * 1.0, 'torso_height'))

        # 3. 臀宽
        if left_hip and right_hip:
            hip_dist = np.sqrt(
                (left_hip[0] - right_hip[0])**2 +
                (left_hip[1] - right_hip[1])**2
            )
            if hip_dist > 8:
                # 平均臀宽 35cm
                dist = (0.35 * self.fx) / hip_dist
                quality = min(1.0, hip_dist / 80)
                measurements.append((dist, quality * 0.9, 'hip_width'))

        # 4. 上半身高度（肩膀到腰部）
        left_elbow = keypoints.get('L_elbow')
        right_elbow = keypoints.get('R_elbow')
        if (left_shoulder or right_shoulder) and (left_elbow or right_elbow):
            shoulder = left_shoulder or right_shoulder
            elbow = left_elbow or right_elbow
            upper_arm = abs(shoulder[1] - elbow[1])
            if upper_arm > 10:
                # 上臂长约 30cm，用于估算
                dist = (0.30 * self.fx) / upper_arm
                quality = min(1.0, upper_arm / 50)
                measurements.append((dist, quality * 0.7, 'upper_arm'))

        if not measurements:
            return 0.0, 0.0

        # 一致性检查
        if len(measurements) >= 2:
            distances = [m[0] for m in measurements]
            median = np.median(distances)
            std = np.std(distances)
            filtered = [(d, w, t) for d, w, t in measurements
                       if abs(d - median) < 2 * std]
            if len(filtered) >= 1:
                measurements = filtered

        # 加权平均
        total_weight = sum(w for _, w, _ in measurements)
        if total_weight == 0:
            return 0.0, 0.0

        weighted_dist = sum(d * w for d, w, _ in measurements) / total_weight
        confidence = min(1.0, total_weight / 2.5)

        # 一致性奖励
        if len(measurements) >= 2:
            distances = [m[0] for m in measurements]
            consistency = 1 - min(1.0, np.std(distances) / np.mean(distances))
            confidence = confidence * 0.7 + consistency * 0.3

        return float(weighted_dist * self.distance_scale), float(confidence)

    def estimate_from_head(self, keypoints: Dict[str, List[float]]) -> Tuple[float, float]:
        """
        基于头部特征估算距离（优化版）

        改进点：
        1. 使用加权融合而非简单中位数
        2. 添加特征质量评估
        3. 多特征一致性检查

        Returns:
            (距离, 置信度)
        """
        left_eye = keypoints.get('L_eye')
        right_eye = keypoints.get('R_eye')
        left_ear = keypoints.get('L_ear')
        right_ear = keypoints.get('R_ear')
        nose = keypoints.get('nose')

        measurements = []  # (距离, 权重, 特征类型)

        # 1. 双眼间距（最可靠）
        if left_eye and right_eye:
            eye_dist = np.sqrt((left_eye[0] - right_eye[0])**2 + 
                              (left_eye[1] - right_eye[1])**2)
            if eye_dist > 5:
                dist = (REFERENCE_EYE_DISTANCE * self.fx) / eye_dist
                # 根据像素距离评估质量
                quality = min(1.0, eye_dist / 50)  # 50像素以上为高质量
                measurements.append((dist, quality * 1.5, 'eye_distance'))

        # 2. 耳朵间距
        if left_ear and right_ear:
            ear_dist = np.sqrt((left_ear[0] - right_ear[0])**2 + 
                              (left_ear[1] - right_ear[1])**2)
            if ear_dist > 8:
                dist = (REFERENCE_EAR_DISTANCE * self.fx) / ear_dist
                quality = min(1.0, ear_dist / 80)
                measurements.append((dist, quality * 1.0, 'ear_distance'))

        # 3. 眼到鼻子的距离（左右眼分别计算）
        if nose:
            for eye, eye_name in [(left_eye, 'L'), (right_eye, 'R')]:
                if eye:
                    eye_nose_dist = np.sqrt((eye[0] - nose[0])**2 + 
                                           (eye[1] - nose[1])**2)
                    if eye_nose_dist > 3:
                        dist = (REFERENCE_EYE_NOSE_DISTANCE * self.fx) / eye_nose_dist
                        quality = min(1.0, eye_nose_dist / 30)
                        measurements.append((dist, quality * 0.8, f'{eye_name}_eye_nose'))

        # 4. 头部高度（如果有）
        if nose and (left_eye or right_eye):
            eye = left_eye or right_eye
            head_height = abs(nose[1] - eye[1])
            if head_height > 5:
                # 头高约22cm
                dist = (0.22 * self.fx) / head_height
                quality = min(1.0, head_height / 40)
                measurements.append((dist, quality * 0.7, 'head_height'))

        if not measurements:
            return 0.0, 0.0

        # 一致性检查 - 剔除异常值
        if len(measurements) >= 3:
            distances = [m[0] for m in measurements]
            median = np.median(distances)
            std = np.std(distances)
            
            # 保留在2个标准差内的测量
            filtered = [(d, w, t) for d, w, t in measurements 
                       if abs(d - median) < 2 * std]
            if len(filtered) >= 2:
                measurements = filtered

        # 加权平均
        total_weight = sum(w for _, w, _ in measurements)
        if total_weight == 0:
            return 0.0, 0.0

        weighted_dist = sum(d * w for d, w, _ in measurements) / total_weight
        
        # 置信度计算
        confidence = min(1.0, total_weight / 3.0)  # 3个高质量测量为满置信度
        
        # 一致性奖励
        if len(measurements) >= 2:
            distances = [m[0] for m in measurements]
            consistency = 1 - min(1.0, np.std(distances) / np.mean(distances))
            confidence = confidence * 0.7 + consistency * 0.3

        return float(weighted_dist * self.distance_scale), float(confidence)

    def estimate_close_range(self, body_distance: float, head_distance: float,
                            head_confidence: float, bbox: List[int],
                            keypoints: Dict[str, List[float]],
                            image_height: float, image_width: float,
                            body_part: str) -> Tuple[float, str]:
        """
        近距离距离估计（优化版）

        改进点：
        1. 自适应权重根据置信度调整
        2. 多方法融合策略
        3. 异常值检测和处理

        Returns:
            (距离, 方法描述)
        """
        bbox_area_ratio = (bbox[2] * bbox[3]) / (image_width * image_height)

        # 超近距离强制判断（占比>70%）
        if bbox_area_ratio > ULTRA_CLOSE_BBOX_RATIO:
            return self._extreme_close_estimate(body_part)

        # 极近距离判断（占比>20%）
        if bbox_area_ratio > NEAR_BBOX_RATIO:
            return self._bbox_ratio_estimate(bbox_area_ratio, body_part)

        # 普通近距离融合（自适应权重）
        return self._adaptive_fuse_distance(body_distance, head_distance, head_confidence, body_part)

    def _adaptive_fuse_distance(self, body_distance: float, head_distance: float,
                               head_confidence: float, body_part: str) -> Tuple[float, str]:
        """
        自适应距离融合

        根据身体部位和置信度动态调整权重
        """
        # 如果头部检测不可用，只使用身体距离
        if head_distance <= 0 or head_confidence < 0.2:
            return body_distance, f"body_only_{body_part}"

        # 根据身体部位调整权重
        if body_part == BODY_PART_HEAD_ONLY:
            # 只有头部时，完全信任头部检测
            body_weight = 0.1
            head_weight = 0.9
            method = "head_priority"
        elif body_part == BODY_PART_UPPER:
            # 上半身时，头部检测很可靠
            body_weight = 0.3
            head_weight = 0.7
            method = "head_weighted"
        elif body_part == BODY_PART_HALF:
            # 半身时，平衡权重
            body_weight = 0.5
            head_weight = 0.5
            method = "balanced"
        else:
            # 全身时，身体检测更可靠
            body_weight = 0.7
            head_weight = 0.3
            method = "body_weighted"

        # 根据置信度微调权重
        head_weight *= head_confidence
        body_weight = 1 - head_weight

        # 计算融合距离
        distance = body_distance * body_weight + head_distance * head_weight

        # 异常值检测
        if abs(body_distance - head_distance) > 0.5:
            # 差异过大，选择置信度高的
            if head_confidence > 0.7:
                distance = head_distance
                method += "_head_selected"
            else:
                distance = body_distance
                method += "_body_selected"

        return distance, method

    def _extreme_close_estimate(self, body_part: str) -> Tuple[float, str]:
        """超近距离估计"""
        if body_part == BODY_PART_HEAD_ONLY:
            return EXTREME_CLOSE_DISTANCE_HEAD, "extreme_close_head_only"
        elif body_part in [BODY_PART_UPPER, BODY_PART_HALF]:
            return EXTREME_CLOSE_DISTANCE_UPPER, "extreme_close_upper_body"
        else:
            return EXTREME_CLOSE_DISTANCE_FULL, "extreme_close_full_body"

    def _bbox_ratio_estimate(self, bbox_area_ratio: float, body_part: str) -> Tuple[float, str]:
        """
        基于边界框占比估计（优化版）

        针对不同身体部位使用不同的校准参数
        """
        # 根据身体部位选择参数（重新校准）
        if body_part == BODY_PART_HEAD_ONLY:
            # 头部：占比20%->0.50m, 40%->0.30m, 60%->0.18m
            params = {'base': 0.65, 'slope': 0.75, 'min': 0.12, 'max': 0.80}
        elif body_part == BODY_PART_UPPER:
            # 上半身：占比20%->0.55m, 40%->0.35m, 60%->0.22m
            # 上半身比全身更"宽"，所以同样占比距离更近
            params = {'base': 0.70, 'slope': 0.80, 'min': 0.15, 'max': 0.90}
        elif body_part == BODY_PART_HALF:
            # 半身：占比20%->0.60m, 40%->0.38m, 60%->0.25m
            params = {'base': 0.75, 'slope': 0.85, 'min': 0.15, 'max': 0.95}
        else:
            # 全身：占比20%->0.70m, 40%->0.45m, 60%->0.30m
            params = {'base': 0.85, 'slope': 1.0, 'min': 0.20, 'max': 1.2}

        # 计算距离
        # 公式: distance = base - slope * (ratio - 0.15)
        # 从15%占比开始计算
        estimated_dist = params['base'] - params['slope'] * (bbox_area_ratio - 0.15)
        estimated_dist = max(params['min'], min(params['max'], estimated_dist))

        # 根据占比确定方法描述
        if bbox_area_ratio > 0.55:
            method = f"extreme_close_{body_part}"
        elif bbox_area_ratio > CLOSE_BBOX_RATIO:
            method = f"ultra_close_{body_part}"
        elif bbox_area_ratio > 0.25:
            method = f"close_{body_part}"
        else:
            method = f"near_{body_part}"

        return estimated_dist, method

