"""
身体部位检测模块
"""

from typing import Tuple, Dict, List
from loguru import logger

from src.utils.constants import (
    BODY_PART_FULL, BODY_PART_HALF, BODY_PART_UPPER, BODY_PART_LOWER,
    BODY_PART_HEAD_ONLY, BODY_PART_UNKNOWN,
    BODY_PART_MIN_KEYPOINTS,
    ULTRA_CLOSE_BBOX_RATIO, CLOSE_BBOX_RATIO
)


class BodyPartDetector:
    """身体部位检测器"""

    # 关键点定义
    HEAD_POINTS = ['nose', 'L_eye', 'R_eye', 'L_ear', 'R_ear']
    UPPER_POINTS = ['L_shoulder', 'R_shoulder', 'L_elbow', 'R_elbow', 'L_wrist', 'R_wrist']
    LOWER_POINTS = ['L_hip', 'R_hip', 'L_knee', 'R_knee', 'L_ankle', 'R_ankle']

    def detect(self, keypoints: Dict[str, List[float]], bbox: List[int],
               image_height: float, image_width: float) -> Tuple[str, float, Dict]:
        """
        检测身体部位（优化版）

        Returns:
            (部位类型, 置信度, 详细信息)
        """
        # 获取各部位关键点
        head_kps = {p: keypoints[p] for p in self.HEAD_POINTS if p in keypoints}
        upper_kps = {p: keypoints[p] for p in self.UPPER_POINTS if p in keypoints}
        lower_kps = {p: keypoints[p] for p in self.LOWER_POINTS if p in keypoints}

        # 计算关键点数量和质量
        head_count = len(head_kps)
        upper_count = len(upper_kps)
        lower_count = len(lower_kps)

        # 评估关键点质量
        head_quality = self._evaluate_keypoint_quality(head_kps)
        upper_quality = self._evaluate_keypoint_quality(upper_kps)
        lower_quality = self._evaluate_keypoint_quality(lower_kps)

        # 判断各部位是否存在
        has_head = head_count >= BODY_PART_MIN_KEYPOINTS and head_quality > 0.3
        has_upper = upper_count >= BODY_PART_MIN_KEYPOINTS and upper_quality > 0.3
        has_lower = lower_count >= BODY_PART_MIN_KEYPOINTS and lower_quality > 0.3

        # 边界框分析
        bbox_y1 = bbox[1]
        bbox_y2 = bbox[1] + bbox[3]
        bbox_height = bbox[3]
        bbox_width = bbox[2]
        bbox_area = bbox_width * bbox_height
        image_area = image_width * image_height

        bbox_height_ratio = bbox_height / image_height
        bbox_width_ratio = bbox_width / image_width
        bbox_area_ratio = bbox_area / image_area
        bbox_bottom_ratio = bbox_y2 / image_height
        bbox_top_ratio = bbox_y1 / image_height

        # 检测是否被裁剪
        body_clipped = self._check_clipped(bbox, image_width, image_height)

        # 边界框与关键点一致性检查
        bbox_keypoint_consistent = self._check_bbox_keypoint_consistency(
            bbox, head_kps, upper_kps, lower_kps, image_height
        )

        # 判断身体部位
        body_part, confidence = self._determine_body_part(
            has_head, has_upper, has_lower,
            bbox_height_ratio, bbox_area_ratio,
            bbox_bottom_ratio, bbox_top_ratio,
            body_clipped, keypoints, image_height
        )

        # 如果不一致，降低置信度
        if not bbox_keypoint_consistent:
            confidence *= 0.8

        # 极近距离修正
        body_part, confidence = self._apply_close_range_correction(
            body_part, confidence, bbox_area_ratio, bbox_height_ratio
        )

        info = {
            'has_head': has_head,
            'has_upper': has_upper,
            'has_lower': has_lower,
            'bbox_area_ratio': bbox_area_ratio,
            'bbox_height_ratio': bbox_height_ratio,
            'bbox_width_ratio': bbox_width_ratio,
            'body_clipped': body_clipped,
            'keypoint_count': len(keypoints),
            'head_quality': head_quality,
            'upper_quality': upper_quality,
            'lower_quality': lower_quality
        }

        return body_part, confidence, info

    def _check_clipped(self, bbox: List[int], image_width: float, image_height: float) -> bool:
        """检查身体是否被画面裁剪"""
        bbox_y2 = bbox[1] + bbox[3]
        bbox_x2 = bbox[0] + bbox[2]

        return (
            bbox[1] < 5 or
            bbox_y2 > image_height - 5 or
            bbox[0] < 5 or
            bbox_x2 > image_width - 5
        )

    def _determine_body_part(self, has_head, has_upper, has_lower,
                            bbox_height_ratio, bbox_area_ratio,
                            bbox_bottom_ratio, bbox_top_ratio,
                            body_clipped, keypoints, image_height) -> Tuple[str, float]:
        """判断身体部位类型"""

        if has_head and has_upper and has_lower:
            return self._analyze_full_body(keypoints, image_height, bbox_height_ratio, body_clipped)

        elif has_head and has_upper and not has_lower:
            return BODY_PART_UPPER, 0.9

        elif not has_head and has_upper and has_lower:
            return BODY_PART_LOWER, 0.8

        elif has_head and not has_upper and not has_lower:
            if bbox_area_ratio > 0.15 or bbox_height_ratio > 0.25:
                return BODY_PART_HEAD_ONLY, 0.9
            return BODY_PART_HEAD_ONLY, 0.7

        elif has_head and has_lower and not has_upper:
            return BODY_PART_HALF, 0.6

        else:
            return self._infer_from_bbox(bbox_height_ratio, bbox_area_ratio,
                                         bbox_bottom_ratio, bbox_top_ratio)

    def _analyze_full_body(self, keypoints, image_height, bbox_height_ratio, body_clipped) -> Tuple[str, float]:
        """分析全身情况"""
        nose_y = keypoints.get('nose', [0, image_height/2])[1]
        shoulder_y = min(
            keypoints.get('L_shoulder', [0, image_height])[1],
            keypoints.get('R_shoulder', [0, image_height])[1]
        )
        hip_y = min(
            keypoints.get('L_hip', [0, image_height])[1],
            keypoints.get('R_hip', [0, image_height])[1]
        )
        ankle_y = max(
            keypoints.get('L_ankle', [0, 0])[1],
            keypoints.get('R_ankle', [0, 0])[1]
        )

        head_to_shoulder = shoulder_y - nose_y
        shoulder_to_hip = hip_y - shoulder_y
        hip_to_ankle = ankle_y - hip_y

        if hip_to_ankle < head_to_shoulder * 1.5:
            return BODY_PART_HALF, 0.75

        if body_clipped:
            return BODY_PART_FULL, 0.85

        if bbox_height_ratio > 0.6:
            return BODY_PART_FULL, 0.9

        return BODY_PART_FULL, 0.8

    def _infer_from_bbox(self, bbox_height_ratio, bbox_area_ratio,
                        bbox_bottom_ratio, bbox_top_ratio) -> Tuple[str, float]:
        """从边界框推断身体部位"""
        if bbox_height_ratio > 0.5 and bbox_area_ratio > 0.2:
            return BODY_PART_FULL, 0.4
        elif bbox_height_ratio > 0.3:
            if bbox_bottom_ratio > 0.8:
                return BODY_PART_LOWER, 0.4
            elif bbox_top_ratio < 0.2:
                return BODY_PART_UPPER, 0.4
            else:
                return BODY_PART_HALF, 0.4
        elif bbox_height_ratio > 0.15:
            return BODY_PART_HEAD_ONLY, 0.5
        else:
            return BODY_PART_UNKNOWN, 0.2

    def _apply_close_range_correction(self, body_part, confidence,
                                     bbox_area_ratio, bbox_height_ratio) -> Tuple[str, float]:
        """极近距离修正"""
        if bbox_area_ratio > ULTRA_CLOSE_BBOX_RATIO:
            return BODY_PART_HEAD_ONLY, 0.95

        if bbox_area_ratio > CLOSE_BBOX_RATIO:
            if body_part == BODY_PART_FULL:
                if bbox_height_ratio > 0.6:
                    return BODY_PART_UPPER, 0.85
                elif bbox_area_ratio > 0.5:
                    return BODY_PART_UPPER, 0.8
            elif body_part == BODY_PART_UPPER and bbox_area_ratio > 0.45:
                return BODY_PART_HEAD_ONLY, 0.9
            elif body_part == BODY_PART_HALF and bbox_area_ratio > 0.4:
                return BODY_PART_UPPER, 0.85

        return body_part, confidence

    def _evaluate_keypoint_quality(self, keypoints: Dict[str, List[float]]) -> float:
        """评估关键点质量"""
        if len(keypoints) < 2:
            return 0.0

        points = list(keypoints.values())
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        area = width * height

        if area < 100:
            return 0.3
        elif area > 10000:
            return 0.5
        else:
            return min(1.0, area / 5000)

    def _check_bbox_keypoint_consistency(self, bbox: List[int],
                                        head_kps: Dict, upper_kps: Dict,
                                        lower_kps: Dict, image_height: float) -> bool:
        """检查边界框与关键点是否一致"""
        bbox_y1 = bbox[1]
        bbox_y2 = bbox[1] + bbox[3]

        all_ys = []
        for kps in [head_kps, upper_kps, lower_kps]:
            for pt in kps.values():
                all_ys.append(pt[1])

        if not all_ys:
            return True

        min_y = min(all_ys)
        max_y = max(all_ys)

        margin = 20
        if min_y < bbox_y1 - margin or max_y > bbox_y2 + margin:
            return False

        keypoint_span = max_y - min_y
        bbox_height = bbox[3]

        if bbox_height > 0:
            fill_ratio = keypoint_span / bbox_height
            return fill_ratio > 0.5

        return True
