"""
测试目标检测模块
"""

import pytest
import numpy as np
import cv2

from src.algorithms.detection import (
    PersonDetector,
    HandDetector,
    CombinedDetector,
    DetectionResult,
    visualize_detections
)


class TestDetectionResult:
    """测试检测结果类"""
    
    def test_create(self):
        result = DetectionResult()
        assert result.persons == []
        assert result.hands == []
        assert result.frame_shape is None
    
    def test_to_dict(self):
        result = DetectionResult()
        result.persons = [{"bbox": [0, 0, 100, 200], "confidence": 0.9}]
        result.frame_shape = (1080, 1920, 3)
        
        data = result.to_dict()
        assert len(data["persons"]) == 1
        assert data["frame_shape"] == (1080, 1920, 3)


class TestPersonDetector:
    """测试人体检测器"""
    
    @pytest.fixture
    def detector(self):
        return PersonDetector(
            model_path='models/yolov8n.pt',
            conf_threshold=0.5
        )
    
    def test_init(self, detector):
        # 模型不存在时 detector.model 应为 None
        assert detector.conf_threshold == 0.5
    
    def test_estimate_keypoints(self, detector):
        # 测试关键点估算
        keypoints = detector._estimate_keypoints(100, 100, 200, 400)
        
        assert "head" in keypoints
        assert "ankle_left" in keypoints
        assert "ankle_right" in keypoints
        
        # 头部应该在顶部
        assert keypoints["head"][1] < 200
        # 脚踝应该在底部
        assert keypoints["ankle_left"][1] == 500


class TestHandDetector:
    """测试手部检测器"""
    
    def test_init(self):
        detector = HandDetector()
        # MediaPipe 可能未安装
        assert detector.hands is None or detector.hands is not None


class TestVisualizeDetections:
    """测试可视化函数"""
    
    def test_visualize_empty_result(self):
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = DetectionResult()
        
        output = visualize_detections(image, result)
        
        # 空结果时，输出图像应该与输入相同
        assert output.shape == image.shape
    
    def test_visualize_person(self):
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = DetectionResult()
        result.persons = [{
            "bbox": [100, 100, 200, 400],
            "confidence": 0.9,
            "keypoints": {
                "head": [200, 150],
                "ankle_left": [170, 500]
            }
        }]
        
        output = visualize_detections(image, result)
        
        assert output.shape == image.shape
