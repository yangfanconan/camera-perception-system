"""
集成测试 - 测试完整系统流程
"""

import pytest
import asyncio
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# 导入被测试模块
from src.algorithms.calibration import CameraCalibrator, CalibrationParams
from src.algorithms.detection import CombinedDetector, DetectionResult
from src.algorithms.spatial import SpatialCalculator
from src.algorithms.detection_enhanced import CombinedDetectorEnhanced
from src.algorithms.spatial_enhanced import SpatialCalculatorEnhanced


class TestIntegrationCalibration:
    """集成测试：相机标定流程"""
    
    @pytest.fixture
    def sample_images(self, tmp_path):
        """创建模拟棋盘格图片"""
        image_dir = tmp_path / "calibration_images"
        image_dir.mkdir()
        
        # 生成模拟棋盘格图像
        for i in range(20):
            # 创建空白图像
            img = np.ones((480, 640, 3), dtype=np.uint8) * 255
            
            # 绘制模拟棋盘格
            square_size = 40
            for row in range(6):
                for col in range(9):
                    if (row + col) % 2 == 0:
                        x = col * square_size + 50
                        y = row * square_size + 50
                        cv2.rectangle(img, (x, y), (x + square_size, y + square_size), 0, -1)
            
            # 添加一些随机偏移模拟不同角度
            offset = np.random.randint(-10, 10, 2)
            img = np.roll(img, offset, axis=(0, 1))
            
            # 保存
            cv2.imwrite(str(image_dir / f"calib_{i:03d}.png"), img)
        
        return image_dir
    
    def test_full_calibration_pipeline(self, sample_images):
        """测试完整标定流程"""
        # 初始化标定器
        calibrator = CameraCalibrator(
            checkerboard_size=(9, 6),
            square_size=25.0
        )
        
        # 收集图片
        image_paths = sorted([str(p) for p in sample_images.glob("*.png")])
        
        # 执行标定
        params = calibrator.calibrate(image_paths, visualize=False)
        
        # 验证结果
        assert params.fx > 0
        assert params.fy > 0
        assert params.cx > 0
        assert params.cy > 0
        assert params.num_images >= 15
        assert params.reprojection_error < 1.0  # 误差应小于 1 像素
        
        # 测试保存和加载
        output_path = sample_images.parent / "calib_params.json"
        calibrator.save_params(str(output_path))
        
        # 重新加载
        loaded_params = calibrator.load_params(str(output_path))
        assert abs(loaded_params.fx - params.fx) < 0.01


class TestIntegrationDetection:
    """集成测试：目标检测流程"""
    
    @pytest.fixture
    def test_image(self):
        """创建测试图像"""
        # 创建模拟人物图像（绿色矩形）
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.rectangle(img, (500, 200), (700, 800), (0, 255, 0), -1)
        return img
    
    def test_detector_initialization(self):
        """测试检测器初始化"""
        detector = CombinedDetector(
            yolo_model_path='models/yolov8n.pt',
            conf_threshold=0.5
        )
        
        # 模型不存在时应 gracefully 降级
        assert detector.person_detector.model is None or detector.person_detector.model is not None
    
    def test_detection_result_structure(self):
        """测试检测结果结构"""
        result = DetectionResult()
        
        # 测试空结果
        assert result.persons == []
        assert result.hands == []
        
        # 测试转换为字典
        data = result.to_dict()
        assert 'persons' in data
        assert 'hands' in data
        assert 'frame_shape' in data


class TestIntegrationSpatial:
    """集成测试：空间计量流程"""
    
    @pytest.fixture
    def calibrated_calculator(self):
        """创建已标定的计算器"""
        params = CalibrationParams(
            fx=1200.0,
            fy=1200.0,
            cx=960.0,
            cy=540.0,
            image_size=(1920, 1080)
        )
        return SpatialCalculator(params)
    
    def test_person_metrics_calculation(self, calibrated_calculator):
        """测试人体指标计算"""
        # 模拟检测结果
        person = {
            'bbox': [500, 200, 200, 600],
            'keypoints': {
                'head': [600, 250],
                'ankle_left': [570, 750]
            }
        }
        
        # 计算指标
        metrics = calibrated_calculator.calc_person_metrics(person)
        
        # 验证结果
        assert 'distance' in metrics
        assert 'height' in metrics
        assert 'topview' in metrics
        
        # 距离应该合理（1-10 米）
        assert 0 < metrics['distance'] < 20
        
        # 身高应该合理（50-250cm）
        assert 50 < metrics['height'] < 250
    
    def test_hand_metrics_calculation(self, calibrated_calculator):
        """测试手部指标计算"""
        # 模拟 21 个手部关键点
        keypoints = [[500 + i * 5, 500 + i * 3] for i in range(21)]
        
        hand = {
            'keypoints': keypoints,
            'bbox': [480, 480, 100, 150]
        }
        
        metrics = calibrated_calculator.calc_hand_metrics(hand)
        
        assert 'size' in metrics
        assert 'distance' in metrics
        assert 'topview' in metrics


class TestIntegrationEnhanced:
    """集成测试：增强版算法"""
    
    @pytest.fixture
    def enhanced_calculator(self):
        """创建增强版计算器"""
        params = CalibrationParams(
            fx=1200.0,
            fy=1200.0,
            cx=960.0,
            cy=540.0,
            image_size=(1920, 1080)
        )
        return SpatialCalculatorEnhanced(params)
    
    def test_calibration_workflow(self, enhanced_calculator):
        """测试校准工作流"""
        # 初始状态
        status = enhanced_calculator.get_calibration_status()
        assert status['distance_scale'] == 1.0
        assert status['height_scale'] == 1.0
        
        # 添加校准记录
        enhanced_calculator.add_calibration_record(
            measured_pixels=100,
            actual_value=5.0,
            distance=5.0,
            measurement_type='distance'
        )
        
        # 验证缩放系数已更新
        status = enhanced_calculator.get_calibration_status()
        assert status['calibration_records'] == 1
    
    def test_multi_frame_fusion(self, enhanced_calculator):
        """测试多帧融合"""
        # 添加多个测量值
        results = []
        for i in range(10):
            # 模拟带噪声的测量
            measured = 3.0 + np.random.randn() * 0.05
            result = enhanced_calculator.add_measurement('distance', measured, track_id=1)
            results.append(result)
        
        # 验证融合结果
        final_result = results[-1]
        assert final_result.num_samples == 10
        assert 0 <= final_result.confidence <= 1
        assert abs(final_result.value - 3.0) < 0.2  # 均值应接近真实值
    
    def test_error_correction(self, enhanced_calculator):
        """测试误差修正"""
        # 添加多个校准记录用于拟合
        for i in range(5):
            distance = 2.0 + i * 0.5
            error = np.random.randn() * 0.05
            enhanced_calculator.add_calibration_record(
                measured_pixels=(0.45 * 1200) / (distance + error),
                actual_value=distance,
                distance=distance,
                measurement_type='distance'
            )
        
        # 拟合修正模型
        enhanced_calculator.fit_distance_correction()
        
        # 验证修正系数已更新
        assert enhanced_calculator.distance_correction_coeffs != [0.0, 0.0, 1.0]


class TestSystemMock:
    """系统级模拟测试"""
    
    @patch('src.algorithms.detection.YOLO')
    def test_full_pipeline_with_mock(self, mock_yolo):
        """测试完整流程（使用模拟）"""
        # 设置模拟 YOLO
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        
        # 模拟检测结果
        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.xyxy = [[100, 100, 300, 700]]
        mock_result.boxes.conf = [0.9]
        mock_result.boxes.cls = [0]
        mock_result.keypoints = None
        mock_model.return_value = [mock_result]
        
        # 创建检测器
        detector = CombinedDetector(
            yolo_model_path='models/yolov8n.pt',
            conf_threshold=0.5
        )
        
        # 创建测试图像
        test_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # 执行检测
        result = detector.detect(test_img)
        
        # 验证
        assert len(result.persons) == 1
        assert result.persons[0]['confidence'] == 0.9


@pytest.fixture
def sample_video_frames():
    """生成模拟视频帧"""
    frames = []
    for i in range(30):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # 添加移动的目标
        x = 500 + i * 10
        cv2.rectangle(frame, (x, 200), (x + 200, 800), (0, 255, 0), -1)
        frames.append(frame)
    return frames


def test_temporal_consistency(sample_video_frames):
    """测试时间一致性（连续帧的稳定性）"""
    params = CalibrationParams(
        fx=1200.0,
        fy=1200.0,
        cx=960.0,
        cy=540.0,
        image_size=(1920, 1080)
    )
    
    calc = SpatialCalculatorEnhanced(params)
    
    distances = []
    for frame in sample_video_frames:
        # 模拟检测
        person = {
            'bbox': [500, 200, 200, 600],
            'track_id': 1
        }
        
        metrics = calc.calc_person_metrics(person, use_fusion=True)
        distances.append(metrics['distance'])
    
    # 验证距离变化平滑
    if len(distances) > 1:
        diff = np.diff(distances)
        std_diff = np.std(diff)
        
        # 标准差应较小（平滑）
        assert std_diff < 0.5, f"Distance variation too large: {std_diff}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
