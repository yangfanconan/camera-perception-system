"""
测试相机标定模块
"""

import pytest
import numpy as np
from pathlib import Path

from src.algorithms.calibration import CameraCalibrator, CalibrationParams


class TestCalibrationParams:
    """测试标定参数类"""
    
    def test_create_default(self):
        params = CalibrationParams()
        assert params.fx == 0.0
        assert params.fy == 0.0
        assert len(params.dist_coeffs) == 5
    
    def test_to_dict(self):
        params = CalibrationParams(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0)
        data = params.to_dict()
        assert data['fx'] == 1000.0
        assert data['fy'] == 1000.0
    
    def test_get_camera_matrix(self):
        params = CalibrationParams(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0)
        matrix = params.get_camera_matrix()
        assert matrix.shape == (3, 3)
        assert matrix[0, 0] == 1000.0
        assert matrix[1, 1] == 1000.0
        assert matrix[0, 2] == 960.0
        assert matrix[1, 2] == 540.0


class TestCameraCalibrator:
    """测试相机标定器"""
    
    def test_init(self):
        calibrator = CameraCalibrator(
            checkerboard_size=(9, 6),
            square_size=25.0
        )
        assert calibrator.checkerboard_size == (9, 6)
        assert calibrator.square_size == 25.0
    
    def test_prepare_object_points(self):
        calibrator = CameraCalibrator(
            checkerboard_size=(9, 6),
            square_size=25.0
        )
        objpoints = calibrator.prepare_object_points()
        assert objpoints.shape == (54, 3)  # 9*6=54 个角点
        assert objpoints.dtype == np.float32


class TestCalibrationIntegration:
    """集成测试"""
    
    @pytest.fixture
    def calibrator(self):
        return CameraCalibrator(
            checkerboard_size=(9, 6),
            square_size=25.0
        )
    
    def test_save_load_params(self, calibrator, tmp_path):
        # 创建模拟参数
        calibrator.calib_params = CalibrationParams(
            fx=1200.0,
            fy=1200.0,
            cx=960.0,
            cy=540.0
        )
        
        # 保存
        filepath = str(tmp_path / "calib.json")
        calibrator.save_params(filepath)
        
        # 加载
        loaded_params = calibrator.load_params(filepath)
        
        assert loaded_params.fx == 1200.0
        assert loaded_params.cx == 960.0
