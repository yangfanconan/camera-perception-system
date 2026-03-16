"""
测试空间计量模块
"""

import pytest
import numpy as np

from src.algorithms.calibration import CalibrationParams
from src.algorithms.spatial import (
    SpatialCalculator,
    WorldPoint,
    TopViewPoint
)


@pytest.fixture
def sample_calib_params():
    """示例标定参数"""
    return CalibrationParams(
        fx=1200.0,
        fy=1200.0,
        cx=960.0,
        cy=540.0,
        image_size=(1920, 1080)
    )


@pytest.fixture
def calculator(sample_calib_params):
    """空间计算器"""
    return SpatialCalculator(sample_calib_params)


class TestSpatialCalculator:
    """测试空间计算器"""
    
    def test_pixel_to_camera_coords(self, calculator):
        # 图像中心点，深度 3 米
        X_c, Y_c, Z_c = calculator.pixel_to_camera_coords(960, 540, 3.0)
        
        assert np.isclose(X_c, 0.0, atol=0.01)
        assert np.isclose(Y_c, 0.0, atol=0.01)
        assert Z_c == 3.0
    
    def test_calc_distance(self, calculator):
        # 肩宽 0.45m，在图像中占 100 像素
        distance = calculator.calc_person_distance(
            person_bbox=[0, 0, 100, 300],
            ref_shoulder_width=0.45
        )
        
        # Z = (0.45 * 1200) / 100 = 5.4m
        assert np.isclose(distance, 5.4, atol=0.1)
    
    def test_calc_height(self, calculator):
        # 头部 (960, 200), 脚踝 (960, 800), 距离 3 米
        head = [960, 200]
        ankle = [960, 800]
        distance = 3.0
        
        height = calculator.calc_height(head, ankle, distance)
        
        # H = (600 * 3) / 1200 = 1.5m = 150cm
        assert np.isclose(height, 150.0, atol=1.0)
    
    def test_calc_hand_size(self, calculator):
        # 掌根 (500, 500), 指尖 (500, 600), 距离 1 米
        palm = [500, 500]
        finger = [500, 600]
        distance = 1.0
        
        size = calculator.calc_hand_size(palm, finger, distance)
        
        # D = (100 * 1) / 1200 = 0.083m = 8.3cm
        assert np.isclose(size, 8.33, atol=0.5)
    
    def test_world_to_topview(self, calculator):
        calculator.set_camera_extrinsics(height=1.8, pitch_angle=30.0)
        
        world_point = WorldPoint(x=0.0, y=0.0, z=3.0)
        topview = calculator.world_to_topview(world_point)
        
        # x_top = 400 + 0 * 10 = 400
        # y_top = 300 - 3 * 10 = 270
        assert topview.x == 400.0
        assert topview.y == 270.0


class TestWorldPoint:
    """测试世界坐标点"""
    
    def test_create(self):
        point = WorldPoint(x=1.0, y=2.0, z=3.0)
        assert point.x == 1.0
        assert point.y == 2.0
        assert point.z == 3.0
    
    def test_to_array(self):
        point = WorldPoint(x=1.0, y=2.0, z=3.0)
        arr = point.to_array()
        assert np.array_equal(arr, np.array([1.0, 2.0, 3.0]))
