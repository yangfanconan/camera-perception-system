"""
Pytest 配置文件
"""

import pytest
import numpy as np
import cv2


def pytest_configure(config):
    """Pytest 配置"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


@pytest.fixture
def sample_calibration_params():
    """示例标定参数"""
    from src.algorithms.calibration import CalibrationParams
    
    return CalibrationParams(
        fx=1200.0,
        fy=1200.0,
        cx=960.0,
        cy=540.0,
        dist_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
        image_size=(1920, 1080),
        checkerboard_size=(9, 6),
        square_size=25.0
    )


@pytest.fixture
def sample_image():
    """示例测试图像"""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def sample_checkerboard_image():
    """示例棋盘格图像"""
    # 创建 480x640 的棋盘格图像
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    square_size = 40
    
    for row in range(6):
        for col in range(9):
            if (row + col) % 2 == 0:
                x = col * square_size + 50
                y = row * square_size + 50
                cv2.rectangle(img, (x, y), (x + square_size, y + square_size), 0, -1)
    
    return img


@pytest.fixture
def temp_config_file(tmp_path):
    """临时配置文件"""
    config_content = """
camera:
  id: 0
  resolution: [1920, 1080]
  fps: 20

calibration:
  checkerboard: [9, 6]
  square_size: 25.0
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    return config_file
