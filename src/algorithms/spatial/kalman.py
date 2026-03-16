"""
卡尔曼滤波器模块
用于距离测量的平滑和预测
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class KalmanParams:
    """卡尔曼滤波参数"""
    max_speed: float = 3.0
    process_noise: float = 0.1
    measurement_noise: float = 0.3


class DistanceKalmanFilter:
    """
    距离卡尔曼滤波器

    状态向量: [距离, 速度]
    测量: 距离
    """

    def __init__(self,
                 initial_distance: float = 1.0,
                 max_speed: float = 3.0,
                 process_noise: float = 0.1,
                 measurement_noise: float = 0.3):
        """
        初始化卡尔曼滤波器

        Args:
            initial_distance: 初始距离估计
            max_speed: 最大移动速度（米/秒）
            process_noise: 过程噪声
            measurement_noise: 测量噪声
        """
        self.max_speed = max_speed

        # 状态向量 [距离, 速度]
        self.x = np.array([initial_distance, 0.0])

        # 状态协方差矩阵
        self.P = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])

        # 过程噪声协方差
        self.Q = np.array([
            [process_noise ** 2, 0.0],
            [0.0, (max_speed / 10) ** 2]
        ])

        # 测量噪声协方差
        self.R = np.array([[measurement_noise ** 2]])

        # 测量矩阵（只测量距离）
        self.H = np.array([[1.0, 0.0]])

        # 状态转移矩阵（在 update 中动态设置）
        self.F = np.eye(2)

        logger.debug(f"Kalman filter initialized: distance={initial_distance:.2f}m")

    def predict(self, dt: float):
        """预测步骤"""
        # 状态转移矩阵
        self.F = np.array([
            [1.0, dt],
            [0.0, 1.0]
        ])

        # 预测状态
        self.x = self.F @ self.x

        # 限制速度
        self.x[1] = np.clip(self.x[1], -self.max_speed, self.max_speed)

        # 预测协方差
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement: float, confidence: float = 1.0) -> np.ndarray:
        """
        更新步骤

        Args:
            measurement: 距离测量值
            confidence: 测量置信度 (0-1)

        Returns:
            更新后的状态向量
        """
        # 根据置信度调整测量噪声
        R = self.R / max(confidence, 0.1)

        # 测量残差
        y = measurement - self.H @ self.x

        # 残差协方差
        S = self.H @ self.P @ self.H.T + R

        # 卡尔曼增益
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 更新状态
        self.x = self.x + K @ y

        # 更新协方差
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P

        return self.x

    def filter(self, measurement: float, dt: float, confidence: float = 1.0) -> Tuple[float, float]:
        """
        执行完整的滤波步骤

        Args:
            measurement: 距离测量值
            dt: 时间间隔（秒）
            confidence: 测量置信度

        Returns:
            (滤波后的距离, 估计速度)
        """
        self.predict(dt)
        self.update(measurement, confidence)
        return float(self.x[0]), float(self.x[1])

    def get_motion_state(self) -> str:
        """获取运动状态"""
        velocity = abs(self.x[1])
        if velocity < 0.1:
            return 'static'
        elif velocity < 1.5:
            return 'walking'
        elif velocity < 3.0:
            return 'running'
        else:
            return 'fast'

    def get_state(self) -> dict:
        """获取当前状态"""
        return {
            'distance': float(self.x[0]),
            'velocity': float(self.x[1]),
            'uncertainty': float(np.trace(self.P)),
            'motion_state': self.get_motion_state()
        }

    def reset(self, distance: float):
        """重置滤波器"""
        self.x = np.array([distance, 0.0])
        self.P = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
