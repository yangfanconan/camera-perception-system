"""
空间计量模块 - 像素坐标 ↔ 世界坐标转换
功能：
1. 像素坐标→世界坐标转换
2. 距离计算（人到摄像头）
3. 身高计算
4. 手大小计算
5. 顶视图映射（3D→2D 鸟瞰图）
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
from loguru import logger

from .calibration import CalibrationParams


@dataclass
class WorldPoint:
    """世界坐标点"""
    x: float  # 水平方向 (米)
    y: float  # 垂直方向 (米)
    z: float  # 深度方向 (米)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class TopViewPoint:
    """顶视图坐标点"""
    x: float  # 顶视图 X 坐标 (像素)
    y: float  # 顶视图 Y 坐标 (像素)


class SpatialCalculator:
    """空间计量计算器"""
    
    def __init__(self, calibration_params: CalibrationParams):
        """
        初始化空间计算器
        
        Args:
            calibration_params: 相机标定参数
        """
        self.calib = calibration_params
        
        # 相机内参
        self.fx = calibration_params.fx
        self.fy = calibration_params.fy
        self.cx = calibration_params.cx
        self.cy = calibration_params.cy
        
        # 相机外参
        self.camera_height = 1.8  # 默认安装高度 (米)
        self.pitch_angle = 30.0   # 默认俯角 (度)
        
        # 顶视图配置
        self.topview_scale = 10.0  # 1 米 = 10 像素
        self.topview_origin = (400, 300)  # 顶视图原点 (画布中心)
        
        logger.info("SpatialCalculator initialized")
        logger.info(f"fx={self.fx:.2f}, fy={self.fy:.2f}, "
                   f"cx={self.cx:.2f}, cy={self.cy:.2f}")
    
    def set_camera_extrinsics(
        self, 
        height: float, 
        pitch_angle: float
    ) -> None:
        """
        设置相机外参
        
        Args:
            height: 摄像头安装高度 (米)
            pitch_angle: 俯角 (度)
        """
        self.camera_height = height
        self.pitch_angle = pitch_angle
        logger.info(f"Camera extrinsics: height={height}m, pitch={pitch_angle}°")
    
    def pixel_to_camera_coords(
        self, 
        u: float, 
        v: float, 
        z_c: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """
        像素坐标→相机坐标系
        
        公式：
        X_c = (u - cx) * Z_c / fx
        Y_c = (v - cy) * Z_c / fy
        Z_c = Z_c
        
        Args:
            u: 像素 x 坐标
            v: 像素 y 坐标
            z_c: 相机坐标系中的深度 (米)，如果为 None 则返回归一化方向向量
            
        Returns:
            (X_c, Y_c, Z_c) 相机坐标 (米)
        """
        # 归一化坐标
        x_norm = (u - self.cx) / self.fx
        y_norm = (v - self.cy) / self.fy
        
        if z_c is None:
            # 返回归一化方向向量
            return (x_norm, y_norm, 1.0)
        
        X_c = x_norm * z_c
        Y_c = y_norm * z_c
        
        return (X_c, Y_c, z_c)
    
    def calc_distance_to_camera(
        self,
        pixel_height: float,
        ref_height: float
    ) -> float:
        """
        通过相似三角形计算到摄像头的距离
        
        公式：Z = (ref_height * fx) / pixel_height
        
        Args:
            pixel_height: 物体的像素高度
            ref_height: 物体的实际参考高度 (米)
            
        Returns:
            距离 (米)
        """
        if pixel_height <= 0:
            return 0.0
        
        # 简化公式：Z = K * (ref_height / pixel_height)
        # 其中 K = fx (当传感器宽度已知时)
        distance = (ref_height * self.fx) / pixel_height
        
        return distance
    
    def calc_person_distance(
        self,
        person_bbox: List[int],
        ref_shoulder_width: float = 0.45
    ) -> float:
        """
        计算人到摄像头的距离
        
        Args:
            person_bbox: 人体边界框 [x, y, w, h]
            ref_shoulder_width: 参考肩宽 (米)
            
        Returns:
            距离 (米)
        """
        w = person_bbox[2]
        
        if w <= 0:
            return 0.0
        
        # 使用肩宽估算距离
        distance = (ref_shoulder_width * self.fx) / w
        
        logger.debug(f"Person distance: {distance:.2f}m (bbox width={w})")
        
        return distance
    
    def calc_height(
        self,
        head_point: List[int],
        ankle_point: List[int],
        distance: float
    ) -> float:
        """
        计算身高
        
        公式：H = (pixel_height * Z) / fx
        
        Args:
            head_point: 头部关键点 [x, y]
            ankle_point: 脚踝关键点 [x, y]
            distance: 人到摄像头的距离 (米)
            
        Returns:
            身高 (厘米)
        """
        # 像素高度
        pixel_height = abs(ankle_point[1] - head_point[1])
        
        # 实际高度 (米)
        height_m = (pixel_height * distance) / self.fx
        
        # 转换为厘米
        height_cm = height_m * 100
        
        logger.debug(f"Height: {height_cm:.1f}cm (pixel={pixel_height}, dist={distance:.2f}m)")
        
        return height_cm
    
    def calc_hand_size(
        self,
        palm_point: List[int],
        finger_point: List[int],
        distance: float
    ) -> float:
        """
        计算手大小（掌根到指尖的距离）
        
        Args:
            palm_point: 掌根关键点 [x, y]
            finger_point: 指尖关键点 [x, y]
            distance: 手到摄像头的距离 (米)
            
        Returns:
            手大小 (厘米)
        """
        # 像素距离
        pixel_dist = np.sqrt(
            (finger_point[0] - palm_point[0]) ** 2 +
            (finger_point[1] - palm_point[1]) ** 2
        )
        
        # 实际距离 (米)
        size_m = (pixel_dist * distance) / self.fx
        
        # 转换为厘米
        size_cm = size_m * 100
        
        logger.debug(f"Hand size: {size_cm:.1f}cm (pixel={pixel_dist:.1f}, dist={distance:.2f}m)")
        
        return size_cm
    
    def camera_to_world_coords(
        self,
        X_c: float,
        Y_c: float,
        Z_c: float
    ) -> WorldPoint:
        """
        相机坐标→世界坐标
        
        假设地面为 Y=0 平面，摄像头在 (0, height, 0) 位置
        
        Args:
            X_c: 相机 X 坐标
            Y_c: 相机 Y 坐标
            Z_c: 相机 Z 坐标 (深度)
            
        Returns:
            WorldPoint: 世界坐标点
        """
        # 考虑摄像头俯角
        pitch_rad = np.radians(self.pitch_angle)
        
        # 旋转后的坐标
        X_w = X_c
        Y_w = Z_c * np.sin(pitch_rad) - Y_c * np.cos(pitch_rad)
        Z_w = Z_c * np.cos(pitch_rad) + Y_c * np.sin(pitch_rad)
        
        # 平移到地面坐标系
        Y_w = self.camera_height - Y_w
        
        return WorldPoint(x=X_w, y=Y_w, z=Z_w)
    
    def world_to_topview(
        self,
        world_point: WorldPoint
    ) -> TopViewPoint:
        """
        世界坐标→顶视图坐标
        
        顶视图使用 X-Z 平面（忽略 Y 高度）
        
        Args:
            world_point: 世界坐标点
            
        Returns:
            TopViewPoint: 顶视图坐标点
        """
        # 世界坐标 (米) → 画布像素
        x_top = self.topview_origin[0] + world_point.x * self.topview_scale
        y_top = self.topview_origin[1] - world_point.z * self.topview_scale
        
        return TopViewPoint(x=x_top, y=y_top)
    
    def pixel_to_topview(
        self,
        u: float,
        v: float,
        distance: float
    ) -> TopViewPoint:
        """
        像素坐标→顶视图坐标（完整流程）
        
        Args:
            u: 像素 x 坐标
            v: 像素 y 坐标
            distance: 到摄像头的距离 (米)
            
        Returns:
            TopViewPoint: 顶视图坐标点
        """
        # 像素→相机坐标
        X_c, Y_c, Z_c = self.pixel_to_camera_coords(u, v, distance)
        
        # 相机坐标→世界坐标
        world_point = self.camera_to_world_coords(X_c, Y_c, Z_c)
        
        # 世界坐标→顶视图
        topview_point = self.world_to_topview(world_point)
        
        return topview_point
    
    def calc_person_metrics(
        self,
        person: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        计算人体各项指标
        
        Args:
            person: 人体检测结果
            
        Returns:
            包含各项指标的字典
        """
        bbox = person['bbox']
        keypoints = person.get('keypoints', {})
        
        # 计算距离
        distance = self.calc_person_distance(bbox)
        
        # 计算身高
        height = 0.0
        if 'head' in keypoints and 'ankle_left' in keypoints:
            height = self.calc_height(
                keypoints['head'],
                keypoints['ankle_left'],
                distance
            )
        
        # 计算顶视图坐标（使用人体中心）
        x_center = bbox[0] + bbox[2] // 2
        y_center = bbox[1] + bbox[3] // 2
        topview = self.pixel_to_topview(x_center, y_center, distance)
        
        return {
            "distance": round(distance, 2),
            "height": round(height, 1),
            "topview": {
                "x": round(topview.x, 1),
                "y": round(topview.y, 1)
            },
            "bbox": bbox,
            "keypoints": keypoints
        }
    
    def calc_hand_metrics(
        self,
        hand: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        计算手部各项指标
        
        Args:
            hand: 手部检测结果
            
        Returns:
            包含各项指标的字典
        """
        keypoints = hand.get('keypoints', [])
        bbox = hand['bbox']
        
        if len(keypoints) < 21:
            return {}
        
        # 掌根 (0 号) 到中指指尖 (12 号) 的距离
        palm = keypoints[HandKeypoints.WRIST]
        middle_finger = keypoints[HandKeypoints.MIDDLE_TIP]
        
        # 估算手到摄像头的距离（使用边界框大小）
        hand_width = bbox[2]
        ref_hand_width = 0.08  # 平均手宽 8cm
        distance = (ref_hand_width * self.fx) / hand_width if hand_width > 0 else 1.0
        
        # 计算手大小
        hand_size = self.calc_hand_size(palm, middle_finger, distance)
        
        # 计算顶视图坐标（使用手掌中心）
        palm_center = [
            bbox[0] + bbox[2] // 2,
            bbox[1] + bbox[3] // 2
        ]
        topview = self.pixel_to_topview(palm_center[0], palm_center[1], distance)
        
        return {
            "size": round(hand_size, 1),
            "distance": round(distance, 2),
            "topview": {
                "x": round(topview.x, 1),
                "y": round(topview.y, 1)
            },
            "keypoints": keypoints
        }


class HandKeypoints:
    """手部关键点索引"""
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20


def create_sample_calib_params() -> CalibrationParams:
    """创建示例标定参数（用于测试）"""
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


def main():
    """测试空间计算器"""
    # 创建示例标定参数
    calib_params = create_sample_calib_params()
    
    # 初始化计算器
    calculator = SpatialCalculator(calib_params)
    calculator.set_camera_extrinsics(height=1.8, pitch_angle=30.0)
    
    # 测试：像素→世界坐标
    u, v = 960, 540  # 图像中心
    distance = 3.0   # 3 米远
    
    X_c, Y_c, Z_c = calculator.pixel_to_camera_coords(u, v, distance)
    print(f"Camera coords: ({X_c:.2f}, {Y_c:.2f}, {Z_c:.2f})m")
    
    world_point = calculator.camera_to_world_coords(X_c, Y_c, Z_c)
    print(f"World coords: ({world_point.x:.2f}, {world_point.y:.2f}, {world_point.z:.2f})m")
    
    topview = calculator.world_to_topview(world_point)
    print(f"TopView coords: ({topview.x:.1f}, {topview.y:.1f})px")
    
    # 测试：身高计算
    head = [960, 200]
    ankle = [960, 800]
    height = calculator.calc_height(head, ankle, distance)
    print(f"Estimated height: {height:.1f}cm")


if __name__ == '__main__':
    main()
