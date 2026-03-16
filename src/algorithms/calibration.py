"""
相机标定模块 - 张正友棋盘格标定法
功能：
1. 从棋盘格图片中提取角点
2. 计算相机内参、畸变系数、外参
3. 保存/加载标定参数
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, asdict
from loguru import logger


@dataclass
class CalibrationParams:
    """标定参数数据类"""
    # 相机内参
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    
    # 畸变系数 [k1, k2, p1, p2, k3]
    dist_coeffs: List[float] = None
    
    # 外参
    rotation_matrix: List[List[float]] = None  # 3x3
    translation_vector: List[float] = None     # 3x1
    
    # 标定信息
    image_size: Tuple[int, int] = None  # [width, height]
    checkerboard_size: Tuple[int, int] = None  # [cols, rows]
    square_size: float = 25.0  # mm
    reprojection_error: float = 0.0
    num_images: int = 0
    
    def __post_init__(self):
        if self.dist_coeffs is None:
            self.dist_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
        if self.rotation_matrix is None:
            self.rotation_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        if self.translation_vector is None:
            self.translation_vector = [0.0, 0.0, 0.0]
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CalibrationParams':
        """从字典创建"""
        return cls(**data)
    
    def get_camera_matrix(self) -> np.ndarray:
        """获取相机内参矩阵"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)
    
    def get_dist_coeffs(self) -> np.ndarray:
        """获取畸变系数向量"""
        return np.array(self.dist_coeffs, dtype=np.float64)
    
    def get_rotation_matrix(self) -> np.ndarray:
        """获取旋转矩阵"""
        return np.array(self.rotation_matrix, dtype=np.float64)
    
    def get_translation_vector(self) -> np.ndarray:
        """获取平移向量"""
        return np.array(self.translation_vector, dtype=np.float64)


class CameraCalibrator:
    """相机标定器"""
    
    def __init__(self, checkerboard_size: Tuple[int, int] = (9, 6), 
                 square_size: float = 25.0):
        """
        初始化标定器
        
        Args:
            checkerboard_size: 棋盘格角点数量 [cols, rows]
            square_size: 棋盘格边长 (mm)
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        
        # 角点检测子像素搜索参数
        self.subpix_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,  # 最大迭代次数
            0.001  # 精度阈值
        )
        
        # 标定参数
        self.calib_params: Optional[CalibrationParams] = None
        
        logger.info(f"CameraCalibrator initialized with checkerboard {checkerboard_size}")
    
    def find_checkerboard_corners(
        self, 
        image: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        检测棋盘格角点
        
        Args:
            image: 灰度图像
            
        Returns:
            (success, corners): 是否成功检测，角点坐标
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 检测角点
        found, corners = cv2.findChessboardCorners(
            gray, 
            self.checkerboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if found:
            # 亚像素级精化
            corners = cv2.cornerSubPix(
                gray, corners, 
                winSize=(11, 11), 
                zeroZone=(-1, -1),
                criteria=self.subpix_criteria
            )
        
        return found, corners
    
    def prepare_object_points(self) -> np.ndarray:
        """准备物体坐标系中的 3D 点"""
        objp = np.zeros(
            (self.checkerboard_size[0] * self.checkerboard_size[1], 3), 
            np.float32
        )
        objp[:, :2] = np.mgrid[
            0:self.checkerboard_size[0], 
            0:self.checkerboard_size[1]
        ].T.reshape(-1, 2)
        objp *= self.square_size / 1000.0  # 转换为米
        return objp
    
    def calibrate(
        self, 
        image_paths: List[str],
        visualize: bool = True
    ) -> CalibrationParams:
        """
        执行相机标定
        
        Args:
            image_paths: 棋盘格图片路径列表
            visualize: 是否可视化检测结果
            
        Returns:
            CalibrationParams: 标定参数
        """
        objpoints = []  # 3D 物体点
        imgpoints = []  # 2D 图像点
        valid_images = 0
        
        logger.info(f"Starting calibration with {len(image_paths)} images...")
        
        for idx, img_path in enumerate(image_paths):
            image = cv2.imread(img_path)
            if image is None:
                logger.warning(f"Failed to load image: {img_path}")
                continue
            
            found, corners = self.find_checkerboard_corners(image)
            
            if found:
                objpoints.append(self.prepare_object_points())
                imgpoints.append(corners)
                valid_images += 1
                
                if visualize:
                    cv2.drawChessboardCorners(
                        image, self.checkerboard_size, corners, found
                    )
                    cv2.imshow(f'Calibration {idx}', image)
                    cv2.waitKey(100)
            else:
                logger.warning(f"No checkerboard found in: {img_path}")
        
        if valid_images < 3:
            raise ValueError("Need at least 3 valid images for calibration")
        
        logger.info(f"Found {valid_images} valid images")
        
        # 执行标定
        gray_size = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE).shape[:2]
        img_size = (gray_size[1], gray_size[0])  # (width, height)
        
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray_size[::-1], None, None,
            flags=cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST
        )
        
        logger.info(f"Calibration RMS error: {ret:.6f}")
        
        # 计算重投影误差
        mean_error = self._calc_reprojection_error(
            objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs
        )
        logger.info(f"Mean reprojection error: {mean_error:.6f} pixels")
        
        # 构建标定参数
        self.calib_params = CalibrationParams(
            fx=float(camera_matrix[0, 0]),
            fy=float(camera_matrix[1, 1]),
            cx=float(camera_matrix[0, 2]),
            cy=float(camera_matrix[1, 2]),
            dist_coeffs=dist_coeffs.flatten().tolist(),
            rotation_matrix=rvecs[0].flatten().tolist() if rvecs else [0, 0, 0],
            translation_vector=tvecs[0].flatten().tolist() if tvecs else [0, 0, 0],
            image_size=img_size,
            checkerboard_size=self.checkerboard_size,
            square_size=self.square_size,
            reprojection_error=mean_error,
            num_images=valid_images
        )
        
        if visualize:
            cv2.destroyAllWindows()
        
        return self.calib_params
    
    def _calc_reprojection_error(
        self, 
        objpoints, imgpoints, rvecs, tvecs, 
        camera_matrix, dist_coeffs
    ) -> float:
        """计算重投影误差"""
        total_error = 0
        num_points = 0
        
        for i, objpts in enumerate(objpoints):
            imgpts_est, _ = cv2.projectPoints(
                objpts, rvecs[i], tvecs[i], camera_matrix, dist_coeffs
            )
            error = cv2.norm(imgpoints[i], imgpts_est, cv2.NORM_L2) / len(imgpts_est)
            total_error += error
            num_points += 1
        
        return total_error / num_points if num_points > 0 else 0.0
    
    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """
        去畸变图像
        
        Args:
            image: 输入图像
            
        Returns:
            去畸变后的图像
        """
        if self.calib_params is None:
            raise ValueError("Camera not calibrated yet")
        
        camera_matrix = self.calib_params.get_camera_matrix()
        dist_coeffs = self.calib_params.get_dist_coeffs()
        
        return cv2.undistort(image, camera_matrix, dist_coeffs)
    
    def save_params(self, filepath: str) -> None:
        """保存标定参数到文件"""
        if self.calib_params is None:
            raise ValueError("No calibration parameters to save")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.calib_params.to_json())
        
        logger.info(f"Calibration parameters saved to {filepath}")
    
    def load_params(self, filepath: str) -> CalibrationParams:
        """从文件加载标定参数"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.calib_params = CalibrationParams.from_dict(data)
        logger.info(f"Calibration parameters loaded from {filepath}")
        return self.calib_params


def main():
    """命令行标定工具"""
    import argparse
    
    parser = argparse.ArgumentParser(description='相机标定工具')
    parser.add_argument(
        '--images', '-i', 
        type=str, 
        required=True,
        help='棋盘格图片目录'
    )
    parser.add_argument(
        '--output', '-o', 
        type=str, 
        default='calibration_data/calib_params.json',
        help='输出标定参数文件路径'
    )
    parser.add_argument(
        '--checkerboard', '-c', 
        type=int, 
        nargs=2, 
        default=[9, 6],
        help='棋盘格角点数 [cols, rows]'
    )
    parser.add_argument(
        '--square-size', '-s', 
        type=float, 
        default=25.0,
        help='棋盘格边长 (mm)'
    )
    
    args = parser.parse_args()
    
    # 收集图片
    image_dir = Path(args.images)
    image_paths = sorted([
        str(p) for p in image_dir.glob('*.jpg')
    ] + [
        str(p) for p in image_dir.glob('*.png')
    ])
    
    if not image_paths:
        logger.error(f"No images found in {image_dir}")
        return
    
    # 执行标定
    calibrator = CameraCalibrator(
        checkerboard_size=tuple(args.checkerboard),
        square_size=args.square_size
    )
    
    try:
        params = calibrator.calibrate(image_paths, visualize=True)
        calibrator.save_params(args.output)
        logger.success(f"Calibration completed! Parameters saved to {args.output}")
        logger.info(f"fx={params.fx:.2f}, fy={params.fy:.2f}, "
                   f"cx={params.cx:.2f}, cy={params.cy:.2f}")
    except Exception as e:
        logger.error(f"Calibration failed: {e}")


if __name__ == '__main__':
    main()
