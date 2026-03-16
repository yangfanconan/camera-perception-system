#!/usr/bin/env python3
"""
交互式相机标定工具
功能：
1. 实时摄像头画面显示
2. 自动检测棋盘格并提示
3. 采集足够图片后自动执行标定
4. 显示标定结果和重投影误差
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
from loguru import logger

from src.algorithms.calibration import CameraCalibrator, CalibrationParams


class InteractiveCalibrator:
    """交互式标定器"""
    
    def __init__(
        self,
        camera_id: int = 0,
        checkerboard_size: Tuple[int, int] = (9, 6),
        square_size: float = 25.0,
        min_images: int = 15
    ):
        """
        初始化交互式标定器
        
        Args:
            camera_id: 摄像头设备 ID
            checkerboard_size: 棋盘格角点数
            square_size: 棋盘格边长 (mm)
            min_images: 最少需要的图片数量
        """
        self.camera_id = camera_id
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.min_images = min_images
        
        self.calibrator = CameraCalibrator(
            checkerboard_size=checkerboard_size,
            square_size=square_size
        )
        
        self.captured_images: List[np.ndarray] = []
        self.captured_corners: List[np.ndarray] = []
        
        # UI 配置
        self.ui_colors = {
            'found': (0, 255, 0),      # 绿色 - 检测到棋盘格
            'not_found': (0, 0, 255),  # 红色 - 未检测到
            'saved': (255, 0, 255)     # 品红 - 已保存
        }
        
        logger.info(f"InteractiveCalibrator initialized: camera={camera_id}, "
                   f"checkerboard={checkerboard_size}, min_images={min_images}")
    
    def capture_and_calibrate(self, output_path: str = 'calibration_data/calib_params.json'):
        """
        执行交互式标定流程
        
        Args:
            output_path: 标定参数输出路径
        """
        # 打开摄像头
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        if not cap.isOpened():
            logger.error(f"无法打开摄像头 {self.camera_id}")
            return False
        
        logger.info("摄像头已打开，开始标定...")
        logger.info(f"需要采集至少 {self.min_images} 张有效图片")
        logger.info("操作说明:")
        logger.info("  - 将棋盘格放在摄像头前不同角度/位置")
        logger.info("  - 检测到棋盘格时自动采集（绿色框）")
        logger.info("  - 按 'r' 手动重新采集当前帧")
        logger.info("  - 按 's' 跳过自动采集，手动保存")
        logger.info("  - 按 'q' 退出不保存")
        logger.info("  - 采集完成后自动执行标定")
        
        last_capture_time = 0
        capture_interval = 2.0  # 自动采集间隔（秒）
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = datetime.now().timestamp()
                
                # 检测棋盘格
                found, corners = self.calibrator.find_checkerboard_corners(frame)
                
                # 绘制检测结果
                display_frame = frame.copy()
                
                if found:
                    # 绘制角点
                    cv2.drawChessboardCorners(
                        display_frame, 
                        self.checkerboard_size, 
                        corners, 
                        found
                    )
                    
                    # 自动采集（间隔 capture_interval 秒）
                    if current_time - last_capture_time > capture_interval:
                        if len(self.captured_images) < self.min_images:
                            self.captured_images.append(frame.copy())
                            self.captured_corners.append(corners.copy())
                            last_capture_time = current_time
                            logger.info(f"已采集 {len(self.captured_images)}/{self.min_images} 张")
                
                # 显示 UI 信息
                self._draw_ui(display_frame, found)
                
                cv2.imshow('Interactive Calibration', display_frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    logger.warning("用户取消标定")
                    break
                elif key == ord('s'):
                    # 手动保存当前帧
                    if found:
                        self.captured_images.append(frame.copy())
                        self.captured_corners.append(corners.copy())
                        last_capture_time = current_time
                        logger.info(f"手动保存：{len(self.captured_images)} 张")
                elif key == ord('r'):
                    # 重新采集
                    if found:
                        self.captured_images = []
                        self.captured_corners = []
                        logger.info("已清空，重新采集")
                
                # 检查是否采集足够
                if len(self.captured_images) >= self.min_images:
                    logger.info(f"已采集足够的图片 ({len(self.captured_images)} 张)")
                    break
            
            # 执行标定
            if len(self.captured_images) >= 3:
                logger.info("开始执行标定...")
                
                # 使用采集的数据直接标定
                objpoints = []
                for _ in self.captured_corners:
                    objpoints.append(self.calibrator.prepare_object_points())
                
                # 计算
                gray_size = self.captured_images[0].shape[:2]
                
                ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints, 
                    self.captured_corners, 
                    gray_size[::-1], 
                    None, None,
                    flags=cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST
                )
                
                # 计算重投影误差
                mean_error = self.calibrator._calc_reprojection_error(
                    objpoints, self.captured_corners, rvecs, tvecs, 
                    camera_matrix, dist_coeffs
                )
                
                # 构建标定参数
                calib_params = CalibrationParams(
                    fx=float(camera_matrix[0, 0]),
                    fy=float(camera_matrix[1, 1]),
                    cx=float(camera_matrix[0, 2]),
                    cy=float(camera_matrix[1, 2]),
                    dist_coeffs=dist_coeffs.flatten().tolist(),
                    rotation_matrix=rvecs[0].flatten().tolist() if rvecs else [0, 0, 0],
                    translation_vector=tvecs[0].flatten().tolist() if tvecs else [0, 0, 0],
                    image_size=(gray_size[1], gray_size[0]),
                    checkerboard_size=self.checkerboard_size,
                    square_size=self.square_size,
                    reprojection_error=mean_error,
                    num_images=len(self.captured_images)
                )
                
                # 显示标定结果
                self._show_calibration_result(calib_params)
                
                # 保存
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(calib_params.to_dict(), f, indent=2)
                
                logger.success(f"标定完成！参数已保存到 {output_path}")
                logger.info(f"fx={calib_params.fx:.2f}, fy={calib_params.fy:.2f}, "
                           f"cx={calib_params.cx:.2f}, cy={calib_params.cy:.2f}")
                logger.info(f"重投影误差：{mean_error:.4f} 像素")
                
                return True
            else:
                logger.error("采集图片数量不足，标定失败")
                return False
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def _draw_ui(self, frame: np.ndarray, found: bool):
        """绘制 UI 信息"""
        h, w = frame.shape[:2]
        
        # 标题
        cv2.putText(frame, "Interactive Camera Calibration", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 采集进度
        progress = f"Collected: {len(self.captured_images)}/{self.min_images}"
        cv2.putText(frame, progress, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 状态
        status = "Found" if found else "Not Found"
        color = self.ui_colors['found'] if found else self.ui_colors['not_found']
        cv2.putText(frame, f"Status: {status}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 提示信息
        if len(self.captured_images) < self.min_images:
            hint = f"Need {self.min_images - len(self.captured_images)} more images"
            cv2.putText(frame, hint, (w - 350, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            hint = "Ready! Press any key to calibrate..."
            cv2.putText(frame, hint, (w - 350, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 进度条
        bar_width = int(200 * len(self.captured_images) / self.min_images)
        cv2.rectangle(frame, (w - 220, 50), (w - 20, 70), (50, 50, 50), -1)
        cv2.rectangle(frame, (w - 220, 50), (w - 220 + bar_width, 70), (0, 255, 0), -1)
    
    def _show_calibration_result(self, params: CalibrationParams):
        """显示标定结果窗口"""
        result = np.zeros((600, 800, 3), dtype=np.uint8)
        
        y_offset = 50
        x_offset = 50
        
        # 标题
        cv2.putText(result, "Calibration Result", (x_offset, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        y_offset += 50
        
        # 参数
        params_text = [
            f"Images: {params.num_images}",
            f"RMS Error: {params.reprojection_error:.4f} px",
            f"",
            f"fx = {params.fx:.2f}",
            fy = {params.fy:.2f}",
            f"cx = {params.cx:.2f}",
            f"cy = {params.cy:.2f}",
            f"",
            f"Distortion:",
            f"  k1 = {params.dist_coeffs[0]:.6f}",
            f"  k2 = {params.dist_coeffs[1]:.6f}",
            f"  p1 = {params.dist_coeffs[2]:.6f}",
            f"  p2 = {params.dist_coeffs[3]:.6f}",
        ]
        
        for line in params_text:
            cv2.putText(result, line, (x_offset, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y_offset += 30
        
        cv2.imshow('Calibration Result', result)
        cv2.waitKey(3000)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='交互式相机标定工具')
    parser.add_argument('--camera', '-c', type=int, default=0,
                       help='摄像头设备 ID')
    parser.add_argument('--output', '-o', type=str, 
                       default='calibration_data/calib_params.json',
                       help='输出标定参数文件路径')
    parser.add_argument('--checkerboard', '-cb', type=int, nargs=2, 
                       default=[9, 6],
                       help='棋盘格角点数 [cols, rows]')
    parser.add_argument('--square-size', '-s', type=float, default=25.0,
                       help='棋盘格边长 (mm)')
    parser.add_argument('--min-images', '-n', type=int, default=15,
                       help='最少标定图片数量')
    
    args = parser.parse_args()
    
    # 创建标定器并执行
    calibrator = InteractiveCalibrator(
        camera_id=args.camera,
        checkerboard_size=tuple(args.checkerboard),
        square_size=args.square_size,
        min_images=args.min_images
    )
    
    success = calibrator.capture_and_calibrate(output_path=args.output)
    
    if success:
        print("\n✅ 标定成功!")
    else:
        print("\n❌ 标定失败!")


if __name__ == '__main__':
    main()
