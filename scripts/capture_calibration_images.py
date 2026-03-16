#!/usr/bin/env python3
"""
相机标定图片采集脚本
功能：打开摄像头，按空格键保存当前帧用于标定
"""

import cv2
import argparse
from pathlib import Path
from datetime import datetime
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description='相机标定图片采集')
    parser.add_argument('--camera', '-c', type=int, default=0,
                       help='摄像头设备 ID')
    parser.add_argument('--output', '-o', type=str, 
                       default='calibration_data/images',
                       help='输出目录')
    parser.add_argument('--resolution', '-r', type=int, nargs=2,
                       default=[1920, 1080],
                       help='分辨率 [width, height]')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 打开摄像头
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.resolution[1])
    
    if not cap.isOpened():
        logger.error(f"无法打开摄像头 {args.camera}")
        return
    
    logger.info(f"摄像头已打开，分辨率：{args.resolution}")
    logger.info("操作说明：")
    logger.info("  - 空格键：保存当前帧")
    logger.info("  - 'q' 键：退出")
    
    image_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 显示
            cv2.putText(frame, f"Images: {image_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Space: Save | q: Quit",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Calibration Capture', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                # 保存图像
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = output_dir / f"calib_{timestamp}_{image_count:03d}.jpg"
                cv2.imwrite(str(filename), frame)
                logger.info(f"已保存：{filename}")
                image_count += 1
            elif key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info(f"共保存 {image_count} 张图像")


if __name__ == '__main__':
    main()
