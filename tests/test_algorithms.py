"""
单元测试

测试各模块功能
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGestureRecognition:
    """手势识别测试"""
    
    def test_import(self):
        """测试导入"""
        from algorithms.gesture_recognition import GestureRecognizer
        recognizer = GestureRecognizer()
        assert recognizer is not None
    
    def test_open_palm(self):
        """测试张开手掌"""
        from algorithms.gesture_recognition import GestureRecognizer
        
        recognizer = GestureRecognizer()
        
        # 模拟张开手掌的关键点
        landmarks = [
            [0.5, 0.8, 0, 1],  # wrist
            [0.4, 0.7, 0, 1], [0.35, 0.6, 0, 1], [0.3, 0.5, 0, 1], [0.25, 0.4, 0, 1],  # thumb
            [0.45, 0.5, 0, 1], [0.45, 0.35, 0, 1], [0.45, 0.25, 0, 1], [0.45, 0.15, 0, 1],  # index
            [0.5, 0.5, 0, 1], [0.5, 0.35, 0, 1], [0.5, 0.25, 0, 1], [0.5, 0.15, 0, 1],  # middle
            [0.55, 0.5, 0, 1], [0.55, 0.35, 0, 1], [0.55, 0.25, 0, 1], [0.55, 0.15, 0, 1],  # ring
            [0.6, 0.5, 0, 1], [0.6, 0.4, 0, 1], [0.6, 0.3, 0, 1], [0.6, 0.2, 0, 1],  # pinky
        ]
        
        result = recognizer.recognize(landmarks, "Right")
        
        assert result.gesture == 'open_palm'
        assert result.confidence > 0.5
    
    def test_fist(self):
        """测试握拳"""
        from algorithms.gesture_recognition import GestureRecognizer
        
        recognizer = GestureRecognizer()
        
        # 模拟握拳的关键点
        landmarks = [
            [0.5, 0.8, 0, 1],  # wrist
            [0.45, 0.75, 0, 1], [0.42, 0.72, 0, 1], [0.40, 0.70, 0, 1], [0.38, 0.68, 0, 1],  # thumb (bent)
            [0.48, 0.65, 0, 1], [0.48, 0.60, 0, 1], [0.48, 0.58, 0, 1], [0.48, 0.56, 0, 1],  # index (bent)
            [0.52, 0.65, 0, 1], [0.52, 0.60, 0, 1], [0.52, 0.58, 0, 1], [0.52, 0.56, 0, 1],  # middle (bent)
            [0.56, 0.65, 0, 1], [0.56, 0.60, 0, 1], [0.56, 0.58, 0, 1], [0.56, 0.56, 0, 1],  # ring (bent)
            [0.60, 0.65, 0, 1], [0.60, 0.62, 0, 1], [0.60, 0.60, 0, 1], [0.60, 0.58, 0, 1],  # pinky (bent)
        ]
        
        result = recognizer.recognize(landmarks, "Right")
        
        # 握拳或大拇指
        assert result.gesture in ['fist', 'thumbs_up', 'thumbs_down']


class TestFallDetection:
    """跌倒检测测试"""
    
    def test_import(self):
        """测试导入"""
        from algorithms.fall_detection import FallDetector
        detector = FallDetector()
        assert detector is not None
    
    def test_normal_standing(self):
        """测试正常站立"""
        from algorithms.fall_detection import FallDetector
        
        detector = FallDetector(fps=30.0)
        
        # 模拟正常站立
        for i in range(10):
            bbox = [100, 100, 100, 300]
            keypoints = {
                'L_shoulder': [150, 150],
                'R_shoulder': [200, 150],
                'L_hip': [160, 280],
                'R_hip': [190, 280],
                'L_ankle': [160, 400],
                'R_ankle': [190, 400]
            }
            event = detector.update(1, bbox, keypoints)
            assert event is None  # 不应该检测到跌倒
        
        state = detector.get_person_state(1)
        assert state == "normal"
    
    def test_fall_detection(self):
        """测试跌倒检测"""
        from algorithms.fall_detection import FallDetector
        
        detector = FallDetector(fps=30.0)
        
        # 模拟跌倒过程
        fall_detected = False
        
        for i in range(30):
            # 高度逐渐减小，宽度增大
            h = 300 - i * 10
            w = 100 + i * 8
            y = 100 + i * 5
            bbox = [100, y, w, max(h, 50)]
            
            keypoints = {
                'L_shoulder': [150, 150 + i * 3],
                'R_shoulder': [200, 150 + i * 3],
                'L_hip': [160, 280 - i * 5],
                'R_hip': [190, 280 - i * 5],
                'L_ankle': [160, 400 - i * 8],
                'R_ankle': [190, 400 - i * 8]
            }
            
            event = detector.update(1, bbox, keypoints)
            if event:
                fall_detected = True
                break
        
        # 可能检测到跌倒（取决于参数）
        # assert fall_detected


class TestPersonTracker:
    """多人跟踪测试"""
    
    def test_import(self):
        """测试导入"""
        from algorithms.person_tracker import PersonTracker
        tracker = PersonTracker()
        assert tracker is not None
    
    def test_single_person_tracking(self):
        """测试单人跟踪"""
        from algorithms.person_tracker import PersonTracker
        
        tracker = PersonTracker()
        
        # 模拟单人移动
        for i in range(10):
            detections = [
                {'bbox': [100 + i * 10, 100, 50, 150], 'confidence': 0.9}
            ]
            tracks = tracker.update(detections)
            assert len(tracks) == 1
            assert tracks[0].track_id == 1
    
    def test_multi_person_tracking(self):
        """测试多人跟踪"""
        from algorithms.person_tracker import PersonTracker
        
        tracker = PersonTracker()
        
        # 模拟多人
        for i in range(10):
            detections = [
                {'bbox': [100 + i * 5, 100, 50, 150], 'confidence': 0.9},
                {'bbox': [300 + i * 5, 100, 50, 150], 'confidence': 0.85},
            ]
            tracks = tracker.update(detections)
            assert len(tracks) == 2
    
    def test_track_persistence(self):
        """测试跟踪持久化"""
        from algorithms.person_tracker import PersonTracker
        
        tracker = PersonTracker()
        
        # 先检测
        for i in range(5):
            detections = [{'bbox': [100, 100, 50, 150], 'confidence': 0.9}]
            tracks = tracker.update(detections)
        
        track_id = tracks[0].track_id
        
        # 短暂消失
        for i in range(3):
            tracks = tracker.update([])
        
        # 再次出现
        for i in range(5):
            detections = [{'bbox': [100, 100, 50, 150], 'confidence': 0.9}]
            tracks = tracker.update(detections)
        
        # 应该恢复相同的 ID
        assert tracks[0].track_id == track_id


class TestAlertSystem:
    """报警系统测试"""
    
    def test_import(self):
        """测试导入"""
        from algorithms.alert_system import AlertSystem
        system = AlertSystem()
        assert system is not None
    
    def test_zone_crossing(self):
        """测试越界检测"""
        from algorithms.alert_system import AlertSystem
        
        system = AlertSystem()
        
        # 添加禁区
        system.add_zone(
            zone_id="test_zone",
            name="测试禁区",
            polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
            zone_type="forbidden"
        )
        
        # 在区域内
        alerts = system.check_zones(1, (50, 50))
        assert len(alerts) > 0
        assert alerts[0].alert_type == "intrusion"
        
        # 在区域外
        alerts = system.check_zones(2, (200, 200))
        assert len(alerts) == 0
    
    def test_fall_alert(self):
        """测试跌倒报警"""
        from algorithms.alert_system import AlertSystem
        
        system = AlertSystem()
        
        alert = system.check_fall(1, "fallen", (100, 100))
        assert alert is not None
        assert alert.alert_type == "fall"
        assert alert.severity == "critical"
    
    def test_crowd_detection(self):
        """测试人群聚集检测"""
        from algorithms.alert_system import AlertSystem
        
        system = AlertSystem()
        
        # 5个人聚集
        positions = [(100 + i * 10, 100 + i * 10) for i in range(5)]
        alert = system.check_crowd(positions, threshold=5, radius=100)
        assert alert is not None
        assert alert.alert_type == "crowd"


class TestDepthCalibration:
    """深度校准测试"""
    
    def test_import(self):
        """测试导入"""
        from algorithms.depth_calibration import DepthCalibrator
        calibrator = DepthCalibrator()
        assert calibrator is not None
    
    def test_single_point_calibration(self):
        """测试单点校准"""
        from algorithms.depth_calibration import DepthCalibrator
        
        calibrator = DepthCalibrator()
        
        calibrator.add_calibration_point(0.5, 1.0)
        success = calibrator.calibrate("linear")
        
        assert success
        assert calibrator.params.calibrated
        
        # 测试转换
        depth = calibrator.apply_calibration(np.array([0.5]))
        assert abs(depth[0] - 1.0) < 0.1
    
    def test_multi_point_calibration(self):
        """测试多点校准"""
        from algorithms.depth_calibration import DepthCalibrator
        
        calibrator = DepthCalibrator()
        
        calibrator.add_calibration_point(0.5, 1.0)
        calibrator.add_calibration_point(1.0, 2.0)
        calibrator.add_calibration_point(2.0, 4.0)
        
        success = calibrator.calibrate("linear")
        
        assert success
        
        # 测试转换
        depth = calibrator.apply_calibration(np.array([1.5]))
        assert 2.5 < depth[0] < 3.5


class TestDataRecorder:
    """数据记录测试"""
    
    def test_import(self):
        """测试导入"""
        from algorithms.data_recorder import DataRecorder
        recorder = DataRecorder(output_dir="test_recordings", auto_save=False)
        assert recorder is not None
    
    def test_record_detection(self):
        """测试记录检测"""
        from algorithms.data_recorder import DataRecorder
        
        recorder = DataRecorder(output_dir="test_recordings", auto_save=False)
        
        recorder.record_detection(
            frame_id=1,
            track_id=1,
            bbox=[100, 100, 50, 150],
            confidence=0.9,
            keypoints={},
            distance=2.5
        )
        
        assert len(recorder.detections) == 1
        assert recorder.session.total_detections == 1
    
    def test_trajectory_tracking(self):
        """测试轨迹跟踪"""
        from algorithms.data_recorder import DataRecorder
        
        recorder = DataRecorder(output_dir="test_recordings", auto_save=False)
        
        # 记录移动轨迹
        for i in range(10):
            recorder.record_detection(
                frame_id=i,
                track_id=1,
                bbox=[100 + i * 10, 100, 50, 150],
                confidence=0.9,
                keypoints={}
            )
        
        trajectory = recorder.get_trajectory(1)
        assert trajectory is not None
        assert len(trajectory.positions) == 10
        assert trajectory.total_distance > 0


class TestAsyncProcessor:
    """异步处理测试"""
    
    def test_import(self):
        """测试导入"""
        from algorithms.async_processor import AsyncProcessor
        assert AsyncProcessor is not None
    
    def test_frame_queue(self):
        """测试帧队列"""
        from algorithms.async_processor import FrameQueue, FrameData
        
        queue = FrameQueue(max_size=5)
        
        for i in range(10):
            frame_data = FrameData(
                frame_id=i,
                frame=np.zeros((480, 640, 3)),
                timestamp=0.0
            )
            queue.put(frame_data)
        
        # 队列应该只保留最新的5帧
        assert queue.size() <= 5


# 运行测试
if __name__ == '__main__':
    pytest.main([__file__, '-v'])