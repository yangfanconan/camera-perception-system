"""
系统常量定义
所有魔法数字和硬编码值都应该定义在这里
"""

# ==================== 图像处理常量 ====================

# 默认图像尺寸（用于回退）
DEFAULT_IMAGE_WIDTH = 1920
DEFAULT_IMAGE_HEIGHT = 1080
DEFAULT_IMAGE_SIZE = (DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT)

# 常见摄像头分辨率
COMMON_RESOLUTIONS = {
    '480p': (640, 480),
    '720p': (1280, 720),
    '1080p': (1920, 1080),
    '2k': (2560, 1440),
    '4k': (3840, 2160),
}

# YOLO 模型输入尺寸
YOLO_INPUT_SIZE = 640

# ==================== 检测常量 ====================

# 关键点检测阈值
KEYPOINT_CONFIDENCE_THRESHOLD = 0.3  # 近距离时降低阈值
KEYPOINT_CONFIDENCE_THRESHOLD_NORMAL = 0.5

# 人体关键点索引（COCO格式）
KEYPOINT_NOSE = 0
KEYPOINT_LEFT_EYE = 1
KEYPOINT_RIGHT_EYE = 2
KEYPOINT_LEFT_EAR = 3
KEYPOINT_RIGHT_EAR = 4
KEYPOINT_LEFT_SHOULDER = 5
KEYPOINT_RIGHT_SHOULDER = 6
KEYPOINT_LEFT_ELBOW = 7
KEYPOINT_RIGHT_ELBOW = 8
KEYPOINT_LEFT_WRIST = 9
KEYPOINT_RIGHT_WRIST = 10
KEYPOINT_LEFT_HIP = 11
KEYPOINT_RIGHT_HIP = 12
KEYPOINT_LEFT_KNEE = 13
KEYPOINT_RIGHT_KNEE = 14
KEYPOINT_LEFT_ANKLE = 15
KEYPOINT_RIGHT_ANKLE = 16

# 关键点组定义
HEAD_KEYPOINTS = [KEYPOINT_NOSE, KEYPOINT_LEFT_EYE, KEYPOINT_RIGHT_EYE, 
                  KEYPOINT_LEFT_EAR, KEYPOINT_RIGHT_EAR]
UPPER_BODY_KEYPOINTS = [KEYPOINT_LEFT_SHOULDER, KEYPOINT_RIGHT_SHOULDER,
                        KEYPOINT_LEFT_ELBOW, KEYPOINT_RIGHT_ELBOW,
                        KEYPOINT_LEFT_WRIST, KEYPOINT_RIGHT_WRIST]
LOWER_BODY_KEYPOINTS = [KEYPOINT_LEFT_HIP, KEYPOINT_RIGHT_HIP,
                        KEYPOINT_LEFT_KNEE, KEYPOINT_RIGHT_KNEE,
                        KEYPOINT_LEFT_ANKLE, KEYPOINT_RIGHT_ANKLE]

# ==================== 身体部位检测常量 ====================

# 身体部位类型
BODY_PART_FULL = 'full_body'
BODY_PART_HALF = 'half_body'
BODY_PART_UPPER = 'upper_body'
BODY_PART_LOWER = 'lower_body'
BODY_PART_HEAD_ONLY = 'head_only'
BODY_PART_UNKNOWN = 'unknown'

# 身体部位判断阈值
BODY_PART_MIN_KEYPOINTS = 2  # 判断某部位存在的最小关键点数
BODY_PART_BBOX_RATIO_THRESHOLD_LOW = 0.15
BODY_PART_BBOX_RATIO_THRESHOLD_MID = 0.30
BODY_PART_BBOX_RATIO_THRESHOLD_HIGH = 0.50

# 极近距离判断阈值
ULTRA_CLOSE_BBOX_RATIO = 0.70  # 画面占比超过70%为超近距离
CLOSE_BBOX_RATIO = 0.45        # 画面占比超过45%为近距离
NEAR_BBOX_RATIO = 0.20         # 画面占比超过20%为较近

# ==================== 距离估计常量 ====================

# 人体参考尺寸（单位：米）
REFERENCE_SHOULDER_WIDTH = 0.45   # 平均肩宽
REFERENCE_HEAD_WIDTH = 0.15       # 平均头宽
REFERENCE_UPPER_BODY_WIDTH = 0.30 # 上半身宽
REFERENCE_HEIGHT = 1.70           # 平均身高
REFERENCE_HEAD_HEIGHT = 0.22      # 头高

# 头部特征尺寸
REFERENCE_EYE_DISTANCE = 0.063    # 双眼间距
REFERENCE_EAR_DISTANCE = 0.145    # 双耳间距
REFERENCE_EYE_NOSE_DISTANCE = 0.035  # 眼鼻间距

# 距离估计阈值
DISTANCE_THRESHOLD_CLOSE = 2.0    # 近距离阈值（米）
DISTANCE_THRESHOLD_ULTRA_CLOSE = 0.8  # 超近距离阈值（米）

# 边界框占比距离估算参数
# 公式：distance = base - slope * (ratio - offset)
BBOX_DISTANCE_PARAMS = {
    'head_only': {
        'base': 0.70,
        'slope': 1.1,
        'offset': 0.2,
        'min': 0.12,
        'max': 0.70
    },
    'upper_body': {
        'base': 0.85,
        'slope': 1.25,
        'offset': 0.2,
        'min': 0.15,
        'max': 0.85
    },
    'full_body': {
        'base': 0.95,
        'slope': 1.3,
        'offset': 0.2,
        'min': 0.15,
        'max': 1.0
    }
}

# 超近距离固定距离值
EXTREME_CLOSE_DISTANCE_HEAD = 0.25
EXTREME_CLOSE_DISTANCE_UPPER = 0.30
EXTREME_CLOSE_DISTANCE_FULL = 0.35

# ==================== 相机参数常量 ====================

# 默认相机内参（未标定时使用）
# 笔记本摄像头典型值
DEFAULT_FX = 650.0
DEFAULT_FY = 650.0
DEFAULT_CX = 320.0  # 假设640x480分辨率
DEFAULT_CY = 240.0

# 相机外参默认值
DEFAULT_CAMERA_HEIGHT = 1.8  # 米
DEFAULT_PITCH_ANGLE = 30.0   # 度
DEFAULT_FOV_VERTICAL = 60.0  # 垂直视场角
DEFAULT_FOV_HORIZONTAL = 90.0  # 水平视场角

# ==================== 卡尔曼滤波常量 ====================

# 卡尔曼滤波参数
KALMAN_MAX_SPEED = 3.0           # 最大移动速度（米/秒）
KALMAN_PROCESS_NOISE = 0.1       # 过程噪声
KALMAN_MEASUREMENT_NOISE = 0.3   # 测量噪声
KALMAN_DT_MIN = 0.01             # 最小时间间隔
KALMAN_DT_MAX = 0.5              # 最大时间间隔

# 运动状态阈值
MOTION_STATIC_THRESHOLD = 0.1    # 静止速度阈值
MOTION_WALKING_THRESHOLD = 1.5   # 行走速度阈值
MOTION_RUNNING_THRESHOLD = 3.0   # 奔跑速度阈值

# ==================== 标定常量 ====================

# 棋盘格标定参数
CHECKERBOARD_DEFAULT_SIZE = (9, 6)  # 默认棋盘格尺寸
CHECKERBOARD_DEFAULT_SQUARE_SIZE = 25.0  # 默认方格大小（mm）
CALIBRATION_MIN_IMAGES = 15         # 最小标定图片数
CALIBRATION_MAX_IMAGES = 50         # 最大标定图片数

# ==================== 多帧融合常量 ====================

# 测量缓冲区大小
MEASUREMENT_BUFFER_SIZE = 10

# 置信度阈值
CONFIDENCE_HIGH = 0.8
CONFIDENCE_MEDIUM = 0.5
CONFIDENCE_LOW = 0.3

# ==================== API 常量 ====================

# WebSocket 配置
WS_VIDEO_ENDPOINT = "/ws/video"
WS_DATA_ENDPOINT = "/ws/data"
WS_HEARTBEAT_INTERVAL = 5.0  # 心跳间隔（秒）

# 数据推送频率
DATA_PUSH_RATE = 20  # Hz
DATA_PUSH_INTERVAL = 0.05  # 秒

# 视频编码质量
VIDEO_QUALITY = 80
VIDEO_BITRATE = 2000000

# ==================== 日志常量 ====================

LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
LOG_ROTATION = "10 MB"
LOG_RETENTION = "7 days"

# ==================== 调试常量 ====================

# 调试模式
DEBUG_MODE = False

# 调试信息开关
DEBUG_SHOW_KEYPOINTS = True
DEBUG_SHOW_BBOX = True
DEBUG_SHOW_DISTANCE = True
DEBUG_SHOW_METHOD = True
