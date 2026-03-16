# 摄像头实时感知 + 空间计量系统 - 项目交付文档

## 📋 交付清单

### 1. 源代码文件

| 目录 | 文件数 | 说明 |
|------|--------|------|
| `src/algorithms/` | 6 | 核心算法模块（标定、检测、空间计量） |
| `src/data/` | 4 | 数据采集与处理（视频采集、编码、数据库） |
| `src/api/` | 2 | Web 后端 API |
| `src/utils/` | 2 | 工具模块（配置管理、日志） |
| `web/src/` | 2 | Vue3 前端 |
| `tests/` | 6 | 测试用例 |

**总计**: 约 22 个核心源代码文件

### 2. 配置文件

| 文件 | 说明 |
|------|------|
| `configs/camera.yaml` | 相机和系统配置 |
| `requirements.txt` | Python 依赖 |
| `web/package.json` | 前端依赖 |
| `pyproject.toml` | 项目配置 |
| `.env.example` | 环境变量示例 |
| `Dockerfile` | Docker 镜像配置 |
| `docker-compose.yml` | Docker Compose 配置 |

### 3. 文档文件

| 文件 | 说明 |
|------|------|
| `README.md` | 项目说明 |
| `USAGE.md` | 使用文档 |
| `DELIVERY.md` | 本文件 - 交付文档 |

### 4. 脚本文件

| 文件 | 说明 |
|------|------|
| `start.sh` | Linux/Mac启动脚本 |
| `start.bat` | Windows 启动脚本 |
| `scripts/capture_calibration_images.py` | 标定图片采集脚本 |

---

## 🏗️ 系统架构

### 技术栈总览

```
┌─────────────────────────────────────────────────────────────┐
│                      前端层 (Vue3 + Canvas)                   │
│  - 实时监控页面  - 标定工具  - 参数配置  - 顶视图展示          │
└─────────────────────────────────────────────────────────────┘
                            ↕ WebSocket (视频流 + 数据)
┌─────────────────────────────────────────────────────────────┐
│                    后端层 (FastAPI + Python)                  │
│  - REST API  - WebSocket 推送  - 配置管理  - 数据持久化        │
└─────────────────────────────────────────────────────────────┘
                            ↕ 函数调用
┌─────────────────────────────────────────────────────────────┐
│                    算法层 (OpenCV + PyTorch)                  │
│  - 相机标定  - YOLOv8 检测  - MediaPipe  - 空间计量           │
└─────────────────────────────────────────────────────────────┘
                            ↕ 硬件访问
┌─────────────────────────────────────────────────────────────┐
│                      硬件层 (摄像头 + GPU)                    │
│  - USB/IP 摄像头  - GPU 加速（可选）                          │
└─────────────────────────────────────────────────────────────┘
```

### 模块依赖关系

```
src/
├── algorithms/           # 核心算法（无外部依赖）
│   ├── calibration.py    # 相机标定 → OpenCV
│   ├── detection.py      # 目标检测 → YOLOv8, MediaPipe
│   └── spatial.py        # 空间计量 → NumPy
├── data/                 # 数据处理
│   ├── video_capture.py  # → OpenCV
│   ├── video_encoder.py  # → PyAV, aiortc (可选)
│   └── database.py       # → SQLite
├── api/                  # Web 后端
│   └── main_full.py      # → FastAPI, WebSocket
└── utils/                # 工具模块
    ├── config_manager.py # → PyYAML, watchdog
    └── logger.py         # → loguru
```

---

## 📊 功能特性清单

### ✅ 已实现功能

#### 1. 相机标定
- [x] 张正友棋盘格标定法
- [x] 交互式标定 GUI
- [x] 标定参数保存/加载（JSON）
- [x] 重投影误差计算
- [x] 在线标定图片采集
- [x] 外参配置（高度、俯角）

#### 2. 目标检测
- [x] YOLOv8 人体检测
- [x] YOLOv8-Pose 精确关键点（18 个 COCO 关键点）
- [x] MediaPipe 手部检测（21 个关键点）
- [x] 卡尔曼滤波关键点平滑
- [x] 多人物/多手部追踪
- [x] 检测置信度阈值配置

#### 3. 空间计量
- [x] 像素坐标→世界坐标转换
- [x] 距离计算（人到摄像头）
- [x] 身高计算（厘米）
- [x] 手大小计算（厘米）
- [x] 动态校准（用户输入参考值）
- [x] 误差修正模型（多项式拟合）
- [x] 多帧融合（提高稳定性）
- [x] 置信度评估

#### 4. 顶视图映射
- [x] 3D→2D 鸟瞰图转换
- [x] XY 坐标系实时展示
- [x] 多人物位置追踪
- [x] 可配置比例尺

#### 5. 视频流
- [x] 摄像头实时采集
- [x] 分辨率/帧率配置
- [x] MJPEG 流编码
- [x] H.264 编码支持（PyAV/aiortc）
- [x] FLV 流支持
- [x] 去畸变处理

#### 6. Web 后端
- [x] FastAPI REST API
- [x] WebSocket 视频流推送
- [x] WebSocket 数据推送
- [x] 配置管理 API
- [x] 标定 API
- [x] 数据导出 API
- [x] 健康检查端点
- [x] CORS 支持

#### 7. Web 前端
- [x] Vue3 实时监控页面
- [x] 视频流渲染（Canvas）
- [x] 关键点/边界框标注
- [x] 数据面板（人数、手数、身高、距离）
- [x] 顶视图画布
- [x] 相机标定工具界面
- [x] 参数配置面板
- [x] 系统状态显示
- [x] 响应式设计

#### 8. 数据持久化
- [x] SQLite 数据库
- [x] 检测记录存储
- [x] 人体/手部指标存储
- [x] 历史数据查询
- [x] 统计分析
- [x] 数据导出（JSON）

#### 9. 配置管理
- [x] YAML 配置文件
- [x] 环境变量覆盖
- [x] 配置热重载
- [x] 配置验证

#### 10. 日志和错误处理
- [x] 统一日志配置（loguru）
- [x] 异常捕获和记录
- [x] 自定义异常类
- [x] 性能监控
- [x] 请求日志中间件

#### 11. 部署支持
- [x] Docker 镜像（多阶段构建）
- [x] Docker Compose 配置
- [x] 开发/生产环境分离
- [x] Nginx 反向代理配置
- [x] 一键启动脚本

#### 12. 测试
- [x] 单元测试（pytest）
- [x] 集成测试
- [x] API 测试
- [x] 测试夹具（fixtures）

---

## 📈 性能指标

| 指标 | 目标值 | 测试条件 |
|------|--------|----------|
| 视频帧率 | ≥15fps | 1920×1080, CPU |
| 端到端延迟 | ≤100ms | 局域网 |
| 身高误差 | ±5cm | 距离 3m 内 |
| 距离误差 | ±0.2m | 距离 5m 内 |
| 手大小误差 | ±1cm | 距离 1m 内 |
| 标定重投影误差 | <0.5px | 15+ 张图片 |
| API 响应时间 | <50ms | 本地 |

---

## 🚀 部署指南

### 方式一：本地开发部署

```bash
# 1. 克隆项目
cd camera-perception-system

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动服务
./start.sh  # 或 start.bat

# 5. 访问系统
# 前端：http://localhost:5173
# API: http://localhost:8000
# 文档：http://localhost:8000/docs
```

### 方式二：Docker 部署

```bash
# 1. 构建并启动
docker-compose up -d

# 2. 查看日志
docker-compose logs -f backend

# 3. 停止服务
docker-compose down

# 4. 开发模式（带前端热重载）
docker-compose --profile dev up -d
```

### 方式三：生产环境部署

```bash
# 1. 构建生产镜像
docker build -t camera-perception:latest --target production .

# 2. 运行容器
docker run -d \
  --name camera-perception \
  -p 8000:8000 \
  -v ./data:/app/data \
  -v ./calibration_data:/app/calibration_data \
  -v ./models:/app/models \
  --device /dev/video0 \
  camera-perception:latest

# 3. 配置 Nginx 反向代理
# 参考 web/nginx.conf
```

---

## 🔧 配置说明

### 核心配置项（configs/camera.yaml）

```yaml
camera:
  id: 0                    # 摄像头设备 ID
  resolution: [1920, 1080] # 分辨率
  fps: 20                  # 帧率

calibration:
  checkerboard: [9, 6]     # 棋盘格角点数
  square_size: 25.0        # 棋盘格边长 (mm)
  min_images: 15           # 最少标定图片数

detection:
  conf_threshold: 0.5      # 置信度阈值
  smooth_enabled: true     # 关键点平滑

spatial:
  ref_shoulder_width: 0.45 # 参考肩宽 (米)
  camera_height: 1.8       # 摄像头安装高度 (米)
  pitch_angle: 30.0        # 俯角 (度)
```

### 环境变量覆盖

```bash
# .env 文件
CAMERA_ID=0
RESOLUTION=1920x1080
FPS=20
CONF_THRESHOLD=0.5
CAMERA_HEIGHT=1.8
LOG_LEVEL=INFO
```

---

## 📁 目录结构

```
camera-perception-system/
├── src/
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── calibration.py          # 相机标定
│   │   ├── calibration_gui.py      # 交互式标定
│   │   ├── detection.py            # 目标检测
│   │   ├── detection_enhanced.py   # 增强检测
│   │   ├── spatial.py              # 空间计量
│   │   └── spatial_enhanced.py     # 增强空间计量
│   ├── data/
│   │   ├── __init__.py
│   │   ├── video_capture.py        # 视频采集
│   │   ├── video_encoder.py        # 视频编码
│   │   └── database.py             # 数据库
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                 # 基础 API
│   │   └── main_full.py            # 完整 API
│   └── utils/
│       ├── __init__.py
│       ├── config_manager.py       # 配置管理
│       └── logger.py               # 日志处理
├── web/
│   ├── src/
│   │   ├── App.vue                 # 主页面
│   │   └── main.js
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── Dockerfile
│   └── nginx.conf
├── tests/
│   ├── conftest.py
│   ├── test_calibration.py
│   ├── test_detection.py
│   ├── test_spatial.py
│   ├── test_integration.py
│   └── test_api.py
├── configs/
│   └── camera.yaml
├── scripts/
│   └── capture_calibration_images.py
├── models/                     # 模型目录
├── calibration_data/           # 标定数据
├── data/                       # 数据库数据
├── logs/                       # 日志文件
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── README.md
├── USAGE.md
├── DELIVERY.md                 # 本文件
├── start.sh
├── start.bat
└── .env.example
```

---

## 🧪 测试指南

### 运行测试

```bash
# 安装测试依赖
pip install pytest pytest-cov pytest-asyncio

# 运行所有测试
pytest tests/ -v

# 运行特定模块测试
pytest tests/test_calibration.py -v
pytest tests/test_integration.py -v

# 生成覆盖率报告
pytest tests/ --cov=src --cov-report=html

# 查看覆盖率报告
open htmlcov/index.html  # Mac
xdg-open htmlcov/index.html  # Linux
```

### 测试覆盖率目标

| 模块 | 目标覆盖率 | 当前覆盖率 |
|------|------------|------------|
| algorithms/ | ≥80% | - |
| data/ | ≥70% | - |
| api/ | ≥60% | - |
| utils/ | ≥70% | - |

---

## 🐛 常见问题排查

### 1. 摄像头无法打开

```bash
# 检查摄像头设备
ls -l /dev/video*  # Linux
system_profiler SPCameraDataType  # Mac

# 测试摄像头
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"

# 解决方案
# - 检查设备 ID（尝试 0, 1, 2...）
# - 关闭其他占用摄像头的程序
# - 检查权限设置
```

### 2. YOLO 模型下载失败

```bash
# 手动下载模型
yolo download yolov8n.pt
yolo download yolov8n-pose.pt

# 或从 GitHub 下载
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

### 3. 标定失败

```bash
# 检查棋盘格图片
# - 确保至少 15 张有效图片
# - 棋盘格应覆盖图像不同区域
# - 角点检测清晰

# 重新采集
python scripts/capture_calibration_images.py
```

### 4. WebSocket 连接失败

```bash
# 检查防火墙
# 检查 CORS 配置
# 使用 ws:// 而非 http://
```

---

## 📞 技术支持

### 问题反馈

1. 查看日志文件：`logs/app.log`
2. 启用调试模式：`LOG_LEVEL=DEBUG`
3. 提交 Issue 时附上：
   - 系统环境（OS、Python 版本）
   - 错误日志
   - 复现步骤

### 联系方式

- GitHub Issues: [项目 Issue 页面]
- 邮箱：[开发者邮箱]

---

## 📝 版本信息

| 版本 | 日期 | 说明 |
|------|------|------|
| 0.1.0 | 2024-01-01 | 初始版本 |
| 0.2.0 | 2024-01-15 | 完整功能版本（本次交付） |

---

## ✅ 验收标准

- [ ] 所有核心功能正常运行
- [ ] 单元测试通过率 ≥90%
- [ ] 文档完整准确
- [ ] Docker 部署成功
- [ ] 性能指标达标
- [ ] 无严重 Bug

---

**交付完成！** 🎉
