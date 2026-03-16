# Camera Perception System | 摄像头实时感知系统

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Vue.js](https://img.shields.io/badge/Vue.js-3.0-4FC08D.svg)](https://vuejs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![YOLO](https://img.shields.io/badge/YOLOv8-v8-00FFFF.svg)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**English** | [中文](#中文文档)

A real-time computer vision system for human body/hand detection and spatial measurement, supporting video streaming, size measurement, and top-view mapping.

基于计算机视觉的实时人体/手部检测与空间计量系统，支持视频流展示、尺寸测量和顶视图映射。

[Live Demo](#) · [Documentation](#) · [Report Bug](../../issues)

<img src="docs/demo.gif" alt="Demo" width="800"/>

</div>

---

## 📋 Table of Contents | 目录

- [Features](#-features--功能特性)
- [Architecture](#-architecture--技术架构)
- [Quick Start](#-quick-start--快速开始)
- [Installation](#-installation--安装指南)
- [Usage](#-usage--使用说明)
- [API Reference](#-api-reference--api文档)
- [Performance](#-performance--性能指标)
- [Contributing](#-contributing--贡献指南)
- [License](#-license--许可证)

---

## ✨ Features | 功能特性

### 🎥 Video Streaming | 视频流采集
- Real-time USB/IP camera capture, supporting 1080P@20fps
- Automatic frame rate control and quality optimization
- Hardware acceleration support (Apple Silicon, CUDA)

### 📐 Camera Calibration | 相机标定
- Zhang's checkerboard calibration method
- Intrinsic/extrinsic parameters and distortion coefficients
- Interactive calibration tool with real-time preview
- Automatic checkerboard detection

### 🔍 Object Detection | 目标检测
- **Human Detection**: YOLOv8n + YOLOv8-Pose (17 COCO keypoints)
- **Hand Detection**: MediaPipe Hands (21 keypoints)
- **Keypoint Smoothing**: Kalman filter for temporal consistency
- **Multi-person Tracking**: ID assignment and tracking

### 📏 Spatial Measurement | 空间计量
| Measurement | Accuracy | Method |
|-------------|----------|--------|
| Distance | ±0.15m | Multi-method fusion |
| Height | ±5cm | Pose-based estimation |
| Hand Size | ±1cm | Keypoint analysis |
| Body Part | 95%+ | AI classification |

- **Adaptive Distance Estimation**:
  - Head-based (eyes, ears, nose) - Primary method
  - Body keypoints (shoulders, hips) - Fallback for no-head scenarios
  - Bounding box ratio - For extreme close-range
- **Dynamic Calibration**: User-defined reference sizes
- **Multi-frame Fusion**: Temporal stability improvement

### 🗺️ Top-View Mapping | 顶视图映射
- 3D spatial coordinates → 2D bird's eye view
- Real-time position tracking
- Multi-target trajectory visualization

### 🌐 Web Interface | Web界面
- **Frontend**: Vue 3 + Vite + Canvas annotation
- **Backend**: FastAPI + WebSocket real-time push
- **Features**:
  - Real-time video with detection overlay
  - Spatial data dashboard
  - Calibration tool interface
  - Parameter configuration panel
  - Performance monitoring

---

## 🏗️ Architecture | 技术架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web Frontend (Vue3)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Video Render│  │ Data Panel  │  │ Top-View    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                              ↕ WebSocket / HTTP
┌─────────────────────────────────────────────────────────────────┐
│                      Backend API (FastAPI)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │Video Stream │  │ Data Push   │  │ REST API    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                    Algorithm Layer (Python)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Calibration │  │ Detection   │  │ Measurement │             │
│  │ (OpenCV)    │  │ (YOLO+MP)   │  │ (Spatial)   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

### Tech Stack | 技术栈
- **Backend**: Python 3.10+, FastAPI, WebSocket, NumPy, OpenCV
- **AI Models**: YOLOv8 (Ultralytics), MediaPipe
- **Frontend**: Vue 3, Vite, Canvas API, Axios
- **Deployment**: Uvicorn, Docker (optional)

---

## 🚀 Quick Start | 快速开始

### Prerequisites | 环境要求
- Python 3.10+
- Node.js 16+
- USB Camera or Laptop Built-in Camera
- macOS / Linux / Windows

### One-Line Start | 一键启动

```bash
# Clone repository
git clone https://github.com/yangfanconan/camera-perception-system.git
cd camera-perception-system

# Setup environment
python setup.py  # Auto-install dependencies

# Start services
./start.sh  # Or: python scripts/start_all.py
```

Access the system:
- Web Interface: http://localhost:8100
- API Docs: http://localhost:8001/docs

---

## 📦 Installation | 安装指南

### Step 1: Backend Setup | 后端安装

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download YOLO models
python scripts/download_models.py
```

### Step 2: Frontend Setup | 前端安装

```bash
cd web
npm install
npm run dev  # Development mode
# npm run build  # Production build
```

### Step 3: Start Services | 启动服务

```bash
# Terminal 1: Backend
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8001

# Terminal 2: Frontend (dev)
cd web && npm run dev

# Or use the startup script
python scripts/start_all.py
```

---

## 📖 Usage | 使用说明

### 1. Camera Calibration | 相机标定

Before measurement, calibrate your camera for accurate results:

1. Print a checkerboard pattern (9x6 squares, 25mm each)
2. Open web interface → "Calibration" tab
3. Click "Start Capture" and show the checkerboard from different angles
4. Capture 15+ images with good coverage
5. Click "Calibrate" to compute parameters

### 2. Real-Time Detection | 实时检测

1. Go to "Monitor" tab
2. Click "Start Camera"
3. View real-time detection with:
   - Bounding boxes and keypoints
   - Distance and height measurements
   - Body part classification
   - Top-view position mapping

### 3. Spatial Measurement | 空间计量

The system automatically estimates:
- **Distance**: From camera to person (0.15m - 10m range)
- **Height**: Person's height based on pose
- **Hand Size**: Palm size when hands are detected

Measurement methods (adaptive selection):
- `head` - Eye/ear distance (most accurate)
- `body_keypoints` - Shoulder/hip width (fallback)
- `bbox_ratio` - Bounding box coverage (extreme close-range)

### 4. API Usage | API使用

```python
import requests

# Get status
r = requests.get("http://localhost:8001/api/status")
print(r.json())

# Start camera
requests.post("http://localhost:8001/api/camera/start", json={
    "camera_id": 0,
    "resolution": [1920, 1080],
    "fps": 20
})

# Get spatial config
r = requests.get("http://localhost:8001/api/spatial/config")
print(r.json())
```

---

## 📚 API Reference | API文档

### REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | System status |
| `/api/camera/start` | POST | Start camera |
| `/api/camera/stop` | POST | Stop camera |
| `/api/calibration/capture` | POST | Capture calibration frame |
| `/api/calibration/calibrate` | POST | Run calibration |
| `/api/spatial/config` | GET/POST | Spatial measurement config |
| `/api/performance` | GET | Performance metrics |
| `/api/errors` | GET | Error statistics |

### WebSocket Endpoints

| Endpoint | Type | Description |
|----------|------|-------------|
| `/ws/video` | Binary | Video stream (MJPEG) |
| `/ws/data` | JSON | Detection data (20Hz) |

Full API documentation: http://localhost:8001/docs (Swagger UI)

---

## 📊 Performance | 性能指标

Tested on MacBook Pro M1 / Intel i7 / RTX 3060:

| Metric | Value |
|--------|-------|
| Video Resolution | 1920x1080 @ 20fps |
| Detection Latency | 30-50ms |
| Spatial Calculation | <10ms |
| WebSocket Push | 20Hz |
| End-to-End Delay | <100ms |
| Distance Accuracy | ±0.15m (close-range) |
| Body Part Detection | 95%+ |

### Optimization Features
- Smart caching for repeated calculations
- Kalman filtering for temporal stability
- Multi-method fusion for robustness
- Hardware acceleration (MPS/CUDA)

---

## 🤝 Contributing | 贡献指南

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black src/
isort src/

# Type checking
mypy src/
```

---

## 📄 License | 许可证

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## 🙏 Acknowledgments | 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [MediaPipe](https://mediapipe.dev) for hand tracking
- [FastAPI](https://fastapi.tiangolo.com) for backend framework
- [Vue.js](https://vuejs.org) for frontend framework

---

## 📞 Contact | 联系方式

- **Author**: yangfanconan
- **Email**: [your-email@example.com]
- **GitHub**: [@yangfanconan](https://github.com/yangfanconan)
- **Issues**: [GitHub Issues](../../issues)

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

**如果这个项目对你有帮助，请给个 Star！**

</div>

---

## 中文文档

### 项目简介

摄像头实时感知系统是一个基于计算机视觉的智能测量平台，能够实时检测人体和手部，并进行精确的空间计量。

### 核心功能

1. **实时视频采集**: 支持USB/IP摄像头，最高1080P@20fps
2. **AI目标检测**: YOLOv8人体检测 + MediaPipe手部检测
3. **智能距离估计**: 多方法融合（头部/身体关键点/边界框）
4. **相机标定**: 张正友棋盘格标定法
5. **Web实时展示**: Vue3前端 + FastAPI后端

### 快速开始

```bash
# 克隆仓库
git clone https://github.com/yangfanconan/camera-perception-system.git
cd camera-perception-system

# 安装依赖并启动
python setup.py
./start.sh
```

访问 http://localhost:8100 查看系统界面。

### 技术亮点

- **自适应距离估计**: 根据场景自动选择最佳估计方法
- **极近距离优化**: 画面占比>70%时自动切换高精度模式
- **模块化架构**: 易于扩展和维护
- **完整错误处理**: 全局异常捕获和监控

### 使用场景

- 智能安防监控
- 人体测量分析
- 交互式展示系统
- 运动姿态分析

---

<div align="center">

Made with ❤️ by yangfanconan

</div>
