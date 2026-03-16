# 摄像头实时感知系统 - 使用文档

## 一、快速开始

### 1. 环境准备

```bash
# 进入项目目录
cd camera-perception-system

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 相机标定

#### 步骤 1：准备棋盘格
1. 打印 `calibration_data/checkerboard.pdf`（9×6 棋盘格）
2. 将棋盘格贴在平整的板上

#### 步骤 2：采集标定图片
```bash
# 打开摄像头采集窗口
python -m src.algorithms.calibration --help

# 使用摄像头实时采集（按空格保存图像）
python scripts/capture_calibration_images.py --output calibration_data/images/
```

或使用已有图片：
- 在不同角度/位置拍摄 15-20 张棋盘格图片
- 确保棋盘格覆盖图像的不同区域
- 将图片保存到 `calibration_data/images/` 目录

#### 步骤 3：执行标定
```bash
python -m src.algorithms.calibration \
  --images calibration_data/images/ \
  --output calibration_data/calib_params.json
```

输出示例：
```
2024-01-01 10:00:00 INFO Starting calibration with 18 images...
2024-01-01 10:00:05 INFO Found 18 valid images
2024-01-01 10:00:06 INFO Calibration RMS error: 0.123456
2024-01-01 10:00:06 INFO Mean reprojection error: 0.098765 pixels
2024-01-01 10:00:06 INFO Calibration parameters saved to calibration_data/calib_params.json
2024-01-01 10:00:06 SUCCESS Calibration completed!
fx=1200.50, fy=1199.80, cx=960.20, cy=540.10
```

### 3. 启动服务

#### 后端服务
```bash
# 启动 FastAPI 服务
python -m src.api.main --host 0.0.0.0 --port 8000
```

#### 前端服务
```bash
# 进入 web 目录
cd web

# 安装依赖（首次运行）
npm install

# 启动开发服务器
npm run dev
```

访问 http://localhost:5173

---

## 二、系统配置

### 相机配置（configs/camera.yaml）

```yaml
camera:
  id: 0                    # 摄像头设备 ID
  resolution: [1920, 1080] # 分辨率
  fps: 20                  # 帧率
  
calibration:
  checkerboard: [9, 6]     # 棋盘格角点数
  square_size: 25.0        # 棋盘格边长 (mm)
  
spatial:
  ref_shoulder_width: 0.45 # 参考肩宽 (米)
  camera_height: 1.8       # 摄像头安装高度 (米)
  pitch_angle: 30.0        # 俯角 (度)
```

### 修改配置
1. 编辑 `configs/camera.yaml`
2. 重启后端服务

---

## 三、API 接口

### REST API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/status` | GET | 获取系统状态 |
| `/api/camera/start` | POST | 启动摄像头 |
| `/api/camera/stop` | POST | 停止摄像头 |
| `/api/calibration/load` | POST | 加载标定参数 |
| `/api/calibration/status` | GET | 获取标定状态 |

### WebSocket

| 端点 | 说明 |
|------|------|
| `/ws/video` | 视频流推送（JPEG 帧） |
| `/ws/data` | 检测数据推送（JSON） |

### 使用示例

```python
import requests
import websockets
import asyncio

# 启动摄像头
requests.post('http://localhost:8000/api/camera/start')

# 获取状态
status = requests.get('http://localhost:8000/api/status')
print(status.json())

# 接收检测数据
async def receive_data():
    async with websockets.connect('ws://localhost:8000/ws/data') as ws:
        while True:
            data = await ws.recv()
            print(data)

asyncio.run(receive_data())
```

---

## 四、常见问题

### 1. 摄像头无法打开
- 检查摄像头设备 ID（尝试 0, 1, 2...）
- 确保没有其他程序占用摄像头
- 检查摄像头权限设置

### 2. 标定失败
- 确保棋盘格图片数量 ≥ 15 张
- 棋盘格应覆盖图像的不同区域
- 确保棋盘格角点清晰可见

### 3. 尺寸计算误差大
- 重新标定相机
- 准确测量摄像头安装高度和俯角
- 使用已知尺寸的物体进行校准

### 4. 帧率低
- 降低分辨率（如 1280×720）
- 降低帧率设置（如 15fps）
- 使用 GPU 加速 YOLO 推理

---

## 五、性能指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 帧率 | ≥15fps | 1920×1080 分辨率 |
| 端到端延迟 | ≤100ms | 采集→检测→展示 |
| 身高误差 | ±5cm | 距离 3m 内 |
| 距离误差 | ±0.2m | 距离 5m 内 |
| 手大小误差 | ±1cm | 距离 1m 内 |

---

## 六、项目结构

```
camera-perception-system/
├── src/
│   ├── algorithms/       # 核心算法
│   │   ├── calibration.py    # 相机标定
│   │   ├── detection.py      # 目标检测
│   │   └── spatial.py        # 空间计量
│   ├── data/             # 数据采集
│   │   └── video_capture.py  # 视频采集
│   ├── api/              # Web 后端
│   │   └── main.py           # FastAPI
│   └── utils/            # 工具函数
├── web/                  # Vue3 前端
├── configs/              # 配置文件
├── calibration_data/     # 标定数据
├── models/               # 模型文件
└── tests/                # 测试用例
```

---

## 七、开发指南

### 添加新算法
1. 在 `src/algorithms/` 创建新模块
2. 在 `__init__.py` 中导出
3. 在 `src/api/main.py` 中集成

### 修改前端
1. 编辑 `web/src/App.vue`
2. 运行 `npm run dev` 预览
3. 运行 `npm run build` 构建

### 运行测试
```bash
pytest tests/ -v
```

---

## 八、许可证

MIT License
