# 摄像头实时感知系统 - M5 Pro 快速启动指南

## 🍎 为您的设备优化

已检测到您的设备：
- **芯片**: Apple M5 Pro (24GB 统一内存)
- **系统**: macOS 26.3.1 ARM64
- **PyTorch**: 2.10.0 (MPS 加速可用)

## 🚀 一键启动

### 首次运行

```bash
cd /Users/yangfan/camera-perception-system

# 1. 安装依赖（约 5 分钟）
./install_m5.sh

# 2. 启动优化服务
./start_m5_optimized.sh
```

### 后续运行

```bash
cd /Users/yangfan/camera-perception-system
source venv/bin/activate
./start_m5_optimized.sh
```

## 📊 性能预期

在 M5 Pro 芯片上的性能：

| 功能 | 性能 |
|------|------|
| YOLOv8 推理 | ~45 FPS (MPS + FP16) |
| 视频采集 | 1080P @ 30fps |
| 端到端延迟 | ~30ms |
| CPU 占用 | ~40% (硬件编码) |

## 🔧 常用命令

```bash
# 测试 MPS 加速
python -m src.algorithms.detection_mps --source 0 --device auto

# 测试视频采集
python -m src.data.video_capture_macos --camera 0

# 查看系统优化状态
python -c "from src.utils import get_optimizer; o = get_optimizer(); print(f'MPS: {o.mps_available}, Device: {o.device}')"

# 使用优化配置
export CONFIG_FILE=configs/camera_apple_silicon.yaml
python -m src.api.main_full
```

## 📁 重要文件

| 文件 | 说明 |
|------|------|
| `configs/camera_apple_silicon.yaml` | M5 Pro 优化配置 |
| `src/algorithms/detection_mps.py` | MPS 加速检测模块 |
| `src/data/video_capture_macos.py` | macOS 视频采集 |
| `src/utils/apple_silicon.py` | Apple Silicon 优化器 |
| `APPLE_SILICON_OPTIMIZATION.md` | 详细优化文档 |

## ⚠️ 常见问题

### MPS 加速未启用

```bash
# 检查 MPS 状态
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"

# 如果返回 False，升级 PyTorch
pip install --upgrade torch torchvision
```

### 摄像头无法打开

```bash
# 列出可用摄像头
python -m src.data.video_capture_macos

# 尝试不同后端
python -m src.data.video_capture_macos --backend builtin
```

### 内存不足

编辑 `configs/camera_apple_silicon.yaml`：
```yaml
apple_silicon:
  mps_memory_limit: 0.6  # 降低到 60%
```

## 📖 完整文档

- [README.md](README.md) - 项目总览
- [APPLE_SILICON_OPTIMIZATION.md](APPLE_SILICON_OPTIMIZATION.md) - M5 Pro 优化详解
- [USAGE.md](USAGE.md) - 使用指南
- [DELIVERY.md](DELIVERY.md) - 交付文档

## 🎯 下一步

1. ✅ 运行 `./install_m5.sh` 安装依赖
2. ✅ 运行 `./start_m5_optimized.sh` 启动服务
3. ✅ 访问 http://localhost:5173 查看前端
4. ✅ 查看 `APPLE_SILICON_OPTIMIZATION.md` 了解优化详情

---

**准备就绪！开始使用 M5 Pro 优化版本吧！** 🚀
