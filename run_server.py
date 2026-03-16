#!/usr/bin/env python3
"""
简化版启动脚本 - 直接运行 FastAPI 服务
"""

import uvicorn
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

print("\n" + "="*60)
print("🔧 摄像头感知系统 - 调试模式")
print("="*60)
print()
print("正在启动服务器...")
print()

# 直接运行 uvicorn
uvicorn.run(
    "src.api.debug_server:app",
    host="0.0.0.0",
    port=8100,
    reload=False,
    log_level="info",
    access_log=True
)
