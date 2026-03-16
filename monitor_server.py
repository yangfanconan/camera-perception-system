#!/usr/bin/env python3
"""
服务器启动与实时监控工具
"""

import subprocess
import sys
import time
import signal
import os
from datetime import datetime
from pathlib import Path

class ServerMonitor:
    def __init__(self):
        self.proc = None
        self.running = True
        self.log_file = Path("logs/server_monitor.log")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 绑定信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        print("\n\n⏹️  收到停止信号，正在关闭服务器...")
        self.running = False
    
    def start(self):
        print("="*70)
        print("🚀 摄像头感知系统 v2 - M5 Pro 优化版")
        print("="*70)
        print()
        
        # 检查端口
        print("📋 检查端口 8100...")
        result = subprocess.run(
            f"lsof -ti :8100 | xargs kill -9 2>/dev/null || true",
            shell=True,
            capture_output=True
        )
        print("✅ 端口已清理")
        print()
        
        # 启动服务器
        print("🚀 启动服务器进程...")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "src.api.full_server_v2"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(Path(__file__).parent)
        )
        
        print(f"✅ 服务器已启动 (PID: {self.proc.pid})")
        print()
        print("📊 实时日志监控 (按 Ctrl+C 停止)")
        print("-" * 70)
        print()
        
        # 日志统计
        stats = {
            'info': 0,
            'warning': 0,
            'error': 0,
            'success': 0
        }
        
        try:
            with open(self.log_file, 'w') as log_f:
                for line in self.proc.stdout:
                    line = line.rstrip()
                    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                    log_line = f"[{timestamp}] {line}"
                    
                    # 写入日志文件
                    log_f.write(log_line + '\n')
                    log_f.flush()
                    
                    # 彩色输出
                    if 'ERROR' in line or 'Error' in line or 'Exception' in line:
                        print(f"\033[91m[{timestamp}] {line}\033[0m")
                        stats['error'] += 1
                    elif 'WARNING' in line or 'Warning' in line:
                        print(f"\033[93m[{timestamp}] {line}\033[0m")
                        stats['warning'] += 1
                    elif '✅' in line or 'success' in line.lower() or 'Success' in line:
                        print(f"\033[92m[{timestamp}] {line}\033[0m")
                        stats['success'] += 1
                    elif 'INFO' in line or '✅' in line:
                        print(f"\033[94m[{timestamp}] {line}\033[0m")
                        stats['info'] += 1
                    else:
                        print(f"[{timestamp}] {line}")
                    
                    if not self.running:
                        break
                        
        except KeyboardInterrupt:
            pass
        finally:
            print("\n")
            print("="*70)
            print("📊 日志统计")
            print("="*70)
            print(f"  INFO:    {stats['info']}")
            print(f"  WARNING: {stats['warning']}")
            print(f"  ERROR:   {stats['error']}")
            print(f"  SUCCESS: {stats['success']}")
            print()
            print(f"📁 完整日志：{self.log_file.absolute()}")
            print()
            
            if self.proc:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            
            print("✅ 服务器已停止")


if __name__ == '__main__':
    monitor = ServerMonitor()
    monitor.start()
