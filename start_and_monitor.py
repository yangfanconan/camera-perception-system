#!/usr/bin/env python3
"""
实时日志监控工具
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

def main():
    log_file = Path("logs/server.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("🚀 启动服务器并监控日志")
    print("="*70)
    print()
    
    # 启动服务器
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.api.full_server_v2"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    print(f"✅ 服务器已启动 (PID: {proc.pid})")
    print()
    print("📊 实时日志 (按 Ctrl+C 停止)")
    print("-" * 70)
    print()
    
    stats = {'info': 0, 'warning': 0, 'error': 0, 'success': 0}
    
    try:
        with open(log_file, 'w') as f:
            for line in proc.stdout:
                line = line.rstrip()
                ts = datetime.now().strftime('%H:%M:%S')
                f.write(f"[{ts}] {line}\n")
                f.flush()
                
                if 'ERROR' in line or 'Error' in line:
                    print(f"\033[91m[{ts}] {line}\033[0m")
                    stats['error'] += 1
                elif 'WARNING' in line or 'Warning' in line:
                    print(f"\033[93m[{ts}] {line}\033[0m")
                    stats['warning'] += 1
                elif '✅' in line or 'success' in line.lower():
                    print(f"\033[92m[{ts}] {line}\033[0m")
                    stats['success'] += 1
                else:
                    print(f"\033[94m[{ts}] {line}\033[0m")
                    stats['info'] += 1
                    
    except KeyboardInterrupt:
        print("\n\n⏹️  停止服务器...")
        proc.terminate()
        proc.wait()
        
        print("\n📊 统计:")
        print(f"  INFO: {stats['info']}")
        print(f"  WARNING: {stats['warning']}")
        print(f"  ERROR: {stats['error']}")
        print(f"\n📁 日志文件：{log_file.absolute()}")

if __name__ == '__main__':
    main()
