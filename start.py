#!/usr/bin/env python3
"""
跨平台启动脚本
支持 Windows / macOS / Linux
"""

import subprocess
import sys
import os
import socket
from pathlib import Path

def is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port))
            return True
        except OSError:
            return False

def kill_port_process(port: int) -> bool:
    """终止占用指定端口的进程（仅 Windows）"""
    if sys.platform != "win32":
        return False
    
    try:
        result = subprocess.run(
            f'netstat -ano | findstr ":{port}" | findstr "LISTENING"',
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            pids = set()
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    pids.add(parts[-1])
            
            for pid in pids:
                if pid.isdigit():
                    subprocess.run(f'taskkill /F /PID {pid}', shell=True, capture_output=True)
                    print(f"[清理] 已终止占用端口 {port} 的进程 (PID: {pid})")
            return True
    except Exception:
        pass
    return False

def main():
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)
    
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
    
    print("=" * 50)
    print("[启动] 私人专属面试顾问 V1.0")
    print("=" * 50)
    print()
    
    env_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not env_key:
        print("[警告] 未检测到 API 密钥环境变量！")
        print("请设置环境变量：")
        print("  Windows: set DASHSCOPE_API_KEY=your_api_key")
        print("  macOS/Linux: export DASHSCOPE_API_KEY=your_api_key")
        print()
        print("或者创建 .env 文件配置 OPENAI_API_KEY")
        print()
    
    port = 8000
    
    if not is_port_available(port):
        print(f"[提示] 端口 {port} 已被占用，正在清理...")
        kill_port_process(port)
        import time
        time.sleep(1)
    
    print(f"正在启动服务... (端口: {port})")
    print()
    
    try:
        subprocess.run(
            [sys.executable, "-m", "chainlit", "run", "app/main.py", "--port", str(port)],
            cwd=script_dir,
            check=True
        )
    except KeyboardInterrupt:
        print("\n服务已停止")
    except FileNotFoundError:
        print("[错误] 未找到 chainlit，请先安装依赖：")
        print("   pip install -e .")
        sys.exit(1)
    except Exception as e:
        print(f"[错误] 启动失败：{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
