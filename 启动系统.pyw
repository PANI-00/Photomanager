"""CLIP 检索系统一键启动器
双击运行：自动安装依赖 → 启动服务 → 打开浏览器
"""
import importlib
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"
PROJECT_DIR = Path(__file__).parent
MAIN_PY = PROJECT_DIR / "main.py"
REQ_TXT = PROJECT_DIR / "requirements.txt"

# 需要检查的核心依赖
CORE_DEPS = ["torch", "clip", "transformers", "fastapi", "uvicorn", "PIL"]


def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((HOST, port)) == 0


def check_dependencies() -> list[str]:
    """返回缺失的依赖列表"""
    missing = []
    for mod in CORE_DEPS:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    return missing


def install_dependencies():
    """安装 requirements.txt 中的所有依赖"""
    print("=" * 55)
    print("  CLIP 检索系统 — 首次运行准备")
    print("=" * 55)
    print("\n📦 检测到依赖未安装，正在自动安装...")
    print("   首次安装可能需要 5~15 分钟，请耐心等待\n")

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(REQ_TXT)],
    )
    if result.returncode != 0:
        print("\n❌ 自动安装失败，请手动运行：")
        print(f"   cd {PROJECT_DIR}")
        print("   pip install -r requirements.txt")
        input("\n按 Enter 退出...")
        sys.exit(1)

    # 安装后再次验证
    still_missing = check_dependencies()
    if still_missing:
        print(f"\n❌ 以下依赖仍未安装成功: {', '.join(still_missing)}")
        print("请尝试手动安装。")
        input("按 Enter 退出...")
        sys.exit(1)

    print("\n✅ 依赖安装完成！\n")


def start_server():
    """启动 FastAPI 服务（后台无窗口进程）"""
    print("🚀 正在启动服务...")
    process = subprocess.Popen(
        [sys.executable, str(MAIN_PY)],
        cwd=str(PROJECT_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return process


def wait_for_server(timeout: int = 60) -> bool:
    """等待服务就绪，最多等待 timeout 秒"""
    for i in range(timeout):
        if is_port_in_use(PORT):
            return True
        time.sleep(1)
    return False


def main():
    # 关闭旧服务
    if is_port_in_use(PORT):
        print("检测到旧服务正在运行，先关闭...")
        import os
        os.system(f"taskkill /F /FI \"PID eq {subprocess.check_output(['netstat', '-ano'], shell=True).decode('gbk').split()}\" >nul 2>&1")
        time.sleep(2)

    # 第一步：检查并安装依赖
    missing = check_dependencies()
    if missing:
        install_dependencies()
    else:
        print("✅ 依赖已就绪")

    # 第二步：启动服务
    start_server()

    # 第三步：等待服务就绪
    print("⏳ 等待服务启动...")
    if wait_for_server():
        print(f"\n✅ 服务已启动: {URL}")
        webbrowser.open(URL)
        print("\n🔒 关闭此窗口即可停止服务")
        print("📂 浏览器保持打开即可继续使用")
    else:
        print("\n❌ 服务启动超时")
        print("请尝试手动运行: python main.py")

    # 保持窗口打开，显示服务日志
    print("\n按 Ctrl+C 停止服务...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n服务已停止。")
        sys.exit(0)
