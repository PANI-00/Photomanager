"""Photomanager 启动入口 — 双击启动桌面客户端（无控制台窗口）
通过 pythonw.exe 或 VBScript 调用时不显示命令窗口
启动日志写入 entry.log 以方便排查（pythonw.exe 模式下没有控制台输出）
"""
import os
import sys
import time
import traceback
from pathlib import Path

_LOG_FILE = Path(__file__).with_name("entry.log")


def _log(msg: str):
    """将日志写入文件（pythonw.exe 模式下 stdout/stderr 不可见）"""
    try:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except Exception:
        pass  # 无法写日志就忽略


def main():
    # pythonw.exe 下 stdin/stdout/stderr 均为 None，替换为 devnull 防止崩溃
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    if sys.stdin is None:
        sys.stdin = open(os.devnull, "r", encoding="utf-8")

    try:
        _log("正在启动桌面客户端…")
        from config import IMAGES_DIR, DATA_DIR
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # 直接启动 desktop_app
        import desktop_app
        app = desktop_app.PhotoManager()
        app.protocol("WM_DELETE_WINDOW", app.on_close)
        app.mainloop()
    except Exception:
        err = traceback.format_exc()
        _log(f"启动失败:\n{err}")
        # 尝试用消息框通知用户
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0, f"Photomanager 启动失败，详情请查看 entry.log\n\n{err[-500:]}",
                "Photomanager 错误", 0x10  # MB_ICONERROR
            )
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
