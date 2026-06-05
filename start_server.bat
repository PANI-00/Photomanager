@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Photomanager 桌面客户端

echo.
echo  [INFO] 启动 Photomanager 桌面客户端 ...
echo  [INFO] 关闭本窗口 = 退出程序
echo.
echo  [TIP]  想无窗口运行？双击「启动 Photomanager.vbs」即可
echo  ============================================
echo.

".venv\Scripts\pythonw.exe" "entry.py"

echo.
echo  程序已退出，按任意键关闭...
pause >nul
