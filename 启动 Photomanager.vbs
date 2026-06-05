' Photomanager 启动器 — 双击运行，无控制台窗口
' 使用 pythonw.exe 避免显示命令窗口

Set WshShell = CreateObject("WScript.Shell")
Set FSO = CreateObject("Scripting.FileSystemObject")

' 切换到 VBS 文件所在目录（项目根目录）
strPath = FSO.GetParentFolderName(WScript.ScriptFullName)

' 用 pythonw.exe 后台运行入口脚本
strCmd = """" & strPath & "\.venv\Scripts\pythonw.exe"" """ & strPath & "\entry.py"""
intRet = WshShell.Run(strCmd, 0, False)
