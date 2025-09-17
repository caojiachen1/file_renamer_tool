@echo off
setlocal enableextensions
cd /d "%~dp0"
chcp 65001 >nul

:: 强制 Python 使用 UTF-8（对 Python 3.7+ 有效），避免 subprocess/输出中文乱码
set PYTHONUTF8=1
:: 确保 Python 标准输入/输出/错误使用 UTF-8 编码
set PYTHONIOENCODING=utf-8

:: 提示：如果仍然看到中文乱码，请用编辑器将本文件另存为 "UTF-8 with BOM"（Windows cmd 在无 BOM 时有时无法正确识别）
:: 并在管理员模式下运行 cmd.exe；上述步骤通常能解决大部分中文显示问题。

@REM set VENV_DIR=.venv
@REM set PYTHON=python

@REM where %PYTHON% >nul 2>nul
@REM if errorlevel 1 (
@REM   echo 找不到 Python，请安装 Python 3.9+ 并确保在 PATH 中。
@REM   pause
@REM   exit /b 1
@REM )

@REM if not exist "%VENV_DIR%" (
@REM   echo [*] 正在创建虚拟环境...
@REM   %PYTHON% -m venv "%VENV_DIR%"
@REM   if errorlevel 1 (
@REM     echo 创建虚拟环境失败。
@REM     pause
@REM     exit /b 1
@REM   )
@REM )

@REM call "%VENV_DIR%\Scripts\activate.bat"
@REM if errorlevel 1 (
@REM   echo 激活虚拟环境失败。
@REM   pause
@REM   exit /b 1
@REM )

echo [*] 安装依赖...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo 依赖安装失败。请检查网络或以管理员身份运行本脚本。
  pause
  exit /b 1
)

echo [*] 启动 Web 服务...
set FLASK_ENV=production

:: 默认 HOST/PORT，可通过环境变量覆盖
if "%HOST%"=="" set HOST=127.0.0.1
if "%PORT%"=="" set PORT=5000

echo [*] 使用监听地址: http://%HOST%:%PORT%/

python web\server.py

endlocal
