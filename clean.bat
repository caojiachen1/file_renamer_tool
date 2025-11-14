@echo off
echo Cleaning up old build files...
echo ================================

REM 删除旧的分离版本可执行文件
if exist "file_renamer_cli.exe" (
    echo Removing old CPU-only executable: file_renamer_cli.exe
    del "file_renamer_cli.exe"
)


REM 删除测试用的可执行文件
if exist "file_renamer_cli_test.exe" (
    echo Removing test executable: file_renamer_cli_test.exe
    del "file_renamer_cli_test.exe"
)

REM 删除编译的临时文件
if exist "*.o" (
    echo Removing object files...
    del "*.o"
)

if exist "*.obj" (
    echo Removing object files...
    del "*.obj"
)

REM 清理build目录
if exist "build" (
    echo Cleaning build directory...
    rmdir /s /q "build"
)

echo.
echo Cleanup completed!
echo.
echo To build the unified executable, run:
echo   build.bat
echo.
echo This will create file_renamer.exe with unified device selection support.
