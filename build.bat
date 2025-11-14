@echo off
setlocal enabledelayedexpansion

echo ========================================
echo File Renamer Tool - Build
echo ========================================
echo.
echo Building file_renamer.exe ^(CPU version^)
echo Device options: -d cpu, -d auto
echo.

REM 检查是否安装了CMake
cmake --version >nul 2>&1
if errorlevel 1 (
    echo Error: CMake not found in PATH
    echo Please install CMake and add it to your PATH
    pause
    exit /b 1
)

REM 检查是否安装了Visual Studio 2022
where cl.exe >nul 2>&1
if errorlevel 1 (
    echo Error: Visual Studio 2022 not found in PATH
    echo Please run this from a Visual Studio 2022 Developer Command Prompt
    pause
    exit /b 1
)

REM 创建构建目录
if not exist "build" mkdir build

REM 进入构建目录
cd build
REM 构建（仅支持CPU）
echo.
echo ========================================
echo Building Release and Debug...
echo ========================================

echo Configuring project...
cmake .. -G "Visual Studio 17 2022" -A x64
if errorlevel 1 (
    echo Failed to configure project
    cd..
    pause
    exit /b 1
)

echo Building Release version...
cmake --build . --config Release
if errorlevel 1 (
    echo Failed to build Release version
    cd..
    pause
    exit /b 1
)

REM 也构建Debug版本
echo.
echo Building Debug version...
cmake --build . --config Debug
if errorlevel 1 (
    echo Warning: Failed to build Debug version
)

cd..

echo.
echo ========================================
echo Build completed successfully
echo ========================================
echo.

echo.
echo Executable generated:
if exist "build\Release\file_renamer_cli.exe" (
    echo   Release: build\Release\file_renamer_cli.exe
    echo   Copying executable to project root...
    
    copy "build\Release\file_renamer_cli.exe" "file_renamer.exe" >nul
    if errorlevel 1 (
        echo   Warning: Failed to copy executable to project root
    ) else (
        echo   Successfully copied to: file_renamer.exe
    )
) else (
    echo   Release: Not found
)

if exist "build\Debug\file_renamer_cli_d.exe" (
    echo   Debug: build\Debug\file_renamer_cli_d.exe
) else (
    echo   Debug: Not found
)

echo.
echo Usage examples:
echo   file_renamer.exe C:\MyFiles -d list                    ^(List devices^)
echo   file_renamer.exe C:\MyFiles -d auto                    ^(Auto-select device^)
echo   file_renamer.exe C:\MyFiles -d cpu                     ^(Force CPU processing^)
echo   file_renamer.exe C:\MyFiles -d -1                      ^(Force CPU processing^)
echo   file_renamer.exe C:\MyFiles -x jpg,png -e               ^(Process jpg/png files^)
echo.

echo Build Summary:
echo   Build Type: CPU-only
echo   Executable: file_renamer.exe
echo   Device Selection: cpu / auto / list / numeric ids
echo   Performance: High ^(optimized multi-threaded CPU pipeline^)
echo.

