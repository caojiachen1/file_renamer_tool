@echo off
setlocal enabledelayedexpansion

echo ========================================
echo File Renamer Tool - Unified Build
echo ========================================
echo.
echo Building unified executable with device selection support
echo Device options: -d cpu, -d auto, -d 0, -d 1, etc.
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

REM 检查是否支持CUDA
set CUDA_AVAILABLE=OFF
set BUILD_TYPE=Unified-CPU-only
where nvcc.exe >nul 2>&1
if not errorlevel 1 (
    echo CUDA compiler found, attempting unified build with CUDA support...
    set CUDA_AVAILABLE=ON
    set BUILD_TYPE=Unified-CUDA-enabled
) else (
    echo CUDA compiler not found, building unified CPU-only version...
)

REM 进入构建目录
cd build

REM 尝试CUDA构建（如果CUDA可用）
if "%CUDA_AVAILABLE%" == "ON" (
    echo.
    echo ========================================
    echo Building unified executable with CUDA support...
    echo ========================================
    
    echo Configuring project with CUDA support...
    cmake .. -G "Visual Studio 17 2022" -A x64 -DUSE_CUDA=ON
    if not errorlevel 1 (
        echo Building unified version with CUDA...
        cmake --build . --config Release
        if not errorlevel 1 (
            echo Unified CUDA build successful!
            set BUILD_TYPE=Unified-CUDA-enabled
            goto build_success
        ) else (
            echo CUDA build failed, falling back to CPU-only unified build...
        )
    ) else (
        echo CUDA configuration failed, falling back to CPU-only unified build...
    )
    
    REM 清理失败的CUDA构建
    echo Cleaning up failed CUDA build...
    cd ..
    rmdir /s /q build >nul 2>&1
    mkdir build
    cd build
    set BUILD_TYPE=Unified-CPU-only
)

REM 统一CPU构建（作为回退或默认选项）
echo.
echo ========================================
echo Building unified CPU-only version...
echo ========================================

echo Configuring project without CUDA...
cmake .. -G "Visual Studio 17 2022" -A x64 -DUSE_CUDA=OFF
if errorlevel 1 (
    echo Failed to configure unified CPU-only project
    cd..
    pause
    exit /b 1
)

echo Building unified CPU version...
cmake --build . --config Release
if errorlevel 1 (
    echo Failed to build unified CPU-only project
    cd..
    pause
    exit /b 1
)

:build_success
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
echo Unified Build completed successfully!
echo ========================================
echo.

if "%BUILD_TYPE%" == "Unified-CUDA-enabled" (
    echo Device Support: CPU + GPU (CUDA)
    echo   CPU Device: -1 ^(always available^)
    echo   GPU Devices: 0, 1, 2, ... ^(auto-detected^)
    echo   GPU acceleration for large files ^(^>4KB^)
    echo   Supported GPU algorithms: MD5, SHA1, SHA256, CRC32
) else (
    echo Device Support: CPU only
    echo   CPU Device: -1 ^(always available^)
    echo   GPU acceleration: Not available ^(CUDA not found^)
    echo   All device selections will use CPU processing
)

echo.
echo Unified executable generated:
if exist "build\Release\file_renamer_cli.exe" (
    echo   Release: build\Release\file_renamer_cli.exe
    echo   Copying unified executable to project root...
    
    copy "build\Release\file_renamer_cli.exe" "file_renamer.exe" >nul
    if errorlevel 1 (
        echo   Warning: Failed to copy unified executable to project root
    ) else (
        echo   Successfully copied to: file_renamer.exe
        echo   ^(Unified version with device selection support^)
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
echo   file_renamer.exe C:\MyFiles -d list                    ^(List all devices^)
echo   file_renamer.exe C:\MyFiles -d auto                    ^(Auto-select best device^)
echo   file_renamer.exe C:\MyFiles -d cpu                     ^(Force CPU processing^)
echo   file_renamer.exe C:\MyFiles -d -1                      ^(Force CPU processing^)
if "%BUILD_TYPE%" == "Unified-CUDA-enabled" (
    echo   file_renamer.exe C:\MyFiles -d 0                       ^(Use GPU device 0^)
    echo   file_renamer.exe C:\MyFiles -d 1 -a SHA256             ^(Use GPU device 1 with SHA256^)
)
echo   file_renamer.exe C:\MyFiles -x jpg,png -e               ^(Process jpg/png files^)
echo.

echo Build Summary:
echo   Build Type: %BUILD_TYPE%
echo   Executable: file_renamer.exe ^(unified version^)
if "%BUILD_TYPE%" == "Unified-CUDA-enabled" (
    echo   CUDA Support: Available and enabled
    echo   Device Selection: CPU ^(-1^) + GPU ^(0, 1, 2, ...^)
    echo   Performance: High ^(GPU acceleration for large files^)
) else (
    echo   CUDA Support: Not available
    echo   Device Selection: CPU only ^(-1^)
    echo   Performance: Standard ^(CPU-only processing^)
)
echo   Device Parameter: Use -d ^<device^> to specify processing device
echo.

