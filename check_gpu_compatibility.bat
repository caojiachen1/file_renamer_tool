@echo off
echo FileRenamer Tool - GPU Compatibility Checker
echo =============================================
echo.

REM Check if CUDA is available
where nvcc.exe >nul 2>&1
if errorlevel 1 (
    echo [ERROR] CUDA compiler nvcc not found in PATH
    echo Please install CUDA Toolkit 11.0 or higher
    echo Download from: https://developer.nvidia.com/cuda-toolkit
    goto end
) else (
    echo [OK] CUDA compiler found
)

REM Check if nvidia-smi is available
where nvidia-smi.exe >nul 2>&1
if errorlevel 1 (
    echo [ERROR] nvidia-smi not found - NVIDIA drivers may not be installed
    echo Please install the latest NVIDIA drivers
    goto end
) else (
    echo [OK] NVIDIA drivers found
)

echo.
echo Detecting GPU information...
echo ==========================================

REM Run nvidia-smi to get GPU info
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader,nounits 2>nul
if errorlevel 1 (
    echo [ERROR] Failed to query GPU information
    echo Make sure NVIDIA GPU is properly installed and drivers are up to date
    goto end
)

echo.
echo Supported Compute Capabilities:
echo ==========================================
echo 5.0-5.2: Maxwell (GTX 900 series)
echo 6.0-6.1: Pascal (GTX 10 series, Titan X)
echo 7.0:     Volta (Titan V, Tesla V100)
echo 7.5:     Turing (GTX 16 series, RTX 20 series)
echo 8.0-8.7: Ampere (RTX 30 series, A100, Jetson AGX Orin)
echo 8.9:     Ada Lovelace (RTX 40 series)
echo 9.0:     Blackwell (RTX 50 series)
echo.

echo Your GPU must have compute capability 5.0 or higher to use GPU acceleration.
echo.

REM Check CUDA version
nvcc --version | findstr "release"
echo.

echo Performance Recommendations:
echo ==========================================
echo - RTX 50 series: Best performance
echo - RTX 40 series: Excellent performance  
echo - RTX 30 series: Good performance
echo - RTX 20 series: Moderate performance
echo - GTX 16/10 series: Basic performance
echo - GTX 900 series: Minimum supported
echo.

:end
echo.
echo To build with GPU acceleration: build.bat
echo To build CPU-only version: build.bat
echo.
pause
