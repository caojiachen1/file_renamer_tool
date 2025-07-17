@echo off
setlocal enabledelayedexpansion

echo ========================================
echo File Renamer Tool - CLI Only Build
echo ========================================
echo.

REM 检查是否安装了CMake
cmake --version >nul 2>&1
if errorlevel 1 (
    echo Error: CMake not found in PATH
    echo Please install CMake and add it to your PATH
    exit /b 1
)

REM 创建构建目录
if not exist "build" mkdir build

REM 进入构建目录
cd build

echo Configuring project with CMake...
cmake .. -G "Visual Studio 17 2022" -A x64
if errorlevel 1 (
    echo Failed to configure project
    cd..
    exit /b 1
)

echo.
echo Building CLI version...
cmake --build . --config Release
if errorlevel 1 (
    echo Failed to build project
    cd..
    exit /b 1
)

echo.
echo Building Debug version...
cmake --build . --config Debug
if errorlevel 1 (
    echo Failed to build Debug version
    cd..
    exit /b 1
)

cd..

echo.
echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo Executables generated:
if exist "build\Release\file_renamer_cli.exe" (
    echo   Release: build\Release\file_renamer_cli.exe
    echo   Copying Release executable to project root...
    copy "build\Release\file_renamer_cli.exe" "file_renamer_cli.exe" >nul
    if errorlevel 1 (
        echo   Warning: Failed to copy Release executable to project root
    ) else (
        echo   Successfully copied to: file_renamer_cli.exe
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
echo Usage example:
echo   build\Release\file_renamer_cli.exe C:\MyFiles -x jpg,png -e
echo.

