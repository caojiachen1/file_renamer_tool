@echo off
setlocal enabledelayedexpansion

echo File Renamer Tool - Build

cmake --version >nul 2>&1
if errorlevel 1 (
    echo Error: CMake not found in PATH
    echo Please install CMake and add it to your PATH
    pause
    exit /b 1
)

where cl.exe >nul 2>&1
if errorlevel 1 (
    echo Error: Visual Studio 2022 not found in PATH
    echo Please run this from a Visual Studio 2022 Developer Command Prompt
    pause
    exit /b 1
)

if not exist "build" mkdir build

cd build

echo Configuring project...
cmake .. -G "Visual Studio 17 2022" -A x64
if errorlevel 1 (
    echo Failed to configure project
    cd ..
    pause
    exit /b 1
)

cmake --build . --config Release
if errorlevel 1 (
    echo Failed to build Release version
    cd ..
    pause
    exit /b 1
)

cd ..

echo Build completed successfully
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

