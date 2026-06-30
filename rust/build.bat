@echo off
setlocal enabledelayedexpansion

echo File Renamer Tool - Rust CLI Build

where cargo >nul 2>&1
if errorlevel 1 (
    echo Error: Rust/cargo not found in PATH
    echo Please install Rust from https://rustup.rs/
    pause
    exit /b 1
)

echo Building Rust CLI in release mode...
cargo build --release
if errorlevel 1 (
    echo Failed to build Rust CLI
    pause
    exit /b 1
)

echo Build completed successfully
echo Executable: target\release\file_renamer.exe

echo Copying to project root...
copy "target\release\file_renamer.exe" "..\file_renamer.exe" >nul
if errorlevel 1 (
    echo   Warning: Failed to copy to root
) else (
    echo   Successfully copied to: ..\file_renamer.exe
)

echo Copying to src-tauri\binaries\...
copy "target\release\file_renamer.exe" "..\src-tauri\binaries\file_renamer-x86_64-pc-windows-msvc.exe" >nul
if errorlevel 1 (
    echo   Warning: Failed to copy to binaries
) else (
    echo   Successfully copied to: ..\src-tauri\binaries\file_renamer-x86_64-pc-windows-msvc.exe
)
