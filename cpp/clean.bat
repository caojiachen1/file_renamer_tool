@echo off
echo Cleaning up old build files...

if exist "file_renamer_cli.exe" (
    echo Removing CLI executable: file_renamer_cli.exe
    del "file_renamer_cli.exe"
)

if exist "file_renamer.exe" (
    echo Removing root executable: file_renamer.exe
    del "file_renamer.exe"
)

if exist "*.o" (
    echo Removing object files...
    del "*.o"
)

if exist "*.obj" (
    echo Removing object files...
    del "*.obj"
)

if exist "build" (
    echo Cleaning build directory...
    rmdir /s /q "build"
)

echo Cleanup completed!
echo To rebuild the CLI executable, run:
echo   build.bat