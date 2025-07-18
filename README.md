# File Renamer Tool - Unified CLI Edition

A command-line utility for batch renaming files using hash algorithms (MD5, SHA1, SHA256, SHA512, CRC32, BLAKE2B). This tool helps organize files by renaming them with their hash values while preserving file extensions. Features unified CPU/GPU processing with intelligent device selection.

## Features

- **Hash-based renaming**: Rename files using multiple hash algorithms (MD5, SHA1, SHA256, SHA512, CRC32, BLAKE2B)
- **Unified Device Selection**: Single executable with intelligent CPU/GPU switching via `--device` parameter
- **GPU Acceleration**: CUDA GPU support for faster hash calculation on large files
- **Extension filtering**: Process only files with specific extensions
- **Recursive scanning**: Option to scan subdirectories
- **Preview mode**: See what changes will be made before executing
- **Multi-threading**: Optimized CPU processing with configurable thread count
- **Cross-platform**: Windows support with Visual Studio 2022
- **Wide GPU Support**: Maxwell, Pascal, Volta, Turing, Ampere, Ada Lovelace, Blackwell architectures

## Device Selection

The unified executable `file_renamer.exe` supports intelligent device selection:

- **CPU Processing**: `-d cpu` or `-d -1` (always available)
- **Auto Selection**: `-d auto` (default - chooses best available device)
- **GPU Processing**: `-d 0`, `-d 1`, etc. (specific GPU devices)
- **Device Discovery**: `-d list` (shows all available devices)

## Requirements

- Windows 10/11
- Visual Studio 2022 with C++ development tools
- CMake 3.20 or later
- **Optional**: NVIDIA GPU with CUDA support (GTX 900 series or newer) for GPU acceleration

## GPU Support

This tool supports CUDA GPU acceleration for faster hash computation on large files:

- **Supported GPUs**: NVIDIA GTX 900 series and newer (compute capability 5.0+)
- **Performance**: 2-10x faster hash calculation for large files
- **Auto-fallback**: Automatically uses CPU if GPU is unavailable
- **Supported Architectures**: Maxwell, Pascal, Volta, Turing, Ampere, Ada Lovelace, Blackwell

To check GPU compatibility:
```cmd
check_gpu_compatibility.bat
```

## Quick Start

### Building

#### Unified Build (Recommended)
1. Clone or download this repository
2. Check GPU compatibility (optional):
   ```cmd
   check_gpu_compatibility.bat
   ```
3. Build unified executable:
   ```cmd
   build.bat
   ```
   This creates `file_renamer.exe` with automatic CUDA support detection.

#### Option 2: CPU-Only Version
1. Clone or download this repository
2. Run the build script:
   ```cmd
   build.bat
   ```

**Note**: The GPU version automatically falls back to CPU processing if no compatible GPU is found.

### Usage

The unified `file_renamer.exe` supports intelligent device selection:

```cmd
file_renamer.exe <directory> [options]
```

#### Device Selection Options

- `-d, --device <specification>` - Choose processing device:
  - `auto` - Auto-select best available device (default)
  - `cpu` or `-1` - Force CPU processing
  - `0`, `1`, `2`, ... - Use specific GPU device
  - `list` - Show all available devices

#### Core Options

- `-a, --algorithm <hash>` - Hash algorithm (MD5, SHA1, SHA256, SHA512, CRC32, BLAKE2B) [default: MD5]
- `-r, --recursive` - Scan subdirectories recursively
- `-e, --execute` - Execute renaming (default is preview mode)
- `-x, --extensions <ext>` - Only process files with specified extensions
- `-t, --threads <n>` - Number of processing threads [default: auto-detect]
- `-q, --quick` - Enable quick check for already-named files
- `-y, --yes` - Auto-confirm without user interaction
- `-h, --help` - Show help message

#### Examples

**Device Selection:**
```cmd
# List all available devices
file_renamer.exe C:\MyFiles -d list

# Auto-select best device (default)
file_renamer.exe C:\MyFiles

# Force CPU processing
file_renamer.exe C:\MyFiles -d cpu
file_renamer.exe C:\MyFiles -d -1

# Use specific GPU device
file_renamer.exe C:\MyFiles -d 0
file_renamer.exe C:\MyFiles -d 1 -a SHA256
```

**Basic Operations:**
```cmd
# Preview all files with MD5
file_renamer.exe C:\MyFiles

# Preview only jpg and png files
file_renamer.exe C:\MyFiles -x jpg,png

# Execute renaming with SHA256, recursive, using GPU device 0
file_renamer.exe C:\MyFiles -a SHA256 -r -e -d 0

# Execute renaming for txt files using CPU, 8 threads
file_renamer.exe C:\MyFiles -e -x txt -d cpu -t 8

# Auto-confirm execution with quick check
file_renamer.exe C:\MyFiles -e -y -q
```

## How It Works

1. **Device Selection**: Automatically detects available devices (CPU + GPU)
2. **Scan**: The tool scans the specified directory for files
3. **Filter**: Applies extension filters if specified
4. **Hash**: Calculates the hash value using selected device (CPU or GPU)
5. **Rename**: Renames files to `<hash>.<extension>` format

## Safety Features

- **Preview mode by default**: Shows what changes will be made without executing
- **Confirmation prompt**: Asks for confirmation before executing rename operations
- **Duplicate detection**: Warns if target filename already exists
- **Error handling**: Graceful error handling with detailed messages

## File Processing

The tool processes files as follows:
1. Original: `document.pdf`
2. Renamed: `d41d8cd98f00b204e9800998ecf8427e.pdf` (MD5 hash + original extension)