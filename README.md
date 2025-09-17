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
- `-t, --threads <n>` - Number of processing threads [default: auto]
- `-q, --quick` - Enable quick check for already-named files
- `--ultra-fast` - Enable ultra-fast pipeline (aggressive I/O + scheduling)
   - 默认会使用机器的全部逻辑CPU线程以获得最大吞吐（可用 `-t` 手动指定覆盖）
- `--extreme` - Extreme performance mode (very aggressive auto-tuning; higher RAM/VRAM & I/O usage)
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

## Auto-tuned Defaults (Fastest by default)

When you don’t specify tuning flags, the tool picks aggressive, high-throughput defaults based on your hardware:

- I/O buffer (`--buffer-kb`): 2–4 MB depending on system RAM (≥8GB → 4MB)
- mmap feed chunk (`--mmap-chunk-mb`): 8–16 MB depending on system RAM (≥8GB → 16MB)
- Threads (`-t`): up to 3× logical cores (capped at 64)
- Batch size (`-b`): 4 × threads
- GPU per-file in-memory cap (`--gpu-file-cap-mb`): ~1/16 VRAM (128–1024 MB)
- GPU mini-batch cap (`--gpu-batch-bytes-mb`): ~1/4 VRAM (512–8192 MB)
- GPU single-file chunk (`--gpu-chunk-mb`): 默认 16 MB
- GPU min size (`--gpu-min-kb`): 4 KB

You can override any of these at runtime by passing the corresponding flag.

## Extreme mode (--extreme)

For maximum throughput, add `--extreme`. This applies an even more aggressive tuning profile on top of the auto defaults:

- Threads: up to 4× logical cores (capped at 96)
- Batch size: 6 × threads
- I/O buffer (`--buffer-kb`): 4–8 MB (RAM-dependent)
- mmap feed chunk (`--mmap-chunk-mb`): 16–32 MB (RAM-dependent)
- GPU per-file in-memory cap (`--gpu-file-cap-mb`): ~1/8 VRAM (256–2048 MB)
- GPU mini-batch cap (`--gpu-batch-bytes-mb`): ~1/2 VRAM (1024–12288 MB)
- GPU single-file chunk (`--gpu-chunk-mb`): 32 MB

Notes:
- `--extreme` increases memory and I/O pressure; ensure your system has sufficient RAM/VRAM and fast storage.
- Any explicit flag you pass (e.g., `-t`, `--buffer-kb`, `--gpu-chunk-mb`) overrides the auto/`--extreme` values.

### Examples

Preview, ultra-fast + extreme, auto device selection:
```cmd
file_renamer.exe C:\MyFiles --ultra-fast --extreme -d auto -q -r
```

Note: Ultra-fast 模式在未显式指定 `-t` 时将默认使用机器全部逻辑线程数（`std::thread::hardware_concurrency()`）。

Execute on GPU 0 with extreme tuning and SHA256:
```cmd
file_renamer.exe C:\MyFiles -a SHA256 -e -r -d 0 --ultra-fast --extreme -y
```

## Web 界面（本地）

项目包含一个简洁的本地 Web UI，方便可视化配置与运行：

### 启动（Windows）

1. 运行 `start_web.bat`（双击或在命令行）。
   - 自动创建 `.venv` 虚拟环境
   - 安装依赖（Flask）
   - 启动 Web 服务（默认 http://127.0.0.1:5000）
2. 打开浏览器访问 `http://127.0.0.1:5000`。
3. 在页面中填写：目录路径、算法、模式、设备（auto/CPU/GPU id），是否递归，是否执行（默认预览）、线程/批大小、扩展名过滤，以及可选高级参数。
4. 点击“运行”查看流式输出；可随时“停止”。

设备列表：点击页面“刷新”按钮将调用 `-d list` 并展示可用设备 ID。

### 可执行文件查找顺序

Web 后端会按顺序寻找下列可执行文件：

1. `file_renamer.exe`
2. `file_renamer_cli.exe`
3. `build/Release/file_renamer_cli.exe`
4. `build/Debug/file_renamer_cli_d.exe`

若均不存在，会在界面与接口报错提示。

### 自定义监听

默认监听 127.0.0.1:5000。若要修改：

```cmd
set HOST=0.0.0.0
set PORT=6000
start_web.bat
```

注意：对外开放时请注意网络环境与权限控制。
