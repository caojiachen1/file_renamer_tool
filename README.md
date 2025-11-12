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
   - By default uses ALL logical CPU threads for maximum throughput (you can override with `-t`)
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
- GPU single-file chunk (`--gpu-chunk-mb`): default 16 MB
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

Note: In ultra-fast mode, if you do not explicitly specify `-t`, the tool will default to using ALL logical CPU threads (`std::thread::hardware_concurrency()`).

Execute on GPU 0 with extreme tuning and SHA256:
```cmd
file_renamer.exe C:\MyFiles -a SHA256 -e -r -d 0 --ultra-fast --extreme -y
```

## Web Interface (Local)

The project includes a simple local Web UI for visual configuration and execution.

### Launch (Windows)

1. Run `start_web.bat` (double-click or from the command line):
   - Automatically creates a `.venv` virtual environment
   - Installs dependencies (Flask)
   - Starts the web server (default http://127.0.0.1:5000)
2. Open your browser at `http://127.0.0.1:5000`.
3. Fill in: directory path, algorithm, mode, device (auto / CPU / GPU id), recursive option, execute (default is preview), threads / batch size, extension filters, and optional advanced parameters.
4. Click "Run" to view streaming output; you can click "Stop" at any time.

Device list: Clicking the "Refresh" button triggers `-d list` and displays available device IDs.

### Executable Lookup Order

The web backend searches for an executable in the following order:

1. `file_renamer.exe`
2. `file_renamer_cli.exe`
3. `build/Release/file_renamer_cli.exe`
4. `build/Debug/file_renamer_cli_d.exe`

If none are found, an error will be shown in the UI and API response.

### Custom Host/Port

Default bind is 127.0.0.1:5000. To modify:

```cmd
set HOST=0.0.0.0
set PORT=6000
start_web.bat
```

Note: When exposing externally, ensure proper network environment and access control.

## Desktop Edition (Tauri)

In addition to the local Flask-based Web UI, you can package a native desktop application using Tauri (Windows executable with windowing, system dialogs, and automated packaging).

### Feature Mapping
The desktop app reuses the `web/static/index.html` frontend and auto-detects if it runs inside a Tauri environment:

- Directory selection: native folder chooser (Tauri dialog API)
- Device list: `list_devices` command invokes CLI `-d list`
- Start task: `start_run` command launches a child process and streams output via `cli-output` events
- Stop task: `stop_run` command terminates the child process
- Exit code: `cli-exit` event returns the process exit code

If not running in Tauri (e.g. still served by Flask), the frontend automatically falls back to the original HTTP endpoints without manual switching.

### Development Run (Windows)
Prerequisites:

1. Install Node.js (>=16) and npm
2. Install Rust (recommend rustup) and ensure `cargo` is on PATH
3. Build or have an existing executable: `file_renamer.exe` / `file_renamer_cli.exe` etc. (used for actual work)

Launch the development window:

1. `npm install` to install `@tauri-apps/cli`
2. `npm run dev` to start the Tauri development window (loading `web/static/index.html`)

### Build Release Package

```cmd
npm run build
```

The generated installer / executables are located under `src-tauri/target/release` (plus the `target` cache directory). If you need a WiX installer, install the WiX Toolset per Tauri docs; or adjust `bundle.targets` in `tauri.conf.json` to exclude the installer target.

### Structure Overview

```
src-tauri/
   Cargo.toml          # Rust project & Tauri dependencies
   tauri.conf.json     # Tauri config (distDir points to web/static)
   src/main.rs         # Backend command implementations (which_exe, list_devices, start_run, stop_run)
package.json          # Provides Tauri dev/build scripts
```

### Security Notes
The desktop edition directly invokes the local CLI and can access user-specified directories. Do not place untrusted external executables in the same directory where they could be mistakenly invoked.

