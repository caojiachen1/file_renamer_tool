# File Renamer Tool - CLI Edition

A command-line utility for batch renaming files using hash algorithms (MD5, SHA1, SHA256, SHA512, CRC32, BLAKE2B). This tool helps organize files by renaming them with their hash values while preserving file extensions.

## Features

- **Hash-based renaming**: Rename files using multiple hash algorithms (MD5, SHA1, SHA256, SHA512, CRC32, BLAKE2B)
- **CPU-focused design**: Modern, highly-optimized multi-threaded CPU pipeline
- **Extension filtering**: Process only files with specific extensions
- **Recursive scanning**: Option to scan subdirectories
- **Preview mode**: See what changes will be made before executing
- **Multi-threading**: Optimized CPU processing with configurable thread count
- **Cross-platform**: Windows support with Visual Studio 2022

## Runtime Device

The current CLI only supports CPU execution with multi-threading.

## Requirements

- Windows 10/11
- Visual Studio 2022 with C++ development tools
- CMake 3.20 or later

## Quick Start

### Building

1. Clone or download this repository
2. Run the build script:
   ```cmd
   build.bat
   ```

### CLI Usage

Basic invocation:

```cmd
file_renamer_cli.exe <directory> [options]
```

Or, depending on your build output:

```cmd
file_renamer.exe <directory> [options]
```

#### Core Options

- `-a, --algorithm <hash>`: Hash algorithm (`MD5`, `SHA1`, `SHA256`, `SHA512`, `CRC32`, `BLAKE2B`), default is `MD5`
- `-r, --recursive`: Scan subdirectories recursively
- `-e, --execute`: Actually perform renaming (default is preview-only)
- `-x, --extensions <ext>`: Only process files with given extensions, e.g. `jpg,png,txt` or `.jpg,.png,.txt`
- `-q, --quick`: Enable quick check (skip files that already look correctly hash-named, default on)
- `--no-quick`: Disable quick check and force full hash calculation for all files
- `-y, --yes`: Auto-confirm without interactive prompts
- `-t, --threads <n>`: Number of worker threads; default is auto-detected from CPU cores
- `-b, --batch <n>`: Batch size (files per batch in batch mode); default is auto-calculated
- `--buffer-kb <n>`: Streaming I/O buffer size in KB; default is auto
- `--mmap-chunk-mb <n>`: Memory-mapped feed chunk size in MB; default is auto
- `--single-thread`: Use single-threaded processing (original sequential mode)
- `--multi-thread`: Use multi-threaded processing
- `--batch-mode`: Use batch processing mode (best for huge file sets)
<!-- `--extreme`: Extreme performance tuning (removed; kept as a backward-compatible no-op flag in CLI) -->
- `-h, --help`: Show help message

#### Examples

**Basic examples:**
```cmd
# Preview all files with MD5
file_renamer_cli.exe C:\MyFiles

# Preview only jpg and png files
file_renamer_cli.exe C:\MyFiles -x jpg,png

# Execute renaming with SHA256, recursive
file_renamer_cli.exe C:\MyFiles -a SHA256 -r -e

# Execute renaming for txt files using 8 threads
file_renamer_cli.exe C:\MyFiles -e -x txt -t 8

# Auto-confirm execution with quick check
file_renamer_cli.exe C:\MyFiles -e -y -q
```

## How It Works

1. **Device Selection**: Automatically handles device selection
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

When you don’t specify tuning flags, the tool picks aggressive, high-throughput defaults based on your hardware, and runs in ultra-fast mode by default (thread pool + CPU affinity):

- I/O buffer (`--buffer-kb`): 2–4 MB depending on system RAM (≥8GB → 4MB)
- mmap feed chunk (`--mmap-chunk-mb`): 8–16 MB depending on system RAM (≥8GB → 16MB)
- Threads (`-t`): up to 3× logical cores (capped at 64)
- Batch size (`-b`): 4 × threads

You can override any of these at runtime by passing the corresponding flag.

## Auto-tuned performance

The CLI automatically tunes threads, I/O buffer, and mmap chunk sizes based on system RAM. The deprecated `--extreme` flag is accepted for backward compatibility but no longer changes behavior; for more control, explicitly set:

- `-t, --threads <n>`
- `-b, --batch <n>`
- `--buffer-kb <n>`
- `--mmap-chunk-mb <n>`

Note: By default, if you do not specify `-t`, the tool uses `std::thread::hardware_concurrency()` threads (all logical CPU cores).

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

