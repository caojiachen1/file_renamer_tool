# File Renamer Tool

A batch file renaming tool that uses hash algorithms (MD5, SHA1, SHA256, SHA512, CRC32, BLAKE2B) to rename files. Available as CLI tools (C++ and Rust), a Tauri desktop application, and an iced native GUI.

## Project Structure

```
FileRenamerTool/
├── cpp/                    # C++ CLI implementation
│   ├── file_renamer_cli.cpp
│   ├── CMakeLists.txt
│   └── build.bat
├── rust/                   # Rust library + CLI
│   ├── Cargo.toml
│   ├── build.bat
│   └── src/
├── gui/                    # iced native GUI (Fluent Design)
│   ├── Cargo.toml
│   └── src/
├── src-tauri/              # Tauri desktop application
│   ├── src/
│   ├── Cargo.toml
│   └── tauri.conf.json
└── package.json
```

## CLI Versions

Both C++ and Rust versions provide identical functionality with the same command-line interface.

### C++ Version

Uses Windows CryptoAPI for hash computation. Built with Visual Studio and CMake.

**Requirements:**
- Windows 10/11
- Visual Studio 2022 with C++ development tools
- CMake 3.20 or later

**Build:**
```cmd
cd cpp
build.bat
```

### Rust Version

Uses SIMD-optimized hash libraries (md-5, sha1, sha2, crc32fast, blake2). Generally faster than the C++ version due to optimized crypto implementations.

**Requirements:**
- Windows 10/11
- Rust toolchain (https://rustup.rs/)

**Build:**
```cmd
cd rust
build.bat
```

### CLI Usage

Both versions share the same interface:

```cmd
file_renamer.exe <directory> [options]
```

**Options:**
- `-a, --algorithm <hash>`: Hash algorithm (MD5, SHA1, SHA256, SHA512, CRC32, BLAKE2B) [default: MD5]
- `-r, --recursive`: Scan subdirectories recursively
- `-e, --execute`: Execute renaming (default is preview mode)
- `-x, --extensions <ext>`: Only process files with specified extensions (e.g., jpg,png,txt)
- `-y, --yes`: Auto-confirm without user interaction
- `-t, --threads <n>`: Number of processing threads [default: auto-detect]
- `-b, --batch <n>`: Batch size for processing [default: auto-calculate]
- `--single-thread`: Use single-threaded processing
- `--multi-thread`: Use multi-threaded processing [default]
- `--batch-mode`: Use batch processing mode (best for large datasets)
- `-h, --help`: Show help message

**Examples:**
```cmd
# Preview all files with MD5
file_renamer.exe C:\MyFiles

# Preview only jpg and png files
file_renamer.exe C:\MyFiles -x jpg,png

# Execute renaming with SHA256, recursive
file_renamer.exe C:\MyFiles -a SHA256 -r -e

# Execute renaming for txt files using 8 threads
file_renamer.exe C:\MyFiles -e -x txt -t 8

# Auto-confirm execution
file_renamer.exe C:\MyFiles -e -y
```

## iced Native GUI

A native desktop application built with iced, featuring a Fluent Design dark theme.

**Features:**
- Native GPU-accelerated rendering
- Fluent Design dark theme (accent colors, rounded corners, clean typography)
- Real-time streaming output with scrollable text area
- All CLI options available in the GUI
- Direct library integration (no external process spawning)

**Build:**
```cmd
cd gui
cargo build --release
```

**Run:**
```cmd
gui\target\release\FileRenamer.exe
```

## Tauri Desktop Application

A desktop GUI built with Tauri 2, providing a visual interface for the CLI tool.

**Features:**
- Browse and select directories
- Choose hash algorithm and processing mode
- Configure thread count and batch size
- Real-time streaming output
- Start/stop processing

**Development:**
```cmd
npm install
npm run dev
```

**Build:**
```cmd
npm run build
```

The Tauri app looks for the CLI executable in the following order:
1. `file_renamer.exe` (project root)
2. `rust/target/release/file_renamer.exe`
3. `rust/target/debug/file_renamer.exe`
4. `cpp/build/Release/file_renamer_cli.exe`
5. `cpp/build/Debug/file_renamer_cli_d.exe`

## How It Works

1. **Scan**: The tool scans the specified directory for files
2. **Filter**: Applies extension filters if specified
3. **Hash**: Calculates the hash value using the selected algorithm
4. **Rename**: Renames files to `<hash>.<extension>` format

## Safety Features

- **Preview mode by default**: Shows what changes will be made without executing
- **Confirmation prompt**: Asks for confirmation before executing rename operations
- **Duplicate detection**: Warns if target filename already exists
- **Error handling**: Graceful error handling with detailed messages

## Error Reporting

The tool provides comprehensive error reporting for failed rename operations:

- **FILE_EXISTS**: Target filename already exists
- **FILE_READONLY**: Source file has read-only attribute
- **SOURCE_ACCESS_ERROR**: Permission denied or file locked
- **RENAME_FAILED_5**: Access denied
- **RENAME_FAILED_32**: File locked by another process
- **RENAME_FAILED_112**: Disk full
- **RENAME_FAILED_19**: Disk write-protected