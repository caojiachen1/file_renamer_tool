# File Renamer Tool - CLI Edition

A command-line utility for batch renaming files using hash algorithms (MD5, SHA1). This tool helps organize files by renaming them with their hash values while preserving file extensions.

## Features

- **Hash-based renaming**: Rename files using MD5 or SHA1 hashes
- **Extension filtering**: Process only files with specific extensions
- **Recursive scanning**: Option to scan subdirectories
- **Preview mode**: See what changes will be made before executing
- **Cross-platform**: Windows support with Visual Studio 2022

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

### Usage

Basic usage:
```cmd
file_renamer_cli.exe <directory> [options]
```

#### Options

- `-a, --algorithm <hash>` - Hash algorithm (MD5, SHA1) [default: MD5]
- `-r, --recursive` - Scan subdirectories recursively
- `-e, --execute` - Execute renaming (default is preview mode)
- `-x, --extensions <ext>` - Only process files with specified extensions
- `-h, --help` - Show help message

#### Examples

Preview all files with MD5:
```cmd
file_renamer_cli.exe C:\MyFiles
```

Preview only jpg and png files:
```cmd
file_renamer_cli.exe C:\MyFiles -x jpg,png
```

Execute renaming for txt and log files with SHA1, recursive:
```cmd
file_renamer_cli.exe C:\MyFiles -a SHA1 -r -x .txt,.log -e
```

Execute renaming for jpg files only:
```cmd
file_renamer_cli.exe C:\MyFiles -e -x jpg
```

## How It Works

1. **Scan**: The tool scans the specified directory for files
2. **Filter**: Applies extension filters if specified
3. **Hash**: Calculates the hash value for each file
4. **Rename**: Renames files to `<hash>.<extension>` format

## Safety Features

- **Preview mode by default**: Shows what changes will be made without executing
- **Confirmation prompt**: Asks for confirmation before executing rename operations
- **Duplicate detection**: Warns if target filename already exists
- **Error handling**: Graceful error handling with detailed messages

## File Processing

The tool processes files as follows:
1. Original: `document.pdf`
2. Renamed: `d41d8cd98f00b204e9800998ecf8427e.pdf` (MD5 hash + original extension)