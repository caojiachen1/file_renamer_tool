cmake_minimum_required(VERSION 3.20)
project(FileRenamerTool VERSION 1.0.0 LANGUAGES CXX)

# Prefer PowerShell Core (pwsh) if available, otherwise fall back to Windows PowerShell
find_program(PWSH_EXECUTABLE NAMES pwsh pwsh.exe)
if(PWSH_EXECUTABLE)
    set(CMAKE_VS_MSBUILD_COMMAND "${PWSH_EXECUTABLE}")
else()
    set(CMAKE_VS_MSBUILD_COMMAND "powershell.exe")
endif()

# Require Visual Studio 2022 (MSVC v143) or later
if(NOT MSVC OR MSVC_VERSION LESS 1930)
    message(FATAL_ERROR "This project requires Visual Studio 2022 (MSVC v143) or later")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use UTF-8 for C++ source files to avoid encoding issues
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)


# Windows system libraries
set(WINDOWS_LIBS
    advapi32       # Windows CryptoAPI
)

# Common compile definitions
set(COMMON_DEFINITIONS
    _UNICODE 
    UNICODE
    WIN32_LEAN_AND_MEAN
    NOMINMAX
)

# Visual Studio 2022 specific compile options (C++ only)
set(MSVC_COMPILE_OPTIONS
    /W3          # Warning level 3
    /O2          # Optimization level 2 (Release)
    /MP          # Multi-processor compilation
    /permissive- # Strict conformance mode
    /Zc:__cplusplus # Enable updated __cplusplus macro
)

# Disable specific warnings (C++ only)
set(MSVC_DISABLE_WARNINGS
    /wd4819      # Disable encoding warning
    /wd4996      # Disable "unsafe function" warning
)

# =============================================================================
# CLI executable target
# =============================================================================

# Source files for the target
set(CLI_SOURCES
    file_renamer_cli.cpp
)


add_executable(file_renamer_cli ${CLI_SOURCES})

target_link_libraries(file_renamer_cli ${WINDOWS_LIBS})


target_compile_definitions(file_renamer_cli PRIVATE ${COMMON_DEFINITIONS})

# Apply MSVC options to C++ sources only
target_compile_options(file_renamer_cli PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:${MSVC_COMPILE_OPTIONS}>
    $<$<COMPILE_LANGUAGE:CXX>:${MSVC_DISABLE_WARNINGS}>
)

# Console application
set_target_properties(file_renamer_cli PROPERTIES
    WIN32_EXECUTABLE FALSE
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# =============================================================================
# Debug/Release configuration
# =============================================================================
# Debug configuration
set_target_properties(file_renamer_cli PROPERTIES
    DEBUG_POSTFIX "_d"
)

target_compile_definitions(file_renamer_cli PRIVATE 
    $<$<CONFIG:Debug>:_DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
)

# Disable optimization and enable debug info for Debug (C++ only)
target_compile_options(file_renamer_cli PRIVATE 
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:/Od /Zi>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/O2>
)

# =============================================================================
# Visual Studio project settings
# =============================================================================
# Set startup project to CLI target
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT file_renamer_cli)

# Put target under a solution folder
set_target_properties(file_renamer_cli PROPERTIES FOLDER "Console Applications")

# Build info
message(STATUS "=== FileRenamer Tool Build Configuration ===")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Targets:")
message(STATUS "  - file_renamer_cli (Console)")
message(STATUS "Visual Studio Version: ${MSVC_VERSION}")
message(STATUS "==============================================")