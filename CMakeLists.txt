cmake_minimum_required(VERSION 3.20)
project(FileRenamerTool VERSION 1.0.0 LANGUAGES CXX)

# Require Visual Studio 2022 (MSVC v143) or later
if(NOT MSVC OR MSVC_VERSION LESS 1930)
    message(FATAL_ERROR "This project requires Visual Studio 2022 (MSVC v143) or later")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use UTF-8 for C++ source files to avoid encoding issues
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)

# CUDA support (optional)
option(USE_CUDA "Enable CUDA GPU acceleration" OFF)

if(USE_CUDA)
    # Enable CUDA language
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    # CUDA settings
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)  # Disable separable compilation to avoid command line issues
    
    message(STATUS "CUDA support enabled")
    message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA Runtime Library: ${CUDAToolkit_LIBRARY_DIR}")
endif()

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

# Add CUDA source file when CUDA is enabled
if(USE_CUDA)
    list(APPEND CLI_SOURCES cuda_hash.cu)
endif()

add_executable(file_renamer_cli ${CLI_SOURCES})

target_link_libraries(file_renamer_cli ${WINDOWS_LIBS})

# Link CUDA libraries when enabled
if(USE_CUDA)
    target_link_libraries(file_renamer_cli CUDA::cudart)
    target_compile_definitions(file_renamer_cli PRIVATE USE_CUDA)
    
    # Auto-select CUDA architectures:
    # - CUDA 13.0 and above no longer support < sm_80, so enable 80+ only (including 120)
    # - Older CUDA versions progressively add newer archs for broader compatibility
    set(CUDA_ARCHS "")
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
        # CUDA 13 removed support for < sm_80
        set(CUDA_ARCHS "80;86;87;89;90;120")
    else()
        # Preload older archs to support CUDA 10.x-era devices (may increase build time)
        set(CUDA_ARCHS "60;61;70;75")
        # Append newer archs based on CUDA version support
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.0")
            list(APPEND CUDA_ARCHS "80")      # Ampere SM80
        endif()
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.1")
            list(APPEND CUDA_ARCHS "86")      # Ampere SM86 (GA10x)
        endif()
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.4")
            list(APPEND CUDA_ARCHS "87")      # Ampere SM87 (Orin)
        endif()
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.8")
            list(APPEND CUDA_ARCHS "89")      # Ada SM89 (RTX 40)
        endif()
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0")
            list(APPEND CUDA_ARCHS "90")      # Hopper SM90 (H100)
        endif()
        # Blackwell SM120 is only available in newer toolchains; avoid setting unknown arch on old CUDA
        # Therefore, SM120 is appended only for 13.0+ in the branch above
    endif()

    message(STATUS "Selected CUDA architectures: ${CUDA_ARCHS}")

    # CUDA-specific target properties
    set_target_properties(file_renamer_cli PROPERTIES 
        CUDA_ARCHITECTURES "${CUDA_ARCHS}"
        CUDA_RUNTIME_LIBRARY Static
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    )

    # 60,61: Pascal architecture (GTX 10 series, Titan X)
    # 70: Volta architecture (Titan V, V100)
    # 75: Turing architecture (GTX 16 series, RTX 20 series)
    # 80,86: Ampere architecture (RTX 30 series, A100)
    # 87: Ampere architecture (Jetson AGX Orin)
    # 89: Ada Lovelace architecture (RTX 40 series)
    # 90: Hopper architecture (H100)
    # 120: Blackwell architecture (newer CUDA toolchains)
endif()

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
if(USE_CUDA)
    message(STATUS "CUDA: Enabled (${CUDAToolkit_VERSION})")
else()
    message(STATUS "CUDA: Disabled")
endif()
message(STATUS "Targets:")
message(STATUS "  - file_renamer_cli (Console)")
message(STATUS "Visual Studio Version: ${MSVC_VERSION}")
message(STATUS "==============================================")

# Build guidance for users
if(NOT USE_CUDA)
    message(STATUS "")
    message(STATUS "To enable CUDA GPU acceleration:")
    message(STATUS "  cmake -DUSE_CUDA=ON ..")
    message(STATUS "  or configure with CMake GUI")
    message(STATUS "")
endif()