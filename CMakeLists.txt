cmake_minimum_required(VERSION 3.20)
project(FileRenamerTool VERSION 1.0.0 LANGUAGES CXX)

# 要求使用Visual Studio 2022 (MSVC v143)
if(NOT MSVC OR MSVC_VERSION LESS 1930)
    message(FATAL_ERROR "This project requires Visual Studio 2022 (MSVC v143) or later")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编码为UTF-8，避免中文编码问题，仅对C++源文件生效
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
    set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)  # 关闭可分离编译以避免命令行问题
    
    message(STATUS "CUDA support enabled")
    message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA Runtime Library: ${CUDAToolkit_LIBRARY_DIR}")
endif()

# Windows系统库
set(WINDOWS_LIBS
    advapi32       # Windows CryptoAPI
)

# 通用编译定义
set(COMMON_DEFINITIONS
    _UNICODE 
    UNICODE
    WIN32_LEAN_AND_MEAN
    NOMINMAX
)

# Visual Studio 2022 特定编译选项 (仅用于C++源文件)
set(MSVC_COMPILE_OPTIONS
    /W3          # Warning level 3
    /O2          # Optimization level 2 (Release)
    /MP          # Multi-processor compilation
    /permissive- # Strict conformance mode
    /Zc:__cplusplus # Enable updated __cplusplus macro
)

# 禁用特定警告 (仅用于C++源文件)
set(MSVC_DISABLE_WARNINGS
    /wd4819      # 禁用编码警告
    /wd4996      # 禁用不安全函数警告
)

# =============================================================================
# CLI版本可执行文件
# =============================================================================

# 创建目标源文件列表
set(CLI_SOURCES
    file_renamer_cli.cpp
)

# 如果启用CUDA，添加CUDA源文件
if(USE_CUDA)
    list(APPEND CLI_SOURCES cuda_hash.cu)
endif()

add_executable(file_renamer_cli ${CLI_SOURCES})

target_link_libraries(file_renamer_cli ${WINDOWS_LIBS})

# 如果启用CUDA，链接CUDA库
if(USE_CUDA)
    target_link_libraries(file_renamer_cli CUDA::cudart)
    target_compile_definitions(file_renamer_cli PRIVATE USE_CUDA)
    
    # CUDA特定编译选项 - 支持多种GPU架构
    # 包括从Maxwell到最新Blackwell架构的广泛GPU支持
    set_target_properties(file_renamer_cli PROPERTIES 
        CUDA_ARCHITECTURES "50;52;60;61;70;75;80;86;87;89;90"  # 支持Maxwell到Blackwell架构
        CUDA_RUNTIME_LIBRARY Static
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    )
    
    # 为不同架构添加编译注释说明
    # 50,52: Maxwell架构 (GTX 900系列)
    # 60,61: Pascal架构 (GTX 10系列, Titan X)  
    # 70: Volta架构 (Titan V, V100)
    # 75: Turing架构 (GTX 16系列, RTX 20系列)
    # 80,86: Ampere架构 (RTX 30系列, A100)
    # 87: Ampere架构 (Jetson AGX Orin)
    # 89: Ada Lovelace架构 (RTX 40系列)
    # 90: Blackwell架构 (RTX 50系列)
endif()

target_compile_definitions(file_renamer_cli PRIVATE ${COMMON_DEFINITIONS})

# 仅对C++源文件应用MSVC编译选项
target_compile_options(file_renamer_cli PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:${MSVC_COMPILE_OPTIONS}>
    $<$<COMPILE_LANGUAGE:CXX>:${MSVC_DISABLE_WARNINGS}>
)

# 设置为控制台应用程序
set_target_properties(file_renamer_cli PROPERTIES
    WIN32_EXECUTABLE FALSE
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# =============================================================================
# Debug/Release 配置
# =============================================================================
# Debug配置
set_target_properties(file_renamer_cli PROPERTIES
    DEBUG_POSTFIX "_d"
)

target_compile_definitions(file_renamer_cli PRIVATE 
    $<$<CONFIG:Debug>:_DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
)

# Debug模式下禁用优化，启用调试信息 (仅C++文件)
target_compile_options(file_renamer_cli PRIVATE 
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:/Od /Zi>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/O2>
)

# =============================================================================
# Visual Studio 项目设置
# =============================================================================
# 设置启动项目为CLI版本
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT file_renamer_cli)

# 设置解决方案文件夹
set_target_properties(file_renamer_cli PROPERTIES FOLDER "Console Applications")

# 输出信息
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

# 为用户提供构建指导
if(NOT USE_CUDA)
    message(STATUS "")
    message(STATUS "To enable CUDA GPU acceleration:")
    message(STATUS "  cmake -DUSE_CUDA=ON ..")
    message(STATUS "  or configure with CMake GUI")
    message(STATUS "")
endif()