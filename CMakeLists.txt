cmake_minimum_required(VERSION 3.20)
project(FileRenamerTool VERSION 1.0.0 LANGUAGES CXX)

# 要求使用Visual Studio 2022 (MSVC v143)
if(NOT MSVC OR MSVC_VERSION LESS 1930)
    message(FATAL_ERROR "This project requires Visual Studio 2022 (MSVC v143) or later")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编码为UTF-8，避免中文编码问题
add_compile_options(/utf-8)

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

# Visual Studio 2022 特定编译选项
set(MSVC_COMPILE_OPTIONS
    /W3          # Warning level 3
    /O2          # Optimization level 2 (Release)
    /MP          # Multi-processor compilation
    /permissive- # Strict conformance mode
    /Zc:__cplusplus # Enable updated __cplusplus macro
)

# 禁用特定警告
set(MSVC_DISABLE_WARNINGS
    /wd4819      # 禁用编码警告
    /wd4996      # 禁用不安全函数警告
)

# =============================================================================
# CLI版本可执行文件
# =============================================================================
add_executable(file_renamer_cli file_renamer_cli.cpp)

target_link_libraries(file_renamer_cli ${WINDOWS_LIBS})

target_compile_definitions(file_renamer_cli PRIVATE ${COMMON_DEFINITIONS})

target_compile_options(file_renamer_cli PRIVATE 
    ${MSVC_COMPILE_OPTIONS}
    ${MSVC_DISABLE_WARNINGS}
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

# Debug模式下禁用优化，启用调试信息
target_compile_options(file_renamer_cli PRIVATE 
    $<$<CONFIG:Debug>:/Od /Zi>
    $<$<CONFIG:Release>:/O2>
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
message(STATUS "Targets:")
message(STATUS "  - file_renamer_cli (Console)")
message(STATUS "Visual Studio Version: ${MSVC_VERSION}")
message(STATUS "==============================================")
