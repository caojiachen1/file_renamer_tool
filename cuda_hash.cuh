#ifndef CUDA_HASH_CUH
#define CUDA_HASH_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <string>
#include <vector>
#include <mutex>

// CUDA error checking helpers (return boolean; callers decide how to handle)
inline bool CUDA_CHECK(cudaError_t call, const char* file, int line) {
    if (call != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d - %s\n", file, line, cudaGetErrorString(call));
        return false;
    }
    return true;
}

inline bool CUDA_CHECK_KERNEL(const char* file, int line) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA kernel launch error at %s:%d - %s\n", file, line, cudaGetErrorString(error));
        return false;
    }
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA kernel execution error at %s:%d - %s\n", file, line, cudaGetErrorString(error));
        return false;
    }
    return true;
}

// Variant that only checks launch errors, without device-wide synchronization.
inline bool CUDA_CHECK_KERNEL_NOSYNC(const char* file, int line) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA kernel launch error at %s:%d - %s\n", file, line, cudaGetErrorString(error));
        return false;
    }
    return true;
}

// Convenience macros to capture file/line automatically
#define CUDA_TRY(call) CUDA_CHECK((call), __FILE__, __LINE__)
#define CUDA_TRY_KERNEL() CUDA_CHECK_KERNEL(__FILE__, __LINE__)
#define CUDA_TRY_KERNEL_NOSYNC() CUDA_CHECK_KERNEL_NOSYNC(__FILE__, __LINE__)

// GPU device information structure
struct GPUInfo {
    int deviceCount;
    std::vector<cudaDeviceProp> devices;
    int selectedDevice;
    bool isAvailable;
    
    GPUInfo() : deviceCount(0), selectedDevice(0), isAvailable(false) {}
};

// CUDA Hash Calculator class
class CudaHashCalculator {
private:
    static bool initialized;
    static GPUInfo gpuInfo;
    static bool isCudaAvailable;
    static std::mutex s_mutex; // serialize access to persistent CUDA resources
    
    // GPU memory pools for better performance
    static void* d_input_pool;
    static void* d_output_pool;
    static size_t pool_size;

    // Persistent resources for batch processing
    static cudaStream_t s_stream1;
    static cudaStream_t s_stream2;
    static cudaEvent_t  s_evtPtrReady; // persistent event to fence pointer/length copies
    static unsigned char** s_d_inputs;   // device array of input pointers
    static unsigned int**  s_d_outputs;  // device array of output pointers
    static size_t*         s_d_lengths;  // device array of lengths
    static int             s_ptrCapacity; // capacity of pointer arrays
        // Persistent batch buffers (host pinned and device) for packed inputs/outputs
        static unsigned char*  s_h_all_inputs;   // pinned host buffer for all inputs in a batch
        static unsigned int*   s_h_all_outputs;  // pinned host buffer for all digests in a batch
        static unsigned char*  s_d_all_inputs;   // device buffer for all inputs in a batch
        static unsigned int*   s_d_all_outputs;  // device buffer for all digests in a batch
        static size_t          s_h_in_capacity;  // bytes
        static size_t          s_h_out_capacity; // bytes
        static size_t          s_d_in_capacity;  // bytes
        static size_t          s_d_out_capacity; // bytes
        static bool EnsureHostBatchBufferCapacity(size_t inBytes, size_t outBytes);
        static bool EnsureDeviceBatchBufferCapacity(size_t inBytes, size_t outBytes);
    
public:
    // Initialize CUDA environment
    static bool Initialize();
    
    // Cleanup CUDA resources
    static void Cleanup();
    
    // Check if CUDA is available
    static bool IsCudaAvailable() { return isCudaAvailable; }
    
    // Get GPU information
    static const GPUInfo& GetGPUInfo() { return gpuInfo; }
    
    // GPU-accelerated hash calculations
    static std::string CalculateMD5_GPU(const std::vector<unsigned char>& data);
    static std::string CalculateSHA1_GPU(const std::vector<unsigned char>& data);
    static std::string CalculateSHA256_GPU(const std::vector<unsigned char>& data);
    static std::string CalculateCRC32_GPU(const std::vector<unsigned char>& data);
    
    // Batch processing for multiple files
    static std::vector<std::string> CalculateBatchMD5_GPU(const std::vector<std::vector<unsigned char>>& dataList);
    static std::vector<std::string> CalculateBatchSHA1_GPU(const std::vector<std::vector<unsigned char>>& dataList);
    static std::vector<std::string> CalculateBatchSHA256_GPU(const std::vector<std::vector<unsigned char>>& dataList);
    static std::vector<std::string> CalculateBatchCRC32_GPU(const std::vector<std::vector<unsigned char>>& dataList);
    
    // Utility functions
    static bool SelectBestDevice();
    static void PrintGPUInfo();
    static size_t GetAvailableGPUMemory();
    static size_t GetOptimalBatchSize(size_t averageFileSize);
    
private:
    // Helper functions
    static bool AllocateGPUMemoryPools();
    static void FreeGPUMemoryPools();
    static std::string BytesToHexString(const unsigned char* data, size_t length);
    static bool SelfTest();
    static bool EnsureDevicePointerCapacity(int n);
};

// Note: Kernel declarations are intentionally omitted from the header to avoid
// linkage conflicts. They are defined internally in the .cu translation unit.

// Constants for GPU processing
namespace CudaConstants {
    const int MAX_THREADS_PER_BLOCK = 256;
    const int MAX_BLOCKS_PER_GRID = 65535;
    const size_t MAX_GPU_MEMORY_USAGE = 1024 * 1024 * 1024; // 1GB limit
    const size_t MIN_FILE_SIZE_FOR_GPU = 1024; // 1KB minimum for GPU processing
    const int OPTIMAL_BATCH_SIZE = 32;
    const size_t MEMORY_POOL_SIZE = 512 * 1024 * 1024; // 512MB memory pool
}

#endif // CUDA_HASH_CUH
