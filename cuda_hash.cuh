#ifndef CUDA_HASH_CUH
#define CUDA_HASH_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string>
#include <vector>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            return false; \
        } \
    } while(0)

// CUDA error checking macro for kernels
#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA kernel error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            return false; \
        } \
        cudaDeviceSynchronize(); \
        error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA kernel execution error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            return false; \
        } \
    } while(0)

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
    
    // GPU memory pools for better performance
    static void* d_input_pool;
    static void* d_output_pool;
    static size_t pool_size;
    
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
};

// CUDA kernel declarations
extern "C" {
    // MD5 kernels
    __global__ void md5_kernel(const unsigned char* input, unsigned int* output, size_t length);
    __global__ void md5_batch_kernel(const unsigned char** inputs, unsigned int** outputs, 
                                   const size_t* lengths, int batch_size);
    
    // SHA1 kernels
    __global__ void sha1_kernel(const unsigned char* input, unsigned int* output, size_t length);
    __global__ void sha1_batch_kernel(const unsigned char** inputs, unsigned int** outputs, 
                                    const size_t* lengths, int batch_size);
    
    // SHA256 kernels
    __global__ void sha256_kernel(const unsigned char* input, unsigned int* output, size_t length);
    __global__ void sha256_batch_kernel(const unsigned char** inputs, unsigned int** outputs, 
                                      const size_t* lengths, int batch_size);
    
    // CRC32 kernels
    __global__ void crc32_kernel(const unsigned char* input, unsigned int* output, size_t length);
    __global__ void crc32_batch_kernel(const unsigned char** inputs, unsigned int** outputs, 
                                     const size_t* lengths, int batch_size);
}

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
