#include "cuda_hash.cuh"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cstring>

// Static member initialization
bool CudaHashCalculator::initialized = false;
GPUInfo CudaHashCalculator::gpuInfo;
bool CudaHashCalculator::isCudaAvailable = false;
void* CudaHashCalculator::d_input_pool = nullptr;
void* CudaHashCalculator::d_output_pool = nullptr;
size_t CudaHashCalculator::pool_size = 0;

// MD5 constants and functions for GPU
__constant__ unsigned int d_md5_k[64];
__constant__ unsigned int d_md5_r[64];

__device__ unsigned int md5_f(unsigned int x, unsigned int y, unsigned int z, int round) {
    if (round < 16) return (x & y) | (~x & z);
    if (round < 32) return (z & x) | (~z & y);
    if (round < 48) return x ^ y ^ z;
    return y ^ (x | ~z);
}

__device__ unsigned int md5_g(int round) {
    if (round < 16) return round;
    if (round < 32) return (5 * round + 1) % 16;
    if (round < 48) return (3 * round + 5) % 16;
    return (7 * round) % 16;
}

__device__ unsigned int rotate_left(unsigned int value, int shift) {
    return (value << shift) | (value >> (32 - shift));
}

// MD5 device function implementation
__device__ void md5_compute(const unsigned char* input, unsigned int* output, size_t length) {
    // MD5 initialization
    unsigned int h[4] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476};
    
    // Prepare padded message
    unsigned long long bit_len = length * 8;
    size_t padded_len = ((length + 8) / 64 + 1) * 64;
    
    // Process message in chunks of 64 bytes
    for (size_t chunk_start = 0; chunk_start < padded_len; chunk_start += 64) {
        unsigned int w[16] = {0};
        
        // Fill w array with data
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 4; j++) {
                size_t byte_idx = chunk_start + i * 4 + j;
                if (byte_idx < length) {
                    w[i] |= ((unsigned int)input[byte_idx]) << (8 * j);
                } else if (byte_idx == length) {
                    w[i] |= 0x80 << (8 * j);
                } else if (chunk_start + 56 <= byte_idx && byte_idx < chunk_start + 64) {
                    int bit_pos = (byte_idx - chunk_start - 56) * 8;
                    w[i] |= ((unsigned int)(bit_len >> bit_pos)) << (8 * j);
                }
            }
        }
        
        // MD5 algorithm main loop
        unsigned int a = h[0], b = h[1], c = h[2], d = h[3];
        
        for (int i = 0; i < 64; i++) {
            unsigned int f_val = md5_f(b, c, d, i);
            unsigned int g = md5_g(i);
            
            unsigned int temp = d;
            d = c;
            c = b;
            b = b + rotate_left(a + f_val + d_md5_k[i] + w[g], d_md5_r[i]);
            a = temp;
        }
        
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
    }
    
    // Store result
    output[0] = h[0];
    output[1] = h[1];
    output[2] = h[2];
    output[3] = h[3];
}

// MD5 kernel implementation
__global__ void md5_kernel(const unsigned char* input, unsigned int* output, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only process if within bounds
    if (idx >= 1) return;
    
    md5_compute(input, output, length);
}

// Batch MD5 kernel
__global__ void md5_batch_kernel(const unsigned char** inputs, unsigned int** outputs, 
                               const size_t* lengths, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size) return;
    
    md5_compute(inputs[idx], outputs[idx], lengths[idx]);
}

// CRC32 lookup table for GPU
__constant__ unsigned int d_crc32_table[256];

// CRC32 kernel implementation
// CRC32 device function implementation
__device__ void crc32_compute(const unsigned char* input, unsigned int* output, size_t length) {
    unsigned int crc = 0xFFFFFFFF;
    
    for (size_t i = 0; i < length; i++) {
        crc = d_crc32_table[(crc ^ input[i]) & 0xFF] ^ (crc >> 8);
    }
    
    *output = crc ^ 0xFFFFFFFF;
}

__global__ void crc32_kernel(const unsigned char* input, unsigned int* output, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= 1) return;
    
    crc32_compute(input, output, length);
}

// Batch CRC32 kernel
__global__ void crc32_batch_kernel(const unsigned char** inputs, unsigned int** outputs, 
                                 const size_t* lengths, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size) return;
    
    crc32_compute(inputs[idx], outputs[idx], lengths[idx]);
}

// SHA1 constants
__constant__ unsigned int d_sha1_k[4] = {0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xCA62C1D6};

__device__ unsigned int sha1_f(unsigned int x, unsigned int y, unsigned int z, int round) {
    if (round < 20) return (x & y) | (~x & z);
    if (round < 40) return x ^ y ^ z;
    if (round < 60) return (x & y) | (x & z) | (y & z);
    return x ^ y ^ z;
}

// SHA1 device function implementation
__device__ void sha1_compute(const unsigned char* input, unsigned int* output, size_t length) {
    // SHA1 initialization
    unsigned int h[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
    
    unsigned long long bit_len = length * 8;
    size_t padded_len = ((length + 8) / 64 + 1) * 64;
    
    // Process message in chunks of 64 bytes
    for (size_t chunk_start = 0; chunk_start < padded_len; chunk_start += 64) {
        unsigned int w[80] = {0};
        
        // Fill first 16 words
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 4; j++) {
                size_t byte_idx = chunk_start + i * 4 + (3 - j); // Big-endian
                if (byte_idx < length) {
                    w[i] |= ((unsigned int)input[byte_idx]) << (8 * j);
                } else if (byte_idx == length) {
                    w[i] |= 0x80 << (8 * j);
                } else if (chunk_start + 56 <= byte_idx && byte_idx < chunk_start + 64) {
                    int bit_pos = (63 - byte_idx + chunk_start) * 8;
                    w[i] |= ((unsigned int)(bit_len >> bit_pos)) << (8 * j);
                }
            }
        }
        
        // Extend the sixteen 32-bit words into eighty 32-bit words
        for (int i = 16; i < 80; i++) {
            w[i] = rotate_left(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);
        }
        
        unsigned int a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
        
        for (int i = 0; i < 80; i++) {
            unsigned int f_val = sha1_f(b, c, d, i);
            unsigned int k = d_sha1_k[i / 20];
            
            unsigned int temp = rotate_left(a, 5) + f_val + e + k + w[i];
            e = d;
            d = c;
            c = rotate_left(b, 30);
            b = a;
            a = temp;
        }
        
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
    }
    
    // Store result
    for (int i = 0; i < 5; i++) {
        output[i] = h[i];
    }
}

// SHA1 kernel implementation
__global__ void sha1_kernel(const unsigned char* input, unsigned int* output, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= 1) return;
    
    sha1_compute(input, output, length);
}

// Batch SHA1 kernel
__global__ void sha1_batch_kernel(const unsigned char** inputs, unsigned int** outputs, 
                                const size_t* lengths, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size) return;
    
    sha1_compute(inputs[idx], outputs[idx], lengths[idx]);
}

// SHA256 constants
__constant__ unsigned int d_sha256_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__device__ unsigned int sha256_ch(unsigned int x, unsigned int y, unsigned int z) {
    return (x & y) ^ (~x & z);
}

__device__ unsigned int sha256_maj(unsigned int x, unsigned int y, unsigned int z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ unsigned int sha256_sigma0(unsigned int x) {
    return rotate_left(x, 30) ^ rotate_left(x, 19) ^ rotate_left(x, 10);
}

__device__ unsigned int sha256_sigma1(unsigned int x) {
    return rotate_left(x, 26) ^ rotate_left(x, 21) ^ rotate_left(x, 7);
}

__device__ unsigned int sha256_gamma0(unsigned int x) {
    return rotate_left(x, 25) ^ rotate_left(x, 14) ^ (x >> 3);
}

__device__ unsigned int sha256_gamma1(unsigned int x) {
    return rotate_left(x, 15) ^ rotate_left(x, 13) ^ (x >> 10);
}

// SHA256 device function implementation
__device__ void sha256_compute(const unsigned char* input, unsigned int* output, size_t length) {
    // SHA256 initialization
    unsigned int h[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    unsigned long long bit_len = length * 8;
    size_t padded_len = ((length + 8) / 64 + 1) * 64;
    
    // Process message in chunks of 64 bytes
    for (size_t chunk_start = 0; chunk_start < padded_len; chunk_start += 64) {
        unsigned int w[64] = {0};
        
        // Fill first 16 words
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 4; j++) {
                size_t byte_idx = chunk_start + i * 4 + (3 - j); // Big-endian
                if (byte_idx < length) {
                    w[i] |= ((unsigned int)input[byte_idx]) << (8 * j);
                } else if (byte_idx == length) {
                    w[i] |= 0x80 << (8 * j);
                } else if (chunk_start + 56 <= byte_idx && byte_idx < chunk_start + 64) {
                    int bit_pos = (63 - byte_idx + chunk_start) * 8;
                    w[i] |= ((unsigned int)(bit_len >> bit_pos)) << (8 * j);
                }
            }
        }
        
        // Extend the sixteen 32-bit words into sixty-four 32-bit words
        for (int i = 16; i < 64; i++) {
            w[i] = sha256_gamma1(w[i-2]) + w[i-7] + sha256_gamma0(w[i-15]) + w[i-16];
        }
        
        unsigned int a = h[0], b = h[1], c = h[2], d = h[3];
        unsigned int e = h[4], f = h[5], g = h[6], h_temp = h[7];
        
        for (int i = 0; i < 64; i++) {
            unsigned int t1 = h_temp + sha256_sigma1(e) + sha256_ch(e, f, g) + d_sha256_k[i] + w[i];
            unsigned int t2 = sha256_sigma0(a) + sha256_maj(a, b, c);
            
            h_temp = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += h_temp;
    }
    
    // Store result
    for (int i = 0; i < 8; i++) {
        output[i] = h[i];
    }
}

// SHA256 kernel implementation
__global__ void sha256_kernel(const unsigned char* input, unsigned int* output, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= 1) return;
    
    sha256_compute(input, output, length);
}

// Batch SHA256 kernel
__global__ void sha256_batch_kernel(const unsigned char** inputs, unsigned int** outputs, 
                                  const size_t* lengths, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size) return;
    
    sha256_compute(inputs[idx], outputs[idx], lengths[idx]);
}

// Host function implementations
bool CudaHashCalculator::Initialize() {
    if (initialized) return true;
    
    // Check for CUDA runtime
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess || deviceCount == 0) {
        std::cerr << "CUDA not available: " << cudaGetErrorString(error) << std::endl;
        isCudaAvailable = false;
        return false;
    }
    
    // Initialize GPU info
    gpuInfo.deviceCount = deviceCount;
    gpuInfo.devices.resize(deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        CUDA_CHECK(cudaGetDeviceProperties(&gpuInfo.devices[i], i));
    }
    
    // Select best device
    if (!SelectBestDevice()) {
        return false;
    }
    
    // Initialize constant memory
    unsigned int md5_k[64] = {
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
        0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
        0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
        0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
        0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
        0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
        0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
    };
    
    unsigned int md5_r[64] = {
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
        5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
        4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
        6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21
    };
    
    // CRC32 table
    unsigned int crc32_table[256];
    for (unsigned int i = 0; i < 256; i++) {
        unsigned int crc = i;
        for (int j = 0; j < 8; j++) {
            if (crc & 1) {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
        crc32_table[i] = crc;
    }
    
    // Copy constants to GPU
    CUDA_CHECK(cudaMemcpyToSymbol(d_md5_k, md5_k, sizeof(md5_k)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_md5_r, md5_r, sizeof(md5_r)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_crc32_table, crc32_table, sizeof(crc32_table)));
    
    // Allocate GPU memory pools
    if (!AllocateGPUMemoryPools()) {
        return false;
    }
    
    isCudaAvailable = true;
    initialized = true;
    
    std::cout << "CUDA initialized successfully!" << std::endl;
    PrintGPUInfo();
    
    return true;
}

void CudaHashCalculator::Cleanup() {
    if (!initialized) return;
    
    FreeGPUMemoryPools();
    cudaDeviceReset();
    
    initialized = false;
    isCudaAvailable = false;
}

bool CudaHashCalculator::SelectBestDevice() {
    if (gpuInfo.deviceCount == 0) return false;
    
    int bestDevice = 0;
    int maxCores = 0;
    
    for (int i = 0; i < gpuInfo.deviceCount; i++) {
        const auto& prop = gpuInfo.devices[i];
        int cores = prop.multiProcessorCount * 
                   (prop.major >= 6 ? 128 : (prop.major >= 3 ? 192 : 32));
        
        if (cores > maxCores) {
            maxCores = cores;
            bestDevice = i;
        }
    }
    
    CUDA_CHECK(cudaSetDevice(bestDevice));
    gpuInfo.selectedDevice = bestDevice;
    gpuInfo.isAvailable = true;
    
    return true;
}

void CudaHashCalculator::PrintGPUInfo() {
    if (!isCudaAvailable) {
        std::cout << "CUDA is not available." << std::endl;
        return;
    }
    
    std::cout << "GPU Information:" << std::endl;
    std::cout << "  Devices found: " << gpuInfo.deviceCount << std::endl;
    
    const auto& prop = gpuInfo.devices[gpuInfo.selectedDevice];
    std::cout << "  Selected device: " << gpuInfo.selectedDevice << " (" << prop.name << ")" << std::endl;
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "  Global memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
    std::cout << "  Shared memory per block: " << (prop.sharedMemPerBlock / 1024) << " KB" << std::endl;
    std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
}

size_t CudaHashCalculator::GetAvailableGPUMemory() {
    if (!isCudaAvailable) return 0;
    
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    return free_mem;
}

size_t CudaHashCalculator::GetOptimalBatchSize(size_t averageFileSize) {
    if (!isCudaAvailable) return 1;
    
    size_t available_memory = GetAvailableGPUMemory();
    size_t max_files = available_memory / (averageFileSize + 1024); // Extra overhead
    
    return std::min(max_files, (size_t)CudaConstants::OPTIMAL_BATCH_SIZE);
}

bool CudaHashCalculator::AllocateGPUMemoryPools() {
    pool_size = CudaConstants::MEMORY_POOL_SIZE;
    
    // Allocate input pool
    cudaError_t error = cudaMalloc(&d_input_pool, pool_size);
    if (error != cudaSuccess) {
        pool_size /= 2; // Try with half size
        error = cudaMalloc(&d_input_pool, pool_size);
        if (error != cudaSuccess) {
            std::cerr << "Failed to allocate GPU input memory pool: " << cudaGetErrorString(error) << std::endl;
            return false;
        }
    }
    
    // Allocate output pool (much smaller)
    size_t output_pool_size = 1024 * 1024; // 1MB for outputs
    error = cudaMalloc(&d_output_pool, output_pool_size);
    if (error != cudaSuccess) {
        cudaFree(d_input_pool);
        d_input_pool = nullptr;
        std::cerr << "Failed to allocate GPU output memory pool: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    std::cout << "GPU memory pools allocated: " << (pool_size / (1024*1024)) << " MB" << std::endl;
    
    return true;
}

void CudaHashCalculator::FreeGPUMemoryPools() {
    if (d_input_pool) {
        cudaFree(d_input_pool);
        d_input_pool = nullptr;
    }
    
    if (d_output_pool) {
        cudaFree(d_output_pool);
        d_output_pool = nullptr;
    }
    
    pool_size = 0;
}

std::string CudaHashCalculator::BytesToHexString(const unsigned char* data, size_t length) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    
    for (size_t i = 0; i < length; i++) {
        ss << std::setw(2) << (unsigned int)data[i];
    }
    
    return ss.str();
}

// GPU hash calculation implementations
std::string CudaHashCalculator::CalculateMD5_GPU(const std::vector<unsigned char>& data) {
    if (!isCudaAvailable || data.empty()) return "";
    
    // Allocate GPU memory
    unsigned char* d_input;
    unsigned int* d_output;
    
    CUDA_CHECK(cudaMalloc(&d_input, data.size()));
    CUDA_CHECK(cudaMalloc(&d_output, 16)); // MD5 is 16 bytes
    
    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_input, data.data(), data.size(), cudaMemcpyHostToDevice));
    
    // Launch kernel
    md5_kernel<<<1, 1>>>(d_input, d_output, data.size());
    CUDA_CHECK_KERNEL();
    
    // Copy result back
    unsigned int result[4];
    CUDA_CHECK(cudaMemcpy(result, d_output, 16, cudaMemcpyDeviceToHost));
    
    // Convert to hex string
    std::string hex_result = BytesToHexString(reinterpret_cast<unsigned char*>(result), 16);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    
    return hex_result;
}

std::string CudaHashCalculator::CalculateSHA1_GPU(const std::vector<unsigned char>& data) {
    if (!isCudaAvailable || data.empty()) return "";
    
    // Allocate GPU memory
    unsigned char* d_input;
    unsigned int* d_output;
    
    CUDA_CHECK(cudaMalloc(&d_input, data.size()));
    CUDA_CHECK(cudaMalloc(&d_output, 20)); // SHA1 is 20 bytes
    
    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_input, data.data(), data.size(), cudaMemcpyHostToDevice));
    
    // Launch kernel
    sha1_kernel<<<1, 1>>>(d_input, d_output, data.size());
    CUDA_CHECK_KERNEL();
    
    // Copy result back
    unsigned int result[5];
    CUDA_CHECK(cudaMemcpy(result, d_output, 20, cudaMemcpyDeviceToHost));
    
    // Convert to hex string (big-endian)
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (int i = 0; i < 5; i++) {
        // Convert to big-endian for display
        unsigned int be_val = ((result[i] & 0xFF) << 24) | 
                              ((result[i] & 0xFF00) << 8) |
                              ((result[i] & 0xFF0000) >> 8) |
                              ((result[i] & 0xFF000000) >> 24);
        ss << std::setw(8) << be_val;
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    
    return ss.str();
}

std::string CudaHashCalculator::CalculateSHA256_GPU(const std::vector<unsigned char>& data) {
    if (!isCudaAvailable || data.empty()) return "";
    
    // Allocate GPU memory
    unsigned char* d_input;
    unsigned int* d_output;
    
    CUDA_CHECK(cudaMalloc(&d_input, data.size()));
    CUDA_CHECK(cudaMalloc(&d_output, 32)); // SHA256 is 32 bytes
    
    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_input, data.data(), data.size(), cudaMemcpyHostToDevice));
    
    // Launch kernel
    sha256_kernel<<<1, 1>>>(d_input, d_output, data.size());
    CUDA_CHECK_KERNEL();
    
    // Copy result back
    unsigned int result[8];
    CUDA_CHECK(cudaMemcpy(result, d_output, 32, cudaMemcpyDeviceToHost));
    
    // Convert to hex string (big-endian)
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (int i = 0; i < 8; i++) {
        // Convert to big-endian for display
        unsigned int be_val = ((result[i] & 0xFF) << 24) | 
                              ((result[i] & 0xFF00) << 8) |
                              ((result[i] & 0xFF0000) >> 8) |
                              ((result[i] & 0xFF000000) >> 24);
        ss << std::setw(8) << be_val;
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    
    return ss.str();
}

std::string CudaHashCalculator::CalculateCRC32_GPU(const std::vector<unsigned char>& data) {
    if (!isCudaAvailable || data.empty()) return "";
    
    // Allocate GPU memory
    unsigned char* d_input;
    unsigned int* d_output;
    
    CUDA_CHECK(cudaMalloc(&d_input, data.size()));
    CUDA_CHECK(cudaMalloc(&d_output, 4)); // CRC32 is 4 bytes
    
    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_input, data.data(), data.size(), cudaMemcpyHostToDevice));
    
    // Launch kernel
    crc32_kernel<<<1, 1>>>(d_input, d_output, data.size());
    CUDA_CHECK_KERNEL();
    
    // Copy result back
    unsigned int result;
    CUDA_CHECK(cudaMemcpy(&result, d_output, 4, cudaMemcpyDeviceToHost));
    
    // Convert to hex string
    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(8) << result;
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    
    return ss.str();
}

// Batch processing functions (simplified implementation)
std::vector<std::string> CudaHashCalculator::CalculateBatchMD5_GPU(const std::vector<std::vector<unsigned char>>& dataList) {
    std::vector<std::string> results;
    
    // For simplicity, process one by one (can be optimized for true batch processing)
    for (const auto& data : dataList) {
        results.push_back(CalculateMD5_GPU(data));
    }
    
    return results;
}

std::vector<std::string> CudaHashCalculator::CalculateBatchSHA1_GPU(const std::vector<std::vector<unsigned char>>& dataList) {
    std::vector<std::string> results;
    
    for (const auto& data : dataList) {
        results.push_back(CalculateSHA1_GPU(data));
    }
    
    return results;
}

std::vector<std::string> CudaHashCalculator::CalculateBatchSHA256_GPU(const std::vector<std::vector<unsigned char>>& dataList) {
    std::vector<std::string> results;
    
    for (const auto& data : dataList) {
        results.push_back(CalculateSHA256_GPU(data));
    }
    
    return results;
}

std::vector<std::string> CudaHashCalculator::CalculateBatchCRC32_GPU(const std::vector<std::vector<unsigned char>>& dataList) {
    std::vector<std::string> results;
    
    for (const auto& data : dataList) {
        results.push_back(CalculateCRC32_GPU(data));
    }
    
    return results;
}
