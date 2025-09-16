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
std::mutex CudaHashCalculator::s_mutex;
void* CudaHashCalculator::d_input_pool = nullptr;
void* CudaHashCalculator::d_output_pool = nullptr;
size_t CudaHashCalculator::pool_size = 0;
cudaStream_t CudaHashCalculator::s_stream1 = nullptr;
cudaStream_t CudaHashCalculator::s_stream2 = nullptr;
cudaEvent_t  CudaHashCalculator::s_evtPtrReady = nullptr;
unsigned char** CudaHashCalculator::s_d_inputs = nullptr;
unsigned int**  CudaHashCalculator::s_d_outputs = nullptr;
size_t*         CudaHashCalculator::s_d_lengths = nullptr;
int             CudaHashCalculator::s_ptrCapacity = 0;
unsigned char*  CudaHashCalculator::s_h_all_inputs = nullptr;
unsigned int*   CudaHashCalculator::s_h_all_outputs = nullptr;
unsigned char*  CudaHashCalculator::s_d_all_inputs = nullptr;
unsigned int*   CudaHashCalculator::s_d_all_outputs = nullptr;
size_t          CudaHashCalculator::s_h_in_capacity = 0;
size_t          CudaHashCalculator::s_h_out_capacity = 0;
size_t          CudaHashCalculator::s_d_in_capacity = 0;
size_t          CudaHashCalculator::s_d_out_capacity = 0;

// MD5 constants and functions for GPU
__constant__ unsigned int d_md5_k[64];
__constant__ unsigned int d_md5_r[64];

__device__ __forceinline__ unsigned int md5_f(unsigned int x, unsigned int y, unsigned int z, int round) {
    if (round < 16) return (x & y) | (~x & z);
    if (round < 32) return (z & x) | (~z & y);
    if (round < 48) return x ^ y ^ z;
    return y ^ (x | ~z);
}

__device__ __forceinline__ unsigned int md5_g(int round) {
    if (round < 16) return round;
    if (round < 32) return (5 * round + 1) % 16;
    if (round < 48) return (3 * round + 5) % 16;
    return (7 * round) % 16;
}

__device__ __forceinline__ unsigned int rotate_left(unsigned int value, int shift) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 320)
    // Use hardware funnel shift when available for better performance
    return __funnelshift_l(value, value, shift & 31);
#else
    return (value << (shift & 31)) | (value >> ((32 - (shift & 31)) & 31));
#endif
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
                } else if ((chunk_start + 64 == padded_len) && (chunk_start + 56 <= byte_idx && byte_idx < chunk_start + 64)) {
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
__global__ void md5_batch_kernel(const unsigned char* const* __restrict__ inputs, unsigned int* const* __restrict__ outputs, 
                               const size_t* __restrict__ lengths, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size) return;
    
    md5_compute(inputs[idx], outputs[idx], lengths[idx]);
}

// CRC32 lookup table for GPU
__constant__ unsigned int d_crc32_table[256];

// CRC32 kernel implementation
// CRC32 device function implementation
__device__ __forceinline__ void crc32_compute(const unsigned char* input, unsigned int* output, size_t length) {
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
__global__ void crc32_batch_kernel(const unsigned char* const* __restrict__ inputs, unsigned int* const* __restrict__ outputs, 
                                 const size_t* __restrict__ lengths, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size) return;
    
    crc32_compute(inputs[idx], outputs[idx], lengths[idx]);
}

// SHA1 constants
__constant__ unsigned int d_sha1_k[4] = {0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xCA62C1D6};

__device__ __forceinline__ unsigned int sha1_f(unsigned int x, unsigned int y, unsigned int z, int round) {
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
        unsigned char chunk[64] = {0};

        // Fill chunk with message/padding
        for (int i = 0; i < 64; ++i) {
            size_t idx = chunk_start + i;
            if (idx < length) {
                chunk[i] = input[idx];
            } else if (idx == length) {
                chunk[i] = 0x80;
            } else {
                // remains 0
            }
        }
        // Write 64-bit big-endian length in the final chunk
        if (chunk_start + 64 == padded_len) {
            for (int k = 0; k < 8; ++k) {
                chunk[56 + k] = (unsigned char)((bit_len >> (8 * (7 - k))) & 0xFF);
            }
        }

        // Convert to big-endian 32-bit words
        for (int i = 0; i < 16; i++) {
            w[i] = ((unsigned int)chunk[i * 4 + 0] << 24) |
                   ((unsigned int)chunk[i * 4 + 1] << 16) |
                   ((unsigned int)chunk[i * 4 + 2] << 8)  |
                   ((unsigned int)chunk[i * 4 + 3] << 0);
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
__global__ void sha1_batch_kernel(const unsigned char* const* __restrict__ inputs, unsigned int* const* __restrict__ outputs, 
                                const size_t* __restrict__ lengths, int batch_size) {
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

__device__ __forceinline__ unsigned int sha256_ch(unsigned int x, unsigned int y, unsigned int z) {
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ unsigned int sha256_maj(unsigned int x, unsigned int y, unsigned int z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ unsigned int sha256_sigma0(unsigned int x) {
    return rotate_left(x, 30) ^ rotate_left(x, 19) ^ rotate_left(x, 10);
}

__device__ __forceinline__ unsigned int sha256_sigma1(unsigned int x) {
    return rotate_left(x, 26) ^ rotate_left(x, 21) ^ rotate_left(x, 7);
}

__device__ __forceinline__ unsigned int sha256_gamma0(unsigned int x) {
    return rotate_left(x, 25) ^ rotate_left(x, 14) ^ (x >> 3);
}

__device__ __forceinline__ unsigned int sha256_gamma1(unsigned int x) {
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
        unsigned char chunk[64] = {0};

        // Fill chunk with message/padding
        for (int i = 0; i < 64; ++i) {
            size_t idx = chunk_start + i;
            if (idx < length) {
                chunk[i] = input[idx];
            } else if (idx == length) {
                chunk[i] = 0x80;
            } else {
                // remains 0
            }
        }
        // Write 64-bit big-endian length in the final chunk
        if (chunk_start + 64 == padded_len) {
            for (int k = 0; k < 8; ++k) {
                chunk[56 + k] = (unsigned char)((bit_len >> (8 * (7 - k))) & 0xFF);
            }
        }

        // Convert to big-endian 32-bit words
        for (int i = 0; i < 16; i++) {
            w[i] = ((unsigned int)chunk[i * 4 + 0] << 24) |
                   ((unsigned int)chunk[i * 4 + 1] << 16) |
                   ((unsigned int)chunk[i * 4 + 2] << 8)  |
                   ((unsigned int)chunk[i * 4 + 3] << 0);
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
__global__ void sha256_batch_kernel(const unsigned char* const* __restrict__ inputs, unsigned int* const* __restrict__ outputs, 
                                  const size_t* __restrict__ lengths, int batch_size) {
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
        if (!CUDA_TRY(cudaGetDeviceProperties(&gpuInfo.devices[i], i))) {
            isCudaAvailable = false;
            return false;
        }
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
    if (!CUDA_TRY(cudaMemcpyToSymbol(d_md5_k, md5_k, sizeof(md5_k)))) return false;
    if (!CUDA_TRY(cudaMemcpyToSymbol(d_md5_r, md5_r, sizeof(md5_r)))) return false;
    if (!CUDA_TRY(cudaMemcpyToSymbol(d_crc32_table, crc32_table, sizeof(crc32_table)))) return false;
    
    // Allocate GPU memory pools
    if (!AllocateGPUMemoryPools()) {
        return false;
    }
    
    // Create persistent streams (non-blocking)
    if (!CUDA_TRY(cudaStreamCreateWithFlags(&s_stream1, cudaStreamNonBlocking))) {
        FreeGPUMemoryPools();
        return false;
    }
    if (!CUDA_TRY(cudaStreamCreateWithFlags(&s_stream2, cudaStreamNonBlocking))) {
        cudaStreamDestroy(s_stream1); s_stream1 = nullptr;
        FreeGPUMemoryPools();
        return false;
    }

    // Create persistent event (no timing)
    if (!CUDA_TRY(cudaEventCreateWithFlags(&s_evtPtrReady, cudaEventDisableTiming))) {
        cudaStreamDestroy(s_stream2); s_stream2 = nullptr;
        cudaStreamDestroy(s_stream1); s_stream1 = nullptr;
        FreeGPUMemoryPools();
        return false;
    }

    isCudaAvailable = true;
    initialized = true;

    // Run a quick self-test to validate correctness
    if (!SelfTest()) {
        std::cerr << "CUDA self-test failed. Disabling GPU acceleration." << std::endl;
        Cleanup();
        return false;
    }

    std::cout << "CUDA initialized successfully!" << std::endl;
    PrintGPUInfo();
    
    return true;
}

void CudaHashCalculator::Cleanup() {
    if (!initialized) return;
    
    if (s_evtPtrReady) { cudaEventDestroy(s_evtPtrReady); s_evtPtrReady = nullptr; }
    if (s_stream2) { cudaStreamDestroy(s_stream2); s_stream2 = nullptr; }
    if (s_stream1) { cudaStreamDestroy(s_stream1); s_stream1 = nullptr; }
    if (s_d_lengths) { cudaFree(s_d_lengths); s_d_lengths = nullptr; }
    if (s_d_outputs) { cudaFree(s_d_outputs); s_d_outputs = nullptr; }
    if (s_d_inputs) { cudaFree(s_d_inputs); s_d_inputs = nullptr; }
    if (s_d_all_outputs) { cudaFree(s_d_all_outputs); s_d_all_outputs = nullptr; s_d_out_capacity = 0; }
    if (s_d_all_inputs) { cudaFree(s_d_all_inputs); s_d_all_inputs = nullptr; s_d_in_capacity = 0; }
    if (s_h_all_outputs) { cudaFreeHost(s_h_all_outputs); s_h_all_outputs = nullptr; s_h_out_capacity = 0; }
    if (s_h_all_inputs) { cudaFreeHost(s_h_all_inputs); s_h_all_inputs = nullptr; s_h_in_capacity = 0; }
    s_ptrCapacity = 0;

    FreeGPUMemoryPools();
    cudaDeviceReset();
    
    initialized = false;
    isCudaAvailable = false;
}

bool CudaHashCalculator::EnsureHostBatchBufferCapacity(size_t inBytes, size_t outBytes) {
    if (!isCudaAvailable) return false;
    bool ok = true;
    // Input pinned host buffer: prefer write-combined for H2D throughput (CPU write-only, then DMA)
    if (inBytes > s_h_in_capacity) {
        // geometric growth with a reasonable floor to reduce realloc frequency
        size_t newCap = s_h_in_capacity ? s_h_in_capacity * 2 : 256 * 1024; // 256KB min
        if (newCap < inBytes) newCap = inBytes;
        if (s_h_all_inputs) { cudaFreeHost(s_h_all_inputs); s_h_all_inputs = nullptr; s_h_in_capacity = 0; }
        ok = CUDA_TRY(cudaHostAlloc(&s_h_all_inputs, newCap, cudaHostAllocWriteCombined));
        if (!ok) return false;
        s_h_in_capacity = newCap;
    }
    // Output pinned host buffer: default pinned memory (CPU will read results)
    if (outBytes > s_h_out_capacity) {
        size_t newCap = s_h_out_capacity ? s_h_out_capacity * 2 : 64 * 1024; // 64KB min
        if (newCap < outBytes) newCap = outBytes;
        if (s_h_all_outputs) { cudaFreeHost(s_h_all_outputs); s_h_all_outputs = nullptr; s_h_out_capacity = 0; }
        ok = CUDA_TRY(cudaHostAlloc(&s_h_all_outputs, newCap, cudaHostAllocDefault));
        if (!ok) return false;
        s_h_out_capacity = newCap;
    }
    return true;
}

bool CudaHashCalculator::EnsureDeviceBatchBufferCapacity(size_t inBytes, size_t outBytes) {
    if (!isCudaAvailable) return false;
    if (inBytes > s_d_in_capacity) {
        size_t newCap = s_d_in_capacity ? s_d_in_capacity * 2 : 256 * 1024; // 256KB min
        if (newCap < inBytes) newCap = inBytes;
        if (s_d_all_inputs) { cudaFree(s_d_all_inputs); s_d_all_inputs = nullptr; s_d_in_capacity = 0; }
        if (!CUDA_TRY(cudaMalloc(&s_d_all_inputs, newCap))) return false;
        s_d_in_capacity = newCap;
    }
    if (outBytes > s_d_out_capacity) {
        size_t newCap = s_d_out_capacity ? s_d_out_capacity * 2 : 64 * 1024; // 64KB min
        if (newCap < outBytes) newCap = outBytes;
        if (s_d_all_outputs) { cudaFree(s_d_all_outputs); s_d_all_outputs = nullptr; s_d_out_capacity = 0; }
        if (!CUDA_TRY(cudaMalloc(&s_d_all_outputs, newCap))) return false;
        s_d_out_capacity = newCap;
    }
    return true;
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
    
    if (!CUDA_TRY(cudaSetDevice(bestDevice))) {
        return false;
    }
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
    if (!CUDA_TRY(cudaMemGetInfo(&free_mem, &total_mem))) {
        return 0;
    }
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

bool CudaHashCalculator::EnsureDevicePointerCapacity(int n) {
    if (!isCudaAvailable) return false;
    if (n <= s_ptrCapacity && s_d_inputs && s_d_outputs && s_d_lengths) return true;

    // Free existing if any
    if (s_d_lengths) { cudaFree(s_d_lengths); s_d_lengths = nullptr; }
    if (s_d_outputs) { cudaFree(s_d_outputs); s_d_outputs = nullptr; }
    if (s_d_inputs)  { cudaFree(s_d_inputs);  s_d_inputs  = nullptr; }
    s_ptrCapacity = 0;

    if (n <= 0) return true;

    // Allocate with headroom to reduce reallocations
    int newCap = s_ptrCapacity ? std::max(n, s_ptrCapacity * 2) : std::max(n, 64);
    if (!CUDA_TRY(cudaMalloc(&s_d_inputs,  sizeof(unsigned char*) * newCap))) return false;
    if (!CUDA_TRY(cudaMalloc(&s_d_outputs, sizeof(unsigned int*)  * newCap))) { cudaFree(s_d_inputs); s_d_inputs = nullptr; return false; }
    if (!CUDA_TRY(cudaMalloc(&s_d_lengths, sizeof(size_t) * newCap)))       { cudaFree(s_d_outputs); s_d_outputs = nullptr; cudaFree(s_d_inputs); s_d_inputs = nullptr; return false; }

    s_ptrCapacity = newCap;
    return true;
}

bool CudaHashCalculator::SelfTest() {
    // Known test vector: empty string
    const std::vector<unsigned char> empty;
    // MD5("") = d41d8cd98f00b204e9800998ecf8427e
    // SHA1("") = da39a3ee5e6b4b0d3255bfef95601890afd80709
    // SHA256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    // CRC32("") = 00000000
    // Our GPU path returns empty string for empty input by design; test small payload instead

    const char* msg = "abc"; // common test vector
    std::vector<unsigned char> data(reinterpret_cast<const unsigned char*>(msg),
                                    reinterpret_cast<const unsigned char*>(msg) + 3);

    // Expected values
    const std::string md5_exp = "900150983cd24fb0d6963f7d28e17f72";
    const std::string sha1_exp = "a9993e364706816aba3e25717850c26c9cd0d89d";
    const std::string sha256_exp = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    const std::string crc32_exp = "352441c2"; // standard IEEE CRC-32 of "abc"

    std::string md5 = CalculateMD5_GPU(data);
    std::string sha1 = CalculateSHA1_GPU(data);
    std::string sha256 = CalculateSHA256_GPU(data);
    std::string crc32 = CalculateCRC32_GPU(data);

    auto tolower_str = [](std::string s){
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
        return s;
    };

    if (tolower_str(md5) != md5_exp) {
        std::cerr << "MD5 self-test failed: got " << md5 << ", expected " << md5_exp << std::endl;
        return false;
    }
    if (tolower_str(sha1) != sha1_exp) {
        std::cerr << "SHA1 self-test failed: got " << sha1 << ", expected " << sha1_exp << std::endl;
        return false;
    }
    if (tolower_str(sha256) != sha256_exp) {
        std::cerr << "SHA256 self-test failed: got " << sha256 << ", expected " << sha256_exp << std::endl;
        return false;
    }
    if (tolower_str(crc32) != crc32_exp) {
        std::cerr << "CRC32 self-test failed: got " << crc32 << ", expected " << crc32_exp << std::endl;
        return false;
    }

    return true;
}

// GPU hash calculation implementations
std::string CudaHashCalculator::CalculateMD5_GPU(const std::vector<unsigned char>& data) {
    std::lock_guard<std::mutex> lock(s_mutex);
    if (!isCudaAvailable) return "";
    if (data.empty()) {
        // MD5("")
        return std::string("d41d8cd98f00b204e9800998ecf8427e");
    }

    const size_t inBytes = data.size();
    const size_t outBytes = 16; // 4 words

    // Ensure persistent pinned host/device buffers
    if (!EnsureHostBatchBufferCapacity(inBytes, outBytes)) return "";
    if (!EnsureDeviceBatchBufferCapacity(inBytes, outBytes)) return "";

    // Copy input into pinned host buffer and H2D async on stream1
    std::memcpy(s_h_all_inputs, data.data(), inBytes);
    if (!CUDA_TRY(cudaMemcpyAsync(s_d_all_inputs, s_h_all_inputs, inBytes, cudaMemcpyHostToDevice, s_stream1))) return "";

    // Launch kernel (single item)
    md5_kernel<<<1, 1, 0, s_stream1>>>(s_d_all_inputs, s_d_all_outputs, inBytes);
    if (!CUDA_TRY_KERNEL_NOSYNC()) return "";

    // D2H async and synchronize stream1
    if (!CUDA_TRY(cudaMemcpyAsync(s_h_all_outputs, s_d_all_outputs, outBytes, cudaMemcpyDeviceToHost, s_stream1))) return "";
    if (!CUDA_TRY(cudaStreamSynchronize(s_stream1))) return "";

    // Convert to hex string (MD5 uses byte order of output)
    std::string hex_result = BytesToHexString(reinterpret_cast<unsigned char*>(s_h_all_outputs), 16);
    return hex_result;
}

std::string CudaHashCalculator::CalculateSHA1_GPU(const std::vector<unsigned char>& data) {
    std::lock_guard<std::mutex> lock(s_mutex);
    if (!isCudaAvailable) return "";
    if (data.empty()) {
        // SHA1("")
        return std::string("da39a3ee5e6b4b0d3255bfef95601890afd80709");
    }

    const size_t inBytes = data.size();
    const size_t outWords = 5;
    const size_t outBytes = outWords * sizeof(unsigned int);

    if (!EnsureHostBatchBufferCapacity(inBytes, outBytes)) return "";
    if (!EnsureDeviceBatchBufferCapacity(inBytes, outBytes)) return "";

    std::memcpy(s_h_all_inputs, data.data(), inBytes);
    if (!CUDA_TRY(cudaMemcpyAsync(s_d_all_inputs, s_h_all_inputs, inBytes, cudaMemcpyHostToDevice, s_stream1))) return "";

    sha1_kernel<<<1, 1, 0, s_stream1>>>(s_d_all_inputs, s_d_all_outputs, inBytes);
    if (!CUDA_TRY_KERNEL_NOSYNC()) return "";

    if (!CUDA_TRY(cudaMemcpyAsync(s_h_all_outputs, s_d_all_outputs, outBytes, cudaMemcpyDeviceToHost, s_stream1))) return "";
    if (!CUDA_TRY(cudaStreamSynchronize(s_stream1))) return "";

    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (int i = 0; i < static_cast<int>(outWords); i++) ss << std::setw(8) << s_h_all_outputs[i];
    return ss.str();
}

std::string CudaHashCalculator::CalculateSHA256_GPU(const std::vector<unsigned char>& data) {
    std::lock_guard<std::mutex> lock(s_mutex);
    if (!isCudaAvailable) return "";
    if (data.empty()) {
        // SHA256("")
        return std::string("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    }

    const size_t inBytes = data.size();
    const size_t outWords = 8;
    const size_t outBytes = outWords * sizeof(unsigned int);

    if (!EnsureHostBatchBufferCapacity(inBytes, outBytes)) return "";
    if (!EnsureDeviceBatchBufferCapacity(inBytes, outBytes)) return "";

    std::memcpy(s_h_all_inputs, data.data(), inBytes);
    if (!CUDA_TRY(cudaMemcpyAsync(s_d_all_inputs, s_h_all_inputs, inBytes, cudaMemcpyHostToDevice, s_stream1))) return "";

    sha256_kernel<<<1, 1, 0, s_stream1>>>(s_d_all_inputs, s_d_all_outputs, inBytes);
    if (!CUDA_TRY_KERNEL_NOSYNC()) return "";

    if (!CUDA_TRY(cudaMemcpyAsync(s_h_all_outputs, s_d_all_outputs, outBytes, cudaMemcpyDeviceToHost, s_stream1))) return "";
    if (!CUDA_TRY(cudaStreamSynchronize(s_stream1))) return "";

    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (int i = 0; i < static_cast<int>(outWords); i++) ss << std::setw(8) << s_h_all_outputs[i];
    return ss.str();
}

std::string CudaHashCalculator::CalculateCRC32_GPU(const std::vector<unsigned char>& data) {
    std::lock_guard<std::mutex> lock(s_mutex);
    if (!isCudaAvailable) return "";
    if (data.empty()) {
        // CRC32("")
        return std::string("00000000");
    }

    const size_t inBytes = data.size();
    const size_t outWords = 1;
    const size_t outBytes = outWords * sizeof(unsigned int);

    if (!EnsureHostBatchBufferCapacity(inBytes, outBytes)) return "";
    if (!EnsureDeviceBatchBufferCapacity(inBytes, outBytes)) return "";

    std::memcpy(s_h_all_inputs, data.data(), inBytes);
    if (!CUDA_TRY(cudaMemcpyAsync(s_d_all_inputs, s_h_all_inputs, inBytes, cudaMemcpyHostToDevice, s_stream1))) return "";

    crc32_kernel<<<1, 1, 0, s_stream1>>>(s_d_all_inputs, s_d_all_outputs, inBytes);
    if (!CUDA_TRY_KERNEL_NOSYNC()) return "";

    if (!CUDA_TRY(cudaMemcpyAsync(s_h_all_outputs, s_d_all_outputs, outBytes, cudaMemcpyDeviceToHost, s_stream1))) return "";
    if (!CUDA_TRY(cudaStreamSynchronize(s_stream1))) return "";

    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(8) << s_h_all_outputs[0];
    return ss.str();
}

// Batch processing functions (simplified implementation)
std::vector<std::string> CudaHashCalculator::CalculateBatchMD5_GPU(const std::vector<std::vector<unsigned char>>& dataList) {
    std::lock_guard<std::mutex> lock(s_mutex);
    std::vector<std::string> results;
    if (!isCudaAvailable || dataList.empty()) return results;

    const int n = static_cast<int>(dataList.size());
    results.resize(n);

    // Pre-declare host pointer arrays and launch config to avoid goto crossing initialization
    std::vector<unsigned char*> h_inputs(n, nullptr);
    std::vector<unsigned int*> h_outputs_ptrs(n, nullptr);
    // Predeclare compact arrays to avoid goto-bypassed initialization
    int m = 0;
    std::vector<unsigned char*> h_inputs_c;
    std::vector<unsigned int*> h_outputs_ptrs_c;
    std::vector<size_t> lengths_c;
    std::vector<size_t> offsets_c;
    dim3 block;
    dim3 grid;
    // Predeclare device pointers to avoid goto-bypassed initialization warnings
    unsigned char* d_all_inputs = nullptr;
    unsigned int* d_all_outputs = nullptr;
    unsigned char** d_inputs = nullptr;
    unsigned int** d_outputs = nullptr;
    size_t* d_lengths = nullptr;
    // Predeclare split variables to avoid goto-bypassed initialization warnings
    int mid;
    int cnt1;
    int cnt2;
    size_t bytes1;
    size_t bytes2;
    const size_t outWordsPerItem = 4;
    size_t outBytes1;
    size_t outBytes2;
    size_t startOffset2;

    // Handle empty inputs: directly set standard digest for empty string
    int nonEmptyCount = 0;
    std::vector<int> nonEmptyIdx;
    std::vector<size_t> lengths(n, 0);
    std::vector<size_t> offsets(n, 0);
    size_t totalBytes = 0;
    for (int i = 0; i < n; ++i) {
        lengths[i] = dataList[i].size();
        if (lengths[i] == 0) {
            results[i] = "d41d8cd98f00b204e9800998ecf8427e"; // MD5("")
        } else {
            offsets[i] = totalBytes;
            totalBytes += lengths[i];
            ++nonEmptyCount;
            nonEmptyIdx.push_back(i);
        }
    }
    if (nonEmptyCount == 0) return results;

    // Ensure pinned host buffers
    if (!EnsureHostBatchBufferCapacity(totalBytes, sizeof(unsigned int) * 4 * n)) return results;
    unsigned char* h_all_inputs = s_h_all_inputs;
    unsigned int* h_all_outputs = s_h_all_outputs; // 4 words per MD5

    // Pack inputs
    for (int i = 0; i < n; ++i) {
        if (lengths[i] > 0) {
            memcpy(h_all_inputs + offsets[i], dataList[i].data(), lengths[i]);
        }
    }

    bool ok = true;
    bool started1 = false, started2 = false;
    do {
        // Ensure device buffers
        if (!EnsureDeviceBatchBufferCapacity(totalBytes, sizeof(unsigned int) * 4 * n)) { ok = false; break; }
        d_all_inputs = s_d_all_inputs;
        d_all_outputs = s_d_all_outputs;

        if (!EnsureDevicePointerCapacity(n)) { ok = false; break; }
        d_inputs = s_d_inputs; d_outputs = s_d_outputs; d_lengths = s_d_lengths;

        // Build compact host arrays (only non-empty items)
        m = nonEmptyCount;
        h_inputs_c.resize(m, nullptr);
        h_outputs_ptrs_c.resize(m, nullptr);
        lengths_c.resize(m, 0);
        offsets_c.resize(m, 0);
        for (int j = 0; j < m; ++j) {
            int i = nonEmptyIdx[j];
            lengths_c[j] = lengths[i];
            offsets_c[j] = offsets[i];
            h_inputs_c[j] = d_all_inputs + offsets_c[j];
            h_outputs_ptrs_c[j] = d_all_outputs + j * 4; // compact contiguous outputs
        }

        // Split work into two halves (bytes-balanced on compact arrays)
        {
            const size_t target = totalBytes / 2;
            int splitIndex = m; // default: all in first half
            for (int j = 0; j < m; ++j) {
                if (offsets_c[j] >= target) { splitIndex = j; break; }
            }
            mid = splitIndex;
        }
        cnt1 = mid;
        cnt2 = m - mid;
        if (cnt1 > 0 && mid < m) {
            bytes1 = offsets_c[mid];
        } else if (cnt1 > 0 && mid == m) {
            bytes1 = totalBytes;
        } else {
            bytes1 = 0;
        }
        startOffset2 = (mid < m) ? offsets_c[mid] : totalBytes;
        bytes2 = totalBytes - startOffset2;
        outBytes1 = outWordsPerItem * sizeof(unsigned int) * cnt1;
        outBytes2 = outWordsPerItem * sizeof(unsigned int) * cnt2;

        // Copy pointer arrays and lengths asynchronously on stream1, and fence with an event for stream2
        if (!CUDA_TRY(cudaMemcpyAsync(d_inputs, h_inputs_c.data(), sizeof(unsigned char*) * m, cudaMemcpyHostToDevice, s_stream1))) { ok = false; break; }
        if (!CUDA_TRY(cudaMemcpyAsync(d_outputs, h_outputs_ptrs_c.data(), sizeof(unsigned int*) * m, cudaMemcpyHostToDevice, s_stream1))) { ok = false; break; }
        if (!CUDA_TRY(cudaMemcpyAsync(d_lengths, lengths_c.data(), sizeof(size_t) * m, cudaMemcpyHostToDevice, s_stream1))) { ok = false; break; }
        if (!CUDA_TRY(cudaEventRecord(s_evtPtrReady, s_stream1))) { ok = false; break; }
        started1 = true;

        // H2D for each half
        if (cnt1 > 0 && bytes1 > 0) {
            if (!CUDA_TRY(cudaMemcpyAsync(d_all_inputs, h_all_inputs, bytes1, cudaMemcpyHostToDevice, s_stream1))) { ok = false; break; }
            started1 = true;
        }
        if (cnt2 > 0 && bytes2 > 0) {
            if (!CUDA_TRY(cudaMemcpyAsync(d_all_inputs + startOffset2, h_all_inputs + startOffset2, bytes2, cudaMemcpyHostToDevice, s_stream2))) { ok = false; break; }
            started2 = true;
        }

        // Launch kernels per half
        block = dim3(256);
        if (cnt1 > 0) {
            grid = dim3((cnt1 + block.x - 1) / block.x);
            md5_batch_kernel<<<grid, block, 0, s_stream1>>>(const_cast<const unsigned char**>(d_inputs), const_cast<unsigned int**>(d_outputs), d_lengths, cnt1);
            if (!CUDA_TRY_KERNEL_NOSYNC()) { ok = false; break; }
        }
        if (cnt2 > 0) {
            grid = dim3((cnt2 + block.x - 1) / block.x);
            // Ensure stream2 observes pointer arrays/lengths updates
            if (!CUDA_TRY(cudaStreamWaitEvent(s_stream2, s_evtPtrReady, 0))) { ok = false; break; }
            md5_batch_kernel<<<grid, block, 0, s_stream2>>>(const_cast<const unsigned char**>(d_inputs + mid), const_cast<unsigned int**>(d_outputs + mid), d_lengths + mid, cnt2);
            if (!CUDA_TRY_KERNEL_NOSYNC()) { ok = false; break; }
        }

        // D2H for each half
        if (cnt1 > 0 && outBytes1 > 0) {
            if (!CUDA_TRY(cudaMemcpyAsync(h_all_outputs, d_all_outputs, outBytes1, cudaMemcpyDeviceToHost, s_stream1))) { ok = false; break; }
        }
        if (cnt2 > 0 && outBytes2 > 0) {
            if (!CUDA_TRY(cudaMemcpyAsync(h_all_outputs + outWordsPerItem * mid, d_all_outputs + outWordsPerItem * mid, outBytes2, cudaMemcpyDeviceToHost, s_stream2))) { ok = false; break; }
        }

        // Sync
        if (!CUDA_TRY(cudaStreamSynchronize(s_stream1))) { ok = false; break; }
        if (!CUDA_TRY(cudaStreamSynchronize(s_stream2))) { ok = false; break; }

        // Fill result strings
        for (int j = 0; j < m; ++j) {
            int i = nonEmptyIdx[j];
            unsigned char* bytes = reinterpret_cast<unsigned char*>(h_all_outputs + j * 4);
            results[i] = BytesToHexString(bytes, 16);
        }
    } while (false);

    if (!ok) {
        // Drain any started async ops to keep streams consistent for next calls
        if (started1) cudaStreamSynchronize(s_stream1);
        if (started2) cudaStreamSynchronize(s_stream2);
    }

    return results;
}

std::vector<std::string> CudaHashCalculator::CalculateBatchSHA1_GPU(const std::vector<std::vector<unsigned char>>& dataList) {
    std::lock_guard<std::mutex> lock(s_mutex);
    std::vector<std::string> results;
    if (!isCudaAvailable || dataList.empty()) return results;

    const int n = static_cast<int>(dataList.size());
    results.resize(n);

    // Pre-declare to avoid goto crossing initialization
    std::vector<unsigned char*> h_inputs(n, nullptr);
    std::vector<unsigned int*> h_outputs_ptrs(n, nullptr);
    // Predeclare compact arrays to avoid goto-bypassed initialization
    int m1 = 0;
    std::vector<unsigned char*> h_inputs_c1;
    std::vector<unsigned int*> h_outputs_ptrs_c1;
    std::vector<size_t> lengths_c1;
    std::vector<size_t> offsets_c1;
    dim3 block;
    dim3 grid;
    // Predeclare split variables
    int mid1;
    int cnt1_1;
    int cnt2_1;
    size_t bytes1_1;
    size_t bytes2_1;
    const size_t sha1Words = 5;
    size_t outBytes1_1;
    size_t outBytes2_1;
    size_t startOffset2_1;

    int nonEmptyCount = 0;
    std::vector<int> nonEmptyIdx1;
    std::vector<size_t> lengths(n, 0);
    std::vector<size_t> offsets(n, 0);
    size_t totalBytes = 0;
    for (int i = 0; i < n; ++i) {
        lengths[i] = dataList[i].size();
        if (lengths[i] == 0) {
            // SHA1("") standard digest
            results[i] = "da39a3ee5e6b4b0d3255bfef95601890afd80709";
        } else {
            offsets[i] = totalBytes;
            totalBytes += lengths[i];
            ++nonEmptyCount;
            nonEmptyIdx1.push_back(i);
        }
    }
    if (nonEmptyCount == 0) return results;

    if (!EnsureDeviceBatchBufferCapacity(totalBytes, sizeof(unsigned int) * 5 * n)) return results;
    unsigned char* h_all_inputs = s_h_all_inputs;
    unsigned int* h_all_outputs = s_h_all_outputs; // 5 words per SHA1
    for (int i = 0; i < n; ++i) if (lengths[i] > 0) memcpy(h_all_inputs + offsets[i], dataList[i].data(), lengths[i]);

    unsigned char* d_all_inputs = nullptr;
    unsigned int* d_all_outputs = nullptr;
    unsigned char** d_inputs = nullptr;
    unsigned int** d_outputs = nullptr;
    size_t* d_lengths = nullptr;

    bool ok = true; bool started1 = false, started2 = false;
    do {
        if (!EnsureDeviceBatchBufferCapacity(totalBytes, sizeof(unsigned int) * 5 * n)) { ok = false; break; }
        d_all_inputs = s_d_all_inputs;
        d_all_outputs = s_d_all_outputs;
        m1 = nonEmptyCount;
        if (!EnsureDevicePointerCapacity(m1)) { ok = false; break; }
        d_inputs = s_d_inputs; d_outputs = s_d_outputs; d_lengths = s_d_lengths;

        // Build compact arrays for non-empty entries
        h_inputs_c1.resize(m1, nullptr);
        h_outputs_ptrs_c1.resize(m1, nullptr);
        lengths_c1.resize(m1, 0);
        offsets_c1.resize(m1, 0);
        for (int j = 0; j < m1; ++j) {
            int i = nonEmptyIdx1[j];
            lengths_c1[j] = lengths[i];
            offsets_c1[j] = offsets[i];
            h_inputs_c1[j] = d_all_inputs + offsets_c1[j];
            h_outputs_ptrs_c1[j] = d_all_outputs + j * 5;
        }

        // Compute split variables
        {
            const size_t target1 = totalBytes / 2;
            int splitIndex1 = m1;
            for (int j = 0; j < m1; ++j) {
                if (offsets_c1[j] >= target1) { splitIndex1 = j; break; }
            }
            mid1 = splitIndex1;
        }
        cnt1_1 = mid1;
        cnt2_1 = m1 - mid1;
        if (cnt1_1 > 0 && mid1 < m1) {
            bytes1_1 = offsets_c1[mid1];
        } else if (cnt1_1 > 0 && mid1 == m1) {
            bytes1_1 = totalBytes;
        } else {
            bytes1_1 = 0;
        }
        startOffset2_1 = (mid1 < m1) ? offsets_c1[mid1] : totalBytes;
        bytes2_1 = totalBytes - startOffset2_1;
        outBytes1_1 = sha1Words * sizeof(unsigned int) * cnt1_1;
        outBytes2_1 = sha1Words * sizeof(unsigned int) * cnt2_1;

        // Copy pointer arrays and lengths asynchronously on stream1, and fence with an event for stream2
        if (!CUDA_TRY(cudaMemcpyAsync(d_inputs, h_inputs_c1.data(), sizeof(unsigned char*) * m1, cudaMemcpyHostToDevice, s_stream1))) { ok = false; break; }
        if (!CUDA_TRY(cudaMemcpyAsync(d_outputs, h_outputs_ptrs_c1.data(), sizeof(unsigned int*) * m1, cudaMemcpyHostToDevice, s_stream1))) { ok = false; break; }
        if (!CUDA_TRY(cudaMemcpyAsync(d_lengths, lengths_c1.data(), sizeof(size_t) * m1, cudaMemcpyHostToDevice, s_stream1))) { ok = false; break; }
        if (!CUDA_TRY(cudaEventRecord(s_evtPtrReady, s_stream1))) { ok = false; break; }
        started1 = true;

        if (cnt1_1 > 0 && bytes1_1 > 0) {
            if (!CUDA_TRY(cudaMemcpyAsync(d_all_inputs, h_all_inputs, bytes1_1, cudaMemcpyHostToDevice, s_stream1))) { ok = false; break; }
            started1 = true;
        }
        if (cnt2_1 > 0 && bytes2_1 > 0) {
            if (!CUDA_TRY(cudaMemcpyAsync(d_all_inputs + startOffset2_1, h_all_inputs + startOffset2_1, bytes2_1, cudaMemcpyHostToDevice, s_stream2))) { ok = false; break; }
            started2 = true;
        }

        block = dim3(256);
        if (cnt1_1 > 0) {
            grid = dim3((cnt1_1 + block.x - 1) / block.x);
            sha1_batch_kernel<<<grid, block, 0, s_stream1>>>(const_cast<const unsigned char**>(d_inputs), const_cast<unsigned int**>(d_outputs), d_lengths, cnt1_1);
            if (!CUDA_TRY_KERNEL_NOSYNC()) { ok = false; break; }
        }
        if (cnt2_1 > 0) {
            grid = dim3((cnt2_1 + block.x - 1) / block.x);
            // Ensure stream2 observes pointer arrays/lengths updates
            if (!CUDA_TRY(cudaStreamWaitEvent(s_stream2, s_evtPtrReady, 0))) { ok = false; break; }
            sha1_batch_kernel<<<grid, block, 0, s_stream2>>>(const_cast<const unsigned char**>(d_inputs + mid1), const_cast<unsigned int**>(d_outputs + mid1), d_lengths + mid1, cnt2_1);
            if (!CUDA_TRY_KERNEL_NOSYNC()) { ok = false; break; }
        }

        if (cnt1_1 > 0 && outBytes1_1 > 0) {
            if (!CUDA_TRY(cudaMemcpyAsync(h_all_outputs, d_all_outputs, outBytes1_1, cudaMemcpyDeviceToHost, s_stream1))) { ok = false; break; }
        }
        if (cnt2_1 > 0 && outBytes2_1 > 0) {
            if (!CUDA_TRY(cudaMemcpyAsync(h_all_outputs + sha1Words * mid1, d_all_outputs + sha1Words * mid1, outBytes2_1, cudaMemcpyDeviceToHost, s_stream2))) { ok = false; break; }
        }

        if (!CUDA_TRY(cudaStreamSynchronize(s_stream1))) { ok = false; break; }
        if (!CUDA_TRY(cudaStreamSynchronize(s_stream2))) { ok = false; break; }

        for (int j = 0; j < m1; ++j) {
            int i = nonEmptyIdx1[j];
            std::stringstream ss;
            ss << std::hex << std::setfill('0');
            for (int w = 0; w < 5; ++w) ss << std::setw(8) << h_all_outputs[j * 5 + w];
            results[i] = ss.str();
        }
    } while(false);

    if (!ok) {
        if (started1) cudaStreamSynchronize(s_stream1);
        if (started2) cudaStreamSynchronize(s_stream2);
    }

    return results;
}

std::vector<std::string> CudaHashCalculator::CalculateBatchSHA256_GPU(const std::vector<std::vector<unsigned char>>& dataList) {
    std::lock_guard<std::mutex> lock(s_mutex);
    std::vector<std::string> results;
    if (!isCudaAvailable || dataList.empty()) return results;

    const int n = static_cast<int>(dataList.size());
    results.resize(n);

    // Pre-declare to avoid goto crossing initialization
    std::vector<unsigned char*> h_inputs(n, nullptr);
    std::vector<unsigned int*> h_outputs_ptrs(n, nullptr);
    // Predeclare compact arrays to avoid goto-bypassed initialization
    int m2 = 0;
    std::vector<unsigned char*> h_inputs_c2;
    std::vector<unsigned int*> h_outputs_ptrs_c2;
    std::vector<size_t> lengths_c2;
    std::vector<size_t> offsets_c2;
    dim3 block;
    dim3 grid;
    // Predeclare split variables
    int mid2;
    int cnt1_2;
    int cnt2_2;
    size_t bytes1_2;
    size_t bytes2_2;
    const size_t sha256Words = 8;
    size_t outBytes1_2;
    size_t outBytes2_2;
    size_t startOffset2_2;

    int nonEmptyCount = 0;
    std::vector<int> nonEmptyIdx2;
    std::vector<size_t> lengths(n, 0);
    std::vector<size_t> offsets(n, 0);
    size_t totalBytes = 0;
    for (int i = 0; i < n; ++i) {
        lengths[i] = dataList[i].size();
        if (lengths[i] == 0) {
            // SHA256("") standard digest
            results[i] = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        } else {
            offsets[i] = totalBytes;
            totalBytes += lengths[i];
            ++nonEmptyCount;
            nonEmptyIdx2.push_back(i);
        }
    }
    if (nonEmptyCount == 0) return results;

    if (!EnsureHostBatchBufferCapacity(totalBytes, sizeof(unsigned int) * 8 * n)) return results;
    unsigned char* h_all_inputs = s_h_all_inputs;
    unsigned int* h_all_outputs = s_h_all_outputs; // 8 words per SHA256
    for (int i = 0; i < n; ++i) if (lengths[i] > 0) memcpy(h_all_inputs + offsets[i], dataList[i].data(), lengths[i]);

    unsigned char* d_all_inputs = nullptr;
    unsigned int* d_all_outputs = nullptr;
    unsigned char** d_inputs = nullptr;
    unsigned int** d_outputs = nullptr;
    size_t* d_lengths = nullptr;

    bool ok = true; bool started1 = false, started2 = false;
    do {
        if (!EnsureDeviceBatchBufferCapacity(totalBytes, sizeof(unsigned int) * 8 * n)) { ok = false; break; }
        d_all_inputs = s_d_all_inputs;
        d_all_outputs = s_d_all_outputs;
        m2 = nonEmptyCount;
        if (!EnsureDevicePointerCapacity(m2)) { ok = false; break; }
        d_inputs = s_d_inputs; d_outputs = s_d_outputs; d_lengths = s_d_lengths;

        // Build compact arrays for non-empty entries
        h_inputs_c2.resize(m2, nullptr);
        h_outputs_ptrs_c2.resize(m2, nullptr);
        lengths_c2.resize(m2, 0);
        offsets_c2.resize(m2, 0);
        for (int j = 0; j < m2; ++j) {
            int i = nonEmptyIdx2[j];
            lengths_c2[j] = lengths[i];
            offsets_c2[j] = offsets[i];
            h_inputs_c2[j] = d_all_inputs + offsets_c2[j];
            h_outputs_ptrs_c2[j] = d_all_outputs + j * 8;
        }

        if (!CUDA_TRY(cudaMemcpyAsync(d_inputs, h_inputs_c2.data(), sizeof(unsigned char*) * m2, cudaMemcpyHostToDevice, s_stream1))) { ok = false; break; }
        if (!CUDA_TRY(cudaMemcpyAsync(d_outputs, h_outputs_ptrs_c2.data(), sizeof(unsigned int*) * m2, cudaMemcpyHostToDevice, s_stream1))) { ok = false; break; }
        if (!CUDA_TRY(cudaMemcpyAsync(d_lengths, lengths_c2.data(), sizeof(size_t) * m2, cudaMemcpyHostToDevice, s_stream1))) { ok = false; break; }
        if (!CUDA_TRY(cudaEventRecord(s_evtPtrReady, s_stream1))) { ok = false; break; }
        started1 = true;

        {
            const size_t target2 = totalBytes / 2;
            int splitIndex2 = m2;
            for (int j = 0; j < m2; ++j) {
                if (offsets_c2[j] >= target2) { splitIndex2 = j; break; }
            }
            mid2 = splitIndex2;
        }
        cnt1_2 = mid2;
        cnt2_2 = m2 - mid2;
        if (cnt1_2 > 0 && mid2 < m2) {
            bytes1_2 = offsets_c2[mid2];
        } else if (cnt1_2 > 0 && mid2 == m2) {
            bytes1_2 = totalBytes;
        } else {
            bytes1_2 = 0;
        }
        startOffset2_2 = (mid2 < m2) ? offsets_c2[mid2] : totalBytes;
        bytes2_2 = totalBytes - startOffset2_2;

        if (cnt1_2 > 0 && bytes1_2 > 0) {
            if (!CUDA_TRY(cudaMemcpyAsync(d_all_inputs, h_all_inputs, bytes1_2, cudaMemcpyHostToDevice, s_stream1))) { ok = false; break; }
            started1 = true;
        }
        if (cnt2_2 > 0 && bytes2_2 > 0) {
            if (!CUDA_TRY(cudaMemcpyAsync(d_all_inputs + startOffset2_2, h_all_inputs + startOffset2_2, bytes2_2, cudaMemcpyHostToDevice, s_stream2))) { ok = false; break; }
            started2 = true;
        }

        block = dim3(256);
        if (cnt1_2 > 0) {
            grid = dim3((cnt1_2 + block.x - 1) / block.x);
            sha256_batch_kernel<<<grid, block, 0, s_stream1>>>(const_cast<const unsigned char**>(d_inputs), const_cast<unsigned int**>(d_outputs), d_lengths, cnt1_2);
            if (!CUDA_TRY_KERNEL_NOSYNC()) { ok = false; break; }
        }
        if (cnt2_2 > 0) {
            grid = dim3((cnt2_2 + block.x - 1) / block.x);
            // Ensure stream2 observes pointer arrays/lengths updates
            if (!CUDA_TRY(cudaStreamWaitEvent(s_stream2, s_evtPtrReady, 0))) { ok = false; break; }
            sha256_batch_kernel<<<grid, block, 0, s_stream2>>>(const_cast<const unsigned char**>(d_inputs + mid2), const_cast<unsigned int**>(d_outputs + mid2), d_lengths + mid2, cnt2_2);
            if (!CUDA_TRY_KERNEL_NOSYNC()) { ok = false; break; }
        }

        outBytes1_2 = sha256Words * sizeof(unsigned int) * cnt1_2;
        outBytes2_2 = sha256Words * sizeof(unsigned int) * cnt2_2;
        if (cnt1_2 > 0 && outBytes1_2 > 0) {
            if (!CUDA_TRY(cudaMemcpyAsync(h_all_outputs, d_all_outputs, outBytes1_2, cudaMemcpyDeviceToHost, s_stream1))) { ok = false; break; }
        }
        if (cnt2_2 > 0 && outBytes2_2 > 0) {
            if (!CUDA_TRY(cudaMemcpyAsync(h_all_outputs + sha256Words * mid2, d_all_outputs + sha256Words * mid2, outBytes2_2, cudaMemcpyDeviceToHost, s_stream2))) { ok = false; break; }
        }

        if (!CUDA_TRY(cudaStreamSynchronize(s_stream1))) { ok = false; break; }
        if (!CUDA_TRY(cudaStreamSynchronize(s_stream2))) { ok = false; break; }

        for (int j = 0; j < m2; ++j) {
            int i = nonEmptyIdx2[j];
            std::stringstream ss;
            ss << std::hex << std::setfill('0');
            for (int w = 0; w < 8; ++w) ss << std::setw(8) << h_all_outputs[j * 8 + w];
            results[i] = ss.str();
        }
    } while(false);

    if (!ok) {
        if (started1) cudaStreamSynchronize(s_stream1);
        if (started2) cudaStreamSynchronize(s_stream2);
    }

    return results;
}

std::vector<std::string> CudaHashCalculator::CalculateBatchCRC32_GPU(const std::vector<std::vector<unsigned char>>& dataList) {
    std::lock_guard<std::mutex> lock(s_mutex);
    std::vector<std::string> results;
    if (!isCudaAvailable || dataList.empty()) return results;

    const int n = static_cast<int>(dataList.size());
    results.resize(n);

    // Pre-declare to avoid goto crossing initialization
    std::vector<unsigned char*> h_inputs(n, nullptr);
    std::vector<unsigned int*> h_outputs_ptrs(n, nullptr);
    // Predeclare compact arrays to avoid goto-bypassed initialization
    int m3 = 0;
    std::vector<unsigned char*> h_inputs_c3;
    std::vector<unsigned int*> h_outputs_ptrs_c3;
    std::vector<size_t> lengths_c3;
    std::vector<size_t> offsets_c3;
    dim3 block;
    dim3 grid;
    // Predeclare split variables
    int mid3;
    int cnt1_3;
    int cnt2_3;
    size_t bytes1_3;
    size_t bytes2_3;
    const size_t crcWords = 1;
    size_t outBytes1_3;
    size_t outBytes2_3;
    size_t startOffset2_3;

    int nonEmptyCount = 0;
    std::vector<int> nonEmptyIdx3;
    std::vector<size_t> lengths(n, 0);
    std::vector<size_t> offsets(n, 0);
    size_t totalBytes = 0;
    for (int i = 0; i < n; ++i) {
        lengths[i] = dataList[i].size();
        if (lengths[i] == 0) {
            // CRC32("") standard digest with init 0xFFFFFFFF and final XOR
            results[i] = "00000000";
        } else {
            offsets[i] = totalBytes;
            totalBytes += lengths[i];
            ++nonEmptyCount;
            nonEmptyIdx3.push_back(i);
        }
    }
    if (nonEmptyCount == 0) return results;

    if (!EnsureHostBatchBufferCapacity(totalBytes, sizeof(unsigned int) * n)) return results;
    unsigned char* h_all_inputs = s_h_all_inputs;
    unsigned int* h_all_outputs = s_h_all_outputs; // 1 word per CRC32
    for (int i = 0; i < n; ++i) if (lengths[i] > 0) memcpy(h_all_inputs + offsets[i], dataList[i].data(), lengths[i]);

    unsigned char* d_all_inputs = nullptr;
    unsigned int* d_all_outputs = nullptr;
    unsigned char** d_inputs = nullptr;
    unsigned int** d_outputs = nullptr;
    size_t* d_lengths = nullptr;

    bool ok = true; bool started1 = false, started2 = false;
    do {
        if (!EnsureDeviceBatchBufferCapacity(totalBytes, sizeof(unsigned int) * n)) { ok = false; break; }
        d_all_inputs = s_d_all_inputs;
        d_all_outputs = s_d_all_outputs;
        m3 = nonEmptyCount;
        if (!EnsureDevicePointerCapacity(m3)) { ok = false; break; }
        d_inputs = s_d_inputs; d_outputs = s_d_outputs; d_lengths = s_d_lengths;

        // Build compact arrays for non-empty entries
        h_inputs_c3.resize(m3, nullptr);
        h_outputs_ptrs_c3.resize(m3, nullptr);
        lengths_c3.resize(m3, 0);
        offsets_c3.resize(m3, 0);
        for (int j = 0; j < m3; ++j) {
            int i = nonEmptyIdx3[j];
            lengths_c3[j] = lengths[i];
            offsets_c3[j] = offsets[i];
            h_inputs_c3[j] = d_all_inputs + offsets_c3[j];
            h_outputs_ptrs_c3[j] = d_all_outputs + j;
        }

        if (!CUDA_TRY(cudaMemcpyAsync(d_inputs, h_inputs_c3.data(), sizeof(unsigned char*) * m3, cudaMemcpyHostToDevice, s_stream1))) { ok = false; break; }
        if (!CUDA_TRY(cudaMemcpyAsync(d_outputs, h_outputs_ptrs_c3.data(), sizeof(unsigned int*) * m3, cudaMemcpyHostToDevice, s_stream1))) { ok = false; break; }
        if (!CUDA_TRY(cudaMemcpyAsync(d_lengths, lengths_c3.data(), sizeof(size_t) * m3, cudaMemcpyHostToDevice, s_stream1))) { ok = false; break; }
        if (!CUDA_TRY(cudaEventRecord(s_evtPtrReady, s_stream1))) { ok = false; break; }
        started1 = true;

        {
            const size_t target3 = totalBytes / 2;
            int splitIndex3 = m3;
            for (int j = 0; j < m3; ++j) {
                if (offsets_c3[j] >= target3) { splitIndex3 = j; break; }
            }
            mid3 = splitIndex3;
        }
        cnt1_3 = mid3;
        cnt2_3 = m3 - mid3;
        if (cnt1_3 > 0 && mid3 < m3) {
            bytes1_3 = offsets_c3[mid3];
        } else if (cnt1_3 > 0 && mid3 == m3) {
            bytes1_3 = totalBytes;
        } else {
            bytes1_3 = 0;
        }
        startOffset2_3 = (mid3 < m3) ? offsets_c3[mid3] : totalBytes;
        bytes2_3 = totalBytes - startOffset2_3;

        if (cnt1_3 > 0 && bytes1_3 > 0) {
            if (!CUDA_TRY(cudaMemcpyAsync(d_all_inputs, h_all_inputs, bytes1_3, cudaMemcpyHostToDevice, s_stream1))) { ok = false; break; }
            started1 = true;
        }
        if (cnt2_3 > 0 && bytes2_3 > 0) {
            if (!CUDA_TRY(cudaMemcpyAsync(d_all_inputs + startOffset2_3, h_all_inputs + startOffset2_3, bytes2_3, cudaMemcpyHostToDevice, s_stream2))) { ok = false; break; }
            started2 = true;
        }

        block = dim3(256);
        if (cnt1_3 > 0) {
            grid = dim3((cnt1_3 + block.x - 1) / block.x);
            crc32_batch_kernel<<<grid, block, 0, s_stream1>>>(const_cast<const unsigned char**>(d_inputs), const_cast<unsigned int**>(d_outputs), d_lengths, cnt1_3);
            if (!CUDA_TRY_KERNEL_NOSYNC()) { ok = false; break; }
        }
        if (cnt2_3 > 0) {
            grid = dim3((cnt2_3 + block.x - 1) / block.x);
            // Ensure stream2 observes pointer arrays/lengths updates
            if (!CUDA_TRY(cudaStreamWaitEvent(s_stream2, s_evtPtrReady, 0))) { ok = false; break; }
            crc32_batch_kernel<<<grid, block, 0, s_stream2>>>(const_cast<const unsigned char**>(d_inputs + mid3), const_cast<unsigned int**>(d_outputs + mid3), d_lengths + mid3, cnt2_3);
            if (!CUDA_TRY_KERNEL_NOSYNC()) { ok = false; break; }
        }

        outBytes1_3 = crcWords * sizeof(unsigned int) * cnt1_3;
        outBytes2_3 = crcWords * sizeof(unsigned int) * cnt2_3;
        if (cnt1_3 > 0 && outBytes1_3 > 0) {
            if (!CUDA_TRY(cudaMemcpyAsync(h_all_outputs, d_all_outputs, outBytes1_3, cudaMemcpyDeviceToHost, s_stream1))) { ok = false; break; }
        }
        if (cnt2_3 > 0 && outBytes2_3 > 0) {
            if (!CUDA_TRY(cudaMemcpyAsync(h_all_outputs + crcWords * mid3, d_all_outputs + crcWords * mid3, outBytes2_3, cudaMemcpyDeviceToHost, s_stream2))) { ok = false; break; }
        }

        if (!CUDA_TRY(cudaStreamSynchronize(s_stream1))) { ok = false; break; }
        if (!CUDA_TRY(cudaStreamSynchronize(s_stream2))) { ok = false; break; }

        for (int j = 0; j < m3; ++j) {
            int i = nonEmptyIdx3[j];
            std::stringstream ss;
            ss << std::hex << std::setfill('0') << std::setw(8) << h_all_outputs[j];
            results[i] = ss.str();
        }
    } while(false);

    if (!ok) {
        if (started1) cudaStreamSynchronize(s_stream1);
        if (started2) cudaStreamSynchronize(s_stream2);
    }

    return results;
}
