#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <windows.h>
#include <wincrypt.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <thread>
#include <future>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>

#pragma comment(lib, "advapi32.lib")

// CUDA support (optional compilation)
#ifdef USE_CUDA
    #include "cuda_hash.cuh"
    #pragma comment(lib, "cudart.lib")
    #define CUDA_AVAILABLE true
#else
    #define CUDA_AVAILABLE false
#endif

namespace fs = std::filesystem;

// High-performance thread pool with work-stealing and CPU affinity
class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
    std::atomic<size_t> active_threads{0};

    // Set CPU affinity for better performance
    void SetThreadAffinity(size_t thread_id) {
        DWORD_PTR mask = 1ULL << (thread_id % std::thread::hardware_concurrency());
        SetThreadAffinityMask(GetCurrentThread(), mask);
    }

public:
    ThreadPool(size_t threads) : stop(false) {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this, i] {
                SetThreadAffinity(i);
                
                // Set high priority for hash calculation threads
                SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
                
                for (;;) {
                    std::function<void()> task;
                    
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                        
                        if (this->stop && this->tasks.empty())
                            return;
                        
                        if (!this->tasks.empty()) {
                            task = std::move(this->tasks.front());
                            this->tasks.pop();
                            active_threads.fetch_add(1);
                        }
                    }
                    
                    if (task) {
                        task();
                        active_threads.fetch_sub(1);
                    }
                }
            });
        }
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);

            if (stop)
                throw std::runtime_error("enqueue on stopped ThreadPool");

            tasks.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    size_t active_count() const {
        return active_threads.load();
    }

    size_t queue_size() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(queue_mutex));
        return tasks.size();
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers)
            worker.join();
    }
};

class FileRenamerCLI {
private:
    // Optimized hex conversion using lookup table with SIMD-friendly approach
    static const char HEX_CHARS[16];
    
    // GPU acceleration support
    static bool gpuInitialized;
    static bool useGPU;
    static size_t gpuMinFileSize;
    static int selectedDevice;
    
    // I/O performance tuning
    static size_t ioBufferKB;      // streaming buffer size in KB (for ReadFile)
    static size_t mmapChunkMB;     // chunk size for hashing memory-mapped views
    static size_t gpuFileCapMB;    // max file size (MB) to load into memory for GPU hashing
    static size_t gpuBatchBytesMB; // max total bytes per GPU mini-batch in batch mode
    static size_t gpuChunkMB;      // chunk size (MB) for single-file CRC32 GPU chunking
    
    // CPU-optimized hex conversion for better performance
    static std::string BytesToHexOptimized(const BYTE* data, DWORD length) {
        std::string result;
        result.reserve(length * 2);
        
        // Process multiple bytes at once when possible
        const DWORD* data32 = reinterpret_cast<const DWORD*>(data);
        DWORD fullWords = length / 4;
        
        // Process 4 bytes at a time for better cache performance
        for (DWORD i = 0; i < fullWords; i++) {
            DWORD word = data32[i];
            // Extract each byte and convert
            for (int j = 0; j < 4; j++) {
                BYTE b = (word >> (j * 8)) & 0xFF;
                result += HEX_CHARS[b >> 4];
                result += HEX_CHARS[b & 0x0F];
            }
        }
        
        // Handle remaining bytes
        for (DWORD i = fullWords * 4; i < length; i++) {
            result += HEX_CHARS[data[i] >> 4];
            result += HEX_CHARS[data[i] & 0x0F];
        }
        
        return result;
    }
    
    static std::string BytesToHex(const BYTE* data, DWORD length) {
        return BytesToHexOptimized(data, length);
    }

public:
    // Tuning setters
    static void SetGPUMinFileSizeBytes(size_t bytes) { gpuMinFileSize = bytes; }
    static void SetIOBufferKB(size_t kb) { ioBufferKB = (kb == 0 ? 64 : kb); }
    static void SetMmapChunkMB(size_t mb) { mmapChunkMB = (mb == 0 ? 1 : mb); }
    static void SetGPUFileCapMB(size_t mb) { gpuFileCapMB = (mb == 0 ? 64 : mb); }
    static void SetGPUBatchBytesMB(size_t mb) { gpuBatchBytesMB = (mb == 0 ? 256 : mb); }
    static void SetGPUChunkMB(size_t mb) { gpuChunkMB = (mb == 0 ? 8 : mb); }

private:
    static bool IsGPUSupportedAlgorithm(const std::string& algorithm) {
        return (algorithm == "MD5" || algorithm == "SHA1" || algorithm == "SHA256" || algorithm == "CRC32");
    }
public:
    // CRC32 combine using GF(2) matrices, operates on post-XOR (finalized) CRC values
    static uint32_t CRC32_Combine(uint32_t crc1, uint32_t crc2, uint64_t len2) {
        if (len2 == 0) return crc1 ^ crc2; // concatenating empty -> xor yields combined
        // Build the operator for len2 bytes of zero appended, then apply to crc1 and xor crc2.
        auto gf2_times = [](const uint32_t mat[32], uint32_t vec) -> uint32_t {
            uint32_t sum = 0;
            for (int i = 0; i < 32; ++i) {
                if (vec & 1u) sum ^= mat[i];
                vec >>= 1u;
            }
            return sum;
        };
        auto gf2_square = [&](uint32_t square[32], const uint32_t mat[32]) {
            for (int i = 0; i < 32; ++i) {
                square[i] = gf2_times(mat, mat[i]);
            }
        };
        const uint32_t POLY = 0xEDB88320u; // reversed polynomial
        uint32_t odd[32] = {0};
        uint32_t even[32] = {0};
        // operator for one zero bit
        odd[0] = POLY;
        uint32_t row = 1;
        for (int i = 1; i < 32; ++i) { odd[i] = row; row <<= 1; }
        // transform to one byte (8 bits) operator by repeated squaring
        gf2_square(even, odd);  // 2 bits
        gf2_square(odd, even);  // 4 bits
        gf2_square(even, odd);  // 8 bits -> even now holds 8-bit operator
        // Raise operator to len2 (bytes) power and apply to crc1
        uint32_t op[32];
        // start with op = even (8-bit)
        for (int i = 0; i < 32; ++i) op[i] = even[i];
        uint64_t n = len2;
        uint32_t combined = crc1;
        while (n) {
            if (n & 1ull) combined = gf2_times(op, combined);
            n >>= 1ull;
            if (!n) break;
            uint32_t next[32];
            gf2_square(next, op);
            for (int i = 0; i < 32; ++i) op[i] = next[i];
        }
        return combined ^ crc2;
    }

private:
    static std::string CalculateFileHashCRC32_ChunkedGPU(const fs::path& filePath) {
#ifndef USE_CUDA
        (void)filePath; return "";
#else
        // Open file
        HANDLE hFile = CreateFileW(filePath.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
        if (hFile == INVALID_HANDLE_VALUE) return "";
        LARGE_INTEGER liSize; if (!GetFileSizeEx(hFile, &liSize)) { CloseHandle(hFile); return ""; }
        uint64_t totalSize = static_cast<uint64_t>(liSize.QuadPart);
        if (totalSize == 0) { CloseHandle(hFile); return std::string("00000000"); }

        const size_t chunkBytes = std::max<size_t>(gpuChunkMB * 1024ull * 1024ull, 1ull * 1024ull * 1024ull); // at least 1MB
        const size_t batchCapBytes = std::max<size_t>(gpuBatchBytesMB * 1024ull * 1024ull, chunkBytes);

        uint64_t offset = 0;
        uint32_t crcTotal = 0x00000000u; // finalized representation for empty prefix
        while (offset < totalSize) {
            // prepare one GPU mini-batch
            std::vector<std::vector<unsigned char>> chunks;
            chunks.reserve(static_cast<size_t>(batchCapBytes / chunkBytes) + 1);
            uint64_t bytesQueued = 0;
            std::vector<uint32_t> chunkLens;

            while (offset < totalSize && bytesQueued < batchCapBytes) {
                size_t toRead = static_cast<size_t>(std::min<uint64_t>(chunkBytes, totalSize - offset));
                std::vector<unsigned char> buf(toRead);
                DWORD read = 0; BOOL ok = ReadFile(hFile, buf.data(), static_cast<DWORD>(toRead), &read, NULL);
                if (!ok) { CloseHandle(hFile); return ""; }
                if (read == 0) break;
                buf.resize(read);
                chunks.emplace_back(std::move(buf));
                chunkLens.emplace_back(read);
                bytesQueued += read;
                offset += read;
            }
            if (chunks.empty()) break;

            // GPU batch CRC32 on chunks
            auto hashes = CudaHashCalculator::CalculateBatchCRC32_GPU(chunks);
            if (hashes.size() != chunks.size()) { CloseHandle(hFile); return ""; }

            // Combine with running CRC
            for (size_t i = 0; i < hashes.size(); ++i) {
                const std::string& hx = hashes[i];
                if (hx.size() != 8) { CloseHandle(hFile); return ""; }
                // parse hex to uint32
                uint32_t c = 0;
                for (char ch : hx) {
                    c <<= 4;
                    if (ch >= '0' && ch <= '9') c |= static_cast<uint32_t>(ch - '0');
                    else if (ch >= 'a' && ch <= 'f') c |= static_cast<uint32_t>(ch - 'a' + 10);
                    else if (ch >= 'A' && ch <= 'F') c |= static_cast<uint32_t>(ch - 'A' + 10);
                }
                crcTotal = CRC32_Combine(crcTotal, c, chunkLens[i]);
            }
        }
        CloseHandle(hFile);
        std::stringstream ss; ss << std::hex << std::setfill('0') << std::setw(8) << crcTotal;
        return ss.str();
#endif
    }
public:

    // GPU acceleration functions
    static bool InitializeGPU(int deviceId = -2) { // -2 means auto, -1 means CPU, >=0 means specific GPU
        if (gpuInitialized) return useGPU;
        
        gpuInitialized = true;
        
        // If specifically requesting CPU
        if (deviceId == -1) {
            ForceCPUMode();
            return false; // Return false for GPU, but this is expected for CPU
        }
        
#ifdef USE_CUDA
        // Lightweight CUDA availability check first
        int deviceCount = 0;
        cudaError_t cudaError = cudaGetDeviceCount(&deviceCount);
        
        if (cudaError != cudaSuccess || deviceCount == 0) {
            if (deviceId >= 0) {
                std::cout << "CUDA not available (" << cudaGetErrorString(cudaError) 
                          << "), cannot use GPU device " << deviceId << std::endl;
            }
            std::cout << "Using CPU processing (CUDA not available)" << std::endl;
            useGPU = false;
            selectedDevice = -1;
            return false;
        }
        
        if (CudaHashCalculator::Initialize()) {
            // If deviceId is specified and >= 0, try to select that device
            if (deviceId >= 0) {
                const auto& gpuInfo = CudaHashCalculator::GetGPUInfo();
                if (deviceId < gpuInfo.deviceCount) {
                    cudaError_t error = cudaSetDevice(deviceId);
                    if (error == cudaSuccess) {
                        selectedDevice = deviceId;
                        useGPU = true;
                        gpuMinFileSize = 4096; // 4KB minimum for GPU processing
                        std::cout << "GPU acceleration enabled on device " << deviceId 
                                  << " (" << gpuInfo.devices[deviceId].name << ")" << std::endl;
                        return true;
                    } else {
                        std::cout << "Failed to select device " << deviceId 
                                  << ": " << cudaGetErrorString(error) << std::endl;
                        std::cout << "Falling back to CPU processing..." << std::endl;
                    }
                } else {
                    std::cout << "Invalid device ID " << deviceId 
                              << ". Available GPU devices: 0-" << (gpuInfo.deviceCount - 1) << std::endl;
                    std::cout << "Falling back to CPU processing..." << std::endl;
                }
            }
            
            // Auto-selection (deviceId == -2) or fallback
            else if (deviceId == -2) {
                if (CudaHashCalculator::SelectBestDevice()) {
                    selectedDevice = CudaHashCalculator::GetGPUInfo().selectedDevice;
                    useGPU = true;
                    gpuMinFileSize = 4096; // 4KB minimum for GPU processing
                    std::cout << "GPU acceleration auto-enabled on device " << selectedDevice
                              << " (" << CudaHashCalculator::GetGPUInfo().devices[selectedDevice].name << ")" << std::endl;
                    return true;
                } else {
                    std::cout << "No suitable GPU found, using CPU processing" << std::endl;
                }
            }
        } else {
            if (deviceId >= 0) {
                std::cout << "CUDA initialization failed, cannot use GPU device " << deviceId << std::endl;
            }
            std::cout << "GPU acceleration not available, using CPU processing" << std::endl;
        }
#else
        if (deviceId >= 0) {
            std::cout << "CUDA support not compiled, cannot use GPU device " << deviceId << std::endl;
        }
        std::cout << "Using CPU processing (CUDA not available)" << std::endl;
#endif
        
        // Fallback to CPU
        useGPU = false;
        selectedDevice = -1;
        return false;
    }
    
    // Force CPU-only mode and cleanup any GPU resources
    static void ForceCPUMode() {
        if (gpuInitialized) {
            CleanupGPU();
        }
        useGPU = false;
        selectedDevice = -1;
        gpuInitialized = true; // Mark as initialized but in CPU mode
        std::cout << "Forced CPU-only mode enabled" << std::endl;
    }
    
    // Get current selected device
    static int GetSelectedDevice() {
        return selectedDevice;
    }
    
    // List available devices (CPU and GPU)
    static void ListAvailableDevices() {
        std::cout << "Available computing devices:" << std::endl;
        
        // Show detailed CPU information as device -1
        // Get CPU name from registry
        std::string cpuName = "Unknown CPU";
        HKEY hKey;
        if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0, KEY_READ, &hKey) == ERROR_SUCCESS) {
            char processorName[256];
            DWORD bufferSize = sizeof(processorName);
            if (RegQueryValueExA(hKey, "ProcessorNameString", NULL, NULL, (LPBYTE)processorName, &bufferSize) == ERROR_SUCCESS) {
                cpuName = std::string(processorName);
                // Trim whitespace
                cpuName.erase(0, cpuName.find_first_not_of(" \t\r\n"));
                cpuName.erase(cpuName.find_last_not_of(" \t\r\n") + 1);
            }
            RegCloseKey(hKey);
        }
        
        std::cout << "  Device -1: " << cpuName << std::endl;
        std::cout << "    Type: CPU (Multi-threaded)" << std::endl;
        std::cout << "    Logical Cores: " << std::thread::hardware_concurrency() << std::endl;
        
        // Get additional CPU information from Windows
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        std::cout << "    Physical Processors: " << sysInfo.dwNumberOfProcessors << std::endl;
        std::cout << "    Processor Architecture: ";
        switch (sysInfo.wProcessorArchitecture) {
            case PROCESSOR_ARCHITECTURE_AMD64:
                std::cout << "x64 (AMD64)";
                break;
            case PROCESSOR_ARCHITECTURE_ARM:
                std::cout << "ARM";
                break;
            case PROCESSOR_ARCHITECTURE_ARM64:
                std::cout << "ARM64";
                break;
            case PROCESSOR_ARCHITECTURE_INTEL:
                std::cout << "x86 (Intel)";
                break;
            default:
                std::cout << "Unknown (" << sysInfo.wProcessorArchitecture << ")";
                break;
        }
        std::cout << std::endl;
        
        // Get memory information
        MEMORYSTATUSEX memInfo;
        memInfo.dwLength = sizeof(MEMORYSTATUSEX);
        if (GlobalMemoryStatusEx(&memInfo)) {
            std::cout << "    Total RAM: " << (memInfo.ullTotalPhys / (1024*1024)) << " MB" << std::endl;
            std::cout << "    Available RAM: " << (memInfo.ullAvailPhys / (1024*1024)) << " MB" << std::endl;
            std::cout << "    Memory Usage: " << (100 - (memInfo.ullAvailPhys * 100 / memInfo.ullTotalPhys)) << "%" << std::endl;
        }
        
        if (selectedDevice == -1) {
            std::cout << "    Status: Currently selected" << std::endl;
        } else {
            std::cout << "    Status: Always available" << std::endl;
        }
        std::cout << std::endl;
        
#ifdef USE_CUDA
        // Don't force initialize CUDA just to list devices
        // Only check if CUDA is available but don't initialize full context
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        bool cudaAvailable = (error == cudaSuccess && deviceCount > 0);
        
        if (cudaAvailable) {
            if (gpuInitialized) {
                // If GPU is already initialized, show full info
                const auto& gpuInfo = CudaHashCalculator::GetGPUInfo();
                
                if (gpuInfo.deviceCount == 0) {
                    std::cout << "No CUDA-capable GPU devices found." << std::endl;
                } else {
                    for (int i = 0; i < gpuInfo.deviceCount; i++) {
                        const auto& device = gpuInfo.devices[i];
                        std::cout << "  Device " << i << ": " << device.name << std::endl;
                        std::cout << "    Type: GPU (CUDA)" << std::endl;
                        std::cout << "    Compute Capability: " << device.major << "." << device.minor << std::endl;
                        std::cout << "    Global Memory: " << (device.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
                        std::cout << "    Multiprocessors: " << device.multiProcessorCount << std::endl;
                        std::cout << "    Max Threads per Block: " << device.maxThreadsPerBlock << std::endl;
                        if (selectedDevice == i && useGPU) {
                            std::cout << "    Status: Currently selected" << std::endl;
                        }
                        std::cout << std::endl;
                    }
                }
            } else {
                // Show basic GPU info without full initialization
                std::cout << "CUDA-capable GPU devices detected: " << deviceCount << " device(s)" << std::endl;
                
                // Get basic device properties without initializing CUDA context
                for (int i = 0; i < deviceCount; i++) {
                    cudaDeviceProp prop;
                    cudaError_t error = cudaGetDeviceProperties(&prop, i);
                    
                    if (error == cudaSuccess) {
                        std::cout << "  Device " << i << ": " << prop.name << std::endl;
                        std::cout << "    Type: GPU (CUDA)" << std::endl;
                        std::cout << "    Compute Capability: " << prop.major << "." << prop.minor << std::endl;
                        std::cout << "    Global Memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
                        std::cout << "    Multiprocessors: " << prop.multiProcessorCount << std::endl;
                        std::cout << "    Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
                        std::cout << "    Status: Available (not initialized)" << std::endl;
                        std::cout << std::endl;
                    } else {
                        std::cout << "  Device " << i << ": <Error getting properties>" << std::endl;
                        std::cout << "    Error: " << cudaGetErrorString(error) << std::endl;
                        std::cout << std::endl;
                    }
                }
                
                std::cout << "Use -d <id> or -d auto to enable GPU acceleration." << std::endl;
            }
        } else {
            std::cout << "No CUDA-capable GPU devices available." << std::endl;
        }
#else
        std::cout << "CUDA support not compiled. Only CPU processing available." << std::endl;
#endif
        
        if (selectedDevice == -1) {
            std::cout << "Currently selected: Device -1 (CPU)" << std::endl;
        }
    }
    
    // Cleanup GPU resources
    static void CleanupGPU() {
#ifdef USE_CUDA
        if (useGPU || gpuInitialized) {
            CudaHashCalculator::Cleanup();
            // Force reset CUDA context to free memory
            cudaDeviceReset();
        }
#endif
        useGPU = false;
        selectedDevice = -1;
        gpuInitialized = false;
    }
    
    // Determine if file should be processed on GPU
    static bool ShouldUseGPU(size_t fileSize) {
        return useGPU && fileSize >= gpuMinFileSize;
    }
    
    static std::string CalculateMD5(const std::vector<BYTE>& data) {
        HCRYPTPROV hProv = 0;
        HCRYPTHASH hHash = 0;
        BYTE rgbHash[16];
        DWORD cbHash = 16;
        
        if (!CryptAcquireContext(&hProv, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT)) {
            return "";
        }
        
        if (!CryptCreateHash(hProv, CALG_MD5, 0, 0, &hHash)) {
            CryptReleaseContext(hProv, 0);
            return "";
        }
        
        if (!CryptHashData(hHash, data.data(), (DWORD)data.size(), 0)) {
            CryptDestroyHash(hHash);
            CryptReleaseContext(hProv, 0);
            return "";
        }
        
        if (!CryptGetHashParam(hHash, HP_HASHVAL, rgbHash, &cbHash, 0)) {
            CryptDestroyHash(hHash);
            CryptReleaseContext(hProv, 0);
            return "";
        }
        
        std::string result = BytesToHex(rgbHash, cbHash);
        
        CryptDestroyHash(hHash);
        CryptReleaseContext(hProv, 0);
        
        return result;
    }
    
    static std::string CalculateSHA1(const std::vector<BYTE>& data) {
        HCRYPTPROV hProv = 0;
        HCRYPTHASH hHash = 0;
        BYTE rgbHash[20];
        DWORD cbHash = 20;
        
        if (!CryptAcquireContext(&hProv, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT)) {
            return "";
        }
        
        if (!CryptCreateHash(hProv, CALG_SHA1, 0, 0, &hHash)) {
            CryptReleaseContext(hProv, 0);
            return "";
        }
        
        if (!CryptHashData(hHash, data.data(), (DWORD)data.size(), 0)) {
            CryptDestroyHash(hHash);
            CryptReleaseContext(hProv, 0);
            return "";
        }
        
        if (!CryptGetHashParam(hHash, HP_HASHVAL, rgbHash, &cbHash, 0)) {
            CryptDestroyHash(hHash);
            CryptReleaseContext(hProv, 0);
            return "";
        }
        
        std::string result = BytesToHex(rgbHash, cbHash);
        
        CryptDestroyHash(hHash);
        CryptReleaseContext(hProv, 0);
        
        return result;
    }
    
    // SHA256 implementation using Windows CNG API
    static std::string CalculateSHA256(const std::vector<BYTE>& data) {
        HCRYPTPROV hProv = 0;
        HCRYPTHASH hHash = 0;
        BYTE rgbHash[32];
        DWORD cbHash = 32;
        
        // Try to use enhanced provider for SHA256
        if (!CryptAcquireContext(&hProv, NULL, MS_ENH_RSA_AES_PROV, PROV_RSA_AES, CRYPT_VERIFYCONTEXT)) {
            // Fallback to RSA_FULL provider
            if (!CryptAcquireContext(&hProv, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT)) {
                return "";
            }
        }
        
        if (!CryptCreateHash(hProv, CALG_SHA_256, 0, 0, &hHash)) {
            CryptReleaseContext(hProv, 0);
            return "";
        }
        
        if (!CryptHashData(hHash, data.data(), (DWORD)data.size(), 0)) {
            CryptDestroyHash(hHash);
            CryptReleaseContext(hProv, 0);
            return "";
        }
        
        if (!CryptGetHashParam(hHash, HP_HASHVAL, rgbHash, &cbHash, 0)) {
            CryptDestroyHash(hHash);
            CryptReleaseContext(hProv, 0);
            return "";
        }
        
        std::string result = BytesToHex(rgbHash, cbHash);
        
        CryptDestroyHash(hHash);
        CryptReleaseContext(hProv, 0);
        
        return result;
    }
    
    // SHA512 implementation using Windows CNG API
    static std::string CalculateSHA512(const std::vector<BYTE>& data) {
        HCRYPTPROV hProv = 0;
        HCRYPTHASH hHash = 0;
        BYTE rgbHash[64];
        DWORD cbHash = 64;
        
        // Try to use enhanced provider for SHA512
        if (!CryptAcquireContext(&hProv, NULL, MS_ENH_RSA_AES_PROV, PROV_RSA_AES, CRYPT_VERIFYCONTEXT)) {
            // Fallback to RSA_FULL provider
            if (!CryptAcquireContext(&hProv, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT)) {
                return "";
            }
        }
        
        if (!CryptCreateHash(hProv, CALG_SHA_512, 0, 0, &hHash)) {
            CryptDestroyHash(hHash);
            CryptReleaseContext(hProv, 0);
            return "";
        }
        
        if (!CryptHashData(hHash, data.data(), (DWORD)data.size(), 0)) {
            CryptDestroyHash(hHash);
            CryptReleaseContext(hProv, 0);
            return "";
        }
        
        if (!CryptGetHashParam(hHash, HP_HASHVAL, rgbHash, &cbHash, 0)) {
            CryptDestroyHash(hHash);
            CryptReleaseContext(hProv, 0);
            return "";
        }
        
        std::string result = BytesToHex(rgbHash, cbHash);
        
        CryptDestroyHash(hHash);
        CryptReleaseContext(hProv, 0);
        
        return result;
    }
    
    // CRC32 implementation (IEEE 802.3)
    static std::string CalculateCRC32(const std::vector<BYTE>& data) {
        // CRC32 polynomial (IEEE 802.3)
        static const uint32_t CRC32_POLY = 0xEDB88320;
        static uint32_t crc_table[256];
        static bool table_initialized = false;
        
        // Initialize CRC table if not done already
        if (!table_initialized) {
            for (uint32_t i = 0; i < 256; i++) {
                uint32_t crc = i;
                for (int j = 0; j < 8; j++) {
                    if (crc & 1) {
                        crc = (crc >> 1) ^ CRC32_POLY;
                    } else {
                        crc >>= 1;
                    }
                }
                crc_table[i] = crc;
            }
            table_initialized = true;
        }
        
        uint32_t crc = 0xFFFFFFFF;
        for (BYTE byte : data) {
            crc = crc_table[(crc ^ byte) & 0xFF] ^ (crc >> 8);
        }
        crc ^= 0xFFFFFFFF;
        
        // Convert to hex string
        std::stringstream ss;
        ss << std::hex << std::setfill('0') << std::setw(8) << crc;
        return ss.str();
    }
    
    // Simple BLAKE2B implementation (256-bit output)
    static std::string CalculateBLAKE2B(const std::vector<BYTE>& data) {
        // Note: This is a simplified BLAKE2B implementation for demonstration
        // For production use, consider using a proper BLAKE2B library
        
        // BLAKE2B-256 initialization vector
        static const uint64_t IV[8] = {
            0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
            0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
            0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
            0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
        };
        
        // Simple hash based on modified SHA-1 for BLAKE2B-like behavior
        // This is a placeholder implementation - for real use, implement full BLAKE2B
        uint64_t h[8];
        for (int i = 0; i < 8; i++) {
            h[i] = IV[i] ^ 0x01010000 ^ 32; // 32-byte output length
        }
        
        // Process data in simple blocks (simplified)
        uint64_t hash = 0;
        for (size_t i = 0; i < data.size(); i++) {
            hash = ((hash << 5) + hash) + data[i];
            hash ^= IV[i % 8];
        }
        
        // Generate 32-byte hash (simplified)
        std::stringstream ss;
        for (int i = 0; i < 4; i++) {
            uint64_t part = hash ^ IV[i] ^ (i * 0x123456789ABCDEFULL);
            ss << std::hex << std::setfill('0') << std::setw(16) << part;
        }
        
        return ss.str();
    }
    
    // Generic hash calculation dispatcher with GPU acceleration
    static std::string CalculateHash(const std::vector<BYTE>& data, const std::string& algorithm) {
        // Initialize GPU if not already done
        if (!gpuInitialized) {
            InitializeGPU(-2); // Use auto-selection
        }
        
        // Try GPU acceleration for supported algorithms and large files
        if (ShouldUseGPU(data.size())) {
#ifdef USE_CUDA
            std::vector<unsigned char> gpu_data(data.begin(), data.end());
            
            if (algorithm == "MD5") {
                std::string result = CudaHashCalculator::CalculateMD5_GPU(gpu_data);
                if (!result.empty()) return result;
            } else if (algorithm == "SHA1") {
                std::string result = CudaHashCalculator::CalculateSHA1_GPU(gpu_data);
                if (!result.empty()) return result;
            } else if (algorithm == "SHA256") {
                std::string result = CudaHashCalculator::CalculateSHA256_GPU(gpu_data);
                if (!result.empty()) return result;
            } else if (algorithm == "CRC32") {
                std::string result = CudaHashCalculator::CalculateCRC32_GPU(gpu_data);
                if (!result.empty()) return result;
            }
            // Fall back to CPU if GPU fails
#endif
        }
        
        // CPU fallback
        if (algorithm == "MD5") {
            return CalculateMD5(data);
        } else if (algorithm == "SHA1") {
            return CalculateSHA1(data);
        } else if (algorithm == "SHA256") {
            return CalculateSHA256(data);
        } else if (algorithm == "SHA512") {
            return CalculateSHA512(data);
        } else if (algorithm == "CRC32") {
            return CalculateCRC32(data);
        } else if (algorithm == "BLAKE2B") {
            return CalculateBLAKE2B(data);
        } else {
            return "";
        }
    }
    
    static std::vector<BYTE> ReadFileData(const fs::path& filePath) {
        std::ifstream file(filePath, std::ios::binary);
        if (!file.is_open()) {
            return {};
        }
        
        file.seekg(0, std::ios::end);
        size_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);
        
        // For very large files, consider memory limitations
        if (fileSize > 100 * 1024 * 1024) { // 100MB threshold
            std::cerr << "Warning: Large file detected (" << fileSize / (1024*1024) << "MB): " << filePath.u8string() << std::endl;
        }
        
        std::vector<BYTE> data;
        data.reserve(fileSize);
        data.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        
        return data;
    }
    
    // High-throughput full-file read using WinAPI (for optional GPU hashing path)
    static std::vector<BYTE> ReadFileEntireWin(const fs::path& filePath) {
        std::vector<BYTE> data;
        HANDLE hFile = CreateFileW(filePath.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
        if (hFile == INVALID_HANDLE_VALUE) return data;
        LARGE_INTEGER li; li.QuadPart = 0;
        if (!GetFileSizeEx(hFile, &li) || li.QuadPart < 0) { CloseHandle(hFile); return data; }
        size_t total = static_cast<size_t>(li.QuadPart);
        data.resize(total);
        size_t offset = 0;
        const DWORD chunk = static_cast<DWORD>(std::min<size_t>(ioBufferKB * 1024ull, 4ull * 1024ull * 1024ull));
        DWORD bytesRead = 0; BOOL ok = FALSE;
        while (offset < total) {
            DWORD toRead = static_cast<DWORD>(std::min<size_t>(chunk, total - offset));
            ok = ReadFile(hFile, data.data() + offset, toRead, &bytesRead, NULL);
            if (!ok) { data.clear(); break; }
            if (bytesRead == 0) break;
            offset += bytesRead;
        }
        CloseHandle(hFile);
        if (offset != total) data.clear();
        return data;
    }
    
    // Helper function to get hash size and algorithm ID
    static std::pair<DWORD, ALG_ID> GetHashInfo(const std::string& algorithm) {
        if (algorithm == "MD5") {
            return {16, CALG_MD5};
        } else if (algorithm == "SHA1") {
            return {20, CALG_SHA1};
        } else if (algorithm == "SHA256") {
            return {32, CALG_SHA_256};
        } else if (algorithm == "SHA512") {
            return {64, CALG_SHA_512};
        } else if (algorithm == "CRC32") {
            return {4, 0}; // CRC32 doesn't use Windows Crypto API
        } else if (algorithm == "BLAKE2B") {
            return {32, 0}; // BLAKE2B doesn't use Windows Crypto API
        } else {
            return {0, 0};
        }
    }
    
    // Memory-mapped file hash calculation for very large files
    static std::string CalculateFileHashMemoryMapped(const fs::path& filePath, const std::string& algorithm = "MD5") {
        // For algorithms not supported by Windows Crypto API, use streaming (true streaming for CRC32/BLAKE2B)
        if (algorithm == "CRC32" || algorithm == "BLAKE2B") {
            return CalculateFileHashStreaming(filePath, algorithm);
        }
        
        HANDLE hFile = CreateFileW(filePath.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) {
            return "";
        }
        
        LARGE_INTEGER fileSize;
        if (!GetFileSizeEx(hFile, &fileSize)) {
            CloseHandle(hFile);
            return "";
        }
        
        // For files larger than 100MB, use memory mapping
        if (fileSize.QuadPart > 100 * 1024 * 1024) {
            HANDLE hMapping = CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
            if (hMapping == NULL) {
                CloseHandle(hFile);
                return CalculateFileHashStreaming(filePath, algorithm); // Fallback
            }
            
            const BYTE* pData = static_cast<const BYTE*>(MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0));
            if (pData == NULL) {
                CloseHandle(hMapping);
                CloseHandle(hFile);
                return CalculateFileHashStreaming(filePath, algorithm); // Fallback
            }
            
            // Calculate hash using memory-mapped data
            auto hashInfo = GetHashInfo(algorithm);
            DWORD hashSize = hashInfo.first;
            ALG_ID algId = hashInfo.second;
            
            if (hashSize == 0) {
                UnmapViewOfFile(pData);
                CloseHandle(hMapping);
                CloseHandle(hFile);
                return CalculateFileHashStreaming(filePath, algorithm);
            }
            
            HCRYPTPROV hProv = 0;
            HCRYPTHASH hHash = 0;
            std::string result;
            
            // Try enhanced provider first for newer algorithms
            bool useEnhanced = (algorithm == "SHA256" || algorithm == "SHA512");
            if (useEnhanced) {
                if (!CryptAcquireContext(&hProv, NULL, MS_ENH_RSA_AES_PROV, PROV_RSA_AES, CRYPT_VERIFYCONTEXT)) {
                    if (!CryptAcquireContext(&hProv, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT)) {
                        UnmapViewOfFile(pData);
                        CloseHandle(hMapping);
                        CloseHandle(hFile);
                        return CalculateFileHashStreaming(filePath, algorithm);
                    }
                }
            } else {
                if (!CryptAcquireContext(&hProv, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT)) {
                    UnmapViewOfFile(pData);
                    CloseHandle(hMapping);
                    CloseHandle(hFile);
                    return CalculateFileHashStreaming(filePath, algorithm);
                }
            }
            
            if (CryptCreateHash(hProv, algId, 0, 0, &hHash)) {
                // Process in chunks to avoid overwhelming the hash function
                const DWORD CHUNK_SIZE = static_cast<DWORD>(mmapChunkMB * 1024ull * 1024ull); // configurable MB
                const DWORD CHUNK_SIZE_SAFE = (CHUNK_SIZE == 0 ? (4 * 1024 * 1024) : CHUNK_SIZE);
                DWORD totalSize = static_cast<DWORD>(fileSize.QuadPart);
                
                for (DWORD offset = 0; offset < totalSize; offset += CHUNK_SIZE_SAFE) {
                    DWORD chunkSize = (CHUNK_SIZE_SAFE < totalSize - offset) ? CHUNK_SIZE_SAFE : (totalSize - offset);
                    if (!CryptHashData(hHash, pData + offset, chunkSize, 0)) {
                        break;
                    }
                }
                
                std::vector<BYTE> rgbHash(hashSize);
                DWORD cbHash = hashSize;
                
                if (CryptGetHashParam(hHash, HP_HASHVAL, rgbHash.data(), &cbHash, 0)) {
                    result = BytesToHex(rgbHash.data(), cbHash);
                }
                
                CryptDestroyHash(hHash);
            }
            CryptReleaseContext(hProv, 0);
            
            UnmapViewOfFile(pData);
            CloseHandle(hMapping);
            CloseHandle(hFile);
            
            return result.empty() ? CalculateFileHashStreaming(filePath, algorithm) : result;
        }
        
        CloseHandle(hFile);
        return CalculateFileHashStreaming(filePath, algorithm);
    }

    // Optimized streaming hash calculation for large files
    static std::string CalculateFileHashStreaming(const fs::path& filePath, const std::string& algorithm = "MD5") {
        // True streaming for CRC32 and BLAKE2B placeholder; WinCrypto for others with WinAPI I/O
        const DWORD desiredAccess = GENERIC_READ;
        const DWORD shareMode = FILE_SHARE_READ;
        const DWORD creationDisposition = OPEN_EXISTING;
        const DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN;

        HANDLE hFile = CreateFileW(filePath.wstring().c_str(), desiredAccess, shareMode, NULL, creationDisposition, flags, NULL);
        if (hFile == INVALID_HANDLE_VALUE) {
            return "";
        }

        const size_t BUFFER_SIZE = std::max<size_t>(ioBufferKB * 1024ull, 64 * 1024ull);
        std::vector<BYTE> buffer(BUFFER_SIZE);

        if (algorithm == "CRC32") {
            // Initialize CRC32 table
            static const uint32_t CRC32_POLY = 0xEDB88320;
            static uint32_t crc_table[256];
            static std::once_flag crc_once;
            std::call_once(crc_once, [](){
                for (uint32_t i = 0; i < 256; i++) {
                    uint32_t crc = i;
                    for (int j = 0; j < 8; j++) crc = (crc & 1) ? (crc >> 1) ^ CRC32_POLY : (crc >> 1);
                    crc_table[i] = crc;
                }
            });

            uint32_t crc = 0xFFFFFFFF;
            DWORD bytesRead = 0;
            BOOL ok = FALSE;
            do {
                ok = ReadFile(hFile, buffer.data(), static_cast<DWORD>(BUFFER_SIZE), &bytesRead, NULL);
                if (!ok) break;
                if (bytesRead == 0) break;
                for (DWORD i = 0; i < bytesRead; ++i) crc = crc_table[(crc ^ buffer[i]) & 0xFFu] ^ (crc >> 8);
            } while (bytesRead > 0);

            CloseHandle(hFile);
            if (!ok) return "";
            crc ^= 0xFFFFFFFFu;
            std::stringstream ss; ss << std::hex << std::setfill('0') << std::setw(8) << crc;
            return ss.str();
        }

        if (algorithm == "BLAKE2B") {
            // Stream the placeholder implementation
            static const uint64_t IV[8] = {
                0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
                0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
                0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
                0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
            };
            uint64_t hash = 0;
            DWORD bytesRead = 0; BOOL ok = FALSE;
            do {
                ok = ReadFile(hFile, buffer.data(), static_cast<DWORD>(BUFFER_SIZE), &bytesRead, NULL);
                if (!ok) break;
                if (bytesRead == 0) break;
                for (DWORD i = 0; i < bytesRead; ++i) {
                    hash = ((hash << 5) + hash) + buffer[i];
                    hash ^= IV[i % 8];
                }
            } while (bytesRead > 0);
            CloseHandle(hFile);
            if (!ok) return "";
            std::stringstream ss;
            for (int i = 0; i < 4; i++) {
                uint64_t part = hash ^ IV[i] ^ (static_cast<uint64_t>(i) * 0x123456789ABCDEFULL);
                ss << std::hex << std::setfill('0') << std::setw(16) << part;
            }
            return ss.str();
        }

        // Windows Crypto API streaming (MD5/SHA1/SHA256/SHA512)
        auto hashInfo = GetHashInfo(algorithm);
        DWORD hashSize = hashInfo.first;
        ALG_ID algId = hashInfo.second;
        if (hashSize == 0) { CloseHandle(hFile); return ""; }

        HCRYPTPROV hProv = 0;
        bool useEnhanced = (algorithm == "SHA256" || algorithm == "SHA512");
        if (useEnhanced) {
            if (!CryptAcquireContext(&hProv, NULL, MS_ENH_RSA_AES_PROV, PROV_RSA_AES, CRYPT_VERIFYCONTEXT)) {
                if (!CryptAcquireContext(&hProv, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT)) {
                    CloseHandle(hFile);
                    return "";
                }
            }
        } else {
            if (!CryptAcquireContext(&hProv, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT)) {
                CloseHandle(hFile);
                return "";
            }
        }

        HCRYPTHASH hHash = 0;
        if (!CryptCreateHash(hProv, algId, 0, 0, &hHash)) {
            CryptReleaseContext(hProv, 0);
            CloseHandle(hFile);
            return "";
        }

        DWORD bytesRead = 0; BOOL ok = FALSE;
        do {
            ok = ReadFile(hFile, buffer.data(), static_cast<DWORD>(BUFFER_SIZE), &bytesRead, NULL);
            if (!ok) break;
            if (bytesRead == 0) break;
            if (!CryptHashData(hHash, buffer.data(), bytesRead, 0)) { ok = FALSE; break; }
        } while (bytesRead > 0);

        CloseHandle(hFile);

        if (!ok) {
            CryptDestroyHash(hHash);
            CryptReleaseContext(hProv, 0);
            return "";
        }

        std::vector<BYTE> rgbHash(hashSize); DWORD cbHash = hashSize;
        if (!CryptGetHashParam(hHash, HP_HASHVAL, rgbHash.data(), &cbHash, 0)) {
            CryptDestroyHash(hHash);
            CryptReleaseContext(hProv, 0);
            return "";
        }

        std::string result = BytesToHex(rgbHash.data(), cbHash);
        CryptDestroyHash(hHash);
        CryptReleaseContext(hProv, 0);
        return result;
    }
    
    // Optimized hash calculation using Windows Crypto API with better error handling
    static std::string CalculateFileHashOptimized(const fs::path& filePath, const std::string& algorithm = "MD5") {
        // Get file size first for optimization decisions
        std::error_code ec;
        auto fileSize = fs::file_size(filePath, ec);
        
        if (ec) {
            return "";
        }
        
        // Special fast path: CRC32 single-file GPU chunked processing for large files
        if (algorithm == "CRC32" && useGPU) {
            if (fileSize >= gpuMinFileSize) {
                // Prefer chunked GPU path when file is larger than in-memory cap
                if (static_cast<uint64_t>(fileSize) > gpuFileCapMB * 1024ull * 1024ull) {
                    auto res = CalculateFileHashCRC32_ChunkedGPU(filePath);
                    if (!res.empty()) return res;
                    // on failure, fall back
                }
            }
        }

        // Fast path: if GPU available and file size within cap, load entire file and hash on GPU for supported algos
        if (useGPU && (algorithm == "MD5" || algorithm == "SHA1" || algorithm == "SHA256" || algorithm == "CRC32")) {
            if (fileSize >= gpuMinFileSize && fileSize <= gpuFileCapMB * 1024ull * 1024ull) {
                auto data = ReadFileEntireWin(filePath);
                if (!data.empty()) {
                    return CalculateHash(data, algorithm); // will route to GPU if above min size
                }
                // fallback if read failed -> continue
            }
        }
        
        // Use memory mapping for very large files (>100MB)
        if (fileSize > 100 * 1024 * 1024) {
            std::string result = CalculateFileHashMemoryMapped(filePath, algorithm);
            if (!result.empty()) {
                return result;
            }
        }
        
        // Use streaming for better memory efficiency and performance
        return CalculateFileHashStreaming(filePath, algorithm);
    }

    // Thread-safe hash calculation for parallel processing
    struct FileHashResult {
        fs::path originalPath;
        fs::path newPath;
        std::string hash;
        bool success;
        std::string error;
        bool needsRename;
    };

    static FileHashResult ProcessSingleFile(const fs::path& filePath, const std::string& algorithm, bool quickCheck) {
        FileHashResult result;
        result.originalPath = filePath;
        result.success = false;
        result.needsRename = false;
        
        try {
            // Quick check optimization
            if (quickCheck && QuickHashCheck(filePath, algorithm)) {
                result.success = true;
                result.needsRename = false;
                result.hash = "quick_skip";
                return result;
            }
            
            std::string hash = CalculateFileHashOptimized(filePath, algorithm);
            if (hash.empty()) {
                result.error = "Could not calculate hash";
                return result;
            }
            
            std::string extension = filePath.extension().string();
            std::string newFileName = hash + extension;
            fs::path newPath = filePath.parent_path() / newFileName;
            
            result.hash = hash;
            result.newPath = newPath;
            result.success = true;
            result.needsRename = (filePath.filename() != newFileName);
            
        } catch (const std::exception& ex) {
            result.error = ex.what();
        }
        
        return result;
    }
    
    static std::vector<fs::path> ScanDirectory(const fs::path& directory, bool recursive = false, const std::vector<std::string>& allowedExtensions = {}) {
        std::vector<fs::path> files;
        files.reserve(5000); // Increased reservation for better performance
        
        try {
            if (recursive) {
                for (const auto& entry : fs::recursive_directory_iterator(
                    directory, 
                    fs::directory_options::skip_permission_denied)) {
                    try {
                        if (entry.is_regular_file()) {
                            // Pre-filter by extension during scanning for better performance
                            if (allowedExtensions.empty() || ShouldProcessFileQuick(entry.path(), allowedExtensions)) {
                                files.push_back(entry.path());
                            }
                        }
                    } catch (const fs::filesystem_error& ex) {
                        std::cerr << "Warning: Error accessing file: " << ex.what() << std::endl;
                        continue;
                    }
                }
            } else {
                for (const auto& entry : fs::directory_iterator(
                    directory,
                    fs::directory_options::skip_permission_denied)) {
                    try {
                        if (entry.is_regular_file()) {
                            // Pre-filter by extension during scanning for better performance
                            if (allowedExtensions.empty() || ShouldProcessFileQuick(entry.path(), allowedExtensions)) {
                                files.push_back(entry.path());
                            }
                        }
                    } catch (const fs::filesystem_error& ex) {
                        std::cerr << "Warning: Error accessing file: " << ex.what() << std::endl;
                        continue;
                    }
                }
            }
        } catch (const fs::filesystem_error& ex) {
            std::cerr << "Error scanning directory: " << ex.what() << std::endl;
        }
        
        // Sort files by size for better processing order (small files first)
        std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
            std::error_code ec1, ec2;
            auto sizeA = fs::file_size(a, ec1);
            auto sizeB = fs::file_size(b, ec2);
            if (ec1 || ec2) return false;
            return sizeA < sizeB;
        });
        
        return files;
    }
    
    static bool RenameFile(const fs::path& oldPath, const fs::path& newPath) {
        try {
            if (fs::exists(newPath)) {
                std::cout << "Warning: Target file already exists: " << newPath.u8string() << std::endl;
                return false;
            }
            
            fs::rename(oldPath, newPath);
            return true;
        } catch (const fs::filesystem_error& ex) {
            std::cerr << "Error renaming file: " << ex.what() << std::endl;
            return false;
        }
    }
    
    // Check if filename looks like a valid hash
    static bool IsValidHashFilename(const std::string& filename, const std::string& algorithm) {
        size_t expectedLength;
        if (algorithm == "MD5") {
            expectedLength = 32;
        } else if (algorithm == "SHA1") {
            expectedLength = 40;
        } else if (algorithm == "SHA256") {
            expectedLength = 64;
        } else if (algorithm == "SHA512") {
            expectedLength = 128;
        } else if (algorithm == "CRC32") {
            expectedLength = 8;
        } else if (algorithm == "BLAKE2B") {
            expectedLength = 64; // 32 bytes * 2 hex chars = 64 chars
        } else {
            return false; // Unknown algorithm
        }
        
        if (filename.length() < expectedLength) {
            return false;
        }
        
        // Check if first part is all hex characters
        for (size_t i = 0; i < expectedLength; i++) {
            char c = filename[i];
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) {
                return false;
            }
        }
        
        return true;
    }
    
    // Fast verification using partial hash (first 1KB of file)
    static std::string CalculatePartialHash(const fs::path& filePath, const std::string& algorithm = "MD5") {
        std::ifstream file(filePath, std::ios::binary);
        if (!file.is_open()) {
            return "";
        }
        
        const size_t PARTIAL_SIZE = 1024; // Read first 1KB for quick verification
        std::vector<BYTE> buffer(PARTIAL_SIZE);
        
        file.read(reinterpret_cast<char*>(buffer.data()), PARTIAL_SIZE);
        std::streamsize bytesRead = file.gcount();
        
        if (bytesRead <= 0) {
            return "";
        }
        
        buffer.resize(static_cast<size_t>(bytesRead));
        
        return CalculateHash(buffer, algorithm);
    }
    
    // Check if file likely already has correct hash name (quick check)
    static bool QuickHashCheck(const fs::path& filePath, const std::string& algorithm) {
        std::string stem = filePath.stem().string();
        
        // Convert to lowercase for comparison
        std::transform(stem.begin(), stem.end(), stem.begin(), ::tolower);
        
        if (!IsValidHashFilename(stem, algorithm)) {
            return false;
        }
        
        // Quick partial hash check
        std::string partialHash = CalculatePartialHash(filePath, algorithm);
        if (partialHash.empty()) {
            return false;
        }
        
        // Convert partial hash to lowercase
        std::transform(partialHash.begin(), partialHash.end(), partialHash.begin(), ::tolower);
        
        // Check if stem starts with partial hash (first 8 characters)
        return stem.substr(0, 8) == partialHash.substr(0, 8);
    }

    static bool ShouldProcessFile(const fs::path& filePath, const std::vector<std::string>& allowedExtensions) {
        if (allowedExtensions.empty()) {
            return true; // Process all files if no filter specified
        }
        
        std::string fileExt = filePath.extension().string();
        
        // Convert to lowercase for case-insensitive comparison
        std::transform(fileExt.begin(), fileExt.end(), fileExt.begin(), ::tolower);
        
        for (const auto& ext : allowedExtensions) {
            std::string lowerExt = ext;
            std::transform(lowerExt.begin(), lowerExt.end(), lowerExt.begin(), ::tolower);
            
            // Add dot if not present
            if (!lowerExt.empty() && lowerExt[0] != '.') {
                lowerExt = "." + lowerExt;
            }
            
            if (fileExt == lowerExt) {
                return true;
            }
        }
        
        return false;
    }

    // Optimized version for scanning - avoids string operations where possible
    static bool ShouldProcessFileQuick(const fs::path& filePath, const std::vector<std::string>& allowedExtensions) {
        if (allowedExtensions.empty()) {
            return true;
        }
        
        const std::string& fileExt = filePath.extension().string();
        
        // Quick case-insensitive comparison without creating new strings
        for (const auto& ext : allowedExtensions) {
            if (fileExt.size() == ext.size() || 
                (ext[0] != '.' && fileExt.size() == ext.size() + 1)) {
                
                bool match = true;
                size_t start = (ext[0] == '.') ? 0 : 1; // Skip dot if needed
                
                for (size_t i = start; i < fileExt.size() && match; ++i) {
                    char fileChar = std::tolower(fileExt[i]);
                    char extChar = std::tolower(ext[i - start]);
                    if (fileChar != extChar) {
                        match = false;
                    }
                }
                
                if (match) return true;
            }
        }
        
        return false;
    }
    
    // Multi-threaded processing with buffered output to prevent output mixing
    static void ProcessDirectoryMultiThreaded(const std::string& directoryPath, const std::string& algorithm = "MD5", bool recursive = false, bool dryRun = true, const std::vector<std::string>& allowedExtensions = {}, bool quickCheck = true, int numThreads = 0) {
        auto startTime = std::chrono::high_resolution_clock::now();
        
        fs::path dir(directoryPath);
        
        if (!fs::exists(dir) || !fs::is_directory(dir)) {
            std::cerr << "Error: Invalid directory path: " << directoryPath << std::endl;
            return;
        }
        
        // Auto-detect number of threads if not specified
        if (numThreads <= 0) {
            numThreads = std::thread::hardware_concurrency();
            if (numThreads <= 0) numThreads = 4; // Fallback to 4 threads
        }
        
        std::cout << "Scanning directory: " << directoryPath << std::endl;
        std::cout << "Algorithm: " << algorithm << std::endl;
        std::cout << "Recursive: " << (recursive ? "Yes" : "No") << std::endl;
        std::cout << "Mode: " << (dryRun ? "Preview" : "Execute") << std::endl;
        std::cout << "Quick check: " << (quickCheck ? "Enabled" : "Disabled") << std::endl;
        std::cout << "Threads: " << numThreads << std::endl;
        
        // Display device status
        if (!gpuInitialized) {
            InitializeGPU(-2); // Use auto-selection
        }
        std::cout << "Processing device: " << (useGPU ? ("GPU " + std::to_string(selectedDevice)) : "CPU") << std::endl;
        if (useGPU) {
            std::cout << "GPU min file size: " << (gpuMinFileSize / 1024) << " KB" << std::endl;
        }
        
        if (!allowedExtensions.empty()) {
            std::cout << "Extensions filter: ";
            for (size_t i = 0; i < allowedExtensions.size(); i++) {
                std::cout << "\"" << allowedExtensions[i] << "\"";
                if (i < allowedExtensions.size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << std::endl;
        } else {
            std::cout << "Extensions filter: All files" << std::endl;
        }
        
        std::cout << "===========================================" << std::endl;
        
        auto scanStartTime = std::chrono::high_resolution_clock::now();
        auto files = ScanDirectory(dir, recursive, allowedExtensions);
        auto scanEndTime = std::chrono::high_resolution_clock::now();
        
        auto scanDuration = std::chrono::duration_cast<std::chrono::milliseconds>(scanEndTime - scanStartTime);
        std::cout << "Found " << files.size() << " files in " << scanDuration.count() << "ms." << std::endl << std::endl;
        
        if (files.empty()) {
            std::cout << "No files found to process." << std::endl;
            return;
        }
        
        // Thread-safe counters
        std::atomic<int> processedCount{0};
        std::atomic<int> successCount{0};
        std::atomic<int> skippedCount{0};
        std::atomic<int> noChangeCount{0};
        std::atomic<int> quickSkipCount{0};
        std::atomic<int> currentIndex{0};
        
        // Output buffer system to prevent mixing
        struct OutputBuffer {
            std::string content;
            size_t fileIndex;
            bool ready;
            
            OutputBuffer() : fileIndex(0), ready(false) {}
        };
        
        std::vector<OutputBuffer> outputBuffers(files.size());
        std::mutex outputMutex;
        std::atomic<size_t> nextOutputIndex{0};
        
        // Progress tracking
        std::atomic<size_t> progress{0};
        
        // Create thread pool
        std::vector<std::future<void>> futures;
        
        // Worker function
        auto worker = [&]() {
            while (true) {
                // Get next file index atomically
                size_t fileIndex = currentIndex.fetch_add(1);
                if (fileIndex >= files.size()) {
                    break;
                }
                
                const auto& file = files[fileIndex];
                std::stringstream buffer; // Local buffer for this file's output
                
                try {
                    std::string fileName = file.filename().string();
                    
                    // Build output in local buffer first
                    buffer << "[" << (fileIndex + 1) << "/" << files.size() << "] Processing: \"" << fileName << "\"" << std::endl;
                    
                    // Check if file extension matches filter
                    if (!ShouldProcessFile(file, allowedExtensions)) {
                        std::string extension = file.extension().string();
                        buffer << "  Extension: \"" << extension << "\"" << std::endl;
                        buffer << "  Status: Skipped (extension not in filter)" << std::endl;
                        buffer << std::endl;
                        skippedCount++;
                        
                        // Store in output buffer
                        outputBuffers[fileIndex].content = buffer.str();
                        outputBuffers[fileIndex].fileIndex = fileIndex;
                        outputBuffers[fileIndex].ready = true;
                        continue;
                    }
                    
                    // Quick check optimization for files that likely already have correct hash names
                    if (quickCheck && QuickHashCheck(file, algorithm)) {
                        buffer << "  Status: Likely already correctly named (quick check passed)" << std::endl;
                        buffer << std::endl;
                        quickSkipCount++;
                        
                        // Store in output buffer
                        outputBuffers[fileIndex].content = buffer.str();
                        outputBuffers[fileIndex].fileIndex = fileIndex;
                        outputBuffers[fileIndex].ready = true;
                        continue;
                    }
                    
                    processedCount++;
                    
                    std::string hash = CalculateFileHashOptimized(file, algorithm);
                    if (hash.empty()) {
                        buffer << "  Error: Could not calculate hash" << std::endl;
                        buffer << std::endl;
                        
                        // Store in output buffer
                        outputBuffers[fileIndex].content = buffer.str();
                        outputBuffers[fileIndex].fileIndex = fileIndex;
                        outputBuffers[fileIndex].ready = true;
                        continue;
                    }
                    
                    std::string extension = file.extension().string();
                    std::string newFileName = hash + extension;
                    fs::path newPath = file.parent_path() / newFileName;
                    
                    buffer << "  Hash (" << algorithm << "): " << hash << std::endl;
                    buffer << "  New name: " << newFileName << std::endl;
                    
                    // Check if the new filename is the same as the current filename
                    if (file.filename() == newFileName) {
                        buffer << "  Status: No change needed (filename already matches hash)" << std::endl;
                        buffer << std::endl;
                        noChangeCount++;
                    } else if (!dryRun) {
                        if (RenameFile(file, newPath)) {
                            buffer << "  Status: Renamed successfully" << std::endl;
                            buffer << std::endl;
                            successCount++;
                        } else {
                            buffer << "  Status: Failed to rename" << std::endl;
                            buffer << std::endl;
                        }
                    } else {
                        buffer << "  Status: Preview only (will be renamed)" << std::endl;
                        buffer << std::endl;
                    }
                    
                } catch (const std::exception& ex) {
                    buffer << "[" << (fileIndex + 1) << "/" << files.size() << "] Processing: <Error reading filename>" << std::endl;
                    buffer << "  Error: " << ex.what() << std::endl;
                    buffer << std::endl;
                    skippedCount++;
                }
                
                // Store completed output in buffer
                outputBuffers[fileIndex].content = buffer.str();
                outputBuffers[fileIndex].fileIndex = fileIndex;
                outputBuffers[fileIndex].ready = true;
                
                // Update progress
                progress.fetch_add(1);
            }
        };
        
        // Output thread to maintain proper order
        auto outputWorker = [&]() {
            while (nextOutputIndex.load() < files.size()) {
                size_t currentOutput = nextOutputIndex.load();
                if (currentOutput < files.size() && outputBuffers[currentOutput].ready) {
                    {
                        std::lock_guard<std::mutex> lock(outputMutex);
                        std::cout << outputBuffers[currentOutput].content << std::flush;
                    }
                    nextOutputIndex.fetch_add(1);
                } else {
                    // Wait a bit before checking again
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }
        };
        
        // Launch output thread
        std::future<void> outputFuture = std::async(std::launch::async, outputWorker);
        
        // Launch worker threads
        for (int i = 0; i < numThreads; i++) {
            futures.push_back(std::async(std::launch::async, worker));
        }
        
        // Wait for all threads to complete
        for (auto& future : futures) {
            future.wait();
        }
        
        // Wait for output thread to finish
        outputFuture.wait();
        
        std::cout << "===========================================" << std::endl;
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        std::cout << "Summary:" << std::endl;
        std::cout << "Total files: " << files.size() << std::endl;
        std::cout << "Processed: " << processedCount.load() << std::endl;
        std::cout << "Skipped (filter): " << skippedCount.load() << std::endl;
        if (quickCheck) {
            std::cout << "Quick-skipped (likely correct): " << quickSkipCount.load() << std::endl;
        }
        std::cout << "No change needed: " << noChangeCount.load() << std::endl;
        if (!dryRun) {
            std::cout << "Successfully renamed: " << successCount.load() << std::endl;
            std::cout << "Failed: " << (processedCount.load() - successCount.load() - noChangeCount.load()) << std::endl;
        }
        std::cout << "Total execution time: " << totalDuration.count() << "ms" << std::endl;
        
        if (processedCount.load() > 0) {
            double avgTimePerFile = static_cast<double>(totalDuration.count()) / processedCount.load();
            std::cout << "Average time per file: " << std::fixed << std::setprecision(2) << avgTimePerFile << "ms" << std::endl;
        }
        
        if (quickCheck && quickSkipCount.load() > 0) {
            std::cout << "Time saved by quick check: ~" << (quickSkipCount.load() * 10) << "ms (estimated)" << std::endl;
        }
        
        std::cout << "Performance: " << numThreads << " threads utilized with sequential output" << std::endl;
    }

    static void ProcessDirectory(const std::string& directoryPath, const std::string& algorithm = "MD5", bool recursive = false, bool dryRun = true, const std::vector<std::string>& allowedExtensions = {}, bool quickCheck = true) {
        auto startTime = std::chrono::high_resolution_clock::now();
        
        fs::path dir(directoryPath);
        
        if (!fs::exists(dir) || !fs::is_directory(dir)) {
            std::cerr << "Error: Invalid directory path: " << directoryPath << std::endl;
            return;
        }
        
        std::cout << "Scanning directory: " << directoryPath << std::endl;
        std::cout << "Algorithm: " << algorithm << std::endl;
        std::cout << "Recursive: " << (recursive ? "Yes" : "No") << std::endl;
        std::cout << "Mode: " << (dryRun ? "Preview" : "Execute") << std::endl;
        std::cout << "Quick check: " << (quickCheck ? "Enabled" : "Disabled") << std::endl;
        
        // Display device status
        if (!gpuInitialized) {
            InitializeGPU(-2); // Use auto-selection
        }
        std::cout << "Processing device: " << (useGPU ? ("GPU " + std::to_string(selectedDevice)) : "CPU") << std::endl;
        if (useGPU) {
            std::cout << "GPU min file size: " << (gpuMinFileSize / 1024) << " KB" << std::endl;
        }
        
        if (!allowedExtensions.empty()) {
            std::cout << "Extensions filter: ";
            for (size_t i = 0; i < allowedExtensions.size(); i++) {
                std::cout << "\"" << allowedExtensions[i] << "\"";
                if (i < allowedExtensions.size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << std::endl;
        } else {
            std::cout << "Extensions filter: All files" << std::endl;
        }
        
        std::cout << "===========================================" << std::endl;
        
        auto scanStartTime = std::chrono::high_resolution_clock::now();
        auto files = ScanDirectory(dir, recursive, allowedExtensions);
        auto scanEndTime = std::chrono::high_resolution_clock::now();
        
        auto scanDuration = std::chrono::duration_cast<std::chrono::milliseconds>(scanEndTime - scanStartTime);
        std::cout << "Found " << files.size() << " files in " << scanDuration.count() << "ms." << std::endl << std::endl;
        
        int processedCount = 0;
        int successCount = 0;
        int skippedCount = 0;
        int noChangeCount = 0;
        int quickSkipCount = 0;
        
        for (const auto& file : files) {
            try {
                std::string fileName = file.filename().string();
                std::cout << "[" << (processedCount + skippedCount + quickSkipCount + 1) << "/" << files.size() << "] Processing: \"" << fileName << "\"" << std::endl;
                
                // Check if file extension matches filter
                if (!ShouldProcessFile(file, allowedExtensions)) {
                    std::string extension = file.extension().string();
                    std::cout << "  Extension: \"" << extension << "\"" << std::endl;
                    std::cout << "  Status: Skipped (extension not in filter)" << std::endl;
                    skippedCount++;
                    std::cout << std::endl;
                    continue;
                }
                
                // Quick check optimization for files that likely already have correct hash names
                if (quickCheck && QuickHashCheck(file, algorithm)) {
                    std::cout << "  Status: Likely already correctly named (quick check passed)" << std::endl;
                    quickSkipCount++;
                    std::cout << std::endl;
                    continue;
                }
                
            } catch (const std::exception& ex) {
                std::cout << "[" << (processedCount + skippedCount + quickSkipCount + 1) << "/" << files.size() << "] Processing: <Error reading filename>" << std::endl;
                std::cout << "  Error: " << ex.what() << std::endl;
                skippedCount++;
                std::cout << std::endl;
                continue;
            }
            
            processedCount++;
            
            std::string hash = CalculateFileHashOptimized(file, algorithm);
            if (hash.empty()) {
                std::cout << "  Error: Could not calculate hash" << std::endl;
                continue;
            }
            
            std::string extension = file.extension().string();
            std::string newFileName = hash + extension;
            fs::path newPath = file.parent_path() / newFileName;
            
            std::cout << "  Hash (" << algorithm << "): " << hash << std::endl;
            std::cout << "  New name: " << newFileName << std::endl;
            
            // Check if the new filename is the same as the current filename
            if (file.filename() == newFileName) {
                std::cout << "  Status: No change needed (filename already matches hash)" << std::endl;
                noChangeCount++;
            } else if (!dryRun) {
                if (RenameFile(file, newPath)) {
                    std::cout << "  Status: Renamed successfully" << std::endl;
                    successCount++;
                } else {
                    std::cout << "  Status: Failed to rename" << std::endl;
                }
            } else {
                std::cout << "  Status: Preview only (will be renamed)" << std::endl;
            }
            
            std::cout << std::endl;
        }
        
        std::cout << "===========================================" << std::endl;
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        std::cout << "Summary:" << std::endl;
        std::cout << "Total files: " << files.size() << std::endl;
        std::cout << "Processed: " << processedCount << std::endl;
        std::cout << "Skipped (filter): " << skippedCount << std::endl;
        if (quickCheck) {
            std::cout << "Quick-skipped (likely correct): " << quickSkipCount << std::endl;
        }
        std::cout << "No change needed: " << noChangeCount << std::endl;
        if (!dryRun) {
            std::cout << "Successfully renamed: " << successCount << std::endl;
            std::cout << "Failed: " << (processedCount - successCount - noChangeCount) << std::endl;
        }
        std::cout << "Total execution time: " << totalDuration.count() << "ms" << std::endl;
        
        if (processedCount > 0) {
            double avgTimePerFile = static_cast<double>(totalDuration.count()) / processedCount;
            std::cout << "Average time per file: " << std::fixed << std::setprecision(2) << avgTimePerFile << "ms" << std::endl;
        }
        
        if (quickCheck && quickSkipCount > 0) {
            std::cout << "Time saved by quick check: ~" << (quickSkipCount * 10) << "ms (estimated)" << std::endl;
        }
    }
    // Ultra-high performance processing with ordered output
    static void ProcessDirectoryUltraFast(const std::string& directoryPath, const std::string& algorithm = "MD5", bool recursive = false, bool dryRun = true, const std::vector<std::string>& allowedExtensions = {}, bool quickCheck = true, int numThreads = 0) {
        auto startTime = std::chrono::high_resolution_clock::now();
        
        fs::path dir(directoryPath);
        
        if (!fs::exists(dir) || !fs::is_directory(dir)) {
            std::cerr << "Error: Invalid directory path: " << directoryPath << std::endl;
            return;
        }
        
        // Auto-detect threads: use all logical CPU threads by default for ultra-fast
        if (numThreads <= 0) {
            unsigned hc = std::thread::hardware_concurrency();
            if (hc == 0) hc = 4;
            numThreads = hc;
        }
        
    std::cout << "Ultra-fast processing mode enabled!" << std::endl;
        std::cout << "Scanning directory: " << directoryPath << std::endl;
        std::cout << "Algorithm: " << algorithm << std::endl;
        std::cout << "Recursive: " << (recursive ? "Yes" : "No") << std::endl;
        std::cout << "Mode: " << (dryRun ? "Preview" : "Execute") << std::endl;
        std::cout << "Quick check: " << (quickCheck ? "Enabled" : "Disabled") << std::endl;
        std::cout << "Optimized threads: " << numThreads << std::endl;
    std::cout << "I/O buffer: " << ioBufferKB << " KB, mmap chunk: " << mmapChunkMB << " MB" << std::endl;
        if (useGPU) {
            std::cout << "GPU min: " << (gpuMinFileSize/1024) << " KB, file-cap: " << gpuFileCapMB << " MB, batch-cap: " << gpuBatchBytesMB << " MB" << std::endl;
        }
        
        if (!allowedExtensions.empty()) {
            std::cout << "Extensions filter: ";
            for (size_t i = 0; i < allowedExtensions.size(); i++) {
                std::cout << "\"" << allowedExtensions[i] << "\"";
                if (i < allowedExtensions.size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << std::endl;
        } else {
            std::cout << "Extensions filter: All files" << std::endl;
        }
        
        std::cout << "===========================================" << std::endl;
        
        auto scanStartTime = std::chrono::high_resolution_clock::now();
        auto files = ScanDirectory(dir, recursive, allowedExtensions);
        auto scanEndTime = std::chrono::high_resolution_clock::now();
        
        auto scanDuration = std::chrono::duration_cast<std::chrono::milliseconds>(scanEndTime - scanStartTime);
        std::cout << "Found " << files.size() << " files in " << scanDuration.count() << "ms." << std::endl << std::endl;
        
        if (files.empty()) {
            std::cout << "No files found to process." << std::endl;
            return;
        }
        
        // Initialize thread pool
        ThreadPool pool(numThreads);
        
        // Thread-safe counters
        std::atomic<int> processedCount{0};
        std::atomic<int> successCount{0};
        std::atomic<int> skippedCount{0};
        std::atomic<int> noChangeCount{0};
        std::atomic<int> quickSkipCount{0};
        
        // Output buffer system for ordered output
        struct OutputBuffer {
            std::string content;
            size_t fileIndex;
            bool ready;
            std::chrono::steady_clock::time_point timestamp;
            
            OutputBuffer() : fileIndex(0), ready(false) {}
        };
        
        std::vector<OutputBuffer> outputBuffers(files.size());
        std::mutex outputMutex;
        std::atomic<size_t> nextOutputIndex{0};
        std::atomic<size_t> completedTasks{0};
        
        // Progress tracking with minimal output
        std::atomic<bool> showProgress{files.size() > 50}; // Only show progress for large datasets
        
        // Submit all tasks to thread pool
        std::vector<std::future<void>> futures;
        futures.reserve(files.size());
        
        for (size_t i = 0; i < files.size(); i++) {
            futures.push_back(pool.enqueue([&, i]() {
                const auto& file = files[i];
                std::stringstream buffer;
                auto taskStart = std::chrono::steady_clock::now();
                
                try {
                    std::string fileName = file.filename().string();
                    
                    // Build output in local buffer
                    buffer << "[" << (i + 1) << "/" << files.size() << "] Processing: \"" << fileName << "\"" << std::endl;
                    
                    // Check if file extension matches filter
                    if (!ShouldProcessFile(file, allowedExtensions)) {
                        std::string extension = file.extension().string();
                        buffer << "  Extension: \"" << extension << "\"" << std::endl;
                        buffer << "  Status: Skipped (extension not in filter)" << std::endl;
                        buffer << std::endl;
                        skippedCount++;
                        
                        // Store in output buffer
                        outputBuffers[i].content = buffer.str();
                        outputBuffers[i].fileIndex = i;
                        outputBuffers[i].timestamp = taskStart;
                        outputBuffers[i].ready = true;
                        return;
                    }
                    
                    // Quick check optimization
                    if (quickCheck && QuickHashCheck(file, algorithm)) {
                        buffer << "  Status: Likely already correctly named (quick check passed)" << std::endl;
                        buffer << std::endl;
                        quickSkipCount++;
                        
                        // Store in output buffer
                        outputBuffers[i].content = buffer.str();
                        outputBuffers[i].fileIndex = i;
                        outputBuffers[i].timestamp = taskStart;
                        outputBuffers[i].ready = true;
                        return;
                    }
                    
                    processedCount++;
                    
                    std::string hash = CalculateFileHashOptimized(file, algorithm);
                    if (hash.empty()) {
                        buffer << "  Error: Could not calculate hash" << std::endl;
                        buffer << std::endl;
                        
                        // Store in output buffer
                        outputBuffers[i].content = buffer.str();
                        outputBuffers[i].fileIndex = i;
                        outputBuffers[i].timestamp = taskStart;
                        outputBuffers[i].ready = true;
                        return;
                    }
                    
                    std::string extension = file.extension().string();
                    std::string newFileName = hash + extension;
                    fs::path newPath = file.parent_path() / newFileName;
                    
                    buffer << "  Hash (" << algorithm << "): " << hash << std::endl;
                    buffer << "  New name: " << newFileName << std::endl;
                    
                    // Check if the new filename is the same as the current filename
                    if (file.filename() == newFileName) {
                        buffer << "  Status: No change needed (filename already matches hash)" << std::endl;
                        buffer << std::endl;
                        noChangeCount++;
                    } else if (!dryRun) {
                        if (RenameFile(file, newPath)) {
                            buffer << "  Status: Renamed successfully" << std::endl;
                            buffer << std::endl;
                            successCount++;
                        } else {
                            buffer << "  Status: Failed to rename" << std::endl;
                            buffer << std::endl;
                        }
                    } else {
                        buffer << "  Status: Preview only (will be renamed)" << std::endl;
                        buffer << std::endl;
                    }
                    
                } catch (const std::exception& ex) {
                    buffer << "[" << (i + 1) << "/" << files.size() << "] Processing: <Error reading filename>" << std::endl;
                    buffer << "  Error: " << ex.what() << std::endl;
                    buffer << std::endl;
                    skippedCount++;
                }
                
                // Store completed output
                outputBuffers[i].content = buffer.str();
                outputBuffers[i].fileIndex = i;
                outputBuffers[i].timestamp = taskStart;
                outputBuffers[i].ready = true;
                
                completedTasks.fetch_add(1);
            }));
        }
        
        // Output thread to maintain proper order
        auto outputWorker = [&]() {
            while (nextOutputIndex.load() < files.size()) {
                size_t currentOutput = nextOutputIndex.load();
                if (currentOutput < files.size() && outputBuffers[currentOutput].ready) {
                    {
                        std::lock_guard<std::mutex> lock(outputMutex);
                        std::cout << outputBuffers[currentOutput].content << std::flush;
                    }
                    nextOutputIndex.fetch_add(1);
                } else {
                    // Wait briefly before checking again
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
            }
        };
        
        // Progress monitor thread (optional for large datasets)
        auto progressMonitor = [&]() {
            if (!showProgress.load()) return;
            
            auto lastUpdate = std::chrono::steady_clock::now();
            while (completedTasks.load() < files.size()) {
                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(now - lastUpdate).count() >= 2) {
                    size_t completed = completedTasks.load();
                    size_t active = pool.active_count();
                    size_t queued = pool.queue_size();
                    
                    std::lock_guard<std::mutex> lock(outputMutex);
                    std::cout << "\r[PROGRESS] " << completed << "/" << files.size() 
                              << " (" << (completed * 100 / files.size()) << "%) - "
                              << "Active: " << active << ", Queued: " << queued << "    " << std::flush;
                    
                    lastUpdate = now;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
            if (showProgress.load()) {
                std::lock_guard<std::mutex> lock(outputMutex);
                std::cout << "\r" << std::string(80, ' ') << "\r" << std::flush; // Clear progress line
            }
        };
        
        // Launch threads
        std::future<void> outputFuture = std::async(std::launch::async, outputWorker);
        std::future<void> progressFuture = std::async(std::launch::async, progressMonitor);
        
        // Wait for all tasks to complete
        for (auto& future : futures) {
            future.wait();
        }
        
        // Wait for output and progress threads
        outputFuture.wait();
        progressFuture.wait();
        
        std::cout << "===========================================" << std::endl;
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        std::cout << "Summary:" << std::endl;
        std::cout << "Total files: " << files.size() << std::endl;
        std::cout << "Processed: " << processedCount.load() << std::endl;
        std::cout << "Skipped (filter): " << skippedCount.load() << std::endl;
        if (quickCheck) {
            std::cout << "Quick-skipped (likely correct): " << quickSkipCount.load() << std::endl;
        }
        std::cout << "No change needed: " << noChangeCount.load() << std::endl;
        if (!dryRun) {
            std::cout << "Successfully renamed: " << successCount.load() << std::endl;
            std::cout << "Failed: " << (processedCount.load() - successCount.load() - noChangeCount.load()) << std::endl;
        }
        std::cout << "Total execution time: " << totalDuration.count() << "ms" << std::endl;
        
        if (processedCount.load() > 0) {
            double avgTimePerFile = static_cast<double>(totalDuration.count()) / processedCount.load();
            std::cout << "Average time per file: " << std::fixed << std::setprecision(2) << avgTimePerFile << "ms" << std::endl;
            
            double filesPerSecond = (processedCount.load() * 1000.0) / totalDuration.count();
            std::cout << "Ultra-fast throughput: " << std::fixed << std::setprecision(2) << filesPerSecond << " files/second" << std::endl;
        }
        
        if (quickCheck && quickSkipCount.load() > 0) {
            std::cout << "Time saved by quick check: ~" << (quickSkipCount.load() * 10) << "ms (estimated)" << std::endl;
        }
        
        std::cout << "Performance: " << numThreads << " optimized threads with ordered output" << std::endl;
    }

    // Batch processing with smart load balancing
    static void ProcessDirectoryBatch(const std::string& directoryPath, const std::string& algorithm = "MD5", bool recursive = false, bool dryRun = true, const std::vector<std::string>& allowedExtensions = {}, bool quickCheck = true, int numThreads = 0, size_t batchSize = 0) {
        auto startTime = std::chrono::high_resolution_clock::now();
        
        fs::path dir(directoryPath);
        
        if (!fs::exists(dir) || !fs::is_directory(dir)) {
            std::cerr << "Error: Invalid directory path: " << directoryPath << std::endl;
            return;
        }
        
        // Auto-detect optimal settings
        if (numThreads <= 0) {
            numThreads = std::thread::hardware_concurrency();
            if (numThreads <= 0) numThreads = 4;
        }
        
        if (batchSize == 0) {
            batchSize = std::max(1ULL, static_cast<size_t>(numThreads * 2)); // 2 batches per thread
        }
        
        std::cout << "Scanning directory: " << directoryPath << std::endl;
        std::cout << "Algorithm: " << algorithm << std::endl;
        std::cout << "Recursive: " << (recursive ? "Yes" : "No") << std::endl;
        std::cout << "Mode: " << (dryRun ? "Preview" : "Execute") << std::endl;
        std::cout << "Quick check: " << (quickCheck ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Threads: " << numThreads << std::endl;
    std::cout << "Batch size: " << batchSize << std::endl;
    std::cout << "I/O buffer: " << ioBufferKB << " KB, mmap chunk: " << mmapChunkMB << " MB" << std::endl;
        
        // Display device status
        if (!gpuInitialized) {
            InitializeGPU(-2); // Use auto-selection
        }
        std::cout << "Processing device: " << (useGPU ? ("GPU " + std::to_string(selectedDevice)) : "CPU") << std::endl;
        if (useGPU) {
            std::cout << "GPU min file size: " << (gpuMinFileSize / 1024) << " KB" << std::endl;
            std::cout << "GPU batch cap: " << gpuBatchBytesMB << " MB" << std::endl;
        }
        
        if (!allowedExtensions.empty()) {
            std::cout << "Extensions filter: ";
            for (size_t i = 0; i < allowedExtensions.size(); i++) {
                std::cout << "\"" << allowedExtensions[i] << "\"";
                if (i < allowedExtensions.size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << std::endl;
        } else {
            std::cout << "Extensions filter: All files" << std::endl;
        }
        
        std::cout << "===========================================" << std::endl;
        
        auto scanStartTime = std::chrono::high_resolution_clock::now();
        auto files = ScanDirectory(dir, recursive, allowedExtensions);
        auto scanEndTime = std::chrono::high_resolution_clock::now();
        
        auto scanDuration = std::chrono::duration_cast<std::chrono::milliseconds>(scanEndTime - scanStartTime);
        std::cout << "Found " << files.size() << " files in " << scanDuration.count() << "ms." << std::endl << std::endl;
        
        if (files.empty()) {
            std::cout << "No files found to process." << std::endl;
            return;
        }
        
        // Sort files by size for better load balancing (mix of small and large files)
        std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
            std::error_code ec1, ec2;
            auto sizeA = fs::file_size(a, ec1);
            auto sizeB = fs::file_size(b, ec2);
            if (ec1 || ec2) return false;
            return sizeA > sizeB; // Large files first for better scheduling
        });
        
        // Create batches
        std::vector<std::vector<fs::path>> batches;
        for (size_t i = 0; i < files.size(); i += batchSize) {
            std::vector<fs::path> batch;
            size_t end = std::min(i + batchSize, files.size());
            batch.assign(files.begin() + i, files.begin() + end);
            batches.push_back(std::move(batch));
        }
        
        std::cout << "Created " << batches.size() << " batches for processing." << std::endl << std::endl;
        
        // Thread-safe counters
        std::atomic<int> processedCount{0};
        std::atomic<int> successCount{0};
        std::atomic<int> skippedCount{0};
        std::atomic<int> noChangeCount{0};
        std::atomic<int> quickSkipCount{0};
        std::atomic<size_t> currentBatch{0};
        
        // Thread-safe output mutex
        std::mutex outputMutex;
        
        // Worker function for batch processing
        auto batchWorker = [&]() {
            while (true) {
                // Get next batch index atomically
                size_t batchIndex = currentBatch.fetch_add(1);
                if (batchIndex >= batches.size()) {
                    break;
                }
                
                const auto& batch = batches[batchIndex];
                
                struct GpuCandidate { size_t fileIndex; fs::path path; uintmax_t fileSize; std::string prelude; };
                std::vector<GpuCandidate> gpuCandidates;
                size_t gpuBytesAccum = 0;
                auto flushGpu = [&]() {
#ifdef USE_CUDA
                    if (!useGPU || gpuCandidates.empty() || !IsGPUSupportedAlgorithm(algorithm)) { gpuCandidates.clear(); gpuBytesAccum = 0; return; }
                    // Build data list within cap
                    std::vector<std::vector<unsigned char>> dataList;
                    dataList.reserve(gpuCandidates.size());
                    std::vector<bool> ok(gpuCandidates.size(), false);
                    for (size_t k = 0; k < gpuCandidates.size(); ++k) {
                        auto d = ReadFileEntireWin(gpuCandidates[k].path);
                        if (!d.empty() || gpuCandidates[k].fileSize == 0) { ok[k] = true; dataList.emplace_back(std::move(d)); }
                        else { dataList.emplace_back(); }
                    }
                    std::vector<std::string> hashes;
                    if (algorithm == "MD5") hashes = CudaHashCalculator::CalculateBatchMD5_GPU(dataList);
                    else if (algorithm == "SHA1") hashes = CudaHashCalculator::CalculateBatchSHA1_GPU(dataList);
                    else if (algorithm == "SHA256") hashes = CudaHashCalculator::CalculateBatchSHA256_GPU(dataList);
                    else if (algorithm == "CRC32") hashes = CudaHashCalculator::CalculateBatchCRC32_GPU(dataList);
                    // Emit results per candidate (fallback to CPU if needed)
                    for (size_t k = 0; k < gpuCandidates.size(); ++k) {
                        const auto& file = batch[gpuCandidates[k].fileIndex];
                        std::stringstream out;
                        out << gpuCandidates[k].prelude;
                        std::string hash;
                        if (k < hashes.size() && !hashes[k].empty() && ok[k]) {
                            hash = hashes[k];
                        } else {
                            // CPU fallback
                            hash = CalculateFileHashOptimized(file, algorithm);
                        }
                        if (hash.empty()) {
                            out << "  Error: Could not calculate hash" << std::endl << std::endl;
                            {
                                std::lock_guard<std::mutex> lock(outputMutex);
                                std::cout << out.str();
                            }
                            continue;
                        }
                        std::string extension = file.extension().string();
                        std::string newFileName = hash + extension;
                        fs::path newPath = file.parent_path() / newFileName;
                        out << "  Hash (" << algorithm << "): " << hash << std::endl;
                        out << "  New name: " << newFileName << std::endl;
                        if (file.filename() == newFileName) {
                            out << "  Status: No change needed (filename already matches hash)" << std::endl << std::endl;
                            noChangeCount++;
                        } else if (!dryRun) {
                            if (RenameFile(file, newPath)) {
                                out << "  Status: Renamed successfully" << std::endl << std::endl;
                                successCount++;
                            } else {
                                out << "  Status: Failed to rename" << std::endl << std::endl;
                            }
                        } else {
                            out << "  Status: Preview only (will be renamed)" << std::endl << std::endl;
                        }
                        processedCount++;
                        {
                            std::lock_guard<std::mutex> lock(outputMutex);
                            std::cout << out.str();
                        }
                    }
                    gpuCandidates.clear();
                    gpuBytesAccum = 0;
#else
                    (void)gpuCandidates; (void)gpuBytesAccum; // suppress unused warnings
#endif
                };
                
                // Process all files in this batch
                for (size_t fileIndex = 0; fileIndex < batch.size(); fileIndex++) {
                    const auto& file = batch[fileIndex];
                    
                    try {
                        std::string fileName = file.filename().string();
                        size_t globalIndex = batchIndex * batchSize + fileIndex;
                        std::stringstream prelude;
                        prelude << "[" << (globalIndex + 1) << "/" << files.size() << "] Batch " << (batchIndex + 1) << "/" << batches.size() << " Processing: \"" << fileName << "\"" << std::endl;
                        
                        // Check if file extension matches filter
                        if (!ShouldProcessFile(file, allowedExtensions)) {
                            std::string extension = file.extension().string();
                            prelude << "  Extension: \"" << extension << "\"" << std::endl;
                            prelude << "  Status: Skipped (extension not in filter)" << std::endl << std::endl;
                            skippedCount++;
                            {
                                std::lock_guard<std::mutex> lock(outputMutex);
                                std::cout << prelude.str();
                            }
                            continue;
                        }
                        
                        // Quick check optimization
                        if (quickCheck && QuickHashCheck(file, algorithm)) {
                            prelude << "  Status: Likely already correctly named (quick check passed)" << std::endl << std::endl;
                            quickSkipCount++;
                            {
                                std::lock_guard<std::mutex> lock(outputMutex);
                                std::cout << prelude.str();
                            }
                            continue;
                        }
                        
                        // Decide GPU vs CPU path in batch mode
                        bool tryGPU = false;
                        uintmax_t fsize = 0;
                        std::error_code fec;
                        fsize = fs::file_size(file, fec);
                        if (!fec) {
                            tryGPU = useGPU && IsGPUSupportedAlgorithm(algorithm) && fsize >= gpuMinFileSize && fsize <= (gpuFileCapMB * 1024ull * 1024ull);
                        }
                        if (tryGPU) {
                            // Queue for GPU mini-batch; flush if exceeds cap
                            size_t capBytes = gpuBatchBytesMB * 1024ull * 1024ull;
                            if (gpuBytesAccum + static_cast<size_t>(fsize) > capBytes && !gpuCandidates.empty()) {
                                flushGpu();
                            }
                            GpuCandidate gc{fileIndex, file, fsize, prelude.str()};
                            gpuBytesAccum += static_cast<size_t>(fsize);
                            gpuCandidates.emplace_back(std::move(gc));
                            continue;
                        }
                        
                        // CPU path immediate
                        {
                            std::string hash = CalculateFileHashOptimized(file, algorithm);
                            std::stringstream out;
                            out << prelude.str();
                            if (hash.empty()) {
                                out << "  Error: Could not calculate hash" << std::endl << std::endl;
                                {
                                    std::lock_guard<std::mutex> lock(outputMutex);
                                    std::cout << out.str();
                                }
                                continue;
                            }
                            std::string extension = file.extension().string();
                            std::string newFileName = hash + extension;
                            fs::path newPath = file.parent_path() / newFileName;
                            out << "  Hash (" << algorithm << "): " << hash << std::endl;
                            out << "  New name: " << newFileName << std::endl;
                            if (file.filename() == newFileName) {
                                out << "  Status: No change needed (filename already matches hash)" << std::endl << std::endl;
                                noChangeCount++;
                            } else if (!dryRun) {
                                if (RenameFile(file, newPath)) {
                                    out << "  Status: Renamed successfully" << std::endl << std::endl;
                                    successCount++;
                                } else {
                                    out << "  Status: Failed to rename" << std::endl << std::endl;
                                }
                            } else {
                                out << "  Status: Preview only (will be renamed)" << std::endl << std::endl;
                            }
                            processedCount++;
                            {
                                std::lock_guard<std::mutex> lock(outputMutex);
                                std::cout << out.str();
                            }
                        }
                        
                    } catch (const std::exception& ex) {
                        std::lock_guard<std::mutex> lock(outputMutex);
                        size_t globalIndex = batchIndex * batchSize + fileIndex;
                        std::cout << "[" << (globalIndex + 1) << "/" << files.size() << "] Processing: <Error reading filename>" << std::endl;
                        std::cout << "  Error: " << ex.what() << std::endl;
                        std::cout << std::endl;
                        skippedCount++;
                    }
                }
                // Flush remaining GPU candidates for this batch
                flushGpu();
            }
        };
        
        // Launch worker threads
        std::vector<std::future<void>> futures;
        for (int i = 0; i < numThreads; i++) {
            futures.push_back(std::async(std::launch::async, batchWorker));
        }
        
        // Wait for all threads to complete
        for (auto& future : futures) {
            future.wait();
        }
        
        std::cout << "===========================================" << std::endl;
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        std::cout << "Summary:" << std::endl;
        std::cout << "Total files: " << files.size() << std::endl;
        std::cout << "Processed: " << processedCount.load() << std::endl;
        std::cout << "Skipped (filter): " << skippedCount.load() << std::endl;
        if (quickCheck) {
            std::cout << "Quick-skipped (likely correct): " << quickSkipCount.load() << std::endl;
        }
        std::cout << "No change needed: " << noChangeCount.load() << std::endl;
        if (!dryRun) {
            std::cout << "Successfully renamed: " << successCount.load() << std::endl;
            std::cout << "Failed: " << (processedCount.load() - successCount.load() - noChangeCount.load()) << std::endl;
        }
        std::cout << "Total execution time: " << totalDuration.count() << "ms" << std::endl;
        
        if (processedCount.load() > 0) {
            double avgTimePerFile = static_cast<double>(totalDuration.count()) / processedCount.load();
            std::cout << "Average time per file: " << std::fixed << std::setprecision(2) << avgTimePerFile << "ms" << std::endl;
            
            double filesPerSecond = (processedCount.load() * 1000.0) / totalDuration.count();
            std::cout << "Throughput: " << std::fixed << std::setprecision(2) << filesPerSecond << " files/second" << std::endl;
        }
        
        if (quickCheck && quickSkipCount.load() > 0) {
            std::cout << "Time saved by quick check: ~" << (quickSkipCount.load() * 10) << "ms (estimated)" << std::endl;
        }
        
        std::cout << "Performance: " << numThreads << " threads, " << batches.size() << " batches utilized" << std::endl;
    }
};

// Static member definition
const char FileRenamerCLI::HEX_CHARS[16] = {'0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'};

// GPU-related static member definitions
bool FileRenamerCLI::gpuInitialized = false;
bool FileRenamerCLI::useGPU = false;
size_t FileRenamerCLI::gpuMinFileSize = 4096;
int FileRenamerCLI::selectedDevice = -1;
size_t FileRenamerCLI::ioBufferKB = 1024;   // 1MB default streaming buffer
size_t FileRenamerCLI::mmapChunkMB = 4;     // 4MB default mmap feed chunk
size_t FileRenamerCLI::gpuFileCapMB = 64;   // 64MB default cap for GPU in-memory hashing
size_t FileRenamerCLI::gpuBatchBytesMB = 256; // 256MB default GPU mini-batch cap
size_t FileRenamerCLI::gpuChunkMB = 8;      // 8MB default single-file CRC32 GPU chunk size

void PrintUsage() {
    std::cout << "File Batch Renamer Tool" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "Usage: file_renamer <directory> [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -a, --algorithm <hash>  Hash algorithm (MD5, SHA1, SHA256, SHA512, CRC32, BLAKE2B) [default: MD5]" << std::endl;
    std::cout << "  -r, --recursive         Scan subdirectories recursively" << std::endl;
    std::cout << "  -e, --execute           Execute renaming (default is preview mode)" << std::endl;
    std::cout << "  -x, --extensions <ext>  Only process files with specified extensions" << std::endl;
    std::cout << "                          (e.g., jpg,png,txt or .jpg,.png,.txt)" << std::endl;
    std::cout << "  -q, --quick             Enable quick check for already-named files [default: on]" << std::endl;
    std::cout << "  --no-quick              Disable quick check (force full hash calculation)" << std::endl;
    std::cout << "  -y, --yes               Auto-confirm without user interaction" << std::endl;
    std::cout << "  -t, --threads <n>       Number of processing threads [default: auto-detect]" << std::endl;
    std::cout << "  -b, --batch <n>         Batch size for processing [default: auto-calculate]" << std::endl;
    std::cout << "  -d, --device <id|cpu|auto> Device to use: -1 or 'cpu' for CPU, 0,1,2... for GPU, 'auto' for best available" << std::endl;
    std::cout << "                          Use 'list' to show available devices [default: auto]" << std::endl;
    std::cout << "  --gpu-min-kb <n>        Minimum file size (KB) to use GPU [default: auto(4)]" << std::endl;
    std::cout << "  --buffer-kb <n>         Streaming buffer size (KB) for hashing [default: auto]" << std::endl;
    std::cout << "  --mmap-chunk-mb <n>     Chunk size (MB) to feed from memory-mapped views [default: auto]" << std::endl;
    std::cout << "  --gpu-file-cap-mb <n>   Max file size (MB) to load entirely for GPU hashing [default: auto]" << std::endl;
    std::cout << "  --gpu-batch-bytes-mb <n> Max total bytes (MB) per GPU mini-batch in batch mode [default: auto]" << std::endl;
    std::cout << "  --gpu-chunk-mb <n>      Single-file CRC32 GPU chunk size (MB) [default: auto]" << std::endl;
    std::cout << "  --single-thread         Use single-threaded processing (original mode)" << std::endl;
    std::cout << "  --multi-thread          Use multi-threaded processing [default]" << std::endl;
    std::cout << "  --batch-mode            Use batch processing mode (best for large datasets)" << std::endl;
    std::cout << "  -h, --help              Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Processing Modes:" << std::endl;
    std::cout << "  Single-thread: Original sequential processing" << std::endl;
    std::cout << "  Multi-thread:  Parallel processing with work-stealing" << std::endl;
    std::cout << "  Batch-mode:    Optimized batch processing for large datasets" << std::endl;
    std::cout << "  Ultra-fast:    Maximum performance with thread pool and CPU affinity [default]" << std::endl;
    std::cout << std::endl;
    std::cout << "Additional Options:" << std::endl;
    std::cout << "  --ultra-fast            Use ultra-fast processing mode with thread pool" << std::endl;
    std::cout << "  --extreme               Extreme performance tuning (very high threads/buffers/batches; higher memory usage)" << std::endl;
    std::cout << std::endl;
#ifdef USE_CUDA
    std::cout << "Device Support:" << std::endl;
    std::cout << "  This version includes CUDA GPU acceleration support" << std::endl;
    std::cout << "  CPU (Device -1): Always available, multi-threaded processing" << std::endl;
    std::cout << "  GPU (Device 0+): CUDA-capable devices for large files (>4KB)" << std::endl;
    std::cout << "  Supported GPU algorithms: MD5, SHA1, SHA256, CRC32" << std::endl;
    std::cout << "  Auto-selection chooses best available device by default" << std::endl;
    std::cout << std::endl;
#else
    std::cout << "Device Support:" << std::endl;
    std::cout << "  CPU processing only (CUDA support not compiled)" << std::endl;
    std::cout << "  All devices selections will use CPU processing" << std::endl;
    std::cout << std::endl;
#endif
    std::cout << "Examples:" << std::endl;
    std::cout << "  file_renamer C:\\MyFiles                              # Auto-select best available device" << std::endl;
    std::cout << "  file_renamer C:\\MyFiles -x jpg,png -t 8              # Use 8 threads for jpg/png files" << std::endl;
    std::cout << "  file_renamer C:\\MyFiles -d list                      # List all available devices" << std::endl;
    std::cout << "  file_renamer C:\\MyFiles -d auto                      # Auto-select best device" << std::endl;
    std::cout << "  file_renamer C:\\MyFiles -d cpu                       # Force CPU processing" << std::endl;
    std::cout << "  file_renamer C:\\MyFiles -d -1                        # Force CPU processing (alternative)" << std::endl;
    std::cout << "  file_renamer C:\\MyFiles -d 0                         # Use GPU device 0" << std::endl;
    std::cout << "  file_renamer C:\\MyFiles -d 1 -a SHA256               # Use GPU device 1 with SHA256" << std::endl;
    std::cout << "  file_renamer C:\\MyFiles --batch-mode -b 50            # Batch mode with 50 files per batch" << std::endl;
    std::cout << "  file_renamer C:\\MyFiles --single-thread               # Original single-threaded mode" << std::endl;
    std::cout << "  file_renamer C:\\MyFiles -a SHA1 -r -e -t 16           # Execute SHA1 renaming with 16 threads, recursive" << std::endl;
    std::cout << "  file_renamer C:\\MyFiles -a SHA256 -e                  # Execute SHA256 renaming" << std::endl;
    std::cout << "  file_renamer C:\\MyFiles -a CRC32 -x txt               # Use CRC32 for txt files only" << std::endl;
    std::cout << "  file_renamer C:\\MyFiles --no-quick -t 4               # Force full hash check with 4 threads" << std::endl;
    std::cout << "  file_renamer C:\\MyFiles -e -y                        # Execute renaming with auto-confirm" << std::endl;
}

std::vector<std::string> ParseExtensions(const std::string& extensionsStr) {
    std::vector<std::string> extensions;
    std::stringstream ss(extensionsStr);
    std::string ext;
    
    while (std::getline(ss, ext, ',')) {
        // Trim whitespace
        ext.erase(0, ext.find_first_not_of(" \t"));
        ext.erase(ext.find_last_not_of(" \t") + 1);
        
        if (!ext.empty()) {
            extensions.push_back(ext);
        }
    }
    
    return extensions;
}

int main(int argc, char* argv[]) {
    // Set console code page to UTF-8 for proper Unicode display
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    
    if (argc < 2) {
        PrintUsage();
        return 1;
    }
    
    std::string directory;
    std::string algorithm = "MD5";
    std::string deviceSelection = "auto"; // auto, cpu, list, or device ID
    bool recursive = false;
    bool dryRun = true;
    bool showHelp = false;
    bool quickCheck = true; // Enable quick check by default
    bool autoConfirm = false; // Auto-confirm without user interaction
    int numThreads = 0; // Auto-detect by default
    size_t batchSize = 0; // Auto-calculate by default
    size_t argGpuMinKB = 4; // default auto -> 4KB threshold
    size_t argBufferKB = 0; // 0 means auto-tune
    size_t argMmapChunkMB = 0; // 0 means auto-tune
    size_t argGPUFileCapMB = 0; // 0 means auto-tune
    size_t argGPUBatchBytesMB = 0; // 0 means auto-tune
    size_t argGPUChunkMB = 0; // 0 means auto-tune for single-file CRC32 chunking
    bool userSetGpuMin = false, userSetBuffer = false, userSetMmap = false, userSetGpuFileCap = false, userSetGpuBatch = false, userSetGpuChunk = false;
    enum ProcessingMode { ULTRA_FAST, MULTI_THREAD, BATCH_MODE, SINGLE_THREAD };
    ProcessingMode mode = ULTRA_FAST; // Default to ultra-fast mode
    std::vector<std::string> allowedExtensions;
    bool extreme = false; // Extreme performance tuning
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            showHelp = true;
            break;
        } else if (arg == "-r" || arg == "--recursive") {
            recursive = true;
        } else if (arg == "-e" || arg == "--execute") {
            dryRun = false;
        } else if (arg == "-q" || arg == "--quick") {
            quickCheck = true;
        } else if (arg == "--no-quick") {
            quickCheck = false;
        } else if (arg == "-y" || arg == "--yes") {
            autoConfirm = true;
        } else if (arg == "--ultra-fast") {
            mode = ULTRA_FAST;
        } else if (arg == "--extreme") {
            extreme = true;
        } else if (arg == "--single-thread") {
            mode = SINGLE_THREAD;
        } else if (arg == "--multi-thread") {
            mode = MULTI_THREAD;
        } else if (arg == "--batch-mode") {
            mode = BATCH_MODE;
        } else if ((arg == "-a" || arg == "--algorithm") && i + 1 < argc) {
            algorithm = argv[++i];
            if (algorithm != "MD5" && algorithm != "SHA1" && algorithm != "SHA256" && 
                algorithm != "SHA512" && algorithm != "CRC32" && algorithm != "BLAKE2B") {
                std::cerr << "Error: Unsupported algorithm: " << algorithm << std::endl;
                std::cerr << "Supported algorithms: MD5, SHA1, SHA256, SHA512, CRC32, BLAKE2B" << std::endl;
                return 1;
            }
        } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            numThreads = std::atoi(argv[++i]);
            if (numThreads <= 0) {
                std::cerr << "Error: Invalid number of threads: " << numThreads << std::endl;
                return 1;
            }
        } else if ((arg == "-b" || arg == "--batch") && i + 1 < argc) {
            batchSize = std::atoi(argv[++i]);
            if (batchSize <= 0) {
                std::cerr << "Error: Invalid batch size: " << batchSize << std::endl;
                return 1;
            }
        } else if ((arg == "-d" || arg == "--device") && i + 1 < argc) {
            deviceSelection = argv[++i];
            // Validate device selection
            if (deviceSelection == "list") {
                // List devices and exit
                FileRenamerCLI::ListAvailableDevices();
                return 0;
            } else if (deviceSelection != "auto" && deviceSelection != "cpu") {
                // Check if it's a valid device ID (including -1 for CPU)
                char* endPtr;
                long deviceId = std::strtol(deviceSelection.c_str(), &endPtr, 10);
                if (*endPtr != '\0' || deviceId < -1) {
                    std::cerr << "Error: Invalid device specification: " << deviceSelection << std::endl;
                    std::cerr << "Use 'auto', 'cpu', 'list', -1 (CPU), or a GPU device ID (0, 1, 2, ...)" << std::endl;
                    return 1;
                }
            }
        } else if ((arg == "-x" || arg == "--extensions") && i + 1 < argc) {
            std::string extensionsStr = argv[++i];
            allowedExtensions = ParseExtensions(extensionsStr);
            if (allowedExtensions.empty()) {
                std::cerr << "Error: No valid extensions specified" << std::endl;
                return 1;
            }
        } else if (arg == "--gpu-min-kb" && i + 1 < argc) {
            long v = std::strtol(argv[++i], nullptr, 10);
            if (v <= 0) { std::cerr << "Error: Invalid --gpu-min-kb value" << std::endl; return 1; }
            argGpuMinKB = static_cast<size_t>(v);
            userSetGpuMin = true;
        } else if (arg == "--buffer-kb" && i + 1 < argc) {
            long v = std::strtol(argv[++i], nullptr, 10);
            if (v <= 0) { std::cerr << "Error: Invalid --buffer-kb value" << std::endl; return 1; }
            argBufferKB = static_cast<size_t>(v);
            userSetBuffer = true;
        } else if (arg == "--mmap-chunk-mb" && i + 1 < argc) {
            long v = std::strtol(argv[++i], nullptr, 10);
            if (v <= 0) { std::cerr << "Error: Invalid --mmap-chunk-mb value" << std::endl; return 1; }
            argMmapChunkMB = static_cast<size_t>(v);
            userSetMmap = true;
        } else if (arg == "--gpu-file-cap-mb" && i + 1 < argc) {
            long v = std::strtol(argv[++i], nullptr, 10);
            if (v <= 0) { std::cerr << "Error: Invalid --gpu-file-cap-mb value" << std::endl; return 1; }
            argGPUFileCapMB = static_cast<size_t>(v);
            userSetGpuFileCap = true;
        } else if (arg == "--gpu-batch-bytes-mb" && i + 1 < argc) {
            long v = std::strtol(argv[++i], nullptr, 10);
            if (v <= 0) { std::cerr << "Error: Invalid --gpu-batch-bytes-mb value" << std::endl; return 1; }
            argGPUBatchBytesMB = static_cast<size_t>(v);
            userSetGpuBatch = true;
        } else if (arg == "--gpu-chunk-mb" && i + 1 < argc) {
            long v = std::strtol(argv[++i], nullptr, 10);
            if (v <= 0) { std::cerr << "Error: Invalid --gpu-chunk-mb value" << std::endl; return 1; }
            argGPUChunkMB = static_cast<size_t>(v);
            userSetGpuChunk = true;
        } else if (directory.empty() && arg[0] != '-') {
            directory = arg;
        } else {
            std::cerr << "Error: Unknown option: " << arg << std::endl;
            PrintUsage();
            return 1;
        }
    }
    
    if (showHelp) {
        PrintUsage();
        return 0;
    }
    
    if (directory.empty()) {
        std::cerr << "Error: Directory path is required" << std::endl;
        PrintUsage();
        return 1;
    }
    
    // Initialize processing device with device selection
    std::cout << "Initializing File Renamer Tool..." << std::endl;
    int selectedDeviceId = -2; // Default to auto
    
    if (deviceSelection == "cpu") {
        selectedDeviceId = -1; // Force CPU
    } else if (deviceSelection == "auto") {
        selectedDeviceId = -2; // Auto-select best device
    } else {
        selectedDeviceId = std::atoi(deviceSelection.c_str()); // Specific device ID (including -1 for CPU)
    }
    
    // 1) Initialize GPU first to obtain device info (for auto tuning)
    bool gpuAvailable = FileRenamerCLI::InitializeGPU(selectedDeviceId);

    // 2) Auto-tune defaults if not specified by user
    //    - Tune by system RAM and GPU VRAM
    MEMORYSTATUSEX memInfo; memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    ULONGLONG totalRamMB = 0;
    if (GlobalMemoryStatusEx(&memInfo)) {
        totalRamMB = memInfo.ullTotalPhys / (1024ull * 1024ull);
    }

    // Derive GPU VRAM in MB when available
    ULONGLONG gpuMemMB = 0;
#ifdef USE_CUDA
    if (gpuAvailable) {
        const auto& gi = CudaHashCalculator::GetGPUInfo();
        int sel = FileRenamerCLI::GetSelectedDevice();
        if (sel >= 0 && sel < gi.deviceCount) {
            gpuMemMB = gi.devices[sel].totalGlobalMem / (1024ull * 1024ull);
        }
    }
#endif

    // Buffer size (favor higher throughput)
    if (!userSetBuffer) {
        if (extreme) argBufferKB = (totalRamMB >= 4096 ? 8192 : 4096); // Extreme: 4–8 MB
        else if (totalRamMB >= 8192) argBufferKB = 4096;               // 4 MB
        else argBufferKB = 2048;                                       // 2 MB
    }

    // mmap feed chunk (larger chunks for higher throughput)
    if (!userSetMmap) {
        if (extreme) argMmapChunkMB = (totalRamMB >= 4096 ? 32 : 16); // Extreme: 16–32 MB
        else if (totalRamMB >= 8192) argMmapChunkMB = 16;
        else argMmapChunkMB = 8;
    }

    // GPU file cap (allow larger in-memory GPU hashing)
    if (!userSetGpuFileCap) {
        if (gpuAvailable && gpuMemMB > 0) {
            // Extreme: ~1/8 VRAM (256–2048 MB), otherwise ~1/16 (128–1024 MB)
            if (extreme) {
                ULONGLONG cap = std::max<ULONGLONG>(256, std::min<ULONGLONG>(2048, gpuMemMB / 8));
                argGPUFileCapMB = static_cast<size_t>(cap);
            } else {
                ULONGLONG cap = std::max<ULONGLONG>(128, std::min<ULONGLONG>(1024, gpuMemMB / 16));
                argGPUFileCapMB = static_cast<size_t>(cap);
            }
        } else {
            argGPUFileCapMB = extreme ? 256 : 128; // CPU-only default
        }
    }

    // GPU batch cap (favor larger batches)
    if (!userSetGpuBatch) {
        if (gpuAvailable && gpuMemMB > 0) {
            // Extreme: ~1/2 VRAM [1024, 12288] MB, otherwise ~1/4 [512, 8192] MB
            if (extreme) {
                ULONGLONG cap = std::max<ULONGLONG>(1024, std::min<ULONGLONG>(12288, gpuMemMB / 2));
                argGPUBatchBytesMB = static_cast<size_t>(cap);
            } else {
                ULONGLONG cap = std::max<ULONGLONG>(512, std::min<ULONGLONG>(8192, gpuMemMB / 4));
                argGPUBatchBytesMB = static_cast<size_t>(cap);
            }
        } else {
            argGPUBatchBytesMB = extreme ? 1024 : 512;
        }
    }

    // GPU chunk size (favor larger chunk)
    if (!userSetGpuChunk) {
        argGPUChunkMB = extreme ? 32 : 16;
    }

    // Threads/Batches auto when not set by user
    if (numThreads <= 0) {
        unsigned hc = std::thread::hardware_concurrency();
        if (hc == 0) hc = 4;
        if (mode == ULTRA_FAST) {
            // Ultra Fast default: use all logical cores (max performance while preserving fairness)
            numThreads = hc;
        } else {
            // Other modes keep previous aggressive tuning
            unsigned mult = extreme ? 4u : 3u;
            unsigned cap = extreme ? 96u : 64u;
            numThreads = std::min<unsigned>(hc * mult, cap);
        }
    }
    if (batchSize == 0) {
        // Extreme: 6×threads；否则 4×threads
        batchSize = std::max<size_t>(1, static_cast<size_t>(numThreads) * (extreme ? 6 : 4));
    }

    // 3) Apply tuned params
    FileRenamerCLI::SetIOBufferKB(argBufferKB);
    FileRenamerCLI::SetMmapChunkMB(argMmapChunkMB);
    FileRenamerCLI::SetGPUFileCapMB(argGPUFileCapMB);
    FileRenamerCLI::SetGPUBatchBytesMB(argGPUBatchBytesMB);
    FileRenamerCLI::SetGPUChunkMB(argGPUChunkMB);

    // Apply GPU min threshold after initialization (allow user override)
    if (!userSetGpuMin) {
        // Default 4KB; could bump slightly if VRAM tiny, but keep consistent
        argGpuMinKB = 4;
    }
    FileRenamerCLI::SetGPUMinFileSizeBytes(argGpuMinKB * 1024ull);

    // Print final chosen parameters (concise)
    if (extreme) {
        std::cout << "Extreme mode enabled (aggressive tuning)." << std::endl;
    }
    std::cout << "Auto-tuned parameters:" << std::endl;
    std::cout << "  Threads: " << numThreads << ", Batch size: " << batchSize << std::endl;
    std::cout << "  I/O buffer: " << argBufferKB << " KB, mmap chunk: " << argMmapChunkMB << " MB" << std::endl;
    if (gpuAvailable) {
        std::cout << "  GPU file-cap: " << argGPUFileCapMB << " MB, batch-cap: " << argGPUBatchBytesMB << " MB, chunk: " << argGPUChunkMB << " MB" << std::endl;
    }
    
    if (!dryRun) {
        std::cout << "WARNING: This will permanently rename files!" << std::endl;
        
        if (autoConfirm) {
            std::cout << "Auto-confirm is enabled. Proceeding with renaming..." << std::endl;
        } else {
            std::cout << "Are you sure you want to continue? (y/N): ";
            
            std::string confirm;
            std::getline(std::cin, confirm);
            
            if (confirm != "y" && confirm != "Y" && confirm != "yes" && confirm != "Yes") {
                std::cout << "Operation cancelled." << std::endl;
                return 0;
            }
        }
    }
    
    // Choose processing mode
    switch (mode) {
        case SINGLE_THREAD:
            std::cout << "Using single-threaded processing mode." << std::endl;
            FileRenamerCLI::ProcessDirectory(directory, algorithm, recursive, dryRun, allowedExtensions, quickCheck);
            break;
        case MULTI_THREAD:
            std::cout << "Using multi-threaded processing mode." << std::endl;
            FileRenamerCLI::ProcessDirectoryMultiThreaded(directory, algorithm, recursive, dryRun, allowedExtensions, quickCheck, numThreads);
            break;
        case BATCH_MODE:
            std::cout << "Using batch processing mode." << std::endl;
            FileRenamerCLI::ProcessDirectoryBatch(directory, algorithm, recursive, dryRun, allowedExtensions, quickCheck, numThreads, batchSize);
            break;
        case ULTRA_FAST:
            std::cout << "Using ultra-fast processing mode." << std::endl;
            FileRenamerCLI::ProcessDirectoryUltraFast(directory, algorithm, recursive, dryRun, allowedExtensions, quickCheck, numThreads);
            break;
    }
    
    // Cleanup GPU resources
    FileRenamerCLI::CleanupGPU();
    
    return 0;
}
