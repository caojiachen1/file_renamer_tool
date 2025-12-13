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
#include <unordered_set>

#pragma comment(lib, "advapi32.lib")

// CUDA/GPU support removed - CPU processing only

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
    
    // CPU processing only
    
    // I/O performance tuning
    static size_t ioBufferKB;      // streaming buffer size in KB (for ReadFile)
    static size_t mmapChunkMB;     // chunk size for hashing memory-mapped views

    
    // CPU-optimized hex conversion for better performance
    // Pre-computed lookup table for byte-to-hex conversion (space-time tradeoff)
    static const char* HEX_LUT;
    static std::once_flag hex_lut_once;
    static char HEX_LUT_DATA[512]; // 256 * 2 bytes for lookup
    
    static void InitHexLUT() {
        for (int i = 0; i < 256; ++i) {
            HEX_LUT_DATA[i * 2] = HEX_CHARS[i >> 4];
            HEX_LUT_DATA[i * 2 + 1] = HEX_CHARS[i & 0x0F];
        }
    }
    
    static std::string BytesToHexOptimized(const BYTE* data, DWORD length) {
        // Thread-safe initialization of lookup table
        std::call_once(hex_lut_once, InitHexLUT);
        
        std::string result;
        result.resize(length * 2); // Pre-allocate exact size for better performance
        
        char* out = &result[0];
        
        // Use lookup table for O(1) per-byte conversion
        // Process 4 bytes at a time for better cache utilization
        DWORD i = 0;
        for (; i + 4 <= length; i += 4) {
            // Unrolled loop for better instruction-level parallelism
            const BYTE* ptr = data + i;
            memcpy(out, HEX_LUT_DATA + ptr[0] * 2, 2);
            memcpy(out + 2, HEX_LUT_DATA + ptr[1] * 2, 2);
            memcpy(out + 4, HEX_LUT_DATA + ptr[2] * 2, 2);
            memcpy(out + 6, HEX_LUT_DATA + ptr[3] * 2, 2);
            out += 8;
        }
        
        // Handle remaining bytes
        for (; i < length; ++i) {
            memcpy(out, HEX_LUT_DATA + data[i] * 2, 2);
            out += 2;
        }
        
        return result;
    }
    
    static std::string BytesToHex(const BYTE* data, DWORD length) {
        return BytesToHexOptimized(data, length);
    }

public:
    // Tuning setters
    static void SetGPUMinFileSizeBytes(size_t bytes) { /* GPU support removed */ }
    static void SetIOBufferKB(size_t kb) { ioBufferKB = (kb == 0 ? 64 : kb); }
    static void SetMmapChunkMB(size_t mb) { mmapChunkMB = (mb == 0 ? 1 : mb); }
    static void SetGPUFileCapMB(size_t mb) { /* GPU support removed */ }
    static void SetGPUBatchBytesMB(size_t mb) { /* GPU support removed */ }
    static void SetGPUChunkMB(size_t mb) { /* GPU support removed */ }


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
        // GPU support removed - use standard CPU CRC32 processing
        return CalculateFileHashOptimized(filePath, "CRC32");
    }
public:

    // CPU processing only
    

    
    // Device listing removed - CPU processing only
    
    // Cleanup function removed - CPU processing only
    
    // GPU processing check removed - CPU only
    
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
            // Note: Do not call CryptDestroyHash when hHash is 0 (creation failed)
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
        static std::once_flag crc_once_flag;
        
        // Thread-safe CRC table initialization using std::call_once
        std::call_once(crc_once_flag, [&]() {
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
        });
        
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
    
    // Generic hash calculation dispatcher (CPU-only in main branch)
    static std::string CalculateHash(const std::vector<BYTE>& data, const std::string& algorithm) {
        // CPU-only implementation
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
                // Use uint64_t to support files > 4GB properly
                const uint64_t CHUNK_SIZE = static_cast<uint64_t>(mmapChunkMB) * 1024ull * 1024ull;
                const uint64_t CHUNK_SIZE_SAFE = (CHUNK_SIZE == 0 ? (4ull * 1024ull * 1024ull) : CHUNK_SIZE);
                uint64_t totalSize = static_cast<uint64_t>(fileSize.QuadPart);
                
                for (uint64_t offset = 0; offset < totalSize; offset += CHUNK_SIZE_SAFE) {
                    // Calculate chunk size, ensuring it fits in DWORD for CryptHashData
                    uint64_t remaining = totalSize - offset;
                    DWORD chunkSize = static_cast<DWORD>(std::min(CHUNK_SIZE_SAFE, remaining));
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

    static FileHashResult ProcessSingleFile(const fs::path& filePath, const std::string& algorithm) {
        FileHashResult result;
        result.originalPath = filePath;
        result.success = false;
        result.needsRename = false;
        
        try {
            std::string hash = CalculateFileHashOptimized(filePath, algorithm);
            if (hash.empty()) {
                result.error = "Could not calculate hash";
                return result;
            }
            
            std::string extension = filePath.extension().u8string();
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
    
    // Enhanced error message structure for detailed rename failure reporting
    struct RenameErrorInfo {
        std::string errorCode;
        std::string errorMessage;
        std::string suggestion;
    };
    
    // Get thread-local error info cache (space-time tradeoff)
    static RenameErrorInfo& GetLastRenameError() {
        thread_local RenameErrorInfo lastRenameError;
        return lastRenameError;
    }
    
    static std::string GetWindowsErrorMessage(DWORD errorCode) {
        LPVOID lpMsgBuf = nullptr;
        FormatMessageW(
            FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL, errorCode, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            (LPWSTR)&lpMsgBuf, 0, NULL);
        
        std::wstring wMsg;
        if (lpMsgBuf) {
            wMsg = (LPCWSTR)lpMsgBuf;
            LocalFree(lpMsgBuf);
            // Remove trailing newlines
            while (!wMsg.empty() && (wMsg.back() == L'\n' || wMsg.back() == L'\r')) {
                wMsg.pop_back();
            }
        }
        
        // Convert to UTF-8
        if (wMsg.empty()) return "Unknown error";
        int size_needed = WideCharToMultiByte(CP_UTF8, 0, wMsg.c_str(), (int)wMsg.size(), NULL, 0, NULL, NULL);
        std::string result(size_needed, 0);
        WideCharToMultiByte(CP_UTF8, 0, wMsg.c_str(), (int)wMsg.size(), &result[0], size_needed, NULL, NULL);
        return result;
    }
    
    static bool RenameFile(const fs::path& oldPath, const fs::path& newPath) {
        // Clear previous error
        auto& lastRenameError = GetLastRenameError();
        lastRenameError = {};
        
        try {
            // Check if target already exists
            if (fs::exists(newPath)) {
                lastRenameError.errorCode = "FILE_EXISTS";
                lastRenameError.errorMessage = "Target file already exists: " + newPath.u8string();
                lastRenameError.suggestion = "Delete the existing file or use a different naming scheme";
                std::cerr << "  [ERROR] " << lastRenameError.errorCode << ": " << lastRenameError.errorMessage << std::endl;
                std::cerr << "  [HINT] " << lastRenameError.suggestion << std::endl;
                return false;
            }
            
            // Check source file accessibility
            std::error_code ec;
            auto status = fs::status(oldPath, ec);
            if (ec) {
                lastRenameError.errorCode = "SOURCE_ACCESS_ERROR";
                lastRenameError.errorMessage = "Cannot access source file: " + ec.message();
                lastRenameError.suggestion = "Check file permissions or if file is locked by another process";
                std::cerr << "  [ERROR] " << lastRenameError.errorCode << ": " << lastRenameError.errorMessage << std::endl;
                std::cerr << "  [HINT] " << lastRenameError.suggestion << std::endl;
                return false;
            }
            
            // Check if source is read-only (Windows specific)
            DWORD attrs = GetFileAttributesW(oldPath.wstring().c_str());
            if (attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_READONLY)) {
                lastRenameError.errorCode = "FILE_READONLY";
                lastRenameError.errorMessage = "Source file is read-only";
                lastRenameError.suggestion = "Remove read-only attribute: attrib -r \"" + oldPath.u8string() + "\"";
                std::cerr << "  [ERROR] " << lastRenameError.errorCode << ": " << lastRenameError.errorMessage << std::endl;
                std::cerr << "  [HINT] " << lastRenameError.suggestion << std::endl;
                return false;
            }
            
            // Attempt rename
            fs::rename(oldPath, newPath);
            return true;
            
        } catch (const fs::filesystem_error& ex) {
            // Parse the underlying error code for detailed diagnostics
            std::error_code ec = ex.code();
            DWORD winError = static_cast<DWORD>(ec.value());
            
            lastRenameError.errorCode = "RENAME_FAILED_" + std::to_string(winError);
            lastRenameError.errorMessage = ex.what();
            
            // Provide specific suggestions based on common error codes
            switch (winError) {
                case ERROR_ACCESS_DENIED: // 5
                    lastRenameError.suggestion = "Access denied. Check file permissions or if file is locked by another application (e.g., antivirus, editor)";
                    break;
                case ERROR_SHARING_VIOLATION: // 32
                    lastRenameError.suggestion = "File is locked by another process. Close any applications using this file";
                    break;
                case ERROR_FILE_NOT_FOUND: // 2
                    lastRenameError.suggestion = "Source file was moved or deleted during processing";
                    break;
                case ERROR_PATH_NOT_FOUND: // 3
                    lastRenameError.suggestion = "Target directory does not exist or path is invalid";
                    break;
                case ERROR_INVALID_NAME: // 123
                    lastRenameError.suggestion = "Invalid filename. Check for illegal characters in the hash output";
                    break;
                case ERROR_DISK_FULL: // 112
                    lastRenameError.suggestion = "Disk is full. Free up disk space and retry";
                    break;
                case ERROR_WRITE_PROTECT: // 19
                    lastRenameError.suggestion = "Disk is write-protected. Remove write protection";
                    break;
                default:
                    lastRenameError.suggestion = "Windows error " + std::to_string(winError) + ": " + GetWindowsErrorMessage(winError);
                    break;
            }
            
            std::cerr << "  [ERROR] " << lastRenameError.errorCode << ": " << lastRenameError.errorMessage << std::endl;
            std::cerr << "  [HINT] " << lastRenameError.suggestion << std::endl;
            return false;
        }
    }
    
    static bool ShouldProcessFile(const fs::path& filePath, const std::vector<std::string>& allowedExtensions) {
        if (allowedExtensions.empty()) {
            return true; // Process all files if no filter specified
        }
        
        std::string fileExt = filePath.extension().u8string();
        
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

    // Thread-local extension cache for O(1) lookup (space-time tradeoff)
    // This avoids repeated string transformations during high-throughput processing
    struct ExtensionCache {
        std::unordered_set<std::string> normalizedExtensions;
        bool initialized = false;
        
        void initialize(const std::vector<std::string>& allowedExtensions) {
            if (initialized) return;
            normalizedExtensions.reserve(allowedExtensions.size());
            for (const auto& ext : allowedExtensions) {
                std::string normalizedExt = ext;
                // Convert to lowercase
                std::transform(normalizedExt.begin(), normalizedExt.end(), normalizedExt.begin(), ::tolower);
                // Ensure dot prefix
                if (!normalizedExt.empty() && normalizedExt[0] != '.') {
                    normalizedExt = "." + normalizedExt;
                }
                normalizedExtensions.insert(normalizedExt);
            }
            initialized = true;
        }
        
        bool contains(const std::string& ext) const {
            std::string lowerExt = ext;
            std::transform(lowerExt.begin(), lowerExt.end(), lowerExt.begin(), ::tolower);
            return normalizedExtensions.find(lowerExt) != normalizedExtensions.end();
        }
    };

    // Optimized version for scanning - uses O(1) hash set lookup instead of O(n) linear search
    static bool ShouldProcessFileQuick(const fs::path& filePath, const std::vector<std::string>& allowedExtensions) {
        if (allowedExtensions.empty()) {
            return true;
        }
        
        // Thread-local cache to avoid contention (initialized on first use)
        thread_local ExtensionCache extensionCache;
        extensionCache.initialize(allowedExtensions);
        
        const std::string fileExt = filePath.extension().u8string();
        return extensionCache.contains(fileExt);
    }
    
    // Multi-threaded processing with buffered output to prevent output mixing
    static void ProcessDirectoryMultiThreaded(const std::string& directoryPath, const std::string& algorithm = "MD5", bool recursive = false, bool dryRun = true, const std::vector<std::string>& allowedExtensions = {}, int numThreads = 0) {
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
        std::cout << "Threads: " << numThreads << std::endl;
        
        std::cout << "Processing device: CPU" << std::endl;
        
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
                    std::string fileName = file.filename().u8string();
                    
                    // Build output in local buffer first
                    buffer << "[" << (fileIndex + 1) << "/" << files.size() << "] Processing: \"" << fileName << "\"" << std::endl;
                    
                    // Check if file extension matches filter
                    if (!ShouldProcessFile(file, allowedExtensions)) {
                        std::string extension = file.extension().u8string();
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
                    
                    std::string extension = file.extension().u8string();
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
            std::cout << "Throughput: " << std::fixed << std::setprecision(2)
                      << filesPerSecond << " files/second" << std::endl;
        }
        
        std::cout << "Performance: " << numThreads << " threads utilized with sequential output" << std::endl;
    }

    static void ProcessDirectory(const std::string& directoryPath, const std::string& algorithm = "MD5", bool recursive = false, bool dryRun = true, const std::vector<std::string>& allowedExtensions = {}) {
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
        
        // Display device status
        
        std::cout << "Processing device: CPU" << std::endl;
        
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
        
        for (const auto& file : files) {
            try {
                std::string fileName = file.filename().u8string();
                std::cout << "[" << (processedCount + skippedCount + 1) << "/" << files.size() << "] Processing: \"" << fileName << "\"" << std::endl;
                
                // Check if file extension matches filter
                if (!ShouldProcessFile(file, allowedExtensions)) {
                    std::string extension = file.extension().u8string();
                    std::cout << "  Extension: \"" << extension << "\"" << std::endl;
                    std::cout << "  Status: Skipped (extension not in filter)" << std::endl;
                    skippedCount++;
                    std::cout << std::endl;
                    continue;
                }
                
            } catch (const std::exception& ex) {
                std::cout << "[" << (processedCount + skippedCount + 1) << "/" << files.size() << "] Processing: <Error reading filename>" << std::endl;
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
            
            std::string extension = file.extension().u8string();
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
        std::cout << "No change needed: " << noChangeCount << std::endl;
        if (!dryRun) {
            std::cout << "Successfully renamed: " << successCount << std::endl;
            std::cout << "Failed: " << (processedCount - successCount - noChangeCount) << std::endl;
        }
        std::cout << "Total execution time: " << totalDuration.count() << "ms" << std::endl;
        
        if (processedCount > 0) {
            double avgTimePerFile = static_cast<double>(totalDuration.count()) / processedCount;
            std::cout << "Average time per file: " << std::fixed << std::setprecision(2) << avgTimePerFile << "ms" << std::endl;

            double filesPerSecond = (processedCount * 1000.0) / totalDuration.count();
            std::cout << "Throughput: " << std::fixed << std::setprecision(2)
                      << filesPerSecond << " files/second" << std::endl;
        }
        
    }

    // Batch processing with smart load balancing
    static void ProcessDirectoryBatch(const std::string& directoryPath, const std::string& algorithm = "MD5", bool recursive = false, bool dryRun = true, const std::vector<std::string>& allowedExtensions = {}, int numThreads = 0, size_t batchSize = 0) {
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
        std::cout << "Threads: " << numThreads << std::endl;
        std::cout << "Batch size: " << batchSize << std::endl;
        std::cout << "I/O buffer: " << ioBufferKB << " KB, mmap chunk: " << mmapChunkMB << " MB" << std::endl;
        std::cout << "Processing device: CPU" << std::endl;
        
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
                
                // Process all files in this batch
                for (size_t fileIndex = 0; fileIndex < batch.size(); fileIndex++) {
                    const auto& file = batch[fileIndex];
                    
                    try {
                        std::string fileName = file.filename().u8string();
                        size_t globalIndex = batchIndex * batchSize + fileIndex;
                        std::stringstream prelude;
                        prelude << "[" << (globalIndex + 1) << "/" << files.size() << "] Batch " << (batchIndex + 1) << "/" << batches.size() << " Processing: \"" << fileName << "\"" << std::endl;
                        
                        // Check if file extension matches filter
                        if (!ShouldProcessFile(file, allowedExtensions)) {
                            std::string extension = file.extension().u8string();
                            prelude << "  Extension: \"" << extension << "\"" << std::endl;
                            prelude << "  Status: Skipped (extension not in filter)" << std::endl << std::endl;
                            skippedCount++;
                            {
                                std::lock_guard<std::mutex> lock(outputMutex);
                                std::cout << prelude.str();
                            }
                            continue;
                        }
                        
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
                            std::string extension = file.extension().u8string();
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
        
        std::cout << "Performance: " << numThreads << " threads, " << batches.size() << " batches utilized" << std::endl;
    }
};

// Static member definition
const char FileRenamerCLI::HEX_CHARS[16] = {'0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'};

// Static member definitions for hex lookup table (space-time tradeoff optimization)
const char* FileRenamerCLI::HEX_LUT = nullptr;
std::once_flag FileRenamerCLI::hex_lut_once;
char FileRenamerCLI::HEX_LUT_DATA[512];

// Static member definitions - CPU only (GPU support removed)
size_t FileRenamerCLI::ioBufferKB = 1024;     // 1MB default streaming buffer
size_t FileRenamerCLI::mmapChunkMB = 4;       // 4MB default mmap feed chunk

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
    std::cout << "  -y, --yes               Auto-confirm without user interaction" << std::endl;
    std::cout << "  -t, --threads <n>       Number of processing threads [default: auto-detect]" << std::endl;
    std::cout << "  -b, --batch <n>         Batch size for processing [default: auto-calculate]" << std::endl;
    std::cout << "  --buffer-kb <n>         Streaming buffer size (KB) for hashing [default: auto]" << std::endl;
    std::cout << "  --mmap-chunk-mb <n>     Chunk size (MB) to feed from memory-mapped views [default: auto]" << std::endl;
    std::cout << "  --single-thread         Use single-threaded processing (original mode)" << std::endl;
    std::cout << "  --multi-thread          Use multi-threaded processing [default]" << std::endl;
    std::cout << "  --batch-mode            Use batch processing mode (best for large datasets)" << std::endl;
    std::cout << "  -h, --help              Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Processing Modes:" << std::endl;
    std::cout << "  Single-thread: Original sequential processing" << std::endl;
    std::cout << "  Multi-thread:  Parallel processing with work-stealing" << std::endl;
    std::cout << "  Batch-mode:    Optimized batch processing for large datasets" << std::endl;
    std::cout << std::endl;
    std::cout << "Device Support:" << std::endl;
    std::cout << "  CPU processing only" << std::endl;
    std::cout << "  Multi-threaded processing with thread pool" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  file_renamer C:\\MyFiles                              # Process files in C:\\MyFiles" << std::endl;
    std::cout << "  file_renamer C:\\MyFiles -x jpg,png -t 8              # Use 8 threads for jpg/png files" << std::endl;
    std::cout << "  file_renamer C:\\MyFiles --batch-mode -b 50            # Batch mode with 50 files per batch" << std::endl;
    std::cout << "  file_renamer C:\\MyFiles --single-thread               # Original single-threaded mode" << std::endl;
    std::cout << "  file_renamer C:\\MyFiles -a SHA1 -r -e -t 16           # Execute SHA1 renaming with 16 threads, recursive" << std::endl;
    std::cout << "  file_renamer C:\\MyFiles -a SHA256 -e                  # Execute SHA256 renaming" << std::endl;
    std::cout << "  file_renamer C:\\MyFiles -a CRC32 -x txt               # Use CRC32 for txt files only" << std::endl;
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
    bool recursive = false;
    bool dryRun = true;
    bool showHelp = false;
    bool autoConfirm = false; // Auto-confirm without user interaction
    int numThreads = 0; // Auto-detect by default
    size_t batchSize = 0; // Auto-calculate by default
    size_t argBufferKB = 0; // 0 means auto-tune
    size_t argMmapChunkMB = 0; // 0 means auto-tune
    bool userSetBuffer = false, userSetMmap = false;
    enum ProcessingMode { MULTI_THREAD, BATCH_MODE, SINGLE_THREAD };
    ProcessingMode mode = MULTI_THREAD; // Default to multi-threaded ultra-fast mode
    std::vector<std::string> allowedExtensions;
    
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
        } else if (arg == "-q" || arg == "--quick" || arg == "--no-quick") {
            // Backward-compatible no-op: quick check has been removed.
        } else if (arg == "-y" || arg == "--yes") {
            autoConfirm = true;
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
        } else if ((arg == "-x" || arg == "--extensions") && i + 1 < argc) {
            std::string extensionsStr = argv[++i];
            allowedExtensions = ParseExtensions(extensionsStr);
            if (allowedExtensions.empty()) {
                std::cerr << "Error: No valid extensions specified" << std::endl;
                return 1;
            }
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
    
    // Initialize processing
    std::cout << "Initializing File Renamer Tool" << std::endl;

    // Auto-tune defaults based on system RAM
    MEMORYSTATUSEX memInfo; memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    ULONGLONG totalRamMB = 0;
    if (GlobalMemoryStatusEx(&memInfo)) {
        totalRamMB = memInfo.ullTotalPhys / (1024ull * 1024ull);
    }

    // Buffer size (favor higher throughput)
    if (!userSetBuffer) {
        if (totalRamMB >= 8192) argBufferKB = 4096; // 4 MB
        else argBufferKB = 2048;                    // 2 MB
    }

    // mmap feed chunk (larger chunks for higher throughput)
    if (!userSetMmap) {
        if (totalRamMB >= 8192) argMmapChunkMB = 16;
        else argMmapChunkMB = 8;
    }


    // Threads/Batches auto when not set by user
    if (numThreads <= 0) {
        unsigned hc = std::thread::hardware_concurrency();
        if (hc == 0) hc = 4;
        // Default/"ultra-fast" behavior: use all logical cores
        numThreads = hc;
    }
    if (batchSize == 0) {
        batchSize = std::max<size_t>(1, static_cast<size_t>(numThreads) * 4);
    }

    // 3) Apply tuned params
    FileRenamerCLI::SetIOBufferKB(argBufferKB);
    FileRenamerCLI::SetMmapChunkMB(argMmapChunkMB);

    // Print final chosen parameters (concise)
    std::cout << "Auto-tuned parameters:" << std::endl;
    std::cout << "  Threads: " << numThreads << ", Batch size: " << batchSize << std::endl;
    std::cout << "  I/O buffer: " << argBufferKB << " KB, mmap chunk: " << argMmapChunkMB << " MB" << std::endl;
    
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
            FileRenamerCLI::ProcessDirectory(directory, algorithm, recursive, dryRun, allowedExtensions);
            break;
        case MULTI_THREAD:
            std::cout << "Using multi-threaded processing mode." << std::endl;
            FileRenamerCLI::ProcessDirectoryMultiThreaded(directory, algorithm, recursive, dryRun, allowedExtensions, numThreads);
            break;
        case BATCH_MODE:
            std::cout << "Using batch processing mode." << std::endl;
            FileRenamerCLI::ProcessDirectoryBatch(directory, algorithm, recursive, dryRun, allowedExtensions, numThreads, batchSize);
            break;
    }
    
    
    return 0;
}
