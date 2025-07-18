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

#pragma comment(lib, "advapi32.lib")

namespace fs = std::filesystem;

class FileRenamerCLI {
private:
    // Optimized hex conversion using lookup table with SIMD-friendly approach
    static const char HEX_CHARS[16];
    
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
    
    // Memory-mapped file hash calculation for very large files
    static std::string CalculateFileHashMemoryMapped(const fs::path& filePath, const std::string& algorithm = "MD5") {
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
            HCRYPTPROV hProv = 0;
            HCRYPTHASH hHash = 0;
            DWORD hashSize = (algorithm == "MD5") ? 16 : 20;
            ALG_ID algId = (algorithm == "MD5") ? CALG_MD5 : CALG_SHA1;
            std::string result;
            
            if (CryptAcquireContext(&hProv, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT)) {
                if (CryptCreateHash(hProv, algId, 0, 0, &hHash)) {
                    // Process in chunks to avoid overwhelming the hash function
                    const DWORD CHUNK_SIZE = 1024 * 1024; // 1MB chunks
                    DWORD totalSize = static_cast<DWORD>(fileSize.QuadPart);
                    
                    for (DWORD offset = 0; offset < totalSize; offset += CHUNK_SIZE) {
                        DWORD chunkSize = (CHUNK_SIZE < totalSize - offset) ? CHUNK_SIZE : (totalSize - offset);
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
            }
            
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
        std::ifstream file(filePath, std::ios::binary);
        if (!file.is_open()) {
            return "";
        }
        
        HCRYPTPROV hProv = 0;
        HCRYPTHASH hHash = 0;
        DWORD hashSize = (algorithm == "MD5") ? 16 : 20;
        ALG_ID algId = (algorithm == "MD5") ? CALG_MD5 : CALG_SHA1;
        
        if (!CryptAcquireContext(&hProv, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT)) {
            return "";
        }
        
        if (!CryptCreateHash(hProv, algId, 0, 0, &hHash)) {
            CryptReleaseContext(hProv, 0);
            return "";
        }
        
        // Process file in chunks to save memory
        const size_t BUFFER_SIZE = 64 * 1024; // 64KB chunks
        std::vector<BYTE> buffer(BUFFER_SIZE);
        
        while (file.good()) {
            file.read(reinterpret_cast<char*>(buffer.data()), BUFFER_SIZE);
            std::streamsize bytesRead = file.gcount();
            
            if (bytesRead > 0) {
                if (!CryptHashData(hHash, buffer.data(), static_cast<DWORD>(bytesRead), 0)) {
                    CryptDestroyHash(hHash);
                    CryptReleaseContext(hProv, 0);
                    return "";
                }
            }
        }
        
        std::vector<BYTE> rgbHash(hashSize);
        DWORD cbHash = hashSize;
        
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
        size_t expectedLength = (algorithm == "MD5") ? 32 : 40;
        
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
        
        if (algorithm == "MD5") {
            return CalculateMD5(buffer);
        } else if (algorithm == "SHA1") {
            return CalculateSHA1(buffer);
        }
        
        return "";
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
};

// Static member definition
const char FileRenamerCLI::HEX_CHARS[16] = {'0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'};

void PrintUsage() {
    std::cout << "File Batch Renamer Tool" << std::endl;
    std::cout << "=======================" << std::endl;
    std::cout << "Usage: file_renamer <directory> [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -a, --algorithm <hash>  Hash algorithm (MD5, SHA1) [default: MD5]" << std::endl;
    std::cout << "  -r, --recursive         Scan subdirectories recursively" << std::endl;
    std::cout << "  -e, --execute           Execute renaming (default is preview mode)" << std::endl;
    std::cout << "  -x, --extensions <ext>  Only process files with specified extensions" << std::endl;
    std::cout << "                          (e.g., jpg,png,txt or .jpg,.png,.txt)" << std::endl;
    std::cout << "  -q, --quick             Enable quick check for already-named files [default: on]" << std::endl;
    std::cout << "  --no-quick              Disable quick check (force full hash calculation)" << std::endl;
    std::cout << "  -h, --help              Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  file_renamer C:\\MyFiles                              # Preview all files with MD5" << std::endl;
    std::cout << "  file_renamer C:\\MyFiles -x jpg,png                   # Preview only jpg and png files" << std::endl;
    std::cout << "  file_renamer C:\\MyFiles -a SHA1 -r -x .txt,.log      # Preview txt/log files with SHA1, recursive" << std::endl;
    std::cout << "  file_renamer C:\\MyFiles -e -x jpg                    # Execute renaming for jpg files only" << std::endl;
    std::cout << "  file_renamer C:\\MyFiles --no-quick                   # Force full hash check on all files" << std::endl;
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
    bool quickCheck = true; // Enable quick check by default
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
        } else if (arg == "-q" || arg == "--quick") {
            quickCheck = true;
        } else if (arg == "--no-quick") {
            quickCheck = false;
        } else if ((arg == "-a" || arg == "--algorithm") && i + 1 < argc) {
            algorithm = argv[++i];
            if (algorithm != "MD5" && algorithm != "SHA1") {
                std::cerr << "Error: Unsupported algorithm: " << algorithm << std::endl;
                std::cerr << "Supported algorithms: MD5, SHA1" << std::endl;
                return 1;
            }
        } else if ((arg == "-x" || arg == "--extensions") && i + 1 < argc) {
            std::string extensionsStr = argv[++i];
            allowedExtensions = ParseExtensions(extensionsStr);
            if (allowedExtensions.empty()) {
                std::cerr << "Error: No valid extensions specified" << std::endl;
                return 1;
            }
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
    
    if (!dryRun) {
        std::cout << "WARNING: This will permanently rename files!" << std::endl;
        std::cout << "Are you sure you want to continue? (y/N): ";
        
        std::string confirm;
        std::getline(std::cin, confirm);
        
        if (confirm != "y" && confirm != "Y" && confirm != "yes" && confirm != "Yes") {
            std::cout << "Operation cancelled." << std::endl;
            return 0;
        }
    }
    
    FileRenamerCLI::ProcessDirectory(directory, algorithm, recursive, dryRun, allowedExtensions, quickCheck);
    
    return 0;
}
