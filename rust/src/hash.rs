use md5::Md5;
use sha1::Sha1;
use sha2::{Sha256, Sha512};
use blake2::{Blake2b, Digest};
use std::fs::File;
use std::io::{Read, BufReader};
use std::path::Path;

const CRC32_POLY: u32 = 0xEDB88320;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    Md5,
    Sha1,
    Sha256,
    Sha512,
    Crc32,
    Blake2b,
}

impl std::fmt::Display for Algorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Algorithm::Md5 => write!(f, "MD5"),
            Algorithm::Sha1 => write!(f, "SHA1"),
            Algorithm::Sha256 => write!(f, "SHA256"),
            Algorithm::Sha512 => write!(f, "SHA512"),
            Algorithm::Crc32 => write!(f, "CRC32"),
            Algorithm::Blake2b => write!(f, "BLAKE2B"),
        }
    }
}

impl Algorithm {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "MD5" => Some(Algorithm::Md5),
            "SHA1" => Some(Algorithm::Sha1),
            "SHA256" => Some(Algorithm::Sha256),
            "SHA512" => Some(Algorithm::Sha512),
            "CRC32" => Some(Algorithm::Crc32),
            "BLAKE2B" => Some(Algorithm::Blake2b),
            _ => None,
        }
    }
}

fn build_crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    for i in 0..256 {
        let mut crc = i as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ CRC32_POLY
            } else {
                crc >> 1
            };
        }
        table[i] = crc;
    }
    table
}

static CRC32_TABLE: std::sync::LazyLock<[u32; 256]> = std::sync::LazyLock::new(build_crc32_table);

pub fn hash_bytes(data: &[u8], algo: Algorithm) -> Option<String> {
    match algo {
        Algorithm::Md5 => {
            let mut hasher = Md5::new();
            hasher.update(data);
            Some(hex::encode(hasher.finalize()))
        }
        Algorithm::Sha1 => {
            let mut hasher = Sha1::new();
            hasher.update(data);
            Some(hex::encode(hasher.finalize()))
        }
        Algorithm::Sha256 => {
            let mut hasher = Sha256::new();
            hasher.update(data);
            Some(hex::encode(hasher.finalize()))
        }
        Algorithm::Sha512 => {
            let mut hasher = Sha512::new();
            hasher.update(data);
            Some(hex::encode(hasher.finalize()))
        }
        Algorithm::Crc32 => {
            let mut crc: u32 = 0xFFFFFFFF;
            for &byte in data {
                crc = CRC32_TABLE[((crc ^ byte as u32) & 0xFF) as usize] ^ (crc >> 8);
            }
            crc ^= 0xFFFFFFFF;
            Some(format!("{:08x}", crc))
        }
        Algorithm::Blake2b => {
            let mut hasher = Blake2b::<blake2::digest::consts::U32>::new();
            hasher.update(data);
            Some(hex::encode(hasher.finalize()))
        }
    }
}

pub fn hash_file(path: &Path, algo: Algorithm) -> Option<String> {
    match algo {
        Algorithm::Crc32 => hash_file_crc32_streaming(path),
        Algorithm::Blake2b => hash_file_blake2b_streaming(path),
        _ => hash_file_buffered(path, algo),
    }
}

fn hash_file_crc32_streaming(path: &Path) -> Option<String> {
    let file = File::open(path).ok()?;
    let mut reader = BufReader::with_capacity(256 * 1024, file);
    let mut crc: u32 = 0xFFFFFFFF;
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        for &byte in &buf[..n] {
            crc = CRC32_TABLE[((crc ^ byte as u32) & 0xFF) as usize] ^ (crc >> 8);
        }
    }
    crc ^= 0xFFFFFFFF;
    Some(format!("{:08x}", crc))
}

fn hash_file_blake2b_streaming(path: &Path) -> Option<String> {
    let file = File::open(path).ok()?;
    let mut reader = BufReader::with_capacity(256 * 1024, file);
    let mut hasher = Blake2b::<blake2::digest::consts::U32>::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Some(hex::encode(hasher.finalize()))
}

fn hash_file_buffered(path: &Path, algo: Algorithm) -> Option<String> {
    let file = File::open(path).ok()?;
    let mut reader = BufReader::with_capacity(256 * 1024, file);

    match algo {
        Algorithm::Md5 => {
            let mut hasher = Md5::new();
            let mut buf = [0u8; 64 * 1024];
            loop {
                let n = reader.read(&mut buf).ok()?;
                if n == 0 { break; }
                hasher.update(&buf[..n]);
            }
            Some(hex::encode(hasher.finalize()))
        }
        Algorithm::Sha1 => {
            let mut hasher = Sha1::new();
            let mut buf = [0u8; 64 * 1024];
            loop {
                let n = reader.read(&mut buf).ok()?;
                if n == 0 { break; }
                hasher.update(&buf[..n]);
            }
            Some(hex::encode(hasher.finalize()))
        }
        Algorithm::Sha256 => {
            let mut hasher = Sha256::new();
            let mut buf = [0u8; 64 * 1024];
            loop {
                let n = reader.read(&mut buf).ok()?;
                if n == 0 { break; }
                hasher.update(&buf[..n]);
            }
            Some(hex::encode(hasher.finalize()))
        }
        Algorithm::Sha512 => {
            let mut hasher = Sha512::new();
            let mut buf = [0u8; 64 * 1024];
            loop {
                let n = reader.read(&mut buf).ok()?;
                if n == 0 { break; }
                hasher.update(&buf[..n]);
            }
            Some(hex::encode(hasher.finalize()))
        }
        _ => None,
    }
}

pub fn hash_file_optimized(path: &Path, algo: Algorithm) -> Option<String> {
    match algo {
        Algorithm::Crc32 | Algorithm::Blake2b => hash_file(path, algo),
        _ => {
            let meta = std::fs::metadata(path).ok()?;
            let file_size = meta.len();

            if file_size > 100 * 1024 * 1024 {
                if let Some(result) = hash_file_memory_mapped(path, algo) {
                    return Some(result);
                }
            }

            hash_file(path, algo)
        }
    }
}

fn hash_file_memory_mapped(path: &Path, algo: Algorithm) -> Option<String> {
    let file = File::open(path).ok()?;
    let mmap = unsafe { memmap2::Mmap::map(&file).ok()? };
    let data = mmap.as_ref();
    hash_bytes(data, algo)
}

pub fn process_single_file(path: &Path, algo: Algorithm) -> FileHashResult {
    let mut result = FileHashResult {
        original_path: path.to_path_buf(),
        new_path: path.to_path_buf(),
        hash: String::new(),
        success: false,
        error: String::new(),
        needs_rename: false,
    };

    match hash_file_optimized(path, algo) {
        Some(hash) => {
            let extension = path.extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");
            let new_filename = format!("{}.{}", hash, extension);
            let new_path = path.parent().unwrap_or(path).join(&new_filename);

            result.hash = hash;
            result.new_path = new_path;
            result.success = true;
            result.needs_rename = path.file_name() != result.new_path.file_name();
        }
        None => {
            result.error = "Could not calculate hash".to_string();
        }
    }

    result
}

pub struct FileHashResult {
    pub original_path: std::path::PathBuf,
    pub new_path: std::path::PathBuf,
    pub hash: String,
    pub success: bool,
    pub error: String,
    pub needs_rename: bool,
}
