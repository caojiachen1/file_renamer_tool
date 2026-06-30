use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "file_renamer", about = "File Batch Renamer Tool")]
pub struct Args {
    /// Directory path to process
    pub directory: PathBuf,

    /// Hash algorithm (MD5, SHA1, SHA256, SHA512, CRC32, BLAKE2B)
    #[arg(short, long, default_value = "MD5")]
    pub algorithm: String,

    /// Scan subdirectories recursively
    #[arg(short, long)]
    pub recursive: bool,

    /// Execute renaming (default is preview mode)
    #[arg(short, long)]
    pub execute: bool,

    /// Only process files with specified extensions (e.g., jpg,png,txt)
    #[arg(short = 'x', long, value_delimiter = ',')]
    pub extensions: Option<Vec<String>>,

    /// Auto-confirm without user interaction
    #[arg(short = 'y', long)]
    pub yes: bool,

    /// Number of processing threads [default: auto-detect]
    #[arg(short, long)]
    pub threads: Option<usize>,

    /// Batch size for processing [default: auto-calculate]
    #[arg(short, long)]
    pub batch: Option<usize>,

    /// Streaming buffer size (KB) for hashing
    #[arg(long)]
    pub buffer_kb: Option<usize>,

    /// Chunk size (MB) to feed from memory-mapped views
    #[arg(long)]
    pub mmap_chunk_mb: Option<usize>,

    /// Use single-threaded processing
    #[arg(long)]
    pub single_thread: bool,

    /// Use multi-threaded processing (default)
    #[arg(long)]
    pub multi_thread: bool,

    /// Use batch processing mode (best for large datasets)
    #[arg(long)]
    pub batch_mode: bool,
}

pub fn parse_extensions(exts: &[String]) -> Vec<String> {
    exts.iter()
        .map(|e| {
            let trimmed = e.trim();
            if trimmed.starts_with('.') {
                trimmed.to_lowercase()
            } else {
                format!(".{}", trimmed.to_lowercase())
            }
        })
        .filter(|e| !e.is_empty() && e != ".")
        .collect()
}
