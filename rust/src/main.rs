mod cli;
mod file_ops;
mod hash;
mod scanner;
mod workers;

use clap::Parser;
use std::collections::HashSet;
use std::io::{self, Write};

fn main() {
    let args = cli::Args::parse();

    if args.directory.as_os_str().is_empty() {
        workers::print_usage();
        std::process::exit(1);
    }

    let directory = args.directory;

    if !directory.exists() || !directory.is_dir() {
        eprintln!("Error: Invalid directory path: {}", directory.display());
        std::process::exit(1);
    }

    let algorithm = match hash::Algorithm::from_str(&args.algorithm) {
        Some(a) => a,
        None => {
            eprintln!("Error: Unsupported algorithm: {}", args.algorithm);
            eprintln!("Supported algorithms: MD5, SHA1, SHA256, SHA512, CRC32, BLAKE2B");
            std::process::exit(1);
        }
    };

    let dry_run = !args.execute;
    let auto_confirm = args.yes;
    let recursive = args.recursive;

    enum ProcessingMode {
        MultiThread,
        BatchMode,
        SingleThread,
    }

    let mode = if args.batch_mode {
        ProcessingMode::BatchMode
    } else if args.single_thread {
        ProcessingMode::SingleThread
    } else {
        ProcessingMode::MultiThread
    };

    let allowed_extensions: HashSet<String> = match &args.extensions {
        Some(exts) => cli::parse_extensions(exts).into_iter().collect(),
        None => HashSet::new(),
    };

    let total_ram_mb = sys_info();
    let _buffer_kb = args.buffer_kb.unwrap_or_else(|| {
        if total_ram_mb >= 8192 { 4096 } else { 2048 }
    });
    let _mmap_chunk_mb = args.mmap_chunk_mb.unwrap_or_else(|| {
        if total_ram_mb >= 8192 { 16 } else { 8 }
    });

    let num_threads = args.threads.unwrap_or_else(|| {
        let n = num_cpus::get();
        if n == 0 { 4 } else { n }
    });

    let batch_size = args.batch.unwrap_or_else(|| {
        std::cmp::max(1, num_threads * 4)
    });

    println!("Initializing File Renamer Tool");
    println!("Auto-tuned parameters:");
    println!("  Threads: {}, Batch size: {}", num_threads, batch_size);

    if !dry_run {
        println!("WARNING: This will permanently rename files!");

        if auto_confirm {
            println!("Auto-confirm is enabled. Proceeding with renaming...");
        } else {
            print!("Are you sure you want to continue? (y/N): ");
            io::stdout().flush().unwrap();

            let mut confirm = String::new();
            io::stdin().read_line(&mut confirm).unwrap();
            let confirm = confirm.trim();

            if confirm != "y" && confirm != "Y" && confirm != "yes" && confirm != "Yes" {
                println!("Operation cancelled.");
                return;
            }
        }
    }

    match mode {
        ProcessingMode::SingleThread => {
            println!("Using single-threaded processing mode.");
            workers::process_single_threaded(
                &directory,
                algorithm,
                recursive,
                dry_run,
                allowed_extensions,
            );
        }
        ProcessingMode::MultiThread => {
            println!("Using multi-threaded processing mode.");
            workers::process_multi_threaded(
                &directory,
                algorithm,
                recursive,
                dry_run,
                allowed_extensions,
                num_threads,
            );
        }
        ProcessingMode::BatchMode => {
            println!("Using batch processing mode.");
            workers::process_batch_mode(
                &directory,
                algorithm,
                recursive,
                dry_run,
                allowed_extensions,
                num_threads,
                batch_size,
            );
        }
    }
}

fn sys_info() -> u64 {
    #[cfg(target_os = "windows")]
    {
        use std::mem;
        #[repr(C)]
        struct MEMORYSTATUSEX {
            dw_length: u32,
            dw_memory_load: u32,
            ull_total_phys: u64,
            ull_avail_phys: u64,
            ull_total_page_file: u64,
            ull_avail_page_file: u64,
            ull_total_virtual: u64,
            ull_avail_virtual: u64,
            ull_avail_extended_virtual: u64,
        }

        extern "system" {
            fn GlobalMemoryStatusEx(lpBuffer: *mut MEMORYSTATUSEX) -> i32;
        }

        let mut mem_info: MEMORYSTATUSEX = unsafe { mem::zeroed() };
        mem_info.dw_length = mem::size_of::<MEMORYSTATUSEX>() as u32;
        let success = unsafe { GlobalMemoryStatusEx(&mut mem_info) };
        if success != 0 {
            mem_info.ull_total_phys / (1024 * 1024)
        } else {
            4096
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        4096
    }
}
