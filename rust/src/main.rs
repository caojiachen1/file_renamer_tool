mod cli;

use clap::Parser;
use std::collections::HashSet;
use std::io::{self, Write};
use std::path::PathBuf;

use file_renamer_lib::{Algorithm, ProcessOptions, ProcessingMode, ProgressReporter, process_directory};
use std::sync::Arc;

struct StdoutReporter;

impl ProgressReporter for StdoutReporter {
    fn on_output(&self, text: &str) {
        print!("{}", text);
        let _ = io::stdout().flush();
    }
}

fn main() {
    let args = cli::Args::parse();

    if args.directory.as_os_str().is_empty() {
        eprintln!("Usage: file_renamer <directory> [options]");
        eprintln!("Run with --help for full options.");
        std::process::exit(1);
    }

    let directory = args.directory;

    if !directory.exists() || !directory.is_dir() {
        eprintln!("Error: Invalid directory path: {}", directory.display());
        std::process::exit(1);
    }

    let algorithm = match Algorithm::from_str(&args.algorithm) {
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
            println!("Using single-threaded processing mode.\n");
        }
        ProcessingMode::MultiThread => {
            println!("Using multi-threaded processing mode.\n");
        }
        ProcessingMode::BatchMode => {
            println!("Using batch processing mode.\n");
        }
    }

    let options = ProcessOptions {
        directory,
        algorithm,
        recursive,
        dry_run,
        allowed_extensions,
        num_threads,
        batch_size,
        mode,
    };

    let reporter = Arc::new(StdoutReporter);
    process_directory(options, reporter);
}
