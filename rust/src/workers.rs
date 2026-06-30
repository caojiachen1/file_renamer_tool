use std::collections::{HashMap, HashSet};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Condvar};
use std::thread;
use std::time::Instant;

use crate::file_ops;
use crate::hash::{self, Algorithm};
use crate::scanner::{self, ScannedFile};

#[derive(Clone)]
pub struct FailedRenameInfo {
    pub file_path: PathBuf,
    pub file_name: String,
    pub error_code: String,
    pub error_message: String,
    pub suggestion: String,
    pub file_index: usize,
}

pub struct FailedRenameCollector {
    failures: Mutex<Vec<FailedRenameInfo>>,
}

impl FailedRenameCollector {
    pub fn new() -> Self {
        Self { failures: Mutex::new(Vec::new()) }
    }

    pub fn add(&self, info: FailedRenameInfo) {
        self.failures.lock().unwrap().push(info);
    }

    pub fn get_all(&self) -> Vec<FailedRenameInfo> {
        self.failures.lock().unwrap().clone()
    }

    pub fn count(&self) -> usize {
        self.failures.lock().unwrap().len()
    }
}

pub fn print_usage() {
    println!("File Batch Renamer Tool");
    println!("=========================================");
    println!("Usage: file_renamer <directory> [options]");
    println!();
    println!("Options:");
    println!("  -a, --algorithm <hash>  Hash algorithm (MD5, SHA1, SHA256, SHA512, CRC32, BLAKE2B) [default: MD5]");
    println!("  -r, --recursive         Scan subdirectories recursively");
    println!("  -e, --execute           Execute renaming (default is preview mode)");
    println!("  -x, --extensions <ext>  Only process files with specified extensions");
    println!("  -y, --yes               Auto-confirm without user interaction");
    println!("  -t, --threads <n>       Number of processing threads [default: auto-detect]");
    println!("  -b, --batch <n>         Batch size for processing [default: auto-calculate]");
    println!("  --single-thread         Use single-threaded processing");
    println!("  --multi-thread          Use multi-threaded processing [default]");
    println!("  --batch-mode            Use batch processing mode");
    println!("  -h, --help              Show this help message");
}

struct WorkerContext {
    files: Vec<ScannedFile>,
    algorithm: Algorithm,
    dry_run: bool,
    next_index: AtomicUsize,
    processed_count: AtomicUsize,
    success_count: AtomicUsize,
    skipped_count: AtomicUsize,
    no_change_count: AtomicUsize,
    failed_renames: FailedRenameCollector,
}

#[derive(Clone)]
struct OutputEntry {
    content: String,
    ready: bool,
}

fn store_output(
    buffers: &Arc<Mutex<HashMap<usize, OutputEntry>>>,
    cv: &Arc<Condvar>,
    cv_mutex: &Arc<Mutex<()>>,
    idx: usize,
    content: String,
) {
    {
        let mut map = buffers.lock().unwrap();
        map.insert(idx, OutputEntry { content, ready: true });
    }
    let _lock = cv_mutex.lock().unwrap();
    cv.notify_all();
}

pub fn process_single_threaded(
    directory: &Path,
    algorithm: Algorithm,
    recursive: bool,
    dry_run: bool,
    allowed_extensions: HashSet<String>,
) {
    let start = Instant::now();

    if !directory.exists() || !directory.is_dir() {
        eprintln!("Error: Invalid directory path: {}", directory.display());
        return;
    }

    println!("Scanning directory: {}", directory.display());
    println!("Algorithm: {}", algorithm);
    println!("Recursive: {}", if recursive { "Yes" } else { "No" });
    println!("Mode: {}", if dry_run { "Preview" } else { "Execute" });
    println!("Processing device: CPU");

    if !allowed_extensions.is_empty() {
        print!("Extensions filter: ");
        for (i, ext) in allowed_extensions.iter().enumerate() {
            print!("\"{}\"", ext);
            if i < allowed_extensions.len() - 1 {
                print!(", ");
            }
        }
        println!();
    } else {
        println!("Extensions filter: All files");
    }

    println!("===========================================");

    let scan_start = Instant::now();
    let files = scanner::scan_directory(directory, recursive, &allowed_extensions);
    let scan_duration = scan_start.elapsed();
    println!(
        "Found {} files in {}ms.",
        files.len(),
        scan_duration.as_millis()
    );
    println!();

    if files.is_empty() {
        println!("No files found to process.");
        return;
    }

    let mut processed_count = 0usize;
    let mut success_count = 0usize;
    let skipped_count = 0usize;
    let mut no_change_count = 0usize;
    let failed_renames = FailedRenameCollector::new();

    for (idx, scanned) in files.iter().enumerate() {
        let file = &scanned.path;
        let file_name = file.file_name().and_then(|n| n.to_str()).unwrap_or("<Error reading filename>");
        println!(
            "[{}/{}] Processing: \"{}\"",
            idx + 1,
            files.len(),
            file_name
        );

        processed_count += 1;

        let hash_result = hash::process_single_file(file, algorithm);
        if !hash_result.success {
            println!("  Error: {}", hash_result.error);
            println!();
            continue;
        }

        let extension = file.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        let new_filename = format!("{}.{}", hash_result.hash, extension);
        let new_path = file.parent().unwrap_or(file).join(&new_filename);

        println!("  Hash ({}): {}", algorithm, hash_result.hash);
        println!("  New name: {}", new_filename);

        if file.file_name() == new_path.file_name() {
            println!("  Status: No change needed (filename already matches hash)");
            no_change_count += 1;
        } else if !dry_run {
            match file_ops::rename_file(file, &new_path) {
                Ok(()) => {
                    println!("  Status: Renamed successfully");
                    success_count += 1;
                }
                Err(e) => {
                    println!("  Status: Failed to rename");
                    failed_renames.add(FailedRenameInfo {
                        file_path: file.clone(),
                        file_name: file_name.to_string(),
                        error_code: e.error_code,
                        error_message: e.error_message,
                        suggestion: e.suggestion,
                        file_index: idx + 1,
                    });
                }
            }
        } else {
            println!("  Status: Preview only (will be renamed)");
        }
        println!();
        let _ = io::stdout().flush();
    }

    println!("===========================================");

    let total_duration = start.elapsed();

    println!("Summary:");
    println!("Total files: {}", files.len());
    println!("Processed: {}", processed_count);
    println!("Skipped (filter): {}", skipped_count);
    println!("No change needed: {}", no_change_count);
    if !dry_run {
        println!("Successfully renamed: {}", success_count);
        let failed_count = processed_count - success_count - no_change_count;
        println!("Failed: {}", failed_count);

        if failed_count > 0 && failed_renames.count() > 0 {
            println!();
            println!("============================================");
            println!("FAILED RENAMES ({} files):", failed_renames.count());
            println!("============================================");

            for fail in failed_renames.get_all() {
                println!();
                println!("[{}] File: {}", fail.file_index, fail.file_name);
                println!("    Path: {}", fail.file_path.display());
                println!("    Error: {}", fail.error_code);
                println!("    Message: {}", fail.error_message);
                if !fail.suggestion.is_empty() {
                    println!("    Hint: {}", fail.suggestion);
                }
            }
            println!();
            println!("============================================");
        }
    }

    println!(
        "Total execution time: {}ms",
        total_duration.as_millis()
    );

    if processed_count > 0 {
        let avg = total_duration.as_millis() as f64 / processed_count as f64;
        println!("Average time per file: {:.2}ms", avg);
        let throughput = processed_count as f64 * 1000.0 / total_duration.as_millis() as f64;
        println!("Throughput: {:.2} files/second", throughput);
    }
}

pub fn process_multi_threaded(
    directory: &Path,
    algorithm: Algorithm,
    recursive: bool,
    dry_run: bool,
    allowed_extensions: HashSet<String>,
    num_threads: usize,
) {
    let start = Instant::now();

    if !directory.exists() || !directory.is_dir() {
        eprintln!("Error: Invalid directory path: {}", directory.display());
        return;
    }

    println!("Scanning directory: {}", directory.display());
    println!("Algorithm: {}", algorithm);
    println!("Recursive: {}", if recursive { "Yes" } else { "No" });
    println!("Mode: {}", if dry_run { "Preview" } else { "Execute" });
    println!("Threads: {}", num_threads);
    println!("Processing device: CPU");

    if !allowed_extensions.is_empty() {
        print!("Extensions filter: ");
        for (i, ext) in allowed_extensions.iter().enumerate() {
            print!("\"{}\"", ext);
            if i < allowed_extensions.len() - 1 {
                print!(", ");
            }
        }
        println!();
    } else {
        println!("Extensions filter: All files");
    }

    println!("===========================================");

    let scan_start = Instant::now();
    let files = scanner::scan_directory(directory, recursive, &allowed_extensions);
    let scan_duration = scan_start.elapsed();
    println!(
        "Found {} files in {}ms.",
        files.len(),
        scan_duration.as_millis()
    );
    println!();

    if files.is_empty() {
        println!("No files found to process.");
        return;
    }

    let total_files = files.len();

    let ctx = Arc::new(WorkerContext {
        files,
        algorithm,
        dry_run,
        next_index: AtomicUsize::new(0),
        processed_count: AtomicUsize::new(0),
        success_count: AtomicUsize::new(0),
        skipped_count: AtomicUsize::new(0),
        no_change_count: AtomicUsize::new(0),
        failed_renames: FailedRenameCollector::new(),
    });

    let output_buffers: Arc<Mutex<HashMap<usize, OutputEntry>>> =
        Arc::new(Mutex::new(HashMap::new()));
    let next_output: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));
    let output_cv_mutex: Arc<Mutex<()>> = Arc::new(Mutex::new(()));
    let output_cv: Arc<Condvar> = Arc::new(Condvar::new());
    let workers_done: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));

    let mut handles = Vec::with_capacity(num_threads);
    for _ in 0..num_threads {
        let ctx_clone = Arc::clone(&ctx);
        let bufs_clone = Arc::clone(&output_buffers);
        let cv_clone = Arc::clone(&output_cv);
        let cv_mutex_clone = Arc::clone(&output_cv_mutex);
        let done_clone = Arc::clone(&workers_done);
        handles.push(thread::spawn(move || {
            worker_fn(ctx_clone, bufs_clone, cv_clone, cv_mutex_clone, done_clone);
        }));
    }

    let out_bufs = Arc::clone(&output_buffers);
    let out_next = Arc::clone(&next_output);
    let out_cv = Arc::clone(&output_cv);
    let out_cv_mutex = Arc::clone(&output_cv_mutex);
    let out_done = Arc::clone(&workers_done);
    let output_handle = thread::spawn(move || {
        output_worker_fn(out_bufs, out_next, out_cv, out_cv_mutex, out_done, total_files);
    });

    for h in handles {
        let _ = h.join();
    }

    workers_done.store(true, Ordering::SeqCst);
    {
        let _lock = output_cv_mutex.lock().unwrap();
    }
    output_cv.notify_one();

    let _ = output_handle.join();

    println!("===========================================");

    let total_duration = start.elapsed();

    println!("Summary:");
    println!("Total files: {}", total_files);
    println!("Processed: {}", ctx.processed_count.load(Ordering::Relaxed));
    println!("Skipped (filter): {}", ctx.skipped_count.load(Ordering::Relaxed));
    println!("No change needed: {}", ctx.no_change_count.load(Ordering::Relaxed));
    if !dry_run {
        println!("Successfully renamed: {}", ctx.success_count.load(Ordering::Relaxed));
        let failed_count = ctx.processed_count.load(Ordering::Relaxed)
            - ctx.success_count.load(Ordering::Relaxed)
            - ctx.no_change_count.load(Ordering::Relaxed);
        println!("Failed: {}", failed_count);

        if failed_count > 0 && ctx.failed_renames.count() > 0 {
            println!();
            println!("============================================");
            println!("FAILED RENAMES ({} files):", ctx.failed_renames.count());
            println!("============================================");

            for fail in ctx.failed_renames.get_all() {
                println!();
                println!("[{}] File: {}", fail.file_index, fail.file_name);
                println!("    Path: {}", fail.file_path.display());
                println!("    Error: {}", fail.error_code);
                println!("    Message: {}", fail.error_message);
                if !fail.suggestion.is_empty() {
                    println!("    Hint: {}", fail.suggestion);
                }
            }
            println!();
            println!("============================================");
        }
    }

    println!("Total execution time: {}ms", total_duration.as_millis());

    if ctx.processed_count.load(Ordering::Relaxed) > 0 {
        let pc = ctx.processed_count.load(Ordering::Relaxed);
        let avg = total_duration.as_millis() as f64 / pc as f64;
        println!("Average time per file: {:.2}ms", avg);
        let throughput = pc as f64 * 1000.0 / total_duration.as_millis() as f64;
        println!("Throughput: {:.2} files/second", throughput);
    }

    println!("Performance: {} threads utilized with sequential output", num_threads);
}

fn worker_fn(
    ctx: Arc<WorkerContext>,
    output_buffers: Arc<Mutex<HashMap<usize, OutputEntry>>>,
    cv: Arc<Condvar>,
    cv_mutex: Arc<Mutex<()>>,
    _workers_done: Arc<AtomicBool>,
) {
    loop {
        let file_idx = ctx.next_index.fetch_add(1, Ordering::Relaxed);
        if file_idx >= ctx.files.len() {
            break;
        }

        let file = &ctx.files[file_idx].path;
        let mut buffer = String::new();

        match file.file_name().and_then(|n| n.to_str()) {
            Some(file_name) => {
                buffer.push_str(&format!(
                    "[{}/{}] Processing: \"{}\"\n",
                    file_idx + 1,
                    ctx.files.len(),
                    file_name
                ));
            }
            None => {
                buffer.push_str(&format!(
                    "[{}/{}] Processing: <Error reading filename>\n",
                    file_idx + 1,
                    ctx.files.len()
                ));
                ctx.skipped_count.fetch_add(1, Ordering::Relaxed);
                store_output(&output_buffers, &cv, &cv_mutex, file_idx, buffer);
                continue;
            }
        }

        ctx.processed_count.fetch_add(1, Ordering::Relaxed);

        let hash_result = hash::process_single_file(file, ctx.algorithm);
        if !hash_result.success {
            buffer.push_str(&format!("  Error: {}\n\n", hash_result.error));
            store_output(&output_buffers, &cv, &cv_mutex, file_idx, buffer);
            continue;
        }

        let extension = file.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        let new_filename = format!("{}.{}", hash_result.hash, extension);
        let new_path = file.parent().unwrap_or(file).join(&new_filename);

        buffer.push_str(&format!("  Hash ({}): {}\n", ctx.algorithm, hash_result.hash));
        buffer.push_str(&format!("  New name: {}\n", new_filename));

        if file.file_name() == new_path.file_name() {
            buffer.push_str("  Status: No change needed (filename already matches hash)\n\n");
            ctx.no_change_count.fetch_add(1, Ordering::Relaxed);
        } else if !ctx.dry_run {
            match file_ops::rename_file(file, &new_path) {
                Ok(()) => {
                    buffer.push_str("  Status: Renamed successfully\n\n");
                    ctx.success_count.fetch_add(1, Ordering::Relaxed);
                }
                Err(e) => {
                    buffer.push_str("  Status: Failed to rename\n\n");
                    ctx.failed_renames.add(FailedRenameInfo {
                        file_path: file.clone(),
                        file_name: file.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string(),
                        error_code: e.error_code,
                        error_message: e.error_message,
                        suggestion: e.suggestion,
                        file_index: file_idx + 1,
                    });
                }
            }
        } else {
            buffer.push_str("  Status: Preview only (will be renamed)\n\n");
        }

        store_output(&output_buffers, &cv, &cv_mutex, file_idx, buffer);
    }
}

fn output_worker_fn(
    buffers: Arc<Mutex<HashMap<usize, OutputEntry>>>,
    next_output: Arc<AtomicUsize>,
    cv: Arc<Condvar>,
    cv_mutex: Arc<Mutex<()>>,
    workers_done: Arc<AtomicBool>,
    total_files: usize,
) {
    loop {
        let current = next_output.load(Ordering::Relaxed);
        if current >= total_files {
            break;
        }

        let is_ready = {
            let map = buffers.lock().unwrap();
            map.get(&current).map_or(false, |e| e.ready)
        };

        if !is_ready {
            let lock = cv_mutex.lock().unwrap();
            let is_ready = {
                let map = buffers.lock().unwrap();
                map.get(&current).map_or(false, |e| e.ready)
            };
            if !is_ready {
                if workers_done.load(Ordering::Relaxed) {
                    break;
                }
                let _guard = cv.wait(lock);
                continue;
            }
        }

        loop {
            let idx = next_output.load(Ordering::Relaxed);
            if idx >= total_files {
                break;
            }
            let entry = {
                let map = buffers.lock().unwrap();
                map.get(&idx).cloned()
            };
            match entry {
                Some(e) if e.ready => {
                    print!("{}", e.content);
                    let _ = io::stdout().flush();
                    next_output.fetch_add(1, Ordering::Relaxed);
                }
                _ => break,
            }
        }
    }
}

pub fn process_batch_mode(
    directory: &Path,
    algorithm: Algorithm,
    recursive: bool,
    dry_run: bool,
    allowed_extensions: HashSet<String>,
    num_threads: usize,
    batch_size: usize,
) {
    let start = Instant::now();

    if !directory.exists() || !directory.is_dir() {
        eprintln!("Error: Invalid directory path: {}", directory.display());
        return;
    }

    println!("Scanning directory: {}", directory.display());
    println!("Algorithm: {}", algorithm);
    println!("Recursive: {}", if recursive { "Yes" } else { "No" });
    println!("Mode: {}", if dry_run { "Preview" } else { "Execute" });
    println!("Threads: {}", num_threads);
    println!("Batch size: {}", batch_size);
    println!("Processing device: CPU");

    if !allowed_extensions.is_empty() {
        print!("Extensions filter: ");
        for (i, ext) in allowed_extensions.iter().enumerate() {
            print!("\"{}\"", ext);
            if i < allowed_extensions.len() - 1 {
                print!(", ");
            }
        }
        println!();
    } else {
        println!("Extensions filter: All files");
    }

    println!("===========================================");

    let scan_start = Instant::now();
    let mut files = scanner::scan_directory(directory, recursive, &allowed_extensions);
    let scan_duration = scan_start.elapsed();
    println!(
        "Found {} files in {}ms.",
        files.len(),
        scan_duration.as_millis()
    );
    println!();

    if files.is_empty() {
        println!("No files found to process.");
        return;
    }

    let total_files = files.len();

    files.sort_by(|a, b| b.size.cmp(&a.size));

    let batches: Vec<Vec<ScannedFile>> = files
        .chunks(batch_size)
        .map(|chunk| chunk.to_vec())
        .collect();

    println!(
        "Created {} batches for processing.",
        batches.len()
    );
    println!();

    let ctx = Arc::new(WorkerContext {
        files,
        algorithm,
        dry_run,
        next_index: AtomicUsize::new(0),
        processed_count: AtomicUsize::new(0),
        success_count: AtomicUsize::new(0),
        skipped_count: AtomicUsize::new(0),
        no_change_count: AtomicUsize::new(0),
        failed_renames: FailedRenameCollector::new(),
    });

    let stdout_mutex: Arc<Mutex<()>> = Arc::new(Mutex::new(()));

    let mut handles = Vec::with_capacity(num_threads);
    for _ in 0..num_threads {
        let ctx_clone = Arc::clone(&ctx);
        let batches_clone = batches.clone();
        let stdout_clone = Arc::clone(&stdout_mutex);
        handles.push(thread::spawn(move || {
            batch_worker_fn(ctx_clone, batches_clone, stdout_clone);
        }));
    }

    for h in handles {
        let _ = h.join();
    }

    println!("===========================================");

    let total_duration = start.elapsed();

    println!("Summary:");
    println!("Total files: {}", total_files);
    println!("Processed: {}", ctx.processed_count.load(Ordering::Relaxed));
    println!("Skipped (filter): {}", ctx.skipped_count.load(Ordering::Relaxed));
    println!("No change needed: {}", ctx.no_change_count.load(Ordering::Relaxed));
    if !dry_run {
        println!("Successfully renamed: {}", ctx.success_count.load(Ordering::Relaxed));
        let failed_count = ctx.processed_count.load(Ordering::Relaxed)
            - ctx.success_count.load(Ordering::Relaxed)
            - ctx.no_change_count.load(Ordering::Relaxed);
        println!("Failed: {}", failed_count);

        if failed_count > 0 && ctx.failed_renames.count() > 0 {
            println!();
            println!("============================================");
            println!("FAILED RENAMES ({} files):", ctx.failed_renames.count());
            println!("============================================");

            for fail in ctx.failed_renames.get_all() {
                println!();
                println!("[{}] File: {}", fail.file_index, fail.file_name);
                println!("    Path: {}", fail.file_path.display());
                println!("    Error: {}", fail.error_code);
                println!("    Message: {}", fail.error_message);
                if !fail.suggestion.is_empty() {
                    println!("    Hint: {}", fail.suggestion);
                }
            }
            println!();
            println!("============================================");
        }
    }

    println!("Total execution time: {}ms", total_duration.as_millis());

    if ctx.processed_count.load(Ordering::Relaxed) > 0 {
        let pc = ctx.processed_count.load(Ordering::Relaxed);
        let avg = total_duration.as_millis() as f64 / pc as f64;
        println!("Average time per file: {:.2}ms", avg);
        let throughput = pc as f64 * 1000.0 / total_duration.as_millis() as f64;
        println!("Throughput: {:.2} files/second", throughput);
    }

    println!(
        "Performance: {} threads, {} batches utilized",
        num_threads,
        batches.len()
    );
}

fn batch_worker_fn(
    ctx: Arc<WorkerContext>,
    batches: Vec<Vec<ScannedFile>>,
    stdout_mutex: Arc<Mutex<()>>,
) {
    let batch_size = match batches.first() {
        Some(first) => first.len(),
        None => return,
    };

    loop {
        let batch_idx = ctx.next_index.fetch_add(1, Ordering::Relaxed);
        if batch_idx >= batches.len() {
            break;
        }

        let batch = &batches[batch_idx];

        for (file_idx, scanned) in batch.iter().enumerate() {
            let file = &scanned.path;
            let global_index = batch_idx * batch_size + file_idx;

            let mut out = String::new();
            let file_name_str = file.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");

            if file_name_str.is_empty() {
                out.push_str(&format!(
                    "[{}/{}] Processing: <Error reading filename>\n",
                    global_index + 1,
                    ctx.files.len()
                ));
                ctx.skipped_count.fetch_add(1, Ordering::Relaxed);
                let _guard = stdout_mutex.lock().unwrap();
                print!("{}", out);
                let _ = io::stdout().flush();
                continue;
            }

            out.push_str(&format!(
                "[{}/{}] Batch {}/{} Processing: \"{}\"\n",
                global_index + 1,
                ctx.files.len(),
                batch_idx + 1,
                batches.len(),
                file_name_str
            ));

            let hash_result = hash::process_single_file(file, ctx.algorithm);
            if !hash_result.success {
                out.push_str(&format!("  Error: {}\n\n", hash_result.error));
                let _guard = stdout_mutex.lock().unwrap();
                print!("{}", out);
                let _ = io::stdout().flush();
                continue;
            }

            let extension = file.extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");
            let new_filename = format!("{}.{}", hash_result.hash, extension);
            let new_path = file.parent().unwrap_or(file).join(&new_filename);

            out.push_str(&format!("  Hash ({}): {}\n", ctx.algorithm, hash_result.hash));
            out.push_str(&format!("  New name: {}\n", new_filename));

            if file.file_name() == new_path.file_name() {
                out.push_str("  Status: No change needed (filename already matches hash)\n\n");
                ctx.no_change_count.fetch_add(1, Ordering::Relaxed);
            } else if !ctx.dry_run {
                match file_ops::rename_file(file, &new_path) {
                    Ok(()) => {
                        out.push_str("  Status: Renamed successfully\n\n");
                        ctx.success_count.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(e) => {
                        out.push_str("  Status: Failed to rename\n\n");
                        ctx.failed_renames.add(FailedRenameInfo {
                            file_path: file.clone(),
                            file_name: file_name_str.to_string(),
                            error_code: e.error_code,
                            error_message: e.error_message,
                            suggestion: e.suggestion,
                            file_index: global_index + 1,
                        });
                    }
                }
            } else {
                out.push_str("  Status: Preview only (will be renamed)\n\n");
            }

            ctx.processed_count.fetch_add(1, Ordering::Relaxed);

            let _guard = stdout_mutex.lock().unwrap();
            print!("{}", out);
            let _ = io::stdout().flush();
        }
    }
}
