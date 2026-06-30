use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tauri::Emitter;
use file_renamer_lib::{Algorithm, ProcessOptions, ProcessingMode, ProgressReporter, process_directory};

/// Shared state for the processing task
pub struct AppState {
    cancel_flag: Arc<AtomicBool>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            cancel_flag: Arc::new(AtomicBool::new(false)),
        }
    }
}

/// Tauri event reporter that batches output to reduce IPC overhead
struct TauriReporter {
    app: tauri::AppHandle,
    buffer: Mutex<String>,
    last_flush: Mutex<Instant>,
}

impl TauriReporter {
    fn new(app: tauri::AppHandle) -> Self {
        Self {
            app,
            buffer: Mutex::new(String::with_capacity(8192)),
            last_flush: Mutex::new(Instant::now()),
        }
    }

    fn flush(&self) {
        let mut buf = self.buffer.lock().unwrap();
        if buf.is_empty() {
            return;
        }
        let text = std::mem::take(&mut *buf);
        drop(buf);
        let _ = self.app.emit("cli-output", serde_json::json!({"text": text}));
    }

    fn try_flush(&self) {
        let should_flush = {
            let buf = self.buffer.lock().unwrap();
            let last = self.last_flush.lock().unwrap();
            buf.len() > 4096 || last.elapsed() > Duration::from_millis(50)
        };
        if should_flush {
            self.flush();
            *self.last_flush.lock().unwrap() = Instant::now();
        }
    }
}

impl ProgressReporter for TauriReporter {
    fn on_output(&self, text: &str) {
        self.buffer.lock().unwrap().push_str(text);
        self.try_flush();
    }
}

#[tauri::command]
pub async fn run_cli(
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
    directory: String,
    algorithm: Option<String>,
    mode: Option<String>,
    extensions: Option<String>,
    threads: Option<String>,
    batch: Option<String>,
    execute: Option<bool>,
    _yes: Option<bool>,
    recursive: Option<bool>,
    _buffer_kb: Option<String>,
    _mmap_chunk_mb: Option<String>,
) -> Result<(), String> {
    let dir = directory.trim().to_string();
    if dir.is_empty() {
        return Err("目录路径不能为空".to_string());
    }
    let dir_path = PathBuf::from(&dir);
    if !dir_path.exists() {
        return Err(format!("目录不存在: {}", dir));
    }
    if !dir_path.is_dir() {
        return Err(format!("路径不是文件夹: {}", dir));
    }

    // Reset cancel flag
    state.cancel_flag.store(false, Ordering::Relaxed);

    let algo = algorithm.unwrap_or_else(|| "MD5".to_string()).to_uppercase();
    let algorithm = match Algorithm::from_str(&algo) {
        Some(a) => a,
        None => {
            return Err(format!("不支持的算法: {}", algo));
        }
    };

    let m = mode.unwrap_or_else(|| "multi-thread".to_string()).to_lowercase();
    let processing_mode = match m.as_str() {
        "single-thread" => ProcessingMode::SingleThread,
        "batch-mode" => ProcessingMode::BatchMode,
        _ => ProcessingMode::MultiThread,
    };

    let dry_run = !execute.unwrap_or(false);

    let allowed_extensions: HashSet<String> = match extensions.as_ref() {
        Some(ext) if !ext.trim().is_empty() => {
            ext.split(|c: char| c == ',' || c.is_whitespace())
                .filter(|s| !s.is_empty())
                .map(|s| {
                    let trimmed = s.trim().trim_start_matches('.').to_lowercase();
                    if trimmed.starts_with('.') { trimmed } else { format!(".{}", trimmed) }
                })
                .collect()
        }
        _ => HashSet::new(),
    };

    let num_threads: usize = threads
        .as_ref()
        .filter(|s| !s.trim().is_empty())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or_else(|| {
            let n = num_cpus::get();
            if n == 0 { 4 } else { n }
        });

    let batch_size: usize = batch
        .as_ref()
        .filter(|s| !s.trim().is_empty())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or_else(|| std::cmp::max(1, num_threads * 4));

    let options = ProcessOptions {
        directory: dir_path,
        algorithm,
        recursive: recursive.unwrap_or(false),
        dry_run,
        allowed_extensions,
        num_threads,
        batch_size,
        mode: processing_mode,
    };

    let reporter = Arc::new(TauriReporter::new(app.clone()));

    // Run processing in a blocking thread to avoid blocking the async runtime
    let _cancel_flag = state.cancel_flag.clone();
    let reporter_for_done = reporter.clone();
    tokio::task::spawn_blocking(move || {
        process_directory(options, reporter);

        // Flush any remaining buffered output
        reporter_for_done.flush();

        let _ = app.emit(
            "cli-output",
            serde_json::json!({"text": "", "done": true}),
        );
    });

    Ok(())
}

#[tauri::command]
pub async fn stop_cli(state: tauri::State<'_, AppState>) -> Result<bool, String> {
    state.cancel_flag.store(true, Ordering::Relaxed);
    Ok(true)
}

#[tauri::command]
pub fn which_exe() -> serde_json::Value {
    // Library mode - no external exe needed
    serde_json::json!({
        "path": "built-in",
        "exists": true,
    })
}

#[tauri::command]
pub async fn browse_folder(_app: tauri::AppHandle) -> Result<String, String> {
    use tauri_plugin_dialog::DialogExt;
    let folder = _app
        .dialog()
        .file()
        .set_title("选择文件夹")
        .blocking_pick_folder();
    match folder {
        Some(path) => Ok(path.to_string()),
        None => Err("cancelled".to_string()),
    }
}
