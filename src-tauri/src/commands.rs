use std::path::PathBuf;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use tauri::Emitter;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};

#[derive(Default)]
pub struct AppState {
    current_child: Arc<Mutex<Option<Child>>>,
}

fn find_executable() -> Option<PathBuf> {
    if let Ok(env_path) = std::env::var("FILE_RENAMER_EXE") {
        let p = PathBuf::from(&env_path);
        if p.exists() {
            return Some(p);
        }
    }

    let sidecar_names = [
        "file_renamer.exe",
        "file_renamer-cli.exe",
    ];

    let bases: Vec<PathBuf> = vec![
        std::env::current_dir()
            .ok()
            .and_then(|d| d.parent().map(|p| p.to_path_buf()))
            .unwrap_or_default(),
        std::env::current_exe()
            .ok()
            .and_then(|d| d.parent().map(|p| p.to_path_buf()))
            .unwrap_or_default(),
        std::env::current_dir().unwrap_or_default(),
    ];

    for base in &bases {
        for name in &sidecar_names {
            let p = base.join(name);
            if p.exists() {
                return Some(p);
            }
        }
    }

    let build_rel = [
        "build/Release/file_renamer_cli.exe",
        "build/Debug/file_renamer_cli_d.exe",
        "build/Release/file_renamer.exe",
    ];
    for base in &bases {
        for rel in &build_rel {
            let p = base.join(rel);
            if p.exists() {
                return Some(p);
            }
        }
    }
    None
}

#[tauri::command]
pub fn which_exe() -> serde_json::Value {
    let exe = find_executable();
    serde_json::json!({
        "path": exe.as_ref().map(|p| p.to_string_lossy().to_string()),
        "exists": exe.as_ref().map_or(false, |p| p.exists()),
    })
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
    yes: Option<bool>,
    recursive: Option<bool>,
    buffer_kb: Option<String>,
    mmap_chunk_mb: Option<String>,
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

    let exe = find_executable().ok_or("找不到可执行文件 file_renamer.exe")?;

    let mut args: Vec<String> = vec![exe.to_string_lossy().to_string(), dir];

    let algo = algorithm.unwrap_or_else(|| "MD5".to_string()).to_uppercase();
    if !["MD5", "SHA1", "SHA256", "SHA512", "CRC32", "BLAKE2B"].contains(&algo.as_str()) {
        return Err(format!("不支持的算法: {}", algo));
    }
    if algo != "MD5" {
        args.extend_from_slice(&["-a".to_string(), algo]);
    }

    let m = mode.unwrap_or_else(|| "multi-thread".to_string()).to_lowercase();
    if m == "single-thread" {
        args.push("--single-thread".to_string());
    } else if m == "batch-mode" {
        args.push("--batch-mode".to_string());
    }

    if recursive.unwrap_or(false) {
        args.push("-r".to_string());
    }

    if execute.unwrap_or(false) {
        args.push("-e".to_string());
        if yes.unwrap_or(false) {
            args.push("-y".to_string());
        }
    }

    if let Some(t) = threads.as_ref().filter(|s| !s.trim().is_empty()) {
        args.extend_from_slice(&["-t".to_string(), t.trim().to_string()]);
    }
    if let Some(b) = batch.as_ref().filter(|s| !s.trim().is_empty()) {
        args.extend_from_slice(&["-b".to_string(), b.trim().to_string()]);
    }
    if let Some(ext) = extensions.as_ref().filter(|s| !s.trim().is_empty()) {
        let parts: Vec<String> = ext
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|s| !s.is_empty())
            .map(|s| s.trim().trim_start_matches('.').to_lowercase())
            .collect();
        let norm: Vec<&str> = parts.iter().map(|s| s.as_str()).collect();
        if !norm.is_empty() {
            args.extend_from_slice(&["-x".to_string(), norm.join(",")]);
        }
    }

    if let Some(v) = buffer_kb.as_ref().filter(|s| !s.trim().is_empty()) {
        args.extend_from_slice(&["--buffer-kb".to_string(), v.trim().to_string()]);
    }
    if let Some(v) = mmap_chunk_mb.as_ref().filter(|s| !s.trim().is_empty()) {
        args.extend_from_slice(&["--mmap-chunk-mb".to_string(), v.trim().to_string()]);
    }

    let cmd_display = args.join(" ");
    app.emit("cli-output", serde_json::json!({"text": format!("运行命令: {}\n\n", cmd_display)}))
        .map_err(|e| e.to_string())?;

    {
        let old_child = {
            let mut guard = state.current_child.lock().map_err(|e| e.to_string())?;
            guard.take()
        };
        if let Some(mut child) = old_child {
            let _ = child.kill().await;
        }
    }

    let mut child = Command::new(&args[0])
        .args(&args[1..])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .creation_flags(0x08000000)
        .spawn()
        .map_err(|e| format!("启动失败: {}", e))?;

    let stdout = child.stdout.take().ok_or("无法获取进程输出")?;

    {
        let mut guard = state.current_child.lock().map_err(|e| e.to_string())?;
        *guard = Some(child);
    }

    let app_clone = app.clone();
    let state_clone = state.inner().current_child.clone();

    tokio::spawn(async move {
        let reader = BufReader::new(stdout);
        let mut lines = reader.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let _ = app_clone.emit(
                "cli-output",
                serde_json::json!({"text": format!("{}\n", line)}),
            );
        }

        let child = {
            let mut guard = state_clone.lock().unwrap();
            guard.take()
        };
        let exit_code = if let Some(child) = child {
            match child.wait_with_output().await {
                Ok(status) => status.status.code().unwrap_or(-1),
                Err(_) => -1,
            }
        } else {
            -1
        };

        let _ = app_clone.emit(
            "cli-output",
            serde_json::json!({"text": format!("\n[exit-code] {}\n", exit_code), "done": true}),
        );
    });

    Ok(())
}

#[tauri::command]
pub async fn stop_cli(state: tauri::State<'_, AppState>) -> Result<bool, String> {
    let child = {
        let mut guard = state.current_child.lock().map_err(|e| e.to_string())?;
        guard.take()
    };
    if let Some(mut child) = child {
        child.kill().await.map_err(|e| e.to_string())?;
        Ok(true)
    } else {
        Err("没有正在运行的任务。".to_string())
    }
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
