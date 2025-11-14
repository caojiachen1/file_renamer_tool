#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};
use tauri::{AppHandle, Manager, State};
fn device_list_timeout() -> Duration {
    if let Ok(ms) = std::env::var("DEVICE_LIST_TIMEOUT_MS") {
        if let Ok(v) = ms.trim().parse::<u64>() { return Duration::from_millis(v.max(1000)); }
    }
    if let Ok(secs) = std::env::var("DEVICE_LIST_TIMEOUT_S") {
        if let Ok(v) = secs.trim().parse::<u64>() { return Duration::from_secs(v.max(1)); }
    }
    Duration::from_secs(15)
}

#[derive(Default)]
struct ProcState {
    child: Option<Child>,
}

type SharedProc = Arc<Mutex<ProcState>>;

fn repo_root_candidates() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        let mut cur = exe.parent().map(|p| p.to_path_buf());
        for _ in 0..8 { // walk up several levels to reach repo root
            if let Some(ref c) = cur { v.push(c.clone()); }
            cur = cur.and_then(|p| p.parent().map(|pp| pp.to_path_buf()));
            if cur.is_none() { break; }
        }
    }
    if let Ok(mut cwd) = std::env::current_dir() {
        for _ in 0..8 {
            v.push(cwd.clone());
            if let Some(p) = cwd.parent() { cwd = p.to_path_buf(); } else { break; }
        }
    }
    v
}

fn find_executable() -> Option<PathBuf> {
    // 1) explicit env override
    if let Ok(p) = std::env::var("FILE_RENAMER_EXE") {
        let pb = PathBuf::from(p);
        if pb.exists() { return Some(pb); }
    }

    let names = [
        "file_renamer.exe",
        "file_renamer_cli.exe",
        "build/Release/file_renamer_cli.exe",
        "build/Debug/file_renamer_cli_d.exe",
        "build/Release/file_renamer.exe",
    ];
    for root in repo_root_candidates() {
        for n in &names {
            let p = root.join(n);
            if p.exists() {
                return Some(p);
            }
        }
    }
    None
}

fn find_executable_with_app(app: Option<&AppHandle>) -> Option<PathBuf> {
    // env override always first
    if let Some(p) = std::env::var("FILE_RENAMER_EXE").ok().map(PathBuf::from) {
        if p.exists() { return Some(p); }
    }
    // packaged resource
    if let Some(app) = app {
        if let Some(p) = app.path_resolver().resolve_resource("file_renamer.exe") {
            if p.exists() { return Some(p); }
        }
        // defensive: some installers may place resources under a sibling "resources" or "_up_" folder
        if let Ok(me) = std::env::current_exe() {
            if let Some(dir) = me.parent() {
                let candidates = [
                    PathBuf::from("file_renamer.exe"),
                    PathBuf::from("resources").join("file_renamer.exe"),
                    PathBuf::from("_up_").join("file_renamer.exe"),
                    PathBuf::from("..\\").join("resources").join("file_renamer.exe"),
                ];
                for rel in candidates {
                    let p = dir.join(&rel);
                    if p.exists() { return Some(p); }
                }
            }
        }
    }
    // fallback to filesystem search
    find_executable()
}

#[derive(Serialize)]
struct WhichExeResp {
    path: Option<String>,
    exists: bool,
}

#[tauri::command]
fn which_exe(app: AppHandle) -> WhichExeResp {
    let exe = find_executable_with_app(Some(&app));
    WhichExeResp {
        path: exe.as_ref().map(|p| p.to_string_lossy().to_string()),
        exists: exe.map(|p| p.exists()).unwrap_or(false),
    }
}

#[derive(Serialize)]
struct DevicesResp {
    ok: bool,
    output: Option<String>,
    error: Option<String>,
    #[serde(rename = "deviceIds")] // keep JSON field camelCase for frontend
    device_ids: Option<Vec<i32>>,
}

#[tauri::command]
fn list_devices(app: AppHandle) -> DevicesResp {
    let exe = match find_executable_with_app(Some(&app)) {
        Some(p) => p,
        None => {
            return DevicesResp {
                ok: false,
                output: None,
                error: Some("找不到可执行文件 file_renamer.exe".to_string()),
                device_ids: None,
            }
        }
    };

    let mut child = match Command::new(&exe)
        .arg("-d")
        .arg("list")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn() {
        Ok(c) => c,
        Err(e) => {
            return DevicesResp {
                ok: false,
                output: None,
                error: Some(format!("执行失败: {}", e)),
                device_ids: None,
            };
        }
    };

    // Read stdout in background while we monitor timeout
    let mut stdout = match child.stdout.take() {
        Some(s) => s,
        None => {
            return DevicesResp { ok: false, output: None, error: Some("无法捕获输出".into()), device_ids: None };
        }
    };
    let output_buf: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
    let out_clone = output_buf.clone();
    let reader_handle = thread::spawn(move || {
        let reader = BufReader::new(&mut stdout);
        for line in reader.lines() {
            if let Ok(l) = line { let mut g = out_clone.lock().unwrap(); g.push_str(&l); g.push('\n'); }
            else { break; }
        }
    });

    let start = Instant::now();
    let timeout = device_list_timeout();
    let mut timed_out = false;
    loop {
        match child.try_wait() {
            Ok(Some(_status)) => break,
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    timed_out = true;
                    break;
                }
                thread::sleep(Duration::from_millis(50));
            }
            Err(_e) => { break; }
        }
    }

    // Ensure reader thread exits
    let _ = reader_handle.join();
    let out = output_buf.lock().unwrap().clone();

    let re = Regex::new(r"Device\s+(-?\d+):").unwrap();
    let mut ids = vec![];
    for cap in re.captures_iter(&out) {
        if let Some(m) = cap.get(1) {
            if let Ok(id) = m.as_str().parse::<i32>() { ids.push(id); }
        }
    }

    if timed_out {
        return DevicesResp {
            ok: false,
            output: if out.is_empty() { None } else { Some(out) },
            error: Some("设备查询超时(>3s)，请点击刷新重试或稍后再试。".into()),
            device_ids: if ids.is_empty() { None } else { Some(ids) },
        };
    }

    DevicesResp { ok: true, output: Some(out), error: None, device_ids: Some(ids) }
}

#[derive(Debug, Deserialize, Clone)]
struct RunPayload {
    directory: String,
    algorithm: Option<String>,
    mode: Option<String>,
    device: Option<String>,
    extensions: Option<String>,
    threads: Option<String>,
    batch: Option<String>,
    quick: Option<bool>,
    execute: Option<bool>,
    yes: Option<bool>,
    recursive: Option<bool>,
    extreme: Option<bool>,
    buffer_kb: Option<String>,
    mmap_chunk_mb: Option<String>,
}

fn add_opt(args: &mut Vec<String>, key: Option<String>, flag: &str) {
    if let Some(v) = key { if !v.trim().is_empty() { args.push(flag.to_string()); args.push(v.trim().to_string()); } }
}

fn build_args(exe: &Path, p: &RunPayload) -> anyhow::Result<Vec<String>> {
    let dir = p.directory.trim();
    if dir.is_empty() { anyhow::bail!("目录路径不能为空"); }
    let dpath = Path::new(dir);
    if !dpath.exists() || !dpath.is_dir() { anyhow::bail!(format!("目录不存在或不是文件夹: {}", dir)); }

    let mut args: Vec<String> = vec![exe.to_string_lossy().to_string(), dpath.to_string_lossy().to_string()];

    let algo = p.algorithm.clone().unwrap_or_else(|| "MD5".into()).to_uppercase();
    let allowed = ["MD5", "SHA1", "SHA256", "SHA512", "CRC32", "BLAKE2B"];
    if !allowed.contains(&algo.as_str()) { anyhow::bail!(format!("不支持的算法: {}", algo)); }
    if algo != "MD5" { args.push("-a".into()); args.push(algo); }

    match p.mode.clone().unwrap_or_else(|| "ultra-fast".into()).to_lowercase().as_str() {
        "single-thread" => args.push("--single-thread".into()),
        "multi-thread" => args.push("--multi-thread".into()),
        "batch-mode" => args.push("--batch-mode".into()),
        _ => args.push("--ultra-fast".into()),
    }

    if p.recursive.unwrap_or(false) { args.push("-r".into()); }
    if p.execute.unwrap_or(false) { args.push("-e".into()); if p.yes.unwrap_or(false) { args.push("-y".into()); } }
    if let Some(q) = p.quick { args.push(if q { "-q" } else { "--no-quick" }.into()); }

    if let Some(t) = &p.threads { if !t.trim().is_empty() { args.push("-t".into()); args.push(t.trim().into()); } }
    else if p.mode.clone().unwrap_or_else(|| "ultra-fast".into()).to_lowercase() == "ultra-fast" {
        if let Some(n) = std::thread::available_parallelism().ok().map(|n| n.get()) { if n > 0 { args.push("-t".into()); args.push(n.to_string()); } }
    }
    if let Some(b) = &p.batch { if !b.trim().is_empty() { args.push("-b".into()); args.push(b.trim().into()); } }

    if let Some(exts_raw) = &p.extensions {
        let s = exts_raw.trim();
        if !s.is_empty() {
            let parts: Vec<&str> = s.split(|c: char| c == ',' || c.is_whitespace()).collect();
            let mut uniq: Vec<String> = parts.into_iter().filter(|p| !p.is_empty()).map(|p| p.trim_start_matches('.').to_lowercase()).collect();
            uniq.sort(); uniq.dedup();
            if !uniq.is_empty() { args.push("-x".into()); args.push(uniq.join(",")); }
        }
    }

    if let Some(d) = &p.device {
        let dv = d.trim();
        if !dv.is_empty() {
            let low = dv.to_lowercase();
            if ["auto", "cpu", "list"].contains(&low.as_str()) { args.push("-d".into()); args.push(low); }
            else if dv.parse::<i32>().is_ok() { args.push("-d".into()); args.push(dv.into()); }
            else { anyhow::bail!(format!("设备选择无效: {}", dv)); }
        }
    }

    if p.extreme.unwrap_or(false) { args.push("--extreme".into()); }

    add_opt(&mut args, p.buffer_kb.clone(), "--buffer-kb");
    add_opt(&mut args, p.mmap_chunk_mb.clone(), "--mmap-chunk-mb");

    Ok(args)
}

#[tauri::command]
fn start_run(app: AppHandle, state: State<SharedProc>, payload: RunPayload) -> Result<String, String> {
    let exe = find_executable_with_app(Some(&app)).ok_or_else(|| "找不到可执行文件 file_renamer.exe".to_string())?;
    let args = build_args(&exe, &payload).map_err(|e| e.to_string())?;
    let cmd_line = args.join(" ");

    let mut guard = state.lock().unwrap();
    if guard.child.is_some() { return Err("已有任务在运行中，请先停止或等待完成。".into()); }

    let mut cmd = Command::new(&args[0]);
    cmd.args(&args[1..])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| format!("启动失败：{}", e))?;
    let stdout = child.stdout.take().ok_or_else(|| "无法捕获输出".to_string())?;

    let app_handle = app.clone();
    app_handle.emit_all("cli-output", format!("运行命令: {}\n\n", &cmd_line)).ok();

    guard.child = Some(child);
    drop(guard);
    let state_arc = state.inner().clone();

    thread::spawn(move || {
        // Switch to chunked reading to reduce event overhead and mitigate perceived lag.
        // We batch output and flush every ~80ms or when buffer grows large.
        let mut reader = stdout;
        let mut buf = [0u8; 4096];
        let mut acc = String::new();
        let flush_interval = Duration::from_millis(80);
        let mut last_flush = Instant::now();
        loop {
            match reader.read(&mut buf) {
                Ok(0) => break, // EOF
                Ok(n) => {
                    // decode incremental chunk
                    let chunk = String::from_utf8_lossy(&buf[..n]);
                    acc.push_str(&chunk);
                    let now = Instant::now();
                    if acc.len() >= 8192 || now.duration_since(last_flush) >= flush_interval {
                        let send = acc.clone();
                        acc.clear();
                        last_flush = now;
                        let _ = app_handle.emit_all("cli-output", send);
                    }
                }
                Err(_) => break,
            }
        }
        if !acc.is_empty() { let _ = app_handle.emit_all("cli-output", acc); }
        // wait for exit code using shared state
        let code = {
            let mut guard = state_arc.lock().unwrap();
            if let Some(mut c) = guard.child.take() {
                match c.wait() { Ok(s) => s.code().unwrap_or(-1), Err(_) => -1 }
            } else { -1 }
        };
        let _ = app_handle.emit_all("cli-output", format!("\n[exit-code] {}\n", code));
        let _ = app_handle.emit_all("cli-exit", code);
    });

    Ok(cmd_line)
}

#[tauri::command]
fn stop_run(state: State<SharedProc>) -> Result<(), String> {
    let mut guard = state.lock().unwrap();
    if let Some(mut c) = guard.child.take() {
        c.kill().map_err(|e| e.to_string())?;
        Ok(())
    } else {
        Err("没有正在运行的任务。".into())
    }
}

fn main() {
    tauri::Builder::default()
        .manage::<SharedProc>(Arc::new(Mutex::new(ProcState::default())))
        .invoke_handler(tauri::generate_handler![which_exe, list_devices, start_run, stop_run])
    .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
