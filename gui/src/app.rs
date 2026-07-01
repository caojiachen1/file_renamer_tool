use iced::widget::{button, checkbox, column, container, row, scrollable, text, text_input};
use iced::{Color, Element, Length, Subscription, Task, Theme, time};
use std::collections::VecDeque;
use std::io::Read;
use std::sync::mpsc;
use std::time::Duration;

use crate::theme::colors;

// ── Constants ───────────────────────────────────────────────────────────────

const ALG: &[&str] = &["MD5","SHA1","SHA256","SHA512","CRC32","BLAKE2B"];
const MOD: &[(&str, &str)] = &[
    ("multi-thread", "Multi Thread (默认)"),
    ("batch-mode",   "Batch Mode"),
    ("single-thread","Single Thread"),
];

const READ_BUF: usize = 256 * 1024;   // 256 KB per read syscall
const BATCH_BUF: usize = 128 * 1024;  // 128 KB send batch — smaller = smoother UI
const MAX_LINES: usize = 6000;         // keep all logs, cap display at last N lines
const DISPLAY_LINES: usize = 500;      // lines actually rendered in the widget

// ── Message ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Message {
    DirectoryChanged(String), AlgorithmChanged(usize), ModeChanged(usize),
    ExtensionsChanged(String), ThreadsChanged(String), BatchSizeChanged(String),
    ToggleRecursive(bool), ToggleExecute(bool), BrowseFolder, FolderPicked(Option<String>),
    Run, Stop, ClearOutput, Tick,
}

// ── App state ───────────────────────────────────────────────────────────────

pub struct App {
    directory: String, algorithm_index: usize, mode_index: usize,
    extensions: String, threads: String, batch_size: String,
    recursive: bool, execute: bool, is_running: bool,
    lines: VecDeque<String>,     // all log lines (O(1) push_back / pop_front)
    display_cache: String,       // last DISPLAY_LINES joined — only rebuilt when lines change
    cache_dirty: bool,
    rx: Option<mpsc::Receiver<Vec<u8>>>,
}

pub fn new() -> (App, Task<Message>) {
    (App {
        directory: String::new(), algorithm_index: 0, mode_index: 0,
        extensions: "jpg,jpeg,png".into(), threads: String::new(), batch_size: String::new(),
        recursive: false, execute: false, is_running: false,
        lines: VecDeque::new(), display_cache: String::new(), cache_dirty: false,
        rx: None,
    }, Task::none())
}

// ── Update ──────────────────────────────────────────────────────────────────

pub fn update(app: &mut App, msg: Message) -> Task<Message> {
    match msg {
        Message::DirectoryChanged(v) => app.directory = v,
        Message::AlgorithmChanged(i) => app.algorithm_index = i.min(ALG.len() - 1),
        Message::ModeChanged(i)      => app.mode_index = i.min(MOD.len() - 1),
        Message::ExtensionsChanged(v) => app.extensions = v,
        Message::ThreadsChanged(v)    => app.threads = v,
        Message::BatchSizeChanged(v)  => app.batch_size = v,
        Message::ToggleRecursive(v)   => app.recursive = v,
        Message::ToggleExecute(v)     => app.execute = v,

        Message::BrowseFolder => {
            return Task::perform(async {
                rfd::AsyncFileDialog::new().set_title("选择文件夹").pick_folder().await
                    .map(|h| h.path().to_string_lossy().to_string())
            }, Message::FolderPicked);
        }
        Message::FolderPicked(Some(p)) => app.directory = p,
        Message::FolderPicked(None)    => {}

        Message::Run => {
            if app.is_running || app.directory.is_empty() { return Task::none(); }
            app.is_running = true;
            app.lines.clear();
            app.display_cache.clear();
            app.cache_dirty = false;
            app.lines.push_back("Starting...".into());
            app.cache_dirty = true;

            let exe = find_cli_exe();
            let mut args: Vec<String> = Vec::with_capacity(12);
            args.push(app.directory.clone());
            let a = ALG[app.algorithm_index];
            if a != "MD5" { args.push("-a".into()); args.push(a.into()); }
            match MOD[app.mode_index].0 {
                "single-thread" => args.push("--single-thread".into()),
                "batch-mode"    => args.push("--batch-mode".into()),
                _ => {}
            }
            if app.recursive  { args.push("-r".into()); }
            if app.execute    { args.push("-e".into()); args.push("-y".into()); }
            if !app.extensions.is_empty() { args.push("-x".into()); args.push(app.extensions.clone()); }
            if !app.threads.is_empty()    { args.push("-t".into()); args.push(app.threads.clone()); }
            if !app.batch_size.is_empty() { args.push("-b".into()); args.push(app.batch_size.clone()); }

            let (tx, rx) = mpsc::channel();
            app.rx = Some(rx);
            std::thread::spawn(move || {
                if let Err(e) = run_cli_process(&exe, &args, &tx) {
                    let _ = tx.send(format!("[ERROR] {}\n", e).into_bytes());
                }
            });
        }

        Message::Stop => app.is_running = false,
        Message::ClearOutput => {
            app.lines.clear();
            app.display_cache.clear();
            app.cache_dirty = false;
        }

        Message::Tick => {
            // 1. Drain all pending chunks from channel
            let mut chunks: Vec<Vec<u8>> = Vec::new();
            if let Some(rx) = &app.rx {
                loop {
                    match rx.try_recv() {
                        Ok(c) => chunks.push(c),
                        Err(mpsc::TryRecvError::Empty) => break,
                        Err(mpsc::TryRecvError::Disconnected) => {
                            app.is_running = false;
                            break;
                        }
                    }
                }
            }

            if chunks.is_empty() && !app.cache_dirty {
                return Task::none();
            }

            // 2. Decode + split into lines, push to deque — O(n) where n = new bytes
            for chunk in &chunks {
                let decoded = String::from_utf8_lossy(chunk);
                for line in decoded.split('\n') {
                    if line.is_empty() { continue; }
                    app.lines.push_back(line.to_string());
                    app.cache_dirty = true;
                }
            }

            // 3. Trim to MAX_LINES — O(1) pop_front per line dropped
            while app.lines.len() > MAX_LINES {
                app.lines.pop_front();
            }

            // 4. Rebuild display cache only when dirty — O(DISPLAY_LINES) not O(total)
            if app.cache_dirty {
                let skip = app.lines.len().saturating_sub(DISPLAY_LINES);
                let tail = app.lines.iter().skip(skip);
                // Pre-calculate capacity to avoid reallocations
                let cap: usize = tail.clone().map(|l| l.len() + 1).sum();
                let mut buf = String::with_capacity(cap);
                for line in tail {
                    buf.push_str(line);
                    buf.push('\n');
                }
                app.display_cache = buf;
                app.cache_dirty = false;
            }
        }
    }
    Task::none()
}

// ── Process runner ──────────────────────────────────────────────────────────

fn run_cli_process(exe: &str, args: &[String], tx: &mpsc::Sender<Vec<u8>>) -> Result<(), String> {
    use std::process::{Command, Stdio};
    use std::os::windows::process::CommandExt;

    const CREATE_NO_WINDOW: u32 = 0x08000000;

    let mut child = Command::new(exe)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .creation_flags(CREATE_NO_WINDOW)
        .spawn()
        .map_err(|e| format!("Failed to spawn '{}': {}", exe, e))?;

    let stdout = child.stdout.take().ok_or("No stdout handle")?;
    let stderr = child.stderr.take().ok_or("No stderr handle")?;

    let tx1 = tx.clone();
    std::thread::spawn(move || {
        let mut batch = Vec::with_capacity(BATCH_BUF);
        let mut buf = [0u8; READ_BUF];
        let mut r = std::io::BufReader::with_capacity(READ_BUF, stdout);
        loop {
            match r.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    batch.extend_from_slice(&buf[..n]);
                    if batch.len() >= BATCH_BUF {
                        let _ = tx1.send(std::mem::replace(&mut batch, Vec::with_capacity(BATCH_BUF)));
                    }
                }
            }
        }
        if !batch.is_empty() { let _ = tx1.send(batch); }
    });

    let tx2 = tx.clone();
    std::thread::spawn(move || {
        let mut batch = Vec::with_capacity(BATCH_BUF);
        let mut buf = [0u8; READ_BUF];
        let mut r = std::io::BufReader::with_capacity(READ_BUF, stderr);
        loop {
            match r.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    batch.extend_from_slice(&buf[..n]);
                    if batch.len() >= BATCH_BUF {
                        let _ = tx2.send(std::mem::replace(&mut batch, Vec::with_capacity(BATCH_BUF)));
                    }
                }
            }
        }
        if !batch.is_empty() { let _ = tx2.send(batch); }
    });

    let _ = child.wait();
    Ok(())
}

fn find_cli_exe() -> String {
    let rel = ["file_renamer.exe","../file_renamer.exe","rust/target/release/file_renamer.exe","../rust/target/release/file_renamer.exe"];
    for c in &rel {
        if std::path::Path::new(c).exists() {
            return std::fs::canonicalize(c).unwrap_or_else(|_| std::path::PathBuf::from(c))
                .to_string_lossy().to_string();
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            for c in &["file_renamer.exe","rust/target/release/file_renamer.exe"] {
                let p = dir.join(c);
                if p.exists() { return p.to_string_lossy().to_string(); }
                if let Some(parent) = dir.parent() {
                    let p2 = parent.join(c);
                    if p2.exists() { return p2.to_string_lossy().to_string(); }
                }
            }
        }
    }
    "file_renamer.exe".to_string()
}

// ── Subscription ────────────────────────────────────────────────────────────

pub fn subscription(app: &App) -> Subscription<Message> {
    if app.is_running {
        time::every(Duration::from_millis(16)).map(|_| Message::Tick) // 60 fps
    } else {
        Subscription::none()
    }
}

pub fn theme(_: &App) -> Theme { crate::theme::fluent_dark() }

// ── View ────────────────────────────────────────────────────────────────────

pub fn view(app: &App) -> Element<'_, Message> {
    container(
        row![view_settings(app), view_output(app)]
            .spacing(16).padding(16)
            .width(Length::Fill).height(Length::Fill)
    )
    .width(Length::Fill).height(Length::Fill)
    .style(|_| container::Style {
        background: Some(colors::BG.into()),
        ..Default::default()
    }).into()
}

fn view_settings(app: &App) -> Element<'_, Message> {
    let dir_row = row![
        text_input("例如：D:\\Downloads", &app.directory)
            .on_input(Message::DirectoryChanged).width(Length::Fill),
        button("选择...").on_press(Message::BrowseFolder).style(btn_sec).height(36),
    ].spacing(8).align_y(iced::Alignment::Center);

    let ab: Vec<Element<'_, Message>> = ALG.iter().enumerate()
        .map(|(i, n)| button(*n).on_press(Message::AlgorithmChanged(i))
            .style(if i == app.algorithm_index { btn_acc } else { btn_tog })
            .height(32).into())
        .collect();
    let mb: Vec<Element<'_, Message>> = MOD.iter().enumerate()
        .map(|(i, (_, l))| button(*l).on_press(Message::ModeChanged(i))
            .style(if i == app.mode_index { btn_acc } else { btn_tog })
            .height(32).into())
        .collect();

    let adv = column![
        lbl("高级参数"),
        row![
            column![lbl("线程数"), text_input("留空=自动", &app.threads)
                .on_input(Message::ThreadsChanged)].spacing(4).width(Length::Fill),
            column![lbl("批大小"), text_input("留空=自动", &app.batch_size)
                .on_input(Message::BatchSizeChanged)].spacing(4).width(Length::Fill),
        ].spacing(12)
    ].spacing(4);

    let run = if app.is_running {
        button("运行").style(btn_dis).height(36)
    } else {
        button("运行").on_press(Message::Run).style(btn_suc).height(36)
    };

    let card = column![
        column![lbl("目录路径"), dir_row,
            hint("请输入本机磁盘中的文件夹路径，或使用\"选择...\"打开文件夹对话框")].spacing(4),
        column![lbl("算法"), row(ab).spacing(4)].spacing(4),
        column![lbl("模式"), row(mb).spacing(4)].spacing(4),
        column![lbl("扩展名过滤"),
            text_input("jpg,png,txt", &app.extensions)
                .on_input(Message::ExtensionsChanged)].spacing(4),
        row![
            row![checkbox::Checkbox::new(app.execute).on_toggle(Message::ToggleExecute),
                 lbl("执行重命名")].spacing(6).align_y(iced::Alignment::Center),
            row![checkbox::Checkbox::new(app.recursive).on_toggle(Message::ToggleRecursive),
                 lbl("包含子目录")].spacing(6).align_y(iced::Alignment::Center),
        ].spacing(20),
        adv,
        row![run,
            button("停止").on_press(Message::Stop).style(btn_dan).height(36),
            button("清空").on_press(Message::ClearOutput).style(btn_sec).height(36),
        ].spacing(12),
    ].spacing(16).padding(24);

    column![
        row![text("设置").size(16).color(colors::ACCENT)],
        container(card).width(Length::Fill).style(fcard)
    ].spacing(8).width(Length::Fill).into()
}

fn view_output(app: &App) -> Element<'_, Message> {
    // Only render the last DISPLAY_LINES lines — widget never sees the full log
    let content = text(&app.display_cache).size(13).color(colors::TEXT);
    let scroll = scrollable(content).height(Length::Fill).width(Length::Fill).anchor_bottom();
    let card = container(scroll).padding(16).width(Length::Fill).height(Length::Fill).style(fcard);
    let total = app.lines.len();
    let status = if total > 0 {
        format!("输出 ({} 行)", total)
    } else {
        "输出".into()
    };
    column![
        row![text(status).size(16).color(colors::ACCENT)],
        card
    ].spacing(8).width(Length::Fill).height(Length::Fill).into()
}

// ── Style helpers ───────────────────────────────────────────────────────────

fn lbl(s: &str) -> Element<'_, Message> { text(s).size(14).color(colors::TEXT_SECONDARY).into() }
fn hint(s: &str) -> Element<'_, Message> { text(s).size(12).color(colors::TEXT_DISABLED).into() }

fn fcard(_: &Theme) -> container::Style {
    container::Style {
        background: Some(colors::CARD.into()),
        border: iced::Border { width: 1.0, color: colors::BORDER, radius: 8.0.into() },
        ..Default::default()
    }
}
fn btn_suc(_: &Theme, s: button::Status) -> button::Style {
    match s {
        button::Status::Active | button::Status::Hovered => button::Style {
            background: Some(colors::BTN_SUCCESS.into()), text_color: colors::BG,
            border: iced::Border::default().rounded(4), ..Default::default()
        },
        _ => button::Style::default()
    }
}
fn btn_dan(_: &Theme, s: button::Status) -> button::Style {
    match s {
        button::Status::Active | button::Status::Hovered => button::Style {
            background: Some(colors::BTN_DANGER.into()), text_color: colors::TEXT,
            border: iced::Border::default().rounded(4), ..Default::default()
        },
        _ => button::Style::default()
    }
}
fn btn_sec(_: &Theme, s: button::Status) -> button::Style {
    match s {
        button::Status::Active | button::Status::Hovered => button::Style {
            background: Some(colors::BTN_SECONDARY.into()), text_color: colors::TEXT,
            border: iced::Border { width: 1.0, color: colors::BORDER, radius: 4.0.into() },
            ..Default::default()
        },
        _ => button::Style::default()
    }
}
fn btn_acc(_: &Theme, s: button::Status) -> button::Style {
    match s {
        button::Status::Active | button::Status::Hovered => button::Style {
            background: Some(colors::ACCENT.into()), text_color: colors::TEXT,
            border: iced::Border::default().rounded(4), ..Default::default()
        },
        _ => button::Style::default()
    }
}
fn btn_tog(_: &Theme, s: button::Status) -> button::Style {
    match s {
        button::Status::Active => button::Style {
            background: Some(colors::ELEVATED.into()), text_color: colors::TEXT_SECONDARY,
            border: iced::Border { width: 1.0, color: colors::BORDER, radius: 4.0.into() },
            ..Default::default()
        },
        button::Status::Hovered => button::Style {
            background: Some(colors::SURFACE.into()), text_color: colors::TEXT,
            border: iced::Border { width: 1.0, color: colors::BORDER_SUBTLE, radius: 4.0.into() },
            ..Default::default()
        },
        _ => button::Style::default()
    }
}
fn btn_dis(_: &Theme, _: button::Status) -> button::Style {
    button::Style {
        background: Some(Color::from_rgba8(0, 120, 212, 0.5).into()),
        text_color: Color::from_rgba8(255, 255, 255, 0.5),
        border: iced::Border::default().rounded(4), ..Default::default()
    }
}
