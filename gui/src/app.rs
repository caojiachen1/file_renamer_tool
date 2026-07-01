use iced::widget::{button, checkbox, column, container, row, scrollable, text, text_input};
use iced::{Color, Element, Length, Subscription, Task, Theme, time};
use std::io::Read;
use std::sync::mpsc;
use std::time::Duration;

use crate::theme::colors;

// ── Raw-byte ring buffer (zero-copy between reader thread and UI) ───────────

struct RingBuffer {
    data: Vec<u8>,
    head: usize,
    len: usize,
    cap: usize,
}

impl RingBuffer {
    fn new(cap: usize) -> Self {
        Self { data: vec![0u8; cap], head: 0, len: 0, cap }
    }

    fn push(&mut self, chunk: &[u8]) {
        let n = chunk.len().min(self.cap);
        if n == 0 { return; }
        let st = (self.head + self.len) % self.cap;
        if st + n <= self.cap {
            self.data[st..st + n].copy_from_slice(&chunk[..n]);
        } else {
            let f = self.cap - st;
            self.data[st..].copy_from_slice(&chunk[..f]);
            self.data[..n - f].copy_from_slice(&chunk[f..]);
        }
        if self.len + n > self.cap {
            self.head = (self.head + (self.len + n - self.cap)) % self.cap;
            self.len = self.cap;
        } else {
            self.len += n;
        }
    }

    /// Drain all buffered bytes, returning owned Vec<u8>.
    fn drain_all(&mut self) -> Vec<u8> {
        if self.len == 0 { return Vec::new(); }
        let st = self.head;
        let take = self.len;
        self.len = 0;
        self.head = 0;
        let mut out = Vec::with_capacity(take);
        if st + take <= self.cap {
            out.extend_from_slice(&self.data[st..st + take]);
        } else {
            let f = self.cap - st;
            out.extend_from_slice(&self.data[st..]);
            out.extend_from_slice(&self.data[..take - f]);
        }
        out
    }

    fn clear(&mut self) { self.head = 0; self.len = 0; }
}

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
    ring: RingBuffer,
    output_text: String,
    line_count: usize,
    rx: Option<mpsc::Receiver<Vec<u8>>>,
}

const ALG: &[&str] = &["MD5","SHA1","SHA256","SHA512","CRC32","BLAKE2B"];
const MOD: &[(&str, &str)] = &[
    ("multi-thread", "Multi Thread (默认)"),
    ("batch-mode",   "Batch Mode"),
    ("single-thread","Single Thread"),
];

const RING_CAP: usize = 4 * 1024 * 1024; // 4 MB ring
const MAX_LINES: usize = 4000;            // keep last N lines in view
const READ_BUF: usize = 256 * 1024;       // 256 KB read buffer (fewer syscalls)

pub fn new() -> (App, Task<Message>) {
    (App {
        directory: String::new(), algorithm_index: 0, mode_index: 0,
        extensions: "jpg,jpeg,png".into(), threads: String::new(), batch_size: String::new(),
        recursive: false, execute: false, is_running: false,
        ring: RingBuffer::new(RING_CAP), output_text: String::new(), line_count: 0,
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
            app.ring.clear();
            app.output_text.clear();
            app.line_count = 0;
            app.output_text.push_str("Starting...\n");
            app.line_count += 1;

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
        Message::ClearOutput => { app.ring.clear(); app.output_text.clear(); app.line_count = 0; }

        Message::Tick => {
            // 1. Bulk-drain ring buffer — one allocation per tick
            if let Some(rx) = &app.rx {
                loop {
                    match rx.try_recv() {
                        Ok(chunk) => app.ring.push(&chunk),
                        Err(mpsc::TryRecvError::Empty) => break,
                        Err(mpsc::TryRecvError::Disconnected) => { app.is_running = false; break; }
                    }
                }
            }

            // 2. Extract raw bytes from ring, decode as UTF-8 (lossy), split into lines
            let raw = app.ring.drain_all();
            if raw.is_empty() { return Task::none(); }

            let decoded = String::from_utf8_lossy(&raw);
            for line in decoded.split('\n') {
                if line.is_empty() { continue; }
                app.output_text.push_str(line);
                app.output_text.push('\n');
                app.line_count += 1;
            }

            // 3. Trim to MAX_LINES — drop oldest complete lines from front
            while app.line_count > MAX_LINES {
                if let Some(pos) = app.output_text.find('\n') {
                    app.output_text.drain(..pos + 1);
                    app.line_count -= 1;
                } else { break; }
            }
        }
    }
    Task::none()
}

// ── Process runner (direct spawn, no bat file) ──────────────────────────────

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
    let t1 = std::thread::spawn(move || {
        let mut r = std::io::BufReader::with_capacity(READ_BUF, stdout);
        let mut buf = vec![0u8; READ_BUF];
        loop {
            match r.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => { let _ = tx1.send(buf[..n].to_vec()); }
            }
        }
    });

    let tx2 = tx.clone();
    let t2 = std::thread::spawn(move || {
        let mut r = std::io::BufReader::with_capacity(READ_BUF, stderr);
        let mut buf = vec![0u8; READ_BUF];
        loop {
            match r.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => { let _ = tx2.send(buf[..n].to_vec()); }
            }
        }
    });

    let _ = t1.join();
    let _ = t2.join();
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

// ── Subscription (30 fps tick while running) ────────────────────────────────

pub fn subscription(app: &App) -> Subscription<Message> {
    if app.is_running {
        time::every(Duration::from_millis(32)).map(|_| Message::Tick)
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
    let content = text(&app.output_text).size(13).color(colors::TEXT);
    let scroll = scrollable(content).height(Length::Fill).width(Length::Fill).anchor_bottom();
    let card = container(scroll).padding(16).width(Length::Fill).height(Length::Fill).style(fcard);
    column![
        row![text("输出").size(16).color(colors::ACCENT)],
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
