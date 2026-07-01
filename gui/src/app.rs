use iced::widget::{button, checkbox, column, container, row, scrollable, text, text_input};
use iced::{Color, Element, Length, Subscription, Task, Theme, time};
use std::collections::{HashSet, VecDeque};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use crossbeam_channel::{bounded, TryRecvError};
use file_renamer_lib::{Algorithm, ProcessingMode, ProcessOptions, ProgressReporter};

use crate::theme::colors;

const ALG: &[&str] = &["MD5","SHA1","SHA256","SHA512","CRC32","BLAKE2B"];
const MOD: &[(&str,&str)] = &[("multi-thread","Multi Thread (默认)"),("batch-mode","Batch Mode"),("single-thread","Single Thread")];
const DISPLAY_LINES: usize = 2000;
const LOG_DIR: &str = "logs";

#[derive(Debug, Clone)]
pub enum Message {
    DirectoryChanged(String), AlgorithmChanged(usize), ModeChanged(usize),
    ExtensionsChanged(String), ThreadsChanged(String), BatchSizeChanged(String),
    ToggleRecursive(bool), ToggleExecute(bool), BrowseFolder, FolderPicked(Option<String>),
    Run, Stop, ClearOutput, Tick, LoadFullLog,
}

pub struct App {
    directory: String, algorithm_index: usize, mode_index: usize,
    extensions: String, threads: String, batch_size: String,
    recursive: bool, execute: bool, is_running: bool,
    lines: VecDeque<String>,
    display_cache: String,
    cache_dirty: bool,
    log_writer: Option<BufWriter<File>>,
    log_path: Option<String>,
    total_lines: usize,
    rx: Option<crossbeam_channel::Receiver<String>>,
}

pub fn new() -> (App, Task<Message>) {
    (App {
        directory: String::new(), algorithm_index: 0, mode_index: 0,
        extensions: "jpg,jpeg,png".into(), threads: String::new(), batch_size: String::new(),
        recursive: false, execute: false, is_running: false,
        lines: VecDeque::new(), display_cache: String::new(), cache_dirty: false,
        log_writer: None, log_path: None, total_lines: 0, rx: None,
    }, Task::none())
}

pub fn update(app: &mut App, msg: Message) -> Task<Message> {
    match msg {
        Message::DirectoryChanged(v) => app.directory = v,
        Message::AlgorithmChanged(i) => app.algorithm_index = i.min(ALG.len()-1),
        Message::ModeChanged(i) => app.mode_index = i.min(MOD.len()-1),
        Message::ExtensionsChanged(v) => app.extensions = v,
        Message::ThreadsChanged(v) => app.threads = v,
        Message::BatchSizeChanged(v) => app.batch_size = v,
        Message::ToggleRecursive(v) => app.recursive = v,
        Message::ToggleExecute(v) => app.execute = v,
        Message::BrowseFolder => {
            return Task::perform(async {
                rfd::AsyncFileDialog::new().set_title("选择文件夹").pick_folder().await
                    .map(|h| h.path().to_string_lossy().to_string())
            }, Message::FolderPicked);
        }
        Message::FolderPicked(Some(p)) => app.directory = p,
        Message::FolderPicked(None) => {}

        Message::Run => {
            if app.is_running || app.directory.is_empty() { return Task::none(); }
            app.is_running = true;
            app.lines.clear();
            app.display_cache.clear();
            app.cache_dirty = false;
            app.total_lines = 0;
            app.lines.push_back("Starting...".into());
            app.total_lines = 1;
            rebuild_display(app);

            // Create log file
            let _ = std::fs::create_dir_all(LOG_DIR);
            let ts = timestamp_tag();
            let path = format!("{}/run_{}.log", LOG_DIR, ts);
            app.log_path = Some(path.clone());
            app.log_writer = File::create(&path).ok().map(|f| BufWriter::with_capacity(64*1024, f));

            // Build options
            let dir = PathBuf::from(&app.directory);
            let a = match ALG[app.algorithm_index] {
                "SHA1" => Algorithm::Sha1,
                "SHA256" => Algorithm::Sha256,
                "SHA512" => Algorithm::Sha512,
                "CRC32" => Algorithm::Crc32,
                "BLAKE2B" => Algorithm::Blake2b,
                _ => Algorithm::Md5,
            };
            let mode = match MOD[app.mode_index].0 {
                "single-thread" => ProcessingMode::SingleThread,
                "batch-mode" => ProcessingMode::BatchMode,
                _ => ProcessingMode::MultiThread,
            };
            let exts: HashSet<String> = if app.extensions.is_empty() {
                HashSet::new()
            } else {
                app.extensions.split(',').map(|s| {
                    let e = s.trim().to_lowercase();
                    if e.starts_with('.') { e } else { format!(".{}", e) }
                }).collect()
            };
            let threads: usize = app.threads.parse().unwrap_or_else(|_| num_cpus::get());
            let batch: usize = app.batch_size.parse().unwrap_or(0);

            let options = ProcessOptions {
                directory: dir, algorithm: a, recursive: app.recursive,
                dry_run: !app.execute, allowed_extensions: exts,
                num_threads: threads, batch_size: batch, mode,
            };

            let (tx, rx) = bounded::<String>(256);
            app.rx = Some(rx);

            std::thread::spawn(move || {
                let reporter = Arc::new(ChannelReporter { tx });
                file_renamer_lib::process_directory(options, reporter);
            });
        }

        Message::Stop => {
            app.is_running = false;
            app.rx = None;
            app.log_writer = None;
        }
        Message::ClearOutput => {
            app.lines.clear();
            app.display_cache.clear();
            app.cache_dirty = false;
            app.total_lines = 0;
        }
        Message::LoadFullLog => { load_full_log(app); }

        Message::Tick => {
            let mut texts: Vec<String> = Vec::new();
            let mut disconnected = false;
            if let Some(rx) = &app.rx {
                loop {
                    match rx.try_recv() {
                        Ok(t) => texts.push(t),
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => { disconnected = true; break; }
                    }
                }
            }

            if disconnected {
                for t in &texts {
                    if let Some(ref mut w) = app.log_writer { let _ = w.write_all(t.as_bytes()); }
                    for line in t.split('\n') {
                        if line.is_empty() { continue; }
                        app.total_lines += 1;
                        if app.lines.len() >= DISPLAY_LINES { app.lines.pop_front(); }
                        app.lines.push_back(line.to_string());
                        app.cache_dirty = true;
                    }
                }
                if let Some(ref mut w) = app.log_writer { let _ = w.flush(); }
                app.is_running = false;
                app.rx = None;
                app.log_writer = None;
                if app.cache_dirty { rebuild_display(app); }
                return Task::none();
            }

            if texts.is_empty() && !app.cache_dirty { return Task::none(); }

            if let Some(ref mut w) = app.log_writer {
                for t in &texts { let _ = w.write_all(t.as_bytes()); }
                let _ = w.flush();
            }

            for t in &texts {
                for line in t.split('\n') {
                    if line.is_empty() { continue; }
                    app.total_lines += 1;
                    if app.lines.len() >= DISPLAY_LINES { app.lines.pop_front(); }
                    app.lines.push_back(line.to_string());
                    app.cache_dirty = true;
                }
            }
            if app.cache_dirty { rebuild_display(app); }
        }
    }
    Task::none()
}

// ── Channel reporter ────────────────────────────────────────────────────────

struct ChannelReporter {
    tx: crossbeam_channel::Sender<String>,
}

impl ProgressReporter for ChannelReporter {
    fn on_output(&self, text: &str) {
        let _ = self.tx.send(text.to_string());
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn load_full_log(app: &mut App) {
    let Some(ref path) = app.log_path else { return };
    let Ok(content) = std::fs::read_to_string(path) else { return };
    app.lines.clear();
    app.total_lines = 0;
    for line in content.lines() {
        app.lines.push_back(line.to_string());
        app.total_lines += 1;
    }
    rebuild_display(app);
}

fn rebuild_display(app: &mut App) {
    let cap: usize = app.lines.iter().map(|l| l.len()+1).sum();
    let mut buf = String::with_capacity(cap);
    for line in &app.lines {
        buf.push_str(line);
        buf.push('\n');
    }
    app.display_cache = buf;
    app.cache_dirty = false;
}

fn timestamp_tag() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    let d = secs / 86400; let t = secs % 86400;
    let y = 1970u32 + (d as u32 * 4 / 1461);
    let rem = d - (((y-1970) as u64 * 1461) / 4);
    let m = (rem / 31 + 1).min(12); let day = (rem % 31 + 1) as u32;
    let h = (t / 3600) as u32; let mi = ((t % 3600) / 60) as u32; let s = (t % 60) as u32;
    format!("{:04}{:02}{:02}_{:02}{:02}{:02}", y, m, day, h, mi, s)
}

// ── Subscription ────────────────────────────────────────────────────────────

pub fn subscription(app: &App) -> Subscription<Message> {
    if app.is_running { time::every(Duration::from_millis(16)).map(|_| Message::Tick) }
    else { Subscription::none() }
}

pub fn theme(_: &App) -> Theme { crate::theme::fluent_dark() }

// ── View ────────────────────────────────────────────────────────────────────

pub fn view(app: &App) -> Element<'_, Message> {
    container(
        row![view_settings(app), view_output(app)]
            .spacing(16).padding(16)
            .width(Length::Fill).height(Length::Fill)
    ).width(Length::Fill).height(Length::Fill)
     .style(|_| container::Style{background:Some(colors::BG.into()),..Default::default()}).into()
}

fn view_settings(app: &App) -> Element<'_, Message> {
    let dir_row = row![
        text_input("例如：D:\\Downloads", &app.directory).on_input(Message::DirectoryChanged).width(Length::Fill),
        button("选择...").on_press(Message::BrowseFolder).style(btn_sec).height(36),
    ].spacing(8).align_y(iced::Alignment::Center);

    let ab: Vec<Element<'_,Message>> = ALG.iter().enumerate()
        .map(|(i,n)| button(*n).on_press(Message::AlgorithmChanged(i))
            .style(if i==app.algorithm_index{btn_acc}else{btn_tog}).height(32).into()).collect();
    let mb: Vec<Element<'_,Message>> = MOD.iter().enumerate()
        .map(|(i,(_,l))| button(*l).on_press(Message::ModeChanged(i))
            .style(if i==app.mode_index{btn_acc}else{btn_tog}).height(32).into()).collect();

    let adv = column![lbl("高级参数"),row![
        column![lbl("线程数"),text_input("留空=自动",&app.threads).on_input(Message::ThreadsChanged)].spacing(4).width(Length::Fill),
        column![lbl("批大小"),text_input("留空=自动",&app.batch_size).on_input(Message::BatchSizeChanged)].spacing(4).width(Length::Fill),
    ].spacing(12)].spacing(4);

    let run = if app.is_running{button("运行").style(btn_dis).height(36)}else{button("运行").on_press(Message::Run).style(btn_suc).height(36)};

    let card = column![
        column![lbl("目录路径"),dir_row,hint("请输入本机磁盘中的文件夹路径，或使用\"选择...\"打开文件夹对话框")].spacing(4),
        column![lbl("算法"),row(ab).spacing(4)].spacing(4),
        column![lbl("模式"),row(mb).spacing(4)].spacing(4),
        column![lbl("扩展名过滤"),text_input("jpg,png,txt",&app.extensions).on_input(Message::ExtensionsChanged)].spacing(4),
        row![row![checkbox::Checkbox::new(app.execute).on_toggle(Message::ToggleExecute),lbl("执行重命名")].spacing(6).align_y(iced::Alignment::Center),
             row![checkbox::Checkbox::new(app.recursive).on_toggle(Message::ToggleRecursive),lbl("包含子目录")].spacing(6).align_y(iced::Alignment::Center)].spacing(20),
        adv,
        row![run,button("停止").on_press(Message::Stop).style(btn_dan).height(36),button("清空").on_press(Message::ClearOutput).style(btn_sec).height(36)].spacing(12),
    ].spacing(16).padding(24);

    column![row![text("设置").size(16).color(colors::ACCENT)],container(card).width(Length::Fill).style(fcard)].spacing(8).width(Length::Fill).into()
}

fn view_output(app: &App) -> Element<'_, Message> {
    let content = text(&app.display_cache).size(13).color(colors::TEXT);
    let scroll = scrollable(content).height(Length::Fill).width(Length::Fill).anchor_bottom();
    let card = container(scroll).padding(16).width(Length::Fill).height(Length::Fill).style(fcard);

    let status = if app.total_lines > 0 { format!("输出 ({} 行)", app.total_lines) } else { "输出".into() };
    let load_btn = if app.log_path.is_some() && !app.is_running {
        row![button("加载完整日志").on_press(Message::LoadFullLog).style(btn_acc).height(28)]
    } else { row![] };

    column![
        row![text(status).size(16).color(colors::ACCENT), load_btn].spacing(8).align_y(iced::Alignment::Center),
        card
    ].spacing(8).width(Length::Fill).height(Length::Fill).into()
}

// ── Styles ──────────────────────────────────────────────────────────────────

fn lbl(s:&str)->Element<'_,Message>{text(s).size(14).color(colors::TEXT_SECONDARY).into()}
fn hint(s:&str)->Element<'_,Message>{text(s).size(12).color(colors::TEXT_DISABLED).into()}
fn fcard(_:&Theme)->container::Style{container::Style{background:Some(colors::CARD.into()),border:iced::Border{width:1.0,color:colors::BORDER,radius:8.0.into()},..Default::default()}}
fn btn_suc(_:&Theme,s:button::Status)->button::Style{match s{button::Status::Active|button::Status::Hovered=>button::Style{background:Some(colors::BTN_SUCCESS.into()),text_color:colors::BG,border:iced::Border::default().rounded(4),..Default::default()},_=>button::Style::default()}}
fn btn_dan(_:&Theme,s:button::Status)->button::Style{match s{button::Status::Active|button::Status::Hovered=>button::Style{background:Some(colors::BTN_DANGER.into()),text_color:colors::TEXT,border:iced::Border::default().rounded(4),..Default::default()},_=>button::Style::default()}}
fn btn_sec(_:&Theme,s:button::Status)->button::Style{match s{button::Status::Active|button::Status::Hovered=>button::Style{background:Some(colors::BTN_SECONDARY.into()),text_color:colors::TEXT,border:iced::Border{width:1.0,color:colors::BORDER,radius:4.0.into()},..Default::default()},_=>button::Style::default()}}
fn btn_acc(_:&Theme,s:button::Status)->button::Style{match s{button::Status::Active|button::Status::Hovered=>button::Style{background:Some(colors::ACCENT.into()),text_color:colors::TEXT,border:iced::Border::default().rounded(4),..Default::default()},_=>button::Style::default()}}
fn btn_tog(_:&Theme,s:button::Status)->button::Style{match s{button::Status::Active=>button::Style{background:Some(colors::ELEVATED.into()),text_color:colors::TEXT_SECONDARY,border:iced::Border{width:1.0,color:colors::BORDER,radius:4.0.into()},..Default::default()},button::Status::Hovered=>button::Style{background:Some(colors::SURFACE.into()),text_color:colors::TEXT,border:iced::Border{width:1.0,color:colors::BORDER_SUBTLE,radius:4.0.into()},..Default::default()},_=>button::Style::default()}}
fn btn_dis(_:&Theme,_:button::Status)->button::Style{button::Style{background:Some(Color::from_rgba8(0,120,212,0.5).into()),text_color:Color::from_rgba8(255,255,255,0.5),border:iced::Border::default().rounded(4),..Default::default()}}
