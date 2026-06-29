#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod commands;

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .manage(commands::AppState::default())
        .invoke_handler(tauri::generate_handler![
            commands::which_exe,
            commands::run_cli,
            commands::stop_cli,
            commands::browse_folder,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
