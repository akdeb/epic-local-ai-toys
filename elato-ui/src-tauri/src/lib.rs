use tauri_plugin_shell::ShellExt;
use tauri_plugin_shell::process::CommandEvent;

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let sidecar_command = app.shell().sidecar("api").unwrap()
                .env("ELATO_DB_PATH", "/Users/akashdeepdeb/Desktop/neutts-openai-api/elato.db"); // Point to dev DB for now

            let (mut rx, mut _child) = sidecar_command
                .spawn()
                .expect("Failed to spawn sidecar");

            tauri::async_runtime::spawn(async move {
                // read events such as stdout
                while let Some(event) = rx.recv().await {
                    if let CommandEvent::Stdout(line) = event {
                        println!("[SIDECAR] {}", String::from_utf8_lossy(&line));
                    }
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![greet])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
