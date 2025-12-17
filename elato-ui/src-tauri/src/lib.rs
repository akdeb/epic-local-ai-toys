use tauri_plugin_shell::ShellExt;
use tauri_plugin_shell::process::CommandEvent;
use tauri::Manager;
use std::path::PathBuf;
use std::{env, fs};
use std::io::Write;
use std::net::TcpStream;
use std::sync::Mutex;
use std::time::Duration;

#[allow(dead_code)]
struct SidecarChild(Mutex<Option<tauri_plugin_shell::process::CommandChild>>);

fn ensure_port_8000_free() {
    let addr = ("127.0.0.1", 8000);

    // If something is already listening, ask it to shutdown.
    if TcpStream::connect(addr).is_ok() {
        let _ = TcpStream::connect(addr).and_then(|mut stream| {
            let req = b"POST /shutdown HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 0\r\n\r\n";
            stream.write_all(req)
        });

        // Wait a bit for the port to become free.
        for _ in 0..30 {
            std::thread::sleep(Duration::from_millis(100));
            if TcpStream::connect(addr).is_err() {
                break;
            }
        }
    }
}

fn stop_sidecar(app: &tauri::AppHandle) {
    let _ = TcpStream::connect(("127.0.0.1", 8000)).and_then(|mut stream| {
        let req = b"POST /shutdown HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 0\r\n\r\n";
        stream.write_all(req)
    });

    std::thread::sleep(Duration::from_millis(200));

    let state = app.state::<SidecarChild>();
    let child = state.0.lock().ok().and_then(|mut guard| guard.take());
    if let Some(child) = child {
        let _ = child.kill();
    }
}

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let db_path: PathBuf = match env::var("ELATO_DB_PATH") {
                Ok(v) if !v.trim().is_empty() => PathBuf::from(v),
                _ => {
                    let dir = app
                        .path()
                        .app_data_dir()
                        .expect("Failed to resolve app data directory");
                    let _ = fs::create_dir_all(&dir);
                    dir.join("elato.db")
                }
            };

            let db_path_str = db_path.to_string_lossy().to_string();

            println!("[TAURI] ELATO_DB_PATH={}", db_path_str);

            let migrations_dir: Option<PathBuf> = app
                .path()
                .resource_dir()
                .ok()
                .map(|p| p.join("migrations"))
                .filter(|p| fs::metadata(p).is_ok());

            if let Some(dir) = &migrations_dir {
                println!("[TAURI] ELATO_MIGRATIONS_DIR={}", dir.to_string_lossy());
            }

            let sidecar_command = app
                .shell()
                .sidecar("api")
                .unwrap()
                .env("ELATO_DB_PATH", db_path_str.clone())
                .arg("--db-path")
                .arg(db_path_str.clone());

            let sidecar_command = if let Some(dir) = migrations_dir {
                sidecar_command.env("ELATO_MIGRATIONS_DIR", dir.to_string_lossy().to_string())
            } else {
                sidecar_command
            };

            ensure_port_8000_free();

            let (mut rx, child) = sidecar_command
                .spawn()
                .expect("Failed to spawn sidecar");

            // Keep the child alive for the duration of the app.
            // If dropped at end of setup, the sidecar may terminate immediately.
            app.manage(SidecarChild(Mutex::new(Some(child))));

            tauri::async_runtime::spawn(async move {
                // read events such as stdout
                while let Some(event) = rx.recv().await {
                    match event {
                        CommandEvent::Stdout(line) => {
                            println!("[SIDECAR:stdout] {}", String::from_utf8_lossy(&line));
                        }
                        CommandEvent::Stderr(line) => {
                            eprintln!("[SIDECAR:stderr] {}", String::from_utf8_lossy(&line));
                        }
                        CommandEvent::Error(err) => {
                            eprintln!("[SIDECAR:error] {}", err);
                        }
                        CommandEvent::Terminated(payload) => {
                            eprintln!("[SIDECAR:terminated] {:?}", payload);
                        }
                        _ => {}
                    }
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![greet])
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|app_handle, event| match event {
        tauri::RunEvent::ExitRequested { .. } => {
            stop_sidecar(app_handle);
        }
        tauri::RunEvent::WindowEvent { event, .. } => {
            if matches!(event, tauri::WindowEvent::CloseRequested { .. }) {
                stop_sidecar(app_handle);
            }
        }
        _ => {}
    });
}
