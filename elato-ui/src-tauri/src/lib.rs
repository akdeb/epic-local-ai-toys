use tauri::{Manager, AppHandle, Emitter};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Command, Child, Stdio};
use std::sync::Mutex;
use std::time::Duration;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::io::Write;
use std::ffi::OsStr;

mod python_setup;

struct ApiProcess(Mutex<Option<Child>>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetupStatus {
    pub python_installed: bool,
    pub python_version: Option<String>,
    pub python_path: Option<String>,
    pub venv_exists: bool,
    pub venv_path: Option<String>,
    pub deps_installed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub model_type: String,
    pub repo_id: String,
    pub downloaded: bool,
    pub size_estimate: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatus {
    pub models: Vec<ModelInfo>,
    pub all_downloaded: bool,
}

fn get_elato_dir(app: &AppHandle) -> PathBuf {
    app.path()
        .app_data_dir()
        .expect("Failed to resolve app data directory")
}

fn get_venv_path(app: &AppHandle) -> PathBuf {
    get_elato_dir(app).join("python_env")
}

fn get_venv_python(app: &AppHandle) -> PathBuf {
    let venv = get_venv_path(app);
    if cfg!(target_os = "windows") {
        venv.join("Scripts").join("python.exe")
    } else {
        venv.join("bin").join("python")
    }
}

fn get_venv_pip(app: &AppHandle) -> PathBuf {
    let venv = get_venv_path(app);
    if cfg!(target_os = "windows") {
        venv.join("Scripts").join("pip.exe")
    } else {
        venv.join("bin").join("pip")
    }
}

 fn get_dir_size(path: &PathBuf) -> u64 {
     let mut total_size = 0;
     if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.filter_map(Result::ok) {
            let entry_path = entry.path();
            if entry_path.is_file() {
                if let Ok(metadata) = entry.metadata() {
                    total_size += metadata.len();
                }
            } else if entry_path.is_dir() {
                 total_size += get_dir_size(&entry_path);
             }
         }
     }
     total_size
 }

 fn format_size(bytes: u64) -> String {
     const KB: u64 = 1024;
     const MB: u64 = KB * 1024;
     const GB: u64 = MB * 1024;

     if bytes >= GB {
         format!("{:.2} GB", bytes as f64 / GB as f64)
     } else if bytes >= MB {
         format!("{:.2} MB", bytes as f64 / MB as f64)
     } else if bytes >= KB {
         format!("{:.2} KB", bytes as f64 / KB as f64)
     } else {
         format!("{} B", bytes)
     }
 }

 fn guess_model_type(repo_id: &str) -> String {
     let lower = repo_id.to_lowercase();
     if lower.contains("whisper") || lower.contains("stt") {
         "stt".to_string()
     } else if lower.contains("tts") || lower.contains("chatterbox") {
         "tts".to_string()
     } else {
         "llm".to_string()
     }
 }

fn ensure_port_free(port: u16) {
    let addr = ("127.0.0.1", port);

    // If something is already listening, try to shut it down
    if TcpStream::connect(addr).is_ok() {
        // Try graceful shutdown first for API
        if port == 8000 {
            let _ = TcpStream::connect(addr).and_then(|mut stream| {
                let req = b"POST /shutdown HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 0\r\n\r\n";
                stream.write_all(req)
            });
            std::thread::sleep(Duration::from_millis(500));
        }

        // Force kill using lsof on macOS/Linux
        if cfg!(unix) {
            let _ = Command::new("sh")
                .arg("-c")
                .arg(format!("lsof -ti:{} | xargs kill -9", port))
                .output();
        }

        // Wait for port to become free
        for _ in 0..30 {
            std::thread::sleep(Duration::from_millis(100));
            if TcpStream::connect(addr).is_err() {
                break;
            }
        }
    }
}

fn stop_api_server(app: &tauri::AppHandle) {
    // Try graceful shutdown via HTTP
    let _ = TcpStream::connect(("127.0.0.1", 8000)).and_then(|mut stream| {
        let req = b"POST /shutdown HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 0\r\n\r\n";
        stream.write_all(req)
    });

    std::thread::sleep(Duration::from_millis(200));

    // Kill the managed process
    if let Some(state) = app.try_state::<ApiProcess>() {
        if let Ok(mut guard) = state.0.lock() {
            if let Some(mut child) = guard.take() {
                let _ = child.kill();
            }
        }
    }

    // Force kill port just in case
    if cfg!(unix) {
        let _ = Command::new("sh")
            .arg("-c")
            .arg("lsof -ti:8000 | xargs kill -9")
            .output();
    }
}

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[tauri::command]
async fn check_setup_status(app: AppHandle) -> Result<SetupStatus, String> {
    let venv_path = get_venv_path(&app);
    let venv_python = get_venv_python(&app);
    let venv_exists = venv_python.exists();

    // Check system Python
    let python_check = Command::new("python3")
        .arg("--version")
        .output();

    let (python_installed, python_version, python_path) = match python_check {
        Ok(output) if output.status.success() => {
            let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let path_output = Command::new("which")
                .arg("python3")
                .output()
                .ok()
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
            (true, Some(version), path_output)
        }
        _ => (false, None, None),
    };

    // Check if deps are installed in venv
    let deps_installed = if venv_exists {
        let check = Command::new(venv_python.to_str().unwrap())
            .arg("-c")
            .arg("import mlx; import mlx_audio; import fastapi; import uvicorn; import serial; import esptool")
            .output();
        check.map(|o| o.status.success()).unwrap_or(false)
    } else {
        false
    };

    Ok(SetupStatus {
        python_installed,
        python_version,
        python_path,
        venv_exists,
        venv_path: if venv_exists { Some(venv_path.to_string_lossy().to_string()) } else { None },
        deps_installed,
    })
}

#[tauri::command]
async fn create_python_venv(app: AppHandle) -> Result<String, String> {
    let venv_path = get_venv_path(&app);
    let venv_python = get_venv_python(&app);

    // 1. Check if a valid venv already exists (has python binary)
    if venv_python.exists() {
        app.emit("setup-progress", "Virtual environment already exists...").ok();
        return Ok(venv_path.to_string_lossy().to_string());
    }

    // 2. If path exists (but invalid venv), clean it out
    if venv_path.exists() || fs::symlink_metadata(&venv_path).is_ok() {
        app.emit("setup-progress", "Cleaning up existing invalid environment...").ok();
        if venv_path.is_dir() {
            fs::remove_dir_all(&venv_path).map_err(|e| e.to_string())?;
        } else {
            fs::remove_file(&venv_path).map_err(|e| e.to_string())?;
        }
    }
    
    // Create parent directory if needed
    if let Some(parent) = venv_path.parent() {
        fs::create_dir_all(parent).map_err(|e: std::io::Error| e.to_string())?;
    }

    app.emit("setup-progress", "Creating Python virtual environment...").ok();

    let output = Command::new("python3")
        .arg("-m")
        .arg("venv")
        .arg("--clear")
        .arg(&venv_path)
        .output()
        .map_err(|e| format!("Failed to create venv: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "Failed to create venv: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    Ok(venv_path.to_string_lossy().to_string())
}

#[tauri::command]
async fn install_python_deps(app: AppHandle) -> Result<String, String> {
    let pip = get_venv_pip(&app);

    app.emit("setup-progress", "Installing Python dependencies (this may take a few minutes)...").ok();

    let result = python_setup::install_python_deps(&app, pip)?;
    app.emit("setup-progress", "Dependencies installed successfully!").ok();
    Ok(result)
}

#[tauri::command]
async fn check_models_status(_app: AppHandle) -> Result<ModelStatus, String> {
    let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let hf_cache = PathBuf::from(&home).join(".cache").join("huggingface").join("hub");

    let mut models = vec![
        ModelInfo {
            id: "stt".to_string(),
            name: "Whisper Large V3 Turbo".to_string(),
            model_type: "stt".to_string(),
            repo_id: "mlx-community/whisper-large-v3-turbo".to_string(),
            downloaded: false,
            size_estimate: None,
        },
        ModelInfo {
            id: "llm".to_string(),
            name: "Qwen 2.5 0.5B Instruct (4-bit)".to_string(),
            model_type: "llm".to_string(),
            repo_id: "mlx-community/Qwen2.5-0.5B-Instruct-4bit".to_string(),
            downloaded: false,
            size_estimate: None,
        },
        ModelInfo {
            id: "tts".to_string(),
            name: "Chatterbox TTS Turbo (4-bit)".to_string(),
            model_type: "tts".to_string(),
            repo_id: "mlx-community/chatterbox-turbo-4bit".to_string(),
            downloaded: false,
            size_estimate: None,
        },
    ];

    for model in &mut models {
        if let Some(path) = get_model_path(&hf_cache, &model.repo_id) {
            model.downloaded = true;
            let size = get_dir_size(&path);
            model.size_estimate = Some(format_size(size));
        }
    }

    let all_downloaded = models.iter().all(|m| m.downloaded);

    Ok(ModelStatus {
        models,
        all_downloaded,
    })
}

#[tauri::command]
async fn scan_local_models(_app: AppHandle) -> Result<Vec<ModelInfo>, String> {
    let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let hf_cache = PathBuf::from(&home).join(".cache").join("huggingface").join("hub");
    
    let mut models = Vec::new();

    if let Ok(entries) = fs::read_dir(&hf_cache) {
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            if !path.is_dir() { continue; }
            
            if let Some(name) = path.file_name().and_then(|n: &OsStr| n.to_str()) {
                if name.starts_with("models--") {
                    // Parse repo_id from folder name: models--org--repo
                    let without_prefix = name.trim_start_matches("models--");
                    if let Some((org, repo)) = without_prefix.split_once("--") {
                        let repo_id = format!("{}/{}", org, repo);
                        let repo_name = repo.to_string();
                        
                        let model_type = guess_model_type(&repo_id);
                        
                        // Calculate size
                        let size_str = if let Some(model_path) = get_model_path(&hf_cache, &repo_id) {
                            let size = get_dir_size(&model_path);
                            Some(format_size(size))
                        } else {
                            None
                        };

                        // Only add if it seems to be downloaded (has size)
                        if let Some(size) = size_str {
                            models.push(ModelInfo {
                                id: repo_id.clone(),
                                name: repo_name,
                                model_type,
                                repo_id,
                                downloaded: true,
                                size_estimate: Some(size),
                            });
                        }
                    }
                }
            }
        }
    }

    // Sort by name
    models.sort_by(|a, b| a.name.cmp(&b.name));
    
    Ok(models)
}

fn get_model_path(hf_cache: &PathBuf, repo_id: &str) -> Option<PathBuf> {
    let cache_name = format!("models--{}", repo_id.replace('/', "--"));
    let model_dir = hf_cache.join(&cache_name).join("snapshots");
    
    if model_dir.exists() {
        // Check if there's at least one snapshot directory with files
        if let Ok(entries) = fs::read_dir(&model_dir) {
            for entry in entries.filter_map(Result::ok) {
                if entry.path().is_dir() {
                    // Return the first snapshot dir found
                    return Some(entry.path());
                }
            }
         }
     }
     None
 }

#[tauri::command]
async fn download_model(app: AppHandle, repo_id: String) -> Result<String, String> {
    let venv_python = get_venv_python(&app);
    
    if !venv_python.exists() {
        return Err("Python environment not set up. Please complete setup first.".to_string());
    }

    app.emit("model-download-progress", format!("Downloading {}...", repo_id)).ok();

    // Use huggingface_hub to download the model
    let script = format!(
        r#"from huggingface_hub import snapshot_download; snapshot_download(repo_id="{}")"#,
        repo_id
    );

    let output = Command::new(venv_python.to_str().unwrap())
        .arg("-c")
        .arg(&script)
        .output()
        .map_err(|e| format!("Failed to download model: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "Failed to download model: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    app.emit("model-download-progress", format!("Downloaded {} successfully!", repo_id)).ok();
    Ok(format!("Model {} downloaded successfully", repo_id))
}

#[tauri::command]
async fn download_all_models(app: AppHandle) -> Result<String, String> {
    let models = vec![
        "mlx-community/whisper-large-v3-turbo",
        "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "mlx-community/chatterbox-turbo-4bit",
    ];

    for repo_id in models {
        download_model(app.clone(), repo_id.to_string()).await?;
    }

    Ok("All models downloaded successfully".to_string())
}

#[tauri::command]
async fn get_setup_complete(app: AppHandle) -> Result<bool, String> {
    let status = check_setup_status(app.clone()).await?;
    let models = check_models_status(app).await?;
    
    Ok(status.deps_installed && models.all_downloaded)
}

#[tauri::command]
async fn mark_setup_complete(app: AppHandle) -> Result<(), String> {
    let elato_dir = get_elato_dir(&app);
    let marker_file = elato_dir.join(".setup_complete");
    fs::create_dir_all(&elato_dir).map_err(|e: std::io::Error| e.to_string())?;
    fs::write(&marker_file, "1").map_err(|e: std::io::Error| e.to_string())?;
    Ok(())
}

#[tauri::command]
async fn is_first_launch(app: AppHandle) -> Result<bool, String> {
    let elato_dir = get_elato_dir(&app);
    let marker_file = elato_dir.join(".setup_complete");
    Ok(!marker_file.exists())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            ensure_port_free(8000);

            // Get paths
            let venv_python = get_venv_python(&app.handle());
            
            // In dev mode, resource_dir points to target/debug which doesn't have python
            // We need to find the actual resources/python folder
            let python_dir = {
                let resource_dir = app.path().resource_dir().ok();
                let bundled_path = resource_dir.as_ref().map(|r| r.join("python"));
                
                // Check if bundled path exists (production), otherwise use dev path
                if bundled_path.as_ref().map(|p| p.exists()).unwrap_or(false) {
                    bundled_path.unwrap()
                } else {
                    // Dev mode: go up from src-tauri to find resources/python
                    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
                    manifest_dir.parent().unwrap().parent().unwrap().join("resources").join("python")
                }
            };

            // Determine which python to use
            let python_path = if venv_python.exists() {
                venv_python
            } else {
                PathBuf::from("python3")
            };

            println!("[TAURI] Starting Python API server...");
            println!("[TAURI] Python: {:?}", python_path);
            println!("[TAURI] Server dir: {:?}", python_dir);

            let elato_db_path = get_elato_dir(&app.handle()).join("elato.db");
            println!("[TAURI] DB Path: {:?}", elato_db_path);

            // Spawn the Python server with inherited stdio so we can see output
            let child = Command::new(&python_path)
                .arg("-m")
                .arg("uvicorn")
                .arg("server:app")
                .arg("--host")
                .arg("127.0.0.1")
                .arg("--port")
                .arg("8000")
                .current_dir(&python_dir)
                .env("ELATO_DB_PATH", elato_db_path.to_string_lossy().to_string())
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit())
                .spawn();

            match child {
                Ok(child) => {
                    println!("[TAURI] Python API server started (PID: {})", child.id());
                    app.manage(ApiProcess(Mutex::new(Some(child))));
                }
                Err(e) => {
                    eprintln!("[TAURI] Failed to start Python API server: {}", e);
                }
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            greet,
            check_setup_status,
            create_python_venv,
            install_python_deps,
            check_models_status,
            scan_local_models,
            download_model,
            download_all_models,
            get_setup_complete,
            mark_setup_complete,
            is_first_launch
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|app_handle, event| match event {
        tauri::RunEvent::ExitRequested { .. } => {
            stop_api_server(app_handle);
        }
        tauri::RunEvent::WindowEvent { event, .. } => {
            if matches!(event, tauri::WindowEvent::CloseRequested { .. }) {
                stop_api_server(app_handle);
            }
        }
        _ => {}
    });
}
