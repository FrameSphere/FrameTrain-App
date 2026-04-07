// Plugin Management Commands for Tauri Backend

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::io::{BufRead, BufReader};
use tauri::{AppHandle, Window};
use tauri::Emitter;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PluginInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub category: String,
    pub icon: String,
    pub built_in: bool,
    pub train_plugin: Option<String>,
    pub test_plugin: Option<String>,
    pub required_packages: Vec<String>,
    pub optional_packages: Vec<String>,
    pub estimated_size_mb: i32,
    pub install_time_minutes: i32,
    pub github_path: Option<String>,
    pub priority: i32,
    #[serde(default)]
    pub is_selected: bool,
    #[serde(default)]
    pub is_installed: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PluginInstallProgress {
    pub plugin_id: String,
    pub status: String,
    pub message: String,
    pub progress: Option<i32>,
}

/// Get list of available plugins for first launch selection
#[tauri::command]
pub async fn get_available_plugins(app_handle: AppHandle) -> Result<Vec<PluginInfo>, String> {
    println!("[PluginCmd] Getting available plugins...");
    
    // Get Python executable
    let python_exe = get_python_executable(&app_handle)?;
    
    // Get plugin_manager.py path
    let plugin_manager_path = get_resource_path(&app_handle, "python/plugin_manager.py")?;
    
    println!("[PluginCmd] Python: {}", python_exe);
    println!("[PluginCmd] Plugin Manager: {}", plugin_manager_path.display());
    
    // Run plugin_manager.py --first-launch
    let output = Command::new(&python_exe)
        .arg(&plugin_manager_path)
        .arg("--first-launch")
        .output()
        .map_err(|e| format!("Failed to run plugin_manager: {}", e))?;
    
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        println!("[PluginCmd] Error: {}", stderr);
        return Err(format!("Plugin manager failed: {}", stderr));
    }
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("[PluginCmd] Output: {}", stdout);
    
    // Parse JSON output - the Python script outputs JSON directly
    // Parse the entire output as JSON array
    let plugins: Vec<PluginInfo> = serde_json::from_str(&stdout)
        .map_err(|e| format!("Failed to parse plugins: {}", e))?;
    
    println!("[PluginCmd] Found {} plugins", plugins.len());
    Ok(plugins)
}

/// Check if first launch setup is needed
#[tauri::command]
pub async fn check_first_launch() -> Result<bool, String> {
    println!("[PluginCmd] Checking first launch...");
    
    // TEMP: Force first launch for testing
    // Uncomment the next line to always trigger first launch dialog:
    // return Ok(true);
    
    let settings_path = dirs::home_dir()
        .ok_or("Could not get home directory")?
        .join(".frametrain")
        .join("settings.json");
    
    if !settings_path.exists() {
        println!("[PluginCmd] No settings file found - first launch");
        return Ok(true);
    }
    
    // Read settings
    let settings_json = std::fs::read_to_string(&settings_path)
        .map_err(|e| format!("Failed to read settings: {}", e))?;
    
    let settings: serde_json::Value = serde_json::from_str(&settings_json)
        .map_err(|e| format!("Failed to parse settings: {}", e))?;
    
    let first_launch_completed = settings["first_launch_completed"].as_bool().unwrap_or(false);
    
    println!("[PluginCmd] First launch completed: {}", first_launch_completed);
    Ok(!first_launch_completed)
}

/// Install selected plugins
#[tauri::command]
pub async fn install_plugins(
    app_handle: AppHandle,
    plugin_ids: Vec<String>,
    window: Window,
) -> Result<(), String> {
    println!("[PluginCmd] Installing plugins: {:?}", plugin_ids);
    
    // Get Python executable
    let python_exe = get_python_executable(&app_handle)?;
    
    // Get plugin_manager.py path
    let plugin_manager_path = get_resource_path(&app_handle, "python/plugin_manager.py")?;
    
    // Spawn installation in background thread
    tauri::async_runtime::spawn(async move {
        for plugin_id in plugin_ids.iter() {
            println!("[PluginCmd] Installing {}", plugin_id);
            
            // Emit progress: starting installation
            let _ = window.emit(
                "plugin-install-progress",
                PluginInstallProgress {
                    plugin_id: plugin_id.clone(),
                    status: "installing_dependencies".to_string(),
                    message: format!("Installing {}...", plugin_id),
                    progress: Some(0),
                },
            );
            
            // Run plugin installation
            let mut child = match Command::new(&python_exe)
                .arg(&plugin_manager_path)
                .arg("--install")
                .arg(plugin_id)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
            {
                Ok(child) => child,
                Err(e) => {
                    eprintln!("[PluginCmd] Failed to spawn installer: {}", e);
                    let _ = window.emit(
                        "plugin-install-progress",
                        PluginInstallProgress {
                            plugin_id: plugin_id.clone(),
                            status: "failed".to_string(),
                            message: format!("Failed to start installer: {}", e),
                            progress: None,
                        },
                    );
                    continue;
                }
            };
            
            // Stream output
            if let Some(stdout) = child.stdout.take() {
                let reader = BufReader::new(stdout);
                for line in reader.lines() {
                    if let Ok(line) = line {
                        println!("[PluginCmd] {}: {}", plugin_id, line);
                        
                        // Emit progress updates
                        let _ = window.emit(
                            "plugin-install-progress",
                            PluginInstallProgress {
                                plugin_id: plugin_id.clone(),
                                status: "installing_package".to_string(),
                                message: line,
                                progress: Some(50),
                            },
                        );
                    }
                }
            }
            
            let status = child.wait().expect("Failed to wait for plugin installer");
            
            if status.success() {
                println!("[PluginCmd] ✓ {} installed successfully", plugin_id);
                let _ = window.emit(
                    "plugin-install-progress",
                    PluginInstallProgress {
                        plugin_id: plugin_id.clone(),
                        status: "complete".to_string(),
                        message: format!("{} installed successfully", plugin_id),
                        progress: Some(100),
                    },
                );
            } else {
                println!("[PluginCmd] ✗ Failed to install {}", plugin_id);
                let _ = window.emit(
                    "plugin-install-progress",
                    PluginInstallProgress {
                        plugin_id: plugin_id.clone(),
                        status: "failed".to_string(),
                        message: format!("Failed to install {}", plugin_id),
                        progress: None,
                    },
                );
            }
        }
        
        // Mark first launch as complete
        let _ = mark_first_launch_complete();
        
        // Emit completion event
        let _ = window.emit("plugin-install-complete", ());
        println!("[PluginCmd] All installations complete");
    });
    
    Ok(())
}

/// Handle plugin install approval request from Python
#[tauri::command]
pub async fn handle_plugin_approval(
    plugin_id: String,
    approved: bool,
    remember: bool,
) -> Result<(), String> {
    println!("[PluginCmd] Plugin approval: {} = {}", plugin_id, approved);
    
    // Write approval to file that Python is waiting for
    let approval_path = dirs::home_dir()
        .ok_or("Could not get home directory")?
        .join(".frametrain")
        .join(format!("approval_{}.json", plugin_id));
    
    let approval_data = serde_json::json!({
        "approved": approved,
        "remember": remember,
    });
    
    std::fs::create_dir_all(approval_path.parent().unwrap())
        .map_err(|e| format!("Failed to create directory: {}", e))?;
    
    std::fs::write(&approval_path, approval_data.to_string())
        .map_err(|e| format!("Failed to write approval: {}", e))?;
    
    Ok(())
}



// ===== Helper Functions =====

fn get_python_executable(_app_handle: &AppHandle) -> Result<String, String> {
    // Try to find Python executable
    
    // Check if running from bundled app
    if let Ok(exe_path) = std::env::current_exe() {
        let bundled_python = exe_path
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("Resources").join("python").join("bin").join("python3"));
        
        if let Some(py_path) = bundled_python {
            if py_path.exists() {
                return Ok(py_path.to_string_lossy().to_string());
            }
        }
    }
    
    // Fall back to system Python
    #[cfg(target_os = "windows")]
    return Ok("python.exe".to_string());
    
    #[cfg(not(target_os = "windows"))]
    return Ok("python3".to_string());
}

fn get_resource_path(_app_handle: &AppHandle, relative: &str) -> Result<PathBuf, String> {
    // Get resource path based on environment
    
    // Development: use src-tauri/../python/
    if cfg!(debug_assertions) {
        // In dev mode, resources are in src-tauri/python/ (not going up to parent)
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .map_err(|_| "Could not get CARGO_MANIFEST_DIR".to_string())?;
        
        let path = PathBuf::from(manifest_dir)
            .join(relative);
        
        return Ok(path);
    }
    
    // Production: use bundled resources
    std::env::current_exe()
        .ok()
        .and_then(|exe| {
            exe.parent()
                .and_then(|p| p.parent())
                .map(|p| p.join("Resources").join(relative))
        })
        .ok_or_else(|| "Could not determine resource path".to_string())
}

fn mark_first_launch_complete() -> Result<(), String> {
    let settings_path = dirs::home_dir()
        .ok_or("Could not get home directory")?
        .join(".frametrain")
        .join("settings.json");
    
    let mut settings = if settings_path.exists() {
        let json = std::fs::read_to_string(&settings_path)
            .unwrap_or_else(|_| "{}".to_string());
        serde_json::from_str(&json).unwrap_or_else(|_| serde_json::json!({}))
    } else {
        serde_json::json!({})
    };
    
    settings["first_launch_completed"] = serde_json::json!(true);
    
    std::fs::create_dir_all(settings_path.parent().unwrap())
        .map_err(|e| format!("Failed to create settings dir: {}", e))?;
    
    std::fs::write(&settings_path, serde_json::to_string_pretty(&settings).unwrap())
        .map_err(|e| format!("Failed to write settings: {}", e))?;
    
    println!("[PluginCmd] First launch marked as complete");
    Ok(())
}
