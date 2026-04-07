// Test Manager - Handles model testing and evaluation

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::io::{BufRead, BufReader};
use std::thread;
use tauri::{Emitter, Manager};
use crate::AppState;
use std::sync::{Arc, Mutex};

// ============ Data Structures ============

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestJob {
    pub id: String,
    pub model_id: String,
    pub model_name: String,
    pub version_id: String,
    pub version_name: String,
    pub dataset_id: String,
    pub dataset_name: String,
    pub status: TestStatus,
    pub created_at: String,
    pub started_at: Option<String>,
    pub completed_at: Option<String>,
    pub progress: TestProgress,
    pub results: Option<TestResults>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum TestStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Stopped,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TestProgress {
    pub current_sample: usize,
    pub total_samples: usize,
    pub progress_percent: f64,
    pub samples_per_second: f64,
    pub estimated_time_remaining: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResults {
    pub total_samples: usize,
    pub correct_predictions: usize,
    pub incorrect_predictions: usize,
    pub accuracy: f64,
    pub average_loss: Option<f64>,
    pub average_inference_time: f64,
    pub predictions: Vec<PredictionResult>,
    #[serde(default)]
    pub metrics: std::collections::HashMap<String, f64>,
    // Optional fields that might be present
    #[serde(default)]
    pub total_time: Option<f64>,
    #[serde(default)]
    pub samples_per_second: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    pub sample_id: usize,
    #[serde(default)]
    pub input_text: String,
    pub expected_output: Option<String>,
    pub predicted_output: String,
    pub is_correct: bool,
    pub loss: Option<f64>,
    #[serde(default)]
    pub confidence: Option<f64>,
    pub inference_time: f64,
    #[serde(default)]
    pub error_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestConfig {
    pub model_path: String,
    pub dataset_path: String,
    pub output_path: String,
    pub batch_size: usize,
    pub max_samples: Option<usize>,
}

// ============ Global Test State ============

pub struct TestState {
    pub current_job: Option<TestJob>,
    pub jobs_history: Vec<TestJob>,
}

impl Default for TestState {
    fn default() -> Self {
        Self {
            current_job: None,
            jobs_history: Vec::new(),
        }
    }
}

// ============ Helper Functions ============

fn get_python_path() -> String {
    println!("[Test Python] 🔍 ROBUST SEARCH: Finding best Python installation...");
    
    #[derive(Debug)]
    struct PythonCandidate {
        path: String,
        version: (u32, u32, u32),
    }
    
    let mut candidates: Vec<PythonCandidate> = Vec::new();
    
    // STRATEGY 1: Search common installation directories on macOS/Linux
    if !cfg!(target_os = "windows") {
        let search_paths = vec![
            // Homebrew (Intel)
            "/usr/local/bin",
            // Homebrew (Apple Silicon)
            "/opt/homebrew/bin",
            // Python.org installations
            "/Library/Frameworks/Python.framework/Versions/3.13/bin",
            "/Library/Frameworks/Python.framework/Versions/3.12/bin",
            "/Library/Frameworks/Python.framework/Versions/3.11/bin",
            "/Library/Frameworks/Python.framework/Versions/3.10/bin",
            "/Library/Frameworks/Python.framework/Versions/3.9/bin",
            // System Python
            "/usr/bin",
        ];
        
        for base_path in search_paths {
            // Check for python3.X and python3
            for name in &["python3.13", "python3.12", "python3.11", "python3.10", "python3.9", "python3"] {
                let full_path = format!("{}/{}", base_path, name);
                
                if let Ok(output) = Command::new(&full_path).arg("--version").output() {
                    if output.status.success() {
                        let version_str = String::from_utf8_lossy(&output.stdout);
                        if let Some(version) = parse_python_version(&version_str) {
                            println!("[Test Python] ✅ Found: {} -> v{}.{}.{}", full_path, version.0, version.1, version.2);
                            candidates.push(PythonCandidate {
                                path: full_path,
                                version,
                            });
                        }
                    }
                }
            }
        }
    }
    
    // STRATEGY 2: Use 'which' command to find python in PATH
    for cmd in &["python3", "python"] {
        if let Ok(output) = Command::new("which").arg(cmd).output() {
            if output.status.success() {
                let which_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !which_path.is_empty() {
                    if let Ok(version_output) = Command::new(&which_path).arg("--version").output() {
                        if version_output.status.success() {
                            let version_str = String::from_utf8_lossy(&version_output.stdout);
                            if let Some(version) = parse_python_version(&version_str) {
                                println!("[Test Python] ✅ Found via 'which': {} -> v{}.{}.{}", which_path, version.0, version.1, version.2);
                                candidates.push(PythonCandidate {
                                    path: which_path,
                                    version,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    
    // STRATEGY 3: Try command names directly (fallback for Windows or restricted environments)
    let cmd_names = if cfg!(target_os = "windows") {
        vec!["python", "python3", "py"]
    } else {
        vec!["python3", "python"]
    };
    
    for cmd in cmd_names {
        if let Ok(output) = Command::new(cmd).arg("--version").output() {
            if output.status.success() {
                let version_str = String::from_utf8_lossy(&output.stdout);
                if let Some(version) = parse_python_version(&version_str) {
                    println!("[Test Python] ✅ Found via command: {} -> v{}.{}.{}", cmd, version.0, version.1, version.2);
                    candidates.push(PythonCandidate {
                        path: cmd.to_string(),
                        version,
                    });
                }
            }
        }
    }
    
    // Remove duplicates (same version, different paths)
    candidates.sort_by(|a, b| b.version.cmp(&a.version));
    candidates.dedup_by(|a, b| a.version == b.version);
    
    // Select the best (highest version)
    if let Some(best) = candidates.first() {
        println!("[Test Python] 🎯 SELECTED: {} (v{}.{}.{})", best.path, best.version.0, best.version.1, best.version.2);
        println!("[Test Python] 📊 All candidates found: {:?}", candidates);
        return best.path.clone();
    }
    
    // Fallback
    println!("[Test Python] ⚠️  WARNING: No Python found! Using fallback.");
    if cfg!(target_os = "windows") {
        "python".to_string()
    } else {
        "python3".to_string()
    }
}

fn parse_python_version(version_str: &str) -> Option<(u32, u32, u32)> {
    // Parse version from strings like "Python 3.11.1"
    let parts: Vec<&str> = version_str.split_whitespace().collect();
    if parts.len() < 2 {
        return None;
    }
    
    let version_part = parts[1];
    let nums: Vec<&str> = version_part.split('.').collect();
    
    if nums.len() < 2 {
        return None;
    }
    
    let major = nums[0].parse::<u32>().ok()?;
    let minor = nums[1].parse::<u32>().ok()?;
    let patch = nums.get(2).and_then(|p| p.parse::<u32>().ok()).unwrap_or(0);
    
    Some((major, minor, patch))
}

fn get_test_engine_path(app_handle: &tauri::AppHandle) -> Result<PathBuf, String> {
    // NEW: Updated path for test_engine folder structure
    let resource_path = app_handle
        .path()
        .resource_dir()
        .map_err(|e| format!("Could not get resource dir: {}", e))?;
    
    println!("[Test Engine] Resource path: {:?}", resource_path);
    
    // Try new structure: python/test_engine/test_engine.py
    let engine_path = resource_path.join("python").join("test_engine").join("test_engine.py");
    println!("[Test Engine] Trying path 1: {:?}", engine_path);
    
    if engine_path.exists() {
        println!("[Test Engine] ✅ Found at path 1");
        return Ok(engine_path);
    }
    
    // Fallback: old structure python/test_engine.py
    let engine_path_old = resource_path.join("python").join("test_engine.py");
    println!("[Test Engine] Trying path 2: {:?}", engine_path_old);
    if engine_path_old.exists() {
        println!("[Test Engine] ✅ Found at path 2");
        return Ok(engine_path_old);
    }
    
    // Fallback: Development - relative to src-tauri
    let local_path = PathBuf::from("src-tauri/python/test_engine/test_engine.py");
    println!("[Test Engine] Trying path 3: {:?}", local_path);
    if local_path.exists() {
        println!("[Test Engine] ✅ Found at path 3");
        return Ok(local_path);
    }
    
    // Last fallback: old development path
    let local_path_old = PathBuf::from("src-tauri/python/test_engine.py");
    println!("[Test Engine] Trying path 4: {:?}", local_path_old);
    if local_path_old.exists() {
        println!("[Test Engine] ✅ Found at path 4");
        return Ok(local_path_old);
    }
    
    // ABSOLUTE PATH FALLBACK for development
    let absolute_dev_path = PathBuf::from("/Users/karol/Desktop/Laufende_Projekte/FrameTrain/desktop-app2/src-tauri/python/test_engine/test_engine.py");
    println!("[Test Engine] Trying path 5 (absolute): {:?}", absolute_dev_path);
    if absolute_dev_path.exists() {
        println!("[Test Engine] ✅ Found at path 5 (absolute dev path)");
        return Ok(absolute_dev_path);
    }
    
    Err("Test engine not found in any location".to_string())
}

fn get_models_dir(app_handle: &tauri::AppHandle) -> Result<PathBuf, String> {
    let data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Could not get app data dir: {}", e))?;
    Ok(data_dir.join("models"))
}

fn get_test_output_dir(app_handle: &tauri::AppHandle, test_id: &str) -> Result<PathBuf, String> {
    let data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Could not get app data dir: {}", e))?;
    
    let output_dir = data_dir.join("test_outputs").join(test_id);
    fs::create_dir_all(&output_dir)
        .map_err(|e| format!("Could not create output dir: {}", e))?;
    
    Ok(output_dir)
}

fn save_test_jobs(app_handle: &tauri::AppHandle, jobs: &[TestJob]) -> Result<(), String> {
    let data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Could not get app data dir: {}", e))?;
    
    let jobs_path = data_dir.join("test_jobs.json");
    let content = serde_json::to_string_pretty(jobs)
        .map_err(|e| format!("Could not serialize jobs: {}", e))?;
    
    fs::write(&jobs_path, content)
        .map_err(|e| format!("Could not save jobs: {}", e))?;
    
    Ok(())
}

fn load_test_jobs(app_handle: &tauri::AppHandle) -> Result<Vec<TestJob>, String> {
    let data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Could not get app data dir: {}", e))?;
    
    let jobs_path = data_dir.join("test_jobs.json");
    
    if !jobs_path.exists() {
        return Ok(Vec::new());
    }
    
    let content = fs::read_to_string(&jobs_path)
        .map_err(|e| format!("Could not read jobs: {}", e))?;
    
    let jobs: Vec<TestJob> = serde_json::from_str(&content)
        .map_err(|e| format!("Could not parse jobs: {}", e))?;
    
    Ok(jobs)
}

fn save_test_results_to_db(
    app_handle: &tauri::AppHandle,
    version_id: &str,
    results: &TestResults,
) -> Result<(), String> {
    let state = app_handle.state::<AppState>();
    let db = state.db.lock()
        .map_err(|e| format!("Database lock error: {}", e))?;
    
    // Serialize results to JSON
    let results_json = serde_json::to_string(results)
        .map_err(|e| format!("Failed to serialize results: {}", e))?;
    
    db.save_test_result(
        version_id,
        results.total_samples as i32,
        results.correct_predictions as i32,
        results.accuracy,
        results.average_loss.unwrap_or(0.0),
        results.average_inference_time,
        &results_json,
    ).map_err(|e| format!("Failed to save test results: {}", e))?;
    
    println!("[Test] ✅ Test results saved to database for version: {}", version_id);
    Ok(())
}

// ============ Tauri Commands ============

#[tauri::command]
pub async fn start_test(
    app_handle: tauri::AppHandle,
    model_id: String,
    model_name: String,
    version_id: String,
    version_name: String,
    dataset_id: String,
    dataset_name: String,
    batch_size: Option<usize>,
    max_samples: Option<usize>,
    state: tauri::State<'_, Arc<Mutex<TestState>>>,
) -> Result<TestJob, String> {
    let mut state_lock = state.lock().map_err(|e| format!("Lock error: {}", e))?;
    
    if state_lock.current_job.is_some() {
        return Err("Ein Test läuft bereits".to_string());
    }
    
    // Generate test ID
    let test_id = format!("test_{}", uuid::Uuid::new_v4().to_string().replace("-", "")[..12].to_string());
    
    // Get paths
    let models_dir = get_models_dir(&app_handle)?;
    
    // Get version path from database
    let db_path = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Could not get app data dir: {}", e))?
        .join("frametrain.db");
    
    let conn = rusqlite::Connection::open(&db_path)
        .map_err(|e| format!("Database error: {}", e))?;
    
    let version_path: String = conn.query_row(
        "SELECT path FROM model_versions_new WHERE id = ?1",
        [&version_id],
        |row| row.get(0),
    ).map_err(|e| format!("Version nicht gefunden: {}", e))?;
    
    let model_path = PathBuf::from(version_path);
    let dataset_path = models_dir.join(&model_id).join("datasets").join(&dataset_id);
    let output_dir = get_test_output_dir(&app_handle, &test_id)?;
    
    // Create config
    let config = TestConfig {
        model_path: model_path.to_string_lossy().to_string(),
        dataset_path: dataset_path.to_string_lossy().to_string(),
        output_path: output_dir.to_string_lossy().to_string(),
        batch_size: batch_size.unwrap_or(8),
        max_samples,
    };
    
    // Write config to file
    let config_path = output_dir.join("test_config.json");
    let config_json = serde_json::to_string_pretty(&config)
        .map_err(|e| format!("Could not serialize config: {}", e))?;
    fs::write(&config_path, &config_json)
        .map_err(|e| format!("Could not write config: {}", e))?;
    
    // Create job
    let job = TestJob {
        id: test_id.clone(),
        model_id,
        model_name,
        version_id: version_id.clone(),
        version_name,
        dataset_id,
        dataset_name,
        status: TestStatus::Pending,
        created_at: chrono::Utc::now().to_rfc3339(),
        started_at: None,
        completed_at: None,
        progress: TestProgress::default(),
        results: None,
        error: None,
    };
    
    state_lock.current_job = Some(job.clone());
    
    // Start test process in background
    let app_handle_clone = app_handle.clone();
    let config_path_str = config_path.to_string_lossy().to_string();
    let version_id_clone = version_id.clone();
    let state_clone = Arc::clone(&state);
    
    drop(state_lock);
    
    thread::spawn(move || {
        run_test_process(app_handle_clone, test_id, config_path_str, version_id_clone, state_clone);
    });
    
    Ok(job)
}

fn run_test_process(
    app_handle: tauri::AppHandle,
    test_id: String,
    config_path: String,
    version_id: String,
    state: Arc<Mutex<TestState>>,
) {
    let python = get_python_path();
    
    println!("[Test] Using Python: {}", python);
    println!("[Test] Test ID: {}", test_id);
    println!("[Test] Config: {}", config_path);
    
    let engine_path = match get_test_engine_path(&app_handle) {
        Ok(p) => {
            println!("[Test] Engine path: {:?}", p);
            p
        },
        Err(e) => {
            eprintln!("[Test] Error finding engine: {}", e);
            let _ = app_handle.emit("test-error", serde_json::json!({
                "test_id": test_id,
                "error": e
            }));
            return;
        }
    };
    
    let _ = app_handle.emit("test-started", serde_json::json!({
        "test_id": test_id
    }));
    
    println!("[Test] Starting Python process...");
    
    let mut child = match Command::new(&python)
        .arg(engine_path.to_string_lossy().to_string())
        .arg("--config")
        .arg(&config_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => {
            println!("[Test] Python process started successfully");
            c
        },
        Err(e) => {
            eprintln!("[Test] Failed to start Python: {}", e);
            let _ = app_handle.emit("test-error", serde_json::json!({
                "test_id": test_id,
                "error": format!("Failed to start Python: {}", e)
            }));
            return;
        }
    };
    
    // Read stderr in separate thread
    if let Some(stderr) = child.stderr.take() {
        let stderr_reader = BufReader::new(stderr);
        thread::spawn(move || {
            for line in stderr_reader.lines() {
                if let Ok(line) = line {
                    eprintln!("[Test STDERR] {}", line);
                }
            }
        });
    }
    
    // Read stdout for progress updates
    if let Some(stdout) = child.stdout.take() {
        let reader = BufReader::new(stdout);
        let app_handle_clone = app_handle.clone();
        let test_id_clone = test_id.clone();
        
        for line in reader.lines() {
            if let Ok(line) = line {
                println!("[Test OUTPUT] {}", line);
                
                if let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) {
                    let msg_type = msg.get("type").and_then(|t| t.as_str()).unwrap_or("");
                    
                    match msg_type {
                        "progress" => {
                            let _ = app_handle_clone.emit("test-progress", serde_json::json!({
                                "test_id": test_id_clone,
                                "data": msg.get("data")
                            }));
                        }
                        "status" => {
                            let _ = app_handle_clone.emit("test-status", serde_json::json!({
                                "test_id": test_id_clone,
                                "data": msg.get("data")
                            }));
                        }
                        "complete" => {
                            // CRITICAL FIX: Read full results from file before saving to DB
                            if let Some(data) = msg.get("data") {
                                // Check if results_file is provided
                                if let Some(results_file) = data.get("results_file").and_then(|f| f.as_str()) {
                                    println!("[Test] Reading full results from file: {}", results_file);
                                    
                                    // Read full results from file
                                    match fs::read_to_string(results_file) {
                                        Ok(file_content) => {
                                            match serde_json::from_str::<TestResults>(&file_content) {
                                                Ok(full_results) => {
                                                    // Save FULL results (with predictions) to DB
                                                    if let Err(e) = save_test_results_to_db(&app_handle_clone, &version_id, &full_results) {
                                                        eprintln!("[Test] Failed to save results to DB: {}", e);
                                                    } else {
                                                        println!("[Test] ✅ Saved {} predictions to database", full_results.predictions.len());
                                                    }
                                                }
                                                Err(e) => {
                                                    eprintln!("[Test] Failed to parse results file: {}", e);
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            eprintln!("[Test] Failed to read results file: {}", e);
                                        }
                                    }
                                } else {
                                    // Fallback: Try to parse data directly (old format)
                                    if let Ok(results) = serde_json::from_value::<TestResults>(data.clone()) {
                                        if let Err(e) = save_test_results_to_db(&app_handle_clone, &version_id, &results) {
                                            eprintln!("[Test] Failed to save results to DB: {}", e);
                                        }
                                    }
                                }
                            }
                            
                            // CRITICAL FIX: Include version_id in event payload
                            let event_data = serde_json::json!({
                                "test_id": test_id_clone,
                                "version_id": version_id.clone(),  // ADD version_id!
                                "data": msg.get("data")
                            });
                            
                            let _ = app_handle_clone.emit("test-complete", event_data);
                        }
                        "error" => {
                            let _ = app_handle_clone.emit("test-error", serde_json::json!({
                                "test_id": test_id_clone,
                                "data": msg.get("data")
                            }));
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    
    let status = child.wait();
    println!("[Test] Process finished with status: {:?}", status);
    
    // Reset state
    if let Ok(mut state_lock) = state.lock() {
        state_lock.current_job = None;
    }
    
    let _ = app_handle.emit("test-finished", serde_json::json!({
        "test_id": test_id,
        "success": status.map(|s| s.success()).unwrap_or(false)
    }));
    
    // CRITICAL: Send final done event to unlock UI
    println!("[Test] Sending test-done event");
    let _ = app_handle.emit("test-done", serde_json::json!({
        "test_id": test_id
    }));
}

#[tauri::command]
pub fn stop_test(
    state: tauri::State<'_, Arc<Mutex<TestState>>>,
) -> Result<(), String> {
    let mut state_lock = state.lock().map_err(|e| format!("Lock error: {}", e))?;
    
    if let Some(ref mut job) = state_lock.current_job {
        job.status = TestStatus::Stopped;
        job.completed_at = Some(chrono::Utc::now().to_rfc3339());
    }
    
    Ok(())
}

#[tauri::command]
pub fn get_current_test(
    state: tauri::State<'_, Arc<Mutex<TestState>>>,
) -> Result<Option<TestJob>, String> {
    let state_lock = state.lock().map_err(|e| format!("Lock error: {}", e))?;
    Ok(state_lock.current_job.clone())
}

#[tauri::command]
pub fn get_active_test_job(
    state: tauri::State<'_, Arc<Mutex<TestState>>>,
) -> Result<Option<TestJob>, String> {
    let state_lock = state.lock().map_err(|e| format!("Lock error: {}", e))?;
    
    // Return current job only if it's running or pending
    if let Some(ref job) = state_lock.current_job {
        if job.status == TestStatus::Running || job.status == TestStatus::Pending {
            return Ok(Some(job.clone()));
        }
    }
    
    Ok(None)
}

#[tauri::command]
pub fn get_test_history(
    app_handle: tauri::AppHandle,
) -> Result<Vec<TestJob>, String> {
    load_test_jobs(&app_handle)
}

#[tauri::command]
pub fn get_test_results_for_version(
    app_handle: tauri::AppHandle,
    version_id: String,
) -> Result<Vec<TestResults>, String> {
    let state = app_handle.state::<AppState>();
    let db = state.db.lock()
        .map_err(|e| format!("Database lock error: {}", e))?;
    
    let results_json = db.get_test_results_for_version(&version_id)
        .map_err(|e| format!("Query error: {}", e))?;
    
    let mut test_results = Vec::new();
    for json_str in results_json {
        if let Ok(test_result) = serde_json::from_str::<TestResults>(&json_str) {
            test_results.push(test_result);
        }
    }
    
    Ok(test_results)
}

#[tauri::command]
pub fn export_hard_examples(
    app_handle: tauri::AppHandle,
    predictions: Vec<PredictionResult>,
    format: String,
) -> Result<String, String> {
    let data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Could not get app data dir: {}", e))?;
    
    let export_dir = data_dir.join("exports");
    fs::create_dir_all(&export_dir)
        .map_err(|e| format!("Could not create export dir: {}", e))?;
    
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let filename = format!("hard_examples_{}.{}", timestamp, format);
    let export_path = export_dir.join(&filename);
    
    match format.as_str() {
        "json" => {
            let json = serde_json::to_string_pretty(&predictions)
                .map_err(|e| format!("Serialization error: {}", e))?;
            fs::write(&export_path, json)
                .map_err(|e| format!("Write error: {}", e))?;
        }
        "jsonl" => {
            let mut lines = Vec::new();
            for pred in predictions {
                let line = serde_json::to_string(&pred)
                    .map_err(|e| format!("Serialization error: {}", e))?;
                lines.push(line);
            }
            fs::write(&export_path, lines.join("\n"))
                .map_err(|e| format!("Write error: {}", e))?;
        }
        "csv" => {
            let mut csv = String::from("input,expected,predicted,is_correct,loss,confidence\n");
            for pred in predictions {
                let input = pred.input_text.replace("\"", "\"\"");
                let expected = pred.expected_output.unwrap_or_default().replace("\"", "\"\"");
                let predicted = pred.predicted_output.replace("\"", "\"\"");
                let loss = pred.loss.map(|l| l.to_string()).unwrap_or_default();
                let conf = pred.confidence.map(|c| c.to_string()).unwrap_or_default();
                csv.push_str(&format!(
                    "\"{}\",\"{}\",\"{}\",{},{},{}\n",
                    input, expected, predicted, pred.is_correct, loss, conf
                ));
            }
            fs::write(&export_path, csv)
                .map_err(|e| format!("Write error: {}", e))?;
        }
        "txt" => {
            let mut text = String::new();
            for (i, pred) in predictions.iter().enumerate() {
                text.push_str(&format!("Example {}\n", i + 1));
                text.push_str(&format!("Input: {}\n", pred.input_text));
                if let Some(expected) = &pred.expected_output {
                    text.push_str(&format!("Expected: {}\n", expected));
                }
                text.push_str(&format!("Predicted: {}\n", pred.predicted_output));
                text.push_str(&format!("Correct: {}\n", pred.is_correct));
                text.push_str("\n---\n\n");
            }
            fs::write(&export_path, text)
                .map_err(|e| format!("Write error: {}", e))?;
        }
        _ => return Err(format!("Unsupported format: {}", format)),
    }
    
    Ok(export_path.to_string_lossy().to_string())
}
