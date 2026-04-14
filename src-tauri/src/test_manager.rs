// Test Manager - Handles model testing and evaluation
// FrameTrain v2 — Erweitert für alle Modalitäten (NLP, Vision, Audio, Detection, Tabular)
// Unterstützt Dataset-Modus (ganzen Datensatz) + Single-Modus (einzelnen Input)

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
    // Neue Felder
    #[serde(default)]
    pub task_type: String,
    #[serde(default = "default_mode")]
    pub mode: String,
}

fn default_mode() -> String { "dataset".to_string() }

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
    #[serde(default)]
    pub correct_predictions: Option<usize>,
    #[serde(default)]
    pub incorrect_predictions: Option<usize>,
    #[serde(default)]
    pub accuracy: Option<f64>,
    #[serde(default)]
    pub average_loss: Option<f64>,
    #[serde(default)]
    pub average_inference_time: f64,
    #[serde(default)]
    pub predictions: Vec<PredictionResult>,
    #[serde(default)]
    pub metrics: std::collections::HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub total_time: Option<f64>,
    #[serde(default)]
    pub samples_per_second: Option<f64>,
    #[serde(default)]
    pub task_type: String,
    #[serde(default)]
    pub hard_examples_file: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    pub sample_id: usize,
    #[serde(default)]
    pub input_text: String,
    #[serde(default)]
    pub input_path: Option<String>,
    pub expected_output: Option<String>,
    pub predicted_output: String,
    pub is_correct: bool,
    pub loss: Option<f64>,
    #[serde(default)]
    pub confidence: Option<f64>,
    pub inference_time: f64,
    #[serde(default)]
    pub error_type: Option<String>,
    // Modalitäts-spezifische Felder
    #[serde(default)]
    pub top_predictions: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub detections: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub wer: Option<f64>,
}

/// Config-Struct das an Python übergeben wird
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestConfig {
    pub model_path: String,
    pub dataset_path: String,
    pub output_path: String,
    pub batch_size: usize,
    pub max_samples: Option<usize>,
    pub task_type: String,
    pub mode: String,
    pub single_input: String,
    pub single_input_type: String,
}

/// Ergebnis eines Single-Modus Tests (modalitätsagnostisch)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingleTestResult {
    pub task_type: String,
    pub input: String,
    pub input_type: String,
    pub result: serde_json::Value,
    pub inference_time_ms: f64,
}

// ============ Global Test State ============

pub struct TestState {
    pub current_job: Option<TestJob>,
    pub jobs_history: Vec<TestJob>,
    pub stop_signal: bool,
}

impl Default for TestState {
    fn default() -> Self {
        Self {
            current_job: None,
            jobs_history: Vec::new(),
            stop_signal: false,
        }
    }
}

// ============ Helper Functions ============

fn get_python_path() -> String {
    println!("[Test Python] 🔍 Suche Python-Installation…");

    struct Candidate { path: String, version: (u32, u32, u32) }
    let mut candidates: Vec<Candidate> = Vec::new();

    if !cfg!(target_os = "windows") {
        let dirs = vec![
            "/opt/homebrew/bin", "/usr/local/bin",
            "/Library/Frameworks/Python.framework/Versions/3.13/bin",
            "/Library/Frameworks/Python.framework/Versions/3.12/bin",
            "/Library/Frameworks/Python.framework/Versions/3.11/bin",
            "/Library/Frameworks/Python.framework/Versions/3.10/bin",
            "/usr/bin",
        ];
        for base in dirs {
            for name in &["python3.13","python3.12","python3.11","python3.10","python3.9","python3"] {
                let full = format!("{}/{}", base, name);
                if let Ok(out) = Command::new(&full).arg("--version").output() {
                    if out.status.success() {
                        let vs = String::from_utf8_lossy(&out.stdout);
                        if let Some(v) = parse_python_version(&vs) {
                            println!("[Test Python] ✅ {full} → v{}.{}.{}", v.0, v.1, v.2);
                            candidates.push(Candidate { path: full, version: v });
                        }
                    }
                }
            }
        }
    }

    for cmd in &["python3", "python"] {
        if let Ok(out) = Command::new("which").arg(cmd).output() {
            let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !p.is_empty() {
                if let Ok(vo) = Command::new(&p).arg("--version").output() {
                    let vs = String::from_utf8_lossy(&vo.stdout);
                    if let Some(v) = parse_python_version(&vs) {
                        candidates.push(Candidate { path: p, version: v });
                    }
                }
            }
        }
    }

    candidates.sort_by(|a, b| b.version.cmp(&a.version));
    candidates.dedup_by(|a, b| a.version == b.version);

    if let Some(best) = candidates.first() {
        println!("[Test Python] 🎯 Gewählt: {} (v{}.{}.{})", best.path, best.version.0, best.version.1, best.version.2);
        return best.path.clone();
    }

    if cfg!(target_os = "windows") { "python".to_string() } else { "python3".to_string() }
}

fn parse_python_version(s: &str) -> Option<(u32, u32, u32)> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() < 2 { return None; }
    let nums: Vec<&str> = parts[1].split('.').collect();
    if nums.len() < 2 { return None; }
    let major = nums[0].parse::<u32>().ok()?;
    let minor = nums[1].parse::<u32>().ok()?;
    let patch = nums.get(2).and_then(|p| p.trim_end_matches(|c: char| !c.is_ascii_digit()).parse::<u32>().ok()).unwrap_or(0);
    Some((major, minor, patch))
}

fn get_test_engine_path(app_handle: &tauri::AppHandle) -> Result<PathBuf, String> {
    let try_paths: Vec<PathBuf> = vec![
        // Resourcen-Verzeichnis (Produktion)
        app_handle.path().resource_dir()
            .map(|p| p.join("python").join("test_engine").join("test_engine.py"))
            .unwrap_or_default(),
        // Entwicklung – relativ
        PathBuf::from("src-tauri/python/test_engine/test_engine.py"),
        // Entwicklung – absolut
        PathBuf::from("/Users/karol/Desktop/Laufende_Projekte/FrameTrain/desktop-app2/src-tauri/python/test_engine/test_engine.py"),
    ];

    for p in try_paths {
        if p.exists() {
            println!("[Test Engine] ✅ Gefunden: {:?}", p);
            return Ok(p);
        }
    }
    Err("Test-Engine nicht gefunden".to_string())
}

fn get_models_dir(app_handle: &tauri::AppHandle) -> Result<PathBuf, String> {
    let data = app_handle.path().app_data_dir()
        .map_err(|e| format!("AppDataDir Fehler: {}", e))?;
    Ok(data.join("models"))
}

fn get_test_output_dir(app_handle: &tauri::AppHandle, test_id: &str) -> Result<PathBuf, String> {
    let data = app_handle.path().app_data_dir()
        .map_err(|e| format!("AppDataDir Fehler: {}", e))?;
    let dir = data.join("test_outputs").join(test_id);
    fs::create_dir_all(&dir).map_err(|e| format!("Output-Ordner Fehler: {}", e))?;
    Ok(dir)
}

fn get_version_path(app_handle: &tauri::AppHandle, version_id: &str) -> Result<String, String> {
    let db_path = app_handle.path().app_data_dir()
        .map_err(|e| format!("AppDataDir Fehler: {}", e))?
        .join("frametrain.db");

    let conn = rusqlite::Connection::open(&db_path)
        .map_err(|e| format!("Datenbank-Fehler: {}", e))?;

    conn.query_row(
        "SELECT path FROM model_versions_new WHERE id = ?1",
        [version_id],
        |row| row.get(0),
    ).map_err(|e| format!("Version nicht gefunden: {}", e))
}

fn get_task_type_for_version(app_handle: &tauri::AppHandle, version_id: &str) -> String {
    let db_path = match app_handle.path().app_data_dir() {
        Ok(p) => p.join("frametrain.db"),
        Err(_) => return "auto".to_string(),
    };

    let conn = match rusqlite::Connection::open(&db_path) {
        Ok(c) => c,
        Err(_) => return "auto".to_string(),
    };

    // Versuche task_type aus der Trainings-Config zu lesen
    let result: Result<String, _> = conn.query_row(
        "SELECT training_config FROM model_versions_new WHERE id = ?1",
        [version_id],
        |row| row.get::<_, String>(0),
    );

    if let Ok(config_json) = result {
        if let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&config_json) {
            if let Some(tt) = cfg.get("task_type").and_then(|v| v.as_str()) {
                return tt.to_string();
            }
        }
    }
    "auto".to_string()
}

fn save_test_results_to_db(
    app_handle: &tauri::AppHandle,
    version_id: &str,
    results: &TestResults,
) -> Result<(), String> {
    let state = app_handle.state::<AppState>();
    let db = state.db.lock()
        .map_err(|e| format!("DB-Lock Fehler: {}", e))?;

    let results_json = serde_json::to_string(results)
        .map_err(|e| format!("Serialisierungs-Fehler: {}", e))?;

    db.save_test_result(
        version_id,
        results.total_samples as i32,
        results.correct_predictions.unwrap_or(0) as i32,
        results.accuracy.unwrap_or(0.0),
        results.average_loss.unwrap_or(0.0),
        results.average_inference_time,
        &results_json,
    ).map_err(|e| format!("DB-Speicher-Fehler: {}", e))?;

    println!("[Test] ✅ Ergebnisse in DB gespeichert für Version: {}", version_id);
    Ok(())
}

// ============ Tauri Commands ============

/// Startet einen Dataset-Test (ganzen Datensatz durchlaufen)
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
    let mut state_lock = state.lock().map_err(|e| format!("Lock Fehler: {}", e))?;
    if state_lock.current_job.is_some() {
        return Err("Ein Test läuft bereits".to_string());
    }
    state_lock.stop_signal = false;

    let test_id = format!(
        "test_{}",
        &uuid::Uuid::new_v4().to_string().replace("-", "")[..12]
    );

    let model_path = get_version_path(&app_handle, &version_id)?;
    let task_type = get_task_type_for_version(&app_handle, &version_id);
    let models_dir = get_models_dir(&app_handle)?;
    let dataset_path = models_dir.join(&model_id).join("datasets").join(&dataset_id);
    let output_dir = get_test_output_dir(&app_handle, &test_id)?;

    let config = TestConfig {
        model_path,
        dataset_path: dataset_path.to_string_lossy().to_string(),
        output_path: output_dir.to_string_lossy().to_string(),
        batch_size: batch_size.unwrap_or(8),
        max_samples,
        task_type: task_type.clone(),
        mode: "dataset".to_string(),
        single_input: String::new(),
        single_input_type: "text".to_string(),
    };

    let config_path = output_dir.join("test_config.json");
    let config_json = serde_json::to_string_pretty(&config)
        .map_err(|e| format!("Config-Serialisierung: {}", e))?;
    fs::write(&config_path, &config_json)
        .map_err(|e| format!("Config schreiben: {}", e))?;

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
        task_type: task_type.clone(),
        mode: "dataset".to_string(),
    };

    state_lock.current_job = Some(job.clone());
    drop(state_lock);

    let app_clone = app_handle.clone();
    let config_path_str = config_path.to_string_lossy().to_string();
    let version_id_clone = version_id.clone();
    let state_clone = Arc::clone(&state);

    thread::spawn(move || {
        run_test_process(
            app_clone, test_id, config_path_str,
            version_id_clone, state_clone, false,
        );
    });

    Ok(job)
}

/// Testet einen einzelnen Input (Text, Bildpfad, Audiopfad, JSON)
#[tauri::command]
pub async fn test_single_input(
    app_handle: tauri::AppHandle,
    version_id: String,
    single_input: String,
    single_input_type: String,  // "text" | "image_path" | "audio_path" | "json"
    state: tauri::State<'_, Arc<Mutex<TestState>>>,
) -> Result<String, String> {
    // Single-Tests blockieren keinen Dataset-Test – sie laufen parallel
    let test_id = format!(
        "single_{}",
        &uuid::Uuid::new_v4().to_string().replace("-", "")[..8]
    );

    let model_path = get_version_path(&app_handle, &version_id)?;
    let task_type = get_task_type_for_version(&app_handle, &version_id);
    let output_dir = get_test_output_dir(&app_handle, &test_id)?;

    let config = TestConfig {
        model_path,
        dataset_path: String::new(),
        output_path: output_dir.to_string_lossy().to_string(),
        batch_size: 1,
        max_samples: Some(1),
        task_type: task_type.clone(),
        mode: "single".to_string(),
        single_input: single_input.clone(),
        single_input_type: single_input_type.clone(),
    };

    let config_path = output_dir.join("test_config.json");
    fs::write(&config_path, serde_json::to_string_pretty(&config)
        .map_err(|e| format!("Config-Fehler: {}", e))?)
        .map_err(|e| format!("Config schreiben: {}", e))?;

    let app_clone = app_handle.clone();
    let config_path_str = config_path.to_string_lossy().to_string();
    let test_id_clone = test_id.clone();
    let state_clone = Arc::clone(&state);

    // Single-Test in eigenem Thread, Ergebnis via Event
    thread::spawn(move || {
        run_test_process(
            app_clone, test_id_clone, config_path_str,
            version_id, state_clone, true,
        );
    });

    Ok(test_id)
}

// ============ Process Runner ============

fn run_test_process(
    app_handle: tauri::AppHandle,
    test_id: String,
    config_path: String,
    version_id: String,
    state: Arc<Mutex<TestState>>,
    is_single: bool,
) {
    let python = get_python_path();

    let engine_path = match get_test_engine_path(&app_handle) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[Test] Engine nicht gefunden: {}", e);
            let _ = app_handle.emit("test-error", serde_json::json!({
                "test_id": test_id,
                "error": e,
            }));
            return;
        }
    };

    println!("[Test] Python: {} | Engine: {:?}", python, engine_path);

    let mut child = match Command::new(&python)
        .arg(engine_path.to_string_lossy().to_string())
        .arg("--config")
        .arg(&config_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            let _ = app_handle.emit("test-error", serde_json::json!({
                "test_id": test_id,
                "error": format!("Python konnte nicht gestartet werden: {}", e),
            }));
            return;
        }
    };

    // stderr → Logs
    if let Some(stderr) = child.stderr.take() {
        thread::spawn(move || {
            for line in BufReader::new(stderr).lines().flatten() {
                eprintln!("[Test STDERR] {}", line);
            }
        });
    }

    // stdout → Events
    if let Some(stdout) = child.stdout.take() {
        let ah = app_handle.clone();
        let tid = test_id.clone();
        let vid = version_id.clone();

        for line in BufReader::new(stdout).lines().flatten() {
            // Stop-Signal prüfen
            if let Ok(s) = state.lock() {
                if s.stop_signal && !is_single {
                    let _ = child.kill();
                    break;
                }
            }

            println!("[Test OUTPUT] {}", line);

            let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) else { continue };
            let msg_type = msg.get("type").and_then(|t| t.as_str()).unwrap_or("");

            match msg_type {
                "progress" => {
                    let _ = ah.emit("test-progress", serde_json::json!({
                        "test_id": tid,
                        "data": msg.get("data"),
                    }));
                }
                "status" => {
                    let _ = ah.emit("test-status", serde_json::json!({
                        "test_id": tid,
                        "data": msg.get("data"),
                    }));
                }
                "complete" => {
                    let data = msg.get("data");
                    let mode = data
                        .and_then(|d| d.get("mode"))
                        .and_then(|m| m.as_str())
                        .unwrap_or("dataset");

                    if mode == "single" || is_single {
                        // Single-Modus: Ergebnis direkt ans Frontend
                        let _ = ah.emit("test-single-complete", serde_json::json!({
                            "test_id": tid,
                            "data": data,
                        }));
                    } else {
                        // Dataset-Modus: Ergebnisse aus Datei lesen + in DB speichern
                        if let Some(d) = data {
                            if let Some(results_file) = d.get("results_file").and_then(|f| f.as_str()) {
                                println!("[Test] Lese Ergebnisse aus: {}", results_file);
                                match fs::read_to_string(results_file) {
                                    Ok(content) => {
                                        match serde_json::from_str::<TestResults>(&content) {
                                            Ok(full_results) => {
                                                if let Err(e) = save_test_results_to_db(&ah, &vid, &full_results) {
                                                    eprintln!("[Test] DB-Fehler: {}", e);
                                                } else {
                                                    println!("[Test] ✅ {} Predictions gespeichert", full_results.predictions.len());
                                                }
                                            }
                                            Err(e) => eprintln!("[Test] Parse-Fehler: {}", e),
                                        }
                                    }
                                    Err(e) => eprintln!("[Test] Datei-Fehler: {}", e),
                                }
                            }
                        }

                        let _ = ah.emit("test-complete", serde_json::json!({
                            "test_id": tid,
                            "version_id": vid,
                            "data": data,
                        }));
                    }
                }
                "error" => {
                    let _ = ah.emit("test-error", serde_json::json!({
                        "test_id": tid,
                        "data": msg.get("data"),
                        "error": msg.get("data")
                            .and_then(|d| d.get("error"))
                            .and_then(|e| e.as_str())
                            .unwrap_or("Unbekannter Fehler"),
                    }));
                }
                "warning" => {
                    let _ = ah.emit("test-status", serde_json::json!({
                        "test_id": tid,
                        "data": {"status": "warning", "message": msg.get("data")},
                    }));
                }
                _ => {}
            }
        }
    }

    let status = child.wait();
    println!("[Test] Prozess beendet: {:?}", status);

    if !is_single {
        if let Ok(mut s) = state.lock() {
            s.current_job = None;
            s.stop_signal = false;
        }
    }

    let _ = app_handle.emit("test-finished", serde_json::json!({
        "test_id": test_id,
        "is_single": is_single,
        "success": status.map(|s| s.success()).unwrap_or(false),
    }));
    let _ = app_handle.emit("test-done", serde_json::json!({ "test_id": test_id }));
}

// ============ Weitere Commands ============

#[tauri::command]
pub fn stop_test(
    state: tauri::State<'_, Arc<Mutex<TestState>>>,
) -> Result<(), String> {
    let mut s = state.lock().map_err(|e| format!("Lock Fehler: {}", e))?;
    s.stop_signal = true;
    if let Some(ref mut job) = s.current_job {
        job.status = TestStatus::Stopped;
        job.completed_at = Some(chrono::Utc::now().to_rfc3339());
    }
    Ok(())
}

#[tauri::command]
pub fn get_active_test_job(
    state: tauri::State<'_, Arc<Mutex<TestState>>>,
) -> Result<Option<TestJob>, String> {
    let s = state.lock().map_err(|e| format!("Lock Fehler: {}", e))?;
    if let Some(ref job) = s.current_job {
        if job.status == TestStatus::Running || job.status == TestStatus::Pending {
            return Ok(Some(job.clone()));
        }
    }
    Ok(None)
}

#[tauri::command]
pub fn get_current_test(
    state: tauri::State<'_, Arc<Mutex<TestState>>>,
) -> Result<Option<TestJob>, String> {
    let s = state.lock().map_err(|e| format!("Lock Fehler: {}", e))?;
    Ok(s.current_job.clone())
}

#[tauri::command]
pub fn get_test_history(
    state: tauri::State<'_, Arc<Mutex<TestState>>>,
) -> Result<Vec<TestJob>, String> {
    let s = state.lock().map_err(|e| format!("Lock Fehler: {}", e))?;
    Ok(s.jobs_history.clone())
}

#[tauri::command]
pub fn get_test_results_for_version(
    app_handle: tauri::AppHandle,
    version_id: String,
) -> Result<Vec<TestResults>, String> {
    let state = app_handle.state::<AppState>();
    let db = state.db.lock()
        .map_err(|e| format!("DB-Lock Fehler: {}", e))?;

    let results_json = db.get_test_results_for_version(&version_id)
        .map_err(|e| format!("DB-Fehler: {}", e))?;

    let mut out = Vec::new();
    for json_str in results_json {
        if let Ok(r) = serde_json::from_str::<TestResults>(&json_str) {
            out.push(r);
        }
    }
    Ok(out)
}

#[tauri::command]
pub fn export_hard_examples(
    app_handle: tauri::AppHandle,
    predictions: Vec<PredictionResult>,
    format: String,
) -> Result<String, String> {
    let data_dir = app_handle.path().app_data_dir()
        .map_err(|e| format!("AppDataDir: {}", e))?;
    let export_dir = data_dir.join("exports");
    fs::create_dir_all(&export_dir)
        .map_err(|e| format!("Export-Ordner: {}", e))?;

    let ts = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let filename = format!("hard_examples_{}.{}", ts, format);
    let path = export_dir.join(&filename);

    match format.as_str() {
        "json" => {
            let j = serde_json::to_string_pretty(&predictions)
                .map_err(|e| format!("Serialisierung: {}", e))?;
            fs::write(&path, j).map_err(|e| format!("Schreiben: {}", e))?;
        }
        "jsonl" => {
            let mut lines = Vec::new();
            for p in &predictions {
                lines.push(serde_json::to_string(p).map_err(|e| format!("Serialisierung: {}", e))?);
            }
            fs::write(&path, lines.join("\n")).map_err(|e| format!("Schreiben: {}", e))?;
        }
        "csv" => {
            let mut csv = String::from("input,expected,predicted,is_correct,confidence,inference_time_ms\n");
            for p in &predictions {
                let input = p.input_text.replace('"', "\"\"");
                let exp = p.expected_output.clone().unwrap_or_default().replace('"', "\"\"");
                let pred = p.predicted_output.replace('"', "\"\"");
                let conf = p.confidence.map(|c| format!("{:.4}", c)).unwrap_or_default();
                csv.push_str(&format!(
                    "\"{}\",\"{}\",\"{}\",{},{},{:.1}\n",
                    input, exp, pred, p.is_correct, conf, p.inference_time * 1000.0
                ));
            }
            fs::write(&path, csv).map_err(|e| format!("Schreiben: {}", e))?;
        }
        "txt" => {
            let mut text = String::new();
            for (i, p) in predictions.iter().enumerate() {
                text.push_str(&format!("── Example {} ──\n", i + 1));
                text.push_str(&format!("Input:     {}\n", p.input_text));
                if let Some(exp) = &p.expected_output {
                    text.push_str(&format!("Expected:  {}\n", exp));
                }
                text.push_str(&format!("Predicted: {}\n", p.predicted_output));
                text.push_str(&format!("Correct:   {}\n", p.is_correct));
                if let Some(conf) = p.confidence {
                    text.push_str(&format!("Confidence:{:.2}%\n", conf * 100.0));
                }
                text.push_str(&format!("Time:      {:.0}ms\n\n", p.inference_time * 1000.0));
            }
            fs::write(&path, text).map_err(|e| format!("Schreiben: {}", e))?;
        }
        _ => return Err(format!("Unbekanntes Format: {}", format)),
    }

    Ok(path.to_string_lossy().to_string())
}
