// test_manager.rs – Sequenzklassifikations-Test-Engine

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::io::{BufRead, BufReader};
use std::thread;
use tauri::{Emitter, Manager};
use crate::AppState;
use std::sync::{Arc, Mutex};

// ============ Typen ============

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
    #[serde(default = "default_task_type")] pub task_type: String,
    #[serde(default = "default_mode")]      pub mode: String,
}
fn default_task_type() -> String { "seq_classification".to_string() }
fn default_mode()      -> String { "dataset".to_string() }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum TestStatus { Pending, Running, Completed, Failed, Stopped }

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
    #[serde(default)] pub correct_predictions: Option<usize>,
    #[serde(default)] pub incorrect_predictions: Option<usize>,
    #[serde(default)] pub accuracy: Option<f64>,
    #[serde(default)] pub average_loss: Option<f64>,
    #[serde(default)] pub average_inference_time: f64,
    #[serde(default)] pub predictions: Vec<PredictionResult>,
    #[serde(default)] pub metrics: std::collections::HashMap<String, serde_json::Value>,
    #[serde(default)] pub total_time: Option<f64>,
    #[serde(default)] pub samples_per_second: Option<f64>,
    #[serde(default)] pub task_type: String,
    #[serde(default)] pub hard_examples_file: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    pub sample_id: usize,
    #[serde(default)] pub input_text: String,
    #[serde(default)] pub input_path: Option<String>,
    pub expected_output: Option<String>,
    pub predicted_output: String,
    pub is_correct: bool,
    pub loss: Option<f64>,
    #[serde(default)] pub confidence: Option<f64>,
    pub inference_time: f64,
    #[serde(default)] pub error_type: Option<String>,
    #[serde(default)] pub top_predictions: Option<Vec<serde_json::Value>>,
    #[serde(default)] pub detections: Option<Vec<serde_json::Value>>,
    #[serde(default)] pub wer: Option<f64>,
}

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

// ============ State ============

pub struct TestState {
    pub current_job: Option<TestJob>,
    pub jobs_history: Vec<TestJob>,
    pub stop_signal: bool,
}
impl Default for TestState {
    fn default() -> Self { Self { current_job: None, jobs_history: Vec::new(), stop_signal: false } }
}

// ============ Hilfsfunktionen ============

fn get_python_path() -> String {
    struct C { path: String, version: (u32,u32,u32) }
    let mut candidates: Vec<C> = Vec::new();

    if !cfg!(target_os = "windows") {
        for base in &["/opt/homebrew/bin","/usr/local/bin","/usr/bin"] {
            for name in &["python3.13","python3.12","python3.11","python3.10","python3.9","python3"] {
                let full = format!("{}/{}", base, name);
                if let Ok(out) = Command::new(&full).arg("--version").output() {
                    if out.status.success() {
                        let vs = String::from_utf8_lossy(&out.stdout);
                        if let Some(v) = parse_version(&vs) { candidates.push(C { path: full, version: v }); }
                    }
                }
            }
        }
    }
    for cmd in &["python3","python"] {
        if let Ok(out) = Command::new(cmd).arg("--version").output() {
            if out.status.success() {
                let vs = String::from_utf8_lossy(&out.stdout);
                if let Some(v) = parse_version(&vs) { candidates.push(C { path: cmd.to_string(), version: v }); }
            }
        }
    }
    candidates.sort_by(|a,b| b.version.cmp(&a.version));
    candidates.dedup_by(|a,b| a.version == b.version);
    for c in &candidates {
        let ok = Command::new(&c.path).args(["-c","import torch"]).output()
            .map(|o| o.status.success()).unwrap_or(false);
        if ok { return c.path.clone(); }
    }
    candidates.first().map(|c| c.path.clone())
        .unwrap_or_else(|| if cfg!(target_os="windows") { "python".to_string() } else { "python3".to_string() })
}

fn parse_version(s: &str) -> Option<(u32,u32,u32)> {
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
    let candidates = vec![
        app_handle.path().resource_dir().ok().map(|p| p.join("python").join("test_engine").join("test_engine.py")),
        Some(PathBuf::from("src-tauri/python/test_engine/test_engine.py")),
        Some(PathBuf::from("/Users/karol/Desktop/Laufende_Projekte/FrameTrain/desktop-app/src-tauri/python/test_engine/test_engine.py")),
    ];
    for p in candidates.into_iter().flatten() {
        if p.exists() { println!("[Test Engine] ✅ {:?}", p); return Ok(p); }
    }
    Err("Test-Engine nicht gefunden".to_string())
}

fn get_models_dir(app_handle: &tauri::AppHandle) -> Result<PathBuf, String> {
    app_handle.path().app_data_dir().map(|p| p.join("models")).map_err(|e| format!("AppDataDir: {}", e))
}

fn get_test_output_dir(app_handle: &tauri::AppHandle, id: &str) -> Result<PathBuf, String> {
    let dir = app_handle.path().app_data_dir().map_err(|e| format!("AppDataDir: {}", e))?
        .join("test_outputs").join(id);
    fs::create_dir_all(&dir).map_err(|e| format!("Output-Dir: {}", e))?;
    Ok(dir)
}

fn get_version_path(app_handle: &tauri::AppHandle, version_id: &str) -> Result<String, String> {
    let db_path = app_handle.path().app_data_dir().map_err(|e| format!("AppDataDir: {}", e))?
        .join("frametrain.db");
    let conn = rusqlite::Connection::open(&db_path).map_err(|e| format!("DB: {}", e))?;
    conn.query_row("SELECT path FROM model_versions_new WHERE id = ?1", [version_id], |r| r.get(0))
        .map_err(|e| format!("Version nicht gefunden: {}", e))
}

fn save_test_results(app_handle: &tauri::AppHandle, version_id: &str, results: &TestResults) -> Result<(), String> {
    let state = app_handle.state::<AppState>();
    let db = state.db.lock().map_err(|e| format!("DB Lock: {}", e))?;
    let json = serde_json::to_string(results).map_err(|e| format!("JSON: {}", e))?;
    db.save_test_result(
        version_id, results.total_samples as i32,
        results.correct_predictions.unwrap_or(0) as i32,
        results.accuracy.unwrap_or(0.0),
        results.average_loss.unwrap_or(0.0),
        results.average_inference_time, &json,
    ).map(|_| ()).map_err(|e| format!("DB speichern: {}", e))
}

// ============ Commands ============

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
    let mut sl = state.lock().map_err(|e| format!("Lock: {}", e))?;
    if sl.current_job.is_some() { return Err("Ein Test läuft bereits".to_string()); }
    sl.stop_signal = false;

    let test_id = format!("test_{}", &uuid::Uuid::new_v4().to_string().replace("-","")[..12]);
    let model_path = get_version_path(&app_handle, &version_id)?;

    // Dataset-Pfad aus DB
    let models_dir = get_models_dir(&app_handle)?;
    let dataset_path = {
        let db_path = app_handle.path().app_data_dir().map_err(|e| format!("AppDataDir: {}", e))?.join("frametrain.db");
        let conn = rusqlite::Connection::open(&db_path).map_err(|e| format!("DB: {}", e))?;
        let res: Result<String,_> = conn.query_row("SELECT file_path FROM datasets WHERE id = ?1", [&dataset_id], |r| r.get(0));
        match res {
            Ok(p) if !p.is_empty() => PathBuf::from(p),
            _ => models_dir.join(&model_id).join("datasets").join(&dataset_id),
        }
    };

    let output_dir = get_test_output_dir(&app_handle, &test_id)?;
    let config = TestConfig {
        model_path,
        dataset_path: dataset_path.to_string_lossy().to_string(),
        output_path:  output_dir.to_string_lossy().to_string(),
        batch_size:   batch_size.unwrap_or(16),
        max_samples,
        task_type:    "seq_classification".to_string(),
        mode:         "dataset".to_string(),
        single_input: String::new(),
        single_input_type: "text".to_string(),
    };

    let config_path = output_dir.join("test_config.json");
    fs::write(&config_path, serde_json::to_string_pretty(&config).map_err(|e| format!("JSON: {}", e))?)
        .map_err(|e| format!("Config: {}", e))?;

    let job = TestJob {
        id: test_id.clone(), model_id, model_name, version_id: version_id.clone(),
        version_name, dataset_id, dataset_name,
        status: TestStatus::Pending, created_at: chrono::Utc::now().to_rfc3339(),
        started_at: None, completed_at: None, progress: TestProgress::default(),
        results: None, error: None,
        task_type: "seq_classification".to_string(), mode: "dataset".to_string(),
    };

    sl.current_job = Some(job.clone());
    drop(sl);

    let ah = app_handle.clone();
    let cfg_str = config_path.to_string_lossy().to_string();
    let vid = version_id.clone();
    let sc = Arc::clone(&state);

    thread::spawn(move || { run_test(ah, test_id, cfg_str, vid, sc, false); });
    Ok(job)
}

#[tauri::command]
pub async fn test_single_input(
    app_handle: tauri::AppHandle,
    version_id: String,
    single_input: String,
    single_input_type: String,
    state: tauri::State<'_, Arc<Mutex<TestState>>>,
) -> Result<String, String> {
    let test_id = format!("single_{}", &uuid::Uuid::new_v4().to_string().replace("-","")[..8]);
    let model_path = get_version_path(&app_handle, &version_id)?;
    let output_dir = get_test_output_dir(&app_handle, &test_id)?;

    let config = TestConfig {
        model_path,
        dataset_path: String::new(),
        output_path:  output_dir.to_string_lossy().to_string(),
        batch_size:   1, max_samples: Some(1),
        task_type:    "seq_classification".to_string(),
        mode:         "single".to_string(),
        single_input: single_input.clone(),
        single_input_type: single_input_type.clone(),
    };

    let config_path = output_dir.join("test_config.json");
    fs::write(&config_path, serde_json::to_string_pretty(&config).map_err(|e| format!("JSON: {}", e))?)
        .map_err(|e| format!("Config: {}", e))?;

    let ah = app_handle.clone();
    let cfg_str = config_path.to_string_lossy().to_string();
    let tid = test_id.clone();
    let vid = version_id.clone();
    let sc = Arc::clone(&state);

    thread::spawn(move || { run_test(ah, tid, cfg_str, vid, sc, true); });
    Ok(test_id)
}

fn run_test(
    app_handle: tauri::AppHandle, test_id: String, config_path: String,
    version_id: String, state: Arc<Mutex<TestState>>, is_single: bool,
) {
    let python = get_python_path();
    let engine_path = match get_test_engine_path(&app_handle) {
        Ok(p) => p,
        Err(e) => {
            let _ = app_handle.emit("test-error", serde_json::json!({"test_id":test_id,"error":e}));
            return;
        }
    };

    let mut child = match Command::new(&python)
        .arg(engine_path.to_string_lossy().to_string())
        .arg("--config").arg(&config_path)
        .stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()
    {
        Ok(c) => c,
        Err(e) => {
            let _ = app_handle.emit("test-error", serde_json::json!({"test_id":test_id,"error":format!("Python: {}",e)}));
            return;
        }
    };

    if let Some(stderr) = child.stderr.take() {
        thread::spawn(move || { for l in BufReader::new(stderr).lines().flatten() { eprintln!("[Test STDERR] {}", l); } });
    }

    if let Some(stdout) = child.stdout.take() {
        let ah = app_handle.clone();
        let tid = test_id.clone();
        let vid = version_id.clone();

        for line in BufReader::new(stdout).lines().flatten() {
            if let Ok(s) = state.lock() {
                if s.stop_signal && !is_single { let _ = child.kill(); break; }
            }
            println!("[Test] {}", line);
            let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) else { continue };
            let typ = msg.get("type").and_then(|t| t.as_str()).unwrap_or("");

            match typ {
                "progress" => { let _ = ah.emit("test-progress", serde_json::json!({"test_id":tid,"data":msg.get("data")})); }
                "status"   => { let _ = ah.emit("test-status",   serde_json::json!({"test_id":tid,"data":msg.get("data")})); }
                "complete" => {
                    let data = msg.get("data");
                    let mode = data.and_then(|d| d.get("mode")).and_then(|m| m.as_str()).unwrap_or("dataset");

                    if mode == "single" || is_single {
                        let _ = ah.emit("test-single-complete", serde_json::json!({"test_id":tid,"data":data}));
                    } else if let Some(d) = data {
                        // Ergebnisse aus results_file lesen
                        if let Some(rf) = d.get("results_file").and_then(|f| f.as_str()) {
                            if let Ok(content) = fs::read_to_string(rf) {
                                if let Ok(fj) = serde_json::from_str::<serde_json::Value>(&content) {
                                    let metrics = fj.get("metrics").cloned().unwrap_or_default();
                                    let preds_raw = fj.get("predictions").and_then(|p| p.as_array()).cloned().unwrap_or_default();
                                    let mut preds: Vec<PredictionResult> = Vec::new();
                                    for p in &preds_raw {
                                        if let Ok(pr) = serde_json::from_value::<PredictionResult>(p.clone()) { preds.push(pr); }
                                    }
                                    let total    = d.get("total_samples").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                                    let acc      = d.get("accuracy").and_then(|v| v.as_f64()).or_else(|| metrics.get("accuracy").and_then(|v| v.as_f64()));
                                    let correct  = d.get("correct_predictions").and_then(|v| v.as_u64()).map(|v| v as usize);
                                    let avg_loss = d.get("average_loss").and_then(|v| v.as_f64());
                                    let avg_time = d.get("average_inference_time").and_then(|v| v.as_f64()).unwrap_or(0.0);
                                    let sps      = d.get("samples_per_second").and_then(|v| v.as_f64());
                                    let mut mm: std::collections::HashMap<String, serde_json::Value> = Default::default();
                                    if let Some(obj) = metrics.as_object() { for (k,v) in obj { mm.insert(k.clone(), v.clone()); } }

                                    let full = TestResults {
                                        total_samples: total,
                                        correct_predictions: correct,
                                        incorrect_predictions: correct.map(|c| total.saturating_sub(c)),
                                        accuracy: acc, average_loss: avg_loss,
                                        average_inference_time: avg_time, predictions: preds,
                                        metrics: mm, total_time: metrics.get("total_time").and_then(|v| v.as_f64()),
                                        samples_per_second: sps,
                                        task_type: "seq_classification".to_string(),
                                        hard_examples_file: d.get("hard_examples_file").and_then(|v| v.as_str()).map(|s| s.to_string()),
                                    };
                                    if let Err(e) = save_test_results(&ah, &vid, &full) {
                                        eprintln!("[Test] DB: {}", e);
                                    }
                                }
                            }
                        }
                        let _ = ah.emit("test-complete", serde_json::json!({"test_id":tid,"version_id":vid,"data":data}));
                    }
                }
                "error" => { let _ = ah.emit("test-error", serde_json::json!({"test_id":tid,"data":msg.get("data")})); }
                _ => {}
            }
        }
    }

    let status = child.wait();
    if !is_single {
        if let Ok(mut s) = state.lock() { s.current_job = None; s.stop_signal = false; }
    }
    let _ = app_handle.emit("test-finished", serde_json::json!({"test_id":test_id,"is_single":is_single,"success":status.map(|s| s.success()).unwrap_or(false)}));
    let _ = app_handle.emit("test-done", serde_json::json!({"test_id":test_id}));
}

// ============ Weitere Commands ============

#[tauri::command]
pub fn stop_test(state: tauri::State<'_, Arc<Mutex<TestState>>>) -> Result<(), String> {
    let mut s = state.lock().map_err(|e| format!("Lock: {}", e))?;
    s.stop_signal = true;
    if let Some(ref mut job) = s.current_job {
        job.status = TestStatus::Stopped;
        job.completed_at = Some(chrono::Utc::now().to_rfc3339());
    }
    Ok(())
}

#[tauri::command]
pub fn get_active_test_job(state: tauri::State<'_, Arc<Mutex<TestState>>>) -> Result<Option<TestJob>, String> {
    let s = state.lock().map_err(|e| format!("Lock: {}", e))?;
    if let Some(ref job) = s.current_job {
        if job.status == TestStatus::Running || job.status == TestStatus::Pending { return Ok(Some(job.clone())); }
    }
    Ok(None)
}

#[tauri::command]
pub fn get_current_test(state: tauri::State<'_, Arc<Mutex<TestState>>>) -> Result<Option<TestJob>, String> {
    Ok(state.lock().map_err(|e| format!("Lock: {}", e))?.current_job.clone())
}

#[tauri::command]
pub fn get_test_history(state: tauri::State<'_, Arc<Mutex<TestState>>>) -> Result<Vec<TestJob>, String> {
    Ok(state.lock().map_err(|e| format!("Lock: {}", e))?.jobs_history.clone())
}

#[tauri::command]
pub fn get_test_results_for_version(app_handle: tauri::AppHandle, version_id: String) -> Result<Vec<TestResults>, String> {
    let state = app_handle.state::<AppState>();
    let db = state.db.lock().map_err(|e| format!("DB Lock: {}", e))?;
    let raw = db.get_test_results_for_version(&version_id).map_err(|e| format!("DB: {}", e))?;
    let mut out = Vec::new();
    for s in raw { if let Ok(r) = serde_json::from_str::<TestResults>(&s) { out.push(r); } }
    Ok(out)
}

#[tauri::command]
pub fn export_hard_examples(app_handle: tauri::AppHandle, predictions: Vec<PredictionResult>, format: String) -> Result<String, String> {
    let dir = app_handle.path().app_data_dir().map_err(|e| format!("AppDataDir: {}", e))?.join("exports");
    fs::create_dir_all(&dir).map_err(|e| format!("Export-Dir: {}", e))?;
    let ts = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let path = dir.join(format!("hard_examples_{}.{}", ts, format));
    match format.as_str() {
        "json"  => fs::write(&path, serde_json::to_string_pretty(&predictions).map_err(|e| format!("JSON: {}", e))?).map_err(|e| format!("Write: {}", e))?,
        "jsonl" => {
            let lines: Result<Vec<String>,_> = predictions.iter().map(|p| serde_json::to_string(p)).collect();
            fs::write(&path, lines.map_err(|e| format!("JSON: {}", e))?.join("\n")).map_err(|e| format!("Write: {}", e))?;
        }
        "csv" => {
            let mut csv = "input,expected,predicted,is_correct,confidence,label_id\n".to_string();
            for p in &predictions {
                let i = p.input_text.replace('"',"\"\"");
                let e = p.expected_output.clone().unwrap_or_default().replace('"',"\"\"");
                let pr = p.predicted_output.replace('"',"\"\"");
                let conf = p.confidence.map(|c| format!("{:.4}",c)).unwrap_or_default();
                csv.push_str(&format!("\"{}\",\"{}\",\"{}\",{},{}\n", i, e, pr, p.is_correct, conf));
            }
            fs::write(&path, csv).map_err(|e| format!("Write: {}", e))?;
        }
        _ => return Err(format!("Unbekanntes Format: {}", format)),
    }
    Ok(path.to_string_lossy().to_string())
}
