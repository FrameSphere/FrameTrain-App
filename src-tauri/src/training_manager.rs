use std::fs;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::io::{BufRead, BufReader};
use std::thread;
use serde::{Deserialize, Serialize};
use tauri::{Emitter, Manager};
use chrono::{DateTime, Utc};
use std::sync::Mutex as StdMutex;

// ============ Typen ============

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum TrainingStatus {
    Pending, Running, Completed, Failed, Stopped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingJob {
    pub id: String,
    pub model_id: String,
    pub model_name: String,
    pub dataset_id: String,
    pub dataset_name: String,
    pub status: TrainingStatus,
    pub config: TrainingConfig,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub progress: TrainingProgress,
    pub output_path: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainingProgress {
    pub epoch: u32,
    pub total_epochs: u32,
    pub step: u32,
    pub total_steps: u32,
    pub train_loss: f64,
    pub val_loss: Option<f64>,
    pub learning_rate: f64,
    pub progress_percent: f64,
    pub metrics: std::collections::HashMap<String, f64>,
}

/// TrainingConfig – spiegelt alle Felder aus dem TypeScript-Frontend 1:1 wider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    #[serde(default)] pub model_path: String,
    #[serde(default)] pub dataset_path: String,
    #[serde(default)] pub output_path: String,
    #[serde(default)] pub checkpoint_dir: String,

    #[serde(default = "default_epochs")]       pub epochs: u32,
    #[serde(default = "default_batch_size")]   pub batch_size: u32,
    #[serde(default = "default_one")]          pub gradient_accumulation_steps: u32,
    #[serde(default = "default_minus_one")]    pub max_steps: i32,

    #[serde(default = "default_lr")]           pub learning_rate: f64,
    #[serde(default = "default_weight_decay")] pub weight_decay: f64,
    #[serde(default)]                          pub warmup_steps: u32,
    #[serde(default)]                          pub warmup_ratio: f64,

    #[serde(default = "default_optimizer")]    pub optimizer: String,
    #[serde(default = "default_beta1")]        pub adam_beta1: f64,
    #[serde(default = "default_beta2")]        pub adam_beta2: f64,
    #[serde(default = "default_epsilon")]      pub adam_epsilon: f64,
    #[serde(default = "default_momentum")]     pub sgd_momentum: f64,

    #[serde(default = "default_scheduler")]    pub scheduler: String,
    #[serde(default = "default_one")]          pub scheduler_step_size: u32,
    #[serde(default = "default_gamma")]        pub scheduler_gamma: f64,
    #[serde(default)]                          pub cosine_min_lr: f64,

    #[serde(default = "default_dropout")]      pub dropout: f64,
    #[serde(default = "default_grad_norm")]    pub max_grad_norm: f64,
    #[serde(default)]                          pub label_smoothing: f64,

    #[serde(default)] pub fp16: bool,
    #[serde(default)] pub bf16: bool,

    #[serde(default)]                         pub use_lora: bool,
    #[serde(default = "default_lora_r")]      pub lora_r: u32,
    #[serde(default = "default_lora_alpha")]  pub lora_alpha: u32,
    #[serde(default = "default_dropout")]     pub lora_dropout: f64,
    #[serde(default = "default_lora_mods")]   pub lora_target_modules: Vec<String>,

    #[serde(default)] pub load_in_8bit: bool,
    #[serde(default)] pub load_in_4bit: bool,

    #[serde(default = "default_seq_len")]      pub max_seq_length: u32,
    #[serde(default = "default_workers")]      pub num_workers: u32,
    #[serde(default = "default_true")]         pub pin_memory: bool,

    #[serde(default = "default_eval_steps")]   pub eval_steps: u32,
    #[serde(default = "default_strategy")]     pub eval_strategy: String,
    #[serde(default = "default_eval_steps")]   pub save_steps: u32,
    #[serde(default = "default_strategy")]     pub save_strategy: String,
    #[serde(default = "default_save_limit")]   pub save_total_limit: u32,
    #[serde(default = "default_log_steps")]    pub logging_steps: u32,

    #[serde(default = "default_seed")]         pub seed: u32,
    #[serde(default)] pub dataloader_drop_last: bool,
    #[serde(default)] pub group_by_length: bool,
    #[serde(default)] pub gradient_checkpointing: bool,

    #[serde(default = "default_training_type")] pub training_type: String,
    /// task_type steuert das Python-Plugin.
    /// Für Sequenzklassifikation: "seq_classification"
    #[serde(default = "default_task_type")]     pub task_type: String,
}

fn default_epochs() -> u32 { 3 }
fn default_batch_size() -> u32 { 8 }
fn default_one() -> u32 { 1 }
fn default_minus_one() -> i32 { -1 }
fn default_lr() -> f64 { 2e-5 }
fn default_weight_decay() -> f64 { 0.01 }
fn default_optimizer() -> String { "adamw".to_string() }
fn default_beta1() -> f64 { 0.9 }
fn default_beta2() -> f64 { 0.999 }
fn default_epsilon() -> f64 { 1e-8 }
fn default_momentum() -> f64 { 0.9 }
fn default_scheduler() -> String { "linear".to_string() }
fn default_gamma() -> f64 { 0.1 }
fn default_dropout() -> f64 { 0.1 }
fn default_grad_norm() -> f64 { 1.0 }
fn default_lora_r() -> u32 { 8 }
fn default_lora_alpha() -> u32 { 32 }
fn default_lora_mods() -> Vec<String> { vec!["query".to_string(), "value".to_string()] }
fn default_seq_len() -> u32 { 128 }
fn default_workers() -> u32 { 0 }
fn default_true() -> bool { true }
fn default_eval_steps() -> u32 { 500 }
fn default_strategy() -> String { "epoch".to_string() }
fn default_save_limit() -> u32 { 3 }
fn default_log_steps() -> u32 { 10 }
fn default_seed() -> u32 { 42 }
fn default_training_type() -> String { "fine_tuning".to_string() }
fn default_task_type() -> String { "seq_classification".to_string() }

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(), dataset_path: String::new(),
            output_path: String::new(), checkpoint_dir: String::new(),
            epochs: 3, batch_size: 8, gradient_accumulation_steps: 1, max_steps: -1,
            learning_rate: 2e-5, weight_decay: 0.01, warmup_steps: 0, warmup_ratio: 0.0,
            optimizer: "adamw".to_string(),
            adam_beta1: 0.9, adam_beta2: 0.999, adam_epsilon: 1e-8, sgd_momentum: 0.9,
            scheduler: "linear".to_string(),
            scheduler_step_size: 1, scheduler_gamma: 0.1, cosine_min_lr: 0.0,
            dropout: 0.1, max_grad_norm: 1.0, label_smoothing: 0.0,
            fp16: false, bf16: false,
            use_lora: false, lora_r: 8, lora_alpha: 32, lora_dropout: 0.1,
            lora_target_modules: vec!["query".to_string(), "value".to_string()],
            load_in_8bit: false, load_in_4bit: false,
            max_seq_length: 128, num_workers: 0, pin_memory: false,
            eval_steps: 500, eval_strategy: "epoch".to_string(),
            save_steps: 500, save_strategy: "epoch".to_string(), save_total_limit: 3,
            logging_steps: 10, seed: 42,
            dataloader_drop_last: false, group_by_length: false, gradient_checkpointing: false,
            training_type: "fine_tuning".to_string(),
            task_type: "seq_classification".to_string(),
        }
    }
}

// ============ Presets ============

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetConfig {
    pub id: String,
    pub name: String,
    pub description: String,
    pub config: TrainingConfig,
}

#[tauri::command]
pub fn get_training_presets() -> Result<Vec<PresetConfig>, String> {
    Ok(vec![
        PresetConfig {
            id: "standard_classification".to_string(),
            name: "Standard Klassifikation".to_string(),
            description: "Bewährte Einstellungen für Text-Klassifikation. Funktioniert gut für die meisten Aufgaben.".to_string(),
            config: TrainingConfig {
                learning_rate: 2e-5,
                batch_size: 8,
                epochs: 3,
                optimizer: "adamw".to_string(),
                scheduler: "linear".to_string(),
                warmup_ratio: 0.1,
                weight_decay: 0.01,
                max_seq_length: 128,
                eval_strategy: "epoch".to_string(),
                save_strategy: "epoch".to_string(),
                logging_steps: 10,
                task_type: "seq_classification".to_string(),
                ..Default::default()
            },
        },
        PresetConfig {
            id: "long_texts".to_string(),
            name: "Lange Texte".to_string(),
            description: "Für längere Dokumente oder Artikel. Höhere Sequenzlänge, kleinere Batch-Size.".to_string(),
            config: TrainingConfig {
                learning_rate: 2e-5,
                batch_size: 4,
                gradient_accumulation_steps: 4,
                epochs: 3,
                max_seq_length: 512,
                optimizer: "adamw".to_string(),
                scheduler: "linear".to_string(),
                warmup_ratio: 0.1,
                weight_decay: 0.01,
                eval_strategy: "epoch".to_string(),
                save_strategy: "epoch".to_string(),
                logging_steps: 10,
                task_type: "seq_classification".to_string(),
                ..Default::default()
            },
        },
        PresetConfig {
            id: "quick_test".to_string(),
            name: "Schnelltest".to_string(),
            description: "Für schnelle Experimente: 1 Epoche, kleine Batch-Size.".to_string(),
            config: TrainingConfig {
                learning_rate: 2e-5,
                batch_size: 8,
                epochs: 1,
                max_seq_length: 64,
                logging_steps: 5,
                eval_strategy: "epoch".to_string(),
                save_strategy: "epoch".to_string(),
                task_type: "seq_classification".to_string(),
                ..Default::default()
            },
        },
        PresetConfig {
            id: "conservative_stable".to_string(),
            name: "Konservativ & Stabil".to_string(),
            description: "Kleinere Lernrate, mehr Epochen. Weniger Overfitting-Risiko.".to_string(),
            config: TrainingConfig {
                learning_rate: 1e-5,
                batch_size: 8,
                epochs: 5,
                max_seq_length: 128,
                optimizer: "adamw".to_string(),
                scheduler: "cosine".to_string(),
                warmup_ratio: 0.1,
                weight_decay: 0.01,
                max_grad_norm: 1.0,
                eval_strategy: "epoch".to_string(),
                save_strategy: "epoch".to_string(),
                logging_steps: 10,
                task_type: "seq_classification".to_string(),
                ..Default::default()
            },
        },
        PresetConfig {
            id: "ram_efficient".to_string(),
            name: "RAM-Schonend".to_string(),
            description: "Für Rechner mit wenig RAM. Sehr kleine Batch-Size, Gradient-Akkumulation.".to_string(),
            config: TrainingConfig {
                learning_rate: 2e-5,
                batch_size: 2,
                gradient_accumulation_steps: 8,
                epochs: 3,
                max_seq_length: 64,
                gradient_checkpointing: true,
                eval_strategy: "epoch".to_string(),
                save_strategy: "epoch".to_string(),
                logging_steps: 10,
                task_type: "seq_classification".to_string(),
                ..Default::default()
            },
        },
    ])
}

// ============ Rating ============

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterRating {
    pub score: u32,
    pub rating: String,
    pub rating_info: RatingInfo,
    pub issues: Vec<String>,
    pub warnings: Vec<String>,
    pub tips: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RatingInfo {
    pub score: u32,
    pub label: String,
    pub color: String,
}

#[tauri::command]
pub fn rate_training_config(config: TrainingConfig) -> Result<ParameterRating, String> {
    let mut score: i32 = 100;
    let mut issues   = Vec::new();
    let mut warnings = Vec::new();
    let mut tips     = Vec::new();

    // Lernrate
    if config.learning_rate > 1e-3 {
        issues.push("Learning Rate > 1e-3 ist sehr hoch — Training wird instabil.".to_string());
        score -= 30;
    } else if config.learning_rate > 5e-4 {
        warnings.push("Learning Rate ist relativ hoch (> 5e-4). Für Klassifikation empfohlen: 1e-5 – 3e-5.".to_string());
        score -= 10;
    } else if config.learning_rate < 1e-6 {
        warnings.push("Learning Rate sehr niedrig — Konvergenz kann sehr lang dauern.".to_string());
        score -= 10;
    } else {
        tips.push("Learning Rate ist im optimalen Bereich für Textklassifikation.".to_string());
    }

    // Batch Size
    if config.batch_size < 2 {
        warnings.push("Sehr kleine Batch-Size (< 2) — verrauschte Gradienten möglich.".to_string());
        score -= 10;
    }

    // Sequenzlänge
    if config.max_seq_length > 512 {
        warnings.push("max_seq_length > 512 benötigt sehr viel RAM.".to_string());
        score -= 5;
    } else if config.max_seq_length < 32 {
        warnings.push("max_seq_length < 32 könnte zu viel Informationsverlust führen.".to_string());
        score -= 10;
    }

    // Epochen
    if config.epochs > 10 {
        warnings.push("Mehr als 10 Epochen kann zu Overfitting führen.".to_string());
        score -= 5;
    } else if config.epochs < 2 {
        warnings.push("Weniger als 2 Epochen — Modell konvergiert möglicherweise nicht.".to_string());
        score -= 10;
    } else {
        tips.push("Epochen-Anzahl ist in einem guten Bereich.".to_string());
    }

    // Warmup
    if config.warmup_ratio > 0.0 && config.warmup_ratio <= 0.15 {
        tips.push(format!("Warmup-Ratio von {:.0}% hilft bei stabilem Start.", config.warmup_ratio * 100.0));
    } else if config.warmup_ratio > 0.2 {
        warnings.push("Sehr langes Warmup (> 20%) reduziert die effektive Trainingszeit.".to_string());
        score -= 5;
    }

    // Weight Decay
    if config.weight_decay > 0.1 {
        warnings.push("Hoher Weight Decay (> 0.1) kann zu Underfitting führen.".to_string());
        score -= 5;
    }

    let score = (score.max(0) as u32).min(100);
    let (rating, label, color) = if score >= 90 { ("excellent","Exzellent","green") }
        else if score >= 75 { ("good","Gut","blue") }
        else if score >= 60 { ("okay","Okay","yellow") }
        else if score >= 40 { ("risky","Riskant","orange") }
        else { ("bad","Schlecht","red") };

    Ok(ParameterRating {
        score,
        rating: rating.to_string(),
        rating_info: RatingInfo {
            score: match rating { "excellent" => 5, "good" => 4, "okay" => 3, "risky" => 2, _ => 1 },
            label: label.to_string(),
            color: color.to_string(),
        },
        issues, warnings, tips,
    })
}

// ============ Training State ============

pub struct TrainingState {
    pub current_job: Option<TrainingJob>,
    pub process: Option<Child>,
    pub process_pid: Option<u32>,
    pub jobs_history: Vec<TrainingJob>,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self { current_job: None, process: None, process_pid: None, jobs_history: Vec::new() }
    }
}

// ============ Hilfsfunktionen ============

fn get_python_path() -> String {
    println!("[Python] Suche Python mit torch...");

    struct Candidate { path: String, version: (u32,u32,u32) }
    let mut candidates: Vec<Candidate> = Vec::new();

    if !cfg!(target_os = "windows") {
        let dirs = ["/opt/homebrew/bin","/usr/local/bin","/usr/bin"];
        let names = ["python3.13","python3.12","python3.11","python3.10","python3.9","python3"];
        for d in &dirs {
            for n in &names {
                let full = format!("{}/{}", d, n);
                if let Ok(out) = Command::new(&full).arg("--version").output() {
                    if out.status.success() {
                        let vs = String::from_utf8_lossy(&out.stdout);
                        if let Some(v) = parse_version(&vs) {
                            candidates.push(Candidate { path: full, version: v });
                        }
                    }
                }
            }
        }
    }

    for cmd in &["python3","python"] {
        if let Ok(out) = Command::new(cmd).arg("--version").output() {
            if out.status.success() {
                let vs = String::from_utf8_lossy(&out.stdout);
                if let Some(v) = parse_version(&vs) {
                    candidates.push(Candidate { path: cmd.to_string(), version: v });
                }
            }
        }
    }

    candidates.sort_by(|a,b| b.version.cmp(&a.version));
    candidates.dedup_by(|a,b| a.version == b.version);

    for c in &candidates {
        let ok = Command::new(&c.path).args(["-c","import torch"]).output()
            .map(|o| o.status.success()).unwrap_or(false);
        if ok {
            println!("[Python] ✅ Gewählt (torch vorhanden): {}", c.path);
            return c.path.clone();
        }
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

fn get_train_engine_path(app_handle: &tauri::AppHandle) -> Result<PathBuf, String> {
    let candidates = vec![
        // Produktion: gebundelte Ressourcen
        app_handle.path().resource_dir().ok()
            .map(|p| p.join("python").join("train_engine").join("train_engine.py")),
        // Entwicklung: relativ zum Projekt
        Some(PathBuf::from("src-tauri/python/train_engine/train_engine.py")),
        // Absoluter Dev-Pfad (dieses Projekt)
        Some(PathBuf::from("/Users/karol/Desktop/Laufende_Projekte/FrameTrain/desktop-app/src-tauri/python/train_engine/train_engine.py")),
    ];

    for candidate in candidates.into_iter().flatten() {
        if candidate.exists() {
            println!("[Engine] ✅ Gefunden: {:?}", candidate);
            return Ok(candidate);
        }
    }
    Err("Train-Engine nicht gefunden".to_string())
}

fn get_models_dir(app_handle: &tauri::AppHandle) -> Result<PathBuf, String> {
    app_handle.path().app_data_dir()
        .map(|p| p.join("models"))
        .map_err(|e| format!("AppDataDir: {}", e))
}

fn get_output_dir(app_handle: &tauri::AppHandle, job_id: &str) -> Result<PathBuf, String> {
    let dir = app_handle.path().app_data_dir()
        .map_err(|e| format!("AppDataDir: {}", e))?
        .join("training_outputs").join(job_id);
    fs::create_dir_all(&dir).map_err(|e| format!("Output-Dir: {}", e))?;
    Ok(dir)
}

fn save_job(app_handle: &tauri::AppHandle, job: TrainingJob) -> Result<(), String> {
    let mut jobs = load_jobs(app_handle).unwrap_or_default();
    if let Some(pos) = jobs.iter().position(|j| j.id == job.id) {
        jobs[pos] = job;
    } else {
        jobs.insert(0, job);
    }
    jobs.truncate(200);
    write_jobs(app_handle, &jobs)
}

fn write_jobs(app_handle: &tauri::AppHandle, jobs: &[TrainingJob]) -> Result<(), String> {
    let path = app_handle.path().app_data_dir()
        .map_err(|e| format!("AppDataDir: {}", e))?
        .join("training_jobs.json");
    let content = serde_json::to_string_pretty(jobs).map_err(|e| format!("JSON: {}", e))?;
    fs::write(&path, content).map_err(|e| format!("Schreiben: {}", e))
}

fn load_jobs(app_handle: &tauri::AppHandle) -> Result<Vec<TrainingJob>, String> {
    let path = app_handle.path().app_data_dir()
        .map_err(|e| format!("AppDataDir: {}", e))?
        .join("training_jobs.json");
    if !path.exists() { return Ok(Vec::new()); }
    let content = fs::read_to_string(&path).map_err(|e| format!("Lesen: {}", e))?;
    serde_json::from_str(&content).map_err(|e| format!("JSON: {}", e))
}

// ============ Tauri Commands ============

#[tauri::command]
pub async fn start_training(
    app_handle: tauri::AppHandle,
    model_id: String,
    model_name: String,
    dataset_id: String,
    dataset_name: String,
    config: TrainingConfig,
    version_id: Option<String>,
    state: tauri::State<'_, Arc<Mutex<TrainingState>>>,
) -> Result<TrainingJob, String> {
    let mut sl = state.lock().map_err(|e| format!("Lock: {}", e))?;
    if sl.current_job.is_some() {
        return Err("Ein Training läuft bereits".to_string());
    }

    // Anti-Sleep direkt im Backend aktivieren (robust, unabhängig vom Frontend).
    if let Err(e) = crate::power_manager::enable_prevent_sleep(
        app_handle.state::<StdMutex<crate::power_manager::PowerState>>(),
    ) {
        eprintln!("[PowerManager] ⚠️ enable_prevent_sleep fehlgeschlagen: {}", e);
    }

    let job_id = format!("train_{}", &uuid::Uuid::new_v4().to_string().replace("-","")[..12]);
    let models_dir = get_models_dir(&app_handle)?;

    // Modell-Pfad: Aus Version-DB lesen wenn version_id gesetzt, sonst models_dir/model_id
    let model_path = if let Some(ref vid) = version_id {
        let db_path = app_handle.path().app_data_dir()
            .map_err(|e| format!("AppDataDir: {}", e))?.join("frametrain.db");
        let conn = rusqlite::Connection::open(&db_path).map_err(|e| format!("DB: {}", e))?;
        let vpath: String = conn.query_row(
            "SELECT path FROM model_versions_new WHERE id = ?1", [vid], |r| r.get(0),
        ).map_err(|e| format!("Version nicht gefunden: {}", e))?;
        PathBuf::from(vpath)
    } else {
        models_dir.join(&model_id)
    };

    // Dataset-Pfad aus DB
    let dataset_path = {
        let db_path = app_handle.path().app_data_dir()
            .map_err(|e| format!("AppDataDir: {}", e))?.join("frametrain.db");
        let conn = rusqlite::Connection::open(&db_path).map_err(|e| format!("DB: {}", e))?;
        let res: Result<String, _> = conn.query_row(
            "SELECT file_path FROM datasets WHERE id = ?1", [&dataset_id], |r| r.get(0),
        );
        match res {
            Ok(p) if !p.is_empty() => PathBuf::from(p),
            _ => {
                // Fallback: Verwende standard datasets_dir statt model-spezifischen Pfad
                let datasets_dir = app_handle.path().app_data_dir()
                    .map_err(|e| format!("AppDataDir: {}", e))?
                    .join("datasets");
                datasets_dir.join(&dataset_id)
            }
        }
    };

    let output_dir   = get_output_dir(&app_handle, &job_id)?;
    let checkpoint_dir = output_dir.join("checkpoints");
    fs::create_dir_all(&checkpoint_dir).map_err(|e| format!("Checkpoint-Dir: {}", e))?;

    let mut final_config = config.clone();
    final_config.model_path    = model_path.to_string_lossy().to_string();
    final_config.dataset_path  = dataset_path.to_string_lossy().to_string();
    final_config.output_path   = output_dir.join("final_model").to_string_lossy().to_string();
    final_config.checkpoint_dir= checkpoint_dir.to_string_lossy().to_string();
    // Sicherstellen dass task_type immer seq_classification ist
    final_config.task_type     = "seq_classification".to_string();

    let config_path = output_dir.join("config.json");
    fs::write(&config_path, serde_json::to_string_pretty(&final_config)
        .map_err(|e| format!("Config JSON: {}", e))?)
        .map_err(|e| format!("Config schreiben: {}", e))?;

    let user_id = {
        let app_state = app_handle.state::<crate::AppState>();
        let db = app_state.db.lock().map_err(|e| format!("DB Lock: {}", e))?;
        db.get_current_user_id().ok_or_else(|| "Kein User eingeloggt".to_string())?
    };

    let job = TrainingJob {
        id: job_id.clone(), model_id: model_id.clone(), model_name: model_name.clone(),
        dataset_id: dataset_id.clone(), dataset_name,
        status: TrainingStatus::Pending, config: final_config,
        created_at: Utc::now(), started_at: None, completed_at: None,
        progress: TrainingProgress::default(),
        output_path: Some(output_dir.to_string_lossy().to_string()), error: None,
    };

    sl.current_job = Some(job.clone());
    drop(sl);

    let ah = app_handle.clone();
    let cfg_path_str = config_path.to_string_lossy().to_string();
    let state_clone = Arc::clone(&state);
    let vid_clone = version_id.clone();

    thread::spawn(move || {
        run_training(ah, job_id, cfg_path_str, model_id, model_name, vid_clone, user_id, dataset_id, state_clone);
    });

    Ok(job)
}

fn create_version(
    app_handle: &tauri::AppHandle,
    model_id: &str,
    model_name: &str,
    parent_version_id: Option<String>,
    output_path: &str,
    user_id: &str,
) -> Result<String, String> {
    let db_path = app_handle.path().app_data_dir()
        .map_err(|e| format!("AppDataDir: {}", e))?.join("frametrain.db");
    let conn = rusqlite::Connection::open(&db_path).map_err(|e| format!("DB: {}", e))?;

    conn.execute("PRAGMA foreign_keys = OFF", []).ok();

    conn.execute("CREATE TABLE IF NOT EXISTS models (
        id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT, base_model TEXT,
        model_path TEXT, status TEXT NOT NULL DEFAULT 'created',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP, UNIQUE(name))", []).ok();

    conn.execute("CREATE TABLE IF NOT EXISTS model_versions_new (
        id TEXT PRIMARY KEY, model_id TEXT NOT NULL, version_name TEXT NOT NULL,
        version_number INTEGER NOT NULL, path TEXT NOT NULL,
        size_bytes INTEGER NOT NULL DEFAULT 0, file_count INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL, is_root INTEGER NOT NULL DEFAULT 0,
        parent_version_id TEXT, user_id TEXT)", []).ok();
    let _ = conn.execute("ALTER TABLE model_versions_new ADD COLUMN user_id TEXT", []);

    conn.execute("CREATE TABLE IF NOT EXISTS training_metrics_new (
        id TEXT PRIMARY KEY, version_id TEXT NOT NULL UNIQUE,
        final_train_loss REAL NOT NULL, final_val_loss REAL,
        total_epochs INTEGER NOT NULL, total_steps INTEGER NOT NULL,
        best_epoch INTEGER, training_duration_seconds INTEGER,
        created_at TEXT NOT NULL, user_id TEXT)", []).ok();
    let _ = conn.execute("ALTER TABLE training_metrics_new ADD COLUMN user_id TEXT", []);

    conn.execute("CREATE INDEX IF NOT EXISTS idx_versions_model ON model_versions_new(model_id)", []).ok();
    conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_version ON training_metrics_new(version_id)", []).ok();

    // Model-Eintrag sicherstellen
    let model_exists: i32 = conn.query_row(
        "SELECT COUNT(*) FROM models WHERE id = ?1", [model_id], |r| r.get(0)).unwrap_or(0);
    if model_exists == 0 {
        let now = Utc::now().to_rfc3339();
        let models_dir = get_models_dir(app_handle)?;
        let mp = models_dir.join(model_id).to_string_lossy().to_string();
        let unique_name = format!("{} ({})", model_name, &model_id[..8.min(model_id.len())]);
        conn.execute(
            "INSERT INTO models (id, name, model_path, status, created_at, updated_at) VALUES (?1,?2,?3,?4,?5,?6)",
            rusqlite::params![model_id, &unique_name, &mp, "trained", &now, &now],
        ).ok();
    }

    let version_id = format!("ver_{}", &uuid::Uuid::new_v4().to_string().replace("-","")[..12]);
    let version_number: i32 = conn.query_row(
        "SELECT COALESCE(MAX(version_number),0)+1 FROM model_versions_new WHERE model_id=?1",
        [model_id], |r| r.get(0)).unwrap_or(1);

    let models_dir  = get_models_dir(app_handle)?;
    let version_path = models_dir.join(model_id).join("versions").join(&version_id);
    fs::create_dir_all(&version_path).map_err(|e| format!("Version-Dir: {}", e))?;

    let src = PathBuf::from(output_path);
    if src.exists() {
        copy_dir(&src, &version_path)?;
    } else {
        return Err(format!("Output-Pfad existiert nicht: {}", src.display()));
    }

    let (size, files) = dir_size(&version_path).unwrap_or((0, 0));
    let now = Utc::now().to_rfc3339();
    conn.execute(
        "INSERT INTO model_versions_new (id,model_id,version_name,version_number,path,size_bytes,file_count,created_at,is_root,parent_version_id,user_id) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
        rusqlite::params![&version_id, model_id, format!("{} v{}", model_name, version_number),
            version_number, version_path.to_string_lossy().to_string(), size, files, &now, 0i32, parent_version_id, user_id],
    ).map_err(|e| format!("Version-Record: {}", e))?;

    Ok(version_id)
}

fn copy_dir(src: &PathBuf, dst: &PathBuf) -> Result<(), String> {
    if !dst.exists() { fs::create_dir_all(dst).map_err(|e| format!("mkdir: {}", e))?; }
    for entry in fs::read_dir(src).map_err(|e| format!("readdir: {}", e))? {
        let entry = entry.map_err(|e| format!("entry: {}", e))?;
        let sp = entry.path();
        let dp = dst.join(entry.file_name());
        if sp.is_dir() { copy_dir(&sp, &dp)?; }
        else { fs::copy(&sp, &dp).map_err(|e| format!("copy: {}", e))?; }
    }
    Ok(())
}

fn dir_size(path: &PathBuf) -> Result<(i64, i32), String> {
    let mut size: i64 = 0; let mut count: i32 = 0;
    fn visit(dir: &PathBuf, s: &mut i64, c: &mut i32) -> Result<(), String> {
        if dir.is_dir() {
            for e in fs::read_dir(dir).map_err(|e| e.to_string())? {
                let p = e.map_err(|e| e.to_string())?.path();
                if p.is_dir() { visit(&p, s, c)?; }
                else { if let Ok(m) = fs::metadata(&p) { *s += m.len() as i64; *c += 1; } }
            }
        }
        Ok(())
    }
    visit(path, &mut size, &mut count)?;
    Ok((size, count))
}

fn save_metrics(app_handle: &tauri::AppHandle, version_id: &str, data: &serde_json::Value, user_id: &str) -> Result<(), String> {
    let metrics = data.get("final_metrics").unwrap_or(data);

    let train_loss = metrics.get("final_train_loss").and_then(|v| v.as_f64())
        .or_else(|| data.get("train_loss").and_then(|v| v.as_f64())).unwrap_or(0.0);
    let val_loss   = metrics.get("final_val_loss").and_then(|v| v.as_f64());
    let epochs     = metrics.get("total_epochs").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
    let steps      = metrics.get("total_steps").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
    let best_ep    = metrics.get("best_epoch").and_then(|v| v.as_i64()).map(|v| v as i32);
    let duration   = data.get("training_duration_seconds").and_then(|v| v.as_i64());

    if epochs == 0 { return Err("Keine Metriken (epochs=0)".to_string()); }

    let db_path = app_handle.path().app_data_dir()
        .map_err(|e| format!("AppDataDir: {}", e))?.join("frametrain.db");
    let conn = rusqlite::Connection::open(&db_path).map_err(|e| format!("DB: {}", e))?;
    let id  = format!("metrics_{}", uuid::Uuid::new_v4());
    let now = Utc::now().to_rfc3339();
    conn.execute(
        "INSERT OR REPLACE INTO training_metrics_new (id,version_id,final_train_loss,final_val_loss,total_epochs,total_steps,best_epoch,training_duration_seconds,created_at,user_id) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
        rusqlite::params![id, version_id, train_loss, val_loss, epochs, steps, best_ep, duration, now, user_id],
    ).map_err(|e| format!("Metriken speichern: {}", e))?;
    Ok(())
}

fn run_training(
    app_handle: tauri::AppHandle, job_id: String, config_path: String,
    model_id: String, model_name: String, version_id: Option<String>,
    user_id: String, dataset_id: String, state: Arc<Mutex<TrainingState>>,
) {
    let python = get_python_path();
    let engine_path = match get_train_engine_path(&app_handle) {
        Ok(p) => p,
        Err(e) => {
            let _ = app_handle.emit("training-error", serde_json::json!({"job_id":job_id,"error":e}));
            return;
        }
    };

    let _ = app_handle.emit("training-started", serde_json::json!({"job_id":job_id}));

    let mut child = match Command::new(&python)
        .arg(engine_path.to_string_lossy().to_string())
        .arg("--config").arg(&config_path)
        .stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()
    {
        Ok(c) => c,
        Err(e) => {
            let _ = app_handle.emit("training-error", serde_json::json!({"job_id":job_id,"error":format!("Python start: {}",e)}));
            return;
        }
    };

    if let Some(pid) = child.id().into() {
        if let Ok(mut s) = state.lock() { s.process_pid = Some(pid); }
    }
    let stderr_lines: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    if let Some(pid) = Some(child.id()) {
        if let Ok(mut s) = state.lock() { s.process_pid = Some(pid); }
    }

    if let Some(stderr) = child.stderr.take() {
        let sl = Arc::clone(&stderr_lines);
        thread::spawn(move || {
            for line in BufReader::new(stderr).lines().flatten() {
                eprintln!("[Train STDERR] {}", line);
                if let Ok(mut v) = sl.lock() {
                    v.push(line);
                    if v.len() > 50 { let n = v.len() - 50; v.drain(0..n); }
                }
            }
        });
    }

    // ── Step-Log-Akkumulation für Analyse ────────────────────────────────
    let started_at_secs = Utc::now().timestamp();
    let mut step_logs: Vec<serde_json::Value> = Vec::new();
    let final_config_json: serde_json::Value = fs::read_to_string(&config_path)
        .ok()
        .and_then(|c| serde_json::from_str(&c).ok())
        .unwrap_or(serde_json::Value::Null);

    let mut json_error = false;

    if let Some(stdout) = child.stdout.take() {
        let ah = app_handle.clone();
        let jid = job_id.clone();
        let mid = model_id.clone();
        let mname = model_name.clone();
        let vid = version_id.clone();
        let uid = user_id.clone();

        for line in BufReader::new(stdout).lines().flatten() {
            println!("[Train] {}", line);
            let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) else { continue };
            let typ = msg.get("type").and_then(|t| t.as_str()).unwrap_or("");

            match typ {
                "progress" => {
                    // Step-Log akkumulieren
                    if let Some(data) = msg.get("data") {
                        let log_entry = serde_json::json!({
                            "epoch":         data.get("epoch").and_then(|v| v.as_i64()).unwrap_or(0),
                            "step":          data.get("step").and_then(|v| v.as_i64()).unwrap_or(0),
                            "train_loss":    data.get("train_loss").and_then(|v| v.as_f64()).unwrap_or(0.0),
                            "val_loss":      data.get("val_loss").and_then(|v| v.as_f64()),
                            "learning_rate": data.get("learning_rate").and_then(|v| v.as_f64()).unwrap_or(0.0),
                            "grad_norm":     data.get("grad_norm").and_then(|v| v.as_f64()),
                            "elapsed_seconds": Utc::now().timestamp() - started_at_secs,
                            "timestamp":     Utc::now().to_rfc3339(),
                        });
                        step_logs.push(log_entry);
                    }
                    let _ = ah.emit("training-progress", serde_json::json!({"job_id":jid,"data":msg.get("data")}));
                }
                "status"   => { let _ = ah.emit("training-status",   serde_json::json!({"job_id":jid,"data":msg.get("data")})); }
                "checkpoint"=>{ let _ = ah.emit("training-checkpoint",serde_json::json!({"job_id":jid,"data":msg.get("data")})); }
                "complete" => {
                    if let Some(data) = msg.get("data") {
                        if let Some(mp) = data.get("model_path").and_then(|v| v.as_str()) {
                            match create_version(&ah, &mid, &mname, vid.clone(), mp, &uid) {
                                Ok(new_vid) => {
                                    if let Err(e) = save_metrics(&ah, &new_vid, data, &uid) {
                                        eprintln!("[Train] Metriken: {}", e);
                                    }
                                    // Full-Data + Step-Logs für Analyse-Seite speichern
                                    crate::analysis_manager::save_full_analysis_data(
                                        &ah,
                                        &new_vid,
                                        data,
                                        &step_logs,
                                        &final_config_json,
                                        started_at_secs,
                                    );
                                    if let Ok(db_guard) = ah.state::<crate::AppState>().db.lock() {
                                        let _ = db_guard.mark_dataset_used(&dataset_id);
                                    }
                                    let _ = ah.emit("training-complete", serde_json::json!({"job_id":jid,"data":data,"new_version_id":new_vid}));
                                }
                                Err(e) => {
                                    eprintln!("[Train] Version: {}", e);
                                    let _ = ah.emit("training-complete", serde_json::json!({"job_id":jid,"data":data,"version_error":e}));
                                }
                            }
                        } else {
                            let _ = ah.emit("training-complete", serde_json::json!({"job_id":jid,"data":data}));
                        }
                    }
                }
                "error" => {
                    json_error = true;
                    let _ = ah.emit("training-error", serde_json::json!({"job_id":jid,"data":msg.get("data")}));
                }
                _ => {}
            }
        }
    }

    let status = child.wait();
    let ok = status.as_ref().map(|s| s.success()).unwrap_or(false);

    if !ok && !json_error {
        let stderr_ctx = stderr_lines.lock().ok()
            .map(|v| if v.is_empty() { String::new() } else { format!("\n\nStderr:\n{}", v.join("\n")) })
            .unwrap_or_default();
        let _ = app_handle.emit("training-error", serde_json::json!({
            "job_id": job_id,
            "data": { "error": "Training unerwartet beendet", "details": format!("Exit: {:?}{}", status.as_ref().map(|s| s.code()), stderr_ctx) }
        }));
    }

    if let Ok(mut sl) = state.lock() {
        if let Some(ref mut job) = sl.current_job {
            if job.completed_at.is_none() { job.completed_at = Some(Utc::now()); }
            if job.status == TrainingStatus::Pending || job.status == TrainingStatus::Running {
                job.status = if ok { TrainingStatus::Completed } else { TrainingStatus::Failed };
            }
            let _ = save_job(&app_handle, job.clone());
        }
        sl.current_job = None; sl.process = None; sl.process_pid = None;
    }

    // Anti-Sleep deaktivieren sobald der Prozess endet (egal ob Success/Fail).
    if let Err(e) = crate::power_manager::disable_prevent_sleep(
        app_handle.state::<StdMutex<crate::power_manager::PowerState>>(),
    ) {
        eprintln!("[PowerManager] ⚠️ disable_prevent_sleep fehlgeschlagen: {}", e);
    }

    let _ = app_handle.emit("training-finished", serde_json::json!({"job_id":job_id,"success":ok}));
}

#[tauri::command]
pub fn stop_training(
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, Arc<Mutex<TrainingState>>>,
) -> Result<(), String> {
    let mut sl = state.lock().map_err(|e| format!("Lock: {}", e))?;
    if let Some(ref mut p) = sl.process { let _ = p.kill(); }
    if let Some(pid) = sl.process_pid {
        #[cfg(unix)] {
            let _ = Command::new("kill").args(["-TERM", &pid.to_string()]).output();
            thread::sleep(std::time::Duration::from_millis(300));
            let _ = Command::new("kill").args(["-KILL", &pid.to_string()]).output();
            let _ = Command::new("pkill").args(["-KILL","-P",&pid.to_string()]).output();
        }
        #[cfg(windows)] { let _ = Command::new("taskkill").args(["/F","/PID",&pid.to_string(),"/T"]).output(); }
    }
    if let Some(ref mut job) = sl.current_job {
        job.status = TrainingStatus::Stopped;
        job.completed_at = Some(Utc::now());
        let _ = save_job(&app_handle, job.clone());
    }
    sl.process = None; sl.process_pid = None; sl.current_job = None;

    if let Err(e) = crate::power_manager::disable_prevent_sleep(
        app_handle.state::<StdMutex<crate::power_manager::PowerState>>(),
    ) {
        eprintln!("[PowerManager] ⚠️ disable_prevent_sleep fehlgeschlagen: {}", e);
    }

    Ok(())
}

#[tauri::command]
pub fn get_current_training(state: tauri::State<'_, Arc<Mutex<TrainingState>>>) -> Result<Option<TrainingJob>, String> {
    Ok(state.lock().map_err(|e| format!("Lock: {}", e))?.current_job.clone())
}

#[tauri::command]
pub fn get_training_history(app_handle: tauri::AppHandle) -> Result<Vec<TrainingJob>, String> {
    load_jobs(&app_handle)
}

#[tauri::command]
pub fn delete_training_job(app_handle: tauri::AppHandle, job_id: String) -> Result<(), String> {
    let mut jobs = load_jobs(&app_handle)?;
    jobs.retain(|j| j.id != job_id);
    write_jobs(&app_handle, &jobs)?;
    let out = app_handle.path().app_data_dir().map_err(|e| format!("AppDataDir: {}", e))?
        .join("training_outputs").join(&job_id);
    if out.exists() { fs::remove_dir_all(&out).ok(); }
    Ok(())
}

#[tauri::command]
pub fn get_system_ram_gb() -> f64 {
    #[cfg(target_os = "macos")] {
        if let Ok(out) = Command::new("sysctl").args(["-n","hw.memsize"]).output() {
            if let Ok(s) = String::from_utf8(out.stdout) {
                if let Ok(b) = s.trim().parse::<u64>() { return b as f64 / (1024.0_f64).powi(3); }
            }
        }
    }
    #[cfg(target_os = "linux")] {
        if let Ok(c) = fs::read_to_string("/proc/meminfo") {
            for line in c.lines() {
                if line.starts_with("MemTotal:") {
                    let p: Vec<&str> = line.split_whitespace().collect();
                    if p.len() >= 2 { if let Ok(kb) = p[1].parse::<u64>() { return kb as f64 / (1024.0*1024.0); } }
                }
            }
        }
    }
    16.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRamInfo {
    pub param_billion: f64,
    pub model_type: String,
    pub readable_size: String,
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
}

#[tauri::command]
pub fn get_model_ram_info(app_handle: tauri::AppHandle, model_id: String) -> Result<ModelRamInfo, String> {
    let cfg_path = get_models_dir(&app_handle)?.join(&model_id).join("config.json");
    if !cfg_path.exists() {
        return Ok(ModelRamInfo { param_billion: 0.28, model_type: "xlm-roberta".to_string(), readable_size: "278M".to_string(), hidden_size: 768, num_hidden_layers: 12 });
    }
    let content = fs::read_to_string(&cfg_path).map_err(|e| format!("config.json: {}", e))?;
    let cfg: serde_json::Value = serde_json::from_str(&content).map_err(|e| format!("JSON: {}", e))?;
    let h = cfg.get("hidden_size").and_then(|v| v.as_f64()).unwrap_or(768.0);
    let layers = cfg.get("num_hidden_layers").and_then(|v| v.as_f64()).unwrap_or(12.0);
    let vocab  = cfg.get("vocab_size").and_then(|v| v.as_f64()).unwrap_or(250002.0);
    let params = vocab * h + layers * (4.0 * h * h + 2.0 * h * 4.0 * h);
    let pb = params / 1e9;
    let model_type = cfg.get("model_type").and_then(|v| v.as_str()).unwrap_or("xlm-roberta").to_string();
    let readable = if pb < 0.5 { format!("{:.0}M", pb*1000.0) } else { format!("{:.1}B", pb) };
    Ok(ModelRamInfo { param_billion: pb, model_type, readable_size: readable, hidden_size: h as u32, num_hidden_layers: layers as u32 })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequirementsCheck {
    pub python_installed: bool,
    pub python_version: String,
    pub torch_installed: bool,
    pub torch_version: String,
    pub cuda_available: bool,
    pub mps_available: bool,
    pub transformers_installed: bool,
    pub transformers_version: String,
    pub peft_installed: bool,
    pub peft_version: String,
    pub ready: bool,
}

#[tauri::command]
pub async fn check_training_requirements() -> Result<RequirementsCheck, String> {
    let python = get_python_path();

    let py_out = Command::new(&python).arg("--version").output();
    let py_ok  = py_out.is_ok() && py_out.as_ref().unwrap().status.success();
    let py_ver = if py_ok { String::from_utf8_lossy(&py_out.unwrap().stdout).trim().to_string() } else { "Nicht gefunden".to_string() };

    let torch_out = Command::new(&python).args(["-c","import torch; print(torch.__version__)"]).output();
    let torch_ok  = torch_out.is_ok() && torch_out.as_ref().unwrap().status.success();
    let torch_ver = if torch_ok { String::from_utf8_lossy(&torch_out.unwrap().stdout).trim().to_string() } else { "Nicht installiert".to_string() };

    let cuda = Command::new(&python).args(["-c","import torch; print(torch.cuda.is_available())"]).output();
    let cuda_ok = cuda.is_ok() && String::from_utf8_lossy(&cuda.unwrap().stdout).trim() == "True";

    let mps = Command::new(&python).args(["-c","import torch; print(hasattr(torch.backends,'mps') and torch.backends.mps.is_available())"]).output();
    let mps_ok = mps.is_ok() && String::from_utf8_lossy(&mps.unwrap().stdout).trim() == "True";

    let tf_out = Command::new(&python).args(["-c","import transformers; print(transformers.__version__)"]).output();
    let tf_ok  = tf_out.is_ok() && tf_out.as_ref().unwrap().status.success();
    let tf_ver = if tf_ok { String::from_utf8_lossy(&tf_out.unwrap().stdout).trim().to_string() } else { "Nicht installiert".to_string() };

    let peft_out = Command::new(&python).args(["-c","import peft; print(peft.__version__)"]).output();
    let peft_ok  = peft_out.is_ok() && peft_out.as_ref().unwrap().status.success();
    let peft_ver = if peft_ok { String::from_utf8_lossy(&peft_out.unwrap().stdout).trim().to_string() } else { "Nicht installiert".to_string() };

    Ok(RequirementsCheck {
        python_installed: py_ok, python_version: py_ver,
        torch_installed: torch_ok, torch_version: torch_ver,
        cuda_available: cuda_ok, mps_available: mps_ok,
        transformers_installed: tf_ok, transformers_version: tf_ver,
        peft_installed: peft_ok, peft_version: peft_ver,
        ready: py_ok && torch_ok && tf_ok,
    })
}

// ============ Metrics Templates ============

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsTemplate {
    pub id: String, pub name: String, pub description: String,
    pub config: TrainingConfig, pub created_at: String, pub source: String,
}

fn templates_path(app_handle: &tauri::AppHandle) -> Result<PathBuf, String> {
    app_handle.path().app_data_dir().map_err(|e| format!("AppDataDir: {}", e)).map(|d| d.join("metrics_templates.json"))
}

#[tauri::command]
pub fn save_metrics_template(app_handle: tauri::AppHandle, name: String, description: String, config: TrainingConfig, source: String) -> Result<MetricsTemplate, String> {
    let path = templates_path(&app_handle)?;
    let mut templates: Vec<MetricsTemplate> = if path.exists() {
        serde_json::from_str(&fs::read_to_string(&path).unwrap_or_default()).unwrap_or_default()
    } else { vec![] };
    let tmpl = MetricsTemplate {
        id: format!("tmpl_{}", &uuid::Uuid::new_v4().to_string().replace("-","")[..8]),
        name, description, config, created_at: Utc::now().to_rfc3339(), source,
    };
    templates.push(tmpl.clone());
    fs::write(&path, serde_json::to_string_pretty(&templates).unwrap_or_default()).map_err(|e| format!("Write: {}", e))?;
    Ok(tmpl)
}

#[tauri::command]
pub fn get_metrics_templates(app_handle: tauri::AppHandle) -> Result<Vec<MetricsTemplate>, String> {
    let path = templates_path(&app_handle)?;
    if !path.exists() { return Ok(vec![]); }
    serde_json::from_str(&fs::read_to_string(&path).map_err(|e| format!("Read: {}", e))?).map_err(|e| format!("JSON: {}", e))
}

#[tauri::command]
pub fn delete_metrics_template(app_handle: tauri::AppHandle, template_id: String) -> Result<(), String> {
    let path = templates_path(&app_handle)?;
    if !path.exists() { return Ok(()); }
    let mut templates: Vec<MetricsTemplate> = serde_json::from_str(&fs::read_to_string(&path).unwrap_or_default()).unwrap_or_default();
    templates.retain(|t| t.id != template_id);
    fs::write(&path, serde_json::to_string_pretty(&templates).unwrap_or_default()).map_err(|e| format!("Write: {}", e))
}
