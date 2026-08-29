use std::fs;
use crate::command_ext::NoWindow;
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
    /// FIX: user_id für Isolation zwischen Accounts
    #[serde(default)]
    pub user_id: String,
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
    /// 0 = kompletten Eval-Split nutzen, sonst nur die ersten N Beispiele.
    #[serde(default)]                          pub max_eval_samples: u32,
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

    /// Plugin-spezifische Parameter — werden 1:1 an Python durchgereicht.
    /// Jedes Plugin kann hier eigene Werte ablegen (image_size, num_classes, etc.)
    #[serde(default)]                           pub plugin_config: serde_json::Value,

    // ──────────────────────────────────────────────────────────────────────────
    // PHASE 4: Canvas Integration
    // ──────────────────────────────────────────────────────────────────────────
    /// Canvas-generierter PyTorch nn.Module Code
    /// Falls gesetzt: Canvas wird als Trainingsquelle verwendet (statt model_id)
    #[serde(default)]                           pub canvas_model_code: String,
    
    /// Canvas Graph Metadaten für Debugging
    #[serde(default)]                           pub canvas_graph_metadata: serde_json::Value,

    /// Canvas Graph IR (JSON) — runtime training source for Synapse Builder
    #[serde(default)]                           pub canvas_graph: serde_json::Value,
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
            eval_steps: 500, max_eval_samples: 0, eval_strategy: "epoch".to_string(),
            save_steps: 500, save_strategy: "epoch".to_string(), save_total_limit: 3,
            logging_steps: 10, seed: 42,
            dataloader_drop_last: false, group_by_length: false, gradient_checkpointing: false,
            training_type: "fine_tuning".to_string(),
            task_type: "seq_classification".to_string(),
            plugin_config: serde_json::Value::Object(serde_json::Map::new()),
            canvas_model_code: String::new(),
            canvas_graph_metadata: serde_json::Value::Object(serde_json::Map::new()),
            canvas_graph: serde_json::Value::Object(serde_json::Map::new()),
        }
    }
}

/// True when canvas_graph IR has nodes (not empty object/array).
fn has_canvas_graph_ir(cg: &serde_json::Value) -> bool {
    match cg {
        serde_json::Value::Object(m) => {
            m.get("nodes")
                .and_then(|n| n.as_array())
                .map(|a| !a.is_empty())
                .unwrap_or(false)
        }
        _ => false,
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

/// Interpreter-Auswahl fuer alle Trainingswege (auch Dev Train/Dev Test).
pub fn resolve_python_path() -> String { get_python_path() }

fn get_python_path() -> String {
    // Gemeinsame Auswahl fuer Training, Tests, Labor und Einrichtung.
    crate::python_env::resolve_python()
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
    let output_dir   = get_output_dir(&app_handle, &job_id)?;
    let checkpoint_dir = output_dir.join("checkpoints");
    fs::create_dir_all(&checkpoint_dir).map_err(|e| format!("Checkpoint-Dir: {}", e))?;

    let mut final_config = config.clone();
    
    // ──────────────────────────────────────────────────────────────────────────
    // Canvas / Synapse: IR first, then legacy code, else model_id path
    // ──────────────────────────────────────────────────────────────────────────
    // FIX Bug 1: user-scoped JSON-Metadaten first (storage_path), SQLite als Fallback.
    // Vorher wurde nur SQLite file_path gelesen – der stimmt nicht mit dem
    // user-isolierten datasets/<user_id>/<dataset_id>/ Pfad überein.
    let resolve_dataset_path = || -> PathBuf {
        // Ohne Dataset-ID gibt es nichts aufzulösen. WICHTIG: vorher lief die
        // leere ID bis in den Fallback `datasets.join("")` durch — das ergab den
        // datasets-STAMMORDNER, den der image_loader dann als ImageFolder
        // scannte ("Found no valid file for the classes <dataset-ids>...").
        if dataset_id.is_empty() {
            return PathBuf::new();
        }
        // 1. user-scoped Metadaten-JSON → storage_path (Primary Source)
        if let Ok(app_data) = app_handle.path().app_data_dir() {
            // Eigener Block: app_state + db_guard werden hier sofort gedropt
            let maybe_uid: Option<String> = {
                let app_state = app_handle.state::<crate::AppState>();
                app_state.db.lock().ok().and_then(|g| g.get_current_user_id())
            };
            if let Some(uid) = maybe_uid {
                let sanitized: String = uid.chars()
                    .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
                    .collect();
                let user_dir = app_data.join("datasets").join(sanitized);
                let meta_file = user_dir.join("datasets_metadata.json");
                if let Ok(raw) = fs::read_to_string(&meta_file) {
                    if let Ok(list) = serde_json::from_str::<Vec<serde_json::Value>>(&raw) {
                        if let Some(entry) = list.iter().find(|e| {
                            e.get("id").and_then(|v| v.as_str()) == Some(&dataset_id)
                        }) {
                            if let Some(sp) = entry.get("storage_path").and_then(|v| v.as_str()) {
                                let pb = PathBuf::from(sp);
                                if pb.exists() { return pb; }
                            }
                            let fallback = user_dir.join(&dataset_id);
                            if fallback.exists() { return fallback; }
                        }
                    }
                }
            }
        }
        // 2. SQLite file_path (Legacy-Fallback für ältere Einträge)
        if let Ok(app_data) = app_handle.path().app_data_dir() {
            if let Ok(conn) = rusqlite::Connection::open(app_data.join("frametrain.db")) {
                let res: Result<String, _> = conn.query_row(
                    "SELECT file_path FROM datasets WHERE id = ?1", [&dataset_id], |r| r.get(0),
                );
                if let Ok(p) = res {
                    if !p.is_empty() {
                        let pb = PathBuf::from(&p);
                        if pb.exists() { return pb; }
                    }
                }
            }
        }
        // 3. Bare fallback
        app_handle.path().app_data_dir()
            .map(|d| d.join("datasets").join(&dataset_id))
            .unwrap_or_else(|_| PathBuf::from(&dataset_id))
    };

    // ──────────────────────────────────────────────────────────────────────────
    // Canvas-Modell aus Modellbibliothek: graph_metadata.json auto-injizieren
    // wenn canvas_graph noch leer ist (Aufruf von TrainingPanel ohne IR)
    // ──────────────────────────────────────────────────────────────────────────
    if model_id.starts_with("canvas_") && !has_canvas_graph_ir(&final_config.canvas_graph) {
        let meta_path = models_dir.join(&model_id).join("graph_metadata.json");
        if meta_path.exists() {
            if let Ok(content) = fs::read_to_string(&meta_path) {
                if let Ok(meta) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let Some(ir) = meta.get("graphIR") {
                        if has_canvas_graph_ir(ir) {
                            final_config.canvas_graph = ir.clone();
                            eprintln!("[Canvas] ✓ graph_metadata.json → canvas_graph ({} nodes)",
                                ir.get("nodes").and_then(|n| n.as_array()).map(|a| a.len()).unwrap_or(0));
                        }
                    }
                }
            }
        }
    }

    // Iteratives Training: letzten Checkpoint suchen und in plugin_config hinterlegen
    if model_id.starts_with("canvas_") {
        let db_path_val = app_handle.path().app_data_dir()
            .map(|p| p.join("frametrain.db"));
        if let Ok(db_path_val) = db_path_val {
            if let Ok(conn) = rusqlite::Connection::open(&db_path_val) {
                let res: Result<String, _> = conn.query_row(
                    "SELECT path FROM model_versions_new WHERE model_id=?1 AND is_root=0 ORDER BY version_number DESC LIMIT 1",
                    [&model_id], |r| r.get(0),
                );
                if let Ok(ver_path) = res {
                    let pt = PathBuf::from(&ver_path).join("model.pt");
                    if pt.exists() {
                        let mut pc_obj = final_config.plugin_config
                            .as_object()
                            .cloned()
                            .unwrap_or_default();
                        pc_obj.insert(
                            "prev_checkpoint".to_string(),
                            serde_json::Value::String(pt.to_string_lossy().to_string()),
                        );
                        final_config.plugin_config = serde_json::Value::Object(pc_obj);
                        eprintln!("[Canvas] ✓ Iteratives Training: prev_checkpoint = {:?}", pt);
                    }
                }
            }
        }
    }

    if has_canvas_graph_ir(&final_config.canvas_graph) {
        eprintln!("[Canvas] ✓ Graph-IR erkannt ({} nodes)", 
            final_config.canvas_graph.get("nodes").and_then(|n| n.as_array()).map(|a| a.len()).unwrap_or(0));
        final_config.task_type = "canvas".to_string();
        final_config.model_path = String::new();
        final_config.canvas_model_code = String::new();
        let dp = resolve_dataset_path();
        if !dp.as_os_str().is_empty() {
            final_config.dataset_path = dp.to_string_lossy().to_string();
        }
        // sonst: dataset_path vom Frontend behalten (ggf. leer → Engine meldet
        // sauber "kein Dataset-Pfad angegeben" statt den Stammordner zu laden)
        let ir_path = output_dir.join("canvas_graph.json");
        fs::write(&ir_path, serde_json::to_string_pretty(&final_config.canvas_graph)
            .unwrap_or_default())
            .ok();
    } else if !final_config.canvas_model_code.is_empty() {
        eprintln!("[Canvas] ✓ Canvas-Modell erkannt ({} chars Code)", final_config.canvas_model_code.len());
        let canvas_code_path = output_dir.join("canvas_model.py");
        fs::write(&canvas_code_path, &final_config.canvas_model_code)
            .map_err(|e| format!("Canvas-Code speichern: {}", e))?;
        final_config.task_type = "canvas".to_string();
        final_config.model_path = String::new();
        let dp = resolve_dataset_path();
        if !dp.as_os_str().is_empty() {
            final_config.dataset_path = dp.to_string_lossy().to_string();
        }
    } else {
        // Traditionelles Modell - lade von Festplatte
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
        
    // FIX Bug 1: Dataset-Pfad — user-scoped Metadaten-JSON first, SQLite als Fallback.
        let dataset_path = resolve_dataset_path();
        
        // dataset.yaml in plugin_config eintragen wenn vorhanden
        // (YOLO-Plugins und Custom-Scripts können den Pfad direkt aus plugin_config lesen)
        let yaml_path = dataset_path.join("dataset.yaml");
        if yaml_path.exists() {
            let mut pc = final_config.plugin_config.as_object().cloned().unwrap_or_default();
            pc.entry("dataset_yaml_path".to_string()).or_insert_with(|| {
                serde_json::Value::String(yaml_path.to_string_lossy().to_string())
            });
            final_config.plugin_config = serde_json::Value::Object(pc);
            eprintln!("[Training] dataset.yaml -> plugin_config: {:?}", yaml_path);
        }
        
        final_config.model_path = model_path.to_string_lossy().to_string();
        final_config.dataset_path = dataset_path.to_string_lossy().to_string();
    }
    
    final_config.output_path   = output_dir.join("final_model").to_string_lossy().to_string();
    final_config.checkpoint_dir= checkpoint_dir.to_string_lossy().to_string();
    // task_type kommt vom Frontend – nur Fallback wenn leer
    if final_config.task_type.is_empty() {
        final_config.task_type = "seq_classification".to_string();
    }

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
        user_id: user_id.clone(),
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
            "INSERT INTO models (id, name, model_path, status, user_id, created_at, updated_at) VALUES (?1,?2,?3,?4,?5,?6,?7)",
            rusqlite::params![model_id, &unique_name, &mp, "trained", user_id, &now, &now],
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

    // Canvas-Modelle: die trainierten Gewichte zusaetzlich neben graph_metadata.json
    // legen. Der Inferenz-Tab im Synapse Builder sucht dort nach model.pt und
    // meldete sonst "(kein model.pt)" — auch direkt nach einem erfolgreichen Lauf.
    if model_id.starts_with("canvas_") || model_id.starts_with("synapse_") {
        let model_root = models_dir.join(model_id);
        for name in ["model.pt", "model_best.pt", "checkpoint.pt"] {
            let found = version_path.join(name);
            let found = if found.exists() { Some(found) } else {
                let nested = version_path.join("final_model").join(name);
                if nested.exists() { Some(nested) } else { None }
            };
            if let Some(src_pt) = found {
                match fs::copy(&src_pt, model_root.join("model.pt")) {
                    Ok(_)  => eprintln!("[Canvas] Gewichte -> {:?}", model_root.join("model.pt")),
                    Err(e) => eprintln!("[Canvas] Gewichte kopieren fehlgeschlagen: {}", e),
                }
                break;
            }
        }
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
    // Die Trainings-Engine liefert die Dauer innerhalb von "final_metrics"
    // (siehe MessageProtocol.complete). Früher wurde hier nur die oberste
    // Ebene gelesen – dadurch blieb die Spalte immer NULL und die UI zeigte
    // überall "-". Top-Level bleibt als Fallback für ältere Payloads.
    let duration   = metrics.get("training_duration_seconds").and_then(|v| v.as_i64())
        .or_else(|| data.get("training_duration_seconds").and_then(|v| v.as_i64()));

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

    let mut child = match Command::new(&python).no_window()
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

    // PID merken + Job als "running" markieren (vorher blieb der Status dauerhaft "pending")
    if let Ok(mut s) = state.lock() {
        s.process_pid = Some(child.id());
        if let Some(ref mut job) = s.current_job {
            job.status = TrainingStatus::Running;
            job.started_at = Some(Utc::now());
        }
    }
    let stderr_lines: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

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
    let mut json_complete = false;

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

                        // Progress auch im State mitführen — sonst liefert
                        // list_active_trainings (Polling-Fallback des globalen
                        // Widgets) dauerhaft veraltete Nullwerte.
                        if let Ok(mut sl) = state.lock() {
                            if let Some(ref mut job) = sl.current_job {
                                job.status = TrainingStatus::Running;
                                let p = &mut job.progress;
                                if let Some(v) = data.get("epoch").and_then(|v| v.as_u64()) { p.epoch = v as u32; }
                                if let Some(v) = data.get("total_epochs").and_then(|v| v.as_u64()) { p.total_epochs = v as u32; }
                                if let Some(v) = data.get("step").and_then(|v| v.as_u64()) { p.step = v as u32; }
                                if let Some(v) = data.get("total_steps").and_then(|v| v.as_u64()) { p.total_steps = v as u32; }
                                if let Some(v) = data.get("train_loss").and_then(|v| v.as_f64()) { p.train_loss = v; }
                                p.val_loss = data.get("val_loss").and_then(|v| v.as_f64()).or(p.val_loss);
                                if let Some(v) = data.get("learning_rate").and_then(|v| v.as_f64()) { p.learning_rate = v; }
                                if let Some(v) = data.get("progress_percent").and_then(|v| v.as_f64()) { p.progress_percent = v; }
                            }
                        }
                    }
                    let _ = ah.emit("training-progress", serde_json::json!({"job_id":jid,"data":msg.get("data")}));
                }
                "status"   => { let _ = ah.emit("training-status",   serde_json::json!({"job_id":jid,"data":msg.get("data")})); }
                "checkpoint"=>{ let _ = ah.emit("training-checkpoint",serde_json::json!({"job_id":jid,"data":msg.get("data")})); }
                "complete" => {
                    json_complete = true;
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

    // Bewusster Stop: stop_training setzt current_job auf None BEVOR es killt.
    // Ohne diesen Guard würde der Kill hier als "Training unerwartet beendet"
    // gemeldet und der "Gestoppt"-Status im UI mit "Fehlgeschlagen" überschrieben.
    let was_stopped = state.lock().ok()
        .map(|sl| sl.current_job.is_none())
        .unwrap_or(false);

    if !ok && !json_error && !was_stopped {
        let stderr_ctx = stderr_lines.lock().ok()
            .map(|v| if v.is_empty() { String::new() } else { format!("\n\nStderr:\n{}", v.join("\n")) })
            .unwrap_or_default();
        let _ = app_handle.emit("training-error", serde_json::json!({
            "job_id": job_id,
            "data": { "error": "Training unerwartet beendet", "details": format!("Exit: {:?}{}", status.as_ref().map(|s| s.code()), stderr_ctx) }
        }));
    }

    // Exit 0 OHNE complete/error-Event: die Engine ist still gestorben (z. B.
    // extern gekillt oder App-Neustart während tauri dev). Vorher bekam das
    // Frontend dann NIE ein Event und das Dashboard blieb ewig auf "läuft".
    if ok && !json_error && !json_complete && !was_stopped {
        let _ = app_handle.emit("training-error", serde_json::json!({
            "job_id": job_id,
            "data": {
                "error": "Training ohne Ergebnis beendet",
                "details": "Der Trainingsprozess hat sich beendet, ohne ein Ergebnis oder einen Fehler zu melden.\nMögliche Ursachen: Prozess wurde extern beendet (z. B. App-Neustart, System) oder die Engine wurde unterbrochen.\nStarte das Training danach erneut."
            }
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

    // Job-Info vor dem Kill sichern
    let job_snapshot = sl.current_job.clone();

    if let Some(ref mut p) = sl.process { let _ = p.kill(); }
    if let Some(pid) = sl.process_pid {
        #[cfg(unix)] {
            let _ = Command::new("kill").no_window().args(["-TERM", &pid.to_string()]).output();
            thread::sleep(std::time::Duration::from_millis(300));
            // Kinder VOR dem Parent killen — nach dem Parent-Kill werden sie
            // an launchd/init umgehängt und pkill -P findet sie nicht mehr.
            let _ = Command::new("pkill").no_window().args(["-KILL","-P",&pid.to_string()]).output();
            let _ = Command::new("kill").no_window().args(["-KILL", &pid.to_string()]).output();
        }
        #[cfg(windows)] { let _ = Command::new("taskkill").no_window().args(["/F","/PID",&pid.to_string(),"/T"]).output(); }
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

    // ── Neuesten Checkpoint als Version registrieren ─────────────────────────────────
    // Nach dem Kill pruefen ob HF Trainer bereits Checkpoints geschrieben hat.
    // Neuesten Checkpoint-Ordner finden und als gestoppte Version speichern.
    if let Some(job) = job_snapshot {
        let checkpoint_dir = PathBuf::from(&job.config.checkpoint_dir);
        if checkpoint_dir.exists() {
            // Suche nach checkpoint-{N}-Ordnern, nimm den mit der hoechsten Schritt-Nummer
            let best_checkpoint = fs::read_dir(&checkpoint_dir)
                .ok()
                .into_iter()
                .flatten()
                .flatten()
                .filter_map(|entry| {
                    let p = entry.path();
                    if !p.is_dir() { return None; }
                    let name = p.file_name()?.to_str()?.to_string();
                    if !name.starts_with("checkpoint-") { return None; }
                    let step: u64 = name.strip_prefix("checkpoint-")?.parse().ok()?;
                    Some((step, p))
                })
                .max_by_key(|(step, _)| *step)
                .map(|(_, path)| path);

            if let Some(ckpt_path) = best_checkpoint {
                // Modell-Dateien aus Checkpoint exportieren (HF-Format: pytorch_model.bin / model.safetensors + config.json)
                let has_model = ckpt_path.join("pytorch_model.bin").exists()
                    || ckpt_path.join("model.safetensors").exists()
                    || ckpt_path.join("model.pt").exists();

                if has_model {
                    let ah = app_handle.clone();
                    let model_id = job.model_id.clone();
                    let model_name = job.model_name.clone();
                    let user_id = job.user_id.clone();
                    let ckpt_str = ckpt_path.to_string_lossy().to_string();
                    let job_id = job.id.clone();

                    // In separatem Thread damit stop_training sofort zurückgibt
                    thread::spawn(move || {
                        match create_version(&ah, &model_id, &model_name, None, &ckpt_str, &user_id) {
                            Ok(version_id) => {
                                eprintln!("[Train] Stop-Checkpoint als Version registriert: {}", version_id);
                                let _ = ah.emit("training-stopped-with-checkpoint", serde_json::json!({
                                    "job_id": job_id,
                                    "version_id": version_id,
                                    "checkpoint_path": ckpt_str,
                                    "message": "Training gestoppt. Letzter Checkpoint wurde als Version gespeichert.",
                                }));
                            }
                            Err(e) => {
                                eprintln!("[Train] Stop-Checkpoint konnte nicht gespeichert werden: {}", e);
                                let _ = ah.emit("training-stopped", serde_json::json!({
                                    "job_id": job_id,
                                    "message": "Training gestoppt. Kein Checkpoint gespeichert.",
                                }));
                            }
                        }
                    });
                } else {
                    eprintln!("[Train] Stop: Checkpoint-Ordner gefunden aber keine Modell-Dateien: {:?}", ckpt_path);
                    let _ = app_handle.emit("training-stopped", serde_json::json!({
                        "job_id": job.id,
                        "message": "Training gestoppt. Kein vollst\u{00e4}ndiger Checkpoint vorhanden.",
                    }));
                }
            } else {
                eprintln!("[Train] Stop: Kein Checkpoint-Ordner gefunden in {:?}", checkpoint_dir);
                let _ = app_handle.emit("training-stopped", serde_json::json!({
                    "job_id": job.id,
                    "message": "Training gestoppt. Es wurde noch kein Checkpoint geschrieben.",
                }));
            }
        }
    }

    Ok(())
}

#[tauri::command]
pub fn get_current_training(state: tauri::State<'_, Arc<Mutex<TrainingState>>>) -> Result<Option<TrainingJob>, String> {
    Ok(state.lock().map_err(|e| format!("Lock: {}", e))?.current_job.clone())
}

/// Liste der aktuell laufenden Trainings (für das globale Progress-Widget).
/// Format entspricht dem ActiveTraining-Interface im Frontend.
#[tauri::command]
pub fn list_active_trainings(state: tauri::State<'_, Arc<Mutex<TrainingState>>>) -> Result<Vec<serde_json::Value>, String> {
    let sl = state.lock().map_err(|e| format!("Lock: {}", e))?;
    let Some(job) = sl.current_job.as_ref() else { return Ok(vec![]) };
    if !matches!(job.status, TrainingStatus::Pending | TrainingStatus::Running) {
        return Ok(vec![]);
    }
    let p = &job.progress;
    let elapsed = job.started_at.or(Some(job.created_at))
        .map(|t| (Utc::now() - t).num_seconds().max(0))
        .unwrap_or(0);
    let status = match job.status {
        TrainingStatus::Pending => "pending", TrainingStatus::Running => "running",
        TrainingStatus::Completed => "completed", TrainingStatus::Failed => "failed",
        TrainingStatus::Stopped => "stopped",
    };
    Ok(vec![serde_json::json!({
        "training_id": job.id,
        "status": status,
        "current_epoch": p.epoch,
        "total_epochs": p.total_epochs,
        "current_step": p.step,
        "total_steps": p.total_steps,
        "progress_percentage": p.progress_percent,
        "train_loss": p.train_loss,
        "val_loss": p.val_loss,
        "learning_rate": p.learning_rate,
        "elapsed_time_seconds": elapsed,
        "estimated_time_remaining_seconds": null,
    })])
}

#[tauri::command]
pub fn get_training_history(
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, crate::AppState>,
) -> Result<Vec<TrainingJob>, String> {
    let current_user_id = {
        let db = state.db.lock().map_err(|e| format!("Lock: {}", e))?;
        db.get_current_user_id()
    };
    let mut jobs = load_jobs(&app_handle)?;
    match current_user_id {
        Some(uid) => {
            // Migration: alte Jobs ohne user_id beim ersten Zugriff dem aktuellen User zuweisen
            let mut changed = false;
            for job in jobs.iter_mut() {
                if job.user_id.is_empty() {
                    job.user_id = uid.clone();
                    changed = true;
                }
            }
            if changed {
                let _ = write_jobs(&app_handle, &jobs);
            }
            Ok(jobs.into_iter().filter(|j| j.user_id == uid).collect())
        }
        None => Ok(vec![]),
    }
}

#[tauri::command]
pub fn delete_training_job(
    app_handle: tauri::AppHandle,
    job_id: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<(), String> {
    // FIX: Nur eigene Jobs löschen dürfen
    let current_user_id = {
        let db = state.db.lock().map_err(|e| format!("Lock: {}", e))?;
        db.get_current_user_id()
    };
    // FIX: job_id validieren (Path-Traversal) und Output-Ordner nur löschen,
    // wenn der Job tatsächlich dem User gehörte und entfernt wurde.
    if job_id.is_empty() || !job_id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
        return Err("Ungültige Job-ID".to_string());
    }
    let mut jobs = load_jobs(&app_handle)?;
    let before = jobs.len();
    jobs.retain(|j| {
        if j.id != job_id { return true; }
        // Job gehört dem User (oder ist ein alter Job ohne user_id)
        match &current_user_id {
            Some(uid) => j.user_id != *uid && !j.user_id.is_empty(),
            None => true,
        }
    });
    let removed = jobs.len() < before;
    write_jobs(&app_handle, &jobs)?;
    if removed {
        let out = app_handle.path().app_data_dir().map_err(|e| format!("AppDataDir: {}", e))?
            .join("training_outputs").join(&job_id);
        if out.exists() { fs::remove_dir_all(&out).ok(); }
    }
    Ok(())
}

#[tauri::command]
pub fn get_system_ram_gb() -> f64 {
    #[cfg(target_os = "macos")] {
        if let Ok(out) = Command::new("sysctl").no_window().args(["-n","hw.memsize"]).output() {
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

    let py_out = Command::new(&python).no_window().arg("--version").output();
    let py_ok  = py_out.is_ok() && py_out.as_ref().unwrap().status.success();
    let py_ver = if py_ok { String::from_utf8_lossy(&py_out.unwrap().stdout).trim().to_string() } else { "Nicht gefunden".to_string() };

    let torch_out = Command::new(&python).no_window().args(["-c","import torch; print(torch.__version__)"]).output();
    let torch_ok  = torch_out.is_ok() && torch_out.as_ref().unwrap().status.success();
    let torch_ver = if torch_ok { String::from_utf8_lossy(&torch_out.unwrap().stdout).trim().to_string() } else { "Nicht installiert".to_string() };

    let cuda = Command::new(&python).no_window().args(["-c","import torch; print(torch.cuda.is_available())"]).output();
    let cuda_ok = cuda.is_ok() && String::from_utf8_lossy(&cuda.unwrap().stdout).trim() == "True";

    let mps = Command::new(&python).no_window().args(["-c","import torch; print(hasattr(torch.backends,'mps') and torch.backends.mps.is_available())"]).output();
    let mps_ok = mps.is_ok() && String::from_utf8_lossy(&mps.unwrap().stdout).trim() == "True";

    let tf_out = Command::new(&python).no_window().args(["-c","import transformers; print(transformers.__version__)"]).output();
    let tf_ok  = tf_out.is_ok() && tf_out.as_ref().unwrap().status.success();
    let tf_ver = if tf_ok { String::from_utf8_lossy(&tf_out.unwrap().stdout).trim().to_string() } else { "Nicht installiert".to_string() };

    let peft_out = Command::new(&python).no_window().args(["-c","import peft; print(peft.__version__)"]).output();
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

fn templates_path(app_handle: &tauri::AppHandle, user_id: &str) -> Result<PathBuf, String> {
    app_handle.path().app_data_dir().map_err(|e| format!("AppDataDir: {}", e)).map(|d| d.join(format!("metrics_templates_{}.json", user_id)))
}

#[tauri::command]
pub fn save_metrics_template(
    app_handle: tauri::AppHandle,
    name: String,
    description: String,
    config: TrainingConfig,
    source: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<MetricsTemplate, String> {
    let user_id = {
        let db = state.db.lock().map_err(|e| format!("Lock: {}", e))?;
        db.get_current_user_id().unwrap_or_else(|| "default".to_string())
    };
    let path = templates_path(&app_handle, &user_id)?;
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
pub fn get_metrics_templates(
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, crate::AppState>,
) -> Result<Vec<MetricsTemplate>, String> {
    let user_id = {
        let db = state.db.lock().map_err(|e| format!("Lock: {}", e))?;
        db.get_current_user_id().unwrap_or_else(|| "default".to_string())
    };
    let path = templates_path(&app_handle, &user_id)?;
    if !path.exists() { return Ok(vec![]); }
    serde_json::from_str(&fs::read_to_string(&path).map_err(|e| format!("Read: {}", e))?).map_err(|e| format!("JSON: {}", e))
}

#[tauri::command]
pub fn delete_metrics_template(
    app_handle: tauri::AppHandle,
    template_id: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<(), String> {
    let user_id = {
        let db = state.db.lock().map_err(|e| format!("Lock: {}", e))?;
        db.get_current_user_id().unwrap_or_else(|| "default".to_string())
    };
    let path = templates_path(&app_handle, &user_id)?;
    if !path.exists() { return Ok(()); }
    let mut templates: Vec<MetricsTemplate> = serde_json::from_str(&fs::read_to_string(&path).unwrap_or_default()).unwrap_or_default();
    templates.retain(|t| t.id != template_id);
    fs::write(&path, serde_json::to_string_pretty(&templates).unwrap_or_default()).map_err(|e| format!("Write: {}", e))
}

// ─── Synapse Builder: register trained canvas output as a FrameTrain version ───

#[derive(Debug, Serialize, Deserialize)]
pub struct SynapseVersionResult {
    pub version_id: String,
    pub model_id: String,
}

/// Copies Synapse training output (model.pt, metrics.json, …) into the model library.
#[tauri::command]
pub async fn register_synapse_training_version(
    app_handle: tauri::AppHandle,
    model_name: String,
    output_dir: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<SynapseVersionResult, String> {
    let user_id = {
        let db = state.db.lock().map_err(|e| format!("DB Lock: {}", e))?;
        db.get_current_user_id().ok_or_else(|| "Kein User eingeloggt".to_string())?
    };
    let model_id = format!(
        "synapse_{}",
        &uuid::Uuid::new_v4().to_string().replace('-', "")[..12]
    );
    let version_id = create_version(
        &app_handle,
        &model_id,
        &model_name,
        None,
        &output_dir,
        &user_id,
    )?;
    Ok(SynapseVersionResult { version_id, model_id })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CanvasNetworkResult {
    pub model_id: String,
    pub path: String,
}

#[tauri::command]
pub async fn create_canvas_network_model(
    app_handle: tauri::AppHandle,
    model_name: String,
    metadata: String,
    python_code: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<CanvasNetworkResult, String> {
    let user_id = {
        let db = state.db.lock().map_err(|e| format!("DB Lock: {}", e))?;
        db.get_current_user_id().ok_or_else(|| "Kein User eingeloggt".to_string())?
    };
    let model_id = format!(
        "canvas_{}",
        &uuid::Uuid::new_v4().to_string().replace('-', "")[..12]
    );
    let models_dir = get_models_dir(&app_handle)?;
    let model_path = models_dir.join(&model_id);
    fs::create_dir_all(&model_path).map_err(|e| format!("mkdir: {}", e))?;
    fs::write(model_path.join("canvas_model.py"), &python_code)
        .map_err(|e| format!("canvas_model.py: {}", e))?;
    fs::write(model_path.join("graph_metadata.json"), &metadata)
        .map_err(|e| format!("graph_metadata.json: {}", e))?;

    let db_path = app_handle.path().app_data_dir()
        .map_err(|e| format!("AppDataDir: {}", e))?
        .join("frametrain.db");
    if let Ok(conn) = rusqlite::Connection::open(&db_path) {
        let now = Utc::now().to_rfc3339();
        let unique_name = format!("{} ({})", model_name, &model_id[..8.min(model_id.len())]);
        conn.execute(
            "INSERT OR IGNORE INTO models (id, name, model_path, status, user_id, created_at, updated_at) VALUES (?1,?2,?3,?4,?5,?6,?7)",
            rusqlite::params![
                &model_id,
                &unique_name,
                model_path.to_string_lossy().to_string(),
                "canvas",
                &user_id,
                &now,
                &now
            ],
        ).ok();
        let _ = conn.execute(
            "INSERT OR IGNORE INTO model_versions_new (id,model_id,version_name,version_number,path,size_bytes,file_count,created_at,is_root,parent_version_id,user_id) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
            rusqlite::params![
                format!("ver_{}", &uuid::Uuid::new_v4().to_string().replace('-', "")[..12]),
                &model_id,
                "Canvas v1",
                1i32,
                model_path.to_string_lossy().to_string(),
                0i64,
                0i32,
                &now,
                1i32,
                Option::<String>::None,
                &user_id
            ],
        );
    }

    Ok(CanvasNetworkResult {
        model_id: model_id.clone(),
        path: model_path.to_string_lossy().to_string(),
    })
}

#[tauri::command]
pub async fn update_canvas_network_model(
    app_handle: tauri::AppHandle,
    model_id: String,
    metadata: String,
    python_code: String,
) -> Result<(), String> {
    let models_dir = get_models_dir(&app_handle)?;
    let model_path = models_dir.join(&model_id);
    if !model_path.exists() {
        return Err(format!("Modell nicht gefunden: {}", model_id));
    }
    fs::write(model_path.join("canvas_model.py"), &python_code)
        .map_err(|e| format!("canvas_model.py: {}", e))?;
    fs::write(model_path.join("graph_metadata.json"), &metadata)
        .map_err(|e| format!("graph_metadata.json: {}", e))?;
    // updated_at in DB aktualisieren
    let db_path = app_handle.path().app_data_dir()
        .map_err(|e| format!("AppDataDir: {}", e))?.join("frametrain.db");
    if let Ok(conn) = rusqlite::Connection::open(&db_path) {
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE models SET updated_at = ?1 WHERE id = ?2",
            rusqlite::params![&now, &model_id],
        ).ok();
    }
    eprintln!("[Canvas] ✓ Modell {} aktualisiert", model_id);
    Ok(())
}

#[tauri::command]
pub async fn is_canvas_network_model(model_id: String) -> Result<bool, String> {
    Ok(model_id.starts_with("canvas_") || model_id.starts_with("synapse_"))
}

#[tauri::command]
pub async fn get_canvas_network_code(
    app_handle: tauri::AppHandle,
    model_id: String,
) -> Result<String, String> {
    let models_dir = get_models_dir(&app_handle)?;
    let code_path = models_dir.join(&model_id).join("canvas_model.py");
    if !code_path.exists() {
        return Err(format!("Canvas-Code nicht gefunden: {}", code_path.display()));
    }
    fs::read_to_string(&code_path).map_err(|e| format!("Lesen: {}", e))
}

// ─── Canvas Inference ─────────────────────────────────────────────────────────

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct CanvasModelInfo {
    pub model_id: String,
    pub name: String,
    pub has_weights: bool,
    pub model_pt_path: String,
    pub metadata_path: String,
    pub task_type: String,
    pub num_classes: i64,
}

/// Listet alle Canvas-Modelle auf, die ein model.pt haben (inferenzbereit).
#[tauri::command]
pub async fn list_canvas_models_with_pt(
    app_handle: tauri::AppHandle,
    user_id: String,
) -> Result<Vec<CanvasModelInfo>, String> {
    let models_dir = get_models_dir(&app_handle)?;
    let db_path = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("AppDataDir: {}", e))?
        .join("frametrain.db");

    let conn = rusqlite::Connection::open(&db_path)
        .map_err(|e| format!("DB: {}", e))?;

    // Alle Canvas-Modelle des Users laden
    let mut stmt = conn
        .prepare(
            "SELECT id, name FROM models WHERE user_id = ?1 AND (id LIKE 'canvas_%' OR id LIKE 'synapse_%')",
        )
        .map_err(|e| format!("Stmt: {}", e))?;

    let rows = stmt
        .query_map(rusqlite::params![&user_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(|e| format!("Query: {}", e))?;

    let mut result = Vec::new();
    for row in rows.flatten() {
        let (model_id, name) = row;
        let model_path = models_dir.join(&model_id);
        let metadata_path = model_path.join("graph_metadata.json");
        if !metadata_path.exists() {
            continue;
        }

        // Prüfen ob model.pt vorhanden
        let pt_candidates = ["model.pt", "model_best.pt", "checkpoint.pt"];
        let pt_path = pt_candidates
            .iter()
            .map(|n| model_path.join(n))
            .find(|p| p.exists());

        let has_weights = pt_path.is_some();
        let model_pt_path = pt_path
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();

        // task_type + num_classes aus graph_metadata.json lesen
        let (task_type, num_classes) = if let Ok(raw) = fs::read_to_string(&metadata_path) {
            if let Ok(meta) = serde_json::from_str::<serde_json::Value>(&raw) {
                let ir = meta.get("graphIR").or_else(|| meta.get("graph_ir"));
                let task = ir
                    .and_then(|v| v.get("training"))
                    .and_then(|v| v.get("taskType"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("classification")
                    .to_string();
                let classes = ir
                    .and_then(|v| v.get("training"))
                    .and_then(|v| v.get("numClasses"))
                    .and_then(|v| v.as_i64())
                    .unwrap_or(10);
                (task, classes)
            } else {
                ("classification".to_string(), 10)
            }
        } else {
            ("classification".to_string(), 10)
        };

        result.push(CanvasModelInfo {
            model_id,
            name,
            has_weights,
            model_pt_path,
            metadata_path: metadata_path.to_string_lossy().to_string(),
            task_type,
            num_classes,
        });
    }

    Ok(result)
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct CanvasInferenceResult {
    pub predicted_class: Option<i64>,
    pub confidence: Option<f64>,
    pub predicted_value: Option<serde_json::Value>,
    pub top_predictions: Option<Vec<serde_json::Value>>,
    pub all_probs: Option<Vec<f64>>,
    pub inference_ms: f64,
    pub task_type: String,
    pub error: Option<String>,
}

/// Führt eine einzelne Inferenz auf einem Canvas-Modell durch.
/// Spawnt canvas_inference_server.py, sendet Input, liest Result, beendet Prozess.
#[tauri::command]
pub async fn run_canvas_inference(
    app_handle: tauri::AppHandle,
    model_id: String,
    input: Vec<f64>,
) -> Result<CanvasInferenceResult, String> {
    let models_dir = get_models_dir(&app_handle)?;
    let model_dir = models_dir.join(&model_id);
    if !model_dir.exists() {
        return Err(format!("Modell-Ordner nicht gefunden: {}", model_id));
    }

    let python_path = get_python_path();
    let server_script = get_train_engine_path(&app_handle)?
        .parent()
        .ok_or_else(|| "train_engine.py hat kein Parent-Verzeichnis".to_string())?
        .join("plugins")
        .join("canvas")
        .join("canvas_inference_server.py");

    if !server_script.exists() {
        return Err(format!(
            "canvas_inference_server.py nicht gefunden: {}",
            server_script.display()
        ));
    }

    use std::io::Write;
    use std::process::{Command, Stdio};

    let mut child = Command::new(&python_path).no_window()
        .arg(&server_script)
        .arg("--model-dir")
        .arg(model_dir.to_string_lossy().as_ref())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Python spawn: {}", e))?;

    let stdin = child.stdin.as_mut().ok_or("Kein stdin")?;
    let stdout = child.stdout.take().ok_or("Kein stdout")?;

    // Warte auf "ready" Signal
    use std::io::BufRead;
    let mut reader = std::io::BufReader::new(stdout);
    let mut ready_line = String::new();
    reader
        .read_line(&mut ready_line)
        .map_err(|e| format!("Ready lesen: {}", e))?;

    let ready: serde_json::Value = serde_json::from_str(ready_line.trim())
        .map_err(|e| format!("Ready parse: {} | raw: {}", e, ready_line.trim()))?;

    if ready.get("type").and_then(|v| v.as_str()) == Some("error") {
        let _ = child.kill();
        return Err(ready
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("Unbekannter Fehler beim Laden")
            .to_string()
            .into());
    }

    // Input senden
    let req = serde_json::json!({ "input": input, "input_type": "tensor" });
    writeln!(stdin, "{}", req).map_err(|e| format!("Stdin write: {}", e))?;

    // Ergebnis lesen
    let mut result_line = String::new();
    reader
        .read_line(&mut result_line)
        .map_err(|e| format!("Result lesen: {}", e))?;

    // Shutdown
    let _ = writeln!(stdin, "{{\"cmd\": \"shutdown\"}}");
    let _ = child.wait();

    let val: serde_json::Value = serde_json::from_str(result_line.trim())
        .map_err(|e| format!("Result parse: {} | raw: {}", e, result_line.trim()))?;

    if val.get("type").and_then(|v| v.as_str()) == Some("error") {
        return Ok(CanvasInferenceResult {
            predicted_class: None,
            confidence: None,
            predicted_value: None,
            top_predictions: None,
            all_probs: None,
            inference_ms: 0.0,
            task_type: "classification".to_string(),
            error: val.get("message").and_then(|v| v.as_str()).map(|s| s.to_string()),
        });
    }

    Ok(CanvasInferenceResult {
        predicted_class: val.get("predicted_class").and_then(|v| v.as_i64()),
        confidence: val.get("confidence").and_then(|v| v.as_f64()),
        predicted_value: val.get("predicted_value").cloned(),
        top_predictions: val
            .get("top_predictions")
            .and_then(|v| v.as_array())
            .map(|a| a.clone()),
        all_probs: val
            .get("all_probs")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_f64()).collect()),
        inference_ms: val.get("inference_ms").and_then(|v| v.as_f64()).unwrap_or(0.0),
        task_type: val
            .get("task_type")
            .and_then(|v| v.as_str())
            .unwrap_or("classification")
            .to_string(),
        error: None,
    })
}
