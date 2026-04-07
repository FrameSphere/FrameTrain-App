use std::fs;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::io::{BufRead, BufReader};
use std::thread;
use serde::{Deserialize, Serialize};
use tauri::{Emitter, Manager};
use chrono::{DateTime, Utc};

/// Training Job Status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum TrainingStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Stopped,
}

/// Training Job Info
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

/// Training Progress
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

/// Training Configuration (mirrors Python config)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    // Paths (will be filled by backend)
    #[serde(default)]
    pub model_path: String,
    #[serde(default)]
    pub dataset_path: String,
    #[serde(default)]
    pub output_path: String,
    #[serde(default)]
    pub checkpoint_dir: String,
    
    // Training Basics
    #[serde(default = "default_epochs")]
    pub epochs: u32,
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,
    #[serde(default = "default_one")]
    pub gradient_accumulation_steps: u32,
    #[serde(default = "default_minus_one")]
    pub max_steps: i32,
    
    // Learning Rate
    #[serde(default = "default_lr")]
    pub learning_rate: f64,
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,
    #[serde(default)]
    pub warmup_steps: u32,
    #[serde(default)]
    pub warmup_ratio: f64,
    
    // Optimizer
    #[serde(default = "default_optimizer")]
    pub optimizer: String,
    #[serde(default = "default_beta1")]
    pub adam_beta1: f64,
    #[serde(default = "default_beta2")]
    pub adam_beta2: f64,
    #[serde(default = "default_epsilon")]
    pub adam_epsilon: f64,
    #[serde(default = "default_momentum")]
    pub sgd_momentum: f64,
    
    // Scheduler
    #[serde(default = "default_scheduler")]
    pub scheduler: String,
    #[serde(default = "default_one")]
    pub scheduler_step_size: u32,
    #[serde(default = "default_gamma")]
    pub scheduler_gamma: f64,
    #[serde(default)]
    pub cosine_min_lr: f64,
    
    // Regularization
    #[serde(default = "default_dropout")]
    pub dropout: f64,
    #[serde(default = "default_grad_norm")]
    pub max_grad_norm: f64,
    #[serde(default)]
    pub label_smoothing: f64,
    
    // Mixed Precision
    #[serde(default)]
    pub fp16: bool,
    #[serde(default)]
    pub bf16: bool,
    
    // LoRA / PEFT
    #[serde(default)]
    pub use_lora: bool,
    #[serde(default = "default_lora_r")]
    pub lora_r: u32,
    #[serde(default = "default_lora_alpha")]
    pub lora_alpha: u32,
    #[serde(default = "default_dropout")]
    pub lora_dropout: f64,
    #[serde(default = "default_lora_modules")]
    pub lora_target_modules: Vec<String>,
    
    // Quantization
    #[serde(default)]
    pub load_in_8bit: bool,
    #[serde(default)]
    pub load_in_4bit: bool,
    
    // Data
    #[serde(default = "default_seq_length")]
    pub max_seq_length: u32,
    #[serde(default = "default_workers")]
    pub num_workers: u32,
    #[serde(default = "default_true")]
    pub pin_memory: bool,
    
    // Evaluation
    #[serde(default = "default_eval_steps")]
    pub eval_steps: u32,
    #[serde(default = "default_strategy")]
    pub eval_strategy: String,
    #[serde(default = "default_eval_steps")]
    pub save_steps: u32,
    #[serde(default = "default_strategy")]
    pub save_strategy: String,
    #[serde(default = "default_save_limit")]
    pub save_total_limit: u32,
    
    // Logging
    #[serde(default = "default_logging_steps")]
    pub logging_steps: u32,
    
    // Advanced
    #[serde(default = "default_seed")]
    pub seed: u32,
    #[serde(default)]
    pub dataloader_drop_last: bool,
    #[serde(default)]
    pub group_by_length: bool,
    
    // Training Type
    #[serde(default = "default_training_type")]
    pub training_type: String,
    #[serde(default = "default_task_type")]
    pub task_type: String,
}

// Default value functions
fn default_epochs() -> u32 { 3 }
fn default_batch_size() -> u32 { 8 }
fn default_one() -> u32 { 1 }
fn default_minus_one() -> i32 { -1 }
fn default_lr() -> f64 { 5e-5 }
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
fn default_lora_modules() -> Vec<String> { vec!["q_proj".to_string(), "v_proj".to_string()] }
fn default_seq_length() -> u32 { 512 }
fn default_workers() -> u32 { 4 }
fn default_true() -> bool { true }
fn default_eval_steps() -> u32 { 500 }
fn default_strategy() -> String { "steps".to_string() }
fn default_save_limit() -> u32 { 3 }
fn default_logging_steps() -> u32 { 100 }
fn default_seed() -> u32 { 42 }
fn default_training_type() -> String { "fine_tuning".to_string() }
fn default_task_type() -> String { "causal_lm".to_string() }

/// Preset Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetConfig {
    pub id: String,
    pub name: String,
    pub description: String,
    pub config: TrainingConfig,
}

/// Parameter Rating Response
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

/// Global state for running training processes
pub struct TrainingState {
    pub current_job: Option<TrainingJob>,
    pub process: Option<Child>,
    pub jobs_history: Vec<TrainingJob>,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            current_job: None,
            process: None,
            jobs_history: Vec::new(),
        }
    }
}

// ============ Helper Functions ============

fn get_python_path() -> String {
    println!("[Python] 🔍 ROBUST SEARCH: Finding best Python installation...");
    
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
                            println!("[Python] ✅ Found: {} -> v{}.{}.{}", full_path, version.0, version.1, version.2);
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
                                println!("[Python] ✅ Found via 'which': {} -> v{}.{}.{}", which_path, version.0, version.1, version.2);
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
                    println!("[Python] ✅ Found via command: {} -> v{}.{}.{}", cmd, version.0, version.1, version.2);
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
        println!("[Python] 🎯 SELECTED: {} (v{}.{}.{})", best.path, best.version.0, best.version.1, best.version.2);
        println!("[Python] 📊 All candidates found: {:?}", candidates);
        return best.path.clone();
    }
    
    // Fallback
    println!("[Python] ⚠️  WARNING: No Python found! Using fallback.");
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

fn get_training_engine_path(app_handle: &tauri::AppHandle) -> Result<PathBuf, String> {
    // NEW: Updated path for train_engine folder structure
    let resource_path = app_handle
        .path()
        .resource_dir()
        .map_err(|e| format!("Could not get resource dir: {}", e))?;
    
    println!("[Engine] Resource path: {:?}", resource_path);
    
    // Try new structure: python/train_engine/train_engine.py
    let engine_path = resource_path.join("python").join("train_engine").join("train_engine.py");
    println!("[Engine] Trying path 1: {:?}", engine_path);
    
    if engine_path.exists() {
        println!("[Engine] ✅ Found at path 1");
        return Ok(engine_path);
    }
    
    // Fallback: old structure python/train_engine.py
    let engine_path_old = resource_path.join("python").join("train_engine.py");
    println!("[Engine] Trying path 2: {:?}", engine_path_old);
    if engine_path_old.exists() {
        println!("[Engine] ✅ Found at path 2");
        return Ok(engine_path_old);
    }
    
    // Fallback: Development - relative to src-tauri
    let local_path = PathBuf::from("src-tauri/python/train_engine/train_engine.py");
    println!("[Engine] Trying path 3: {:?}", local_path);
    if local_path.exists() {
        println!("[Engine] ✅ Found at path 3");
        return Ok(local_path);
    }
    
    // Last fallback: old development path
    let local_path_old = PathBuf::from("src-tauri/python/train_engine.py");
    println!("[Engine] Trying path 4: {:?}", local_path_old);
    if local_path_old.exists() {
        println!("[Engine] ✅ Found at path 4");
        return Ok(local_path_old);
    }
    
    // ABSOLUTE PATH FALLBACK for development
    let absolute_dev_path = PathBuf::from("/Users/karol/Desktop/Laufende_Projekte/FrameTrain/desktop-app2/src-tauri/python/train_engine/train_engine.py");
    println!("[Engine] Trying path 5 (absolute): {:?}", absolute_dev_path);
    if absolute_dev_path.exists() {
        println!("[Engine] ✅ Found at path 5 (absolute dev path)");
        return Ok(absolute_dev_path);
    }
    
    Err("Training engine not found in any location".to_string())
}

fn get_models_dir(app_handle: &tauri::AppHandle) -> Result<PathBuf, String> {
    let data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Could not get app data dir: {}", e))?;
    Ok(data_dir.join("models"))
}

fn get_training_output_dir(app_handle: &tauri::AppHandle, job_id: &str) -> Result<PathBuf, String> {
    let data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Could not get app data dir: {}", e))?;
    
    let output_dir = data_dir.join("training_outputs").join(job_id);
    fs::create_dir_all(&output_dir)
        .map_err(|e| format!("Could not create output dir: {}", e))?;
    
    Ok(output_dir)
}

fn save_training_jobs(app_handle: &tauri::AppHandle, jobs: &[TrainingJob]) -> Result<(), String> {
    let data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Could not get app data dir: {}", e))?;
    
    let jobs_path = data_dir.join("training_jobs.json");
    let content = serde_json::to_string_pretty(jobs)
        .map_err(|e| format!("Could not serialize jobs: {}", e))?;
    
    fs::write(&jobs_path, content)
        .map_err(|e| format!("Could not save jobs: {}", e))?;
    
    Ok(())
}

fn load_training_jobs(app_handle: &tauri::AppHandle) -> Result<Vec<TrainingJob>, String> {
    let data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Could not get app data dir: {}", e))?;
    
    let jobs_path = data_dir.join("training_jobs.json");
    
    if !jobs_path.exists() {
        return Ok(Vec::new());
    }
    
    let content = fs::read_to_string(&jobs_path)
        .map_err(|e| format!("Could not read jobs: {}", e))?;
    
    let jobs: Vec<TrainingJob> = serde_json::from_str(&content)
        .map_err(|e| format!("Could not parse jobs: {}", e))?;
    
    Ok(jobs)
}

// ============ TAURI COMMANDS ============

/// Gibt verfügbare Presets zurück
#[tauri::command]
pub fn get_training_presets() -> Result<Vec<PresetConfig>, String> {
    let presets = vec![
        PresetConfig {
            id: "llm_qlora_efficient".to_string(),
            name: "LLM QLoRA (Speichereffizient)".to_string(),
            description: "Optimiert für Fine-Tuning großer Sprachmodelle mit wenig VRAM".to_string(),
            config: TrainingConfig {
                learning_rate: 2e-4,
                batch_size: 4,
                gradient_accumulation_steps: 4,
                epochs: 3,
                optimizer: "adamw".to_string(),
                scheduler: "cosine".to_string(),
                warmup_ratio: 0.03,
                weight_decay: 0.01,
                use_lora: true,
                lora_r: 16,
                lora_alpha: 32,
                lora_dropout: 0.05,
                load_in_4bit: true,
                fp16: true,
                max_grad_norm: 0.3,
                max_seq_length: 512,
                ..Default::default()
            },
        },
        PresetConfig {
            id: "llm_lora_balanced".to_string(),
            name: "LLM LoRA (Ausgewogen)".to_string(),
            description: "Gute Balance zwischen Qualität und Geschwindigkeit".to_string(),
            config: TrainingConfig {
                learning_rate: 1e-4,
                batch_size: 8,
                gradient_accumulation_steps: 2,
                epochs: 3,
                optimizer: "adamw".to_string(),
                scheduler: "cosine".to_string(),
                warmup_ratio: 0.05,
                weight_decay: 0.01,
                use_lora: true,
                lora_r: 32,
                lora_alpha: 64,
                lora_dropout: 0.1,
                fp16: true,
                max_grad_norm: 1.0,
                max_seq_length: 1024,
                ..Default::default()
            },
        },
        PresetConfig {
            id: "classification_standard".to_string(),
            name: "Text-Klassifikation".to_string(),
            description: "Für Sentiment, Topic Classification etc.".to_string(),
            config: TrainingConfig {
                learning_rate: 5e-5,
                batch_size: 16,
                epochs: 5,
                optimizer: "adamw".to_string(),
                scheduler: "linear".to_string(),
                warmup_ratio: 0.1,
                weight_decay: 0.01,
                fp16: true,
                max_seq_length: 256,
                task_type: "seq_classification".to_string(),
                ..Default::default()
            },
        },
        PresetConfig {
            id: "quick_test".to_string(),
            name: "Schnelltest".to_string(),
            description: "Zum schnellen Testen ob alles funktioniert".to_string(),
            config: TrainingConfig {
                learning_rate: 1e-4,
                batch_size: 2,
                epochs: 1,
                optimizer: "adam".to_string(),
                scheduler: "constant".to_string(),
                max_seq_length: 128,
                logging_steps: 10,
                eval_steps: 50,
                ..Default::default()
            },
        },
        PresetConfig {
            id: "stable_conservative".to_string(),
            name: "Stabil & Konservativ".to_string(),
            description: "Langsames aber stabiles Training".to_string(),
            config: TrainingConfig {
                learning_rate: 1e-5,
                batch_size: 8,
                gradient_accumulation_steps: 4,
                epochs: 5,
                optimizer: "adamw".to_string(),
                scheduler: "cosine".to_string(),
                warmup_ratio: 0.1,
                weight_decay: 0.01,
                use_lora: true,
                lora_r: 8,
                lora_alpha: 16,
                lora_dropout: 0.1,
                fp16: true,
                max_grad_norm: 0.5,
                ..Default::default()
            },
        },
    ];
    
    Ok(presets)
}

/// Bewertet eine Konfiguration
#[tauri::command]
pub fn rate_training_config(config: TrainingConfig) -> Result<ParameterRating, String> {
    let mut score: i32 = 100;
    let mut issues = Vec::new();
    let mut warnings = Vec::new();
    let mut tips = Vec::new();
    
    // Learning Rate Check
    if config.learning_rate > 1e-3 {
        issues.push("Learning Rate ist sehr hoch (>1e-3). Das kann zu instabilem Training führen.".to_string());
        score -= 25;
    } else if config.learning_rate > 5e-4 {
        warnings.push("Learning Rate ist relativ hoch. Beobachte den Loss genau.".to_string());
        score -= 10;
    } else if config.learning_rate < 1e-6 {
        warnings.push("Learning Rate ist sehr niedrig. Training könnte sehr langsam sein.".to_string());
        score -= 10;
    } else {
        tips.push("Learning Rate ist im guten Bereich.".to_string());
    }
    
    // Batch Size Check
    if config.batch_size < 4 {
        warnings.push("Kleine Batch Size kann zu verrauschten Gradienten führen.".to_string());
        score -= 5;
    }
    if config.batch_size > 64 && !config.use_lora {
        warnings.push("Große Batch Size benötigt viel GPU-Speicher.".to_string());
        score -= 5;
    }
    
    // Epochs Check
    if config.epochs > 10 {
        warnings.push("Viele Epochen können zu Overfitting führen.".to_string());
        score -= 5;
    } else if config.epochs < 2 {
        warnings.push("Wenige Epochen könnten nicht ausreichen für gute Konvergenz.".to_string());
        score -= 5;
    }
    
    // Optimizer + LR Kombination
    if config.optimizer == "sgd" && config.learning_rate > 0.01 {
        warnings.push("SGD mit hoher LR kann instabil sein.".to_string());
        score -= 10;
    }
    if (config.optimizer == "adam" || config.optimizer == "adamw") && config.learning_rate > 1e-3 {
        warnings.push("Adam-Optimizer funktionieren meist besser mit LR < 1e-3.".to_string());
        score -= 10;
    }
    
    // LoRA Checks
    if config.use_lora {
        if config.lora_alpha < config.lora_r {
            warnings.push("LoRA Alpha sollte >= LoRA R sein für stabiles Training.".to_string());
            score -= 10;
        }
        if config.lora_r > 64 {
            warnings.push("Sehr hoher LoRA Rank kann zu Overfitting führen.".to_string());
            score -= 5;
        }
        tips.push("LoRA ist aktiviert - gute Wahl für effizientes Fine-Tuning!".to_string());
    }
    
    // 4-bit Quantization
    if config.load_in_4bit {
        if !config.use_lora {
            issues.push("4-bit Quantisierung sollte mit LoRA/QLoRA verwendet werden.".to_string());
            score -= 20;
        } else {
            tips.push("QLoRA-Setup erkannt - sehr speichereffizient!".to_string());
        }
    }
    
    // Warmup
    if config.warmup_ratio == 0.0 && config.epochs >= 3 {
        tips.push("Tipp: Ein Warmup von 0.03-0.1 kann Training stabilisieren.".to_string());
    } else if config.warmup_ratio > 0.2 {
        warnings.push("Sehr langes Warmup kann effektive Trainingszeit reduzieren.".to_string());
        score -= 5;
    }
    
    // Weight Decay
    if config.weight_decay > 0.1 {
        warnings.push("Hoher Weight Decay kann zu Underfitting führen.".to_string());
        score -= 10;
    }
    
    // Gradient Clipping
    if config.max_grad_norm > 5.0 {
        warnings.push("Hohes Gradient Clipping kann explodierende Gradienten nicht verhindern.".to_string());
        score -= 5;
    }
    
    // Determine rating
    let score = score.max(0).min(100) as u32;
    let (rating, label, color) = if score >= 90 {
        ("excellent", "Exzellent", "green")
    } else if score >= 75 {
        ("good", "Gut", "blue")
    } else if score >= 60 {
        ("okay", "Okay", "yellow")
    } else if score >= 40 {
        ("risky", "Riskant", "orange")
    } else {
        ("bad", "Schlecht", "red")
    };
    
    Ok(ParameterRating {
        score,
        rating: rating.to_string(),
        rating_info: RatingInfo {
            score: match rating {
                "excellent" => 5,
                "good" => 4,
                "okay" => 3,
                "risky" => 2,
                _ => 1,
            },
            label: label.to_string(),
            color: color.to_string(),
        },
        issues,
        warnings,
        tips,
    })
}

/// Startet ein Training
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
    let mut state_lock = state.lock().map_err(|e| format!("Lock error: {}", e))?;
    
    // Check if training is already running
    if state_lock.current_job.is_some() {
        return Err("Ein Training läuft bereits".to_string());
    }
    
    // Generate job ID
    let job_id = format!("train_{}", uuid::Uuid::new_v4().to_string().replace("-", "")[..12].to_string());
    
    // Get paths
    let models_dir = get_models_dir(&app_handle)?;
    
    // CRITICAL: Resolve model_id if it's a local path-based ID (do this FIRST before any other DB operations)
    // CRITICAL FIX: ALWAYS resolve model_id by querying database with model_name
// This ensures we use the existing database ID, not a temporary/new ID
let resolved_model_id = {
let db_path = app_handle
.path()
.app_data_dir()
.map_err(|e| format!("Could not get app data dir: {}", e))?;
let db_file = db_path.join("frametrain.db");

let conn = rusqlite::Connection::open(&db_file)
.map_err(|e| format!("Database error: {}", e))?;

println!("[Training] Resolving model by name: {}", model_name);

// ALWAYS query by name to get the EXISTING database ID
let result: Result<String, _> = conn.query_row(
"SELECT id FROM models WHERE name = ?1",
[&model_name],
|row| row.get(0),
);

match result {
Ok(db_id) => {
println!("[Training] ✅ Found existing model: {} (ID: {})", model_name, db_id);
db_id
},
Err(_) => {
println!("[Training] ⚠️  Model not found in database, using provided ID: {}", model_id);
// Only use provided ID if model doesn't exist yet
model_id.clone()
}
}
};
    
    // Determine model path based on version
    let model_path = if let Some(ref vid) = version_id {
        // Use version path from database
        let db_path = app_handle
            .path()
            .app_data_dir()
            .map_err(|e| format!("Could not get app data dir: {}", e))?;
        let db_file = db_path.join("frametrain.db");
        
        let conn = rusqlite::Connection::open(&db_file)
            .map_err(|e| format!("Database error: {}", e))?;
        
        let version_path: String = conn.query_row(
            "SELECT path FROM model_versions_new WHERE id = ?1",
            [vid],
            |row| row.get(0),
        ).map_err(|e| format!("Version nicht gefunden: {}", e))?;
        
        PathBuf::from(version_path)
    } else {
        // Use base model path
        models_dir.join(&resolved_model_id)
    };
    
    let dataset_path = models_dir.join(&resolved_model_id).join("datasets").join(&dataset_id);
    let output_dir = get_training_output_dir(&app_handle, &job_id)?;
    let checkpoint_dir = output_dir.join("checkpoints");
    
    fs::create_dir_all(&checkpoint_dir)
        .map_err(|e| format!("Could not create checkpoint dir: {}", e))?;
    
    // Update config with paths
    let mut final_config = config.clone();
    final_config.model_path = model_path.to_string_lossy().to_string();
    final_config.dataset_path = dataset_path.to_string_lossy().to_string();
    final_config.output_path = output_dir.join("final_model").to_string_lossy().to_string();
    final_config.checkpoint_dir = checkpoint_dir.to_string_lossy().to_string();
    
    // Write config to file
    let config_path = output_dir.join("config.json");
    let config_json = serde_json::to_string_pretty(&final_config)
        .map_err(|e| format!("Could not serialize config: {}", e))?;
    fs::write(&config_path, &config_json)
        .map_err(|e| format!("Could not write config: {}", e))?;
    
    // Create job
    let job = TrainingJob {
        id: job_id.clone(),
        model_id,
        model_name,
        dataset_id,
        dataset_name,
        status: TrainingStatus::Pending,
        config: final_config,
        created_at: Utc::now(),
        started_at: None,
        completed_at: None,
        progress: TrainingProgress::default(),
        output_path: Some(output_dir.to_string_lossy().to_string()),
        error: None,
    };
    
    state_lock.current_job = Some(job.clone());
    
    // CRITICAL: Get current user_id from AppState before spawning thread
    let user_id = {
        let app_state = app_handle.state::<crate::AppState>();
        let db = app_state.db.lock().map_err(|e| format!("Failed to lock database: {}", e))?;
        db.get_current_user_id().ok_or_else(|| "No user logged in".to_string())?
    };
    
    println!("[Training] Starting training for user: {}", user_id);
    
    // Start Python process in background
    let app_handle_clone = app_handle.clone();
    let config_path_str = config_path.to_string_lossy().to_string();
    let model_id_clone = resolved_model_id.clone();  // Use resolved_model_id instead of job.model_id
    let model_name_clone = job.model_name.clone();
    let version_id_clone = version_id.clone();
    let user_id_clone = user_id.clone();
    
    let state_clone = Arc::clone(&state);
    drop(state_lock);  // Release lock before spawning thread
    
    thread::spawn(move || {
        run_training_process(app_handle_clone, job_id, config_path_str, model_id_clone, model_name_clone, version_id_clone, user_id_clone, state_clone);
    });
    
    Ok(job)
}

fn create_new_model_version(
    app_handle: &tauri::AppHandle,
    model_id: &str,
    model_name: &str,
    parent_version_id: Option<String>,
    output_path: &str,
    user_id: &str,  // CRITICAL: Add user_id parameter
) -> Result<String, String> {
    // Get correct database path using Tauri's app_data_dir
    let db_path = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Could not get app data dir: {}", e))?
        .join("frametrain.db");
    
    println!("[Version] Using database at: {:?}", db_path);
    
    let conn = rusqlite::Connection::open(&db_path)
        .map_err(|e| format!("Database error: {}", e))?;
    
    // CRITICAL: Disable foreign key constraints to avoid FK errors
    conn.execute("PRAGMA foreign_keys = OFF", [])
        .map_err(|e| format!("Failed to disable foreign keys: {}", e))?;
    println!("[Version] Foreign key constraints disabled");
    
    // Ensure ALL required tables exist
    println!("[Version] Ensuring ALL required tables exist...");
    
    // 1. Create models table first (base table)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            base_model TEXT,
            model_path TEXT,
            status TEXT NOT NULL DEFAULT 'created',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(name)
        )",
        [],
    ).map_err(|e| format!("Failed to create models table: {}", e))?;
    println!("[Version] ✅ models table created/verified");
    
    // 2. Create model_versions_new table
    conn.execute(
        "CREATE TABLE IF NOT EXISTS model_versions_new (
            id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            version_name TEXT NOT NULL,
            version_number INTEGER NOT NULL,
            path TEXT NOT NULL,
            size_bytes INTEGER NOT NULL DEFAULT 0,
            file_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            is_root INTEGER NOT NULL DEFAULT 0,
            parent_version_id TEXT
        )",
        [],
    ).map_err(|e| format!("Failed to create model_versions_new table: {}", e))?;
    println!("[Version] ✅ model_versions_new table created/verified");
    
    // 3. Create training_metrics_new table
    conn.execute(
        "CREATE TABLE IF NOT EXISTS training_metrics_new (
            id TEXT PRIMARY KEY,
            version_id TEXT NOT NULL UNIQUE,
            final_train_loss REAL NOT NULL,
            final_val_loss REAL,
            total_epochs INTEGER NOT NULL,
            total_steps INTEGER NOT NULL,
            best_epoch INTEGER,
            training_duration_seconds INTEGER,
            created_at TEXT NOT NULL
        )",
        [],
    ).map_err(|e| format!("Failed to create training_metrics_new table: {}", e))?;
    println!("[Version] ✅ training_metrics_new table created/verified");

    // 4. Create indices
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_versions_model ON model_versions_new(model_id)",
        [],
    ).map_err(|e| format!("Failed to create index: {}", e))?;

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_metrics_version ON training_metrics_new(version_id)",
        [],
    ).map_err(|e| format!("Failed to create index: {}", e))?;
    
    println!("[Version] ✅ ALL tables and indices created/verified");
    
    // CRITICAL: Verify the model_id exists in the models table
    println!("[Version] Checking if model {} exists...", model_id);
    let model_exists: i32 = conn.query_row(
        "SELECT COUNT(*) FROM models WHERE id = ?",
        [model_id],
        |row| row.get(0),
    ).unwrap_or(0);
    
    if model_exists == 0 {
        eprintln!("[Version] ❌ ERROR: Model ID '{}' not found in models table!", model_id);
        eprintln!("[Version] This version will be orphaned!");
        
        // List all available model IDs
        let mut stmt = conn.prepare("SELECT id, name FROM models").unwrap();
        let ids = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        }).unwrap();
        
        eprintln!("[Version] Available models in database:");
        for id in ids {
            if let Ok((db_id, name)) = id {
                eprintln!("[Version]   - {} (ID: {})", name, db_id);
            }
        }
    } else {
        println!("[Version] ✅ Model exists");
    }
    
    // CRITICAL: Ensure the model exists in the models table
    println!("[Version] Checking if model {} exists...", model_id);
    let model_exists: i32 = conn.query_row(
        "SELECT COUNT(*) FROM models WHERE id = ?1",
        [model_id],
        |row| row.get(0),
    ).unwrap_or(0);
    
    if model_exists == 0 {
        println!("[Version] Model {} not found, creating placeholder entry...", model_id);
        // Create a placeholder model entry if it doesn't exist
        let now = Utc::now().to_rfc3339();
        
        // Use the models directory path as model_path
        let models_dir = get_models_dir(app_handle)?;
        let model_path = models_dir.join(model_id);
        let model_path_str = model_path.to_string_lossy().to_string();
        
        // Generate a unique name if needed
        let unique_model_name = format!("{} ({})", model_name, &model_id[..8]);
        
        println!("[Version] Creating model with name: {}", unique_model_name);
        
        conn.execute(
            "INSERT INTO models (id, name, model_path, status, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![model_id, &unique_model_name, &model_path_str, "trained", &now, &now],
        ).map_err(|e| format!("Failed to create model placeholder: {}", e))?;
        println!("[Version] ✅ Model placeholder created with path: {}", model_path_str);
    } else {
        println!("[Version] ✅ Model exists");
    }
    
    // Generate version ID
    let version_id = format!("ver_{}", uuid::Uuid::new_v4().to_string().replace("-", "")[..12].to_string());
    
    // Determine version number
    let version_number: i32 = conn.query_row(
        "SELECT COALESCE(MAX(version_number), 0) + 1 FROM model_versions_new WHERE model_id = ?1",

        [model_id],
        |row| row.get(0),
    ).unwrap_or(1);
    
    // Copy model to new version directory
    let models_dir = get_models_dir(app_handle)?;
    let version_path = models_dir.join(model_id).join("versions").join(&version_id);
    
    fs::create_dir_all(&version_path)
        .map_err(|e| format!("Could not create version dir: {}", e))?;
    
    // Copy trained model
    // CRITICAL FIX: output_path is already the final_model directory from Python
    let output_model_path = PathBuf::from(output_path);
    
    println!("[Version] Copying from: {}", output_model_path.display());
    println!("[Version] Copying to: {}", version_path.display());
    
    if output_model_path.exists() {
        println!("[Version] Source path exists, starting copy...");
        copy_dir_recursive_internal(&output_model_path, &version_path)?;
        println!("[Version] ✅ Copy completed");
    } else {
        eprintln!("[Version] ❌ ERROR: Source path does not exist!");
        return Err(format!("Model output path does not exist: {}", output_model_path.display()));
    }
    
    // Calculate size and file count
    let (size_bytes, file_count) = calculate_dir_size(&version_path).unwrap_or((0, 0));
    
    // Insert into database with user_id
    let now = Utc::now().to_rfc3339();
    conn.execute(
        "INSERT INTO model_versions_new (id, model_id, version_name, version_number, path, size_bytes, file_count, created_at, is_root, parent_version_id, user_id) 
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        rusqlite::params![
            &version_id,
            model_id,
            format!("{} v{}", model_name, version_number),
            version_number,
            version_path.to_string_lossy().to_string(),
            size_bytes,
            file_count,
            &now,
            0, // is_root = false
            parent_version_id,
            user_id,  // CRITICAL: Include user_id
        ],
    ).map_err(|e| format!("Failed to create version: {}", e))?;
    
    Ok(version_id)
}

fn copy_dir_recursive_internal(src: &PathBuf, dst: &PathBuf) -> Result<(), String> {
    if !dst.exists() {
        fs::create_dir_all(dst)
            .map_err(|e| format!("Could not create dir: {}", e))?;
    }
    
    let entries = fs::read_dir(src)
        .map_err(|e| format!("Could not read dir: {}", e))?;
    
    for entry in entries {
        let entry = entry.map_err(|e| format!("Entry error: {}", e))?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        
        if src_path.is_dir() {
            copy_dir_recursive_internal(&src_path, &dst_path)?;
        } else {
            fs::copy(&src_path, &dst_path)
                .map_err(|e| format!("Could not copy file: {}", e))?;
        }
    }
    
    Ok(())
}

fn calculate_dir_size(path: &PathBuf) -> Result<(i64, i32), String> {
    let mut total_size: i64 = 0;
    let mut file_count: i32 = 0;
    
    fn visit_dirs(dir: &PathBuf, total_size: &mut i64, file_count: &mut i32) -> Result<(), String> {
        if dir.is_dir() {
            for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
                let entry = entry.map_err(|e| e.to_string())?;
                let path = entry.path();
                if path.is_dir() {
                    visit_dirs(&path, total_size, file_count)?;
                } else {
                    if let Ok(metadata) = fs::metadata(&path) {
                        *total_size += metadata.len() as i64;
                        *file_count += 1;
                    }
                }
            }
        }
        Ok(())
    }
    
    visit_dirs(path, &mut total_size, &mut file_count)?;
    Ok((total_size, file_count))
}

fn save_training_metrics_from_output(
    app_handle: &tauri::AppHandle,
    version_id: &str,
    output_path: &PathBuf,
    data: &serde_json::Value,
    user_id: &str,  // CRITICAL: Add user_id parameter
) -> Result<(), String> {
    println!("[Metrics] Attempting to save metrics for version: {}", version_id);
    
    // Try to read metrics.json from output directory
    let metrics_file = output_path.join("metrics.json");
    let parent_metrics = output_path.parent().and_then(|p| Some(p.join("metrics.json")));
    
    let metrics_content = if metrics_file.exists() {
        println!("[Metrics] Found metrics.json at: {}", metrics_file.display());
        fs::read_to_string(&metrics_file)
            .map_err(|e| format!("Failed to read metrics: {}", e))?
    } else if let Some(parent_path) = parent_metrics {
        if parent_path.exists() {
            println!("[Metrics] Found metrics.json at: {}", parent_path.display());
            fs::read_to_string(&parent_path)
                .map_err(|e| format!("Failed to read metrics: {}", e))?
        } else {
            // Extract metrics from data parameter
            println!("[Metrics] No metrics.json found, extracting from completion data");
            return save_metrics_from_data(app_handle, version_id, data, user_id);
        }
    } else {
        println!("[Metrics] No metrics.json found, extracting from completion data");
        return save_metrics_from_data(app_handle, version_id, data, user_id);
    };
    
    let metrics: serde_json::Value = serde_json::from_str(&metrics_content)
        .map_err(|e| format!("Failed to parse metrics: {}", e))?;
    
    // Extract values
    let final_train_loss = metrics.get("final_train_loss")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let final_val_loss = metrics.get("final_val_loss")
        .and_then(|v| v.as_f64());
    let total_epochs = metrics.get("total_epochs")
        .and_then(|v| v.as_i64())
        .unwrap_or(0) as i32;
    let total_steps = metrics.get("total_steps")
        .and_then(|v| v.as_i64())
        .unwrap_or(0) as i32;
    let best_epoch = metrics.get("best_epoch")
        .and_then(|v| v.as_i64())
        .map(|v| v as i32);
    let training_duration = metrics.get("training_duration_seconds")
        .and_then(|v| v.as_i64());
    
    // Save to database
    let db_path = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Could not get app data dir: {}", e))?
        .join("frametrain.db");
    
    let conn = rusqlite::Connection::open(&db_path)
        .map_err(|e| format!("Database error: {}", e))?;
    
    let id = format!("metrics_{}", uuid::Uuid::new_v4());
    let created_at = Utc::now().to_rfc3339();
    
    conn.execute(
        "INSERT OR REPLACE INTO training_metrics_new 
         (id, version_id, final_train_loss, final_val_loss, total_epochs, total_steps, 
          best_epoch, training_duration_seconds, created_at, user_id)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        rusqlite::params![
            id,
            version_id,
            final_train_loss,
            final_val_loss,
            total_epochs,
            total_steps,
            best_epoch,
            training_duration,
            created_at,
            user_id,  // CRITICAL: Include user_id
        ],
    ).map_err(|e| format!("Failed to save training metrics: {}", e))?;
    
    println!("[Metrics] ✅ Successfully saved metrics for version: {}", version_id);
    Ok(())
}

fn save_metrics_from_data(
    app_handle: &tauri::AppHandle,
    version_id: &str,
    data: &serde_json::Value,
    user_id: &str,  // CRITICAL: Add user_id parameter
) -> Result<(), String> {
    // Try to extract metrics from the completion data
    // CRITICAL: Check if metrics are in 'final_metrics' object first
    let metrics_obj = data.get("final_metrics").unwrap_or(data);
    
    println!("[Metrics] Extracting from data: {}", serde_json::to_string_pretty(metrics_obj).unwrap_or_default());
    
    let final_train_loss = metrics_obj.get("final_train_loss")
        .and_then(|v| v.as_f64())
        .or_else(|| data.get("train_loss").and_then(|v| v.as_f64()))
        .unwrap_or(0.0);
    
    let final_val_loss = metrics_obj.get("final_val_loss")
        .and_then(|v| v.as_f64())
        .or_else(|| metrics_obj.get("best_val_loss").and_then(|v| v.as_f64()))
        .or_else(|| data.get("val_loss").and_then(|v| v.as_f64()));
    
    let total_epochs = metrics_obj.get("total_epochs")
        .and_then(|v| v.as_i64())
        .or_else(|| data.get("epochs").and_then(|v| v.as_i64()))
        .unwrap_or(0) as i32;
    
    let total_steps = metrics_obj.get("total_steps")
        .and_then(|v| v.as_i64())
        .or_else(|| data.get("steps").and_then(|v| v.as_i64()))
        .unwrap_or(0) as i32;
    
    let best_epoch = metrics_obj.get("best_epoch")
        .and_then(|v| v.as_i64())
        .map(|v| v as i32);
    
    let training_duration = data.get("training_duration_seconds")
        .and_then(|v| v.as_i64())
        .or_else(|| data.get("duration").and_then(|v| v.as_i64()));
    
    println!("[Metrics] Extracted values: train_loss={}, val_loss={:?}, epochs={}, steps={}", 
             final_train_loss, final_val_loss, total_epochs, total_steps);
    
    if total_epochs == 0 || final_train_loss == 0.0 {
        return Err(format!("No valid metrics found - epochs: {}, loss: {}", total_epochs, final_train_loss));
    }
    
    // Save to database
    let db_path = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Could not get app data dir: {}", e))?
        .join("frametrain.db");
    
    let conn = rusqlite::Connection::open(&db_path)
        .map_err(|e| format!("Database error: {}", e))?;
    
    let id = format!("metrics_{}", uuid::Uuid::new_v4());
    let created_at = Utc::now().to_rfc3339();
    
    conn.execute(
        "INSERT OR REPLACE INTO training_metrics_new 
         (id, version_id, final_train_loss, final_val_loss, total_epochs, total_steps, 
          best_epoch, training_duration_seconds, created_at, user_id)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        rusqlite::params![
            id,
            version_id,
            final_train_loss,
            final_val_loss,
            total_epochs,
            total_steps,
            best_epoch,
            training_duration,
            created_at,
            user_id,  // CRITICAL: Include user_id
        ],
    ).map_err(|e| format!("Failed to save training metrics: {}", e))?;
    
    println!("[Metrics] ✅ Successfully saved metrics from data for version: {}", version_id);
    Ok(())
}

fn run_training_process(app_handle: tauri::AppHandle, job_id: String, config_path: String, model_id: String, model_name: String, version_id: Option<String>, user_id: String, state: Arc<Mutex<TrainingState>>) {
    let python = get_python_path();
    
    println!("[Training] Using Python: {}", python);
    println!("[Training] Job ID: {}", job_id);
    println!("[Training] Config: {}", config_path);
    
    // Versuche Engine-Pfad zu finden
    let engine_path = match get_training_engine_path(&app_handle) {
        Ok(p) => {
            println!("[Training] Engine path: {:?}", p);
            p
        },
        Err(e) => {
            eprintln!("[Training] Error finding engine: {}", e);
            let _ = app_handle.emit("training-error", serde_json::json!({
                "job_id": job_id,
                "error": e
            }));
            return;
        }
    };
    
    // Emit start event
    let _ = app_handle.emit("training-started", serde_json::json!({
        "job_id": job_id
    }));
    
    println!("[Training] Starting Python process...");
    
    // Start Python process
    let mut child = match Command::new(&python)
        .arg(engine_path.to_string_lossy().to_string())
        .arg("--config")
        .arg(&config_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => {
            println!("[Training] Python process started successfully");
            c
        },
        Err(e) => {
            eprintln!("[Training] Failed to start Python: {}", e);
            let _ = app_handle.emit("training-error", serde_json::json!({
                "job_id": job_id,
                "error": format!("Failed to start Python: {}", e)
            }));
            return;
        }
    };
    
    // Read stderr in separate thread for debugging
    if let Some(stderr) = child.stderr.take() {
        let stderr_reader = BufReader::new(stderr);
        thread::spawn(move || {
            for line in stderr_reader.lines() {
                if let Ok(line) = line {
                    eprintln!("[Training STDERR] {}", line);
                }
            }
        });
    }
    
    // Read stdout for progress updates
    if let Some(stdout) = child.stdout.take() {
        let reader = BufReader::new(stdout);
        let app_handle_clone = app_handle.clone();
        let job_id_clone = job_id.clone();
        let model_id_clone = model_id.clone();
        let model_name_clone = model_name.clone();
        let version_id_clone = version_id.clone();
        let user_id_clone = user_id.clone();
        
        for line in reader.lines() {
            if let Ok(line) = line {
                println!("[Training OUTPUT] {}", line);
                
                // Parse JSON message from Python
                if let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) {
                    let msg_type = msg.get("type").and_then(|t| t.as_str()).unwrap_or("");
                    
                    println!("[Training] Message type: {}", msg_type);
                    
                    match msg_type {
                        "progress" => {
                            let _ = app_handle_clone.emit("training-progress", serde_json::json!({
                                "job_id": job_id_clone,
                                "data": msg.get("data")
                            }));
                        }
                        "status" => {
                            let _ = app_handle_clone.emit("training-status", serde_json::json!({
                                "job_id": job_id_clone,
                                "data": msg.get("data")
                            }));
                        }
                        "checkpoint" => {
                            let _ = app_handle_clone.emit("training-checkpoint", serde_json::json!({
                                "job_id": job_id_clone,
                                "data": msg.get("data")
                            }));
                        }
                        "complete" => {
                            // Create new model version after successful training
                            if let Some(data) = msg.get("data") {
                                // CRITICAL FIX: model_path is already the final_model directory
                                // We need to pass this directory directly, not go up levels
                                if let Some(model_path_str) = data.get("model_path").and_then(|v| v.as_str()) {
                                    let model_path = PathBuf::from(model_path_str);
                                    
                                    println!("[Training] Complete message received");
                                    println!("[Training] Model path from Python: {}", model_path.display());
                                    
                                    // Use the model_path directly as it's already the final_model directory
                                    let output_path = model_path_str.to_string();
                                    
                                    match create_new_model_version(
                                        &app_handle_clone,
                                        &model_id_clone,
                                        &model_name_clone,
                                        version_id_clone.clone(),
                                        &output_path,
                                        &user_id_clone,  // CRITICAL: Pass user_id
                                    ) {
                                        Ok(new_version_id) => {
                                            println!("[Training] Created new version: {}", new_version_id);
                                            
                                            // NEW: Save training metrics if available
                                            if let Err(e) = save_training_metrics_from_output(
                                                &app_handle_clone,
                                                &new_version_id,
                                                &model_path,
                                                data,
                                                &user_id_clone,  // CRITICAL: Pass user_id
                                            ) {
                                                eprintln!("[Training] Warning: Could not save metrics: {}", e);
                                            }
                                            
                                            let _ = app_handle_clone.emit("training-complete", serde_json::json!({
                                                "job_id": job_id_clone,
                                                "data": msg.get("data"),
                                                "new_version_id": new_version_id
                                            }));
                                        }
                                        Err(e) => {
                                            eprintln!("[Training] Failed to create version: {}", e);
                                            let _ = app_handle_clone.emit("training-complete", serde_json::json!({
                                                "job_id": job_id_clone,
                                                "data": msg.get("data"),
                                                "version_error": e
                                            }));
                                        }
                                    }
                                } else {
                                    eprintln!("[Training] No model_path in complete message");
                                    let _ = app_handle_clone.emit("training-complete", serde_json::json!({
                                        "job_id": job_id_clone,
                                        "data": msg.get("data")
                                    }));
                                }
                                 // Close the if let Some(model_path_str)
                            } else {
                                let _ = app_handle_clone.emit("training-complete", serde_json::json!({
                                    "job_id": job_id_clone,
                                    "data": msg.get("data")
                                }));
                            }
                        }
                        "error" => {
                            let _ = app_handle_clone.emit("training-error", serde_json::json!({
                                "job_id": job_id_clone,
                                "data": msg.get("data")
                            }));
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    
    // Wait for process to finish
    let status = child.wait();
    
    println!("[Training] Process finished with status: {:?}", status);
    
    // CRITICAL FIX: Reset training state after process finishes
    if let Ok(mut state_lock) = state.lock() {
        println!("[Training] Resetting training state...");
        state_lock.current_job = None;
        state_lock.process = None;
        println!("[Training] ✅ Training state reset");
    } else {
        eprintln!("[Training] ❌ Failed to lock state for reset");
    }
    
    let _ = app_handle.emit("training-finished", serde_json::json!({
        "job_id": job_id,
        "success": status.map(|s| s.success()).unwrap_or(false)
    }));
}

/// Stoppt das laufende Training
#[tauri::command]
pub fn stop_training(
    state: tauri::State<'_, Arc<Mutex<TrainingState>>>,
) -> Result<(), String> {
    let mut state_lock = state.lock().map_err(|e| format!("Lock error: {}", e))?;
    
    if let Some(ref mut process) = state_lock.process {
        process.kill().map_err(|e| format!("Could not kill process: {}", e))?;
    }
    
    if let Some(ref mut job) = state_lock.current_job {
        job.status = TrainingStatus::Stopped;
        job.completed_at = Some(Utc::now());
    }
    
    state_lock.process = None;
    
    Ok(())
}

/// Holt den aktuellen Training-Status
#[tauri::command]
pub fn get_current_training(
    state: tauri::State<'_, Arc<Mutex<TrainingState>>>,
) -> Result<Option<TrainingJob>, String> {
    let state_lock = state.lock().map_err(|e| format!("Lock error: {}", e))?;
    Ok(state_lock.current_job.clone())
}

/// Holt die Training-History
#[tauri::command]
pub fn get_training_history(
    app_handle: tauri::AppHandle,
) -> Result<Vec<TrainingJob>, String> {
    load_training_jobs(&app_handle)
}

/// Löscht einen Training-Job aus der History
#[tauri::command]
pub fn delete_training_job(
    app_handle: tauri::AppHandle,
    job_id: String,
) -> Result<(), String> {
    let mut jobs = load_training_jobs(&app_handle)?;
    jobs.retain(|j| j.id != job_id);
    save_training_jobs(&app_handle, &jobs)?;
    
    // Lösche auch den Output-Ordner
    let data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Could not get app data dir: {}", e))?;
    
    let output_dir = data_dir.join("training_outputs").join(&job_id);
    if output_dir.exists() {
        fs::remove_dir_all(&output_dir).ok();
    }
    
    Ok(())
}

/// Prüft ob Python und PyTorch installiert sind
#[tauri::command]
pub async fn check_training_requirements() -> Result<RequirementsCheck, String> {
    let python = get_python_path();
    
    println!("[Requirements] Using Python: {}", python);
    
    // Check Python
    let python_version = Command::new(&python)
        .arg("--version")
        .output();
    
    let python_ok = python_version.is_ok() && python_version.as_ref().unwrap().status.success();
    let python_version_str = if python_ok {
        String::from_utf8_lossy(&python_version.unwrap().stdout).trim().to_string()
    } else {
        "Nicht gefunden".to_string()
    };
    
    println!("[Requirements] Python: {} - {}", python_ok, python_version_str);
    
    // Check PyTorch
    let torch_check = Command::new(&python)
        .arg("-c")
        .arg("import torch; print(torch.__version__)")
        .output();
    
    let torch_ok = torch_check.is_ok() && torch_check.as_ref().unwrap().status.success();
    let torch_version = if torch_ok {
        String::from_utf8_lossy(&torch_check.unwrap().stdout).trim().to_string()
    } else {
        "Nicht installiert".to_string()
    };
    
    println!("[Requirements] PyTorch: {} - {}", torch_ok, torch_version);
    
    // Check CUDA
    let cuda_check = Command::new(&python)
        .arg("-c")
        .arg("import torch; print(torch.cuda.is_available())")
        .output();
    
    let cuda_available = cuda_check.is_ok() && 
        String::from_utf8_lossy(&cuda_check.unwrap().stdout).trim() == "True";
    
    println!("[Requirements] CUDA: {}", cuda_available);
    
    // Check MPS (Apple Silicon)
    let mps_check = Command::new(&python)
        .arg("-c")
        .arg("import torch; print(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())")
        .output();
    
    let mps_available = mps_check.is_ok() && 
        String::from_utf8_lossy(&mps_check.unwrap().stdout).trim() == "True";
    
    println!("[Requirements] MPS: {}", mps_available);
    
    // Check Transformers
    let transformers_check = Command::new(&python)
        .arg("-c")
        .arg("import transformers; print(transformers.__version__)")
        .output();
    
    let transformers_ok = transformers_check.is_ok() && transformers_check.as_ref().unwrap().status.success();
    let transformers_version = if transformers_ok {
        String::from_utf8_lossy(&transformers_check.unwrap().stdout).trim().to_string()
    } else {
        "Nicht installiert".to_string()
    };
    
    println!("[Requirements] Transformers: {} - {}", transformers_ok, transformers_version);
    
    // Check PEFT
    let peft_check = Command::new(&python)
        .arg("-c")
        .arg("import peft; print(peft.__version__)")
        .output();
    
    let peft_ok = peft_check.is_ok() && peft_check.as_ref().unwrap().status.success();
    let peft_version = if peft_ok {
        String::from_utf8_lossy(&peft_check.unwrap().stdout).trim().to_string()
    } else {
        "Nicht installiert".to_string()
    };
    
    println!("[Requirements] PEFT: {} - {}", peft_ok, peft_version);
    
    Ok(RequirementsCheck {
        python_installed: python_ok,
        python_version: python_version_str,
        torch_installed: torch_ok,
        torch_version,
        cuda_available,
        mps_available,
        transformers_installed: transformers_ok,
        transformers_version,
        peft_installed: peft_ok,
        peft_version,
        ready: python_ok && torch_ok,
    })
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

// ============ Default Implementation ============

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            dataset_path: String::new(),
            output_path: String::new(),
            checkpoint_dir: String::new(),
            epochs: 3,
            batch_size: 8,
            gradient_accumulation_steps: 1,
            max_steps: -1,
            learning_rate: 5e-5,
            weight_decay: 0.01,
            warmup_steps: 0,
            warmup_ratio: 0.0,
            optimizer: "adamw".to_string(),
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_epsilon: 1e-8,
            sgd_momentum: 0.9,
            scheduler: "linear".to_string(),
            scheduler_step_size: 1,
            scheduler_gamma: 0.1,
            cosine_min_lr: 0.0,
            dropout: 0.1,
            max_grad_norm: 1.0,
            label_smoothing: 0.0,
            fp16: false,
            bf16: false,
            use_lora: false,
            lora_r: 8,
            lora_alpha: 32,
            lora_dropout: 0.1,
            lora_target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            load_in_8bit: false,
            load_in_4bit: false,
            max_seq_length: 512,
            num_workers: 4,
            pin_memory: true,
            eval_steps: 500,
            eval_strategy: "steps".to_string(),
            save_steps: 500,
            save_strategy: "steps".to_string(),
            save_total_limit: 3,
            logging_steps: 100,
            seed: 42,
            dataloader_drop_last: false,
            group_by_length: false,
            training_type: "fine_tuning".to_string(),
            task_type: "causal_lm".to_string(),
        }
    }
}
