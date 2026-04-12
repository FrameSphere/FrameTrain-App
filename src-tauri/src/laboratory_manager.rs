/// Laboratory Manager – Interaktives Lernen mit Echtzeit-Feedback
///
/// Workflow:
/// 1. Sample aus Dataset laden
/// 2. Python-Inferenz ausführen
/// 3. Ergebnis anzeigen
/// 4. Nutzer gibt Feedback
/// 5. Feedback → neues Trainingsbeispiel (Active Learning)

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use tauri::Manager;

// ============ Types ============

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabSample {
    pub path: String,
    pub filename: String,
    pub sample_type: String, // "image", "text", "json", "csv", "audio", "unknown"
    pub content: Option<String>, // Textinhalt für text/json
    pub dataset_id: String,
    pub sample_index: usize,
    pub total_samples: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabInferenceResult {
    pub sample_path: String,
    pub model_output_type: String, // "classification", "detection", "ner", "generation", "raw"
    pub raw_output: serde_json::Value,
    pub rendered: RenderedOutput,
    pub inference_time_ms: u64,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RenderedOutput {
    pub primary_label: Option<String>,
    pub confidence: Option<f64>,
    pub labels: Vec<LabelResult>,
    pub bounding_boxes: Vec<BoundingBox>,
    pub highlighted_spans: Vec<TextSpan>,
    pub generated_text: Option<String>,
    pub key_values: Vec<(String, String)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelResult {
    pub label: String,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub label: String,
    pub score: f64,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSpan {
    pub start: usize,
    pub end: usize,
    pub label: String,
    pub score: f64,
    pub color: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabFeedback {
    pub rating: String, // "correct", "partial", "incorrect"
    pub comment: String,
    pub corrected_label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabSession {
    pub id: String,
    pub model_id: String,
    pub model_name: String,
    pub version_id: String,
    pub version_name: String,
    pub dataset_id: String,
    pub dataset_name: String,
    pub sample: LabSample,
    pub inference_result: LabInferenceResult,
    pub feedback: LabFeedback,
    pub created_at: DateTime<Utc>,
    pub added_to_dataset: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabStats {
    pub total_sessions: usize,
    pub correct: usize,
    pub partial: usize,
    pub incorrect: usize,
    pub accuracy_rate: f64,
    pub exported_samples: usize,
}

// ============ Helpers ============

fn get_lab_dir(app_handle: &tauri::AppHandle) -> Result<PathBuf, String> {
    let data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Could not get app data dir: {}", e))?;
    let lab_dir = data_dir.join("laboratory");
    fs::create_dir_all(&lab_dir).map_err(|e| e.to_string())?;
    Ok(lab_dir)
}

fn get_sessions_path(app_handle: &tauri::AppHandle) -> Result<PathBuf, String> {
    Ok(get_lab_dir(app_handle)?.join("sessions.json"))
}

fn load_sessions(app_handle: &tauri::AppHandle) -> Result<Vec<LabSession>, String> {
    let path = get_sessions_path(app_handle)?;
    if !path.exists() {
        return Ok(Vec::new());
    }
    let content = fs::read_to_string(&path).map_err(|e| e.to_string())?;
    serde_json::from_str(&content).map_err(|e| e.to_string())
}

fn save_sessions(app_handle: &tauri::AppHandle, sessions: &[LabSession]) -> Result<(), String> {
    let path = get_sessions_path(app_handle)?;
    let content = serde_json::to_string_pretty(sessions).map_err(|e| e.to_string())?;
    fs::write(&path, content).map_err(|e| e.to_string())
}

fn detect_sample_type(path: &str) -> String {
    let ext = PathBuf::from(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "jpg" | "jpeg" | "png" | "bmp" | "gif" | "webp" | "tiff" => "image".to_string(),
        "txt" | "md" => "text".to_string(),
        "json" | "jsonl" => "json".to_string(),
        "csv" | "tsv" => "csv".to_string(),
        "wav" | "mp3" | "flac" | "ogg" | "m4a" => "audio".to_string(),
        _ => "unknown".to_string(),
    }
}

fn get_python_path() -> String {
    for cmd in &["python3", "python"] {
        if let Ok(output) = Command::new(cmd).arg("--version").output() {
            if output.status.success() {
                return cmd.to_string();
            }
        }
    }
    "python3".to_string()
}

fn get_lab_script_path(app_handle: &tauri::AppHandle) -> Result<PathBuf, String> {
    let resource_path = app_handle
        .path()
        .resource_dir()
        .map_err(|e| e.to_string())?;

    let candidates = vec![
        resource_path.join("python").join("lab_inference.py"),
        PathBuf::from("src-tauri/python/lab_inference.py"),
        PathBuf::from("/Users/karol/Desktop/Laufende_Projekte/FrameTrain/desktop-app2/src-tauri/python/lab_inference.py"),
    ];

    for path in candidates {
        if path.exists() {
            return Ok(path);
        }
    }

    Err("lab_inference.py nicht gefunden".to_string())
}

fn get_model_path(app_handle: &tauri::AppHandle, model_id: &str) -> Result<PathBuf, String> {
    let data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?;
    Ok(data_dir.join("models").join(model_id))
}

fn get_version_path(app_handle: &tauri::AppHandle, model_id: &str, version_id: &str) -> Result<PathBuf, String> {
    let db_path = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?
        .join("frametrain.db");

    let conn = rusqlite::Connection::open(&db_path).map_err(|e| e.to_string())?;

    let path: String = conn
        .query_row(
            "SELECT path FROM model_versions_new WHERE id = ?1",
            [version_id],
            |row| row.get(0),
        )
        .map_err(|_| {
            // Fallback: base model path
            get_model_path(app_handle, model_id)
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default()
        })?;

    Ok(PathBuf::from(path))
}

// ============ Tauri Commands ============

/// Lädt ein Sample aus einem Dataset
#[tauri::command]
pub fn lab_load_sample(
    app_handle: tauri::AppHandle,
    model_id: String,
    dataset_id: String,
    sample_index: Option<usize>,
) -> Result<LabSample, String> {
    let data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?;

    // Dataset-Ordner finden
    let dataset_base = data_dir.join("models").join(&model_id).join("datasets").join(&dataset_id);

    // Suche in train/ oder direkt im Ordner
    let search_dirs = vec![
        dataset_base.join("train"),
        dataset_base.clone(),
        dataset_base.join("data"),
    ];

    let mut all_files: Vec<PathBuf> = Vec::new();
    for dir in &search_dirs {
        if dir.exists() {
            if let Ok(entries) = fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() {
                        let ext = path.extension()
                            .and_then(|e| e.to_str())
                            .unwrap_or("")
                            .to_lowercase();
                        // Exclude metadata files
                        if !matches!(ext.as_str(), "json" | "") || path.file_name()
                            .and_then(|n| n.to_str())
                            .map(|n| n != "metadata.json" && n != "config.json")
                            .unwrap_or(false)
                        {
                            all_files.push(path);
                        }
                    }
                }
            }
            if !all_files.is_empty() {
                break;
            }
        }
    }

    // Auch jsonl direkt lesen
    let jsonl_files: Vec<PathBuf> = {
        let mut files = Vec::new();
        for dir in &search_dirs {
            if dir.exists() {
                if let Ok(entries) = fs::read_dir(dir) {
                    for entry in entries.flatten() {
                        let p = entry.path();
                        if p.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                            files.push(p);
                        }
                    }
                }
            }
        }
        files
    };

    // Falls jsonl vorhanden → Zeilen als Samples verwenden
    if all_files.is_empty() && !jsonl_files.is_empty() {
        let jsonl_path = &jsonl_files[0];
        let content = fs::read_to_string(jsonl_path).map_err(|e| e.to_string())?;
        let lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();

        if lines.is_empty() {
            return Err("Dataset leer".to_string());
        }

        let idx = sample_index.unwrap_or(0) % lines.len();
        let line = lines[idx];

        return Ok(LabSample {
            path: jsonl_path.to_string_lossy().to_string(),
            filename: format!("Zeile {}", idx + 1),
            sample_type: "json".to_string(),
            content: Some(line.to_string()),
            dataset_id,
            sample_index: idx,
            total_samples: lines.len(),
        });
    }

    if all_files.is_empty() {
        return Err("Keine Samples im Dataset gefunden".to_string());
    }

    all_files.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

    let idx = sample_index.unwrap_or(0) % all_files.len();
    let file_path = &all_files[idx];
    let sample_type = detect_sample_type(&file_path.to_string_lossy());

    let content = if sample_type == "text" || sample_type == "json" || sample_type == "csv" {
        fs::read_to_string(file_path).ok().map(|s| {
            if s.len() > 4000 { s[..4000].to_string() } else { s }
        })
    } else {
        None
    };

    Ok(LabSample {
        path: file_path.to_string_lossy().to_string(),
        filename: file_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string(),
        sample_type,
        content,
        dataset_id,
        sample_index: idx,
        total_samples: all_files.len(),
    })
}

/// Führt Python-Inferenz auf einem Sample aus
#[tauri::command]
pub async fn lab_run_inference(
    app_handle: tauri::AppHandle,
    model_id: String,
    version_id: String,
    sample_path: String,
    task_type: Option<String>,
) -> Result<LabInferenceResult, String> {
    let python = get_python_path();

    // Modell-Pfad bestimmen
    let model_path = if version_id.is_empty() || version_id == "root" {
        get_model_path(&app_handle, &model_id)?
    } else {
        get_version_path(&app_handle, &model_id, &version_id)
            .unwrap_or_else(|_| get_model_path(&app_handle, &model_id).unwrap_or_default())
    };

    let script_path = match get_lab_script_path(&app_handle) {
        Ok(p) => p,
        Err(e) => {
            // Script nicht gefunden – simuliere Dummy-Output für Demo
            return Ok(dummy_inference_result(&sample_path, &e));
        }
    };

    let task = task_type.unwrap_or_else(|| "auto".to_string());

    let output = Command::new(&python)
        .arg(script_path.to_string_lossy().to_string())
        .arg("--model_path")
        .arg(model_path.to_string_lossy().to_string())
        .arg("--sample_path")
        .arg(&sample_path)
        .arg("--task_type")
        .arg(&task)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("Python konnte nicht gestartet werden: {}", e))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if !output.status.success() || stdout.trim().is_empty() {
        return Ok(LabInferenceResult {
            sample_path: sample_path.clone(),
            model_output_type: "raw".to_string(),
            raw_output: serde_json::Value::Null,
            rendered: RenderedOutput {
                key_values: vec![("Fehler".to_string(), stderr.clone())],
                ..Default::default()
            },
            inference_time_ms: 0,
            error: Some(if stderr.is_empty() { "Keine Ausgabe vom Modell".to_string() } else { stderr }),
        });
    }

    // Parse JSON output
    serde_json::from_str::<LabInferenceResult>(stdout.trim())
        .map_err(|e| format!("Konnte Inferenz-Ergebnis nicht parsen: {} | Output: {}", e, &stdout[..stdout.len().min(200)]))
}

/// Dummy-Ergebnis wenn Python-Script fehlt (für UI-Testing)
fn dummy_inference_result(sample_path: &str, note: &str) -> LabInferenceResult {
    let sample_type = detect_sample_type(sample_path);

    let rendered = match sample_type.as_str() {
        "image" => RenderedOutput {
            primary_label: Some("Katze".to_string()),
            confidence: Some(0.94),
            labels: vec![
                LabelResult { label: "Katze".to_string(), score: 0.94 },
                LabelResult { label: "Hund".to_string(), score: 0.04 },
                LabelResult { label: "Vogel".to_string(), score: 0.02 },
            ],
            ..Default::default()
        },
        "text" | "json" => RenderedOutput {
            primary_label: Some("Positiv".to_string()),
            confidence: Some(0.87),
            labels: vec![
                LabelResult { label: "Positiv".to_string(), score: 0.87 },
                LabelResult { label: "Negativ".to_string(), score: 0.13 },
            ],
            ..Default::default()
        },
        _ => RenderedOutput {
            primary_label: Some("Unbekannt".to_string()),
            key_values: vec![
                ("Hinweis".to_string(), format!("Demo-Modus: {}", note)),
            ],
            ..Default::default()
        },
    };

    LabInferenceResult {
        sample_path: sample_path.to_string(),
        model_output_type: if sample_type == "image" { "classification" } else { "classification" }.to_string(),
        raw_output: serde_json::json!({"demo": true}),
        rendered,
        inference_time_ms: 42,
        error: None,
    }
}

/// Speichert eine Feedback-Session
#[tauri::command]
pub fn lab_save_session(
    app_handle: tauri::AppHandle,
    model_id: String,
    model_name: String,
    version_id: String,
    version_name: String,
    dataset_id: String,
    dataset_name: String,
    sample: LabSample,
    inference_result: LabInferenceResult,
    feedback: LabFeedback,
) -> Result<String, String> {
    let session_id = format!("lab_{}", uuid::Uuid::new_v4().to_string().replace("-", "")[..12].to_string());

    let session = LabSession {
        id: session_id.clone(),
        model_id,
        model_name,
        version_id,
        version_name,
        dataset_id,
        dataset_name,
        sample,
        inference_result,
        feedback,
        created_at: Utc::now(),
        added_to_dataset: false,
    };

    let mut sessions = load_sessions(&app_handle)?;
    sessions.push(session);
    save_sessions(&app_handle, &sessions)?;

    Ok(session_id)
}

/// Gibt alle Sessions zurück (optional gefiltert nach Modell)
#[tauri::command]
pub fn lab_get_sessions(
    app_handle: tauri::AppHandle,
    model_id: Option<String>,
) -> Result<Vec<LabSession>, String> {
    let mut sessions = load_sessions(&app_handle)?;

    if let Some(mid) = model_id {
        sessions.retain(|s| s.model_id == mid);
    }

    // Neueste zuerst
    sessions.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(sessions)
}

/// Löscht eine Session
#[tauri::command]
pub fn lab_delete_session(
    app_handle: tauri::AppHandle,
    session_id: String,
) -> Result<(), String> {
    let mut sessions = load_sessions(&app_handle)?;
    sessions.retain(|s| s.id != session_id);
    save_sessions(&app_handle, &sessions)
}

/// Exportiert Feedback-Sessions als neues Dataset für weiteres Training
#[tauri::command]
pub fn lab_export_as_dataset(
    app_handle: tauri::AppHandle,
    model_id: String,
    dataset_name: String,
    only_incorrect: bool,
) -> Result<String, String> {
    let sessions = load_sessions(&app_handle)?;

    let relevant: Vec<&LabSession> = sessions.iter()
        .filter(|s| s.model_id == model_id)
        .filter(|s| {
            if only_incorrect {
                s.feedback.rating == "incorrect" || s.feedback.rating == "partial"
            } else {
                true
            }
        })
        .collect();

    if relevant.is_empty() {
        return Err("Keine passenden Sessions zum Exportieren".to_string());
    }

    let data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?;

    let dataset_id = format!("lab_{}", uuid::Uuid::new_v4().to_string().replace("-", "")[..8].to_string());
    let dataset_dir = data_dir
        .join("models")
        .join(&model_id)
        .join("datasets")
        .join(&dataset_id)
        .join("train");

    fs::create_dir_all(&dataset_dir).map_err(|e| e.to_string())?;

    // Export als JSONL – universelles Format
    let jsonl_path = dataset_dir.join("lab_data.jsonl");
    let mut lines = Vec::new();

    for session in &relevant {
        let entry = serde_json::json!({
            "sample_path": session.sample.path,
            "sample_type": session.sample.sample_type,
            "content": session.sample.content,
            "model_output": session.inference_result.raw_output,
            "feedback": {
                "rating": session.feedback.rating,
                "comment": session.feedback.comment,
                "corrected_label": session.feedback.corrected_label,
            },
            "source": "laboratory",
            "session_id": session.id,
        });
        lines.push(serde_json::to_string(&entry).map_err(|e| e.to_string())?);
    }

    fs::write(&jsonl_path, lines.join("\n")).map_err(|e| e.to_string())?;

    // Metadata
    let meta = serde_json::json!({
        "id": dataset_id,
        "name": dataset_name,
        "source": "laboratory",
        "sample_count": relevant.len(),
        "model_id": model_id,
        "created_at": Utc::now().to_rfc3339(),
    });
    let meta_dir = data_dir.join("models").join(&model_id).join("datasets").join(&dataset_id);
    fs::write(meta_dir.join("metadata.json"), serde_json::to_string_pretty(&meta).unwrap())
        .map_err(|e| e.to_string())?;

    Ok(dataset_id)
}

/// Gibt Statistiken zu Laboratory-Sessions zurück
#[tauri::command]
pub fn lab_get_stats(
    app_handle: tauri::AppHandle,
    model_id: Option<String>,
) -> Result<LabStats, String> {
    let sessions = load_sessions(&app_handle)?;

    let relevant: Vec<&LabSession> = sessions.iter()
        .filter(|s| model_id.as_ref().map(|m| &s.model_id == m).unwrap_or(true))
        .collect();

    let total = relevant.len();
    let correct = relevant.iter().filter(|s| s.feedback.rating == "correct").count();
    let partial = relevant.iter().filter(|s| s.feedback.rating == "partial").count();
    let incorrect = relevant.iter().filter(|s| s.feedback.rating == "incorrect").count();
    let exported = relevant.iter().filter(|s| s.added_to_dataset).count();

    Ok(LabStats {
        total_sessions: total,
        correct,
        partial,
        incorrect,
        accuracy_rate: if total > 0 { correct as f64 / total as f64 } else { 0.0 },
        exported_samples: exported,
    })
}
