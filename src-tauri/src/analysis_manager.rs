// Analysis Manager - Handles training analysis and metrics retrieval

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use tauri::State;
use crate::database::Database;
use crate::AppState;

// ============ Data Structures ============

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainingMetrics {
    pub id: String,
    pub version_id: String,
    pub final_train_loss: f64,
    pub final_val_loss: Option<f64>,
    pub total_epochs: i32,
    pub total_steps: i32,
    pub best_epoch: Option<i32>,
    pub training_duration_seconds: Option<i64>,
    pub created_at: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LogEntry {
    pub epoch: i32,
    pub step: i32,
    pub train_loss: f64,
    pub val_loss: Option<f64>,
    pub learning_rate: f64,
    pub timestamp: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VersionDetails {
    pub id: String,
    pub model_id: String,
    pub version_name: String,
    pub version_number: i32,
    pub path: String,
    pub size_bytes: i64,
    pub file_count: i32,
    pub created_at: String,
    pub is_root: bool,
    pub parent_version_id: Option<String>,
}

// ============ Database Extensions ============

impl Database {
    pub fn get_training_metrics_by_version(&self, version_id: &str) -> Result<Option<TrainingMetrics>, String> {
        let result = self.conn.query_row(
            "SELECT id, version_id, final_train_loss, final_val_loss, total_epochs, total_steps, 
                    best_epoch, training_duration_seconds, created_at
             FROM training_metrics_new 
             WHERE version_id = ?1",
            &[version_id],
            |row| {
                Ok(TrainingMetrics {
                    id: row.get(0)?,
                    version_id: row.get(1)?,
                    final_train_loss: row.get(2)?,
                    final_val_loss: row.get(3)?,
                    total_epochs: row.get(4)?,
                    total_steps: row.get(5)?,
                    best_epoch: row.get(6)?,
                    training_duration_seconds: row.get(7)?,
                    created_at: row.get(8)?,
                })
            },
        );

        match result {
            Ok(metrics) => Ok(Some(metrics)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(format!("Database error: {}", e)),
        }
    }

    pub fn get_version_details_by_id(&self, version_id: &str) -> Result<Option<VersionDetails>, String> {
        let result = self.conn.query_row(
            "SELECT id, model_id, version_name, version_number, path, size_bytes, file_count, 
                    created_at, is_root, parent_version_id
             FROM model_versions_new 
             WHERE id = ?1",
            &[version_id],
            |row| {
                Ok(VersionDetails {
                    id: row.get(0)?,
                    model_id: row.get(1)?,
                    version_name: row.get(2)?,
                    version_number: row.get(3)?,
                    path: row.get(4)?,
                    size_bytes: row.get(5)?,
                    file_count: row.get(6)?,
                    created_at: row.get(7)?,
                    is_root: row.get::<_, i32>(8)? != 0,
                    parent_version_id: row.get(9)?,
                })
            },
        );

        match result {
            Ok(details) => Ok(Some(details)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(format!("Database error: {}", e)),
        }
    }
}

// ============ Tauri Commands ============

#[tauri::command]
pub async fn get_training_metrics(
    state: State<'_, AppState>,
    version_id: String,
) -> Result<TrainingMetrics, String> {
    let db = state.db.lock().map_err(|e| format!("Lock error: {}", e))?;
    
    match db.get_training_metrics_by_version(&version_id)? {
        Some(metrics) => Ok(metrics),
        None => Err(format!("No training metrics found for version {}", version_id)),
    }
}

#[tauri::command]
pub async fn get_version_details(
    state: State<'_, AppState>,
    version_id: String,
) -> Result<VersionDetails, String> {
    let db = state.db.lock().map_err(|e| format!("Lock error: {}", e))?;
    
    match db.get_version_details_by_id(&version_id)? {
        Some(details) => Ok(details),
        None => Err(format!("Version {} not found", version_id)),
    }
}

#[tauri::command]
pub async fn get_training_logs(
    state: State<'_, AppState>,
    version_id: String,
) -> Result<Vec<LogEntry>, String> {
    let db = state.db.lock().map_err(|e| format!("Lock error: {}", e))?;
    
    // Get version details to find the logs file
    let version = match db.get_version_details_by_id(&version_id)? {
        Some(v) => v,
        None => return Err(format!("Version {} not found", version_id)),
    };

    // Try to find logs file in the version directory
    let version_path = PathBuf::from(&version.path);
    let logs_file = version_path.join("training_logs.json");

    if !logs_file.exists() {
        // Also try in a "logs" subdirectory
        let logs_subdir = version_path.join("logs").join("training_logs.json");
        if !logs_subdir.exists() {
            return Ok(Vec::new()); // Return empty vec if no logs found
        } else {
            return read_logs_file(&logs_subdir);
        }
    }

    read_logs_file(&logs_file)
}

fn read_logs_file(path: &PathBuf) -> Result<Vec<LogEntry>, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read logs file: {}", e))?;

    let logs: Vec<LogEntry> = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse logs JSON: {}", e))?;

    Ok(logs)
}

#[tauri::command]
pub async fn save_training_metrics(
    state: State<'_, AppState>,
    version_id: String,
    final_train_loss: f64,
    final_val_loss: Option<f64>,
    total_epochs: i32,
    total_steps: i32,
    best_epoch: Option<i32>,
    training_duration_seconds: Option<i64>,
) -> Result<(), String> {
    let db = state.db.lock().map_err(|e| format!("Lock error: {}", e))?;

    let id = format!("metrics_{}", uuid::Uuid::new_v4());
    let created_at = chrono::Utc::now().to_rfc3339();

    db.conn.execute(
        "INSERT OR REPLACE INTO training_metrics_new 
         (id, version_id, final_train_loss, final_val_loss, total_epochs, total_steps, 
          best_epoch, training_duration_seconds, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        rusqlite::params![
            id,
            version_id,
            final_train_loss,
            final_val_loss,
            total_epochs,
            total_steps,
            best_epoch,
            training_duration_seconds,
            created_at,
        ],
    ).map_err(|e| format!("Failed to save training metrics: {}", e))?;

    Ok(())
}

#[tauri::command]
pub async fn save_training_logs(
    version_id: String,
    logs: Vec<LogEntry>,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let db = state.db.lock().map_err(|e| format!("Lock error: {}", e))?;
    
    // Get version details to find where to save logs
    let version = match db.get_version_details_by_id(&version_id)? {
        Some(v) => v,
        None => return Err(format!("Version {} not found", version_id)),
    };

    let version_path = PathBuf::from(&version.path);
    
    // Create logs directory if it doesn't exist
    let logs_dir = version_path.join("logs");
    fs::create_dir_all(&logs_dir)
        .map_err(|e| format!("Failed to create logs directory: {}", e))?;

    let logs_file = logs_dir.join("training_logs.json");

    // Write logs to file
    let json_content = serde_json::to_string_pretty(&logs)
        .map_err(|e| format!("Failed to serialize logs: {}", e))?;

    fs::write(&logs_file, json_content)
        .map_err(|e| format!("Failed to write logs file: {}", e))?;

    Ok(())
}

// Helper function to update metrics during training
#[tauri::command]
pub async fn update_training_progress(
    state: State<'_, AppState>,
    version_id: String,
    epoch: i32,
    step: i32,
    train_loss: f64,
    val_loss: Option<f64>,
    learning_rate: f64,
) -> Result<(), String> {
    let db = state.db.lock().map_err(|e| format!("Lock error: {}", e))?;
    
    // Get version details
    let version = match db.get_version_details_by_id(&version_id)? {
        Some(v) => v,
        None => return Err(format!("Version {} not found", version_id)),
    };

    let version_path = PathBuf::from(&version.path);
    let logs_dir = version_path.join("logs");
    fs::create_dir_all(&logs_dir)
        .map_err(|e| format!("Failed to create logs directory: {}", e))?;

    let logs_file = logs_dir.join("training_logs.json");

    // Read existing logs or create new
    let mut logs: Vec<LogEntry> = if logs_file.exists() {
        let content = fs::read_to_string(&logs_file)
            .map_err(|e| format!("Failed to read logs: {}", e))?;
        serde_json::from_str(&content).unwrap_or_default()
    } else {
        Vec::new()
    };

    // Add new entry
    logs.push(LogEntry {
        epoch,
        step,
        train_loss,
        val_loss,
        learning_rate,
        timestamp: chrono::Utc::now().to_rfc3339(),
    });

    // Write updated logs
    let json_content = serde_json::to_string_pretty(&logs)
        .map_err(|e| format!("Failed to serialize logs: {}", e))?;

    fs::write(&logs_file, json_content)
        .map_err(|e| format!("Failed to write logs: {}", e))?;

    Ok(())
}
