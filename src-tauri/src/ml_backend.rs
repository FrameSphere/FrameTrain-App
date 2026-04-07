// ML Backend - Placeholder
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingProgress {
    pub epoch: i32,
    pub total_epochs: i32,
    pub loss: f64,
    pub accuracy: Option<f64>,
}

// Moved to training_manager.rs

#[tauri::command]
pub async fn get_training_progress(_training_id: String) -> Result<TrainingProgress, String> {
    Err("Training-Feature in Entwicklung".to_string())
}

// Moved to training_manager.rs

#[tauri::command]
pub async fn download_model(_model_id: String, _destination: String) -> Result<(), String> {
    Err("Feature in Entwicklung".to_string())
}

#[tauri::command]
pub async fn get_local_models() -> Result<Vec<String>, String> {
    Ok(vec![])
}

#[tauri::command]
pub async fn validate_dataset(_dataset_path: String) -> Result<bool, String> {
    Err("Feature in Entwicklung".to_string())
}
