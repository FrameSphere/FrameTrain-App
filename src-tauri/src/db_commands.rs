// Database Commands - Placeholder
use crate::AppState;
use crate::database::{Model, Dataset};
use tauri::State;

#[tauri::command]
pub fn db_create_model(
    _state: State<AppState>,
    _name: String,
    _description: Option<String>,
    _base_model: Option<String>,
) -> Result<Model, String> {
    Err("Feature in Entwicklung".to_string())
}

#[tauri::command]
pub fn db_list_models(_state: State<AppState>) -> Result<Vec<Model>, String> {
    Ok(vec![])
}

#[tauri::command]
pub fn db_get_model(_state: State<AppState>, _id: String) -> Result<Model, String> {
    Err("Feature in Entwicklung".to_string())
}

#[tauri::command]
pub fn db_delete_model(_state: State<AppState>, _id: String) -> Result<(), String> {
    Err("Feature in Entwicklung".to_string())
}

#[tauri::command]
pub fn db_list_datasets(_state: State<AppState>) -> Result<Vec<Dataset>, String> {
    Ok(vec![])
}

#[tauri::command]
pub fn db_save_dataset(
    _state: State<AppState>,
    _dataset: Dataset,
) -> Result<(), String> {
    Err("Feature in Entwicklung".to_string())
}
