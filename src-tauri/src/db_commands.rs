// Database Commands
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

// ==================== CANVAS MODEL DESIGNS (W3) ====================

/// Speichert oder aktualisiert den vollständigen Editor-Zustand (nodes/edges/viewport)
/// eines Canvas-Modells in SQLite. Wird von SynapseBuilder aufgerufen.
#[tauri::command]
pub fn save_canvas_model_design(
    state: State<AppState>,
    model_id: String,
    design_json: String,
) -> Result<(), String> {
    let db = state.db.lock().map_err(|e| format!("DB Lock: {}", e))?;
    db.save_canvas_model_design(&model_id, &design_json)
}

/// Lädt den Editor-Zustand eines Canvas-Modells aus SQLite.
/// Gibt null zurück wenn noch kein Design gespeichert wurde.
#[tauri::command]
pub fn load_canvas_model_design(
    state: State<AppState>,
    model_id: String,
) -> Result<Option<String>, String> {
    let db = state.db.lock().map_err(|e| format!("DB Lock: {}", e))?;
    db.load_canvas_model_design(&model_id)
}

/// Löscht den Editor-Zustand eines Canvas-Modells aus SQLite.
#[tauri::command]
pub fn delete_canvas_model_design(
    state: State<AppState>,
    model_id: String,
) -> Result<(), String> {
    let db = state.db.lock().map_err(|e| format!("DB Lock: {}", e))?;
    db.delete_canvas_model_design(&model_id)
}
