use std::fs;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use tauri::Manager;
use chrono::{DateTime, Utc};
use rand::seq::SliceRandom;
use rand::thread_rng;

/// Metadata für ein Dataset
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DatasetInfo {
    pub id: String,
    pub name: String,
    pub model_id: String,           // Zugehöriges Modell
    pub source: String,             // "local" oder "huggingface"
    pub source_path: Option<String>,
    pub size_bytes: u64,
    pub file_count: usize,
    pub created_at: DateTime<Utc>,
    pub status: DatasetStatus,      // unused, split, etc.
    pub split_info: Option<SplitInfo>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum DatasetStatus {
    Unused,     // Noch nicht gesplittet
    Split,      // Bereits in train/val/test aufgeteilt
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SplitInfo {
    pub train_count: usize,
    pub val_count: usize,
    pub test_count: usize,
    pub train_ratio: f32,
    pub val_ratio: f32,
    pub test_ratio: f32,
}

/// HuggingFace Dataset Struktur
#[derive(Debug, Serialize, Deserialize)]
pub struct HuggingFaceDataset {
    pub id: String,
    #[serde(rename = "author")]
    pub author: Option<String>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    pub tags: Option<Vec<String>>,
    #[serde(rename = "lastModified")]
    pub last_modified: Option<String>,
    pub description: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HuggingFaceDatasetFile {
    #[serde(alias = "rfilename", alias = "path")]
    pub filename: String,
    pub size: Option<u64>,
    #[serde(rename = "type")]
    pub file_type: Option<String>,
}

// ============ Helper Functions ============

/// Holt den Pfad zum Dataset-Verzeichnis eines Modells
fn get_model_datasets_dir(app_handle: &tauri::AppHandle, model_id: &str) -> Result<PathBuf, String> {
    let data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Konnte App-Daten-Verzeichnis nicht finden: {}", e))?;
    
    let datasets_dir = data_dir.join("models").join(model_id).join("datasets");
    
    if !datasets_dir.exists() {
        fs::create_dir_all(&datasets_dir)
            .map_err(|e| format!("Konnte Datasets-Verzeichnis nicht erstellen: {}", e))?;
    }
    
    Ok(datasets_dir)
}

/// Holt den globalen Datasets-Metadata-Pfad
fn get_datasets_metadata_path(app_handle: &tauri::AppHandle) -> Result<PathBuf, String> {
    let data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Konnte App-Daten-Verzeichnis nicht finden: {}", e))?;
    
    Ok(data_dir.join("datasets_metadata.json"))
}

/// Berechnet die Größe eines Verzeichnisses rekursiv
fn calculate_dir_size(path: &Path) -> Result<(u64, usize), String> {
    let mut total_size: u64 = 0;
    let mut file_count: usize = 0;
    
    if path.is_file() {
        let metadata = fs::metadata(path)
            .map_err(|e| format!("Konnte Metadaten nicht lesen: {}", e))?;
        return Ok((metadata.len(), 1));
    }
    
    if !path.exists() {
        return Ok((0, 0));
    }
    
    let entries = fs::read_dir(path)
        .map_err(|e| format!("Konnte Verzeichnis nicht lesen: {}", e))?;
    
    for entry in entries {
        let entry = entry.map_err(|e| format!("Fehler beim Lesen: {}", e))?;
        let entry_path = entry.path();
        
        if entry_path.is_file() {
            let metadata = fs::metadata(&entry_path)
                .map_err(|e| format!("Konnte Metadaten nicht lesen: {}", e))?;
            total_size += metadata.len();
            file_count += 1;
        } else if entry_path.is_dir() {
            let (sub_size, sub_count) = calculate_dir_size(&entry_path)?;
            total_size += sub_size;
            file_count += sub_count;
        }
    }
    
    Ok((total_size, file_count))
}

/// Kopiert ein Verzeichnis rekursiv
fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<(), String> {
    if !dst.exists() {
        fs::create_dir_all(dst)
            .map_err(|e| format!("Konnte Zielverzeichnis nicht erstellen: {}", e))?;
    }
    
    let entries = fs::read_dir(src)
        .map_err(|e| format!("Konnte Quellverzeichnis nicht lesen: {}", e))?;
    
    for entry in entries {
        let entry = entry.map_err(|e| format!("Fehler beim Lesen: {}", e))?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        
        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else {
            fs::copy(&src_path, &dst_path)
                .map_err(|e| format!("Konnte Datei nicht kopieren: {} - {}", src_path.display(), e))?;
        }
    }
    
    Ok(())
}

/// Speichert Dataset-Metadata
fn save_dataset_metadata(app_handle: &tauri::AppHandle, dataset_info: &DatasetInfo) -> Result<(), String> {
    let metadata_path = get_datasets_metadata_path(app_handle)?;
    
    let mut datasets: Vec<DatasetInfo> = if metadata_path.exists() {
        let content = fs::read_to_string(&metadata_path)
            .map_err(|e| format!("Konnte Metadata nicht lesen: {}", e))?;
        serde_json::from_str(&content).unwrap_or_default()
    } else {
        Vec::new()
    };
    
    // Entferne altes Dataset mit gleicher ID falls vorhanden
    datasets.retain(|d| d.id != dataset_info.id);
    datasets.push(dataset_info.clone());
    
    let content = serde_json::to_string_pretty(&datasets)
        .map_err(|e| format!("Konnte Metadata nicht serialisieren: {}", e))?;
    
    // Erstelle Parent-Verzeichnis falls nötig
    if let Some(parent) = metadata_path.parent() {
        fs::create_dir_all(parent).ok();
    }
    
    fs::write(&metadata_path, content)
        .map_err(|e| format!("Konnte Metadata nicht speichern: {}", e))?;
    
    Ok(())
}

/// Holt alle Dateien in einem Verzeichnis (flach)
fn get_files_in_dir(path: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    
    if !path.exists() {
        return Ok(files);
    }
    
    let entries = fs::read_dir(path)
        .map_err(|e| format!("Konnte Verzeichnis nicht lesen: {}", e))?;
    
    for entry in entries {
        let entry = entry.map_err(|e| format!("Fehler beim Lesen: {}", e))?;
        let entry_path = entry.path();
        
        if entry_path.is_file() {
            files.push(entry_path);
        }
    }
    
    Ok(files)
}

/// Konvertiert ein DB-Dataset zu DatasetInfo
fn convert_db_dataset_to_info(
    db_dataset: crate::database::Dataset,
    _app_handle: &tauri::AppHandle,
) -> DatasetInfo {
    let dataset_path = Path::new(&db_dataset.file_path);
    
    // Check if dataset has been split by looking for train/val/test dirs
    let has_split_dirs = dataset_path.join("train").exists() 
        || dataset_path.join("val").exists() 
        || dataset_path.join("test").exists();
    
    // Calculate size from appropriate directory
    let (size_bytes, file_count) = if has_split_dirs {
        // Calculate total from all split directories
        let mut total_size = 0u64;
        let mut total_files = 0usize;
        
        for split in &["train", "val", "test"] {
            let split_path = dataset_path.join(split);
            if split_path.exists() {
                if let Ok((size, count)) = calculate_dir_size(&split_path) {
                    total_size += size;
                    total_files += count;
                }
            }
        }
        (total_size, total_files)
    } else {
        // Not split yet, check unused directory or root
        let unused_path = dataset_path.join("unused");
        if unused_path.exists() {
            calculate_dir_size(&unused_path).unwrap_or((0, 0))
        } else {
            calculate_dir_size(dataset_path).unwrap_or((0, 0))
        }
    };
    
    // Extract model_id from path (format: models/{model_id}/datasets/{dataset_id})
    let model_id = dataset_path
        .to_string_lossy()
        .split("/models/")
        .nth(1)
        .and_then(|s| s.split("/datasets/").next())
        .unwrap_or("unknown")
        .to_string();
    
    // Determine status and split_info from filesystem
    let status = if has_split_dirs {
        DatasetStatus::Split
    } else {
        DatasetStatus::Unused
    };
    
    // Calculate split_info if split
    let split_info = if status == DatasetStatus::Split {
        Some(calculate_split_info(dataset_path))
    } else {
        None
    };
    
    DatasetInfo {
        id: db_dataset.id,
        name: db_dataset.name,
        model_id,
        source: "local".to_string(),
        source_path: Some(db_dataset.file_path),
        size_bytes,
        file_count,
        created_at: Utc::now(),
        status,
        split_info,
    }
}

/// Berechnet Split-Info aus Filesystem
fn calculate_split_info(dataset_path: &Path) -> SplitInfo {
    let train_count = get_files_in_dir(&dataset_path.join("train"))
        .map(|f| f.len())
        .unwrap_or(0);
    let val_count = get_files_in_dir(&dataset_path.join("val"))
        .map(|f| f.len())
        .unwrap_or(0);
    let test_count = get_files_in_dir(&dataset_path.join("test"))
        .map(|f| f.len())
        .unwrap_or(0);
    
    let total = (train_count + val_count + test_count) as f32;
    
    SplitInfo {
        train_count,
        val_count,
        test_count,
        train_ratio: if total > 0.0 { train_count as f32 / total } else { 0.0 },
        val_ratio: if total > 0.0 { val_count as f32 / total } else { 0.0 },
        test_ratio: if total > 0.0 { test_count as f32 / total } else { 0.0 },
    }
}

// ============ TAURI COMMANDS ============

/// Listet alle Datasets für ein bestimmtes Modell
#[tauri::command]
pub fn list_datasets_for_model(
    app_handle: tauri::AppHandle,
    model_id: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<Vec<DatasetInfo>, String> {
    // CRITICAL: Use database with user_id filtering!
    let db = state.db.lock()
        .map_err(|e| format!("Database lock error: {}", e))?;
    
    let db_datasets = db.list_datasets()
        .map_err(|e| format!("Konnte Datasets nicht laden: {}", e))?;
    
    // Filter by model_id (from file_path) and convert to DatasetInfo
    let filtered: Vec<DatasetInfo> = db_datasets
        .into_iter()
        .map(|d| convert_db_dataset_to_info(d, &app_handle))
        .filter(|d| d.model_id == model_id)
        .filter(|d| {
            // Only return datasets that still exist in filesystem
            Path::new(&d.source_path.as_ref().unwrap_or(&String::new())).exists()
        })
        .collect();
    
    Ok(filtered)
}

/// Listet nur Test-bereite Datasets für ein bestimmtes Modell (für TestPanel)
#[tauri::command]
pub fn list_test_datasets_for_model(
    app_handle: tauri::AppHandle,
    model_id: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<Vec<DatasetInfo>, String> {
    let all_datasets = list_datasets_for_model(app_handle, model_id, state)?;
    
    let filtered: Vec<DatasetInfo> = all_datasets
        .into_iter()
        .filter(|d| {
            // Only include datasets that have been split and have a test set
            if d.status != DatasetStatus::Split {
                return false;
            }
            
            // Check if test_count > 0
            if let Some(ref split_info) = d.split_info {
                split_info.test_count > 0
            } else {
                false
            }
        })
        .collect();
    
    Ok(filtered)
}

/// Listet alle Datasets
#[tauri::command]
pub fn list_all_datasets(
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, crate::AppState>,
) -> Result<Vec<DatasetInfo>, String> {
    // CRITICAL: Use database with user_id filtering!
    let db = state.db.lock()
        .map_err(|e| format!("Database lock error: {}", e))?;
    
    let db_datasets = db.list_datasets()
        .map_err(|e| format!("Konnte Datasets nicht laden: {}", e))?;
    
    // Convert all DB datasets to DatasetInfo
    let datasets: Vec<DatasetInfo> = db_datasets
        .into_iter()
        .map(|d| convert_db_dataset_to_info(d, &app_handle))
        .filter(|d| {
            // Only return datasets that still exist in filesystem
            Path::new(&d.source_path.as_ref().unwrap_or(&String::new())).exists()
        })
        .collect();
    
    Ok(datasets)
}

/// Importiert ein lokales Dataset
#[tauri::command]
pub async fn import_local_dataset(
    app_handle: tauri::AppHandle,
    source_path: String,
    dataset_name: String,
    model_id: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<DatasetInfo, String> {
    let source = Path::new(&source_path);
    
    if !source.exists() {
        return Err("Quellverzeichnis existiert nicht".to_string());
    }
    
    if !source.is_dir() {
        return Err("Quelle muss ein Verzeichnis sein".to_string());
    }
    
    // Generiere eindeutige ID
    let dataset_id = format!("ds_local_{}", uuid::Uuid::new_v4().to_string().replace("-", "")[..12].to_string());
    
    // Zielverzeichnis: models/{model_id}/datasets/{dataset_id}/unused
    let datasets_dir = get_model_datasets_dir(&app_handle, &model_id)?;
    let target_dir = datasets_dir.join(&dataset_id).join("unused");
    
    fs::create_dir_all(&target_dir)
        .map_err(|e| format!("Konnte Zielverzeichnis nicht erstellen: {}", e))?;
    
    // Berechne Größe
    let (size_bytes, file_count) = calculate_dir_size(source)?;
    
    // Kopiere Dateien
    copy_dir_recursive(source, &target_dir)?;
    
    // Erstelle DatasetInfo
    let dataset_info = DatasetInfo {
        id: dataset_id.clone(),
        name: dataset_name.clone(),
        model_id: model_id.clone(),
        source: "local".to_string(),
        source_path: Some(source_path.clone()),
        size_bytes,
        file_count,
        created_at: Utc::now(),
        status: DatasetStatus::Unused,
        split_info: None,
    };
    
    // Speichere Metadata JSON (deprecated but kept for backward compatibility)
    save_dataset_metadata(&app_handle, &dataset_info)?;
    
    // CRITICAL: Use database with user isolation!
    println!("[Dataset] Saving to database with user isolation: {}", dataset_id);
    
    let db = state.db.lock()
        .map_err(|e| format!("Failed to lock database: {}", e))?;
    
    // CRITICAL: Save dataset root path, not unused subdir!
    let dataset_root_path = datasets_dir.join(&dataset_id);
    
    let db_dataset = crate::database::Dataset {
        id: dataset_id.clone(),
        name: dataset_name.clone(),
        file_path: dataset_root_path.to_string_lossy().to_string(),
        file_type: "directory".to_string(),
        size_bytes: Some(size_bytes as i64),
        rows_count: None,
        columns_count: None,
        validated: false,
        created_at: Utc::now().to_rfc3339(),
    };
    
    db.save_dataset(&db_dataset)
        .map_err(|e| format!("Failed to save to database: {}", e))?;
    
    println!("[Dataset] ✅ Dataset saved to database with user_id: {}", dataset_id);
    
    Ok(dataset_info)
}

/// Löscht ein Dataset
#[tauri::command]
pub fn delete_dataset(
    app_handle: tauri::AppHandle,
    dataset_id: String,
    model_id: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<(), String> {
    let datasets_dir = get_model_datasets_dir(&app_handle, &model_id)?;
    let dataset_path = datasets_dir.join(&dataset_id);
    
    // Lösche Verzeichnis
    if dataset_path.exists() {
        fs::remove_dir_all(&dataset_path)
            .map_err(|e| format!("Konnte Dataset nicht löschen: {}", e))?;
    }
    
    // Aktualisiere Metadata JSON (deprecated but kept for backward compatibility)
    let metadata_path = get_datasets_metadata_path(&app_handle)?;
    if metadata_path.exists() {
        let content = fs::read_to_string(&metadata_path)
            .map_err(|e| format!("Konnte Metadata nicht lesen: {}", e))?;
        
        let mut datasets: Vec<DatasetInfo> = serde_json::from_str(&content).unwrap_or_default();
        datasets.retain(|d| d.id != dataset_id);
        
        let content = serde_json::to_string_pretty(&datasets)
            .map_err(|e| format!("Konnte Metadata nicht serialisieren: {}", e))?;
        
        fs::write(&metadata_path, content)
            .map_err(|e| format!("Konnte Metadata nicht speichern: {}", e))?;
    }
    
    // CRITICAL: Delete from database (checks user_id automatically!)
    println!("[Dataset] Deleting from database with user isolation: {}", dataset_id);
    
    let db = state.db.lock()
        .map_err(|e| format!("Failed to lock database: {}", e))?;
    
    db.conn.execute("DELETE FROM datasets WHERE id = ?1", [&dataset_id])
        .map_err(|e| format!("Failed to delete from database: {}", e))?;
    
    println!("[Dataset] ✅ Dataset deleted from database: {}", dataset_id);
    
    Ok(())
}

/// Splittet ein Dataset in train/val/test
#[tauri::command]
pub fn split_dataset(
    app_handle: tauri::AppHandle,
    dataset_id: String,
    model_id: String,
    train_ratio: f32,
    val_ratio: f32,
    test_ratio: f32,
    state: tauri::State<'_, crate::AppState>,
) -> Result<DatasetInfo, String> {
    // Validiere Ratios
    let total = train_ratio + val_ratio + test_ratio;
    if (total - 1.0).abs() > 0.01 {
        return Err(format!("Split-Verhältnisse müssen zusammen 1.0 ergeben (aktuell: {})", total));
    }
    
    let datasets_dir = get_model_datasets_dir(&app_handle, &model_id)?;
    let dataset_dir = datasets_dir.join(&dataset_id);
    let unused_dir = dataset_dir.join("unused");
    
    if !unused_dir.exists() {
        return Err("Keine ungeteilten Daten gefunden".to_string());
    }
    
    // Hole alle Dateien mit Größe
    let files = get_files_in_dir(&unused_dir)?;
    
    if files.is_empty() {
        return Err("Keine Dateien zum Splitten gefunden".to_string());
    }
    
    // Erstelle Vektor mit (Pfad, Größe)
    let mut files_with_size: Vec<(PathBuf, u64)> = files
        .iter()
        .filter_map(|path| {
            fs::metadata(path).ok().map(|meta| (path.clone(), meta.len()))
        })
        .collect();
    
    // Berechne Gesamtgröße
    let total_size: u64 = files_with_size.iter().map(|(_, size)| size).sum();
    let target_train_size = (total_size as f64 * train_ratio as f64) as u64;
    let target_val_size = (total_size as f64 * val_ratio as f64) as u64;
    
    // Sortiere nach Größe (optional: für bessere Verteilung)
    files_with_size.sort_by_key(|(_, size)| *size);
    
    // Mische für Zufälligkeit
    let mut rng = thread_rng();
    files_with_size.shuffle(&mut rng);
    
    // Verteile Dateien nach Gesamtgröße
    let mut train_files = Vec::new();
    let mut val_files = Vec::new();
    let mut test_files = Vec::new();
    
    let mut train_current_size = 0u64;
    let mut val_current_size = 0u64;
    
    for (path, size) in files_with_size {
        if train_current_size < target_train_size {
            train_current_size += size;
            train_files.push(path);
        } else if val_current_size < target_val_size {
            val_current_size += size;
            val_files.push(path);
        } else {
            test_files.push(path);
        }
    }
    
    let train_count = train_files.len();
    let val_count = val_files.len();
    let test_count = test_files.len();
    
    // Erstelle Split-Verzeichnisse
    let train_dir = dataset_dir.join("train");
    let val_dir = dataset_dir.join("val");
    let test_dir = dataset_dir.join("test");
    
    fs::create_dir_all(&train_dir).map_err(|e| format!("Konnte train-Verzeichnis nicht erstellen: {}", e))?;
    fs::create_dir_all(&val_dir).map_err(|e| format!("Konnte val-Verzeichnis nicht erstellen: {}", e))?;
    fs::create_dir_all(&test_dir).map_err(|e| format!("Konnte test-Verzeichnis nicht erstellen: {}", e))?;
    
    // Verschiebe Dateien in Train
    for file in &train_files {
        let filename = file.file_name().unwrap();
        let target_path = train_dir.join(filename);
        fs::rename(file, &target_path)
            .or_else(|_| -> Result<(), std::io::Error> {
                fs::copy(file, &target_path)?;
                fs::remove_file(file)?;
                Ok(())
            })
            .map_err(|e| format!("Konnte Datei nicht verschieben: {}", e))?;
    }
    
    // Verschiebe Dateien in Val
    for file in &val_files {
        let filename = file.file_name().unwrap();
        let target_path = val_dir.join(filename);
        fs::rename(file, &target_path)
            .or_else(|_| -> Result<(), std::io::Error> {
                fs::copy(file, &target_path)?;
                fs::remove_file(file)?;
                Ok(())
            })
            .map_err(|e| format!("Konnte Datei nicht verschieben: {}", e))?;
    }
    
    // Verschiebe Dateien in Test
    for file in &test_files {
        let filename = file.file_name().unwrap();
        let target_path = test_dir.join(filename);
        fs::rename(file, &target_path)
            .or_else(|_| -> Result<(), std::io::Error> {
                // Falls rename fehlschlägt (z.B. cross-device), kopiere und lösche
                fs::copy(file, &target_path)?;
                fs::remove_file(file)
            })
            .map_err(|e| format!("Konnte Datei nicht verschieben: {}", e))?;
    }
    
    // Lösche unused-Verzeichnis wenn leer
    if get_files_in_dir(&unused_dir)?.is_empty() {
        fs::remove_dir(&unused_dir).ok();
    }
    
    // Update Dataset Metadata
    let metadata_path = get_datasets_metadata_path(&app_handle)?;
    let content = fs::read_to_string(&metadata_path)
        .map_err(|e| format!("Konnte Metadata nicht lesen: {}", e))?;
    
    let mut datasets: Vec<DatasetInfo> = serde_json::from_str(&content).unwrap_or_default();
    
    let dataset = datasets.iter_mut()
        .find(|d| d.id == dataset_id)
        .ok_or_else(|| "Dataset nicht gefunden".to_string())?;
    
    dataset.status = DatasetStatus::Split;
    dataset.split_info = Some(SplitInfo {
        train_count,
        val_count,
        test_count,
        train_ratio,
        val_ratio,
        test_ratio,
    });
    
    let updated_dataset = dataset.clone();
    
    let content = serde_json::to_string_pretty(&datasets)
        .map_err(|e| format!("Konnte Metadata nicht serialisieren: {}", e))?;
    
    fs::write(&metadata_path, content)
        .map_err(|e| format!("Konnte Metadata nicht speichern: {}", e))?;
    
    // CRITICAL: Update database file_path to point to dataset root, not unused!
    println!("[Dataset] Updating database after split for: {}", dataset_id);
    
    let db = state.db.lock()
        .map_err(|e| format!("Failed to lock database: {}", e))?;
    
    // Update file_path to dataset root directory (remove /unused)
    let new_file_path = dataset_dir.to_string_lossy().to_string();
    
    db.conn.execute(
        "UPDATE datasets SET file_path = ?1 WHERE id = ?2",
        [&new_file_path, &dataset_id]
    ).map_err(|e| format!("Failed to update database: {}", e))?;
    
    println!("[Dataset] ✅ Database updated with new file_path: {}", new_file_path);
    
    Ok(updated_dataset)
}

// ============ HUGGING FACE DATASET INTEGRATION ============

/// Sucht nach Datasets auf HuggingFace
#[tauri::command]
pub async fn search_huggingface_datasets(
    query: String, 
    limit: Option<u32>,
    filter_task: Option<String>,
    filter_language: Option<String>,
    filter_size: Option<String>,
) -> Result<Vec<HuggingFaceDataset>, String> {
    let limit = limit.unwrap_or(20);
    
    // Baue URL mit Filtern
    let mut url = format!(
        "https://huggingface.co/api/datasets?search={}&limit={}&sort=downloads&direction=-1",
        urlencoding::encode(&query),
        limit
    );
    
    // Füge Filter hinzu
    if let Some(task) = filter_task {
        if !task.is_empty() {
            url.push_str(&format!("&task_categories={}", urlencoding::encode(&task)));
        }
    }
    
    if let Some(lang) = filter_language {
        if !lang.is_empty() {
            url.push_str(&format!("&language={}", urlencoding::encode(&lang)));
        }
    }
    
    if let Some(size) = filter_size {
        if !size.is_empty() {
            url.push_str(&format!("&size_categories={}", urlencoding::encode(&size)));
        }
    }
    
    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .header("User-Agent", "FrameTrain-Desktop/1.0")
        .send()
        .await
        .map_err(|e| format!("HTTP Fehler: {}", e))?;
    
    if !response.status().is_success() {
        return Err(format!("API Fehler: {}", response.status()));
    }
    
    let datasets: Vec<HuggingFaceDataset> = response
        .json()
        .await
        .map_err(|e| format!("JSON Parse Fehler: {}", e))?;
    
    Ok(datasets)
}

/// Holt Dateiliste eines HuggingFace Datasets
#[tauri::command]
pub async fn get_huggingface_dataset_files(repo_id: String) -> Result<Vec<HuggingFaceDatasetFile>, String> {
    let url = format!("https://huggingface.co/api/datasets/{}/tree/main", repo_id);
    
    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .header("User-Agent", "FrameTrain-Desktop/1.0")
        .send()
        .await
        .map_err(|e| format!("HTTP Fehler: {}", e))?;
    
    if !response.status().is_success() {
        return Err(format!("API Fehler: {}", response.status()));
    }
    
    let files: Vec<HuggingFaceDatasetFile> = response
        .json()
        .await
        .map_err(|e| format!("JSON Parse Fehler: {}", e))?;
    
    Ok(files)
}

/// Lädt ein Dataset von HuggingFace herunter
#[tauri::command]
pub async fn download_huggingface_dataset(
    app_handle: tauri::AppHandle,
    repo_id: String,
    dataset_name: String,
    model_id: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<DatasetInfo, String> {
    // Generiere eindeutige ID
    let dataset_id = format!("ds_hf_{}", uuid::Uuid::new_v4().to_string().replace("-", "")[..12].to_string());
    
    // Zielverzeichnis
    let datasets_dir = get_model_datasets_dir(&app_handle, &model_id)?;
    let target_dir = datasets_dir.join(&dataset_id).join("unused");
    
    fs::create_dir_all(&target_dir)
        .map_err(|e| format!("Konnte Zielverzeichnis nicht erstellen: {}", e))?;
    
    // Hole Dateiliste
    let files = get_huggingface_dataset_files(repo_id.clone()).await?;
    
    let client = reqwest::Client::new();
    let mut total_size: u64 = 0;
    let mut file_count: usize = 0;

    // Erkenne ob das Dataset bereits in train/validation/test aufgeteilt ist
    let has_presplit = files.iter().any(|f| {
        let name = f.filename.to_lowercase();
        name.starts_with("train/") || name.starts_with("validation/") || name.starts_with("test/")
    });

    println!("[Dataset] HF pre-split detected: {}", has_presplit);

    // Bei pre-split: dataset_root direkt verwenden (kein /unused Unterordner)
    let download_base_dir = if has_presplit {
        let root = datasets_dir.join(&dataset_id);
        fs::create_dir_all(&root)
            .map_err(|e| format!("Konnte Root-Verzeichnis nicht erstellen: {}", e))?;
        root
    } else {
        target_dir.clone()
    };

    // Filtere relevante Dateien (keine reinen Verzeichnis-Einträge)
    let important_files: Vec<&HuggingFaceDatasetFile> = files.iter()
        .filter(|f| {
            if let Some(ref ft) = f.file_type {
                if ft == "directory" {
                    return false;
                }
            }
            let name = f.filename.to_lowercase();
            name.ends_with(".json") ||
            name.ends_with(".jsonl") ||
            name.ends_with(".csv") ||
            name.ends_with(".tsv") ||
            name.ends_with(".txt") ||
            name.ends_with(".parquet") ||
            name.ends_with(".arrow") ||
            name.ends_with(".md") ||
            name.contains("train") ||
            name.contains("test") ||
            name.contains("val") ||
            name.contains("data")
        })
        .collect();

    for file in important_files {
        // Normalisiere 'validation/' -> 'val/' für interne Konsistenz
        let normalized_filename = file.filename
            .replace("validation/", "val/")
            .replace("valid/", "val/");

        let encoded_filename = file.filename
            .split('/')
            .map(|part| urlencoding::encode(part).into_owned())
            .collect::<Vec<_>>()
            .join("/");

        let file_url = format!(
            "https://huggingface.co/datasets/{}/resolve/main/{}",
            repo_id,
            encoded_filename
        );

        let response = client
            .get(&file_url)
            .header("User-Agent", "FrameTrain-Desktop/1.0")
            .send()
            .await
            .map_err(|e| format!("Download Fehler für {}: {}", file.filename, e))?;

        if response.status().is_success() {
            let bytes = response.bytes().await
                .map_err(|e| format!("Konnte Datei nicht lesen: {}", e))?;

            let file_path = download_base_dir.join(&normalized_filename);
            if let Some(parent) = file_path.parent() {
                fs::create_dir_all(parent).ok();
            }

            fs::write(&file_path, &bytes)
                .map_err(|e| format!("Konnte Datei nicht speichern: {}", e))?;

            total_size += bytes.len() as u64;
            file_count += 1;
        }
    }
    
    // Erstelle DatasetInfo
    let dataset_info = DatasetInfo {
        id: dataset_id,
        name: dataset_name,
        model_id,
        source: "huggingface".to_string(),
        source_path: Some(repo_id),
        size_bytes: total_size,
        file_count,
        created_at: Utc::now(),
        status: if has_presplit { DatasetStatus::Split } else { DatasetStatus::Unused },
        split_info: None,
    };
    
    save_dataset_metadata(&app_handle, &dataset_info)?;

    // CRITICAL: Save to database so list_datasets_for_model can find it!
    println!("[Dataset] Saving HF dataset to database: {}", dataset_info.id);

    let db = state.db.lock()
        .map_err(|e| format!("Failed to lock database: {}", e))?;

    let dataset_root_path = datasets_dir.join(&dataset_info.id);

    let db_dataset = crate::database::Dataset {
        id: dataset_info.id.clone(),
        name: dataset_info.name.clone(),
        file_path: dataset_root_path.to_string_lossy().to_string(),
        file_type: "directory".to_string(),
        size_bytes: Some(total_size as i64),
        rows_count: None,
        columns_count: None,
        validated: false,
        created_at: Utc::now().to_rfc3339(),
    };

    db.save_dataset(&db_dataset)
        .map_err(|e| format!("Failed to save HF dataset to database: {}", e))?;

    println!("[Dataset] ✅ HF dataset saved to database: {}", dataset_info.id);

    Ok(dataset_info)
}

/// Holt verfügbare Filter-Optionen für HuggingFace
#[tauri::command]
pub fn get_dataset_filter_options() -> Result<DatasetFilterOptions, String> {
    Ok(DatasetFilterOptions {
        tasks: vec![
            "text-classification".to_string(),
            "token-classification".to_string(),
            "question-answering".to_string(),
            "summarization".to_string(),
            "translation".to_string(),
            "text-generation".to_string(),
            "fill-mask".to_string(),
            "conversational".to_string(),
            "image-classification".to_string(),
            "object-detection".to_string(),
            "image-segmentation".to_string(),
            "audio-classification".to_string(),
            "automatic-speech-recognition".to_string(),
        ],
        languages: vec![
            "en".to_string(),
            "de".to_string(),
            "fr".to_string(),
            "es".to_string(),
            "zh".to_string(),
            "ja".to_string(),
            "ko".to_string(),
            "ru".to_string(),
            "ar".to_string(),
            "pt".to_string(),
            "it".to_string(),
            "nl".to_string(),
            "pl".to_string(),
            "multilingual".to_string(),
        ],
        sizes: vec![
            "n<1K".to_string(),
            "1K<n<10K".to_string(),
            "10K<n<100K".to_string(),
            "100K<n<1M".to_string(),
            "1M<n<10M".to_string(),
            "10M<n<100M".to_string(),
            "100M<n<1B".to_string(),
            "n>1B".to_string(),
        ],
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DatasetFilterOptions {
    pub tasks: Vec<String>,
    pub languages: Vec<String>,
    pub sizes: Vec<String>,
}

// ============ File Management Functions ============

#[derive(Debug, Serialize, Deserialize)]
pub struct FileInfo {
    pub name: String,
    pub path: String,
    pub size: u64,
    pub is_dir: bool,
    pub split: String, // "train", "val", "test", or "unsplit"
}

/// Holt alle Dateien eines Datasets mit Split-Information
#[tauri::command]
pub fn get_dataset_files(
    app_handle: tauri::AppHandle,
    dataset_id: String
) -> Result<Vec<FileInfo>, String> {
    let metadata_path = get_datasets_metadata_path(&app_handle)?;
    let datasets: Vec<DatasetInfo> = if metadata_path.exists() {
        let content = fs::read_to_string(&metadata_path)
            .map_err(|e| format!("Konnte Metadata nicht lesen: {}", e))?;
        serde_json::from_str(&content).unwrap_or_default()
    } else {
        return Err("Keine Datasets gefunden".to_string());
    };
    
    let dataset = datasets.iter()
        .find(|d| d.id == dataset_id)
        .ok_or_else(|| "Dataset nicht gefunden".to_string())?;
    
    let dataset_dir = get_model_datasets_dir(&app_handle, &dataset.model_id)?
        .join(&dataset.id);
    
    let mut files = Vec::new();
    
    // Sammle Dateien aus allen Split-Ordnern
    for split in &["train", "val", "test"] {
        let split_dir = dataset_dir.join(split);
        if split_dir.exists() {
            collect_files_recursive(&split_dir, split, &mut files)?;
        }
    }
    
    // Sammle Dateien aus Hauptverzeichnis (unsplit)
    if dataset_dir.exists() {
        let entries = fs::read_dir(&dataset_dir)
            .map_err(|e| format!("Konnte Verzeichnis nicht lesen: {}", e))?;
        
        for entry in entries {
            let entry = entry.map_err(|e| format!("Fehler beim Lesen: {}", e))?;
            let path = entry.path();
            let file_name = path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
            
            // Überspringe Split-Ordner
            if file_name == "train" || file_name == "val" || file_name == "test" {
                continue;
            }
            
            if path.is_file() {
                let metadata = fs::metadata(&path)
                    .map_err(|e| format!("Konnte Metadaten nicht lesen: {}", e))?;
                
                files.push(FileInfo {
                    name: file_name.to_string(),
                    path: path.to_string_lossy().to_string(),
                    size: metadata.len(),
                    is_dir: false,
                    split: "unsplit".to_string(),
                });
            }
        }
    }
    
    Ok(files)
}

fn collect_files_recursive(
    dir: &Path,
    split: &str,
    files: &mut Vec<FileInfo>
) -> Result<(), String> {
    let entries = fs::read_dir(dir)
        .map_err(|e| format!("Konnte Verzeichnis nicht lesen: {}", e))?;
    
    for entry in entries {
        let entry = entry.map_err(|e| format!("Fehler beim Lesen: {}", e))?;
        let path = entry.path();
        
        if path.is_file() {
            let metadata = fs::metadata(&path)
                .map_err(|e| format!("Konnte Metadaten nicht lesen: {}", e))?;
            
            let file_name = path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();
            
            files.push(FileInfo {
                name: file_name,
                path: path.to_string_lossy().to_string(),
                size: metadata.len(),
                is_dir: false,
                split: split.to_string(),
            });
        } else if path.is_dir() {
            collect_files_recursive(&path, split, files)?;
        }
    }
    
    Ok(())
}

/// Liest den Inhalt einer Datei
#[tauri::command]
pub fn read_dataset_file(file_path: String) -> Result<String, String> {
    let path = Path::new(&file_path);
    
    if !path.exists() {
        return Err("Datei nicht gefunden".to_string());
    }
    
    // Versuche als Text zu lesen
    let content = fs::read_to_string(path)
        .map_err(|e| {
            // Falls Textlesen fehlschlägt, versuche als Binary und zeige Hex
            if let Ok(bytes) = fs::read(path) {
                if bytes.len() > 10000 {
                    return format!("[Binärdatei - {} Bytes - Zu groß zur Anzeige]", bytes.len());
                }
                // Zeige ersten Teil als Hex
                let hex = bytes.iter()
                    .take(1000)
                    .map(|b| format!("{:02x}", b))
                    .collect::<Vec<_>>()
                    .join(" ");
                return format!("[Binärdatei - Hex Preview]\n{}", hex);
            }
            format!("Konnte Datei nicht lesen: {}", e)
        })?;
    
    // Begrenze Textlänge für UI
    if content.len() > 100000 {
        Ok(format!("{}\n\n[... {} weitere Zeichen ...]", 
            &content[..100000], 
            content.len() - 100000))
    } else {
        Ok(content)
    }
}

/// Verschiebt Dateien zu einem anderen Split
#[tauri::command]
pub fn move_dataset_files(
    app_handle: tauri::AppHandle,
    dataset_id: String,
    file_paths: Vec<String>,
    target_split: String
) -> Result<(), String> {
    let metadata_path = get_datasets_metadata_path(&app_handle)?;
    let datasets: Vec<DatasetInfo> = if metadata_path.exists() {
        let content = fs::read_to_string(&metadata_path)
            .map_err(|e| format!("Konnte Metadata nicht lesen: {}", e))?;
        serde_json::from_str(&content).unwrap_or_default()
    } else {
        return Err("Keine Datasets gefunden".to_string());
    };
    
    let dataset = datasets.iter()
        .find(|d| d.id == dataset_id)
        .ok_or_else(|| "Dataset nicht gefunden".to_string())?;
    
    let dataset_dir = get_model_datasets_dir(&app_handle, &dataset.model_id)?
        .join(&dataset.id);
    
    let target_dir = dataset_dir.join(&target_split);
    fs::create_dir_all(&target_dir)
        .map_err(|e| format!("Konnte Zielverzeichnis nicht erstellen: {}", e))?;
    
    for file_path_str in file_paths {
        let source_path = Path::new(&file_path_str);
        if !source_path.exists() {
            continue;
        }
        
        let file_name = source_path.file_name()
            .ok_or_else(|| "Ungültiger Dateipfad".to_string())?;
        
        let target_path = target_dir.join(file_name);
        
        fs::rename(source_path, &target_path)
            .map_err(|e| format!("Konnte Datei nicht verschieben: {}", e))?;
    }
    
    Ok(())
}

/// Löscht Dateien aus einem Dataset
#[tauri::command]
pub fn delete_dataset_files(
    _dataset_id: String,
    file_paths: Vec<String>
) -> Result<(), String> {
    for file_path_str in file_paths {
        let path = Path::new(&file_path_str);
        if path.exists() && path.is_file() {
            fs::remove_file(path)
                .map_err(|e| format!("Konnte Datei nicht löschen: {}", e))?;
        }
    }
    
    Ok(())
}

/// Fügt neue Dateien zu einem Dataset hinzu (über Frontend-Dialog)
#[tauri::command]
pub async fn add_files_to_dataset(
    app_handle: tauri::AppHandle,
    dataset_id: String,
    file_paths: Vec<String>
) -> Result<usize, String> {
    let metadata_path = get_datasets_metadata_path(&app_handle)?;
    let datasets: Vec<DatasetInfo> = if metadata_path.exists() {
        let content = fs::read_to_string(&metadata_path)
            .map_err(|e| format!("Konnte Metadata nicht lesen: {}", e))?;
        serde_json::from_str(&content).unwrap_or_default()
    } else {
        return Err("Keine Datasets gefunden".to_string());
    };
    
    let dataset = datasets.iter()
        .find(|d| d.id == dataset_id)
        .ok_or_else(|| "Dataset nicht gefunden".to_string())?;
    
    let dataset_dir = get_model_datasets_dir(&app_handle, &dataset.model_id)?
        .join(&dataset.id);
    
    let mut copied_count = 0;
    
    for file_path_str in file_paths {
        let source_file = Path::new(&file_path_str);
        
        if !source_file.exists() {
            continue;
        }
        
        let file_name = source_file.file_name()
            .ok_or_else(|| "Ungültiger Dateiname".to_string())?;
        
        let target_path = dataset_dir.join(file_name);
        
        fs::copy(&source_file, &target_path)
            .map_err(|e| format!("Konnte Datei nicht kopieren: {}", e))?;
        
        copied_count += 1;
    }
    
    Ok(copied_count)
}
