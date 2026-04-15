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
    // Nutzungs-Tracking
    pub training_count: i32,        // Wie oft für Training benutzt
    pub last_used_at: Option<String>, // ISO-8601 Zeitstempel letzter Benutzung
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
        training_count: db_dataset.training_count,
        last_used_at: db_dataset.last_used_at,
    }
}

/// Berechnet Split-Info aus Filesystem
fn calculate_split_info(dataset_path: &Path) -> SplitInfo {
    // Use recursive counting so nested files (e.g. sharded parquet) are all counted
    let train_count = calculate_dir_size(&dataset_path.join("train"))
        .map(|(_, c)| c)
        .unwrap_or(0);
    let val_count = calculate_dir_size(&dataset_path.join("val"))
        .map(|(_, c)| c)
        .unwrap_or(0);
    let test_count = calculate_dir_size(&dataset_path.join("test"))
        .map(|(_, c)| c)
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
        training_count: 0,
        last_used_at: None,
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
        training_count: 0,
        last_used_at: None,
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
    
    // Sammle alle splittbaren Dateien: aus unused/ und direkt aus Root
    let mut all_candidate_files: Vec<PathBuf> = Vec::new();
    
    if unused_dir.exists() {
        let mut unused_files = get_files_in_dir(&unused_dir)?;
        all_candidate_files.append(&mut unused_files);
    }
    
    // Auch Root-Dateien (die nicht in train/val/test/unused liegen) einschließen
    if let Ok(root_entries) = fs::read_dir(&dataset_dir) {
        for entry in root_entries.flatten() {
            let p = entry.path();
            let fname = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if p.is_file() && !matches!(fname, "train" | "val" | "test" | "unused") {
                all_candidate_files.push(p);
            }
        }
    }
    
    let files = all_candidate_files;
    
    if !unused_dir.exists() && files.is_empty() {
        return Err("Keine ungeteilten Daten gefunden".to_string());
    }
    
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
    // recursive=true fetches all files in all subdirectories (e.g. data/train-*.parquet)
    let url = format!("https://huggingface.co/api/datasets/{}/tree/main?recursive=true", repo_id);
    
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
    
    // Hole Dateiliste
    let files = get_huggingface_dataset_files(repo_id.clone()).await?;
    
    let client = reqwest::Client::new();
    let mut total_size: u64 = 0;
    let mut file_count: usize = 0;

    // Dataset-Root (immer ohne /unused)
    let dataset_root = datasets_dir.join(&dataset_id);
    fs::create_dir_all(&dataset_root)
        .map_err(|e| format!("Konnte Root-Verzeichnis nicht erstellen: {}", e))?;

    // Filtere relevante Dateien (keine reinen Verzeichnis-Einträge)
    let important_files: Vec<&HuggingFaceDatasetFile> = files.iter()
        .filter(|f| {
            if let Some(ref ft) = f.file_type {
                if ft == "directory" {
                    return false;
                }
            }
            // Filtere System-/Konfigurationsdateien aus
            let name = f.filename.to_lowercase();
            !name.ends_with(".gitattributes")
                && !name.ends_with(".gitignore")
                && name != ".git"
                && (name.ends_with(".json")
                    || name.ends_with(".jsonl")
                    || name.ends_with(".csv")
                    || name.ends_with(".tsv")
                    || name.ends_with(".txt")
                    || name.ends_with(".parquet")
                    || name.ends_with(".arrow")
                    || name.ends_with(".md")
                    || name.contains("train")
                    || name.contains("test")
                    || name.contains("val")
                    || name.contains("data"))
        })
        .collect();

    let mut has_any_split = false;

    for file in important_files {
        // Klassifiziere die Datei in einen Split
        let split = classify_hf_file(&file.filename);
        if split != "unused" {
            has_any_split = true;
        }

        // Ziel-Verzeichnis: {dataset_root}/{split}/{basename}
        let basename = file.filename.split('/').last()
            .unwrap_or(&file.filename)
            .to_string();

        let target_subdir = dataset_root.join(split);
        fs::create_dir_all(&target_subdir).ok();
        let file_path = target_subdir.join(&basename);

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

        println!("[Dataset] Downloading {} -> {}/{}/{}", file.filename, dataset_id, split, basename);

        let response = client
            .get(&file_url)
            .header("User-Agent", "FrameTrain-Desktop/1.0")
            .send()
            .await
            .map_err(|e| format!("Download Fehler für {}: {}", file.filename, e))?;

        if response.status().is_success() {
            let bytes = response.bytes().await
                .map_err(|e| format!("Konnte Datei nicht lesen: {}", e))?;

            fs::write(&file_path, &bytes)
                .map_err(|e| format!("Konnte Datei nicht speichern: {}", e))?;

            total_size += bytes.len() as u64;
            file_count += 1;
        } else {
            println!("[Dataset] Skipped {} (HTTP {})", file.filename, response.status());
        }
    }

    // Berechne split_info aus dem Filesystem
    let (final_status, final_split_info) = if has_any_split {
        let si = calculate_split_info(&dataset_root);
        (DatasetStatus::Split, Some(si))
    } else {
        (DatasetStatus::Unused, None)
    };

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
        status: final_status,
        split_info: final_split_info,
        training_count: 0,
        last_used_at: None,
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
        training_count: 0,
        last_used_at: None,
    };

    db.save_dataset(&db_dataset)
        .map_err(|e| format!("Failed to save HF dataset to database: {}", e))?;

    println!("[Dataset] ✅ HF dataset saved to database: {}", dataset_info.id);

    Ok(dataset_info)
}

// ============ HF FILE CLASSIFICATION ============

/// Klassifiziert eine HF-Datei anhand des Pfades/Namens in einen Split
/// Gibt "train", "val", "test" oder "unused" zurück
fn classify_hf_file(filename: &str) -> &'static str {
    let lower = filename.to_lowercase();
    let parts: Vec<&str> = lower.split('/').collect();
    let n = parts.len();

    // Verzeichnisbasierte Erkennung (z.B. "train/file.parquet")
    for part in &parts[..n.saturating_sub(1)] {
        match *part {
            "train" | "training" => return "train",
            "val" | "valid" | "validation" | "dev" | "develop" => return "val",
            "test" | "testing" => return "test",
            _ => {}
        }
    }

    // Dateinamensbasierte Erkennung (z.B. "train-00000-of-00005.parquet")
    let basename = parts.last().unwrap_or(&"");
    if basename.starts_with("train") || basename.contains("-train-") || basename.contains("_train") {
        return "train";
    }
    if basename.starts_with("val") || basename.starts_with("valid")
        || basename.starts_with("dev-") || basename.starts_with("dev_")
        || basename.contains("-val-") || basename.contains("_val")
        || basename.contains("-valid-") || basename.contains("_valid")
        || basename.contains("validation")
    {
        return "val";
    }
    if basename.starts_with("test") || basename.contains("-test-") || basename.contains("_test") {
        return "test";
    }

    "unused"
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
    dataset_id: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<Vec<FileInfo>, String> {
    // Primär: DB ist autoritativ (enthält immer den korrekten lokalen Pfad)
    let dataset_dir: PathBuf = {
        let db = state.db.lock()
            .map_err(|e| format!("DB-Lock-Fehler: {}", e))?;
        let all_db = db.list_datasets()
            .map_err(|e| format!("DB-Lesefehler: {}", e))?;
        if let Some(db_ds) = all_db.iter().find(|d| d.id == dataset_id) {
            PathBuf::from(&db_ds.file_path)
        } else {
            // Fallback: JSON-Metadata (rückwärtskompatibel)
            let metadata_path = get_datasets_metadata_path(&app_handle)?;
            let json_datasets: Vec<DatasetInfo> = if metadata_path.exists() {
                let content = fs::read_to_string(&metadata_path)
                    .map_err(|e| format!("Konnte Metadata nicht lesen: {}", e))?;
                serde_json::from_str(&content).unwrap_or_default()
            } else {
                Vec::new()
            };
            if let Some(ds) = json_datasets.iter().find(|d| d.id == dataset_id) {
                get_model_datasets_dir(&app_handle, &ds.model_id)?.join(&ds.id)
            } else {
                return Err(format!("Dataset '{}' nicht gefunden", dataset_id));
            }
        }
    };

    if !dataset_dir.exists() {
        return Err(format!("Dataset-Verzeichnis existiert nicht: {}", dataset_dir.display()));
    }

    let mut files = Vec::new();

    // 1. Sammle Dateien aus Split-Ordnern (train / val / test)
    for split in &["train", "val", "test"] {
        let split_dir = dataset_dir.join(split);
        if split_dir.exists() {
            collect_files_recursive(&split_dir, split, &mut files)?;
        }
    }

    // 2. Sammle Dateien aus "unused/" (noch nicht gesplittet)
    let unused_dir = dataset_dir.join("unused");
    if unused_dir.exists() {
        collect_files_recursive(&unused_dir, "unsplit", &mut files)?;
    }

    // 3. Sammle Dateien direkt im Root (weder in Split- noch in unused-Ordnern)
    let entries = fs::read_dir(&dataset_dir)
        .map_err(|e| format!("Konnte Verzeichnis nicht lesen: {}", e))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("Fehler beim Lesen: {}", e))?;
        let path = entry.path();
        let file_name = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");

        // Überspringe bekannte Unterordner
        if matches!(file_name, "train" | "val" | "test" | "unused") {
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

// ============ DATASET SPLIT IN HALF ============

#[derive(Debug, Serialize, Deserialize)]
pub struct SplitHalfResult {
    pub dataset_a: DatasetInfo,
    pub dataset_b: DatasetInfo,
}

/// Teilt einen Datensatz in zwei gleich große Hälften auf
#[tauri::command]
pub async fn split_dataset_in_half(
    app_handle: tauri::AppHandle,
    dataset_id: String,
    model_id: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<SplitHalfResult, String> {
    use std::process::Command;

    // 1. Finde den Datensatz-Pfad
    let db = state.db.lock().map_err(|e| format!("DB-Fehler: {}", e))?;
    let all_db = db.list_datasets().map_err(|e| format!("DB-Lesefehler: {}", e))?;
    let db_ds = all_db.iter().find(|d| d.id == dataset_id)
        .ok_or_else(|| format!("Datensatz '{}' nicht gefunden", dataset_id))?;
    let source_root = PathBuf::from(&db_ds.file_path);
    let ds_name = db_ds.name.clone();
    drop(db);

    if !source_root.exists() {
        return Err(format!("Datensatz-Verzeichnis existiert nicht: {}", source_root.display()));
    }

    // 2. Finde das Verzeichnis mit den eigentlichen Daten
    let data_dir = {
        let train = source_root.join("train");
        let unused = source_root.join("unused");
        if train.exists() {
            train
        } else if unused.exists() {
            unused
        } else {
            source_root.clone()
        }
    };

    // 3. Erstelle IDs + Zielverzeichnisse für die zwei neuen Datensätze
    let datasets_dir = get_model_datasets_dir(&app_handle, &model_id)?;

    let id_a = format!("ds_half_{}", uuid::Uuid::new_v4().to_string().replace("-", "")[..12].to_string());
    let id_b = format!("ds_half_{}", uuid::Uuid::new_v4().to_string().replace("-", "")[..12].to_string());

    let out_a = datasets_dir.join(&id_a).join("train");
    let out_b = datasets_dir.join(&id_b).join("train");

    fs::create_dir_all(&out_a).map_err(|e| format!("Konnte Zielordner A nicht erstellen: {}", e))?;
    fs::create_dir_all(&out_b).map_err(|e| format!("Konnte Zielordner B nicht erstellen: {}", e))?;

    // 4. Finde split_dataset.py
    let script_path = {
        let resource_path = app_handle.path().resource_dir()
            .map_err(|e| format!("Resource-Dir nicht gefunden: {}", e))?;
        let p1 = resource_path.join("python").join("split_dataset.py");
        let p2 = PathBuf::from("src-tauri/python/split_dataset.py");
        let p3 = PathBuf::from("/Users/karol/Desktop/Laufende_Projekte/FrameTrain/desktop-app2/src-tauri/python/split_dataset.py");
        if p1.exists() { p1 } else if p2.exists() { p2 } else if p3.exists() { p3 }
        else { return Err("split_dataset.py nicht gefunden".to_string()); }
    };

    // 5. Finde Python
    let python = {
        let candidates = ["python3", "python"];
        candidates.iter()
            .find(|p| Command::new(p).arg("--version").output().map(|o| o.status.success()).unwrap_or(false))
            .map(|s| s.to_string())
            .unwrap_or_else(|| "python3".to_string())
    };

    // 6. Führe Split-Script aus
    let output = Command::new(&python)
        .arg(script_path.to_string_lossy().to_string())
        .arg(data_dir.to_string_lossy().to_string())
        .arg(out_a.to_string_lossy().to_string())
        .arg(out_b.to_string_lossy().to_string())
        .output()
        .map_err(|e| format!("Konnte Split-Script nicht starten: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!("Split-Script fehlgeschlagen:\n{}\n{}", stderr, stdout));
    }

    // 7. Parse Ergebnis (letzte JSON-Zeile mit type=done)
    let stdout_str = String::from_utf8_lossy(&output.stdout);
    let mut total_a: usize = 0;
    let mut total_b: usize = 0;
    for line in stdout_str.lines() {
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
            if val.get("type").and_then(|t| t.as_str()) == Some("done") {
                total_a = val.get("total_a").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                total_b = val.get("total_b").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            }
            if val.get("type").and_then(|t| t.as_str()) == Some("error") {
                let msg = val.get("message").and_then(|m| m.as_str()).unwrap_or("Unbekannter Fehler");
                return Err(format!("Split-Fehler: {}", msg));
            }
        }
    }

    // 8. Berechne Größen
    let root_a = datasets_dir.join(&id_a);
    let root_b = datasets_dir.join(&id_b);
    let (size_a, files_a) = calculate_dir_size(&root_a).unwrap_or((0, 0));
    let (size_b, files_b) = calculate_dir_size(&root_b).unwrap_or((0, 0));

    // 9. Speichere in DB
    let db = state.db.lock().map_err(|e| format!("DB-Fehler: {}", e))?;
    let now = Utc::now().to_rfc3339();

    let name_a = format!("{} (Hälfte 1/2)", ds_name);
    let name_b = format!("{} (Hälfte 2/2)", ds_name);

    for (id, name, root, size, _files) in [
        (&id_a, &name_a, &root_a, size_a, files_a),
        (&id_b, &name_b, &root_b, size_b, files_b),
    ] {
        let db_ds_new = crate::database::Dataset {
            id: id.clone(),
            name: name.clone(),
            file_path: root.to_string_lossy().to_string(),
            file_type: "directory".to_string(),
            size_bytes: Some(size as i64),
            rows_count: None,
            columns_count: None,
            validated: false,
            created_at: now.clone(),
            training_count: 0,
            last_used_at: None,
        };
        db.save_dataset(&db_ds_new)
            .map_err(|e| format!("DB-Speicherfehler: {}", e))?;
    }

    // 10. Baue DatasetInfo für beide
    let make_info = |id: &str, name: &str, root: &PathBuf, size: u64, file_count: usize| DatasetInfo {
        id: id.to_string(),
        name: name.to_string(),
        model_id: model_id.clone(),
        source: "local".to_string(),
        source_path: Some(root.to_string_lossy().to_string()),
        size_bytes: size,
        file_count,
        created_at: Utc::now(),
        status: DatasetStatus::Split,
        split_info: Some(SplitInfo {
            train_count: if id == &id_a { total_a } else { total_b },
            val_count: 0,
            test_count: 0,
            train_ratio: 1.0,
            val_ratio: 0.0,
            test_ratio: 0.0,
        }),
        training_count: 0,
        last_used_at: None,
    };

    Ok(SplitHalfResult {
        dataset_a: make_info(&id_a, &name_a, &root_a, size_a, files_a),
        dataset_b: make_info(&id_b, &name_b, &root_b, size_b, files_b),
    })
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
    target_split: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<(), String> {
    // Primär: DB (wie get_dataset_files und add_files_to_dataset)
    let dataset_dir: PathBuf = {
        let db = state.db.lock()
            .map_err(|e| format!("DB-Lock-Fehler: {}", e))?;
        let all_db = db.list_datasets()
            .map_err(|e| format!("DB-Lesefehler: {}", e))?;

        if let Some(db_ds) = all_db.iter().find(|d| d.id == dataset_id) {
            PathBuf::from(&db_ds.file_path)
        } else {
            // Fallback: alte JSON-Metadata (Rückwärtskompatibilität)
            let metadata_path = get_datasets_metadata_path(&app_handle)?;
            let json_datasets: Vec<DatasetInfo> = if metadata_path.exists() {
                let content = fs::read_to_string(&metadata_path)
                    .map_err(|e| format!("Konnte Metadata nicht lesen: {}", e))?;
                serde_json::from_str(&content).unwrap_or_default()
            } else {
                Vec::new()
            };
            if let Some(ds) = json_datasets.iter().find(|d| d.id == dataset_id) {
                get_model_datasets_dir(&app_handle, &ds.model_id)?.join(&ds.id)
            } else {
                return Err(format!(
                    "Dataset '{}' nicht gefunden (weder in DB noch in Metadata-Datei)",
                    dataset_id
                ));
            }
        }
    };

    if !dataset_dir.exists() {
        return Err(format!(
            "Dataset-Verzeichnis existiert nicht: {}",
            dataset_dir.display()
        ));
    }

    // Zielordner anlegen (z.B. val/, train/, test/)
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
    file_paths: Vec<String>,
    state: tauri::State<'_, crate::AppState>,
) -> Result<usize, String> {
    // 1. Primär: Pfad aus der Datenbank lesen (wie get_dataset_files)
    let dataset_dir: PathBuf = {
        let db = state.db.lock()
            .map_err(|e| format!("DB-Lock-Fehler: {}", e))?;
        let all_db = db.list_datasets()
            .map_err(|e| format!("DB-Lesefehler: {}", e))?;

        if let Some(db_ds) = all_db.iter().find(|d| d.id == dataset_id) {
            PathBuf::from(&db_ds.file_path)
        } else {
            // 2. Fallback: alte JSON-Metadata (Rückwärtskompatibilität)
            let metadata_path = get_datasets_metadata_path(&app_handle)?;
            let json_datasets: Vec<DatasetInfo> = if metadata_path.exists() {
                let content = fs::read_to_string(&metadata_path)
                    .map_err(|e| format!("Konnte Metadata nicht lesen: {}", e))?;
                serde_json::from_str(&content).unwrap_or_default()
            } else {
                Vec::new()
            };

            if let Some(ds) = json_datasets.iter().find(|d| d.id == dataset_id) {
                get_model_datasets_dir(&app_handle, &ds.model_id)?.join(&ds.id)
            } else {
                return Err(format!(
                    "Dataset '{}' nicht gefunden (weder in DB noch in Metadata-Datei)",
                    dataset_id
                ));
            }
        }
    };

    if !dataset_dir.exists() {
        return Err(format!(
            "Dataset-Verzeichnis existiert nicht: {}",
            dataset_dir.display()
        ));
    }

    let mut copied_count = 0;

    for file_path_str in file_paths {
        let source_file = Path::new(&file_path_str);

        if !source_file.exists() {
            println!("[Dataset] Datei übersprungen (nicht gefunden): {}", file_path_str);
            continue;
        }

        let file_name = source_file.file_name()
            .ok_or_else(|| format!("Ungültiger Dateiname: {}", file_path_str))?;

        // Neue Dateien kommen in unused/ (split_dataset erkennt sie dort)
        let unused_dir = dataset_dir.join("unused");
        fs::create_dir_all(&unused_dir)
            .map_err(|e| format!("Konnte unused-Verzeichnis nicht erstellen: {}", e))?;

        let target_path = unused_dir.join(file_name);

        fs::copy(source_file, &target_path)
            .map_err(|e| format!(
                "Konnte '{}' nicht kopieren: {}",
                source_file.display(), e
            ))?;

        println!("[Dataset] ✅ Datei hinzugefügt: {:?}", file_name);
        copied_count += 1;
    }

    Ok(copied_count)
}
