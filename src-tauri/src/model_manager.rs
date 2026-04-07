use std::fs;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use tauri::Manager;
use chrono::{DateTime, Utc};

/// Metadata für ein gespeichertes Modell
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub source: String, // "local" oder "huggingface"
    pub source_path: Option<String>, // Original-Pfad oder HF-Repo
    pub size_bytes: u64,
    pub file_count: usize,
    pub created_at: DateTime<Utc>,
    pub model_type: Option<String>, // z.B. "transformer", "diffusion", etc.
}

/// Ergebnis eines Kopiervorgangs
#[derive(Debug, Serialize, Deserialize)]
pub struct CopyProgress {
    pub current_file: String,
    pub files_copied: usize,
    pub total_files: usize,
    pub bytes_copied: u64,
    pub total_bytes: u64,
}

/// Holt den Pfad zum Models-Verzeichnis innerhalb der App
fn get_models_dir(app_handle: &tauri::AppHandle) -> Result<PathBuf, String> {
    let data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Konnte App-Daten-Verzeichnis nicht finden: {}", e))?;
    
    let models_dir = data_dir.join("models");
    
    // Erstelle Verzeichnis falls nicht vorhanden
    if !models_dir.exists() {
        fs::create_dir_all(&models_dir)
            .map_err(|e| format!("Konnte Models-Verzeichnis nicht erstellen: {}", e))?;
    }
    
    Ok(models_dir)
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

/// Erkennt den Modelltyp anhand der Dateien im Verzeichnis
fn detect_model_type(path: &Path) -> Option<String> {
    let entries: Vec<_> = fs::read_dir(path).ok()?.filter_map(|e| e.ok()).collect();
    
    for entry in &entries {
        let name = entry.file_name().to_string_lossy().to_lowercase();
        
        // PyTorch / Transformers
        if name.contains("config.json") || name.contains("pytorch_model") || name.contains("model.safetensors") {
            return Some("transformer".to_string());
        }
        
        // Diffusion Models
        if name.contains("unet") || name.contains("vae") || name.contains("text_encoder") {
            return Some("diffusion".to_string());
        }
        
        // GGUF/GGML (llama.cpp)
        if name.ends_with(".gguf") || name.ends_with(".ggml") {
            return Some("gguf".to_string());
        }
        
        // ONNX
        if name.ends_with(".onnx") {
            return Some("onnx".to_string());
        }
        
        // TensorFlow
        if name.contains("saved_model") || name.ends_with(".pb") {
            return Some("tensorflow".to_string());
        }
    }
    
    None
}

/// Speichert Model-Metadata in einer JSON-Datei
fn save_model_metadata(models_dir: &Path, model_info: &ModelInfo) -> Result<(), String> {
    let metadata_path = models_dir.join("models_metadata.json");
    
    // Bestehende Metadata laden oder neue erstellen
    let mut models: Vec<ModelInfo> = if metadata_path.exists() {
        let content = fs::read_to_string(&metadata_path)
            .map_err(|e| format!("Konnte Metadata nicht lesen: {}", e))?;
        serde_json::from_str(&content).unwrap_or_default()
    } else {
        Vec::new()
    };
    
    // Prüfe ob Modell bereits existiert (nach ID)
    models.retain(|m| m.id != model_info.id);
    
    // Neues Modell hinzufügen
    models.push(model_info.clone());
    
    // Speichern
    let content = serde_json::to_string_pretty(&models)
        .map_err(|e| format!("Konnte Metadata nicht serialisieren: {}", e))?;
    
    fs::write(&metadata_path, content)
        .map_err(|e| format!("Konnte Metadata nicht speichern: {}", e))?;
    
    Ok(())
}

// ============ TAURI COMMANDS ============

/// Gibt den Pfad zum Models-Verzeichnis zurück
#[tauri::command]
pub fn get_models_directory(app_handle: tauri::AppHandle) -> Result<String, String> {
    let models_dir = get_models_dir(&app_handle)?;
    Ok(models_dir.to_string_lossy().to_string())
}

/// Listet alle installierten Modelle
#[tauri::command]
pub fn list_models(
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, crate::AppState>,
) -> Result<Vec<ModelInfo>, String> {
    // CRITICAL: Use database.rs list_models() which filters by user_id!
    let db = state.db.lock()
        .map_err(|e| format!("Database lock error: {}", e))?;
    
    let db_models = db.list_models()
        .map_err(|e| format!("Konnte Modelle nicht laden: {}", e))?;
    
    let models_dir = get_models_dir(&app_handle)?;
    
    // Convert DB models to ModelInfo
    let model_infos: Vec<ModelInfo> = db_models
        .into_iter()
        .filter(|m| {
            // Only return models that still exist in filesystem
            if let Some(ref path) = m.model_path {
                Path::new(path).exists()
            } else {
                models_dir.join(&m.id).exists()
            }
        })
        .map(|m| {
            let default_path = models_dir.join(&m.id);
            let model_path = m.model_path.as_ref()
                .map(|p| Path::new(p))
                .unwrap_or(&default_path);
            
            let (size_bytes, file_count) = calculate_dir_size(model_path)
                .unwrap_or((0, 0));
            
            let model_type = detect_model_type(model_path);
            
            ModelInfo {
                id: m.id,
                name: m.name,
                source: m.base_model.unwrap_or_else(|| "local".to_string()),
                source_path: m.model_path.clone(),
                size_bytes,
                file_count,
                created_at: Utc::now(), // DB doesn't parse timestamp, use current
                model_type,
            }
        })
        .collect();
    
    Ok(model_infos)
}

/// Kopiert ein lokales Modell-Verzeichnis in die App
#[tauri::command]
pub async fn import_local_model(
    app_handle: tauri::AppHandle,
    source_path: String,
    model_name: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<ModelInfo, String> {
    let source = Path::new(&source_path);
    
    // Validierung
    if !source.exists() {
        return Err("Quellverzeichnis existiert nicht".to_string());
    }
    
    if !source.is_dir() {
        return Err("Quelle muss ein Verzeichnis sein".to_string());
    }
    
    // Generiere eindeutige ID
    let model_id = format!("local_{}", uuid::Uuid::new_v4().to_string().replace("-", "")[..12].to_string());
    
    // Zielverzeichnis
    let models_dir = get_models_dir(&app_handle)?;
    let target_dir = models_dir.join(&model_id);
    
    println!("[Model] Importing local model: {} to {}", source_path, target_dir.display());
    
    // Berechne Größe vorher
    let (size_bytes, file_count) = calculate_dir_size(source)?;
    
    // Kopiere das Verzeichnis
    copy_dir_recursive(source, &target_dir)?;
    
    // Erkenne Modelltyp
    let model_type = detect_model_type(&target_dir);
    
    // Erstelle ModelInfo
    let model_info = ModelInfo {
        id: model_id.clone(),
        name: model_name.clone(),
        source: "local".to_string(),
        source_path: Some(source_path.clone()),
        size_bytes,
        file_count,
        created_at: Utc::now(),
        model_type: model_type.clone(),
    };
    
    // Speichere Metadata JSON (deprecated but kept for backward compatibility)
    save_model_metadata(&models_dir, &model_info)?;
    
    // CRITICAL: Use database.rs create_model() for user isolation!
    println!("[Model] Saving to database with user isolation: {}", model_id);
    
    let db = state.db.lock()
        .map_err(|e| format!("Failed to lock database: {}", e))?;
    
    let model_path_str = target_dir.to_string_lossy().to_string();
    let now = Utc::now().to_rfc3339();
    
    let db_model = crate::database::Model {
        id: model_id.clone(),
        name: model_name.clone(),
        description: Some(format!("Imported from local: {}", source_path)),
        base_model: Some("local".to_string()),
        model_path: Some(model_path_str.clone()),  // CRITICAL: Clone here so we can use it again later
        status: "ready".to_string(),
        created_at: now.clone(),
        updated_at: now,
    };
    
    db.create_model(&db_model)
        .map_err(|e| format!("Failed to insert into database: {}", e))?;
    
    println!("[Model] ✅ Model saved to database with user_id: {}", model_id);
    
    // CRITICAL: Create root version immediately after model import!
    println!("[Model] Creating root version for model: {}", model_id);
    match db.create_root_version(&model_id, &model_path_str) {
        Ok(version_id) => {
            println!("[Model] ✅ Root version created: {}", version_id);
        }
        Err(e) => {
            eprintln!("[Model] ⚠️  Failed to create root version: {}", e);
            // Don't fail the entire import, just log the error
        }
    }
    
    Ok(model_info)
}

/// Löscht ein Modell KOMPLETT (inkl. aller DB-Einträge, Versionen, Datasets, Training-Logs)
#[tauri::command]
pub fn delete_model(
    app_handle: tauri::AppHandle, 
    model_id: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<(), String> {
    println!("[DELETE] Starting complete deletion of model: {}", model_id);
    
    let models_dir = get_models_dir(&app_handle)?;
    let model_path = models_dir.join(&model_id);
    
    // 1. Lösche Modell-Verzeichnis (inkl. aller Unterordner)
    if model_path.exists() {
        println!("[DELETE] Deleting model directory: {:?}", model_path);
        fs::remove_dir_all(&model_path)
            .map_err(|e| format!("Konnte Modell-Verzeichnis nicht löschen: {}", e))?;
        println!("[DELETE] ✅ Model directory deleted");
    }
    
    // 2. Lösche alle Datasets für dieses Modell
    let datasets_path = models_dir.join(&model_id).join("datasets");
    if datasets_path.exists() {
        println!("[DELETE] Deleting datasets directory: {:?}", datasets_path);
        fs::remove_dir_all(&datasets_path)
            .map_err(|e| format!("Konnte Datasets nicht löschen: {}", e))?;
        println!("[DELETE] ✅ Datasets deleted");
    }
    
    // 3. Lösche alle Versionen für dieses Modell
    let versions_path = models_dir.join(&model_id).join("versions");
    if versions_path.exists() {
        println!("[DELETE] Deleting versions directory: {:?}", versions_path);
        fs::remove_dir_all(&versions_path)
            .map_err(|e| format!("Konnte Versionen nicht löschen: {}", e))?;
        println!("[DELETE] ✅ Versions deleted");
    }
    
    // 4. Lösche alle Training-Outputs für dieses Modell
    let data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Could not get app data dir: {}", e))?;
    
    let training_outputs_dir = data_dir.join("training_outputs");
    if training_outputs_dir.exists() {
        println!("[DELETE] Scanning training outputs for model {}", model_id);
        // Lösche alle Training-Outputs die zu diesem Modell gehören
        if let Ok(entries) = fs::read_dir(&training_outputs_dir) {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                // Prüfe config.json ob es zu diesem Modell gehört
                let config_path = path.join("config.json");
                if config_path.exists() {
                    if let Ok(config_str) = fs::read_to_string(&config_path) {
                        if config_str.contains(&model_id) {
                            println!("[DELETE] Deleting training output: {:?}", path);
                            fs::remove_dir_all(&path).ok();
                        }
                    }
                }
            }
        }
        println!("[DELETE] ✅ Training outputs cleaned");
    }
    
    // 5. Lösche alle Datenbank-Einträge
    println!("[DELETE] Cleaning database entries for model {}", model_id);
    
    let db = state.db.lock()
        .map_err(|e| format!("Failed to lock database: {}", e))?;
        
    // Lösche aus allen Tabellen
    
    // a) Lösche training_metrics_new für alle Versionen dieses Modells
    db.conn.execute(
        "DELETE FROM training_metrics_new WHERE version_id IN 
         (SELECT id FROM model_versions_new WHERE model_id = ?1)",
        [&model_id],
    ).ok();
    println!("[DELETE] ✅ Training metrics deleted");
    
    // b) Lösche model_versions_new
    db.conn.execute(
        "DELETE FROM model_versions_new WHERE model_id = ?1",
        [&model_id],
    ).ok();
    println!("[DELETE] ✅ Model versions deleted");
    
    // c) Lösche alte model_versions (falls vorhanden)
    db.conn.execute(
        "DELETE FROM model_versions WHERE model_id = ?1",
        [&model_id],
    ).ok();
    
    // d) Lösche training_metrics (alte Tabelle)
    db.conn.execute(
        "DELETE FROM training_metrics WHERE version_id IN 
         (SELECT id FROM model_versions WHERE model_id = ?1)",
        [&model_id],
    ).ok();
    
    // e) Lösche training_configs
    db.conn.execute(
        "DELETE FROM training_configs WHERE version_id IN 
         (SELECT id FROM model_versions WHERE model_id = ?1)",
        [&model_id],
    ).ok();
    db.conn.execute(
        "DELETE FROM training_configs WHERE version_id IN 
         (SELECT id FROM model_versions_new WHERE model_id = ?1)",
        [&model_id],
    ).ok();
    println!("[DELETE] ✅ Training configs deleted");
    
    // f) Lösche training_sessions
    db.conn.execute(
        "DELETE FROM training_sessions WHERE version_id IN 
         (SELECT id FROM model_versions WHERE model_id = ?1)",
        [&model_id],
    ).ok();
    db.conn.execute(
        "DELETE FROM training_sessions WHERE version_id IN 
         (SELECT id FROM model_versions_new WHERE model_id = ?1)",
        [&model_id],
    ).ok();
    println!("[DELETE] ✅ Training sessions deleted");
    
    // g) Lösche datasets (die zu diesem Modell gehören)
    // Hinweis: Datasets haben keine direkte model_id, aber wir können sie über den file_path filtern
    if let Ok(mut stmt) = db.conn.prepare("SELECT id, file_path FROM datasets") {
        if let Ok(rows) = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        }) {
            for row in rows.filter_map(|r| r.ok()) {
                let (dataset_id, file_path) = row;
                if file_path.contains(&model_id) {
                    db.conn.execute("DELETE FROM datasets WHERE id = ?1", [&dataset_id]).ok();
                }
            }
        }
    }
    println!("[DELETE] ✅ Datasets deleted");
    
    // h) Lösche das Model selbst aus der models Tabelle
    db.conn.execute(
        "DELETE FROM models WHERE id = ?1",
        [&model_id],
    ).ok();
    println!("[DELETE] ✅ Model entry deleted from database");
    
    // 6. Aktualisiere Metadata JSON
    let metadata_path = models_dir.join("models_metadata.json");
    if metadata_path.exists() {
        println!("[DELETE] Updating models_metadata.json");
        let content = fs::read_to_string(&metadata_path)
            .map_err(|e| format!("Konnte Metadata nicht lesen: {}", e))?;
        
        let mut models: Vec<ModelInfo> = serde_json::from_str(&content).unwrap_or_default();
        models.retain(|m| m.id != model_id);
        
        let content = serde_json::to_string_pretty(&models)
            .map_err(|e| format!("Konnte Metadata nicht serialisieren: {}", e))?;
        
        fs::write(&metadata_path, content)
            .map_err(|e| format!("Konnte Metadata nicht speichern: {}", e))?;
        println!("[DELETE] ✅ Metadata updated");
    }
    
    println!("[DELETE] ✅ COMPLETE DELETION FINISHED for model {}", model_id);
    Ok(())
}

/// Holt Infos zu einem spezifischen Modell
#[tauri::command]
pub fn get_model_info(
    app_handle: tauri::AppHandle,
    model_id: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<ModelInfo, String> {
    let models = list_models(app_handle, state)?;
    
    models
        .into_iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| "Modell nicht gefunden".to_string())
}

/// Validiert ob ein Verzeichnis ein gültiges Modell enthält
#[tauri::command]
pub fn validate_model_directory(path: String) -> Result<bool, String> {
    let dir = Path::new(&path);
    
    if !dir.exists() || !dir.is_dir() {
        return Ok(false);
    }
    
    // Prüfe auf typische Modell-Dateien
    let entries: Vec<_> = fs::read_dir(dir)
        .map_err(|e| format!("Konnte Verzeichnis nicht lesen: {}", e))?
        .filter_map(|e| e.ok())
        .collect();
    
    // Mindestens eine erkennbare Modell-Datei
    let valid_extensions = [
        "safetensors", "bin", "pt", "pth", "onnx", "gguf", "ggml", "pb", "h5", "keras"
    ];
    
    let has_model_file = entries.iter().any(|entry| {
        let name = entry.file_name().to_string_lossy().to_lowercase();
        valid_extensions.iter().any(|ext| name.ends_with(ext)) ||
        name == "config.json" ||
        name == "model_index.json"
    });
    
    Ok(has_model_file)
}

/// Berechnet die Größe eines Verzeichnisses (für UI-Vorschau)
#[tauri::command]
pub fn get_directory_size(path: String) -> Result<(u64, usize), String> {
    let dir = Path::new(&path);
    calculate_dir_size(dir)
}

// ============ HUGGING FACE INTEGRATION ============

#[derive(Debug, Serialize, Deserialize)]
pub struct HuggingFaceModel {
    pub id: String,
    #[serde(rename = "modelId")]
    pub model_id: Option<String>,
    pub author: Option<String>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    pub tags: Option<Vec<String>>,
    pub pipeline_tag: Option<String>,
    #[serde(rename = "lastModified")]
    pub last_modified: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HuggingFaceFile {
    #[serde(alias = "rfilename", alias = "path")]
    pub filename: String,
    pub size: Option<u64>,
    #[serde(rename = "type")]
    pub file_type: Option<String>,
}

/// Sucht nach Modellen auf Hugging Face
#[tauri::command]
pub async fn search_huggingface_models(query: String, limit: Option<u32>) -> Result<Vec<HuggingFaceModel>, String> {
    let limit = limit.unwrap_or(20);
    let url = format!(
        "https://huggingface.co/api/models?search={}&limit={}&sort=downloads&direction=-1",
        urlencoding::encode(&query),
        limit
    );
    
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
    
    let models: Vec<HuggingFaceModel> = response
        .json()
        .await
        .map_err(|e| format!("JSON Parse Fehler: {}", e))?;
    
    Ok(models)
}

/// Holt die Dateiliste eines HuggingFace Repos
#[tauri::command]
pub async fn get_huggingface_model_files(repo_id: String) -> Result<Vec<HuggingFaceFile>, String> {
    let url = format!("https://huggingface.co/api/models/{}/tree/main", repo_id);
    
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
    
    let files: Vec<HuggingFaceFile> = response
        .json()
        .await
        .map_err(|e| format!("JSON Parse Fehler: {}", e))?;
    
    Ok(files)
}

/// Lädt ein Modell von HuggingFace herunter
#[tauri::command]
pub async fn download_huggingface_model(
    app_handle: tauri::AppHandle,
    repo_id: String,
    model_name: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<ModelInfo, String> {
    let models_dir = get_models_dir(&app_handle)?;
    
    // Generiere eindeutige ID
    let model_id = format!("hf_{}", uuid::Uuid::new_v4().to_string().replace("-", "")[..12].to_string());
    let target_dir = models_dir.join(&model_id);
    
    println!("[Model] Downloading HuggingFace model: {} to {}", repo_id, target_dir.display());
    
    fs::create_dir_all(&target_dir)
        .map_err(|e| format!("Konnte Zielverzeichnis nicht erstellen: {}", e))?;
    
    // Hole Dateiliste
    let files = get_huggingface_model_files(repo_id.clone()).await?;
    
    let client = reqwest::Client::new();
    let mut total_size: u64 = 0;
    let mut file_count: usize = 0;
    
    // Filtere nur wichtige Dateien (keine riesigen Binaries für den Anfang)
    let important_files: Vec<&HuggingFaceFile> = files.iter()
        .filter(|f| {
            // Nur Dateien, keine Verzeichnisse
            if let Some(ref ft) = f.file_type {
                if ft == "directory" {
                    return false;
                }
            }
            let name = f.filename.to_lowercase();
            name.ends_with(".json") ||
            name.ends_with(".txt") ||
            name.ends_with(".md") ||
            name.ends_with(".safetensors") ||
            name.ends_with(".bin") ||
            name.ends_with(".model") ||
            name == "tokenizer.model" ||
            name.contains("config")
        })
        .collect();
    
    for file in important_files {
        // URL-encode den Dateinamen für Sonderzeichen
        let encoded_filename = urlencoding::encode(&file.filename);
        let file_url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
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
            
            // Erstelle Unterverzeichnisse falls nötig
            let file_path = target_dir.join(&file.filename);
            if let Some(parent) = file_path.parent() {
                fs::create_dir_all(parent).ok();
            }
            
            fs::write(&file_path, &bytes)
                .map_err(|e| format!("Konnte Datei nicht speichern: {}", e))?;
            
            total_size += bytes.len() as u64;
            file_count += 1;
        }
    }
    
    // Erkenne Modelltyp
    let model_type = detect_model_type(&target_dir);
    
    // Erstelle ModelInfo
    let model_info = ModelInfo {
        id: model_id.clone(),
        name: model_name.clone(),
        source: "huggingface".to_string(),
        source_path: Some(repo_id.clone()),
        size_bytes: total_size,
        file_count,
        created_at: Utc::now(),
        model_type: model_type.clone(),
    };
    
    // Speichere Metadata JSON (deprecated but kept for backward compatibility)
    save_model_metadata(&models_dir, &model_info)?;
    
    // CRITICAL: Use database.rs create_model() for user isolation!
    println!("[Model] Saving to database with user isolation: {}", model_id);
    
    let db = state.db.lock()
        .map_err(|e| format!("Failed to lock database: {}", e))?;
    
    let model_path_str = target_dir.to_string_lossy().to_string();
    let now = Utc::now().to_rfc3339();
    
    let db_model = crate::database::Model {
        id: model_id.clone(),
        name: model_name.clone(),
        description: Some(format!("Downloaded from HuggingFace: {}", repo_id)),
        base_model: Some("huggingface".to_string()),
        model_path: Some(model_path_str.clone()),  // CRITICAL: Clone here so we can use it again later
        status: "ready".to_string(),
        created_at: now.clone(),
        updated_at: now,
    };
    
    db.create_model(&db_model)
        .map_err(|e| format!("Failed to insert into database: {}", e))?;
    
    println!("[Model] ✅ Model saved to database with user_id: {}", model_id);
    
    // CRITICAL: Create root version immediately after model download!
    println!("[Model] Creating root version for model: {}", model_id);
    match db.create_root_version(&model_id, &model_path_str) {
        Ok(version_id) => {
            println!("[Model] ✅ Root version created: {}", version_id);
        }
        Err(e) => {
            eprintln!("[Model] ⚠️  Failed to create root version: {}", e);
            // Don't fail the entire download, just log the error
        }
    }
    
    Ok(model_info)
}
