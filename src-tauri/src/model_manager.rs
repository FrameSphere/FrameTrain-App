use std::fs;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use tauri::Manager;
use chrono::{DateTime, Utc};

// ============ Typen ============

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub source: String,
    pub source_path: Option<String>,  // HF Repo-ID oder originaler lokaler Pfad
    pub local_path: String,           // NEU: tatsächlicher Pfad auf der Festplatte
    pub size_bytes: u64,
    pub file_count: usize,
    pub created_at: DateTime<Utc>,
    pub model_type: Option<String>,
}

// ============ Interne Helfer ============

fn get_models_dir(app_handle: &tauri::AppHandle) -> Result<PathBuf, String> {
    let data_dir = app_handle.path().app_data_dir()
        .map_err(|e| format!("AppDataDir: {}", e))?;
    let dir = data_dir.join("models");
    if !dir.exists() {
        fs::create_dir_all(&dir).map_err(|e| format!("Models-Dir erstellen: {}", e))?;
    }
    Ok(dir)
}

fn calculate_dir_size(path: &Path) -> Result<(u64, usize), String> {
    let mut size: u64 = 0;
    let mut count: usize = 0;
    if path.is_file() {
        return Ok((fs::metadata(path).map_err(|e| e.to_string())?.len(), 1));
    }
    for entry in fs::read_dir(path).map_err(|e| format!("readdir: {}", e))? {
        let p = entry.map_err(|e| format!("entry: {}", e))?.path();
        if p.is_file() {
            size += fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            count += 1;
        } else if p.is_dir() {
            let (s, c) = calculate_dir_size(&p)?;
            size += s; count += c;
        }
    }
    Ok((size, count))
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<(), String> {
    if !dst.exists() {
        fs::create_dir_all(dst).map_err(|e| format!("mkdir: {}", e))?;
    }
    for entry in fs::read_dir(src).map_err(|e| format!("readdir: {}", e))? {
        let entry = entry.map_err(|e| format!("entry: {}", e))?;
        let sp = entry.path();
        let dp = dst.join(entry.file_name());
        if sp.is_dir() { copy_dir_recursive(&sp, &dp)?; }
        else { fs::copy(&sp, &dp).map_err(|e| format!("copy {}: {}", sp.display(), e))?; }
    }
    Ok(())
}

fn detect_model_type(path: &Path) -> Option<String> {
    // 1. Aus config.json lesen (exakt, wie HuggingFace es speichert)
    let cfg_path = path.join("config.json");
    if cfg_path.exists() {
        if let Ok(content) = fs::read_to_string(&cfg_path) {
            if let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&content) {
                // model_type direkt zurückgeben (z.B. "xlm-roberta", "bert", "deberta")
                if let Some(mt) = cfg.get("model_type").and_then(|v| v.as_str()) {
                    return Some(mt.to_string());
                }
            }
        }
    }
    // 2. Fallback: Datei-Erweiterungen
    let entries: Vec<_> = fs::read_dir(path).ok()?.filter_map(|e| e.ok()).collect();
    for entry in &entries {
        let name = entry.file_name().to_string_lossy().to_lowercase();
        if name.contains("unet") || name.contains("vae") { return Some("diffusion".to_string()); }
        if name.ends_with(".gguf") || name.ends_with(".ggml") { return Some("gguf".to_string()); }
        if name.ends_with(".onnx") { return Some("onnx".to_string()); }
        if name.ends_with(".pt") || name.ends_with(".pth") { return Some("pytorch".to_string()); }
        if name.contains("pytorch_model") || name.contains(".safetensors") { return Some("transformer".to_string()); }
    }
    None
}

fn save_metadata(models_dir: &Path, info: &ModelInfo) -> Result<(), String> {
    let path = models_dir.join("models_metadata.json");
    let mut models: Vec<ModelInfo> = if path.exists() {
        serde_json::from_str(&fs::read_to_string(&path).unwrap_or_default()).unwrap_or_default()
    } else { vec![] };
    models.retain(|m| m.id != info.id);
    models.push(info.clone());
    fs::write(&path, serde_json::to_string_pretty(&models).map_err(|e| format!("JSON: {}", e))?)
        .map_err(|e| format!("Metadata schreiben: {}", e))
}

// ============ Tauri Commands ============

#[tauri::command]
pub fn get_models_directory(app_handle: tauri::AppHandle) -> Result<String, String> {
    Ok(get_models_dir(&app_handle)?.to_string_lossy().to_string())
}

#[tauri::command]
pub fn list_models(
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, crate::AppState>,
) -> Result<Vec<ModelInfo>, String> {
    let db = state.db.lock().map_err(|e| format!("DB Lock: {}", e))?;
    let db_models = db.list_models().map_err(|e| format!("DB list_models: {}", e))?;
    let models_dir = get_models_dir(&app_handle)?;

    let infos = db_models.into_iter()
        .filter(|m| {
            if let Some(ref p) = m.model_path { Path::new(p).exists() }
            else { models_dir.join(&m.id).exists() }
        })
        .map(|m| {
            let default = models_dir.join(&m.id);
            let model_path = m.model_path.as_ref().map(|p| Path::new(p).to_path_buf()).unwrap_or(default);
            let (size, files) = calculate_dir_size(&model_path).unwrap_or((0, 0));
            let mtype = detect_model_type(&model_path);
            let source = m.base_model.clone().unwrap_or_else(|| "local".to_string());
            // HF Repo-ID aus description extrahieren ("HuggingFace: {repo_id}")
            let source_path = if source == "huggingface" {
                m.description.as_ref()
                    .and_then(|d| d.strip_prefix("HuggingFace: "))
                    .map(|s| s.to_string())
                    .or_else(|| m.model_path.clone())
            } else {
                m.model_path.clone()
            };
            ModelInfo {
                id: m.id,
                name: m.name,
                source,
                source_path,
                local_path: model_path.to_string_lossy().to_string(),
                size_bytes: size,
                file_count: files,
                created_at: Utc::now(),
                model_type: mtype,
            }
        })
        .collect();

    Ok(infos)
}

/// Parameternamen müssen 1:1 dem Frontend-invoke entsprechen (camelCase → snake_case)
#[tauri::command]
pub async fn import_local_model(
    app_handle: tauri::AppHandle,
    source_path: String,   // Frontend: sourcePath
    model_name: String,    // Frontend: modelName
    state: tauri::State<'_, crate::AppState>,
) -> Result<ModelInfo, String> {
    let src = Path::new(&source_path);
    if !src.exists() { return Err("Quellpfad existiert nicht".to_string()); }
    if !src.is_dir()  { return Err("Quelle muss ein Verzeichnis sein".to_string()); }

    let model_id   = format!("local_{}", &uuid::Uuid::new_v4().to_string().replace("-","")[..12]);
    let models_dir = get_models_dir(&app_handle)?;
    let target     = models_dir.join(&model_id);

    let (size, files) = calculate_dir_size(src)?;
    copy_dir_recursive(src, &target)?;
    let mtype = detect_model_type(&target);

    let info = ModelInfo {
        id: model_id.clone(), name: model_name.clone(), source: "local".to_string(),
        source_path: Some(source_path.clone()), local_path: target.to_string_lossy().to_string(),
        size_bytes: size, file_count: files,
        created_at: Utc::now(), model_type: mtype,
    };
    save_metadata(&models_dir, &info)?;

    let db = state.db.lock().map_err(|e| format!("DB Lock: {}", e))?;
    let model_path_str = target.to_string_lossy().to_string();
    let now = Utc::now().to_rfc3339();
    let db_model = crate::database::Model {
        id: model_id.clone(), name: model_name.clone(),
        description: Some(format!("Importiert von: {}", source_path)),
        base_model: Some("local".to_string()),
        model_path: Some(model_path_str.clone()),
        status: "ready".to_string(), created_at: now.clone(), updated_at: now,
    };
    db.create_model(&db_model).map_err(|e| format!("DB create_model: {}", e))?;

    if let Err(e) = db.create_root_version(&model_id, &model_path_str) {
        eprintln!("[Model] Root-Version konnte nicht erstellt werden: {}", e);
    }

    Ok(info)
}

#[tauri::command]
pub fn delete_model(
    app_handle: tauri::AppHandle,
    model_id: String,      // Frontend: modelId
    state: tauri::State<'_, crate::AppState>,
) -> Result<(), String> {
    let models_dir = get_models_dir(&app_handle)?;
    let model_path = models_dir.join(&model_id);
    if model_path.exists() {
        fs::remove_dir_all(&model_path).map_err(|e| format!("Dir löschen: {}", e))?;
    }

    // Training-Outputs bereinigen
    if let Ok(data_dir) = app_handle.path().app_data_dir() {
        let outputs = data_dir.join("training_outputs");
        if outputs.exists() {
            if let Ok(entries) = fs::read_dir(&outputs) {
                for e in entries.filter_map(|e| e.ok()) {
                    let cfg = e.path().join("config.json");
                    if cfg.exists() {
                        if let Ok(s) = fs::read_to_string(&cfg) {
                            if s.contains(&model_id) { fs::remove_dir_all(e.path()).ok(); }
                        }
                    }
                }
            }
        }
    }

    let db = state.db.lock().map_err(|e| format!("DB Lock: {}", e))?;
    db.conn.execute("DELETE FROM training_metrics_new WHERE version_id IN (SELECT id FROM model_versions_new WHERE model_id = ?1)", [&model_id]).ok();
    db.conn.execute("DELETE FROM model_versions_new WHERE model_id = ?1", [&model_id]).ok();
    db.conn.execute("DELETE FROM models WHERE id = ?1", [&model_id]).ok();

    // Metadata JSON aktualisieren
    let meta = models_dir.join("models_metadata.json");
    if meta.exists() {
        let mut models: Vec<ModelInfo> = serde_json::from_str(
            &fs::read_to_string(&meta).unwrap_or_default()
        ).unwrap_or_default();
        models.retain(|m| m.id != model_id);
        fs::write(&meta, serde_json::to_string_pretty(&models).unwrap_or_default()).ok();
    }

    Ok(())
}

#[tauri::command]
pub fn get_model_info(
    app_handle: tauri::AppHandle,
    model_id: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<ModelInfo, String> {
    list_models(app_handle, state)?
        .into_iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| "Modell nicht gefunden".to_string())
}

#[tauri::command]
pub fn validate_model_directory(path: String) -> Result<bool, String> {
    let dir = Path::new(&path);
    if !dir.exists() || !dir.is_dir() { return Ok(false); }
    let valid_ext = ["safetensors","bin","pt","pth","onnx","gguf","ggml","pb","h5","keras"];
    let has_file = fs::read_dir(dir).map_err(|e| format!("readdir: {}", e))?.filter_map(|e| e.ok()).any(|entry| {
        let name = entry.file_name().to_string_lossy().to_lowercase();
        valid_ext.iter().any(|ext| name.ends_with(ext)) || name == "config.json" || name == "model_index.json"
    });
    Ok(has_file)
}

#[tauri::command]
pub fn get_directory_size(path: String) -> Result<(u64, usize), String> {
    calculate_dir_size(Path::new(&path))
}

// ============ HuggingFace ============

#[derive(Debug, Serialize, Deserialize)]
pub struct HuggingFaceModel {
    pub id: String,
    pub author: Option<String>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
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

/// Frontend ruft auf: invoke('search_huggingface_models', { query, limit })
#[tauri::command]
pub async fn search_huggingface_models(
    query: String,
    limit: Option<u32>,
) -> Result<Vec<HuggingFaceModel>, String> {
    let limit = limit.unwrap_or(20);
    let url = format!(
        "https://huggingface.co/api/models?search={}&limit={}&sort=downloads&direction=-1",
        urlencoding::encode(&query), limit
    );

    println!("[HF] Suche: {}", url);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|e| format!("HTTP Client: {}", e))?;

    let resp = client.get(&url)
        .header("User-Agent", "FrameTrain-Desktop/1.0")
        .send()
        .await
        .map_err(|e| format!("HTTP: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!("HuggingFace API: {}", resp.status()));
    }

    let models: Vec<HuggingFaceModel> = resp.json().await
        .map_err(|e| format!("JSON: {}", e))?;

    println!("[HF] {} Modelle gefunden", models.len());
    Ok(models)
}

#[tauri::command]
pub async fn get_huggingface_model_files(
    repo_id: String,
) -> Result<Vec<HuggingFaceFile>, String> {
    let url = format!("https://huggingface.co/api/models/{}/tree/main", repo_id);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|e| format!("HTTP Client: {}", e))?;
    let resp = client.get(&url)
        .header("User-Agent", "FrameTrain-Desktop/1.0")
        .send().await.map_err(|e| format!("HTTP: {}", e))?;
    if !resp.status().is_success() { return Err(format!("HF API: {}", resp.status())); }
    resp.json::<Vec<HuggingFaceFile>>().await.map_err(|e| format!("JSON: {}", e))
}

/// Frontend ruft auf: invoke('download_huggingface_model', { repoId, modelName })
#[tauri::command]
pub async fn download_huggingface_model(
    app_handle: tauri::AppHandle,
    repo_id: String,       // Frontend: repoId
    model_name: String,    // Frontend: modelName
    state: tauri::State<'_, crate::AppState>,
) -> Result<ModelInfo, String> {
    let models_dir = get_models_dir(&app_handle)?;
    let model_id   = format!("hf_{}", &uuid::Uuid::new_v4().to_string().replace("-","")[..12]);
    let target     = models_dir.join(&model_id);
    fs::create_dir_all(&target).map_err(|e| format!("mkdir: {}", e))?;

    println!("[HF Download] {} → {}", repo_id, target.display());

    let files = get_huggingface_model_files(repo_id.clone()).await?;
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .map_err(|e| format!("HTTP Client: {}", e))?;

    let mut total: u64 = 0;
    let mut count: usize = 0;

    // Nur relevante Dateien (keine Verzeichnisse, keine riesigen Shards)
    let relevant: Vec<&HuggingFaceFile> = files.iter().filter(|f| {
        if let Some(ref t) = f.file_type { if t == "directory" { return false; } }
        let n = f.filename.to_lowercase();
        // Für Klassifikationsmodelle: config, tokenizer, safetensors/bin
        n.ends_with(".json") || n.ends_with(".txt") || n.ends_with(".md")
            || n.ends_with(".safetensors") || n.ends_with(".bin")
            || n.ends_with(".model") || n.contains("config")
    }).collect();

    println!("[HF Download] {} Dateien herunterladen", relevant.len());

    for file in relevant {
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            repo_id, urlencoding::encode(&file.filename)
        );
        println!("[HF Download] ← {}", file.filename);

        let resp = client.get(&url)
            .header("User-Agent", "FrameTrain-Desktop/1.0")
            .send().await.map_err(|e| format!("GET {}: {}", file.filename, e))?;

        if resp.status().is_success() {
            let bytes = resp.bytes().await.map_err(|e| format!("Bytes {}: {}", file.filename, e))?;
            let dest = target.join(&file.filename);
            if let Some(p) = dest.parent() { fs::create_dir_all(p).ok(); }
            fs::write(&dest, &bytes).map_err(|e| format!("Write {}: {}", file.filename, e))?;
            total += bytes.len() as u64;
            count += 1;
        }
    }

    let mtype = detect_model_type(&target);
    let info = ModelInfo {
        id: model_id.clone(), name: model_name.clone(), source: "huggingface".to_string(),
        source_path: Some(repo_id.clone()), local_path: target.to_string_lossy().to_string(),
        size_bytes: total, file_count: count,
        created_at: Utc::now(), model_type: mtype,
    };
    save_metadata(&models_dir, &info)?;

    let db = state.db.lock().map_err(|e| format!("DB Lock: {}", e))?;
    let path_str = target.to_string_lossy().to_string();
    let now = Utc::now().to_rfc3339();
    db.create_model(&crate::database::Model {
        id: model_id.clone(), name: model_name.clone(),
        description: Some(format!("HuggingFace: {}", repo_id)),
        base_model: Some("huggingface".to_string()),
        model_path: Some(path_str.clone()),
        status: "ready".to_string(), created_at: now.clone(), updated_at: now,
    }).map_err(|e| format!("DB create_model: {}", e))?;

    if let Err(e) = db.create_root_version(&model_id, &path_str) {
        eprintln!("[HF Download] Root-Version: {}", e);
    }

    println!("[HF Download] ✅ {} Dateien, {} bytes", count, total);
    Ok(info)
}
