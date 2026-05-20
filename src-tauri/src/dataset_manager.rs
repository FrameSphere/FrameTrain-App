// dataset_manager.rs – vollständige Implementierung

use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use tauri::{Manager, Emitter};
use chrono::Utc;
use serde_json;
use futures_util::StreamExt;
use tokio::io::AsyncWriteExt;

// ============ Typen ============

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitInfo {
    pub train_count: usize,
    pub val_count:   usize,
    pub test_count:  usize,
    pub train_ratio: f64,
    pub val_ratio:   f64,
    pub test_ratio:  f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    pub id:             String,
    pub name:           String,
    pub model_id:       String,
    pub source:         String,
    pub source_path:    Option<String>,
    #[serde(default)]
    pub storage_path:   String,
    pub size_bytes:     u64,
    pub file_count:     usize,
    pub created_at:     String,
    pub status:         String,
    pub split_info:     Option<SplitInfo>,
    pub training_count: i64,
    pub last_used_at:   Option<String>,
    #[serde(default)]
    pub extensions:     Vec<String>,
}

/// Progress-Event für HuggingFace Dataset-Downloads (analog zu ModelDownloadProgress)
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DatasetDownloadProgress {
    pub status:               String,   // "connecting" | "downloading" | "complete" | "error"
    pub current_file:         String,
    pub current_file_index:   usize,
    pub total_files:          usize,
    pub downloaded_bytes:     u64,
    pub total_bytes:          u64,
    pub progress_percent:     i32,
    pub speed_mbs:            f32,
    pub elapsed_secs:         u64,
    pub eta_secs:             u64,
    pub message:              String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HuggingFaceDataset {
    pub id:        String,
    pub author:    Option<String>,
    pub downloads: Option<u64>,
    pub likes:     Option<u64>,
    pub tags:      Option<Vec<String>>,
}

// ============ Interne Helfer ============

fn get_datasets_dir(app_handle: &tauri::AppHandle) -> Result<PathBuf, String> {
    let dir = app_handle.path().app_data_dir()
        .map_err(|e| format!("AppDataDir: {}", e))?
        .join("datasets");
    fs::create_dir_all(&dir).ok();
    Ok(dir)
}

fn meta_path(datasets_dir: &Path) -> PathBuf {
    datasets_dir.join("datasets_metadata.json")
}

fn load_metadata(datasets_dir: &Path) -> Vec<DatasetInfo> {
    let path = meta_path(datasets_dir);
    if !path.exists() { return vec![]; }
    serde_json::from_str(&fs::read_to_string(&path).unwrap_or_default()).unwrap_or_default()
}

fn save_metadata(datasets_dir: &Path, datasets: &[DatasetInfo]) -> Result<(), String> {
    let path = meta_path(datasets_dir);
    fs::write(&path, serde_json::to_string_pretty(datasets)
        .map_err(|e| format!("JSON: {}", e))?)
        .map_err(|e| format!("Write: {}", e))
}

fn upsert_metadata(datasets_dir: &Path, info: &DatasetInfo) -> Result<(), String> {
    let mut all = load_metadata(datasets_dir);
    all.retain(|d| d.id != info.id);
    all.push(info.clone());
    save_metadata(datasets_dir, &all)
}

fn dir_size(path: &Path) -> (u64, usize) {
    if !path.exists() { return (0, 0); }
    let mut size = 0u64; let mut count = 0usize;
    fn walk(p: &Path, s: &mut u64, c: &mut usize) {
        if let Ok(entries) = fs::read_dir(p) {
            for e in entries.flatten() {
                let ep = e.path();
                if ep.is_file() { *s += fs::metadata(&ep).map(|m| m.len()).unwrap_or(0); *c += 1; }
                else if ep.is_dir() { walk(&ep, s, c); }
            }
        }
    }
    walk(path, &mut size, &mut count);
    (size, count)
}

fn copy_dir(src: &Path, dst: &Path) -> Result<(), String> {
    fs::create_dir_all(dst).ok();
    for e in fs::read_dir(src).map_err(|e| format!("readdir: {}", e))? {
        let e = e.map_err(|e| format!("entry: {}", e))?;
        let sp = e.path(); let dp = dst.join(e.file_name());
        if sp.is_dir() { copy_dir(&sp, &dp)?; }
        else { fs::copy(&sp, &dp).map_err(|e| format!("copy: {}", e))?; }
    }
    Ok(())
}

fn collect_extensions(dir: &Path) -> Vec<String> {
    let mut exts = std::collections::HashSet::new();
    fn walk(p: &Path, out: &mut std::collections::HashSet<String>) {
        if let Ok(entries) = fs::read_dir(p) {
            for e in entries.flatten() {
                let ep = e.path();
                if ep.is_file() {
                    if let Some(ext) = ep.extension().and_then(|s| s.to_str()) {
                        out.insert(format!(".{}", ext.to_lowercase()));
                    }
                } else if ep.is_dir() { walk(&ep, out); }
            }
        }
    }
    walk(dir, &mut exts);
    let mut v: Vec<String> = exts.into_iter().collect();
    v.sort();
    v
}

fn collect_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() { continue; }
            let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if matches!(file_name, "dataset_infos.json" | "metadata.json" | ".gitkeep" | ".DS_Store") {
                continue;
            }
            files.push(path);
        }
    }
    files
}

fn collect_files_recursive(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    fn walk(p: &Path, out: &mut Vec<PathBuf>) {
        if let Ok(entries) = fs::read_dir(p) {
            for e in entries.flatten() {
                let ep = e.path();
                if ep.is_file() { out.push(ep); }
                else if ep.is_dir() { walk(&ep, out); }
            }
        }
    }
    walk(dir, &mut files);
    files
}

fn make_info(
    id: &str, name: &str, model_id: &str, source: &str,
    source_path: Option<String>, target: &Path,
    size_bytes: u64, file_count: usize,
    status: &str, split_info: Option<SplitInfo>,
) -> DatasetInfo {
    let storage_path = target.to_string_lossy().to_string();
    let extensions   = collect_extensions(target);
    DatasetInfo {
        id: id.to_string(), name: name.to_string(),
        model_id: model_id.to_string(), source: source.to_string(),
        source_path, storage_path,
        size_bytes, file_count,
        created_at: Utc::now().to_rfc3339(),
        status: status.to_string(), split_info,
        training_count: 0, last_used_at: None, extensions,
    }
}

// ============ Tauri Commands ============

#[tauri::command]
pub async fn list_datasets_for_model(
    app_handle: tauri::AppHandle,
    model_id:   String,
) -> Result<Vec<DatasetInfo>, String> {
    let dir = get_datasets_dir(&app_handle)?;
    let all = load_metadata(&dir);

    let mut result: Vec<DatasetInfo> = all.into_iter()
        .filter(|d| d.model_id == model_id)
        .map(|mut d| {
            let storage = dir.join(&d.id);
            d.storage_path = storage.to_string_lossy().to_string();
            if storage.exists() {
                let exts = collect_extensions(&storage);
                if !exts.is_empty() { d.extensions = exts; }
            }
            d
        })
        .collect();

    result.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    let mut seen = std::collections::HashSet::new();
    result.retain(|d| seen.insert(d.name.clone()));
    result.sort_by(|a, b| a.created_at.cmp(&b.created_at));

    if let Ok(db_path) = app_handle.path().app_data_dir().map(|p| p.join("frametrain.db")) {
        if let Ok(conn) = rusqlite::Connection::open(&db_path) {
            for ds in &mut result {
                ds.training_count = conn.query_row(
                    "SELECT COALESCE(training_count,0) FROM datasets WHERE id=?1",
                    [&ds.id], |r| r.get(0),
                ).unwrap_or(0);
                ds.last_used_at = conn.query_row(
                    "SELECT last_used_at FROM datasets WHERE id=?1",
                    [&ds.id], |r| r.get(0),
                ).unwrap_or(None);
            }
        }
    }

    Ok(result)
}

#[tauri::command]
pub async fn list_test_datasets_for_model(
    app_handle: tauri::AppHandle,
    model_id:   String,
) -> Result<Vec<DatasetInfo>, String> {
    list_datasets_for_model(app_handle, model_id).await
}

#[tauri::command]
pub async fn list_all_datasets(app_handle: tauri::AppHandle) -> Result<Vec<DatasetInfo>, String> {
    Ok(load_metadata(&get_datasets_dir(&app_handle)?))
}

#[tauri::command]
pub async fn import_local_dataset(
    app_handle:   tauri::AppHandle,
    source_path:  String,
    dataset_name: String,
    model_id:     String,
) -> Result<DatasetInfo, String> {
    let src = Path::new(&source_path);
    if !src.exists() { return Err(format!("Pfad nicht gefunden: {}", source_path)); }

    let dataset_id   = format!("ds_{}", &uuid::Uuid::new_v4().to_string().replace("-","")[..12]);
    let datasets_dir = get_datasets_dir(&app_handle)?;
    let target       = datasets_dir.join(&dataset_id);

    if src.is_dir() { copy_dir(src, &target)?; }
    else {
        fs::create_dir_all(&target).ok();
        fs::copy(src, target.join(src.file_name().unwrap()))
            .map_err(|e| format!("Copy: {}", e))?;
    }

    let (size, files) = dir_size(&target);
    let info = make_info(&dataset_id, &dataset_name, &model_id, "local",
        Some(source_path), &target, size, files, "unused", None);
    upsert_metadata(&datasets_dir, &info)?;

    if let Ok(db_path) = app_handle.path().app_data_dir().map(|p| p.join("frametrain.db")) {
        if let Ok(conn) = rusqlite::Connection::open(&db_path) {
            let now = Utc::now().to_rfc3339();
            conn.execute(
                "INSERT OR IGNORE INTO datasets (id,name,file_path,file_type,size_bytes,validated,user_id,created_at) VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
                rusqlite::params![&dataset_id, &info.name, target.to_string_lossy().to_string(), "local", size as i64, 0, "default_user", &now],
            ).ok();
        }
    }

    Ok(info)
}

#[tauri::command]
pub async fn delete_dataset(
    app_handle: tauri::AppHandle,
    dataset_id: String,
    model_id:   String,
) -> Result<(), String> {
    let datasets_dir = get_datasets_dir(&app_handle)?;
    let target = datasets_dir.join(&dataset_id);
    if target.exists() {
        fs::remove_dir_all(&target).map_err(|e| format!("Delete: {}", e))?;
    }
    let mut all = load_metadata(&datasets_dir);
    all.retain(|d| !(d.id == dataset_id && d.model_id == model_id));
    save_metadata(&datasets_dir, &all)?;
    if let Ok(db_path) = app_handle.path().app_data_dir().map(|p| p.join("frametrain.db")) {
        if let Ok(conn) = rusqlite::Connection::open(&db_path) {
            conn.execute("DELETE FROM datasets WHERE id=?1", [&dataset_id]).ok();
        }
    }
    Ok(())
}

#[tauri::command]
pub async fn split_dataset(
    app_handle:  tauri::AppHandle,
    dataset_id:  String,
    model_id:    String,
    train_ratio: f64,
    val_ratio:   f64,
    test_ratio:  f64,
) -> Result<DatasetInfo, String> {
    let datasets_dir = get_datasets_dir(&app_handle)?;
    let base = datasets_dir.join(&dataset_id);
    let mut all = load_metadata(&datasets_dir);
    let ds = all.iter().find(|d| d.id == dataset_id && d.model_id == model_id)
        .ok_or("Dataset nicht gefunden")?.clone();

    let files = collect_files(&base);
    let n = files.len();
    if n == 0 { return Err("Keine Dateien im Dataset zum Splitten".to_string()); }

    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut indices: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let mut h = DefaultHasher::new();
        i.hash(&mut h);
        let j = (h.finish() as usize) % (i + 1);
        indices.swap(i, j);
    }

    let train_n = (n as f64 * train_ratio).round() as usize;
    let val_n   = (n as f64 * val_ratio).round()   as usize;
    let test_n  = n - train_n - val_n;

    let train_dir = base.join("train");
    let val_dir   = base.join("val");
    let test_dir  = base.join("test");
    fs::create_dir_all(&train_dir).ok();
    fs::create_dir_all(&val_dir).ok();
    fs::create_dir_all(&test_dir).ok();

    for (slot_idx, file_idx) in indices.iter().enumerate() {
        let src = &files[*file_idx];
        let fname = src.file_name().unwrap_or_default();
        let dst_dir = if slot_idx < train_n { &train_dir }
            else if slot_idx < train_n + val_n { &val_dir }
            else { &test_dir };
        fs::rename(src, dst_dir.join(fname))
            .or_else(|_| fs::copy(src, dst_dir.join(fname)).map(|_| ()))
            .ok();
    }

    let split_info = SplitInfo { train_count: train_n, val_count: val_n, test_count: test_n, train_ratio, val_ratio, test_ratio };
    let (size, fc) = dir_size(&base);
    let updated = DatasetInfo {
        status: "split".to_string(),
        split_info: Some(split_info),
        storage_path: base.to_string_lossy().to_string(),
        extensions: collect_extensions(&base),
        size_bytes: size, file_count: fc,
        ..ds
    };
    all.retain(|d| d.id != dataset_id);
    all.push(updated.clone());
    save_metadata(&datasets_dir, &all)?;
    Ok(updated)
}

#[tauri::command]
pub async fn split_dataset_in_half(
    app_handle: tauri::AppHandle,
    dataset_id: String,
    model_id:   String,
) -> Result<serde_json::Value, String> {
    let datasets_dir = get_datasets_dir(&app_handle)?;
    let base = datasets_dir.join(&dataset_id);
    let all  = load_metadata(&datasets_dir);
    let ds   = all.iter().find(|d| d.id == dataset_id && d.model_id == model_id)
        .ok_or("Dataset nicht gefunden")?.clone();

    let files = collect_files(&base);
    let n = files.len();
    if n == 0 { return Err("Keine Dateien im Dataset".to_string()); }
    let half = n / 2;

    let id_a = format!("ds_{}", &uuid::Uuid::new_v4().to_string().replace("-","")[..12]);
    let id_b = format!("ds_{}", &uuid::Uuid::new_v4().to_string().replace("-","")[..12]);
    let dir_a = datasets_dir.join(&id_a);
    let dir_b = datasets_dir.join(&id_b);
    fs::create_dir_all(&dir_a).ok();
    fs::create_dir_all(&dir_b).ok();

    for (i, f) in files.iter().enumerate() {
        let fname = f.file_name().unwrap_or_default();
        let dst = if i < half { dir_a.join(fname) } else { dir_b.join(fname) };
        fs::copy(f, dst).ok();
    }

    let (sa, fa) = dir_size(&dir_a);
    let (sb, fb) = dir_size(&dir_b);
    let ds_a = make_info(&id_a, &format!("{} (Hälfte 1)", ds.name), &model_id, "local", None, &dir_a, sa, fa, "unused", None);
    let ds_b = make_info(&id_b, &format!("{} (Hälfte 2)", ds.name), &model_id, "local", None, &dir_b, sb, fb, "unused", None);

    let mut all = load_metadata(&datasets_dir);
    all.push(ds_a.clone());
    all.push(ds_b.clone());
    save_metadata(&datasets_dir, &all)?;
    Ok(serde_json::json!({ "dataset_a": ds_a, "dataset_b": ds_b }))
}

#[tauri::command]
pub async fn search_huggingface_datasets(
    query:           String,
    limit:           Option<u32>,
    filter_task:     Option<String>,
    filter_language: Option<String>,
    _filter_size:    Option<String>,
) -> Result<Vec<HuggingFaceDataset>, String> {
    let limit = limit.unwrap_or(15);
    let mut url = format!(
        "https://huggingface.co/api/datasets?search={}&limit={}&sort=downloads&direction=-1",
        urlencoding::encode(&query), limit
    );
    if let Some(t) = &filter_task     { url.push_str(&format!("&pipeline_tag={}", urlencoding::encode(t))); }
    if let Some(l) = &filter_language { url.push_str(&format!("&language={}", urlencoding::encode(l))); }
    println!("[HF Datasets] {}", url);

    let client = reqwest::Client::builder().timeout(std::time::Duration::from_secs(15)).build()
        .map_err(|e| format!("HTTP: {}", e))?;
    let resp = client.get(&url).header("User-Agent", "FrameTrain-Desktop/1.0").send().await
        .map_err(|e| format!("HTTP: {}", e))?;
    if !resp.status().is_success() { return Err(format!("HF API: {}", resp.status())); }

    let raw: Vec<serde_json::Value> = resp.json().await.map_err(|e| format!("JSON: {}", e))?;
    let datasets = raw.iter().filter_map(|v| Some(HuggingFaceDataset {
        id:        v.get("id")?.as_str()?.to_string(),
        author:    v.get("author").and_then(|a| a.as_str()).map(String::from),
        downloads: v.get("downloads").and_then(|d| d.as_u64()),
        likes:     v.get("likes").and_then(|l| l.as_u64()),
        tags:      v.get("tags").and_then(|t| t.as_array()).map(|arr|
            arr.iter().filter_map(|s| s.as_str()).map(String::from).collect()
        ),
    })).collect();
    println!("[HF Datasets] {} Ergebnisse", raw.len());
    Ok(datasets)
}

#[tauri::command]
pub async fn get_dataset_filter_options() -> Result<serde_json::Value, String> {
    Ok(serde_json::json!({
        "tasks": ["text-classification","token-classification","question-answering",
                  "summarization","translation","text-generation","fill-mask",
                  "image-classification","automatic-speech-recognition"],
        "languages": ["de","en","fr","es","it","zh","ja","pt","ru","ar"],
        "sizes": ["n<1K","1K<n<10K","10K<n<100K","100K<n<1M","n>1M"]
    }))
}

/// Parquet-Eintrag aus dem HuggingFace Datasets Server
#[derive(Debug, Deserialize)]
struct HfParquetFile {
    url:      String,
    filename: String,
    size:     Option<u64>,
    split:    Option<String>,
}

/// Dataset-Download von HuggingFace.
/// Strategie:
///   1. Datasets-Server API → direkte Parquet-URLs → echtes Byte-Streaming mit echter Progress Bar
///   2. Fallback: Python `load_dataset()` → Parquet schreiben (bei nicht indizierten Datasets)
#[tauri::command]
pub async fn download_huggingface_dataset(
    app_handle:   tauri::AppHandle,
    repo_id:      String,
    dataset_name: String,
    model_id:     String,
) -> Result<DatasetInfo, String> {
    use std::time::Instant;

    let datasets_dir = get_datasets_dir(&app_handle)?;
    let dataset_id   = format!("ds_{}", &uuid::Uuid::new_v4().to_string().replace("-","")[..12]);
    let target       = datasets_dir.join(&dataset_id);
    fs::create_dir_all(&target).map_err(|e| format!("mkdir: {}", e))?;

    println!("[HF Dataset] {} → {}", repo_id, target.display());

    // Sofortiges "Verbinde…"-Event
    let _ = app_handle.emit("dataset-download-progress", DatasetDownloadProgress {
        status: "connecting".to_string(),
        current_file: String::new(),
        current_file_index: 0, total_files: 0,
        downloaded_bytes: 0, total_bytes: 0, progress_percent: 0,
        speed_mbs: 0.0, elapsed_secs: 0, eta_secs: 0,
        message: "Verbinde mit Hugging Face…".to_string(),
    });

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|e| format!("HTTP client: {}", e))?;

    // ── Schritt 1: Datasets Server API abfragen ──
    let api_url = format!(
        "https://datasets-server.huggingface.co/parquet?dataset={}",
        urlencoding::encode(&repo_id)
    );
    println!("[HF Dataset] Datasets Server API: {}", api_url);

    let api_result = client.get(&api_url)
        .header("User-Agent", "FrameTrain-Desktop/1.0")
        .send().await;

    if let Ok(resp) = api_result {
        if resp.status().is_success() {
            if let Ok(api_json) = resp.json::<serde_json::Value>().await {
                let parquet_files: Vec<HfParquetFile> = api_json
                    .get("parquet_files")
                    .and_then(|f| f.as_array())
                    .map(|arr| arr.iter().filter_map(|v| serde_json::from_value(v.clone()).ok()).collect())
                    .unwrap_or_default();

                if !parquet_files.is_empty() {
                    println!("[HF Dataset] Datasets Server: {} Parquet-Dateien gefunden", parquet_files.len());
                    return download_parquet_direct(
                        app_handle, client, parquet_files, target,
                        dataset_id, dataset_name, model_id, datasets_dir, repo_id,
                    ).await;
                }
            }
        }
    }

    // ── Schritt 2: Fallback → Python ──
    println!("[HF Dataset] Datasets Server nicht verfügbar – Fallback auf Python");
    download_via_python(
        app_handle, repo_id, target,
        dataset_id, dataset_name, model_id, datasets_dir,
    ).await
}

/// Lädt Parquet-Dateien direkt von HuggingFace mit echtem Byte-Streaming und Progress-Events.
async fn download_parquet_direct(
    app_handle:   tauri::AppHandle,
    client:       reqwest::Client,
    files:        Vec<HfParquetFile>,
    target:       PathBuf,
    dataset_id:   String,
    dataset_name: String,
    model_id:     String,
    datasets_dir: PathBuf,
    repo_id:      String,
) -> Result<DatasetInfo, String> {
    use std::time::Instant;

    let total_files  = files.len();
    let total_bytes: u64 = files.iter().filter_map(|f| f.size).sum();
    let download_start = Instant::now();
    let mut global_downloaded: u64 = 0;

    for (file_idx, hf_file) in files.iter().enumerate() {
        let file_name = if hf_file.filename.is_empty() {
            format!("file_{}.parquet", file_idx)
        } else {
            hf_file.filename.clone()
        };
        let output_path = target.join(&file_name);

        println!("[HF Dataset] Lade Datei {}/{}: {}", file_idx + 1, total_files, file_name);

        // Datei-Download mit Chunk-Streaming
        let response = client.get(&hf_file.url)
            .header("User-Agent", "FrameTrain-Desktop/1.0")
            .send().await
            .map_err(|e| format!("HTTP GET '{}': {}", file_name, e))?;

        if !response.status().is_success() {
            return Err(format!("HTTP {} für Datei '{}'", response.status(), file_name));
        }

        // Content-Length für diese Datei ermitteln (falls nicht in API)
        let file_total = hf_file.size
            .or_else(|| response.content_length())
            .unwrap_or(0);

        let mut output_file = tokio::fs::File::create(&output_path).await
            .map_err(|e| format!("Datei erstellen '{}': {}", file_name, e))?;

        let mut file_downloaded: u64 = 0;
        let mut stream = response.bytes_stream();
        let mut last_emit = Instant::now();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| format!("Chunk-Fehler: {}", e))?;
            output_file.write_all(&chunk).await
                .map_err(|e| format!("Schreiben '{}': {}", file_name, e))?;

            file_downloaded  += chunk.len() as u64;
            global_downloaded += chunk.len() as u64;

            // Nicht jeden Chunk emitten – max. ~30 mal pro Sekunde
            if last_emit.elapsed().as_millis() >= 33 {
                last_emit = Instant::now();
                let elapsed      = download_start.elapsed().as_secs();
                let global_pct   = if total_bytes > 0 {
                    ((global_downloaded as f64 / total_bytes as f64) * 100.0) as i32
                } else {
                    let file_pct = if file_total > 0 { (file_downloaded as f64 / file_total as f64) * 100.0 } else { 0.0 };
                    (((file_idx as f64 + file_pct / 100.0) / total_files as f64) * 100.0) as i32
                };
                let speed = if elapsed > 0 {
                    (global_downloaded as f32 / 1_048_576.0) / elapsed as f32
                } else { 0.0 };
                let eta = if speed > 0.0 && total_bytes > global_downloaded {
                    ((total_bytes - global_downloaded) as f32 / (speed * 1_048_576.0)) as u64
                } else { 0 };

                let split_label = hf_file.split.as_deref().unwrap_or("?");
                let _ = app_handle.emit("dataset-download-progress", DatasetDownloadProgress {
                    status: "downloading".to_string(),
                    current_file: format!("{} ({})", file_name, split_label),
                    current_file_index: file_idx + 1,
                    total_files,
                    downloaded_bytes: global_downloaded,
                    total_bytes,
                    progress_percent: global_pct.clamp(0, 99),
                    speed_mbs: speed,
                    elapsed_secs: elapsed,
                    eta_secs: eta,
                    message: format!("Datei {}/{}: {}", file_idx + 1, total_files, file_name),
                });
            }
        }

        // Datei-Handle schließen
        drop(output_file);

        let actual_size = fs::metadata(&output_path).map(|m| m.len()).unwrap_or(file_downloaded);
        let elapsed = download_start.elapsed().as_secs();
        let speed = if elapsed > 0 { (global_downloaded as f32 / 1_048_576.0) / elapsed as f32 } else { 0.0 };

        let global_pct = if total_bytes > 0 {
            ((global_downloaded as f64 / total_bytes as f64) * 100.0) as i32
        } else {
            (((file_idx + 1) as f64 / total_files as f64) * 100.0) as i32
        };
        let eta = if speed > 0.0 && total_bytes > global_downloaded {
            ((total_bytes - global_downloaded) as f32 / (speed * 1_048_576.0)) as u64
        } else { 0 };

        let split_label = hf_file.split.as_deref().unwrap_or("?");
        let _ = app_handle.emit("dataset-download-progress", DatasetDownloadProgress {
            status: "downloading".to_string(),
            current_file: format!("{} ({})", file_name, split_label),
            current_file_index: file_idx + 1,
            total_files,
            downloaded_bytes: global_downloaded,
            total_bytes,
            progress_percent: global_pct.clamp(0, 99),
            speed_mbs: speed,
            elapsed_secs: elapsed,
            eta_secs: eta,
            message: format!("{} ✓ ({:.1} MB)", file_name, actual_size as f64 / 1_048_576.0),
        });

        println!("[HF Dataset] ✓ {} ({} bytes)", file_name, actual_size);
    }

    // Abschluss-Event
    let elapsed = download_start.elapsed().as_secs();
    let (total_size, file_count) = dir_size(&target);
    let speed = if elapsed > 0 { (total_size as f32 / 1_048_576.0) / elapsed as f32 } else { 0.0 };

    let _ = app_handle.emit("dataset-download-progress", DatasetDownloadProgress {
        status: "complete".to_string(),
        current_file: String::new(),
        current_file_index: file_count,
        total_files: file_count,
        downloaded_bytes: total_size,
        total_bytes: total_size,
        progress_percent: 100,
        speed_mbs: speed,
        elapsed_secs: elapsed,
        eta_secs: 0,
        message: format!("Fertig! ({} Dateien, {:.1} MB)", file_count, total_size as f64 / 1_048_576.0),
    });

    let info = make_info(&dataset_id, &dataset_name, &model_id, "huggingface",
        Some(repo_id), &target, total_size, file_count, "unused", None);
    upsert_metadata(&datasets_dir, &info)?;

    println!("[HF Dataset] ✅ {} Dateien, {} bytes", file_count, total_size);
    Ok(info)
}

/// Fallback: Download via Python `datasets`-Library (für nicht-indexierte Datasets).
/// Fortschritt kann nur grob per Split-Event gemeldet werden.
async fn download_via_python(
    app_handle:   tauri::AppHandle,
    repo_id:      String,
    target:       PathBuf,
    dataset_id:   String,
    dataset_name: String,
    model_id:     String,
    datasets_dir: PathBuf,
) -> Result<DatasetInfo, String> {
    use std::time::Instant;
    use std::process::{Command, Stdio};

    let _ = app_handle.emit("dataset-download-progress", DatasetDownloadProgress {
        status: "preparing".to_string(),
        current_file: String::new(),
        current_file_index: 0, total_files: 0,
        downloaded_bytes: 0, total_bytes: 0, progress_percent: 0,
        speed_mbs: 0.0, elapsed_secs: 0, eta_secs: 0,
        message: format!("Lade '{}' via Python datasets…", repo_id),
    });

    let python_script = r#"
import sys
import json
from datasets import load_dataset, get_dataset_config_names
from pathlib import Path

repo_id = sys.argv[1]
target = Path(sys.argv[2])
target.mkdir(parents=True, exist_ok=True)

def emit(obj):
    print(json.dumps(obj), flush=True)

try:
    dataset = None
    config_name = None

    emit({"type": "status", "phase": "loading", "message": f"Lade '{repo_id}' von Hugging Face..."})

    try:
        dataset = load_dataset(repo_id)
    except Exception as e:
        error_msg = str(e)
        if "Config name is missing" in error_msg or "Please pick one among" in error_msg:
            emit({"type": "status", "phase": "config", "message": "Dataset benötigt eine Config – lade verfügbare Configs..."})
            configs = get_dataset_config_names(repo_id)
            if not configs:
                raise Exception(f"Keine Configs für '{repo_id}' gefunden.")
            config_name = configs[0]
            emit({"type": "status", "phase": "config", "message": f"Verwende Config '{config_name}' (verfügbar: {configs})"})
            emit({"type": "status", "phase": "loading", "message": f"Lade '{repo_id}' mit Config '{config_name}'..."})
            dataset = load_dataset(repo_id, config_name)
        else:
            raise

    splits = dataset if isinstance(dataset, dict) else {"default": dataset}
    total_splits = len(splits)
    emit({"type": "status", "phase": "saving", "message": f"Dataset geladen! Speichere {total_splits} Split(s) als Parquet..."})

    file_count = 0
    total_size = 0
    for split_name, split_data in splits.items():
        emit({"type": "status", "phase": "saving", "message": f"Schreibe Split '{split_name}' ({len(split_data)} Zeilen)..."})
        output_file = target / f"{split_name}.parquet"
        split_data.to_parquet(str(output_file))
        file_size = output_file.stat().st_size
        total_size += file_size
        file_count += 1
        emit({"type": "file_done", "split": split_name, "size": file_size, "total_files": total_splits, "file_index": file_count})

    emit({"type": "complete", "files": file_count, "total_size": total_size, "config": config_name or "default"})

except Exception as e:
    print(json.dumps({"type": "error", "message": str(e)}), file=sys.stderr, flush=True)
    sys.exit(1)
"#;

    let script_file = std::env::temp_dir().join("hf_dataset_download.py");
    fs::write(&script_file, python_script)
        .map_err(|e| format!("Script schreiben: {}", e))?;

    let python_cmd = if Command::new("python3").arg("--version").output().is_ok() { "python3" }
                     else if Command::new("python").arg("--version").output().is_ok() { "python" }
                     else { return Err("Python nicht gefunden".to_string()); };

    let mut child = Command::new(python_cmd)
        .arg(&script_file)
        .arg(&repo_id)
        .arg(target.to_string_lossy().to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Python spawn: {}", e))?;

    let stdout_pipe = child.stdout.take().expect("stdout pipe");
    let stderr_pipe = child.stderr.take().expect("stderr pipe");

    let stderr_handle = std::thread::spawn(move || {
        let mut buf = String::new();
        BufReader::new(stderr_pipe).read_to_string(&mut buf).ok();
        buf
    });

    let download_start = Instant::now();
    let mut count = 0usize;
    let mut total_written = 0u64;
    let mut total_splits_known = 0usize;

    // stdout-Loop in spawn_blocking, damit der tokio-Thread nicht blockiert wird
    let app_clone = app_handle.clone();
    let stdout_result: Result<(), String> = tokio::task::spawn_blocking(move || {
        for line in BufReader::new(stdout_pipe).lines().flatten() {
            println!("[HF Dataset] py> {}", line);

            if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(&line) {
                let elapsed = download_start.elapsed().as_secs();

                match json_val.get("type").and_then(|t| t.as_str()).unwrap_or("") {
                    "status" => {
                        let msg = json_val.get("message").and_then(|m| m.as_str()).unwrap_or("").to_string();
                        let _ = app_clone.emit("dataset-download-progress", DatasetDownloadProgress {
                            status: "preparing".to_string(),
                            current_file: String::new(),
                            current_file_index: 0,
                            total_files: total_splits_known,
                            downloaded_bytes: 0,
                            total_bytes: 0,
                            progress_percent: 0,
                            speed_mbs: 0.0,
                            elapsed_secs: elapsed,
                            eta_secs: 0,
                            message: msg,
                        });
                    }
                    "file_done" => {
                        if let Some(size) = json_val.get("size").and_then(|s| s.as_u64()) {
                            let split = json_val.get("split").and_then(|s| s.as_str()).unwrap_or("split").to_string();
                            total_splits_known = json_val.get("total_files").and_then(|t| t.as_u64()).unwrap_or(0) as usize;
                            let file_index = json_val.get("file_index").and_then(|i| i.as_u64()).unwrap_or(1) as usize;
                            count += 1;
                            total_written += size;
                            let pct = if total_splits_known > 0 {
                                ((file_index as f32 / total_splits_known as f32) * 100.0) as i32
                            } else { 0 };
                            let speed = if elapsed > 0 { (total_written as f32 / 1_048_576.0) / elapsed as f32 } else { 0.0 };
                            let _ = app_clone.emit("dataset-download-progress", DatasetDownloadProgress {
                                status: "downloading".to_string(),
                                current_file: format!("{}.parquet", split),
                                current_file_index: file_index,
                                total_files: total_splits_known,
                                downloaded_bytes: total_written,
                                total_bytes: total_written,
                                progress_percent: pct,
                                speed_mbs: speed,
                                elapsed_secs: elapsed,
                                eta_secs: 0,
                                message: format!("{} ✓ ({:.1} MB)", split, size as f64 / 1_048_576.0),
                            });
                        }
                    }
                    "complete" => {
                        if let (Some(files), Some(size)) = (
                            json_val.get("files").and_then(|f| f.as_u64()),
                            json_val.get("total_size").and_then(|s| s.as_u64()),
                        ) {
                            let speed = if elapsed > 0 { (size as f32 / 1_048_576.0) / elapsed as f32 } else { 0.0 };
                            let _ = app_clone.emit("dataset-download-progress", DatasetDownloadProgress {
                                status: "complete".to_string(),
                                current_file: String::new(),
                                current_file_index: files as usize,
                                total_files: files as usize,
                                downloaded_bytes: size,
                                total_bytes: size,
                                progress_percent: 100,
                                speed_mbs: speed,
                                elapsed_secs: elapsed,
                                eta_secs: 0,
                                message: format!("Fertig! ({} Splits, {:.1} MB)", files, size as f64 / 1_048_576.0),
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }).await.map_err(|e| format!("spawn_blocking: {}", e))?;

    stdout_result?;

    let exit_status = child.wait().map_err(|e| format!("Wait: {}", e))?;
    let stderr = stderr_handle.join().unwrap_or_default();
    fs::remove_file(&script_file).ok();

    if !exit_status.success() {
        fs::remove_dir_all(&target).ok();
        let user_error = stderr.lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .find(|v| v.get("type").and_then(|t| t.as_str()) == Some("error"))
            .and_then(|v| v.get("message").and_then(|m| m.as_str()).map(String::from))
            .unwrap_or_else(|| stderr.trim().to_string());
        return Err(format!("Dataset download fehlgeschlagen: {}", user_error));
    }

    // Fallback: Ordner scannen wenn keine Events ankamen
    if count == 0 {
        if let Ok(entries) = fs::read_dir(&target) {
            for entry in entries.flatten() {
                if let Ok(meta) = entry.metadata() {
                    if meta.is_file() { count += 1; total_written += meta.len(); }
                }
            }
        }
    }

    if count == 0 {
        fs::remove_dir_all(&target).ok();
        return Err(format!(
            "Keine Dateien von '{}' heruntergeladen. Das Dataset könnte nicht zugänglich sein.",
            repo_id
        ));
    }

    let info = make_info(&dataset_id, &dataset_name, &model_id, "huggingface",
        Some(repo_id), &target, total_written, count, "unused", None);
    upsert_metadata(&datasets_dir, &info)?;

    println!("[HF Dataset] ✅ {} Dateien, {} bytes", count, total_written);
    Ok(info)
}

#[tauri::command]
pub async fn get_dataset_files(
    app_handle: tauri::AppHandle,
    dataset_id: String,
) -> Result<Vec<serde_json::Value>, String> {
    let datasets_dir = get_datasets_dir(&app_handle)?;
    let dataset_dir  = datasets_dir.join(&dataset_id);
    if !dataset_dir.exists() { return Ok(vec![]); }

    let mut files: Vec<serde_json::Value> = Vec::new();

    for split in &["train", "val", "test"] {
        let split_dir = dataset_dir.join(split);
        if split_dir.exists() {
            for file in collect_files_recursive(&split_dir) {
                if let Ok(meta) = fs::metadata(&file) {
                    files.push(serde_json::json!({
                        "name":   file.file_name().unwrap_or_default().to_string_lossy(),
                        "path":   file.to_string_lossy(),
                        "size":   meta.len(),
                        "is_dir": false,
                        "split":  split,
                    }));
                }
            }
        }
    }

    let unused_dir = dataset_dir.join("unused");
    if unused_dir.exists() {
        for file in collect_files_recursive(&unused_dir) {
            if let Ok(meta) = fs::metadata(&file) {
                files.push(serde_json::json!({
                    "name":   file.file_name().unwrap_or_default().to_string_lossy(),
                    "path":   file.to_string_lossy(),
                    "size":   meta.len(),
                    "is_dir": false,
                    "split":  "unsplit",
                }));
            }
        }
    }

    if let Ok(entries) = fs::read_dir(&dataset_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if matches!(file_name, "train" | "val" | "test" | "unused" | "images" | "labels") { continue; }
            if path.is_file() {
                if let Ok(meta) = fs::metadata(&path) {
                    let tag = if matches!(file_name, "dataset_infos.json" | "metadata.json" | ".gitkeep" | ".DS_Store") {
                        "info"
                    } else { "unsplit" };
                    files.push(serde_json::json!({
                        "name":   file_name,
                        "path":   path.to_string_lossy(),
                        "size":   meta.len(),
                        "is_dir": false,
                        "split":  tag,
                    }));
                }
            }
        }
    }

    Ok(files)
}

#[tauri::command]
pub async fn read_dataset_file(file_path: String) -> Result<String, String> {
    let path = Path::new(&file_path);
    if !path.exists() { return Err(format!("Datei nicht gefunden: {}", file_path)); }
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    if matches!(ext.as_str(), "txt"|"json"|"jsonl"|"csv"|"tsv"|"md"|"log"|"xml"|"yaml"|"yml") {
        let content = fs::read_to_string(path).map_err(|e| format!("Lesen: {}", e))?;
        let lines: Vec<&str> = content.lines().collect();
        let preview = lines.iter().take(200).cloned().collect::<Vec<_>>().join("\n");
        if lines.len() > 200 {
            return Ok(format!("{}\n\n--- [Vorschau: 200 von {} Zeilen] ---", preview, lines.len()));
        }
        return Ok(preview);
    }
    if ext == "parquet" {
        let size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        return Ok(format!("[Parquet] {} bytes – Binärformat, kein Preview.", size));
    }
    let size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    Ok(format!("[{}.{}] {} bytes – kein Preview.", path.file_name().unwrap_or_default().to_string_lossy(), ext, size))
}

#[tauri::command]
pub async fn move_dataset_files(
    app_handle: tauri::AppHandle,
    dataset_id: String,
    file_paths: Vec<String>,
    target_split: String,
) -> Result<(), String> {
    let datasets_dir = get_datasets_dir(&app_handle)?;
    let dataset_dir = datasets_dir.join(&dataset_id);
    let target_dir = dataset_dir.join(&target_split);
    fs::create_dir_all(&target_dir).map_err(|e| format!("mkdir: {}", e))?;

    for fp in &file_paths {
        let src = Path::new(fp);
        if src.exists() && src.is_file() {
            let fname = src.file_name().unwrap_or_default();
            let dst = target_dir.join(fname);
            if let Err(_) = fs::rename(src, &dst) {
                fs::copy(src, &dst).map_err(|e| format!("Copy: {}", e))?;
                fs::remove_file(src).map_err(|e| format!("Delete original: {}", e))?;
            }
        }
    }
    Ok(())
}

#[tauri::command]
pub async fn delete_dataset_files(file_paths: Vec<String>) -> Result<(), String> {
    for fp in &file_paths {
        let p = Path::new(fp);
        if p.exists() { fs::remove_file(p).map_err(|e| format!("Delete: {}", e))?; }
    }
    Ok(())
}

#[tauri::command]
pub async fn add_files_to_dataset(
    app_handle: tauri::AppHandle,
    dataset_id: String,
    file_paths: Vec<String>,
) -> Result<serde_json::Value, String> {
    let dst = get_datasets_dir(&app_handle)?.join(&dataset_id);
    fs::create_dir_all(&dst).ok();
    let mut added = 0usize;
    for fp in &file_paths {
        let src = Path::new(fp);
        if src.exists() {
            fs::copy(src, dst.join(src.file_name().unwrap_or_default()))
                .map_err(|e| format!("Copy: {}", e))?;
            added += 1;
        }
    }
    Ok(serde_json::json!({ "added": added }))
}

#[tauri::command]
pub async fn validate_image_label_folders(path: String) -> Result<serde_json::Value, String> {
    let p = Path::new(&path);
    let valid = p.is_dir() && fs::read_dir(p).map(|mut e| e.next().is_some()).unwrap_or(false);
    Ok(serde_json::json!({ "valid": valid }))
}

#[tauri::command]
pub async fn import_structured_dataset(
    app_handle: tauri::AppHandle, source_path: String,
    dataset_name: String, model_id: String,
) -> Result<DatasetInfo, String> {
    import_local_dataset(app_handle, source_path, dataset_name, model_id).await
}
