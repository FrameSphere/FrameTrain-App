// dataset_manager.rs – vollständige Implementierung

use std::fs;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use tauri::Manager;
use chrono::Utc;

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
    pub storage_path:   String,          // lokaler Speicherpfad
    pub size_bytes:     u64,
    pub file_count:     usize,
    pub created_at:     String,
    pub status:         String,
    pub split_info:     Option<SplitInfo>,
    pub training_count: i64,
    pub last_used_at:   Option<String>,
    #[serde(default)]
    pub extensions:     Vec<String>,     // Dateiendungen im Dataset
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HuggingFaceDataset {
    pub id:        String,
    pub author:    Option<String>,
    pub downloads: Option<u64>,
    pub likes:     Option<u64>,
    pub tags:      Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct HFTreeEntry {
    #[serde(alias = "rfilename", alias = "path")]
    path: String,
    size: Option<u64>,
    #[serde(rename = "type")]
    entry_type: Option<String>,
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

/// Einzigartige Dateiendungen (lowercase, mit Punkt) in einem Verzeichnis
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

// Sammelt nur Dateien im Root-Verzeichnis (nicht in train/val/test/unused)
// Ignoriert Metadateien
fn collect_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            
            let file_name = path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
            
            // Ignoriere Metadateien
            if matches!(file_name, "dataset_infos.json" | "metadata.json" | ".gitkeep" | ".DS_Store") {
                continue;
            }
            
            files.push(path);
        }
    }
    files
}

// Hilfsfunktion: Sammelt Dateien rekursiv in einem Verzeichnis
fn collect_files_recursive(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    fn walk(p: &Path, out: &mut Vec<PathBuf>) {
        if let Ok(entries) = fs::read_dir(p) {
            for e in entries.flatten() {
                let ep = e.path();
                if ep.is_file() {
                    out.push(ep);
                } else if ep.is_dir() {
                    walk(&ep, out);
                }
            }
        }
    }
    walk(dir, &mut files);
    files
}

// Erstellt leeres DatasetInfo mit storage_path und extensions
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

    // Deduplizieren: bei gleichem Namen nur den neuesten behalten
    result.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    let mut seen = std::collections::HashSet::new();
    result.retain(|d| seen.insert(d.name.clone()));
    result.sort_by(|a, b| a.created_at.cmp(&b.created_at));

    // training_count aus DB
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

    // Sammle Dateien nur aus Root (nicht aus Split-Ordnern)
    let files = collect_files(&base);
    let n = files.len();
    if n == 0 { return Err("Keine Dateien im Dataset zum Splitten (nur Root-Dateien werden gesplittet)".to_string()); }

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
        // Verschiebe (nicht kopiere) die Datei
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
    _filter_size:    Option<String>,  // reserviert für spätere Nutzung
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
pub async fn get_huggingface_dataset_files(dataset_id: String) -> Result<Vec<serde_json::Value>, String> {
    let url = format!("https://huggingface.co/api/datasets/{}/tree/main", dataset_id);
    let client = reqwest::Client::builder().timeout(std::time::Duration::from_secs(15)).build()
        .map_err(|e| format!("HTTP: {}", e))?;
    let resp = client.get(&url).header("User-Agent", "FrameTrain-Desktop/1.0").send().await
        .map_err(|e| format!("HTTP: {}", e))?;
    if !resp.status().is_success() { return Err(format!("HF API: {}", resp.status())); }
    resp.json().await.map_err(|e| format!("JSON: {}", e))
}

#[tauri::command]
pub async fn download_huggingface_dataset(
    app_handle:   tauri::AppHandle,
    repo_id:      String,
    dataset_name: String,
    model_id:     String,
) -> Result<DatasetInfo, String> {
    let datasets_dir = get_datasets_dir(&app_handle)?;
    let dataset_id   = format!("ds_{}", &uuid::Uuid::new_v4().to_string().replace("-","")[..12]);
    let target       = datasets_dir.join(&dataset_id);
    fs::create_dir_all(&target).ok();

    let url = format!("https://huggingface.co/api/datasets/{}/tree/main", repo_id);
    let client = reqwest::Client::builder().timeout(std::time::Duration::from_secs(600)).build()
        .map_err(|e| format!("HTTP: {}", e))?;

    let resp = client.get(&url).header("User-Agent", "FrameTrain-Desktop/1.0").send().await
        .map_err(|e| format!("HTTP: {}", e))?;
    let files: Vec<HFTreeEntry> = if resp.status().is_success() {
        resp.json().await.unwrap_or_default()
    } else { vec![] };

    let relevant: Vec<&HFTreeEntry> = files.iter().filter(|f| {
        if let Some(ref t) = f.entry_type { if t == "directory" { return false; } }
        let n = f.path.to_lowercase();
        n.ends_with(".json") || n.ends_with(".jsonl") || n.ends_with(".csv")
            || n.ends_with(".parquet") || n.ends_with(".txt") || n.ends_with(".arrow")
            || n.contains("train") || n.contains("test")
    }).collect();

    let mut total = 0u64; let mut count = 0usize;

    if relevant.is_empty() {
        for split in &["train", "validation", "test"] {
            for ext in &["csv", "jsonl", "parquet"] {
                let fname = format!("{}.{}", split, ext);
                let dl_url = format!("https://huggingface.co/datasets/{}/resolve/main/{}", repo_id, fname);
                if let Ok(r) = client.get(&dl_url).header("User-Agent", "FrameTrain-Desktop/1.0").send().await {
                    if r.status().is_success() {
                        if let Ok(bytes) = r.bytes().await {
                            fs::write(target.join(&fname), &bytes).ok();
                            total += bytes.len() as u64; count += 1; break;
                        }
                    }
                }
            }
        }
    } else {
        for f in &relevant {
            let dl_url = format!("https://huggingface.co/datasets/{}/resolve/main/{}", repo_id, urlencoding::encode(&f.path));
            if let Ok(r) = client.get(&dl_url).header("User-Agent", "FrameTrain-Desktop/1.0").send().await {
                if r.status().is_success() {
                    if let Ok(bytes) = r.bytes().await {
                        let dest = target.join(&f.path);
                        if let Some(p) = dest.parent() { fs::create_dir_all(p).ok(); }
                        fs::write(&dest, &bytes).ok();
                        total += bytes.len() as u64; count += 1;
                    }
                }
            }
        }
    }

    if count == 0 {
        fs::remove_dir_all(&target).ok();
        return Err(format!("Keine Dateien von '{}' heruntergeladen. Versuche lokalen Import.", repo_id));
    }

    let info = make_info(&dataset_id, &dataset_name, &model_id, "huggingface",
        Some(repo_id), &target, total, count, "unused", None);
    upsert_metadata(&datasets_dir, &info)?;
    Ok(info)
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

fn detect_file_split(file_path: &Path, dataset_dir: &Path) -> &'static str {
    if let Ok(rel) = file_path.strip_prefix(dataset_dir) {
        let comps: Vec<_> = rel.components().collect();
        if comps.len() >= 2 {
            let f = comps[0].as_os_str().to_string_lossy().to_lowercase();
            match f.as_str() {
                "train" | "training"             => return "train",
                "val" | "valid" | "validation"   => return "val",
                "test" | "testing" | "eval"      => return "test",
                _ => {}
            }
        }
    }
    "unsplit"
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
    
    // 1. Sammle Dateien aus Split-Ordnern (train / val / test) mit rekursiver Suche
    for split in &["train", "val", "test"] {
        let split_dir = dataset_dir.join(split);
        if split_dir.exists() {
            let split_files = collect_files_recursive(&split_dir);
            for file in split_files {
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
    
    // 2. Sammle Dateien aus "unused/" (noch nicht gesplittet)
    let unused_dir = dataset_dir.join("unused");
    if unused_dir.exists() {
        let unused_files = collect_files_recursive(&unused_dir);
        for file in unused_files {
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
    
    // 3. Sammle Dateien direkt im Root (überspringen wir bekannte Unterordner)
    if let Ok(entries) = fs::read_dir(&dataset_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let file_name = path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
            
            // Überspringe bekannte Unterordner
            if matches!(file_name, "train" | "val" | "test" | "unused" | "images" | "labels") {
                continue;
            }
            
            if path.is_file() {
                if let Ok(meta) = fs::metadata(&path) {
                    // Bestimme das Tag basierend auf Dateiname
                    let tag = if matches!(file_name, "dataset_infos.json" | "metadata.json" | ".gitkeep" | ".DS_Store") {
                        "info"
                    } else {
                        "unsplit"
                    };
                    
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
    
    // Erstelle das Ziel-Split-Verzeichnis falls nötig
    let target_dir = dataset_dir.join(&target_split);
    fs::create_dir_all(&target_dir).map_err(|e| format!("Konnte Verzeichnis nicht erstellen: {}", e))?;
    
    for fp in &file_paths {
        let src = Path::new(fp);
        if src.exists() && src.is_file() {
            let fname = src.file_name().unwrap_or_default();
            let dst = target_dir.join(fname);
            
            // Versuche zu verschieben, bei Fehler kopieren und original löschen
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
