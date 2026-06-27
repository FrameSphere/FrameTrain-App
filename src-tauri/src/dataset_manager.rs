// dataset_manager.rs – v2: User-Isolation + DatasetType-Erkennung + Pairing-Validierung
//
// ARCHITEKTUR:
//  - Dateien liegen in: <app_data_dir>/datasets/<user_id>/<dataset_id>/
//  - Metadaten:         <app_data_dir>/datasets/<user_id>/datasets_metadata.json
//  - SQLite nur für training_count / last_used_at (kein Primary Source)
//  - Jeder Command mit Datenzugriff bekommt State<'_, AppState> → user_id

use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use tauri::{Manager, Emitter, State};
use chrono::Utc;
use serde_json;
use futures_util::StreamExt;
use tokio::io::AsyncWriteExt;

use crate::AppState;

// ══════════════════════════════════════════════════════════════════
// TYPEN
// ══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum DatasetType {
    FlatFile,
    YoloBbox,
    CocoJson,
    PascalVoc,
    FolderClass,
    AudioTranscript,
    CommonVoice,
    PreSplit,
    MultiShard,
    #[default]
    Unknown,
}

impl DatasetType {
    pub fn as_str(&self) -> &'static str {
        match self {
            DatasetType::FlatFile        => "flat_file",
            DatasetType::YoloBbox        => "yolo_bbox",
            DatasetType::CocoJson        => "coco_json",
            DatasetType::PascalVoc       => "pascal_voc",
            DatasetType::FolderClass     => "folder_class",
            DatasetType::AudioTranscript => "audio_transcript",
            DatasetType::CommonVoice     => "common_voice",
            DatasetType::PreSplit        => "pre_split",
            DatasetType::MultiShard      => "multi_shard",
            DatasetType::Unknown         => "unknown",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PairingStatus {
    pub is_paired:          bool,
    pub primary_count:      usize,
    pub paired_count:       usize,
    pub orphan_primaries:   Vec<String>,
    pub orphan_secondaries: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DatasetAnalysis {
    pub detected_type:  DatasetType,
    pub confidence:     u8,
    pub pairing_status: Option<PairingStatus>,
    pub warnings:       Vec<String>,
    pub file_count:     usize,
    pub dir_count:      usize,
    pub extensions:     Vec<String>,
    pub schema_hint:    Option<serde_json::Value>,
}

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
    #[serde(default)]
    pub dataset_type:   DatasetType,
    #[serde(default)]
    pub pairing_status: Option<PairingStatus>,
    #[serde(default)]
    pub warnings:       Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DatasetDownloadProgress {
    pub status:             String,
    pub current_file:       String,
    pub current_file_index: usize,
    pub total_files:        usize,
    pub downloaded_bytes:   u64,
    pub total_bytes:        u64,
    pub progress_percent:   i32,
    pub speed_mbs:          f32,
    pub elapsed_secs:       u64,
    pub eta_secs:           u64,
    pub message:            String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HuggingFaceDataset {
    pub id:        String,
    pub author:    Option<String>,
    pub downloads: Option<u64>,
    pub likes:     Option<u64>,
    pub tags:      Option<Vec<String>>,
}

// ══════════════════════════════════════════════════════════════════
// USER-SCOPED FILESYSTEM HELPERS
// ══════════════════════════════════════════════════════════════════

fn get_user_id(state: &State<'_, AppState>) -> Result<String, String> {
    let db = state.db.lock().map_err(|e| format!("DB lock: {}", e))?;
    db.get_current_user_id()
        .ok_or_else(|| "Kein Benutzer angemeldet".to_string())
}

fn sanitize_user_id(user_id: &str) -> String {
    user_id.chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}

fn get_datasets_dir(app_handle: &tauri::AppHandle, user_id: &str) -> Result<PathBuf, String> {
    let dir = app_handle.path().app_data_dir()
        .map_err(|e| format!("AppDataDir: {}", e))?
        .join("datasets")
        .join(sanitize_user_id(user_id));
    fs::create_dir_all(&dir).map_err(|e| format!("mkdir datasets: {}", e))?;
    Ok(dir)
}

// ══════════════════════════════════════════════════════════════════
// METADATA HELPERS
// ══════════════════════════════════════════════════════════════════

fn meta_path(datasets_dir: &Path) -> PathBuf { datasets_dir.join("datasets_metadata.json") }

fn load_metadata(datasets_dir: &Path) -> Vec<DatasetInfo> {
    let path = meta_path(datasets_dir);
    if !path.exists() { return vec![]; }
    serde_json::from_str(&fs::read_to_string(&path).unwrap_or_default()).unwrap_or_default()
}

fn save_metadata(datasets_dir: &Path, datasets: &[DatasetInfo]) -> Result<(), String> {
    let path = meta_path(datasets_dir);
    fs::write(&path, serde_json::to_string_pretty(datasets)
        .map_err(|e| format!("JSON: {}", e))?)
        .map_err(|e| format!("Metadaten schreiben: {}", e))
}

fn upsert_metadata(datasets_dir: &Path, info: &DatasetInfo) -> Result<(), String> {
    let mut all = load_metadata(datasets_dir);
    all.retain(|d| d.id != info.id);
    all.push(info.clone());
    save_metadata(datasets_dir, &all)
}

// ══════════════════════════════════════════════════════════════════
// FILESYSTEM UTILITIES
// ══════════════════════════════════════════════════════════════════

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
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if matches!(name, "dataset_infos.json" | "metadata.json" | ".gitkeep" | ".DS_Store") { continue; }
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
    dataset_type: DatasetType,
    pairing_status: Option<PairingStatus>,
    warnings: Vec<String>,
) -> DatasetInfo {
    DatasetInfo {
        id: id.to_string(), name: name.to_string(),
        model_id: model_id.to_string(), source: source.to_string(),
        source_path, storage_path: target.to_string_lossy().to_string(),
        size_bytes, file_count,
        created_at: Utc::now().to_rfc3339(),
        status: status.to_string(), split_info,
        training_count: 0, last_used_at: None,
        extensions: collect_extensions(target),
        dataset_type, pairing_status, warnings,
    }
}

// ══════════════════════════════════════════════════════════════════
// DATASET TYPE DETECTION ENGINE
// ══════════════════════════════════════════════════════════════════

const IMAGE_EXTS: &[&str] = &["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp", "gif"];
const AUDIO_EXTS: &[&str] = &["wav", "mp3", "flac", "ogg", "m4a", "aac", "opus"];
const FLAT_EXTS:  &[&str] = &["jsonl", "json", "csv", "tsv", "parquet", "arrow", "txt"];

fn is_image(ext: &str) -> bool { IMAGE_EXTS.contains(&ext.to_lowercase().as_str()) }
fn is_audio(ext: &str) -> bool { AUDIO_EXTS.contains(&ext.to_lowercase().as_str()) }
fn is_flat(ext: &str)  -> bool { FLAT_EXTS.contains(&ext.to_lowercase().as_str())  }

fn get_basename(path: &Path) -> String {
    path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_lowercase()
}

fn list_subdir_names(dir: &Path) -> Vec<String> {
    if let Ok(entries) = fs::read_dir(dir) {
        entries.flatten()
            .filter(|e| e.path().is_dir())
            .filter_map(|e| e.file_name().into_string().ok())
            .collect()
    } else { vec![] }
}

fn list_files_in_dir(dir: &Path) -> Vec<PathBuf> {
    if !dir.exists() { return vec![]; }
    if let Ok(entries) = fs::read_dir(dir) {
        entries.flatten().filter(|e| e.path().is_file()).map(|e| e.path()).collect()
    } else { vec![] }
}

fn check_basename_pairing(primary_dir: &Path, secondary_dir: &Path) -> PairingStatus {
    let primary_files   = list_files_in_dir(primary_dir);
    let secondary_files = list_files_in_dir(secondary_dir);
    let primaries:   std::collections::HashSet<String> = primary_files.iter().map(|p| get_basename(p)).collect();
    let secondaries: std::collections::HashSet<String> = secondary_files.iter().map(|p| get_basename(p)).collect();
    let paired_count        = primaries.intersection(&secondaries).count();
    let orphan_primaries:   Vec<String> = primaries.difference(&secondaries).take(20).cloned().collect();
    let orphan_secondaries: Vec<String> = secondaries.difference(&primaries).take(20).cloned().collect();
    PairingStatus {
        is_paired: orphan_primaries.is_empty() && orphan_secondaries.is_empty(),
        primary_count: primary_files.len(),
        paired_count, orphan_primaries, orphan_secondaries,
    }
}

pub fn detect_dataset_type(path: &Path) -> DatasetAnalysis {
    let mut warnings  = Vec::new();
    let dir_names     = list_subdir_names(path);
    let dir_names_lc: Vec<String> = dir_names.iter().map(|s| s.to_lowercase()).collect();
    let root_files    = list_files_in_dir(path);
    let root_exts: std::collections::HashSet<String> = root_files.iter()
        .filter_map(|f| f.extension().and_then(|e| e.to_str()).map(|s| s.to_lowercase()))
        .collect();
    let (_, total_file_count) = dir_size(path);
    let all_extensions = collect_extensions(path);

    // 1. Pre-Split
    let split_dirs: Vec<&str> = ["train","val","test","validation","training","testing"]
        .iter().filter(|&&s| dir_names_lc.contains(&s.to_string())).copied().collect();
    if split_dirs.len() >= 2 {
        return DatasetAnalysis { detected_type: DatasetType::PreSplit, confidence: 95,
            pairing_status: None, warnings: vec![], file_count: total_file_count,
            dir_count: dir_names.len(), extensions: all_extensions,
            schema_hint: Some(serde_json::json!({ "split_dirs": split_dirs })) };
    }

    // 2. YOLO / PascalVOC
    // Zuerst: existierende dataset.yaml / data.yaml auslesen wenn vorhanden
    let existing_yaml: Option<serde_json::Value> = [
        "dataset.yaml", "data.yaml", "yolov5.yaml", "yolov8.yaml"
    ].iter().find_map(|name| {
        let p = path.join(name);
        if !p.exists() { return None; }
        let content = fs::read_to_string(&p).ok()?;
        // Mini-YAML-Parser: nur Schluesselpfade die wir brauchen
        let mut map = serde_json::Map::new();
        for line in content.lines() {
            let line = line.trim();
            if line.starts_with('#') || line.is_empty() { continue; }
            if let Some((k, v)) = line.split_once(':') {
                let k = k.trim().to_string();
                let v = v.trim().trim_matches('"').to_string();
                if !v.is_empty() {
                    map.insert(k, serde_json::Value::String(v));
                }
            }
        }
        Some(serde_json::Value::Object(map))
    });

    let img_dir_name = dir_names.iter()
        .find(|d| matches!(d.to_lowercase().as_str(), "images"|"imgs"|"image")).cloned();
    let lbl_dir_name = dir_names.iter()
        .find(|d| matches!(d.to_lowercase().as_str(), "labels"|"label"|"annotations"|"annotation")).cloned();

    if let (Some(img_dir_name), Some(lbl_dir_name)) = (img_dir_name, lbl_dir_name) {
        let img_dir   = path.join(&img_dir_name);
        let lbl_dir   = path.join(&lbl_dir_name);
        let img_files = list_files_in_dir(&img_dir);
        let lbl_files = list_files_in_dir(&lbl_dir);
        let has_images = img_files.iter().any(|f| f.extension().and_then(|e| e.to_str()).map(|e| is_image(e)).unwrap_or(false));
        if has_images {
            let has_xml = lbl_files.iter().any(|f| f.extension().and_then(|e| e.to_str()).map(|e| e == "xml").unwrap_or(false));
            let pairing = check_basename_pairing(&img_dir, &lbl_dir);
            if !pairing.is_paired { warnings.push(format!("{} Bild(er) ohne Label.", pairing.orphan_primaries.len())); }
            let has_classes = path.join("classes.txt").exists() || path.join("obj.names").exists();
            if has_xml {
                return DatasetAnalysis { detected_type: DatasetType::PascalVoc, confidence: 90,
                    pairing_status: Some(pairing), warnings, file_count: total_file_count,
                    dir_count: dir_names.len(), extensions: all_extensions,
                    schema_hint: Some(serde_json::json!({ "images_dir": img_dir_name, "annotations_dir": lbl_dir_name })) };
            }
            return DatasetAnalysis { detected_type: DatasetType::YoloBbox,
                confidence: if has_classes { 97 } else { 88 },
                pairing_status: Some(pairing), warnings, file_count: total_file_count,
                dir_count: dir_names.len(), extensions: all_extensions,
                schema_hint: Some(serde_json::json!({
                    "images_dir": img_dir_name,
                    "labels_dir": lbl_dir_name,
                    "has_classes_file": has_classes,
                    "dataset_yaml": existing_yaml,
                    "dataset_yaml_path": path.join("dataset.yaml").to_string_lossy()
                })) };
        }
    }

    // 3. COCO JSON
    let ann_file = root_files.iter().find(|f| {
        let n = f.file_name().and_then(|n| n.to_str()).unwrap_or("").to_lowercase();
        n == "annotations.json" || (n.ends_with(".json") && n.contains("annotation")) || n.contains("instances_")
    });
    let has_img_dir = dir_names_lc.iter().any(|d| matches!(d.as_str(), "images"|"imgs"|"image"));
    if ann_file.is_some() && has_img_dir {
        let ann_name = ann_file.unwrap().file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
        let img_dir  = dir_names.iter().find(|d| matches!(d.to_lowercase().as_str(), "images"|"imgs"|"image")).cloned().unwrap_or_default();
        return DatasetAnalysis { detected_type: DatasetType::CocoJson, confidence: 95,
            pairing_status: None, warnings: vec![], file_count: total_file_count,
            dir_count: dir_names.len(), extensions: all_extensions,
            schema_hint: Some(serde_json::json!({ "annotations_file": ann_name, "images_dir": img_dir })) };
    }

    // 4. Folder Classification
    if !dir_names.is_empty() && root_files.is_empty() {
        let all_leaf = dir_names.iter().all(|d| list_subdir_names(&path.join(d)).is_empty());
        if all_leaf && dir_names.len() >= 2 {
            let sample_files = list_files_in_dir(&path.join(&dir_names[0]));
            let has_img = sample_files.iter().any(|f| f.extension().and_then(|e| e.to_str()).map(|e| is_image(e)).unwrap_or(false));
            let has_aud = sample_files.iter().any(|f| f.extension().and_then(|e| e.to_str()).map(|e| is_audio(e)).unwrap_or(false));
            if has_img || has_aud {
                let classes: Vec<&str> = dir_names.iter().map(String::as_str).take(10).collect();
                return DatasetAnalysis { detected_type: DatasetType::FolderClass, confidence: 92,
                    pairing_status: None, warnings: vec![], file_count: total_file_count,
                    dir_count: dir_names.len(), extensions: all_extensions,
                    schema_hint: Some(serde_json::json!({ "class_count": dir_names.len(), "classes": classes, "media_type": if has_img { "image" } else { "audio" } })) };
            }
        }
    }

    // 5. Common Voice
    if dir_names_lc.contains(&"clips".to_string()) {
        let tsv = root_files.iter().find(|f| {
            let n = f.file_name().and_then(|n| n.to_str()).unwrap_or("").to_lowercase();
            matches!(n.as_str(), "metadata.tsv"|"validated.tsv"|"train.tsv"|"test.tsv")
        });
        if let Some(tsv_file) = tsv {
            let tsv_name = tsv_file.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
            return DatasetAnalysis { detected_type: DatasetType::CommonVoice, confidence: 97,
                pairing_status: None, warnings: vec![], file_count: total_file_count,
                dir_count: dir_names.len(), extensions: all_extensions,
                schema_hint: Some(serde_json::json!({ "clips_dir": "clips", "metadata_file": tsv_name })) };
        }
    }

    // 6. Audio + Transcript
    let audio_files: Vec<_> = root_files.iter()
        .filter(|f| f.extension().and_then(|e| e.to_str()).map(|e| is_audio(e)).unwrap_or(false)).collect();
    let txt_files: Vec<_> = root_files.iter()
        .filter(|f| f.extension().and_then(|e| e.to_str()).map(|e| e == "txt").unwrap_or(false)).collect();
    if !audio_files.is_empty() && !txt_files.is_empty() {
        let audio_bns: std::collections::HashSet<String> = audio_files.iter().map(|f| get_basename(f)).collect();
        let txt_bns:   std::collections::HashSet<String> = txt_files.iter().map(|f| get_basename(f)).collect();
        let overlap = audio_bns.intersection(&txt_bns).count();
        if overlap > 0 {
            let orphan_audio: Vec<String> = audio_bns.difference(&txt_bns).take(20).cloned().collect();
            let orphan_txt:   Vec<String> = txt_bns.difference(&audio_bns).take(20).cloned().collect();
            if !orphan_audio.is_empty() { warnings.push(format!("{} Audio-Datei(en) ohne Transkript.", orphan_audio.len())); }
            let pairing = PairingStatus {
                is_paired: orphan_audio.is_empty() && orphan_txt.is_empty(),
                primary_count: audio_files.len(), paired_count: overlap,
                orphan_primaries: orphan_audio, orphan_secondaries: orphan_txt,
            };
            return DatasetAnalysis { detected_type: DatasetType::AudioTranscript, confidence: 93,
                pairing_status: Some(pairing), warnings, file_count: total_file_count,
                dir_count: dir_names.len(), extensions: all_extensions, schema_hint: None };
        }
    }

    // 7. Multi-Shard Parquet
    let parquet_files: Vec<_> = root_files.iter()
        .filter(|f| f.extension().and_then(|e| e.to_str()).map(|e| e == "parquet").unwrap_or(false)).collect();
    if parquet_files.len() > 1 {
        let is_sharded = parquet_files.iter().any(|f| {
            let n = f.file_name().and_then(|n| n.to_str()).unwrap_or("");
            n.contains("part-") || n.contains("-of-") || n.contains("train-") || n.contains("test-")
        });
        if is_sharded {
            return DatasetAnalysis { detected_type: DatasetType::MultiShard, confidence: 90,
                pairing_status: None, warnings: vec![], file_count: total_file_count,
                dir_count: dir_names.len(), extensions: all_extensions,
                schema_hint: Some(serde_json::json!({ "shard_count": parquet_files.len() })) };
        }
    }

    // 8. Flat File
    let flat_count = root_files.iter()
        .filter(|f| f.extension().and_then(|e| e.to_str()).map(|e| is_flat(e)).unwrap_or(false)).count();
    if flat_count > 0 {
        let all_flat = root_exts.iter().all(|e| is_flat(e));
        if !all_flat { warnings.push("Dataset enthaelt gemischte Dateitypen.".to_string()); }
        return DatasetAnalysis { detected_type: DatasetType::FlatFile,
            confidence: if all_flat { 92 } else { 70 },
            pairing_status: None, warnings, file_count: total_file_count,
            dir_count: dir_names.len(), extensions: all_extensions, schema_hint: None };
    }

    // 9. Unknown
    warnings.push("Dataset-Typ konnte nicht erkannt werden.".to_string());
    DatasetAnalysis { detected_type: DatasetType::Unknown, confidence: 0,
        pairing_status: None, warnings, file_count: total_file_count,
        dir_count: dir_names.len(), extensions: all_extensions, schema_hint: None }
}

// ══════════════════════════════════════════════════════════════════
// TYPE-AWARE SPLIT HELPERS
// ══════════════════════════════════════════════════════════════════

fn shuffle_indices(n: usize) -> Vec<usize> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut indices: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let mut h = DefaultHasher::new();
        i.hash(&mut h);
        let j = (h.finish() as usize) % (i + 1);
        indices.swap(i, j);
    }
    indices
}

fn split_counts(n: usize, train_r: f64, val_r: f64) -> (usize, usize, usize) {
    let train_n = (n as f64 * train_r).round() as usize;
    let val_n   = (n as f64 * val_r).round()   as usize;
    let test_n  = n.saturating_sub(train_n + val_n);
    (train_n, val_n, test_n)
}

fn move_or_copy(src: &Path, dst: &Path) -> Result<(), String> {
    if let Some(p) = dst.parent() { fs::create_dir_all(p).ok(); }
    if fs::rename(src, dst).is_err() {
        fs::copy(src, dst).map_err(|e| format!("copy {:?}: {}", src, e))?;
        fs::remove_file(src).ok();
    }
    Ok(())
}

// ══════════════════════════════════════════════════════════════════
// DATA.YAML GENERATOR
// ══════════════════════════════════════════════════════════════════

/// Liest vorhandene classes.txt / obj.names aus base und gibt die Liste zurueck.
fn read_class_names(base: &Path) -> Vec<String> {
    for candidate in &["classes.txt", "obj.names", "obj.data"] {
        let p = base.join(candidate);
        if p.exists() {
            if let Ok(content) = fs::read_to_string(&p) {
                let names: Vec<String> = content.lines()
                    .map(|l| l.trim().to_string())
                    .filter(|l| !l.is_empty() && !l.starts_with('#'))
                    // obj.data hat "classes = N" Zeilen – diese ueberspringen
                    .filter(|l| !l.contains('='))
                    .collect();
                if !names.is_empty() { return names; }
            }
        }
    }
    vec![]
}

/// Generiert eine dataset.yaml die YOLO / Ultralytics direkt lesen kann.
/// Wird nach Import und nach Split aufgerufen.
/// WICHTIG: Kein labels_dir-Feld -- Ultralytics leitet labels/ automatisch ab
/// indem es im train/val-Pfad 'images' durch 'labels' ersetzt.
pub fn generate_dataset_yaml(
    base: &Path,
    images_dir: &str,
    _labels_dir: &str,
    is_split: bool,
) -> Result<(), String> {
    let class_names = read_class_names(base);
    let nc = class_names.len();

    // names: als YAML-Liste mit Quotes -- Ultralytics-Standard
    let names_block = if nc > 0 {
        class_names.iter()
            .map(|n| format!("  - '{}'", n.replace('\'', "\\\'")))
            .collect::<Vec<_>>().join("\n")
    } else {
        "  # Klassen hier eintragen:\n  - 'KlasseA'\n  - 'KlasseB'".to_string()
    };

    let yaml = if is_split {
        format!(
            "# FrameTrain – dataset.yaml (Ultralytics/YOLO-kompatibel)\n\
             # Pfade sind relativ zum 'path'-Eintrag.\n\
             # Labels werden automatisch gesucht: 'images' im Pfad wird zu 'labels' ersetzt.\n\
             path: {}  # absoluter Pfad zum Dataset-Root\n\
             train: {}/train  # Trainings-Bilder\n\
             val:   {}/val    # Validierungs-Bilder\n\
             test:  {}/test   # Test-Bilder (optional)\n\
             \n\
             nc: {}\n\
             names:\n{}\n",
            base.display(),
            images_dir, images_dir, images_dir,
            nc, names_block
        )
    } else {
        format!(
            "# FrameTrain – dataset.yaml (Ultralytics/YOLO-kompatibel)\n\
             # Pfade sind relativ zum 'path'-Eintrag.\n\
             # Labels werden automatisch gesucht: 'images' im Pfad wird zu 'labels' ersetzt.\n\
             # Nach dem Split werden train/val/test automatisch eingetragen.\n\
             path: {}  # absoluter Pfad zum Dataset-Root\n\
             train: {}  # Trainings-Bilder (vor Split: Haupt-Bilder-Ordner)\n\
             val:   # nach Split verfuegbar\n\
             \n\
             nc: {}\n\
             names:\n{}\n",
            base.display(),
            images_dir,
            nc, names_block
        )
    };

    let yaml_path = base.join("dataset.yaml");
    fs::write(&yaml_path, &yaml).map_err(|e| format!("dataset.yaml schreiben: {}", e))?;
    eprintln!("[Dataset] \u{2713} dataset.yaml generiert: {:?}", yaml_path);
    Ok(())
}

/// Splittet zwei gepaarte Ordner (YOLO: images/+labels/, Pascal: images/+annotations/).
/// Garantie: Alle Dateien mit demselben Basename (z.B. img01.jpg + img01.txt)
/// landen IMMER im selben Split – auch bei mehreren Primaerdateien per Basename.
fn split_paired_dirs(base: &Path, primary_dir: &str, secondary_dir: &str,
                     train_r: f64, val_r: f64, test_r: f64) -> Result<SplitInfo, String> {
    let pdir = base.join(primary_dir);
    let sdir = base.join(secondary_dir);
    if !pdir.exists() { return Err(format!("Ordner '{}' nicht gefunden.", primary_dir)); }
    let primaries = list_files_in_dir(&pdir);
    let n = primaries.len();
    if n == 0 { return Err(format!("Keine Dateien in '{}'", primary_dir)); }

    // Sekundaerdateien nach Basename vorindexieren
    let sec_files = if sdir.exists() { list_files_in_dir(&sdir) } else { vec![] };
    let mut sec_map: std::collections::HashMap<String, Vec<PathBuf>> = std::collections::HashMap::new();
    for f in &sec_files {
        let bn = get_basename(f);
        sec_map.entry(bn).or_default().push(f.clone());
    }

    // PAIRING-GARANTIE: Nach Basename gruppieren damit beide immer in denselben Split kommen.
    // Primaerdateien mit gleichem Basename bilden eine Gruppe die atomar zugeteilt wird.
    let mut basename_groups: Vec<(String, Vec<PathBuf>)> = {
        let mut map: std::collections::HashMap<String, Vec<PathBuf>> = std::collections::HashMap::new();
        for f in &primaries {
            let bn = get_basename(f);
            map.entry(bn).or_default().push(f.clone());
        }
        let mut v: Vec<(String, Vec<PathBuf>)> = map.into_iter().collect();
        v.sort_by(|a, b| a.0.cmp(&b.0)); // deterministisch sortieren
        v
    };
    let num_groups = basename_groups.len();
    let indices = shuffle_indices(num_groups);
    let (train_n, val_n, test_n) = split_counts(num_groups, train_r, val_r);

    let splits = ["train", "val", "test"];
    for s in &splits {
        fs::create_dir_all(base.join(s).join(primary_dir)).ok();
        if sdir.exists() { fs::create_dir_all(base.join(s).join(secondary_dir)).ok(); }
    }

    let mut actual_train = 0usize;
    let mut actual_val   = 0usize;
    let mut actual_test  = 0usize;

    for (slot, &group_idx) in indices.iter().enumerate() {
        let split = if slot < train_n { "train" } else if slot < train_n + val_n { "val" } else { "test" };
        let (basename, pfiles) = &basename_groups[group_idx];

        // Alle Primaerdateien dieser Gruppe in denselben Split
        for pf in pfiles {
            move_or_copy(pf, &base.join(split).join(primary_dir).join(pf.file_name().unwrap_or_default()))?;
        }
        // Alle Sekundaerdateien dieser Gruppe in denselben Split
        if let Some(partners) = sec_map.get(basename) {
            for sf in partners {
                move_or_copy(sf, &base.join(split).join(secondary_dir).join(sf.file_name().unwrap_or_default()))?;
            }
        }

        match split { "train" => actual_train += 1, "val" => actual_val += 1, _ => actual_test += 1 }
    }

    // Hilfsdateien in alle Splits kopieren
    for extra in &["classes.txt", "obj.names", "obj.data", "data.yaml"] {
        let src = base.join(extra);
        if src.exists() { for s in &splits { fs::copy(&src, base.join(s).join(extra)).ok(); } }
    }
    // dataset.yaml nach Split neu generieren (Pfade zeigen jetzt auf train/val/test/images)
    generate_dataset_yaml(base, primary_dir, secondary_dir, true).ok();
    Ok(SplitInfo { train_count: actual_train, val_count: actual_val, test_count: actual_test,
                   train_ratio: train_r, val_ratio: val_r, test_ratio: test_r })
}

/// Splittet Audio+Transkript-Paare (gleicher Basename, verschiedene Extension).
fn split_audio_transcript(base: &Path, train_r: f64, val_r: f64, test_r: f64) -> Result<SplitInfo, String> {
    let audio_files: Vec<_> = list_files_in_dir(base).into_iter()
        .filter(|f| f.extension().and_then(|e| e.to_str()).map(|e| is_audio(e)).unwrap_or(false)).collect();
    let n = audio_files.len();
    if n == 0 { return Err("Keine Audio-Dateien gefunden.".to_string()); }
    let indices = shuffle_indices(n);
    let (train_n, val_n, test_n) = split_counts(n, train_r, val_r);
    for s in &["train", "val", "test"] { fs::create_dir_all(base.join(s)).ok(); }
    for (slot, &file_idx) in indices.iter().enumerate() {
        let split   = if slot < train_n { "train" } else if slot < train_n + val_n { "val" } else { "test" };
        let audio   = &audio_files[file_idx];
        let basename = audio.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        move_or_copy(audio, &base.join(split).join(audio.file_name().unwrap_or_default()))?;
        for ext in &["txt", "srt", "vtt"] {
            let txt = base.join(format!("{}.{}", basename, ext));
            if txt.exists() { move_or_copy(&txt, &base.join(split).join(format!("{}.{}", basename, ext)))?; }
        }
    }
    Ok(SplitInfo { train_count: train_n, val_count: val_n, test_count: test_n,
                   train_ratio: train_r, val_ratio: val_r, test_ratio: test_r })
}

/// Stratifizierter Split fuer Ordner-Klassenstruktur.
fn split_folder_class(base: &Path, train_r: f64, val_r: f64, test_r: f64) -> Result<SplitInfo, String> {
    let class_dirs = list_subdir_names(base);
    if class_dirs.is_empty() { return Err("Keine Klassen-Unterordner gefunden.".to_string()); }
    let (mut tt, mut tv, mut tx) = (0usize, 0usize, 0usize);
    for class in &class_dirs {
        if matches!(class.as_str(), "train"|"val"|"test"|"validation") { continue; }
        let files = list_files_in_dir(&base.join(class));
        let n = files.len(); if n == 0 { continue; }
        let indices = shuffle_indices(n);
        let (train_n, val_n, test_n) = split_counts(n, train_r, val_r);
        for s in &["train", "val", "test"] { fs::create_dir_all(base.join(s).join(class)).ok(); }
        for (slot, &file_idx) in indices.iter().enumerate() {
            let split = if slot < train_n { "train" } else if slot < train_n + val_n { "val" } else { "test" };
            let f = &files[file_idx];
            move_or_copy(f, &base.join(split).join(class).join(f.file_name().unwrap_or_default()))?;
        }
        tt += train_n; tv += val_n; tx += test_n;
    }
    if tt + tv + tx == 0 { return Err("Keine Dateien gefunden.".to_string()); }
    Ok(SplitInfo { train_count: tt, val_count: tv, test_count: tx,
                   train_ratio: train_r, val_ratio: val_r, test_ratio: test_r })
}

/// Flacher Split fuer Root-Dateien (FlatFile, MultiShard).
fn split_flat_files(base: &Path, train_r: f64, val_r: f64, test_r: f64) -> Result<SplitInfo, String> {
    let files = collect_files(base);
    let n = files.len();
    if n == 0 { return Err("Keine Dateien im Dataset-Root.".to_string()); }
    let indices = shuffle_indices(n);
    let (train_n, val_n, test_n) = split_counts(n, train_r, val_r);
    let train_dir = base.join("train"); let val_dir = base.join("val"); let test_dir = base.join("test");
    fs::create_dir_all(&train_dir).ok(); fs::create_dir_all(&val_dir).ok(); fs::create_dir_all(&test_dir).ok();
    for (slot, &file_idx) in indices.iter().enumerate() {
        let src = &files[file_idx];
        let fname = src.file_name().unwrap_or_default();
        let dst_dir = if slot < train_n { &train_dir } else if slot < train_n + val_n { &val_dir } else { &test_dir };
        move_or_copy(src, &dst_dir.join(fname))?;
    }
    Ok(SplitInfo { train_count: train_n, val_count: val_n, test_count: test_n,
                   train_ratio: train_r, val_ratio: val_r, test_ratio: test_r })
}

// ══════════════════════════════════════════════════════════════════
// TAURI COMMANDS
// ══════════════════════════════════════════════════════════════════

#[tauri::command]
pub async fn analyze_dataset_path(path: String) -> Result<DatasetAnalysis, String> {
    let p = Path::new(&path);
    if !p.exists() { return Err(format!("Pfad nicht gefunden: {}", path)); }
    if p.is_file() {
        let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        return Ok(DatasetAnalysis { detected_type: DatasetType::FlatFile, confidence: 85,
            pairing_status: None, warnings: vec![], file_count: 1, dir_count: 0,
            extensions: vec![format!(".{}", ext)], schema_hint: None });
    }
    Ok(detect_dataset_type(p))
}

#[tauri::command]
pub async fn list_datasets_for_model(
    app_handle: tauri::AppHandle, state: State<'_, AppState>, model_id: String,
) -> Result<Vec<DatasetInfo>, String> {
    let user_id = get_user_id(&state)?;
    let dir = get_datasets_dir(&app_handle, &user_id)?;
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
        }).collect();
    result.sort_by(|a, b| a.created_at.cmp(&b.created_at));
    if let Ok(db_path) = app_handle.path().app_data_dir().map(|p| p.join("frametrain.db")) {
        if let Ok(conn) = rusqlite::Connection::open(&db_path) {
            for ds in &mut result {
                ds.training_count = conn.query_row(
                    "SELECT COALESCE(training_count,0) FROM datasets WHERE id=?1", [&ds.id], |r| r.get(0)).unwrap_or(0);
                ds.last_used_at = conn.query_row(
                    "SELECT last_used_at FROM datasets WHERE id=?1", [&ds.id], |r| r.get(0)).unwrap_or(None);
            }
        }
    }
    Ok(result)
}

#[tauri::command]
pub async fn list_test_datasets_for_model(
    app_handle: tauri::AppHandle, state: State<'_, AppState>, model_id: String,
) -> Result<Vec<DatasetInfo>, String> {
    list_datasets_for_model(app_handle, state, model_id).await
}

#[tauri::command]
pub async fn list_all_datasets(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
) -> Result<Vec<DatasetInfo>, String> {
    let user_id = get_user_id(&state)?;
    let dir = get_datasets_dir(&app_handle, &user_id)?;
    Ok(load_metadata(&dir))
}

#[tauri::command]
pub async fn list_datasets(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
) -> Result<Vec<serde_json::Value>, String> {
    let user_id = get_user_id(&state)?;
    let dir = get_datasets_dir(&app_handle, &user_id)?;
    let datasets = load_metadata(&dir);
    Ok(datasets.into_iter().map(|d| {
        serde_json::json!({ "id": d.id, "name": d.name })
    }).collect())
}

#[tauri::command]
pub async fn import_local_dataset(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    source_path: String, dataset_name: String, model_id: String,
) -> Result<DatasetInfo, String> {
    let user_id = get_user_id(&state)?;
    let src = Path::new(&source_path);
    if !src.exists() { return Err(format!("Pfad nicht gefunden: {}", source_path)); }
    let analysis = if src.is_dir() { detect_dataset_type(src) } else {
        let ext = src.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        DatasetAnalysis { detected_type: DatasetType::FlatFile, confidence: 85,
            pairing_status: None, warnings: vec![], file_count: 1, dir_count: 0,
            extensions: vec![format!(".{}", ext)], schema_hint: None }
    };
    let dataset_id   = format!("ds_{}", &uuid::Uuid::new_v4().to_string().replace("-", "")[..12]);
    let datasets_dir = get_datasets_dir(&app_handle, &user_id)?;
    let target       = datasets_dir.join(&dataset_id);
    if src.is_dir() { copy_dir(src, &target)?; }
    else {
        fs::create_dir_all(&target).ok();
        fs::copy(src, target.join(src.file_name().unwrap())).map_err(|e| format!("Copy: {}", e))?;
    }
    let (size, files) = dir_size(&target);
    let info = make_info(&dataset_id, &dataset_name, &model_id, "local",
        Some(source_path), &target, size, files, "unused", None,
        analysis.detected_type.clone(), analysis.pairing_status, analysis.warnings);
    // dataset.yaml für YOLO/Pascal generieren falls noch nicht vorhanden
    if matches!(info.dataset_type, DatasetType::YoloBbox | DatasetType::PascalVoc) {
        let hint = &analysis.schema_hint;
        let img_dir = hint.as_ref().and_then(|s| s.get("images_dir")).and_then(|v| v.as_str()).unwrap_or("images");
        let lbl_dir = hint.as_ref()
            .and_then(|s| s.get("labels_dir").or_else(|| s.get("annotations_dir")))
            .and_then(|v| v.as_str()).unwrap_or("labels");
        generate_dataset_yaml(&target, img_dir, lbl_dir, false).ok();
        eprintln!("[Dataset] Import: dataset.yaml -> {:?}", target.join("dataset.yaml"));
    }
    upsert_metadata(&datasets_dir, &info)?;
    if let Ok(db_path) = app_handle.path().app_data_dir().map(|p| p.join("frametrain.db")) {
        if let Ok(conn) = rusqlite::Connection::open(&db_path) {
            let now = Utc::now().to_rfc3339();
            conn.execute("INSERT OR IGNORE INTO datasets (id,name,file_path,file_type,size_bytes,validated,user_id,created_at) VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
                rusqlite::params![&dataset_id, &info.name, target.to_string_lossy().to_string(), "local", size as i64, 0, &user_id, &now]).ok();
        }
    }
    Ok(info)
}

#[tauri::command]
pub async fn delete_dataset(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    dataset_id: String, model_id: String,
) -> Result<(), String> {
    let user_id      = get_user_id(&state)?;
    let datasets_dir = get_datasets_dir(&app_handle, &user_id)?;
    let target       = datasets_dir.join(&dataset_id);
    if target.exists() { fs::remove_dir_all(&target).map_err(|e| format!("Delete: {}", e))?; }
    let mut all = load_metadata(&datasets_dir);
    all.retain(|d| !(d.id == dataset_id && d.model_id == model_id));
    save_metadata(&datasets_dir, &all)?;
    if let Ok(db_path) = app_handle.path().app_data_dir().map(|p| p.join("frametrain.db")) {
        if let Ok(conn) = rusqlite::Connection::open(&db_path) {
            conn.execute("DELETE FROM datasets WHERE id=?1 AND user_id=?2",
                rusqlite::params![&dataset_id, &user_id]).ok();
        }
    }
    Ok(())
}

#[tauri::command]
pub async fn split_dataset(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    dataset_id: String, model_id: String,
    train_ratio: f64, val_ratio: f64, test_ratio: f64,
) -> Result<DatasetInfo, String> {
    let user_id      = get_user_id(&state)?;
    let datasets_dir = get_datasets_dir(&app_handle, &user_id)?;
    let base         = datasets_dir.join(&dataset_id);
    let mut all      = load_metadata(&datasets_dir);
    let ds           = all.iter().find(|d| d.id == dataset_id && d.model_id == model_id)
        .ok_or("Dataset nicht gefunden")?.clone();

    if ds.status == "split" { return Err("Dataset ist bereits aufgeteilt.".to_string()); }

    let split_info = match &ds.dataset_type {
        DatasetType::YoloBbox => {
            let re      = detect_dataset_type(&base);
            let img_dir = re.schema_hint.as_ref().and_then(|s| s.get("images_dir")).and_then(|v| v.as_str()).unwrap_or("images").to_string();
            let lbl_dir = re.schema_hint.as_ref().and_then(|s| s.get("labels_dir")).and_then(|v| v.as_str()).unwrap_or("labels").to_string();
            split_paired_dirs(&base, &img_dir, &lbl_dir, train_ratio, val_ratio, test_ratio)?
        }
        DatasetType::PascalVoc => {
            let re      = detect_dataset_type(&base);
            let img_dir = re.schema_hint.as_ref().and_then(|s| s.get("images_dir")).and_then(|v| v.as_str()).unwrap_or("images").to_string();
            let ann_dir = re.schema_hint.as_ref().and_then(|s| s.get("annotations_dir")).and_then(|v| v.as_str()).unwrap_or("annotations").to_string();
            split_paired_dirs(&base, &img_dir, &ann_dir, train_ratio, val_ratio, test_ratio)?
        }
        DatasetType::AudioTranscript => split_audio_transcript(&base, train_ratio, val_ratio, test_ratio)?,
        DatasetType::FolderClass     => split_folder_class(&base, train_ratio, val_ratio, test_ratio)?,
        // FIX Bug 3: Klarere Fehlermeldung mit konkreten Alternativen.
        DatasetType::CocoJson   => return Err("COCO JSON kann nicht automatisch gesplittet werden – die annotations.json referenziert Bilder per ID, ein Split wuerde diese Verknuepfungen zerreissen. Alternativen: (1) Dataset direkt ohne Split im Training nutzen, (2) manuell mit einem COCO-Split-Tool aufteilen, oder (3) zuerst ins YOLO-Format konvertieren.".to_string()),
        DatasetType::PreSplit   => return Err("Dieses Dataset ist bereits aufgeteilt.".to_string()),
        DatasetType::CommonVoice => return Err("Common Voice hat eigene Splits (metadata.tsv). Kein automatischer Split noetig.".to_string()),
        _ => split_flat_files(&base, train_ratio, val_ratio, test_ratio)?,
    };

    let (size, fc) = dir_size(&base);
    let updated = DatasetInfo { status: "split".to_string(), split_info: Some(split_info),
        storage_path: base.to_string_lossy().to_string(),
        extensions: collect_extensions(&base), size_bytes: size, file_count: fc, ..ds };
    all.retain(|d| d.id != dataset_id);
    all.push(updated.clone());
    save_metadata(&datasets_dir, &all)?;
    Ok(updated)
}

#[tauri::command]
pub async fn split_dataset_in_half(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    dataset_id: String, model_id: String,
) -> Result<serde_json::Value, String> {
    let user_id      = get_user_id(&state)?;
    let datasets_dir = get_datasets_dir(&app_handle, &user_id)?;
    let base         = datasets_dir.join(&dataset_id);
    let all          = load_metadata(&datasets_dir);
    let ds           = all.iter().find(|d| d.id == dataset_id && d.model_id == model_id)
        .ok_or("Dataset nicht gefunden")?.clone();

    let id_a = format!("ds_{}", &uuid::Uuid::new_v4().to_string().replace("-", "")[..12]);
    let id_b = format!("ds_{}", &uuid::Uuid::new_v4().to_string().replace("-", "")[..12]);
    let dir_a = datasets_dir.join(&id_a);
    let dir_b = datasets_dir.join(&id_b);
    fs::create_dir_all(&dir_a).ok();
    fs::create_dir_all(&dir_b).ok();

    match &ds.dataset_type {
        DatasetType::YoloBbox | DatasetType::PascalVoc => {
            let re = detect_dataset_type(&base);
            let primary_dir = re.schema_hint.as_ref()
                .and_then(|s| s.get("images_dir")).and_then(|v| v.as_str()).unwrap_or("images").to_string();
            let secondary_dir = if matches!(ds.dataset_type, DatasetType::PascalVoc) {
                re.schema_hint.as_ref().and_then(|s| s.get("annotations_dir")).and_then(|v| v.as_str()).unwrap_or("annotations").to_string()
            } else {
                re.schema_hint.as_ref().and_then(|s| s.get("labels_dir")).and_then(|v| v.as_str()).unwrap_or("labels").to_string()
            };
            let pdir  = base.join(&primary_dir);
            let sdir  = base.join(&secondary_dir);
            let files = list_files_in_dir(&pdir);
            let n     = files.len();
            if n == 0 { return Err(format!("Keine Bilder in '{}'", primary_dir)); }
            let half  = n / 2;
            let sec_files = if sdir.exists() { list_files_in_dir(&sdir) } else { vec![] };
            let mut sec_map: std::collections::HashMap<String, Vec<PathBuf>> = std::collections::HashMap::new();
            for f in &sec_files {
                let bn = f.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
                sec_map.entry(bn).or_default().push(f.clone());
            }
            for (i, pf) in files.iter().enumerate() {
                let dst_base = if i < half { &dir_a } else { &dir_b };
                let basename = pf.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
                let dst_p = dst_base.join(&primary_dir).join(pf.file_name().unwrap_or_default());
                fs::create_dir_all(dst_p.parent().unwrap()).ok();
                fs::copy(pf, &dst_p).ok();
                if let Some(partners) = sec_map.get(&basename) {
                    for sf in partners {
                        let dst_s = dst_base.join(&secondary_dir).join(sf.file_name().unwrap_or_default());
                        fs::create_dir_all(dst_s.parent().unwrap()).ok();
                        fs::copy(sf, &dst_s).ok();
                    }
                }
            }
        }
        DatasetType::AudioTranscript => {
            let audio_files: Vec<_> = list_files_in_dir(&base).into_iter()
                .filter(|f| f.extension().and_then(|e| e.to_str()).map(|e| is_audio(e)).unwrap_or(false)).collect();
            let half = audio_files.len() / 2;
            for (i, af) in audio_files.iter().enumerate() {
                let dst_base = if i < half { &dir_a } else { &dir_b };
                let basename = af.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                fs::copy(af, dst_base.join(af.file_name().unwrap_or_default())).ok();
                for ext in &["txt", "srt", "vtt"] {
                    let txt = base.join(format!("{}.{}", basename, ext));
                    if txt.exists() { fs::copy(&txt, dst_base.join(format!("{}.{}", basename, ext))).ok(); }
                }
            }
        }
        DatasetType::FolderClass => {
            for class in list_subdir_names(&base) {
                if matches!(class.as_str(), "train"|"val"|"test") { continue; }
                let files = list_files_in_dir(&base.join(&class));
                let half  = files.len() / 2;
                fs::create_dir_all(dir_a.join(&class)).ok();
                fs::create_dir_all(dir_b.join(&class)).ok();
                for (i, f) in files.iter().enumerate() {
                    let dst = if i < half { dir_a.join(&class) } else { dir_b.join(&class) };
                    fs::copy(f, dst.join(f.file_name().unwrap_or_default())).ok();
                }
            }
        }
        _ => {
            // FIX Bug 4: collect_files_recursive statt collect_files damit MultiShard-Shards
            // in Unterordnern (z.B. train/part-0.parquet) vollstaendig erfasst werden.
            // Relative Pfadstruktur bleibt erhalten.
            let files = collect_files_recursive(&base);
            if files.is_empty() { return Err("Keine Dateien im Dataset.".to_string()); }
            let half = files.len() / 2;
            for (i, f) in files.iter().enumerate() {
                let dst_root = if i < half { &dir_a } else { &dir_b };
                let rel = f.strip_prefix(&base).unwrap_or(f.as_path());
                let dst = dst_root.join(rel);
                if let Some(p) = dst.parent() { fs::create_dir_all(p).ok(); }
                fs::copy(f, &dst).ok();
            }
        }
    }

    let (sa, fa) = dir_size(&dir_a);
    let (sb, fb) = dir_size(&dir_b);
    let name_a = format!("{} (Haelfte 1)", ds.name);
    let name_b = format!("{} (Haelfte 2)", ds.name);
    let ds_a = make_info(&id_a, &name_a, &model_id, "local", None, &dir_a, sa, fa, "unused", None, ds.dataset_type.clone(), None, vec![]);
    let ds_b = make_info(&id_b, &name_b, &model_id, "local", None, &dir_b, sb, fb, "unused", None, ds.dataset_type, None, vec![]);

    let mut all = load_metadata(&datasets_dir);
    all.push(ds_a.clone());
    all.push(ds_b.clone());
    save_metadata(&datasets_dir, &all)?;
    Ok(serde_json::json!({ "dataset_a": ds_a, "dataset_b": ds_b }))
}

#[tauri::command]
pub async fn get_dataset_files(
    app_handle: tauri::AppHandle, state: State<'_, AppState>, dataset_id: String,
) -> Result<Vec<serde_json::Value>, String> {
    let user_id      = get_user_id(&state)?;
    let datasets_dir = get_datasets_dir(&app_handle, &user_id)?;
    let dataset_dir  = datasets_dir.join(&dataset_id);
    if !dataset_dir.exists() { return Ok(vec![]); }
    let mut files: Vec<serde_json::Value> = Vec::new();
    for split in &["train", "val", "test"] {
        let split_dir = dataset_dir.join(split);
        if split_dir.exists() {
            for file in collect_files_recursive(&split_dir) {
                if let Ok(meta) = fs::metadata(&file) {
                    files.push(serde_json::json!({ "name": file.file_name().unwrap_or_default().to_string_lossy(), "path": file.to_string_lossy(), "size": meta.len(), "is_dir": false, "split": split }));
                }
            }
        }
    }
    // Unterordner wie images/ und labels/ ebenfalls listen (mit split="subdir")
    let known_subdirs = ["images", "labels", "annotations", "imgs", "clips"];
    for subdir in &known_subdirs {
        let subdir_path = dataset_dir.join(subdir);
        if subdir_path.exists() {
            for file in list_files_in_dir(&subdir_path) {
                if let Ok(meta) = fs::metadata(&file) {
                    files.push(serde_json::json!({ "name": file.file_name().unwrap_or_default().to_string_lossy(), "path": file.to_string_lossy(), "size": meta.len(), "is_dir": false, "split": subdir }));
                }
            }
        }
    }
    if let Ok(entries) = fs::read_dir(&dataset_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if matches!(file_name, "train"|"val"|"test"|"unused"|"images"|"labels"|"clips") { continue; }
            if path.is_file() {
                if let Ok(meta) = fs::metadata(&path) {
                    let tag = if matches!(file_name, "dataset_infos.json"|"metadata.json"|".gitkeep"|".DS_Store") { "info" } else { "unsplit" };
                    files.push(serde_json::json!({ "name": file_name, "path": path.to_string_lossy(), "size": meta.len(), "is_dir": false, "split": tag }));
                }
            }
        }
    }
    Ok(files)
}

#[tauri::command]
pub async fn move_dataset_files(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    dataset_id: String, file_paths: Vec<String>, target_split: String,
) -> Result<(), String> {
    let user_id      = get_user_id(&state)?;
    let datasets_dir = get_datasets_dir(&app_handle, &user_id)?;
    let target_dir   = datasets_dir.join(&dataset_id).join(&target_split);
    fs::create_dir_all(&target_dir).map_err(|e| format!("mkdir: {}", e))?;
    for fp in &file_paths {
        let src = Path::new(fp);
        if src.exists() && src.is_file() {
            let dst = target_dir.join(src.file_name().unwrap_or_default());
            if fs::rename(src, &dst).is_err() {
                fs::copy(src, &dst).map_err(|e| format!("Copy: {}", e))?;
                fs::remove_file(src).map_err(|e| format!("Delete: {}", e))?;
            }
        }
    }
    Ok(())
}

#[tauri::command]
pub async fn add_files_to_dataset(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    dataset_id: String, file_paths: Vec<String>,
) -> Result<serde_json::Value, String> {
    let user_id = get_user_id(&state)?;
    let dst = get_datasets_dir(&app_handle, &user_id)?.join(&dataset_id);
    fs::create_dir_all(&dst).ok();
    let mut added = 0usize;
    for fp in &file_paths {
        let src = Path::new(fp);
        if src.exists() {
            fs::copy(src, dst.join(src.file_name().unwrap_or_default())).map_err(|e| format!("Copy: {}", e))?;
            added += 1;
        }
    }
    Ok(serde_json::json!({ "added": added }))
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
pub async fn read_dataset_file(file_path: String) -> Result<String, String> {
    let path = Path::new(&file_path);
    if !path.exists() { return Err(format!("Datei nicht gefunden: {}", file_path)); }
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    match ext.as_str() {
        "txt"|"json"|"jsonl"|"csv"|"tsv"|"md"|"log"|"xml"|"yaml"|"yml"|"conll"|"bio"|"iob" => {
            let content = fs::read_to_string(path).map_err(|e| format!("Lesen: {}", e))?;
            let lines: Vec<&str> = content.lines().collect();
            let preview = lines.iter().take(200).cloned().collect::<Vec<_>>().join("\n");
            if lines.len() > 200 { return Ok(format!("{}\n\n--- [Vorschau: 200 von {} Zeilen] ---", preview, lines.len())); }
            Ok(preview)
        }
        "jpg"|"jpeg"|"png"|"bmp"|"webp"|"gif"|"tiff" => {
            let size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            Ok(format!("[Bild] {}.{} -- {} bytes", path.file_stem().unwrap_or_default().to_string_lossy(), ext, size))
        }
        "wav"|"mp3"|"flac"|"ogg"|"m4a" => {
            let size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            Ok(format!("[Audio] {}.{} -- {} bytes", path.file_stem().unwrap_or_default().to_string_lossy(), ext, size))
        }
        "parquet" => {
            let size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            Ok(format!("[Parquet] {} bytes -- Binaerformat, kein Preview.", size))
        }
        _ => {
            let size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            Ok(format!("[{}.{}] {} bytes -- kein Preview.", path.file_stem().unwrap_or_default().to_string_lossy(), ext, size))
        }
    }
}

// ── HuggingFace ────────────────────────────────────────────────────────────

#[tauri::command]
pub async fn search_huggingface_datasets(
    query: String, limit: Option<u32>,
    filter_task: Option<String>, filter_language: Option<String>, _filter_size: Option<String>,
) -> Result<Vec<HuggingFaceDataset>, String> {
    let limit = limit.unwrap_or(15);
    let mut url = format!("https://huggingface.co/api/datasets?search={}&limit={}&sort=downloads&direction=-1",
        urlencoding::encode(&query), limit);
    if let Some(t) = &filter_task     { url.push_str(&format!("&pipeline_tag={}", urlencoding::encode(t))); }
    if let Some(l) = &filter_language { url.push_str(&format!("&language={}", urlencoding::encode(l))); }
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
            arr.iter().filter_map(|s| s.as_str()).map(String::from).collect()),
    })).collect();
    Ok(datasets)
}

#[tauri::command]
pub async fn get_dataset_filter_options() -> Result<serde_json::Value, String> {
    Ok(serde_json::json!({
        "tasks": ["text-classification","token-classification","question-answering","summarization","translation","text-generation","fill-mask","image-classification","object-detection","automatic-speech-recognition"],
        "languages": ["de","en","fr","es","it","zh","ja","pt","ru","ar"],
        "sizes": ["n<1K","1K<n<10K","10K<n<100K","100K<n<1M","n>1M"]
    }))
}

#[derive(Debug, Deserialize)]
struct HfParquetFile { url: String, filename: String, size: Option<u64>, split: Option<String> }

#[tauri::command]
pub async fn download_huggingface_dataset(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    repo_id: String, dataset_name: String, model_id: String,
) -> Result<DatasetInfo, String> {
    let user_id      = get_user_id(&state)?;
    let datasets_dir = get_datasets_dir(&app_handle, &user_id)?;
    let dataset_id   = format!("ds_{}", &uuid::Uuid::new_v4().to_string().replace("-", "")[..12]);
    let target       = datasets_dir.join(&dataset_id);
    fs::create_dir_all(&target).map_err(|e| format!("mkdir: {}", e))?;
    let _ = app_handle.emit("dataset-download-progress", DatasetDownloadProgress {
        status: "connecting".to_string(), current_file: String::new(),
        current_file_index: 0, total_files: 0, downloaded_bytes: 0, total_bytes: 0,
        progress_percent: 0, speed_mbs: 0.0, elapsed_secs: 0, eta_secs: 0,
        message: "Verbinde mit Hugging Face...".to_string() });
    let client = reqwest::Client::builder().timeout(std::time::Duration::from_secs(60)).build()
        .map_err(|e| format!("HTTP client: {}", e))?;
    let api_url = format!("https://datasets-server.huggingface.co/parquet?dataset={}", urlencoding::encode(&repo_id));
    if let Ok(resp) = client.get(&api_url).header("User-Agent", "FrameTrain-Desktop/1.0").send().await {
        if resp.status().is_success() {
            if let Ok(api_json) = resp.json::<serde_json::Value>().await {
                let parquet_files: Vec<HfParquetFile> = api_json.get("parquet_files")
                    .and_then(|f| f.as_array())
                    .map(|arr| arr.iter().filter_map(|v| serde_json::from_value(v.clone()).ok()).collect())
                    .unwrap_or_default();
                if !parquet_files.is_empty() {
                    return download_parquet_direct(app_handle, client, parquet_files, target,
                        dataset_id, dataset_name, model_id, datasets_dir, repo_id, user_id).await;
                }
            }
        }
    }
    download_via_python(app_handle, repo_id, target, dataset_id, dataset_name, model_id, datasets_dir, user_id).await
}

async fn download_parquet_direct(
    app_handle: tauri::AppHandle, client: reqwest::Client, files: Vec<HfParquetFile>,
    target: PathBuf, dataset_id: String, dataset_name: String, model_id: String,
    datasets_dir: PathBuf, repo_id: String, user_id: String,
) -> Result<DatasetInfo, String> {
    use std::time::Instant;
    let total_files  = files.len();
    let total_bytes: u64 = files.iter().filter_map(|f| f.size).sum();
    let t0 = Instant::now();
    let mut global_dl: u64 = 0;
    for (file_idx, hf_file) in files.iter().enumerate() {
        let fname = if hf_file.filename.is_empty() { format!("file_{}.parquet", file_idx) } else { hf_file.filename.clone() };
        let response = client.get(&hf_file.url).header("User-Agent", "FrameTrain-Desktop/1.0").send().await
            .map_err(|e| format!("HTTP GET '{}': {}", fname, e))?;
        if !response.status().is_success() { return Err(format!("HTTP {} fuer '{}'", response.status(), fname)); }
        let file_total = hf_file.size.or_else(|| response.content_length()).unwrap_or(0);
        let mut out_file = tokio::fs::File::create(target.join(&fname)).await
            .map_err(|e| format!("Erstellen '{}': {}", fname, e))?;
        let mut file_dl: u64 = 0;
        let mut stream = response.bytes_stream();
        let mut last_emit = Instant::now();
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| format!("Chunk: {}", e))?;
            out_file.write_all(&chunk).await.map_err(|e| format!("Schreiben: {}", e))?;
            file_dl += chunk.len() as u64; global_dl += chunk.len() as u64;
            if last_emit.elapsed().as_millis() >= 33 {
                last_emit = Instant::now();
                let elapsed = t0.elapsed().as_secs();
                let pct = if total_bytes > 0 { ((global_dl as f64 / total_bytes as f64) * 100.0) as i32 }
                    else { let fp = if file_total > 0 { (file_dl as f64 / file_total as f64) * 100.0 } else { 0.0 };
                           (((file_idx as f64 + fp / 100.0) / total_files as f64) * 100.0) as i32 };
                let speed = if elapsed > 0 { (global_dl as f32 / 1_048_576.0) / elapsed as f32 } else { 0.0 };
                let eta   = if speed > 0.0 && total_bytes > global_dl { ((total_bytes - global_dl) as f32 / (speed * 1_048_576.0)) as u64 } else { 0 };
                let _ = app_handle.emit("dataset-download-progress", DatasetDownloadProgress {
                    status: "downloading".to_string(),
                    current_file: format!("{} ({})", fname, hf_file.split.as_deref().unwrap_or("?")),
                    current_file_index: file_idx + 1, total_files,
                    downloaded_bytes: global_dl, total_bytes,
                    progress_percent: pct.clamp(0, 99),
                    speed_mbs: speed, elapsed_secs: elapsed, eta_secs: eta,
                    message: format!("Datei {}/{}: {}", file_idx + 1, total_files, fname) });
            }
        }
        drop(out_file);
    }
    let elapsed = t0.elapsed().as_secs();
    let (total_size, file_count) = dir_size(&target);
    let speed = if elapsed > 0 { (total_size as f32 / 1_048_576.0) / elapsed as f32 } else { 0.0 };
    let _ = app_handle.emit("dataset-download-progress", DatasetDownloadProgress {
        status: "complete".to_string(), current_file: String::new(),
        current_file_index: file_count, total_files: file_count,
        downloaded_bytes: total_size, total_bytes: total_size, progress_percent: 100,
        speed_mbs: speed, elapsed_secs: elapsed, eta_secs: 0,
        message: format!("Fertig! ({} Dateien, {:.1} MB)", file_count, total_size as f64 / 1_048_576.0) });
    // FIX Bug 2: Typ nach Download neu erkennen statt immer MultiShard zu setzen.
    let detected = detect_dataset_type(&target);
    let info = make_info(&dataset_id, &dataset_name, &model_id, "huggingface",
        Some(repo_id), &target, total_size, file_count, "unused", None,
        detected.detected_type, detected.pairing_status, detected.warnings);
    upsert_metadata(&datasets_dir, &info)?;
    if let Ok(db_path) = app_handle.path().app_data_dir().map(|p| p.join("frametrain.db")) {
        if let Ok(conn) = rusqlite::Connection::open(&db_path) {
            let now = Utc::now().to_rfc3339();
            conn.execute("INSERT OR IGNORE INTO datasets (id,name,file_path,file_type,size_bytes,validated,user_id,created_at) VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
                rusqlite::params![&dataset_id, &dataset_name, target.to_string_lossy().to_string(), "huggingface", total_size as i64, 0, &user_id, &now]).ok();
        }
    }
    Ok(info)
}

async fn download_via_python(
    app_handle: tauri::AppHandle, repo_id: String, target: PathBuf,
    dataset_id: String, dataset_name: String, model_id: String,
    datasets_dir: PathBuf, user_id: String,
) -> Result<DatasetInfo, String> {
    use std::time::Instant;
    use std::process::{Command, Stdio};
    let _ = app_handle.emit("dataset-download-progress", DatasetDownloadProgress {
        status: "preparing".to_string(), current_file: String::new(),
        current_file_index: 0, total_files: 0, downloaded_bytes: 0, total_bytes: 0,
        progress_percent: 0, speed_mbs: 0.0, elapsed_secs: 0, eta_secs: 0,
        message: format!("Lade '{}' via Python datasets...", repo_id) });
    let python_script = r#"
import sys, json
from datasets import load_dataset, get_dataset_config_names
from pathlib import Path
repo_id = sys.argv[1]; target = Path(sys.argv[2]); target.mkdir(parents=True, exist_ok=True)
def emit(obj): print(json.dumps(obj), flush=True)
try:
    dataset = None
    emit({"type": "status", "message": f"Lade '{repo_id}'..."})
    try: dataset = load_dataset(repo_id)
    except Exception as e:
        msg = str(e)
        if "Config name is missing" in msg or "Please pick one among" in msg:
            configs = get_dataset_config_names(repo_id)
            if not configs: raise Exception(f"Keine Configs fuer '{repo_id}' gefunden.")
            emit({"type": "status", "message": f"Verwende Config '{configs[0]}'"})
            dataset = load_dataset(repo_id, configs[0])
        else: raise
    splits = dataset if isinstance(dataset, dict) else {"default": dataset}
    total_splits = len(splits)
    emit({"type": "status", "message": f"Speichere {total_splits} Split(s)..."})
    fc = 0; ts = 0
    for sn, sd in splits.items():
        emit({"type": "status", "message": f"Schreibe '{sn}' ({len(sd)} Zeilen)..."})
        out = target / f"{sn}.parquet"; sd.to_parquet(str(out))
        size = out.stat().st_size; ts += size; fc += 1
        emit({"type": "file_done", "split": sn, "size": size, "total_files": total_splits, "file_index": fc})
    emit({"type": "complete", "files": fc, "total_size": ts})
except Exception as e:
    print(json.dumps({"type": "error", "message": str(e)}), file=sys.stderr, flush=True); sys.exit(1)
"#;
    let script_file = std::env::temp_dir().join("hf_dataset_download.py");
    fs::write(&script_file, python_script).map_err(|e| format!("Script: {}", e))?;
    let python_cmd = if Command::new("python3").arg("--version").output().is_ok() { "python3" }
        else if Command::new("python").arg("--version").output().is_ok() { "python" }
        else { return Err("Python nicht gefunden".to_string()); };
    let mut child = Command::new(python_cmd)
        .arg(&script_file).arg(&repo_id).arg(target.to_string_lossy().to_string())
        .stdout(Stdio::piped()).stderr(Stdio::piped())
        .spawn().map_err(|e| format!("Python spawn: {}", e))?;
    let stdout_pipe = child.stdout.take().expect("stdout");
    let stderr_pipe = child.stderr.take().expect("stderr");
    let stderr_handle = std::thread::spawn(move || {
        let mut buf = String::new(); BufReader::new(stderr_pipe).read_to_string(&mut buf).ok(); buf
    });
    let t0 = Instant::now();
    let mut count = 0usize; let mut total_written = 0u64; let mut total_splits_known = 0usize;
    let app_clone = app_handle.clone();
    let _: Result<(), String> = tokio::task::spawn_blocking(move || {
        for line in BufReader::new(stdout_pipe).lines().flatten() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                let elapsed = t0.elapsed().as_secs();
                match v.get("type").and_then(|t| t.as_str()).unwrap_or("") {
                    "status" => {
                        let msg = v.get("message").and_then(|m| m.as_str()).unwrap_or("").to_string();
                        let _ = app_clone.emit("dataset-download-progress", DatasetDownloadProgress {
                            status: "preparing".to_string(), current_file: String::new(),
                            current_file_index: 0, total_files: total_splits_known,
                            downloaded_bytes: 0, total_bytes: 0, progress_percent: 0,
                            speed_mbs: 0.0, elapsed_secs: elapsed, eta_secs: 0, message: msg });
                    }
                    "file_done" => {
                        if let Some(size) = v.get("size").and_then(|s| s.as_u64()) {
                            let split = v.get("split").and_then(|s| s.as_str()).unwrap_or("split").to_string();
                            total_splits_known = v.get("total_files").and_then(|t| t.as_u64()).unwrap_or(0) as usize;
                            let fi = v.get("file_index").and_then(|i| i.as_u64()).unwrap_or(1) as usize;
                            let _count = count; total_written += size;
                            let pct   = if total_splits_known > 0 { ((fi as f32 / total_splits_known as f32) * 100.0) as i32 } else { 0 };
                            let speed = if elapsed > 0 { (total_written as f32 / 1_048_576.0) / elapsed as f32 } else { 0.0 };
                            let _ = app_clone.emit("dataset-download-progress", DatasetDownloadProgress {
                                status: "downloading".to_string(),
                                current_file: format!("{}.parquet", split),
                                current_file_index: fi, total_files: total_splits_known,
                                downloaded_bytes: total_written, total_bytes: total_written,
                                progress_percent: pct, speed_mbs: speed, elapsed_secs: elapsed, eta_secs: 0,
                                message: format!("{} ok ({:.1} MB)", split, size as f64 / 1_048_576.0) });
                        }
                    }
                    "complete" => {
                        if let (Some(files), Some(size)) = (v.get("files").and_then(|f| f.as_u64()), v.get("total_size").and_then(|s| s.as_u64())) {
                            let speed = if elapsed > 0 { (size as f32 / 1_048_576.0) / elapsed as f32 } else { 0.0 };
                            let _ = app_clone.emit("dataset-download-progress", DatasetDownloadProgress {
                                status: "complete".to_string(), current_file: String::new(),
                                current_file_index: files as usize, total_files: files as usize,
                                downloaded_bytes: size, total_bytes: size, progress_percent: 100,
                                speed_mbs: speed, elapsed_secs: elapsed, eta_secs: 0,
                                message: format!("Fertig! ({} Splits, {:.1} MB)", files, size as f64 / 1_048_576.0) });
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }).await.map_err(|e| format!("spawn_blocking: {}", e))?;
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
        return Err(format!("Download fehlgeschlagen: {}", user_error));
    }
    if count == 0 {
        if let Ok(entries) = fs::read_dir(&target) {
            for entry in entries.flatten() {
                if let Ok(meta) = entry.metadata() {
                    if meta.is_file() { count += 1; total_written += meta.len(); }
                }
            }
        }
    }
    if count == 0 { fs::remove_dir_all(&target).ok(); return Err(format!("Keine Dateien von '{}' heruntergeladen.", repo_id)); }
    // FIX Bug 2: Typ nach Python-Download neu erkennen.
    let detected = detect_dataset_type(&target);
    let info = make_info(&dataset_id, &dataset_name, &model_id, "huggingface",
        Some(repo_id), &target, total_written, count, "unused", None,
        detected.detected_type, detected.pairing_status, detected.warnings);
    upsert_metadata(&datasets_dir, &info)?;
    if let Ok(db_path) = app_handle.path().app_data_dir().map(|p| p.join("frametrain.db")) {
        if let Ok(conn) = rusqlite::Connection::open(&db_path) {
            let now = Utc::now().to_rfc3339();
            conn.execute("INSERT OR IGNORE INTO datasets (id,name,file_path,file_type,size_bytes,validated,user_id,created_at) VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
                rusqlite::params![&dataset_id, &dataset_name, target.to_string_lossy().to_string(), "huggingface", total_written as i64, 0, &user_id, &now]).ok();
        }
    }
    Ok(info)
}

#[tauri::command]
pub async fn validate_image_label_folders(path: String) -> Result<serde_json::Value, String> {
    let p = Path::new(&path);
    if !p.is_dir() { return Ok(serde_json::json!({ "valid": false, "error": "Kein Ordner" })); }
    let analysis = detect_dataset_type(p);
    Ok(serde_json::json!({
        "valid": !matches!(analysis.detected_type, DatasetType::Unknown),
        "detected_type": analysis.detected_type.as_str(),
        "confidence": analysis.confidence,
        "warnings": analysis.warnings,
    }))
}

#[tauri::command]
pub async fn import_structured_dataset(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    source_path: String, dataset_name: String, model_id: String,
) -> Result<DatasetInfo, String> {
    import_local_dataset(app_handle, state, source_path, dataset_name, model_id).await
}

/// Resolve dataset storage path for Synapse / training env vars.
#[tauri::command]
pub async fn get_dataset_path(
    app_handle: tauri::AppHandle,
    state: State<'_, AppState>,
    dataset_id: String,
) -> Result<String, String> {
    if dataset_id.is_empty() {
        return Err("Keine Dataset-ID angegeben".to_string());
    }
    let user_id = get_user_id(&state)?;
    let dir = get_datasets_dir(&app_handle, &user_id)?;
    let datasets = load_metadata(&dir);
    if let Some(d) = datasets.iter().find(|d| d.id == dataset_id) {
        if !d.storage_path.is_empty() {
            let p = PathBuf::from(&d.storage_path);
            if p.exists() {
                return Ok(p.to_string_lossy().to_string());
            }
        }
    }
    let db_path = app_handle.path().app_data_dir()
        .map_err(|e| format!("AppDataDir: {}", e))?
        .join("frametrain.db");
    if db_path.exists() {
        if let Ok(conn) = rusqlite::Connection::open(&db_path) {
            let res: Result<String, _> = conn.query_row(
                "SELECT file_path FROM datasets WHERE id = ?1",
                [&dataset_id],
                |r| r.get(0),
            );
            if let Ok(p) = res {
                if !p.is_empty() && Path::new(&p).exists() {
                    return Ok(p);
                }
            }
        }
    }
    let fallback = dir.join(&dataset_id);
    if fallback.exists() {
        return Ok(fallback.to_string_lossy().to_string());
    }
    Err(format!("Dataset-Pfad nicht gefunden: {}", dataset_id))
}

// ════════════════════════════════════════════════════════════════
// DATASET.YAML EDITOR COMMANDS
// ════════════════════════════════════════════════════════════════

/// Liefert den Inhalt der dataset.yaml eines Datasets (falls vorhanden).
/// Wird vom Frontend-Editor geladen.
#[tauri::command]
pub async fn get_dataset_yaml(
    app_handle: tauri::AppHandle,
    state: State<'_, AppState>,
    dataset_id: String,
) -> Result<serde_json::Value, String> {
    let user_id = get_user_id(&state)?;
    let dir     = get_datasets_dir(&app_handle, &user_id)?;
    let ds_dir  = dir.join(&dataset_id);
    let yaml_path = ds_dir.join("dataset.yaml");
    if !yaml_path.exists() {
        return Ok(serde_json::json!({ "exists": false }));
    }
    let raw = fs::read_to_string(&yaml_path).map_err(|e| format!("Lesen: {}", e))?;
    // Mini-Parser: relevante Felder extrahieren
    let mut train_path  = String::new();
    let mut val_path    = String::new();
    let mut nc: usize   = 0;
    let mut names: Vec<String> = Vec::new();
    let mut in_names    = false;
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') { continue; }
        if in_names {
            if trimmed.starts_with("- ") {
                let name = trimmed.trim_start_matches("- ").trim().trim_matches('\'').trim_matches('"').to_string();
                if !name.is_empty() { names.push(name); }
                continue;
            } else if trimmed.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                // Altes Format: "0: cat" -- Wert nach dem ersten ':' extrahieren
                if let Some((_, v)) = trimmed.split_once(':') {
                    let name = v.trim().trim_matches('\'').trim_matches('"').to_string();
                    if !name.is_empty() { names.push(name); }
                }
                continue;
            } else if !trimmed.is_empty() && !trimmed.starts_with('-') {
                in_names = false; // Naechstes Feld beginnt
            } else { continue; }
        }
        if let Some((k, v)) = trimmed.split_once(':') {
            let k = k.trim(); let v = v.trim().trim_matches('"').trim_matches('\'').to_string();
            match k {
                "train" => { if !v.is_empty() && v != "#" { train_path = v; } }
                "val"   => { if !v.is_empty() && v != "#" { val_path   = v; } }
                "nc"    => { nc = v.parse().unwrap_or(0); }
                "names" => { in_names = true; }
                _ => {}
            }
        }
    }
    // Falls nc nicht gesetzt war, aus names ableiten
    if nc == 0 && !names.is_empty() { nc = names.len(); }
    Ok(serde_json::json!({
        "exists":     true,
        "train_path": train_path,
        "val_path":   val_path,
        "nc":         nc,
        "names":      names,
        "raw":        raw,
        "yaml_path":  yaml_path.to_string_lossy(),
    }))
}

/// Schreibt eine editierte dataset.yaml zurueck auf Disk.
/// Generiert das korrekte Ultralytics-Format.
#[tauri::command]
pub async fn save_dataset_yaml(
    app_handle: tauri::AppHandle,
    state: State<'_, AppState>,
    dataset_id: String,
    train_path: String,
    val_path: String,
    names: Vec<String>,
) -> Result<String, String> {
    let user_id  = get_user_id(&state)?;
    let dir      = get_datasets_dir(&app_handle, &user_id)?;
    let ds_dir   = dir.join(&dataset_id);
    if !ds_dir.exists() { return Err("Dataset-Ordner nicht gefunden".to_string()); }
    let nc = names.len();
    let names_block: String = names.iter()
        .map(|n| format!("  - '{}'", n.replace('\'', "\\'")))
        .collect::<Vec<_>>().join("\n");
    // Pfad = absoluter Dataset-Root
    let yaml = format!(
        "# FrameTrain \u{2013} dataset.yaml (Ultralytics-kompatibel)\n\
         # Pfade sind relativ zum 'path'-Eintrag unten.\n\
         path: {}  # absoluter Pfad zum Dataset-Root\n\
         train: {}  # relativer Pfad zum Trainings-Bilder-Ordner\n\
         val:   {}  # relativer Pfad zum Validierungs-Bilder-Ordner\n\
         \n\
         nc: {}\n\
         names:\n{}\n",
        ds_dir.display(),
        train_path.trim(),
        val_path.trim(),
        nc,
        if names_block.is_empty() { "  # Noch keine Klassen eingetragen".to_string() } else { names_block },
    );
    let yaml_path = ds_dir.join("dataset.yaml");
    fs::write(&yaml_path, &yaml).map_err(|e| format!("Schreiben: {}", e))?;
    eprintln!("[Dataset] dataset.yaml gespeichert: {:?}", yaml_path);
    Ok(yaml_path.to_string_lossy().to_string())
}