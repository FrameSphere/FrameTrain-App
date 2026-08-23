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

/// IDs dürfen nur aus [A-Za-z0-9_-] bestehen — verhindert Path-Traversal.
fn is_safe_id(id: &str) -> bool {
    crate::model_manager::is_safe_id(id)
}

/// Prüft, dass `path` innerhalb von `base` liegt (nach Kanonisierung).
/// Schutz gegen absichtlich/versehentlich übergebene fremde Pfade.
fn is_within_dir(path: &Path, base: &Path) -> bool {
    let (Ok(canon_path), Ok(canon_base)) = (path.canonicalize(), base.canonicalize()) else {
        return false;
    };
    canon_path.starts_with(&canon_base)
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

/// Ein YOLO/VOC-Dataset, das bereits in train/val/test aufgeteilt auf der
/// Platte liegt. Ultralytics kennt zwei Schreibweisen, beide sind hier gemeint:
///   Nested:  images/train + labels/train
///   Grouped: train/images + train/labels
#[derive(Debug, Clone)]
pub struct SplitLayout {
    /// "nested" oder "grouped"
    pub style:      &'static str,
    pub images_dir: String,
    pub labels_dir: String,
    /// Nur Splits, die tatsaechlich Bilder enthalten: (kanonischer Name, Ordnername, Bildanzahl)
    pub splits:     Vec<(String, String, usize)>,
    pub has_xml:    bool,
}

impl SplitLayout {
    /// Relativer Pfad zum Bilderordner eines Splits, wie ihn die dataset.yaml braucht.
    pub fn images_path_for(&self, dir_name: &str) -> String {
        if self.style == "nested" { format!("{}/{}", self.images_dir, dir_name) }
        else                      { format!("{}/{}", dir_name, self.images_dir) }
    }
    pub fn count_of(&self, canonical: &str) -> usize {
        self.splits.iter().find(|(c, _, _)| c == canonical).map(|(_, _, n)| *n).unwrap_or(0)
    }
}

/// Ordnernamen wie "training"/"validation" auf train/val/test normalisieren.
fn canonical_split_name(name: &str) -> Option<&'static str> {
    match name.to_lowercase().as_str() {
        "train" | "training"                 => Some("train"),
        "val" | "valid" | "validation" | "dev" => Some("val"),
        "test" | "testing"                   => Some("test"),
        _ => None,
    }
}

fn count_images_in(dir: &Path) -> usize {
    list_files_in_dir(dir).iter()
        .filter(|f| f.extension().and_then(|e| e.to_str()).map(is_image).unwrap_or(false))
        .count()
}

fn dir_has_xml(dir: &Path) -> bool {
    list_files_in_dir(dir).iter()
        .any(|f| f.extension().and_then(|e| e.to_str()).map(|e| e.eq_ignore_ascii_case("xml")).unwrap_or(false))
}

/// Erkennt ein bereits aufgeteiltes Bild+Label-Dataset.
///
/// Vorher fiel genau dieses Layout durch alle Zweige und landete bei
/// "Unbekannt, 0 % Konfidenz": `list_files_in_dir("images")` findet keine
/// Dateien, weil dort nur die Split-Unterordner liegen.
pub fn detect_split_layout(path: &Path) -> Option<SplitLayout> {
    let dir_names = list_subdir_names(path);
    let find_dir = |cands: &[&str]| -> Option<String> {
        dir_names.iter().find(|d| cands.contains(&d.to_lowercase().as_str())).cloned()
    };
    const IMG_DIRS: &[&str] = &["images", "imgs", "image"];
    const LBL_DIRS: &[&str] = &["labels", "label", "annotations", "annotation"];

    // Nested: images/<split> + labels/<split>
    if let (Some(img_dir), Some(lbl_dir)) = (find_dir(IMG_DIRS), find_dir(LBL_DIRS)) {
        let img_base = path.join(&img_dir);
        let lbl_base = path.join(&lbl_dir);
        let mut splits  = Vec::new();
        let mut has_xml = false;
        for sub in list_subdir_names(&img_base) {
            let Some(canon) = canonical_split_name(&sub) else { continue };
            let n = count_images_in(&img_base.join(&sub));
            if n == 0 { continue; }
            // Der passende Label-Ordner muss existieren, sonst ist es kein Paar.
            let lbl_sub = list_subdir_names(&lbl_base).into_iter()
                .find(|d| canonical_split_name(d) == Some(canon))?;
            has_xml |= dir_has_xml(&lbl_base.join(&lbl_sub));
            splits.push((canon.to_string(), sub, n));
        }
        if !splits.is_empty() {
            splits.sort_by_key(|(c, _, _)| match c.as_str() { "train" => 0, "val" => 1, _ => 2 });
            return Some(SplitLayout { style: "nested", images_dir: img_dir, labels_dir: lbl_dir, splits, has_xml });
        }
    }

    // Grouped: <split>/images + <split>/labels
    let mut splits  = Vec::new();
    let mut has_xml = false;
    let mut img_name = String::new();
    let mut lbl_name = String::new();
    for dir in &dir_names {
        let Some(canon) = canonical_split_name(dir) else { continue };
        let split_root = path.join(dir);
        let subs = list_subdir_names(&split_root);
        let Some(img_sub) = subs.iter().find(|d| IMG_DIRS.contains(&d.to_lowercase().as_str())) else { continue };
        let Some(lbl_sub) = subs.iter().find(|d| LBL_DIRS.contains(&d.to_lowercase().as_str())) else { continue };
        let n = count_images_in(&split_root.join(img_sub));
        if n == 0 { continue; }
        has_xml |= dir_has_xml(&split_root.join(lbl_sub));
        img_name = img_sub.clone();
        lbl_name = lbl_sub.clone();
        splits.push((canon.to_string(), dir.clone(), n));
    }
    if !splits.is_empty() {
        splits.sort_by_key(|(c, _, _)| match c.as_str() { "train" => 0, "val" => 1, _ => 2 });
        return Some(SplitLayout { style: "grouped", images_dir: img_name, labels_dir: lbl_name, splits, has_xml });
    }
    None
}

/// Prueft Bild/Label-Paarung ueber alle Splits hinweg.
fn pairing_across_splits(path: &Path, layout: &SplitLayout) -> PairingStatus {
    let mut total = PairingStatus { is_paired: true, ..Default::default() };
    for (_, dir_name, _) in &layout.splits {
        let (img_dir, lbl_dir) = if layout.style == "nested" {
            (path.join(&layout.images_dir).join(dir_name), path.join(&layout.labels_dir).join(dir_name))
        } else {
            (path.join(dir_name).join(&layout.images_dir), path.join(dir_name).join(&layout.labels_dir))
        };
        let p = check_basename_pairing(&img_dir, &lbl_dir);
        total.primary_count += p.primary_count;
        total.paired_count  += p.paired_count;
        total.is_paired     &= p.is_paired;
        for o in p.orphan_primaries   { if total.orphan_primaries.len()   < 20 { total.orphan_primaries.push(o); } }
        for o in p.orphan_secondaries { if total.orphan_secondaries.len() < 20 { total.orphan_secondaries.push(o); } }
    }
    total
}

/// Liest eine vorhandene data.yaml/dataset.yaml als flache Key-Value-Map.
fn read_existing_dataset_yaml(path: &Path) -> Option<serde_json::Value> {
    ["dataset.yaml", "data.yaml", "yolov5.yaml", "yolov8.yaml"].iter().find_map(|name| {
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
    })
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

    // 0. Bereits aufgeteiltes YOLO / Pascal VOC (images/train + labels/train o. train/images ...).
    //    Muss vor dem Pre-Split-Zweig stehen: sonst wird "train/images" nur als
    //    generischer Pre-Split gemeldet und die Bild/Label-Paarung geht verloren.
    if let Some(layout) = detect_split_layout(path) {
        let pairing = pairing_across_splits(path, &layout);
        if !pairing.is_paired {
            warnings.push(format!("{} Bild(er) ohne Label.", pairing.orphan_primaries.len()));
        }
        if layout.count_of("val") == 0 {
            warnings.push("Kein Validierungs-Split gefunden – nur Training.".to_string());
        }
        let splits_json: serde_json::Map<String, serde_json::Value> = layout.splits.iter()
            .map(|(canon, dir, n)| (canon.clone(), serde_json::json!({ "dir": dir, "count": n })))
            .collect();
        return DatasetAnalysis {
            detected_type: if layout.has_xml { DatasetType::PascalVoc } else { DatasetType::YoloBbox },
            confidence: 95,
            pairing_status: Some(pairing), warnings, file_count: total_file_count,
            dir_count: dir_names.len(), extensions: all_extensions,
            schema_hint: Some(serde_json::json!({
                "images_dir":   layout.images_dir,
                "labels_dir":   layout.labels_dir,
                "is_split":     true,
                "split_style":  layout.style,
                "splits":       splits_json,
                "dataset_yaml": read_existing_dataset_yaml(path),
            })),
        };
    }

    // 1. Pre-Split
    let split_dirs: Vec<&str> = ["train","val","valid","test","validation","training","testing"]
        .iter().filter(|&&s| dir_names_lc.contains(&s.to_string())).copied().collect();
    if split_dirs.len() >= 2 {
        return DatasetAnalysis { detected_type: DatasetType::PreSplit, confidence: 95,
            pairing_status: None, warnings: vec![], file_count: total_file_count,
            dir_count: dir_names.len(), extensions: all_extensions,
            schema_hint: Some(serde_json::json!({ "split_dirs": split_dirs })) };
    }

    // 2. YOLO / PascalVOC
    // Zuerst: existierende dataset.yaml / data.yaml auslesen wenn vorhanden
    let existing_yaml: Option<serde_json::Value> = read_existing_dataset_yaml(path);

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
    // Viele YOLO-Datensaetze bringen keine classes.txt mit, sondern nur eine
    // data.yaml mit dem names-Block. Ohne diesen Zweig landeten die
    // Platzhalter 'KlasseA'/'KlasseB' in der generierten dataset.yaml.
    for candidate in &["dataset.yaml", "data.yaml", "yolov5.yaml", "yolov8.yaml", "data.yml"] {
        let p = base.join(candidate);
        if !p.exists() { continue; }
        if let Ok(content) = fs::read_to_string(&p) {
            let names = parse_yaml_class_names(&content);
            if !names.is_empty() { return names; }
        }
    }
    vec![]
}

/// Liest den `names:`-Block einer Ultralytics-data.yaml.
/// Unterstuetzt alle drei gebraeuchlichen Schreibweisen:
///   names: ['a', 'b']        (Inline-Liste)
///   names:\n  - 'a'\n  - 'b' (Block-Liste)
///   names:\n  0: a\n  1: b   (Index-Map, YOLOv8-Stil)
fn parse_yaml_class_names(content: &str) -> Vec<String> {
    let unquote = |s: &str| -> String {
        s.trim().trim_matches(|c| c == '"' || c == '\'').trim().to_string()
    };
    let mut lines = content.lines().peekable();
    while let Some(line) = lines.next() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') { continue; }
        let Some(rest) = trimmed.strip_prefix("names:") else { continue };
        // Kommentar am Zeilenende abschneiden
        let rest = rest.split('#').next().unwrap_or("").trim();

        // Fall 1: Inline-Liste
        if let Some(inner) = rest.strip_prefix('[').and_then(|r| r.strip_suffix(']')) {
            let names: Vec<String> = inner.split(',')
                .map(unquote).filter(|s| !s.is_empty()).collect();
            if !names.is_empty() { return names; }
            continue;
        }
        if !rest.is_empty() { continue; }

        // Fall 2/3: Folgezeilen einsammeln, bis ein neuer Top-Level-Key kommt.
        let mut indexed: Vec<(usize, String)> = Vec::new();
        let mut listed:  Vec<String> = Vec::new();
        while let Some(next) = lines.peek() {
            let raw = *next;
            let t = raw.trim();
            if t.is_empty() || t.starts_with('#') { lines.next(); continue; }
            // Ein Eintrag muss eingerueckt sein, sonst beginnt ein neuer Key.
            let indented = raw.starts_with(' ') || raw.starts_with('\t');
            if !indented { break; }
            let t = t.split(" #").next().unwrap_or(t).trim();
            if let Some(item) = t.strip_prefix("- ") {
                listed.push(unquote(item));
            } else if let Some((k, v)) = t.split_once(':') {
                if let Ok(idx) = k.trim().parse::<usize>() {
                    indexed.push((idx, unquote(v)));
                } else { break; }
            } else { break; }
            lines.next();
        }
        if !listed.is_empty() { return listed.into_iter().filter(|s| !s.is_empty()).collect(); }
        if !indexed.is_empty() {
            indexed.sort_by_key(|(i, _)| *i);
            return indexed.into_iter().map(|(_, n)| n).filter(|s| !s.is_empty()).collect();
        }
    }
    vec![]
}

/// Schreibt die dataset.yaml fuer ein bereits aufgeteiltes Dataset.
///
/// Im Gegensatz zu `generate_dataset_yaml` werden nur Splits eingetragen, die
/// wirklich Bilder enthalten – ein leerer images/test-Ordner (der in vielen
/// Datensaetzen herumliegt) laesst Ultralytics sonst mit
/// "No images found" abbrechen.
pub fn generate_split_dataset_yaml(base: &Path, layout: &SplitLayout) -> Result<(), String> {
    let class_names = read_class_names(base);
    let nc = class_names.len();
    let names_block = if nc > 0 {
        class_names.iter()
            .map(|n| format!("  - '{}'", n.replace('\'', "\\\'")))
            .collect::<Vec<_>>().join("\n")
    } else {
        "  # Klassen hier eintragen:\n  - 'KlasseA'\n  - 'KlasseB'".to_string()
    };

    let mut split_lines = String::new();
    for (canon, dir_name, count) in &layout.splits {
        split_lines.push_str(&format!(
            "{}: {}  # {} Bilder\n", canon, layout.images_path_for(dir_name), count
        ));
    }
    // Ultralytics braucht einen val-Eintrag; ohne eigenen Val-Split auf train zeigen.
    if layout.count_of("val") == 0 {
        if let Some((_, train_dir, _)) = layout.splits.iter().find(|(c, _, _)| c == "train") {
            split_lines.push_str(&format!(
                "val: {}  # kein eigener Val-Split vorhanden\n", layout.images_path_for(train_dir)
            ));
        }
    }

    let yaml = format!(
        "# FrameTrain \u{2013} dataset.yaml (Ultralytics/YOLO-kompatibel)\n\
         # Dataset war beim Import bereits aufgeteilt, die Splits wurden uebernommen.\n\
         # Labels werden automatisch gesucht: 'images' im Pfad wird zu 'labels' ersetzt.\n\
         path: {}  # absoluter Pfad zum Dataset-Root\n\
         {}\n\
         nc: {}\n\
         names:\n{}\n",
        base.display(), split_lines, nc, names_block
    );
    fs::write(base.join("dataset.yaml"), &yaml).map_err(|e| format!("Schreiben: {}", e))?;
    Ok(())
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

/// Erkennt ob eine Datei zeilenweise/Sample-weise gesplittet werden kann
/// (strukturiertes Format mit mehreren Datensaetzen pro Datei), statt als
/// atomare Einheit verschoben zu werden.
fn is_row_splittable(ext: &str) -> bool {
    matches!(ext.to_lowercase().as_str(), "parquet" | "csv" | "tsv" | "jsonl" | "json")
}

/// Der Interpreter der App — derselbe wie fuer Training, Tests und Labor.
///
/// Frueher suchte diese Datei an drei Stellen selbst nach "python3". Unter
/// Windows heisst der Interpreter aber "python"; dort scheiterten Parquet-Split,
/// Parquet-Vorschau und HuggingFace-Download. python_env::resolve_python kennt
/// beide Namen und bevorzugt ausserdem ein torch-faehiges Python.
fn find_python_cmd() -> Result<String, String> {
    let py = crate::python_env::resolve_python();
    if std::process::Command::new(&py).arg("--version").output().is_ok() {
        return Ok(py);
    }
    Err("Python nicht gefunden -- wird für das Splitten strukturierter Dateien (Parquet/CSV/JSON) benötigt.".to_string())
}

/// Splittet eine einzelne strukturierte Datei (Parquet/CSV/TSV/JSONL/JSON) intern
/// nach Zeilen/Samples, NICHT nach ganzen Dateien. Schema/Header bleibt in jeder
/// Split-Datei erhalten (z.B. CSV-Header wird in train/val/test-Datei dupliziert).
/// Schreibt train.<ext>, val.<ext>, test.<ext> (val/test nur wenn > 0 Zeilen) in den
/// jeweiligen Split-Ordner und gibt (train_n, val_n, test_n) Zeilenanzahl zurueck.
/// Bei zu wenigen Zeilen fuer eine sinnvolle Aufteilung wird gewarnt statt zu splitten.
fn split_row_file(
    src: &Path, train_dir: &Path, val_dir: &Path, test_dir: &Path,
    train_r: f64, val_r: f64, _test_r: f64,
    warnings: &mut Vec<String>,
) -> Result<(usize, usize, usize), String> {
    let ext  = src.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    let stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or("data");
    fs::create_dir_all(train_dir).ok();
    fs::create_dir_all(val_dir).ok();
    fs::create_dir_all(test_dir).ok();

    // Namenskollisions-Schutz: falls eine Datei mit gleichem Namen im Zielordner
    // schon existiert (z.B. zwei Shards heissen beide 0000.parquet nach stem-Extraktion),
    // wird ein Suffix angehaengt: 0000_1.parquet, 0000_2.parquet, ...
    let unique_name = |dir: &Path, base: &str, extension: &str| -> String {
        let candidate = format!("{}.{}", base, extension);
        if !dir.join(&candidate).exists() { return candidate; }
        let mut i = 1usize;
        loop {
            let c = format!("{}_{}.{}", base, i, extension);
            if !dir.join(&c).exists() { return c; }
            i += 1;
        }
    };

    let train_out = train_dir.join(unique_name(train_dir, stem, &ext));
    let val_out   = val_dir.join(unique_name(val_dir, stem, &ext));
    let test_out  = test_dir.join(unique_name(test_dir, stem, &ext));

    match ext.as_str() {
        "jsonl" => {
            let content = fs::read_to_string(src).map_err(|e| format!("Lesen '{}': {}", src.display(), e))?;
            let lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
            let n = lines.len();
            if n < 3 {
                warnings.push(format!("'{}' hat nur {} Zeile(n) -- zu wenig für einen sinnvollen Split, komplett in train übernommen.", src.file_name().unwrap_or_default().to_string_lossy(), n));
                fs::write(&train_out, &content).map_err(|e| format!("Schreiben: {}", e))?;
                return Ok((n, 0, 0));
            }
            let indices = shuffle_indices(n);
            let (tn, vn, _tt) = split_counts(n, train_r, val_r);
            let mut train_lines = Vec::new(); let mut val_lines = Vec::new(); let mut test_lines = Vec::new();
            for (slot, &idx) in indices.iter().enumerate() {
                if slot < tn { train_lines.push(lines[idx]); }
                else if slot < tn + vn { val_lines.push(lines[idx]); }
                else { test_lines.push(lines[idx]); }
            }
            fs::write(&train_out, train_lines.join("\n") + "\n").map_err(|e| format!("Schreiben train: {}", e))?;
            if !val_lines.is_empty()  { fs::write(&val_out,  val_lines.join("\n") + "\n").map_err(|e| format!("Schreiben val: {}", e))?; }
            if !test_lines.is_empty() { fs::write(&test_out, test_lines.join("\n") + "\n").map_err(|e| format!("Schreiben test: {}", e))?; }
            Ok((train_lines.len(), val_lines.len(), test_lines.len()))
        }
        "csv" | "tsv" => {
            let content = fs::read_to_string(src).map_err(|e| format!("Lesen '{}': {}", src.display(), e))?;
            let mut all_lines = content.lines();
            let header = all_lines.next().ok_or_else(|| format!("'{}' ist leer.", src.display()))?.to_string();
            let data_lines: Vec<&str> = all_lines.filter(|l| !l.trim().is_empty()).collect();
            let n = data_lines.len();
            if n < 3 {
                warnings.push(format!("'{}' hat nur {} Datenzeile(n) -- zu wenig für einen sinnvollen Split, komplett in train übernommen.", src.file_name().unwrap_or_default().to_string_lossy(), n));
                fs::write(&train_out, &content).map_err(|e| format!("Schreiben: {}", e))?;
                return Ok((n, 0, 0));
            }
            let indices = shuffle_indices(n);
            let (tn, vn, _tt) = split_counts(n, train_r, val_r);
            let mut train_lines = vec![header.clone()];
            let mut val_lines   = vec![header.clone()];
            let mut test_lines  = vec![header.clone()];
            for (slot, &idx) in indices.iter().enumerate() {
                if slot < tn { train_lines.push(data_lines[idx].to_string()); }
                else if slot < tn + vn { val_lines.push(data_lines[idx].to_string()); }
                else { test_lines.push(data_lines[idx].to_string()); }
            }
            // Header zaehlt nicht als Datenzeile -- Split-Dateien immer mit Header schreiben,
            // damit Spaltennamen (CSV-Schema) in jedem Split erhalten bleiben.
            fs::write(&train_out, train_lines.join("\n") + "\n").map_err(|e| format!("Schreiben train: {}", e))?;
            let val_n  = val_lines.len().saturating_sub(1);
            let test_n = test_lines.len().saturating_sub(1);
            if val_n  > 0 { fs::write(&val_out,  val_lines.join("\n") + "\n").map_err(|e| format!("Schreiben val: {}", e))?; }
            if test_n > 0 { fs::write(&test_out, test_lines.join("\n") + "\n").map_err(|e| format!("Schreiben test: {}", e))?; }
            Ok((train_lines.len() - 1, val_n, test_n))
        }
        "json" => {
            let content = fs::read_to_string(src).map_err(|e| format!("Lesen '{}': {}", src.display(), e))?;
            let parsed: serde_json::Value = serde_json::from_str(&content).map_err(|e| format!("JSON-Parse '{}': {}", src.display(), e))?;
            let arr = parsed.as_array().ok_or_else(|| format!("'{}' ist kein JSON-Array von Samples -- Zeilen-Split nicht moeglich.", src.display()))?;
            let n = arr.len();
            if n < 3 {
                warnings.push(format!("'{}' hat nur {} Einträge -- zu wenig für einen sinnvollen Split, komplett in train übernommen.", src.file_name().unwrap_or_default().to_string_lossy(), n));
                fs::write(&train_out, &content).map_err(|e| format!("Schreiben: {}", e))?;
                return Ok((n, 0, 0));
            }
            let indices = shuffle_indices(n);
            let (tn, vn, _tt) = split_counts(n, train_r, val_r);
            let mut train_items = Vec::new(); let mut val_items = Vec::new(); let mut test_items = Vec::new();
            for (slot, &idx) in indices.iter().enumerate() {
                if slot < tn { train_items.push(arr[idx].clone()); }
                else if slot < tn + vn { val_items.push(arr[idx].clone()); }
                else { test_items.push(arr[idx].clone()); }
            }
            fs::write(&train_out, serde_json::to_string_pretty(&train_items).unwrap()).map_err(|e| format!("Schreiben train: {}", e))?;
            if !val_items.is_empty()  { fs::write(&val_out,  serde_json::to_string_pretty(&val_items).unwrap()).map_err(|e| format!("Schreiben val: {}", e))?; }
            if !test_items.is_empty() { fs::write(&test_out, serde_json::to_string_pretty(&test_items).unwrap()).map_err(|e| format!("Schreiben test: {}", e))?; }
            Ok((train_items.len(), val_items.len(), test_items.len()))
        }
        "parquet" => split_parquet_file(src, &train_out, &val_out, &test_out, train_r, val_r, warnings),
        _ => Err(format!("Format '.{}' kann nicht zeilenweise gesplittet werden.", ext)),
    }
}

/// Splittet eine Parquet-Datei zeilenweise via Python (pandas), da Parquet ein
/// binaeres Spaltenformat ist und nicht ohne pyarrow/pandas geparst werden kann.
/// Mischt Zeilen, schreibt train/val/test-Parquet-Dateien mit identischem Schema.
fn split_parquet_file(
    src: &Path, train_out: &Path, val_out: &Path, test_out: &Path,
    train_r: f64, val_r: f64, warnings: &mut Vec<String>,
) -> Result<(usize, usize, usize), String> {
    use std::process::Command;
    let python = find_python_cmd()?;
    let script = format!(r#"
import sys, json
import pandas as pd
import numpy as np

src       = sys.argv[1]
train_out = sys.argv[2]
val_out   = sys.argv[3]
test_out  = sys.argv[4]
train_r   = float(sys.argv[5])
val_r     = float(sys.argv[6])

df = pd.read_parquet(src)
n = len(df)
if n < 3:
    df.to_parquet(train_out)
    print(json.dumps({{"train": n, "val": 0, "test": 0, "too_small": True}}))
    sys.exit(0)

rng = np.random.default_rng(42)
idx = rng.permutation(n)
train_n = round(n * train_r)
val_n   = round(n * val_r)

train_idx = idx[:train_n]
if train_r + val_r >= 0.999:
    # Kein Test-Split vorgesehen (z.B. Halbieren 0.5/0.5): Rest komplett zu val.
    # Vorher konnte durch Banker's Rounding (round(2.5)=2) bei ungerader
    # Zeilenzahl eine Restzeile im test-Output landen, der beim Halbieren
    # verworfen wurde -- 1 Zeile Datenverlust pro Datei.
    val_idx  = idx[train_n:]
    test_idx = idx[:0]
else:
    val_idx   = idx[train_n:train_n+val_n]
    test_idx  = idx[train_n+val_n:]

df.iloc[train_idx].to_parquet(train_out)
val_count = 0
test_count = 0
if len(val_idx) > 0:
    df.iloc[val_idx].to_parquet(val_out)
    val_count = len(val_idx)
if len(test_idx) > 0:
    df.iloc[test_idx].to_parquet(test_out)
    test_count = len(test_idx)

print(json.dumps({{"train": len(train_idx), "val": val_count, "test": test_count, "too_small": False}}))
"#);
    let output = Command::new(python)
        .arg("-c").arg(&script)
        .arg(src.to_string_lossy().to_string())
        .arg(train_out.to_string_lossy().to_string())
        .arg(val_out.to_string_lossy().to_string())
        .arg(test_out.to_string_lossy().to_string())
        .arg(train_r.to_string())
        .arg(val_r.to_string())
        .output()
        .map_err(|e| format!("Python spawn: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Parquet-Split fehlgeschlagen fuer '{}': {}", src.display(), stderr.trim()));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let result: serde_json::Value = serde_json::from_str(stdout.trim())
        .map_err(|e| format!("JSON-Parse Parquet-Split-Ergebnis: {} (raw: {})", e, stdout.trim()))?;
    let train_n = result.get("train").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let val_n   = result.get("val").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let test_n  = result.get("test").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    if result.get("too_small").and_then(|v| v.as_bool()).unwrap_or(false) {
        warnings.push(format!("'{}' hat nur {} Zeile(n) -- zu wenig für einen sinnvollen Split, komplett in train übernommen.", src.file_name().unwrap_or_default().to_string_lossy(), train_n));
    }
    Ok((train_n, val_n, test_n))
}

/// Flacher Split fuer Root-Dateien (FlatFile, MultiShard).
///
/// WICHTIG: Strukturierte Dateien (Parquet/CSV/TSV/JSONL/JSON) werden NICHT als
/// atomare Einheit verschoben, sondern zeilenweise/Sample-weise intern gesplittet --
/// jede Datei landet anteilig (train_r/val_r/test_r) in train/val/test, mit Header/Schema
/// in jeder resultierenden Datei erhalten. Nur unbekannte/binaere Formate (die nicht
/// row-splittable sind) werden weiterhin als ganze Datei einem Split zugeteilt.
/// Dateien, die im Dataset-Root herumliegen, aber keine Trainingsdaten sind:
/// Hilfsskripte, Konfigurationen, READMEs. Diese landeten frueher wahllos in
/// train/ bzw. test/ – bei einem YOLO-Ordner wanderte so die data.yaml
/// (die einzige Quelle der Klassennamen) nach test/.
fn is_auxiliary_file(path: &Path) -> bool {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_lowercase();
    let ext  = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    if matches!(ext.as_str(), "py" | "yaml" | "yml" | "md" | "sh" | "toml" | "ini" | "cfg" | "log" | "cache" | "pyc") {
        return true;
    }
    matches!(name.as_str(), "readme" | "license" | "classes.txt" | "obj.names" | "obj.data" | "requirements.txt")
}

fn split_flat_files(base: &Path, train_r: f64, val_r: f64, test_r: f64) -> Result<SplitInfo, String> {
    let all_files = collect_files(base);
    let skipped   = all_files.iter().filter(|f| is_auxiliary_file(f)).count();
    let files: Vec<PathBuf> = all_files.into_iter().filter(|f| !is_auxiliary_file(f)).collect();
    let n = files.len();
    if n == 0 {
        return Err(if skipped > 0 {
            format!("Im Dataset-Root liegen nur Hilfsdateien ({} Stueck, z. B. Skripte oder data.yaml), keine Trainingsdaten. \
                     Falls das Dataset bereits train/val-Ordner mitbringt, ist kein Split noetig.", skipped)
        } else {
            "Keine Dateien im Dataset-Root.".to_string()
        });
    }

    let train_dir = base.join("train"); let val_dir = base.join("val"); let test_dir = base.join("test");
    let mut warnings: Vec<String> = Vec::new();
    let mut total_train = 0usize; let mut total_val = 0usize; let mut total_test = 0usize;
    let mut whole_file_fallback: Vec<&Path> = Vec::new();

    for f in &files {
        let ext = f.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        if is_row_splittable(&ext) {
            match split_row_file(f, &train_dir, &val_dir, &test_dir, train_r, val_r, test_r, &mut warnings) {
                Ok((tn, vn, tt)) => {
                    total_train += tn; total_val += vn; total_test += tt;
                    // FIX: Quelldatei wurde bereits vollstaendig zeilenweise in train/val/test
                    // aufgeteilt (Inhalt liegt jetzt in train.<ext>/val.<ext>/test.<ext>) -- die
                    // urspruengliche Datei im Dataset-Root MUSS geloescht werden, sonst bleibt sie
                    // liegen und taucht zusaetzlich als "unsplit" auf (Bug: train+val+unsplit
                    // gleichzeitig sichtbar, wirkt wie verdoppelte Daten).
                    fs::remove_file(f).ok();
                }
                Err(e) => {
                    // Bei Fehler (z.B. Python fehlt fuer Parquet) auf ganze-Datei-Split zurueckfallen,
                    // damit der Split nicht komplett scheitert -- User wird gewarnt.
                    warnings.push(format!("Zeilen-Split fuer '{}' fehlgeschlagen ({}), Datei wird als Ganzes zugeteilt.", f.file_name().unwrap_or_default().to_string_lossy(), e));
                    whole_file_fallback.push(f.as_path());
                }
            }
        } else {
            whole_file_fallback.push(f.as_path());
        }
    }

    // Fallback: nicht-splittable Dateien (oder Fehlerfaelle) als ganze Einheiten verteilen
    if !whole_file_fallback.is_empty() {
        fs::create_dir_all(&train_dir).ok(); fs::create_dir_all(&val_dir).ok(); fs::create_dir_all(&test_dir).ok();
        let fc = whole_file_fallback.len();
        let indices = shuffle_indices(fc);
        let (tn, vn, _tt) = split_counts(fc, train_r, val_r);
        for (slot, &file_idx) in indices.iter().enumerate() {
            let src = whole_file_fallback[file_idx];
            let fname = src.file_name().unwrap_or_default();
            let dst_dir = if slot < tn { &train_dir } else if slot < tn + vn { &val_dir } else { &test_dir };
            move_or_copy(src, &dst_dir.join(fname))?;
            if slot < tn { total_train += 1; } else if slot < tn + vn { total_val += 1; } else { total_test += 1; }
        }
    }

    if total_train + total_val + total_test == 0 {
        return Err("Split ergab keine Zeilen/Dateien -- Dataset pruefen.".to_string());
    }

    Ok(SplitInfo { train_count: total_train, val_count: total_val, test_count: total_test,
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

                // FIX: split_info live aus den tatsaechlichen train/val/test-Ordnern
                // berechnen statt dem evtl. veralteten gespeicherten Wert zu vertrauen.
                // Betrifft v.a. PreSplit-Datasets von HuggingFace, die bereits mit
                // train/test-Ordnern ankommen ohne dass split_dataset() je lief.
                let train_dir = storage.join("train");
                let val_dir   = storage.join("val");
                let test_dir  = storage.join("test");
                let has_split_dirs = train_dir.exists() || val_dir.exists() || test_dir.exists();
                if has_split_dirs {
                    let train_count = collect_files_recursive(&train_dir).len();
                    let val_count   = collect_files_recursive(&val_dir).len();
                    let test_count  = collect_files_recursive(&test_dir).len();
                    let total = (train_count + val_count + test_count).max(1) as f64;
                    d.split_info = Some(SplitInfo {
                        train_count, val_count, test_count,
                        train_ratio: train_count as f64 / total,
                        val_ratio:   val_count   as f64 / total,
                        test_ratio:  test_count  as f64 / total,
                    });
                    d.status = "split".to_string();
                } else if let Some(info) = detect_flat_split_files(&storage) {
                    // Flache Dateien train.csv / val.csv / test.csv — genau die
                    // Struktur, die der Hilfe-Dialog zeigt. Ohne diesen Zweig
                    // meldete die App "Kein Split" für ihr eigenes dokumentiertes
                    // Layout und verlangte einen manuellen Split.
                    d.split_info = Some(info);
                    d.status = "split".to_string();
                }
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

    // Bringt das Dataset seine train/val/test-Aufteilung schon mit, wird sie
    // uebernommen. Ohne das galt ein fertiges YOLO-Dataset als "Kein Split",
    // das Training blieb gesperrt und der erzwungene Split im Dataset-Manager
    // haette die Bild/Label-Paare zerrissen.
    let existing_layout = if src.is_dir() { detect_split_layout(&target) } else { None };
    let (status, split_info) = match &existing_layout {
        Some(layout) => {
            let (tr, va, te) = (layout.count_of("train"), layout.count_of("val"), layout.count_of("test"));
            let total = (tr + va + te).max(1) as f64;
            ("split", Some(SplitInfo {
                train_count: tr, val_count: va, test_count: te,
                train_ratio: tr as f64 / total,
                val_ratio:   va as f64 / total,
                test_ratio:  te as f64 / total,
            }))
        }
        None => ("unused", None),
    };

    let info = make_info(&dataset_id, &dataset_name, &model_id, "local",
        Some(source_path), &target, size, files, status, split_info,
        analysis.detected_type.clone(), analysis.pairing_status, analysis.warnings);
    // dataset.yaml für YOLO/Pascal generieren falls noch nicht vorhanden
    if matches!(info.dataset_type, DatasetType::YoloBbox | DatasetType::PascalVoc) {
        if let Some(layout) = &existing_layout {
            generate_split_dataset_yaml(&target, layout).ok();
        } else {
            let hint = &analysis.schema_hint;
            let img_dir = hint.as_ref().and_then(|s| s.get("images_dir")).and_then(|v| v.as_str()).unwrap_or("images");
            let lbl_dir = hint.as_ref()
                .and_then(|s| s.get("labels_dir").or_else(|| s.get("annotations_dir")))
                .and_then(|v| v.as_str()).unwrap_or("labels");
            generate_dataset_yaml(&target, img_dir, lbl_dir, false).ok();
        }
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
    if !is_safe_id(&dataset_id) { return Err("Ungültige Dataset-ID".to_string()); }
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
    if !is_safe_id(&dataset_id) { return Err("Ungültige Dataset-ID".to_string()); }
    if !(0.0..=1.0).contains(&train_ratio) || !(0.0..=1.0).contains(&val_ratio) || !(0.0..=1.0).contains(&test_ratio)
        || (train_ratio + val_ratio + test_ratio) > 1.0001 || train_ratio <= 0.0 {
        return Err("Ungültige Split-Verhältnisse (train muss > 0 sein, Summe ≤ 1.0).".to_string());
    }
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

/// Baut die Namen der beiden Hälften.
///
/// Früher wurde bei jedem Split erneut " (Hälfte N)" angehängt, sodass Namen
/// nach mehrfachem Splitten unbegrenzt wuchsen
/// ("ds (Haelfte 1) (Haelfte 1) (Haelfte 2)"). Stattdessen wird ein bereits
/// vorhandener Hälften-Pfad vertieft:
/// "ds" → "ds (Hälfte 1)" → "ds (Hälfte 1.1)" → "ds (Hälfte 1.1.2)".
/// Der ASCII-Suffix älterer Datensätze wird dabei mit erkannt.
fn half_names(name: &str) -> (String, String) {
    const PREFIX: &str = " (Hälfte ";
    const LEGACY_PREFIX: &str = " (Haelfte ";

    if name.ends_with(')') {
        for prefix in [PREFIX, LEGACY_PREFIX] {
            let Some(idx) = name.rfind(prefix) else { continue };
            let inner = &name[idx + prefix.len()..name.len() - 1];
            // Nur echte Hälften-Pfade vertiefen, z. B. "1" oder "1.2.1".
            if inner.is_empty() || !inner.chars().all(|c| c.is_ascii_digit() || c == '.') {
                continue;
            }
            let base = &name[..idx];
            return (
                format!("{}{}{}.1)", base, PREFIX, inner),
                format!("{}{}{}.2)", base, PREFIX, inner),
            );
        }
    }

    (format!("{}{}1)", name, PREFIX), format!("{}{}2)", name, PREFIX))
}

#[cfg(test)]
mod half_names_tests {
    use super::half_names;

    #[test]
    fn erster_split_haengt_suffix_an() {
        assert_eq!(
            half_names("kp20k2"),
            ("kp20k2 (Hälfte 1)".to_string(), "kp20k2 (Hälfte 2)".to_string())
        );
    }

    #[test]
    fn wiederholter_split_vertieft_statt_anzuhaengen() {
        let (a, b) = half_names("kp20k2 (Hälfte 1)");
        assert_eq!(a, "kp20k2 (Hälfte 1.1)");
        assert_eq!(b, "kp20k2 (Hälfte 1.2)");

        let (c, _) = half_names(&a);
        assert_eq!(c, "kp20k2 (Hälfte 1.1.1)");
    }

    #[test]
    fn alte_ascii_namen_werden_erkannt() {
        let (a, b) = half_names("kp20k2 (Haelfte 2)");
        assert_eq!(a, "kp20k2 (Hälfte 2.1)");
        assert_eq!(b, "kp20k2 (Hälfte 2.2)");
    }

    #[test]
    fn klammern_ohne_haelften_pfad_bleiben_unangetastet() {
        let (a, _) = half_names("dataset (Kopie)");
        assert_eq!(a, "dataset (Kopie) (Hälfte 1)");
    }
}

#[tauri::command]
pub async fn split_dataset_in_half(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    dataset_id: String, model_id: String,
    preserve_splits: Option<bool>,
) -> Result<serde_json::Value, String> {
    // Standard: vorhandene train/val/test-Struktur an beide Hälften vererben.
    let preserve = preserve_splits.unwrap_or(true);
    if !is_safe_id(&dataset_id) { return Err("Ungültige Dataset-ID".to_string()); }
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
            // Vorher wurden train/val/test-Ordner komplett ÜBERSPRUNGEN —
            // das Halbieren eines bereits gesplitteten FolderClass-Datasets
            // erzeugte damit zwei LEERE Hälften. Jetzt: Split-Ordner werden
            // pro Split pro Klasse halbiert (Struktur vererbt) bzw. bei
            // preserve=false pro Klasse zusammengelegt.
            let split_dirs: Vec<String> = ["train", "val", "valid", "test"].iter()
                .map(|s| s.to_string())
                .filter(|s| base.join(s).is_dir())
                .collect();

            for sd in &split_dirs {
                for class in list_subdir_names(&base.join(sd)) {
                    let files = list_files_in_dir(&base.join(sd).join(&class));
                    let half  = files.len() / 2;
                    let (ta, tb) = if preserve {
                        (dir_a.join(sd).join(&class), dir_b.join(sd).join(&class))
                    } else {
                        (dir_a.join(&class), dir_b.join(&class))
                    };
                    fs::create_dir_all(&ta).ok();
                    fs::create_dir_all(&tb).ok();
                    for (i, f) in files.iter().enumerate() {
                        let dst_dir = if i < half { &ta } else { &tb };
                        let fname = f.file_name().unwrap_or_default().to_string_lossy().to_string();
                        // Beim Zusammenlegen (preserve=false) können gleiche
                        // Dateinamen aus train/ und val/ kollidieren → Split-Präfix
                        let mut dst = dst_dir.join(&fname);
                        if dst.exists() { dst = dst_dir.join(format!("{}_{}", sd, fname)); }
                        fs::copy(f, &dst).ok();
                    }
                }
            }

            // Klassen-Ordner auf Root-Ebene (ungesplitteter Anteil) wie bisher
            for class in list_subdir_names(&base) {
                if matches!(class.as_str(), "train"|"val"|"valid"|"test") { continue; }
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
            // FIX: Strukturierte Dateien (Parquet/CSV/TSV/JSONL/JSON) werden zeilenweise
            // gesplittet statt als ganze Datei einer Haelfte zugeteilt zu werden --
            // sonst landet bei 2 Dateien Datei 1 komplett in Haelfte A, Datei 2 komplett
            // in Haelfte B (das urspruengliche Bug-Symptom). Jede splittable Datei wird
            // intern in zwei Haelften (50/50) aufgeteilt, Header/Schema bleibt erhalten.
            // Nicht-splittable Dateien werden weiterhin pfad-erhaltend kopiert.
            let files = collect_files_recursive(&base);
            if files.is_empty() { return Err("Keine Dateien im Dataset.".to_string()); }
            let mut whole_files: Vec<&PathBuf> = Vec::new();
            let mut half_warnings: Vec<String> = Vec::new();
            for f in &files {
                let ext = f.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
                if is_row_splittable(&ext) {
                    // preserve: relative Struktur (z.B. train/, val/) in beiden Hälften
                    // spiegeln — vorher landete alles flach im Root und die Hälften
                    // verloren ihre Split-Struktur. preserve=false = bewusst flach.
                    let rel_parent = if preserve {
                        f.parent()
                            .and_then(|p| p.strip_prefix(&base).ok())
                            .map(|p| p.to_path_buf())
                            .unwrap_or_default()
                    } else {
                        PathBuf::new()
                    };
                    let out_a = dir_a.join(&rel_parent);
                    let out_b = dir_b.join(&rel_parent);
                    // train_dir/val_dir Parameter zweckentfremdet als Haelfte A / Haelfte B,
                    // test_dir wird nicht genutzt (ratio 0.5/0.5/0.0).
                    let unused_test = std::env::temp_dir().join(format!("ft_unused_split_{}", uuid::Uuid::new_v4()));
                    match split_row_file(f, &out_a, &out_b, &unused_test, 0.5, 0.5, 0.0, &mut half_warnings) {
                        Ok(_) => {
                            // Quelldatei bleibt absichtlich erhalten: Halbieren erzeugt zwei NEUE
                            // Datasets (Kopie-Semantik, wie bei allen anderen Typen in dieser
                            // Funktion) — das Original-Dataset bleibt unangetastet bestehen.
                        }
                        Err(e) => {
                            half_warnings.push(format!("Zeilen-Split fuer '{}' fehlgeschlagen ({}), Datei wird als Ganzes kopiert.", f.file_name().unwrap_or_default().to_string_lossy(), e));
                            whole_files.push(f);
                        }
                    }
                    fs::remove_dir_all(&unused_test).ok();
                } else {
                    whole_files.push(f);
                }
            }
            let half = whole_files.len() / 2;
            for (i, f) in whole_files.iter().enumerate() {
                let dst_root = if i < half { &dir_a } else { &dir_b };
                // preserve: Pfadstruktur erhalten; sonst flach ablegen (mit
                // Kollisionsschutz, falls z.B. train/x.bin und val/x.bin existieren)
                let dst = if preserve {
                    let rel = f.strip_prefix(&base).unwrap_or(f.as_path());
                    dst_root.join(rel)
                } else {
                    let fname = f.file_name().unwrap_or_default().to_string_lossy().to_string();
                    let mut d = dst_root.join(&fname);
                    if d.exists() {
                        let prefix = f.parent()
                            .and_then(|p| p.strip_prefix(&base).ok())
                            .map(|p| p.to_string_lossy().replace('/', "_"))
                            .unwrap_or_default();
                        d = dst_root.join(format!("{}_{}", prefix, fname));
                    }
                    d
                };
                if let Some(p) = dst.parent() { fs::create_dir_all(p).ok(); }
                fs::copy(f, &dst).ok();
            }
        }
    }

    let (sa, fa) = dir_size(&dir_a);
    let (sb, fb) = dir_size(&dir_b);
    let (name_a, name_b) = half_names(&ds.name);

    // Geerbte Split-Struktur → Hälfte ist direkt trainierbar (Status "split"
    // + SplitInfo aus den tatsächlichen Datei-Zahlen der Split-Ordner).
    let has_split_dirs = |dir: &Path| -> bool {
        ["train", "val", "valid", "test"].iter().any(|s| dir.join(s).is_dir())
    };
    let count_split_info = |dir: &Path| -> SplitInfo {
        let cnt = |s: &str| -> usize {
            let d = dir.join(s);
            if d.is_dir() { collect_files_recursive(&d).len() } else { 0 }
        };
        let train = cnt("train");
        let val   = cnt("val") + cnt("valid");
        let test  = cnt("test");
        let tot   = (train + val + test).max(1) as f64;
        SplitInfo {
            train_count: train, val_count: val, test_count: test,
            train_ratio: train as f64 / tot,
            val_ratio:   val as f64 / tot,
            test_ratio:  test as f64 / tot,
        }
    };
    let inherited_a = preserve && has_split_dirs(&dir_a);
    let inherited_b = preserve && has_split_dirs(&dir_b);

    let ds_a = make_info(&id_a, &name_a, &model_id, "local", None, &dir_a, sa, fa,
        if inherited_a { "split" } else { "unused" },
        if inherited_a { Some(count_split_info(&dir_a)) } else { None },
        ds.dataset_type.clone(), None, vec![]);
    let ds_b = make_info(&id_b, &name_b, &model_id, "local", None, &dir_b, sb, fb,
        if inherited_b { "split" } else { "unused" },
        if inherited_b { Some(count_split_info(&dir_b)) } else { None },
        ds.dataset_type, None, vec![]);

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
    if !is_safe_id(&dataset_id) { return Err("Ungültige Dataset-ID".to_string()); }
    let user_id      = get_user_id(&state)?;
    let datasets_dir = get_datasets_dir(&app_handle, &user_id)?;
    let dataset_dir  = datasets_dir.join(&dataset_id);
    if !dataset_dir.exists() { return Ok(vec![]); }
    let mut files: Vec<serde_json::Value> = Vec::new();
    let mut seen: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
    let mut push = |file: &Path, split: &str, files: &mut Vec<serde_json::Value>, seen: &mut std::collections::HashSet<PathBuf>| {
        if !seen.insert(file.to_path_buf()) { return; }
        if let Ok(meta) = fs::metadata(file) {
            files.push(serde_json::json!({
                "name": file.file_name().unwrap_or_default().to_string_lossy(),
                "path": file.to_string_lossy(), "size": meta.len(),
                "is_dir": false, "split": split,
            }));
        }
    };

    // Bereits aufgeteiltes Bild/Label-Dataset: die Bilder liegen unter
    // images/train bzw. train/images. Ohne diesen Zweig meldete das Labor
    // "Keine Dateien für Split \"val\" gefunden. Verfügbare Splits: images, labels".
    if let Some(layout) = detect_split_layout(&dataset_dir) {
        for (canon, dir_name, _) in &layout.splits {
            let (img_dir, lbl_dir) = if layout.style == "nested" {
                (dataset_dir.join(&layout.images_dir).join(dir_name),
                 dataset_dir.join(&layout.labels_dir).join(dir_name))
            } else {
                (dataset_dir.join(dir_name).join(&layout.images_dir),
                 dataset_dir.join(dir_name).join(&layout.labels_dir))
            };
            for file in list_files_in_dir(&img_dir) {
                push(&file, canon, &mut files, &mut seen);
            }
            // Labels bekommen ein eigenes Tag, damit die Split-Auswahl im Labor
            // nur Bilder liefert und nicht zur Haelfte .txt-Dateien.
            for file in list_files_in_dir(&lbl_dir) {
                push(&file, "labels", &mut files, &mut seen);
            }
        }
    }

    for split in &["train", "val", "test"] {
        let split_dir = dataset_dir.join(split);
        if split_dir.exists() {
            for file in collect_files_recursive(&split_dir) {
                push(&file, split, &mut files, &mut seen);
            }
        }
    }
    // Unterordner wie images/ und labels/ ebenfalls listen (mit split="subdir")
    let known_subdirs = ["images", "labels", "annotations", "imgs", "clips"];
    for subdir in &known_subdirs {
        let subdir_path = dataset_dir.join(subdir);
        if subdir_path.exists() {
            for file in list_files_in_dir(&subdir_path) {
                push(&file, subdir, &mut files, &mut seen);
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
    if !is_safe_id(&dataset_id) { return Err("Ungültige Dataset-ID".to_string()); }
    if !matches!(target_split.as_str(), "train" | "val" | "test" | "unused") {
        return Err(format!("Ungültiger Ziel-Split: {}", target_split));
    }
    let user_id      = get_user_id(&state)?;
    let datasets_dir = get_datasets_dir(&app_handle, &user_id)?;
    let target_dir   = datasets_dir.join(&dataset_id).join(&target_split);
    fs::create_dir_all(&target_dir).map_err(|e| format!("mkdir: {}", e))?;
    for fp in &file_paths {
        let src = Path::new(fp);
        if !src.exists() { continue; }
        // Nur Dateien innerhalb des eigenen Datasets-Verzeichnisses verschieben
        if !is_within_dir(src, &datasets_dir) {
            return Err(format!("Pfad liegt außerhalb des Dataset-Ordners: {}", fp));
        }
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
    if !is_safe_id(&dataset_id) { return Err("Ungültige Dataset-ID".to_string()); }
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
pub async fn delete_dataset_files(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    file_paths: Vec<String>,
) -> Result<(), String> {
    // FIX Sicherheit: Vorher konnte hierüber JEDE Datei auf der Festplatte gelöscht
    // werden. Jetzt nur noch Dateien innerhalb des eigenen Datasets-Verzeichnisses.
    let user_id      = get_user_id(&state)?;
    let datasets_dir = get_datasets_dir(&app_handle, &user_id)?;
    for fp in &file_paths {
        let p = Path::new(fp);
        if !p.exists() { continue; }
        if !is_within_dir(p, &datasets_dir) {
            return Err(format!("Pfad liegt außerhalb des Dataset-Ordners: {}", fp));
        }
        fs::remove_file(p).map_err(|e| format!("Delete: {}", e))?;
    }
    Ok(())
}

#[tauri::command]
pub async fn preview_parquet_file(file_path: String, max_rows: Option<usize>) -> Result<serde_json::Value, String> {
    use std::process::{Command, Stdio};
    let path = Path::new(&file_path);
    if !path.exists() { return Err(format!("Datei nicht gefunden: {}", file_path)); }
    let limit = max_rows.unwrap_or(50).min(500);

    let python_script = format!(r#"
import sys, json
import pandas as pd

try:
    df = pd.read_parquet(sys.argv[1])
    total_rows = len(df)
    total_cols = len(df.columns)
    columns = [{{"name": str(c), "dtype": str(df[c].dtype)}} for c in df.columns]
    preview = df.head({limit})
    rows = json.loads(preview.to_json(orient="records", date_format="iso", default_handler=str))
    print(json.dumps({{
        "columns": columns,
        "rows": rows,
        "total_rows": total_rows,
        "total_cols": total_cols,
        "shown_rows": len(rows),
    }}))
except Exception as e:
    print(json.dumps({{"error": str(e)}}), file=sys.stderr)
    sys.exit(1)
"#);

    let python_cmd = find_python_cmd()
        .map_err(|_| "Python nicht gefunden — Parquet-Preview benötigt Python mit pandas/pyarrow.".to_string())?;

    let output = Command::new(python_cmd)
        .arg("-c").arg(&python_script).arg(&file_path)
        .stdout(Stdio::piped()).stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("Python spawn: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let user_error = serde_json::from_str::<serde_json::Value>(stderr.trim())
            .ok()
            .and_then(|v| v.get("error").and_then(|e| e.as_str()).map(String::from))
            .unwrap_or_else(|| stderr.trim().to_string());
        return Err(format!("Parquet-Preview fehlgeschlagen: {}", user_error));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(stdout.trim()).map_err(|e| format!("JSON-Parse: {} (raw: {})", e, stdout.trim()))
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
            Ok(format!("[Parquet] {} bytes -- Binärformat, kein Preview.", size))
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

#[derive(Debug, Deserialize, Clone)]
struct HfParquetFile {
    url: String,
    filename: String,
    size: Option<u64>,
    split: Option<String>,
    #[serde(default)]
    config: Option<String>,
}

/// Haengt bei Namenskollision ein Suffix an: 0000.parquet, 0000_1.parquet, ...
fn unique_file_name(dir: &Path, base: &str, extension: &str) -> String {
    let candidate = format!("{}.{}", base, extension);
    if !dir.join(&candidate).exists() { return candidate; }
    let mut i = 1usize;
    loop {
        let c = format!("{}_{}.{}", base, i, extension);
        if !dir.join(&c).exists() { return c; }
        i += 1;
    }
}

/// Erkennt Datasets, deren Splits als flache Dateien vorliegen
/// (`train.csv`, `val.jsonl`, `test.parquet`, …) statt als Unterordner.
///
/// Genau dieses Layout zeigt der Hilfe-Dialog beim lokalen Import; es wurde
/// bisher trotzdem als "Kein Split" eingestuft.
/// Holt die Zeilenzahlen je Split vom HuggingFace datasets-server.
///
/// Der `/parquet`-Endpunkt liefert nur Dateigroessen. Ohne die Zeilenzahlen
/// zeigte die Dataset-Karte die Anzahl Dateien an — bei imdb also "1 / 0 / 1"
/// statt "25000 / 0 / 25000". Schlaegt der Aufruf fehl, gibt es `None` und die
/// Karte verhaelt sich wie bisher.
async fn fetch_hf_split_sizes(
    repo_id: &str,
    selected: &[(HfParquetFile, String)],
) -> Option<SplitInfo> {
    if selected.is_empty() { return None; }
    let url = format!(
        "https://datasets-server.huggingface.co/size?dataset={}",
        urlencoding::encode(repo_id)
    );
    let body: serde_json::Value = reqwest::Client::new()
        .get(&url)
        .send().await.ok()?
        .json().await.ok()?;
    let splits = body.get("size")?.get("splits")?.as_array()?;

    // Nur die Konfiguration zaehlen, aus der wir tatsaechlich Dateien geladen haben.
    let used_config = selected.first().and_then(|(f, _)| f.config.clone());
    let (mut train, mut val, mut test) = (0usize, 0usize, 0usize);
    let mut found = false;
    for entry in splits {
        let cfg = entry.get("config").and_then(|v| v.as_str());
        if let (Some(used), Some(cfg)) = (used_config.as_deref(), cfg) {
            if used != cfg { continue; }
        }
        let name = entry.get("split").and_then(|v| v.as_str()).unwrap_or("");
        let rows = entry.get("num_rows").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        // Nur Splits zaehlen, die wir auch heruntergeladen haben.
        let dir = match map_hf_split(name) { Some(d) => d, None => continue };
        if !selected.iter().any(|(_, d)| d == dir) { continue; }
        match dir {
            "train" => train += rows,
            "val"   => val   += rows,
            _        => test += rows,
        }
        found = true;
    }
    if !found || train + val + test == 0 { return None; }
    let total = (train + val + test) as f64;
    Some(SplitInfo {
        train_count: train, val_count: val, test_count: test,
        train_ratio: train as f64 / total,
        val_ratio:   val as f64 / total,
        test_ratio:  test as f64 / total,
    })
}

/// Zaehlt die Datenzeilen einer zeilenbasierten Datei (CSV/TSV ohne Kopfzeile).
/// Fuer Formate, die sich so nicht lesen lassen (z.B. Parquet), gibt es `None`.
fn count_text_rows(path: &Path) -> Option<usize> {
    let ext = path.extension().and_then(|e| e.to_str())?.to_lowercase();
    if !matches!(ext.as_str(), "csv" | "tsv" | "jsonl" | "txt") { return None; }
    let content = fs::read_to_string(path).ok()?;
    let lines = content.lines().filter(|l| !l.trim().is_empty()).count();
    // Bei CSV/TSV ist die erste Zeile der Header und keine Datenzeile.
    Some(if matches!(ext.as_str(), "csv" | "tsv") { lines.saturating_sub(1) } else { lines })
}

fn detect_flat_split_files(storage: &Path) -> Option<SplitInfo> {
    let entries = fs::read_dir(storage).ok()?;
    let (mut train_count, mut val_count, mut test_count) = (0usize, 0usize, 0usize);
    let mut found_any = false;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() { continue; }
        let ext_ok = path.extension().and_then(|e| e.to_str())
            .map(|e| is_row_splittable(e) || e.eq_ignore_ascii_case("txt")
                 || e.eq_ignore_ascii_case("arrow"))
            .unwrap_or(false);
        if !ext_ok { continue; }
        let stem = match path.file_stem().and_then(|s| s.to_str()) {
            Some(s) => s.to_lowercase(),
            None => continue,
        };
        // Zeilen zaehlen statt Dateien: sonst meldete die App fuer
        // train.csv/val.csv/test.csv "1 / 1 / 1" und 33/33/33 Prozent,
        // egal wie die Daten tatsaechlich verteilt sind.
        let n = count_text_rows(&path).unwrap_or(1);
        match stem.as_str() {
            "train" | "training" => { train_count += n; found_any = true; }
            "val" | "valid" | "validation" | "dev" => { val_count += n; found_any = true; }
            "test" | "testing" | "eval" => { test_count += n; found_any = true; }
            _ => {}
        }
    }
    if !found_any { return None; }
    // Ein einzelnes train.csv ist noch kein Split.
    if train_count == 0 || (val_count == 0 && test_count == 0) { return None; }
    let total = (train_count + val_count + test_count).max(1) as f64;
    Some(SplitInfo {
        train_count, val_count, test_count,
        train_ratio: train_count as f64 / total,
        val_ratio:   val_count   as f64 / total,
        test_ratio:  test_count  as f64 / total,
    })
}

/// Ordnet einen HuggingFace-Splitnamen einem lokalen Split-Verzeichnis zu.
/// `None` = Split ohne verwertbare Labels bzw. unbekannt -> wird uebersprungen.
fn map_hf_split(split: &str) -> Option<&'static str> {
    match split.to_lowercase().as_str() {
        "train" | "training" => Some("train"),
        "validation" | "valid" | "val" | "dev" => Some("val"),
        "test" | "testing" | "eval" => Some("test"),
        _ => None,
    }
}

/// Waehlt aus der HF-Parquet-Liste genau eine Config und deren beschriftete
/// Splits aus.
///
/// Ohne diese Auswahl landeten alle Splits als gleichnamige `0000.parquet` im
/// selben Zielordner und ueberschrieben sich gegenseitig — uebrig blieb der
/// zuletzt geschriebene. Bei IMDB war das `unsupervised`, wo jedes Label -1 ist;
/// das Training bekam dadurch nur eine einzige Klasse zu sehen.
fn select_hf_parquet_files(files: &[HfParquetFile]) -> (Vec<(HfParquetFile, String)>, Vec<String>) {
    let mut warnings: Vec<String> = Vec::new();
    if files.is_empty() { return (Vec::new(), warnings); }

    // Configs in Reihenfolge des ersten Auftretens sammeln.
    let mut configs: Vec<String> = Vec::new();
    for f in files {
        let c = f.config.clone().unwrap_or_else(|| "default".to_string());
        if !configs.contains(&c) { configs.push(c); }
    }

    let files_of = |cfg: &str| -> Vec<HfParquetFile> {
        files.iter()
            .filter(|f| f.config.as_deref().unwrap_or("default") == cfg)
            .cloned().collect()
    };

    // Bevorzugt die Config, die einen train-Split mitbringt.
    let chosen = configs.iter()
        .find(|c| files_of(c).iter().any(|f|
            f.split.as_deref().map(|s| map_hf_split(s) == Some("train")).unwrap_or(false)))
        .cloned()
        .unwrap_or_else(|| configs[0].clone());

    if configs.len() > 1 {
        warnings.push(format!(
            "Dataset hat {} Konfigurationen ({}). Importiert wurde '{}'.",
            configs.len(), configs.join(", "), chosen));
    }

    let candidates = files_of(&chosen);
    let mut selected: Vec<(HfParquetFile, String)> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    for f in &candidates {
        let split_name = f.split.clone().unwrap_or_default();
        match map_hf_split(&split_name) {
            Some(dir) => selected.push((f.clone(), dir.to_string())),
            None => {
                if !split_name.is_empty() && !skipped.contains(&split_name) {
                    skipped.push(split_name);
                }
            }
        }
    }
    if !skipped.is_empty() {
        warnings.push(format!(
            "Splits ohne verwertbare Labels uebersprungen: {}.", skipped.join(", ")));
    }

    // Notfall: keine bekannten Splitnamen -> alles als train behandeln,
    // damit ungewoehnlich benannte Datasets nicht komplett leer ankommen.
    if selected.is_empty() {
        warnings.push("Keine Standard-Splits erkannt — alle Dateien werden als 'train' importiert.".to_string());
        selected = candidates.into_iter().map(|f| (f, "train".to_string())).collect();
    }
    (selected, warnings)
}

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
    // Genau eine Config + nur beschriftete Splits, jeder in sein eigenes
    // Unterverzeichnis — sonst ueberschreiben sich die gleichnamigen
    // 0000.parquet-Dateien der einzelnen Splits gegenseitig.
    let (selected, mut select_warnings) = select_hf_parquet_files(&files);
    if selected.is_empty() {
        return Err("Keine importierbaren Parquet-Dateien im Dataset gefunden.".to_string());
    }
    let total_files  = selected.len();
    let total_bytes: u64 = selected.iter().filter_map(|(f, _)| f.size).sum();
    let t0 = Instant::now();
    let mut global_dl: u64 = 0;
    for (file_idx, (hf_file, split_dir)) in selected.iter().enumerate() {
        let base_name = if hf_file.filename.is_empty() { format!("file_{}.parquet", file_idx) } else { hf_file.filename.clone() };
        let split_path = target.join(split_dir);
        fs::create_dir_all(&split_path).map_err(|e| format!("mkdir '{}': {}", split_dir, e))?;
        // Innerhalb eines Splits kann es mehrere Shards mit gleichem Namen geben.
        let stem = Path::new(&base_name).file_stem().and_then(|s| s.to_str()).unwrap_or("data");
        let ext  = Path::new(&base_name).extension().and_then(|s| s.to_str()).unwrap_or("parquet");
        let fname = unique_file_name(&split_path, stem, ext);
        let response = client.get(&hf_file.url).header("User-Agent", "FrameTrain-Desktop/1.0").send().await
            .map_err(|e| format!("HTTP GET '{}': {}", fname, e))?;
        if !response.status().is_success() { return Err(format!("HTTP {} fuer '{}'", response.status(), fname)); }
        let file_total = hf_file.size.or_else(|| response.content_length()).unwrap_or(0);
        let mut out_file = tokio::fs::File::create(split_path.join(&fname)).await
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
    let mut warnings = detected.warnings;
    warnings.append(&mut select_warnings);
    // Zeilenzahlen je Split von HuggingFace nachladen. Ohne sie stand auf der
    // Dataset-Karte "1 / 0 / 1" (Datei-Zahlen) statt "25000 / 0 / 25000" — der
    // Nutzer konnte nicht erkennen, wie viele Daten er ueberhaupt hat.
    let hf_split_info = fetch_hf_split_sizes(&repo_id, &selected).await;
    let hf_status = if hf_split_info.is_some() { "split" } else { "unused" };
    let info = make_info(&dataset_id, &dataset_name, &model_id, "huggingface",
        Some(repo_id), &target, total_size, file_count, hf_status, hf_split_info,
        detected.detected_type, detected.pairing_status, warnings);
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
    let python_cmd = find_python_cmd()?;
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
    if !is_safe_id(&dataset_id) { return Err("Ungültige Dataset-ID".to_string()); }
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
    // Kommentare am Zeilenende abschneiden. Ohne das landete "images/train  # 463
    // Bilder" komplett im Pfad-Feld des Editors und wurde beim Speichern als
    // Pfad zurueckgeschrieben — Ultralytics fand den Ordner dann nicht mehr.
    // Betrifft auch mitgebrachte data.yaml-Dateien, die fast immer Kommentare haben.
    let strip_comment = |s: &str| -> String {
        match s.find(" #").or_else(|| if s.starts_with('#') { Some(0) } else { None }) {
            Some(i) => s[..i].trim_end().to_string(),
            None => s.to_string(),
        }
    };
    for line in raw.lines() {
        let trimmed_raw = line.trim();
        if trimmed_raw.starts_with('#') { continue; }
        let stripped = strip_comment(trimmed_raw);
        let trimmed = stripped.as_str();
        if trimmed.is_empty() { continue; }
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
    if !is_safe_id(&dataset_id) { return Err("Ungültige Dataset-ID".to_string()); }
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
#[cfg(test)]
mod hf_split_selection_tests {
    use super::{select_hf_parquet_files, map_hf_split, HfParquetFile};

    fn f(config: &str, split: &str, size: u64) -> HfParquetFile {
        HfParquetFile {
            url: format!("https://example/{}/{}", config, split),
            // Auf HuggingFace heissen die Dateien ALLER Splits gleich.
            filename: "0000.parquet".to_string(),
            size: Some(size),
            split: Some(split.to_string()),
            config: Some(config.to_string()),
        }
    }

    #[test]
    fn imdb_nimmt_train_und_test_aber_nicht_unsupervised() {
        // Realer Aufbau von stanfordnlp/imdb. Der unsupervised-Split hat
        // durchgehend label = -1; landete er als letzte Datei im Zielordner,
        // sah das Training nur eine einzige Klasse.
        let files = vec![
            f("plain_text", "test", 19_500_000),
            f("plain_text", "train", 20_000_000),
            f("plain_text", "unsupervised", 40_100_000),
        ];
        let (selected, warnings) = select_hf_parquet_files(&files);

        let dirs: Vec<&str> = selected.iter().map(|(_, d)| d.as_str()).collect();
        assert_eq!(selected.len(), 2, "unsupervised muss aussortiert werden");
        assert!(dirs.contains(&"train"));
        assert!(dirs.contains(&"test"));
        assert!(!dirs.contains(&"unsupervised"));
        assert!(warnings.iter().any(|w| w.contains("unsupervised")));
    }

    #[test]
    fn jeder_split_bekommt_ein_eigenes_verzeichnis() {
        // Kernpunkt: gleiche Dateinamen duerfen sich nicht ueberschreiben.
        let files = vec![f("default", "train", 10), f("default", "validation", 5)];
        let (selected, _) = select_hf_parquet_files(&files);
        assert_eq!(selected.len(), 2);
        assert!(selected.iter().all(|(file, _)| file.filename == "0000.parquet"));
        let mut dirs: Vec<&str> = selected.iter().map(|(_, d)| d.as_str()).collect();
        dirs.sort();
        assert_eq!(dirs, vec!["train", "val"]);
    }

    #[test]
    fn mehrere_configs_werden_auf_eine_reduziert() {
        let files = vec![
            f("en", "train", 10),
            f("en", "test", 5),
            f("de", "train", 10),
        ];
        let (selected, warnings) = select_hf_parquet_files(&files);
        assert!(selected.iter().all(|(file, _)| file.config.as_deref() == Some("en")));
        assert!(warnings.iter().any(|w| w.contains("Konfigurationen")));
    }

    #[test]
    fn unbekannte_splitnamen_landen_als_train() {
        let files = vec![f("default", "kompletter_datensatz", 10)];
        let (selected, warnings) = select_hf_parquet_files(&files);
        assert_eq!(selected.len(), 1, "nichts darf verloren gehen");
        assert_eq!(selected[0].1, "train");
        assert!(!warnings.is_empty());
    }

    #[test]
    fn splitnamen_werden_normalisiert() {
        assert_eq!(map_hf_split("validation"), Some("val"));
        assert_eq!(map_hf_split("Training"), Some("train"));
        assert_eq!(map_hf_split("unsupervised"), None);
    }
}

#[cfg(test)]
mod flat_split_tests {
    use super::*;
    use std::fs;

    fn write(dir: &Path, name: &str, rows: usize) {
        let mut s = String::from("text,label\n");
        for i in 0..rows {
            s.push_str(&format!("Beispiel {},{}\n", i, i % 2));
        }
        fs::write(dir.join(name), s).unwrap();
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("ft_flat_{}_{}", name, std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn flache_train_val_test_dateien_gelten_als_split() {
        let d = tmp("basic");
        write(&d, "train.csv", 80);
        write(&d, "val.csv", 10);
        write(&d, "test.csv", 10);
        let info = detect_flat_split_files(&d).expect("Split sollte erkannt werden");
        // Zeilen, nicht Dateien — sonst stuende hier ueberall 1.
        assert_eq!(info.train_count, 80);
        assert_eq!(info.val_count, 10);
        assert_eq!(info.test_count, 10);
        assert!((info.train_ratio - 0.8).abs() < 1e-9);
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn einzelnes_train_csv_ist_kein_split() {
        let d = tmp("only_train");
        write(&d, "train.csv", 50);
        assert!(detect_flat_split_files(&d).is_none());
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn fremde_dateinamen_ergeben_keinen_split() {
        let d = tmp("foreign");
        write(&d, "daten.csv", 50);
        write(&d, "mehr.csv", 10);
        assert!(detect_flat_split_files(&d).is_none());
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn validation_und_dev_zaehlen_als_val() {
        let d = tmp("aliases");
        write(&d, "training.csv", 30);
        write(&d, "validation.csv", 6);
        let info = detect_flat_split_files(&d).expect("Split sollte erkannt werden");
        assert_eq!(info.train_count, 30);
        assert_eq!(info.val_count, 6);
        assert_eq!(info.test_count, 0);
        let _ = fs::remove_dir_all(&d);
    }
}

#[cfg(test)]
mod split_layout_tests {
    use super::{detect_split_layout, detect_dataset_type, parse_yaml_class_names,
                read_class_names, generate_split_dataset_yaml, is_auxiliary_file, DatasetType};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let n = COUNTER.fetch_add(1, Ordering::SeqCst);
            let p = std::env::temp_dir().join(format!("ft_ds_test_{}_{}_{}", tag, std::process::id(), n));
            let _ = fs::remove_dir_all(&p);
            fs::create_dir_all(&p).unwrap();
            TempDir(p)
        }
        fn path(&self) -> &Path { &self.0 }
    }
    impl Drop for TempDir {
        fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); }
    }

    /// Legt ein Bild/Label-Paar an. Der Bildinhalt ist egal, nur die Endung zaehlt.
    fn make_pair(img_dir: &Path, lbl_dir: &Path, stem: &str) {
        fs::create_dir_all(img_dir).unwrap();
        fs::create_dir_all(lbl_dir).unwrap();
        fs::write(img_dir.join(format!("{}.jpg", stem)), b"x").unwrap();
        fs::write(lbl_dir.join(format!("{}.txt", stem)), b"0 0.5 0.5 0.1 0.1").unwrap();
    }

    /// Der Aufbau von Karols Ski-Dataset: images/train + labels/train,
    /// dazu ein leerer images/test-Ordner und eine data.yaml im Root.
    fn build_nested(root: &Path) {
        for (split, n) in [("train", 3), ("val", 2)] {
            for i in 0..n {
                make_pair(&root.join("images").join(split), &root.join("labels").join(split), &format!("{}_{}", split, i));
            }
        }
        fs::create_dir_all(root.join("images/test")).unwrap();
        fs::create_dir_all(root.join("labels/test")).unwrap();
        fs::write(root.join("data.yaml"),
            "train: images/train\nval: images/val\n\nnc: 3\nnames:\n  - 'Tree'\n  - 'Stone'\n  - 'Person'\n").unwrap();
    }

    #[test]
    fn nested_yolo_split_wird_erkannt() {
        let dir = TempDir::new("nested");
        build_nested(dir.path());
        let layout = detect_split_layout(dir.path()).expect("images/train + labels/train muss erkannt werden");
        assert_eq!(layout.style, "nested");
        assert_eq!(layout.images_dir, "images");
        assert_eq!(layout.labels_dir, "labels");
        assert_eq!(layout.count_of("train"), 3);
        assert_eq!(layout.count_of("val"), 2);
        // Leerer test-Ordner darf nicht als Split gelten.
        assert_eq!(layout.count_of("test"), 0);
        assert_eq!(layout.splits.len(), 2);
    }

    #[test]
    fn grouped_yolo_split_wird_erkannt() {
        let dir = TempDir::new("grouped");
        for (split, n) in [("train", 4), ("valid", 1)] {
            for i in 0..n {
                make_pair(&dir.path().join(split).join("images"), &dir.path().join(split).join("labels"), &format!("{}_{}", split, i));
            }
        }
        let layout = detect_split_layout(dir.path()).expect("train/images + train/labels muss erkannt werden");
        assert_eq!(layout.style, "grouped");
        assert_eq!(layout.count_of("train"), 4);
        // "valid" muss auf "val" normalisiert werden.
        assert_eq!(layout.count_of("val"), 1);
    }

    #[test]
    fn detect_meldet_yolo_statt_unbekannt() {
        // Das war der eigentliche Fehler: 0 % Konfidenz, "Dataset-Typ konnte
        // nicht erkannt werden" – und damit blieb das Training gesperrt.
        let dir = TempDir::new("detect");
        build_nested(dir.path());
        // Beiwerk, das im echten Ordner ebenfalls herumliegt.
        fs::write(dir.path().join("shuffel.py"), b"# helper").unwrap();
        fs::create_dir_all(dir.path().join("raw")).unwrap();

        let a = detect_dataset_type(dir.path());
        assert!(matches!(a.detected_type, DatasetType::YoloBbox), "erkannt als {:?}", a.detected_type);
        assert!(a.confidence >= 90, "Konfidenz war {}", a.confidence);
        let hint = a.schema_hint.expect("schema_hint fehlt");
        assert_eq!(hint["is_split"], serde_json::json!(true));
        assert_eq!(hint["splits"]["train"]["count"], serde_json::json!(3));
        assert!(a.pairing_status.unwrap().is_paired);
    }

    #[test]
    fn pascal_voc_split_wird_als_voc_erkannt() {
        let dir = TempDir::new("voc");
        fs::create_dir_all(dir.path().join("images/train")).unwrap();
        fs::create_dir_all(dir.path().join("annotations/train")).unwrap();
        fs::write(dir.path().join("images/train/a.jpg"), b"x").unwrap();
        fs::write(dir.path().join("annotations/train/a.xml"), b"<annotation/>").unwrap();
        let a = detect_dataset_type(dir.path());
        assert!(matches!(a.detected_type, DatasetType::PascalVoc), "erkannt als {:?}", a.detected_type);
    }

    #[test]
    fn klassennamen_kommen_aus_der_data_yaml() {
        let dir = TempDir::new("names");
        build_nested(dir.path());
        assert_eq!(read_class_names(dir.path()), vec!["Tree", "Stone", "Person"]);
    }

    #[test]
    fn yaml_namen_in_allen_drei_schreibweisen() {
        assert_eq!(parse_yaml_class_names("nc: 2\nnames: ['a', \"b\"]\n"), vec!["a", "b"]);
        assert_eq!(parse_yaml_class_names("names:\n  - 'a'\n  - b\nnc: 2\n"), vec!["a", "b"]);
        assert_eq!(parse_yaml_class_names("names:\n  1: b\n  0: a\n"), vec!["a", "b"]);
        // Ein Kommentar hinter dem Key darf nicht als Inline-Liste durchgehen.
        assert_eq!(parse_yaml_class_names("names:  # Klassen\n  - 'a'\n"), vec!["a"]);
        assert!(parse_yaml_class_names("train: images/train\nnc: 0\n").is_empty());
    }

    #[test]
    fn generierte_yaml_laesst_leere_splits_weg() {
        let dir = TempDir::new("yaml");
        build_nested(dir.path());
        let layout = detect_split_layout(dir.path()).unwrap();
        generate_split_dataset_yaml(dir.path(), &layout).unwrap();
        let yaml = fs::read_to_string(dir.path().join("dataset.yaml")).unwrap();
        assert!(yaml.contains("train: images/train"), "{}", yaml);
        assert!(yaml.contains("val: images/val"), "{}", yaml);
        // images/test ist leer – ein test-Eintrag laesst Ultralytics abbrechen.
        assert!(!yaml.contains("test:"), "{}", yaml);
        assert!(yaml.contains("nc: 3"), "{}", yaml);
        assert!(yaml.contains("- 'Tree'"), "{}", yaml);
    }

    #[test]
    fn ohne_val_split_zeigt_val_auf_train() {
        let dir = TempDir::new("noval");
        make_pair(&dir.path().join("images/train"), &dir.path().join("labels/train"), "a");
        let layout = detect_split_layout(dir.path()).unwrap();
        generate_split_dataset_yaml(dir.path(), &layout).unwrap();
        let yaml = fs::read_to_string(dir.path().join("dataset.yaml")).unwrap();
        assert!(yaml.contains("val: images/train"), "{}", yaml);
    }

    #[test]
    fn hilfsdateien_werden_vom_split_ausgenommen() {
        assert!(is_auxiliary_file(Path::new("/x/data.yaml")));
        assert!(is_auxiliary_file(Path::new("/x/shuffel.py")));
        assert!(is_auxiliary_file(Path::new("/x/README.md")));
        assert!(!is_auxiliary_file(Path::new("/x/train.csv")));
        assert!(!is_auxiliary_file(Path::new("/x/bild.jpg")));
    }

    #[test]
    fn flache_datasets_bleiben_unveraendert() {
        // Regression: die neue Split-Erkennung darf normale Ordner nicht kapern.
        let dir = TempDir::new("flat");
        fs::write(dir.path().join("train.csv"), b"a,b\n1,2\n").unwrap();
        assert!(detect_split_layout(dir.path()).is_none());
        let a = detect_dataset_type(dir.path());
        assert!(matches!(a.detected_type, DatasetType::FlatFile), "erkannt als {:?}", a.detected_type);
    }

    #[test]
    fn ordner_klassifikation_bleibt_erhalten() {
        // katzen/ und hunde/ heissen nicht train/val – darf kein Split-Layout werden.
        let dir = TempDir::new("folderclass");
        for cls in ["katzen", "hunde"] {
            fs::create_dir_all(dir.path().join(cls)).unwrap();
            fs::write(dir.path().join(cls).join("a.jpg"), b"x").unwrap();
        }
        assert!(detect_split_layout(dir.path()).is_none());
        let a = detect_dataset_type(dir.path());
        assert!(matches!(a.detected_type, DatasetType::FolderClass), "erkannt als {:?}", a.detected_type);
    }
}

#[cfg(test)]
mod yaml_comment_tests {
    // Der Editor las "images/train  # 463 Bilder" komplett als Pfad ein und
    // schrieb ihn beim Speichern zurueck. Der Parser sitzt in get_dataset_yaml
    // (async, braucht AppHandle) — hier wird die Kommentar-Regel selbst geprueft.
    fn strip_comment(s: &str) -> String {
        match s.find(" #").or_else(|| if s.starts_with('#') { Some(0) } else { None }) {
            Some(i) => s[..i].trim_end().to_string(),
            None => s.to_string(),
        }
    }

    #[test]
    fn zeilenkommentar_wird_abgeschnitten() {
        assert_eq!(strip_comment("train: images/train  # 463 Bilder"), "train: images/train");
        assert_eq!(strip_comment("val: images/val  # relativer Pfad"), "val: images/val");
        assert_eq!(strip_comment("nc: 13"), "nc: 13");
    }

    #[test]
    fn raute_ohne_leerzeichen_bleibt_teil_des_werts() {
        // Dateinamen duerfen ein '#' enthalten; nur " #" leitet einen Kommentar ein.
        assert_eq!(strip_comment("train: bilder#1/train"), "train: bilder#1/train");
    }

    #[test]
    fn ganze_kommentarzeile_wird_leer() {
        assert_eq!(strip_comment("# nur ein Kommentar"), "");
    }
}
