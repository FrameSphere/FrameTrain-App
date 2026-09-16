// studio_manager.rs – Dataset Studio: Projekte, Samples, Export
//
// ARCHITEKTUR (ausfuehrlich in DATASET_STUDIO_PLAN.md):
//   <app_data>/datasets/<user_id>/_studio/<project_id>/
//     project.json     Name, Modalitaet, Klassen, Zielformat
//     media/ab/<hash>  inhaltsadressiert (sha256) — gleiches Bild, gleicher Pfad
//     samples.jsonl    eine Zeile je Sample (Basiszustand)
//     events.jsonl     jede Annotationsaenderung, angehaengt statt neu geschrieben
//     exports/         erzeugte Datensaetze
//
// Warum zwei Dateien statt einer: beim Labeln folgt Bestaetigung auf
// Bestaetigung. Wuerde jede davon samples.jsonl neu schreiben, kostet ein
// einziger Tastendruck bei 20 000 Samples mehrere Megabyte Schreiblast. Das
// Event wird angehaengt (O(1)); der Basiszustand wird nur beim Kompaktieren
// neu geschrieben. Stuerzt die App dazwischen ab, ist kein Label verloren.
//
// Das Studio kennt YOLO nur an einer Stelle: im Export. Intern liegen Boxen
// normalisiert (Mittelpunkt + Groesse, 0..1). Ein zweites Zielformat braucht
// deshalb einen neuen Writer und sonst nichts.

use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use chrono::Utc;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::{Emitter, Manager, State};

use crate::AppState;

// ══════════════════════════════════════════════════════════════════
// TYPEN
// ══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudioProject {
    pub id:            String,
    pub name:          String,
    /// "image" — Text und Audio folgen in spaeteren Phasen.
    pub modality:      String,
    /// "bbox"
    pub task:          String,
    /// Zielformat des Exports, z.B. "yolo_bbox".
    pub target_format: String,
    pub classes:       Vec<String>,
    pub created_at:    String,
    pub updated_at:    String,
    /// Live beim Auflisten gezaehlt, nie gespeichert — gespeicherte Zaehler
    /// laufen frueher oder spaeter aus dem Tritt mit dem echten Inhalt.
    #[serde(default, skip_serializing)]
    pub sample_count:    usize,
    #[serde(default, skip_serializing)]
    pub confirmed_count: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct BoxAnn {
    pub cls: usize,
    /// Mittelpunkt und Groesse, normalisiert auf 0..1 (YOLO-Konvention).
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Annotation {
    #[serde(default)]
    pub boxes: Vec<BoxAnn>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleSource {
    /// "import" | "record" | "web" | "create"
    pub kind:    String,
    /// Urspruenglicher Pfad oder URL — die Herkunft ist Pflichtfeld, damit der
    /// Export eine belastbare PROVENANCE.csv schreiben kann.
    #[serde(default)]
    pub origin:  Option<String>,
    #[serde(default)]
    pub license: Option<String>,
    pub at:      String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SampleMeta {
    #[serde(default)]
    pub w: u32,
    #[serde(default)]
    pub h: u32,
    /// Video, Sprecher, Aufnahmesession, Quelldomain. Noch nicht ausgewertet —
    /// der gruppenbewusste Split kommt in einer spaeteren Phase, das Feld wird
    /// aber ab jetzt mitgeschrieben, damit alte Projekte ihn spaeter bekommen.
    #[serde(default)]
    pub group: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudioSample {
    pub id:     String,
    /// Pfad relativ zu media/, z.B. "ab/ab3f….jpg".
    pub media:  String,
    pub mime:   String,
    /// "new" | "suggested" | "confirmed" | "skipped"
    pub status: String,
    #[serde(default)]
    pub ann:    Annotation,
    pub src:    SampleSource,
    #[serde(default)]
    pub meta:   SampleMeta,
    /// Absoluter Pfad — nur fuer das Frontend, nicht gespeichert.
    #[serde(default, skip_deserializing)]
    pub abs_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnnEvent {
    sample_id: String,
    status:    String,
    boxes:     Vec<BoxAnn>,
    at:        String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImportReport {
    pub added:            usize,
    pub duplicates:       usize,
    pub unreadable:       usize,
    pub with_labels:      usize,
    pub classes_added:    Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SamplePage {
    pub total: usize,
    pub items: Vec<StudioSample>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudioStats {
    pub total:       usize,
    pub new:         usize,
    pub suggested:   usize,
    pub confirmed:   usize,
    pub skipped:     usize,
    pub boxes_total: usize,
    /// Boxen je Klasse, gleiche Reihenfolge wie project.classes.
    pub per_class:   Vec<usize>,
    pub empty_confirmed: usize,
}

// ══════════════════════════════════════════════════════════════════
// PFADE
// ══════════════════════════════════════════════════════════════════

fn get_user_id(state: &State<'_, AppState>) -> Result<String, String> {
    let db = state.db.lock().map_err(|e| format!("DB lock: {}", e))?;
    db.get_current_user_id().ok_or_else(|| "Kein Benutzer angemeldet".to_string())
}

fn sanitize_user_id(user_id: &str) -> String {
    user_id.chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}

fn studio_dir(app_handle: &tauri::AppHandle, user_id: &str) -> Result<PathBuf, String> {
    let dir = app_handle.path().app_data_dir()
        .map_err(|e| format!("AppDataDir: {}", e))?
        .join("datasets")
        .join(sanitize_user_id(user_id))
        .join("_studio");
    fs::create_dir_all(&dir).map_err(|e| format!("mkdir _studio: {}", e))?;
    Ok(dir)
}

fn project_dir(app_handle: &tauri::AppHandle, user_id: &str, project_id: &str) -> Result<PathBuf, String> {
    if !crate::model_manager::is_safe_id(project_id) {
        return Err("Ungueltige Projekt-ID".to_string());
    }
    let dir = studio_dir(app_handle, user_id)?.join(project_id);
    if !dir.exists() { return Err(format!("Projekt nicht gefunden: {}", project_id)); }
    Ok(dir)
}

// ══════════════════════════════════════════════════════════════════
// JSONL
// ══════════════════════════════════════════════════════════════════

fn read_jsonl<T: DeserializeOwned>(path: &Path) -> Vec<T> {
    let Ok(file) = fs::File::open(path) else { return vec![]; };
    BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter(|l| !l.trim().is_empty())
        // Eine kaputte Zeile (halber Schreibvorgang bei Stromausfall) darf nicht
        // das ganze Projekt unlesbar machen.
        .filter_map(|l| serde_json::from_str::<T>(&l).ok())
        .collect()
}

fn append_jsonl<T: Serialize>(path: &Path, item: &T) -> Result<(), String> {
    let line = serde_json::to_string(item).map_err(|e| format!("JSON: {}", e))?;
    let mut f = OpenOptions::new().create(true).append(true).open(path)
        .map_err(|e| format!("Oeffnen {}: {}", path.display(), e))?;
    writeln!(f, "{}", line).map_err(|e| format!("Schreiben: {}", e))
}

fn write_jsonl<T: Serialize>(path: &Path, items: &[T]) -> Result<(), String> {
    let mut out = String::new();
    for it in items {
        out.push_str(&serde_json::to_string(it).map_err(|e| format!("JSON: {}", e))?);
        out.push('\n');
    }
    let tmp = path.with_extension("jsonl.tmp");
    fs::write(&tmp, out).map_err(|e| format!("Schreiben: {}", e))?;
    fs::rename(&tmp, path).map_err(|e| format!("Ersetzen: {}", e))
}

fn samples_path(dir: &Path) -> PathBuf { dir.join("samples.jsonl") }
fn events_path(dir: &Path)  -> PathBuf { dir.join("events.jsonl") }

/// Basiszustand plus alle angehaengten Aenderungen.
fn load_samples(dir: &Path) -> Vec<StudioSample> {
    let mut samples: Vec<StudioSample> = read_jsonl(&samples_path(dir));
    let events: Vec<AnnEvent> = read_jsonl(&events_path(dir));
    if events.is_empty() { return samples; }
    let index: HashMap<String, usize> = samples.iter().enumerate()
        .map(|(i, s)| (s.id.clone(), i)).collect();
    for ev in events {
        if let Some(&i) = index.get(&ev.sample_id) {
            samples[i].status = ev.status;
            samples[i].ann.boxes = ev.boxes;
        }
    }
    samples
}

/// Ab dieser Zahl angehaengter Events lohnt das Neuschreiben des Basiszustands.
const COMPACT_AFTER_EVENTS: usize = 2000;

fn maybe_compact(dir: &Path) -> Result<(), String> {
    let ev_path = events_path(dir);
    let Ok(meta) = fs::metadata(&ev_path) else { return Ok(()); };
    // Grobe Schaetzung ueber die Dateigroesse — eine Ereigniszeile liegt bei
    // etwa 120 Byte. Zeilen zaehlen hiesse die Datei bei jedem Tastendruck
    // vollstaendig zu lesen, genau das soll das Anhaengen ja vermeiden.
    if meta.len() < (COMPACT_AFTER_EVENTS as u64) * 120 { return Ok(()); }
    let samples = load_samples(dir);
    write_jsonl(&samples_path(dir), &samples)?;
    fs::write(&ev_path, b"").map_err(|e| format!("Events leeren: {}", e))?;
    Ok(())
}

// ══════════════════════════════════════════════════════════════════
// BILDER: MASSE AUS DEM DATEIKOPF
// ══════════════════════════════════════════════════════════════════

fn be16(b: &[u8]) -> u16 { u16::from_be_bytes([b[0], b[1]]) }
fn be32(b: &[u8]) -> u32 { u32::from_be_bytes([b[0], b[1], b[2], b[3]]) }
fn le16(b: &[u8]) -> u16 { u16::from_le_bytes([b[0], b[1]]) }
fn le32(b: &[u8]) -> u32 { u32::from_le_bytes([b[0], b[1], b[2], b[3]]) }
fn le24(b: &[u8]) -> u32 { u32::from_le_bytes([b[0], b[1], b[2], 0]) }

/// Breite und Hoehe aus dem Dateikopf, ohne das Bild zu dekodieren.
///
/// Die Masse werden fuer die Box-Anzeige gebraucht (normalisierte Boxen mal
/// Bildgroesse). Eine Bildbibliothek dafuer einzubinden waere teuer: sie muss
/// jedes Bild vollstaendig einlesen, und das Studio sieht beim Import zehn-
/// tausende. Der Kopf reicht und kostet ein paar Byte.
pub fn image_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.len() < 16 { return None; }

    if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        if bytes.len() >= 24 && &bytes[12..16] == b"IHDR" {
            return Some((be32(&bytes[16..20]), be32(&bytes[20..24])));
        }
        return None;
    }

    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return Some((le16(&bytes[6..8]) as u32, le16(&bytes[8..10]) as u32));
    }

    if bytes.starts_with(b"BM") && bytes.len() >= 26 {
        let w = le32(&bytes[18..22]) as i32;
        let h = le32(&bytes[22..26]) as i32;
        return Some((w.unsigned_abs(), h.unsigned_abs()));
    }

    if bytes.len() >= 30 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        return match &bytes[12..16] {
            b"VP8X" => Some((le24(&bytes[24..27]) + 1, le24(&bytes[27..30]) + 1)),
            b"VP8L" if bytes[20] == 0x2F => {
                let bits = le32(&bytes[21..25]);
                Some(((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1))
            }
            b"VP8 " if bytes[23] == 0x9D && bytes[24] == 0x01 && bytes[25] == 0x2A => {
                Some(((le16(&bytes[26..28]) & 0x3FFF) as u32, (le16(&bytes[28..30]) & 0x3FFF) as u32))
            }
            _ => None,
        };
    }

    if bytes.starts_with(&[0xFF, 0xD8]) {
        let mut i = 2usize;
        while i + 9 < bytes.len() {
            if bytes[i] != 0xFF { i += 1; continue; }
            let marker = bytes[i + 1];
            // Fuellbytes und Marker ohne Nutzlast ueberspringen.
            if marker == 0xFF { i += 1; continue; }
            if marker == 0x01 || (0xD0..=0xD9).contains(&marker) { i += 2; continue; }
            let len = be16(&bytes[i + 2..i + 4]) as usize;
            let is_sof = matches!(marker, 0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF);
            if is_sof {
                return Some((be16(&bytes[i + 7..i + 9]) as u32, be16(&bytes[i + 5..i + 7]) as u32));
            }
            if len < 2 { return None; }
            i += 2 + len;
        }
    }
    None
}

// ══════════════════════════════════════════════════════════════════
// YOLO: LESEN UND SCHREIBEN
// ══════════════════════════════════════════════════════════════════

fn clamp01(v: f64) -> f64 { v.clamp(0.0, 1.0) }

/// Liest eine YOLO-Labeldatei. Segmentierungszeilen (Polygone) bekommen die
/// umschliessende Box — dasselbe Verhalten wie im Labor, damit ein Dataset
/// nicht je nach Ansicht anders aussieht.
pub fn parse_yolo_label(text: &str) -> Vec<BoxAnn> {
    let mut out = Vec::new();
    for line in text.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 5 { continue; }
        let Ok(cls) = parts[0].parse::<f64>() else { continue; };
        let nums: Vec<f64> = parts[1..].iter().filter_map(|p| p.parse::<f64>().ok()).collect();
        if nums.len() != parts.len() - 1 { continue; }
        let cls = cls as usize;
        if nums.len() == 4 {
            out.push(BoxAnn { cls, x: nums[0], y: nums[1], w: nums[2], h: nums[3] });
        } else if nums.len() >= 6 && nums.len() % 2 == 0 {
            let xs: Vec<f64> = nums.iter().step_by(2).copied().collect();
            let ys: Vec<f64> = nums.iter().skip(1).step_by(2).copied().collect();
            let (x0, x1) = (xs.iter().cloned().fold(f64::MAX, f64::min), xs.iter().cloned().fold(f64::MIN, f64::max));
            let (y0, y1) = (ys.iter().cloned().fold(f64::MAX, f64::min), ys.iter().cloned().fold(f64::MIN, f64::max));
            out.push(BoxAnn { cls, x: (x0 + x1) / 2.0, y: (y0 + y1) / 2.0, w: x1 - x0, h: y1 - y0 });
        }
    }
    out
}

/// Moegliche Labeldateien zu einem Bild — Ultralytics legt sie in einem
/// Geschwisterordner "labels" ab, manche Datensaetze direkt daneben.
pub fn label_paths_for_image(image: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    out.push(image.with_extension("txt"));
    let comps: Vec<String> = image.components()
        .map(|c| c.as_os_str().to_string_lossy().to_string()).collect();
    if let Some(pos) = comps.iter().rposition(|c| c.eq_ignore_ascii_case("images")) {
        let mut swapped = comps.clone();
        swapped[pos] = "labels".to_string();
        let mut p = PathBuf::from(swapped.join(std::path::MAIN_SEPARATOR_STR));
        p.set_extension("txt");
        out.push(p);
    }
    out
}

/// Klassennamen aus classes.txt / obj.names / data.yaml des Quellordners.
/// FrameTrain haelt keine Klassenliste vor — sie kommt immer aus den Daten.
pub fn read_source_classes(dir: &Path) -> Vec<String> {
    for name in ["classes.txt", "obj.names"] {
        if let Ok(text) = fs::read_to_string(dir.join(name)) {
            let names: Vec<String> = text.lines().map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty()).collect();
            if !names.is_empty() { return names; }
        }
    }
    for name in ["data.yaml", "dataset.yaml"] {
        if let Ok(text) = fs::read_to_string(dir.join(name)) {
            let names = class_names_from_yaml(&text);
            if !names.is_empty() { return names; }
        }
    }
    vec![]
}

/// names aus einer data.yaml — sowohl `names: [a, b]` als auch die Listen- und
/// Mapping-Schreibweise ueber mehrere Zeilen.
pub fn class_names_from_yaml(text: &str) -> Vec<String> {
    let mut map: HashMap<usize, String> = HashMap::new();
    let mut seq: Vec<String> = Vec::new();
    let mut in_names = false;
    for raw in text.lines() {
        let line = raw.trim_end();
        let trimmed = line.trim();
        if trimmed.starts_with('#') { continue; }
        if let Some(rest) = trimmed.strip_prefix("names:") {
            in_names = true;
            let rest = rest.trim();
            if rest.starts_with('[') {
                let inner = rest.trim_start_matches('[').trim_end_matches(']');
                for part in inner.split(',') {
                    let v = part.trim().trim_matches(|c| c == '\'' || c == '"').to_string();
                    if !v.is_empty() { seq.push(v); }
                }
                in_names = false;
            }
            continue;
        }
        if !in_names { continue; }
        // Ende des Blocks: eine Zeile ohne Einrueckung, die nicht zur Liste gehoert.
        if !line.starts_with(' ') && !line.starts_with('-') && !trimmed.is_empty() { break; }
        if let Some(item) = trimmed.strip_prefix("- ") {
            seq.push(item.trim().trim_matches(|c| c == '\'' || c == '"').to_string());
            continue;
        }
        if let Some((k, v)) = trimmed.split_once(':') {
            if let Ok(idx) = k.trim().parse::<usize>() {
                map.insert(idx, v.trim().trim_matches(|c| c == '\'' || c == '"').to_string());
            }
        }
    }
    if !map.is_empty() {
        let max = map.keys().max().copied().unwrap_or(0);
        return (0..=max).map(|i| map.get(&i).cloned().unwrap_or_else(|| format!("Klasse {}", i))).collect();
    }
    seq
}

// ══════════════════════════════════════════════════════════════════
// PROJEKTE
// ══════════════════════════════════════════════════════════════════

fn load_project(dir: &Path) -> Result<StudioProject, String> {
    let text = fs::read_to_string(dir.join("project.json"))
        .map_err(|e| format!("project.json: {}", e))?;
    serde_json::from_str(&text).map_err(|e| format!("project.json unlesbar: {}", e))
}

fn save_project(dir: &Path, p: &StudioProject) -> Result<(), String> {
    fs::write(dir.join("project.json"),
        serde_json::to_string_pretty(p).map_err(|e| format!("JSON: {}", e))?)
        .map_err(|e| format!("project.json schreiben: {}", e))
}

#[tauri::command]
pub async fn studio_list_projects(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
) -> Result<Vec<StudioProject>, String> {
    let user_id = get_user_id(&state)?;
    let root = studio_dir(&app_handle, &user_id)?;
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(&root) else { return Ok(out); };
    for entry in entries.flatten() {
        let dir = entry.path();
        if !dir.is_dir() { continue; }
        let Ok(mut project) = load_project(&dir) else { continue; };
        let samples = load_samples(&dir);
        project.sample_count = samples.len();
        project.confirmed_count = samples.iter().filter(|s| s.status == "confirmed").count();
        out.push(project);
    }
    out.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    Ok(out)
}

#[tauri::command]
pub async fn studio_create_project(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    name: String, modality: String, target_format: String, classes: Vec<String>,
) -> Result<StudioProject, String> {
    let name = name.trim().to_string();
    if name.is_empty() { return Err("Name fehlt".to_string()); }
    let user_id = get_user_id(&state)?;
    let root = studio_dir(&app_handle, &user_id)?;
    let id = format!("sp_{}", &uuid::Uuid::new_v4().to_string().replace('-', "")[..12]);
    let dir = root.join(&id);
    fs::create_dir_all(dir.join("media")).map_err(|e| format!("mkdir: {}", e))?;
    fs::create_dir_all(dir.join("exports")).map_err(|e| format!("mkdir: {}", e))?;
    let now = Utc::now().to_rfc3339();
    let project = StudioProject {
        id, name, modality, task: "bbox".to_string(), target_format,
        classes: classes.into_iter().map(|c| c.trim().to_string()).filter(|c| !c.is_empty()).collect(),
        created_at: now.clone(), updated_at: now,
        sample_count: 0, confirmed_count: 0,
    };
    save_project(&dir, &project)?;
    Ok(project)
}

#[tauri::command]
pub async fn studio_update_project(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    project_id: String, name: Option<String>, classes: Option<Vec<String>>,
) -> Result<StudioProject, String> {
    let user_id = get_user_id(&state)?;
    let dir = project_dir(&app_handle, &user_id, &project_id)?;
    let mut project = load_project(&dir)?;
    if let Some(n) = name {
        let n = n.trim().to_string();
        if n.is_empty() { return Err("Name fehlt".to_string()); }
        project.name = n;
    }
    if let Some(c) = classes {
        let cleaned: Vec<String> = c.into_iter().map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect();
        // Eine Klasse zu entfernen wuerde alle Boxen dahinter auf die falsche
        // Klasse schieben, weil YOLO ueber den Index geht. Daher nur anhaengen
        // und umbenennen, nie kuerzen.
        if cleaned.len() < project.classes.len() {
            return Err("Klassen koennen umbenannt und ergaenzt, aber nicht entfernt werden".to_string());
        }
        project.classes = cleaned;
    }
    project.updated_at = Utc::now().to_rfc3339();
    save_project(&dir, &project)?;
    Ok(project)
}

#[tauri::command]
pub async fn studio_delete_project(
    app_handle: tauri::AppHandle, state: State<'_, AppState>, project_id: String,
) -> Result<(), String> {
    let user_id = get_user_id(&state)?;
    let dir = project_dir(&app_handle, &user_id, &project_id)?;
    fs::remove_dir_all(&dir).map_err(|e| format!("Loeschen: {}", e))
}

// ══════════════════════════════════════════════════════════════════
// IMPORT
// ══════════════════════════════════════════════════════════════════

const IMAGE_EXTS: &[&str] = &["jpg", "jpeg", "png", "bmp", "webp", "gif", "tif", "tiff"];

fn mime_for(ext: &str) -> String {
    match ext {
        "jpg" | "jpeg" => "image/jpeg",
        "png"          => "image/png",
        "bmp"          => "image/bmp",
        "webp"         => "image/webp",
        "gif"          => "image/gif",
        "tif" | "tiff" => "image/tiff",
        _              => "application/octet-stream",
    }.to_string()
}

fn collect_images(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else { continue; };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if name.starts_with('.') { continue; }
                stack.push(p);
            } else {
                let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
                if IMAGE_EXTS.contains(&ext.as_str()) { out.push(p); }
            }
        }
    }
    out.sort();
    out
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[tauri::command]
pub async fn studio_import_folder(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    project_id: String, source_path: String,
) -> Result<ImportReport, String> {
    let user_id = get_user_id(&state)?;
    let dir = project_dir(&app_handle, &user_id, &project_id)?;
    let mut project = load_project(&dir)?;
    let src = Path::new(&source_path);
    if !src.is_dir() { return Err(format!("Ordner nicht gefunden: {}", source_path)); }

    let files = collect_images(src);
    if files.is_empty() { return Err("Keine Bilder in diesem Ordner gefunden".to_string()); }

    let existing = load_samples(&dir);
    let mut known: std::collections::HashSet<String> =
        existing.iter().map(|s| s.media.clone()).collect();

    // Klassennamen aus dem Quellordner uebernehmen, solange das Projekt noch
    // keine hat — sonst stehen im Editor Zahlen statt Namen.
    let source_classes = read_source_classes(src);
    let mut classes_added: Vec<String> = Vec::new();
    if project.classes.is_empty() && !source_classes.is_empty() {
        project.classes = source_classes.clone();
        classes_added = source_classes.clone();
    }

    let mut report = ImportReport { added: 0, duplicates: 0, unreadable: 0, with_labels: 0, classes_added: vec![] };
    let now = Utc::now().to_rfc3339();
    let total = files.len();

    for (i, file) in files.iter().enumerate() {
        if i % 25 == 0 {
            let _ = app_handle.emit("studio-import-progress", serde_json::json!({
                "project_id": project_id, "current": i, "total": total,
            }));
        }
        let Ok(bytes) = fs::read(file) else { report.unreadable += 1; continue; };
        let Some((w, h)) = image_dimensions(&bytes) else { report.unreadable += 1; continue; };
        let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("bin").to_lowercase();
        let hash = sha256_hex(&bytes);
        let rel = format!("{}/{}.{}", &hash[..2], &hash, ext);
        if known.contains(&rel) { report.duplicates += 1; continue; }

        let target = dir.join("media").join(&rel);
        if let Some(parent) = target.parent() { fs::create_dir_all(parent).ok(); }
        if !target.exists() {
            fs::write(&target, &bytes).map_err(|e| format!("Kopieren: {}", e))?;
        }

        // Liegt eine YOLO-Labeldatei daneben, ist das Bild bereits gelabelt.
        // Ein halb fertiges Dataset laesst sich damit weiterbearbeiten, statt
        // bei null anzufangen.
        let mut boxes: Vec<BoxAnn> = Vec::new();
        for lp in label_paths_for_image(file) {
            if let Ok(text) = fs::read_to_string(&lp) {
                boxes = parse_yolo_label(&text);
                break;
            }
        }
        let has_labels = !boxes.is_empty();
        if has_labels { report.with_labels += 1; }

        // Klassen-IDs, fuer die es noch keinen Namen gibt, bekommen einen
        // Platzhalter — sonst zeigt der Editor eine Box ohne Klasse.
        if let Some(max_cls) = boxes.iter().map(|b| b.cls).max() {
            while project.classes.len() <= max_cls {
                let name = source_classes.get(project.classes.len()).cloned()
                    .unwrap_or_else(|| format!("Klasse {}", project.classes.len()));
                classes_added.push(name.clone());
                project.classes.push(name);
            }
        }

        let sample = StudioSample {
            id: format!("s_{}", &uuid::Uuid::new_v4().to_string().replace('-', "")[..10]),
            media: rel.clone(),
            mime: mime_for(&ext),
            status: if has_labels { "confirmed".to_string() } else { "new".to_string() },
            ann: Annotation { boxes },
            src: SampleSource {
                kind: "import".to_string(),
                origin: Some(file.to_string_lossy().to_string()),
                license: None,
                at: now.clone(),
            },
            meta: SampleMeta { w, h, group: None },
            abs_path: String::new(),
        };
        append_jsonl(&samples_path(&dir), &sample)?;
        known.insert(rel);
        report.added += 1;
    }

    report.classes_added = classes_added;
    project.updated_at = Utc::now().to_rfc3339();
    save_project(&dir, &project)?;
    let _ = app_handle.emit("studio-import-progress", serde_json::json!({
        "project_id": project_id, "current": total, "total": total, "done": true,
    }));
    Ok(report)
}

// ══════════════════════════════════════════════════════════════════
// SAMPLES
// ══════════════════════════════════════════════════════════════════

#[tauri::command]
pub async fn studio_list_samples(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    project_id: String, status: Option<String>, offset: usize, limit: usize,
) -> Result<SamplePage, String> {
    let user_id = get_user_id(&state)?;
    let dir = project_dir(&app_handle, &user_id, &project_id)?;
    let media_dir = dir.join("media");
    let all = load_samples(&dir);
    let filtered: Vec<&StudioSample> = match status.as_deref() {
        None | Some("") | Some("all") => all.iter().collect(),
        Some("open") => all.iter().filter(|s| s.status == "new" || s.status == "suggested").collect(),
        Some(st) => all.iter().filter(|s| s.status == st).collect(),
    };
    let total = filtered.len();
    let items = filtered.into_iter().skip(offset).take(limit.clamp(1, 500))
        .map(|s| {
            let mut c = s.clone();
            c.abs_path = media_dir.join(&s.media).to_string_lossy().to_string();
            c
        })
        .collect();
    Ok(SamplePage { total, items })
}

#[tauri::command]
pub async fn studio_set_annotation(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    project_id: String, sample_id: String, boxes: Vec<BoxAnn>, status: String,
) -> Result<(), String> {
    if !matches!(status.as_str(), "new" | "suggested" | "confirmed" | "skipped") {
        return Err(format!("Unbekannter Status: {}", status));
    }
    let user_id = get_user_id(&state)?;
    let dir = project_dir(&app_handle, &user_id, &project_id)?;
    let event = AnnEvent {
        sample_id, status,
        boxes: boxes.into_iter()
            .map(|b| BoxAnn { cls: b.cls, x: clamp01(b.x), y: clamp01(b.y), w: clamp01(b.w), h: clamp01(b.h) })
            .filter(|b| b.w > 0.0005 && b.h > 0.0005)
            .collect(),
        at: Utc::now().to_rfc3339(),
    };
    append_jsonl(&events_path(&dir), &event)?;
    maybe_compact(&dir)?;
    Ok(())
}

#[tauri::command]
pub async fn studio_stats(
    app_handle: tauri::AppHandle, state: State<'_, AppState>, project_id: String,
) -> Result<StudioStats, String> {
    let user_id = get_user_id(&state)?;
    let dir = project_dir(&app_handle, &user_id, &project_id)?;
    let project = load_project(&dir)?;
    let samples = load_samples(&dir);
    let mut stats = StudioStats {
        total: samples.len(), new: 0, suggested: 0, confirmed: 0, skipped: 0,
        boxes_total: 0, per_class: vec![0; project.classes.len()], empty_confirmed: 0,
    };
    for s in &samples {
        match s.status.as_str() {
            "new"       => stats.new += 1,
            "suggested" => stats.suggested += 1,
            "confirmed" => stats.confirmed += 1,
            "skipped"   => stats.skipped += 1,
            _ => {}
        }
        if s.status == "confirmed" && s.ann.boxes.is_empty() { stats.empty_confirmed += 1; }
        for b in &s.ann.boxes {
            stats.boxes_total += 1;
            if b.cls < stats.per_class.len() { stats.per_class[b.cls] += 1; }
        }
    }
    Ok(stats)
}

// ══════════════════════════════════════════════════════════════════
// EXPORT
// ══════════════════════════════════════════════════════════════════

fn csv_field(v: &str) -> String {
    if v.contains(',') || v.contains('"') || v.contains('\n') {
        format!("\"{}\"", v.replace('"', "\"\""))
    } else {
        v.to_string()
    }
}

/// Schreibt das Projekt als ungeteiltes YOLO-Dataset. Bewusst ohne Split: die
/// App teilt typbewusst und paarerhaltend bereits auf, ein zweiter Splitter im
/// Studio waere eine Kopie mit eigenen Fehlern. Der gruppenbewusste Split
/// kommt dort, nicht hier.
///
/// images/, labels/ und classes.txt schreibt `yolo_export` — derselbe Weg, den
/// auch der Korrektur-Export des Labors nimmt. Studio-eigen sind nur die
/// beiden Beipackzettel darunter.
fn write_yolo_export(
    project: &StudioProject, samples: &[&StudioSample], media_dir: &Path, out: &Path,
) -> Result<(), String> {
    let items: Vec<crate::yolo_export::ExportItem> = samples.iter()
        .map(|s| crate::yolo_export::ExportItem {
            source: media_dir.join(&s.media),
            stem:   s.id.clone(),
            boxes:  s.ann.boxes.iter()
                .filter_map(|b| crate::yolo_export::norm_box(b.cls, b.x, b.y, b.w, b.h))
                .collect(),
        })
        .collect();
    let file_names = crate::yolo_export::write_yolo_layout(out, &items, &project.classes)?;

    let mut provenance = String::from("sample_id,datei,herkunft,lizenz,status,boxen\n");
    for ((s, item), file_name) in samples.iter().zip(items.iter()).zip(file_names.iter()) {
        provenance.push_str(&format!("{},{},{},{},{},{}\n",
            csv_field(&s.id),
            csv_field(file_name),
            csv_field(s.src.origin.as_deref().unwrap_or("")),
            csv_field(s.src.license.as_deref().unwrap_or("")),
            csv_field(&s.status),
            item.boxes.len()));
    }
    fs::write(out.join("PROVENANCE.csv"), provenance)
        .map_err(|e| format!("PROVENANCE.csv: {}", e))?;

    let confirmed = samples.iter().filter(|s| s.status == "confirmed").count();
    let suggested = samples.len() - confirmed;
    // Dieselbe Quelle wie PROVENANCE.csv: eine entartete Box zaehlt in
    // beiden Dateien nicht mit, sonst widersprechen sich die Zahlen.
    let boxes: usize = items.iter().map(|i| i.boxes.len()).sum();
    let card = format!(
"# {name}

Erzeugt vom FrameTrain Dataset Studio am {date}.

- Bilder: {count}
- davon bestaetigt: {confirmed}
- davon uebernommene Vorschlaege: {suggested}
- Boxen insgesamt: {boxes}
- Klassen: {classes}

Aufteilung in train/val/test ist noch nicht erfolgt — dafuer den Split im
Dataset-Bereich nutzen, er haelt Bild- und Labelpaare zusammen.

Herkunft je Bild steht in PROVENANCE.csv.
",
        name = project.name,
        date = Utc::now().format("%Y-%m-%d"),
        count = samples.len(),
        confirmed = confirmed,
        suggested = suggested,
        boxes = boxes,
        classes = project.classes.join(", "));
    fs::write(out.join("DATA_CARD.md"), card).map_err(|e| format!("DATA_CARD.md: {}", e))?;
    Ok(())
}

#[tauri::command]
pub async fn studio_export(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    project_id: String, model_id: String, dataset_name: String, include_suggested: bool,
) -> Result<crate::dataset_manager::DatasetInfo, String> {
    let user_id = get_user_id(&state)?;
    let dir = project_dir(&app_handle, &user_id, &project_id)?;
    let project = load_project(&dir)?;
    if project.classes.is_empty() { return Err("Das Projekt hat noch keine Klassen".to_string()); }

    let samples = load_samples(&dir);
    let selected: Vec<&StudioSample> = samples.iter()
        .filter(|s| s.status == "confirmed" || (include_suggested && s.status == "suggested"))
        .collect();
    if selected.is_empty() {
        return Err("Keine bestaetigten Samples — es gibt nichts zu exportieren".to_string());
    }

    let stamp = Utc::now().format("%Y%m%d_%H%M%S").to_string();
    let out = dir.join("exports").join(&stamp);
    fs::create_dir_all(&out).map_err(|e| format!("mkdir export: {}", e))?;
    write_yolo_export(&project, &selected, &dir.join("media"), &out)?;

    let name = if dataset_name.trim().is_empty() { project.name.clone() } else { dataset_name };
    crate::dataset_manager::import_local_dataset(
        app_handle, state, out.to_string_lossy().to_string(), name, model_id,
    ).await
}

// ══════════════════════════════════════════════════════════════════
// TESTS
// ══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_masse_aus_dem_kopf() {
        let mut b = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        b.extend_from_slice(&[0, 0, 0, 13]);
        b.extend_from_slice(b"IHDR");
        b.extend_from_slice(&640u32.to_be_bytes());
        b.extend_from_slice(&480u32.to_be_bytes());
        assert_eq!(image_dimensions(&b), Some((640, 480)));
    }

    #[test]
    fn jpeg_masse_ueberspringt_vorspann() {
        // FFD8, dann ein APP0-Segment das uebersprungen werden muss, dann SOF0.
        let mut b = vec![0xFF, 0xD8];
        b.extend_from_slice(&[0xFF, 0xE0, 0x00, 0x08, 1, 2, 3, 4, 5, 6]);
        b.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x11, 0x08]);
        b.extend_from_slice(&1080u16.to_be_bytes());
        b.extend_from_slice(&1920u16.to_be_bytes());
        b.extend_from_slice(&[0; 8]);
        assert_eq!(image_dimensions(&b), Some((1920, 1080)));
    }

    #[test]
    fn unbekannte_bytes_geben_nichts_zurueck() {
        assert_eq!(image_dimensions(b"nur irgendein textinhalt hier"), None);
        assert_eq!(image_dimensions(b"kurz"), None);
    }

    #[test]
    fn yolo_zeile_wird_begrenzt() {
        let b = BoxAnn { cls: 2, x: 0.5, y: 0.25, w: 1.5, h: 0.1 };
        let n = crate::yolo_export::norm_box(b.cls, b.x, b.y, b.w, b.h).unwrap();
        assert_eq!(crate::yolo_export::label_line(&n), "2 0.500000 0.250000 1.000000 0.100000");
    }

    #[test]
    fn yolo_datei_lesen_inklusive_polygon() {
        let text = "0 0.5 0.5 0.2 0.4\n\
                    # Kommentar wird ignoriert\n\
                    1 0.1 0.1 0.3 0.1 0.3 0.5 0.1 0.5\n\
                    kaputt\n";
        let boxes = parse_yolo_label(text);
        assert_eq!(boxes.len(), 2);
        assert_eq!(boxes[0], BoxAnn { cls: 0, x: 0.5, y: 0.5, w: 0.2, h: 0.4 });
        // Das Polygon bekommt seine umschliessende Box.
        let p = boxes[1];
        assert_eq!(p.cls, 1);
        assert!((p.x - 0.2).abs() < 1e-9, "x war {}", p.x);
        assert!((p.w - 0.2).abs() < 1e-9, "w war {}", p.w);
        assert!((p.h - 0.4).abs() < 1e-9, "h war {}", p.h);
    }

    #[test]
    fn labelpfade_folgen_der_ultralytics_konvention() {
        let paths = label_paths_for_image(Path::new("/daten/train/images/bild_01.jpg"));
        assert!(paths.iter().any(|p| p.ends_with("train/images/bild_01.txt")));
        assert!(paths.iter().any(|p| p.ends_with("train/labels/bild_01.txt")));
    }

    #[test]
    fn klassennamen_aus_yaml_in_allen_schreibweisen() {
        assert_eq!(class_names_from_yaml("names: [Lift, Sky]"), vec!["Lift", "Sky"]);
        assert_eq!(class_names_from_yaml("nc: 2\nnames:\n  - Lift\n  - Sky\n"), vec!["Lift", "Sky"]);
        assert_eq!(class_names_from_yaml("names:\n  0: Lift\n  1: Sky\n"), vec!["Lift", "Sky"]);
        assert!(class_names_from_yaml("path: /daten\n").is_empty());
    }

    #[test]
    fn klassenluecke_im_mapping_bekommt_platzhalter() {
        let names = class_names_from_yaml("names:\n  0: Lift\n  2: Mast\n");
        assert_eq!(names, vec!["Lift", "Klasse 1", "Mast"]);
    }

    /// Eigener Temp-Ordner, damit die Tests nichts im Projekt hinterlassen.
    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("ft_studio_{}_{}", tag, uuid::Uuid::new_v4()));
            fs::create_dir_all(&dir).unwrap();
            TempDir(dir)
        }
        fn path(&self) -> &Path { &self.0 }
    }
    impl Drop for TempDir {
        fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); }
    }

    fn sample(id: &str, status: &str) -> StudioSample {
        StudioSample {
            id: id.to_string(), media: format!("ab/{}.jpg", id), mime: "image/jpeg".to_string(),
            status: status.to_string(), ann: Annotation::default(),
            src: SampleSource { kind: "import".to_string(), origin: Some(format!("/daten/{}.jpg", id)),
                license: None, at: "2026-09-16T08:00:00Z".to_string() },
            meta: SampleMeta { w: 1000, h: 500, group: None },
            abs_path: String::new(),
        }
    }

    #[test]
    fn angehaengte_events_ueberschreiben_den_basiszustand() {
        // Das ist der Kern der Ablage: samples.jsonl bleibt beim Labeln
        // unberuehrt, die Aenderung steht in events.jsonl. Wird beides nicht
        // zusammengefaltet, sieht der Nutzer nach dem Neustart seine Arbeit nicht.
        let dir = TempDir::new("events");
        append_jsonl(&samples_path(dir.path()), &sample("s_1", "new")).unwrap();
        append_jsonl(&samples_path(dir.path()), &sample("s_2", "new")).unwrap();
        append_jsonl(&events_path(dir.path()), &AnnEvent {
            sample_id: "s_2".to_string(), status: "confirmed".to_string(),
            boxes: vec![BoxAnn { cls: 1, x: 0.5, y: 0.5, w: 0.2, h: 0.2 }],
            at: "2026-09-16T09:00:00Z".to_string(),
        }).unwrap();

        let loaded = load_samples(dir.path());
        assert_eq!(loaded.len(), 2, "kein Sample darf durch ein Event dazukommen");
        assert_eq!(loaded[0].status, "new");
        assert_eq!(loaded[1].status, "confirmed");
        assert_eq!(loaded[1].ann.boxes.len(), 1);
    }

    #[test]
    fn spaeteres_event_gewinnt() {
        let dir = TempDir::new("letztes");
        append_jsonl(&samples_path(dir.path()), &sample("s_1", "new")).unwrap();
        for status in ["confirmed", "skipped"] {
            append_jsonl(&events_path(dir.path()), &AnnEvent {
                sample_id: "s_1".to_string(), status: status.to_string(),
                boxes: vec![], at: "2026-09-16T09:00:00Z".to_string(),
            }).unwrap();
        }
        assert_eq!(load_samples(dir.path())[0].status, "skipped");
    }

    #[test]
    fn halbe_zeile_macht_das_projekt_nicht_unlesbar() {
        // Stromausfall mitten im Schreiben: die kaputte Zeile faellt weg,
        // der Rest bleibt lesbar.
        let dir = TempDir::new("kaputt");
        append_jsonl(&samples_path(dir.path()), &sample("s_1", "new")).unwrap();
        let mut f = OpenOptions::new().append(true).open(samples_path(dir.path())).unwrap();
        write!(f, "{{\"id\":\"s_2\",\"med").unwrap();
        drop(f);
        assert_eq!(load_samples(dir.path()).len(), 1);
    }

    #[test]
    fn export_wird_als_yolo_wiedererkannt() {
        // Der Export ist nur dann etwas wert, wenn die App ihn hinterher als
        // YOLO erkennt — sonst bleibt das Training gesperrt.
        let dir = TempDir::new("export");
        let media = dir.path().join("media/ab");
        fs::create_dir_all(&media).unwrap();
        let mut png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        png.extend_from_slice(&[0, 0, 0, 13]);
        png.extend_from_slice(b"IHDR");
        png.extend_from_slice(&64u32.to_be_bytes());
        png.extend_from_slice(&64u32.to_be_bytes());

        let mut s1 = sample("s_1", "confirmed");
        s1.media = "ab/s_1.png".to_string();
        s1.ann.boxes = vec![BoxAnn { cls: 0, x: 0.5, y: 0.5, w: 0.25, h: 0.5 }];
        let mut s2 = sample("s_2", "confirmed");
        s2.media = "ab/s_2.png".to_string();
        s2.src.origin = Some("/daten/mit,komma.png".to_string());
        fs::write(media.join("s_1.png"), &png).unwrap();
        fs::write(media.join("s_2.png"), &png).unwrap();

        let project = StudioProject {
            id: "sp_1".to_string(), name: "Ski".to_string(), modality: "image".to_string(),
            task: "bbox".to_string(), target_format: "yolo_bbox".to_string(),
            classes: vec!["Lift".to_string(), "Sky".to_string()],
            created_at: "x".to_string(), updated_at: "x".to_string(),
            sample_count: 0, confirmed_count: 0,
        };
        let out = dir.path().join("out");
        fs::create_dir_all(&out).unwrap();
        write_yolo_export(&project, &[&s1, &s2], &dir.path().join("media"), &out).unwrap();

        let analysis = crate::dataset_manager::detect_dataset_type(&out);
        assert!(matches!(analysis.detected_type, crate::dataset_manager::DatasetType::YoloBbox),
            "erkannt als {:?}", analysis.detected_type);

        assert_eq!(fs::read_to_string(out.join("labels/s_1.txt")).unwrap(),
            "0 0.500000 0.500000 0.250000 0.500000\n");
        // Ein Bild ohne Box bekommt eine leere Labeldatei, keine fehlende —
        // sonst gilt es als unbeschriftet statt als Negativbeispiel.
        assert_eq!(fs::read_to_string(out.join("labels/s_2.txt")).unwrap(), "");
        assert_eq!(fs::read_to_string(out.join("classes.txt")).unwrap(), "Lift\nSky\n");

        let prov = fs::read_to_string(out.join("PROVENANCE.csv")).unwrap();
        assert!(prov.contains("\"/daten/mit,komma.png\""), "Herkunft fehlt oder ist unquotiert: {}", prov);
        assert!(out.join("DATA_CARD.md").exists());
    }

    #[test]
    fn csv_feld_mit_komma_wird_gequotet() {
        assert_eq!(csv_field("a,b"), "\"a,b\"");
        assert_eq!(csv_field("sagt \"hallo\""), "\"sagt \"\"hallo\"\"\"");
        assert_eq!(csv_field("schlicht"), "schlicht");
    }
}
