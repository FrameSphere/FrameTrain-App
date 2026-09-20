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
use std::process::{Command, Stdio};
use std::sync::mpsc::{channel, Receiver};
use std::time::Duration;

use crate::command_ext::{NoWindow, PythonUtf8};

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
}

/// Projekt plus die live gezaehlten Stände, wie die Liste sie zeigt.
///
/// Warum ein eigener Typ: die Zaehler gehoeren nicht in project.json —
/// gespeicherte Zaehler laufen frueher oder spaeter aus dem Tritt mit dem
/// echten Inhalt. Sie per `skip_serializing` aus dem Projekt-Typ zu nehmen
/// hat sie auch aus der Antwort an die Oberflaeche entfernt: die Karte zeigte
/// "0 Bilder", obwohl elf auf der Platte lagen. Ablage und Antwort sind
/// deshalb zwei Typen.
#[derive(Debug, Clone, Serialize)]
pub struct StudioProjectView {
    #[serde(flatten)]
    pub project:         StudioProject,
    pub sample_count:    usize,
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
    /// Objekterkennung.
    #[serde(default)]
    pub boxes: Vec<BoxAnn>,
    /// Klassifikation: die zugewiesene Klasse.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Seq2Seq: der Zieltext.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
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
    /// Pfad relativ zu media/, z.B. "ab/ab3f….jpg". Bei Text leer.
    pub media:  String,
    pub mime:   String,
    /// Der Text selbst. Fuer jede Textzeile eine Datei anzulegen waere bei
    /// zehntausend Zeilen zehntausend Dateien — der Inhalt steht deshalb in
    /// der Zeile, so wie er aus der Quelle kam.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
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
    /// Widerspruch zwischen Modell und bestaetigtem Label, falls geprueft.
    /// Liegt in doubts.json, nicht in samples.jsonl.
    #[serde(default, skip_deserializing, skip_serializing_if = "Option::is_none")]
    pub doubt: Option<Doubt>,
}

/// Was das Modell anders sieht als das bestaetigte Label.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Doubt {
    /// Klassen, die das Modell findet und das Label nicht hat.
    pub missing: Vec<String>,
    /// Klassen im Label, die das Modell nicht bestaetigt.
    pub extra:   Vec<String>,
    pub at:      String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReviewReport {
    pub checked: usize,
    pub doubts:  usize,
    pub agree:   usize,
    pub failed:  usize,
}

/// Eine Aenderung an einem Sample.
///
/// Welche Felder gesetzt sind, haengt an der Modalitaet: Bilder schicken
/// Boxen, Klassifikation ein Label, Seq2Seq einen Zieltext. Beim Falten wird
/// uebernommen, was das Ereignis traegt — auch ein leeres Feld, denn das ist
/// die Art, eine Zuweisung wieder zu entfernen.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnnEvent {
    sample_id: String,
    status:    String,
    #[serde(default)]
    boxes:     Vec<BoxAnn>,
    #[serde(default)]
    label:     Option<String>,
    #[serde(default)]
    target:    Option<String>,
    at:        String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImportReport {
    pub added:            usize,
    pub duplicates:       usize,
    pub unreadable:       usize,
    pub with_labels:      usize,
    pub classes_added:    Vec<String>,
    /// Klassen-IDs in den Labeldateien, fuer die die Namensliste zu kurz war.
    pub unknown_ids:      Vec<usize>,
    /// Labels lagen da, wurden aber auf Wunsch nicht uebernommen.
    pub labels_ignored:   usize,
}

/// Was ein Ordner mitbringt — abgefragt, bevor importiert wird.
///
/// Der Grund: Labeldateien enthalten nur Zahlen. Ohne die Liste, zu der diese
/// Zahlen gehoeren, ist eine 0 bedeutungslos. Sie stillschweigend auf die
/// Klassenliste des Projekts zu legen, macht aus einem "Tree" der Quelle ein
/// "Ski" im Projekt — und das faellt erst im Training auf.
#[derive(Debug, Clone, Serialize)]
pub struct FolderInspection {
    pub images:            usize,
    pub with_labels:       usize,
    /// Aus classes.txt, obj.names oder data.yaml des Ordners.
    pub source_classes:    Vec<String>,
    /// Hoechste Klassen-ID, die in den gefundenen Labels vorkommt.
    pub max_class_id:      Option<usize>,
    /// Labels vorhanden, aber keine Liste dazu — hier muss gefragt werden.
    pub needs_class_list:  bool,
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
    /// Bilder, bei denen das Modell dem bestaetigten Label widerspricht.
    pub doubts:      usize,
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
        return Err("Ungültige Projekt-ID".to_string());
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
fn doubts_path(dir: &Path)  -> PathBuf { dir.join("doubts.json") }

/// Zweifel sind eine Momentaufnahme eines Durchlaufs, kein Verlauf — deshalb
/// eine Datei, die jeder Lauf ersetzt, und kein angehaengtes Ereignis.
fn load_doubts(dir: &Path) -> HashMap<String, Doubt> {
    fs::read_to_string(doubts_path(dir)).ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default()
}

fn save_doubts(dir: &Path, doubts: &HashMap<String, Doubt>) -> Result<(), String> {
    fs::write(doubts_path(dir), serde_json::to_string_pretty(doubts)
        .map_err(|e| format!("JSON: {}", e))?)
        .map_err(|e| format!("doubts.json: {}", e))
}

/// Mengenvergleich zweier Klassenlisten.
///
/// Verglichen werden Mengen, keine Reihenfolgen und keine Anzahlen: drei Baeume
/// statt zwei sind kein Widerspruch, ein fehlender Lift schon.
pub fn compare_class_sets(label: &[String], modell: &[String]) -> (Vec<String>, Vec<String>) {
    let norm = |v: &[String]| -> Vec<String> {
        let mut out: Vec<String> = v.iter().map(|s| s.trim().to_lowercase()).collect();
        out.sort(); out.dedup(); out
    };
    let l = norm(label);
    let m = norm(modell);
    let missing = m.iter().filter(|c| !l.contains(c)).cloned().collect();
    let extra   = l.iter().filter(|c| !m.contains(c)).cloned().collect();
    (missing, extra)
}
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
            samples[i].ann.label = ev.label;
            samples[i].ann.target = ev.target;
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

/// Schreibt die Klassen-IDs einer Labeldatei auf die Klassen des Projekts um.
///
/// Abgebildet wird ueber den **Namen**, nie ueber die Zahl: die 0 der Quelle
/// und die 0 des Projekts meinen im Regelfall verschiedene Dinge. Namen, die
/// das Projekt noch nicht kennt, werden angehaengt.
///
/// IDs jenseits der Namensliste bekommen einen Platzhalter und werden gemeldet
/// — stillschweigend verschwinden darf keine Box.
pub fn remap_boxes(
    boxes: &[BoxAnn], source_names: &[String], project_classes: &mut Vec<String>,
) -> (Vec<BoxAnn>, Vec<String>, Vec<usize>) {
    let mut out = Vec::with_capacity(boxes.len());
    let mut added = Vec::new();
    let mut unknown = Vec::new();
    for b in boxes {
        let name = match source_names.get(b.cls) {
            Some(n) => n.clone(),
            None => {
                if !unknown.contains(&b.cls) { unknown.push(b.cls); }
                format!("Klasse {}", b.cls)
            }
        };
        let cls = match class_index_for(&name, project_classes) {
            Some(i) => i,
            None => {
                project_classes.push(name.clone());
                added.push(name);
                project_classes.len() - 1
            }
        };
        out.push(BoxAnn { cls, ..*b });
    }
    (out, added, unknown)
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
) -> Result<Vec<StudioProjectView>, String> {
    let user_id = get_user_id(&state)?;
    let root = studio_dir(&app_handle, &user_id)?;
    let mut out: Vec<StudioProjectView> = Vec::new();
    let Ok(entries) = fs::read_dir(&root) else { return Ok(out); };
    for entry in entries.flatten() {
        let dir = entry.path();
        if !dir.is_dir() { continue; }
        let Ok(project) = load_project(&dir) else { continue; };
        let samples = load_samples(&dir);
        out.push(StudioProjectView {
            sample_count:    samples.len(),
            confirmed_count: samples.iter().filter(|s| s.status == "confirmed").count(),
            project,
        });
    }
    out.sort_by(|a, b| b.project.updated_at.cmp(&a.project.updated_at));
    Ok(out)
}

#[tauri::command]
pub async fn studio_create_project(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    name: String, modality: String, task: Option<String>, target_format: String,
    classes: Vec<String>,
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
        id, name, modality, target_format,
        // Ohne Angabe bleibt es bei Boxen — so verhalten sich Projekte aus
        // der Zeit vor den Textmodalitaeten unveraendert.
        task: task.unwrap_or_else(|| "bbox".to_string()),
        classes: classes.into_iter().map(|c| c.trim().to_string()).filter(|c| !c.is_empty()).collect(),
        created_at: now.clone(), updated_at: now,
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
            return Err("Klassen können umbenannt und ergänzt, aber nicht entfernt werden".to_string());
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

/// Was liegt in dem Ordner? Wird vor dem Import gefragt, damit die Klassenfrage
/// geklaert ist, bevor 463 Bilder falsch beschriftet im Projekt landen.
#[tauri::command]
pub async fn studio_inspect_folder(source_path: String) -> Result<FolderInspection, String> {
    let src = Path::new(&source_path);
    if !src.is_dir() { return Err(format!("Ordner nicht gefunden: {}", source_path)); }
    let files = collect_images(src);
    if files.is_empty() { return Err("Keine Bilder in diesem Ordner gefunden".to_string()); }

    let source_classes = read_source_classes(src);
    let mut with_labels = 0usize;
    let mut max_class_id: Option<usize> = None;
    // Eine Stichprobe reicht, um die Frage zu beantworten; bei 50 000 Bildern
    // soll der Dialog nicht eine Minute auf sich warten lassen.
    for file in files.iter().take(500) {
        for lp in label_paths_for_image(file) {
            let Ok(text) = fs::read_to_string(&lp) else { continue; };
            let boxes = parse_yolo_label(&text);
            if boxes.is_empty() { break; }
            with_labels += 1;
            let hoechste = boxes.iter().map(|b| b.cls).max().unwrap_or(0);
            max_class_id = Some(max_class_id.map_or(hoechste, |m: usize| m.max(hoechste)));
            break;
        }
    }

    Ok(FolderInspection {
        images: files.len(),
        with_labels,
        needs_class_list: with_labels > 0 && source_classes.is_empty(),
        source_classes,
        max_class_id,
    })
}

/// Klassennamen eines trainierten Modells, in der Reihenfolge seiner IDs.
///
/// Der einzige verlaessliche Weg zu den Namen eines fremden Label-Ordners:
/// das Modell, mit dem er entstanden ist, traegt sie im Checkpoint.
#[tauri::command]
pub async fn studio_model_classes(
    app_handle: tauri::AppHandle, version_id: String,
) -> Result<Vec<String>, String> {
    let (server, classes, _) = start_inference_server(&app_handle, &version_id)?;
    server.shutdown();
    Ok(classes)
}

#[tauri::command]
pub async fn studio_import_folder(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    project_id: String, source_path: String,
    label_classes: Option<Vec<String>>, ignore_labels: bool,
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

    // Zu welcher Liste gehoeren die Zahlen in den Labeldateien? Vorrang hat,
    // was der Aufrufer mitgibt (aus einem Modell oder von Hand), sonst die
    // Liste im Ordner selbst.
    let source_classes: Vec<String> = label_classes
        .filter(|c| !c.is_empty())
        .unwrap_or_else(|| read_source_classes(src));

    // Labels ohne Namensliste: nicht raten. Eine 0 in der Labeldatei ist eine
    // Aussage ueber eine fremde Klassenliste — fehlt sie, fehlt die Bedeutung.
    if !ignore_labels && source_classes.is_empty() {
        let hat_labels = files.iter().take(200).any(|f| {
            label_paths_for_image(f).iter().any(|lp| {
                fs::read_to_string(lp).map(|t| !parse_yolo_label(&t).is_empty()).unwrap_or(false)
            })
        });
        if hat_labels {
            return Err("Der Ordner bringt Labeldateien mit, aber keine Klassenliste. \
                Die Zahlen darin sagen allein nicht, welche Klasse gemeint ist. \
                Bitte die Liste angeben — aus einem trainierten Modell oder von Hand — \
                oder die Labels beim Import weglassen.".to_string());
        }
    }

    let mut classes_added: Vec<String> = Vec::new();
    if project.classes.is_empty() && !source_classes.is_empty() {
        project.classes = source_classes.clone();
        classes_added = source_classes.clone();
    }

    let mut report = ImportReport { added: 0, duplicates: 0, unreadable: 0, with_labels: 0,
        classes_added: vec![], unknown_ids: vec![], labels_ignored: 0 };
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
        let mut roh: Vec<BoxAnn> = Vec::new();
        for lp in label_paths_for_image(file) {
            if let Ok(text) = fs::read_to_string(&lp) {
                roh = parse_yolo_label(&text);
                break;
            }
        }
        if ignore_labels && !roh.is_empty() {
            report.labels_ignored += 1;
            roh.clear();
        }

        // Umschreiben ueber die Namen. Ohne Namensliste waere die Zahl in der
        // Labeldatei eine Behauptung ueber eine fremde Klassenliste.
        let (boxes, neu, unbekannt) = remap_boxes(&roh, &source_classes, &mut project.classes);
        classes_added.extend(neu);
        for id in unbekannt {
            if !report.unknown_ids.contains(&id) { report.unknown_ids.push(id); }
        }

        let has_labels = !boxes.is_empty();
        if has_labels { report.with_labels += 1; }

        let sample = StudioSample {
            id: format!("s_{}", &uuid::Uuid::new_v4().to_string().replace('-', "")[..10]),
            media: rel.clone(),
            mime: mime_for(&ext),
            content: None,
            status: if has_labels { "confirmed".to_string() } else { "new".to_string() },
            ann: Annotation { boxes, label: None, target: None },
            src: SampleSource {
                kind: "import".to_string(),
                origin: Some(file.to_string_lossy().to_string()),
                license: None,
                at: now.clone(),
            },
            meta: SampleMeta { w, h, group: None },
            abs_path: String::new(),
            doubt: None,
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

/// Pfad zum Skript, das Einzelbilder aus einem Video schreibt.
fn frames_script(app_handle: &tauri::AppHandle) -> Result<PathBuf, String> {
    let rel = Path::new("python").join("studio").join("extract_frames.py");
    let kandidaten = [
        app_handle.path().resource_dir().ok().map(|p| p.join(&rel)),
        Some(PathBuf::from("src-tauri").join(&rel)),
        Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(&rel)),
    ];
    for p in kandidaten.into_iter().flatten() {
        if p.exists() { return Ok(p); }
    }
    Err("extract_frames.py nicht gefunden".to_string())
}

/// Einzelbilder aus einem Video ins Projekt holen.
///
/// Jedes Bild bekommt das Video als Gruppe. Beim Export bleibt eine Gruppe
/// zusammen — sonst pruefte das Training gegen fast dieselben Bilder, mit
/// denen es gelernt hat.
#[tauri::command]
pub async fn studio_import_video(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    project_id: String, video_path: String, every_n: usize, max_frames: usize,
) -> Result<ImportReport, String> {
    let user_id = get_user_id(&state)?;
    let dir = project_dir(&app_handle, &user_id, &project_id)?;
    let mut project = load_project(&dir)?;
    let video = PathBuf::from(&video_path);
    if !video.is_file() { return Err(format!("Video nicht gefunden: {}", video_path)); }

    let gruppe = video.file_stem().map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "video".to_string());
    let tmp = app_handle.path().app_data_dir()
        .map_err(|e| format!("AppDataDir: {}", e))?
        .join("tmp").join(format!("frames_{}", &uuid::Uuid::new_v4().to_string().replace('-', "")[..12]));
    fs::create_dir_all(&tmp).map_err(|e| format!("Zwischenordner: {}", e))?;

    let script = frames_script(&app_handle)?;
    let python = crate::python_env::resolve_python();
    let mut child = Command::new(&python).no_window().python_utf8()
        .arg(script.to_string_lossy().to_string())
        .arg("--video").arg(&video_path)
        .arg("--out-dir").arg(tmp.to_string_lossy().to_string())
        .arg("--every-n").arg(every_n.max(1).to_string())
        .arg("--max-frames").arg(max_frames.clamp(1, 20000).to_string())
        .stdout(Stdio::piped()).stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Bildextraktion ließ sich nicht starten: {}", e))?;

    if let Some(err) = child.stderr.take() {
        std::thread::spawn(move || {
            for line in BufReader::new(err).lines().map_while(Result::ok) {
                eprintln!("[StudioFrames] {}", line);
            }
        });
    }

    let mut fehler: Option<String> = None;
    if let Some(out) = child.stdout.take() {
        for line in BufReader::new(out).lines().map_while(Result::ok) {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line.trim()) else { continue };
            match v.get("type").and_then(|t| t.as_str()) {
                Some("progress") => {
                    let _ = app_handle.emit("studio-import-progress", serde_json::json!({
                        "project_id": project_id,
                        "current": v.get("current").and_then(|x| x.as_u64()).unwrap_or(0),
                        "total": v.get("total").and_then(|x| x.as_u64()).unwrap_or(0),
                    }));
                }
                Some("error") => {
                    fehler = Some(v.get("message").and_then(|m| m.as_str())
                        .unwrap_or("Bildextraktion fehlgeschlagen").to_string());
                }
                _ => {}
            }
        }
    }
    let _ = child.wait();
    if let Some(msg) = fehler {
        let _ = fs::remove_dir_all(&tmp);
        return Err(msg);
    }

    // Die geschriebenen Bilder wie einen Ordnerimport aufnehmen, nur ohne
    // Labelsuche — frische Einzelbilder bringen keine mit.
    let files = collect_images(&tmp);
    let existing = load_samples(&dir);
    let mut known: std::collections::HashSet<String> =
        existing.iter().map(|s| s.media.clone()).collect();
    let mut report = ImportReport { added: 0, duplicates: 0, unreadable: 0, with_labels: 0,
        classes_added: vec![], unknown_ids: vec![], labels_ignored: 0 };
    let now = Utc::now().to_rfc3339();

    for file in &files {
        let Ok(bytes) = fs::read(file) else { report.unreadable += 1; continue; };
        let Some((w, h)) = image_dimensions(&bytes) else { report.unreadable += 1; continue; };
        let hash = sha256_hex(&bytes);
        let rel = format!("{}/{}.jpg", &hash[..2], &hash);
        if known.contains(&rel) { report.duplicates += 1; continue; }

        let target = dir.join("media").join(&rel);
        if let Some(parent) = target.parent() { fs::create_dir_all(parent).ok(); }
        if !target.exists() {
            fs::write(&target, &bytes).map_err(|e| format!("Kopieren: {}", e))?;
        }

        append_jsonl(&samples_path(&dir), &StudioSample {
            id: format!("s_{}", &uuid::Uuid::new_v4().to_string().replace('-', "")[..10]),
            media: rel.clone(),
            mime: "image/jpeg".to_string(),
            content: None,
            status: "new".to_string(),
            ann: Annotation::default(),
            src: SampleSource {
                kind: "video".to_string(),
                origin: Some(format!("{} ({})", video_path,
                    file.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default())),
                license: None,
                at: now.clone(),
            },
            meta: SampleMeta { w, h, group: Some(gruppe.clone()) },
            abs_path: String::new(),
            doubt: None,
        })?;
        known.insert(rel);
        report.added += 1;
    }

    let _ = fs::remove_dir_all(&tmp);
    project.updated_at = Utc::now().to_rfc3339();
    save_project(&dir, &project)?;
    let _ = app_handle.emit("studio-import-progress", serde_json::json!({
        "project_id": project_id, "current": files.len(), "total": files.len(), "done": true,
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
    let doubts = load_doubts(&dir);
    let filtered: Vec<&StudioSample> = match status.as_deref() {
        None | Some("") | Some("all") => all.iter().collect(),
        Some("open") => all.iter().filter(|s| s.status == "new" || s.status == "suggested").collect(),
        Some("doubt") => all.iter().filter(|s| doubts.contains_key(&s.id)).collect(),
        Some(st) => all.iter().filter(|s| s.status == st).collect(),
    };
    let total = filtered.len();
    let items = filtered.into_iter().skip(offset).take(limit.clamp(1, 500))
        .map(|s| {
            let mut c = s.clone();
            c.abs_path = media_dir.join(&s.media).to_string_lossy().to_string();
            c.doubt = doubts.get(&s.id).cloned();
            c
        })
        .collect();
    Ok(SamplePage { total, items })
}

#[tauri::command]
pub async fn studio_set_annotation(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    project_id: String, sample_id: String, boxes: Vec<BoxAnn>, status: String,
    label: Option<String>, target: Option<String>,
) -> Result<(), String> {
    if !matches!(status.as_str(), "new" | "suggested" | "confirmed" | "skipped") {
        return Err(format!("Unbekannter Status: {}", status));
    }
    let user_id = get_user_id(&state)?;
    let dir = project_dir(&app_handle, &user_id, &project_id)?;
    let event = AnnEvent {
        sample_id, status, label, target,
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
        doubts: load_doubts(&dir).len(),
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
// TEXT
//
// Texte liegen nicht als Datei je Sample in media/, sondern in der Zeile
// selbst: eine CSV mit 20 000 Zeilen wuerde sonst 20 000 Dateien anlegen.
// Alles andere — Ereignisse, Status, Export, Statistik — ist dasselbe.
// ══════════════════════════════════════════════════════════════════

/// Zerlegt CSV-Text nach RFC 4180.
///
/// Selbst geschrieben, weil die Alternative eine weitere Abhaengigkeit waere
/// und der Fall klar umrissen ist: Anfuehrungszeichen schuetzen Kommas,
/// Zeilenumbrueche und verdoppelte Anfuehrungszeichen. Genau das produziert
/// auch der Export des Labors.
pub fn parse_csv(text: &str) -> Vec<Vec<String>> {
    let mut zeilen = Vec::new();
    let mut feld = String::new();
    let mut zeile: Vec<String> = Vec::new();
    let mut in_quotes = false;
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if in_quotes {
            if c == '"' {
                if chars.peek() == Some(&'"') { chars.next(); feld.push('"'); }
                else { in_quotes = false; }
            } else {
                feld.push(c);
            }
            continue;
        }
        match c {
            '"' if feld.is_empty() => in_quotes = true,
            ',' => zeile.push(std::mem::take(&mut feld)),
            '\r' => { if chars.peek() == Some(&'\n') { chars.next(); }
                      zeile.push(std::mem::take(&mut feld));
                      zeilen.push(std::mem::take(&mut zeile)); }
            '\n' => { zeile.push(std::mem::take(&mut feld));
                      zeilen.push(std::mem::take(&mut zeile)); }
            _ => feld.push(c),
        }
    }
    if !feld.is_empty() || !zeile.is_empty() {
        zeile.push(feld);
        zeilen.push(zeile);
    }
    // Leerzeilen am Ende sind kein Datensatz.
    zeilen.retain(|z| !(z.len() == 1 && z[0].trim().is_empty()));
    zeilen
}

/// Eine Textzeile aus der Quelle, noch ohne Projektbezug.
#[derive(Debug, Clone)]
pub struct TextRow {
    pub text:  String,
    pub label: Option<String>,
    pub origin: String,
}

/// Was eine Textquelle hergibt.
#[derive(Debug, Clone, Serialize)]
pub struct TextInspection {
    /// "csv" | "jsonl" | "folder"
    pub kind:       String,
    pub rows:       usize,
    /// Spaltennamen einer CSV, Schluessel einer JSONL — leer bei Ordnern.
    pub columns:    Vec<String>,
    /// Vorschlag, welche Spalte der Text ist.
    pub text_column:  Option<String>,
    /// Vorschlag, welche Spalte das Label ist.
    pub label_column: Option<String>,
    /// Die ersten Zeilen, damit man sieht, was ankommt.
    pub preview:    Vec<String>,
    /// Klassen, die in der Quelle vorkommen.
    pub labels:     Vec<String>,
}

const TEXT_EXTS: &[&str] = &["txt", "md"];

fn spalte_finden(spalten: &[String], kandidaten: &[&str]) -> Option<String> {
    spalten.iter()
        .find(|s| kandidaten.iter().any(|k| s.trim().eq_ignore_ascii_case(k)))
        .cloned()
}

/// Liest eine Textquelle: CSV, JSONL oder ein Ordner mit .txt-Dateien.
fn read_text_source(
    path: &Path, text_column: Option<&str>, label_column: Option<&str>,
) -> Result<(String, Vec<String>, Vec<TextRow>), String> {
    if path.is_dir() {
        // Ordner: jede .txt ist ein Sample. Liegt sie in einem Unterordner,
        // ist dessen Name das Label — dieselbe Konvention wie bei Bildern.
        let mut rows = Vec::new();
        let mut stack = vec![path.to_path_buf()];
        while let Some(dir) = stack.pop() {
            let Ok(entries) = fs::read_dir(&dir) else { continue };
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    if !p.file_name().and_then(|n| n.to_str()).unwrap_or("").starts_with('.') {
                        stack.push(p);
                    }
                    continue;
                }
                let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
                if !TEXT_EXTS.contains(&ext.as_str()) { continue; }
                let Ok(text) = fs::read_to_string(&p) else { continue };
                if text.trim().is_empty() { continue; }
                let label = p.parent()
                    .filter(|parent| *parent != path)
                    .and_then(|parent| parent.file_name())
                    .map(|n| n.to_string_lossy().to_string());
                rows.push(TextRow { text, label, origin: p.to_string_lossy().to_string() });
            }
        }
        rows.sort_by(|a, b| a.origin.cmp(&b.origin));
        return Ok(("folder".to_string(), vec![], rows));
    }

    let inhalt = fs::read_to_string(path).map_err(|e| format!("Datei lesen: {}", e))?;
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    let quelle = path.to_string_lossy().to_string();

    if ext == "jsonl" || ext == "ndjson" {
        let mut spalten: Vec<String> = Vec::new();
        let mut werte: Vec<serde_json::Map<String, serde_json::Value>> = Vec::new();
        for line in inhalt.lines().filter(|l| !l.trim().is_empty()) {
            let Ok(serde_json::Value::Object(obj)) = serde_json::from_str(line) else { continue };
            for k in obj.keys() {
                if !spalten.contains(k) { spalten.push(k.clone()); }
            }
            werte.push(obj);
        }
        let tcol = text_column.map(str::to_string)
            .or_else(|| spalte_finden(&spalten, &["text", "source", "input", "sentence"]))
            .or_else(|| spalten.first().cloned())
            .ok_or("Die Datei enthaelt keine lesbaren Felder")?;
        let lcol = label_column.map(str::to_string)
            .or_else(|| spalte_finden(&spalten, &["label", "target", "class", "klasse"]));
        let rows = werte.into_iter().filter_map(|obj| {
            let text = obj.get(&tcol).and_then(|v| v.as_str())?.to_string();
            if text.trim().is_empty() { return None; }
            let label = lcol.as_ref().and_then(|c| obj.get(c))
                .and_then(|v| v.as_str().map(str::to_string));
            Some(TextRow { text, label, origin: quelle.clone() })
        }).collect();
        return Ok(("jsonl".to_string(), spalten, rows));
    }

    // Alles andere als CSV lesen. Eine .txt ohne Kommas ergibt dabei eine
    // einspaltige Tabelle — jede Zeile ein Sample, was genau richtig ist.
    let tabelle = parse_csv(&inhalt);
    if tabelle.is_empty() { return Err("Die Datei ist leer".to_string()); }

    let kopf = &tabelle[0];
    // Kopfzeile nur annehmen, wenn sie nach Spaltennamen aussieht.
    let hat_kopf = kopf.iter().any(|f| {
        let f = f.trim().to_lowercase();
        ["text", "label", "source", "target", "class", "klasse", "input", "sentence"].contains(&f.as_str())
    });
    let spalten: Vec<String> = if hat_kopf {
        kopf.clone()
    } else {
        (0..kopf.len()).map(|i| format!("Spalte {}", i + 1)).collect()
    };
    let tcol = text_column.map(str::to_string)
        .or_else(|| spalte_finden(&spalten, &["text", "source", "input", "sentence"]))
        .or_else(|| spalten.first().cloned())
        .ok_or("Keine Spalte gefunden")?;
    let lcol = label_column.map(str::to_string)
        .or_else(|| spalte_finden(&spalten, &["label", "target", "class", "klasse"]));
    let ti = spalten.iter().position(|s| *s == tcol).unwrap_or(0);
    let li = lcol.as_ref().and_then(|c| spalten.iter().position(|s| s == c));

    let daten = if hat_kopf { &tabelle[1..] } else { &tabelle[..] };
    let rows = daten.iter().filter_map(|z| {
        let text = z.get(ti)?.trim().to_string();
        if text.is_empty() { return None; }
        let label = li.and_then(|i| z.get(i)).map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
        Some(TextRow { text, label, origin: quelle.clone() })
    }).collect();
    Ok(("csv".to_string(), spalten, rows))
}

#[tauri::command]
pub async fn studio_inspect_text(
    source_path: String, text_column: Option<String>, label_column: Option<String>,
) -> Result<TextInspection, String> {
    let path = Path::new(&source_path);
    if !path.exists() { return Err(format!("Nicht gefunden: {}", source_path)); }
    let (kind, spalten, rows) = read_text_source(path, text_column.as_deref(), label_column.as_deref())?;
    if rows.is_empty() { return Err("Keine verwertbaren Texte gefunden".to_string()); }

    let mut labels: Vec<String> = Vec::new();
    for r in &rows {
        if let Some(l) = &r.label {
            if !labels.iter().any(|x| x.eq_ignore_ascii_case(l)) { labels.push(l.clone()); }
        }
    }
    labels.sort();

    Ok(TextInspection {
        text_column: spalte_finden(&spalten, &["text", "source", "input", "sentence"])
            .or_else(|| spalten.first().cloned()),
        label_column: spalte_finden(&spalten, &["label", "target", "class", "klasse"]),
        preview: rows.iter().take(3)
            .map(|r| r.text.chars().take(140).collect::<String>()).collect(),
        rows: rows.len(),
        kind, columns: spalten, labels,
    })
}

#[tauri::command]
pub async fn studio_import_text(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    project_id: String, source_path: String,
    text_column: Option<String>, label_column: Option<String>, ignore_labels: bool,
) -> Result<ImportReport, String> {
    let user_id = get_user_id(&state)?;
    let dir = project_dir(&app_handle, &user_id, &project_id)?;
    let mut project = load_project(&dir)?;
    let path = Path::new(&source_path);
    if !path.exists() { return Err(format!("Nicht gefunden: {}", source_path)); }

    let (_kind, _spalten, rows) = read_text_source(path, text_column.as_deref(), label_column.as_deref())?;
    if rows.is_empty() { return Err("Keine verwertbaren Texte gefunden".to_string()); }

    let existing = load_samples(&dir);
    // Gleicher Text zweimal ist auch hier ein Duplikat.
    let mut known: std::collections::HashSet<String> = existing.iter()
        .filter_map(|s| s.content.as_ref().map(|c| sha256_hex(c.as_bytes())))
        .collect();

    let mut report = ImportReport { added: 0, duplicates: 0, unreadable: 0, with_labels: 0,
        classes_added: vec![], unknown_ids: vec![], labels_ignored: 0 };
    let now = Utc::now().to_rfc3339();
    let total = rows.len();

    for (i, row) in rows.iter().enumerate() {
        if i % 100 == 0 {
            let _ = app_handle.emit("studio-import-progress", serde_json::json!({
                "project_id": project_id, "current": i, "total": total,
            }));
        }
        let hash = sha256_hex(row.text.as_bytes());
        if known.contains(&hash) { report.duplicates += 1; continue; }

        let mut label = row.label.clone();
        if ignore_labels && label.is_some() { report.labels_ignored += 1; label = None; }

        // Klassen aus der Quelle uebernehmen — unter dem Namen, den das
        // Projekt schon fuehrt, damit "Spam" und "spam" nicht zweimal stehen.
        if let Some(l) = &label {
            match class_index_for(l, &project.classes) {
                Some(idx) => label = Some(project.classes[idx].clone()),
                None => {
                    project.classes.push(l.clone());
                    report.classes_added.push(l.clone());
                }
            }
        }
        if label.is_some() { report.with_labels += 1; }

        append_jsonl(&samples_path(&dir), &StudioSample {
            id: format!("s_{}", &uuid::Uuid::new_v4().to_string().replace('-', "")[..10]),
            media: String::new(),
            mime: "text/plain".to_string(),
            content: Some(row.text.clone()),
            status: if label.is_some() { "confirmed".to_string() } else { "new".to_string() },
            ann: Annotation { boxes: vec![], label, target: None },
            src: SampleSource {
                kind: "import".to_string(),
                origin: Some(row.origin.clone()),
                license: None,
                at: now.clone(),
            },
            meta: SampleMeta { w: 0, h: 0, group: None },
            abs_path: String::new(),
            doubt: None,
        })?;
        known.insert(hash);
        report.added += 1;
    }

    project.updated_at = Utc::now().to_rfc3339();
    save_project(&dir, &project)?;
    let _ = app_handle.emit("studio-import-progress", serde_json::json!({
        "project_id": project_id, "current": total, "total": total, "done": true,
    }));
    Ok(report)
}

// ══════════════════════════════════════════════════════════════════
// AUDIO
//
// Dieselbe Ablage wie Bilder: die Datei liegt inhaltsadressiert in media/,
// das Label in der Annotation. Unterschiedlich ist nur, was in der Mitte der
// Werkbank steht — ein Abspieler statt eines Bildes.
// ══════════════════════════════════════════════════════════════════

const AUDIO_EXTS: &[&str] = &["wav", "mp3", "flac", "ogg", "m4a", "aac", "webm", "aiff", "aif"];

fn audio_mime(ext: &str) -> String {
    match ext {
        "wav"          => "audio/wav",
        "mp3"          => "audio/mpeg",
        "flac"         => "audio/flac",
        "ogg"          => "audio/ogg",
        "m4a" | "aac"  => "audio/mp4",
        "webm"         => "audio/webm",
        "aiff" | "aif" => "audio/aiff",
        _              => "application/octet-stream",
    }.to_string()
}

fn collect_audio(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                if !p.file_name().and_then(|n| n.to_str()).unwrap_or("").starts_with('.') {
                    stack.push(p);
                }
                continue;
            }
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
            if AUDIO_EXTS.contains(&ext.as_str()) { out.push(p); }
        }
    }
    out.sort();
    out
}

/// Findet den Transkript-Text zu einer Audiodatei: gleicher Name, .txt daneben.
fn transcript_for(audio: &Path) -> Option<String> {
    for ext in ["txt", "TXT"] {
        let p = audio.with_extension(ext);
        if let Ok(text) = fs::read_to_string(&p) {
            let text = text.trim().to_string();
            if !text.is_empty() { return Some(text); }
        }
    }
    None
}

#[tauri::command]
pub async fn studio_import_audio(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    project_id: String, source_path: String, ignore_labels: bool,
) -> Result<ImportReport, String> {
    let user_id = get_user_id(&state)?;
    let dir = project_dir(&app_handle, &user_id, &project_id)?;
    let mut project = load_project(&dir)?;
    let src = Path::new(&source_path);
    if !src.is_dir() { return Err(format!("Ordner nicht gefunden: {}", source_path)); }

    let dateien = collect_audio(src);
    if dateien.is_empty() { return Err("Keine Audiodateien in diesem Ordner gefunden".to_string()); }

    let existing = load_samples(&dir);
    let mut known: std::collections::HashSet<String> =
        existing.iter().map(|s| s.media.clone()).collect();
    let transkript = project.task == "transcript";

    let mut report = ImportReport { added: 0, duplicates: 0, unreadable: 0, with_labels: 0,
        classes_added: vec![], unknown_ids: vec![], labels_ignored: 0 };
    let now = Utc::now().to_rfc3339();
    let total = dateien.len();

    for (i, datei) in dateien.iter().enumerate() {
        if i % 20 == 0 {
            let _ = app_handle.emit("studio-import-progress", serde_json::json!({
                "project_id": project_id, "current": i, "total": total,
            }));
        }
        let Ok(bytes) = fs::read(datei) else { report.unreadable += 1; continue; };
        if bytes.is_empty() { report.unreadable += 1; continue; }
        let ext = datei.extension().and_then(|e| e.to_str()).unwrap_or("wav").to_lowercase();
        let hash = sha256_hex(&bytes);
        let rel = format!("{}/{}.{}", &hash[..2], &hash, ext);
        if known.contains(&rel) { report.duplicates += 1; continue; }

        let ziel = dir.join("media").join(&rel);
        if let Some(parent) = ziel.parent() { fs::create_dir_all(parent).ok(); }
        if !ziel.exists() {
            fs::write(&ziel, &bytes).map_err(|e| format!("Kopieren: {}", e))?;
        }

        // Bei Transkription zaehlt die .txt daneben, bei Klassifikation der
        // Ordnername — dieselben Konventionen wie bei Bild und Text.
        let (mut label, mut target) = if transkript {
            (None, transcript_for(datei))
        } else {
            let klasse = datei.parent()
                .filter(|parent| *parent != src)
                .and_then(|parent| parent.file_name())
                .map(|n| n.to_string_lossy().to_string());
            (klasse, None)
        };
        if ignore_labels { label = None; target = None; }

        if let Some(l) = &label {
            match class_index_for(l, &project.classes) {
                Some(idx) => label = Some(project.classes[idx].clone()),
                None => {
                    project.classes.push(l.clone());
                    report.classes_added.push(l.clone());
                }
            }
        }
        let fertig = label.is_some() || target.is_some();
        if fertig { report.with_labels += 1; }

        append_jsonl(&samples_path(&dir), &StudioSample {
            id: format!("s_{}", &uuid::Uuid::new_v4().to_string().replace('-', "")[..10]),
            media: rel.clone(),
            mime: audio_mime(&ext),
            content: None,
            status: if fertig { "confirmed".to_string() } else { "new".to_string() },
            ann: Annotation { boxes: vec![], label, target },
            src: SampleSource {
                kind: "import".to_string(),
                origin: Some(datei.to_string_lossy().to_string()),
                license: None,
                at: now.clone(),
            },
            meta: SampleMeta::default(),
            abs_path: String::new(),
            doubt: None,
        })?;
        known.insert(rel);
        report.added += 1;
    }

    project.updated_at = Utc::now().to_rfc3339();
    save_project(&dir, &project)?;
    let _ = app_handle.emit("studio-import-progress", serde_json::json!({
        "project_id": project_id, "current": total, "total": total, "done": true,
    }));
    Ok(report)
}

/// Eine Aufnahme aus der App ablegen.
#[tauri::command]
pub async fn studio_add_audio(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    project_id: String, bytes: Vec<u8>, ext: String, label: Option<String>,
) -> Result<ImportReport, String> {
    let user_id = get_user_id(&state)?;
    let dir = project_dir(&app_handle, &user_id, &project_id)?;
    let mut project = load_project(&dir)?;
    if bytes.is_empty() { return Err("Die Aufnahme ist leer".to_string()); }

    let ext = ext.trim().trim_start_matches('.').to_lowercase();
    let ext = if AUDIO_EXTS.contains(&ext.as_str()) { ext } else { "webm".to_string() };
    let hash = sha256_hex(&bytes);
    let rel = format!("{}/{}.{}", &hash[..2], &hash, ext);

    let mut report = ImportReport { added: 0, duplicates: 0, unreadable: 0, with_labels: 0,
        classes_added: vec![], unknown_ids: vec![], labels_ignored: 0 };
    if load_samples(&dir).iter().any(|s| s.media == rel) {
        report.duplicates = 1;
        return Ok(report);
    }

    let ziel = dir.join("media").join(&rel);
    if let Some(parent) = ziel.parent() { fs::create_dir_all(parent).ok(); }
    fs::write(&ziel, &bytes).map_err(|e| format!("Speichern: {}", e))?;

    let label = match label.as_deref().map(str::trim).filter(|l| !l.is_empty()) {
        Some(l) => match class_index_for(l, &project.classes) {
            Some(i) => Some(project.classes[i].clone()),
            None => {
                project.classes.push(l.to_string());
                report.classes_added.push(l.to_string());
                Some(l.to_string())
            }
        },
        None => None,
    };
    if label.is_some() { report.with_labels += 1; }

    append_jsonl(&samples_path(&dir), &StudioSample {
        id: format!("s_{}", &uuid::Uuid::new_v4().to_string().replace('-', "")[..10]),
        media: rel,
        mime: audio_mime(&ext),
        content: None,
        status: if label.is_some() { "confirmed".to_string() } else { "new".to_string() },
        ann: Annotation { boxes: vec![], label, target: None },
        src: SampleSource {
            kind: "record".to_string(),
            origin: Some("Aufnahme".to_string()),
            license: None,
            at: Utc::now().to_rfc3339(),
        },
        meta: SampleMeta::default(),
        abs_path: String::new(),
        doubt: None,
    })?;
    report.added = 1;
    project.updated_at = Utc::now().to_rfc3339();
    save_project(&dir, &project)?;
    Ok(report)
}

// ══════════════════════════════════════════════════════════════════
// SELBST ERSTELLEN
//
// Nicht jeder Datensatz liegt schon irgendwo. Wer eine Klasse mit zwei
// Beispielen hat, braucht einen Weg, das dritte zu schreiben — ohne den Umweg
// ueber eine CSV in einem anderen Programm.
// ══════════════════════════════════════════════════════════════════

/// Texte von Hand anlegen. Mehrere auf einmal, weil man beim Schreiben selten
/// bei einem bleibt und eine eingefuegte Liste sonst Zeile fuer Zeile muesste.
#[tauri::command]
pub async fn studio_add_texts(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    project_id: String, texts: Vec<String>, label: Option<String>, target: Option<String>,
) -> Result<ImportReport, String> {
    let user_id = get_user_id(&state)?;
    let dir = project_dir(&app_handle, &user_id, &project_id)?;
    let mut project = load_project(&dir)?;

    let sauber: Vec<String> = texts.into_iter()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect();
    if sauber.is_empty() { return Err("Kein Text eingegeben".to_string()); }

    let existing = load_samples(&dir);
    let mut known: std::collections::HashSet<String> = existing.iter()
        .filter_map(|s| s.content.as_ref().map(|c| sha256_hex(c.as_bytes())))
        .collect();

    // Die Klasse muss es im Projekt geben — sonst steht im Export ein Name,
    // den die Klassenliste nicht kennt.
    let label = match label.as_deref().map(str::trim).filter(|l| !l.is_empty()) {
        Some(l) => {
            match class_index_for(l, &project.classes) {
                Some(i) => Some(project.classes[i].clone()),
                None => {
                    project.classes.push(l.to_string());
                    save_project(&dir, &project)?;
                    Some(l.to_string())
                }
            }
        }
        None => None,
    };
    let target = target.map(|t| t.trim().to_string()).filter(|t| !t.is_empty());

    let mut report = ImportReport { added: 0, duplicates: 0, unreadable: 0, with_labels: 0,
        classes_added: vec![], unknown_ids: vec![], labels_ignored: 0 };
    let now = Utc::now().to_rfc3339();

    for text in sauber {
        let hash = sha256_hex(text.as_bytes());
        if known.contains(&hash) { report.duplicates += 1; continue; }
        let fertig = label.is_some() || target.is_some();
        if fertig { report.with_labels += 1; }

        append_jsonl(&samples_path(&dir), &StudioSample {
            id: format!("s_{}", &uuid::Uuid::new_v4().to_string().replace('-', "")[..10]),
            media: String::new(),
            mime: "text/plain".to_string(),
            content: Some(text),
            status: if fertig { "confirmed".to_string() } else { "new".to_string() },
            ann: Annotation { boxes: vec![], label: label.clone(), target: target.clone() },
            src: SampleSource {
                kind: "create".to_string(),
                origin: Some("von Hand angelegt".to_string()),
                license: None,
                at: now.clone(),
            },
            meta: SampleMeta::default(),
            abs_path: String::new(),
            doubt: None,
        })?;
        known.insert(hash);
        report.added += 1;
    }

    project.updated_at = Utc::now().to_rfc3339();
    save_project(&dir, &project)?;
    Ok(report)
}

/// Ein Bild aus der Zwischenablage oder von einem Drop ins Projekt legen.
///
/// Derselbe Weg wie beim Ordnerimport: inhaltsadressiert, Masse aus dem
/// Dateikopf, Duplikate kosten nichts.
#[tauri::command]
pub async fn studio_add_image(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    project_id: String, bytes: Vec<u8>, origin: Option<String>,
) -> Result<ImportReport, String> {
    let user_id = get_user_id(&state)?;
    let dir = project_dir(&app_handle, &user_id, &project_id)?;

    let Some((w, h)) = image_dimensions(&bytes) else {
        return Err("Das ist kein lesbares Bild (PNG, JPEG, GIF, BMP oder WebP).".to_string());
    };
    let ext = match &bytes[..4] {
        [0x89, b'P', b'N', b'G'] => "png",
        [0xFF, 0xD8, _, _]       => "jpg",
        [b'G', b'I', b'F', _]    => "gif",
        [b'B', b'M', _, _]       => "bmp",
        _                        => "webp",
    };

    let existing = load_samples(&dir);
    let hash = sha256_hex(&bytes);
    let rel = format!("{}/{}.{}", &hash[..2], &hash, ext);
    let mut report = ImportReport { added: 0, duplicates: 0, unreadable: 0, with_labels: 0,
        classes_added: vec![], unknown_ids: vec![], labels_ignored: 0 };
    if existing.iter().any(|s| s.media == rel) {
        report.duplicates = 1;
        return Ok(report);
    }

    let target = dir.join("media").join(&rel);
    if let Some(parent) = target.parent() { fs::create_dir_all(parent).ok(); }
    fs::write(&target, &bytes).map_err(|e| format!("Speichern: {}", e))?;

    append_jsonl(&samples_path(&dir), &StudioSample {
        id: format!("s_{}", &uuid::Uuid::new_v4().to_string().replace('-', "")[..10]),
        media: rel,
        mime: mime_for(ext),
        content: None,
        status: "new".to_string(),
        ann: Annotation::default(),
        src: SampleSource {
            kind: "create".to_string(),
            origin: origin.or_else(|| Some("Zwischenablage".to_string())),
            license: None,
            at: Utc::now().to_rfc3339(),
        },
        meta: SampleMeta { w, h, group: None },
        abs_path: String::new(),
        doubt: None,
    })?;
    report.added = 1;
    Ok(report)
}

// ══════════════════════════════════════════════════════════════════
// VORSCHLAEGE (Stufe 2)
//
// Ein bereits trainiertes Modell laeuft ueber die noch offenen Bilder und
// legt Boxen als Vorschlag hin. Bestaetigen ist schneller als zeichnen —
// aber der Vorschlag wird nie von allein zur Wahrheit: er bekommt den Status
// "suggested", und der Export nimmt ihn nur auf ausdruecklichen Wunsch mit.
//
// Benutzt wird derselbe Inferenz-Server wie im Labor (yolo_inference_server.py,
// stdin/stdout, gemessene 45-57 ms je Bild). Ein zweiter Weg zum Modell waere
// ein zweiter Weg, auf dem etwas anderes herauskommen kann.
// ══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize)]
pub struct SuggestReport {
    /// Bilder, die der Server gesehen hat.
    pub processed:         usize,
    /// Davon mit mindestens einer uebernommenen Box.
    pub with_boxes:        usize,
    pub boxes_total:       usize,
    /// Bestaetigte Bilder werden nie angefasst.
    pub left_confirmed:    usize,
    /// Bilder, bei denen das Modell nichts gefunden hat — sie bleiben offen,
    /// statt als "nichts drauf" bestaetigt zu werden.
    pub without_boxes:     usize,
    /// Klassen des Modells, fuer die es im Projekt keine Entsprechung gibt.
    pub unmapped_classes:  Vec<String>,
    pub classes_added:     Vec<String>,
    pub model_classes:     Vec<String>,
    pub failed:            usize,
}

/// Klassenindex im Projekt zu einem Label des Modells.
///
/// Verglichen wird ohne Ruecksicht auf Gross- und Kleinschreibung: das
/// Ski-Modell meldet "Tree", "Person", "Generallobstacle", im Projekt stehen
/// "tree", "person", "generallobstacle". Bei genauem Vergleich waere von
/// sechs Projektklassen genau eine getroffen worden.
///
/// to_lowercase statt eq_ignore_ascii_case, damit "Bäume" und "bäume"
/// ebenfalls zusammenfinden — deutsche Klassennamen sind der Normalfall.
pub fn class_index_for(label: &str, classes: &[String]) -> Option<usize> {
    let needle = label.trim().to_lowercase();
    classes.iter().position(|c| c.trim().to_lowercase() == needle)
}

struct InferenceServer {
    child: std::process::Child,
    stdin: std::io::BufWriter<std::process::ChildStdin>,
    rx:    Receiver<String>,
}

impl InferenceServer {
    /// Eine Zeile mit JSON-Objekt abwarten; alles andere (Ladeausgaben von
    /// Ultralytics) wird uebersprungen.
    fn next_json(&self, timeout: Duration) -> Result<serde_json::Value, String> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let rest = deadline.saturating_duration_since(std::time::Instant::now());
            if rest.is_zero() { return Err("Zeitüberschreitung beim Warten auf den Modell-Server".to_string()); }
            let line = self.rx.recv_timeout(rest)
                .map_err(|_| "Zeitüberschreitung beim Warten auf den Modell-Server".to_string())?;
            let trimmed = line.trim();
            if !trimmed.starts_with('{') { continue; }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) { return Ok(v); }
        }
    }

    fn send(&mut self, payload: &serde_json::Value) -> Result<(), String> {
        writeln!(self.stdin, "{}", payload).map_err(|e| format!("Schreiben an den Server: {}", e))?;
        self.stdin.flush().map_err(|e| format!("Flush: {}", e))
    }

    fn shutdown(mut self) {
        let _ = self.send(&serde_json::json!({ "cmd": "shutdown" }));
        std::thread::sleep(Duration::from_millis(120));
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Startet den passenden Inferenz-Server zu einer trainierten Version.
///
/// Welcher es ist, entscheidet das Modell, nicht der Aufrufer: liegt ein
/// Ultralytics-Checkpoint im Ordner, ist es Objekterkennung, sonst ein
/// HuggingFace-Modell. Dieselbe Unterscheidung trifft das Labor — sie hier zu
/// wiederholen hiesse, zwei Stellen koennten sich uneinig werden.
///
/// Rueckgabe: Server, bekannte Klassen (nur YOLO meldet sie vorab) und die
/// Modalitaet, die der Server selbst nennt.
fn start_inference_server(
    app_handle: &tauri::AppHandle, version_id: &str,
) -> Result<(InferenceServer, Vec<String>, String), String> {
    let (version_path, model_id) = crate::laboratory_manager::get_version_info(app_handle, version_id)?;
    let vp = PathBuf::from(&version_path);
    let model_dir_of_model = app_handle.path().app_data_dir()
        .map_err(|e| format!("AppDataDir: {}", e))?
        .join("models").join(&model_id);

    // Die Gewichte des eigenen Laufs haben Vorrang vor dem Ausgangsmodell.
    let (script, pfad_arg, pfad) = if crate::model_manager::dir_has_ultralytics_checkpoint(&vp) {
        (crate::laboratory_manager::get_yolo_server_path(app_handle)?, "--model-dir", vp.clone())
    } else if crate::model_manager::dir_has_ultralytics_checkpoint(&model_dir_of_model) {
        (crate::laboratory_manager::get_yolo_server_path(app_handle)?, "--model-dir", model_dir_of_model)
    } else if vp.join("config.json").exists() {
        (crate::laboratory_manager::get_model_server_path(app_handle)?, "--model-path", vp.clone())
    } else {
        return Err("Zu dieser Version liess sich kein Modell laden — weder ein YOLO-Checkpoint noch eine config.json.".to_string());
    };

    let python = crate::python_env::resolve_python();

    let mut child = Command::new(&python).no_window().python_utf8()
        .arg(script.to_string_lossy().to_string())
        .arg(pfad_arg).arg(pfad.to_string_lossy().to_string())
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Modell-Server ließ sich nicht starten: {}", e))?;

    let stdout = child.stdout.take().ok_or("Kein stdout des Modell-Servers")?;
    let (tx, rx) = channel::<String>();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if tx.send(line).is_err() { break; }
        }
    });
    // stderr muss gelesen werden, sonst blockiert der Server, sobald die Pipe voll ist.
    if let Some(err) = child.stderr.take() {
        std::thread::spawn(move || {
            for line in BufReader::new(err).lines().map_while(Result::ok) {
                eprintln!("[StudioSuggest] {}", line);
            }
        });
    }

    let stdin = std::io::BufWriter::new(child.stdin.take().ok_or("Kein stdin des Modell-Servers")?);
    let server = InferenceServer { child, stdin, rx };

    // Das Laden der Gewichte dauert; drei Minuten sind grosszuegig, aber ein
    // haengender Start soll nicht ewig blockieren.
    let ready = server.next_json(Duration::from_secs(180))?;
    match ready.get("type").and_then(|t| t.as_str()) {
        Some("ready") => {
            let classes = ready.get("classes").and_then(|c| c.as_array())
                .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
                .unwrap_or_default();
            let modalitaet = ready.get("modality").and_then(|m| m.as_str())
                .unwrap_or("detect").to_string();
            Ok((server, classes, modalitaet))
        }
        _ => {
            let msg = ready.get("message").and_then(|m| m.as_str())
                .unwrap_or("Der Modell-Server meldete einen Fehler").to_string();
            server.shutdown();
            Err(msg)
        }
    }
}

#[tauri::command]
pub async fn studio_suggest(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    project_id: String, version_id: String,
    min_confidence: f64, add_unknown_classes: bool,
) -> Result<SuggestReport, String> {
    let user_id = get_user_id(&state)?;
    let dir = project_dir(&app_handle, &user_id, &project_id)?;
    let mut project = load_project(&dir)?;
    let samples = load_samples(&dir);
    let media_dir = dir.join("media");

    // Bestaetigte Arbeit wird nie ueberschrieben.
    let offen: Vec<&StudioSample> = samples.iter().filter(|s| s.status != "confirmed").collect();
    let left_confirmed = samples.len() - offen.len();
    if offen.is_empty() {
        return Err("Alle Bilder sind bereits bestätigt — es gibt nichts vorzuschlagen.".to_string());
    }

    let (mut server, model_classes, _modalitaet) = start_inference_server(&app_handle, &version_id)?;
    let ist_text = project.modality == "text";

    let mut report = SuggestReport {
        processed: 0, with_boxes: 0, boxes_total: 0, left_confirmed,
        without_boxes: 0, unmapped_classes: vec![], classes_added: vec![],
        model_classes: model_classes.clone(), failed: 0,
    };
    let total = offen.len();

    for (i, sample) in offen.iter().enumerate() {
        let _ = app_handle.emit("studio-suggest-progress", serde_json::json!({
            "project_id": project_id, "current": i, "total": total,
        }));

        // Text schickt den Inhalt, Bild den Pfad — beides derselbe Server-Weg.
        let anfrage = if ist_text {
            match sample.content.as_ref() {
                Some(text) => serde_json::json!({ "text": text }),
                None => { report.failed += 1; continue; }
            }
        } else {
            serde_json::json!({ "file_path": media_dir.join(&sample.media).to_string_lossy() })
        };
        if server.send(&anfrage).is_err() {
            report.failed += 1;
            continue;
        }
        let answer = match server.next_json(Duration::from_secs(120)) {
            Ok(v) => v,
            Err(_) => { report.failed += 1; continue; }
        };
        if answer.get("type").and_then(|t| t.as_str()) == Some("error") {
            report.failed += 1;
            continue;
        }
        report.processed += 1;

        if ist_text {
            // Klassifikation: eine Vorhersage, ein Label.
            let konfidenz = answer.get("confidence").and_then(|v| v.as_f64()).unwrap_or(1.0);
            let Some(vorhersage) = answer.get("predicted").and_then(|v| v.as_str()) else {
                report.without_boxes += 1; continue;
            };
            if konfidenz < min_confidence { report.without_boxes += 1; continue; }

            let name = match class_index_for(vorhersage, &project.classes) {
                Some(i) => project.classes[i].clone(),
                None if add_unknown_classes => {
                    project.classes.push(vorhersage.to_string());
                    report.classes_added.push(vorhersage.to_string());
                    vorhersage.to_string()
                }
                None => {
                    if !report.unmapped_classes.iter().any(|c| c == vorhersage) {
                        report.unmapped_classes.push(vorhersage.to_string());
                    }
                    report.without_boxes += 1;
                    continue;
                }
            };

            report.with_boxes += 1;
            append_jsonl(&events_path(&dir), &AnnEvent {
                sample_id: sample.id.clone(),
                status:    "suggested".to_string(),
                boxes:     vec![],
                label:     Some(name),
                target:    None,
                at:        Utc::now().to_rfc3339(),
            })?;
            continue;
        }

        // Die Masse des Servers zaehlen; kennt er sie nicht, die aus dem Import.
        let width  = answer.get("image_width").and_then(|v| v.as_f64()).filter(|v| *v > 0.0)
            .unwrap_or(sample.meta.w as f64);
        let height = answer.get("image_height").and_then(|v| v.as_f64()).filter(|v| *v > 0.0)
            .unwrap_or(sample.meta.h as f64);

        let mut boxes: Vec<BoxAnn> = Vec::new();
        for b in answer.get("boxes").and_then(|v| v.as_array()).map(|a| a.as_slice()).unwrap_or(&[]) {
            let conf = b.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.0);
            if conf < min_confidence { continue; }
            let Some(label) = b.get("label").and_then(|v| v.as_str()) else { continue; };

            let cls = match class_index_for(label, &project.classes) {
                Some(idx) => idx,
                None if add_unknown_classes => {
                    project.classes.push(label.to_string());
                    report.classes_added.push(label.to_string());
                    project.classes.len() - 1
                }
                None => {
                    if !report.unmapped_classes.iter().any(|c| c == label) {
                        report.unmapped_classes.push(label.to_string());
                    }
                    continue;
                }
            };

            let (x1, y1, x2, y2) = (
                b.get("x1").and_then(|v| v.as_f64()).unwrap_or(0.0),
                b.get("y1").and_then(|v| v.as_f64()).unwrap_or(0.0),
                b.get("x2").and_then(|v| v.as_f64()).unwrap_or(0.0),
                b.get("y2").and_then(|v| v.as_f64()).unwrap_or(0.0),
            );
            if let Some(n) = crate::yolo_export::normalize_box(cls, x1, y1, x2, y2, width, height) {
                boxes.push(BoxAnn { cls: n.cls, x: n.x, y: n.y, w: n.w, h: n.h });
            }
        }

        // Ohne Fund bleibt das Bild offen. Es als "Vorschlag: nichts drauf" zu
        // markieren wuerde dazu einladen, ein uebersehenes Objekt wegzudruecken.
        if boxes.is_empty() { report.without_boxes += 1; continue; }

        report.with_boxes += 1;
        report.boxes_total += boxes.len();
        append_jsonl(&events_path(&dir), &AnnEvent {
            sample_id: sample.id.clone(),
            status:    "suggested".to_string(),
            boxes,
            label:     None,
            target:    None,
            at:        Utc::now().to_rfc3339(),
        })?;
    }

    server.shutdown();
    maybe_compact(&dir)?;
    if !report.classes_added.is_empty() {
        project.updated_at = Utc::now().to_rfc3339();
        save_project(&dir, &project)?;
    }
    let _ = app_handle.emit("studio-suggest-progress", serde_json::json!({
        "project_id": project_id, "current": total, "total": total, "done": true,
    }));
    Ok(report)
}

/// Prueft die bestaetigten Labels gegen das Modell.
///
/// Bei uebernommenen Fremdlabels ist das oft wertvoller als neue Labels: wo
/// Modell und Label sich widersprechen, steckt haeufig ein Fehler im Label.
/// Geaendert wird dabei nichts — nur markiert.
#[tauri::command]
pub async fn studio_review(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    project_id: String, version_id: String, min_confidence: f64,
) -> Result<ReviewReport, String> {
    let user_id = get_user_id(&state)?;
    let dir = project_dir(&app_handle, &user_id, &project_id)?;
    let project = load_project(&dir)?;
    let samples = load_samples(&dir);
    let media_dir = dir.join("media");

    let bestaetigt: Vec<&StudioSample> = samples.iter().filter(|s| s.status == "confirmed").collect();
    if bestaetigt.is_empty() {
        return Err("Es gibt noch keine bestätigten Bilder zum Prüfen.".to_string());
    }

    let (mut server, _model_classes, _modalitaet) = start_inference_server(&app_handle, &version_id)?;
    let ist_text = project.modality == "text";
    let mut doubts: HashMap<String, Doubt> = HashMap::new();
    let mut report = ReviewReport { checked: 0, doubts: 0, agree: 0, failed: 0 };
    let total = bestaetigt.len();
    let now = Utc::now().to_rfc3339();

    for (i, sample) in bestaetigt.iter().enumerate() {
        let _ = app_handle.emit("studio-review-progress", serde_json::json!({
            "project_id": project_id, "current": i, "total": total,
        }));

        let anfrage = if ist_text {
            match sample.content.as_ref() {
                Some(text) => serde_json::json!({ "text": text }),
                None => { report.failed += 1; continue; }
            }
        } else {
            serde_json::json!({ "file_path": media_dir.join(&sample.media).to_string_lossy() })
        };
        if server.send(&anfrage).is_err() {
            report.failed += 1;
            continue;
        }
        let answer = match server.next_json(Duration::from_secs(120)) {
            Ok(v) => v,
            Err(_) => { report.failed += 1; continue; }
        };
        if answer.get("type").and_then(|t| t.as_str()) == Some("error") {
            report.failed += 1;
            continue;
        }
        report.checked += 1;

        if ist_text {
            // Klassifikation: das Modell sagt eine Klasse, im Label steht eine.
            let vorhersage = answer.get("predicted").and_then(|v| v.as_str()).unwrap_or("");
            let konfidenz = answer.get("confidence").and_then(|v| v.as_f64()).unwrap_or(1.0);
            let im_label = sample.ann.label.clone().unwrap_or_default();
            let einig = konfidenz < min_confidence
                || vorhersage.trim().to_lowercase() == im_label.trim().to_lowercase();
            if einig {
                report.agree += 1;
            } else {
                report.doubts += 1;
                doubts.insert(sample.id.clone(), Doubt {
                    missing: vec![vorhersage.to_string()],
                    extra:   vec![im_label],
                    at:      now.clone(),
                });
            }
            continue;
        }

        // Was das Modell sieht — nur Klassen, die das Projekt ueberhaupt kennt.
        // Sonst stuende bei jedem Bild "Offroad fehlt", obwohl das Projekt die
        // Klasse gar nicht fuehrt.
        let mut gesehen: Vec<String> = Vec::new();
        for b in answer.get("boxes").and_then(|v| v.as_array()).map(|a| a.as_slice()).unwrap_or(&[]) {
            if b.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.0) < min_confidence { continue; }
            let Some(label) = b.get("label").and_then(|v| v.as_str()) else { continue; };
            if let Some(idx) = class_index_for(label, &project.classes) {
                gesehen.push(project.classes[idx].clone());
            }
        }

        let im_label: Vec<String> = sample.ann.boxes.iter()
            .filter_map(|b| project.classes.get(b.cls).cloned())
            .collect();

        let (missing, extra) = compare_class_sets(&im_label, &gesehen);
        if missing.is_empty() && extra.is_empty() {
            report.agree += 1;
        } else {
            report.doubts += 1;
            doubts.insert(sample.id.clone(), Doubt { missing, extra, at: now.clone() });
        }
    }

    server.shutdown();
    save_doubts(&dir, &doubts)?;
    let _ = app_handle.emit("studio-review-progress", serde_json::json!({
        "project_id": project_id, "current": total, "total": total, "done": true,
    }));
    Ok(report)
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
    train_ratio: f64, val_ratio: f64,
) -> Result<(), String> {
    // Gruppenbewusst aufteilen: Einzelbilder eines Videos duerfen nicht
    // gleichzeitig in Train und Val landen. Ohne Gruppe steht jedes Bild fuer
    // sich, dann ist es ein gewoehnlicher Zufallssplit.
    let splits: Vec<Option<String>> = if train_ratio > 0.0 && train_ratio < 1.0 {
        let gruppen: Vec<String> = samples.iter()
            .map(|s| s.meta.group.clone().unwrap_or_else(|| s.id.clone()))
            .collect();
        crate::yolo_export::assign_splits(&gruppen, train_ratio, val_ratio)
            .into_iter().map(Some).collect()
    } else {
        vec![None; samples.len()]
    };

    let items: Vec<crate::yolo_export::ExportItem> = samples.iter().zip(splits)
        .map(|(s, split)| crate::yolo_export::ExportItem {
            source: media_dir.join(&s.media),
            stem:   s.id.clone(),
            boxes:  s.ann.boxes.iter()
                .filter_map(|b| crate::yolo_export::norm_box(b.cls, b.x, b.y, b.w, b.h))
                .collect(),
            split,
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

{split_hinweis}

Herkunft je Bild steht in PROVENANCE.csv.
",
        name = project.name,
        date = Utc::now().format("%Y-%m-%d"),
        count = samples.len(),
        confirmed = confirmed,
        suggested = suggested,
        boxes = boxes,
        classes = project.classes.join(", "),
        split_hinweis = if train_ratio > 0.0 && train_ratio < 1.0 {
            "Aufgeteilt in train/val/test. Bilder derselben Gruppe (z. B. eines Videos)\nliegen immer im selben Teil, damit die Validierung aussagekräftig bleibt."
        } else {
            "Aufteilung in train/val/test ist noch nicht erfolgt — dafür den Split im\nDataset-Bereich nutzen, er hält Bild- und Labelpaare zusammen."
        });
    fs::write(out.join("DATA_CARD.md"), card).map_err(|e| format!("DATA_CARD.md: {}", e))?;
    Ok(())
}

/// Schreibt das Projekt als Textdatensatz.
///
/// Die Spalten sind nicht frei gewaehlt: die Train Engine liest bei
/// Klassifikation text/label aus einer CSV und bei Seq2Seq source/target aus
/// einer JSONL. Was hier entsteht, muss dort ohne Zwischenschritt passen.
fn write_text_export(
    project: &StudioProject, samples: &[&StudioSample], out: &Path,
) -> Result<(), String> {
    let seq2seq = project.task == "pairs";
    let mut provenance = String::from("sample_id,herkunft,lizenz,status\n");

    if seq2seq {
        let mut zeilen = String::new();
        for s in samples {
            let (Some(quelle), Some(ziel)) = (s.content.as_ref(), s.ann.target.as_ref()) else { continue };
            zeilen.push_str(&serde_json::json!({ "source": quelle, "target": ziel }).to_string());
            zeilen.push('\n');
            provenance.push_str(&format!("{},{},{},{}\n", csv_field(&s.id),
                csv_field(s.src.origin.as_deref().unwrap_or("")),
                csv_field(s.src.license.as_deref().unwrap_or("")),
                csv_field(&s.status)));
        }
        fs::write(out.join("daten.jsonl"), zeilen).map_err(|e| format!("JSONL: {}", e))?;
    } else {
        let mut zeilen = String::from("text,label\n");
        for s in samples {
            let (Some(text), Some(label)) = (s.content.as_ref(), s.ann.label.as_ref()) else { continue };
            zeilen.push_str(&format!("{},{}\n", csv_field(text), csv_field(label)));
            provenance.push_str(&format!("{},{},{},{}\n", csv_field(&s.id),
                csv_field(s.src.origin.as_deref().unwrap_or("")),
                csv_field(s.src.license.as_deref().unwrap_or("")),
                csv_field(&s.status)));
        }
        fs::write(out.join("daten.csv"), zeilen).map_err(|e| format!("CSV: {}", e))?;
    }

    fs::write(out.join("PROVENANCE.csv"), provenance)
        .map_err(|e| format!("PROVENANCE.csv: {}", e))?;

    let card = format!(
"# {name}

Erzeugt vom FrameTrain Dataset Studio am {date}.

- Texte: {count}
- Klassen: {classes}
- Format: {format}

Herkunft je Text steht in PROVENANCE.csv.
",
        name = project.name,
        date = Utc::now().format("%Y-%m-%d"),
        count = samples.len(),
        classes = if project.classes.is_empty() { "—".to_string() } else { project.classes.join(", ") },
        format = if seq2seq { "daten.jsonl mit source und target" } else { "daten.csv mit text und label" });
    fs::write(out.join("DATA_CARD.md"), card).map_err(|e| format!("DATA_CARD.md: {}", e))?;
    Ok(())
}

/// Schreibt das Projekt als Audiodatensatz.
///
/// Zwei Formen, beide erkennt der Dataset-Import von selbst: Klassifikation
/// als Ordner je Klasse (FolderClass), Transkription als Audiodatei mit
/// gleichnamiger .txt daneben (AudioTranscript).
fn write_audio_export(
    project: &StudioProject, samples: &[&StudioSample], media_dir: &Path, out: &Path,
) -> Result<PathBuf, String> {
    let transkript = project.task == "transcript";
    let mut provenance = String::from("sample_id,datei,herkunft,lizenz,status,label\n");

    // Klassenordner werden nur erkannt, wenn im Wurzelordner keine Dateien
    // liegen — PROVENANCE.csv und DATA_CARD.md dort wuerden die Erkennung auf
    // "flat_file" kippen und das Training blockieren. Deshalb liegen die
    // Klassen eine Ebene tiefer, die Beipackzettel bleiben darueber.
    let daten = if transkript { out.to_path_buf() } else { out.join("dataset") };
    fs::create_dir_all(&daten).map_err(|e| format!("mkdir: {}", e))?;

    for s in samples {
        let ext = Path::new(&s.media).extension().and_then(|e| e.to_str()).unwrap_or("wav");
        let quelle = media_dir.join(&s.media);

        let (ordner, name) = if transkript {
            (daten.clone(), format!("{}.{}", s.id, ext))
        } else {
            // Ohne Klasse waere der Ordnername leer — solche Samples gehoeren
            // nicht in einen Klassifikations-Export.
            let Some(klasse) = s.ann.label.as_ref() else { continue };
            (daten.join(klasse.replace('/', "_")), format!("{}.{}", s.id, ext))
        };
        fs::create_dir_all(&ordner).map_err(|e| format!("mkdir: {}", e))?;
        fs::copy(&quelle, ordner.join(&name)).map_err(|e| format!("Audio kopieren: {}", e))?;

        if transkript {
            let Some(text) = s.ann.target.as_ref() else { continue };
            fs::write(ordner.join(format!("{}.txt", s.id)), text)
                .map_err(|e| format!("Transkript schreiben: {}", e))?;
        }

        provenance.push_str(&format!("{},{},{},{},{},{}\n",
            csv_field(&s.id), csv_field(&name),
            csv_field(s.src.origin.as_deref().unwrap_or("")),
            csv_field(s.src.license.as_deref().unwrap_or("")),
            csv_field(&s.status),
            csv_field(s.ann.label.as_deref().unwrap_or(""))));
    }

    fs::write(out.join("PROVENANCE.csv"), provenance)
        .map_err(|e| format!("PROVENANCE.csv: {}", e))?;

    let card = format!(
"# {name}

Erzeugt vom FrameTrain Dataset Studio am {date}.

- Aufnahmen: {count}
- Form: {form}
{klassen}
Herkunft je Aufnahme steht in PROVENANCE.csv.
",
        name = project.name,
        date = Utc::now().format("%Y-%m-%d"),
        count = samples.len(),
        form = if transkript { "Audiodatei mit gleichnamiger .txt daneben" }
               else { "ein Ordner je Klasse" },
        klassen = if transkript { String::new() }
                  else { format!("- Klassen: {}\n", project.classes.join(", ")) });
    fs::write(out.join("DATA_CARD.md"), card).map_err(|e| format!("DATA_CARD.md: {}", e))?;
    Ok(daten)
}

#[tauri::command]
pub async fn studio_export(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    project_id: String, model_id: String, dataset_name: String, include_suggested: bool,
    train_ratio: f64, val_ratio: f64,
) -> Result<crate::dataset_manager::DatasetInfo, String> {
    let user_id = get_user_id(&state)?;
    let dir = project_dir(&app_handle, &user_id, &project_id)?;
    let project = load_project(&dir)?;
    if project.classes.is_empty() && project.task != "pairs" && project.task != "transcript" {
        return Err("Das Projekt hat noch keine Klassen".to_string());
    }

    let samples = load_samples(&dir);
    let selected: Vec<&StudioSample> = samples.iter()
        .filter(|s| s.status == "confirmed" || (include_suggested && s.status == "suggested"))
        .collect();
    if selected.is_empty() {
        return Err("Keine bestätigten Samples — es gibt nichts zu exportieren".to_string());
    }
    let ist_text  = project.modality == "text";
    let ist_audio = project.modality == "audio";

    let stamp = Utc::now().format("%Y%m%d_%H%M%S").to_string();
    let out = dir.join("exports").join(&stamp);
    fs::create_dir_all(&out).map_err(|e| format!("mkdir export: {}", e))?;
    // Welcher Ordner am Ende registriert wird, entscheidet der Writer: bei
    // Klassenordnern muss die Wurzel dateifrei bleiben, sonst wird der Typ
    // falsch erkannt.
    let zu_registrieren = if ist_text {
        write_text_export(&project, &selected, &out)?;
        out.clone()
    } else if ist_audio {
        write_audio_export(&project, &selected, &dir.join("media"), &out)?
    } else {
        write_yolo_export(&project, &selected, &dir.join("media"), &out, train_ratio, val_ratio)?;
        out.clone()
    };

    let name = if dataset_name.trim().is_empty() { project.name.clone() } else { dataset_name };
    crate::dataset_manager::import_local_dataset(
        app_handle, state, zu_registrieren.to_string_lossy().to_string(), name, model_id,
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
            content: None, status: status.to_string(), ann: Annotation::default(),
            src: SampleSource { kind: "import".to_string(), origin: Some(format!("/daten/{}.jpg", id)),
                license: None, at: "2026-09-16T08:00:00Z".to_string() },
            meta: SampleMeta { w: 1000, h: 500, group: None },
            abs_path: String::new(), doubt: None,
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
            label: None, target: None,
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
                boxes: vec![], label: None, target: None,
                at: "2026-09-16T09:00:00Z".to_string(),
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

    fn textsample(id: &str, text: &str, label: Option<&str>) -> StudioSample {
        StudioSample {
            id: id.to_string(), media: String::new(), mime: "text/plain".to_string(),
            content: Some(text.to_string()), status: "confirmed".to_string(),
            ann: Annotation { boxes: vec![], label: label.map(str::to_string), target: None },
            src: SampleSource { kind: "import".to_string(), origin: Some("/daten/x.csv".to_string()),
                license: None, at: "2026-09-20T08:00:00Z".to_string() },
            meta: SampleMeta::default(), abs_path: String::new(), doubt: None,
        }
    }

    fn audiosample(id: &str, label: Option<&str>, ziel: Option<&str>) -> StudioSample {
        StudioSample {
            id: id.to_string(), media: format!("ab/{}.wav", id), mime: "audio/wav".to_string(),
            content: None, status: "confirmed".to_string(),
            ann: Annotation { boxes: vec![], label: label.map(str::to_string),
                target: ziel.map(str::to_string) },
            src: SampleSource { kind: "record".to_string(), origin: Some("Aufnahme".to_string()),
                license: None, at: "2026-09-20T08:00:00Z".to_string() },
            meta: SampleMeta::default(), abs_path: String::new(), doubt: None,
        }
    }

    #[test]
    fn audioexport_klassifikation_wird_als_klassenordner_erkannt() {
        let dir = TempDir::new("audioklassen");
        let media = dir.path().join("media/ab");
        fs::create_dir_all(&media).unwrap();
        for id in ["s_1", "s_2"] {
            fs::write(media.join(format!("{}.wav", id)), vec![0u8; 32]).unwrap();
        }
        let project = StudioProject {
            id: "sp_a".to_string(), name: "Ansagen".to_string(), modality: "audio".to_string(),
            task: "classification".to_string(), target_format: "folder_class".to_string(),
            classes: vec!["ansage".to_string(), "stoerung".to_string()],
            created_at: "x".to_string(), updated_at: "x".to_string(),
        };
        let s1 = audiosample("s_1", Some("ansage"), None);
        let s2 = audiosample("s_2", Some("stoerung"), None);
        // Ohne Klasse gehoert die Aufnahme nicht in einen Klassifikations-Export.
        let s3 = audiosample("s_3", None, None);
        let out = dir.path().join("out");
        fs::create_dir_all(&out).unwrap();
        let daten = write_audio_export(&project, &[&s1, &s2, &s3], &dir.path().join("media"), &out).unwrap();

        assert!(daten.join("ansage/s_1.wav").exists());
        assert!(daten.join("stoerung/s_2.wav").exists());
        assert!(!daten.join("s_3.wav").exists(), "Aufnahme ohne Klasse darf nicht mit");
        // Die Beipackzettel liegen ueber dem registrierten Ordner, damit dessen
        // Wurzel dateifrei bleibt.
        assert!(out.join("PROVENANCE.csv").exists());
        assert!(!daten.join("PROVENANCE.csv").exists());

        let analysis = crate::dataset_manager::detect_dataset_type(&daten);
        assert_eq!(analysis.detected_type.as_str(), "folder_class",
            "erkannt als {:?}", analysis.detected_type);
    }

    #[test]
    fn audioexport_transkript_legt_die_txt_daneben() {
        let dir = TempDir::new("audiotext");
        let media = dir.path().join("media/ab");
        fs::create_dir_all(&media).unwrap();
        fs::write(media.join("s_1.wav"), vec![0u8; 32]).unwrap();
        let project = StudioProject {
            id: "sp_t".to_string(), name: "Durchsagen".to_string(), modality: "audio".to_string(),
            task: "transcript".to_string(), target_format: "audio_transcript".to_string(),
            classes: vec![], created_at: "x".to_string(), updated_at: "x".to_string(),
        };
        let s1 = audiosample("s_1", None, Some("Der Lift faehrt gleich weiter"));
        let out = dir.path().join("out");
        fs::create_dir_all(&out).unwrap();
        let daten = write_audio_export(&project, &[&s1], &dir.path().join("media"), &out).unwrap();

        assert_eq!(daten, out, "Transkripte brauchen keinen Unterordner");
        assert!(out.join("s_1.wav").exists());
        assert_eq!(fs::read_to_string(out.join("s_1.txt")).unwrap(),
            "Der Lift faehrt gleich weiter");

        let analysis = crate::dataset_manager::detect_dataset_type(&out);
        assert_eq!(analysis.detected_type.as_str(), "audio_transcript",
            "erkannt als {:?}", analysis.detected_type);
    }

    #[test]
    fn textexport_schreibt_die_spalten_der_train_engine() {
        // text,label ist nicht frei gewaehlt — seq_classification liest genau
        // diese Spalten. Und ein Komma im Text darf die Datei nicht zerreissen.
        let dir = TempDir::new("textexport");
        let project = StudioProject {
            id: "sp_t".to_string(), name: "Rueckmeldungen".to_string(),
            modality: "text".to_string(), task: "classification".to_string(),
            target_format: "flat_file".to_string(),
            classes: vec!["beschwerde".to_string(), "lob".to_string()],
            created_at: "x".to_string(), updated_at: "x".to_string(),
        };
        let s1 = textsample("s_1", "Der Lift stand still, zweimal", Some("beschwerde"));
        let s2 = textsample("s_2", "Tolle Piste", Some("lob"));
        let out = dir.path().join("out");
        fs::create_dir_all(&out).unwrap();
        write_text_export(&project, &[&s1, &s2], &out).unwrap();

        let csv = fs::read_to_string(out.join("daten.csv")).unwrap();
        let tabelle = parse_csv(&csv);
        assert_eq!(tabelle[0], vec!["text", "label"]);
        assert_eq!(tabelle[1], vec!["Der Lift stand still, zweimal", "beschwerde"]);
        assert_eq!(tabelle[2], vec!["Tolle Piste", "lob"]);
        assert!(out.join("PROVENANCE.csv").exists());
        assert!(out.join("DATA_CARD.md").exists());
    }

    #[test]
    fn textexport_schreibt_paare_als_jsonl() {
        let dir = TempDir::new("paare");
        let project = StudioProject {
            id: "sp_p".to_string(), name: "Umformulieren".to_string(),
            modality: "text".to_string(), task: "pairs".to_string(),
            target_format: "flat_file".to_string(), classes: vec![],
            created_at: "x".to_string(), updated_at: "x".to_string(),
        };
        let mut s1 = textsample("s_1", "Der Lift steht", None);
        s1.ann.target = Some("Die Bahn ist ausser Betrieb".to_string());
        // Ohne Zieltext gehoert die Zeile nicht in den Export.
        let s2 = textsample("s_2", "Ohne Ziel", None);
        let out = dir.path().join("out");
        fs::create_dir_all(&out).unwrap();
        write_text_export(&project, &[&s1, &s2], &out).unwrap();

        let jsonl = fs::read_to_string(out.join("daten.jsonl")).unwrap();
        let zeilen: Vec<&str> = jsonl.lines().filter(|l| !l.trim().is_empty()).collect();
        assert_eq!(zeilen.len(), 1, "die Zeile ohne Ziel darf nicht mit");
        let v: serde_json::Value = serde_json::from_str(zeilen[0]).unwrap();
        assert_eq!(v["source"], "Der Lift steht");
        assert_eq!(v["target"], "Die Bahn ist ausser Betrieb");
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
        };
        let out = dir.path().join("out");
        fs::create_dir_all(&out).unwrap();
        write_yolo_export(&project, &[&s1, &s2], &dir.path().join("media"), &out, 0.0, 0.0).unwrap();

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
    fn csv_lesen_haelt_kommas_und_anfuehrungszeichen_zusammen() {
        let t = "text,label\n\"Hallo, Welt\",gruss\n\"er sagte \"\"hi\"\"\",zitat\n";
        let tabelle = parse_csv(t);
        assert_eq!(tabelle.len(), 3);
        assert_eq!(tabelle[1], vec!["Hallo, Welt", "gruss"]);
        assert_eq!(tabelle[2], vec!["er sagte \"hi\"", "zitat"]);
    }

    #[test]
    fn csv_lesen_kommt_mit_zeilenumbruch_im_feld_klar() {
        // Ein Textfeld darf einen Absatz enthalten — das zerriss frueher jede
        // selbstgebaute Zerlegung an \n.
        let tabelle = parse_csv("text,label\n\"Zeile eins\nZeile zwei\",lang\n");
        assert_eq!(tabelle.len(), 2);
        assert_eq!(tabelle[1][0], "Zeile eins\nZeile zwei");
        assert_eq!(tabelle[1][1], "lang");
    }

    #[test]
    fn csv_lesen_versteht_windows_zeilenenden() {
        let tabelle = parse_csv("a,b\r\n1,2\r\n");
        assert_eq!(tabelle, vec![vec!["a", "b"], vec!["1", "2"]]);
    }

    #[test]
    fn textquelle_csv_erkennt_spalten_und_labels() {
        let dir = TempDir::new("csv");
        let datei = dir.path().join("daten.csv");
        fs::write(&datei, "text,label\nDer Lift steht,defekt\nSchoene Piste,ok\n").unwrap();

        let (kind, spalten, rows) = read_text_source(&datei, None, None).unwrap();
        assert_eq!(kind, "csv");
        assert_eq!(spalten, vec!["text", "label"]);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].text, "Der Lift steht");
        assert_eq!(rows[0].label.as_deref(), Some("defekt"));
    }

    #[test]
    fn textquelle_ohne_kopfzeile_nimmt_die_erste_spalte() {
        let dir = TempDir::new("nokopf");
        let datei = dir.path().join("roh.csv");
        fs::write(&datei, "Der Lift steht\nSchoene Piste\n").unwrap();

        let (_, spalten, rows) = read_text_source(&datei, None, None).unwrap();
        assert_eq!(spalten, vec!["Spalte 1"]);
        assert_eq!(rows.len(), 2, "beide Zeilen sind Daten, keine davon ist ein Kopf");
        assert_eq!(rows[0].text, "Der Lift steht");
        assert!(rows[0].label.is_none());
    }

    #[test]
    fn textquelle_ordner_nimmt_den_unterordner_als_klasse() {
        let dir = TempDir::new("ordner");
        fs::create_dir_all(dir.path().join("beschwerde")).unwrap();
        fs::create_dir_all(dir.path().join("lob")).unwrap();
        fs::write(dir.path().join("beschwerde/a.txt"), "Der Lift stand still").unwrap();
        fs::write(dir.path().join("lob/b.txt"), "Tolle Piste").unwrap();
        // Direkt im Wurzelordner: kein Unterordner, also kein Label.
        fs::write(dir.path().join("c.txt"), "Ohne Zuordnung").unwrap();

        let (kind, _, rows) = read_text_source(dir.path(), None, None).unwrap();
        assert_eq!(kind, "folder");
        assert_eq!(rows.len(), 3);
        let label_von = |t: &str| rows.iter().find(|r| r.text.contains(t))
            .and_then(|r| r.label.clone());
        assert_eq!(label_von("Lift").as_deref(), Some("beschwerde"));
        assert_eq!(label_von("Piste").as_deref(), Some("lob"));
        assert_eq!(label_von("Zuordnung"), None);
    }

    #[test]
    fn textquelle_jsonl_liest_source_und_target() {
        let dir = TempDir::new("jsonl");
        let datei = dir.path().join("paare.jsonl");
        fs::write(&datei, "{\"source\":\"Hallo\",\"target\":\"Moin\"}\n{\"source\":\"Tschuess\",\"target\":\"Ciao\"}\n").unwrap();

        let (kind, spalten, rows) = read_text_source(&datei, None, None).unwrap();
        assert_eq!(kind, "jsonl");
        assert!(spalten.contains(&"source".to_string()));
        assert_eq!(rows[0].text, "Hallo");
        assert_eq!(rows[0].label.as_deref(), Some("Moin"));
    }

    #[test]
    fn mengenvergleich_meldet_nur_echte_widersprueche() {
        let v = |xs: &[&str]| -> Vec<String> { xs.iter().map(|s| s.to_string()).collect() };

        // Drei Baeume statt zwei sind kein Widerspruch — verglichen werden
        // Mengen, nicht Anzahlen.
        let (missing, extra) = compare_class_sets(&v(&["tree", "tree"]), &v(&["Tree", "Tree", "Tree"]));
        assert!(missing.is_empty() && extra.is_empty(), "missing {:?}, extra {:?}", missing, extra);

        // Das Modell sieht einen Lift, im Label steht keiner.
        let (missing, _) = compare_class_sets(&v(&["tree"]), &v(&["Tree", "Lift"]));
        assert_eq!(missing, vec!["lift"]);

        // Im Label steht Ski, das Modell sieht keinen.
        let (_, extra) = compare_class_sets(&v(&["Ski", "tree"]), &v(&["tree"]));
        assert_eq!(extra, vec!["ski"]);
    }

    #[test]
    fn labels_werden_ueber_namen_umgeschrieben_nicht_ueber_zahlen() {
        // Der echte Fall: die Labeldateien in yolo8n_data stammen aus einem
        // Datensatz mit 13 Klassen, das Projekt hat sechs in anderer Reihenfolge.
        // Eine 0 heisst dort "Tree" und hier "Ski" — wer die Zahl uebernimmt,
        // beschriftet 463 Bilder falsch und merkt es erst im Training.
        let quelle: Vec<String> = ["Tree", "Stone", "Person", "Hole", "Building", "Stick",
            "Emptyspace", "Lift", "Slopesign", "Slopeborder", "Sky", "Generallobstacle", "Offroad"]
            .iter().map(|s| s.to_string()).collect();
        let mut projekt: Vec<String> = ["Ski", "Emptyspace", "generallobstacle", "tree", "person", "sky"]
            .iter().map(|s| s.to_string()).collect();

        let b = |cls: usize| BoxAnn { cls, x: 0.5, y: 0.5, w: 0.2, h: 0.2 };
        let (boxen, neu, unbekannt) = remap_boxes(&[b(0), b(2), b(10), b(1)], &quelle, &mut projekt);

        // Tree -> "tree" (Index 3), Person -> "person" (4), Sky -> "sky" (5).
        assert_eq!(boxen[0].cls, 3, "Tree landete auf {}", projekt[boxen[0].cls]);
        assert_eq!(boxen[1].cls, 4);
        assert_eq!(boxen[2].cls, 5);
        // Stone kennt das Projekt nicht und bekommt einen eigenen Platz.
        assert_eq!(projekt[boxen[3].cls], "Stone");
        assert_eq!(neu, vec!["Stone"]);
        assert!(unbekannt.is_empty());

        // Und das Entscheidende: keine Box ist auf "Ski" gelandet.
        assert!(boxen.iter().all(|x| x.cls != 0), "eine Box wurde zu Ski");
    }

    #[test]
    fn geometrie_bleibt_beim_umschreiben_unangetastet() {
        let mut projekt = vec!["a".to_string()];
        let quelle = vec!["b".to_string()];
        let (boxen, _, _) = remap_boxes(
            &[BoxAnn { cls: 0, x: 0.25, y: 0.75, w: 0.1, h: 0.2 }], &quelle, &mut projekt);
        assert_eq!(boxen[0].x, 0.25);
        assert_eq!(boxen[0].y, 0.75);
        assert_eq!(boxen[0].w, 0.1);
        assert_eq!(boxen[0].h, 0.2);
    }

    #[test]
    fn id_jenseits_der_namensliste_wird_gemeldet_statt_verschluckt() {
        let mut projekt: Vec<String> = vec![];
        let quelle = vec!["Tree".to_string()];
        let (boxen, neu, unbekannt) = remap_boxes(
            &[BoxAnn { cls: 4, x: 0.5, y: 0.5, w: 0.2, h: 0.2 }], &quelle, &mut projekt);
        assert_eq!(boxen.len(), 1, "die Box darf nicht verschwinden");
        assert_eq!(projekt[boxen[0].cls], "Klasse 4");
        assert_eq!(neu, vec!["Klasse 4"]);
        assert_eq!(unbekannt, vec![4]);
    }

    #[test]
    fn modellklassen_finden_die_projektklassen_trotz_schreibweise() {
        // Die echten Namen: links was das Ski-Modell meldet, rechts was im
        // Projekt steht. Ohne den Vergleich ohne Gross-/Kleinschreibung haette
        // von sechs Klassen nur "Emptyspace" gepasst.
        let projekt: Vec<String> = ["Ski", "Emptyspace", "generallobstacle", "tree", "person", "sky"]
            .iter().map(|s| s.to_string()).collect();

        assert_eq!(class_index_for("Tree", &projekt), Some(3));
        assert_eq!(class_index_for("Person", &projekt), Some(4));
        assert_eq!(class_index_for("Sky", &projekt), Some(5));
        assert_eq!(class_index_for("Emptyspace", &projekt), Some(1));
        assert_eq!(class_index_for("Generallobstacle", &projekt), Some(2));

        // Was das Modell kennt und das Projekt nicht, darf nicht auf gut Glueck
        // irgendeiner Klasse zugeschlagen werden.
        assert_eq!(class_index_for("Offroad", &projekt), None);
        assert_eq!(class_index_for("Slopeborder", &projekt), None);
    }

    #[test]
    fn zuordnung_kommt_mit_umlauten_und_leerzeichen_klar() {
        let projekt = vec!["Bäume".to_string(), "Piste".to_string()];
        assert_eq!(class_index_for("BÄUME", &projekt), Some(0));
        assert_eq!(class_index_for("  piste  ", &projekt), Some(1));
        assert_eq!(class_index_for("Baeume", &projekt), None);
    }

    #[test]
    fn die_liste_liefert_die_zaehler_mit() {
        // Genau das fehlte: project.json soll die Zaehler nicht enthalten,
        // die Antwort an die Oberflaeche aber sehr wohl. Ein serde-Attribut
        // am Projekt-Typ trifft beides — deshalb zwei Typen, und deshalb
        // dieser Test.
        let project = StudioProject {
            id: "sp_1".to_string(), name: "Ski".to_string(), modality: "image".to_string(),
            task: "bbox".to_string(), target_format: "yolo_bbox".to_string(),
            classes: vec!["Ski".to_string()],
            created_at: "x".to_string(), updated_at: "y".to_string(),
        };

        let gespeichert = serde_json::to_value(&project).unwrap();
        assert!(gespeichert.get("sample_count").is_none(),
            "Zaehler haben in project.json nichts verloren: {}", gespeichert);

        let antwort = serde_json::to_value(StudioProjectView {
            project, sample_count: 11, confirmed_count: 10,
        }).unwrap();
        assert_eq!(antwort["sample_count"], 11);
        assert_eq!(antwort["confirmed_count"], 10);
        // flatten darf die Projektfelder nicht verschlucken.
        assert_eq!(antwort["name"], "Ski");
        assert_eq!(antwort["classes"][0], "Ski");
    }

    #[test]
    fn csv_feld_mit_komma_wird_gequotet() {
        assert_eq!(csv_field("a,b"), "\"a,b\"");
        assert_eq!(csv_field("sagt \"hallo\""), "\"sagt \"\"hallo\"\"\"");
        assert_eq!(csv_field("schlicht"), "schlicht");
    }
}
