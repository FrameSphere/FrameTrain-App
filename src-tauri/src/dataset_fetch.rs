// dataset_fetch – Datensaetze gezielt holen und pruefen (python/ft_data/fetch.py)
//
//   hf_dataset_files   Ordner, Groessen, Formate eines Hugging-Face-Datensatzes
//   fetch_dataset      Hugging Face (mit Unterordner/Muster, Groessenlimit, YOLO-Umwandlung)
//                      oder Archiv von einer https-URL holen und als Dataset importieren
//   inspect_dataset    Was steckt in einem Dataset? Bilder, Labels, Klassen, Probleme
//
// Der eigentliche Import laeuft ueber import_local_dataset – dieselben Regeln
// (Splits, dataset.yaml, Metadaten) wie beim Import eines lokalen Ordners.

use crate::command_ext::{NoWindow, PythonUtf8};
use crate::dataset_manager::{get_dataset_path, import_local_dataset};
use crate::AppState;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime};
use tauri::{Emitter, Manager, State};

pub const PROGRESS_EVENT: &str = "dataset-fetch-progress";
pub const CANCELLED_TEXT: &str = "Download cancelled.";

// Laufende Python-Prozesse je fetch_id, damit ein Download abgebrochen werden kann.
type Running = Mutex<HashMap<String, Arc<Mutex<Child>>>>;

fn running() -> &'static Running {
    static R: OnceLock<Running> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

fn cancelled() -> &'static Mutex<HashSet<String>> {
    static C: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Bricht einen laufenden Download ab. true = es lief einer.
pub fn cancel(fetch_id: &str) -> bool {
    let child = running().lock().unwrap().get(fetch_id).cloned();
    match child {
        Some(c) => {
            cancelled().lock().unwrap().insert(fetch_id.to_string());
            let _ = c.lock().unwrap().kill();
            true
        }
        None => false,
    }
}

/// Ergebnis aus den JSON-Zeilen von ft_data.fetch (letzte result- oder error-Zeile).
pub fn parse_fetch_output(stdout: &str) -> Result<Value, String> {
    let mut last: Option<Value> = None;
    for line in stdout.lines() {
        if let Ok(v) = serde_json::from_str::<Value>(line) {
            match v.get("type").and_then(|t| t.as_str()) {
                Some("result") | Some("error") => last = Some(v),
                _ => {}
            }
        }
    }
    match last {
        Some(v) if v.get("type").and_then(|t| t.as_str()) == Some("result") => {
            let mut v = v;
            if let Some(o) = v.as_object_mut() { o.remove("type"); }
            Ok(v)
        }
        Some(v) => Err(v.get("message").and_then(|m| m.as_str()).unwrap_or("Unbekannter Fehler").to_string()),
        None => Err("Keine Antwort vom Python-Helfer".to_string()),
    }
}

fn run_fetch(app: &tauri::AppHandle, request: Value, fetch_id: Option<String>) -> Result<Value, String> {
    run_module_tracked(app, "ft_data.fetch", request, fetch_id)
}

fn run_module(app: &tauri::AppHandle, module: &str, request: Value) -> Result<Value, String> {
    run_module_tracked(app, module, request, None)
}

fn run_module_tracked(app: &tauri::AppHandle, module: &str, request: Value, fetch_id: Option<String>) -> Result<Value, String> {
    let mut cmd = Command::new(crate::training_manager::resolve_python_path());
    cmd.no_window();
    cmd.python_utf8();
    if let Some(root) = crate::training_manager::python_root(app) {
        let mut paths = vec![root];
        if let Some(existing) = std::env::var_os("PYTHONPATH") {
            paths.extend(std::env::split_paths(&existing));
        }
        if let Ok(joined) = std::env::join_paths(paths) {
            cmd.env("PYTHONPATH", joined);
        }
    }
    cmd.args(["-m", module, &request.to_string()]);
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| format!("Python konnte nicht gestartet werden: {}", e))?;
    let stdout = child.stdout.take().ok_or("stdout fehlt")?;
    let mut stderr = child.stderr.take().ok_or("stderr fehlt")?;
    let child = Arc::new(Mutex::new(child));
    if let Some(id) = &fetch_id {
        running().lock().unwrap().insert(id.clone(), child.clone());
    }
    let err_thread = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = stderr.read_to_string(&mut s);
        s
    });
    let mut collected = String::new();
    for line in BufReader::new(stdout).lines().map_while(Result::ok) {
        if let Ok(v) = serde_json::from_str::<Value>(&line) {
            if matches!(v.get("type").and_then(|t| t.as_str()), Some("status") | Some("progress")) {
                let _ = app.emit(PROGRESS_EVENT, &v);
            }
        }
        collected.push_str(&line);
        collected.push('\n');
    }
    let _ = child.lock().unwrap().wait();
    let err = err_thread.join().unwrap_or_default();
    if let Some(id) = &fetch_id {
        running().lock().unwrap().remove(id);
        if cancelled().lock().unwrap().remove(id) {
            return Err(CANCELLED_TEXT.to_string());
        }
    }
    parse_fetch_output(&collected).map_err(|e| {
        if e == "Keine Antwort vom Python-Helfer" {
            let last = err.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("");
            format!("{} {}", e, last).trim().to_string()
        } else {
            e
        }
    })
}

async fn run_fetch_async(app: tauri::AppHandle, request: Value) -> Result<Value, String> {
    run_fetch_tracked(app, request, None).await
}

async fn run_fetch_tracked(app: tauri::AppHandle, request: Value, fetch_id: Option<String>) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || run_fetch(&app, request, fetch_id))
        .await
        .map_err(|e| format!("Abgebrochen: {}", e))?
}

/// Ordner, Groessen und Formate eines Hugging-Face-Datensatzes – vor dem Download.
#[tauri::command]
pub async fn hf_dataset_files(app_handle: tauri::AppHandle, repo_id: String) -> Result<Value, String> {
    if !is_safe_repo_id(&repo_id) {
        return Err("Ungültige Repo-ID".into());
    }
    run_fetch_async(app_handle, json!({ "kind": "hf_files", "repo_id": repo_id })).await
}

/// Holt einen Datensatz (source.kind = "hf" | "url") und importiert ihn fuer ein Modell.
/// source: { kind, repo_id?, include?, url?, max_mb?, max_rows?, bbox_format?, yolo? }
#[tauri::command]
pub async fn fetch_dataset(
    app_handle: tauri::AppHandle,
    state: State<'_, AppState>,
    source: Value,
    dataset_name: String,
    model_id: String,
    fetch_id: Option<String>,
) -> Result<Value, String> {
    let kind = source.get("kind").and_then(|k| k.as_str()).unwrap_or("");
    match kind {
        "hf" => {
            let repo = source.get("repo_id").and_then(|r| r.as_str()).unwrap_or("");
            if !is_safe_repo_id(repo) { return Err("Ungültige Repo-ID".into()); }
        }
        "url" => {
            // Der MCP-Import schickt "urls" (Liste), der Dialog "url". Vorher wurde nur
            // "url" geprueft: jede MCP-Anfrage scheiterte mit "Nur https://".
            let urls = source_urls(&source);
            if urls.is_empty() { return Err("Keine Adresse angegeben".into()); }
            if let Some(bad) = urls.iter().find(|u| !u.to_ascii_lowercase().starts_with("https://")) {
                return Err(format!("Nur https://-Adressen sind erlaubt: {}", bad));
            }
        }
        _ => return Err("source.kind muss \"hf\" oder \"url\" sein".into()),
    }
    let work = temp_dir(&app_handle)?;
    let mut req = source.clone();
    if let Some(o) = req.as_object_mut() {
        o.insert("target".into(), json!(work.to_string_lossy()));
    }
    let result = run_fetch_tracked(app_handle.clone(), req, fetch_id).await;
    let outcome = match result {
        Ok(v) => {
            let root = v.get("root").and_then(|r| r.as_str()).unwrap_or("").to_string();
            match import_local_dataset(app_handle.clone(), state, root, dataset_name, model_id).await {
                Ok(dataset) => Ok(json!({ "dataset": dataset, "report": v.get("report"), "conversion": v.get("conversion") })),
                Err(e) => Err(e),
            }
        }
        Err(e) => Err(e),
    };
    let _ = std::fs::remove_dir_all(&work);
    outcome
}

/// Inhalt eines vorhandenen Datasets pruefen (Bilder, Labels, Klassen, Probleme).
#[tauri::command]
pub async fn inspect_dataset(
    app_handle: tauri::AppHandle,
    state: State<'_, AppState>,
    dataset_id: String,
) -> Result<Value, String> {
    let path = get_dataset_path(app_handle.clone(), state, dataset_id).await?;
    run_fetch_async(app_handle, json!({ "kind": "inspect", "path": path })).await
}

/// Ein Dataset fuer den Dataset Builder vorbereiten (Klassen umbenennen/zusammenlegen/
/// weglassen, Stichprobe). request: { project_kind, class_map?, keep_unmapped?, max_samples?,
/// per_class?, keep_empty?, text_column?, label_column? }. Ergebnis enthaelt den Pfad und
/// welcher Studio-Import ihn liest; danach discard_staged aufrufen.
#[tauri::command]
pub async fn stage_dataset(
    app_handle: tauri::AppHandle,
    state: State<'_, AppState>,
    dataset_id: String,
    request: Value,
) -> Result<Value, String> {
    let source = get_dataset_path(app_handle.clone(), state, dataset_id).await?;
    let work = temp_dir(&app_handle)?;
    let mut req = request;
    if let Some(o) = req.as_object_mut() {
        o.insert("source".into(), json!(source));
        o.insert("target".into(), json!(work.to_string_lossy()));
    }
    let app = app_handle.clone();
    let res = tauri::async_runtime::spawn_blocking(move || run_module(&app, "ft_data.stage", req))
        .await
        .map_err(|e| format!("Abgebrochen: {}", e))?;
    match res {
        Ok(mut v) => {
            if let Some(o) = v.as_object_mut() {
                o.insert("work_dir".into(), json!(work.to_string_lossy()));
            }
            Ok(v)
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&work);
            Err(e)
        }
    }
}

/// Raeumt einen Arbeitsordner von stage_dataset weg – nur innerhalb des eigenen Cache-Ordners.
#[tauri::command]
pub fn discard_staged(app_handle: tauri::AppHandle, work_dir: String) -> Result<(), String> {
    let base = app_handle.path().app_cache_dir().map_err(|e| e.to_string())?.join("dataset-fetch");
    let p = PathBuf::from(&work_dir);
    let (Ok(base), Ok(p)) = (base.canonicalize(), p.canonicalize()) else { return Ok(()) };
    if p.starts_with(&base) && p != base {
        std::fs::remove_dir_all(&p).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Vorschaubild (Bild oder Video-Frame) mit eingezeichneten Boxen als base64-JPEG –
/// damit KI-Assistenten Labels sehen und pruefen koennen. Nur Dateien unter den
/// App-Daten (Datasets, Dataset-Builder-Projekte).
#[tauri::command]
pub async fn preview_media(
    app_handle: tauri::AppHandle,
    path: String,
    boxes: Option<Value>,
    classes: Option<Vec<String>>,
    max_size: Option<u32>,
    at: Option<f64>,
) -> Result<Value, String> {
    let data = app_handle.path().app_data_dir().map_err(|e| e.to_string())?;
    let p = PathBuf::from(&path).canonicalize().map_err(|_| "Datei nicht gefunden".to_string())?;
    let allowed = data.canonicalize().map(|d| p.starts_with(d)).unwrap_or(false);
    if !allowed {
        return Err("Vorschau nur fuer Dateien aus FrameTrain-Datasets und -Projekten".into());
    }
    let mut req = json!({ "path": p.to_string_lossy(), "max_size": max_size.unwrap_or(1024) });
    if let Some(b) = boxes { req["boxes"] = b; }
    if let Some(c) = classes { req["classes"] = json!(c); }
    if let Some(a) = at { req["at"] = json!(a); }
    let app = app_handle.clone();
    tauri::async_runtime::spawn_blocking(move || run_module(&app, "ft_data.preview", req))
        .await
        .map_err(|e| format!("Abgebrochen: {}", e))?
}

/// Stoppt einen laufenden fetch_dataset (gleiche fetch_id). Der halb geladene
/// Ordner wird danach von fetch_dataset selbst geloescht.
#[tauri::command]
pub fn cancel_dataset_fetch(fetch_id: String) -> bool {
    cancel(&fetch_id)
}

/// Juengste Aenderung in einem Ordnerbaum (fuer das Aufraeumen beim Start).
fn newest_mtime(dir: &Path) -> Option<SystemTime> {
    let mut newest = std::fs::metadata(dir).and_then(|m| m.modified()).ok();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for e in entries.flatten() {
            let Ok(meta) = e.metadata() else { continue };
            if let Ok(m) = meta.modified() {
                if newest.map(|n| m > n).unwrap_or(true) {
                    newest = Some(m);
                }
            }
            if meta.is_dir() {
                stack.push(e.path());
            }
        }
    }
    newest
}

/// Loescht liegengebliebene Download-Ordner (App waehrend eines Downloads beendet).
/// Nur Ordner ohne Aenderung seit `idle` – ein laufender Download schreibt staendig.
pub fn cleanup_stale(base: &Path, idle: Duration) -> usize {
    let Ok(entries) = std::fs::read_dir(base) else { return 0 };
    let now = SystemTime::now();
    let mut removed = 0;
    for e in entries.flatten() {
        let p = e.path();
        if !p.is_dir() {
            continue;
        }
        let old = newest_mtime(&p)
            .and_then(|m| now.duration_since(m).ok())
            .map(|age| age >= idle)
            .unwrap_or(false);
        if old && std::fs::remove_dir_all(&p).is_ok() {
            removed += 1;
        }
    }
    removed
}

pub fn cleanup_stale_on_start(app: &tauri::AppHandle) {
    if let Ok(base) = app.path().app_cache_dir() {
        let base = base.join("dataset-fetch");
        std::thread::spawn(move || {
            let n = cleanup_stale(&base, Duration::from_secs(30 * 60));
            if n > 0 {
                println!("[dataset-fetch] {} liegengebliebene Download-Ordner entfernt", n);
            }
        });
    }
}

fn temp_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let base = app.path().app_cache_dir().map_err(|e| format!("Cache-Ordner: {}", e))?;
    let dir = base.join("dataset-fetch").join(uuid::Uuid::new_v4().to_string());
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {}", e))?;
    Ok(dir)
}

/// Adressen eines URL-Imports: "urls" (Liste, MCP) und/oder "url" (einzeln, Dialog).
pub fn source_urls(source: &Value) -> Vec<String> {
    let mut out: Vec<String> = source.get("urls").and_then(|u| u.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str()).map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect())
        .unwrap_or_default();
    if let Some(u) = source.get("url").and_then(|u| u.as_str()).map(str::trim).filter(|s| !s.is_empty()) {
        if !out.iter().any(|x| x == u) { out.push(u.to_string()); }
    }
    out
}

pub fn is_safe_repo_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 200
        && !id.contains("..")
        && id.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/'))
        && id.matches('/').count() <= 1
        && !id.starts_with('/')
        && !id.ends_with('/')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_result_and_error_lines() {
        let ok = "{\"type\":\"status\",\"message\":\"x\"}\n{\"type\":\"result\",\"root\":\"/r\",\"report\":{}}\n";
        assert_eq!(parse_fetch_output(ok).unwrap()["root"], "/r");
        let err = "{\"type\":\"progress\"}\n{\"type\":\"error\",\"message\":\"too big\"}\n";
        assert_eq!(parse_fetch_output(err).unwrap_err(), "too big");
        assert!(parse_fetch_output("Traceback ...").is_err());
    }

    #[test]
    fn url_liste_und_einzeladresse() {
        let mcp = serde_json::json!({"kind": "url", "urls": ["https://a/x.zip", " https://b/y.zip "]});
        assert_eq!(source_urls(&mcp), vec!["https://a/x.zip", "https://b/y.zip"]);
        let dialog = serde_json::json!({"kind": "url", "url": "https://a/x.zip"});
        assert_eq!(source_urls(&dialog), vec!["https://a/x.zip"]);
        assert!(source_urls(&serde_json::json!({"kind": "url"})).is_empty());
    }

    #[test]
    fn repo_ids() {
        assert!(is_safe_repo_id("rishitdagli/cppe-5"));
        assert!(is_safe_repo_id("imdb"));
        assert!(!is_safe_repo_id("../etc"));
        assert!(!is_safe_repo_id("a/b/c"));
        assert!(!is_safe_repo_id("a b"));
    }
}

#[cfg(test)]
mod cleanup_tests {
    use super::*;

    #[test]
    fn raeumt_nur_alte_ordner() {
        let base = std::env::temp_dir().join(format!("ft-cleanup-{}", uuid::Uuid::new_v4()));
        let fresh = base.join("fresh");
        std::fs::create_dir_all(fresh.join("raw")).unwrap();
        std::fs::write(fresh.join("raw/a.jpg"), b"x").unwrap();
        // idle 0: alles gilt als alt
        assert_eq!(cleanup_stale(&base, Duration::from_secs(3600)), 0);
        assert!(fresh.exists());
        assert_eq!(cleanup_stale(&base, Duration::from_secs(0)), 1);
        assert!(!fresh.exists());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[cfg(unix)]
    #[test]
    fn abbruch_beendet_den_prozess() {
        let child = Command::new("sleep").arg("30").spawn().unwrap();
        let c = Arc::new(Mutex::new(child));
        running().lock().unwrap().insert("t1".into(), c.clone());
        let t = std::time::Instant::now();
        assert!(cancel("t1"));
        let status = c.lock().unwrap().wait().unwrap();
        assert!(!status.success());
        assert!(t.elapsed() < Duration::from_secs(5));
        assert!(cancelled().lock().unwrap().remove("t1"));
        running().lock().unwrap().remove("t1");
    }

    #[test]
    fn abbruch_ohne_laufenden_download() {
        assert!(!cancel("gibt-es-nicht"));
    }
}
