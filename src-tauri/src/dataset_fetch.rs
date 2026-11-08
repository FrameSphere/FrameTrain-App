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
use std::path::PathBuf;
use std::process::{Command, Stdio};
use tauri::{Emitter, Manager, State};

pub const PROGRESS_EVENT: &str = "dataset-fetch-progress";

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

fn run_fetch(app: &tauri::AppHandle, request: Value) -> Result<Value, String> {
    run_module(app, "ft_data.fetch", request)
}

fn run_module(app: &tauri::AppHandle, module: &str, request: Value) -> Result<Value, String> {
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
    let _ = child.wait();
    let err = err_thread.join().unwrap_or_default();
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
    tauri::async_runtime::spawn_blocking(move || run_fetch(&app, request))
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
) -> Result<Value, String> {
    let kind = source.get("kind").and_then(|k| k.as_str()).unwrap_or("");
    match kind {
        "hf" => {
            let repo = source.get("repo_id").and_then(|r| r.as_str()).unwrap_or("");
            if !is_safe_repo_id(repo) { return Err("Ungültige Repo-ID".into()); }
        }
        "url" => {
            let url = source.get("url").and_then(|u| u.as_str()).unwrap_or("");
            if !url.to_ascii_lowercase().starts_with("https://") { return Err("Nur https://-Adressen sind erlaubt".into()); }
        }
        _ => return Err("source.kind muss \"hf\" oder \"url\" sein".into()),
    }
    let work = temp_dir(&app_handle)?;
    let mut req = source.clone();
    if let Some(o) = req.as_object_mut() {
        o.insert("target".into(), json!(work.to_string_lossy()));
    }
    let result = run_fetch_async(app_handle.clone(), req).await;
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

fn temp_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let base = app.path().app_cache_dir().map_err(|e| format!("Cache-Ordner: {}", e))?;
    let dir = base.join("dataset-fetch").join(uuid::Uuid::new_v4().to_string());
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {}", e))?;
    Ok(dir)
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
    fn repo_ids() {
        assert!(is_safe_repo_id("rishitdagli/cppe-5"));
        assert!(is_safe_repo_id("imdb"));
        assert!(!is_safe_repo_id("../etc"));
        assert!(!is_safe_repo_id("a/b/c"));
        assert!(!is_safe_repo_id("a b"));
    }
}
