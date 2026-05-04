// laboratory_manager.rs – Persistenter Model-Server fuer Lab-Inferenz
//
// Architektur: Rust startet einmalig einen Python-Prozess der das Modell
// laedt und dann via stdin/stdout auf Inferenz-Anfragen wartet.
// Jeder Sample-Test braucht nur noch ~50ms statt 3-5s.

use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::{Emitter, Manager};

// ============ Typen ============

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ServerStatus {
    Idle,
    Loading,
    Ready,
    Error,
}

impl Default for ServerStatus {
    fn default() -> Self { ServerStatus::Idle }
}

pub struct LabServer {
    pub child:      Child,
    pub stdin:      std::io::BufWriter<std::process::ChildStdin>,
    pub receiver:   std::sync::mpsc::Receiver<String>,
    pub version_id: String,
    pub model_path: String,
}

#[derive(Default)]
pub struct LabState {
    pub server: Option<LabServer>,
    pub status: ServerStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferResult {
    pub predicted:        String,
    pub confidence:       Option<f64>,
    pub top_predictions:  Option<Vec<serde_json::Value>>,
    pub inference_ms:     f64,
}

// ============ Hilfsfunktionen ============

fn get_python_path() -> String {
    struct C { path: String, version: (u32, u32, u32) }
    let mut candidates: Vec<C> = Vec::new();

    if !cfg!(target_os = "windows") {
        for base in &["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"] {
            for name in &["python3.13", "python3.12", "python3.11", "python3.10", "python3.9", "python3"] {
                let full = format!("{}/{}", base, name);
                if let Ok(out) = Command::new(&full).arg("--version").output() {
                    if out.status.success() {
                        let vs = String::from_utf8_lossy(&out.stdout);
                        let vs2 = String::from_utf8_lossy(&out.stderr);
                        let combined = format!("{}{}", vs, vs2);
                        if let Some(v) = parse_version(&combined) {
                            candidates.push(C { path: full, version: v });
                        }
                    }
                }
            }
        }
    }
    for cmd in &["python3", "python"] {
        if let Ok(out) = Command::new(cmd).arg("--version").output() {
            if out.status.success() {
                let vs = String::from_utf8_lossy(&out.stdout);
                let vs2 = String::from_utf8_lossy(&out.stderr);
                let combined = format!("{}{}", vs, vs2);
                if let Some(v) = parse_version(&combined) {
                    candidates.push(C { path: cmd.to_string(), version: v });
                }
            }
        }
    }
    candidates.sort_by(|a, b| b.version.cmp(&a.version));
    candidates.dedup_by(|a, b| a.version == b.version);
    for c in &candidates {
        let ok = Command::new(&c.path).args(["-c", "import torch"]).output()
            .map(|o| o.status.success()).unwrap_or(false);
        if ok { return c.path.clone(); }
    }
    candidates.first().map(|c| c.path.clone())
        .unwrap_or_else(|| if cfg!(target_os = "windows") { "python".to_string() } else { "python3".to_string() })
}

fn parse_version(s: &str) -> Option<(u32, u32, u32)> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() < 2 { return None; }
    let nums: Vec<&str> = parts[1].split('.').collect();
    if nums.len() < 2 { return None; }
    let major = nums[0].parse::<u32>().ok()?;
    let minor = nums[1].parse::<u32>().ok()?;
    let patch = nums.get(2)
        .and_then(|p| p.trim_end_matches(|c: char| !c.is_ascii_digit()).parse::<u32>().ok())
        .unwrap_or(0);
    Some((major, minor, patch))
}

fn get_model_server_path(app_handle: &tauri::AppHandle) -> Result<std::path::PathBuf, String> {
    let candidates = vec![
        app_handle.path().resource_dir().ok()
            .map(|p| p.join("python").join("test_engine").join("model_server.py")),
        Some(std::path::PathBuf::from("src-tauri/python/test_engine/model_server.py")),
        Some(std::path::PathBuf::from(
            "/Users/karol/Desktop/Laufende_Projekte/FrameTrain/desktop-app/src-tauri/python/test_engine/model_server.py"
        )),
    ];
    for p in candidates.into_iter().flatten() {
        if p.exists() {
            println!("[LabServer] Script gefunden: {:?}", p);
            return Ok(p);
        }
    }
    Err("model_server.py nicht gefunden".to_string())
}

fn get_version_path(app_handle: &tauri::AppHandle, version_id: &str) -> Result<String, String> {
    let db_path = app_handle.path().app_data_dir()
        .map_err(|e| format!("AppDataDir: {}", e))?
        .join("frametrain.db");
    let conn = rusqlite::Connection::open(&db_path)
        .map_err(|e| format!("DB: {}", e))?;
    conn.query_row(
        "SELECT path FROM model_versions_new WHERE id = ?1",
        [version_id],
        |r| r.get(0),
    ).map_err(|e| format!("Version nicht gefunden: {}", e))
}

// ============ Commands ============

/// Startet (oder ersetzt) den persistenten Modell-Server fuer eine Version.
/// Kehrt sofort zurueck; der eigentliche Start laeuft im Hintergrund.
/// Events: "lab-server-status" { status: "loading" | "ready" | "error", message?, version_id? }
#[tauri::command]
pub async fn lab_start_model_server(
    app_handle: tauri::AppHandle,
    version_id: String,
    state: tauri::State<'_, Arc<Mutex<LabState>>>,
) -> Result<(), String> {
    // Alten Server beenden
    {
        let mut s = state.lock().map_err(|e| format!("Lock: {}", e))?;
        if let Some(ref mut srv) = s.server {
            let _ = srv.child.kill();
        }
        s.server = None;
        s.status = ServerStatus::Loading;
    }

    let _ = app_handle.emit("lab-server-status", serde_json::json!({ "status": "loading" }));

    let model_path = match get_version_path(&app_handle, &version_id) {
        Ok(p) => p,
        Err(e) => {
            let _ = app_handle.emit("lab-server-status",
                serde_json::json!({ "status": "error", "message": e }));
            let mut s = state.lock().unwrap();
            s.status = ServerStatus::Error;
            return Err(e);
        }
    };

    let python        = get_python_path();
    let server_script = match get_model_server_path(&app_handle) {
        Ok(p) => p,
        Err(e) => {
            let _ = app_handle.emit("lab-server-status",
                serde_json::json!({ "status": "error", "message": e }));
            let mut s = state.lock().unwrap();
            s.status = ServerStatus::Error;
            return Err(e);
        }
    };

    // Hintergrund-Thread fuer den blockierenden Startup
    let state_arc = Arc::clone(&*state);
    let ah        = app_handle.clone();
    let vid       = version_id.clone();
    let mp        = model_path.clone();

    std::thread::spawn(move || {
        println!("[LabServer] Starte Python: {} --model-path {}", python, mp);

        let mut child = match Command::new(&python)
            .arg(server_script.to_string_lossy().to_string())
            .arg("--model-path").arg(&mp)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                let msg = format!("Python konnte nicht gestartet werden: {}", e);
                let _ = ah.emit("lab-server-status", serde_json::json!({ "status": "error", "message": msg }));
                if let Ok(mut s) = state_arc.lock() { s.status = ServerStatus::Error; }
                return;
            }
        };

        // Stderr in separatem Thread loggen
        if let Some(stderr) = child.stderr.take() {
            std::thread::spawn(move || {
                for line in BufReader::new(stderr).lines().flatten() {
                    eprintln!("[LabServer STDERR] {}", line);
                }
            });
        }

        let stdin = match child.stdin.take() {
            Some(s) => s,
            None => {
                let _ = child.kill();
                let _ = ah.emit("lab-server-status", serde_json::json!({ "status": "error", "message": "Kein stdin" }));
                return;
            }
        };

        let stdout = match child.stdout.take() {
            Some(s) => s,
            None => {
                let _ = child.kill();
                let _ = ah.emit("lab-server-status", serde_json::json!({ "status": "error", "message": "Kein stdout" }));
                return;
            }
        };

        // stdout-Lese-Thread -> Channel
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().flatten() {
                if tx.send(line).is_err() { break; }
            }
        });

        // Auf "ready" warten (max. 120 Sekunden – grosse Modelle auf CPU brauchen Zeit)
        let deadline = Instant::now() + Duration::from_secs(120);
        let mut ready = false;

        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                let _ = child.kill();
                let msg = "Timeout beim Laden des Modells (120s). Versuche es erneut.".to_string();
                let _ = ah.emit("lab-server-status", serde_json::json!({ "status": "error", "message": msg }));
                if let Ok(mut s) = state_arc.lock() { s.status = ServerStatus::Error; }
                return;
            }

            match rx.recv_timeout(remaining) {
                Ok(line) => {
                    let line = line.trim().to_string();
                    println!("[LabServer] Startup-Zeile: {}", line);
                    if let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) {
                        match msg.get("type").and_then(|t| t.as_str()) {
                            Some("ready") => { ready = true; break; }
                            Some("error") => {
                                let m = msg.get("message").and_then(|m| m.as_str())
                                    .unwrap_or("Unbekannter Fehler").to_string();
                                let _ = child.kill();
                                let _ = ah.emit("lab-server-status", serde_json::json!({ "status": "error", "message": m }));
                                if let Ok(mut s) = state_arc.lock() { s.status = ServerStatus::Error; }
                                return;
                            }
                            _ => { /* Ignoriere andere Nachrichten waehrend Startup */ }
                        }
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    let _ = child.kill();
                    let _ = ah.emit("lab-server-status", serde_json::json!({ "status": "error", "message": "Timeout beim Modell-Laden" }));
                    if let Ok(mut s) = state_arc.lock() { s.status = ServerStatus::Error; }
                    return;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    let _ = ah.emit("lab-server-status", serde_json::json!({ "status": "error", "message": "Server-Prozess unerwartet beendet" }));
                    if let Ok(mut s) = state_arc.lock() { s.status = ServerStatus::Error; }
                    return;
                }
            }
        }

        if ready {
            if let Ok(mut s) = state_arc.lock() {
                s.server = Some(LabServer {
                    child,
                    stdin: std::io::BufWriter::new(stdin),
                    receiver: rx,
                    version_id: vid.clone(),
                    model_path: mp,
                });
                s.status = ServerStatus::Ready;
            }
            let _ = ah.emit("lab-server-status",
                serde_json::json!({ "status": "ready", "version_id": vid }));
            println!("[LabServer] Bereit fuer Inferenz.");
        }
    });

    Ok(())
}

/// Fuehrt Inferenz auf einem einzelnen Text durch.
/// Schnell (~50ms) weil das Modell bereits geladen ist.
#[tauri::command]
pub fn lab_infer_sample(
    text: String,
    state: tauri::State<'_, Arc<Mutex<LabState>>>,
) -> Result<InferResult, String> {
    let mut s = state.lock().map_err(|e| format!("Lock: {}", e))?;

    // Schreiben + Lesen atomar (Mutex haelt waehrend beider Operationen)
    let recv_result = {
        let server = s.server.as_mut()
            .ok_or_else(|| "Kein Modell geladen. Bitte warte bis das Modell fertig geladen ist.".to_string())?;

        let req = serde_json::json!({ "text": text }).to_string();
        writeln!(server.stdin, "{}", req).map_err(|e| format!("Schreibfehler: {}", e))?;
        server.stdin.flush().map_err(|e| format!("Flush-Fehler: {}", e))?;

        // Auf Antwort warten (max. 30s)
        server.receiver.recv_timeout(Duration::from_secs(30))
    }; // server-Borrow endet hier

    match recv_result {
        Ok(line) => {
            let resp: serde_json::Value = serde_json::from_str(line.trim())
                .map_err(|e| format!("JSON parse: {} (Zeile: {})", e, line))?;

            if let Some("error") = resp.get("type").and_then(|t| t.as_str()) {
                return Err(resp.get("message").and_then(|m| m.as_str())
                    .unwrap_or("Unbekannter Inferenz-Fehler").to_string());
            }

            Ok(InferResult {
                predicted: resp["predicted"].as_str().unwrap_or("?").to_string(),
                confidence: resp["confidence"].as_f64(),
                top_predictions: resp["top_predictions"].as_array().cloned(),
                inference_ms: resp["inference_time"].as_f64().unwrap_or(0.0) * 1000.0,
            })
        }
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            Err("Inferenz-Timeout (30s) – Modell antwortet nicht. Bitte neu laden.".to_string())
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            // Prozess ist abgestuerzt – Server-Referenz bereinigen
            s.server = None;
            s.status = ServerStatus::Error;
            Err("Modell-Server ist abgestuerzt. Bitte Modell neu laden.".to_string())
        }
    }
}

/// Beendet den laufenden Modell-Server.
#[tauri::command]
pub fn lab_stop_model_server(
    state: tauri::State<'_, Arc<Mutex<LabState>>>,
) -> Result<(), String> {
    let mut s = state.lock().map_err(|e| format!("Lock: {}", e))?;
    if let Some(ref mut srv) = s.server {
        let _ = srv.child.kill();
        println!("[LabServer] Server gestoppt.");
    }
    s.server = None;
    s.status = ServerStatus::Idle;
    Ok(())
}

/// Gibt den aktuellen Server-Status zurueck.
#[tauri::command]
pub fn lab_get_server_status(
    state: tauri::State<'_, Arc<Mutex<LabState>>>,
) -> Result<serde_json::Value, String> {
    let s = state.lock().map_err(|e| format!("Lock: {}", e))?;
    Ok(serde_json::json!({
        "status": s.status,
        "version_id": s.server.as_ref().map(|srv| &srv.version_id),
        "model_path": s.server.as_ref().map(|srv| &srv.model_path),
    }))
}

/// Fuehrt ein Dev-Script fuer ein einzelnes Sample aus.
/// Script wird als Temp-Datei gespeichert, mit ENV-Variablen gestartet,
/// stdout (erste JSON-Zeile) wird als lab-script-result Event emittiert.
#[tauri::command]
pub async fn run_lab_script_sample(
    app_handle: tauri::AppHandle,
    script: String,
    sample_input: String,
    refs: std::collections::HashMap<String, String>,
) -> Result<(), String> {
    use std::io::Write as IoWrite;

    let python = get_python_path();

    // Script in temp-Datei schreiben
    let tmp_path = std::env::temp_dir()
        .join(format!("ft_lab_{}.py", uuid::Uuid::new_v4()));
    {
        let mut f = std::fs::File::create(&tmp_path)
            .map_err(|e| format!("Temp-Datei: {}", e))?;
        f.write_all(script.as_bytes())
            .map_err(|e| format!("Schreiben: {}", e))?;
    }

    let ah = app_handle.clone();
    let tp = tmp_path.clone();

    std::thread::spawn(move || {
        let mut cmd = Command::new(&python);
        cmd.arg(tp.to_string_lossy().to_string())
           .env("LAB_SAMPLE_INPUT", &sample_input)
           .stdout(Stdio::piped())
           .stderr(Stdio::piped());

        for (k, v) in &refs {
            cmd.env(k, v);
        }

        let result: Result<serde_json::Value, String> = match cmd.spawn() {
            Err(e) => {
                let _ = std::fs::remove_file(&tp);
                Err(format!("Python konnte nicht gestartet werden: {}", e))
            }
            Ok(mut child) => {
                // Stderr loggen
                if let Some(stderr) = child.stderr.take() {
                    std::thread::spawn(move || {
                        for l in BufReader::new(stderr).lines().flatten() {
                            eprintln!("[LabScript STDERR] {}", l);
                        }
                    });
                }

                // Erste JSON-Zeile aus stdout lesen
                let first_line = child.stdout.take().and_then(|s| {
                    BufReader::new(s).lines().flatten()
                        .find(|l| !l.trim().is_empty())
                });

                let _ = child.wait();
                let _ = std::fs::remove_file(&tp);

                match first_line {
                    None => Err("Skript hat keine Ausgabe produziert".to_string()),
                    Some(line) => serde_json::from_str::<serde_json::Value>(&line)
                        .map_err(|e| format!("JSON parse: {} (Output: {})", e, line)),
                }
            }
        };

        match result {
            Ok(v) => { let _ = ah.emit("lab-script-result", v); }
            Err(e) => { let _ = ah.emit("lab-script-result", serde_json::json!({ "error": e })); }
        }
    });

    Ok(())
}

// ============ Alte Stubs (unveraendert) ============

#[tauri::command]
pub async fn lab_load_sample(
    _app_handle: tauri::AppHandle,
    _version_id: String,
    _dataset_id: Option<String>,
) -> Result<serde_json::Value, String> {
    Err("Verwende lab_infer_sample fuer direkte Inferenz".to_string())
}

#[tauri::command]
pub async fn lab_run_inference(
    _app_handle: tauri::AppHandle,
    _version_id: String,
    _input: String,
) -> Result<serde_json::Value, String> {
    Err("Verwende lab_infer_sample fuer direkte Inferenz".to_string())
}

#[tauri::command]
pub async fn lab_save_session(
    _app_handle: tauri::AppHandle,
    _session: serde_json::Value,
) -> Result<String, String> {
    Err("Sessions werden im Frontend gespeichert".to_string())
}

#[tauri::command]
pub async fn lab_get_sessions(_app_handle: tauri::AppHandle) -> Result<Vec<serde_json::Value>, String> {
    Ok(vec![])
}

#[tauri::command]
pub async fn lab_delete_session(
    _app_handle: tauri::AppHandle,
    _session_id: String,
) -> Result<(), String> {
    Ok(())
}

#[tauri::command]
pub async fn lab_export_as_dataset(
    _app_handle: tauri::AppHandle,
    _session_id: String,
    _name: Option<String>,
) -> Result<serde_json::Value, String> {
    Err("Noch nicht implementiert".to_string())
}

#[tauri::command]
pub async fn lab_get_stats(_app_handle: tauri::AppHandle) -> Result<serde_json::Value, String> {
    Ok(serde_json::json!({ "total_sessions": 0, "total_inferences": 0 }))
}
