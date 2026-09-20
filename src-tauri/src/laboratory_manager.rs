// laboratory_manager.rs – Persistenter Model-Server fuer Lab-Inferenz
//
// Architektur: Rust startet einmalig einen Python-Prozess der das Modell
// laedt und dann via stdin/stdout auf Inferenz-Anfragen wartet.
// Jeder Sample-Test braucht nur noch ~50ms statt 3-5s.

use serde::{Deserialize, Serialize};
use crate::command_ext::{NoWindow, PythonUtf8};
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
    /// Canvas-Modell (DynamicGraphModule) statt HuggingFace — anderes Request-Format
    pub is_canvas:  bool,
    /// Was der Server erwartet: "text" | "image" | "audio" (vom Python-Server gemeldet)
    pub input_kind: String,
    /// Aufgabenbereich: "text" | "image" | "audio" | "seq2seq" | "canvas"
    pub modality:   String,
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
    /// Objekterkennung: Boxen in Pixelkoordinaten des Originalbildes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub boxes:            Option<Vec<serde_json::Value>>,
    /// Bildmasse zu den Boxen – ohne sie laesst sich nichts massstabsgetreu zeichnen.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_width:      Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_height:     Option<u32>,
}

// ============ Hilfsfunktionen ============

fn get_python_path() -> String {
    // Gemeinsame Auswahl fuer Training, Tests, Labor und Einrichtung.
    crate::python_env::resolve_python()
}

pub(crate) fn get_model_server_path(app_handle: &tauri::AppHandle) -> Result<std::path::PathBuf, String> {
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
    get_version_info(app_handle, version_id).map(|(p, _)| p)
}

/// Liefert (Versions-Pfad, model_id) — model_id wird für Canvas-Modelle gebraucht,
/// deren Inferenz-Dateien im Modell-Ordner liegen (nicht zwingend im Versions-Pfad).
pub(crate) fn get_version_info(app_handle: &tauri::AppHandle, version_id: &str) -> Result<(String, String), String> {
    let db_path = app_handle.path().app_data_dir()
        .map_err(|e| format!("AppDataDir: {}", e))?
        .join("frametrain.db");
    let conn = rusqlite::Connection::open(&db_path)
        .map_err(|e| format!("DB: {}", e))?;
    conn.query_row(
        "SELECT path, model_id FROM model_versions_new WHERE id = ?1",
        [version_id],
        |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
    ).map_err(|e| format!("Version nicht gefunden: {}", e))
}

/// Script für Canvas-Modelle (gleiches stdin/stdout-Protokoll wie model_server.py)
/// Pfad zum YOLO-Server – dritter Servertyp neben HuggingFace und Canvas.
pub(crate) fn get_yolo_server_path(app_handle: &tauri::AppHandle) -> Result<std::path::PathBuf, String> {
    let rel = std::path::Path::new("python").join("train_engine").join("plugins")
        .join("yolo").join("yolo_inference_server.py");
    let candidates = vec![
        app_handle.path().resource_dir().ok().map(|p| p.join(&rel)),
        Some(std::path::PathBuf::from("src-tauri").join(&rel)),
        Some(std::path::PathBuf::from(
            "/Users/karol/Desktop/Laufende_Projekte/FrameTrain/desktop-app/src-tauri"
        ).join(&rel)),
    ];
    for p in candidates.into_iter().flatten() {
        if p.exists() {
            println!("[LabServer] YOLO-Script gefunden: {:?}", p);
            return Ok(p);
        }
    }
    Err("yolo_inference_server.py nicht gefunden".to_string())
}

fn get_canvas_server_path(app_handle: &tauri::AppHandle) -> Result<std::path::PathBuf, String> {
    let rel = std::path::Path::new("python").join("train_engine").join("plugins")
        .join("canvas").join("canvas_inference_server.py");
    let candidates = vec![
        app_handle.path().resource_dir().ok().map(|p| p.join(&rel)),
        Some(std::path::PathBuf::from("src-tauri").join(&rel)),
        Some(std::path::PathBuf::from(
            "/Users/karol/Desktop/Laufende_Projekte/FrameTrain/desktop-app/src-tauri"
        ).join(&rel)),
    ];
    for p in candidates.into_iter().flatten() {
        if p.exists() {
            println!("[LabServer] Canvas-Script gefunden: {:?}", p);
            return Ok(p);
        }
    }
    Err("canvas_inference_server.py nicht gefunden".to_string())
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

    let (version_path, model_id) = match get_version_info(&app_handle, &version_id) {
        Ok(v) => v,
        Err(e) => {
            let _ = app_handle.emit("lab-server-status",
                serde_json::json!({ "status": "error", "message": e }));
            let mut s = state.lock().unwrap();
            s.status = ServerStatus::Error;
            return Err(e);
        }
    };

    let fail = |msg: String| -> Result<(), String> {
        let _ = app_handle.emit("lab-server-status",
            serde_json::json!({ "status": "error", "message": msg.clone() }));
        if let Ok(mut s) = state.lock() { s.status = ServerStatus::Error; }
        Err(msg)
    };

    // ── Preflight + Server-Typ bestimmen (HuggingFace vs. Canvas) ─────────
    let vp = std::path::PathBuf::from(&version_path);
    let models_root = app_handle.path().app_data_dir()
        .map(|d| d.join("models"))
        .unwrap_or_default();
    let canvas_model_dir = models_root.join(&model_id);

    let is_canvas = model_id.starts_with("canvas_")
        || vp.join("graph_metadata.json").exists()
        || canvas_model_dir.join("graph_metadata.json").exists();

    // YOLO: eigener Server. Erkannt wird es am Checkpoint selbst (Ultralytics
    // schreibt seine Modulpfade in die .pt) oder an der Zuordnung, die der
    // Nutzer beim Import getroffen hat — Dateinamen wie best.pt sagen nichts.
    let is_yolo = !is_canvas && (
        crate::model_manager::read_plugin_override(&canvas_model_dir).as_deref() == Some("yolo")
            || crate::model_manager::dir_has_ultralytics_checkpoint(&vp)
            || crate::model_manager::dir_has_ultralytics_checkpoint(&canvas_model_dir)
    );

    let (model_path, is_canvas) = if is_yolo {
        // Die Version hat Vorrang: dort liegen die Gewichte des eigenen Laufs.
        let dir = if crate::model_manager::dir_has_ultralytics_checkpoint(&vp) {
            vp.clone()
        } else {
            canvas_model_dir.clone()
        };
        (dir.to_string_lossy().to_string(), false)
    } else if is_canvas {
        // Canvas braucht graph_metadata.json + model.pt im selben Ordner.
        // Versions-Pfad bevorzugen, sonst der Modell-Ordner (dorthin kopiert
        // das Training die Gewichte für list_canvas_models_with_pt).
        let dir = if vp.join("graph_metadata.json").exists() && vp.join("model.pt").exists() {
            vp.clone()
        } else {
            canvas_model_dir.clone()
        };
        if !dir.join("graph_metadata.json").exists() {
            return fail(format!(
                "Canvas-Modell: graph_metadata.json nicht gefunden in {} — \
                 Modell im Synapse Builder erneut speichern.", dir.display()
            ));
        }
        if !dir.join("model.pt").exists() {
            return fail(
                "Canvas-Modell ist noch nicht trainiert (kein model.pt). \
                 Trainiere es zuerst im Synapse Builder oder Training-Panel — \
                 danach kann es hier geladen werden.".to_string()
            );
        }
        (dir.to_string_lossy().to_string(), true)
    } else {
        if !vp.exists() {
            return fail(format!(
                "Versions-Pfad existiert nicht: {} — das Modell wurde evtl. verschoben oder gelöscht.",
                version_path
            ));
        }
        if !vp.join("config.json").exists() {
            let contents: Vec<String> = std::fs::read_dir(&vp).ok().into_iter().flatten().flatten()
                .filter_map(|e| e.file_name().to_str().map(|s| s.to_string()))
                .filter(|n| !n.starts_with('.'))
                .take(8)
                .collect();
            return fail(format!(
                "Keine config.json in {} — kein HuggingFace-Format. \
                 Die Lab-Inferenz benötigt ein HuggingFace-Modell \
                 (Text, Bild, Audio oder Seq2Seq). Vorhandene Dateien: {}",
                version_path,
                if contents.is_empty() { "(leer)".to_string() } else { contents.join(", ") }
            ));
        }
        (version_path.clone(), false)
    };

    let python        = get_python_path();
    let server_script = match if is_yolo {
        get_yolo_server_path(&app_handle)
    } else if is_canvas {
        get_canvas_server_path(&app_handle)
    } else {
        get_model_server_path(&app_handle)
    } {
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
    let canvas    = is_canvas;
    let yolo      = is_yolo;
    // Canvas- und YOLO-Server erwarten --model-dir, der HF-Server --model-path
    let path_arg  = if is_canvas || is_yolo { "--model-dir" } else { "--model-path" };

    std::thread::spawn(move || {
        println!("[LabServer] Starte Python: {} {} {} (canvas={}, yolo={})", python, path_arg, mp, canvas, yolo);

        let mut child = match Command::new(&python).no_window().python_utf8()
            .arg(server_script.to_string_lossy().to_string())
            .arg(path_arg).arg(&mp)
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
        let mut server_ready = false;
        let mut input_kind   = if canvas { "tensor".to_string() }
                               else if yolo { "image".to_string() }
                               else { "text".to_string() };
        let mut modality     = if canvas { "canvas".to_string() }
                               else if yolo { "detect".to_string() }
                               else { "text".to_string() };
        // Klassennamen des Modells. Sie kommen aus dem Checkpoint, nicht aus
        // einer Liste in FrameTrain — jedes Modell bringt seine eigenen mit.
        let mut classes: Vec<String> = Vec::new();

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
                            Some("ready") => {
                                if let Some(k) = msg.get("input_kind").and_then(|v| v.as_str()) {
                                    input_kind = k.to_string();
                                }
                                if let Some(m) = msg.get("modality").and_then(|v| v.as_str()) {
                                    modality = m.to_string();
                                }
                                if let Some(c) = msg.get("classes").and_then(|v| v.as_array()) {
                                    classes = c.iter()
                                        .filter_map(|v| v.as_str().map(str::to_string))
                                        .collect();
                                }
                                server_ready = true;
                                break;
                            }
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

        if server_ready {
            if let Ok(mut s) = state_arc.lock() {
                s.server = Some(LabServer {
                    child,
                    stdin: std::io::BufWriter::new(stdin),
                    receiver: rx,
                    version_id: vid.clone(),
                    model_path: mp,
                    is_canvas: canvas,
                    input_kind: input_kind.clone(),
                    modality: modality.clone(),
                });
                s.status = ServerStatus::Ready;
            }
            let _ = ah.emit("lab-server-status", serde_json::json!({
                "status": "ready",
                "version_id": vid,
                "input_kind": input_kind,
                "modality": modality,
                "classes": classes,
            }));
            println!("[LabServer] Bereit fuer Inferenz.");
        }
    });

    Ok(())
}

/// Fuehrt Inferenz auf einem einzelnen Sample durch (Text oder Datei).
/// Schnell (~50ms) weil das Modell bereits geladen ist.
#[tauri::command]
pub fn lab_infer_sample(
    text: String,
    file_path: Option<String>,
    state: tauri::State<'_, Arc<Mutex<LabState>>>,
) -> Result<InferResult, String> {
    let mut s = state.lock().map_err(|e| format!("Lock: {}", e))?;

    // Schreiben + Lesen atomar (Mutex haelt waehrend beider Operationen)
    let recv_result = {
        let server = s.server.as_mut()
            .ok_or_else(|| "Kein Modell geladen. Bitte warte bis das Modell fertig geladen ist.".to_string())?;

        // Datei-Sample (Bild/Audio): Pfad statt Text an den Server
        let file = file_path.as_deref().map(str::trim).filter(|p| !p.is_empty());

        let req = if let (Some(path), true) = (file, server.is_canvas) {
            // Canvas: Preprocessing per IR im Python
            serde_json::json!({ "input": path, "input_type": "image" }).to_string()
        } else if !server.is_canvas && matches!(server.input_kind.as_str(), "image" | "audio") {
            let kind = if server.input_kind == "image" { "Bild" } else { "Audio" };
            let path = file.ok_or_else(|| format!(
                "Dieses Modell erwartet eine {}-Datei. Lade im Labor {}-Samples aus einem Dataset.",
                kind, kind
            ))?;
            serde_json::json!({ "file_path": path }).to_string()
        } else if !server.is_canvas && file.is_some() {
            return Err(format!(
                "Dieses Modell erwartet {}, es wurde aber eine Datei ausgewählt.                  Passt das Dataset zum Modell?",
                if server.modality == "seq2seq" { "Text zum Umformulieren" } else { "Text" }
            ));
        } else if server.is_canvas {
            // Canvas-Modelle erwarten einen Zahlen-Tensor statt Text
            let nums: Vec<f64> = text
                .split(|c: char| c == ',' || c == ';' || c.is_whitespace())
                .filter(|s| !s.is_empty())
                .map(|s| s.parse::<f64>())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| "Canvas-Modell erwartet numerische Eingaben, z.B. \"0.5, 1.2, 3.0\" — freier Text wird nicht unterstützt.".to_string())?;
            if nums.is_empty() {
                return Err("Keine Zahlen in der Eingabe. Canvas-Modelle erwarten einen Feature-Vektor, z.B. \"0.5, 1.2, 3.0\".".to_string());
            }
            serde_json::json!({ "input": nums, "input_type": "tensor" }).to_string()
        } else {
            serde_json::json!({ "text": text }).to_string()
        };
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
                boxes: resp["boxes"].as_array().cloned(),
                image_width: resp["image_width"].as_u64().map(|v| v as u32),
                image_height: resp["image_height"].as_u64().map(|v| v as u32),
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
        "input_kind": s.server.as_ref().map(|srv| &srv.input_kind),
        "modality":   s.server.as_ref().map(|srv| &srv.modality),
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
        cmd.no_window();
        cmd.python_utf8();
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

// ══════════════════════════════════════════════════════════════════
// KORREKTUREN ALS DATASET EXPORTIEREN
//
// Das Labor sammelt, was das Modell falsch gemacht hat — und seit 1.2.85
// auch, wie es richtig gewesen waere. Ohne Export bliebe das in einer
// Session-Datei liegen. Hier werden daraus Dateien in dem Format, das die
// Train Engine ohnehin liest, und der Import registriert sie als Dataset.
//
// Das Ursprungs-Dataset wird dabei nie angefasst.
// ══════════════════════════════════════════════════════════════════

#[derive(Debug, Deserialize, Clone)]
pub struct CorrectionBox {
    pub label: String,
    pub x1: f64, pub y1: f64, pub x2: f64, pub y2: f64,
}

/// Die Felder kommen so aus dem Frontend. Tauri wandelt nur die Argumente
/// eines Befehls von camelCase um, nicht die Felder darin — ohne dieses
/// rename_all scheitert das Deserialisieren der Liste.
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CorrectionItem {
    /// "boxes" | "label" | "text"
    pub kind: String,
    pub input_text: String,
    pub file_path: Option<String>,
    pub boxes: Option<Vec<CorrectionBox>>,
    pub label: Option<String>,
    pub text: Option<String>,
    pub image_width: Option<f64>,
    pub image_height: Option<f64>,
}

/// Eine Korrektur-Box in eine normierte Box.
///
/// Das Beschneiden und Drehen liegt in `yolo_export` — dieselbe Rechnung
/// benutzt das Dataset Studio.
fn norm_box_of(class_id: usize, b: &CorrectionBox, width: f64, height: f64)
    -> Option<crate::yolo_export::NormBox>
{
    crate::yolo_export::normalize_box(class_id, b.x1, b.y1, b.x2, b.y2, width, height)
}

/// CSV-Feld nach RFC 4180 — Anfuehrungszeichen verdoppeln, Feld einpacken.
/// Korrigierte Texte enthalten Kommas und Zeilenumbrueche, sonst zerfaellt
/// die Datei beim Einlesen.
pub fn csv_field(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

/// Klassenliste in der Reihenfolge des ersten Auftretens.
///
/// Alphabetisch waere willkuerlicher: so entspricht die Reihenfolge dem, was
/// der Nutzer im Labor zuerst korrigiert hat, und bleibt bei erneutem Export
/// derselben Session gleich.
pub fn collect_class_names(items: &[CorrectionItem]) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for item in items {
        for b in item.boxes.iter().flatten() {
            if !names.contains(&b.label) { names.push(b.label.clone()); }
        }
    }
    names
}

#[tauri::command]
pub async fn lab_export_corrections(
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, crate::AppState>,
    model_id: String,
    dataset_name: String,
    items: Vec<CorrectionItem>,
) -> Result<serde_json::Value, String> {
    if items.is_empty() {
        return Err("Keine Korrekturen zum Exportieren.".to_string());
    }
    let name = dataset_name.trim();
    if name.is_empty() { return Err("Bitte einen Namen für das Dataset angeben.".to_string()); }

    let tmp_root = app_handle.path().app_data_dir()
        .map_err(|e| format!("AppDataDir: {}", e))?
        .join("tmp")
        .join(format!("lab_export_{}", &uuid::Uuid::new_v4().to_string().replace('-', "")[..12]));
    std::fs::create_dir_all(&tmp_root).map_err(|e| format!("Export-Ordner: {}", e))?;

    let has_boxes = items.iter().any(|i| i.kind == "boxes");
    let mut written = 0usize;
    let mut skipped: Vec<String> = Vec::new();

    if has_boxes {
        let classes = collect_class_names(&items);
        if classes.is_empty() {
            let _ = std::fs::remove_dir_all(&tmp_root);
            return Err("Die Korrekturen enthalten keine Boxen.".to_string());
        }
        let mut export_items: Vec<crate::yolo_export::ExportItem> = Vec::new();
        for (idx, item) in items.iter().enumerate() {
            let (Some(src), Some(w), Some(h)) = (item.file_path.as_deref(), item.image_width, item.image_height) else {
                skipped.push(item.input_text.clone());
                continue;
            };
            let src_path = std::path::Path::new(src);
            if !src_path.exists() { skipped.push(item.input_text.clone()); continue; }

            // Eindeutiger Name: zwei Datasets koennen "0001.jpg" heissen.
            let stem = src_path.file_stem().map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| format!("sample_{}", idx));

            export_items.push(crate::yolo_export::ExportItem {
                source: src_path.to_path_buf(),
                stem:   format!("{:04}_{}", idx + 1, stem),
                boxes:  item.boxes.iter().flatten()
                    .filter_map(|b| {
                        let class_id = classes.iter().position(|c| c == &b.label)?;
                        norm_box_of(class_id, b, w, h)
                    })
                    .collect(),
                // Korrekturen sind zu wenige, um sie sinnvoll aufzuteilen.
                split: None,
            });
        }

        // images/, labels/ und classes.txt schreibt der gemeinsame Exporter —
        // classes.txt wird beim Import gelesen und landet als names-Block in
        // der dataset.yaml, sonst stuenden dort Platzhalter.
        written = crate::yolo_export::write_yolo_layout(&tmp_root, &export_items, &classes)?.len();
    } else {
        let is_text = items.iter().any(|i| i.kind == "text");
        if is_text {
            // Seq2Seq liest source/target (siehe ft_data/seq2seq.py).
            let mut lines = Vec::new();
            for item in &items {
                let Some(target) = item.text.as_deref().or(item.label.as_deref()) else {
                    skipped.push(item.input_text.clone()); continue;
                };
                lines.push(serde_json::json!({ "source": item.input_text, "target": target }).to_string());
                written += 1;
            }
            std::fs::write(tmp_root.join("korrekturen.jsonl"), lines.join("\n") + "\n")
                .map_err(|e| format!("JSONL schreiben: {}", e))?;
        } else {
            // Klassifikation liest text/label (siehe seq_classification/plugin.py).
            let mut lines = vec!["text,label".to_string()];
            for item in &items {
                let Some(label) = item.label.as_deref().or(item.text.as_deref()) else {
                    skipped.push(item.input_text.clone()); continue;
                };
                lines.push(format!("{},{}", csv_field(&item.input_text), csv_field(label)));
                written += 1;
            }
            std::fs::write(tmp_root.join("korrekturen.csv"), lines.join("\n") + "\n")
                .map_err(|e| format!("CSV schreiben: {}", e))?;
        }
    }

    if written == 0 {
        let _ = std::fs::remove_dir_all(&tmp_root);
        return Err("Keine der Korrekturen liess sich exportieren (fehlende Dateien oder Werte).".to_string());
    }

    let info = crate::dataset_manager::import_local_dataset(
        app_handle.clone(), state,
        tmp_root.to_string_lossy().to_string(), name.to_string(), model_id,
    ).await;
    // Der Zwischenstand wird nicht gebraucht – der Import hat kopiert.
    let _ = std::fs::remove_dir_all(&tmp_root);
    let info = info?;

    Ok(serde_json::json!({
        "dataset": info,
        "written": written,
        "skipped": skipped.len(),
    }))
}

#[cfg(test)]
mod export_tests {
    use super::*;

    fn boxed(label: &str, x1: f64, y1: f64, x2: f64, y2: f64) -> CorrectionBox {
        CorrectionBox { label: label.to_string(), x1, y1, x2, y2 }
    }

    /// Weg einer Korrektur-Box bis zur fertigen Labelzeile.
    fn yolo_label_line(class_id: usize, b: &CorrectionBox, width: f64, height: f64) -> Option<String> {
        norm_box_of(class_id, b, width, height).map(|n| crate::yolo_export::label_line(&n))
    }

    #[test]
    fn box_wird_auf_null_bis_eins_normiert() {
        let line = yolo_label_line(0, &boxed("Tree", 100.0, 100.0, 300.0, 200.0), 400.0, 400.0).unwrap();
        assert_eq!(line, "0 0.500000 0.375000 0.500000 0.250000");
    }

    #[test]
    fn rueckwaerts_gezogene_box_wird_gedreht() {
        let a = yolo_label_line(1, &boxed("Sky", 300.0, 200.0, 100.0, 100.0), 400.0, 400.0).unwrap();
        let b = yolo_label_line(1, &boxed("Sky", 100.0, 100.0, 300.0, 200.0), 400.0, 400.0).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn ueber_den_rand_gezogene_box_wird_beschnitten() {
        // Beim Zeichnen rutscht die Maus schnell aus dem Bild.
        let line = yolo_label_line(0, &boxed("Tree", -50.0, -50.0, 200.0, 200.0), 400.0, 400.0).unwrap();
        assert_eq!(line, "0 0.250000 0.250000 0.500000 0.500000");
    }

    #[test]
    fn entartete_box_liefert_keine_zeile() {
        assert!(yolo_label_line(0, &boxed("x", 10.0, 10.0, 10.0, 60.0), 400.0, 400.0).is_none());
        assert!(yolo_label_line(0, &boxed("x", 10.0, 10.0, 60.0, 60.0), 0.0, 0.0).is_none());
    }

    #[test]
    fn csv_feld_haelt_komma_und_anfuehrungszeichen_aus() {
        assert_eq!(csv_field("a,b"), "\"a,b\"");
        assert_eq!(csv_field("er sagte \"hallo\""), "\"er sagte \"\"hallo\"\"\"");
        assert_eq!(csv_field("zeile1\nzeile2"), "\"zeile1\nzeile2\"");
    }

    /// Der Vertrag zwischen Export und Import: was hier geschrieben wird,
    /// muss der Dataset-Import als YOLO erkennen – sonst landet es als
    /// "unbekannt" in der Liste und ist fuers Training gesperrt.
    #[test]
    fn exportierter_ordner_wird_als_yolo_dataset_erkannt() {
        use std::fs;
        let base = std::env::temp_dir().join(format!("ft_labexport_{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(base.join("images")).unwrap();
        fs::create_dir_all(base.join("labels")).unwrap();
        fs::write(base.join("images/0001_a.jpg"), vec![0u8; 16]).unwrap();
        let line = yolo_label_line(0, &boxed("Tree", 10.0, 10.0, 60.0, 60.0), 100.0, 100.0).unwrap();
        fs::write(base.join("labels/0001_a.txt"), line + "\n").unwrap();
        fs::write(base.join("classes.txt"), "Tree\nSky\n").unwrap();

        let analysis = crate::dataset_manager::detect_dataset_type(&base);
        assert_eq!(analysis.detected_type.as_str(), "yolo_bbox",
                   "Export-Layout muss als YOLO durchgehen");

        // classes.txt traegt die echten Namen in die generierte dataset.yaml.
        crate::dataset_manager::generate_dataset_yaml(&base, "images", "labels", false).unwrap();
        let yaml = fs::read_to_string(base.join("dataset.yaml")).unwrap();
        assert!(yaml.contains("Tree"), "Klassennamen fehlen: {}", yaml);
        assert!(!yaml.contains("KlasseA"), "Platzhalter statt echter Namen: {}", yaml);

        let _ = fs::remove_dir_all(&base);
    }

    /// Das Frontend schickt camelCase. Ohne serde(rename_all) waere die
    /// Liste beim Export leer angekommen – und der Fehler erst zur Laufzeit
    /// sichtbar geworden.
    #[test]
    fn frontend_json_wird_gelesen() {
        let json = r#"[{
            "kind": "boxes", "inputText": "ski_0001.jpg",
            "filePath": "/tmp/ski_0001.jpg",
            "boxes": [{"label": "Tree", "x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0}],
            "label": null, "text": null,
            "imageWidth": 512.0, "imageHeight": 512.0
        }]"#;
        let items: Vec<CorrectionItem> = serde_json::from_str(json).expect("camelCase muss passen");
        assert_eq!(items[0].file_path.as_deref(), Some("/tmp/ski_0001.jpg"));
        assert_eq!(items[0].image_width, Some(512.0));
        assert_eq!(collect_class_names(&items), vec!["Tree"]);
    }

    #[test]
    fn klassenliste_folgt_dem_ersten_auftreten() {
        let item = |labels: &[&str]| CorrectionItem {
            kind: "boxes".into(), input_text: String::new(), file_path: None,
            boxes: Some(labels.iter().map(|l| boxed(l, 0.0, 0.0, 1.0, 1.0)).collect()),
            label: None, text: None, image_width: None, image_height: None,
        };
        let items = vec![item(&["Sky", "Tree"]), item(&["Tree", "Person"])];
        assert_eq!(collect_class_names(&items), vec!["Sky", "Tree", "Person"]);
    }
}
