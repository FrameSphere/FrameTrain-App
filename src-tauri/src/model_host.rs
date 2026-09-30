// model_host.rs – Gemeinsamer Kern fuer persistente Modell-Server
//
// Labor und Hosting starten dieselben Python-Server (HuggingFace, YOLO,
// Canvas) und sprechen dasselbe Protokoll: eine JSON-Zeile rein, eine
// JSON-Zeile raus. Beim Streaming (nur LLM) kommen vorher Zeilen
// {"type": "token", "text": …}. Hier liegt alles, was beide brauchen:
// Modell-Ordner finden, Server starten, Anfrage bauen, Antwort lesen.

use crate::command_ext::{NoWindow, PythonUtf8};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use tauri::Manager;

// ============ Typen ============

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
    /// Die ganze Antwort des Modell-Servers. Jede Aufgabe meldet eigene
    /// Felder (Entitaeten, Aehnlichkeit, Bildpfad, YOLO-Task, Frage …); ohne
    /// sie konnte das Labor nur Klasse und Konfidenz zeigen.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra:            Option<serde_json::Value>,
}

/// Wo die Gewichte liegen und welcher Server sie laedt.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub model_path: String,
    pub is_canvas:  bool,
    pub is_yolo:    bool,
}

/// Ein laufender, fertig geladener Modell-Server.
pub struct ServerProc {
    pub child:      Child,
    pub stdin:      std::io::BufWriter<std::process::ChildStdin>,
    pub receiver:   std::sync::mpsc::Receiver<String>,
    pub model_path: String,
    /// Canvas-Modell (DynamicGraphModule) statt HuggingFace — anderes Request-Format
    pub is_canvas:  bool,
    /// Was der Server erwartet: "text" | "image" | "audio" | "video" | "tensor"
    pub input_kind: String,
    /// Aufgabenbereich: "text" | "image" | "audio" | "seq2seq" | "causal_lm" | "canvas" | "detect" …
    pub modality:   String,
    /// Klassennamen des Modells. Sie kommen aus dem Checkpoint, nicht aus
    /// einer Liste in FrameTrain — jedes Modell bringt seine eigenen mit.
    pub classes:    Vec<String>,
    /// YOLO-Aufgabe (detect/segment/pose/obb/classify).
    pub task:       Option<String>,
}

impl Drop for ServerProc {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Eingabe fuer eine Inferenz — Labor und Hosting fuellen, was sie haben.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InferInput {
    #[serde(default)]
    pub text:      String,
    pub file_path: Option<String>,
    /// Frage zum Bild (VLM). Andere Modelle ignorieren das Feld.
    pub question:  Option<String>,
    /// Clip-Grenzen in Sekunden (Video). Ohne sie gilt das ganze Video.
    pub start:     Option<f64>,
    pub end:       Option<f64>,
    /// Chat-Verlauf fuer LLMs: [{role, content}]. Ohne ihn gilt `text` als einzige Nachricht.
    pub messages:  Option<Vec<serde_json::Value>>,
    /// Token einzeln melden (nur LLM).
    #[serde(default)]
    pub stream:    bool,
    /// Zusaetzliche Generierungs-Parameter (max_new_tokens, temperature, seed …).
    pub params:    Option<serde_json::Value>,
}

#[derive(Debug)]
pub enum RequestError {
    Timeout(u64),
    Crashed,
    Server(String),
    Io(String),
}

impl RequestError {
    pub fn message(&self) -> String {
        match self {
            RequestError::Timeout(s) => format!("Inferenz-Timeout ({}s) – Modell antwortet nicht. Bitte neu laden.", s),
            RequestError::Crashed => "Modell-Server ist abgestuerzt. Bitte Modell neu laden.".to_string(),
            RequestError::Server(m) | RequestError::Io(m) => m.clone(),
        }
    }
}

// ============ Pfade ============

fn find_script(app_handle: &tauri::AppHandle, rel: &Path, what: &str) -> Result<PathBuf, String> {
    let candidates = vec![
        app_handle.path().resource_dir().ok().map(|p| p.join(rel)),
        Some(PathBuf::from("src-tauri").join(rel)),
        Some(PathBuf::from(
            "/Users/karol/Desktop/Laufende_Projekte/FrameTrain/desktop-app/src-tauri"
        ).join(rel)),
    ];
    for p in candidates.into_iter().flatten() {
        if p.exists() {
            return Ok(p);
        }
    }
    Err(format!("{} nicht gefunden", what))
}

pub(crate) fn get_model_server_path(app_handle: &tauri::AppHandle) -> Result<PathBuf, String> {
    let rel = Path::new("python").join("test_engine").join("model_server.py");
    find_script(app_handle, &rel, "model_server.py")
}

/// Pfad zum YOLO-Server – dritter Servertyp neben HuggingFace und Canvas.
pub(crate) fn get_yolo_server_path(app_handle: &tauri::AppHandle) -> Result<PathBuf, String> {
    let rel = Path::new("python").join("train_engine").join("plugins")
        .join("yolo").join("yolo_inference_server.py");
    find_script(app_handle, &rel, "yolo_inference_server.py")
}

/// Script für Canvas-Modelle (gleiches stdin/stdout-Protokoll wie model_server.py)
pub(crate) fn get_canvas_server_path(app_handle: &tauri::AppHandle) -> Result<PathBuf, String> {
    let rel = Path::new("python").join("train_engine").join("plugins")
        .join("canvas").join("canvas_inference_server.py");
    find_script(app_handle, &rel, "canvas_inference_server.py")
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

/// Kann der Modell-Server diesen Ordner laden? HuggingFace-Modelle tragen
/// config.json; Diffusers-Pipelines stattdessen model_index.json, der
/// LoRA-Export des Text-zu-Bild-Trainings text_to_image_lora.json (wie
/// is_diffusion_model in model_server.py). Nur config.json zu verlangen
/// sperrte jedes Stable-Diffusion-Modell aus dem Labor aus (Live-Test 1.4.2).
pub(crate) fn has_lab_model_marker(dir: &Path) -> bool {
    ["config.json", "model_index.json", "text_to_image_lora.json"].iter().any(|f| dir.join(f).is_file())
}

/// Wartezeit auf eine Antwort des Modell-Servers je Modalitaet.
pub(crate) fn infer_timeout_secs(modality: &str) -> u64 {
    match modality {
        "text_to_image" => 600,
        "vlm" => 120,
        // Ein LLM schreibt Token fuer Token — 256 Tokens eines 7B-Modells
        // brauchen auf dem Mac leicht ueber 30 s.
        "causal_lm" => 180,
        // Lange Aufnahmen laufen in 30-s-Stuecken durch, Videos werden erst dekodiert.
        "asr" | "video" => 120,
        _ => 30,
    }
}

// ============ Modell finden ============

/// Findet Ordner und Servertyp einer Version (HuggingFace, YOLO oder Canvas).
pub fn resolve_model(app_handle: &tauri::AppHandle, version_id: &str) -> Result<ResolvedModel, String> {
    let (version_path, model_id) = get_version_info(app_handle, version_id)?;

    let vp = PathBuf::from(&version_path);
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

    if is_yolo {
        // Die Version hat Vorrang: dort liegen die Gewichte des eigenen Laufs.
        let dir = if crate::model_manager::dir_has_ultralytics_checkpoint(&vp) {
            vp.clone()
        } else {
            canvas_model_dir.clone()
        };
        return Ok(ResolvedModel { model_path: dir.to_string_lossy().to_string(), is_canvas: false, is_yolo: true });
    }

    if is_canvas {
        // Canvas braucht graph_metadata.json + model.pt im selben Ordner.
        // Versions-Pfad bevorzugen, sonst der Modell-Ordner (dorthin kopiert
        // das Training die Gewichte für list_canvas_models_with_pt).
        let dir = if vp.join("graph_metadata.json").exists() && vp.join("model.pt").exists() {
            vp.clone()
        } else {
            canvas_model_dir.clone()
        };
        if !dir.join("graph_metadata.json").exists() {
            return Err(format!(
                "Canvas-Modell: graph_metadata.json nicht gefunden in {} — \
                 Modell im Synapse Builder erneut speichern.", dir.display()
            ));
        }
        if !dir.join("model.pt").exists() {
            return Err(
                "Canvas-Modell ist noch nicht trainiert (kein model.pt). \
                 Trainiere es zuerst im Synapse Builder oder Training-Panel — \
                 danach kann es hier geladen werden.".to_string()
            );
        }
        return Ok(ResolvedModel { model_path: dir.to_string_lossy().to_string(), is_canvas: true, is_yolo: false });
    }

    if !vp.exists() {
        return Err(format!(
            "Versions-Pfad existiert nicht: {} — das Modell wurde evtl. verschoben oder gelöscht.",
            version_path
        ));
    }
    if !has_lab_model_marker(&vp) {
        let contents: Vec<String> = std::fs::read_dir(&vp).ok().into_iter().flatten().flatten()
            .filter_map(|e| e.file_name().to_str().map(|s| s.to_string()))
            .filter(|n| !n.starts_with('.'))
            .take(8)
            .collect();
        return Err(format!(
            "Keine config.json in {} — kein HuggingFace-Format. \
             Die Lab-Inferenz benötigt ein HuggingFace-Modell \
             (Text, Bild, Audio oder Seq2Seq). Vorhandene Dateien: {}",
            version_path,
            if contents.is_empty() { "(leer)".to_string() } else { contents.join(", ") }
        ));
    }
    Ok(ResolvedModel { model_path: version_path, is_canvas: false, is_yolo: false })
}

// ============ Server starten ============

/// Startet den passenden Python-Server und wartet (blockierend, max. 120 s),
/// bis er "ready" meldet. `tag` steht vor jeder Log-Zeile.
pub fn spawn_server(app_handle: &tauri::AppHandle, model: &ResolvedModel, tag: &str) -> Result<ServerProc, String> {
    let script = if model.is_yolo {
        get_yolo_server_path(app_handle)?
    } else if model.is_canvas {
        get_canvas_server_path(app_handle)?
    } else {
        get_model_server_path(app_handle)?
    };
    let python = crate::python_env::resolve_python();
    // Canvas- und YOLO-Server erwarten --model-dir, der HF-Server --model-path
    let path_arg = if model.is_canvas || model.is_yolo { "--model-dir" } else { "--model-path" };
    println!("[{}] Starte Python: {} {} {} (canvas={}, yolo={})",
        tag, python, path_arg, model.model_path, model.is_canvas, model.is_yolo);

    let mut child = Command::new(&python).no_window().python_utf8()
        .arg(script.to_string_lossy().to_string())
        .arg(path_arg).arg(&model.model_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Python konnte nicht gestartet werden: {}", e))?;

    // Stderr in separatem Thread loggen
    if let Some(stderr) = child.stderr.take() {
        let tag = tag.to_string();
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().flatten() {
                eprintln!("[{} STDERR] {}", tag, line);
            }
        });
    }

    let stdin = child.stdin.take();
    let stdout = child.stdout.take();
    let (stdin, stdout) = match (stdin, stdout) {
        (Some(i), Some(o)) => (i, o),
        _ => {
            let _ = child.kill();
            return Err("Kein stdin/stdout zum Modell-Server".to_string());
        }
    };

    // stdout-Lese-Thread -> Channel
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().flatten() {
            if tx.send(line).is_err() { break; }
        }
    });

    let mut proc = ServerProc {
        child,
        stdin: std::io::BufWriter::new(stdin),
        receiver: rx,
        model_path: model.model_path.clone(),
        is_canvas: model.is_canvas,
        input_kind: if model.is_canvas { "tensor".into() } else if model.is_yolo { "image".into() } else { "text".into() },
        modality: if model.is_canvas { "canvas".into() } else if model.is_yolo { "detect".into() } else { "text".into() },
        classes: Vec::new(),
        task: None,
    };

    // Auf "ready" warten (max. 120 Sekunden – grosse Modelle auf CPU brauchen Zeit)
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err("Timeout beim Laden des Modells (120s). Versuche es erneut.".to_string());
        }
        match proc.receiver.recv_timeout(remaining) {
            Ok(line) => {
                let line = line.trim().to_string();
                println!("[{}] Startup-Zeile: {}", tag, line);
                let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) else { continue };
                match msg.get("type").and_then(|t| t.as_str()) {
                    Some("ready") => {
                        apply_ready(&mut proc, &msg);
                        return Ok(proc);
                    }
                    Some("error") => {
                        return Err(msg.get("message").and_then(|m| m.as_str())
                            .unwrap_or("Unbekannter Fehler").to_string());
                    }
                    _ => { /* Ignoriere andere Nachrichten waehrend Startup */ }
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                return Err("Timeout beim Modell-Laden".to_string());
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                return Err("Server-Prozess unerwartet beendet".to_string());
            }
        }
    }
}

fn apply_ready(proc: &mut ServerProc, msg: &serde_json::Value) {
    if let Some(k) = msg.get("input_kind").and_then(|v| v.as_str()) {
        proc.input_kind = k.to_string();
    }
    if let Some(m) = msg.get("modality").and_then(|v| v.as_str()) {
        proc.modality = m.to_string();
    }
    proc.task = msg.get("task_type").and_then(|v| v.as_str()).map(str::to_string);
    if let Some(c) = msg.get("classes").and_then(|v| v.as_array()) {
        proc.classes = c.iter().filter_map(|v| v.as_str().map(str::to_string)).collect();
    }
}

// ============ Anfrage / Antwort ============

/// Baut die JSON-Anfrage passend zum Server: Datei, Text, Chat oder Zahlen.
pub fn build_request(is_canvas: bool, input_kind: &str, modality: &str, input: &InferInput) -> Result<serde_json::Value, String> {
    // Datei-Sample (Bild/Audio): Pfad statt Text an den Server
    let file = input.file_path.as_deref().map(str::trim).filter(|p| !p.is_empty());
    let text = input.text.as_str();

    let mut req = if let (Some(path), true) = (file, is_canvas) {
        // Canvas: Preprocessing per IR im Python
        serde_json::json!({ "input": path, "input_type": "image" })
    } else if !is_canvas && input_kind == "video" {
        let path = file.ok_or_else(||
            "Dieses Modell erwartet eine Video-Datei.".to_string())?;
        serde_json::json!({ "file_path": path, "start": input.start, "end": input.end })
    } else if !is_canvas && matches!(input_kind, "image" | "audio") {
        let kind = if input_kind == "image" { "Bild" } else { "Audio" };
        let path = file.ok_or_else(|| format!("Dieses Modell erwartet eine {}-Datei.", kind))?;
        let q = input.question.as_deref().map(str::trim).filter(|q| !q.is_empty());
        match q {
            Some(q) if modality == "vlm" => serde_json::json!({ "file_path": path, "question": q }),
            _ => serde_json::json!({ "file_path": path }),
        }
    } else if !is_canvas && file.is_some() {
        return Err(format!(
            "Dieses Modell erwartet {}, es wurde aber eine Datei ausgewählt. Passt das Dataset zum Modell?",
            if modality == "seq2seq" { "Text zum Umformulieren" } else { "Text" }
        ));
    } else if is_canvas {
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
        serde_json::json!({ "input": nums, "input_type": "tensor" })
    } else {
        serde_json::json!({ "text": text })
    };

    // Chat-Verlauf und Streaming versteht nur der LLM-Zweig des HF-Servers;
    // andere Server bekommen die Felder gar nicht erst.
    if modality == "causal_lm" {
        if let Some(msgs) = input.messages.as_ref().filter(|m| !m.is_empty()) {
            req["messages"] = serde_json::Value::Array(msgs.clone());
        }
        if input.stream {
            req["stream"] = serde_json::Value::Bool(true);
        }
    }
    if let (Some(params), Some(obj)) = (input.params.as_ref().and_then(|p| p.as_object()), req.as_object_mut()) {
        for (k, v) in params {
            // Eingabefelder duerfen die Parameter nicht ueberschreiben.
            if !obj.contains_key(k) {
                obj.insert(k.clone(), v.clone());
            }
        }
    }
    Ok(req)
}

/// Wandelt die Server-Antwort in das Ergebnis fuer die Oberflaeche.
pub fn parse_result(resp: serde_json::Value) -> Result<InferResult, String> {
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
        extra: Some(resp),
    })
}

impl ServerProc {
    /// Schickt eine Anfrage und liest bis zur Ergebniszeile. Token-Zeilen
    /// (Streaming) gehen an `on_token`; das Zeitlimit gilt je Zeile, damit ein
    /// lange schreibendes LLM nicht abgebrochen wird, solange es schreibt.
    pub fn request(
        &mut self,
        req: &serde_json::Value,
        timeout_secs: u64,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<serde_json::Value, RequestError> {
        // Reste einer abgebrochenen Anfrage verwerfen, sonst bekaeme diese
        // Anfrage die Antwort der vorigen.
        while self.receiver.try_recv().is_ok() {}
        writeln!(self.stdin, "{}", req).map_err(|e| RequestError::Io(format!("Schreibfehler: {}", e)))?;
        self.stdin.flush().map_err(|e| RequestError::Io(format!("Flush-Fehler: {}", e)))?;
        loop {
            match self.receiver.recv_timeout(Duration::from_secs(timeout_secs)) {
                Ok(line) => {
                    let resp: serde_json::Value = match serde_json::from_str(line.trim()) {
                        Ok(v) => v,
                        // Fremde Ausgaben (print einer Bibliothek) ueberspringen.
                        Err(_) => continue,
                    };
                    if resp.get("type").and_then(|t| t.as_str()) == Some("token") {
                        if let Some(t) = resp.get("text").and_then(|t| t.as_str()) {
                            on_token(t);
                        }
                        continue;
                    }
                    if resp.get("type").and_then(|t| t.as_str()) == Some("error") {
                        return Err(RequestError::Server(resp.get("message").and_then(|m| m.as_str())
                            .unwrap_or("Unbekannter Inferenz-Fehler").to_string()));
                    }
                    return Ok(resp);
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => return Err(RequestError::Timeout(timeout_secs)),
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return Err(RequestError::Crashed),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(text: &str) -> InferInput { InferInput { text: text.into(), ..Default::default() } }

    #[test]
    fn text_anfrage_fuer_klassifikation() {
        let r = build_request(false, "text", "text", &input("hallo")).unwrap();
        assert_eq!(r, serde_json::json!({ "text": "hallo" }));
    }

    #[test]
    fn llm_bekommt_verlauf_und_streaming() {
        let mut i = input("zweite Frage");
        i.stream = true;
        i.messages = Some(vec![serde_json::json!({ "role": "user", "content": "erste" })]);
        let r = build_request(false, "text", "causal_lm", &i).unwrap();
        assert_eq!(r["stream"], true);
        assert_eq!(r["messages"][0]["content"], "erste");
        assert_eq!(r["text"], "zweite Frage");
    }

    #[test]
    fn nicht_llm_bekommt_kein_streaming() {
        let mut i = input("x");
        i.stream = true;
        i.messages = Some(vec![serde_json::json!({ "role": "user", "content": "x" })]);
        let r = build_request(false, "text", "seq2seq", &i).unwrap();
        assert!(r.get("stream").is_none());
        assert!(r.get("messages").is_none());
    }

    #[test]
    fn bildmodell_verlangt_datei() {
        assert!(build_request(false, "image", "image", &input("x")).is_err());
        let mut i = input("");
        i.file_path = Some("/tmp/a.png".into());
        let r = build_request(false, "image", "image", &i).unwrap();
        assert_eq!(r["file_path"], "/tmp/a.png");
    }

    #[test]
    fn vlm_frage_wird_mitgeschickt() {
        let mut i = input("");
        i.file_path = Some("/tmp/a.png".into());
        i.question = Some("Was ist das?".into());
        let r = build_request(false, "image", "vlm", &i).unwrap();
        assert_eq!(r["question"], "Was ist das?");
    }

    #[test]
    fn parameter_ueberschreiben_keine_eingabe() {
        let mut i = input("prompt");
        i.params = Some(serde_json::json!({ "text": "boese", "seed": 7 }));
        let r = build_request(false, "text", "text_to_image", &i).unwrap();
        assert_eq!(r["text"], "prompt");
        assert_eq!(r["seed"], 7);
    }

    #[test]
    fn canvas_liest_zahlen() {
        let r = build_request(true, "tensor", "canvas", &input("0.5, 1; 2")).unwrap();
        assert_eq!(r["input"], serde_json::json!([0.5, 1.0, 2.0]));
        assert!(build_request(true, "tensor", "canvas", &input("abc")).is_err());
    }

    #[test]
    fn fehlerantwort_wird_zum_fehler() {
        let e = parse_result(serde_json::json!({ "type": "error", "message": "kaputt" })).unwrap_err();
        assert_eq!(e, "kaputt");
        let ok = parse_result(serde_json::json!({ "predicted": "a", "inference_time": 0.5 })).unwrap();
        assert_eq!(ok.inference_ms, 500.0);
    }

    #[test]
    fn diffusions_pipelines_sind_lab_modelle() {
        let dir = std::env::temp_dir().join(format!("ft_marker_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!has_lab_model_marker(&dir));
        std::fs::write(dir.join("model_index.json"), "{}").unwrap();
        assert!(has_lab_model_marker(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
