// synapse_service.rs – Formen-Dienst des Synapse Builders
//
// Ein dauerhafter Python-Prozess (python/synapse/service.py) prueft den Graphen
// nach jeder Aenderung auf Meta-Tensoren. Er laeuft dauerhaft, weil das Laden
// von torch rund eine Sekunde dauert, die Pruefung danach nur Millisekunden.

use crate::command_ext::{NoWindow, PythonUtf8};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError};
use std::sync::Mutex;
use std::time::Duration;

const START_TIMEOUT: Duration = Duration::from_secs(60);
const CHECK_TIMEOUT: Duration = Duration::from_secs(120);

struct Service {
    child: Child,
    stdin: ChildStdin,
    receiver: Receiver<String>,
    next_id: u64,
}

static SERVICE: Mutex<Option<Service>> = Mutex::new(None);

fn start(app_handle: &tauri::AppHandle) -> Result<Service, String> {
    let rel = Path::new("python").join("synapse").join("service.py");
    let script = crate::model_host::find_script(app_handle, &rel, "synapse/service.py")?;
    let python = crate::python_env::resolve_python();
    let mut child = Command::new(&python).no_window().python_utf8()
        .arg(script.to_string_lossy().to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("Python konnte nicht gestartet werden: {}", e))?;
    let stdin = child.stdin.take().ok_or("Kein stdin")?;
    let stdout = child.stdout.take().ok_or("Kein stdout")?;
    let (tx, receiver) = channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().flatten() {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let mut service = Service { child, stdin, receiver, next_id: 0 };
    match service.receiver.recv_timeout(START_TIMEOUT) {
        Ok(line) => {
            let msg: serde_json::Value = serde_json::from_str(&line).map_err(|e| e.to_string())?;
            if msg["type"] == "ready" {
                Ok(service)
            } else {
                let _ = service.child.kill();
                Err(msg["message"].as_str().unwrap_or("Dienst startet nicht").to_string())
            }
        }
        Err(_) => {
            let _ = service.child.kill();
            Err("Formen-Dienst hat nicht geantwortet".to_string())
        }
    }
}

/// Eine Anfrage senden und die Zeile mit derselben id abwarten.
fn exchange(service: &mut Service, mut request: serde_json::Value) -> Result<serde_json::Value, String> {
    service.next_id += 1;
    let id = service.next_id;
    request["id"] = serde_json::json!(id);
    writeln!(service.stdin, "{}", request).map_err(|e| e.to_string())?;
    service.stdin.flush().map_err(|e| e.to_string())?;
    loop {
        let line = service.receiver.recv_timeout(CHECK_TIMEOUT).map_err(|e| match e {
            RecvTimeoutError::Timeout => "Formen-Pruefung dauert zu lange".to_string(),
            RecvTimeoutError::Disconnected => "Formen-Dienst wurde beendet".to_string(),
        })?;
        let msg: serde_json::Value = serde_json::from_str(&line).map_err(|e| e.to_string())?;
        if msg["id"] != serde_json::json!(id) {
            continue;
        }
        if let Some(err) = msg["error"].as_str() {
            return Err(err.to_string());
        }
        return Ok(msg["result"].clone());
    }
}

fn request(app_handle: &tauri::AppHandle, payload: serde_json::Value) -> Result<serde_json::Value, String> {
    let mut slot = SERVICE.lock().map_err(|e| e.to_string())?;
    // Ein abgestuerzter Dienst wird einmal neu gestartet.
    for attempt in 0..2 {
        if slot.is_none() {
            *slot = Some(start(app_handle)?);
        }
        let service = slot.as_mut().ok_or("Dienst fehlt")?;
        match exchange(service, payload.clone()) {
            Ok(result) => return Ok(result),
            Err(e) if e.contains("Dienst wurde beendet") && attempt == 0 => *slot = None,
            Err(e) => {
                if let Ok(Some(_)) = service.child.try_wait() {
                    *slot = None;
                }
                return Err(e);
            }
        }
    }
    Err("Formen-Dienst startet nicht".to_string())
}

/// Prueft einen Graphen. Lang laufend beim ersten Aufruf, daher async.
#[tauri::command]
pub async fn synapse_check(
    app_handle: tauri::AppHandle,
    graph: serde_json::Value,
) -> Result<serde_json::Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        request(&app_handle, serde_json::json!({ "cmd": "check", "graph": graph }))
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Liest Ports und Einstellungen aus dem Code eines eigenen Nodes und macht einen Probelauf.
#[tauri::command]
pub async fn synapse_describe_code(
    app_handle: tauri::AppHandle,
    code: String,
    values: serde_json::Value,
    shape: Vec<i64>,
) -> Result<serde_json::Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        request(&app_handle, serde_json::json!({
            "cmd": "describe_code", "code": code, "values": values, "shape": shape,
        }))
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Ein Lernschritt mit Zufallsdaten: Ausgabe, Gradient und Update je Node.
#[tauri::command]
pub async fn synapse_step(
    app_handle: tauri::AppHandle,
    graph: serde_json::Value,
    lr: f64,
) -> Result<serde_json::Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        request(&app_handle, serde_json::json!({ "cmd": "step", "graph": graph, "lr": lr }))
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Exportiert ein trainiertes Modell (pytorch, ftmodel, torchscript, onnx, safetensors).
#[tauri::command]
pub async fn synapse_export(
    app_handle: tauri::AppHandle,
    model_id: String,
    format: String,
    target: String,
) -> Result<serde_json::Value, String> {
    let models = crate::model_manager::models_dir_for_export(&app_handle)?;
    let model_dir = models.join(&model_id);
    if !model_dir.is_dir() {
        return Err(format!("Modell-Ordner nicht gefunden: {}", model_id));
    }
    tauri::async_runtime::spawn_blocking(move || {
        request(&app_handle, serde_json::json!({
            "cmd": "export", "format": format, "target": target,
            "model_dir": model_dir.to_string_lossy(),
        }))
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Vorhersage mit einem trainierten Modell: tensor, text, file oder generate.
#[tauri::command]
pub async fn synapse_infer(
    app_handle: tauri::AppHandle,
    model_id: String,
    kind: String,
    value: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let model_dir = crate::model_manager::models_dir_for_export(&app_handle)?.join(&model_id);
    if !model_dir.join("model.pt").is_file() {
        return Err("Dieses Modell hat noch keine Gewichte. Trainiere es zuerst.".to_string());
    }
    tauri::async_runtime::spawn_blocking(move || {
        request(&app_handle, serde_json::json!({
            "cmd": "infer", "kind": kind, "value": value,
            "model_dir": model_dir.to_string_lossy(),
        }))
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Beendet den Dienst, wenn der Builder geschlossen wird.
#[tauri::command]
pub async fn synapse_stop() -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(|| {
        if let Ok(mut slot) = SERVICE.lock() {
            if let Some(mut service) = slot.take() {
                let _ = writeln!(service.stdin, "{{\"cmd\":\"shutdown\"}}");
                let _ = service.child.wait();
            }
        }
    })
    .await
    .map_err(|e| e.to_string())
}
