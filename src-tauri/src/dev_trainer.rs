// dev_trainer.rs – Führt ein user-geschriebenes Python-Script aus
// und emitiert die gleichen Events wie der normale Trainer.
// Enthält außerdem den Dev-Test-Modus (start_dev_test / stop_dev_test).

use std::fs;
use crate::command_ext::{NoWindow, PythonUtf8};
use std::process::{Command, Stdio};
use std::io::{BufRead, BufReader};
use std::thread;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use tauri::{Emitter, Manager};
use chrono::Utc;
use uuid::Uuid;
use std::sync::Mutex as StdMutex;

use crate::training_manager::{TrainingJob, TrainingStatus, TrainingProgress, TrainingConfig};

#[derive(Debug, Serialize, Deserialize)]
pub struct DevTrainingRefs {
    #[serde(flatten)]
    pub vars: HashMap<String, String>,
}

// ── Prozess-Registry für Dev-Train / Dev-Test ─────────────────────────────
// Ermöglicht sauberes Stoppen der Dev-Prozesse (vorher gab es dafür keinen Weg).

struct DevProcEntry {
    pid: Option<u32>,
    stop_requested: bool,
}

static DEV_TRAIN_PROC: StdMutex<DevProcEntry> = StdMutex::new(DevProcEntry { pid: None, stop_requested: false });
static DEV_TEST_PROC:  StdMutex<DevProcEntry> = StdMutex::new(DevProcEntry { pid: None, stop_requested: false });

fn kill_process_tree(pid: u32) {
    #[cfg(unix)]
    {
        let _ = Command::new("kill").no_window().args(["-TERM", &pid.to_string()]).output();
        thread::sleep(std::time::Duration::from_millis(300));
        let _ = Command::new("pkill").no_window().args(["-KILL", "-P", &pid.to_string()]).output();
        let _ = Command::new("kill").no_window().args(["-KILL", &pid.to_string()]).output();
    }
    #[cfg(windows)]
    {
        let _ = Command::new("taskkill").no_window().args(["/F", "/PID", &pid.to_string(), "/T"]).output();
    }
}

fn registry_start(reg: &StdMutex<DevProcEntry>) -> Result<(), String> {
    let mut e = reg.lock().map_err(|e| format!("Lock: {}", e))?;
    if e.pid.is_some() {
        return Err("Es läuft bereits ein Dev-Prozess. Bitte zuerst stoppen.".to_string());
    }
    e.stop_requested = false;
    Ok(())
}

fn registry_set_pid(reg: &StdMutex<DevProcEntry>, pid: u32) {
    if let Ok(mut e) = reg.lock() { e.pid = Some(pid); }
}

fn registry_clear(reg: &StdMutex<DevProcEntry>) -> bool {
    // Gibt zurück ob ein Stop angefordert wurde (Events dann unterdrücken).
    if let Ok(mut e) = reg.lock() {
        e.pid = None;
        let stopped = e.stop_requested;
        e.stop_requested = false;
        stopped
    } else { false }
}

fn registry_stop(reg: &StdMutex<DevProcEntry>) {
    let pid = {
        if let Ok(mut e) = reg.lock() {
            e.stop_requested = true;
            e.pid
        } else { None }
    };
    if let Some(pid) = pid { kill_process_tree(pid); }
}

// Dieselbe Python-Auswahl wie das normale Training.
//
// Vorher nahm der Dev-Trainer schlicht das erste `python3` auf dem PATH — ohne
// zu pruefen, ob dort ueberhaupt torch installiert ist. Ein Script, das im
// normalen Training laeuft, starb im Dev-Train mit ModuleNotFoundError, weil
// beide Wege unterschiedliche Interpreter benutzten.
fn get_python_path() -> String {
    crate::training_manager::resolve_python_path()
}

/// Startet ein user-geschriebenes Python-Script.
/// Das Script bekommt die übergebenen env-Variablen + OUTPUT_PATH.
/// Stdout-Output wird geparst (JSON-Events wie train_engine) oder als Rohtextzeile emitiert.
#[tauri::command]
pub async fn start_dev_training(
    app_handle:   tauri::AppHandle,
    script:       String,
    model_id:     String,
    model_name:   String,
    dataset_id:   String,
    dataset_name: String,
    refs:         HashMap<String, String>,
) -> Result<TrainingJob, String> {
    let python = get_python_path();

    // Nur ein Dev-Training gleichzeitig
    registry_start(&DEV_TRAIN_PROC)?;

    // Anti-Sleep direkt im Backend aktivieren (robust, unabhängig vom Frontend).
    if let Err(e) = crate::power_manager::enable_prevent_sleep(
        app_handle.state::<StdMutex<crate::power_manager::PowerState>>(),
    ) {
        eprintln!("[PowerManager] ⚠️ enable_prevent_sleep fehlgeschlagen: {}", e);
    }

    // Tmp-Verzeichnis für Script + Output
    let tmp_dir = app_handle.path().app_data_dir()
        .map_err(|e| format!("AppDataDir: {}", e))?
        .join("dev_scripts");
    fs::create_dir_all(&tmp_dir).ok();

    let job_id      = format!("dev_{}", &Uuid::new_v4().to_string().replace('-', "")[..12]);
    let script_path = tmp_dir.join(format!("{}.py", job_id));
    let output_dir  = tmp_dir.join(&job_id);
    fs::create_dir_all(&output_dir).ok();

    // Script schreiben
    fs::write(&script_path, &script).map_err(|e| format!("Script schreiben: {}", e))?;

    let output_path = output_dir.to_string_lossy().to_string();

    let job = TrainingJob {
        id: job_id.clone(),
        model_id: model_id.clone(),
        model_name: model_name.clone(),
        dataset_id: dataset_id.clone(),
        dataset_name: dataset_name.clone(),
        status: TrainingStatus::Running,
        config: TrainingConfig::default(),
        created_at: Utc::now(),
        started_at: Some(Utc::now()),
        completed_at: None,
        progress: TrainingProgress::default(),
        output_path: Some(output_path.clone()),
        error: None,
        user_id: String::new(), // Dev-Training läuft lokal, kein Account-Kontext nötig
    };

    let ah       = app_handle.clone();
    let jid      = job_id.clone();
    let script_p = script_path.clone();
    let out_p    = output_path.clone();
    let env_vars = refs;

    thread::spawn(move || {
        let mut cmd = Command::new(&python);
        cmd.no_window();
        cmd.python_utf8();
        cmd.arg(script_p.to_string_lossy().to_string())
           .stdout(Stdio::piped())
           .stderr(Stdio::piped());

        // Env-Variablen setzen
        cmd.env("OUTPUT_PATH", &out_p);
        // Ohne das puffert Python seine Ausgabe blockweise, sobald stdout eine
        // Pipe ist: Der Nutzer sieht waehrend des gesamten Laufs nichts und
        // bekommt alle print()-Zeilen erst am Ende auf einmal.
        cmd.env("PYTHONUNBUFFERED", "1");
        for (k, v) in &env_vars {
            cmd.env(k, v);
        }

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                let _ = registry_clear(&DEV_TRAIN_PROC);
                let _ = ah.emit("training-error", serde_json::json!({
                    "job_id": jid,
                    "data": { "error": format!("Python konnte nicht gestartet werden: {}", e) }
                }));
                return;
            }
        };
        registry_set_pid(&DEV_TRAIN_PROC, child.id());

        // Stderr in separatem Thread loggen. Die letzten Zeilen werden
        // aufgehoben und der Fehlermeldung angehängt — ohne sie stand im
        // Fehlerdialog nur der nackte Exit-Code, und der Python-Traceback
        // (also die einzige verwertbare Information) war nirgends zu sehen.
        let stderr_tail: std::sync::Arc<StdMutex<Vec<String>>> =
            std::sync::Arc::new(StdMutex::new(Vec::new()));
        let mut stderr_handle: Option<thread::JoinHandle<()>> = None;
        if let Some(stderr) = child.stderr.take() {
            let jid2 = jid.clone();
            let ah2  = ah.clone();
            let tail = std::sync::Arc::clone(&stderr_tail);
            stderr_handle = Some(thread::spawn(move || {
                for line in BufReader::new(stderr).lines().flatten() {
                    eprintln!("[DevTrain STDERR] {}", line);
                    if let Ok(mut t) = tail.lock() {
                        t.push(line.clone());
                        if t.len() > 40 { let excess = t.len() - 40; t.drain(0..excess); }
                    }
                    // Stderr-Zeilen auch als Output-Event senden
                    let _ = ah2.emit("dev-training-output", serde_json::json!({
                        "job_id": jid2, "line": format!("[ERR] {}", line)
                    }));
                }
            }));
        }

        let mut json_error = false;
        let mut step = 0u32;

        // Stdout verarbeiten
        if let Some(stdout) = child.stdout.take() {
            for line in BufReader::new(stdout).lines().flatten() {
                println!("[DevTrain] {}", line);

                // Output-Zeile immer ans Frontend senden
                let _ = ah.emit("dev-training-output", serde_json::json!({
                    "job_id": jid, "line": line.clone()
                }));

                // JSON-Events aus der train_engine parsen (falls vorhanden)
                if line.trim_start().starts_with('{') {
                    if let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) {
                        let typ = msg.get("type").and_then(|t| t.as_str()).unwrap_or("");
                        match typ {
                            "progress"  => { let _ = ah.emit("training-progress", serde_json::json!({"job_id": jid, "data": msg.get("data")})); }
                            "status"    => { let _ = ah.emit("training-status",   serde_json::json!({"job_id": jid, "data": msg.get("data")})); }
                            "complete"  => { let _ = ah.emit("training-complete", serde_json::json!({"job_id": jid, "data": msg.get("data")})); }
                            "error"     => {
                                json_error = true;
                                let _ = ah.emit("training-error", serde_json::json!({"job_id": jid, "data": msg.get("data")}));
                            }
                            _ => {}
                        }
                        continue;
                    }
                }

                // Kein JSON → Loss aus Ausgabe versuchen zu parsen (z.B. "loss: 0.345")
                // Unterstützt HuggingFace Trainer Output-Format
                if let Some(loss) = parse_loss_from_line(&line) {
                    step += 1;
                    let _ = ah.emit("training-progress", serde_json::json!({
                        "job_id": jid,
                        "data": {
                            "step": step,
                            "total_steps": 0,
                            "epoch": 0,
                            "total_epochs": 0,
                            "train_loss": loss,
                            "val_loss": null,
                            "learning_rate": 0.0,
                            "progress_percent": 0.0
                        }
                    }));
                }
            }
        }

        let status = child.wait().ok();
        // Auf den stderr-Leser warten, bevor der Tail gelesen wird — sonst
        // meldet die App "keine Fehlerausgabe erhalten", obwohl der Traceback
        // nur noch im Puffer stand.
        if let Some(h) = stderr_handle { let _ = h.join(); }
        let success = status.map(|s| s.success()).unwrap_or(false);
        let was_stopped = registry_clear(&DEV_TRAIN_PROC);

        if was_stopped {
            // User hat gestoppt – kein Fehler-/Complete-Event, das Frontend hat die UI bereits aktualisiert.
        } else if success && !json_error {
            let _ = ah.emit("training-complete", serde_json::json!({
                "job_id": jid,
                "data": { "model_path": out_p, "final_metrics": { "total_epochs": 0, "total_steps": step } }
            }));
        } else if !json_error {
            let code = status.and_then(|s| s.code());
            let code_text = match code {
                Some(c) => format!("Exit-Code {}", c),
                None    => "durch ein Signal beendet".to_string(),
            };
            let tail: Vec<String> = stderr_tail.lock().map(|t| t.clone()).unwrap_or_default();
            let error_msg = if tail.is_empty() {
                format!(
                    "Script beendet mit {} — keine Fehlerausgabe erhalten.\n\n\
                     Prüfe, ob das Skript gespeichert wurde und ob es auf stdout/stderr schreibt.",
                    code_text
                )
            } else {
                format!("Script beendet mit {}.\n\n{}", code_text, tail.join("\n"))
            };
            let _ = ah.emit("training-error", serde_json::json!({
                "job_id": jid,
                "data": { "error": error_msg, "exit_code": code, "stderr": tail }
            }));
        }

        let _ = ah.emit("training-finished", serde_json::json!({"job_id": jid, "success": success}));

        // Anti-Sleep deaktivieren sobald der Prozess endet (egal ob Success/Fail).
        if let Err(e) = crate::power_manager::disable_prevent_sleep(
            ah.state::<StdMutex<crate::power_manager::PowerState>>(),
        ) {
            eprintln!("[PowerManager] ⚠️ disable_prevent_sleep fehlgeschlagen: {}", e);
        }

        // Aufräumen
        fs::remove_file(&script_path).ok();
    });

    Ok(job)
}

/// Stoppt das laufende Dev-Training (killt den Python-Prozess samt Kindern).
#[tauri::command]
pub fn stop_dev_training(app_handle: tauri::AppHandle) -> Result<(), String> {
    registry_stop(&DEV_TRAIN_PROC);
    if let Err(e) = crate::power_manager::disable_prevent_sleep(
        app_handle.state::<StdMutex<crate::power_manager::PowerState>>(),
    ) {
        eprintln!("[PowerManager] ⚠️ disable_prevent_sleep fehlgeschlagen: {}", e);
    }
    Ok(())
}

// ══════════════════════════════════════════════════════════════════
// DEV TEST MODE
// Frontend (DevTestPanel) erwartet:
//   Events: "dev-test-output"   { job_id, line }
//           "dev-test-complete" { job_id, exit_code, data?: { error, details } }
//   Output-Verzeichnis: <app_data>/test_outputs/dev_<job_id>
// ══════════════════════════════════════════════════════════════════

/// Startet ein user-geschriebenes Test-Script (analog zu start_dev_training).
#[tauri::command]
pub async fn start_dev_test(
    app_handle:   tauri::AppHandle,
    script:       String,
    model_id:     String,
    model_name:   String,
    dataset_id:   String,
    dataset_name: String,
    refs:         HashMap<String, String>,
) -> Result<String, String> {
    let _ = (model_id, model_name, dataset_id, dataset_name); // aktuell nur fürs Logging/API-Symmetrie
    let python = get_python_path();

    registry_start(&DEV_TEST_PROC)?;

    let app_data = app_handle.path().app_data_dir()
        .map_err(|e| format!("AppDataDir: {}", e))?;

    let job_id      = format!("dev_{}", &Uuid::new_v4().to_string().replace('-', "")[..12]);
    let scripts_dir = app_data.join("dev_scripts");
    fs::create_dir_all(&scripts_dir).ok();
    let script_path = scripts_dir.join(format!("{}_test.py", job_id));
    let output_dir  = app_data.join("test_outputs").join(&job_id);
    fs::create_dir_all(&output_dir).ok();

    fs::write(&script_path, &script).map_err(|e| format!("Script schreiben: {}", e))?;

    if let Err(e) = crate::power_manager::enable_prevent_sleep(
        app_handle.state::<StdMutex<crate::power_manager::PowerState>>(),
    ) {
        eprintln!("[PowerManager] ⚠️ enable_prevent_sleep fehlgeschlagen: {}", e);
    }

    let ah       = app_handle.clone();
    let jid      = job_id.clone();
    let out_p    = output_dir.to_string_lossy().to_string();
    let env_vars = refs;

    thread::spawn(move || {
        let mut cmd = Command::new(&python);
        cmd.no_window();
        cmd.python_utf8();
        cmd.arg(script_path.to_string_lossy().to_string())
           .stdout(Stdio::piped())
           .stderr(Stdio::piped());

        cmd.env("OUTPUT_PATH", &out_p);
        cmd.env("PYTHONUNBUFFERED", "1");
        for (k, v) in &env_vars {
            cmd.env(k, v);
        }

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                let _ = registry_clear(&DEV_TEST_PROC);
                let _ = ah.emit("dev-test-complete", serde_json::json!({
                    "job_id": jid,
                    "exit_code": -1,
                    "data": { "error": format!("Python konnte nicht gestartet werden: {}", e) }
                }));
                return;
            }
        };
        registry_set_pid(&DEV_TEST_PROC, child.id());

        // Stderr: loggen, als Output-Event senden, letzte Zeilen für Fehlerdetails sammeln
        let stderr_tail: std::sync::Arc<StdMutex<Vec<String>>> =
            std::sync::Arc::new(StdMutex::new(Vec::new()));
        let mut stderr_handle: Option<thread::JoinHandle<()>> = None;
        if let Some(stderr) = child.stderr.take() {
            let jid2 = jid.clone();
            let ah2  = ah.clone();
            let tail = std::sync::Arc::clone(&stderr_tail);
            stderr_handle = Some(thread::spawn(move || {
                for line in BufReader::new(stderr).lines().flatten() {
                    eprintln!("[DevTest STDERR] {}", line);
                    let _ = ah2.emit("dev-test-output", serde_json::json!({
                        "job_id": jid2, "line": format!("[ERR] {}", line)
                    }));
                    if let Ok(mut v) = tail.lock() {
                        v.push(line);
                        if v.len() > 30 { let n = v.len() - 30; v.drain(0..n); }
                    }
                }
            }));
        }

        if let Some(stdout) = child.stdout.take() {
            for line in BufReader::new(stdout).lines().flatten() {
                println!("[DevTest] {}", line);
                let _ = ah.emit("dev-test-output", serde_json::json!({
                    "job_id": jid, "line": line
                }));
            }
        }

        let status = child.wait().ok();
        if let Some(h) = stderr_handle { let _ = h.join(); }
        let exit_code = status.and_then(|s| s.code()).unwrap_or(-1);
        let was_stopped = registry_clear(&DEV_TEST_PROC);

        if !was_stopped {
            if exit_code == 0 {
                let _ = ah.emit("dev-test-complete", serde_json::json!({
                    "job_id": jid, "exit_code": 0
                }));
            } else {
                let details = stderr_tail.lock().ok()
                    .map(|v| v.join("\n"))
                    .unwrap_or_default();
                let _ = ah.emit("dev-test-complete", serde_json::json!({
                    "job_id": jid,
                    "exit_code": exit_code,
                    "data": {
                        "error": if details.is_empty() {
                            format!(
                                "Script beendet mit Exit-Code {} — keine Fehlerausgabe erhalten.",
                                exit_code
                            )
                        } else {
                            format!("Script beendet mit Exit-Code {}.\n\n{}", exit_code, details)
                        },
                        "details": details
                    }
                }));
            }
        }

        if let Err(e) = crate::power_manager::disable_prevent_sleep(
            ah.state::<StdMutex<crate::power_manager::PowerState>>(),
        ) {
            eprintln!("[PowerManager] ⚠️ disable_prevent_sleep fehlgeschlagen: {}", e);
        }

        fs::remove_file(&script_path).ok();
    });

    Ok(job_id)
}

/// Stoppt den laufenden Dev-Test.
#[tauri::command]
pub fn stop_dev_test(app_handle: tauri::AppHandle) -> Result<(), String> {
    registry_stop(&DEV_TEST_PROC);
    if let Err(e) = crate::power_manager::disable_prevent_sleep(
        app_handle.state::<StdMutex<crate::power_manager::PowerState>>(),
    ) {
        eprintln!("[PowerManager] ⚠️ disable_prevent_sleep fehlgeschlagen: {}", e);
    }
    Ok(())
}

/// Parst den Loss-Wert aus einer HuggingFace Trainer-Ausgabezeile.
/// Beispiele:
///   "{'loss': 0.3452, 'learning_rate': 1e-05, 'epoch': 1.0}"
///   "  loss: 0.3452"
///   "[100/200] loss=0.3452"
fn parse_loss_from_line(line: &str) -> Option<f64> {
    // HuggingFace Trainer JSON-ähnliche Ausgabe
    // (line.get statt Slice — verhindert Panic, wenn die Zeile direkt nach 'loss' endet)
    if line.contains("'loss'") || line.contains("\"loss\"") {
        let re_sq = line.find("'loss'").and_then(|i| line.get(i + 6..));
        let re_dq = line.find("\"loss\"").and_then(|i| line.get(i + 6..));
        let after = re_sq.or(re_dq)?;
        let after = after.trim_start_matches([' ', ':', '\t']);
        let end = after.find([',', '}', '\n', ' ']).unwrap_or(after.len());
        after[..end].trim().parse::<f64>().ok()
    } else {
        None
    }
}
