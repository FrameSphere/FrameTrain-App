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

/// Macht die gemeinsame Dataset-Logik (python/ft_data) im Dev-Skript importierbar.
/// Die Vorlagen nutzen sie fuer Klassenordner, Splits und HF-Parquet — dieselben
/// Regeln wie Training und Test.
fn add_ft_data_path(cmd: &mut Command, app_handle: &tauri::AppHandle) {
    if let Some(root) = crate::training_manager::python_root(app_handle) {
        let mut paths = vec![root];
        if let Some(existing) = std::env::var_os("PYTHONPATH") {
            paths.extend(std::env::split_paths(&existing));
        }
        if let Ok(joined) = std::env::join_paths(paths) {
            cmd.env("PYTHONPATH", joined);
        }
    }
}

/// Ergaenzt zu jedem DATASET_PATH[_n] ein DATASET_YAML[_n], wenn das Dataset eine
/// (gepruefte) dataset.yaml hat. YOLO-Skripte brauchen die yaml, nicht den Ordner —
/// vorher mussten sie sie selbst suchen und wussten nicht, ob sie noch stimmt.
fn add_dataset_yaml_refs(env_vars: &mut HashMap<String, String>) {
    let extra: Vec<(String, String)> = env_vars.iter()
        .filter_map(|(k, v)| {
            let suffix = k.strip_prefix("DATASET_PATH")?;
            if v.is_empty() { return None; }
            let key = format!("DATASET_YAML{}", suffix);
            if env_vars.contains_key(&key) { return None; }
            let yaml = crate::dataset_manager::ensure_dataset_yaml(std::path::Path::new(v))?;
            Some((key, yaml.to_string_lossy().to_string()))
        })
        .collect();
    env_vars.extend(extra);
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
    version_id:   Option<String>,
) -> Result<TrainingJob, String> {
    let python = get_python_path();

    // Nur ein Dev-Training gleichzeitig
    registry_start(&DEV_TRAIN_PROC)?;

    let user_id = app_handle.state::<crate::AppState>().db.lock().ok()
        .and_then(|db| db.get_current_user_id())
        .unwrap_or_default();

    // Anti-Sleep direkt im Backend aktivieren (robust, unabhängig vom Frontend).
    if let Err(e) = crate::power_manager::enable_prevent_sleep(
        app_handle.state::<StdMutex<crate::power_manager::PowerState>>(),
    ) {
        eprintln!("[PowerManager] ⚠️ enable_prevent_sleep fehlgeschlagen: {}", e);
    }

    // Script liegt temporaer in dev_scripts/, der Output unter
    // training_outputs/dev_<id> — genau dort, wo Pfade-Panel und Coach ihn
    // nennen. Vorher landete er unbemerkt in dev_scripts/dev_<id>.
    let app_data = app_handle.path().app_data_dir()
        .map_err(|e| format!("AppDataDir: {}", e))?;
    let tmp_dir = app_data.join("dev_scripts");
    fs::create_dir_all(&tmp_dir).ok();

    let job_id      = format!("dev_{}", &Uuid::new_v4().to_string().replace('-', "")[..12]);
    let script_path = tmp_dir.join(format!("{}.py", job_id));
    let output_dir  = app_data.join("training_outputs").join(&job_id);
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
    let mut env_vars = refs;
    add_dataset_yaml_refs(&mut env_vars);

    thread::spawn(move || {
        let mut cmd = Command::new(&python);
        cmd.no_window();
        cmd.python_utf8();
        cmd.arg(script_p.to_string_lossy().to_string())
           .stdout(Stdio::piped())
           .stderr(Stdio::piped());

        // Env-Variablen setzen
        add_ft_data_path(&mut cmd, &ah);
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
        // Fuer die Version: Step-Logs (Analyse-Seite) und das complete-Event des
        // Skripts. Die Version entsteht erst nach Exit-Code 0 — ein Skript, das
        // nach "complete" noch abstuerzt, soll keine halbe Version hinterlassen.
        let started_at_secs = Utc::now().timestamp();
        let mut step_logs: Vec<serde_json::Value> = Vec::new();
        let mut complete_data: Option<serde_json::Value> = None;

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
                            "progress"  => {
                                if let Some(entry) = msg.get("data").and_then(|d| step_log_entry(d, started_at_secs)) {
                                    step_logs.push(entry);
                                }
                                let _ = ah.emit("training-progress", serde_json::json!({"job_id": jid, "data": msg.get("data")}));
                            }
                            "status"    => { let _ = ah.emit("training-status",   serde_json::json!({"job_id": jid, "data": msg.get("data")})); }
                            // Erst nach dem Prozessende melden (siehe unten) — vorher kam
                            // training-complete doppelt: einmal hier, einmal beim Exit.
                            "complete"  => { complete_data = msg.get("data").cloned(); }
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
                    if let Some(entry) = step_log_entry(&serde_json::json!({"step": step, "train_loss": loss}), started_at_secs) {
                        step_logs.push(entry);
                    }
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
            let mut data = complete_data.unwrap_or_else(|| serde_json::json!({}));
            fill_final_metrics(&mut data, &step_logs, step, Utc::now().timestamp() - started_at_secs);
            let model_path = data.get("model_path").and_then(|v| v.as_str())
                .filter(|p| std::path::Path::new(p).exists())
                .unwrap_or(&out_p).to_string();
            if let Some(obj) = data.as_object_mut() {
                obj.insert("model_path".into(), serde_json::json!(model_path));
            }

            // Wie das normale Training: Ergebnis als Modellversion ablegen, damit
            // es in Modelle, Analyse, Tests und im Labor auftaucht. Vorher blieb
            // es unbemerkt in training_outputs/dev_<id> liegen.
            let mut payload = serde_json::json!({ "job_id": jid, "data": data });
            if !crate::training_manager::has_model_weights(std::path::Path::new(&model_path)) {
                payload["version_error"] = serde_json::json!(format!(
                    "Im Ausgabeordner liegen keine Modellgewichte (.safetensors, .bin, .pt) — keine Version angelegt. \
                     Speichere das Modell nach OUTPUT_PATH (z. B. model.save_pretrained(OUTPUT_PATH)). Ordner: {}",
                    model_path
                ));
            } else {
                match crate::training_manager::create_version(&ah, &model_id, &model_name, version_id.clone(), &model_path, &user_id) {
                    Ok(new_vid) => {
                        if let Err(e) = crate::training_manager::save_metrics(&ah, &new_vid, &data, &user_id) {
                            eprintln!("[DevTrain] Metriken: {}", e);
                        }
                        let config = serde_json::json!({ "mode": "dev", "dataset_id": dataset_id, "dataset_name": dataset_name });
                        crate::analysis_manager::save_full_analysis_data(&ah, &new_vid, &data, &step_logs, &config, started_at_secs);
                        if let Ok(db) = ah.state::<crate::AppState>().db.lock() {
                            let _ = db.mark_dataset_used(&dataset_id);
                        }
                        payload["new_version_id"] = serde_json::json!(new_vid);
                    }
                    Err(e) => {
                        eprintln!("[DevTrain] Version: {}", e);
                        payload["version_error"] = serde_json::json!(e);
                    }
                }
            }
            let _ = ah.emit("training-complete", payload);
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
    let mut env_vars = refs;
    add_dataset_yaml_refs(&mut env_vars);

    thread::spawn(move || {
        let mut cmd = Command::new(&python);
        cmd.no_window();
        cmd.python_utf8();
        cmd.arg(script_path.to_string_lossy().to_string())
           .stdout(Stdio::piped())
           .stderr(Stdio::piped());

        add_ft_data_path(&mut cmd, &ah);
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

/// Beschreibt Modell, Dataset und installierte Pakete fuer die Code-KI in
/// Dev Train / Dev Test (python/ft_data/dev_context.py). Vorher kannte die KI
/// nur die Pfade und riet bei Chronos-2 + Zeitreihen-Tabelle ein Skript mit
/// YOLO-, Klassifikations- und LLM-Zweig zusammen.
#[tauri::command]
pub async fn describe_dev_context(
    app_handle:   tauri::AppHandle,
    model_path:   String,
    dataset_path: Option<String>,
) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let mut cmd = Command::new(get_python_path());
        cmd.no_window();
        cmd.python_utf8();
        add_ft_data_path(&mut cmd, &app_handle);
        cmd.args(["-m", "ft_data.dev_context", &model_path, dataset_path.as_deref().unwrap_or("")]);
        let out = cmd.output().map_err(|e| format!("Python konnte nicht gestartet werden: {}", e))?;
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            let last = err.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("ohne Fehlerausgabe");
            return Err(format!("Modell/Dataset nicht lesbar: {}", last));
        }
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    })
    .await
    .map_err(|e| format!("describe_dev_context: {}", e))?
}

/// Step-Log-Eintrag fuer die Analyse-Seite (gleiches Format wie das normale
/// Training). Events ohne train_loss (reine Eval-Logs) zaehlen nicht — sonst
/// stuende die Loss-Kurve an diesen Stellen auf 0.
fn step_log_entry(data: &serde_json::Value, started_at_secs: i64) -> Option<serde_json::Value> {
    let train_loss = data.get("train_loss").and_then(|v| v.as_f64())?;
    Some(serde_json::json!({
        "epoch":         data.get("epoch").and_then(|v| v.as_i64()).unwrap_or(1),
        "step":          data.get("step").and_then(|v| v.as_i64()).unwrap_or(0),
        "train_loss":    train_loss,
        "val_loss":      data.get("val_loss").and_then(|v| v.as_f64()),
        "learning_rate": data.get("learning_rate").and_then(|v| v.as_f64()).unwrap_or(0.0),
        "grad_norm":     data.get("grad_norm").and_then(|v| v.as_f64()),
        "elapsed_seconds": Utc::now().timestamp() - started_at_secs,
        "timestamp":     Utc::now().to_rfc3339(),
    }))
}

/// Ergaenzt final_metrics um das, was eigene Skripte meist weglassen
/// (final_train_loss, total_epochs, total_steps) — aus den Step-Logs.
/// Ohne total_epochs > 0 legt save_metrics keinen Eintrag an, und die
/// Version stuende ohne Kennzahlen in der Liste.
fn fill_final_metrics(data: &mut serde_json::Value, step_logs: &[serde_json::Value], parsed_steps: u32, duration_secs: i64) {
    if !data.is_object() { *data = serde_json::json!({}); }
    if !data.get("final_metrics").map(|m| m.is_object()).unwrap_or(false) {
        data["final_metrics"] = serde_json::json!({});
    }
    let last = step_logs.last();
    let m = &mut data["final_metrics"];
    if m.get("final_train_loss").and_then(|v| v.as_f64()).is_none() {
        if let Some(l) = m.get("train_loss").and_then(|v| v.as_f64())
            .or_else(|| last.and_then(|e| e["train_loss"].as_f64())) {
            m["final_train_loss"] = serde_json::json!(l);
        }
    }
    if m.get("final_val_loss").and_then(|v| v.as_f64()).is_none() {
        if let Some(v) = m.get("val_loss").and_then(|v| v.as_f64())
            .or_else(|| step_logs.iter().rev().find_map(|e| e["val_loss"].as_f64())) {
            m["final_val_loss"] = serde_json::json!(v);
        }
    }
    if m.get("total_epochs").and_then(|v| v.as_i64()).unwrap_or(0) <= 0 {
        let max_ep = step_logs.iter().filter_map(|e| e["epoch"].as_i64()).max().unwrap_or(0);
        m["total_epochs"] = serde_json::json!(max_ep.max(1));
    }
    if m.get("total_steps").and_then(|v| v.as_i64()).unwrap_or(0) <= 0 {
        let steps = last.and_then(|e| e["step"].as_i64()).unwrap_or(parsed_steps as i64);
        m["total_steps"] = serde_json::json!(steps);
    }
    // Ohne Dauer zeigte die Analyse "Training time: -".
    if m.get("training_duration_seconds").and_then(|v| v.as_i64()).unwrap_or(0) <= 0 {
        m["training_duration_seconds"] = serde_json::json!(duration_secs.max(1));
    }
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
        // transformers 5 schreibt die Werte in Anfuehrungszeichen
        // ({'loss': '0.3582', …}) — ohne trim_matches blieb die Loss-Kurve leer.
        after[..end].trim().trim_matches(['\'', '"']).parse::<f64>().ok()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loss_wird_auch_in_anfuehrungszeichen_gelesen() {
        assert_eq!(parse_loss_from_line("{'loss': 0.3452, 'learning_rate': 1e-05}"), Some(0.3452));
        assert_eq!(parse_loss_from_line("{'loss': '0.3582', 'grad_norm': '15.26', 'epoch': '1'}"), Some(0.3582));
        assert_eq!(parse_loss_from_line("{\"loss\": \"0.5\"}"), Some(0.5));
        assert_eq!(parse_loss_from_line("{'train_runtime': '29.09'}"), None);
    }

    #[test]
    fn eval_logs_ohne_train_loss_zaehlen_nicht() {
        assert!(step_log_entry(&serde_json::json!({"step": 5, "val_loss": 0.4}), 0).is_none());
        let e = step_log_entry(&serde_json::json!({"step": 5, "epoch": 2, "train_loss": 0.7}), 0).unwrap();
        assert_eq!(e["epoch"], 2);
        assert_eq!(e["train_loss"], 0.7);
    }

    #[test]
    fn final_metrics_werden_aus_step_logs_ergaenzt() {
        let logs = vec![
            step_log_entry(&serde_json::json!({"step": 5, "epoch": 1, "train_loss": 0.9}), 0).unwrap(),
            step_log_entry(&serde_json::json!({"step": 10, "epoch": 1, "train_loss": 0.6, "val_loss": 0.5}), 0).unwrap(),
        ];
        let mut data = serde_json::json!({"model_path": "/x", "final_metrics": {"accuracy": 0.8, "val_loss": 0.55, "total_epochs": 1}});
        fill_final_metrics(&mut data, &logs, 0, 40);
        let m = &data["final_metrics"];
        assert_eq!(m["final_train_loss"], 0.6);
        assert_eq!(m["final_val_loss"], 0.55);
        assert_eq!(m["total_epochs"], 1);
        assert_eq!(m["total_steps"], 10);
        assert_eq!(m["accuracy"], 0.8);
        assert_eq!(m["training_duration_seconds"], 40);
    }

    #[test]
    fn skript_ohne_complete_event_bekommt_trotzdem_metriken() {
        let mut data = serde_json::json!({});
        fill_final_metrics(&mut data, &[], 3, 0);
        assert_eq!(data["final_metrics"]["total_epochs"], 1);
        assert_eq!(data["final_metrics"]["total_steps"], 3);
        assert_eq!(data["final_metrics"]["training_duration_seconds"], 1);
    }
}
