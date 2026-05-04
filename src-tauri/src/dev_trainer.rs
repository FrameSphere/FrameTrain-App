// dev_trainer.rs – Führt ein user-geschriebenes Python-Script aus
// und emitiert die gleichen Events wie der normale Trainer

use std::fs;
use std::path::PathBuf;
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

// Gleiche Python-Pfad-Erkennung wie training_manager
fn get_python_path() -> String {
    let candidates = if cfg!(target_os = "windows") {
        vec!["python", "python3"]
    } else {
        vec!["python3", "python"]
    };
    for cmd in &candidates {
        if Command::new(cmd).arg("--version").output().map(|o| o.status.success()).unwrap_or(false) {
            return cmd.to_string();
        }
    }
    "python3".to_string()
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
    };

    let ah       = app_handle.clone();
    let jid      = job_id.clone();
    let script_p = script_path.clone();
    let out_p    = output_path.clone();
    let env_vars = refs;

    thread::spawn(move || {
        let mut cmd = Command::new(&python);
        cmd.arg(script_p.to_string_lossy().to_string())
           .stdout(Stdio::piped())
           .stderr(Stdio::piped());

        // Env-Variablen setzen
        cmd.env("OUTPUT_PATH", &out_p);
        for (k, v) in &env_vars {
            cmd.env(k, v);
        }

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                let _ = ah.emit("training-error", serde_json::json!({
                    "job_id": jid,
                    "data": { "error": format!("Python konnte nicht gestartet werden: {}", e) }
                }));
                return;
            }
        };

        // Stderr in separatem Thread loggen
        if let Some(stderr) = child.stderr.take() {
            let jid2 = jid.clone();
            let ah2  = ah.clone();
            thread::spawn(move || {
                for line in BufReader::new(stderr).lines().flatten() {
                    eprintln!("[DevTrain STDERR] {}", line);
                    // Stderr-Zeilen auch als Output-Event senden
                    let _ = ah2.emit("dev-training-output", serde_json::json!({
                        "job_id": jid2, "line": format!("[ERR] {}", line)
                    }));
                }
            });
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
        let success = status.map(|s| s.success()).unwrap_or(false);

        if success && !json_error {
            let _ = ah.emit("training-complete", serde_json::json!({
                "job_id": jid,
                "data": { "model_path": out_p, "final_metrics": { "total_epochs": 0, "total_steps": step } }
            }));
        } else if !json_error {
            let _ = ah.emit("training-error", serde_json::json!({
                "job_id": jid,
                "data": { "error": format!("Script beendet mit Exit-Code {:?}", status.and_then(|s| s.code())) }
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

/// Parst den Loss-Wert aus einer HuggingFace Trainer-Ausgabezeile.
/// Beispiele:
///   "{'loss': 0.3452, 'learning_rate': 1e-05, 'epoch': 1.0}"
///   "  loss: 0.3452"
///   "[100/200] loss=0.3452"
fn parse_loss_from_line(line: &str) -> Option<f64> {
    // HuggingFace Trainer JSON-ähnliche Ausgabe
    if line.contains("'loss'") || line.contains("\"loss\"") {
        let re_sq = line.find("'loss'").map(|i| &line[i + 7..]);
        let re_dq = line.find("\"loss\"").map(|i| &line[i + 7..]);
        let after = re_sq.or(re_dq)?;
        let after = after.trim_start_matches([' ', ':', '\t']);
        let end = after.find([',', '}', '\n', ' ']).unwrap_or(after.len());
        after[..end].trim().parse::<f64>().ok()
    } else {
        None
    }
}
