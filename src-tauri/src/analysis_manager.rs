// analysis_manager.rs – Vollständige Implementierung
// Liest Trainings-Metriken aus DB, Step-Logs + Full-Data + AI-Reports aus Dateien

use std::fs;
use crate::command_ext::NoWindow;
use std::path::PathBuf;
use tauri::Manager;
use serde_json::{json, Value};
use rusqlite::Connection;
use chrono::Utc;

// ─────────────────────────────────────────────────────────────────────────────
// Hilfsfunktionen
// ─────────────────────────────────────────────────────────────────────────────

fn db_path(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    app.path().app_data_dir()
        .map_err(|e| format!("AppDataDir: {}", e))
        .map(|d| d.join("frametrain.db"))
}

pub fn analysis_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let dir = app.path().app_data_dir()
        .map_err(|e| format!("AppDataDir: {}", e))?
        .join("analysis");
    fs::create_dir_all(&dir).map_err(|e| format!("Mkdir analysis_dir: {}", e))?;
    Ok(dir)
}

/// FIX: Prüft ob version_id dem aktuellen User gehört
fn check_version_ownership(
    app: &tauri::AppHandle,
    version_id: &str,
    state: &tauri::State<'_, crate::AppState>,
) -> Result<(), String> {
    let user_id = {
        let db = state.db.lock().map_err(|e| format!("Lock: {}", e))?;
        db.get_current_user_id().ok_or_else(|| "Kein Nutzer eingeloggt".to_string())?
    };
    let db_file = db_path(app)?;
    if !db_file.exists() { return Ok(()); }
    let conn = Connection::open(&db_file).map_err(|e| format!("DB: {}", e))?;
    let count: i32 = conn.query_row(
        "SELECT COUNT(*) FROM model_versions_new WHERE id = ?1 AND user_id = ?2",
        rusqlite::params![version_id, &user_id],
        |row| row.get(0),
    ).unwrap_or(0);
    if count == 0 {
        return Err(format!("Zugriff verweigert: Version gehört nicht dem aktuellen Nutzer"));
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Training Metriken – aus training_metrics_new DB-Tabelle
// ─────────────────────────────────────────────────────────────────────────────

#[tauri::command]
pub async fn get_training_metrics(
    app_handle: tauri::AppHandle,
    version_id: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<Value, String> {
    // FIX: Ownership prüfen
    check_version_ownership(&app_handle, &version_id, &state)?;

    let db = db_path(&app_handle)?;
    if !db.exists() { return Ok(json!(null)); }

    let conn = Connection::open(&db).map_err(|e| format!("DB öffnen: {}", e))?;

    let res = conn.query_row(
        "SELECT id, version_id, final_train_loss, final_val_loss, \
         total_epochs, total_steps, best_epoch, training_duration_seconds, created_at \
         FROM training_metrics_new WHERE version_id = ?1",
        [&version_id],
        |row| {
            Ok(json!({
                "id":                        row.get::<_, String>(0).unwrap_or_default(),
                "version_id":                row.get::<_, String>(1).unwrap_or_default(),
                "final_train_loss":          row.get::<_, f64>(2).unwrap_or(0.0),
                "final_val_loss":            row.get::<_, Option<f64>>(3).unwrap_or(None),
                "total_epochs":              row.get::<_, i32>(4).unwrap_or(0),
                "total_steps":               row.get::<_, i32>(5).unwrap_or(0),
                "best_epoch":                row.get::<_, Option<i32>>(6).unwrap_or(None),
                "training_duration_seconds": row.get::<_, Option<i64>>(7).unwrap_or(None),
                "created_at":                row.get::<_, String>(8).unwrap_or_default(),
            }))
        },
    );

    match res {
        Ok(v) => Ok(v),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(json!(null)),
        Err(e) => Err(format!("Query training_metrics: {}", e)),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Version Details – aus model_versions_new DB-Tabelle
// ─────────────────────────────────────────────────────────────────────────────

#[tauri::command]
pub async fn get_version_details(
    app_handle: tauri::AppHandle,
    version_id: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<Value, String> {
    // FIX: Ownership prüfen
    check_version_ownership(&app_handle, &version_id, &state)?;

    let db = db_path(&app_handle)?;
    if !db.exists() { return Err("Datenbank nicht gefunden".to_string()); }

    let conn = Connection::open(&db).map_err(|e| format!("DB öffnen: {}", e))?;

    let res = conn.query_row(
        "SELECT id, model_id, version_name, version_number, path, \
         size_bytes, file_count, created_at, is_root, parent_version_id \
         FROM model_versions_new WHERE id = ?1",
        [&version_id],
        |row| {
            Ok(json!({
                "id":               row.get::<_, String>(0).unwrap_or_default(),
                "model_id":         row.get::<_, String>(1).unwrap_or_default(),
                "version_name":     row.get::<_, String>(2).unwrap_or_default(),
                "version_number":   row.get::<_, i32>(3).unwrap_or(0),
                "path":             row.get::<_, String>(4).unwrap_or_default(),
                "size_bytes":       row.get::<_, i64>(5).unwrap_or(0),
                "file_count":       row.get::<_, i32>(6).unwrap_or(0),
                "created_at":       row.get::<_, String>(7).unwrap_or_default(),
                "is_root":          row.get::<_, i32>(8).unwrap_or(0) == 1,
                "parent_version_id":row.get::<_, Option<String>>(9).unwrap_or(None),
            }))
        },
    );

    match res {
        Ok(v) => Ok(v),
        Err(rusqlite::Error::QueryReturnedNoRows) => Err("Version nicht gefunden".to_string()),
        Err(e) => Err(format!("Query version_details: {}", e)),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Step-Logs – aus Datei analysis/logs_{version_id}.json
// ─────────────────────────────────────────────────────────────────────────────

#[tauri::command]
pub async fn get_training_logs(
    app_handle: tauri::AppHandle,
    version_id: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<Vec<Value>, String> {
    // FIX: Ownership prüfen
    check_version_ownership(&app_handle, &version_id, &state)?;
    let dir = analysis_dir(&app_handle)?;
    let path = dir.join(format!("logs_{}.json", version_id));
    if !path.exists() { return Ok(vec![]); }
    let content = fs::read_to_string(&path).map_err(|e| format!("Logs lesen: {}", e))?;
    serde_json::from_str::<Vec<Value>>(&content).map_err(|e| format!("Logs JSON: {}", e))
}

#[tauri::command]
pub async fn save_training_logs(
    app_handle: tauri::AppHandle,
    version_id: String,
    logs: Vec<Value>,
) -> Result<(), String> {
    let dir = analysis_dir(&app_handle)?;
    let path = dir.join(format!("logs_{}.json", version_id));
    fs::write(&path, serde_json::to_string_pretty(&logs).map_err(|e| format!("JSON: {}", e))?)
        .map_err(|e| format!("Logs schreiben: {}", e))
}

// ─────────────────────────────────────────────────────────────────────────────
// Full Training Data – aus analysis/full_{version_id}.json
// ─────────────────────────────────────────────────────────────────────────────

#[tauri::command]
pub async fn get_training_full_data(
    app_handle: tauri::AppHandle,
    version_id: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<Value, String> {
    // FIX: Ownership prüfen
    check_version_ownership(&app_handle, &version_id, &state)?;
    let dir = analysis_dir(&app_handle)?;
    let path = dir.join(format!("full_{}.json", version_id));
    if !path.exists() { return Ok(json!(null)); }
    let content = fs::read_to_string(&path).map_err(|e| format!("Full-Data lesen: {}", e))?;
    serde_json::from_str::<Value>(&content).map_err(|e| format!("Full-Data JSON: {}", e))
}

// ─────────────────────────────────────────────────────────────────────────────
// Training Metrics speichern (von externen Aufrufen – intern via training_manager)
// ─────────────────────────────────────────────────────────────────────────────

#[tauri::command]
pub async fn save_training_metrics(
    _app_handle: tauri::AppHandle,
    _version_id: String,
    _metrics: Value,
) -> Result<(), String> {
    // Metriken werden direkt von training_manager.rs in die DB geschrieben.
    // Dieser Command ist für externe Aufrufe reserviert.
    Ok(())
}

#[tauri::command]
pub async fn update_training_progress(
    _app_handle: tauri::AppHandle,
    _job_id: String,
    _progress: Value,
) -> Result<(), String> {
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// AI-Analyse-Berichte – aus analysis/ai_report_{version_id}.json
// ─────────────────────────────────────────────────────────────────────────────

#[tauri::command]
pub async fn save_ai_analysis_report(
    app_handle: tauri::AppHandle,
    version_id: String,
    report_text: String,
    provider: String,
    model: String,
    // `language`: Sprache, in der der Bericht erzeugt wurde ("de" / "en").
    // Berichte aus älteren Versionen haben das Feld nicht — dort bleibt es null.
    language: Option<String>,
) -> Result<(), String> {
    let dir = analysis_dir(&app_handle)?;
    let path = dir.join(format!("ai_report_{}.json", version_id));
    let report = json!({
        "version_id":    version_id,
        "report_text":   report_text,
        "provider":      provider,
        "model":         model,
        "language":      language,
        "generated_at":  Utc::now().to_rfc3339(),
    });
    fs::write(
        &path,
        serde_json::to_string_pretty(&report).map_err(|e| format!("JSON: {}", e))?,
    ).map_err(|e| format!("AI-Bericht schreiben: {}", e))
}

#[tauri::command]
pub async fn get_ai_analysis_report(
    app_handle: tauri::AppHandle,
    version_id: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<Value, String> {
    // FIX: Ownership prüfen
    check_version_ownership(&app_handle, &version_id, &state)?;
    let dir = analysis_dir(&app_handle)?;
    let path = dir.join(format!("ai_report_{}.json", version_id));
    if !path.exists() { return Ok(json!(null)); }
    let content = fs::read_to_string(&path).map_err(|e| format!("AI-Bericht lesen: {}", e))?;
    serde_json::from_str::<Value>(&content).map_err(|e| format!("AI-Bericht JSON: {}", e))
}

#[tauri::command]
pub async fn delete_ai_analysis_report(
    app_handle: tauri::AppHandle,
    version_id: String,
) -> Result<(), String> {
    let dir = analysis_dir(&app_handle)?;
    let path = dir.join(format!("ai_report_{}.json", version_id));
    if path.exists() {
        fs::remove_file(&path).map_err(|e| format!("AI-Bericht löschen: {}", e))?;
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Interne Funktion: Full-Data nach Training speichern (wird von training_manager aufgerufen)
// ─────────────────────────────────────────────────────────────────────────────

pub fn save_full_analysis_data(
    app_handle: &tauri::AppHandle,
    version_id: &str,
    complete_data: &Value,
    step_logs: &[Value],
    config_json: &Value,
    started_at_secs: i64,
) {
    let dir = match analysis_dir(app_handle) {
        Ok(d) => d,
        Err(e) => { eprintln!("[Analysis] Verzeichnis: {}", e); return; }
    };

    // Bevorzugt die von der Engine gemessene Trainingsdauer. Die Differenz zu
    // "jetzt minus Startzeit" erklaerte, warum Trainingsdialog, Analyse-Header
    // und "Dauer pro Epoche" drei verschiedene Laufzeiten anzeigten.
    let wall_clock_secs = Utc::now().timestamp() - started_at_secs;
    let duration_secs = complete_data
        .get("final_metrics")
        .and_then(|m| m.get("training_duration_seconds"))
        .and_then(|v| v.as_i64())
        .filter(|d| *d > 0)
        .unwrap_or(wall_clock_secs);

    // ── Epoch Summaries aus Step-Logs berechnen ──────────────────────────────
    let mut epoch_losses: std::collections::BTreeMap<i64, Vec<f64>> = std::collections::BTreeMap::new();
    let mut epoch_val:    std::collections::BTreeMap<i64, Vec<f64>> = std::collections::BTreeMap::new();

    for log in step_logs {
        if let (Some(ep), Some(loss)) = (log["epoch"].as_i64(), log["train_loss"].as_f64()) {
            epoch_losses.entry(ep).or_default().push(loss);
        }
        if let (Some(ep), Some(val)) = (log["epoch"].as_i64(), log["val_loss"].as_f64()) {
            epoch_val.entry(ep).or_default().push(val);
        }
    }

    let epoch_summaries: Vec<Value> = epoch_losses.iter().map(|(ep, losses)| {
        let steps = losses.len();
        let avg   = losses.iter().sum::<f64>() / steps as f64;
        let min   = losses.iter().cloned().fold(f64::MAX, f64::min);
        let max   = losses.iter().cloned().fold(f64::MIN, f64::max);
        let val_l = epoch_val.get(ep).and_then(|v| {
            let s: f64 = v.iter().sum();
            Some(s / v.len() as f64)
        });
        // Geschätzte Dauer: Gesamtdauer / Epochen
        let ep_count = epoch_losses.len().max(1) as i64;
        let ep_duration = duration_secs / ep_count;
        json!({
            "epoch":          ep,
            "avg_train_loss": avg,
            "min_train_loss": min,
            "max_train_loss": max,
            "val_loss":       val_l,
            "duration_seconds": ep_duration,
            "steps":          steps,
        })
    }).collect();

    // ── Derived Stats ────────────────────────────────────────────────────────
    let all_losses: Vec<f64> = step_logs.iter()
        .filter_map(|l| l["train_loss"].as_f64()).collect();
    let initial_loss = all_losses.first().copied();
    let final_loss   = all_losses.last().copied();
    let min_loss     = all_losses.iter().cloned().fold(f64::MAX, f64::min);
    let max_loss     = all_losses.iter().cloned().fold(f64::MIN, f64::max);
    let loss_reduction_pct = match (initial_loss, final_loss) {
        (Some(i), Some(f)) if i > 0.0 => {
            Some(((i - f) / i * 1000.0).round() / 10.0) // 1 Dezimalstelle
        }
        _ => None,
    };

    // Val losses für Overfitting-Gap
    let all_val: Vec<f64> = step_logs.iter()
        .filter_map(|l| l["val_loss"].as_f64()).collect();
    let final_val = all_val.last().copied();
    let overfitting_gap_pct = match (final_loss, final_val) {
        (Some(t), Some(v)) if t > 0.0 => {
            Some(((v - t) / t * 1000.0).round() / 10.0)
        }
        _ => None,
    };

    // Grad norms
    let all_grads: Vec<f64> = step_logs.iter()
        .filter_map(|l| l["grad_norm"].as_f64()).collect();
    let avg_grad = if !all_grads.is_empty() {
        Some(((all_grads.iter().sum::<f64>() / all_grads.len() as f64) * 1000.0).round() / 1000.0)
    } else { None };
    let max_grad = all_grads.iter().cloned().fold(f64::MIN, f64::max);
    let max_grad = if max_grad > f64::MIN { Some((max_grad * 1000.0).round() / 1000.0) } else { None };

    // LR start/end
    let lr_data: Vec<f64> = step_logs.iter()
        .filter_map(|l| l["learning_rate"].as_f64()).collect();
    let initial_lr = lr_data.first().copied();
    let final_lr   = lr_data.last().copied();

    // ── Summary aus complete_data oder berechneten Werten ───────────────────
    let final_metrics = complete_data.get("final_metrics").unwrap_or(complete_data);
    let sum_train_loss = final_metrics.get("final_train_loss").and_then(|v| v.as_f64())
        .or(final_loss).unwrap_or(0.0);
    let sum_val_loss   = final_metrics.get("final_val_loss").and_then(|v| v.as_f64())
        .or(final_val);
    // Die Engine meldet die tatsaechlich durchlaufenen Epochen. Der Wert aus
    // der Config ist nur der Plan — bei Max Steps bricht das Training mitten
    // in der ersten Epoche ab, die Analyse behauptete trotzdem "3 Epochen".
    let total_epochs   = final_metrics.get("total_epochs").and_then(|v| v.as_u64())
        .or_else(|| config_json.get("epochs").and_then(|v| v.as_u64()))
        .unwrap_or(0);
    let total_steps    = final_metrics.get("total_steps").and_then(|v| v.as_i64())
        .or_else(|| step_logs.last().and_then(|l| l["step"].as_i64()))
        .unwrap_or(0);
    // best_epoch kommt ebenfalls aus der Engine statt fest auf null zu stehen.
    let best_epoch = final_metrics.get("best_epoch").and_then(|v| v.as_i64());

    // ── Config aufbereiten ───────────────────────────────────────────────────
    // Das tatsaechlich benutzte Geraet meldet die Engine. Die alte Ableitung
    // aus den fp16/bf16-Flags war schlicht geraten und zeigte auf Apple
    // Silicon immer "cpu", auch wenn auf MPS trainiert wurde.
    let hardware_device: String = final_metrics
        .get("device").and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            if config_json.get("fp16").and_then(|v| v.as_bool()).unwrap_or(false)
                || config_json.get("bf16").and_then(|v| v.as_bool()).unwrap_or(false) { "gpu".to_string() }
            else { "cpu".to_string() }
        });

    let system_ram: f64 = {
        #[cfg(target_os = "macos")] {
            std::process::Command::new("sysctl").no_window()
                .args(["-n", "hw.memsize"])
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .and_then(|s| s.trim().parse::<u64>().ok())
                .map(|b| (b as f64) / (1024.0_f64).powi(3))
                .unwrap_or(16.0)
        }
        #[cfg(not(target_os = "macos"))] { 16.0 }
    };

    let full_data = json!({
        "exported_at": Utc::now().to_rfc3339(),
        "training_summary": {
            "final_train_loss":          sum_train_loss,
            "final_val_loss":            sum_val_loss,
            "total_epochs":              total_epochs,
            "total_steps":               total_steps,
            "best_epoch":                best_epoch,
            "training_duration_seconds": duration_secs,
        },
        "config": config_json,
        "hardware": {
            "device":         hardware_device,
            "system_ram_gb":  (system_ram * 10.0).round() / 10.0,
        },
        "model_info": {
            // Werte kommen aus final_metrics der Engine. Früher stand hier
            // fest "xlm-roberta" bzw. 0 — unabhängig davon, was trainiert
            // wurde. Das machte die Analyse-Seite als Beleg wertlos.
            "architecture": final_metrics.get("architecture").and_then(|v| v.as_str()).unwrap_or("unbekannt"),
            "num_labels":   final_metrics.get("num_labels").and_then(|v| v.as_u64()),
            "lora_active":  config_json.get("use_lora").and_then(|v| v.as_bool()).unwrap_or(false),
        },
        "dataset_info": {
            "n_train":       final_metrics.get("n_train").and_then(|v| v.as_u64()).unwrap_or(0),
            "has_validation": sum_val_loss.is_some(),
            "n_val":         final_metrics.get("n_val").and_then(|v| v.as_u64()).unwrap_or(0),
            "max_seq_length": config_json.get("max_seq_length").and_then(|v| v.as_u64()).unwrap_or(128),
        },
        "classification_metrics": {
            "accuracy":  final_metrics.get("accuracy").and_then(|v| v.as_f64()),
            "f1":        final_metrics.get("f1").and_then(|v| v.as_f64()),
            "precision": final_metrics.get("precision").and_then(|v| v.as_f64()),
            "recall":    final_metrics.get("recall").and_then(|v| v.as_f64()),
        },
        // Objekterkennung wird nicht ueber Accuracy beurteilt, sondern ueber
        // mAP. Die Werte lagen in final_metrics, wurden aber nirgends gezeigt.
        "detection_metrics": {
            "map50":     final_metrics.get("mAP50").and_then(|v| v.as_f64()),
            "map50_95":  final_metrics.get("mAP50-95").and_then(|v| v.as_f64()),
            "precision": final_metrics.get("precision").and_then(|v| v.as_f64()),
            "recall":    final_metrics.get("recall").and_then(|v| v.as_f64()),
        },
        "epoch_summaries": epoch_summaries,
        "step_logs": step_logs,
        "derived_stats": {
            "initial_train_loss":  initial_loss,
            "final_train_loss":    final_loss,
            "min_train_loss":      if min_loss < f64::MAX { Some(min_loss) } else { None },
            "max_train_loss":      if max_loss > f64::MIN { Some(max_loss) } else { None },
            "loss_reduction_pct":  loss_reduction_pct,
            "overfitting_gap_pct": overfitting_gap_pct,
            "avg_grad_norm":       avg_grad,
            "max_grad_norm":       max_grad,
            "initial_lr":          initial_lr,
            "final_lr":            final_lr,
            "total_log_entries":   step_logs.len(),
        },
    });

    // Full-Data speichern
    let full_path = dir.join(format!("full_{}.json", version_id));
    if let Err(e) = fs::write(
        &full_path,
        serde_json::to_string_pretty(&full_data).unwrap_or_default(),
    ) {
        eprintln!("[Analysis] Full-Data schreiben: {}", e);
    } else {
        println!("[Analysis] ✅ Full-Data gespeichert: {:?}", full_path);
    }

    // Step-Logs separat speichern (für get_training_logs)
    let logs_path = dir.join(format!("logs_{}.json", version_id));
    if let Err(e) = fs::write(
        &logs_path,
        serde_json::to_string_pretty(step_logs).unwrap_or_default(),
    ) {
        eprintln!("[Analysis] Logs schreiben: {}", e);
    } else {
        println!("[Analysis] ✅ {} Step-Logs gespeichert: {:?}", step_logs.len(), logs_path);
    }
}
