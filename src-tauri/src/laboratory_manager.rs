// laboratory_manager.rs – Persistenter Model-Server fuer Lab-Inferenz
//
// Architektur: Rust startet einmalig einen Python-Prozess der das Modell
// laedt und dann via stdin/stdout auf Inferenz-Anfragen wartet.
// Jeder Sample-Test braucht nur noch ~50ms statt 3-5s.
// Start, Anfrage und Antwort teilt sich das Labor mit dem Hosting (model_host.rs).

use serde::{Deserialize, Serialize};
use crate::command_ext::{NoWindow, PythonUtf8};
use crate::model_host::{self, InferInput, ServerProc};
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use tauri::{Emitter, Manager};

fn get_python_path() -> String {
    // Gemeinsame Auswahl fuer Training, Tests, Labor und Einrichtung.
    crate::python_env::resolve_python()
}

pub use crate::model_host::InferResult;
#[allow(unused_imports)]
pub(crate) use crate::model_host::{get_model_server_path, get_version_info, get_yolo_server_path};

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
    pub proc:       ServerProc,
    pub version_id: String,
}

#[derive(Default)]
pub struct LabState {
    pub server: Option<LabServer>,
    pub status: ServerStatus,
    /// Zaehlt die Ladeversuche. Ein langsamer alter Start darf das Modell
    /// eines neueren nicht mehr ersetzen.
    pub generation: u64,
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
    // Alten Server beenden (Drop von ServerProc beendet den Prozess)
    let generation = {
        let mut s = state.lock().map_err(|e| format!("Lock: {}", e))?;
        s.server = None;
        s.status = ServerStatus::Loading;
        s.generation += 1;
        s.generation
    };

    let _ = app_handle.emit("lab-server-status", serde_json::json!({ "status": "loading" }));

    let fail = |msg: String| -> Result<(), String> {
        let _ = app_handle.emit("lab-server-status",
            serde_json::json!({ "status": "error", "message": msg.clone() }));
        if let Ok(mut s) = state.lock() { s.status = ServerStatus::Error; }
        Err(msg)
    };

    // ── Preflight + Server-Typ bestimmen (HuggingFace vs. YOLO vs. Canvas) ──
    let model = match model_host::resolve_model(&app_handle, &version_id) {
        Ok(m) => m,
        Err(e) => return fail(e),
    };

    // Hintergrund-Thread fuer den blockierenden Startup
    let state_arc = Arc::clone(&*state);
    let ah        = app_handle.clone();
    let vid       = version_id.clone();

    std::thread::spawn(move || {
        match model_host::spawn_server(&ah, &model, "LabServer") {
            Ok(proc) => {
                let payload = serde_json::json!({
                    "status": "ready",
                    "version_id": vid,
                    "input_kind": proc.input_kind,
                    "modality": proc.modality,
                    "classes": proc.classes,
                    "task": proc.task,
                });
                if let Ok(mut s) = state_arc.lock() {
                    if s.generation != generation {
                        // Inzwischen wurde ein anderes Modell angefordert.
                        return;
                    }
                    s.server = Some(LabServer { proc, version_id: vid });
                    s.status = ServerStatus::Ready;
                }
                let _ = ah.emit("lab-server-status", payload);
                println!("[LabServer] Bereit fuer Inferenz.");
            }
            Err(msg) => {
                if let Ok(mut s) = state_arc.lock() {
                    if s.generation != generation { return; }
                    s.status = ServerStatus::Error;
                }
                let _ = ah.emit("lab-server-status", serde_json::json!({ "status": "error", "message": msg }));
            }
        }
    });

    Ok(())
}

/// Fuehrt Inferenz auf einem einzelnen Sample durch (Text oder Datei).
/// Schnell (~50ms) weil das Modell bereits geladen ist.
#[tauri::command]
pub async fn lab_infer_sample(
    text: String,
    file_path: Option<String>,
    // Frage zum Bild (VLM). Andere Modelle ignorieren das Feld.
    question: Option<String>,
    // Clip-Grenzen in Sekunden (Video). Ohne sie gilt das ganze Video.
    start: Option<f64>,
    end: Option<f64>,
    state: tauri::State<'_, Arc<Mutex<LabState>>>,
) -> Result<InferResult, String> {
    // Asynchron: ein Text-zu-Bild-Lauf dauert Minuten. Als synchroner Befehl
    // lief er auf dem Haupt-Thread und die ganze App stand still.
    let state = state.inner().clone();
    let input = InferInput { text, file_path, question, start, end, ..Default::default() };
    tauri::async_runtime::spawn_blocking(move || infer_blocking(input, &state))
        .await
        .map_err(|e| e.to_string())?
}

fn infer_blocking(input: InferInput, state: &Arc<Mutex<LabState>>) -> Result<InferResult, String> {
    // Schreiben + Lesen atomar (Mutex haelt waehrend beider Operationen)
    let mut s = state.lock().map_err(|e| format!("Lock: {}", e))?;
    let server = s.server.as_mut()
        .ok_or_else(|| "Kein Modell geladen. Bitte warte bis das Modell fertig geladen ist.".to_string())?;
    let p = &mut server.proc;
    let req = model_host::build_request(p.is_canvas, &p.input_kind, &p.modality, &input)
        .map_err(|e| if p.is_canvas || !matches!(p.input_kind.as_str(), "image" | "audio" | "video") {
            e
        } else {
            format!("{} Lade im Labor passende Samples aus einem Dataset.", e)
        })?;
    // Bilderzeugung braucht je nach Modell und Geraet deutlich laenger als
    // eine Klassifikation (SD 1.5 auf der CPU: Minuten).
    let timeout = model_host::infer_timeout_secs(&p.modality);
    match p.request(&req, timeout, &mut |_| {}) {
        Ok(resp) => model_host::parse_result(resp),
        Err(model_host::RequestError::Crashed) => {
            // Prozess ist abgestuerzt – Server-Referenz bereinigen
            s.server = None;
            s.status = ServerStatus::Error;
            Err(model_host::RequestError::Crashed.message())
        }
        Err(e) => Err(e.message()),
    }
}

/// Beendet den laufenden Modell-Server.
#[tauri::command]
pub fn lab_stop_model_server(
    state: tauri::State<'_, Arc<Mutex<LabState>>>,
) -> Result<(), String> {
    let mut s = state.lock().map_err(|e| format!("Lock: {}", e))?;
    s.generation += 1;
    if s.server.take().is_some() {
        println!("[LabServer] Server gestoppt.");
    }
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
        "model_path": s.server.as_ref().map(|srv| &srv.proc.model_path),
        "input_kind": s.server.as_ref().map(|srv| &srv.proc.input_kind),
        "modality":   s.server.as_ref().map(|srv| &srv.proc.modality),
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

// Frueher stand hier lab_export_as_dataset, ein Platzhalter ohne Aufrufer
// ("Noch nicht implementiert"). Korrekturen als Datensatz: lab_export_corrections.
// Ergebnisse in ein Werkstatt-Projekt: studio_manager::transfer::studio_add_from_lab.

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


#[cfg(test)]
mod infer_timeout_tests {
    use crate::model_host::{has_lab_model_marker, infer_timeout_secs};

    #[test]
    fn bilderzeugung_bekommt_mehr_zeit_als_klassifikation() {
        assert_eq!(infer_timeout_secs("text"), 30);
        assert_eq!(infer_timeout_secs("image"), 30);
        assert!(infer_timeout_secs("text_to_image") >= 300);
        assert!(infer_timeout_secs("vlm") > 30);
    }

    #[test]
    fn diffusions_pipelines_sind_lab_modelle() {
        let dir = std::env::temp_dir().join(format!("ft_lab_marker_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        assert!(!has_lab_model_marker(&dir), "leerer Ordner");
        std::fs::write(dir.join("model_index.json"), "{}").unwrap();
        assert!(has_lab_model_marker(&dir), "Diffusers-Pipeline");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
