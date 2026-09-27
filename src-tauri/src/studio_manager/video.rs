// studio_manager/video.rs – Videos fuer Modelle, die Videos erkennen
//
// Ein Videoprojekt labelt Clips: je Clip eine Klasse ("springt", "faellt",
// "steht"). Ein langes Video muss dafuer nicht vorab in Dateien zerschnitten
// werden — ein Abschnitt ist ein Sample mit Start und Ende im Originalvideo.
// Das kostet keinen Speicher, und ein Abschnitt laesst sich teilen, ohne dass
// eine Datei neu entsteht. Geschnitten wird erst beim Export (video_tools.py),
// weil das Training je Datei eine Klasse liest.
//
// Alle Abschnitte eines Videos tragen dieselbe Gruppe. Beim Aufteilen bleiben
// sie zusammen: zwei Sekunden aus demselben Video in Train und Val messen,
// wie gut sich das Modell das Video gemerkt hat, nicht wie gut es erkennt.

use super::*;

/// sha256 ohne die ganze Datei in den Speicher zu laden — Videos haben
/// schnell ein paar Gigabyte.
pub(super) fn sha256_datei(p: &Path) -> std::io::Result<String> {
    use std::io::Read;
    let mut f = fs::File::open(p)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 { break; }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn collect_videos(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                if !p.file_name().and_then(|n| n.to_str()).unwrap_or("").starts_with('.') { stack.push(p); }
                continue;
            }
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
            if VIDEO_EXTS.contains(&ext.as_str()) { out.push(p); }
        }
    }
    out.sort();
    out
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct VideoInfo {
    #[serde(default)] pub duration: f64,
    #[serde(default)] pub fps:      f64,
    #[serde(default)] pub width:    u32,
    #[serde(default)] pub height:   u32,
}

/// Laenge und Masse ueber OpenCV. Fehlt Python oder OpenCV, kommt die
/// Meldung des Skripts zurueck — sie sagt, was zu installieren ist.
fn probe(app_handle: &tauri::AppHandle, video: &Path) -> Result<VideoInfo, String> {
    let script = studio_script(app_handle, "video_tools.py")?;
    let out = Command::new(crate::python_env::resolve_python()).no_window().python_utf8()
        .arg(script).arg("probe").arg("--video").arg(video)
        .output()
        .map_err(|e| format!("Python ließ sich nicht starten: {}", e))?;
    let text = String::from_utf8_lossy(&out.stdout);
    for zeile in text.lines().rev() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(zeile.trim()) else { continue };
        match v.get("type").and_then(|t| t.as_str()) {
            Some("done") => return serde_json::from_value(v).map_err(|e| format!("Videoinfo: {}", e)),
            Some("error") => return Err(v.get("message").and_then(|m| m.as_str()).unwrap_or("Fehler").to_string()),
            _ => {}
        }
    }
    Err(format!("Video ließ sich nicht lesen: {}", String::from_utf8_lossy(&out.stderr).lines().last().unwrap_or("")))
}

/// Abschnitte fester Laenge. Ein kurzer Rest am Ende wird an den letzten
/// Abschnitt gehaengt statt ein eigener Schnipsel von 0,4 Sekunden zu werden.
pub fn abschnitte(dauer: f64, laenge: f64) -> Vec<(f64, f64)> {
    if dauer <= 0.0 || laenge <= 0.0 { return vec![]; }
    let mut out: Vec<(f64, f64)> = Vec::new();
    let mut start = 0.0;
    while start < dauer - 1e-6 {
        let ende = (start + laenge).min(dauer);
        let kurz = ende - start < (laenge * 0.3).max(0.5);
        if kurz && !out.is_empty() {
            out.last_mut().unwrap().1 = ende;
        } else {
            out.push((start, ende));
        }
        start = ende;
    }
    out.into_iter().map(|(a, b)| ((a * 1000.0).round() / 1000.0, (b * 1000.0).round() / 1000.0)).collect()
}

/// Videos ins Projekt holen — Ordner (Ordnername = Klasse) oder einzelne Dateien.
///
/// `clip_seconds` zerlegt jedes Video in Abschnitte dieser Laenge; ohne
/// Angabe ist das ganze Video ein Sample.
#[tauri::command]
pub async fn studio_import_videos(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    project_id: String, paths: Vec<String>, clip_seconds: Option<f64>, ignore_labels: bool,
) -> Result<ImportReport, String> {
    let user_id = get_user_id(&state)?;
    let dir = project_dir(&app_handle, &user_id, &project_id)?;
    let mut project = load_project(&dir)?;

    // (Datei, Wurzel fuer den Klassennamen)
    let mut dateien: Vec<(PathBuf, PathBuf)> = Vec::new();
    for p in &paths {
        let p = PathBuf::from(p);
        if p.is_dir() {
            for v in collect_videos(&p) { dateien.push((v, p.clone())); }
        } else if p.is_file() {
            let wurzel = p.parent().map(Path::to_path_buf).unwrap_or_default();
            dateien.push((p, wurzel));
        }
    }
    if dateien.is_empty() {
        return Err(format!("Keine Videos gefunden ({})", VIDEO_EXTS.join(", ")));
    }

    let existing = load_samples(&dir);
    let mut bekannt: std::collections::HashSet<String> = existing.iter()
        .map(|s| format!("{}#{:.3}", s.media, s.meta.start.unwrap_or(0.0)))
        .collect();
    let laenge = clip_seconds.filter(|s| *s > 0.0);

    let mut report = ImportReport { added: 0, duplicates: 0, unreadable: 0, with_labels: 0,
        classes_added: vec![], unknown_ids: vec![], labels_ignored: 0 };
    let now = Utc::now().to_rfc3339();
    let total = dateien.len();

    for (i, (datei, wurzel)) in dateien.iter().enumerate() {
        let _ = app_handle.emit("studio-import-progress", serde_json::json!({
            "project_id": project_id, "current": i, "total": total,
        }));
        let Ok(hash) = sha256_datei(datei) else { report.unreadable += 1; continue };
        let kopf = datei_kopf(datei, 64).unwrap_or_default();
        let ext = video_ext_from_bytes(&kopf).map(str::to_string).unwrap_or_else(|| {
            datei.extension().and_then(|e| e.to_str()).unwrap_or("mp4").to_lowercase()
        });
        let rel = format!("{}/{}.{}", &hash[..2], &hash, ext);
        let ziel = dir.join("media").join(&rel);
        if let Some(parent) = ziel.parent() { fs::create_dir_all(parent).ok(); }
        if !ziel.exists() {
            fs::copy(datei, &ziel).map_err(|e| format!("Kopieren: {}", e))?;
        }

        // Laenge und Masse sind fuer Abschnitte Pflicht, sonst nur Beiwerk.
        let info = match probe(&app_handle, &ziel) {
            Ok(i) => Some(i),
            Err(e) if laenge.is_some() => return Err(e),
            Err(_) => None,
        };
        let stuecke: Vec<(Option<f64>, Option<f64>)> = match (laenge, info.as_ref()) {
            (Some(l), Some(i)) if i.duration > 0.0 => abschnitte(i.duration, l).into_iter()
                .map(|(a, b)| (Some(a), Some(b))).collect(),
            _ => vec![(None, None)],
        };

        let label = if ignore_labels { None } else {
            klasse_aus_ordner(datei, wurzel).map(|l| match class_index_for(&l, &project.classes) {
                Some(idx) => project.classes[idx].clone(),
                None => { project.classes.push(l.clone()); report.classes_added.push(l.clone()); l }
            })
        };

        for (start, ende) in stuecke {
            let schluessel = format!("{}#{:.3}", rel, start.unwrap_or(0.0));
            if !bekannt.insert(schluessel) { report.duplicates += 1; continue; }
            if label.is_some() { report.with_labels += 1; }
            append_jsonl(&samples_path(&dir), &StudioSample {
                id: format!("s_{}", &uuid::Uuid::new_v4().to_string().replace('-', "")[..10]),
                media: rel.clone(),
                mime: video_mime(&ext),
                content: None,
                status: if label.is_some() { "confirmed".to_string() } else { "new".to_string() },
                ann: Annotation { confidence: None, boxes: vec![], label: label.clone(), target: None },
                src: SampleSource { page: None, kind: "import".to_string(),
                    origin: Some(datei.to_string_lossy().to_string()), license: None, at: now.clone() },
                meta: SampleMeta {
                    w: info.as_ref().map(|i| i.width).unwrap_or(0),
                    h: info.as_ref().map(|i| i.height).unwrap_or(0),
                    group: Some(format!("video:{}", &hash[..16])),
                    start, end: ende,
                },
                abs_path: String::new(), doubt: None,
            })?;
            report.added += 1;
        }
    }

    project.updated_at = Utc::now().to_rfc3339();
    save_project(&dir, &project)?;
    let _ = app_handle.emit("studio-import-progress", serde_json::json!({
        "project_id": project_id, "current": total, "total": total, "done": true,
    }));
    Ok(report)
}

/// Teilt einen Abschnitt an einer Stelle. Der erste Teil behaelt ID, Label
/// und Status; der zweite ist neu und offen — wer teilt, tut das meist, weil
/// ab dort etwas anderes passiert.
pub fn abschnitt_teilen(dir: &Path, sample_id: &str, at: f64, dauer: Option<f64>) -> Result<String, String> {
    let mut alle: Vec<StudioSample> = read_jsonl(&samples_path(dir));
    let pos = alle.iter().position(|s| s.id == sample_id).ok_or("Sample nicht gefunden")?;
    let s = &alle[pos];
    if !s.mime.starts_with("video/") { return Err("Nur Videoabschnitte lassen sich teilen".to_string()); }
    let start = s.meta.start.unwrap_or(0.0);
    let ende = s.meta.end.or(dauer).ok_or("Die Länge des Videos ist unbekannt")?;
    // Weniger als eine Viertelsekunde je Teil ist kein Clip mehr.
    if at <= start + 0.25 || at >= ende - 0.25 {
        return Err("Die Stelle liegt zu nah am Anfang oder Ende des Abschnitts".to_string());
    }
    let mut zweiter = s.clone();
    zweiter.id = format!("s_{}", &uuid::Uuid::new_v4().to_string().replace('-', "")[..10]);
    zweiter.status = "new".to_string();
    zweiter.ann = Annotation::default();
    zweiter.meta.start = Some(at);
    zweiter.meta.end = Some(ende);
    let neu_id = zweiter.id.clone();
    alle[pos].meta.start = Some(start);
    alle[pos].meta.end = Some(at);
    alle.insert(pos + 1, zweiter);
    write_jsonl(&samples_path(dir), &alle)?;
    Ok(neu_id)
}

#[tauri::command]
pub async fn studio_split_segment(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    project_id: String, sample_id: String, at: f64, duration: Option<f64>,
) -> Result<String, String> {
    let user_id = get_user_id(&state)?;
    let dir = project_dir(&app_handle, &user_id, &project_id)?;
    let id = abschnitt_teilen(&dir, &sample_id, at, duration)?;
    let mut project = load_project(&dir)?;
    project.updated_at = Utc::now().to_rfc3339();
    save_project(&dir, &project)?;
    Ok(id)
}

/// Schneidet Abschnitte fuer den Export. Rueckgabe: die Ausgabedateien, die
/// nicht geschrieben werden konnten.
pub(super) fn clips_schneiden(
    app_handle: &tauri::AppHandle, jobs: &[serde_json::Value], arbeitsordner: &Path,
) -> Result<Vec<String>, String> {
    if jobs.is_empty() { return Ok(vec![]); }
    let script = studio_script(app_handle, "video_tools.py")?;
    let jobs_datei = arbeitsordner.join("clips_jobs.json");
    fs::write(&jobs_datei, serde_json::to_string(jobs).map_err(|e| e.to_string())?)
        .map_err(|e| format!("Clip-Liste: {}", e))?;
    let out = Command::new(crate::python_env::resolve_python()).no_window().python_utf8()
        .arg(script).arg("cut").arg("--jobs").arg(&jobs_datei)
        .output()
        .map_err(|e| format!("Python ließ sich nicht starten: {}", e))?;
    let _ = fs::remove_file(&jobs_datei);
    let text = String::from_utf8_lossy(&out.stdout);
    for zeile in text.lines().rev() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(zeile.trim()) else { continue };
        match v.get("type").and_then(|t| t.as_str()) {
            Some("done") => {
                return Ok(v.get("failed").and_then(|f| f.as_array()).map(|a| a.iter()
                    .filter_map(|x| x.get("out").and_then(|o| o.as_str()).map(str::to_string)).collect())
                    .unwrap_or_default());
            }
            Some("error") => return Err(v.get("message").and_then(|m| m.as_str()).unwrap_or("Fehler").to_string()),
            _ => {}
        }
    }
    Err(format!("Clips ließen sich nicht schneiden: {}", String::from_utf8_lossy(&out.stderr).lines().last().unwrap_or("")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abschnitte_mit_kurzem_rest() {
        assert_eq!(abschnitte(10.0, 4.0), vec![(0.0, 4.0), (4.0, 8.0), (8.0, 10.0)]);
        // 1 s Rest bei 4 s Laenge ist kein eigener Clip.
        assert_eq!(abschnitte(9.0, 4.0), vec![(0.0, 4.0), (4.0, 9.0)]);
        assert_eq!(abschnitte(9.0, 3.0), vec![(0.0, 3.0), (3.0, 6.0), (6.0, 9.0)]);
        assert_eq!(abschnitte(2.0, 5.0), vec![(0.0, 2.0)], "kurzes Video bleibt ein Abschnitt");
        assert!(abschnitte(0.0, 5.0).is_empty());
    }

    #[test]
    fn video_art_an_den_bytes() {
        let mut mp4 = vec![0, 0, 0, 0x1c];
        mp4.extend_from_slice(b"ftypisom");
        assert_eq!(video_ext_from_bytes(&mp4), Some("mp4"));
        let mut mov = vec![0, 0, 0, 0x14];
        mov.extend_from_slice(b"ftypqt  ");
        assert_eq!(video_ext_from_bytes(&mov), Some("mov"));
        let mut webm = vec![0x1A, 0x45, 0xDF, 0xA3, 0x9f, 0x42, 0x82, 0x84];
        webm.extend_from_slice(b"webm");
        assert_eq!(video_ext_from_bytes(&webm), Some("webm"));
        assert_eq!(video_ext_from_bytes(&[0x1A, 0x45, 0xDF, 0xA3, 0, 0, 0, 0]), Some("mkv"));
        let mut avi = b"RIFF".to_vec();
        avi.extend_from_slice(&[0, 0, 0, 0]);
        avi.extend_from_slice(b"AVI LIST");
        assert_eq!(video_ext_from_bytes(&avi), Some("avi"));
        assert_eq!(video_ext_from_bytes(b"kein video"), None);
    }

    #[test]
    fn abschnitt_teilen_behaelt_den_ersten_teil() {
        let dir = std::env::temp_dir().join(format!("ft_video_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let s = StudioSample {
            id: "s_v".to_string(), media: "ab/x.mp4".to_string(), mime: "video/mp4".to_string(),
            content: None, status: "confirmed".to_string(),
            ann: Annotation { label: Some("springt".to_string()), ..Default::default() },
            src: SampleSource { kind: "import".to_string(), origin: None, license: None,
                at: "2026-09-27T00:00:00Z".to_string(), page: None },
            meta: SampleMeta { group: Some("video:ab".to_string()), start: Some(2.0), end: Some(10.0), ..Default::default() },
            abs_path: String::new(), doubt: None,
        };
        append_jsonl(&samples_path(&dir), &s).unwrap();
        assert!(abschnitt_teilen(&dir, "s_v", 2.1, None).is_err(), "zu nah am Anfang");
        let neu = abschnitt_teilen(&dir, "s_v", 6.0, None).unwrap();
        let alle: Vec<StudioSample> = read_jsonl(&samples_path(&dir));
        assert_eq!(alle.len(), 2);
        assert_eq!((alle[0].meta.start, alle[0].meta.end), (Some(2.0), Some(6.0)));
        assert_eq!(alle[0].ann.label.as_deref(), Some("springt"));
        assert_eq!(alle[1].id, neu);
        assert_eq!((alle[1].meta.start, alle[1].meta.end), (Some(6.0), Some(10.0)));
        assert_eq!(alle[1].status, "new");
        assert_eq!(alle[1].meta.group.as_deref(), Some("video:ab"), "Gruppe bleibt, sonst Leck im Split");
        let _ = fs::remove_dir_all(&dir);
    }
}
