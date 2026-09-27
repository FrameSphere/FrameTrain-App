// studio_manager/transfer.rs – Medien ersetzen und Ergebnisse aus dem Labor
//
// Zwei Wege, auf denen Samples von aussen veraendert oder befuellt werden:
//
//   * Eine Aufnahme als WAV ersetzen. MediaRecorder nimmt je nach Webview MP4
//     (macOS) oder WebM (Windows) auf. WebM liest das Training ohne ffmpeg gar
//     nicht, M4A nur dort, wo das Betriebssystem es dekodiert. Die Oberflaeche
//     wandelt deshalb jede Aufnahme in 16-kHz-Mono-WAV um — neue gleich beim
//     Speichern, alte beim naechsten Oeffnen des Projekts ueber diesen Befehl.
//
//   * Aktives Lernen ueber das Labor: was dort falsch oder unsicher war, geht
//     in ein Werkstatt-Projekt. Korrigierte Ergebnisse kommen bestaetigt an,
//     die Vorhersage des Modells nur als Vorschlag — mit ihrer Sicherheit,
//     damit "Unsicherste zuerst" sie richtig einsortiert.

use super::*;

/// Ersetzt die Mediendatei eines Samples durch neue Bytes (gleiche Art).
pub fn medium_ersetzen(dir: &Path, sample_id: &str, bytes: &[u8], ext: &str) -> Result<String, String> {
    let mut alle: Vec<StudioSample> = read_jsonl(&samples_path(dir));
    let pos = alle.iter().position(|s| s.id == sample_id).ok_or("Sample nicht gefunden")?;
    let alt = alle[pos].media.clone();
    let hash = sha256_hex(bytes);
    let rel = format!("{}/{}.{}", &hash[..2], &hash, ext);
    if rel == alt { return Ok(rel); }
    let ziel = dir.join("media").join(&rel);
    if let Some(parent) = ziel.parent() { fs::create_dir_all(parent).ok(); }
    if !ziel.exists() { fs::write(&ziel, bytes).map_err(|e| format!("Speichern: {}", e))?; }
    alle[pos].media = rel.clone();
    alle[pos].mime = audio_mime(ext);
    write_jsonl(&samples_path(dir), &alle)?;
    if !alle.iter().any(|s| s.media == alt) {
        let _ = fs::remove_file(dir.join("media").join(&alt));
    }
    Ok(rel)
}

#[tauri::command]
pub async fn studio_replace_audio(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    project_id: String, sample_id: String, bytes: Vec<u8>,
) -> Result<String, String> {
    let user_id = get_user_id(&state)?;
    let dir = project_dir(&app_handle, &user_id, &project_id)?;
    // Nur WAV: genau dafuer ist der Befehl da, und die Bytes sagen es selbst.
    if audio_ext_from_bytes(&bytes) != Some("wav") {
        return Err("Erwartet wird eine WAV-Datei".to_string());
    }
    medium_ersetzen(&dir, &sample_id, &bytes, "wav")
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LabItem {
    /// Datei-Sample (Bild, Audio, Video) — sonst Text.
    pub file_path:  Option<String>,
    pub text:       Option<String>,
    /// Was das Modell gesagt hat.
    pub predicted:  Option<String>,
    pub confidence: Option<f64>,
    /// Korrektur des Nutzers: Klasse, Zieltext oder Boxen.
    pub label:      Option<String>,
    pub target:     Option<String>,
    pub boxes:      Option<Vec<crate::laboratory_manager::CorrectionBox>>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct LabTransferReport {
    pub added:      usize,
    pub confirmed:  usize,
    pub suggested:  usize,
    pub duplicates: usize,
    pub skipped:    usize,
}

fn klasse_sichern(project: &mut StudioProject, name: &str) -> usize {
    match class_index_for(name, &project.classes) {
        Some(i) => i,
        None => { project.classes.push(name.trim().to_string()); project.classes.len() - 1 }
    }
}

/// Legt Labor-Ergebnisse als Samples ab. Rein und testbar; der Befehl
/// darunter besorgt nur Pfade.
pub fn aus_labor(dir: &Path, project: &mut StudioProject, items: &[LabItem]) -> Result<LabTransferReport, String> {
    let mut r = LabTransferReport::default();
    let existing = load_samples(dir);
    let mut medien: std::collections::HashSet<String> = existing.iter().map(|s| s.media.clone()).collect();
    let mut texte: std::collections::HashSet<String> = existing.iter()
        .filter_map(|s| s.content.as_ref().map(|c| sha256_hex(c.as_bytes()))).collect();
    let boxen_projekt = project.modality == "image" && project.task != "classify";
    let ziel_projekt = project.task == "pairs" || project.task == "transcript";
    let now = Utc::now().to_rfc3339();

    for item in items {
        // Die Datei bzw. den Text ablegen.
        let (media, mime, content, w, h, origin) = if let Some(fp) = item.file_path.as_deref().filter(|p| !p.is_empty()) {
            let Ok(bytes) = fs::read(fp) else { r.skipped += 1; continue };
            let (ext, mime, w, h) = match project.modality.as_str() {
                "image" => match image_dimensions(&bytes) {
                    Some((w, h)) => { let e = image_ext_from_bytes(&bytes); (e.to_string(), mime_for(e), w, h) }
                    None => { r.skipped += 1; continue }
                },
                "audio" => match audio_ext_from_bytes(&bytes) {
                    Some(e) => (e.to_string(), audio_mime(e), 0, 0),
                    None => { r.skipped += 1; continue }
                },
                "video" => match video_ext_from_bytes(&bytes) {
                    Some(e) => (e.to_string(), video_mime(e), 0, 0),
                    None => { r.skipped += 1; continue }
                },
                _ => { r.skipped += 1; continue }
            };
            let hash = sha256_hex(&bytes);
            let rel = format!("{}/{}.{}", &hash[..2], &hash, ext);
            if !medien.insert(rel.clone()) { r.duplicates += 1; continue; }
            let ziel = dir.join("media").join(&rel);
            if let Some(parent) = ziel.parent() { fs::create_dir_all(parent).ok(); }
            if !ziel.exists() { fs::write(&ziel, &bytes).map_err(|e| format!("Speichern: {}", e))?; }
            (rel, mime, None, w, h, fp.to_string())
        } else if let Some(t) = item.text.as_deref().map(str::trim).filter(|t| !t.is_empty()) {
            if project.modality != "text" { r.skipped += 1; continue; }
            if !texte.insert(sha256_hex(t.as_bytes())) { r.duplicates += 1; continue; }
            (String::new(), "text/plain".to_string(), Some(t.to_string()), 0, 0, "Labor".to_string())
        } else {
            r.skipped += 1;
            continue;
        };

        // Korrektur schlaegt Vorhersage: eine Korrektur hat ein Mensch gesehen.
        let mut ann = Annotation::default();
        let status;
        if boxen_projekt {
            if let (Some(boxes), true) = (item.boxes.as_ref(), w > 0 && h > 0) {
                for b in boxes {
                    let cls = klasse_sichern(project, &b.label);
                    if let Some(n) = crate::yolo_export::normalize_box(cls, b.x1, b.y1, b.x2, b.y2, w as f64, h as f64) {
                        ann.boxes.push(BoxAnn { cls: n.cls, x: n.x, y: n.y, w: n.w, h: n.h });
                    }
                }
                status = "confirmed";
            } else {
                status = "new";
            }
        } else if ziel_projekt {
            if let Some(t) = item.target.as_deref().filter(|t| !t.trim().is_empty()) {
                ann.target = Some(t.trim().to_string());
                status = "confirmed";
            } else if let Some(p) = item.predicted.as_deref().filter(|t| !t.trim().is_empty()) {
                ann.target = Some(p.trim().to_string());
                ann.confidence = item.confidence;
                status = "suggested";
            } else {
                status = "new";
            }
        } else if let Some(l) = item.label.as_deref().filter(|l| !l.trim().is_empty()) {
            let i = klasse_sichern(project, l);
            ann.label = Some(project.classes[i].clone());
            status = "confirmed";
        } else if let Some(p) = item.predicted.as_deref().filter(|l| !l.trim().is_empty()) {
            let i = klasse_sichern(project, p);
            ann.label = Some(project.classes[i].clone());
            ann.confidence = item.confidence;
            status = "suggested";
        } else {
            status = "new";
        }
        match status { "confirmed" => r.confirmed += 1, "suggested" => r.suggested += 1, _ => {} }

        append_jsonl(&samples_path(dir), &StudioSample {
            id: format!("s_{}", &uuid::Uuid::new_v4().to_string().replace('-', "")[..10]),
            media, mime, content, status: status.to_string(), ann,
            src: SampleSource { kind: "lab".to_string(), origin: Some(origin), license: None, at: now.clone(), page: None },
            meta: SampleMeta { w, h, ..Default::default() },
            abs_path: String::new(), doubt: None,
        })?;
        r.added += 1;
    }
    Ok(r)
}

#[tauri::command]
pub async fn studio_add_from_lab(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    project_id: String, items: Vec<LabItem>,
) -> Result<LabTransferReport, String> {
    let user_id = get_user_id(&state)?;
    let dir = project_dir(&app_handle, &user_id, &project_id)?;
    let mut project = load_project(&dir)?;
    let r = aus_labor(&dir, &mut project, &items)?;
    project.updated_at = Utc::now().to_rfc3339();
    save_project(&dir, &project)?;
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn projekt(modality: &str, task: &str) -> StudioProject {
        StudioProject { id: "p".into(), name: "P".into(), modality: modality.into(), task: task.into(),
            target_format: String::new(), classes: vec!["ja".into()], created_at: String::new(), updated_at: String::new() }
    }

    fn tempdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ft_transfer_{}_{}", tag, uuid::Uuid::new_v4()));
        fs::create_dir_all(d.join("media")).unwrap();
        d
    }

    #[test]
    fn korrektur_bestaetigt_vorhersage_schlaegt_nur_vor() {
        let dir = tempdir("text");
        let mut p = projekt("text", "classification");
        let items = vec![
            LabItem { file_path: None, text: Some("Das war gut".into()), predicted: Some("nein".into()),
                confidence: Some(0.4), label: Some("ja".into()), target: None, boxes: None },
            LabItem { file_path: None, text: Some("Das war schlecht".into()), predicted: Some("nein".into()),
                confidence: Some(0.55), label: None, target: None, boxes: None },
            LabItem { file_path: None, text: Some("Das war gut".into()), predicted: None,
                confidence: None, label: None, target: None, boxes: None },
        ];
        let r = aus_labor(&dir, &mut p, &items).unwrap();
        assert_eq!((r.added, r.confirmed, r.suggested, r.duplicates), (2, 1, 1, 1));
        assert_eq!(p.classes, vec!["ja".to_string(), "nein".to_string()], "unbekannte Klasse wird angelegt");
        let s = load_samples(&dir);
        assert_eq!(s[0].status, "confirmed");
        assert_eq!(s[0].ann.label.as_deref(), Some("ja"));
        assert_eq!(s[1].status, "suggested");
        assert_eq!(s[1].ann.confidence, Some(0.55));
        assert_eq!(s[1].src.kind, "lab");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn wav_ersetzt_die_alte_aufnahme() {
        let dir = tempdir("wav");
        fs::create_dir_all(dir.join("media/ab")).unwrap();
        fs::write(dir.join("media/ab/alt.m4a"), b"alt").unwrap();
        append_jsonl(&samples_path(&dir), &StudioSample {
            id: "s_1".into(), media: "ab/alt.m4a".into(), mime: "audio/mp4".into(), content: None,
            status: "confirmed".into(), ann: Annotation { label: Some("ja".into()), ..Default::default() },
            src: SampleSource { kind: "record".into(), origin: None, license: None, at: String::new(), page: None },
            meta: SampleMeta::default(), abs_path: String::new(), doubt: None,
        }).unwrap();
        let mut wav = b"RIFF\0\0\0\0WAVEfmt ".to_vec();
        wav.extend_from_slice(&[0u8; 32]);
        let rel = medium_ersetzen(&dir, "s_1", &wav, "wav").unwrap();
        assert!(rel.ends_with(".wav"));
        let s = load_samples(&dir);
        assert_eq!(s[0].media, rel);
        assert_eq!(s[0].mime, "audio/wav");
        assert_eq!(s[0].ann.label.as_deref(), Some("ja"), "Label bleibt");
        assert!(!dir.join("media/ab/alt.m4a").exists(), "alte Datei ist weg");
        let _ = fs::remove_dir_all(&dir);
    }
}
