// studio_manager/regeln.rs – Was in ein Projekt darf und was in den Export
//
// Die Oberflaeche bietet je Projektart nur passende Wege an. Ueber MCP kommt
// dagegen jede Kombination an: Text fuer ein Bildprojekt, Boxen in Pixeln,
// eine Klasse, die es nicht gibt. Die Pruefungen stehen deshalb hier und nicht
// in der Oberflaeche — der Testlauf vom 22.11. hat gezeigt, dass sonst erst
// das Training auffaellt (oder gar nicht).

use super::*;

/// Passt das Medium zum Projekt? `medium` ist die Modalitaet, die hereinkommt.
pub(super) fn medium_pruefen(project: &StudioProject, medium: &str) -> Result<(), String> {
    if project.modality == medium { return Ok(()); }
    let name = |m: &str| match m {
        "image" => "Bilder", "text" => "Texte", "audio" => "Audio", "video" => "Videos", _ => "andere Daten",
    };
    Err(format!("Das Projekt „{}“ sammelt {} – {} passen nicht hinein. Bitte ein passendes Projekt wählen.",
        project.name, name(&project.modality), name(medium)))
}

/// Arbeitet das Projekt mit Zieltexten statt mit Klassen?
pub(super) fn mit_zieltext(project: &StudioProject) -> bool {
    matches!(project.task.as_str(), "pairs" | "transcript")
}

fn ist_boxenprojekt(project: &StudioProject) -> bool {
    project.modality == "image" && project.task != "classify"
}

// ── Schreiben ───────────────────────────────────────────────────────────────

/// Neuer Basiszustand; die Ereignisse sind darin schon enthalten.
fn basis_schreiben(dir: &Path, samples: &[StudioSample]) -> Result<(), String> {
    write_jsonl(&samples_path(dir), samples)?;
    if events_path(dir).exists() {
        fs::write(events_path(dir), b"").map_err(|e| format!("Events leeren: {}", e))?;
    }
    Ok(())
}

/// Gibt es das Sample? Bewusst eine Textsuche: die Frage kommt bei jedem
/// Tastendruck, und 20 000 Zeilen dafuer zu parsen waere zu teuer.
fn sample_vorhanden(dir: &Path, sample_id: &str) -> bool {
    let Ok(text) = fs::read_to_string(samples_path(dir)) else { return false };
    let anfang = format!("{{\"id\":\"{}\"", sample_id);
    text.lines().any(|l| l.starts_with(&anfang))
}

// ── Annotation ──────────────────────────────────────────────────────────────

/// Was eine Anfrage am Sample aendern soll. `None` heisst "nicht angegeben".
pub(super) struct Wunsch {
    pub status: String,
    pub boxes:  Option<Vec<BoxAnn>>,
    pub label:  Option<String>,
    pub target: Option<String>,
    /// Nicht angegebene Felder behalten ihren Wert. Die Oberflaeche schickt
    /// immer den ganzen Stand (leer heisst dort leer), MCP nur das Geaenderte.
    pub keep:   bool,
}

/// Box auf das Bild beschnitten. `None`: zu klein, um etwas zu zeigen.
fn box_pruefen(b: &BoxAnn) -> Result<Option<BoxAnn>, String> {
    let ok = |v: f64| v.is_finite() && (-0.01..=1.01).contains(&v);
    if ![b.x, b.y, b.w, b.h].into_iter().all(ok) {
        return Err(format!(
            "Box außerhalb des Bildes (x {}, y {}, w {}, h {}). Erwartet werden Mittelpunkt und Größe als Anteil des Bildes von 0 bis 1, keine Pixel.",
            b.x, b.y, b.w, b.h));
    }
    Ok(crate::yolo_export::norm_box(b.cls, b.x, b.y, b.w, b.h)
        .map(|n| BoxAnn { cls: n.cls, x: n.x, y: n.y, w: n.w, h: n.h })
        .filter(|b| b.w > 0.0005 && b.h > 0.0005))
}

/// Prueft den Wunsch gegen Projekt und Sample und macht ein Ereignis daraus.
pub(super) fn annotation_ereignis(
    dir: &Path, project: &StudioProject, sample_id: &str, w: Wunsch,
) -> Result<AnnEvent, String> {
    if !matches!(w.status.as_str(), "new" | "suggested" | "confirmed" | "skipped") {
        return Err(format!("Unbekannter Status: {}", w.status));
    }
    if !sample_vorhanden(dir, sample_id) {
        return Err(format!("Sample nicht gefunden: {}", sample_id));
    }
    let bisher = || load_samples(dir).into_iter().find(|s| s.id == sample_id).map(|s| s.ann).unwrap_or_default();
    let alt = if w.keep { Some(bisher()) } else { None };

    // Boxen. Eine ungueltige Klasse, die schon am Sample steht (aeltere
    // Projekte), darf das Speichern der uebrigen Aenderungen nicht verhindern.
    let boxes = match (w.boxes, &alt) {
        (None, Some(a)) => a.boxes.clone(),
        (None, None) => vec![],
        (Some(neu), _) => {
            let mut out = Vec::with_capacity(neu.len());
            for b in &neu {
                if b.cls >= project.classes.len() && !bisher().boxes.iter().any(|a| a.cls == b.cls) {
                    return Err(format!("Die Klasse mit der Nummer {} gibt es nicht. Das Projekt hat {} Klasse(n): {}.",
                        b.cls, project.classes.len(), project.classes.join(", ")));
                }
                if let Some(b) = box_pruefen(b)? { out.push(b); }
            }
            out
        }
    };

    let gesetzt = |v: Option<String>, alt_wert: Option<String>| match v {
        Some(t) => Some(t).filter(|t| !t.trim().is_empty()),
        None => alt_wert,
    };
    let mut label = gesetzt(w.label, alt.as_ref().and_then(|a| a.label.clone()));
    let target = gesetzt(w.target, alt.as_ref().and_then(|a| a.target.clone()));
    if mit_zieltext(project) || ist_boxenprojekt(project) {
        label = None;
    } else if let Some(l) = label.clone() {
        match class_index_for(&l, &project.classes) {
            Some(i) => label = Some(project.classes[i].clone()),
            None if bisher().label.as_deref() == Some(l.as_str()) => {}
            None => return Err(format!("Die Klasse „{}“ gibt es im Projekt nicht. Vorhanden: {}.",
                l, project.classes.join(", "))),
        }
    }

    // Ueber MCP ist ein leeres "bestaetigt" immer ein Versehen. Ein Bild ohne
    // Box bleibt erlaubt: das ist ein Negativbeispiel.
    if w.keep && w.status == "confirmed" {
        if mit_zieltext(project) && target.is_none() {
            return Err("Zum Bestätigen fehlt der Zieltext (target).".to_string());
        }
        if !mit_zieltext(project) && !ist_boxenprojekt(project) && label.is_none() {
            return Err("Zum Bestätigen fehlt die Klasse (label).".to_string());
        }
    }

    let confidence = if w.status == "suggested" { alt.and_then(|a| a.confidence) } else { None };
    Ok(AnnEvent {
        sample_id: sample_id.to_string(), status: w.status, boxes, label, target, confidence,
        at: Utc::now().to_rfc3339(),
    })
}

// ── Klassenliste ────────────────────────────────────────────────────────────

fn gleich(a: &str, b: &str) -> bool { a.trim().to_lowercase() == b.trim().to_lowercase() }

/// Klassen ohne Dubletten ("A", "a", " A " sind eine Klasse).
pub(super) fn klassen_eindeutig(klassen: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for k in klassen {
        let k = k.trim().to_string();
        if !k.is_empty() && !out.iter().any(|o| gleich(o, &k)) { out.push(k); }
    }
    out
}

/// Wo steht jede bisherige Klasse in der neuen Liste?
///
/// Ein Name, den es noch gibt, ist nur verschoben. Einer, den es nicht mehr
/// gibt, wurde an seiner Stelle umbenannt — aber nur, wenn dort wirklich ein
/// neuer Name steht. Sonst ist nicht zu sagen, was gemeint war.
pub(super) fn klassen_abbilden(alt: &[String], neu: &[String]) -> Result<Vec<usize>, String> {
    if neu.len() < alt.len() {
        return Err("Klassen können umbenannt und ergänzt, aber nicht entfernt werden".to_string());
    }
    for (i, n) in neu.iter().enumerate() {
        let unveraendert = alt.get(i).map(|a| a == n).unwrap_or(false);
        if !unveraendert && neu.iter().enumerate().any(|(j, m)| j != i && gleich(m, n)) {
            return Err(format!("Die Klasse „{}“ steht mehrfach in der Liste", n));
        }
    }
    alt.iter().enumerate().map(|(i, a)| {
        if neu.get(i).map(|n| gleich(n, a)).unwrap_or(false) { return Ok(i); }
        if let Some(j) = neu.iter().position(|n| gleich(n, a)) { return Ok(j); }
        if alt.iter().any(|b| gleich(b, &neu[i])) {
            return Err(format!("„{}“ ist nicht mehr in der Liste und an ihrer Stelle steht „{}“, das es schon gab. Bitte erst umbenennen, dann umsortieren.", a, neu[i]));
        }
        Ok(i)
    }).collect()
}

/// Zieht die Labels nach, wenn sich die Klassenliste geaendert hat. Boxen
/// verweisen ueber die Nummer, Klassifikation ueber den Namen.
pub(super) fn klassen_umschreiben(
    dir: &Path, boxen: bool, alt: &[String], neu: &[String],
) -> Result<(), String> {
    let abb = klassen_abbilden(alt, neu)?;
    let noetig = if boxen { abb.iter().enumerate().any(|(i, j)| i != *j) }
                 else { alt.iter().zip(&abb).any(|(a, j)| *a != neu[*j]) };
    if !noetig { return Ok(()); }
    let mut samples = load_samples(dir);
    for s in samples.iter_mut() {
        if boxen {
            for b in s.ann.boxes.iter_mut() {
                if let Some(j) = abb.get(b.cls) { b.cls = *j; }
            }
        } else if let Some(l) = s.ann.label.clone() {
            if let Some(i) = class_index_for(&l, alt) { s.ann.label = Some(neu[abb[i]].clone()); }
        }
    }
    basis_schreiben(dir, &samples)?;
    // Auch die Zweifel nennen Klassen beim Namen.
    let mut doubts = load_doubts(dir);
    if !doubts.is_empty() {
        for d in doubts.values_mut() {
            for k in d.missing.iter_mut().chain(d.extra.iter_mut()) {
                if let Some(i) = class_index_for(k, alt) { *k = neu[abb[i]].clone(); }
            }
        }
        save_doubts(dir, &doubts)?;
    }
    Ok(())
}

// ── Modellklassen zuordnen ──────────────────────────────────────────────────

/// Modellklasse -> Projektklasse. Ein Detektor meldet "pedestrian", das Projekt
/// fuehrt "person" — ohne Zuordnung kaeme kein einziger Vorschlag an.
pub(super) struct Zuordnung(HashMap<String, Option<String>>);

impl Zuordnung {
    pub fn neu(map: Option<HashMap<String, Option<String>>>) -> Self {
        Zuordnung(map.unwrap_or_default().into_iter()
            .map(|(k, v)| (k.trim().to_lowercase(), v.map(|v| v.trim().to_string()).filter(|v| !v.is_empty())))
            .collect())
    }
    /// Name im Projekt; `None`: ausdruecklich ignoriert.
    pub fn ziel(&self, modellklasse: &str) -> Option<String> {
        match self.0.get(&modellklasse.trim().to_lowercase()) {
            Some(z) => z.clone(),
            None => Some(modellklasse.trim().to_string()),
        }
    }
    /// Projektklassen, ueber die das Modell ueberhaupt etwas sagen kann.
    pub fn bekannte_klassen(&self, modell: &[String], projekt: &[String]) -> Vec<String> {
        let ziele: Vec<String> = modell.iter().filter_map(|m| self.ziel(m)).collect();
        projekt.iter().filter(|p| ziele.iter().any(|z| gleich(z, p))).cloned().collect()
    }
}

// ── Paare reparieren ────────────────────────────────────────────────────────

/// Bis 1.10.0 legte der Dateiimport den Zieltext eines Paares als Klasse ab.
/// Solche Projekte werden beim Oeffnen richtiggestellt, sonst fehlten die
/// Paare im Export. Rueckgabe: Zahl der geretteten Paare.
pub(super) fn paare_reparieren(dir: &Path, project: &mut StudioProject) -> Result<usize, String> {
    if !mit_zieltext(project) || project.classes.is_empty() { return Ok(0); }
    let mut samples = load_samples(dir);
    let mut n = 0;
    for s in samples.iter_mut().filter(|s| s.ann.label.is_some()) {
        let label = s.ann.label.take();
        // Nur was aus einer Datei kam, war ein Zieltext. Ein von Hand
        // mitgegebenes Label hat in einem Paar keine Bedeutung.
        if s.src.kind == "import" && s.ann.target.is_none() {
            s.ann.target = label;
            n += 1;
        } else if s.ann.target.is_none() && s.status == "confirmed" {
            s.status = "new".to_string();
        }
    }
    basis_schreiben(dir, &samples)?;
    project.classes.clear();
    save_project(dir, project)?;
    Ok(n)
}

// ── Export ──────────────────────────────────────────────────────────────────

/// Hat das Sample alles, was das Projekt fuer einen Trainingseintrag braucht?
pub(super) fn exportierbar(project: &StudioProject, s: &StudioSample) -> bool {
    let voll = |t: &Option<String>| t.as_deref().map(|t| !t.trim().is_empty()).unwrap_or(false);
    let medium = match project.modality.as_str() {
        "text" => voll(&s.content),
        m => !s.media.is_empty() && s.mime.starts_with(&format!("{}/", m)),
    };
    medium && (ist_boxenprojekt(project)
        || if mit_zieltext(project) { voll(&s.ann.target) } else { voll(&s.ann.label) })
}

/// Boxen, deren Klasse es nicht gibt, nimmt der Export nicht mit: eine
/// Klassennummer jenseits der Liste laesst Ultralytics das ganze Bild verwerfen.
pub(super) fn ohne_fremde_boxen(project: &StudioProject, samples: &[&StudioSample]) -> (Vec<StudioSample>, usize) {
    let mut weg = 0;
    let out = samples.iter().map(|s| {
        let mut s = (*s).clone();
        let vorher = s.ann.boxes.len();
        s.ann.boxes.retain(|b| b.cls < project.classes.len());
        weg += vorher - s.ann.boxes.len();
        s
    }).collect();
    (out, weg)
}

/// Ein eigener Ordner je Export. Zwei Exporte in derselben Sekunde teilten
/// sich vorher einen — der zweite Datensatz enthielt dann beide.
pub(super) fn export_ordner(dir: &Path) -> Result<PathBuf, String> {
    let exports = dir.join("exports");
    fs::create_dir_all(&exports).map_err(|e| format!("mkdir export: {}", e))?;
    let stamp = Utc::now().format("%Y%m%d_%H%M%S").to_string();
    for n in 1..10_000 {
        let out = exports.join(if n == 1 { stamp.clone() } else { format!("{}_{}", stamp, n) });
        match fs::create_dir(&out) {
            Ok(()) => return Ok(out),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(format!("mkdir export: {}", e)),
        }
    }
    Err("Kein freier Exportordner gefunden".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Ordner(PathBuf);
    impl Ordner {
        fn neu() -> Self {
            let p = std::env::temp_dir().join(format!("ft_regeln_{}", uuid::Uuid::new_v4()));
            fs::create_dir_all(&p).unwrap();
            Ordner(p)
        }
    }
    impl Drop for Ordner { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }

    fn liste(k: &[&str]) -> Vec<String> { k.iter().map(|s| s.to_string()).collect() }

    fn projekt(modality: &str, task: &str, klassen: &[&str]) -> StudioProject {
        StudioProject { id: "sp_r".into(), name: "P".into(), modality: modality.into(), task: task.into(),
            target_format: "x".into(), classes: liste(klassen), created_at: "x".into(), updated_at: "x".into() }
    }

    fn sample(id: &str, label: Option<&str>) -> StudioSample {
        StudioSample { id: id.into(), media: String::new(), mime: "text/plain".into(), content: Some(format!("Text {}", id)),
            status: "confirmed".into(), ann: Annotation { label: label.map(str::to_string), ..Default::default() },
            src: SampleSource { kind: "import".into(), origin: None, license: None, at: "x".into(), page: None },
            meta: SampleMeta::default(), abs_path: String::new(), doubt: None }
    }

    fn wunsch(status: &str, label: Option<&str>, keep: bool) -> Wunsch {
        Wunsch { status: status.into(), boxes: None, label: label.map(str::to_string), target: None, keep }
    }

    #[test]
    fn falsches_medium_wird_abgelehnt() {
        let p = projekt("image", "bbox", &["a"]);
        assert!(medium_pruefen(&p, "image").is_ok());
        let e = medium_pruefen(&p, "text").unwrap_err();
        assert!(e.contains("Bilder") && e.contains("Texte"), "{}", e);
    }

    #[test]
    fn ueberspringen_behaelt_das_label() {
        let d = Ordner::neu();
        append_jsonl(&samples_path(&d.0), &sample("s_1", Some("ham"))).unwrap();
        let p = projekt("text", "classification", &["spam", "ham"]);
        let ev = annotation_ereignis(&d.0, &p, "s_1", wunsch("skipped", None, true)).unwrap();
        assert_eq!(ev.label.as_deref(), Some("ham"));
        // Die Oberflaeche schickt immer den ganzen Stand: leer heisst dort leer.
        let ev = annotation_ereignis(&d.0, &p, "s_1", wunsch("new", None, false)).unwrap();
        assert_eq!(ev.label, None);
        // Ausdruecklich leer entfernt auch ueber MCP.
        let ev = annotation_ereignis(&d.0, &p, "s_1", wunsch("new", Some(""), true)).unwrap();
        assert_eq!(ev.label, None);
    }

    #[test]
    fn unbekanntes_sample_und_unbekannte_klasse_werden_gemeldet() {
        let d = Ordner::neu();
        append_jsonl(&samples_path(&d.0), &sample("s_1", None)).unwrap();
        let p = projekt("text", "classification", &["spam", "ham"]);
        assert!(annotation_ereignis(&d.0, &p, "s_weg", wunsch("new", None, true)).is_err());
        assert!(annotation_ereignis(&d.0, &p, "s_1", wunsch("confirmed", Some("phishing"), true)).is_err());
        assert!(annotation_ereignis(&d.0, &p, "s_1", wunsch("confirmed", None, true)).is_err(), "bestaetigt ohne Klasse");
        let ev = annotation_ereignis(&d.0, &p, "s_1", wunsch("confirmed", Some("HAM"), true)).unwrap();
        assert_eq!(ev.label.as_deref(), Some("ham"), "Schreibweise des Projekts");
    }

    #[test]
    fn boxen_werden_geprueft_und_beschnitten() {
        let d = Ordner::neu();
        append_jsonl(&samples_path(&d.0), &sample("s_1", None)).unwrap();
        let p = projekt("image", "bbox", &["a", "b"]);
        let mit = |b: BoxAnn| annotation_ereignis(&d.0, &p, "s_1",
            Wunsch { boxes: Some(vec![b]), ..wunsch("confirmed", None, true) });
        assert!(mit(BoxAnn { cls: 99, x: 0.5, y: 0.5, w: 0.2, h: 0.2 }).is_err());
        assert!(mit(BoxAnn { cls: 0, x: 320.0, y: 240.0, w: 100.0, h: 80.0 }).is_err(), "Pixel");
        assert!(mit(BoxAnn { cls: 0, x: -0.2, y: 0.5, w: 0.1, h: 0.1 }).is_err());
        let b = mit(BoxAnn { cls: 1, x: 0.95, y: 0.95, w: 0.5, h: 0.5 }).unwrap().boxes[0];
        assert!((b.x - 0.85).abs() < 1e-9 && (b.w - 0.3).abs() < 1e-9, "{:?}", b);
        // Kein Feld angegeben: die Boxen bleiben stehen.
        let mut s = sample("s_2", None);
        s.ann.boxes = vec![BoxAnn { cls: 0, x: 0.5, y: 0.5, w: 0.2, h: 0.2 }, BoxAnn { cls: 99, x: 0.5, y: 0.5, w: 0.2, h: 0.2 }];
        append_jsonl(&samples_path(&d.0), &s).unwrap();
        assert_eq!(annotation_ereignis(&d.0, &p, "s_2", wunsch("confirmed", None, true)).unwrap().boxes.len(), 2);
        // Eine alte, ungueltige Box blockiert die Werkbank nicht.
        assert!(annotation_ereignis(&d.0, &p, "s_2", Wunsch { boxes: Some(s.ann.boxes.clone()), ..wunsch("skipped", None, false) }).is_ok());
    }

    #[test]
    fn klassenliste_abbilden() {
        assert_eq!(klassen_abbilden(&liste(&["a", "b"]), &liste(&["a", "b", "c"])).unwrap(), vec![0, 1]);
        assert_eq!(klassen_abbilden(&liste(&["a", "b"]), &liste(&["x", "b"])).unwrap(), vec![0, 1], "umbenannt");
        assert_eq!(klassen_abbilden(&liste(&["a", "b"]), &liste(&["b", "a"])).unwrap(), vec![1, 0], "umsortiert");
        assert_eq!(klassen_abbilden(&liste(&["a", "b"]), &liste(&["A", "b"])).unwrap(), vec![0, 1], "Schreibweise");
        assert!(klassen_abbilden(&liste(&["a", "b"]), &liste(&["a"])).is_err(), "gekuerzt");
        assert!(klassen_abbilden(&liste(&["a", "b"]), &liste(&["a", "b", "B"])).is_err(), "doppelt");
        assert!(klassen_abbilden(&liste(&["a", "b"]), &liste(&["b", "c"])).is_err(), "mehrdeutig");
        // Alte Projekte mit doppelten Klassen duerfen weiter ergaenzt werden.
        assert!(klassen_abbilden(&liste(&["a", "A"]), &liste(&["a", "A", "c"])).is_ok());
        assert_eq!(klassen_eindeutig(liste(&["A", "a", " A ", "", "b"])), liste(&["A", "b"]));
    }

    #[test]
    fn umbenennen_und_umsortieren_zieht_die_labels_nach() {
        let d = Ordner::neu();
        append_jsonl(&samples_path(&d.0), &sample("s_1", Some("spam"))).unwrap();
        // Die letzte Aenderung steht nur als Ereignis da und darf nicht verloren gehen.
        append_jsonl(&samples_path(&d.0), &sample("s_2", None)).unwrap();
        append_jsonl(&events_path(&d.0), &AnnEvent { sample_id: "s_2".into(), status: "confirmed".into(), boxes: vec![],
            label: Some("ham".into()), target: None, confidence: None, at: "x".into() }).unwrap();
        klassen_umschreiben(&d.0, false, &liste(&["spam", "ham"]), &liste(&["werbung", "ham"])).unwrap();
        let s = load_samples(&d.0);
        assert_eq!(s[0].ann.label.as_deref(), Some("werbung"));
        assert_eq!(s[1].ann.label.as_deref(), Some("ham"));
        assert_eq!(s[1].status, "confirmed");

        let d = Ordner::neu();
        let mut b = sample("s_b", None);
        b.ann.boxes = vec![BoxAnn { cls: 0, x: 0.5, y: 0.5, w: 0.2, h: 0.2 }];
        append_jsonl(&samples_path(&d.0), &b).unwrap();
        klassen_umschreiben(&d.0, true, &liste(&["person", "auto"]), &liste(&["auto", "person"])).unwrap();
        assert_eq!(load_samples(&d.0)[0].ann.boxes[0].cls, 1, "die Box bleibt eine Person");
    }

    #[test]
    fn modellklassen_zuordnen() {
        let z = Zuordnung::neu(Some([("Pedestrian".to_string(), Some("person".to_string())), ("3-wheeler".to_string(), None)].into()));
        assert_eq!(z.ziel("pedestrian").as_deref(), Some("person"));
        assert_eq!(z.ziel("3-wheeler"), None);
        assert_eq!(z.ziel("car").as_deref(), Some("car"));
        assert_eq!(z.bekannte_klassen(&liste(&["pedestrian", "3-wheeler", "car"]), &liste(&["Person", "Car", "dog"])), liste(&["Person", "Car"]));
        assert!(Zuordnung::neu(None).bekannte_klassen(&liste(&["x"]), &liste(&["y"])).is_empty());
    }

    #[test]
    fn beschaedigte_paare_werden_gerettet() {
        let d = Ordner::neu();
        append_jsonl(&samples_path(&d.0), &sample("s_1", Some("Guten Morgen"))).unwrap();
        let mut hand = sample("s_2", Some("x"));
        hand.src.kind = "create".into();
        append_jsonl(&samples_path(&d.0), &hand).unwrap();
        let mut p = projekt("text", "pairs", &["Guten Morgen", "x"]);
        assert_eq!(paare_reparieren(&d.0, &mut p).unwrap(), 1);
        let s = load_samples(&d.0);
        assert_eq!((s[0].ann.target.as_deref(), s[0].ann.label.as_deref()), (Some("Guten Morgen"), None));
        assert_eq!((s[1].ann.target.as_deref(), s[1].status.as_str()), (None, "new"));
        assert!(p.classes.is_empty());
        assert_eq!(paare_reparieren(&d.0, &mut p).unwrap(), 0, "nur einmal");
        let mut k = projekt("text", "classification", &["a"]);
        assert_eq!(paare_reparieren(&d.0, &mut k).unwrap(), 0);
        assert_eq!(k.classes.len(), 1, "Klassenprojekte bleiben, wie sie sind");
    }

    #[test]
    fn nur_vollstaendiges_wird_exportiert() {
        let paare = projekt("text", "pairs", &[]);
        let mut s = sample("s_1", None);
        assert!(!exportierbar(&paare, &s));
        s.ann.target = Some("Ziel".into());
        assert!(exportierbar(&paare, &s));
        assert!(!exportierbar(&projekt("text", "classification", &["a"]), &s));
        // Ein Text in einem Bildprojekt (aeltere Projekte) kommt nicht mit.
        assert!(!exportierbar(&projekt("image", "bbox", &["a"]), &s));
        let mut b = sample("s_b", None);
        b.media = "ab/x.jpg".into(); b.mime = "image/jpeg".into(); b.content = None;
        assert!(exportierbar(&projekt("image", "bbox", &["a"]), &b), "ohne Box ist ein Negativbeispiel");
        b.ann.boxes = vec![BoxAnn { cls: 0, x: 0.5, y: 0.5, w: 0.1, h: 0.1 }, BoxAnn { cls: 9, x: 0.5, y: 0.5, w: 0.1, h: 0.1 }];
        let (sauber, weg) = ohne_fremde_boxen(&projekt("image", "bbox", &["a"]), &[&b]);
        assert_eq!((sauber[0].ann.boxes.len(), weg), (1, 1));
    }

    #[test]
    fn jeder_export_bekommt_einen_eigenen_ordner() {
        let d = Ordner::neu();
        let a = export_ordner(&d.0).unwrap();
        let b = export_ordner(&d.0).unwrap();
        let c = export_ordner(&d.0).unwrap();
        assert!(a != b && b != c && a != c);
        assert!(a.is_dir() && b.is_dir() && c.is_dir());
    }
}
