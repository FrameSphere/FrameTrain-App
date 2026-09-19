// yolo_export.rs – der eine Ort, an dem FrameTrain ein YOLO-Dataset schreibt.
//
// Es gab zwei: das Labor exportiert Korrekturen (Boxen in Bildpixeln), das
// Studio exportiert Projekte (Boxen bereits normiert). Beide legten images/,
// labels/ und classes.txt an, beide formatierten dieselbe Zeile, und beide
// haetten bei einer Aenderung am Format einzeln nachgezogen werden muessen.
//
// Geteilt ist alles, was das Format ausmacht. Nicht geteilt ist, was die
// Aufrufer unterscheidet: welche Dateien wie heissen, was uebersprungen wird
// und welche Beipackzettel daneben liegen.

use std::fs;
use std::path::{Path, PathBuf};

/// Eine Box in YOLO-Schreibweise: Mittelpunkt und Groesse, normiert auf 0..1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormBox {
    pub cls: usize,
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

/// Box aus Pixelkanten.
///
/// Rueckwaerts gezogene Rechtecke werden gedreht, ueber den Bildrand
/// hinausgezogene Ecken beschnitten — beim Zeichnen rutscht die Maus schnell
/// aus dem Bild, und Ultralytics verlangt Werte zwischen 0 und 1.
/// `None` bei entarteter Box oder unbekanntem Bildmass: eine Box ohne Flaeche
/// ist kein Objekt, sondern ein Fehlklick.
pub fn normalize_box(
    cls: usize, x1: f64, y1: f64, x2: f64, y2: f64, width: f64, height: f64,
) -> Option<NormBox> {
    if width <= 0.0 || height <= 0.0 { return None; }
    let left   = x1.min(x2).max(0.0);
    let right  = x1.max(x2).min(width);
    let top    = y1.min(y2).max(0.0);
    let bottom = y1.max(y2).min(height);
    let (w, h) = (right - left, bottom - top);
    if w <= 0.0 || h <= 0.0 { return None; }
    Some(NormBox {
        cls,
        x: (left + w / 2.0) / width,
        y: (top + h / 2.0) / height,
        w: w / width,
        h: h / height,
    })
}

/// Box, die schon normiert vorliegt.
///
/// Auch hier wird beschnitten: wer Mittelpunkt und Groesse getrennt auf 0..1
/// begrenzt, kann eine Box bekommen, die rechnerisch ueber den Bildrand
/// hinausragt (Mittelpunkt 0.9 bei Breite 0.4). Deshalb derselbe Weg ueber
/// die Kanten wie bei Pixelwerten, nur in einem Bild der Groesse 1 x 1.
pub fn norm_box(cls: usize, x: f64, y: f64, w: f64, h: f64) -> Option<NormBox> {
    normalize_box(cls, x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0, 1.0, 1.0)
}

/// Eine Zeile einer YOLO-Labeldatei: "cls cx cy w h".
pub fn label_line(b: &NormBox) -> String {
    format!("{} {:.6} {:.6} {:.6} {:.6}", b.cls, b.x, b.y, b.w, b.h)
}

/// Ein Bild mit seinen Boxen, so wie es im Export landen soll.
pub struct ExportItem {
    /// Bild, das kopiert wird. Der Aufrufer stellt sicher, dass es existiert.
    pub source: PathBuf,
    /// Dateiname ohne Endung im Export. Die Aufrufer vergeben ihn
    /// unterschiedlich (laufende Nummer im Labor, Sample-ID im Studio) — er
    /// muss nur innerhalb eines Exports eindeutig sein.
    pub stem: String,
    pub boxes: Vec<NormBox>,
    /// "train" | "val" | "test", oder None fuer ein ungeteiltes Dataset.
    /// Mit Split landet das Bild in images/<split>/ statt in images/ —
    /// das Layout, das der Dataset-Import als fertig aufgeteilt erkennt.
    pub split: Option<String>,
}

/// Schreibt images/, labels/ und classes.txt.
///
/// Rueckgabe sind die geschriebenen Bilddateinamen in der Reihenfolge der
/// Eingabe — die Aufrufer brauchen sie zum Zaehlen und fuer ihre eigenen
/// Beipackzettel.
///
/// Das Layout ist der Vertrag mit `dataset_manager::detect_dataset_type`:
/// images/ + labels/ + classes.txt wird als `yolo_bbox` erkannt, und die
/// Namen aus classes.txt landen in der generierten dataset.yaml statt
/// Platzhaltern.
pub fn write_yolo_layout(
    dir: &Path, items: &[ExportItem], classes: &[String],
) -> Result<Vec<String>, String> {
    let images_dir = dir.join("images");
    let labels_dir = dir.join("labels");
    fs::create_dir_all(&images_dir).map_err(|e| format!("images/: {}", e))?;
    fs::create_dir_all(&labels_dir).map_err(|e| format!("labels/: {}", e))?;

    let mut written = Vec::with_capacity(items.len());
    for item in items {
        let ext = item.source.extension()
            .map(|e| e.to_string_lossy().to_string())
            .unwrap_or_else(|| "jpg".to_string());
        let file_name = format!("{}.{}", item.stem, ext);
        let (img_ziel, lbl_ziel) = match &item.split {
            Some(split) => {
                let i = images_dir.join(split);
                let l = labels_dir.join(split);
                fs::create_dir_all(&i).map_err(|e| format!("images/{}: {}", split, e))?;
                fs::create_dir_all(&l).map_err(|e| format!("labels/{}: {}", split, e))?;
                (i, l)
            }
            None => (images_dir.clone(), labels_dir.clone()),
        };
        fs::copy(&item.source, img_ziel.join(&file_name))
            .map_err(|e| format!("Bild kopieren: {}", e))?;

        // Ein Bild ohne Box bekommt eine leere Labeldatei, keine fehlende:
        // fehlt sie, gilt das Bild als unbeschriftet; ist sie leer, ist es
        // ein Negativbeispiel.
        let text: String = item.boxes.iter()
            .map(|b| label_line(b) + "\n")
            .collect();
        fs::write(lbl_ziel.join(format!("{}.txt", item.stem)), text)
            .map_err(|e| format!("Label schreiben: {}", e))?;
        written.push(file_name);
    }

    fs::write(dir.join("classes.txt"), classes.join("\n") + "\n")
        .map_err(|e| format!("classes.txt: {}", e))?;
    Ok(written)
}

/// Teilt Samples auf train, val und test auf, ohne eine Gruppe zu zerreissen.
///
/// Eine Gruppe ist, was zusammengehoert: Einzelbilder eines Videos, Aufnahmen
/// einer Session, Bilder derselben Quelle. Landen Bilder einer Gruppe in Train
/// und gleichzeitig in Val, prueft das Training gegen fast dieselben Bilder,
/// mit denen es gelernt hat — die Validierung sieht grossartig aus und sagt
/// nichts. Das ist die haeufigste stille Ursache fuer zu gute Werte.
///
/// Die Reihenfolge ist deterministisch: derselbe Datenbestand ergibt dieselbe
/// Aufteilung, sonst waeren zwei Exporte nicht vergleichbar.
pub fn assign_splits(group_of: &[String], train_ratio: f64, val_ratio: f64) -> Vec<String> {
    use std::collections::HashMap;

    // Gruppen in stabiler, aber nicht alphabetischer Reihenfolge: nach Namen
    // sortiert landeten sonst alle "video_01"-Bilder immer im selben Split.
    let mut reihenfolge: Vec<(u64, &String)> = Vec::new();
    let mut gesehen: HashMap<&String, usize> = HashMap::new();
    for g in group_of {
        if let Some(n) = gesehen.get_mut(g) { *n += 1; continue; }
        gesehen.insert(g, 1);
        reihenfolge.push((stabiler_hash(g), g));
    }
    reihenfolge.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(b.1)));

    let gesamt = group_of.len() as f64;
    let ziel_train = gesamt * train_ratio;
    let ziel_val = gesamt * (train_ratio + val_ratio);

    let mut split_der_gruppe: HashMap<&String, &str> = HashMap::new();
    let mut vergeben = 0f64;
    for (_, g) in &reihenfolge {
        let n = *gesehen.get(g).unwrap_or(&0) as f64;
        // Die Gruppe geht dorthin, wo ihre Mitte landet — so kippt eine grosse
        // Gruppe die Quote nicht komplett in den naechsten Split.
        let mitte = vergeben + n / 2.0;
        let split = if mitte < ziel_train { "train" }
            else if mitte < ziel_val { "val" }
            else { "test" };
        split_der_gruppe.insert(g, split);
        vergeben += n;
    }

    group_of.iter()
        .map(|g| split_der_gruppe.get(g).copied().unwrap_or("train").to_string())
        .collect()
}

/// FNV-1a. Nicht kryptographisch, aber ueber Laeufe und Plattformen gleich —
/// anders als DefaultHasher, dessen Ergebnis nicht garantiert stabil ist.
fn stabiler_hash(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixelbox_wird_normiert() {
        let b = normalize_box(0, 100.0, 100.0, 300.0, 200.0, 400.0, 400.0).unwrap();
        assert_eq!(label_line(&b), "0 0.500000 0.375000 0.500000 0.250000");
    }

    #[test]
    fn rueckwaerts_gezogene_box_wird_gedreht() {
        let a = normalize_box(1, 300.0, 200.0, 100.0, 100.0, 400.0, 400.0).unwrap();
        let b = normalize_box(1, 100.0, 100.0, 300.0, 200.0, 400.0, 400.0).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn ueber_den_rand_gezogene_box_wird_beschnitten() {
        let b = normalize_box(0, -50.0, -50.0, 200.0, 200.0, 400.0, 400.0).unwrap();
        assert_eq!(label_line(&b), "0 0.250000 0.250000 0.500000 0.500000");
    }

    #[test]
    fn entartete_box_und_unbekanntes_bildmass_liefern_nichts() {
        assert!(normalize_box(0, 10.0, 10.0, 10.0, 60.0, 400.0, 400.0).is_none());
        assert!(normalize_box(0, 10.0, 10.0, 60.0, 60.0, 0.0, 0.0).is_none());
    }

    #[test]
    fn schon_normierte_box_wird_ebenfalls_beschnitten() {
        // Mittelpunkt und Groesse getrennt begrenzt ergeben eine Box, die
        // ueber den rechten Rand hinausragt. Die Kanten bringen sie zurueck.
        let b = norm_box(2, 0.5, 0.25, 1.5, 0.1).unwrap();
        assert_eq!(label_line(&b), "2 0.500000 0.250000 1.000000 0.100000");

        let am_rand = norm_box(0, 0.9, 0.5, 0.4, 0.2).unwrap();
        assert!((am_rand.x + am_rand.w / 2.0 - 1.0).abs() < 1e-12,
            "Box ragt weiterhin ueber den Rand: {:?}", am_rand);
    }

    #[test]
    fn eine_gruppe_landet_nie_in_zwei_splits() {
        // Einzelbilder aus drei Videos. Wuerden sie einzeln verteilt, saehe die
        // Validierung fast dieselben Bilder wie das Training.
        let mut gruppen: Vec<String> = Vec::new();
        for video in ["video_a", "video_b", "video_c"] {
            for _ in 0..30 { gruppen.push(video.to_string()); }
        }
        let splits = assign_splits(&gruppen, 0.7, 0.2);

        use std::collections::{HashMap, HashSet};
        let mut je_gruppe: HashMap<&str, HashSet<&str>> = HashMap::new();
        for (g, sp) in gruppen.iter().zip(splits.iter()) {
            je_gruppe.entry(g).or_default().insert(sp);
        }
        for (g, s) in &je_gruppe {
            assert_eq!(s.len(), 1, "Gruppe {} verteilt auf {:?}", g, s);
        }
    }

    #[test]
    fn aufteilung_haelt_die_quote_ungefaehr_ein() {
        // Ohne Gruppen ist jede Aufnahme fuer sich — dann muss die Quote passen.
        let gruppen: Vec<String> = (0..100).map(|i| format!("s_{}", i)).collect();
        let splits = assign_splits(&gruppen, 0.7, 0.2);
        let zaehle = |name: &str| splits.iter().filter(|s| s.as_str() == name).count();
        assert!((zaehle("train") as i32 - 70).abs() <= 2, "train: {}", zaehle("train"));
        assert!((zaehle("val") as i32 - 20).abs() <= 2, "val: {}", zaehle("val"));
        assert!((zaehle("test") as i32 - 10).abs() <= 2, "test: {}", zaehle("test"));
    }

    #[test]
    fn aufteilung_ist_bei_gleichen_daten_gleich() {
        let gruppen: Vec<String> = (0..40).map(|i| format!("g_{}", i % 7)).collect();
        assert_eq!(assign_splits(&gruppen, 0.7, 0.2), assign_splits(&gruppen, 0.7, 0.2));
    }

    #[test]
    fn split_schreibt_in_unterordner() {
        let dir = std::env::temp_dir().join(format!("ft_yolo_split_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let quelle = dir.join("q.png");
        fs::write(&quelle, vec![0u8; 8]).unwrap();
        let items = vec![
            ExportItem { source: quelle.clone(), stem: "a".into(), boxes: vec![], split: Some("train".into()) },
            ExportItem { source: quelle.clone(), stem: "b".into(), boxes: vec![], split: Some("val".into()) },
        ];
        let out = dir.join("out");
        write_yolo_layout(&out, &items, &["Lift".to_string()]).unwrap();

        assert!(out.join("images/train/a.png").exists());
        assert!(out.join("labels/train/a.txt").exists());
        assert!(out.join("images/val/b.png").exists());

        // Das Layout muss als fertig aufgeteiltes YOLO durchgehen.
        let analysis = crate::dataset_manager::detect_dataset_type(&out);
        assert_eq!(analysis.detected_type.as_str(), "yolo_bbox");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn layout_schreibt_bild_label_und_klassen() {
        let dir = std::env::temp_dir().join(format!("ft_yolo_export_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let quelle = dir.join("quelle.png");
        fs::write(&quelle, vec![0u8; 8]).unwrap();

        let items = vec![
            ExportItem {
                source: quelle.clone(), stem: "a".to_string(),
                boxes: vec![norm_box(0, 0.5, 0.5, 0.2, 0.2).unwrap()], split: None,
            },
            ExportItem { source: quelle.clone(), stem: "b".to_string(), boxes: vec![], split: None },
        ];
        let out = dir.join("out");
        let names = write_yolo_layout(&out, &items, &["Lift".to_string(), "Sky".to_string()]).unwrap();

        assert_eq!(names, vec!["a.png", "b.png"]);
        assert_eq!(fs::read_to_string(out.join("labels/a.txt")).unwrap(),
            "0 0.500000 0.500000 0.200000 0.200000\n");
        assert_eq!(fs::read_to_string(out.join("labels/b.txt")).unwrap(), "");
        assert_eq!(fs::read_to_string(out.join("classes.txt")).unwrap(), "Lift\nSky\n");
        assert!(out.join("images/a.png").exists());

        // Der Vertrag mit dem Import: das Ergebnis muss als YOLO durchgehen.
        let analysis = crate::dataset_manager::detect_dataset_type(&out);
        assert_eq!(analysis.detected_type.as_str(), "yolo_bbox");

        let _ = fs::remove_dir_all(&dir);
    }
}
