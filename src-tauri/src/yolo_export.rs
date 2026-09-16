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
        fs::copy(&item.source, images_dir.join(&file_name))
            .map_err(|e| format!("Bild kopieren: {}", e))?;

        // Ein Bild ohne Box bekommt eine leere Labeldatei, keine fehlende:
        // fehlt sie, gilt das Bild als unbeschriftet; ist sie leer, ist es
        // ein Negativbeispiel.
        let text: String = item.boxes.iter()
            .map(|b| label_line(b) + "\n")
            .collect();
        fs::write(labels_dir.join(format!("{}.txt", item.stem)), text)
            .map_err(|e| format!("Label schreiben: {}", e))?;
        written.push(file_name);
    }

    fs::write(dir.join("classes.txt"), classes.join("\n") + "\n")
        .map_err(|e| format!("classes.txt: {}", e))?;
    Ok(written)
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
    fn layout_schreibt_bild_label_und_klassen() {
        let dir = std::env::temp_dir().join(format!("ft_yolo_export_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let quelle = dir.join("quelle.png");
        fs::write(&quelle, vec![0u8; 8]).unwrap();

        let items = vec![
            ExportItem {
                source: quelle.clone(), stem: "a".to_string(),
                boxes: vec![norm_box(0, 0.5, 0.5, 0.2, 0.2).unwrap()],
            },
            ExportItem { source: quelle.clone(), stem: "b".to_string(), boxes: vec![] },
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
