// studio_manager/ablage.rs – Dateien und Ordner, die ins Fenster gezogen werden
//
// Ein Drop bringt nur Pfade. Ob ein Pfad ein Ordner ist und welche Datenart
// eine Datei hat, entscheidet sich hier und nicht am Namen in der Oberflaeche:
// "aufnahmen.v2" ist ein Ordner, und .webm kann Ton oder Bild sein.

use super::*;

#[derive(Debug, Clone, Serialize, Default, PartialEq)]
pub struct Ablage {
    pub folders: Vec<String>,
    pub images:  Vec<String>,
    pub audio:   Vec<String>,
    pub videos:  Vec<String>,
    pub texts:   Vec<String>,
    /// Was es gibt, womit das Studio aber nichts anfangen kann.
    pub other:   usize,
}

const TEXTQUELLEN: &[&str] = &["csv", "tsv", "jsonl", "ndjson", "txt", "md"];

fn endung(p: &Path) -> String {
    p.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase()
}

fn sortieren(paths: &[String]) -> Ablage {
    let mut a = Ablage::default();
    for s in paths {
        let p = Path::new(s);
        if p.is_dir() { a.folders.push(s.clone()); continue; }
        if !p.is_file() { a.other += 1; continue; }
        let e = endung(p);
        let mut bekannt = false;
        for (liste, exts) in [
            (&mut a.images, IMAGE_EXTS), (&mut a.audio, AUDIO_EXTS),
            (&mut a.videos, VIDEO_EXTS), (&mut a.texts, TEXTQUELLEN),
        ] {
            if exts.contains(&e.as_str()) { liste.push(s.clone()); bekannt = true; }
        }
        if !bekannt { a.other += 1; }
    }
    a
}

#[tauri::command]
pub async fn studio_sort_drop(paths: Vec<String>) -> Result<Ablage, String> {
    Ok(sortieren(&paths))
}

/// Die Dateien eines Imports: alle aus einem Ordner oder einzeln gewaehlte.
/// Der zweite Wert sagt, ob es einzelne sind: dann ist ihr Ordner nur zufaellig
/// der, in dem sie liegen, und weder Klasse noch Wurzel einer Labelstruktur.
pub(super) fn quellen(
    source_path: &str, files: Option<Vec<String>>, exts: &[&str], sammeln: fn(&Path) -> Vec<PathBuf>,
) -> Result<(Vec<PathBuf>, bool), String> {
    match files.filter(|f| !f.is_empty()) {
        Some(liste) => Ok((liste.into_iter().map(PathBuf::from)
            .filter(|p| p.is_file() && exts.contains(&endung(p).as_str())).collect(), true)),
        None => {
            let src = Path::new(source_path);
            if !src.is_dir() { return Err(format!("Ordner nicht gefunden: {}", source_path)); }
            Ok((sammeln(src), false))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ordner() -> PathBuf {
        let d = std::env::temp_dir().join(format!("ft_ablage_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(d.join("aufnahmen.v2")).unwrap();
        for n in ["a.PNG", "b.wav", "c.webm", "d.csv", "e.pdf", "aufnahmen.v2/f.jpg"] {
            fs::write(d.join(n), b"x").unwrap();
        }
        d
    }

    #[test]
    fn sortiert_nach_datenart() {
        let d = ordner();
        let s = |n: &str| d.join(n).to_string_lossy().to_string();
        let a = sortieren(&["a.PNG", "b.wav", "c.webm", "d.csv", "e.pdf", "aufnahmen.v2", "fehlt.png"].map(s));
        assert_eq!(a, Ablage {
            folders: vec![s("aufnahmen.v2")], images: vec![s("a.PNG")],
            audio: vec![s("b.wav"), s("c.webm")], videos: vec![s("c.webm")],
            texts: vec![s("d.csv")], other: 2,
        });
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn einzelne_dateien_oder_ganzer_ordner() {
        let d = ordner();
        let s = |n: &str| d.join(n).to_string_lossy().to_string();
        let (f, lose) = quellen("", Some(vec![s("a.PNG"), s("b.wav"), s("fehlt.png")]), IMAGE_EXTS, collect_images).unwrap();
        assert_eq!((f, lose), (vec![d.join("a.PNG")], true));
        let (f, lose) = quellen(&d.to_string_lossy(), None, IMAGE_EXTS, collect_images).unwrap();
        assert_eq!((f.len(), lose), (2, false));
        assert!(quellen("/gibt/es/nicht", None, IMAGE_EXTS, collect_images).is_err());
        fs::remove_dir_all(&d).ok();
    }
}
