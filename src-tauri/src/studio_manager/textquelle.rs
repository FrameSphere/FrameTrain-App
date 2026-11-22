// studio_manager/textquelle.rs – Textdateien lesen, wie sie wirklich aussehen
//
// Eine "CSV" aus der Praxis ist selten eine nach Lehrbuch: deutsches Excel
// trennt mit Semikolon, aeltere Exporte sind in Windows-1252 statt UTF-8, und
// die Spalten heissen "v1" und "v2". Vorher wurde so eine Datei ohne Meldung
// falsch gelesen (Testlauf 22.11.: von 58 SMS blieben 3 uebrig — die
// Klassennamen). Deshalb werden Zeichensatz, Trennzeichen, Kopfzeile und
// Spalten aus dem Inhalt erkannt; was ausdruecklich angegeben ist, geht vor.

use super::*;

/// Was beim Lesen herauskam, samt der Entscheidungen.
pub(super) struct Quelle {
    pub kind:    String,
    pub spalten: Vec<String>,
    /// Spalte, aus der der Text kam (leer bei Ordnern).
    pub text_spalte:  Option<String>,
    pub label_spalte: Option<String>,
    pub rows:    Vec<TextRow>,
    /// Die ersten Zeilen mit allen Spalten — zum Ansehen.
    pub proben:  Vec<Vec<String>>,
}

/// Spaltennamen, die nach dem Text aussehen. Paare bringen eigene mit.
pub(super) fn text_spalten(paare: bool) -> &'static [&'static str] {
    if paare { &["text", "source", "input", "sentence", "prompt", "question", "instruction"] }
    else { &["text", "source", "input", "sentence", "content", "message", "review"] }
}

/// Spaltennamen fuer die Klasse bzw. — bei Paaren — fuer den Zieltext.
pub(super) fn label_spalten(paare: bool) -> &'static [&'static str] {
    if paare { &["target", "output", "answer", "response", "completion", "label"] }
    else { &["label", "target", "class", "klasse", "category", "kategorie", "sentiment", "intent"] }
}

// ── Zeichensatz ─────────────────────────────────────────────────────────────

/// Windows-1252 fuer 0x80–0x9F; der Rest deckt sich mit Latin-1.
const CP1252: [char; 32] = [
    '€', '\u{81}', '‚', 'ƒ', '„', '…', '†', '‡', 'ˆ', '‰', 'Š', '‹', 'Œ', '\u{8d}', 'Ž', '\u{8f}',
    '\u{90}', '‘', '’', '“', '”', '•', '–', '—', '˜', '™', 'š', '›', 'œ', '\u{9d}', 'ž', 'Ÿ'];

/// Bytes zu Text: UTF-8 (mit oder ohne BOM), UTF-16 mit BOM, sonst Windows-1252.
/// Letzteres geht immer auf — jedes Byte ist dort ein Zeichen.
pub fn als_text(bytes: &[u8]) -> String {
    let utf16 = |rest: &[u8], lese: fn([u8; 2]) -> u16| {
        let werte: Vec<u16> = rest.chunks_exact(2).map(|c| lese([c[0], c[1]])).collect();
        String::from_utf16_lossy(&werte)
    };
    if let Some(r) = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]) { return String::from_utf8_lossy(r).into_owned(); }
    if let Some(r) = bytes.strip_prefix(&[0xFF, 0xFE]) { return utf16(r, u16::from_le_bytes); }
    if let Some(r) = bytes.strip_prefix(&[0xFE, 0xFF]) { return utf16(r, u16::from_be_bytes); }
    match std::str::from_utf8(bytes) {
        Ok(t) => t.to_string(),
        Err(_) => bytes.iter().map(|b| match b {
            0x80..=0x9F => CP1252[(*b - 0x80) as usize],
            _ => *b as char,
        }).collect(),
    }
}

// ── Tabelle ─────────────────────────────────────────────────────────────────

/// Trennzeichen aus dem Anfang der Datei. Gewinnt, was die meisten Zeilen in
/// gleich viele (mindestens zwei) Spalten teilt.
pub fn trennzeichen(text: &str) -> Option<char> {
    let mut anfang = 0;
    for (i, _) in text.match_indices('\n').take(60) { anfang = i + 1; }
    let probe = if text.matches('\n').count() > 60 { &text[..anfang] } else { text };
    let mut bestes: Option<(char, f64, usize)> = None;
    for z in [',', ';', '\t', '|'] {
        let t = parse_csv_mit(probe, z);
        if t.is_empty() { continue; }
        let mut zahl: HashMap<usize, usize> = HashMap::new();
        for zeile in &t { *zahl.entry(zeile.len()).or_default() += 1; }
        let Some((spalten, n)) = zahl.into_iter().max_by_key(|(s, n)| (*n, *s)) else { continue };
        if spalten < 2 { continue; }
        let anteil = n as f64 / t.len() as f64;
        // Einheitlichkeit zaehlt; bei Gleichstand bleibt es beim frueheren
        // Zeichen — das Komma ist der Normalfall.
        if bestes.map(|(_, a, _)| anteil > a + 1e-9).unwrap_or(true) { bestes = Some((z, anteil, spalten)); }
    }
    bestes.filter(|(_, a, _)| *a >= 0.8).map(|(z, _, _)| z)
}

struct Spalte { verschieden: usize, gefuellt: usize, laenge: f64 }

fn spalten_profil(daten: &[Vec<String>], n: usize) -> Vec<Spalte> {
    (0..n).map(|i| {
        let werte: Vec<&str> = daten.iter().filter_map(|z| z.get(i)).map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
        let mut v: Vec<String> = werte.iter().map(|s| s.to_lowercase()).collect();
        v.sort(); v.dedup();
        let laenge = if werte.is_empty() { 0.0 } else { werte.iter().map(|s| s.chars().count()).sum::<usize>() as f64 / werte.len() as f64 };
        Spalte { verschieden: v.len(), gefuellt: werte.len(), laenge }
    }).collect()
}

/// Sieht die Spalte nach Klassen aus: wenige kurze Werte, die sich wiederholen?
fn klassenartig(s: &Spalte) -> bool {
    s.gefuellt >= 4 && s.verschieden >= 2 && s.verschieden <= 100
        && s.verschieden * 2 <= s.gefuellt && s.laenge <= 40.0
}

/// Ist die erste Zeile eine Kopfzeile? Bekannte Namen entscheiden sofort. Sonst
/// hilft die Klassenspalte: steht ihr erster Wert ("v1") nie wieder darunter,
/// ist er ein Name und keine Klasse.
pub fn hat_kopfzeile(tabelle: &[Vec<String>], paare: bool) -> bool {
    let Some(kopf) = tabelle.first() else { return false };
    let bekannt = |f: &String| {
        let f = f.trim().to_lowercase();
        text_spalten(paare).contains(&f.as_str()) || label_spalten(paare).contains(&f.as_str())
    };
    if kopf.iter().any(bekannt) { return true; }
    let daten = &tabelle[1..];
    if kopf.len() < 2 || daten.len() < 4 { return false; }
    let namen: Vec<String> = kopf.iter().map(|f| f.trim().to_lowercase()).collect();
    let mut eindeutig = namen.clone(); eindeutig.sort(); eindeutig.dedup();
    let wie_namen = eindeutig.len() == namen.len() && namen.iter().all(|f|
        !f.is_empty() && f.chars().count() <= 30 && f.split_whitespace().count() <= 3 && f.parse::<f64>().is_err());
    if !wie_namen { return false; }
    let profil = spalten_profil(daten, kopf.len());
    let mut klassen = 0;
    for (i, p) in profil.iter().enumerate().filter(|(_, p)| klassenartig(p)) {
        let _ = p;
        if daten.iter().filter_map(|z| z.get(i)).any(|v| v.trim().to_lowercase() == namen[i]) { return false; }
        klassen += 1;
    }
    klassen > 0
}

fn finde(spalten: &[String], namen: &[&str]) -> Option<usize> {
    namen.iter().find_map(|k| spalten.iter().position(|s| s.trim().eq_ignore_ascii_case(k)))
}

/// Ausdruecklich genannte Spalte. Ein Tippfehler soll nicht still die erste
/// Spalte nehmen, sondern sagen, welche es gibt.
fn genannt(spalten: &[String], name: Option<&str>) -> Result<Option<usize>, String> {
    let Some(n) = name.map(str::trim).filter(|n| !n.is_empty()) else { return Ok(None) };
    spalten.iter().position(|s| s.trim() == n).or_else(|| spalten.iter().position(|s| s.trim().eq_ignore_ascii_case(n)))
        .map(Some)
        .ok_or_else(|| format!("Die Spalte „{}“ gibt es nicht. Vorhanden: {}.", n, spalten.join(", ")))
}

/// Text- und Klassenspalte: genannt, am Namen erkannt, sonst am Inhalt.
fn spalten_waehlen(
    spalten: &[String], daten: &[Vec<String>], text: Option<&str>, label: Option<&str>, paare: bool,
) -> Result<(usize, Option<usize>), String> {
    let t_genannt = genannt(spalten, text)?;
    let l_genannt = genannt(spalten, label)?;
    let profil = spalten_profil(daten, spalten.len());
    let t = t_genannt
        .or_else(|| finde(spalten, text_spalten(paare)).filter(|i| Some(*i) != l_genannt))
        .unwrap_or_else(|| {
            // Der laengste Inhalt ist der Text; die Klasse ist kurz.
            (0..spalten.len()).filter(|i| Some(*i) != l_genannt)
                .fold(None::<usize>, |b, i| match b {
                    Some(j) if profil[j].laenge >= profil[i].laenge => Some(j),
                    _ => Some(i),
                }).unwrap_or(0)
        });
    let l = l_genannt
        .or_else(|| finde(spalten, label_spalten(paare)).filter(|i| *i != t))
        .or_else(|| {
            if paare { return if spalten.len() == 2 { Some(1 - t) } else { None }; }
            // Nur wenn genau eine Spalte in Frage kommt: raten waere schlimmer
            // als offen lassen.
            let k: Vec<usize> = (0..spalten.len()).filter(|i| *i != t && klassenartig(&profil[*i])).collect();
            if k.len() == 1 { Some(k[0]) } else { None }
        });
    Ok((t, l))
}

/// Klasse oder Zieltext aus einem JSON-Feld. Zahlen und Wahrheitswerte zaehlen
/// mit: `"label": 0` ist die haeufigste Schreibweise ueberhaupt.
fn feld_als_text(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::String(t) => Some(t.clone()).filter(|t| !t.trim().is_empty()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

fn aus_tabelle(
    kind: &str, spalten: Vec<String>, daten: Vec<Vec<String>>, herkunft: &str,
    text: Option<&str>, label: Option<&str>, paare: bool,
) -> Result<Quelle, String> {
    let (ti, li) = spalten_waehlen(&spalten, &daten, text, label, paare)?;
    let rows = daten.iter().filter_map(|z| {
        let t = z.get(ti)?.trim().to_string();
        if t.is_empty() { return None; }
        let l = li.and_then(|i| z.get(i)).map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
        Some(TextRow { text: t, label: l, origin: herkunft.to_string() })
    }).collect();
    Ok(Quelle {
        kind: kind.to_string(),
        text_spalte: spalten.get(ti).cloned(),
        label_spalte: li.and_then(|i| spalten.get(i).cloned()),
        proben: daten.iter().take(3).map(|z| z.iter().map(|f| f.chars().take(140).collect()).collect()).collect(),
        spalten, rows,
    })
}

fn aus_ordner(path: &Path) -> Quelle {
    // Jede .txt ist ein Sample. Liegt sie in einem Unterordner, ist dessen Name
    // das Label — dieselbe Konvention wie bei Bildern.
    let mut rows = Vec::new();
    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                if !p.file_name().and_then(|n| n.to_str()).unwrap_or("").starts_with('.') { stack.push(p); }
                continue;
            }
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
            if !TEXT_EXTS.contains(&ext.as_str()) { continue; }
            let Ok(bytes) = fs::read(&p) else { continue };
            let text = als_text(&bytes);
            if text.trim().is_empty() { continue; }
            rows.push(TextRow { text, label: klasse_aus_ordner(&p, path), origin: p.to_string_lossy().to_string() });
        }
    }
    rows.sort_by(|a, b| a.origin.cmp(&b.origin));
    Quelle { kind: "folder".to_string(), spalten: vec![], text_spalte: None, label_spalte: None, proben: vec![], rows }
}

/// Liest eine Textquelle: CSV/TSV, JSONL, TXT oder einen Ordner mit .txt-Dateien.
pub(super) fn lesen(path: &Path, text: Option<&str>, label: Option<&str>, paare: bool) -> Result<Quelle, String> {
    if path.is_dir() { return Ok(aus_ordner(path)); }
    let inhalt = als_text(&fs::read(path).map_err(|e| format!("Datei lesen: {}", e))?);
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    let herkunft = path.to_string_lossy().to_string();

    if ext == "jsonl" || ext == "ndjson" {
        let mut spalten: Vec<String> = Vec::new();
        let mut werte: Vec<serde_json::Map<String, serde_json::Value>> = Vec::new();
        for line in inhalt.lines().filter(|l| !l.trim().is_empty()) {
            let Ok(serde_json::Value::Object(obj)) = serde_json::from_str(line) else { continue };
            for k in obj.keys() { if !spalten.contains(k) { spalten.push(k.clone()); } }
            werte.push(obj);
        }
        if spalten.is_empty() { return Err("Die Datei enthält keine lesbaren Felder".to_string()); }
        let daten = werte.iter().map(|o| spalten.iter()
            .map(|k| o.get(k).and_then(feld_als_text).unwrap_or_default()).collect()).collect();
        return aus_tabelle("jsonl", spalten, daten, &herkunft, text, label, paare);
    }

    // Eine .txt ist eine Liste von Texten, eine Zeile je Sample — auch wenn
    // Kommas darin stehen. Nur mit erkennbarer Kopfzeile ist sie eine Tabelle.
    let zeichen = if ext == "tsv" { Some('\t') } else { trennzeichen(&inhalt) };
    let mut tabelle = match zeichen {
        Some(z) => parse_csv_mit(&inhalt, z),
        None => parse_csv_mit(&inhalt, '\0'),
    };
    if ext == "txt" && !(zeichen.is_some() && tabelle.first().map(|k| k.iter().any(|f| {
        let f = f.trim().to_lowercase();
        text_spalten(paare).contains(&f.as_str()) || label_spalten(paare).contains(&f.as_str())
    })).unwrap_or(false)) {
        tabelle = inhalt.lines().filter(|l| !l.trim().is_empty()).map(|l| vec![l.to_string()]).collect();
    }
    if tabelle.is_empty() { return Err("Die Datei ist leer".to_string()); }

    let kopf = hat_kopfzeile(&tabelle, paare);
    let breite = tabelle.iter().map(Vec::len).max().unwrap_or(1);
    let mut spalten: Vec<String> = if kopf { tabelle.remove(0) } else { vec![] };
    for i in spalten.len()..breite { spalten.push(format!("Spalte {}", i + 1)); }
    aus_tabelle("csv", spalten, tabelle, &herkunft, text, label, paare)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Datei(PathBuf);
    impl Datei {
        fn neu(name: &str, bytes: &[u8]) -> Self {
            let d = std::env::temp_dir().join(format!("ft_tq_{}", uuid::Uuid::new_v4()));
            fs::create_dir_all(&d).unwrap();
            let p = d.join(name);
            fs::write(&p, bytes).unwrap();
            Datei(p)
        }
        fn lies(&self, paare: bool) -> Quelle { lesen(&self.0, None, None, paare).unwrap() }
    }
    impl Drop for Datei { fn drop(&mut self) { let _ = fs::remove_dir_all(self.0.parent().unwrap()); } }

    const SMS: [(&str, &str); 6] = [("ham", "Hi, wie geht es dir heute?"), ("spam", "Gewinne jetzt, ruf an!"),
        ("ham", "Bis morgen, dann um acht"), ("ham", "Alles klar"), ("spam", "Gratis; nur heute"), ("ham", "Ok")];

    fn tabelle(kopf: &str, z: &str) -> String {
        let mut t = if kopf.is_empty() { String::new() } else { format!("{}\n", kopf) };
        for (l, s) in SMS { t.push_str(&format!("{}{}\"{}\"\n", l, z, s)); }
        t
    }

    #[test]
    fn unbekannte_kopfzeile_und_spalten_werden_erkannt() {
        // Die UCI-SMS-Sammlung als CSV: v1 = Klasse, v2 = Text.
        let q = Datei::neu("a.csv", tabelle("v1,v2", ",").as_bytes()).lies(false);
        assert_eq!(q.rows.len(), 6);
        assert_eq!((q.text_spalte.as_deref(), q.label_spalte.as_deref()), (Some("v2"), Some("v1")));
        assert_eq!((q.rows[1].text.as_str(), q.rows[1].label.as_deref()), ("Gewinne jetzt, ruf an!", Some("spam")));
        // Ohne Kopfzeile darf die erste Zeile nicht verloren gehen.
        let q = Datei::neu("a.csv", tabelle("", ",").as_bytes()).lies(false);
        assert_eq!(q.rows.len(), 6);
        assert_eq!(q.rows[0].label.as_deref(), Some("ham"));
    }

    #[test]
    fn semikolon_tab_und_senkrechter_strich() {
        for (name, z) in [("a.csv", ";"), ("a.tsv", "\t"), ("a.csv", "\t"), ("a.csv", "|")] {
            let q = Datei::neu(name, tabelle(&format!("label{}text", z), z).as_bytes()).lies(false);
            assert_eq!(q.rows.len(), 6, "{:?}", z);
            assert_eq!(q.rows[4].text, "Gratis; nur heute", "{:?}", z);
            assert_eq!(q.rows[4].label.as_deref(), Some("spam"), "{:?}", z);
        }
    }

    #[test]
    fn zeichensaetze() {
        let erwartet = "Grüße – 5 €";
        let cp: Vec<u8> = "text,label\nGr".bytes().chain([0xFC, 0xDF]).chain("e ".bytes()).chain([0x96]).chain(" 5 ".bytes()).chain([0x80]).chain(",gruss\n".bytes()).collect();
        assert_eq!(Datei::neu("a.csv", &cp).lies(false).rows[0].text, erwartet);
        let utf8 = format!("\u{feff}text,label\n{},gruss\n", erwartet);
        let q = Datei::neu("a.csv", utf8.as_bytes()).lies(false);
        assert_eq!((q.rows[0].text.as_str(), q.text_spalte.as_deref()), (erwartet, Some("text")));
        let mut le = vec![0xFF, 0xFE];
        for u in format!("text;label\n{};gruss\n", erwartet).encode_utf16() { le.extend(u.to_le_bytes()); }
        assert_eq!(Datei::neu("a.csv", &le).lies(false).rows[0].text, erwartet);
    }

    #[test]
    fn textzeilen_mit_kommas_bleiben_ganz() {
        let q = Datei::neu("a.txt", b"Hallo, wie geht es?\nGut, danke, und dir?\nAuch, ja\n").lies(false);
        assert_eq!(q.rows.iter().map(|r| r.text.as_str()).collect::<Vec<_>>(),
            vec!["Hallo, wie geht es?", "Gut, danke, und dir?", "Auch, ja"]);
        // Einspaltige CSV mit einzelnen Kommas im Text.
        let q = Datei::neu("a.csv", b"text\nHallo\nGut\nJa\nNein\nDoch\n").lies(false);
        assert_eq!(q.rows.len(), 5);
        // Mit Kopfzeile ist auch eine .txt eine Tabelle.
        let q = Datei::neu("a.txt", b"text;label\nHallo;gruss\nTschuess;abschied\n").lies(false);
        assert_eq!(q.rows[1].label.as_deref(), Some("abschied"));
    }

    #[test]
    fn unbekannte_spalte_wird_genannt() {
        let d = Datei::neu("a.csv", b"text,label\nHallo,gruss\n");
        let e = lesen(&d.0, Some("gibtsnicht"), None, false).err().unwrap();
        assert!(e.contains("gibtsnicht") && e.contains("text, label"), "{}", e);
        assert_eq!(lesen(&d.0, Some("LABEL"), Some("text"), false).unwrap().rows[0].text, "gruss");
    }

    #[test]
    fn mehrere_klassenartige_spalten_werden_nicht_geraten() {
        let t = "message,lang,topic\nEin laengerer Satz hier,de,a\nNoch ein laengerer Satz,en,b\nUnd ein dritter Satz,de,a\nVierter Satz im Bunde,en,b\n";
        let q = Datei::neu("a.csv", t.as_bytes()).lies(false);
        assert_eq!((q.text_spalte.as_deref(), q.label_spalte.as_deref()), (Some("message"), None));
    }
}
