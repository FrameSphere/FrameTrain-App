// studio_manager/quality.rs – Qualitaet vor dem Export
//
// Drei Fragen, die ein Datensatz beantworten muss, bevor trainiert wird:
//
//   * Stehen Beinahe-Dubletten darin? Exakte Kopien faengt der Inhalts-Hash
//     schon beim Import ab. Ein Bild in anderer Groesse, ein Text mit einem
//     anderen Satzzeichen ist aber auch eine Kopie — liegt eine in Train und
//     die andere in Val, misst die Validierung Auswendiglernen.
//   * Ist er ausgewogen? 58 zu 2 trainiert ein Modell, das immer die grosse
//     Klasse sagt und dabei 97 % Accuracy meldet.
//   * Ist die Aufteilung dicht? Eine Gruppe (Video, Seite, Sprecher) darf nur
//     in einem Teil liegen.
//
// Texte vergleicht MinHash ueber Wort-Dreiergruppen, Bilder ein dHash aus
// 9x8 Graustufen. Den rechnet die Oberflaeche aus — der Webview dekodiert
// jedes Bildformat, das Backend muesste dafuer eine Bildbibliothek mitbringen.
// Die Hashes liegen in hashes.json, je Mediendatei; weil die Ablage
// inhaltsadressiert ist, veraltet ein Eintrag nie.

use super::*;
use std::collections::HashSet;

// ══════════════════════════════════════════════════════════════════
// TEXT: MINHASH
// ══════════════════════════════════════════════════════════════════

/// Deterministisches Mischen (splitmix64) — std-Hasher sind je Lauf zufaellig.
fn mische(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn hash_str(s: &str) -> u64 {
    // FNV-1a, danach gemischt.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() { h ^= *b as u64; h = h.wrapping_mul(0x100_0000_01b3); }
    mische(h)
}

/// Wort-Dreiergruppen eines Textes, klein und ohne Satzzeichen. Kurze Texte
/// (unter drei Woertern) nehmen Zeichen-Fuenfergruppen, sonst haette "Ja
/// bitte" genau eine Gruppe und waere zu jedem "Ja bitte!" identisch — was
/// hier sogar stimmt, bei "Nein danke" aber nicht mehr trennscharf ist.
pub fn schindeln(text: &str) -> HashSet<u64> {
    let norm: String = text.to_lowercase().chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' }).collect();
    let woerter: Vec<&str> = norm.split_whitespace().collect();
    let mut out = HashSet::new();
    if woerter.len() >= 3 {
        for w in woerter.windows(3) { out.insert(hash_str(&w.join(" "))); }
    } else {
        let zeichen: Vec<char> = woerter.join(" ").chars().collect();
        if zeichen.len() <= 5 {
            out.insert(hash_str(&zeichen.iter().collect::<String>()));
        } else {
            for w in zeichen.windows(5) { out.insert(hash_str(&w.iter().collect::<String>())); }
        }
    }
    out
}

pub fn jaccard(a: &HashSet<u64>, b: &HashSet<u64>) -> f64 {
    if a.is_empty() && b.is_empty() { return 1.0; }
    let schnitt = a.intersection(b).count() as f64;
    schnitt / (a.len() as f64 + b.len() as f64 - schnitt)
}

const PERMUTATIONEN: usize = 64;
const BAENDER: usize = 16;   // 16 Baender zu 4 Zeilen: ab ~0,6 Aehnlichkeit meist Kandidat

fn signatur(s: &HashSet<u64>) -> [u64; PERMUTATIONEN] {
    let mut sig = [u64::MAX; PERMUTATIONEN];
    for &h in s {
        for (i, slot) in sig.iter_mut().enumerate() {
            let v = mische(h ^ (i as u64).wrapping_mul(0xA24B_AED4_963E_E407));
            if v < *slot { *slot = v; }
        }
    }
    sig
}

struct Vereinigung { eltern: Vec<usize> }
impl Vereinigung {
    fn new(n: usize) -> Self { Vereinigung { eltern: (0..n).collect() } }
    fn wurzel(&mut self, mut i: usize) -> usize {
        while self.eltern[i] != i { self.eltern[i] = self.eltern[self.eltern[i]]; i = self.eltern[i]; }
        i
    }
    fn vereine(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.wurzel(a), self.wurzel(b));
        if ra != rb { self.eltern[rb] = ra; }
    }
    fn gruppen(&mut self) -> Vec<Vec<usize>> {
        let mut m: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..self.eltern.len() { let r = self.wurzel(i); m.entry(r).or_default().push(i); }
        let mut g: Vec<Vec<usize>> = m.into_values().filter(|v| v.len() > 1).collect();
        for v in g.iter_mut() { v.sort(); }
        g.sort();
        g
    }
}

/// Gruppen beinahe gleicher Texte (Indizes in `texte`). Kandidaten liefert
/// LSH ueber die MinHash-Signatur, bestaetigt wird mit echtem Jaccard — so
/// bleibt es auch bei 20 000 Texten schnell.
pub fn text_gruppen(texte: &[&str], schwelle: f64) -> Vec<Vec<usize>> {
    let sets: Vec<HashSet<u64>> = texte.iter().map(|t| schindeln(t)).collect();
    let sigs: Vec<[u64; PERMUTATIONEN]> = sets.iter().map(signatur).collect();
    let zeilen = PERMUTATIONEN / BAENDER;
    let mut kandidaten: HashSet<(usize, usize)> = HashSet::new();
    for band in 0..BAENDER {
        let mut eimer: HashMap<u64, Vec<usize>> = HashMap::new();
        for (i, sig) in sigs.iter().enumerate() {
            if sets[i].is_empty() { continue; }
            let mut h: u64 = band as u64;
            for z in 0..zeilen { h = mische(h ^ sig[band * zeilen + z]); }
            eimer.entry(h).or_default().push(i);
        }
        for v in eimer.values() {
            if v.len() < 2 || v.len() > 500 { continue; }
            for a in 0..v.len() { for b in a + 1..v.len() { kandidaten.insert((v[a], v[b])); } }
        }
    }
    let mut uf = Vereinigung::new(texte.len());
    for (a, b) in kandidaten {
        if jaccard(&sets[a], &sets[b]) >= schwelle { uf.vereine(a, b); }
    }
    uf.gruppen()
}

/// Gruppen beinahe gleicher Bilder: dHash mit hoechstens `max_abstand`
/// abweichenden Bits (von 64). 6 findet Groessenaenderung, leichte
/// Kompression und Helligkeit, aber keine zwei verschiedenen Fotos derselben Szene.
pub fn bild_gruppen(hashes: &[u64], max_abstand: u32) -> Vec<Vec<usize>> {
    let mut uf = Vereinigung::new(hashes.len());
    for a in 0..hashes.len() {
        for b in a + 1..hashes.len() {
            if (hashes[a] ^ hashes[b]).count_ones() <= max_abstand { uf.vereine(a, b); }
        }
    }
    uf.gruppen()
}

// ══════════════════════════════════════════════════════════════════
// BILD-HASHES (von der Oberflaeche gerechnet)
// ══════════════════════════════════════════════════════════════════

fn hashes_path(dir: &Path) -> PathBuf { dir.join("hashes.json") }

pub(super) fn load_hashes(dir: &Path) -> HashMap<String, u64> {
    let roh: HashMap<String, String> = fs::read_to_string(hashes_path(dir)).ok()
        .and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
    roh.into_iter().filter_map(|(k, v)| u64::from_str_radix(&v, 16).ok().map(|h| (k, h))).collect()
}

#[tauri::command]
pub async fn studio_save_hashes(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    project_id: String, hashes: HashMap<String, String>,
) -> Result<usize, String> {
    let user_id = get_user_id(&state)?;
    let dir = project_dir(&app_handle, &user_id, &project_id)?;
    let mut alle: HashMap<String, String> = fs::read_to_string(hashes_path(&dir)).ok()
        .and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
    for (k, v) in hashes {
        if u64::from_str_radix(&v, 16).is_ok() { alle.insert(k, v); }
    }
    fs::write(hashes_path(&dir), serde_json::to_string(&alle).map_err(|e| e.to_string())?)
        .map_err(|e| format!("hashes.json: {}", e))?;
    Ok(alle.len())
}

#[derive(Debug, Clone, Serialize)]
pub struct NearDupReport {
    /// "text" | "image" | "exact_only" (Audio, Video: nur exakte Dubletten)
    pub kind:    String,
    pub groups:  Vec<Vec<String>>,
    pub checked: usize,
    /// Bilder ohne Hash — die Oberflaeche rechnet sie und fragt erneut.
    pub missing: Vec<StudioSample>,
}

#[tauri::command]
pub async fn studio_near_duplicates(
    app_handle: tauri::AppHandle, state: State<'_, AppState>, project_id: String,
) -> Result<NearDupReport, String> {
    let user_id = get_user_id(&state)?;
    let dir = project_dir(&app_handle, &user_id, &project_id)?;
    let project = load_project(&dir)?;
    let samples = load_samples(&dir);
    let media_dir = dir.join("media");

    match project.modality.as_str() {
        "text" => {
            let texte: Vec<&str> = samples.iter().map(|s| s.content.as_deref().unwrap_or("")).collect();
            let groups = text_gruppen(&texte, 0.8).into_iter()
                .map(|g| g.into_iter().map(|i| samples[i].id.clone()).collect()).collect();
            Ok(NearDupReport { kind: "text".to_string(), groups, checked: samples.len(), missing: vec![] })
        }
        "image" => {
            let hashes = load_hashes(&dir);
            let mut mit: Vec<(usize, u64)> = Vec::new();
            let mut missing = Vec::new();
            for (i, s) in samples.iter().enumerate() {
                match hashes.get(&s.media) {
                    Some(h) => mit.push((i, *h)),
                    None => {
                        let mut c = s.clone();
                        c.abs_path = media_dir.join(&s.media).to_string_lossy().to_string();
                        missing.push(c);
                    }
                }
            }
            let nur: Vec<u64> = mit.iter().map(|(_, h)| *h).collect();
            let groups = bild_gruppen(&nur, 6).into_iter()
                .map(|g| g.into_iter().map(|j| samples[mit[j].0].id.clone()).collect()).collect();
            Ok(NearDupReport { kind: "image".to_string(), groups, checked: mit.len(), missing })
        }
        _ => Ok(NearDupReport { kind: "exact_only".to_string(), groups: vec![], checked: samples.len(), missing: vec![] }),
    }
}

// ══════════════════════════════════════════════════════════════════
// BALANCE
// ══════════════════════════════════════════════════════════════════

/// Unter so vielen Beispielen lernt eine Klasse kaum etwas Verlaessliches.
pub const MIN_JE_KLASSE: usize = 10;
/// Ab diesem Verhaeltnis groesste zu kleinster Klasse wird gewarnt.
pub const MAX_VERHAELTNIS: f64 = 5.0;

pub fn balance_warnungen(per_class: &[(String, usize)]) -> Vec<String> {
    let mut w = Vec::new();
    if per_class.is_empty() { return w; }
    for (name, n) in per_class {
        if *n == 0 {
            w.push(format!("Klasse „{}“ hat keine Beispiele.", name));
        } else if *n < MIN_JE_KLASSE {
            w.push(format!("Klasse „{}“ hat nur {} Beispiele (empfohlen: mindestens {}).", name, n, MIN_JE_KLASSE));
        }
    }
    let belegt: Vec<&(String, usize)> = per_class.iter().filter(|(_, n)| *n > 0).collect();
    if belegt.len() >= 2 {
        let gross = belegt.iter().max_by_key(|(_, n)| *n).unwrap();
        let klein = belegt.iter().min_by_key(|(_, n)| *n).unwrap();
        if gross.1 as f64 / klein.1 as f64 > MAX_VERHAELTNIS {
            w.push(format!("Unausgewogen: „{}“ hat {} Beispiele, „{}“ nur {}. Ein Modell lernt dann vor allem, die große Klasse zu sagen.",
                gross.0, gross.1, klein.0, klein.1));
        }
    }
    w
}

// ══════════════════════════════════════════════════════════════════
// EXPORT-REPORT
// ══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Default, Serialize)]
pub struct ExportReport {
    pub total:                  usize,
    pub per_class:              Vec<(String, usize)>,
    pub per_split:              Vec<(String, usize)>,
    pub groups:                 usize,
    /// Gruppen, die in mehr als einem Teil liegen — sollte leer sein.
    pub group_leaks:            Vec<String>,
    /// Paare beinahe gleicher Samples im Export.
    pub near_duplicates:        usize,
    /// Davon ueber Teilgrenzen hinweg: das eigentliche Leck.
    pub near_duplicates_across: usize,
    /// false, wenn nicht geprueft werden konnte (Audio, Video, Bilder ohne Hash).
    pub near_checked:           bool,
    pub licenses:               Vec<(String, usize)>,
    pub without_license:        usize,
    pub sources:                Vec<(String, usize)>,
    pub warnings:               Vec<String>,
}

fn zaehle<I: IntoIterator<Item = String>>(it: I) -> Vec<(String, usize)> {
    let mut m: HashMap<String, usize> = HashMap::new();
    for k in it { *m.entry(k).or_default() += 1; }
    let mut v: Vec<(String, usize)> = m.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v
}

/// Der Bericht zu einem Export. `splits` hat dieselbe Laenge wie `samples`.
pub fn baue_report(
    project: &StudioProject, samples: &[&StudioSample], splits: &[Option<String>],
    bild_hashes: &HashMap<String, u64>,
) -> ExportReport {
    let mut r = ExportReport { total: samples.len(), ..Default::default() };

    // Verteilung: bei Boxen die Boxen je Klasse, sonst die Labels.
    let boxen = project.modality == "image" && project.task != "classify";
    let ohne_klassen = project.task == "pairs" || project.task == "transcript";
    if !ohne_klassen {
        let mut je: Vec<usize> = vec![0; project.classes.len()];
        for s in samples {
            if boxen {
                for b in &s.ann.boxes { if b.cls < je.len() { je[b.cls] += 1; } }
            } else if let Some(i) = s.ann.label.as_deref().and_then(|l| class_index_for(l, &project.classes)) {
                je[i] += 1;
            }
        }
        r.per_class = project.classes.iter().cloned().zip(je).collect();
        r.warnings.extend(balance_warnungen(&r.per_class));
    }

    let geteilt = splits.iter().any(|s| s.is_some());
    if geteilt {
        r.per_split = zaehle(splits.iter().map(|s| s.clone().unwrap_or_else(|| "—".to_string())));
        let mut wo: HashMap<String, HashSet<String>> = HashMap::new();
        for (s, sp) in samples.iter().zip(splits) {
            let g = s.meta.group.clone().unwrap_or_else(|| s.id.clone());
            if let Some(sp) = sp { wo.entry(g).or_default().insert(sp.clone()); }
        }
        r.groups = wo.len();
        r.group_leaks = wo.into_iter().filter(|(_, v)| v.len() > 1).map(|(g, _)| g).collect();
        r.group_leaks.sort();
        if !r.group_leaks.is_empty() {
            r.warnings.push(format!("{} Gruppe(n) liegen in mehr als einem Teil.", r.group_leaks.len()));
        }
    } else {
        r.groups = samples.iter().map(|s| s.meta.group.clone().unwrap_or_else(|| s.id.clone()))
            .collect::<HashSet<_>>().len();
    }

    // Beinahe-Dubletten.
    let gruppen: Option<Vec<Vec<usize>>> = match project.modality.as_str() {
        "text" => {
            let texte: Vec<&str> = samples.iter().map(|s| s.content.as_deref().unwrap_or("")).collect();
            Some(text_gruppen(&texte, 0.8))
        }
        "image" => {
            let mit: Vec<(usize, u64)> = samples.iter().enumerate()
                .filter_map(|(i, s)| bild_hashes.get(&s.media).map(|h| (i, *h))).collect();
            if mit.len() == samples.len() {
                let nur: Vec<u64> = mit.iter().map(|(_, h)| *h).collect();
                Some(bild_gruppen(&nur, 6).into_iter().map(|g| g.into_iter().map(|j| mit[j].0).collect()).collect())
            } else { None }
        }
        _ => None,
    };
    if let Some(gruppen) = gruppen {
        r.near_checked = true;
        for g in &gruppen {
            for a in 0..g.len() {
                for b in a + 1..g.len() {
                    r.near_duplicates += 1;
                    if geteilt && splits[g[a]] != splits[g[b]] { r.near_duplicates_across += 1; }
                }
            }
        }
        if r.near_duplicates_across > 0 {
            r.warnings.push(format!("{} Paar(e) beinahe gleicher Samples liegen in verschiedenen Teilen — die Validierung misst dort Wiedererkennen.",
                r.near_duplicates_across));
        } else if r.near_duplicates > 0 {
            r.warnings.push(format!("{} Paar(e) beinahe gleicher Samples im Datensatz.", r.near_duplicates));
        }
    }

    r.licenses = zaehle(samples.iter().filter_map(|s| s.src.license.clone().filter(|l| !l.trim().is_empty())));
    r.without_license = samples.iter()
        .filter(|s| s.src.kind == "web" && s.src.license.as_deref().map(|l| l.trim().is_empty()).unwrap_or(true))
        .count();
    if r.without_license > 0 {
        r.warnings.push(format!("{} aus dem Netz geholte Samples ohne Lizenzangabe.", r.without_license));
    }
    r.sources = zaehle(samples.iter().map(|s| s.src.kind.clone()));
    r
}

pub fn report_markdown(project: &StudioProject, r: &ExportReport) -> String {
    let mut md = format!("# Export-Report: {}\n\nErzeugt am {}.\n\n- Samples: {}\n- Gruppen: {}\n",
        project.name, Utc::now().format("%Y-%m-%d %H:%M"), r.total, r.groups);
    if !r.per_class.is_empty() {
        md.push_str("\n## Klassen\n\n| Klasse | Anzahl |\n|---|---:|\n");
        for (k, n) in &r.per_class { md.push_str(&format!("| {} | {} |\n", k, n)); }
    }
    if !r.per_split.is_empty() {
        md.push_str("\n## Aufteilung\n\n");
        for (k, n) in &r.per_split { md.push_str(&format!("- {}: {}\n", k, n)); }
        md.push_str(&format!("- Gruppen in mehr als einem Teil: {}\n", r.group_leaks.len()));
    }
    md.push_str("\n## Beinahe-Dubletten\n\n");
    if r.near_checked {
        md.push_str(&format!("- Paare: {}\n- davon über Teilgrenzen: {}\n", r.near_duplicates, r.near_duplicates_across));
    } else {
        md.push_str("- nicht geprüft (nur exakte Dubletten; bei Bildern: Prüfung in der Werkstatt ausführen)\n");
    }
    md.push_str("\n## Herkunft\n\n");
    for (k, n) in &r.sources { md.push_str(&format!("- {}: {}\n", k, n)); }
    if !r.licenses.is_empty() {
        md.push_str("\n## Lizenzen\n\n");
        for (k, n) in &r.licenses { md.push_str(&format!("- {}: {}\n", k, n)); }
    }
    if !r.warnings.is_empty() {
        md.push_str("\n## Hinweise\n\n");
        for w in &r.warnings { md.push_str(&format!("- {}\n", w)); }
    }
    md
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beinahe_gleiche_texte_finden_sich() {
        let texte = [
            "Der Sessellift fährt ab acht Uhr und bringt bis zu 2400 Personen je Stunde nach oben.",
            "Der Sessellift fährt ab acht Uhr und bringt bis zu 2400 Personen je Stunde nach oben!",
            "Bei Sturm bleibt die Anlage geschlossen, das zeigt die Tafel an der Talstation.",
            "Der Sessellift faehrt ab neun Uhr, heute nur bis Mittag, wegen Wartung an der Bergstation.",
        ];
        let g = text_gruppen(&texte, 0.8);
        assert_eq!(g, vec![vec![0, 1]]);
    }

    #[test]
    fn jaccard_rechnet_richtig() {
        let a: HashSet<u64> = [1, 2, 3, 4].into_iter().collect();
        let b: HashSet<u64> = [3, 4, 5, 6].into_iter().collect();
        assert!((jaccard(&a, &b) - 2.0 / 6.0).abs() < 1e-9);
        assert!((jaccard(&a, &a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn bildhashes_mit_wenigen_abweichenden_bits() {
        let h = [0xF0F0_F0F0_F0F0_F0F0u64, 0xF0F0_F0F0_F0F0_F0F1, 0x0F0F_0F0F_0F0F_0F0F, 0xF0F0_F0F0_F0F0_F0F0 ^ 0xFF];
        // 0 und 1 unterscheiden sich in einem Bit, 3 von beiden in sieben oder acht — zu viel.
        assert_eq!(bild_gruppen(&h, 6), vec![vec![0, 1]]);
    }

    #[test]
    fn balance_warnt_bei_wenig_und_schieflage() {
        let w = balance_warnungen(&[("ja".into(), 60), ("nein".into(), 4), ("vielleicht".into(), 0)]);
        assert_eq!(w.len(), 3, "{:?}", w);
        assert!(w[0].contains("nein") && w[0].contains("nur 4"));
        assert!(w[1].contains("vielleicht") && w[1].contains("keine"));
        assert!(w[2].starts_with("Unausgewogen"));
        assert!(balance_warnungen(&[("a".into(), 20), ("b".into(), 30)]).is_empty());
    }

    fn sample(id: &str, label: &str, group: &str, text: &str) -> StudioSample {
        StudioSample {
            id: id.into(), media: String::new(), mime: "text/plain".into(), content: Some(text.into()),
            status: "confirmed".into(),
            ann: Annotation { label: Some(label.into()), ..Default::default() },
            src: SampleSource { kind: "web".into(), origin: None, license: None, at: String::new(), page: None },
            meta: SampleMeta { group: Some(group.into()), ..Default::default() },
            abs_path: String::new(), doubt: None,
        }
    }

    #[test]
    fn report_findet_leck_ueber_teilgrenzen() {
        let p = StudioProject { id: "p".into(), name: "T".into(), modality: "text".into(), task: "classification".into(),
            target_format: "flat_file".into(), classes: vec!["a".into(), "b".into()],
            created_at: String::new(), updated_at: String::new() };
        let s = [
            sample("1", "a", "g1", "Der Sessellift fährt ab acht Uhr und bringt bis zu 2400 Personen je Stunde nach oben."),
            sample("2", "a", "g2", "Der Sessellift fährt ab acht Uhr und bringt bis zu 2400 Personen je Stunde nach oben!"),
            sample("3", "b", "g3", "Bei Sturm bleibt die Anlage geschlossen, das zeigt die Tafel an der Talstation."),
        ];
        let refs: Vec<&StudioSample> = s.iter().collect();
        let splits = vec![Some("train".to_string()), Some("val".to_string()), Some("train".to_string())];
        let r = baue_report(&p, &refs, &splits, &HashMap::new());
        assert!(r.near_checked);
        assert_eq!((r.near_duplicates, r.near_duplicates_across), (1, 1));
        assert!(r.group_leaks.is_empty());
        assert_eq!(r.per_class, vec![("a".to_string(), 2), ("b".to_string(), 1)]);
        assert_eq!(r.without_license, 3);
        assert!(r.warnings.iter().any(|w| w.contains("verschiedenen Teilen")), "{:?}", r.warnings);
        let md = report_markdown(&p, &r);
        assert!(md.contains("davon über Teilgrenzen: 1"), "{}", md);
    }
}
