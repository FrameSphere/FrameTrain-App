// studio_manager/web.rs – Daten aus dem Netz holen
//
// Kein Knopf, der das Netz absaugt, aber auch nicht mehr nur eine Liste
// direkter Dateiadressen. Drei Wege:
//
//   * Adressen:  jede Adresse einzeln. Ist es eine Seite, wird sie nach dem
//                durchsucht, was das Projekt braucht (Bilder, Audio, Video,
//                Absaetze) — eine Seitenadresse im Bildprojekt war bisher
//                "falscher Typ".
//   * Website:   von den Startseiten aus Links folgen, bis zur gewaehlten
//                Tiefe und hoechstens so viele Seiten wie eingestellt.
//   * Sitemap:   die Seiten aus sitemap.xml (auch Sitemap-Verzeichnisse und
//                "Sitemap:"-Zeilen der robots.txt).
//
// Mit Anstand, und zwar immer:
//   * robots.txt wird gelesen und befolgt, samt Crawl-delay,
//   * je Server mindestens eine Sekunde Abstand,
//   * Seiten ausserhalb der Allowlist werden nicht besucht — ohne Allowlist
//     bleibt ein Crawl auf den Domains der Startadressen,
//   * keine Datei ueber dem Groessenlimit; gemessen wird beim Laden, nicht nur
//     am Content-Length-Kopf, den Server auch weglassen,
//   * jede Datei traegt Adresse, Fundseite und angegebene Lizenz mit.
//
// Der letzte Punkt ist kein Beiwerk: ein gesammelter Datensatz ohne Herkunft
// darf dieses Geraet nie verlassen, und das merkt man erst, wenn es zu spaet
// ist.

use super::*;
use reqwest::Url;
use std::collections::{HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};

const AGENT: &str = "FrameTrain-DatasetStudio";

/// Gesetzt von "Abbrechen"; der Lauf prueft es vor jeder Anfrage.
static ABBRUCH: AtomicBool = AtomicBool::new(false);

// ══════════════════════════════════════════════════════════════════
// OPTIONEN UND BERICHT
// ══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct WebOptions {
    /// "urls" | "crawl" | "sitemap"
    pub mode:       String,
    /// Wie viele Klicks von der Startseite weg (nur "crawl").
    pub max_depth:  usize,
    pub max_pages:  usize,
    pub max_files:  usize,
    /// Domains, die besucht werden duerfen. Eine Domain schliesst ihre
    /// Subdomains ein ("example.org" erlaubt "img.example.org").
    pub allowlist:  Vec<String>,
    /// Groessenlimit je Datei in MB.
    pub max_mb:     f64,
    /// Bilder, deren kuerzere Seite darunter liegt, sind Symbole und Knoepfe.
    pub min_side:   u32,
    /// Text in Absaetze zerlegen statt eine Seite als ein Sample.
    pub paragraphs: bool,
    pub min_chars:  usize,
    pub license:    Option<String>,
}

impl Default for WebOptions {
    fn default() -> Self {
        WebOptions {
            mode: "urls".to_string(), max_depth: 1, max_pages: 30, max_files: 300,
            allowlist: vec![], max_mb: 20.0, min_side: 64, paragraphs: true, min_chars: 40,
            license: None,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct FetchReport {
    pub fetched:           usize,
    pub duplicates:        usize,
    /// Von robots.txt untersagt — mit der Adresse, damit es nachvollziehbar ist.
    pub blocked:           Vec<String>,
    pub failed:            Vec<String>,
    pub skipped_type:      Vec<String>,
    pub pages_visited:     usize,
    /// Ueber dem Groessenlimit, nicht geladen.
    pub too_large:         Vec<String>,
    /// Bilder unter der Mindestgroesse (Symbole, Zaehlpixel).
    pub too_small:         usize,
    /// Links auf Domains ausserhalb der Allowlist, nicht besucht.
    pub outside_allowlist: usize,
    /// Seiten- oder Dateigrenze erreicht — es gaebe noch mehr.
    pub limit_reached:     bool,
    pub cancelled:         bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Art { Text, Bild, Audio, Video }

fn art_von(project: &StudioProject) -> Art {
    match project.modality.as_str() {
        "text"  => Art::Text,
        "audio" => Art::Audio,
        "video" => Art::Video,
        _       => Art::Bild,
    }
}

// ══════════════════════════════════════════════════════════════════
// ROBOTS.TXT
// ══════════════════════════════════════════════════════════════════

/// Die Zeilen der Gruppe, die fuer uns gilt: eine Gruppe auf unseren Namen
/// schlaegt den Stern, und dann gilt nur sie, nicht beide zusammen.
fn robots_gruppe(robots: &str, agent: &str) -> Option<Vec<(String, String)>> {
    let agent = agent.to_lowercase();
    // Mehrere User-agent-Zeilen hintereinander gehoeren zu denselben Regeln.
    let mut gruppen: Vec<(Vec<String>, Vec<(String, String)>)> = Vec::new();
    let mut namen: Vec<String> = Vec::new();
    let mut regeln: Vec<(String, String)> = Vec::new();
    let mut zuletzt_regel = false;

    for zeile in robots.lines() {
        let zeile = zeile.split('#').next().unwrap_or("").trim();
        if zeile.is_empty() { continue; }
        let Some((schluessel, wert)) = zeile.split_once(':') else { continue };
        let schluessel = schluessel.trim().to_lowercase();
        let wert = wert.trim().to_string();
        if schluessel == "user-agent" {
            if zuletzt_regel && !namen.is_empty() {
                gruppen.push((std::mem::take(&mut namen), std::mem::take(&mut regeln)));
            }
            zuletzt_regel = false;
            namen.push(wert.to_lowercase());
        } else if schluessel != "sitemap" {
            zuletzt_regel = true;
            if !namen.is_empty() { regeln.push((schluessel, wert)); }
        }
    }
    if !namen.is_empty() { gruppen.push((namen, regeln)); }

    let eigene = gruppen.iter()
        .find(|(namen, _)| namen.iter().any(|n| n != "*" && agent.contains(n.as_str())));
    let stern = gruppen.iter().find(|(namen, _)| namen.iter().any(|n| n == "*"));
    eigene.or(stern).map(|(_, r)| r.clone())
}

/// Darf dieser Pfad geholt werden?
///
/// Innerhalb der Gruppe gewinnt die laengste Uebereinstimmung, bei gleicher
/// Laenge das erlaubende Allow — so steht es im Standard (RFC 9309) und so
/// verhalten sich die grossen Crawler.
pub fn robots_erlaubt(robots: &str, agent: &str, pfad: &str) -> bool {
    let Some(regeln) = robots_gruppe(robots, agent) else { return true };
    let mut treffer: Vec<(usize, bool)> = Vec::new();
    for (art, wert) in &regeln {
        let erlaubt = match art.as_str() { "allow" => true, "disallow" => false, _ => continue };
        if wert.is_empty() {
            // "Disallow:" ohne Wert erlaubt alles.
            if !erlaubt { treffer.push((0, true)); }
            continue;
        }
        if pfad.starts_with(wert.as_str()) { treffer.push((wert.len(), erlaubt)); }
    }
    match treffer.iter().max_by_key(|(len, erlaubt)| (*len, *erlaubt)) {
        Some((_, erlaubt)) => *erlaubt,
        None => true,
    }
}

/// Crawl-delay der fuer uns geltenden Gruppe, gedeckelt auf zehn Sekunden —
/// ein Wert von 3600 wuerde den Lauf sonst eine Stunde je Seite anhalten.
pub fn robots_crawl_delay(robots: &str, agent: &str) -> Option<f64> {
    let regeln = robots_gruppe(robots, agent)?;
    regeln.iter()
        .find(|(k, _)| k == "crawl-delay")
        .and_then(|(_, v)| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(|v| v.min(10.0))
}

/// "Sitemap:"-Zeilen — sie gelten unabhaengig von jeder Gruppe.
pub fn robots_sitemaps(robots: &str) -> Vec<String> {
    robots.lines()
        .filter_map(|z| {
            let z = z.split('#').next().unwrap_or("").trim();
            let (k, v) = z.split_once(':')?;
            if k.trim().eq_ignore_ascii_case("sitemap") { Some(v.trim().to_string()) } else { None }
        })
        .filter(|v| v.starts_with("http"))
        .collect()
}

// ══════════════════════════════════════════════════════════════════
// HTML LESEN
//
// Bewusst ohne HTML-Bibliothek: gebraucht werden Attribute weniger Tags und
// der sichtbare Text. Ein kleiner Scanner, der unbekanntes ueberspringt statt
// daran zu scheitern, ist hier robuster als ein halber Parser.
// ══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct Tag {
    pub name:  String,
    pub attrs: Vec<(String, String)>,
}

impl Tag {
    fn attr(&self, name: &str) -> Option<&str> {
        self.attrs.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }
}

/// Entities fuer die haeufigen Faelle und alle numerischen.
pub fn entities(s: &str) -> String {
    if !s.contains('&') { return s.to_string(); }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(pos) = rest.find('&') {
        out.push_str(&rest[..pos]);
        rest = &rest[pos..];
        let ende = rest[..rest.len().min(12)].find(';');
        let ersetzt = ende.and_then(|e| {
            let name = &rest[1..e];
            let zeichen = match name {
                "amp" => Some('&'), "lt" => Some('<'), "gt" => Some('>'),
                "quot" => Some('"'), "apos" => Some('\''), "nbsp" => Some(' '),
                "ndash" => Some('–'), "mdash" => Some('—'), "hellip" => Some('…'),
                "auml" => Some('ä'), "ouml" => Some('ö'), "uuml" => Some('ü'),
                "Auml" => Some('Ä'), "Ouml" => Some('Ö'), "Uuml" => Some('Ü'), "szlig" => Some('ß'),
                _ if name.starts_with("#x") || name.starts_with("#X") =>
                    u32::from_str_radix(&name[2..], 16).ok().and_then(char::from_u32),
                _ if name.starts_with('#') => name[1..].parse::<u32>().ok().and_then(char::from_u32),
                _ => None,
            };
            zeichen.map(|z| (z, e))
        });
        match ersetzt {
            Some((z, e)) => { out.push(z); rest = &rest[e + 1..]; }
            None => { out.push('&'); rest = &rest[1..]; }
        }
    }
    out.push_str(rest);
    out
}

/// Tags mit ihren Attributen, in Dokumentreihenfolge. Inhalte von script und
/// style werden uebersprungen, Kommentare ebenso.
pub fn tags(html: &str) -> Vec<Tag> {
    let b = html.as_bytes();
    let n = b.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        if b[i] != b'<' { i += 1; continue; }
        if html[i..].starts_with("<!--") {
            i = html[i..].find("-->").map(|e| i + e + 3).unwrap_or(n);
            continue;
        }
        let mut j = i + 1;
        if j < n && (b[j] == b'/' || b[j] == b'!' || b[j] == b'?') {
            i = html[j..].find('>').map(|e| j + e + 1).unwrap_or(n);
            continue;
        }
        let name_start = j;
        while j < n && (b[j].is_ascii_alphanumeric() || b[j] == b'-') { j += 1; }
        if j == name_start { i += 1; continue; }
        let name = html[name_start..j].to_ascii_lowercase();
        let mut attrs = Vec::new();
        loop {
            while j < n && b[j].is_ascii_whitespace() { j += 1; }
            if j >= n { break; }
            if b[j] == b'>' { j += 1; break; }
            if b[j] == b'/' { j += 1; continue; }
            let a_start = j;
            while j < n && !b[j].is_ascii_whitespace() && b[j] != b'=' && b[j] != b'>' && b[j] != b'/' { j += 1; }
            let a_name = html[a_start..j].to_ascii_lowercase();
            while j < n && b[j].is_ascii_whitespace() { j += 1; }
            let mut wert = String::new();
            if j < n && b[j] == b'=' {
                j += 1;
                while j < n && b[j].is_ascii_whitespace() { j += 1; }
                if j < n && (b[j] == b'"' || b[j] == b'\'') {
                    let q = b[j];
                    let v_start = j + 1;
                    let v_end = b[v_start..].iter().position(|c| *c == q).map(|e| v_start + e).unwrap_or(n);
                    wert = entities(&html[v_start..v_end]);
                    j = (v_end + 1).min(n);
                } else {
                    let v_start = j;
                    while j < n && !b[j].is_ascii_whitespace() && b[j] != b'>' { j += 1; }
                    wert = entities(&html[v_start..j]);
                }
            }
            if !a_name.is_empty() { attrs.push((a_name, wert)); }
            if a_start == j { j += 1; }
        }
        let ueberspringen = name == "script" || name == "style";
        out.push(Tag { name: name.clone(), attrs });
        i = j;
        if ueberspringen {
            let ende = format!("</{}", name);
            let rest_lc = html[i..].to_ascii_lowercase();
            i = rest_lc.find(&ende).map(|e| i + e).unwrap_or(n);
        }
    }
    out
}

/// Die groesste Variante aus einem srcset ("a.jpg 480w, b.jpg 1080w").
fn groesstes_aus_srcset(srcset: &str) -> Option<String> {
    srcset.split(',')
        .filter_map(|teil| {
            let mut it = teil.split_whitespace();
            let url = it.next()?.to_string();
            let wert = it.next()
                .and_then(|d| d.trim_end_matches(|c| c == 'w' || c == 'x').parse::<f64>().ok())
                .unwrap_or(1.0);
            Some((url, wert))
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(u, _)| u)
}

/// Was eine Seite verlinkt und einbettet, nach Art sortiert.
#[derive(Debug, Default, Clone)]
pub struct Funde {
    pub seiten: Vec<Url>,
    pub bilder: Vec<Url>,
    pub audio:  Vec<Url>,
    pub video:  Vec<Url>,
}

fn endung(url: &Url) -> String {
    Path::new(url.path()).extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase()
}

fn art_nach_endung(url: &Url) -> Option<Art> {
    let e = endung(url);
    if IMAGE_EXTS.contains(&e.as_str()) || e == "svg" { return Some(Art::Bild); }
    if AUDIO_EXTS.contains(&e.as_str()) { return Some(Art::Audio); }
    if VIDEO_EXTS.contains(&e.as_str()) { return Some(Art::Video); }
    None
}

pub fn funde(html: &str, seite: &Url) -> Funde {
    let alle = tags(html);
    // <base href> verschiebt, wogegen relative Links aufgeloest werden.
    let basis = alle.iter()
        .find(|t| t.name == "base")
        .and_then(|t| t.attr("href"))
        .and_then(|h| seite.join(h).ok())
        .unwrap_or_else(|| seite.clone());

    let mut f = Funde::default();
    let mut gesehen: HashSet<String> = HashSet::new();
    let mut ablegen = |f: &mut Funde, roh: &str, art: Option<Art>| {
        let roh = roh.trim();
        if roh.is_empty() || roh.starts_with('#') || roh.starts_with("data:")
            || roh.starts_with("javascript:") || roh.starts_with("mailto:") || roh.starts_with("tel:") {
            return;
        }
        let Ok(mut url) = basis.join(roh) else { return };
        if url.scheme() != "http" && url.scheme() != "https" { return; }
        url.set_fragment(None);
        if !gesehen.insert(url.to_string()) { return; }
        match art.or_else(|| art_nach_endung(&url)) {
            Some(Art::Bild) if endung(&url) != "svg" => f.bilder.push(url),
            Some(Art::Bild) => {}
            Some(Art::Audio) => f.audio.push(url),
            Some(Art::Video) => f.video.push(url),
            Some(Art::Text) | None => f.seiten.push(url),
        }
    };

    // Welches Medien-Tag gerade offen ist: <source> in <video> ist Video.
    let mut offen: Option<Art> = None;
    for t in &alle {
        match t.name.as_str() {
            "a" => {
                if let Some(h) = t.attr("href") {
                    // rel=nofollow ist ein Wunsch des Betreibers — kein Verbot,
                    // aber ein Grund, dem Link nicht weiter zu folgen.
                    let nofollow = t.attr("rel").map(|r| r.contains("nofollow")).unwrap_or(false);
                    if !nofollow { ablegen(&mut f, h, None); }
                }
            }
            "img" => {
                for a in ["src", "data-src", "data-original", "data-lazy-src"] {
                    if let Some(v) = t.attr(a) { ablegen(&mut f, v, Some(Art::Bild)); }
                }
                for a in ["srcset", "data-srcset"] {
                    if let Some(g) = t.attr(a).and_then(groesstes_aus_srcset) { ablegen(&mut f, &g, Some(Art::Bild)); }
                }
            }
            "video" => {
                offen = Some(Art::Video);
                if let Some(v) = t.attr("src") { ablegen(&mut f, v, Some(Art::Video)); }
            }
            "audio" => {
                offen = Some(Art::Audio);
                if let Some(v) = t.attr("src") { ablegen(&mut f, v, Some(Art::Audio)); }
            }
            "picture" => offen = Some(Art::Bild),
            "source" => {
                let typ = t.attr("type").unwrap_or("").to_lowercase();
                let art = if typ.starts_with("video/") { Some(Art::Video) }
                    else if typ.starts_with("audio/") { Some(Art::Audio) }
                    else if typ.starts_with("image/") { Some(Art::Bild) }
                    else { offen };
                if let Some(v) = t.attr("src") { ablegen(&mut f, v, art); }
                if let Some(g) = t.attr("srcset").and_then(groesstes_aus_srcset) { ablegen(&mut f, &g, art.or(Some(Art::Bild))); }
            }
            "meta" => {
                let prop = t.attr("property").or_else(|| t.attr("name")).unwrap_or("").to_lowercase();
                if let Some(c) = t.attr("content") {
                    match prop.as_str() {
                        "og:image" | "og:image:url" | "twitter:image" => ablegen(&mut f, c, Some(Art::Bild)),
                        "og:video" | "og:video:url"                   => ablegen(&mut f, c, Some(Art::Video)),
                        "og:audio" | "og:audio:url"                   => ablegen(&mut f, c, Some(Art::Audio)),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    f
}

/// Die Seitenadressen einer Sitemap, und ob es ein Verzeichnis weiterer
/// Sitemaps ist (dann sind die Adressen selbst Sitemaps).
pub fn sitemap_locs(xml: &str) -> (Vec<String>, bool) {
    let ist_index = xml.contains("<sitemapindex");
    let mut out = Vec::new();
    let mut rest = xml;
    while let Some(a) = rest.find("<loc>") {
        rest = &rest[a + 5..];
        let Some(e) = rest.find("</loc>") else { break };
        let roh = rest[..e].trim().trim_start_matches("<![CDATA[").trim_end_matches("]]>").trim();
        if !roh.is_empty() { out.push(entities(roh)); }
        rest = &rest[e + 6..];
    }
    (out, ist_index)
}

/// Sichtbarer Text aus HTML, als eine Zeichenkette.
///
/// Bewusst grob: Skript- und Stilbloecke raus, Tags raus, Entities. Das reicht
/// fuer eine Seite als Ganzes; wer Absaetze will, nimmt textbloecke().
pub fn html_zu_text(html: &str) -> String {
    textbloecke(html, 1, false).join("\n")
}

/// Bereiche, die fast nie zum Inhalt gehoeren: Navigation, Kopf- und
/// Fusszeilen, Formulare. Ohne sie besteht ein Absatz-Datensatz zur Haelfte
/// aus "Startseite", "Impressum", "Cookies akzeptieren".
const RAHMEN: &[&str] = &["script", "style", "noscript", "svg", "template", "nav", "header",
    "footer", "aside", "form", "button", "select", "iframe"];

const BLOCK: &[&str] = &["p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "br",
    "section", "article", "main", "td", "th", "tr", "table", "blockquote", "pre", "dd", "dt",
    "figcaption", "hr", "body"];

/// Der sichtbare Text in Absaetzen.
///
/// `ohne_rahmen` laesst Navigation, Kopf und Fuss weg. Absaetze unter
/// `min_zeichen` fallen raus (Menuepunkte, Datumszeilen), gleiche Absaetze
/// stehen nur einmal da (Teaser, die unten wiederholt werden).
pub fn textbloecke(html: &str, min_zeichen: usize, ohne_rahmen: bool) -> Vec<String> {
    let b = html.as_bytes();
    let n = b.len();
    let mut roh = String::with_capacity(n / 2);
    let mut i = 0;
    let mut text_start = 0;

    // Nimmt den Text zwischen zwei Tags mit.
    let flush = |roh: &mut String, von: usize, bis: usize| {
        if bis > von { roh.push_str(&html[von..bis]); }
    };

    while i < n {
        if b[i] != b'<' { i += 1; continue; }
        flush(&mut roh, text_start, i);
        if html[i..].starts_with("<!--") {
            i = html[i..].find("-->").map(|e| i + e + 3).unwrap_or(n);
            text_start = i;
            continue;
        }
        let schliessend = i + 1 < n && b[i + 1] == b'/';
        let mut j = i + 1 + usize::from(schliessend);
        let name_start = j;
        while j < n && (b[j].is_ascii_alphanumeric() || b[j] == b'-') { j += 1; }
        let name = html[name_start..j].to_ascii_lowercase();
        let tag_ende = html[j..].find('>').map(|e| j + e + 1).unwrap_or(n);
        let selbst_schliessend = tag_ende >= 2 && b[tag_ende - 2] == b'/';

        let immer_weg = name == "script" || name == "style" || name == "noscript"
            || name == "template" || name == "svg";
        if !schliessend && !selbst_schliessend && !name.is_empty()
            && (immer_weg || (ohne_rahmen && RAHMEN.contains(&name.as_str()))) {
            // Bis zum passenden Ende springen; gleichnamige Verschachtelung
            // (nav in nav) wird mitgezaehlt.
            let auf = format!("<{}", name);
            let zu = format!("</{}", name);
            let lc = html[tag_ende..].to_ascii_lowercase();
            let mut tiefe = 1usize;
            let mut pos = 0usize;
            let mut ende = n;
            while tiefe > 0 {
                let naechst_zu = lc[pos..].find(&zu).map(|p| p + pos);
                let naechst_auf = lc[pos..].find(&auf).map(|p| p + pos)
                    .filter(|p| lc.as_bytes().get(p + auf.len()).map(|c| !c.is_ascii_alphanumeric()).unwrap_or(false));
                match (naechst_auf, naechst_zu) {
                    (Some(a), Some(z)) if a < z => { tiefe += 1; pos = a + auf.len(); }
                    (_, Some(z)) => {
                        tiefe -= 1;
                        pos = z + zu.len();
                        if tiefe == 0 {
                            ende = lc[pos..].find('>').map(|e| tag_ende + pos + e + 1).unwrap_or(n);
                        }
                    }
                    _ => { tiefe = 0; ende = n; }
                }
            }
            i = ende;
            text_start = i;
            roh.push('\n');
            continue;
        }
        if BLOCK.contains(&name.as_str()) { roh.push('\n'); } else { roh.push(' '); }
        i = tag_ende;
        text_start = i;
    }
    flush(&mut roh, text_start, n);

    let text = entities(&roh);
    let mut out: Vec<String> = Vec::new();
    let mut gesehen: HashSet<String> = HashSet::new();
    for zeile in text.lines() {
        let z = zeile.split_whitespace().collect::<Vec<_>>().join(" ");
        if z.chars().count() < min_zeichen.max(1) { continue; }
        if gesehen.insert(z.clone()) { out.push(z); }
    }
    out
}

// ══════════════════════════════════════════════════════════════════
// ADRESSEN UND DOMAINS
// ══════════════════════════════════════════════════════════════════

fn ohne_www(host: &str) -> &str { host.strip_prefix("www.").unwrap_or(host) }

/// Liegt der Host in einer der Domains? "example.org" erlaubt auch
/// "img.example.org", aber nicht "badexample.org".
pub fn host_erlaubt(host: &str, domains: &[String]) -> bool {
    let host = ohne_www(&host.to_lowercase()).to_string();
    domains.iter().any(|d| {
        let d = d.trim().trim_start_matches("https://").trim_start_matches("http://");
        let d = ohne_www(d.split('/').next().unwrap_or("")).to_lowercase();
        !d.is_empty() && (host == d || host.ends_with(&format!(".{}", d)))
    })
}

/// Die Allowlist aus einem Eingabefeld: Kommas, Leerzeichen, Zeilen.
pub fn allowlist_aus(text: &[String]) -> Vec<String> {
    text.iter()
        .flat_map(|t| t.split(|c: char| c == ',' || c.is_whitespace()))
        .map(|d| d.trim().to_string())
        .filter(|d| !d.is_empty())
        .collect()
}

/// Dateiendung aus dem Content-Type, sonst aus der Adresse (nur Bilder).
pub fn ext_aus_typ(content_type: &str, url: &str) -> Option<String> {
    let t = content_type.split(';').next().unwrap_or("").trim().to_lowercase();
    let aus_typ = match t.as_str() {
        "image/jpeg" | "image/jpg" => Some("jpg"),
        "image/png"  => Some("png"),
        "image/webp" => Some("webp"),
        "image/gif"  => Some("gif"),
        "image/bmp"  => Some("bmp"),
        _ => None,
    };
    if let Some(e) = aus_typ { return Some(e.to_string()); }
    let pfad = url.split('?').next().unwrap_or(url);
    let e = Path::new(pfad).extension()?.to_str()?.to_lowercase();
    if IMAGE_EXTS.contains(&e.as_str()) { Some(e) } else { None }
}

// ══════════════════════════════════════════════════════════════════
// LADEN
// ══════════════════════════════════════════════════════════════════

enum Holfehler { ZuGross, Fehlgeschlagen }

/// Laedt eine Adresse, aber nie mehr als `max_bytes`. Der Content-Length-Kopf
/// allein reicht nicht: Server lassen ihn weg, und dann stand frueher eine
/// 2-GB-Datei komplett im Speicher, bevor irgendetwas geprueft wurde.
async fn hole(client: &reqwest::Client, url: &Url, max_bytes: u64) -> Result<(String, Vec<u8>), Holfehler> {
    let mut antwort = client.get(url.clone()).send().await.map_err(|_| Holfehler::Fehlgeschlagen)?;
    if !antwort.status().is_success() { return Err(Holfehler::Fehlgeschlagen); }
    if antwort.content_length().map(|l| l > max_bytes).unwrap_or(false) { return Err(Holfehler::ZuGross); }
    let typ = antwort.headers().get("content-type")
        .and_then(|v| v.to_str().ok()).unwrap_or("").to_lowercase();
    let mut daten: Vec<u8> = Vec::new();
    while let Some(stueck) = antwort.chunk().await.map_err(|_| Holfehler::Fehlgeschlagen)? {
        daten.extend_from_slice(&stueck);
        if daten.len() as u64 > max_bytes { return Err(Holfehler::ZuGross); }
    }
    Ok((typ, daten))
}

/// robots.txt und Abstand je Server.
struct Hoeflichkeit {
    robots:  HashMap<String, String>,
    zuletzt: HashMap<String, std::time::Instant>,
}

impl Hoeflichkeit {
    fn new() -> Self { Hoeflichkeit { robots: HashMap::new(), zuletzt: HashMap::new() } }

    fn schluessel(url: &Url) -> String {
        format!("{}://{}", url.scheme(), url.host_str().unwrap_or(""))
            + &url.port().map(|p| format!(":{}", p)).unwrap_or_default()
    }

    async fn robots_fuer(&mut self, client: &reqwest::Client, url: &Url) -> String {
        let k = Self::schluessel(url);
        if let Some(r) = self.robots.get(&k) { return r.clone(); }
        let text = match Url::parse(&format!("{}/robots.txt", k)) {
            Ok(r) => match hole(client, &r, 512 * 1024).await {
                Ok((_, b)) => String::from_utf8_lossy(&b).to_string(),
                Err(_) => String::new(),   // keine robots.txt = keine Einschraenkung
            },
            Err(_) => String::new(),
        };
        self.robots.insert(k, text.clone());
        text
    }

    async fn darf(&mut self, client: &reqwest::Client, url: &Url) -> bool {
        let robots = self.robots_fuer(client, url).await;
        let pfad = match url.query() {
            Some(q) => format!("{}?{}", url.path(), q),
            None => url.path().to_string(),
        };
        robots_erlaubt(&robots, AGENT, &pfad)
    }

    /// Nicht prasseln: je Server mindestens eine Sekunde, oder was dessen
    /// robots.txt als Crawl-delay verlangt.
    async fn warte(&mut self, url: &Url) {
        let k = Self::schluessel(url);
        let abstand = self.robots.get(&k)
            .and_then(|r| robots_crawl_delay(r, AGENT))
            .map(|s| Duration::from_millis((s * 1000.0) as u64))
            .unwrap_or(Duration::from_millis(1000))
            .max(Duration::from_millis(1000));
        if let Some(t) = self.zuletzt.get(&k) {
            let vergangen = t.elapsed();
            if vergangen < abstand { tokio::time::sleep(abstand - vergangen).await; }
        }
        self.zuletzt.insert(k, std::time::Instant::now());
    }
}

// ══════════════════════════════════════════════════════════════════
// ABLEGEN
// ══════════════════════════════════════════════════════════════════

struct Ablage<'a> {
    dir:            &'a Path,
    art:            Art,
    opts:           &'a WebOptions,
    now:            String,
    bekannte_medien: HashSet<String>,
    bekannte_texte:  HashSet<String>,
}

impl<'a> Ablage<'a> {
    fn quelle(&self, url: &str, seite: Option<&str>) -> SampleSource {
        SampleSource {
            kind: "web".to_string(),
            origin: Some(url.to_string()),
            license: self.opts.license.clone().filter(|l| !l.trim().is_empty()),
            at: self.now.clone(),
            page: seite.filter(|s| *s != url).map(str::to_string),
        }
    }

    fn neue_id() -> String {
        format!("s_{}", &uuid::Uuid::new_v4().to_string().replace('-', "")[..10])
    }

    /// Ein Text als Sample. Gruppe ist die Seite: Absaetze eines Artikels
    /// duerfen beim Aufteilen nicht auf Train und Val verteilt werden, sonst
    /// misst die Validierung, wie gut das Modell den Artikel kennt.
    fn text(&mut self, text: String, url: &str, seite: Option<&str>, report: &mut FetchReport) -> Result<(), String> {
        let text = text.trim().to_string();
        if text.is_empty() { return Ok(()); }
        let hash = sha256_hex(text.as_bytes());
        if !self.bekannte_texte.insert(hash) { report.duplicates += 1; return Ok(()); }
        append_jsonl(&samples_path(self.dir), &StudioSample {
            id: Self::neue_id(), media: String::new(), mime: "text/plain".to_string(),
            content: Some(text), status: "new".to_string(), ann: Annotation::default(),
            src: self.quelle(url, seite),
            meta: SampleMeta { start: None, end: None, w: 0, h: 0, group: Some(seite.unwrap_or(url).to_string()) },
            abs_path: String::new(), doubt: None,
        })?;
        report.fetched += 1;
        Ok(())
    }

    /// Eine Mediendatei. Was sie ist, sagen die Bytes, nicht die Adresse.
    fn datei(&mut self, bytes: &[u8], url: &str, seite: Option<&str>, report: &mut FetchReport) -> Result<(), String> {
        let (ext, mime, w, h) = match self.art {
            Art::Bild => {
                let Some((w, h)) = image_dimensions(bytes) else { report.skipped_type.push(url.to_string()); return Ok(()) };
                if w.min(h) < self.opts.min_side { report.too_small += 1; return Ok(()); }
                let ext = image_ext_from_bytes(bytes);
                (ext.to_string(), mime_for(ext), w, h)
            }
            Art::Audio => {
                let Some(ext) = audio_ext_from_bytes(bytes) else { report.skipped_type.push(url.to_string()); return Ok(()) };
                (ext.to_string(), audio_mime(ext), 0, 0)
            }
            Art::Video => {
                let Some(ext) = video_ext_from_bytes(bytes) else { report.skipped_type.push(url.to_string()); return Ok(()) };
                (ext.to_string(), video_mime(ext), 0, 0)
            }
            Art::Text => { report.skipped_type.push(url.to_string()); return Ok(()); }
        };
        let hash = sha256_hex(bytes);
        let rel = format!("{}/{}.{}", &hash[..2], &hash, ext);
        if !self.bekannte_medien.insert(rel.clone()) { report.duplicates += 1; return Ok(()); }

        let ziel = self.dir.join("media").join(&rel);
        if let Some(parent) = ziel.parent() { fs::create_dir_all(parent).ok(); }
        fs::write(&ziel, bytes).map_err(|e| format!("Speichern: {}", e))?;

        append_jsonl(&samples_path(self.dir), &StudioSample {
            id: Self::neue_id(), media: rel, mime, content: None,
            status: "new".to_string(), ann: Annotation::default(),
            src: self.quelle(url, seite),
            // Die Fundseite ist die Gruppe: Bilder einer Seite sind sich oft
            // aehnlich und duerfen beim Split nicht auseinanderfallen. Frueher
            // war es die Domain — bei einem Crawl einer einzigen Website lag
            // dann alles in einer Gruppe, und der Split steckte alles in train.
            meta: SampleMeta { start: None, end: None, w, h, group: Some(seite.unwrap_or(url).to_string()) },
            abs_path: String::new(), doubt: None,
        })?;
        report.fetched += 1;
        Ok(())
    }
}

// ══════════════════════════════════════════════════════════════════
// DER LAUF
// ══════════════════════════════════════════════════════════════════

fn ist_html(typ: &str, bytes: &[u8]) -> bool {
    if typ.contains("html") || typ.contains("xhtml") { return true; }
    if !typ.is_empty() { return false; }
    let kopf = String::from_utf8_lossy(&bytes[..bytes.len().min(256)]).to_lowercase();
    kopf.trim_start().starts_with("<!doctype html") || kopf.contains("<html")
}

#[tauri::command]
pub async fn studio_fetch_cancel() -> Result<(), String> {
    ABBRUCH.store(true, Ordering::SeqCst);
    Ok(())
}

#[tauri::command]
pub async fn studio_fetch_web(
    app_handle: tauri::AppHandle, state: State<'_, AppState>,
    project_id: String, urls: Vec<String>, options: Option<WebOptions>,
) -> Result<FetchReport, String> {
    let user_id = get_user_id(&state)?;
    let dir = project_dir(&app_handle, &user_id, &project_id)?;
    let project = load_project(&dir)?;
    let mut opts = options.unwrap_or_default();
    opts.allowlist = allowlist_aus(&opts.allowlist);
    opts.max_pages = opts.max_pages.clamp(1, 5000);
    opts.max_files = opts.max_files.clamp(1, 20000);
    opts.max_depth = opts.max_depth.min(5);
    let art = art_von(&project);
    ABBRUCH.store(false, Ordering::SeqCst);

    let startadressen: Vec<Url> = urls.iter()
        .flat_map(|u| u.split_whitespace())
        .map(|u| u.trim().trim_end_matches(','))
        .filter(|u| u.starts_with("http://") || u.starts_with("https://"))
        .filter_map(|u| Url::parse(u).ok())
        .collect();
    if startadressen.is_empty() { return Err("Keine gültigen Adressen (http:// oder https://)".to_string()); }

    let client = reqwest::Client::builder()
        .user_agent(format!("{}/1.0 (lokales Werkzeug)", AGENT))
        .timeout(Duration::from_secs(30))
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()
        .map_err(|e| format!("HTTP-Client: {}", e))?;

    // Ohne eigene Allowlist bleibt ein Crawl auf den Domains der Startadressen.
    // Bei einzelnen Adressen gilt nur eine ausdrueckliche Allowlist.
    let domains: Vec<String> = if !opts.allowlist.is_empty() {
        opts.allowlist.clone()
    } else if opts.mode != "urls" {
        startadressen.iter().filter_map(|u| u.host_str().map(|h| ohne_www(h).to_string())).collect()
    } else {
        vec![]
    };
    let seite_erlaubt = |u: &Url| domains.is_empty() || u.host_str().map(|h| host_erlaubt(h, &domains)).unwrap_or(false);
    // Eingebettete Dateien liegen oft auf einem CDN mit anderer Domain; fuer
    // sie gilt nur eine ausdrueckliche Allowlist.
    let datei_erlaubt = |u: &Url| opts.allowlist.is_empty()
        || u.host_str().map(|h| host_erlaubt(h, &opts.allowlist)).unwrap_or(false);

    let existing = load_samples(&dir);
    let mut ablage = Ablage {
        dir: &dir, art, opts: &opts, now: Utc::now().to_rfc3339(),
        bekannte_medien: existing.iter().filter(|s| !s.media.is_empty()).map(|s| s.media.clone()).collect(),
        bekannte_texte: existing.iter().filter_map(|s| s.content.as_ref().map(|c| sha256_hex(c.as_bytes()))).collect(),
    };
    drop(existing);

    let mut hoeflich = Hoeflichkeit::new();
    let mut report = FetchReport::default();
    let max_bytes = (opts.max_mb.max(0.1) * 1024.0 * 1024.0) as u64;
    // Seiten selbst duerfen groesser sein als ein kleines Dateilimit — eine
    // Nachrichtenseite hat schnell 2 MB HTML.
    let max_seite = max_bytes.max(8 * 1024 * 1024);

    let melde = |report: &FetchReport, url: &str| {
        let _ = app_handle.emit("studio-fetch-progress", serde_json::json!({
            "project_id": project_id,
            "current": report.fetched, "total": opts.max_files,
            "pages": report.pages_visited, "max_pages": opts.max_pages,
            "url": url,
        }));
    };

    // (Adresse, Tiefe, Links folgen?)
    let mut schlange: VecDeque<(Url, usize, bool)> = VecDeque::new();
    let mut besucht: HashSet<String> = HashSet::new();
    let mut dateien_gesehen: HashSet<String> = HashSet::new();

    if opts.mode == "sitemap" {
        let mut sitemaps: VecDeque<Url> = VecDeque::new();
        for s in &startadressen {
            if s.path().ends_with(".xml") || s.path().ends_with(".xml.gz") {
                sitemaps.push_back(s.clone());
            } else {
                let robots = hoeflich.robots_fuer(&client, s).await;
                let aus_robots = robots_sitemaps(&robots);
                if aus_robots.is_empty() {
                    if let Ok(u) = s.join("/sitemap.xml") { sitemaps.push_back(u); }
                } else {
                    sitemaps.extend(aus_robots.iter().filter_map(|u| Url::parse(u).ok()));
                }
            }
        }
        let mut gelesen = 0usize;
        while let Some(sm) = sitemaps.pop_front() {
            if gelesen >= 25 || schlange.len() >= opts.max_pages || ABBRUCH.load(Ordering::SeqCst) { break; }
            gelesen += 1;
            if !hoeflich.darf(&client, &sm).await { report.blocked.push(sm.to_string()); continue; }
            hoeflich.warte(&sm).await;
            let Ok((_, bytes)) = hole(&client, &sm, 32 * 1024 * 1024).await else {
                report.failed.push(sm.to_string()); continue;
            };
            let (locs, ist_index) = sitemap_locs(&String::from_utf8_lossy(&bytes));
            for loc in locs {
                let Ok(u) = Url::parse(&loc) else { continue };
                if ist_index { sitemaps.push_back(u); continue; }
                if !seite_erlaubt(&u) { report.outside_allowlist += 1; continue; }
                if schlange.len() >= opts.max_pages { report.limit_reached = true; break; }
                if besucht.insert(u.to_string()) { schlange.push_back((u, 0, false)); }
            }
        }
        if schlange.is_empty() && report.failed.is_empty() && report.blocked.is_empty() {
            return Err("In der Sitemap standen keine Seiten.".to_string());
        }
    } else {
        for s in &startadressen {
            if besucht.insert(s.to_string()) { schlange.push_back((s.clone(), 0, opts.mode == "crawl")); }
        }
    }

    'lauf: while let Some((url, tiefe, folgen)) = schlange.pop_front() {
        if ABBRUCH.load(Ordering::SeqCst) { report.cancelled = true; break; }
        if report.fetched >= opts.max_files { report.limit_reached = true; break; }
        if !seite_erlaubt(&url) { report.outside_allowlist += 1; continue; }
        if !hoeflich.darf(&client, &url).await { report.blocked.push(url.to_string()); continue; }
        hoeflich.warte(&url).await;
        melde(&report, url.as_str());

        let (typ, bytes) = match hole(&client, &url, max_seite).await {
            Ok(x) => x,
            Err(Holfehler::ZuGross) => { report.too_large.push(url.to_string()); continue; }
            Err(Holfehler::Fehlgeschlagen) => { report.failed.push(url.to_string()); continue; }
        };

        if !ist_html(&typ, &bytes) {
            // Eine Datei direkt: Text als Text, alles andere nach den Bytes.
            if art == Art::Text {
                if typ.starts_with("text/") || typ.is_empty() {
                    let inhalt = String::from_utf8_lossy(&bytes).to_string();
                    let teile: Vec<String> = if opts.paragraphs {
                        inhalt.split("\n\n").map(|t| t.split_whitespace().collect::<Vec<_>>().join(" "))
                            .filter(|t| t.chars().count() >= opts.min_chars).collect()
                    } else { vec![inhalt] };
                    for t in teile { ablage.text(t, url.as_str(), None, &mut report)?; }
                } else {
                    report.skipped_type.push(url.to_string());
                }
            } else if bytes.len() as u64 > max_bytes {
                report.too_large.push(url.to_string());
            } else {
                ablage.datei(&bytes, url.as_str(), None, &mut report)?;
            }
            continue;
        }

        // Eine Seite.
        report.pages_visited += 1;
        let html = String::from_utf8_lossy(&bytes).to_string();
        let seite = url.to_string();

        if art == Art::Text {
            if opts.paragraphs {
                for block in textbloecke(&html, opts.min_chars, true) {
                    ablage.text(block, &seite, Some(&seite), &mut report)?;
                    if report.fetched >= opts.max_files { report.limit_reached = true; break 'lauf; }
                }
            } else {
                ablage.text(textbloecke(&html, 1, true).join("\n"), &seite, None, &mut report)?;
            }
        }

        let f = funde(&html, &url);
        let dateien = match art {
            Art::Bild => f.bilder.clone(),
            Art::Audio => f.audio.clone(),
            Art::Video => f.video.clone(),
            Art::Text => vec![],
        };
        if art != Art::Text && dateien.is_empty() && !folgen {
            report.skipped_type.push(seite.clone());
        }
        for datei in dateien {
            if ABBRUCH.load(Ordering::SeqCst) { report.cancelled = true; break 'lauf; }
            if report.fetched >= opts.max_files { report.limit_reached = true; break 'lauf; }
            if !dateien_gesehen.insert(datei.to_string()) { continue; }
            if !datei_erlaubt(&datei) { report.outside_allowlist += 1; continue; }
            if !hoeflich.darf(&client, &datei).await { report.blocked.push(datei.to_string()); continue; }
            hoeflich.warte(&datei).await;
            melde(&report, datei.as_str());
            match hole(&client, &datei, max_bytes).await {
                Ok((_, b)) => ablage.datei(&b, datei.as_str(), Some(&seite), &mut report)?,
                Err(Holfehler::ZuGross) => report.too_large.push(datei.to_string()),
                Err(Holfehler::Fehlgeschlagen) => report.failed.push(datei.to_string()),
            }
        }

        if folgen && tiefe < opts.max_depth {
            for s in f.seiten {
                if !seite_erlaubt(&s) { report.outside_allowlist += 1; continue; }
                if besucht.len() >= opts.max_pages { report.limit_reached = true; break; }
                if besucht.insert(s.to_string()) { schlange.push_back((s, tiefe + 1, true)); }
            }
        }
        if report.pages_visited >= opts.max_pages && !schlange.is_empty() {
            report.limit_reached = true;
            break;
        }
    }

    let _ = app_handle.emit("studio-fetch-progress", serde_json::json!({
        "project_id": project_id, "current": report.fetched, "total": opts.max_files,
        "pages": report.pages_visited, "max_pages": opts.max_pages, "done": true,
    }));
    if report.fetched > 0 {
        let mut p = project;
        p.updated_at = Utc::now().to_rfc3339();
        let _ = save_project(&dir, &p);
    }
    Ok(report)
}

// ══════════════════════════════════════════════════════════════════
// TESTS
// ══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn url(s: &str) -> Url { Url::parse(s).unwrap() }

    #[test]
    fn bilder_einer_seite_werden_gefunden() {
        let html = r#"<html><head><meta property="og:image" content="/og.jpg">
            <base href="https://cdn.example.org/assets/"></head><body>
            <img src="a.jpg" alt="x"><img data-src='lazy.png'>
            <img srcset="klein.jpg 480w, gross.jpg 1600w">
            <picture><source srcset="bild.webp 1x, bild@2x.webp 2x" type="image/webp"></picture>
            <a href="/foto.jpeg">Foto</a><a href="../seite.html">Weiter</a>
            <img src="data:image/png;base64,AAA"><img src="icon.svg">
            </body></html>"#;
        let f = funde(html, &url("https://example.org/galerie/"));
        let bilder: Vec<String> = f.bilder.iter().map(|u| u.to_string()).collect();
        assert!(bilder.contains(&"https://cdn.example.org/og.jpg".to_string()), "{:?}", bilder);
        assert!(bilder.contains(&"https://cdn.example.org/assets/a.jpg".to_string()), "{:?}", bilder);
        assert!(bilder.contains(&"https://cdn.example.org/assets/lazy.png".to_string()), "{:?}", bilder);
        assert!(bilder.contains(&"https://cdn.example.org/assets/gross.jpg".to_string()), "{:?}", bilder);
        assert!(!bilder.iter().any(|b| b.ends_with("klein.jpg")), "kleinste srcset-Variante statt groesster");
        assert!(bilder.contains(&"https://cdn.example.org/assets/bild@2x.webp".to_string()), "{:?}", bilder);
        assert!(bilder.contains(&"https://cdn.example.org/foto.jpeg".to_string()), "{:?}", bilder);
        assert!(!bilder.iter().any(|b| b.starts_with("data:") || b.ends_with(".svg")));
        assert_eq!(f.seiten.iter().map(|u| u.to_string()).collect::<Vec<_>>(),
            vec!["https://cdn.example.org/seite.html".to_string()]);
    }

    #[test]
    fn video_und_audio_quellen_nach_ihrem_tag() {
        let html = r#"<video poster="p.jpg"><source src="clip" type="video/mp4"><source src="clip2.webm"></video>
            <audio controls><source src="ton"></audio><a href="song.mp3">mp3</a>"#;
        let f = funde(html, &url("https://x.de/"));
        assert_eq!(f.video.len(), 2, "{:?}", f.video);
        assert_eq!(f.audio.iter().map(|u| u.path().to_string()).collect::<Vec<_>>(), vec!["/ton", "/song.mp3"]);
    }

    #[test]
    fn nofollow_und_fragmente() {
        let html = r##"<a href="/a#oben">a</a><a href="/a#unten">a2</a><a rel="nofollow" href="/b">b</a>
            <a href="mailto:x@y.de">m</a><a href="javascript:void(0)">j</a>"##;
        let f = funde(html, &url("https://x.de/"));
        assert_eq!(f.seiten.iter().map(|u| u.to_string()).collect::<Vec<_>>(), vec!["https://x.de/a".to_string()]);
    }

    #[test]
    fn absaetze_ohne_navigation_und_fuss() {
        let html = r#"<html><body><header><nav><a>Start</a><a>Impressum und Kontaktformular</a></nav></header>
            <article><h1>Der Lift am Berg</h1>
            <p>Der Sessellift f&auml;hrt ab acht Uhr &amp; bringt bis zu 2400 Personen je Stunde nach oben.</p>
            <p>Kurz.</p>
            <div>Bei Sturm bleibt die Anlage geschlossen, das zeigt die Tafel an der Talstation.</div>
            <script>var x = "<p>kein Absatz aus einem Skript, auch wenn er lang genug waere</p>";</script>
            </article><footer><p>Alle Rechte vorbehalten, Musterfirma GmbH, Musterstrasse 1</p></footer></body></html>"#;
        let b = textbloecke(html, 30, true);
        assert_eq!(b.len(), 2, "{:?}", b);
        assert!(b[0].starts_with("Der Sessellift fährt ab acht Uhr & bringt"), "{:?}", b);
        assert!(b[1].starts_with("Bei Sturm"), "{:?}", b);
        // Mit Rahmen kommt der Fuss dazu, das Skript nie.
        let mit = textbloecke(html, 30, false);
        assert!(mit.iter().any(|t| t.contains("Musterfirma")), "{:?}", mit);
        assert!(!mit.iter().any(|t| t.contains("Skript")), "{:?}", mit);
    }

    #[test]
    fn verschachtelte_navigation_wird_ganz_entfernt() {
        let html = "<nav><nav><p>innen innen innen innen innen</p></nav><p>aussen aussen aussen aussen</p></nav>\
                    <p>Inhalt Inhalt Inhalt Inhalt Inhalt</p>";
        assert_eq!(textbloecke(html, 10, true), vec!["Inhalt Inhalt Inhalt Inhalt Inhalt".to_string()]);
    }

    #[test]
    fn entities_numerisch_und_benannt() {
        assert_eq!(entities("a &amp; b &#228; &#x00FC; &lt;x&gt; &unbekannt; &"), "a & b ä ü <x> &unbekannt; &");
    }

    #[test]
    fn sitemap_und_verzeichnis() {
        let xml = "<?xml version=\"1.0\"?><urlset><url><loc>https://x.de/a</loc></url>\
                   <url><loc> <![CDATA[https://x.de/b?x=1&amp;y=2]]> </loc></url></urlset>";
        assert_eq!(sitemap_locs(xml), (vec!["https://x.de/a".to_string(), "https://x.de/b?x=1&y=2".to_string()], false));
        let idx = "<sitemapindex><sitemap><loc>https://x.de/s1.xml</loc></sitemap></sitemapindex>";
        assert_eq!(sitemap_locs(idx), (vec!["https://x.de/s1.xml".to_string()], true));
    }

    #[test]
    fn allowlist_mit_subdomains_aber_ohne_namensvettern() {
        let d = allowlist_aus(&["example.org, https://www.wiki.de/pfad".to_string()]);
        assert_eq!(d, vec!["example.org".to_string(), "https://www.wiki.de/pfad".to_string()]);
        assert!(host_erlaubt("example.org", &d));
        assert!(host_erlaubt("www.example.org", &d));
        assert!(host_erlaubt("img.example.org", &d));
        assert!(!host_erlaubt("badexample.org", &d));
        assert!(host_erlaubt("wiki.de", &d));
        assert!(!host_erlaubt("anderswo.de", &d));
    }

    #[test]
    fn crawl_delay_und_sitemaps_aus_robots() {
        let robots = "Sitemap: https://x.de/sm.xml\nUser-agent: *\nCrawl-delay: 4\nDisallow: /intern\n\
                      User-agent: gierig\nCrawl-delay: 3600\n";
        assert_eq!(robots_crawl_delay(robots, AGENT), Some(4.0));
        assert_eq!(robots_crawl_delay(robots, "gierig"), Some(10.0), "gedeckelt");
        assert_eq!(robots_sitemaps(robots), vec!["https://x.de/sm.xml".to_string()]);
        assert!(!robots_erlaubt(robots, AGENT, "/intern/a"));
    }

    #[test]
    fn html_wird_an_bytes_erkannt_wenn_der_typ_fehlt() {
        assert!(ist_html("text/html; charset=utf-8", b""));
        assert!(ist_html("", b"  <!DOCTYPE html><html>"));
        assert!(!ist_html("image/jpeg", b"<html>"));
        assert!(!ist_html("", &[0xFF, 0xD8, 0xFF]));
    }
}
