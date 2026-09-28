// Auswahl des Python-Interpreters — eine Stelle fuer die ganze App.
//
// Bisher lag dieselbe Suche viermal im Code (Training, Dev-Training, Tests,
// Labor) und der Erststart nahm zusaetzlich einfach das erste `python3` aus dem
// PATH. Auf einem Rechner mit mehreren Interpretern hiess das: die Einrichtung
// installiert die Pakete in Interpreter A, trainiert wird aber mit B.
//
// Regeln:
//  - Unterstuetzt ist Python 3.10 bis 3.13. torch, transformers 5, peft,
//    accelerate, datasets und sentence-transformers verlangen >= 3.10; 3.9 bekaeme
//    nur alte Versionen, gegen die die Plugins nie getestet wurden.
//  - Unter den unterstuetzten gewinnt einer mit funktionierendem torch, sonst
//    die hoechste Version. Ein altes 3.9 mit torch schlaegt KEIN neues 3.12 —
//    sonst bliebe die App nach "bitte Python 3.12 installieren" beim alten.
//  - Eine Mac-App aus dem Finder hat einen kurzen PATH (ohne Homebrew). Darum
//    werden die ueblichen Installationsorte direkt abgesucht: Homebrew,
//    python.org (/Library/Frameworks), unter Windows der py-Starter und
//    %LOCALAPPDATA%\Programs\Python.
//  - /usr/bin/python3 ist auf einem Mac ohne Entwicklerwerkzeuge nur ein
//    Platzhalter, der beim Aufruf einen Installationsdialog oeffnet. Er wird nur
//    gefragt, wenn die Command Line Tools wirklich da sind.

use std::process::Command;
use crate::command_ext::NoWindow;

/// Kleinste unterstuetzte Version.
pub const MIN_SUPPORTED: (u32, u32) = (3, 10);
/// Groesste getestete Version. Neuere werden genommen, aber erst danach.
pub const MAX_TESTED: (u32, u32) = (3, 13);

/// Version aus der Ausgabe von `python --version` ("Python 3.11.1").
pub fn parse_version(s: &str) -> Option<(u32, u32, u32)> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() < 2 { return None; }
    let nums: Vec<&str> = parts[1].split('.').collect();
    if nums.len() < 2 { return None; }
    let major = nums[0].parse::<u32>().ok()?;
    let minor = nums[1].parse::<u32>().ok()?;
    let patch = nums.get(2)
        .and_then(|p| p.trim_end_matches(|c: char| !c.is_ascii_digit()).parse::<u32>().ok())
        .unwrap_or(0);
    Some((major, minor, patch))
}

/// Liegt die Version im unterstuetzten Bereich (>= 3.10)?
pub fn is_supported(v: (u32, u32, u32)) -> bool {
    v.0 == 3 && (v.0, v.1) >= MIN_SUPPORTED
}

#[derive(Clone, Debug)]
struct Candidate {
    path: String,
    version: (u32, u32, u32),
}

fn version_of(cmd: &str) -> Option<(u32, u32, u32)> {
    let out = Command::new(cmd).no_window().arg("--version").output().ok()?;
    if !out.status.success() { return None; }
    // Aeltere Versionen schreiben nach stderr statt stdout
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    parse_version(&combined)
}

/// Auf dem Mac: sind die Command Line Tools (oder Xcode) installiert? Ohne sie
/// ist /usr/bin/python3 nur der Platzhalter mit Installationsdialog.
fn macos_clt_present() -> bool {
    std::path::Path::new("/Library/Developer/CommandLineTools/usr/bin/python3").exists()
        || std::path::Path::new("/Applications/Xcode.app").exists()
}

/// Pfade aus `py -0p` (" -V:3.12 *   C:\Program Files\Python312\python.exe").
/// Der Pfad beginnt beim Laufwerksbuchstaben und darf Leerzeichen enthalten.
fn parse_py_launcher(out: &str) -> Vec<String> {
    out.lines()
        .filter_map(|line| {
            let pos = line.find(":\\")?;
            if pos == 0 { return None; }
            let p = line[pos - 1..].trim();
            p.to_lowercase().ends_with(".exe").then(|| p.to_string())
        })
        .collect()
}

/// Alle Pfade, die als Interpreter in Frage kommen (noch ohne Versionspruefung).
fn candidate_paths() -> Vec<String> {
    let mut paths: Vec<String> = Vec::new();
    let names = ["python3.13", "python3.12", "python3.11", "python3.10", "python3"];

    if cfg!(target_os = "windows") {
        // py-Starter: listet alle installierten Interpreter mit Pfad ("-V:3.12 *  C:\...\python.exe")
        if let Ok(out) = Command::new("py").no_window().arg("-0p").output() {
            paths.extend(parse_py_launcher(&String::from_utf8_lossy(&out.stdout)));
        }
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            let base = std::path::Path::new(&local).join("Programs").join("Python");
            if let Ok(entries) = std::fs::read_dir(&base) {
                for e in entries.flatten() {
                    let exe = e.path().join("python.exe");
                    if exe.exists() { paths.push(exe.to_string_lossy().to_string()); }
                }
            }
        }
        paths.push("python".to_string());
        paths.push("python3".to_string());
        return paths;
    }

    let clt = !cfg!(target_os = "macos") || macos_clt_present();
    let mut bases: Vec<String> = vec!["/opt/homebrew/bin".into(), "/usr/local/bin".into()];
    // python.org-Installer: /Library/Frameworks/Python.framework/Versions/3.x/bin
    if cfg!(target_os = "macos") {
        if let Ok(entries) = std::fs::read_dir("/Library/Frameworks/Python.framework/Versions") {
            let mut vers: Vec<String> = entries.flatten()
                .map(|e| e.path().join("bin").to_string_lossy().to_string())
                .collect();
            vers.sort();
            bases.extend(vers);
        }
    }
    if clt { bases.push("/usr/bin".into()); }
    for base in &bases {
        for name in &names {
            paths.push(format!("{}/{}", base, name));
        }
    }
    // Zuletzt, was der PATH hergibt (Linux, pyenv, conda). "python3" aus einem
    // Finder-PATH waere auf dem Mac wieder /usr/bin/python3 — nur mit CLT.
    if clt {
        paths.push("python3".to_string());
        paths.push("python".to_string());
    }
    paths
}

fn candidates() -> Vec<Candidate> {
    let mut found: Vec<Candidate> = Vec::new();
    for p in candidate_paths() {
        // Nicht existierende absolute Pfade gar nicht erst starten.
        if p.contains('/') || p.contains('\\') {
            if !std::path::Path::new(&p).exists() { continue; }
        }
        if let Some(v) = version_of(&p) {
            found.push(Candidate { path: p, version: v });
        }
    }
    // Nur echte Duplikate (Symlink auf dieselbe Binaerdatei) entfernen — Version
    // allein reicht nicht: /opt/homebrew und /usr/local koennen dieselbe Version
    // mit unterschiedlichen site-packages haben (nur eine davon hat torch).
    let mut seen: Vec<std::path::PathBuf> = Vec::new();
    found.retain(|c| {
        let key = std::fs::canonicalize(&c.path).unwrap_or_else(|_| std::path::PathBuf::from(&c.path));
        if seen.contains(&key) { false } else { seen.push(key); true }
    });
    found
}

/// Reihenfolge der Kandidaten: unterstuetzt und getestet (3.10-3.13, hoechste
/// zuerst), dann neuere als getestet, dann zu alte.
fn rank(list: &mut Vec<Candidate>) {
    list.sort_by_key(|c| {
        let mm = (c.version.0, c.version.1);
        let group = if !is_supported(c.version) { 2 } else if mm > MAX_TESTED { 1 } else { 0 };
        (group, std::cmp::Reverse(c.version))
    });
}

/// Auswahlregel ohne Prozessstarts — `has_torch` wird nur fuer unterstuetzte
/// Kandidaten gefragt. Getrennt, damit sie testbar ist.
fn choose(mut list: Vec<Candidate>, has_torch: &dyn Fn(&str) -> bool) -> Option<Candidate> {
    rank(&mut list);
    if let Some(c) = list.iter().filter(|c| is_supported(c.version)).find(|c| has_torch(&c.path)) {
        return Some(c.clone());
    }
    list.into_iter().next()
}

fn torch_works(path: &str) -> bool {
    // torch + torchvision/torchaudio (falls installiert) muessen zusammenpassen
    let torch_check = "import torch\nfor _m in ('torchvision', 'torchaudio'):\n    try:\n        __import__(_m)\n    except ImportError:\n        pass";
    let ok = Command::new(path).no_window().args(["-c", torch_check]).output()
        .map(|o| o.status.success()).unwrap_or(false);
    if ok { return true; }
    let bare = Command::new(path).no_window().args(["-c", "import torch"]).output()
        .map(|o| o.status.success()).unwrap_or(false);
    if bare {
        println!("[Python] torchvision/torchaudio defekt oder inkompatibel bei {} — Fix: {} -m pip install --upgrade torch torchvision torchaudio", path, path);
    }
    bare
}

fn fallback() -> String {
    if cfg!(target_os = "windows") { "python".to_string() } else { "python3".to_string() }
}

pub fn resolve_python() -> String {
    choose(candidates(), &torch_works).map(|c| c.path).unwrap_or_else(fallback)
}

/// Wie `resolve_python`, zusaetzlich die Versionsnummer ("3.11.1") des
/// gewaehlten Interpreters — fuer den System-Check beim Erststart.
pub fn resolve_python_with_version() -> (Option<String>, Option<String>) {
    let path = resolve_python();
    let version = version_of(&path).map(|(a, b, c)| format!("{}.{}.{}", a, b, c));
    if version.is_none() {
        // Interpreter laesst sich nicht starten
        return (None, None);
    }
    (Some(path), version)
}

/// Wie man ein passendes Python bekommt — je Betriebssystem.
pub fn install_hint() -> String {
    if cfg!(target_os = "macos") {
        "Installiere Python 3.12 von https://www.python.org/downloads/macos/ (oder mit Homebrew: brew install python@3.12) und starte FrameTrain neu.".to_string()
    } else if cfg!(target_os = "windows") {
        "Installiere Python 3.12 von https://www.python.org/downloads/windows/ (Haken bei \"Add python.exe to PATH\" setzen) und starte FrameTrain neu.".to_string()
    } else {
        "Installiere Python 3.12 über die Paketverwaltung (z. B. sudo apt install python3.12 python3.12-venv python3-pip) und starte FrameTrain neu.".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(path: &str, v: (u32, u32, u32)) -> Candidate { Candidate { path: path.into(), version: v } }

    #[test]
    fn neues_python_schlaegt_altes_mit_torch() {
        // Genau der Fall "altes 3.9 hat torch, Nutzer installiert auf Hinweis 3.12"
        let list = vec![c("/usr/bin/python3", (3, 9, 6)), c("/usr/local/bin/python3.12", (3, 12, 4))];
        let torch = |p: &str| p == "/usr/bin/python3";
        assert_eq!(choose(list, &torch).unwrap().path, "/usr/local/bin/python3.12");
    }

    #[test]
    fn torch_entscheidet_unter_den_unterstuetzten() {
        let list = vec![c("a3.13", (3, 13, 0)), c("b3.11", (3, 11, 1))];
        let torch = |p: &str| p == "b3.11";
        assert_eq!(choose(list, &torch).unwrap().path, "b3.11");
    }

    #[test]
    fn getestete_version_vor_neuerer() {
        let list = vec![c("neu", (3, 14, 0)), c("getestet", (3, 12, 0))];
        assert_eq!(choose(list, &|_| false).unwrap().path, "getestet");
    }

    #[test]
    fn nur_zu_altes_python_wird_trotzdem_gemeldet() {
        // Der System-Check braucht einen Pfad, um "zu alt" sagen zu koennen.
        let list = vec![c("/usr/bin/python3", (3, 9, 6))];
        let chosen = choose(list, &|_| true).unwrap();
        assert!(!is_supported(chosen.version));
    }

    #[test]
    fn py_starter_mit_leerzeichen_im_pfad() {
        let out = " -V:3.12 *        C:\\Program Files\\Python312\\python.exe\n -V:3.10          C:\\Users\\k\\AppData\\Local\\Programs\\Python\\Python310\\python.exe\n";
        assert_eq!(parse_py_launcher(out), vec![
            "C:\\Program Files\\Python312\\python.exe".to_string(),
            "C:\\Users\\k\\AppData\\Local\\Programs\\Python\\Python310\\python.exe".to_string(),
        ]);
    }

    #[test]
    fn versionsausgabe() {
        assert_eq!(parse_version("Python 3.12.1"), Some((3, 12, 1)));
        assert_eq!(parse_version("Python 3.13.0rc1"), Some((3, 13, 0)));
        assert!(is_supported((3, 10, 0)) && !is_supported((3, 9, 18)));
    }
}
