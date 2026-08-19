// Auswahl des Python-Interpreters — eine Stelle fuer die ganze App.
//
// Bisher lag dieselbe Suche viermal im Code (Training, Dev-Training, Tests,
// Labor) und der Erststart nahm zusaetzlich einfach das erste `python3` aus dem
// PATH. Auf einem Rechner mit mehreren Interpretern hiess das: die Einrichtung
// installiert die Pakete in Interpreter A, trainiert wird aber mit B.
//
// Regel: hoechste Version zuerst, aber ein Interpreter mit funktionierendem
// torch schlaegt eine hoehere Version ohne torch.

use std::process::Command;

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

struct Candidate {
    path: String,
    version: (u32, u32, u32),
}

fn version_of(cmd: &str) -> Option<(u32, u32, u32)> {
    let out = Command::new(cmd).arg("--version").output().ok()?;
    if !out.status.success() { return None; }
    // Aeltere Versionen schreiben nach stderr statt stdout
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    parse_version(&combined)
}

fn candidates() -> Vec<Candidate> {
    let mut found: Vec<Candidate> = Vec::new();

    if !cfg!(target_os = "windows") {
        for base in &["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"] {
            for name in &["python3.13", "python3.12", "python3.11", "python3.10", "python3.9", "python3"] {
                let full = format!("{}/{}", base, name);
                if let Some(v) = version_of(&full) {
                    found.push(Candidate { path: full, version: v });
                }
            }
        }
    }
    for cmd in &["python3", "python"] {
        if let Some(v) = version_of(cmd) {
            found.push(Candidate { path: cmd.to_string(), version: v });
        }
    }

    found.sort_by(|a, b| b.version.cmp(&a.version));
    // Nur echte Duplikate (Symlink auf dieselbe Binaerdatei) entfernen — Version
    // allein reicht nicht: /opt/homebrew und /usr/local koennen dieselbe Version
    // mit unterschiedlichen site-packages haben (nur eine davon hat torch).
    found.dedup_by(|a, b| {
        match (std::fs::canonicalize(&a.path), std::fs::canonicalize(&b.path)) {
            (Ok(x), Ok(y)) => x == y,
            _ => a.path == b.path,
        }
    });
    found
}

fn fallback() -> String {
    if cfg!(target_os = "windows") { "python".to_string() } else { "python3".to_string() }
}

/// Der Interpreter, mit dem die App arbeitet — Training, Tests, Labor und
/// Paketinstallation nutzen denselben.
pub fn resolve_python() -> String {
    let list = candidates();

    // torch + torchvision/torchaudio (falls installiert) muessen zusammenpassen
    let torch_check = "import torch\nfor _m in ('torchvision', 'torchaudio'):\n    try:\n        __import__(_m)\n    except ImportError:\n        pass";
    for c in &list {
        let ok = Command::new(&c.path).args(["-c", torch_check]).output()
            .map(|o| o.status.success()).unwrap_or(false);
        if ok { return c.path.clone(); }
    }
    // Fallback: torch vorhanden, torchvision/torchaudio defekt
    for c in &list {
        let ok = Command::new(&c.path).args(["-c", "import torch"]).output()
            .map(|o| o.status.success()).unwrap_or(false);
        if ok {
            println!("[Python] torchvision/torchaudio defekt oder inkompatibel bei {} — Fix: pip install --upgrade torch torchvision torchaudio", c.path);
            return c.path.clone();
        }
    }
    list.first().map(|c| c.path.clone()).unwrap_or_else(fallback)
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
