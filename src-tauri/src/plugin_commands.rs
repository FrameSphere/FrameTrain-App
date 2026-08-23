// plugin_commands.rs
// Verwaltet First-Launch-Check, Python-Dependency-Installation und YOLO-Inferenz.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::io::{BufRead, BufReader};
use std::time::{Instant, Duration};
use tauri::{AppHandle, Window};
use tauri::Emitter;

// ══════════════════════════════════════════════════════════════════
// PRE-FLIGHT TYPEN
// ══════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PreFlightCheck {
    pub ok: bool,
    pub python_found: bool,
    pub python_version: Option<String>,
    pub python_version_ok: bool,   // >= 3.8
    pub pip_found: bool,
    pub free_gb: f64,
    pub free_gb_ok: bool,          // >= 6 GB
    pub gpu_info: GpuInfo,
    pub platform: String,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GpuInfo {
    pub has_nvidia_gpu: bool,
    pub cuda_available: bool,
    pub cuda_version: Option<String>,
    pub gpu_name: Option<String>,
    pub recommended_torch_index: String,  // z.B. "cu121" oder "cpu"
}

// ══════════════════════════════════════════════════════════════════
// TYPEN
// ══════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PluginInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub category: String,
    pub icon: String,
    pub built_in: bool,
    pub required_packages: Vec<String>,
    pub optional_packages: Vec<String>,
    pub estimated_size_mb: i32,
    pub install_time_minutes: i32,
    pub priority: i32,
    #[serde(default)] pub is_selected: bool,
    #[serde(default)] pub is_installed: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PluginInstallProgress {
    pub plugin_id: String,
    pub status: String,
    pub message: String,
    pub progress: Option<i32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DependencyStatus {
    pub package: String,
    pub installed: bool,
    pub version: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct YoloDetection {
    pub label: String,
    pub confidence: f32,
    pub bbox: [f32; 4], // x1, y1, x2, y2
}

#[derive(Debug, Serialize, Deserialize)]
pub struct YoloInferenceResult {
    pub detections: Vec<YoloDetection>,
    pub inference_time_ms: f64,
    pub image_path: String,
}

// ══════════════════════════════════════════════════════════════════
// PYTHON HELPERS
// ══════════════════════════════════════════════════════════════════

fn verify_python_available() -> Result<(), String> {
    let candidates: Vec<&str> = if cfg!(target_os = "windows") {
        vec!["python", "python3"]
    } else {
        vec!["python3", "python"]
    };
    for cmd in &candidates {
        if let Ok(out) = Command::new(cmd).arg("--version").output() {
            if out.status.success() {
                let version = String::from_utf8_lossy(&out.stdout);
                println!("[Deps] Python gefunden: {} ({})", cmd, version.trim());
                return Ok(());
            }
        }
    }
    Err("Python ist nicht installiert oder nicht im PATH verfügbar. Bitte installiere Python 3.8+ von python.org.".to_string())
}

fn get_python_executable() -> String {
    // Derselbe Interpreter, mit dem spaeter trainiert und getestet wird —
    // sonst landen die Pakete in einem anderen Python als die Engines nutzen.
    crate::python_env::resolve_python()
}

/// Prueft ob eine installierte Version eine Obergrenze ueberschreitet (z.B. numpy < 2.0.0).
/// Simpler Parser fuer "major.minor.patch" Versions-Strings.
fn version_exceeds_max(version: &str, max_major: u32, max_minor: u32) -> bool {
    let parts: Vec<u32> = version.split('.')
        .filter_map(|p| p.chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().ok())
        .collect();
    let major = parts.first().copied().unwrap_or(0);
    let minor = parts.get(1).copied().unwrap_or(0);
    major > max_major || (major == max_major && minor >= max_minor)
}

/// Pakete mit harten Obergrenzen die trotz erfolgreichem Import erzwungen werden muessen.
/// (package, max_major, max_minor) -- "< max_major.max_minor"
const VERSION_CEILINGS: &[(&str, u32, u32)] = &[
    ("numpy", 2, 0),
];

fn check_package_installed(python: &str, package: &str) -> DependencyStatus {
    let import_name = match package {
        "scikit-learn"    => "sklearn",
        "opencv-python"   => "cv2",
        "pillow"          => "PIL",
        other             => other,
    };
    // Nicht nur Metadaten lesen, sondern auch echten Import versuchen.
    // Faengt binaere Inkompatibilitaeten ab (z.B. numpy/pandas ABI-Mismatch),
    // die importlib.metadata nicht erkennt weil das Paket "installiert" aber kaputt ist.
    let check = Command::new(python)
        .args(["-c", &format!(
            "import importlib.metadata, {}; print(importlib.metadata.version('{}'))",
            import_name, package
        )])
        .output();
    match check {
        Ok(out) if out.status.success() => {
            let version = String::from_utf8_lossy(&out.stdout).trim().to_string();
            // Versions-Obergrenze pruefen: z.B. numpy>=2.0 gilt als "nicht korrekt installiert",
            // weil andere Pakete (pandas etc.) noch gegen numpy<2.0 kompiliert sein koennen.
            // Ein erfolgreicher eigener Import reicht hier nicht -- die Versionsnummer entscheidet.
            if let Some(&(_, max_major, max_minor)) = VERSION_CEILINGS.iter().find(|(p, _, _)| *p == package) {
                if version_exceeds_max(&version, max_major, max_minor) {
                    eprintln!("[Deps] {} Version {} ueberschreitet Obergrenze {}.{} -- wird neu installiert",
                        package, version, max_major, max_minor);
                    return DependencyStatus { package: package.to_string(), installed: false, version: Some(version) };
                }
            }
            DependencyStatus { package: package.to_string(), installed: true, version: Some(version) }
        }
        Ok(out) => {
            // Import ist fehlgeschlagen -- Paket gilt als nicht (korrekt) installiert,
            // damit install_plugins es per --force-reinstall neu installiert.
            let stderr = String::from_utf8_lossy(&out.stderr);
            if stderr.contains("binary incompatibility") || stderr.contains("dtype size changed") {
                eprintln!("[Deps] {} hat ABI-Konflikt (numpy/pandas Mismatch) -- wird neu installiert", package);
            }
            DependencyStatus { package: package.to_string(), installed: false, version: None }
        }
        _ => DependencyStatus { package: package.to_string(), installed: false, version: None },
    }
}

fn settings_path() -> Result<PathBuf, String> {
    dirs::home_dir()
        .ok_or("Konnte Home-Verzeichnis nicht finden".to_string())
        .map(|h| h.join(".frametrain").join("settings.json"))
}

fn mark_first_launch_complete() -> Result<(), String> {
    let path = settings_path()?;
    let mut settings = if path.exists() {
        let json = std::fs::read_to_string(&path).unwrap_or_else(|_| "{}".to_string());
        serde_json::from_str(&json).unwrap_or_else(|_| serde_json::json!({}))
    } else { serde_json::json!({}) };
    settings["first_launch_completed"] = serde_json::json!(true);
    std::fs::create_dir_all(path.parent().unwrap()).map_err(|e| format!("mkdir: {}", e))?;
    std::fs::write(&path, serde_json::to_string_pretty(&settings).unwrap()).map_err(|e| format!("write: {}", e))?;
    println!("[Deps] First launch als abgeschlossen markiert");
    Ok(())
}

// ══════════════════════════════════════════════════════════════════
// PRE-FLIGHT HELPERS
// ══════════════════════════════════════════════════════════════════

/// Gibt (major, minor) zurück wenn Python-Version >= 3.8, sonst None
fn parse_python_version(version_str: &str) -> Option<(u32, u32)> {
    // "Python 3.11.4" oder "3.11.4"
    let trimmed = version_str.trim().trim_start_matches("Python ");
    let parts: Vec<&str> = trimmed.split('.').collect();
    if parts.len() >= 2 {
        let major = parts[0].parse::<u32>().ok()?;
        let minor = parts[1].parse::<u32>().ok()?;
        return Some((major, minor));
    }
    None
}

/// Prüft Python-Version aus stdout+stderr (manche Pythons schreiben auf stderr)
fn get_python_version_string(cmd: &str) -> Option<String> {
    if let Ok(out) = Command::new(cmd).arg("--version").output() {
        let from_stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
        let from_stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
        let version_str = if !from_stdout.is_empty() { from_stdout } else { from_stderr };
        if !version_str.is_empty() { return Some(version_str); }
    }
    None
}

/// Findet Python-Executable UND prüft Version >= 3.8.
/// Gibt (executable, version_string, version_ok) zurück.
fn find_valid_python() -> (Option<String>, Option<String>, bool) {
    // Zeigt genau den Interpreter, den die App nutzt (siehe python_env).
    let (path, version) = crate::python_env::resolve_python_with_version();
    let ok = version.as_deref()
        .and_then(parse_python_version)
        .map(|(major, minor)| major == 3 && minor >= 8)
        .unwrap_or(false);
    (path, version, ok)
}

/// Prüft ob pip verfügbar ist für das gegebene Python
fn check_pip(python: &str) -> bool {
    Command::new(python)
        .args(["-m", "pip", "--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Freier Speicherplatz in GB für das Home-Verzeichnis
fn get_free_disk_gb() -> f64 {
    // Wir nutzen ein Python-Einzeiler weil cross-platform sys-Crate nicht im Projekt ist
    let python = get_python_executable();
    let script = if cfg!(target_os = "windows") {
        "import shutil,os; s=shutil.disk_usage(os.environ.get('USERPROFILE','C:\\\\')); print(s.free/1e9)".to_string()
    } else {
        "import shutil,os; s=shutil.disk_usage(os.path.expanduser('~')); print(s.free/1e9)".to_string()
    };
    if let Ok(out) = Command::new(&python).args(["-c", &script]).output() {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            return s.parse::<f64>().unwrap_or(0.0);
        }
    }
    // Fallback: kein Python — versuche os-native Methode
    #[cfg(unix)]
    {
        if let Ok(out) = Command::new("df").args(["-BG", "/"]).output() {
            let s = String::from_utf8_lossy(&out.stdout);
            if let Some(line) = s.lines().nth(1) {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    if let Ok(gb) = parts[3].trim_end_matches('G').parse::<f64>() {
                        return gb;
                    }
                }
            }
        }
    }
    0.0
}

/// GPU-Detection: NVIDIA/CUDA via nvidia-smi + Python torch-check
fn detect_gpu() -> GpuInfo {
    // 1. nvidia-smi prüfen
    let (has_nvidia, cuda_ver, gpu_name) = if let Ok(out) = Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
    {
        if out.status.success() {
            let name = String::from_utf8_lossy(&out.stdout).trim().to_string();
            // CUDA-Version aus nvidia-smi
            let cuda = Command::new("nvidia-smi")
                .output()
                .ok()
                .and_then(|o| {
                    let text = String::from_utf8_lossy(&o.stdout).to_string();
                    // "CUDA Version: 12.1" aus dem Header
                    text.lines()
                        .find(|l| l.contains("CUDA Version"))
                        .and_then(|l| l.split(':').nth(1))
                        .map(|s| s.trim().to_string())
                });
            (true, cuda, if name.is_empty() { None } else { Some(name) })
        } else { (false, None, None) }
    } else { (false, None, None) };

    // 2. macOS Apple Silicon — MPS
    let is_apple_silicon = cfg!(target_os = "macos") && {
        Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).contains("Apple"))
            .unwrap_or(false)
    };

    // 3. Empfohlenen torch-Index ableiten
    let recommended_torch_index = if has_nvidia {
        // CUDA-Version parsen: "12.1" → "cu121"
        cuda_ver.as_deref()
            .and_then(|v| {
                let parts: Vec<&str> = v.split('.').collect();
                if parts.len() >= 2 {
                    let major = parts[0].trim();
                    let minor = parts[1].trim();
                    // Torch unterstützt: cu118, cu121, cu124
                    let major_n = major.parse::<u32>().unwrap_or(0);
                    let minor_n = minor.parse::<u32>().unwrap_or(0);
                    Some(match (major_n, minor_n) {
                        (11, _)            => "cu118".to_string(),
                        (12, 0..=2)        => "cu121".to_string(),
                        (12, 3..=u32::MAX) => "cu124".to_string(),
                        _                  => "cu121".to_string(),
                    })
                } else { None }
            })
            .unwrap_or_else(|| "cu121".to_string())
    } else if is_apple_silicon {
        "cpu".to_string()  // PyTorch für MPS kommt von PyPI direkt, kein Index nötig
    } else {
        "cpu".to_string()
    };

    GpuInfo {
        has_nvidia_gpu: has_nvidia,
        cuda_available: has_nvidia,
        cuda_version: cuda_ver,
        gpu_name,
        recommended_torch_index,
    }
}

/// Erstellt das torch-Install-Spec abhängig von der GPU
fn torch_install_args(gpu: &GpuInfo) -> Vec<String> {
    if gpu.has_nvidia_gpu && gpu.recommended_torch_index != "cpu" {
        // CUDA-Torch über extra-index-url
        let index_url = format!(
            "https://download.pytorch.org/whl/{}",
            gpu.recommended_torch_index
        );
        vec![
            "torch".to_string(),
            "torchvision".to_string(),
            "torchaudio".to_string(),
            "--index-url".to_string(),
            index_url,
        ]
    } else {
        // CPU oder MPS (macOS) — normaler PyPI-Torch
        vec!["torch".to_string(), "torchvision".to_string(), "torchaudio".to_string()]
    }
}

/// Prüft ob torchvision/torchaudio (falls installiert) zur torch-Version passen.
/// Inkompatible Kombinationen (z. B. nach torch-Update ohne Begleitpakete)
/// crashen sonst später beim transformers-Import mit
/// "operator torchvision::nms does not exist".
fn torch_ecosystem_broken(python: &str) -> bool {
    let check = "import torch\nfor _m in ('torchvision', 'torchaudio'):\n    try:\n        __import__(_m)\n    except ImportError:\n        pass";
    Command::new(python)
        .args(["-c", check])
        .output()
        .map(|o| !o.status.success())
        .unwrap_or(false)
}

/// Prüft ob torch bereits mit korrekter CUDA-Unterstützung installiert ist
fn torch_needs_reinstall(python: &str, gpu: &GpuInfo) -> bool {
    if !gpu.has_nvidia_gpu { return false; }
    // Prüfe ob torch.cuda.is_available()
    let check = Command::new(python)
        .args(["-c", "import torch; print('1' if torch.cuda.is_available() else '0')"])
        .output();
    match check {
        Ok(out) if out.status.success() => {
            let result = String::from_utf8_lossy(&out.stdout).trim().to_string();
            result != "1"  // true = braucht Reinstall weil CUDA nicht funktioniert
        }
        _ => false,  // torch gar nicht installiert — normaler Install-Flow
    }
}

// ══════════════════════════════════════════════════════════════════
// TAURI COMMANDS
// ══════════════════════════════════════════════════════════════════

/// Pre-Flight-Check: alles prüfen bevor Installation startet
#[tauri::command]
pub async fn run_preflight_check() -> Result<PreFlightCheck, String> {
    let mut errors: Vec<String> = vec![];
    let mut warnings: Vec<String> = vec![];

    let platform = format!("{}/{}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );

    // 1. Python finden + Version prüfen
    let (python_exe, python_ver, python_ver_ok) = find_valid_python();
    let python_found = python_exe.is_some();

    if !python_found {
        errors.push("Python wurde nicht gefunden. Bitte installiere Python 3.8+ von python.org".to_string());
    } else if !python_ver_ok {
        let ver = python_ver.as_deref().unwrap_or("?");
        errors.push(format!(
            "Python {} ist zu alt. FrameTrain benötigt Python 3.8 oder neuer.",
            ver
        ));
    }

    // 2. pip prüfen
    let pip_found = python_exe.as_deref()
        .map(check_pip)
        .unwrap_or(false);

    if python_found && python_ver_ok && !pip_found {
        errors.push(
            "pip ist nicht verfügbar. Führe aus: python3 -m ensurepip --upgrade".to_string()
        );
    }

    // 3. Speicherplatz
    let free_gb = get_free_disk_gb();
    let free_gb_ok = free_gb >= 6.0;
    if !free_gb_ok && free_gb > 0.0 {
        errors.push(format!(
            "Zu wenig Speicherplatz: {:.1} GB frei. FrameTrain benötigt mindestens 6 GB.",
            free_gb
        ));
    }
    if free_gb > 0.0 && free_gb < 10.0 {
        warnings.push(format!(
            "Nur {:.1} GB frei. PyTorch + Modelle benötigen typischerweise 8-15 GB.",
            free_gb
        ));
    }

    // 4. GPU-Detection
    let gpu_info = detect_gpu();
    if gpu_info.has_nvidia_gpu {
        println!("[PreFlight] NVIDIA GPU gefunden: {:?}, CUDA: {:?}",
            gpu_info.gpu_name, gpu_info.cuda_version);
    } else {
        warnings.push("Keine NVIDIA-GPU erkannt. Training läuft auf CPU (langsamer).".to_string());
    }

    let ok = errors.is_empty();
    Ok(PreFlightCheck {
        ok,
        python_found,
        python_version: python_ver,
        python_version_ok: python_ver_ok,
        pip_found,
        free_gb,
        free_gb_ok,
        gpu_info,
        platform,
        errors,
        warnings,
    })
}

#[tauri::command]
pub async fn get_available_plugins(_app_handle: AppHandle) -> Result<Vec<PluginInfo>, String> {
    verify_python_available()?;
    let python = get_python_executable();

    // HuggingFace-Stack: deckt Text, Bild, Audio und Seq2Seq ab.
    // librosa/soundfile (Audio) und pillow (Bild) gehoeren dazu — ohne sie
    // brechen Audio-Training, Audio-Test und die Bild-Vorlagen ab.
    let nlp_packages = vec!["torch", "transformers", "datasets", "huggingface_hub", "scikit-learn",
                            "numpy", "accelerate", "librosa", "soundfile", "pillow"];
    let nlp_installed = nlp_packages.iter().all(|p| check_package_installed(&python, p).installed);
    let nlp_plugin = PluginInfo {
        id: "seq_classification".to_string(),
        name: "firstLaunch.pluginRegistry.seq_classification.name".to_string(),
        description: "firstLaunch.pluginRegistry.seq_classification.description".to_string(),
        category: "NLP".to_string(), icon: String::new(), built_in: true,   // Symbol kommt aus der UI (lucide)
        required_packages: nlp_packages.iter().map(|s| s.to_string()).collect(),
        optional_packages: vec!["peft".to_string()],
        estimated_size_mb: 2800, install_time_minutes: 4, priority: 1,
        is_selected: true, is_installed: nlp_installed,
    };

    // YOLO-Stack
    let yolo_packages = vec!["ultralytics", "torch", "numpy", "pillow", "opencv-python"];
    let yolo_installed = yolo_packages.iter().all(|p| check_package_installed(&python, p).installed);
    let yolo_plugin = PluginInfo {
        id: "yolo".to_string(),
        name: "firstLaunch.pluginRegistry.yolo.name".to_string(),
        description: "firstLaunch.pluginRegistry.yolo.description".to_string(),
        category: "Vision".to_string(), icon: String::new(), built_in: true,
        required_packages: yolo_packages.iter().map(|s| s.to_string()).collect(),
        optional_packages: vec![],
        estimated_size_mb: 1500, install_time_minutes: 2, priority: 2,
        is_selected: false, is_installed: yolo_installed,
    };

    Ok(vec![nlp_plugin, yolo_plugin])
}

#[tauri::command]
pub async fn check_dependency_status() -> Result<Vec<DependencyStatus>, String> {
    verify_python_available()?;
    let python = get_python_executable();
    let packages = vec!["torch", "transformers", "datasets", "huggingface_hub", "scikit-learn",
                        "numpy", "pandas", "pyarrow", "accelerate", "librosa", "soundfile",
                        "pillow", "ultralytics"];
    let status: Vec<DependencyStatus> = packages.iter().map(|p| check_package_installed(&python, p)).collect();
    let missing: Vec<_> = status.iter().filter(|s| !s.installed).map(|s| s.package.as_str()).collect();
    if missing.is_empty() { println!("[Deps] Alle Pakete installiert"); }
    else { println!("[Deps] Fehlende Pakete: {:?}", missing); }
    Ok(status)
}

#[tauri::command]
pub async fn check_first_launch() -> Result<bool, String> {
    let path = settings_path()?;
    if !path.exists() { return Ok(true); }
    let json = std::fs::read_to_string(&path).map_err(|e| format!("Settings lesen: {}", e))?;
    let settings: serde_json::Value = serde_json::from_str(&json).map_err(|e| format!("Settings parsen: {}", e))?;
    let completed = settings["first_launch_completed"].as_bool().unwrap_or(false);
    Ok(!completed)
}

#[tauri::command]
pub async fn install_plugins(_app_handle: AppHandle, plugin_ids: Vec<String>, window: Window) -> Result<(), String> {
    // --- Schritt 0: Pre-Flight inline ---
    let preflight = match run_preflight_check().await {
        Ok(p) => p,
        Err(e) => {
            let _ = window.emit("plugin-install-progress", PluginInstallProgress {
                plugin_id: "system".to_string(), status: "failed".to_string(),
                message: format!("Pre-Flight-Check fehlgeschlagen: {}", e), progress: Some(0),
            });
            return Err(e);
        }
    };
    if !preflight.ok {
        let msg = preflight.errors.join(" | ");
        let _ = window.emit("plugin-install-progress", PluginInstallProgress {
            plugin_id: "system".to_string(), status: "failed".to_string(),
            message: msg.clone(), progress: Some(0),
        });
        return Err(msg);
    }

    // --- Schritt 1: Paketliste aufbauen ---
    // Torch wird separat behandelt (GPU-aware), daher hier raus
    // pandas/pyarrow: immer installiert, werden für Dataset-Vorschau (Parquet-Preview)
    // und generelle Dataset-Verarbeitung gebraucht, unabhängig vom gewählten Plugin.
    let mut packages: Vec<(&'static str, &'static str)> = vec![
        ("numpy",           "NumPy"),
        ("pandas",          "Pandas"),
        ("pyarrow",         "PyArrow (Parquet)"),
    ];
    if plugin_ids.iter().any(|id| id == "seq_classification") || plugin_ids.is_empty() {
        packages.extend([
            ("transformers",    "Transformers"),
            ("datasets",        "HuggingFace Datasets"),
            ("huggingface_hub", "HuggingFace Hub"),
            ("scikit-learn",    "Scikit-Learn"),
            ("accelerate",      "Accelerate"),
            // Bild und Audio laufen ueber denselben Stack:
            ("pillow",          "Pillow (Bild)"),
            ("librosa",         "Librosa (Audio)"),
            ("soundfile",       "SoundFile (Audio)"),
        ]);
    }
    if plugin_ids.iter().any(|id| id == "yolo") {
        packages.extend([
            ("ultralytics",    "Ultralytics YOLO"),
            ("pillow",         "Pillow"),
            ("opencv-python",  "OpenCV"),
        ]);
    }
    packages.dedup_by_key(|(p, _)| *p);

    // Torch-Args nach GPU-Typ
    let gpu_info = preflight.gpu_info.clone();
    let torch_args = torch_install_args(&gpu_info);
    let torch_description = if gpu_info.has_nvidia_gpu {
        format!("PyTorch (CUDA {})", gpu_info.recommended_torch_index)
    } else {
        "PyTorch (CPU)".to_string()
    };

    // Gesamt-Schritte: pip-upgrade + torch + alle packages
    let total_steps = 1 + 1 + packages.len();

    tauri::async_runtime::spawn(async move {
        let python = get_python_executable();
        let t0 = Instant::now();
        let mut step = 0usize;

        let emit_progress = |w: &Window, msg: &str, status: &str, pct: i32| {
            let _ = w.emit("plugin-install-progress", PluginInstallProgress {
                plugin_id: "system".to_string(),
                status: status.to_string(),
                message: msg.to_string(),
                progress: Some(pct),
            });
        };

        let pct = |s: usize| ((s as f32 / total_steps as f32) * 100.0) as i32;

        // --- Schritt 2: pip selbst upgraden ---
        emit_progress(&window, "pip wird aktualisiert...", "installing_package", pct(step));
        let _ = Command::new(&python)
            .args(["-m", "pip", "install", "--quiet", "--upgrade", "pip"])
            .output();
        step += 1;
        emit_progress(&window, "pip aktualisiert ✓", "package_complete", pct(step));

        // --- Schritt 3: Torch installieren (GPU-aware, mit Retry) ---
        emit_progress(
            &window,
            &format!("Installiere {} (kann mehrere Minuten dauern)...", torch_description),
            "installing_package",
            pct(step),
        );

        // Prüfe ob Torch Reinstall nötig (CPU-Torch aber NVIDIA-GPU vorhanden)
        let torch_status = check_package_installed(&python, "torch");
        let needs_reinstall = torch_status.installed && torch_needs_reinstall(&python, &gpu_info);
        // torch da, aber torchvision/torchaudio passen nicht zur torch-Version
        let ecosystem_broken = torch_status.installed && torch_ecosystem_broken(&python);

        if torch_status.installed && !needs_reinstall && !ecosystem_broken {
            emit_progress(
                &window,
                &format!("PyTorch bereits vorhanden ({}) ✓",
                    torch_status.version.as_deref().unwrap_or("✓")),
                "package_complete",
                pct(step + 1),
            );
        } else {
            // Torch installieren (oder neu installieren für CUDA / Versionskonflikt)
            let mut torch_cmd_args = vec!["-m".to_string(), "pip".to_string(),
                "install".to_string(), "--quiet".to_string()];
            if needs_reinstall {
                torch_cmd_args.push("--force-reinstall".to_string());
                emit_progress(
                    &window,
                    "PyTorch wird für CUDA neu installiert...",
                    "installing_package",
                    pct(step),
                );
            } else if ecosystem_broken {
                // --upgrade nötig: sonst meldet pip "already satisfied" und lässt
                // die inkompatiblen torchvision/torchaudio-Versionen stehen
                torch_cmd_args.push("--upgrade".to_string());
                emit_progress(
                    &window,
                    "torchvision/torchaudio passen nicht zur torch-Version — werden aktualisiert...",
                    "installing_package",
                    pct(step),
                );
            }
            torch_cmd_args.extend(torch_args.clone());

            let torch_ok = run_pip_with_retry(&python, &torch_cmd_args, &window, 2);

            if !torch_ok {
                // Fallback: CPU-Torch wenn CUDA-Install fehlschlägt
                if gpu_info.has_nvidia_gpu {
                    emit_progress(
                        &window,
                        "CUDA-Torch fehlgeschlagen, installiere CPU-Variante als Fallback...",
                        "installing_package",
                        pct(step),
                    );
                    let fallback_args = vec!["-m".to_string(), "pip".to_string(),
                        "install".to_string(), "--quiet".to_string(),
                        "torch".to_string(), "torchvision".to_string(), "torchaudio".to_string()];
                    let fallback_ok = run_pip_with_retry(&python, &fallback_args, &window, 1);
                    if !fallback_ok {
                        emit_progress(&window, "PyTorch konnte nicht installiert werden. Prüfe Internetverbindung und Speicherplatz.", "failed", pct(step));
                        return;
                    }
                } else {
                    emit_progress(&window, "PyTorch konnte nicht installiert werden. Prüfe Internetverbindung und Speicherplatz.", "failed", pct(step));
                    return;
                }
            }
        }
        step += 1;

        // --- Schritt 4: Rest-Pakete ---
        for (package, description) in &packages {
            let current_pct = pct(step);

            // Skip wenn schon installiert
            let already = check_package_installed(&python, package);
            if already.installed {
                println!("[Deps] {} bereits installiert ({}), überspringe",
                    package, already.version.as_deref().unwrap_or("?"));
                emit_progress(
                    &window,
                    &format!("{} bereits vorhanden ({}) ✓",
                        description, already.version.as_deref().unwrap_or("✓")),
                    "package_complete",
                    current_pct,
                );
                step += 1;
                continue;
            }

            emit_progress(
                &window,
                &format!("Installiere {} ({}/{})...", description, step, total_steps),
                "installing_package",
                current_pct,
            );

            // Versionsconstraints
            let install_spec = match *package {
                "transformers"    => "transformers>=4.35.0",
                "datasets"        => "datasets>=2.14.0",
                "huggingface_hub" => "huggingface_hub>=0.19.0",
                "scikit-learn"    => "scikit-learn>=1.3.0",
                "numpy"           => "numpy>=1.24.0,<2.0.0",
                "pandas"          => "pandas>=2.0.0,<2.2.0",  // kompatibel zu numpy<2.0
                "pyarrow"         => "pyarrow>=14.0.0",
                "accelerate"      => "accelerate>=0.24.0",
                "ultralytics"     => "ultralytics>=8.0.0",
                "pillow"          => "pillow>=9.0.0",
                "librosa"         => "librosa>=0.10.0",
                "soundfile"       => "soundfile>=0.12.0",
                "opencv-python"   => "opencv-python>=4.8.0",
                other             => other,
            };

            let pip_args = vec!["-m".to_string(), "pip".to_string(),
                "install".to_string(), "--quiet".to_string(),
                install_spec.to_string()];

            let ok = run_pip_with_retry(&python, &pip_args, &window, 2);
            if !ok {
                emit_progress(
                    &window,
                    &format!("Fehler beim Installieren von {}. Prüfe Internetverbindung.", package),
                    "failed",
                    current_pct,
                );
                return;
            }

            emit_progress(
                &window,
                &format!("{} installiert ✓", description),
                "package_complete",
                current_pct,
            );
            step += 1;
        }

        // --- Abschluss ---
        let secs = t0.elapsed().as_secs();
        let _ = mark_first_launch_complete();
        emit_progress(
            &window,
            &format!("Alle Dependencies installiert! ({}m {}s)", secs / 60, secs % 60),
            "complete",
            100,
        );
        let _ = window.emit("plugin-install-complete", ());
    });
    Ok(())
}

/// Führt einen pip-Befehl mit bis zu `retries` Versuchen aus.
/// Gibt stderr-Meldungen über Events weiter. Gibt true zurück bei Erfolg.
fn run_pip_with_retry(python: &str, args: &[String], window: &Window, retries: u32) -> bool {
    for attempt in 0..=retries {
        if attempt > 0 {
            let _ = window.emit("plugin-install-progress", PluginInstallProgress {
                plugin_id: "system".to_string(),
                status: "installing_package".to_string(),
                message: format!("Versuch {} von {}...", attempt + 1, retries + 1),
                progress: None,
            });
            std::thread::sleep(Duration::from_secs(3));
        }

        let mut child = match Command::new(python)
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                let _ = window.emit("plugin-install-progress", PluginInstallProgress {
                    plugin_id: "system".to_string(),
                    status: "installing_package".to_string(),
                    message: format!("Spawn-Fehler: {}", e),
                    progress: None,
                });
                continue;
            }
        };

        // stdout live streamen (rate-limited)
        if let Some(stdout) = child.stdout.take() {
            let wc = window.clone();
            std::thread::spawn(move || {
                let mut last = Instant::now();
                for line in BufReader::new(stdout).lines().flatten() {
                    if last.elapsed() > Duration::from_millis(500) {
                        let _ = wc.emit("plugin-install-progress", PluginInstallProgress {
                            plugin_id: "system".to_string(),
                            status: "installing_package".to_string(),
                            message: line,
                            progress: None,
                        });
                        last = Instant::now();
                    }
                }
            });
        }

        // stderr in Buffer sammeln
        let stderr_buf = child.stderr.take().map(|e| {
            BufReader::new(e).lines().flatten().collect::<Vec<_>>().join("\n")
        }).unwrap_or_default();

        let status = match child.wait() {
            Ok(s) => s,
            Err(_) => continue,
        };

        if status.success() {
            return true;
        }

        // Fehlerdetails weitergeben
        if !stderr_buf.is_empty() {
            // Nur die letzten 3 Zeilen — pip gibt oft lange Stacktraces
            let last_lines: Vec<&str> = stderr_buf.lines().rev().take(3).collect();
            let short_err = last_lines.into_iter().rev().collect::<Vec<_>>().join(" | ");
            let _ = window.emit("plugin-install-progress", PluginInstallProgress {
                plugin_id: "system".to_string(),
                status: "installing_package".to_string(),
                message: format!("pip-Fehler: {}", short_err),
                progress: None,
            });
        }
    }
    false
}

#[tauri::command]
pub async fn handle_plugin_approval(_plugin_id: String, _approved: bool, _remember: bool) -> Result<(), String> {
    Ok(())
}

// ══════════════════════════════════════════════════════════════════
// YOLO INFERENZ COMMAND
// ══════════════════════════════════════════════════════════════════

/// Letzte aussagekräftige Zeile einer Python-Ausgabe.
///
/// Tracebacks und Ultralytics-Hinweise sind vielzeilig; die entscheidende
/// Information steht am Ende. Vorher landete der rohe Anfang der Ausgabe
/// (inklusive Warn-Emoji) ungefiltert in der Oberfläche.
fn last_meaningful_line(text: &str) -> String {
    let line = text.lines().rev()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with("Traceback") && !l.starts_with("File \""))
        .unwrap_or("keine Fehlerausgabe");
    let cleaned: String = line.chars().filter(|c| !matches!(c, '\u{26a0}' | '\u{fe0f}')).collect();
    let cleaned = cleaned.trim();
    if cleaned.chars().count() > 300 {
        format!("{}…", cleaned.chars().take(300).collect::<String>())
    } else {
        cleaned.to_string()
    }
}

/// Führt YOLO-Inferenz auf einem einzelnen Bild aus via Ultralytics Python-API.
#[tauri::command]
pub async fn run_yolo_inference(
    model_path: String,
    image_path: String,
    conf_threshold: f32,
    iou_threshold: f32,
) -> Result<YoloInferenceResult, String> {
    let python = get_python_executable();

    // FIX Sicherheit: Pfade werden als sys.argv übergeben statt in den Python-Code
    // interpoliert zu werden (vorher: Code-Injection / Crash bei Anführungszeichen im Pfad).
    let script = r#"
import sys, json, time, os, contextlib

# Ultralytics schreibt Hinweise ("Unable to automatically guess model task ...")
# nach stdout. Frueher landeten die vor dem Ergebnis-JSON und der Parser auf der
# Rust-Seite scheiterte mit "expected value at line 1 column 1". Deshalb geht
# waehrend der Inferenz alles nach stderr, das JSON wird am Ende bewusst
# auf das echte stdout geschrieben.
_real_stdout = sys.stdout


def emit(payload):
    print(json.dumps(payload), file=_real_stdout)


def find_weights(path):
    """Akzeptiert eine .pt-Datei oder einen Modell-/Versionsordner."""
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        return None
    preferred = [
        "model.pt",
        os.path.join("weights", "best.pt"),
        os.path.join("weights", "last.pt"),
        os.path.join("train", "weights", "best.pt"),
        os.path.join("train", "weights", "last.pt"),
        "best.pt", "last.pt",
    ]
    for rel in preferred:
        cand = os.path.join(path, rel)
        if os.path.isfile(cand):
            return cand
    for root, _dirs, files in os.walk(path):
        pts = sorted(f for f in files if f.endswith(".pt"))
        if pts:
            return os.path.join(root, pts[0])
    return None


try:
    with contextlib.redirect_stdout(sys.stderr):
        from ultralytics import YOLO

        model_arg   = sys.argv[1]
        image_path  = sys.argv[2]
        conf        = float(sys.argv[3])
        iou         = float(sys.argv[4])
        task        = sys.argv[5] if len(sys.argv) > 5 else "detect"

        if not os.path.exists(model_arg):
            emit({"error": f"Modell nicht gefunden: {model_arg}"})
            sys.exit(1)
        weights = find_weights(model_arg)
        if weights is None:
            emit({"error": f"Keine Gewichtsdatei (.pt) in {model_arg} gefunden."})
            sys.exit(1)
        if not os.path.exists(image_path):
            emit({"error": f"Bild nicht gefunden: {image_path}"})
            sys.exit(1)

        # task explizit setzen, sonst raet Ultralytics und warnt dabei.
        model = YOLO(weights, task=task)
        t0 = time.perf_counter()
        results = model(image_path, conf=conf, iou=iou, verbose=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            names = r.names
            for box in boxes:
                cls_id = int(box.cls[0].item())
                label  = names.get(cls_id, str(cls_id)) if hasattr(names, "get") else str(cls_id)
                conf_v = float(box.conf[0].item())
                xyxy   = box.xyxy[0].tolist()
                detections.append({
                    "label":      label,
                    "confidence": conf_v,
                    "bbox":       xyxy,
                })

    emit({
        "detections":        detections,
        "inference_time_ms": elapsed_ms,
        "image_path":        image_path,
        "weights_path":      weights,
    })

except ImportError:
    emit({"error": "ultralytics ist nicht installiert. Installiere es über den First-Launch-Setup."})
    sys.exit(1)
except SystemExit:
    raise
except Exception as e:
    emit({"error": f"{type(e).__name__}: {e}"})
    sys.exit(1)
"#;

    let tmp_path = std::env::temp_dir().join(format!("ft_yolo_infer_{}.py", uuid::Uuid::new_v4()));
    std::fs::write(&tmp_path, &script).map_err(|e| format!("Script schreiben: {}", e))?;

    let out = Command::new(&python)
        .arg(tmp_path.to_string_lossy().to_string())
        .arg(&model_path)
        .arg(&image_path)
        .arg(format!("{:.4}", conf_threshold))
        .arg(format!("{:.4}", iou_threshold))
        .arg("detect")
        .output()
        .map_err(|e| format!("Python spawn: {}", e))?;

    std::fs::remove_file(&tmp_path).ok();

    let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if stdout.is_empty() {
        let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
        return Err(format!("YOLO-Inferenz fehlgeschlagen: {}", last_meaningful_line(&stderr)));
    }

    // Robust gegen Fremdausgaben: die letzte Zeile nehmen, die als JSON durchgeht.
    // Ein roher stdout-Auszug in der Fehlermeldung half niemandem weiter.
    let json: serde_json::Value = stdout.lines().rev()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l.trim()).ok())
        .find(|v| v.is_object())
        .ok_or_else(|| {
            let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
            format!("YOLO-Inferenz lieferte kein Ergebnis: {}",
                last_meaningful_line(if stderr.is_empty() { &stdout } else { &stderr }))
        })?;

    if let Some(err) = json.get("error").and_then(|e| e.as_str()) {
        return Err(err.to_string());
    }

    let detections: Vec<YoloDetection> = json.get("detections")
        .and_then(|d| d.as_array())
        .map(|arr| arr.iter().filter_map(|v| {
            let label = v.get("label")?.as_str()?.to_string();
            let confidence = v.get("confidence")?.as_f64()? as f32;
            let bbox_arr = v.get("bbox")?.as_array()?;
            if bbox_arr.len() < 4 { return None; }
            let bbox = [
                bbox_arr[0].as_f64()? as f32,
                bbox_arr[1].as_f64()? as f32,
                bbox_arr[2].as_f64()? as f32,
                bbox_arr[3].as_f64()? as f32,
            ];
            Some(YoloDetection { label, confidence, bbox })
        }).collect())
        .unwrap_or_default();

    let inference_time_ms = json.get("inference_time_ms").and_then(|t| t.as_f64()).unwrap_or(0.0);

    Ok(YoloInferenceResult { detections, inference_time_ms, image_path })
}

#[cfg(test)]
mod yolo_error_tests {
    use super::last_meaningful_line;

    #[test]
    fn nimmt_die_letzte_zeile_eines_tracebacks() {
        let tb = "Traceback (most recent call last):\n  File \"/x/y.py\", line 3, in <module>\n\
                  TypeError: model='/pfad' is not a supported model format.";
        assert_eq!(last_meaningful_line(tb),
            "TypeError: model='/pfad' is not a supported model format.");
    }

    #[test]
    fn warn_emoji_landet_nicht_in_der_oberflaeche() {
        // Ultralytics schreibt "WARNING ⚠️ ..." – die App zeigt keine Emojis.
        let out = last_meaningful_line("WARNING \u{26a0}\u{fe0f} Unable to guess model task");
        assert!(!out.contains('\u{26a0}'), "{}", out);
        assert!(out.starts_with("WARNING"), "{}", out);
    }

    #[test]
    fn leere_ausgabe_liefert_einen_hinweis() {
        assert_eq!(last_meaningful_line("   \n\n"), "keine Fehlerausgabe");
    }

    #[test]
    fn sehr_lange_zeilen_werden_gekuerzt() {
        let long = "x".repeat(500);
        let out = last_meaningful_line(&long);
        assert_eq!(out.chars().count(), 301, "300 Zeichen plus Auslassungszeichen");
    }
}
