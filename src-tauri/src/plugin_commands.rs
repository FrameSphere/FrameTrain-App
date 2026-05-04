// plugin_commands.rs
// Verwaltet den First-Launch-Check und die Installation der Python-Dependencies
// für die FrameTrain Sequenzklassifikations-Engine.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::io::{BufRead, BufReader};
use std::time::{Instant, Duration};
use tauri::{AppHandle, Window};
use tauri::Emitter;

// ============ Typen ============

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
    #[serde(default)]
    pub is_selected: bool,
    #[serde(default)]
    pub is_installed: bool,
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

// ============ Hilfsfunktionen ============

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
                println!("[Deps] ✅ Python gefunden: {} ({})", cmd, version.trim());
                return Ok(());
            }
        }
    }
    
    Err(
        "Python ist nicht installiert oder nicht im PATH verfügbar. \
        Bitte installiere Python 3.8+ von python.org oder nutze einen \
        Package Manager (brew/apt/choco) und versuche es dann erneut."
            .to_string()
    )
}

fn get_python_executable() -> String {
    // Gleiche Logik wie training_manager / test_manager:
    // Python mit torch bevorzugen, falls mehrere Versionen installiert.
    let candidates: Vec<&str> = if cfg!(target_os = "windows") {
        vec!["python", "python3"]
    } else {
        vec!["python3", "python"]
    };

    for cmd in &candidates {
        if let Ok(out) = Command::new(cmd).arg("--version").output() {
            if out.status.success() {
                return cmd.to_string();
            }
        }
    }
    "python3".to_string()
}

fn check_package_installed(python: &str, package: &str) -> DependencyStatus {
    // Normiert package für import (z.B. scikit-learn → sklearn)
    let import_name = match package {
        "scikit-learn" => "sklearn",
        "torch"        => "torch",
        "transformers" => "transformers",
        "datasets"     => "datasets",
        "numpy"        => "numpy",
        "accelerate"   => "accelerate",
        other          => other,
    };

    let check = Command::new(python)
        .args(["-c", &format!(
            "import importlib.metadata, {}; print(importlib.metadata.version('{}'))",
            import_name, package
        )])
        .output();

    match check {
        Ok(out) if out.status.success() => {
            let version = String::from_utf8_lossy(&out.stdout).trim().to_string();
            DependencyStatus { package: package.to_string(), installed: true, version: Some(version) }
        }
        _ => {
            DependencyStatus { package: package.to_string(), installed: false, version: None }
        }
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
    } else {
        serde_json::json!({})
    };

    settings["first_launch_completed"] = serde_json::json!(true);
    std::fs::create_dir_all(path.parent().unwrap())
        .map_err(|e| format!("Verzeichnis erstellen: {}", e))?;
    std::fs::write(&path, serde_json::to_string_pretty(&settings).unwrap())
        .map_err(|e| format!("Settings schreiben: {}", e))?;

    println!("[Deps] ✅ First launch als abgeschlossen markiert");
    Ok(())
}

// ============ Tauri Commands ============

/// Gibt die benötigten Dependencies als PluginInfo-Liste zurück.
/// Das Frontend zeigt diese auf der First-Launch-Seite an.
#[tauri::command]
pub async fn get_available_plugins(_app_handle: AppHandle) -> Result<Vec<PluginInfo>, String> {
    // Prüfe Python Verfügbarkeit
    verify_python_available()?;
    
    let python = get_python_executable();

    // Die eine "Plugin-Gruppe" ist die Sequenzklassifikations-Engine
    let packages = vec![
        "torch", "transformers", "datasets", "scikit-learn", "numpy", "accelerate",
    ];

    let all_installed = packages.iter().all(|p| {
        check_package_installed(&python, p).installed
    });

    let plugin = PluginInfo {
        id:                   "seq_classification".to_string(),
        name:                 "Text-Klassifikation (HuggingFace)".to_string(),
        description:          "Trainiert encoder-basierte Modelle (z.B. XLM-RoBERTa, BERT, DeBERTa) für Textklassifikation. Unterstützt Sentiment-Analyse, Topic-Detection, Spam-Filter u.v.m.".to_string(),
        category:             "NLP".to_string(),
        icon:                 "🤗".to_string(),
        built_in:             true,
        required_packages:    packages.iter().map(|s| s.to_string()).collect(),
        optional_packages:    vec!["peft".to_string()],
        estimated_size_mb:    2500,
        install_time_minutes: 3,
        priority:             1,
        is_selected:          true,
        is_installed:         all_installed,
    };

    Ok(vec![plugin])
}

/// Prüft den Status aller erforderlichen Python-Pakete.
/// Gibt eine Liste von DependencyStatus für jedes Paket zurück.
#[tauri::command]
pub async fn check_dependency_status() -> Result<Vec<DependencyStatus>, String> {
    println!("[Deps] Prüfe Abhängigkeitsstatus...");
    
    // Prüfe zuerst ob Python überhaupt vorhanden ist
    verify_python_available()?;
    
    let python = get_python_executable();
    let packages = vec![
        "torch",
        "transformers",
        "datasets",
        "scikit-learn",
        "numpy",
        "accelerate",
    ];
    
    let status: Vec<DependencyStatus> = packages
        .iter()
        .map(|p| check_package_installed(&python, p))
        .collect();
    
    let all_installed = status.iter().all(|s| s.installed);
    if all_installed {
        println!("[Deps] ✅ Alle Pakete installiert");
    } else {
        let missing: Vec<_> = status.iter()
            .filter(|s| !s.installed)
            .map(|s| s.package.as_str())
            .collect();
        println!("[Deps] ⚠️ Fehlende Pakete: {:?}", missing);
    }
    
    Ok(status)
}

/// Prüft ob der First-Launch-Setup noch ausgeführt werden muss.
/// Gibt true zurück wenn die Dependencies noch nicht installiert sind.
#[tauri::command]
pub async fn check_first_launch() -> Result<bool, String> {
    println!("[Deps] Prüfe First-Launch-Status...");

    let path = settings_path()?;
    if !path.exists() {
        println!("[Deps] Keine Settings-Datei → First Launch");
        return Ok(true);
    }

    let json = std::fs::read_to_string(&path)
        .map_err(|e| format!("Settings lesen: {}", e))?;
    let settings: serde_json::Value = serde_json::from_str(&json)
        .map_err(|e| format!("Settings parsen: {}", e))?;

    let completed = settings["first_launch_completed"].as_bool().unwrap_or(false);
    println!("[Deps] First launch completed: {}", completed);

    if !completed {
        return Ok(true);
    }

    // Auch wenn completed=true: prüfe ob die Core-Packages noch vorhanden sind.
    // So erkennen wir wenn jemand Python neu installiert hat.
    let python = get_python_executable();
    let core_packages = ["torch", "transformers", "datasets"];
    let all_ok = core_packages.iter().all(|p| check_package_installed(&python, p).installed);

    if !all_ok {
        println!("[Deps] Core-Packages fehlen trotz completed=true → First Launch erneut");
        return Ok(true);
    }

    Ok(false)
}

/// Installiert die ausgewählten Dependencies via pip.
/// Sendet Fortschritts-Events ans Frontend.
#[tauri::command]
pub async fn install_plugins(
    _app_handle: AppHandle,
    plugin_ids: Vec<String>,
    window: Window,
) -> Result<(), String> {
    println!("[Deps] Installiere Dependencies für: {:?}", plugin_ids);
    
    // Prüfe Python Verfügbarkeit bevor Installation startet
    if let Err(e) = verify_python_available() {
        eprintln!("[Deps] ✗ Python-Fehler: {}", e);
        let _ = window.emit("plugin-install-progress", PluginInstallProgress {
            plugin_id: "seq_classification".to_string(),
            status: "error".to_string(),
            message: e,
            progress: None,
        });
        return Err("Python konnte nicht gefunden werden".to_string());
    }

    tauri::async_runtime::spawn(async move {
        let python = get_python_executable();
        println!("[Deps] Verwende Python: {}", python);

        // Installiere in dieser Reihenfolge (torch first - das ist das größte)
        let packages = vec![
            ("torch", "PyTorch (enthält große Modelle)"),
            ("numpy", "NumPy (Numerische Berechnungen)"),
            ("transformers", "Transformers (Sprachmodelle)"),
            ("datasets", "Datasets (Datenverwaltung)"),
            ("scikit-learn", "Scikit-Learn (ML-Tools)"),
            ("accelerate", "Accelerate (GPU-Unterstützung)"),
        ];

        let total = packages.len();
        let install_start = Instant::now();

        for (i, (package, description)) in packages.iter().enumerate() {
            let progress_pct = ((i as f32 / total as f32) * 100.0) as i32;

            let _ = window.emit("plugin-install-progress", PluginInstallProgress {
                plugin_id:  "seq_classification".to_string(),
                status:     "installing_package".to_string(),
                message:    format!("Installiere {} - {} ({}/{})", package, description, i + 1, total),
                progress:   Some(progress_pct),
            });

            println!("[Deps] pip install {} ({})", package, description);
            let pkg_start = Instant::now();

            let mut child = match Command::new(&python)
                .args(["-m", "pip", "install", "--upgrade", "--quiet", package])
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
            {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("[Deps] Fehler beim Starten von pip: {}", e);
                    let msg = format!(
                        "pip konnte nicht gestartet werden: {}. Stelle sicher, dass Python richtig installiert ist.",
                        e
                    );
                    let _ = window.emit("plugin-install-progress", PluginInstallProgress {
                        plugin_id: "seq_classification".to_string(),
                        status:    "failed".to_string(),
                        message:   msg,
                        progress:  None,
                    });
                    return;
                }
            };

            // Nur relevante pip-Output-Zeilen streamen (mit Throttling)
            if let Some(stdout) = child.stdout.take() {
                let win_clone = window.clone();
                let pkg_clone = package.to_string();
                std::thread::spawn(move || {
                    let mut last_update = Instant::now();
                    let mut current_line_buffer = String::new();
                    
                    for line in BufReader::new(stdout).lines().flatten() {
                        // Nur Zeilen mit wichtigen Keywords anzeigen
                        if line.contains("Downloading")
                            || line.contains("downloading")
                            || line.contains("Installing")
                            || line.contains("Collecting")
                            || line.contains("ERROR")
                            || line.contains("error")
                        {
                            // Maximal eine Update pro 500ms senden (Throttling)
                            if last_update.elapsed() > Duration::from_millis(500) {
                                println!("[pip] {} → {}", pkg_clone, line);
                                let _ = win_clone.emit("plugin-install-progress", PluginInstallProgress {
                                    plugin_id: "seq_classification".to_string(),
                                    status:    "installing_package".to_string(),
                                    message:   line.clone(),
                                    progress:  None,
                                });
                                last_update = Instant::now();
                            } else {
                                current_line_buffer = line.clone();
                            }
                        }
                    }
                    
                    // Letzte gepufferte Zeile senden
                    if !current_line_buffer.is_empty() {
                        println!("[pip] {}", current_line_buffer);
                        let _ = win_clone.emit("plugin-install-progress", PluginInstallProgress {
                            plugin_id: "seq_classification".to_string(),
                            status:    "installing_package".to_string(),
                            message:   current_line_buffer,
                            progress:  None,
                        });
                    }
                });
            }

            let status = child.wait().expect("Warten auf pip fehlgeschlagen");
            let elapsed = pkg_start.elapsed();

            if !status.success() {
                eprintln!("[Deps] ✗ {} konnte nicht installiert werden", package);
                let _ = window.emit("plugin-install-progress", PluginInstallProgress {
                    plugin_id: "seq_classification".to_string(),
                    status:    "failed".to_string(),
                    message:   format!(
                        "Fehler beim Installieren von {}.\n\n\
                        Mögliche Lösungen:\n\
                        - Überprüfe deine Internetverbindung\n\
                        - Stelle sicher, dass genug Festplattenspeicher verfügbar ist (mindestens 20GB)\n\
                        - Versuche: pip install --upgrade pip setuptools wheel", 
                        package
                    ),
                    progress:  None,
                });
                return;
            }

            let elapsed_secs = elapsed.as_secs();
            println!("[Deps] ✓ {} installiert ({:.1}s)", package, elapsed_secs);
            
            let _ = window.emit("plugin-install-progress", PluginInstallProgress {
                plugin_id: "seq_classification".to_string(),
                status: "package_complete".to_string(),
                message: format!("{} fertig ({} Sekunden)", package, elapsed_secs),
                progress: Some(progress_pct),
            });
        }

        // Abschluss mit Gesamtdauer
        let total_elapsed = install_start.elapsed();
        let total_secs = total_elapsed.as_secs();
        let total_mins = total_secs / 60;
        
        let _ = mark_first_launch_complete();
        let _ = window.emit("plugin-install-progress", PluginInstallProgress {
            plugin_id: "seq_classification".to_string(),
            status:    "complete".to_string(),
            message:   format!(
                "✅ Alle Dependencies erfolgreich installiert!\n\nGesamtdauer: {}m {}s",
                total_mins,
                total_secs % 60
            ),
            progress:  Some(100),
        });
        let _ = window.emit("plugin-install-complete", ());
        println!("[Deps] ✅ Installation abgeschlossen in {:.1}s", total_secs);
    });

    Ok(())
}

/// Wird nicht mehr benötigt, aber bleibt für API-Kompatibilität mit dem Frontend.
#[tauri::command]
pub async fn handle_plugin_approval(
    _plugin_id: String,
    _approved: bool,
    _remember: bool,
) -> Result<(), String> {
    Ok(())
}
