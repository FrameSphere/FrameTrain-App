/// Power Manager – verhindert, dass der Computer während des Trainings schläft.
///
/// macOS  → startet `caffeinate -i` als Subprocess (kein extra Crate nötig)
/// Linux  → nutzt `systemd-inhibit` falls vorhanden, sonst xdg-screensaver
/// Windows → ruft SetThreadExecutionState via FFI direkt auf (kein extra Crate)

use std::process::{Child, Command};
use std::sync::Mutex;

// ============ Windows FFI ============

#[cfg(target_os = "windows")]
mod win_power {
    // kernel32 ist auf Windows immer gelinkt – kein extra Crate nötig
    extern "system" {
        fn SetThreadExecutionState(esFlags: u32) -> u32;
    }

    const ES_CONTINUOUS: u32 = 0x8000_0000;
    const ES_SYSTEM_REQUIRED: u32 = 0x0000_0001;
    const ES_AWAYMODE_REQUIRED: u32 = 0x0000_0040;

    pub fn prevent_sleep() {
        unsafe {
            SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED);
        }
    }

    pub fn allow_sleep() {
        unsafe {
            SetThreadExecutionState(ES_CONTINUOUS);
        }
    }
}

// ============ State ============

pub struct PowerState {
    pub sleep_prevented: bool,
    /// Handle auf den laufenden Inhibitor-Prozess (macOS / Linux)
    inhibitor_process: Option<Child>,
}

impl Default for PowerState {
    fn default() -> Self {
        Self {
            sleep_prevented: false,
            inhibitor_process: None,
        }
    }
}

// ============ Tauri Commands ============

/// Aktiviert Sleep-Prevention.
#[tauri::command]
pub fn enable_prevent_sleep(
    state: tauri::State<'_, Mutex<PowerState>>,
) -> Result<bool, String> {
    let mut s = state.lock().map_err(|e| e.to_string())?;

    if s.sleep_prevented {
        return Ok(true);
    }

    #[cfg(target_os = "macos")]
    {
        // caffeinate -dims:
        //   -d  Display-Sleep verhindern
        //   -i  Idle-Sleep verhindern  
        //   -m  Disk-Sleep verhindern
        //   -s  System-Sleep verhindern (AC-Power vorausgesetzt)
        // Damit läuft das Training auch über Nacht ohne Unterbrechung.
        match Command::new("caffeinate").args(["-dims"]).spawn() {
            Ok(child) => {
                s.inhibitor_process = Some(child);
                s.sleep_prevented = true;
                println!("[PowerManager] ✅ macOS: caffeinate gestartet");
            }
            Err(e) => {
                return Err(format!("caffeinate konnte nicht gestartet werden: {}", e));
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        // Primär: systemd-inhibit (moderne Distros mit systemd)
        let result = Command::new("systemd-inhibit")
            .args([
                "--what=sleep:idle",
                "--who=FrameTrain",
                "--why=Training laeuft",
                "--mode=block",
                "sleep",
                "infinity",
            ])
            .spawn();

        match result {
            Ok(child) => {
                s.inhibitor_process = Some(child);
                s.sleep_prevented = true;
                println!("[PowerManager] ✅ Linux: systemd-inhibit gestartet");
            }
            Err(_) => {
                // Fallback: xdg-screensaver reset (X11-Umgebungen)
                match Command::new("xdg-screensaver").arg("reset").spawn() {
                    Ok(_) => {
                        s.sleep_prevented = true;
                        println!("[PowerManager] ⚠️  Linux: nur xdg-screensaver verfügbar");
                    }
                    Err(e) => {
                        return Err(format!(
                            "Weder systemd-inhibit noch xdg-screensaver verfügbar: {}",
                            e
                        ));
                    }
                }
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        win_power::prevent_sleep();
        s.sleep_prevented = true;
        println!("[PowerManager] ✅ Windows: SetThreadExecutionState gesetzt");
    }

    Ok(true)
}

/// Deaktiviert Sleep-Prevention.
#[tauri::command]
pub fn disable_prevent_sleep(
    state: tauri::State<'_, Mutex<PowerState>>,
) -> Result<bool, String> {
    let mut s = state.lock().map_err(|e| e.to_string())?;

    if !s.sleep_prevented {
        return Ok(false);
    }

    if let Some(mut child) = s.inhibitor_process.take() {
        let _ = child.kill();
        let _ = child.wait();
        println!("[PowerManager] ✅ Inhibitor-Prozess beendet");
    }

    #[cfg(target_os = "windows")]
    {
        win_power::allow_sleep();
        println!("[PowerManager] ✅ Windows: SetThreadExecutionState zurückgesetzt");
    }

    s.sleep_prevented = false;
    Ok(false)
}

/// Gibt zurück ob Sleep-Prevention gerade aktiv ist.
#[tauri::command]
pub fn get_prevent_sleep_status(
    state: tauri::State<'_, Mutex<PowerState>>,
) -> Result<bool, String> {
    let s = state.lock().map_err(|e| e.to_string())?;
    Ok(s.sleep_prevented)
}
