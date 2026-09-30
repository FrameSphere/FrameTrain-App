// hosting_desktop.rs – Zugriff auf gehostete Modelle ausserhalb des Hauptfensters
//
// • Schnell-Chat: rahmenloses Fenster "quickchat", immer im Vordergrund,
//   erscheint auf dem Bildschirm mit dem Mauszeiger (bzw. unter dem Tray-Symbol).
// • Tastenkuerzel (tauri-plugin-global-shortcut) und Doppeltipp (hotkey_tap.rs).
// • Tray-Symbol: macOS-Menueleiste, Windows-Infobereich, Linux-AppIndicator.
// • Bildschirmfoto mit den Bordmitteln des Systems.
// • Hauptfenster verstecken statt beenden, solange Modelle gehostet sind.

#[cfg(windows)]
use crate::command_ext::NoWindow;
use crate::hosting_manager::{self, HostStatus, SharedHosting};
use std::process::Command;
use std::sync::Mutex;
use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{Emitter, Manager, PhysicalPosition, PhysicalSize, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};

pub const QUICKCHAT: &str = "quickchat";
const TRAY_ID: &str = "frametrain-hosting";
const QC_WIDTH: f64 = 640.0;

/// Wo der Schnell-Chat haengt: oben fest (Kuerzel, macOS-Menueleiste) oder
/// unten fest (Windows-Taskleiste) — beim Wachsen bleibt diese Kante stehen.
#[derive(Clone, Copy, Debug)]
struct Anchor { x_center: f64, y: f64, from_bottom: bool }

#[derive(Default)]
pub struct DesktopState {
    anchor: Mutex<Option<Anchor>>,
    /// Aktuell registriertes Kuerzel (zum Abmelden beim Wechsel).
    shortcut: Mutex<Option<String>>,
    pub shortcut_error: Mutex<Option<String>>,
    /// Sprache der Oberflaeche fuer Tray-Texte ("de" | "en").
    lang: Mutex<String>,
    /// Zeitpunkt des letzten Oeffnens — ein Fokusverlust direkt danach
    /// (Tray-Klick auf Windows) soll das Fenster nicht gleich wieder schliessen.
    pub shown_at: Mutex<Option<std::time::Instant>>,
    /// Waehrend in den Einstellungen ein neues Kuerzel aufgenommen wird, darf
    /// das alte nicht greifen — sonst schluckt das System den Tastendruck.
    paused: Mutex<bool>,
}

fn tr(app: &tauri::AppHandle, de: &'static str, en: &'static str) -> &'static str {
    let lang = app.state::<DesktopState>().lang.lock().map(|l| l.clone()).unwrap_or_default();
    if lang == "en" { en } else { de }
}

// ============ Schnell-Chat-Fenster ============

fn quickchat_window(app: &tauri::AppHandle) -> Result<tauri::WebviewWindow, String> {
    if let Some(w) = app.get_webview_window(QUICKCHAT) {
        return Ok(w);
    }
    let b = WebviewWindowBuilder::new(app, QUICKCHAT, WebviewUrl::App("index.html".into()))
        .title("FrameTrain")
        .inner_size(QC_WIDTH, 96.0)
        .decorations(false)
        .transparent(true)
        .shadow(false)
        .always_on_top(true)
        .skip_taskbar(true)
        .resizable(false)
        .visible(false)
        .focused(true)
        .visible_on_all_workspaces(true);
    b.build().map_err(|e| format!("Schnell-Chat-Fenster: {}", e))
}

/// Legt das Fenster beim Start an, damit das erste Oeffnen ohne Ladezeit geht.
pub fn prewarm(app: &tauri::AppHandle) {
    let _ = quickchat_window(app);
}

fn monitor_at(app: &tauri::AppHandle, x: f64, y: f64) -> Option<tauri::Monitor> {
    app.monitor_from_point(x, y).ok().flatten()
        .or_else(|| app.primary_monitor().ok().flatten())
}

fn place(win: &tauri::WebviewWindow, a: Anchor) {
    let scale = win.scale_factor().unwrap_or(1.0);
    let size = win.outer_size().unwrap_or(PhysicalSize::new((QC_WIDTH * scale) as u32, (96.0 * scale) as u32));
    let x = a.x_center - size.width as f64 / 2.0;
    let y = if a.from_bottom { a.y - size.height as f64 } else { a.y };
    let _ = win.set_position(PhysicalPosition::new(x.round() as i32, y.round() as i32));
}

fn show_with_anchor(app: &tauri::AppHandle, anchor: Anchor, mode: &str) -> Result<(), String> {
    let win = quickchat_window(app)?;
    println!("[Hosting] Schnell-Chat oeffnen ({}) bei {:?}", mode, anchor);
    *app.state::<DesktopState>().anchor.lock().unwrap() = Some(anchor);
    place(&win, anchor);
    *app.state::<DesktopState>().shown_at.lock().unwrap() = Some(std::time::Instant::now());
    win.show().map_err(|e| e.to_string())?;
    let _ = win.set_focus();
    let _ = app.emit_to(QUICKCHAT, "quickchat-shown", serde_json::json!({ "mode": mode }));
    Ok(())
}

/// Oeffnet den Schnell-Chat mittig im oberen Drittel des Bildschirms mit dem Mauszeiger.
pub fn show_quickchat(app: &tauri::AppHandle) -> Result<(), String> {
    let cursor = app.cursor_position().unwrap_or(PhysicalPosition::new(0.0, 0.0));
    let mon = monitor_at(app, cursor.x, cursor.y).ok_or("Kein Bildschirm gefunden")?;
    let (pos, size) = (mon.position(), mon.size());
    let anchor = Anchor {
        x_center: pos.x as f64 + size.width as f64 / 2.0,
        y: pos.y as f64 + size.height as f64 * 0.22,
        from_bottom: false,
    };
    show_with_anchor(app, anchor, "hotkey")
}

pub fn toggle_quickchat(app: &tauri::AppHandle) {
    if let Some(w) = app.get_webview_window(QUICKCHAT) {
        if w.is_visible().unwrap_or(false) && w.is_focused().unwrap_or(false) {
            let _ = w.hide();
            return;
        }
    }
    if let Err(e) = show_quickchat(app) {
        eprintln!("[Hosting] Schnell-Chat: {}", e);
    }
}

/// Unter dem Tray-Symbol oeffnen (macOS: Menueleiste oben; Windows: Taskleiste
/// meist unten — dann waechst das Fenster nach oben).
fn show_at_tray(app: &tauri::AppHandle, rect: tauri::Rect) {
    let scale = app.primary_monitor().ok().flatten().map(|m| m.scale_factor()).unwrap_or(1.0);
    let pos = rect.position.to_physical::<f64>(scale);
    let size = rect.size.to_physical::<f64>(scale);
    let mon = monitor_at(app, pos.x, pos.y);
    let (mon_y, mon_h) = mon.as_ref()
        .map(|m| (m.position().y as f64, m.size().height as f64))
        .unwrap_or((0.0, 1080.0));
    let icon_mid_y = pos.y + size.height / 2.0;
    let from_bottom = icon_mid_y > mon_y + mon_h / 2.0;
    let gap = 6.0 * scale;
    let mut x_center = pos.x + size.width / 2.0;
    // Nicht ueber den Bildschirmrand hinaus
    if let Some(m) = mon.as_ref() {
        let half = QC_WIDTH * m.scale_factor() / 2.0 + 8.0;
        let left = m.position().x as f64 + half;
        let right = m.position().x as f64 + m.size().width as f64 - half;
        x_center = x_center.clamp(left, right.max(left));
    }
    let anchor = Anchor {
        x_center,
        y: if from_bottom { pos.y - gap } else { pos.y + size.height + gap },
        from_bottom,
    };
    let _ = show_with_anchor(app, anchor, "tray");
}

pub fn show_main(app: &tauri::AppHandle, view: Option<&str>) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.show();
        let _ = w.unminimize();
        let _ = w.set_focus();
    }
    if let Some(v) = view {
        let _ = app.emit_to("main", "hosting-open-view", v);
    }
}

// ============ Tray ============

fn build_tray_menu(app: &tauri::AppHandle) -> tauri::Result<Menu<tauri::Wry>> {
    let hosting = app.state::<SharedHosting>();
    let list = hosting_manager::list(&hosting);
    let shortcut = hosting.settings.lock().unwrap().shortcut.clone();
    let menu = Menu::new(app)?;
    menu.append(&MenuItem::with_id(app, "qc", tr(app, "Schnell-Chat öffnen", "Open quick chat"), true,
        if shortcut.is_empty() { None } else { Some(shortcut.as_str()) })?)?;
    menu.append(&MenuItem::with_id(app, "main", tr(app, "FrameTrain anzeigen", "Show FrameTrain"), true, None::<&str>)?)?;
    menu.append(&PredefinedMenuItem::separator(app)?)?;
    if list.is_empty() {
        menu.append(&MenuItem::with_id(app, "none", tr(app, "Keine Modelle gehostet", "No models hosted"), false, None::<&str>)?)?;
    }
    for h in &list {
        let state = match h.status {
            HostStatus::Ready => tr(app, "läuft", "running"),
            HostStatus::Loading => tr(app, "lädt", "loading"),
            HostStatus::Sleeping => tr(app, "schläft", "sleeping"),
            HostStatus::Error => tr(app, "Fehler", "error"),
            HostStatus::Idle => tr(app, "gestoppt", "stopped"),
        };
        let mark = if h.is_default { "● " } else { "   " };
        menu.append(&MenuItem::with_id(app, format!("model:{}", h.id), format!("{}{} — {}", mark, h.name, state), true, None::<&str>)?)?;
    }
    menu.append(&PredefinedMenuItem::separator(app)?)?;
    menu.append(&MenuItem::with_id(app, "stopall", tr(app, "Alle Modelle stoppen", "Stop all models"), !list.is_empty(), None::<&str>)?)?;
    menu.append(&MenuItem::with_id(app, "quit", tr(app, "FrameTrain beenden", "Quit FrameTrain"), true, None::<&str>)?)?;
    Ok(menu)
}

fn on_menu(app: &tauri::AppHandle, id: &str) {
    match id {
        "qc" => { let _ = show_quickchat(app); }
        "main" => show_main(app, None),
        "stopall" => hosting_manager::stop_all(app, &app.state::<SharedHosting>()),
        "quit" => {
            hosting_manager::stop_all(app, &app.state::<SharedHosting>());
            app.exit(0);
        }
        other => {
            if let Some(vid) = other.strip_prefix("model:") {
                // Modell zum Standard machen und den Schnell-Chat damit oeffnen.
                let st = app.state::<SharedHosting>();
                let _ = hosting_manager::hosting_update_model(app.clone(), st, vid.to_string(), None, Some(true));
                let _ = show_quickchat(app);
            }
        }
    }
}

fn ensure_tray(app: &tauri::AppHandle) -> tauri::Result<()> {
    if app.tray_by_id(TRAY_ID).is_some() {
        return Ok(());
    }
    // macOS: einfarbiges Vorlagenbild, das sich hell/dunkel der Menueleiste anpasst.
    // Windows/Linux: das farbige App-Symbol — Schwarz waere auf dunkler Taskleiste unsichtbar.
    let bytes: &[u8] = if cfg!(target_os = "macos") {
        include_bytes!("../icons/tray.png")
    } else {
        include_bytes!("../icons/32x32.png")
    };
    let icon = tauri::image::Image::from_bytes(bytes)?;
    let menu = build_tray_menu(app)?;
    let b = TrayIconBuilder::with_id(TRAY_ID)
        .icon(icon)
        .icon_as_template(cfg!(target_os = "macos"))
        .tooltip("FrameTrain Hosting")
        .menu(&menu)
        // Linksklick oeffnet den Schnell-Chat, Rechtsklick das Menue. Linux
        // (AppIndicator) kennt keine Klick-Events und zeigt immer das Menue.
        .show_menu_on_left_click(false)
        .on_menu_event(|app, e| on_menu(app, e.id().as_ref()))
        .on_tray_icon_event(|tray, e| {
            if let TrayIconEvent::Click { button: MouseButton::Left, button_state: MouseButtonState::Up, rect, .. } = e {
                let app = tray.app_handle();
                if let Some(w) = app.get_webview_window(QUICKCHAT) {
                    if w.is_visible().unwrap_or(false) {
                        let _ = w.hide();
                        return;
                    }
                }
                show_at_tray(app, rect);
            }
        });
    b.build(app)?;
    Ok(())
}

pub fn refresh_tray(app: &tauri::AppHandle) {
    let Some(tray) = app.tray_by_id(TRAY_ID) else { return };
    if let Ok(menu) = build_tray_menu(app) {
        let _ = tray.set_menu(Some(menu));
    }
}

fn remove_tray(app: &tauri::AppHandle) {
    let _ = app.remove_tray_by_id(TRAY_ID);
}

// ============ Einstellungen anwenden ============

fn apply_shortcut(app: &tauri::AppHandle, wanted: &str) {
    let ds = app.state::<DesktopState>();
    let mut cur = ds.shortcut.lock().unwrap();
    if cur.as_deref() == Some(wanted) && ds.shortcut_error.lock().unwrap().is_none() {
        return;
    }
    let gs = app.global_shortcut();
    if let Some(old) = cur.take() {
        let _ = gs.unregister(old.as_str());
    }
    *ds.shortcut_error.lock().unwrap() = None;
    if wanted.trim().is_empty() {
        return;
    }
    match gs.on_shortcut(wanted, |app, _sc, ev| {
        if ev.state() == ShortcutState::Pressed {
            println!("[Hosting] Kuerzel ausgeloest");
            toggle_quickchat(app);
        }
    }) {
        Ok(()) => *cur = Some(wanted.to_string()),
        Err(e) => {
            eprintln!("[Hosting] Kuerzel {} nicht registriert: {}", wanted, e);
            *ds.shortcut_error.lock().unwrap() = Some(format!("{}", e));
        }
    }
}

/// Setzt Kuerzel, Doppeltipp, Tray und API nach den gespeicherten Einstellungen.
pub fn apply_settings(app: &tauri::AppHandle) {
    let s = app.state::<SharedHosting>().settings.lock().unwrap().clone();
    let paused = *app.state::<DesktopState>().paused.lock().unwrap();
    // Hauptschalter aus → weder Kuerzel noch Doppeltipp noch Tray.
    apply_shortcut(app, if s.quick_enabled && !paused { &s.shortcut } else { "" });
    crate::hotkey_tap::set_key(if s.quick_enabled && !paused { &s.double_tap } else { "off" });
    if s.quick_enabled && s.tray_enabled {
        if let Err(e) = ensure_tray(app) { eprintln!("[Hosting] Tray: {}", e); }
        refresh_tray(app);
    } else {
        remove_tray(app);
    }
    crate::hosting_api::apply(app, s.api_enabled, s.api_port);
}

/// Einmal beim App-Start.
pub fn setup(app: &tauri::AppHandle) {
    let hosting = app.state::<SharedHosting>().inner().clone();
    apply_settings(app);
    prewarm(app);
    let ah = app.clone();
    crate::hotkey_tap::spawn(move || {
        let ah2 = ah.clone();
        let _ = ah.run_on_main_thread(move || toggle_quickchat(&ah2));
    });
    hosting_manager::prime_tasks(app, &hosting);
    hosting_manager::spawn_idle_watch(app.clone(), hosting.clone());
    hosting_manager::autoload(app, &hosting);
    hosting_manager::clean_uploads(app);
}

// ============ Bildschirmfoto ============

#[cfg(target_os = "macos")]
fn capture_to(path: &std::path::Path) -> Result<bool, String> {
    // -i: Bereich waehlen (Leertaste: Fenster), -x: ohne Ton. Abbruch mit Esc
    // liefert Status 0 ohne Datei.
    Command::new("screencapture").args(["-i", "-x"]).arg(path)
        .status().map_err(|e| format!("screencapture: {}", e))?;
    Ok(path.exists())
}

#[cfg(target_os = "windows")]
fn capture_to(path: &std::path::Path) -> Result<bool, String> {
    // Windows-Ausschneidewerkzeug (Win+Umschalt+S) oeffnen und warten, bis ein
    // neues Bild in der Zwischenablage liegt. Die Sequenznummer zeigt, ob sich
    // die Zwischenablage seit dem Start geaendert hat — ein altes Bild zaehlt nicht.
    let p = path.to_string_lossy().replace('\'', "''");
    let script = format!(r#"
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
Add-Type -Name Clip -Namespace FT -MemberDefinition '[DllImport("user32.dll")] public static extern uint GetClipboardSequenceNumber();'
$start = [FT.Clip]::GetClipboardSequenceNumber()
Start-Process 'ms-screenclip:'
$deadline = (Get-Date).AddSeconds(90)
while ((Get-Date) -lt $deadline) {{
  Start-Sleep -Milliseconds 250
  if ([FT.Clip]::GetClipboardSequenceNumber() -ne $start -and [System.Windows.Forms.Clipboard]::ContainsImage()) {{
    $img = [System.Windows.Forms.Clipboard]::GetImage()
    $img.Save('{}', [System.Drawing.Imaging.ImageFormat]::Png)
    exit 0
  }}
}}
exit 1
"#, p);
    let st = Command::new("powershell").no_window()
        .args(["-NoProfile", "-STA", "-ExecutionPolicy", "Bypass", "-Command", &script])
        .status().map_err(|e| format!("PowerShell: {}", e))?;
    Ok(st.success() && path.exists())
}

#[cfg(all(unix, not(target_os = "macos")))]
fn capture_to(path: &std::path::Path) -> Result<bool, String> {
    let p = path.to_string_lossy().to_string();
    let quoted = format!("'{}'", p.replace('\'', "'\\''"));
    // Erstes vorhandene Werkzeug gewinnt: GNOME, KDE, wlroots (grim+slurp), X11.
    let tools: Vec<(&str, Vec<String>)> = vec![
        ("gnome-screenshot", vec!["-a".into(), "-f".into(), p.clone()]),
        ("spectacle", vec!["-r".into(), "-b".into(), "-n".into(), "-o".into(), p.clone()]),
        ("sh", vec!["-c".into(), format!("command -v grim >/dev/null && command -v slurp >/dev/null && grim -g \"$(slurp)\" {}", quoted)]),
        ("flameshot", vec!["gui".into(), "-r".into()]),
        ("scrot", vec!["-s".into(), p.clone()]),
        ("import", vec![p.clone()]),
    ];
    let mut tried = false;
    for (bin, args) in tools {
        if bin != "sh" && Command::new("sh").args(["-c", &format!("command -v {} >/dev/null", bin)])
            .status().map(|s| !s.success()).unwrap_or(true) {
            continue;
        }
        tried = true;
        if bin == "flameshot" {
            // flameshot -r schreibt das PNG nach stdout.
            if let Ok(out) = Command::new(bin).args(&args).output() {
                if out.status.success() && !out.stdout.is_empty() {
                    std::fs::write(path, &out.stdout).map_err(|e| e.to_string())?;
                }
            }
        } else {
            let _ = Command::new(bin).args(&args).status();
        }
        if path.exists() {
            return Ok(true);
        }
        if bin != "sh" {
            // Werkzeug war da, der Nutzer hat abgebrochen.
            return Ok(false);
        }
    }
    if !tried {
        return Err("Kein Screenshot-Werkzeug gefunden. Installiere gnome-screenshot, spectacle, grim+slurp, flameshot oder scrot.".into());
    }
    Ok(false)
}

/// Bildschirmfoto fuer den Schnell-Chat: Fenster verstecken, Bereich waehlen
/// lassen, Pfad zurueck (None = abgebrochen).
#[tauri::command]
pub async fn hosting_capture_screenshot(app: tauri::AppHandle) -> Result<Option<String>, String> {
    let path = hosting_manager::new_capture_path(&app)?;
    let qc = app.get_webview_window(QUICKCHAT);
    let was_visible = qc.as_ref().map(|w| w.is_visible().unwrap_or(false)).unwrap_or(false);
    if let Some(w) = qc.as_ref() { let _ = w.hide(); }
    let main = app.get_webview_window("main");
    let main_visible = main.as_ref().map(|w| w.is_visible().unwrap_or(false) && w.is_focused().unwrap_or(false)).unwrap_or(false);
    // Kurz warten, bis das Fenster wirklich weg ist — sonst ist es auf dem Bild.
    tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    let p2 = path.clone();
    let res = tauri::async_runtime::spawn_blocking(move || capture_to(&p2)).await.map_err(|e| e.to_string())?;
    if was_visible {
        if let Some(w) = qc.as_ref() {
            *app.state::<DesktopState>().shown_at.lock().unwrap() = Some(std::time::Instant::now());
            let _ = w.show();
            let _ = w.set_focus();
        }
    } else if main_visible {
        if let Some(w) = main.as_ref() { let _ = w.set_focus(); }
    }
    match res {
        Ok(true) => Ok(Some(path.to_string_lossy().to_string())),
        Ok(false) => Ok(None),
        Err(e) => Err(e),
    }
}

// ============ Commands fuer die Fenster ============

#[tauri::command]
pub fn hosting_show_quickchat(app: tauri::AppHandle) -> Result<(), String> {
    show_quickchat(&app)
}

#[tauri::command]
pub fn hosting_hide_quickchat(app: tauri::AppHandle) {
    if let Some(w) = app.get_webview_window(QUICKCHAT) {
        let _ = w.hide();
    }
}

/// Der Schnell-Chat meldet seine Inhaltshoehe; die verankerte Kante bleibt stehen.
#[tauri::command]
pub fn hosting_quickchat_resize(app: tauri::AppHandle, height: f64) -> Result<(), String> {
    let win = app.get_webview_window(QUICKCHAT).ok_or("Kein Schnell-Chat")?;
    let h = height.clamp(56.0, 720.0);
    win.set_size(tauri::LogicalSize::new(QC_WIDTH, h)).map_err(|e| e.to_string())?;
    if let Some(a) = *app.state::<DesktopState>().anchor.lock().unwrap() {
        place(&win, a);
    }
    Ok(())
}

/// Blendet den Schnell-Chat bei Fokusverlust aus — ausser direkt nach dem
/// Oeffnen (der Tray-Klick selbst nimmt auf Windows kurz den Fokus).
#[tauri::command]
pub fn hosting_quickchat_blur(app: tauri::AppHandle) {
    let recent = app.state::<DesktopState>().shown_at.lock().unwrap()
        .map(|t| t.elapsed() < std::time::Duration::from_millis(400)).unwrap_or(false);
    if recent { return; }
    if let Some(w) = app.get_webview_window(QUICKCHAT) {
        let _ = w.hide();
    }
}

#[tauri::command]
pub fn hosting_open_main(app: tauri::AppHandle, view: Option<String>) {
    if let Some(w) = app.get_webview_window(QUICKCHAT) { let _ = w.hide(); }
    show_main(&app, view.as_deref());
}

/// Hauptfenster verstecken; Modelle, Tray und Kuerzel laufen weiter.
#[tauri::command]
pub fn hosting_hide_main(app: tauri::AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.hide();
    }
}

/// Kuerzel und Doppeltipp kurz aussetzen (Aufnahme eines neuen Kuerzels).
#[tauri::command]
pub fn hosting_pause_shortcut(app: tauri::AppHandle, paused: bool) {
    *app.state::<DesktopState>().paused.lock().unwrap() = paused;
    apply_settings(&app);
}

#[tauri::command]
pub fn hosting_set_ui_language(app: tauri::AppHandle, lang: String) {
    *app.state::<DesktopState>().lang.lock().unwrap() = if lang == "en" { "en".into() } else { "de".into() };
    refresh_tray(&app);
}

/// Laufzeit-Zustand fuer die Einstellungen: Kuerzel-Fehler, Doppeltipp-Unterstuetzung, API.
#[tauri::command]
pub fn hosting_desktop_status(app: tauri::AppHandle) -> serde_json::Value {
    let ds = app.state::<DesktopState>();
    let hosting = app.state::<SharedHosting>();
    serde_json::json!({
        "shortcut_error": ds.shortcut_error.lock().unwrap().clone(),
        "double_tap_supported": crate::hotkey_tap::supported(),
        "api": crate::hosting_api::status(&app),
        "any_loaded": hosting_manager::any_loaded(&hosting),
        "platform": std::env::consts::OS,
    })
}
