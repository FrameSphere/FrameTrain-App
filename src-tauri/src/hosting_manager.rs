// hosting_manager.rs – Modelle lokal hosten
//
// Anders als das Labor (ein Modell, wechselt mit der Auswahl) haelt das
// Hosting mehrere Modelle gleichzeitig geladen, jedes in seinem eigenen
// Python-Prozess. Angesprochen werden sie aus der Hosting-Seite, dem
// Schnell-Chat-Fenster, dem Tray-Menue und der lokalen API — alle gehen
// ueber `infer_on`. Start und Protokoll teilt es mit dem Labor (model_host.rs).
//
// Events:
//   "hosting-status" – HostInfo eines Modells nach jeder Aenderung
//   "hosting-token"  – { host_id, request_id, text } beim Streaming (LLM)

use crate::model_host::{self, InferInput, InferResult, ServerProc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tauri::{Emitter, Manager};

pub const EV_STATUS: &str = "hosting-status";
pub const EV_TOKEN: &str = "hosting-token";

/// Standard-Kuerzel fuer den Schnell-Chat: ⌘⇧Leertaste bzw. Strg+Umschalt+Leertaste.
/// Nicht ⌥Leertaste (Claude, ChatGPT, Raycast), nicht ⌘Leertaste (Spotlight),
/// nicht ⌃Leertaste (Eingabequelle). Der schnellere Weg ist ohnehin der
/// Doppeltipp auf Control (Standard, braucht keine Freigabe).
pub const DEFAULT_SHORTCUT: &str = "Super+Shift+Space";
pub const DEFAULT_SHORTCUT_OTHER: &str = "Control+Shift+Space";
/// Das Kuerzel aus 1.5.0 — viel zu umstaendlich, wird beim Laden ersetzt.
const OLD_SHORTCUTS: [&str; 2] = ["Control+Alt+Super+K", "Control+Alt+Shift+K"];
/// Stand des Einstellungsformats (fuer Umstellungen alter hosting.json).
const SETTINGS_VERSION: u32 = 2;
/// Port der lokalen API. 7860 (Gradio), 8000/8080 (Dev-Server), 11434
/// (Ollama) und 1234 (LM Studio) sind haeufig schon belegt.
pub const DEFAULT_API_PORT: u16 = 47_860;

// ============ Einstellungen (hosting.json) ============

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HostedModel {
    pub version_id: String,
    pub model_id:   String,
    /// Anzeigename, z. B. "SmolLM2 · v4"
    pub name:       String,
    /// Beim Start der App laden.
    #[serde(default)]
    pub autoload:   bool,
    /// Zuletzt gemeldete Aufgabe — damit Schnell-Chat und Seite schon vor dem
    /// Laden die richtige Eingabe zeigen (Bild statt Text bei YOLO).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modality:   Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task:       Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct HostingSettings {
    pub hosted:       Vec<HostedModel>,
    pub version:      u32,
    /// Hauptschalter fuer den Schnell-Zugriff (Kuerzel, Doppeltipp, Tray).
    pub quick_enabled: bool,
    /// Modell, das der Schnell-Chat anspricht.
    pub default_id:   Option<String>,
    /// Tastenkombination fuer den Schnell-Chat ("" = aus).
    pub shortcut:     String,
    /// Doppeltipp auf eine Sondertaste: "off" | "control" | "alt" | "shift" | "meta"
    pub double_tap:   String,
    /// Nach so vielen Minuten ohne Anfrage entladen (0 = nie). Die naechste
    /// Anfrage laedt das Modell wieder.
    pub idle_minutes: u32,
    pub api_enabled:  bool,
    pub api_port:     u16,
    pub api_token:    String,
    /// Symbol in der Menueleiste / im Infobereich zeigen.
    pub tray_enabled: bool,
    /// Wann der Schnell-Chat beim Oeffnen neu beginnt: "smart" (neu, ausser
    /// nach kurzer Unterbrechung oder wenn eine Antwort wartet) | "always" | "never"
    pub quick_session: String,
}

impl Default for HostingSettings {
    fn default() -> Self {
        HostingSettings {
            hosted: Vec::new(),
            version: SETTINGS_VERSION,
            quick_enabled: true,
            default_id: None,
            shortcut: default_shortcut().to_string(),
            double_tap: "control".to_string(),
            idle_minutes: 30,
            api_enabled: false,
            api_port: DEFAULT_API_PORT,
            api_token: String::new(),
            tray_enabled: true,
            quick_session: "smart".to_string(),
        }
    }
}

pub fn default_shortcut() -> &'static str {
    if cfg!(target_os = "macos") { DEFAULT_SHORTCUT } else { DEFAULT_SHORTCUT_OTHER }
}

pub fn new_token() -> String {
    use rand::Rng;
    const CHARS: &[u8] = b"abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789";
    let mut rng = rand::thread_rng();
    let body: String = (0..32).map(|_| CHARS[rng.gen_range(0..CHARS.len())] as char).collect();
    format!("ft-{}", body)
}

fn settings_path(app: &tauri::AppHandle) -> Option<PathBuf> {
    app.path().app_data_dir().ok().map(|d| d.join("hosting.json"))
}

/// Alte hosting.json (1.5.0) auf die neuen Standards heben: das
/// umstaendliche Kuerzel ersetzen und den Doppeltipp einschalten.
pub fn migrate(s: &mut HostingSettings) {
    if s.version < 2 {
        if OLD_SHORTCUTS.contains(&s.shortcut.as_str()) {
            s.shortcut = default_shortcut().to_string();
        }
        if s.double_tap == "off" {
            s.double_tap = "control".to_string();
        }
        s.quick_enabled = true;
    }
    s.version = SETTINGS_VERSION;
}

pub fn load_settings(app: &tauri::AppHandle) -> HostingSettings {
    let mut s: HostingSettings = settings_path(app)
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|t| {
            let mut v: serde_json::Value = serde_json::from_str(&t).ok()?;
            // Fehlt "version", stammt die Datei aus 1.5.0.
            if v.get("version").is_none() { v["version"] = 1.into(); }
            serde_json::from_value(v).ok()
        })
        .unwrap_or_default();
    migrate(&mut s);
    if s.api_token.is_empty() {
        s.api_token = new_token();
    }
    if s.api_port == 0 {
        s.api_port = DEFAULT_API_PORT;
    }
    s
}

pub fn save_settings_file(app: &tauri::AppHandle, s: &HostingSettings) -> Result<(), String> {
    let p = settings_path(app).ok_or("Kein App-Datenordner")?;
    if let Some(dir) = p.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let text = serde_json::to_string_pretty(s).map_err(|e| e.to_string())?;
    // Erst schreiben, dann umbenennen: ein Absturz mittendrin laesst die alte Datei stehen.
    let tmp = p.with_extension("json.tmp");
    std::fs::write(&tmp, text).map_err(|e| format!("hosting.json: {}", e))?;
    std::fs::rename(&tmp, &p).map_err(|e| format!("hosting.json: {}", e))
}

// ============ Laufzeit ============

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum HostStatus {
    /// Gehostet, aber nicht geladen (gestoppt oder nie gestartet).
    Idle,
    Loading,
    Ready,
    Error,
    /// Wegen Leerlauf entladen — die naechste Anfrage laedt es wieder.
    Sleeping,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostInfo {
    pub id:          String,
    pub model_id:    String,
    pub name:        String,
    /// Name fuer die API ("smollm2-v4").
    pub api_name:    String,
    pub status:      HostStatus,
    pub error:       Option<String>,
    pub modality:    Option<String>,
    pub input_kind:  Option<String>,
    pub classes:     Vec<String>,
    pub task:        Option<String>,
    pub is_default:  bool,
    pub autoload:    bool,
    /// Gerade eine Anfrage in Arbeit.
    pub busy:        bool,
    /// Groesse der Gewichte auf der Platte in GB (Richtwert fuer den RAM).
    pub size_gb:     Option<f64>,
    pub loaded_at:   Option<u64>,
    pub last_used:   Option<u64>,
    pub requests:    u64,
}

struct Slot {
    info: HostInfo,
    proc: Option<Arc<Mutex<ServerProc>>>,
    /// Steigt mit jedem Start/Stopp; ein veralteter Ladevorgang verwirft sein Ergebnis.
    generation: u64,
}

pub struct HostingState {
    slots:        Mutex<HashMap<String, Slot>>,
    pub settings: Mutex<HostingSettings>,
}

impl HostingState {
    pub fn new(settings: HostingSettings) -> Self {
        HostingState { slots: Mutex::new(HashMap::new()), settings: Mutex::new(settings) }
    }
}

pub type SharedHosting = Arc<HostingState>;

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// "SmolLM2 · v4" → "smollm2-v4": kleingeschrieben, nur a-z0-9 und Bindestriche.
pub fn api_slug(name: &str) -> String {
    let mut out = String::new();
    let mut dash = false;
    for c in name.chars().flat_map(|c| c.to_lowercase()) {
        let c = match c { 'ä' => 'a', 'ö' => 'o', 'ü' => 'u', 'ß' => 's', c => c };
        if c.is_ascii_alphanumeric() || c == '.' {
            out.push(c);
            dash = false;
        } else if !dash && !out.is_empty() {
            out.push('-');
            dash = true;
        }
    }
    while out.ends_with('-') { out.pop(); }
    if out.is_empty() { "modell".to_string() } else { out }
}

fn dir_size_gb(p: &Path) -> Option<f64> {
    fn walk(p: &Path, depth: u32) -> u64 {
        if depth > 6 { return 0; }
        let Ok(rd) = std::fs::read_dir(p) else { return 0 };
        rd.flatten().map(|e| {
            let Ok(md) = e.metadata() else { return 0 };
            if md.is_dir() {
                // Checkpoints und alte Versionen zaehlen nicht zu dem, was geladen wird.
                let n = e.file_name().to_string_lossy().to_string();
                if n.starts_with("checkpoint") || n == "versions" || n.starts_with('.') { 0 } else { walk(&e.path(), depth + 1) }
            } else { md.len() }
        }).sum()
    }
    let b = walk(p, 0);
    if b == 0 { None } else { Some((b as f64 / 1e9 * 100.0).round() / 100.0) }
}

fn info_for(cfg: &HostedModel, default_id: &Option<String>) -> HostInfo {
    HostInfo {
        id: cfg.version_id.clone(),
        model_id: cfg.model_id.clone(),
        name: cfg.name.clone(),
        api_name: api_slug(&cfg.name),
        status: HostStatus::Idle,
        error: None,
        modality: cfg.modality.clone(),
        input_kind: cfg.input_kind.clone(),
        classes: Vec::new(),
        task: cfg.task.clone(),
        is_default: default_id.as_deref() == Some(cfg.version_id.as_str()),
        autoload: cfg.autoload,
        busy: false,
        size_gb: None,
        loaded_at: None,
        last_used: None,
        requests: 0,
    }
}

/// Alle Slots an die Einstellungen angleichen (neue anlegen, entfernte beenden).
fn sync_slots(state: &HostingState) {
    let settings = state.settings.lock().unwrap().clone();
    let mut slots = state.slots.lock().unwrap();
    slots.retain(|id, _| settings.hosted.iter().any(|h| &h.version_id == id));
    for cfg in &settings.hosted {
        let slot = slots.entry(cfg.version_id.clone()).or_insert_with(|| Slot {
            info: info_for(cfg, &settings.default_id),
            proc: None,
            generation: 0,
        });
        slot.info.name = cfg.name.clone();
        slot.info.api_name = api_slug(&cfg.name);
        slot.info.autoload = cfg.autoload;
        slot.info.is_default = settings.default_id.as_deref() == Some(cfg.version_id.as_str());
    }
}

fn emit_info(app: &tauri::AppHandle, info: &HostInfo) {
    let _ = app.emit(EV_STATUS, info);
    crate::hosting_desktop::refresh_tray(app);
}

fn update_slot<F: FnOnce(&mut Slot)>(app: &tauri::AppHandle, state: &HostingState, id: &str, f: F) {
    let info = {
        let mut slots = state.slots.lock().unwrap();
        let Some(slot) = slots.get_mut(id) else { return };
        f(slot);
        slot.info.clone()
    };
    emit_info(app, &info);
}

pub fn list(state: &HostingState) -> Vec<HostInfo> {
    sync_slots(state);
    let order: Vec<String> = state.settings.lock().unwrap().hosted.iter().map(|h| h.version_id.clone()).collect();
    let slots = state.slots.lock().unwrap();
    order.iter().filter_map(|id| slots.get(id).map(|s| s.info.clone())).collect()
}

/// Laedt ein Modell (blockierend). Laeuft im Hintergrund-Thread oder in einer
/// Anfrage, die ein schlafendes Modell weckt.
fn load_blocking(app: &tauri::AppHandle, state: &HostingState, id: &str) -> Result<Arc<Mutex<ServerProc>>, String> {
    let generation = {
        let mut slots = state.slots.lock().unwrap();
        let slot = slots.get_mut(id).ok_or("Modell ist nicht gehostet")?;
        slot.proc = None;
        slot.generation += 1;
        slot.info.status = HostStatus::Loading;
        slot.info.error = None;
        slot.generation
    };
    update_slot(app, state, id, |_| {});

    let result = model_host::resolve_model(app, id)
        .and_then(|m| {
            let size = dir_size_gb(Path::new(&m.model_path));
            update_slot(app, state, id, |s| s.info.size_gb = size);
            model_host::spawn_server(app, &m, "Hosting")
        });

    let mut slots = state.slots.lock().unwrap();
    let Some(slot) = slots.get_mut(id) else {
        return Err("Modell wurde waehrend des Ladens entfernt".into());
    };
    if slot.generation != generation {
        return Err("Laden wurde abgebrochen".into());
    }
    let out = match result {
        Ok(proc) => {
            slot.info.status = HostStatus::Ready;
            slot.info.modality = Some(proc.modality.clone());
            slot.info.input_kind = Some(proc.input_kind.clone());
            slot.info.classes = proc.classes.clone();
            slot.info.task = proc.task.clone();
            slot.info.loaded_at = Some(now_secs());
            slot.info.last_used = Some(now_secs());
            let arc = Arc::new(Mutex::new(proc));
            slot.proc = Some(arc.clone());
            Ok(arc)
        }
        Err(e) => {
            slot.info.status = HostStatus::Error;
            slot.info.error = Some(e.clone());
            Err(e)
        }
    };
    let info = slot.info.clone();
    drop(slots);
    emit_info(app, &info);
    if out.is_ok() {
        remember_task(app, state, &info);
    }
    out
}

/// Vorlaeufige Aufgabe aus den Dateien, ohne Python zu starten — damit ein
/// nie geladenes Modell trotzdem die passende Eingabe zeigt (YOLO: Bild).
/// Vereinfachte Form von detect_modality in model_server.py; beim ersten
/// Laden gilt, was der Server meldet.
pub fn guess_task(dir: &Path, is_yolo: bool, is_canvas: bool) -> (String, String) {
    let pair = |m: &str, k: &str| (m.to_string(), k.to_string());
    if is_yolo { return pair("detect", "image"); }
    if is_canvas { return pair("canvas", "tensor"); }
    if dir.join("model_index.json").is_file() || dir.join("text_to_image_lora.json").is_file() {
        return pair("text_to_image", "text");
    }
    if dir.join("modules.json").is_file() || dir.join("sentence_bert_config.json").is_file() {
        return pair("embedding", "text");
    }
    let cfg: serde_json::Value = std::fs::read_to_string(dir.join("config.json")).ok()
        .and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
    let arch = cfg["architectures"][0].as_str().unwrap_or("");
    let mt = cfg["model_type"].as_str().unwrap_or("").to_lowercase();
    let audio = ["wav2vec2", "hubert", "wavlm", "whisper", "audio-spectrogram-transformer", "ast", "data2vec-audio"];
    let image = ["resnet", "vit", "deit", "beit", "convnext", "swin", "efficientnet", "mobilenet_v2", "mobilevit", "regnet", "dinov2"];
    let video = ["videomae", "timesformer", "vivit"];
    let vlm = cfg.get("vision_config").is_some() && cfg.get("text_config").is_some() && !mt.contains("whisper");
    if vlm { return pair("vlm", "image"); }
    if arch.ends_with("ForImageClassification") || image.contains(&mt.as_str()) { return pair("image", "image"); }
    if arch.ends_with("ForVideoClassification") || video.contains(&mt.as_str()) { return pair("video", "video"); }
    if arch.ends_with("ForCTC") || arch.ends_with("ForSpeechSeq2Seq") || mt == "whisper" { return pair("asr", "audio"); }
    if arch.ends_with("ForAudioClassification") || audio.contains(&mt.as_str()) { return pair("audio", "audio"); }
    if arch.ends_with("ForTokenClassification") { return pair("token", "text"); }
    if arch.ends_with("ForConditionalGeneration") || arch.ends_with("ForSeq2SeqLM") { return pair("seq2seq", "text"); }
    if arch.ends_with("ForCausalLM") || arch.ends_with("LMHeadModel") { return pair("causal_lm", "text"); }
    pair("text", "text")
}

/// Fuer gehostete Modelle ohne gemerkte Aufgabe eine vorlaeufige eintragen.
pub fn prime_tasks(app: &tauri::AppHandle, state: &HostingState) {
    let missing: Vec<String> = state.settings.lock().unwrap().hosted.iter()
        .filter(|h| h.modality.is_none()).map(|h| h.version_id.clone()).collect();
    if missing.is_empty() { return; }
    for id in &missing {
        let Ok(m) = model_host::resolve_model(app, id) else { continue };
        let (modality, input_kind) = guess_task(Path::new(&m.model_path), m.is_yolo, m.is_canvas);
        {
            let mut s = state.settings.lock().unwrap();
            if let Some(h) = s.hosted.iter_mut().find(|h| &h.version_id == id) {
                h.modality = Some(modality.clone());
                h.input_kind = Some(input_kind.clone());
            }
        }
        let mut slots = state.slots.lock().unwrap();
        if let Some(slot) = slots.get_mut(id) {
            if slot.info.modality.is_none() {
                slot.info.modality = Some(modality);
                slot.info.input_kind = Some(input_kind);
            }
        }
    }
    let snapshot = state.settings.lock().unwrap().clone();
    let _ = save_settings_file(app, &snapshot);
}

/// Aufgabe in hosting.json merken (nur wenn sie sich geaendert hat).
fn remember_task(app: &tauri::AppHandle, state: &HostingState, info: &HostInfo) {
    let snapshot = {
        let mut s = state.settings.lock().unwrap();
        let Some(h) = s.hosted.iter_mut().find(|h| h.version_id == info.id) else { return };
        if h.modality == info.modality && h.input_kind == info.input_kind && h.task == info.task {
            return;
        }
        h.modality = info.modality.clone();
        h.input_kind = info.input_kind.clone();
        h.task = info.task.clone();
        s.clone()
    };
    let _ = save_settings_file(app, &snapshot);
}

pub fn start_in_background(app: &tauri::AppHandle, state: &SharedHosting, id: &str) {
    let (app, state, id) = (app.clone(), state.clone(), id.to_string());
    std::thread::spawn(move || {
        if let Err(e) = load_blocking(&app, &state, &id) {
            eprintln!("[Hosting] {} konnte nicht geladen werden: {}", id, e);
        }
    });
}

/// Liefert den Server eines Modells und laedt es bei Bedarf (schlafend,
/// gestoppt oder gerade im Laden).
fn ensure_loaded(app: &tauri::AppHandle, state: &HostingState, id: &str) -> Result<Arc<Mutex<ServerProc>>, String> {
    let deadline = std::time::Instant::now() + Duration::from_secs(150);
    loop {
        let (status, proc) = {
            let slots = state.slots.lock().unwrap();
            let slot = slots.get(id).ok_or("Modell ist nicht gehostet")?;
            (slot.info.status.clone(), slot.proc.clone())
        };
        match (status, proc) {
            (HostStatus::Ready, Some(p)) => return Ok(p),
            (HostStatus::Loading, _) => {
                if std::time::Instant::now() > deadline {
                    return Err("Modell laedt noch — bitte gleich noch einmal versuchen.".into());
                }
                std::thread::sleep(Duration::from_millis(250));
            }
            _ => return load_blocking(app, state, id),
        }
    }
}

/// Findet ein Modell ueber Versions-ID, API-Namen oder Anzeigenamen.
pub fn find_id(state: &HostingState, key: &str) -> Option<String> {
    sync_slots(state);
    let slots = state.slots.lock().unwrap();
    let key_l = key.trim().to_lowercase();
    slots.values().find(|s| s.info.id == key).map(|s| s.info.id.clone())
        .or_else(|| slots.values().find(|s| s.info.api_name == key_l).map(|s| s.info.id.clone()))
        .or_else(|| slots.values().find(|s| s.info.name.to_lowercase() == key_l).map(|s| s.info.id.clone()))
}

pub fn default_or_first(state: &HostingState) -> Option<String> {
    let d = state.settings.lock().unwrap().default_id.clone();
    let list = list(state);
    d.filter(|d| list.iter().any(|h| &h.id == d))
        .or_else(|| list.iter().find(|h| h.status == HostStatus::Ready).map(|h| h.id.clone()))
        .or_else(|| list.first().map(|h| h.id.clone()))
}

/// Eine Anfrage an ein gehostetes Modell. Schlafende Modelle werden geweckt,
/// Token gehen beim Streaming an `on_token`.
pub fn infer_on(
    app: &tauri::AppHandle,
    state: &HostingState,
    id: &str,
    input: &InferInput,
    on_token: &mut dyn FnMut(&str),
) -> Result<InferResult, String> {
    let proc = ensure_loaded(app, state, id)?;
    update_slot(app, state, id, |s| s.info.busy = true);
    let outcome = {
        let mut p = proc.lock().map_err(|e| format!("Lock: {}", e))?;
        model_host::build_request(p.is_canvas, &p.input_kind, &p.modality, input)
            .and_then(|req| {
                let timeout = model_host::infer_timeout_secs(&p.modality);
                p.request(&req, timeout, on_token).map_err(|e| {
                    if matches!(e, model_host::RequestError::Crashed) { "__crashed__".to_string() } else { e.message() }
                })
            })
    };
    let crashed = matches!(&outcome, Err(e) if e == "__crashed__");
    update_slot(app, state, id, |s| {
        s.info.busy = false;
        s.info.last_used = Some(now_secs());
        if outcome.is_ok() { s.info.requests += 1; }
        if crashed {
            s.proc = None;
            s.info.status = HostStatus::Error;
            s.info.error = Some(model_host::RequestError::Crashed.message());
        }
    });
    match outcome {
        Ok(resp) => model_host::parse_result(resp),
        Err(_) if crashed => Err(model_host::RequestError::Crashed.message()),
        Err(e) => Err(e),
    }
}

fn stop_slot(app: &tauri::AppHandle, state: &HostingState, id: &str, status: HostStatus) {
    update_slot(app, state, id, |s| {
        s.generation += 1;
        s.proc = None; // Drop beendet den Prozess
        s.info.status = status;
        s.info.busy = false;
    });
}

/// Entlaedt Modelle, die laenger als `idle_minutes` nicht gefragt wurden.
pub fn spawn_idle_watch(app: tauri::AppHandle, state: SharedHosting) {
    std::thread::spawn(move || loop {
        std::thread::sleep(Duration::from_secs(30));
        let minutes = state.settings.lock().unwrap().idle_minutes;
        if minutes == 0 { continue; }
        let limit = minutes as u64 * 60;
        let now = now_secs();
        let sleepy: Vec<String> = {
            let slots = state.slots.lock().unwrap();
            slots.values()
                .filter(|s| s.info.status == HostStatus::Ready && !s.info.busy)
                .filter(|s| now.saturating_sub(s.info.last_used.unwrap_or(now)) >= limit)
                .map(|s| s.info.id.clone())
                .collect()
        };
        for id in sleepy {
            // Nur, wenn gerade niemand rechnet (try_lock statt warten).
            let free = {
                let slots = state.slots.lock().unwrap();
                slots.get(&id).and_then(|s| s.proc.clone()).map(|p| p.try_lock().is_ok()).unwrap_or(false)
            };
            if free {
                println!("[Hosting] {} nach {} min Leerlauf entladen", id, minutes);
                stop_slot(&app, &state, &id, HostStatus::Sleeping);
            }
        }
    });
}

/// Beim App-Start: Modelle mit "beim Start laden" hochfahren.
pub fn autoload(app: &tauri::AppHandle, state: &SharedHosting) {
    sync_slots(state);
    let ids: Vec<String> = state.settings.lock().unwrap().hosted.iter()
        .filter(|h| h.autoload).map(|h| h.version_id.clone()).collect();
    for id in ids {
        start_in_background(app, state, &id);
    }
}

pub fn any_loaded(state: &HostingState) -> bool {
    state.slots.lock().unwrap().values()
        .any(|s| matches!(s.info.status, HostStatus::Ready | HostStatus::Loading | HostStatus::Sleeping))
}

pub fn stop_all(app: &tauri::AppHandle, state: &HostingState) {
    let ids: Vec<String> = state.slots.lock().unwrap().keys().cloned().collect();
    for id in ids {
        stop_slot(app, state, &id, HostStatus::Idle);
    }
}

// ============ Commands ============

type St<'a> = tauri::State<'a, SharedHosting>;

#[tauri::command]
pub fn hosting_list(app: tauri::AppHandle, state: St<'_>) -> Vec<HostInfo> {
    sync_slots(&state);
    prime_tasks(&app, &state);
    list(&state)
}

#[tauri::command]
pub fn hosting_get_settings(state: St<'_>) -> HostingSettings {
    state.settings.lock().unwrap().clone()
}

/// Uebernimmt geaenderte Einstellungen (Kuerzel, Doppeltipp, API, Tray, Leerlauf).
/// Die Liste der gehosteten Modelle bleibt dabei unangetastet — dafuer gibt es
/// eigene Befehle.
#[tauri::command]
pub fn hosting_save_settings(app: tauri::AppHandle, state: St<'_>, settings: HostingSettings) -> Result<HostingSettings, String> {
    let merged = {
        let mut cur = state.settings.lock().unwrap();
        let hosted = cur.hosted.clone();
        let default_id = cur.default_id.clone();
        *cur = HostingSettings { hosted, default_id, version: SETTINGS_VERSION, ..settings };
        if cur.api_token.is_empty() { cur.api_token = new_token(); }
        if cur.api_port < 1024 { cur.api_port = DEFAULT_API_PORT; }
        cur.clone()
    };
    save_settings_file(&app, &merged)?;
    crate::hosting_desktop::apply_settings(&app);
    Ok(merged)
}

#[tauri::command]
pub fn hosting_rotate_token(app: tauri::AppHandle, state: St<'_>) -> Result<String, String> {
    let s = {
        let mut cur = state.settings.lock().unwrap();
        cur.api_token = new_token();
        cur.clone()
    };
    save_settings_file(&app, &s)?;
    Ok(s.api_token)
}

/// Nimmt ein Modell ins Hosting auf und laedt es im Hintergrund.
#[tauri::command]
pub fn hosting_start(
    app: tauri::AppHandle,
    state: St<'_>,
    version_id: String,
    model_id: String,
    name: String,
    autoload: Option<bool>,
    make_default: Option<bool>,
) -> Result<(), String> {
    {
        let mut s = state.settings.lock().unwrap();
        match s.hosted.iter_mut().find(|h| h.version_id == version_id) {
            Some(h) => {
                h.name = name.clone();
                if let Some(a) = autoload { h.autoload = a; }
            }
            None => s.hosted.push(HostedModel {
                version_id: version_id.clone(),
                model_id,
                name,
                autoload: autoload.unwrap_or(false),
                modality: None,
                input_kind: None,
                task: None,
            }),
        }
        if make_default.unwrap_or(false) || s.default_id.is_none() {
            s.default_id = Some(version_id.clone());
        }
        save_settings_file(&app, &s)?;
    }
    sync_slots(&state);
    start_in_background(&app, &state, &version_id);
    Ok(())
}

#[tauri::command]
pub fn hosting_stop(app: tauri::AppHandle, state: St<'_>, id: String) {
    stop_slot(&app, &state, &id, HostStatus::Idle);
}

#[tauri::command]
pub fn hosting_remove(app: tauri::AppHandle, state: St<'_>, id: String) -> Result<(), String> {
    stop_slot(&app, &state, &id, HostStatus::Idle);
    let s = {
        let mut s = state.settings.lock().unwrap();
        s.hosted.retain(|h| h.version_id != id);
        if s.default_id.as_deref() == Some(id.as_str()) {
            s.default_id = s.hosted.first().map(|h| h.version_id.clone());
        }
        s.clone()
    };
    save_settings_file(&app, &s)?;
    sync_slots(&state);
    let _ = app.emit(EV_STATUS, serde_json::json!({ "id": id, "removed": true }));
    for h in list(&state) { emit_info(&app, &h); }
    Ok(())
}

#[tauri::command]
pub fn hosting_update_model(app: tauri::AppHandle, state: St<'_>, id: String, autoload: Option<bool>, make_default: Option<bool>) -> Result<(), String> {
    let s = {
        let mut s = state.settings.lock().unwrap();
        if let Some(h) = s.hosted.iter_mut().find(|h| h.version_id == id) {
            if let Some(a) = autoload { h.autoload = a; }
        }
        if make_default == Some(true) { s.default_id = Some(id.clone()); }
        s.clone()
    };
    save_settings_file(&app, &s)?;
    sync_slots(&state);
    for h in list(&state) { emit_info(&app, &h); }
    Ok(())
}

/// Inferenz aus Hosting-Seite oder Schnell-Chat. `request_id` ordnet die
/// Token-Events der richtigen Antwort zu.
#[tauri::command]
pub async fn hosting_infer(
    app: tauri::AppHandle,
    state: St<'_>,
    id: String,
    request_id: String,
    input: InferInput,
) -> Result<InferResult, String> {
    let state = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let ah = app.clone();
        let (hid, rid) = (id.clone(), request_id.clone());
        let mut on_token = |t: &str| {
            let _ = ah.emit(EV_TOKEN, serde_json::json!({ "host_id": hid, "request_id": rid, "text": t }));
        };
        infer_on(&app, &state, &id, &input, &mut on_token)
    })
    .await
    .map_err(|e| e.to_string())?
}

fn uploads_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let d = app.path().app_cache_dir().map_err(|e| e.to_string())?.join("hosting_uploads");
    std::fs::create_dir_all(&d).map_err(|e| e.to_string())?;
    Ok(d)
}

/// Aufraeumen: Uploads und Screenshots aelter als einen Tag.
pub fn clean_uploads(app: &tauri::AppHandle) {
    let Ok(d) = uploads_dir(app) else { return };
    let limit = SystemTime::now() - Duration::from_secs(24 * 3600);
    for e in std::fs::read_dir(d).into_iter().flatten().flatten() {
        if e.metadata().and_then(|m| m.modified()).map(|t| t < limit).unwrap_or(false) {
            let _ = std::fs::remove_file(e.path());
        }
    }
}

pub fn safe_ext(ext: &str) -> String {
    let e: String = ext.trim_start_matches('.').chars().filter(|c| c.is_ascii_alphanumeric()).take(5).collect::<String>().to_lowercase();
    if e.is_empty() { "bin".into() } else { e }
}

pub fn write_upload(app: &tauri::AppHandle, bytes: &[u8], ext: &str) -> Result<String, String> {
    if bytes.is_empty() { return Err("Leere Datei".into()); }
    let p = uploads_dir(app)?.join(format!("{}_{}.{}", now_secs(), uuid::Uuid::new_v4().simple(), safe_ext(ext)));
    std::fs::write(&p, bytes).map_err(|e| e.to_string())?;
    Ok(p.to_string_lossy().to_string())
}

/// Eingefuegte Bilder, Mikrofonaufnahmen usw.: der Python-Server liest Dateien,
/// also landen sie als Datei im Cache.
#[tauri::command]
pub fn hosting_save_upload(app: tauri::AppHandle, bytes: Vec<u8>, ext: String) -> Result<String, String> {
    write_upload(&app, &bytes, &ext)
}

pub fn new_capture_path(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    Ok(uploads_dir(app)?.join(format!("screenshot_{}.png", now_secs())))
}

fn chats_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let d = app.path().app_data_dir().map_err(|e| e.to_string())?.join("hosting_chats");
    std::fs::create_dir_all(&d).map_err(|e| e.to_string())?;
    Ok(d)
}

fn chat_file(app: &tauri::AppHandle, id: &str) -> Result<PathBuf, String> {
    let safe: String = id.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_').collect();
    if safe.is_empty() { return Err("Ungueltige ID".into()); }
    Ok(chats_dir(app)?.join(format!("{}.json", safe)))
}

#[tauri::command]
pub fn hosting_chat_load(app: tauri::AppHandle, id: String) -> Result<serde_json::Value, String> {
    let p = chat_file(&app, &id)?;
    Ok(std::fs::read_to_string(p).ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_else(|| serde_json::json!([])))
}

/// Speichert den Verlauf eines Modells (die neuesten 200 Nachrichten).
#[tauri::command]
pub fn hosting_chat_save(app: tauri::AppHandle, id: String, messages: Vec<serde_json::Value>) -> Result<(), String> {
    let p = chat_file(&app, &id)?;
    let start = messages.len().saturating_sub(200);
    let text = serde_json::to_string(&messages[start..]).map_err(|e| e.to_string())?;
    std::fs::write(p, text).map_err(|e| e.to_string())
}

/// Haengt Nachrichten aus dem Schnell-Chat an den Verlauf des Modells an —
/// so findet man sie spaeter auf der Hosting-Seite wieder.
#[tauri::command]
pub fn hosting_chat_append(app: tauri::AppHandle, id: String, messages: Vec<serde_json::Value>) -> Result<(), String> {
    if messages.is_empty() { return Ok(()); }
    let p = chat_file(&app, &id)?;
    let mut all: Vec<serde_json::Value> = std::fs::read_to_string(&p).ok()
        .and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
    // Dieselbe Nachricht nicht doppelt (ein Chat wird nach jeder Antwort gesichert).
    for m in messages {
        let mid = m.get("id").and_then(|v| v.as_str()).map(str::to_string);
        match mid.and_then(|mid| all.iter().position(|x| x.get("id").and_then(|v| v.as_str()) == Some(mid.as_str()))) {
            Some(i) => all[i] = m,
            None => all.push(m),
        }
    }
    let start = all.len().saturating_sub(200);
    let text = serde_json::to_string(&all[start..]).map_err(|e| e.to_string())?;
    std::fs::write(p, text).map_err(|e| e.to_string())?;
    let _ = app.emit("hosting-chat-changed", serde_json::json!({ "id": id }));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_namen_sind_url_tauglich() {
        assert_eq!(api_slug("SmolLM2 · v4"), "smollm2-v4");
        assert_eq!(api_slug("Größe/Übung 2"), "grose-ubung-2");
        assert_eq!(api_slug("qwen2.5-0.5b"), "qwen2.5-0.5b");
        assert_eq!(api_slug("  ·  "), "modell");
    }

    #[test]
    fn standard_kuerzel_kollidiert_nicht() {
        let s = default_shortcut().to_lowercase();
        // ⌥Leertaste (Claude/Raycast), ⌘Leertaste (Spotlight), ⌃Leertaste (Eingabequelle)
        assert!(!["alt+space", "super+space", "control+space"].contains(&s.as_str()));
        assert!(s.contains("shift"), "mit Umschalt, damit es frei ist");
        assert_eq!(s.matches('+').count(), 2, "nur zwei Sondertasten – einfach zu druecken");
    }

    #[test]
    fn neue_einstellungen_haben_doppeltipp_und_schnellzugriff() {
        let s = HostingSettings::default();
        assert_eq!(s.double_tap, "control");
        assert!(s.quick_enabled && s.tray_enabled && !s.api_enabled);
        assert_eq!(s.api_port, DEFAULT_API_PORT);
    }

    #[test]
    fn datei_aus_150_wird_umgestellt() {
        let mut s: HostingSettings = serde_json::from_str(
            r#"{"hosted":[],"version":1,"shortcut":"Control+Alt+Super+K","double_tap":"off"}"#).unwrap();
        migrate(&mut s);
        assert_eq!(s.shortcut, default_shortcut());
        assert_eq!(s.double_tap, "control");
        assert_eq!(s.version, SETTINGS_VERSION);
        // Eigene Wahl bleibt
        let mut e: HostingSettings = serde_json::from_str(
            r#"{"hosted":[],"version":2,"shortcut":"Alt+Shift+J","double_tap":"off","quick_enabled":false}"#).unwrap();
        migrate(&mut e);
        assert_eq!(e.shortcut, "Alt+Shift+J");
        assert_eq!(e.double_tap, "off");
        assert!(!e.quick_enabled);
    }

    #[test]
    fn gemerkte_aufgabe_gilt_schon_vor_dem_laden() {
        let mut s = HostingSettings::default();
        s.hosted.push(HostedModel { version_id: "y".into(), model_id: "m".into(), name: "YOLO".into(), autoload: false,
            modality: Some("detect".into()), input_kind: Some("image".into()), task: Some("detect".into()) });
        let st = HostingState::new(s);
        let l = list(&st);
        assert_eq!(l[0].input_kind.as_deref(), Some("image"));
        assert_eq!(l[0].status, HostStatus::Idle);
    }

    #[test]
    fn token_ist_lang_und_zufaellig() {
        let a = new_token();
        assert!(a.starts_with("ft-") && a.len() == 35);
        assert_ne!(a, new_token());
    }

    #[test]
    fn aufgabe_wird_ohne_python_geschaetzt() {
        let dir = std::env::temp_dir().join(format!("ft_guess_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(guess_task(&dir, true, false).1, "image");
        std::fs::write(dir.join("config.json"), r#"{"architectures":["LlamaForCausalLM"],"model_type":"llama"}"#).unwrap();
        assert_eq!(guess_task(&dir, false, false).0, "causal_lm");
        std::fs::write(dir.join("config.json"), r#"{"architectures":["WhisperForConditionalGeneration"],"model_type":"whisper"}"#).unwrap();
        assert_eq!(guess_task(&dir, false, false), ("asr".to_string(), "audio".to_string()));
        std::fs::write(dir.join("config.json"), r#"{"architectures":["ViTForImageClassification"],"model_type":"vit"}"#).unwrap();
        assert_eq!(guess_task(&dir, false, false).1, "image");
        std::fs::write(dir.join("config.json"), r#"{"model_type":"idefics3","vision_config":{},"text_config":{}}"#).unwrap();
        assert_eq!(guess_task(&dir, false, false).0, "vlm");
        std::fs::write(dir.join("modules.json"), "[]").unwrap();
        assert_eq!(guess_task(&dir, false, false).0, "embedding");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dateiendung_wird_bereinigt() {
        assert_eq!(safe_ext(".PNG"), "png");
        assert_eq!(safe_ext("../../x"), "x");
        assert_eq!(safe_ext(""), "bin");
    }

    #[test]
    fn slots_folgen_den_einstellungen() {
        let mut s = HostingSettings::default();
        s.hosted.push(HostedModel { version_id: "v1".into(), model_id: "m".into(), name: "A".into(), autoload: false, modality: None, input_kind: None, task: None });
        s.hosted.push(HostedModel { version_id: "v2".into(), model_id: "m".into(), name: "B".into(), autoload: true, modality: None, input_kind: None, task: None });
        s.default_id = Some("v2".into());
        let st = HostingState::new(s);
        let l = list(&st);
        assert_eq!(l.len(), 2);
        assert_eq!(l[0].id, "v1");
        assert!(l[1].is_default && l[1].autoload);
        assert_eq!(find_id(&st, "b").as_deref(), Some("v2"));
        assert_eq!(find_id(&st, "v1").as_deref(), Some("v1"));
        assert_eq!(default_or_first(&st).as_deref(), Some("v2"));
        st.settings.lock().unwrap().hosted.remove(0);
        assert_eq!(list(&st).len(), 1);
    }
}
