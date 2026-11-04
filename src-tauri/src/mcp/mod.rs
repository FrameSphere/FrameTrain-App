// mcp – FrameTrain per MCP steuerbar machen (Claude Desktop/Code, Cursor, ChatGPT, Claude.ai)
//
//   bridge.rs  Rust <-> Webview (Manifest, mcp://call, mcp_tool_result, system_ping)
//   server.rs  lokaler Endpunkt http://127.0.0.1:47861/mcp (Streamable HTTP, JSON)
//   stdio.rs   `--mcp-stdio`: Proxy stdin/stdout -> lokaler Endpunkt (fuer Claude Desktop)
//   relay.rs   Fernzugriff ueber frame-train.com (opt-in)
//   config.rs  <app_data>/mcp.json
//
// Vertrag: docs/mcp/CONTRACT.md

pub mod bridge;
pub mod config;
pub mod relay;
pub mod server;
pub mod stdio;

use bridge::{Bridge, ClientInfo, Source};
use config::McpConfig;
use serde_json::{json, Value};
use server::{Backend, ServerState};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tauri::Manager;

pub struct McpState {
    pub config: Mutex<McpConfig>,
    pub bridge: Arc<Bridge>,
    pub server: Arc<ServerState>,
    pub relay: relay::RelayState,
}

struct AppBackend {
    app: tauri::AppHandle,
}

impl Backend for AppBackend {
    fn version(&self) -> String {
        self.app.package_info().version.to_string()
    }

    fn tools(&self) -> Vec<Value> {
        let bridge = self.app.state::<McpState>().bridge.clone();
        // Gerade erst (z.B. von der Stdio-Bruecke) gestartet: kurz auf das Manifest warten.
        if !bridge.has_manifest() && bridge.recently_started() {
            bridge.wait_manifest(bridge::MANIFEST_WAIT);
        }
        bridge.tools()
    }

    fn call(&self, name: &str, args: Value, client: &ClientInfo) -> Value {
        bridge::call_via_app(&self.app, name, args, Source::Local, client.clone())
    }
}

/// Aus main().setup: Konfiguration laden, Endpunkt und ggf. Relay starten.
pub fn setup(app: &tauri::AppHandle) {
    let cfg = config::load_or_init(&config::config_path());
    app.manage(McpState {
        config: Mutex::new(cfg.clone()),
        bridge: Arc::new(Bridge::default()),
        server: Arc::new(ServerState::default()),
        relay: relay::RelayState::default(),
    });
    apply_server(app, &cfg);
    relay::apply(app, cfg.remote_enabled);
}

fn apply_server(app: &tauri::AppHandle, cfg: &McpConfig) {
    let st = app.state::<McpState>();
    let app2 = app.clone();
    server::apply(st.server.clone(), cfg.enabled, cfg.port, move || {
        let token = app2.state::<McpState>().config.lock().unwrap().token.clone();
        (token, Arc::new(AppBackend { app: app2.clone() }) as Arc<dyn Backend>)
    });
}

fn save(app: &tauri::AppHandle, cfg: &McpConfig) -> Result<(), String> {
    config::write_to(&config::config_path(), cfg)?;
    *app.state::<McpState>().config.lock().unwrap() = cfg.clone();
    Ok(())
}

// ============ Claude Desktop ============

pub fn claude_desktop_config_path() -> Option<PathBuf> {
    // macOS: ~/Library/Application Support/Claude, Windows: %APPDATA%\Claude, Linux: ~/.config/Claude
    dirs::config_dir().map(|d| d.join("Claude").join("claude_desktop_config.json"))
}

pub const SERVER_KEY: &str = "frametrain";

/// Traegt `frametrain` in eine bestehende claude_desktop_config.json ein, alles andere bleibt.
pub fn merge_claude_config(existing: Option<&str>, exe: &str) -> Result<String, String> {
    let mut root: Value = match existing.map(str::trim).filter(|s| !s.is_empty()) {
        None => json!({}),
        Some(s) => serde_json::from_str(s).map_err(|e| format!("CLAUDE_CONFIG_INVALID: {}", e))?,
    };
    let obj = root.as_object_mut().ok_or("CLAUDE_CONFIG_INVALID: kein JSON-Objekt")?;
    let servers = obj.entry("mcpServers").or_insert_with(|| json!({}));
    let servers = servers.as_object_mut().ok_or("CLAUDE_CONFIG_INVALID: mcpServers ist kein Objekt")?;
    servers.insert(SERVER_KEY.into(), json!({ "command": exe, "args": [stdio::FLAG] }));
    serde_json::to_string_pretty(&root).map_err(|e| e.to_string())
}

pub fn claude_config_has_entry(raw: &str, exe: &str) -> bool {
    serde_json::from_str::<Value>(raw).ok()
        .and_then(|v| v.get("mcpServers")?.get(SERVER_KEY)?.get("command")?.as_str().map(|c| c == exe))
        .unwrap_or(false)
}

/// Schreibt den Eintrag mit Sicherheitskopie. Rueckgabe: Pfad der Sicherung (falls es eine Datei gab).
pub fn write_claude_config(path: &Path, exe: &str) -> Result<Option<PathBuf>, String> {
    let dir = path.parent().ok_or("CLAUDE_NOT_FOUND")?;
    if !dir.exists() {
        return Err("CLAUDE_NOT_FOUND".into());
    }
    let existing = std::fs::read_to_string(path).ok();
    let merged = merge_claude_config(existing.as_deref(), exe)?;
    let backup = if existing.is_some() {
        let b = path.with_file_name(format!("claude_desktop_config.json.bak-frametrain-{}", chrono::Local::now().format("%Y%m%d-%H%M%S")));
        std::fs::copy(path, &b).map_err(|e| format!("Sicherung fehlgeschlagen: {}", e))?;
        Some(b)
    } else {
        None
    };
    let tmp = path.with_extension("json.frametrain-tmp");
    std::fs::write(&tmp, merged).map_err(|e| format!("Schreiben fehlgeschlagen: {}", e))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("Schreiben fehlgeschlagen: {}", e))?;
    Ok(backup)
}

fn exe_path() -> String {
    std::env::current_exe().map(|p| p.to_string_lossy().to_string()).unwrap_or_default()
}

// ============ Tauri-Commands ============

/// Webview meldet ihr Werkzeug-Manifest (beim Start und bei Aenderung).
#[tauri::command]
pub fn mcp_register_manifest(state: tauri::State<'_, McpState>, tools: Vec<Value>) -> usize {
    state.bridge.set_manifest(tools);
    state.bridge.tools().len()
}

/// Antwort der Webview auf ein `mcp://call`-Event.
#[tauri::command]
pub fn mcp_tool_result(state: tauri::State<'_, McpState>, call_id: String, result: Value) -> bool {
    state.bridge.resolve(&call_id, result)
}

#[tauri::command]
pub fn mcp_get_status(app: tauri::AppHandle) -> Value {
    let st = app.state::<McpState>();
    let cfg = st.config.lock().unwrap().clone();
    let exe = exe_path();
    let claude_path = claude_desktop_config_path();
    let claude_configured = claude_path.as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|raw| claude_config_has_entry(&raw, &exe))
        .unwrap_or(false);
    let relay = st.relay.status.lock().unwrap().clone();
    json!({
        "config": cfg,
        "server": st.server.status(),
        "relay": relay,
        "manifestTools": st.bridge.tools().len(),
        "manifestLoaded": st.bridge.has_manifest(),
        "pendingCalls": st.bridge.pending_count(),
        "recentCalls": st.bridge.recent_calls(),
        "localClients": st.server.sessions.clients(),
        "exePath": exe,
        // macOS startet nicht verschobene Apps aus einem Zufallsordner – der Pfad waere nach dem Neustart falsch.
        "exeTranslocated": exe.contains("/AppTranslocation/"),
        "claudeDesktop": {
            "configPath": claude_path.as_ref().map(|p| p.to_string_lossy().to_string()),
            "installed": claude_path.as_ref().and_then(|p| p.parent()).map(|d| d.exists()).unwrap_or(false),
            "configured": claude_configured,
        },
    })
}

#[tauri::command]
pub fn mcp_save_settings(app: tauri::AppHandle, enabled: bool, port: u16, remote_enabled: bool) -> Result<Value, String> {
    if port < 1024 {
        return Err("Port muss zwischen 1024 und 65535 liegen".into());
    }
    let mut cfg = app.state::<McpState>().config.lock().unwrap().clone();
    let remote_changed = cfg.remote_enabled != remote_enabled;
    cfg.enabled = enabled;
    cfg.port = port;
    cfg.remote_enabled = remote_enabled;
    save(&app, &cfg)?;
    apply_server(&app, &cfg);
    if remote_changed {
        relay::apply(&app, remote_enabled);
    }
    Ok(mcp_get_status(app))
}

#[tauri::command]
pub fn mcp_rotate_token(app: tauri::AppHandle) -> Result<Value, String> {
    let mut cfg = app.state::<McpState>().config.lock().unwrap().clone();
    cfg.token = config::new_token();
    save(&app, &cfg)?;
    Ok(mcp_get_status(app))
}

/// Traegt FrameTrain in Claude Desktop ein und schaltet den lokalen Endpunkt ein.
#[tauri::command]
pub fn mcp_connect_claude_desktop(app: tauri::AppHandle) -> Result<Value, String> {
    let path = claude_desktop_config_path().ok_or("CLAUDE_NOT_FOUND")?;
    let exe = exe_path();
    if exe.is_empty() {
        return Err("Programmpfad unbekannt".into());
    }
    let backup = write_claude_config(&path, &exe)?;
    let mut cfg = app.state::<McpState>().config.lock().unwrap().clone();
    if !cfg.enabled {
        cfg.enabled = true;
        save(&app, &cfg)?;
        apply_server(&app, &cfg);
    }
    Ok(json!({
        "configPath": path.to_string_lossy(),
        "backupPath": backup.map(|b| b.to_string_lossy().to_string()),
        "status": mcp_get_status(app),
    }))
}

/// Verbundene Clients und letzte Aufrufe laut Website (/api/desktop/mcp/status).
#[tauri::command]
pub async fn mcp_remote_status(app: tauri::AppHandle) -> Result<Value, String> {
    relay::remote_request(&app, "status", json!({})).await
}

#[tauri::command]
pub async fn mcp_remote_disconnect(app: tauri::AppHandle, connection_id: String) -> Result<Value, String> {
    relay::remote_request(&app, "disconnect", json!({ "connectionId": connection_id })).await
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXE: &str = "/Applications/FrameTrain.app/Contents/MacOS/FrameTrain";

    #[test]
    fn merge_into_empty_and_missing() {
        for existing in [None, Some(""), Some("  \n")] {
            let out: Value = serde_json::from_str(&merge_claude_config(existing, EXE).unwrap()).unwrap();
            assert_eq!(out["mcpServers"]["frametrain"]["command"], EXE);
            assert_eq!(out["mcpServers"]["frametrain"]["args"], json!(["--mcp-stdio"]));
        }
    }

    #[test]
    fn merge_keeps_other_entries() {
        let existing = r#"{
            "globalShortcut": "Alt+Space",
            "mcpServers": {
                "filesystem": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"] },
                "frametrain": { "command": "/alt/pfad", "args": [] }
            }
        }"#;
        let out: Value = serde_json::from_str(&merge_claude_config(Some(existing), EXE).unwrap()).unwrap();
        assert_eq!(out["globalShortcut"], "Alt+Space");
        assert_eq!(out["mcpServers"]["filesystem"]["command"], "npx");
        assert_eq!(out["mcpServers"]["frametrain"]["command"], EXE);
        assert!(claude_config_has_entry(&out.to_string(), EXE));
        assert!(!claude_config_has_entry(existing, EXE));
    }

    #[test]
    fn merge_refuses_broken_config() {
        assert!(merge_claude_config(Some("{kaputt"), EXE).unwrap_err().starts_with("CLAUDE_CONFIG_INVALID"));
        assert!(merge_claude_config(Some("[1,2]"), EXE).is_err());
        assert!(merge_claude_config(Some(r#"{"mcpServers": 5}"#), EXE).is_err());
    }

    #[test]
    fn write_with_backup() {
        let dir = std::env::temp_dir().join(format!("ft-claude-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("claude_desktop_config.json");
        // Ohne Datei: keine Sicherung
        assert!(write_claude_config(&path, EXE).unwrap().is_none());
        std::fs::write(&path, r#"{"mcpServers":{"x":{"command":"y"}}}"#).unwrap();
        let backup = write_claude_config(&path, EXE).unwrap().expect("Sicherung");
        assert_eq!(std::fs::read_to_string(&backup).unwrap(), r#"{"mcpServers":{"x":{"command":"y"}}}"#);
        let now: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(now["mcpServers"]["x"]["command"], "y");
        assert_eq!(now["mcpServers"]["frametrain"]["command"], EXE);
        // Kaputte Datei wird NICHT ueberschrieben
        std::fs::write(&path, "{kaputt").unwrap();
        assert!(write_claude_config(&path, EXE).is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "{kaputt");
        // Claude nicht installiert
        assert_eq!(write_claude_config(&dir.join("fehlt").join("c.json"), EXE).unwrap_err(), "CLAUDE_NOT_FOUND");
        let _ = std::fs::remove_dir_all(dir);
    }
}
