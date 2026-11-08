// mcp/relay.rs – Relay-Client fuer den Fernzugriff ueber frame-train.com (Vertrag Abschnitt 5)
//
// Laeuft nur, wenn `remoteEnabled` an ist. Zwei Schleifen:
//   - Heartbeat alle 30 s (+ Manifest hochladen, wenn die Website es will)
//   - Long-Poll auf Jobs; jeder Job geht durch dieselbe Bridge mit source "remote"
//     (also auch durch den Freigabe-Dialog), das Ergebnis per /result zurueck.
// Netzfehler: Backoff 2 -> 60 s. Auth wie alle /api/desktop/*-Routen: apiKey im Body.

use super::bridge::{self, ClientInfo, Source};
use serde::Serialize;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;
use tauri::Manager;

const HEARTBEAT_EVERY: Duration = Duration::from_secs(30);
const POLL_TIMEOUT: Duration = Duration::from_secs(40); // Server haelt bis 25 s
const BACKOFF_MIN: u64 = 2;
const BACKOFF_MAX: u64 = 60;

pub fn url(path: &str) -> String {
    format!("{}/desktop/mcp/{}", crate::api_config::get_api_base_url(), path)
}

#[derive(Debug, Clone, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct RelayStatus {
    pub running: bool,
    pub connected: bool,
    pub last_heartbeat: Option<String>,
    pub last_error: Option<String>,
    pub jobs_handled: u64,
}

#[derive(Default)]
pub struct RelayState {
    /// Jede Aenderung (an/aus) erhoeht die Generation; alte Schleifen beenden sich.
    generation: AtomicU64,
    pub status: Mutex<RelayStatus>,
}

/// Naechste Wartezeit nach einem Fehler.
pub fn next_backoff(current: u64) -> u64 {
    if current == 0 { BACKOFF_MIN } else { (current * 2).min(BACKOFF_MAX) }
}

/// API-Key des angemeldeten Nutzers – aus der laufenden Sitzung. Abgemeldet
/// heisst: kein Key, die Relay wartet. (Vorher las sie config.json, in der
/// Passwort und Key im Klartext standen.)
pub fn read_api_key(app: &tauri::AppHandle) -> Option<String> {
    let uid = crate::database::session_user_id().ok();
    let bound = app.state::<super::McpState>().config.lock().ok()?.remote_active_for(uid.as_deref());
    if !bound { return None; }
    crate::secret_store::session_api_key().filter(|k| k.starts_with("ft_"))
}

#[cfg(test)]
pub fn api_key_from_config(raw: &str) -> Option<String> {
    let v: Value = serde_json::from_str(raw).ok()?;
    v.get("api_key").and_then(|k| k.as_str()).filter(|k| k.starts_with("ft_")).map(str::to_string)
}

pub fn device_name() -> String {
    #[cfg(target_os = "macos")]
    {
        use crate::command_ext::NoWindow;
        if let Ok(o) = std::process::Command::new("scutil").args(["--get", "ComputerName"]).no_window().output() {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if o.status.success() && !s.is_empty() {
                return s;
            }
        }
    }
    for var in ["COMPUTERNAME", "HOSTNAME"] {
        if let Ok(v) = std::env::var(var) {
            if !v.trim().is_empty() {
                return v.trim().to_string();
            }
        }
    }
    std::fs::read_to_string("/etc/hostname").ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
        .unwrap_or_else(|| "FrameTrain Desktop".to_string())
}

/// Jobs aus der /poll-Antwort.
pub fn parse_jobs(v: &Value) -> Vec<(String, String, Value, ClientInfo)> {
    v.get("jobs").and_then(|j| j.as_array()).into_iter().flatten().filter_map(|j| {
        let id = j.get("id")?.as_str()?.to_string();
        let tool = j.get("tool")?.as_str()?.to_string();
        let args = j.get("args").cloned().filter(|a| a.is_object()).unwrap_or_else(|| json!({}));
        let name = j.get("client").and_then(|c| c.get("name")).and_then(|n| n.as_str()).unwrap_or("Remote").to_string();
        Some((id, tool, args, ClientInfo { name, version: None }))
    }).collect()
}

fn http() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(POLL_TIMEOUT)
        .user_agent(format!("FrameTrain-Desktop/{}", env!("CARGO_PKG_VERSION")))
        .build()
        .unwrap_or_default()
}

async fn post(client: &reqwest::Client, path: &str, body: Value) -> Result<Value, String> {
    let resp = client.post(url(path)).json(&body).send().await.map_err(|e| format!("Netzwerk: {}", e))?;
    let status = resp.status();
    let v: Value = resp.json().await.unwrap_or(Value::Null);
    if !status.is_success() {
        let msg = v.get("error").and_then(|e| e.as_str()).map(str::to_string).unwrap_or_else(|| status.to_string());
        return Err(format!("{} ({})", msg, status.as_u16()));
    }
    Ok(v)
}

fn set_status(app: &tauri::AppHandle, f: impl FnOnce(&mut RelayStatus)) {
    let st = app.state::<super::McpState>();
    f(&mut st.relay.status.lock().unwrap());
}

/// Startet oder stoppt den Client passend zur Einstellung.
pub fn apply(app: &tauri::AppHandle, enabled: bool) {
    let st = app.state::<super::McpState>();
    let gen = st.relay.generation.fetch_add(1, Ordering::SeqCst) + 1;
    {
        let mut s = st.relay.status.lock().unwrap();
        *s = RelayStatus { running: enabled, ..Default::default() };
    }
    if !enabled {
        return;
    }
    let app1 = app.clone();
    tauri::async_runtime::spawn(async move { heartbeat_loop(app1, gen).await });
    let app2 = app.clone();
    tauri::async_runtime::spawn(async move { poll_loop(app2, gen).await });
}

fn alive(app: &tauri::AppHandle, gen: u64) -> bool {
    app.state::<super::McpState>().relay.generation.load(Ordering::SeqCst) == gen
}

async fn heartbeat_loop(app: tauri::AppHandle, gen: u64) {
    let client = http();
    let st = app.state::<super::McpState>();
    let bridge = st.bridge.clone();
    let device_id = st.config.lock().unwrap().device_id.clone();
    let device_name = tauri::async_runtime::spawn_blocking(device_name).await.unwrap_or_else(|_| "FrameTrain Desktop".into());
    let version = app.package_info().version.to_string();
    let mut backoff = 0u64;
    while alive(&app, gen) {
        let wait = match read_api_key(&app) {
            None => {
                set_status(&app, |s| { s.connected = false; s.last_error = Some("Nicht angemeldet oder Fernzugriff gehoert zu einem anderen Account".into()); });
                BACKOFF_MAX
            }
            Some(key) => {
                let hash = bridge.current_hash();
                let res = post(&client, "heartbeat", json!({
                    "apiKey": key, "deviceId": device_id, "deviceName": device_name,
                    "appVersion": version, "manifestHash": hash,
                })).await;
                match res {
                    Ok(v) => {
                        backoff = 0;
                        let mut err = None;
                        if v.get("wantManifest").and_then(|w| w.as_bool()).unwrap_or(false) && bridge.has_manifest() {
                            let tools = bridge.tools();
                            if let Err(e) = post(&client, "manifest", json!({
                                "apiKey": key, "deviceId": device_id, "manifestHash": hash, "tools": tools,
                            })).await {
                                err = Some(format!("Manifest: {}", e));
                            }
                        }
                        set_status(&app, |s| {
                            s.connected = true;
                            s.last_heartbeat = Some(chrono::Utc::now().to_rfc3339());
                            s.last_error = err;
                        });
                        HEARTBEAT_EVERY.as_secs()
                    }
                    Err(e) => {
                        backoff = next_backoff(backoff);
                        set_status(&app, |s| { s.connected = false; s.last_error = Some(e); });
                        backoff.min(HEARTBEAT_EVERY.as_secs())
                    }
                }
            }
        };
        sleep_while_alive(&app, gen, wait).await;
    }
}

async fn poll_loop(app: tauri::AppHandle, gen: u64) {
    let client = http();
    let device_id = app.state::<super::McpState>().config.lock().unwrap().device_id.clone();
    let mut backoff = 0u64;
    while alive(&app, gen) {
        let Some(key) = read_api_key(&app) else {
            sleep_while_alive(&app, gen, BACKOFF_MAX).await;
            continue;
        };
        match post(&client, "poll", json!({ "apiKey": key, "deviceId": device_id })).await {
            Ok(v) => {
                backoff = 0;
                if !alive(&app, gen) { break; }
                for (job_id, tool, args, client_info) in parse_jobs(&v) {
                    let app = app.clone();
                    let key = key.clone();
                    let device_id = device_id.clone();
                    let http = client.clone();
                    tauri::async_runtime::spawn(async move {
                        let app2 = app.clone();
                        let result = tauri::async_runtime::spawn_blocking(move || {
                            bridge::call_via_app(&app2, &tool, args, Source::Remote, client_info)
                        }).await.unwrap_or_else(|e| bridge::error_result(&format!("Interner Fehler: {}", e)));
                        let r = post(&http, "result", json!({
                            "apiKey": key, "deviceId": device_id, "jobId": job_id, "result": result,
                        })).await;
                        set_status(&app, |s| {
                            s.jobs_handled += 1;
                            if let Err(e) = r { s.last_error = Some(format!("Ergebnis: {}", e)); }
                        });
                    });
                }
            }
            Err(e) => {
                backoff = next_backoff(backoff);
                set_status(&app, |s| s.last_error = Some(e));
                sleep_while_alive(&app, gen, backoff).await;
            }
        }
    }
}

async fn sleep_while_alive(app: &tauri::AppHandle, gen: u64, secs: u64) {
    for _ in 0..secs {
        if !alive(app, gen) { return; }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

/// /status bzw. /disconnect fuer die Einstellungen.
pub async fn remote_request(app: &tauri::AppHandle, path: &str, extra: Value) -> Result<Value, String> {
    let key = read_api_key(app).ok_or("Nicht angemeldet")?;
    let mut body = json!({ "apiKey": key });
    if let (Some(b), Some(e)) = (body.as_object_mut(), extra.as_object()) {
        for (k, v) in e { b.insert(k.clone(), v.clone()); }
    }
    let client = reqwest::Client::builder().timeout(Duration::from_secs(15)).build().map_err(|e| e.to_string())?;
    post(&client, path, body).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_doubles_and_caps() {
        let mut b = 0;
        let mut seen = vec![];
        for _ in 0..8 { b = next_backoff(b); seen.push(b); }
        assert_eq!(seen, vec![2, 4, 8, 16, 32, 60, 60, 60]);
    }

    #[test]
    fn api_key_parsing() {
        assert_eq!(api_key_from_config(r#"{"api_key":"ft_abcdefghijklmnopqrstuvwx","password":"x"}"#).as_deref(), Some("ft_abcdefghijklmnopqrstuvwx"));
        assert_eq!(api_key_from_config(r#"{"api_key":"nope"}"#), None);
        assert_eq!(api_key_from_config(r#"{"theme":"dark"}"#), None);
        assert_eq!(api_key_from_config("kaputt"), None);
    }

    #[test]
    fn jobs_parsing() {
        let v = json!({ "jobs": [
            { "id": "j1", "tool": "models_list", "args": { "a": 1 }, "client": { "name": "ChatGPT" }, "createdAt": "x" },
            { "id": "j2", "tool": "datasets_list", "args": null },
            { "tool": "ohne_id" }
        ]});
        let jobs = parse_jobs(&v);
        assert_eq!(jobs.len(), 2);
        assert_eq!(jobs[0].0, "j1");
        assert_eq!(jobs[0].3.name, "ChatGPT");
        assert_eq!(jobs[1].2, json!({}));
        assert_eq!(jobs[1].3.name, "Remote");
        assert!(parse_jobs(&json!({})).is_empty());
    }

    #[test]
    fn urls() {
        assert!(url("poll").ends_with("/desktop/mcp/poll"));
    }
}
