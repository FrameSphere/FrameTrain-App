// mcp/bridge.rs – Rust <-> Webview (Vertrag Abschnitt 3)
//
// Die Webview meldet ihr Werkzeug-Manifest per `mcp_register_manifest`. Jeder
// Aufruf (lokal oder ueber das Relay) wird als Event `mcp://call` an das
// Hauptfenster geschickt; die Antwort kommt per `mcp_tool_result` zurueck.
// Der Kern (Bridge) haengt nicht an Tauri, damit er sich testen laesst.

use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

pub const CALL_TIMEOUT: Duration = Duration::from_secs(60);
pub const MANIFEST_WAIT: Duration = Duration::from_secs(15);
pub const TIMEOUT_TEXT: &str = "FrameTrain did not respond in time";
pub const PING_TOOL: &str = "system_ping";

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    Local,
    Remote,
}

#[derive(Debug, Clone, Serialize, Default, PartialEq)]
pub struct ClientInfo {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

/// Ein Eintrag fuer "letzte Aufrufe" in den Einstellungen.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CallLogEntry {
    pub tool: String,
    pub client_name: String,
    pub source: Source,
    pub is_error: bool,
    pub at: String,
    pub duration_ms: u64,
}

pub fn error_result(text: &str) -> Value {
    json!({ "content": [{ "type": "text", "text": text }], "isError": true })
}

pub fn ping_tool_def() -> Value {
    json!({
        "name": PING_TOOL,
        "title": "FrameTrain ping",
        "description": "Checks that the FrameTrain desktop app is reachable. Returns the app version and whether the tool registry is loaded yet. Call this first if other FrameTrain tools are missing.",
        "inputSchema": { "type": "object", "properties": {}, "additionalProperties": false },
        "annotations": { "readOnlyHint": true, "destructiveHint": false, "openWorldHint": false },
        "_meta": { "frametrain/domain": "system", "frametrain/approval": "none" }
    })
}

pub fn ping_result(version: &str, ready: bool, tool_count: usize) -> Value {
    let text = if ready {
        format!("FrameTrain {} is running with {} tools available.", version, tool_count)
    } else {
        format!("FrameTrain {} is running, but the app is still starting. Try tools/list again in a few seconds.", version)
    };
    json!({
        "content": [{ "type": "text", "text": text }],
        "structuredContent": { "app": "FrameTrain", "version": version, "ready": ready, "tools": tool_count }
    })
}

/// Stabiler Hash des Manifests (Relay: manifestHash).
pub fn manifest_hash(tools: &[Value]) -> String {
    let mut h = Sha256::new();
    h.update(serde_json::to_vec(tools).unwrap_or_default());
    h.finalize().iter().take(16).map(|b| format!("{:02x}", b)).collect()
}

pub struct Bridge {
    manifest: Mutex<Option<Vec<Value>>>,
    manifest_cv: Condvar,
    pending: Mutex<HashMap<String, mpsc::Sender<Value>>>,
    log: Mutex<Vec<CallLogEntry>>,
    started: Instant,
}

impl Default for Bridge {
    fn default() -> Self {
        Self {
            manifest: Mutex::new(None),
            manifest_cv: Condvar::new(),
            pending: Mutex::new(HashMap::new()),
            log: Mutex::new(Vec::new()),
            started: Instant::now(),
        }
    }
}

const LOG_MAX: usize = 50;

impl Bridge {
    pub fn set_manifest(&self, tools: Vec<Value>) {
        // Nur Eintraege mit Namen; doppelte Namen: der erste gewinnt.
        let mut seen = std::collections::HashSet::new();
        let clean: Vec<Value> = tools
            .into_iter()
            .filter(|t| t.get("name").and_then(|n| n.as_str()).map(|n| !n.is_empty() && seen.insert(n.to_string())).unwrap_or(false))
            .collect();
        *self.manifest.lock().unwrap() = Some(clean);
        self.manifest_cv.notify_all();
    }

    pub fn has_manifest(&self) -> bool {
        self.manifest.lock().unwrap().is_some()
    }

    /// Wartet hoechstens `max` auf ein Manifest. true = vorhanden.
    pub fn wait_manifest(&self, max: Duration) -> bool {
        let guard = self.manifest.lock().unwrap();
        let (guard, _) = self.manifest_cv.wait_timeout_while(guard, max, |m| m.is_none()).unwrap();
        guard.is_some()
    }

    /// Kurz nach dem Start (z.B. von der Stdio-Bruecke geweckt) lohnt sich das Warten.
    pub fn recently_started(&self) -> bool {
        self.started.elapsed() < Duration::from_secs(60)
    }

    /// Werkzeugliste: Manifest + system_ping, falls das Manifest es nicht selbst hat.
    pub fn tools(&self) -> Vec<Value> {
        let mut list = self.manifest.lock().unwrap().clone().unwrap_or_default();
        if !list.iter().any(|t| t.get("name").and_then(|n| n.as_str()) == Some(PING_TOOL)) {
            list.insert(0, ping_tool_def());
        }
        list
    }

    pub fn current_hash(&self) -> String {
        manifest_hash(&self.manifest.lock().unwrap().clone().unwrap_or_default())
    }

    fn manifest_has(&self, tool: &str) -> bool {
        self.manifest.lock().unwrap().as_ref()
            .map(|m| m.iter().any(|t| t.get("name").and_then(|n| n.as_str()) == Some(tool)))
            .unwrap_or(false)
    }

    /// Antwort der Webview. false = unbekannte/abgelaufene callId.
    pub fn resolve(&self, call_id: &str, result: Value) -> bool {
        match self.pending.lock().unwrap().remove(call_id) {
            Some(tx) => tx.send(result).is_ok(),
            None => false,
        }
    }

    pub fn pending_count(&self) -> usize {
        self.pending.lock().unwrap().len()
    }

    pub fn recent_calls(&self) -> Vec<CallLogEntry> {
        self.log.lock().unwrap().iter().rev().cloned().collect()
    }

    fn record(&self, tool: &str, client: &ClientInfo, source: &Source, result: &Value, started: Instant) {
        let mut log = self.log.lock().unwrap();
        log.push(CallLogEntry {
            tool: tool.to_string(),
            client_name: client.name.clone(),
            source: source.clone(),
            is_error: result.get("isError").and_then(|v| v.as_bool()).unwrap_or(false),
            at: chrono::Utc::now().to_rfc3339(),
            duration_ms: started.elapsed().as_millis() as u64,
        });
        if log.len() > LOG_MAX {
            let cut = log.len() - LOG_MAX;
            log.drain(..cut);
        }
    }

    /// Fuehrt einen Aufruf aus (blockiert bis Antwort oder Timeout).
    /// `emit` schickt das `mcp://call`-Event an die Webview, `on_missing_manifest`
    /// darf das Hauptfenster zeigen.
    pub fn call(
        &self,
        tool: &str,
        args: Value,
        source: Source,
        client: ClientInfo,
        version: &str,
        timeout: Duration,
        on_missing_manifest: impl FnOnce(),
        emit: impl FnOnce(&Value) -> Result<(), String>,
    ) -> Value {
        let started = Instant::now();
        if !self.has_manifest() {
            on_missing_manifest();
            if tool != PING_TOOL {
                self.wait_manifest(MANIFEST_WAIT.min(timeout));
            }
        }
        // system_ping beantwortet Rust selbst, solange die Webview es nicht anbietet.
        if tool == PING_TOOL && !self.manifest_has(PING_TOOL) {
            let n = self.manifest.lock().unwrap().as_ref().map(|m| m.len()).unwrap_or(0);
            let r = ping_result(version, self.has_manifest(), n);
            self.record(tool, &client, &source, &r, started);
            return r;
        }
        let result = if !self.has_manifest() {
            error_result("FrameTrain is still starting. Please try again in a few seconds.")
        } else if !self.manifest_has(tool) {
            error_result(&format!("Unbekanntes Werkzeug: {}", tool))
        } else {
            let call_id = uuid::Uuid::new_v4().to_string();
            let (tx, rx) = mpsc::channel();
            self.pending.lock().unwrap().insert(call_id.clone(), tx);
            let payload = json!({
                "callId": call_id,
                "tool": tool,
                "args": if args.is_object() { args } else { json!({}) },
                "source": source,
                "client": client,
            });
            match emit(&payload) {
                Err(e) => {
                    self.pending.lock().unwrap().remove(&call_id);
                    error_result(&format!("FrameTrain konnte den Aufruf nicht weitergeben: {}", e))
                }
                Ok(()) => match rx.recv_timeout(timeout) {
                    Ok(r) => normalize_result(r),
                    Err(_) => {
                        self.pending.lock().unwrap().remove(&call_id);
                        error_result(TIMEOUT_TEXT)
                    }
                },
            }
        };
        self.record(tool, &client, &source, &result, started);
        result
    }
}

/// Stellt sicher, dass ein CallToolResult mindestens `content` hat.
pub fn normalize_result(r: Value) -> Value {
    match r {
        Value::Object(mut m) => {
            if !m.get("content").map(|c| c.is_array()).unwrap_or(false) {
                let text = m.get("structuredContent").map(|s| s.to_string()).unwrap_or_default();
                m.insert("content".into(), json!([{ "type": "text", "text": text }]));
            }
            Value::Object(m)
        }
        other => json!({ "content": [{ "type": "text", "text": other.to_string() }] }),
    }
}

// ============ Tauri-Anbindung ============

pub const CALL_EVENT: &str = "mcp://call";

/// Aufruf aus Server/Relay: Event ans Hauptfenster, ggf. Fenster zeigen.
pub fn call_via_app(app: &tauri::AppHandle, tool: &str, args: Value, source: Source, client: ClientInfo) -> Value {
    use tauri::{Emitter, Manager};
    let bridge = app.state::<super::McpState>().bridge.clone();
    let version = app.package_info().version.to_string();
    let app2 = app.clone();
    bridge.call(
        tool,
        args,
        source,
        client,
        &version,
        CALL_TIMEOUT,
        move || {
            // Kein Manifest: Hauptfenster zeigen (die Webview registriert beim Laden).
            if app2.get_webview_window("main").map(|w| !w.is_visible().unwrap_or(false)).unwrap_or(true) {
                crate::hosting_desktop::show_main(&app2, None);
            }
        },
        |payload| app.emit_to("main", CALL_EVENT, payload.clone()).map_err(|e| e.to_string()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn tool(name: &str) -> Value {
        json!({ "name": name, "inputSchema": { "type": "object" } })
    }

    fn client() -> ClientInfo {
        ClientInfo { name: "test-client".into(), version: Some("1".into()) }
    }

    #[test]
    fn ping_without_manifest() {
        let b = Bridge::default();
        let r = b.call(PING_TOOL, json!({}), Source::Local, client(), "9.9.9", Duration::from_millis(50), || {}, |_| panic!("kein Event fuer ping"));
        assert!(r.get("isError").is_none());
        assert_eq!(r["structuredContent"]["ready"], false);
        assert!(r["content"][0]["text"].as_str().unwrap().contains("still starting"));
        assert_eq!(b.tools()[0]["name"], PING_TOOL);
    }

    #[test]
    fn manifest_dedup_and_ping_added() {
        let b = Bridge::default();
        b.set_manifest(vec![tool("models_list"), tool("models_list"), json!({ "title": "ohne Name" })]);
        let t = b.tools();
        assert_eq!(t.len(), 2);
        assert_eq!(t[0]["name"], PING_TOOL);
        assert_eq!(t[1]["name"], "models_list");
        // Webview liefert selbst system_ping -> kein doppelter Eintrag
        b.set_manifest(vec![tool(PING_TOOL)]);
        assert_eq!(b.tools().len(), 1);
    }

    #[test]
    fn call_roundtrip_through_event() {
        let b = Arc::new(Bridge::default());
        b.set_manifest(vec![tool("datasets_list")]);
        let b2 = b.clone();
        let r = b.call("datasets_list", json!({"a": 1}), Source::Remote, client(), "1", Duration::from_secs(5), || {}, move |p| {
            assert_eq!(p["tool"], "datasets_list");
            assert_eq!(p["source"], "remote");
            assert_eq!(p["client"]["name"], "test-client");
            assert_eq!(p["args"]["a"], 1);
            let id = p["callId"].as_str().unwrap().to_string();
            let b3 = b2.clone();
            std::thread::spawn(move || {
                assert!(b3.resolve(&id, json!({ "content": [{ "type": "text", "text": "ok" }], "structuredContent": { "n": 3 } })));
            });
            Ok(())
        });
        assert_eq!(r["structuredContent"]["n"], 3);
        assert_eq!(b.pending_count(), 0);
        let log = b.recent_calls();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].tool, "datasets_list");
        assert!(!log[0].is_error);
    }

    #[test]
    fn call_timeout() {
        let b = Bridge::default();
        b.set_manifest(vec![tool("training_start")]);
        let r = b.call("training_start", json!({}), Source::Local, client(), "1", Duration::from_millis(30), || {}, |_| Ok(()));
        assert_eq!(r["isError"], true);
        assert_eq!(r["content"][0]["text"], TIMEOUT_TEXT);
        assert_eq!(b.pending_count(), 0);
        assert!(!b.resolve("gibt-es-nicht", json!({})));
    }

    #[test]
    fn unknown_tool_and_emit_error() {
        let b = Bridge::default();
        b.set_manifest(vec![tool("a_b")]);
        let r = b.call("x_y", json!({}), Source::Local, client(), "1", Duration::from_millis(30), || {}, |_| Ok(()));
        assert_eq!(r["isError"], true);
        let r = b.call("a_b", json!({}), Source::Local, client(), "1", Duration::from_millis(30), || {}, |_| Err("kein Fenster".into()));
        assert_eq!(r["isError"], true);
        assert!(r["content"][0]["text"].as_str().unwrap().contains("kein Fenster"));
    }

    #[test]
    fn missing_manifest_waits_and_calls_hook() {
        let b = Arc::new(Bridge::default());
        let b2 = b.clone();
        let hooked = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let h2 = hooked.clone();
        // Manifest kommt waehrend des Wartens
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            b2.set_manifest(vec![tool("a_b")]);
        });
        let b3 = b.clone();
        let r = b.call("a_b", json!({}), Source::Local, client(), "1", Duration::from_secs(5), move || h2.store(true, std::sync::atomic::Ordering::SeqCst), move |p| {
            let id = p["callId"].as_str().unwrap().to_string();
            let b4 = b3.clone();
            std::thread::spawn(move || { b4.resolve(&id, json!({ "structuredContent": { "x": 1 } })); });
            Ok(())
        });
        assert!(hooked.load(std::sync::atomic::Ordering::SeqCst));
        // normalize_result ergaenzt content
        assert!(r["content"].is_array());
        assert_eq!(r["structuredContent"]["x"], 1);
    }

    #[test]
    fn hash_changes_with_manifest() {
        let a = manifest_hash(&[tool("a_b")]);
        let b = manifest_hash(&[tool("a_c")]);
        assert_ne!(a, b);
        assert_eq!(a.len(), 32);
    }
}
