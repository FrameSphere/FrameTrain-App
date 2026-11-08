// mcp/server.rs – lokaler MCP-Endpunkt (Vertrag Abschnitt 4)
//
// Streamable HTTP (MCP 2025-06-18) mit reinen JSON-Antworten:
//   POST /mcp     JSON-RPC (initialize, notifications/initialized, ping, tools/list, tools/call)
//   GET  /mcp     405 (kein SSE-Strom)
//   GET  /health  ohne Token, fuer die Stdio-Bruecke ("laeuft die App?")
// Nur 127.0.0.1, Host- und Origin-Pruefung gegen DNS-Rebinding,
// Authorization: Bearer ftmcp_…

use super::bridge::ClientInfo;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::Read;
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub const SUPPORTED_VERSIONS: [&str; 3] = ["2025-06-18", "2025-03-26", "2024-11-05"];
pub const MAX_BODY: usize = 16 * 1024 * 1024;
const MAX_SESSIONS: usize = 200;

// JSON-RPC-Fehlercodes
pub const PARSE_ERROR: i64 = -32700;
pub const INVALID_REQUEST: i64 = -32600;
pub const METHOD_NOT_FOUND: i64 = -32601;
pub const INVALID_PARAMS: i64 = -32602;

/// Was der Server von der App braucht – im Test durch eine Attrappe ersetzt.
pub trait Backend: Send + Sync {
    fn version(&self) -> String;
    fn tools(&self) -> Vec<Value>;
    fn call(&self, name: &str, args: Value, client: &ClientInfo) -> Value;
}

fn rpc_error(id: Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

fn rpc_result(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

pub fn negotiate_version(requested: Option<&str>) -> &'static str {
    requested
        .and_then(|r| SUPPORTED_VERSIONS.iter().find(|v| **v == r).copied())
        .unwrap_or(SUPPORTED_VERSIONS[0])
}

pub fn initialize_result(version: &str, requested: Option<&str>) -> Value {
    json!({
        "protocolVersion": negotiate_version(requested),
        "capabilities": { "tools": { "listChanged": false } },
        "serverInfo": { "name": "frametrain", "title": "FrameTrain", "version": version },
        "instructions": "FrameTrain trains AI models locally on this computer. The tools manage models and datasets, build datasets, start and monitor training, and test or host the results. Long-running work returns a job_id; the matching *_status tool reports its progress. Keep polling (system_job_status with wait_seconds 45) until the job is done or failed, even if that takes several minutes, and never report a running job as a failure. Good small starting sources: for people and vehicles (and 78 other everyday COCO classes) the 128 labeled images at https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip via datasets_import_url; for larger samples a Hugging Face dataset via datasets_download_hf with max_rows, which only downloads the parquet files it needs. Some actions ask the user for approval inside the FrameTrain app. Datasets can come from Hugging Face (datasets_hf_files, datasets_download_hf), from https URLs including partial reads of large ZIP archives (datasets_import_url), or from local files. datasets_check and datasets_view show what a dataset contains. The Dataset Builder (studio_*) merges several datasets into one project with a common class list (studio_import with source dataset and class_map), shows samples as pictures (studio_view_samples), takes corrections (studio_set_annotation) and exports a training dataset (studio_export). This works for images with boxes, image, audio and video classification, audio with transcripts, and text classification or text pairs."
    })
}

/// Bearbeitet eine einzelne JSON-RPC-Nachricht. None = keine Antwort (Notification/Response).
pub fn handle_message(msg: &Value, backend: &dyn Backend, client: &ClientInfo) -> Option<Value> {
    let Some(obj) = msg.as_object() else {
        return Some(rpc_error(Value::Null, INVALID_REQUEST, "Invalid Request"));
    };
    let id = obj.get("id").cloned();
    let method = obj.get("method").and_then(|m| m.as_str());
    if method.is_none() && (obj.contains_key("result") || obj.contains_key("error")) {
        return None; // Antwort des Clients auf eine Server-Anfrage – gibt es bei uns nicht
    }
    if obj.get("jsonrpc").and_then(|v| v.as_str()) != Some("2.0") || method.is_none() {
        return Some(rpc_error(id.unwrap_or(Value::Null), INVALID_REQUEST, "Invalid Request"));
    }
    let method = method.unwrap();
    let Some(id) = id else {
        return None; // Notification (notifications/initialized, notifications/cancelled, …)
    };
    let params = obj.get("params").cloned().unwrap_or(Value::Null);
    Some(match method {
        "initialize" => rpc_result(id, initialize_result(&backend.version(), params.get("protocolVersion").and_then(|v| v.as_str()))),
        "ping" => rpc_result(id, json!({})),
        "tools/list" => rpc_result(id, json!({ "tools": backend.tools() })),
        "tools/call" => match params.get("name").and_then(|n| n.as_str()) {
            None => rpc_error(id, INVALID_PARAMS, "Missing tool name"),
            Some(name) => {
                let args = params.get("arguments").cloned().unwrap_or_else(|| json!({}));
                if !args.is_object() {
                    rpc_error(id, INVALID_PARAMS, "arguments must be an object")
                } else {
                    rpc_result(id, backend.call(name, args, client))
                }
            }
        },
        // Leere Listen statt Fehler – manche Clients fragen das unaufgefordert ab.
        "resources/list" => rpc_result(id, json!({ "resources": [] })),
        "resources/templates/list" => rpc_result(id, json!({ "resourceTemplates": [] })),
        "prompts/list" => rpc_result(id, json!({ "prompts": [] })),
        _ => rpc_error(id, METHOD_NOT_FOUND, &format!("Method not found: {}", method)),
    })
}

pub fn client_from_initialize(msg: &Value) -> Option<ClientInfo> {
    if msg.get("method").and_then(|m| m.as_str()) != Some("initialize") {
        return None;
    }
    let ci = msg.get("params")?.get("clientInfo")?;
    let name = ci.get("name").and_then(|n| n.as_str()).filter(|n| !n.trim().is_empty())?;
    Some(ClientInfo {
        name: name.chars().take(80).collect(),
        version: ci.get("version").and_then(|v| v.as_str()).map(|v| v.chars().take(40).collect()),
    })
}

// ============ HTTP-Schicht (ohne Netzwerk testbar) ============

/// Nur lokale Namen im Host-Header — Schutz vor DNS-Rebinding.
pub fn host_allowed(host: Option<&str>, port: u16) -> bool {
    crate::hosting_api::host_allowed(host, port)
}

/// Browser schicken Origin mit; nur lokale Seiten duerfen (der Token wird trotzdem geprueft).
pub fn origin_allowed(origin: Option<&str>) -> bool {
    let Some(o) = origin else { return true };
    let o = o.trim().to_lowercase();
    let rest = o.strip_prefix("http://").or_else(|| o.strip_prefix("https://"));
    let Some(rest) = rest else { return false };
    let host = rest.split('/').next().unwrap_or("");
    let host = if host.starts_with('[') {
        host.split(']').next().map(|h| format!("{}]", h)).unwrap_or_default()
    } else {
        host.split(':').next().unwrap_or("").to_string()
    };
    matches!(host.as_str(), "localhost" | "127.0.0.1" | "[::1]")
}

pub fn token_ok(auth: Option<&str>, token: &str) -> bool {
    crate::hosting_api::token_ok(auth, None, token)
}

pub struct HttpReq<'a> {
    pub method: &'a str,
    pub path: &'a str,
    pub host: Option<&'a str>,
    pub origin: Option<&'a str>,
    pub auth: Option<&'a str>,
    pub session: Option<&'a str>,
    pub body: &'a [u8],
}

#[derive(Debug)]
pub struct HttpResp {
    pub status: u16,
    pub body: Option<Value>,
    pub session: Option<String>,
    pub allow: Option<&'static str>,
}

impl HttpResp {
    fn new(status: u16, body: Option<Value>) -> Self {
        Self { status, body, session: None, allow: None }
    }
}

/// Sitzungen nur, um den Client-Namen (Freigaben je Client) ueber Anfragen hinweg zu kennen.
#[derive(Default)]
pub struct Sessions {
    map: Mutex<HashMap<String, (ClientInfo, Instant)>>,
}

impl Sessions {
    pub fn create(&self, client: ClientInfo) -> String {
        let id = uuid::Uuid::new_v4().simple().to_string();
        let mut m = self.map.lock().unwrap();
        if m.len() >= MAX_SESSIONS {
            if let Some(oldest) = m.iter().min_by_key(|(_, (_, t))| *t).map(|(k, _)| k.clone()) {
                m.remove(&oldest);
            }
        }
        m.insert(id.clone(), (client, Instant::now()));
        id
    }

    pub fn client(&self, id: Option<&str>) -> Option<ClientInfo> {
        let mut m = self.map.lock().unwrap();
        let e = m.get_mut(id?)?;
        e.1 = Instant::now();
        Some(e.0.clone())
    }

    pub fn remove(&self, id: &str) -> bool {
        self.map.lock().unwrap().remove(id).is_some()
    }

    pub fn clients(&self) -> Vec<ClientInfo> {
        let mut seen = std::collections::HashSet::new();
        self.map.lock().unwrap().values()
            .filter(|(c, _)| seen.insert(c.name.clone()))
            .map(|(c, _)| c.clone())
            .collect()
    }
}

pub fn handle_http(req: &HttpReq, port: u16, token: &str, sessions: &Sessions, backend: &dyn Backend) -> HttpResp {
    if !host_allowed(req.host, port) {
        return HttpResp::new(403, Some(json!({ "error": "Host not allowed" })));
    }
    if !origin_allowed(req.origin) {
        return HttpResp::new(403, Some(json!({ "error": "Origin not allowed" })));
    }
    if req.path == "/health" && req.method == "GET" {
        return HttpResp::new(200, Some(json!({ "ok": true, "app": "FrameTrain", "mcp": true, "version": backend.version() })));
    }
    if req.path != "/mcp" {
        return HttpResp::new(404, Some(json!({ "error": "Not found. MCP endpoint: POST /mcp" })));
    }
    if !token_ok(req.auth, token) {
        return HttpResp::new(401, Some(json!({ "error": "Missing or wrong token (Authorization: Bearer ftmcp_…)" })));
    }
    match req.method {
        "POST" => {}
        "DELETE" => {
            if let Some(s) = req.session { sessions.remove(s); }
            return HttpResp::new(200, None);
        }
        _ => {
            let mut r = HttpResp::new(405, None);
            r.allow = Some("POST, DELETE");
            return r;
        }
    }
    let msg: Value = match serde_json::from_slice(req.body) {
        Ok(v) => v,
        Err(_) => return HttpResp::new(400, Some(rpc_error(Value::Null, PARSE_ERROR, "Parse error"))),
    };
    if msg.is_array() {
        return HttpResp::new(400, Some(rpc_error(Value::Null, INVALID_REQUEST, "Batching is not supported")));
    }
    let mut new_session = None;
    let client = match client_from_initialize(&msg) {
        Some(c) => {
            new_session = Some(sessions.create(c.clone()));
            c
        }
        // Unbekannte Sitzung (z.B. nach App-Neustart): grosszuegig weitermachen statt 404,
        // sonst haengt ein Client, der nicht neu initialisiert.
        None => sessions.client(req.session).unwrap_or_else(|| ClientInfo { name: "unknown".into(), version: None }),
    };
    match handle_message(&msg, backend, &client) {
        None => HttpResp::new(202, None),
        Some(resp) => {
            let mut r = HttpResp::new(200, Some(resp));
            r.session = new_session;
            r
        }
    }
}

// ============ tiny_http-Server ============

struct Running {
    port: u16,
    server: Arc<tiny_http::Server>,
}

#[derive(Default)]
pub struct ServerState {
    running: Mutex<Option<Running>>,
    error: Mutex<Option<String>>,
    pub sessions: Sessions,
}

impl ServerState {
    pub fn status(&self) -> Value {
        let running = self.running.lock().unwrap();
        json!({
            "running": running.is_some(),
            "url": running.as_ref().map(|r| format!("http://127.0.0.1:{}/mcp", r.port)),
            "error": self.error.lock().unwrap().clone(),
        })
    }
}

fn header(req: &tiny_http::Request, name: &str) -> Option<String> {
    req.headers().iter()
        .find(|h| h.field.as_str().as_str().eq_ignore_ascii_case(name))
        .map(|h| h.value.as_str().to_string())
}

/// Startet, stoppt oder verlegt den Server. `ctx` liefert Token + Backend zur Laufzeit,
/// damit ein neuer Token sofort gilt.
pub fn apply<F>(state: Arc<ServerState>, enabled: bool, port: u16, ctx: F)
where
    F: Fn() -> (String, Arc<dyn Backend>) + Send + Sync + 'static,
{
    let mut running = state.running.lock().unwrap();
    if let Some(r) = running.as_ref() {
        if enabled && r.port == port {
            return;
        }
        r.server.unblock();
        *running = None;
        println!("[MCP] Server gestoppt");
    }
    *state.error.lock().unwrap() = None;
    if !enabled {
        return;
    }
    match tiny_http::Server::http(("127.0.0.1", port)) {
        Ok(server) => {
            let server = Arc::new(server);
            *running = Some(Running { port, server: server.clone() });
            let ctx = Arc::new(ctx);
            let state2 = state.clone();
            std::thread::spawn(move || {
                for req in server.incoming_requests() {
                    let ctx = ctx.clone();
                    let state = state2.clone();
                    std::thread::spawn(move || serve(req, port, &state, &*ctx));
                }
            });
            println!("[MCP] laeuft auf http://127.0.0.1:{}/mcp", port);
        }
        Err(e) => {
            let msg = format!("Port {} ist belegt oder gesperrt: {}", port, e);
            eprintln!("[MCP] {}", msg);
            *state.error.lock().unwrap() = Some(msg);
        }
    }
}

fn serve<F>(mut req: tiny_http::Request, port: u16, state: &ServerState, ctx: &F)
where
    F: Fn() -> (String, Arc<dyn Backend>),
{
    let method = req.method().as_str().to_uppercase();
    let path = req.url().split('?').next().unwrap_or("").trim_end_matches('/').to_string();
    let mut buf = Vec::new();
    if method == "POST" && (req.as_reader().take(MAX_BODY as u64 + 1).read_to_end(&mut buf).is_err() || buf.len() > MAX_BODY) {
        let _ = req.respond(tiny_http::Response::from_string("Request too large").with_status_code(413));
        return;
    }
    let (token, backend) = ctx();
    let host = header(&req, "Host");
    let origin = header(&req, "Origin");
    let auth = header(&req, "Authorization");
    let session = header(&req, "Mcp-Session-Id");
    let r = handle_http(
        &HttpReq {
            method: &method,
            path: &path,
            host: host.as_deref(),
            origin: origin.as_deref(),
            auth: auth.as_deref(),
            session: session.as_deref(),
            body: &buf,
        },
        port,
        &token,
        &state.sessions,
        &*backend,
    );
    let data = r.body.map(|b| b.to_string().into_bytes()).unwrap_or_default();
    let mut resp = tiny_http::Response::from_data(data).with_status_code(r.status);
    if r.status != 202 {
        resp = resp.with_header(tiny_http::Header::from_bytes("Content-Type", "application/json; charset=utf-8").unwrap());
    }
    if let Some(s) = r.session {
        resp = resp.with_header(tiny_http::Header::from_bytes("Mcp-Session-Id", s.as_bytes()).unwrap());
    }
    if let Some(a) = r.allow {
        resp = resp.with_header(tiny_http::Header::from_bytes("Allow", a).unwrap());
    }
    if r.status == 401 {
        resp = resp.with_header(tiny_http::Header::from_bytes("WWW-Authenticate", "Bearer").unwrap());
    }
    let _ = req.respond(resp);
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::bridge::{self, Bridge, Source};
    use std::time::Duration;

    struct Fake {
        bridge: Bridge,
    }

    impl Backend for Fake {
        fn version(&self) -> String { "1.2.3".into() }
        fn tools(&self) -> Vec<Value> { self.bridge.tools() }
        fn call(&self, name: &str, args: Value, client: &ClientInfo) -> Value {
            self.bridge.call(name, args, Source::Local, client.clone(), "1.2.3", Duration::from_millis(20), || {}, |_| Ok(()))
        }
    }

    fn fake() -> Fake {
        Fake { bridge: Bridge::default() }
    }

    const TOKEN: &str = "ftmcp_0123456789abcdef0123456789abcdef";

    fn post<'a>(body: &'a [u8], auth: Option<&'a str>, session: Option<&'a str>) -> HttpReq<'a> {
        HttpReq { method: "POST", path: "/mcp", host: Some("127.0.0.1:47861"), origin: None, auth, session, body }
    }

    fn c() -> ClientInfo { ClientInfo { name: "t".into(), version: None } }

    #[test]
    fn initialize_negotiates_version() {
        let f = fake();
        let r = handle_message(&json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"claude-ai","version":"0.1"}}}), &f, &c()).unwrap();
        assert_eq!(r["id"], 1);
        assert_eq!(r["result"]["protocolVersion"], "2025-03-26");
        assert_eq!(r["result"]["serverInfo"]["name"], "frametrain");
        assert_eq!(r["result"]["serverInfo"]["version"], "1.2.3");
        assert!(r["result"]["capabilities"]["tools"].is_object());
        let r = handle_message(&json!({"jsonrpc":"2.0","id":"x","method":"initialize","params":{"protocolVersion":"1999-01-01"}}), &f, &c()).unwrap();
        assert_eq!(r["result"]["protocolVersion"], "2025-06-18");
    }

    #[test]
    fn notifications_and_responses_get_no_reply() {
        let f = fake();
        assert!(handle_message(&json!({"jsonrpc":"2.0","method":"notifications/initialized"}), &f, &c()).is_none());
        assert!(handle_message(&json!({"jsonrpc":"2.0","id":4,"result":{}}), &f, &c()).is_none());
    }

    #[test]
    fn ping_list_call_and_errors() {
        let f = fake();
        let r = handle_message(&json!({"jsonrpc":"2.0","id":2,"method":"ping"}), &f, &c()).unwrap();
        assert_eq!(r["result"], json!({}));
        let r = handle_message(&json!({"jsonrpc":"2.0","id":3,"method":"tools/list"}), &f, &c()).unwrap();
        assert_eq!(r["result"]["tools"][0]["name"], bridge::PING_TOOL);
        let r = handle_message(&json!({"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"system_ping"}}), &f, &c()).unwrap();
        assert_eq!(r["result"]["structuredContent"]["version"], "1.2.3");
        let r = handle_message(&json!({"jsonrpc":"2.0","id":6,"method":"tools/call","params":{}}), &f, &c()).unwrap();
        assert_eq!(r["error"]["code"], INVALID_PARAMS);
        let r = handle_message(&json!({"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"system_ping","arguments":[1]}}), &f, &c()).unwrap();
        assert_eq!(r["error"]["code"], INVALID_PARAMS);
        let r = handle_message(&json!({"jsonrpc":"2.0","id":8,"method":"foo/bar"}), &f, &c()).unwrap();
        assert_eq!(r["error"]["code"], METHOD_NOT_FOUND);
        let r = handle_message(&json!({"id":9,"method":"ping"}), &f, &c()).unwrap();
        assert_eq!(r["error"]["code"], INVALID_REQUEST);
        let r = handle_message(&json!([1]), &f, &c()).unwrap();
        assert_eq!(r["error"]["code"], INVALID_REQUEST);
    }

    #[test]
    fn host_origin_token_checks() {
        assert!(host_allowed(Some("127.0.0.1:47861"), 47861));
        assert!(host_allowed(Some("localhost:47861"), 47861));
        assert!(!host_allowed(Some("evil.com:47861"), 47861));
        assert!(!host_allowed(Some("127.0.0.1:1234"), 47861));
        assert!(!host_allowed(None, 47861));
        assert!(origin_allowed(None));
        assert!(origin_allowed(Some("http://localhost:3000")));
        assert!(origin_allowed(Some("http://127.0.0.1")));
        assert!(origin_allowed(Some("http://[::1]:5")));
        assert!(!origin_allowed(Some("https://evil.com")));
        assert!(!origin_allowed(Some("http://localhost.evil.com")));
        assert!(!origin_allowed(Some("null")));
        assert!(token_ok(Some(&format!("Bearer {}", TOKEN)), TOKEN));
        assert!(!token_ok(Some("Bearer ftmcp_00000000000000000000000000000000"), TOKEN));
        assert!(!token_ok(None, TOKEN));
        assert!(!token_ok(Some("Bearer "), ""));
    }

    #[test]
    fn http_layer() {
        let f = fake();
        let s = Sessions::default();
        let auth = format!("Bearer {}", TOKEN);
        // falscher Host
        let mut req = post(b"{}", Some(&auth), None);
        req.host = Some("rebind.example:47861");
        assert_eq!(handle_http(&req, 47861, TOKEN, &s, &f).status, 403);
        // fremder Origin
        let mut req = post(b"{}", Some(&auth), None);
        req.origin = Some("https://evil.com");
        assert_eq!(handle_http(&req, 47861, TOKEN, &s, &f).status, 403);
        // ohne Token
        assert_eq!(handle_http(&post(b"{}", None, None), 47861, TOKEN, &s, &f).status, 401);
        // health ohne Token
        let req = HttpReq { method: "GET", path: "/health", host: Some("127.0.0.1:47861"), origin: None, auth: None, session: None, body: b"" };
        assert_eq!(handle_http(&req, 47861, TOKEN, &s, &f).status, 200);
        // GET /mcp -> 405
        let req = HttpReq { method: "GET", path: "/mcp", host: Some("127.0.0.1:47861"), origin: None, auth: Some(&auth), session: None, body: b"" };
        let r = handle_http(&req, 47861, TOKEN, &s, &f);
        assert_eq!(r.status, 405);
        assert_eq!(r.allow, Some("POST, DELETE"));
        // kaputtes JSON / Batch
        assert_eq!(handle_http(&post(b"{x", Some(&auth), None), 47861, TOKEN, &s, &f).status, 400);
        assert_eq!(handle_http(&post(b"[]", Some(&auth), None), 47861, TOKEN, &s, &f).status, 400);
        // initialize -> Sitzung, Client-Name bleibt erhalten
        let init = br#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","clientInfo":{"name":"Claude Desktop","version":"1"}}}"#;
        let r = handle_http(&post(init, Some(&auth), None), 47861, TOKEN, &s, &f);
        assert_eq!(r.status, 200);
        let sid = r.session.expect("Sitzung");
        assert_eq!(s.client(Some(&sid)).unwrap().name, "Claude Desktop");
        // Notification -> 202
        let r = handle_http(&post(br#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#, Some(&auth), Some(&sid)), 47861, TOKEN, &s, &f);
        assert_eq!(r.status, 202);
        assert!(r.body.is_none());
        // tools/list
        let r = handle_http(&post(br#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#, Some(&auth), Some(&sid)), 47861, TOKEN, &s, &f);
        assert_eq!(r.body.unwrap()["result"]["tools"][0]["name"], "system_ping");
        assert_eq!(s.clients().len(), 1);
        // DELETE beendet Sitzung
        let req = HttpReq { method: "DELETE", path: "/mcp", host: Some("127.0.0.1:47861"), origin: None, auth: Some(&auth), session: Some(&sid), body: b"" };
        assert_eq!(handle_http(&req, 47861, TOKEN, &s, &f).status, 200);
        assert!(s.client(Some(&sid)).is_none());
        // unbekannter Pfad
        let req = HttpReq { method: "POST", path: "/v1/x", host: Some("127.0.0.1:47861"), origin: None, auth: Some(&auth), session: None, body: b"{}" };
        assert_eq!(handle_http(&req, 47861, TOKEN, &s, &f).status, 404);
    }

    #[test]
    fn real_server_roundtrip() {
        // Echter Socket auf einem freien Port
        let port = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let state = Arc::new(ServerState::default());
        let backend: Arc<dyn Backend> = Arc::new(fake());
        let b2 = backend.clone();
        apply(state.clone(), true, port, move || (TOKEN.to_string(), b2.clone()));
        assert_eq!(state.status()["running"], true);
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let body: Value = rt.block_on(async {
            reqwest::Client::new()
                .post(format!("http://127.0.0.1:{}/mcp", port))
                .header("Authorization", format!("Bearer {}", TOKEN))
                .json(&json!({"jsonrpc":"2.0","id":1,"method":"tools/list"}))
                .send().await.unwrap().json().await.unwrap()
        });
        assert_eq!(body["result"]["tools"][0]["name"], "system_ping");
        apply(state.clone(), false, port, move || (TOKEN.to_string(), backend.clone()));
        assert_eq!(state.status()["running"], false);
    }
}
