// hosting_api.rs – Lokale HTTP-API fuer gehostete Modelle
//
// Nur an 127.0.0.1 gebunden, jede Anfrage braucht den Token
// (Authorization: Bearer ft-…). Der Host-Header muss localhost/127.0.0.1
// sein — so kann eine Webseite im Browser die API auch per DNS-Rebinding
// nicht erreichen; CORS-Header gibt es bewusst keine.
//
//   GET  /health                 ohne Token, {"ok": true}
//   GET  /v1/models              gehostete Modelle (OpenAI-Format)
//   POST /v1/chat/completions    OpenAI-kompatibel, auch "stream": true (SSE)
//   POST /v1/infer               jede Aufgabe: text | file_path | file_base64, question, params

use crate::hosting_manager::{self, SharedHosting};
use crate::model_host::InferInput;
use std::io::{Read, Write};
use std::sync::{Arc, Mutex};
use tauri::Manager;

const MAX_BODY: usize = 64 * 1024 * 1024;

struct Running {
    port: u16,
    server: Arc<tiny_http::Server>,
}

#[derive(Default)]
pub struct ApiState {
    running: Mutex<Option<Running>>,
    error: Mutex<Option<String>>,
}

pub fn status(app: &tauri::AppHandle) -> serde_json::Value {
    let st = app.state::<ApiState>();
    let running = st.running.lock().unwrap();
    serde_json::json!({
        "running": running.is_some(),
        "url": running.as_ref().map(|r| format!("http://127.0.0.1:{}/v1", r.port)),
        "error": st.error.lock().unwrap().clone(),
    })
}

/// Startet, stoppt oder verlegt den Server passend zu den Einstellungen.
pub fn apply(app: &tauri::AppHandle, enabled: bool, port: u16) {
    let st = app.state::<ApiState>();
    let mut running = st.running.lock().unwrap();
    if let Some(r) = running.as_ref() {
        if enabled && r.port == port {
            return;
        }
        r.server.unblock();
        *running = None;
        println!("[HostingAPI] gestoppt");
    }
    *st.error.lock().unwrap() = None;
    if !enabled {
        return;
    }
    match tiny_http::Server::http(("127.0.0.1", port)) {
        Ok(server) => {
            let server = Arc::new(server);
            *running = Some(Running { port, server: server.clone() });
            let app = app.clone();
            std::thread::spawn(move || {
                for req in server.incoming_requests() {
                    let app = app.clone();
                    std::thread::spawn(move || handle(app, req, port));
                }
            });
            println!("[HostingAPI] laeuft auf http://127.0.0.1:{}", port);
        }
        Err(e) => {
            let msg = format!("Port {} ist belegt oder gesperrt: {}", port, e);
            eprintln!("[HostingAPI] {}", msg);
            *st.error.lock().unwrap() = Some(msg);
        }
    }
}

// ============ Hilfen ============

fn header(req: &tiny_http::Request, name: &str) -> Option<String> {
    req.headers().iter()
        .find(|h| h.field.as_str().as_str().eq_ignore_ascii_case(name))
        .map(|h| h.value.as_str().to_string())
}

fn json_response(code: u16, body: &serde_json::Value) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    tiny_http::Response::from_data(body.to_string().into_bytes())
        .with_status_code(code)
        .with_header(tiny_http::Header::from_bytes("Content-Type", "application/json; charset=utf-8").unwrap())
}

fn error_body(msg: &str, kind: &str) -> serde_json::Value {
    serde_json::json!({ "error": { "message": msg, "type": kind } })
}

/// Nur lokale Namen im Host-Header — Schutz vor DNS-Rebinding.
pub fn host_allowed(host: Option<&str>, port: u16) -> bool {
    let Some(h) = host else { return false };
    let h = h.trim().to_lowercase();
    [format!("127.0.0.1:{}", port), format!("localhost:{}", port), format!("[::1]:{}", port),
     "127.0.0.1".to_string(), "localhost".to_string()].contains(&h)
}

pub fn token_ok(auth: Option<&str>, api_key: Option<&str>, token: &str) -> bool {
    if token.is_empty() { return false; }
    let given = auth.and_then(|a| a.strip_prefix("Bearer ").or_else(|| a.strip_prefix("bearer ")))
        .or(api_key)
        .map(str::trim);
    // Vergleich in konstanter Zeit
    match given {
        Some(g) if g.len() == token.len() => g.bytes().zip(token.bytes()).fold(0u8, |acc, (a, b)| acc | (a ^ b)) == 0,
        _ => false,
    }
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Text aus einem OpenAI-"content" (String oder Liste von Teilen) und ggf. ein Bild (data:-URL).
pub fn split_content(content: &serde_json::Value) -> (String, Option<(Vec<u8>, String)>) {
    if let Some(s) = content.as_str() {
        return (s.to_string(), None);
    }
    let mut text = Vec::new();
    let mut image = None;
    for part in content.as_array().into_iter().flatten() {
        match part.get("type").and_then(|t| t.as_str()) {
            Some("text") => if let Some(t) = part.get("text").and_then(|t| t.as_str()) { text.push(t.to_string()) },
            Some("image_url") => {
                let url = part.get("image_url").and_then(|u| u.get("url").or(Some(u))).and_then(|u| u.as_str()).unwrap_or("");
                if let Some(d) = decode_data_url(url) { image = Some(d); }
            }
            _ => {}
        }
    }
    (text.join("\n"), image)
}

/// "data:image/png;base64,…" → (Bytes, Endung)
pub fn decode_data_url(url: &str) -> Option<(Vec<u8>, String)> {
    use base64::Engine;
    let rest = url.strip_prefix("data:")?;
    let (meta, data) = rest.split_once(',')?;
    if !meta.ends_with(";base64") { return None; }
    let mime = meta.trim_end_matches(";base64");
    let ext = mime.rsplit('/').next().unwrap_or("bin").replace("jpeg", "jpg").replace("x-wav", "wav").replace("mpeg", "mp3");
    let bytes = base64::engine::general_purpose::STANDARD.decode(data.trim()).ok()?;
    Some((bytes, ext))
}

fn resolve_model(state: &SharedHosting, key: Option<&str>) -> Result<String, String> {
    match key.filter(|k| !k.trim().is_empty()) {
        Some(k) => hosting_manager::find_id(state, k)
            .ok_or_else(|| format!("Modell '{}' ist nicht gehostet. GET /v1/models listet alle.", k)),
        None => hosting_manager::default_or_first(state).ok_or_else(|| "Es ist kein Modell gehostet.".to_string()),
    }
}

// ============ Anfragen ============

fn handle(app: tauri::AppHandle, mut req: tiny_http::Request, port: u16) {
    let method = req.method().clone();
    let url = req.url().split('?').next().unwrap_or("").trim_end_matches('/').to_string();

    if !host_allowed(header(&req, "Host").as_deref(), port) {
        let _ = req.respond(json_response(403, &error_body("Host nicht erlaubt", "forbidden")));
        return;
    }
    if url == "/health" {
        let _ = req.respond(json_response(200, &serde_json::json!({ "ok": true, "app": "FrameTrain" })));
        return;
    }
    let hosting = app.state::<SharedHosting>().inner().clone();
    let token = hosting.settings.lock().unwrap().api_token.clone();
    if !token_ok(header(&req, "Authorization").as_deref(), header(&req, "X-API-Key").as_deref(), &token) {
        let _ = req.respond(json_response(401, &error_body("Token fehlt oder ist falsch (Authorization: Bearer …)", "invalid_api_key")));
        return;
    }

    let body: serde_json::Value = if method == tiny_http::Method::Post {
        let mut buf = Vec::new();
        if req.as_reader().take(MAX_BODY as u64 + 1).read_to_end(&mut buf).is_err() || buf.len() > MAX_BODY {
            let _ = req.respond(json_response(413, &error_body("Anfrage zu gross (max. 64 MB)", "invalid_request_error")));
            return;
        }
        match serde_json::from_slice(&buf) {
            Ok(v) => v,
            Err(e) => {
                let _ = req.respond(json_response(400, &error_body(&format!("Kein gueltiges JSON: {}", e), "invalid_request_error")));
                return;
            }
        }
    } else {
        serde_json::Value::Null
    };

    match (method, url.as_str()) {
        (tiny_http::Method::Get, "/v1/models") => {
            let data: Vec<serde_json::Value> = hosting_manager::list(&hosting).iter().map(|h| serde_json::json!({
                "id": h.api_name,
                "object": "model",
                "created": h.loaded_at.unwrap_or(0),
                "owned_by": "frametrain",
                "frametrain": {
                    "version_id": h.id, "name": h.name, "status": h.status,
                    "modality": h.modality, "input_kind": h.input_kind, "default": h.is_default,
                },
            })).collect();
            let _ = req.respond(json_response(200, &serde_json::json!({ "object": "list", "data": data })));
        }
        (tiny_http::Method::Post, "/v1/chat/completions") => chat(app, hosting, req, body),
        (tiny_http::Method::Post, "/v1/infer") => infer(app, hosting, req, body),
        _ => {
            let _ = req.respond(json_response(404, &error_body("Unbekannter Pfad", "not_found")));
        }
    }
}

fn infer(app: tauri::AppHandle, hosting: SharedHosting, req: tiny_http::Request, body: serde_json::Value) {
    let id = match resolve_model(&hosting, body.get("model").and_then(|m| m.as_str())) {
        Ok(id) => id,
        Err(e) => { let _ = req.respond(json_response(404, &error_body(&e, "model_not_found"))); return; }
    };
    let mut file_path = body.get("file_path").and_then(|v| v.as_str()).map(str::to_string);
    if let Some(b64) = body.get("file_base64").and_then(|v| v.as_str()) {
        use base64::Engine;
        let ext = body.get("file_ext").and_then(|v| v.as_str()).unwrap_or("bin");
        match base64::engine::general_purpose::STANDARD.decode(b64.trim())
            .map_err(|e| e.to_string())
            .and_then(|bytes| hosting_manager::write_upload(&app, &bytes, ext)) {
            Ok(p) => file_path = Some(p),
            Err(e) => { let _ = req.respond(json_response(400, &error_body(&format!("file_base64: {}", e), "invalid_request_error"))); return; }
        }
    }
    let input = InferInput {
        text: body.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        file_path,
        question: body.get("question").and_then(|v| v.as_str()).map(str::to_string),
        start: body.get("start").and_then(|v| v.as_f64()),
        end: body.get("end").and_then(|v| v.as_f64()),
        messages: body.get("messages").and_then(|v| v.as_array()).cloned(),
        stream: false,
        params: body.get("params").cloned(),
    };
    match hosting_manager::infer_on(&app, &hosting, &id, &input, &mut |_| {}) {
        Ok(r) => {
            let name = hosting_manager::list(&hosting).into_iter().find(|h| h.id == id).map(|h| h.api_name).unwrap_or_default();
            let _ = req.respond(json_response(200, &serde_json::json!({ "model": name, "result": r })));
        }
        Err(e) => { let _ = req.respond(json_response(500, &error_body(&e, "inference_error"))); }
    }
}

fn chat(app: tauri::AppHandle, hosting: SharedHosting, req: tiny_http::Request, body: serde_json::Value) {
    let id = match resolve_model(&hosting, body.get("model").and_then(|m| m.as_str())) {
        Ok(id) => id,
        Err(e) => { let _ = req.respond(json_response(404, &error_body(&e, "model_not_found"))); return; }
    };
    let info = hosting_manager::list(&hosting).into_iter().find(|h| h.id == id);
    let model_name = info.as_ref().map(|h| h.api_name.clone()).unwrap_or_default();
    let modality = info.as_ref().and_then(|h| h.modality.clone()).unwrap_or_default();

    let msgs = body.get("messages").and_then(|m| m.as_array()).cloned().unwrap_or_default();
    let Some(last_user) = msgs.iter().rev().find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user")) else {
        let _ = req.respond(json_response(400, &error_body("messages braucht mindestens eine user-Nachricht", "invalid_request_error")));
        return;
    };
    let (text, image) = split_content(last_user.get("content").unwrap_or(&serde_json::Value::Null));
    let file_path = match image {
        Some((bytes, ext)) => match hosting_manager::write_upload(&app, &bytes, &ext) {
            Ok(p) => Some(p),
            Err(e) => { let _ = req.respond(json_response(400, &error_body(&e, "invalid_request_error"))); return; }
        },
        None => None,
    };
    // Verlauf fuer LLMs: nur Text-Inhalte
    let history: Vec<serde_json::Value> = msgs.iter().map(|m| {
        let (t, _) = split_content(m.get("content").unwrap_or(&serde_json::Value::Null));
        serde_json::json!({ "role": m.get("role").cloned().unwrap_or_default(), "content": t })
    }).collect();
    let mut params = serde_json::Map::new();
    if let Some(v) = body.get("max_tokens").or(body.get("max_completion_tokens")) { params.insert("max_new_tokens".into(), v.clone()); }
    if let Some(v) = body.get("temperature") { params.insert("temperature".into(), v.clone()); }
    if let Some(v) = body.get("top_p") { params.insert("top_p".into(), v.clone()); }
    if let Some(v) = body.get("seed") { params.insert("seed".into(), v.clone()); }
    let stream = body.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);
    let is_image_model = modality == "vlm" || info.as_ref().and_then(|h| h.input_kind.as_deref()) == Some("image");
    let input = InferInput {
        text: text.clone(),
        question: if is_image_model && !text.is_empty() { Some(text.clone()) } else { None },
        file_path,
        messages: Some(history),
        stream,
        params: Some(serde_json::Value::Object(params)),
        ..Default::default()
    };
    let cid = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let created = now();

    if !stream {
        match hosting_manager::infer_on(&app, &hosting, &id, &input, &mut |_| {}) {
            Ok(r) => {
                let extra = r.extra.clone().unwrap_or_default();
                let usage = serde_json::json!({
                    "prompt_tokens": extra.get("prompt_tokens").cloned().unwrap_or(0.into()),
                    "completion_tokens": extra.get("completion_tokens").cloned().unwrap_or(0.into()),
                });
                let _ = req.respond(json_response(200, &serde_json::json!({
                    "id": cid, "object": "chat.completion", "created": created, "model": model_name,
                    "choices": [{ "index": 0, "message": { "role": "assistant", "content": r.predicted }, "finish_reason": "stop" }],
                    "usage": usage,
                    "frametrain": r,
                })));
            }
            Err(e) => { let _ = req.respond(json_response(500, &error_body(&e, "inference_error"))); }
        }
        return;
    }

    // Streaming (Server-Sent Events): Antwort von Hand schreiben, damit jedes
    // Stueck sofort rausgeht — tiny_http puffert eigene Antworten.
    let mut w = req.into_writer();
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    if w.write_all(head.as_bytes()).and_then(|_| w.flush()).is_err() { return; }
    let chunk = |delta: serde_json::Value, finish: Option<&str>| serde_json::json!({
        "id": cid, "object": "chat.completion.chunk", "created": created, "model": model_name,
        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
    });
    let send = |w: &mut Box<dyn Write + Send>, v: &serde_json::Value| -> bool {
        w.write_all(format!("data: {}\n\n", v).as_bytes()).and_then(|_| w.flush()).is_ok()
    };
    send(&mut w, &chunk(serde_json::json!({ "role": "assistant" }), None));
    let mut streamed = false;
    let result = {
        let mut on_token = |t: &str| {
            streamed = true;
            let _ = send(&mut w, &chunk(serde_json::json!({ "content": t }), None));
        };
        hosting_manager::infer_on(&app, &hosting, &id, &input, &mut on_token)
    };
    match result {
        Ok(r) => {
            // Nicht-LLM-Modelle streamen nicht: die ganze Antwort als ein Stueck.
            if !streamed {
                send(&mut w, &chunk(serde_json::json!({ "content": r.predicted }), None));
            }
            send(&mut w, &chunk(serde_json::json!({}), Some("stop")));
        }
        Err(e) => {
            send(&mut w, &error_body(&e, "inference_error"));
        }
    }
    let _ = w.write_all(b"data: [DONE]\n\n");
    let _ = w.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nur_lokale_hosts() {
        assert!(host_allowed(Some("127.0.0.1:47860"), 47860));
        assert!(host_allowed(Some("localhost:47860"), 47860));
        assert!(!host_allowed(Some("evil.example:47860"), 47860));
        assert!(!host_allowed(Some("127.0.0.1:1"), 47860));
        assert!(!host_allowed(None, 47860));
    }

    #[test]
    fn token_pruefung() {
        assert!(token_ok(Some("Bearer ft-abc"), None, "ft-abc"));
        assert!(token_ok(None, Some("ft-abc"), "ft-abc"));
        assert!(!token_ok(Some("Bearer ft-abd"), None, "ft-abc"));
        assert!(!token_ok(Some("ft-abc"), None, "ft-abc"));
        assert!(!token_ok(None, None, "ft-abc"));
        assert!(!token_ok(Some("Bearer "), None, ""));
    }

    #[test]
    fn openai_inhalt_mit_bild() {
        let c = serde_json::json!([
            { "type": "text", "text": "Was ist das?" },
            { "type": "image_url", "image_url": { "url": "data:image/png;base64,aGFsbG8=" } }
        ]);
        let (t, img) = split_content(&c);
        assert_eq!(t, "Was ist das?");
        let (bytes, ext) = img.unwrap();
        assert_eq!(bytes, b"hallo");
        assert_eq!(ext, "png");
        assert_eq!(split_content(&serde_json::json!("nur text")).0, "nur text");
    }

    #[test]
    fn data_url_endungen() {
        assert_eq!(decode_data_url("data:image/jpeg;base64,aGk=").unwrap().1, "jpg");
        assert!(decode_data_url("https://x/y.png").is_none());
        assert!(decode_data_url("data:text/plain,hi").is_none());
    }
}
