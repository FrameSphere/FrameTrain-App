// ai_proxy.rs
//
// Leitet KI-HTTP-Aufrufe serverseitig (Rust/reqwest) statt aus dem WebView.
//
// Grund: Anthropic (und je nach Org-Einstellung auch andere) blockieren
// CORS-Anfragen aus dem Browser ("CORS requests are not allowed for this
// Organization"). Abo-/OAuth-Orgs koennen CORS nicht aktivieren. Die
// Claude-Code-CLI umgeht das, weil sie serverseitig laeuft — genau das machen
// wir hier: ein simpler POST-Proxy ohne Origin/Preflight.

use std::collections::HashMap;
use serde::Serialize;

#[derive(Serialize)]
pub struct AiHttpResponse {
    pub status: u16,
    pub body: String,
}

/// Fuehrt einen POST mit beliebigen Headern + JSON-Body aus und gibt Status
/// und Rohtext zurueck. Das Frontend interpretiert Status/Body wie bei fetch().
#[tauri::command]
pub async fn ai_http_post(
    url: String,
    headers: HashMap<String, String>,
    body: String,
) -> Result<AiHttpResponse, String> {
    let client = reqwest::Client::new();
    let mut req = client.post(&url).body(body);
    for (k, v) in headers {
        req = req.header(k, v);
    }
    let resp = req
        .send()
        .await
        .map_err(|e| format!("Netzwerkfehler: {}", e))?;
    let status = resp.status().as_u16();
    let text = resp
        .text()
        .await
        .map_err(|e| format!("Antwort konnte nicht gelesen werden: {}", e))?;
    Ok(AiHttpResponse { status, body: text })
}
