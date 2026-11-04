// mcp/stdio.rs – Stdio-Bruecke fuer Claude Desktop (Vertrag Abschnitt 4)
//
// `FrameTrain --mcp-stdio` laeuft OHNE Tauri: JSON-RPC zeilenweise von stdin,
// per HTTP an den lokalen Endpunkt der laufenden App, Antwort als eine Zeile
// auf stdout. Auf stdout darf NICHTS anderes landen – Meldungen gehen nach stderr.
// Laeuft die App nicht, wird sie gestartet und bis 30 s gewartet.

use super::config::{self, McpConfig};
use crate::command_ext::NoWindow;
use serde_json::{json, Value};
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const FLAG: &str = "--mcp-stdio";
const START_WAIT: Duration = Duration::from_secs(30);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

macro_rules! log {
    ($($t:tt)*) => { eprintln!("[FrameTrain MCP] {}", format!($($t)*)) };
}

pub fn requested(args: &[String]) -> bool {
    args.iter().skip(1).any(|a| a == FLAG)
}

/// Einstieg aus main(). Rueckgabe = Exit-Code.
pub fn run() -> i32 {
    let rt = match tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            log!("Tokio-Runtime: {}", e);
            return 1;
        }
    };
    rt.block_on(run_async())
}

struct Proxy {
    http: reqwest::Client,
    session: Mutex<Option<String>>,
    launched: AtomicBool,
    start_lock: tokio::sync::Mutex<()>,
    out: Mutex<std::io::Stdout>,
}

impl Proxy {
    fn write(&self, v: &Value) {
        // serde_json::to_string erzeugt nie Zeilenumbrueche -> eine Nachricht pro Zeile
        let line = v.to_string();
        let mut out = self.out.lock().unwrap();
        let _ = writeln!(out, "{}", line);
        let _ = out.flush();
    }

    async fn healthy(&self, port: u16) -> bool {
        let url = format!("http://127.0.0.1:{}/health", port);
        match self.http.get(url).timeout(Duration::from_millis(1500)).send().await {
            Ok(r) if r.status().is_success() => r.json::<Value>().await.map(|v| v["mcp"] == true).unwrap_or(false),
            _ => false,
        }
    }

    /// Stellt sicher, dass die App mit eingeschaltetem Endpunkt laeuft.
    async fn ensure_app(&self) -> Result<McpConfig, String> {
        let _g = self.start_lock.lock().await;
        let cfg = config::read_from(&config::config_path());
        if self.healthy(cfg.port).await {
            return Ok(cfg);
        }
        let already = self.launched.swap(true, Ordering::SeqCst);
        if !already {
            log!("FrameTrain laeuft nicht, starte die App …");
            if let Err(e) = launch_app() {
                log!("Start fehlgeschlagen: {}", e);
            }
        }
        // Schon einmal gestartet und Zugang aus: nicht bei jeder Anfrage 30 s warten.
        let until = Instant::now() + if already && !cfg.enabled { Duration::from_secs(2) } else { START_WAIT };
        while Instant::now() < until {
            tokio::time::sleep(Duration::from_millis(500)).await;
            let cfg = config::read_from(&config::config_path());
            if self.healthy(cfg.port).await {
                return Ok(cfg);
            }
        }
        let cfg = config::read_from(&config::config_path());
        Err(if !cfg.enabled {
            "Der lokale MCP-Zugang ist in FrameTrain ausgeschaltet. Bitte in FrameTrain unter Einstellungen > KI-Verbindungen einschalten. (Local MCP access is turned off in FrameTrain: Settings > AI connections.)".to_string()
        } else {
            format!("FrameTrain ist nicht erreichbar (Port {}). Bitte die App starten. (FrameTrain is not reachable, please open the app.)", cfg.port)
        })
    }

    async fn post(&self, cfg: &McpConfig, msg: &Value) -> Result<reqwest::Response, reqwest::Error> {
        let mut req = self.http
            .post(format!("http://127.0.0.1:{}/mcp", cfg.port))
            .header("Authorization", format!("Bearer {}", cfg.token))
            .header("Accept", "application/json, text/event-stream")
            .header("MCP-Protocol-Version", "2025-06-18")
            .json(msg);
        if let Some(s) = self.session.lock().unwrap().clone() {
            req = req.header("Mcp-Session-Id", s);
        }
        req.send().await
    }

    /// Leitet eine Nachricht weiter. Ok(None) = keine Antwort (Notification).
    async fn forward(&self, msg: &Value) -> Result<Option<Value>, String> {
        let mut cfg = self.ensure_app().await?;
        for attempt in 0..2 {
            match self.post(&cfg, msg).await {
                Ok(resp) => {
                    let status = resp.status().as_u16();
                    if let Some(s) = resp.headers().get("mcp-session-id").and_then(|v| v.to_str().ok()) {
                        *self.session.lock().unwrap() = Some(s.to_string());
                    }
                    match status {
                        202 => return Ok(None),
                        200 => return resp.json::<Value>().await.map(Some).map_err(|e| format!("Antwort unlesbar: {}", e)),
                        401 if attempt == 0 => {
                            // Token wurde womoeglich gerade neu erzeugt
                            cfg = config::read_from(&config::config_path());
                            continue;
                        }
                        401 => return Err("FrameTrain hat den Token abgelehnt. In Einstellungen > KI-Verbindungen erneut 'Mit Claude Desktop verbinden'.".into()),
                        s => {
                            let body = resp.text().await.unwrap_or_default();
                            if let Ok(v) = serde_json::from_str::<Value>(&body) {
                                if v.get("jsonrpc").is_some() {
                                    return Ok(Some(v));
                                }
                            }
                            return Err(format!("FrameTrain antwortete mit HTTP {}: {}", s, body.chars().take(200).collect::<String>()));
                        }
                    }
                }
                Err(e) if attempt == 0 => {
                    log!("Verbindung fehlgeschlagen ({}), pruefe die App erneut", e);
                    cfg = self.ensure_app().await?;
                }
                Err(e) => return Err(format!("Keine Verbindung zu FrameTrain: {}", e)),
            }
        }
        Err("Keine Verbindung zu FrameTrain".into())
    }

    async fn handle_line(&self, line: &str) {
        let msg: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => {
                self.write(&json!({ "jsonrpc": "2.0", "id": null, "error": { "code": -32700, "message": "Parse error" } }));
                return;
            }
        };
        let id = msg.get("id").cloned();
        let is_request = id.is_some() && msg.get("method").is_some();
        match self.forward(&msg).await {
            Ok(Some(resp)) => {
                if is_request {
                    self.write(&resp);
                }
            }
            Ok(None) => {}
            Err(e) => {
                log!("{}", e);
                if is_request {
                    self.write(&json!({ "jsonrpc": "2.0", "id": id, "error": { "code": -32000, "message": e } }));
                }
            }
        }
    }
}

async fn run_async() -> i32 {
    use tokio::io::AsyncBufReadExt;
    let http = match reqwest::Client::builder().timeout(REQUEST_TIMEOUT).no_proxy().build() {
        Ok(c) => c,
        Err(e) => {
            log!("HTTP-Client: {}", e);
            return 1;
        }
    };
    let proxy = Arc::new(Proxy {
        http,
        session: Mutex::new(None),
        launched: AtomicBool::new(false),
        start_lock: tokio::sync::Mutex::new(()),
        out: Mutex::new(std::io::stdout()),
    });
    let mut lines = tokio::io::BufReader::new(tokio::io::stdin()).lines();
    let mut tasks = tokio::task::JoinSet::new();
    loop {
        match lines.next_line().await {
            Ok(Some(line)) => {
                let line = line.trim().to_string();
                if line.is_empty() {
                    continue;
                }
                let p = proxy.clone();
                tasks.spawn(async move { p.handle_line(&line).await });
            }
            Ok(None) => break,
            Err(e) => {
                log!("stdin: {}", e);
                break;
            }
        }
    }
    // stdin zu: offene Anfragen noch beantworten, dann Ende
    while tasks.join_next().await.is_some() {}
    0
}

/// App starten: macOS ueber die Bundle-ID (startet das installierte Bundle,
/// bringt es nicht nach vorn), sonst das eigene Binary ohne Argumente.
fn launch_app() -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    #[cfg(target_os = "macos")]
    {
        if exe.to_string_lossy().contains(".app/Contents/MacOS/") {
            let ok = std::process::Command::new("open")
                .args(["-g", "-b", crate::database::APP_ID])
                .no_window()
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            if ok {
                return Ok(());
            }
        }
    }
    std::process::Command::new(&exe)
        .no_window()
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map(|_| ())
        .map_err(|e| format!("{}: {}", exe.display(), e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag_detection() {
        assert!(requested(&["/x/FrameTrain".into(), "--mcp-stdio".into()]));
        assert!(!requested(&["/x/FrameTrain".into()]));
        // Programmpfad selbst zaehlt nicht
        assert!(!requested(&["--mcp-stdio".into()]));
    }
}
