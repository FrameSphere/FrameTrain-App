// mcp/config.rs – <app_data>/mcp.json
//
// Wird von der App UND von der Stdio-Bruecke (`--mcp-stdio`, ohne Tauri) gelesen,
// deshalb ohne AppHandle: der Pfad kommt aus database::get_data_directory(),
// das genauso aufgebaut ist wie Tauris app_data_dir.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const DEFAULT_PORT: u16 = 47861;
pub const TOKEN_PREFIX: &str = "ftmcp_";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct McpConfig {
    /// Lokaler Endpunkt an/aus (Standard aus, "Mit Claude Desktop verbinden" schaltet ihn ein)
    pub enabled: bool,
    pub port: u16,
    /// ftmcp_<32 hex>
    pub token: String,
    /// UUID, einmal erzeugt – Geraete-Kennung fuer das Relay
    pub device_id: String,
    /// Fernzugriff ueber frame-train.com (Standard aus)
    pub remote_enabled: bool,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self { enabled: false, port: DEFAULT_PORT, token: String::new(), device_id: String::new(), remote_enabled: false }
    }
}

pub fn new_token() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let bytes: [u8; 16] = rng.gen();
    format!("{}{}", TOKEN_PREFIX, bytes.iter().map(|b| format!("{:02x}", b)).collect::<String>())
}

pub fn token_valid_format(t: &str) -> bool {
    t.strip_prefix(TOKEN_PREFIX)
        .map(|h| h.len() == 32 && h.bytes().all(|b| b.is_ascii_hexdigit()))
        .unwrap_or(false)
}

impl McpConfig {
    /// Fehlende/kaputte Felder ergaenzen. true = es wurde etwas geaendert (speichern).
    pub fn fill_missing(&mut self) -> bool {
        let mut changed = false;
        if !token_valid_format(&self.token) {
            self.token = new_token();
            changed = true;
        }
        if uuid::Uuid::parse_str(&self.device_id).is_err() {
            self.device_id = uuid::Uuid::new_v4().to_string();
            changed = true;
        }
        if self.port == 0 {
            self.port = DEFAULT_PORT;
            changed = true;
        }
        changed
    }
}

pub fn config_path() -> PathBuf {
    crate::database::get_data_directory().join("mcp.json")
}

/// Liest die Datei; fehlt sie oder ist sie kaputt -> Standardwerte.
pub fn read_from(path: &Path) -> McpConfig {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str::<McpConfig>(&s).ok())
        .unwrap_or_default()
}

pub fn write_to(path: &Path, cfg: &McpConfig) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("mcp.json: Ordner anlegen: {}", e))?;
    }
    let json = serde_json::to_string_pretty(cfg).map_err(|e| e.to_string())?;
    // Erst in eine Nachbardatei, dann umbenennen – die Stdio-Bruecke liest parallel.
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json).map_err(|e| format!("mcp.json schreiben: {}", e))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600));
    }
    std::fs::rename(&tmp, path).map_err(|e| format!("mcp.json ersetzen: {}", e))
}

/// Laden, fehlende Werte erzeugen und ggf. sofort speichern.
pub fn load_or_init(path: &Path) -> McpConfig {
    let mut cfg = read_from(path);
    if cfg.fill_missing() {
        if let Err(e) = write_to(path, &cfg) {
            eprintln!("[MCP] {}", e);
        }
    }
    cfg
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ft-mcp-test-{}-{}", name, uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn token_format() {
        let t = new_token();
        assert!(t.starts_with("ftmcp_"));
        assert_eq!(t.len(), 6 + 32);
        assert!(token_valid_format(&t));
        assert!(!token_valid_format("ftmcp_xyz"));
        assert!(!token_valid_format("ft-0123456789abcdef0123456789abcdef"));
        assert_ne!(new_token(), new_token());
    }

    #[test]
    fn missing_file_gives_defaults_and_init_writes() {
        let d = tmpdir("init");
        let p = d.join("mcp.json");
        assert_eq!(read_from(&p), McpConfig::default());
        let c = load_or_init(&p);
        assert!(!c.enabled);
        assert!(!c.remote_enabled);
        assert_eq!(c.port, DEFAULT_PORT);
        assert!(token_valid_format(&c.token));
        assert!(uuid::Uuid::parse_str(&c.device_id).is_ok());
        // Zweites Laden aendert nichts mehr
        assert_eq!(load_or_init(&p), c);
        let raw = std::fs::read_to_string(&p).unwrap();
        assert!(raw.contains("\"deviceId\""));
        assert!(raw.contains("\"remoteEnabled\""));
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn partial_and_broken_files() {
        let d = tmpdir("partial");
        let p = d.join("mcp.json");
        std::fs::write(&p, r#"{"enabled":true,"port":5000}"#).unwrap();
        let c = load_or_init(&p);
        assert!(c.enabled);
        assert_eq!(c.port, 5000);
        assert!(token_valid_format(&c.token));
        std::fs::write(&p, "{kaputt").unwrap();
        assert_eq!(read_from(&p), McpConfig::default());
        let _ = std::fs::remove_dir_all(d);
    }
}
