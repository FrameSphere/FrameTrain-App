// FrameTrain Business Stufe 2: Team-Katalog und Team-Speicher.
//
// Der Katalog (geteilte Modellversionen und Datasets als Metadaten) liegt auf
// dem FrameTrain-Server. Dateien liegen nie dort, sondern im Team-Speicher der
// Firma (NAS, Netzlaufwerk, Cloud-Ordner). Jeder Rechner merkt sich selbst, wo
// dieser Ordner bei ihm eingehaengt ist (team_storage.json im App-Datenordner),
// denn der Pfad ist auf jedem Geraet anders.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Component, Path, PathBuf};
use tauri::{Emitter, Manager};

use crate::api_config::get_api_base_url;

/// Markierung im Team-Speicher: verhindert, dass ein Rechner versehentlich den
/// Ordner eines anderen Teams nutzt.
const MARKER_FILE: &str = ".frametrain-team.json";
/// Beschreibung neben den Dateien eines Eintrags (selbsterklaerender Speicher).
pub const ENTRY_MANIFEST: &str = ".frametrain-entry.json";

#[derive(Debug, Default, Serialize, Deserialize)]
struct StorageConfig {
    /// Team-ID -> lokaler Pfad des Team-Speichers
    #[serde(default)]
    paths: HashMap<String, String>,
}

fn config_file(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let dir = app.path().app_data_dir().map_err(|e| format!("AppDataDir: {}", e))?;
    fs::create_dir_all(&dir).map_err(|e| format!("AppDataDir anlegen: {}", e))?;
    Ok(dir.join("team_storage.json"))
}

fn read_config(app: &tauri::AppHandle) -> StorageConfig {
    config_file(app)
        .ok()
        .and_then(|p| fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn write_config(app: &tauri::AppHandle, cfg: &StorageConfig) -> Result<(), String> {
    let path = config_file(app)?;
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, serde_json::to_string_pretty(cfg).map_err(|e| e.to_string())?)
        .map_err(|e| format!("Einstellung speichern: {}", e))?;
    fs::rename(&tmp, &path).map_err(|e| format!("Einstellung speichern: {}", e))
}

/// Relativer Pfad im Team-Speicher: keine absoluten Pfade, kein "..".
pub fn safe_relative(rel: &str) -> Result<PathBuf, String> {
    let normalized = rel.replace('\\', "/");
    let p = Path::new(&normalized);
    if normalized.is_empty() || p.is_absolute() || normalized.starts_with('/') {
        return Err("Ungueltiger Pfad im Team-Speicher".into());
    }
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::Normal(s) => out.push(s),
            _ => return Err("Ungueltiger Pfad im Team-Speicher".into()),
        }
    }
    if out.as_os_str().is_empty() {
        return Err("Ungueltiger Pfad im Team-Speicher".into());
    }
    Ok(out)
}

fn storage_root(app: &tauri::AppHandle, team_id: &str) -> Result<PathBuf, String> {
    let cfg = read_config(app);
    let root = cfg.paths.get(team_id).ok_or("Kein Team-Speicher eingerichtet")?;
    let root = PathBuf::from(root);
    if !root.is_dir() {
        return Err(format!("Team-Speicher nicht erreichbar: {}", root.display()));
    }
    Ok(root)
}

#[derive(Debug, Serialize)]
pub struct StorageStatus {
    /// Gespeicherter Pfad auf diesem Rechner (None = nicht eingerichtet)
    path: Option<String>,
    /// Ordner gerade erreichbar (NAS eingehaengt, Laufwerk verbunden)
    available: bool,
    /// Ordner gehoert laut Markierung zu einem anderen Team
    foreign_team: Option<String>,
}

fn read_marker(root: &Path) -> Option<serde_json::Value> {
    fs::read_to_string(root.join(MARKER_FILE)).ok().and_then(|s| serde_json::from_str(&s).ok())
}

fn marker_team(root: &Path) -> Option<String> {
    read_marker(root).and_then(|v| v.get("teamId").and_then(|t| t.as_str()).map(str::to_string))
}

#[tauri::command]
pub fn team_storage_get(app: tauri::AppHandle, team_id: String) -> StorageStatus {
    let cfg = read_config(&app);
    match cfg.paths.get(&team_id) {
        None => StorageStatus { path: None, available: false, foreign_team: None },
        Some(p) => {
            let root = PathBuf::from(p);
            let available = root.is_dir();
            let foreign = if available { marker_team(&root).filter(|t| t != &team_id) } else { None };
            StorageStatus { path: Some(p.clone()), available, foreign_team: foreign }
        }
    }
}

/// Team-Speicher fuer dieses Geraet festlegen (None = entfernen). Legt die
/// Markierung an, wenn der Ordner noch keine hat, und prueft Schreibrechte.
#[tauri::command]
pub fn team_storage_set(app: tauri::AppHandle, team_id: String, team_name: String, path: Option<String>) -> Result<StorageStatus, String> {
    let mut cfg = read_config(&app);
    match path {
        None => {
            cfg.paths.remove(&team_id);
        }
        Some(p) => {
            let root = PathBuf::from(&p);
            if !root.is_dir() {
                return Err("Der Ordner existiert nicht oder ist nicht erreichbar.".into());
            }
            match marker_team(&root) {
                Some(other) if other != team_id => {
                    return Err("Dieser Ordner gehoert zu einem anderen FrameTrain-Team.".into());
                }
                Some(_) => {}
                None => {
                    let marker = serde_json::json!({ "teamId": team_id, "teamName": team_name, "app": "FrameTrain" });
                    fs::write(root.join(MARKER_FILE), serde_json::to_string_pretty(&marker).unwrap_or_default())
                        .map_err(|e| format!("Keine Schreibrechte im Ordner: {}", e))?;
                }
            }
            // Schreibprobe (eine vorhandene Markierung beweist keine Schreibrechte)
            let probe = root.join(".frametrain-write-test");
            fs::write(&probe, b"ok").map_err(|e| format!("Keine Schreibrechte im Ordner: {}", e))?;
            let _ = fs::remove_file(&probe);
            cfg.paths.insert(team_id.clone(), p);
        }
    }
    write_config(&app, &cfg)?;
    Ok(team_storage_get(app, team_id))
}

fn dir_size(path: &Path) -> u64 {
    if path.is_file() {
        return fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    }
    let mut total = 0;
    if let Ok(rd) = fs::read_dir(path) {
        for e in rd.flatten() {
            total += dir_size(&e.path());
        }
    }
    total
}

#[derive(Clone, Serialize)]
struct CopyProgress {
    rel_path: String,
    bytes_done: u64,
    bytes_total: u64,
}

struct Copier<'a> {
    app: &'a tauri::AppHandle,
    rel: String,
    done: u64,
    total: u64,
    last_emit: std::time::Instant,
}

impl Copier<'_> {
    fn copy(&mut self, src: &Path, dst: &Path) -> Result<(), String> {
        if src.is_file() {
            if let Some(parent) = dst.parent() {
                fs::create_dir_all(parent).map_err(|e| format!("Ordner anlegen: {}", e))?;
            }
            let n = fs::copy(src, dst).map_err(|e| format!("Kopieren {}: {}", src.display(), e))?;
            self.done += n;
            if self.last_emit.elapsed().as_millis() > 250 {
                self.last_emit = std::time::Instant::now();
                let _ = self.app.emit("team-storage-progress", CopyProgress {
                    rel_path: self.rel.clone(), bytes_done: self.done, bytes_total: self.total,
                });
            }
            return Ok(());
        }
        fs::create_dir_all(dst).map_err(|e| format!("Ordner anlegen: {}", e))?;
        for e in fs::read_dir(src).map_err(|e| format!("Lesen {}: {}", src.display(), e))?.flatten() {
            let name = e.file_name();
            // Systemmuell nicht mitnehmen
            if name == ".DS_Store" || name == "Thumbs.db" {
                continue;
            }
            self.copy(&e.path(), &dst.join(name))?;
        }
        Ok(())
    }
}

/// Dateien eines Eintrags in den Team-Speicher legen. Kopiert erst in einen
/// Zwischenordner und benennt dann um — andere sehen nie halbe Kopien.
/// `manifest` landet als .frametrain-entry.json daneben.
#[tauri::command]
pub async fn team_storage_put(
    app: tauri::AppHandle,
    team_id: String,
    source_path: String,
    rel_path: String,
    manifest: serde_json::Value,
) -> Result<u64, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let root = storage_root(&app, &team_id)?;
        let rel = safe_relative(&rel_path)?;
        let src = PathBuf::from(&source_path);
        if !src.exists() {
            return Err(format!("Quelle nicht gefunden: {}", source_path));
        }
        let target = root.join(&rel);
        let tmp = root.join(format!("{}.uploading-{}", rel.display(), std::process::id()));
        if tmp.exists() {
            let _ = fs::remove_dir_all(&tmp);
        }
        let total = dir_size(&src);
        let mut copier = Copier { app: &app, rel: rel_path.clone(), done: 0, total, last_emit: std::time::Instant::now() };
        let dst = if src.is_file() { tmp.join(src.file_name().unwrap_or_default()) } else { tmp.clone() };
        if let Err(e) = copier.copy(&src, &dst) {
            let _ = fs::remove_dir_all(&tmp);
            return Err(e);
        }
        fs::write(tmp.join(ENTRY_MANIFEST), serde_json::to_string_pretty(&manifest).unwrap_or_default())
            .map_err(|e| format!("Beschreibung schreiben: {}", e))?;
        if target.exists() {
            fs::remove_dir_all(&target).map_err(|e| format!("Alte Fassung entfernen: {}", e))?;
        }
        fs::rename(&tmp, &target).map_err(|e| format!("Abschliessen: {}", e))?;
        let _ = app.emit("team-storage-progress", CopyProgress { rel_path, bytes_done: total, bytes_total: total });
        Ok(total)
    })
    .await
    .map_err(|e| format!("Kopieren abgebrochen: {}", e))?
}

/// Absoluter Pfad eines Eintrags im Team-Speicher (zum Importieren).
/// Liegt dort genau eine Datei (z. B. eine CSV), kommt deren Pfad zurueck.
#[tauri::command]
pub fn team_storage_resolve(app: tauri::AppHandle, team_id: String, rel_path: String) -> Result<String, String> {
    let root = storage_root(&app, &team_id)?;
    let target = root.join(safe_relative(&rel_path)?);
    if !target.exists() {
        return Err("Die Dateien liegen nicht (mehr) im Team-Speicher.".into());
    }
    let files: Vec<PathBuf> = fs::read_dir(&target)
        .map_err(|e| e.to_string())?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.file_name().map(|n| n != ENTRY_MANIFEST).unwrap_or(false))
        .collect();
    if files.len() == 1 && files[0].is_file() {
        return Ok(files[0].to_string_lossy().to_string());
    }
    Ok(target.to_string_lossy().to_string())
}

/// Dateien eines Eintrags aus dem Team-Speicher loeschen.
#[tauri::command]
pub async fn team_storage_remove(app: tauri::AppHandle, team_id: String, rel_path: String) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        let root = storage_root(&app, &team_id)?;
        let target = root.join(safe_relative(&rel_path)?);
        if target.exists() {
            fs::remove_dir_all(&target).map_err(|e| format!("Loeschen: {}", e))?;
        }
        Ok(())
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Erlaubte Team-Schnittstellen des Servers (nichts Beliebiges durchreichen).
fn team_endpoint(path: &str) -> Option<String> {
    const ALLOWED: [&str; 5] = ["catalog", "catalog/share", "catalog/remove", "catalog/fetched", "settings"];
    ALLOWED.contains(&path).then(|| format!("{}/desktop/team/{}", get_api_base_url(), path))
}

/// Anfrage an den Team-Katalog des Servers; API-Key und Geraet setzt das Backend.
#[tauri::command]
pub async fn team_request(api_key: String, path: String, body: Option<serde_json::Value>) -> Result<serde_json::Value, String> {
    let url = team_endpoint(&path).ok_or("Unbekannte Team-Anfrage")?;
    let (device_id, _) = crate::auth::this_device().await;
    let mut payload = match body {
        Some(serde_json::Value::Object(m)) => m,
        _ => serde_json::Map::new(),
    };
    payload.insert("apiKey".into(), serde_json::Value::String(api_key));
    payload.insert("deviceId".into(), serde_json::Value::String(device_id));
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(20))
        .build()
        .map_err(|e| format!("HTTP-Client: {}", e))?;
    let response = client.post(&url).json(&payload).send().await.map_err(|e| format!("Netzwerkfehler: {}", e))?;
    let status = response.status();
    let value: serde_json::Value = response.json().await.unwrap_or(serde_json::Value::Null);
    if !status.is_success() {
        return Err(value.get("error").and_then(|e| e.as_str()).map(str::to_string)
            .unwrap_or_else(|| format!("Server antwortet mit {}", status)));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_pfade_bleiben_im_team_speicher() {
        assert_eq!(safe_relative("models/abc").unwrap(), PathBuf::from("models").join("abc"));
        assert_eq!(safe_relative("models\\abc").unwrap(), PathBuf::from("models").join("abc"));
        assert!(safe_relative("../etc").is_err());
        assert!(safe_relative("models/../../x").is_err());
        assert!(safe_relative("/abs").is_err());
        assert!(safe_relative("").is_err());
    }

    #[test]
    fn nur_bekannte_endpunkte() {
        assert!(team_endpoint("catalog/share").unwrap().ends_with("/desktop/team/catalog/share"));
        assert!(team_endpoint("../keys").is_none());
        assert!(team_endpoint("seat-password").is_none());
    }

    #[test]
    fn markierung_und_groesse() {
        let dir = std::env::temp_dir().join(format!("ft-team-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("sub")).unwrap();
        fs::write(dir.join("a.bin"), vec![0u8; 10]).unwrap();
        fs::write(dir.join("sub/b.bin"), vec![0u8; 5]).unwrap();
        assert_eq!(dir_size(&dir), 15);
        assert!(marker_team(&dir).is_none());
        fs::write(dir.join(MARKER_FILE), r#"{"teamId":"t1"}"#).unwrap();
        assert_eq!(marker_team(&dir).as_deref(), Some("t1"));
        let _ = fs::remove_dir_all(&dir);
    }
}
