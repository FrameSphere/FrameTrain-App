// secret_store.rs
//
// Sichere Ablage sensibler Werte (v.a. KI-API-Keys) im Betriebssystem-Schluesselbund
// statt im Klartext im localStorage der Webview.
//
//   - macOS   -> Keychain
//   - Windows -> Credential Manager
//
// Der `keyring`-Crate uebernimmt die plattformspezifische Anbindung. Jeder Wert
// wird unter dem festen Service-Namen (App-Identifier) und einem frei waehlbaren
// Konto-Schluessel (`key`) abgelegt, sodass wir pro Nutzer/Zweck getrennte
// Eintraege fuehren koennen (z.B. `ft_ai_key_<userId>`).

use keyring::{Entry, Error as KeyringError};

/// Fester Service-Name im Schluesselbund (entspricht dem App-Identifier).
const SERVICE: &str = "com.frametrain.desktop";

fn entry(key: &str) -> Result<Entry, String> {
    Entry::new(SERVICE, key).map_err(|e| format!("Schluesselbund nicht verfuegbar: {}", e))
}

/// Legt einen Wert sicher im OS-Schluesselbund ab (ueberschreibt vorhandene Eintraege).
#[tauri::command]
pub fn secret_set(key: String, value: String) -> Result<(), String> {
    entry(&key)?
        .set_password(&value)
        .map_err(|e| format!("Konnte Geheimnis nicht speichern: {}", e))
}

/// Liest einen Wert aus dem Schluesselbund. Gibt `None` zurueck, wenn kein
/// Eintrag existiert (kein Fehler) — so kann das Frontend sauber unterscheiden.
#[tauri::command]
pub fn secret_get(key: String) -> Result<Option<String>, String> {
    match entry(&key)?.get_password() {
        Ok(v) => Ok(Some(v)),
        Err(KeyringError::NoEntry) => Ok(None),
        Err(e) => Err(format!("Konnte Geheimnis nicht lesen: {}", e)),
    }
}

/// Loescht einen Eintrag. Ein bereits fehlender Eintrag gilt als Erfolg
/// (idempotent), damit "Key leeren" immer durchlaeuft.
#[tauri::command]
pub fn secret_delete(key: String) -> Result<(), String> {
    match entry(&key)?.delete_credential() {
        Ok(()) => Ok(()),
        Err(KeyringError::NoEntry) => Ok(()),
        Err(e) => Err(format!("Konnte Geheimnis nicht loeschen: {}", e)),
    }
}
