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

/// Ob ein Eintrag existiert — ohne das Geheimnis zu lesen. Unter macOS fragt
/// jedes Auslesen nach dem Schluesselbund-Passwort, sobald sich die Signatur
/// der App geaendert hat (ad-hoc, also bei jedem Release). Die Einstellungen
/// lasen den HF-Token deshalb schon beim Oeffnen und loesten einen
/// unerklaerten Dialog aus. Die Attribute eines Eintrags sind nicht
/// geschuetzt; die Abfrage kommt ohne Dialog aus.
#[tauri::command]
pub fn secret_exists(key: String) -> Result<bool, String> {
    #[cfg(target_os = "macos")]
    {
        use security_framework::item::{ItemClass, ItemSearchOptions, Limit};
        // errSecItemNotFound
        const NICHT_DA: i32 = -25300;
        return match ItemSearchOptions::new()
            .class(ItemClass::generic_password())
            .service(SERVICE)
            .account(&key)
            .load_attributes(true)
            .limit(Limit::Max(1))
            .search()
        {
            Ok(treffer) => Ok(!treffer.is_empty()),
            Err(e) if e.code() == NICHT_DA => Ok(false),
            Err(e) => Err(format!("Schluesselbund nicht verfuegbar: {}", e)),
        };
    }
    #[cfg(not(target_os = "macos"))]
    {
        // Windows-Anmeldeinformationen fragen nicht nach; Lesen ist hier unkritisch.
        secret_get(key).map(|v| v.is_some())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exists_findet_was_keyring_ablegt() {
        let key = format!("ft_test_exists_{}", std::process::id());
        assert_eq!(secret_exists(key.clone()), Ok(false));
        secret_set(key.clone(), "geheim".into()).unwrap();
        assert_eq!(secret_exists(key.clone()), Ok(true), "gleicher Dienst und gleiches Konto wie keyring");
        secret_delete(key.clone()).unwrap();
        assert_eq!(secret_exists(key), Ok(false));
    }
}
