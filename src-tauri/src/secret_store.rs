// secret_store.rs
//
// Sichere Ablage sensibler Werte (KI-API-Keys, HF-Token, Login-Daten) im
// Betriebssystem-Schluesselbund statt im Klartext auf der Platte.
//
//   - macOS   -> Keychain
//   - Windows -> Credential Manager
//
// Konto-Trennung: Das Frontend nennt nur noch einen LOGISCHEN Namen
// ("hf_token", "ai_key:anthropic", ...). Den Eintrag im Schluesselbund bildet
// das Backend selbst aus diesem Namen und dem gerade angemeldeten Nutzer.
// Vorher reichte die Webview den fertigen Kontonamen durch und konnte damit
// jeden Eintrag lesen — auch die Keys anderer FrameTrain-Accounts, und der
// HF-Token lag unter einem festen Namen fuer alle Accounts.

use keyring::{Entry, Error as KeyringError};
use std::sync::Mutex;
use tauri::State;

use crate::AppState;

/// Fester Service-Name im Schluesselbund (entspricht dem App-Identifier).
const SERVICE: &str = crate::database::APP_ID;

/// Anbieter, fuer die ein KI-Key abgelegt werden darf.
const AI_PROVIDERS: &[&str] = &["anthropic", "openai", "groq"];

/// Schluesselbund-Konto der gespeicherten Anmeldung (API-Key + Passwort).
/// Nur das Backend kommt daran; es gibt keinen Command, der es frei liest.
const LOGIN_ACCOUNT: &str = "ft_login";

fn entry(account: &str) -> Result<Entry, String> {
    Entry::new(SERVICE, account).map_err(|e| format!("Schluesselbund nicht verfuegbar: {}", e))
}

/// Nutzer-ID so, wie sie in einem Kontonamen stehen darf.
fn sanitize_user_id(user_id: &str) -> String {
    user_id.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_').collect()
}

/// Logischer Name + Nutzer -> Kontoname im Schluesselbund.
///
/// Die Namen der KI-Keys entsprechen genau dem bisherigen Schema
/// (`ft_ai_key_<userId>_<provider>`), damit gespeicherte Keys erhalten bleiben.
pub fn account_for(user_id: &str, key: &str) -> Result<String, String> {
    let uid = sanitize_user_id(user_id);
    if uid.is_empty() {
        return Err("Nicht angemeldet".to_string());
    }
    match key {
        "hf_token" => Ok(format!("ft_hf_token_{}", uid)),
        // Altes gemeinsames KI-Konto (vor der Trennung nach Anbieter)
        "ai_key" => Ok(format!("ft_ai_key_{}", uid)),
        k => match k.strip_prefix("ai_key:") {
            Some(p) if AI_PROVIDERS.contains(&p) => Ok(format!("ft_ai_key_{}_{}", uid, p)),
            _ => Err(format!("Unbekannter Schluessel: {}", key)),
        },
    }
}

fn current_account(state: &State<'_, AppState>, key: &str) -> Result<String, String> {
    let uid = state.db.lock()
        .map_err(|e| format!("Database lock error: {}", e))?
        .get_current_user_id()
        .ok_or_else(|| "Nicht angemeldet".to_string())?;
    account_for(&uid, key)
}

// ============ Rohzugriff auf ein Konto (nur intern) ============

fn raw_set(account: &str, value: &str) -> Result<(), String> {
    entry(account)?
        .set_password(value)
        .map_err(|e| format!("Konnte Geheimnis nicht speichern: {}", e))
}

fn raw_get(account: &str) -> Result<Option<String>, String> {
    match entry(account)?.get_password() {
        Ok(v) => Ok(Some(v)),
        Err(KeyringError::NoEntry) => Ok(None),
        Err(e) => Err(format!("Konnte Geheimnis nicht lesen: {}", e)),
    }
}

/// Ob ein Eintrag existiert — ohne das Geheimnis zu lesen. Unter macOS fragt
/// jedes Auslesen nach dem Schluesselbund-Passwort, sobald sich die Signatur
/// der App geaendert hat (ad-hoc, also bei jedem Release). Die Attribute eines
/// Eintrags sind nicht geschuetzt; die Abfrage kommt ohne Dialog aus.
fn raw_exists(account: &str) -> Result<bool, String> {
    #[cfg(target_os = "macos")]
    {
        use security_framework::item::{ItemClass, ItemSearchOptions, Limit};
        // errSecItemNotFound
        const NICHT_DA: i32 = -25300;
        return match ItemSearchOptions::new()
            .class(ItemClass::generic_password())
            .service(SERVICE)
            .account(account)
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
        raw_get(account).map(|v| v.is_some())
    }
}

/// Ein bereits fehlender Eintrag gilt als Erfolg (idempotent).
fn raw_delete(account: &str) -> Result<(), String> {
    match entry(account)?.delete_credential() {
        Ok(()) => Ok(()),
        Err(KeyringError::NoEntry) => Ok(()),
        Err(e) => Err(format!("Konnte Geheimnis nicht loeschen: {}", e)),
    }
}

// ============ Commands fuer das Frontend (immer im Konto des angemeldeten Nutzers) ============

#[tauri::command]
pub fn secret_set(state: State<'_, AppState>, key: String, value: String) -> Result<(), String> {
    raw_set(&current_account(&state, &key)?, &value)
}

/// Gibt `None` zurueck, wenn kein Eintrag existiert (kein Fehler).
#[tauri::command]
pub fn secret_get(state: State<'_, AppState>, key: String) -> Result<Option<String>, String> {
    raw_get(&current_account(&state, &key)?)
}

#[tauri::command]
pub fn secret_exists(state: State<'_, AppState>, key: String) -> Result<bool, String> {
    raw_exists(&current_account(&state, &key)?)
}

#[tauri::command]
pub fn secret_delete(state: State<'_, AppState>, key: String) -> Result<(), String> {
    raw_delete(&current_account(&state, &key)?)
}

// ============ Gespeicherte Anmeldung (nur Backend) ============

/// API-Key der laufenden Sitzung, fuer Hintergrunddienste (Fern-Relay).
/// Wird beim Login gesetzt und beim Logout geleert — so muss niemand die
/// Anmeldung erneut aus dem Schluesselbund lesen.
static SESSION_API_KEY: Mutex<Option<String>> = Mutex::new(None);

pub fn set_session_api_key(key: Option<String>) {
    if let Ok(mut g) = SESSION_API_KEY.lock() {
        *g = key;
    }
}

pub fn session_api_key() -> Option<String> {
    SESSION_API_KEY.lock().ok().and_then(|g| g.clone())
}

pub fn login_save(config: &str) -> Result<(), String> {
    raw_set(LOGIN_ACCOUNT, config)
}

pub fn login_load() -> Result<Option<String>, String> {
    raw_get(LOGIN_ACCOUNT)
}

pub fn login_clear() -> Result<(), String> {
    raw_delete(LOGIN_ACCOUNT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn konten_sind_pro_nutzer_getrennt() {
        assert_eq!(account_for("user_a", "hf_token").unwrap(), "ft_hf_token_user_a");
        assert_ne!(account_for("user_a", "hf_token"), account_for("user_b", "hf_token"));
        assert_eq!(account_for("u1", "ai_key:groq").unwrap(), "ft_ai_key_u1_groq");
        assert_eq!(account_for("u1", "ai_key").unwrap(), "ft_ai_key_u1");
    }

    #[test]
    fn fremde_und_freie_namen_werden_abgelehnt() {
        // Frueher reichte die Webview den Kontonamen direkt durch.
        assert!(account_for("u1", "ft_ai_key_u2_groq").is_err());
        assert!(account_for("u1", "ft_login").is_err());
        assert!(account_for("u1", "ai_key:evil").is_err());
        assert!(account_for("u1", "ft_hf_token").is_err());
        // Ohne Nutzer kein gemeinsames "anon"-Konto mehr
        assert!(account_for("", "hf_token").is_err());
        // Nutzer-ID kann den Namen nicht verbiegen
        assert_eq!(account_for("a_b/../c", "hf_token").unwrap(), "ft_hf_token_a_bc");
    }

    #[test]
    fn exists_findet_was_keyring_ablegt() {
        let account = format!("ft_test_exists_{}", std::process::id());
        assert_eq!(raw_exists(&account), Ok(false));
        raw_set(&account, "geheim").unwrap();
        assert_eq!(raw_exists(&account), Ok(true), "gleicher Dienst und gleiches Konto wie keyring");
        raw_delete(&account).unwrap();
        assert_eq!(raw_exists(&account), Ok(false));
    }
}
