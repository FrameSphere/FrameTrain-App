// Authentifizierungs-Modul für Backend-API Integration
// Verwendet das cloud-hosted Backend anstatt direkter Datenbankverbindung

use serde::{Deserialize, Serialize};
use reqwest;
use crate::api_config::endpoints;

#[derive(Debug, Serialize, Deserialize)]
pub struct ApiKeyValidation {
    pub user_id: String,
    pub email: String,
    pub is_valid: bool,
    /// None = aeltere Server-Version ohne das Feld (dann nichts erzwingen).
    pub terms_accepted: Option<bool>,
    /// Frueher zugestimmt, aber einer aelteren AGB-Fassung.
    pub terms_updated: Option<bool>,
    pub diagnostics_consent: Option<bool>,
    /// Business: Team, Rolle und Platz; None = Einzelkonto oder aeltere Server-Version.
    pub team: Option<serde_json::Value>,
    /// Name der Person auf einem Business-Platz (sonst None).
    pub member_name: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CredentialRequest {
    #[serde(rename = "apiKey")]
    api_key: String,
    password: String, // leer für OAuth-Accounts, Server entscheidet
    /// Geraet fuer Business-Plaetze (ein Platz = ein Geraet gleichzeitig)
    #[serde(rename = "deviceId")]
    device_id: String,
    #[serde(rename = "deviceName")]
    device_name: String,
    /// Platz von einem anderen Geraet uebernehmen
    takeover: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct CredentialResponse {
    #[serde(rename = "isValid")]
    is_valid: bool,
    #[serde(rename = "userId")]
    user_id: Option<String>,
    email: Option<String>,
    message: Option<String>,
    error: Option<String>,
    #[serde(rename = "termsAccepted")]
    terms_accepted: Option<bool>,
    #[serde(rename = "termsUpdated")]
    terms_updated: Option<bool>,
    #[serde(rename = "diagnosticsConsent")]
    diagnostics_consent: Option<bool>,
    team: Option<serde_json::Value>,
    #[serde(rename = "memberName")]
    member_name: Option<String>,
    // Business-Platz: Zustaende, auf die die Anmeldung gezielt reagiert
    #[serde(rename = "needsSeatPassword")]
    needs_seat_password: Option<bool>,
    #[serde(rename = "seatName")]
    seat_name: Option<String>,
    #[serde(rename = "teamName")]
    team_name: Option<String>,
    #[serde(rename = "seatInUse")]
    seat_in_use: Option<bool>,
    #[serde(rename = "deviceName")]
    device_name: Option<String>,
    #[serde(rename = "seatLocked")]
    seat_locked: Option<bool>,
}

/// Fehler, auf die die Anmeldemaske gezielt reagiert, gehen als JSON-Text
/// zurueck: {"code": "...", "message": "...", ...}. Alles andere bleibt Klartext.
fn structured_login_error(r: &CredentialResponse, message: &str) -> Option<String> {
    let v = if r.needs_seat_password == Some(true) {
        serde_json::json!({ "code": "seat_password", "message": message,
            "seatName": r.seat_name, "teamName": r.team_name })
    } else if r.seat_in_use == Some(true) {
        serde_json::json!({ "code": "seat_in_use", "message": message, "deviceName": r.device_name })
    } else if r.seat_locked == Some(true) {
        serde_json::json!({ "code": "seat_locked", "message": message })
    } else {
        return None;
    };
    Some(v.to_string())
}

/// Geraete-ID (dieselbe wie fuer KI-Verbindungen, liegt in mcp.json) und Name.
pub async fn this_device() -> (String, String) {
    tauri::async_runtime::spawn_blocking(|| {
        let id = crate::mcp::config::load_or_init(&crate::mcp::config::config_path()).device_id;
        (id, crate::mcp::relay::device_name())
    })
    .await
    .unwrap_or_else(|_| (String::new(), "FrameTrain Desktop".into()))
}

/// Validiert API-Key und Password über das Backend-API
#[tauri::command]
pub async fn validate_credentials(api_key: String, password: String, takeover: Option<bool>) -> Result<ApiKeyValidation, String> {
    println!("🔐 validate_credentials called (via Backend API)");
    // Hinweis: API-Key/Passwort werden bewusst NICHT geloggt.

    // 1. Prüfe API-Key Format (lokale Validierung für schnelles Feedback)
    if !api_key.starts_with("ft_") || api_key.len() < 24 {
        return Err("Ungültiges API-Key Format".to_string());
    }

    // Hinweis: Passwort kann leer sein (OAuth-Accounts), Server prüft das

    // 2. Leeres Passwort entscheidet der Server: ein neuer Business-Platz hat
    //    noch keins (Antwort "Passwort festlegen"), alle anderen bekommen
    //    "Passwort ist erforderlich".

    // 3. Erstelle Request Body
    let (device_id, device_name) = this_device().await;
    let request_body = CredentialRequest {
        api_key: api_key.clone(),
        password: password.clone(),
        device_id,
        device_name,
        takeover: takeover.unwrap_or(false),
    };

    println!("📡 Sending validation request to backend API...");
    
    // 4. Erstelle HTTP Client mit Timeout
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| format!("Fehler beim Erstellen des HTTP-Clients: {}", e))?;

    // 5. Sende POST-Request an Backend-API
    let api_url = endpoints::validate_credentials();
    println!("   API URL: {}", api_url);

    let response = client
        .post(&api_url)
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send()
        .await
        .map_err(|e| {
            eprintln!("❌ Network error: {}", e);
            format!("Netzwerkfehler: Keine Verbindung zum Server möglich. {}", e)
        })?;

    let status = response.status();
    println!("   Response status: {}", status);

    // 6. Parse Response
    let response_text = response
        .text()
        .await
        .map_err(|e| format!("Fehler beim Lesen der Antwort: {}", e))?;

    #[cfg(debug_assertions)]
    println!("   Response body: {}", response_text);

    let credential_response: CredentialResponse = serde_json::from_str(&response_text)
        .map_err(|e| {
            eprintln!("❌ JSON parse error: {}", e);
            format!("Ungültige Antwort vom Server: {}", e)
        })?;

    // 7. Prüfe Response
    if !credential_response.is_valid {
        let error_message = credential_response.error.clone()
            .unwrap_or_else(|| "Validierung fehlgeschlagen".to_string());
        println!("   ❌ Validation failed: {}", error_message);
        return Err(structured_login_error(&credential_response, &error_message).unwrap_or(error_message));
    }

    // 8. Extrahiere User-Daten
    let user_id = credential_response.user_id
        .ok_or_else(|| "Keine User-ID in der Antwort".to_string())?;
    
    let email = credential_response.email
        .ok_or_else(|| "Keine E-Mail in der Antwort".to_string())?;

    println!("✅ Login successful for: {}", email);

    // 9. Erfolgreiche Validierung
    Ok(ApiKeyValidation {
        user_id,
        email,
        is_valid: true,
        terms_accepted: credential_response.terms_accepted,
        terms_updated: credential_response.terms_updated,
        diagnostics_consent: credential_response.diagnostics_consent,
        team: credential_response.team.filter(|t| !t.is_null()),
        member_name: credential_response.member_name,
    })
}

async fn post_json(url: String, body: serde_json::Value) -> Result<serde_json::Value, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|e| format!("Fehler beim Erstellen des HTTP-Clients: {}", e))?;
    let response = client.post(&url).json(&body).send().await
        .map_err(|e| format!("Netzwerkfehler: {}", e))?;
    let status = response.status();
    let value: serde_json::Value = response.json().await.unwrap_or(serde_json::Value::Null);
    if !status.is_success() {
        let msg = value.get("error").and_then(|e| e.as_str()).map(str::to_string)
            .unwrap_or_else(|| format!("Server antwortet mit {}", status));
        return Err(msg);
    }
    Ok(value)
}

/// Erste Anmeldung auf einem Business-Platz: eigenes Passwort festlegen.
#[tauri::command]
pub async fn team_set_seat_password(api_key: String, password: String) -> Result<(), String> {
    post_json(endpoints::team_seat_password(), serde_json::json!({ "apiKey": api_key, "password": password }))
        .await
        .map(|_| ())
}

/// Team-Seite und Lebenszeichen: Team, Mitglieder und ob der Platz auf
/// diesem Geraet noch aktiv ist ({active, reason, team}).
#[tauri::command]
pub async fn team_overview(api_key: String) -> Result<serde_json::Value, String> {
    let (device_id, _) = this_device().await;
    post_json(endpoints::team(), serde_json::json!({ "apiKey": api_key, "deviceId": device_id })).await
}

/// POST mit API-Key an einen Desktop-Endpoint; Fehler als lesbarer Text.
async fn post_with_api_key(url: String, body: serde_json::Value) -> Result<(), String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|e| format!("Fehler beim Erstellen des HTTP-Clients: {}", e))?;
    let response = client.post(&url).json(&body).send().await
        .map_err(|e| format!("Netzwerkfehler: {}", e))?;
    if !response.status().is_success() {
        return Err(format!("Server antwortet mit {}", response.status()));
    }
    Ok(())
}

/// Schreibt den Schalter "Fehler automatisch melden" ins Konto.
#[tauri::command]
pub async fn set_diagnostics_consent(api_key: String, consent: bool) -> Result<(), String> {
    post_with_api_key(
        endpoints::diagnostics_consent(),
        serde_json::json!({ "apiKey": api_key, "consent": consent }),
    ).await
}

/// Holt die Zustimmung zur aktuellen Fassung von AGB & Datenschutz nach.
#[tauri::command]
pub async fn accept_terms(api_key: String) -> Result<(), String> {
    post_with_api_key(endpoints::accept_terms(), serde_json::json!({ "apiKey": api_key })).await
}

/// Gleicht Support-Tickets mit dem Konto ab: schickt die lokal bekannten
/// (ticket_id, user_token, subject) und liefert alle Tickets des Kontos – so
/// erscheinen Anfragen aus der App auch auf der Website und umgekehrt.
#[tauri::command]
pub async fn sync_support_tickets(api_key: String, tickets: serde_json::Value) -> Result<serde_json::Value, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|e| format!("Fehler beim Erstellen des HTTP-Clients: {}", e))?;
    let response = client
        .post(endpoints::support_tickets())
        .json(&serde_json::json!({ "apiKey": api_key, "tickets": tickets }))
        .send()
        .await
        .map_err(|e| format!("Netzwerkfehler: {}", e))?;
    if !response.status().is_success() {
        return Err(format!("Server antwortet mit {}", response.status()));
    }
    let body: serde_json::Value = response.json().await.map_err(|e| format!("Antwort unlesbar: {}", e))?;
    Ok(body.get("tickets").cloned().unwrap_or_else(|| serde_json::json!([])))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response(json: &str) -> CredentialResponse {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn platz_fehler_gehen_als_json_an_die_anmeldung() {
        let r = response(r#"{"isValid":false,"needsSeatPassword":true,"seatName":"Anna","teamName":"Muster GmbH"}"#);
        let v: serde_json::Value = serde_json::from_str(&structured_login_error(&r, "x").unwrap()).unwrap();
        assert_eq!(v["code"], "seat_password");
        assert_eq!(v["teamName"], "Muster GmbH");

        let r = response(r#"{"isValid":false,"seatInUse":true,"deviceName":"Annas Mac"}"#);
        let v: serde_json::Value = serde_json::from_str(&structured_login_error(&r, "x").unwrap()).unwrap();
        assert_eq!(v["code"], "seat_in_use");
        assert_eq!(v["deviceName"], "Annas Mac");

        let r = response(r#"{"isValid":false,"error":"Falsches Passwort"}"#);
        assert!(structured_login_error(&r, "Falsches Passwort").is_none());
    }

    #[test]
    fn team_fehlt_bei_aelterem_server() {
        let r = response(r#"{"isValid":true,"userId":"u","email":"e"}"#);
        assert!(r.team.is_none());
        let r = response(r#"{"isValid":true,"userId":"u","email":"e","team":null}"#);
        assert!(r.team.filter(|t| !t.is_null()).is_none());
    }

    #[tokio::test]
    async fn test_api_key_format_validation() {
        // Test ungültiges Format
        let result = validate_credentials("invalid".to_string(), "password123".to_string(), None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Ungültiges API-Key Format"));
    }
}
