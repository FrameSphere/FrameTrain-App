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
}

#[derive(Debug, Serialize, Deserialize)]
struct CredentialRequest {
    #[serde(rename = "apiKey")]
    api_key: String,
    password: String,
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
}

/// Validiert API-Key und Password über das Backend-API
#[tauri::command]
pub async fn validate_credentials(api_key: String, password: String) -> Result<ApiKeyValidation, String> {
    println!("🔐 validate_credentials called (via Backend API)");
    println!("   API Key: {}...", &api_key.get(..10).unwrap_or(""));
    println!("   Password length: {}", password.len());

    // 1. Prüfe API-Key Format (lokale Validierung für schnelles Feedback)
    if !api_key.starts_with("ft_") || api_key.len() < 24 {
        return Err("Ungültiges API-Key Format".to_string());
    }

    // 2. Prüfe Password-Länge
    if password.is_empty() {
        return Err("Passwort darf nicht leer sein".to_string());
    }

    // 3. Erstelle Request Body
    let request_body = CredentialRequest {
        api_key: api_key.clone(),
        password: password.clone(),
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

    println!("   Response body: {}", response_text);

    let credential_response: CredentialResponse = serde_json::from_str(&response_text)
        .map_err(|e| {
            eprintln!("❌ JSON parse error: {}", e);
            format!("Ungültige Antwort vom Server: {}", e)
        })?;

    // 7. Prüfe Response
    if !credential_response.is_valid {
        let error_message = credential_response.error
            .unwrap_or_else(|| "Validierung fehlgeschlagen".to_string());
        println!("   ❌ Validation failed: {}", error_message);
        return Err(error_message);
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
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_key_format_validation() {
        // Test ungültiges Format
        let result = tokio_test::block_on(async {
            validate_credentials("invalid".to_string(), "password123".to_string()).await
        });
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Ungültiges API-Key Format"));
    }

    #[test]
    fn test_empty_password() {
        let result = tokio_test::block_on(async {
            validate_credentials("ft_1234567890123456789012".to_string(), "".to_string()).await
        });
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Passwort darf nicht leer sein"));
    }
}
