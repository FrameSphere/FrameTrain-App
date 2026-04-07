// User State Management for Multi-User Data Isolation
use crate::AppState;
use crate::auth::validate_credentials;
use tauri::State;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct UserSession {
    pub user_id: String,
    pub email: String,
    pub logged_in_at: String,
}

/// Login Command - Sets the current user in the database
#[tauri::command]
pub async fn login_user(
    state: State<'_, AppState>,
    api_key: String,
    password: String,
) -> Result<UserSession, String> {
    // 1. Validate credentials against Supabase
    let validation = validate_credentials(api_key.clone(), password).await?;
    
    if !validation.is_valid {
        return Err("Ungültige Anmeldedaten".to_string());
    }
    
    // 2. Set current user in database
    {
        let mut db = state.db.lock()
            .map_err(|e| format!("Database lock error: {}", e))?;
        
        db.set_current_user(validation.user_id.clone());
    }
    
    // 3. Create session
    let session = UserSession {
        user_id: validation.user_id,
        email: validation.email,
        logged_in_at: chrono::Utc::now().to_rfc3339(),
    };
    
    println!("✅ User logged in: {}", session.email);
    
    Ok(session)
}

/// Logout Command - Clears the current user from the database
#[tauri::command]
pub fn logout_user(state: State<'_, AppState>) -> Result<(), String> {
    let mut db = state.db.lock()
        .map_err(|e| format!("Database lock error: {}", e))?;
    
    db.clear_current_user();
    
    println!("✅ User logged out");
    
    Ok(())
}

/// Check if a user is currently logged in
#[tauri::command]
pub fn is_user_logged_in(state: State<'_, AppState>) -> Result<bool, String> {
    let db = state.db.lock()
        .map_err(|e| format!("Database lock error: {}", e))?;
    
    // Check if current_user_id is set (via a dummy operation)
    match db.list_models() {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}
