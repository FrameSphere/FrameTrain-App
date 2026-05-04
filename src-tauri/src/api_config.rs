// API Configuration for FrameTrain Desktop App
// This file contains the API endpoints for communicating with the cloud backend

use std::sync::OnceLock;

/// Production API base URL
pub const PRODUCTION_API_URL: &str = "https://frame-train.vercel.app/api";

/// Development API base URL (for local testing)
pub const DEVELOPMENT_API_URL: &str = "http://localhost:3000/api";

/// Get the current API base URL based on build configuration
pub fn get_api_base_url() -> &'static str {
    #[cfg(debug_assertions)]
    {
        static DEBUG_API_URL: OnceLock<String> = OnceLock::new();
        let url = DEBUG_API_URL.get_or_init(|| {
            std::env::var("FRAMETRAIN_API_URL")
                .ok()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| PRODUCTION_API_URL.to_string())
        });
        return url.as_str();
    }

    #[cfg(not(debug_assertions))]
    {
        static RELEASE_API_URL: OnceLock<String> = OnceLock::new();
        let url = RELEASE_API_URL.get_or_init(|| {
            std::env::var("FRAMETRAIN_API_URL")
                .ok()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| PRODUCTION_API_URL.to_string())
        });
        return url.as_str();
    }
}

/// Returns true when the app is explicitly configured to talk to a local dev API.
pub fn is_local_dev_api() -> bool {
    let url = get_api_base_url().to_ascii_lowercase();
    url.contains("localhost") || url.contains("127.0.0.1")
}

/// Desktop API endpoints
pub mod endpoints {
    use super::get_api_base_url;
    
    /// Get the full URL for the credential validation endpoint
    pub fn validate_credentials() -> String {
        format!("{}/desktop/validate-credentials", get_api_base_url())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_production_url() {
        assert_eq!(PRODUCTION_API_URL, "https://frame-train.vercel.app/api");
    }
    
    #[test]
    fn test_endpoint_construction() {
        let url = endpoints::validate_credentials();
        assert!(url.ends_with("/desktop/validate-credentials"));
    }
}
