// API Configuration for FrameTrain Desktop App
// This file contains the API endpoints for communicating with the cloud backend

/// Production API base URL
pub const PRODUCTION_API_URL: &str = "https://frame-train.vercel.app/api";

/// Development API base URL (for local testing)
pub const DEVELOPMENT_API_URL: &str = "http://localhost:3000/api";

/// Get the current API base URL based on build configuration
pub fn get_api_base_url() -> &'static str {
    #[cfg(debug_assertions)]
    {
        // In debug mode, check for environment variable override
        if let Ok(url) = std::env::var("FRAMETRAIN_API_URL") {
            // Note: This is a static string, so we can't return the env var directly
            // For development, you can change this to your local URL
            return DEVELOPMENT_API_URL;
        }
        DEVELOPMENT_API_URL
    }
    
    #[cfg(not(debug_assertions))]
    {
        PRODUCTION_API_URL
    }
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
