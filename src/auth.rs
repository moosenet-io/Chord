//! JWT validation for incoming Chord proxy requests.
//!
//! Validates HS256 JWT tokens. The token must have:
//! - A valid HMAC-SHA256 signature using CHORD_JWT_SECRET
//! - `sub` claim equal to "lumina"
//! - `exp` claim in the future
//!
//! When CHORD_JWT_SECRET is empty, auth is disabled (all requests pass through).

use base64::Engine;
use hmac::{Hmac, Mac};
use serde::Deserialize;
use sha2::Sha256;

use crate::error::AuthError;

type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Deserialize, Clone)]
pub struct Claims {
    pub sub: String,
    pub exp: u64,
    /// Optional role claim ("admin", "user", "guest"). Defaults to "user" if absent.
    pub role: Option<String>,
}

/// Validate a JWT token. Returns Claims on success, AuthError on failure.
/// If secret is empty, auth is disabled and returns a synthetic "lumina" claim.
pub fn validate_jwt(token: &str, secret: &str) -> Result<Claims, AuthError> {
    if secret.is_empty() {
        return Ok(Claims {
            sub: "lumina".into(),
            exp: u64::MAX,
            role: None,
        });
    }

    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return Err(AuthError::InvalidToken("Expected 3 parts".into()));
    }

    // Pin the algorithm: reject anything that isn't HS256
    let header_bytes = base64url_decode(parts[0])
        .map_err(|e| AuthError::InvalidToken(format!("Bad header encoding: {e}")))?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes)
        .map_err(|e| AuthError::InvalidToken(format!("Bad header JSON: {e}")))?;
    let alg = header.get("alg").and_then(|a| a.as_str()).unwrap_or("");
    if alg != "HS256" {
        return Err(AuthError::InvalidToken(format!("Unsupported algorithm: {alg}")));
    }

    let signing_input = format!("{}.{}", parts[0], parts[1]);

    // Verify signature
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
        .map_err(|e| AuthError::InvalidToken(format!("Bad secret: {e}")))?;
    mac.update(signing_input.as_bytes());

    let expected_sig = mac.finalize().into_bytes();
    let expected_encoded = base64url_encode(&expected_sig);

    if !constant_time_eq(parts[2], &expected_encoded) {
        return Err(AuthError::InvalidToken("Signature mismatch".into()));
    }

    // Decode payload
    let payload_bytes = base64url_decode(parts[1])
        .map_err(|e| AuthError::InvalidToken(format!("Bad payload encoding: {e}")))?;
    let claims: Claims = serde_json::from_slice(&payload_bytes)
        .map_err(|e| AuthError::InvalidToken(format!("Bad payload JSON: {e}")))?;

    // Check expiry
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|_| AuthError::InvalidToken("System time error".into()))?
        .as_secs();

    if claims.exp < now {
        return Err(AuthError::Expired);
    }

    // Validate subject
    if claims.sub != "lumina" {
        return Err(AuthError::InvalidSubject);
    }

    Ok(claims)
}

/// Extract JWT from "Bearer <token>" Authorization header.
pub fn extract_bearer(header_value: &str) -> Result<&str, AuthError> {
    let value = header_value.trim();
    if let Some(token) = value.strip_prefix("Bearer ") {
        Ok(token.trim())
    } else {
        Err(AuthError::InvalidFormat)
    }
}

fn base64url_encode(input: &[u8]) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(input)
}

fn base64url_decode(input: &str) -> Result<Vec<u8>, base64::DecodeError> {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(input)
}

fn constant_time_eq(a: &str, b: &str) -> bool {
    use subtle::ConstantTimeEq;
    a.as_bytes().ct_eq(b.as_bytes()).into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;
    use hmac::{Hmac, Mac};
    use sha2::Sha256;

    fn make_jwt(sub: &str, exp_offset_secs: i64, secret: &str) -> String {
        let header = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(r#"{"alg":"HS256","typ":"JWT"}"#);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let exp = (now as i64 + exp_offset_secs) as u64;

        let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(format!(r#"{{"sub":"{sub}","exp":{exp}}}"#));

        let signing_input = format!("{header}.{payload}");

        let mut mac: Hmac<Sha256> = Hmac::new_from_slice(secret.as_bytes()).unwrap();
        mac.update(signing_input.as_bytes());
        let sig = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(mac.finalize().into_bytes());

        format!("{signing_input}.{sig}")
    }

    #[test]
    fn test_valid_jwt_accepted() {
        let token = make_jwt("lumina", 3600, "test-secret");
        let claims = validate_jwt(&token, "test-secret").unwrap();
        assert_eq!(claims.sub, "lumina");
    }

    #[test]
    fn test_expired_jwt_rejected() {
        let token = make_jwt("lumina", -100, "test-secret");
        let err = validate_jwt(&token, "test-secret").unwrap_err();
        assert!(matches!(err, AuthError::Expired));
    }

    #[test]
    fn test_wrong_secret_rejected() {
        let token = make_jwt("lumina", 3600, "correct-secret");
        let err = validate_jwt(&token, "wrong-secret").unwrap_err();
        assert!(matches!(err, AuthError::InvalidToken(_)));
    }

    #[test]
    fn test_wrong_subject_rejected() {
        let token = make_jwt("attacker", 3600, "test-secret");
        let err = validate_jwt(&token, "test-secret").unwrap_err();
        assert!(matches!(err, AuthError::InvalidSubject));
    }

    #[test]
    fn test_malformed_token_rejected() {
        let err = validate_jwt("not.a.valid.jwt.here", "secret").unwrap_err();
        assert!(matches!(err, AuthError::InvalidToken(_)));
    }

    #[test]
    fn test_empty_secret_bypasses_auth() {
        // Any token (even invalid) passes when secret is empty
        let claims = validate_jwt("anything", "").unwrap();
        assert_eq!(claims.sub, "lumina");
    }

    #[test]
    fn test_extract_bearer_valid() {
        let token = extract_bearer("Bearer my-token-123").unwrap();
        assert_eq!(token, "my-token-123");
    }

    #[test]
    fn test_extract_bearer_missing_prefix() {
        let err = extract_bearer("my-token-123").unwrap_err();
        assert!(matches!(err, AuthError::InvalidFormat));
    }

    #[test]
    fn test_extract_bearer_trims_whitespace() {
        let token = extract_bearer("  Bearer   my-token  ").unwrap();
        assert_eq!(token, "my-token");
    }

    #[test]
    fn test_alg_none_rejected() {
        // Craft a token with alg:"none" — must be rejected even if signature is empty
        let header = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(r#"{"alg":"none","typ":"JWT"}"#);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(format!(r#"{{"sub":"lumina","exp":{}}}"#, now + 3600));
        let token = format!("{header}.{payload}."); // empty signature

        let err = validate_jwt(&token, "test-secret").unwrap_err();
        assert!(matches!(err, AuthError::InvalidToken(ref msg) if msg.contains("Unsupported algorithm")));
    }

    #[test]
    fn test_alg_rs256_rejected() {
        let header = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(r#"{"alg":"RS256","typ":"JWT"}"#);
        let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(r#"{"sub":"lumina","exp":9999999999}"#);
        let token = format!("{header}.{payload}.fakesig");

        let err = validate_jwt(&token, "test-secret").unwrap_err();
        assert!(matches!(err, AuthError::InvalidToken(ref msg) if msg.contains("Unsupported algorithm")));
    }
}
