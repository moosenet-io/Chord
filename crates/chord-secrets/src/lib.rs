//! `chord-secrets` — Chord's own <secret-manager> Universal Auth client (CSEC-01).
//!
//! Standing architectural decision: Chord authenticates to <secret-manager> directly with
//! its own bootstrap identity. This is NOT brokered through `terminus_personal` or
//! any other fleet service over internal HTTP — several internal hops aren't
//! TLS-terminated, so secrets never travel that path.
//!
//! Modeled on the proven Universal Auth flow already shipped twice in this fleet
//! (`moosenet/Terminus`'s `src/<secret-manager>/mod.rs`, `moosenet/harmony`'s
//! `harmony-core/src/secrets/<secret-manager>.rs`): POST `clientId`/`clientSecret` to
//! `{INFISICAL_URL}/api/v1/auth/universal-auth/login` for a bearer token, then
//! `GET {INFISICAL_URL}/api/v3/secrets/raw` (scoped by `workspaceId`/`environment`/
//! `secretPath`) for the secret values.
//!
//! This crate deliberately does NOT keep a background refresh thread or TTL cache
//! (unlike Harmony's fuller `vault::manager` reference) — both of Chord's consumers
//! (CSEC-02's one-shot startup fetch, CSEC-03's chord-tui `SecretManager`) only need
//! a fetch-at-process-start shape, matching the simpler, fresh-auth-per-call
//! Terminus pattern. If a future consumer needs periodic refresh, add it there
//! rather than growing this shared client speculatively.
//!
//! Security:
//! - Secret values are never logged, printed, or included in `Debug`/`Display` of
//!   any error type in this crate — errors carry only key names, HTTP statuses, and
//!   non-secret diagnostic text.
//! - `InfisicalConfig::from_env()` returns a clean "not configured" signal rather
//!   than an error when the required env vars are unset, so callers can fall back
//!   to the static environment without treating an un-migrated deployment as
//!   broken.

use std::collections::HashMap;

use serde::Deserialize;
use serde_json::json;

/// Env var names this crate reads. No infrastructure values are hardcoded — every
/// URL/id/secret comes from the process environment at call time.
mod env_keys {
    pub const URL: &str = "INFISICAL_URL";
    pub const CLIENT_ID: &str = "INFISICAL_CLIENT_ID";
    pub const CLIENT_SECRET: &str = "INFISICAL_CLIENT_SECRET";
}

/// Bootstrap configuration for Chord's <secret-manager> Universal Auth client.
///
/// `project_id`/`environment`/`secret_path` are per-fetch parameters (passed to
/// [`fetch_secrets_batch`]) rather than part of this struct, since a single Chord
/// process may want to fetch from more than one path — but the three bootstrap
/// credential fields (url/client_id/client_secret) are the same for every call.
#[derive(Clone)]
pub struct InfisicalConfig {
    url: Option<String>,
    client_id: Option<String>,
    client_secret: Option<String>,
}

impl InfisicalConfig {
    /// Load bootstrap credentials from the process environment. Never panics and
    /// never treats missing values as an error — `is_configured()` tells the
    /// caller whether a fetch should even be attempted.
    pub fn from_env() -> Self {
        Self {
            url: non_empty_env(env_keys::URL),
            client_id: non_empty_env(env_keys::CLIENT_ID),
            client_secret: non_empty_env(env_keys::CLIENT_SECRET),
        }
    }

    /// `true` only when URL + client id + client secret are all present and
    /// non-empty. Callers should treat `false` as "not configured yet", not an
    /// error — e.g. a deployment that hasn't migrated to <secret-manager>-backed secrets.
    pub fn is_configured(&self) -> bool {
        self.url.is_some() && self.client_id.is_some() && self.client_secret.is_some()
    }

    fn base_url(&self) -> Result<&str, SecretError> {
        self.url
            .as_deref()
            .ok_or_else(|| SecretError::NotConfigured(format!("missing {}", env_keys::URL)))
    }

    fn client_id(&self) -> Result<&str, SecretError> {
        self.client_id
            .as_deref()
            .ok_or_else(|| SecretError::NotConfigured(format!("missing {}", env_keys::CLIENT_ID)))
    }

    fn client_secret(&self) -> Result<&str, SecretError> {
        self.client_secret.as_deref().ok_or_else(|| {
            SecretError::NotConfigured(format!("missing {}", env_keys::CLIENT_SECRET))
        })
    }
}

fn non_empty_env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|s| !s.trim().is_empty())
}

/// Clean, typed errors — never a panic, never a secret value. `Display`/`Debug`
/// output is safe to log directly (key names and HTTP statuses only).
#[derive(Debug, thiserror::Error)]
pub enum SecretError {
    /// Required bootstrap env vars aren't set. Not necessarily a failure — callers
    /// use this to decide "skip the fetch, use the static environment" rather than
    /// treating it as fatal.
    #[error("<secret-manager> not configured: {0}")]
    NotConfigured(String),

    /// The <secret-manager> Universal Auth login call failed (network error or non-2xx).
    #[error("<secret-manager> authentication failed: {0}")]
    Auth(String),

    /// The secrets-fetch HTTP call itself failed (network error or non-2xx).
    #[error("<secret-manager> secrets fetch failed: {0}")]
    Http(String),

    /// A response body didn't parse into the shape this client expects.
    #[error("<secret-manager> response malformed: {0}")]
    Malformed(String),
}

#[derive(Deserialize)]
struct LoginResponse {
    #[serde(rename = "accessToken")]
    access_token: String,
}

#[derive(Deserialize)]
struct SecretsRawResponse {
    secrets: Vec<SecretEntry>,
}

#[derive(Deserialize)]
struct SecretEntry {
    #[serde(rename = "secretKey")]
    secret_key: String,
    #[serde(rename = "secretValue")]
    secret_value: String,
}

fn build_client() -> Result<reqwest::Client, SecretError> {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|e| SecretError::Http(format!("failed to build HTTP client: {e}")))
}

/// Authenticate via <secret-manager> Universal Auth and return a short-lived bearer
/// access token. Fresh per call — this crate keeps no cached token or background
/// refresh thread (see module docs for why).
async fn get_access_token(
    client: &reqwest::Client,
    config: &InfisicalConfig,
) -> Result<String, SecretError> {
    let base = config.base_url()?.trim_end_matches('/');
    let url = format!("{base}/api/v1/auth/universal-auth/login");

    let resp = client
        .post(&url)
        .json(&json!({
            "clientId": config.client_id()?,
            "clientSecret": config.client_secret()?,
        }))
        .send()
        .await
        .map_err(|e| SecretError::Auth(format!("request failed: {e}")))?;

    let status = resp.status();
    if !status.is_success() {
        // Never echo the response body here — it could reflect request contents.
        return Err(SecretError::Auth(format!("HTTP {status}")));
    }

    let body: LoginResponse = resp
        .json()
        .await
        .map_err(|e| SecretError::Malformed(format!("auth response not JSON: {e}")))?;

    Ok(body.access_token)
}

/// Fetch every secret at the given project/environment/path and return a
/// key→value map. Never logs or returns the fetched values anywhere except in
/// the returned map itself — callers are responsible for not logging them
/// further.
pub async fn fetch_secrets_batch(
    config: &InfisicalConfig,
    project_id: &str,
    environment: &str,
    secret_path: &str,
) -> Result<HashMap<String, String>, SecretError> {
    if !config.is_configured() {
        return Err(SecretError::NotConfigured(
            "INFISICAL_URL/INFISICAL_CLIENT_ID/INFISICAL_CLIENT_SECRET not fully set".into(),
        ));
    }

    let client = build_client()?;
    let token = get_access_token(&client, config).await?;
    let base = config.base_url()?.trim_end_matches('/');

    let resp = client
        .get(format!("{base}/api/v3/secrets/raw"))
        .bearer_auth(&token)
        .query(&[
            ("workspaceId", project_id),
            ("environment", environment),
            ("secretPath", secret_path),
        ])
        .send()
        .await
        .map_err(|e| SecretError::Http(format!("request failed: {e}")))?;

    let status = resp.status();
    if !status.is_success() {
        return Err(SecretError::Http(format!("HTTP {status}")));
    }

    let body: SecretsRawResponse = resp
        .json()
        .await
        .map_err(|e| SecretError::Malformed(format!("secrets response not JSON: {e}")))?;

    Ok(body
        .secrets
        .into_iter()
        .map(|s| (s.secret_key, s.secret_value))
        .collect())
}

/// Fetch a single secret by key. Convenience wrapper over
/// [`fetch_secrets_batch`] for callers (like chord-tui's `SecretManager`) that
/// want one value rather than the whole path. Returns `Ok(None)` when <secret-manager>
/// is reachable but the key isn't present at that path — this is NOT an error
/// state, since a missing key is a normal "not set" outcome.
pub async fn fetch_secret(
    config: &InfisicalConfig,
    project_id: &str,
    environment: &str,
    secret_path: &str,
    key: &str,
) -> Result<Option<String>, SecretError> {
    let mut all = fetch_secrets_batch(config, project_id, environment, secret_path).await?;
    Ok(all.remove(key))
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::prelude::*;
    use serial_test::serial;

    fn set_bootstrap_env(base_url: &str) {
        std::env::set_var(env_keys::URL, base_url);
        std::env::set_var(env_keys::CLIENT_ID, "test-client-id"); // pii-test-fixture
        std::env::set_var(env_keys::CLIENT_SECRET, "test-client-secret"); // pii-test-fixture
    }

    fn clear_bootstrap_env() {
        std::env::remove_var(env_keys::URL);
        std::env::remove_var(env_keys::CLIENT_ID);
        std::env::remove_var(env_keys::CLIENT_SECRET);
    }

    // ── is_configured / edge cases ───────────────────────────────────────────

    #[test]
    #[serial]
    fn not_configured_when_env_unset() {
        clear_bootstrap_env();
        let cfg = InfisicalConfig::from_env();
        assert!(!cfg.is_configured());
    }

    #[test]
    #[serial]
    fn not_configured_when_partially_set() {
        clear_bootstrap_env();
        std::env::set_var(env_keys::URL, "http://example.test");
        std::env::set_var(env_keys::CLIENT_ID, "id-only");
        // client secret deliberately left unset
        let cfg = InfisicalConfig::from_env();
        assert!(!cfg.is_configured());
        clear_bootstrap_env();
    }

    #[test]
    #[serial]
    fn empty_string_env_vars_treated_as_unset() {
        clear_bootstrap_env();
        std::env::set_var(env_keys::URL, "");
        std::env::set_var(env_keys::CLIENT_ID, "id");
        std::env::set_var(env_keys::CLIENT_SECRET, "secret"); // pii-test-fixture
        let cfg = InfisicalConfig::from_env();
        assert!(!cfg.is_configured());
        clear_bootstrap_env();
    }

    #[tokio::test]
    #[serial]
    async fn fetch_returns_not_configured_without_panicking_when_unset() {
        clear_bootstrap_env();
        let cfg = InfisicalConfig::from_env();
        let result = fetch_secrets_batch(&cfg, "proj", "prod", "/").await;
        assert!(matches!(result, Err(SecretError::NotConfigured(_))));
    }

    // ── success path (mocked <secret-manager>) ──────────────────────────────────────

    #[tokio::test]
    #[serial]
    async fn success_fetches_and_maps_secrets() {
        let server = MockServer::start();
        let login_mock = server.mock(|when, then| {
            when.method(POST).path("/api/v1/auth/universal-auth/login");
            then.status(200)
                .json_body(json!({ "accessToken": "test-bearer-token" })); // pii-test-fixture
        });
        let secrets_mock = server.mock(|when, then| {
            when.method(GET)
                .path("/api/v3/secrets/raw")
                .query_param("workspaceId", "proj1")
                .query_param("environment", "prod")
                .query_param("secretPath", "/");
            then.status(200).json_body(json!({
                "secrets": [
                    { "secretKey": "CHORD_JWT_SECRET", "secretValue": "jwt-val" }, // pii-test-fixture
                    { "secretKey": "CHORD_API_KEY", "secretValue": "api-val" } // pii-test-fixture
                ]
            }));
        });

        set_bootstrap_env(&server.base_url());
        let cfg = InfisicalConfig::from_env();
        let result = fetch_secrets_batch(&cfg, "proj1", "prod", "/").await.unwrap();

        login_mock.assert();
        secrets_mock.assert();
        assert_eq!(result.get("CHORD_JWT_SECRET").map(String::as_str), Some("jwt-val"));
        assert_eq!(result.get("CHORD_API_KEY").map(String::as_str), Some("api-val"));
        clear_bootstrap_env();
    }

    #[tokio::test]
    #[serial]
    async fn fetch_secret_returns_single_value() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/api/v1/auth/universal-auth/login");
            then.status(200).json_body(json!({ "accessToken": "tok" }));
        });
        server.mock(|when, then| {
            when.method(GET).path("/api/v3/secrets/raw");
            then.status(200).json_body(json!({
                "secrets": [{ "secretKey": "ONLY_KEY", "secretValue": "only-val" }] // pii-test-fixture
            }));
        });

        set_bootstrap_env(&server.base_url());
        let cfg = InfisicalConfig::from_env();
        let found = fetch_secret(&cfg, "proj1", "prod", "/", "ONLY_KEY").await.unwrap();
        let missing = fetch_secret(&cfg, "proj1", "prod", "/", "NOT_THERE").await.unwrap();

        assert_eq!(found.as_deref(), Some("only-val"));
        assert_eq!(missing, None);
        clear_bootstrap_env();
    }

    // ── failure paths ─────────────────────────────────────────────────────────

    #[tokio::test]
    #[serial]
    async fn auth_failure_returns_auth_error() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/api/v1/auth/universal-auth/login");
            then.status(401).json_body(json!({ "message": "invalid client credentials" }));
        });

        set_bootstrap_env(&server.base_url());
        let cfg = InfisicalConfig::from_env();
        let result = fetch_secrets_batch(&cfg, "proj1", "prod", "/").await;

        assert!(matches!(result, Err(SecretError::Auth(_))));
        clear_bootstrap_env();
    }

    #[tokio::test]
    #[serial]
    async fn unreachable_host_returns_auth_error_not_panic() {
        // Port 1 should refuse/never accept a connection.
        set_bootstrap_env("http://127.0.0.1:1");
        let cfg = InfisicalConfig::from_env();
        let result = fetch_secrets_batch(&cfg, "proj1", "prod", "/").await;

        assert!(result.is_err());
        clear_bootstrap_env();
    }

    #[tokio::test]
    #[serial]
    async fn malformed_auth_response_returns_malformed_error() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/api/v1/auth/universal-auth/login");
            then.status(200).body("not json at all");
        });

        set_bootstrap_env(&server.base_url());
        let cfg = InfisicalConfig::from_env();
        let result = fetch_secrets_batch(&cfg, "proj1", "prod", "/").await;

        assert!(matches!(result, Err(SecretError::Malformed(_))));
        clear_bootstrap_env();
    }

    #[tokio::test]
    #[serial]
    async fn malformed_secrets_response_returns_malformed_error() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/api/v1/auth/universal-auth/login");
            then.status(200).json_body(json!({ "accessToken": "tok" }));
        });
        server.mock(|when, then| {
            when.method(GET).path("/api/v3/secrets/raw");
            then.status(200).body("{not valid json");
        });

        set_bootstrap_env(&server.base_url());
        let cfg = InfisicalConfig::from_env();
        let result = fetch_secrets_batch(&cfg, "proj1", "prod", "/").await;

        assert!(matches!(result, Err(SecretError::Malformed(_))));
        clear_bootstrap_env();
    }

    #[tokio::test]
    #[serial]
    async fn secrets_endpoint_failure_returns_http_error() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/api/v1/auth/universal-auth/login");
            then.status(200).json_body(json!({ "accessToken": "tok" }));
        });
        server.mock(|when, then| {
            when.method(GET).path("/api/v3/secrets/raw");
            then.status(500).body("internal error");
        });

        set_bootstrap_env(&server.base_url());
        let cfg = InfisicalConfig::from_env();
        let result = fetch_secrets_batch(&cfg, "proj1", "prod", "/").await;

        assert!(matches!(result, Err(SecretError::Http(_))));
        clear_bootstrap_env();
    }

    // ── never logs/echoes secret values ──────────────────────────────────────

    #[tokio::test]
    #[serial]
    async fn errors_never_contain_secret_values() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/api/v1/auth/universal-auth/login");
            then.status(401).body("nope");
        });

        set_bootstrap_env(&server.base_url());
        std::env::set_var(env_keys::CLIENT_SECRET, "super-secret-value-xyz"); // pii-test-fixture
        let cfg = InfisicalConfig::from_env();
        let result = fetch_secrets_batch(&cfg, "proj1", "prod", "/").await;

        let msg = format!("{}", result.unwrap_err());
        assert!(!msg.contains("super-secret-value-xyz"));
        clear_bootstrap_env();
    }
}
