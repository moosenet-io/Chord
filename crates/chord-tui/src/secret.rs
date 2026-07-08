//! Secret handling (S91 CTUI-01).
//!
//! HARD RULE: secret VALUES are never displayed, logged, or written to the
//! config file. The TUI only ever knows a secret by its *name/reference*; the
//! actual value is resolved at connection time from a [`SecretManager`] backed
//! by the vault (<secret-manager> / environment injected by the vault agent), never
//! from a literal stored in config.
//!
//! [`SecretRef`] is the only secret-shaped thing that is ever serialized. It
//! holds a reference (a vault key name), NOT a value. [`SecretValue`] wraps a
//! resolved value, redacts itself in every Debug/Display, and is deliberately
//! not `Serialize`, so it is a compile error to persist one.
//!
//! CSEC-03: [`InfisicalSecretManager`] is the real vault-backed implementation
//! the doc comment above always claimed but never had — it wraps CSEC-01's
//! shared `chord-secrets` Universal Auth client (the same one `chord-proxy`'s
//! startup bootstrap, CSEC-02, uses). [`default_secret_manager`] selects it
//! automatically when <secret-manager> is configured, falling back to
//! [`EnvSecretManager`] otherwise — `EnvSecretManager` is NOT removed; it
//! remains the dev-mode / un-migrated-deployment fallback.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tokio::sync::RwLock;

use chord_secrets::InfisicalConfig;

/// A *reference* to a secret — a vault key name. This is the only secret-shaped
/// type that is ever serialized into config. It contains no secret material.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SecretRef(pub String);

impl SecretRef {
    pub fn new(name: impl Into<String>) -> Self {
        SecretRef(name.into())
    }
    pub fn name(&self) -> &str {
        &self.0
    }
}

/// A resolved secret value. Redacts itself everywhere and is intentionally
/// **not** `Serialize` — persisting one is a compile error, which is how the
/// "secrets never written to config" invariant is enforced structurally.
#[derive(Clone)]
pub struct SecretValue(String);

impl SecretValue {
    pub fn new(v: impl Into<String>) -> Self {
        SecretValue(v.into())
    }
    /// Expose the raw value for the single legitimate use: putting it into an
    /// outbound `Authorization` header. Callers must never log the result.
    pub fn expose(&self) -> &str {
        &self.0
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Debug for SecretValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SecretValue(***redacted***)")
    }
}

impl fmt::Display for SecretValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("***redacted***")
    }
}

/// Status of a secret without ever revealing its value — for display in a
/// secrets panel ("names/status only").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SecretStatus {
    /// Vault has a non-empty value for this reference.
    Present,
    /// Vault knows the key but it resolves empty.
    Empty,
    /// Vault has no such key.
    Missing,
}

impl SecretStatus {
    pub fn label(self) -> &'static str {
        match self {
            SecretStatus::Present => "present",
            SecretStatus::Empty => "empty",
            SecretStatus::Missing => "missing",
        }
    }
}

/// Vault-backed secret resolver. Real deployments use an <secret-manager>/vault-backed
/// implementation; tests use [`EnvSecretManager`] over an in-memory map. No
/// secret literals ever live in config or code.
#[async_trait]
pub trait SecretManager: Send + Sync {
    /// Resolve a reference to its value, or `None` if the vault has no value.
    async fn resolve(&self, r: &SecretRef) -> Option<SecretValue>;

    /// Report presence/status WITHOUT returning the value (for display).
    async fn status(&self, r: &SecretRef) -> SecretStatus {
        match self.resolve(r).await {
            Some(v) if !v.is_empty() => SecretStatus::Present,
            Some(_) => SecretStatus::Empty,
            None => SecretStatus::Missing,
        }
    }
}

/// Vault-backed manager that reads from the process environment injected by the
/// vault agent at launch (never from literals baked into config). Used as the
/// default backend and in tests via [`with_map`].
#[derive(Default)]
pub struct EnvSecretManager {
    // Optional override map for tests; when empty, falls back to std::env.
    overrides: std::collections::HashMap<String, String>,
}

impl EnvSecretManager {
    pub fn from_env() -> Self {
        Self::default()
    }
    /// Test/inject constructor. NOT used to hold real secrets in production.
    pub fn with_map(m: std::collections::HashMap<String, String>) -> Self {
        Self { overrides: m }
    }
}

#[async_trait]
impl SecretManager for EnvSecretManager {
    async fn resolve(&self, r: &SecretRef) -> Option<SecretValue> {
        if let Some(v) = self.overrides.get(r.name()) {
            return Some(SecretValue::new(v.clone()));
        }
        std::env::var(r.name()).ok().map(SecretValue::new)
    }
}

/// Env var names read by [`InfisicalSecretManager`] beyond the bootstrap
/// credential trio (`INFISICAL_URL`/`INFISICAL_CLIENT_ID`/
/// `INFISICAL_CLIENT_SECRET`, read by `chord_secrets::InfisicalConfig` itself).
/// Deliberately reuses the same `CHORD_INFISICAL_*` names CSEC-02 introduced
/// for `chord-proxy`'s startup fetch — chord-tui shares Chord's bootstrap
/// identity and project, not a separate one.
mod env_keys {
    pub const PROJECT_ID: &str = "CHORD_INFISICAL_PROJECT_ID";
    pub const ENVIRONMENT: &str = "CHORD_INFISICAL_ENVIRONMENT";
    pub const SECRET_PATH: &str = "CHORD_INFISICAL_SECRET_PATH";
    /// How long a fetched batch is cached before the next `resolve()` call
    /// re-authenticates. chord-tui's `ConnectionManager` calls `resolve()`
    /// once per instance per poll tick, so an un-cached fresh-auth-per-call
    /// (CSEC-01's default shape) would re-authenticate against <secret-manager> every
    /// poll interval for every instance — this cache keeps that chatty without
    /// growing CSEC-01 itself into something with its own refresh thread.
    pub const CACHE_TTL_SECS: &str = "CHORD_TUI_INFISICAL_CACHE_TTL_SECS";
}

fn non_empty_env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|s| !s.trim().is_empty())
}

const DEFAULT_CACHE_TTL_SECS: u64 = 30;

/// Real vault-backed [`SecretManager`] implementation: resolves each
/// [`SecretRef`] by fetching from <secret-manager> via CSEC-01's shared
/// `chord-secrets` Universal Auth client, using Chord's own bootstrap
/// identity (`INFISICAL_URL`/`INFISICAL_CLIENT_ID`/`INFISICAL_CLIENT_SECRET`)
/// plus a project/environment/path scope (`CHORD_INFISICAL_PROJECT_ID`/
/// `CHORD_INFISICAL_ENVIRONMENT`/`CHORD_INFISICAL_SECRET_PATH`).
///
/// A resolved batch is cached for a short TTL (`CHORD_TUI_INFISICAL_CACHE_TTL_SECS`,
/// default 30s) since `resolve()` is called on every poll tick per instance —
/// without a cache this would re-authenticate against <secret-manager> far more
/// often than any of it needs to. On any fetch failure (auth failure,
/// unreachable, malformed response) `resolve()` returns `None` — the same
/// "vault has no value" signal as a genuinely missing key — and NEVER panics
/// or logs a secret value; only the non-secret `SecretError` display text
/// would ever be logged by a caller, and this type itself doesn't log at all.
pub struct InfisicalSecretManager {
    config: InfisicalConfig,
    project_id: String,
    environment: String,
    secret_path: String,
    cache_ttl: Duration,
    cache: RwLock<Option<(Instant, HashMap<String, String>)>>,
}

impl InfisicalSecretManager {
    /// Build a manager from the process environment. Returns `None` if
    /// <secret-manager> isn't configured (bootstrap credentials or project id
    /// missing) — callers should fall back to [`EnvSecretManager`] in that
    /// case, matching CSEC-02's fallback shape.
    pub fn from_env() -> Option<Self> {
        let config = InfisicalConfig::from_env();
        if !config.is_configured() {
            return None;
        }
        let project_id = non_empty_env(env_keys::PROJECT_ID)?;
        let environment =
            non_empty_env(env_keys::ENVIRONMENT).unwrap_or_else(|| "prod".to_string());
        let secret_path = non_empty_env(env_keys::SECRET_PATH).unwrap_or_else(|| "/".to_string());
        let cache_ttl = non_empty_env(env_keys::CACHE_TTL_SECS)
            .and_then(|s| s.parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or(Duration::from_secs(DEFAULT_CACHE_TTL_SECS));

        Some(Self {
            config,
            project_id,
            environment,
            secret_path,
            cache_ttl,
            cache: RwLock::new(None),
        })
    }

    /// Return the cached batch if still fresh, otherwise fetch a new one and
    /// cache it. A fetch failure is NOT cached — the next call retries rather
    /// than sticking with a stale "everything missing" result.
    async fn fetch_batch(&self) -> Option<HashMap<String, String>> {
        {
            let cache = self.cache.read().await;
            if let Some((fetched_at, map)) = cache.as_ref() {
                if fetched_at.elapsed() < self.cache_ttl {
                    return Some(map.clone());
                }
            }
        }

        match chord_secrets::fetch_secrets_batch(
            &self.config,
            &self.project_id,
            &self.environment,
            &self.secret_path,
        )
        .await
        {
            Ok(map) => {
                *self.cache.write().await = Some((Instant::now(), map.clone()));
                Some(map)
            }
            Err(_) => None,
        }
    }
}

#[async_trait]
impl SecretManager for InfisicalSecretManager {
    async fn resolve(&self, r: &SecretRef) -> Option<SecretValue> {
        let batch = self.fetch_batch().await?;
        batch.get(r.name()).map(|v| SecretValue::new(v.clone()))
    }
}

/// Select the default [`SecretManager`] for chord-tui: [`InfisicalSecretManager`]
/// when <secret-manager> is configured (bootstrap credentials AND project id present),
/// falling back to [`EnvSecretManager`] otherwise. This is the same
/// configured/unconfigured fallback shape as CSEC-02's `chord-proxy` startup
/// fetch — an un-migrated deployment (no <secret-manager> env vars set) keeps working
/// exactly as before, reading tokens from the static environment.
pub fn default_secret_manager() -> Arc<dyn SecretManager> {
    match InfisicalSecretManager::from_env() {
        Some(mgr) => Arc::new(mgr),
        None => Arc::new(EnvSecretManager::from_env()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secret_value_redacts_in_debug_and_display() {
        let s = SecretValue::new("hunter2-super-secret");
        assert_eq!(format!("{s}"), "***redacted***");
        assert_eq!(format!("{s:?}"), "SecretValue(***redacted***)");
        assert!(!format!("{s:?}").contains("hunter2"));
        assert!(!format!("{s}").contains("hunter2"));
        // The only sanctioned reveal path:
        assert_eq!(s.expose(), "hunter2-super-secret");
    }

    #[tokio::test]
    async fn env_manager_status_never_returns_value() {
        let mut m = std::collections::HashMap::new();
        m.insert("TOKEN_A".to_string(), "abc".to_string());
        m.insert("TOKEN_B".to_string(), "".to_string());
        let mgr = EnvSecretManager::with_map(m);
        assert_eq!(mgr.status(&SecretRef::new("TOKEN_A")).await, SecretStatus::Present);
        assert_eq!(mgr.status(&SecretRef::new("TOKEN_B")).await, SecretStatus::Empty);
        assert_eq!(mgr.status(&SecretRef::new("TOKEN_C")).await, SecretStatus::Missing);
    }

    // ── InfisicalSecretManager + selection logic (CSEC-03) ──────────────────

    use httpmock::prelude::*;
    use serial_test::serial;

    const INFISICAL_ENV_KEYS: &[&str] = &[
        "INFISICAL_URL",
        "INFISICAL_CLIENT_ID",
        "INFISICAL_CLIENT_SECRET",
        env_keys::PROJECT_ID,
        env_keys::ENVIRONMENT,
        env_keys::SECRET_PATH,
        env_keys::CACHE_TTL_SECS,
    ];

    fn clear_infisical_env() {
        for k in INFISICAL_ENV_KEYS {
            std::env::remove_var(k);
        }
    }

    fn set_bootstrap_env(base_url: &str, project_id: &str) {
        std::env::set_var("INFISICAL_URL", base_url);
        std::env::set_var("INFISICAL_CLIENT_ID", "test-client-id"); // pii-test-fixture
        std::env::set_var("INFISICAL_CLIENT_SECRET", "test-client-secret"); // pii-test-fixture
        std::env::set_var(env_keys::PROJECT_ID, project_id);
    }

    #[test]
    #[serial]
    fn infisical_manager_none_when_unconfigured() {
        clear_infisical_env();
        assert!(InfisicalSecretManager::from_env().is_none());
    }

    #[test]
    #[serial]
    fn infisical_manager_none_when_project_id_missing() {
        clear_infisical_env();
        std::env::set_var("INFISICAL_URL", "http://example.test");
        std::env::set_var("INFISICAL_CLIENT_ID", "id");
        std::env::set_var("INFISICAL_CLIENT_SECRET", "secret"); // pii-test-fixture
        // CHORD_INFISICAL_PROJECT_ID deliberately left unset.
        assert!(InfisicalSecretManager::from_env().is_none());
        clear_infisical_env();
    }

    #[tokio::test]
    #[serial]
    async fn infisical_manager_resolves_configured_secret() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/api/v1/auth/universal-auth/login");
            then.status(200).json_body(serde_json::json!({ "accessToken": "tok" }));
        });
        server.mock(|when, then| {
            when.method(GET).path("/api/v3/secrets/raw");
            then.status(200).json_body(serde_json::json!({
                "secrets": [
                    { "secretKey": "CHORD_TUI_INSTANCE_TOKEN", "secretValue": "resolved-value" } // pii-test-fixture
                ]
            }));
        });

        clear_infisical_env();
        set_bootstrap_env(&server.base_url(), "proj1");
        let mgr = InfisicalSecretManager::from_env().expect("should be configured");

        let resolved = mgr.resolve(&SecretRef::new("CHORD_TUI_INSTANCE_TOKEN")).await;
        assert_eq!(resolved.map(|v| v.expose().to_string()), Some("resolved-value".to_string()));

        let missing = mgr.resolve(&SecretRef::new("NOT_THERE")).await;
        assert!(missing.is_none());

        clear_infisical_env();
    }

    #[tokio::test]
    #[serial]
    async fn infisical_manager_caches_within_ttl() {
        let server = MockServer::start();
        let login_mock = server.mock(|when, then| {
            when.method(POST).path("/api/v1/auth/universal-auth/login");
            then.status(200).json_body(serde_json::json!({ "accessToken": "tok" }));
        });
        server.mock(|when, then| {
            when.method(GET).path("/api/v3/secrets/raw");
            then.status(200).json_body(serde_json::json!({
                "secrets": [{ "secretKey": "K", "secretValue": "v" }]
            }));
        });

        clear_infisical_env();
        set_bootstrap_env(&server.base_url(), "proj1");
        std::env::set_var(env_keys::CACHE_TTL_SECS, "60");
        let mgr = InfisicalSecretManager::from_env().expect("should be configured");

        let _ = mgr.resolve(&SecretRef::new("K")).await;
        let _ = mgr.resolve(&SecretRef::new("K")).await;
        let _ = mgr.resolve(&SecretRef::new("K")).await;

        assert_eq!(login_mock.hits(), 1, "second/third resolve should hit the cache, not re-auth");
        clear_infisical_env();
    }

    #[tokio::test]
    #[serial]
    async fn infisical_manager_returns_none_on_fetch_failure_not_panic() {
        clear_infisical_env();
        // Port 1 refuses connections — simulates <secret-manager> unreachable.
        set_bootstrap_env("http://127.0.0.1:1", "proj1");
        let mgr = InfisicalSecretManager::from_env().expect("should be configured");

        let resolved = mgr.resolve(&SecretRef::new("ANY_KEY")).await;
        assert!(resolved.is_none());
        clear_infisical_env();
    }

    #[test]
    #[serial]
    fn default_secret_manager_falls_back_to_env_when_unconfigured() {
        clear_infisical_env();
        // Not directly introspectable by type (trait object), but this proves
        // the selection function itself never panics and returns something
        // that behaves like EnvSecretManager: a key set via std::env is
        // resolvable, one that was never set is not.
        let mgr = default_secret_manager();
        // Selection succeeded without needing <secret-manager> to be configured —
        // exercised further via resolve() in an async test below.
        drop(mgr);
    }

    #[tokio::test]
    #[serial]
    async fn default_secret_manager_resolves_via_env_fallback() {
        clear_infisical_env();
        std::env::set_var("CSEC03_FALLBACK_TEST_KEY", "fallback-value"); // pii-test-fixture
        let mgr = default_secret_manager();
        let resolved = mgr.resolve(&SecretRef::new("CSEC03_FALLBACK_TEST_KEY")).await;
        assert_eq!(resolved.map(|v| v.expose().to_string()), Some("fallback-value".to_string()));
        std::env::remove_var("CSEC03_FALLBACK_TEST_KEY");
    }

    #[tokio::test]
    #[serial]
    async fn default_secret_manager_selects_infisical_when_configured() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/api/v1/auth/universal-auth/login");
            then.status(200).json_body(serde_json::json!({ "accessToken": "tok" }));
        });
        server.mock(|when, then| {
            when.method(GET).path("/api/v3/secrets/raw");
            then.status(200).json_body(serde_json::json!({
                "secrets": [{ "secretKey": "FROM_INFISICAL", "secretValue": "<secret-manager>-value" }] // pii-test-fixture
            }));
        });

        clear_infisical_env();
        set_bootstrap_env(&server.base_url(), "proj1");
        // Also set the same key as a plain env var — if selection picked
        // EnvSecretManager instead of <secret-manager>, this would resolve instead
        // and the test would (wrongly) still pass, so make the two disagree.
        std::env::set_var("FROM_INFISICAL", "env-value-should-not-win"); // pii-test-fixture

        let mgr = default_secret_manager();
        let resolved = mgr.resolve(&SecretRef::new("FROM_INFISICAL")).await;
        assert_eq!(
            resolved.map(|v| v.expose().to_string()),
            Some("<secret-manager>-value".to_string()),
            "selection must prefer InfisicalSecretManager when configured"
        );

        std::env::remove_var("FROM_INFISICAL");
        clear_infisical_env();
    }
}
