//! Startup-time <secret-manager> secret fetch for Chord (CSEC-02).
//!
//! Before `Config::from_env()` reads `CHORD_JWT_SECRET` (`config.rs`) or
//! `HarnessVramManager::from_env()` reads `CHORD_API_KEY`
//! (`harness/vram_lifecycle.rs`), `main()` calls
//! [`fetch_and_apply_downstream_secrets`], which — when
//! `INFISICAL_URL`/`INFISICAL_CLIENT_ID`/`INFISICAL_CLIENT_SECRET` (Chord's own
//! bootstrap identity, per the standing "Chord authenticates to <secret-manager>
//! directly" decision) plus `CHORD_INFISICAL_PROJECT_ID` are configured —
//! fetches `CHORD_JWT_SECRET`/`CHORD_API_KEY` fresh from <secret-manager> and sets them
//! into the process environment via `std::env::set_var`. Because this happens
//! once, early in `main()`, before either `from_env()` call runs, downstream
//! code needs NO changes: both continue to do a plain `std::env::var(...)` read
//! and transparently see whichever value ended up in the environment.
//!
//! This is the same shape already proven this session for `terminus_personal`
//! (PSEC-02): reuses [`chord_secrets`], Chord's own <secret-manager> Universal Auth
//! client (CSEC-01) — NOT brokered through `terminus_personal` or any other
//! fleet service, per the operator's standing decision that Chord authenticates
//! to <secret-manager> with its own bootstrap identity (several internal hops aren't
//! TLS-terminated).
//!
//! Falls back cleanly (never panics, never hangs, never hard-fails startup) to
//! whatever is already in the process environment (e.g. a static `.env` loaded
//! by the systemd unit's `EnvironmentFile=`) when <secret-manager> isn't configured or
//! the fetch fails for any reason (auth failure, network error, malformed
//! response). No secret value is ever logged; only counts and (for missing
//! keys) key NAMES are logged.
//!
//! Additional env vars for this fetch (all optional; the fetch is skipped —
//! falling back to the static environment — unless the bootstrap credential
//! AND the project id are both present):
//! - `CHORD_INFISICAL_PROJECT_ID` — the <secret-manager> workspace/project ID to fetch
//!   from. No default (deliberately not hardcoded — see S1).
//! - `CHORD_INFISICAL_ENVIRONMENT` — <secret-manager> environment slug. Defaults to
//!   `prod`.
//! - `CHORD_INFISICAL_SECRET_PATH` — folder path within the environment.
//!   Defaults to `/`.

use chord_secrets::{fetch_secrets_batch, InfisicalConfig};

/// The downstream secret keys this process needs, fetched from <secret-manager> at
/// startup rather than relying on a static `.env`. Deliberately a fixed, named
/// allowlist (not "set every key found at this path") so a shared <secret-manager>
/// path containing secrets for other services never leaks into this process's
/// environment.
/// EMBED-01: `OPENROUTER_API_KEY` — the bearer key `src/embeddings.rs`'s
/// OpenRouter fallback path sends as `Authorization: Bearer <key>`. Fetched
/// here (never a literal, never logged) exactly like `CHORD_JWT_SECRET`/
/// `CHORD_API_KEY`; `embeddings::openrouter_api_key()` reads it back from the
/// process environment fresh on every request.
const DOWNSTREAM_SECRET_KEYS: &[&str] =
    &["CHORD_JWT_SECRET", "CHORD_API_KEY", "OPENROUTER_API_KEY"];

/// Outcome of the startup <secret-manager> fetch attempt, for the caller (`main()`)
/// to log and for tests to assert on directly rather than scraping log text.
#[derive(Debug, PartialEq, Eq)]
pub enum SecretFetchOutcome {
    /// `INFISICAL_URL`/`INFISICAL_CLIENT_ID`/`INFISICAL_CLIENT_SECRET` or the
    /// project-id env var aren't configured — nothing was attempted.
    NotConfigured,
    /// The fetch succeeded; `count` downstream keys were found and set into the
    /// process environment. `missing` names (never values) any of
    /// `DOWNSTREAM_SECRET_KEYS` that <secret-manager> didn't have at this path.
    Fetched { count: usize, missing: Vec<String> },
    /// The fetch was attempted but failed (auth failure, network error,
    /// malformed response, ...) — callers fall back to whatever is already in
    /// the process environment. `reason` is a display-formatted `SecretError`
    /// — never a secret value.
    Failed { reason: String },
}

/// Attempt to fetch `CHORD_JWT_SECRET`/`CHORD_API_KEY` fresh from <secret-manager> and
/// set them into the process environment, so `Config::from_env()` and
/// `HarnessVramManager::from_env()` (called later, at any point) see the
/// current value. Falls back cleanly — never panics, never hangs, never
/// hard-fails startup — when <secret-manager> isn't configured or the fetch fails.
///
/// Never logs or echoes any fetched secret value — only counts and, for
/// missing keys, key NAMES (never values).
pub async fn fetch_and_apply_downstream_secrets() -> SecretFetchOutcome {
    let config = InfisicalConfig::from_env();
    if !config.is_configured() {
        return SecretFetchOutcome::NotConfigured;
    }

    let project_id = match std::env::var("CHORD_INFISICAL_PROJECT_ID")
        .ok()
        .filter(|s| !s.is_empty())
    {
        Some(p) => p,
        None => return SecretFetchOutcome::NotConfigured,
    };
    let environment = std::env::var("CHORD_INFISICAL_ENVIRONMENT")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "prod".to_string());
    let secret_path = std::env::var("CHORD_INFISICAL_SECRET_PATH")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "/".to_string());

    match fetch_secrets_batch(&config, &project_id, &environment, &secret_path).await {
        Ok(fetched) => {
            let mut count = 0usize;
            let mut missing = Vec::new();
            for key in DOWNSTREAM_SECRET_KEYS {
                match fetched.get(*key) {
                    Some(value) => {
                        std::env::set_var(key, value);
                        count += 1;
                    }
                    None => missing.push((*key).to_string()),
                }
            }
            SecretFetchOutcome::Fetched { count, missing }
        }
        Err(e) => SecretFetchOutcome::Failed {
            reason: e.to_string(),
        },
    }
}

/// Log the outcome of the startup <secret-manager> fetch. Split out from
/// `fetch_and_apply_downstream_secrets()` so tests can assert on the returned
/// enum directly without needing to capture tracing output.
pub fn log_secret_fetch_outcome(outcome: &SecretFetchOutcome) {
    match outcome {
        SecretFetchOutcome::NotConfigured => {
            tracing::info!(
                "chord-proxy: <secret-manager> not configured (INFISICAL_URL/INFISICAL_CLIENT_ID/\
INFISICAL_CLIENT_SECRET/CHORD_INFISICAL_PROJECT_ID unset), using static environment"
            );
        }
        SecretFetchOutcome::Fetched { count, missing } => {
            tracing::info!("chord-proxy: fetched {count} secret(s) from <secret-manager>");
            if !missing.is_empty() {
                tracing::warn!(
                    "chord-proxy: <secret-manager> fetch did not include: {} (using static environment for these, if present)",
                    missing.join(", ")
                );
            }
        }
        SecretFetchOutcome::Failed { reason } => {
            tracing::warn!(
                "chord-proxy: <secret-manager> fetch failed ({reason}), falling back to static environment"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::prelude::*;
    use serial_test::serial;

    const ALL_ENV_KEYS: &[&str] = &[
        "INFISICAL_URL",
        "INFISICAL_CLIENT_ID",
        "INFISICAL_CLIENT_SECRET",
        "CHORD_INFISICAL_PROJECT_ID",
        "CHORD_INFISICAL_ENVIRONMENT",
        "CHORD_INFISICAL_SECRET_PATH",
        "CHORD_JWT_SECRET",
        "CHORD_API_KEY",
        "OPENROUTER_API_KEY",
    ];

    fn clear_all() {
        for k in ALL_ENV_KEYS {
            std::env::remove_var(k);
        }
    }

    #[tokio::test]
    #[serial]
    async fn not_configured_when_infisical_env_unset() {
        clear_all();
        let outcome = fetch_and_apply_downstream_secrets().await;
        assert_eq!(outcome, SecretFetchOutcome::NotConfigured);
        clear_all();
    }

    #[tokio::test]
    #[serial]
    async fn not_configured_when_project_id_unset() {
        clear_all();
        std::env::set_var("INFISICAL_URL", "http://127.0.0.1:1");
        std::env::set_var("INFISICAL_CLIENT_ID", "cid"); // pii-test-fixture
        std::env::set_var("INFISICAL_CLIENT_SECRET", "csecret"); // pii-test-fixture
        // CHORD_INFISICAL_PROJECT_ID deliberately left unset
        let outcome = fetch_and_apply_downstream_secrets().await;
        assert_eq!(outcome, SecretFetchOutcome::NotConfigured);
        clear_all();
    }

    #[tokio::test]
    #[serial]
    async fn fetch_sets_env_vars_before_downstream_reads_them() {
        clear_all();
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/api/v1/auth/universal-auth/login");
            then.status(200).json_body(serde_json::json!({ "accessToken": "tok" }));
        });
        server.mock(|when, then| {
            when.method(GET).path("/api/v3/secrets/raw");
            then.status(200).json_body(serde_json::json!({
                "secrets": [
                    { "secretKey": "CHORD_JWT_SECRET", "secretValue": "fetched-jwt" }, // pii-test-fixture
                    { "secretKey": "CHORD_API_KEY", "secretValue": "fetched-key" }, // pii-test-fixture
                    { "secretKey": "OPENROUTER_API_KEY", "secretValue": "fetched-openrouter-key" } // pii-test-fixture
                ]
            }));
        });

        std::env::set_var("INFISICAL_URL", server.base_url());
        std::env::set_var("INFISICAL_CLIENT_ID", "cid"); // pii-test-fixture
        std::env::set_var("INFISICAL_CLIENT_SECRET", "csecret"); // pii-test-fixture
        std::env::set_var("CHORD_INFISICAL_PROJECT_ID", "proj1");

        let outcome = fetch_and_apply_downstream_secrets().await;
        assert_eq!(
            outcome,
            SecretFetchOutcome::Fetched { count: 3, missing: vec![] }
        );

        // Simulate the downstream `from_env()` reads that happen later in main().
        assert_eq!(std::env::var("CHORD_JWT_SECRET").unwrap(), "fetched-jwt");
        assert_eq!(std::env::var("CHORD_API_KEY").unwrap(), "fetched-key");
        assert_eq!(
            std::env::var("OPENROUTER_API_KEY").unwrap(),
            "fetched-openrouter-key"
        );
        clear_all();
    }

    #[tokio::test]
    #[serial]
    async fn missing_keys_are_named_not_set_and_static_env_is_untouched() {
        clear_all();
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/api/v1/auth/universal-auth/login");
            then.status(200).json_body(serde_json::json!({ "accessToken": "tok" }));
        });
        server.mock(|when, then| {
            when.method(GET).path("/api/v3/secrets/raw");
            // <secret-manager> has neither downstream key at this path.
            then.status(200).json_body(serde_json::json!({ "secrets": [] }));
        });

        std::env::set_var("INFISICAL_URL", server.base_url());
        std::env::set_var("INFISICAL_CLIENT_ID", "cid"); // pii-test-fixture
        std::env::set_var("INFISICAL_CLIENT_SECRET", "csecret"); // pii-test-fixture
        std::env::set_var("CHORD_INFISICAL_PROJECT_ID", "proj1");
        // A pre-existing static value (e.g. from a systemd EnvironmentFile=) —
        // must survive untouched since <secret-manager> didn't have this key.
        std::env::set_var("CHORD_API_KEY", "static-fallback-key"); // pii-test-fixture

        let outcome = fetch_and_apply_downstream_secrets().await;
        match &outcome {
            SecretFetchOutcome::Fetched { count, missing } => {
                assert_eq!(*count, 0);
                assert!(missing.contains(&"CHORD_JWT_SECRET".to_string()));
                assert!(missing.contains(&"CHORD_API_KEY".to_string()));
                assert!(missing.contains(&"OPENROUTER_API_KEY".to_string()));
            }
            other => panic!("expected Fetched outcome, got {other:?}"),
        }
        // Untouched static value still present.
        assert_eq!(std::env::var("CHORD_API_KEY").unwrap(), "static-fallback-key");
        clear_all();
    }

    #[tokio::test]
    #[serial]
    async fn fetch_failure_falls_back_cleanly_without_panicking() {
        clear_all();
        std::env::set_var("INFISICAL_URL", "http://127.0.0.1:1"); // refuses connections
        std::env::set_var("INFISICAL_CLIENT_ID", "cid"); // pii-test-fixture
        std::env::set_var("INFISICAL_CLIENT_SECRET", "csecret"); // pii-test-fixture
        std::env::set_var("CHORD_INFISICAL_PROJECT_ID", "proj1");
        std::env::set_var("CHORD_JWT_SECRET", "static-jwt"); // pii-test-fixture

        let outcome = fetch_and_apply_downstream_secrets().await;
        assert!(matches!(outcome, SecretFetchOutcome::Failed { .. }));
        // Static value must be left completely alone on failure.
        assert_eq!(std::env::var("CHORD_JWT_SECRET").unwrap(), "static-jwt");
        clear_all();
    }

    #[tokio::test]
    #[serial]
    async fn log_outcome_never_panics_for_any_variant() {
        log_secret_fetch_outcome(&SecretFetchOutcome::NotConfigured);
        log_secret_fetch_outcome(&SecretFetchOutcome::Fetched {
            count: 2,
            missing: vec!["CHORD_API_KEY".to_string()],
        });
        log_secret_fetch_outcome(&SecretFetchOutcome::Failed {
            reason: "HTTP 401".to_string(),
        });
    }

    #[tokio::test]
    #[serial]
    async fn failure_reason_never_contains_secret_value() {
        clear_all();
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/api/v1/auth/universal-auth/login");
            then.status(401).body("nope");
        });

        std::env::set_var("INFISICAL_URL", server.base_url());
        std::env::set_var("INFISICAL_CLIENT_ID", "cid"); // pii-test-fixture
        std::env::set_var("INFISICAL_CLIENT_SECRET", "super-secret-bootstrap-value"); // pii-test-fixture
        std::env::set_var("CHORD_INFISICAL_PROJECT_ID", "proj1");

        let outcome = fetch_and_apply_downstream_secrets().await;
        if let SecretFetchOutcome::Failed { reason } = &outcome {
            assert!(!reason.contains("super-secret-bootstrap-value"));
        } else {
            panic!("expected Failed outcome, got {outcome:?}");
        }
        clear_all();
    }
}
