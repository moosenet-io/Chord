//! Launch-environment scrubbing — S88 ISO-01.
//!
//! Builds the environment a runtime child is spawned with, starting from a
//! **minimal** base (NOT the supervisor's full inherited environment) and
//! injecting only what a runtime needs plus a set of telemetry-off / offline
//! opt-outs. It also STRIPS HTTP/HTTPS/ALL proxy variables unless Chord is
//! explicitly configured with a proxy.
//!
//! ## Honest scope (ADVISORY)
//! This is env-scrubbing, not network isolation. It relies on the runtimes
//! HONOURING the documented opt-out variables (ollama/HF/etc.). A binary that
//! ignores them is not stopped by this layer. The KERNEL guarantee — a network
//! namespace that physically blocks egress — is **ISO-02 and is not yet built**.
//! Treat the vars set here as defense-in-depth, not a boundary.
//!
//! ## Integration
//! [`super::egress_policy::posture_for`] decides the posture; this module turns a
//! launch into a scrubbed env. The serving launcher
//! ([`crate::serving::launcher::build_launch_command`]) merges these pairs into
//! the `LaunchCommand::env` it hands the spawner — additively, so an existing
//! launch's behaviour is preserved aside from the new telemetry-off vars.

use crate::config::Config;

use super::egress_policy::{posture_for, EgressPosture, RuntimeClass};

/// Proxy env var names stripped from a runtime launch unless Chord has an explicit
/// proxy configured. Both upper- and lower-case forms are honoured by HTTP stacks,
/// so both are stripped.
const PROXY_VARS: &[&str] = &[
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
];

/// Build the scrubbed environment for spawning a runtime of `class`.
///
/// Starts from a MINIMAL env (only the handful of vars a process needs to run —
/// `PATH`, `HOME`, `LANG`, etc. when present in the supervisor env), then injects:
///   * the telemetry-off / offline opt-outs (always), and
///   * the proxy vars ONLY when `cfg` carries an explicit Chord proxy
///     ([`Config::outbound_proxy`]); otherwise proxy vars are NOT present (they are
///     never inherited).
///
/// The returned pairs are the COMPLETE env the child should run with — the caller
/// spawns with a cleared environment plus these pairs (or merges them over a
/// minimal base), so no ambient telemetry/proxy var leaks in.
///
/// Posture: [`posture_for`] is consulted for `class`. In ISO-01 the posture only
/// influences telemetry/offline framing (it is advisory); ISO-02 will use it to
/// build the network namespace. We compute it here so the seam exists and the
/// posture is observable in logs.
pub fn build_runtime_env(class: RuntimeClass, base: &Config) -> Vec<(String, String)> {
    let posture = posture_for(class, base);

    let mut env: Vec<(String, String)> = Vec::new();

    // (1) Minimal passthrough: carry only the few vars a process legitimately needs
    // to execute. We do NOT inherit the full environment (that is the scrub).
    for key in MINIMAL_PASSTHROUGH {
        if let Ok(val) = std::env::var(key) {
            if !val.is_empty() {
                env.push(((*key).to_string(), val));
            }
        }
    }

    // (2) Telemetry-off / offline opt-outs — always set (defense in depth).
    for (k, v) in TELEMETRY_OFF_VARS {
        env.push(((*k).to_string(), (*v).to_string()));
    }

    // (3) Proxy: stripped by default (never inherited). Only when Chord is
    // explicitly configured with an outbound proxy do we re-introduce it — and
    // only for a Pull (a Serve has no egress, so a proxy is meaningless there).
    debug_assert!(
        !env.iter().any(|(k, _)| PROXY_VARS.contains(&k.as_str())),
        "proxy vars must never be carried by the minimal passthrough"
    );
    if let Some(proxy) = base.outbound_proxy.as_deref() {
        if matches!(class, RuntimeClass::Pull) && !matches!(posture, EgressPosture::Denied) {
            // Set both HTTPS_PROXY and HTTP_PROXY to the configured value so the
            // runtime's HTTP stack uses Chord's proxy for the allow-listed pull.
            env.push(("HTTPS_PROXY".to_string(), proxy.to_string()));
            env.push(("HTTP_PROXY".to_string(), proxy.to_string()));
        }
    }

    tracing::debug!(
        target: "chord.supervisor",
        class = ?class,
        posture = ?posture,
        proxy_configured = base.outbound_proxy.is_some(),
        "built scrubbed runtime launch env (ISO-01, advisory)"
    );

    env
}

/// The minimal set of variables a spawned runtime legitimately needs to execute.
/// Anything outside this list is NOT inherited from the supervisor (the scrub).
/// Deliberately small: process basics + locale, no proxy, no secrets, no inherited
/// telemetry toggles (we set those ourselves below).
const MINIMAL_PASSTHROUGH: &[&str] = &["PATH", "HOME", "LANG", "LC_ALL", "TZ", "TMPDIR"];

/// Telemetry-off / offline opt-out variables set on EVERY runtime launch.
///
/// These are the documented public opt-outs:
///   * `DO_NOT_TRACK=1` — the cross-tool do-not-track convention.
///   * Ollama: disable analytics + update checks + model pruning side-effects.
///   * HuggingFace / Transformers: force fully offline (no hub calls).
const TELEMETRY_OFF_VARS: &[(&str, &str)] = &[
    // Cross-tool do-not-track convention.
    ("DO_NOT_TRACK", "1"),
    // Ollama opt-outs (analytics + update check off; no auto-prune side effects).
    ("OLLAMA_NO_ANALYTICS", "1"),
    ("OLLAMA_NOPRUNE", "1"),
    ("OLLAMA_NO_UPDATE_CHECK", "1"),
    // HuggingFace Hub / Transformers — fully offline (never reach the hub at serve).
    ("HF_HUB_OFFLINE", "1"),
    ("TRANSFORMERS_OFFLINE", "1"),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use serial_test::serial;

    fn has(env: &[(String, String)], key: &str) -> bool {
        env.iter().any(|(k, _)| k == key)
    }
    fn val<'a>(env: &'a [(String, String)], key: &str) -> Option<&'a str> {
        env.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    #[test]
    #[serial]
    fn env_builder_sets_all_telemetry_off_vars() {
        let cfg = Config::test_default();
        let env = build_runtime_env(RuntimeClass::Serve, &cfg);
        assert_eq!(val(&env, "DO_NOT_TRACK"), Some("1"));
        assert_eq!(val(&env, "OLLAMA_NO_ANALYTICS"), Some("1"));
        assert_eq!(val(&env, "OLLAMA_NOPRUNE"), Some("1"));
        assert_eq!(val(&env, "OLLAMA_NO_UPDATE_CHECK"), Some("1"));
        assert_eq!(val(&env, "HF_HUB_OFFLINE"), Some("1"));
        assert_eq!(val(&env, "TRANSFORMERS_OFFLINE"), Some("1"));
    }

    #[test]
    #[serial]
    fn proxy_vars_stripped_when_no_chord_proxy_configured() {
        // Even if the supervisor's own environment carries a proxy, the scrubbed
        // child env must NOT (we start minimal, never inherit it).
        std::env::set_var("HTTP_PROXY", "http://evil.invalid:3128");
        std::env::set_var("https_proxy", "http://evil.invalid:3128");
        let mut cfg = Config::test_default();
        cfg.outbound_proxy = None;
        let env = build_runtime_env(RuntimeClass::Pull, &cfg);
        for p in PROXY_VARS {
            assert!(!has(&env, p), "proxy var {p} must be stripped");
        }
        std::env::remove_var("HTTP_PROXY");
        std::env::remove_var("https_proxy");
    }

    #[test]
    #[serial]
    fn proxy_set_only_when_configured_and_pull_allowed() {
        let mut cfg = Config::test_default();
        cfg.outbound_proxy = Some("http://chord-proxy.local:3128".to_string());
        cfg.model_source_allowlist = vec!["registry.ollama.ai".to_string()];
        // Pull with an allow-list → proxy is applied.
        let env = build_runtime_env(RuntimeClass::Pull, &cfg);
        assert_eq!(val(&env, "HTTPS_PROXY"), Some("http://chord-proxy.local:3128"));
        assert_eq!(val(&env, "HTTP_PROXY"), Some("http://chord-proxy.local:3128"));
        // Serve is always Denied → no proxy even when configured.
        let env_serve = build_runtime_env(RuntimeClass::Serve, &cfg);
        assert!(!has(&env_serve, "HTTPS_PROXY"));
        assert!(!has(&env_serve, "HTTP_PROXY"));
    }

    #[test]
    #[serial]
    fn proxy_not_set_for_pull_when_allow_list_empty_even_if_proxy_configured() {
        // Allow-list unset → Pull posture is Denied → no egress → no proxy applied,
        // even though a proxy is configured.
        let mut cfg = Config::test_default();
        cfg.outbound_proxy = Some("http://chord-proxy.local:3128".to_string());
        cfg.model_source_allowlist = vec![];
        let env = build_runtime_env(RuntimeClass::Pull, &cfg);
        assert!(!has(&env, "HTTPS_PROXY"));
        assert!(!has(&env, "HTTP_PROXY"));
    }

    #[test]
    #[serial]
    fn does_not_inherit_arbitrary_supervisor_vars() {
        // A random secret-ish var in the supervisor env must not leak into the
        // scrubbed child env.
        std::env::set_var("SOME_SECRET_TOKEN", "super-secret");
        let cfg = Config::test_default();
        let env = build_runtime_env(RuntimeClass::Serve, &cfg);
        assert!(!has(&env, "SOME_SECRET_TOKEN"), "minimal env must not inherit arbitrary vars");
        std::env::remove_var("SOME_SECRET_TOKEN");
    }
}
