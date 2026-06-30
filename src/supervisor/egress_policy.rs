//! Egress policy — the CONFIG SURFACE for S88 ISO-01.
//!
//! ISO-01 declares *what* network posture a runtime launch SHOULD have; it does
//! NOT enforce it at the kernel. Enforcement (a network namespace that physically
//! blocks egress) is ISO-02 and is **not built yet**. Until then this module is
//! ADVISORY: it produces a posture that the launch-env scrubbing
//! ([`super::launch_env`]) reflects via telemetry/offline opt-outs and proxy
//! stripping, relying on the runtimes honouring those opt-outs.
//!
//! ## What is config, what is hardcoded
//! The two postures and the two runtime classes are fixed *policy* (Serve never
//! needs the network; Pull does). The **allow-list of model sources is CONFIG** —
//! it comes from [`Config::model_source_allowlist`](crate::config) (the
//! `MODEL_SOURCE_ALLOWLIST` env var), never a baked-in host. An empty/unset
//! allow-list means Pull is [`EgressPosture::Denied`]: we FAIL CLOSED, never
//! default to allow-all.

use crate::config::Config;

/// The network posture a runtime launch should have.
///
/// ADVISORY in ISO-01 (relies on tool opt-outs); becomes a kernel guarantee under
/// ISO-02's network namespace (not yet built).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EgressPosture {
    /// No egress at all. A serving runtime answers local inference requests; it
    /// never needs to reach the internet, so the posture is full denial.
    Denied,
    /// Egress permitted ONLY to the listed model-source hosts/domains. The list is
    /// config (`MODEL_SOURCE_ALLOWLIST`); an empty list never appears here —
    /// [`posture_for`] collapses an empty allow-list to [`EgressPosture::Denied`]
    /// (fail closed).
    AllowList(Vec<String>),
}

/// What a runtime is being launched to DO — which decides its egress posture.
///
/// Distinct from `terminus_rs::intake::serving::Runtime` (which backend binary):
/// `RuntimeClass` is about the *operation*, not the tier. A `Pull` may use any
/// runtime/binary; a `Serve` likewise. Only the operation decides egress.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeClass {
    /// Serving inference for a model that is already present locally. Needs no
    /// egress → [`EgressPosture::Denied`].
    Serve,
    /// Pulling/acquiring model weights from a model source. Needs egress, but ONLY
    /// to the configured allow-list → [`EgressPosture::AllowList`] (or `Denied`
    /// when the allow-list is unset).
    Pull,
}

/// Decide the egress posture for a runtime `class` given the current `cfg`.
///
/// * `Serve` → always [`EgressPosture::Denied`] (a server never needs the net).
/// * `Pull`  → [`EgressPosture::AllowList`] of the configured model sources, OR
///   [`EgressPosture::Denied`] when no sources are configured (FAIL CLOSED — we
///   never default a pull to allow-all).
pub fn posture_for(class: RuntimeClass, cfg: &Config) -> EgressPosture {
    match class {
        RuntimeClass::Serve => EgressPosture::Denied,
        RuntimeClass::Pull => {
            let allow = cfg.model_source_allowlist.clone();
            if allow.is_empty() {
                // Fail closed: an unconfigured allow-list denies the pull rather
                // than silently permitting all egress.
                EgressPosture::Denied
            } else {
                EgressPosture::AllowList(allow)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    /// A minimal Config with an explicit allow-list, avoiding env reads in tests.
    fn cfg_with_allow(allow: Vec<&str>) -> Config {
        let mut c = Config::test_default();
        c.model_source_allowlist = allow.into_iter().map(String::from).collect();
        c
    }

    #[test]
    fn serve_is_always_denied() {
        let cfg = cfg_with_allow(vec!["registry.ollama.ai", "huggingface.co"]);
        assert_eq!(posture_for(RuntimeClass::Serve, &cfg), EgressPosture::Denied);
    }

    #[test]
    fn pull_uses_configured_allow_list() {
        let cfg = cfg_with_allow(vec!["registry.ollama.ai", "huggingface.co"]);
        assert_eq!(
            posture_for(RuntimeClass::Pull, &cfg),
            EgressPosture::AllowList(vec![
                "registry.ollama.ai".to_string(),
                "huggingface.co".to_string(),
            ])
        );
    }

    #[test]
    fn pull_with_unset_allow_list_is_denied_not_allow_all() {
        // The negative test: an empty allow-list must FAIL CLOSED, never produce
        // an allow-all (or even a non-empty) posture.
        let cfg = cfg_with_allow(vec![]);
        let posture = posture_for(RuntimeClass::Pull, &cfg);
        assert_eq!(posture, EgressPosture::Denied);
        assert!(
            !matches!(posture, EgressPosture::AllowList(_)),
            "unset allow-list must never become an AllowList (no allow-all default)"
        );
    }
}
