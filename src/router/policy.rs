//! DOCGEN-03: explicit routing policy for the Chord SLM router.
//!
//! This is pure decision logic — thresholds and allow/deny rules — with no
//! network I/O. [`RoutingPolicy::decide`] maps a [`RoutingRequest`] to a
//! [`RoutingDestination`] + a human-readable reason; [`RoutingPolicy::fallback_for`]
//! defines the degrade-gracefully chain the router walks when a destination is
//! unavailable; [`RoutingPolicy::is_cloud_egress_allowed`] is the ISO egress
//! gate consulted before any cloud network call is attempted.
//!
//! No secret values live here — only model NAMES and an egress allow-list of
//! HOSTNAMES (not tokens). The actual `OPENROUTER_API_KEY`-shaped credential is
//! resolved in [`super::slm_router`] via the existing `Backend::api_key_env`
//! indirection (`src/models/backends.rs`, `src/models/routing.rs`) — the same
//! pattern already used for the "openrouter" backend elsewhere in this crate:
//! the ENV VAR NAME is config, the value is read fresh at call time via a
//! variable — never a literal env-var lookup naming the secret directly in
//! source.

/// Where a generation request can be routed. Recomputed fresh per request —
/// never cached across requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingDestination {
    /// A local model with a large context window — used when the request's
    /// estimated token count exceeds the cheap-model threshold but still fits
    /// within local high-context capacity.
    LocalHighContext,
    /// A local, cheap/fast model — the default for simple requests.
    LocalCheap,
    /// OpenRouter's frontier-free tier — used when a request exceeds even
    /// local high-context capacity (so it is never silently truncated), or as
    /// an explicit policy escalation.
    CloudFrontierFree,
}

/// A generation request to be routed. `estimated_tokens` is the caller's
/// (the doc engine's) best estimate of the swept input's size — this router
/// does not tokenize itself, it trusts the caller's estimate per its own
/// APPROACH (destination decision + execute, not content analysis).
#[derive(Debug, Clone)]
pub struct RoutingRequest {
    pub prompt: String,
    pub estimated_tokens: usize,
}

/// Explicit, config-driven routing policy. Every threshold and model
/// reference is read from the environment (via [`RoutingPolicy::from_env`])
/// or supplied directly in tests — nothing is a compiled-in infrastructure
/// value beyond the documented default-fallback strings (same convention as
/// `AgenticModelRouter`/`ComplexityHeuristic` in `agentic/model_router.rs`).
#[derive(Debug, Clone)]
pub struct RoutingPolicy {
    /// Requests with an estimated token count at or below this go to the
    /// cheap local model. Above it, they need the high-context local model
    /// (or cloud, if they exceed `local_high_ctx_max_tokens` too).
    pub context_threshold_tokens: usize,
    /// The ceiling of what the local high-context model can serve. A request
    /// larger than this is routed to the cloud frontier-free model (if
    /// allowed) rather than truncated — never routed back down to
    /// `LocalHighContext` and silently truncated.
    pub local_high_ctx_max_tokens: usize,
    /// Local high-context model name/tag.
    pub local_high_ctx_model: String,
    /// Local cheap/fast model name/tag.
    pub local_cheap_model: String,
    /// OpenRouter frontier-free model id.
    pub cloud_frontier_model: String,
    /// Master on/off switch for cloud routing, independent of the allow-list.
    /// `false` means cloud is never used, even as a fallback.
    pub allow_cloud: bool,
    /// ISO egress allow-list of hostnames cloud calls may reach. Mirrors the
    /// fail-closed posture of `supervisor::egress_policy::posture_for` — an
    /// EMPTY allow-list denies all cloud egress, it never means "allow all".
    pub cloud_egress_allowlist: Vec<String>,
}

const DEFAULT_LOCAL_HIGH_CTX_MODEL: &str = "qwen2.5:120b";
const DEFAULT_LOCAL_CHEAP_MODEL: &str = "qwen2.5:20b";
const DEFAULT_CLOUD_EGRESS_HOST: &str = "openrouter.ai";

impl RoutingPolicy {
    /// Load policy from the environment. Every var has a safe, documented
    /// default (see field docs); nothing panics on a missing/malformed var.
    ///
    /// Env vars:
    ///   `SLM_ROUTER_CONTEXT_THRESHOLD_TOKENS`   (default 6000)
    ///   `SLM_ROUTER_LOCAL_HIGH_CTX_MAX_TOKENS`  (default 32000)
    ///   `SLM_ROUTER_LOCAL_HIGH_CTX_MODEL`       (default "qwen2.5:120b")
    ///   `SLM_ROUTER_LOCAL_CHEAP_MODEL`          (default "qwen2.5:20b")
    ///   `SLM_ROUTER_CLOUD_MODEL`                (default the fleet's existing
    ///                                             frontier-free OpenRouter
    ///                                             model id, `registry::OWL_ALPHA_MODEL_ID`)
    ///   `SLM_ROUTER_CLOUD_ALLOWED`              (default true; "false"/"0" disables)
    ///   `SLM_ROUTER_CLOUD_EGRESS_ALLOWLIST`     (comma-separated hostnames;
    ///                                             default "openrouter.ai")
    pub fn from_env() -> Self {
        Self {
            context_threshold_tokens: read_usize("SLM_ROUTER_CONTEXT_THRESHOLD_TOKENS", 6_000),
            local_high_ctx_max_tokens: read_usize("SLM_ROUTER_LOCAL_HIGH_CTX_MAX_TOKENS", 32_000),
            local_high_ctx_model: read_string(
                "SLM_ROUTER_LOCAL_HIGH_CTX_MODEL",
                DEFAULT_LOCAL_HIGH_CTX_MODEL,
            ),
            local_cheap_model: read_string("SLM_ROUTER_LOCAL_CHEAP_MODEL", DEFAULT_LOCAL_CHEAP_MODEL),
            cloud_frontier_model: read_string(
                "SLM_ROUTER_CLOUD_MODEL",
                crate::models::registry::OWL_ALPHA_MODEL_ID,
            ),
            allow_cloud: std::env::var("SLM_ROUTER_CLOUD_ALLOWED")
                .ok()
                .map(|v| !(v.eq_ignore_ascii_case("false") || v.trim() == "0"))
                .unwrap_or(true),
            cloud_egress_allowlist: std::env::var("SLM_ROUTER_CLOUD_EGRESS_ALLOWLIST")
                .ok()
                .map(|v| {
                    v.split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect::<Vec<_>>()
                })
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| vec![DEFAULT_CLOUD_EGRESS_HOST.to_string()]),
        }
    }

    /// Decide a destination for `request`, per policy. Never truncates: a
    /// request larger than the local high-context ceiling is routed to the
    /// cloud frontier-free model (if cloud is allowed) rather than served
    /// (and silently truncated) locally.
    pub fn decide(&self, request: &RoutingRequest) -> (RoutingDestination, String) {
        if request.estimated_tokens > self.local_high_ctx_max_tokens {
            if self.allow_cloud {
                return (
                    RoutingDestination::CloudFrontierFree,
                    format!(
                        "estimated_tokens {} exceeds local high-context capacity ({} tokens): \
                         routed to cloud frontier-free model rather than truncating",
                        request.estimated_tokens, self.local_high_ctx_max_tokens
                    ),
                );
            }
            // Cloud disallowed: still route to the highest-context option
            // available locally rather than silently truncating — the caller
            // (the doc engine) is responsible for chunking if this proves
            // insufficient; the router itself never truncates.
            return (
                RoutingDestination::LocalHighContext,
                format!(
                    "estimated_tokens {} exceeds local high-context capacity ({} tokens) but cloud \
                     routing is disabled: routed to local high-context model as the highest-context \
                     option available (not truncated)",
                    request.estimated_tokens, self.local_high_ctx_max_tokens
                ),
            );
        }

        if request.estimated_tokens > self.context_threshold_tokens {
            return (
                RoutingDestination::LocalHighContext,
                format!(
                    "estimated_tokens {} exceeds context_threshold_tokens {}: routed to local \
                     high-context model",
                    request.estimated_tokens, self.context_threshold_tokens
                ),
            );
        }

        (
            RoutingDestination::LocalCheap,
            format!(
                "estimated_tokens {} within context_threshold_tokens {}: routed to cheap local model",
                request.estimated_tokens, self.context_threshold_tokens
            ),
        )
    }

    /// The model name/tag configured for `destination`.
    pub fn model_for(&self, destination: RoutingDestination) -> &str {
        match destination {
            RoutingDestination::LocalHighContext => &self.local_high_ctx_model,
            RoutingDestination::LocalCheap => &self.local_cheap_model,
            RoutingDestination::CloudFrontierFree => &self.cloud_frontier_model,
        }
    }

    /// The next destination to try when `destination` is unavailable.
    /// Degrades gracefully: cloud → local high-context → local cheap. Local
    /// cheap is the floor — it falls back to itself, which callers must treat
    /// as "no further fallback" (never a silent no-op: if the floor also
    /// fails, the router must surface a hard error, not swallow it).
    pub fn fallback_for(&self, destination: RoutingDestination) -> RoutingDestination {
        match destination {
            RoutingDestination::CloudFrontierFree => RoutingDestination::LocalHighContext,
            RoutingDestination::LocalHighContext => RoutingDestination::LocalCheap,
            RoutingDestination::LocalCheap => RoutingDestination::LocalCheap,
        }
    }

    /// ISO egress gate: `true` only when cloud routing is enabled AND `host`
    /// is present (case-insensitively) in the configured allow-list. Fails
    /// CLOSED — an empty allow-list (which [`Self::from_env`] never actually
    /// produces, since it falls back to a default host, but a test or a
    /// future explicit config might) denies rather than allows, mirroring
    /// `supervisor::egress_policy::posture_for`'s "unset allow-list is
    /// Denied, never allow-all" invariant.
    pub fn is_cloud_egress_allowed(&self, host: &str) -> bool {
        if !self.allow_cloud {
            return false;
        }
        if host.is_empty() {
            return false;
        }
        if self.cloud_egress_allowlist.is_empty() {
            return false;
        }
        self.cloud_egress_allowlist
            .iter()
            .any(|h| h.eq_ignore_ascii_case(host))
    }
}

fn read_usize(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn read_string(var: &str, default: &str) -> String {
    std::env::var(var)
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_policy() -> RoutingPolicy {
        RoutingPolicy {
            context_threshold_tokens: 1_000,
            local_high_ctx_max_tokens: 10_000,
            local_high_ctx_model: "local-high-ctx".into(),
            local_cheap_model: "local-cheap".into(),
            cloud_frontier_model: "cloud-frontier".into(),
            allow_cloud: true,
            cloud_egress_allowlist: vec!["openrouter.ai".into()],
        }
    }

    fn req(tokens: usize) -> RoutingRequest {
        RoutingRequest {
            prompt: "x".repeat(10),
            estimated_tokens: tokens,
        }
    }

    // ── decide: simple → cheap, high-context → high-ctx destination ────────

    #[test]
    fn simple_request_routes_to_local_cheap() {
        let p = test_policy();
        let (dest, _) = p.decide(&req(100));
        assert_eq!(dest, RoutingDestination::LocalCheap);
    }

    #[test]
    fn exactly_at_threshold_stays_cheap() {
        let p = test_policy();
        let (dest, _) = p.decide(&req(1_000));
        assert_eq!(dest, RoutingDestination::LocalCheap);
    }

    #[test]
    fn high_context_request_routes_to_local_high_context() {
        let p = test_policy();
        let (dest, reason) = p.decide(&req(5_000));
        assert_eq!(dest, RoutingDestination::LocalHighContext);
        assert!(reason.contains("high-context"));
    }

    // ── decide: over-ceiling → cloud (never truncated) ──────────────────────

    #[test]
    fn over_local_ceiling_routes_to_cloud_when_allowed() {
        let p = test_policy();
        let (dest, reason) = p.decide(&req(20_000));
        assert_eq!(dest, RoutingDestination::CloudFrontierFree);
        assert!(reason.contains("truncat"));
    }

    #[test]
    fn over_local_ceiling_with_cloud_disabled_stays_local_high_context_not_truncated() {
        let mut p = test_policy();
        p.allow_cloud = false;
        let (dest, reason) = p.decide(&req(20_000));
        // Never silently truncated: still routes to the highest-context local
        // option, and the reason says so explicitly, rather than dropping
        // content.
        assert_eq!(dest, RoutingDestination::LocalHighContext);
        assert!(reason.contains("not truncated"));
    }

    // ── model_for ────────────────────────────────────────────────────────

    #[test]
    fn model_for_maps_each_destination() {
        let p = test_policy();
        assert_eq!(p.model_for(RoutingDestination::LocalCheap), "local-cheap");
        assert_eq!(
            p.model_for(RoutingDestination::LocalHighContext),
            "local-high-ctx"
        );
        assert_eq!(
            p.model_for(RoutingDestination::CloudFrontierFree),
            "cloud-frontier"
        );
    }

    // ── fallback_for ─────────────────────────────────────────────────────

    #[test]
    fn fallback_chain_degrades_cloud_to_high_ctx_to_cheap() {
        let p = test_policy();
        assert_eq!(
            p.fallback_for(RoutingDestination::CloudFrontierFree),
            RoutingDestination::LocalHighContext
        );
        assert_eq!(
            p.fallback_for(RoutingDestination::LocalHighContext),
            RoutingDestination::LocalCheap
        );
    }

    #[test]
    fn fallback_floor_is_local_cheap_pointing_to_itself() {
        // The floor: callers must detect "next == destination" as "no further
        // fallback exists" rather than looping forever.
        let p = test_policy();
        assert_eq!(
            p.fallback_for(RoutingDestination::LocalCheap),
            RoutingDestination::LocalCheap
        );
    }

    // ── is_cloud_egress_allowed: ISO egress gate ────────────────────────────

    #[test]
    fn egress_allowed_for_listed_host() {
        let p = test_policy();
        assert!(p.is_cloud_egress_allowed("openrouter.ai"));
        assert!(p.is_cloud_egress_allowed("OpenRouter.AI"), "case-insensitive");
    }

    #[test]
    fn egress_denied_for_unlisted_host() {
        let p = test_policy();
        assert!(!p.is_cloud_egress_allowed("evil.example.com"));
    }

    #[test]
    fn egress_denied_when_cloud_disabled_even_for_listed_host() {
        let mut p = test_policy();
        p.allow_cloud = false;
        assert!(!p.is_cloud_egress_allowed("openrouter.ai"));
    }

    #[test]
    fn egress_denied_when_allowlist_empty_fail_closed() {
        // Negative test: an empty allow-list must never mean "allow all" —
        // mirrors supervisor::egress_policy's fail-closed invariant.
        let mut p = test_policy();
        p.cloud_egress_allowlist = vec![];
        assert!(!p.is_cloud_egress_allowed("openrouter.ai"));
    }

    #[test]
    fn egress_denied_for_empty_host() {
        let p = test_policy();
        assert!(!p.is_cloud_egress_allowed(""));
    }

    // ── from_env: safe defaults, no panics ──────────────────────────────────

    #[test]
    #[serial_test::serial]
    fn from_env_has_safe_defaults_when_unset() {
        for var in [
            "SLM_ROUTER_CONTEXT_THRESHOLD_TOKENS",
            "SLM_ROUTER_LOCAL_HIGH_CTX_MAX_TOKENS",
            "SLM_ROUTER_LOCAL_HIGH_CTX_MODEL",
            "SLM_ROUTER_LOCAL_CHEAP_MODEL",
            "SLM_ROUTER_CLOUD_MODEL",
            "SLM_ROUTER_CLOUD_ALLOWED",
            "SLM_ROUTER_CLOUD_EGRESS_ALLOWLIST",
        ] {
            std::env::remove_var(var);
        }
        let p = RoutingPolicy::from_env();
        assert_eq!(p.context_threshold_tokens, 6_000);
        assert_eq!(p.local_high_ctx_max_tokens, 32_000);
        assert_eq!(p.local_high_ctx_model, DEFAULT_LOCAL_HIGH_CTX_MODEL);
        assert_eq!(p.local_cheap_model, DEFAULT_LOCAL_CHEAP_MODEL);
        assert!(p.allow_cloud);
        assert_eq!(p.cloud_egress_allowlist, vec![DEFAULT_CLOUD_EGRESS_HOST.to_string()]);
    }

    #[test]
    #[serial_test::serial]
    fn from_env_cloud_allowed_false_disables() {
        std::env::set_var("SLM_ROUTER_CLOUD_ALLOWED", "false");
        let p = RoutingPolicy::from_env();
        assert!(!p.allow_cloud);
        std::env::remove_var("SLM_ROUTER_CLOUD_ALLOWED");
    }

    #[test]
    fn test_no_hardcoded_infrastructure_values() {
        // Documentation/guard test, same convention as router_classifier.rs:
        // this module contains no private IPs or org domains.
        let src = include_str!("policy.rs");
        let private_ip_prefix = ["192", "168", "."].concat();
        let org_domain = ["moosenet", ".online"].concat();
        assert!(!src.contains(&private_ip_prefix));
        assert!(!src.contains(&org_domain));
    }
}
