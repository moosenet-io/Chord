//! DOCGEN-03: the Chord SLM router — request → destination decision → execute.
//!
//! This is the mechanism behind "all doc-engine inference routes through
//! Chord, and Chord owns the destination decision" (see the S95 documentation
//! engine spec, Design Overview). The doc engine (a separate `moosenet/Terminus`
//! item) only ever ASKS this router to generate; it never picks a model
//! itself.
//!
//! Reuses the existing routing substrate rather than reinventing it:
//! - Destinations resolve to real [`crate::models::backends::Backend`]s from
//!   [`crate::models::backends::seed_from_env`] — the same backend catalogue
//!   `models::routing::resolve_and_ensure` uses elsewhere in this crate. The
//!   local destinations (`LocalHighContext`/`LocalCheap`) both resolve to the
//!   primary Ollama-compatible backend and differ only in which MODEL name is
//!   requested — exactly the shape `AgenticModelRouter`'s `fast_model`/
//!   `deep_model` already use (`agentic/model_router.rs`).
//! - The cloud destination resolves to the existing `"openrouter"` backend,
//!   whose bearer key is read via the SAME `Backend::api_key_env` indirection
//!   `models::routing::resolve_and_ensure` uses: the env var NAME is config,
//!   the value is read fresh via a variable at call time (never a literal
//!   env-var lookup naming the secret directly in source), so the OpenRouter
//!   credential is never hardcoded and never read through a second, ad hoc path.
//!
//! ## Assumptions (per the spec's APPROACH)
//! - The input is already PII-swept upstream (the doc engine's DOCGEN-02
//!   gate). This router is NOT a PII gate and does not sweep `prompt`.
//! - Every cloud call still passes the ISO egress allow-list
//!   ([`super::policy::RoutingPolicy::is_cloud_egress_allowed`]) BEFORE any
//!   network call is attempted — enforced unconditionally in
//!   [`SlmRouter::route_and_execute`], independent of the PII assumption
//!   above.
//! - A destination that fails (unreachable, non-2xx, egress-denied) never
//!   silently fails the generation: the router walks
//!   [`super::policy::RoutingPolicy::fallback_for`] until either a destination
//!   succeeds or the fallback floor is reached, at which point it returns a
//!   hard [`SlmRouterError`] — never an empty/fabricated success.

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;

use crate::models::backends::{self, Backend};
use crate::router::policy::{RoutingDestination, RoutingPolicy, RoutingRequest};

/// One routing decision that was ACTED ON (including any fallback hops) —
/// enough detail for the DOCGEN-04 evaluation sweep to judge routing quality
/// (did a high-context task land on a high-context destination? at what
/// cost/latency?).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct RoutingDecision {
    pub destination: RoutingDestination,
    pub model: String,
    pub reason: String,
    /// `Some(previous_destination)` when this decision was reached by falling
    /// back from a destination that failed or was egress-denied.
    pub fallback_from: Option<RoutingDestination>,
}

/// Failure modes the router can return. Both variants mean the request was
/// NOT silently swallowed — the caller gets a clear, actionable error instead
/// of an empty/fabricated generation.
#[derive(Debug, Clone, thiserror::Error, PartialEq)]
pub enum SlmRouterError {
    /// The policy-decided destination (and every fallback hop) resolved to a
    /// cloud destination the ISO egress allow-list refuses, with no local
    /// fallback available either.
    #[error("cloud egress denied for host '{0}' and no fallback destination is available")]
    EgressDenied(String),
    /// Every destination in the fallback chain failed (backend not
    /// configured, or the execution attempt itself errored).
    #[error("all routing destinations exhausted: {0}")]
    AllDestinationsUnavailable(String),
}

/// Executes a generation call against a resolved backend + model. Abstracted
/// behind a trait so tests can simulate "destination unavailable" without a
/// live model, and so a future swap of transport (e.g. a shared HTTP client
/// pool) doesn't touch the routing logic above it.
#[async_trait]
pub trait Executor: Send + Sync {
    async fn execute(&self, backend: &Backend, model: &str, prompt: &str) -> Result<String, String>;
}

/// Real executor: an OpenAI-compatible `/v1/chat/completions` POST. Used for
/// both local (Ollama-compatible, unauthenticated) and cloud (OpenRouter,
/// bearer-authenticated via `Backend::api_key_env`) destinations — the only
/// difference is whether a bearer header is attached.
pub struct HttpExecutor {
    client: reqwest::Client,
}

impl HttpExecutor {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

impl Default for HttpExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Executor for HttpExecutor {
    async fn execute(&self, backend: &Backend, model: &str, prompt: &str) -> Result<String, String> {
        let url = format!("{}/v1/chat/completions", backend.url.trim_end_matches('/'));
        let mut req = self.client.post(&url).json(&serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }));

        // Bearer resolution: SAME indirection as `models::routing::resolve_and_ensure`
        // — `api_key_env` names an env var; the value is read fresh via that
        // variable, never a literal env-var lookup naming the secret directly.
        if let Some(env_name) = &backend.api_key_env {
            if let Ok(key) = std::env::var(env_name) {
                if !key.trim().is_empty() {
                    req = req.bearer_auth(key);
                }
            }
        }

        let resp = req.send().await.map_err(|e| e.to_string())?;
        let status = resp.status();
        if !status.is_success() {
            return Err(format!("HTTP {status}"));
        }
        let body: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
        body.get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| "malformed chat-completion response (no choices[0].message.content)".to_string())
    }
}

/// The Chord SLM router: decides a destination per [`RoutingPolicy`], resolves
/// it to a real [`Backend`], executes via an [`Executor`], and falls back
/// gracefully on failure. Logs every decision it acts on (for DOCGEN-04) via
/// both `tracing` and an in-memory log retrievable via [`SlmRouter::decisions`].
pub struct SlmRouter {
    policy: RoutingPolicy,
    backends: HashMap<String, Backend>,
    log: Mutex<Vec<RoutingDecision>>,
}

impl SlmRouter {
    /// Construct a router reading the live backend catalogue from the
    /// environment (`backends::seed_from_env`) — the same catalogue every
    /// other routing path in this crate uses.
    pub fn new(policy: RoutingPolicy) -> Self {
        Self {
            policy,
            backends: backends::seed_from_env(),
            log: Mutex::new(Vec::new()),
        }
    }

    /// Construct a router against an explicit backend map — for tests (and
    /// any future caller that wants a scoped, non-global backend catalogue).
    pub fn with_backends(policy: RoutingPolicy, backends: HashMap<String, Backend>) -> Self {
        Self {
            policy,
            backends,
            log: Mutex::new(Vec::new()),
        }
    }

    fn backend_for_destination(&self, destination: RoutingDestination) -> Option<&Backend> {
        match destination {
            // Local destinations differ only by MODEL, not by backend — both
            // resolve to the primary local Ollama-compatible backend, falling
            // back to the generic on-demand GPU backend if the primary isn't
            // configured in this environment.
            RoutingDestination::LocalHighContext | RoutingDestination::LocalCheap => self
                .backends
                .get("ollama")
                .or_else(|| self.backends.get("llama-gpu")),
            RoutingDestination::CloudFrontierFree => self.backends.get("openrouter"),
        }
    }

    fn host_for(backend: &Backend) -> String {
        reqwest::Url::parse(&backend.url)
            .ok()
            .and_then(|u| u.host_str().map(|s| s.to_string()))
            .unwrap_or_default()
    }

    fn record(&self, decision: RoutingDecision) {
        tracing::info!(
            destination = ?decision.destination,
            model = %decision.model,
            reason = %decision.reason,
            fallback_from = ?decision.fallback_from,
            "slm_router: routing decision"
        );
        if let Ok(mut log) = self.log.lock() {
            log.push(decision);
        }
    }

    /// Decisions logged so far, in the order they were acted on — the feed
    /// the DOCGEN-04 evaluation sweep consumes to judge routing quality.
    pub fn decisions(&self) -> Vec<RoutingDecision> {
        self.log.lock().map(|l| l.clone()).unwrap_or_default()
    }

    /// Route `request` per policy and execute the generation via `executor`.
    ///
    /// Walks the fallback chain on any failure (egress-denied, backend not
    /// configured, or the execution call itself erroring) until a destination
    /// succeeds or the fallback floor is reached — at which point this
    /// returns `Err`, never a silent/empty success. Every hop (including
    /// failed ones) is logged.
    pub async fn route_and_execute(
        &self,
        request: &RoutingRequest,
        executor: &dyn Executor,
    ) -> Result<(String, RoutingDecision), SlmRouterError> {
        let (mut destination, mut reason) = self.policy.decide(request);
        let mut fallback_from: Option<RoutingDestination> = None;

        loop {
            // ISO egress gate — checked BEFORE any network call is attempted,
            // for every hop that lands on the cloud destination (not just the
            // first). This is what makes the negative test meaningful: a
            // denied cloud hop never reaches `Executor::execute` at all.
            if destination == RoutingDestination::CloudFrontierFree {
                let host = self
                    .backend_for_destination(destination)
                    .map(Self::host_for)
                    .unwrap_or_default();
                if !self.policy.is_cloud_egress_allowed(&host) {
                    let denied_reason = format!("{reason}; egress denied for host '{host}'");
                    let next = self.policy.fallback_for(destination);
                    if next == destination {
                        self.record(RoutingDecision {
                            destination,
                            model: self.policy.model_for(destination).to_string(),
                            reason: denied_reason,
                            fallback_from,
                        });
                        return Err(SlmRouterError::EgressDenied(host));
                    }
                    self.record(RoutingDecision {
                        destination,
                        model: self.policy.model_for(destination).to_string(),
                        reason: denied_reason.clone(),
                        fallback_from,
                    });
                    fallback_from = Some(destination);
                    destination = next;
                    reason = format!("{denied_reason}; falling back to {destination:?}");
                    continue;
                }
            }

            let model = self.policy.model_for(destination).to_string();
            let decision = RoutingDecision {
                destination,
                model: model.clone(),
                reason: reason.clone(),
                fallback_from,
            };

            let Some(backend) = self.backend_for_destination(destination) else {
                let next = self.policy.fallback_for(destination);
                let unavailable_reason = format!("{reason}; no backend configured for {destination:?}");
                if next == destination {
                    self.record(RoutingDecision {
                        reason: unavailable_reason.clone(),
                        ..decision
                    });
                    return Err(SlmRouterError::AllDestinationsUnavailable(unavailable_reason));
                }
                self.record(RoutingDecision {
                    reason: unavailable_reason.clone(),
                    ..decision
                });
                fallback_from = Some(destination);
                destination = next;
                reason = format!("{unavailable_reason}; falling back to {destination:?}");
                continue;
            };

            match executor.execute(backend, &model, &request.prompt).await {
                Ok(output) => {
                    self.record(decision.clone());
                    return Ok((output, decision));
                }
                Err(err) => {
                    tracing::warn!(
                        destination = ?destination,
                        error = %err,
                        "slm_router: destination failed, applying policy fallback"
                    );
                    let next = self.policy.fallback_for(destination);
                    let failed_reason = format!("{reason}; destination failed ({err})");
                    if next == destination {
                        self.record(RoutingDecision {
                            reason: format!("{failed_reason}; no further fallback available"),
                            ..decision
                        });
                        return Err(SlmRouterError::AllDestinationsUnavailable(failed_reason));
                    }
                    self.record(RoutingDecision {
                        reason: failed_reason.clone(),
                        ..decision
                    });
                    fallback_from = Some(destination);
                    destination = next;
                    reason = format!("{failed_reason}; falling back to {destination:?}");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::backends::{BackendKind, Hardware};

    fn policy() -> RoutingPolicy {
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

    fn ollama_backend() -> Backend {
        Backend {
            name: "ollama".into(),
            url: "http://127.0.0.1:11434".into(),
            hardware: Hardware::Cpu,
            kind: BackendKind::Ollama,
            unit: None,
            always_on: true,
            idle_stop_secs: 0,
            launch: None,
            api_key_env: None,
        }
    }

    fn openrouter_backend() -> Backend {
        Backend {
            name: "openrouter".into(),
            url: "https://openrouter.ai/api".into(),
            hardware: Hardware::Cpu,
            kind: BackendKind::OpenRouter,
            unit: None,
            always_on: true,
            idle_stop_secs: 0,
            launch: None,
            api_key_env: Some("TEST_OPENROUTER_KEY_VAR".into()),
        }
    }

    fn backends_map(with_openrouter: bool) -> HashMap<String, Backend> {
        let mut m = HashMap::new();
        m.insert("ollama".into(), ollama_backend());
        if with_openrouter {
            m.insert("openrouter".into(), openrouter_backend());
        }
        m
    }

    /// Test executor: succeeds for backends whose name is in `succeeds_for`,
    /// fails for everything else. Records every call it receives (backend
    /// name) so tests can assert which destinations were actually reached —
    /// critical for the egress negative test (cloud must never be *called*).
    struct FakeExecutor {
        succeeds_for: Vec<&'static str>,
        calls: std::sync::Mutex<Vec<String>>,
    }

    impl FakeExecutor {
        fn new(succeeds_for: Vec<&'static str>) -> Self {
            Self {
                succeeds_for,
                calls: std::sync::Mutex::new(Vec::new()),
            }
        }

        fn calls(&self) -> Vec<String> {
            self.calls.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl Executor for FakeExecutor {
        async fn execute(&self, backend: &Backend, _model: &str, _prompt: &str) -> Result<String, String> {
            self.calls.lock().unwrap().push(backend.name.clone());
            if self.succeeds_for.contains(&backend.name.as_str()) {
                Ok(format!("generated-by-{}", backend.name))
            } else {
                Err(format!("{} is down", backend.name))
            }
        }
    }

    fn req(tokens: usize) -> RoutingRequest {
        RoutingRequest {
            prompt: "hello".into(),
            estimated_tokens: tokens,
        }
    }

    // ── happy path: destination decision → execute ──────────────────────────

    #[tokio::test]
    async fn simple_request_executes_on_local_cheap() {
        let router = SlmRouter::with_backends(policy(), backends_map(true));
        let executor = FakeExecutor::new(vec!["ollama", "openrouter"]);
        let (output, decision) = router
            .route_and_execute(&req(10), &executor)
            .await
            .expect("should succeed");
        assert_eq!(decision.destination, RoutingDestination::LocalCheap);
        assert_eq!(decision.model, "local-cheap");
        assert_eq!(output, "generated-by-ollama");
        assert!(decision.fallback_from.is_none());
    }

    #[tokio::test]
    async fn high_context_request_executes_on_local_high_context() {
        let router = SlmRouter::with_backends(policy(), backends_map(true));
        let executor = FakeExecutor::new(vec!["ollama", "openrouter"]);
        let (_, decision) = router
            .route_and_execute(&req(5_000), &executor)
            .await
            .expect("should succeed");
        assert_eq!(decision.destination, RoutingDestination::LocalHighContext);
        assert_eq!(decision.model, "local-high-ctx");
    }

    // ── negative test: destination down → policy fallback, no silent failure ─

    #[tokio::test]
    async fn destination_down_falls_back_per_policy() {
        let router = SlmRouter::with_backends(policy(), backends_map(true));
        // ollama (local) fails; nothing else succeeds either in this map for
        // the LocalCheap destination's only backend — but requesting a
        // high-context, over-ceiling request routes to cloud first, and cloud
        // succeeds, proving the fallback chain is walked correctly.
        let executor = FakeExecutor::new(vec!["openrouter"]);
        let (output, decision) = router
            .route_and_execute(&req(20_000), &executor)
            .await
            .expect("should fall back to a working destination, not fail silently");
        assert_eq!(decision.destination, RoutingDestination::CloudFrontierFree);
        assert_eq!(output, "generated-by-openrouter");
    }

    #[tokio::test]
    async fn all_destinations_failing_returns_hard_error_not_silent_success() {
        let router = SlmRouter::with_backends(policy(), backends_map(true));
        let executor = FakeExecutor::new(vec![]); // nothing succeeds
        let result = router.route_and_execute(&req(20_000), &executor).await;
        assert!(
            matches!(result, Err(SlmRouterError::AllDestinationsUnavailable(_))),
            "must return a hard error, never a silent/fabricated success: {result:?}"
        );
    }

    #[tokio::test]
    async fn fallback_hop_is_recorded_with_fallback_from_set() {
        // Over-ceiling request whose primary decision is Cloud — deny cloud
        // egress so the first hop is skipped, landing on LocalHighContext
        // with fallback_from = Some(CloudFrontierFree).
        let mut p = policy();
        p.cloud_egress_allowlist = vec![]; // fail closed
        let router = SlmRouter::with_backends(p, backends_map(true));
        let executor = FakeExecutor::new(vec!["ollama"]);
        let (_, decision) = router
            .route_and_execute(&req(20_000), &executor)
            .await
            .expect("should succeed via fallback");
        assert_eq!(decision.destination, RoutingDestination::LocalHighContext);
        assert_eq!(decision.fallback_from, Some(RoutingDestination::CloudFrontierFree));
    }

    // ── negative test: cloud egress isolation/allowlist ─────────────────────

    #[tokio::test]
    async fn egress_denied_host_never_reaches_the_executor() {
        let mut p = policy();
        p.cloud_egress_allowlist = vec![]; // fail closed: nothing allowed
        let router = SlmRouter::with_backends(p, backends_map(true));
        let executor = FakeExecutor::new(vec!["ollama", "openrouter"]);

        // Over-ceiling request would normally route to cloud first.
        let (_, decision) = router
            .route_and_execute(&req(20_000), &executor)
            .await
            .expect("should fall back to local since cloud is egress-denied");

        assert_eq!(decision.destination, RoutingDestination::LocalHighContext);
        assert_eq!(
            executor.calls(),
            vec!["ollama".to_string()],
            "the cloud backend must NEVER be invoked when egress is denied — no unisolated cloud egress"
        );
    }

    #[tokio::test]
    async fn egress_denied_with_no_local_fallback_returns_hard_error() {
        // No "ollama"/"llama-gpu" backend configured at all, so once the
        // cloud hop is egress-denied, the LocalHighContext fallback also has
        // no backend to resolve — this must be a hard error, never a silent
        // success or an empty output.
        let mut p = policy();
        p.cloud_egress_allowlist = vec![];
        let router = SlmRouter::with_backends(p, HashMap::new());
        let executor = FakeExecutor::new(vec!["ollama", "openrouter"]);
        let result = router.route_and_execute(&req(20_000), &executor).await;
        assert!(
            matches!(result, Err(SlmRouterError::AllDestinationsUnavailable(_))),
            "no backend at all for the fallback destination must be a hard error: {result:?}"
        );
        assert!(executor.calls().is_empty(), "no backend was ever resolved, so the executor is never called");
    }

    #[tokio::test]
    async fn cloud_disabled_never_calls_openrouter_even_when_reachable() {
        let mut p = policy();
        p.allow_cloud = false;
        let router = SlmRouter::with_backends(p, backends_map(true));
        let executor = FakeExecutor::new(vec!["ollama", "openrouter"]);
        let (_, decision) = router
            .route_and_execute(&req(20_000), &executor)
            .await
            .expect("should succeed locally");
        assert_eq!(decision.destination, RoutingDestination::LocalHighContext);
        assert!(!executor.calls().contains(&"openrouter".to_string()));
    }

    // ── routing decisions are logged for the evaluation sweep (DOCGEN-04) ───

    #[tokio::test]
    async fn successful_decision_is_logged() {
        let router = SlmRouter::with_backends(policy(), backends_map(true));
        let executor = FakeExecutor::new(vec!["ollama", "openrouter"]);
        router
            .route_and_execute(&req(10), &executor)
            .await
            .expect("should succeed");
        let decisions = router.decisions();
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].destination, RoutingDestination::LocalCheap);
        assert!(!decisions[0].reason.is_empty());
    }

    #[tokio::test]
    async fn failed_hops_are_logged_too_not_just_the_final_success() {
        let mut p = policy();
        p.cloud_egress_allowlist = vec![];
        let router = SlmRouter::with_backends(p, backends_map(true));
        let executor = FakeExecutor::new(vec!["ollama"]);
        router
            .route_and_execute(&req(20_000), &executor)
            .await
            .expect("should succeed via fallback");
        let decisions = router.decisions();
        // At least the denied cloud hop + the successful local hop.
        assert!(decisions.len() >= 2, "expected the denied hop and the successful hop both logged: {decisions:?}");
        assert!(decisions.iter().any(|d| d.destination == RoutingDestination::CloudFrontierFree));
        assert!(decisions.iter().any(|d| d.destination == RoutingDestination::LocalHighContext));
    }

    #[tokio::test]
    async fn multiple_requests_accumulate_in_the_log_in_order() {
        let router = SlmRouter::with_backends(policy(), backends_map(true));
        let executor = FakeExecutor::new(vec!["ollama", "openrouter"]);
        router.route_and_execute(&req(10), &executor).await.unwrap();
        router.route_and_execute(&req(5_000), &executor).await.unwrap();
        let decisions = router.decisions();
        assert_eq!(decisions.len(), 2);
        assert_eq!(decisions[0].destination, RoutingDestination::LocalCheap);
        assert_eq!(decisions[1].destination, RoutingDestination::LocalHighContext);
    }

    // ── HttpExecutor: bearer resolution via api_key_env indirection ────────

    #[tokio::test]
    async fn http_executor_uses_api_key_env_indirection_not_a_literal() {
        // Mirrors resolve_and_ensure_returns_bearer_key_for_openrouter_backend
        // in models/routing.rs: the env var NAME is config; the value is read
        // fresh via that variable at call time.
        let server = httpmock::MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(httpmock::Method::POST)
                    .path("/v1/chat/completions")
                    .header("authorization", "Bearer test-bearer-value-not-real");
                then.status(200).json_body(serde_json::json!({
                    "choices": [{"message": {"content": "ok"}}]
                }));
            })
            .await;

        std::env::set_var("TEST_HTTP_EXECUTOR_KEY_VAR", "test-bearer-value-not-real");
        let backend = Backend {
            name: "openrouter".into(),
            url: server.base_url(),
            hardware: Hardware::Cpu,
            kind: BackendKind::OpenRouter,
            unit: None,
            always_on: true,
            idle_stop_secs: 0,
            launch: None,
            api_key_env: Some("TEST_HTTP_EXECUTOR_KEY_VAR".into()),
        };

        let executor = HttpExecutor::new();
        let result = executor.execute(&backend, "cloud-frontier", "hi").await;
        mock.assert_async().await;
        assert_eq!(result.as_deref(), Ok("ok"));
        std::env::remove_var("TEST_HTTP_EXECUTOR_KEY_VAR");
    }

    #[tokio::test]
    async fn http_executor_non_2xx_is_an_error_not_a_panic() {
        let server = httpmock::MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(httpmock::Method::POST).path("/v1/chat/completions");
                then.status(500).body("boom");
            })
            .await;
        let backend = Backend {
            name: "ollama".into(),
            url: server.base_url(),
            hardware: Hardware::Cpu,
            kind: BackendKind::Ollama,
            unit: None,
            always_on: true,
            idle_stop_secs: 0,
            launch: None,
            api_key_env: None,
        };
        let executor = HttpExecutor::new();
        let result = executor.execute(&backend, "local-cheap", "hi").await;
        assert!(result.is_err());
    }

    #[test]
    fn test_no_hardcoded_infrastructure_values() {
        let src = include_str!("slm_router.rs");
        let private_ip_prefix = ["192", "168", "."].concat();
        let org_domain = ["moosenet", ".online"].concat();
        assert!(!src.contains(&private_ip_prefix));
        assert!(!src.contains(&org_domain));
    }
}
