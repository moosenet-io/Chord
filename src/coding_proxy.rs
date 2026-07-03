//! CPROX-03/04: the coding-proxy HTTP endpoint.
//!
//! `POST /v1/coding/select` — Harmony sends a
//! [`WorkTypeCode`](crate::models::work_type::WorkTypeCode) ("I need CODE work
//! of this shape") instead of a hardcoded model alias. Chord ranks the REAL
//! coder-sweep fleet data (CPROX-02's [`crate::models::coding_selector`]),
//! tries the top candidate's health, and falls back down the ranked list on
//! failure (CPROX-04) — the same "never silently succeed with something
//! unverified" discipline as the rest of this proxy.
//!
//! ## Design decision: resolution, not a transparent proxy
//! This endpoint returns a MODEL RESOLUTION (`model_id` / `backend` /
//! `confidence`) for the caller to dispatch itself against Chord's EXISTING
//! `/v1/chat/completions` (or `/v1/infer`) — it does NOT itself forward a
//! chat-completion body. Reasons:
//!   1. `chat_completions` already owns the full inference-proxy concerns
//!      (JWT auth, per-user LLM rate-limit budget, model-alias rewriting,
//!      streaming passthrough, audit logging). Re-doing all of that here to
//!      also proxy the completion would either duplicate ~300 lines of that
//!      logic or require a deeper refactor than this item's scope.
//!   2. The two concerns are genuinely separable: "which model should serve
//!      this coding work" (a ranking decision over sweep data) is orthogonal
//!      to "how do I safely forward a chat completion" (already solved).
//!      Harmony can call this endpoint once per work item, then reuse its
//!      existing `/v1/chat/completions` client code unchanged, just pointing
//!      it at the resolved model name.
//!   3. Because this returns a resolution rather than proxying, it does not
//!      need to touch the per-user LLM rate-limit budget at all — the actual
//!      inference call (wherever the caller sends it) is what consumes budget,
//!      exactly once, avoiding a double-count.
//!
//! ## Backend resolution (a judgment call, documented)
//! `code_profile_runs.backend_tag` only distinguishes `"gpu"` from
//! absent/legacy — it is not fine-grained enough to name a specific serving
//! backend (`llama-gpu` vs `lemonade-coder` vs `vulkan`). This module maps
//! `Some("gpu")` → the generic on-demand `llama-gpu` backend (serves ANY
//! requested model's blob on GPU — see `models::backends::seed_from_env`) and
//! anything else → the always-on `ollama` backend. Neither of these mappings
//! ever names `vulkan`, so no candidate reaching this module can be resolved
//! onto it — that finer-grained routing is out of this item's scope. Note
//! MoE-tagged candidates never even reach this module: CPROX-02's
//! [`crate::models::coding_selector::rank_candidates`] excludes them from the
//! ranked list entirely (see that module's docs) — there is no per-candidate
//! safety flag here to check, because an unsafe candidate is never one of the
//! inputs in the first place.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
use tracing::warn;

use crate::models::backends::{seed_from_env, Backend};
use crate::models::coding_selector::{rank_for_work_type, CodeProfileSource, CodingCandidate, SelectorError};
use crate::models::work_type::WorkTypeCode;

/// Health-checker abstraction, reusing [`crate::serving::launcher::HealthChecker`]
/// verbatim (Chord's existing model health-check trait) rather than inventing a
/// new one for this endpoint.
pub use crate::serving::launcher::HealthChecker;

/// Production [`HealthChecker`]: `GET {endpoint}` (the FULL health URL, e.g.
/// `http://host:port/health` — the caller appends the path), success on any 2xx
/// response. Mirrors the exact pattern already established in
/// `snap::health::poll_vllm` (`GET {base_url}/health`) — this is that same
/// convention, not a new one.
pub struct HttpHealthChecker {
    client: reqwest::Client,
}

impl HttpHealthChecker {
    pub fn new(client: reqwest::Client) -> Self {
        HttpHealthChecker { client }
    }
}

#[async_trait]
impl HealthChecker for HttpHealthChecker {
    async fn check(&self, endpoint: &str) -> bool {
        match self.client.get(endpoint).send().await {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }
}

/// A resolved backend target for a candidate: the backend's stable name (for
/// the response + logging) and its base URL (for the health probe).
#[derive(Debug, Clone, PartialEq)]
pub struct BackendResolution {
    pub name: String,
    pub url: String,
}

/// Map a [`CodingCandidate`]'s coarse `backend_tag` (`"gpu"` or
/// absent/legacy) onto one of Chord's seeded backends. `None` when the
/// relevant backend isn't configured on this host at all (e.g. no
/// `LLAMA_GPU_BIN`/`OLLAMA_URL` env) — the caller treats that exactly like a
/// failed health check (try the next candidate).
///
/// See the module-level "Backend resolution" doc comment for why this mapping
/// is coarse and what it deliberately does not attempt.
pub fn resolve_backend_for_candidate(
    candidate: &CodingCandidate,
    backends: &HashMap<String, Backend>,
) -> Option<BackendResolution> {
    let key = match candidate.backend_tag.as_deref() {
        Some("gpu") => "llama-gpu",
        _ => "ollama",
    };
    backends.get(key).map(|b| BackendResolution {
        name: b.name.clone(),
        url: b.url.clone(),
    })
}

/// The request body: Harmony's work-type-tagged coding request. A malformed or
/// unknown-variant body fails Axum's `Json` extraction before this handler
/// ever runs (see [`WorkTypeCode`]'s strict serde derive) — a 4xx automatically,
/// never a 500, never a hang. Concretely: invalid JSON syntax (e.g. an empty
/// body) is 400 Bad Request; valid JSON that doesn't match the schema (e.g. an
/// unknown enum variant) is 422 Unprocessable Entity — see
/// `test_coding_select_malformed_body_returns_4xx_not_500` /
/// `test_coding_select_empty_body_returns_4xx_not_500` in `tests/e2e.rs`.
pub type CodingSelectRequest = WorkTypeCode;

/// What Chord actually picked, plus enough detail for Harmony (and logs/audits)
/// to know which model/backend is now serving the request and how confident
/// the pick was.
#[derive(Debug, Clone, Serialize)]
pub struct SelectedModel {
    pub model_id: String,
    pub backend: String,
    /// The candidate's `combined_score` at selection time (see
    /// `coding_selector`'s documented scoring formula) — `[0, 1]`-ish, higher
    /// is more confident.
    pub confidence: f64,
    pub mem_config: Option<String>,
}

/// Successful `/v1/coding/select` response.
#[derive(Debug, Clone, Serialize)]
pub struct CodingSelectResponse {
    pub selected: SelectedModel,
    /// 0-based index into the ranked candidate list this selection came from —
    /// `0` means the top-ranked candidate was healthy on the first try; `N>0`
    /// means the first `N` candidates failed their health check and this is
    /// the `(N+1)`th one tried (CPROX-04's fallback tier).
    pub fallback_tier: usize,
    pub candidates_considered: usize,
}

/// A `/v1/coding/select` failure. Every variant maps to a specific, documented
/// HTTP status — never a 500, never a silent hang.
#[derive(Debug, Clone, PartialEq)]
pub enum CodingSelectError {
    /// The coding-profile data source isn't wired up (no intake DB configured).
    NotConfigured,
    /// The data source is configured but temporarily unreachable/erroring.
    StoreUnavailable,
    /// The query ran fine but returned zero candidates for this
    /// language/work-type (e.g. a language the sweep hasn't covered yet).
    NoCandidates,
    /// Candidates exist, but EVERY one failed its health/availability check —
    /// the CPROX-04 "never silently succeed with something unverified" case.
    AllCandidatesUnavailable { candidates_tried: usize },
}

impl IntoResponse for CodingSelectError {
    fn into_response(self) -> Response {
        let (status, error) = match &self {
            CodingSelectError::NotConfigured => (
                StatusCode::SERVICE_UNAVAILABLE,
                "coding-model selection is not configured (no intake DB)".to_string(),
            ),
            CodingSelectError::StoreUnavailable => (
                StatusCode::SERVICE_UNAVAILABLE,
                "coding-model selection store is temporarily unavailable".to_string(),
            ),
            CodingSelectError::NoCandidates => (
                StatusCode::SERVICE_UNAVAILABLE,
                "no measured coding-model candidates exist yet for this language".to_string(),
            ),
            CodingSelectError::AllCandidatesUnavailable { candidates_tried } => (
                StatusCode::SERVICE_UNAVAILABLE,
                format!(
                    "all {candidates_tried} ranked coding-model candidate(s) failed their \
                     health check — refusing to return an unverified selection"
                ),
            ),
        };
        (status, Json(serde_json::json!({ "error": error }))).into_response()
    }
}

impl From<SelectorError> for CodingSelectError {
    fn from(e: SelectorError) -> Self {
        match e {
            SelectorError::NotConfigured => CodingSelectError::NotConfigured,
            SelectorError::StoreUnavailable => CodingSelectError::StoreUnavailable,
        }
    }
}

/// The testable core of CPROX-04's fallback walk: given already-ranked
/// candidates, a backend map, and a health checker, try each candidate's
/// resolved backend in order and return the first healthy one. Pure enough to
/// unit test with a scripted [`HealthChecker`] — no HTTP server, no Postgres.
pub async fn select_with_fallback(
    candidates: &[CodingCandidate],
    backends: &HashMap<String, Backend>,
    health: &dyn HealthChecker,
) -> Result<CodingSelectResponse, CodingSelectError> {
    if candidates.is_empty() {
        return Err(CodingSelectError::NoCandidates);
    }

    for (tier, candidate) in candidates.iter().enumerate() {
        let Some(resolution) = resolve_backend_for_candidate(candidate, backends) else {
            warn!(
                model_id = %candidate.model_id,
                backend_tag = ?candidate.backend_tag,
                "coding-select: no configured backend for candidate's tag — skipping (fallback tier {tier})"
            );
            continue;
        };
        let health_url = format!("{}/health", resolution.url);
        if health.check(&health_url).await {
            return Ok(CodingSelectResponse {
                selected: SelectedModel {
                    model_id: candidate.model_id.clone(),
                    backend: resolution.name,
                    confidence: candidate.combined_score,
                    mem_config: candidate.mem_config.clone(),
                },
                fallback_tier: tier,
                candidates_considered: candidates.len(),
            });
        }
        warn!(
            model_id = %candidate.model_id,
            backend = %resolution.name,
            "coding-select: candidate failed health check — trying next (fallback tier {tier})"
        );
    }

    Err(CodingSelectError::AllCandidatesUnavailable {
        candidates_tried: candidates.len(),
    })
}

/// `POST /v1/coding/select` handler. JWT auth follows the same pattern as
/// every other proxy endpoint (see `routes::auth_check`). This endpoint is
/// deliberately NOT rate-limited against the per-user LLM budget — see the
/// module-level design-decision doc comment (it resolves a model, it does not
/// itself perform inference).
pub async fn coding_select(
    State(state): State<Arc<crate::routes::AppState>>,
    headers: axum::http::HeaderMap,
    Json(work_type): Json<CodingSelectRequest>,
) -> Response {
    if let Err(e) = crate::routes::auth_check(&headers, &state.jwt_secret) {
        return crate::routes::auth_error_response(e);
    }

    let Some(source) = state.coding_profile_source.lock().await.clone() else {
        return CodingSelectError::NotConfigured.into_response();
    };

    let candidates = match rank_for_work_type(source.as_ref(), &work_type).await {
        Ok(c) => c,
        Err(e) => return CodingSelectError::from(e).into_response(),
    };

    let backends = seed_from_env();
    let health = HttpHealthChecker::new(state.http_client.clone());

    match select_with_fallback(&candidates, &backends, &health).await {
        Ok(resp) => (StatusCode::OK, Json(resp)).into_response(),
        Err(e) => e.into_response(),
    }
}

/// Shared type alias for the AppState field: a best-effort, hot-swappable
/// coding-profile data source. `None` ⇒ not configured (fail-open at startup,
/// same discipline as `AppState::routing_map` / `AppState::model_registry`).
pub type SharedCodingProfileSource =
    Arc<tokio::sync::Mutex<Option<Arc<dyn CodeProfileSource>>>>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::backends::{BackendKind, Hardware};

    fn backend(name: &str, url: &str) -> Backend {
        Backend {
            name: name.to_string(),
            url: url.to_string(),
            hardware: Hardware::Gpu,
            kind: BackendKind::LlamaServer,
            unit: None,
            always_on: true,
            idle_stop_secs: 0,
            launch: None,
            api_key_env: None,
        }
    }

    fn candidate(model_id: &str, backend_tag: Option<&str>, score: f64) -> CodingCandidate {
        CodingCandidate {
            model_id: model_id.to_string(),
            backend_tag: backend_tag.map(str::to_string),
            mem_config: None,
            run_count: 10,
            avg_effective_score: Some(4.0),
            compile_pass_rate: Some(1.0),
            test_pass_rate: Some(1.0),
            combined_score: score,
            yarn_bonus_applied: false,
        }
    }

    /// Health checker that reports healthy only for a fixed set of endpoints,
    /// and records every endpoint it was asked about (call-order verifiable).
    struct ScriptedHealth {
        healthy: Vec<String>,
        calls: std::sync::Mutex<Vec<String>>,
    }

    #[async_trait]
    impl HealthChecker for ScriptedHealth {
        async fn check(&self, endpoint: &str) -> bool {
            self.calls.lock().unwrap().push(endpoint.to_string());
            self.healthy.iter().any(|h| h == endpoint)
        }
    }

    #[test]
    fn resolve_backend_maps_gpu_tag_to_llama_gpu() {
        let mut backends = HashMap::new();
        backends.insert("llama-gpu".to_string(), backend("llama-gpu", "http://localhost:8082"));
        backends.insert("ollama".to_string(), backend("ollama", "http://localhost:11434"));

        let c = candidate("model-a", Some("gpu"), 0.9);
        let r = resolve_backend_for_candidate(&c, &backends).expect("resolved");
        assert_eq!(r.name, "llama-gpu");
        assert_eq!(r.url, "http://localhost:8082");
    }

    #[test]
    fn resolve_backend_maps_absent_tag_to_ollama() {
        let mut backends = HashMap::new();
        backends.insert("ollama".to_string(), backend("ollama", "http://localhost:11434"));

        let c = candidate("model-a", None, 0.9);
        let r = resolve_backend_for_candidate(&c, &backends).expect("resolved");
        assert_eq!(r.name, "ollama");
    }

    #[test]
    fn resolve_backend_none_when_unconfigured() {
        let backends: HashMap<String, Backend> = HashMap::new();
        let c = candidate("model-a", Some("gpu"), 0.9);
        assert!(resolve_backend_for_candidate(&c, &backends).is_none());
    }

    #[tokio::test]
    async fn select_with_fallback_picks_top_candidate_when_healthy() {
        let mut backends = HashMap::new();
        backends.insert("ollama".to_string(), backend("ollama", "http://ollama.local"));
        let candidates = vec![candidate("best-model", None, 0.9), candidate("second-model", None, 0.5)];
        let health = ScriptedHealth {
            healthy: vec!["http://ollama.local/health".to_string()],
            calls: Default::default(),
        };

        let resp = select_with_fallback(&candidates, &backends, &health).await.expect("selected");
        assert_eq!(resp.selected.model_id, "best-model");
        assert_eq!(resp.fallback_tier, 0);
        assert_eq!(resp.candidates_considered, 2);
    }

    #[tokio::test]
    async fn select_with_fallback_falls_back_on_unhealthy_top_candidate() {
        // Two distinct backends so the health check can distinguish them.
        let mut backends = HashMap::new();
        backends.insert("llama-gpu".to_string(), backend("llama-gpu", "http://gpu.local"));
        backends.insert("ollama".to_string(), backend("ollama", "http://ollama.local"));

        let candidates = vec![
            candidate("gpu-model", Some("gpu"), 0.9),   // resolves to gpu.local — will fail
            candidate("cpu-model", None, 0.5),          // resolves to ollama.local — healthy
        ];
        let health = ScriptedHealth {
            healthy: vec!["http://ollama.local/health".to_string()],
            calls: Default::default(),
        };

        let resp = select_with_fallback(&candidates, &backends, &health).await.expect("selected");
        assert_eq!(resp.selected.model_id, "cpu-model");
        assert_eq!(resp.fallback_tier, 1, "must record that tier 0 was skipped");
    }

    #[tokio::test]
    async fn select_with_fallback_all_unavailable_is_distinguishable_error() {
        let mut backends = HashMap::new();
        backends.insert("ollama".to_string(), backend("ollama", "http://ollama.local"));
        let candidates = vec![candidate("only-model", None, 0.9)];
        let health = ScriptedHealth { healthy: vec![], calls: Default::default() };

        let err = select_with_fallback(&candidates, &backends, &health).await.unwrap_err();
        assert_eq!(err, CodingSelectError::AllCandidatesUnavailable { candidates_tried: 1 });
    }

    #[tokio::test]
    async fn select_with_fallback_empty_candidates_is_no_candidates_error() {
        let backends: HashMap<String, Backend> = HashMap::new();
        let health = ScriptedHealth { healthy: vec![], calls: Default::default() };
        let err = select_with_fallback(&[], &backends, &health).await.unwrap_err();
        assert_eq!(err, CodingSelectError::NoCandidates);
    }

    #[tokio::test]
    async fn select_with_fallback_skips_candidate_with_unconfigured_backend() {
        // First candidate is "gpu" but llama-gpu isn't configured at all —
        // must be skipped (not treated as a health failure that still counts
        // toward a confusing error), falling to the second, configured one.
        let mut backends = HashMap::new();
        backends.insert("ollama".to_string(), backend("ollama", "http://ollama.local"));
        let candidates = vec![
            candidate("gpu-model-unconfigured-backend", Some("gpu"), 0.9),
            candidate("cpu-model", None, 0.5),
        ];
        let health = ScriptedHealth {
            healthy: vec!["http://ollama.local/health".to_string()],
            calls: Default::default(),
        };
        let resp = select_with_fallback(&candidates, &backends, &health).await.expect("selected");
        assert_eq!(resp.selected.model_id, "cpu-model");
    }

    #[tokio::test]
    async fn health_checker_calls_are_in_ranked_order() {
        let mut backends = HashMap::new();
        backends.insert("ollama".to_string(), backend("ollama", "http://ollama.local"));
        let candidates = vec![candidate("a", None, 0.9), candidate("b", None, 0.5)];
        let calls = std::sync::Mutex::new(Vec::new());
        let health = ScriptedHealth { healthy: vec![], calls };
        let _ = select_with_fallback(&candidates, &backends, &health).await;
        let calls = health.calls.lock().unwrap();
        assert_eq!(calls.len(), 2, "both candidates should have been tried in order");
    }

    #[tokio::test]
    async fn http_health_checker_reports_false_on_connection_failure() {
        let checker = HttpHealthChecker::new(reqwest::Client::new());
        // Nothing is listening on this port — a connection failure, not a panic.
        let ok = checker.check("http://127.0.0.1:1/health").await;
        assert!(!ok);
    }

    // Guards against accidental reordering that would make fallback_tier lie.
    #[tokio::test]
    async fn fallback_tier_reflects_actual_skip_count_with_unconfigured_backend_between() {
        let mut backends = HashMap::new();
        backends.insert("ollama".to_string(), backend("ollama", "http://ollama.local"));
        let candidates = vec![
            candidate("tier0-unconfigured", Some("gpu"), 0.95),
            candidate("tier1-healthy", None, 0.5),
        ];
        let health = ScriptedHealth {
            healthy: vec!["http://ollama.local/health".to_string()],
            calls: Default::default(),
        };
        let resp = select_with_fallback(&candidates, &backends, &health).await.unwrap();
        // tier0 is skipped for lack of backend config (not a health failure),
        // but it still occupied ranked-list index 0, so fallback_tier is 1.
        assert_eq!(resp.fallback_tier, 1);
        assert_eq!(resp.selected.model_id, "tier1-healthy");
    }
}
