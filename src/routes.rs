//! Axum route handlers for the MCP proxy + LLM inference endpoints.
//!
//! All endpoints require JWT auth (Bearer token, same secret as LLM calls).
//! Endpoints:
//!   POST /v1/tools/list        → return merged tool catalog
//!   POST /v1/tools/call        → execute a tool by name
//!   POST /v1/tools/discover    → search catalog by query
//!   POST /v1/agent/execute     → guarded agentic tool-calling loop
//!   POST /v1/chat/completions  → OpenAI-compatible LLM proxy (→ CHORD_LLM_URL)
//!   POST /v1/coding/select     → CPROX-03/04: fleet-driven coding-model resolution
//!   GET  /health               → health check (no auth)

use axum::{
    body::Body,
    extract::{Json, State},
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tracing::warn;

use crate::agentic::{AgenticExecutor, AgenticRequest, AgenticResponse};
use crate::audit::{AuditLogger, AuditSummary, Status as AuditStatus};
use crate::auth::{extract_bearer, validate_jwt, Claims};
use crate::catalog::ToolEntry;
use crate::error::{AuthError, ProxyError};
use crate::mcp_proxy::McpProxy;
use crate::rate_limiter::{CallType, ProxyRateLimiter, RateLimitResult, UserRole};

/// Shared application state.
pub struct AppState {
    pub proxy: McpProxy,
    pub jwt_secret: String,
    pub audit_logger: Arc<AuditLogger>,
    pub rate_limiter: Arc<Mutex<ProxyRateLimiter>>,
    pub agentic_executor: Arc<AgenticExecutor>,
    /// Upstream LLM backend URL for `/v1/chat/completions`. `None` → endpoint disabled (503).
    pub llm_backend_url: Option<String>,
    /// Model alias → real model map (from CHORD_MODEL_ALIASES). Empty → no rewriting.
    pub model_aliases: std::collections::HashMap<String, String>,
    /// Shared HTTP client for proxying LLM requests (connection pooling).
    pub http_client: reqwest::Client,
    /// TIER-01/02 model registry (tier/size/timestamps). Shared with
    /// `pull_coordinator`; locked briefly in `chat_completions` to look up a
    /// resolved model's tier for the pull-on-miss hook.
    pub model_registry: Arc<Mutex<crate::models::registry::ModelRegistry>>,
    /// TIER-02 archive pull coordinator (cold → warm). Wraps a clone of the same
    /// `model_registry` and dedups concurrent pulls per model.
    pub pull_coordinator: Arc<crate::models::transfer::PullCoordinator>,
    /// TIER-05: GC-aware local evictor used by the control API's manual archive
    /// endpoint and sweep. The same evictor instance the background sweep uses.
    pub local_evictor: Arc<dyn crate::models::eviction::LocalEvictor>,
    /// TIER-05: shared disk-op lock serialising the control sweep / archive with
    /// the background sweep and pre-pull eviction so destructive ops never race.
    pub disk_op_lock: crate::models::eviction::DiskOpLock,
    /// TIER-05: disk-space probe for `GET /api/storage` and the manual sweep.
    pub disk_probe: Arc<dyn crate::models::transfer::DiskSpaceProbe>,
    /// TIER-05: disk-pressure threshold (percent) the manual sweep evicts above.
    pub disk_pressure_percent: u8,
    /// TIER-05: cooldown window for the manual sweep (hours before warm model is eligible).
    pub model_warm_cooldown_hours: u64,
    /// YARN-06: SRV-04 serving-profile routing map — the source of a model's
    /// [`crate::serving::profile::ThinkingConfig`] (capability advertisement +
    /// per-request thinking honoring in `chat_completions`). Best-effort: when
    /// the intake DB is unreachable/unconfigured this is
    /// [`crate::serving::profile::RoutingMap::empty`] (every lookup misses,
    /// `thinking_available` reports `false`) rather than blocking startup —
    /// the same fail-open discipline as `model_registry`/the eviction sweep.
    /// A background refresh (if added later) replaces the whole map, same as
    /// [`crate::serving::profile::RoutingMap::load`] already guarantees.
    pub routing_map: Arc<Mutex<crate::serving::profile::RoutingMap>>,
    /// CPROX-02/03: fleet-driven coding-model data source. Best-effort, same
    /// fail-open discipline as `routing_map`/`model_registry`: `None` when the
    /// intake DB isn't configured/reachable, in which case
    /// `POST /v1/coding/select` returns a clear 503
    /// ([`crate::coding_proxy::CodingSelectError::NotConfigured`]) rather than
    /// blocking startup or panicking.
    pub coding_profile_source: crate::coding_proxy::SharedCodingProfileSource,
}

// ── Auth helpers ──────────────────────────────────────────────────────────────

/// Validates JWT and returns the claims. Returns Err(AuthError) if invalid.
/// When jwt_secret is empty, auth is disabled and a synthetic lumina claim is returned.
/// `pub(crate)` so the TIER-05 control router (`control.rs`) gates its endpoints
/// with the exact same auth as the proxy port.
pub(crate) fn auth_check(headers: &HeaderMap, jwt_secret: &str) -> Result<Claims, AuthError> {
    // Auth disabled when no secret configured
    if jwt_secret.is_empty() {
        return Ok(Claims {
            sub: "lumina".into(),
            exp: u64::MAX,
            role: None,
        });
    }
    let auth_header = headers
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .ok_or(AuthError::MissingHeader)?;
    let token = extract_bearer(auth_header)?;
    validate_jwt(token, jwt_secret)
}

pub(crate) fn auth_error_response(err: AuthError) -> Response {
    let body = serde_json::json!({"error": err.to_string()});
    (StatusCode::UNAUTHORIZED, Json(body)).into_response()
}

fn proxy_error_response(err: ProxyError) -> Response {
    let status = match &err {
        ProxyError::ToolNotFound(_) => StatusCode::NOT_FOUND,
        ProxyError::Timeout(_) => StatusCode::GATEWAY_TIMEOUT,
        _ => StatusCode::BAD_GATEWAY,
    };
    let body = serde_json::json!({"error": err.to_string()});
    (status, Json(body)).into_response()
}

/// Build response headers for rate limit information.
fn rate_limit_headers(result: &RateLimitResult) -> HeaderMap {
    let mut headers = HeaderMap::new();
    // X-RateLimit-Limit
    if let Ok(v) = HeaderValue::from_str(&result.limit.to_string()) {
        headers.insert("X-RateLimit-Limit", v);
    }
    // X-RateLimit-Remaining
    if let Ok(v) = HeaderValue::from_str(&result.remaining.to_string()) {
        headers.insert("X-RateLimit-Remaining", v);
    }
    // X-RateLimit-Reset
    if let Ok(v) = HeaderValue::from_str(&result.reset_at.to_string()) {
        headers.insert("X-RateLimit-Reset", v);
    }
    headers
}

/// Build a 429 Too Many Requests response.
fn rate_limit_exceeded_response(result: &RateLimitResult, call_type: CallType) -> Response {
    let kind = match call_type {
        CallType::Llm | CallType::Deep => "llm",
        CallType::Tool => "tool",
    };
    let body = serde_json::json!({
        "error": format!("Daily {kind} limit reached. Resets at midnight UTC.")
    });
    let mut headers = rate_limit_headers(result);
    if let Ok(v) = HeaderValue::from_str(&result.retry_after_secs.to_string()) {
        headers.insert("Retry-After", v);
    }
    let mut response = (StatusCode::TOO_MANY_REQUESTS, Json(body)).into_response();
    response.headers_mut().extend(headers);
    response
}

// ── /v1/tools/list ────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct ToolListResponse {
    pub tools: Vec<ToolEntry>,
    pub count: usize,
}

pub async fn tools_list(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    let claims = match auth_check(&headers, &state.jwt_secret) {
        Ok(c) => c,
        Err(e) => return auth_error_response(e),
    };

    // Tool calls (list) count against the tool budget.
    let role = UserRole::from_claim(claims.role.as_deref());
    let rl_result = {
        let mut rl = state.rate_limiter.lock().await;
        rl.check_and_record(&claims.sub, role, CallType::Tool)
    };

    if !rl_result.allowed {
        return rate_limit_exceeded_response(&rl_result, CallType::Tool);
    }

    let rl_headers = rate_limit_headers(&rl_result);
    match state.proxy.tool_list().await {
        Ok(tools) => {
            let count = tools.len();
            let mut response = Json(ToolListResponse { tools, count }).into_response();
            response.headers_mut().extend(rl_headers);
            response
        }
        Err(e) => {
            warn!("tools/list error: {e}");
            proxy_error_response(e)
        }
    }
}

// ── /v1/tools/call ────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct ToolCallRequest {
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}

#[derive(Serialize)]
pub struct ToolCallResponse {
    pub result: String,
    pub source: Option<String>,
}

pub async fn tools_call(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ToolCallRequest>,
) -> Response {
    let claims = match auth_check(&headers, &state.jwt_secret) {
        Ok(c) => c,
        Err(e) => return auth_error_response(e),
    };

    let role = UserRole::from_claim(claims.role.as_deref());
    let rl_result = {
        let mut rl = state.rate_limiter.lock().await;
        rl.check_and_record(&claims.sub, role, CallType::Tool)
    };

    if !rl_result.allowed {
        return rate_limit_exceeded_response(&rl_result, CallType::Tool);
    }

    let rl_headers = rate_limit_headers(&rl_result);
    match state.proxy.tool_call(&req.name, req.arguments).await {
        Ok((result, source)) => {
            let mut response = Json(ToolCallResponse {
                result,
                source: Some(source.to_string()),
            }).into_response();
            response.headers_mut().extend(rl_headers);
            response
        }
        Err(e) => {
            warn!("tools/call error for {}: {e}", req.name);
            proxy_error_response(e)
        }
    }
}

// ── /v1/tools/discover ────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct ToolDiscoverRequest {
    pub query: String,
    #[serde(default = "default_max_results")]
    pub max_results: usize,
}

fn default_max_results() -> usize { 10 }

#[derive(Serialize)]
pub struct ToolDiscoverResponse {
    pub tools: Vec<ToolEntry>,
    pub query: String,
    pub count: usize,
}

pub async fn tools_discover(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ToolDiscoverRequest>,
) -> Response {
    let claims = match auth_check(&headers, &state.jwt_secret) {
        Ok(c) => c,
        Err(e) => return auth_error_response(e),
    };

    let role = UserRole::from_claim(claims.role.as_deref());
    let rl_result = {
        let mut rl = state.rate_limiter.lock().await;
        rl.check_and_record(&claims.sub, role, CallType::Tool)
    };

    if !rl_result.allowed {
        return rate_limit_exceeded_response(&rl_result, CallType::Tool);
    }

    let rl_headers = rate_limit_headers(&rl_result);
    let max = req.max_results.min(100); // cap at 100
    match state.proxy.tool_discover(&req.query, max).await {
        Ok(tools) => {
            let count = tools.len();
            let query = req.query.clone();
            let mut response = Json(ToolDiscoverResponse { tools, query, count }).into_response();
            response.headers_mut().extend(rl_headers);
            response
        }
        Err(e) => {
            warn!("tools/discover error: {e}");
            proxy_error_response(e)
        }
    }
}

// ── /v1/agent/execute ─────────────────────────────────────────────────────────

/// POST /v1/agent/execute — run the guarded agentic tool-calling loop on Chord.
///
/// Requires JWT auth.  Accepts an `AgenticRequest` (full conversation context)
/// and returns an `AgenticResponse` (final text + metadata-only execution log).
///
/// Tool arguments and raw results never leave Chord — only the final answer and
/// metadata (step type, tool name, duration, status) are returned.
pub async fn agent_execute(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<AgenticRequest>,
) -> Response {
    let claims = match auth_check(&headers, &state.jwt_secret) {
        Ok(c) => c,
        Err(e) => return auth_error_response(e),
    };

    // Count an agentic execution against the LLM budget (it makes LLM calls).
    let role = UserRole::from_claim(claims.role.as_deref());
    let rl_result = {
        let mut rl = state.rate_limiter.lock().await;
        rl.check_and_record(&claims.sub, role, CallType::Llm)
    };

    if !rl_result.allowed {
        return rate_limit_exceeded_response(&rl_result, CallType::Llm);
    }

    let rl_headers = rate_limit_headers(&rl_result);

    // RESP-04: stream ProgressEvents as SSE when the caller requests it; otherwise
    // return the legacy single buffered JSON response.
    if req.stream {
        use crate::agentic::streaming::ProgressEvent;
        use futures_util::StreamExt;
        use tokio_stream::wrappers::UnboundedReceiverStream;

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<ProgressEvent>();
        let exec = state.agentic_executor.clone();
        tokio::spawn(async move {
            let _ = exec.execute(req, Some(tx)).await;
        });

        let stream = UnboundedReceiverStream::new(rx).map(|ev| {
            Ok::<axum::response::sse::Event, std::convert::Infallible>(
                axum::response::sse::Event::default()
                    .json_data(&ev)
                    .unwrap_or_else(|_| axum::response::sse::Event::default().data("{}")),
            )
        });

        let mut response = axum::response::sse::Sse::new(stream).into_response();
        response.headers_mut().extend(rl_headers);
        return response;
    }

    let resp: AgenticResponse = state.agentic_executor.execute(req, None).await;

    let mut response = Json(resp).into_response();
    response.headers_mut().extend(rl_headers);
    response
}

// ── /v1/chat/completions ──────────────────────────────────────────────────────

/// Hop-by-hop headers that must NOT be forwarded between connections (RFC 7230 §6.1),
/// plus length/encoding headers that reqwest recomputes for the upstream request.
fn is_unforwardable_request_header(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        "host"
            | "content-length"
            | "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
            // Auth header is the proxy's own JWT, never forwarded to the LLM backend.
            | "authorization"
    )
}

/// POST /v1/chat/completions — OpenAI-compatible LLM inference proxy.
///
/// Validates JWT (same auth as every other endpoint), applies the per-user LLM
/// rate limit, then forwards the request body verbatim to `CHORD_LLM_URL`
/// (the local Ollama OpenAI-compatible endpoint). Supports both non-streaming
/// (JSON) and streaming (`stream: true` → `text/event-stream`) responses by
/// streaming the upstream body straight back to the caller.
///
/// If `CHORD_LLM_URL` is not configured, returns 503 Service Unavailable.
///
/// ## TIER-02 pull-on-miss
/// Immediately after the model alias is resolved and before the upstream request,
/// the resolved model's tier is looked up in the [`AppState::model_registry`]. If
/// (and only if) it is [`StorageTier::Cold`], the model is transparently pulled
/// from the archive via [`PullCoordinator::ensure_local`] before inference. Hot,
/// Warm, and registry-*unknown* models are passed through unchanged (no pull, no
/// regression for models the registry doesn't track). Known models always get
/// their `last_requested` timestamp bumped.
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let start = Instant::now();

    let claims = match auth_check(&headers, &state.jwt_secret) {
        Ok(c) => c,
        Err(e) => {
            // Record the auth failure (hashes the token; never stores it).
            let raw = headers
                .get("Authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|h| extract_bearer(h).ok());
            state
                .audit_logger
                .log_auth_failure(raw, start.elapsed().as_millis() as u64);
            return auth_error_response(e);
        }
    };

    // Endpoint disabled when no upstream LLM URL is configured.
    let Some(llm_url) = state.llm_backend_url.clone() else {
        let model = parse_model_from_body(&body);
        state.audit_logger.log_llm_call(
            &claims.sub,
            &model,
            start.elapsed().as_millis() as u64,
            AuditStatus::Error,
            Some("CHORD_LLM_URL not configured".into()),
        );
        let resp_body = serde_json::json!({
            "error": "LLM backend not configured (CHORD_LLM_URL unset)"
        });
        return (StatusCode::SERVICE_UNAVAILABLE, Json(resp_body)).into_response();
    };

    // LLM inference counts against the user's LLM budget.
    let role = UserRole::from_claim(claims.role.as_deref());
    let rl_result = {
        let mut rl = state.rate_limiter.lock().await;
        rl.check_and_record(&claims.sub, role, CallType::Llm)
    };
    if !rl_result.allowed {
        return rate_limit_exceeded_response(&rl_result, CallType::Llm);
    }
    let rl_headers = rate_limit_headers(&rl_result);

    // Resolve any model alias (e.g. lumina-fast → gpt-oss:20b) before forwarding.
    // lumina-core sends alias names that Ollama does not know; without this the
    // upstream returns HTTP 404 "model lumina-fast not found" (the F1 outage).
    // `forward_body` carries the rewritten model when an alias matched, else the
    // original bytes untouched; `model` is the resolved name (for audit logging).
    let requested_model = parse_model_from_body(&body);
    let resolved_model =
        crate::config::resolve_model_alias(&state.model_aliases, &requested_model).to_string();

    // The registry is keyed by the fully-tagged Ollama name (e.g. "model:latest").
    // Ollama treats an untagged request as ":latest", so normalize before any
    // registry lookup/pull — otherwise an untagged request misses the registry
    // entry and a cold model is never pulled. The untagged `resolved_model` is
    // still what we forward upstream (Ollama resolves the tag itself).
    let registry_key = if resolved_model.contains(':') {
        resolved_model.clone()
    } else {
        format!("{resolved_model}:latest")
    };

    // ── TIER-02 pull-on-miss ──────────────────────────────────────────────────
    // Look up the resolved model's tier under a brief lock, bumping its
    // last_requested timestamp for any model the registry knows. ONLY a Cold
    // model triggers an archive pull; Hot/Warm/unknown pass straight through to
    // the upstream below (no regression for models the registry doesn't track).
    // The lock is released before the (potentially long) pull and the upstream
    // HTTP call.
    let needs_pull = {
        use crate::models::registry::StorageTier;
        let mut reg = state.model_registry.lock().await;
        reg.update_last_requested(&registry_key);
        matches!(
            reg.get(&registry_key).map(|r| r.tier.clone()),
            Some(StorageTier::Cold)
        )
    };
    if needs_pull {
        if let Err(e) = state
            .pull_coordinator
            .ensure_local(&registry_key, None)
            .await
        {
            warn!(
                "chat/completions: cold model {resolved_model} could not be retrieved: {e}"
            );
            state.audit_logger.log_llm_call(
                &claims.sub,
                &resolved_model,
                start.elapsed().as_millis() as u64,
                AuditStatus::Error,
                Some(format!("archive pull failed: {e}")),
            );
            let resp_body = serde_json::json!({
                "error": format!("model could not be retrieved from archive: {e}")
            });
            return (StatusCode::SERVICE_UNAVAILABLE, Json(resp_body)).into_response();
        }
    }

    // ── P5 tag-aware routing ──────────────────────────────────────────────────
    // Forward to the model's tagged backend (GPU vs CPU), starting an on-demand
    // backend if needed. Untagged models resolve to the default backend (whose
    // URL matches CHORD_LLM_URL), so behavior is unchanged until models are
    // tagged. On any failure we fall back to CHORD_LLM_URL.
    // `bearer_key` is `Some` only for backends with `api_key_env` set (e.g.
    // OpenRouter) — re-injected as an outbound Authorization header below,
    // after the inbound-header copy loop strips the caller's own JWT.
    let (llm_url, bearer_key) = crate::models::routing::resolve_and_ensure(
        &state.model_registry,
        &registry_key,
        &resolved_model,
    )
    .await
    .unwrap_or((llm_url, None));

    // ── YARN-06: per-request thinking honoring ──────────────────────────────
    // Harmony (THINK-01/02) may send an optional top-level `thinking: "on"|"off"`
    // hint. Chord makes NO decision about WHEN to think — that step-type
    // heuristic is entirely Harmony's; this only resolves whether the hint CAN
    // be honored (the model is supporting + validated, per its
    // `serving_profile.env_json.thinking` block) and, if so, honors it for
    // THIS request via the llama.cpp/Qwen3-template per-request toggle
    // (`chat_template_kwargs.enable_thinking`) — an already-warm/resident model
    // reads this from the request body on every call, no relaunch required.
    // A hint against a non-supporting or unvalidated model is ignored (model
    // default mode) with a note, never an error; a malformed value degrades the
    // same way. See `crate::serving::profile::resolve_thinking_request`.
    let raw_thinking = parse_thinking_from_body(&body);
    let (requested_thinking, malformed_thinking_note) =
        crate::serving::profile::parse_thinking_request(raw_thinking.as_deref());
    let thinking_capability = {
        let routing = state.routing_map.lock().await;
        routing
            .get(&terminus_rs::intake::serving::ModelId::from(
                resolved_model.as_str(),
            ))
            .and_then(|entry| entry.env.thinking)
    };
    let mut thinking_decision = crate::serving::profile::resolve_thinking_request(
        requested_thinking,
        thinking_capability.as_ref(),
    );
    if let Some(note) = malformed_thinking_note {
        // A malformed value takes precedence over resolve_thinking_request's own
        // (empty, since `requested == None` short-circuits) note — the caller
        // sent SOMETHING, it just wasn't recognized.
        thinking_decision.note = Some(note);
    }
    if let Some(note) = &thinking_decision.note {
        tracing::debug!(model = %resolved_model, "chat/completions: thinking hint not honored as requested: {note}");
    }

    let need_model_rewrite = resolved_model != requested_model;
    let need_any_rewrite = need_model_rewrite || raw_thinking.is_some();
    let (forward_body, model) = if need_any_rewrite {
        let new_model = need_model_rewrite.then(|| resolved_model.as_str());
        match apply_request_rewrites(&body, new_model, &thinking_decision) {
            Some(rewritten) => (
                axum::body::Bytes::from(rewritten),
                if need_model_rewrite {
                    resolved_model.clone()
                } else {
                    requested_model.clone()
                },
            ),
            // Body wasn't valid JSON we could rewrite — forward verbatim, log original.
            None => (body.clone(), requested_model.clone()),
        }
    } else {
        (body.clone(), requested_model.clone())
    };
    let mut upstream_req = state.http_client.post(&llm_url).body(forward_body);
    let mut had_content_type = false;
    for (name, value) in headers.iter() {
        if is_unforwardable_request_header(name) {
            continue;
        }
        if name.as_str() == "content-type" {
            had_content_type = true;
        }
        upstream_req = upstream_req.header(name, value);
    }
    if !had_content_type {
        upstream_req = upstream_req.header("content-type", "application/json");
    }
    // Re-add an Authorization header ONLY for backends that need one (e.g.
    // OpenRouter) — the loop above stripped the caller's own JWT via
    // `is_unforwardable_request_header`, so this can never leak the caller's
    // token to the upstream provider; it's a fresh value from `bearer_key`,
    // resolved by `resolve_and_ensure` from the backend's own `api_key_env`.
    if let Some(key) = &bearer_key {
        upstream_req = upstream_req.header("authorization", format!("Bearer {key}"));
    }

    let upstream = match upstream_req.send().await {
        Ok(r) => r,
        Err(e) => {
            warn!("chat/completions: upstream request to {llm_url} failed: {e}");
            state.audit_logger.log_llm_call(
                &claims.sub,
                &model,
                start.elapsed().as_millis() as u64,
                AuditStatus::Error,
                Some(format!("upstream request failed: {e}")),
            );
            let resp_body = serde_json::json!({
                "error": format!("LLM backend unreachable: {e}")
            });
            let mut response =
                (StatusCode::BAD_GATEWAY, Json(resp_body)).into_response();
            response.headers_mut().extend(rl_headers);
            return response;
        }
    };

    let status = upstream.status();
    // Capture the upstream content-type so streaming (text/event-stream) is preserved.
    let content_type = upstream
        .headers()
        .get("content-type")
        .cloned()
        .unwrap_or_else(|| HeaderValue::from_static("application/json"));

    let audit_status = if status.is_success() {
        AuditStatus::Success
    } else {
        AuditStatus::Error
    };
    state.audit_logger.log_llm_call(
        &claims.sub,
        &model,
        start.elapsed().as_millis() as u64,
        audit_status,
        if status.is_success() {
            None
        } else {
            Some(format!("upstream returned HTTP {status}"))
        },
    );

    // Stream the upstream body straight back to the caller. This passes through
    // both non-streaming JSON and streaming SSE (text/event-stream) untouched.
    use futures_util::TryStreamExt;
    let stream = upstream
        .bytes_stream()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e));
    let body = Body::from_stream(stream);

    let mut response = Response::builder()
        .status(status)
        .header("content-type", content_type)
        .body(body)
        .unwrap_or_else(|e| {
            warn!("chat/completions: failed to build response: {e}");
            (StatusCode::INTERNAL_SERVER_ERROR, "response build error").into_response()
        });
    response.headers_mut().extend(rl_headers);
    response
}

/// Extract the `model` field from an OpenAI-style request body for audit logging.
/// Returns `"unknown"` when the body is not valid JSON or has no model field.
fn parse_model_from_body(body: &[u8]) -> String {
    serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|v| v.get("model").and_then(|m| m.as_str().map(String::from)))
        .unwrap_or_else(|| "unknown".to_string())
}

/// YARN-06: extract the optional top-level `thinking` string hint from an
/// incoming request body (Chord's own request-time contract field, e.g.
/// `{"thinking": "on"}`) — NOT an OpenAI/upstream-standard field, and never
/// forwarded upstream verbatim (see [`apply_request_rewrites`]). `None` when
/// absent, the body isn't a JSON object, or the value isn't a string (an
/// absent/non-string value is treated as "no hint sent", matching
/// [`crate::serving::profile::parse_thinking_request`]'s `None` input case;
/// a present-but-unrecognized STRING value like `"maybe"` is a different,
/// malformed case that function itself handles).
fn parse_thinking_from_body(body: &[u8]) -> Option<String> {
    serde_json::from_slice::<Value>(body)
        .ok()?
        .get("thinking")
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

/// YARN-06: apply the resolved per-request thinking decision to the outgoing
/// body, and (when needed) the existing model-alias rewrite — both are small,
/// targeted mutations of the same JSON object, so they share one
/// parse/re-serialize pass. Returns `None` when the body isn't a JSON object
/// (caller forwards verbatim; nothing to honor/rewrite).
///
/// The Chord-specific top-level `thinking` field is ALWAYS stripped before
/// forwarding — it is Chord's own request-time hint, not something an
/// upstream OpenAI-compatible backend understands. When the decision is
/// `ForcedOn`/`ForcedOff` (only ever returned for a model that is supporting +
/// validated — see [`crate::serving::profile::resolve_thinking_request`]),
/// the corresponding llama.cpp/Qwen3-chat-template toggle
/// (`chat_template_kwargs.enable_thinking`) is set/merged into the body: this
/// is the actual per-request honoring mechanism for an already-warm/resident
/// model — llama-server (and vLLM/SGLang) read `enable_thinking` from the
/// chat-template kwargs on EVERY request, no relaunch required. `ModelDefault`
/// leaves any caller-supplied `chat_template_kwargs` entirely untouched.
fn apply_request_rewrites(
    body: &[u8],
    new_model: Option<&str>,
    decision: &crate::serving::profile::ThinkingDecision,
) -> Option<Vec<u8>> {
    use crate::serving::profile::EffectiveThinking;

    let mut v: Value = serde_json::from_slice(body).ok()?;
    let obj = v.as_object_mut()?;

    if let Some(model) = new_model {
        obj.insert("model".to_string(), Value::String(model.to_string()));
    }

    // Chord's own hint never reaches the backend verbatim.
    obj.remove("thinking");

    match decision.effective {
        EffectiveThinking::ModelDefault => {}
        EffectiveThinking::ForcedOn | EffectiveThinking::ForcedOff => {
            let enable = matches!(decision.effective, EffectiveThinking::ForcedOn);
            let kwargs = obj
                .entry("chat_template_kwargs".to_string())
                .or_insert_with(|| Value::Object(serde_json::Map::new()));
            if !kwargs.is_object() {
                *kwargs = Value::Object(serde_json::Map::new());
            }
            if let Some(kwargs_obj) = kwargs.as_object_mut() {
                kwargs_obj.insert("enable_thinking".to_string(), Value::Bool(enable));
            }
        }
    }

    serde_json::to_vec(&v).ok()
}

// ── /health ───────────────────────────────────────────────────────────────────

pub async fn health() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "service": "chord-proxy",
        "version": crate::version::version(),
        "commit": crate::version::commit(),
        "terminus_rs": terminus_rs::VERSION,
    }))
}

// ── /v1/audit/summary ─────────────────────────────────────────────────────────

/// GET /v1/audit/summary — aggregate counts for the last 24h.
/// No auth required (returns aggregate counts only, no user identities).
pub async fn audit_summary(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let mut summary: AuditSummary = state.audit_logger.daily_summary();
    summary.window_hours = 24;
    Json(summary)
}

/// Build the Axum router.
pub fn build_router(state: Arc<AppState>) -> axum::Router {
    axum::Router::new()
        .route("/health", axum::routing::get(health))
        .route("/v1/audit/summary", axum::routing::get(audit_summary))
        .route("/v1/tools/list", axum::routing::post(tools_list))
        .route("/v1/tools/call", axum::routing::post(tools_call))
        .route("/v1/tools/discover", axum::routing::post(tools_discover))
        .route("/v1/agent/execute", axum::routing::post(agent_execute))
        .route("/v1/chat/completions", axum::routing::post(chat_completions))
        .route("/v1/infer", axum::routing::post(infer))
        // Sweep-status observability (no auth, same bar as /health and
        // /v1/audit/summary — aggregate health, no identities/secrets).
        .merge(crate::sweep_status::api::sweep_status_routes())
        .route("/v1/coding/select", axum::routing::post(crate::coding_proxy::coding_select))
        .with_state(state)
}

/// `POST /v1/infer` request: run one prompt on the model's tagged backend and
/// return normalized metrics (throughput/TTFT/tokens/VRAM + backend attribution).
#[derive(serde::Deserialize)]
pub struct InferRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub timeout_secs: Option<u64>,
}

/// `POST /v1/infer` — surfaces the same backend-aware, metric-capturing
/// inference the harness uses, so external clients get per-backend metrics
/// without hitting backends directly. JWT-gated like the other endpoints.
pub async fn infer(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<InferRequest>,
) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }
    let timeout = std::time::Duration::from_secs(req.timeout_secs.unwrap_or(300));
    let metrics = terminus_rs::intake::infer::infer_with_metrics(
        &state.http_client,
        &req.model,
        &req.prompt,
        timeout,
    )
    .await;
    Json(metrics).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Method, Request};
    use tower::ServiceExt;

    use crate::agentic::AgenticExecutor;
    use crate::audit::AuditLogger;
    use crate::config::{Config, RateLimitConfig};
    use crate::mcp_proxy::{FallbackRegistry, FallbackTool, McpProxy};

    struct PingTool;
    #[async_trait::async_trait]
    impl FallbackTool for PingTool {
        fn name(&self) -> &str { "ping" }
        fn description(&self) -> &str { "Ping" }
        fn parameters(&self) -> Value { serde_json::json!({}) }
        async fn execute(&self, _: Value) -> Result<String, ProxyError> { Ok("pong".into()) }
    }

    fn default_rate_config() -> RateLimitConfig {
        RateLimitConfig {
            user_llm_limit: 200,
            user_tool_limit: 500,
            user_deep_limit: 50,
            guest_llm_limit: 20,
            guest_tool_limit: 50,
            guest_deep_limit: 5,
        }
    }

    /// Build an empty model registry (pointing at throwaway paths) and a pull
    /// coordinator over it, for AppState test constructors that don't care about
    /// the pull-on-miss path. Returns `(registry, coordinator)` sharing the same
    /// inner registry. Paths are non-existent on purpose: an empty registry knows
    /// no models, so `chat_completions` treats every model as "unknown" → pass-
    /// through, exactly the legacy behaviour these tests assert.
    fn empty_model_state() -> (
        Arc<Mutex<crate::models::registry::ModelRegistry>>,
        Arc<crate::models::transfer::PullCoordinator>,
    ) {
        use crate::models::registry::ModelRegistry;
        use crate::models::transfer::PullCoordinator;
        let reg = ModelRegistry::new(
            std::path::PathBuf::from("/nonexistent/chord-test-registry.json"),
            std::path::PathBuf::from("/nonexistent/local"),
            std::path::PathBuf::from("/nonexistent/archive"),
            vec![],
        );
        let registry = Arc::new(Mutex::new(reg));
        let coordinator = Arc::new(PullCoordinator::new(
            registry.clone(),
            std::time::Duration::from_secs(5),
        ));
        (registry, coordinator)
    }

    /// An empty [`crate::serving::profile::RoutingMap`] for state builders that
    /// don't care about thinking-capability routing — every lookup misses,
    /// `thinking_available()` is `false` (same fail-open default as production
    /// when the intake DB is unconfigured).
    fn empty_routing_map() -> Arc<Mutex<crate::serving::profile::RoutingMap>> {
        Arc::new(Mutex::new(crate::serving::profile::RoutingMap::empty()))
    }

    /// An unconfigured coding-profile source for state builders that don't
    /// exercise CPROX-03 — `POST /v1/coding/select` reports 503 `NotConfigured`.
    fn empty_coding_profile_source() -> crate::coding_proxy::SharedCodingProfileSource {
        Arc::new(Mutex::new(None))
    }

    fn test_state(mcp_url: String) -> Arc<AppState> {
        let mut reg = FallbackRegistry::new();
        reg.register(Box::new(PingTool));
        let config = Config {
            mcp_backend_url: mcp_url,
            jwt_secret: String::new(), // auth disabled for most tests
            tool_timeout_secs: 5,
            catalog_cache_secs: 300,
            listen_port: 9099,
            control_port: 8090,
            rate_limits: default_rate_config(),
            llm_backend_url: None,
            model_aliases: std::collections::HashMap::new(),
            model_archive_path: "/var/lib/model-archive".into(),
            model_local_path: "/opt/ollama-models".into(),
            model_protected: vec![],
            model_pull_timeout_secs: 600,
            model_registry_path: "<path>/model-registry.json".into(),
            model_disk_pressure_percent: 80,
            model_sweep_interval_secs: 1800,
            model_warm_cooldown_hours: 168,
            model_source_allowlist: Vec::new(),
            outbound_proxy: None,
            runtime_telemetry_off: true,
        };
        let proxy = McpProxy::new(&config, Arc::new(reg));
        let proxy_arc = Arc::new(McpProxy::new(
            &Config {
                mcp_backend_url: "http://does-not-exist:9999".into(),
                jwt_secret: String::new(),
                tool_timeout_secs: 5,
                catalog_cache_secs: 300,
                listen_port: 9099,
                control_port: 8090,
                rate_limits: default_rate_config(),
                llm_backend_url: None,
                model_aliases: std::collections::HashMap::new(),
                model_archive_path: "/var/lib/model-archive".into(),
                model_local_path: "/opt/ollama-models".into(),
                model_protected: vec![],
                model_pull_timeout_secs: 600,
                model_registry_path: "<path>/model-registry.json".into(),
                model_disk_pressure_percent: 80,
                model_sweep_interval_secs: 1800,
            model_warm_cooldown_hours: 168,
            model_source_allowlist: Vec::new(),
            outbound_proxy: None,
            runtime_telemetry_off: true,
            },
            Arc::new(FallbackRegistry::new()),
        ));
        let agentic_executor = Arc::new(AgenticExecutor::new(proxy_arc));
        let rate_limiter = Arc::new(Mutex::new(ProxyRateLimiter::new(default_rate_config())));
        let audit_logger = Arc::new(AuditLogger::new(std::path::PathBuf::from("/dev/null")));
        let (model_registry, pull_coordinator) = empty_model_state();
        Arc::new(AppState { proxy, jwt_secret: String::new(), audit_logger, rate_limiter, agentic_executor, llm_backend_url: None, model_aliases: std::collections::HashMap::new(), http_client: reqwest::Client::new(), model_registry, pull_coordinator, local_evictor: std::sync::Arc::new(crate::models::eviction::FsLocalEvictor::new(std::path::PathBuf::from("/tmp"))), disk_op_lock: crate::models::eviction::new_disk_op_lock(), disk_probe: std::sync::Arc::new(crate::models::transfer::StatvfsProbe), disk_pressure_percent: 80, model_warm_cooldown_hours: 168, routing_map: empty_routing_map(), coding_profile_source: empty_coding_profile_source() })
    }

    /// State with auth disabled and an explicit upstream LLM URL for chat/completions tests.
    fn test_state_with_llm(llm_url: Option<String>) -> Arc<AppState> {
        test_state_with_llm_aliases(llm_url, std::collections::HashMap::new())
    }

    /// Like `test_state_with_llm` but with an explicit model alias map so alias
    /// rewriting in the chat/completions proxy can be exercised.
    fn test_state_with_llm_aliases(
        llm_url: Option<String>,
        model_aliases: std::collections::HashMap<String, String>,
    ) -> Arc<AppState> {
        test_state_with_llm_aliases_and_routing(llm_url, model_aliases, empty_routing_map())
    }

    /// Like `test_state_with_llm_aliases` but with an explicit
    /// [`crate::serving::profile::RoutingMap`] so YARN-06 thinking-capability
    /// routing (capability lookup + per-request honoring) can be exercised.
    fn test_state_with_llm_aliases_and_routing(
        llm_url: Option<String>,
        model_aliases: std::collections::HashMap<String, String>,
        routing_map: Arc<Mutex<crate::serving::profile::RoutingMap>>,
    ) -> Arc<AppState> {
        let config = Config {
            mcp_backend_url: "http://does-not-exist:9999".into(),
            jwt_secret: String::new(),
            tool_timeout_secs: 5,
            catalog_cache_secs: 300,
            listen_port: 9099,
            control_port: 8090,
            rate_limits: default_rate_config(),
            llm_backend_url: llm_url.clone(),
            model_aliases: model_aliases.clone(),
            model_archive_path: "/var/lib/model-archive".into(),
            model_local_path: "/opt/ollama-models".into(),
            model_protected: vec![],
            model_pull_timeout_secs: 600,
            model_registry_path: "<path>/model-registry.json".into(),
            model_disk_pressure_percent: 80,
            model_sweep_interval_secs: 1800,
            model_warm_cooldown_hours: 168,
            model_source_allowlist: Vec::new(),
            outbound_proxy: None,
            runtime_telemetry_off: true,
        };
        let proxy = McpProxy::new(&config, Arc::new(FallbackRegistry::new()));
        let proxy_arc = Arc::new(McpProxy::new(&config, Arc::new(FallbackRegistry::new())));
        let agentic_executor = Arc::new(AgenticExecutor::new(proxy_arc));
        let rate_limiter = Arc::new(Mutex::new(ProxyRateLimiter::new(default_rate_config())));
        let audit_logger = Arc::new(AuditLogger::new(std::path::PathBuf::from("/dev/null")));
        let (model_registry, pull_coordinator) = empty_model_state();
        Arc::new(AppState {
            proxy,
            jwt_secret: String::new(),
            audit_logger,
            rate_limiter,
            agentic_executor,
            llm_backend_url: llm_url,
            model_aliases,
            http_client: reqwest::Client::new(),
            model_registry,
            pull_coordinator,
            local_evictor: std::sync::Arc::new(crate::models::eviction::FsLocalEvictor::new(std::path::PathBuf::from("/tmp"))), disk_op_lock: crate::models::eviction::new_disk_op_lock(), disk_probe: std::sync::Arc::new(crate::models::transfer::StatvfsProbe), disk_pressure_percent: 80, model_warm_cooldown_hours: 168,
            routing_map,
            coding_profile_source: empty_coding_profile_source(),
        })
    }

    fn test_state_with_secret(mcp_url: String, secret: String) -> Arc<AppState> {
        let config = Config {
            mcp_backend_url: mcp_url,
            jwt_secret: secret.clone(),
            tool_timeout_secs: 5,
            catalog_cache_secs: 300,
            listen_port: 9099,
            control_port: 8090,
            rate_limits: default_rate_config(),
            llm_backend_url: None,
            model_aliases: std::collections::HashMap::new(),
            model_archive_path: "/var/lib/model-archive".into(),
            model_local_path: "/opt/ollama-models".into(),
            model_protected: vec![],
            model_pull_timeout_secs: 600,
            model_registry_path: "<path>/model-registry.json".into(),
            model_disk_pressure_percent: 80,
            model_sweep_interval_secs: 1800,
            model_warm_cooldown_hours: 168,
            model_source_allowlist: Vec::new(),
            outbound_proxy: None,
            runtime_telemetry_off: true,
        };
        let proxy = McpProxy::new(&config, Arc::new(FallbackRegistry::new()));
        let proxy_arc = Arc::new(McpProxy::new(
            &Config {
                mcp_backend_url: "http://does-not-exist:9999".into(),
                jwt_secret: String::new(),
                tool_timeout_secs: 5,
                catalog_cache_secs: 300,
                listen_port: 9099,
                control_port: 8090,
                rate_limits: default_rate_config(),
                llm_backend_url: None,
                model_aliases: std::collections::HashMap::new(),
                model_archive_path: "/var/lib/model-archive".into(),
                model_local_path: "/opt/ollama-models".into(),
                model_protected: vec![],
                model_pull_timeout_secs: 600,
                model_registry_path: "<path>/model-registry.json".into(),
                model_disk_pressure_percent: 80,
                model_sweep_interval_secs: 1800,
            model_warm_cooldown_hours: 168,
            model_source_allowlist: Vec::new(),
            outbound_proxy: None,
            runtime_telemetry_off: true,
            },
            Arc::new(FallbackRegistry::new()),
        ));
        let agentic_executor = Arc::new(AgenticExecutor::new(proxy_arc));
        let rate_limiter = Arc::new(Mutex::new(ProxyRateLimiter::new(default_rate_config())));
        let audit_logger = Arc::new(AuditLogger::new(std::path::PathBuf::from("/dev/null")));
        let (model_registry, pull_coordinator) = empty_model_state();
        Arc::new(AppState { proxy, jwt_secret: secret, audit_logger, rate_limiter, agentic_executor, llm_backend_url: None, model_aliases: std::collections::HashMap::new(), http_client: reqwest::Client::new(), model_registry, pull_coordinator, local_evictor: std::sync::Arc::new(crate::models::eviction::FsLocalEvictor::new(std::path::PathBuf::from("/tmp"))), disk_op_lock: crate::models::eviction::new_disk_op_lock(), disk_probe: std::sync::Arc::new(crate::models::transfer::StatvfsProbe), disk_pressure_percent: 80, model_warm_cooldown_hours: 168, routing_map: empty_routing_map(), coding_profile_source: empty_coding_profile_source() })
    }

    /// Build a state with a very tight user tool limit for rate limit tests.
    /// Auth is disabled → synthetic claim has no role → defaults to User role.
    fn test_state_tight_limits(mcp_url: String) -> Arc<AppState> {
        let mut reg = FallbackRegistry::new();
        reg.register(Box::new(PingTool));
        let tight = RateLimitConfig {
            user_llm_limit: 200,
            user_tool_limit: 2, // low limit for fast test (auth-disabled → User role)
            user_deep_limit: 50,
            guest_llm_limit: 3,
            guest_tool_limit: 2,
            guest_deep_limit: 1,
        };
        let config = Config {
            mcp_backend_url: mcp_url,
            jwt_secret: String::new(),
            tool_timeout_secs: 5,
            catalog_cache_secs: 300,
            listen_port: 9099,
            control_port: 8090,
            rate_limits: tight.clone(),
            llm_backend_url: None,
            model_aliases: std::collections::HashMap::new(),
            model_archive_path: "/var/lib/model-archive".into(),
            model_local_path: "/opt/ollama-models".into(),
            model_protected: vec![],
            model_pull_timeout_secs: 600,
            model_registry_path: "<path>/model-registry.json".into(),
            model_disk_pressure_percent: 80,
            model_sweep_interval_secs: 1800,
            model_warm_cooldown_hours: 168,
            model_source_allowlist: Vec::new(),
            outbound_proxy: None,
            runtime_telemetry_off: true,
        };
        let proxy = McpProxy::new(&config, Arc::new(reg));
        let proxy_arc = Arc::new(McpProxy::new(
            &Config {
                mcp_backend_url: "http://does-not-exist:9999".into(),
                jwt_secret: String::new(),
                tool_timeout_secs: 5,
                catalog_cache_secs: 300,
                listen_port: 9099,
                control_port: 8090,
                rate_limits: tight.clone(),
                llm_backend_url: None,
                model_aliases: std::collections::HashMap::new(),
                model_archive_path: "/var/lib/model-archive".into(),
                model_local_path: "/opt/ollama-models".into(),
                model_protected: vec![],
                model_pull_timeout_secs: 600,
                model_registry_path: "<path>/model-registry.json".into(),
                model_disk_pressure_percent: 80,
                model_sweep_interval_secs: 1800,
            model_warm_cooldown_hours: 168,
            model_source_allowlist: Vec::new(),
            outbound_proxy: None,
            runtime_telemetry_off: true,
            },
            Arc::new(FallbackRegistry::new()),
        ));
        let agentic_executor = Arc::new(AgenticExecutor::new(proxy_arc));
        let rate_limiter = Arc::new(Mutex::new(ProxyRateLimiter::new(tight)));
        let audit_logger = Arc::new(AuditLogger::new(std::path::PathBuf::from("/dev/null")));
        let (model_registry, pull_coordinator) = empty_model_state();
        Arc::new(AppState { proxy, jwt_secret: String::new(), audit_logger, rate_limiter, agentic_executor, llm_backend_url: None, model_aliases: std::collections::HashMap::new(), http_client: reqwest::Client::new(), model_registry, pull_coordinator, local_evictor: std::sync::Arc::new(crate::models::eviction::FsLocalEvictor::new(std::path::PathBuf::from("/tmp"))), disk_op_lock: crate::models::eviction::new_disk_op_lock(), disk_probe: std::sync::Arc::new(crate::models::transfer::StatvfsProbe), disk_pressure_percent: 80, model_warm_cooldown_hours: 168, routing_map: empty_routing_map(), coding_profile_source: empty_coding_profile_source() })
    }

    #[tokio::test]
    async fn test_health_endpoint_no_auth() {
        let state = test_state("http://does-not-exist:9999".into());
        let app = build_router(state);

        let req = Request::builder()
            .method(Method::GET)
            .uri("/health")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_health_includes_version_fields() {
        let state = test_state("http://does-not-exist:9999".into());
        let app = build_router(state);

        let req = Request::builder()
            .method(Method::GET)
            .uri("/health")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(&bytes).unwrap();

        // Compiled-in version fields must be present in the health payload.
        assert_eq!(body["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(body["terminus_rs"], terminus_rs::VERSION);
        assert!(body.get("commit").is_some(), "commit field present");
        assert_eq!(body["status"], "ok");
        assert_eq!(body["service"], "chord-proxy");
    }

    #[tokio::test]
    async fn test_tools_list_requires_auth() {
        let state = test_state_with_secret(
            "http://does-not-exist:9999".into(),
            "test-secret".into(),
        );
        let app = build_router(state);

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/tools/list")
            .header("Content-Type", "application/json")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_tools_call_requires_auth() {
        let state = test_state_with_secret(
            "http://does-not-exist:9999".into(),
            "test-secret".into(),
        );
        let app = build_router(state);

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/tools/call")
            .header("Content-Type", "application/json")
            .body(Body::from(r#"{"name":"ping","arguments":{}}"#))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_tools_discover_requires_auth() {
        let state = test_state_with_secret(
            "http://does-not-exist:9999".into(),
            "test-secret".into(),
        );
        let app = build_router(state);

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/tools/discover")
            .header("Content-Type", "application/json")
            .body(Body::from(r#"{"query":"ping"}"#))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_tools_list_no_auth_secret_returns_200() {
        let state = test_state("http://does-not-exist:9999".into());
        let app = build_router(state);

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/tools/list")
            .header("Content-Type", "application/json")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // Auth disabled (no secret) → should proceed; MCP down → returns Rust-only catalog
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_tools_call_rust_fallback_route() {
        let state = test_state("http://does-not-exist:9999".into());
        let app = build_router(state);

        let body = serde_json::to_string(&serde_json::json!({
            "name": "ping",
            "arguments": {}
        })).unwrap();

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/tools/call")
            .header("Content-Type", "application/json")
            .body(Body::from(body))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["result"], "pong");
    }

    #[tokio::test]
    async fn test_tools_call_not_found_returns_404() {
        let state = test_state("http://does-not-exist:9999".into());
        let app = build_router(state);

        let body = serde_json::to_string(&serde_json::json!({
            "name": "nonexistent_tool",
            "arguments": {}
        })).unwrap();

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/tools/call")
            .header("Content-Type", "application/json")
            .body(Body::from(body))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_tools_discover_returns_results() {
        let state = test_state("http://does-not-exist:9999".into());
        let app = build_router(state);

        let body = serde_json::to_string(&serde_json::json!({
            "query": "ping",
            "max_results": 5
        })).unwrap();

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/tools/discover")
            .header("Content-Type", "application/json")
            .body(Body::from(body))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(json["tools"].is_array());
        assert_eq!(json["query"], "ping");
    }

    // ── Rate limit header tests ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_rate_limit_headers_present_on_200() {
        let state = test_state("http://does-not-exist:9999".into());
        let app = build_router(state);

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/tools/list")
            .header("Content-Type", "application/json")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let headers = resp.headers();
        assert!(headers.contains_key("X-RateLimit-Limit"), "X-RateLimit-Limit must be present");
        assert!(headers.contains_key("X-RateLimit-Remaining"), "X-RateLimit-Remaining must be present");
        assert!(headers.contains_key("X-RateLimit-Reset"), "X-RateLimit-Reset must be present");
    }

    #[tokio::test]
    async fn test_rate_limit_exceeded_returns_429_with_retry_after() {
        let state = test_state_tight_limits("http://does-not-exist:9999".into());
        let app = build_router(state);

        // Exhaust the 2-call guest tool limit, then verify 429.
        for _ in 0..2 {
            let req = Request::builder()
                .method(Method::POST)
                .uri("/v1/tools/list")
                .header("Content-Type", "application/json")
                .body(Body::empty())
                .unwrap();
            let _ = app.clone().oneshot(req).await.unwrap();
        }

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/tools/list")
            .header("Content-Type", "application/json")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);

        let headers = resp.headers();
        assert!(headers.contains_key("Retry-After"), "Retry-After must be present on 429");
        assert!(headers.contains_key("X-RateLimit-Limit"), "X-RateLimit-Limit must be present on 429");
        assert!(headers.contains_key("X-RateLimit-Remaining"), "X-RateLimit-Remaining must be present on 429");
        assert!(headers.contains_key("X-RateLimit-Reset"), "X-RateLimit-Reset must be present on 429");

        // Body should contain error message.
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(json["error"].as_str().unwrap().contains("limit reached"));
    }

    // ── /v1/chat/completions tests ────────────────────────────────────────────

    fn chat_request_body(model: &str, stream: bool) -> String {
        serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": stream,
        })
        .to_string()
    }

    /// Like `chat_request_body` but with an optional top-level `thinking` hint
    /// (Chord's own request-time contract field — see `parse_thinking_from_body`).
    fn chat_request_body_with_thinking(model: &str, thinking: &str) -> String {
        serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false,
            "thinking": thinking,
        })
        .to_string()
    }

    /// YARN-06: a one-row [`crate::serving::profile::RoutingMap`] for `model`
    /// with the given `thinking` env_json fragment (or `"{}"` for no thinking
    /// block at all — a non-supporting model).
    fn routing_map_with(
        model: &str,
        thinking_env_json: &str,
    ) -> Arc<Mutex<crate::serving::profile::RoutingMap>> {
        use terminus_rs::intake::serving::{
            ExclusionReason, ModelId, RecheckTrigger, Runtime, ServingBackend, ServingProfile,
        };
        let row = ServingProfile {
            model_id: ModelId::from(model),
            backend_tag: ServingBackend::LlamaGpu,
            best_runtime: Runtime::LlamaCpp,
            env_json: thinking_env_json.into(),
            tok_s: Some(30.0),
            vram_or_ram_peak_gb: Some(8.0),
            cold_load_s: Some(10.0),
            keep_warm: false,
            fallback_runtime: None,
            exclusion_reason: ExclusionReason::None,
            recheck_trigger: RecheckTrigger::None,
            provenance: None,
        };
        Arc::new(Mutex::new(crate::serving::profile::RoutingMap::load_from(
            vec![row],
        )))
    }

    /// YARN-06: `thinking:"on"` against a supporting + validated model — the
    /// forwarded body must carry `chat_template_kwargs.enable_thinking: true`
    /// (the actual per-request honoring mechanism for an already-warm model;
    /// llama-server/vLLM/SGLang read this chat-template kwarg on every
    /// request, no relaunch required) and the Chord-only `thinking` field must
    /// NOT be forwarded upstream.
    #[tokio::test]
    async fn test_chat_completions_thinking_on_sets_enable_thinking_true() {
        let server = httpmock::MockServer::start_async().await;
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/v1/chat/completions")
                .json_body_partial(r#"{"chat_template_kwargs":{"enable_thinking":true}}"#)
                .matches(|req| {
                    let body = req.body.as_deref().unwrap_or(&[]);
                    !String::from_utf8_lossy(body).contains("\"thinking\"")
                });
            then.status(200)
                .header("content-type", "application/json")
                .json_body(serde_json::json!({"choices": []}));
        });

        let routing = routing_map_with(
            "reasoner:30b",
            r#"{"thinking":{"supports_thinking":true,"validated":true}}"#,
        );
        let llm_url = format!("{}/v1/chat/completions", server.base_url());
        let state = test_state_with_llm_aliases_and_routing(
            Some(llm_url),
            std::collections::HashMap::new(),
            routing,
        );
        let app = build_router(state);

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("Content-Type", "application/json")
            .body(Body::from(chat_request_body_with_thinking(
                "reasoner:30b",
                "on",
            )))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        mock.assert_async().await;
    }

    /// YARN-06: `thinking:"off"` against a supporting + validated model — the
    /// forwarded body must carry `chat_template_kwargs.enable_thinking: false`.
    #[tokio::test]
    async fn test_chat_completions_thinking_off_sets_enable_thinking_false() {
        let server = httpmock::MockServer::start_async().await;
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/v1/chat/completions")
                .json_body_partial(r#"{"chat_template_kwargs":{"enable_thinking":false}}"#);
            then.status(200)
                .header("content-type", "application/json")
                .json_body(serde_json::json!({"choices": []}));
        });

        let routing = routing_map_with(
            "reasoner:30b",
            r#"{"thinking":{"supports_thinking":true,"validated":true}}"#,
        );
        let llm_url = format!("{}/v1/chat/completions", server.base_url());
        let state = test_state_with_llm_aliases_and_routing(
            Some(llm_url),
            std::collections::HashMap::new(),
            routing,
        );
        let app = build_router(state);

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("Content-Type", "application/json")
            .body(Body::from(chat_request_body_with_thinking(
                "reasoner:30b",
                "off",
            )))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        mock.assert_async().await;
    }

    /// NEGATIVE TEST: `thinking:"on"` against a model with NO thinking block at
    /// all — ignored, NOT an error (still 200), and no `chat_template_kwargs`
    /// is injected into the forwarded body.
    #[tokio::test]
    async fn test_chat_completions_thinking_ignored_for_non_supporting_model() {
        let server = httpmock::MockServer::start_async().await;
        // Trap: fails the test (unmatched → 404) if chat_template_kwargs leaks in.
        let trap = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/v1/chat/completions")
                .body_contains("chat_template_kwargs");
            then.status(200).json_body(serde_json::json!({"choices": []}));
        });
        let real = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/v1/chat/completions");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(serde_json::json!({"choices": []}));
        });

        // No thinking block at all in the profile ⇒ non-supporting.
        let routing = routing_map_with("plain:7b", "{}");
        let llm_url = format!("{}/v1/chat/completions", server.base_url());
        let state = test_state_with_llm_aliases_and_routing(
            Some(llm_url),
            std::collections::HashMap::new(),
            routing,
        );
        let app = build_router(state);

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("Content-Type", "application/json")
            .body(Body::from(chat_request_body_with_thinking("plain:7b", "on")))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "ignored hint must never error");
        trap.assert_hits_async(0).await;
        real.assert_hits_async(1).await;
    }

    /// EDGE CASE: `thinking:"on"` against a model whose thinking config is
    /// present but UNVALIDATED — must degrade EXACTLY like a non-supporting
    /// model (never serve an untrusted mode): ignored, no `chat_template_kwargs`.
    #[tokio::test]
    async fn test_chat_completions_thinking_ignored_for_unvalidated_model() {
        let server = httpmock::MockServer::start_async().await;
        let trap = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/v1/chat/completions")
                .body_contains("chat_template_kwargs");
            then.status(200).json_body(serde_json::json!({"choices": []}));
        });
        let real = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/v1/chat/completions");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(serde_json::json!({"choices": []}));
        });

        let routing = routing_map_with(
            "reasoner:30b",
            r#"{"thinking":{"supports_thinking":true,"validated":false}}"#,
        );
        let llm_url = format!("{}/v1/chat/completions", server.base_url());
        let state = test_state_with_llm_aliases_and_routing(
            Some(llm_url),
            std::collections::HashMap::new(),
            routing,
        );
        let app = build_router(state);

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("Content-Type", "application/json")
            .body(Body::from(chat_request_body_with_thinking(
                "reasoner:30b",
                "on",
            )))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        trap.assert_hits_async(0).await;
        real.assert_hits_async(1).await;
    }

    /// REGRESSION: no `thinking` param sent at all ⇒ unchanged behavior — the
    /// body is forwarded exactly as `chat_request_body` produces it (byte-for-
    /// byte pass-through, same as before this item), no `chat_template_kwargs`
    /// ever appears.
    #[tokio::test]
    async fn test_chat_completions_no_thinking_param_leaves_body_unchanged() {
        let server = httpmock::MockServer::start_async().await;
        let trap = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/v1/chat/completions")
                .body_contains("chat_template_kwargs");
            then.status(200).json_body(serde_json::json!({"choices": []}));
        });
        let real = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/v1/chat/completions");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(serde_json::json!({"choices": []}));
        });

        // Model IS supporting + validated — proves absence of the param, not
        // absence of capability, is what keeps behavior unchanged.
        let routing = routing_map_with(
            "reasoner:30b",
            r#"{"thinking":{"supports_thinking":true,"validated":true}}"#,
        );
        let llm_url = format!("{}/v1/chat/completions", server.base_url());
        let state = test_state_with_llm_aliases_and_routing(
            Some(llm_url),
            std::collections::HashMap::new(),
            routing,
        );
        let app = build_router(state);

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("Content-Type", "application/json")
            .body(Body::from(chat_request_body("reasoner:30b", false)))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        trap.assert_hits_async(0).await;
        real.assert_hits_async(1).await;
    }

    /// NEGATIVE TEST: a malformed `thinking` value (anything but "on"/"off")
    /// degrades to model default — never a crash, never a 4xx/5xx, and no
    /// `chat_template_kwargs` is injected.
    #[tokio::test]
    async fn test_chat_completions_malformed_thinking_value_defaults_no_crash() {
        let server = httpmock::MockServer::start_async().await;
        let trap = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/v1/chat/completions")
                .body_contains("chat_template_kwargs");
            then.status(200).json_body(serde_json::json!({"choices": []}));
        });
        let real = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/v1/chat/completions");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(serde_json::json!({"choices": []}));
        });

        let routing = routing_map_with(
            "reasoner:30b",
            r#"{"thinking":{"supports_thinking":true,"validated":true}}"#,
        );
        let llm_url = format!("{}/v1/chat/completions", server.base_url());
        let state = test_state_with_llm_aliases_and_routing(
            Some(llm_url),
            std::collections::HashMap::new(),
            routing,
        );
        let app = build_router(state);

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("Content-Type", "application/json")
            .body(Body::from(chat_request_body_with_thinking(
                "reasoner:30b",
                "maybe",
            )))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "malformed value must never crash/error");
        trap.assert_hits_async(0).await;
        real.assert_hits_async(1).await;
    }

    #[tokio::test]
    async fn test_chat_completions_requires_auth() {
        // Auth enabled (secret set) but no Authorization header → 401.
        let mut state = test_state_with_llm(Some("http://does-not-exist:9999".into()));
        // Rebuild with a secret to force auth on.
        Arc::get_mut(&mut state).unwrap().jwt_secret = "<REDACTED-SECRET>".into();
        let app = build_router(state);

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("Content-Type", "application/json")
            .body(Body::from(chat_request_body("gpt-oss:20b", false)))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_chat_completions_503_when_llm_url_unset() {
        // Auth disabled, llm_backend_url is None → 503.
        let state = test_state_with_llm(None);
        let app = build_router(state);

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("Content-Type", "application/json")
            .body(Body::from(chat_request_body("gpt-oss:20b", false)))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(json["error"].as_str().unwrap().contains("not configured"));
    }

    #[tokio::test]
    async fn test_chat_completions_non_streaming_proxies_json() {
        let server = httpmock::MockServer::start_async().await;
        let upstream_body = serde_json::json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hi there!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        });
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/v1/chat/completions");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(upstream_body.clone());
        });

        let llm_url = format!("{}/v1/chat/completions", server.base_url());
        let state = test_state_with_llm(Some(llm_url));
        let app = build_router(state);

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("Content-Type", "application/json")
            .body(Body::from(chat_request_body("gpt-oss:20b", false)))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        // Rate limit headers must be present on the proxied response.
        assert!(resp.headers().contains_key("X-RateLimit-Limit"));
        assert_eq!(
            resp.headers().get("content-type").unwrap(),
            "application/json"
        );

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["choices"][0]["message"]["content"], "Hi there!");

        mock.assert_async().await;
    }

    /// F1 regression: when the client sends a model alias (lumina-fast), the proxy
    /// must rewrite the `model` field to the real backend model (gpt-oss:20b) before
    /// forwarding, so Ollama no longer returns 404 "model lumina-fast not found".
    #[tokio::test]
    async fn test_chat_completions_resolves_model_alias_before_forwarding() {
        let server = httpmock::MockServer::start_async().await;
        let upstream_body = serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "gpt-oss:20b",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hi"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
        });
        // The mock only matches when the forwarded body carries the RESOLVED model.
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/v1/chat/completions")
                .json_body_partial(r#"{"model":"gpt-oss:20b"}"#);
            then.status(200)
                .header("content-type", "application/json")
                .json_body(upstream_body.clone());
        });

        let mut aliases = std::collections::HashMap::new();
        aliases.insert("lumina-fast".to_string(), "gpt-oss:20b".to_string());
        let llm_url = format!("{}/v1/chat/completions", server.base_url());
        let state = test_state_with_llm_aliases(Some(llm_url), aliases);
        let app = build_router(state);

        // Client sends the ALIAS, which Ollama would 404 on.
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("Content-Type", "application/json")
            .body(Body::from(chat_request_body("lumina-fast", false)))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        // Mock asserts the upstream received model=gpt-oss:20b, proving the rewrite.
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_chat_completions_streaming_preserves_sse_content_type() {
        let server = httpmock::MockServer::start_async().await;
        // Simulate an SSE stream body the way Ollama returns it.
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\" there\"}}]}\n\n\
                   data: [DONE]\n\n";
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/v1/chat/completions");
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(sse);
        });

        let llm_url = format!("{}/v1/chat/completions", server.base_url());
        let state = test_state_with_llm(Some(llm_url));
        let app = build_router(state);

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("Content-Type", "application/json")
            .body(Body::from(chat_request_body("gpt-oss:20b", true)))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        // Streaming content-type must be passed through untouched.
        assert_eq!(
            resp.headers().get("content-type").unwrap(),
            "text/event-stream"
        );

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(text.contains("data: "));
        assert!(text.contains("[DONE]"));

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_chat_completions_passes_through_upstream_error_status() {
        let server = httpmock::MockServer::start_async().await;
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/v1/chat/completions");
            then.status(404)
                .header("content-type", "application/json")
                .json_body(serde_json::json!({
                    "error": {"message": "model lumina-fast not found"}
                }));
        });

        let llm_url = format!("{}/v1/chat/completions", server.base_url());
        let state = test_state_with_llm(Some(llm_url));
        let app = build_router(state);

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("Content-Type", "application/json")
            .body(Body::from(chat_request_body("lumina-fast", false)))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // The upstream 404 must be surfaced verbatim, not masked as 502.
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("not found"));

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_chat_completions_502_when_backend_unreachable() {
        // Point at an unroutable address so the upstream send() fails.
        let state = test_state_with_llm(Some(
            "http://127.0.0.1:1/v1/chat/completions".into(),
        ));
        let app = build_router(state);

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("Content-Type", "application/json")
            .body(Body::from(chat_request_body("gpt-oss:20b", false)))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(json["error"].as_str().unwrap().contains("unreachable"));
    }

    #[tokio::test]
    async fn test_parse_model_from_body_extracts_model() {
        assert_eq!(
            parse_model_from_body(br#"{"model":"gpt-oss:120b","messages":[]}"#),
            "gpt-oss:120b"
        );
        assert_eq!(parse_model_from_body(b"not json"), "unknown");
        assert_eq!(parse_model_from_body(br#"{"messages":[]}"#), "unknown");
    }

    // ── TIER-02 pull-on-miss wiring tests ──────────────────────────────────────

    /// Build an AppState whose registry/coordinator are provided by the caller
    /// (so a populated registry can be wired in), with an explicit upstream LLM
    /// URL. Auth disabled.
    fn test_state_with_registry(
        llm_url: Option<String>,
        registry: Arc<Mutex<crate::models::registry::ModelRegistry>>,
        coordinator: Arc<crate::models::transfer::PullCoordinator>,
    ) -> Arc<AppState> {
        let config = Config {
            mcp_backend_url: "http://does-not-exist:9999".into(),
            jwt_secret: String::new(),
            tool_timeout_secs: 5,
            catalog_cache_secs: 300,
            listen_port: 9099,
            control_port: 8090,
            rate_limits: default_rate_config(),
            llm_backend_url: llm_url.clone(),
            model_aliases: std::collections::HashMap::new(),
            model_archive_path: "/var/lib/model-archive".into(),
            model_local_path: "/opt/ollama-models".into(),
            model_protected: vec![],
            model_pull_timeout_secs: 600,
            model_registry_path: "<path>/model-registry.json".into(),
            model_disk_pressure_percent: 80,
            model_sweep_interval_secs: 1800,
            model_warm_cooldown_hours: 168,
            model_source_allowlist: Vec::new(),
            outbound_proxy: None,
            runtime_telemetry_off: true,
        };
        let proxy = McpProxy::new(&config, Arc::new(FallbackRegistry::new()));
        let proxy_arc = Arc::new(McpProxy::new(&config, Arc::new(FallbackRegistry::new())));
        let agentic_executor = Arc::new(AgenticExecutor::new(proxy_arc));
        let rate_limiter = Arc::new(Mutex::new(ProxyRateLimiter::new(default_rate_config())));
        let audit_logger = Arc::new(AuditLogger::new(std::path::PathBuf::from("/dev/null")));
        Arc::new(AppState {
            proxy,
            jwt_secret: String::new(),
            audit_logger,
            rate_limiter,
            agentic_executor,
            llm_backend_url: llm_url,
            model_aliases: std::collections::HashMap::new(),
            http_client: reqwest::Client::new(),
            model_registry: registry,
            pull_coordinator: coordinator,
            local_evictor: std::sync::Arc::new(crate::models::eviction::FsLocalEvictor::new(std::path::PathBuf::from("/tmp"))), disk_op_lock: crate::models::eviction::new_disk_op_lock(), disk_probe: std::sync::Arc::new(crate::models::transfer::StatvfsProbe), disk_pressure_percent: 80, model_warm_cooldown_hours: 168,
            routing_map: empty_routing_map(),
            coding_profile_source: empty_coding_profile_source(),
        })
    }

    /// Write a minimal Ollama manifest + its blobs under `<root>` for `<model>:<tag>`,
    /// returning the model name. Mirrors the transfer-test layout.
    fn write_archive_model(root: &std::path::Path, model: &str, tag: &str, sizes: &[u64]) -> String {
        use std::fs;
        let manifests = root
            .join("manifests")
            .join("registry.ollama.ai")
            .join("library")
            .join(model);
        fs::create_dir_all(&manifests).unwrap();
        let blobs = root.join("blobs");
        fs::create_dir_all(&blobs).unwrap();
        let mut layers = Vec::new();
        for (i, size) in sizes.iter().enumerate() {
            let digest = format!("sha256:{model}{i}");
            fs::write(blobs.join(digest.replacen(':', "-", 1)), vec![b'x'; *size as usize]).unwrap();
            layers.push(serde_json::json!({ "size": size, "digest": digest }));
        }
        let cfg = format!("sha256:{model}cfg");
        fs::write(blobs.join(cfg.replacen(':', "-", 1)), b"cfg").unwrap();
        let body = serde_json::json!({
            "config": { "size": 3, "digest": cfg },
            "layers": layers,
        });
        fs::write(manifests.join(tag), serde_json::to_string(&body).unwrap()).unwrap();
        format!("{model}:{tag}")
    }

    /// A chat request for a *Cold* model (present only in the archive) triggers a
    /// transparent archive pull before the upstream inference call: the model is
    /// copied to local disk, promoted to Warm in the registry, and the upstream
    /// mock is hit exactly once.
    #[tokio::test]
    async fn test_chat_completions_pulls_cold_model_before_forwarding() {
        use crate::models::registry::{ModelRegistry, StorageTier};
        use crate::models::transfer::PullCoordinator;

        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        let model = write_archive_model(&base.join("archive"), "coldmodel", "1", &[64, 64]);

        let mut reg = ModelRegistry::new(
            base.join("registry.json"),
            base.join("local"),
            base.join("archive"),
            vec![],
        );
        reg.reconcile();
        assert_eq!(reg.get(&model).unwrap().tier, StorageTier::Cold);
        let registry = Arc::new(Mutex::new(reg));
        let coordinator = Arc::new(PullCoordinator::new(
            registry.clone(),
            std::time::Duration::from_secs(30),
        ));

        let server = httpmock::MockServer::start_async().await;
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/v1/chat/completions");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(serde_json::json!({"choices": []}));
        });
        let llm_url = format!("{}/v1/chat/completions", server.base_url());

        let state = test_state_with_registry(Some(llm_url), registry.clone(), coordinator);
        let app = build_router(state);
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("Content-Type", "application/json")
            .body(Body::from(chat_request_body(&model, false)))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Upstream was reached → the pull succeeded and inference proceeded.
        mock.assert_async().await;
        // The cold model was copied locally and promoted to Warm.
        assert_eq!(
            registry.lock().await.get(&model).unwrap().tier,
            StorageTier::Warm
        );
        assert!(base.join("local/blobs/sha256-coldmodel0").is_file());
    }

    #[tokio::test]
    async fn test_chat_completions_pulls_cold_model_for_untagged_request() {
        // Regression: the registry is keyed by the fully-tagged name
        // ("coldmodel:latest"), but clients often request the untagged name
        // ("coldmodel"). The pull-on-miss hook must normalize to ":latest" so the
        // cold model is still found and pulled.
        use crate::models::registry::{ModelRegistry, StorageTier};
        use crate::models::transfer::PullCoordinator;

        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        let model = write_archive_model(&base.join("archive"), "coldmodel", "latest", &[64, 64]);
        assert_eq!(model, "coldmodel:latest");

        let mut reg = ModelRegistry::new(
            base.join("registry.json"),
            base.join("local"),
            base.join("archive"),
            vec![],
        );
        reg.reconcile();
        assert_eq!(reg.get("coldmodel:latest").unwrap().tier, StorageTier::Cold);
        let registry = Arc::new(Mutex::new(reg));
        let coordinator = Arc::new(PullCoordinator::new(
            registry.clone(),
            std::time::Duration::from_secs(30),
        ));

        let server = httpmock::MockServer::start_async().await;
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/v1/chat/completions");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(serde_json::json!({"choices": []}));
        });
        let llm_url = format!("{}/v1/chat/completions", server.base_url());

        let state = test_state_with_registry(Some(llm_url), registry.clone(), coordinator);
        let app = build_router(state);
        // Request the UNTAGGED name.
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("Content-Type", "application/json")
            .body(Body::from(chat_request_body("coldmodel", false)))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        mock.assert_async().await;
        // Untagged request normalized to ":latest" → cold model pulled + warmed.
        assert_eq!(
            registry.lock().await.get("coldmodel:latest").unwrap().tier,
            StorageTier::Warm
        );
    }

    /// A chat request for a model the registry does NOT know passes straight
    /// through to the upstream unchanged (no pull attempted, no error) — the
    /// no-regression guarantee for unknown models.
    #[tokio::test]
    async fn test_chat_completions_unknown_model_passes_through() {
        let server = httpmock::MockServer::start_async().await;
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/v1/chat/completions");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(serde_json::json!({"choices": []}));
        });
        let llm_url = format!("{}/v1/chat/completions", server.base_url());

        // Empty registry → every model is "unknown" → pass-through.
        let (registry, coordinator) = empty_model_state();
        let state = test_state_with_registry(Some(llm_url), registry, coordinator);
        let app = build_router(state);
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("Content-Type", "application/json")
            .body(Body::from(chat_request_body("some-unknown-model:42", false)))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        mock.assert_async().await;
    }
}
