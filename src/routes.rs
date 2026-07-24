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
//!
//! ## Audit logging (tools/list, tools/call, tools/discover)
//! Every reachable outcome of these three handlers — auth failure, rate-limit
//! rejection, and the underlying proxy call's success/error — produces exactly
//! one `AuditLogger` entry (never tool arguments or raw discover query text;
//! see `crate::audit` and `proxy_error_kind` below). One outcome is NOT
//! covered: a malformed request body fails Axum's `Json<T>` extractor before
//! the handler body runs, so it never reaches an audit call. Closing that gap
//! would require moving auth/audit into a `tower` layer ahead of the
//! extractor (the existing `AuditLayer` in `middleware.rs` sketches this but
//! is not wired in) — tracked as a follow-up, not attempted here to keep this
//! change scoped to the handler-body audit gap the tool-allowlist review
//! flagged.

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
    /// MSM-02/04: maximum duration (seconds) for a single warm→cold eviction
    /// copy before it aborts, cleans up partial archive files, and leaves the
    /// model Warm for retry. Shared by the background sweep, the manual
    /// `/api/models/sweep` trigger, and `/api/models/:name/archive`.
    pub model_archive_copy_timeout_secs: u64,
    /// MSM-03/04: minimum age (seconds) a local blob must have before the
    /// orphan-GC pass (`POST /api/storage/gc`) will delete it — the B1
    /// defense-in-depth grace window that keeps an in-flight archive pull's
    /// just-copied blobs from being treated as orphans.
    pub model_gc_min_age_secs: u64,
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
    /// Task 2 (federation): a second, independent `McpProxy` instance pointed
    /// at the standalone `terminus_personal` Rust MCP binary, when
    /// `PERSONAL_BACKEND_URL` is configured. `None` when unset — the
    /// `/v1/personal/*` routes then return a clean 503 rather than panicking
    /// or hanging. Constructed with `McpProxy::new_unfiltered` (no
    /// `tool_allowlist::is_core_tool` scoping) and deliberately never merged
    /// into `proxy` / `/v1/tools/list` — reachable only via
    /// `/v1/personal/tools/list` and `/v1/personal/tools/call`.
    pub personal_proxy: Option<Arc<McpProxy>>,
    /// EMBED-01: config for the local-first/OpenRouter-fallback `POST
    /// /v1/embeddings` proxy. See `crate::embeddings`.
    pub embeddings_config: crate::embeddings::EmbeddingsConfig,
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

/// Coarse-grained, argument-free classification of a `ProxyError` for audit
/// logging. Deliberately NEVER uses the error's `Display`/`to_string()` text:
/// `ProxyError::ToolExecution`/`McpBackend`/`Session` wrap strings that can
/// originate from a tool backend's own response, which may echo back content
/// derived from the (potentially sensitive) call arguments. The audit log gets
/// only this fixed, safe classification — full detail stays in the (separate,
/// non-persistent) `tracing::warn!` diagnostic already logged alongside it.
fn proxy_error_kind(err: &ProxyError) -> &'static str {
    match err {
        ProxyError::McpBackend(_) => "mcp_backend_error",
        ProxyError::Session(_) => "session_error",
        ProxyError::ToolNotFound(_) => "tool_not_found",
        ProxyError::ToolExecution(_) => "tool_execution_error",
        ProxyError::Timeout(_) => "timeout",
        ProxyError::Http(_) => "http_error",
        ProxyError::Json(_) => "json_error",
        ProxyError::Config(_) => "config_error",
    }
}

/// Map a `ProxyError` to the audit `Status` it represents.
fn proxy_error_audit_status(err: &ProxyError) -> AuditStatus {
    match err {
        ProxyError::Timeout(_) => AuditStatus::Timeout,
        _ => AuditStatus::Error,
    }
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

pub async fn tools_list(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let start = Instant::now();

    let claims = match auth_check(&headers, &state.jwt_secret) {
        Ok(c) => c,
        Err(e) => {
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

    // Tool calls (list) count against the tool budget.
    let role = UserRole::from_claim(claims.role.as_deref());
    let rl_result = {
        let mut rl = state.rate_limiter.lock().await;
        rl.check_and_record(&claims.sub, role, CallType::Tool)
    };

    if !rl_result.allowed {
        state.audit_logger.log_tool_list(
            &claims.sub,
            start.elapsed().as_millis() as u64,
            AuditStatus::Error,
            Some("rate_limited".to_string()),
        );
        return rate_limit_exceeded_response(&rl_result, CallType::Tool);
    }

    let rl_headers = rate_limit_headers(&rl_result);
    match state.proxy.tool_list().await {
        Ok(tools) => {
            let count = tools.len();
            state.audit_logger.log_tool_list(
                &claims.sub,
                start.elapsed().as_millis() as u64,
                AuditStatus::Success,
                None,
            );
            let mut response = Json(ToolListResponse { tools, count }).into_response();
            response.headers_mut().extend(rl_headers);
            response
        }
        Err(e) => {
            warn!("tools/list error: {e}");
            state.audit_logger.log_tool_list(
                &claims.sub,
                start.elapsed().as_millis() as u64,
                proxy_error_audit_status(&e),
                Some(proxy_error_kind(&e).to_string()),
            );
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
    let start = Instant::now();

    let claims = match auth_check(&headers, &state.jwt_secret) {
        Ok(c) => c,
        Err(e) => {
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

    let role = UserRole::from_claim(claims.role.as_deref());
    // Captured before `req.arguments` is moved into `tool_call` below (and
    // before the rate-limit check, so a 429 can still be audited with the
    // tool name) — the audit log records WHICH tool was invoked, never the
    // (potentially sensitive) `req.arguments` payload itself.
    let tool_name = req.name.clone();
    let rl_result = {
        let mut rl = state.rate_limiter.lock().await;
        rl.check_and_record(&claims.sub, role, CallType::Tool)
    };

    if !rl_result.allowed {
        state.audit_logger.log_tool_call(
            &claims.sub,
            &tool_name,
            start.elapsed().as_millis() as u64,
            AuditStatus::Error,
            Some("rate_limited".to_string()),
        );
        return rate_limit_exceeded_response(&rl_result, CallType::Tool);
    }

    let rl_headers = rate_limit_headers(&rl_result);
    match state.proxy.tool_call(&req.name, req.arguments).await {
        Ok((result, source)) => {
            state.audit_logger.log_tool_call(
                &claims.sub,
                &tool_name,
                start.elapsed().as_millis() as u64,
                AuditStatus::Success,
                None,
            );
            let mut response = Json(ToolCallResponse {
                result,
                source: Some(source.to_string()),
            })
            .into_response();
            response.headers_mut().extend(rl_headers);
            response
        }
        Err(e) => {
            warn!("tools/call error for {}: {e}", tool_name);
            state.audit_logger.log_tool_call(
                &claims.sub,
                &tool_name,
                start.elapsed().as_millis() as u64,
                proxy_error_audit_status(&e),
                Some(proxy_error_kind(&e).to_string()),
            );
            proxy_error_response(e)
        }
    }
}

// ── /v1/personal/tools/list, /v1/personal/tools/call ─────────────────────────
//
// Task 2 (federation): these mirror `tools_list`/`tools_call` above but
// operate on `state.personal_proxy` — the second, unfiltered `McpProxy`
// instance pointed at `terminus_personal`. Kept as separate handler functions
// (rather than parameterizing the existing ones) so the default
// `/v1/tools/*` catalog's code path is untouched by this change, per the
// regression guard in
// `tests::test_default_tools_list_unchanged_when_personal_unconfigured`.
//
// Auth/rate-limiting/audit logging reuse the exact same JWT + `ProxyRateLimiter`
// + `AuditLogger` machinery as the default tool routes — this is a second
// catalog, not a second security posture.

/// Returns a clean 503 (never a panic or hang) when `PERSONAL_BACKEND_URL` is
/// unset, i.e. `state.personal_proxy` is `None`.
fn personal_backend_unconfigured_response() -> Response {
    let body = serde_json::json!({
        "error": "personal backend not configured (PERSONAL_BACKEND_URL unset)"
    });
    (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response()
}

pub async fn personal_tools_list(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    let start = Instant::now();

    let Some(proxy) = state.personal_proxy.as_ref() else {
        return personal_backend_unconfigured_response();
    };

    let claims = match auth_check(&headers, &state.jwt_secret) {
        Ok(c) => c,
        Err(e) => {
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

    let role = UserRole::from_claim(claims.role.as_deref());
    let rl_result = {
        let mut rl = state.rate_limiter.lock().await;
        rl.check_and_record(&claims.sub, role, CallType::Tool)
    };

    if !rl_result.allowed {
        state.audit_logger.log_tool_list(
            &claims.sub,
            start.elapsed().as_millis() as u64,
            AuditStatus::Error,
            Some("rate_limited".to_string()),
        );
        return rate_limit_exceeded_response(&rl_result, CallType::Tool);
    }

    let rl_headers = rate_limit_headers(&rl_result);
    match proxy.tool_list().await {
        Ok(tools) => {
            let count = tools.len();
            state.audit_logger.log_tool_list(
                &claims.sub,
                start.elapsed().as_millis() as u64,
                AuditStatus::Success,
                None,
            );
            let mut response = Json(ToolListResponse { tools, count }).into_response();
            response.headers_mut().extend(rl_headers);
            response
        }
        Err(e) => {
            warn!("personal tools/list error: {e}");
            state.audit_logger.log_tool_list(
                &claims.sub,
                start.elapsed().as_millis() as u64,
                proxy_error_audit_status(&e),
                Some(proxy_error_kind(&e).to_string()),
            );
            proxy_error_response(e)
        }
    }
}

pub async fn personal_tools_call(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ToolCallRequest>,
) -> Response {
    let start = Instant::now();

    let Some(proxy) = state.personal_proxy.as_ref() else {
        return personal_backend_unconfigured_response();
    };

    let claims = match auth_check(&headers, &state.jwt_secret) {
        Ok(c) => c,
        Err(e) => {
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

    let role = UserRole::from_claim(claims.role.as_deref());
    let tool_name = req.name.clone();
    let rl_result = {
        let mut rl = state.rate_limiter.lock().await;
        rl.check_and_record(&claims.sub, role, CallType::Tool)
    };

    if !rl_result.allowed {
        state.audit_logger.log_tool_call(
            &claims.sub,
            &tool_name,
            start.elapsed().as_millis() as u64,
            AuditStatus::Error,
            Some("rate_limited".to_string()),
        );
        return rate_limit_exceeded_response(&rl_result, CallType::Tool);
    }

    let rl_headers = rate_limit_headers(&rl_result);
    match proxy.tool_call(&req.name, req.arguments).await {
        Ok((result, source)) => {
            state.audit_logger.log_tool_call(
                &claims.sub,
                &tool_name,
                start.elapsed().as_millis() as u64,
                AuditStatus::Success,
                None,
            );
            let mut response = Json(ToolCallResponse {
                result,
                source: Some(source.to_string()),
            })
            .into_response();
            response.headers_mut().extend(rl_headers);
            response
        }
        Err(e) => {
            warn!("personal tools/call error for {}: {e}", tool_name);
            state.audit_logger.log_tool_call(
                &claims.sub,
                &tool_name,
                start.elapsed().as_millis() as u64,
                proxy_error_audit_status(&e),
                Some(proxy_error_kind(&e).to_string()),
            );
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

fn default_max_results() -> usize {
    10
}

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
    let start = Instant::now();

    let claims = match auth_check(&headers, &state.jwt_secret) {
        Ok(c) => c,
        Err(e) => {
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

    let role = UserRole::from_claim(claims.role.as_deref());
    let rl_result = {
        let mut rl = state.rate_limiter.lock().await;
        rl.check_and_record(&claims.sub, role, CallType::Tool)
    };

    if !rl_result.allowed {
        state.audit_logger.log_tool_discover(
            &claims.sub,
            "",
            start.elapsed().as_millis() as u64,
            AuditStatus::Error,
            Some("rate_limited".to_string()),
        );
        return rate_limit_exceeded_response(&rl_result, CallType::Tool);
    }

    let rl_headers = rate_limit_headers(&rl_result);
    let max = req.max_results.min(100); // cap at 100
    match state.proxy.tool_discover(&req.query, max).await {
        Ok(tools) => {
            let count = tools.len();
            // The raw query text is NEVER logged — only a safe, pre-summarised
            // result count. See `AuditLogger::log_tool_discover`.
            state.audit_logger.log_tool_discover(
                &claims.sub,
                &format!("results={count}"),
                start.elapsed().as_millis() as u64,
                AuditStatus::Success,
                None,
            );
            let query = req.query.clone();
            let mut response = Json(ToolDiscoverResponse {
                tools,
                query,
                count,
            })
            .into_response();
            response.headers_mut().extend(rl_headers);
            response
        }
        Err(e) => {
            warn!("tools/discover error: {e}");
            state.audit_logger.log_tool_discover(
                &claims.sub,
                "",
                start.elapsed().as_millis() as u64,
                proxy_error_audit_status(&e),
                Some(proxy_error_kind(&e).to_string()),
            );
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

    // ── BLD-09 idle-mode admission (see `chat_completions`). This route dispatches
    // LLM calls, so it must join the closed-world drain / lazy-restore path too. ──
    // Held for the whole request; for the STREAMING branch it is moved into the
    // spawned executor task so the in-flight count stays incremented until the actual
    // inference completes — NOT merely until this handler returns the SSE Response
    // (cycle-2 fix #3). For the non-streaming branch it is held across the awaited
    // `execute(...)` below.
    let idle_inflight = match crate::admin::idle::admit_inference(&state).await {
        crate::admin::idle::Admission::Admitted(g) => g,
        crate::admin::idle::Admission::Rejected(resp) => return resp,
    };

    // ── GPU-exclusive gate ────────────────────────────────────────────────────
    // Same gate as `chat_completions`/`infer`: this route makes LLM calls too
    // (via `agentic_executor.execute`, which dispatches to CHORD_LLM_URL), so it
    // must not be allowed to load a model / contend for the GPU while the intake
    // harness holds exclusive access. Placed AFTER auth, BEFORE the rate-limit
    // check or any dispatch work.
    if let Some(record) =
        crate::gpu_exclusive::GPU_EXCLUSIVE.active_holder(crate::gpu_exclusive::now_epoch())
    {
        return gpu_exclusively_held_response(&record);
    }

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
        // Move the in-flight guard INTO the detached executor task: the guard (and so
        // the in-flight count) lives until the real inference finishes, regardless of
        // when the SSE Response is returned or whether the client disconnects early.
        tokio::spawn(async move {
            let _idle_inflight = idle_inflight;
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

    // ── BLD-09 idle-mode admission: join the closed-world drain (so `POST /admin/idle`
    // can drain before releasing) and lazily restore if idle. Once idle-mode has begun
    // entering (EnteringIdle), admission is refused with a retryable 503 so no new
    // request slips into the in-flight set after the drain window opens. ────────────
    // On every EARLY return below (GPU-exclusive 503, rate-limit, upstream error) the
    // guard simply drops. On the SUCCESS path it is moved INTO the streamed response
    // body so the in-flight count stays incremented until the stream is fully consumed
    // — not merely until this handler returns the Response (cycle-2 fix #3).
    let idle_inflight = match crate::admin::idle::admit_inference(&state).await {
        crate::admin::idle::Admission::Admitted(g) => g,
        crate::admin::idle::Admission::Rejected(resp) => return resp,
    };

    // ── GPU-exclusive gate ────────────────────────────────────────────────────
    // While the GPU is exclusively held (by the intake harness), do NOT load a
    // model / dispatch inference and contend for VRAM — return a clear, structured
    // 503 instead. Placed AFTER auth (so an unauthenticated caller still gets 401)
    // and BEFORE any rate-limit/pull/upstream work.
    if let Some(record) =
        crate::gpu_exclusive::GPU_EXCLUSIVE.active_holder(crate::gpu_exclusive::now_epoch())
    {
        return gpu_exclusively_held_response(&record);
    }

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
        // PROMEX-02: no registry lookup has happened yet at this point, so
        // there's no known-served evidence for `model` — bound to <unknown>.
        crate::metrics::record_inference(
            &crate::metrics::bounded_model_label(&model, false),
            false,
            start.elapsed(),
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
    // PROMEX-02: `is_known_served` is Chord's own validated known-model
    // evidence for the `chord_inference_*` metrics' bounded `model` label —
    // the registry actually tracking `registry_key` (regardless of tier),
    // never a parse of the raw client-supplied model string. See
    // `crate::metrics::bounded_model_label`.
    let (needs_pull, is_known_served) = {
        use crate::models::registry::StorageTier;
        let mut reg = state.model_registry.lock().await;
        reg.update_last_requested(&registry_key);
        let record = reg.get(&registry_key);
        let is_known_served = record.is_some();
        let needs_pull = matches!(record.map(|r| r.tier.clone()), Some(StorageTier::Cold));
        (needs_pull, is_known_served)
    };
    if needs_pull {
        if let Err(e) = state
            .pull_coordinator
            .ensure_local(&registry_key, None)
            .await
        {
            warn!("chat/completions: cold model {resolved_model} could not be retrieved: {e}");
            state.audit_logger.log_llm_call(
                &claims.sub,
                &resolved_model,
                start.elapsed().as_millis() as u64,
                AuditStatus::Error,
                Some(format!("archive pull failed: {e}")),
            );
            crate::metrics::record_inference(
                &crate::metrics::bounded_model_label(&resolved_model, is_known_served),
                false,
                start.elapsed(),
            );
            let resp_body = serde_json::json!({
                "error": format!("model could not be retrieved from archive: {e}")
            });
            return (StatusCode::SERVICE_UNAVAILABLE, Json(resp_body)).into_response();
        }
    }

    // ── CHRD-DIFF-01 / CHRD-DIFF-503: DiffusionGemma managed-daemon routing ──
    // Takes precedence over P5 tag-aware routing below: a request for the
    // Chord-managed diffusion model (`diffusion-gemma` by default, see
    // `crate::diffusion`) is lazy-started on demand and SERVED via the
    // daemon's own `/generate` API — never `CHORD_LLM_URL` and never the
    // ModelRegistry backend table (that daemon was never registered as a
    // `Backend`; it is owned entirely by the `diffusion` module's own
    // process manager). This is a fully separate serving path (not a raw
    // forward): CHRD-DIFF-503 found the daemon has NO `/v1/chat/completions`
    // route (verified live: POST → 404, `/v1/models` → 404 — only `/health`
    // and `/generate` exist), so the old raw-forward-to-OpenAI-shape 404'd
    // and was mapped to a 503 on every diffusion request. `ensure_running`'s
    // adopt/gate/spawn behavior (including the gpu_exclusive `Blocked` case,
    // which still returns a structured 503 rather than silently falling
    // back to the wrong non-diffusion upstream) is unchanged.
    if crate::diffusion::global().is_diffusion_model(&resolved_model) {
        return serve_diffusion_chat_completion(
            &state,
            &claims,
            &resolved_model,
            &body,
            start,
            rl_headers,
            is_known_served,
            idle_inflight,
        )
        .await;
    }

    // ── P5 tag-aware routing ───────────────────────────────────────────────
    // Forward to the model's tagged backend (GPU vs CPU), starting an
    // on-demand backend if needed. Untagged models resolve to the default
    // backend (whose URL matches CHORD_LLM_URL), so behavior is unchanged
    // until models are tagged. On any failure we fall back to CHORD_LLM_URL.
    // `bearer_key` is `Some` only for backends with `api_key_env` set (e.g.
    // OpenRouter) — re-injected as an outbound Authorization header below,
    // after the inbound-header copy loop strips the caller's own JWT.
    let (llm_url, bearer_key) =
        crate::models::routing::resolve_and_ensure(&state.model_registry, &registry_key, &resolved_model)
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
            crate::metrics::record_inference(
                &crate::metrics::bounded_model_label(&model, is_known_served),
                false,
                start.elapsed(),
            );
            let resp_body = serde_json::json!({
                "error": format!("LLM backend unreachable: {e}")
            });
            let mut response = (StatusCode::BAD_GATEWAY, Json(resp_body)).into_response();
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
    crate::metrics::record_inference(
        &crate::metrics::bounded_model_label(&model, is_known_served),
        status.is_success(),
        start.elapsed(),
    );

    // Stream the upstream body straight back to the caller. This passes through
    // both non-streaming JSON and streaming SSE (text/event-stream) untouched.
    use futures_util::{StreamExt, TryStreamExt};
    // Move the idle-mode in-flight guard INTO the body stream so it is dropped only
    // when the streamed body is fully consumed (or dropped on client disconnect),
    // never when this handler returns the Response (cycle-2 fix #3). The `move`
    // inspect closure OWNS the guard, tying its lifetime to the stream's.
    let idle_guard = idle_inflight;
    let stream = upstream
        .bytes_stream()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
        .inspect(move |_chunk| {
            let _ = &idle_guard; // keepalive: forces the closure to own the guard
        });
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

/// CHRD-DIFF-503: serve a `/v1/chat/completions` request whose (alias-resolved)
/// model is the Chord-managed DiffusionGemma model. Calls
/// [`crate::diffusion::global`]'s `ensure_running` (unchanged adopt/gate/spawn
/// behavior, including the structured 503 on `gpu_exclusive`-Blocked), then
/// SERVES via the daemon's real `POST /generate` API — never a raw forward to
/// an OpenAI-shaped endpoint the daemon doesn't implement (that was the bug:
/// the daemon 404s on `/v1/chat/completions`, which `chat_completions` mapped
/// to a 503 on every diffusion request). The `/generate` result is translated
/// into a standard OpenAI chat-completion JSON response.
///
/// DiffusionGemma generates in fixed canvas blocks, not a token stream, so
/// (v1) this always returns a single non-streamed JSON response even when the
/// caller set `"stream": true` — correct behavior, just not SSE yet.
async fn serve_diffusion_chat_completion(
    state: &Arc<AppState>,
    claims: &Claims,
    resolved_model: &str,
    body: &[u8],
    start: Instant,
    rl_headers: HeaderMap,
    is_known_served: bool,
    _idle_inflight: crate::admin::idle::InflightGuard,
) -> Response {
    let manager = crate::diffusion::global();

    let generate_url = match manager.ensure_running().await {
        Ok(url) => url,
        Err(e) => {
            warn!("diffusion: could not serve {resolved_model}: {e}");
            state.audit_logger.log_llm_call(
                &claims.sub,
                resolved_model,
                start.elapsed().as_millis() as u64,
                AuditStatus::Error,
                Some(e.clone()),
            );
            crate::metrics::record_inference(
                &crate::metrics::bounded_model_label(resolved_model, is_known_served),
                false,
                start.elapsed(),
            );
            let resp_body = serde_json::json!({
                "error": format!("diffusion backend unavailable: {e}")
            });
            let mut response = (StatusCode::SERVICE_UNAVAILABLE, Json(resp_body)).into_response();
            response.headers_mut().extend(rl_headers);
            return response;
        }
    };

    let (system_prompt, user_prompt) = extract_diffusion_prompt(body);
    let max_tokens = parse_max_tokens_from_body(body).unwrap_or(1024);

    let gen = match manager
        .generate(&generate_url, &system_prompt, &user_prompt, max_tokens)
        .await
    {
        Ok(g) => g,
        Err(e) => {
            warn!("diffusion: /generate failed for {resolved_model}: {e}");
            state.audit_logger.log_llm_call(
                &claims.sub,
                resolved_model,
                start.elapsed().as_millis() as u64,
                AuditStatus::Error,
                Some(e.clone()),
            );
            crate::metrics::record_inference(
                &crate::metrics::bounded_model_label(resolved_model, is_known_served),
                false,
                start.elapsed(),
            );
            let resp_body = serde_json::json!({
                "error": format!("diffusion backend unavailable: {e}")
            });
            let mut response = (StatusCode::SERVICE_UNAVAILABLE, Json(resp_body)).into_response();
            response.headers_mut().extend(rl_headers);
            return response;
        }
    };

    state.audit_logger.log_llm_call(
        &claims.sub,
        resolved_model,
        start.elapsed().as_millis() as u64,
        AuditStatus::Success,
        None,
    );
    crate::metrics::record_inference(
        &crate::metrics::bounded_model_label(resolved_model, is_known_served),
        true,
        start.elapsed(),
    );

    let resp_body = diffusion_generate_to_openai_response(resolved_model, &gen);
    let mut response = (StatusCode::OK, Json(resp_body)).into_response();
    response.headers_mut().extend(rl_headers);
    // `_idle_inflight` drops here (function return), releasing the in-flight
    // count — correct for this handler because, unlike the streamed P5/
    // fallback path, the response body above is already fully buffered JSON;
    // there is no later point at which "fully consumed" would differ from
    // "handler returned".
    response
}

/// Extract a (system, user) prompt pair from an OpenAI-style chat-completions
/// request body for the diffusion `/generate` API, which — unlike an OpenAI
/// backend — takes a single flat `system` + `prompt` string rather than a
/// `messages` array. All `system`-role messages are concatenated (in order)
/// into the system prompt; every other message (`user`/`assistant`/tool,
/// etc.) is rendered as `"<Role>: <content>"` lines and concatenated into the
/// prompt, preserving conversation order — a simple, standard flattening
/// (not a model-specific chat template) that keeps multi-turn context legible
/// to the model. A `content` array (OpenAI's multi-part message shape) has
/// its string `text` parts joined; non-text parts (e.g. images) are skipped,
/// as DiffusionGemma here is text-only. Malformed/non-object JSON yields two
/// empty strings rather than erroring — the daemon call will simply see an
/// empty prompt.
fn extract_diffusion_prompt(body: &[u8]) -> (String, String) {
    let Some(messages) = serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|v| v.get("messages").cloned())
        .and_then(|m| m.as_array().cloned())
    else {
        return (String::new(), String::new());
    };

    let mut system = String::new();
    let mut convo = String::new();
    for msg in &messages {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        let content = message_content_to_text(msg.get("content"));
        if content.is_empty() {
            continue;
        }
        if role == "system" {
            if !system.is_empty() {
                system.push_str("\n\n");
            }
            system.push_str(&content);
        } else {
            if !convo.is_empty() {
                convo.push('\n');
            }
            let label = match role {
                "assistant" => "Assistant",
                "tool" => "Tool",
                _ => "User",
            };
            convo.push_str(&format!("{label}: {content}"));
        }
    }
    (system, convo)
}

/// Render an OpenAI message `content` field (either a plain string, or the
/// multi-part `[{"type":"text","text":"..."}, ...]` shape) as flat text.
fn message_content_to_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Extract an optional top-level `max_tokens` integer from the request body
/// (standard OpenAI field). `None` when absent/malformed — the caller applies
/// the daemon-side default.
fn parse_max_tokens_from_body(body: &[u8]) -> Option<u32> {
    serde_json::from_slice::<Value>(body)
        .ok()?
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .map(|n| n as u32)
}

/// Translate a DiffusionGemma `/generate` response into a standard OpenAI
/// `chat.completion` response object.
fn diffusion_generate_to_openai_response(
    model: &str,
    gen: &crate::diffusion::DiffusionGenerateResponse,
) -> Value {
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    serde_json::json!({
        "id": format!("chatcmpl-dgem-{created}"),
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": gen.text,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": gen.input_tokens,
            "completion_tokens": gen.tokens,
            "total_tokens": gen.input_tokens + gen.tokens,
        },
    })
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

// ── /v1/embeddings ────────────────────────────────────────────────────────────

/// POST /v1/embeddings — OpenAI-compatible embeddings proxy (EMBED-01).
///
/// Serves LOCAL-FIRST from the fleet Ollama (`EMBED_LOCAL_URL`/
/// `EMBED_LOCAL_MODEL`) and falls back to OpenRouter
/// (`EMBED_FALLBACK_MODEL`, `OPENROUTER_API_KEY`) when local is unreachable,
/// errors, times out, or returns a vector whose dimensionality doesn't match
/// `EMBED_DIM`. Never returns a wrong-dimension vector — a mismatch on both
/// paths is a structured 502, never a partial/garbage response.
///
/// Same JWT auth as every other endpoint; counted against the caller's LLM
/// rate-limit budget (an embedding call is model inference, same as chat).
///
/// ## No GPU-exclusive gate (deliberate)
/// Unlike `chat_completions`/`agent_execute`, this route is NOT gated on the
/// GPU-exclusive lock, on unanimous reviewer consensus: (a) the OpenRouter
/// fallback path uses zero local GPU, so gating would defeat the whole
/// local-first→fallback resilience story while a sweep holds the GPU; (b) an
/// agent driving `/v1/agent/execute` can hold the GPU-exclusive lock and then
/// need `/v1/embeddings` for RAG/semantic search — a hard gate there would
/// deadlock it against its own lock; (c) embeddings are lightweight. The
/// local backend contending for a briefly-held GPU is acceptable; a hard deny
/// is not.
/// S125 CH-RRK-01: `POST /v1/rerank` — cross-encoder reranking. Proxies `{query,
/// documents[], top_n?}` to the sovereign reranker serve (`RERANK_URL`, default the local
/// bge-reranker-v2-m3 on `:8091`) and returns `{results:[{index,score}], model}` sorted by
/// score desc. Same JWT auth as every endpoint. No GPU-exclusive gate: the reranker runs
/// CPU-only, so it never contends for the GPU a sweep may hold.
#[derive(serde::Deserialize)]
pub struct RerankRequest {
    pub query: String,
    #[serde(default)]
    pub documents: Vec<String>,
    #[serde(default)]
    pub top_n: Option<usize>,
}

pub async fn rerank(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<RerankRequest>,
) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }
    if req.query.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "query is required" })),
        )
            .into_response();
    }
    if req.documents.is_empty() {
        return (
            StatusCode::OK,
            Json(serde_json::json!({ "results": [] })),
        )
            .into_response();
    }
    let url = std::env::var("RERANK_URL")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "http://127.0.0.1:8091/rerank".to_string());
    let body = serde_json::json!({
        "query": req.query, "documents": req.documents, "top_n": req.top_n,
    });
    match state
        .http_client
        .post(&url)
        .json(&body)
        .timeout(std::time::Duration::from_secs(120))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => match resp.json::<serde_json::Value>().await {
            Ok(v) => (StatusCode::OK, Json(v)).into_response(),
            Err(e) => (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({ "error": "rerank upstream parse error", "detail": e.to_string() })),
            )
                .into_response(),
        },
        Ok(resp) => {
            let code = resp.status().as_u16();
            (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({ "error": "rerank upstream error", "upstream_status": code })),
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({ "error": "rerank serve unreachable", "detail": e.to_string() })),
        )
            .into_response(),
    }
}

pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<crate::embeddings::EmbeddingsRequest>,
) -> Response {
    let claims = match auth_check(&headers, &state.jwt_secret) {
        Ok(c) => c,
        Err(e) => {
            let raw = headers
                .get("Authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|h| extract_bearer(h).ok());
            state.audit_logger.log_auth_failure(raw, 0);
            return auth_error_response(e);
        }
    };

    // ── BLD-09 idle-mode admission ──────────────────────────────────────────────
    // /v1/embeddings is an inference entry point: the local-first path dispatches to
    // a GPU-resident embedding model and contends for VRAM, so it MUST join the
    // closed-world drain / lazy-restore transition exactly like the other inference
    // routes (cycle-2 fix #2 — previously this route bypassed admission entirely).
    // Non-streaming (buffered JSON), so the guard held to handler return is correct.
    let _idle_inflight = match crate::admin::idle::admit_inference(&state).await {
        crate::admin::idle::Admission::Admitted(g) => g,
        crate::admin::idle::Admission::Rejected(resp) => return resp,
    };

    let inputs = match crate::embeddings::validate_inputs(req, state.embeddings_config.max_batch) {
        Ok(inputs) => inputs,
        Err(e) => {
            let status = e.status();
            return (status, Json(e.to_json())).into_response();
        }
    };

    let role = UserRole::from_claim(claims.role.as_deref());
    let rl_result = {
        let mut rl = state.rate_limiter.lock().await;
        rl.check_and_record(&claims.sub, role, CallType::Llm)
    };
    if !rl_result.allowed {
        return rate_limit_exceeded_response(&rl_result, CallType::Llm);
    }
    let rl_headers = rate_limit_headers(&rl_result);

    let openrouter_key = crate::embeddings::openrouter_api_key();
    match crate::embeddings::route_embeddings(
        &state.http_client,
        &state.embeddings_config,
        &inputs,
        openrouter_key.as_deref(),
    )
    .await
    {
        Ok(outcome) => {
            // Never logs input text — only the count and which path served it.
            tracing::info!(
                source = outcome.source.as_str(),
                count = inputs.len(),
                "embeddings: served"
            );
            let resp = crate::embeddings::build_response(&inputs, outcome);
            let mut response = Json(resp).into_response();
            response.headers_mut().extend(rl_headers);
            response
        }
        Err(e) => {
            warn!("embeddings: both backends failed: {e:?}");
            let mut response = (e.status(), Json(e.to_json())).into_response();
            response.headers_mut().extend(rl_headers);
            response
        }
    }
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
pub async fn audit_summary(State(state): State<Arc<AppState>>) -> impl IntoResponse {
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
        // Task 2 (federation): terminus_personal's ~147-tool catalog,
        // reachable ONLY here — never merged into /v1/tools/list above.
        .route(
            "/v1/personal/tools/list",
            axum::routing::post(personal_tools_list),
        )
        .route(
            "/v1/personal/tools/call",
            axum::routing::post(personal_tools_call),
        )
        .route("/v1/agent/execute", axum::routing::post(agent_execute))
        .route(
            "/v1/chat/completions",
            axum::routing::post(chat_completions),
        )
        .route("/v1/embeddings", axum::routing::post(embeddings))
        // S125 CH-RRK-01: cross-encoder reranking (proxied to the sovereign reranker serve).
        .route("/v1/rerank", axum::routing::post(rerank))
        .route("/v1/infer", axum::routing::post(infer))
        // GPU-exclusive coordination: the intake harness ACQUIREs the GPU here
        // instead of `systemctl stop chord.service` — Chord stays up, only its
        // inference paths gate. Same JWT auth as every other endpoint.
        .route(
            "/v1/gpu-exclusive/acquire",
            axum::routing::post(gpu_exclusive_acquire),
        )
        .route(
            "/v1/gpu-exclusive/release",
            axum::routing::post(gpu_exclusive_release),
        )
        .route(
            "/v1/gpu-exclusive/status",
            axum::routing::get(gpu_exclusive_status),
        )
        // Sweep-status observability (no auth, same bar as /health and
        // /v1/audit/summary — aggregate health, no identities/secrets).
        .merge(crate::sweep_status::api::sweep_status_routes())
        .route(
            "/v1/coding/select",
            axum::routing::post(crate::coding_proxy::coding_select),
        )
        // S125 CH-SEL-01: capability-aware model resolution from a TaskDescriptor.
        .route("/v1/resolve", axum::routing::post(resolve_task))
        .with_state(state)
}

/// S125 CH-SEL-01/CH-CAP-02: `POST /v1/resolve` — resolve a [`TaskDescriptor`] to a concrete
/// `(model, backend)` using the capability registry + live model registry. Clients send *what
/// the work needs* (task + constraints) instead of a raw model name. 400 on a self-
/// contradictory descriptor (advisory modality check); 404 when no model can serve the task.
pub async fn resolve_task(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(desc): Json<crate::models::selector::TaskDescriptor>,
) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }
    if let Err(reason) = desc.modality_consistent() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "modality_inconsistent", "reason": reason })),
        )
            .into_response();
    }
    // Capabilities are config-driven (CHORD_CAPABILITIES_PATH) + MINT-refreshable. Loaded per
    // request so config/MINT updates are picked up; this is a routing-decision endpoint, not
    // the hot inference path.
    let caps = crate::models::capability::CapabilityRegistry::from_env();
    let candidates = {
        let reg = state.model_registry.lock().await;
        crate::models::selector::candidates_from(&reg, &caps)
    };
    match crate::models::selector::resolve(&desc, &caps, &candidates) {
        Some(sel) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "model": sel.model, "backend": sel.backend, "reason": sel.reason
            })),
        )
            .into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "no_model_for_task",
                "task": format!("{:?}", desc.task),
            })),
        )
            .into_response(),
    }
}

// ── GPU-exclusive coordination endpoints ─────────────────────────────────────
//
// The intake benchmarking harness (Terminus `intake::gpu_authority`) calls
// these to take exclusive GPU control WITHOUT taking Chord down. See
// `crate::gpu_exclusive` for the lock/TTL/heartbeat model.

/// Build the structured 503 an inference path returns while the GPU is
/// exclusively held. Shared by `chat_completions` and `infer` so the contract is
/// identical on every gated path.
fn gpu_exclusively_held_response(record: &crate::gpu_exclusive::LockRecord) -> Response {
    let body = serde_json::json!({
        "error": "gpu_exclusively_held",
        "holder": record.holder,
        "since": crate::gpu_exclusive::iso_utc(record.acquired_at),
    });
    (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response()
}

#[derive(Deserialize)]
pub struct GpuExclusiveBody {
    /// Short label identifying the acquirer (e.g. `intake_coder_sweep`).
    pub holder: Option<String>,
    /// S125 CHRD-GPUX: force the grant even while a client request is in flight
    /// (bypass the client-yield guard). Default false — a fresh grab is refused while
    /// Chord is actively serving so MINT yields to the client.
    #[serde(default)]
    pub force: bool,
}

/// `POST /v1/gpu-exclusive/acquire` — grant the GPU to `holder`. On a fresh
/// grant Chord best-effort evicts any resident Ollama model; a re-acquire by the
/// same holder is a heartbeat refresh (no re-eviction). 409 if a DIFFERENT live
/// holder owns it.
pub async fn gpu_exclusive_acquire(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<GpuExclusiveBody>,
) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }
    let holder = body.holder.unwrap_or_default();
    if holder.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "holder required" })),
        )
            .into_response();
    }
    let now = crate::gpu_exclusive::now_epoch();
    match crate::gpu_exclusive::GPU_EXCLUSIVE.acquire(&holder, now) {
        crate::gpu_exclusive::AcquireOutcome::Granted { record, new_grant } => {
            // S125 CHRD-GPUX client-guard: the ATOMIC, all-paths backstop to MINT's own
            // fast-path probe. On a FRESH grant, if a client request is in flight, RELEASE
            // and yield BEFORE any eviction — so the harmful action (evicting the client's
            // resident model) never runs while a client is being served. This is atomic
            // w.r.t. eviction: once we hold the lock, new clients are 503'd (they see the
            // lock held), so `inflight` reflects only clients admitted before the lock;
            // refusing here leaves them untouched. A heartbeat (new_grant=false) never
            // evicts, and `force` overrides. Covers every MINT acquisition path (all route
            // through here), closing the probe→acquire window from the fast-path check.
            if new_grant && !body.force {
                let client_inflight = crate::admin::idle::IDLE_MODE.inflight_count();
                if client_inflight > 0 {
                    let _ = crate::gpu_exclusive::GPU_EXCLUSIVE.release(&holder);
                    tracing::info!(
                        requested_by = %holder, client_inflight,
                        "gpu-exclusive: yielding to in-flight client — released without evicting (S125)"
                    );
                    return (
                        StatusCode::CONFLICT,
                        Json(serde_json::json!({
                            "error": "gpu_yield_client_busy",
                            "inflight": client_inflight,
                        })),
                    )
                        .into_response();
                }
            }
            if new_grant {
                tracing::info!(holder = %record.holder, "gpu-exclusive: granted — evicting resident models");
                if let Some(base) = crate::gpu_exclusive::ollama_base_from_env() {
                    let unloaded =
                        crate::gpu_exclusive::evict_resident_models(&state.http_client, &base)
                            .await;
                    tracing::info!(holder = %record.holder, unloaded, "gpu-exclusive: acquire eviction complete");
                }
                // CHRD-DIFF-01: the managed DiffusionGemma daemon is not tracked
                // by Ollama's /api/ps (it's a bare llama-diffusion-daemon
                // process), so evict_resident_models above never sees it — stop
                // it explicitly so a fresh exclusive grant never contends with
                // Chord's own managed daemon for VRAM.
                if crate::diffusion::global().stop().await {
                    tracing::info!(holder = %record.holder, "gpu-exclusive: stopped managed DiffusionGemma daemon");
                }
            }
            let body = serde_json::json!({
                "status": "acquired",
                "holder": record.holder,
                "since": crate::gpu_exclusive::iso_utc(record.acquired_at),
                "new_grant": new_grant,
            });
            (StatusCode::OK, Json(body)).into_response()
        }
        crate::gpu_exclusive::AcquireOutcome::Blocked { record } => {
            tracing::warn!(
                requested_by = %holder, current_holder = %record.holder,
                "gpu-exclusive: acquire refused — held by another"
            );
            (
                StatusCode::CONFLICT,
                Json(serde_json::json!({
                    "error": "gpu_exclusively_held",
                    "holder": record.holder,
                    "since": crate::gpu_exclusive::iso_utc(record.acquired_at),
                })),
            )
                .into_response()
        }
    }
}

/// `POST /v1/gpu-exclusive/release` — clear `holder`'s lock and resume normal
/// serving. Idempotent (no lock ⇒ 200). 409 if a DIFFERENT holder owns it.
pub async fn gpu_exclusive_release(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<GpuExclusiveBody>,
) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }
    let holder = body.holder.unwrap_or_default();
    if holder.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "holder required" })),
        )
            .into_response();
    }
    match crate::gpu_exclusive::GPU_EXCLUSIVE.release(&holder) {
        crate::gpu_exclusive::ReleaseOutcome::Released => {
            tracing::info!(holder = %holder, "gpu-exclusive: released — normal serving resumed");
            (
                StatusCode::OK,
                Json(serde_json::json!({ "status": "released" })),
            )
                .into_response()
        }
        crate::gpu_exclusive::ReleaseOutcome::Mismatch { record } => {
            tracing::warn!(
                requested_by = %holder, current_holder = %record.holder,
                "gpu-exclusive: release refused — held by another"
            );
            (
                StatusCode::CONFLICT,
                Json(serde_json::json!({
                    "error": "lock_held_by_other",
                    "holder": record.holder,
                    "since": crate::gpu_exclusive::iso_utc(record.acquired_at),
                })),
            )
                .into_response()
        }
    }
}

/// `GET /v1/gpu-exclusive/status` — inspect the current lock (holder, since,
/// last heartbeat, whether it has expired/abandoned) for stale-lock diagnosis.
pub async fn gpu_exclusive_status(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }
    let now = crate::gpu_exclusive::now_epoch();
    let ttl = crate::gpu_exclusive::GPU_EXCLUSIVE.ttl();
    let body = match crate::gpu_exclusive::GPU_EXCLUSIVE.snapshot(now) {
        Some((record, expired)) => serde_json::json!({
            "held": !expired,
            "holder": record.holder,
            "since": crate::gpu_exclusive::iso_utc(record.acquired_at),
            "last_heartbeat": crate::gpu_exclusive::iso_utc(record.last_heartbeat),
            "expired": expired,
            "ttl_secs": ttl,
        }),
        None => serde_json::json!({ "held": false, "ttl_secs": ttl }),
    };
    (StatusCode::OK, Json(body)).into_response()
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
    // BLD-09 idle-mode admission (see `chat_completions`): join the closed-world drain
    // / lazy-restore path, or short-circuit with a retryable 503 during a transition.
    let _idle_inflight = match crate::admin::idle::admit_inference(&state).await {
        crate::admin::idle::Admission::Admitted(g) => g,
        crate::admin::idle::Admission::Rejected(resp) => return resp,
    };

    // GPU-exclusive gate (see `chat_completions`): never run inference while the
    // GPU is exclusively held — return the same structured 503 instead.
    if let Some(record) =
        crate::gpu_exclusive::GPU_EXCLUSIVE.active_holder(crate::gpu_exclusive::now_epoch())
    {
        return gpu_exclusively_held_response(&record);
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
    use serial_test::serial;
    use tower::ServiceExt;

    use crate::agentic::AgenticExecutor;
    use crate::audit::AuditLogger;
    use crate::config::{Config, RateLimitConfig};
    use crate::mcp_proxy::{FallbackRegistry, FallbackTool, McpProxy};

    struct PingTool;
    #[async_trait::async_trait]
    impl FallbackTool for PingTool {
        fn name(&self) -> &str {
            "gitea_ping"
        }
        fn description(&self) -> &str {
            "Ping"
        }
        fn parameters(&self) -> Value {
            serde_json::json!({})
        }
        async fn execute(&self, _: Value) -> Result<String, ProxyError> {
            Ok("pong".into())
        }
    }

    /// A tool whose backend failure message embeds content that could plausibly
    /// derive from caller-supplied arguments (e.g. an upstream API echoing back
    /// an invalid field value). Used to prove `proxy_error_kind()` strips this
    /// down to a fixed classification rather than persisting it verbatim.
    struct FailingTool;
    #[async_trait::async_trait]
    impl FallbackTool for FailingTool {
        fn name(&self) -> &str {
            "gitea_fail_test"
        }
        fn description(&self) -> &str {
            "Always fails with backend-echoed content"
        }
        fn parameters(&self) -> Value {
            serde_json::json!({})
        }
        async fn execute(&self, _: Value) -> Result<String, ProxyError> {
            Err(ProxyError::ToolExecution(
                "backend rejected field api_key=<REDACTED-SECRET>".into(),
            ))
        }
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
            model_archive_copy_timeout_secs: 1800,
            model_gc_min_age_secs: 300,
            model_source_allowlist: Vec::new(),
            outbound_proxy: None,
            runtime_telemetry_off: true,
            mcp_backend_token: None,
            personal_backend_url: None,
            personal_backend_token: None,
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
                model_archive_copy_timeout_secs: 1800,
                model_gc_min_age_secs: 300,
                model_source_allowlist: Vec::new(),
                outbound_proxy: None,
                runtime_telemetry_off: true,
                mcp_backend_token: None,
                personal_backend_url: None,
                personal_backend_token: None,
            },
            Arc::new(FallbackRegistry::new()),
        ));
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
            llm_backend_url: None,
            model_aliases: std::collections::HashMap::new(),
            http_client: reqwest::Client::new(),
            model_registry,
            pull_coordinator,
            local_evictor: std::sync::Arc::new(crate::models::eviction::FsLocalEvictor::new(
                std::path::PathBuf::from("/tmp"),
            )),
            disk_op_lock: crate::models::eviction::new_disk_op_lock(),
            disk_probe: std::sync::Arc::new(crate::models::transfer::StatvfsProbe),
            disk_pressure_percent: 80,
            model_warm_cooldown_hours: 168,
            model_archive_copy_timeout_secs: 1800,
            model_gc_min_age_secs: 300,
            routing_map: empty_routing_map(),
            coding_profile_source: empty_coding_profile_source(),
            personal_proxy: None,
            embeddings_config: crate::embeddings::EmbeddingsConfig::test_default(
                None,
                "http://127.0.0.1:1".to_string(),
            ),
        })
    }

    /// Like `test_state` but with the embeddings fallback pointed at
    /// `fallback_base` (local disabled) — used by the GPU-exclusive gate test
    /// to prove `/v1/embeddings` reaches its fallback path while the GPU lock
    /// is held.
    fn test_state_with_embeddings_fallback(fallback_base: String) -> Arc<AppState> {
        let base = test_state("http://mcp.invalid:3200".into());
        // AppState is behind an Arc and not Clone; rebuild the one field we need
        // to change by constructing a fresh state that shares the base's config.
        Arc::new(AppState {
            proxy: McpProxy::new(&Config::test_default(), Arc::new(FallbackRegistry::new())),
            jwt_secret: String::new(),
            audit_logger: base.audit_logger.clone(),
            rate_limiter: base.rate_limiter.clone(),
            agentic_executor: base.agentic_executor.clone(),
            llm_backend_url: None,
            model_aliases: std::collections::HashMap::new(),
            http_client: reqwest::Client::new(),
            model_registry: base.model_registry.clone(),
            pull_coordinator: base.pull_coordinator.clone(),
            local_evictor: base.local_evictor.clone(),
            disk_op_lock: crate::models::eviction::new_disk_op_lock(),
            disk_probe: std::sync::Arc::new(crate::models::transfer::StatvfsProbe),
            disk_pressure_percent: 80,
            model_warm_cooldown_hours: 168,
            model_archive_copy_timeout_secs: 1800,
            model_gc_min_age_secs: 300,
            routing_map: empty_routing_map(),
            coding_profile_source: empty_coding_profile_source(),
            personal_proxy: None,
            embeddings_config: crate::embeddings::EmbeddingsConfig::test_default(
                None,
                fallback_base,
            ),
        })
    }

    /// Task 2: like `test_state`, but with `personal_proxy` set to an
    /// unfiltered `McpProxy` pointed at `personal_url`. Used by the
    /// `/v1/personal/tools/*` tests below. `mcp_url` still backs the
    /// default/core `proxy` exactly as in `test_state` — this builder only
    /// adds the second, independent federation proxy on top.
    fn test_state_with_personal(mcp_url: String, personal_url: String) -> Arc<AppState> {
        let mut reg = FallbackRegistry::new();
        reg.register(Box::new(PingTool));
        let config = Config {
            mcp_backend_url: mcp_url,
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
            model_archive_copy_timeout_secs: 1800,
            model_gc_min_age_secs: 300,
            model_source_allowlist: Vec::new(),
            outbound_proxy: None,
            runtime_telemetry_off: true,
            mcp_backend_token: None,
            personal_backend_url: Some(personal_url.clone()),
            personal_backend_token: None,
        };
        let proxy = McpProxy::new(&config, Arc::new(reg));

        let personal_config = Config {
            mcp_backend_url: personal_url,
            ..config.clone()
        };
        let personal_proxy = Some(Arc::new(McpProxy::new_unfiltered(
            &personal_config,
            Arc::new(FallbackRegistry::new()),
        )));

        let proxy_arc = Arc::new(McpProxy::new(
            &Config {
                mcp_backend_url: "http://does-not-exist:9999".into(),
                ..config.clone()
            },
            Arc::new(FallbackRegistry::new()),
        ));
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
            llm_backend_url: None,
            model_aliases: std::collections::HashMap::new(),
            http_client: reqwest::Client::new(),
            model_registry,
            pull_coordinator,
            local_evictor: std::sync::Arc::new(crate::models::eviction::FsLocalEvictor::new(
                std::path::PathBuf::from("/tmp"),
            )),
            disk_op_lock: crate::models::eviction::new_disk_op_lock(),
            disk_probe: std::sync::Arc::new(crate::models::transfer::StatvfsProbe),
            disk_pressure_percent: 80,
            model_warm_cooldown_hours: 168,
            model_archive_copy_timeout_secs: 1800,
            model_gc_min_age_secs: 300,
            routing_map: empty_routing_map(),
            coding_profile_source: empty_coding_profile_source(),
            personal_proxy,
            embeddings_config: crate::embeddings::EmbeddingsConfig::test_default(
                None,
                "http://127.0.0.1:1".to_string(),
            ),
        })
    }

    /// Like `test_state`, but the audit logger writes to a real file (instead of
    /// `/dev/null`) so tests can assert on the JSONL entries it produces.
    fn test_state_with_audit_path(
        mcp_url: String,
        audit_path: std::path::PathBuf,
    ) -> Arc<AppState> {
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
            model_archive_copy_timeout_secs: 1800,
            model_gc_min_age_secs: 300,
            model_source_allowlist: Vec::new(),
            outbound_proxy: None,
            runtime_telemetry_off: true,
            mcp_backend_token: None,
            personal_backend_url: None,
            personal_backend_token: None,
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
                model_archive_copy_timeout_secs: 1800,
                model_gc_min_age_secs: 300,
                model_source_allowlist: Vec::new(),
                outbound_proxy: None,
                runtime_telemetry_off: true,
                mcp_backend_token: None,
                personal_backend_url: None,
                personal_backend_token: None,
            },
            Arc::new(FallbackRegistry::new()),
        ));
        let agentic_executor = Arc::new(AgenticExecutor::new(proxy_arc));
        let rate_limiter = Arc::new(Mutex::new(ProxyRateLimiter::new(default_rate_config())));
        let audit_logger = Arc::new(AuditLogger::new(audit_path));
        let (model_registry, pull_coordinator) = empty_model_state();
        Arc::new(AppState {
            proxy,
            jwt_secret: String::new(),
            audit_logger,
            rate_limiter,
            agentic_executor,
            llm_backend_url: None,
            model_aliases: std::collections::HashMap::new(),
            http_client: reqwest::Client::new(),
            model_registry,
            pull_coordinator,
            local_evictor: std::sync::Arc::new(crate::models::eviction::FsLocalEvictor::new(
                std::path::PathBuf::from("/tmp"),
            )),
            disk_op_lock: crate::models::eviction::new_disk_op_lock(),
            disk_probe: std::sync::Arc::new(crate::models::transfer::StatvfsProbe),
            disk_pressure_percent: 80,
            model_warm_cooldown_hours: 168,
            model_archive_copy_timeout_secs: 1800,
            model_gc_min_age_secs: 300,
            routing_map: empty_routing_map(),
            coding_profile_source: empty_coding_profile_source(),
            personal_proxy: None,
            embeddings_config: crate::embeddings::EmbeddingsConfig::test_default(
                None,
                "http://127.0.0.1:1".to_string(),
            ),
        })
    }

    /// Like `test_state_with_audit_path`, but registers `FailingTool` instead of
    /// `PingTool` — for tests proving a `ToolExecution` error (whose Display text
    /// may echo backend-side content derived from caller arguments) never
    /// reaches the persisted audit log.
    fn test_state_with_failing_tool_and_audit_path(
        mcp_url: String,
        audit_path: std::path::PathBuf,
    ) -> Arc<AppState> {
        let mut reg = FallbackRegistry::new();
        reg.register(Box::new(FailingTool));
        let config = Config {
            mcp_backend_url: mcp_url,
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
            model_archive_copy_timeout_secs: 1800,
            model_gc_min_age_secs: 300,
            model_source_allowlist: Vec::new(),
            outbound_proxy: None,
            runtime_telemetry_off: true,
            mcp_backend_token: None,
            personal_backend_url: None,
            personal_backend_token: None,
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
                model_archive_copy_timeout_secs: 1800,
                model_gc_min_age_secs: 300,
                model_source_allowlist: Vec::new(),
                outbound_proxy: None,
                runtime_telemetry_off: true,
                mcp_backend_token: None,
                personal_backend_url: None,
                personal_backend_token: None,
            },
            Arc::new(FallbackRegistry::new()),
        ));
        let agentic_executor = Arc::new(AgenticExecutor::new(proxy_arc));
        let rate_limiter = Arc::new(Mutex::new(ProxyRateLimiter::new(default_rate_config())));
        let audit_logger = Arc::new(AuditLogger::new(audit_path));
        let (model_registry, pull_coordinator) = empty_model_state();
        Arc::new(AppState {
            proxy,
            jwt_secret: String::new(),
            audit_logger,
            rate_limiter,
            agentic_executor,
            llm_backend_url: None,
            model_aliases: std::collections::HashMap::new(),
            http_client: reqwest::Client::new(),
            model_registry,
            pull_coordinator,
            local_evictor: std::sync::Arc::new(crate::models::eviction::FsLocalEvictor::new(
                std::path::PathBuf::from("/tmp"),
            )),
            disk_op_lock: crate::models::eviction::new_disk_op_lock(),
            disk_probe: std::sync::Arc::new(crate::models::transfer::StatvfsProbe),
            disk_pressure_percent: 80,
            model_warm_cooldown_hours: 168,
            model_archive_copy_timeout_secs: 1800,
            model_gc_min_age_secs: 300,
            routing_map: empty_routing_map(),
            coding_profile_source: empty_coding_profile_source(),
            personal_proxy: None,
            embeddings_config: crate::embeddings::EmbeddingsConfig::test_default(
                None,
                "http://127.0.0.1:1".to_string(),
            ),
        })
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
            model_archive_copy_timeout_secs: 1800,
            model_gc_min_age_secs: 300,
            model_source_allowlist: Vec::new(),
            outbound_proxy: None,
            runtime_telemetry_off: true,
            mcp_backend_token: None,
            personal_backend_url: None,
            personal_backend_token: None,
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
            local_evictor: std::sync::Arc::new(crate::models::eviction::FsLocalEvictor::new(
                std::path::PathBuf::from("/tmp"),
            )),
            disk_op_lock: crate::models::eviction::new_disk_op_lock(),
            disk_probe: std::sync::Arc::new(crate::models::transfer::StatvfsProbe),
            disk_pressure_percent: 80,
            model_warm_cooldown_hours: 168,
            model_archive_copy_timeout_secs: 1800,
            model_gc_min_age_secs: 300,
            routing_map,
            coding_profile_source: empty_coding_profile_source(),
            personal_proxy: None,
            embeddings_config: crate::embeddings::EmbeddingsConfig::test_default(
                None,
                "http://127.0.0.1:1".to_string(),
            ),
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
            model_archive_copy_timeout_secs: 1800,
            model_gc_min_age_secs: 300,
            model_source_allowlist: Vec::new(),
            outbound_proxy: None,
            runtime_telemetry_off: true,
            mcp_backend_token: None,
            personal_backend_url: None,
            personal_backend_token: None,
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
                model_archive_copy_timeout_secs: 1800,
                model_gc_min_age_secs: 300,
                model_source_allowlist: Vec::new(),
                outbound_proxy: None,
                runtime_telemetry_off: true,
                mcp_backend_token: None,
                personal_backend_url: None,
                personal_backend_token: None,
            },
            Arc::new(FallbackRegistry::new()),
        ));
        let agentic_executor = Arc::new(AgenticExecutor::new(proxy_arc));
        let rate_limiter = Arc::new(Mutex::new(ProxyRateLimiter::new(default_rate_config())));
        let audit_logger = Arc::new(AuditLogger::new(std::path::PathBuf::from("/dev/null")));
        let (model_registry, pull_coordinator) = empty_model_state();
        Arc::new(AppState {
            proxy,
            jwt_secret: secret,
            audit_logger,
            rate_limiter,
            agentic_executor,
            llm_backend_url: None,
            model_aliases: std::collections::HashMap::new(),
            http_client: reqwest::Client::new(),
            model_registry,
            pull_coordinator,
            local_evictor: std::sync::Arc::new(crate::models::eviction::FsLocalEvictor::new(
                std::path::PathBuf::from("/tmp"),
            )),
            disk_op_lock: crate::models::eviction::new_disk_op_lock(),
            disk_probe: std::sync::Arc::new(crate::models::transfer::StatvfsProbe),
            disk_pressure_percent: 80,
            model_warm_cooldown_hours: 168,
            model_archive_copy_timeout_secs: 1800,
            model_gc_min_age_secs: 300,
            routing_map: empty_routing_map(),
            coding_profile_source: empty_coding_profile_source(),
            personal_proxy: None,
            embeddings_config: crate::embeddings::EmbeddingsConfig::test_default(
                None,
                "http://127.0.0.1:1".to_string(),
            ),
        })
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
            model_archive_copy_timeout_secs: 1800,
            model_gc_min_age_secs: 300,
            model_source_allowlist: Vec::new(),
            outbound_proxy: None,
            runtime_telemetry_off: true,
            mcp_backend_token: None,
            personal_backend_url: None,
            personal_backend_token: None,
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
                model_archive_copy_timeout_secs: 1800,
                model_gc_min_age_secs: 300,
                model_source_allowlist: Vec::new(),
                outbound_proxy: None,
                runtime_telemetry_off: true,
                mcp_backend_token: None,
                personal_backend_url: None,
                personal_backend_token: None,
            },
            Arc::new(FallbackRegistry::new()),
        ));
        let agentic_executor = Arc::new(AgenticExecutor::new(proxy_arc));
        let rate_limiter = Arc::new(Mutex::new(ProxyRateLimiter::new(tight)));
        let audit_logger = Arc::new(AuditLogger::new(std::path::PathBuf::from("/dev/null")));
        let (model_registry, pull_coordinator) = empty_model_state();
        Arc::new(AppState {
            proxy,
            jwt_secret: String::new(),
            audit_logger,
            rate_limiter,
            agentic_executor,
            llm_backend_url: None,
            model_aliases: std::collections::HashMap::new(),
            http_client: reqwest::Client::new(),
            model_registry,
            pull_coordinator,
            local_evictor: std::sync::Arc::new(crate::models::eviction::FsLocalEvictor::new(
                std::path::PathBuf::from("/tmp"),
            )),
            disk_op_lock: crate::models::eviction::new_disk_op_lock(),
            disk_probe: std::sync::Arc::new(crate::models::transfer::StatvfsProbe),
            disk_pressure_percent: 80,
            model_warm_cooldown_hours: 168,
            model_archive_copy_timeout_secs: 1800,
            model_gc_min_age_secs: 300,
            routing_map: empty_routing_map(),
            coding_profile_source: empty_coding_profile_source(),
            personal_proxy: None,
            embeddings_config: crate::embeddings::EmbeddingsConfig::test_default(
                None,
                "http://127.0.0.1:1".to_string(),
            ),
        })
    }

    /// Like `test_state_tight_limits`, but the audit logger writes to a real
    /// file so a 429 rate-limit rejection's audit entry can be asserted on.
    fn test_state_tight_limits_with_audit_path(
        mcp_url: String,
        audit_path: std::path::PathBuf,
    ) -> Arc<AppState> {
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
            model_archive_copy_timeout_secs: 1800,
            model_gc_min_age_secs: 300,
            model_source_allowlist: Vec::new(),
            outbound_proxy: None,
            runtime_telemetry_off: true,
            mcp_backend_token: None,
            personal_backend_url: None,
            personal_backend_token: None,
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
                model_archive_copy_timeout_secs: 1800,
                model_gc_min_age_secs: 300,
                model_source_allowlist: Vec::new(),
                outbound_proxy: None,
                runtime_telemetry_off: true,
                mcp_backend_token: None,
                personal_backend_url: None,
                personal_backend_token: None,
            },
            Arc::new(FallbackRegistry::new()),
        ));
        let agentic_executor = Arc::new(AgenticExecutor::new(proxy_arc));
        let rate_limiter = Arc::new(Mutex::new(ProxyRateLimiter::new(tight)));
        let audit_logger = Arc::new(AuditLogger::new(audit_path));
        let (model_registry, pull_coordinator) = empty_model_state();
        Arc::new(AppState {
            proxy,
            jwt_secret: String::new(),
            audit_logger,
            rate_limiter,
            agentic_executor,
            llm_backend_url: None,
            model_aliases: std::collections::HashMap::new(),
            http_client: reqwest::Client::new(),
            model_registry,
            pull_coordinator,
            local_evictor: std::sync::Arc::new(crate::models::eviction::FsLocalEvictor::new(
                std::path::PathBuf::from("/tmp"),
            )),
            disk_op_lock: crate::models::eviction::new_disk_op_lock(),
            disk_probe: std::sync::Arc::new(crate::models::transfer::StatvfsProbe),
            disk_pressure_percent: 80,
            model_warm_cooldown_hours: 168,
            model_archive_copy_timeout_secs: 1800,
            model_gc_min_age_secs: 300,
            routing_map: empty_routing_map(),
            coding_profile_source: empty_coding_profile_source(),
            personal_proxy: None,
            embeddings_config: crate::embeddings::EmbeddingsConfig::test_default(
                None,
                "http://127.0.0.1:1".to_string(),
            ),
        })
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
        let state =
            test_state_with_secret("http://does-not-exist:9999".into(), "test-secret".into());
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
        let state =
            test_state_with_secret("http://does-not-exist:9999".into(), "test-secret".into());
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
        let state =
            test_state_with_secret("http://does-not-exist:9999".into(), "test-secret".into());
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
            "name": "gitea_ping",
            "arguments": {}
        }))
        .unwrap();

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/tools/call")
            .header("Content-Type", "application/json")
            .body(Body::from(body))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
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
        }))
        .unwrap();

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
        }))
        .unwrap();

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/tools/discover")
            .header("Content-Type", "application/json")
            .body(Body::from(body))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(json["tools"].is_array());
        assert_eq!(json["query"], "ping");
    }

    // ── Tool-dispatch audit logging (HOFIX: tool_list/tool_call/tool_discover
    // were never audited — AuditLogger only covered LLM calls) ────────────────

    fn read_audit_entries(path: &std::path::Path) -> Vec<crate::audit::AuditEntry> {
        let contents = std::fs::read_to_string(path).unwrap_or_default();
        contents
            .lines()
            .map(|l| serde_json::from_str(l).expect("audit line must be valid JSON"))
            .collect()
    }

    #[tokio::test]
    async fn test_tool_call_success_produces_audit_entry_with_tool_name() {
        let dir = tempfile::tempdir().unwrap();
        let audit_path = dir.path().join("chord-audit.jsonl");
        let state =
            test_state_with_audit_path("http://does-not-exist:9999".into(), audit_path.clone());
        let app = build_router(state);

        // Arguments deliberately contain secret-shaped content — it must never
        // reach the audit log.
        let body = serde_json::to_string(&serde_json::json!({
            "name": "gitea_ping",
            "arguments": {"token": "<REDACTED-SECRET>", "note": "hunter2"}
        }))
        .unwrap();

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/tools/call")
            .header("Content-Type", "application/json")
            .body(Body::from(body))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let entries = read_audit_entries(&audit_path);
        assert_eq!(entries.len(), 1, "exactly one audit entry expected");
        let entry = &entries[0];
        assert_eq!(entry.request_type, crate::audit::RequestType::ToolCall);
        assert_eq!(entry.target, "gitea_ping");
        assert_eq!(entry.status, crate::audit::Status::Success);

        // No argument content leaks into the log file at all.
        let raw = std::fs::read_to_string(&audit_path).unwrap();
        assert!(!raw.contains("<REDACTED-SECRET>"));
        assert!(!raw.contains("hunter2"));
    }

    #[tokio::test]
    async fn test_tool_call_error_produces_audit_entry_no_leak() {
        let dir = tempfile::tempdir().unwrap();
        let audit_path = dir.path().join("chord-audit.jsonl");
        let state =
            test_state_with_audit_path("http://does-not-exist:9999".into(), audit_path.clone());
        let app = build_router(state);

        let body = serde_json::to_string(&serde_json::json!({
            "name": "nonexistent_tool",
            "arguments": {"api_key": "<REDACTED-SECRET>"}
        }))
        .unwrap();

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/tools/call")
            .header("Content-Type", "application/json")
            .body(Body::from(body))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        let entries = read_audit_entries(&audit_path);
        assert_eq!(entries.len(), 1, "a failed tool_call must still be audited");
        let entry = &entries[0];
        assert_eq!(entry.request_type, crate::audit::RequestType::ToolCall);
        assert_eq!(entry.target, "nonexistent_tool");
        assert_eq!(entry.status, crate::audit::Status::Error);
        assert_eq!(entry.error_message.as_deref(), Some("tool_not_found"));

        let raw = std::fs::read_to_string(&audit_path).unwrap();
        assert!(!raw.contains("<REDACTED-SECRET>"));
    }

    #[tokio::test]
    async fn test_tools_list_produces_audit_entry() {
        let dir = tempfile::tempdir().unwrap();
        let audit_path = dir.path().join("chord-audit.jsonl");
        let state =
            test_state_with_audit_path("http://does-not-exist:9999".into(), audit_path.clone());
        let app = build_router(state);

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/tools/list")
            .header("Content-Type", "application/json")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let entries = read_audit_entries(&audit_path);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].request_type, crate::audit::RequestType::ToolList);
        assert_eq!(entries[0].status, crate::audit::Status::Success);
    }

    #[tokio::test]
    async fn test_tools_discover_produces_audit_entry_without_query_text() {
        let dir = tempfile::tempdir().unwrap();
        let audit_path = dir.path().join("chord-audit.jsonl");
        let state =
            test_state_with_audit_path("http://does-not-exist:9999".into(), audit_path.clone());
        let app = build_router(state);

        let sensitive_query = "find the tool that stores api_key <REDACTED-SECRET>";
        let body = serde_json::to_string(&serde_json::json!({
            "query": sensitive_query,
            "max_results": 5
        }))
        .unwrap();

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/tools/discover")
            .header("Content-Type", "application/json")
            .body(Body::from(body))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let entries = read_audit_entries(&audit_path);
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].request_type,
            crate::audit::RequestType::ToolDiscover
        );
        assert_eq!(entries[0].status, crate::audit::Status::Success);

        // The raw query text must never appear in the audit log.
        let raw = std::fs::read_to_string(&audit_path).unwrap();
        assert!(!raw.contains("<REDACTED-SECRET>"));
        assert!(!raw.contains(sensitive_query));
    }

    #[tokio::test]
    async fn test_tool_call_error_execution_variant_redacts_backend_text() {
        // ProxyError::ToolExecution's Display text can echo backend content
        // derived from caller-supplied arguments (unlike ToolNotFound, whose
        // Display carries only the tool name). Prove proxy_error_kind() strips
        // this down to a fixed classification rather than persisting it.
        let dir = tempfile::tempdir().unwrap();
        let audit_path = dir.path().join("chord-audit.jsonl");
        let state = test_state_with_failing_tool_and_audit_path(
            "http://does-not-exist:9999".into(),
            audit_path.clone(),
        );
        let app = build_router(state);

        let body = serde_json::to_string(&serde_json::json!({
            "name": "gitea_fail_test",
            "arguments": {}
        }))
        .unwrap();

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/tools/call")
            .header("Content-Type", "application/json")
            .body(Body::from(body))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);

        let entries = read_audit_entries(&audit_path);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].request_type, crate::audit::RequestType::ToolCall);
        assert_eq!(entries[0].target, "gitea_fail_test");
        assert_eq!(entries[0].status, crate::audit::Status::Error);
        assert_eq!(
            entries[0].error_message.as_deref(),
            Some("tool_execution_error")
        );

        // The backend's echoed "secret" must never appear in the audit log,
        // even though FailingTool's ProxyError::ToolExecution Display text
        // contains it.
        let raw = std::fs::read_to_string(&audit_path).unwrap();
        assert!(!raw.contains("<REDACTED-SECRET>"));
        assert!(!raw.contains("api_key"));
    }

    #[tokio::test]
    async fn test_tool_call_rate_limited_produces_audit_entry() {
        let dir = tempfile::tempdir().unwrap();
        let audit_path = dir.path().join("chord-audit.jsonl");
        let state = test_state_tight_limits_with_audit_path(
            "http://does-not-exist:9999".into(),
            audit_path.clone(),
        );
        let app = build_router(state);

        // user_tool_limit is 2 — the 3rd call in the same test should 429.
        for _ in 0..2 {
            let body = serde_json::to_string(&serde_json::json!({
                "name": "gitea_ping",
                "arguments": {}
            }))
            .unwrap();
            let req = Request::builder()
                .method(Method::POST)
                .uri("/v1/tools/call")
                .header("Content-Type", "application/json")
                .body(Body::from(body))
                .unwrap();
            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
        }

        let body = serde_json::to_string(&serde_json::json!({
            "name": "gitea_ping",
            "arguments": {"secret_arg": "<REDACTED-SECRET>"}
        }))
        .unwrap();
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/tools/call")
            .header("Content-Type", "application/json")
            .body(Body::from(body))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);

        let entries = read_audit_entries(&audit_path);
        assert_eq!(entries.len(), 3, "2 successful calls + 1 rate-limited call");
        let last = entries.last().unwrap();
        assert_eq!(last.request_type, crate::audit::RequestType::ToolCall);
        assert_eq!(last.target, "gitea_ping");
        assert_eq!(last.status, crate::audit::Status::Error);
        assert_eq!(last.error_message.as_deref(), Some("rate_limited"));

        let raw = std::fs::read_to_string(&audit_path).unwrap();
        assert!(!raw.contains("<REDACTED-SECRET>"));
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
        assert!(
            headers.contains_key("X-RateLimit-Limit"),
            "X-RateLimit-Limit must be present"
        );
        assert!(
            headers.contains_key("X-RateLimit-Remaining"),
            "X-RateLimit-Remaining must be present"
        );
        assert!(
            headers.contains_key("X-RateLimit-Reset"),
            "X-RateLimit-Reset must be present"
        );
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
        assert!(
            headers.contains_key("Retry-After"),
            "Retry-After must be present on 429"
        );
        assert!(
            headers.contains_key("X-RateLimit-Limit"),
            "X-RateLimit-Limit must be present on 429"
        );
        assert!(
            headers.contains_key("X-RateLimit-Remaining"),
            "X-RateLimit-Remaining must be present on 429"
        );
        assert!(
            headers.contains_key("X-RateLimit-Reset"),
            "X-RateLimit-Reset must be present on 429"
        );

        // Body should contain error message.
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
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
    #[serial(gpu_gate)]
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
    #[serial(gpu_gate)]
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
    #[serial(gpu_gate)]
    async fn test_chat_completions_thinking_ignored_for_non_supporting_model() {
        let server = httpmock::MockServer::start_async().await;
        // Trap: fails the test (unmatched → 404) if chat_template_kwargs leaks in.
        let trap = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/v1/chat/completions")
                .body_contains("chat_template_kwargs");
            then.status(200)
                .json_body(serde_json::json!({"choices": []}));
        });
        let real = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/v1/chat/completions");
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
            .body(Body::from(chat_request_body_with_thinking(
                "plain:7b", "on",
            )))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "ignored hint must never error"
        );
        trap.assert_hits_async(0).await;
        real.assert_hits_async(1).await;
    }

    /// EDGE CASE: `thinking:"on"` against a model whose thinking config is
    /// present but UNVALIDATED — must degrade EXACTLY like a non-supporting
    /// model (never serve an untrusted mode): ignored, no `chat_template_kwargs`.
    #[tokio::test]
    #[serial(gpu_gate)]
    async fn test_chat_completions_thinking_ignored_for_unvalidated_model() {
        let server = httpmock::MockServer::start_async().await;
        let trap = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/v1/chat/completions")
                .body_contains("chat_template_kwargs");
            then.status(200)
                .json_body(serde_json::json!({"choices": []}));
        });
        let real = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/v1/chat/completions");
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
    #[serial(gpu_gate)]
    async fn test_chat_completions_no_thinking_param_leaves_body_unchanged() {
        let server = httpmock::MockServer::start_async().await;
        let trap = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/v1/chat/completions")
                .body_contains("chat_template_kwargs");
            then.status(200)
                .json_body(serde_json::json!({"choices": []}));
        });
        let real = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/v1/chat/completions");
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
    #[serial(gpu_gate)]
    async fn test_chat_completions_malformed_thinking_value_defaults_no_crash() {
        let server = httpmock::MockServer::start_async().await;
        let trap = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/v1/chat/completions")
                .body_contains("chat_template_kwargs");
            then.status(200)
                .json_body(serde_json::json!({"choices": []}));
        });
        let real = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/v1/chat/completions");
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
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "malformed value must never crash/error"
        );
        trap.assert_hits_async(0).await;
        real.assert_hits_async(1).await;
    }

    #[tokio::test]
    #[serial(gpu_gate)]
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
    #[serial(gpu_gate)]
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

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(json["error"].as_str().unwrap().contains("not configured"));
    }

    /// EMBED-01 (reviewer fix 1): `/v1/embeddings` must NOT be gated on the
    /// GPU-exclusive lock. Acquire the lock as some holder, then confirm an
    /// embeddings request still reaches its (mocked OpenRouter) fallback and
    /// returns 200 — never the `gpu_exclusively_held` 503 that
    /// `chat_completions`/`agent_execute` return. Guards against a deadlock
    /// where an agent holding the GPU lock needs embeddings for RAG.
    #[tokio::test]
    #[serial(gpu_gate)]
    async fn test_embeddings_not_gpu_exclusive_gated() {
        let fallback = httpmock::MockServer::start_async().await;
        fallback.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200).json_body(serde_json::json!({
                "data": [{ "embedding": vec![0.1_f64; 1024], "index": 0 }]
            }));
        });

        let state = test_state_with_embeddings_fallback(fallback.base_url());
        // The fallback path needs a key present (read fresh from env at dispatch).
        std::env::set_var("OPENROUTER_API_KEY", "<REDACTED-SECRET>"); // pii-test-fixture

        // Take the process-global GPU-exclusive lock as an unrelated holder,
        // exactly as the intake sweep (or a GPU-holding agent) would.
        let holder = "embed01-gpu-gate-test";
        let now = crate::gpu_exclusive::now_epoch();
        let _ = crate::gpu_exclusive::GPU_EXCLUSIVE.acquire(holder, now);
        assert!(
            crate::gpu_exclusive::GPU_EXCLUSIVE
                .active_holder(now)
                .is_some(),
            "precondition: GPU-exclusive lock must be held for this test"
        );

        let app = build_router(state);
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/embeddings")
            .header("Content-Type", "application/json")
            .body(Body::from(r#"{"input":"hello"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status();

        // Release the global lock + env BEFORE asserting, so a failure never
        // leaks process-global state into other tests.
        let _ = crate::gpu_exclusive::GPU_EXCLUSIVE.release(holder);
        std::env::remove_var("OPENROUTER_API_KEY");

        assert_eq!(
            status,
            StatusCode::OK,
            "embeddings must succeed via fallback while the GPU is exclusively held (no GPU gate)"
        );
    }

    #[tokio::test]
    #[serial(gpu_gate)]
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
            when.method(httpmock::Method::POST)
                .path("/v1/chat/completions");
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

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["choices"][0]["message"]["content"], "Hi there!");

        mock.assert_async().await;
    }

    /// F1 regression: when the client sends a model alias (lumina-fast), the proxy
    /// must rewrite the `model` field to the real backend model (gpt-oss:20b) before
    /// forwarding, so Ollama no longer returns 404 "model lumina-fast not found".
    #[tokio::test]
    #[serial(gpu_gate)]
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
    #[serial(gpu_gate)]
    async fn test_chat_completions_streaming_preserves_sse_content_type() {
        let server = httpmock::MockServer::start_async().await;
        // Simulate an SSE stream body the way Ollama returns it.
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\" there\"}}]}\n\n\
                   data: [DONE]\n\n";
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/v1/chat/completions");
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

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(text.contains("data: "));
        assert!(text.contains("[DONE]"));

        mock.assert_async().await;
    }

    #[tokio::test]
    #[serial(gpu_gate)]
    async fn test_chat_completions_passes_through_upstream_error_status() {
        let server = httpmock::MockServer::start_async().await;
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/v1/chat/completions");
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

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("not found"));

        mock.assert_async().await;
    }

    #[tokio::test]
    #[serial(gpu_gate)]
    async fn test_chat_completions_502_when_backend_unreachable() {
        // Point at an unroutable address so the upstream send() fails.
        let state = test_state_with_llm(Some("http://127.0.0.1:1/v1/chat/completions".into()));
        let app = build_router(state);

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("Content-Type", "application/json")
            .body(Body::from(chat_request_body("gpt-oss:20b", false)))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
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
            model_archive_copy_timeout_secs: 1800,
            model_gc_min_age_secs: 300,
            model_source_allowlist: Vec::new(),
            outbound_proxy: None,
            runtime_telemetry_off: true,
            mcp_backend_token: None,
            personal_backend_url: None,
            personal_backend_token: None,
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
            local_evictor: std::sync::Arc::new(crate::models::eviction::FsLocalEvictor::new(
                std::path::PathBuf::from("/tmp"),
            )),
            disk_op_lock: crate::models::eviction::new_disk_op_lock(),
            disk_probe: std::sync::Arc::new(crate::models::transfer::StatvfsProbe),
            disk_pressure_percent: 80,
            model_warm_cooldown_hours: 168,
            model_archive_copy_timeout_secs: 1800,
            model_gc_min_age_secs: 300,
            routing_map: empty_routing_map(),
            coding_profile_source: empty_coding_profile_source(),
            personal_proxy: None,
            embeddings_config: crate::embeddings::EmbeddingsConfig::test_default(
                None,
                "http://127.0.0.1:1".to_string(),
            ),
        })
    }

    /// Write a minimal Ollama manifest + its blobs under `<root>` for `<model>:<tag>`,
    /// returning the model name. Mirrors the transfer-test layout.
    fn write_archive_model(
        root: &std::path::Path,
        model: &str,
        tag: &str,
        sizes: &[u64],
    ) -> String {
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
            fs::write(
                blobs.join(digest.replacen(':', "-", 1)),
                vec![b'x'; *size as usize],
            )
            .unwrap();
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
    #[serial(gpu_gate)]
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
            when.method(httpmock::Method::POST)
                .path("/v1/chat/completions");
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
    #[serial(gpu_gate)]
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
            when.method(httpmock::Method::POST)
                .path("/v1/chat/completions");
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
    #[serial(gpu_gate)]
    async fn test_chat_completions_unknown_model_passes_through() {
        let server = httpmock::MockServer::start_async().await;
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/v1/chat/completions");
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
            .body(Body::from(chat_request_body(
                "some-unknown-model:42",
                false,
            )))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        mock.assert_async().await;
    }

    // ── Task 2 (federation): /v1/personal/tools/* ────────────────────────────

    /// Regression guard: `/v1/tools/list`'s core catalog must be unaffected by
    /// this change when `PERSONAL_BACKEND_URL` is unset — `test_state` builds
    /// an `AppState` with `personal_proxy: None`, exactly the pre-Task-2
    /// shape. Asserts both the status and that the known Rust-fallback tool
    /// (`gitea_ping`, registered on the CORE registry only) is present,
    /// proving the core `proxy`'s tool_list path is untouched.
    #[tokio::test]
    async fn test_default_tools_list_unchanged_when_personal_unconfigured() {
        let state = test_state("http://does-not-exist:9999".into());
        assert!(
            state.personal_proxy.is_none(),
            "test_state must not configure a personal proxy"
        );
        let app = build_router(state);

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/tools/list")
            .header("Content-Type", "application/json")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        let names: Vec<&str> = json["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert!(
            names.contains(&"gitea_ping"),
            "core catalog must still contain the Rust-fallback test tool: {names:?}"
        );
    }

    /// `/v1/personal/tools/list` and `/v1/personal/tools/call` must return a
    /// clean 503 (never panic, never hang) when `PERSONAL_BACKEND_URL` is
    /// unset — i.e. `state.personal_proxy` is `None`.
    #[tokio::test]
    async fn test_personal_routes_return_503_when_unconfigured() {
        let state = test_state("http://does-not-exist:9999".into());
        let app = build_router(state);

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/personal/tools/list")
            .header("Content-Type", "application/json")
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/personal/tools/call")
            .header("Content-Type", "application/json")
            .body(Body::from(r#"{"name":"health","arguments":{}}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    /// Happy path against a mocked `terminus_personal` backend: proves
    /// `/v1/personal/tools/list` returns the backend's full (unfiltered)
    /// catalog — including a tool name that is NOT on Chord's
    /// `tool_allowlist::is_core_tool` list — and that `/v1/tools/list` (the
    /// default/core catalog) does NOT include it, demonstrating the two
    /// catalogs stay separate.
    #[tokio::test]
    async fn test_personal_tools_list_happy_path_mocked_backend() {
        let mock_server = httpmock::MockServer::start_async().await;
        mock_server.mock(|when, then| {
            when.body_contains(r#""method":"initialize""#);
            then.status(200)
                .header("Mcp-Session-Id", "personal-test")
                .json_body(serde_json::json!({"jsonrpc":"2.0","id":1,"result":{}}));
        });
        mock_server.mock(|when, then| {
            when.body_contains("notifications/initialized");
            then.status(200).body("");
        });
        mock_server.mock(|when, then| {
            when.body_contains("tools/list");
            then.status(200).json_body(serde_json::json!({
                "jsonrpc": "2.0", "id": 2,
                "result": {
                    "tools": [
                        // Deliberately NOT on tool_allowlist::is_core_tool — proves
                        // the personal proxy's catalog is unfiltered.
                        {"name": "vitals_today", "description": "Today's health vitals", "inputSchema": {}}
                    ]
                }
            }));
        });

        let state =
            test_state_with_personal("http://does-not-exist:9999".into(), mock_server.base_url());
        let app = build_router(state);

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/personal/tools/list")
            .header("Content-Type", "application/json")
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        let names: Vec<&str> = json["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert!(
            names.contains(&"vitals_today"),
            "personal catalog must include the non-core tool: {names:?}"
        );

        // The default/core catalog must NOT pick up the personal-only tool —
        // the two catalogs never merge.
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/tools/list")
            .header("Content-Type", "application/json")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        let core_names: Vec<&str> = json["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert!(
            !core_names.contains(&"vitals_today"),
            "core catalog must never merge in the personal-only tool: {core_names:?}"
        );
    }

    /// Happy path: `/v1/personal/tools/call` executes a tool through the
    /// mocked personal backend and returns its result.
    #[tokio::test]
    async fn test_personal_tools_call_happy_path_mocked_backend() {
        let mock_server = httpmock::MockServer::start_async().await;
        mock_server.mock(|when, then| {
            when.body_contains(r#""method":"initialize""#);
            then.status(200)
                .header("Mcp-Session-Id", "personal-call-test")
                .json_body(serde_json::json!({"jsonrpc":"2.0","id":1,"result":{}}));
        });
        mock_server.mock(|when, then| {
            when.body_contains("notifications/initialized");
            then.status(200).body("");
        });
        mock_server.mock(|when, then| {
            when.body_contains("tools/call");
            then.status(200).json_body(serde_json::json!({
                "jsonrpc": "2.0", "id": 2,
                "result": {
                    "content": [{"type": "text", "text": "pong-from-terminus-personal"}]
                }
            }));
        });

        let state =
            test_state_with_personal("http://does-not-exist:9999".into(), mock_server.base_url());
        let app = build_router(state);

        // Deliberately a non-core tool name (NOT on tool_allowlist::is_core_tool) —
        // proves this exercises the filter_core_tools == false bypass in
        // tool_call's hard gate, not just a name that would also pass the core
        // allowlist (a prior version of this test used "health", which is
        // core-allowlisted and so could not distinguish "bypass works" from
        // "gate was never actually reached").
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/personal/tools/call")
            .header("Content-Type", "application/json")
            .body(Body::from(r#"{"name":"vitals_today","arguments":{}}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["result"], "pong-from-terminus-personal");
        assert_eq!(json["source"], "mcp");
    }

    /// `/v1/personal/tools/list` and `/v1/personal/tools/call` must enforce the
    /// same JWT auth as the default tool routes — a request with no bearer
    /// token is rejected with 401 even when a personal backend IS configured
    /// (using a mocked server here just to prove auth is checked before the
    /// personal proxy is ever reached).
    #[tokio::test]
    async fn test_personal_routes_require_auth() {
        let mock_server = httpmock::MockServer::start_async().await;
        // No mocks registered: if auth were skipped and the proxy were reached,
        // the unmocked request would 5xx/timeout rather than 401 — making this
        // a meaningful negative test, not a vacuous pass.

        let mut reg = FallbackRegistry::new();
        reg.register(Box::new(PingTool));
        let config = Config {
            mcp_backend_url: "http://does-not-exist:9999".into(),
            jwt_secret: "<REDACTED-SECRET>".into(),
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
            model_archive_copy_timeout_secs: 1800,
            model_gc_min_age_secs: 300,
            model_source_allowlist: Vec::new(),
            outbound_proxy: None,
            runtime_telemetry_off: true,
            mcp_backend_token: None,
            personal_backend_url: Some(mock_server.base_url()),
            personal_backend_token: None,
        };
        let proxy = McpProxy::new(&config, Arc::new(reg));
        let personal_proxy = Some(Arc::new(McpProxy::new_unfiltered(
            &config,
            Arc::new(FallbackRegistry::new()),
        )));
        let proxy_arc = Arc::new(McpProxy::new(
            &Config {
                mcp_backend_url: "http://does-not-exist:9999".into(),
                ..config.clone()
            },
            Arc::new(FallbackRegistry::new()),
        ));
        let agentic_executor = Arc::new(AgenticExecutor::new(proxy_arc));
        let rate_limiter = Arc::new(Mutex::new(ProxyRateLimiter::new(default_rate_config())));
        let audit_logger = Arc::new(AuditLogger::new(std::path::PathBuf::from("/dev/null")));
        let (model_registry, pull_coordinator) = empty_model_state();
        let state = Arc::new(AppState {
            proxy,
            jwt_secret: "<REDACTED-SECRET>".into(),
            audit_logger,
            rate_limiter,
            agentic_executor,
            llm_backend_url: None,
            model_aliases: std::collections::HashMap::new(),
            http_client: reqwest::Client::new(),
            model_registry,
            pull_coordinator,
            local_evictor: std::sync::Arc::new(crate::models::eviction::FsLocalEvictor::new(
                std::path::PathBuf::from("/tmp"),
            )),
            disk_op_lock: crate::models::eviction::new_disk_op_lock(),
            disk_probe: std::sync::Arc::new(crate::models::transfer::StatvfsProbe),
            disk_pressure_percent: 80,
            model_warm_cooldown_hours: 168,
            model_archive_copy_timeout_secs: 1800,
            model_gc_min_age_secs: 300,
            routing_map: empty_routing_map(),
            coding_profile_source: empty_coding_profile_source(),
            personal_proxy,
            embeddings_config: crate::embeddings::EmbeddingsConfig::test_default(
                None,
                "http://127.0.0.1:1".to_string(),
            ),
        });
        let app = build_router(state);

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/personal/tools/list")
            .header("Content-Type", "application/json")
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/personal/tools/call")
            .header("Content-Type", "application/json")
            .body(Body::from(r#"{"name":"vitals_today","arguments":{}}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    /// LIVE integration test (Task 2): actually calls through to the REAL
    /// running `terminus_personal` backend and asserts a real tool list comes
    /// back. Feature-gated behind `personal-live-test` — NOT part of default
    /// `cargo test --workspace`. Run explicitly from a host that can reach
    /// the deployed `terminus_personal` backend with:
    ///   `PERSONAL_BACKEND_URL=<terminus_personal's base URL> \
    ///    PERSONAL_BACKEND_TOKEN=<terminus_personal's TERMINUS_PERSONAL_TOKEN> \
    ///    cargo test --features personal-live-test test_personal_live_backend -- --ignored`
    #[cfg(feature = "personal-live-test")]
    #[tokio::test]
    #[ignore = "requires network access to the real terminus_personal backend"]
    async fn test_personal_live_backend_tools_list() {
        let personal_url = std::env::var("PERSONAL_BACKEND_URL")
            .expect("set PERSONAL_BACKEND_URL to run this live test");
        let personal_token = std::env::var("PERSONAL_BACKEND_TOKEN").ok();

        let config = Config {
            mcp_backend_url: personal_url,
            jwt_secret: String::new(),
            tool_timeout_secs: 15,
            catalog_cache_secs: 0,
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
            model_archive_copy_timeout_secs: 1800,
            model_gc_min_age_secs: 300,
            model_source_allowlist: Vec::new(),
            outbound_proxy: None,
            runtime_telemetry_off: true,
            mcp_backend_token: personal_token,
            personal_backend_url: None,
            personal_backend_token: None,
        };
        let proxy = McpProxy::new_unfiltered(&config, Arc::new(FallbackRegistry::new()));
        let tools = proxy
            .tool_list()
            .await
            .expect("live terminus_personal tools/list call failed");
        assert!(
            !tools.is_empty(),
            "expected a real, non-empty tool catalog from terminus_personal"
        );
        println!(
            "LIVE terminus_personal catalog: {} tools: {:?}",
            tools.len(),
            tools.iter().map(|t| &t.name).collect::<Vec<_>>()
        );
    }

    // ── CHRD-DIFF-503: diffusion serving translation (pure helpers) ─────────
    //
    // `serve_diffusion_chat_completion` itself dispatches through the
    // process-global `crate::diffusion::global()` singleton, so it can't be
    // pointed at a per-test mock daemon the way `resolve_and_ensure`'s
    // registry-backed path can. Instead these prove the translation logic —
    // OpenAI request → (system, prompt) → daemon response → OpenAI
    // chat-completion — directly against the pure helper functions, and
    // `diffusion::tests::generate_posts_to_slash_generate_and_parses_the_daemon_contract`
    // (in `diffusion.rs`) proves the actual `/generate` HTTP call shape
    // against a mocked daemon.

    #[test]
    fn extract_diffusion_prompt_splits_system_and_flattens_conversation() {
        let body = serde_json::json!({
            "model": "diffusion-gemma",
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "how are you"},
            ],
        })
        .to_string();
        let (system, prompt) = extract_diffusion_prompt(body.as_bytes());
        assert_eq!(system, "be terse");
        assert_eq!(
            prompt,
            "User: hi\nAssistant: hello\nUser: how are you"
        );
    }

    #[test]
    fn extract_diffusion_prompt_joins_multiple_system_messages() {
        let body = serde_json::json!({
            "messages": [
                {"role": "system", "content": "rule one"},
                {"role": "system", "content": "rule two"},
                {"role": "user", "content": "go"},
            ],
        })
        .to_string();
        let (system, prompt) = extract_diffusion_prompt(body.as_bytes());
        assert_eq!(system, "rule one\n\nrule two");
        assert_eq!(prompt, "User: go");
    }

    #[test]
    fn extract_diffusion_prompt_handles_multipart_content_and_skips_non_text() {
        let body = serde_json::json!({
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "part one"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
                    {"type": "text", "text": "part two"},
                ]},
            ],
        })
        .to_string();
        let (_, prompt) = extract_diffusion_prompt(body.as_bytes());
        assert_eq!(prompt, "User: part one\npart two");
    }

    #[test]
    fn extract_diffusion_prompt_malformed_body_yields_empty_strings() {
        let (system, prompt) = extract_diffusion_prompt(b"not json");
        assert!(system.is_empty());
        assert!(prompt.is_empty());
    }

    #[test]
    fn parse_max_tokens_from_body_reads_and_defaults() {
        let body = serde_json::json!({"model": "x", "max_tokens": 256}).to_string();
        assert_eq!(parse_max_tokens_from_body(body.as_bytes()), Some(256));
        assert_eq!(parse_max_tokens_from_body(b"{}"), None);
        assert_eq!(parse_max_tokens_from_body(b"not json"), None);
    }

    #[test]
    fn diffusion_generate_wraps_as_valid_openai_chat_completion() {
        let gen = crate::diffusion::DiffusionGenerateResponse {
            text: "the answer is 42".to_string(),
            time_ms: 500,
            model_load_ms: 0,
            input_tokens: 10,
            tokens: 6,
            blocks: 1,
        };
        let resp = diffusion_generate_to_openai_response("diffusion-gemma", &gen);

        assert_eq!(resp["object"], "chat.completion");
        assert_eq!(resp["model"], "diffusion-gemma");
        assert_eq!(
            resp["choices"][0]["message"]["role"],
            "assistant"
        );
        assert_eq!(
            resp["choices"][0]["message"]["content"],
            "the answer is 42"
        );
        assert_eq!(resp["choices"][0]["finish_reason"], "stop");
        assert_eq!(resp["choices"][0]["index"], 0);
        assert_eq!(resp["usage"]["prompt_tokens"], 10);
        assert_eq!(resp["usage"]["completion_tokens"], 6);
        assert_eq!(resp["usage"]["total_tokens"], 16);
        assert!(resp["id"].as_str().unwrap().starts_with("chatcmpl-"));
    }

    /// End-to-end through the real `chat_completions` handler: a diffusion
    /// model request must NEVER reach `CHORD_LLM_URL` (proving the old
    /// raw-forward-to-a-nonexistent-endpoint bug is gone — the diffusion
    /// branch now returns its own structured 503 from `ensure_running`
    /// failing in this sandbox, where no real daemon binary/process exists,
    /// rather than falling through to any other upstream).
    #[tokio::test]
    #[serial(gpu_gate)]
    async fn test_chat_completions_diffusion_model_never_forwards_to_chord_llm_url() {
        let server = httpmock::MockServer::start_async().await;
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/v1/chat/completions");
            then.status(200).json_body(serde_json::json!({"choices": []}));
        });
        let llm_url = format!("{}/v1/chat/completions", server.base_url());

        let state = test_state_with_llm(Some(llm_url));
        let app = build_router(state);
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("Content-Type", "application/json")
            .body(Body::from(chat_request_body("diffusion-gemma", false)))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // No real llama-diffusion-daemon binary in this sandbox ⇒
        // `ensure_running` fails to spawn ⇒ structured 503 from the
        // diffusion branch itself.
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(
            json["error"].as_str().unwrap().contains("diffusion"),
            "expected a diffusion-specific error, got: {json}"
        );
        // The decisive assertion: CHORD_LLM_URL was NEVER hit. This is the
        // bug this fix closes — the old code, on a diffusion-branch failure,
        // still couldn't reach CHORD_LLM_URL either (it returned its own
        // 503), but a regression that fell through to the P5/CHORD_LLM_URL
        // path for a diffusion model would silently answer with the WRONG
        // (non-diffusion) model — this guards against ever reintroducing
        // that.
        mock.assert_hits_async(0).await;
    }
}
