//! TIER-05: model-tier control API.
//!
//! A second axum router (bound on `CHORD_CONTROL_PORT`, default 8090) that
//! exposes the model registry and tiering controls so an operator (or the Soma
//! admin dashboard, which lives in a separate repo) can see where every model
//! lives, how much space each uses, when it was last requested, and manually
//! trigger tier changes.
//!
//! ## Endpoints
//! | Method | Path                          | Auth | Purpose |
//! |--------|-------------------------------|------|---------|
//! | GET    | `/api/models`                 | yes  | list all registry records |
//! | GET    | `/api/models/:name`          | yes  | single model detail (404 unknown) |
//! | POST   | `/api/models/:name/archive`  | yes  | archive a warm model (warm → cold) |
//! | POST   | `/api/models/:name/pull`     | yes  | pull a cold model (cold → warm) |
//! | POST   | `/api/models/:name/protect`  | yes  | toggle/set the protected flag |
//! | GET    | `/api/storage`                | yes  | disk usage summary (local + archive) |
//! | POST   | `/api/models/sweep`           | yes  | trigger a disk-pressure eviction sweep |
//!
//! ## Auth choice
//! **All** endpoints — including the GETs — require the same JWT auth as the
//! proxy port (`auth_check(&headers, &state.jwt_secret)`), returning the proxy's
//! identical 401 response on failure. The registry exposes model names, sizes,
//! and storage layout (operationally sensitive), so read endpoints are gated for
//! consistency with the mutating ones rather than left open. When `jwt_secret`
//! is empty, auth is disabled cluster-wide (same behaviour as the proxy), which
//! is what the router-oneshot unit tests rely on.

use std::sync::Arc;

use axum::{
    extract::{Path as AxumPath, Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::models::eviction::{self, run_eviction_sweep, EvictError, FsLocalEvictor};
use crate::models::registry::{ModelRecord, StorageTier};
use crate::models::transfer::DiskSpaceProbe;
use crate::routes::{auth_check, auth_error_response, AppState};

// ── JSON DTOs ────────────────────────────────────────────────────────────────

/// JSON view of a single [`ModelRecord`] returned by the control API. Mirrors the
/// record fields but renders the tier as a lowercase string (`hot`/`warm`/`cold`)
/// so dashboard clients don't depend on serde's enum encoding.
#[derive(Serialize)]
pub struct ModelView {
    pub name: String,
    pub tier: String,
    pub size_bytes: u64,
    pub local_path: Option<String>,
    pub archive_path: Option<String>,
    pub last_requested: Option<i64>,
    pub last_loaded: Option<i64>,
    pub protected: bool,
    /// Lifecycle manager: `"ollama"` (default) or e.g. `"llama-diffusion"` for DiffusionGemma.
    pub managed_by: String,
    /// YARN-06: capability advertisement — whether this model currently
    /// supports thinking (reasoning-trace) mode, derived from
    /// [`crate::serving::profile::RoutingMap::thinking_available`]:
    /// `true` only when the model's serving profile has a `thinking` block
    /// AND `supports_thinking` AND `validated` are all true. An unvalidated
    /// config is NEVER advertised as available (same "inert until validated"
    /// discipline as the launcher's own emission gate) — Harmony's THINK-02
    /// can rely on this field to decide whether sending a per-request
    /// `thinking:on` hint on `/v1/chat/completions` is worth attempting at
    /// all. Computed fresh from the process's in-memory `RoutingMap` on every
    /// request (no separate result cache sitting on top of it) — but that map
    /// itself is loaded ONCE at process startup (see `main.rs`) with no
    /// background refresh, so a model reprofiled/newly-validated after Chord
    /// starts is NOT reflected here until the next restart. See the "Load
    /// cadence" note in `docs/serving.md`.
    pub supports_thinking: bool,
}

fn tier_str(tier: &StorageTier) -> &'static str {
    match tier {
        StorageTier::Hot => "hot",
        StorageTier::Warm => "warm",
        StorageTier::Cold => "cold",
    }
}

impl From<&ModelRecord> for ModelView {
    /// Baseline conversion — `supports_thinking` defaults to `false` here
    /// since a bare [`ModelRecord`] carries no serving-profile/routing
    /// context. Callers with access to the [`crate::serving::profile::RoutingMap`]
    /// (`list_models`/`get_model`) use [`model_view_with_capability`] instead
    /// so the field reflects the model's actual thinking capability.
    fn from(r: &ModelRecord) -> Self {
        ModelView {
            name: r.name.clone(),
            tier: tier_str(&r.tier).to_string(),
            size_bytes: r.size_bytes,
            local_path: r.local_path.clone(),
            archive_path: r.archive_path.clone(),
            last_requested: r.last_requested,
            last_loaded: r.last_loaded,
            protected: r.protected,
            managed_by: r.managed_by.clone(),
            supports_thinking: false,
        }
    }
}

/// YARN-06: build a [`ModelView`] with `supports_thinking` populated from the
/// live [`crate::serving::profile::RoutingMap`] — the capability-advertisement
/// surface Harmony's THINK-02 (built against a stub that always assumes
/// non-supporting) would query over HTTP via `GET /api/models`/`GET
/// /api/models/:name`.
fn model_view_with_capability(
    r: &ModelRecord,
    routing: &crate::serving::profile::RoutingMap,
) -> ModelView {
    let mut view = ModelView::from(r);
    view.supports_thinking = routing.thinking_available(
        &terminus_rs::intake::serving::ModelId::from_registry_key(r.name.clone()),
    );
    view
}

/// Disk usage for one filesystem. `null` fields mean the probe couldn't read the
/// path (e.g. an unmounted archive) — the API reports that rather than erroring.
#[derive(Serialize)]
pub struct DiskUsage {
    /// Whether the path is present/mounted.
    pub available: bool,
    pub total_bytes: Option<u64>,
    pub free_bytes: Option<u64>,
    pub used_bytes: Option<u64>,
}

#[derive(Serialize)]
pub struct StorageView {
    pub local: DiskUsage,
    pub archive: DiskUsage,
}

fn error_response(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(serde_json::json!({ "error": message.into() }))).into_response()
}

// ── GET /api/models ──────────────────────────────────────────────────────────

/// List every registry record (sorted by name for a stable dashboard view).
pub async fn list_models(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }
    let reg = state.model_registry.lock().await;
    let routing = state.routing_map.lock().await;
    let mut models: Vec<ModelView> = reg
        .all_records()
        .map(|r| model_view_with_capability(r, &routing))
        .collect();
    models.sort_by(|a, b| a.name.cmp(&b.name));
    let count = models.len();
    Json(serde_json::json!({ "models": models, "count": count })).into_response()
}

// ── GET /api/models/:name ───────────────────────────────────────────────────

/// Single model detail; 404 when the registry doesn't know the name.
pub async fn get_model(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    AxumPath(name): AxumPath<String>,
) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }
    let reg = state.model_registry.lock().await;
    let routing = state.routing_map.lock().await;
    match reg.get(&name) {
        Some(rec) => Json(model_view_with_capability(rec, &routing)).into_response(),
        None => error_response(StatusCode::NOT_FOUND, format!("unknown model: {name}")),
    }
}

// ── POST /api/models/:name/archive ──────────────────────────────────────────

/// Archive a warm model to the cold tier via [`eviction::evict_to_archive`].
///
/// Edge cases (per spec):
/// - Hot (loaded in VRAM) → 409 "model is currently loaded, unload first".
/// - protected → 403 with an explanation.
/// - unknown → 404.
pub async fn archive_model(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    AxumPath(name): AxumPath<String>,
) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }

    // Pre-flight under a short lock so we can return precise status codes the
    // generic evict error mapping can't (Hot → 409, unknown → 404).
    {
        let reg = state.model_registry.lock().await;
        match reg.get(&name) {
            None => {
                return error_response(StatusCode::NOT_FOUND, format!("unknown model: {name}"));
            }
            Some(rec) => {
                if rec.tier == StorageTier::Hot {
                    return error_response(
                        StatusCode::CONFLICT,
                        "model is currently loaded, unload first",
                    );
                }
            }
        }
        if reg.is_protected(&name) {
            return error_response(
                StatusCode::FORBIDDEN,
                format!("model {name} is protected and cannot be archived; unprotect it first"),
            );
        }
    }

    match eviction::evict_to_archive(&state.model_registry, &name, state.local_evictor.as_ref())
        .await
    {
        Ok(ev) => Json(serde_json::json!({
            "status": "archived",
            "model": name,
            "freed_bytes": ev.freed_bytes,
        }))
        .into_response(),
        Err(EvictError::UnknownModel(_)) => {
            error_response(StatusCode::NOT_FOUND, format!("unknown model: {name}"))
        }
        Err(EvictError::Protected(_)) => error_response(
            StatusCode::FORBIDDEN,
            format!("model {name} is protected and cannot be archived; unprotect it first"),
        ),
        Err(EvictError::NotWarm(_)) => error_response(
            StatusCode::CONFLICT,
            format!("model {name} is not warm; only warm models can be archived"),
        ),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

// ── POST /api/models/:name/pull ─────────────────────────────────────────────

/// Pull a cold model to the warm tier via `pull_coordinator.ensure_local`. The
/// TIER-03 pre-pull eviction (if wired) runs inside `ensure_local`; an
/// insufficient-space failure surfaces here as a 507.
pub async fn pull_model(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    AxumPath(name): AxumPath<String>,
) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }

    // Surface a precise 404 for unknown models (ensure_local also returns
    // UnknownModel, but checking first avoids touching the pull machinery).
    {
        let reg = state.model_registry.lock().await;
        if reg.get(&name).is_none() {
            return error_response(StatusCode::NOT_FOUND, format!("unknown model: {name}"));
        }
    }

    match state.pull_coordinator.ensure_local(&name, None).await {
        Ok(()) => Json(serde_json::json!({ "status": "warm", "model": name })).into_response(),
        Err(crate::models::transfer::PullError::UnknownModel(_)) => {
            error_response(StatusCode::NOT_FOUND, format!("unknown model: {name}"))
        }
        Err(crate::models::transfer::PullError::MissingArchive(_)) => error_response(
            StatusCode::NOT_FOUND,
            format!("model {name} is not present in the archive"),
        ),
        Err(e @ crate::models::transfer::PullError::InsufficientDiskSpace { .. }) => {
            error_response(StatusCode::INSUFFICIENT_STORAGE, e.to_string())
        }
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

// ── POST /api/models/:name/protect ──────────────────────────────────────────

/// Optional desired protected state. May arrive as a query param
/// (`?protected=true`) or a JSON body (`{"protected": false}`). When absent, the
/// current flag is **toggled**.
#[derive(Deserialize, Default)]
pub struct ProtectQuery {
    pub protected: Option<bool>,
}

#[derive(Deserialize, Default)]
pub struct ProtectBody {
    pub protected: Option<bool>,
}

/// Toggle or set a model's `protected` flag.
///
/// Contract: the desired state is taken from `?protected=<bool>` first, then the
/// JSON body `{"protected": <bool>}`; if neither is present the current flag is
/// inverted. Persisted via `registry.save()` (best-effort; a save error is
/// logged, the in-memory change still applies). 404 for unknown models.
///
/// Note: a model whose name is in the configured `MODEL_PROTECTED` set stays
/// protected regardless of this flag — the response's `protected` reflects the
/// authoritative `is_protected()` so a no-op clear is visible to the caller.
pub async fn protect_model(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    AxumPath(name): AxumPath<String>,
    Query(q): Query<ProtectQuery>,
    body: Option<Json<ProtectBody>>,
) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }

    let desired = q
        .protected
        .or_else(|| body.and_then(|b| b.0.protected));

    let mut reg = state.model_registry.lock().await;
    let current = match reg.get(&name) {
        Some(rec) => rec.protected,
        None => {
            return error_response(StatusCode::NOT_FOUND, format!("unknown model: {name}"));
        }
    };
    let target = desired.unwrap_or(!current);
    reg.set_protected(&name, target);
    if let Err(e) = reg.save() {
        warn!("protect_model: failed to persist registry for {name}: {e}");
    }
    // Report the authoritative protection state (config list may force-protect).
    let effective = reg.is_protected(&name);
    Json(serde_json::json!({
        "model": name,
        "protected": effective,
        "flag": target,
    }))
    .into_response()
}

// ── GET /api/storage ─────────────────────────────────────────────────────────

fn disk_usage(probe: &dyn DiskSpaceProbe, path: &std::path::Path) -> DiskUsage {
    // Probe the nearest existing ancestor so a not-yet-created leaf still reports
    // its filesystem's usage; a wholly-unmounted archive yields nulls.
    let target = crate::models::transfer::nearest_existing_ancestor(path);
    let available = path.exists();
    let total = probe.total_bytes(&target);
    let free = probe.available_bytes(&target);
    let used = match (total, free) {
        (Some(t), Some(f)) => Some(t.saturating_sub(f)),
        _ => None,
    };
    DiskUsage {
        available,
        total_bytes: total,
        free_bytes: free,
        used_bytes: used,
    }
}

/// Disk usage summary for the local and archive roots. An unmounted/unavailable
/// archive reports `available: false` with null byte counts rather than erroring.
pub async fn storage_summary(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }
    let (local_path, archive_path) = {
        let reg = state.model_registry.lock().await;
        (
            reg.local_path().to_path_buf(),
            reg.archive_path().to_path_buf(),
        )
    };
    let probe = state.disk_probe.as_ref();
    let view = StorageView {
        local: disk_usage(probe, &local_path),
        archive: disk_usage(probe, &archive_path),
    };
    Json(view).into_response()
}

// ── POST /api/models/sweep ───────────────────────────────────────────────────

/// Manually trigger a disk-pressure eviction sweep. The sweep is spawned (it may
/// archive several models and is long-running) and the call returns 202 Accepted
/// immediately. The sweep itself no-ops when disk usage is below threshold or the
/// archive isn't mounted (see [`run_eviction_sweep`]).
pub async fn trigger_sweep(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }
    let registry = state.model_registry.clone();
    let evictor = state.local_evictor.clone();
    let lock = state.disk_op_lock.clone();
    let probe = state.disk_probe.clone();
    let threshold = state.disk_pressure_percent;
    let cooldown_hours = state.model_warm_cooldown_hours;
    info!("control API: manual eviction sweep triggered");
    tokio::spawn(async move {
        run_eviction_sweep(&registry, threshold, cooldown_hours, probe.as_ref(), evictor.as_ref(), &lock).await;
    });
    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({ "status": "sweep started" })),
    )
        .into_response()
}

// ── GET /health (control port) ───────────────────────────────────────────────

/// Health/version handler for the control port. No auth — version metadata only.
pub async fn control_health() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "service": "chord-proxy-control",
        "version": crate::version::version(),
        "commit": crate::version::commit(),
        "terminus_rs": terminus_rs::VERSION,
    }))
}

// ── Router ───────────────────────────────────────────────────────────────────

/// Build the TIER-05 control router over the shared [`AppState`].
pub fn build_control_router(state: Arc<AppState>) -> axum::Router {
    use axum::routing::{get, post};
    axum::Router::new()
        .route("/health", get(control_health))
        .route("/api/models", get(list_models))
        .route("/api/models/sweep", post(trigger_sweep))
        .route("/api/models/:name", get(get_model))
        .route("/api/models/:name/archive", post(archive_model))
        .route("/api/models/:name/pull", post(pull_model))
        .route("/api/models/:name/protect", post(protect_model))
        .route("/api/storage", get(storage_summary))
        // SNAP observability routes (additive; distinct paths, same JWT auth):
        // /api/vram, /api/activity, /api/inventory, /api/analytics/*.
        .merge(crate::snap::api::snap_routes())
        .with_state(state)
}

// Suppress unused import when FsLocalEvictor is only referenced by main.rs/tests.
#[allow(unused_imports)]
use FsLocalEvictor as _ControlFsLocalEvictor;

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Method, Request};
    use serde_json::Value;
    use std::path::Path;
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;

    use crate::models::eviction::{new_disk_op_lock, FsLocalEvictor, LocalEvictor};
    use crate::models::registry::{ModelRegistry, StorageTier};
    use crate::models::transfer::{DiskSpaceProbe, PullCoordinator, StatvfsProbe};

    /// Build a control-router AppState over the given registry, with a real
    /// FsLocalEvictor rooted at `local_root` and an injected disk probe. Auth is
    /// disabled (empty jwt_secret) so the router-oneshot tests don't need a token.
    fn control_state(
        registry: Arc<Mutex<ModelRegistry>>,
        local_root: std::path::PathBuf,
        probe: Arc<dyn DiskSpaceProbe>,
    ) -> Arc<AppState> {
        control_state_with_routing(
            registry,
            local_root,
            probe,
            Arc::new(Mutex::new(crate::serving::profile::RoutingMap::empty())),
        )
    }

    /// Like `control_state` but with an explicit
    /// [`crate::serving::profile::RoutingMap`] so YARN-06's `supports_thinking`
    /// capability field can be exercised.
    fn control_state_with_routing(
        registry: Arc<Mutex<ModelRegistry>>,
        local_root: std::path::PathBuf,
        probe: Arc<dyn DiskSpaceProbe>,
        routing_map: Arc<Mutex<crate::serving::profile::RoutingMap>>,
    ) -> Arc<AppState> {
        use crate::agentic::AgenticExecutor;
        use crate::audit::AuditLogger;
        use crate::config::{Config, RateLimitConfig};
        use crate::mcp_proxy::{FallbackRegistry, McpProxy};
        use crate::rate_limiter::ProxyRateLimiter;

        let config = Config {
            mcp_backend_url: "http://does-not-exist:9999".into(),
            jwt_secret: String::new(),
            tool_timeout_secs: 5,
            catalog_cache_secs: 300,
            listen_port: 9099,
            control_port: 8090,
            rate_limits: RateLimitConfig::default(),
            llm_backend_url: None,
            model_aliases: std::collections::HashMap::new(),
            model_archive_path: "/archive".into(),
            model_local_path: "/local".into(),
            model_protected: vec![],
            model_pull_timeout_secs: 600,
            model_registry_path: "/registry.json".into(),
            model_disk_pressure_percent: 80,
            model_sweep_interval_secs: 1800,
            model_warm_cooldown_hours: 168,
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
        let rate_limiter = Arc::new(Mutex::new(ProxyRateLimiter::new(RateLimitConfig::default())));
        let audit_logger = Arc::new(AuditLogger::new(std::path::PathBuf::from("/dev/null")));
        let pull_coordinator = Arc::new(PullCoordinator::new(
            registry.clone(),
            std::time::Duration::from_secs(30),
        ));
        let local_evictor: Arc<dyn LocalEvictor> =
            Arc::new(FsLocalEvictor::new(local_root));
        Arc::new(AppState {
            proxy,
            jwt_secret: String::new(),
            audit_logger,
            rate_limiter,
            agentic_executor,
            llm_backend_url: None,
            model_aliases: std::collections::HashMap::new(),
            http_client: reqwest::Client::new(),
            model_registry: registry,
            pull_coordinator,
            local_evictor,
            disk_op_lock: new_disk_op_lock(),
            disk_probe: probe,
            disk_pressure_percent: 80,
            model_warm_cooldown_hours: 168,
            routing_map,
            coding_profile_source: Arc::new(Mutex::new(None)),
            personal_proxy: None,
        })
    }

    /// Write a manifest + referenced blobs under `root`, returning the model name.
    fn make_model(root: &Path, model: &str, tag: &str, blob_sizes: &[u64]) -> String {
        use std::fs;
        let manifests = root
            .join("manifests")
            .join("registry.ollama.ai")
            .join("library")
            .join(model);
        fs::create_dir_all(&manifests).unwrap();
        let blobs_dir = root.join("blobs");
        fs::create_dir_all(&blobs_dir).unwrap();
        let mut layers = Vec::new();
        for (i, size) in blob_sizes.iter().enumerate() {
            let digest = format!("sha256:{model}{i}");
            fs::write(blobs_dir.join(digest.replacen(':', "-", 1)), vec![b'x'; *size as usize])
                .unwrap();
            layers.push(serde_json::json!({ "size": size, "digest": digest }));
        }
        let cfg = format!("sha256:{model}cfg");
        fs::write(blobs_dir.join(cfg.replacen(':', "-", 1)), b"cfg").unwrap();
        let body = serde_json::json!({
            "config": { "size": 3, "digest": cfg },
            "layers": layers,
        });
        fs::write(manifests.join(tag), serde_json::to_string(&body).unwrap()).unwrap();
        format!("{model}:{tag}")
    }

    fn reg_at(base: &Path, protected: Vec<String>) -> ModelRegistry {
        ModelRegistry::new(
            base.join("registry.json"),
            base.join("local"),
            base.join("archive"),
            protected,
        )
    }

    #[tokio::test]
    async fn get_models_returns_records() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        make_model(&base.join("local"), "alpha", "1", &[100]);
        make_model(&base.join("local"), "beta", "1", &[200]);
        let mut reg = reg_at(base, vec![]);
        reg.reconcile();
        let registry = Arc::new(Mutex::new(reg));
        let state = control_state(registry, base.join("local"), Arc::new(StatvfsProbe));
        let app = build_control_router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["count"], 2);
        let names: Vec<&str> = json["models"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, vec!["alpha:1", "beta:1"]);
        assert_eq!(json["models"][0]["tier"], "warm");
    }

    /// YARN-06: build a one-row [`crate::serving::profile::RoutingMap`] for
    /// `model_name` with the given `thinking` env_json fragment (or `"{}"` for
    /// no thinking block at all).
    fn routing_map_with(
        model_name: &str,
        thinking_env_json: &str,
    ) -> Arc<Mutex<crate::serving::profile::RoutingMap>> {
        use terminus_rs::intake::serving::{
            ExclusionReason, ModelId, RecheckTrigger, Runtime, ServingBackend, ServingProfile,
        };
        let row = ServingProfile {
            model_id: ModelId::from(model_name),
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

    #[tokio::test]
    async fn list_models_reports_supports_thinking_only_for_validated_model() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        make_model(&base.join("local"), "alpha", "1", &[100]);
        make_model(&base.join("local"), "beta", "1", &[200]);
        let mut reg = reg_at(base, vec![]);
        reg.reconcile();
        let registry = Arc::new(Mutex::new(reg));
        // Only "alpha:1" has a supporting + validated thinking config.
        let routing = routing_map_with(
            "alpha:1",
            r#"{"thinking":{"supports_thinking":true,"validated":true}}"#,
        );
        let state = control_state_with_routing(
            registry,
            base.join("local"),
            Arc::new(StatvfsProbe),
            routing,
        );
        let app = build_control_router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        let models = json["models"].as_array().unwrap();
        let alpha = models.iter().find(|m| m["name"] == "alpha:1").unwrap();
        let beta = models.iter().find(|m| m["name"] == "beta:1").unwrap();
        assert_eq!(alpha["supports_thinking"], true);
        // Negative: "beta:1" has no serving profile at all ⇒ not advertised.
        assert_eq!(beta["supports_thinking"], false);
    }

    #[tokio::test]
    async fn get_model_reports_false_for_unvalidated_thinking_config() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        make_model(&base.join("local"), "alpha", "1", &[100]);
        let mut reg = reg_at(base, vec![]);
        reg.reconcile();
        let registry = Arc::new(Mutex::new(reg));
        // Negative test: supports_thinking=true but validated=false ⇒ never advertised.
        let routing = routing_map_with(
            "alpha:1",
            r#"{"thinking":{"supports_thinking":true,"validated":false}}"#,
        );
        let state = control_state_with_routing(
            registry,
            base.join("local"),
            Arc::new(StatvfsProbe),
            routing,
        );
        let app = build_control_router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/models/alpha:1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["supports_thinking"], false);
    }

    #[tokio::test]
    async fn get_unknown_model_returns_404() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        let reg = reg_at(base, vec![]);
        let registry = Arc::new(Mutex::new(reg));
        let state = control_state(registry, base.join("local"), Arc::new(StatvfsProbe));
        let app = build_control_router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/models/nope:1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn archive_protected_model_returns_403() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        let model = make_model(&base.join("local"), "keepme", "1", &[100]);
        let mut reg = reg_at(base, vec![model.clone()]);
        reg.reconcile();
        assert!(reg.is_protected(&model));
        let registry = Arc::new(Mutex::new(reg));
        let state = control_state(registry, base.join("local"), Arc::new(StatvfsProbe));
        let app = build_control_router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(&format!("/api/models/{model}/archive"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(json["error"].as_str().unwrap().contains("protected"));
    }

    #[tokio::test]
    async fn archive_warm_model_triggers_eviction_to_cold() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        std::fs::create_dir_all(base.join("archive")).unwrap();
        let model = make_model(&base.join("local"), "warm", "1", &[100, 200]);
        let mut reg = reg_at(base, vec![]);
        reg.reconcile();
        assert_eq!(reg.get(&model).unwrap().tier, StorageTier::Warm);
        let registry = Arc::new(Mutex::new(reg));
        let state = control_state(registry.clone(), base.join("local"), Arc::new(StatvfsProbe));
        let app = build_control_router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(&format!("/api/models/{model}/archive"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        // Model is now cold; archive holds the copy, local is gone.
        assert_eq!(
            registry.lock().await.get(&model).unwrap().tier,
            StorageTier::Cold
        );
        assert!(base
            .join("archive/manifests/registry.ollama.ai/library/warm/1")
            .is_file());
        assert!(!base
            .join("local/manifests/registry.ollama.ai/library/warm/1")
            .is_file());
    }

    #[tokio::test]
    async fn archive_hot_model_returns_409() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        let model = make_model(&base.join("local"), "hotmodel", "1", &[100]);
        let mut reg = reg_at(base, vec![]);
        reg.reconcile();
        reg.set_tier(&model, StorageTier::Hot);
        let registry = Arc::new(Mutex::new(reg));
        let state = control_state(registry, base.join("local"), Arc::new(StatvfsProbe));
        let app = build_control_router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(&format!("/api/models/{model}/archive"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(json["error"].as_str().unwrap().contains("loaded"));
    }

    #[tokio::test]
    async fn protect_toggles_flag() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        let model = make_model(&base.join("local"), "togg", "1", &[100]);
        let mut reg = reg_at(base, vec![]);
        reg.reconcile();
        assert!(!reg.get(&model).unwrap().protected);
        let registry = Arc::new(Mutex::new(reg));
        let state = control_state(registry.clone(), base.join("local"), Arc::new(StatvfsProbe));
        let app = build_control_router(state);

        // No body/query → toggle to true.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(&format!("/api/models/{model}/protect"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["protected"], true);

        // Explicit ?protected=false → clears it.
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(&format!("/api/models/{model}/protect?protected=false"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(!registry.lock().await.get(&model).unwrap().protected);
    }

    #[tokio::test]
    async fn storage_returns_usage_json() {
        // Injected probe so the result is deterministic regardless of the host FS.
        struct FixedProbe;
        impl DiskSpaceProbe for FixedProbe {
            fn available_bytes(&self, _: &Path) -> Option<u64> {
                Some(40)
            }
            fn total_bytes(&self, _: &Path) -> Option<u64> {
                Some(100)
            }
        }
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        std::fs::create_dir_all(base.join("local")).unwrap();
        std::fs::create_dir_all(base.join("archive")).unwrap();
        let reg = reg_at(base, vec![]);
        let registry = Arc::new(Mutex::new(reg));
        let state = control_state(registry, base.join("local"), Arc::new(FixedProbe));
        let app = build_control_router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/storage")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["local"]["total_bytes"], 100);
        assert_eq!(json["local"]["free_bytes"], 40);
        assert_eq!(json["local"]["used_bytes"], 60);
    }

    #[tokio::test]
    async fn sweep_returns_202() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        let reg = reg_at(base, vec![]);
        let registry = Arc::new(Mutex::new(reg));
        let state = control_state(registry, base.join("local"), Arc::new(StatvfsProbe));
        let app = build_control_router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/models/sweep")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::ACCEPTED);
    }
}
