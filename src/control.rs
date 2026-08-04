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
//! | GET    | `/metrics`                    | no   | PROMEX-02: Prometheus text-exposition application metrics |
//! | GET    | `/api/models`                 | yes  | list all registry records |
//! | GET    | `/api/models/:name`          | yes  | single model detail (404 unknown) |
//! | POST   | `/api/models/:name/archive`  | yes  | archive a warm model (warm → cold) |
//! | POST   | `/api/models/:name/pull`     | yes  | pull a cold model (cold → warm) |
//! | POST   | `/api/models/ingest`          | yes  | ASK4-P2A: pull a HuggingFace model into cold storage (default-OFF gate) |
//! | POST   | `/api/models/:name/protect`  | yes  | toggle/set the protected flag |
//! | GET    | `/api/routes`                 | yes  | CHRD-100: the logical ROUTE catalog (named routes, locality, availability) |
//! | GET    | `/api/routes/:id`             | yes  | CHRD-100: one route (404 = no such route, which is not "unavailable") |
//! | GET    | `/api/storage`                | yes  | disk usage summary (local + archive) |
//! | POST   | `/api/models/sweep`           | yes  | trigger a disk-pressure eviction sweep |
//! | POST   | `/api/models/reconcile`       | yes  | MSM-04: reconcile + persist the registry, return before/after tier counts |
//! | POST   | `/api/storage/gc`             | yes  | MSM-04: run the MSM-03 orphan-blob GC pass, return freed bytes |
//! | POST   | `/api/sweep/session`          | yes  | RESIL-02: register/upsert a sweep's action queue (idempotent) |
//! | GET    | `/api/sweep/session/:id`      | yes  | RESIL-02: remaining keys (queue order) + counts (404 unknown) |
//! | POST   | `/api/sweep/session/:id/advance` | yes | RESIL-02: mark keys done (append-only, idempotent) |
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
use crate::models::gc;
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

    let copy_timeout = std::time::Duration::from_secs(state.model_archive_copy_timeout_secs);
    match eviction::evict_to_archive(
        &state.model_registry,
        &name,
        state.local_evictor.as_ref(),
        copy_timeout,
    )
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
        Err(e @ EvictError::Timeout(..)) => {
            error_response(StatusCode::GATEWAY_TIMEOUT, e.to_string())
        }
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

// ── GET /admin/resident-set ──────────────────────────────────────────────────

/// TRTR-07: report the assistant-mode resident set so residency is OBSERVABLE
/// rather than inferred.
///
/// Per role (`personality` / `router` / `embedding`, reported in that degradation
/// priority order): the alias key it resolves through, the model that alias
/// currently points at, its state (`warm` / `released` / `unresolved` / `missing`
/// / `warm-failed` / `dropped-vram` / `disabled`), `warm`, `last_used`, and the
/// registry size. Plus the set-level `active` flag (false between a mode-swap
/// release and the next re-warm) and the residency exemption in force.
///
/// Read-only and auth-gated with the same posture as `/admin/idle`.
pub async fn resident_set_status(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }
    let status = crate::routing::resident_set::global().status().await;
    // Cross-check the set's own view against the registry's live exemption so a
    // drift between the two is visible instead of silently believed.
    let registry_exempt = state.model_registry.lock().await.residency_exempt();
    let mut body = serde_json::to_value(&status).unwrap_or_else(|_| serde_json::json!({}));
    if let Some(obj) = body.as_object_mut() {
        obj.insert(
            "registry_exempt".to_string(),
            serde_json::json!(registry_exempt),
        );
    }
    Json(body).into_response()
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
    let copy_timeout = std::time::Duration::from_secs(state.model_archive_copy_timeout_secs);
    info!("control API: manual eviction sweep triggered");
    tokio::spawn(async move {
        run_eviction_sweep(
            &registry,
            threshold,
            cooldown_hours,
            probe.as_ref(),
            evictor.as_ref(),
            &lock,
            copy_timeout,
        )
        .await;
    });
    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({ "status": "sweep started" })),
    )
        .into_response()
}

// ── POST /api/models/reconcile ───────────────────────────────────────────────

/// MSM-04: reconcile the registry against on-disk reality and persist it,
/// returning the tier counts before and after. Synchronous from the caller's
/// point of view (returns the result, not a 202), but — critically, S111 — it
/// runs the SLOW manifest scan (local + NFS archive) OFF the registry lock via
/// `spawn_blocking`, then takes the lock only for the fast in-memory apply +
/// persist. Holding the registry `Mutex` across the ~84s NFS scan would freeze
/// every concurrent `chat_completions`/`update_last_requested`, wedging all
/// inference (the regression this fixes). Takes only the registry lock, never
/// `disk_op_lock`, so it can't invert the canonical disk_op → registry order.
pub async fn trigger_reconcile(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }

    // Snapshot paths + before-counts under a BRIEF lock, then release it.
    let (local_path, archive_path, hot_before, warm_before, cold_before) = {
        let reg = state.model_registry.lock().await;
        let (h, w, c) = reg.tier_counts();
        (
            reg.local_path().to_path_buf(),
            reg.archive_path().to_path_buf(),
            h,
            w,
            c,
        )
    };

    // SLOW manifest scan OFF the lock.
    let scan = match tokio::task::spawn_blocking(move || {
        crate::models::registry::ModelRegistry::scan_disk(local_path, archive_path)
    })
    .await
    {
        Ok(scan) => scan,
        Err(e) => {
            warn!("control API: reconcile scan task failed: {e}");
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("reconcile scan failed: {e}"),
            );
        }
    };

    // Apply + persist under a BRIEF lock (in-memory, milliseconds).
    let mut reg = state.model_registry.lock().await;
    reg.apply_reconcile(scan);
    let (hot_after, warm_after, cold_after) = reg.tier_counts();
    let persisted = match reg.save() {
        Ok(()) => true,
        Err(e) => {
            warn!("control API: failed to persist registry after reconcile: {e}");
            false
        }
    };
    info!(
        hot_before, warm_before, cold_before, hot_after, warm_after, cold_after, persisted,
        "control API: manual reconcile complete"
    );
    Json(serde_json::json!({
        "status": "reconciled",
        "persisted": persisted,
        "before": { "hot": hot_before, "warm": warm_before, "cold": cold_before },
        "after": { "hot": hot_after, "warm": warm_after, "cold": cold_after },
    }))
    .into_response()
}

// ── POST /api/storage/gc ─────────────────────────────────────────────────────

/// MSM-04: run the MSM-03 orphan-blob GC pass synchronously and return freed
/// bytes. Bounded and non-destructive-by-default (only ever removes blobs
/// confirmed orphaned — see `models::gc`), so unlike the sweep trigger this
/// returns the result directly rather than 202-and-forget.
pub async fn trigger_gc(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
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
    let result = gc::run_gc(
        &state.model_registry,
        &local_path,
        &archive_path,
        &state.disk_op_lock,
        state.model_gc_min_age_secs,
    )
    .await;
    info!(
        orphans_deleted = result.orphans_deleted,
        freed_bytes = result.freed_bytes,
        errors = result.errors.len(),
        "control API: manual GC pass complete"
    );
    Json(serde_json::json!({
        "status": "gc complete",
        "orphans_deleted": result.orphans_deleted,
        "freed_bytes": result.freed_bytes,
        "errors": result.errors,
    }))
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

/// PROMEX-02: `GET /metrics` — encodes the process-global `crate::metrics`
/// registry (inference request counts + latency histogram) in the standard
/// Prometheus text exposition format. Takes no `State` — the registry is
/// process-global, not per-server-instance. Unauthenticated, same as
/// `/health` above (see this module's doc, "Auth choice" section — JWT auth
/// is checked INSIDE the `/api/*`/`/admin/*` handlers, not a router-wide
/// layer, and metrics are equally non-sensitive: bounded model names, counts,
/// and timings only).
pub async fn handle_metrics() -> impl IntoResponse {
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4")],
        crate::metrics::gather_text(),
    )
}

// ── Sweep session cache (RESIL-02) ───────────────────────────────────────────
//
// A durable, session-keyed cache of a sweep's planned action queue + progress
// cursor, so a restarted sweep can resume from Chord. Chord only RECORDS and
// SERVES the queue; the Terminus sweep executes it. Backed by the process-global
// [`crate::sweep_session::SWEEP_SESSIONS`] store (host-singleton, like the GPU
// lock), persisted under `CHORD_STATE_DIR` when configured. Same JWT auth as the
// other control routes.

/// Body for `POST /api/sweep/session` — register/upsert a queue.
#[derive(Deserialize)]
pub struct SweepSessionRegisterBody {
    pub session_id: String,
    #[serde(default)]
    pub queue: Vec<String>,
}

/// Body for `POST /api/sweep/session/:id/advance` — mark keys done.
#[derive(Deserialize)]
pub struct SweepSessionAdvanceBody {
    #[serde(default)]
    pub keys: Vec<String>,
}

/// `POST /api/sweep/session` — idempotent register/upsert of a sweep's action
/// queue. Same queue ⇒ no-op (preserves progress); different queue ⇒ replaces it
/// and resets progress (a replanned sweep).
pub async fn sweep_session_register(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<SweepSessionRegisterBody>,
) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }
    if body.session_id.trim().is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "session_id is required");
    }
    let now = crate::gpu_exclusive::now_epoch();
    let summary =
        crate::sweep_session::SWEEP_SESSIONS.register(&body.session_id, body.queue, now);
    Json(summary).into_response()
}

/// `GET /api/sweep/session/:id` — the remaining keys (in queue order) + counts.
/// 404 for an unknown session.
pub async fn sweep_session_get(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }
    match crate::sweep_session::SWEEP_SESSIONS.get(&id) {
        Some(summary) => Json(summary).into_response(),
        None => error_response(StatusCode::NOT_FOUND, format!("unknown sweep session: {id}")),
    }
}

/// `POST /api/sweep/session/:id/advance` — mark `keys` done (append-only,
/// idempotent; keys not in the queue are ignored). 404 for an unknown session.
pub async fn sweep_session_advance(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
    Json(body): Json<SweepSessionAdvanceBody>,
) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }
    match crate::sweep_session::SWEEP_SESSIONS.advance(&id, &body.keys) {
        Some(summary) => Json(summary).into_response(),
        None => error_response(StatusCode::NOT_FOUND, format!("unknown sweep session: {id}")),
    }
}

// ── POST /api/models/ingest (ASK4-P2A) ───────────────────────────────────────

/// Request body for `POST /api/models/ingest`.
///
/// Re-exported shape of [`crate::models::ingest::IngestRequest`] — declared here
/// so axum's `Json<...>` extractor rejects a malformed body with a 422 before
/// the handler runs, consistent with the other control DTOs.
pub use crate::models::ingest::IngestRequest;

/// The real [`crate::models::ingest::IngestOps`] over the live [`AppState`]:
/// Ollama for the HF download, the model registry + eviction machinery for the
/// warm→cold demote. Constructed per-request (cheap; holds only references).
struct AppStateIngestOps<'a> {
    state: &'a AppState,
    ollama_url: Option<String>,
    /// CHORD_MODEL_INGEST_PULL_TIMEOUT_SECS — bounds the (potentially long) HF
    /// download step. Distinct from `archive_copy_timeout`.
    pull_timeout: std::time::Duration,
    /// MODEL_ARCHIVE_COPY_TIMEOUT_SECS — bounds the warm→cold copy.
    archive_copy_timeout: std::time::Duration,
}

impl<'a> AppStateIngestOps<'a> {
    /// Map a registry [`ModelRecord`] tier to the module-local
    /// [`crate::models::ingest::ModelTier`].
    fn tier_of(rec: &ModelRecord) -> crate::models::ingest::ModelTier {
        match rec.tier {
            StorageTier::Cold => crate::models::ingest::ModelTier::Cold,
            StorageTier::Warm => crate::models::ingest::ModelTier::Warm,
            StorageTier::Hot => crate::models::ingest::ModelTier::Hot,
        }
    }
}

#[async_trait::async_trait]
impl<'a> crate::models::ingest::IngestOps for AppStateIngestOps<'a> {
    async fn find_existing(
        &self,
        candidates: &[String],
    ) -> Option<crate::models::ingest::ExistingModel> {
        // Candidates are the host-stripped registry keys (`org/model[:tag]`),
        // exactly what `reconcile()` records — never a `hf.co/...` literal.
        let reg = self.state.model_registry.lock().await;
        for candidate in candidates {
            if let Some(rec) = reg.get(candidate) {
                return Some(crate::models::ingest::ExistingModel {
                    name: rec.name.clone(),
                    tier: Self::tier_of(rec),
                });
            }
        }
        None
    }

    async fn probe(
        &self,
        hf_repo: &str,
        revision: Option<&str>,
        token: Option<&str>,
    ) -> Result<crate::models::ingest::HfProbe, String> {
        // HuggingFace Hub model-info API. A gated repo returns 401/403 without
        // an accepted token; a public one returns JSON with a `gated` field and
        // (with ?blobs=true) per-sibling sizes. `revision` here is an Ollama
        // quant selector, NOT an HF git ref — so it is used only to FILTER the
        // sibling sizes, never placed in the URL path.
        let url = format!("https://huggingface.co/api/models/{hf_repo}?blobs=true");
        let mut rb = self
            .state
            .http_client
            .get(&url)
            .header("User-Agent", "chord-model-ingest");
        if let Some(t) = token {
            rb = rb.bearer_auth(t); // never logged
        }
        let resp = rb.send().await.map_err(|e| format!("request error: {e}"))?;
        let status = resp.status();
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            // Access-restricted and unreadable (gated without an accepted token,
            // or private/absent). Report gated so it's refused up front — the
            // credential-free Ollama pull couldn't authenticate it anyway.
            return Ok(crate::models::ingest::HfProbe {
                gated: true,
                private: false,
                total_bytes: None,
            });
        }
        if !status.is_success() {
            return Err(format!("HTTP {status}"));
        }
        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| format!("response not JSON: {e}"))?;
        // Both `gated` (bool | "auto"/"manual") and `private` (bool) mark a repo
        // the credential-free Ollama pull cannot fetch → refused by the caller.
        let flag = |key: &str| match body.get(key) {
            Some(serde_json::Value::Bool(b)) => *b,
            Some(serde_json::Value::String(s)) => !s.eq_ignore_ascii_case("false"),
            _ => false,
        };
        let total_bytes = hf_repo_size_bytes(&body, revision);
        Ok(crate::models::ingest::HfProbe {
            gated: flag("gated"),
            private: flag("private"),
            total_bytes,
        })
    }

    async fn pull_warm(&self, ollama_ref: &str) -> Result<(), String> {
        // Delegate the actual download to Ollama's native HF ingestion (public
        // repos only — gated repos are refused before this step). The download
        // writes blobs into the SAME local model store that the sweep / archive
        // pull / orphan-GC manage, so it MUST hold `disk_op_lock` for its whole
        // duration — otherwise a concurrent GC/sweep could reap the in-progress
        // (or sharded/long) download's blobs before its manifest lands. Same
        // discipline as `PullCoordinator::ensure_local` (holds the lock across
        // the cold→warm copy). Acquired FIRST, no registry lock held, so the
        // canonical disk_op→registry order is preserved; `land_cold` then takes
        // its own guard afterwards (nothing is held between the two).
        let _guard = self.state.disk_op_lock.lock().await;

        let base = self
            .ollama_url
            .as_deref()
            .ok_or_else(|| "OLLAMA_URL is not configured".to_string())?;
        let url = format!("{}/api/pull", base.trim_end_matches('/'));
        let fut = self
            .state
            .http_client
            .post(&url)
            .json(&serde_json::json!({ "name": ollama_ref, "stream": false }))
            .send();
        let resp = tokio::time::timeout(self.pull_timeout, fut)
            .await
            .map_err(|_| "ollama pull timed out".to_string())?
            .map_err(|e| format!("ollama pull request error: {e}"))?;
        if !resp.status().is_success() {
            return Err(format!("ollama pull returned HTTP {}", resp.status()));
        }
        Ok(())
    }

    async fn land_cold(
        &self,
        candidates: &[String],
    ) -> Result<crate::models::ingest::ColdLanded, String> {
        // Serialise with the sweep / archive / GC via the shared disk-op lock,
        // acquired FIRST (canonical disk_op → registry order), and never hold
        // the registry mutex across the blocking NFS scan (S111 discipline:
        // scan off-lock via spawn_blocking, apply under a brief lock).
        let _guard = self.state.disk_op_lock.lock().await;

        let (local_path, archive_path_root) = {
            let reg = self.state.model_registry.lock().await;
            (
                reg.local_path().to_path_buf(),
                reg.archive_path().to_path_buf(),
            )
        };
        let scan = tokio::task::spawn_blocking(move || {
            crate::models::registry::ModelRegistry::scan_disk(local_path, archive_path_root)
        })
        .await
        .map_err(|e| format!("reconcile scan task failed: {e}"))?;

        // Locate the model by the host-stripped key reconcile actually records.
        // A concurrent ingest may have already demoted it to Cold between our
        // pull and this reconcile — in that case the requested state is already
        // achieved, so return success rather than a spurious "not found as warm".
        enum Found {
            Warm(String),
            AlreadyCold(crate::models::ingest::ColdLanded),
        }
        let found = {
            let mut reg = self.state.model_registry.lock().await;
            reg.apply_reconcile(scan);
            let matches = |r: &ModelRecord| candidates.iter().any(|c| c == &r.name);
            // Bind to a local (note the trailing `;`) so the `all_records()`
            // iterator borrow of `reg` is dropped before the block's tail.
            let result: Found = if let Some(rec) = reg
                .all_records()
                .find(|r| r.tier == StorageTier::Warm && matches(r))
            {
                Found::Warm(rec.name.clone())
            } else if let Some(rec) = reg
                .all_records()
                .find(|r| r.tier == StorageTier::Cold && matches(r))
            {
                Found::AlreadyCold(crate::models::ingest::ColdLanded {
                    registry_name: rec.name.clone(),
                    archive_path: rec
                        .archive_path
                        .clone()
                        .unwrap_or_else(|| reg.archive_path().display().to_string()),
                    bytes: rec.size_bytes,
                })
            } else {
                return Err("pulled model not found as warm or cold after reconcile".to_string());
            };
            result
        };

        let registry_name = match found {
            Found::AlreadyCold(landed) => return Ok(landed),
            Found::Warm(name) => name,
        };

        let evicted = eviction::evict_to_archive(
            &self.state.model_registry,
            &registry_name,
            self.state.local_evictor.as_ref(),
            self.archive_copy_timeout,
        )
        .await
        .map_err(|e| e.to_string())?;

        let archive_path = {
            let reg = self.state.model_registry.lock().await;
            reg.get(&registry_name)
                .and_then(|r| r.archive_path.clone())
                .unwrap_or_else(|| reg.archive_path().display().to_string())
        };

        Ok(crate::models::ingest::ColdLanded {
            registry_name,
            archive_path,
            bytes: evicted.freed_bytes,
        })
    }
}

/// Sum the on-disk size of an HF model from its `?blobs=true` model-info JSON.
///
/// When `revision` (an Ollama quant selector, e.g. `Q4_K_M`) is given, only
/// sibling files whose filename contains that tag (case-insensitive) are summed,
/// so a multi-quant GGUF repo's whole-repo `usedStorage` doesn't trigger a false
/// `too_large`. When no quant is given, or no per-file sizes are available, falls
/// back to `usedStorage`. `None` ⇒ size unknown (the caller skips the size gate).
fn hf_repo_size_bytes(body: &serde_json::Value, revision: Option<&str>) -> Option<u64> {
    let quant = revision.map(str::trim).filter(|r| !r.is_empty());
    if let Some(siblings) = body.get("siblings").and_then(|s| s.as_array()) {
        let mut sum: u64 = 0;
        let mut matched = false;
        for sib in siblings {
            let rfilename = sib.get("rfilename").and_then(|v| v.as_str()).unwrap_or("");
            if let Some(q) = quant {
                if !rfilename
                    .to_ascii_lowercase()
                    .contains(&q.to_ascii_lowercase())
                {
                    continue;
                }
            }
            // ?blobs=true surfaces `size` (and `lfs.size` for LFS objects).
            let size = sib.get("size").and_then(|v| v.as_u64()).or_else(|| {
                sib.get("lfs")
                    .and_then(|l| l.get("size"))
                    .and_then(|v| v.as_u64())
            });
            if let Some(sz) = size {
                sum += sz;
                matched = true;
            }
        }
        if matched && sum > 0 {
            return Some(sum);
        }
    }
    // Fallback: whole-repo storage (only meaningful when no quant filter, but
    // better than nothing when sibling sizes are absent).
    body.get("usedStorage")
        .and_then(|v| v.as_u64())
        .filter(|b| *b > 0)
}

/// `POST /api/models/ingest` — ASK4-P2A: pull a HuggingFace model into cold
/// storage so the MINT sweep can later promote+test it.
///
/// Same JWT auth as every other control endpoint. Master-gated OFF by default
/// (`CHORD_MODEL_INGEST_ENABLED`) — the gate is checked BEFORE input validation,
/// so a disabled endpoint returns `disabled` even for a malformed body. Returns
/// a structured [`crate::models::ingest::IngestOutcome`] as JSON; the HTTP status
/// code is derived from the outcome. Never panics; never leaks the HF token.
pub async fn ingest_model(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<IngestRequest>,
) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }

    let cfg = crate::models::ingest::IngestConfig::from_env();

    // Gate FIRST — a disabled feature does no work and leaks no validation
    // detail (mirrors the module's gate→validate order).
    if !cfg.enabled {
        let outcome = crate::models::ingest::ingest_model(&NoopIngestOps, &cfg, &req).await;
        let code = StatusCode::from_u16(outcome.http_status())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        return (code, Json(outcome)).into_response();
    }

    // Precise 400 for malformed input, once enabled.
    if let Err(msg) = crate::models::ingest::validate_request(&req) {
        return error_response(StatusCode::BAD_REQUEST, msg);
    }

    let ops = AppStateIngestOps {
        state: &state,
        ollama_url: cfg.ollama_url.clone(),
        pull_timeout: cfg.pull_timeout,
        archive_copy_timeout: std::time::Duration::from_secs(state.model_archive_copy_timeout_secs),
    };

    let outcome = crate::models::ingest::ingest_model(&ops, &cfg, &req).await;

    // TIER-05 cold-quota pre-flight (trigger a): a fresh cold pull just grew the
    // archive — if it is now over quota, prune the least-qualified cold models
    // back under it. DRY-RUN by default (logs a would-prune plan, deletes
    // nothing). Only after a NEW landing (Ingested); an AlreadyPresent/gated/error
    // outcome added no bytes. Best-effort + non-blocking to the ingest response.
    if matches!(outcome.status, crate::models::ingest::IngestStatus::Ingested) {
        let archive_root = {
            let reg = state.model_registry.lock().await;
            reg.archive_path().to_path_buf()
        };
        let cold_cfg = crate::models::cold_quota::ColdQuotaConfig::from_env();
        // CQH-01 (F2/F4): LIVE keep set — dynamic lumina targets re-queried at
        // delete time, unioned with the VRAM keep-resident pins.
        let keep = crate::models::cold_quota::LuminaResidentKeepSet::new(
            state.lumina_aliases.clone(),
            state.model_aliases.clone(),
            crate::routing::resident_set::ResidentSetConfig::from_env(),
        );
        crate::models::cold_quota::run_cold_quota_pass_with_source(
            &state.model_registry,
            &archive_root,
            state.disk_probe.as_ref(),
            &state.cold_score_source,
            &keep,
            &cold_cfg,
            &state.disk_op_lock,
        )
        .await;
    }

    let code =
        StatusCode::from_u16(outcome.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (code, Json(outcome)).into_response()
}

/// A no-op [`crate::models::ingest::IngestOps`] used only for the disabled-gate
/// path: `ingest_model` returns `disabled` before touching any of these, so they
/// are never actually called.
struct NoopIngestOps;

#[async_trait::async_trait]
impl crate::models::ingest::IngestOps for NoopIngestOps {
    async fn find_existing(
        &self,
        _candidates: &[String],
    ) -> Option<crate::models::ingest::ExistingModel> {
        None
    }
    async fn probe(
        &self,
        _hf_repo: &str,
        _revision: Option<&str>,
        _token: Option<&str>,
    ) -> Result<crate::models::ingest::HfProbe, String> {
        Err("ingest disabled".into())
    }
    async fn pull_warm(&self, _ollama_ref: &str) -> Result<(), String> {
        Err("ingest disabled".into())
    }
    async fn land_cold(
        &self,
        _candidates: &[String],
    ) -> Result<crate::models::ingest::ColdLanded, String> {
        Err("ingest disabled".into())
    }
}

// ── GET /api/routes, GET /api/routes/:id (CHRD-100) ──────────────────────────
//
// The LOGICAL route catalog. `/api/models` publishes the model INVENTORY — what
// is on disk, how big it is, which tier it sits in. That is an operator's view
// of storage, and it is the wrong list to put in front of anyone choosing where
// to send a conversation: it names models, it changes when a sweep archives
// something, and it says nothing about whether a name is one a caller may
// target. This endpoint publishes the other thing: the named routes, what each
// is FOR, whether it runs on fleet hardware or leaves it, and whether it works
// right now.
//
// The two endpoints are deliberately not merged. A route's identity outlives
// its target — the assistant-fit updater repoints the lumina tiers at runtime —
// so a catalog that was a projection of the model list would change identity
// under its consumers every time a model was promoted.
//
// All resolution logic and the no-names guarantee live in
// `crate::routing::route_catalog`; this handler is the I/O around it.

/// Every route id Chord currently serves: the static alias table plus the
/// runtime-mutable lumina tiers. A `BTreeSet` so the catalog order is stable.
/// Every route id Chord currently serves: the static alias table plus the
/// runtime-mutable lumina tiers. A `BTreeSet` so the catalog order is stable.
///
/// Alias keys that are not shaped like a ROUTE NAME are omitted, not published.
/// An alias table is free to contain a key like `qwen2.5:7b`; that is a
/// pass-through, not a logical route, and publishing it would put a model
/// reference into the one field resolution cannot reach (`id`). Fail-closed,
/// counted in the log so a missing route is explained rather than mysterious.
fn route_ids(state: &AppState) -> std::collections::BTreeSet<String> {
    let mut all: std::collections::BTreeSet<String> =
        state.model_aliases.keys().cloned().collect();
    all.extend(state.lumina_aliases.snapshot().into_keys());
    let total = all.len();
    let ids: std::collections::BTreeSet<String> = all
        .into_iter()
        .filter(|id| crate::routing::route_catalog::is_route_id(id))
        .collect();
    if ids.len() != total {
        // The rejected keys are NOT logged: they are exactly the strings most
        // likely to be model references, and a log line is still a place a name
        // ends up. The count is what an operator needs to go look.
        tracing::warn!(
            omitted = total - ids.len(),
            "route catalog: alias keys that are not route names were omitted — a route id              must be lowercase alphanumerics/-/_ so it cannot be a model reference"
        );
    }
    ids
}

/// Resolve a route id to its current target model, in exactly the order the
/// chat hot path uses (`routes::chat_completions`): the runtime lumina store
/// first, then the static alias map. A catalog that resolved differently from
/// the hot path would describe a system that does not exist.
fn route_target(state: &AppState, id: &str) -> Option<String> {
    state
        .lumina_aliases
        .resolve(id)
        .or_else(|| state.model_aliases.get(id).cloned())
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
}

/// Gather per-route facts. Takes the registry lock ONCE for the whole catalog
/// and returns owned data, so the lock is released before any probe I/O.
async fn collect_route_facts(
    state: &Arc<AppState>,
) -> (
    std::collections::BTreeMap<String, crate::routing::route_catalog::RouteFacts>,
    std::collections::BTreeMap<String, crate::models::backends::Backend>,
) {
    use crate::routing::route_catalog::RouteFacts;

    let ids = route_ids(state);
    let mut facts = std::collections::BTreeMap::new();
    let mut backends: std::collections::BTreeMap<String, crate::models::backends::Backend> =
        std::collections::BTreeMap::new();

    let reg = state.model_registry.lock().await;
    for id in ids {
        let target = route_target(state, &id);
        let (target_known, backend) = match target.as_deref() {
            None => (false, None),
            Some(t) => {
                // Two key shapes live in one registry, and a lookup that knows
                // only one of them reports a perfectly good route as broken.
                //
                //  * Ollama-managed records are keyed by the FULLY-TAGGED name,
                //    and an untagged reference means `:latest` — the same
                //    normalization `chat_completions` does before its own
                //    registry lookup. Without it every untagged alias target
                //    would report `unknown_model`.
                //  * Remote-API records (`register_remote_api_model`) are keyed
                //    by the bare name they were registered under, with no tag
                //    at all — a `:latest` suffix would never match one. This
                //    was found by the cloud-route test failing, not reasoned
                //    about in advance.
                //
                // So: exact first, then the Ollama normalization.
                let key = if reg.get(t).is_some() {
                    t.to_string()
                } else if t.contains(':') {
                    t.to_string()
                } else {
                    format!("{t}:latest")
                };
                let known = reg.get(&key).is_some();
                // `backend_for` falls back to the default backend for a model
                // it does not know, which would let an unknown target answer
                // "local and available". Only ask about a target the registry
                // actually has.
                let b = if known {
                    reg.backend_for(&key).cloned()
                } else {
                    None
                };
                (known, b)
            }
        };
        if let Some(b) = &backend {
            backends.insert(id.clone(), b.clone());
        }
        facts.insert(
            id.clone(),
            RouteFacts {
                id,
                has_target: target.is_some(),
                target_known,
                backend_kind: backend.as_ref().map(|b| b.kind),
                // Filled in after the probes below — resolution and liveness are
                // two different questions and are answered in that order.
                liveness: None,
            },
        );
    }
    drop(reg);
    (facts, backends)
}

/// Resolve liveness for every route's backend and fold it into the facts.
async fn apply_liveness(
    state: &Arc<AppState>,
    facts: &mut std::collections::BTreeMap<String, crate::routing::route_catalog::RouteFacts>,
    backends: &std::collections::BTreeMap<String, crate::models::backends::Backend>,
) {
    use crate::routing::route_catalog::classify_liveness;

    // Probe each DISTINCT always-on local backend once, not once per route.
    let urls: std::collections::BTreeSet<String> = backends
        .values()
        .filter(|b| {
            b.kind != crate::models::backends::BackendKind::OpenRouter && !b.on_demand()
        })
        .map(|b| b.url.clone())
        .collect();
    let probed = crate::routing::route_catalog::probe_always_on(&state.http_client, &urls).await;

    for (id, b) in backends {
        // Presence-oriented credential read. `var_os` is used rather than
        // `var` so the value is never decoded into a `String`, and the
        // `OsString` is reduced to a bool inside the closure without ever being
        // bound to a name, logged, returned, or compared against anything.
        //
        // The residual is stated rather than glossed: Rust's standard library
        // has no presence-only environment read — `var_os` still materializes
        // the value — so "presence only" is a property of what this code DOES
        // with it, not of the API it calls. What matters operationally is that
        // a remote route with no provisioned key would fail on first use, and
        // publishing it as available would be a lie the user only discovers
        // mid-conversation.
        let live = classify_liveness(
            b,
            |env_key| {
                std::env::var_os(env_key)
                    .is_some_and(|v| !v.as_encoded_bytes().iter().all(u8::is_ascii_whitespace))
            },
            |url| probed.get(url).copied().unwrap_or(false),
        );
        if let Some(f) = facts.get_mut(id) {
            f.liveness = Some(live);
        }
    }
}

/// `GET /api/routes` — the route catalog.
///
/// Same JWT auth as every other `/api/*` route (checked in-handler): the
/// catalog describes the fleet's serving topology, and it is gated for the same
/// reason `/api/models` is.
pub async fn list_routes(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }
    let (mut facts, backends) = collect_route_facts(&state).await;
    apply_liveness(&state, &mut facts, &backends).await;
    let decls = crate::routing::route_catalog::declarations_from_env();
    let routes = crate::routing::route_catalog::build_catalog(&facts, &decls);
    let count = routes.len();
    Json(serde_json::json!({ "routes": routes, "count": count })).into_response()
}

/// `GET /api/routes/:id` — one route, or a 404.
///
/// The 404 is the point: "no such route" is a different answer from "the route
/// exists and is unavailable", and a client that cannot tell them apart will
/// eventually render a typo as an outage.
pub async fn get_route(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }
    let (mut facts, backends) = collect_route_facts(&state).await;
    if !facts.contains_key(&id) {
        // The requested id is NOT echoed. It was caller-supplied, so echoing it
        // introduces no NEW information — but it does make this endpoint a
        // reflector, and a reflector on an endpoint whose entire contract is
        // "no model or provider name appears in my responses" is a hole
        // somebody will eventually walk a model id through. The status code is
        // the whole answer; the caller already knows what it asked for.
        return error_response(
            StatusCode::NOT_FOUND,
            "unknown route".to_string(),
        );
    }
    // Probe ONLY this route's backend. The list endpoint fans out across every
    // always-on backend because it has to; a single-route read that did the
    // same would make one route's page load depend on every other backend's
    // liveness, which is both slower and a worse answer.
    let mine: std::collections::BTreeMap<String, crate::models::backends::Backend> = backends
        .into_iter()
        .filter(|(k, _)| k == &id)
        .collect();
    apply_liveness(&state, &mut facts, &mine).await;
    let decls = crate::routing::route_catalog::declarations_from_env();
    let view = crate::routing::route_catalog::build_view(&facts[&id], decls.get(&id));
    Json(view).into_response()
}

// ── Router ───────────────────────────────────────────────────────────────────

/// Build the TIER-05 control router over the shared [`AppState`].
pub fn build_control_router(state: Arc<AppState>) -> axum::Router {
    use axum::routing::{get, post};
    axum::Router::new()
        .route("/health", get(control_health))
        // PROMEX-02: application metrics, unauthenticated like /health above.
        .route("/metrics", get(handle_metrics))
        .route("/api/models", get(list_models))
        .route("/api/models/sweep", post(trigger_sweep))
        .route("/api/models/reconcile", post(trigger_reconcile))
        .route("/api/models/:name", get(get_model))
        .route("/api/models/:name/archive", post(archive_model))
        .route("/api/models/:name/pull", post(pull_model))
        .route("/api/models/:name/protect", post(protect_model))
        // ASK4-P2A: pull a HuggingFace model into cold storage (default-OFF gate).
        .route("/api/models/ingest", post(ingest_model))
        // CHRD-100: the LOGICAL route catalog — the named routes a caller may
        // target, with locality/availability resolved BY Chord. Distinct from
        // `/api/models` (the on-disk model inventory) by design; see the
        // handler docs. Same JWT auth (checked in-handler).
        .route("/api/routes", get(list_routes))
        .route("/api/routes/:id", get(get_route))
        .route("/api/storage", get(storage_summary))
        .route("/api/storage/gc", post(trigger_gc))
        // RESIL-02: sweep action-queue cache (durable resume).
        .route("/api/sweep/session", post(sweep_session_register))
        .route("/api/sweep/session/:id", get(sweep_session_get))
        .route("/api/sweep/session/:id/advance", post(sweep_session_advance))
        // BLD-09: idle-mode admin surface — free the heavy host for the compiler.
        // `POST /admin/idle` enters idle (drain + release providers/GPU/models/RAM,
        // report freed RAM); `GET /admin/idle` reports status; `POST /admin/activate`
        // restores. Same JWT auth as every route above (checked inside the handlers).
        .route(
            "/admin/idle",
            post(crate::admin::idle::admin_idle_enter).get(crate::admin::idle::admin_idle_status),
        )
        .route("/admin/activate", post(crate::admin::idle::admin_activate))
        // CHORD-ACT-01: live serving-activity signal. Distinct from the idle-MODE phase
        // above — `GET /admin/activity` reports whether inference is actually in flight
        // and, if not, how long Chord has been quiet, so the compiler scheduler can
        // dispatch heavy builds into genuine idle windows. Same JWT auth (in-handler).
        .route(
            "/admin/activity",
            get(crate::admin::idle::admin_activity_status),
        )
        // TRTR-07: assistant-mode resident-set state — per role (embedding /
        // router / personality): the alias it resolves through, the model that
        // alias currently points at, whether it is warm, and when it was last
        // used. Residency is OBSERVABLE rather than inferred from `/api/models`.
        // Same JWT auth as every route above (checked inside the handler).
        .route("/admin/resident-set", get(resident_set_status))
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
            model_archive_copy_timeout_secs: 1800,
            // Tests write fresh blobs and assert immediate GC collection, so the
            // grace window is disabled here (0) — the B1 age-guard itself is
            // exercised directly in `models::gc`'s unit tests.
            model_gc_min_age_secs: 0,
            routing_map,
            coding_profile_source: Arc::new(Mutex::new(None)),
            score_source: Arc::new(Mutex::new(None)),
            cold_score_source: Arc::new(Mutex::new(None)),
            lumina_aliases: crate::routing::lumina_alias::LuminaAliasStore::empty(),
            personal_proxy: None,
            embeddings_config: crate::embeddings::EmbeddingsConfig::test_default(
                None,
                "http://127.0.0.1:1".to_string(),
            ),
        })
    }

    /// Like `control_state` but with a real (non-empty) `jwt_secret`, so auth is
    /// enforced — used by tests asserting the 401 path on the new MSM-04
    /// endpoints.
    fn control_state_with_secret(
        registry: Arc<Mutex<ModelRegistry>>,
        local_root: std::path::PathBuf,
        secret: &str,
    ) -> Arc<AppState> {
        let state = control_state(registry, local_root, Arc::new(StatvfsProbe));
        // Freshly constructed above with a single owner, so this never fails.
        let mut owned = Arc::try_unwrap(state).ok().expect("sole Arc owner");
        owned.jwt_secret = secret.to_string();
        Arc::new(owned)
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

    // RESIL-02: sweep-session cache routes end-to-end. Uses a UNIQUE session id
    // so it never collides with the process-global SWEEP_SESSIONS store.
    #[tokio::test]
    async fn sweep_session_register_get_advance_and_404() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        let registry = Arc::new(Mutex::new(reg_at(base, vec![])));
        let state = control_state(registry, base.join("local"), Arc::new(StatvfsProbe));
        let app = build_control_router(state);

        let sid = "resil02-ctrl-test-unique-abc123";

        // Register a 3-item queue.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/sweep/session")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({ "session_id": sid, "queue": ["a|1", "b|2", "c|3"] })
                            .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["total"], 3);
        assert_eq!(json["done_count"], 0);

        // Advance one key.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(format!("/api/sweep/session/{sid}/advance"))
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::json!({ "keys": ["b|2"] }).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["done_count"], 1);
        let remaining: Vec<&str> = json["remaining"].as_array().unwrap().iter()
            .map(|v| v.as_str().unwrap()).collect();
        assert_eq!(remaining, vec!["a|1", "c|3"]);

        // GET reflects the smaller remaining set.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri(format!("/api/sweep/session/{sid}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["remaining"].as_array().unwrap().len(), 2);

        // Unknown session ⇒ 404.
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/sweep/session/resil02-does-not-exist-xyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
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

    // ── TRTR-07: GET /admin/resident-set ────────────────────────────────────────

    /// The report must expose EVERY role slot (in priority order) with its alias,
    /// state, and warm flag — residency observable, not inferred. Auth disabled
    /// (empty secret) so this exercises the handler, not the gate.
    #[tokio::test]
    async fn resident_set_report_lists_every_role() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        let registry = Arc::new(Mutex::new(reg_at(base, vec![])));
        let state = control_state_with_secret(registry, base.join("local"), "");
        let app = build_control_router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/admin/resident-set")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        let roles = v.get("roles").and_then(|r| r.as_array()).unwrap();
        assert_eq!(roles.len(), 3);
        let ids: Vec<&str> = roles
            .iter()
            .map(|r| r.get("role").and_then(|x| x.as_str()).unwrap())
            .collect();
        assert_eq!(ids, vec!["personality", "router", "embedding"]);
        for r in roles {
            assert!(r.get("alias").and_then(|x| x.as_str()).is_some());
            assert!(r.get("state").is_some());
            assert!(r.get("warm").and_then(|x| x.as_bool()).is_some());
            assert!(r.get("last_used").is_some(), "last_used reported (may be null)");
        }
        assert!(v.get("active").is_some());
        assert!(v.get("registry_exempt").and_then(|x| x.as_array()).is_some());
    }

    // ── MSM-04: /api/models/reconcile, /api/storage/gc ─────────────────────────

    // ── BLD-09: /admin/idle, /admin/activate ────────────────────────────────────

    // Auth-wiring only: a non-empty secret with no token ⇒ 401 on every idle-mode
    // route, so the request never reaches `enter_idle`/`activate` and the
    // process-global IDLE_MODE is never touched (keeping this test parallel-safe).
    #[tokio::test]
    async fn idle_mode_routes_require_auth() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        let registry = Arc::new(Mutex::new(reg_at(base, vec![])));
        let state = control_state_with_secret(registry, base.join("local"), "s");
        let app = build_control_router(state);

        for (method, uri) in [
            (Method::POST, "/admin/idle"),
            (Method::GET, "/admin/idle"),
            (Method::POST, "/admin/activate"),
            (Method::GET, "/admin/activity"),
            // TRTR-07: the resident-set report is auth-gated the same way.
            (Method::GET, "/admin/resident-set"),
        ] {
            let resp = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method(method.clone())
                        .uri(uri)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::UNAUTHORIZED,
                "{method} {uri} must require auth"
            );
        }
    }

    #[tokio::test]
    async fn reconcile_requires_auth() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        let reg = reg_at(base, vec![]);
        let registry = Arc::new(Mutex::new(reg));
        let state = control_state_with_secret(registry, base.join("local"), "s");
        let app = build_control_router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/models/reconcile")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn reconcile_returns_before_after_tier_counts_and_persists() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        // A model on disk that the in-memory registry doesn't know about yet —
        // reconcile must discover it and the response must reflect the change.
        make_model(&base.join("local"), "fresh", "1", &[100]);
        let reg = reg_at(base, vec![]); // deliberately NOT reconciled yet
        let registry = Arc::new(Mutex::new(reg));
        let state = control_state(registry.clone(), base.join("local"), Arc::new(StatvfsProbe));
        let app = build_control_router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/models/reconcile")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["before"]["warm"], 0);
        assert_eq!(json["after"]["warm"], 1);
        assert_eq!(json["persisted"], true);

        // Registry actually updated in place.
        assert_eq!(registry.lock().await.get("fresh:1").unwrap().tier, StorageTier::Warm);
        // And persisted to disk (MSM-01).
        assert!(base.join("registry.json").is_file());
    }

    #[tokio::test]
    async fn gc_endpoint_deletes_orphan_and_reports_freed_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        // Local orphan blob (no referencing manifest) also present in archive.
        std::fs::create_dir_all(base.join("local/blobs")).unwrap();
        std::fs::create_dir_all(base.join("archive/blobs")).unwrap();
        std::fs::write(base.join("local/blobs/sha256-orphan1"), vec![b'x'; 64]).unwrap();
        std::fs::write(base.join("archive/blobs/sha256-orphan1"), vec![b'x'; 64]).unwrap();
        let reg = reg_at(base, vec![]);
        let registry = Arc::new(Mutex::new(reg));
        let state = control_state(registry, base.join("local"), Arc::new(StatvfsProbe));
        let app = build_control_router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/storage/gc")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["orphans_deleted"], 1);
        assert_eq!(json["freed_bytes"], 64);
        assert!(!base.join("local/blobs/sha256-orphan1").exists());
    }

    // ── ASK4-P2A: /api/models/ingest ─────────────────────────────────────────

    #[tokio::test]
    async fn ingest_requires_auth() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        let registry = Arc::new(Mutex::new(reg_at(base, vec![])));
        let state = control_state_with_secret(registry, base.join("local"), "s");
        let app = build_control_router(state);

        // Valid JSON body so the Json extractor succeeds and the in-handler
        // auth_check is what rejects the request.
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/models/ingest")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({ "hf_repo": "org/model", "model_name": "m" })
                            .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn ingest_disabled_by_default_refuses() {
        // Default (gate unset) ⇒ disabled ⇒ 403 with status "disabled",
        // touching no network/registry state.
        std::env::remove_var("CHORD_MODEL_INGEST_ENABLED");
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        let registry = Arc::new(Mutex::new(reg_at(base, vec![])));
        // Empty jwt_secret ⇒ auth disabled, so we reach the gate.
        let state = control_state(registry, base.join("local"), Arc::new(StatvfsProbe));
        let app = build_control_router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/models/ingest")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({ "hf_repo": "org/model", "model_name": "m" })
                            .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["status"], "disabled");
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn ingest_enabled_rejects_malformed_repo_with_400() {
        // Gate is checked BEFORE validation, so the 400 path is only reachable
        // once the feature is enabled.
        std::env::set_var("CHORD_MODEL_INGEST_ENABLED", "1");
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        let registry = Arc::new(Mutex::new(reg_at(base, vec![])));
        let state = control_state(registry, base.join("local"), Arc::new(StatvfsProbe));
        let app = build_control_router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/models/ingest")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({ "hf_repo": "not a repo", "model_name": "m" })
                            .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        std::env::remove_var("CHORD_MODEL_INGEST_ENABLED");
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn ingest_disabled_refuses_even_malformed_before_validation() {
        // Gate-first: a malformed body under a disabled gate returns `disabled`
        // (403), never a 400 — the disabled endpoint leaks no validation detail.
        std::env::remove_var("CHORD_MODEL_INGEST_ENABLED");
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        let registry = Arc::new(Mutex::new(reg_at(base, vec![])));
        let state = control_state(registry, base.join("local"), Arc::new(StatvfsProbe));
        let app = build_control_router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/models/ingest")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({ "hf_repo": "not a repo", "model_name": "m" })
                            .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["status"], "disabled");
    }

    // ── CHRD-100: the route catalog ──────────────────────────────────────────

    /// A control state whose alias table (and therefore route set) is seeded.
    fn control_state_with_aliases(
        registry: Arc<Mutex<ModelRegistry>>,
        local_root: std::path::PathBuf,
        aliases: &[(&str, &str)],
    ) -> Arc<AppState> {
        let state = control_state(registry, local_root, Arc::new(StatvfsProbe));
        let mut owned = Arc::try_unwrap(state).ok().expect("sole Arc owner");
        owned.model_aliases = aliases
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        Arc::new(owned)
    }

    /// An ON-DEMAND local backend. On-demand on purpose: these tests must never
    /// make a network call, and `classify_liveness` never probes an on-demand
    /// backend (being stopped is its resting state).
    fn on_demand_backend(name: &str) -> crate::models::backends::Backend {
        crate::models::backends::Backend {
            name: name.to_string(),
            url: format!("http://127.0.0.1:1/{name}"),
            hardware: crate::models::backends::Hardware::Gpu,
            kind: crate::models::backends::BackendKind::LlamaServer,
            unit: None,
            always_on: false,
            idle_stop_secs: 600,
            launch: None,
            api_key_env: None,
        }
    }

    async fn get_json(app: axum::Router, uri: &str) -> (StatusCode, Value) {
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri(uri)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (status, json)
    }

    fn route_of<'a>(json: &'a Value, id: &str) -> &'a Value {
        json["routes"]
            .as_array()
            .expect("a catalog always has a routes array")
            .iter()
            .find(|r| r["id"] == id)
            .unwrap_or_else(|| panic!("route {id} missing from the catalog"))
    }

    #[tokio::test]
    async fn an_alias_is_published_as_a_route_with_locality_derived_from_its_backend() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        make_model(&base.join("local"), "alpha", "1", &[100]);
        let mut reg = reg_at(base, vec![]);
        reg.upsert_backend(on_demand_backend("llama-gpu"));
        reg.reconcile();
        let registry = Arc::new(Mutex::new(reg));
        let state = control_state_with_aliases(
            registry,
            base.join("local"),
            &[("lumina-fast", "alpha:1")],
        );
        let (status, json) = get_json(build_control_router(state), "/api/routes").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["count"], 1);
        let r = route_of(&json, "lumina-fast");
        assert_eq!(r["available"], true);
        assert_eq!(
            r["locality"], "local",
            "locality must come from the resolved backend's kind"
        );
        assert!(r.get("unavailable_reason").is_none());
        // Chord's own tier has a purpose it can state; it is not the id.
        assert_eq!(r["label"], "Quick conversational answers");
    }

    #[tokio::test]
    async fn a_route_whose_target_the_registry_never_heard_of_is_a_fault_not_an_available_local_route()
    {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        make_model(&base.join("local"), "alpha", "1", &[100]);
        let mut reg = reg_at(base, vec![]);
        reg.upsert_backend(on_demand_backend("llama-gpu"));
        reg.reconcile();
        let registry = Arc::new(Mutex::new(reg));
        let state = control_state_with_aliases(
            registry,
            base.join("local"),
            &[("ghost-route", "not-a-model:9")],
        );
        let (_, json) = get_json(build_control_router(state), "/api/routes").await;
        let r = route_of(&json, "ghost-route");
        assert_eq!(r["available"], false);
        assert_eq!(r["unavailable_reason"], "unknown_model");
        assert!(
            r.get("locality").is_none(),
            "a route whose target cannot be placed has NO locality — the registry's default \
             backend must not be borrowed to manufacture one"
        );
    }

    #[tokio::test]
    async fn an_untagged_alias_target_resolves_the_same_way_the_chat_hot_path_does() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        make_model(&base.join("local"), "alpha", "latest", &[100]);
        let mut reg = reg_at(base, vec![]);
        reg.upsert_backend(on_demand_backend("llama-gpu"));
        reg.reconcile();
        let registry = Arc::new(Mutex::new(reg));
        // The alias target carries no tag, exactly as an operator would write it.
        let state =
            control_state_with_aliases(registry, base.join("local"), &[("untagged", "alpha")]);
        let (_, json) = get_json(build_control_router(state), "/api/routes").await;
        let r = route_of(&json, "untagged");
        assert_eq!(
            r["available"], true,
            "the registry is keyed by the fully-tagged name; without the `:latest` \
             normalization every untagged alias would falsely report unknown_model"
        );
    }

    #[tokio::test]
    async fn a_remote_route_is_reported_cloud_and_disabled_when_its_credential_is_unprovisioned() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        let mut reg = reg_at(base, vec![]);
        let mut remote = on_demand_backend("openrouter");
        remote.kind = crate::models::backends::BackendKind::OpenRouter;
        remote.always_on = true;
        remote.api_key_env = Some("CHRD100_TEST_KEY_THAT_IS_NEVER_SET".to_string());
        reg.upsert_backend(remote);
        assert!(reg.register_remote_api_model("owl-alpha", "remote-api", "openrouter"));
        let registry = Arc::new(Mutex::new(reg));
        let state = control_state_with_aliases(
            registry,
            base.join("local"),
            &[("frontier", "owl-alpha")],
        );
        let (_, json) = get_json(build_control_router(state), "/api/routes").await;
        let r = route_of(&json, "frontier");
        assert_eq!(
            r["locality"], "cloud",
            "a remote bearer-authenticated backend is cloud — this is the fact SCOUT cannot \
             derive for itself"
        );
        assert_eq!(r["available"], false);
        assert_eq!(r["unavailable_reason"], "disabled");
    }

    #[tokio::test]
    async fn the_published_catalog_contains_no_model_backend_or_url_string() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        make_model(&base.join("local"), "alpha", "1", &[100]);
        let mut reg = reg_at(base, vec![]);
        reg.upsert_backend(on_demand_backend("llama-gpu"));
        reg.reconcile();
        assert!(reg.set_model_backend("alpha:1", Some("llama-gpu".to_string())));
        let registry = Arc::new(Mutex::new(reg));
        let state = control_state_with_aliases(
            registry,
            base.join("local"),
            &[("lumina-deep", "alpha:1")],
        );
        let (_, json) = get_json(build_control_router(state), "/api/routes").await;
        let body = serde_json::to_string(&json).unwrap();
        for leak in ["alpha", "llama-gpu", "127.0.0.1", "http://"] {
            assert!(
                !body.contains(leak),
                "the route catalog leaked {leak:?} into its response: {body}"
            );
        }
    }

    #[tokio::test]
    async fn an_unknown_route_id_is_404_which_is_not_the_same_as_unavailable() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        make_model(&base.join("local"), "alpha", "1", &[100]);
        let mut reg = reg_at(base, vec![]);
        reg.upsert_backend(on_demand_backend("llama-gpu"));
        reg.reconcile();
        let registry = Arc::new(Mutex::new(reg));
        let state = control_state_with_aliases(
            registry,
            base.join("local"),
            &[("lumina-fast", "alpha:1")],
        );
        let app = build_control_router(state);
        let (status, _) = get_json(app.clone(), "/api/routes/no-such-route").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let (status, json) = get_json(app, "/api/routes/lumina-fast").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["id"], "lumina-fast");
        assert_eq!(json["available"], true);
    }

    #[tokio::test]
    async fn the_route_catalog_is_auth_gated_like_every_other_api_route() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        let reg = reg_at(base, vec![]);
        let registry = Arc::new(Mutex::new(reg));
        let state = control_state_with_secret(registry, base.join("local"), "a-secret");
        let app = build_control_router(state);
        let (status, _) = get_json(app.clone(), "/api/routes").await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        let (status, _) = get_json(app, "/api/routes/lumina-fast").await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn an_alias_key_that_is_a_model_reference_is_not_published_as_a_route() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        make_model(&base.join("local"), "alpha", "1", &[100]);
        let mut reg = reg_at(base, vec![]);
        reg.upsert_backend(on_demand_backend("llama-gpu"));
        reg.reconcile();
        let registry = Arc::new(Mutex::new(reg));
        let state = control_state_with_aliases(
            registry,
            base.join("local"),
            &[("lumina-fast", "alpha:1"), ("alpha:1", "alpha:1")],
        );
        let (_, json) = get_json(build_control_router(state), "/api/routes").await;
        assert_eq!(json["count"], 1, "the model-shaped alias key must be omitted");
        assert_eq!(route_of(&json, "lumina-fast")["id"], "lumina-fast");
        assert!(
            !serde_json::to_string(&json).unwrap().contains("alpha"),
            "`id` is the one string resolution cannot reach — it must not carry a model \
             reference either"
        );
    }

    #[tokio::test]
    async fn a_404_does_not_reflect_the_requested_id_back() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        let reg = reg_at(base, vec![]);
        let registry = Arc::new(Mutex::new(reg));
        let state = control_state_with_aliases(registry, base.join("local"), &[]);
        let (status, json) = get_json(
            build_control_router(state),
            "/api/routes/qwen-something-distinctive",
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(
            !serde_json::to_string(&json)
                .unwrap()
                .contains("qwen-something-distinctive"),
            "an endpoint whose contract is 'no model name appears in my responses' must not \
             be a reflector"
        );
    }

    #[tokio::test]
    async fn every_string_in_the_response_comes_from_the_route_id_or_a_declaration() {
        // A value-level allowlist, not a list of four fixture substrings: walk
        // the whole response and prove each string is one this endpoint is
        // entitled to emit. A refactor that leaked some OTHER model or backend
        // name — not just the ones this test happens to name — fails here.
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        make_model(&base.join("local"), "zephyrine", "9", &[100]);
        let mut reg = reg_at(base, vec![]);
        reg.upsert_backend(on_demand_backend("bespoke-serving-process"));
        reg.reconcile();
        assert!(reg.set_model_backend("zephyrine:9", Some("bespoke-serving-process".to_string())));
        let registry = Arc::new(Mutex::new(reg));
        let state = control_state_with_aliases(
            registry,
            base.join("local"),
            &[("lumina-deep", "zephyrine:9")],
        );
        let (_, json) = get_json(build_control_router(state), "/api/routes").await;

        let allowed: std::collections::BTreeSet<String> = ["lumina-deep", "local", "cloud"]
            .iter()
            .map(|s| (*s).to_string())
            .chain(
                crate::routing::route_catalog::declarations_from_env()
                    .values()
                    .filter_map(|d| d.label.clone()),
            )
            .chain(
                crate::routing::route_catalog::COST_TIERS
                    .iter()
                    .map(|s| (*s).to_string()),
            )
            .chain(
                ["no_target", "unknown_model", "no_backend", "unreachable", "disabled"]
                    .iter()
                    .map(|s| (*s).to_string()),
            )
            .collect();

        fn strings(v: &Value, out: &mut Vec<String>) {
            match v {
                Value::String(s) => out.push(s.clone()),
                Value::Array(a) => a.iter().for_each(|x| strings(x, out)),
                Value::Object(o) => o.values().for_each(|x| strings(x, out)),
                _ => {}
            }
        }
        let mut found = Vec::new();
        strings(&json, &mut found);
        assert!(!found.is_empty());
        for s in found {
            assert!(
                allowed.contains(&s),
                "the catalog emitted the string {s:?}, which is neither a route id, a \
                 declared label/cost tier, a locality, nor a reason code"
            );
        }
    }

    #[tokio::test]
    async fn the_runtime_lumina_target_wins_over_the_static_alias_entry() {
        // The chat hot path resolves the lumina store FIRST. A catalog that
        // resolved the static map first would describe a route pointing
        // somewhere the traffic does not go.
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        make_model(&base.join("local"), "stale", "1", &[100]);
        let mut reg = reg_at(base, vec![]);
        reg.upsert_backend(on_demand_backend("llama-gpu"));
        reg.reconcile();
        let registry = Arc::new(Mutex::new(reg));
        // Static map points at a KNOWN model; the runtime store has been
        // repointed at one the registry does not have. Hot-path order means the
        // route must report the RUNTIME target's fault, not the stale success.
        let state = control_state_with_aliases(
            registry,
            base.join("local"),
            &[("lumina-fast", "stale:1")],
        );
        state
            .lumina_aliases
            .set("lumina-fast", "freshly-promoted:1".to_string())
            .await;
        let (_, json) = get_json(build_control_router(state), "/api/routes").await;
        let r = route_of(&json, "lumina-fast");
        assert_eq!(
            r["unavailable_reason"], "unknown_model",
            "the runtime lumina target must win — resolving the static map first would have \
             reported this route as available"
        );
    }

    #[tokio::test]
    async fn an_exact_registry_key_wins_over_the_latest_normalization() {
        // Both shapes exist for the same bare name, tagged to backends of
        // DIFFERENT locality. Preferring the normalized key would report the
        // wrong side of the local/cloud line — the one fact SCOUT cannot check.
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        make_model(&base.join("local"), "duo", "latest", &[100]);
        let mut reg = reg_at(base, vec![]);
        reg.upsert_backend(on_demand_backend("llama-gpu"));
        let mut remote = on_demand_backend("openrouter");
        remote.kind = crate::models::backends::BackendKind::OpenRouter;
        remote.always_on = true;
        remote.api_key_env = Some("CHRD100_TEST_KEY_THAT_IS_NEVER_SET".to_string());
        reg.upsert_backend(remote);
        reg.reconcile();
        assert!(reg.set_model_backend("duo:latest", Some("llama-gpu".to_string())));
        // The bare name is a remote-API record — a different model that merely
        // shares a prefix.
        assert!(reg.register_remote_api_model("duo", "remote-api", "openrouter"));
        let registry = Arc::new(Mutex::new(reg));
        let state =
            control_state_with_aliases(registry, base.join("local"), &[("ambiguous", "duo")]);
        let (_, json) = get_json(build_control_router(state), "/api/routes").await;
        assert_eq!(
            route_of(&json, "ambiguous")["locality"], "cloud",
            "the exact key must win; falling through to `duo:latest` would report a cloud \
             route as local"
        );
    }
}
