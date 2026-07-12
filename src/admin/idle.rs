//! BLD-09: Chord idle-mode admin API — release providers, GPU, models, and RAM.
//!
//! ## Why this exists
//! The constellation CI/CD compiler (S117) builds on the heavy GPU/big-RAM host.
//! To hand that host to a build WITHOUT taking Chord down, the compiler asks Chord
//! to go *idle*: stop accepting new inference, drain what's in flight, stop the
//! on-demand inference backends, unload every resident model from VRAM (demoting
//! them back to warm storage so the system RAM/VRAM they held is freed), and enter
//! a low-footprint wait. When the build finishes — or lazily, on the first real
//! request afterwards — Chord *activates* and resumes normal serving (models
//! reload on demand, exactly as they do from a cold start).
//!
//! This mirrors the [`crate::gpu_exclusive`] subsystem's shape almost exactly
//! (process-global state + durable persistence + a decision core separated from
//! IO), but where GPU-exclusive *gates* inference with a 503, idle-mode instead
//! *lazily restores* on the next request — the compiler never wants a hard 503,
//! it wants Chord to quietly get out of the way and quietly come back.
//!
//! ## Contract (see `README.md`)
//! - `POST /admin/idle`      → enter idle; reports freed RAM. Idempotent.
//! - `POST /admin/activate`  → restore service. Idempotent. Also happens lazily
//!                             on the next inference request ([`lazy_activate`]).
//! - `GET  /admin/idle`      → current idle/active status + resume manifest.
//! - A watchdog ([`watchdog_loop`]) re-activates on timeout so the proxy is never
//!   left silently dead; it holds off only while a compiler GPU-exclusive lease is
//!   actively held.
//!
//! ## Testability
//! The pure decision logic ([`decide_enter`], [`decide_activate`],
//! [`ResumeManifest::watchdog_expired`], and the in-memory [`IdleController`]
//! transitions) is separated from the clock, the filesystem, and the network so it
//! is exhaustively unit-testable offline with no global state and no sleeping. The
//! release/restore *side effects* (stopping backends, evicting VRAM, reading
//! `/proc/meminfo`) live in the async orchestration functions and are best-effort.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::gpu_exclusive::now_epoch;

/// Default hard-timeout (seconds) after which the watchdog re-activates an idle
/// proxy if no compiler GPU-exclusive lease is held. 1 hour: comfortably longer
/// than a heavy fleet build, short enough that a crashed/forgotten compiler never
/// wedges Chord idle indefinitely. Override with `CHORD_IDLE_WATCHDOG_SECS`.
pub const DEFAULT_WATCHDOG_SECS: u64 = 3600;

/// Default bound (seconds) on draining in-flight inference before releasing
/// resources. Kept short — real chat/agent turns finish in seconds, and a request
/// that overruns this bound is left to complete on its own while release proceeds
/// (the report flags `inflight_remaining > 0`). Override `CHORD_IDLE_DRAIN_SECS`.
pub const DEFAULT_DRAIN_SECS: u64 = 30;

/// Resolve the watchdog timeout from `CHORD_IDLE_WATCHDOG_SECS` (seconds); a
/// missing/blank/zero/unparseable value falls back to [`DEFAULT_WATCHDOG_SECS`].
pub fn watchdog_secs_from_env() -> u64 {
    parse_positive_env("CHORD_IDLE_WATCHDOG_SECS", DEFAULT_WATCHDOG_SECS)
}

/// Resolve the in-flight drain bound from `CHORD_IDLE_DRAIN_SECS` (seconds); a
/// missing/blank/zero/unparseable value falls back to [`DEFAULT_DRAIN_SECS`].
pub fn drain_secs_from_env() -> u64 {
    parse_positive_env("CHORD_IDLE_DRAIN_SECS", DEFAULT_DRAIN_SECS)
}

fn parse_positive_env(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default)
}

// ── Resume manifest ───────────────────────────────────────────────────────────

/// What to restore when leaving idle, plus the bookkeeping the idle response and
/// the watchdog need. Persisted (when `CHORD_STATE_DIR` is set) so a crash mid-idle
/// leaves a record the watchdog can act on after restart.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResumeManifest {
    /// Who/what requested idle (e.g. `"compiler"`), for diagnostics. Never a secret.
    pub reason: String,
    /// Epoch seconds idle was entered.
    pub entered_at: u64,
    /// Epoch seconds after which the watchdog will auto-activate (unless a
    /// compiler GPU-exclusive lease is still held).
    pub watchdog_deadline: u64,
    /// Names of the models that were resident in VRAM when idle was entered, so
    /// activate can note them. Restoration is LAZY — Ollama/llama-server reload a
    /// model on its next request exactly as from cold — so this list is
    /// informational, not a force-reload instruction.
    pub resident_models: Vec<String>,
    /// `MemAvailable` (GiB) sampled just before release, for the freed-RAM delta.
    pub mem_available_before_gb: f64,
}

impl ResumeManifest {
    /// Has the watchdog deadline passed at `now`? (Pure — the lease-held override
    /// is applied by the watchdog, not here.)
    pub fn watchdog_expired(&self, now: u64) -> bool {
        now >= self.watchdog_deadline
    }
}

// ── Pure decisions ────────────────────────────────────────────────────────────

/// Pure decision for a `POST /admin/idle`, given the current state.
#[derive(Debug, PartialEq, Eq)]
pub enum EnterDecision {
    /// Currently active ⇒ run the release side effects and enter idle.
    Enter,
    /// Already idle ⇒ idempotent no-op (do NOT re-run release).
    AlreadyIdle,
}

pub fn decide_enter(current: Option<&ResumeManifest>) -> EnterDecision {
    match current {
        None => EnterDecision::Enter,
        Some(_) => EnterDecision::AlreadyIdle,
    }
}

/// Pure decision for a `POST /admin/activate`, given the current state.
#[derive(Debug, PartialEq, Eq)]
pub enum ActivateDecision {
    /// Currently idle ⇒ restore.
    Restore,
    /// Already active ⇒ idempotent no-op.
    AlreadyActive,
}

pub fn decide_activate(current: Option<&ResumeManifest>) -> ActivateDecision {
    match current {
        Some(_) => ActivateDecision::Restore,
        None => ActivateDecision::AlreadyActive,
    }
}

// ── In-memory controller + durable persistence ───────────────────────────────

/// Outcome of applying an enter against the live state.
#[derive(Debug, PartialEq)]
pub enum EnterOutcome {
    /// Transitioned active → idle; carries the stored manifest.
    Entered(ResumeManifest),
    /// Already idle; carries the existing manifest (idempotent).
    AlreadyIdle(ResumeManifest),
}

/// Outcome of applying an activate against the live state.
#[derive(Debug, PartialEq)]
pub enum ActivateOutcome {
    /// Transitioned idle → active; carries the manifest that was cleared.
    Activated(ResumeManifest),
    /// Already active (idempotent no-op).
    AlreadyActive,
}

/// Process-global idle-mode state. `Some(manifest)` ⇒ idle; `None` ⇒ active. One
/// Chord process serves one host, so this is a singleton, like `GPU_EXCLUSIVE`.
pub struct IdleController {
    inner: RwLock<Option<ResumeManifest>>,
    /// Where the manifest is persisted across restarts. `None` ⇒ persistence
    /// disabled (in-memory only) — behaviourally fine, the watchdog still bounds it.
    state_path: Option<PathBuf>,
}

impl Default for IdleController {
    fn default() -> Self {
        Self::new()
    }
}

impl IdleController {
    /// In-memory-only controller (no persistence). Used by unit tests.
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(None),
            state_path: None,
        }
    }

    /// Construct with durable persistence at `state_path`, seeding any persisted
    /// manifest. A missing/corrupt file seeds `None` (active) and never panics.
    pub fn with_state(state_path: Option<PathBuf>) -> Self {
        let seed = state_path.as_deref().and_then(load_persisted);
        if let Some(m) = &seed {
            info!(
                reason = %m.reason,
                entered_at = m.entered_at,
                "idle-mode: reloaded persisted idle state across restart (watchdog will bound it)"
            );
        }
        Self {
            inner: RwLock::new(seed),
            state_path,
        }
    }

    /// From `CHORD_STATE_DIR` (see [`crate::config::admin_idle_state_path`]).
    pub fn from_env() -> Self {
        Self::with_state(crate::config::admin_idle_state_path())
    }

    fn persist_locked(&self, current: &Option<ResumeManifest>) {
        if let Some(path) = self.state_path.as_deref() {
            persist_state(path, current);
        }
    }

    /// Is Chord currently idle?
    pub fn is_idle(&self) -> bool {
        self.inner.read().expect("idle lock poisoned").is_some()
    }

    /// A snapshot of the current manifest (if idle) for the status endpoint.
    pub fn snapshot(&self) -> Option<ResumeManifest> {
        self.inner.read().expect("idle lock poisoned").clone()
    }

    /// Enter idle with `manifest`. Idempotent: if already idle the existing
    /// manifest is preserved (no clobber) and returned as `AlreadyIdle`.
    pub fn enter(&self, manifest: ResumeManifest) -> EnterOutcome {
        let mut guard = self.inner.write().expect("idle lock poisoned");
        match decide_enter(guard.as_ref()) {
            EnterDecision::Enter => {
                *guard = Some(manifest.clone());
                self.persist_locked(&guard);
                EnterOutcome::Entered(manifest)
            }
            EnterDecision::AlreadyIdle => {
                EnterOutcome::AlreadyIdle(guard.clone().expect("already-idle implies Some"))
            }
        }
    }

    /// Leave idle. Idempotent: if already active, `AlreadyActive`.
    pub fn exit(&self) -> ActivateOutcome {
        let mut guard = self.inner.write().expect("idle lock poisoned");
        match decide_activate(guard.as_ref()) {
            ActivateDecision::Restore => {
                let manifest = guard.take().expect("restore implies Some");
                self.persist_locked(&guard);
                ActivateOutcome::Activated(manifest)
            }
            ActivateDecision::AlreadyActive => ActivateOutcome::AlreadyActive,
        }
    }
}

/// The process-global idle-mode controller. Handlers, the lazy-activate hook, and
/// the watchdog reference this; unit tests use isolated [`IdleController::new`]
/// instances so they never touch global state.
pub static IDLE_MODE: once_cell::sync::Lazy<IdleController> =
    once_cell::sync::Lazy::new(IdleController::from_env);

/// Load a persisted manifest from `path`. Missing/unreadable/malformed ⇒ `None`
/// with a warn (never a panic). The file stores `Option<ResumeManifest>`; a stored
/// `null` (last write was an activate) also yields `None`.
fn load_persisted(path: &Path) -> Option<ResumeManifest> {
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Err(e) => {
            warn!(path = %path.display(), error = %e,
                "idle-mode: could not read persisted state (starting active)");
            return None;
        }
    };
    match serde_json::from_str::<Option<ResumeManifest>>(&data) {
        Ok(m) => m,
        Err(e) => {
            warn!(path = %path.display(), error = %e,
                "idle-mode: persisted state is corrupt/unrecognized (starting active)");
            None
        }
    }
}

/// Atomically persist the current state (tempfile + rename). Best-effort: any
/// IO/serde error is logged at warn and swallowed — persistence must never break
/// the idle/activate transition itself.
fn persist_state(path: &Path, state: &Option<ResumeManifest>) {
    let json = match serde_json::to_string(state) {
        Ok(j) => j,
        Err(e) => {
            warn!(error = %e, "idle-mode: failed to serialize state (not persisted)");
            return;
        }
    };
    if let Some(dir) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(dir) {
            warn!(dir = %dir.display(), error = %e,
                "idle-mode: could not create state dir (state not persisted)");
            return;
        }
    }
    let tmp = path.with_extension("json.tmp");
    if let Err(e) = std::fs::write(&tmp, json.as_bytes()) {
        warn!(path = %tmp.display(), error = %e,
            "idle-mode: could not write temp state file (state not persisted)");
        return;
    }
    if let Err(e) = std::fs::rename(&tmp, path) {
        warn!(path = %path.display(), error = %e,
            "idle-mode: could not atomically install state file (state not persisted)");
        let _ = std::fs::remove_file(&tmp);
    }
}

// ── In-flight gauge + drain ───────────────────────────────────────────────────

/// Process-global count of inference requests currently executing. Incremented on
/// entry to the inference handlers (via [`InflightGuard`]) and decremented on drop,
/// so idle-mode can drain before releasing resources.
static INFLIGHT: AtomicUsize = AtomicUsize::new(0);

/// RAII guard: `InflightGuard::enter()` at the top of an inference handler counts
/// one in-flight request; it is decremented when the guard drops (even on panic /
/// early return / `?`), so the gauge can never leak.
#[must_use = "hold the guard for the duration of the request"]
pub struct InflightGuard(());

impl InflightGuard {
    pub fn enter() -> Self {
        INFLIGHT.fetch_add(1, Ordering::SeqCst);
        InflightGuard(())
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        INFLIGHT.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Current number of in-flight inference requests.
pub fn inflight_count() -> usize {
    INFLIGHT.load(Ordering::SeqCst)
}

/// Wait (bounded by `timeout`) for in-flight inference to drain to zero. Returns
/// the number still in flight when it returned (0 = fully drained; >0 = the bound
/// was hit and release proceeds anyway). Polls at 100ms.
pub async fn drain_inflight(timeout: Duration) -> usize {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let n = inflight_count();
        if n == 0 {
            return 0;
        }
        if tokio::time::Instant::now() >= deadline {
            return n;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

// ── Freed-RAM report ──────────────────────────────────────────────────────────

/// The observable result of entering idle, surfaced in the `POST /admin/idle`
/// response so the compiler knows how much headroom it just gained.
#[derive(Debug, Clone, Serialize)]
pub struct IdleReport {
    /// `MemAvailable` (GiB) sampled before release; `null` if `/proc/meminfo`
    /// was unreadable.
    pub mem_available_before_gb: Option<f64>,
    /// `MemAvailable` (GiB) sampled after release; `null` if unreadable.
    pub mem_available_after_gb: Option<f64>,
    /// `after - before`, clamped at 0 (a transient negative from other activity
    /// is reported as 0 freed). `null` if either sample was unreadable.
    pub freed_gb: Option<f64>,
    /// On-demand inference backends stopped.
    pub backends_stopped: usize,
    /// Resident models unloaded from VRAM.
    pub models_unloaded: usize,
    /// Registry records demoted Hot → Warm (VRAM-resident → on-disk).
    pub models_demoted: usize,
    /// In-flight requests still running when release proceeded (0 = clean drain).
    pub inflight_remaining: usize,
    /// If a GPU-exclusive lease is held by ANOTHER holder, its label — reported,
    /// NOT force-released/killed (that lease may be a legitimate external GPU job).
    /// `None` when no foreign lease is held.
    pub foreign_gpu_lock_holder: Option<String>,
}

fn freed_gb(before: Option<f64>, after: Option<f64>) -> Option<f64> {
    match (before, after) {
        (Some(b), Some(a)) => Some((a - b).max(0.0)),
        _ => None,
    }
}

// ── Orchestration (async, best-effort side effects) ──────────────────────────

/// Enter idle: drain, stop providers, unload VRAM, demote resident models, free
/// RAM, record the resume manifest. Idempotent — a second call while already idle
/// returns the existing state with no side effects. Returns the controller outcome
/// plus (only on a real transition) the freed-RAM report.
pub async fn enter_idle(
    state: &Arc<crate::routes::AppState>,
    reason: &str,
) -> (EnterOutcome, Option<IdleReport>) {
    // Idempotency FIRST, so we never re-run release on an already-idle proxy.
    if let Some(existing) = IDLE_MODE.snapshot() {
        return (EnterOutcome::AlreadyIdle(existing), None);
    }

    info!(reason, "idle-mode: entering — draining and releasing host resources");

    // 1. Drain in-flight inference (bounded).
    let inflight_remaining = drain_inflight(Duration::from_secs(drain_secs_from_env())).await;
    if inflight_remaining > 0 {
        warn!(
            inflight_remaining,
            "idle-mode: drain bound hit — releasing anyway (overrunning requests finish on their own)"
        );
    }

    // 2. Snapshot resident models (for the manifest) BEFORE we unload them.
    let resident_models = list_resident_models(state).await;

    // 3. Sample RAM before release.
    let mem_before = crate::config::read_cpu_free_gb();

    // 4. Stop on-demand inference backends (llama-server processes).
    let backends_stopped =
        crate::models::routing::stop_all_on_demand_backends(&state.model_registry).await;

    // 5. Unload every resident model from VRAM (best-effort; skipped if OLLAMA_URL
    //    unset). This is the real "release the GPU + the RAM the models held".
    let models_unloaded = match crate::gpu_exclusive::ollama_base_from_env() {
        Some(base) => crate::gpu_exclusive::evict_resident_models(&state.http_client, &base).await,
        None => {
            info!("idle-mode: OLLAMA_URL unset — skipping VRAM eviction (best-effort)");
            0
        }
    };

    // 6. Demote Hot (VRAM-resident) registry records to Warm (on disk). The blobs
    //    stay local; only the tier bookkeeping changes so a later reconcile/status
    //    reflects that nothing is loaded. Best-effort; a save error is logged.
    let models_demoted = demote_hot_to_warm(&state.model_registry).await;

    // 7. Report (but never force-clear) any foreign GPU-exclusive lease.
    let foreign_gpu_lock_holder =
        crate::gpu_exclusive::GPU_EXCLUSIVE
            .active_holder(now_epoch())
            .map(|r| {
                warn!(
                    holder = %r.holder,
                    "idle-mode: a GPU-exclusive lease is held by another job — reporting, not force-releasing"
                );
                r.holder
            });

    // 8. Sample RAM after release.
    let mem_after = crate::config::read_cpu_free_gb();

    let now = now_epoch();
    let manifest = ResumeManifest {
        reason: reason.to_string(),
        entered_at: now,
        watchdog_deadline: now.saturating_add(watchdog_secs_from_env()),
        resident_models,
        mem_available_before_gb: mem_before.unwrap_or(0.0),
    };
    let outcome = IDLE_MODE.enter(manifest);

    let report = IdleReport {
        mem_available_before_gb: mem_before,
        mem_available_after_gb: mem_after,
        freed_gb: freed_gb(mem_before, mem_after),
        backends_stopped,
        models_unloaded,
        models_demoted,
        inflight_remaining,
        foreign_gpu_lock_holder,
    };
    info!(
        reason,
        backends_stopped,
        models_unloaded,
        models_demoted,
        freed_gb = report.freed_gb.unwrap_or(0.0),
        "idle-mode: entered — host resources released for the compiler"
    );
    (outcome, Some(report))
}

/// Leave idle and resume normal serving. Idempotent. Models reload LAZILY on their
/// next request (Ollama/llama-server cold-load on demand), so there is nothing to
/// force-start here — activate simply clears the gate and logs.
pub async fn activate(state: &Arc<crate::routes::AppState>, reason: &str) -> ActivateOutcome {
    let _ = state; // reserved: a future eager pre-warm could use the registry/client.
    match IDLE_MODE.exit() {
        ActivateOutcome::Activated(m) => {
            info!(
                reason,
                resident_models = m.resident_models.len(),
                "idle-mode: activated — normal serving resumed (models reload on demand)"
            );
            ActivateOutcome::Activated(m)
        }
        ActivateOutcome::AlreadyActive => ActivateOutcome::AlreadyActive,
    }
}

/// Lazy-activate hook for the inference handlers: if Chord is idle, restore before
/// serving so the request succeeds (rather than hitting a stopped backend). A cheap
/// atomic read in the common (active) case — no lock, no await — so it is safe to
/// call at the top of every inference request.
pub async fn lazy_activate(state: &Arc<crate::routes::AppState>) {
    if IDLE_MODE.is_idle() {
        info!("idle-mode: lazy activate — a real request arrived while idle");
        let _ = activate(state, "lazy-on-request").await;
    }
}

/// Best-effort list of the models Ollama currently has resident (`/api/ps`), for
/// the resume manifest. Empty on any error / when `OLLAMA_URL` is unset.
async fn list_resident_models(state: &Arc<crate::routes::AppState>) -> Vec<String> {
    let Some(base) = crate::gpu_exclusive::ollama_base_from_env() else {
        return Vec::new();
    };
    let base = base.trim_end_matches('/');
    let stats = crate::sweep_status::ollama::query_ollama_ps(&state.http_client, base).await;
    if !stats.available {
        return Vec::new();
    }
    stats
        .models
        .into_iter()
        .map(|m| m.name)
        .filter(|n| !n.is_empty())
        .collect()
}

/// Demote every Hot (VRAM-resident) registry record to Warm (on local disk), so the
/// registry reflects that no model is loaded after idle. Returns the count demoted.
/// Persists once at the end (best-effort). Protected models are demoted too — the
/// protection flag guards against *archival/eviction to cold*, not against
/// unloading from VRAM, which is exactly what idle does.
async fn demote_hot_to_warm(
    registry: &Arc<tokio::sync::Mutex<crate::models::registry::ModelRegistry>>,
) -> usize {
    use crate::models::registry::StorageTier;
    let mut reg = registry.lock().await;
    let hot: Vec<String> = reg
        .all_records()
        .filter(|r| r.tier == StorageTier::Hot)
        .map(|r| r.name.clone())
        .collect();
    let mut demoted = 0usize;
    for name in &hot {
        if reg.set_tier(name, StorageTier::Warm) {
            demoted += 1;
        }
    }
    if demoted > 0 {
        if let Err(e) = reg.save() {
            warn!(error = %e, "idle-mode: failed to persist registry after Hot→Warm demote");
        }
    }
    demoted
}

// ── HTTP handlers (control port) ──────────────────────────────────────────────

use std::sync::Arc as StdArc;

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};

use crate::routes::{auth_check, auth_error_response, AppState};

/// Optional body for `POST /admin/idle` and `POST /admin/activate`.
#[derive(Deserialize, Default)]
pub struct IdleBody {
    /// Short label identifying who requested the transition (e.g. `"compiler"`).
    /// Diagnostics only — never a secret. Defaults to `"compiler"` / `"operator"`.
    pub reason: Option<String>,
}

/// `POST /admin/idle` — enter idle mode. Auth-gated. Idempotent (already idle ⇒
/// 200 with the current state, no re-release). Reports freed RAM.
pub async fn admin_idle_enter(
    State(state): State<StdArc<AppState>>,
    headers: HeaderMap,
    body: Option<Json<IdleBody>>,
) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }
    let reason = body
        .and_then(|b| b.0.reason)
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "compiler".to_string());

    let (outcome, report) = enter_idle(&state, &reason).await;
    match outcome {
        EnterOutcome::Entered(m) => {
            let report = report.expect("a real transition always carries a report");
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "status": "idle",
                    "changed": true,
                    "reason": m.reason,
                    "entered_at": crate::gpu_exclusive::iso_utc(m.entered_at),
                    "watchdog_deadline": crate::gpu_exclusive::iso_utc(m.watchdog_deadline),
                    "resident_models": m.resident_models,
                    "freed": report,
                })),
            )
                .into_response()
        }
        EnterOutcome::AlreadyIdle(m) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "idle",
                "changed": false,
                "reason": m.reason,
                "entered_at": crate::gpu_exclusive::iso_utc(m.entered_at),
                "watchdog_deadline": crate::gpu_exclusive::iso_utc(m.watchdog_deadline),
                "resident_models": m.resident_models,
            })),
        )
            .into_response(),
    }
}

/// `POST /admin/activate` — restore full service. Auth-gated. Idempotent
/// (already active ⇒ 200 `changed:false`).
pub async fn admin_activate(
    State(state): State<StdArc<AppState>>,
    headers: HeaderMap,
    body: Option<Json<IdleBody>>,
) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }
    let reason = body
        .and_then(|b| b.0.reason)
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "operator".to_string());

    match activate(&state, &reason).await {
        ActivateOutcome::Activated(m) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "active",
                "changed": true,
                "was_idle_since": crate::gpu_exclusive::iso_utc(m.entered_at),
                "resident_models": m.resident_models,
            })),
        )
            .into_response(),
        ActivateOutcome::AlreadyActive => (
            StatusCode::OK,
            Json(serde_json::json!({ "status": "active", "changed": false })),
        )
            .into_response(),
    }
}

/// `GET /admin/idle` — current idle/active status + resume manifest for observability.
pub async fn admin_idle_status(
    State(state): State<StdArc<AppState>>,
    headers: HeaderMap,
) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }
    let body = match IDLE_MODE.snapshot() {
        Some(m) => serde_json::json!({
            "status": "idle",
            "reason": m.reason,
            "entered_at": crate::gpu_exclusive::iso_utc(m.entered_at),
            "watchdog_deadline": crate::gpu_exclusive::iso_utc(m.watchdog_deadline),
            "watchdog_expired": m.watchdog_expired(now_epoch()),
            "resident_models": m.resident_models,
            "inflight": inflight_count(),
        }),
        None => serde_json::json!({ "status": "active", "inflight": inflight_count() }),
    };
    (StatusCode::OK, Json(body)).into_response()
}

// ── Watchdog ──────────────────────────────────────────────────────────────────

/// Background fail-safe: every `interval`, if idle and the watchdog deadline has
/// passed AND no compiler GPU-exclusive lease is currently held, auto-activate so
/// the proxy is never left silently dead (a crashed/forgotten compiler, or a stale
/// idle state reloaded after a Chord restart). While a lease IS held the deadline
/// is deferred — a legitimately long build keeps Chord idle as long as it holds the
/// GPU. Follows the `idle_stop_sweep` spawn pattern in `main.rs`.
pub async fn watchdog_loop(state: Arc<crate::routes::AppState>, interval: Duration) {
    info!(
        interval_secs = interval.as_secs(),
        "idle-mode watchdog started"
    );
    loop {
        tokio::time::sleep(interval).await;
        let Some(m) = IDLE_MODE.snapshot() else {
            continue;
        };
        let now = now_epoch();
        if !m.watchdog_expired(now) {
            continue;
        }
        if crate::gpu_exclusive::GPU_EXCLUSIVE.active_holder(now).is_some() {
            // A compiler lease is still held — a long build; defer.
            continue;
        }
        warn!(
            reason = %m.reason,
            "idle-mode watchdog: deadline passed with no active GPU lease — auto-activating (fail-safe)"
        );
        let _ = activate(&state, "watchdog-timeout").await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest(reason: &str, entered_at: u64, deadline: u64) -> ResumeManifest {
        ResumeManifest {
            reason: reason.into(),
            entered_at,
            watchdog_deadline: deadline,
            resident_models: vec!["qwen3-coder:30b".into()],
            mem_available_before_gb: 12.0,
        }
    }

    // ── pure decisions ───────────────────────────────────────────────────────

    #[test]
    fn enter_when_active_enters_when_idle_is_noop() {
        assert_eq!(decide_enter(None), EnterDecision::Enter);
        let m = manifest("compiler", 100, 3700);
        assert_eq!(decide_enter(Some(&m)), EnterDecision::AlreadyIdle);
    }

    #[test]
    fn activate_when_idle_restores_when_active_is_noop() {
        let m = manifest("compiler", 100, 3700);
        assert_eq!(decide_activate(Some(&m)), ActivateDecision::Restore);
        assert_eq!(decide_activate(None), ActivateDecision::AlreadyActive);
    }

    #[test]
    fn watchdog_expiry_is_deadline_relative() {
        let m = manifest("compiler", 100, 3700);
        assert!(!m.watchdog_expired(3699));
        assert!(m.watchdog_expired(3700)); // exactly at the deadline ⇒ expired
        assert!(m.watchdog_expired(9999));
    }

    // ── controller transitions (isolated instance, no globals) ───────────────

    #[test]
    fn enter_then_exit_cycle() {
        let ctl = IdleController::new();
        assert!(!ctl.is_idle());
        assert!(ctl.snapshot().is_none());

        let m = manifest("compiler", 100, 3700);
        match ctl.enter(m.clone()) {
            EnterOutcome::Entered(got) => assert_eq!(got, m),
            other => panic!("expected Entered, got {other:?}"),
        }
        assert!(ctl.is_idle());
        assert_eq!(ctl.snapshot().unwrap(), m);

        match ctl.exit() {
            ActivateOutcome::Activated(got) => assert_eq!(got, m),
            other => panic!("expected Activated, got {other:?}"),
        }
        assert!(!ctl.is_idle());
    }

    #[test]
    fn enter_is_idempotent_and_does_not_clobber() {
        let ctl = IdleController::new();
        let first = manifest("compiler", 100, 3700);
        let second = manifest("someone-else", 999, 9999);
        assert!(matches!(ctl.enter(first.clone()), EnterOutcome::Entered(_)));

        // A second enter must NOT overwrite the original manifest.
        match ctl.enter(second) {
            EnterOutcome::AlreadyIdle(got) => assert_eq!(got, first),
            other => panic!("expected AlreadyIdle, got {other:?}"),
        }
        assert_eq!(ctl.snapshot().unwrap(), first);
    }

    #[test]
    fn exit_is_idempotent() {
        let ctl = IdleController::new();
        assert!(matches!(ctl.exit(), ActivateOutcome::AlreadyActive));
        ctl.enter(manifest("compiler", 1, 2));
        assert!(matches!(ctl.exit(), ActivateOutcome::Activated(_)));
        assert!(matches!(ctl.exit(), ActivateOutcome::AlreadyActive));
    }

    // ── freed-RAM arithmetic ─────────────────────────────────────────────────

    #[test]
    fn freed_gb_clamps_and_handles_missing() {
        assert_eq!(freed_gb(Some(10.0), Some(25.0)), Some(15.0));
        // A transient negative (other activity ate RAM) reports 0, not negative.
        assert_eq!(freed_gb(Some(25.0), Some(24.0)), Some(0.0));
        assert_eq!(freed_gb(None, Some(25.0)), None);
        assert_eq!(freed_gb(Some(10.0), None), None);
    }

    // ── in-flight gauge ──────────────────────────────────────────────────────

    #[test]
    fn inflight_guard_increments_and_decrements() {
        let base = inflight_count();
        {
            let _g1 = InflightGuard::enter();
            assert_eq!(inflight_count(), base + 1);
            {
                let _g2 = InflightGuard::enter();
                assert_eq!(inflight_count(), base + 2);
            }
            assert_eq!(inflight_count(), base + 1);
        }
        assert_eq!(inflight_count(), base);
    }

    // ── durable persistence (mirrors gpu_exclusive) ──────────────────────────

    #[test]
    fn with_state_reloads_idle_across_restart() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("admin_idle_state.json");

        let ctl = IdleController::with_state(Some(path.clone()));
        let m = manifest("compiler", 100, 3700);
        assert!(matches!(ctl.enter(m.clone()), EnterOutcome::Entered(_)));
        assert!(path.exists(), "state file should be written on enter");

        // Simulate a Chord restart: a fresh controller reloads the same file.
        let restarted = IdleController::with_state(Some(path.clone()));
        assert!(restarted.is_idle());
        assert_eq!(restarted.snapshot().unwrap(), m);
    }

    #[test]
    fn exit_clears_persisted_state() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("admin_idle_state.json");

        let ctl = IdleController::with_state(Some(path.clone()));
        ctl.enter(manifest("compiler", 1, 2));
        assert!(matches!(ctl.exit(), ActivateOutcome::Activated(_)));

        // A restart after activate must see no idle state.
        let restarted = IdleController::with_state(Some(path.clone()));
        assert!(!restarted.is_idle());
    }

    #[test]
    fn with_state_corrupt_file_starts_active_no_panic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("admin_idle_state.json");
        std::fs::write(&path, b"{ not valid json ").unwrap();

        let ctl = IdleController::with_state(Some(path));
        assert!(!ctl.is_idle());
        // Still fully functional after ignoring the corrupt file.
        assert!(matches!(
            ctl.enter(manifest("compiler", 1, 2)),
            EnterOutcome::Entered(_)
        ));
    }

    #[test]
    fn with_state_missing_file_starts_active() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nope.json");
        let ctl = IdleController::with_state(Some(path));
        assert!(!ctl.is_idle());
    }

    #[test]
    fn no_state_path_writes_nothing_and_still_works() {
        let ctl = IdleController::new(); // in-memory only
        assert!(matches!(
            ctl.enter(manifest("compiler", 1, 2)),
            EnterOutcome::Entered(_)
        ));
        assert!(ctl.is_idle());
        assert!(matches!(ctl.exit(), ActivateOutcome::Activated(_)));
    }

    // ── env parsing ──────────────────────────────────────────────────────────

    #[test]
    fn positive_env_falls_back_on_junk() {
        std::env::set_var("CHORD_IDLE_TEST_KEY", "not-a-number");
        assert_eq!(parse_positive_env("CHORD_IDLE_TEST_KEY", 42), 42);
        std::env::set_var("CHORD_IDLE_TEST_KEY", "0");
        assert_eq!(parse_positive_env("CHORD_IDLE_TEST_KEY", 42), 42);
        std::env::set_var("CHORD_IDLE_TEST_KEY", "900");
        assert_eq!(parse_positive_env("CHORD_IDLE_TEST_KEY", 42), 900);
        std::env::remove_var("CHORD_IDLE_TEST_KEY");
    }
}
