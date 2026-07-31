//! GPU-exclusive coordination — a "service mode" that hands the single host GPU
//! to an external, GPU-heavy job (the Terminus intake benchmarking harness on
//! the GPU inference host) WITHOUT ever taking Chord down.
//!
//! ## Why this exists
//! The GPU inference host is a dedicated, single-GPU (gfx1151 APU, no multi-tenancy) host. Chord
//! is the always-on backbone proxy for the whole fleet. The benchmarking harness
//! (`intake_coder_sweep` / `intake_assistant_sweep`, in the Terminus repo) needs
//! EXCLUSIVE GPU access while it profiles models — two inference jobs stacked in
//! the shared 96GB VRAM ceiling produce false "wedge" timeouts (see the
//! `gfx1151-vram-contention` memory).
//!
//! The harness's `intake::gpu_authority` module used to get that exclusivity by
//! literally `systemctl stop chord.service` — which left Chord, the backbone,
//! `inactive (dead)` for the ENTIRE multi-day sweep (discovered after 3 days
//! down). This module is the fix: Chord stays up, keeps its HTTP listener,
//! health checks, routing decisions, and read-only DB tools serving normally,
//! and only GATES the GPU/model-inference paths for the duration of the lock.
//!
//! ## Model
//! - The lock is a single, PROCESS-GLOBAL record ([`GPU_EXCLUSIVE`]) — there is
//!   one physical GPU per Chord process, so this is a hardware resource, not a
//!   per-request/per-connection thing.
//! - [`GpuExclusive::acquire`] grants the lock to a `holder` label. A grant from
//!   FREE (or from an expired/abandoned lock) is a NEW grant; a re-acquire by the
//!   SAME holder is a heartbeat REFRESH (bumps `last_heartbeat`, no re-eviction).
//!   A live lock held by a DIFFERENT holder BLOCKS (409) — Chord never silently
//!   lets two jobs race the GPU.
//! - [`GpuExclusive::active_holder`] is the GATE the inference handlers consult:
//!   `Some(record)` ⇒ the request path returns a structured 503
//!   `gpu_exclusively_held` INSTEAD of loading a model / dispatching inference.
//! - **TTL / heartbeat safety.** The harness runs for DAYS but is a REMOTE
//!   process — Chord cannot check a remote PID the way `gpu_authority`'s own
//!   `LockState` self-heals a local crashed PID. So this is TIME-based: a lock
//!   whose `last_heartbeat` is older than the TTL ([`DEFAULT_TTL_SECS`], override
//!   `CHORD_GPU_EXCLUSIVE_TTL_SECS`) is treated as ABANDONED — [`active_holder`]
//!   stops gating and a fresh `acquire` (by anyone) is granted. The harness MUST
//!   therefore periodically re-`acquire` (heartbeat) at an interval well under
//!   the TTL to hold the GPU across a long sweep; if it crashes, the missed
//!   heartbeats let the lock expire and Chord auto-resumes serving. This mirrors
//!   `gpu_authority`'s "a crashed sweep must never wedge the GPU forever"
//!   philosophy, ported from local-PID-liveness to remote-safe wall-clock TTL.
//!
//! The pure decision logic ([`decide_acquire`], [`decide_release`],
//! [`LockRecord::is_expired`]) is separated from the `RwLock`/clock/HTTP so it is
//! exhaustively unit-testable with no global state, no sleeping, and no network.

use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

/// Default TTL (seconds) after which a lock with no fresh heartbeat is treated
/// as abandoned. 10 minutes: long enough to survive a transient harness stall /
/// network blip, short enough that a genuinely crashed sweep hands the GPU back
/// to Chord promptly. The harness heartbeats far more often than this.
pub const DEFAULT_TTL_SECS: u64 = 600;

/// Resolve the abandoned-lock TTL from `CHORD_GPU_EXCLUSIVE_TTL_SECS` (seconds);
/// a missing/blank/zero/unparseable value falls back to [`DEFAULT_TTL_SECS`].
pub fn ttl_secs_from_env() -> u64 {
    std::env::var("CHORD_GPU_EXCLUSIVE_TTL_SECS")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_TTL_SECS)
}

/// Current wall-clock epoch seconds. Thin wrapper (not pure) so the decision
/// functions that take `now` stay pure/testable.
pub fn now_epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Render an epoch-seconds timestamp as an RFC3339/ISO-8601 UTC string for the
/// `since` field of API/gate bodies (so a stale lock is diagnosable at a glance).
/// Falls back to the raw epoch string if the timestamp is somehow out of range.
pub fn iso_utc(epoch_secs: u64) -> String {
    chrono::DateTime::<chrono::Utc>::from_timestamp(epoch_secs as i64, 0)
        .map(|dt| dt.to_rfc3339())
        .unwrap_or_else(|| epoch_secs.to_string())
}

/// The GPU-exclusive lock, in memory. `holder` is a short label supplied by the
/// acquirer (e.g. `intake_coder_sweep`); `acquired_at` is the first-grant time
/// (stable across heartbeats, for "since" reporting); `last_heartbeat` is the
/// most recent (re)acquire, which drives TTL expiry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LockRecord {
    pub holder: String,
    pub acquired_at: u64,
    pub last_heartbeat: u64,
}

impl LockRecord {
    /// Is this lock abandoned — no heartbeat within `ttl` seconds of `now`?
    /// `saturating_sub` so a clock that briefly goes backwards can't underflow
    /// into a spuriously-huge age (it just reads as age 0 = not expired).
    pub fn is_expired(&self, now: u64, ttl: u64) -> bool {
        now.saturating_sub(self.last_heartbeat) > ttl
    }
}

/// Pure decision for an [`GpuExclusive::acquire`] by `holder` at `now`, given the
/// current lock (if any) and the `ttl`. No IO, no clock, no lock — the whole
/// policy in one exhaustively-testable place.
#[derive(Debug, PartialEq, Eq)]
pub enum AcquireDecision {
    /// Free, or the existing lock is expired/abandoned ⇒ grant fresh (the caller
    /// should evict any resident model on this transition).
    GrantNew,
    /// Same holder re-acquiring a live lock ⇒ heartbeat refresh only (NO
    /// re-eviction — the model set is already whatever the holder wants).
    Refresh,
    /// A DIFFERENT holder owns a live lock ⇒ blocked. Carries the current
    /// holder + its first-acquire time for the 409 body.
    HeldBy { holder: String, since: u64 },
}

pub fn decide_acquire(
    existing: Option<&LockRecord>,
    holder: &str,
    now: u64,
    ttl: u64,
) -> AcquireDecision {
    match existing {
        None => AcquireDecision::GrantNew,
        Some(r) if r.is_expired(now, ttl) => AcquireDecision::GrantNew,
        Some(r) if r.holder == holder => AcquireDecision::Refresh,
        Some(r) => AcquireDecision::HeldBy {
            holder: r.holder.clone(),
            since: r.acquired_at,
        },
    }
}

/// Pure decision for a [`GpuExclusive::release`] by `holder`, given the current
/// lock (if any). A release NEVER clears someone else's lock.
#[derive(Debug, PartialEq, Eq)]
pub enum ReleaseDecision {
    /// `holder` owns the lock ⇒ clear it.
    Release,
    /// No lock at all ⇒ idempotent no-op success.
    NotHeld,
    /// A DIFFERENT holder owns it ⇒ refuse. Carries the real holder.
    Mismatch { holder: String },
}

pub fn decide_release(existing: Option<&LockRecord>, holder: &str) -> ReleaseDecision {
    match existing {
        None => ReleaseDecision::NotHeld,
        Some(r) if r.holder == holder => ReleaseDecision::Release,
        Some(r) => ReleaseDecision::Mismatch {
            holder: r.holder.clone(),
        },
    }
}

/// Outcome of applying an acquire against the live state.
#[derive(Debug, PartialEq, Eq)]
pub enum AcquireOutcome {
    /// Granted. `new_grant` is true only on a FREE→HELD (or expired-takeover)
    /// transition — the caller evicts resident models only then, not on a
    /// heartbeat refresh. `record` is the resulting live lock.
    Granted { record: LockRecord, new_grant: bool },
    /// Blocked by a live lock held by someone else.
    Blocked { record: LockRecord },
}

/// Outcome of applying a release against the live state.
#[derive(Debug, PartialEq, Eq)]
pub enum ReleaseOutcome {
    /// Cleared (or was already free).
    Released,
    /// Refused — a different holder owns the lock.
    Mismatch { record: LockRecord },
}

// ── RESIL-01: durable lease persistence across a Chord restart ────────────────
//
// The lock above is otherwise IN-MEMORY only, so a Chord process restart mid-sweep
// dropped the lease — the sweep that legitimately owns the GPU appeared un-held
// ("CHORD LOCK GAP DETECTED" on the harness side) and a competing job could slip
// in. When a state path is configured (`CHORD_STATE_DIR`), every mutation writes
// the current `Option<LockRecord>` and startup reloads it (respecting TTL), so a
// restarted Chord keeps honoring a live lease. Persistence is best-effort: a
// missing/corrupt/unwritable file NEVER panics Chord — it degrades to the prior
// in-memory-only behavior and logs at warn. Path unset ⇒ persistence is disabled
// entirely (no file writes), preserving the exact current behavior.

/// Load a persisted lease from `path`. Returns the stored `Option<LockRecord>`
/// (which may itself be `None` when the last write was a release). A missing,
/// unreadable, or malformed file yields `None` with a warn — never a panic.
fn load_persisted(path: &Path) -> Option<LockRecord> {
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Err(e) => {
            warn!(path = %path.display(), error = %e,
                "gpu-exclusive: could not read persisted lease (starting unheld)");
            return None;
        }
    };
    match serde_json::from_str::<Option<LockRecord>>(&data) {
        Ok(rec) => rec,
        Err(e) => {
            warn!(path = %path.display(), error = %e,
                "gpu-exclusive: persisted lease is corrupt/unrecognized (starting unheld)");
            None
        }
    }
}

/// Atomically persist the current lock state to `path` (tempfile + rename), so a
/// crash mid-write can never leave a torn file. Best-effort: any IO/serde error
/// is logged at warn and swallowed — persistence must never break acquire/release.
fn persist_state(path: &Path, rec: &Option<LockRecord>) {
    let json = match serde_json::to_string(rec) {
        Ok(j) => j,
        Err(e) => {
            warn!(error = %e, "gpu-exclusive: failed to serialize lease (state not persisted)");
            return;
        }
    };
    if let Some(dir) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(dir) {
            warn!(dir = %dir.display(), error = %e,
                "gpu-exclusive: could not create state dir (lease not persisted)");
            return;
        }
    }
    let tmp = path.with_extension("json.tmp");
    if let Err(e) = std::fs::write(&tmp, json.as_bytes()) {
        warn!(path = %tmp.display(), error = %e,
            "gpu-exclusive: could not write temp lease file (lease not persisted)");
        return;
    }
    if let Err(e) = std::fs::rename(&tmp, path) {
        warn!(path = %path.display(), error = %e,
            "gpu-exclusive: could not atomically install lease file (lease not persisted)");
        let _ = std::fs::remove_file(&tmp);
    }
}

/// The process-global GPU-exclusive lock. One physical GPU ⇒ one lock.
pub struct GpuExclusive {
    inner: RwLock<Option<LockRecord>>,
    ttl: u64,
    /// Where the lease is persisted across restarts. `None` ⇒ persistence
    /// disabled (in-memory only) — the original behavior.
    state_path: Option<PathBuf>,
}

impl GpuExclusive {
    pub fn new(ttl: u64) -> Self {
        Self {
            inner: RwLock::new(None),
            ttl,
            state_path: None,
        }
    }

    /// Construct with durable persistence at `state_path`. On construction the
    /// persisted lease (if any) is reloaded and seeded into the in-memory lock,
    /// UNLESS it is already expired at `now` (an abandoned lease from before the
    /// restart must not relock the GPU). A missing/corrupt file seeds nothing.
    pub fn with_state(ttl: u64, state_path: Option<PathBuf>, now: u64) -> Self {
        let seed = match state_path.as_deref() {
            Some(p) => match load_persisted(p) {
                Some(rec) if rec.is_expired(now, ttl) => {
                    info!(holder = %rec.holder,
                        "gpu-exclusive: persisted lease is expired — starting unheld");
                    None
                }
                Some(rec) => {
                    info!(holder = %rec.holder, acquired_at = rec.acquired_at,
                        "gpu-exclusive: reloaded live lease across restart");
                    Some(rec)
                }
                None => None,
            },
            None => None,
        };
        Self {
            inner: RwLock::new(seed),
            ttl,
            state_path,
        }
    }

    pub fn from_env() -> Self {
        Self::with_state(
            ttl_secs_from_env(),
            crate::config::gpu_exclusive_state_path(),
            now_epoch(),
        )
    }

    /// Persist the current in-memory state (called while holding the write lock,
    /// after every mutation). No-op when persistence is disabled.
    fn persist_locked(&self, current: &Option<LockRecord>) {
        if let Some(path) = self.state_path.as_deref() {
            persist_state(path, current);
        }
    }

    pub fn ttl(&self) -> u64 {
        self.ttl
    }

    /// The GATE the inference handlers consult. `Some(record)` ⇒ the GPU is
    /// exclusively held by a LIVE (non-expired) lock; the caller must return the
    /// structured 503 instead of touching a model. `None` ⇒ free, or the lock is
    /// abandoned (expired) and should no longer gate anything.
    pub fn active_holder(&self, now: u64) -> Option<LockRecord> {
        let guard = self.inner.read().expect("gpu-exclusive lock poisoned");
        match &*guard {
            Some(r) if !r.is_expired(now, self.ttl) => Some(r.clone()),
            _ => None,
        }
    }

    /// Apply an acquire by `holder` at `now`. See [`AcquireOutcome`].
    pub fn acquire(&self, holder: &str, now: u64) -> AcquireOutcome {
        let mut guard = self.inner.write().expect("gpu-exclusive lock poisoned");
        match decide_acquire(guard.as_ref(), holder, now, self.ttl) {
            AcquireDecision::GrantNew => {
                let record = LockRecord {
                    holder: holder.to_string(),
                    acquired_at: now,
                    last_heartbeat: now,
                };
                *guard = Some(record.clone());
                self.persist_locked(&guard);
                AcquireOutcome::Granted {
                    record,
                    new_grant: true,
                }
            }
            AcquireDecision::Refresh => {
                // Preserve the original acquired_at; only bump the heartbeat.
                let record = {
                    let r = guard.as_mut().expect("refresh implies a live lock");
                    r.last_heartbeat = now;
                    r.clone()
                };
                self.persist_locked(&guard);
                AcquireOutcome::Granted {
                    record,
                    new_grant: false,
                }
            }
            AcquireDecision::HeldBy { .. } => {
                let record = guard.as_ref().expect("held implies a live lock").clone();
                AcquireOutcome::Blocked { record }
            }
        }
    }

    /// Apply a release by `holder`. See [`ReleaseOutcome`].
    pub fn release(&self, holder: &str) -> ReleaseOutcome {
        let mut guard = self.inner.write().expect("gpu-exclusive lock poisoned");
        match decide_release(guard.as_ref(), holder) {
            ReleaseDecision::Release | ReleaseDecision::NotHeld => {
                *guard = None;
                self.persist_locked(&guard);
                ReleaseOutcome::Released
            }
            ReleaseDecision::Mismatch { .. } => {
                let record = guard
                    .as_ref()
                    .expect("mismatch implies a live lock")
                    .clone();
                ReleaseOutcome::Mismatch { record }
            }
        }
    }

    /// A point-in-time snapshot for the status endpoint: the current lock (if
    /// any) plus whether it is expired/abandoned right now.
    pub fn snapshot(&self, now: u64) -> Option<(LockRecord, bool)> {
        self.inner
            .read()
            .expect("gpu-exclusive lock poisoned")
            .as_ref()
            .map(|r| (r.clone(), r.is_expired(now, self.ttl)))
    }
}

/// The process-global lock instance. Handlers and the inference gate reference
/// this; unit tests exercise isolated [`GpuExclusive::new`] instances instead so
/// they never touch (or race on) global state.
pub static GPU_EXCLUSIVE: once_cell::sync::Lazy<GpuExclusive> =
    once_cell::sync::Lazy::new(GpuExclusive::from_env);

/// Ollama base URL to evict resident models against, from `OLLAMA_URL` (the same
/// env `chord.service` already requires — it points at the SAME local Ollama the
/// intake harness contends with). `None` ⇒ eviction is skipped (best-effort).
pub fn ollama_base_from_env() -> Option<String> {
    std::env::var("OLLAMA_URL")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Phase 1 (lumina resident mode): is `name` in the keep-resident `exempt` set?
/// Pure/testable — exact-match against the operator's `MODEL_KEEP_RESIDENT` names
/// (same names Ollama reports in `/api/ps` and that the warm path pins). An empty
/// `exempt` (the default) never matches, so eviction behaves exactly as before.
pub fn is_keep_resident(name: &str, exempt: &[String]) -> bool {
    exempt.iter().any(|e| e == name)
}

/// Phase 1: split a list of resident model names into `(to_evict, kept)` given the
/// keep-resident `exempt` set. Pure — the eviction policy in one testable place,
/// no network. `kept` is the keep-resident models that were resident and are being
/// left loaded (for `[keep-resident]` logging).
pub fn partition_evictable(resident: &[String], exempt: &[String]) -> (Vec<String>, Vec<String>) {
    let mut to_evict = Vec::new();
    let mut kept = Vec::new();
    for name in resident {
        if name.is_empty() {
            continue;
        }
        if is_keep_resident(name, exempt) {
            kept.push(name.clone());
        } else {
            to_evict.push(name.clone());
        }
    }
    (to_evict, kept)
}

/// Best-effort: unload the models Ollama currently has resident so the GPU is clear
/// for the exclusive holder — EXCEPT any name in `exempt` (the Phase 1 keep-resident
/// working-set), which is left loaded so the assistant stays hot through a sweep.
/// Pass `&[]` to unload everything (idle-mode's whole-GPU release keeps that shape).
/// Non-fatal by construction — a missing `OLLAMA_URL`, an unreachable Ollama, or
/// nothing loaded all yield `0` with a log line, never an error. Reuses the harness's
/// own `/api/ps` poll shape. Returns the count actually unloaded.
pub async fn evict_resident_models(
    client: &reqwest::Client,
    ollama_base: &str,
    exempt: &[String],
) -> usize {
    let base = ollama_base.trim_end_matches('/');
    let stats = crate::sweep_status::ollama::query_ollama_ps(client, base).await;
    if !stats.available {
        info!("gpu-exclusive: Ollama /api/ps unavailable — nothing to evict (best-effort)");
        return 0;
    }
    let resident: Vec<String> = stats.models.into_iter().map(|m| m.name).collect();
    let (to_evict, kept) = partition_evictable(&resident, exempt);
    for name in &kept {
        info!(model = %name, "[keep-resident] gpu-exclusive: exempting model from eviction (staying VRAM-resident)");
    }
    let mut unloaded = 0usize;
    for name in to_evict {
        // Ollama unloads a resident model when handed keep_alive:0.
        let url = format!("{base}/api/generate");
        let body = serde_json::json!({ "model": name, "keep_alive": 0 });
        match client
            .post(&url)
            .json(&body)
            .timeout(Duration::from_secs(30))
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => {
                info!(model = %name, "gpu-exclusive: evicted resident model");
                unloaded += 1;
            }
            Ok(r) => warn!(
                model = %name,
                status = r.status().as_u16(),
                "gpu-exclusive: unload request rejected (best-effort, continuing)"
            ),
            Err(e) => warn!(
                model = %name,
                error = %e,
                "gpu-exclusive: unload request failed (best-effort, continuing)"
            ),
        }
    }
    if unloaded > 0 {
        info!(
            count = unloaded,
            "gpu-exclusive: resident models evicted for exclusive holder"
        );
    }
    unloaded
}

/// Phase 1: the request body that PINS a model VRAM-resident — `keep_alive:-1`
/// (indefinite) with an empty prompt, so Ollama loads the model (if cold) and sets
/// its keep_alive so a normal request can't let it drift to the 2h default and out
/// of VRAM. Pure/testable (asserts the `-1` sentinel). The `/api/generate` empty-
/// prompt "just load / set keep_alive" shape is the mirror of the `keep_alive:0`
/// unload used above.
pub fn keep_resident_warm_body(model: &str) -> serde_json::Value {
    serde_json::json!({ "model": model, "prompt": "", "keep_alive": -1 })
}

/// Phase 1.1: the request body that PINS an **embedding** model VRAM-resident via
/// Ollama's `/api/embeddings` endpoint — `keep_alive:-1` (indefinite) with an
/// empty prompt. Embedding models (e.g. `qwen3-embedding:0.6b`) reject
/// `/api/generate` with a 400 "does not support generate" and so can never be
/// pinned by [`keep_resident_warm_body`]; this is the fallback that durably pins
/// them. Pure/testable (asserts the `-1` sentinel).
pub fn keep_resident_embed_warm_body(model: &str) -> serde_json::Value {
    serde_json::json!({ "model": model, "prompt": "", "keep_alive": -1 })
}

/// Phase 1.1: heuristic for "this model name is an embedding model" — a cheap
/// pre-check so we can go straight to the `/api/embeddings` warm path instead of
/// wasting a doomed `/api/generate` round trip. The `/api/generate` 400
/// "does not support generate" response is still handled as a second signal (see
/// [`warm_keep_resident_models`]) for embedding models whose name doesn't match.
pub fn is_embedding_model(name: &str) -> bool {
    name.to_ascii_lowercase().contains("embed")
}

/// Phase 1.1: pin one embedding model VRAM-resident via `/api/embeddings` with
/// `keep_alive:-1`. Best-effort / fail-SOFT: returns `true` only on a 2xx,
/// logging and returning `false` otherwise (never errors/panics).
async fn warm_embed_keep_resident(client: &reqwest::Client, base: &str, model: &str) -> bool {
    let url = format!("{base}/api/embeddings");
    let body = keep_resident_embed_warm_body(model);
    match client
        .post(&url)
        .json(&body)
        .timeout(Duration::from_secs(180))
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => {
            info!(model = %model, "[keep-resident] pinned EMBEDDING model VRAM-resident via /api/embeddings (keep_alive=-1)");
            true
        }
        Ok(r) => {
            warn!(
                model = %model,
                status = r.status().as_u16(),
                "[keep-resident] embed warm request rejected (best-effort, continuing)"
            );
            false
        }
        Err(e) => {
            warn!(
                model = %model,
                error = %e,
                "[keep-resident] embed warm request failed (best-effort, continuing)"
            );
            false
        }
    }
}

/// Phase 1: pin each `models` name VRAM-resident by issuing a tiny `keep_alive:-1`
/// warm request. Best-effort / fail-SOFT by construction: a warm failure for one
/// model logs and moves on — it NEVER errors or panics (so a startup pre-warm or a
/// periodic re-warm can't take Chord down). Returns the count successfully warmed.
/// An empty `models` (the default keep-resident set) is a no-op.
///
/// Phase 1.1: embedding models (e.g. `qwen3-embedding:0.6b`) do NOT support
/// `/api/generate` — Ollama rejects the warm with a 400 "does not support
/// generate", so they would never get durably pinned. For those we FALL BACK to
/// a `/api/embeddings` warm (also `keep_alive:-1`). The fallback fires either
/// when the name looks like an embedding model ([`is_embedding_model`]) — a
/// cheap pre-check that skips the doomed `/api/generate` — or when `/api/generate`
/// itself returns a 400 (the "does not support generate"-class signal).
pub async fn warm_keep_resident_models(
    client: &reqwest::Client,
    ollama_base: &str,
    models: &[String],
) -> usize {
    let base = ollama_base.trim_end_matches('/');
    let url = format!("{base}/api/generate");
    let mut warmed = 0usize;
    for model in models {
        if model.is_empty() {
            continue;
        }
        // Phase 1.1: known embedding models skip the doomed /api/generate and go
        // straight to the /api/embeddings warm path.
        if is_embedding_model(model) {
            if warm_embed_keep_resident(client, base, model).await {
                warmed += 1;
            }
            continue;
        }
        let body = keep_resident_warm_body(model);
        match client
            .post(&url)
            .json(&body)
            // A cold model can take a while to load into VRAM; give the warm a
            // generous bound but still cap it so a wedged Ollama never hangs the
            // re-warm timer forever.
            .timeout(Duration::from_secs(180))
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => {
                info!(model = %model, "[keep-resident] pinned model VRAM-resident (keep_alive=-1)");
                warmed += 1;
            }
            // Phase 1.1: a 400 from /api/generate is the "does not support
            // generate"-class signal for an embedding model whose name didn't
            // match the heuristic — fall back to the /api/embeddings warm.
            Ok(r) if r.status() == reqwest::StatusCode::BAD_REQUEST => {
                info!(
                    model = %model,
                    "[keep-resident] /api/generate rejected (400) — falling back to /api/embeddings warm"
                );
                if warm_embed_keep_resident(client, base, model).await {
                    warmed += 1;
                }
            }
            Ok(r) => warn!(
                model = %model,
                status = r.status().as_u16(),
                "[keep-resident] warm request rejected (best-effort, continuing)"
            ),
            Err(e) => warn!(
                model = %model,
                error = %e,
                "[keep-resident] warm request failed (best-effort, continuing)"
            ),
        }
    }
    if warmed > 0 {
        info!(count = warmed, "[keep-resident] re-warm pass complete");
    }
    warmed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(holder: &str, acquired_at: u64, last_heartbeat: u64) -> LockRecord {
        LockRecord {
            holder: holder.into(),
            acquired_at,
            last_heartbeat,
        }
    }

    // ── is_expired ───────────────────────────────────────────────────────────

    #[test]
    fn not_expired_within_ttl() {
        let r = rec("sweep", 100, 100);
        assert!(!r.is_expired(100, 600));
        assert!(!r.is_expired(700, 600)); // exactly ttl old ⇒ not yet expired
    }

    #[test]
    fn expired_past_ttl() {
        let r = rec("sweep", 100, 100);
        assert!(r.is_expired(701, 600)); // 601s since heartbeat > 600 ttl
    }

    #[test]
    fn clock_going_backwards_reads_as_not_expired() {
        let r = rec("sweep", 100, 500);
        // now < last_heartbeat — saturating_sub ⇒ age 0, never spuriously expired.
        assert!(!r.is_expired(400, 600));
    }

    // ── decide_acquire ───────────────────────────────────────────────────────

    #[test]
    fn acquire_when_free_grants_new() {
        assert_eq!(
            decide_acquire(None, "sweep", 10, 600),
            AcquireDecision::GrantNew
        );
    }

    #[test]
    fn acquire_same_holder_live_lock_is_refresh() {
        let r = rec("sweep", 100, 100);
        assert_eq!(
            decide_acquire(Some(&r), "sweep", 200, 600),
            AcquireDecision::Refresh
        );
    }

    #[test]
    fn acquire_different_holder_live_lock_blocks() {
        let r = rec("sweep", 100, 150);
        assert_eq!(
            decide_acquire(Some(&r), "other", 200, 600),
            AcquireDecision::HeldBy {
                holder: "sweep".into(),
                since: 100
            }
        );
    }

    #[test]
    fn acquire_expired_lock_grants_new_even_to_different_holder() {
        // A crashed holder (no heartbeat past ttl) must never wedge the GPU:
        // anyone can take over an abandoned lock.
        let r = rec("sweep", 100, 100);
        assert_eq!(
            decide_acquire(Some(&r), "other", 1000, 600),
            AcquireDecision::GrantNew
        );
    }

    // ── decide_release ───────────────────────────────────────────────────────

    #[test]
    fn release_no_lock_is_notheld() {
        assert_eq!(decide_release(None, "sweep"), ReleaseDecision::NotHeld);
    }

    #[test]
    fn release_own_lock_releases() {
        let r = rec("sweep", 100, 100);
        assert_eq!(decide_release(Some(&r), "sweep"), ReleaseDecision::Release);
    }

    #[test]
    fn release_others_lock_is_mismatch() {
        let r = rec("sweep", 100, 100);
        assert_eq!(
            decide_release(Some(&r), "other"),
            ReleaseDecision::Mismatch {
                holder: "sweep".into()
            }
        );
    }

    // ── GpuExclusive (stateful, isolated instance) ───────────────────────────

    #[test]
    fn acquire_release_cycle() {
        let gpu = GpuExclusive::new(600);
        assert!(gpu.active_holder(0).is_none());

        match gpu.acquire("sweep", 10) {
            AcquireOutcome::Granted { new_grant, record } => {
                assert!(new_grant);
                assert_eq!(record.acquired_at, 10);
            }
            other => panic!("expected new grant, got {other:?}"),
        }

        // Now gated.
        let held = gpu.active_holder(20).expect("should be held");
        assert_eq!(held.holder, "sweep");

        // Same-holder re-acquire = heartbeat refresh (not a new grant), preserves
        // acquired_at, bumps last_heartbeat.
        match gpu.acquire("sweep", 300) {
            AcquireOutcome::Granted { new_grant, record } => {
                assert!(!new_grant);
                assert_eq!(record.acquired_at, 10);
                assert_eq!(record.last_heartbeat, 300);
            }
            other => panic!("expected refresh, got {other:?}"),
        }

        // Release clears the gate.
        assert_eq!(gpu.release("sweep"), ReleaseOutcome::Released);
        assert!(gpu.active_holder(310).is_none());
    }

    #[test]
    fn heartbeat_holds_across_would_be_ttl_but_silence_expires() {
        let gpu = GpuExclusive::new(600);
        gpu.acquire("sweep", 0);
        // Heartbeat at 500 keeps it live well past the original 600 window.
        gpu.acquire("sweep", 500);
        assert!(gpu.active_holder(1000).is_some()); // 500s since last heartbeat
                                                    // But no further heartbeat ⇒ expires 600s after the last one (500).
        assert!(gpu.active_holder(1101).is_none());
    }

    #[test]
    fn different_holder_blocked_while_live() {
        let gpu = GpuExclusive::new(600);
        gpu.acquire("sweep", 0);
        match gpu.acquire("other", 10) {
            AcquireOutcome::Blocked { record } => assert_eq!(record.holder, "sweep"),
            other => panic!("expected blocked, got {other:?}"),
        }
        // Original holder still gates.
        assert_eq!(gpu.active_holder(10).unwrap().holder, "sweep");
    }

    #[test]
    fn expired_lock_no_longer_gates_and_is_takeable() {
        let gpu = GpuExclusive::new(600);
        gpu.acquire("sweep", 0);
        assert!(gpu.active_holder(601).is_none()); // abandoned
                                                   // A new holder takes over cleanly.
        match gpu.acquire("other", 601) {
            AcquireOutcome::Granted { new_grant, .. } => assert!(new_grant),
            other => panic!("expected takeover grant, got {other:?}"),
        }
        assert_eq!(gpu.active_holder(602).unwrap().holder, "other");
    }

    #[test]
    fn release_mismatch_leaves_lock_intact() {
        let gpu = GpuExclusive::new(600);
        gpu.acquire("sweep", 0);
        match gpu.release("other") {
            ReleaseOutcome::Mismatch { record } => assert_eq!(record.holder, "sweep"),
            other => panic!("expected mismatch, got {other:?}"),
        }
        assert!(gpu.active_holder(1).is_some());
    }

    #[test]
    fn ttl_env_parsing_falls_back_on_junk() {
        // (Env is process-global; set+remove within this one test only.)
        std::env::set_var("CHORD_GPU_EXCLUSIVE_TTL_SECS", "not-a-number");
        assert_eq!(ttl_secs_from_env(), DEFAULT_TTL_SECS);
        std::env::set_var("CHORD_GPU_EXCLUSIVE_TTL_SECS", "0");
        assert_eq!(ttl_secs_from_env(), DEFAULT_TTL_SECS);
        std::env::set_var("CHORD_GPU_EXCLUSIVE_TTL_SECS", "1200");
        assert_eq!(ttl_secs_from_env(), 1200);
        std::env::remove_var("CHORD_GPU_EXCLUSIVE_TTL_SECS");
    }

    #[test]
    fn iso_utc_is_rfc3339() {
        // 2021-01-01T00:00:00Z
        assert!(iso_utc(1609459200).starts_with("2021-01-01T00:00:00"));
    }

    // ── Phase 1: keep-resident exemption + warm ──────────────────────────────

    #[test]
    fn keep_resident_membership() {
        let exempt = vec!["granite4.1:8b".to_string(), "lumina:latest".to_string()];
        assert!(is_keep_resident("granite4.1:8b", &exempt));
        assert!(is_keep_resident("lumina:latest", &exempt));
        assert!(!is_keep_resident("qwen3-coder:30b", &exempt));
        // Empty exempt set (the default) never matches → today's behavior.
        assert!(!is_keep_resident("granite4.1:8b", &[]));
    }

    #[test]
    fn partition_exempts_keep_resident_only() {
        // granite/embedding are keep-resident; the qwen coder is a sweep model
        // (→ evict); the empty name is dropped entirely.
        let resident = vec![
            "granite4.1:8b".to_string(),
            "qwen3-coder:30b".to_string(),
            "qwen3-embedding:0.6b".to_string(),
            "".to_string(),
        ];
        let exempt = vec![
            "granite4.1:8b".to_string(),
            "qwen3-embedding:0.6b".to_string(),
        ];
        let (to_evict, kept) = partition_evictable(&resident, &exempt);
        // The keep-resident models are NOT in the evict list...
        assert_eq!(to_evict, vec!["qwen3-coder:30b".to_string()]);
        // ...and the non-resident sweep model IS.
        assert!(to_evict.contains(&"qwen3-coder:30b".to_string()));
        assert!(!to_evict.contains(&"granite4.1:8b".to_string()));
        assert!(!to_evict.contains(&"qwen3-embedding:0.6b".to_string()));
        assert_eq!(
            kept,
            vec![
                "granite4.1:8b".to_string(),
                "qwen3-embedding:0.6b".to_string()
            ]
        );
    }

    #[test]
    fn partition_empty_exempt_evicts_everything() {
        // Default (empty exempt) preserves the original "unload all" behavior.
        let resident = vec!["a".to_string(), "b".to_string()];
        let (to_evict, kept) = partition_evictable(&resident, &[]);
        assert_eq!(to_evict, vec!["a".to_string(), "b".to_string()]);
        assert!(kept.is_empty());
    }

    #[tokio::test]
    async fn warm_is_fail_soft_on_unreachable_ollama() {
        // A warm against an unreachable Ollama must return 0 and NEVER panic/error
        // (a startup pre-warm or periodic re-warm can't be allowed to take Chord down).
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(200))
            .build()
            .unwrap();
        let models = vec!["granite4.1:8b".to_string()];
        // Reserved-for-docs TEST-NET-1 (RFC 5737) → connection fails fast.
        let warmed = warm_keep_resident_models(&client, "http://192.0.2.1:65535", &models).await;
        assert_eq!(warmed, 0);
    }

    #[tokio::test]
    async fn evict_is_fail_soft_on_unreachable_ollama() {
        // Same fail-soft guarantee for the eviction path (Ollama /api/ps unavailable
        // → 0, no error), with a keep-resident exempt set passed through.
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(200))
            .build()
            .unwrap();
        let exempt = vec!["granite4.1:8b".to_string()];
        let unloaded = evict_resident_models(&client, "http://192.0.2.1:65535", &exempt).await;
        assert_eq!(unloaded, 0);
    }

    #[test]
    fn warm_body_pins_indefinitely() {
        // The re-warm logic must issue keep_alive:-1 (indefinite residency).
        let body = keep_resident_warm_body("granite4.1:8b");
        assert_eq!(body["model"], "granite4.1:8b");
        assert_eq!(body["keep_alive"], -1);
        // Empty prompt: load / set keep_alive without generating.
        assert_eq!(body["prompt"], "");
    }

    // ── Phase 1.1: embed-model keep-resident fallback ────────────────────────

    #[test]
    fn embed_warm_body_pins_indefinitely() {
        // The embed keep-resident warm must also issue keep_alive:-1.
        let body = keep_resident_embed_warm_body("qwen3-embedding:0.6b");
        assert_eq!(body["model"], "qwen3-embedding:0.6b");
        assert_eq!(body["keep_alive"], -1);
        assert_eq!(body["prompt"], "");
    }

    #[test]
    fn is_embedding_model_matches_embed_substring() {
        assert!(is_embedding_model("qwen3-embedding:0.6b"));
        assert!(is_embedding_model("mxbai-embed-large:latest"));
        assert!(is_embedding_model("nomic-embed-text"));
        assert!(!is_embedding_model("granite4.1:8b"));
        assert!(!is_embedding_model("qwen3-coder:30b"));
    }

    #[tokio::test]
    async fn embed_model_uses_embeddings_endpoint_with_keep_alive_minus_one() {
        // A model whose name marks it as an embedding model must be warmed via
        // /api/embeddings with keep_alive:-1 — NOT /api/generate.
        let server = httpmock::MockServer::start_async().await;
        let embed_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/api/embeddings")
                .body_contains(r#""keep_alive":-1"#)
                .body_contains(r#""model":"qwen3-embedding:0.6b""#);
            then.status(200)
                .header("Content-Type", "application/json")
                .json_body(serde_json::json!({ "embedding": [0.0] }));
        });
        // /api/generate must NOT be hit for a name-matched embedding model.
        let generate_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/api/generate");
            then.status(200).body("");
        });

        let client = reqwest::Client::new();
        let models = vec!["qwen3-embedding:0.6b".to_string()];
        let warmed = warm_keep_resident_models(&client, &server.base_url(), &models).await;

        assert_eq!(
            warmed, 1,
            "embedding model should be pinned via /api/embeddings"
        );
        embed_mock.assert();
        assert_eq!(
            generate_mock.hits(),
            0,
            "/api/generate must be skipped for embed models"
        );
    }

    #[tokio::test]
    async fn generate_400_falls_back_to_embeddings_warm() {
        // A model whose name does NOT look like an embedding model but which
        // Ollama rejects from /api/generate with a 400 ("does not support
        // generate") must fall back to the /api/embeddings warm.
        let server = httpmock::MockServer::start_async().await;
        let generate_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/api/generate");
            then.status(400).json_body(
                serde_json::json!({ "error": "\"embedder\" does not support generate" }),
            );
        });
        let embed_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/api/embeddings")
                .body_contains(r#""keep_alive":-1"#);
            then.status(200)
                .json_body(serde_json::json!({ "embedding": [0.0] }));
        });

        let client = reqwest::Client::new();
        // Name has no "embed" substring → hits /api/generate first, gets 400,
        // then falls back to /api/embeddings.
        let models = vec!["mystery-vectorizer:latest".to_string()];
        let warmed = warm_keep_resident_models(&client, &server.base_url(), &models).await;

        assert_eq!(
            warmed, 1,
            "400 from /api/generate should trigger embed fallback"
        );
        generate_mock.assert();
        embed_mock.assert();
    }

    // ── RESIL-01: durable lease persistence ──────────────────────────────────

    #[test]
    fn with_state_reloads_live_lease_across_restart() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gpu_exclusive_lease.json");

        // First instance acquires and persists.
        let gpu = GpuExclusive::with_state(600, Some(path.clone()), 0);
        assert!(matches!(
            gpu.acquire("sweep", 10),
            AcquireOutcome::Granted {
                new_grant: true,
                ..
            }
        ));
        assert!(path.exists(), "lease file should be written on acquire");

        // Simulate a Chord restart: a brand-new instance loads the same file.
        let restarted = GpuExclusive::with_state(600, Some(path.clone()), 20);
        let held = restarted
            .active_holder(20)
            .expect("live lease should survive the restart");
        assert_eq!(held.holder, "sweep");
        assert_eq!(held.acquired_at, 10);
    }

    #[test]
    fn with_state_drops_expired_lease_on_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gpu_exclusive_lease.json");

        let gpu = GpuExclusive::with_state(600, Some(path.clone()), 0);
        gpu.acquire("sweep", 0);

        // Reload far past the TTL: the abandoned lease must NOT relock the GPU.
        let restarted = GpuExclusive::with_state(600, Some(path.clone()), 700);
        assert!(restarted.active_holder(700).is_none());
        // And a fresh holder can take over cleanly.
        assert!(matches!(
            restarted.acquire("other", 700),
            AcquireOutcome::Granted {
                new_grant: true,
                ..
            }
        ));
    }

    #[test]
    fn with_state_corrupt_file_starts_unheld_no_panic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gpu_exclusive_lease.json");
        std::fs::write(&path, b"{ this is not valid json ").unwrap();

        let gpu = GpuExclusive::with_state(600, Some(path.clone()), 0);
        assert!(gpu.active_holder(0).is_none());
        // Still fully functional after ignoring the corrupt file.
        assert!(matches!(
            gpu.acquire("sweep", 0),
            AcquireOutcome::Granted {
                new_grant: true,
                ..
            }
        ));
    }

    #[test]
    fn with_state_missing_file_starts_unheld() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does-not-exist.json");
        let gpu = GpuExclusive::with_state(600, Some(path), 0);
        assert!(gpu.active_holder(0).is_none());
    }

    #[test]
    fn release_clears_persisted_lease() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gpu_exclusive_lease.json");

        let gpu = GpuExclusive::with_state(600, Some(path.clone()), 0);
        gpu.acquire("sweep", 0);
        assert_eq!(gpu.release("sweep"), ReleaseOutcome::Released);

        // A restart after release must see no lease.
        let restarted = GpuExclusive::with_state(600, Some(path.clone()), 1);
        assert!(restarted.active_holder(1).is_none());
    }

    #[test]
    fn heartbeat_refresh_is_persisted() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gpu_exclusive_lease.json");

        let gpu = GpuExclusive::with_state(600, Some(path.clone()), 0);
        gpu.acquire("sweep", 0);
        // Heartbeat well after the first acquire; the persisted last_heartbeat
        // must advance so a restart sees a still-live (not stale) lease.
        gpu.acquire("sweep", 500);

        let restarted = GpuExclusive::with_state(600, Some(path.clone()), 1000);
        let held = restarted
            .active_holder(1000)
            .expect("refreshed lease should still be live 500s after the heartbeat");
        assert_eq!(held.last_heartbeat, 500);
        assert_eq!(held.acquired_at, 0);
    }

    #[test]
    fn no_state_path_writes_nothing_and_still_works() {
        // The original in-memory-only path: new(ttl) sets no state_path.
        let gpu = GpuExclusive::new(600);
        assert!(matches!(
            gpu.acquire("sweep", 0),
            AcquireOutcome::Granted {
                new_grant: true,
                ..
            }
        ));
        assert_eq!(gpu.active_holder(1).unwrap().holder, "sweep");
        assert_eq!(gpu.release("sweep"), ReleaseOutcome::Released);
    }
}
