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
                let record = guard.as_ref().expect("mismatch implies a live lock").clone();
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

/// Best-effort: unload EVERY model Ollama currently has resident, so the GPU is
/// clear for the exclusive holder. Non-fatal by construction — a missing
/// `OLLAMA_URL`, an unreachable Ollama, or nothing loaded all yield `0` with a
/// log line, never an error. Reuses the harness's own `/api/ps` poll shape.
pub async fn evict_resident_models(client: &reqwest::Client, ollama_base: &str) -> usize {
    let base = ollama_base.trim_end_matches('/');
    let stats = crate::sweep_status::ollama::query_ollama_ps(client, base).await;
    if !stats.available {
        info!("gpu-exclusive: Ollama /api/ps unavailable — nothing to evict (best-effort)");
        return 0;
    }
    let mut unloaded = 0usize;
    for m in stats.models {
        if m.name.is_empty() {
            continue;
        }
        // Ollama unloads a resident model when handed keep_alive:0.
        let url = format!("{base}/api/generate");
        let body = serde_json::json!({ "model": m.name, "keep_alive": 0 });
        match client
            .post(&url)
            .json(&body)
            .timeout(Duration::from_secs(30))
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => {
                info!(model = %m.name, "gpu-exclusive: evicted resident model");
                unloaded += 1;
            }
            Ok(r) => warn!(
                model = %m.name,
                status = r.status().as_u16(),
                "gpu-exclusive: unload request rejected (best-effort, continuing)"
            ),
            Err(e) => warn!(
                model = %m.name,
                error = %e,
                "gpu-exclusive: unload request failed (best-effort, continuing)"
            ),
        }
    }
    if unloaded > 0 {
        info!(count = unloaded, "gpu-exclusive: resident models evicted for exclusive holder");
    }
    unloaded
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
        assert_eq!(decide_acquire(None, "sweep", 10, 600), AcquireDecision::GrantNew);
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

    // ── RESIL-01: durable lease persistence ──────────────────────────────────

    #[test]
    fn with_state_reloads_live_lease_across_restart() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gpu_exclusive_lease.json");

        // First instance acquires and persists.
        let gpu = GpuExclusive::with_state(600, Some(path.clone()), 0);
        assert!(matches!(
            gpu.acquire("sweep", 10),
            AcquireOutcome::Granted { new_grant: true, .. }
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
            AcquireOutcome::Granted { new_grant: true, .. }
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
            AcquireOutcome::Granted { new_grant: true, .. }
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
            AcquireOutcome::Granted { new_grant: true, .. }
        ));
        assert_eq!(gpu.active_holder(1).unwrap().holder, "sweep");
        assert_eq!(gpu.release("sweep"), ReleaseOutcome::Released);
    }
}
