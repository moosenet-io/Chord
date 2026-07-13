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
//! ## Transition state machine (closed-world drain)
//! Idle-mode is a real state machine, not a snapshot-then-act flag, so it is
//! correct under CONCURRENT control calls and concurrent inference:
//!
//! ```text
//!   Active ──begin_enter (CAS)──▶ EnteringIdle ──finish_enter──▶ Idle
//!     ▲                                                            │
//!     └──────────── finish_activate ◀── Activating ◀──begin_activate (CAS)┘
//! ```
//!
//! - The `EnteringIdle`/`Activating` markers are installed ATOMICALLY (compare-and-swap
//!   under the state lock) *before* any side-effect work, so a second concurrent
//!   `enter`/`activate` sees the in-flight transition and returns a `changed:false`
//!   no-op instead of re-running drain/stop/evict/demote.
//! - New inference is admitted ([`IdleController::try_admit`]) only while the state is
//!   `Active`, and the admission increment happens *under the same lock* that flips the
//!   state. Once we flip to `EnteringIdle`, no further request can join the in-flight
//!   set, so the subsequent drain is a genuine CLOSED-WORLD drain — nothing slips in
//!   after the drain window opens.
//!
//! ## Compiler-lease awareness
//! Lazy activation and the watchdog distinguish a *compiler build lease* (see
//! [`is_compiler_lease`]) from any other GPU-exclusive holder (e.g. the intake sweep
//! harness). While a compiler build lease is held, a stray request does NOT tear down
//! the idle manifest, and the watchdog does NOT auto-activate — the build window stays
//! protected. A non-compiler GPU holder does not extend the idle window.
//!
//! ## Contract (see `README.md`)
//! - `POST /admin/idle`      → enter idle; reports freed RAM. Idempotent.
//! - `POST /admin/activate`  → restore service. Idempotent. Also happens lazily
//!                             on the next inference request ([`admit_inference`]).
//! - `GET  /admin/idle`      → current phase + resume manifest.
//! - A watchdog ([`watchdog_loop`]) re-activates on timeout so the proxy is never
//!   left silently dead; it holds off only while a COMPILER GPU-exclusive lease is
//!   actively held.
//!
//! ## Testability
//! The pure decision logic ([`decide_enter`], [`decide_activate`], [`is_compiler_lease`],
//! [`lazy_action`], [`watchdog_should_activate`], [`ResumeManifest::watchdog_expired`],
//! and the in-memory [`IdleController`] transitions) is separated from the clock, the
//! filesystem, and the network so it is exhaustively unit-testable offline with no
//! global state and no sleeping. The release/restore *side effects* (stopping backends,
//! evicting VRAM, reading `/proc/meminfo`) live in the async orchestration functions and
//! are best-effort.

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

/// Default substrings (case-insensitive) that identify a GPU-exclusive holder as a
/// *compiler build* lease, as opposed to some other GPU job (e.g. the intake sweep
/// harness `intake_coder_sweep`). These are role/label conventions, NOT infra
/// identifiers — override with `CHORD_IDLE_COMPILER_LEASE_HOLDERS` (comma-separated)
/// if the compiler adopts a different holder label.
pub const DEFAULT_COMPILER_LEASE_HOLDERS: &str = "compiler,build,bld";

/// Default bound (seconds) after which the watchdog force-resolves a controller
/// stuck in a TRANSIENT phase (`EnteringIdle`/`Activating`) back to `Active`. This
/// is a backstop only: the RAII [`EnterTransition`] guard already rolls a dropped /
/// panicked enter back to `Active` immediately, so this timeout only matters for a
/// pathological wedge. Comfortably longer than any drain+release, short enough to
/// self-heal quickly. Override with `CHORD_IDLE_STALE_TRANSITION_SECS`.
pub const DEFAULT_STALE_TRANSITION_SECS: u64 = 120;

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

/// Resolve the stale-transition backstop bound from `CHORD_IDLE_STALE_TRANSITION_SECS`
/// (seconds); a missing/blank/zero/unparseable value falls back to
/// [`DEFAULT_STALE_TRANSITION_SECS`].
pub fn stale_transition_secs_from_env() -> u64 {
    parse_positive_env(
        "CHORD_IDLE_STALE_TRANSITION_SECS",
        DEFAULT_STALE_TRANSITION_SECS,
    )
}

fn parse_positive_env(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default)
}

/// The configured set of compiler-lease holder substrings (lowercased), from
/// `CHORD_IDLE_COMPILER_LEASE_HOLDERS` or [`DEFAULT_COMPILER_LEASE_HOLDERS`]. Not a
/// secret and not an infra identifier — a list of role labels.
pub fn compiler_lease_holders_from_env() -> Vec<String> {
    let raw = std::env::var("CHORD_IDLE_COMPILER_LEASE_HOLDERS")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_COMPILER_LEASE_HOLDERS.to_string());
    raw.split(',')
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Does `holder` name a COMPILER build lease (per `patterns`)? Case-insensitive
/// substring match. Pure — the caller supplies the patterns and the holder label,
/// so this is fully unit-testable without the global lock.
pub fn is_compiler_lease(holder: &str, patterns: &[String]) -> bool {
    let h = holder.to_ascii_lowercase();
    patterns
        .iter()
        .any(|p| !p.is_empty() && h.contains(p.as_str()))
}

/// Is a COMPILER build lease currently held on the shared GPU? Reads the global
/// GPU-exclusive gate and applies [`is_compiler_lease`] to the live holder. A
/// non-compiler holder (or no holder) ⇒ `false` — only a build lease protects idle.
pub fn compiler_lease_held(now: u64) -> bool {
    match crate::gpu_exclusive::GPU_EXCLUSIVE.active_holder(now) {
        Some(rec) => is_compiler_lease(&rec.holder, &compiler_lease_holders_from_env()),
        None => false,
    }
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

/// Pure decision for the lazy-restore hook: when a real request arrives while idle,
/// should we restore, or preserve idle because a compiler build is still running?
#[derive(Debug, PartialEq, Eq)]
pub enum LazyAction {
    /// No compiler lease ⇒ restore service, then serve the request.
    Restore,
    /// A compiler build lease is still held ⇒ keep the idle manifest + watchdog
    /// protection intact; the request is shed (retryable 503) rather than allowed
    /// to tear the build window down.
    PreserveIdle,
}

pub fn lazy_action(compiler_lease_held: bool) -> LazyAction {
    if compiler_lease_held {
        LazyAction::PreserveIdle
    } else {
        LazyAction::Restore
    }
}

/// Pure decision for the watchdog: given whether the deadline has passed and the
/// current GPU holder (if any), should the watchdog auto-activate now? Defers ONLY
/// for a live compiler build lease; a non-compiler holder does not extend idle.
pub fn watchdog_should_activate(expired: bool, holder: Option<&str>, patterns: &[String]) -> bool {
    if !expired {
        return false;
    }
    match holder {
        Some(h) if is_compiler_lease(h, patterns) => false, // compiler build in progress → defer
        _ => true,                                          // no/other holder → auto-activate
    }
}

// ── In-memory controller + durable persistence ───────────────────────────────

/// The lifecycle phase of idle-mode. `EnteringIdle`/`Activating` are transient
/// transition markers held only for the duration of the (short) release/restore
/// work; they are never persisted (a crash mid-transition reloads as `Active`, and
/// the GPU-exclusive gate + watchdog keep things safe).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Phase {
    Active,
    EnteringIdle,
    Idle,
    Activating,
}

/// Internal state cell. Owns the manifest only in the idle/activating phases. The
/// transient phases carry the epoch second they began (`since`) so the watchdog can
/// detect and force-resolve a wedged transition (a backstop behind the RAII guard).
enum IdleState {
    Active,
    EnteringIdle {
        since: u64,
    },
    Idle(ResumeManifest),
    Activating {
        since: u64,
        manifest: ResumeManifest,
    },
}

impl IdleState {
    fn phase(&self) -> Phase {
        match self {
            IdleState::Active => Phase::Active,
            IdleState::EnteringIdle { .. } => Phase::EnteringIdle,
            IdleState::Idle(_) => Phase::Idle,
            IdleState::Activating { .. } => Phase::Activating,
        }
    }
    /// The manifest to persist for this phase: only a fully-`Idle` proxy persists a
    /// manifest; every other phase (including the transients) persists "not idle".
    fn to_persisted(&self) -> Option<&ResumeManifest> {
        match self {
            IdleState::Idle(m) => Some(m),
            _ => None,
        }
    }
    /// The epoch second a TRANSIENT phase began, or `None` for a steady phase.
    fn transition_since(&self) -> Option<u64> {
        match self {
            IdleState::EnteringIdle { since } | IdleState::Activating { since, .. } => Some(*since),
            _ => None,
        }
    }
}

/// Result of trying to BEGIN entering idle (CAS `Active → EnteringIdle`).
#[derive(Debug, PartialEq)]
pub enum BeginEnter {
    /// Won the CAS: caller MUST run release work then call [`IdleController::finish_enter`].
    Begin,
    /// Already fully idle ⇒ idempotent no-op (carries the existing manifest).
    AlreadyIdle(ResumeManifest),
    /// Another enter/activate transition is in flight ⇒ no-op, do NOT run release.
    InTransition,
}

/// Result of trying to BEGIN activating (CAS `Idle → Activating`).
#[derive(Debug, PartialEq)]
pub enum BeginActivate {
    /// Won the CAS: caller finishes with [`IdleController::finish_activate`].
    Begin(ResumeManifest),
    /// Already active ⇒ idempotent no-op.
    AlreadyActive,
    /// An enter/activate transition is in flight ⇒ no-op.
    InTransition,
}

/// Outcome of a full enter (begin+release+finish) against the live state.
#[derive(Debug, PartialEq)]
pub enum EnterOutcome {
    /// Transitioned active → idle; carries the stored manifest.
    Entered(ResumeManifest),
    /// Already idle; carries the existing manifest (idempotent).
    AlreadyIdle(ResumeManifest),
    /// A concurrent transition was already in flight; nothing was re-run.
    InTransition,
}

/// Outcome of a full activate against the live state.
#[derive(Debug, PartialEq)]
pub enum ActivateOutcome {
    /// Transitioned idle → active; carries the manifest that was cleared.
    Activated(ResumeManifest),
    /// Already active (idempotent no-op).
    AlreadyActive,
    /// A concurrent transition was in flight; nothing was re-run.
    InTransition,
}

/// Outcome of an admission attempt for a new inference request.
pub enum AdmitOutcome {
    /// Admitted while `Active`; holds the in-flight guard (already counted).
    Admitted(InflightGuard),
    /// Steady `Idle`: caller decides restore-vs-preserve (see [`lazy_action`]).
    Idle,
    /// Mid-transition (`EnteringIdle`/`Activating`): brief, retryable — do NOT admit.
    Transitioning,
}

/// Process-global idle-mode state machine. One Chord process serves one host, so
/// this is a singleton, like `GPU_EXCLUSIVE`.
pub struct IdleController {
    inner: RwLock<IdleState>,
    /// Count of admitted in-flight inference requests. Owned per-controller (not a
    /// module global) so unit tests are fully isolated. Shared with each
    /// [`InflightGuard`] via an `Arc` so the guard decrements the RIGHT counter on
    /// drop, no matter how long the request outlives the admission call.
    inflight: Arc<AtomicUsize>,
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
            inner: RwLock::new(IdleState::Active),
            inflight: Arc::new(AtomicUsize::new(0)),
            state_path: None,
        }
    }

    /// Construct with durable persistence at `state_path`, seeding any persisted
    /// manifest. A missing/corrupt file seeds `Active` and never panics.
    pub fn with_state(state_path: Option<PathBuf>) -> Self {
        let seed = match state_path.as_deref().and_then(load_persisted) {
            Some(m) => {
                info!(
                    reason = %m.reason,
                    entered_at = m.entered_at,
                    "idle-mode: reloaded persisted idle state across restart (watchdog will bound it)"
                );
                IdleState::Idle(m)
            }
            None => IdleState::Active,
        };
        Self {
            inner: RwLock::new(seed),
            inflight: Arc::new(AtomicUsize::new(0)),
            state_path,
        }
    }

    /// From `CHORD_STATE_DIR` (see [`crate::config::admin_idle_state_path`]).
    pub fn from_env() -> Self {
        Self::with_state(crate::config::admin_idle_state_path())
    }

    fn persist_locked(&self, current: &IdleState) {
        if let Some(path) = self.state_path.as_deref() {
            persist_state(path, &current.to_persisted().cloned());
        }
    }

    /// Current lifecycle phase (cheap snapshot).
    pub fn phase(&self) -> Phase {
        self.inner.read().expect("idle lock poisoned").phase()
    }

    /// Is Chord fully idle right now? (Transitions do NOT count as idle.)
    pub fn is_idle(&self) -> bool {
        matches!(
            &*self.inner.read().expect("idle lock poisoned"),
            IdleState::Idle(_)
        )
    }

    /// A snapshot of the current manifest (present while idle or activating) for the
    /// status endpoint.
    pub fn snapshot(&self) -> Option<ResumeManifest> {
        match &*self.inner.read().expect("idle lock poisoned") {
            IdleState::Idle(m) | IdleState::Activating { manifest: m, .. } => Some(m.clone()),
            _ => None,
        }
    }

    /// Try to admit ONE new inference request. The in-flight increment happens under
    /// the SAME write lock that flips the phase, so once a concurrent
    /// [`begin_enter`](Self::begin_enter) has installed `EnteringIdle`, this can
    /// never return `Admitted` — the drain that follows is closed-world.
    pub fn try_admit(&self) -> AdmitOutcome {
        let guard = self.inner.write().expect("idle lock poisoned");
        match &*guard {
            IdleState::Active => {
                AdmitOutcome::Admitted(InflightGuard::admit(self.inflight.clone()))
            }
            IdleState::Idle(_) => AdmitOutcome::Idle,
            IdleState::EnteringIdle { .. } | IdleState::Activating { .. } => {
                AdmitOutcome::Transitioning
            }
        }
    }

    /// Current number of admitted in-flight inference requests.
    pub fn inflight_count(&self) -> usize {
        self.inflight.load(Ordering::SeqCst)
    }

    /// Wait (bounded by `timeout`) for in-flight inference to drain to zero. Returns
    /// the number still in flight when it returned (0 = fully drained; >0 = the bound
    /// was hit and release proceeds anyway). Polls at 100ms. Because admission is
    /// closed once the phase left `Active`, the count is monotonically non-increasing
    /// here — a genuine closed-world drain.
    pub async fn drain_inflight(&self, timeout: Duration) -> usize {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let n = self.inflight_count();
            if n == 0 {
                return 0;
            }
            if tokio::time::Instant::now() >= deadline {
                return n;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// CAS `Active → EnteringIdle`. Installs the transition marker atomically BEFORE
    /// any release work, so exactly one caller ever runs the release side effects.
    /// Prefer [`try_begin_enter`](Self::try_begin_enter), whose RAII guard guarantees
    /// the phase is finalized even if the enter future is dropped mid-transition.
    pub fn begin_enter(&self) -> BeginEnter {
        let mut guard = self.inner.write().expect("idle lock poisoned");
        match &*guard {
            IdleState::Active => {
                *guard = IdleState::EnteringIdle { since: now_epoch() }; // transient — not persisted
                BeginEnter::Begin
            }
            IdleState::Idle(m) => BeginEnter::AlreadyIdle(m.clone()),
            IdleState::EnteringIdle { .. } | IdleState::Activating { .. } => {
                BeginEnter::InTransition
            }
        }
    }

    /// CAS into `EnteringIdle` and return an RAII [`EnterTransition`] guard. The guard
    /// MUST be finalized with [`EnterTransition::commit`]; if it is instead dropped
    /// (the enter future is cancelled on client disconnect, panics, or returns early),
    /// its `Drop` deterministically rolls `EnteringIdle → Active` so the controller can
    /// never wedge in the transient phase. `Err` carries the non-`Begin` CAS result.
    pub fn try_begin_enter(&self) -> Result<EnterTransition<'_>, BeginEnter> {
        match self.begin_enter() {
            BeginEnter::Begin => Ok(EnterTransition {
                ctl: self,
                committed: false,
            }),
            other => Err(other),
        }
    }

    /// Complete an enter: `EnteringIdle → Idle(manifest)`, persisting the manifest.
    /// Returns the stored manifest. Defensive: if the phase is not `EnteringIdle`
    /// (should never happen — only the `Begin` winner calls this), it still installs
    /// the idle manifest rather than losing the transition.
    pub fn finish_enter(&self, manifest: ResumeManifest) -> ResumeManifest {
        let mut guard = self.inner.write().expect("idle lock poisoned");
        *guard = IdleState::Idle(manifest.clone());
        self.persist_locked(&guard);
        manifest
    }

    /// Roll an in-progress `EnteringIdle` back to `Active` (the safe resting phase).
    /// Used by the [`EnterTransition`] guard when an enter is dropped before commit.
    /// Only acts while still `EnteringIdle`; if the phase already advanced (finished,
    /// or was recovered by the watchdog) it is left untouched, so a late drop can't
    /// clobber a subsequently-installed state.
    pub fn abort_enter(&self) {
        let mut guard = self.inner.write().expect("idle lock poisoned");
        if matches!(&*guard, IdleState::EnteringIdle { .. }) {
            *guard = IdleState::Active;
            self.persist_locked(&guard);
        }
    }

    /// CAS `Idle → Activating`, returning the manifest to clear. Concurrent
    /// activates: exactly one wins `Begin`; the rest see `InTransition`/`AlreadyActive`.
    pub fn begin_activate(&self) -> BeginActivate {
        let mut guard = self.inner.write().expect("idle lock poisoned");
        match &*guard {
            IdleState::Idle(_) => {
                let m = match std::mem::replace(&mut *guard, IdleState::Active) {
                    IdleState::Idle(m) => m,
                    _ => unreachable!("matched Idle above"),
                };
                *guard = IdleState::Activating {
                    since: now_epoch(),
                    manifest: m.clone(),
                }; // transient — not persisted
                BeginActivate::Begin(m)
            }
            IdleState::Active => BeginActivate::AlreadyActive,
            IdleState::EnteringIdle { .. } | IdleState::Activating { .. } => {
                BeginActivate::InTransition
            }
        }
    }

    /// Complete an activate: `Activating → Active`, persisting the cleared state.
    pub fn finish_activate(&self) {
        let mut guard = self.inner.write().expect("idle lock poisoned");
        *guard = IdleState::Active;
        self.persist_locked(&guard);
    }

    /// Backstop: if the controller has been stuck in a TRANSIENT phase
    /// (`EnteringIdle`/`Activating`) since before `now - max_age`, force-resolve it to
    /// `Active` and return `true`. Never touches a steady `Active`/`Idle` phase. This
    /// is insurance behind the RAII guard for the (should-be-impossible) case of a
    /// genuinely wedged transition; the watchdog calls it each tick.
    pub fn recover_stale_transition(&self, now: u64, max_age: u64) -> bool {
        let mut guard = self.inner.write().expect("idle lock poisoned");
        let Some(since) = guard.transition_since() else {
            return false;
        };
        if now.saturating_sub(since) >= max_age {
            *guard = IdleState::Active;
            self.persist_locked(&guard);
            true
        } else {
            false
        }
    }

    /// Convenience full enter used by unit tests: atomically `Active → Idle` (begin +
    /// finish with no release work in between). Idempotent, like the real path.
    pub fn enter(&self, manifest: ResumeManifest) -> EnterOutcome {
        match self.begin_enter() {
            BeginEnter::Begin => EnterOutcome::Entered(self.finish_enter(manifest)),
            BeginEnter::AlreadyIdle(m) => EnterOutcome::AlreadyIdle(m),
            BeginEnter::InTransition => EnterOutcome::InTransition,
        }
    }

    /// Full leave idle (begin + finish). Idempotent: already active ⇒ `AlreadyActive`.
    pub fn exit(&self) -> ActivateOutcome {
        match self.begin_activate() {
            BeginActivate::Begin(m) => {
                self.finish_activate();
                ActivateOutcome::Activated(m)
            }
            BeginActivate::AlreadyActive => ActivateOutcome::AlreadyActive,
            BeginActivate::InTransition => ActivateOutcome::InTransition,
        }
    }
}

/// RAII guard for an in-progress `EnteringIdle` transition (BLD-09 cycle-2 fix #1).
/// Obtained from [`IdleController::try_begin_enter`]. The transition spans several
/// `.await` points (drain, VRAM eviction, demote); if the enclosing future is
/// dropped/cancelled/panics before [`commit`](Self::commit), this guard's `Drop`
/// deterministically rolls the phase back to `Active`, so a cancelled enter can never
/// leave the controller wedged in `EnteringIdle` (which would 503 all inference and
/// block admin enter/activate indefinitely).
#[must_use = "commit the transition, or it will roll back to Active on drop"]
pub struct EnterTransition<'a> {
    ctl: &'a IdleController,
    committed: bool,
}

impl EnterTransition<'_> {
    /// Complete the transition: `EnteringIdle → Idle(manifest)`. Consumes the guard so
    /// its `Drop` becomes a no-op (nothing to roll back).
    pub fn commit(mut self, manifest: ResumeManifest) -> ResumeManifest {
        self.committed = true;
        self.ctl.finish_enter(manifest)
    }
}

impl Drop for EnterTransition<'_> {
    fn drop(&mut self) {
        if !self.committed {
            self.ctl.abort_enter();
            warn!(
                "idle-mode: enter transition dropped before commit (future cancelled/panicked) \
                 — rolled EnteringIdle back to Active"
            );
        }
    }
}

/// The process-global idle-mode controller. Handlers, the admission hook, and the
/// watchdog reference this; unit tests use isolated [`IdleController::new`]
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

// ── In-flight gauge ───────────────────────────────────────────────────────────

/// RAII guard for one admitted in-flight request. Constructed only via
/// [`IdleController::try_admit`] (which increments under the state lock, and only
/// while `Active`); the decrement on drop is lock-free and can never leak (fires on
/// panic / `?` / early return alike). Holds an `Arc` to its owning controller's
/// counter so it always decrements the counter it incremented.
#[must_use = "hold the guard for the duration of the request"]
pub struct InflightGuard {
    counter: Arc<AtomicUsize>,
}

impl InflightGuard {
    /// Increment `counter` and hand back the guard. Private: callers must go through
    /// [`IdleController::try_admit`] so the increment stays under the state lock.
    fn admit(counter: Arc<AtomicUsize>) -> Self {
        counter.fetch_add(1, Ordering::SeqCst);
        InflightGuard { counter }
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::SeqCst);
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
/// RAM, record the resume manifest. The transition marker is installed atomically
/// FIRST, so concurrent callers never double-run release and no new inference is
/// admitted once release begins. Returns the controller outcome plus (only on a
/// real transition) the freed-RAM report.
pub async fn enter_idle(
    state: &Arc<crate::routes::AppState>,
    reason: &str,
) -> (EnterOutcome, Option<IdleReport>) {
    // Atomically CLAIM the transition FIRST (fixes the TOCTOU: two concurrent enters
    // can no longer both observe "active" and both run release). Only the `Begin`
    // winner proceeds; everyone else returns a no-op with no side effects. The RAII
    // `transition` guard makes this cancellation-safe: if this future is dropped
    // (client disconnect), panics, or returns early before `transition.commit(...)`,
    // the guard's Drop rolls `EnteringIdle → Active` so the controller never wedges.
    let transition = match IDLE_MODE.try_begin_enter() {
        Ok(t) => t,
        Err(BeginEnter::AlreadyIdle(m)) => return (EnterOutcome::AlreadyIdle(m), None),
        Err(_) => return (EnterOutcome::InTransition, None),
    };
    // Phase is now `EnteringIdle`: `try_admit` rejects all new inference, so the
    // drain below is a genuine closed-world drain.

    info!(
        reason,
        "idle-mode: entering — draining and releasing host resources"
    );

    // 1. Drain in-flight inference (bounded, closed-world).
    let inflight_remaining = IDLE_MODE
        .drain_inflight(Duration::from_secs(drain_secs_from_env()))
        .await;
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
    // Complete the transition: `EnteringIdle → Idle` (consumes the guard so its Drop
    // rollback becomes a no-op). Past this point the enter is durably finalized.
    let stored = transition.commit(manifest);

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
    (EnterOutcome::Entered(stored), Some(report))
}

/// Leave idle and resume normal serving. Idempotent, CAS-guarded. Models reload
/// LAZILY on their next request (Ollama/llama-server cold-load on demand), so there
/// is no async release work on this path — hence `Activating` is a nanosecond
/// window and activate is effectively a single atomic transition.
pub async fn activate(state: &Arc<crate::routes::AppState>, reason: &str) -> ActivateOutcome {
    let _ = state; // reserved: a future eager pre-warm could use the registry/client.
    match IDLE_MODE.begin_activate() {
        BeginActivate::Begin(m) => {
            // (restore side effects would go here; lazy reload means none today.)
            IDLE_MODE.finish_activate();
            info!(
                reason,
                resident_models = m.resident_models.len(),
                "idle-mode: activated — normal serving resumed (models reload on demand)"
            );
            ActivateOutcome::Activated(m)
        }
        BeginActivate::AlreadyActive => ActivateOutcome::AlreadyActive,
        BeginActivate::InTransition => ActivateOutcome::InTransition,
    }
}

/// Result of [`admit_inference`]: either a held guard or a ready-to-return response.
pub enum Admission {
    Admitted(InflightGuard),
    Rejected(Response),
}

/// Admission hook for the inference handlers. Returns the in-flight guard to hold
/// for the request, or a `Response` to short-circuit with:
/// - `Active`        ⇒ admitted (guard already counted under the state lock).
/// - `EnteringIdle`/`Activating` ⇒ retryable 503 (a brief, bounded transition window).
/// - `Idle` + no compiler lease ⇒ lazily restore, then admit.
/// - `Idle` + compiler build lease held ⇒ retryable 503 that PRESERVES idle +
///   watchdog protection (the build window is not torn down by stray traffic).
pub async fn admit_inference(state: &Arc<crate::routes::AppState>) -> Admission {
    // Bounded attempts: at most one lazy restore, then a re-admit. A pathological
    // re-idle between the two just yields a retryable 503 rather than spinning.
    for _ in 0..3 {
        match IDLE_MODE.try_admit() {
            AdmitOutcome::Admitted(guard) => return Admission::Admitted(guard),
            AdmitOutcome::Transitioning => return Admission::Rejected(idle_transition_response()),
            AdmitOutcome::Idle => match lazy_action(compiler_lease_held(now_epoch())) {
                LazyAction::PreserveIdle => {
                    info!(
                        "idle-mode: request arrived while idle but a compiler build lease is held — \
                         preserving idle (503, watchdog still protecting the build window)"
                    );
                    return Admission::Rejected(idle_compiler_busy_response());
                }
                LazyAction::Restore => {
                    info!("idle-mode: lazy activate — a real request arrived while idle");
                    let _ = activate(state, "lazy-on-request").await;
                    // loop: re-admit now that we should be Active
                }
            },
        }
    }
    Admission::Rejected(idle_transition_response())
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
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};

use crate::routes::{auth_check, auth_error_response, AppState};

/// Retryable 503 while a short idle/activate transition is in progress.
fn idle_transition_response() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [(header::RETRY_AFTER, HeaderValue::from_static("2"))],
        Json(serde_json::json!({
            "error": "idle_transition_in_progress",
            "status": "transitioning",
        })),
    )
        .into_response()
}

/// Retryable 503 shed while idle because a compiler build lease is still held; idle
/// state + watchdog protection are deliberately PRESERVED.
fn idle_compiler_busy_response() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [(header::RETRY_AFTER, HeaderValue::from_static("5"))],
        Json(serde_json::json!({
            "error": "idle_compiler_build_active",
            "status": "idle",
        })),
    )
        .into_response()
}

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
        EnterOutcome::InTransition => (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "entering_idle",
                "changed": false,
                "note": "another idle/activate transition is in progress",
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
        ActivateOutcome::InTransition => (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "activating",
                "changed": false,
                "note": "another idle/activate transition is in progress",
            })),
        )
            .into_response(),
    }
}

/// `GET /admin/idle` — current phase + resume manifest for observability.
pub async fn admin_idle_status(
    State(state): State<StdArc<AppState>>,
    headers: HeaderMap,
) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }
    let phase = IDLE_MODE.phase();
    let phase_str = match &phase {
        Phase::Active => "active",
        Phase::EnteringIdle => "entering_idle",
        Phase::Idle => "idle",
        Phase::Activating => "activating",
    };
    let body = match IDLE_MODE.snapshot() {
        Some(m) => serde_json::json!({
            "status": if phase == Phase::Idle { "idle" } else { phase_str },
            "phase": phase_str,
            "reason": m.reason,
            "entered_at": crate::gpu_exclusive::iso_utc(m.entered_at),
            "watchdog_deadline": crate::gpu_exclusive::iso_utc(m.watchdog_deadline),
            "watchdog_expired": m.watchdog_expired(now_epoch()),
            "resident_models": m.resident_models,
            "inflight": IDLE_MODE.inflight_count(),
        }),
        None => serde_json::json!({
            "status": "active",
            "phase": phase_str,
            "inflight": IDLE_MODE.inflight_count(),
        }),
    };
    (StatusCode::OK, Json(body)).into_response()
}

// ── Watchdog ──────────────────────────────────────────────────────────────────

/// Background fail-safe: every `interval`, if idle and the watchdog deadline has
/// passed AND no COMPILER build lease is currently held, auto-activate so the proxy
/// is never left silently dead (a crashed/forgotten compiler, or a stale idle state
/// reloaded after a Chord restart). While a compiler build lease IS held the deadline
/// is deferred — a legitimately long build keeps Chord idle as long as it holds the
/// GPU. A NON-compiler GPU holder (e.g. the intake sweep harness) does NOT extend the
/// idle window. Follows the `idle_stop_sweep` spawn pattern in `main.rs`.
pub async fn watchdog_loop(state: Arc<crate::routes::AppState>, interval: Duration) {
    info!(
        interval_secs = interval.as_secs(),
        "idle-mode watchdog started"
    );
    let patterns = compiler_lease_holders_from_env();
    let stale_secs = stale_transition_secs_from_env();
    loop {
        tokio::time::sleep(interval).await;
        let now = now_epoch();
        // Backstop (BLD-09 cycle-2 fix #1): force-resolve a controller wedged in a
        // transient phase (EnteringIdle/Activating) past the stale bound back to
        // Active. The RAII EnterTransition guard normally prevents this; this only
        // fires for a pathological wedge that escaped the guard.
        if IDLE_MODE.recover_stale_transition(now, stale_secs) {
            warn!(
                stale_secs,
                "idle-mode watchdog: force-resolved a stale idle transition back to Active"
            );
            continue;
        }
        let Some(m) = IDLE_MODE.snapshot() else {
            continue;
        };
        let holder = crate::gpu_exclusive::GPU_EXCLUSIVE.active_holder(now);
        let holder_label = holder.as_ref().map(|r| r.holder.as_str());
        if !watchdog_should_activate(m.watchdog_expired(now), holder_label, &patterns) {
            continue;
        }
        warn!(
            reason = %m.reason,
            "idle-mode watchdog: deadline passed with no active compiler lease — auto-activating (fail-safe)"
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

    fn holders() -> Vec<String> {
        vec!["compiler".into(), "build".into(), "bld".into()]
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

    // ── compiler-lease matching (findings #3/#4) ──────────────────────────────

    #[test]
    fn compiler_lease_matches_only_build_holders() {
        let p = holders();
        assert!(is_compiler_lease("compiler", &p));
        assert!(is_compiler_lease("bld-05-compiler", &p));
        assert!(is_compiler_lease("constellation-build", &p));
        assert!(is_compiler_lease("COMPILER", &p)); // case-insensitive
                                                    // a DIFFERENT GPU job must NOT read as a compiler lease:
        assert!(!is_compiler_lease("intake_coder_sweep", &p));
        assert!(!is_compiler_lease("intake_assistant_sweep", &p));
        assert!(!is_compiler_lease("", &p));
    }

    #[test]
    fn lazy_action_preserves_idle_only_under_compiler_lease() {
        // finding #3: a held compiler lease must NOT clear idle.
        assert_eq!(lazy_action(true), LazyAction::PreserveIdle);
        assert_eq!(lazy_action(false), LazyAction::Restore);
    }

    #[test]
    fn watchdog_defers_only_for_compiler_lease() {
        let p = holders();
        // Not expired ⇒ never activate, regardless of holder.
        assert!(!watchdog_should_activate(false, Some("compiler"), &p));
        assert!(!watchdog_should_activate(false, None, &p));
        // Expired + compiler lease held ⇒ defer (finding #4).
        assert!(!watchdog_should_activate(true, Some("bld-05-compiler"), &p));
        // Expired + a NON-compiler GPU holder ⇒ auto-activate anyway.
        assert!(watchdog_should_activate(
            true,
            Some("intake_coder_sweep"),
            &p
        ));
        // Expired + no holder ⇒ auto-activate.
        assert!(watchdog_should_activate(true, None, &p));
    }

    // ── controller transitions (isolated instance, no globals) ───────────────

    #[test]
    fn enter_then_exit_cycle() {
        let ctl = IdleController::new();
        assert_eq!(ctl.phase(), Phase::Active);
        assert!(!ctl.is_idle());
        assert!(ctl.snapshot().is_none());

        let m = manifest("compiler", 100, 3700);
        match ctl.enter(m.clone()) {
            EnterOutcome::Entered(got) => assert_eq!(got, m),
            other => panic!("expected Entered, got {other:?}"),
        }
        assert!(ctl.is_idle());
        assert_eq!(ctl.phase(), Phase::Idle);
        assert_eq!(ctl.snapshot().unwrap(), m);

        match ctl.exit() {
            ActivateOutcome::Activated(got) => assert_eq!(got, m),
            other => panic!("expected Activated, got {other:?}"),
        }
        assert!(!ctl.is_idle());
        assert_eq!(ctl.phase(), Phase::Active);
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

    // ── concurrency-safety: CAS + closed-world drain (findings #1/#2) ─────────

    #[test]
    fn begin_enter_is_exclusive_cas_release_runs_once() {
        // finding #1: only ONE caller may run release. The first begin_enter wins;
        // a second while EnteringIdle must NOT also get Begin.
        let ctl = IdleController::new();
        assert_eq!(ctl.begin_enter(), BeginEnter::Begin);
        assert_eq!(ctl.begin_enter(), BeginEnter::InTransition);
        // finish and confirm a later begin sees AlreadyIdle, never a second Begin.
        let m = manifest("compiler", 1, 2);
        let _ = ctl.finish_enter(m.clone());
        match ctl.begin_enter() {
            BeginEnter::AlreadyIdle(got) => assert_eq!(got, m),
            other => panic!("expected AlreadyIdle, got {other:?}"),
        }
    }

    #[test]
    fn begin_activate_is_exclusive_cas() {
        let ctl = IdleController::new();
        ctl.enter(manifest("compiler", 1, 2));
        assert!(matches!(ctl.begin_activate(), BeginActivate::Begin(_)));
        // While Activating, a second begin must not also win.
        assert_eq!(ctl.begin_activate(), BeginActivate::InTransition);
        ctl.finish_activate();
        assert_eq!(ctl.begin_activate(), BeginActivate::AlreadyActive);
    }

    // ── cancellation-safety of the EnteringIdle transition (finding #1) ───────

    #[test]
    fn dropped_enter_transition_rolls_back_to_active() {
        // A transition guard dropped WITHOUT commit (future cancelled/panicked) must
        // leave the controller recoverable (Active), never wedged in EnteringIdle.
        let ctl = IdleController::new();
        {
            let _t = ctl.try_begin_enter().expect("Active ⇒ transition begins");
            assert_eq!(ctl.phase(), Phase::EnteringIdle);
            // fall out of scope WITHOUT calling commit → Drop rolls back
        }
        assert_eq!(
            ctl.phase(),
            Phase::Active,
            "dropped transition must roll back to Active"
        );
        assert!(!ctl.is_idle());
        // Controller is fully usable afterwards.
        assert!(matches!(
            ctl.enter(manifest("compiler", 1, 2)),
            EnterOutcome::Entered(_)
        ));
    }

    #[test]
    fn committed_enter_transition_reaches_idle() {
        let ctl = IdleController::new();
        let t = ctl.try_begin_enter().expect("Active ⇒ transition begins");
        let m = manifest("compiler", 5, 10);
        assert_eq!(t.commit(m.clone()), m);
        assert!(ctl.is_idle());
        assert_eq!(ctl.snapshot().unwrap(), m);
    }

    #[test]
    fn try_begin_enter_errors_when_not_active() {
        let ctl = IdleController::new();
        ctl.enter(manifest("compiler", 1, 2));
        // Already idle ⇒ Err(AlreadyIdle), and NO transition guard handed out (so no
        // spurious rollback of the live idle state when that Err is dropped).
        match ctl.try_begin_enter() {
            Err(BeginEnter::AlreadyIdle(_)) => {}
            other => panic!("expected Err(AlreadyIdle), got {other:?}"),
        }
        assert!(
            ctl.is_idle(),
            "a rejected try_begin_enter must not disturb idle"
        );
    }

    #[test]
    fn recover_stale_transition_only_when_stale() {
        let ctl = IdleController::new();
        // Put it in EnteringIdle (records `since = now_epoch()`).
        assert_eq!(ctl.begin_enter(), BeginEnter::Begin);
        let now = crate::gpu_exclusive::now_epoch();
        // Fresh transition ⇒ NOT recovered.
        assert!(!ctl.recover_stale_transition(now, 120));
        assert_eq!(ctl.phase(), Phase::EnteringIdle);
        // Well past the bound ⇒ force-resolved to Active.
        assert!(ctl.recover_stale_transition(now + 1_000, 120));
        assert_eq!(ctl.phase(), Phase::Active);
        // A steady phase is never touched.
        assert!(!ctl.recover_stale_transition(now + 1_000, 120));
    }

    #[test]
    fn no_inflight_admitted_after_entering_idle() {
        // finding #2: once we flip to EnteringIdle, try_admit must reject — no new
        // request can join the in-flight set, so the drain is closed-world.
        // The counter is per-controller, so this test is fully isolated from any
        // other test touching in-flight state (no shared global gauge).
        let ctl = IdleController::new();
        // Active ⇒ admits and increments.
        assert_eq!(ctl.inflight_count(), 0);
        let guard = match ctl.try_admit() {
            AdmitOutcome::Admitted(g) => {
                assert_eq!(ctl.inflight_count(), 1);
                g
            }
            _ => panic!("Active must admit"),
        };
        drop(guard);
        assert_eq!(ctl.inflight_count(), 0);

        // Enter the transition; now admission must be refused with NO increment.
        assert_eq!(ctl.begin_enter(), BeginEnter::Begin);
        assert!(matches!(ctl.try_admit(), AdmitOutcome::Transitioning));
        assert_eq!(
            ctl.inflight_count(),
            0,
            "no request admitted after EnteringIdle"
        );

        // Fully idle also refuses admission (caller lazy-activates instead).
        let _ = ctl.finish_enter(manifest("compiler", 1, 2));
        assert!(matches!(ctl.try_admit(), AdmitOutcome::Idle));
        assert_eq!(ctl.inflight_count(), 0);
    }

    #[test]
    fn admit_guard_increments_and_decrements() {
        let ctl = IdleController::new();
        assert_eq!(ctl.inflight_count(), 0);
        match ctl.try_admit() {
            AdmitOutcome::Admitted(g) => {
                assert_eq!(ctl.inflight_count(), 1);
                drop(g);
                assert_eq!(ctl.inflight_count(), 0);
            }
            _ => panic!("Active must admit"),
        }
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
    fn entering_idle_is_not_persisted() {
        // The transient marker must not persist: a crash mid-enter reloads Active.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("admin_idle_state.json");
        let ctl = IdleController::with_state(Some(path.clone()));
        assert_eq!(ctl.begin_enter(), BeginEnter::Begin); // EnteringIdle, no finish
        let restarted = IdleController::with_state(Some(path));
        assert!(
            !restarted.is_idle(),
            "EnteringIdle must not persist as idle"
        );
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

    #[test]
    fn compiler_lease_holders_default_when_unset() {
        std::env::remove_var("CHORD_IDLE_COMPILER_LEASE_HOLDERS");
        let p = compiler_lease_holders_from_env();
        assert!(p.contains(&"compiler".to_string()));
        assert!(is_compiler_lease("compiler", &p));
    }
}
