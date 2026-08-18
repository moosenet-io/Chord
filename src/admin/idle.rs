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
//! - `GET  /admin/activity`  → live *serving* activity (CHORD-ACT-01): whether any
//!                             inference is in flight and, if not, how long Chord has
//!                             been quiet. This is a DIFFERENT signal from `/admin/idle`
//!                             (which reports the idle-MODE phase) — the compiler
//!                             scheduler reads it to dispatch heavy builds only while
//!                             Chord is genuinely idle, without being blocked forever by
//!                             a continuously-serving proxy.
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
use std::sync::atomic::{AtomicI64, AtomicU64, AtomicUsize, Ordering};
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

/// CHRD-135: default bound (seconds) on how long a compiler-build lease may defer
/// the idle-mode watchdog WITHOUT a fresh heartbeat. A live compiler lease
/// ([`is_compiler_lease`]) used to defer the watchdog UNCONDITIONALLY — no bound at
/// all — for as long as `GPU_EXCLUSIVE` reported ANY holder matching a compiler
/// pattern, even one whose last heartbeat was ancient. That let a stale/abandoned
/// lease keep Chord parked off its default assistant-fleet mode forever.
///
/// This is now measured as ABSENCE OF RENEWAL, never elapsed time since idle was
/// entered: a genuinely long-running build that keeps heartbeating
/// (`POST /v1/gpu-exclusive/acquire`, same holder — a heartbeat refresh per
/// [`crate::gpu_exclusive::AcquireDecision::Refresh`]) more often than this window
/// is NEVER auto-reactivated mid-build (reverting mid-build would let models reload
/// and eat the VRAM the lease exists to protect — the exact failure the lease
/// prevents). Once `coder_activity_timeout` seconds pass with no fresh heartbeat,
/// Chord reverts to its default assistant-fleet mode on its own.
///
/// 900s (15 minutes) matches the standing `idle_stop_secs` convention for
/// on-demand backends (`lemonade-coder`, see `src/models/backends.rs`). Override
/// with `CHORD_IDLE_CODER_ACTIVITY_TIMEOUT_SECS`.
///
/// **Coupling with `GPU_EXCLUSIVE`'s own TTL (review, round 2):** this value is
/// meant as a FLOOR — "revert only after AT LEAST this much silence" — but the
/// watchdog reads liveness through `GPU_EXCLUSIVE`, whose OWN, separately
/// configured `ttl` ([`crate::gpu_exclusive::DEFAULT_TTL_SECS`], 600s by
/// default, a shorter and UNRELATED knob governing when a different caller may
/// steal the lock) would silently undercut this floor if used directly — a
/// record could read as gone at 600s even though this constant promises 900.
/// `watchdog_tick` (`src/admin/idle.rs`) closes that gap by computing its own
/// holder-liveness against `effective_ttl = max(gpu.ttl(),
/// coder_activity_timeout)`, so THIS value always wins whenever it is the
/// larger of the two (the default case, 900 > 600) — see the doc on
/// `watchdog_tick` for the full mechanics. Pinned by
/// `watchdog_tick_default_ttls_honor_the_900s_coder_activity_floor` below.
pub const DEFAULT_CODER_ACTIVITY_TIMEOUT_SECS: u64 = 900;

/// Resolve the coder-activity (heartbeat-silence) timeout from
/// `CHORD_IDLE_CODER_ACTIVITY_TIMEOUT_SECS` (seconds); a missing/blank/zero/
/// unparseable value falls back to [`DEFAULT_CODER_ACTIVITY_TIMEOUT_SECS`].
pub fn coder_activity_timeout_secs_from_env() -> u64 {
    parse_positive_env(
        "CHORD_IDLE_CODER_ACTIVITY_TIMEOUT_SECS",
        DEFAULT_CODER_ACTIVITY_TIMEOUT_SECS,
    )
}

/// Default hard budget (seconds) for the ENTIRE `enter_idle` release sequence
/// (drain + stop backends + evict VRAM + demote). Bounding release under an explicit
/// timeout — kept STRICTLY BELOW the stale-recovery threshold — makes it structurally
/// impossible for the watchdog to reopen admission while release is still running: the
/// release either completes (→ `Idle`) or self-aborts via this budget (→ `Active`
/// through the guard, with admission having been closed the whole `EnteringIdle`
/// window). Comfortably above the drain deadline. Override `CHORD_IDLE_RELEASE_BUDGET_SECS`.
pub const DEFAULT_RELEASE_BUDGET_SECS: u64 = 90;

/// Default bound (seconds) after which the watchdog force-resolves a controller stuck
/// in a TRANSIENT phase (`EnteringIdle`/`Activating`) back to `Active`. MUST be
/// strictly greater than [`DEFAULT_RELEASE_BUDGET_SECS`] so the watchdog can only ever
/// recover a transition whose release future has ALREADY self-aborted or vanished —
/// never one still doing live release work (see [`stale_transition_secs_from_env`],
/// which clamps this ordering at runtime). Backstop only: the RAII [`EnterTransition`]
/// guard already rolls a dropped/panicked enter back to `Active` immediately.
/// Override with `CHORD_IDLE_STALE_TRANSITION_SECS`.
pub const DEFAULT_STALE_TRANSITION_SECS: u64 = 300;

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

/// Resolve the whole-release budget from `CHORD_IDLE_RELEASE_BUDGET_SECS` (seconds); a
/// missing/blank/zero/unparseable value falls back to [`DEFAULT_RELEASE_BUDGET_SECS`].
pub fn release_budget_secs_from_env() -> u64 {
    parse_positive_env(
        "CHORD_IDLE_RELEASE_BUDGET_SECS",
        DEFAULT_RELEASE_BUDGET_SECS,
    )
}

/// Resolve the stale-transition backstop bound from `CHORD_IDLE_STALE_TRANSITION_SECS`
/// (seconds), CLAMPED so it is always strictly greater than the release budget. This
/// preserves the core invariant: the release future self-aborts (via its budget) BEFORE
/// the watchdog is ever allowed to force-recover the transition, so stale-recovery can
/// only fire once release is already gone — never concurrently with live release. A
/// misconfiguration (stale ≤ budget) is logged and clamped up to a safe value.
pub fn stale_transition_secs_from_env() -> u64 {
    let stale = parse_positive_env(
        "CHORD_IDLE_STALE_TRANSITION_SECS",
        DEFAULT_STALE_TRANSITION_SECS,
    );
    let budget = release_budget_secs_from_env();
    if stale > budget {
        return stale;
    }
    // Misconfigured: clamp strictly above the budget (≥ 1.5× budget, and never below
    // the safe default), so the ordering invariant always holds.
    let safe = budget
        .saturating_add(budget / 2)
        .max(DEFAULT_STALE_TRANSITION_SECS);
    warn!(
        stale,
        budget,
        clamped_to = safe,
        "CHORD_IDLE_STALE_TRANSITION_SECS ≤ release budget — clamping up to preserve the \
         no-mid-release-admission invariant"
    );
    safe
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

/// CHRD-135 round-4: default substrings (case-insensitive) that identify a
/// GPU-exclusive holder as a real MINT *coder-activity* lease — as opposed to a
/// COMPILER build lease ([`DEFAULT_COMPILER_LEASE_HOLDERS`]), which is an
/// entirely separate, independently-configured pattern set. Override with
/// `CHORD_IDLE_CODER_ACTIVITY_HOLDERS` (comma-separated) if these labels drift.
///
/// **Why a separate set, not a widened `DEFAULT_COMPILER_LEASE_HOLDERS`:** that
/// constant (and [`is_compiler_lease`]/[`compiler_lease_holders_from_env`]) is
/// mirrored verbatim in Terminus `src/mint/idle.rs` and feeds MINT's own
/// `decide_enter`/`decide_activate` — an UNCONDITIONAL, unrelated defer rule.
/// Widening it to catch coder holders would silently change MINT's enter/defer
/// behavior too. This set is used ONLY by the new floor branch in
/// [`watchdog_tick_should_activate`] below — it never touches
/// [`watchdog_should_activate`], [`is_compiler_lease`], or the compiler pattern
/// set, so the existing compiler-lease defer semantics are unchanged byte for
/// byte.
///
/// **Chosen default — `"coder,assistant"`:** covers `intake_coder_sweep` and
/// `intake_coder_case` (via `coder`) and `intake_assistant_sweep` (via
/// `assistant`) — three of the four real production `GPU_EXCLUSIVE` holder
/// labels enumerated by MINT's own `mint_gpu_holders_from_env()` (Terminus
/// `src/intake/{coder_sweep.rs,coder_case.rs,assistant/runner.rs}`). The
/// COMPILER patterns (`compiler,build,bld`) are deliberately NOT folded in
/// here — the Terminus compiler drives Chord via `admin_idle_enter`/`activate`
/// directly (reason `"compiler-heavy-build"`) and never takes a
/// `GPU_EXCLUSIVE` lease at all, so it emits no holder label or heartbeat for
/// this set to match; conflating the two pattern sets would just re-widen the
/// compiler set's blast radius for no benefit.
///
/// **`mint_breakfix` (Terminus `src/intake/breakfix.rs`, `BREAKFIX_GPU_HOLDER`)
/// is deliberately EXCLUDED from the default, not an oversight.** An earlier
/// revision of this constant included it on the reasoning that MINT enumerates
/// it alongside the other three as "the same category" of heartbeat-bearing
/// work — that reasoning does not actually establish the one fact that
/// matters: HOW OFTEN a breakfix run renews its lease while genuinely still
/// working. This module has no visibility into `breakfix.rs`'s renewal
/// cadence (this change is scoped to Chord only). If a legitimate repair run
/// goes quiet for stretches ≥ `coder_activity_timeout` (900s default) while
/// still doing real work — plausible for a "repair" path that may wait on a
/// slow external step between heartbeats, unlike a sweep loop — folding it
/// into this set would let the floor revert Chord mid-repair, which is the
/// exact failure mode ([`DEFAULT_CODER_ACTIVITY_TIMEOUT_SECS`]'s doc, and the
/// compiler-lease floor before it) this whole mechanism exists to prevent.
/// `mint_breakfix` does NOT match `DEFAULT_COMPILER_LEASE_HOLDERS` either, so
/// excluding it here changes nothing about its behavior TODAY — it stays
/// governed by the absolute watchdog deadline alone, exactly as before this
/// item, until someone verifies its actual heartbeat cadence against Terminus
/// source and either confirms it's safe to add here or gives it a
/// `CHORD_IDLE_CODER_ACTIVITY_HOLDERS` override. Do not add `"breakfix"` back
/// without that verification.
pub const DEFAULT_CODER_ACTIVITY_HOLDERS: &str = "coder,assistant";

/// The configured set of coder-activity holder substrings (lowercased), from
/// `CHORD_IDLE_CODER_ACTIVITY_HOLDERS` or [`DEFAULT_CODER_ACTIVITY_HOLDERS`].
/// Mirrors the shape of [`compiler_lease_holders_from_env`] exactly, but is a
/// wholly independent knob — not a secret, not an infra identifier.
pub fn coder_activity_holders_from_env() -> Vec<String> {
    let raw = std::env::var("CHORD_IDLE_CODER_ACTIVITY_HOLDERS")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_CODER_ACTIVITY_HOLDERS.to_string());
    raw.split(',')
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Does `holder` name a real MINT coder-activity lease (per `patterns`)?
/// Case-insensitive substring match — same matching rule as
/// [`is_compiler_lease`], intentionally duplicated rather than shared so that
/// [`is_compiler_lease`] itself (and everything that calls it) stays byte-for-byte
/// unchanged by this addition. Pure — fully unit-testable offline.
pub fn is_coder_activity_holder(holder: &str, patterns: &[String]) -> bool {
    let h = holder.to_ascii_lowercase();
    patterns
        .iter()
        .any(|p| !p.is_empty() && h.contains(p.as_str()))
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
    /// CHRD-135 defect #3: identity token for THIS idle episode, stamped by
    /// [`IdleController::finish_enter`] from the `EnteringIdle` transition's own
    /// generation counter — the caller's initializer value is always overwritten,
    /// so any placeholder (including the pre-CHRD-135 absence of this field) is
    /// safe. Lets an activation DECISION (e.g. the watchdog's) be pinned to the
    /// SPECIFIC episode it evaluated: if a newer episode is current by the time the
    /// activation actually runs, the CAS in
    /// [`IdleController::begin_activate_for_episode`] refuses rather than cancels
    /// it. `#[serde(default)]` so a manifest persisted by a pre-CHRD-135 binary
    /// still deserializes across a rolling restart (defaults to `0`, harmless since
    /// generation counters also reset to `0` on a fresh process).
    #[serde(default)]
    pub episode: u64,
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

/// CHRD-135: has a compiler-build lease's heartbeat gone STALE — no renewal within
/// `timeout` seconds of `now`? `saturating_sub` so a clock that briefly steps
/// backwards reads as fresh (age 0), never spuriously stale — the same
/// clock-safety discipline as [`crate::gpu_exclusive::LockRecord::is_expired`].
/// Pure — no IO, no clock, exhaustively unit-testable.
///
/// Non-blocking review finding #5: `last_heartbeat == None` is treated as
/// immediately stale (never a reason to defer) for callers that construct one
/// directly or in a unit test. Via the ACTUAL production call path
/// (`watchdog_tick` → `watchdog_tick_should_activate`), `LockRecord::last_heartbeat`
/// is a non-`Option<u64>`, and `is_compiler` (hence this function) is only ever
/// evaluated once a real `Some(holder)` was already matched — so that `None` arm
/// is unreachable from `watchdog_loop` itself; it exists for this function's own
/// direct callers (defensive default, and exercised by
/// `coder_activity_stale_no_heartbeat_is_immediately_stale` below).
pub fn coder_activity_stale(now: u64, last_heartbeat: Option<u64>, timeout: u64) -> bool {
    match last_heartbeat {
        None => true,
        Some(hb) => now.saturating_sub(hb) >= timeout,
    }
}

/// CHRD-135: the full per-tick watchdog decision, combining the ABSOLUTE watchdog
/// deadline ([`watchdog_should_activate`]) with the CODER-ACTIVITY (heartbeat)
/// window. `watchdog_should_activate` alone lets a live compiler lease defer the
/// deadline UNCONDITIONALLY — this closes that gap: a compiler lease may defer
/// PAST the absolute deadline only as long as it keeps heartbeating within
/// `coder_activity_timeout`. The moment its heartbeat goes stale
/// ([`coder_activity_stale`]), the watchdog activates regardless of the absolute
/// deadline — so silence, not elapsed wall-clock time, is what bounds the defer.
/// A non-compiler, non-coder-activity holder (or no holder) is unaffected — the
/// absolute deadline alone governs that case exactly as before. Pure — fully
/// unit-testable offline, and used by both `watchdog_loop` and its tests so the
/// tested logic IS the production logic.
///
/// CHRD-135 round-4: `coder_activity_patterns` is a SEPARATE, independently
/// configured set ([`DEFAULT_CODER_ACTIVITY_HOLDERS`]/
/// [`coder_activity_holders_from_env`]) used ONLY here, in the floor branch —
/// [`watchdog_should_activate`] above is called with `patterns` (the COMPILER
/// set) exactly as before and is otherwise untouched. This closes the reachability
/// gap where the floor branch's `is_compiler` check used the SAME compiler
/// pattern set as the unconditional defer rule: every real production
/// `GPU_EXCLUSIVE` holder label (`intake_coder_sweep`, `intake_coder_case`,
/// `intake_assistant_sweep`, `mint_breakfix`) contains none of
/// `compiler`/`build`/`bld`, so `is_compiler` was always `false` for real
/// traffic and this whole floor was dead code in production — governed instead
/// solely by the blind absolute `deadline_expired` one-shot timer. Adding the
/// `is_coder_activity` disjunct only ever WIDENS when this function returns
/// `true`; it can never turn a `true` into a `false`, so this change cannot make
/// Chord stay idle any LONGER than it already does today — it can only make a
/// tick that would have waited for the absolute deadline instead activate
/// earlier, on genuine coder silence past `coder_activity_timeout`.
pub fn watchdog_tick_should_activate(
    now: u64,
    deadline_expired: bool,
    holder: Option<&str>,
    holder_last_heartbeat: Option<u64>,
    patterns: &[String],
    coder_activity_patterns: &[String],
    coder_activity_timeout: u64,
) -> bool {
    if watchdog_should_activate(deadline_expired, holder, patterns) {
        return true;
    }
    let is_compiler = holder
        .map(|h| is_compiler_lease(h, patterns))
        .unwrap_or(false);
    let is_coder_activity = holder
        .map(|h| is_coder_activity_holder(h, coder_activity_patterns))
        .unwrap_or(false);
    (is_compiler || is_coder_activity)
        && coder_activity_stale(now, holder_last_heartbeat, coder_activity_timeout)
}

/// CHORD-ACT-01 pure view: given the in-flight count, the last-activity epoch, and
/// `now`, derive `(serving, idle_secs)` for `GET /admin/activity`. `serving` is simply
/// "something is in flight"; `idle_secs` is `0` while serving, else `now - last_activity`
/// clamped at `0` (a clock that went backwards, or a future stamp, never yields a
/// negative age). Pure and clock-free so it is exhaustively unit-testable.
pub fn activity_summary(inflight: usize, last_activity_unix: i64, now: i64) -> (bool, u64) {
    let serving = inflight > 0;
    let idle_secs = if serving {
        0
    } else {
        now.saturating_sub(last_activity_unix).max(0) as u64
    };
    (serving, idle_secs)
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
/// detect and force-resolve a wedged transition, plus a unique `generation` token so
/// a stale transition guard can prove it still owns the CURRENT transition before
/// finalizing it — preventing a dropped/committed stale guard from clobbering a newer
/// transition (the ABA hazard).
enum IdleState {
    Active,
    EnteringIdle {
        since: u64,
        generation: u64,
    },
    Idle(ResumeManifest),
    Activating {
        since: u64,
        generation: u64,
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
            IdleState::EnteringIdle { since, .. } | IdleState::Activating { since, .. } => {
                Some(*since)
            }
            _ => None,
        }
    }
}

/// Result of trying to BEGIN entering idle (CAS `Active → EnteringIdle`).
#[derive(Debug, PartialEq)]
pub enum BeginEnter {
    /// Won the CAS: caller MUST run release work then commit the transition (via the
    /// RAII [`EnterTransition`] guard's `commit`).
    Begin,
    /// Already fully idle ⇒ idempotent no-op (carries the existing manifest).
    AlreadyIdle(ResumeManifest),
    /// Another enter/activate transition is in flight ⇒ no-op, do NOT run release.
    InTransition,
}

/// Result of trying to BEGIN activating (CAS `Idle → Activating`).
#[derive(Debug, PartialEq)]
pub enum BeginActivate {
    /// Won the CAS: caller finishes the activate transition (see [`IdleController::exit`]
    /// / [`activate`]).
    Begin(ResumeManifest),
    /// Already active ⇒ idempotent no-op.
    AlreadyActive,
    /// An enter/activate transition is in flight ⇒ no-op.
    InTransition,
    /// A state path is configured but clearing the persisted manifest to `Active`
    /// FAILED, so the activate was ABORTED before touching memory — the controller
    /// stays `Idle` (recoverable; a crash would reload `Idle`, consistent with memory).
    /// The caller should surface a retryable error and try again.
    PersistFailed,
    /// CHRD-135 defect #3: an episode-guarded activate
    /// ([`IdleController::begin_activate_for_episode`]) found the controller `Idle`
    /// on a DIFFERENT episode than the one the caller's decision was made about —
    /// i.e. a newer idle episode has been installed since. No-op: the caller's
    /// stale decision must not cancel it.
    Superseded,
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
    /// Activate aborted because the persist-Active-before-restore hard gate failed;
    /// the controller stays `Idle` and the caller should retry (see
    /// [`BeginActivate::PersistFailed`]).
    PersistFailed,
}
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
    /// CHORD-ACT-01: epoch-seconds of the most recent inference ADMISSION (the moment
    /// `try_admit` last increments `inflight`). Seeded to construction time so
    /// "idle for N seconds" is meaningful from boot even before the first request.
    /// Read by `GET /admin/activity`; a single relaxed-cost `SeqCst` store on the hot
    /// admission path, no lock of its own (it piggybacks the admission write lock).
    last_activity: AtomicI64,
    /// Monotonic generation counter. Each `begin_enter`/`begin_activate` mints a fresh
    /// generation (under the `inner` write lock) that is stamped into the transient
    /// phase and captured by the transition's guard. `finish`/`abort` only act while
    /// that same generation is still the live one, so a stale guard (whose transition
    /// was force-recovered by the watchdog and superseded by a newer one) becomes a
    /// no-op instead of clobbering the newer transition (the ABA hazard).
    next_gen: AtomicU64,
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
            last_activity: AtomicI64::new(now_epoch() as i64),
            next_gen: AtomicU64::new(0),
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
            last_activity: AtomicI64::new(now_epoch() as i64),
            next_gen: AtomicU64::new(0),
            state_path,
        }
    }

    /// From `CHORD_STATE_DIR` (see [`crate::config::admin_idle_state_path`]).
    pub fn from_env() -> Self {
        Self::with_state(crate::config::admin_idle_state_path())
    }

    /// Best-effort persist for the non-critical transitions (finish/abort/recover):
    /// a failure here is logged and swallowed because those all move TOWARD a resting
    /// phase and a stale on-disk state is bounded by the watchdog. The one persist that
    /// is NOT best-effort — clearing to `Active` before the activate restore — is
    /// hard-gated directly in [`begin_activate_inner`](Self::begin_activate_inner).
    fn persist_locked(&self, current: &IdleState) {
        if let Some(path) = self.state_path.as_deref() {
            if let Err(e) = persist_state(path, &current.to_persisted().cloned()) {
                warn!(path = %path.display(), error = %e,
                    "idle-mode: state persist failed (best-effort — watchdog still bounds it)");
            }
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
                // CHORD-ACT-01: stamp last-activity at the instant admission increments
                // the in-flight count — one cheap atomic store under the lock we already
                // hold. This is THE inference-request start for every admitted call
                // (chat/completions, embeddings, agentic), so `/admin/activity` reflects
                // real serving without a second counter or a second call site.
                self.last_activity
                    .store(now_epoch() as i64, Ordering::SeqCst);
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

    /// CHORD-ACT-01: epoch-seconds of the most recent inference admission (or the
    /// controller's construction time if it has never served). Read by
    /// `GET /admin/activity`; see [`activity_summary`] for the derived serving/idle view.
    pub fn last_activity_unix(&self) -> i64 {
        self.last_activity.load(Ordering::SeqCst)
    }

    /// Test-only override of the last-activity stamp so the activity view can be
    /// exercised deterministically without sleeping or touching the wall clock.
    #[cfg(test)]
    fn set_last_activity_for_test(&self, unix: i64) {
        self.last_activity.store(unix, Ordering::SeqCst);
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

    /// Internal CAS `Active → EnteringIdle{generation}`. Mints a fresh generation
    /// under the write lock and stamps it into the transient phase. `Ok(gen)` on a
    /// real transition (caller must eventually finish/abort with that `gen`); `Err`
    /// carries the non-transition outcome. Shared by [`begin_enter`](Self::begin_enter)
    /// and [`try_begin_enter`](Self::try_begin_enter) so both mint a generation.
    fn begin_enter_inner(&self) -> Result<u64, BeginEnter> {
        let mut guard = self.inner.write().expect("idle lock poisoned");
        match &*guard {
            IdleState::Active => {
                let generation = self.next_gen.fetch_add(1, Ordering::SeqCst);
                *guard = IdleState::EnteringIdle {
                    since: now_epoch(),
                    generation,
                }; // transient — persists as "not idle" (Active) if persisted at all
                Ok(generation)
            }
            IdleState::Idle(m) => Err(BeginEnter::AlreadyIdle(m.clone())),
            IdleState::EnteringIdle { .. } | IdleState::Activating { .. } => {
                Err(BeginEnter::InTransition)
            }
        }
    }

    /// CAS `Active → EnteringIdle`. Installs the transition marker atomically BEFORE
    /// any release work, so exactly one caller ever runs the release side effects.
    /// Prefer [`try_begin_enter`](Self::try_begin_enter), whose RAII guard guarantees
    /// the phase is finalized even if the enter future is dropped mid-transition.
    pub fn begin_enter(&self) -> BeginEnter {
        match self.begin_enter_inner() {
            Ok(_gen) => BeginEnter::Begin,
            Err(e) => e,
        }
    }

    /// CAS into `EnteringIdle` and return an RAII [`EnterTransition`] guard carrying
    /// this transition's generation. The guard MUST be finalized with
    /// [`EnterTransition::commit`]; if it is instead dropped (the enter future is
    /// cancelled on client disconnect, panics, or returns early), its `Drop`
    /// deterministically rolls `EnteringIdle → Active` — BUT only if its generation is
    /// still the live one, so a stale guard can never roll back a newer transition.
    /// `Err` carries the non-`Begin` CAS result.
    pub fn try_begin_enter(&self) -> Result<EnterTransition<'_>, BeginEnter> {
        match self.begin_enter_inner() {
            Ok(generation) => Ok(EnterTransition {
                ctl: self,
                generation,
                committed: false,
            }),
            Err(e) => Err(e),
        }
    }

    /// Complete an enter: `EnteringIdle{gen} → Idle(manifest)`, persisting the manifest.
    /// GENERATION-GUARDED: only installs the manifest if the controller is STILL in the
    /// same-generation `EnteringIdle` this transition began — otherwise (watchdog
    /// recovered it, or a newer transition superseded it) it is a NO-OP returning
    /// `None`, so a stale commit can never clobber a newer phase.
    fn finish_enter(
        &self,
        generation: u64,
        mut manifest: ResumeManifest,
    ) -> Option<ResumeManifest> {
        // CHRD-135 defect #3: this transition's own generation IS the episode
        // identity — always overwrite whatever the caller set (see the doc on
        // `ResumeManifest::episode`).
        manifest.episode = generation;
        let mut guard = self.inner.write().expect("idle lock poisoned");
        match &*guard {
            IdleState::EnteringIdle {
                generation: cur, ..
            } if *cur == generation => {
                *guard = IdleState::Idle(manifest.clone());
                self.persist_locked(&guard);
                Some(manifest)
            }
            _ => None,
        }
    }

    /// Roll an in-progress `EnteringIdle` back to `Active` (the safe resting phase).
    /// Used by the [`EnterTransition`] guard when an enter is dropped before commit.
    /// GENERATION-GUARDED: only acts while the controller is STILL in the same-generation
    /// `EnteringIdle` this transition installed. If the phase advanced (committed), was
    /// recovered by the watchdog, or was superseded by a newer transition, it is a
    /// NO-OP — so a stale/late drop can never clobber a newer transition's state.
    fn abort_enter(&self, generation: u64) {
        let mut guard = self.inner.write().expect("idle lock poisoned");
        if let IdleState::EnteringIdle {
            generation: cur, ..
        } = &*guard
        {
            if *cur == generation {
                *guard = IdleState::Active;
                self.persist_locked(&guard);
            }
        }
    }

    /// Internal CAS `Idle → Activating{generation}`, returning the manifest to clear
    /// plus the transition's generation. FINDING #2: clearing the persisted manifest to
    /// the RESTING `Active`/none form is a HARD PREREQUISITE done BEFORE mutating memory
    /// — while the controller is still `Idle`. When a state path is configured and that
    /// persist FAILS, the activate is ABORTED (`Err(PersistFailed)`) with memory left
    /// `Idle`, so on-disk and memory stay consistent (a crash reloads `Idle`,
    /// recoverable) — we never proceed into restore with the disk still saying `Idle`
    /// while memory says `Activating`. When no state path is configured, persistence is
    /// a no-op and this gate is skipped (best-effort, as before). Only after the disk
    /// reads `Active` do we flip memory to `Activating`.
    /// CHRD-135 defect #3: shared implementation for [`begin_activate_inner`]
    /// (`expected_episode: None`, unconditional — the admin endpoint and
    /// lazy-on-request path) and [`begin_activate_for_episode`]
    /// (`expected_episode: Some(_)` — the watchdog path only). The episode check
    /// and the rest of the CAS happen under ONE write-lock acquisition, so there is
    /// no gap between "is this still the expected episode" and "commit the
    /// transition" for a second caller to land in.
    fn begin_activate_inner_impl(
        &self,
        expected_episode: Option<u64>,
    ) -> Result<(ResumeManifest, u64), BeginActivate> {
        let mut guard = self.inner.write().expect("idle lock poisoned");
        match &*guard {
            IdleState::Idle(m) => {
                if let Some(expected) = expected_episode {
                    if m.episode != expected {
                        return Err(BeginActivate::Superseded);
                    }
                }
                // HARD GATE: clear disk → Active BEFORE touching memory (still `Idle`).
                if let Some(path) = self.state_path.as_deref() {
                    if let Err(e) = persist_state(path, &None) {
                        warn!(path = %path.display(), error = %e,
                            "idle-mode: could not clear persisted idle state before activate — \
                             aborting activate (staying Idle, retryable)");
                        return Err(BeginActivate::PersistFailed);
                    }
                }
                // Disk now reads `Active`; safe to flip memory to `Activating`.
                let m = match std::mem::replace(&mut *guard, IdleState::Active) {
                    IdleState::Idle(m) => m,
                    _ => unreachable!("matched Idle above"),
                };
                let generation = self.next_gen.fetch_add(1, Ordering::SeqCst);
                *guard = IdleState::Activating {
                    since: now_epoch(),
                    generation,
                    manifest: m.clone(),
                };
                Ok((m, generation))
            }
            IdleState::Active => Err(BeginActivate::AlreadyActive),
            IdleState::EnteringIdle { .. } | IdleState::Activating { .. } => {
                Err(BeginActivate::InTransition)
            }
        }
    }

    fn begin_activate_inner(&self) -> Result<(ResumeManifest, u64), BeginActivate> {
        self.begin_activate_inner_impl(None)
    }

    /// CHRD-135 defect #3: like [`begin_activate_inner`](Self::begin_activate_inner),
    /// but only wins the CAS if the live `Idle` manifest's `episode` still matches
    /// `expected_episode` — see [`ResumeManifest::episode`] and
    /// [`BeginActivate::Superseded`]. Used exclusively by the watchdog so a decision
    /// made about episode N can never activate (and so cancel) episode N+1.
    fn begin_activate_for_episode(
        &self,
        expected_episode: u64,
    ) -> Result<(ResumeManifest, u64), BeginActivate> {
        self.begin_activate_inner_impl(Some(expected_episode))
    }

    /// CAS `Idle → Activating`, returning the manifest to clear. Concurrent
    /// activates: exactly one wins `Begin`; the rest see `InTransition`/`AlreadyActive`.
    pub fn begin_activate(&self) -> BeginActivate {
        match self.begin_activate_inner() {
            Ok((m, _gen)) => BeginActivate::Begin(m),
            Err(e) => e,
        }
    }

    /// Complete an activate: `Activating{gen} → Active`. GENERATION-GUARDED: only acts
    /// while the controller is STILL in the same-generation `Activating` this transition
    /// began; otherwise a NO-OP. Returns whether it finalized. (Disk already reads
    /// `Active` from [`begin_activate_inner`], so this is a memory-only resolution.)
    fn finish_activate(&self, generation: u64) -> bool {
        let mut guard = self.inner.write().expect("idle lock poisoned");
        if let IdleState::Activating {
            generation: cur, ..
        } = &*guard
        {
            if *cur == generation {
                *guard = IdleState::Active;
                self.persist_locked(&guard);
                return true;
            }
        }
        false
    }

    /// Backstop: if the controller has been stuck in a TRANSIENT phase
    /// (`EnteringIdle`/`Activating`) since before `now - max_age`, force-resolve it to
    /// `Active` and return `true`. Never touches a steady `Active`/`Idle` phase. Bumps
    /// the generation so any outstanding guard for the recovered transition is
    /// invalidated (its finish/abort become no-ops). Insurance behind the RAII guard
    /// for the (should-be-impossible) case of a genuinely wedged transition; the
    /// watchdog calls it each tick.
    pub fn recover_stale_transition(&self, now: u64, max_age: u64) -> bool {
        let mut guard = self.inner.write().expect("idle lock poisoned");
        let Some(since) = guard.transition_since() else {
            return false;
        };
        if now.saturating_sub(since) >= max_age {
            // Bump the generation: the abandoned transition's guard now holds a stale
            // generation, so its Drop/commit can't clobber whatever comes next.
            self.next_gen.fetch_add(1, Ordering::SeqCst);
            *guard = IdleState::Active;
            self.persist_locked(&guard);
            true
        } else {
            false
        }
    }

    /// Convenience full enter used by unit tests: atomically `Active → Idle` (begin +
    /// finish with no release work in between, via the RAII guard). Idempotent, like
    /// the real path.
    pub fn enter(&self, manifest: ResumeManifest) -> EnterOutcome {
        match self.try_begin_enter() {
            Ok(t) => match t.commit(manifest) {
                Some(m) => EnterOutcome::Entered(m),
                None => EnterOutcome::InTransition,
            },
            Err(BeginEnter::AlreadyIdle(m)) => EnterOutcome::AlreadyIdle(m),
            Err(_) => EnterOutcome::InTransition,
        }
    }

    /// Full leave idle (begin + finish). Idempotent: already active ⇒ `AlreadyActive`.
    pub fn exit(&self) -> ActivateOutcome {
        match self.begin_activate_inner() {
            Ok((m, generation)) => {
                self.finish_activate(generation);
                ActivateOutcome::Activated(m)
            }
            Err(BeginActivate::AlreadyActive) => ActivateOutcome::AlreadyActive,
            Err(BeginActivate::PersistFailed) => ActivateOutcome::PersistFailed,
            Err(_) => ActivateOutcome::InTransition,
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
    /// The generation minted for THIS transition. `commit`/`Drop` only act while this
    /// is still the controller's live `EnteringIdle` generation (ABA guard).
    generation: u64,
    committed: bool,
}

impl EnterTransition<'_> {
    /// Complete the transition: `EnteringIdle{gen} → Idle(manifest)`. Consumes the
    /// guard so its `Drop` becomes a no-op. Returns `Some(manifest)` if this transition
    /// still owned the live generation, or `None` if it had been superseded/recovered
    /// (in which case nothing was installed — the caller must not report success).
    pub fn commit(mut self, manifest: ResumeManifest) -> Option<ResumeManifest> {
        self.committed = true;
        self.ctl.finish_enter(self.generation, manifest)
    }
}

impl Drop for EnterTransition<'_> {
    fn drop(&mut self) {
        if !self.committed {
            // Generation-guarded inside `abort_enter`: a stale guard (its transition
            // recovered by the watchdog and superseded by a newer one) is a no-op here.
            self.ctl.abort_enter(self.generation);
            warn!(
                "idle-mode: enter transition dropped before commit (future cancelled/panicked) \
                 — rolled EnteringIdle back to Active (if still the live transition)"
            );
        }
    }
}

// Manual `Debug` (the held `&IdleController` isn't `Debug`, so we can't derive):
// prints just the completion marker, which is all a test diagnostic needs.
impl std::fmt::Debug for EnterTransition<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnterTransition")
            .field("generation", &self.generation)
            .field("committed", &self.committed)
            .finish_non_exhaustive()
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

/// Atomically persist the current state (tempfile + rename). Returns the IO error on
/// failure so the CALLER can decide whether it is fatal: most callers treat it as
/// best-effort (see [`IdleController::persist_locked`]), but the activate path
/// hard-gates on it (see [`IdleController::begin_activate_inner`]).
fn persist_state(path: &Path, state: &Option<ResumeManifest>) -> std::io::Result<()> {
    let json = serde_json::to_string(state)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json.as_bytes())?;
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
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

    // FINDING #1: bound the ENTIRE release sequence with an explicit budget kept
    // STRICTLY BELOW the stale-recovery threshold. If release overruns, the timeout
    // fires, we do NOT commit, and the RAII `transition` guard drops → clean rollback
    // to `Active`. Admission was CLOSED for the whole `EnteringIdle` window and only
    // reopens AFTER that consistent rollback — never with half-stopped backends still
    // admitting, and never concurrently with the watchdog's stale-recovery (which, by
    // the config ordering invariant, cannot fire before this budget elapses).
    let release_budget = Duration::from_secs(release_budget_secs_from_env());
    let release = tokio::time::timeout(release_budget, async {
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

        // 4b. CHRD-DIFF-01: stop the Chord-managed DiffusionGemma daemon too —
        //     it isn't a ModelRegistry `Backend` (step 4 doesn't see it) and
        //     isn't tracked by Ollama's /api/ps (step 5 below doesn't see it
        //     either). Idle-mode release must free its VRAM the same as every
        //     other on-demand backend.
        if crate::diffusion::global().stop().await {
            info!("idle-mode: stopped managed DiffusionGemma daemon");
        }

        // 4c. TRTR-07: RELEASE the assistant-mode resident set (embedding /
        //     router / personality). Sequenced deliberately:
        //       - AFTER the closed-world drain in step 1, so an in-flight turn
        //         always finishes — we never yank a model out from under a live
        //         request mid-generation;
        //       - BEFORE the VRAM unload in step 5, because the release is what
        //         drops the eviction exemption; without it step 5's unload would
        //         be immediately undone by the sweep re-protecting these models.
        //     Idempotent and best-effort: an already-released set is a no-op.
        let resident_released = crate::routing::resident_set::global()
            .release(state, reason)
            .await;
        if resident_released.changed {
            info!(
                released = resident_released.released,
                "idle-mode: assistant resident set released for the mode swap"
            );
        }

        // 5. Unload every resident model from VRAM (best-effort; skipped if OLLAMA_URL
        //    unset). This is the real "release the GPU + the RAM the models held".
        let models_unloaded = match crate::gpu_exclusive::ollama_base_from_env() {
            Some(base) => {
                // Idle-mode is a WHOLE-GPU release (the fleet is idle, the assistant
                // included) — unload EVERYTHING. Step 4c above already released the
                // resident set, which is the ONLY thing that holds a model here
                // (CHRD-PIN-01 retired the separate keep-resident pin), so this
                // unload is never fighting another owner.
                crate::gpu_exclusive::evict_resident_models(&state.http_client, &base).await
            }
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
        // `resident_models` goes into the manifest (not the report); hand both back.
        (report, resident_models)
    })
    .await;

    let (report, resident_models) = match release {
        Ok(pair) => pair,
        Err(_elapsed) => {
            warn!(
                reason,
                budget_secs = release_budget.as_secs(),
                "idle-mode: release exceeded its budget — aborting enter; guard rolls EnteringIdle back to Active (admission was closed throughout)"
            );
            // `transition` drops here → clean rollback to Active; admission reopens
            // only now, after a consistent rollback.
            return (EnterOutcome::InTransition, None);
        }
    };

    let now = now_epoch();
    let manifest = ResumeManifest {
        reason: reason.to_string(),
        entered_at: now,
        watchdog_deadline: now.saturating_add(watchdog_secs_from_env()),
        resident_models,
        mem_available_before_gb: report.mem_available_before_gb.unwrap_or(0.0),
        // Overwritten by `finish_enter` with this transition's own generation —
        // see `ResumeManifest::episode`.
        episode: 0,
    };
    // Complete the transition: `EnteringIdle{gen} → Idle` (consumes the guard so its
    // Drop rollback becomes a no-op). Generation-guarded: if our transition was
    // force-recovered by the watchdog while we were releasing (took longer than the
    // stale bound), `commit` returns `None` — our enter did NOT take effect and we
    // must not claim success, so report `InTransition` with no freed-RAM report.
    let Some(stored) = transition.commit(manifest) else {
        warn!(
            reason,
            "idle-mode: enter superseded (watchdog recovered a stale transition mid-release) — not reporting Entered"
        );
        return (EnterOutcome::InTransition, None);
    };

    info!(
        reason,
        backends_stopped = report.backends_stopped,
        models_unloaded = report.models_unloaded,
        models_demoted = report.models_demoted,
        freed_gb = report.freed_gb.unwrap_or(0.0),
        "idle-mode: entered — host resources released for the compiler"
    );
    (EnterOutcome::Entered(stored), Some(report))
}

/// Leave idle and resume normal serving. Idempotent, CAS-guarded. Ordinary models
/// reload LAZILY on their next request (Ollama/llama-server cold-load on demand),
/// so the state transition itself is effectively atomic — `Activating` is a
/// nanosecond window.
///
/// TRTR-07: the ONE eager step is the assistant-mode resident set. The idle lease
/// that just ended is the mode swap back to assistant mode, so the three role
/// models (personality / router / embedding) are re-warmed here rather than
/// waiting to be cold-loaded by the human's next turn. It is debounced (a rapid
/// lease acquire/release cycle must not thrash-warm), best-effort, and fired
/// AFTER the transition completes so a slow warm can never widen the `Activating`
/// window or fail the activate.
pub async fn activate(state: &Arc<crate::routes::AppState>, reason: &str) -> ActivateOutcome {
    let outcome = activate_inner(state, reason).await;
    if matches!(outcome, ActivateOutcome::Activated(_)) {
        let report = crate::routing::resident_set::global()
            .rewarm(state, reason)
            .await;
        if report.changed {
            info!(
                reason,
                warmed = report.warmed,
                dropped = report.dropped,
                failed = report.failed,
                "idle-mode: assistant resident set re-warmed after the mode swap"
            );
        }
    }
    outcome
}

/// The state-machine half of [`activate`], with no resident-set side effect —
/// kept separate so the transition stays a tight, testable CAS.
async fn activate_inner(state: &Arc<crate::routes::AppState>, reason: &str) -> ActivateOutcome {
    let _ = state;
    match IDLE_MODE.begin_activate_inner() {
        Ok((m, generation)) => {
            // (restore side effects would go here; lazy reload means none today.) Disk
            // already reads `Active` from begin_activate_inner; finish_activate resolves
            // memory `Activating{gen} → Active`. Generation-guarded, so a concurrent
            // recovery can't be clobbered.
            IDLE_MODE.finish_activate(generation);
            info!(
                reason,
                resident_models = m.resident_models.len(),
                "idle-mode: activated — normal serving resumed (models reload on demand)"
            );
            ActivateOutcome::Activated(m)
        }
        Err(BeginActivate::AlreadyActive) => ActivateOutcome::AlreadyActive,
        Err(BeginActivate::PersistFailed) => ActivateOutcome::PersistFailed,
        Err(_) => ActivateOutcome::InTransition,
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
                    if activate(state, "lazy-on-request").await == ActivateOutcome::PersistFailed {
                        // Couldn't clear persisted idle safely → stay Idle; shed with a
                        // retryable 503 rather than looping on a persist that keeps failing.
                        return Admission::Rejected(idle_transition_response());
                    }
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
        // Persist-Active-before-restore hard gate failed: the proxy stayed Idle
        // (recoverable). Signal a retryable error so the caller retries.
        ActivateOutcome::PersistFailed => (
            StatusCode::SERVICE_UNAVAILABLE,
            [(header::RETRY_AFTER, HeaderValue::from_static("2"))],
            Json(serde_json::json!({
                "error": "idle_activate_persist_failed",
                "status": "idle",
                "changed": false,
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

/// CHORD-ACT-01 response body, factored out of the handler so its exact shape is unit
/// tested without an HTTP round-trip. Reuses [`activity_summary`] so the wire contract
/// and the pure logic can never drift.
fn activity_json(inflight: usize, last_activity_unix: i64, now: i64) -> serde_json::Value {
    let (serving, idle_secs) = activity_summary(inflight, last_activity_unix, now);
    serde_json::json!({
        "serving": serving,
        "inflight": inflight,
        "idle_secs": idle_secs,
        "last_request_unix": last_activity_unix,
    })
}

/// `GET /admin/activity` — live serving-activity signal (CHORD-ACT-01). Auth-gated with
/// the SAME posture as `/admin/idle`. Distinct from the idle-MODE phase reported there:
/// this answers "is Chord actually serving inference right now, and if not, for how long
/// has it been quiet?" so the constellation compiler scheduler can dispatch heavy builds
/// only into genuine quiet windows without being permanently blocked by a busy proxy.
/// Reuses the one in-flight counter the idle machinery owns — no second source of truth.
pub async fn admin_activity_status(
    State(state): State<StdArc<AppState>>,
    headers: HeaderMap,
) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }
    let inflight = IDLE_MODE.inflight_count();
    let last_activity_unix = IDLE_MODE.last_activity_unix();
    let now = now_epoch() as i64;
    (
        StatusCode::OK,
        Json(activity_json(inflight, last_activity_unix, now)),
    )
        .into_response()
}

// ── Watchdog ──────────────────────────────────────────────────────────────────

/// Background fail-safe: every `interval`, if idle and the watchdog deadline has
/// passed AND no COMPILER build lease is currently held, auto-activate so the proxy
/// is never left silently dead (a crashed/forgotten compiler, or a stale idle state
/// reloaded after a Chord restart). While a compiler build lease IS held the deadline
/// is deferred — a legitimately long build keeps Chord idle as long as it holds the
/// GPU. A NON-compiler GPU holder (e.g. the intake sweep harness) does NOT extend the
/// idle window. Follows the `idle_stop_sweep` spawn pattern in `main.rs`.
/// CHRD-135: the per-tick STATE-MACHINE step — decision plus its CAS effect —
/// taking `ctl`/`gpu` as explicit parameters so it runs identically against
/// isolated test instances and the process globals, and so a test exercising it
/// observes the SAME wiring `watchdog_loop` runs each tick, not a hand-rolled
/// substitute (review, round 1, flagged exactly that gap in the original
/// behavioral test). Returns `Some(cleared manifest)` if it activated
/// (`Idle -> Active`), else `None` — deferred, nothing idle to act on, or the
/// decision was superseded by a newer episode.
///
/// Defect #1 (review, round 1): a holder-liveness check must not treat an
/// abandoned GPU_EXCLUSIVE record (heartbeat that stopped updating) as a live
/// compiler holder deferring the revert forever — raw `.snapshot()` with the
/// `expired` flag discarded did exactly that.
///
/// Defect #1 residual (review, round 2): fixing that by switching to
/// `gpu.active_holder(now)` introduced a NEW bug — `active_holder` expires a
/// record against GPU_EXCLUSIVE's OWN `ttl` ([`crate::gpu_exclusive::
/// DEFAULT_TTL_SECS`], 600s by default), which is UNRELATED to and shorter than
/// `coder_activity_timeout` (900s by default). Once the absolute deadline has
/// passed, `active_holder` returning `None` at 600s of silence — well before the
/// promised 900s floor — let the watchdog revert as early as 600s, silently
/// violating "auto-revert only after AT LEAST 15 minutes of no coder activity."
/// Fixed by computing liveness against `effective_ttl =
/// max(gpu.ttl(), coder_activity_timeout)` here, LOCAL to this decision only —
/// GPU_EXCLUSIVE's own (shorter) `ttl` still governs whether some OTHER caller
/// may steal the lock; that is a separate, correctly-shorter concern this
/// function does not touch. This guarantees the coder-activity floor documented
/// on [`DEFAULT_CODER_ACTIVITY_TIMEOUT_SECS`] is never undercut by
/// GPU_EXCLUSIVE's own TTL, however the two are configured.
///
/// Defect #3 (review, round 1): activation is EPISODE-GUARDED
/// (`begin_activate_for_episode`), pinned to the `episode` captured in the SAME
/// snapshot the decision was computed from. If a newer idle episode has been
/// installed by the time the CAS actually runs, this is a no-op — a stale
/// decision about a FORMER episode can never cancel a fresh one. This narrows,
/// but does not fully eliminate, the analogous race against an external
/// GPU_EXCLUSIVE re-acquire landing between the snapshot above and this CAS —
/// closing that would require a real cross-subsystem lock the two globals don't
/// share today; tracked as a known residual, not solved here.
fn watchdog_tick(
    ctl: &IdleController,
    gpu: &crate::gpu_exclusive::GpuExclusive,
    now: u64,
    patterns: &[String],
    coder_activity_patterns: &[String],
    coder_activity_timeout: u64,
) -> Option<ResumeManifest> {
    let m = ctl.snapshot()?;
    // Defect #1 residual (review, round 2): never let GPU_EXCLUSIVE's own,
    // possibly-shorter TTL undercut the coder-activity floor — see the doc above.
    let effective_ttl = gpu.ttl().max(coder_activity_timeout);
    let holder = gpu.snapshot(now).and_then(|(r, _gpu_own_expired)| {
        if r.is_expired(now, effective_ttl) {
            None
        } else {
            Some(r)
        }
    });
    let holder_label = holder.as_ref().map(|r| r.holder.as_str());
    let holder_last_heartbeat = holder.as_ref().map(|r| r.last_heartbeat);
    if !watchdog_tick_should_activate(
        now,
        m.watchdog_expired(now),
        holder_label,
        holder_last_heartbeat,
        patterns,
        coder_activity_patterns,
        coder_activity_timeout,
    ) {
        return None;
    }
    match ctl.begin_activate_for_episode(m.episode) {
        Ok((manifest, generation)) => {
            ctl.finish_activate(generation);
            Some(manifest)
        }
        Err(BeginActivate::AlreadyActive) | Err(BeginActivate::InTransition) => None,
        Err(BeginActivate::Superseded) => {
            // Expected/benign: a newer idle episode is now current, see the doc above.
            None
        }
        Err(BeginActivate::PersistFailed) => {
            // Non-blocking review finding #4: this WAS a silent `let _ =` discard —
            // log it, since a persistently failing persist would otherwise retry
            // every tick with no diagnostic at all.
            warn!(
                "idle-mode watchdog: episode-guarded activate aborted (persist-to-Active \
                 failed) — will retry next tick"
            );
            None
        }
        Err(BeginActivate::Begin(_)) => {
            // `BeginActivate::Begin` is only ever constructed by the OTHER wrapper
            // (`IdleController::begin_activate`, the public non-episode-guarded
            // entry point used by the HTTP handler) around a *successful* CAS — it
            // is a `BeginActivate` value returned directly, never an `Err` payload
            // out of `begin_activate_inner_impl`/`begin_activate_for_episode`
            // (those return `Ok((ResumeManifest, u64))` on success instead), so
            // this arm is unreachable from this path TODAY.
            //
            // Review, round 2: `unreachable!()` here is fail-DEADLY, not
            // fail-safe — `watchdog_loop` is a spawned task with no supervisor,
            // so a panic on an invariant that's unreachable only by convention
            // (not by the type system) would permanently kill the exact
            // mechanism this ticket exists to guarantee: the return to default
            // mode. Degrade like the sibling `PersistFailed` arm instead — log
            // and retry next tick — at zero behavioral cost today.
            warn!(
                "idle-mode watchdog: begin_activate_for_episode returned an unexpected \
                 BeginActivate::Begin(_) payload — treating as a no-op and retrying next tick"
            );
            None
        }
    }
}

pub async fn watchdog_loop(state: Arc<crate::routes::AppState>, interval: Duration) {
    info!(
        interval_secs = interval.as_secs(),
        "idle-mode watchdog started"
    );
    let patterns = compiler_lease_holders_from_env();
    let coder_activity_patterns = coder_activity_holders_from_env();
    let stale_secs = stale_transition_secs_from_env();
    let coder_activity_timeout = coder_activity_timeout_secs_from_env();
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
        let Some(manifest) = watchdog_tick(
            &IDLE_MODE,
            &crate::gpu_exclusive::GPU_EXCLUSIVE,
            now,
            &patterns,
            &coder_activity_patterns,
            coder_activity_timeout,
        ) else {
            continue;
        };
        warn!(
            reason = %manifest.reason,
            episode = manifest.episode,
            "idle-mode watchdog: deadline passed (or the compiler lease's heartbeat went stale) \
             with no active/renewing compiler lease — auto-activated (fail-safe)"
        );
        // Same side effect `activate()` runs on a real Activated outcome (TRTR-07):
        // re-warm the assistant resident set now that we're back in serving mode.
        //
        // Audited CHRD-135 review, round 2: confirmed `activate() ==
        // begin_activate_inner() + finish_activate() + this same info! log` (see
        // `activate()`'s own "lazy reload means none today" comment) — i.e.
        // `activate() == CAS + rewarm`, exactly matching this watchdog path. This
        // hand-rolled call is NOT a divergent/incomplete substitute for
        // `activate()`; it reproduces its full effect. Re-raise only with a new
        // finding, not a re-read of the same call site.
        let report = crate::routing::resident_set::global()
            .rewarm(&state, "watchdog-timeout")
            .await;
        if report.changed {
            info!(
                reason = "watchdog-timeout",
                warmed = report.warmed,
                dropped = report.dropped,
                failed = report.failed,
                "idle-mode: assistant resident set re-warmed after the mode swap"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn manifest(reason: &str, entered_at: u64, deadline: u64) -> ResumeManifest {
        ResumeManifest {
            reason: reason.into(),
            entered_at,
            watchdog_deadline: deadline,
            resident_models: vec!["qwen3-coder:30b".into()],
            mem_available_before_gb: 12.0,
            episode: 0,
        }
    }

    fn holders() -> Vec<String> {
        vec!["compiler".into(), "build".into(), "bld".into()]
    }

    /// CHRD-135 round-5: the coder-activity pattern set for tests, DERIVED from
    /// [`DEFAULT_CODER_ACTIVITY_HOLDERS`] by construction rather than hand-copied.
    ///
    /// The round-4 version hand-listed `["coder", "assistant", "breakfix"]` while
    /// its doc claimed to mirror the default — but the default deliberately
    /// EXCLUDES `breakfix`. That is the same lying-fixture class that hid the
    /// round-3 dead-code bug: a helper that agrees with a rule production does not
    /// actually apply. Deriving it here makes drift impossible: if the default
    /// changes, every test using this helper changes with it.
    ///
    /// Parsing the constant (rather than calling
    /// [`coder_activity_holders_from_env`]) also keeps callers HERMETIC — a
    /// `CHORD_IDLE_CODER_ACTIVITY_HOLDERS` set in the test process cannot perturb
    /// them. Tests that specifically want the env-resolving behavior should call
    /// that function directly and say so.
    fn coder_holders() -> Vec<String> {
        DEFAULT_CODER_ACTIVITY_HOLDERS
            .split(',')
            .map(|s| s.trim().to_ascii_lowercase())
            .filter(|s| !s.is_empty())
            .collect()
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

    // ── CHRD-135: coder-activity (heartbeat) bounded defer ───────────────────

    #[test]
    fn coder_activity_stale_no_heartbeat_is_immediately_stale() {
        assert!(coder_activity_stale(1_000, None, 900));
    }

    #[test]
    fn coder_activity_stale_within_window_is_fresh() {
        assert!(!coder_activity_stale(1_000, Some(500), 900)); // 500s since heartbeat < 900
    }

    #[test]
    fn coder_activity_stale_at_and_past_timeout() {
        assert!(coder_activity_stale(1_400, Some(500), 900)); // exactly 900s ⇒ stale (>=)
        assert!(coder_activity_stale(2_000, Some(500), 900)); // well past
    }

    #[test]
    fn coder_activity_stale_clock_backwards_reads_fresh() {
        // now < last_heartbeat (clock skew) ⇒ saturating_sub ⇒ age 0, never
        // spuriously stale — same discipline as LockRecord::is_expired.
        assert!(!coder_activity_stale(400, Some(1_000), 900));
    }

    #[test]
    fn watchdog_tick_bounds_compiler_defer_by_heartbeat_silence() {
        // THE BUG THIS CLOSES: `watchdog_should_activate` alone defers a compiler
        // lease FOREVER regardless of how stale its heartbeat is, as long as ANY
        // holder record matching the compiler patterns exists. The combined tick
        // decision bounds that defer by genuine renewal.
        let p = holders();
        let cp = coder_holders();
        let timeout = 900u64;

        // Deadline not yet expired, heartbeat fresh (renewed 10s ago) ⇒ never activate.
        assert!(!watchdog_tick_should_activate(
            1_010,
            false,
            Some("compiler-build-7"),
            Some(1_000),
            &p,
            &cp,
            timeout
        ));

        // Deadline not yet expired, heartbeat stale (901s of silence) ⇒ ACTIVATE —
        // the coder/build has gone silent past the safety window, even though the
        // long absolute deadline hasn't been reached yet.
        assert!(watchdog_tick_should_activate(
            1_901,
            false,
            Some("compiler-build-7"),
            Some(1_000),
            &p,
            &cp,
            timeout
        ));

        // Deadline PASSED, but heartbeat still fresh (renewed 5s ago) ⇒ a live
        // long-running build (past the absolute 1h deadline) must NOT be torn
        // down mid-build — the hard safety constraint.
        assert!(!watchdog_tick_should_activate(
            5_005,
            true,
            Some("compiler-build-7"),
            Some(5_000),
            &p,
            &cp,
            timeout
        ));

        // A non-compiler holder past an EXPIRED deadline is unaffected by the
        // heartbeat check — governed by the absolute deadline alone, exactly as
        // before this change (the new coder-activity floor is additive and only
        // ever reached when `watchdog_should_activate` itself would return
        // false — see `watchdog_tick_should_activate`'s doc).
        assert!(watchdog_tick_should_activate(
            9_999,
            true,
            Some("intake_coder_sweep"),
            Some(9_990),
            &p,
            &cp,
            timeout
        ));
        // Same holder, deadline NOT expired, heartbeat fresh (9s old, well under
        // the 900s floor) ⇒ still must not activate, even though this holder now
        // matches the coder-activity pattern set too.
        assert!(!watchdog_tick_should_activate(
            9_999,
            false,
            Some("intake_coder_sweep"),
            Some(9_990),
            &p,
            &cp,
            timeout
        ));

        // No holder at all: absolute deadline alone.
        assert!(watchdog_tick_should_activate(
            9_999, true, None, None, &p, &cp, timeout
        ));
        assert!(!watchdog_tick_should_activate(
            9_999, false, None, None, &p, &cp, timeout
        ));
    }

    #[test]
    fn coder_activity_holders_default_matches_real_production_gpu_labels() {
        // CHRD-135 round-4: THE REGRESSION GUARD. Every prior round's fixture used
        // a synthetic holder like "compiler-build-42" that can never occur in
        // production — a lying fixture that agreed with a rule real traffic never
        // exercises. This test pins the ACTUAL holder label strings the four real
        // MINT GPU_EXCLUSIVE callers acquire (Terminus
        // src/intake/{coder_sweep.rs,coder_case.rs,assistant/runner.rs,
        // breakfix.rs}, enumerated together by `mint_gpu_holders_from_env()`) and
        // asserts each is recognized by the default coder-activity pattern set —
        // i.e. each one actually REACHES the floor branch in
        // `watchdog_tick_should_activate`, not just some synthetic stand-in for it.
        // Round-5: derived from the CONSTANT, not from the env-resolving wrapper,
        // so a `CHORD_IDLE_CODER_ACTIVITY_HOLDERS` set in the test process cannot
        // perturb this test. The assert_eq! below still pins the constant's actual
        // value, so changing the default without revisiting this guard goes RED.
        let cp = coder_holders();
        assert_eq!(
            cp,
            vec!["coder".to_string(), "assistant".to_string()],
            "sanity: this test must be exercising the real DEFAULT_CODER_ACTIVITY_HOLDERS"
        );

        // The three unambiguous "coder"/"assistant" sweep labels.
        assert!(
            is_coder_activity_holder("intake_coder_sweep", &cp),
            "intake_coder_sweep (Terminus src/intake/coder_sweep.rs GPU_HOLDER) must reach \
             the coder-activity floor"
        );
        assert!(
            is_coder_activity_holder("intake_coder_case", &cp),
            "intake_coder_case (Terminus src/intake/coder_case.rs GPU_HOLDER) must reach \
             the coder-activity floor"
        );
        assert!(
            is_coder_activity_holder("intake_assistant_sweep", &cp),
            "intake_assistant_sweep (Terminus src/intake/assistant/runner.rs GPU_HOLDER) \
             must reach the coder-activity floor"
        );

        // `mint_breakfix` (Terminus src/intake/breakfix.rs BREAKFIX_GPU_HOLDER):
        // deliberately EXCLUDED from the default. Chord has no visibility into
        // breakfix's actual heartbeat/renewal cadence, and folding a holder into
        // this set makes it revert EARLIER on silence — safe for the sweep/assistant
        // labels (renew frequently, per their own loop), but a mistake for a
        // possibly-quiet repair path without first verifying its cadence against
        // Terminus source. See the full reasoning on `DEFAULT_CODER_ACTIVITY_HOLDERS`.
        assert!(
            !is_coder_activity_holder("mint_breakfix", &cp),
            "mint_breakfix must NOT reach the coder-activity floor via the DEFAULT pattern \
             set — unverified heartbeat cadence, deliberately excluded pending that \
             verification (see DEFAULT_CODER_ACTIVITY_HOLDERS doc); it remains governed by \
             the absolute watchdog deadline alone, unchanged from before this item"
        );

        // None of the four real labels contain any COMPILER pattern — confirming
        // the original finding: without this dedicated set, `is_compiler_lease`
        // alone (the pre-round-4 gate on the floor branch) never matches any of
        // them, and the floor was dead code for every real caller.
        let compiler_patterns = holders();
        for real_holder in [
            "intake_coder_sweep",
            "intake_coder_case",
            "intake_assistant_sweep",
            "mint_breakfix",
        ] {
            assert!(
                !is_compiler_lease(real_holder, &compiler_patterns),
                "{real_holder} must NOT match the COMPILER pattern set — that's the whole \
                 reason a separate coder-activity set is needed"
            );
        }
    }

    #[test]
    fn watchdog_tick_should_activate_reaches_floor_for_real_coder_holder_labels() {
        // End-to-end (through the full per-tick decision function, not just
        // `is_coder_activity_holder` in isolation): a REAL production holder label
        // with a genuinely stale heartbeat must activate via the floor even though
        // the absolute watchdog deadline has NOT yet expired — this is the exact
        // "genuine last-activity tracking, not a blind one-shot deadline" behavior
        // CHRD-135 exists to deliver, now actually reachable for real traffic.
        let p = holders();
        // Round-5: derived from the constant (see `coder_holders`) — hermetic.
        let cp = coder_holders();
        let timeout = 900u64;

        // intake_coder_sweep, deadline not expired, heartbeat fresh (10s old) ⇒
        // must NOT activate yet.
        assert!(!watchdog_tick_should_activate(
            1_010,
            false,
            Some("intake_coder_sweep"),
            Some(1_000),
            &p,
            &cp,
            timeout
        ));

        // Same holder, heartbeat now stale (901s of silence), deadline STILL not
        // expired ⇒ must activate — silence, not the absolute deadline, is what
        // bounds this now.
        assert!(watchdog_tick_should_activate(
            1_901,
            false,
            Some("intake_coder_sweep"),
            Some(1_000),
            &p,
            &cp,
            timeout
        ));

        // mint_breakfix: deliberately EXCLUDED from the default coder-activity
        // set (unverified heartbeat cadence — see DEFAULT_CODER_ACTIVITY_HOLDERS
        // doc), so with the deadline NOT yet expired it must NOT activate even
        // with a heartbeat well past the 900s coder-activity window — the floor
        // simply never engages for it today. It stays governed solely by the
        // absolute deadline, exactly as it was before this item existed.
        assert!(!watchdog_tick_should_activate(
            1_901,
            false,
            Some("mint_breakfix"),
            Some(1_000),
            &p,
            &cp,
            timeout
        ));
        // ...but once the deadline itself expires, it activates immediately
        // (unaffected/unchanged — the same "non-compiler, non-coder-activity
        // holder" path any other unrecognized holder takes).
        assert!(watchdog_tick_should_activate(
            1_901,
            true,
            Some("mint_breakfix"),
            Some(1_000),
            &p,
            &cp,
            timeout
        ));
    }

    #[test]
    fn behavioral_coder_silence_reverts_state_machine_to_active() {
        // BEHAVIORAL PROOF, not just the pure decision function (review, round 2 —
        // ending on a hand-called `ctl.exit()` was the round-1 defect verbatim,
        // still present as of round 1's fix): drive the REAL `IdleController`
        // state machine plus a REAL (isolated, non-global) `GpuExclusive` instance
        // through genuine coder heartbeats, calling `watchdog_tick` ITSELF — the
        // exact function `watchdog_loop` calls every tick, decision AND its CAS
        // effect together — and confirm the mode ACTUALLY flips back to `Active`
        // (the default assistant-fleet mode) on its own once the heartbeats stop,
        // and does NOT flip while they keep arriving. Time is INJECTED
        // (`tick_now`/`now_past`), never slept.
        use crate::gpu_exclusive::GpuExclusive;

        let ctl = IdleController::new();
        // GPU_EXCLUSIVE's OWN TTL is a different, unrelated concern (governs when a
        // lock is takeable by someone else) — isolate the variable under test
        // (heartbeat silence vs. the coder-activity timeout) by giving it no bound.
        let gpu = GpuExclusive::new(u64::MAX);
        let patterns = holders();
        let coder_patterns = coder_holders();
        let timeout = 900u64;

        // Coder starts a heavy build: Chord drains into idle mode with a long
        // absolute deadline, exactly as a real build would set.
        let m = manifest("compiler-build-42", 0, 3600);
        assert!(matches!(ctl.enter(m.clone()), EnterOutcome::Entered(_)));
        assert_eq!(ctl.phase(), Phase::Idle);

        // The build acquires the GPU-exclusive lease and heartbeats it every 300s
        // (well under the 900s window) while genuinely progressing. Each tick
        // calls `watchdog_tick` itself — the negative case (no revert) is now
        // behavioral too, not just a predicate assertion.
        for tick_now in [0u64, 300, 600, 900, 1_200, 1_500] {
            gpu.acquire("compiler-build-42", tick_now); // fresh grant, then heartbeat refreshes
            let result = watchdog_tick(&ctl, &gpu, tick_now, &patterns, &coder_patterns, timeout);
            assert!(
                result.is_none(),
                "must NOT revert mid-build while heartbeats keep arriving (t={tick_now})"
            );
            assert_eq!(
                ctl.phase(),
                Phase::Idle,
                "still idle — the build is still genuinely renewing (t={tick_now})"
            );
        }

        // The build finishes (or crashes) at t=1500 and STOPS heartbeating — no
        // further renewal arrives.
        let last_heartbeat = 1_500u64;

        // Advance the tracked clock past the window WITHOUT sleeping.
        let now_past = last_heartbeat + timeout; // exactly at the 900s silence boundary
        let cleared = watchdog_tick(&ctl, &gpu, now_past, &patterns, &coder_patterns, timeout)
            .expect("coder silence past the window must actually revert");
        assert_eq!(cleared.reason, "compiler-build-42");
        assert_eq!(
            ctl.phase(),
            Phase::Active,
            "Chord must actually be back in its default assistant-fleet mode"
        );
        assert!(!ctl.is_idle());
    }

    #[test]
    fn watchdog_tick_treats_expired_gpu_record_as_no_holder() {
        // Defect #1 (review, round 1): a genuinely ABANDONED GPU_EXCLUSIVE record
        // (heartbeat that stopped updating, well past BOTH GPU_EXCLUSIVE's own TTL
        // and the coder-activity timeout) must not defer an already-overdue revert
        // forever. Drive `watchdog_tick` (the exact function `watchdog_loop` calls)
        // directly, not just the pure decision fn.
        //
        // Values are realistic (GPU ttl=600 default, coder timeout=900 default) —
        // NOT an artificially tiny ttl / huge timeout pair, because the round-2 fix
        // (`effective_ttl = max(gpu.ttl(), coder_activity_timeout)`, see
        // `watchdog_tick`'s doc) makes the coder-activity timeout the FLOOR on
        // liveness, not GPU_EXCLUSIVE's own ttl — so "abandoned" now means silence
        // past BOTH, exercised by `watchdog_tick_default_ttls_honor_the_900s_coder_activity_floor`
        // below covers the boundary between the two; this test covers the
        // unambiguous case: nobody has renewed in ages, by either standard.
        use crate::gpu_exclusive::GpuExclusive;

        let ctl = IdleController::new();
        let gpu = GpuExclusive::new(600); // GPU_EXCLUSIVE default TTL
        let patterns = holders();
        let coder_patterns = coder_holders();
        let coder_activity_timeout = 900u64; // coder-activity default

        // Absolute deadline already passed by t=2000 — with NO live compiler
        // holder, the watchdog must revert.
        let m = manifest("compiler-build-expired", 0, 100);
        assert!(matches!(ctl.enter(m.clone()), EnterOutcome::Entered(_)));

        // The lease was acquired at t=0 and never renewed again.
        gpu.acquire("compiler-build-expired", 0);

        // At t=2000, 2000s of silence vastly exceeds BOTH the 600s GPU_EXCLUSIVE
        // TTL and the 900s coder-activity floor — genuinely abandoned by any
        // standard. The watchdog must revert on its overdue deadline instead of
        // deferring on a dead record's stale heartbeat.
        let now = 2000u64;

        let result = watchdog_tick(&ctl, &gpu, now, &patterns, &coder_patterns, coder_activity_timeout);
        assert!(
            result.is_some(),
            "a record abandoned well past both TTLs must not defer an already-overdue revert"
        );
        assert_eq!(
            ctl.phase(),
            Phase::Active,
            "the state machine must have actually flipped back to Active"
        );
    }

    #[test]
    fn watchdog_tick_default_ttls_honor_the_900s_coder_activity_floor() {
        // Review, round 2, blocking finding #3: with the REAL default TTLs
        // (GPU_EXCLUSIVE ttl=600s, coder_activity_timeout=900s), the promised
        // floor is "revert only after AT LEAST 900s of coder silence." Before this
        // fix, `watchdog_tick` read holder-liveness through
        // `gpu.active_holder(now)`, which expires a record at GPU_EXCLUSIVE's own
        // (shorter) 600s TTL — so a build heartbeating, say, every 650s (well
        // under the promised 900s floor, but over GPU_EXCLUSIVE's unrelated 600s
        // TTL) would already read as "no holder" and revert at only ~600s of
        // silence once the absolute deadline had passed. Pins the EFFECTIVE
        // bound, not the constant.
        use crate::gpu_exclusive::GpuExclusive;

        let ctl = IdleController::new();
        let gpu = GpuExclusive::new(crate::gpu_exclusive::DEFAULT_TTL_SECS); // 600s
        let patterns = holders();
        let coder_patterns = coder_holders();
        let coder_activity_timeout = DEFAULT_CODER_ACTIVITY_TIMEOUT_SECS; // 900s
        assert!(
            crate::gpu_exclusive::DEFAULT_TTL_SECS < coder_activity_timeout,
            "sanity: this test only means something when the GPU ttl is the \
             SHORTER of the two, exactly the real default configuration"
        );

        // Absolute deadline already passed by the time we check (so ONLY the
        // coder-activity heartbeat floor is what's deferring the revert).
        let m = manifest("compiler-build-longhaul", 0, 10);
        assert!(matches!(ctl.enter(m), EnterOutcome::Entered(_)));

        let last_heartbeat = 0u64;
        gpu.acquire("compiler-build-longhaul", last_heartbeat);

        // At 650s of silence: past GPU_EXCLUSIVE's own 600s TTL, but comfortably
        // under the promised 900s coder-activity floor. Must NOT revert — this is
        // exactly the regression this finding describes.
        let still_within_floor = last_heartbeat + 650;
        assert!(
            watchdog_tick(&ctl, &gpu, still_within_floor, &patterns, &coder_patterns, coder_activity_timeout)
                .is_none(),
            "650s of silence is past GPU_EXCLUSIVE's own 600s TTL but still under \
             the promised 900s coder-activity floor — must NOT revert"
        );
        assert_eq!(
            ctl.phase(),
            Phase::Idle,
            "still idle — 650s does not yet violate the 900s floor"
        );

        // At exactly 900s of silence: the coder-activity floor itself is now
        // exceeded — must revert.
        let at_floor = last_heartbeat + coder_activity_timeout;
        let result = watchdog_tick(&ctl, &gpu, at_floor, &patterns, &coder_patterns, coder_activity_timeout);
        assert!(
            result.is_some(),
            "900s of silence meets the coder-activity floor — must revert"
        );
        assert_eq!(ctl.phase(), Phase::Active);
    }

    #[test]
    fn watchdog_tick_episode_guard_refuses_superseded_activation() {
        // Defect #3 (review, round 1): an activation decision computed about idle
        // episode N must not be able to cancel a idle episode N+1 that has since
        // replaced it. Exercised directly at the CAS
        // (`begin_activate_for_episode`) — the exact primitive `watchdog_tick`
        // calls — rather than only through the pure decision function.
        let ctl = IdleController::new();

        let m1 = manifest("compiler-build-first", 0, 10);
        let entered1 = match ctl.enter(m1) {
            EnterOutcome::Entered(m) => m,
            other => panic!("expected Entered, got {other:?}"),
        };
        let stale_episode = entered1.episode;

        // The first build's idle episode ends (e.g. a normal activate) and a SECOND,
        // distinct idle episode begins — simulating a fresh decision superseding a
        // stale one still in flight.
        assert!(matches!(ctl.exit(), ActivateOutcome::Activated(_)));
        let m2 = manifest("compiler-build-second", 20, 30);
        let entered2 = match ctl.enter(m2) {
            EnterOutcome::Entered(m) => m,
            other => panic!("expected Entered, got {other:?}"),
        };
        assert_ne!(
            entered2.episode, stale_episode,
            "sanity: the second episode really is a distinct identity"
        );

        // A decision computed against the FIRST (now stale) episode must refuse,
        // not cancel the second episode's live idle state.
        assert_eq!(
            ctl.begin_activate_for_episode(stale_episode),
            Err(BeginActivate::Superseded)
        );
        assert_eq!(
            ctl.phase(),
            Phase::Idle,
            "the newer episode's Idle phase must be untouched by the stale decision"
        );
        assert_eq!(ctl.snapshot().unwrap(), entered2);

        // The CURRENT episode's own decision, by contrast, must still win the CAS.
        let (won, generation) = ctl
            .begin_activate_for_episode(entered2.episode)
            .expect("the current episode's own decision must win the CAS");
        assert_eq!(won, entered2);
        ctl.finish_activate(generation);
        assert_eq!(ctl.phase(), Phase::Active);
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
        // finding #1: only ONE caller may run release. The first begin wins; a second
        // while EnteringIdle must NOT also get a transition.
        let ctl = IdleController::new();
        let t = ctl.try_begin_enter().expect("first begins");
        assert!(matches!(
            ctl.try_begin_enter(),
            Err(BeginEnter::InTransition)
        ));
        assert_eq!(ctl.begin_enter(), BeginEnter::InTransition);
        // commit and confirm a later begin sees AlreadyIdle, never a second Begin.
        let m = manifest("compiler", 1, 2);
        assert_eq!(t.commit(m.clone()), Some(m.clone()));
        match ctl.begin_enter() {
            BeginEnter::AlreadyIdle(got) => assert_eq!(got, m),
            other => panic!("expected AlreadyIdle, got {other:?}"),
        }
    }

    #[test]
    fn begin_activate_is_exclusive_cas() {
        let ctl = IdleController::new();
        ctl.enter(manifest("compiler", 1, 2));
        let (_m, generation) = ctl.begin_activate_inner().expect("Idle ⇒ activate begins");
        // While Activating, a second begin must not also win.
        assert_eq!(ctl.begin_activate(), BeginActivate::InTransition);
        assert!(ctl.finish_activate(generation));
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
        assert_eq!(t.commit(m.clone()), Some(m.clone()));
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
    fn stale_guard_drop_does_not_clobber_newer_transition() {
        // finding #1 (ABA): begin (gen N) → watchdog recovers to Active (gen bumped) →
        // begin again (gen N+2) → drop the FIRST guard. The stale guard's rollback must
        // be a NO-OP: it must NOT roll the SECOND (newer) transition back to Active.
        let ctl = IdleController::new();
        let first = ctl.try_begin_enter().expect("first transition begins");
        assert_eq!(ctl.phase(), Phase::EnteringIdle);

        // Watchdog force-recovers the (now stale) first transition to Active.
        let now = crate::gpu_exclusive::now_epoch();
        assert!(ctl.recover_stale_transition(now + 10_000, 120));
        assert_eq!(ctl.phase(), Phase::Active);

        // A SECOND enter begins — a brand-new, distinct generation.
        let second = ctl.try_begin_enter().expect("second transition begins");
        assert_eq!(ctl.phase(), Phase::EnteringIdle);
        let second_generation = second.generation;

        // Drop the FIRST (stale) guard: its Drop→abort_enter must NOT touch the second
        // transition (generation mismatch), so we stay in EnteringIdle.
        drop(first);
        assert_eq!(
            ctl.phase(),
            Phase::EnteringIdle,
            "stale guard drop must not roll back the newer transition"
        );

        // The second transition can still commit normally. `commit` (via
        // `finish_enter`) stamps `episode` from ITS OWN generation (CHRD-135 defect
        // #3), overwriting the `0` placeholder the `manifest()` helper sets — so the
        // expected value comes from the committed return, not the pre-commit
        // literal.
        let m = manifest("compiler", 7, 9);
        let committed = second.commit(m).expect("second transition still live");
        assert_eq!(committed.episode, second_generation);
        assert!(ctl.is_idle());
        assert_eq!(ctl.snapshot().unwrap(), committed);
    }

    #[test]
    fn stale_guard_commit_does_not_clobber_newer_phase() {
        // Companion to the ABA test: a stale guard's COMMIT must also no-op (return
        // None) rather than install its manifest over whatever phase now exists.
        let ctl = IdleController::new();
        let stale = ctl.try_begin_enter().expect("first transition begins");
        let now = crate::gpu_exclusive::now_epoch();
        assert!(ctl.recover_stale_transition(now + 10_000, 120)); // → Active, gen bumped
                                                                  // A newer enter takes over and reaches Idle. `enter` (via `finish_enter`)
        // stamps `episode` from its own generation (CHRD-135 defect #3), so the
        // expected manifest is the one `Entered` actually returns, not the
        // pre-commit `fresh` literal (whose `episode: 0` placeholder is overwritten).
        let fresh = manifest("compiler", 1, 2);
        let entered = match ctl.enter(fresh) {
            EnterOutcome::Entered(m) => m,
            other => panic!("expected Entered, got {other:?}"),
        };
        // The stale guard commits LATE: must be a no-op (None), not clobber fresh idle.
        let stale_m = manifest("stale", 99, 100);
        assert_eq!(stale.commit(stale_m), None);
        assert_eq!(
            ctl.snapshot().unwrap(),
            entered,
            "stale commit must not overwrite the newer manifest"
        );
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
        let t = ctl.try_begin_enter().expect("Active ⇒ transition begins");
        assert!(matches!(ctl.try_admit(), AdmitOutcome::Transitioning));
        assert_eq!(
            ctl.inflight_count(),
            0,
            "no request admitted after EnteringIdle"
        );

        // Fully idle also refuses admission (caller lazy-activates instead).
        let _ = t.commit(manifest("compiler", 1, 2));
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
    fn crash_during_activating_reloads_active_not_idle() {
        // finding #2: begin_activate clears the persisted manifest to Active BEFORE the
        // (async) restore work. A crash while memory is `Activating` must therefore
        // reload as Active, not Idle — worst case we lose idle, never wedge idle.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("admin_idle_state.json");

        let ctl = IdleController::with_state(Some(path.clone()));
        ctl.enter(manifest("compiler", 1, 2)); // → Idle, disk = Some(manifest)

        // Begin activate: memory → Activating, but disk must be cleared to Active NOW
        // (before finish_activate). Do NOT finish — simulate a crash here.
        let (_m, _gen) = ctl.begin_activate_inner().expect("Idle ⇒ activate begins");
        assert_eq!(ctl.phase(), Phase::Activating);

        // Reload from disk as if the process crashed mid-Activating.
        let reloaded = IdleController::with_state(Some(path));
        assert!(
            !reloaded.is_idle(),
            "crash during Activating must reload Active, not Idle"
        );
        assert_eq!(reloaded.phase(), Phase::Active);
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

    #[test]
    fn begin_activate_persist_failure_stays_recoverable() {
        // finding #2: when a state path is configured and clearing the manifest to
        // Active FAILS, begin_activate must ABORT (PersistFailed) and leave the
        // controller Idle — so on-disk and memory stay consistent (a crash reloads
        // Idle, recoverable), never Activating-with-disk-still-Idle.
        let dir = tempfile::tempdir().unwrap();
        // Make the state file's PARENT a regular file, so create_dir_all/write fails.
        let blocker = dir.path().join("blocker");
        std::fs::write(&blocker, b"x").unwrap();
        let path = blocker.join("admin_idle_state.json"); // parent `blocker` is a file

        let ctl = IdleController::with_state(Some(path));
        // Reach Idle in memory (finish_enter's persist is best-effort and just fails
        // silently against the unwritable path — memory still becomes Idle).
        assert!(matches!(
            ctl.enter(manifest("compiler", 1, 2)),
            EnterOutcome::Entered(_)
        ));
        assert!(ctl.is_idle());

        // The hard-gated persist(None) MUST fail → PersistFailed, memory stays Idle.
        match ctl.begin_activate_inner() {
            Err(BeginActivate::PersistFailed) => {}
            other => panic!("expected PersistFailed, got {other:?}"),
        }
        assert!(
            ctl.is_idle(),
            "activate must abort and remain Idle when the persist gate fails"
        );
        assert_eq!(ctl.phase(), Phase::Idle);
        // The full exit() path surfaces it as a retryable ActivateOutcome too.
        assert_eq!(ctl.exit(), ActivateOutcome::PersistFailed);
        assert!(ctl.is_idle());
    }

    #[tokio::test]
    async fn release_exceeding_budget_rolls_back_with_admission_closed() {
        // finding #1: while EnteringIdle, admission is CLOSED; a release that overruns
        // its budget self-aborts (timeout) and the guard rollback reopens admission
        // ONLY after a consistent return to Active — never mid-release. This mirrors
        // enter_idle's structure using the controller primitives.
        let ctl = IdleController::new();
        let transition = ctl.try_begin_enter().expect("Active ⇒ transition begins");

        // Admission is closed for the whole EnteringIdle window.
        assert!(matches!(ctl.try_admit(), AdmitOutcome::Transitioning));

        // Simulate release work that exceeds a tiny budget.
        let budget = Duration::from_millis(20);
        let release = tokio::time::timeout(budget, async {
            tokio::time::sleep(Duration::from_millis(500)).await;
        })
        .await;
        assert!(release.is_err(), "release must exceed its budget");

        // STILL EnteringIdle and STILL closed — we have not committed or rolled back.
        assert_eq!(ctl.phase(), Phase::EnteringIdle);
        assert!(matches!(ctl.try_admit(), AdmitOutcome::Transitioning));

        // On timeout we drop the guard → clean rollback to Active.
        drop(transition);
        assert_eq!(ctl.phase(), Phase::Active);
        // Admission reopens ONLY now, after the consistent rollback.
        assert!(matches!(ctl.try_admit(), AdmitOutcome::Admitted(_)));
    }

    // ── env parsing ──────────────────────────────────────────────────────────

    #[test]
    #[serial] // CHRD-94: mutates PROCESS-GLOBAL env; must not overlap another test that reads it
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
    #[serial] // CHRD-94: mutates PROCESS-GLOBAL env; must not overlap another test that reads it
    fn compiler_lease_holders_default_when_unset() {
        std::env::remove_var("CHORD_IDLE_COMPILER_LEASE_HOLDERS");
        let p = compiler_lease_holders_from_env();
        assert!(p.contains(&"compiler".to_string()));
        assert!(is_compiler_lease("compiler", &p));
    }

    #[test]
    #[serial] // CHRD-94: mutates PROCESS-GLOBAL env; must not overlap another test that reads it
    fn stale_threshold_always_exceeds_release_budget() {
        // finding #1 invariant: the watchdog stale bound must be strictly greater than
        // the release budget, so stale-recovery can never fire during live release.
        // Default config:
        std::env::remove_var("CHORD_IDLE_STALE_TRANSITION_SECS");
        std::env::remove_var("CHORD_IDLE_RELEASE_BUDGET_SECS");
        assert!(stale_transition_secs_from_env() > release_budget_secs_from_env());

        // Misconfigured so stale ≤ budget ⇒ clamped strictly above the budget.
        std::env::set_var("CHORD_IDLE_RELEASE_BUDGET_SECS", "90");
        std::env::set_var("CHORD_IDLE_STALE_TRANSITION_SECS", "30"); // below budget
        assert!(
            stale_transition_secs_from_env() > release_budget_secs_from_env(),
            "misconfigured stale ≤ budget must be clamped above the budget"
        );

        std::env::remove_var("CHORD_IDLE_STALE_TRANSITION_SECS");
        std::env::remove_var("CHORD_IDLE_RELEASE_BUDGET_SECS");
    }

    // ── CHORD-ACT-01: activity view ──────────────────────────────────────────

    #[test]
    fn activity_summary_serving_when_inflight_positive() {
        // Any in-flight request ⇒ serving, and idle_secs is pinned to 0 regardless of
        // how stale the last-activity stamp looks.
        let (serving, idle_secs) = activity_summary(1, 1_000, 9_999);
        assert!(serving);
        assert_eq!(idle_secs, 0);
        let (serving, idle_secs) = activity_summary(5, 0, 9_999);
        assert!(serving);
        assert_eq!(idle_secs, 0);
    }

    #[test]
    fn activity_summary_idle_secs_grows_from_last_activity() {
        // Not serving ⇒ idle_secs is now - last_activity, growing as `now` advances.
        let (serving, idle_secs) = activity_summary(0, 1_000, 1_000);
        assert!(!serving);
        assert_eq!(idle_secs, 0);
        assert_eq!(activity_summary(0, 1_000, 1_042).1, 42);
        assert_eq!(activity_summary(0, 1_000, 4_600).1, 3_600);
    }

    #[test]
    fn activity_summary_never_negative_on_backward_clock() {
        // A last-activity stamp in the "future" (clock skew / step-back) clamps to 0,
        // never a wrapped/negative age.
        let (serving, idle_secs) = activity_summary(0, 5_000, 1_000);
        assert!(!serving);
        assert_eq!(idle_secs, 0);
    }

    #[test]
    fn activity_json_shape_serving_and_idle() {
        // Serving shape.
        let v = activity_json(2, 1_000, 5_000);
        assert_eq!(v["serving"], serde_json::json!(true));
        assert_eq!(v["inflight"], serde_json::json!(2));
        assert_eq!(v["idle_secs"], serde_json::json!(0));
        assert_eq!(v["last_request_unix"], serde_json::json!(1_000));

        // Idle shape.
        let v = activity_json(0, 1_000, 1_075);
        assert_eq!(v["serving"], serde_json::json!(false));
        assert_eq!(v["inflight"], serde_json::json!(0));
        assert_eq!(v["idle_secs"], serde_json::json!(75));
        assert_eq!(v["last_request_unix"], serde_json::json!(1_000));
    }

    #[test]
    fn controller_records_activity_on_admission() {
        // A fresh isolated controller starts Active with a seeded (non-zero) stamp;
        // admitting a request refreshes it and bumps the in-flight count. This reuses
        // the SAME counter the drain/idle machinery reads — no second gauge.
        let ctl = IdleController::new();
        assert_eq!(ctl.inflight_count(), 0);
        ctl.set_last_activity_for_test(1_000);
        assert_eq!(ctl.last_activity_unix(), 1_000);

        let guard = match ctl.try_admit() {
            AdmitOutcome::Admitted(g) => g,
            _ => panic!("expected admission while Active"),
        };
        assert_eq!(ctl.inflight_count(), 1);
        // Admission stamped a real wall-clock time (>> the 1_000 test seed).
        assert!(
            ctl.last_activity_unix() > 1_000,
            "admission must refresh last_activity from the seed"
        );
        // While admitted the derived view is `serving`.
        assert!(activity_summary(ctl.inflight_count(), ctl.last_activity_unix(), now_epoch() as i64).0);
        drop(guard);
        assert_eq!(ctl.inflight_count(), 0);
        // Once drained, the view reports not-serving with a non-negative age.
        let (serving, _idle) =
            activity_summary(ctl.inflight_count(), ctl.last_activity_unix(), now_epoch() as i64);
        assert!(!serving);
    }
}
