//! RVXR-01: the OPPORTUNISTIC CODER TIER — borrow the reaped window, never the
//! assistant's.
//!
//! ## The window this exploits
//! On the shared GPU host the assistant cohort (personality + router + embedding,
//! held by [`super::resident_set`]) and a 30B-class coder model contend for one
//! unified memory pool. The coder therefore fits in exactly ONE situation: when
//! the assistant models have already been **idle-reaped** (Ollama's keep-alive
//! expired, or an idle-mode release unloaded them) and the memory is genuinely
//! free.
//!
//! ## Measured — and why the OBVIOUS sensors are all wrong here
//! Three separate measurements on this host, each of which independently breaks
//! a naive admission check. The detail lives in [`sensors`]; the summary:
//!
//! - **GTT is invisible to process RSS.** The models live in GTT on this
//!   unified-memory APU. The ollama runners reported **0.52 and 0.40 GB** in `ps`
//!   while holding **28.7 GB** of GTT; every process on the box summed to 10.2 GB
//!   against that 28.7 GB. A fit check reading process memory does not see the
//!   models at all.
//! - **`MemFree` fails LOW** — page cache means near-zero is the healthy state of
//!   a build host, so gating on it refuses forever.
//! - **`MemAvailable` fails HIGH and is anti-correlated with danger** — it read
//!   **89.4 GB while this host was ~10 minutes from hanging**, because the
//!   reclaimable cache it counts is exactly what evaporates under pressure.
//!
//! So capacity is read from GTT, and pressure from `Committed_AS` plus the swap
//! TREND — the two that actually moved during the real incident. An earlier
//! version of this file sized on `MemAvailable`; that was wrong, and it looked
//! right.
//!
//! **No absolute memory figure is hardcoded anywhere.** Fit is re-derived from a
//! fresh reading at every arm-confirm tick and re-checked while resident. One
//! measured RATIO is encoded as a config default: runtime footprint exceeds
//! on-disk size (weights plus KV cache and context) — `granite4.1:8b` is 4.98 GiB
//! on disk and 8.93 GiB resident, ~1.8x — so
//! [`CoderTierConfig::footprint_factor`] inflates the registry size before the
//! fit test. Sizing on the raw disk figure would under-count a 30B coder by most
//! of a model.
//!
//! **A threshold is a claim, and claims need the same scrutiny as code.** Every
//! one lives in [`policy`] with a documented rationale and an env override,
//! because thresholds read as configuration and get waved through review.
//!
//! That is the whole item. This tier does NOT create the window: **it never
//! evicts, releases, unpins, or otherwise touches the assistant cohort.** It
//! observes that the window is already open, opportunistically loads the coder
//! into it so reviews can use a local engine, and gives the memory back the
//! instant a user shows up.
//!
//! ## Three hazards, and where each is closed
//!
//! **1. The eviction race — an incoming user request wins IMMEDIATELY.**
//! [`CoderTier::note_assistant_request`] is a synchronous, non-blocking, no-await
//! call on the inference hot path. It latches every outstanding lease's
//! [`CancelToken`], bumps the epoch, flips the phase to
//! [`Phase::Evicting`], and RETURNS. The actual teardown (stopping the coder
//! backend, handing capacity back) runs in a detached task. **Assistant mode
//! never waits behind a review**, and it never waits behind eviction either —
//! there is no `.await` and no lock held across one anywhere on that path.
//!
//! **2. Partial output is not a result.** A review that was mid-generation when
//! eviction fired must not produce a plausible-looking verdict. Tokens already
//! generated are **discarded**, not returned: [`Lease::commit`] takes the output
//! BY VALUE and, if the lease's epoch no longer matches the tier's (or its token
//! latched), drops it on the floor and returns
//! [`LeaseOutcome::Interrupted`] carrying [`INTERRUPT_REASON`] /
//! [`INTERRUPT_CODE`]. This is the same commit-under-the-lock generation guard
//! [`super::resident_set`] uses for warm passes, applied to output instead of
//! bookkeeping. Downstream (RVXR-02, Terminus) an interrupted seat is ABSENT and
//! must never be counted as a pass — see "Contract with RVXR-02" below.
//!
//! **3. Thrash.** Moving 18.6 GB in and out under intermittent user activity
//! would spend all its time loading and serve nobody. Two independent brakes:
//! - a **minimum idle dwell** ([`CoderTierConfig::min_idle_secs`]) that must hold
//!   before the tier will even ARM, plus an **arm-confirm** delay
//!   ([`CoderTierConfig::arm_confirm_secs`]) that must ALSO elapse before a load
//!   — so the tier structurally cannot load on the first idle tick;
//! - a **post-eviction cooldown** ([`CoderTierConfig::cooldown_secs`]) during
//!   which no amount of idleness will reload.
//!
//! **3b. Headroom does not persist.** Something other than Chord can take the
//! window after the coder is loaded. While resident, every tick re-reads GTT and
//! the pressure signals and evicts on [`EvictReason::GttPressure`] or
//! [`EvictReason::SystemPressure`]. An UNREADABLE sensor does not evict — a
//! sensor gap is not evidence, and evicting on it would thrash on exactly what it
//! cannot see — but it DOES block a load, since taking the window on no
//! information is the cheap thing to refuse. The assistant's protection never
//! depends on this path: hazard 1 covers it unconditionally.
//!
//! ## What it consumes rather than reinvents
//! There is deliberately **no second model list and no second residency rule**
//! here. The coder target is a Chord ALIAS KEY resolved through the very same
//! [`super::resident_set::resolve_alias`] the resident set uses (dynamic lumina
//! store first, static `CHORD_MODEL_ALIASES` second) — no model name appears in
//! this file. Starting and stopping the coder go through the existing on-demand
//! backend lifecycle in [`crate::models::backends`] /
//! [`crate::models::routing`] (`resolve_and_ensure` /
//! `stop_on_demand_backend`), the same machinery that already lazy-starts and
//! idle-stops the coder backend. A duplicated residency rule drifts from the
//! original; this one has no copy to drift from.
//!
//! ## Fail-CLOSED on the load side (a deliberate inversion)
//! [`super::resident_set::plan_warm`] fails SOFT on an unreadable counter —
//! attempting a warm that does not fit is cheap and self-reporting. Here the
//! inversion is correct: loading tens of GB we cannot prove there is room for is
//! how you push a live assistant into swap. So an unresolved alias, an unknown
//! model footprint, an unreadable GTT counter, or an unreadable `Committed_AS`
//! all mean **do not load**.
//!
//! ## Default OFF
//! `CHORD_CODER_TIER_ENABLED` defaults to **false**. This touches a live
//! production service on a shared GPU host; it ships dark and is turned on
//! deliberately. When disabled, [`CoderTier::note_assistant_request`] is a single
//! relaxed atomic load and a return.
//!
//! ## Contract with RVXR-02 (Terminus)
//! Chord PRODUCES the signal; Terminus ACCOUNTS for it.
//! - Chord emits [`InterruptedReport`] — `{"interrupted": true, "code":
//!   "inference_engine_interrupted", "reason": "inference engine interrupted by
//!   user"}` — as the seat result whenever a review's generation was pre-empted.
//! - Terminus maps that code to a seat status of `evicted`, which is **ABSENT**:
//!   it is neither an APPROVE nor a REQUEST_CHANGES, it never counts toward a
//!   pass, and a panel that lost a seat to it is smaller than it looks and must
//!   be reported that way.
//! - The output that seat had already generated does not exist. Chord never sends
//!   it, so there is nothing for Terminus to decide about.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::Serialize;
use tracing::{info, warn};

pub use super::resident_set::CancelToken;

/// Human-readable reason attached to every pre-empted review, in the operator's
/// own words. Stable text — RVXR-02 surfaces it verbatim.
pub const INTERRUPT_REASON: &str = "inference engine interrupted by user";

/// Machine-readable code for the same condition. This is the token RVXR-02
/// switches on to mark a seat `evicted`/ABSENT; the prose may be reworded, this
/// must not be.
pub const INTERRUPT_CODE: &str = "inference_engine_interrupted";

// ─────────────────────────────────────────────────────────────────────────────
// Pure layers, in their own files
// ─────────────────────────────────────────────────────────────────────────────

/// The pure admission POLICY (config, phases, thresholds, `decide`). Split out
/// with ZERO crate-internal dependencies so a standalone harness can
/// `#[path]`-include the real file and mutation-test it without building the
/// crate — the harness tests the shipped source, so it cannot drift from it.
pub mod policy;
/// Memory SENSING: GTT (the only thing that can see model residency) plus
/// `Committed_AS` and the swap trend. Read its module docs before touching a
/// threshold — three obvious-looking sensors are wrong here, and two of them
/// were adopted before being measured.
pub mod sensors;

pub use policy::{
    assistant_quiet, can_admit, coder_fits, decide, pressure_reason, runtime_footprint_gb,
    swap_growth_gb, CoderTierConfig, Decision, EvictReason, Observation, Phase,
};
pub use sensors::{CommitReading, GttReading};
// ─────────────────────────────────────────────────────────────────────────────
// Lease outcome — the signal RVXR-02 consumes
// ─────────────────────────────────────────────────────────────────────────────

/// The serialized "this seat was pre-empted" signal. Chord produces it; Terminus
/// (RVXR-02) maps `code` to a seat status of `evicted`, which is ABSENT — never a
/// pass. See the module docs' "Contract with RVXR-02".
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InterruptedReport {
    /// Always `true`. Present so a consumer can branch without string-matching.
    pub interrupted: bool,
    /// Stable machine code — [`INTERRUPT_CODE`].
    pub code: &'static str,
    /// Human prose — [`INTERRUPT_REASON`].
    pub reason: &'static str,
}

impl InterruptedReport {
    pub fn new() -> Self {
        InterruptedReport {
            interrupted: true,
            code: INTERRUPT_CODE,
            reason: INTERRUPT_REASON,
        }
    }
}

impl Default for InterruptedReport {
    fn default() -> Self {
        Self::new()
    }
}

/// The result of committing a review's output through a lease.
///
/// `Interrupted` carries NO output — not a truncated one, not an empty one. The
/// generated tokens were moved into [`Lease::commit`] and dropped there.
#[derive(Debug, Clone, PartialEq)]
pub enum LeaseOutcome<T> {
    Completed(T),
    Interrupted(InterruptedReport),
}

impl<T> LeaseOutcome<T> {
    pub fn is_interrupted(&self) -> bool {
        matches!(self, LeaseOutcome::Interrupted(_))
    }

    /// The output, or `None` if the seat was pre-empted. There is deliberately no
    /// accessor that can yield partial output.
    pub fn completed(self) -> Option<T> {
        match self {
            LeaseOutcome::Completed(v) => Some(v),
            LeaseOutcome::Interrupted(_) => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The side-effect seam
// ─────────────────────────────────────────────────────────────────────────────

/// Everything the tier does to the outside world, behind one trait — so the
/// lifecycle (the epoch guard, the immediate-eviction path, the commit) is tested
/// by FORCING interleavings rather than by racing a real GPU and hoping.
#[async_trait]
pub trait CoderTierEnv: Send + Sync {
    /// The coder model, resolved through Chord's EXISTING alias machinery.
    /// `None` ⇒ unconfigured or not pulled ⇒ the tier loads nothing, ever.
    async fn resolve_coder(&self, alias: &str) -> Option<String>;
    /// Registry footprint (GB) of `model`. `None` ⇒ unknown ⇒ fail closed.
    async fn model_size_gb(&self, model: &str) -> Option<f64>;
    /// Live GTT capacity — the ONLY capacity signal, because it is the only one
    /// that can see model residency (see `sensors`). `None` ⇒ unreadable ⇒ the
    /// tier refuses to load.
    fn gtt(&self) -> Option<GttReading>;
    /// Live `Committed_AS`/`CommitLimit`/swap-used. `None` ⇒ unreadable ⇒ the
    /// tier refuses to load (but does not evict — see `pressure_reason`).
    fn commit(&self) -> Option<CommitReading>;
    /// Start the coder through the EXISTING on-demand backend lifecycle.
    async fn start_coder(&self, model: &str) -> Result<(), String>;
    /// Stop it through the EXISTING lifecycle stop. MUST be idempotent.
    async fn stop_coder(&self, model: &str) -> Result<(), String>;
    /// Hand assistant capacity back by re-running the EXISTING resident-set
    /// reconcile. Invoked UNCONDITIONALLY on every eviction path, including one
    /// where [`CoderTierEnv::stop_coder`] failed — the cohort must never be left
    /// unrestored because a teardown erred.
    async fn restore_assistant_capacity(&self);
}

// ─────────────────────────────────────────────────────────────────────────────
// Controller
// ─────────────────────────────────────────────────────────────────────────────

struct Inner {
    phase: Phase,
    /// Bumped by EVERY pre-emption. A lease whose captured epoch no longer
    /// matches is invalid, no matter what it has produced.
    epoch: u64,
    /// Live leases' cancel tokens, by lease id.
    leases: HashMap<u64, CancelToken>,
    next_lease_id: u64,
    /// The model currently loaded/loading, when any.
    model: Option<String>,
}

/// The opportunistic coder tier.
pub struct CoderTier {
    cfg: CoderTierConfig,
    /// Lock-free mirror of `cfg.enabled` so the hot-path hook costs one relaxed
    /// atomic load when the tier is off (the default).
    enabled: AtomicBool,
    inner: Mutex<Inner>,
    env: Arc<dyn CoderTierEnv>,
    interrupts: AtomicU64,
    loads: AtomicU64,
    evictions: AtomicU64,
    /// Previous swap-used reading, so the tick can derive a TREND. The level
    /// alone is not a signal; the growth between two observations is.
    last_swap_used_gb: Mutex<Option<f64>>,
}

/// Observable status for `GET /admin/coder-tier`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CoderTierStatus {
    pub enabled: bool,
    pub alias: String,
    pub phase: &'static str,
    pub epoch: u64,
    pub model: Option<String>,
    pub live_leases: usize,
    pub interrupts: u64,
    pub loads: u64,
    pub evictions: u64,
}

impl CoderTier {
    pub fn new(cfg: CoderTierConfig, env: Arc<dyn CoderTierEnv>) -> Arc<Self> {
        let enabled = cfg.enabled;
        Arc::new(CoderTier {
            cfg,
            enabled: AtomicBool::new(enabled),
            inner: Mutex::new(Inner {
                phase: Phase::Assistant,
                epoch: 0,
                leases: HashMap::new(),
                next_lease_id: 0,
                model: None,
            }),
            env,
            interrupts: AtomicU64::new(0),
            loads: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            last_swap_used_gb: Mutex::new(None),
        })
    }

    pub fn config(&self) -> &CoderTierConfig {
        &self.cfg
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        // A poisoned lock here means a panic while holding it. Every critical
        // section below is a handful of field writes with no user code in it, so
        // recovering the guard is strictly better than propagating a panic into
        // the inference hot path.
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn phase(&self) -> Phase {
        self.lock().phase
    }

    pub fn epoch(&self) -> u64 {
        self.lock().epoch
    }

    pub fn status(&self) -> CoderTierStatus {
        let inner = self.lock();
        CoderTierStatus {
            enabled: self.enabled.load(Ordering::Relaxed),
            alias: self.cfg.alias.clone(),
            phase: inner.phase.id(),
            epoch: inner.epoch,
            model: inner.model.clone(),
            live_leases: inner.leases.len(),
            interrupts: self.interrupts.load(Ordering::Relaxed),
            loads: self.loads.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
        }
    }

    // ── Hazard 1: the assistant always wins, immediately ─────────────────────

    /// **The hot-path hook. Synchronous, non-blocking, no `.await`, no I/O.**
    ///
    /// Call this at the start of every ASSISTANT (non-coder-lease) inference
    /// request. It cancels every live review lease, invalidates their output by
    /// bumping the epoch, flips to [`Phase::Evicting`], and returns — the caller
    /// proceeds to serve the user without waiting for a single byte of teardown.
    /// The teardown itself is spawned detached.
    ///
    /// Returns `true` if this call initiated an eviction (useful for logs/tests);
    /// `false` when there was nothing borrowed.
    ///
    /// Disabled tier ⇒ one relaxed atomic load and a `false`.
    pub fn note_assistant_request(self: &Arc<Self>) -> bool {
        if !self.enabled.load(Ordering::Relaxed) {
            return false;
        }
        // Read the clock BEFORE taking the lock: the critical section must contain
        // nothing but field writes.
        let now = crate::gpu_exclusive::now_epoch();
        let (tokens, evict) = {
            let mut inner = self.lock();
            // Bump FIRST: any lease that has already generated output is invalid
            // from this instant, whether or not it ever observes its token.
            inner.epoch = inner.epoch.wrapping_add(1);
            let tokens: Vec<CancelToken> = inner.leases.drain().map(|(_, t)| t).collect();
            let evict = match inner.phase {
                Phase::Loading | Phase::Ready => {
                    inner.phase = Phase::Evicting { since: now };
                    true
                }
                Phase::Arming { .. } => {
                    inner.phase = Phase::Assistant;
                    false
                }
                _ => false,
            };
            (tokens, evict)
        };
        // Latching is cheap and lock-free; do it outside the critical section.
        let interrupted = tokens.len() as u64;
        for t in &tokens {
            t.cancel();
        }
        if interrupted > 0 {
            self.interrupts.fetch_add(interrupted, Ordering::Relaxed);
        }
        if evict {
            let me = self.clone();
            // Detached: the user's request must not wait on teardown. If we are
            // somehow not on a runtime, the tick's budget backstop resolves the
            // phase instead — we still never block here.
            if tokio::runtime::Handle::try_current().is_ok() {
                tokio::spawn(async move {
                    me.run_eviction(EvictReason::AssistantArrived).await;
                });
            } else {
                warn!(
                    "coder tier: no runtime on the eviction path; teardown deferred to the tick"
                );
            }
        }
        evict
    }

    // ── Leases ───────────────────────────────────────────────────────────────

    /// Issue a review lease, or `None` when the coder is not serving. A lease is
    /// only ever granted in [`Phase::Ready`] — never while loading, evicting, or
    /// cooling down.
    pub fn try_acquire_lease(self: &Arc<Self>) -> Option<Lease> {
        if !self.enabled.load(Ordering::Relaxed) {
            return None;
        }
        let mut inner = self.lock();
        if inner.phase != Phase::Ready {
            return None;
        }
        let id = inner.next_lease_id;
        inner.next_lease_id += 1;
        let token = CancelToken::new();
        inner.leases.insert(id, token.clone());
        let epoch = inner.epoch;
        let model = inner.model.clone();
        drop(inner);
        Some(Lease {
            id,
            epoch,
            model,
            token,
            tier: self.clone(),
        })
    }

    /// Commit decision for a lease: is `epoch` still current AND is the lease
    /// still registered? Either failing means pre-empted. Always deregisters.
    ///
    /// ## Three independent guards, deliberately, and each is tested ALONE
    /// A pre-emption trips all three of: deregistration (the hot path drains the
    /// lease map), the epoch bump, and the cancel token. Mutation testing showed
    /// that removing any ONE of them left the other two and no test failed — the
    /// redundancy was real but undefended, which is how a later refactor removes
    /// two of them and nobody notices until the third goes too. So each mechanism
    /// now has a test that isolates it (see `tests.rs`, the
    /// `..._alone_is_sufficient` trio) and the redundancy is kept: this is the
    /// path where a truncated review turns into a verdict, and it is worth
    /// belt-and-braces.
    fn settle_lease(&self, id: u64, epoch: u64) -> bool {
        let mut inner = self.lock();
        let still_registered = inner.leases.remove(&id).is_some();
        still_registered && inner.epoch == epoch
    }

    /// Test-only: advance the epoch WITHOUT draining leases or cancelling tokens,
    /// so the epoch guard can be exercised in isolation from the other two.
    #[cfg(test)]
    fn bump_epoch_only(&self) {
        let mut inner = self.lock();
        inner.epoch = inner.epoch.wrapping_add(1);
    }

    /// Test-only: deregister every lease WITHOUT bumping the epoch or cancelling,
    /// so the deregistration guard can be exercised in isolation.
    #[cfg(test)]
    fn drain_leases_only(&self) {
        let mut inner = self.lock();
        inner.leases.clear();
    }

    // ── Load / evict ─────────────────────────────────────────────────────────

    /// One observation of the world: gather inputs, [`decide`], act.
    ///
    /// `now`, `assistant_idle_secs` and `assistant_inflight` come from the
    /// caller (in production, `GET /admin/activity`'s own counters) so this is
    /// deterministic under test.
    pub async fn tick(self: &Arc<Self>, now: u64, assistant_idle_secs: u64, assistant_inflight: bool) {
        let phase = self.phase();

        // Only pay for resolution/sizing when the phase could actually act on it.
        let (coder_resolved, coder_gb, model) = match phase {
            Phase::Arming { .. } => match self.env.resolve_coder(&self.cfg.alias).await {
                Some(m) => {
                    let gb = self.env.model_size_gb(&m).await;
                    (true, gb, Some(m))
                }
                None => (false, None, None),
            },
            _ => (false, None, None),
        };
        // Sense while ARMING (may we take the window?) and while LOADING/READY (is
        // it still ours, and is the host still healthy?). Other phases have no
        // memory decision to make, so they do not pay for the reads.
        let senses = matches!(
            phase,
            Phase::Arming { .. } | Phase::Loading | Phase::Ready
        );
        let (gtt_free_gb, commit_ratio, swap_growth) = if senses {
            let gtt = self.env.gtt().map(|g| g.free_gb());
            let commit = self.env.commit();
            let growth = match &commit {
                Some(c) => {
                    let mut last = self
                        .last_swap_used_gb
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    let g = policy::swap_growth_gb(*last, c.swap_used_gb);
                    *last = Some(c.swap_used_gb);
                    g
                }
                None => None,
            };
            (gtt, commit.and_then(|c| c.commit_ratio()), growth)
        } else {
            (None, None, None)
        };

        let obs = Observation {
            now,
            assistant_idle_secs,
            assistant_inflight,
            gtt_free_gb,
            commit_ratio,
            swap_growth_gb: swap_growth,
            coder_gb,
            coder_resolved,
        };

        match decide(&self.cfg, phase, &obs) {
            Decision::Hold => {}
            Decision::Rest => {
                let mut inner = self.lock();
                // Only leave a phase we still own — a concurrent hot-path hook may
                // have moved us on.
                if inner.phase == phase {
                    inner.phase = Phase::Assistant;
                }
            }
            Decision::Arm => {
                let mut inner = self.lock();
                if inner.phase == phase {
                    inner.phase = Phase::Arming { armed_at: now };
                }
            }
            Decision::Load => {
                let Some(model) = model else { return };
                {
                    let mut inner = self.lock();
                    if inner.phase != phase {
                        return; // pre-empted between decide and act
                    }
                    inner.phase = Phase::Loading;
                    inner.model = Some(model.clone());
                }
                let started = self.env.start_coder(&model).await;
                let mut inner = self.lock();
                // A user may have arrived DURING the load. If so the phase is
                // already Evicting and the eviction task owns the teardown — do
                // not resurrect Ready over it.
                if inner.phase == Phase::Loading {
                    match started {
                        Ok(()) => {
                            inner.phase = Phase::Ready;
                            self.loads.fetch_add(1, Ordering::Relaxed);
                            info!(model = %model, "coder tier: coder loaded into the reaped window");
                        }
                        Err(e) => {
                            inner.phase = Phase::Cooldown {
                                until: now.saturating_add(self.cfg.cooldown_secs),
                            };
                            inner.model = None;
                            warn!(error = %e, "coder tier: coder load failed — cooling down");
                        }
                    }
                }
            }
            Decision::Evict(reason) => {
                let tokens: Vec<CancelToken> = {
                    let mut inner = self.lock();
                    if inner.phase != phase {
                        return; // pre-empted between decide and act
                    }
                    inner.phase = Phase::Evicting { since: now };
                    inner.epoch = inner.epoch.wrapping_add(1);
                    inner.leases.drain().map(|(_, t)| t).collect()
                };
                let n = tokens.len() as u64;
                for t in &tokens {
                    t.cancel();
                }
                if n > 0 {
                    self.interrupts.fetch_add(n, Ordering::Relaxed);
                }
                self.run_eviction(reason).await;
            }
            Decision::ForceResolveEvict => {
                let mut inner = self.lock();
                if inner.phase == phase {
                    warn!("coder tier: eviction overran its budget — forcing cooldown");
                    inner.model = None;
                    inner.phase = Phase::Cooldown {
                        until: now.saturating_add(self.cfg.cooldown_secs),
                    };
                }
            }
        }
    }

    /// Tear the coder down and hand capacity back. Idempotent and fail-soft:
    /// `restore_assistant_capacity` is called on EVERY path, including when
    /// `stop_coder` errors, so the cohort is never left unrestored.
    ///
    /// The caller has already flipped the phase to [`Phase::Evicting`] and
    /// cancelled the leases — that is the part that must be instantaneous. This is
    /// the slow part, and nothing on the assistant path waits for it.
    pub async fn run_eviction(self: &Arc<Self>, reason: EvictReason) {
        let model = { self.lock().model.clone() };
        if let Some(model) = model.as_deref() {
            if let Err(e) = self.env.stop_coder(model).await {
                // Deliberately not fatal, and deliberately not a reason to skip
                // the restore below: a stop that errored is exactly when the
                // assistant most needs its capacity re-asserted.
                warn!(error = %e, model = %model, "coder tier: coder stop failed — restoring anyway");
            }
        }
        self.env.restore_assistant_capacity().await;
        self.evictions.fetch_add(1, Ordering::Relaxed);
        let mut inner = self.lock();
        inner.model = None;
        // Cooldown is measured from NOW (eviction complete), not from when it was
        // requested — the anti-thrash window must cover the whole move.
        let now = crate::gpu_exclusive::now_epoch();
        inner.phase = Phase::Cooldown {
            until: now.saturating_add(self.cfg.cooldown_secs),
        };
        info!(reason = ?reason, "coder tier: window returned to assistant mode");
    }
}

/// A review's claim on the coder engine.
///
/// Holding one does not make a review safe — **committing** through it does. The
/// lease is what turns "the engine went away mid-generation" from a silent
/// truncation into an explicit, machine-readable absence.
pub struct Lease {
    id: u64,
    epoch: u64,
    model: Option<String>,
    token: CancelToken,
    tier: Arc<CoderTier>,
}

impl Lease {
    /// The model this lease is serving on.
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// The cancellation token to `select!` the generation future against, so an
    /// in-flight HTTP generation is DROPPED rather than merely ignored — the same
    /// discipline [`super::resident_set`] applies to warm requests.
    pub fn cancel_token(&self) -> CancelToken {
        self.token.clone()
    }

    /// Has this lease been pre-empted? Cheap; safe to poll inside a generation
    /// loop to stop early.
    pub fn is_interrupted(&self) -> bool {
        self.token.is_cancelled()
    }

    /// **The commit gate.** Hand over the review's output; get back either the
    /// output or an explicit interruption.
    ///
    /// `output` is taken BY VALUE, so on the interrupted path it is dropped here
    /// and cannot escape. That is the point: a truncated review that still parses
    /// is the most dangerous possible result, because it reads as a verdict.
    ///
    /// The check is made under the tier lock and covers the whole generation
    /// window, not just its end — the epoch was bumped the moment the user
    /// arrived, so a lease that finished a microsecond later still loses.
    pub fn commit<T>(self, output: T) -> LeaseOutcome<T> {
        let ok = self.tier.settle_lease(self.id, self.epoch) && !self.token.is_cancelled();
        if ok {
            LeaseOutcome::Completed(output)
        } else {
            drop(output);
            LeaseOutcome::Interrupted(InterruptedReport::new())
        }
    }

    /// Abandon the lease without output (e.g. the generation itself errored).
    pub fn abandon(self) {
        let _ = self.tier.settle_lease(self.id, self.epoch);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Production env: consumes the EXISTING machinery, adds no second rule
// ─────────────────────────────────────────────────────────────────────────────

/// The production [`CoderTierEnv`]: the real alias store, registry, memory
/// counter, and on-demand backend lifecycle. Every one of these is an existing
/// Chord primitive — nothing here is a second copy of a residency rule.
pub struct AppStateCoderEnv {
    state: Arc<crate::routes::AppState>,
}

impl AppStateCoderEnv {
    pub fn new(state: Arc<crate::routes::AppState>) -> Self {
        AppStateCoderEnv { state }
    }
}

#[async_trait]
impl CoderTierEnv for AppStateCoderEnv {
    async fn resolve_coder(&self, alias: &str) -> Option<String> {
        // EXACTLY the resident set's resolution: dynamic lumina store first, then
        // the static alias map. No model name, no fallback-to-the-key.
        let model = super::resident_set::resolve_alias(
            alias,
            &self.state.lumina_aliases,
            &self.state.model_aliases,
        )?;
        // Presence is VERIFIED, never assumed (the resident set's own rule): an
        // alias that points at a model nobody pulled resolves to nothing here.
        let reg = self.state.model_registry.lock().await;
        if reg.get(&model).is_some() {
            Some(model)
        } else {
            None
        }
    }

    async fn model_size_gb(&self, model: &str) -> Option<f64> {
        let reg = self.state.model_registry.lock().await;
        // On-DISK size. The runtime inflation factor is applied by the fit test,
        // not here — this seam reports what the registry actually knows.
        reg.get(model)
            .map(|r| r.size_bytes as f64 / 1_073_741_824.0)
            .filter(|gb| *gb > 0.0)
    }

    fn gtt(&self) -> Option<GttReading> {
        sensors::read_gtt()
    }

    fn commit(&self) -> Option<CommitReading> {
        sensors::read_commit()
    }

    async fn start_coder(&self, model: &str) -> Result<(), String> {
        // The EXISTING on-demand start: resolve the model's backend and ensure it
        // is up. This is the same call the chat path makes, so the coder is
        // started by the one lifecycle that already knows how.
        crate::models::routing::resolve_and_ensure(
            &self.state.model_registry,
            &self.state.routing_map,
            model,
            model,
        )
        .await
        .map(|_| ())
        .ok_or_else(|| "coder backend did not come up".to_string())
    }

    async fn stop_coder(&self, model: &str) -> Result<(), String> {
        // The EXISTING lifecycle stop, addressed at the one backend this model
        // routes to. Idempotent by construction (stopping a stopped backend is a
        // no-op in `lifecycle::stop`).
        let stopped =
            crate::models::routing::stop_on_demand_backend_for_model(&self.state.model_registry, model)
                .await;
        if stopped {
            Ok(())
        } else {
            Err("no on-demand backend to stop for this model".to_string())
        }
    }

    async fn restore_assistant_capacity(&self) {
        // Hand the window back through the EXISTING resident-set reconcile — the
        // single owner of assistant residency. We do not warm anything ourselves,
        // and we hold no model list of our own to warm FROM.
        let _ = super::resident_set::global().reconcile(&self.state).await;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Process-global instance + wiring helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Request header a review dispatch sets to identify itself as CODER traffic, so
/// its own inference calls do not look like a user arriving and evict the very
/// engine they are using. Everything WITHOUT this header is assistant traffic.
///
/// Absence is the safe default in the right direction: an unlabelled request is
/// treated as a user, so the worst case is an unnecessary eviction — never a
/// missed one.
pub const LEASE_HEADER: &str = "x-chord-coder-lease";

static GLOBAL: std::sync::OnceLock<Arc<CoderTier>> = std::sync::OnceLock::new();

/// Install the process-global tier. Called once at startup, after `AppState`
/// exists. Idempotent — a second call is ignored.
pub fn init_global(state: Arc<crate::routes::AppState>) -> Arc<CoderTier> {
    GLOBAL
        .get_or_init(|| {
            let cfg = CoderTierConfig::from_env();
            info!(
                enabled = cfg.enabled,
                alias = %cfg.alias,
                min_idle_secs = cfg.min_idle_secs,
                arm_confirm_secs = cfg.arm_confirm_secs,
                cooldown_secs = cfg.cooldown_secs,
                "coder tier: initialised"
            );
            CoderTier::new(cfg, Arc::new(AppStateCoderEnv::new(state)))
        })
        .clone()
}

/// The process-global tier, if one has been installed.
pub fn global() -> Option<&'static Arc<CoderTier>> {
    GLOBAL.get()
}

/// Is this request CODER traffic (i.e. a review running on a lease), rather than
/// a user? Header-presence only — the value is not a credential and is never
/// trusted for anything but this classification.
pub fn is_coder_traffic(headers: &axum::http::HeaderMap) -> bool {
    headers.contains_key(LEASE_HEADER)
}

/// **The one call the inference hot path makes.** Classify the request and, if it
/// is a user, pre-empt any review immediately. Synchronous, allocation-free on
/// the common path, and a single relaxed atomic load when the tier is off.
pub fn note_inference_request(headers: &axum::http::HeaderMap) {
    if let Some(tier) = global() {
        if !is_coder_traffic(headers) {
            tier.note_assistant_request();
        }
    }
}

/// Background observation loop. Reads the SAME activity counters
/// `GET /admin/activity` reports (CHORD-ACT-01) — one signal, not a second one.
pub async fn tick_loop(tier: Arc<CoderTier>) {
    let interval = std::time::Duration::from_secs(policy::env_u64("CHORD_CODER_TIER_TICK_SECS", 60).max(1));
    loop {
        tokio::time::sleep(interval).await;
        let now = crate::gpu_exclusive::now_epoch();
        let idle = &*crate::admin::idle::IDLE_MODE;
        let (serving, idle_secs) = crate::admin::idle::activity_summary(
            idle.inflight_count(),
            idle.last_activity_unix(),
            now as i64,
        );
        tier.tick(now, idle_secs, serving).await;
    }
}

#[cfg(test)]
mod tests;
