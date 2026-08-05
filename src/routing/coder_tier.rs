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
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
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
    /// Would `model` be served by an ON-DEMAND backend?
    ///
    /// A safety PRECONDITION the tier checks before loading, deliberately kept in
    /// the seam so it is exercised by tests rather than living only in the
    /// production impl: a coder that resolves to the ALWAYS-ON serve is running on
    /// the assistant's own engine, and the stop gate will (correctly) refuse to
    /// stop it — so it could never be evicted.
    async fn coder_is_on_demand(&self, model: &str) -> bool;
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
    /// Bumped by every eviction ATTEMPT. A `run_eviction` whose captured value is
    /// stale has been abandoned (force-resolved by the tick budget) and must not
    /// write phase/model — otherwise a wedged teardown can wake up long after a
    /// NEW load and stop the new backend / clobber the new phase.
    evict_epoch: u64,
    /// Live leases' cancel tokens, keyed by the lease's UNGUESSABLE SECRET.
    ///
    /// Keyed by the secret rather than a counter because the key doubles as the
    /// bearer credential that identifies coder traffic on the hot path: a
    /// sequential id would be trivially guessable, letting any authenticated
    /// caller label itself as a review and thereby exempt itself from
    /// pre-emption (see [`is_coder_traffic`]).
    leases: HashMap<String, CancelToken>,
    /// The model currently loaded/loading, when any.
    model: Option<String>,
    /// How many teardown tasks are still RUNNING — including ones the
    /// force-resolve budget has already abandoned. See
    /// [`policy::Observation::teardown_inflight`]: a stop that is already in
    /// flight cannot be un-issued by bumping a counter, so a new load must wait
    /// for it to actually finish or the late stop lands on the NEW coder.
    teardowns_inflight: usize,
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
    /// Epoch seconds of the last ASSISTANT request, stamped by the hot-path hook.
    ///
    /// The tier keeps its OWN stamp rather than reading the global
    /// `IdleController` counters, because those count EVERY admitted inference —
    /// including a review's own calls on the coder. Reading them made the feature
    /// self-defeating: a running review kept `idle_secs` near zero, so the very
    /// next tick evicted the coder the review was using.
    last_assistant_unix: AtomicI64,
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
                evict_epoch: 0,
                leases: HashMap::new(),
                model: None,
                teardowns_inflight: 0,
            }),
            env,
            interrupts: AtomicU64::new(0),
            loads: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            last_assistant_unix: AtomicI64::new(crate::gpu_exclusive::now_epoch() as i64),
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

    /// Epoch seconds of the last ASSISTANT request seen by the hot-path hook.
    pub fn last_assistant_unix(&self) -> i64 {
        self.last_assistant_unix.load(Ordering::Relaxed)
    }

    /// How long since the last ASSISTANT request, for a tick at `now`.
    ///
    /// Derived from the tier's OWN stamp, never from `IdleController`'s counters:
    /// those count EVERY admitted inference, including a review's own calls on the
    /// coder, which made the feature self-defeating — a running review held
    /// `idle_secs` near zero, so the next tick evicted the coder the review was
    /// using. Only [`CoderTier::note_assistant_request`] moves this, and it is
    /// called only for non-coder traffic.
    pub fn assistant_idle_secs(&self, now: u64) -> u64 {
        (now as i64)
            .saturating_sub(self.last_assistant_unix())
            .max(0) as u64
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
        // Stamp BEFORE anything else: this is the authoritative record of user
        // activity, and it is deliberately not shared with the coder's own calls.
        self.last_assistant_unix.store(now as i64, Ordering::Relaxed);
        let (tokens, evict) = {
            let mut inner = self.lock();
            // Bump FIRST: any lease that has already generated output is invalid
            // from this instant, whether or not it ever observes its token.
            inner.epoch = inner.epoch.wrapping_add(1);
            let tokens: Vec<CancelToken> = inner.leases.drain().map(|(_, t)| t).collect();
            let evict = match inner.phase {
                Phase::Loading | Phase::Ready => {
                    inner.phase = Phase::Evicting { since: now };
                    // Belt-and-braces, and PROVABLY REDUNDANT TODAY — kept
                    // deliberately, and flagged so nobody "fixes" it.
                    //
                    // What actually makes the abandonment guard work is (a) the
                    // generation being captured HERE, at REQUEST time, and (b)
                    // `ForceResolveEvict` bumping it. This extra bump changes
                    // nothing observable while at most ONE teardown can be in
                    // flight — the `Evicting` phase gates every other eviction
                    // decision, so generations are already serialized. Removing
                    // it is an EQUIVALENT MUTANT (verified: it survives the
                    // suite, and the equivalence is the phase gate, not a missing
                    // test). It exists so that if concurrent teardowns ever become
                    // possible, each still gets a distinct generation.
                    inner.evict_epoch = inner.evict_epoch.wrapping_add(1);
                    inner.teardowns_inflight += 1;
                    true
                }
                Phase::Arming { .. } => {
                    inner.phase = Phase::Assistant;
                    false
                }
                _ => false,
            };
            (tokens, evict.then_some(inner.evict_epoch))
        };
        // Latching is cheap and lock-free; do it outside the critical section.
        let interrupted = tokens.len() as u64;
        for t in &tokens {
            t.cancel();
        }
        if interrupted > 0 {
            self.interrupts.fetch_add(interrupted, Ordering::Relaxed);
        }
        if let Some(my_evict_epoch) = evict {
            let me = self.clone();
            // Detached: the user's request must not wait on teardown. If we are
            // somehow not on a runtime, the tick's budget backstop resolves the
            // phase instead — we still never block here.
            if tokio::runtime::Handle::try_current().is_ok() {
                tokio::spawn(async move {
                    me.run_eviction(EvictReason::AssistantArrived, my_evict_epoch)
                        .await;
                });
            } else {
                warn!(
                    "coder tier: no runtime on the eviction path; teardown deferred to the tick"
                );
            }
        }
        evict.is_some()
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
        // An unguessable, single-use credential. It is what the review dispatch
        // puts in the `x-chord-coder-lease` header, and the ONLY thing that makes
        // a request count as coder traffic.
        let secret = uuid::Uuid::new_v4().to_string();
        let token = CancelToken::new();
        inner.leases.insert(secret.clone(), token.clone());
        let epoch = inner.epoch;
        let model = inner.model.clone();
        drop(inner);
        Some(Lease {
            secret,
            settled: false,
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
    fn settle_lease(&self, secret: &str, epoch: u64, token: &CancelToken) -> bool {
        let mut inner = self.lock();
        // ALL THREE guards are evaluated under the SAME lock acquisition, so the
        // commit has a single linearization point. Checking the token after
        // releasing the lock left a window in which a pre-emption could land
        // between the epoch check and the token read, and the commit would then
        // observe neither — raised in review, closed by making the decision
        // atomic rather than by arguing about whether the window was reachable.
        let still_registered = inner.leases.remove(secret).is_some();
        still_registered && inner.epoch == epoch && !token.is_cancelled()
    }

    /// Is `secret` a LIVE lease? This is what distinguishes coder traffic from a
    /// user on the hot path, and it is deliberately a membership test against
    /// minted secrets rather than mere header presence.
    pub fn is_live_lease(&self, secret: &str) -> bool {
        !secret.is_empty() && self.lock().leases.contains_key(secret)
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

    /// Test-only: how many leases the tier currently tracks.
    #[cfg(test)]
    fn live_lease_count(&self) -> usize {
        self.lock().leases.len()
    }

    /// Test-only: age the assistant stamp, so a fixture can distinguish the
    /// tier's own stamp from any global counter that happens to be seeded at the
    /// same construction time (they coincide in a test process, which made two
    /// mutants survive a fixture that could not tell them apart).
    #[cfg(test)]
    fn set_last_assistant_for_test(&self, unix: i64) {
        self.last_assistant_unix.store(unix, Ordering::Relaxed);
    }

    /// Test-only: the current eviction generation.
    #[cfg(test)]
    fn evict_epoch(&self) -> u64 {
        self.lock().evict_epoch
    }

    /// Test-only: abandon any outstanding teardown WITHOUT running the rest of
    /// the force-resolve path, so each abandonment guard can be exercised alone.
    #[cfg(test)]
    fn bump_evict_epoch_only(&self) {
        let mut inner = self.lock();
        inner.evict_epoch = inner.evict_epoch.wrapping_add(1);
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
            teardown_inflight: { self.lock().teardowns_inflight > 0 },
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
                // PRECONDITION: never take the window on the assistant's own
                // engine. Checked before the phase moves, so a refusal is a clean
                // no-op rather than a load that must be unwound.
                if !self.env.coder_is_on_demand(&model).await {
                    warn!(
                        model = %model,
                        "coder tier: resolved coder would run on the always-on assistant engine — refusing"
                    );
                    let mut inner = self.lock();
                    if inner.phase == phase {
                        inner.phase = Phase::Cooldown {
                            until: now.saturating_add(self.cfg.cooldown_secs),
                        };
                    }
                    return;
                }
                {
                    let mut inner = self.lock();
                    if inner.phase != phase {
                        return; // pre-empted between decide and act
                    }
                    inner.phase = Phase::Loading;
                    inner.model = Some(model.clone());
                }
                let started = self.env.start_coder(&model).await;
                // ONE lock acquisition decides everything about this load.
                //
                // This used to be two — "did I lose the race?" and then, after
                // dropping and retaking the lock, "am I still Loading?" — which
                // left a window between them that no test could force, and a
                // mutant that deleted the second check survived precisely because
                // the window is unreachable from a test seam. The fix is to remove
                // the window rather than to write a test that cannot see it.
                // NOTE: no logging inside this critical section. `tracing` macros
                // can block on a synchronous subscriber's writer, and an incoming
                // user acquires this same mutex on the hot path — so the section is
                // strictly bounded field updates, and the logs happen after.
                let outcome = {
                    let mut inner = self.lock();
                    if inner.phase == Phase::Loading {
                        match &started {
                            Ok(()) => {
                                inner.phase = Phase::Ready;
                                self.loads.fetch_add(1, Ordering::Relaxed);
                                Some(true)
                            }
                            Err(_) => {
                                inner.phase = Phase::Cooldown {
                                    until: now.saturating_add(self.cfg.cooldown_secs),
                                };
                                inner.model = None;
                                Some(false)
                            }
                        }
                    } else {
                        None
                    }
                };
                match (&outcome, &started) {
                    (Some(true), _) => {
                        info!(model = %model, "coder tier: coder loaded into the reaped window")
                    }
                    (Some(false), Err(e)) => {
                        warn!(error = %e, "coder tier: coder load failed — cooling down")
                    }
                    _ => {}
                }
                let lost_the_race = outcome.is_none();
                if lost_the_race && started.is_ok() {
                    // A user arrived DURING the load and the eviction task has
                    // already run. Declining to install `Ready` is NOT enough: the
                    // eviction's `stop_coder` may have run BEFORE this
                    // `start_coder` actually launched the backend, in which case
                    // the coder is now running with the tier recording
                    // `model: None` — borrowed memory nobody will ever give back.
                    // So compensate, exactly as the resident set compensates for a
                    // warm request it could not un-issue. Idempotent: stopping an
                    // already-stopped backend is a no-op.
                    warn!(
                        model = %model,
                        "coder tier: load completed after losing the race — compensating with a stop"
                    );
                    if let Err(e) = self.env.stop_coder(&model).await {
                        warn!(error = %e, model = %model,
                            "coder tier: compensating stop failed");
                    }
                }
            }
            Decision::Evict(reason) => {
                let (tokens, my_evict_epoch): (Vec<CancelToken>, u64) = {
                    let mut inner = self.lock();
                    if inner.phase != phase {
                        return; // pre-empted between decide and act
                    }
                    inner.phase = Phase::Evicting { since: now };
                    inner.epoch = inner.epoch.wrapping_add(1);
                    inner.evict_epoch = inner.evict_epoch.wrapping_add(1);
                    inner.teardowns_inflight += 1;
                    let gen = inner.evict_epoch;
                    (inner.leases.drain().map(|(_, t)| t).collect(), gen)
                };
                let n = tokens.len() as u64;
                for t in &tokens {
                    t.cancel();
                }
                if n > 0 {
                    self.interrupts.fetch_add(n, Ordering::Relaxed);
                }
                self.run_eviction(reason, my_evict_epoch).await;
            }
            Decision::ForceResolveEvict => {
                // Take ownership of the abandonment FIRST: bumping `evict_epoch`
                // makes the wedged `run_eviction` a no-op if it ever wakes, so it
                // cannot later stop a NEWLY loaded coder or clobber a newer phase.
                {
                    let mut inner = self.lock();
                    if inner.phase != phase {
                        return;
                    }
                    inner.evict_epoch = inner.evict_epoch.wrapping_add(1);
                }
                warn!("coder tier: eviction overran its budget — restoring and forcing cooldown");
                // SAFETY 3: the cohort must never be left unrestored. This is
                // PRECISELY the wedged-teardown case, so it is the last place that
                // can restore — and it is idempotent, so doing it here as well as
                // in a teardown that later completes is harmless.
                self.env.restore_assistant_capacity().await;
                let mut inner = self.lock();
                inner.model = None;
                inner.phase = Phase::Cooldown {
                    until: now.saturating_add(self.cfg.cooldown_secs),
                };
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
    /// `my_evict_epoch` is captured by the CALLER at the instant the eviction is
    /// REQUESTED, and passed in — it is deliberately not read here.
    ///
    /// Reading it at the top of this function was a real bug (caught by a test,
    /// after a mutant survived a fixture that could not see it): this body does
    /// not run until the spawned task is first polled, so a tick that
    /// force-resolved the eviction in between would already have bumped the
    /// counter, and this task would then capture the NEW value and consider
    /// itself current. Capturing at request time closes that window.
    pub async fn run_eviction(self: &Arc<Self>, reason: EvictReason, my_evict_epoch: u64) {
        self.run_eviction_inner(reason, my_evict_epoch).await;
        // ALWAYS, on every exit path: only once this decrements can a new load
        // begin, which is what stops a late `stop_coder` from hitting a newer
        // cycle's coder.
        let mut inner = self.lock();
        inner.teardowns_inflight = inner.teardowns_inflight.saturating_sub(1);
    }

    async fn run_eviction_inner(self: &Arc<Self>, reason: EvictReason, my_evict_epoch: u64) {
        let model = {
            let inner = self.lock();
            // Already abandoned before we even started: do NOT stop anything. By
            // now a newer cycle may have loaded its own coder, and `inner.model`
            // would name THAT one.
            if inner.evict_epoch != my_evict_epoch {
                warn!("coder tier: teardown abandoned before it began — standing down");
                return;
            }
            inner.model.clone()
        };
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
        // If this teardown was ABANDONED (force-resolved past its budget) the
        // eviction epoch has moved and a newer cycle may already own the tier.
        // Restoring capacity above was still correct and idempotent; writing
        // phase/model here would not be.
        if inner.evict_epoch != my_evict_epoch {
            warn!("coder tier: abandoned teardown completed late — not touching newer state");
            return;
        }
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
    secret: String,
    /// Has this lease already been settled by `commit`/`abandon`? Drives the
    /// [`Drop`] cleanup below.
    settled: bool,
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
    pub fn commit<T>(mut self, output: T) -> LeaseOutcome<T> {
        let ok = self.tier.settle_lease(&self.secret, self.epoch, &self.token);
        self.settled = true;
        if ok {
            LeaseOutcome::Completed(output)
        } else {
            drop(output);
            LeaseOutcome::Interrupted(InterruptedReport::new())
        }
    }

    /// Abandon the lease without output (e.g. the generation itself errored).
    pub fn abandon(mut self) {
        let _ = self.tier.settle_lease(&self.secret, self.epoch, &self.token);
        self.settled = true;
    }

    /// The bearer credential this lease's inference calls must send in
    /// [`LEASE_HEADER`] so they are recognised as coder traffic.
    pub fn header_value(&self) -> &str {
        &self.secret
    }
}

/// A lease that is DROPPED without being committed or abandoned — a panicking
/// review, a cancelled task, a caller that simply forgot — must not leave its
/// secret registered.
///
/// A stale registered secret keeps passing [`CoderTier::is_live_lease`], so a
/// request bearing it stays classified as coder traffic and is exempt from
/// pre-emption forever. That is the same hole as the original presence-only
/// header check, reintroduced by a leak instead of by a forgery.
impl Drop for Lease {
    fn drop(&mut self) {
        if !self.settled {
            let mut inner = self.tier.lock();
            inner.leases.remove(&self.secret);
        }
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

    async fn coder_is_on_demand(&self, model: &str) -> bool {
        // The SAME arch-aware resolution stop uses, so start and stop can never
        // disagree about which backend they mean.
        crate::models::routing::model_has_on_demand_backend(
            &self.state.model_registry,
            &self.state.routing_map,
            model,
        )
        .await
    }

    async fn start_coder(&self, model: &str) -> Result<(), String> {
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
        let stopped = crate::models::routing::stop_on_demand_backend_for_model(
            &self.state.model_registry,
            &self.state.routing_map,
            model,
        )
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

/// Is this request CODER traffic (a review running on a LIVE lease), rather than
/// a user?
///
/// **Header PRESENCE is not sufficient and was a real hole.** The first version
/// tested only that the header existed, which let any authenticated caller label
/// itself a review and exempt itself from pre-emption — i.e. a user request could
/// leave the coder resident, defeating the whole safety property. The header must
/// now carry the unguessable secret minted by [`CoderTier::try_acquire_lease`],
/// and it is checked for MEMBERSHIP against the live lease set.
///
/// Everything else is a user. Absence, a stale secret, a forged secret, and a
/// malformed value all classify as a user — the safe direction, since the worst
/// case is an unnecessary eviction rather than a missed one.
pub fn is_coder_traffic(tier: &CoderTier, headers: &axum::http::HeaderMap) -> bool {
    headers
        .get(LEASE_HEADER)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|secret| tier.is_live_lease(secret))
}

/// **The one call the inference hot path makes.** Classify the request and, if it
/// is a user, pre-empt any review immediately. Synchronous, allocation-free on
/// the common path, and a single relaxed atomic load when the tier is off.
pub fn note_inference_request(headers: &axum::http::HeaderMap) {
    if let Some(tier) = global() {
        if !is_coder_traffic(tier, headers) {
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
        // `assistant_inflight` is false here on purpose: the synchronous hot-path
        // hook is the authoritative pre-emption, and any request that has STARTED
        // has already moved the tier's stamp, so the tick needs only the idle age.
        tier.tick(now, tier.assistant_idle_secs(now), false).await;
    }
}

#[cfg(test)]
mod tests;
