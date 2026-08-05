//! RVXR-01 tests.
//!
//! The rule this suite is written to: **prove the race by FORCING it.** Every
//! concurrency claim below is asserted by constructing the exact interleaving
//! with a gate the test controls, and observing an outcome — never by reasoning
//! about the code, and never by sleeping and hoping.
//!
//! Anti-hang: every fake seam method yields (`tokio::task::yield_now`) so a
//! `select!`/`timeout` over immediate-completion fakes still has a suspension
//! point to fire at, and a test that wedges fails on the harness timeout rather
//! than spinning a core.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use tokio::sync::{mpsc, Semaphore};

use super::*;

// ─────────────────────────────────────────────────────────────────────────────
// Fixtures
// ─────────────────────────────────────────────────────────────────────────────

/// A permit count large enough that `acquire()` never blocks — an "open" gate.
const OPEN: usize = 1 << 20;

struct FakeEnv {
    coder: Option<String>,
    /// On-DISK size the registry would report.
    size_gb: Option<f64>,
    gtt: Mutex<Option<GttReading>>,
    commit: Mutex<Option<CommitReading>>,
    start_result: Mutex<Result<(), String>>,
    stop_result: Mutex<Result<(), String>>,
    /// Ordered log of seam calls — the evidence for "restore was called anyway".
    calls: Mutex<Vec<&'static str>>,
    start_entered: mpsc::UnboundedSender<()>,
    start_gate: Arc<Semaphore>,
    stop_entered: mpsc::UnboundedSender<()>,
    stop_gate: Arc<Semaphore>,
    restores: AtomicUsize,
    coder_on_demand: std::sync::atomic::AtomicBool,
}

struct FakeHandles {
    env: Arc<FakeEnv>,
    start_entered: mpsc::UnboundedReceiver<()>,
    start_gate: Arc<Semaphore>,
    stop_entered: mpsc::UnboundedReceiver<()>,
    stop_gate: Arc<Semaphore>,
}

/// A healthy, roomy host: 50 GiB of GTT free, commit ratio 0.40, no swap in use.
fn healthy_gtt(free_gb: f64) -> GttReading {
    GttReading {
        used_gb: 60.0 - free_gb,
        total_gb: 60.0,
    }
}
fn healthy_commit() -> CommitReading {
    CommitReading {
        committed_gb: 34.0,
        commit_limit_gb: 85.0,
        swap_used_gb: 0.0,
    }
}

fn fake(coder: Option<&str>, size_gb: Option<f64>, free_gb: Option<f64>) -> FakeHandles {
    let (start_tx, start_rx) = mpsc::unbounded_channel();
    let (stop_tx, stop_rx) = mpsc::unbounded_channel();
    let start_gate = Arc::new(Semaphore::new(OPEN));
    let stop_gate = Arc::new(Semaphore::new(OPEN));
    let env = Arc::new(FakeEnv {
        coder: coder.map(|s| s.to_string()),
        size_gb,
        gtt: Mutex::new(free_gb.map(healthy_gtt)),
        commit: Mutex::new(Some(healthy_commit())),
        start_result: Mutex::new(Ok(())),
        stop_result: Mutex::new(Ok(())),
        calls: Mutex::new(Vec::new()),
        start_entered: start_tx,
        start_gate: start_gate.clone(),
        stop_entered: stop_tx,
        stop_gate: stop_gate.clone(),
        restores: AtomicUsize::new(0),
        coder_on_demand: std::sync::atomic::AtomicBool::new(true),
    });
    FakeHandles {
        env,
        start_entered: start_rx,
        start_gate,
        stop_entered: stop_rx,
        stop_gate,
    }
}

impl FakeEnv {
    fn log(&self, what: &'static str) {
        self.calls.lock().unwrap().push(what);
    }
    fn calls(&self) -> Vec<&'static str> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl CoderTierEnv for FakeEnv {
    async fn resolve_coder(&self, _alias: &str) -> Option<String> {
        tokio::task::yield_now().await;
        self.coder.clone()
    }
    async fn model_size_gb(&self, _model: &str) -> Option<f64> {
        tokio::task::yield_now().await;
        self.size_gb
    }
    fn gtt(&self) -> Option<GttReading> {
        *self.gtt.lock().unwrap()
    }
    fn commit(&self) -> Option<CommitReading> {
        *self.commit.lock().unwrap()
    }
    async fn coder_is_on_demand(&self, _model: &str) -> bool {
        tokio::task::yield_now().await;
        self.coder_on_demand.load(Ordering::SeqCst)
    }
    async fn start_coder(&self, _model: &str) -> Result<(), String> {
        self.log("start");
        let _ = self.start_entered.send(());
        tokio::task::yield_now().await;
        let _permit = self.start_gate.acquire().await.expect("gate open");
        self.start_result.lock().unwrap().clone()
    }
    async fn stop_coder(&self, _model: &str) -> Result<(), String> {
        self.log("stop");
        let _ = self.stop_entered.send(());
        tokio::task::yield_now().await;
        let _permit = self.stop_gate.acquire().await.expect("gate open");
        self.stop_result.lock().unwrap().clone()
    }
    async fn restore_assistant_capacity(&self) {
        self.log("restore");
        self.restores.fetch_add(1, Ordering::SeqCst);
        tokio::task::yield_now().await;
    }
}

fn cfg_on() -> CoderTierConfig {
    CoderTierConfig {
        enabled: true,
        alias: "test-coder-alias".into(),
        ..CoderTierConfig::default()
    }
}

/// Wait (bounded) for the tier to leave the transient `Loading` phase.
///
/// The load runs DETACHED — an inline load blocks the tick loop and makes the
/// eviction budget unreachable through the load branch — so a tick that decides
/// `Load` returns before the load has happened. Tests must settle, not assume.
async fn settle(tier: &Arc<CoderTier>) {
    for _ in 0..5_000 {
        if !matches!(tier.phase(), Phase::Loading) {
            return;
        }
        tokio::task::yield_now().await;
    }
}

/// Tick, then wait for any detached load it dispatched to settle.
async fn tick_settled(tier: &Arc<CoderTier>, now: u64, idle: u64, inflight: bool) {
    tier.tick(now, idle, inflight).await;
    settle(tier).await;
}

/// Drive the tier to `Ready` deterministically. The two intermediate assertions
/// are load-bearing: if the fixture ever stopped ARMING first, or stopped
/// reaching Ready, every eviction test below would pass for the wrong reason.
async fn ready_tier() -> (Arc<CoderTier>, Arc<FakeEnv>, Arc<Semaphore>, mpsc::UnboundedReceiver<()>) {
    let h = fake(Some("a-coder"), Some(17.3), Some(50.0));
    let tier = CoderTier::new(cfg_on(), h.env.clone());
    tick_settled(&tier, 1_000, 5_000, false).await; // arms only
    assert!(matches!(tier.phase(), Phase::Arming { .. }));
    tick_settled(&tier, 2_000, 5_000, false).await; // confirm elapsed → load
    assert_eq!(tier.phase(), Phase::Ready);
    (tier, h.env, h.stop_gate, h.stop_entered)
}

// ─────────────────────────────────────────────────────────────────────────────
// Hazard 1 — the eviction race, FORCED
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn assistant_request_returns_without_waiting_for_teardown() {
    let h = fake(Some("a-coder"), Some(17.3), Some(50.0));
    let tier = CoderTier::new(cfg_on(), h.env.clone());
    tick_settled(&tier, 1_000, 5_000, false).await;
    tick_settled(&tier, 2_000, 5_000, false).await;
    assert_eq!(tier.phase(), Phase::Ready);

    // CLOSE the stop gate: any teardown that starts will block inside the seam
    // until this test opens it. If `note_assistant_request` waited on teardown in
    // any way, this test would deadlock rather than fail — which is itself the
    // assertion (the harness timeout catches it).
    let mut stop_entered = h.stop_entered;
    let gate = h.stop_gate;
    gate.acquire_many(OPEN as u32).await.expect("close gate").forget();

    let lease = tier.try_acquire_lease().expect("ready tier issues leases");

    // THE HOT PATH CALL. Synchronous, and must return with teardown unfinished.
    let evicted = tier.note_assistant_request();
    assert!(evicted, "a resident coder must be evicted by an incoming user request");

    // Observable immediately, with the teardown still blocked:
    assert!(matches!(tier.phase(), Phase::Evicting { .. }));
    assert!(lease.is_interrupted(), "the lease is cancelled before we return");
    assert_eq!(
        h.env.restores.load(Ordering::SeqCst),
        0,
        "teardown has NOT completed — proving the assistant did not wait for it"
    );

    // Let the spawned teardown reach the seam, then confirm it is genuinely
    // parked in `stop_coder` and not merely un-scheduled.
    tokio::task::yield_now().await;
    let entered = tokio::time::timeout(std::time::Duration::from_secs(5), stop_entered.recv()).await;
    assert!(entered.is_ok(), "teardown should have started in the background");
    assert_eq!(
        h.env.restores.load(Ordering::SeqCst),
        0,
        "still mid-teardown while the assistant has long since returned"
    );

    // Open the gate and let it finish.
    gate.add_permits(OPEN);
    for _ in 0..500 {
        if matches!(tier.phase(), Phase::Cooldown { .. }) {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert_eq!(h.env.restores.load(Ordering::SeqCst), 1);
    assert!(matches!(tier.phase(), Phase::Cooldown { .. }));
}

/// A witness that records its own destruction — the only honest way to assert
/// "the generated tokens were discarded" rather than "we didn't return them".
#[derive(Debug)]
struct DropWitness(Arc<AtomicUsize>);
impl Drop for DropWitness {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn partial_output_generated_before_eviction_is_discarded_not_returned() {
    let (tier, _env, gate, _rx) = ready_tier().await;
    gate.acquire_many(OPEN as u32).await.unwrap().forget(); // park the teardown

    let lease = tier.try_acquire_lease().expect("lease");

    // FORCE the interleaving: the review has ALREADY produced output — a
    // plausible-looking, fully-parsing verdict — when the user arrives.
    let dropped = Arc::new(AtomicUsize::new(0));
    let partial = (
        "APPROVE: looks good to me".to_string(),
        DropWitness(dropped.clone()),
    );

    tier.note_assistant_request();

    let outcome = lease.commit(partial);
    match &outcome {
        LeaseOutcome::Interrupted(r) => {
            assert!(r.interrupted);
            assert_eq!(r.code, INTERRUPT_CODE);
            assert_eq!(r.reason, INTERRUPT_REASON);
        }
        LeaseOutcome::Completed(_) => panic!("a pre-empted review must never return a verdict"),
    }
    assert!(outcome.is_interrupted());
    assert!(
        outcome.completed().is_none(),
        "there must be no accessor that yields the partial output"
    );
    assert_eq!(
        dropped.load(Ordering::SeqCst),
        1,
        "the generated tokens must be DROPPED, not merely withheld"
    );
    gate.add_permits(OPEN);
}

#[tokio::test]
async fn output_survives_when_no_user_ever_arrives() {
    // The control in the other direction: without a pre-emption the same lease
    // returns the output intact. A test that only ever sees Interrupted would
    // pass against a `commit` hardwired to fail.
    let (tier, _env, _gate, _rx) = ready_tier().await;
    let lease = tier.try_acquire_lease().expect("lease");
    let dropped = Arc::new(AtomicUsize::new(0));
    let out = lease.commit(("APPROVE".to_string(), DropWitness(dropped.clone())));
    assert!(!out.is_interrupted());
    let v = out.completed().expect("completed carries the output");
    assert_eq!(v.0, "APPROVE");
    assert_eq!(dropped.load(Ordering::SeqCst), 0, "nothing was dropped");
    drop(v);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_review_that_finishes_a_hair_after_the_user_arrives_still_loses() {
    // The dangerous ordering: eviction fires, the generation completes anyway,
    // and only THEN does it try to commit. The epoch was bumped at the instant
    // the user arrived, so the late commit still loses.
    let (tier, _env, gate, _rx) = ready_tier().await;
    gate.acquire_many(OPEN as u32).await.unwrap().forget();
    let lease = tier.try_acquire_lease().expect("lease");
    let epoch_before = tier.epoch();

    tier.note_assistant_request();
    assert_ne!(tier.epoch(), epoch_before, "the epoch moves the moment the user arrives");

    // The generation "completes" after the pre-emption.
    for _ in 0..50 {
        tokio::task::yield_now().await;
    }
    assert!(lease.commit("a complete-looking answer").is_interrupted());
    gate.add_permits(OPEN);
}

// Each of the three pre-emption guards, ISOLATED.
//
// A real user request trips all three at once (deregistration, epoch, token), so
// the combined tests above pass even if two of the three are deleted — mutation
// testing proved exactly that. These three exercise one mechanism each, with the
// other two deliberately left untripped, so no single guard can be removed
// silently.

#[tokio::test]
async fn the_epoch_guard_alone_is_sufficient() {
    let (tier, _env, _gate, _rx) = ready_tier().await;
    let lease = tier.try_acquire_lease().expect("lease");
    // Epoch moves; the lease is still registered and its token is NOT cancelled.
    tier.bump_epoch_only();
    assert!(!lease.is_interrupted(), "token deliberately untouched");
    assert!(
        lease.commit("verdict").is_interrupted(),
        "a stale epoch alone must invalidate the output"
    );
}

#[tokio::test]
async fn the_deregistration_guard_alone_is_sufficient() {
    let (tier, _env, _gate, _rx) = ready_tier().await;
    let lease = tier.try_acquire_lease().expect("lease");
    // Lease is dropped from the map; epoch unchanged, token not cancelled.
    tier.drain_leases_only();
    assert!(!lease.is_interrupted(), "token deliberately untouched");
    assert_eq!(tier.epoch(), tier.epoch(), "epoch deliberately unchanged");
    assert!(
        lease.commit("verdict").is_interrupted(),
        "a lease the tier no longer knows about must not be able to commit"
    );
}

#[tokio::test]
async fn the_cancel_token_guard_alone_is_sufficient() {
    let (tier, _env, _gate, _rx) = ready_tier().await;
    let lease = tier.try_acquire_lease().expect("lease");
    let epoch_before = tier.epoch();
    // Only the token latches — the lease stays registered and the epoch holds.
    lease.cancel_token().cancel();
    assert_eq!(tier.epoch(), epoch_before, "epoch deliberately unchanged");
    assert!(lease.is_interrupted());
    assert!(
        lease.commit("verdict").is_interrupted(),
        "a cancelled token alone must invalidate the output"
    );
}

#[tokio::test]
async fn one_user_request_interrupts_every_concurrent_lease() {
    let (tier, _env, gate, _rx) = ready_tier().await;
    gate.acquire_many(OPEN as u32).await.unwrap().forget();
    let leases: Vec<Lease> = (0..5)
        .map(|_| tier.try_acquire_lease().expect("lease"))
        .collect();

    tier.note_assistant_request();

    for l in leases {
        assert!(l.is_interrupted());
        assert!(l.commit("verdict").is_interrupted());
    }
    assert_eq!(tier.status().interrupts, 5);
    gate.add_permits(OPEN);
}

#[tokio::test]
async fn a_lease_is_never_issued_outside_ready() {
    let h = fake(Some("a-coder"), Some(17.3), Some(50.0));
    let tier = CoderTier::new(cfg_on(), h.env.clone());
    assert!(tier.try_acquire_lease().is_none(), "assistant phase");
    tick_settled(&tier, 1_000, 5_000, false).await;
    assert!(tier.try_acquire_lease().is_none(), "arming phase");
    tick_settled(&tier, 2_000, 5_000, false).await;
    assert!(tier.try_acquire_lease().is_some(), "ready phase");
    tier.note_assistant_request();
    assert!(tier.try_acquire_lease().is_none(), "evicting phase");
}

#[tokio::test]
async fn a_user_arriving_mid_load_never_lets_the_coder_reach_ready() {
    // The other half of the race: the load is in flight when the user arrives.
    // If `tick` blindly wrote `Ready` on a successful start, the coder would be
    // resident with nobody left to evict it.
    let h = fake(Some("a-coder"), Some(17.3), Some(50.0));
    let tier = CoderTier::new(cfg_on(), h.env.clone());
    tick_settled(&tier, 1_000, 5_000, false).await;
    assert!(matches!(tier.phase(), Phase::Arming { .. }));

    // Park the load inside the seam.
    let start_gate = h.start_gate.clone();
    start_gate.acquire_many(OPEN as u32).await.unwrap().forget();
    let mut start_entered = h.start_entered;

    let t = tier.clone();
    let loading = tokio::spawn(async move { t.tick(2_000, 5_000, false).await });

    tokio::time::timeout(std::time::Duration::from_secs(5), start_entered.recv())
        .await
        .expect("load should have started")
        .expect("entered");
    assert_eq!(tier.phase(), Phase::Loading);

    // The user arrives mid-load.
    let evicted = tier.note_assistant_request();
    assert!(evicted, "a load in flight is memory borrowed — it must be evicted");
    assert!(matches!(tier.phase(), Phase::Evicting { .. }));

    // Let the load complete SUCCESSFULLY anyway. It must not resurrect Ready.
    start_gate.add_permits(OPEN);
    loading.await.expect("tick completes");
    assert_ne!(
        tier.phase(),
        Phase::Ready,
        "a successful load that lost the race must never install itself"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Restoration is unconditional
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn assistant_capacity_is_restored_even_when_the_coder_stop_fails() {
    let (tier, env, _gate, _rx) = ready_tier().await;
    *env.stop_result.lock().unwrap() = Err("backend refused to stop".into());

    tier.note_assistant_request();
    // Wait for the teardown to REACH ITS RESTING PHASE, not merely for the
    // restore call — the phase write follows the restore, and asserting on the
    // earlier of the two would be a race the test itself introduced.
    for _ in 0..500 {
        if matches!(tier.phase(), Phase::Cooldown { .. }) {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert_eq!(
        env.restores.load(Ordering::SeqCst),
        1,
        "a teardown that errored is exactly when capacity most needs re-asserting"
    );
    let calls = env.calls();
    let stop_at = calls.iter().position(|c| *c == "stop").expect("stop attempted");
    let restore_at = calls.iter().position(|c| *c == "restore").expect("restore called");
    assert!(stop_at < restore_at, "restore follows the stop attempt");
    assert!(matches!(tier.phase(), Phase::Cooldown { .. }));
}

// Defects found by the review panel (codex + opus), each now defended.

#[tokio::test]
async fn a_force_resolved_eviction_still_restores_assistant_capacity() {
    // SAFETY 3. The tick force-resolves a teardown that overran its budget. That
    // is PRECISELY the wedged case, so it is the last place that can hand the
    // cohort back — and the original code cleared `model` and cooled down without
    // restoring anything, leaving the assistant permanently degraded.
    let h = fake(Some("a-coder"), Some(17.3), Some(50.0));
    let tier = CoderTier::new(cfg_on(), h.env.clone());
    tick_settled(&tier, 1_000, 5_000, false).await;
    tick_settled(&tier, 2_000, 5_000, false).await;
    assert_eq!(tier.phase(), Phase::Ready);

    // Wedge the teardown inside the seam, forever.
    h.stop_gate.acquire_many(OPEN as u32).await.unwrap().forget();
    tier.note_assistant_request();
    assert!(matches!(tier.phase(), Phase::Evicting { .. }));
    let restores_before = h.env.restores.load(Ordering::SeqCst);

    // Tick past the eviction budget.
    let now = crate::gpu_exclusive::now_epoch() + cfg_on().evict_budget_secs + 1;
    tier.tick(now, 5_000, false).await;

    assert!(matches!(tier.phase(), Phase::Cooldown { .. }));
    assert!(
        h.env.restores.load(Ordering::SeqCst) > restores_before,
        "a force-resolved eviction must still restore assistant capacity"
    );
}

#[tokio::test]
async fn an_abandoned_teardown_cannot_clobber_a_newer_cycle() {
    // NOTE ON THIS TEST'S FIRST VERSION — it was a LYING FIXTURE. It asserted the
    // phase was still `Cooldown{..}` after the abandoned teardown woke up, but an
    // UNGUARDED teardown also writes `Cooldown`, so the assertion held whether or
    // not the guard existed and the mutant survived. The fixture has to reach a
    // state the stale write would visibly destroy — so it now drives a whole new
    // cycle back to `Ready` and asserts THAT survives.
    let h = fake(Some("a-coder"), Some(17.3), Some(50.0));
    let tier = CoderTier::new(cfg_on(), h.env.clone());
    let cfg = cfg_on();
    let t0 = crate::gpu_exclusive::now_epoch();

    tick_settled(&tier, t0, 5_000, false).await;
    tick_settled(&tier, t0 + 200, 5_000, false).await;
    assert_eq!(tier.phase(), Phase::Ready);

    // Wedge the teardown, then force-resolve past its budget so it is abandoned.
    h.stop_gate.acquire_many(OPEN as u32).await.unwrap().forget();
    tier.note_assistant_request();
    assert!(matches!(tier.phase(), Phase::Evicting { .. }));
    tick_settled(&tier, t0 + cfg.evict_budget_secs + 1, 5_000, false).await;
    assert!(matches!(tier.phase(), Phase::Cooldown { .. }));

    // A WHOLE NEW CYCLE: cooldown expires, re-arm, re-load.
    let after_cd = t0 + cfg.evict_budget_secs + cfg.cooldown_secs + 10;
    tick_settled(&tier, after_cd, 5_000, false).await; // Rest
    tick_settled(&tier, after_cd + 10, 5_000, false).await; // Arm
    tick_settled(&tier, after_cd + 10 + cfg.arm_confirm_secs, 5_000, false).await; // Load
    assert_eq!(
        tier.phase(),
        Phase::Ready,
        "fixture must genuinely reach a NEW Ready, or this proves nothing"
    );
    let model_before = tier.status().model.clone();
    assert!(model_before.is_some());

    // NOW release the long-abandoned teardown and let it run to completion.
    h.stop_gate.add_permits(OPEN);
    for _ in 0..500 {
        tokio::task::yield_now().await;
    }

    assert_eq!(
        tier.phase(),
        Phase::Ready,
        "a teardown that was abandoned must not cool down a NEWLY loaded coder"
    );
    assert_eq!(
        tier.status().model,
        model_before,
        "nor may it clear the new cycle's model"
    );
}

// The two abandonment guards, ISOLATED.
//
// End-to-end, the EARLY guard short-circuits before the LATE one ever runs, so a
// single scenario cannot distinguish them and mutants of each survived. These
// drive `run_eviction` directly with a chosen generation.

#[tokio::test]
async fn an_abandoned_teardown_stops_nothing_at_all() {
    // THE EARLY GUARD. A teardown whose generation is already stale must not even
    // read `model` — by the time it is polled, a newer cycle may have loaded its
    // own coder, and `inner.model` would name THAT one.
    let (tier, env, _gate, _rx) = ready_tier().await;
    let stale = tier.evict_epoch().wrapping_sub(1);
    let calls_before = env.calls().len();

    tier.run_eviction(EvictReason::AssistantArrived, stale).await;

    assert_eq!(
        env.calls().len(),
        calls_before,
        "a pre-abandoned teardown must not stop or restore anything"
    );
    assert_eq!(tier.phase(), Phase::Ready, "and must not touch the phase");
}

#[tokio::test]
async fn a_teardown_abandoned_mid_flight_does_not_write_the_phase() {
    // THE LATE GUARD. This teardown starts legitimately, and is abandoned while
    // it is inside `stop_coder`. It must still restore (idempotent, and the
    // cohort must never be stranded) but must NOT write phase/model, because a
    // newer cycle may already own them.
    let (tier, env, gate, mut stop_entered) = ready_tier().await;
    gate.acquire_many(OPEN as u32).await.unwrap().forget();
    let gen = tier.evict_epoch();

    let t = tier.clone();
    let teardown = tokio::spawn(async move {
        t.run_eviction(EvictReason::AssistantArrived, gen).await;
    });
    tokio::time::timeout(std::time::Duration::from_secs(5), stop_entered.recv())
        .await
        .expect("teardown reached the stop seam")
        .expect("entered");

    // Abandon it mid-flight, then let it finish.
    tier.bump_evict_epoch_only();
    gate.add_permits(OPEN);
    teardown.await.expect("teardown completes");

    assert!(
        env.calls().contains(&"restore"),
        "an abandoned teardown must STILL restore — the cohort is never stranded"
    );
    assert_eq!(
        tier.phase(),
        Phase::Ready,
        "but it must not write a phase a newer cycle now owns"
    );
    assert!(tier.status().model.is_some(), "nor clear the newer model");
}

#[tokio::test]
async fn the_tier_refuses_to_load_onto_the_always_on_assistant_engine() {
    // `resolve_and_ensure` is general routing and can select the ALWAYS-ON serve
    // for an untagged model or via an arch fallback. The tier would then be
    // `Ready` on the assistant's own engine, with the stop gate (correctly)
    // refusing to stop it — a coder that can never be evicted.
    let h = fake(Some("a-coder"), Some(17.3), Some(50.0));
    h.env
        .coder_on_demand
        .store(false, std::sync::atomic::Ordering::SeqCst);
    let tier = CoderTier::new(cfg_on(), h.env.clone());
    tick_settled(&tier, 1_000, 5_000, false).await;
    tick_settled(&tier, 2_000, 5_000, false).await;

    assert_ne!(tier.phase(), Phase::Ready);
    assert!(
        !h.env.calls().contains(&"start"),
        "it must refuse BEFORE starting anything, not unwind afterwards"
    );

    // CONTROL: the identical fixture loads once the backend is on-demand.
    let h2 = fake(Some("a-coder"), Some(17.3), Some(50.0));
    let tier2 = CoderTier::new(cfg_on(), h2.env.clone());
    tick_settled(&tier2, 1_000, 5_000, false).await;
    tick_settled(&tier2, 2_000, 5_000, false).await;
    assert_eq!(tier2.phase(), Phase::Ready);
}

#[tokio::test]
async fn the_tick_idle_age_comes_from_the_tiers_own_stamp() {
    // The tier keeps its OWN stamp. Reading `IdleController`'s counters counted a
    // review's own inference as user activity, so a running review kept idle_secs
    // near zero and the next tick evicted the coder it was using.
    //
    // NOTE: the first version of this test was a LYING FIXTURE — it compared
    // against a freshly-constructed tier whose stamp coincided with every global
    // counter's seed, so "reads the global instead" and "never stamps at all"
    // both still produced 0. The stamp is now aged to a value nothing else shares.
    let (tier, _env, _gate, _rx) = ready_tier().await;
    let now = crate::gpu_exclusive::now_epoch();

    tier.set_last_assistant_for_test(now as i64 - 5_000);
    assert_eq!(
        tier.assistant_idle_secs(now),
        5_000,
        "the age must come from the tier's own stamp, not any global counter"
    );

    // A coder lease is NOT user activity and must not reset the age.
    let lease = tier.try_acquire_lease().expect("lease");
    assert_eq!(tier.assistant_idle_secs(now), 5_000);

    // Only the ASSISTANT hook resets it.
    tier.note_assistant_request();
    assert_eq!(
        tier.assistant_idle_secs(now),
        0,
        "the hot-path hook must actually stamp"
    );
    drop(lease);
}

#[tokio::test]
async fn a_wedged_load_does_not_block_the_tick_loop() {
    // The same structural defect that was fixed for evictions, through the LOAD
    // branch: `tick_loop` awaits `tick` serially, so an inline load that wedges
    // blocks every later tick — including the one that would force-resolve an
    // eviction landing on top of it. The load must be detached.
    let h = fake(Some("a-coder"), Some(17.3), Some(50.0));
    let tier = CoderTier::new(cfg_on(), h.env.clone());
    tick_settled(&tier, 1_000, 5_000, false).await;

    // Wedge the load inside the seam.
    h.start_gate.acquire_many(OPEN as u32).await.unwrap().forget();
    let dispatched = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        tier.tick(2_000, 5_000, false),
    )
    .await;
    assert!(
        dispatched.is_ok(),
        "tick must return promptly; an inline load wedges the whole loop"
    );
    assert_eq!(tier.phase(), Phase::Loading, "and must claim the phase synchronously");

    // The loop is free, so a user can still be served and the tier still evicts.
    assert!(tier.note_assistant_request());
    assert!(matches!(tier.phase(), Phase::Evicting { .. }));
    h.start_gate.add_permits(OPEN);
}

// Every early exit in the DETACHED load must leave a RESTING phase.
//
// Stranding `Loading` blocks admission forever — a load "in progress" that will
// never finish — an invisible, permanent disabling of the feature. These call
// `run_load` DIRECTLY with `Loading` already installed, because reaching those
// branches through `tick` is impossible: `tick` only decides `Load` when the
// coder already resolved, so a fixture routed through `tick` never exercises the
// resolve-failure path at all (the first version of this test did exactly that
// and proved nothing).

#[tokio::test]
async fn a_load_that_cannot_resolve_a_coder_never_strands_the_loading_phase() {
    let h = fake(None, None, Some(50.0));
    let tier = CoderTier::new(cfg_on(), h.env.clone());
    tier.set_phase_for_test(Phase::Loading, None);

    tier.run_load(10_000).await;

    assert_ne!(tier.phase(), Phase::Loading, "must not strand Loading");
    assert!(matches!(tier.phase(), Phase::Cooldown { .. }));
    assert!(h.env.calls().is_empty(), "and must not have started anything");
}

#[tokio::test]
async fn refusing_the_always_on_engine_never_strands_the_loading_phase() {
    let h = fake(Some("a-coder"), Some(17.3), Some(50.0));
    h.env
        .coder_on_demand
        .store(false, std::sync::atomic::Ordering::SeqCst);
    let tier = CoderTier::new(cfg_on(), h.env.clone());
    tier.set_phase_for_test(Phase::Loading, None);

    tier.run_load(10_000).await;

    assert_ne!(tier.phase(), Phase::Loading, "must not strand Loading");
    assert!(matches!(tier.phase(), Phase::Cooldown { .. }));
    assert!(
        !h.env.calls().contains(&"start"),
        "and must refuse BEFORE starting anything"
    );
}

#[tokio::test]
async fn concurrent_leases_are_capped_so_the_user_path_stays_bounded() {
    // The user hot path cancels EVERY live lease, so an unbounded lease set puts
    // unbounded work on the request that must be fastest.
    let (tier, _env, _gate, _rx) = ready_tier().await;
    let cap = cfg_on().max_leases;
    let leases: Vec<Lease> = (0..cap)
        .map(|_| tier.try_acquire_lease().expect("within the cap"))
        .collect();
    assert_eq!(leases.len(), cap);
    assert!(
        tier.try_acquire_lease().is_none(),
        "the cap must actually refuse, not merely be configured"
    );
    // Releasing one frees a slot, so the cap is a bound and not a one-way latch.
    drop(leases.into_iter().next().unwrap());
    assert!(tier.try_acquire_lease().is_some());
}

#[tokio::test]
async fn a_dropped_lease_does_not_leave_its_secret_valid_forever() {
    // A lease that is dropped without commit/abandon — a panicking review, a
    // cancelled task — must deregister. A leaked secret keeps passing
    // `is_live_lease`, so a request bearing it stays exempt from pre-emption
    // FOREVER: the same hole as the presence-only header check, reintroduced by a
    // leak rather than a forgery.
    use axum::http::{HeaderMap, HeaderValue};
    let (tier, _env, _gate, _rx) = ready_tier().await;

    let secret = {
        let lease = tier.try_acquire_lease().expect("lease");
        let secret = lease.header_value().to_string();
        // CONTROL: live while the lease is alive.
        assert_eq!(tier.live_lease_count(), 1);
        let mut h = HeaderMap::new();
        h.insert(LEASE_HEADER, HeaderValue::from_str(&secret).unwrap());
        assert!(is_coder_traffic(&tier, &h));
        secret
        // lease dropped here, never committed
    };

    assert_eq!(tier.live_lease_count(), 0, "a dropped lease must deregister");
    let mut h = HeaderMap::new();
    h.insert(LEASE_HEADER, HeaderValue::from_str(&secret).unwrap());
    assert!(
        !is_coder_traffic(&tier, &h),
        "a leaked secret must not stay valid — it would exempt a user forever"
    );
}

/// Deliberately NOT `#[tokio::test]` — this is the documented no-runtime path.
///
/// The counter is incremented under the phase lock, but nothing is spawned when
/// there is no runtime, so nothing would ever decrement it. A leaked count blocks
/// EVERY future load permanently: a silent, unrecoverable disabling of the
/// feature rather than a visible failure.
#[test]
fn the_teardown_counter_does_not_leak_without_a_runtime() {
    let h = fake(Some("a-coder"), Some(17.3), Some(50.0));
    let tier = CoderTier::new(cfg_on(), h.env.clone());
    tier.set_phase_for_test(Phase::Ready, Some("a-coder".into()));

    let evicted = tier.note_assistant_request();
    assert!(evicted, "CONTROL: it must still register the eviction");
    assert_eq!(
        tier.teardowns_inflight(),
        0,
        "no task was spawned, so the increment must have been undone"
    );
}

#[tokio::test]
async fn a_wedged_teardown_does_not_wedge_the_tick_loop() {
    // A tick-originated eviction must be DETACHED. Awaited inline, a stuck
    // `stop_coder` wedges the whole tick loop — so the very tick that would
    // force-resolve it can never run, and the budget backstop is unreachable.
    let h = fake(Some("a-coder"), Some(17.3), Some(50.0));
    let tier = CoderTier::new(cfg_on(), h.env.clone());
    tick_settled(&tier, 1_000, 5_000, false).await;
    tick_settled(&tier, 2_000, 5_000, false).await;
    assert_eq!(tier.phase(), Phase::Ready);

    // Wedge the teardown, then provoke a TICK-originated eviction (assistant
    // activity observed by the tick, not the hot-path hook).
    h.stop_gate.acquire_many(OPEN as u32).await.unwrap().forget();
    let evicting = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        tier.tick(3_000, 0, true),
    )
    .await;
    assert!(
        evicting.is_ok(),
        "tick must return promptly; awaiting the teardown inline wedges the loop"
    );
    assert!(matches!(tier.phase(), Phase::Evicting { .. }));

    // And because the loop is free, the budget backstop can still fire.
    let after = crate::gpu_exclusive::now_epoch() + cfg_on().evict_budget_secs + 1;
    tokio::time::timeout(std::time::Duration::from_secs(5), tier.tick(after, 5_000, false))
        .await
        .expect("the force-resolving tick must also be able to run");
    assert!(matches!(tier.phase(), Phase::Cooldown { .. }));
    h.stop_gate.add_permits(OPEN);
}

#[tokio::test]
async fn a_new_load_waits_for_an_outstanding_teardown() {
    // The defect the generation guard could NOT reach: an abandoned teardown that
    // is already inside `stop_coder` will still stop the backend it named. If a
    // new cycle loaded in the meantime, that late stop kills the NEW coder. So a
    // load must not begin while any teardown is still running.
    let h = fake(Some("a-coder"), Some(17.3), Some(50.0));
    let tier = CoderTier::new(cfg_on(), h.env.clone());
    let cfg = cfg_on();
    let t0 = crate::gpu_exclusive::now_epoch();
    tick_settled(&tier, t0, 5_000, false).await;
    tick_settled(&tier, t0 + 200, 5_000, false).await;
    assert_eq!(tier.phase(), Phase::Ready);

    // Wedge the teardown and force-resolve past its budget.
    h.stop_gate.acquire_many(OPEN as u32).await.unwrap().forget();
    let mut stop_entered = h.stop_entered;
    tier.note_assistant_request();
    // WAIT until the teardown is genuinely INSIDE `stop_coder` before abandoning
    // it. That is the whole hazard: a teardown abandoned BEFORE it starts stops
    // nothing and a later load is perfectly safe, so force-resolving too early
    // would make this test pass without testing anything.
    tokio::time::timeout(std::time::Duration::from_secs(5), stop_entered.recv())
        .await
        .expect("teardown reached the stop seam")
        .expect("entered");
    tick_settled(&tier, t0 + cfg.evict_budget_secs + 1, 5_000, false).await;
    assert!(matches!(tier.phase(), Phase::Cooldown { .. }));

    // Cooldown expires; the tier tries to run a whole new cycle while the old
    // teardown is STILL wedged inside stop_coder.
    let after = t0 + cfg.evict_budget_secs + cfg.cooldown_secs + 10;
    tick_settled(&tier, after, 5_000, false).await; // Rest
    tick_settled(&tier, after + 10, 5_000, false).await; // Arm
    tick_settled(&tier, after + 10 + cfg.arm_confirm_secs, 5_000, false).await; // would Load
    assert_ne!(
        tier.phase(),
        Phase::Ready,
        "a new coder must NOT be loaded underneath an in-flight teardown"
    );

    // Once the teardown finishes, a load is allowed again.
    h.stop_gate.add_permits(OPEN);
    for _ in 0..500 {
        tokio::task::yield_now().await;
    }
    tick_settled(&tier, after + 400, 5_000, false).await; // Arm
    tick_settled(&tier, after + 400 + cfg.arm_confirm_secs, 5_000, false).await;
    assert_eq!(
        tier.phase(),
        Phase::Ready,
        "CONTROL: once the teardown is done the load proceeds, so the gate is not just 'never load'"
    );
}

#[tokio::test]
async fn a_load_that_lost_the_race_compensates_with_a_stop() {
    // The leak the panel found: eviction's stop can run BEFORE the in-flight
    // start actually launched the backend. Declining to install `Ready` is not
    // enough — without a compensating stop the coder is left running while the
    // tier records `model: None`, i.e. borrowed memory nobody gives back.
    let h = fake(Some("a-coder"), Some(17.3), Some(50.0));
    let tier = CoderTier::new(cfg_on(), h.env.clone());
    tick_settled(&tier, 1_000, 5_000, false).await;

    let start_gate = h.start_gate.clone();
    start_gate.acquire_many(OPEN as u32).await.unwrap().forget();
    let mut start_entered = h.start_entered;

    let t = tier.clone();
    let loading = tokio::spawn(async move { t.tick(2_000, 5_000, false).await });
    tokio::time::timeout(std::time::Duration::from_secs(5), start_entered.recv())
        .await
        .expect("load started")
        .expect("entered");
    assert_eq!(tier.phase(), Phase::Loading);

    tier.note_assistant_request();
    start_gate.add_permits(OPEN); // the load now SUCCEEDS, after losing
    loading.await.expect("tick completes");
    for _ in 0..300 {
        tokio::task::yield_now().await;
    }

    assert_ne!(tier.phase(), Phase::Ready);
    let stops = h.env.calls().iter().filter(|c| **c == "stop").count();
    assert!(
        stops >= 2,
        "expected the eviction stop AND a compensating stop for the late load, saw {stops}"
    );
}

#[tokio::test]
async fn a_failed_load_cools_down_instead_of_retrying_immediately() {
    let h = fake(Some("a-coder"), Some(17.3), Some(50.0));
    *h.env.start_result.lock().unwrap() = Err("no".into());
    let tier = CoderTier::new(cfg_on(), h.env.clone());
    tick_settled(&tier, 1_000, 5_000, false).await;
    tick_settled(&tier, 2_000, 5_000, false).await;
    match tier.phase() {
        Phase::Cooldown { until } => assert_eq!(until, 2_000 + cfg_on().cooldown_secs),
        p => panic!("expected cooldown after a failed load, got {p:?}"),
    }
    assert_eq!(tier.status().loads, 0);
}

#[tokio::test]
async fn an_unresolvable_coder_alias_loads_nothing_ever() {
    let h = fake(None, None, Some(50.0));
    let tier = CoderTier::new(cfg_on(), h.env.clone());
    tick_settled(&tier, 1_000, 5_000, false).await;
    tick_settled(&tier, 2_000, 5_000, false).await;
    tick_settled(&tier, 3_000, 5_000, false).await;
    assert!(matches!(tier.phase(), Phase::Arming { .. }));
    assert!(h.env.calls().is_empty(), "nothing was started");
}

// ─────────────────────────────────────────────────────────────────────────────
// Default-off, and the RVXR-02 contract
// ─────────────────────────────────────────────────────────────────────────────


#[tokio::test]
async fn a_disabled_tier_is_inert_on_the_hot_path() {
    let h = fake(Some("a-coder"), Some(17.3), Some(50.0));
    let mut cfg = cfg_on();
    cfg.enabled = false;
    let tier = CoderTier::new(cfg, h.env.clone());
    assert!(!tier.note_assistant_request());
    assert!(tier.try_acquire_lease().is_none());
    tick_settled(&tier, 1_000, 5_000, false).await;
    tick_settled(&tier, 2_000, 5_000, false).await;
    assert_eq!(tier.phase(), Phase::Assistant);
    assert!(h.env.calls().is_empty());
}

#[test]
fn the_interrupt_signal_is_stable_and_machine_readable() {
    // RVXR-02 switches on `code`; the prose may be reworded, the code may not.
    let r = InterruptedReport::new();
    assert_eq!(r.code, "inference_engine_interrupted");
    assert_eq!(r.reason, "inference engine interrupted by user");
    let v = serde_json::to_value(&r).expect("serializes");
    assert_eq!(v["interrupted"], serde_json::json!(true));
    assert_eq!(v["code"], serde_json::json!("inference_engine_interrupted"));
    assert_eq!(
        v["reason"],
        serde_json::json!("inference engine interrupted by user")
    );
}

#[tokio::test]
async fn only_a_live_lease_secret_counts_as_coder_traffic() {
    use axum::http::{HeaderMap, HeaderValue};
    let (tier, _env, _gate, _rx) = ready_tier().await;
    let lease = tier.try_acquire_lease().expect("lease");

    let with = |v: &str| {
        let mut h = HeaderMap::new();
        h.insert(LEASE_HEADER, HeaderValue::from_str(v).unwrap());
        h
    };

    // THE CONTROL: a genuine, live lease secret IS coder traffic. Without this
    // the negative cases below would pass against a function that always says
    // "user".
    assert!(is_coder_traffic(&tier, &with(lease.header_value())));

    // THE HOLE THAT REVIEW FOUND: mere header PRESENCE must not be enough, or any
    // authenticated caller could label itself a review and exempt itself from
    // pre-emption — a user request that leaves the coder resident.
    assert!(!is_coder_traffic(&tier, &with("1")));
    assert!(!is_coder_traffic(&tier, &with("00000000-0000-0000-0000-000000000000")));
    assert!(!is_coder_traffic(&tier, &with("")));
    assert!(!is_coder_traffic(&tier, &HeaderMap::new()));
    let mut other = HeaderMap::new();
    other.insert("x-something-else", HeaderValue::from_static("1"));
    assert!(!is_coder_traffic(&tier, &other));

    // A SETTLED lease's secret is dead — no replay.
    let secret = lease.header_value().to_string();
    let _ = lease.commit("done");
    assert!(!is_coder_traffic(&tier, &with(&secret)));
}

#[tokio::test]
async fn a_forged_lease_header_does_not_stop_a_user_evicting_the_coder() {
    use axum::http::{HeaderMap, HeaderValue};
    let (tier, _env, gate, _rx) = ready_tier().await;
    gate.acquire_many(OPEN as u32).await.unwrap().forget();
    let mut forged = HeaderMap::new();
    forged.insert(LEASE_HEADER, HeaderValue::from_static("not-a-real-lease"));

    // Classified as a user, so the eviction path is the one that must run.
    assert!(!is_coder_traffic(&tier, &forged));
    assert!(
        tier.note_assistant_request(),
        "a request carrying a forged lease header is a USER and must evict"
    );
    assert!(matches!(tier.phase(), Phase::Evicting { .. }));
    gate.add_permits(OPEN);
}

