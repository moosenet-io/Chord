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
    free_gb: Mutex<Option<f64>>,
    start_result: Mutex<Result<(), String>>,
    stop_result: Mutex<Result<(), String>>,
    /// Ordered log of seam calls — the evidence for "restore was called anyway".
    calls: Mutex<Vec<&'static str>>,
    start_entered: mpsc::UnboundedSender<()>,
    start_gate: Arc<Semaphore>,
    stop_entered: mpsc::UnboundedSender<()>,
    stop_gate: Arc<Semaphore>,
    restores: AtomicUsize,
}

struct FakeHandles {
    env: Arc<FakeEnv>,
    start_entered: mpsc::UnboundedReceiver<()>,
    start_gate: Arc<Semaphore>,
    stop_entered: mpsc::UnboundedReceiver<()>,
    stop_gate: Arc<Semaphore>,
}

fn fake(coder: Option<&str>, size_gb: Option<f64>, free_gb: Option<f64>) -> FakeHandles {
    let (start_tx, start_rx) = mpsc::unbounded_channel();
    let (stop_tx, stop_rx) = mpsc::unbounded_channel();
    let start_gate = Arc::new(Semaphore::new(OPEN));
    let stop_gate = Arc::new(Semaphore::new(OPEN));
    let env = Arc::new(FakeEnv {
        coder: coder.map(|s| s.to_string()),
        size_gb,
        free_gb: Mutex::new(free_gb),
        start_result: Mutex::new(Ok(())),
        stop_result: Mutex::new(Ok(())),
        calls: Mutex::new(Vec::new()),
        start_entered: start_tx,
        start_gate: start_gate.clone(),
        stop_entered: stop_tx,
        stop_gate: stop_gate.clone(),
        restores: AtomicUsize::new(0),
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
    fn free_gb(&self) -> Option<f64> {
        *self.free_gb.lock().unwrap()
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
        min_idle_secs: 900,
        arm_confirm_secs: 120,
        cooldown_secs: 600,
        headroom_margin_gb: 8.0,
        footprint_factor: 1.8,
        evict_budget_secs: 60,
    }
}

/// Quiet, well-provisioned observation at `now`.
fn quiet(now: u64) -> Observation {
    Observation {
        now,
        assistant_idle_secs: 5_000,
        assistant_inflight: false,
        free_gb: Some(80.0),
        coder_gb: Some(17.3),
        coder_resolved: true,
    }
}

/// Drive the tier to `Ready` deterministically, returning the tier + handles.
async fn ready_tier() -> (Arc<CoderTier>, Arc<FakeEnv>, Arc<Semaphore>, mpsc::UnboundedReceiver<()>) {
    let h = fake(Some("a-coder"), Some(17.3), Some(80.0));
    let tier = CoderTier::new(cfg_on(), h.env.clone());
    tier.tick(1_000, 5_000, false).await; // arms only
    assert!(matches!(tier.phase(), Phase::Arming { .. }));
    tier.tick(2_000, 5_000, false).await; // confirm elapsed → load
    assert_eq!(tier.phase(), Phase::Ready);
    (tier, h.env, h.stop_gate, h.stop_entered)
}

// ─────────────────────────────────────────────────────────────────────────────
// Hazard 3 — thrash: the pure state machine
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn never_loads_on_the_first_idle_tick() {
    let cfg = cfg_on();
    // Perfectly idle, coder resolved, ample memory — everything a load could want.
    let obs = quiet(1_000);
    assert!(coder_fits(&obs, &cfg), "fixture must actually fit, else this test proves nothing");
    // ...and the very first qualifying observation still only ARMS.
    assert_eq!(decide(&cfg, Phase::Assistant, &obs), Decision::Arm);
}

#[test]
fn load_requires_both_the_idle_dwell_and_the_arm_confirm() {
    let cfg = cfg_on();
    // Armed at t=1000. Before arm_confirm elapses: hold, never load.
    let armed = Phase::Arming { armed_at: 1_000 };
    assert_eq!(decide(&cfg, armed, &quiet(1_000)), Decision::Hold);
    assert_eq!(decide(&cfg, armed, &quiet(1_119)), Decision::Hold);
    // Exactly at the boundary: load.
    assert_eq!(decide(&cfg, armed, &quiet(1_120)), Decision::Load);
}

#[test]
fn short_idle_never_arms_however_long_we_wait() {
    let cfg = cfg_on();
    let mut obs = quiet(9_999);
    obs.assistant_idle_secs = cfg.min_idle_secs - 1;
    assert_eq!(decide(&cfg, Phase::Assistant, &obs), Decision::Hold);
    // And an arming tier disarms rather than riding out its confirm window.
    assert_eq!(decide(&cfg, Phase::Arming { armed_at: 1 }, &obs), Decision::Rest);
}

#[test]
fn inflight_assistant_beats_a_large_idle_counter() {
    let cfg = cfg_on();
    let mut obs = quiet(9_999);
    obs.assistant_inflight = true; // idle_secs is still 5000
    assert_eq!(decide(&cfg, Phase::Assistant, &obs), Decision::Hold);
    assert_eq!(
        decide(&cfg, Phase::Ready, &obs),
        Decision::Evict(EvictReason::ActivityObserved)
    );
    assert_eq!(
        decide(&cfg, Phase::Loading, &obs),
        Decision::Evict(EvictReason::ActivityObserved)
    );
    assert_eq!(decide(&cfg, Phase::Arming { armed_at: 1 }, &obs), Decision::Rest);
}

#[test]
fn cooldown_blocks_a_reload_however_idle_the_box_is() {
    let cfg = cfg_on();
    let cd = Phase::Cooldown { until: 5_000 };
    // Fully idle, perfect fit — and still nothing happens.
    assert_eq!(decide(&cfg, cd, &quiet(4_999)), Decision::Hold);
    // When it expires we go to REST, not straight to Load — leaving cooldown
    // costs one more observation, which is one more chance for a user to arrive.
    assert_eq!(decide(&cfg, cd, &quiet(5_000)), Decision::Rest);
}

#[test]
fn evicting_is_bounded_and_cannot_wedge() {
    let cfg = cfg_on();
    let ev = Phase::Evicting { since: 1_000 };
    assert_eq!(decide(&cfg, ev, &quiet(1_059)), Decision::Hold);
    assert_eq!(decide(&cfg, ev, &quiet(1_060)), Decision::ForceResolveEvict);
    // Even a busy assistant does not second-guess an in-flight eviction — it is
    // already doing the thing the assistant wants.
    let mut busy = quiet(1_010);
    busy.assistant_inflight = true;
    assert_eq!(decide(&cfg, ev, &busy), Decision::Hold);
}

// ─────────────────────────────────────────────────────────────────────────────
// Fit: fail-closed, and the measured runtime-inflation factor
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn fit_is_fail_closed_on_every_unknown() {
    let cfg = cfg_on();
    let base = quiet(1_000);

    let mut unresolved = base;
    unresolved.coder_resolved = false;
    assert!(!coder_fits(&unresolved, &cfg));

    let mut no_free = base;
    no_free.free_gb = None;
    assert!(!coder_fits(&no_free, &cfg));

    let mut no_size = base;
    no_size.coder_gb = None;
    assert!(!coder_fits(&no_size, &cfg));

    let mut nan = base;
    nan.free_gb = Some(f64::NAN);
    assert!(!coder_fits(&nan, &cfg));

    // And an armed tier that cannot prove a fit HOLDS — it never loads blind.
    assert_eq!(
        decide(&cfg, Phase::Arming { armed_at: 0 }, &no_free),
        Decision::Hold
    );
}

#[test]
fn fit_sizes_against_runtime_footprint_not_on_disk_size() {
    let cfg = cfg_on(); // factor 1.8, margin 8.0
    // Measured on the live host: the coder is ~17.3 GiB on disk. Its RUNTIME
    // footprint is materially larger (granite4.1:8b: 4.98 GiB disk → 8.93 GiB
    // resident, ×1.79). 17.3 × 1.8 = 31.14, + 8 margin = 39.14.
    assert!((runtime_footprint_gb(17.3, 1.8) - 31.14).abs() < 1e-9);

    let mut obs = quiet(1_000);
    obs.coder_gb = Some(17.3);

    obs.free_gb = Some(40.0);
    assert!(coder_fits(&obs, &cfg), "39.14 needed, 40 free");

    obs.free_gb = Some(39.0);
    assert!(!coder_fits(&obs, &cfg), "39.14 needed, only 39 free");

    // The naive on-disk sizing would have said yes at 26 GB free. It must not.
    obs.free_gb = Some(26.0);
    assert!(
        !coder_fits(&obs, &cfg),
        "sizing against the raw on-disk figure under-counts by most of a coder model"
    );
}

#[test]
fn footprint_factor_is_clamped_so_it_can_only_over_estimate() {
    // A factor below 1 would UNDER-estimate — the one unsafe direction.
    assert_eq!(runtime_footprint_gb(10.0, 0.1), 10.0);
    assert_eq!(runtime_footprint_gb(10.0, 1.0), 10.0);
    assert!((runtime_footprint_gb(10.0, 2.0) - 20.0).abs() < 1e-9);
}

// ─────────────────────────────────────────────────────────────────────────────
// Headroom does not persist
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn memory_pressure_evicts_a_resident_coder() {
    let cfg = cfg_on();
    let mut obs = quiet(1_000);
    // Something that is not an inference request took the window (a build cache
    // ate 64 GB of this host once).
    obs.free_gb = Some(3.0);
    assert!(under_memory_pressure(&obs, &cfg));
    assert_eq!(
        decide(&cfg, Phase::Ready, &obs),
        Decision::Evict(EvictReason::MemoryPressure)
    );
}

#[test]
fn an_unreadable_counter_is_not_evidence_of_pressure() {
    let cfg = cfg_on();
    let mut obs = quiet(1_000);
    obs.free_gb = None;
    assert!(!under_memory_pressure(&obs, &cfg));
    assert_eq!(decide(&cfg, Phase::Ready, &obs), Decision::Hold);
    // NaN is a broken sensor, not a zero.
    obs.free_gb = Some(f64::NAN);
    assert!(!under_memory_pressure(&obs, &cfg));
}

#[test]
fn a_disabled_tier_gives_memory_back_and_then_rests() {
    let mut cfg = cfg_on();
    cfg.enabled = false;
    let obs = quiet(1_000);
    assert_eq!(
        decide(&cfg, Phase::Ready, &obs),
        Decision::Evict(EvictReason::Disabled)
    );
    assert_eq!(
        decide(&cfg, Phase::Loading, &obs),
        Decision::Evict(EvictReason::Disabled)
    );
    assert_eq!(decide(&cfg, Phase::Assistant, &obs), Decision::Hold);
    assert_eq!(decide(&cfg, Phase::Cooldown { until: 9_999 }, &obs), Decision::Rest);
    // A disabled tier still BOUNDS an eviction it is already running.
    assert_eq!(
        decide(&cfg, Phase::Evicting { since: 1_000 }, &quiet(1_060)),
        Decision::ForceResolveEvict
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Hazard 1 — the eviction race, FORCED
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn assistant_request_returns_without_waiting_for_teardown() {
    let h = fake(Some("a-coder"), Some(17.3), Some(80.0));
    let tier = CoderTier::new(cfg_on(), h.env.clone());
    tier.tick(1_000, 5_000, false).await;
    tier.tick(2_000, 5_000, false).await;
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
    let h = fake(Some("a-coder"), Some(17.3), Some(80.0));
    let tier = CoderTier::new(cfg_on(), h.env.clone());
    assert!(tier.try_acquire_lease().is_none(), "assistant phase");
    tier.tick(1_000, 5_000, false).await;
    assert!(tier.try_acquire_lease().is_none(), "arming phase");
    tier.tick(2_000, 5_000, false).await;
    assert!(tier.try_acquire_lease().is_some(), "ready phase");
    tier.note_assistant_request();
    assert!(tier.try_acquire_lease().is_none(), "evicting phase");
}

#[tokio::test]
async fn a_user_arriving_mid_load_never_lets_the_coder_reach_ready() {
    // The other half of the race: the load is in flight when the user arrives.
    // If `tick` blindly wrote `Ready` on a successful start, the coder would be
    // resident with nobody left to evict it.
    let h = fake(Some("a-coder"), Some(17.3), Some(80.0));
    let tier = CoderTier::new(cfg_on(), h.env.clone());
    tier.tick(1_000, 5_000, false).await;
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

#[tokio::test]
async fn a_failed_load_cools_down_instead_of_retrying_immediately() {
    let h = fake(Some("a-coder"), Some(17.3), Some(80.0));
    *h.env.start_result.lock().unwrap() = Err("no".into());
    let tier = CoderTier::new(cfg_on(), h.env.clone());
    tier.tick(1_000, 5_000, false).await;
    tier.tick(2_000, 5_000, false).await;
    match tier.phase() {
        Phase::Cooldown { until } => assert_eq!(until, 2_000 + cfg_on().cooldown_secs),
        p => panic!("expected cooldown after a failed load, got {p:?}"),
    }
    assert_eq!(tier.status().loads, 0);
}

#[tokio::test]
async fn an_unresolvable_coder_alias_loads_nothing_ever() {
    let h = fake(None, None, Some(80.0));
    let tier = CoderTier::new(cfg_on(), h.env.clone());
    tier.tick(1_000, 5_000, false).await;
    tier.tick(2_000, 5_000, false).await;
    tier.tick(3_000, 5_000, false).await;
    assert!(matches!(tier.phase(), Phase::Arming { .. }));
    assert!(h.env.calls().is_empty(), "nothing was started");
}

// ─────────────────────────────────────────────────────────────────────────────
// Default-off, and the RVXR-02 contract
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn the_tier_ships_dark() {
    assert!(
        !CoderTierConfig::default().enabled,
        "this touches a live shared GPU host — it must be opt-in"
    );
}

#[tokio::test]
async fn a_disabled_tier_is_inert_on_the_hot_path() {
    let h = fake(Some("a-coder"), Some(17.3), Some(80.0));
    let mut cfg = cfg_on();
    cfg.enabled = false;
    let tier = CoderTier::new(cfg, h.env.clone());
    assert!(!tier.note_assistant_request());
    assert!(tier.try_acquire_lease().is_none());
    tier.tick(1_000, 5_000, false).await;
    tier.tick(2_000, 5_000, false).await;
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

#[test]
fn request_classification_treats_an_unlabelled_request_as_a_user() {
    use axum::http::{HeaderMap, HeaderValue};
    let mut coder = HeaderMap::new();
    coder.insert(LEASE_HEADER, HeaderValue::from_static("1"));
    assert!(is_coder_traffic(&coder), "a review on a lease is coder traffic");

    // The safe default is in the right direction: an unlabelled request counts as
    // a user, so the worst case is an unnecessary eviction — never a missed one.
    assert!(!is_coder_traffic(&HeaderMap::new()));
    let mut other = HeaderMap::new();
    other.insert("x-something-else", HeaderValue::from_static("1"));
    assert!(!is_coder_traffic(&other));
    // Presence, not value — the header is a classification, not a credential.
    let mut empty_value = HeaderMap::new();
    empty_value.insert(LEASE_HEADER, HeaderValue::from_static(""));
    assert!(is_coder_traffic(&empty_value));
}

#[test]
fn config_defaults_are_read_from_env_with_safe_clamps() {
    let d = CoderTierConfig::default();
    assert!(d.min_idle_secs > 0 && d.arm_confirm_secs > 0 && d.cooldown_secs > 0);
    assert!(d.footprint_factor >= 1.0);
    assert!(d.headroom_margin_gb > 0.0);
}
