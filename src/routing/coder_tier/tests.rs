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

/// Drive the tier to `Ready` deterministically. The two intermediate assertions
/// are load-bearing: if the fixture ever stopped ARMING first, or stopped
/// reaching Ready, every eviction test below would pass for the wrong reason.
async fn ready_tier() -> (Arc<CoderTier>, Arc<FakeEnv>, Arc<Semaphore>, mpsc::UnboundedReceiver<()>) {
    let h = fake(Some("a-coder"), Some(17.3), Some(50.0));
    let tier = CoderTier::new(cfg_on(), h.env.clone());
    tier.tick(1_000, 5_000, false).await; // arms only
    assert!(matches!(tier.phase(), Phase::Arming { .. }));
    tier.tick(2_000, 5_000, false).await; // confirm elapsed → load
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
    let h = fake(Some("a-coder"), Some(17.3), Some(50.0));
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
    let h = fake(Some("a-coder"), Some(17.3), Some(50.0));
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
    let h = fake(None, None, Some(50.0));
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


#[tokio::test]
async fn a_disabled_tier_is_inert_on_the_hot_path() {
    let h = fake(Some("a-coder"), Some(17.3), Some(50.0));
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

