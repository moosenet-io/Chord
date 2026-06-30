//! SRV-05 integration tests: VRAM admission + tier-aware eviction / residency.
//!
//! End-to-end against the real [`VramResidencyManager`] with a mocked launcher
//! (no GPU, no process): admission math, eviction order (transient-first), the
//! pinned chat-role never-evict invariant under sustained pressure, keep-warm
//! contention queue-then-LRU-after-threshold, keep-warm persistence across
//! requests, the unreadable-VRAM fail-safe, the pinned-only stall → CannotAdmit,
//! concurrency (no double-admit), and the atomic JSON state file. These cover the
//! SRV-05 TEST PLAN and the Behavior Spec verify checks.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;

use chord_proxy::serving::residency::{
    EventSink, ResidencyEvent, VramResidencyManager, WarmLauncher,
};
use chord_proxy::serving::{ResidencyError, ResidencyManager};
use terminus_rs::intake::serving::{ModelId, Runtime};

/// A scripted launcher: settable free-VRAM, recorded launch/evict order, a health
/// toggle, and an optional per-evict footprint table so a test can model real
/// residents (the manager itself tracks residents; this only voices the GPU side).
struct MockLauncher {
    free_gb: Mutex<Option<f64>>,
    healthy: Mutex<bool>,
    launches: Mutex<Vec<String>>,
    evictions: Mutex<Vec<String>>,
    /// When set, [`free_vram_gb`] is BUMPED by the evicted footprint as residents
    /// are evicted, so a re-plan after a queue-wait sees the freed VRAM (models a
    /// real host whose free counter rises on eviction).
    footprints: Mutex<std::collections::HashMap<String, f64>>,
}

impl MockLauncher {
    fn new(free: Option<f64>) -> Arc<Self> {
        Arc::new(MockLauncher {
            free_gb: Mutex::new(free),
            healthy: Mutex::new(true),
            launches: Mutex::new(vec![]),
            evictions: Mutex::new(vec![]),
            footprints: Mutex::new(std::collections::HashMap::new()),
        })
    }
    fn with_footprint(self: &Arc<Self>, id: &str, gb: f64) {
        self.footprints.lock().unwrap().insert(id.into(), gb);
    }
}

#[async_trait]
impl WarmLauncher for MockLauncher {
    async fn free_vram_gb(&self) -> Option<f64> {
        *self.free_gb.lock().unwrap()
    }
    async fn launch(&self, model_id: &str, gb: f64) -> Result<(Runtime, String), String> {
        self.launches.lock().unwrap().push(model_id.to_string());
        // A real host's free counter drops by the model's footprint on launch, so a
        // subsequent admission sees the consumption (the no-double-admit invariant
        // depends on this — free does NOT stay constant across launches).
        let mut f = self.free_gb.lock().unwrap();
        *f = Some((f.unwrap_or(0.0) - gb).max(0.0));
        Ok((Runtime::LlamaCpp, format!("http://warm.invalid/{model_id}")))
    }
    async fn health_check(&self, _endpoint: &str) -> bool {
        *self.healthy.lock().unwrap()
    }
    async fn evict(&self, model_id: &str) -> Result<(), String> {
        self.evictions.lock().unwrap().push(model_id.to_string());
        // Free counter rises by the evicted footprint, if known.
        if let Some(gb) = self.footprints.lock().unwrap().get(model_id).copied() {
            let mut f = self.free_gb.lock().unwrap();
            *f = Some(f.unwrap_or(0.0) + gb);
        }
        Ok(())
    }
}

#[derive(Default)]
struct RecordingSink {
    events: Mutex<Vec<ResidencyEvent>>,
}
impl EventSink for RecordingSink {
    fn emit(&self, event: &ResidencyEvent) {
        self.events.lock().unwrap().push(event.clone());
    }
}
impl RecordingSink {
    fn decisions(&self) -> Vec<String> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .map(|e| e.decision.to_string())
            .collect()
    }
}

fn manager(
    launcher: Arc<MockLauncher>,
    sink: Arc<RecordingSink>,
    wait_ms: u64,
    state_path: Option<String>,
) -> VramResidencyManager {
    VramResidencyManager::with_settings(
        launcher,
        sink,
        Duration::from_millis(wait_ms),
        state_path,
    )
}

// ── admission math ────────────────────────────────────────────────────────────

#[tokio::test]
async fn admits_model_that_fits() {
    let l = MockLauncher::new(Some(48.0));
    let s = Arc::new(RecordingSink::default());
    let m = manager(l.clone(), s.clone(), 10, None);
    let slot = m
        .acquire_warm_slot(&ModelId::from("qwen3:8b"), Some(20.0))
        .await
        .unwrap();
    assert_eq!(slot.model_id, "qwen3:8b");
    assert_eq!(*l.launches.lock().unwrap(), vec!["qwen3:8b".to_string()]);
    assert!(s.decisions().contains(&"admit".to_string()));
}

#[tokio::test]
async fn rejects_when_nothing_evictable_and_too_big() {
    // Empty host, free 10, model needs 40 → nothing to evict → CannotAdmit.
    let l = MockLauncher::new(Some(10.0));
    let s = Arc::new(RecordingSink::default());
    let m = manager(l.clone(), s.clone(), 5, None);
    let err = m
        .acquire_warm_slot(&ModelId::from("big:120b"), Some(40.0))
        .await
        .unwrap_err();
    assert!(matches!(err, ResidencyError::CannotAdmit(_)));
    assert!(l.launches.lock().unwrap().is_empty());
    assert!(s.decisions().contains(&"admission-denied".to_string()));
}

// ── eviction order: transient before keep-warm ────────────────────────────────

#[tokio::test]
async fn evicts_transient_before_keep_warm() {
    use chord_proxy::serving::Tier;
    // free 0; a transient (45) alone makes room for a 40 model → keep-warm stays.
    let l = MockLauncher::new(Some(30.0));
    l.with_footprint("t", 45.0);
    let s = Arc::new(RecordingSink::default());
    let m = manager(l.clone(), s.clone(), 10, None);
    // A keep-warm resident (warm-slot path) and a transient resident (the inline
    // cold-launch path registers itself). Then squeeze free to 0 → admitting `new`
    // (40) needs reclamation; the transient (45) alone suffices → keep-warm stays.
    m.acquire_warm_slot(&ModelId::from("kw"), Some(30.0)).await.unwrap();
    m.register_resident("t", Runtime::LlamaCpp, "http://t.invalid", 45.0, Tier::Transient)
        .await;
    *l.free_gb.lock().unwrap() = Some(0.0);
    l.evictions.lock().unwrap().clear();
    m.acquire_warm_slot(&ModelId::from("new"), Some(40.0)).await.unwrap();
    let ev = l.evictions.lock().unwrap().clone();
    assert_eq!(ev, vec!["t".to_string()], "only the transient is evicted");
    assert!(!ev.contains(&"kw".to_string()), "keep-warm NOT evicted");
}

// ── chat-role never evicted, even under sustained pressure (negative test) ─────

#[tokio::test]
async fn chat_role_never_evicted_under_sustained_pressure() {
    let l = MockLauncher::new(Some(0.0));
    let s = Arc::new(RecordingSink::default());
    let m = manager(l.clone(), s.clone(), 5, None);
    // Make the chat model resident, then pin it.
    *l.free_gb.lock().unwrap() = Some(80.0);
    m.acquire_warm_slot(&ModelId::from("chat"), Some(40.0)).await.unwrap();
    m.set_pinned_chat_model(Some("chat")).await;
    *l.free_gb.lock().unwrap() = Some(0.0);
    l.evictions.lock().unwrap().clear();

    // Hammer with several oversized launches; none may evict the pinned chat model.
    for i in 0..5 {
        let id = format!("pressure-{i}");
        let err = m
            .acquire_warm_slot(&ModelId::from(id.as_str()), Some(60.0))
            .await
            .unwrap_err();
        assert!(matches!(err, ResidencyError::CannotAdmit(_)));
    }
    assert!(
        l.evictions.lock().unwrap().is_empty(),
        "pinned chat model must NEVER be evicted under sustained pressure"
    );
}

// ── keep-warm contention: queue then LRU after threshold ──────────────────────

#[tokio::test]
async fn keep_warm_contention_queues_then_evicts_lru() {
    let l = MockLauncher::new(Some(120.0));
    let s = Arc::new(RecordingSink::default());
    // Admit two keep-warm; the FIRST admitted is the LRU.
    let m = manager(l.clone(), s.clone(), 20, None);
    m.acquire_warm_slot(&ModelId::from("kw_old"), Some(40.0)).await.unwrap();
    m.acquire_warm_slot(&ModelId::from("kw_new"), Some(40.0)).await.unwrap();
    l.with_footprint("kw_old", 40.0);
    l.with_footprint("kw_new", 40.0);
    // Squeeze: free 0, new needs 35 → only keep-warm to take → queue then LRU.
    *l.free_gb.lock().unwrap() = Some(0.0);
    l.evictions.lock().unwrap().clear();
    s.events.lock().unwrap().clear();

    let slot = m
        .acquire_warm_slot(&ModelId::from("incoming"), Some(35.0))
        .await
        .unwrap();
    assert_eq!(slot.model_id, "incoming");
    assert_eq!(
        *l.evictions.lock().unwrap(),
        vec!["kw_old".to_string()],
        "LRU keep-warm evicted after the wait"
    );
    let d = s.decisions();
    assert!(d.contains(&"queue".to_string()), "queued before evicting keep-warm");
    assert!(d.contains(&"evict".to_string()));
    assert!(d.contains(&"admit".to_string()));
}

// ── keep-warm persists across requests (never cold-cycled) ────────────────────

#[tokio::test]
async fn keep_warm_stays_resident_across_requests() {
    let l = MockLauncher::new(Some(60.0));
    let s = Arc::new(RecordingSink::default());
    let m = manager(l.clone(), s.clone(), 10, None);
    for _ in 0..4 {
        m.acquire_warm_slot(&ModelId::from("kw"), Some(40.0)).await.unwrap();
    }
    assert_eq!(
        *l.launches.lock().unwrap(),
        vec!["kw".to_string()],
        "keep-warm launched once, reused thereafter"
    );
}

// ── transient launch never evicts keep-warm if transients suffice ─────────────

#[tokio::test]
async fn transient_alone_suffices_keep_warm_untouched() {
    use chord_proxy::serving::Tier;
    let l = MockLauncher::new(Some(30.0));
    l.with_footprint("t", 50.0);
    let s = Arc::new(RecordingSink::default());
    let m = manager(l.clone(), s.clone(), 10, None);
    m.acquire_warm_slot(&ModelId::from("kw"), Some(30.0)).await.unwrap();
    m.register_resident("t", Runtime::LlamaCpp, "http://t.invalid", 50.0, Tier::Transient)
        .await;
    *l.free_gb.lock().unwrap() = Some(0.0);
    l.evictions.lock().unwrap().clear();
    // need 45, transient (50) alone covers it.
    m.acquire_warm_slot(&ModelId::from("new"), Some(45.0)).await.unwrap();
    let ev = l.evictions.lock().unwrap().clone();
    assert_eq!(ev, vec!["t".to_string()]);
}

// ── concurrency: two simultaneous launches don't double-admit ─────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_launches_respect_vram_ceiling() {
    // free 50; three models each need 30. The host counter drops on launch / rises
    // on evict, so the reservation under the admission lock is what stops two from
    // double-spending the same VRAM. The hard invariant the spec requires: the
    // CO-RESIDENT footprint never exceeds the 50GB ceiling (no double-admit past
    // the ceiling) — regardless of how many ultimately admit serially.
    let l = MockLauncher::new(Some(50.0));
    // Free VRAM tracks launches/evicts (each 30GB model consumes/frees on its own
    // footprint).
    for id in ["a", "b", "c"] {
        l.with_footprint(id, 30.0);
    }
    *l.free_gb.lock().unwrap() = Some(50.0);
    let s = Arc::new(RecordingSink::default());
    let m = manager(l.clone(), s.clone(), 40, None);

    let mut handles = Vec::new();
    for id in ["a", "b", "c"] {
        let mc = m.clone();
        handles.push(tokio::spawn(async move {
            mc.acquire_warm_slot(&ModelId::from(id), Some(30.0)).await
        }));
    }
    for h in handles {
        let _ = h.await.unwrap();
    }
    // The co-resident footprint never exceeds the ceiling — two 30GB models can
    // never be resident together under 50GB (that is the double-admit the lock
    // prevents). Since nothing here is evictable (all fresh keep-warm peers), at
    // most one is resident at the end.
    let reg_total = m.resident_total_gb().await;
    assert!(reg_total <= 50.0, "VRAM ceiling respected: {reg_total} > 50");
}

// ── unreadable VRAM fails safe ─────────────────────────────────────────────────

#[tokio::test]
async fn unreadable_vram_does_not_oom_launch() {
    let l = MockLauncher::new(None); // sysfs hiccup
    let s = Arc::new(RecordingSink::default());
    let m = manager(l.clone(), s.clone(), 5, None);
    let err = m
        .acquire_warm_slot(&ModelId::from("a"), Some(20.0))
        .await
        .unwrap_err();
    assert!(matches!(err, ResidencyError::CannotAdmit(_)));
    assert!(
        l.launches.lock().unwrap().is_empty(),
        "must not launch when free VRAM is unreadable"
    );
}

// ── pinned-only stall → CannotAdmit without evicting chat ──────────────────────

#[tokio::test]
async fn pinned_only_stall_denies_without_evicting_chat() {
    let l = MockLauncher::new(Some(80.0));
    let s = Arc::new(RecordingSink::default());
    let m = manager(l.clone(), s.clone(), 5, None);
    m.acquire_warm_slot(&ModelId::from("chat"), Some(40.0)).await.unwrap();
    m.set_pinned_chat_model(Some("chat")).await;
    *l.free_gb.lock().unwrap() = Some(0.0);
    l.evictions.lock().unwrap().clear();
    let err = m
        .acquire_warm_slot(&ModelId::from("huge"), Some(70.0))
        .await
        .unwrap_err();
    assert!(matches!(err, ResidencyError::CannotAdmit(_)));
    assert!(l.evictions.lock().unwrap().is_empty());
    // Behavior-spec verify: chat still resident, admission-denied recorded.
    assert!(s.decisions().contains(&"admission-denied".to_string()));
}

// ── atomic state-file writes with required fields ─────────────────────────────

#[tokio::test]
async fn state_file_is_valid_json_with_required_fields() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("residency-state.json");
    let l = MockLauncher::new(Some(60.0));
    let s = Arc::new(RecordingSink::default());
    let m = manager(
        l.clone(),
        s.clone(),
        5,
        Some(path.to_string_lossy().to_string()),
    );
    m.set_pinned_chat_model(Some("chat-model")).await;
    m.acquire_warm_slot(&ModelId::from("a"), Some(20.0)).await.unwrap();

    let raw = std::fs::read_to_string(&path).unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).expect("valid JSON");
    // required_fields: residents, free_vram_gb, pinned_chat_model
    assert!(v.get("residents").and_then(|x| x.as_array()).is_some());
    assert!(v.get("free_vram_gb").is_some());
    assert_eq!(
        v.get("pinned_chat_model").and_then(|x| x.as_str()),
        Some("chat-model")
    );
    // The state file reflects the SERVING state (resident >= 1).
    let residents = v.get("residents").and_then(|x| x.as_array()).unwrap();
    assert!(!residents.is_empty(), "SERVING: at least one resident");
}

#[tokio::test]
async fn state_file_survives_repeated_atomic_rewrites() {
    // Each admission rewrites the file atomically; after several it must still be
    // valid JSON (no torn write) and carry the latest resident set.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("state.json");
    let l = MockLauncher::new(Some(200.0));
    let s = Arc::new(RecordingSink::default());
    let m = manager(
        l.clone(),
        s.clone(),
        5,
        Some(path.to_string_lossy().to_string()),
    );
    for id in ["m1", "m2", "m3"] {
        m.acquire_warm_slot(&ModelId::from(id), Some(20.0)).await.unwrap();
    }
    let raw = std::fs::read_to_string(&path).unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).expect("valid JSON after rewrites");
    let residents = v.get("residents").and_then(|x| x.as_array()).unwrap();
    assert_eq!(residents.len(), 3, "all three resident in the final state");
}

// ── coalescing: a re-requested model during the wait is reused, not double-queued

#[tokio::test]
async fn requested_again_during_wait_coalesces() {
    // Two concurrent requests for the SAME model under keep-warm contention: the
    // second must reuse the resident the first admits, not double-launch.
    let l = MockLauncher::new(Some(120.0));
    let s = Arc::new(RecordingSink::default());
    let m = manager(l.clone(), s.clone(), 30, None);
    m.acquire_warm_slot(&ModelId::from("kw"), Some(40.0)).await.unwrap();
    l.with_footprint("kw", 40.0);
    *l.free_gb.lock().unwrap() = Some(0.0);
    l.launches.lock().unwrap().clear();

    let m1 = m.clone();
    let m2 = m.clone();
    let h1 = tokio::spawn(async move {
        m1.acquire_warm_slot(&ModelId::from("same"), Some(35.0)).await
    });
    let h2 = tokio::spawn(async move {
        m2.acquire_warm_slot(&ModelId::from("same"), Some(35.0)).await
    });
    let r1 = h1.await.unwrap();
    let r2 = h2.await.unwrap();
    assert!(r1.is_ok() && r2.is_ok(), "both resolve to the same warm slot");
    // The model `same` is launched at most once (coalesced).
    let launched_same = l
        .launches
        .lock()
        .unwrap()
        .iter()
        .filter(|x| *x == "same")
        .count();
    assert_eq!(launched_same, 1, "coalesced: launched once");
}
