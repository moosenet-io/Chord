//! SRV-06 integration tests: chat-role PIN + routing integration.
//!
//! Drives the real [`VramResidencyManager`] (mocked launcher, no GPU) through
//! [`apply_chat_pin`]: the chat-role selection becomes the residency manager's
//! pinned, never-evicted model; the atomic pin transfer loads + pins the new model
//! before releasing the old (Lumina is never without a resident chat model); the
//! serving-profile latency guard blocks an unresponsive-cold-start interactive
//! alias; and the pin survives admission pressure. Covers the SRV-06 TEST PLAN.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;

use chord_proxy::routing::assistant_profile::{apply_chat_pin, ChatPinOutcome};
use chord_proxy::serving::residency::{EventSink, ResidencyEvent, VramResidencyManager, WarmLauncher};
use chord_proxy::serving::profile::RoutingMap;
use chord_proxy::serving::ResidencyManager;
use terminus_rs::intake::assistant::reporting::{ChatRoleSelection, ModelKey};
use terminus_rs::intake::serving::{
    ExclusionReason, ModelId, RecheckTrigger, Runtime, ServingBackend, ServingProfile,
};

use chord_proxy::routing::assistant_profile::{decide_chat_role, ChatRoleDecision};

// ── test doubles ──────────────────────────────────────────────────────────────

/// A scripted launcher with settable free VRAM, recorded launches/evicts, a health
/// toggle, and a per-model footprint table so eviction frees VRAM realistically.
struct MockLauncher {
    free_gb: Mutex<Option<f64>>,
    healthy: Mutex<bool>,
    launches: Mutex<Vec<String>>,
    evictions: Mutex<Vec<String>>,
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
    fn fail_health(self: &Arc<Self>) {
        *self.healthy.lock().unwrap() = false;
    }
}

#[async_trait]
impl WarmLauncher for MockLauncher {
    async fn free_vram_gb(&self) -> Option<f64> {
        *self.free_gb.lock().unwrap()
    }
    async fn launch(&self, model_id: &str, gb: f64) -> Result<(Runtime, String), String> {
        self.launches.lock().unwrap().push(model_id.to_string());
        let mut f = self.free_gb.lock().unwrap();
        *f = Some((f.unwrap_or(0.0) - gb).max(0.0));
        Ok((Runtime::LlamaCpp, format!("http://warm.invalid/{model_id}")))
    }
    async fn health_check(&self, _endpoint: &str) -> bool {
        *self.healthy.lock().unwrap()
    }
    async fn evict(&self, model_id: &str) -> Result<(), String> {
        self.evictions.lock().unwrap().push(model_id.to_string());
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

fn manager(launcher: Arc<MockLauncher>) -> VramResidencyManager {
    VramResidencyManager::with_settings(
        launcher,
        Arc::new(RecordingSink::default()),
        Duration::from_millis(10),
        None,
    )
}

/// Build a serving row. `cold` drives the latency guard; `keep_warm` decides
/// whether a slow cold-load is allowed (flagged) or blocked.
fn profile(model: &str, vram: f64, cold: f64, keep_warm: bool) -> ServingProfile {
    ServingProfile {
        model_id: ModelId::from(model),
        backend_tag: ServingBackend::LlamaGpu,
        best_runtime: Runtime::LlamaCpp,
        env_json: "{}".into(),
        tok_s: Some(40.0),
        vram_or_ram_peak_gb: Some(vram),
        cold_load_s: Some(cold),
        keep_warm,
        fallback_runtime: None,
        exclusion_reason: ExclusionReason::None,
        recheck_trigger: RecheckTrigger::None,
        provenance: None,
    }
}

fn routing(rows: Vec<ServingProfile>) -> RoutingMap {
    RoutingMap::load_from(rows)
}

fn route_decision(model: &str) -> ChatRoleDecision {
    // A guard-cleared selection that routes to `model` (registry-known here).
    let sel = ChatRoleSelection {
        candidates: vec![],
        selected: Some(ModelKey {
            model_id: model.into(),
            backend_tag: "gpu".into(),
        }),
    };
    decide_chat_role(&sel, &[])
}

const MAX_COLD: f64 = 300.0;

// ── tests ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn chat_role_selection_marks_model_pinned() {
    let l = MockLauncher::new(Some(80.0));
    let m = manager(l.clone());
    let map = routing(vec![profile("qwen3:8b", 20.0, 15.0, false)]);

    let out = apply_chat_pin(&route_decision("qwen3:8b"), &map, &m, MAX_COLD).await;
    assert_eq!(out.pinned_model(), Some("qwen3:8b"));
    assert_eq!(m.pinned_chat_model().await.as_deref(), Some("qwen3:8b"));
    // It was actually brought resident.
    assert_eq!(*l.launches.lock().unwrap(), vec!["qwen3:8b".to_string()]);
}

#[tokio::test]
async fn keep_default_decision_leaves_pin_untouched() {
    let l = MockLauncher::new(Some(80.0));
    let m = manager(l.clone());
    m.set_pinned_chat_model(Some("incumbent:8b")).await;
    let map = routing(vec![]);

    let decision = ChatRoleDecision::KeepDefault {
        reason: "no candidate cleared the guard".into(),
    };
    let out = apply_chat_pin(&decision, &map, &m, MAX_COLD).await;
    assert!(matches!(out, ChatPinOutcome::KeptCurrent { .. }));
    // The existing pin is unchanged; no launch happened.
    assert_eq!(m.pinned_chat_model().await.as_deref(), Some("incumbent:8b"));
    assert!(l.launches.lock().unwrap().is_empty());
}

#[tokio::test]
async fn chat_role_switch_transfers_pin_atomically() {
    // Old chat model resident + pinned; switch to a new model. The new model must
    // be loaded + pinned, and the old released — but Lumina is NEVER left without a
    // resident chat model mid-transfer.
    let l = MockLauncher::new(Some(80.0));
    l.with_footprint("old:8b", 20.0);
    let m = manager(l.clone());

    // Establish the incumbent pin via the normal path.
    apply_chat_pin(&route_decision("old:8b"), &routing(vec![profile("old:8b", 20.0, 15.0, false)]), &m, MAX_COLD).await;
    assert_eq!(m.pinned_chat_model().await.as_deref(), Some("old:8b"));

    // Switch.
    let map = routing(vec![profile("new:8b", 20.0, 15.0, false)]);
    let out = apply_chat_pin(&route_decision("new:8b"), &map, &m, MAX_COLD).await;
    assert_eq!(out.pinned_model(), Some("new:8b"));
    assert_eq!(m.pinned_chat_model().await.as_deref(), Some("new:8b"));
    // The new model was loaded BEFORE the old pin was released (the old was never
    // evicted to make room — it stayed resident through the transfer).
    let launches = l.launches.lock().unwrap().clone();
    assert_eq!(launches, vec!["old:8b".to_string(), "new:8b".to_string()]);
    assert!(
        l.evictions.lock().unwrap().is_empty(),
        "the pinned incumbent must not be evicted during an atomic transfer"
    );
}

#[tokio::test]
async fn failed_load_keeps_old_pin() {
    // The new model fails its health check → keep the working incumbent pin (never
    // surrender a working chat alias for a broken one).
    let l = MockLauncher::new(Some(80.0));
    let m = manager(l.clone());
    m.set_pinned_chat_model(Some("incumbent:8b")).await;

    l.fail_health();
    let map = routing(vec![profile("broken:8b", 20.0, 15.0, false)]);
    let out = apply_chat_pin(&route_decision("broken:8b"), &map, &m, MAX_COLD).await;
    assert!(matches!(out, ChatPinOutcome::KeptCurrent { .. }));
    assert_eq!(
        m.pinned_chat_model().await.as_deref(),
        Some("incumbent:8b"),
        "Lumina must never be left without a resident chat model"
    );
}

#[tokio::test]
async fn latency_guard_blocks_unresponsive_non_warm_model() {
    // Slow cold-load AND not keep-warm → cold on every use → blocked as the
    // interactive alias. The pin is left as-is.
    let l = MockLauncher::new(Some(120.0));
    let m = manager(l.clone());
    let map = routing(vec![profile("slow-moe:120b", 90.0, 600.0, false)]);

    let out = apply_chat_pin(&route_decision("slow-moe:120b"), &map, &m, MAX_COLD).await;
    assert!(matches!(out, ChatPinOutcome::KeptCurrent { .. }));
    assert!(m.pinned_chat_model().await.is_none());
    assert!(l.launches.lock().unwrap().is_empty(), "blocked model is never loaded");
}

#[tokio::test]
async fn keep_warm_big_moe_allowed_but_flagged() {
    // Slow cold-load BUT keep-warm → allowed (warm residency mitigates steady-state
    // latency) and FLAGGED (first cold-start still applies).
    let l = MockLauncher::new(Some(120.0));
    let m = manager(l.clone());
    let map = routing(vec![profile("minimax-m2.7", 90.0, 600.0, true)]);

    let out = apply_chat_pin(&route_decision("minimax-m2.7"), &map, &m, MAX_COLD).await;
    match out {
        ChatPinOutcome::Pinned { model_id, cold_start_flagged } => {
            assert_eq!(model_id, "minimax-m2.7");
            assert!(cold_start_flagged, "a slow keep-warm pin must record the cold-start tradeoff");
        }
        other => panic!("expected a flagged pin, got {other:?}"),
    }
    assert_eq!(m.pinned_chat_model().await.as_deref(), Some("minimax-m2.7"));
}

#[tokio::test]
async fn unprofiled_pick_keeps_current_pin() {
    let l = MockLauncher::new(Some(80.0));
    let m = manager(l.clone());
    m.set_pinned_chat_model(Some("incumbent:8b")).await;
    let map = routing(vec![]); // no profile for the pick

    let out = apply_chat_pin(&route_decision("ghost:99b"), &map, &m, MAX_COLD).await;
    assert!(matches!(out, ChatPinOutcome::KeptCurrent { .. }));
    assert_eq!(m.pinned_chat_model().await.as_deref(), Some("incumbent:8b"));
}

#[tokio::test]
async fn pin_survives_admission_pressure() {
    // Pin the chat model, then drive sustained admission pressure with a competing
    // keep-warm launch that can't fit → the pinned chat model is retained.
    let l = MockLauncher::new(Some(40.0));
    let m = manager(l.clone());
    let map = routing(vec![profile("chat:8b", 30.0, 15.0, false)]);
    apply_chat_pin(&route_decision("chat:8b"), &map, &m, MAX_COLD).await;
    assert_eq!(m.pinned_chat_model().await.as_deref(), Some("chat:8b"));

    // A 30GB competitor cannot fit in the ~10GB headroom and the only resident is
    // the pinned chat model → it must be denied, never evict the chat pin.
    let err = m
        .acquire_warm_slot(&ModelId::from("competitor:30b"), Some(30.0))
        .await
        .unwrap_err();
    let _ = err; // CannotAdmit
    assert_eq!(
        m.pinned_chat_model().await.as_deref(),
        Some("chat:8b"),
        "pinned chat model retained under admission pressure"
    );
    assert!(!l.evictions.lock().unwrap().contains(&"chat:8b".to_string()));
}

#[tokio::test]
async fn idempotent_when_already_pinned() {
    let l = MockLauncher::new(Some(80.0));
    let m = manager(l.clone());
    let map = routing(vec![profile("qwen3:8b", 20.0, 15.0, false)]);
    apply_chat_pin(&route_decision("qwen3:8b"), &map, &m, MAX_COLD).await;
    let launches_after_first = l.launches.lock().unwrap().len();

    // Re-applying the same decision must not reload.
    let out = apply_chat_pin(&route_decision("qwen3:8b"), &map, &m, MAX_COLD).await;
    assert_eq!(out.pinned_model(), Some("qwen3:8b"));
    assert_eq!(
        l.launches.lock().unwrap().len(),
        launches_after_first,
        "an already-pinned model must not be cold-cycled"
    );
}
