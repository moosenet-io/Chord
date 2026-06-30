//! SRV-11 integration tests: substrate-aware memory accounting wired into the
//! residency manager.
//!
//! The pure trait/detection maths are unit-tested in
//! `chord_proxy::serving::memory_model`; these tests verify the SELECTION is wired
//! end-to-end: the manager records `assumed_memory_model`, emits the (loud)
//! activation event at construction, and admits against the ACTIVE model — so a
//! UnifiedPool manager sees the cross-pool effect that a SeparateCeilings one does
//! not.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;

use chord_proxy::serving::memory_model::{
    select_memory_model, ActivationEvent, ModelSelection, SeparateCeilings, SubstrateInfo,
    UnifiedPool,
};
use chord_proxy::serving::residency::{
    EventSink, ResidencyEvent, VramResidencyManager, WarmLauncher,
};
use chord_proxy::serving::{ResidencyManager, Tier};
use terminus_rs::intake::serving::{ModelId, Runtime};

struct MockLauncher {
    free_gb: Mutex<Option<f64>>,
    launches: Mutex<Vec<String>>,
    evictions: Mutex<Vec<String>>,
}
impl MockLauncher {
    fn new(free: Option<f64>) -> Arc<Self> {
        Arc::new(MockLauncher {
            free_gb: Mutex::new(free),
            launches: Mutex::new(vec![]),
            evictions: Mutex::new(vec![]),
        })
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
    async fn health_check(&self, _e: &str) -> bool {
        true
    }
    async fn evict(&self, m: &str) -> Result<(), String> {
        self.evictions.lock().unwrap().push(m.to_string());
        Ok(())
    }
}

#[derive(Default)]
struct RecordingSink {
    events: Mutex<Vec<ResidencyEvent>>,
}
impl EventSink for RecordingSink {
    fn emit(&self, e: &ResidencyEvent) {
        self.events.lock().unwrap().push(e.clone());
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

fn mgr_with(selection: ModelSelection, free: Option<f64>) -> (VramResidencyManager, Arc<RecordingSink>) {
    let l = MockLauncher::new(free);
    let s = Arc::new(RecordingSink::default());
    let m = VramResidencyManager::with_memory_model(
        l,
        s.clone(),
        Duration::from_millis(10),
        None,
        selection,
    );
    (m, s)
}

#[tokio::test]
async fn fixed_carveout_selects_separate_ceilings_and_records_it() {
    let sel = select_memory_model(&SubstrateInfo {
        vram_carveout_gb: 96.0,
        gtt_total_gb: 32.0,
        system_ram_gb: 31.0,
    });
    let (m, s) = mgr_with(sel, Some(40.0));
    assert_eq!(m.assumed_memory_model(), "separate-ceilings");
    // The selection is always announced as a structured event (negative test
    // anchor: the activation is never invisible).
    assert!(s.decisions().contains(&"memory-model".to_string()));
}

#[tokio::test]
async fn dynamic_gtt_auto_activates_unified_pool_loudly() {
    let sel = select_memory_model(&SubstrateInfo {
        vram_carveout_gb: 0.5,
        gtt_total_gb: 120.0,
        system_ram_gb: 127.0,
    });
    assert!(sel.event.loud, "dynamic-GTT activation must be loud");
    let (m, s) = mgr_with(sel, Some(100.0));
    assert_eq!(m.assumed_memory_model(), "unified-pool");
    assert!(s.decisions().contains(&"memory-model".to_string()));
}

#[tokio::test]
async fn separate_ceilings_admits_against_gpu_free_only() {
    // Active default behaviour: a GPU model admits against GPU free.
    let sel = ModelSelection {
        model: Arc::new(SeparateCeilings),
        event: ActivationEvent {
            assumed_memory_model: "separate-ceilings",
            trigger: "test".into(),
            gpu_pool_gb: 96.0,
            cpu_pool_gb: 31.0,
            loud: false,
        },
    };
    let (m, l) = (
        VramResidencyManager::with_memory_model(
            MockLauncher::new(Some(40.0)),
            Arc::new(RecordingSink::default()),
            Duration::from_millis(10),
            None,
            sel,
        ),
        (),
    );
    let _ = l;
    let slot = m
        .acquire_warm_slot(&ModelId::from("qwen3:8b"), Some(20.0))
        .await
        .unwrap();
    assert_eq!(slot.model_id, "qwen3:8b");
}

#[tokio::test]
async fn unified_pool_cpu_resident_forces_gpu_eviction_separate_does_not() {
    // The cross-pool effect, observed through the manager. Setup: a generous GPU
    // free counter (50GB) but a 90GB CPU-tier resident on a 100GB physical pool.
    //
    //   - UnifiedPool: admissible = 100 - 90 = 10 < 20 → the GPU candidate cannot
    //     fit until the (keep-warm) CPU resident is evicted. So an eviction FIRES —
    //     a CPU resident forced a GPU admission to reclaim, which SeparateCeilings
    //     can never see.
    //   - SeparateCeilings: the GPU candidate checks GPU free (50) only, ignores the
    //     CPU resident entirely → admits immediately, NO eviction.
    fn unified_mgr() -> (VramResidencyManager, Arc<MockLauncher>) {
        let l = MockLauncher::new(Some(50.0));
        let m = VramResidencyManager::with_memory_model(
            l.clone(),
            Arc::new(RecordingSink::default()),
            Duration::from_millis(10),
            None,
            ModelSelection {
                model: Arc::new(UnifiedPool { physical_total_gb: 100.0 }),
                event: ActivationEvent {
                    assumed_memory_model: "unified-pool",
                    trigger: "test".into(),
                    gpu_pool_gb: 100.0,
                    cpu_pool_gb: 100.0,
                    loud: true,
                },
            },
        );
        (m, l)
    }
    fn separate_mgr() -> (VramResidencyManager, Arc<MockLauncher>) {
        let l = MockLauncher::new(Some(50.0));
        let m = VramResidencyManager::with_memory_model(
            l.clone(),
            Arc::new(RecordingSink::default()),
            Duration::from_millis(10),
            None,
            ModelSelection {
                model: Arc::new(SeparateCeilings),
                event: ActivationEvent {
                    assumed_memory_model: "separate-ceilings",
                    trigger: "test".into(),
                    gpu_pool_gb: 96.0,
                    cpu_pool_gb: 31.0,
                    loud: false,
                },
            },
        );
        (m, l)
    }

    // UnifiedPool → the CPU resident forces an eviction.
    let (mu, lu) = unified_mgr();
    mu.register_resident("cpu-big", Runtime::Cpu, "http://cpu.invalid", 90.0, Tier::KeepWarm)
        .await;
    mu.acquire_warm_slot(&ModelId::from("new"), Some(20.0))
        .await
        .unwrap();
    assert_eq!(
        *lu.evictions.lock().unwrap(),
        vec!["cpu-big".to_string()],
        "unified: a CPU resident must force the GPU candidate to evict it"
    );

    // SeparateCeilings → same setup, NO eviction (GPU free alone suffices).
    let (ms, ls) = separate_mgr();
    ms.register_resident("cpu-big", Runtime::Cpu, "http://cpu.invalid", 90.0, Tier::KeepWarm)
        .await;
    ms.acquire_warm_slot(&ModelId::from("new"), Some(20.0))
        .await
        .unwrap();
    assert!(
        ls.evictions.lock().unwrap().is_empty(),
        "separate-ceilings: a CPU resident must NOT affect a GPU admission"
    );
}

#[tokio::test]
async fn state_file_records_assumed_memory_model() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("residency.json");
    let sel = select_memory_model(&SubstrateInfo {
        vram_carveout_gb: 0.5,
        gtt_total_gb: 120.0,
        system_ram_gb: 127.0,
    });
    let m = VramResidencyManager::with_memory_model(
        MockLauncher::new(Some(80.0)),
        Arc::new(RecordingSink::default()),
        Duration::from_millis(10),
        Some(path.to_string_lossy().to_string()),
        sel,
    );
    m.acquire_warm_slot(&ModelId::from("a"), Some(20.0))
        .await
        .unwrap();
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(
        v.get("assumed_memory_model").and_then(|x| x.as_str()),
        Some("unified-pool")
    );
}
