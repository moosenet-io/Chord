//! SRV-13 integration tests: the operating-mode controller wired into the
//! residency manager — mode is explicit, persisted state; switching OFF
//! assistant-live requires confirm + gracefully unpins; mode survives a restart;
//! and the coordinator state file carries the required fields.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;

use chord_proxy::serving::memory_model::{ActivationEvent, ModelSelection, SeparateCeilings};
use chord_proxy::serving::residency::{
    read_persisted_mode, EventSink, ResidencyEvent, VramResidencyManager, WarmLauncher,
};
use chord_proxy::serving::{ModeAction, ModeError, OperatingMode, ResidencyManager, Tier};
use terminus_rs::intake::serving::{ModelId, Runtime};

struct L(Mutex<Option<f64>>);
#[async_trait]
impl WarmLauncher for L {
    async fn free_vram_gb(&self) -> Option<f64> {
        *self.0.lock().unwrap()
    }
    async fn launch(&self, m: &str, gb: f64) -> Result<(Runtime, String), String> {
        let mut f = self.0.lock().unwrap();
        *f = Some((f.unwrap_or(0.0) - gb).max(0.0));
        Ok((Runtime::LlamaCpp, format!("http://w.invalid/{m}")))
    }
    async fn health_check(&self, _e: &str) -> bool {
        true
    }
    async fn evict(&self, _m: &str) -> Result<(), String> {
        Ok(())
    }
}

#[derive(Default)]
struct Sink(Mutex<Vec<ResidencyEvent>>);
impl EventSink for Sink {
    fn emit(&self, e: &ResidencyEvent) {
        self.0.lock().unwrap().push(e.clone());
    }
}

/// A manager with explicit GPU/CPU ceilings (96/31) under separate-ceilings.
fn mgr(free: f64, state_path: Option<String>) -> VramResidencyManager {
    VramResidencyManager::with_memory_model(
        Arc::new(L(Mutex::new(Some(free)))),
        Arc::new(Sink::default()),
        Duration::from_millis(10),
        state_path,
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
    )
}

#[tokio::test]
async fn defaults_to_assistant_live() {
    let m = mgr(96.0, None);
    assert_eq!(m.mode().await, OperatingMode::AssistantLive);
}

#[tokio::test]
async fn switch_off_assistant_live_requires_confirm_and_unpins() {
    let m = mgr(96.0, None);
    // Pin a chat model (assistant-live).
    m.register_resident("chat", Runtime::LlamaCpp, "http://c.invalid", 20.0, Tier::PinnedChat)
        .await;
    m.set_pinned_chat_model(Some("chat")).await;

    // Without confirm → refused (never silently drop live Lumina).
    assert_eq!(
        m.switch_mode(OperatingMode::BatchCoder, false).await,
        Err(ModeError::NeedsConfirm)
    );
    assert_eq!(m.mode().await, OperatingMode::AssistantLive);
    assert_eq!(m.pinned_chat_model().await.as_deref(), Some("chat"));

    // With confirm → graceful unpin + mode change. The chat model stays RESIDENT
    // (not hard-dropped) but is no longer pinned.
    let action = m.switch_mode(OperatingMode::BatchCoder, true).await.unwrap();
    assert_eq!(action, ModeAction::GracefulUnpin);
    assert_eq!(m.mode().await, OperatingMode::BatchCoder);
    assert!(m.pinned_chat_model().await.is_none(), "assistant gracefully unpinned");
    assert_eq!(m.resident_count().await, 1, "the model is still resident, not dropped");
}

#[tokio::test]
async fn switch_into_assistant_live_returns_load_and_pin() {
    let m = mgr(96.0, None);
    m.switch_mode(OperatingMode::BatchCoder, true).await.unwrap();
    let action = m.switch_mode(OperatingMode::AssistantLive, false).await.unwrap();
    assert_eq!(action, ModeAction::LoadAndPin);
    assert_eq!(m.mode().await, OperatingMode::AssistantLive);
}

#[tokio::test]
async fn assistant_live_headroom_rejects_oversize_coder_reason() {
    let m = mgr(96.0, None);
    // Pinned assistant of 60GB → only 36GB leftover.
    let ctl = {
        // mode_controller reflects the manager's ceilings + mode.
        m.mode_controller().await
    };
    // 80GB coder fits the full 96 GPU but not the 36 leftover → clear reason.
    assert!(ctl.oversize_reason(80.0, 60.0).unwrap().contains("batch-coder"));
    // 30GB coder fits the leftover → no reason.
    assert!(ctl.oversize_reason(30.0, 60.0).is_none());
}

#[tokio::test]
async fn mode_persists_across_restart_via_state_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("coordinator.json");
    let p = path.to_string_lossy().to_string();

    // First process: switch to batch-coder (persisted to the state file).
    let m1 = mgr(96.0, Some(p.clone()));
    m1.switch_mode(OperatingMode::BatchCoder, true).await.unwrap();

    // The state file records the mode + required fields.
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(v.get("mode").and_then(|x| x.as_str()), Some("batch-coder"));
    for f in ["residents", "free_vram_gb", "pinned_chat_model", "assumed_memory_model", "gpu_ceiling_gb", "cpu_ceiling_gb"] {
        assert!(v.get(f).is_some(), "state file missing required field {f}");
    }

    // Second process: restore the persisted mode.
    let restored = read_persisted_mode(&p).unwrap();
    let m2 = mgr(96.0, Some(p.clone()));
    assert_eq!(m2.mode().await, OperatingMode::AssistantLive, "fresh default before restore");
    m2.restore_mode(restored).await;
    assert_eq!(m2.mode().await, OperatingMode::BatchCoder, "mode restored across restart");
}
