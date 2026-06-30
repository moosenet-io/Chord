//! SRV-12 integration tests: the verified clean-swap barrier via the public API.
//!
//! The pure barrier logic is unit-tested in `chord_proxy::serving::swap` /
//! `release_verify`; these exercise the public re-exports end-to-end, including the
//! teardown→escalate→launch path and a CPU-backend swap (the barrier is
//! backend-agnostic — same verify-release contract for system RAM).

use std::sync::Mutex;
use std::time::Duration;

use async_trait::async_trait;

use chord_proxy::serving::{
    clean_swap, ContextDefaults, DeviceProbe, ReleaseConfig, SwapError, SwapEvent, SwapEventSink,
    SwapRequest, Teardown,
};
use chord_proxy::serving::swap::CleanLauncher;

struct Td(Mutex<Vec<String>>);
#[async_trait]
impl Teardown for Td {
    async fn teardown(&self, m: &str) -> Result<(), String> {
        self.0.lock().unwrap().push(m.to_string());
        Ok(())
    }
}

struct Lc(Mutex<Vec<(String, u32)>>);
#[async_trait]
impl CleanLauncher for Lc {
    async fn launch_clean(&self, m: &str, n_ctx: u32) -> Result<String, String> {
        self.0.lock().unwrap().push((m.to_string(), n_ctx));
        Ok(format!("http://clean.invalid/{m}"))
    }
}

#[derive(Default)]
struct Sink(Mutex<Vec<SwapEvent>>);
impl SwapEventSink for Sink {
    fn emit(&self, e: &SwapEvent) {
        self.0.lock().unwrap().push(e.clone());
    }
}

/// A device that stays busy for `busy_polls` reads, has an orphan, and releases
/// after a force-kill — exercising the full teardown→escalate→release path.
struct EscalatingDevice {
    polls: Mutex<u32>,
    killed: Mutex<bool>,
}
#[async_trait]
impl DeviceProbe for EscalatingDevice {
    async fn in_use_gb(&self) -> Option<f64> {
        if *self.killed.lock().unwrap() {
            Some(0.1)
        } else {
            Some(40.0)
        }
    }
    async fn orphan_present(&self) -> bool {
        !*self.killed.lock().unwrap()
    }
    async fn force_kill_orphan(&self) -> Result<(), String> {
        *self.killed.lock().unwrap() = true;
        *self.polls.lock().unwrap() += 1;
        Ok(())
    }
}

fn rcfg() -> ReleaseConfig {
    ReleaseConfig {
        baseline_gb: 0.2,
        tolerance_gb: 0.3,
        timeout: Duration::from_millis(25),
        poll_interval: Duration::from_millis(5),
    }
}
fn ctxd() -> ContextDefaults {
    ContextDefaults { base_ctx: 32768, min_ctx: 8192, large_model_gb: 40.0 }
}

#[tokio::test]
async fn end_to_end_swap_escalates_then_launches_with_explicit_ctx() {
    let td = Td(Mutex::new(vec![]));
    let lc = Lc(Mutex::new(vec![]));
    let sink = Sink::default();
    let dev = EscalatingDevice { polls: Mutex::new(0), killed: Mutex::new(false) };
    let req = SwapRequest {
        from: Some("coder-a"),
        to: "coder-b",
        to_n_ctx: Some(98304),
        to_footprint_gb: Some(18.0),
    };
    let out = clean_swap(&req, &td, &dev, &lc, &rcfg(), &ctxd(), &sink)
        .await
        .unwrap();
    assert!(out.release.escalated(), "the stuck device required a force-kill");
    assert_eq!(out.n_ctx, 98304);
    assert_eq!(*lc.0.lock().unwrap(), vec![("coder-b".to_string(), 98304)]);
    let ev = &sink.0.lock().unwrap()[0];
    assert!(ev.release_verified && ev.escalated && ev.explicit_ctx);
}

#[tokio::test]
async fn cpu_backend_swap_uses_the_same_barrier() {
    // A CPU-tier swap (system RAM) goes through the identical verify-release barrier
    // — the probe just measures a different counter. Here it's already at baseline.
    struct CleanDev;
    #[async_trait]
    impl DeviceProbe for CleanDev {
        async fn in_use_gb(&self) -> Option<f64> {
            Some(0.1)
        }
        async fn orphan_present(&self) -> bool {
            false
        }
        async fn force_kill_orphan(&self) -> Result<(), String> {
            Ok(())
        }
    }
    let td = Td(Mutex::new(vec![]));
    let lc = Lc(Mutex::new(vec![]));
    let sink = Sink::default();
    // No profile ctx, large CPU model → reduced explicit default (still explicit).
    let req = SwapRequest { from: None, to: "cpu-model", to_n_ctx: None, to_footprint_gb: Some(50.0) };
    let out = clean_swap(&req, &td, &CleanDev, &lc, &rcfg(), &ctxd(), &sink)
        .await
        .unwrap();
    assert_eq!(out.n_ctx, 8192, "large model → reduced explicit ctx, never auto-fit");
    assert!(out.release.is_released());
}

#[tokio::test]
async fn stuck_device_refuses_launch() {
    struct StuckDev;
    #[async_trait]
    impl DeviceProbe for StuckDev {
        async fn in_use_gb(&self) -> Option<f64> {
            Some(50.0)
        }
        async fn orphan_present(&self) -> bool {
            false
        }
        async fn force_kill_orphan(&self) -> Result<(), String> {
            Ok(())
        }
    }
    let td = Td(Mutex::new(vec![]));
    let lc = Lc(Mutex::new(vec![]));
    let sink = Sink::default();
    let req = SwapRequest { from: Some("a"), to: "b", to_n_ctx: Some(32768), to_footprint_gb: None };
    let err = clean_swap(&req, &td, &StuckDev, &lc, &rcfg(), &ctxd(), &sink)
        .await
        .unwrap_err();
    assert_eq!(err, SwapError::ReleaseStuck);
    assert!(lc.0.lock().unwrap().is_empty(), "never launch onto a polluted device");
}
