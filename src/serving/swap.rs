//! Verified clean-swap transition barrier (S85 SRV-12).
//!
//! The deterministic barrier run between model changes: tear down the outgoing
//! model, **verify** the device returned to baseline (no leaked memory, no
//! orphaned backend), then launch the incoming model with its correct explicit
//! flags — crucially an explicit `-c` so the launch never depends on any backend's
//! auto-fit, permanently sidestepping the UMA context-default slash. Teardown is
//! not trusted; release is verified ([`super::release_verify`]). If the device is
//! still polluted after escalation, the barrier REFUSES to launch rather than
//! launch onto leaked memory (which would surface as a false OOM).

use async_trait::async_trait;

use super::release_verify::{verify_release, DeviceProbe, ReleaseConfig, ReleaseOutcome};

/// Tears down the outgoing model's backend (signal stop + wait for process exit).
#[async_trait]
pub trait Teardown: Send + Sync {
    /// Stop `model_id`'s backend and wait for it to exit. `Err` ⇒ teardown failed
    /// (the barrier aborts; it never proceeds to verify/launch on a failed teardown).
    async fn teardown(&self, model_id: &str) -> Result<(), String>;
}

/// Launches the incoming model with EXPLICIT flags (incl. `-c <n_ctx>`).
#[async_trait]
pub trait CleanLauncher: Send + Sync {
    /// Launch `model_id` with an explicit `n_ctx` (plus its profile flags) and
    /// health-check it. Returns the served endpoint. `Err` ⇒ launch/health failed.
    async fn launch_clean(&self, model_id: &str, n_ctx: u32) -> Result<String, String>;
}

/// A sanitized swap event (S6/S77): model ids (registry keys, not infra) + the
/// verification facts. Carries NO endpoint/host/path.
#[derive(Debug, Clone, PartialEq)]
pub struct SwapEvent {
    pub from: Option<String>,
    pub to: String,
    /// Whether the device was verified released before the launch.
    pub release_verified: bool,
    /// Whether escalation (force-kill of an orphan) was required.
    pub escalated: bool,
    /// ms to reach baseline (0 when the swap was refused before launch).
    pub baseline_ms: u64,
    /// ALWAYS true on a completed swap — the launch carried an explicit `-c`.
    pub explicit_ctx: bool,
    /// The explicit context window the incoming model launched with.
    pub n_ctx: u32,
}

/// Sink for swap events. Production wires structured logging; tests record them.
pub trait SwapEventSink: Send + Sync {
    fn emit(&self, event: &SwapEvent);
}

/// A no-op sink.
pub struct NoopSwapEventSink;
impl SwapEventSink for NoopSwapEventSink {
    fn emit(&self, _e: &SwapEvent) {}
}

/// Config for computing a SAFE explicit context when the profile carries none —
/// never the backend's auto-fit (which trips the UMA slash). From config helpers.
#[derive(Debug, Clone)]
pub struct ContextDefaults {
    /// The default explicit context for a normal-sized model.
    pub base_ctx: u32,
    /// A reduced context for a large model (footprint above `large_model_gb`).
    pub min_ctx: u32,
    /// Footprint (GB) above which `min_ctx` is used instead of `base_ctx`.
    pub large_model_gb: f64,
}

/// Compute a safe explicit context from the model's footprint — deterministic,
/// never the backend's auto-fit. A larger model gets a smaller default so the
/// (explicit) context always fits without relying on the misread free counter.
pub fn default_ctx_for_footprint(footprint_gb: Option<f64>, cfg: &ContextDefaults) -> u32 {
    match footprint_gb {
        Some(f) if f.is_finite() && f > cfg.large_model_gb => cfg.min_ctx,
        _ => cfg.base_ctx,
    }
}

/// One swap request: tear `from` (if any) down, bring `to` up.
pub struct SwapRequest<'a> {
    /// The outgoing resident model id, or `None` if nothing is resident.
    pub from: Option<&'a str>,
    /// The incoming model id.
    pub to: &'a str,
    /// The incoming model's explicit context from its profile (`env_json.n_ctx`).
    /// `None` ⇒ a safe default is computed from the footprint.
    pub to_n_ctx: Option<u32>,
    /// The incoming model's footprint (GB) — used only to compute a default ctx.
    pub to_footprint_gb: Option<f64>,
}

/// The result of a successful swap.
#[derive(Debug, Clone, PartialEq)]
pub struct SwapOutcome {
    pub from: Option<String>,
    pub to: String,
    pub release: ReleaseOutcome,
    /// The explicit `-c` the incoming model launched with.
    pub n_ctx: u32,
    /// The served endpoint of the incoming model.
    pub endpoint: String,
}

/// Why a swap failed.
#[derive(Debug, Clone, PartialEq)]
pub enum SwapError {
    /// Tearing down the outgoing model failed — never proceeded to launch.
    TeardownFailed(String),
    /// The device did not return to baseline even after escalation — the barrier
    /// REFUSED to launch onto a polluted device (the false-OOM guard).
    ReleaseStuck,
    /// The incoming model failed to launch/health-check on a verified-clean device.
    LaunchFailed(String),
}

/// Run the verified clean-swap barrier. The ONLY path that brings a model resident
/// when another must come down first.
///
/// Steps: teardown(from) → [`verify_release`] → (refuse if stuck) → compute the
/// EXPLICIT context → `launch_clean(to, n_ctx)` → emit a sanitized [`SwapEvent`].
#[allow(clippy::too_many_arguments)]
pub async fn clean_swap(
    req: &SwapRequest<'_>,
    teardown: &dyn Teardown,
    probe: &dyn DeviceProbe,
    launcher: &dyn CleanLauncher,
    release_cfg: &ReleaseConfig,
    ctx_defaults: &ContextDefaults,
    events: &dyn SwapEventSink,
) -> Result<SwapOutcome, SwapError> {
    // (1) Tear the outgoing model down. A failed teardown aborts BEFORE any verify
    // or launch — we never launch over a backend we couldn't stop.
    if let Some(from) = req.from {
        teardown
            .teardown(from)
            .await
            .map_err(SwapError::TeardownFailed)?;
    }

    // (2) Verify the device returned to baseline (escalating to a force-kill if an
    // orphan lingers). Teardown is not trusted — this is the point.
    let release = verify_release(probe, release_cfg).await;
    if !release.is_released() {
        // (3a) REFUSE to launch onto a polluted device. Record the failure.
        events.emit(&SwapEvent {
            from: req.from.map(|s| s.to_string()),
            to: req.to.to_string(),
            release_verified: false,
            escalated: false,
            baseline_ms: 0,
            explicit_ctx: false,
            n_ctx: 0,
        });
        return Err(SwapError::ReleaseStuck);
    }

    // (3b) Compute the EXPLICIT context — from the profile, else a safe footprint
    // default. NEVER a default-context (auto-fit) launch.
    let n_ctx = req
        .to_n_ctx
        .unwrap_or_else(|| default_ctx_for_footprint(req.to_footprint_gb, ctx_defaults));

    // (4) Launch the incoming model clean, with the explicit context.
    let endpoint = launcher
        .launch_clean(req.to, n_ctx)
        .await
        .map_err(SwapError::LaunchFailed)?;

    // (5) Emit the sanitized swap event.
    let baseline_ms = match &release {
        ReleaseOutcome::Clean { baseline_ms } | ReleaseOutcome::Escalated { baseline_ms } => {
            *baseline_ms
        }
        ReleaseOutcome::Stuck => 0,
    };
    events.emit(&SwapEvent {
        from: req.from.map(|s| s.to_string()),
        to: req.to.to_string(),
        release_verified: true,
        escalated: release.escalated(),
        baseline_ms,
        explicit_ctx: true,
        n_ctx,
    });

    Ok(SwapOutcome {
        from: req.from.map(|s| s.to_string()),
        to: req.to.to_string(),
        release,
        n_ctx,
        endpoint,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::time::Duration;

    struct MockTeardown {
        ok: bool,
        torn: Mutex<Vec<String>>,
    }
    #[async_trait]
    impl Teardown for MockTeardown {
        async fn teardown(&self, model_id: &str) -> Result<(), String> {
            self.torn.lock().unwrap().push(model_id.to_string());
            if self.ok {
                Ok(())
            } else {
                Err("teardown failed".into())
            }
        }
    }

    struct MockLauncher {
        ok: bool,
        launched: Mutex<Vec<(String, u32)>>,
    }
    #[async_trait]
    impl CleanLauncher for MockLauncher {
        async fn launch_clean(&self, model_id: &str, n_ctx: u32) -> Result<String, String> {
            self.launched
                .lock()
                .unwrap()
                .push((model_id.to_string(), n_ctx));
            if self.ok {
                Ok(format!("http://clean.invalid/{model_id}"))
            } else {
                Err("launch failed".into())
            }
        }
    }

    struct CleanProbe; // always already at baseline
    #[async_trait]
    impl DeviceProbe for CleanProbe {
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

    struct StuckProbe; // never releases, no orphan
    #[async_trait]
    impl DeviceProbe for StuckProbe {
        async fn in_use_gb(&self) -> Option<f64> {
            Some(40.0)
        }
        async fn orphan_present(&self) -> bool {
            false
        }
        async fn force_kill_orphan(&self) -> Result<(), String> {
            Ok(())
        }
    }

    #[derive(Default)]
    struct RecSink {
        events: Mutex<Vec<SwapEvent>>,
    }
    impl SwapEventSink for RecSink {
        fn emit(&self, e: &SwapEvent) {
            self.events.lock().unwrap().push(e.clone());
        }
    }

    fn rcfg() -> ReleaseConfig {
        ReleaseConfig {
            baseline_gb: 0.2,
            tolerance_gb: 0.3,
            timeout: Duration::from_millis(30),
            poll_interval: Duration::from_millis(5),
        }
    }
    fn ctxd() -> ContextDefaults {
        ContextDefaults {
            base_ctx: 32768,
            min_ctx: 8192,
            large_model_gb: 40.0,
        }
    }

    #[tokio::test]
    async fn happy_swap_tears_down_verifies_then_launches_with_explicit_ctx() {
        let td = MockTeardown { ok: true, torn: Mutex::new(vec![]) };
        let lc = MockLauncher { ok: true, launched: Mutex::new(vec![]) };
        let sink = RecSink::default();
        let req = SwapRequest {
            from: Some("old"),
            to: "new",
            to_n_ctx: Some(65536),
            to_footprint_gb: Some(18.0),
        };
        let out = clean_swap(&req, &td, &CleanProbe, &lc, &rcfg(), &ctxd(), &sink)
            .await
            .unwrap();
        assert_eq!(*td.torn.lock().unwrap(), vec!["old".to_string()]);
        // Launched with the profile's explicit context, not auto-fit.
        assert_eq!(*lc.launched.lock().unwrap(), vec![("new".to_string(), 65536)]);
        assert_eq!(out.n_ctx, 65536);
        let ev = &sink.events.lock().unwrap()[0];
        assert!(ev.release_verified);
        assert!(ev.explicit_ctx, "swap must record an explicit-ctx launch");
        assert_eq!(ev.n_ctx, 65536);
    }

    #[tokio::test]
    async fn computes_explicit_default_ctx_when_profile_has_none() {
        let td = MockTeardown { ok: true, torn: Mutex::new(vec![]) };
        let lc = MockLauncher { ok: true, launched: Mutex::new(vec![]) };
        let sink = RecSink::default();
        // No profile n_ctx; small footprint → base_ctx (still EXPLICIT, never auto).
        let req = SwapRequest { from: None, to: "m", to_n_ctx: None, to_footprint_gb: Some(18.0) };
        let out = clean_swap(&req, &td, &CleanProbe, &lc, &rcfg(), &ctxd(), &sink).await.unwrap();
        assert_eq!(out.n_ctx, 32768);
        // A LARGE model gets the reduced explicit default.
        let req2 = SwapRequest { from: None, to: "big", to_n_ctx: None, to_footprint_gb: Some(60.0) };
        let out2 = clean_swap(&req2, &td, &CleanProbe, &lc, &rcfg(), &ctxd(), &sink).await.unwrap();
        assert_eq!(out2.n_ctx, 8192);
        // EVERY launch carried an explicit -c (asserted: never a default-context launch).
        for (_m, ctx) in lc.launched.lock().unwrap().iter() {
            assert!(*ctx > 0, "every launch must carry an explicit -c");
        }
    }

    #[tokio::test]
    async fn refuses_launch_when_release_stuck() {
        let td = MockTeardown { ok: true, torn: Mutex::new(vec![]) };
        let lc = MockLauncher { ok: true, launched: Mutex::new(vec![]) };
        let sink = RecSink::default();
        let req = SwapRequest { from: Some("old"), to: "new", to_n_ctx: Some(32768), to_footprint_gb: None };
        let err = clean_swap(&req, &td, &StuckProbe, &lc, &rcfg(), &ctxd(), &sink)
            .await
            .unwrap_err();
        assert_eq!(err, SwapError::ReleaseStuck);
        // NEGATIVE TEST: never launched onto the polluted device.
        assert!(lc.launched.lock().unwrap().is_empty());
        let ev = &sink.events.lock().unwrap()[0];
        assert!(!ev.release_verified, "the swap-failure event records the unverified device");
    }

    #[tokio::test]
    async fn aborts_on_teardown_failure_without_launching() {
        let td = MockTeardown { ok: false, torn: Mutex::new(vec![]) };
        let lc = MockLauncher { ok: true, launched: Mutex::new(vec![]) };
        let sink = RecSink::default();
        let req = SwapRequest { from: Some("old"), to: "new", to_n_ctx: Some(32768), to_footprint_gb: None };
        let err = clean_swap(&req, &td, &CleanProbe, &lc, &rcfg(), &ctxd(), &sink)
            .await
            .unwrap_err();
        assert!(matches!(err, SwapError::TeardownFailed(_)));
        assert!(lc.launched.lock().unwrap().is_empty(), "no launch after a failed teardown");
    }

    #[test]
    fn default_ctx_scales_with_footprint() {
        let c = ctxd();
        assert_eq!(default_ctx_for_footprint(Some(10.0), &c), 32768);
        assert_eq!(default_ctx_for_footprint(Some(60.0), &c), 8192);
        assert_eq!(default_ctx_for_footprint(None, &c), 32768);
    }
}
