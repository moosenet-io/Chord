//! Release verification for the clean-swap barrier (S85 SRV-12).
//!
//! Teardown is NOT trusted — ROCm does not always reclaim VRAM the instant a
//! backend exits, and a crashed `llama-server`/`ollama` can linger holding the
//! device. After tearing down the outgoing model the barrier ([`super::swap`])
//! calls [`verify_release`] to CONFIRM the device actually returned to its idle
//! baseline before launching the next model onto it — otherwise the new model
//! launches into leaked memory and reports a false OOM.
//!
//! The verification: poll the device's in-use counter until it falls to
//! `baseline + tolerance`; on timeout, if an orphaned backend is still holding the
//! device, force-kill it (escalation) and re-poll; if it is STILL not released,
//! report [`ReleaseOutcome::Stuck`] so the caller refuses to launch.

use std::time::{Duration, Instant};

use async_trait::async_trait;

/// The device a swap reclaims. Abstracted so tests drive deterministic
/// release/orphan/escalation behaviour without a GPU or real processes; the
/// production impl reads the sysfs counter + checks for an orphaned backend pid.
#[async_trait]
pub trait DeviceProbe: Send + Sync {
    /// Memory currently IN USE on the device, in GB. At/below
    /// `baseline + tolerance` ⇒ released. `None` ⇒ counter unreadable → treated as
    /// "not released" (fail-safe: never assume a clean device we can't measure).
    async fn in_use_gb(&self) -> Option<f64>;

    /// Whether an orphaned backend process is still holding the device after
    /// teardown (e.g. a crashed `llama-server` keeping `/dev/kfd`/VRAM).
    async fn orphan_present(&self) -> bool;

    /// Force-kill the orphaned backend (the escalation step). `Err` ⇒ the kill
    /// itself failed; the caller then reports the device stuck.
    async fn force_kill_orphan(&self) -> Result<(), String>;
}

/// Tunables for [`verify_release`]. All sourced from config (no literals).
#[derive(Debug, Clone)]
pub struct ReleaseConfig {
    /// The device's idle in-use baseline in GB (e.g. amdgpu idles ~0.15 GB).
    pub baseline_gb: f64,
    /// Tolerance above baseline that still counts as "released".
    pub tolerance_gb: f64,
    /// Total wall-clock budget to wait for release before escalating.
    pub timeout: Duration,
    /// Poll interval while waiting.
    pub poll_interval: Duration,
}

/// The verdict of a release verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReleaseOutcome {
    /// Memory returned to baseline cleanly (no escalation needed).
    Clean { baseline_ms: u64 },
    /// Returned to baseline only AFTER force-killing an orphaned backend.
    Escalated { baseline_ms: u64 },
    /// Did NOT return to baseline even after escalation — the device is polluted;
    /// the caller MUST refuse to launch onto it.
    Stuck,
}

impl ReleaseOutcome {
    /// Whether the device is verified clean (cleanly or after escalation).
    pub fn is_released(&self) -> bool {
        !matches!(self, ReleaseOutcome::Stuck)
    }

    /// Whether escalation (force-kill) was required.
    pub fn escalated(&self) -> bool {
        matches!(self, ReleaseOutcome::Escalated { .. })
    }
}

/// Poll the device until its in-use counter falls to `baseline + tolerance`,
/// returning the elapsed ms on success or `None` on timeout.
async fn poll_until_released(probe: &dyn DeviceProbe, cfg: &ReleaseConfig) -> Option<u64> {
    let start = Instant::now();
    let target = cfg.baseline_gb + cfg.tolerance_gb;
    loop {
        if let Some(in_use) = probe.in_use_gb().await {
            if in_use <= target {
                return Some(start.elapsed().as_millis() as u64);
            }
        }
        if start.elapsed() >= cfg.timeout {
            return None;
        }
        tokio::time::sleep(cfg.poll_interval).await;
    }
}

/// Verify the device returned to baseline after a teardown.
///
/// 1. Poll to baseline within `timeout` → [`ReleaseOutcome::Clean`].
/// 2. On timeout, if an orphan is present → force-kill, re-poll → [`Escalated`].
/// 3. Still not released (or the kill failed) → [`Stuck`] (caller refuses launch).
pub async fn verify_release(probe: &dyn DeviceProbe, cfg: &ReleaseConfig) -> ReleaseOutcome {
    if let Some(ms) = poll_until_released(probe, cfg).await {
        return ReleaseOutcome::Clean { baseline_ms: ms };
    }
    // Timeout. Escalate only if something is actually holding the device.
    if probe.orphan_present().await && probe.force_kill_orphan().await.is_ok() {
        if let Some(ms) = poll_until_released(probe, cfg).await {
            return ReleaseOutcome::Escalated { baseline_ms: ms };
        }
    }
    ReleaseOutcome::Stuck
}

/// Production [`DeviceProbe`]: in-use = total VRAM − free VRAM (both via the SRV-11
/// config sysfs helpers); orphan detection + force-kill are wired by the binary
/// (process-table inspection), so the default impl reports no orphan. Kept thin —
/// the heavy process glue lives at the binary edge, not in this pure module.
pub struct SysfsDeviceProbe;

#[async_trait]
impl DeviceProbe for SysfsDeviceProbe {
    async fn in_use_gb(&self) -> Option<f64> {
        let free = crate::config::read_free_vram_gb()?;
        // total via the SRV-11 total-VRAM helper; in-use = total − free.
        let total = crate::config::read_substrate_info().map(|s| s.vram_carveout_gb)?;
        Some((total - free).max(0.0))
    }
    async fn orphan_present(&self) -> bool {
        false
    }
    async fn force_kill_orphan(&self) -> Result<(), String> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// A scripted device: a queue of in-use readings consumed per poll, an orphan
    /// flag, and whether force-kill succeeds + what the post-kill reading becomes.
    struct ScriptedDevice {
        readings: Mutex<Vec<Option<f64>>>,
        last: Mutex<Option<f64>>,
        orphan: Mutex<bool>,
        kill_ok: bool,
        post_kill_in_use: Option<f64>,
        kills: Mutex<u32>,
    }
    impl ScriptedDevice {
        fn new(readings: Vec<Option<f64>>) -> Self {
            ScriptedDevice {
                readings: Mutex::new(readings),
                last: Mutex::new(Some(99.0)),
                orphan: Mutex::new(false),
                kill_ok: true,
                post_kill_in_use: None,
                kills: Mutex::new(0),
            }
        }
        fn with_orphan(mut self, post_kill_in_use: Option<f64>, kill_ok: bool) -> Self {
            *self.orphan.lock().unwrap() = true;
            self.post_kill_in_use = post_kill_in_use;
            self.kill_ok = kill_ok;
            self
        }
    }
    #[async_trait]
    impl DeviceProbe for ScriptedDevice {
        async fn in_use_gb(&self) -> Option<f64> {
            let mut q = self.readings.lock().unwrap();
            let v = if q.is_empty() {
                *self.last.lock().unwrap()
            } else {
                q.remove(0)
            };
            *self.last.lock().unwrap() = v;
            v
        }
        async fn orphan_present(&self) -> bool {
            *self.orphan.lock().unwrap()
        }
        async fn force_kill_orphan(&self) -> Result<(), String> {
            *self.kills.lock().unwrap() += 1;
            if !self.kill_ok {
                return Err("kill failed".into());
            }
            // After the kill the device releases to the configured post-kill value.
            *self.last.lock().unwrap() = self.post_kill_in_use;
            self.readings.lock().unwrap().clear();
            Ok(())
        }
    }

    fn cfg() -> ReleaseConfig {
        ReleaseConfig {
            baseline_gb: 0.2,
            tolerance_gb: 0.3, // released ⇔ in_use <= 0.5
            timeout: Duration::from_millis(60),
            poll_interval: Duration::from_millis(5),
        }
    }

    #[tokio::test]
    async fn clean_release_when_counter_drops() {
        // First poll still high, then it falls to baseline.
        let dev = ScriptedDevice::new(vec![Some(30.0), Some(10.0), Some(0.1)]);
        let out = verify_release(&dev, &cfg()).await;
        assert!(matches!(out, ReleaseOutcome::Clean { .. }));
        assert!(out.is_released());
        assert!(!out.escalated());
        assert_eq!(*dev.kills.lock().unwrap(), 0, "no escalation on a clean release");
    }

    #[tokio::test]
    async fn escalates_force_kill_then_releases() {
        // Counter never drops on its own (stays high through timeout); an orphan is
        // present; force-kill drops it to baseline.
        let dev = ScriptedDevice::new(vec![Some(30.0); 1]).with_orphan(Some(0.1), true);
        let out = verify_release(&dev, &cfg()).await;
        assert!(matches!(out, ReleaseOutcome::Escalated { .. }));
        assert!(out.escalated());
        assert_eq!(*dev.kills.lock().unwrap(), 1, "force-kill fired exactly once");
    }

    #[tokio::test]
    async fn stuck_when_orphan_kill_does_not_release() {
        // Orphan present, force-kill "succeeds" but the device STILL doesn't drop.
        let dev = ScriptedDevice::new(vec![Some(30.0)]).with_orphan(Some(30.0), true);
        let out = verify_release(&dev, &cfg()).await;
        assert_eq!(out, ReleaseOutcome::Stuck);
        assert!(!out.is_released());
    }

    #[tokio::test]
    async fn stuck_when_no_orphan_and_never_releases() {
        // Counter stays high, nothing to escalate → Stuck (don't launch).
        let dev = ScriptedDevice::new(vec![Some(40.0)]);
        let out = verify_release(&dev, &cfg()).await;
        assert_eq!(out, ReleaseOutcome::Stuck);
    }

    #[tokio::test]
    async fn unreadable_counter_is_not_released() {
        // None readings (unreadable) never count as released → Stuck (fail-safe).
        let dev = ScriptedDevice::new(vec![None]);
        let out = verify_release(&dev, &cfg()).await;
        assert_eq!(out, ReleaseOutcome::Stuck);
    }
}
