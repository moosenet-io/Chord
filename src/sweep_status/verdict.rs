//! Pure verdict-computation logic for the sweep-status monitor.
//!
//! This is the judgment core of the whole subsystem and the only part of it
//! worth unit-testing heavily (the DB/HTTP/systemd plumbing around it is
//! thin and best tested end-to-end). Kept dependency-free (no sqlx/reqwest)
//! so it can be exercised with synthetic inputs.
//!
//! Heuristic mirrors the host-level watchdog that already auto-restarts
//! `ollama.service` on this exact failure mode (gfx1151 GPU-MoE wedge):
//! a sweep is `Stuck` when its systemd unit is active, the GPU is pegged
//! (>= threshold busy%), AND the newest DB row is older than the stuck-age
//! threshold. Anything else, while the unit is active, is `Working`. A unit
//! that is not active is always `Idle` — regardless of what the DB/GPU say —
//! because "no process running" is definitionally not "stuck mid-generate".

use serde::{Deserialize, Serialize};

/// Default "no fresh row" age threshold (seconds) — mirrors the host watchdog's
/// 6-minute trigger.
pub const DEFAULT_STUCK_AGE_SECS: i64 = 360;

/// Default GPU-busy threshold (percent) — mirrors the host watchdog's 70%.
pub const DEFAULT_GPU_BUSY_THRESHOLD_PERCENT: f64 = 70.0;

/// Health verdict for a single sweep, or the fleet-wide overall verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    /// Producing rows normally (or at least not showing the stuck signature).
    Working,
    /// GPU pegged + no fresh rows + service active: the gfx1151 GPU-MoE-wedge
    /// signature. This is the "go look at it" state.
    Stuck,
    /// The sweep's systemd unit is not active — nothing running, nothing to
    /// judge as stuck.
    Idle,
}

impl Verdict {
    pub fn as_str(&self) -> &'static str {
        match self {
            Verdict::Working => "working",
            Verdict::Stuck => "stuck",
            Verdict::Idle => "idle",
        }
    }
}

/// Compute the verdict for one sweep from concrete (non-optional) signals.
///
/// `service_active = false` always yields `Idle`, independent of the other
/// two signals — an inactive unit cannot be "stuck mid-generate".
///
/// Otherwise: `Stuck` iff `gpu_busy_percent >= gpu_busy_threshold` AND
/// `latest_row_age_secs > stuck_age_secs`. Both conditions must hold — a
/// busy GPU alone is normal (it's supposed to be busy while working), and an
/// old row alone is normal for a sweep between test cases if the GPU isn't
/// pegged (it may just be doing housekeeping / between-model swaps).
pub fn compute_verdict(
    service_active: bool,
    latest_row_age_secs: i64,
    gpu_busy_percent: f64,
    stuck_age_secs: i64,
    gpu_busy_threshold: f64,
) -> Verdict {
    if !service_active {
        return Verdict::Idle;
    }
    if gpu_busy_percent >= gpu_busy_threshold && latest_row_age_secs > stuck_age_secs {
        Verdict::Stuck
    } else {
        Verdict::Working
    }
}

/// `compute_verdict` variant for the real-world case where a signal may be
/// unavailable (DB unreachable → no row-age; sysfs read failed → no GPU busy%).
///
/// Missing `latest_row_age_secs` is mapped to `i64::MAX` (an unknown last-row
/// time is, if anything, WORSE than a known-old one — we'd rather flag a
/// possible stuck sweep than silently report "working" while blind). Missing
/// `gpu_busy_percent` is mapped to `0.0` (we cannot confirm the GPU is pegged,
/// so a single missing GPU reading does not by itself manufacture a `Stuck`
/// verdict — it takes the busy signal to trigger `Stuck` at all).
///
/// `service_active: None` (systemctl itself unavailable/erroring) is treated
/// as `false` (`Idle`) — the conservative direction: we never claim a sweep
/// is `Stuck` when we could not even confirm its unit is running.
pub fn compute_verdict_optional(
    service_active: Option<bool>,
    latest_row_age_secs: Option<i64>,
    gpu_busy_percent: Option<f64>,
    stuck_age_secs: i64,
    gpu_busy_threshold: f64,
) -> Verdict {
    compute_verdict(
        service_active.unwrap_or(false),
        latest_row_age_secs.unwrap_or(i64::MAX),
        gpu_busy_percent.unwrap_or(0.0),
        stuck_age_secs,
        gpu_busy_threshold,
    )
}

/// Roll up per-sweep verdicts into one overall verdict:
/// - any `Stuck` → overall `Stuck` (a single wedged sweep is worth surfacing
///   even if another sweep is healthy).
/// - else, all `Idle` (including the degenerate empty-list case — nothing to
///   report as working) → overall `Idle`.
/// - else → `Working`.
pub fn overall_verdict(verdicts: &[Verdict]) -> Verdict {
    if verdicts.iter().any(|v| *v == Verdict::Stuck) {
        return Verdict::Stuck;
    }
    if verdicts.iter().all(|v| *v == Verdict::Idle) {
        return Verdict::Idle;
    }
    Verdict::Working
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── compute_verdict: core heuristic ────────────────────────────────────

    #[test]
    fn inactive_service_is_always_idle() {
        // Even with a pegged GPU and an ancient row, an inactive unit is Idle.
        assert_eq!(
            compute_verdict(false, 999_999, 100.0, DEFAULT_STUCK_AGE_SECS, DEFAULT_GPU_BUSY_THRESHOLD_PERCENT),
            Verdict::Idle
        );
        assert_eq!(
            compute_verdict(false, 0, 0.0, DEFAULT_STUCK_AGE_SECS, DEFAULT_GPU_BUSY_THRESHOLD_PERCENT),
            Verdict::Idle
        );
    }

    #[test]
    fn active_busy_old_is_stuck() {
        // The real incident: GPU pegged 99%, zero rows for 7 hours, service active.
        assert_eq!(
            compute_verdict(true, 7 * 3600, 99.0, DEFAULT_STUCK_AGE_SECS, DEFAULT_GPU_BUSY_THRESHOLD_PERCENT),
            Verdict::Stuck
        );
    }

    #[test]
    fn active_busy_fresh_is_working() {
        // GPU busy (expected mid-generate), but a row landed 30s ago — normal.
        assert_eq!(
            compute_verdict(true, 30, 95.0, DEFAULT_STUCK_AGE_SECS, DEFAULT_GPU_BUSY_THRESHOLD_PERCENT),
            Verdict::Working
        );
    }

    #[test]
    fn active_idle_gpu_old_row_is_working() {
        // Old row but GPU not busy: sweep is between test cases / doing
        // housekeeping (swapping models), not wedged.
        assert_eq!(
            compute_verdict(true, 3600, 5.0, DEFAULT_STUCK_AGE_SECS, DEFAULT_GPU_BUSY_THRESHOLD_PERCENT),
            Verdict::Working
        );
    }

    #[test]
    fn active_idle_gpu_fresh_row_is_working() {
        assert_eq!(
            compute_verdict(true, 10, 2.0, DEFAULT_STUCK_AGE_SECS, DEFAULT_GPU_BUSY_THRESHOLD_PERCENT),
            Verdict::Working
        );
    }

    #[test]
    fn boundary_age_exactly_at_threshold_is_not_stuck() {
        // age > threshold is required (strictly greater), so age == threshold
        // is still Working (with GPU busy).
        assert_eq!(
            compute_verdict(true, DEFAULT_STUCK_AGE_SECS, 90.0, DEFAULT_STUCK_AGE_SECS, DEFAULT_GPU_BUSY_THRESHOLD_PERCENT),
            Verdict::Working
        );
        assert_eq!(
            compute_verdict(true, DEFAULT_STUCK_AGE_SECS + 1, 90.0, DEFAULT_STUCK_AGE_SECS, DEFAULT_GPU_BUSY_THRESHOLD_PERCENT),
            Verdict::Stuck
        );
    }

    #[test]
    fn boundary_gpu_busy_exactly_at_threshold_is_stuck() {
        // gpu_busy >= threshold is inclusive.
        assert_eq!(
            compute_verdict(true, 999, DEFAULT_GPU_BUSY_THRESHOLD_PERCENT, DEFAULT_STUCK_AGE_SECS, DEFAULT_GPU_BUSY_THRESHOLD_PERCENT),
            Verdict::Stuck
        );
        assert_eq!(
            compute_verdict(true, 999, DEFAULT_GPU_BUSY_THRESHOLD_PERCENT - 0.1, DEFAULT_STUCK_AGE_SECS, DEFAULT_GPU_BUSY_THRESHOLD_PERCENT),
            Verdict::Working
        );
    }

    #[test]
    fn custom_thresholds_are_honored() {
        // A stricter operator config: 60% busy, 120s age.
        assert_eq!(compute_verdict(true, 121, 61.0, 120, 60.0), Verdict::Stuck);
        assert_eq!(compute_verdict(true, 119, 61.0, 120, 60.0), Verdict::Working);
        assert_eq!(compute_verdict(true, 121, 59.0, 120, 60.0), Verdict::Working);
    }

    // ── compute_verdict_optional: missing-signal handling ──────────────────

    #[test]
    fn optional_missing_service_state_defaults_to_idle() {
        assert_eq!(
            compute_verdict_optional(None, Some(999_999), Some(100.0), DEFAULT_STUCK_AGE_SECS, DEFAULT_GPU_BUSY_THRESHOLD_PERCENT),
            Verdict::Idle
        );
    }

    #[test]
    fn optional_missing_row_age_with_busy_gpu_and_active_service_is_stuck() {
        // Unknown last-row age is treated as "possibly ancient" — worth flagging
        // rather than silently reporting Working while blind to the DB.
        assert_eq!(
            compute_verdict_optional(Some(true), None, Some(85.0), DEFAULT_STUCK_AGE_SECS, DEFAULT_GPU_BUSY_THRESHOLD_PERCENT),
            Verdict::Stuck
        );
    }

    #[test]
    fn optional_missing_gpu_reading_alone_does_not_manufacture_stuck() {
        assert_eq!(
            compute_verdict_optional(Some(true), Some(999_999), None, DEFAULT_STUCK_AGE_SECS, DEFAULT_GPU_BUSY_THRESHOLD_PERCENT),
            Verdict::Working
        );
    }

    #[test]
    fn optional_all_missing_but_active_is_working_not_stuck() {
        assert_eq!(
            compute_verdict_optional(Some(true), None, None, DEFAULT_STUCK_AGE_SECS, DEFAULT_GPU_BUSY_THRESHOLD_PERCENT),
            Verdict::Working
        );
    }

    // ── overall_verdict roll-up ─────────────────────────────────────────────

    #[test]
    fn overall_any_stuck_wins() {
        assert_eq!(overall_verdict(&[Verdict::Working, Verdict::Stuck]), Verdict::Stuck);
        assert_eq!(overall_verdict(&[Verdict::Idle, Verdict::Stuck]), Verdict::Stuck);
    }

    #[test]
    fn overall_all_idle_is_idle() {
        assert_eq!(overall_verdict(&[Verdict::Idle, Verdict::Idle]), Verdict::Idle);
        assert_eq!(overall_verdict(&[Verdict::Idle]), Verdict::Idle);
        assert_eq!(overall_verdict(&[]), Verdict::Idle);
    }

    #[test]
    fn overall_mixed_working_idle_is_working() {
        // One sweep configured and healthy, the other's unit simply isn't
        // deployed/active — the fleet overall is still doing useful work.
        assert_eq!(overall_verdict(&[Verdict::Working, Verdict::Idle]), Verdict::Working);
    }

    #[test]
    fn overall_all_working_is_working() {
        assert_eq!(overall_verdict(&[Verdict::Working, Verdict::Working]), Verdict::Working);
    }

    #[test]
    fn verdict_as_str() {
        assert_eq!(Verdict::Working.as_str(), "working");
        assert_eq!(Verdict::Stuck.as_str(), "stuck");
        assert_eq!(Verdict::Idle.as_str(), "idle");
    }

    #[test]
    fn verdict_serializes_lowercase() {
        assert_eq!(serde_json::to_string(&Verdict::Stuck).unwrap(), "\"stuck\"");
        assert_eq!(serde_json::to_string(&Verdict::Working).unwrap(), "\"working\"");
        assert_eq!(serde_json::to_string(&Verdict::Idle).unwrap(), "\"idle\"");
    }
}
