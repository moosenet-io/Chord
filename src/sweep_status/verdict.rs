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

/// Default bound (seconds) on the "empty table, service active" start-up
/// grace period — matches [`DEFAULT_STUCK_AGE_SECS`] (6 min). Past this
/// window, a service that's active with an unreachable-yet-empty table is
/// no longer given an unconditional `Working` — see
/// [`compute_verdict_optional`].
pub const DEFAULT_STARTUP_GRACE_SECS: i64 = 360;

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
    /// A signal required to judge health is unavailable (systemctl couldn't
    /// run, or the DB itself is unreachable) — we cannot confidently say
    /// `Working`, `Stuck`, or `Idle`, so we say so rather than guessing in
    /// either direction.
    Unknown,
}

impl Verdict {
    pub fn as_str(&self) -> &'static str {
        match self {
            Verdict::Working => "working",
            Verdict::Stuck => "stuck",
            Verdict::Idle => "idle",
            Verdict::Unknown => "unknown",
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
/// unavailable (DB unreachable → no row-age; sysfs read failed → no GPU
/// busy%; systemctl unrunnable → no service-active reading). Availability is
/// threaded through explicitly (`db_available`, plus the `Option`-ness of
/// `service_active`/`gpu_busy_percent`) rather than collapsed into a
/// concrete-but-misleading number before this call, so a missing signal can
/// never silently masquerade as a confident `Working` or `Stuck`.
///
/// Precedence (first match wins):
/// 1. `service_active == Some(false)` → always `Idle`, regardless of any
///    other signal — an inactive unit cannot be "stuck mid-generate".
/// 2. `service_active == None` (systemctl itself unavailable) → `Unknown`.
///    An active-but-unobservable sweep must not be reported as `Idle`.
/// 3. `db_available == false` (DB unreachable, not merely empty) →
///    `Unknown`. We have no row-age signal at all here and cannot rule out
///    `Stuck` — surface the ambiguity instead of forcing a confident verdict
///    in either direction.
/// 4. `db_available == true` but `latest_row_age_secs == None` → the table
///    is reachable and simply has zero rows yet (e.g. a sweep that just
///    started, or `assistant_profile_run` before it's accumulated history).
///    This is a legitimate start-up state, not a failure — graced as
///    `Working`, but ONLY for up to `startup_grace_secs` of continuous
///    empty-table observation (`empty_table_elapsed_secs`, tracked by the
///    caller — see `sweep_status::poll::EmptyTableTracker` — since this
///    pure function has no notion of time passing between calls). Past that
///    window a sweep that is active, has a reachable-but-still-empty table,
///    and has GPU pegged the whole time IS the exact stuck signature this
///    subsystem exists to catch (active + GPU busy + no fresh output for
///    longer than the stuck-age threshold), so that's `Stuck`. If the GPU
///    reading is unavailable past the grace window there's no evidence
///    either way, so that's `Unknown` rather than a confident guess in
///    either direction — "forever confidently Working" (the bug this fixes)
///    is wrong, but so would be an unconditional `Stuck` with no GPU signal
///    to back it up.
/// 5. Otherwise we have a real row age. If `gpu_busy_percent` is known,
///    apply the normal `Stuck` heuristic (busy GPU + stale row). If the GPU
///    reading is missing: a stale row plus an unknown GPU state means we
///    cannot confirm health, so that's `Unknown` — but a *fresh* row is
///    `Working` regardless of the GPU reading (nothing to be suspicious of
///    yet).
#[allow(clippy::too_many_arguments)]
pub fn compute_verdict_optional(
    service_active: Option<bool>,
    db_available: bool,
    latest_row_age_secs: Option<i64>,
    gpu_busy_percent: Option<f64>,
    stuck_age_secs: i64,
    gpu_busy_threshold: f64,
    empty_table_elapsed_secs: Option<i64>,
    startup_grace_secs: i64,
) -> Verdict {
    if service_active == Some(false) {
        return Verdict::Idle;
    }
    if service_active.is_none() {
        return Verdict::Unknown;
    }
    // service_active == Some(true) from here on.

    if !db_available {
        return Verdict::Unknown;
    }

    let age = match latest_row_age_secs {
        Some(age) => age,
        // DB available, table empty (never populated / just started): a
        // *bounded* start-up grace period, not an unconditional Working
        // forever. `empty_table_elapsed_secs` is how long the caller has
        // continuously observed this exact state; `None` means "just
        // arrived at it" (treated the same as 0 elapsed — still within any
        // positive grace window).
        None => {
            let elapsed = empty_table_elapsed_secs.unwrap_or(0);
            return if elapsed <= startup_grace_secs {
                Verdict::Working
            } else {
                match gpu_busy_percent {
                    // Active + GPU pegged + no rows for longer than the
                    // grace window: the stuck signature, just observed via
                    // "zero rows ever" instead of "a stale row".
                    Some(gpu) if gpu >= gpu_busy_threshold => Verdict::Stuck,
                    // GPU not pegged, or unobservable: no positive evidence
                    // of a wedge, but "confidently Working" is no longer
                    // honest either.
                    _ => Verdict::Unknown,
                }
            };
        }
    };

    match gpu_busy_percent {
        Some(gpu) if gpu >= gpu_busy_threshold && age > stuck_age_secs => Verdict::Stuck,
        Some(_) => Verdict::Working,
        None if age > stuck_age_secs => Verdict::Unknown,
        None => Verdict::Working,
    }
}

/// Roll up per-sweep verdicts into one overall verdict:
/// - any `Stuck` → overall `Stuck` (a single wedged sweep is worth surfacing
///   even if another sweep is healthy — this is the "go look at it now"
///   signal and must not be masked by an unrelated `Unknown`).
/// - else, any `Unknown` → overall `Unknown` (we couldn't confirm every
///   sweep's health; that's worth surfacing over a false "all clear").
/// - else, all `Idle` (including the degenerate empty-list case — nothing to
///   report as working) → overall `Idle`.
/// - else → `Working`.
pub fn overall_verdict(verdicts: &[Verdict]) -> Verdict {
    if verdicts.iter().any(|v| *v == Verdict::Stuck) {
        return Verdict::Stuck;
    }
    if verdicts.iter().any(|v| *v == Verdict::Unknown) {
        return Verdict::Unknown;
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

    // Helper for brevity: default thresholds, and `empty_table_elapsed_secs
    // = None` (i.e. "just arrived at empty-table state" / not applicable to
    // the test) since most tests below don't exercise the empty-table
    // branch at all. Tests that specifically exercise the bounded grace
    // period use `cvo_grace` instead.
    fn cvo(
        service_active: Option<bool>,
        db_available: bool,
        latest_row_age_secs: Option<i64>,
        gpu_busy_percent: Option<f64>,
    ) -> Verdict {
        compute_verdict_optional(
            service_active,
            db_available,
            latest_row_age_secs,
            gpu_busy_percent,
            DEFAULT_STUCK_AGE_SECS,
            DEFAULT_GPU_BUSY_THRESHOLD_PERCENT,
            None,
            DEFAULT_STARTUP_GRACE_SECS,
        )
    }

    // Helper for the bounded-startup-grace tests: same defaults as `cvo`,
    // but with an explicit `empty_table_elapsed_secs` so tests can place
    // themselves inside or outside the grace window.
    fn cvo_grace(gpu_busy_percent: Option<f64>, empty_table_elapsed_secs: Option<i64>) -> Verdict {
        compute_verdict_optional(
            Some(true),
            true,
            None,
            gpu_busy_percent,
            DEFAULT_STUCK_AGE_SECS,
            DEFAULT_GPU_BUSY_THRESHOLD_PERCENT,
            empty_table_elapsed_secs,
            DEFAULT_STARTUP_GRACE_SECS,
        )
    }

    #[test]
    fn optional_service_inactive_is_always_idle_even_with_bad_signals() {
        // service_active explicitly false wins over everything, including a
        // DB that's unreachable.
        assert_eq!(cvo(Some(false), false, None, None), Verdict::Idle);
        assert_eq!(cvo(Some(false), true, Some(999_999), Some(100.0)), Verdict::Idle);
    }

    #[test]
    fn optional_systemctl_unavailable_is_unknown_not_idle() {
        // systemctl itself couldn't be run: we never confirmed the unit is
        // inactive, so this must NOT collapse to Idle (that would hide an
        // active-but-unobservable sweep).
        assert_eq!(cvo(None, true, Some(30), Some(10.0)), Verdict::Unknown);
        assert_eq!(cvo(None, false, None, None), Verdict::Unknown);
    }

    #[test]
    fn optional_db_unreachable_is_unknown_not_stuck() {
        // Active service, but the DB itself is unreachable (not just an
        // empty table) — we have no row-age signal and must not force a
        // confident Stuck (or Working) from missing data alone, even with a
        // pegged GPU.
        assert_eq!(cvo(Some(true), false, None, Some(99.0)), Verdict::Unknown);
        assert_eq!(cvo(Some(true), false, None, None), Verdict::Unknown);
    }

    #[test]
    fn optional_db_available_empty_table_is_startup_grace_working() {
        // DB reachable, table simply has zero rows yet (fresh/new sweep, or
        // assistant_profile_run before it's accumulated history) — this is a
        // legitimate start-up state, not Stuck/Unknown, even with the GPU
        // pegged, as long as we're within the grace window (here: just
        // arrived at the state, `empty_table_elapsed_secs = None`).
        assert_eq!(cvo(Some(true), true, None, Some(99.0)), Verdict::Working);
        assert_eq!(cvo(Some(true), true, None, None), Verdict::Working);
    }

    // ── compute_verdict_optional: bounded start-up grace period ────────────
    // (regression coverage for the "Working forever" bug: an empty table
    // with the service active used to grant Working unconditionally, with
    // no time bound, so a sweep that wedges before its very first row could
    // never be flagged.)

    #[test]
    fn grace_within_window_is_working_even_with_gpu_pegged() {
        // Well inside the grace window (well under DEFAULT_STARTUP_GRACE_SECS):
        // still a legitimate start-up state regardless of GPU reading.
        assert_eq!(cvo_grace(Some(99.0), Some(0)), Verdict::Working);
        assert_eq!(cvo_grace(Some(99.0), Some(DEFAULT_STARTUP_GRACE_SECS / 2)), Verdict::Working);
        assert_eq!(cvo_grace(None, Some(DEFAULT_STARTUP_GRACE_SECS / 2)), Verdict::Working);
    }

    #[test]
    fn grace_boundary_exactly_at_window_is_still_working() {
        // elapsed > grace is required (strictly greater), matching the
        // stuck-age boundary convention elsewhere in this module.
        assert_eq!(cvo_grace(Some(99.0), Some(DEFAULT_STARTUP_GRACE_SECS)), Verdict::Working);
    }

    #[test]
    fn grace_past_window_with_gpu_pegged_is_stuck() {
        // Past the grace window, active, empty table, AND the GPU has been
        // pegged >= threshold the whole time: this is the exact stuck
        // signature the whole subsystem exists to catch (active + GPU busy
        // + no fresh output for longer than the tolerated window) — just
        // observed via "zero rows ever" rather than "a stale row present".
        // Chosen deliberately over `Unknown`: GPU-pegged-and-silent past the
        // grace window is positive evidence of a wedge, not merely an
        // unobservable signal, so reporting `Unknown` here would under-alert
        // on a real incident of exactly this shape.
        assert_eq!(
            cvo_grace(Some(DEFAULT_GPU_BUSY_THRESHOLD_PERCENT), Some(DEFAULT_STARTUP_GRACE_SECS + 1)),
            Verdict::Stuck
        );
        assert_eq!(cvo_grace(Some(99.0), Some(DEFAULT_STARTUP_GRACE_SECS + 3600)), Verdict::Stuck);
    }

    #[test]
    fn grace_past_window_with_gpu_not_pegged_is_unknown() {
        // Past the grace window, but the GPU is NOT pegged: no positive
        // evidence of a wedge (a sweep can legitimately sit idle-ish between
        // test cases with an empty table if it just started slowly), so
        // this is `Unknown` rather than a confident `Working` (which is the
        // bug being fixed) or a confident `Stuck` (which isn't justified
        // without the GPU signal backing it up).
        assert_eq!(cvo_grace(Some(5.0), Some(DEFAULT_STARTUP_GRACE_SECS + 1)), Verdict::Unknown);
    }

    #[test]
    fn grace_past_window_with_gpu_unknown_is_unknown() {
        // Past the grace window and the GPU reading itself is unavailable:
        // no evidence either way, so `Unknown` — matches the existing
        // "gpu missing + stale row -> Unknown" precedent elsewhere in this
        // function rather than inventing a different rule for this branch.
        assert_eq!(cvo_grace(None, Some(DEFAULT_STARTUP_GRACE_SECS + 1)), Verdict::Unknown);
    }

    #[test]
    fn optional_gpu_missing_with_stale_row_is_unknown() {
        // Row is stale and we can't read the GPU: an unobservable GPU must
        // not "paper over" a stale DB with a confident Working.
        assert_eq!(
            cvo(Some(true), true, Some(DEFAULT_STUCK_AGE_SECS + 1), None),
            Verdict::Unknown
        );
    }

    #[test]
    fn optional_gpu_missing_with_fresh_row_is_working() {
        // Row is fresh — nothing suspicious yet, regardless of the missing
        // GPU reading.
        assert_eq!(cvo(Some(true), true, Some(10), None), Verdict::Working);
    }

    #[test]
    fn optional_all_signals_present_uses_normal_heuristic() {
        assert_eq!(
            cvo(Some(true), true, Some(7 * 3600), Some(99.0)),
            Verdict::Stuck
        );
        assert_eq!(cvo(Some(true), true, Some(30), Some(95.0)), Verdict::Working);
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
    fn overall_stuck_wins_over_unknown() {
        assert_eq!(overall_verdict(&[Verdict::Unknown, Verdict::Stuck]), Verdict::Stuck);
    }

    #[test]
    fn overall_unknown_wins_over_idle_and_working() {
        assert_eq!(overall_verdict(&[Verdict::Unknown, Verdict::Idle]), Verdict::Unknown);
        assert_eq!(overall_verdict(&[Verdict::Unknown, Verdict::Working]), Verdict::Unknown);
        assert_eq!(overall_verdict(&[Verdict::Unknown]), Verdict::Unknown);
    }

    #[test]
    fn verdict_as_str() {
        assert_eq!(Verdict::Working.as_str(), "working");
        assert_eq!(Verdict::Stuck.as_str(), "stuck");
        assert_eq!(Verdict::Idle.as_str(), "idle");
        assert_eq!(Verdict::Unknown.as_str(), "unknown");
    }

    #[test]
    fn verdict_serializes_lowercase() {
        assert_eq!(serde_json::to_string(&Verdict::Stuck).unwrap(), "\"stuck\"");
        assert_eq!(serde_json::to_string(&Verdict::Working).unwrap(), "\"working\"");
        assert_eq!(serde_json::to_string(&Verdict::Idle).unwrap(), "\"idle\"");
        assert_eq!(serde_json::to_string(&Verdict::Unknown).unwrap(), "\"unknown\"");
    }
}
