//! Snapshot types: the full JSON shape written to the log and served by
//! `GET /v1/sweep/status` / `GET /v1/sweep/status/history`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::db::SweepDbStats;
use super::ollama::OllamaPsStats;
use super::verdict::Verdict;

/// One sweep's (coder or assistant) full status for a single tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepStatus {
    /// `"coder"` or `"assistant"`.
    pub name: String,
    /// The systemd unit checked for this sweep.
    pub service_unit: String,
    /// `Some(true)`/`Some(false)` from `systemctl is-active`; `None` if
    /// systemctl itself could not be run.
    pub service_active: Option<bool>,
    #[serde(flatten)]
    pub db: SweepDbStats,
    pub verdict: Verdict,
}

/// The full snapshot for one poll tick: both sweeps, host-level signals, and
/// the rolled-up overall verdict. This exact shape is what gets appended to
/// the JSONL log and returned by the status endpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepStatusSnapshot {
    pub timestamp: DateTime<Utc>,
    pub coder: SweepStatus,
    /// `None` only if the assistant table/sweep were not configured to be
    /// checked at all (not currently the case — both are always checked; kept
    /// `Option` so a future deploy that drops one sweep degrades cleanly).
    pub assistant: Option<SweepStatus>,
    pub gpu_busy_percent: Option<f64>,
    pub ollama: OllamaPsStats,
    /// `false` when no intake DB URL resolved at all (`INTAKE_DATABASE_URL` /
    /// `DATABASE_URL` both unset) — both sweeps' `db.available` will also be
    /// `false` in that case, but this flag makes "never configured" vs
    /// "configured but briefly unreachable" distinguishable at a glance.
    pub db_configured: bool,
    pub overall_verdict: Verdict,
}

impl SweepStatusSnapshot {
    /// Compute the overall verdict from the two per-sweep verdicts (or just
    /// the coder verdict when assistant wasn't checked).
    pub fn compute_overall(coder: Verdict, assistant: Option<Verdict>) -> Verdict {
        let mut verdicts = vec![coder];
        if let Some(v) = assistant {
            verdicts.push(v);
        }
        super::verdict::overall_verdict(&verdicts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_overall_with_both_sweeps() {
        assert_eq!(
            SweepStatusSnapshot::compute_overall(Verdict::Working, Some(Verdict::Stuck)),
            Verdict::Stuck
        );
        assert_eq!(
            SweepStatusSnapshot::compute_overall(Verdict::Working, Some(Verdict::Idle)),
            Verdict::Working
        );
    }

    #[test]
    fn compute_overall_coder_only() {
        assert_eq!(SweepStatusSnapshot::compute_overall(Verdict::Idle, None), Verdict::Idle);
        assert_eq!(SweepStatusSnapshot::compute_overall(Verdict::Working, None), Verdict::Working);
    }
}
