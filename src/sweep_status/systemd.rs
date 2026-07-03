//! systemd unit-state check via `systemctl is-active`, using
//! `tokio::process::Command` (async — this runs inside the sweep-monitor tick,
//! never the blocking `std::process::Command`).

/// The `ActiveState` values `systemctl is-active` prints when it *did*
/// successfully query the unit and the answer is "not active". These are
/// systemd's own well-known state words (see `systemctl(1)`); seeing one of
/// them on stdout means the query was answered, just answered "no".
const KNOWN_NOT_ACTIVE_STATES: &[&str] =
    &["inactive", "failed", "unknown", "activating", "deactivating", "reloading"];

/// Interpret the raw stdout of `systemctl is-active <unit>` (exit code is
/// intentionally NOT consulted — see module docs) into a confirmed answer.
///
/// Returns:
/// - `Some(true)` — stdout was exactly `active`.
/// - `Some(false)` — stdout was one of [`KNOWN_NOT_ACTIVE_STATES`]: the
///   query was genuinely answered, just answered "not active".
/// - `None` — anything else, including empty stdout. This is the case a
///   real sandbox hit: `systemctl is-active` exits 1 with a bus-access
///   failure and *empty* stdout — systemctl never determined the unit's
///   state at all. That must NOT be conflated with a confirmed "inactive"
///   (which is what naively treating "stdout != active" as `Some(false)`
///   did before this fix).
fn classify_is_active_output(stdout: &str) -> Option<bool> {
    let trimmed = stdout.trim();
    if trimmed == "active" {
        Some(true)
    } else if KNOWN_NOT_ACTIVE_STATES.contains(&trimmed) {
        Some(false)
    } else {
        None
    }
}

/// Check whether a systemd unit is active.
///
/// Returns:
/// - `Some(true)` — `systemctl is-active <unit>` printed `active`.
/// - `Some(false)` — it printed one of a small set of well-known "queried,
///   not active" state words (`inactive`, `failed`, `unknown`,
///   `activating`, `deactivating`, `reloading`). `systemctl is-active`
///   exits non-zero for all of these; we still read stdout rather than
///   trusting the exit code.
/// - `None` — either `systemctl` itself could not be run (missing binary,
///   spawn error), OR it ran but produced empty/unrecognized stdout — e.g.
///   a bus-access failure that exits non-zero with nothing on stdout. In
///   both cases systemctl did NOT determine the unit's state. Callers must
///   treat this as "cannot confirm", not "inactive" —
///   [`crate::sweep_status::verdict::compute_verdict_optional`] maps `None`
///   to `Verdict::Unknown` rather than assuming `Idle` (which would hide an
///   active-but-unobservable sweep) or `Stuck`.
pub async fn is_unit_active(unit: &str) -> Option<bool> {
    match tokio::process::Command::new("systemctl")
        .args(["is-active", unit])
        .output()
        .await
    {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let result = classify_is_active_output(&stdout);
            if result.is_none() {
                tracing::warn!(
                    target: "chord.sweep_status", unit, stdout = %stdout.trim(),
                    "systemctl is-active returned an unrecognized/empty response — cannot confirm unit state"
                );
            }
            result
        }
        Err(e) => {
            tracing::warn!(target: "chord.sweep_status", unit, error = %e, "systemctl is-active failed to run");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn nonexistent_unit_reports_inactive() {
        // A real systemctl (if present) reports "unknown"/"inactive" for a
        // unit name that doesn't exist — either way, not "active".
        let result = is_unit_active("chord-sweep-status-test-does-not-exist.service").await;
        // Environments without systemctl at all (containers, CI) yield None;
        // environments with it yield Some(false). Both are "not confirmed
        // active", which is what matters for the verdict logic.
        assert_ne!(result, Some(true));
    }

    // ── classify_is_active_output: the actual bug fix ──────────────────────

    #[test]
    fn classify_active_stdout_is_confirmed_true() {
        assert_eq!(classify_is_active_output("active\n"), Some(true));
    }

    #[test]
    fn classify_known_not_active_states_are_confirmed_false() {
        for state in KNOWN_NOT_ACTIVE_STATES {
            assert_eq!(
                classify_is_active_output(state),
                Some(false),
                "state {state:?} should be a confirmed not-active answer"
            );
        }
        // Trailing newline (as systemctl actually emits) must still parse.
        assert_eq!(classify_is_active_output("inactive\n"), Some(false));
    }

    #[test]
    fn classify_empty_stdout_is_unknown_not_confirmed_inactive() {
        // Reproduces the real reviewer-sandbox failure mode: `systemctl
        // is-active` exits 1 (bus access failure) with completely empty
        // stdout. systemctl never determined the unit's state here — this
        // must be `None` (Unknown), not `Some(false)` (confirmed Idle),
        // which is exactly the bug: the old code mapped "stdout != active"
        // straight to `Some(false)`, conflating "confirmed inactive" with
        // "couldn't tell".
        assert_eq!(classify_is_active_output(""), None);
        assert_eq!(classify_is_active_output("   \n"), None);
    }

    #[test]
    fn classify_unrecognized_stdout_is_unknown() {
        // Some other unexpected error text on stdout that isn't one of
        // systemd's own ActiveState words — also "couldn't tell", not
        // "confirmed inactive".
        assert_eq!(classify_is_active_output("Failed to connect to bus: No such file or directory"), None);
    }
}
