//! systemd unit-state check via `systemctl is-active`, using
//! `tokio::process::Command` (async — this runs inside the sweep-monitor tick,
//! never the blocking `std::process::Command`).

/// Check whether a systemd unit is active.
///
/// Returns:
/// - `Some(true)` — `systemctl is-active <unit>` printed `active`.
/// - `Some(false)` — it printed anything else (`inactive`, `failed`,
///   `activating`, `unknown`, ...). `systemctl is-active` exits non-zero for
///   all of these; we still read stdout rather than trusting the exit code.
/// - `None` — `systemctl` itself could not be run (missing binary, spawn
///   error). Callers must treat this as "cannot confirm", not "inactive" —
///   [`crate::sweep_status::verdict::compute_verdict_optional`] already maps
///   `None` to the conservative `Idle` direction rather than risking a false
///   `Stuck`.
pub async fn is_unit_active(unit: &str) -> Option<bool> {
    match tokio::process::Command::new("systemctl")
        .args(["is-active", unit])
        .output()
        .await
    {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            Some(stdout.trim() == "active")
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
}
