//! The fail-closed decision layer between an egress posture and a live network
//! namespace — S88 ISO-02.
//!
//! [`netns::prepare`](super::netns::prepare) does the privileged create/configure;
//! this module owns the POLICY around it that the launcher depends on:
//!
//!   * **isolation default-ON** (gated by `CHORD_NETNS_ISOLATION`, off only when
//!     explicitly `0`),
//!   * **FAIL CLOSED** when the namespace cannot be created (missing
//!     `CAP_NET_ADMIN`, non-Linux, missing tooling): the runtime is NOT launched
//!     with full host egress — the caller gets an error,
//!   * **explicit operator override** (`CHORD_ALLOW_UNISOLATED=1`): loud (`warn`),
//!     off by default; only then may the launcher proceed without a namespace.
//!
//! The decision is expressed as [`IsolationDecision`] so it is unit-testable
//! without a privileged host: the launcher asks for a decision, and only an
//! `Isolated(handle)` carries a namespace to spawn into; `Unisolated` is reachable
//! ONLY via the explicit override; `Refused` is the fail-closed terminal that must
//! abort the launch.

use super::egress_policy::EgressPosture;
use super::netns::{self, NetnsError, NetnsHandle};

/// The outcome of deciding how to isolate a runtime launch.
#[derive(Debug)]
pub enum IsolationDecision {
    /// A network namespace was prepared; spawn the runtime INTO `handle` and tear
    /// it down on swap-out. This is the normal, guaranteed path.
    Isolated(NetnsHandle),
    /// Isolation is disabled by config (`CHORD_NETNS_ISOLATION=0`) — the legacy
    /// non-isolated path, used by unprivileged dev/CI. Distinct from the override:
    /// this is the developer opt-out, not a fail-closed bypass.
    DisabledByConfig,
    /// The namespace could NOT be created AND the operator set the explicit
    /// `CHORD_ALLOW_UNISOLATED=1` override. The launcher MAY proceed without
    /// isolation — but a `warn` has already been logged. Carries the reason the
    /// namespace was unavailable (for the log/telemetry).
    UnisolatedOverride(NetnsError),
    /// The namespace could not be created and there is NO override. **FAIL CLOSED**:
    /// the launcher MUST abort and must NOT launch with full host egress.
    Refused(NetnsError),
}

/// Decide isolation for a runtime launch enforcing `posture`, for the runtime
/// identified by `slot_token`.
///
/// Order of decision:
///   1. isolation disabled by config → `DisabledByConfig` (dev/CI opt-out),
///   2. try [`netns::prepare`]; success → `Isolated`,
///   3. failure + `CHORD_ALLOW_UNISOLATED=1` → `UnisolatedOverride` (logged `warn`),
///   4. failure + no override → `Refused` (FAIL CLOSED).
pub fn decide_isolation(slot_token: &str, posture: &EgressPosture) -> IsolationDecision {
    if !netns::isolation_enabled() {
        tracing::debug!(
            target: "chord.supervisor.netns",
            "CHORD_NETNS_ISOLATION=0 — launching without a network namespace (dev opt-out)"
        );
        return IsolationDecision::DisabledByConfig;
    }

    match netns::prepare(slot_token, posture) {
        Ok(handle) => IsolationDecision::Isolated(handle),
        Err(e) => {
            if netns::unisolated_override() {
                // LOUD: the operator explicitly chose to run without the kernel
                // guarantee. This is the only way past fail-closed.
                tracing::warn!(
                    target: "chord.supervisor.netns",
                    reason = %e,
                    "CHORD_ALLOW_UNISOLATED=1 set — launching runtime WITHOUT network-namespace \
                     isolation; the kernel egress guarantee is NOT in effect for this runtime"
                );
                IsolationDecision::UnisolatedOverride(e)
            } else {
                // FAIL CLOSED: do not launch with full host egress.
                tracing::error!(
                    target: "chord.supervisor.netns",
                    reason = %e,
                    "refusing to launch runtime: network-namespace isolation unavailable and no \
                     CHORD_ALLOW_UNISOLATED override — failing closed (no full-egress launch)"
                );
                IsolationDecision::Refused(e)
            }
        }
    }
}

impl IsolationDecision {
    /// The prepared namespace handle to spawn into, if any. `Isolated` → `Some`;
    /// every other variant → `None` (the caller spawns without a namespace prefix,
    /// having already either opted out, overridden, or — for `Refused` — aborted).
    pub fn handle(&self) -> Option<&NetnsHandle> {
        match self {
            IsolationDecision::Isolated(h) => Some(h),
            _ => None,
        }
    }

    /// Whether the launch must be ABORTED (the fail-closed terminal). Only
    /// `Refused` aborts; every other variant permits a spawn (isolated or not).
    pub fn must_abort(&self) -> bool {
        matches!(self, IsolationDecision::Refused(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn clear_flags() {
        std::env::remove_var("CHORD_NETNS_ISOLATION");
        std::env::remove_var("CHORD_ALLOW_UNISOLATED");
    }

    #[test]
    #[serial]
    fn disabled_by_config_takes_the_legacy_path() {
        clear_flags();
        std::env::set_var("CHORD_NETNS_ISOLATION", "0");
        let d = decide_isolation("slot", &EgressPosture::Denied);
        assert!(matches!(d, IsolationDecision::DisabledByConfig));
        assert!(!d.must_abort(), "the dev opt-out must not abort the launch");
        assert!(d.handle().is_none());
        clear_flags();
    }

    #[test]
    #[serial]
    fn fails_closed_when_capability_absent_and_no_override() {
        // NEGATIVE TEST. On this unprivileged build prepare() fails; with isolation
        // ON and NO override, the decision MUST be Refused (abort), never a silent
        // full-egress launch.
        clear_flags();
        std::env::set_var("CHORD_NETNS_ISOLATION", "1"); // explicit ON
        let d = decide_isolation("slot", &EgressPosture::Denied);
        assert!(
            matches!(d, IsolationDecision::Refused(_)),
            "missing capability + no override MUST fail closed (Refused)"
        );
        assert!(d.must_abort(), "Refused must signal the launcher to abort");
        assert!(d.handle().is_none(), "a refused decision must carry no spawnable handle");
        clear_flags();
    }

    #[test]
    #[serial]
    fn explicit_override_permits_unisolated_launch_loudly() {
        // With the explicit operator override, the same missing-capability host
        // yields UnisolatedOverride (NOT Refused) — the only sanctioned bypass.
        clear_flags();
        std::env::set_var("CHORD_NETNS_ISOLATION", "1");
        std::env::set_var("CHORD_ALLOW_UNISOLATED", "1");
        let d = decide_isolation("slot", &EgressPosture::Denied);
        assert!(
            matches!(d, IsolationDecision::UnisolatedOverride(_)),
            "explicit override must permit an unisolated launch"
        );
        assert!(!d.must_abort(), "the override path must NOT abort");
        assert!(d.handle().is_none(), "override path spawns without a namespace");
        clear_flags();
    }

    #[test]
    #[serial]
    fn override_is_ignored_unless_exactly_1() {
        // A non-"1" override value must NOT bypass fail-closed.
        clear_flags();
        std::env::set_var("CHORD_NETNS_ISOLATION", "1");
        std::env::set_var("CHORD_ALLOW_UNISOLATED", "yes");
        let d = decide_isolation("slot", &EgressPosture::Denied);
        assert!(
            matches!(d, IsolationDecision::Refused(_)),
            "only an exact CHORD_ALLOW_UNISOLATED=1 may bypass fail-closed"
        );
        clear_flags();
    }
}
