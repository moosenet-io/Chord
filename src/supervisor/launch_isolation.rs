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
    decide_isolation_with(slot_token, posture, netns::prepare)
}

/// [`decide_isolation`] with the namespace-preparing operation INJECTED (CHRD-97).
///
/// The valuable, host-independent part of this module is the DECISION — which of the
/// four [`IsolationDecision`] variants a given (config flag, prepare outcome,
/// override flag) triple produces. That decision does not need a real namespace to
/// be exercised, but until now the only way in was [`netns::prepare`], so the tests
/// drove the privileged create path and leaked live namespaces on the (root) build
/// host. Passing the operation in lets the tests supply an outcome directly, which
/// also makes the `Isolated` branch testable at all — it previously was not, because
/// it required a real namespace.
///
/// Production calls [`decide_isolation`], which passes [`netns::prepare`], so the
/// behavior of the shipped path is unchanged.
pub(crate) fn decide_isolation_with<P>(
    slot_token: &str,
    posture: &EgressPosture,
    prepare_ns: P,
) -> IsolationDecision
where
    P: FnOnce(&str, &EgressPosture) -> Result<NetnsHandle, NetnsError>,
{
    if !netns::isolation_enabled() {
        tracing::debug!(
            target: "chord.supervisor.netns",
            "CHORD_NETNS_ISOLATION=0 — launching without a network namespace (dev opt-out)"
        );
        return IsolationDecision::DisabledByConfig;
    }

    match prepare_ns(slot_token, posture) {
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
    use std::cell::Cell;

    fn clear_flags() {
        std::env::remove_var("CHORD_NETNS_ISOLATION");
        std::env::remove_var("CHORD_ALLOW_UNISOLATED");
    }

    // ── CHRD-97: the tests drive the DECISION, never the host ────────────────────
    //
    // These tests used to call `decide_isolation`, which called the real
    // `netns::prepare`. Their comments claimed "on this unprivileged build
    // prepare() fails" — but Chord's build/test host runs as ROOT, so the claim was
    // false and each of these tests actually created a live network namespace on the
    // build host and leaked it (`ip netns list` showed the residue after a run). The
    // assertions below are UNCHANGED; only the way the prepare OUTCOME is obtained
    // has moved, from "whatever this host happens to do" to "what this test states".
    //
    // The fake outcomes are built from `netns::prepare_with_probe` — the injectable
    // seam netns.rs already exposes — rather than from a hand-rolled `Ok`/`Err`, so
    // the values flowing into the decision are produced by the same code production
    // runs, just with the privileged probe and create operation supplied.

    /// A `prepare` that reports NO capability. Real `prepare_with_probe`, false
    /// probe, panicking create ⇒ the create path is provably not entered and nothing
    /// can land on the host.
    #[cfg(target_os = "linux")]
    fn prepare_unprivileged(
        slot: &str,
        posture: &EgressPosture,
    ) -> Result<NetnsHandle, NetnsError> {
        netns::prepare_with_probe(slot, posture, || false, |_name, _cfg| {
            unreachable!("fail-closed prepare must never enter the create path")
        })
    }

    #[cfg(not(target_os = "linux"))]
    fn prepare_unprivileged(
        _slot: &str,
        _posture: &EgressPosture,
    ) -> Result<NetnsHandle, NetnsError> {
        Err(NetnsError::Unsupported)
    }

    /// A `prepare` that SUCCEEDS, without touching the kernel: the probe reports the
    /// capability and the create operation is a no-op that returns `Ok`. This is what
    /// makes the `Isolated` branch testable at all — it previously required a real
    /// privileged host, so it had no unit coverage.
    #[cfg(target_os = "linux")]
    fn prepare_succeeding(
        slot: &str,
        posture: &EgressPosture,
    ) -> Result<NetnsHandle, NetnsError> {
        netns::prepare_with_probe(slot, posture, || true, |_name, _cfg| Ok(()))
    }

    #[test]
    #[serial]
    fn disabled_by_config_takes_the_legacy_path() {
        clear_flags();
        std::env::set_var("CHORD_NETNS_ISOLATION", "0");
        // The config opt-out must short-circuit BEFORE prepare is attempted — assert
        // that structurally: a prepare that panics if called proves it was not.
        let d = decide_isolation_with("slot", &EgressPosture::Denied, |_, _| {
            panic!("the CHORD_NETNS_ISOLATION=0 opt-out must not attempt to prepare a namespace")
        });
        assert!(matches!(d, IsolationDecision::DisabledByConfig));
        assert!(!d.must_abort(), "the dev opt-out must not abort the launch");
        assert!(d.handle().is_none());
        clear_flags();
    }

    #[test]
    #[serial]
    fn fails_closed_when_capability_absent_and_no_override() {
        // NEGATIVE TEST. When prepare fails, with isolation ON and NO override, the
        // decision MUST be Refused (abort), never a silent full-egress launch.
        clear_flags();
        std::env::set_var("CHORD_NETNS_ISOLATION", "1"); // explicit ON
        let d = decide_isolation_with("slot", &EgressPosture::Denied, prepare_unprivileged);
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
        // With the explicit operator override, the same missing-capability outcome
        // yields UnisolatedOverride (NOT Refused) — the only sanctioned bypass.
        clear_flags();
        std::env::set_var("CHORD_NETNS_ISOLATION", "1");
        std::env::set_var("CHORD_ALLOW_UNISOLATED", "1");
        let d = decide_isolation_with("slot", &EgressPosture::Denied, prepare_unprivileged);
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
        let d = decide_isolation_with("slot", &EgressPosture::Denied, prepare_unprivileged);
        assert!(
            matches!(d, IsolationDecision::Refused(_)),
            "only an exact CHORD_ALLOW_UNISOLATED=1 may bypass fail-closed"
        );
        clear_flags();
    }

    /// NEW COVERAGE (CHRD-97). The happy path — isolation ON and prepare SUCCEEDS —
    /// must yield `Isolated` carrying a spawnable handle, and must NOT abort. This
    /// branch was previously unreachable in a unit test because it needed a real
    /// namespace; with the outcome injected it is a normal test.
    #[test]
    #[serial]
    #[cfg(target_os = "linux")]
    fn successful_prepare_yields_an_isolated_decision_with_a_spawnable_handle() {
        clear_flags();
        std::env::set_var("CHORD_NETNS_ISOLATION", "1");
        let d = decide_isolation_with("slot", &EgressPosture::Denied, prepare_succeeding);
        assert!(
            matches!(d, IsolationDecision::Isolated(_)),
            "a successful prepare must produce Isolated"
        );
        assert!(!d.must_abort(), "an isolated launch must not abort");
        let h = d.handle().expect("Isolated must carry the handle to spawn into");
        assert_eq!(
            h.name(),
            netns::namespace_name("slot"),
            "the handle must name the namespace derived from the slot token"
        );
        clear_flags();
    }

    /// The override must NOT be consulted when isolation succeeded — a successful
    /// prepare is `Isolated` even with `CHORD_ALLOW_UNISOLATED=1` set. Guards against
    /// a refactor that checks the bypass before the outcome.
    #[test]
    #[serial]
    #[cfg(target_os = "linux")]
    fn override_does_not_downgrade_a_successful_isolation() {
        clear_flags();
        std::env::set_var("CHORD_NETNS_ISOLATION", "1");
        std::env::set_var("CHORD_ALLOW_UNISOLATED", "1");
        let d = decide_isolation_with("slot", &EgressPosture::Denied, prepare_succeeding);
        assert!(
            matches!(d, IsolationDecision::Isolated(_)),
            "the override is a fallback for a FAILED prepare, never a downgrade of a \
             successful one"
        );
        clear_flags();
    }

    /// The decision must pass the caller's slot token and posture through to prepare
    /// unmodified — otherwise a runtime could be isolated under the wrong posture.
    #[test]
    #[serial]
    #[cfg(target_os = "linux")]
    fn slot_token_and_posture_reach_prepare_unmodified() {
        clear_flags();
        std::env::set_var("CHORD_NETNS_ISOLATION", "1");
        let seen = Cell::new(None);
        let posture = EgressPosture::AllowList(vec!["huggingface.co".to_string()]);
        let d = decide_isolation_with("slot-XYZ", &posture, |slot, p| {
            seen.set(Some((slot.to_string(), p.clone())));
            netns::prepare_with_probe(slot, p, || true, |_n, _c| Ok(()))
        });
        let (slot, p) = seen.take().expect("prepare must be called when isolation is on");
        assert_eq!(slot, "slot-XYZ");
        assert_eq!(p, EgressPosture::AllowList(vec!["huggingface.co".to_string()]));
        assert_eq!(
            d.handle().expect("Isolated").posture(),
            &EgressPosture::AllowList(vec!["huggingface.co".to_string()]),
            "the handle must record the posture it was prepared for"
        );
        clear_flags();
    }
}
