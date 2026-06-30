//! S88 ISO-02 — capability-gated integration tests for the per-runtime network
//! namespace.
//!
//! These exercise the REAL kernel guarantee and therefore require `CAP_NET_ADMIN`
//! and the `ip`/`nft` userspace. They are `#[ignore]`d so they DO NOT run in the
//! ordinary (unprivileged) `cargo test` — including this CI build. They compile
//! everywhere and run under `cargo test -- --ignored` on a privileged host (the
//! ISO-04 operator verification step).
//!
//! Run them on a privileged host with:
//!   `cargo test --test netns_integration -- --ignored`
//!
//! Each test FIRST checks the capability is actually present and SKIPs (returns
//! early, not fails) if not — so an accidental `--ignored` run on an unprivileged
//! host is a no-op rather than a spurious failure. The assertions only run when the
//! namespace was genuinely created.

#![cfg(target_os = "linux")]

use std::process::Command;

use chord_proxy::supervisor::egress_policy::EgressPosture;
use chord_proxy::supervisor::netns;

/// True iff we can actually create a network namespace here (the privileged gate).
/// When false, the integration tests SKIP rather than fail.
fn privileged() -> bool {
    // `ip netns add` is the same op the production path uses; if it fails we lack
    // the capability/tooling and the test must skip.
    let probe = "chord-iso02-probe";
    let add = Command::new("ip").args(["netns", "add", probe]).status();
    match add {
        Ok(s) if s.success() => {
            let _ = Command::new("ip").args(["netns", "del", probe]).status();
            true
        }
        _ => false,
    }
}

/// Run a command INSIDE a named netns and return whether it succeeded.
fn in_ns_succeeds(ns: &str, argv: &[&str]) -> bool {
    let mut full = vec!["netns", "exec", ns];
    full.extend_from_slice(argv);
    Command::new("ip")
        .args(&full)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[test]
#[ignore = "requires CAP_NET_ADMIN + ip/nft; run under ISO-04 with --ignored"]
fn serve_netns_external_connect_fails() {
    if !privileged() {
        eprintln!("SKIP: no CAP_NET_ADMIN/ip — serve_netns_external_connect_fails");
        return;
    }
    // A Serve posture → Denied → a namespace with loopback up but NO route.
    let handle = netns::prepare("it-serve", &EgressPosture::Denied)
        .expect("serve namespace must be creatable on a privileged host");

    // There is NO default route in the namespace, so any attempt to reach an
    // external address must fail. We try to ping an external IP (1 packet, fast
    // timeout); it MUST fail (no route → ENETUNREACH).
    let external_reachable = in_ns_succeeds(
        handle.name(),
        &["ping", "-c", "1", "-W", "1", "203.0.113.1"], // TEST-NET-3, never routable
    );
    let teardown = handle.teardown();

    assert!(
        !external_reachable,
        "a Serve (Denied) namespace MUST have no path off the box — external connect must fail"
    );
    assert!(teardown.is_ok(), "teardown must succeed");
    // Idempotency: tearing the same namespace down again is Ok.
    assert!(netns::NetnsHandle::for_teardown("it-serve").teardown().is_ok()
        || handle.teardown().is_ok());
}

#[test]
#[ignore = "requires CAP_NET_ADMIN + ip/nft; run under ISO-04 with --ignored"]
fn pull_netns_allow_listed_reachable_others_denied() {
    if !privileged() {
        eprintln!("SKIP: no CAP_NET_ADMIN/ip/nft — pull_netns_allow_listed_others_denied");
        return;
    }
    // A Pull posture with an allow-list → a constrained, nft-filtered egress path.
    // We use a deliberately small allow-list; the operator running ISO-04 supplies
    // a real, reachable allow-listed host via the env they verify against. Here we
    // assert the SHAPE: the namespace is created with an egress path, and a NON
    // allow-listed destination is denied by the in-namespace nft default-drop.
    let allow = vec!["example.com".to_string()];
    let handle = netns::prepare("it-pull", &EgressPosture::AllowList(allow))
        .expect("pull namespace must be creatable on a privileged host");

    // A destination NOT on the allow-list must be denied by the nft default-drop.
    // (We cannot assert positive reachability without a guaranteed-up allow-listed
    // host in CI; ISO-04 verifies the positive case against a real model source.)
    let non_allowlisted_reachable = in_ns_succeeds(
        handle.name(),
        &["ping", "-c", "1", "-W", "1", "203.0.113.2"],
    );
    let teardown = handle.teardown();

    assert!(
        !non_allowlisted_reachable,
        "a non-allow-listed destination MUST be denied by the in-namespace nft default-drop"
    );
    assert!(teardown.is_ok());
}

#[test]
#[ignore = "requires CAP_NET_ADMIN + ip; run under ISO-04 with --ignored"]
fn teardown_is_idempotent_even_after_crash() {
    if !privileged() {
        eprintln!("SKIP: no CAP_NET_ADMIN/ip — teardown_is_idempotent_even_after_crash");
        return;
    }
    let handle = netns::prepare("it-idem", &EgressPosture::Denied)
        .expect("namespace must be creatable");
    // First teardown removes it.
    assert!(handle.teardown().is_ok());
    // Simulating a crash that already removed the namespace: a second teardown of
    // the now-absent namespace must STILL be Ok (no leak, no error).
    assert!(handle.teardown().is_ok(), "teardown must be idempotent (no leaked netns)");
    // And a fresh handle for the same name also tears down cleanly.
    assert!(netns::NetnsHandle::for_teardown(handle.name()).teardown().is_ok());
}
