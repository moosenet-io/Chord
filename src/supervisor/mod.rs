//! Runtime supervisor — launch-environment scrubbing + egress policy (S88 ISO-01).
//!
//! This module owns the *posture* of a runtime launch: the environment a runtime
//! child is spawned with ([`launch_env`]) and the network egress policy that
//! launch should have ([`egress_policy`]).
//!
//! ## Scope — ISO-01 (advisory) + ISO-02 (the kernel guarantee)
//! ISO-01 ships the env-scrub ([`launch_env`]) and the egress-policy config
//! surface ([`egress_policy`]); it is ADVISORY — it relies on the runtimes
//! HONOURING the telemetry-off / offline opt-outs.
//!
//! ISO-02 ([`netns`], [`egress_filter`], [`launch_isolation`]) is the KERNEL
//! guarantee: a per-runtime network namespace that physically blocks the egress
//! ISO-01 only declared. A `Serve`/`Denied` runtime gets a namespace with NO route
//! (every external `connect()` fails at the kernel); a `Pull`/`AllowList` runtime
//! gets a constrained, nftables-filtered egress path to the configured model
//! sources only. It is **fail-closed**: without `CAP_NET_ADMIN` the launch is
//! refused, not run with full host egress (an explicit `CHORD_ALLOW_UNISOLATED=1`
//! override exists and is loud + off by default).
//!
//! Honest scope: ISO-02 isolates the runtimes Chord LAUNCHES. It does NOT firewall
//! Chord's own process (ISO-03) and does NOT replace the host firewall for
//! non-Chord processes. See `docs/egress.md`.
//!
//! The launcher ([`crate::serving::launcher`]) consumes [`launch_env`] for the
//! scrubbed env AND [`launch_isolation`]/[`netns`] to spawn the runtime inside its
//! namespace; the SRV-12 clean swap ([`crate::serving::swap`]) tears the outgoing
//! runtime's namespace down.

pub mod egress_filter;
pub mod egress_policy;
pub mod launch_env;
pub mod launch_isolation;
pub mod netns;

pub use egress_policy::{posture_for, EgressPosture, RuntimeClass};
pub use launch_env::build_runtime_env;
pub use launch_isolation::{decide_isolation, IsolationDecision};
pub use netns::{NetnsConfig, NetnsError, NetnsHandle};
