//! Runtime supervisor — launch-environment scrubbing + egress policy (S88 ISO-01).
//!
//! This module owns the *posture* of a runtime launch: the environment a runtime
//! child is spawned with ([`launch_env`]) and the network egress policy that
//! launch should have ([`egress_policy`]).
//!
//! ## Honest scope — ISO-01 is ADVISORY
//! ISO-01 ships the env-scrub and the egress-policy **config surface only**. The
//! env-scrub sets documented telemetry-off / offline opt-outs and strips proxy
//! vars; it relies on the runtimes HONOURING those opt-outs. The egress policy is
//! *declared* (Serve = Denied, Pull = allow-list-or-Denied) but **not enforced at
//! the kernel**. The actual guarantee — a network namespace that physically blocks
//! egress so a misbehaving binary cannot reach the internet — is **ISO-02 and is
//! NOT built yet**. Do not treat ISO-01 as a security boundary; it is
//! defense-in-depth and the policy plumbing ISO-02 will enforce.
//!
//! The launcher ([`crate::serving::launcher`]) consumes [`launch_env`] when it
//! assembles a runtime's launch env; ISO-02 will additionally consume
//! [`egress_policy`] to build the netns.

pub mod egress_policy;
pub mod launch_env;

pub use egress_policy::{posture_for, EgressPosture, RuntimeClass};
pub use launch_env::build_runtime_env;
