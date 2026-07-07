//! Validation harnesses for serving-profile claims that must be PROVEN before
//! they're trusted (S86 YARN-03).
//!
//! Nothing in [`crate::serving::profile::RopeScaling`] flips its `validated`
//! flag on say-so — research could not confirm YaRN holds cleanly on gfx1151,
//! so a candidate config only earns `validated: true` after
//! [`yarn_validate::run_validation`] confirms it against injected launch +
//! probe results. This module builds that harness's decision machinery; the
//! actual run against real gfx1151 hardware is a separate, gated, human-action
//! item (YARN-04) — everything here is pure logic over an injectable seam, so
//! it is fully unit-testable without touching real infrastructure.

pub mod yarn_validate;

pub use yarn_validate::{
    evaluate, probe_depths_for, recall_score, run_validation, ContextProber, LaunchReport,
    ProbeResult, ValidationEvidence, YarnLauncher, COLLAPSE_RATIO, PROBE_DEPTH_FRACTIONS,
    WEAK_BASELINE_THRESHOLD,
};
