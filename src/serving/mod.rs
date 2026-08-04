//! Chord serving-profile reader + runtime launcher (S85 SRV-04).
//!
//! The CONSUME side of the S85 serving dimension. SRV-01
//! (`terminus-rs::intake::serving`) defines the `serving_profile` table and the
//! shared types; the SRV-02/03 harness writes the rows. This module teaches Chord
//! to:
//!   1. read those rows into an in-memory [`profile::RoutingMap`] keyed by
//!      `model_id` (on startup + on refresh), and
//!   2. [`launcher`]-launch the correct runtime + env/flags for a requested model,
//!      with `fallback_runtime` on failure and genericized errors (S77).
//!
//! ## Split of responsibility with SRV-05
//! SRV-04 does NOT own VRAM admission or eviction. A `keep_warm` model must never
//! be cold-launched inline; SRV-04 delegates those to the
//! [`launcher::ResidencyManager`] trait it DEFINES here, and ships a trivial
//! [`launcher::PassThroughResidency`] stub so the crate compiles + tests today.
//!
//! SRV-05's implementation of that trait (`serving::residency::VramResidencyManager`)
//! was DELETED by CHRD-98: it was never constructed anywhere in chord's history, and
//! `routing::resident_set::ResidentSet` (CHRD-PIN-01) is the single owner of VRAM
//! residency on the live host. Do not reintroduce a second one behind this seam —
//! two independent residency owners fighting over one idle-reaped GPU is the exact
//! bug CHRD-PIN-01 was written to eliminate.

pub mod eviction;
pub mod launcher;
pub mod memory_model;
pub mod mode;
pub mod profile;
pub mod release_verify;
pub mod swap;

pub use eviction::{plan_admission, EvictTarget, EvictionPlan, ResidentView, Tier};
pub use memory_model::{
    classify_substrate, select_memory_model, ActivationEvent, MemoryModel, MemorySnapshot,
    ModelSelection, Pool, SeparateCeilings, Substrate, SubstrateInfo, UnifiedPool,
};
pub use mode::{ModeAction, ModeController, ModeError, OperatingMode};
pub use release_verify::{
    verify_release, DeviceProbe, ReleaseConfig, ReleaseOutcome, SysfsDeviceProbe,
};
pub use swap::{
    clean_swap, default_ctx_for_footprint, CleanLauncher, ContextDefaults, NetnsReapingTeardown,
    NoopSwapEventSink, SwapError, SwapEvent, SwapEventSink, SwapOutcome, SwapRequest, Teardown,
};
pub use launcher::{
    build_launch_command, scrub_launch_env, teardown_netns, FailureRecorder, HealthChecker,
    LaunchCommand, LaunchError, Launcher, NoopFailureRecorder, PassThroughResidency,
    ResidencyError, ResidencyManager, RuntimeSpawner, ServeHandle, Slot,
};
pub use profile::{
    DbProfileSource, EnvSpec, ProfileLoadError, ProfileSource, RouteEntry, RoutingMap,
    StaticProfileSource,
};
