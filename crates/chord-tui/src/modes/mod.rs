//! The two operating modes (S91 CTUI-01).
//!
//! One binary, two modes that SHARE plumbing (connection manager, config,
//! secrets, event loop) but are NOT blended into one view:
//!   - [`chord`]     — Chord control plane (CTUI-02 live + CTUI-03 stubs)
//!   - [`terminus`]  — Terminus-fleet control plane (CTUI-04 connect+status,
//!     CTUI-05 per-instance config). Shares plumbing with Chord mode but is a
//!     SEPARATE view — the two are never blended.

pub mod chord;
pub mod terminus;

/// Backward-compatible path alias: CTUI-01 code referred to `terminus_fleet`;
/// the fleet build-out (CTUI-04/05) lives under `terminus::fleet`. This keeps the
/// existing shell (`app`, `ui`) compiling unchanged.
pub mod terminus_fleet {
    pub use crate::modes::terminus::fleet::*;
}
