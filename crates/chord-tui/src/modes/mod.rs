//! The two operating modes (S91 CTUI-01).
//!
//! One binary, two modes that SHARE plumbing (connection manager, config,
//! secrets, event loop) but are NOT blended into one view:
//!   - [`chord`]           — Chord control plane (CTUI-02 live + CTUI-03 stubs)
//!   - [`terminus_fleet`]  — Terminus-fleet control plane (shell scaffold; the
//!     fleet panels are a later phase, but the mode + switch exist now so the
//!     shared plumbing is exercised).

pub mod chord;
pub mod terminus_fleet;
