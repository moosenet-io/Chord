//! Admin control surface for Chord.
//!
//! These are operator/orchestrator-facing endpoints served on the control port
//! (default 8090, `CHORD_CONTROL_PORT`) alongside the TIER-05 model-tier API in
//! [`crate::control`]. They are auth-gated with the same JWT check as every other
//! control/proxy route (`auth_check` / `auth_error_response`).
//!
//! - [`idle`] — BLD-09 idle-mode: free the heavy host's RAM/GPU on demand so the
//!   constellation compiler can build there, then restore full inference service.

pub mod idle;
