//! Terminus-fleet mode (S91 CTUI-04 connect+status, CTUI-05 per-instance config).
//!
//! The second top-level mode. It SHARES all CTUI-01 plumbing (connection
//! manager, config, secrets, event loop) with Chord mode but is a SEPARATE view
//! — the two are never blended into one screen.
//!
//! Sub-modules:
//!   - [`fleet`]      — the fleet model: add/remove/select instances, each
//!     declaring transport (stdio|HTTP), endpoint (from config, never a literal)
//!     and kind (local|remote|chord-embedded). CTUI-04.
//!   - [`mcp_client`] — connect to a Terminus MCP server over stdio OR HTTP,
//!     list tools, read health/status; drift/version/auth tolerant. CTUI-04.
//!   - [`tools`]      — enable/disable individual tools + scope view/edit, gated
//!     behind confirm and capability-aware (no faked mutation). CTUI-05.
//!   - [`secrets`]    — per-instance VAULT-backed secret management: names +
//!     status ONLY, change = typed confirmation, writes to vault never file. CTUI-05.
//!   - [`transport`]  — per-instance transport config (stdio/HTTP+endpoint) via
//!     config, no hardcoded infra. CTUI-05.
//!   - [`audit`]      — sanitized audit log of every attempted/applied mutation
//!     (never a secret value). CTUI-05.
//!
//! A single unreachable instance MUST NOT break the others: every per-instance
//! operation is isolated and returns a per-instance error.

pub mod audit;
pub mod fleet;
pub mod mcp_client;
pub mod secrets;
pub mod tools;
pub mod transport;

// Re-export CTUI-01 fleet-panel identity so existing shell code keeps compiling
// unchanged (the mode + switch predate the fleet build-out).
pub use fleet::{FleetPanel, FLEET_SCAFFOLD_NOTE};
