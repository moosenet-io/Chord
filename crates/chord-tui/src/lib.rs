//! chord-tui (S91) — a read-mostly ratatui control TUI client for the Chord and
//! Terminus-fleet control planes.
//!
//! Two modes, one binary, shared plumbing:
//!   - [`app`]        — top-level state, mode enum, confirm state machine
//!   - [`connection`] — async multi-instance health manager (never blocks the loop)
//!   - [`config`]     — persisted fleet + settings (secrets never written here)
//!   - [`secret`]     — vault-backed secret refs/values (values never logged)
//!   - [`confirm`]    — mutation gating (keystroke / typed / inert-stub)
//!   - [`modes`]      — Chord mode (live + stubbed S85 seam) + Terminus-fleet mode
//!   - [`ui`]         — ratatui rendering
//!
//! This crate is a CLIENT. It connects to Chord/Terminus over their stable
//! control endpoints; it never links, restarts, or reconfigures the live proxy
//! binary.

pub mod app;
pub mod config;
pub mod confirm;
pub mod connection;
pub mod modes;
pub mod secret;
pub mod ui;
