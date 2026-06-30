//! Chord routing inputs derived from the S84 assistant-intake profile.
//!
//! Today this exposes the **assistant profile → chat-role selection** bridge
//! ([`assistant_profile`]): chord reads the measured assistant scores (via
//! `terminus_rs::intake::assistant::reporting`) and selects the Lumina chat alias
//! by measured fit, behind a latency/degradation guard. The per-backend on-demand
//! lifecycle routing lives in [`crate::models::routing`]; this module is the
//! *which model should the chat alias point at* decision, not the *how to start
//! it* mechanics.

pub mod assistant_profile;
