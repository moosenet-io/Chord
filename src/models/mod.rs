//! Model storage tiering.
//!
//! Tracks every known model across the three storage tiers (hot / warm / cold)
//! in a file-backed JSON registry that survives Chord restarts and is reconciled
//! against the on-disk Ollama manifest trees at startup.
//!
//! See `specs/S79-model-tiering.md` (TIER-01).

pub mod backends;
pub mod batch_suitability;
pub mod coding_selector;
pub mod eviction;
pub mod registry;
pub mod rope_ingest;
pub mod routing;
pub mod transfer;
pub mod work_type;
