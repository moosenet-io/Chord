//! SNAP shared inference-state types.
//!
//! Vendored verbatim from harmony-chord's `harmony_core::state::inference_state`
//! so the SNAP observability subsystem (vram / activity / health) is fully
//! self-contained inside chord — no `harmony_core` dependency is pulled in.
//!
//! These are pure data types (a live snapshot of the inference infrastructure)
//! plus a shared `Arc<RwLock<_>>` handle updated by the background health
//! monitor (`snap::health`).

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

/// Snapshot of the entire inference infrastructure at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceState {
    pub engines: Vec<EngineEndpoint>,
    pub vram: VRAMState,
    pub timestamp: DateTime<Utc>,
}

impl Default for InferenceState {
    fn default() -> Self {
        Self {
            engines: Vec::new(),
            vram: VRAMState::default(),
            timestamp: Utc::now(),
        }
    }
}

/// One inference engine endpoint with health and loaded models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineEndpoint {
    /// Human-readable name (e.g., "llama-server", "ollama_gpu", "ollama_cpu").
    pub name: String,
    /// Env var that supplies this endpoint's URL (e.g., "LLAMA_SERVER_URL").
    pub endpoint_env_var: String,
    /// Current reachability status.
    pub status: EndpointStatus,
    /// Models currently loaded into VRAM on this engine.
    pub models: Vec<LoadedModel>,
    /// Round-trip time of the last health check in milliseconds.
    pub response_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum EndpointStatus {
    /// Responded in <2 seconds.
    Online,
    /// Responded in 2-5 seconds.
    Degraded,
    /// Failed to respond or timed out.
    Offline,
}

/// A model currently occupying VRAM on an inference engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadedModel {
    pub name: String,
    /// Estimated VRAM used in megabytes (from Ollama /api/ps or llama-server).
    pub size_vram_mb: u64,
    /// Number of requests currently being processed.
    pub active_requests: u32,
    /// Measured tokens per second (None if not available).
    pub tokens_per_sec: Option<f64>,
}

/// GPU VRAM state: total, used, free, and per-model allocations.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VRAMState {
    /// Total GPU VRAM in megabytes.
    pub total_mb: u64,
    /// Currently used VRAM in megabytes.
    pub used_mb: u64,
    /// Free VRAM (total - used).
    pub free_mb: u64,
    /// Per-model VRAM allocations.
    pub allocations: Vec<VRAMAllocation>,
}

impl VRAMState {
    /// Compute free_mb from total and used.
    pub fn compute_free(&mut self) {
        self.free_mb = self.total_mb.saturating_sub(self.used_mb);
    }

    /// Check if a model of `needed_mb` fits in available VRAM.
    pub fn can_fit(&self, needed_mb: u64) -> bool {
        self.free_mb >= needed_mb
    }
}

/// A single model's VRAM allocation record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VRAMAllocation {
    pub model_name: String,
    pub engine: String,
    pub size_mb: u64,
    pub loaded_at: DateTime<Utc>,
}

/// Shared inference state updated by the background health poller.
pub type SharedInferenceState = Arc<RwLock<InferenceState>>;

/// Create a new shared inference state with empty defaults.
pub fn new_shared_state() -> SharedInferenceState {
    Arc::new(RwLock::new(InferenceState::default()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vram_state_compute_free() {
        let mut vram = VRAMState {
            total_mb: 98304,
            used_mb: 18000,
            ..Default::default()
        };
        vram.compute_free();
        assert_eq!(vram.free_mb, 80304);
    }

    #[test]
    fn vram_can_fit_when_sufficient() {
        let vram = VRAMState {
            total_mb: 98304,
            used_mb: 18000,
            free_mb: 80304,
            allocations: vec![],
        };
        assert!(vram.can_fit(17000));
        assert!(!vram.can_fit(90000));
    }

    #[test]
    fn endpoint_status_serializes() {
        let json = serde_json::to_string(&EndpointStatus::Online).unwrap();
        assert_eq!(json, r#""online""#);
        let json = serde_json::to_string(&EndpointStatus::Offline).unwrap();
        assert_eq!(json, r#""offline""#);
    }

    #[test]
    fn inference_state_has_defaults() {
        let state = InferenceState::default();
        assert!(state.engines.is_empty());
        assert_eq!(state.vram.total_mb, 0);
    }
}
