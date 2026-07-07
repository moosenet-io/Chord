// SNAP-04: ActivityTracker — passive observation of active inference requests.
use chrono::{DateTime, Utc};
use std::collections::HashMap;

use crate::snap::state::SharedInferenceState;

/// Per-model activity snapshot.
#[derive(Debug, Clone)]
pub struct ModelActivity {
    pub model: String,
    pub engine: String,
    pub active_requests: u32,
    pub last_seen: DateTime<Utc>,
}

/// Passive activity tracker: reads from SharedInferenceState.
/// A model is "in use" if it has active_requests > 0 and was observed within 10 seconds.
pub struct ActivityTracker {
    pub state: SharedInferenceState,
}

impl ActivityTracker {
    pub fn new(state: SharedInferenceState) -> Self {
        Self { state }
    }

    /// Check if a model is currently in use across all engines.
    pub async fn is_in_use(&self, model: &str) -> bool {
        let s = self.state.read().await;
        let threshold = Utc::now() - chrono::Duration::seconds(10);

        for engine in &s.engines {
            for loaded in &engine.models {
                if loaded.name == model && loaded.active_requests > 0 {
                    // State was updated recently if timestamp > threshold
                    if s.timestamp > threshold {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Get activity state for all engines and models.
    pub async fn all_activity(&self) -> Vec<ModelActivity> {
        let s = self.state.read().await;
        let mut activity = Vec::new();

        for engine in &s.engines {
            for loaded in &engine.models {
                activity.push(ModelActivity {
                    model: loaded.name.clone(),
                    engine: engine.name.clone(),
                    active_requests: loaded.active_requests,
                    last_seen: s.timestamp,
                });
            }
        }

        activity
    }

    /// Get activity grouped by engine then model name.
    pub async fn activity_by_engine(&self) -> HashMap<String, Vec<ModelActivity>> {
        let all = self.all_activity().await;
        let mut grouped: HashMap<String, Vec<ModelActivity>> = HashMap::new();
        for a in all {
            grouped.entry(a.engine.clone()).or_default().push(a);
        }
        grouped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snap::state::{
        EndpointStatus, EngineEndpoint, InferenceState, LoadedModel,
    };
    use std::sync::Arc;
    use tokio::sync::RwLock;

    fn make_state(engine: &str, model: &str, active: u32) -> SharedInferenceState {
        Arc::new(RwLock::new(InferenceState {
            engines: vec![EngineEndpoint {
                name: engine.to_string(),
                endpoint_env_var: "TEST_URL".to_string(),
                status: EndpointStatus::Online,
                models: vec![LoadedModel {
                    name: model.to_string(),
                    size_vram_mb: 0,
                    active_requests: active,
                    tokens_per_sec: None,
                }],
                response_time_ms: 50,
            }],
            vram: Default::default(),
            timestamp: Utc::now(),
        }))
    }

    #[tokio::test]
    async fn model_with_active_requests_is_in_use() {
        let state = make_state("ollama_gpu", "qwen3:8b", 2);
        let tracker = ActivityTracker::new(state);
        assert!(tracker.is_in_use("qwen3:8b").await);
    }

    #[tokio::test]
    async fn model_with_zero_requests_is_not_in_use() {
        let state = make_state("ollama_gpu", "qwen3:8b", 0);
        let tracker = ActivityTracker::new(state);
        assert!(!tracker.is_in_use("qwen3:8b").await);
    }

    #[tokio::test]
    async fn unknown_model_is_not_in_use() {
        let state = make_state("ollama_gpu", "qwen3:8b", 5);
        let tracker = ActivityTracker::new(state);
        assert!(!tracker.is_in_use("llama-3:70b").await);
    }

    #[tokio::test]
    async fn all_activity_returns_all_models() {
        let state = make_state("ollama_gpu", "qwen3:8b", 1);
        let tracker = ActivityTracker::new(state);
        let activity = tracker.all_activity().await;
        assert_eq!(activity.len(), 1);
        assert_eq!(activity[0].model, "qwen3:8b");
        assert_eq!(activity[0].engine, "ollama_gpu");
        assert_eq!(activity[0].active_requests, 1);
    }

    #[tokio::test]
    async fn activity_by_engine_groups_correctly() {
        let state = Arc::new(RwLock::new(InferenceState {
            engines: vec![
                EngineEndpoint {
                    name: "ollama_gpu".to_string(),
                    endpoint_env_var: "OLLAMA_URL".to_string(),
                    status: EndpointStatus::Online,
                    models: vec![
                        LoadedModel {
                            name: "qwen3:8b".to_string(),
                            size_vram_mb: 0,
                            active_requests: 1,
                            tokens_per_sec: None,
                        },
                        LoadedModel {
                            name: "mxbai-embed-large".to_string(),
                            size_vram_mb: 0,
                            active_requests: 0,
                            tokens_per_sec: None,
                        },
                    ],
                    response_time_ms: 50,
                },
                EngineEndpoint {
                    name: "llama-server".to_string(),
                    endpoint_env_var: "LLAMA_SERVER_URL".to_string(),
                    status: EndpointStatus::Online,
                    models: vec![LoadedModel {
                        name: "qwen3-coder:30b".to_string(),
                        size_vram_mb: 0,
                        active_requests: 2,
                        tokens_per_sec: None,
                    }],
                    response_time_ms: 30,
                },
            ],
            vram: Default::default(),
            timestamp: Utc::now(),
        }));

        let tracker = ActivityTracker::new(state);
        let grouped = tracker.activity_by_engine().await;
        assert_eq!(grouped.get("ollama_gpu").unwrap().len(), 2);
        assert_eq!(grouped.get("llama-server").unwrap().len(), 1);
    }

    #[tokio::test]
    async fn stale_state_model_not_in_use() {
        // State with timestamp older than 10 seconds
        let old_timestamp = Utc::now() - chrono::Duration::seconds(15);
        let state = Arc::new(RwLock::new(InferenceState {
            engines: vec![EngineEndpoint {
                name: "ollama_gpu".to_string(),
                endpoint_env_var: "OLLAMA_URL".to_string(),
                status: EndpointStatus::Online,
                models: vec![LoadedModel {
                    name: "qwen3:8b".to_string(),
                    size_vram_mb: 0,
                    active_requests: 5,
                    tokens_per_sec: None,
                }],
                response_time_ms: 50,
            }],
            vram: Default::default(),
            timestamp: old_timestamp,
        }));
        let tracker = ActivityTracker::new(state);
        // Stale state → not considered in use
        assert!(!tracker.is_in_use("qwen3:8b").await);
    }
}
