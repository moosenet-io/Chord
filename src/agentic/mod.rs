pub mod argument_guard;
pub mod behavioral_monitor;
pub mod context;
pub mod harness_integration;
pub mod loop_runner;
pub mod model_router;
pub mod permissions;
pub mod response_guard;
pub mod result_guard;
pub mod streaming;
pub mod synthesis;

// Re-export primary types for ergonomic use from other modules.
pub use context::{AgenticRequest, AgenticResponse, ExecutionStep};
pub use loop_runner::AgenticExecutor;

// SecurityEvent shared across all guards
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SecurityEvent {
    pub guard_name: String,
    pub action: SecurityAction,
    pub tool_name: String,
    pub reason: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum SecurityAction {
    Blocked,
    Sanitized,
    Warned,
}
