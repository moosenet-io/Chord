//! AGENT-01: AgenticRequest / AgenticResponse / ExecutionStep data types.
//!
//! These structs carry the full conversation context into Chord's internal
//! tool-calling loop and return a final answer plus a metadata-only execution
//! log to the caller.  Tool arguments and raw results are NEVER included in
//! any of these types — they stay inside the loop runner.

use serde::{Deserialize, Serialize};
use crate::agentic::SecurityEvent;

// ── Input ─────────────────────────────────────────────────────────────────────

/// A single conversation message (user / assistant / tool).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Role: "user", "assistant", or "tool".
    pub role: String,
    /// Text content of the message.
    pub content: String,
    /// For tool messages: the tool call id that produced this message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// A tool definition forwarded from the orchestrator into the agentic loop.
///
/// The loop passes these to the LLM so it knows what tools are available.
/// Chord validates each tool call against `AgenticRequest.permissions` before
/// executing — the definition itself does not grant access.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name (must match what the MCP proxy accepts).
    pub name: String,
    /// Human-readable description used by the LLM.
    pub description: String,
    /// JSON Schema for the tool's input parameters.
    pub parameters: serde_json::Value,
}

/// Full request sent to `POST /v1/agent/execute`.
///
/// the orchestrator packages the conversation and dispatches it here.  Chord runs the
/// entire tool-calling loop internally and returns only the final response plus
/// execution metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgenticRequest {
    /// Conversation history (user + assistant turns so far).
    pub messages: Vec<Message>,

    /// System prompt (with memories already injected by the orchestrator).
    #[serde(default)]
    pub system_prompt: String,

    /// Tool definitions pre-filtered by the orchestrator for this user / context.
    #[serde(default)]
    pub tools: Vec<ToolDefinition>,

    /// Tool names this user is allowed to call.  Wildcard-suffix patterns like
    /// `"google_calendar_*"` are supported.  `["*"]` = admin (all tools).
    /// `[]` = guest (no tools).
    #[serde(default)]
    pub permissions: Vec<String>,

    /// Maximum tool calls within this execution.  Capped at 10.
    #[serde(default = "default_max_tool_calls")]
    pub max_tool_calls: u8,

    /// Total wall-clock budget for the loop in seconds.
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u32,

    /// User identifier — used for rate limiting and audit logging.
    pub user_id: String,

    /// Preferred model identifier (may be overridden by `model_override`).
    pub model: String,

    /// Force a specific model, bypassing any routing logic.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_override: Option<String>,

    /// RESP-04: when true, `/v1/agent/execute` streams [`ProgressEvent`]s back as
    /// Server-Sent Events instead of returning a single buffered JSON response.
    /// Backward-compatible: absent ⇒ `false` ⇒ the legacy buffered path.
    #[serde(default)]
    pub stream: bool,
}

fn default_max_tool_calls() -> u8 { 5 }
fn default_timeout_secs() -> u32 { 60 }

// ── Output ────────────────────────────────────────────────────────────────────

/// Token usage breakdown returned by the LLM.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// A single step in the execution log.
///
/// CRITICAL: This struct MUST NOT contain tool arguments or results.
/// Only metadata (tool name, timing, status) crosses the wire to the orchestrator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStep {
    /// Step type: "tool_call", "llm_response", "guard_block", "timeout".
    pub step_type: String,
    /// Tool name involved (empty for LLM-only steps).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    /// Wall-clock duration of this step in milliseconds.
    pub duration_ms: u64,
    /// Outcome: "ok", "blocked", "error", "timeout".
    pub status: String,
    /// Human-readable error description (never contains args or results).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
}

/// Response returned by `POST /v1/agent/execute`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgenticResponse {
    /// Final text response delivered to the user.
    pub response: String,

    /// Metadata-only log of every step.  NO tool arguments or results included.
    pub execution_log: Vec<ExecutionStep>,

    /// Aggregated token usage across all LLM calls in the loop.
    pub tokens_used: TokenUsage,

    /// The model that actually handled the request.
    pub model_used: String,

    /// How many tool calls were made during this execution.
    pub tool_calls_made: u8,

    /// Total wall-clock duration of the entire execution in milliseconds.
    pub duration_ms: u64,

    /// Security events emitted by any guard during the loop.
    /// Sent to the orchestrator so administrators can review injection attempts.
    pub security_events: Vec<SecurityEvent>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── Serialisation round-trips ─────────────────────────────────────────────

    #[test]
    fn test_agentic_request_roundtrip() {
        let req = AgenticRequest {
            messages: vec![Message {
                role: "user".into(),
                content: "Hello".into(),
                tool_call_id: None,
            }],
            system_prompt: "You are Lumina.".into(),
            tools: vec![ToolDefinition {
                name: "utc_now".into(),
                description: "Returns UTC time".into(),
                parameters: json!({}),
            }],
            permissions: vec!["utc_now".into()],
            max_tool_calls: 3,
            timeout_secs: 30,
            user_id: "<operator>".into(),
            model: "claude-3-haiku".into(),
            model_override: None,
            stream: false,
        };

        let json = serde_json::to_string(&req).expect("serialize");
        let back: AgenticRequest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.user_id, "<operator>");
        assert_eq!(back.max_tool_calls, 3);
        assert_eq!(back.timeout_secs, 30);
        assert_eq!(back.messages.len(), 1);
        assert_eq!(back.tools.len(), 1);
    }

    #[test]
    fn test_agentic_response_roundtrip() {
        let resp = AgenticResponse {
            response: "Here is the time.".into(),
            execution_log: vec![ExecutionStep {
                step_type: "tool_call".into(),
                tool_name: Some("utc_now".into()),
                duration_ms: 42,
                status: "ok".into(),
                error_message: None,
            }],
            tokens_used: TokenUsage {
                prompt_tokens: 100,
                completion_tokens: 50,
                total_tokens: 150,
            },
            model_used: "claude-3-haiku".into(),
            tool_calls_made: 1,
            duration_ms: 500,
            security_events: vec![],
        };

        let json = serde_json::to_string(&resp).expect("serialize");
        let back: AgenticResponse = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.tool_calls_made, 1);
        assert_eq!(back.execution_log.len(), 1);
        assert_eq!(back.tokens_used.total_tokens, 150);
    }

    // ── Default values ────────────────────────────────────────────────────────

    #[test]
    fn test_default_max_tool_calls_is_5() {
        let json_str = r#"{
            "messages": [],
            "user_id": "test",
            "model": "test-model"
        }"#;
        let req: AgenticRequest = serde_json::from_str(json_str).expect("deserialize");
        assert_eq!(req.max_tool_calls, 5);
    }

    #[test]
    fn test_default_timeout_secs_is_60() {
        let json_str = r#"{
            "messages": [],
            "user_id": "test",
            "model": "test-model"
        }"#;
        let req: AgenticRequest = serde_json::from_str(json_str).expect("deserialize");
        assert_eq!(req.timeout_secs, 60);
    }

    #[test]
    fn test_default_permissions_is_empty() {
        let json_str = r#"{
            "messages": [],
            "user_id": "test",
            "model": "test-model"
        }"#;
        let req: AgenticRequest = serde_json::from_str(json_str).expect("deserialize");
        assert!(req.permissions.is_empty());
    }

    #[test]
    fn test_default_tools_is_empty() {
        let json_str = r#"{
            "messages": [],
            "user_id": "test",
            "model": "test-model"
        }"#;
        let req: AgenticRequest = serde_json::from_str(json_str).expect("deserialize");
        assert!(req.tools.is_empty());
    }

    // ── ExecutionStep — no args or results ────────────────────────────────────

    #[test]
    fn test_execution_step_has_no_args_field() {
        // Verify that ExecutionStep cannot carry tool arguments — the type itself
        // must not have an `args` or `arguments` or `result` field.
        let step = ExecutionStep {
            step_type: "tool_call".into(),
            tool_name: Some("utc_now".into()),
            duration_ms: 10,
            status: "ok".into(),
            error_message: None,
        };
        let json_val = serde_json::to_value(&step).expect("to_value");
        let obj = json_val.as_object().expect("object");
        assert!(!obj.contains_key("args"), "args must NOT be present");
        assert!(!obj.contains_key("arguments"), "arguments must NOT be present");
        assert!(!obj.contains_key("result"), "result must NOT be present");
        assert!(!obj.contains_key("output"), "output must NOT be present");
    }

    #[test]
    fn test_execution_step_serialises_only_metadata() {
        let step = ExecutionStep {
            step_type: "guard_block".into(),
            tool_name: Some("infisical_get_secret".into()),
            duration_ms: 0,
            status: "blocked".into(),
            error_message: Some("permission denied".into()),
        };
        let json_val = serde_json::to_value(&step).expect("to_value");
        let keys: Vec<&str> = json_val
            .as_object()
            .expect("object")
            .keys()
            .map(|s| s.as_str())
            .collect();
        // Only these metadata keys should appear
        for key in &keys {
            assert!(
                matches!(*key, "step_type" | "tool_name" | "duration_ms" | "status" | "error_message"),
                "unexpected key in ExecutionStep: {}",
                key
            );
        }
    }

    // ── Model override ────────────────────────────────────────────────────────

    #[test]
    fn test_model_override_optional() {
        let req = AgenticRequest {
            messages: vec![],
            system_prompt: String::new(),
            tools: vec![],
            permissions: vec![],
            max_tool_calls: 5,
            timeout_secs: 60,
            user_id: "u".into(),
            model: "haiku".into(),
            model_override: Some("opus".into()),
            stream: false,
        };
        let json = serde_json::to_string(&req).expect("serialize");
        let back: AgenticRequest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.model_override, Some("opus".into()));
    }

    #[test]
    fn test_model_override_absent_when_none() {
        let req = AgenticRequest {
            messages: vec![],
            system_prompt: String::new(),
            tools: vec![],
            permissions: vec![],
            max_tool_calls: 5,
            timeout_secs: 60,
            user_id: "u".into(),
            model: "haiku".into(),
            model_override: None,
            stream: false,
        };
        let json_val = serde_json::to_value(&req).expect("to_value");
        // model_override is skip_serializing_if(None) — should not appear
        assert!(
            !json_val.as_object().expect("object").contains_key("model_override"),
            "model_override should be absent when None"
        );
    }

    // ── TokenUsage default ────────────────────────────────────────────────────

    #[test]
    fn test_token_usage_default_zeros() {
        let usage = TokenUsage::default();
        assert_eq!(usage.prompt_tokens, 0);
        assert_eq!(usage.completion_tokens, 0);
        assert_eq!(usage.total_tokens, 0);
    }

    // ── max_tool_calls cap check (behaviour is enforced by loop_runner) ───────

    #[test]
    fn test_max_tool_calls_cap_10_is_documented() {
        // The AgenticRequest type allows up to 10 per the spec comment.
        // The loop_runner enforces the cap — this test just documents the intent.
        let req = AgenticRequest {
            messages: vec![],
            system_prompt: String::new(),
            tools: vec![],
            permissions: vec![],
            max_tool_calls: 10, // spec max
            timeout_secs: 60,
            user_id: "admin".into(),
            model: "opus".into(),
            model_override: None,
            stream: false,
        };
        assert_eq!(req.max_tool_calls, 10);
    }

    // ── No hardcoded IPs in any field name or default value ───────────────────

    #[test]
    fn test_no_hardcoded_ips_in_defaults() {
        let json_str = r#"{
            "messages": [],
            "user_id": "test",
            "model": "test-model"
        }"#;
        let req: AgenticRequest = serde_json::from_str(json_str).expect("deserialize");
        let serialised = serde_json::to_string(&req).expect("serialize");
        assert!(!serialised.contains("192.168"), "no hardcoded IPs");
        assert!(!serialised.contains("10.0."), "no hardcoded IPs");
        assert!(!serialised.contains("172.16"), "no hardcoded IPs");
    }
}
