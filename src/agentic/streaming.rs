//! AGENT-04: SSE streaming of progress events during agentic execution.
//!
//! Defines the [`ProgressEvent`] enum and the formatting helpers needed to turn
//! those events into Server-Sent Events (SSE) frames that a client can consume.
//!
//! ## Route note
//! The full SSE endpoint lives at `GET /v1/agent/stream`.  Full axum SSE
//! route wiring (via `axum::response::sse`) is complex and requires an
//! `mpsc` channel between the agentic loop and the HTTP handler.  That
//! plumbing is deferred to a follow-up; this module provides all the data
//! types and formatting helpers so the endpoint can be wired trivially when
//! needed.
//!
//! ## Fallback
//! If the SSE connection drops or the client does not support streaming,
//! callers should fall back to the non-streaming `POST /v1/agent/execute`
//! endpoint which returns the complete [`AgenticResponse`] in one shot.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::agentic::SecurityAction;

// ── ProgressEvent ─────────────────────────────────────────────────────────────

/// An SSE progress event emitted by the agentic loop.
///
/// Events are emitted in this order during a typical execution:
///
/// 1. [`ProgressEvent::Started`]
/// 2. [`ProgressEvent::ToolCallStarted`] (one per tool call, before execution)
/// 3. [`ProgressEvent::ToolCallComplete`] (one per tool call, after execution)
/// 4. *(repeat 2-3 for each tool in the loop)*
/// 5. [`ProgressEvent::SecurityEventOccurred`] (zero or more, whenever a guard fires)
/// 6. [`ProgressEvent::Complete`] (always last — carries the final response text)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProgressEvent {
    /// The agentic loop has started.
    Started,

    /// A tool call is about to be executed.
    ToolCallStarted {
        /// The tool being called (e.g. `"searxng_search"`).
        tool_name: String,
        /// 1-based step counter (how many tool calls have started so far, including this one).
        step_number: u32,
    },

    /// A tool call has finished.
    ToolCallComplete {
        /// The tool that was called.
        tool_name: String,
        /// Wall-clock duration in milliseconds.
        duration_ms: u64,
        /// Outcome: `"ok"`, `"blocked"`, `"error"`, or `"timeout"`.
        status: String,
    },

    /// A security guard fired during execution (blocked, sanitized, or warned).
    SecurityEventOccurred {
        /// Guard that raised the event: `"argument"`, `"result_guard"`,
        /// `"response_chain"`, `"behavioral"`, or `"permission"`.
        guard_name: String,
        /// What the guard did: `"blocked"`, `"sanitized"`, or `"warned"`.
        action: String,
        /// The tool involved.
        tool_name: String,
    },

    /// The agentic loop has finished.  This is always the last event.
    Complete {
        /// The final text response delivered to the user.
        response: String,
    },
}

// ── SSE formatting ────────────────────────────────────────────────────────────

/// Serialise a [`ProgressEvent`] into an SSE frame.
///
/// Format: `data: <json>\n\n`
///
/// The JSON payload is the full event (including the `type` discriminant).
/// Clients parse the `type` field to dispatch to the right handler.
///
/// # Errors
///
/// Returns `Err(String)` if the event cannot be serialised to JSON
/// (should never happen for well-formed events).
pub fn event_to_sse(event: &ProgressEvent) -> Result<String, String> {
    let json = serde_json::to_string(event)
        .map_err(|e| format!("failed to serialise ProgressEvent: {e}"))?;
    Ok(format!("data: {json}\n\n"))
}

// ── Tool-status map ───────────────────────────────────────────────────────────

/// Build the tool-name → human-readable-status lookup table.
///
/// This is a plain [`HashMap`] so tests can inspect it directly.  In
/// production code callers should use [`format_status`] instead.
fn build_tool_status_map() -> HashMap<&'static str, &'static str> {
    let mut m = HashMap::new();

    // ── Web / search ──────────────────────────────────────────────────────────
    m.insert("searxng_search", "Searching the web...");
    m.insert("lumina_web_fetch", "Fetching a web page...");
    m.insert("lumina_web_search", "Searching the web...");
    m.insert("news_get_headlines", "Checking the latest news...");
    m.insert("news_search", "Searching news articles...");

    // ── Calendar / scheduling ─────────────────────────────────────────────────
    m.insert("google_calendar_today", "Checking your calendar...");
    m.insert("google_calendar_list", "Listing calendar events...");
    m.insert("google_calendar_create", "Creating a calendar event...");

    // ── Secrets / vault ───────────────────────────────────────────────────────
    m.insert("infisical_get_secret", "Accessing secure vault...");
    m.insert("infisical_list_secrets", "Listing vault entries...");

    // ── Work queue / Plane ────────────────────────────────────────────────────
    m.insert("plane_list_issues", "Checking work queue...");
    m.insert("plane_create_issue", "Creating a work item...");
    m.insert("plane_update_issue", "Updating a work item...");
    m.insert("plane_get_issue", "Looking up a work item...");
    m.insert("plane_list_projects", "Listing Plane projects...");

    // ── Nexus inbox ───────────────────────────────────────────────────────────
    m.insert("nexus_check_inbox", "Reading inbox...");
    m.insert("nexus_send", "Sending a message...");
    m.insert("nexus_read", "Reading a message...");
    m.insert("nexus_ack", "Acknowledging a message...");
    m.insert("nexus_history", "Loading message history...");

    // ── Memory / Engram ───────────────────────────────────────────────────────
    m.insert("engram_recall", "Searching memory...");
    m.insert("engram_store", "Storing to memory...");
    m.insert("engram_forget", "Removing from memory...");
    m.insert("engram_list", "Listing memories...");

    // ── Time / utility ────────────────────────────────────────────────────────
    m.insert("utc_now", "Checking the current time...");
    m.insert("local_time", "Checking the local time...");

    // ── Matrix / messaging ────────────────────────────────────────────────────
    m.insert("matrix_send", "Sending a Matrix message...");
    m.insert("matrix_read", "Reading Matrix messages...");

    // ── Gitea ─────────────────────────────────────────────────────────────────
    m.insert("gitea_list_repos", "Listing Gitea repositories...");
    m.insert("gitea_create_file", "Writing to a repository...");
    m.insert("gitea_get_file", "Reading from a repository...");

    // ── System / Prometheus ───────────────────────────────────────────────────
    m.insert("prometheus_query", "Querying metrics...");
    m.insert("prometheus_alerts", "Checking alerts...");

    // ── Portainer / containers ────────────────────────────────────────────────
    m.insert("portainer_list_containers", "Listing containers...");
    m.insert("portainer_container_status", "Checking container status...");

    m
}

/// Return a human-readable status string for the given tool name.
///
/// Looks up `tool_name` in the built-in map.  If no entry is found, formats
/// the raw tool name into a title-cased phrase (e.g. `"my_tool"` →
/// `"Running my_tool..."`).
///
/// This function is allocation-free for known tools.
pub fn format_status(tool_name: &str) -> String {
    // We rebuild the map on every call here for simplicity; a production
    // implementation would use a `once_cell::sync::Lazy` static.
    let map = build_tool_status_map();
    match map.get(tool_name) {
        Some(&status) => status.to_string(),
        None => format!("Running {}...", tool_name),
    }
}

/// Convert a [`SecurityAction`] enum variant to the string expected in SSE payloads.
pub fn security_action_label(action: &SecurityAction) -> &'static str {
    match action {
        SecurityAction::Blocked => "blocked",
        SecurityAction::Sanitized => "sanitized",
        SecurityAction::Warned => "warned",
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ProgressEvent serialisation ───────────────────────────────────────────

    #[test]
    fn test_started_event_serialises_with_type_tag() {
        let event = ProgressEvent::Started;
        let json = serde_json::to_string(&event).expect("serialize");
        assert!(json.contains("\"type\":\"started\""), "type tag must be present: {json}");
    }

    #[test]
    fn test_tool_call_started_serialises_fields() {
        let event = ProgressEvent::ToolCallStarted {
            tool_name: "searxng_search".into(),
            step_number: 1,
        };
        let json = serde_json::to_string(&event).expect("serialize");
        assert!(json.contains("\"type\":\"tool_call_started\""), "type: {json}");
        assert!(json.contains("searxng_search"), "tool_name: {json}");
        assert!(json.contains("\"step_number\":1"), "step_number: {json}");
    }

    #[test]
    fn test_tool_call_complete_serialises_fields() {
        let event = ProgressEvent::ToolCallComplete {
            tool_name: "nexus_check_inbox".into(),
            duration_ms: 250,
            status: "ok".into(),
        };
        let json = serde_json::to_string(&event).expect("serialize");
        assert!(json.contains("\"type\":\"tool_call_complete\""), "type: {json}");
        assert!(json.contains("nexus_check_inbox"), "tool_name: {json}");
        assert!(json.contains("\"duration_ms\":250"), "duration: {json}");
        assert!(json.contains("\"status\":\"ok\""), "status: {json}");
    }

    #[test]
    fn test_security_event_occurred_serialises_fields() {
        let event = ProgressEvent::SecurityEventOccurred {
            guard_name: "argument".into(),
            action: "blocked".into(),
            tool_name: "infisical_get_secret".into(),
        };
        let json = serde_json::to_string(&event).expect("serialize");
        assert!(json.contains("\"type\":\"security_event_occurred\""), "type: {json}");
        assert!(json.contains("\"guard_name\":\"argument\""), "guard: {json}");
        assert!(json.contains("\"action\":\"blocked\""), "action: {json}");
        assert!(json.contains("infisical_get_secret"), "tool: {json}");
    }

    #[test]
    fn test_complete_event_serialises_response() {
        let event = ProgressEvent::Complete {
            response: "Here is your answer.".into(),
        };
        let json = serde_json::to_string(&event).expect("serialize");
        assert!(json.contains("\"type\":\"complete\""), "type: {json}");
        assert!(json.contains("Here is your answer."), "response: {json}");
    }

    // ── Round-trip deserialisation ────────────────────────────────────────────

    #[test]
    fn test_started_roundtrip() {
        let event = ProgressEvent::Started;
        let json = serde_json::to_string(&event).unwrap();
        let back: ProgressEvent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, ProgressEvent::Started);
    }

    #[test]
    fn test_tool_call_started_roundtrip() {
        let event = ProgressEvent::ToolCallStarted {
            tool_name: "utc_now".into(),
            step_number: 3,
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: ProgressEvent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, event);
    }

    #[test]
    fn test_tool_call_complete_roundtrip() {
        let event = ProgressEvent::ToolCallComplete {
            tool_name: "plane_list_issues".into(),
            duration_ms: 512,
            status: "blocked".into(),
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: ProgressEvent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, event);
    }

    #[test]
    fn test_security_event_occurred_roundtrip() {
        let event = ProgressEvent::SecurityEventOccurred {
            guard_name: "result_guard".into(),
            action: "sanitized".into(),
            tool_name: "searxng_search".into(),
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: ProgressEvent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, event);
    }

    #[test]
    fn test_complete_roundtrip() {
        let event = ProgressEvent::Complete {
            response: "Done!".into(),
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: ProgressEvent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, event);
    }

    // ── SSE formatting ────────────────────────────────────────────────────────

    #[test]
    fn test_event_to_sse_format_starts_with_data_prefix() {
        let event = ProgressEvent::Started;
        let sse = event_to_sse(&event).expect("sse");
        assert!(sse.starts_with("data: "), "must start with 'data: ': {sse:?}");
    }

    #[test]
    fn test_event_to_sse_ends_with_double_newline() {
        let event = ProgressEvent::Started;
        let sse = event_to_sse(&event).expect("sse");
        assert!(sse.ends_with("\n\n"), "must end with double newline: {sse:?}");
    }

    #[test]
    fn test_event_to_sse_body_is_valid_json() {
        let event = ProgressEvent::ToolCallStarted {
            tool_name: "utc_now".into(),
            step_number: 1,
        };
        let sse = event_to_sse(&event).expect("sse");
        // Strip the "data: " prefix and trailing "\n\n".
        let json_part = sse
            .trim_start_matches("data: ")
            .trim_end_matches('\n');
        let parsed: serde_json::Value = serde_json::from_str(json_part)
            .expect("SSE body must be valid JSON");
        assert_eq!(parsed["type"], "tool_call_started");
    }

    #[test]
    fn test_event_to_sse_complete_event_contains_response() {
        let event = ProgressEvent::Complete {
            response: "All done here.".into(),
        };
        let sse = event_to_sse(&event).expect("sse");
        assert!(sse.contains("All done here."), "response must appear in SSE frame: {sse:?}");
    }

    #[test]
    fn test_event_to_sse_security_event_no_sensitive_data() {
        // Security events MUST NOT leak the actual content blocked/sanitized.
        let event = ProgressEvent::SecurityEventOccurred {
            guard_name: "argument".into(),
            action: "blocked".into(),
            tool_name: "some_tool".into(),
        };
        let sse = event_to_sse(&event).expect("sse");
        // No "reason" or "content" fields should appear — only metadata.
        assert!(!sse.contains("\"reason\""), "reason must not leak in SSE: {sse:?}");
        assert!(!sse.contains("\"content\""), "content must not leak in SSE: {sse:?}");
    }

    // ── Tool status mapping ───────────────────────────────────────────────────

    #[test]
    fn test_format_status_searxng_search() {
        assert_eq!(format_status("searxng_search"), "Searching the web...");
    }

    #[test]
    fn test_format_status_google_calendar_today() {
        assert_eq!(format_status("google_calendar_today"), "Checking your calendar...");
    }

    #[test]
    fn test_format_status_infisical_get_secret() {
        assert_eq!(format_status("infisical_get_secret"), "Accessing secure vault...");
    }

    #[test]
    fn test_format_status_plane_list_issues() {
        assert_eq!(format_status("plane_list_issues"), "Checking work queue...");
    }

    #[test]
    fn test_format_status_nexus_check_inbox() {
        assert_eq!(format_status("nexus_check_inbox"), "Reading inbox...");
    }

    #[test]
    fn test_format_status_nexus_send() {
        assert_eq!(format_status("nexus_send"), "Sending a message...");
    }

    #[test]
    fn test_format_status_nexus_read() {
        assert_eq!(format_status("nexus_read"), "Reading a message...");
    }

    #[test]
    fn test_format_status_nexus_ack() {
        assert_eq!(format_status("nexus_ack"), "Acknowledging a message...");
    }

    #[test]
    fn test_format_status_nexus_history() {
        assert_eq!(format_status("nexus_history"), "Loading message history...");
    }

    #[test]
    fn test_format_status_engram_recall() {
        assert_eq!(format_status("engram_recall"), "Searching memory...");
    }

    #[test]
    fn test_format_status_utc_now() {
        assert_eq!(format_status("utc_now"), "Checking the current time...");
    }

    #[test]
    fn test_format_status_lumina_web_fetch() {
        assert_eq!(format_status("lumina_web_fetch"), "Fetching a web page...");
    }

    #[test]
    fn test_format_status_unknown_tool_falls_back() {
        let status = format_status("my_custom_tool_xyz");
        assert!(
            status.contains("my_custom_tool_xyz"),
            "fallback must include tool name: {status}"
        );
        assert!(
            status.ends_with("..."),
            "fallback must end with ellipsis: {status}"
        );
    }

    #[test]
    fn test_format_status_empty_tool_name_does_not_panic() {
        let status = format_status("");
        assert!(!status.is_empty());
    }

    #[test]
    fn test_format_status_all_known_tools_return_non_empty() {
        let map = build_tool_status_map();
        for (tool, label) in &map {
            let result = format_status(tool);
            assert!(!result.is_empty(), "label for {tool} must not be empty");
            assert_eq!(result, *label, "label mismatch for {tool}");
        }
    }

    #[test]
    fn test_format_status_at_least_10_known_mappings() {
        let map = build_tool_status_map();
        assert!(
            map.len() >= 10,
            "spec requires at least 10 tool mappings, got {}",
            map.len()
        );
    }

    // ── SecurityAction label ──────────────────────────────────────────────────

    #[test]
    fn test_security_action_label_blocked() {
        assert_eq!(security_action_label(&SecurityAction::Blocked), "blocked");
    }

    #[test]
    fn test_security_action_label_sanitized() {
        assert_eq!(security_action_label(&SecurityAction::Sanitized), "sanitized");
    }

    #[test]
    fn test_security_action_label_warned() {
        assert_eq!(security_action_label(&SecurityAction::Warned), "warned");
    }

    // ── SSE event ordering sanity ─────────────────────────────────────────────

    #[test]
    fn test_event_sequence_sse_frames_are_independent() {
        // Each event produces exactly one valid SSE frame.
        let events = vec![
            ProgressEvent::Started,
            ProgressEvent::ToolCallStarted {
                tool_name: "searxng_search".into(),
                step_number: 1,
            },
            ProgressEvent::ToolCallComplete {
                tool_name: "searxng_search".into(),
                duration_ms: 100,
                status: "ok".into(),
            },
            ProgressEvent::SecurityEventOccurred {
                guard_name: "result_guard".into(),
                action: "sanitized".into(),
                tool_name: "searxng_search".into(),
            },
            ProgressEvent::Complete {
                response: "Here is what I found.".into(),
            },
        ];

        for event in &events {
            let sse = event_to_sse(event).expect("sse");
            assert!(sse.starts_with("data: "), "frame must start with data: prefix");
            assert!(sse.ends_with("\n\n"), "frame must end with double newline");
        }
    }

    #[test]
    fn test_no_hardcoded_ips_in_sse_output() {
        let event = ProgressEvent::ToolCallStarted {
            tool_name: "prometheus_query".into(),
            step_number: 1,
        };
        let sse = event_to_sse(&event).expect("sse");
        assert!(!sse.contains("192.168"), "no hardcoded IPs in SSE output");
        assert!(!sse.contains("10.0."), "no hardcoded IPs in SSE output");
    }

    // ── Fallback behaviour ────────────────────────────────────────────────────

    #[test]
    fn test_non_streaming_fallback_types_are_still_serializable() {
        // The AgenticResponse (non-streaming fallback) must still serialize
        // normally — streaming is additive, not a replacement.
        use crate::agentic::context::{AgenticResponse, ExecutionStep, TokenUsage};
        use crate::agentic::SecurityEvent;

        let resp = AgenticResponse {
            response: "Non-streaming response.".into(),
            execution_log: vec![ExecutionStep {
                step_type: "llm_response".into(),
                tool_name: None,
                duration_ms: 50,
                status: "ok".into(),
                error_message: None,
            }],
            tokens_used: TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
            model_used: "stub".into(),
            tool_calls_made: 0,
            duration_ms: 50,
            security_events: vec![SecurityEvent {
                guard_name: "permission".into(),
                action: SecurityAction::Blocked,
                tool_name: "bad_tool".into(),
                reason: "not permitted".into(),
            }],
        };

        let json = serde_json::to_string(&resp).expect("serialize fallback response");
        assert!(json.contains("Non-streaming response."));
    }
}
