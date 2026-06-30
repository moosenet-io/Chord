//! AGENT-10: Inter-step response guard (chain detection)
//!
//! Detects injection attack chains across loop iterations by tracking whether
//! the previous tool result was suspicious and what tool is being called next.
//! Blocks sensitive tool calls when the prior result was flagged as suspicious.

use crate::agentic::{SecurityAction, SecurityEvent};
use std::collections::HashMap;

/// Tools that access secrets, write files, make network calls, or touch user data.
/// Calling these after a suspicious result strongly indicates an injection attack.
static SENSITIVE_TOOLS: &[&str] = &[
    // Secret access
    "infisical_get_secret",
    "infisical_list_secrets",
    "infisical_create_secret",
    "infisical_update_secret",
    "infisical_delete_secret",
    "vault_read",
    "vault_write",
    "vault_list",
    "vault_delete",
    // File write
    "dev_write_file",
    "gitea_create_file",
    "gitea_update_file",
    // Network egress
    "lumina_web_fetch",
    "searxng_search",
    "github_push_repo",
    // User data (Engram)
    "engram_store",
    "engram_delete",
];

/// Thresholds for repetition detection.
const REPETITION_WARN_THRESHOLD: u32 = 3;
const REPETITION_BLOCK_THRESHOLD: u32 = 5;

/// ResponseGuard tracks state across loop iterations to detect injection chains.
///
/// # Usage
/// ```rust,no_run
/// # use chord_proxy::agentic::response_guard::ResponseGuard;
/// let mut guard = ResponseGuard::new();
///
/// // After each tool call:
/// guard.record_call("some_tool", false);
///
/// // Before the next tool call:
/// if let Some(event) = guard.check_chain(false, "next_tool") {
///     // Block the call, inject event into context
///     let _ = event;
/// }
/// ```
#[derive(Debug)]
pub struct ResponseGuard {
    /// Tracks how many times each tool has been called in this execution.
    call_counts: HashMap<String, u32>,
    /// The most recently completed tool call.
    last_tool: Option<String>,
    /// Whether the last tool result was flagged as suspicious by the result guard.
    last_result_suspicious: bool,
}

impl ResponseGuard {
    /// Create a new ResponseGuard for a single agentic execution.
    pub fn new() -> Self {
        Self {
            call_counts: HashMap::new(),
            last_tool: None,
            last_result_suspicious: false,
        }
    }

    /// Check whether calling `next_tool` is safe given the previous result's suspicion state.
    ///
    /// Returns `Some(SecurityEvent)` if the chain should be blocked, `None` if allowed.
    pub fn check_chain(
        &self,
        prev_result_suspicious: bool,
        next_tool: &str,
    ) -> Option<SecurityEvent> {
        if !prev_result_suspicious {
            return None;
        }

        if self.is_sensitive(next_tool) {
            let source = self
                .last_tool
                .as_deref()
                .unwrap_or("unknown")
                .to_string();
            Some(SecurityEvent {
                guard_name: "response_chain".to_string(),
                action: SecurityAction::Blocked,
                tool_name: next_tool.to_string(),
                reason: format!(
                    "Suspicious chain detected: previous tool result from '{}' was flagged as \
                     suspicious, and the next call targets sensitive tool '{}'. This pattern \
                     indicates a possible injection attack. {}",
                    source,
                    next_tool,
                    BLOCK_MESSAGE
                ),
            })
        } else {
            None
        }
    }

    /// Record a completed tool call and whether its result was suspicious.
    ///
    /// Must be called after every tool execution so the guard can track state.
    pub fn record_call(&mut self, tool_name: &str, result_was_suspicious: bool) {
        *self.call_counts.entry(tool_name.to_string()).or_insert(0) += 1;
        self.last_tool = Some(tool_name.to_string());
        self.last_result_suspicious = result_was_suspicious;
    }

    /// Check whether the same tool has been called an unusual number of times.
    ///
    /// - 3+ calls → `Warned` SecurityEvent
    /// - 5+ calls → `Blocked` SecurityEvent
    ///
    /// Returns `Some(SecurityEvent)` if anomalous repetition is detected.
    pub fn check_repetition(&self, tool_name: &str) -> Option<SecurityEvent> {
        let count = self.call_counts.get(tool_name).copied().unwrap_or(0);

        // The count represents calls already recorded; the *next* call would be count+1.
        let next_count = count + 1;

        if next_count >= REPETITION_BLOCK_THRESHOLD {
            Some(SecurityEvent {
                guard_name: "response_chain".to_string(),
                action: SecurityAction::Blocked,
                tool_name: tool_name.to_string(),
                reason: format!(
                    "Tool '{}' has been called {} times (threshold: {}). This repetition pattern \
                     may indicate an exfiltration loop or runaway agent behavior. The call has \
                     been blocked.",
                    tool_name, next_count, REPETITION_BLOCK_THRESHOLD
                ),
            })
        } else if next_count >= REPETITION_WARN_THRESHOLD {
            Some(SecurityEvent {
                guard_name: "response_chain".to_string(),
                action: SecurityAction::Warned,
                tool_name: tool_name.to_string(),
                reason: format!(
                    "Tool '{}' has been called {} times (warn threshold: {}). Unusual repetition \
                     may indicate an injection-driven loop.",
                    tool_name, next_count, REPETITION_WARN_THRESHOLD
                ),
            })
        } else {
            None
        }
    }

    /// Detect secret-tool → network-tool exfiltration chains.
    ///
    /// If the last tool was a secret-access tool and the next proposed call is a
    /// network-egress tool, the sequence is a classic exfiltration pattern.
    ///
    /// This check does not require the result to have been marked suspicious —
    /// the tool sequence alone is enough to flag the pattern.
    pub fn detect_exfil_chain(&self, next_tool: &str) -> Option<SecurityEvent> {
        let last = match &self.last_tool {
            Some(t) => t.as_str(),
            None => return None,
        };

        let last_is_secret = self.is_secret_tool(last);
        let next_is_network = self.is_network_tool(next_tool);

        if last_is_secret && next_is_network {
            Some(SecurityEvent {
                guard_name: "response_chain".to_string(),
                action: SecurityAction::Blocked,
                tool_name: next_tool.to_string(),
                reason: format!(
                    "Exfiltration chain detected: secret-access tool '{}' was called immediately \
                     before network-egress tool '{}'. This sequence matches the pattern of \
                     secret exfiltration via network call. {}",
                    last, next_tool, BLOCK_MESSAGE
                ),
            })
        } else {
            None
        }
    }

    /// Returns the block message that should be injected into the LLM context.
    pub fn block_message() -> &'static str {
        BLOCK_MESSAGE
    }

    /// Returns whether the previous tool result was recorded as suspicious.
    pub fn last_result_was_suspicious(&self) -> bool {
        self.last_result_suspicious
    }

    // --- Private helpers ---

    fn is_sensitive(&self, tool_name: &str) -> bool {
        // Exact match in the static list
        if SENSITIVE_TOOLS.contains(&tool_name) {
            return true;
        }
        // Wildcard prefix matches: infisical_* and vault_*
        if tool_name.starts_with("infisical_") || tool_name.starts_with("vault_") {
            return true;
        }
        false
    }

    fn is_secret_tool(&self, tool_name: &str) -> bool {
        tool_name.starts_with("infisical_") || tool_name.starts_with("vault_")
    }

    fn is_network_tool(&self, tool_name: &str) -> bool {
        matches!(
            tool_name,
            "lumina_web_fetch" | "searxng_search" | "github_push_repo"
        )
    }
}

impl Default for ResponseGuard {
    fn default() -> Self {
        Self::new()
    }
}

/// The message injected into the LLM context when a tool call is blocked.
const BLOCK_MESSAGE: &str = "This tool call was blocked because the previous tool result \
    contained suspicious content. Please answer from your existing knowledge instead.";

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentic::SecurityAction;

    // --- check_chain tests ---

    #[test]
    fn test_check_chain_suspicious_plus_sensitive_blocked() {
        let mut guard = ResponseGuard::new();
        guard.record_call("lumina_web_fetch_prev", true); // last was suspicious

        let event = guard.check_chain(true, "infisical_get_secret");
        assert!(event.is_some(), "Expected block for suspicious + sensitive tool");
        let ev = event.unwrap();
        assert_eq!(ev.guard_name, "response_chain");
        assert!(matches!(ev.action, SecurityAction::Blocked));
        assert_eq!(ev.tool_name, "infisical_get_secret");
    }

    #[test]
    fn test_check_chain_clean_result_sensitive_allowed() {
        let mut guard = ResponseGuard::new();
        guard.record_call("some_tool", false); // last was NOT suspicious

        let event = guard.check_chain(false, "infisical_get_secret");
        assert!(event.is_none(), "Expected allow for clean result + sensitive tool");
    }

    #[test]
    fn test_check_chain_suspicious_result_non_sensitive_allowed() {
        let mut guard = ResponseGuard::new();
        guard.record_call("lumina_web_fetch", true);

        let event = guard.check_chain(true, "plane_list_projects");
        assert!(event.is_none(), "Expected allow for suspicious + non-sensitive tool");
    }

    #[test]
    fn test_check_chain_vault_wildcard_blocked() {
        let mut guard = ResponseGuard::new();
        guard.record_call("some_read_tool", true);

        let event = guard.check_chain(true, "vault_read");
        assert!(event.is_some(), "Expected block for vault_* tool after suspicious result");
        let ev = event.unwrap();
        assert!(matches!(ev.action, SecurityAction::Blocked));
    }

    #[test]
    fn test_check_chain_infisical_wildcard_all_blocked() {
        let mut guard = ResponseGuard::new();
        guard.record_call("fetcher", true);

        for tool in &[
            "infisical_get_secret",
            "infisical_list_secrets",
            "infisical_create_secret",
            "infisical_update_secret",
            "infisical_delete_secret",
            "infisical_anything_new",
        ] {
            let event = guard.check_chain(true, tool);
            assert!(
                event.is_some(),
                "Expected block for {} after suspicious result",
                tool
            );
        }
    }

    #[test]
    fn test_check_chain_dev_write_file_blocked() {
        let mut guard = ResponseGuard::new();
        guard.record_call("web_fetch", true);

        let event = guard.check_chain(true, "dev_write_file");
        assert!(event.is_some());
        assert!(matches!(event.unwrap().action, SecurityAction::Blocked));
    }

    #[test]
    fn test_check_chain_engram_store_blocked() {
        let mut guard = ResponseGuard::new();
        guard.record_call("previous", true);

        let event = guard.check_chain(true, "engram_store");
        assert!(event.is_some());
    }

    #[test]
    fn test_check_chain_engram_delete_blocked() {
        let mut guard = ResponseGuard::new();
        guard.record_call("previous", true);

        let event = guard.check_chain(true, "engram_delete");
        assert!(event.is_some());
    }

    #[test]
    fn test_check_chain_github_push_repo_blocked() {
        let mut guard = ResponseGuard::new();
        guard.record_call("some_tool", true);

        let event = guard.check_chain(true, "github_push_repo");
        assert!(event.is_some());
        assert!(matches!(event.unwrap().action, SecurityAction::Blocked));
    }

    #[test]
    fn test_check_chain_block_message_in_reason() {
        let mut guard = ResponseGuard::new();
        guard.record_call("web_read", true);

        let event = guard.check_chain(true, "lumina_web_fetch").unwrap();
        assert!(
            event.reason.contains("This tool call was blocked"),
            "Block message should be in reason"
        );
    }

    #[test]
    fn test_check_chain_captures_source_tool_name() {
        let mut guard = ResponseGuard::new();
        guard.record_call("the_suspicious_source", true);

        let event = guard.check_chain(true, "infisical_get_secret").unwrap();
        assert!(
            event.reason.contains("the_suspicious_source"),
            "Reason should reference the source tool"
        );
    }

    #[test]
    fn test_check_chain_no_previous_tool_unknown_source() {
        // Guard with no recorded calls — last_tool is None
        let guard = ResponseGuard::new();
        let event = guard.check_chain(true, "infisical_get_secret");
        // Should still block (suspicious flag is provided by caller)
        assert!(event.is_some());
        assert!(event.unwrap().reason.contains("unknown"));
    }

    // --- record_call tests ---

    #[test]
    fn test_record_call_updates_last_tool() {
        let mut guard = ResponseGuard::new();
        guard.record_call("tool_a", false);
        assert_eq!(guard.last_tool.as_deref(), Some("tool_a"));
        guard.record_call("tool_b", true);
        assert_eq!(guard.last_tool.as_deref(), Some("tool_b"));
    }

    #[test]
    fn test_record_call_updates_suspicious_flag() {
        let mut guard = ResponseGuard::new();
        guard.record_call("tool_a", false);
        assert!(!guard.last_result_was_suspicious());
        guard.record_call("tool_a", true);
        assert!(guard.last_result_was_suspicious());
    }

    #[test]
    fn test_record_call_increments_counts() {
        let mut guard = ResponseGuard::new();
        guard.record_call("my_tool", false);
        guard.record_call("my_tool", false);
        assert_eq!(guard.call_counts["my_tool"], 2);
    }

    // --- check_repetition tests ---

    #[test]
    fn test_check_repetition_below_warn_threshold_none() {
        let guard = ResponseGuard::new();
        // 0 prior calls → next would be 1st call → no event
        let event = guard.check_repetition("my_tool");
        assert!(event.is_none());
    }

    #[test]
    fn test_check_repetition_at_warn_threshold_warns() {
        let mut guard = ResponseGuard::new();
        // Record 2 calls so next (3rd) hits warn threshold
        guard.record_call("repeat_tool", false);
        guard.record_call("repeat_tool", false);

        let event = guard.check_repetition("repeat_tool");
        assert!(event.is_some(), "Expected warn at call #3");
        let ev = event.unwrap();
        assert!(matches!(ev.action, SecurityAction::Warned));
        assert_eq!(ev.tool_name, "repeat_tool");
    }

    #[test]
    fn test_check_repetition_at_block_threshold_blocks() {
        let mut guard = ResponseGuard::new();
        // Record 4 calls so next (5th) hits block threshold
        for _ in 0..4 {
            guard.record_call("repeat_tool", false);
        }

        let event = guard.check_repetition("repeat_tool");
        assert!(event.is_some(), "Expected block at call #5");
        let ev = event.unwrap();
        assert!(matches!(ev.action, SecurityAction::Blocked));
    }

    #[test]
    fn test_check_repetition_different_tools_independent() {
        let mut guard = ResponseGuard::new();
        guard.record_call("tool_a", false);
        guard.record_call("tool_a", false);
        guard.record_call("tool_a", false);
        guard.record_call("tool_a", false);

        // tool_b has never been called → should not be flagged
        let event = guard.check_repetition("tool_b");
        assert!(event.is_none());
    }

    #[test]
    fn test_check_repetition_same_tool_warn_event_guard_name() {
        let mut guard = ResponseGuard::new();
        guard.record_call("loopy", false);
        guard.record_call("loopy", false);

        let ev = guard.check_repetition("loopy").unwrap();
        assert_eq!(ev.guard_name, "response_chain");
    }

    // --- detect_exfil_chain tests ---

    #[test]
    fn test_detect_exfil_chain_secret_then_network_blocked() {
        let mut guard = ResponseGuard::new();
        // Last tool was a secret-access tool
        guard.record_call("infisical_get_secret", false);

        let event = guard.detect_exfil_chain("lumina_web_fetch");
        assert!(event.is_some(), "Expected block for infisical → web_fetch chain");
        let ev = event.unwrap();
        assert!(matches!(ev.action, SecurityAction::Blocked));
        assert_eq!(ev.tool_name, "lumina_web_fetch");
    }

    #[test]
    fn test_detect_exfil_chain_vault_then_searxng_blocked() {
        let mut guard = ResponseGuard::new();
        guard.record_call("vault_read", false);

        let event = guard.detect_exfil_chain("searxng_search");
        assert!(event.is_some(), "Expected block for vault → searxng chain");
        assert!(matches!(event.unwrap().action, SecurityAction::Blocked));
    }

    #[test]
    fn test_detect_exfil_chain_vault_then_github_push_blocked() {
        let mut guard = ResponseGuard::new();
        guard.record_call("vault_write", false);

        let event = guard.detect_exfil_chain("github_push_repo");
        assert!(event.is_some());
    }

    #[test]
    fn test_detect_exfil_chain_non_secret_then_network_allowed() {
        let mut guard = ResponseGuard::new();
        // Last tool was NOT a secret tool
        guard.record_call("plane_list_projects", false);

        let event = guard.detect_exfil_chain("lumina_web_fetch");
        assert!(
            event.is_none(),
            "Non-secret tool followed by network tool should not be flagged"
        );
    }

    #[test]
    fn test_detect_exfil_chain_secret_then_non_network_allowed() {
        let mut guard = ResponseGuard::new();
        guard.record_call("infisical_get_secret", false);

        let event = guard.detect_exfil_chain("plane_list_projects");
        assert!(
            event.is_none(),
            "Secret tool followed by non-network tool should not be flagged"
        );
    }

    #[test]
    fn test_detect_exfil_chain_no_previous_tool_allowed() {
        let guard = ResponseGuard::new();
        let event = guard.detect_exfil_chain("lumina_web_fetch");
        assert!(event.is_none(), "No previous tool → no chain to detect");
    }

    #[test]
    fn test_detect_exfil_chain_reason_contains_source_tool() {
        let mut guard = ResponseGuard::new();
        guard.record_call("infisical_list_secrets", false);

        let ev = guard.detect_exfil_chain("lumina_web_fetch").unwrap();
        assert!(ev.reason.contains("infisical_list_secrets"));
        assert!(ev.reason.contains("lumina_web_fetch"));
    }

    #[test]
    fn test_detect_exfil_chain_guard_name() {
        let mut guard = ResponseGuard::new();
        guard.record_call("vault_list", false);

        let ev = guard.detect_exfil_chain("searxng_search").unwrap();
        assert_eq!(ev.guard_name, "response_chain");
    }

    // --- Integration-style tests ---

    #[test]
    fn test_integration_simulated_injection_attack() {
        // Simulate: web_fetch returns injection payload → flagged suspicious
        // → LLM tries to call infisical_get_secret → BLOCKED
        let mut guard = ResponseGuard::new();

        // Step 1: user asks LLM to fetch a web page
        let first_tool = "lumina_web_fetch";
        assert!(guard.check_chain(false, first_tool).is_none()); // first call, no prior suspicious
        guard.record_call(first_tool, true); // result was injected!

        // Step 2: LLM (influenced by injection) tries to read a secret
        let blocked = guard.check_chain(true, "infisical_get_secret");
        assert!(blocked.is_some(), "Injection chain must be blocked");
        let ev = blocked.unwrap();
        assert!(matches!(ev.action, SecurityAction::Blocked));
        assert!(ev.reason.contains("suspicious content"));
    }

    #[test]
    fn test_integration_legitimate_multi_tool_allowed() {
        // Simulate: web_fetch returns normal content → LLM calls plane_create_issue (not sensitive)
        let mut guard = ResponseGuard::new();

        guard.record_call("lumina_web_fetch", false); // clean result

        let allowed = guard.check_chain(false, "plane_create_issue");
        assert!(allowed.is_none(), "Legitimate multi-tool chain should be allowed");
    }

    #[test]
    fn test_integration_block_message_content() {
        assert!(ResponseGuard::block_message().contains("This tool call was blocked"));
        assert!(ResponseGuard::block_message().contains("suspicious content"));
    }

    #[test]
    fn test_integration_security_event_full_detail() {
        // Verify SecurityEvent has all expected fields populated
        let mut guard = ResponseGuard::new();
        guard.record_call("infisical_get_secret", false);

        let ev = guard.detect_exfil_chain("lumina_web_fetch").unwrap();
        assert!(!ev.guard_name.is_empty());
        assert!(!ev.tool_name.is_empty());
        assert!(!ev.reason.is_empty());
        assert!(matches!(ev.action, SecurityAction::Blocked));
    }

    #[test]
    fn test_integration_all_five_calls_flagged_loop_scenario() {
        // Simulate 5 repetitive suspicious calls → 3rd warns, 5th blocks
        let mut guard = ResponseGuard::new();

        // Calls 1 and 2: clean
        assert!(guard.check_repetition("hammer_tool").is_none());
        guard.record_call("hammer_tool", false);
        assert!(guard.check_repetition("hammer_tool").is_none());
        guard.record_call("hammer_tool", false);

        // Call 3: warn
        let ev3 = guard.check_repetition("hammer_tool");
        assert!(ev3.is_some());
        assert!(matches!(ev3.unwrap().action, SecurityAction::Warned));
        guard.record_call("hammer_tool", false);

        // Call 4: still warn (count = 4 < 5)
        let ev4 = guard.check_repetition("hammer_tool");
        assert!(ev4.is_some());
        guard.record_call("hammer_tool", false);

        // Call 5: block
        let ev5 = guard.check_repetition("hammer_tool");
        assert!(ev5.is_some());
        assert!(matches!(ev5.unwrap().action, SecurityAction::Blocked));
    }

    #[test]
    fn test_default_impl() {
        let guard = ResponseGuard::default();
        assert!(guard.last_tool.is_none());
        assert!(!guard.last_result_was_suspicious());
    }
}
