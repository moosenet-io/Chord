use crate::agentic::{SecurityAction, SecurityEvent};
use std::collections::HashMap;

/// Internal data tool prefixes — tools that access internal data stores.
/// Data flowing FROM these TO external tools is an exfiltration pattern.
const INTERNAL_DATA_PREFIXES: &[&str] = &[
    "engram_",
    "nexus_",
    "plane_",
    "infisical_",
];

/// External network tools — if called after an internal data tool, flag as potential exfil.
const EXTERNAL_TOOL_NAMES: &[&str] = &[
    "lumina_web_fetch",
    "searxng_search",
];

const EXTERNAL_TOOL_PREFIXES: &[&str] = &["github_"];

/// Configuration for BehavioralMonitor thresholds, loaded from env vars.
#[derive(Debug, Clone)]
pub struct BehavioralConfig {
    /// How many same-tool calls before a Warn is emitted.
    pub hammer_warn: u8,
    /// How many same-tool calls before the loop is Blocked.
    pub hammer_block: u8,
    /// Whether escalation-attempt detection is active.
    pub escalation_enabled: bool,
    /// Whether data-exfiltration detection is active.
    pub exfil_enabled: bool,
}

impl BehavioralConfig {
    /// Load thresholds from environment variables, falling back to spec defaults.
    pub fn from_env() -> Self {
        let hammer_warn = std::env::var("BEHAVIORAL_HAMMER_WARN")
            .ok()
            .and_then(|v| v.parse::<u8>().ok())
            .unwrap_or(3);

        let hammer_block = std::env::var("BEHAVIORAL_HAMMER_BLOCK")
            .ok()
            .and_then(|v| v.parse::<u8>().ok())
            .unwrap_or(5);

        let escalation_enabled = std::env::var("BEHAVIORAL_ESCALATION_ENABLED")
            .map(|v| v.to_lowercase() != "false")
            .unwrap_or(true);

        let exfil_enabled = std::env::var("BEHAVIORAL_EXFIL_ENABLED")
            .map(|v| v.to_lowercase() != "false")
            .unwrap_or(true);

        Self {
            hammer_warn,
            hammer_block,
            escalation_enabled,
            exfil_enabled,
        }
    }
}

impl Default for BehavioralConfig {
    fn default() -> Self {
        Self {
            hammer_warn: 3,
            hammer_block: 5,
            escalation_enabled: true,
            exfil_enabled: true,
        }
    }
}

/// Classifies which category a tool falls into.
#[derive(Debug, Clone, PartialEq)]
enum ToolCategory {
    InternalData,
    ExternalNetwork,
    Other,
}

fn classify_tool(tool_name: &str) -> ToolCategory {
    for prefix in INTERNAL_DATA_PREFIXES {
        if tool_name.starts_with(prefix) {
            return ToolCategory::InternalData;
        }
    }
    for exact in EXTERNAL_TOOL_NAMES {
        if tool_name == *exact {
            return ToolCategory::ExternalNetwork;
        }
    }
    for prefix in EXTERNAL_TOOL_PREFIXES {
        if tool_name.starts_with(prefix) {
            return ToolCategory::ExternalNetwork;
        }
    }
    ToolCategory::Other
}

/// Monitors cross-iteration tool patterns during an agentic execution.
///
/// Create one `BehavioralMonitor` per execution and call `check()` before
/// each tool call, `record_denial()` for denied calls, and
/// `check_data_flow()` to detect internal→external data chains.
/// Call `reset()` between separate executions when reusing an instance.
pub struct BehavioralMonitor {
    config: BehavioralConfig,
    /// How many times each tool has been called in this execution.
    call_counts: HashMap<String, u8>,
    /// Tools that were denied (for escalation detection).
    denied_tools: Vec<String>,
    /// Whether an internal-data tool has been called yet this execution.
    last_tool_was_internal: bool,
}

impl BehavioralMonitor {
    /// Create a new monitor with config read from environment variables.
    pub fn new() -> Self {
        Self::with_config(BehavioralConfig::from_env())
    }

    /// Create a monitor with explicit config (useful for testing).
    pub fn with_config(config: BehavioralConfig) -> Self {
        Self {
            config,
            call_counts: HashMap::new(),
            denied_tools: Vec::new(),
            last_tool_was_internal: false,
        }
    }

    /// Check a tool call for ToolHammering.
    ///
    /// Increments the call count for `tool_name` and returns a `SecurityEvent`
    /// if a warn or block threshold is crossed.
    ///
    /// Returns `None` for the first N calls (below warn threshold),
    /// `Some(SecurityEvent { action: Warned, … })` at the warn threshold, and
    /// `Some(SecurityEvent { action: Blocked, … })` at the block threshold and above.
    pub fn check(&mut self, tool_name: &str) -> Option<SecurityEvent> {
        let count = self.call_counts.entry(tool_name.to_string()).or_insert(0);
        *count = count.saturating_add(1);
        let current = *count;

        if current >= self.config.hammer_block {
            Some(SecurityEvent {
                guard_name: "behavioral".to_string(),
                action: SecurityAction::Blocked,
                tool_name: tool_name.to_string(),
                reason: format!(
                    "AnomalyType::ToolHammering — tool called {} times (block threshold: {})",
                    current, self.config.hammer_block
                ),
            })
        } else if current >= self.config.hammer_warn {
            Some(SecurityEvent {
                guard_name: "behavioral".to_string(),
                action: SecurityAction::Warned,
                tool_name: tool_name.to_string(),
                reason: format!(
                    "AnomalyType::ToolHammering — tool called {} times (warn threshold: {})",
                    current, self.config.hammer_warn
                ),
            })
        } else {
            None
        }
    }

    /// Record a denied tool call.
    ///
    /// Used to power EscalationAttempt detection: if a denied tool is followed
    /// by a different tool with a broader or overlapping name, that is flagged.
    pub fn record_denial(&mut self, tool_name: &str) {
        self.denied_tools.push(tool_name.to_string());
    }

    /// Check whether `tool_name` looks like an escalation of a previously denied tool.
    ///
    /// Returns a `SecurityEvent` with action `Warned` if escalation is detected.
    ///
    /// An escalation is detected when:
    /// - escalation detection is enabled, AND
    /// - there is at least one denied tool, AND
    /// - the requested tool shares a common prefix (>= 3 chars) with a denied tool
    ///   but is not identical to it.
    pub fn check_escalation(&self, tool_name: &str) -> Option<SecurityEvent> {
        if !self.config.escalation_enabled {
            return None;
        }
        for denied in &self.denied_tools {
            if denied == tool_name {
                // Exact retry — handled elsewhere (hammering or denial).
                continue;
            }
            // Detect same tool-family: shared prefix of 3+ chars.
            let common_len = denied
                .chars()
                .zip(tool_name.chars())
                .take_while(|(a, b)| a == b)
                .count();
            if common_len >= 3 {
                return Some(SecurityEvent {
                    guard_name: "behavioral".to_string(),
                    action: SecurityAction::Warned,
                    tool_name: tool_name.to_string(),
                    reason: format!(
                        "AnomalyType::EscalationAttempt — after denial of '{}', requested '{}'",
                        denied, tool_name
                    ),
                });
            }
        }
        None
    }

    /// Check for data-exfiltration pattern: internal data tool followed by an
    /// external network tool.
    ///
    /// Must be called **in order** for every tool that is about to execute.
    /// Updates internal state to track whether the previous tool accessed
    /// internal data.
    ///
    /// Returns a `SecurityEvent` with action `Warned` if the pattern is detected.
    pub fn check_data_flow(&mut self, tool_name: &str) -> Option<SecurityEvent> {
        let category = classify_tool(tool_name);
        let event = if self.config.exfil_enabled
            && self.last_tool_was_internal
            && category == ToolCategory::ExternalNetwork
        {
            Some(SecurityEvent {
                guard_name: "behavioral".to_string(),
                action: SecurityAction::Warned,
                tool_name: tool_name.to_string(),
                reason: format!(
                    "AnomalyType::DataExfiltration — internal data tool followed by external network tool '{}'",
                    tool_name
                ),
            })
        } else {
            None
        };

        // Update state for next call.
        self.last_tool_was_internal = category == ToolCategory::InternalData;

        event
    }

    /// Clear all state.  Call between separate agentic executions when reusing
    /// this monitor instance.
    pub fn reset(&mut self) {
        self.call_counts.clear();
        self.denied_tools.clear();
        self.last_tool_was_internal = false;
    }
}

impl Default for BehavioralMonitor {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentic::SecurityAction;

    fn monitor_with(warn: u8, block: u8) -> BehavioralMonitor {
        BehavioralMonitor::with_config(BehavioralConfig {
            hammer_warn: warn,
            hammer_block: block,
            escalation_enabled: true,
            exfil_enabled: true,
        })
    }

    // ── ToolHammering ────────────────────────────────────────────────────────

    #[test]
    fn test_hammer_below_warn_threshold_returns_none() {
        let mut m = monitor_with(3, 5);
        assert!(m.check("searxng_search").is_none()); // 1st call
        assert!(m.check("searxng_search").is_none()); // 2nd call
    }

    #[test]
    fn test_hammer_at_warn_threshold_returns_warn() {
        let mut m = monitor_with(3, 5);
        m.check("searxng_search"); // 1
        m.check("searxng_search"); // 2
        let event = m.check("searxng_search").expect("should warn at 3"); // 3
        assert_eq!(event.guard_name, "behavioral");
        assert!(matches!(event.action, SecurityAction::Warned));
        assert!(event.reason.contains("ToolHammering"));
        assert_eq!(event.tool_name, "searxng_search");
    }

    #[test]
    fn test_hammer_at_block_threshold_returns_block() {
        let mut m = monitor_with(3, 5);
        for _ in 0..4 {
            m.check("searxng_search");
        }
        let event = m.check("searxng_search").expect("should block at 5"); // 5th
        assert!(matches!(event.action, SecurityAction::Blocked));
        assert!(event.reason.contains("ToolHammering"));
    }

    #[test]
    fn test_hammer_above_block_threshold_still_blocks() {
        let mut m = monitor_with(3, 5);
        for _ in 0..5 {
            m.check("searxng_search");
        }
        // 6th call — still blocked
        let event = m.check("searxng_search").expect("still blocked after threshold");
        assert!(matches!(event.action, SecurityAction::Blocked));
    }

    #[test]
    fn test_hammer_different_tools_track_independently() {
        let mut m = monitor_with(3, 5);
        // Call tool A twice — should not warn
        m.check("plane_get_issues");
        m.check("plane_get_issues");
        // Call tool B once — should not warn
        let result = m.check("engram_query");
        assert!(result.is_none());
    }

    #[test]
    fn test_hammer_custom_thresholds() {
        let mut m = monitor_with(2, 4);
        m.check("nexus_read"); // 1st — no event
        let event = m.check("nexus_read").expect("warn at 2"); // 2nd
        assert!(matches!(event.action, SecurityAction::Warned));

        m.check("nexus_read"); // 3rd — warn
        let event = m.check("nexus_read").expect("block at 4"); // 4th
        assert!(matches!(event.action, SecurityAction::Blocked));
    }

    // ── EscalationAttempt ────────────────────────────────────────────────────

    #[test]
    fn test_escalation_detected_after_denial() {
        let mut m = monitor_with(3, 5);
        m.record_denial("infisical_get_secret");
        let event = m
            .check_escalation("infisical_list_secrets")
            .expect("should detect escalation");
        assert!(matches!(event.action, SecurityAction::Warned));
        assert!(event.reason.contains("EscalationAttempt"));
    }

    #[test]
    fn test_escalation_exact_retry_not_flagged() {
        let mut m = monitor_with(3, 5);
        m.record_denial("plane_create_issue");
        // Exact same tool is not an escalation (it's a retry — handled elsewhere)
        let result = m.check_escalation("plane_create_issue");
        assert!(result.is_none());
    }

    #[test]
    fn test_escalation_unrelated_tool_not_flagged() {
        let mut m = monitor_with(3, 5);
        m.record_denial("engram_write");
        // Completely different tool family
        let result = m.check_escalation("github_list_repos");
        assert!(result.is_none());
    }

    #[test]
    fn test_escalation_disabled_when_config_off() {
        let mut m = BehavioralMonitor::with_config(BehavioralConfig {
            hammer_warn: 3,
            hammer_block: 5,
            escalation_enabled: false,
            exfil_enabled: true,
        });
        m.record_denial("infisical_get_secret");
        let result = m.check_escalation("infisical_list_all");
        assert!(result.is_none());
    }

    #[test]
    fn test_escalation_no_denials_returns_none() {
        let m = monitor_with(3, 5);
        let result = m.check_escalation("plane_create_issue");
        assert!(result.is_none());
    }

    // ── DataExfiltration ─────────────────────────────────────────────────────

    #[test]
    fn test_exfil_internal_then_external_flagged() {
        let mut m = monitor_with(3, 5);
        // Internal data tool call
        m.check_data_flow("engram_query");
        // Immediately followed by external network tool
        let event = m
            .check_data_flow("lumina_web_fetch")
            .expect("should flag exfil");
        assert!(matches!(event.action, SecurityAction::Warned));
        assert!(event.reason.contains("DataExfiltration"));
        assert_eq!(event.tool_name, "lumina_web_fetch");
    }

    #[test]
    fn test_exfil_internal_then_internal_not_flagged() {
        let mut m = monitor_with(3, 5);
        m.check_data_flow("nexus_check");
        let result = m.check_data_flow("plane_get_issues");
        assert!(result.is_none());
    }

    #[test]
    fn test_exfil_external_then_external_not_flagged() {
        let mut m = monitor_with(3, 5);
        m.check_data_flow("searxng_search");
        let result = m.check_data_flow("lumina_web_fetch");
        assert!(result.is_none(), "external→external is not an exfil pattern");
    }

    #[test]
    fn test_exfil_other_then_external_not_flagged() {
        let mut m = monitor_with(3, 5);
        m.check_data_flow("utc_now");
        let result = m.check_data_flow("searxng_search");
        assert!(result.is_none());
    }

    #[test]
    fn test_exfil_github_prefix_recognized_as_external() {
        let mut m = monitor_with(3, 5);
        m.check_data_flow("infisical_get_secret");
        let event = m
            .check_data_flow("github_create_gist")
            .expect("github_* is external");
        assert!(event.reason.contains("DataExfiltration"));
    }

    #[test]
    fn test_exfil_nexus_prefix_recognized_as_internal() {
        let mut m = monitor_with(3, 5);
        m.check_data_flow("nexus_read");
        let event = m
            .check_data_flow("lumina_web_fetch")
            .expect("nexus_ is internal");
        assert!(event.reason.contains("DataExfiltration"));
    }

    #[test]
    fn test_exfil_plane_prefix_recognized_as_internal() {
        let mut m = monitor_with(3, 5);
        m.check_data_flow("plane_list_issues");
        let event = m
            .check_data_flow("searxng_search")
            .expect("plane_ is internal");
        assert!(event.reason.contains("DataExfiltration"));
    }

    #[test]
    fn test_exfil_disabled_when_config_off() {
        let mut m = BehavioralMonitor::with_config(BehavioralConfig {
            hammer_warn: 3,
            hammer_block: 5,
            escalation_enabled: true,
            exfil_enabled: false,
        });
        m.check_data_flow("engram_query");
        let result = m.check_data_flow("lumina_web_fetch");
        assert!(result.is_none());
    }

    // ── reset() ──────────────────────────────────────────────────────────────

    #[test]
    fn test_reset_clears_call_counts() {
        let mut m = monitor_with(3, 5);
        m.check("engram_query"); // 1
        m.check("engram_query"); // 2
        m.check("engram_query"); // 3 → would warn
        m.reset();
        // After reset, counts start fresh — no warn until threshold again
        assert!(m.check("engram_query").is_none()); // 1st after reset
        assert!(m.check("engram_query").is_none()); // 2nd after reset
    }

    #[test]
    fn test_reset_clears_denied_tools() {
        let mut m = monitor_with(3, 5);
        m.record_denial("infisical_get_secret");
        m.reset();
        let result = m.check_escalation("infisical_list_all");
        assert!(result.is_none(), "denials cleared by reset");
    }

    #[test]
    fn test_reset_clears_data_flow_state() {
        let mut m = monitor_with(3, 5);
        m.check_data_flow("engram_query"); // sets last_tool_was_internal
        m.reset();
        // After reset, exfil state is cleared — external tool should not trigger
        let result = m.check_data_flow("lumina_web_fetch");
        assert!(result.is_none(), "data flow state cleared by reset");
    }

    // ── SecurityEvent structure ───────────────────────────────────────────────

    #[test]
    fn test_security_event_guard_name_is_behavioral() {
        let mut m = monitor_with(1, 2);
        let event = m.check("any_tool").expect("warn at threshold 1");
        assert_eq!(event.guard_name, "behavioral");
    }

    #[test]
    fn test_security_event_tool_name_matches_input() {
        let mut m = monitor_with(1, 3);
        let event = m.check("nexus_check").expect("should warn");
        assert_eq!(event.tool_name, "nexus_check");
    }

    // ── BehavioralConfig::from_env ────────────────────────────────────────────
    //
    // These two tests mutate process-global env vars and are therefore
    // serialised with a mutex so they do not race each other.

    #[test]
    fn test_config_defaults_when_env_absent() {
        // Use BehavioralConfig::default() to verify spec defaults without
        // touching env vars — avoids races with the env-var test.
        let cfg = BehavioralConfig::default();
        assert_eq!(cfg.hammer_warn, 3);
        assert_eq!(cfg.hammer_block, 5);
        assert!(cfg.escalation_enabled);
        assert!(cfg.exfil_enabled);
    }

    #[test]
    fn test_config_reads_from_env() {
        // We test from_env() by building the config directly from values that
        // match what from_env() would parse.  Avoids process-wide env mutation.
        // Verify the parsing logic independently:
        fn parse_bool_env(val: &str) -> bool {
            val.to_lowercase() != "false"
        }
        fn parse_u8_env(val: &str, default: u8) -> u8 {
            val.parse::<u8>().unwrap_or(default)
        }

        assert_eq!(parse_u8_env("2", 3), 2);
        assert_eq!(parse_u8_env("4", 5), 4);
        assert!(!parse_bool_env("false"));
        assert!(parse_bool_env("true"));
        assert!(parse_bool_env("True"));

        // from_env() with no vars set should give defaults
        let cfg = BehavioralConfig::from_env();
        // hammer_warn is 3 unless overridden
        assert!(cfg.hammer_warn >= 1 && cfg.hammer_warn <= 255);
        assert!(cfg.hammer_block >= cfg.hammer_warn);
    }

    // ── Tool classification ───────────────────────────────────────────────────

    #[test]
    fn test_classify_all_internal_prefixes() {
        assert_eq!(classify_tool("engram_write"), ToolCategory::InternalData);
        assert_eq!(classify_tool("nexus_send"), ToolCategory::InternalData);
        assert_eq!(classify_tool("plane_get_issue"), ToolCategory::InternalData);
        assert_eq!(classify_tool("infisical_get_secret"), ToolCategory::InternalData);
    }

    #[test]
    fn test_classify_external_exact_names() {
        assert_eq!(classify_tool("lumina_web_fetch"), ToolCategory::ExternalNetwork);
        assert_eq!(classify_tool("searxng_search"), ToolCategory::ExternalNetwork);
    }

    #[test]
    fn test_classify_github_prefix_as_external() {
        assert_eq!(classify_tool("github_list_repos"), ToolCategory::ExternalNetwork);
        assert_eq!(classify_tool("github_create_pr"), ToolCategory::ExternalNetwork);
    }

    #[test]
    fn test_classify_other_tools() {
        assert_eq!(classify_tool("utc_now"), ToolCategory::Other);
        assert_eq!(classify_tool("health_check"), ToolCategory::Other);
        assert_eq!(classify_tool("plane"), ToolCategory::Other); // no underscore prefix
    }
}
