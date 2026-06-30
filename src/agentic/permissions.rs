use std::collections::HashSet;

use crate::agentic::{SecurityAction, SecurityEvent};

/// Enforces user permission sets on every tool call within the agentic loop.
///
/// Supports three permission modes:
/// - `["*"]` — Admin: all tools allowed
/// - `[]` — Guest: no tools allowed
/// - `["tool_a", "google_calendar_*", ...]` — Named and wildcard-matched tools
///
/// Wildcard matching: `"google_calendar_*"` matches any tool name whose prefix
/// is `google_calendar_`. Only a single trailing `*` is supported.
///
/// Denials are recorded as [`SecurityEvent`] with `guard_name = "permission"`
/// and `action = SecurityAction::Blocked`.
pub struct PermissionEnforcer {
    /// Exact tool names that are allowed (O(1) lookup).
    exact: HashSet<String>,
    /// Wildcard prefixes (the part before the `*`).
    wildcard_prefixes: Vec<String>,
    /// When `true`, every tool is permitted regardless of exact/wildcard sets.
    allow_all: bool,
    /// When `true`, no tool is permitted (empty permission list).
    deny_all: bool,
}

impl PermissionEnforcer {
    /// Build a new enforcer from the permission list transmitted in `AgenticRequest`.
    ///
    /// # Arguments
    /// * `permissions` – Slice of permission strings from the request. May contain
    ///   `"*"` (allow all), plain tool names, or wildcard patterns like
    ///   `"google_calendar_*"`.
    pub fn new(permissions: &[String]) -> Self {
        if permissions.is_empty() {
            return Self {
                exact: HashSet::new(),
                wildcard_prefixes: Vec::new(),
                allow_all: false,
                deny_all: true,
            };
        }

        let mut exact = HashSet::new();
        let mut wildcard_prefixes = Vec::new();
        let mut allow_all = false;

        for p in permissions {
            if p == "*" {
                allow_all = true;
                break;
            } else if p.ends_with('*') {
                // Strip trailing `*` to obtain the prefix.
                let prefix = p[..p.len() - 1].to_string();
                wildcard_prefixes.push(prefix);
            } else {
                exact.insert(p.clone());
            }
        }

        Self {
            exact,
            wildcard_prefixes,
            allow_all,
            deny_all: false,
        }
    }

    /// Check whether `tool_name` is permitted by this permission set.
    ///
    /// Returns `Ok(())` when the tool is allowed. Returns `Err(SecurityEvent)`
    /// when the tool is denied; the caller should inject the clean error message
    /// returned by [`Self::denial_message`] into the LLM context and record the
    /// event.
    ///
    /// Complexity: O(1) for exact matches plus O(W) for wildcard prefixes, where
    /// W is the number of wildcard entries. In practice W is small (< 20).
    pub fn check(&self, tool_name: &str) -> Result<(), SecurityEvent> {
        if self.allow_all {
            return Ok(());
        }

        if self.deny_all {
            return Err(self.build_event(tool_name, "permission list is empty — no tools available"));
        }

        if self.exact.contains(tool_name) {
            return Ok(());
        }

        for prefix in &self.wildcard_prefixes {
            if tool_name.starts_with(prefix.as_str()) {
                return Ok(());
            }
        }

        Err(self.build_event(
            tool_name,
            &format!("tool '{tool_name}' is not in the user's permission set"),
        ))
    }

    /// Human-readable error message suitable for injection into LLM context.
    ///
    /// Deliberately vague to avoid leaking information about what tools exist.
    pub fn denial_message(tool_name: &str) -> String {
        format!("Tool {tool_name} is not available.")
    }

    fn build_event(&self, tool_name: &str, reason: &str) -> SecurityEvent {
        SecurityEvent {
            guard_name: "permission".to_string(),
            action: SecurityAction::Blocked,
            tool_name: tool_name.to_string(),
            reason: reason.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn perm(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    // TEST PLAN: allowed tool passes
    #[test]
    fn test_allowed_exact_tool_passes() {
        let enforcer = PermissionEnforcer::new(&perm(&["nexus_send", "plane_list_issues"]));
        assert!(enforcer.check("nexus_send").is_ok());
        assert!(enforcer.check("plane_list_issues").is_ok());
    }

    // TEST PLAN: denied tool blocked with clean error
    #[test]
    fn test_denied_tool_blocked() {
        let enforcer = PermissionEnforcer::new(&perm(&["nexus_send"]));
        let result = enforcer.check("infisical_get_secret");
        assert!(result.is_err());
        let event = result.unwrap_err();
        assert_eq!(event.guard_name, "permission");
        assert!(matches!(event.action, SecurityAction::Blocked));
        assert_eq!(event.tool_name, "infisical_get_secret");
    }

    // TEST PLAN: wildcard matching
    #[test]
    fn test_wildcard_matching() {
        let enforcer = PermissionEnforcer::new(&perm(&["google_calendar_*"]));
        assert!(enforcer.check("google_calendar_today").is_ok());
        assert!(enforcer.check("google_calendar_create_event").is_ok());
        assert!(enforcer.check("google_calendar_delete_event").is_ok());
        // Different tool family — must be blocked
        assert!(enforcer.check("google_drive_upload").is_err());
        assert!(enforcer.check("infisical_get_secret").is_err());
    }

    // TEST PLAN: wildcard only matches correct prefix (not suffix or middle)
    #[test]
    fn test_wildcard_prefix_only() {
        let enforcer = PermissionEnforcer::new(&perm(&["nexus_*"]));
        assert!(enforcer.check("nexus_send").is_ok());
        assert!(enforcer.check("nexus_read").is_ok());
        // Does not match tools that merely contain "nexus" in another position
        assert!(enforcer.check("get_nexus_status").is_err());
    }

    // TEST PLAN: empty permissions blocks all
    #[test]
    fn test_empty_permissions_blocks_all() {
        let enforcer = PermissionEnforcer::new(&perm(&[]));
        assert!(enforcer.check("nexus_send").is_err());
        assert!(enforcer.check("plane_list_issues").is_err());
        assert!(enforcer.check("anything_at_all").is_err());
    }

    // TEST PLAN: `*` allows all
    #[test]
    fn test_star_allows_all() {
        let enforcer = PermissionEnforcer::new(&perm(&["*"]));
        assert!(enforcer.check("infisical_get_secret").is_ok());
        assert!(enforcer.check("dev_run_command").is_ok());
        assert!(enforcer.check("any_future_tool").is_ok());
    }

    // TEST PLAN: denial → SecurityEvent logged
    #[test]
    fn test_denial_produces_security_event() {
        let enforcer = PermissionEnforcer::new(&perm(&["nexus_send"]));
        let event = enforcer.check("vault_read_secret").unwrap_err();
        assert_eq!(event.guard_name, "permission");
        assert!(matches!(event.action, SecurityAction::Blocked));
        assert_eq!(event.tool_name, "vault_read_secret");
        assert!(!event.reason.is_empty());
    }

    // Verify no hardcoded IPs in the event reason field
    #[test]
    fn test_denial_event_contains_no_hardcoded_ips() {
        let enforcer = PermissionEnforcer::new(&perm(&[]));
        let event = enforcer.check("some_tool").unwrap_err();
        // Must not contain any private IP subnet strings
        assert!(!event.reason.contains("192.168"));
        assert!(!event.reason.contains("10.0."));
        assert!(!event.reason.contains("172.16"));
    }

    // Verify denial_message is clean (suitable for LLM context injection)
    #[test]
    fn test_denial_message_is_clean() {
        let msg = PermissionEnforcer::denial_message("infisical_get_secret");
        assert_eq!(msg, "Tool infisical_get_secret is not available.");
        // Must not leak permission list details
        assert!(!msg.contains("permission set"));
        assert!(!msg.contains("blocked"));
    }

    // Edge case: mixed exact + wildcard list
    #[test]
    fn test_mixed_exact_and_wildcard() {
        let enforcer = PermissionEnforcer::new(&perm(&[
            "nexus_send",
            "google_calendar_*",
            "plane_list_issues",
        ]));
        assert!(enforcer.check("nexus_send").is_ok());
        assert!(enforcer.check("google_calendar_today").is_ok());
        assert!(enforcer.check("plane_list_issues").is_ok());
        assert!(enforcer.check("nexus_read").is_err()); // not in exact list
        assert!(enforcer.check("infisical_get_secret").is_err());
    }

    // Edge case: wildcard that is just `*` (the whole string) = allow all
    #[test]
    fn test_star_as_only_element_allows_all() {
        let enforcer = PermissionEnforcer::new(&perm(&["nexus_send", "*", "plane_list_issues"]));
        // Once `*` appears, everything is allowed
        assert!(enforcer.check("infisical_get_secret").is_ok());
    }

    // Edge case: very long permission list (HashSet must keep O(1))
    #[test]
    fn test_large_permission_list_performance() {
        let tools: Vec<String> = (0..10_000)
            .map(|i| format!("tool_number_{i}"))
            .collect();
        let enforcer = PermissionEnforcer::new(&tools);
        // The 5000th tool should be allowed
        assert!(enforcer.check("tool_number_5000").is_ok());
        // A tool not in the list should be denied
        assert!(enforcer.check("tool_number_99999").is_err());
    }

    // Edge case: empty string tool name in list
    #[test]
    fn test_empty_string_tool_name_exact() {
        let enforcer = PermissionEnforcer::new(&perm(&[""]));
        // Empty string tool name exact-matches the empty entry
        assert!(enforcer.check("").is_ok());
        // But a real tool name is still denied
        assert!(enforcer.check("nexus_send").is_err());
    }

    // Edge case: wildcard prefix that is empty (i.e., permission = "*" but stored as wildcard)
    // A permission entry of "abc*" should match anything starting with "abc"
    #[test]
    fn test_wildcard_long_prefix() {
        let enforcer = PermissionEnforcer::new(&perm(&["google_calendar_create_*"]));
        assert!(enforcer.check("google_calendar_create_event").is_ok());
        assert!(enforcer.check("google_calendar_create_reminder").is_ok());
        // Shorter match — must be denied
        assert!(enforcer.check("google_calendar_today").is_err());
    }

    // Ensure deny_all path produces descriptive reason
    #[test]
    fn test_deny_all_reason_message() {
        let enforcer = PermissionEnforcer::new(&perm(&[]));
        let event = enforcer.check("nexus_send").unwrap_err();
        assert!(event.reason.contains("empty"), "reason should mention empty permissions: {}", event.reason);
    }

    // Ensure non-empty list denied reason mentions tool name
    #[test]
    fn test_denied_reason_contains_tool_name() {
        let enforcer = PermissionEnforcer::new(&perm(&["other_tool"]));
        let event = enforcer.check("secret_tool").unwrap_err();
        assert!(event.reason.contains("secret_tool"), "reason should mention the tool: {}", event.reason);
    }
}
