//! Per-instance tool + scope config (S91 CTUI-05).
//!
//! Enable/disable individual tools and view/edit scopes on a Terminus instance —
//! ONLY when the server advertises the capability. If a control is not exposed
//! ([`super::mcp_client::ServerCaps`]), the UI shows it read-only as "not
//! supported by this instance" and NEVER fakes a mutation.
//!
//! Every mutation is:
//!   - behind an explicit confirm (a [`crate::confirm::PendingMutation`]),
//!   - applied against the server through a capability-checked write op,
//!   - audit-logged sanitized (tool NAME + outcome, never a secret),
//!   - NON-optimistic: the displayed enabled/scope state is updated only from the
//!     server's confirmed response. A server-side failure leaves the UI showing
//!     the true, unchanged state (no optimistic lie); a rejected scope edit
//!     reverts the display and reports why.

use async_trait::async_trait;

use crate::confirm::PendingMutation;
use crate::modes::terminus::audit::AuditOutcome;
use crate::modes::terminus::mcp_client::{McpError, ServerCaps};
use crate::secret::SecretValue;

/// Result of applying a tool mutation, from the server's confirmed response.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolMutationResult {
    /// Server confirmed the new enabled state — UI updates to this value.
    Applied { enabled: bool },
    /// Server rejected the change; UI reflects the TRUE (unchanged) state + why.
    Rejected { true_enabled: bool, why: String },
    /// The instance became unreachable mid-action; UI must re-sync on reconnect.
    Interrupted { detail: String },
}

/// Result of a scope edit, from the server's confirmed response.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScopeEditResult {
    Applied { scopes: Vec<String> },
    /// Rejected → revert display to `true_scopes` and show `why`.
    Rejected { true_scopes: Vec<String>, why: String },
    Interrupted { detail: String },
}

/// Config-write operations layered on top of the read-only MCP client. A server
/// that does not expose a control returns [`McpError::Transport`] with a
/// "not supported" marker, which the controller maps to a read-only outcome
/// (never a faked success). Injected for testing.
#[async_trait]
pub trait ToolControl: Send + Sync {
    /// Ask the server to set a tool's enabled state. Returns the server's
    /// confirmed post-state, or an error if it refused / became unreachable.
    async fn set_tool_enabled(
        &self,
        tool: &str,
        enabled: bool,
        token: Option<&SecretValue>,
    ) -> Result<bool, McpError>;

    /// Ask the server to set a tool's scopes. Returns confirmed scopes or error.
    async fn set_tool_scopes(
        &self,
        tool: &str,
        scopes: &[String],
        token: Option<&SecretValue>,
    ) -> Result<Vec<String>, McpError>;
}

/// Build the confirm-gated (simple) mutation for a tool toggle. Requires an
/// explicit confirm keystroke — never fires without confirmation.
pub fn toggle_mutation(tool: &str, target_enabled: bool) -> PendingMutation {
    let verb = if target_enabled { "Enable" } else { "Disable" };
    PendingMutation::simple("terminus.tool.toggle", format!("{verb} tool '{tool}'"))
}

/// Build the confirm-gated (simple) mutation for a scope edit.
pub fn scope_edit_mutation(tool: &str) -> PendingMutation {
    PendingMutation::simple("terminus.tool.scope", format!("Edit scopes for tool '{tool}'"))
}

/// Apply a confirmed tool toggle. Precondition: the confirm gate was satisfied
/// AND `caps.tool_toggle` is true. If the capability is absent this returns
/// `None` — the caller renders read-only "not supported" and does NOT attempt a
/// mutation. On a server error the result reflects the TRUE state (no optimistic
/// update). Returns the audit outcome alongside the result.
pub async fn apply_toggle(
    control: &dyn ToolControl,
    caps: ServerCaps,
    tool: &str,
    current_enabled: bool,
    token: Option<&SecretValue>,
) -> Option<(ToolMutationResult, AuditOutcome)> {
    if !caps.tool_toggle {
        // Capability absent → do NOT fake a mutation.
        return None;
    }
    let target = !current_enabled;
    let result = match control.set_tool_enabled(tool, target, token).await {
        Ok(confirmed) => (ToolMutationResult::Applied { enabled: confirmed }, AuditOutcome::Applied),
        Err(McpError::ProcessDied(d)) | Err(McpError::Transport(d)) if d.contains("not supported") => {
            // Server refused because it lacks the control — read-only truth.
            (
                ToolMutationResult::Rejected { true_enabled: current_enabled, why: d },
                AuditOutcome::NotSupported,
            )
        }
        Err(McpError::ProcessDied(d)) => {
            (ToolMutationResult::Interrupted { detail: d }, AuditOutcome::Interrupted)
        }
        Err(e) => (
            // Any other refusal: UI reflects the true unchanged state + why.
            ToolMutationResult::Rejected {
                true_enabled: current_enabled,
                why: format!("{e:?}"),
            },
            AuditOutcome::Rejected,
        ),
    };
    Some(result)
}

/// Apply a confirmed scope edit. Same capability + truth rules as [`apply_toggle`].
pub async fn apply_scope_edit(
    control: &dyn ToolControl,
    caps: ServerCaps,
    tool: &str,
    proposed: Vec<String>,
    current_scopes: Vec<String>,
    token: Option<&SecretValue>,
) -> Option<(ScopeEditResult, AuditOutcome)> {
    if !caps.scope_edit {
        return None; // not supported → read-only, no fake mutation
    }
    let result = match control.set_tool_scopes(tool, &proposed, token).await {
        Ok(confirmed) => (ScopeEditResult::Applied { scopes: confirmed }, AuditOutcome::Applied),
        Err(McpError::ProcessDied(d)) => {
            (ScopeEditResult::Interrupted { detail: d }, AuditOutcome::Interrupted)
        }
        Err(e) => (
            // Rejected → revert display to the true scopes + explain why.
            ScopeEditResult::Rejected { true_scopes: current_scopes, why: format!("{e:?}") },
            AuditOutcome::Rejected,
        ),
    };
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct OkControl;
    #[async_trait]
    impl ToolControl for OkControl {
        async fn set_tool_enabled(&self, _t: &str, enabled: bool, _tok: Option<&SecretValue>) -> Result<bool, McpError> {
            Ok(enabled)
        }
        async fn set_tool_scopes(&self, _t: &str, scopes: &[String], _tok: Option<&SecretValue>) -> Result<Vec<String>, McpError> {
            Ok(scopes.to_vec())
        }
    }

    struct FailControl(McpError);
    #[async_trait]
    impl ToolControl for FailControl {
        async fn set_tool_enabled(&self, _t: &str, _e: bool, _tok: Option<&SecretValue>) -> Result<bool, McpError> {
            Err(self.0.clone())
        }
        async fn set_tool_scopes(&self, _t: &str, _s: &[String], _tok: Option<&SecretValue>) -> Result<Vec<String>, McpError> {
            Err(self.0.clone())
        }
    }

    #[test]
    fn toggle_and_scope_mutations_require_confirm() {
        // A tool toggle is confirm-gated (simple) — needs explicit 'y'.
        let m = toggle_mutation("plane_create_issue", false);
        assert!(!m.satisfied_by_keystroke('n'), "no confirm without the key");
        assert!(m.satisfied_by_keystroke('y'));
        assert!(scope_edit_mutation("x").satisfied_by_keystroke('y'));
    }

    /// NEGATIVE: an instance that does NOT expose tool-toggle shows read-only and
    /// no mutation is attempted (apply returns None, never a faked success).
    #[tokio::test]
    async fn no_toggle_capability_shows_readonly_no_fake() {
        let caps = ServerCaps { tool_toggle: false, scope_edit: false };
        let out = apply_toggle(&OkControl, caps, "tool", true, None).await;
        assert!(out.is_none(), "no capability → no mutation attempted, not supported");
    }

    #[tokio::test]
    async fn capable_toggle_applies_from_server_response() {
        let caps = ServerCaps { tool_toggle: true, scope_edit: false };
        let (res, outcome) = apply_toggle(&OkControl, caps, "tool", true, None).await.unwrap();
        // was enabled(true) → target disable(false); server confirms false.
        assert_eq!(res, ToolMutationResult::Applied { enabled: false });
        assert_eq!(outcome, AuditOutcome::Applied);
    }

    /// NEGATIVE: a server-side toggle FAILURE must not optimistically flip the
    /// UI. The result carries the TRUE (unchanged) state + a reason.
    #[tokio::test]
    async fn server_toggle_failure_reflects_true_state_no_optimism() {
        let caps = ServerCaps { tool_toggle: true, scope_edit: false };
        let ctrl = FailControl(McpError::Transport("server said no".into()));
        let (res, outcome) = apply_toggle(&ctrl, caps, "tool", true, None).await.unwrap();
        match res {
            ToolMutationResult::Rejected { true_enabled, .. } => {
                assert!(true_enabled, "UI shows the unchanged true state, not the target");
            }
            other => panic!("expected rejected, got {other:?}"),
        }
        assert_eq!(outcome, AuditOutcome::Rejected);
    }

    /// NEGATIVE: a rejected scope edit reverts the display to the true scopes and
    /// explains why.
    #[tokio::test]
    async fn rejected_scope_edit_reverts_display() {
        let caps = ServerCaps { tool_toggle: false, scope_edit: true };
        let ctrl = FailControl(McpError::Transport("scope not allowed".into()));
        let current = vec!["read".to_string()];
        let (res, _o) = apply_scope_edit(&ctrl, caps, "tool", vec!["read".into(), "write".into()], current.clone(), None)
            .await
            .unwrap();
        match res {
            ScopeEditResult::Rejected { true_scopes, why } => {
                assert_eq!(true_scopes, current, "reverted to true scopes");
                assert!(why.contains("scope not allowed"));
            }
            other => panic!("expected rejected, got {other:?}"),
        }
    }

    /// NEGATIVE: instance unreachable mid-action fails cleanly (interrupted), so
    /// the caller re-syncs on reconnect rather than lying.
    #[tokio::test]
    async fn unreachable_mid_action_is_interrupted() {
        let caps = ServerCaps { tool_toggle: true, scope_edit: false };
        let ctrl = FailControl(McpError::ProcessDied("conn dropped".into()));
        let (res, outcome) = apply_toggle(&ctrl, caps, "tool", false, None).await.unwrap();
        assert!(matches!(res, ToolMutationResult::Interrupted { .. }));
        assert_eq!(outcome, AuditOutcome::Interrupted);
    }
}
